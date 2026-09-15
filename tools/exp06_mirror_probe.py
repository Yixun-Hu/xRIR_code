"""Gate G1's mirror probe (plan v4 6.1): does the geometry branch tell the two ends apart?

Two parts. The *legacy reproduction* re-runs the 2026-09-14 hallway diagnostic on its
frozen 24-microphone cohort, so the probe's numerics can be checked against the anchors
that diagnostic published. The *full-cohort gate* runs every +y hallway test microphone
through the four models of 6.1 and decides G1.

Nothing here imports the diagnostic scripts and nothing patches a Torch global: the pinned
forward is recomposed from the model's own submodules, and the alignment the pinned method
would allocate on a GPU comes from tools.exp06_probe_align.
"""
import argparse
import datetime
import json
import numbers
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_xRIR_backbone import load_model_state
from model.xRIR_cyl_oriented import build_xrir_exp06
from sim_to_real.haa_dataset import DEFAULT_ROOT, HAADataset
from tools.exp06_heading import GIT_FIELDS, HeadingFrameDataset, read_heading_json
from tools.exp06_heading import _closure_digest as closure_digest
from tools.exp06_probe_align import (SAMPLE_RATE, SPEED_OF_SOUND,  # noqa: F401  (re-export)
                                     shift_and_align_device)
from tools.provenance import (closure_record, environment, git_state, sha256_file,
                              source_closure, write_manifest)
from tools.yaw_rotation import rotate_scene_yaw

ENTRY_MODULE = 'tools.exp06_mirror_probe'

HOP_SIZE = 31                                   # the pinned STFT hop of xRIR.convert_ir_to_spec
C50_SECONDS = 0.05
C50_FRAMES = round(C50_SECONDS * SAMPLE_RATE / HOP_SIZE)
COORD_SCALE = 5.0
LATE_EPS = 1e-12
LOG_SPEC_EPS = 1e-8


def onset_frames(src_loc):
    """Direct-path arrival of each query, in STFT hops, as the diagnostic computed it."""
    if not isinstance(src_loc, torch.Tensor) or src_loc.dim() != 2 or src_loc.shape[1] != 3:
        raise ValueError('src_loc must have shape [B, 3]')
    return (src_loc.norm(dim=1) / SPEED_OF_SOUND * SAMPLE_RATE / HOP_SIZE).round().long()


def spectral_c50(mag, onset):
    """Early/late energy ratio in dB of a magnitude spectrogram [B, F, T], per query."""
    if not isinstance(mag, torch.Tensor) or mag.dim() != 3:
        raise ValueError('mag must have shape [B, F, T]')
    onset = torch.as_tensor(onset).reshape(-1)
    if onset.shape[0] != mag.shape[0] or onset.min().item() < 0:
        raise ValueError('one nonnegative onset frame per query is required')
    energy = (mag ** 2).sum(1)
    out = []
    for b in range(energy.shape[0]):
        start = int(onset[b])
        early = energy[b, start:start + C50_FRAMES].sum()
        late = energy[b, start + C50_FRAMES:].sum()
        out.append(10 * torch.log10(early / (late + LATE_EPS)))
    return torch.stack(out)


def geometry_feature(model, src_locs, depth_coord):
    """The pooled source branch for N positions against one panorama -> [N, C]."""
    if depth_coord.dim() != 3 or src_locs.dim() != 2 or src_locs.shape[1] != 3:
        raise ValueError('src_locs must be [N, 3] and depth_coord [3, H, W]')
    encoded = (src_locs[:, :, None, None] - depth_coord[None]) / COORD_SCALE
    pooled = model.lin_proj_0(model.src_proj(model.source_network(encoded)).permute(0, 2, 1))
    return pooled.squeeze(-1)


def composed_forward(model, depth_coord, ref_irs, src_loc, ref_locs, tgt_wav,
                     align=shift_and_align_device):
    """`xRIR.forward`, line for line, also returning the mixing weights and pooled source.

    The only departures are the device-preserving alignment (the pinned one allocates on
    the default GPU) and the extra return values. `model.times` is a plain CPU tensor, not
    a buffer, so it is transferred exactly as the pinned forward transfers it.
    """
    x = align(model, ref_irs, src_loc, ref_locs)
    times = model.times
    time_embed = model.time_embedder(times.unsqueeze(0).to(ref_locs.device)).repeat(
        ref_locs.shape[0], 1, 1)
    time_out = model.time_proj(time_embed)

    source_out = model.lin_proj_0(model.src_proj(model.source_network(
        (src_loc[:, :, None, None] - depth_coord) / COORD_SCALE)).permute(0, 2, 1)
        ).squeeze(-1).unsqueeze(1)
    receiver_out = model.lin_proj_0(model.src_proj(model.source_network(
        (-depth_coord) / COORD_SCALE)).permute(0, 2, 1)).squeeze(-1).unsqueeze(1)
    ref_geo = torch.cat([model.lin_proj_0(model.src_proj(model.source_network(
        (ref_locs[:, i, :, None, None] - depth_coord) / COORD_SCALE)).permute(0, 2, 1)
        ).squeeze(-1).unsqueeze(1) for i in range(ref_locs.shape[1])], dim=1)

    fuse_geo = torch.cat([receiver_out, source_out], dim=-1)
    fuse_ref_geo = torch.cat([receiver_out.repeat(1, ref_geo.shape[1], 1), ref_geo], dim=-1)
    ref_src = torch.cat([model.src_coord_proj(model.dist_embedder(
        ref_locs[:, i:(i + 1)] / COORD_SCALE).view(ref_locs.shape[0], -1)).unsqueeze(1)
        for i in range(ref_locs.shape[1])], dim=1)
    src_feats = model.src_coord_proj(model.dist_embedder(
        src_loc.unsqueeze(1) / COORD_SCALE).view(src_loc.shape[0], -1)).unsqueeze(1)
    fuse_geo = torch.cat([src_feats, fuse_geo], dim=-1)
    fuse_ref_geo = torch.cat([ref_src, fuse_ref_geo], dim=-1)

    logs, audio = [], []
    for i in range(x.shape[1]):
        spec_i = model.convert_ir_to_spec(x[:, i:(i + 1)])
        logs.append(torch.log(spec_i + LOG_SPEC_EPS))
        audio.append(model.audio_enc(spec_i).unsqueeze(1))
    logs = torch.cat(logs, dim=1)
    audio = torch.cat(audio, dim=1)

    fuse_ref = model.lin_proj_2(torch.cat((fuse_ref_geo, audio), dim=-1))
    fuse_tgt = model.lin_proj_1(fuse_geo)
    fuse = F.softmax(fuse_ref @ fuse_tgt.permute(0, 2, 1) / fuse_tgt.shape[-1], dim=1) * fuse_ref
    weights = fuse @ time_out.permute(0, 2, 1) / fuse.shape[-1]
    out_log_spec = torch.sum(logs * weights.unsqueeze(2), dim=1)
    return out_log_spec, model.convert_ir_to_spec(tgt_wav)[:, 0], weights, source_out.squeeze(1)


def legacy_cohort(test_indices, y_room_frame, n=24):
    """The 2026-09-14 selection: sort one side by room-frame y, take n evenly spaced ranks."""
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
        raise ValueError('n must be a positive integer')
    indices = np.asarray(test_indices)
    y = np.asarray(y_room_frame)
    if indices.ndim != 1 or y.shape != indices.shape:
        raise ValueError('test_indices and y_room_frame must be one aligned 1-D array')
    if not np.isfinite(y).all():
        raise ValueError('room-frame y must be finite')
    cohort = {}
    for label, mask in (('+y', y > 0), ('-y', y < 0)):
        ranked = np.where(mask)[0]
        ranked = ranked[np.argsort(y[ranked])]
        if ranked.size == 0:
            cohort[label] = {'positions': [], 'mic_ids': []}
            continue
        if ranked.size < n:
            raise ValueError('side {} has {} microphones, fewer than the {} the cohort '
                             'rule selects'.format(label, ranked.size, n))
        chosen = ranked[np.linspace(0, ranked.size - 1, n).round().astype(int)]
        cohort[label] = {'positions': [int(i) for i in chosen],
                         'mic_ids': [int(indices[i]) for i in chosen]}
    return cohort


G1_THRESHOLDS = {'cyl_cosine_min': 0.90, 'control_cosine_max': 0.60,
                 'cyl_share_min': 0.70, 'control_share_max': 0.35,
                 'cylor_cosine_max': 0.80, 'cylor_share_max': 0.50}
G1_MODELS = ('cyl', 'control', 'cyl_or')


def _value(stats, model, key):
    try:
        value = float(stats[model][key])
    except (KeyError, TypeError, ValueError):
        raise ValueError('G1 needs {} of {}'.format(key, model))
    if not np.isfinite(value):
        raise ValueError('{} of {} is not finite'.format(key, model))
    return value


def g1_decision(stats, thresholds=G1_THRESHOLDS):
    """pass / fail / inconclusive under 6.1: the bracket first, the oriented model second.

    Every comparison is strict, so a statistic exactly on a threshold never satisfies it:
    an anchor exactly at its bound leaves G1 inconclusive, and an oriented model exactly at
    its bound fails.
    """
    values = {model: {key: _value(stats, model, key) for key in
                      ('mirror_cosine', 'opposite_side_weight_share')} for model in G1_MODELS}
    cosine = {model: values[model]['mirror_cosine'] for model in G1_MODELS}
    share = {model: values[model]['opposite_side_weight_share'] for model in G1_MODELS}
    unbracketed = []
    if not cosine['cyl'] > thresholds['cyl_cosine_min']:
        unbracketed.append('cyl mirror cosine {:.4f} is not above {}'.format(
            cosine['cyl'], thresholds['cyl_cosine_min']))
    if not cosine['control'] < thresholds['control_cosine_max']:
        unbracketed.append('control mirror cosine {:.4f} is not below {}'.format(
            cosine['control'], thresholds['control_cosine_max']))
    if not share['cyl'] > thresholds['cyl_share_min']:
        unbracketed.append('cyl weight share {:.4f} is not above {}'.format(
            share['cyl'], thresholds['cyl_share_min']))
    if not share['control'] < thresholds['control_share_max']:
        unbracketed.append('control weight share {:.4f} is not below {}'.format(
            share['control'], thresholds['control_share_max']))
    if unbracketed:
        return {'outcome': 'inconclusive', 'reasons': unbracketed,
                'thresholds': dict(thresholds), 'values': values}
    failures = []
    if not cosine['cyl_or'] < thresholds['cylor_cosine_max']:
        failures.append('cyl_or mirror cosine {:.4f} is not below {}'.format(
            cosine['cyl_or'], thresholds['cylor_cosine_max']))
    if not share['cyl_or'] < thresholds['cylor_share_max']:
        failures.append('cyl_or weight share {:.4f} is not below {}'.format(
            share['cyl_or'], thresholds['cylor_share_max']))
    return {'outcome': 'fail' if failures else 'pass', 'reasons': failures,
            'thresholds': dict(thresholds), 'values': values}


# --- the probe on a real cache ---------------------------------------------------------

HALLWAY = 'hallway'
LEGACY_COHORT_N = 24
BACKBONE_OF = {'cyl': 'cylindrical', 'control': 'simple',
               'cyl_or': 'cylindrical_oriented', 'cyl_hf': 'cylindrical'}
FRAME_OF = {'cyl': 'room', 'control': 'room', 'cyl_or': 'heading', 'cyl_hf': 'heading'}
GATE_ARMS = ('cyl_or', 'cyl', 'control', 'cyl_hf')
LEGACY_CHECKPOINTS = {'cyl': 'ckpt/xRIR_cyl_8_shot/epoch_12.pth',
                      'control': 'ckpt/xRIR_simple_8_shot/epoch_12.pth'}
# The 2026-09-14 diagnostic's published hallway numbers, and the band the reproduction
# must land in for the probe's numerics to count as validated (plan v4 6.1).
ANCHORS_LEGACY = {'cyl': {'mirror_cosine': 0.975, 'opposite_side_weight_share': 0.85},
                  'control': {'mirror_cosine': 0.435, 'opposite_side_weight_share': 0.24}}
ANCHOR_TOLERANCE = 0.01
# Frozen on 2026-09-15 from ~/data_cache/HAA_xrir/hallway by exactly `legacy_cohort`;
# `test_the_frozen_cohort_is_the_rule_applied_to_the_cache` re-derives them.
HALLWAY_LEGACY_COHORT = {
    '+y': {'positions': [238, 246, 250, 256, 265, 275, 283, 294, 303, 313, 318, 333,
                         338, 342, 343, 352, 365, 372, 382, 391, 401, 406, 419, 422],
           'mic_ids': [324, 335, 340, 350, 362, 375, 387, 402, 414, 427, 434, 454,
                       461, 466, 468, 480, 498, 508, 521, 533, 547, 554, 571, 575]},
    '-y': {'positions': [0, 13, 21, 29, 41, 57, 67, 76, 83, 94, 98, 113,
                         130, 138, 144, 151, 158, 170, 192, 200, 207, 216, 220, 237],
           'mic_ids': [0, 18, 29, 40, 56, 78, 92, 104, 114, 128, 134, 155,
                       178, 188, 196, 206, 216, 232, 262, 272, 282, 294, 300, 323]}}

MIRROR_SIGNS = (-1.0, -1.0, 1.0)


def mirror_positions(positions):
    """The exact mirror of each microphone about the speaker: ``(-x, -y, z)``."""
    if not isinstance(positions, torch.Tensor) or positions.shape[-1:] != torch.Size([3]):
        raise ValueError('positions must end in a 3-vector axis')
    signs = torch.tensor(MIRROR_SIGNS, dtype=positions.dtype, device=positions.device)
    return positions * signs


def _declared_frame_k(dataset, room):
    return int(dataset.k_by_room[room]) if hasattr(dataset, 'k_by_room') else 0


def mirror_stats(model, dataset, room, query_ids, frame_k, side_of_query,
                 device='cpu', batch=8):
    """Mirror cosine, opposite-side weight share and signed spectral-C50 error of one arm.

    Sides are room-frame facts: the label of a query and of a reference is the sign of its
    y before any heading roll, taken from the dataset's own room-frame coordinates. The
    geometry is expressed in the frame the dataset itself serves, so ``frame_k`` must be
    exactly the roll that dataset applies to this room.
    """
    if side_of_query not in (1, -1):
        raise ValueError('side_of_query must be +1 or -1')
    if getattr(dataset, 'eval_seed', None) is None:
        raise ValueError('the probe needs a dataset with a fixed eval_seed')
    if room not in getattr(dataset, 'data', {}):
        raise ValueError('the dataset holds no room ' + str(room))
    if int(frame_k) != _declared_frame_k(dataset, room):
        raise ValueError('frame_k {} is not the roll the dataset applies to {}'.format(
            frame_k, room))
    query_ids = [int(i) for i in query_ids]
    if not query_ids:
        raise ValueError('at least one query is required')
    if any(not 0 <= i < len(dataset) or dataset.items[i][0] != room for i in query_ids):
        raise ValueError('every query id must index an item of room ' + str(room))

    data = dataset.data[room]
    room_y = data['src_local'][:, 1]
    mic_ids = [int(dataset.items[i][1]) for i in query_ids]
    if any(int(torch.sign(room_y[mic]).item()) != side_of_query for mic in mic_ids):
        raise ValueError('every query must lie on the declared side in the room frame')

    target_device = torch.device(device)
    depth_room = data['depth_coord']
    query_room = data['src_local'][mic_ids]
    pair = torch.cat([query_room, mirror_positions(query_room)]).unsqueeze(0)
    depth_frame, _, pair_frame = rotate_scene_yaw(
        depth_room.unsqueeze(0), torch.zeros(1, 3, dtype=depth_room.dtype), pair, int(frame_k))
    depth_frame = depth_frame.squeeze(0)
    query_frame, mirror_frame = pair_frame.squeeze(0).split(len(mic_ids))

    depth_on = depth_frame.to(target_device)
    cosines = []
    for start in range(0, len(mic_ids), batch):
        stop = start + batch
        cosines.append(F.cosine_similarity(
            geometry_feature(model, query_frame[start:stop].to(target_device), depth_on),
            geometry_feature(model, mirror_frame[start:stop].to(target_device), depth_on),
            dim=-1).cpu())
    cosine = torch.cat(cosines)

    shares, errors, opposite_counts = [], [], []
    for start in range(0, len(query_ids), batch):
        chunk = query_ids[start:start + batch]
        items = [dataset[i] for i in chunk]
        _, src, depth_b, target, ref_irs, ref_locs = [
            torch.stack([item[j] for item in items]) for j in range(6)]
        if not torch.equal(depth_b[0], depth_frame):
            raise ValueError('the dataset serves a panorama this frame_k does not explain')
        opposite = torch.zeros(len(chunk), ref_locs.shape[1])
        for j, item in enumerate(chunk):
            ref_ids = torch.as_tensor(np.asarray(dataset._pick_refs(room, dataset.items[item][1])),
                                      dtype=torch.long)
            if not torch.equal(ref_irs[j], data['rirs'][ref_ids]):
                raise ValueError('the reference draw is not reproducible for item {}'.format(item))
            opposite[j] = (torch.sign(room_y[ref_ids]) == -side_of_query).float()
        out_log, tgt, weights, _ = composed_forward(
            model, depth_b.to(target_device), ref_irs.to(target_device),
            src.to(target_device), ref_locs.to(target_device), target.to(target_device))
        magnitude = weights.abs().mean(-1)
        shares.append(((magnitude * opposite.to(target_device)).sum(1)
                       / magnitude.sum(1)).cpu())
        onset = onset_frames(src.to(target_device))
        errors.append((spectral_c50(torch.exp(out_log), onset)
                       - spectral_c50(tgt, onset)).cpu())
        opposite_counts.append(opposite.sum(1))

    share, error = torch.cat(shares), torch.cat(errors)
    counts = torch.cat(opposite_counts)
    references = float(len(mic_ids) * ref_locs.shape[1])
    stats = {'room': room, 'frame_k': int(frame_k), 'side_of_query': int(side_of_query),
             'n_queries': len(mic_ids), 'mic_ids': mic_ids, 'batch': int(batch),
             'device': str(target_device),
             'mirror_cosine': float(cosine.mean()), 'mirror_cosine_sd': float(cosine.std()),
             'opposite_side_weight_share': float(share.mean()),
             'opposite_side_weight_share_sd': float(share.std()),
             'c50_signed_error_db': float(error.mean()),
             'c50_signed_error_sd': float(error.std()),
             'reference_side_counts': {'opposite': int(counts.sum()),
                                       'same': int(references - float(counts.sum()))},
             'reference_opposite_fraction': float(counts.sum()) / references}
    unusable = sorted(key for key, value in stats.items()
                      if isinstance(value, float) and not np.isfinite(value))
    if unusable:
        raise ValueError('the probe produced non-finite ' + ', '.join(unusable))
    return stats


def _load(model_factory, backbone, num_shot, checkpoint, device):
    model = model_factory(backbone, num_shot)
    model.load_state_dict(load_model_state(checkpoint), strict=True)
    return model.to(torch.device(device)).eval()


def _single_room_positions(dataset, room):
    """Positions in ``meta['test']``, which are this dataset's item ids for one room."""
    test = np.asarray(dataset.data[room]['meta']['test'])
    if [item[1] for item in dataset.items] != [int(i) for i in test]:
        raise ValueError('the probe needs a dataset of exactly this room in split order')
    return test


def legacy_reproduction(root=DEFAULT_ROOT, room=HALLWAY, device='cpu', batch=8, num_shot=8,
                        max_len=9600, cohort_size=LEGACY_COHORT_N, checkpoints=None,
                        model_factory=build_xrir_exp06, anchors=ANCHORS_LEGACY,
                        tolerance=ANCHOR_TOLERANCE, verify_frozen_cohort=True):
    """Re-run the 2026-09-14 diagnostic on its frozen cohort and report the deviation."""
    checkpoints = dict(LEGACY_CHECKPOINTS if checkpoints is None else checkpoints)
    dataset = HAADataset([room], 'test', root=root, num_shot=num_shot, max_len=max_len,
                         eval_seed=0)
    test = _single_room_positions(dataset, room)
    cohort = legacy_cohort(test, dataset.data[room]['src_local'].numpy()[test, 1],
                           n=cohort_size)['+y']
    if verify_frozen_cohort and cohort != HALLWAY_LEGACY_COHORT['+y']:
        raise ValueError('the cohort rule no longer selects the frozen 2026-09-14 microphones')
    models, deviations = {}, []
    for name in sorted(checkpoints):
        model = _load(model_factory, BACKBONE_OF[name], num_shot, checkpoints[name], device)
        with torch.no_grad():
            models[name] = mirror_stats(model, dataset, room, cohort['positions'], 0, 1,
                                        device=device, batch=batch)
        for key, anchor in sorted(anchors.get(name, {}).items()):
            deviation = models[name][key] - anchor
            models[name][key + '_anchor'] = anchor
            models[name][key + '_deviation'] = deviation
            if abs(deviation) > tolerance:
                deviations.append('{} {} {:.4f} is {:+.4f} from the 2026-09-14 anchor {}'
                                  .format(name, key, models[name][key], deviation, anchor))
    return {'room': room, 'device': str(device), 'batch': int(batch), 'cohort': cohort,
            'models': models, 'anchors': anchors, 'tolerance': tolerance,
            'reproduced': not deviations, 'deviations': deviations,
            'checkpoints': {name: str(path) for name, path in checkpoints.items()}}


def full_gate(cylor_checkpoint, heading_k, root=DEFAULT_ROOT, room=HALLWAY, device=None,
              batch=8, num_shot=8, max_len=9600, side_of_query=1, checkpoints=None,
              model_factory=build_xrir_exp06):
    """Every test microphone of one side through the four arms of 6.1, and the verdict."""
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    checkpoints = dict(LEGACY_CHECKPOINTS if checkpoints is None else checkpoints)
    paths = {'cyl_or': cylor_checkpoint, 'cyl': checkpoints['cyl'],
             'control': checkpoints['control'], 'cyl_hf': checkpoints['cyl']}
    frames = {'room': HAADataset([room], 'test', root=root, num_shot=num_shot,
                                 max_len=max_len, eval_seed=0),
              'heading': HeadingFrameDataset([room], 'test', root=root, num_shot=num_shot,
                                             max_len=max_len, eval_seed=0,
                                             k_by_room={room: int(heading_k)})}
    test = _single_room_positions(frames['room'], room)
    y = frames['room'].data[room]['src_local'].numpy()[test, 1]
    positions = [int(i) for i in np.where(y > 0 if side_of_query > 0 else y < 0)[0]]
    if not positions:
        raise ValueError('no test microphone lies on the requested side')
    cohort = {'positions': positions, 'mic_ids': [int(test[i]) for i in positions],
              'side': '+y' if side_of_query > 0 else '-y'}
    stats = {}
    for name in GATE_ARMS:
        frame = FRAME_OF[name]
        model = _load(model_factory, BACKBONE_OF[name], num_shot, paths[name], device)
        with torch.no_grad():
            stats[name] = mirror_stats(model, frames[frame], room, positions,
                                       int(heading_k) if frame == 'heading' else 0,
                                       side_of_query, device=device, batch=batch)
        stats[name].update(backbone=BACKBONE_OF[name], frame=frame,
                           checkpoint=str(paths[name]))
        del model
    return {'room': room, 'device': str(device), 'batch': int(batch), 'cohort': cohort,
            'heading_k': int(heading_k), 'stats': stats, 'decision': g1_decision(stats),
            'checkpoints': {name: str(path) for name, path in paths.items()}}
