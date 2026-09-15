"""Gate G1's mirror probe (plan v4 6.1): does the geometry branch tell the two ends apart?

Two parts. The *legacy reproduction* re-runs the 2026-09-14 hallway diagnostic on its
frozen 24-microphone cohort, so the probe's numerics can be checked against the anchors
that diagnostic published. The *full-cohort gate* runs every +y hallway test microphone
through the four models of 6.1 and decides G1.

Nothing here imports the diagnostic scripts and nothing patches a Torch global: the pinned
forward is recomposed from the model's own submodules, and the alignment the pinned method
would allocate on a GPU comes from tools.exp06_probe_align.
"""
import numbers

import numpy as np
import torch
import torch.nn.functional as F

from tools.exp06_probe_align import (SAMPLE_RATE, SPEED_OF_SOUND,  # noqa: F401  (re-export)
                                     shift_and_align_device)

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
