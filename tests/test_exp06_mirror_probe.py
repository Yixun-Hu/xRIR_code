"""Gate G1's mirror probe: the cohort rule, the recomposed forward and the decision."""
import numpy as np
import pytest
import torch

from model.xRIR_cyl_oriented import build_xrir_exp06
from tools import exp06_mirror_probe as subject
from tools.exp06_probe_align import shift_and_align_device

MAX_LEN = 1829          # 1829 // 31 + 1 = 60 frames, so the late C50 window is not empty
N_BINS = 60


@pytest.fixture(scope='module')
def tiny():
    """A two-shot xRIR small enough to run on a CPU, with the pinned pooling contract."""
    torch.manual_seed(6)
    return build_xrir_exp06('simple', 2, n_bins=N_BINS, dim=32, depth=1, heads=1,
                            mlp_dim=32).eval()


def batch_of(shots=2, size=2, seed=1):
    generator = torch.Generator().manual_seed(seed)
    depth = torch.randn(size, 3, 256, 512, generator=generator)
    src = torch.rand(size, 3, generator=generator) * 3 + 1
    refs = torch.rand(size, shots, 3, generator=generator) * 3 + 1
    ref_irs = torch.randn(size, shots, MAX_LEN, generator=generator) * 0.1
    target = torch.randn(size, 1, MAX_LEN, generator=generator) * 0.1
    return depth, ref_irs, src, refs, target


# --- the 2026-09-14 cohort rule -------------------------------------------------------


def test_the_cohort_rule_reproduces_a_hand_computed_selection():
    """Sort each side by room-frame y, then take evenly spaced ranks of that order."""
    y = np.array([3.0, -1.0, 1.0, -3.0, 2.0, -2.0], dtype=np.float32)
    test_indices = np.array([10, 11, 12, 13, 14, 15])
    cohort = subject.legacy_cohort(test_indices, y, n=3)
    assert cohort['+y']['positions'] == [2, 4, 0]         # y 1, 2, 3 -> positions 2, 4, 0
    assert cohort['+y']['mic_ids'] == [12, 14, 10]
    assert cohort['-y']['positions'] == [3, 5, 1]         # y -3, -2, -1
    assert cohort['-y']['mic_ids'] == [13, 15, 11]


def test_the_cohort_rule_spaces_ranks_over_the_whole_side():
    y = np.arange(1.0, 11.0, dtype=np.float32)
    cohort = subject.legacy_cohort(np.arange(100, 110), y, n=4)
    assert cohort['+y']['positions'] == [0, 3, 6, 9]
    assert cohort['-y'] == {'positions': [], 'mic_ids': []}


@pytest.mark.parametrize('case', ['length', 'dims', 'too_few', 'n', 'nonfinite'])
def test_the_cohort_rule_refuses_malformed_input(case):
    y = np.array([1.0, -1.0, 2.0, -2.0])
    ids = np.arange(4)
    kwargs = {'n': 2}
    if case == 'length':
        ids = np.arange(3)
    elif case == 'dims':
        y = y.reshape(2, 2)
        ids = ids.reshape(2, 2)
    elif case == 'too_few':
        kwargs['n'] = 3
    elif case == 'n':
        kwargs['n'] = 0
    else:
        y = np.array([1.0, np.nan, 2.0, -2.0])
    with pytest.raises(ValueError):
        subject.legacy_cohort(ids, y, **kwargs)


# --- the spectral C50 of the diagnostic -----------------------------------------------


def test_spectral_c50_is_the_early_to_late_energy_ratio():
    """Ten units of energy per early frame, one per late frame, from the onset on."""
    frames = 200
    magnitude = torch.zeros(2, 63, frames)
    onset = torch.tensor([4, 10])
    n50 = subject.C50_FRAMES
    for b in range(2):
        start = int(onset[b])
        magnitude[b, 0, start:start + n50] = np.sqrt(10.0)
        magnitude[b, 0, start + n50:] = 1.0
    late = [frames - int(onset[b]) - n50 for b in range(2)]
    expected = [10 * np.log10(10.0 * n50 / (late[b] + 1e-12)) for b in range(2)]
    got = subject.spectral_c50(magnitude, onset)
    assert got.shape == (2,)
    assert np.allclose(got.numpy(), expected, rtol=1e-5)
    assert subject.C50_FRAMES == round(0.05 * 22050 / 31)


def test_spectral_c50_refuses_a_bad_shape_or_onset():
    with pytest.raises(ValueError):
        subject.spectral_c50(torch.zeros(0, 63, 100), torch.zeros(0, dtype=torch.long))
    with pytest.raises(ValueError):
        subject.spectral_c50(torch.zeros(2, 63), torch.tensor([0, 0]))
    with pytest.raises(ValueError):
        subject.spectral_c50(torch.zeros(2, 63, 100), torch.tensor([0]))
    with pytest.raises(ValueError):
        subject.spectral_c50(torch.zeros(2, 63, 100), torch.tensor([0, -1]))


def test_onset_frames_are_the_direct_path_in_hops():
    src = torch.tensor([[3.43, 0.0, 0.0], [0.0, 6.86, 0.0]])
    assert subject.onset_frames(src).tolist() == [
        round(3.43 / 343.0 * 22050 / 31), round(6.86 / 343.0 * 22050 / 31)]


# --- the geometry branch and the recomposed forward -----------------------------------


@torch.no_grad()
def test_geometry_feature_is_the_pinned_pooled_source_branch(tiny):
    depth = torch.randn(3, 256, 512, generator=torch.Generator().manual_seed(4))
    positions = torch.tensor([[1.0, 2.0, 0.5], [-1.0, -2.0, 0.5]])
    feature = subject.geometry_feature(tiny, positions, depth)
    encoded = (positions[:, :, None, None] - depth[None]) / 5.
    expected = tiny.lin_proj_0(tiny.src_proj(tiny.source_network(encoded)).permute(0, 2, 1))
    assert feature.shape == (2, 256)
    assert torch.equal(feature, expected.squeeze(-1))


@torch.no_grad()
def test_the_recomposed_forward_mixes_the_reference_log_spectrograms(tiny):
    depth, ref_irs, src, refs, target = batch_of()
    out_log, tgt, weights, source_out = subject.composed_forward(
        tiny, depth, ref_irs, src, refs, target)
    assert out_log.shape == (2, 63, N_BINS) and tgt.shape == (2, 63, N_BINS)
    assert weights.shape == (2, 2, N_BINS) and source_out.shape == (2, 256)
    aligned = shift_and_align_device(tiny, ref_irs, src, refs)
    logs = torch.cat([torch.log(tiny.convert_ir_to_spec(aligned[:, i:i + 1]) + 1e-8)
                      for i in range(refs.shape[1])], dim=1)
    assert torch.equal(out_log, torch.sum(logs * weights.unsqueeze(2), dim=1))
    assert torch.equal(tgt, tiny.convert_ir_to_spec(target)[:, 0])


@torch.no_grad()
def test_the_recomposed_forward_puts_the_time_grid_on_the_inputs_device(tiny):
    """The pinned forward transfers `model.times`; so does this one (plan v4 R3.2)."""
    depth, ref_irs, src, refs, target = batch_of()
    seen = {}
    original = tiny.time_embedder.forward

    def spy(x):
        seen['device'] = x.device
        return original(x)

    tiny.time_embedder.forward = spy
    try:
        subject.composed_forward(tiny, depth, ref_irs, src, refs, target)
    finally:
        tiny.time_embedder.forward = original
    assert seen['device'] == refs.device


@torch.no_grad()
def test_the_recomposed_forward_uses_the_injected_alignment(tiny):
    depth, ref_irs, src, refs, target = batch_of()
    calls = []

    def align(model, x, src_loc, ref_locs):
        calls.append(model)
        return shift_and_align_device(model, x, src_loc, ref_locs)

    subject.composed_forward(tiny, depth, ref_irs, src, refs, target, align=align)
    assert calls == [tiny]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
@torch.no_grad()
def test_cuda_parity_with_the_pinned_forward():
    model = build_xrir_exp06('simple', 2, n_bins=310, dim=32, depth=1, heads=1,
                             mlp_dim=32).cuda().eval()
    generator = torch.Generator().manual_seed(9)
    depth = torch.randn(2, 3, 256, 512, generator=generator).cuda()
    src = (torch.rand(2, 3, generator=generator) * 3 + 1).cuda()
    refs = (torch.rand(2, 2, 3, generator=generator) * 3 + 1).cuda()
    ref_irs = (torch.randn(2, 2, 9600, generator=generator) * 0.1).cuda()
    target = (torch.randn(2, 1, 9600, generator=generator) * 0.1).cuda()
    out_log, tgt, _, _ = subject.composed_forward(model, depth, ref_irs, src, refs, target)
    pinned_log, pinned_tgt = model(depth, ref_irs, src, refs, target)
    assert torch.allclose(out_log.unsqueeze(-1), pinned_log, atol=1e-5, rtol=1e-5)
    assert torch.allclose(tgt, pinned_tgt[:, 0], atol=1e-5, rtol=1e-5)


# --- the decision ---------------------------------------------------------------------


def stats(cyl_cos=0.97, cyl_share=0.85, control_cos=0.44, control_share=0.24,
          cylor_cos=0.5, cylor_share=0.3):
    def cell(cosine, share):
        return {'mirror_cosine': cosine, 'opposite_side_weight_share': share}
    return {'cyl': cell(cyl_cos, cyl_share), 'control': cell(control_cos, control_share),
            'cyl_or': cell(cylor_cos, cylor_share)}


def test_the_gate_passes_when_the_anchors_bracket_and_the_oriented_model_is_below():
    decision = subject.g1_decision(stats())
    assert decision['outcome'] == 'pass' and decision['reasons'] == []
    assert decision['thresholds'] == subject.G1_THRESHOLDS


@pytest.mark.parametrize('override', [{'cylor_cos': 0.85}, {'cylor_share': 0.6},
                                      {'cylor_cos': 0.80}, {'cylor_share': 0.50}])
def test_the_gate_fails_when_the_oriented_model_still_mirrors(override):
    decision = subject.g1_decision(stats(**override))
    assert decision['outcome'] == 'fail' and decision['reasons']


@pytest.mark.parametrize('override', [{'cyl_cos': 0.80}, {'cyl_cos': 0.90},
                                      {'control_cos': 0.70}, {'control_cos': 0.60},
                                      {'cyl_share': 0.65}, {'cyl_share': 0.70},
                                      {'control_share': 0.40}, {'control_share': 0.35}])
def test_the_gate_is_inconclusive_when_the_anchors_do_not_bracket(override):
    decision = subject.g1_decision(stats(**override))
    assert decision['outcome'] == 'inconclusive' and decision['reasons']


def test_the_gate_refuses_incomplete_statistics():
    incomplete = stats()
    del incomplete['cyl_or']
    with pytest.raises(ValueError):
        subject.g1_decision(incomplete)
    with pytest.raises(ValueError):
        subject.g1_decision(stats(cyl_cos=float('nan')))


# --- the probe on a real dataset -------------------------------------------------------

import json
import os
from pathlib import Path

from sim_to_real.haa_dataset import HAADataset
from tools import provenance
from tools.exp06_heading import HeadingFrameDataset, estimate_room_heading, write_heading_json
from tools.yaw_rotation import rotate_scene_yaw

ROOT = Path(__file__).resolve().parents[1]
HEADING_K = 128
NUM_SHOT = 2


def write_probe_room(room):
    """A cache room with a decidable axis and four test microphones on each side."""
    room.mkdir(parents=True)
    train, test = list(range(12)), list(range(12, 20))
    room.joinpath('meta.json').write_text(json.dumps(
        dict(train=train, valid=[12], test=test, sr=22050)))
    theta = np.deg2rad([-90] * 4 + [90] * 8)
    distance = np.arange(1, 13) / 3
    xyz = np.zeros((20, 3))
    xyz[:12, :2] = np.stack((np.cos(theta), np.sin(theta)), axis=1) * distance[:, None]
    for i, (x, y) in enumerate([(2.0, 1.0), (2.5, 2.0), (1.5, 3.0), (3.0, 4.0)]):
        xyz[12 + i, :2] = (x, y)
        xyz[16 + i, :2] = (-x, -y)
    xyz[:, 2] = 0.3
    np.save(room / 'xyzs.npy', xyz)
    np.save(room / 'speaker_xyz.npy', np.zeros(3))
    generator = np.random.default_rng(6)
    rirs = generator.normal(0, 1e-3, (20, MAX_LEN))
    rirs[:12, 10] += 10 ** (np.asarray([9] * 4 + [0] * 8) / 20) / distance
    np.save(room / 'rirs.npy', rirs.astype(np.float32))
    np.save(room / 'depth.npy', (np.arange(256 * 512) % 7 + 1).reshape(256, 512).astype('float32'))
    return room


@pytest.fixture(scope='module')
def probe_cache(tmp_path_factory, tiny):
    """One room, its confirmatory heading record, and checkpoints of the tiny model."""
    from test_exp06_haa import confirmatory
    base = tmp_path_factory.mktemp('probe')
    root = base / 'HAA_xrir'
    record = estimate_room_heading(write_probe_room(root / 'hallway'))
    assert record['decision'] == 'estimated' and record['k'] == HEADING_K
    heading = base / 'hallway.json'
    write_heading_json(heading, confirmatory(record))
    checkpoints = {}
    for name in ('cyl_or', 'cyl', 'control'):
        path = base / (name + '.pth')
        torch.save(tiny.state_dict(), path)
        checkpoints[name] = str(path)
    return {'root': str(root), 'heading': str(heading), 'checkpoints': checkpoints,
            'base': base}


def probe_factory(backbone, num_shot):
    torch.manual_seed(6)
    return build_xrir_exp06('simple', num_shot, n_bins=N_BINS, dim=32, depth=1, heads=1,
                            mlp_dim=32)


def dataset_of(root, k=None):
    if k is None:
        return HAADataset(['hallway'], 'test', root=root, num_shot=NUM_SHOT,
                          max_len=MAX_LEN, eval_seed=0)
    return HeadingFrameDataset(['hallway'], 'test', root=root, num_shot=NUM_SHOT,
                               max_len=MAX_LEN, eval_seed=0, k_by_room={'hallway': k})


@pytest.mark.parametrize('k', [0, HEADING_K])
def test_a_mirror_pair_stays_a_mirror_pair_in_the_heading_frame(k):
    """(-x, -y, z) is Rz(180 deg), which commutes with the frame's own yaw."""
    positions = torch.tensor([[2.0, 1.0, 0.3], [-1.5, 3.0, 0.3]])
    depth = torch.ones(1, 3, 256, 512)
    zero = torch.zeros(1, 3)
    _, _, rotated = rotate_scene_yaw(depth, zero, positions.unsqueeze(0), k)
    _, _, rotated_mirror = rotate_scene_yaw(
        depth, zero, subject.mirror_positions(positions).unsqueeze(0), k)
    assert torch.allclose(subject.mirror_positions(rotated.squeeze(0)),
                          rotated_mirror.squeeze(0), atol=1e-6)


@torch.no_grad()
def test_mirror_stats_reports_the_cohort_and_finite_statistics(probe_cache, tiny):
    dataset = dataset_of(probe_cache['root'])
    stats = subject.mirror_stats(tiny, dataset, 'hallway', [0, 1, 2, 3], 0, 1, batch=2)
    assert stats['n_queries'] == 4 and stats['mic_ids'] == [12, 13, 14, 15]
    assert stats['side_of_query'] == 1 and stats['frame_k'] == 0 and stats['device'] == 'cpu'
    assert -1.0 <= stats['mirror_cosine'] <= 1.0
    assert 0.0 <= stats['opposite_side_weight_share'] <= 1.0
    assert np.isfinite(stats['c50_signed_error_db'])
    assert 0.0 <= stats['reference_opposite_fraction'] <= 1.0


@torch.no_grad()
def test_mirror_stats_mirror_cosine_matches_the_manual_computation(probe_cache, tiny):
    dataset = dataset_of(probe_cache['root'])
    stats = subject.mirror_stats(tiny, dataset, 'hallway', [0, 1], 0, 1, batch=2)
    data = dataset.data['hallway']
    query = data['src_local'][[12, 13]]
    feature = subject.geometry_feature(tiny, query, data['depth_coord'])
    mirror = subject.geometry_feature(tiny, subject.mirror_positions(query),
                                      data['depth_coord'])
    expected = torch.nn.functional.cosine_similarity(feature, mirror, dim=-1).mean()
    assert stats['mirror_cosine'] == pytest.approx(float(expected), abs=1e-6)


@torch.no_grad()
def test_side_labels_and_references_are_the_same_in_both_frames(probe_cache, tiny):
    """Side is a room-frame fact: rolling the panorama cannot move a microphone's end."""
    room_frame = subject.mirror_stats(tiny, dataset_of(probe_cache['root']), 'hallway',
                                      [0, 1, 2, 3], 0, 1, batch=2)
    heading = subject.mirror_stats(tiny, dataset_of(probe_cache['root'], HEADING_K),
                                   'hallway', [0, 1, 2, 3], HEADING_K, 1, batch=2)
    assert heading['mic_ids'] == room_frame['mic_ids']
    assert heading['reference_side_counts'] == room_frame['reference_side_counts']
    assert heading['reference_opposite_fraction'] == room_frame['reference_opposite_fraction']
    assert heading['frame_k'] == HEADING_K


@torch.no_grad()
@pytest.mark.parametrize('case', ['frame_mismatch', 'room_frame_k', 'wrong_side',
                                  'unknown_room', 'random_references', 'empty'])
def test_mirror_stats_refuses_an_inconsistent_request(probe_cache, tiny, case):
    root = probe_cache['root']
    dataset, ids, k, side = dataset_of(root), [0, 1], 0, 1
    if case == 'frame_mismatch':
        dataset, k = dataset_of(root, HEADING_K), 0
    elif case == 'room_frame_k':
        k = HEADING_K
    elif case == 'wrong_side':
        side = -1
    elif case == 'unknown_room':
        dataset = dataset_of(root)
    elif case == 'random_references':
        dataset = HAADataset(['hallway'], 'test', root=root, num_shot=NUM_SHOT,
                             max_len=MAX_LEN)
    else:
        ids = []
    room = 'class_room' if case == 'unknown_room' else 'hallway'
    with pytest.raises(ValueError):
        subject.mirror_stats(tiny, dataset, room, ids, k, side, batch=2)


@torch.no_grad()
def test_the_legacy_reproduction_reports_its_deviation_from_the_anchors(probe_cache):
    """With a tiny model the anchors cannot reproduce; the report says so, fail-closed."""
    report = subject.legacy_reproduction(
        root=probe_cache['root'], room='hallway', cohort_size=4, num_shot=NUM_SHOT,
        max_len=MAX_LEN, batch=2, model_factory=probe_factory,
        checkpoints={'cyl': probe_cache['checkpoints']['cyl'],
                     'control': probe_cache['checkpoints']['control']},
        verify_frozen_cohort=False)
    assert set(report['models']) == {'cyl', 'control'}
    assert report['reproduced'] is False and report['deviations']
    assert report['cohort']['mic_ids'] == [12, 13, 14, 15]
    assert report['anchors'] == subject.ANCHORS_LEGACY


@torch.no_grad()
def test_the_full_gate_runs_every_arm_and_decides(probe_cache):
    gate = subject.full_gate(
        probe_cache['checkpoints']['cyl_or'], HEADING_K, root=probe_cache['root'],
        room='hallway', device='cpu', batch=2, num_shot=NUM_SHOT, max_len=MAX_LEN,
        model_factory=probe_factory,
        checkpoints={'cyl': probe_cache['checkpoints']['cyl'],
                     'control': probe_cache['checkpoints']['control']})
    assert set(gate['stats']) == {'cyl_or', 'cyl', 'control', 'cyl_hf'}
    assert gate['stats']['cyl_or']['frame_k'] == HEADING_K
    assert gate['stats']['cyl']['frame_k'] == 0 and gate['stats']['cyl_hf']['frame_k'] == HEADING_K
    assert gate['cohort']['mic_ids'] == [12, 13, 14, 15]
    assert gate['decision']['outcome'] in ('pass', 'fail', 'inconclusive')
    assert gate['device'] == 'cpu'


# --- the hash-bound record -------------------------------------------------------------


def canned(monkeypatch, outcome='pass', reproduced=True):
    cells = {'cyl': {'mirror_cosine': 0.97, 'opposite_side_weight_share': 0.85},
             'control': {'mirror_cosine': 0.44, 'opposite_side_weight_share': 0.24},
             'cyl_or': {'mirror_cosine': 0.5 if outcome == 'pass' else 0.95,
                        'opposite_side_weight_share': 0.3},
             'cyl_hf': {'mirror_cosine': 0.9, 'opposite_side_weight_share': 0.8}}
    monkeypatch.setattr(subject, 'full_gate', lambda *a, **k: {
        'stats': cells, 'cohort': {'positions': [0], 'mic_ids': [12], 'side': '+y'},
        'decision': subject.g1_decision(cells), 'device': 'cpu'})
    monkeypatch.setattr(subject, 'legacy_reproduction', lambda *a, **k: {
        'models': {}, 'reproduced': reproduced, 'deviations': [],
        'anchors': subject.ANCHORS_LEGACY,
        'cohort': {'positions': [], 'mic_ids': []}})


def cli(cache, out, *extra):
    return ['--cylor-checkpoint', cache['checkpoints']['cyl_or'],
            '--heading-json', cache['heading'], '--haa-root', cache['root'],
            '--room', 'hallway', '--device', 'cpu', '--out', str(out),
            '--cyl-checkpoint', cache['checkpoints']['cyl'],
            '--control-checkpoint', cache['checkpoints']['control'], *extra]


def test_the_record_binds_every_input_and_refuses_to_overwrite(probe_cache, tmp_path,
                                                               monkeypatch):
    canned(monkeypatch)
    out = tmp_path / 'gate_g1.json'
    assert subject.main(cli(probe_cache, out)) == 0
    record = json.loads(out.read_text())
    assert record['decision']['outcome'] == 'pass'
    for name in ('cyl_or', 'cyl', 'control'):
        binding = record['checkpoints'][name]
        assert binding['sha256'] == provenance.sha256_file(
            probe_cache['checkpoints'][name]) and len(binding['sha256']) == 64
    assert record['heading']['sha256'] == provenance.sha256_file(probe_cache['heading'])
    assert record['heading']['k'] == HEADING_K and record['heading']['room'] == 'hallway'
    assert record['source_closure']['entry_module'] == 'tools.exp06_mirror_probe'
    assert 'tools/exp06_mirror_probe.py' in [f['path'] for f in
                                             record['source_closure']['files']]
    assert record['source_closure']['git']['HEAD'] and record['environment']['torch']
    assert record['timestamp'] and record['legacy_reproduction']['reproduced'] is True
    with pytest.raises(FileExistsError):
        subject.main(cli(probe_cache, out))


def test_a_failed_legacy_reproduction_makes_the_gate_inconclusive(probe_cache, tmp_path,
                                                                  monkeypatch):
    canned(monkeypatch, reproduced=False)
    out = tmp_path / 'gate.json'
    assert subject.main(cli(probe_cache, out)) == 4
    record = json.loads(out.read_text())
    assert record['decision']['outcome'] == 'inconclusive'
    assert any('legacy' in reason for reason in record['decision']['reasons'])


def test_legacy_only_writes_no_verdict(probe_cache, tmp_path, monkeypatch):
    canned(monkeypatch)
    out = tmp_path / 'legacy.json'
    assert subject.main(cli(probe_cache, out, '--legacy-only')) == 4
    record = json.loads(out.read_text())
    assert record['stats'] is None and record['decision']['outcome'] == 'inconclusive'


def test_a_heading_whose_cache_moved_is_refused(probe_cache, tmp_path, monkeypatch):
    canned(monkeypatch)
    moved = tmp_path / 'HAA_xrir'
    moved.mkdir()
    room = moved / 'hallway'
    room.mkdir()
    for name in ('meta.json', 'xyzs.npy', 'speaker_xyz.npy', 'depth.npy'):
        room.joinpath(name).write_bytes(
            Path(probe_cache['root'], 'hallway', name).read_bytes())
    np.save(room / 'rirs.npy', np.zeros((20, MAX_LEN), dtype=np.float32))
    argv = cli(probe_cache, tmp_path / 'g.json')
    argv[argv.index('--haa-root') + 1] = str(moved)
    with pytest.raises(SystemExit):
        subject.main(argv)


# --- the frozen hallway cohort and the published anchors --------------------------------

HALLWAY_CACHE = Path(os.environ.get('HAA_XRIR_ROOT',
                                    os.path.expanduser('~/data_cache/HAA_xrir')), 'hallway')


@pytest.mark.skipif(not HALLWAY_CACHE.is_dir(), reason='needs the HAA cache')
def test_the_frozen_cohort_is_the_rule_applied_to_the_cache():
    meta = json.loads((HALLWAY_CACHE / 'meta.json').read_text())
    xyz = np.load(HALLWAY_CACHE / 'xyzs.npy').astype(np.float32)
    speaker = np.load(HALLWAY_CACHE / 'speaker_xyz.npy').astype(np.float32)
    test = np.asarray(meta['test'])
    cohort = subject.legacy_cohort(test, (xyz - speaker[None])[test, 1])
    assert cohort == subject.HALLWAY_LEGACY_COHORT
    assert len(subject.HALLWAY_LEGACY_COHORT['+y']['mic_ids']) == 24


@pytest.mark.skipif(os.environ.get('EXP06_REAL_PROBE') != '1',
                    reason='set EXP06_REAL_PROBE=1 (two minutes of CPU, real checkpoints)')
@torch.no_grad()
def test_the_legacy_reproduction_matches_the_published_anchors():
    report = subject.legacy_reproduction(device='cpu')
    assert report['cohort']['mic_ids'] == subject.HALLWAY_LEGACY_COHORT['+y']['mic_ids']
    assert report['reproduced'] is True, report['deviations']
