"""Frozen training-only heading rule, records, and heading-frame dataset."""
import json
import hashlib
import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sim_to_real.haa_dataset import HAADataset
from tools.yaw_rotation import rotate_scene_yaw
from tools.exp06_heading import (canonical_heading_deg, compensated_levels, continuous_fit,
                                early_level, heading_roll_k, mean_direction)
from tools.exp06_heading import candidate_contrasts, decide_heading, leave_one_out_stable
from tools.exp06_heading import HeadingFrameDataset
from tools.exp06_heading import estimate_room_heading
from tools.exp06_heading import read_heading_json, verify_heading_inputs, write_heading_json
from tools.provenance import closure_record, git_state, sha256_file, source_closure


@pytest.mark.parametrize('window,n', [(0.005, 110), (0.05, 1102)])
def test_early_energy_strict_onset_and_floor(window, n):
    rir = np.zeros(1200)
    rir[1], rir[2], rir[2 + n - 1], rir[2 + n] = 0.2, 1, 0.1, 0.9
    assert early_level(rir, 22050, window) == pytest.approx(1.01)
    assert early_level([0, 0, 1], 22050, window) == 1  # zero-extended tail
    assert early_level(np.zeros(1200), 22050, window) == 0


def test_distance_compensation():
    rirs = np.array([[0, 2, 0], [0, 1, 0]])
    np.testing.assert_allclose(compensated_levels(rirs, [1, 2], 1000, .005), 10 * np.log10(4))
    with pytest.raises(ValueError):
        compensated_levels(rirs, [0, 2], 1000, .005)
    with pytest.raises(ValueError):
        compensated_levels(np.zeros((2, 5)), [1, 2], 1000, .005)


@pytest.mark.parametrize('phi,k', [(-90, 128), (0, 0), (180, 256), (90, 384), (270, 128),
                                 (.35, 0), (.36, 511), (.3515625, 0),
                                 (.3515625 + .703125, 511), (-.3515625, 1)])
def test_roll_ties_and_wrap(phi, k):
    assert heading_roll_k(phi) == k
    assert canonical_heading_deg(k, 512) == -k * 360 / 512
    assert heading_roll_k(canonical_heading_deg(k, 512)) == k
    assert (heading_roll_k(phi + .703125) + 1) % 512 == k


def test_roll_general_width_and_invalid_inputs():
    assert heading_roll_k(-90, W=16) == 4
    for width in [0, -1, 2.5, True]:
        with pytest.raises(ValueError):
            heading_roll_k(0, width)
    for k in [-1, 512, .5, True]:
        with pytest.raises(ValueError):
            canonical_heading_deg(k, 512)
    with pytest.raises(ValueError):
        heading_roll_k(float('nan'))


def test_descriptive_fit_and_mean():
    theta = np.arange(-180, 180, 30)
    level = 2 + 7 * np.cos(np.deg2rad(theta + 90))
    fit = continuous_fit(theta, level)
    assert fit['phi_deg'] == -90
    assert fit['a'] == pytest.approx(2) and fit['b'] == pytest.approx(7)
    assert fit['mse'] < 1e-20
    assert continuous_fit(theta, np.ones(12))['b'] == pytest.approx(0)
    assert mean_direction([-90, 90], [2, 1]) == pytest.approx(-90)
    assert mean_direction([-90, 90], [1, 1]) is None
    with pytest.raises(ValueError):
        mean_direction([0, 90], [1, -1])


@pytest.mark.parametrize('rir,sr,w', [([], 1000, .005), ([np.nan], 1000, .005),
                                    ([1], 0, .005), ([1], 1000, .0001)])
def test_invalid_energy_inputs(rir, sr, w):
    with pytest.raises(ValueError):
        early_level(rir, sr, w)


def windows(level):
    return {'5ms': np.asarray(level), '50ms': np.asarray(level)}


@pytest.mark.parametrize('phi,axis', [(0, '+x'), (90, '+y'), (180, '-x'), (-90, '-y')])
def test_uniform_cardioid_axes(phi, axis):
    theta = np.arange(-180, 180, 15)
    amplitude = 1 + .8 * np.cos(np.deg2rad(theta - phi))
    level = 20 * np.log10(amplitude)
    assert decide_heading(theta, windows(level))[0] == axis
    assert leave_one_out_stable(theta, windows(level))


def test_hallway_axis_despite_biased_weighted_mean_and_exact_threshold():
    theta = np.array([-90] * 4 + [90] * 8)
    # These two sampled gains come from a cardioid with front/back ratio 10**(3/20).
    level = np.array([3.] * 4 + [0.] * 8)
    assert mean_direction(theta, 10 ** (level / 10)) == pytest.approx(90)
    winner, table, reason = decide_heading(theta, windows(level))
    assert winner == '-y' and reason == 'accepted'
    assert table['5ms']['runner_up_margin_db'] == 6
    assert leave_one_out_stable(theta, windows(level))


def test_singleton_and_inclusive_wrapped_sectors():
    theta = [-90] * 6 + [0, 0, 90, 90, 180, 180]
    winner, table, _ = decide_heading(theta, windows([9] * 6 + [0] * 6))
    assert winner == '-y'
    assert table['5ms']['competitor_count'] == 0
    assert table['5ms']['runner_up_margin_db'] is None
    contrast = candidate_contrasts([135, 180, -135, 0, 10, -10], [3, 3, 3, 0, 0, 0])
    assert contrast['-x'] == 3


@pytest.mark.parametrize('kind,reason', [('weak', 'contrast'), ('disagree', 'disagree'),
                                      ('margin', 'margin'), ('few', 'microphones')])
def test_decision_refusals(kind, reason):
    theta = np.repeat([0, 90, 180, -90], 3)
    level = np.repeat([0., 0., 0., 8.], 3)
    values = windows(level)
    if kind == 'weak':
        values = windows(level * .2)
    elif kind == 'disagree':
        values['50ms'] = -level
    elif kind == 'margin':
        values = windows(np.repeat([7., 0., 0., 8.], 3))
    else:
        theta, values = [0, 0, 90, 90], windows([0, 0, 8, 8])
    winner, _, why = decide_heading(theta, values)
    assert winner is None and reason in why


def test_loo_reapplies_counts_contrast_margin_and_winner():
    theta = np.repeat([0, 90, 180, -90], 3)
    level = np.array([6] * 3 + [0] * 6 + [30, 0, 0])
    assert decide_heading(theta, windows(level))[0] == '-y'
    assert decide_heading(np.delete(theta, 9), windows(np.delete(level, 9)))[0] == '+x'
    assert not leave_one_out_stable(theta, windows(level))
    theta = [-90] * 4 + [90] * 8
    assert not leave_one_out_stable(theta, windows([12, 0, 0, 0] + [0] * 8))
    theta = np.repeat([0, 90, 180, -90], 4)
    level = [4] * 4 + [0] * 8 + [14, 6, 6, 6]
    assert decide_heading(theta, windows(level))[0] == '-y'
    assert not leave_one_out_stable(theta, windows(level))  # margin collapses below 3
    with pytest.raises(ValueError):
        decide_heading(theta, {'5ms': level})


@pytest.fixture
def room_cache(tmp_path):
    room = tmp_path / 'synthetic'
    room.mkdir()
    meta = dict(train=list(range(12)), valid=[12], test=[12, 13], sr=22050)
    (room / 'meta.json').write_text(json.dumps(meta))
    np.save(room / 'rirs.npy', np.random.default_rng(6).normal(size=(14, 64)).astype('float32'))
    np.save(room / 'xyzs.npy', np.arange(42, dtype='float32').reshape(14, 3) / 4)
    np.save(room / 'speaker_xyz.npy', np.array([0, 10, 0], dtype='float32'))
    np.save(room / 'depth.npy', np.ones((256, 512), dtype='float32'))
    return room


@pytest.mark.parametrize('k', [0, 128])
def test_dataset_geometry_only_and_room_side(room_cache, k):
    kwargs = dict(rooms=['synthetic'], split='test', root=str(room_cache.parent),
                  num_shot=8, max_len=64, eval_seed=42)
    base = HAADataset(**kwargs)
    framed = HeadingFrameDataset(**kwargs, k_by_room={'synthetic': k})
    assert base.items == framed.items
    assert HeadingFrameDataset._pick_refs is HAADataset._pick_refs
    for index, (room, idx) in enumerate(base.items):
        left, right = base[index], framed[index]
        np.testing.assert_array_equal(base._pick_refs(room, idx), framed._pick_refs(room, idx))
        depth, src, refs = rotate_scene_yaw(left[2][None], left[1][None], left[5][None], k)
        expected = (left[0], src[0], depth[0], left[3], left[4], refs[0])
        assert all(torch.equal(a, b) for a, b in zip(right, expected))
        if k == 0:
            assert all(torch.equal(a, b) for a, b in zip(left, right))
        assert framed.side_label(room, idx) == int(base.data[room]['src_local'][idx, 1].sign())
    for key, value in base.data['synthetic'].items():
        other = framed.data['synthetic'][key]
        assert torch.equal(value, other) if torch.is_tensor(value) else value == other


def test_dataset_requires_room_roll(room_cache):
    for mapping in [{}, {'synthetic': 512}, {'synthetic': 1.5}]:
        with pytest.raises(ValueError):
            HeadingFrameDataset(['synthetic'], 'test', root=str(room_cache.parent), k_by_room=mapping)


@pytest.fixture
def heading_cache(room_cache):
    theta = np.deg2rad([-90] * 4 + [90] * 8)
    dist = np.arange(1, 13) / 3
    xyz = np.zeros((14, 3))
    xyz[:12, :2] = np.stack((np.cos(theta), np.sin(theta)), axis=1) * dist[:, None]
    xyz[:, 2] = 50  # Only horizontal distance compensates levels.
    np.save(room_cache / 'xyzs.npy', xyz)
    np.save(room_cache / 'speaker_xyz.npy', np.zeros(3))
    rirs = np.zeros((14, 1200))
    rirs[:12, 10] = 10 ** (np.array([9] * 4 + [0] * 8) / 20) / dist
    rirs[12:] = np.nan  # Held-out values must never enter the estimator.
    np.save(room_cache / 'rirs.npy', rirs)
    return room_cache


def test_estimate_training_only_mmap_and_source_hashes(heading_cache, monkeypatch):
    import tools.exp06_heading as heading
    original_load = np.load
    accesses = []
    class TrainingRows:
        def __getitem__(self, rows):
            np.testing.assert_array_equal(rows, np.arange(12))
            accesses.append(list(rows))
            return original_load(heading_cache / 'rirs.npy', mmap_mode='r')[rows]
    def guarded_load(path, **kwargs):
        if Path(path).name == 'rirs.npy':
            assert kwargs.get('mmap_mode') == 'r'
            return TrainingRows()
        return original_load(path, **kwargs)
    monkeypatch.setattr(heading.np, 'load', guarded_load)
    record = estimate_room_heading(heading_cache)
    assert accesses == [list(range(12))]
    assert (record['decision'], record['phi_deg'], record['k']) == ('estimated', -90, 128)
    assert record['estimator']['leave_one_out_winners'] == ['-y'] * 12
    assert record['estimator']['leave_one_out_stable'] is True
    assert [row['index'] for row in record['per_mic']] == list(range(12))
    for name, digest in record['input_sha256'].items():
        assert digest == sha256_file(heading_cache / name)
    repo = Path(__file__).resolve().parents[1]
    names = source_closure('tools.exp06_heading', repo)
    pairs = [[name, sha256_file(repo / name)] for name in names]
    assert record['source_closure']['sha256'] == hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest()
    assert record['source_closure']['basis'] == 'working_tree'
    assert len(record['descriptive']['5ms']['loo_phi_deg']) == 12


def test_override_retains_estimator_and_reason(heading_cache):
    record = estimate_room_heading(heading_cache, 90, 'documented speaker axis')
    assert (record['decision'], record['phi_deg'], record['k']) == ('override', 90, 384)
    assert record['override_reason'] == 'documented speaker axis' and record['override_deg'] == 90
    assert record['estimator']['winner'] == '-y'
    for kwargs in [dict(override_deg=90), dict(override_reason='alone'),
                   dict(override_deg=90, override_reason=' '), dict(override_deg=np.nan, override_reason='bad')]:
        with pytest.raises(ValueError):
            estimate_room_heading(heading_cache, **kwargs)


@pytest.mark.parametrize('room,contrasts,margins', [
    ('class_room', [12.9, 9.0], None), ('dampened_room', [11.4, 8.7], [17.4, 13.1]),
    ('hallway', [18.4, 11.3], [36.8, 22.6]), ('complex_room', [12.0, 10.8], None)])
def test_real_training_readback(room, contrasts, margins):
    path = Path('/home/yixunhu/data_cache/HAA_xrir') / room
    if not path.is_dir():
        pytest.skip('HAA cache absent')
    record = estimate_room_heading(path)
    assert (record['decision'], record['phi_deg'], record['k']) == ('estimated', -90, 128)
    assert record['estimator']['leave_one_out_winners'] == ['-y'] * 12
    assert record['estimator']['leave_one_out_stable'] is True
    for i, window in enumerate(['5ms', '50ms']):
        row = record['estimator']['table'][window]
        assert row['contrasts_db']['-y'] == pytest.approx(contrasts[i], abs=.2)
        if margins is None:
            assert row['competitor_count'] == 0 and row['runner_up_margin_db'] is None
        else:
            assert row['runner_up_margin_db'] == pytest.approx(margins[i], abs=.2)


def test_json_roundtrip_and_missing_fields(heading_cache, tmp_path):
    record = estimate_room_heading(heading_cache)
    path = tmp_path / 'heading.json'
    write_heading_json(path, record)
    assert read_heading_json(path) == record
    for field in record:
        broken = copy.deepcopy(record)
        del broken[field]
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError):
            read_heading_json(path)
    for field, value in [('k', -1), ('k', 512), ('k', 1.5), ('k', True), ('k', 0),
                         ('phi_deg', None), ('decision', 'unknown'), ('timestamp', 'invalid'),
                         ('input_sha256', {}), ('source_closure', {}), ('estimator', {}),
                         ('per_mic', []), ('descriptive', {})]:
        broken = copy.deepcopy(record)
        broken[field] = value
        with pytest.raises(ValueError):
            write_heading_json(path, broken)


def test_json_rejects_malformed_descriptive_numbers(heading_cache, tmp_path):
    record = estimate_room_heading(heading_cache)
    for key, value in [('continuous_fit', dict(phi_deg='bad', a=0, b=1, mse=0)),
                       ('continuous_fit', dict(phi_deg=0, a=None, b=1, mse=0)),
                       ('loo_phi_deg', ['bad'] * 12), ('loo_range_deg', [False, 0]),
                       ('mean_direction_deg', 'bad')]:
        broken = copy.deepcopy(record)
        broken['descriptive']['5ms'][key] = value
        with pytest.raises(ValueError):
            write_heading_json(tmp_path / 'bad.json', broken)


def test_refusal_record_and_override_rescue(heading_cache, tmp_path):
    # Equal compensated energies erase directional evidence in both windows.
    rirs = np.load(heading_cache / 'rirs.npy')
    rirs[:4] /= 10 ** (9 / 20)
    np.save(heading_cache / 'rirs.npy', rirs)
    refused = estimate_room_heading(heading_cache)
    assert refused['decision'] == 'refused' and refused['k'] is None and refused['phi_deg'] is None
    path = tmp_path / 'refused.json'
    write_heading_json(path, refused)
    assert read_heading_json(path) == refused
    override = estimate_room_heading(heading_cache, -90, 'independent documented orientation')
    assert override['decision'] == 'override' and override['k'] == 128
    assert override['estimator'] == refused['estimator']
    write_heading_json(path, override)
    assert read_heading_json(path) == override
    override['override_reason'] = ''
    with pytest.raises(ValueError):
        write_heading_json(path, override)


@pytest.mark.parametrize('mode', ['estimate', 'override', 'refuse'])
def test_cli(heading_cache, tmp_path, mode):
    path = tmp_path / 'cli.json'
    repo = Path(__file__).resolve().parents[1]
    args = [sys.executable, 'tools/exp06_heading.py', '--room-dir', str(heading_cache), '--out', str(path)]
    if mode == 'override':
        args += ['--heading-deg', '90', '--override-reason', 'documented orientation']
    elif mode == 'refuse':
        rirs = np.load(heading_cache / 'rirs.npy')
        rirs[:4] /= 10 ** (9 / 20)
        np.save(heading_cache / 'rirs.npy', rirs)
    completed = subprocess.run(args, cwd=repo, capture_output=True, text=True,
                               env={**os.environ, 'PYTHONPATH': str(repo)})
    assert completed.returncode == (3 if mode == 'refuse' else 0), completed.stderr
    assert path.exists()
    record = read_heading_json(path)
    assert record['decision'] == {'estimate': 'estimated', 'override': 'override', 'refuse': 'refused'}[mode]
    assert record['k'] == {'estimate': 128, 'override': 384, 'refuse': None}[mode]


def test_source_closure_carries_git_identity(heading_cache):
    record = estimate_room_heading(heading_cache)
    repo = Path(__file__).resolve().parents[1]
    state = git_state(repo)
    source = record['source_closure']
    assert source['git'] == {key: state[key] for key in
                             ['HEAD', 'dirty', 'dirty_outside_worklog', 'diff_sha256']}
    assert source['git']['HEAD'] == subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    assert len(source['git']['HEAD']) == 40 and type(source['git']['dirty']) is bool
    records, digest = closure_record([f['path'] for f in source['files']], state['HEAD'], repo)
    assert source['head_sha256'] == digest and len(digest) == 64
    assert [f['head_blob_sha256'] for f in source['files']] == [r['reviewed_blob_sha256'] for r in records]
    assert [f['sha256'] for f in source['files']] == [r['working_tree_sha256'] for r in records]
    clean = (not state['dirty_outside_worklog']
             and all(f['head_blob_sha256'] == f['sha256'] for f in source['files']))
    assert record['admissibility'] == ('confirmatory' if clean else 'diagnostic')
    assert (source['head_sha256'] == source['sha256']) == all(
        f['head_blob_sha256'] == f['sha256'] for f in source['files'])


def test_dirty_closure_record_is_diagnostic_only(heading_cache, tmp_path):
    record = estimate_room_heading(heading_cache)
    edited = copy.deepcopy(record)
    edited['source_closure']['files'][0]['head_blob_sha256'] = '0' * 64
    for admissibility, ok in [('diagnostic', True), ('confirmatory', False)]:
        broken = copy.deepcopy(edited)
        broken['admissibility'] = admissibility
        broken['source_closure']['head_sha256'] = hashlib.sha256(json.dumps(
            [[f['path'], f['head_blob_sha256']] for f in broken['source_closure']['files']],
            sort_keys=True).encode()).hexdigest()
        if ok:
            write_heading_json(tmp_path / 'diagnostic.json', broken)
            assert read_heading_json(tmp_path / 'diagnostic.json')['admissibility'] == 'diagnostic'
        else:
            with pytest.raises(ValueError):
                write_heading_json(tmp_path / 'bad.json', broken)


@pytest.mark.parametrize('field,value', [
    ('git', {}), ('git', {'HEAD': 'z' * 40, 'dirty': False, 'dirty_outside_worklog': False,
                          'diff_sha256': None}),
    ('git', {'HEAD': 'a' * 39, 'dirty': False, 'dirty_outside_worklog': False, 'diff_sha256': None}),
    ('git', {'HEAD': 'a' * 40, 'dirty': 0, 'dirty_outside_worklog': False, 'diff_sha256': None}),
    ('head_sha256', 'f' * 64), ('head_sha256', None), ('head_sha256', 'ff')])
def test_record_requires_valid_git_identity(heading_cache, tmp_path, field, value):
    record = estimate_room_heading(heading_cache)
    record['source_closure'][field] = value
    with pytest.raises(ValueError):
        write_heading_json(tmp_path / 'bad.json', record)
    missing = estimate_room_heading(heading_cache)
    del missing['source_closure'][field]
    with pytest.raises(ValueError):
        write_heading_json(tmp_path / 'bad.json', missing)


def test_verify_heading_inputs_refuses_changed_cache(heading_cache, tmp_path):
    record = estimate_room_heading(heading_cache)
    path = tmp_path / 'heading.json'
    write_heading_json(path, record)
    assert verify_heading_inputs(record, heading_cache) == record
    assert read_heading_json(path, room_dir=heading_cache) == record
    assert read_heading_json(path) == record
    (heading_cache / 'meta.json').write_text((heading_cache / 'meta.json').read_text() + ' ')
    with pytest.raises(ValueError):
        verify_heading_inputs(record, heading_cache)
    with pytest.raises(ValueError):
        read_heading_json(path, room_dir=heading_cache)
    with pytest.raises(ValueError):
        verify_heading_inputs(record, tmp_path / 'absent')


def test_cli_input_error_exits_two_without_writing(heading_cache, tmp_path):
    repo = Path(__file__).resolve().parents[1]
    path = tmp_path / 'never.json'
    for extra in [['--room-dir', str(tmp_path / 'absent'), '--out', str(path)],
                  ['--room-dir', str(heading_cache), '--out', str(path), '--heading-deg', '90'],
                  ['--room-dir', str(heading_cache), '--out', str(path), '--override-reason', 'x']]:
        completed = subprocess.run([sys.executable, 'tools/exp06_heading.py'] + extra, cwd=repo,
                                   capture_output=True, text=True,
                                   env={**os.environ, 'PYTHONPATH': str(repo)})
        assert completed.returncode == 2, completed.stderr
        assert not path.exists()
