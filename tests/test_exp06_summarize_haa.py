"""exp_06's HAA summariser: the legacy receipt, the new-arm branch and the verdicts.

The synthetic trees below are exp_02-shaped (arms A and B) and exp_06-shaped (C, D, F);
the exp_06 children are finalised by ``tools.exp06_finalize`` itself, so their
``completion.json`` files are the real thing rather than a fixture's idea of one.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sim_to_real import summarize_haa as legacy
from tools import exp06_summarize_haa as subject

ROOMS = tuple(legacy.ROOMS)
SIZE = {'class_room': 5, 'dampened_room': 4, 'hallway': 6, 'complex_room': 4}
LEGACY_INITS = ('released', 'released_repomaps', 'control', 'cyl')


def indices(room):
    return list(range(SIZE[room]))


def per_sample(room, backbone, checkpoint, offset=0.0, invalid=()):
    """One exp_02 per-sample file: the writer's arrays, plus this arm's offset."""
    idx = indices(room)
    def column(base):
        # The per-query jitter depends on the arm's offset, so a paired difference varies
        # across queries and the bootstrap interval is never degenerate.
        return [float('nan') if i in invalid else
                base + offset + 0.05 * i + 0.2 * math.sin(3.0 * i + 17.0 * offset)
                for i in idx]
    t60 = [float('nan')] * len(idx) if room == 'dampened_room' else column(3.0)
    return {'index': idx, 'ir_path': ['{}/{}'.format(room, i) for i in idx],
            'meta': {'backbone': backbone, 'checkpoint': checkpoint, 'split': 'test',
                     'num_shot': 8, 'eval_seed': 0},
            'edt': column(0.06), 'c50': column(1.0), 't60': t60,
            'stft_mse': column(0.4), 'loss': column(0.5), 'env': column(5.0)}


def metrics_file(room, backbone, checkpoint, per):
    finite = {key: [v for v in per[key] if not math.isnan(v)] for key in ('edt', 'c50', 't60')}
    def block(values):
        return {'mean': float(np.mean(values)) if values else None,
                'median': float(np.median(values)) if values else None, 'n': len(values)}
    return {'backbone': backbone, 'checkpoint': checkpoint, 'room': room, 'split': 'test',
            'num_shot': 8, 'eval_seed': 0, 'depth_variant': 'default',
            'n_samples': len(per['index']), 'edt_error_s': block(finite['edt']),
            'c50_error_db': block(finite['c50']),
            't60_error_pct': None if room == 'dampened_room' else block(finite['t60']),
            'edt_invalid': 0, 'c50_outliers': 0, 't60_invalid': 0,
            'env_error': block(finite['edt']), 'stft_log_mse': block(finite['edt']),
            'test_loss': 0.5, 'elapsed_min': 1.0}


def write_eval(directory, room, backbone, checkpoint, offset=0.0, invalid=()):
    directory.mkdir(parents=True, exist_ok=True)
    per = per_sample(room, backbone, checkpoint, offset, invalid)
    (directory / 'per_sample_{}.json'.format(room)).write_text(json.dumps(per))
    (directory / 'metrics_{}.json'.format(room)).write_text(
        json.dumps(metrics_file(room, backbone, checkpoint, per)))
    return per


def write_stage(directory, backbone, seed, init, rooms, depth_variant='default'):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'args.json').write_text(json.dumps(
        {'backbone': backbone, 'seed': seed, 'init': init, 'rooms': list(rooms),
         'depth_variant': depth_variant, 'num_shot': 8, 'eval_seed': 0, 'epochs': 4}))
    (directory / 'summary.json').write_text(json.dumps(
        {'backbone': backbone, 'seed': seed, 'init': init, 'rooms': list(rooms),
         'best_epoch': 2, 'best_val_loss': 0.4, 'init_val_loss': 0.6, 'epochs': 4,
         'minutes': 1.0, 'val_rooms': list(rooms)}))


OFFSETS = {'released': 0.2, 'released_repomaps': 0.25, 'control': 0.0, 'cyl': 0.1}


def build_legacy_root(root, offsets=None):
    """A faithful exp_02 tree: the four inits, three seeds each, and the canonical record."""
    offsets = OFFSETS if offsets is None else offsets
    root = Path(root)
    for init in LEGACY_INITS:
        backbone = legacy.BACKBONE_OF[init]
        seeds = ('seed0',) if init == 'released_repomaps' else subject.SEEDS
        variant = legacy.DEPTH_VARIANT_OF.get(init, 'default')
        for seed in seeds:
            number = int(seed[len('seed'):])
            base = root / init / seed
            write_stage(base / 'stage1', backbone, number, legacy.INIT_OF[init], ROOMS, variant)
            for room in ROOMS:
                write_stage(base / ('stage2_' + room), backbone, number,
                            str(base / 'stage1' / 'best.pth'), [room], variant)
                write_eval(base / 'eval', room, backbone,
                           str(base / ('stage2_' + room) / 'best.pth'),
                           offsets[init] + 0.01 * number)
        if init != 'released_repomaps':
            for room in ROOMS:
                write_eval(root / init / 'zeroshot', room, backbone,
                           legacy.INIT_OF[init], offsets[init] + 0.5)
    (root / 'stats.json').write_text(json.dumps({'paired': []}))
    (root / 'summary.txt').write_text('exp_02 canonical summary\n')
    return root


def build_cache(root):
    """The four rooms' meta.json, so exp_02's completeness reads these test indices."""
    root = Path(root)
    for room in ROOMS:
        (root / room).mkdir(parents=True, exist_ok=True)
        (root / room / 'meta.json').write_text(json.dumps(
            {'train': [], 'test': indices(room)}))
        xyz = np.zeros((SIZE[room], 3), dtype='float32')
        xyz[:, 1] = [1.0 if i % 2 else -1.0 for i in indices(room)]
        np.save(root / room / 'xyzs.npy', xyz)
        np.save(root / room / 'speaker_xyz.npy', np.zeros(3, dtype='float32'))
    return root


@pytest.fixture
def legacy_root(tmp_path, monkeypatch):
    cache = build_cache(tmp_path / 'HAA_xrir')
    monkeypatch.setattr(legacy, 'HAA_ROOT', str(cache))
    return build_legacy_root(tmp_path / 'sim2real')


# --- the tables of 6.2 ------------------------------------------------------------------


def test_the_arm_table_is_the_five_arms_of_section_2_3():
    assert tuple(subject.ARMS) == ('control', 'cyl', 'cyl_or', 'control_hf', 'cyl_hf')
    assert subject.LEGACY_ARMS == ('control', 'cyl')
    assert subject.NEW_ARMS == ('cyl_or', 'control_hf', 'cyl_hf')
    assert [subject.ARMS[a]['backbone'] for a in subject.ARMS] == [
        'simple', 'cylindrical', 'cylindrical_oriented', 'simple', 'cylindrical']
    assert [subject.ARMS[a]['frame'] for a in subject.ARMS] == [
        'room', 'room', 'heading', 'heading', 'heading']
    assert subject.ARMS['cyl_or']['root'] == 'ckpt/exp06/sim2real/cyl_or'


def test_the_registered_constants_are_frozen():
    assert subject.H1_MARGIN_DB == 0.23 and (subject.H1_ROOM, subject.H1_METRIC) == (
        'hallway', 'c50')
    assert subject.HEADING_K == 128
    assert subject.SEEDS == ('seed0', 'seed1', 'seed2') and subject.ZEROSHOT == 'zeroshot'
    assert len(subject.CELLS) == 11 and subject.H2_FAMILY == 11
    assert subject.N_BOOT == 10000 and subject.N_BOOT_ADJUSTED == 50000


# --- the legacy receipt ------------------------------------------------------------------


def test_the_receipt_enumerates_every_retained_artifact_of_both_arms(legacy_root):
    names = subject.legacy_artifacts()
    assert len(names) == len(subject.SEEDS) * (2 * 5 + 2 * 4) + 2 * 4
    assert 'seed0/stage1/args.json' in names and 'seed2/stage2_hallway/summary.json' in names
    assert 'seed1/eval/per_sample_class_room.json' in names
    assert 'zeroshot/metrics_complex_room.json' in names
    files = subject.legacy_receipt_files(legacy_root)
    paths = [item['path'] for item in files]
    assert paths[-2:] == ['stats.json', 'summary.txt']
    assert len(paths) == 2 * len(names) + 2 and len(set(paths)) == len(paths)
    assert all(len(item['sha256']) == 64 for item in files)


def test_the_receipt_is_written_once_labelled_reconstructed(legacy_root, tmp_path):
    out = tmp_path / 'receipt' / 'legacy_receipt.json'
    record, digest = subject.write_legacy_receipt(out, legacy_root, strict=False)
    assert record['label'] == 'reconstructed' and record['arms'] == ['control', 'cyl']
    assert record['timestamp'] and len(record['source_closure']['sha256']) == 64
    assert record['source_closure']['entry_module'] == 'tools.exp06_summarize_haa'
    assert digest == subject.provenance.sha256_file(out)
    with pytest.raises(FileExistsError):
        subject.write_legacy_receipt(out, legacy_root, strict=False)


def test_a_faithful_tree_verifies_against_its_receipt(legacy_root, tmp_path):
    out = tmp_path / 'legacy_receipt.json'
    _, digest = subject.write_legacy_receipt(out, legacy_root, strict=False)
    record = subject.verify_legacy_receipt(
        out, legacy_root, {'path': str(out), 'sha256': digest})
    assert record['sha256'] == digest and len(record['files']) > 40


def test_a_tampered_receipt_or_a_changed_artifact_is_refused(legacy_root, tmp_path):
    out = tmp_path / 'legacy_receipt.json'
    _, digest = subject.write_legacy_receipt(out, legacy_root, strict=False)
    approved = {'path': str(out), 'sha256': digest}
    changed = json.loads(out.read_text())
    changed['files'][3]['sha256'] = 'f' * 64
    (tmp_path / 'tampered.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='files_sha256'):
        subject.verify_legacy_receipt(tmp_path / 'tampered.json', legacy_root)
    with pytest.raises(ValueError, match='approved'):
        subject.verify_legacy_receipt(tmp_path / 'tampered.json', legacy_root, approved)
    target = Path(legacy_root) / 'control/seed1/eval/per_sample_hallway.json'
    target.write_text(target.read_text() + ' ')
    with pytest.raises(ValueError, match='changed since the receipt'):
        subject.verify_legacy_receipt(out, legacy_root, approved)


def test_a_receipt_of_an_incomplete_arm_is_never_written(legacy_root, tmp_path):
    (Path(legacy_root) / 'cyl/seed2/eval/metrics_hallway.json').unlink()
    with pytest.raises(ValueError, match='cyl has no cyl/seed2/eval/metrics_hallway'):
        subject.write_legacy_receipt(tmp_path / 'r.json', legacy_root, strict=False)


# --- exp_06's own arms ------------------------------------------------------------------

STAMP = '2026-09-15T00:00:00+00:00'
CLOSURE = 'c' * 64
HEADING = {room: {'k': subject.HEADING_K, 'phi_deg': -90.0, 'decision': 'estimated',
                  'path': '/heading/{}.json'.format(room), 'sha256': 'b' * 64}
           for room in ROOMS}


def sha(path):
    return subject.provenance.sha256_file(path)


def new_child(job_dir, name, arm, seed, offset, closure=CLOSURE, invalid=(), heading=None,
              overrides=None):
    """One exp_06 child, finalised through the finalizer's own writer and schema."""
    cfg = subject.ARMS[arm]
    heading = (HEADING if cfg['frame'] == 'heading' else None) if heading is None else heading
    role = subject.finalizer.child_role(name)
    path = Path(job_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    rooms = [name.rsplit('/', 1)[-1]] if role == 'haa_eval' or name.startswith('stage2_') \
        else list(ROOMS)
    if name.startswith('stage2_'):
        rooms = [name[len('stage2_'):]]
    args = {'backbone': cfg['backbone'], 'frame': cfg['frame'], 'seed': seed, 'rooms': rooms,
            'num_shot': 8, 'eval_seed': 0, 'heading': heading, 'haa_root': str(job_dir),
            'init': 'ckpt/exp06/init.pth', 'split': 'test'}
    (path / 'args.json').write_text(json.dumps(args))
    artifacts = {'args.json': sha(path / 'args.json')}
    if role == 'haa_eval':
        room = rooms[0]
        per = per_sample(room, cfg['backbone'], 'best.pth', offset, invalid)
        per['meta'].update(frame=cfg['frame'], heading=heading, room=room)
        per['side_label'] = [1 if i % 2 else -1 for i in per['index']]
        (path / 'per_sample_{}.json'.format(room)).write_text(json.dumps(per))
        (path / 'metrics_{}.json'.format(room)).write_text(
            json.dumps(metrics_file(room, cfg['backbone'], 'best.pth', per)))
        for item in ('per_sample_{}.json'.format(room), 'metrics_{}.json'.format(room)):
            artifacts[item] = sha(path / item)
        extra = {'room': room, 'checkpoint_sha256': 'd' * 64, 'samples': len(per['index']),
                 'seed': seed}
    else:
        (path / 'history.jsonl').write_text('{"epoch": 1}\n')
        artifacts['history.jsonl'] = sha(path / 'history.jsonl')
        extra = {'rooms': rooms, 'init_sha256': 'e' * 64, 'best_epoch': 2, 'seed': seed}
    record = dict({'schema_version': 1, 'run_type': role, 'run_dir': str(path.resolve()),
                   'child_exit': 0, 'child_exit_time': STAMP, 'diagnostic': False,
                   'log': {'path': str(path / 'child.log'), 'sha256': 'a' * 64},
                   'child_exit_receipt': {'path': str(path / 'child_exit.json'),
                                          'sha256': 'a' * 64, 'child_pid': 999999999},
                   'admissible_arm': True, 'artifacts': artifacts,
                   'backbone': cfg['backbone'], 'frame': cfg['frame'], 'heading': heading,
                   'source_closure_sha256': closure}, **extra)
    record.update(overrides or {})
    subject.finalizer.write_completion(path / 'completion.json', record)
    return record


def new_job(base, job, arm, offset, closure=CLOSURE, invalid=(), job_overrides=None, **kwargs):
    cfg = subject.ARMS[arm]
    expect = subject.EXPECT_OF[job]
    job_dir = Path(base) / job
    job_dir.mkdir(parents=True, exist_ok=True)
    seed = int(job[len('seed'):]) if job in subject.SEEDS else 0
    children = {}
    for name in subject.finalizer.expected_children(expect):
        new_child(job_dir, name, arm, seed, offset, closure, invalid, **kwargs)
        children[name] = sha(job_dir / name / 'completion.json')
    record = {'schema_version': 1, 'run_type': 'haa_job', 'run_dir': str(job_dir.resolve()),
              'expect': expect, 'children': children, 'job_spec_sha256': 'f' * 64,
              'owner': {'pid': 4242, 'path': str(job_dir / 'launch.pid'), 'sha256': 'a' * 64},
              'backbone': cfg['backbone'], 'frame': cfg['frame'],
              'heading': HEADING if cfg['frame'] == 'heading' else None, 'seed': seed,
              'init_sha256': 'e' * 64, 'diagnostic': False, 'admissible_arm': True}
    record.update(job_overrides or {})
    subject.provenance.write_completion(job_dir / 'completion.json', record)
    return record


NEW_OFFSETS = {'cyl_or': 0.02, 'control_hf': 0.04, 'cyl_hf': 0.06}


def build_new_root(root, offsets=None, arms=None):
    offsets = NEW_OFFSETS if offsets is None else offsets
    root = Path(root)
    for arm in (arms or subject.NEW_ARMS):
        base = root / arm
        for job in subject.JOBS:
            number = int(job[len('seed'):]) if job in subject.SEEDS else 5
            new_job(base, job, arm, offsets[arm] + 0.01 * number, closure=CLOSURE + '')
    return root


@pytest.fixture
def new_root(tmp_path):
    return build_new_root(tmp_path / 'exp06_sim2real')


def test_the_legacy_branch_admits_the_complete_historical_root(legacy_root, tmp_path):
    out = tmp_path / 'receipt.json'
    _, digest = subject.write_legacy_receipt(out, legacy_root, strict=False)
    data, receipt = subject.load_legacy(legacy_root, out, {'path': str(out), 'sha256': digest})
    assert sorted(data) == ['control', 'cyl'] and receipt['label'] == 'reconstructed'
    assert sorted(data['cyl']['per']) == ['seed0', 'seed1', 'seed2', 'zeroshot']
    assert sorted(data['control']['per']['seed1']) == sorted(ROOMS)
    assert data['control']['branch'] == 'legacy'


def test_an_incomplete_historical_root_is_refused(legacy_root, tmp_path):
    (Path(legacy_root) / 'released/seed2/eval/per_sample_hallway.json').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        subject.load_legacy(legacy_root)


def test_a_new_arm_is_admitted_with_one_closure_and_every_room(new_root):
    arm = subject.load_new_arm(new_root, 'cyl_or')
    assert arm['closure'] == CLOSURE and arm['branch'] == 'new'
    assert sorted(arm['per']) == ['seed0', 'seed1', 'seed2', 'zeroshot']
    assert sorted(arm['per']['zeroshot']) == sorted(ROOMS)
    assert arm['per']['seed0']['hallway']['meta']['heading']['hallway']['k'] == 128
    assert len(arm['jobs']['seed0']['children']) == 9


NEW_REFUSALS = {
    'missing_seed': lambda root: (root / 'cyl_or/seed2').rename(root / 'cyl_or/_seed2'),
    'missing_completion': lambda root: (root / 'cyl_or/seed0/completion.json').unlink(),
    'missing_child': lambda root: (root / 'cyl_or/seed1/eval/hallway/completion.json').unlink(),
    'job_receipt': lambda root: (root / 'cyl_or/seed0/child_exit.json').write_text('{}'),
    'changed_child': lambda root: (
        root / 'cyl_or/seed0/eval/hallway/per_sample_hallway.json').write_text('{}'),
}


@pytest.mark.parametrize('case', sorted(NEW_REFUSALS))
def test_a_new_arm_missing_or_changed_evidence_is_refused(new_root, case):
    NEW_REFUSALS[case](Path(new_root))
    with pytest.raises(ValueError):
        subject.load_new_arm(new_root, 'cyl_or')


def test_a_wrong_backbone_frame_roll_or_closure_is_refused(tmp_path):
    root = build_new_root(tmp_path / 'a', arms=('cyl_or',))
    child = Path(root) / 'cyl_or/seed0/eval/hallway'
    (child / 'completion.json').unlink()
    new_child(Path(root) / 'cyl_or/seed0', 'eval/hallway', 'cyl_or', 0, 0.02,
              overrides={'backbone': 'simple'})
    with pytest.raises(ValueError, match='backbone'):
        subject.load_new_arm(root, 'cyl_or')
    for case, kwargs in (('roll', {'heading': {r: {'k': 0} for r in ROOMS}}),
                         ('closure', {'closure': 'a' * 64})):
        fresh = build_new_root(tmp_path / case, arms=('cyl_or',))
        target = Path(fresh) / 'cyl_or/seed1/eval/complex_room'
        (target / 'completion.json').unlink()
        new_child(Path(fresh) / 'cyl_or/seed1', 'eval/complex_room', 'cyl_or', 1, 0.03,
                  **kwargs)
        with pytest.raises(ValueError):
            subject.load_new_arm(fresh, 'cyl_or')


# --- pairing, the cohort policy and the verdicts -----------------------------------------

REAL_LEGACY = Path(__file__).resolve().parents[1] / 'ckpt/sim2real'
CANONICAL_STATS = REAL_LEGACY / 'stats.json'


@pytest.fixture
def arms(legacy_root, new_root):
    data, _ = subject.load_legacy(legacy_root)
    for arm in subject.NEW_ARMS:
        data[arm] = subject.load_new_arm(new_root, arm)
    return data


def test_the_pairing_assertions_are_exp02s(arms):
    a = arms['cyl_or']['per']['seed0']['hallway']
    b = arms['control']['per']['seed0']['hallway']
    subject.assert_pairing(a, b, 'ok')
    for change in (lambda p: p.update(index=list(reversed(p['index']))),
                   lambda p: p.update(ir_path=['x'] * len(p['index'])),
                   lambda p: p['meta'].update(num_shot=1),
                   lambda p: p['meta'].update(eval_seed=1)):
        broken = json.loads(json.dumps(a))
        change(broken)
        with pytest.raises(ValueError, match='pair '):
            subject.assert_pairing(broken, b, 'broken')


def test_the_cohort_is_the_queries_finite_in_every_compared_run(tmp_path, monkeypatch):
    build_cache(tmp_path / 'HAA_xrir')
    monkeypatch.setattr(legacy, 'HAA_ROOT', str(tmp_path / 'HAA_xrir'))
    build_legacy_root(tmp_path / 'sim2real')
    root = tmp_path / 'new'
    for arm in subject.NEW_ARMS:
        for job in subject.JOBS:
            invalid = (0, 1) if (arm == 'cyl_or' and job == 'seed1') else ()
            new_job(root / arm, job, arm, 0.02, invalid=invalid)
    data, _ = subject.load_legacy(tmp_path / 'sim2real')
    for arm in subject.NEW_ARMS:
        data[arm] = subject.load_new_arm(root, arm)
    rows = subject.cell_rows(data, 'cyl_or', 'control', 'hallway', 'c50')
    assert rows['n_test'] == SIZE['hallway'] and rows['cohort'] == SIZE['hallway'] - 2
    assert rows['excluded']['cyl_or']['queries'] == 2
    assert rows['excluded']['cyl_or']['seeds'] == {'seed0': 0, 'seed1': 2, 'seed2': 0}
    assert rows['excluded']['control']['queries'] == 0
    assert len(rows['a']) == len(rows['b']) == 3 * rows['cohort']
    assert sorted(set(rows['seeds'])) == list(subject.SEEDS)
    reasons = subject.void_reasons(rows, 'cyl_or', 'control')
    assert len(reasons) == 2 and 'more invalid queries' in reasons[0]
    assert subject.void_reasons(subject.cell_rows(data, 'control_hf', 'control', 'hallway',
                                                 'c50'), 'control_hf', 'control') == []


def test_the_decision_cell_reports_pass_fail_void_or_not_converged(arms):
    cell = subject.decision_cell(arms, subject.H1, 'hallway', 'c50', subject.H1_MARGIN_DB,
                                 n_boot=200)
    assert cell['contrast'] == 'cyl_or - control' and cell['verdict'] in ('pass', 'fail')
    assert cell['convergence']['status'] == 'converged'
    assert cell['cohort'] == SIZE['hallway'] and cell['void_reasons'] == []
    assert sorted(cell['per_seed_diff']) == list(subject.SEEDS)
    assert cell['verdict'] == ('pass' if cell['convergence']['interval'][1]
                               < subject.H1_MARGIN_DB else 'fail')
    assert subject.verdict_of(cell['convergence'], ['void'], 0.23) == 'void'
    assert subject.verdict_of({'status': 'not_converged', 'interval': None}, [], 0.23) == \
        'not_converged'
    assert subject.verdict_of({'status': 'converged', 'interval': (-1.0, 0.229)}, [], 0.23) \
        == 'pass'
    assert subject.verdict_of({'status': 'converged', 'interval': (-1.0, 0.23)}, [], 0.23) \
        == 'fail'


def test_h1b_uses_a_zero_margin_against_the_channel_control(arms):
    cell = subject.decision_cell(arms, subject.H1B, 'hallway', 'c50', 0.0, n_boot=200)
    assert cell['contrast'] == 'cyl_or - cyl_hf' and cell['margin'] == 0.0


def test_the_screen_labels_every_cell_and_adjusts_for_eleven(arms):
    cells = subject.screen_cells(arms, subject.H1, n_boot=200, adjusted_n_boot=200)
    assert len(cells) == 11 and {c['family'] for c in cells} == {11}
    assert {round(c['adjusted_alpha'], 8) for c in cells} == {round(0.05 / 11, 8)}
    assert {c['label'] for c in cells} <= {'detected harm', 'detected improvement',
                                           'no detected difference', 'not converged'}
    assert subject.h2_label((0.1, 0.4)) == 'detected harm'
    assert subject.h2_label((-0.4, -0.1)) == 'detected improvement'
    assert subject.h2_label((-0.1, 0.4)) == 'no detected difference'
    assert subject.h2_label((0.0, 0.4)) == 'no detected difference'


def test_the_descriptive_contrasts_are_the_three_of_section_7(arms):
    cells = subject.descriptive_cells(arms, n_boot=200)
    assert {c['contrast'] for c in cells} == {'cyl_or - control_hf', 'control_hf - control',
                                              'cyl_hf - cyl'}
    assert len(cells) == 3 * 11 and all('adjusted_two_way' not in c for c in cells)


@pytest.mark.skipif(not CANONICAL_STATS.is_file(), reason='needs the exp_02 results')
@pytest.mark.parametrize('room,key', [(c['room'], c['metric']) for c in
                                      json.loads(CANONICAL_STATS.read_text())['paired']]
                         if CANONICAL_STATS.is_file() else [])
def test_the_legacy_rows_reproduce_the_canonical_exp02_cells(room, key, monkeypatch):
    # exp_02 compared stage-2 `init` strings against `<root>/stage1/best.pth`, so the
    # historical root is given exactly as that run gave it: repo-relative.
    monkeypatch.chdir(REAL_LEGACY.parents[1])
    data, _ = subject.load_legacy('ckpt/sim2real')
    rows = subject.cell_rows(data, 'cyl', 'control', room, key)
    got = subject.intervals(rows, subject.ALPHA, subject.N_BOOT)
    want = next(c for c in json.loads(CANONICAL_STATS.read_text())['paired']
                if (c['room'], c['metric']) == (room, key))
    assert rows['cohort'] == want['n_queries'] and len(rows['a']) == want['n_valid']
    assert got['diff'] == want['diff']
    assert (got['query']['lo'], got['query']['hi']) == (want['lo'], want['hi'])
    assert (got['two_way']['lo'], got['two_way']['hi']) == (want['lo_two_way'],
                                                            want['hi_two_way'])


# --- the side split, the approvals gate and the published outputs -------------------------


@pytest.fixture
def cache_root(tmp_path_factory):
    return build_cache(tmp_path_factory.mktemp('cache'))


def test_the_two_side_label_routes_agree_and_a_mismatch_is_refused(arms, tmp_path):
    room = 'hallway'
    from_cache = subject.cache_side_labels(room)
    assert sorted(from_cache) == indices(room)
    assert set(from_cache.values()) == {-1, 1}
    new = arms['cyl_or']['per']['zeroshot'][room]
    assert subject.side_labels(new, room) == new['side_label']
    old = arms['control']['per']['zeroshot'][room]
    assert 'side_label' not in old
    assert subject.side_labels(old, room) == [from_cache[i] for i in old['index']]
    broken = json.loads(json.dumps(new))
    broken['side_label'] = [-s for s in broken['side_label']]
    with pytest.raises(ValueError, match='not the room-frame signs'):
        subject.side_labels(broken, room)


def test_the_side_split_covers_every_arm_room_and_metric(arms):
    table = subject.side_split(arms)
    assert table['job'] == 'zeroshot'
    assert len(table['cells']) == len(arms) * 11
    entry = table['cells']['cyl_or|hallway|c50']
    assert entry['n_minus_y'] + entry['n_plus_y'] == SIZE['hallway']
    assert entry['minus_y'] is not None and entry['plus_y'] is not None


def test_production_refuses_without_the_approvals_module():
    with pytest.raises(subject.approvals_api.ApprovalsUnavailable):
        subject.approvals(False)
    approved, receipt, deviations = subject.approvals(True)
    assert approved is None and receipt is None
    assert deviations and 'not available on this branch' in deviations[0]


def test_the_analysis_binds_its_inputs_and_suppresses_draft_verdicts(arms, tmp_path):
    result = subject.analyse(arms, n_boot=200, adjusted_n_boot=200, exploratory=True,
                             deviations=['no approvals'])
    assert result['H1']['verdict'] == 'suppressed (draft)' == result['H1b']['verdict']
    assert result['margin_db'] == 0.23 and result['n_boot'] == 200
    assert len(result['H2']) == 11 and len(result['D']) == 33
    assert result['bootstrap_seeds'] == [0, 1] and result['heading_k'] == 128
    assert all(path.endswith('completion.json') for path in result['inputs'])
    assert len(result['inputs']) == 3 * 4 * (1 + 9) - 3 * 5  # three arms, four jobs
    assert result['rows']['control|fine-tuned|hallway|c50']['std'] is not None
    assert result['arms']['cyl_or']['closure'] == CLOSURE


def test_the_outputs_are_written_once_and_the_json_binds_the_summary(arms, tmp_path):
    result = subject.analyse(arms, n_boot=200, adjusted_n_boot=200, exploratory=True)
    out, summary = tmp_path / 'stats.json', tmp_path / 'summary.txt'
    record, digest, text = subject.write_outputs(result, out, summary)
    assert summary.read_text() == text and digest == sha(out)
    assert record['summary_sha256'] == subject.hashlib.sha256(text.encode()).hexdigest()
    assert json.loads(out.read_text())['summary_sha256'] == record['summary_sha256']
    assert 'DRAFT' in text and 'H2 screen' in text and 'Room-frame side split' in text
    with pytest.raises(FileExistsError):
        subject.write_outputs(result, out, summary)


def test_the_cli_writes_a_draft_and_the_legacy_receipt(legacy_root, new_root, tmp_path,
                                                       monkeypatch):
    receipt = tmp_path / 'legacy_receipt.json'
    assert subject.main(['--legacy-root', str(legacy_root), '--exploratory',
                         '--write-legacy-receipt', str(receipt)]) == 0
    assert json.loads(receipt.read_text())['label'] == 'reconstructed'
    out, summary = tmp_path / 'stats.json', tmp_path / 'summary.txt'
    assert subject.main(['--legacy-root', str(legacy_root), '--new-root', str(new_root),
                         '--legacy-receipt', str(receipt), '--json', str(out),
                         '--summary', str(summary), '--n-boot', '200',
                         '--n-boot-adjusted', '200', '--exploratory']) == 0
    record = json.loads(out.read_text())
    assert record['exploratory'] is True and record['legacy_receipt']['label'] == \
        'reconstructed'
    assert record['H1']['verdict'] == 'suppressed (draft)'
    assert record['legacy_receipt']['sha256'] == sha(receipt)


def test_the_cli_refuses_a_production_run_on_this_branch(legacy_root, new_root, tmp_path):
    receipt = tmp_path / 'r.json'
    subject.write_legacy_receipt(receipt, legacy_root, strict=False)
    with pytest.raises(subject.approvals_api.ApprovalsUnavailable):
        subject.main(['--legacy-root', str(legacy_root), '--new-root', str(new_root),
                      '--legacy-receipt', str(receipt), '--json', str(tmp_path / 'j.json'),
                      '--summary', str(tmp_path / 's.txt')])
