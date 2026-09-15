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


def test_a_receipt_that_omits_an_arm_never_admits_it(legacy_root, tmp_path):
    """Finding 6: load_legacy returns both arms, so a receipt must enumerate both.

    A receipt whose own ``arms`` list names only ``control`` used to define what the
    verifier expected, so every cylindrical artifact was unenumerated -- and free to
    change -- while the loader still returned the cylindrical arm.
    """
    record = subject.legacy_receipt(legacy_root, strict=False)
    kept = [item for item in record['files'] if not item['path'].startswith('cyl/')]
    partial = dict(record, arms=['control'], files=kept,
                   files_sha256=subject._digest(kept))
    out = tmp_path / 'partial.json'
    out.write_text(json.dumps(partial))
    target = Path(legacy_root) / 'cyl/seed0/eval/metrics_hallway.json'
    target.write_text(json.dumps({'tampered': True}))
    approved = {'path': str(out), 'sha256': sha(out)}
    with pytest.raises(ValueError, match='not the registered'):
        subject.verify_legacy_receipt(out, legacy_root, approved)


@pytest.mark.parametrize('arm', ['control', 'cyl'])
def test_a_changed_artifact_of_either_arm_is_refused(legacy_root, tmp_path, arm):
    out = tmp_path / (arm + '.json')
    _, digest = subject.write_legacy_receipt(out, legacy_root, strict=False)
    target = Path(legacy_root) / arm / 'zeroshot' / 'metrics_hallway.json'
    target.write_text(target.read_text() + ' ')
    with pytest.raises(ValueError, match='changed since the receipt'):
        subject.verify_legacy_receipt(out, legacy_root, {'path': str(out), 'sha256': digest})


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / 'ckpt/sim2real').is_dir(),
                    reason='needs the exp_02 results')
def test_the_real_receipt_enumerates_both_registered_arms():
    paths = [item['path'] for item in subject.legacy_receipt_files(
        Path(__file__).resolve().parents[1] / 'ckpt/sim2real')]
    assert {path.split('/')[0] for path in paths} == {'control', 'cyl', 'stats.json',
                                                      'summary.txt'}
    assert paths[-2:] == ['stats.json', 'summary.txt']


def test_a_receipt_of_an_incomplete_arm_is_never_written(legacy_root, tmp_path):
    (Path(legacy_root) / 'cyl/seed2/eval/metrics_hallway.json').unlink()
    with pytest.raises(ValueError, match='cyl has no cyl/seed2/eval/metrics_hallway'):
        subject.write_legacy_receipt(tmp_path / 'r.json', legacy_root, strict=False)


# --- exp_06's own arms ------------------------------------------------------------------

HEADING = {room: {'k': subject.HEADING_K, 'phi_deg': -90.0, 'decision': 'estimated',
                  'path': '/heading/{}.json'.format(room), 'sha256': 'b' * 64}
           for room in ROOMS}


def sha(path):
    return subject.provenance.sha256_file(path)


import os                                              # noqa: E402  (fixture imports)

from test_exp06_haa import cache                        # noqa: F401,E402  (session fixture)
from test_exp06_haa_pipeline import clone, finetune_seed  # noqa: F401,E402


@pytest.fixture(scope='module')
def real_job(finetune_seed, clone):
    """One seed of nine children, certified exactly as tools/exp06_haa_pipeline.sh does."""
    root, children, spec, joblog = finetune_seed
    if not (Path(root) / 'completion.json').is_file():
        subject.finalizer.finalize(root, 'haa_job', joblog, 0, repo=clone, children=children,
                                   expect='finetune', job_spec=spec, owner_pid=os.getpid())
    return root, clone


def test_the_job_completion_schema_is_the_merged_finalizers(real_job):
    """Amendment A3, reconciled: the finalizer's field names are the authoritative ones."""
    root, _ = real_job
    record = json.loads((Path(root) / 'completion.json').read_text())
    assert set(subject.JOB_FIELDS) <= set(record)
    assert subject.FORBIDDEN_JOB_FIELDS == ('child_exit_time', 'child_exit_receipt')
    assert not set(subject.FORBIDDEN_JOB_FIELDS) & set(record)
    assert type(record['owner_pid']) is int and record['owner_pid'] > 0
    assert set(record['job_spec']) == {'path', 'sha256'}
    assert record['run_type'] == 'haa_job' and record['expect'] == 'finetune'
    assert set(record['children']) == set(subject.finalizer.expected_children('finetune'))


def test_a_real_job_is_verified_through_the_finalizers_own_validators(real_job, cache,
                                                                     monkeypatch):
    """Findings 1-2: every child re-verified, two real closures, one heading per room."""
    root, repo = real_job
    monkeypatch.setattr(legacy, 'HAA_ROOT', cache['root'])
    job = subject.verify_job(root, 'seed0', 'cyl_or', repo=repo)
    assert set(job['children']) == set(subject.finalizer.expected_children('finetune'))
    assert sorted(job['closure']) == ['haa_eval', 'haa_train']
    assert job['closure']['haa_train'] != job['closure']['haa_eval']
    assert all(len(value) == 64 for value in job['closure'].values())
    assert set(job['heading']) == set(ROOMS) and len(set(job['heading'].values())) == 4
    assert sorted(job['per']) == sorted(ROOMS)
    assert job['record']['init_sha256'] == job['spec']['init_sha256']
    assert job['record']['heading'] == {room: subject.HEADING_K for room in ROOMS}


@pytest.mark.parametrize('case', ['spec_bytes', 'child_bytes'])
def test_a_job_whose_bound_evidence_changed_is_refused(real_job, cache, monkeypatch, case):
    root, repo = real_job
    monkeypatch.setattr(legacy, 'HAA_ROOT', cache['root'])
    record = json.loads((Path(root) / 'completion.json').read_text())
    target = Path(record['job_spec']['path']) if case == 'spec_bytes' else \
        Path(root) / 'stage1' / 'completion.json'
    original = target.read_bytes()
    target.write_bytes(original + b' ')
    try:
        with pytest.raises(ValueError, match='is not the (bytes|completion) '):
            subject.verify_job(root, 'seed0', 'cyl_or', repo=repo)
    finally:
        target.write_bytes(original)


NEW_OFFSETS = {'cyl_or': 0.02, 'control_hf': 0.04, 'cyl_hf': 0.06}


@pytest.fixture
def stub_new_arms(monkeypatch):
    """The CLI's own tests: admission has its own, over children the finalizer wrote."""
    monkeypatch.setattr(subject, 'load_new_arm',
                        lambda root, arm, init_sha256=None, repo=subject.REPO,
                        approved=None: synthetic_arm(arm, NEW_OFFSETS[arm]))


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
    with pytest.raises(ValueError, match='root is incomplete'):
        subject.load_legacy(legacy_root)


# --- the arm's execution identities and the frozen protocol -------------------------------

TRAIN_CLOSURE, EVAL_CLOSURE = 'a' * 64, 'b' * 64


def arm_children(train=TRAIN_CLOSURE, evaluate=EVAL_CLOSURE, heading=None):
    heading = HEADING if heading is None else heading
    children = {}
    for name in subject.finalizer.expected_children('finetune'):
        role = subject.finalizer.child_role(name)
        children[name] = {'role': role, 'heading': heading,
                          'source_closure_sha256': train if role == 'haa_train' else evaluate}
    return children


def test_training_and_evaluation_children_carry_their_own_closures():
    """Finding 1: the two entry points have two closures; one of each is admissible."""
    children = arm_children()
    assert subject.arm_closures(children) == {'haa_train': TRAIN_CLOSURE,
                                              'haa_eval': EVAL_CLOSURE}
    mixed = dict(children)
    mixed['stage2_hallway'] = dict(mixed['stage2_hallway'], source_closure_sha256='c' * 64)
    with pytest.raises(ValueError, match='haa_train children .* do not share'):
        subject.arm_closures(mixed)
    assert subject.ROLE_CODE_KEY == {'haa_train': 'haa_finetune', 'haa_eval': 'haa_eval'}


def test_the_role_closures_and_heading_records_must_be_the_approved_ones():
    children = arm_children()
    closures, headings = subject.arm_closures(children), subject.arm_headings(children)
    assert headings == {room: HEADING[room]['sha256'] for room in ROOMS}
    approved = {'code': {'haa_finetune': TRAIN_CLOSURE, 'haa_eval': EVAL_CLOSURE},
                'artifacts': {'heading': dict(headings)}}
    subject.check_arm_identities('cyl_or', closures, headings, approved)
    subject.check_arm_identities('cyl_or', closures, headings, None)
    with pytest.raises(ValueError, match=r'not the approved code\.haa_eval'):
        subject.check_arm_identities('cyl_or', closures, headings, {
            'code': dict(approved['code'], haa_eval='d' * 64),
            'artifacts': approved['artifacts']})
    with pytest.raises(ValueError, match=r'not the approved artifacts\.heading'):
        subject.check_arm_identities('cyl_or', closures, headings, {
            'code': approved['code'],
            'artifacts': {'heading': dict(headings, hallway='d' * 64)}})


def test_one_heading_record_per_room_across_the_whole_arm():
    children = arm_children()
    children['eval/hallway'] = dict(
        children['eval/hallway'],
        heading={room: dict(HEADING[room], sha256='d' * 64) for room in ROOMS})
    with pytest.raises(ValueError, match='two heading records for'):
        subject.arm_headings(children)


def test_the_frozen_evaluation_protocol_is_the_one_exp02_registered():
    assert subject.PROTOCOL == {'num_shot': 8, 'eval_seed': 0, 'split': 'test'}
    args = {'num_shot': 8, 'eval_seed': 0, 'split': 'test'}
    subject.child_protocol(args, 'eval/hallway', 'haa_eval')
    subject.child_protocol({'num_shot': 8, 'eval_seed': 0}, 'stage1', 'haa_train')
    for field, value in (('num_shot', 1), ('eval_seed', 3), ('split', 'val')):
        with pytest.raises(ValueError, match='not the registered'):
            subject.child_protocol(dict(args, **{field: value}), 'eval/hallway', 'haa_eval')


def test_an_evaluation_child_covers_exactly_the_diffrir_test_indices(legacy_root):
    room = 'hallway'
    subject.check_test_indices('eval/' + room, room, indices(room))
    for broken in (indices(room)[:-1], list(reversed(indices(room)))):
        with pytest.raises(ValueError, match='DiffRIR test indices'):
            subject.check_test_indices('eval/' + room, room, broken)


# --- pairing, the cohort policy and the verdicts -----------------------------------------

REAL_LEGACY = Path(__file__).resolve().parents[1] / 'ckpt/sim2real'
CANONICAL_STATS = REAL_LEGACY / 'stats.json'


def synthetic_arm(arm, offset, invalid=(), invalid_job='seed1'):
    """An admitted arm's shape without admission: what the statistics of 7 actually read.

    Admission is exercised against real finalised children (below); the tables are
    exercised against these dicts, so a table test never has to forge provenance.
    """
    cfg = subject.ARMS[arm]
    per = {}
    for job in subject.JOBS:
        number = int(job[len('seed'):]) if job in subject.SEEDS else 5
        rooms = {}
        for room in ROOMS:
            item = per_sample(room, cfg['backbone'], 'best.pth', offset + 0.01 * number,
                              invalid if job == invalid_job else ())
            item['meta'].update(frame=cfg['frame'], room=room,
                                heading=HEADING if cfg['frame'] == 'heading' else None)
            item['side_label'] = [1 if index % 2 else -1 for index in item['index']]
            rooms[room] = item
        per[job] = rooms
    return {'arm': arm, 'branch': 'new', 'per': per, 'jobs': {}, 'inputs': {},
            'root': 'ckpt/exp06/sim2real/' + arm,
            'closure': {'haa_train': TRAIN_CLOSURE, 'haa_eval': EVAL_CLOSURE},
            'heading': {room: HEADING[room]['sha256'] for room in ROOMS}}


@pytest.fixture
def arms(legacy_root):
    data, _ = subject.load_legacy(legacy_root)
    for arm in subject.NEW_ARMS:
        data[arm] = synthetic_arm(arm, NEW_OFFSETS[arm])
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


def test_the_cohort_is_the_queries_finite_in_every_compared_run(legacy_root):
    data, _ = subject.load_legacy(legacy_root)
    for arm in subject.NEW_ARMS:
        data[arm] = synthetic_arm(arm, NEW_OFFSETS[arm],
                                  invalid=(0, 1) if arm == 'cyl_or' else ())
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


def void_hallway_c50(arms, arm='cyl_or'):
    """Every hallway C50 observation of one arm invalid, in every job it was measured."""
    for job in subject.JOBS:
        per = arms[arm]['per'][job]['hallway']
        per['c50'] = [float('nan')] * len(per['index'])
    return arms


def test_an_empty_cohort_is_a_reported_void_cell_not_an_exception(arms):
    """Finding 8: the invalidity policy decides before anything is resampled."""
    void_hallway_c50(arms)
    rows = subject.cell_rows(arms, 'cyl_or', 'control', 'hallway', 'c50')
    assert rows['cohort'] == 0 and rows['n_test'] == SIZE['hallway']
    assert subject.intervals(rows, subject.ALPHA, 200) is None
    reasons = subject.void_reasons(rows, 'cyl_or', 'control')
    assert any('finite in every compared run' in reason for reason in reasons)
    cell = subject.decision_cell(arms, subject.H1, 'hallway', 'c50', subject.H1_MARGIN_DB,
                                 n_boot=200)
    assert cell['verdict'] == 'void' and cell['cohort'] == 0
    assert cell['diff'] is None and cell['two_way'] is None and cell['query'] is None
    assert cell['convergence']['status'] == 'void'
    h1b = subject.decision_cell(arms, subject.H1B, 'hallway', 'c50', 0.0, n_boot=200)
    assert h1b['verdict'] == 'void'
    # The bootstrap helper's own zero-width refusal is untouched.
    with pytest.raises(ValueError, match='zero-width'):
        subject.bootstrap.convergence_endpoints(lambda seed: (0.5, 0.5))


def test_a_void_cell_renders_and_leaves_every_other_table_intact(arms):
    void_hallway_c50(arms)
    result = subject.analyse(arms, n_boot=200, adjusted_n_boot=200, exploratory=True)
    assert result['H1']['verdict'] == 'suppressed (draft)'
    screen = {(cell['room'], cell['metric']): cell for cell in result['H2']}
    assert screen[('hallway', 'c50')]['label'] == 'not available'
    assert screen[('hallway', 'c50')]['nominal_two_way'] is None
    assert screen[('hallway', 'edt')]['nominal_two_way'] is not None
    assert len(result['D']) == 33
    text = subject.render(result)
    assert 'not available' in text


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


def filled_approvals():
    """A complete, well-formed approvals record -- every digest plausible, none correct."""
    api = subject.approvals_api
    return api.validate({
        'schema_version': 1,
        'code': {key: 'a' * 64 for key in api.CODE_KEYS},
        'reused': dict({key: 'b' * 64 for key in api.REUSED_DIGESTS},
                       legacy_receipt={'path': 'r.json', 'sha256': 'c' * 64}),
        'artifacts': {'epoch_012': {'path': 'e.pth', 'epoch': 12, 'sha256': 'd' * 64},
                      'heading': {room: 'e' * 64 for room in api.ROOMS},
                      'gate_g1_sha256': 'f' * 64}})


def test_the_producer_closure_must_be_the_approved_one():
    """Finding 3: a populated digest is not an approval of what actually ran."""
    identity = subject.source_identity(strict=False)
    approved = filled_approvals()
    approved['code']['summarize_haa'] = identity['sha256']
    assert subject.check_producer_identity(approved, 'summarize_haa', identity) == []
    assert subject.check_producer_identity(approved, 'legacy_receipt', identity) == []
    assert subject.check_producer_identity(None, 'summarize_haa', identity) == []
    wrong = filled_approvals()
    with pytest.raises(ValueError, match=r'not the approved code\.summarize_haa'):
        subject.check_producer_identity(wrong, 'summarize_haa', identity)
    drafted = subject.check_producer_identity(wrong, 'summarize_haa', identity, True)
    assert len(drafted) == 1 and 'not the approved code.summarize_haa' in drafted[0]


def test_the_exp02_record_and_the_g1_artifact_must_be_the_approved_ones(legacy_root,
                                                                       tmp_path):
    out = tmp_path / 'receipt.json'
    record, _ = subject.write_legacy_receipt(out, legacy_root, strict=False)
    canonical = {item['path']: item['sha256'] for item in record['files']}
    gate = tmp_path / 'gate_g1.json'
    gate.write_text('{"decision": "pass"}')
    approved = filled_approvals()
    approved['reused']['exp02_stats_sha256'] = canonical['stats.json']
    approved['reused']['exp02_summary_sha256'] = canonical['summary.txt']
    approved['artifacts']['gate_g1_sha256'] = sha(gate)
    assert subject.check_reused_identities(approved, record, str(gate)) == {
        str(gate.resolve()): sha(gate)}
    assert subject.check_reused_identities(None, record, None) == {}
    with pytest.raises(ValueError, match='requires --gate-g1'):
        subject.check_reused_identities(approved, record, None)
    approved['reused']['exp02_stats_sha256'] = '9' * 64
    with pytest.raises(ValueError, match=r'not the approved reused\.exp02_stats_sha256'):
        subject.check_reused_identities(approved, record, str(gate))


def test_production_refuses_the_null_approvals_template():
    with pytest.raises(ValueError, match='approvals incomplete'):
        subject.approvals(False)
    approved, receipt, deviations = subject.approvals(True)
    assert approved is not None and receipt['sha256']
    assert deviations and deviations[0].startswith('not approved: ')


def test_the_analysis_binds_its_inputs_and_suppresses_draft_verdicts(arms, tmp_path):
    result = subject.analyse(arms, n_boot=200, adjusted_n_boot=200, exploratory=True,
                             deviations=['no approvals'])
    assert result['H1']['verdict'] == 'suppressed (draft)' == result['H1b']['verdict']
    assert result['margin_db'] == 0.23 and result['n_boot'] == 200
    assert len(result['H2']) == 11 and len(result['D']) == 33
    assert result['bootstrap_seeds'] == [0, 1] and result['heading_k'] == 128
    assert result['rows']['control|fine-tuned|hallway|c50']['std'] is not None
    assert result['arms']['cyl_or']['closure'] == {'haa_train': TRAIN_CLOSURE,
                                                   'haa_eval': EVAL_CLOSURE}


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


def test_every_byte_the_summary_rests_on_is_bound_and_revalidated(arms, tmp_path):
    """Finding 10: the side-split cache is an input, and inputs are rechecked at publish."""
    result = subject.analyse(arms, n_boot=200, adjusted_n_boot=200, exploratory=True)
    cache = Path(legacy.HAA_ROOT) / 'hallway'
    for name in ('meta.json', 'xyzs.npy', 'speaker_xyz.npy'):
        assert result['inputs'][str((cache / name).resolve())] == sha(cache / name)
    out, summary = tmp_path / 's.json', tmp_path / 's.txt'
    target = cache / 'xyzs.npy'
    target.write_bytes(target.read_bytes() + b'\x00')
    with pytest.raises(ValueError, match='input changed during analysis'):
        subject.write_outputs(result, out, summary)
    assert not out.exists() and not summary.exists()


def test_a_failed_publication_leaves_neither_output_behind(arms, tmp_path, monkeypatch):
    """Finding 12: the output pair is staged, so a failure publishes nothing."""
    result = subject.analyse(arms, n_boot=200, adjusted_n_boot=200, exploratory=True)
    out, summary = tmp_path / 'pair' / 's.json', tmp_path / 'pair' / 's.txt'
    real, calls = subject.os.link, []

    def failing(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError('no space left on device')
        return real(source, target)

    monkeypatch.setattr(subject.os, 'link', failing)
    with pytest.raises(OSError):
        subject.write_outputs(result, out, summary)
    assert not out.exists() and not summary.exists()
    monkeypatch.undo()
    subject.write_outputs(result, out, summary)
    assert out.is_file() and summary.is_file()
    with pytest.raises(FileExistsError):
        subject.write_outputs(result, out, tmp_path / 'other.txt')


def test_the_legacy_receipt_binds_every_artifact_it_enumerated(legacy_root, tmp_path):
    out = tmp_path / 'r.json'
    _, digest = subject.write_legacy_receipt(out, legacy_root, strict=False)
    record = subject.verify_legacy_receipt(out, legacy_root)
    inputs = record['inputs']
    assert inputs[str(out.resolve())] == digest
    assert inputs[str((Path(legacy_root) / 'stats.json').resolve())] == \
        sha(Path(legacy_root) / 'stats.json')
    assert len(inputs) == len(record['files']) + 1


def test_the_cli_writes_a_draft_and_the_legacy_receipt(legacy_root, stub_new_arms,
                                                       tmp_path, monkeypatch):
    receipt = tmp_path / 'legacy_receipt.json'
    assert subject.main(['--legacy-root', str(legacy_root), '--exploratory',
                         '--write-legacy-receipt', str(receipt)]) == 0
    assert json.loads(receipt.read_text())['label'] == 'reconstructed'
    out, summary = tmp_path / 'stats.json', tmp_path / 'summary.txt'
    assert subject.main(['--legacy-root', str(legacy_root), '--new-root', 'unused',
                         '--legacy-receipt', str(receipt), '--json', str(out),
                         '--summary', str(summary), '--n-boot', '200',
                         '--n-boot-adjusted', '200', '--exploratory']) == 0
    record = json.loads(out.read_text())
    assert record['exploratory'] is True and record['legacy_receipt']['label'] == \
        'reconstructed'
    assert record['H1']['verdict'] == 'suppressed (draft)'
    assert record['legacy_receipt']['sha256'] == sha(receipt)
    assert record['producer']['entry_module'] == 'tools.exp06_summarize_haa'
    assert any('not the approved code.summarize_haa' in item
               for item in record['deviations'])


def test_the_cli_refuses_a_production_run_on_this_branch(legacy_root, stub_new_arms,
                                                        tmp_path):
    receipt = tmp_path / 'r.json'
    subject.write_legacy_receipt(receipt, legacy_root, strict=False)
    with pytest.raises(ValueError, match='approvals incomplete'):
        subject.main(['--legacy-root', str(legacy_root), '--new-root', 'unused',
                      '--legacy-receipt', str(receipt), '--json', str(tmp_path / 'j.json'),
                      '--summary', str(tmp_path / 's.txt')])


def test_the_registered_initialisations_are_exp01s_and_the_approved_epoch():
    assert subject.ARMS['control_hf']['init_sha256'] == \
        subject.EXP01_CONTROL['sha256'] is not None
    assert subject.ARMS['cyl_hf']['init_sha256'] == subject.EXP01_CYL['sha256'] is not None
    assert subject.ARMS['cyl_or']['init_sha256'] is None
    inits = subject.expected_inits(None)
    assert inits['cyl_or'] is None and inits['cyl_hf'] == subject.EXP01_CYL['sha256']
    approved = {'artifacts': {'epoch_012': {'sha256': 'c' * 64}}}
    assert subject.expected_inits(approved)['cyl_or'] == 'c' * 64


# --- finding 7: production approvals are the committed, reviewed bytes --------------------


def test_production_approvals_are_bound_to_the_reviewed_commit(tmp_path):
    """Plan 6.4 approves a commit, not a file: an uncommitted record never admits a run."""
    template = Path(subject.approvals_api.approvals_module().TEMPLATE_PATH)
    outside = tmp_path / 'approved_digests.json'
    outside.write_text(template.read_text())
    with pytest.raises(ValueError, match='outside the repository'):
        subject.approvals(False, str(outside))
    approved, receipt, deviations = subject.approvals(True, str(outside))
    assert approved is not None and deviations and 'committed_at' not in receipt
    with pytest.raises(ValueError, match='approvals incomplete'):
        subject.approvals(False)              # the committed template, still all null


def test_approvals_that_were_not_tracked_at_the_reviewed_commit_are_refused():
    import subprocess
    first = subprocess.check_output(['git', 'rev-list', '--max-parents=0', 'HEAD'],
                                    cwd=str(subject.REPO), text=True).split()[0]
    with pytest.raises(ValueError, match='not tracked at'):
        subject.approvals(False, commit=first)
