"""Completion evidence by run type: what the finalizer accepts and what it refuses."""
import copy
import functools
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tools import exp06_finalize, exp06_recipe, exp06_train, provenance

REPO = Path(__file__).resolve().parents[1]
STAMP = '2026-09-15T04:05:06.070809+00:00'
MARKER = 'EXP06_CHILD_EXIT 0 ' + STAMP


def seal(run, log, status=0, text='', stamp=STAMP):
    """What the launcher leaves behind: the end marker and the child-exit receipt."""
    Path(log).write_text(text + 'EXP06_CHILD_EXIT {} {}\n'.format(status, stamp))
    Path(run).mkdir(parents=True, exist_ok=True)
    (Path(run) / 'child_exit.json').write_text(json.dumps(
        {'child_pid': 424242, 'status': status, 'ended_at': stamp,
         'log_sha256_after_marker': provenance.sha256_file(log)}, sort_keys=True, indent=2) + '\n')
    return log
STATE = {'source_network.weight': torch.arange(6.).reshape(2, 3), 'head.bias': torch.zeros(2)}


@pytest.fixture(scope='session')
def clone(tmp_path_factory):
    """A hardlinked clone: its working tree equals HEAD, so drift can be introduced."""
    root = tmp_path_factory.mktemp('exp06clone') / 'repo'
    subprocess.run(['git', 'clone', '--local', '--quiet', '--single-branch', str(REPO), str(root)],
                   check=True)
    assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True) == ''
    return root


@pytest.fixture
def data_root(tmp_path):
    """A tiny stand-in for the AcousticRooms mirror, small enough to rehash in a test."""
    root = tmp_path / 'data'
    (root / 'single_channel_ir').mkdir(parents=True)
    (root / 'single_channel_ir/a.wav').write_bytes(b'first sample')
    (root / 'single_channel_ir/b.wav').write_bytes(b'second sample')
    return root


def inventory_of(root):
    """The shape tools.provenance.train_data_identity records, over a tiny root."""
    names = sorted(str(path.relative_to(root)) for path in Path(root).rglob('*') if path.is_file())
    records = [{'path': name, 'sha256': provenance.sha256_file(Path(root) / name),
                'size': (Path(root) / name).stat().st_size,
                'mtime_ns': (Path(root) / name).stat().st_mtime_ns} for name in names]
    digest = hashlib.sha256(json.dumps([[r['path'], r['sha256']] for r in records],
                                       sort_keys=True).encode()).hexdigest()
    return {'data_root': str(root), 'inventory': records, 'inventory_files': len(records),
            'inventory_missing': 0, 'inventory_bytes': sum(r['size'] for r in records),
            'inventory_sha256': digest, 'split': 'train', 'cache_key': 'a' * 64}


@functools.lru_cache(maxsize=None)
def _base_record(repo):
    """One real import closure per clone; the subprocess import is far too slow per test."""
    return exp06_train.provenance_fields(['--backbone', 'cylindrical_oriented'], 'full',
                                         repo=Path(repo))


def provenance_record(repo=None):
    return copy.deepcopy(_base_record(str(repo if repo is not None else REPO)))


def full_args():
    return dict(exp06_recipe.EXP01_RECIPE, backbone='cylindrical_oriented',
                save_dir='ckpt/exp06/pretrain/attempt_x', num_workers=12, log_interval=50,
                save_every=500, epoch_ckpt_every=1,
                env={'PYTHONHASHSEED': '0', 'XRIR_DATA_PATH': '/data',
                     'OMP_NUM_THREADS': '8', 'CUDA_VISIBLE_DEVICES': '1'},
                max_train_batches=0, max_test_batches=0, test_subset=0, resume=None,
                no_save=False, yaw_aug=0, yaw_aug_seed=0, yaw_aug_width=512,
                train_batches_per_epoch=exp06_recipe.TRAIN_BATCHES_PER_EPOCH, tier='M',
                param_counts=dict(exp06_recipe.expected_param_counts('cylindrical_oriented')),
                exp06_run_type='full', exp06_registry_sha256='a' * 64,
                exp06_source_closure_sha256='b' * 64, exp06_git_head='c' * 40,
                exp06_provenance_path='provenance.json')


def history_rows(epochs=range(1, 13)):
    return [dict(epoch=epoch, train_loss=1.0 / epoch, test_loss=2.0 / epoch,
                 best_test_loss=2.0 / epoch, is_best=True, lr=1e-3, epoch_minutes=155.8)
            for epoch in epochs]


@pytest.fixture
def full_run(tmp_path, clone, data_root):
    """A complete twelve-epoch attempt directory with tiny tensors and a closed log."""
    run = tmp_path / 'attempt_20260916T130000'
    run.mkdir()
    record = provenance_record(clone)
    record['train_data_identity'] = inventory_of(data_root)
    args = full_args()
    record['effective_args'] = args
    (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    (run / 'args.json').write_text(json.dumps(args, indent=2))
    (run / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history_rows()))
    torch.save({'model': STATE, 'optimizer': {}, 'scheduler': {}, 'epoch': 12, 'batch_idx': 0,
                'best_test_loss': 0.1, 'args': args}, run / 'last.pth')
    torch.save(STATE, run / 'epoch_012.pth')
    log = seal(run, tmp_path / 'oriented_cyl_train_full.log',
               text='train samples: 296334  test samples: 6337\n')
    return run, log


def test_full_completion_records_every_artifact(full_run, clone):
    run, log = full_run
    fields = exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    completion = json.loads((run / 'completion.json').read_text())
    assert completion == fields
    assert completion['run_type'] == 'full' and completion['diagnostic'] is False
    assert completion['admissible_arm'] is True and completion['child_exit'] == 0
    assert completion['child_exit_time'] == '2026-09-15T04:05:06.070809+00:00'
    assert set(completion['artifacts']) == {'provenance.json', 'args.json', 'history.jsonl',
                                            'last.pth', 'epoch_012.pth'}
    assert completion['artifacts']['args.json'] == provenance.sha256_file(run / 'args.json')
    assert completion['log']['sha256'] == provenance.sha256_file(log)
    assert completion['epochs'] == 12 and completion['backbone'] == 'cylindrical_oriented'
    assert completion['source_closure_sha256'] == provenance_record(clone)['source_closures']['training']['sha256']


def test_finalize_is_idempotent_and_refuses_a_different_completion(full_run, clone):
    run, log = full_run
    exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    first = (run / 'completion.json').read_bytes()
    exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert (run / 'completion.json').read_bytes() == first
    (run / 'completion.json').write_text('{"run_type": "full"}\n')
    with pytest.raises(ValueError, match='completion'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)


@pytest.mark.parametrize('damage,cause', [
    ('truncated', 'max_train_batches'),
    ('missing_epoch', 'epoch'),
    ('state_mismatch', 'epoch_012'),
    ('missing_epoch_ckpt', 'epoch_012'),
    ('last_epoch', 'epoch'),
    ('args_differ', 'args'),
    ('closure_drift', 'closure'),
    ('run_type', 'run_type'),
])
def test_full_refusals_are_named_and_write_nothing(full_run, clone, damage, cause):
    run, log = full_run
    if damage == 'truncated':
        args = dict(full_args(), max_train_batches=3)
        (run / 'args.json').write_text(json.dumps(args))
        torch.save({'model': STATE, 'epoch': 12, 'batch_idx': 0, 'args': args}, run / 'last.pth')
    elif damage == 'missing_epoch':
        (run / 'history.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in history_rows(range(1, 12))))
    elif damage == 'state_mismatch':
        torch.save({key: value + 1 for key, value in STATE.items()}, run / 'epoch_012.pth')
    elif damage == 'missing_epoch_ckpt':
        (run / 'epoch_012.pth').unlink()
    elif damage == 'last_epoch':
        torch.save({'model': STATE, 'epoch': 11, 'batch_idx': 0, 'args': full_args()}, run / 'last.pth')
    elif damage == 'args_differ':
        torch.save({'model': STATE, 'epoch': 12, 'batch_idx': 0,
                    'args': dict(full_args(), save_dir='elsewhere')}, run / 'last.pth')
    elif damage == 'closure_drift':
        record = json.loads((run / 'provenance.json').read_text())
        files = [{'path': 'tools/exp06_train.py', 'reviewed_blob_sha256': '0' * 64,
                  'working_tree_sha256': '0' * 64, 'commits_after_reviewed': [], 'mtime': 'x'}]
        record['source_closures']['training'] = {
            'files': files, 'sha256': exp06_finalize.closure_digest(files)}
        (run / 'provenance.json').write_text(json.dumps(record))
    else:
        record = json.loads((run / 'provenance.json').read_text())
        record['run_type'] = 'probe'
        (run / 'provenance.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


@pytest.mark.parametrize('line,exit_code', [
    (MARKER, 1), ('EXP06_CHILD_EXIT 1 2026-09-15T04:05:06+00:00', 0),
    ('no marker at all', 0), ('EXP06_CHILD_EXIT 0', 0),
    ('EXP06_CHILD_EXIT 0 2026-09-15T04:05:06+00:00 extra', 0),
    ('EXP06_CHILD_EXIT 0 2026-09-15T04:05:06', 0), ('', 0)])
def test_open_or_disagreeing_logs_are_refused(full_run, clone, line, exit_code):
    run, log = full_run
    log.write_text('train samples: 1\n' + line + ('\n' if line else ''))
    with pytest.raises(ValueError):
        exp06_finalize.finalize(run, 'full', log, exit_code, repo=clone)
    assert not (run / 'completion.json').exists()


def test_non_zero_child_exit_is_refused_for_an_arm(full_run, clone):
    run, log = full_run
    seal(run, log, status=7)
    with pytest.raises(ValueError, match='status'):
        exp06_finalize.finalize(run, 'full', log, 7, repo=clone)
    assert not (run / 'completion.json').exists()


@pytest.mark.parametrize('run_type', ['smoke', 'probe'])
def test_diagnostic_runs_need_no_artifacts(tmp_path, run_type):
    run = tmp_path / run_type
    run.mkdir()
    log = seal(run, tmp_path / 'smoke.log', status=3, text='EXP06_SMOKE {"wall_s": 12.5}\n')
    receipt = tmp_path / 'probe_20260915T040506.json'
    receipt.write_text(json.dumps({'diagnostic': True, 'entry': 'exp06_train'}))
    fields = exp06_finalize.finalize(run, run_type, log, 3, repo=REPO, receipt=receipt)
    assert fields['diagnostic'] is True and fields['admissible_arm'] is False
    assert fields['artifacts'] == {} and fields['child_exit'] == 3
    assert fields['receipt'] == {'path': str(receipt.resolve()),
                                 'sha256': provenance.sha256_file(receipt)}
    assert json.loads((run / 'completion.json').read_text()) == fields
    assert exp06_finalize.finalize(run, run_type, log, 3, repo=REPO, receipt=receipt) == fields


def test_unknown_run_type_and_missing_directory_are_refused(tmp_path, full_run, clone):
    run, log = full_run
    with pytest.raises(ValueError, match='run type'):
        exp06_finalize.finalize(run, 'invented', log, 0, repo=REPO)
    with pytest.raises(ValueError, match='directory'):
        exp06_finalize.finalize(tmp_path / 'absent', 'full', log, 0, repo=clone)


@pytest.fixture(scope='session')
def haa_repo(tmp_path_factory):
    """Round 2b's entries do not exist yet: a stub repo supplies a resolvable closure."""
    root = tmp_path_factory.mktemp('exp06haa') / 'repo'
    (root / 'tools').mkdir(parents=True)
    (root / 'tools/__init__.py').write_text('')
    for name in ('exp06_haa_finetune', 'exp06_haa_eval'):
        (root / 'tools' / (name + '.py')).write_text('"""round 2b stub"""\nimport json\n')
    for command in (['init', '-q'], ['add', '-A'], ['-c', 'user.email=a@b', '-c', 'user.name=t',
                                                    'commit', '-q', '-m', 'stub entries']):
        subprocess.run(['git'] + command, cwd=root, check=True)
    return root


@pytest.fixture(scope='session')
def heading_jsons(tmp_path_factory):
    """One real, validated heading record per room, written where a binding can point."""
    import numpy as np
    from tools import exp06_heading
    cache = tmp_path_factory.mktemp('heading') / 'synthetic'
    cache.mkdir()
    (cache / 'meta.json').write_text(json.dumps(dict(train=list(range(12)), valid=[12],
                                                     test=[12, 13], sr=22050)))
    theta = np.deg2rad([-90] * 4 + [90] * 8)
    dist = np.arange(1, 13) / 3
    xyz = np.zeros((14, 3))
    xyz[:12, :2] = np.stack((np.cos(theta), np.sin(theta)), axis=1) * dist[:, None]
    xyz[:, 2] = 50
    np.save(cache / 'xyzs.npy', xyz)
    np.save(cache / 'speaker_xyz.npy', np.zeros(3))
    rirs = np.zeros((14, 1200))
    rirs[:12, 10] = 10 ** (np.array([9] * 4 + [0] * 8) / 20) / dist
    np.save(cache / 'rirs.npy', rirs)
    record = exp06_heading.estimate_room_heading(cache)
    out = {}
    for room in ROOMS:
        path = cache.parent / (room + '.json')
        exp06_heading.write_heading_json(path, dict(record, room=room))
        out[room] = {'path': str(path), 'sha256': provenance.sha256_file(path),
                     'phi_deg': record['phi_deg'], 'k': record['k'],
                     'decision': record['decision']}
    return out


@functools.lru_cache(maxsize=None)
def state_keys(backbone='cylindrical_oriented', num_shot=8):
    from model.xRIR_cyl_oriented import build_xrir_exp06
    return tuple(build_xrir_exp06(backbone, num_shot).state_dict().keys())


def tiny_state(**overrides):
    """The real parameter names with one-element tensors: the finalizer checks the key set."""
    state = {key: torch.zeros(1) for key in state_keys()}
    state.update(overrides)
    return state


def haa_provenance(repo, run_type, args, data_root):
    entry = exp06_finalize.ENTRY_MODULES[run_type]
    state = provenance.git_state(repo)
    files, digest = provenance.closure_record(provenance.source_closure(entry, repo),
                                              state['HEAD'], repo)
    return dict(repo=str(repo), reviewed_commit=state['HEAD'], run_type=run_type,
                source_closures={'child': {'entry_module': entry, 'files': files,
                                           'sha256': digest}},
                registry_sha256='a' * 64, git_state=state, environment={'python': '3.8.0'},
                command=['--backbone', args['backbone']], effective_args=args,
                data_identity=inventory_of(data_root))


def haa_train_args(heading_jsons, init_sha256, frame='heading', **overrides):
    rooms = ['class_room', 'hallway', 'complex_room']
    args = dict(backbone='cylindrical_oriented', num_shot=8, rooms=rooms, frame=frame,
                save_dir='stage1', seed=0, epochs=20, val_every=10, lr=1e-4,
                weight_decay=1e-4, eval_seed=0, init='init.pth', init_sha256=init_sha256,
                heading={room: dict(heading_jsons[room]) for room in rooms})
    args.update(overrides)
    return args


def write_haa_train(run, args, log, repo, data_root, rows=None, summary=None, state=None):
    run.mkdir(parents=True, exist_ok=True)
    seal(run, log, text='stage output\n')
    (run / 'args.json').write_text(json.dumps(args))
    record = haa_provenance(repo, 'haa_train', args, data_root)
    (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    rows = [{'epoch': 10, 'val_loss': 0.7}, {'epoch': 20, 'val_loss': 0.5}] if rows is None else rows
    (run / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (run / 'summary.json').write_text(json.dumps(
        {'best_val_loss': 0.5, 'best_epoch': 20} if summary is None else summary))
    for name in ('best.pth', 'last.pth'):
        torch.save(tiny_state() if state is None else state, run / name)
    return run


@pytest.fixture
def haa_train_run(tmp_path, haa_repo, heading_jsons, data_root):
    """One fine-tuning child with every binding the frozen contract requires."""
    init = haa_repo / 'init.pth'
    torch.save(tiny_state(), init)
    args = haa_train_args(heading_jsons, provenance.sha256_file(init))
    run = tmp_path / 'stage1'
    log = tmp_path / 'child.log'
    write_haa_train(run, args, log, haa_repo, data_root)
    return run, log, args


def test_haa_train_completion_binds_heading_and_artifacts(haa_train_run, haa_repo):
    run, log, args = haa_train_run
    fields = exp06_finalize.finalize(run, 'haa_train', log, 0, repo=haa_repo)
    assert set(fields['artifacts']) == {'args.json', 'history.jsonl', 'summary.json',
                                        'best.pth', 'last.pth', 'provenance.json'}
    assert fields['frame'] == 'heading' and fields['backbone'] == 'cylindrical_oriented'
    assert fields['heading']['hallway']['k'] == args['heading']['hallway']['k']
    assert fields['init_sha256'] == args['init_sha256']
    assert fields['diagnostic'] is False and fields['admissible_arm'] is True
    assert json.loads((run / 'completion.json').read_text()) == fields


def test_haa_train_in_the_room_frame_needs_no_heading(tmp_path, haa_repo, heading_jsons, data_root):
    init = haa_repo / 'init.pth'
    torch.save(tiny_state(), init)
    args = haa_train_args(heading_jsons, provenance.sha256_file(init), frame='room')
    args.pop('heading')
    run, log = tmp_path / 'stage1_room', tmp_path / 'child.log'
    write_haa_train(run, args, log, haa_repo, data_root)
    fields = exp06_finalize.finalize(run, 'haa_train', log, 0, repo=haa_repo)
    assert fields['frame'] == 'room' and fields['heading'] is None


@pytest.mark.parametrize('damage,cause', [
    ('frame', 'frame'), ('no_heading', 'heading'), ('partial_heading', 'heading'),
    ('bad_k', 'roll'), ('bad_phi', 'phi_deg'), ('no_decision', 'estimated'),
    ('bad_sha', 'sha256'), ('no_path', 'path'), ('changed_json', 'sha256'),
    ('foreign_room', 'disagrees'), ('no_init', 'init_sha256'), ('init_changed', 'init_sha256'),
    ('summary', 'summary.json'), ('summary_keys', 'summary.json'), ('best', 'best.pth'),
    ('best_keys', 'best.pth'), ('rooms', 'rooms'), ('no_provenance', 'provenance.json'),
    ('wrong_run_type', 'run_type'), ('closure_drift', 'drift'), ('absent_entry', 'closure'),
    ('history_epochs', 'history.jsonl'), ('history_loss', 'history.jsonl'),
    ('args_disagree', 'disagree')])
def test_haa_train_refusals_are_named(tmp_path, haa_repo, heading_jsons, data_root, damage, cause):
    init = haa_repo / 'init.pth'
    torch.save(tiny_state(), init)
    args = haa_train_args(heading_jsons, provenance.sha256_file(init))
    kwargs = {}
    if damage == 'frame':
        args['frame'] = 'world'
    elif damage == 'no_heading':
        args.pop('heading')
    elif damage == 'partial_heading':
        args['heading'].pop('hallway')
    elif damage == 'bad_k':
        args['heading']['hallway']['k'] = (args['heading']['hallway']['k'] + 1) % 512
    elif damage == 'bad_phi':
        args['heading']['hallway']['phi_deg'] = None
    elif damage == 'no_decision':
        args['heading']['hallway'].pop('decision')
    elif damage == 'bad_sha':
        args['heading']['hallway']['sha256'] = 'zz'
    elif damage == 'no_path':
        args['heading']['hallway'].pop('path')
    elif damage == 'foreign_room':
        args['heading']['hallway']['path'] = heading_jsons['class_room']['path']
        args['heading']['hallway']['sha256'] = heading_jsons['class_room']['sha256']
    elif damage == 'no_init':
        args.pop('init_sha256')
    elif damage == 'init_changed':
        args['init_sha256'] = 'd' * 64
    elif damage == 'rooms':
        args['rooms'] = []
    elif damage == 'summary_keys':
        kwargs['summary'] = {'best_val_loss': 0.5}
    elif damage == 'best_keys':
        kwargs['state'] = {'source_network.weight': torch.zeros(1)}
    elif damage == 'history_epochs':
        kwargs['rows'] = [{'epoch': 10, 'val_loss': 0.5}]
    elif damage == 'history_loss':
        kwargs['rows'] = [{'epoch': 10, 'val_loss': 0.7},
                          {'epoch': 20, 'val_loss': float('inf')}]
    run, log = tmp_path / 'stage1', tmp_path / 'child.log'
    write_haa_train(run, args, log, haa_repo, data_root, **kwargs)
    if damage == 'changed_json':
        Path(args['heading']['hallway']['path']).write_text('{"tampered": true}\n')
    elif damage in ('summary', 'best'):
        (run / {'summary': 'summary.json', 'best': 'best.pth'}[damage]).unlink()
    elif damage == 'no_provenance':
        (run / 'provenance.json').unlink()
    elif damage in ('wrong_run_type', 'closure_drift', 'absent_entry', 'args_disagree'):
        record = json.loads((run / 'provenance.json').read_text())
        if damage == 'wrong_run_type':
            record['run_type'] = 'haa_eval'
        elif damage == 'absent_entry':
            record['source_closures']['child']['entry_module'] = 'tools.exp06_haa_absent'
        elif damage == 'args_disagree':
            record['effective_args'] = dict(args, seed=1)
        else:
            stub = haa_repo / 'tools/exp06_haa_finetune.py'
            stub.write_text(stub.read_text() + '# drift\n')
        (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    try:
        with pytest.raises(ValueError, match=cause):
            exp06_finalize.finalize(run, 'haa_train', log, 0, repo=haa_repo)
    finally:
        if damage == 'closure_drift':
            subprocess.run(['git', 'checkout', '--', 'tools/exp06_haa_finetune.py'],
                           cwd=haa_repo, check=True)
        if damage == 'changed_json':
            path = Path(args['heading']['hallway']['path'])
            path.write_text(Path(heading_jsons['class_room']['path']).read_text()
                            .replace('"class_room"', '"hallway"'))
    assert not (run / 'completion.json').exists()


def write_haa_eval(run, args, meta, log=None):
    run.mkdir(parents=True, exist_ok=True)
    if log is not None:
        seal(run, log, text='stage output\n')
    (run / 'args.json').write_text(json.dumps(args))
    room = args['rooms'][0]
    (run / 'metrics_{}.json'.format(room)).write_text(json.dumps({'edt': 0.05, 'c50': 1.1}))
    (run / 'per_sample_{}.json'.format(room)).write_text(json.dumps({'meta': meta, 'edt': [0.05]}))
    return run


@pytest.fixture
def closed_log_file(tmp_path):
    return tmp_path / 'child.log'


def test_haa_eval_completion_records_the_single_room(tmp_path, closed_log_file, heading_jsons):
    args = dict(backbone='cylindrical_oriented', rooms=['hallway'], frame='heading',
                checkpoint='ckpt/exp06/sim2real/cyl_or/seed0/stage2_hallway/best.pth',
                eval_seed=0, split='test',
                heading={'hallway': dict(heading_jsons['hallway'])})
    meta = {'backbone': 'cylindrical_oriented', 'checkpoint_sha256': 'd' * 64,
            'frame': 'heading', 'heading': {'hallway': {'k': 128}}}
    run = write_haa_eval(tmp_path / 'eval' / 'hallway', args, meta, closed_log_file)
    fields = exp06_finalize.finalize(run, 'haa_eval', closed_log_file, 0, repo=REPO)
    assert fields['room'] == 'hallway' and fields['frame'] == 'heading'
    assert set(fields['artifacts']) == {'args.json', 'metrics_hallway.json', 'per_sample_hallway.json'}
    assert fields['checkpoint_sha256'] == 'd' * 64


@pytest.mark.parametrize('damage,cause', [
    ('two_rooms', 'room'), ('no_metrics', 'metrics_hallway.json'),
    ('no_per_sample', 'per_sample_hallway.json'), ('no_meta_heading', 'heading'),
    ('no_meta_backbone', 'meta'), ('meta_backbone_differs', 'backbone')])
def test_haa_eval_refusals_are_named(tmp_path, closed_log_file, heading_jsons, damage, cause):
    args = dict(backbone='cylindrical_oriented', rooms=['hallway'], frame='heading',
                checkpoint='best.pth', eval_seed=0, split='test',
                heading={'hallway': dict(heading_jsons['hallway'])})
    meta = {'backbone': 'cylindrical_oriented', 'checkpoint_sha256': 'd' * 64,
            'frame': 'heading', 'heading': {'hallway': {'k': 128}}}
    if damage == 'two_rooms':
        args['rooms'] = ['hallway', 'class_room']
        args['heading']['class_room'] = dict(heading_jsons['class_room'])
    elif damage == 'no_meta_heading':
        meta.pop('heading')
    elif damage == 'no_meta_backbone':
        meta.pop('backbone')
    elif damage == 'meta_backbone_differs':
        meta['backbone'] = 'cylindrical'
    run = write_haa_eval(tmp_path / 'eval' / 'hallway', args, meta, closed_log_file)
    if damage == 'no_metrics':
        (run / 'metrics_hallway.json').unlink()
    elif damage == 'no_per_sample':
        (run / 'per_sample_hallway.json').unlink()
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(run, 'haa_eval', closed_log_file, 0, repo=REPO)
    assert not (run / 'completion.json').exists()


from sim_to_real.haa_dataset import ROOMS


def write_job(tmp_path, expect='finetune', children=None, log=None):
    job = tmp_path / 'seed0'
    names = children if children is not None else exp06_finalize.expected_children(expect)
    for name in names:
        child = job / name
        child.mkdir(parents=True)
        (child / 'completion.json').write_text(json.dumps(
            {'run_type': 'haa_eval' if name.startswith('eval/') else 'haa_train',
             'admissible_arm': True, 'run_dir': str(child)}, sort_keys=True))
    job.mkdir(parents=True, exist_ok=True)
    if log is not None:
        seal(job, log, text='pipeline output\n')
    return job, [str(job / name) for name in names]


def test_job_completion_requires_every_child(tmp_path, closed_log_file):
    job, children = write_job(tmp_path, log=closed_log_file)
    fields = exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=REPO,
                                     children=children, expect='finetune')
    assert set(fields['children']) == set(exp06_finalize.expected_children('finetune'))
    assert len(fields['children']) == 9 and fields['expect'] == 'finetune'
    assert fields['children']['stage1'] == provenance.sha256_file(job / 'stage1/completion.json')
    assert set(exp06_finalize.expected_children('zeroshot')) == {'eval/' + room for room in ROOMS}


@pytest.mark.parametrize('damage,cause', [
    ('missing', 'missing'), ('extra', 'unexpected'), ('no_completion', 'completion.json'),
    ('zeroshot', 'unexpected'), ('outside', 'outside'), ('inadmissible', 'admissible'),
    ('wrong_type', 'run type')])
def test_job_refusals_are_named(tmp_path, closed_log_file, damage, cause):
    expect = 'zeroshot' if damage == 'zeroshot' else 'finetune'
    names = list(exp06_finalize.expected_children('finetune'))
    if damage == 'missing':
        names.remove('stage2_hallway')
    elif damage == 'extra':
        names.append('stage2_invented')
    job, children = write_job(tmp_path, expect, names, log=closed_log_file)
    if damage == 'no_completion':
        (job / 'stage1/completion.json').unlink()
    elif damage == 'outside':
        children.append(str(tmp_path / 'elsewhere'))
        (tmp_path / 'elsewhere').mkdir()
        (tmp_path / 'elsewhere/completion.json').write_text('{}')
    elif damage == 'inadmissible':
        (job / 'stage1/completion.json').write_text(json.dumps(
            {'run_type': 'haa_train', 'admissible_arm': False}, sort_keys=True))
    elif damage == 'wrong_type':
        (job / 'stage1/completion.json').write_text(json.dumps(
            {'run_type': 'full', 'admissible_arm': True}, sort_keys=True))
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=REPO,
                                children=children, expect=expect)
    assert not (job / 'completion.json').exists()


def test_job_requires_a_declared_expectation(tmp_path, closed_log_file):
    job, children = write_job(tmp_path, log=closed_log_file)
    with pytest.raises(ValueError, match='expect'):
        exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=REPO, children=children)


def test_cli_reports_refusals_with_status_two(full_run, clone):
    run, log = full_run
    command = [sys.executable, 'tools/exp06_finalize.py', '--run-dir', str(run),
               '--run-type', 'full', '--log', str(log), '--child-exit', '0', '--repo', str(clone)]
    env = {**os.environ, 'PYTHONPATH': str(REPO)}
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True, env=env)
    assert completed.returncode == 0, completed.stderr
    assert 'EXP06_FINALIZE_OK' in completed.stdout and (run / 'completion.json').is_file()
    labelled = subprocess.run(command[:2] + ['finalize'] + command[2:], cwd=REPO,
                              capture_output=True, text=True, env=env)
    assert labelled.returncode == 0, labelled.stderr
    (run / 'epoch_012.pth').unlink()
    (run / 'completion.json').unlink()
    refused = subprocess.run(command, cwd=REPO, capture_output=True, text=True, env=env)
    assert refused.returncode == 2 and 'epoch_012' in refused.stderr
    assert not (run / 'completion.json').exists()
    usage = subprocess.run(command[:4], cwd=REPO, capture_output=True, text=True, env=env)
    assert usage.returncode == 2 and 'usage' in usage.stderr


@pytest.mark.parametrize('damage,cause', [
    ('last_not_a_dict', 'last.pth'),
    ('last_without_model', 'last.pth'),
    ('last_model_not_a_state_dict', 'last.pth'),
    ('epoch_not_a_dict', 'epoch_012.pth'),
    ('epoch_unpicklable', 'epoch_012.pth'),
    ('args_not_json', 'args.json'),
    ('args_not_an_object', 'args.json'),
    ('provenance_not_json', 'provenance.json'),
    ('provenance_not_an_object', 'provenance.json'),
    ('history_not_json', 'history.jsonl'),
])
def test_malformed_inputs_raise_named_refusals(full_run, clone, damage, cause):
    """Should-fix 9: malformed containers must refuse by name, never raise KeyError."""
    run, log = full_run
    if damage == 'last_not_a_dict':
        torch.save(torch.zeros(3), run / 'last.pth')
    elif damage == 'last_without_model':
        torch.save({'epoch': 12, 'batch_idx': 0, 'args': full_args()}, run / 'last.pth')
    elif damage == 'last_model_not_a_state_dict':
        torch.save({'model': 'not-a-state-dict', 'epoch': 12, 'batch_idx': 0,
                    'args': full_args()}, run / 'last.pth')
    elif damage == 'epoch_not_a_dict':
        torch.save([1, 2, 3], run / 'epoch_012.pth')
    elif damage == 'epoch_unpicklable':
        (run / 'epoch_012.pth').write_bytes(b'not a torch archive')
    elif damage == 'args_not_json':
        (run / 'args.json').write_text('{ this is not json')
    elif damage == 'args_not_an_object':
        (run / 'args.json').write_text('[1, 2, 3]')
    elif damage == 'provenance_not_json':
        (run / 'provenance.json').write_text('{ nope')
    elif damage == 'provenance_not_an_object':
        (run / 'provenance.json').write_text('"a string"')
    else:
        (run / 'history.jsonl').write_text('{"epoch": 1}\nnot json\n')
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


def test_cli_exits_two_on_a_malformed_checkpoint(full_run, clone):
    """The documented refusal interface: exit 2 and a named cause, nothing written."""
    run, log = full_run
    torch.save(torch.zeros(3), run / 'last.pth')
    command = [sys.executable, 'tools/exp06_finalize.py', '--run-dir', str(run),
               '--run-type', 'full', '--log', str(log), '--child-exit', '0', '--repo', str(clone)]
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                               env={**os.environ, 'PYTHONPATH': str(REPO)})
    assert completed.returncode == 2 and 'last.pth' in completed.stderr
    assert 'EXP06_FINALIZE_REFUSED' in completed.stderr
    assert not (run / 'completion.json').exists()


def rewrite_sources(run, mutate_json=None, mutate_checkpoint=None, mutate_provenance=None):
    """Rewrite the three recorded copies of the arguments independently."""
    args = full_args()
    json_args = mutate_json(dict(args)) if mutate_json else dict(args)
    ckpt_args = mutate_checkpoint(dict(args)) if mutate_checkpoint else dict(json_args)
    prov_args = mutate_provenance(dict(args)) if mutate_provenance else dict(json_args)
    (run / 'args.json').write_text(json.dumps(json_args, indent=2))
    torch.save({'model': STATE, 'optimizer': {}, 'scheduler': {}, 'epoch': 12, 'batch_idx': 0,
                'best_test_loss': 0.1, 'args': ckpt_args}, run / 'last.pth')
    record = json.loads((run / 'provenance.json').read_text())
    record['effective_args'] = prov_args
    (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')


def test_three_argument_sources_must_agree(full_run, clone):
    """Blocker 2: startup, retained and checkpoint arguments are compared type-strictly."""
    run, log = full_run
    rewrite_sources(run)
    assert exp06_finalize.finalize(run, 'full', log, 0, repo=clone)['admissible_arm'] is True


@pytest.mark.parametrize('damage,cause', [
    ('provenance_truncated', 'max_train_batches'),
    ('checkpoint_tf32_int', 'tf32'),
    ('checkpoint_no_save_int', 'no_save'),
    ('float_param_counts', 'param_counts'),
    ('missing_operational', 'num_workers'),
    ('missing_exp06', 'exp06_git_head'),
    ('provenance_without_effective_args', 'effective_args'),
    ('checkpoint_without_args', 'args'),
])
def test_argument_source_refusals_are_named(full_run, clone, damage, cause):
    run, log = full_run
    if damage == 'provenance_truncated':
        rewrite_sources(run, mutate_provenance=lambda a: dict(a, max_train_batches=3))
    elif damage == 'checkpoint_tf32_int':
        rewrite_sources(run, mutate_checkpoint=lambda a: dict(a, tf32=1))
    elif damage == 'checkpoint_no_save_int':
        rewrite_sources(run, mutate_checkpoint=lambda a: dict(a, no_save=0))
    elif damage == 'float_param_counts':
        floats = {key: float(value) for key, value in full_args()['param_counts'].items()}
        rewrite_sources(run, mutate_json=lambda a: dict(a, param_counts=floats))
    elif damage == 'missing_operational':
        drop = lambda a: {k: v for k, v in a.items() if k != 'num_workers'}
        rewrite_sources(run, mutate_json=drop)
    elif damage == 'missing_exp06':
        drop = lambda a: {k: v for k, v in a.items() if k != 'exp06_git_head'}
        rewrite_sources(run, mutate_json=drop)
    elif damage == 'provenance_without_effective_args':
        record = json.loads((run / 'provenance.json').read_text())
        record.pop('effective_args')
        (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    else:
        torch.save({'model': STATE, 'epoch': 12, 'batch_idx': 0}, run / 'last.pth')
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


def test_a_modified_closure_file_is_refused_after_provenance_was_written(full_run, clone):
    """Blocker 1: real working-tree bytes change; the reviewed blob digest does not."""
    run, log = full_run
    assert exp06_finalize.finalize(run, 'full', log, 0, repo=clone)['admissible_arm'] is True
    (run / 'completion.json').unlink()
    victim = clone / 'tools/exp06_recipe.py'
    victim.write_text(victim.read_text() + '\n# drift introduced after the run started\n')
    try:
        with pytest.raises(ValueError, match='exp06_recipe.py'):
            exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    finally:
        subprocess.run(['git', 'checkout', '--', 'tools/exp06_recipe.py'], cwd=clone, check=True)
    assert not (run / 'completion.json').exists()


def test_changed_training_data_bytes_are_refused(full_run, clone, data_root):
    """Blocker 1: the recorded inventory is rehashed, not trusted."""
    run, log = full_run
    (data_root / 'single_channel_ir/b.wav').write_bytes(b'second sample, edited')
    with pytest.raises(ValueError, match='revalidation'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()
    (data_root / 'single_channel_ir/a.wav').unlink()
    with pytest.raises(ValueError, match='revalidation'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)


@pytest.mark.parametrize('key', ['train_data_identity', 'environment', 'registry_sha256',
                                 'command', 'git_state', 'reviewed_commit', 'source_closures'])
def test_incomplete_provenance_is_refused(full_run, clone, key):
    """Blocker 1: completion requires the whole execution record, not a fragment."""
    run, log = full_run
    record = json.loads((run / 'provenance.json').read_text())
    record.pop(key)
    (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    with pytest.raises(ValueError, match=key):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


@pytest.mark.parametrize('damage,cause', [
    ('empty_closure', 'empty'),
    ('membership', 'membership'),
    ('no_entry_module', 'entry module'),
    ('foreign_entry_module', 'entry module'),
    ('no_inventory', 'train_data_identity'),
])
def test_source_closure_and_identity_shapes_are_refused(full_run, clone, damage, cause):
    record = json.loads((full_run[0] / 'provenance.json').read_text())
    closure = record['source_closures']['training']
    if damage == 'empty_closure':
        closure['files'] = []
        closure['sha256'] = exp06_finalize.closure_digest([])
    elif damage == 'membership':
        closure['files'] = closure['files'][:-1]
        closure['sha256'] = exp06_finalize.closure_digest(closure['files'])
    elif damage == 'no_entry_module':
        closure.pop('entry_module')
    elif damage == 'foreign_entry_module':
        closure['entry_module'] = 'tools.exp06_recipe'
    else:
        record['train_data_identity'].pop('inventory')
    run, log = full_run
    (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


def test_a_live_launch_pid_refuses_finalization_in_every_mode(full_run, clone, tmp_path):
    """Blocker 4: recovery must not certify a run whose launcher is still alive."""
    run, log = full_run
    (run / 'launch.pid').write_text('{}\n'.format(os.getpid()))
    with pytest.raises(ValueError, match='alive'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()
    (run / 'launch.pid').write_text('999999999\n')
    assert exp06_finalize.finalize(run, 'full', log, 0, repo=clone)['admissible_arm'] is True
    diagnostic = tmp_path / 'probe'
    seal(diagnostic, tmp_path / 'probe.log', status=0)
    (diagnostic / 'launch.pid').write_text('{}\n'.format(os.getpid()))
    with pytest.raises(ValueError, match='alive'):
        exp06_finalize.finalize(diagnostic, 'probe', tmp_path / 'probe.log', 0, repo=clone)


@pytest.mark.parametrize('damage,cause', [
    ('missing', 'child_exit.json'), ('status', 'child_exit.json'), ('hash', 'child_exit.json'),
    ('not_json', 'child_exit.json'), ('no_pid', 'child_exit.json')])
def test_the_child_exit_receipt_must_bind_the_hashed_log(full_run, clone, damage, cause):
    """Blocker 4: the marker alone never proved the writers had gone."""
    run, log = full_run
    if damage == 'missing':
        (run / 'child_exit.json').unlink()
    elif damage == 'not_json':
        (run / 'child_exit.json').write_text('{ truncated')
    else:
        receipt = json.loads((run / 'child_exit.json').read_text())
        if damage == 'status':
            receipt['status'] = 3
        elif damage == 'hash':
            receipt['log_sha256_after_marker'] = 'f' * 64
        else:
            receipt.pop('child_pid')
        (run / 'child_exit.json').write_text(json.dumps(receipt, sort_keys=True))
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


def test_a_log_that_changes_during_validation_is_refused(full_run, clone, monkeypatch):
    """Blocker 4: a surviving writer must not be able to extend a certified log."""
    run, log = full_run
    original = exp06_finalize.full_evidence

    def late_writer(run_dir, repo):
        with open(str(log), 'a') as stream:
            stream.write('late output from a surviving descendant\n')
        return original(run_dir, repo)

    monkeypatch.setattr(exp06_finalize, 'full_evidence', late_writer)
    with pytest.raises(ValueError, match='stale log'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


def test_child_exit_subcommand_appends_the_marker_and_writes_the_receipt(tmp_path):
    """The launcher's closing step, owned in Python so the receipt binds the same bytes."""
    run = tmp_path / 'attempt'
    run.mkdir()
    log = tmp_path / 'child.log'
    log.write_text('child output\n')
    command = [sys.executable, 'tools/exp06_finalize.py', 'child-exit', '--run-dir', str(run),
               '--log', str(log), '--child-pid', '4242', '--status', '0']
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                               env={**os.environ, 'PYTHONPATH': str(REPO)})
    assert completed.returncode == 0, completed.stderr
    lines = log.read_text().splitlines()
    assert lines[-1].startswith('EXP06_CHILD_EXIT 0 ') and lines[0] == 'child output'
    receipt = json.loads((run / 'child_exit.json').read_text())
    assert receipt['status'] == 0 and receipt['child_pid'] == 4242
    assert receipt['log_sha256_after_marker'] == provenance.sha256_file(log)
    assert receipt['ended_at'] == lines[-1].split()[2]
    again = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                           env={**os.environ, 'PYTHONPATH': str(REPO)})
    assert again.returncode == 2 and 'child_exit.json' in again.stderr
