"""Completion evidence by run type: what the finalizer accepts and what it refuses."""
import copy
import functools
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sim_to_real.haa_dataset import NO_T60_ROOMS, ROOMS
from tools import exp06_finalize, exp06_recipe, exp06_train, provenance

REPO = Path(__file__).resolve().parents[1]
STAMP = '2026-09-15T04:05:06.070809+00:00'
STARTED = '2026-09-15T03:00:00+00:00'
MARKER = 'EXP06_CHILD_EXIT 0 ' + STAMP


def seal(run, log, status=0, text='', stamp=STAMP, **overrides):
    """What the launcher leaves behind: the end marker and the child-exit receipt."""
    Path(log).write_text(text + 'EXP06_CHILD_EXIT {} {}\n'.format(status, stamp))
    Path(run).mkdir(parents=True, exist_ok=True)
    receipt = {'child_pid': 424242, 'status': status, 'started_at': STARTED, 'ended_at': stamp,
               'log_sha256_after_marker': provenance.sha256_file(log)}
    receipt.update(overrides)
    (Path(run) / 'child_exit.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
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


CATEGORIES = ('Apartments', 'Bathrooms', 'Cafe', 'LivingRoomsWithHallway', 'Office',
              'Auditorium', 'Bedrooms', 'ListeningRoom', 'MeetingRoom', 'Restaurants')
TRAIN_IRS = ('single_channel_ir/Apartments/Apartments_idx_1/S000_R000_hybrid_IR.wav',
             'single_channel_ir/Apartments/Apartments_idx_1/S001_R000_hybrid_IR.wav')
TEST_IR = 'single_channel_ir/Bathrooms/Bathrooms_idx_18/S000_R000_hybrid_IR.wav'


@pytest.fixture
def data_root(tmp_path):
    """A tiny AcousticRooms mirror the pinned train_data_identity can really walk.

    Blocker 1: the expected membership is the train split of this root, so the test
    root carries an unseen room as well -- its file must never enter the inventory.
    """
    root = tmp_path / 'data'
    for category in CATEGORIES:
        (root / 'single_channel_ir' / category).mkdir(parents=True)
    for index, name in enumerate(TRAIN_IRS + (TEST_IR,)):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b'sample %d' % index)
    return root


def inventory_of(root):
    """Exactly what the pinned tools.provenance helper records for this root."""
    return provenance.train_data_identity(str(root),
                                          cache_path=str(Path(root).parent / 'inventory.json'))


@functools.lru_cache(maxsize=None)
def _base_record(repo):
    """One real import closure per clone; the subprocess import is far too slow per test."""
    return exp06_train.provenance_fields(['--backbone', 'cylindrical_oriented'], 'full',
                                         repo=Path(repo))


def provenance_record(repo=None):
    return copy.deepcopy(_base_record(str(repo if repo is not None else REPO)))


def full_args(**overrides):
    args = dict(exp06_recipe.EXP01_RECIPE, backbone='cylindrical_oriented',
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
    args.update(overrides)
    return args


def bound_args(run, record, **overrides):
    """The arguments a real run records: every exp06_* field bound to its own record."""
    return full_args(exp06_git_head=record['git_state']['HEAD'],
                     exp06_registry_sha256=exp06_train.registry_sha256(),
                     exp06_source_closure_sha256=record['source_closures']['training']['sha256'],
                     exp06_provenance_path=str(Path(run) / 'provenance.json'), **overrides)


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
    record['data_root'] = str(Path(data_root).resolve())
    record['train_data_identity'] = inventory_of(data_root)
    args = bound_args(run, record)
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


def smoke_receipt(path, **overrides):
    record = dict(schema_version=1, diagnostic=True, entry='exp06_train',
                  module='tools.exp06_train', argv=['--backbone', 'simple', '--no-save'],
                  alarm_seconds=300.0, max_gb=3.0, entry_status=0, aborted_memory=False,
                  exit_status=0, outcome='ok', wall_s=12.5, peak_bytes=0,
                  git_head='c' * 40, timestamp=STAMP)
    record.update(overrides)
    Path(path).write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    return path


@pytest.mark.parametrize('run_type', ['smoke', 'probe'])
def test_diagnostic_runs_need_a_receipt_but_no_artifacts(tmp_path, run_type):
    run = tmp_path / run_type
    log = seal(run, tmp_path / 'smoke.log', text='EXP06_SMOKE {"wall_s": 12.5}\n')
    receipt = smoke_receipt(tmp_path / 'probe_20260915T040506.json')
    fields = exp06_finalize.finalize(run, run_type, log, 0, repo=REPO, receipt=receipt)
    assert fields['diagnostic'] is True and fields['admissible_arm'] is False
    assert fields['passed'] is True and fields['artifacts'] == {} and fields['child_exit'] == 0
    assert fields['receipt'] == {'path': str(Path(receipt).resolve()),
                                 'sha256': provenance.sha256_file(receipt),
                                 'entry': 'exp06_train', 'exit_status': 0, 'outcome': 'ok'}
    assert json.loads((run / 'completion.json').read_text()) == fields
    assert exp06_finalize.finalize(run, run_type, log, 0, repo=REPO, receipt=receipt) == fields


def test_a_failed_diagnostic_is_recorded_and_never_admissible(tmp_path):
    """Should-fix 5: a failed smoke still gets a completion, marked passed: false."""
    run = tmp_path / 'smoke'
    log = seal(run, tmp_path / 'smoke.log', status=3)
    receipt = smoke_receipt(tmp_path / 'r.json', exit_status=3, outcome='memory',
                            aborted_memory=True)
    fields = exp06_finalize.finalize(run, 'smoke', log, 3, repo=REPO, receipt=receipt)
    assert fields['passed'] is False and fields['admissible_arm'] is False
    assert fields['diagnostic'] is True and fields['receipt']['outcome'] == 'memory'


@pytest.mark.parametrize('damage,cause', [
    ('absent', 'receipt'), ('no_receipt_flag', 'receipt'), ('not_json', 'receipt'),
    ('not_diagnostic', 'diagnostic'), ('no_argv', 'no-save'), ('saving_argv', 'no-save'),
    ('no_status', 'exit_status'), ('float_status', 'exit_status')])
def test_invalid_diagnostic_receipts_are_refused(tmp_path, damage, cause):
    run = tmp_path / 'smoke'
    log = seal(run, tmp_path / 'smoke.log')
    path = tmp_path / 'r.json'
    receipt = path
    if damage == 'absent':
        receipt = tmp_path / 'gone.json'
    elif damage == 'no_receipt_flag':
        receipt = None
    elif damage == 'not_json':
        path.write_text('{ truncated')
    elif damage == 'not_diagnostic':
        smoke_receipt(path, diagnostic=False)
    elif damage == 'no_argv':
        smoke_receipt(path, argv=['--backbone', 'simple'])
    elif damage == 'saving_argv':
        smoke_receipt(path, argv=['--backbone', 'simple', '--save-dir', 'ckpt/x'])
    elif damage == 'no_status':
        smoke_receipt(path, exit_status=None)
    else:
        smoke_receipt(path, exit_status=0.0)
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(run, 'smoke', log, 0, repo=REPO, receipt=receipt)
    assert not (run / 'completion.json').exists()


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
                registry_sha256=exp06_finalize.registry_sha256(), git_state=state,
                environment={'python': '3.8.0'},
                command=['--backbone', args['backbone']], effective_args=args,
                data_identity=inventory_of(data_root))


def haa_train_args(heading_jsons, init_sha256, frame='heading', rooms=None, **overrides):
    rooms = ['class_room', 'hallway', 'complex_room'] if rooms is None else rooms
    args = dict(backbone='cylindrical_oriented', num_shot=8, rooms=rooms, frame=frame,
                save_dir='stage1', seed=0, epochs=20, val_every=10, lr=1e-4,
                weight_decay=1e-4, eval_seed=0, init='init.pth', init_sha256=init_sha256,
                heading={room: dict(heading_jsons[room]) for room in rooms})
    args.update(overrides)
    return args


def pipeline_history(epochs, val_every, lr=1e-4, val=lambda epoch: 1.0 / (epoch + 2)):
    """Exactly the rows sim_to_real/finetune_haa.py lines 95-140 write.

    Epoch 0 carries the initial validation and no train loss; every later epoch carries a
    train loss, and a validation loss on the cadence and on the final epoch; ``is_best``
    marks a strict improvement only.
    """
    rows = [{'epoch': 0, 'train_loss': None, 'val_loss': val(0), 'lr': lr}]
    best, best_epoch = val(0), 0
    for epoch in range(1, epochs + 1):
        row = {'epoch': epoch, 'train_loss': 0.5 / epoch, 'lr': lr * 0.9 ** epoch}
        if epoch % val_every == 0 or epoch == epochs:
            row['val_loss'] = val(epoch)
            if row['val_loss'] < best:
                best, best_epoch = row['val_loss'], epoch
                row['is_best'] = True
        rows.append(row)
    summary = {'best_epoch': best_epoch, 'best_val_loss': best, 'init_val_loss': val(0),
               'epochs': epochs, 'minutes': 3.5, 'seed': 0}
    return rows, summary


def write_haa_train(run, args, log, repo, data_root, rows=None, summary=None, state=None):
    run.mkdir(parents=True, exist_ok=True)
    seal(run, log, text='stage output\n')
    (run / 'args.json').write_text(json.dumps(args))
    record = haa_provenance(repo, 'haa_train', args, data_root)
    (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    written, computed = pipeline_history(args['epochs'], args['val_every'])
    rows = written if rows is None else rows
    (run / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (run / 'summary.json').write_text(json.dumps(computed if summary is None else summary))
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


def test_an_off_cadence_final_epoch_is_the_pipeline_format(tmp_path, haa_repo, heading_jsons,
                                                           data_root):
    """Blocker 3a: epochs 7 with val_every 2 validates 0, 2, 4, 6 and the final 7."""
    init = haa_repo / 'init.pth'
    torch.save(tiny_state(), init)
    args = haa_train_args(heading_jsons, provenance.sha256_file(init), epochs=7, val_every=2)
    run, log = tmp_path / 'stage1', tmp_path / 'child.log'
    write_haa_train(run, args, log, haa_repo, data_root)
    fields = exp06_finalize.finalize(run, 'haa_train', log, 0, repo=haa_repo)
    assert fields['epochs'] == [0, 2, 4, 6, 7] and fields['best_epoch'] == 7


def test_an_unimproved_initialisation_may_remain_the_best_checkpoint(tmp_path, haa_repo,
                                                                     heading_jsons, data_root):
    """Blocker 3a: best_epoch 0 is valid -- fine-tuning need not improve on the init."""
    init = haa_repo / 'init.pth'
    torch.save(tiny_state(), init)
    args = haa_train_args(heading_jsons, provenance.sha256_file(init), epochs=4, val_every=2)
    rows, summary = pipeline_history(4, 2, val=lambda epoch: 0.1 + epoch / 100.0)
    run, log = tmp_path / 'stage1', tmp_path / 'child.log'
    write_haa_train(run, args, log, haa_repo, data_root, rows=rows, summary=summary)
    fields = exp06_finalize.finalize(run, 'haa_train', log, 0, repo=haa_repo)
    assert fields['best_epoch'] == 0 and summary['best_val_loss'] == rows[0]['val_loss']


EXP02_STAGE1 = REPO / 'ckpt/sim2real/control/seed0/stage1'


@pytest.mark.skipif(not (EXP02_STAGE1 / 'history.jsonl').is_file(),
                    reason='the retained exp_02 stage-1 record is not present')
def test_the_retained_exp02_history_is_accepted(tmp_path):
    """Blocker 3a: the frozen pipeline's own output must pass the validator."""
    run = tmp_path / 'stage1'
    run.mkdir()
    for name in ('history.jsonl', 'summary.json'):
        (run / name).write_text((EXP02_STAGE1 / name).read_text())
    args = json.loads((EXP02_STAGE1 / 'args.json').read_text())
    epochs, summary = exp06_finalize.haa_history(run, args)
    assert epochs[0] == 0 and epochs[-1] == args['epochs'] and len(epochs) == 101
    assert summary['best_epoch'] == 930 and summary['best_epoch'] in epochs


def _drop(rows, epoch):
    return [row for row in rows if row['epoch'] != epoch]


HISTORY_DAMAGE = {
    'history_epochs': lambda rows, summary: (rows[:3], summary),
    'history_loss': lambda rows, summary: (
        [dict(row, val_loss=float('inf')) if 'val_loss' in row else row for row in rows], summary),
    'no_epoch_zero': lambda rows, summary: (rows[1:], summary),
    'epoch_zero_trained': lambda rows, summary: ([dict(rows[0], train_loss=0.4)] + rows[1:],
                                                 summary),
    'missing_epoch': lambda rows, summary: (_drop(rows, 7), summary),
    'off_cadence_val': lambda rows, summary: (
        [dict(row, val_loss=9.0) if row['epoch'] == 7 else row for row in rows], summary),
    'off_cadence_best': lambda rows, summary: (
        [dict(row, is_best=True) if row['epoch'] == 7 else row for row in rows], summary),
    'train_loss_nan': lambda rows, summary: (
        [dict(row, train_loss=float('nan')) if row['epoch'] == 3 else row for row in rows], summary),
    'no_lr': lambda rows, summary: (
        [{key: value for key, value in row.items() if key != 'lr'} if row['epoch'] == 5 else row
         for row in rows], summary),
    'summary_best_val': lambda rows, summary: (rows, dict(summary, best_val_loss=-999.0)),
    'summary_best_epoch': lambda rows, summary: (rows, dict(summary, best_epoch=7)),
    'summary_init_val': lambda rows, summary: (rows, dict(summary, init_val_loss=0.123)),
    'summary_epochs': lambda rows, summary: (rows, dict(summary, epochs=summary['epochs'] + 1)),
}


@pytest.mark.parametrize('damage,cause', [
    ('frame', 'frame'), ('no_heading', 'heading'), ('partial_heading', 'heading'),
    ('bad_k', 'roll'), ('bad_phi', 'phi_deg'), ('no_decision', 'estimated'),
    ('bad_sha', 'sha256'), ('no_path', 'path'), ('changed_json', 'sha256'),
    ('foreign_room', 'disagrees'), ('no_init', 'init_sha256'), ('init_changed', 'init_sha256'),
    ('summary', 'summary.json'), ('summary_keys', 'summary.json'), ('best', 'best.pth'),
    ('best_keys', 'best.pth'), ('rooms', 'rooms'), ('no_provenance', 'provenance.json'),
    ('wrong_run_type', 'run_type'), ('closure_drift', 'drift'), ('absent_entry', 'closure'),
    ('history_epochs', 'history.jsonl'), ('history_loss', 'history.jsonl'),
    ('args_disagree', 'disagree'), ('no_epoch_zero', 'epochs 0'),
    ('epoch_zero_trained', 'epoch 0'), ('missing_epoch', 'history.jsonl'),
    ('off_cadence_val', 'val_loss'), ('off_cadence_best', 'is_best'),
    ('train_loss_nan', 'train_loss'), ('no_lr', 'lr'),
    ('summary_best_val', 'best_val_loss'), ('summary_best_epoch', 'best_epoch'),
    ('summary_init_val', 'init_val_loss'), ('summary_epochs', 'epochs')])
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
    elif damage in HISTORY_DAMAGE:
        rows, summary = pipeline_history(args['epochs'], args['val_every'])
        kwargs['rows'], kwargs['summary'] = HISTORY_DAMAGE[damage](rows, summary)
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


def haa_eval_args(heading_jsons, checkpoint, frame='heading', room='hallway', **overrides):
    args = dict(backbone='cylindrical_oriented', num_shot=8, rooms=[room], frame=frame,
                checkpoint=str(checkpoint), eval_seed=0, seed=0, split='test', epochs=20,
                val_every=10, heading={room: dict(heading_jsons[room])})
    if frame != 'heading':
        args.pop('heading')
    args.update(overrides)
    return args


PER_SAMPLE = {'index': [0, 1, 2],  # ir_path is the writer's '<room>/<index>', per room
              'edt': [0.05, 0.06, 0.07], 'c50': [1.1, 1.3, float('nan')],
              't60': [4.0, 5.0, 6.0], 'stft_mse': [0.2, 0.3, 0.4],
              'loss': [0.03, 0.04, 0.05], 'env': [1.0, 2.0, 3.0], 'side_label': [1, -1, 1]}


def summarize(values):
    """eval_xRIR_backbone.summarize over the finite values, as sim_to_real/eval_haa.py calls it."""
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {'mean': None, 'median': None, 'n': 0}
    return {'mean': sum(finite) / len(finite), 'median': sorted(finite)[len(finite) // 2],
            'n': len(finite)}


def eval_metrics(args, room, per_sample, **overrides):
    """The summary dict sim_to_real/eval_haa.py writes beside the per-sample file."""
    metrics = {'backbone': args['backbone'], 'checkpoint': args['checkpoint'], 'room': room,
               'split': args['split'], 'num_shot': args['num_shot'],
               'eval_seed': args['eval_seed'], 'depth_variant': 'default',
               'n_samples': len(per_sample['index'] or []),
               'edt_error_s': summarize(per_sample['edt']),
               'c50_error_db': summarize(per_sample['c50']),
               't60_error_pct': None if room in NO_T60_ROOMS else summarize(per_sample['t60']),
               'env_error': summarize(per_sample['env']),
               'stft_log_mse': summarize(per_sample['stft_mse']),
               'test_loss': summarize(per_sample['loss']),
               'c50_outliers': 1, 't60_invalid': 0, 'edt_invalid': 0, 'elapsed_min': 2.5}
    metrics.update(overrides)
    return metrics


def write_haa_eval(run, args, log, repo, data_root, meta=None, per_sample=None, metrics=None):
    run.mkdir(parents=True, exist_ok=True)
    seal(run, log, text='eval output\n')
    (run / 'args.json').write_text(json.dumps(args))
    record = haa_provenance(repo, 'haa_eval', args, data_root)
    (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    room = args['rooms'][0]
    body = dict(PER_SAMPLE, meta=meta)
    body.update(per_sample or {})
    if isinstance(body.get('index'), list) and 'ir_path' not in (per_sample or {}):
        body['ir_path'] = ['{}/{}'.format(room, index) for index in body['index']]
    summary = eval_metrics(args, room, body) if metrics is None else metrics
    (run / 'metrics_{}.json'.format(room)).write_text(json.dumps(summary))
    (run / 'per_sample_{}.json'.format(room)).write_text(json.dumps(body))
    return run


def eval_meta(args, checkpoint_sha256, **overrides):
    meta = {'backbone': args['backbone'], 'checkpoint_sha256': checkpoint_sha256,
            'frame': args['frame'], 'split': args['split'], 'num_shot': args['num_shot'],
            'eval_seed': args['eval_seed']}
    if args['frame'] == 'heading':
        meta['heading'] = json.loads(json.dumps(args['heading']))
    meta.update(overrides)
    return meta


@pytest.fixture
def haa_eval_run(tmp_path, haa_repo, heading_jsons, data_root):
    checkpoint = haa_repo / 'stage2_best.pth'
    torch.save(tiny_state(), checkpoint)
    args = haa_eval_args(heading_jsons, checkpoint)
    run, log = tmp_path / 'eval' / 'hallway', tmp_path / 'child.log'
    write_haa_eval(run, args, log, haa_repo, data_root,
                   meta=eval_meta(args, provenance.sha256_file(checkpoint)))
    return run, log, args, checkpoint


def test_haa_eval_completion_records_the_single_room(haa_eval_run, haa_repo):
    run, log, args, checkpoint = haa_eval_run
    fields = exp06_finalize.finalize(run, 'haa_eval', log, 0, repo=haa_repo)
    assert fields['room'] == 'hallway' and fields['frame'] == 'heading'
    assert set(fields['artifacts']) == {'provenance.json', 'args.json', 'metrics_hallway.json',
                                        'per_sample_hallway.json'}
    assert fields['checkpoint_sha256'] == provenance.sha256_file(checkpoint)
    assert fields['samples'] == 3 and fields['heading']['hallway']['k'] == args['heading']['hallway']['k']


def test_haa_eval_in_the_room_frame_records_no_heading(tmp_path, haa_repo, heading_jsons, data_root):
    checkpoint = haa_repo / 'stage2_best.pth'
    torch.save(tiny_state(), checkpoint)
    args = haa_eval_args(heading_jsons, checkpoint, frame='room')
    run, log = tmp_path / 'eval' / 'hallway', tmp_path / 'child.log'
    write_haa_eval(run, args, log, haa_repo, data_root,
                   meta=eval_meta(args, provenance.sha256_file(checkpoint)))
    assert exp06_finalize.finalize(run, 'haa_eval', log, 0, repo=haa_repo)['heading'] is None


@pytest.mark.parametrize('damage,cause', [
    ('two_rooms', 'room'), ('no_metrics', 'metrics_hallway.json'),
    ('no_per_sample', 'per_sample_hallway.json'), ('no_meta_heading', 'heading'),
    ('no_meta_backbone', 'backbone'), ('meta_backbone_differs', 'backbone'),
    ('meta_heading_differs', 'heading'), ('meta_frame_differs', 'frame'),
    ('checkpoint_hash_differs', 'checkpoint_sha256'), ('checkpoint_missing', 'checkpoint'),
    ('checkpoint_keys', 'checkpoint'), ('no_side_label', 'side_label'),
    ('short_side_label', 'side_label'), ('bad_side_label', 'side_label'),
    ('boolean_side_label', 'side_label'), ('no_index', 'index'),
    ('no_provenance', 'provenance.json'), ('wrong_run_type', 'run_type')])
def test_haa_eval_refusals_are_named(tmp_path, haa_repo, heading_jsons, data_root, damage, cause):
    checkpoint = haa_repo / 'stage2_best.pth'
    torch.save(tiny_state(), checkpoint)
    if damage == 'checkpoint_keys':
        torch.save({'source_network.weight': torch.zeros(1)}, checkpoint)
    args = haa_eval_args(heading_jsons, checkpoint)
    digest = provenance.sha256_file(checkpoint)
    if damage == 'two_rooms':
        args['rooms'] = ['hallway', 'class_room']
        args['heading']['class_room'] = dict(heading_jsons['class_room'])
    elif damage == 'checkpoint_missing':
        args['checkpoint'] = str(haa_repo / 'absent.pth')
    meta, extra = eval_meta(args, digest), {}
    if damage == 'no_meta_heading':
        meta.pop('heading')
    elif damage == 'no_meta_backbone':
        meta.pop('backbone')
    elif damage == 'meta_backbone_differs':
        meta['backbone'] = 'cylindrical'
    elif damage == 'meta_heading_differs':
        meta['heading']['hallway']['k'] = (meta['heading']['hallway']['k'] + 1) % 512
    elif damage == 'meta_frame_differs':
        meta['frame'] = 'room'
    elif damage == 'checkpoint_hash_differs':
        meta['checkpoint_sha256'] = 'd' * 64
    elif damage == 'no_side_label':
        extra['side_label'] = None
    elif damage == 'short_side_label':
        extra['side_label'] = [1, -1]
    elif damage == 'bad_side_label':
        extra['side_label'] = [1, 0, -1]
    elif damage == 'boolean_side_label':
        extra['side_label'] = [True, -1, 1]
    elif damage == 'no_index':
        extra['index'] = None
    run, log = tmp_path / 'eval' / 'hallway', tmp_path / 'child.log'
    write_haa_eval(run, args, log, haa_repo, data_root, meta=meta, per_sample=extra)
    if damage == 'no_metrics':
        (run / 'metrics_hallway.json').unlink()
    elif damage == 'no_per_sample':
        (run / 'per_sample_hallway.json').unlink()
    elif damage == 'no_provenance':
        (run / 'provenance.json').unlink()
    elif damage == 'wrong_run_type':
        record = json.loads((run / 'provenance.json').read_text())
        record['run_type'] = 'haa_train'
        (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(run, 'haa_eval', log, 0, repo=haa_repo)
    assert not (run / 'completion.json').exists()


def test_the_dampened_room_records_no_t60(tmp_path, haa_repo, heading_jsons, data_root):
    """Blocker 3b: the paper omits T60 there, so the summary must record null."""
    checkpoint = haa_repo / 'stage2_best.pth'
    torch.save(tiny_state(), checkpoint)
    args = haa_eval_args(heading_jsons, checkpoint, room='dampened_room')
    run, log = tmp_path / 'eval' / 'dampened_room', tmp_path / 'child.log'
    write_haa_eval(run, args, log, haa_repo, data_root,
                   meta=eval_meta(args, provenance.sha256_file(checkpoint)))
    assert exp06_finalize.finalize(run, 'haa_eval', log, 0, repo=haa_repo)['room'] == 'dampened_room'


METRIC_DAMAGE = {
    'metrics_not_json': ('not JSON', 'metrics_hallway.json'),
    'metrics_key': (lambda m: {k: v for k, v in m.items() if k != 'c50_error_db'},
                    'c50_error_db'),
    'metrics_samples': (lambda m: dict(m, n_samples=2), 'n_samples'),
    'metrics_mean': (lambda m: dict(m, edt_error_s=dict(m['edt_error_s'], mean=0.5)), 'mean'),
    'metrics_n': (lambda m: dict(m, c50_error_db=dict(m['c50_error_db'], n=3)), 'c50_error_db'),
    'metrics_room': (lambda m: dict(m, room='class_room'), 'room'),
    'metrics_backbone': (lambda m: dict(m, backbone='cylindrical'), 'backbone'),
    'metrics_seed': (lambda m: dict(m, eval_seed=7), 'eval_seed'),
    'metrics_checkpoint': (lambda m: dict(m, checkpoint='/elsewhere.pth'), 'checkpoint'),
    'metrics_t60': (lambda m: dict(m, t60_error_pct=None), 't60_error_pct'),
    'count_string': (lambda m: dict(m, c50_outliers='nonsense'), 'c50_outliers'),
    'count_negative': (lambda m: dict(m, edt_invalid=-5), 'edt_invalid'),
    'count_object': (lambda m: dict(m, t60_invalid={}), 't60_invalid'),
    'count_boolean': (lambda m: dict(m, edt_invalid=True), 'edt_invalid'),
    'count_above_n': (lambda m: dict(m, c50_outliers=4), 'c50_outliers'),
    'median_object': (lambda m: dict(m, edt_error_s=dict(m['edt_error_s'],
                                                         median={'nonsense': True})), 'median'),
    'median_wrong': (lambda m: dict(m, edt_error_s=dict(m['edt_error_s'], median=9.0)), 'median'),
    'summary_extra_key': (lambda m: dict(m, env_error=dict(m['env_error'], std=0.1)),
                          'env_error'),
    'n_boolean': (lambda m: dict(m, n_samples=True), 'n_samples'),
}


@pytest.mark.parametrize('damage', sorted(METRIC_DAMAGE))
def test_the_metrics_summary_is_parsed_and_cross_checked(tmp_path, haa_repo, heading_jsons,
                                                         data_root, damage):
    """Blocker 3b: metrics were hashed without being read; now they must agree."""
    mutate, cause = METRIC_DAMAGE[damage]
    checkpoint = haa_repo / 'stage2_best.pth'
    torch.save(tiny_state(), checkpoint)
    args = haa_eval_args(heading_jsons, checkpoint)
    run, log = tmp_path / 'eval' / 'hallway', tmp_path / 'child.log'
    write_haa_eval(run, args, log, haa_repo, data_root,
                   meta=eval_meta(args, provenance.sha256_file(checkpoint)))
    metrics = json.loads((run / 'metrics_hallway.json').read_text())
    (run / 'metrics_hallway.json').write_text(
        mutate if isinstance(mutate, str) else json.dumps(mutate(metrics)))
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(run, 'haa_eval', log, 0, repo=haa_repo)
    assert not (run / 'completion.json').exists()


def test_an_evaluation_child_must_sit_in_its_own_room_directory(tmp_path, haa_repo,
                                                                heading_jsons, data_root):
    """Blocker 3c: a child may not claim a room other than the directory it ran in."""
    checkpoint = haa_repo / 'stage2_best.pth'
    torch.save(tiny_state(), checkpoint)
    args = haa_eval_args(heading_jsons, checkpoint, room='class_room')
    run, log = tmp_path / 'eval' / 'hallway', tmp_path / 'child.log'
    write_haa_eval(run, args, log, haa_repo, data_root,
                   meta=eval_meta(args, provenance.sha256_file(checkpoint)))
    with pytest.raises(ValueError, match='room'):
        exp06_finalize.finalize(run, 'haa_eval', log, 0, repo=haa_repo)
    assert not (run / 'completion.json').exists()


@pytest.fixture
def closed_log_file(tmp_path):
    return tmp_path / 'child.log'


PRETRAIN = 'pretrain_epoch_012'


def job_spec_file(job, init, heading_jsons, **overrides):
    """What the pipeline declares for one seed before any child runs (blocker 3c)."""
    spec = {'init': PRETRAIN, 'backbone': 'cylindrical_oriented', 'frame': 'heading',
            'init_sha256': provenance.sha256_file(init), 'seed': 0, 'rooms': sorted(ROOMS),
            'heading': {room: heading_jsons[room]['k'] for room in ROOMS}, 'expect': 'finetune'}
    spec.update(overrides)
    path = Path(job) / 'job_spec.json'
    path.write_text(json.dumps(spec, sort_keys=True, indent=2) + '\n')
    return str(path), spec


def make_train_child(job, name, haa_repo, heading_jsons, data_root, rooms, init, tag):
    """One fine-tuning child, certified by this finalizer exactly as the launcher does."""
    run, log = job / name, job / (name.replace('/', '_') + '.log')
    args = haa_train_args(heading_jsons, provenance.sha256_file(init), rooms=rooms,
                          init=str(init), save_dir=name, epochs=4, val_every=2)
    write_haa_train(run, args, log, haa_repo, data_root,
                    state=tiny_state(**{state_keys()[0]: torch.full((1,), float(tag))}))
    exp06_finalize.finalize(run, 'haa_train', log, 0, repo=haa_repo)
    return run


def make_eval_child(job, name, haa_repo, heading_jsons, data_root, room, checkpoint):
    run, log = job / name, job / (name.replace('/', '_') + '.log')
    args = haa_eval_args(heading_jsons, checkpoint, room=room)
    write_haa_eval(run, args, log, haa_repo, data_root,
                   meta=eval_meta(args, provenance.sha256_file(checkpoint)))
    exp06_finalize.finalize(run, 'haa_eval', log, 0, repo=haa_repo)
    return run


def write_job(tmp_path, haa_repo, heading_jsons, data_root, expect='finetune', log=None,
              mutate=None):
    """A complete pipeline seed whose children carry their real evidence, not claims."""
    job = tmp_path / 'seed0'
    job.mkdir(parents=True, exist_ok=True)
    init = tmp_path / 'pretrain.pth'
    torch.save(tiny_state(**{state_keys()[0]: torch.full((1,), 99.0)}), init)
    if log is not None:
        seal(job, log, text='pipeline output\n')
    names = []
    if expect == 'finetune':
        make_train_child(job, 'stage1', haa_repo, heading_jsons, data_root,
                         ['class_room', 'hallway', 'complex_room'], init, 1)
        names.append('stage1')
    for tag, room in enumerate(sorted(ROOMS), 2):
        checkpoint = init
        if expect == 'finetune':
            make_train_child(job, 'stage2_' + room, haa_repo, heading_jsons, data_root,
                             [room], job / 'stage1/best.pth', tag)
            checkpoint = job / ('stage2_' + room) / 'best.pth'
            names.append('stage2_' + room)
        make_eval_child(job, 'eval/' + room, haa_repo, heading_jsons, data_root, room, checkpoint)
        names.append('eval/' + room)
    spec, _ = job_spec_file(job, init, heading_jsons, expect=expect)
    if mutate is not None:
        names = mutate(job, names)
    return job, [str(job / name) for name in names], spec


@pytest.fixture
def job_run(tmp_path, haa_repo, heading_jsons, data_root, closed_log_file):
    """One finetune seed of nine real children, ready to be bound as a job."""
    def build(expect='finetune', mutate=None):
        return write_job(tmp_path, haa_repo, heading_jsons, data_root, expect=expect,
                         log=closed_log_file, mutate=mutate) + (haa_repo,)
    return build


def test_job_completion_requires_every_child(job_run, closed_log_file):
    job, children, spec, repo = job_run()
    fields = exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=repo,
                                     children=children, expect='finetune', job_spec=spec)
    assert set(fields['children']) == set(exp06_finalize.expected_children('finetune'))
    assert len(fields['children']) == 9 and fields['expect'] == 'finetune'
    assert fields['children']['stage1'] == provenance.sha256_file(job / 'stage1/completion.json')
    assert fields['backbone'] == 'cylindrical_oriented' and fields['frame'] == 'heading'
    assert fields['init'] == PRETRAIN and fields['admissible_arm'] is True
    assert fields['init_sha256'] == json.loads(Path(spec).read_text())['init_sha256']
    assert fields['heading']['hallway'] == json.loads(Path(spec).read_text())['heading']['hallway']
    assert set(exp06_finalize.expected_children('zeroshot')) == {'eval/' + room for room in ROOMS}


def test_zeroshot_job_shares_one_checkpoint(job_run, closed_log_file):
    job, children, spec, repo = job_run('zeroshot')
    fields = exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=repo,
                                     children=children, expect='zeroshot', job_spec=spec)
    assert len(fields['children']) == 4 and fields['expect'] == 'zeroshot'
    assert fields['checkpoint_sha256'] == json.loads(Path(spec).read_text())['init_sha256']


def test_a_job_of_bare_admissible_claims_is_refused(tmp_path, closed_log_file):
    """The review's third counterexample: nine children asserting their own admission."""
    job = tmp_path / 'seed0'
    seal(job, closed_log_file, text='pipeline output\n')
    names = list(exp06_finalize.expected_children('finetune'))
    for name in names:
        (job / name).mkdir(parents=True)
        (job / name / 'completion.json').write_text(
            json.dumps({'run_type': 'haa_eval', 'admissible_arm': True}))
    spec = job / 'job_spec.json'
    spec.write_text(json.dumps({'init': PRETRAIN, 'backbone': 'cylindrical_oriented',
                                'frame': 'heading', 'init_sha256': 'a' * 64, 'seed': 0,
                                'rooms': sorted(ROOMS), 'expect': 'finetune',
                                'heading': {room: 0 for room in ROOMS}}))
    with pytest.raises(ValueError, match='provenance.json'):
        exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=REPO, job_spec=str(spec),
                                children=[str(job / name) for name in names], expect='finetune')
    assert not (job / 'completion.json').exists()


@pytest.mark.parametrize('damage,cause', [
    ('missing', 'missing'), ('extra', 'unexpected'), ('no_completion', 'completion.json'),
    ('outside', 'outside'), ('wrong_role', 'run type'), ('diagnostic', 'diagnostic'),
    ('nonzero_exit', 'child_exit'), ('stale_hash', 'artefact'),
    ('missing_artefact', 'last.pth'), ('backbone', 'backbone'), ('frame', 'frame'),
    ('heading_k', 'heading'), ('lineage_init', 'lineage'), ('lineage_checkpoint', 'lineage'),
    ('schema', 'incomplete'), ('eval_artefacts_removed', 'metrics_hallway.json'),
    ('stale_log', 'log'), ('stale_receipt', 'child_exit_receipt'), ('wrong_room', 'room'),
    ('wrong_seed', 'seed'), ('run_dir', 'run_dir'), ('schema_version', 'schema_version')])
def test_job_refusals_are_named(job_run, closed_log_file, tmp_path, haa_repo, heading_jsons,
                                data_root, damage, cause):
    def rebuild(job, name):
        shutil.rmtree(str(job / name))
        (job / (name.replace('/', '_') + '.log')).unlink()

    def mutate(job, names):
        if damage == 'missing':
            names.remove('stage2_hallway')
        elif damage == 'extra':
            make_train_child(job, 'stage2_invented', haa_repo, heading_jsons, data_root,
                             ['hallway'], tmp_path / 'pretrain.pth', 9)
            names.append('stage2_invented')
        elif damage == 'no_completion':
            (job / 'stage1/completion.json').unlink()
        elif damage == 'outside':
            names.append('../elsewhere')
            make_train_child(job.parent, 'elsewhere', haa_repo, heading_jsons, data_root,
                             ['hallway'], tmp_path / 'pretrain.pth', 9)
        elif damage == 'stale_hash':  # a valid checkpoint, but not the one that was certified
            torch.save(tiny_state(**{state_keys()[0]: torch.full((1,), 42.0)}),
                       job / 'stage1/best.pth')
        elif damage == 'missing_artefact':
            (job / 'stage1/last.pth').unlink()
        elif damage == 'eval_artefacts_removed':
            record = json.loads((job / 'eval/hallway/completion.json').read_text())
            for name in ('metrics_hallway.json', 'per_sample_hallway.json'):
                (job / 'eval/hallway' / name).unlink()
                record['artifacts'].pop(name)
            (job / 'eval/hallway/completion.json').write_text(json.dumps(record, sort_keys=True))
        elif damage == 'stale_log':
            with open(str(job / 'stage1.log'), 'a') as stream:
                stream.write('appended after the completion\n')
        elif damage == 'stale_receipt':
            (job / 'stage1/child_exit.json').write_text('{"child_pid": 1}')
        elif damage == 'lineage_init':  # a valid child, but not started from stage1/best.pth
            rebuild(job, 'stage2_hallway')
            rebuild(job, 'eval/hallway')
            make_train_child(job, 'stage2_hallway', haa_repo, heading_jsons, data_root,
                             ['hallway'], tmp_path / 'pretrain.pth', 7)
            make_eval_child(job, 'eval/hallway', haa_repo, heading_jsons, data_root, 'hallway',
                            job / 'stage2_hallway/best.pth')
        elif damage == 'lineage_checkpoint':
            rebuild(job, 'eval/hallway')
            make_eval_child(job, 'eval/hallway', haa_repo, heading_jsons, data_root, 'hallway',
                            job / 'stage2_class_room/best.pth')
        elif damage == 'wrong_seed':
            rebuild(job, 'stage2_hallway')
            run, log = job / 'stage2_hallway', job / 'stage2_hallway.log'
            args = haa_train_args(heading_jsons, provenance.sha256_file(job / 'stage1/best.pth'),
                                  rooms=['hallway'], init=str(job / 'stage1/best.pth'),
                                  save_dir='stage2_hallway', epochs=4, val_every=2, seed=1)
            write_haa_train(run, args, log, haa_repo, data_root)
            exp06_finalize.finalize(run, 'haa_train', log, 0, repo=haa_repo)
        else:
            path = job / {'wrong_role': 'stage1', 'diagnostic': 'stage1', 'nonzero_exit': 'stage1',
                          'backbone': 'eval/hallway', 'frame': 'eval/hallway',
                          'heading_k': 'eval/hallway', 'wrong_room': 'eval/hallway',
                          'run_dir': 'stage1', 'schema_version': 'stage1', 'schema': 'stage1'}[damage]
            record = json.loads((path / 'completion.json').read_text())
            if damage == 'wrong_role':
                record['run_type'] = 'haa_eval'
            elif damage == 'diagnostic':
                record['diagnostic'] = True
            elif damage == 'nonzero_exit':
                record['child_exit'] = 1
            elif damage == 'backbone':
                record['backbone'] = 'cylindrical'
            elif damage == 'frame':
                record['frame'] = 'room'
            elif damage == 'wrong_room':
                record['room'] = 'class_room'
            elif damage == 'run_dir':
                record['run_dir'] = str(job / 'stage2_hallway')
            elif damage == 'schema_version':
                record['schema_version'] = 2
            elif damage == 'heading_k':
                record['heading'] = copy.deepcopy(record['heading'])
                record['heading']['hallway']['k'] = 0
            else:
                record.pop('artifacts')
            (path / 'completion.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
        return names

    job, children, spec, repo = job_run(mutate=mutate)
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=repo,
                                children=children, expect='finetune', job_spec=spec)
    assert not (job / 'completion.json').exists()


@pytest.mark.parametrize('damage,cause', [
    ('absent', 'job-spec'), ('not_json', 'job spec'), ('no_field', 'incomplete'),
    ('expect', 'expect'), ('backbone', 'backbone'), ('frame', 'frame'),
    ('init_sha256', 'init_sha256'), ('seed', 'seed'), ('rooms', 'rooms'),
    ('heading_room', 'heading'), ('heading_in_room_frame', 'heading')])
def test_the_job_spec_is_required_and_validated(job_run, closed_log_file, tmp_path, damage, cause):
    job, children, spec, repo = job_run()
    record = json.loads(Path(spec).read_text())
    if damage == 'absent':
        spec = None
    elif damage == 'not_json':
        Path(spec).write_text('{ truncated')
    else:
        if damage == 'no_field':
            record.pop('init')
        elif damage == 'expect':
            record['expect'] = 'zeroshot'
        elif damage == 'backbone':
            record['backbone'] = 'invented'
        elif damage == 'frame':
            record['frame'] = 'world'
        elif damage == 'init_sha256':
            record['init_sha256'] = 'zz'
        elif damage == 'seed':
            record['seed'] = '0'
        elif damage == 'rooms':
            record['rooms'] = sorted(ROOMS)[:2]
        elif damage == 'heading_room':
            record['heading'].pop('hallway')
        else:
            record['frame'] = 'room'
        Path(spec).write_text(json.dumps(record, sort_keys=True, indent=2))
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=repo,
                                children=children, expect='finetune', job_spec=spec)
    assert not (job / 'completion.json').exists()


@pytest.mark.parametrize('damage,cause', [
    ('no_marker', 'EXP06_CHILD_EXIT'), ('receipt_status', 'child_exit.json')])
def test_a_child_whose_log_was_never_closed_is_refused(job_run, closed_log_file, damage, cause):
    """Blocker 3c: the job re-validates the marker and the receipt, not just their hashes."""
    def mutate(job, names):
        record = json.loads((job / 'stage1/completion.json').read_text())
        log = job / 'stage1.log'
        if damage == 'no_marker':
            log.write_text('output that was never closed\n')
            record['log']['sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
        else:
            receipt = json.loads((job / 'stage1/child_exit.json').read_text())
            receipt['status'] = 3
            (job / 'stage1/child_exit.json').write_text(json.dumps(receipt, sort_keys=True))
            record['child_exit_receipt']['sha256'] = provenance.sha256_file(
                job / 'stage1/child_exit.json')
        (job / 'stage1/completion.json').write_text(
            json.dumps(record, sort_keys=True, indent=2) + '\n')
        return names

    job, children, spec, repo = job_run(mutate=mutate)
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=repo,
                                children=children, expect='finetune', job_spec=spec)
    assert not (job / 'completion.json').exists()


def test_job_requires_a_declared_expectation(job_run, closed_log_file):
    job, children, spec, repo = job_run()
    with pytest.raises(ValueError, match='expect'):
        exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=repo, children=children,
                                job_spec=spec)


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
    ('model_unsortable_keys', 'last.pth'),
    ('git_state_not_a_record', 'git_state'),
    ('git_state_incomplete', 'git_state'),
    ('mutable_inputs_null', 'mutable_inputs'),
    ('mutable_inputs_pathless', 'mutable_inputs'),
    ('data_identity_not_a_record', 'data_identity'),
    ('environment_not_a_record', 'environment'),
])
def test_malformed_inputs_raise_named_refusals(full_run, clone, damage, cause):
    """Should-fix 9: malformed containers must refuse by name, never raise KeyError."""
    run, log = full_run
    args = json.loads((run / 'args.json').read_text())
    if damage == 'last_not_a_dict':
        torch.save(torch.zeros(3), run / 'last.pth')
    elif damage == 'last_without_model':
        torch.save({'epoch': 12, 'batch_idx': 0, 'args': args}, run / 'last.pth')
    elif damage == 'last_model_not_a_state_dict':
        torch.save({'model': 'not-a-state-dict', 'epoch': 12, 'batch_idx': 0,
                    'args': args}, run / 'last.pth')
    elif damage == 'model_unsortable_keys':
        torch.save({'model': {1: torch.zeros(1), 'bad': 'not a tensor'}, 'epoch': 12,
                    'batch_idx': 0, 'args': args}, run / 'last.pth')
    elif damage in ('git_state_not_a_record', 'git_state_incomplete', 'mutable_inputs_null',
                    'mutable_inputs_pathless', 'data_identity_not_a_record',
                    'environment_not_a_record'):
        record = json.loads((run / 'provenance.json').read_text())
        if damage == 'git_state_not_a_record':
            record['git_state'] = []
        elif damage == 'git_state_incomplete':
            record['git_state'] = {'HEAD': record['git_state']['HEAD']}
        elif damage == 'mutable_inputs_null':
            record['mutable_inputs'] = None
        elif damage == 'mutable_inputs_pathless':
            record['mutable_inputs'] = {'heading': {'sha256': 'a' * 64}}
        elif damage == 'environment_not_a_record':
            record['environment'] = ['python 3.8']
        else:
            record['data_identity'] = []
        (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
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
    args = json.loads((run / 'args.json').read_text())
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
        floats = {key: float(value)
                  for key, value in json.loads((run / 'args.json').read_text())['param_counts'].items()}
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


@pytest.mark.parametrize('field,value', [
    ('exp06_run_type', 'probe'), ('exp06_git_head', 'c' * 40),
    ('exp06_registry_sha256', 'a' * 64), ('exp06_source_closure_sha256', 'b' * 64),
    ('exp06_provenance_path', '/wrong/provenance.json')])
def test_false_exp06_bindings_are_refused_even_when_all_copies_agree(full_run, clone, field, value):
    """Blocker 2: the exp06_* fields bind the execution record, not each other."""
    run, log = full_run
    rewrite_sources(run, mutate_json=lambda a: dict(a, **{field: value}))
    with pytest.raises(ValueError, match=field):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


def test_the_registry_digest_is_recomputed_at_finalisation(full_run, clone):
    """Blocker 2: provenance may not declare a registry other than the one on disk."""
    run, log = full_run
    record = json.loads((run / 'provenance.json').read_text())
    assert record['registry_sha256'] == exp06_train.registry_sha256()
    record['registry_sha256'] = 'a' * 64
    (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    with pytest.raises(ValueError, match='registry_sha256'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


def test_the_provenance_path_must_be_the_validated_record(full_run, clone):
    """Blocker 2: a binding to some other provenance file is not this run's record."""
    run, log = full_run
    other = run.parent / 'provenance.json'
    other.write_text((run / 'provenance.json').read_text())
    rewrite_sources(run, mutate_json=lambda a: dict(a, exp06_provenance_path=str(other)))
    with pytest.raises(ValueError, match='exp06_provenance_path'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    rewrite_sources(run, mutate_json=lambda a: dict(
        a, exp06_provenance_path=os.path.relpath(str(run / 'provenance.json'), str(clone))))
    assert exp06_finalize.finalize(run, 'full', log, 0, repo=clone)['admissible_arm'] is True


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
    (data_root / TRAIN_IRS[1]).write_bytes(b'sample 1, edited')
    with pytest.raises(ValueError, match='revalidation'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()
    (data_root / TRAIN_IRS[0]).unlink()
    with pytest.raises(ValueError, match='inventory|revalidation'):
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


def rewrite_identity(run, **overrides):
    """Replace the recorded training-data identity with a damaged copy."""
    record = json.loads((run / 'provenance.json').read_text())
    record['train_data_identity'] = dict(record['train_data_identity'], **overrides)
    (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    return record


def test_the_expected_membership_comes_from_the_pinned_helper(data_root, monkeypatch):
    """Blocker 1: the membership is derived, never read out of the record."""
    calls = []
    real = provenance.train_data_identity
    monkeypatch.setattr(provenance, 'train_data_identity',
                        lambda root, cache_path=None, **kw: (calls.append((root, cache_path)),
                                                             real(root, cache_path=cache_path))[1])
    paths = exp06_finalize.train_inventory_paths(str(Path(data_root).resolve()))
    assert paths == set(TRAIN_IRS), paths
    assert TEST_IR not in paths and calls[0][0] == str(Path(data_root).resolve())
    assert str(exp06_finalize.TRAIN_INVENTORY).endswith('ckpt/yaw_aug/train_inventory.json')
    with pytest.raises(ValueError, match='training inventory'):
        exp06_finalize.train_inventory_paths(str(data_root / 'absent'))


@pytest.mark.parametrize('damage,cause', [
    ('empty', 'empty'), ('short', 'training split'), ('root', 'data_root'),
    ('files', 'inventory_files'), ('bytes', 'inventory_bytes'), ('digest', 'inventory_sha256'),
    ('entry', 'inventory entry'), ('not_a_list', 'inventory'), ('no_root', 'data_root'),
])
def test_an_incomplete_training_inventory_is_refused(full_run, clone, tmp_path, damage, cause):
    """Blocker 1: an empty or partial inventory establishes no training-data identity."""
    run, log = full_run
    identity = json.loads((run / 'provenance.json').read_text())['train_data_identity']
    entries = identity['inventory']
    if damage == 'empty':
        rewrite_identity(run, inventory=[], inventory_files=0, inventory_bytes=0,
                         inventory_sha256=hashlib.sha256(b'[]').hexdigest())
    elif damage == 'short':
        kept = entries[:-1]
        rewrite_identity(run, inventory=kept, inventory_files=len(kept),
                         inventory_bytes=sum(entry['size'] for entry in kept),
                         inventory_sha256=exp06_finalize.inventory_digest(kept))
    elif damage == 'root':
        rewrite_identity(run, data_root=str(tmp_path / 'elsewhere'))
    elif damage == 'no_root':
        record = json.loads((run / 'provenance.json').read_text())
        record['train_data_identity'].pop('data_root')
        (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    elif damage == 'files':
        rewrite_identity(run, inventory_files=len(entries) + 1)
    elif damage == 'bytes':
        rewrite_identity(run, inventory_bytes=identity['inventory_bytes'] + 1)
    elif damage == 'digest':
        rewrite_identity(run, inventory_sha256='f' * 64)
    elif damage == 'entry':
        broken = [dict(entries[0], sha256='zz')] + entries[1:]
        rewrite_identity(run, inventory=broken)
    else:
        rewrite_identity(run, inventory={'path': entries[0]['path']})
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


def test_the_recorded_data_root_must_be_the_one_the_run_resolved(full_run, clone, tmp_path):
    """Blocker 1: identity and execution record must name the same resolved root."""
    run, log = full_run
    record = json.loads((run / 'provenance.json').read_text())
    record['data_root'] = str(tmp_path / 'another_mirror')
    (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    with pytest.raises(ValueError, match='data_root'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    record.pop('data_root')
    (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    with pytest.raises(ValueError, match='data_root'):
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
    ('not_json', 'child_exit.json'), ('no_pid', 'child_exit.json'),
    ('pid_zero', 'child_pid'), ('pid_string', 'child_pid'), ('no_started_at', 'started_at'),
    ('started_after_ended', 'started_at'), ('ended_not_a_time', 'ended_at'),
    ('ended_naive', 'ended_at'), ('ended_not_the_marker', 'ended_at'),
    ('hash_not_hex', 'log_sha256_after_marker')])
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
        elif damage == 'pid_zero':
            receipt['child_pid'] = 0
        elif damage == 'pid_string':
            receipt['child_pid'] = str(receipt['child_pid'])
        elif damage == 'no_started_at':
            receipt.pop('started_at')
        elif damage == 'started_after_ended':
            receipt['started_at'] = '2026-09-15T05:00:00+00:00'
        elif damage == 'ended_not_a_time':
            receipt['ended_at'] = 'whenever'
        elif damage == 'ended_naive':
            receipt['ended_at'] = STAMP.replace('+00:00', '')
        elif damage == 'ended_not_the_marker':
            receipt['ended_at'] = '2026-09-15T04:05:07.070809+00:00'
        elif damage == 'hash_not_hex':
            receipt['log_sha256_after_marker'] = 'z' * 64
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
               '--log', str(log), '--child-pid', '4242', '--status', '0',
               '--started-at', STARTED]
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                               env={**os.environ, 'PYTHONPATH': str(REPO)})
    assert completed.returncode == 0, completed.stderr
    lines = log.read_text().splitlines()
    assert lines[-1].startswith('EXP06_CHILD_EXIT 0 ') and lines[0] == 'child output'
    receipt = json.loads((run / 'child_exit.json').read_text())
    assert receipt['status'] == 0 and receipt['child_pid'] == 4242
    assert receipt['log_sha256_after_marker'] == provenance.sha256_file(log)
    assert receipt['ended_at'] == lines[-1].split()[2] and receipt['started_at'] == STARTED
    other = tmp_path / 'attempt_two'
    other.mkdir()
    refused = subprocess.run([part if part != str(run) else str(other) for part in command[:-1]]
                             + ['not-a-timestamp'], cwd=REPO, capture_output=True,
                             text=True, env={**os.environ, 'PYTHONPATH': str(REPO)})
    assert refused.returncode == 2 and 'started_at' in refused.stderr
    assert not (other / 'child_exit.json').exists()
    again = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                           env={**os.environ, 'PYTHONPATH': str(REPO)})
    assert again.returncode == 2 and 'child_exit.json' in again.stderr


@pytest.mark.parametrize('child,missing', [('stage1', 'best.pth'), ('stage1', 'last.pth'),
                                           ('eval/hallway', 'args.json')])
def test_a_child_that_records_no_role_artefact_is_refused(job_run, closed_log_file,
                                                          child, missing):
    """Should-fix 9 on the job branch: a thin artefact map must refuse, not raise KeyError."""
    def mutate(job, names):
        record = json.loads((job / child / 'completion.json').read_text())
        record['artifacts'].pop(missing, None)
        (job / child / 'completion.json').write_text(json.dumps(record, sort_keys=True, indent=2))
        return names

    job, children, spec, repo = job_run(mutate=mutate)
    with pytest.raises(ValueError, match=missing):
        exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=repo,
                                children=children, expect='finetune', job_spec=spec)
    assert not (job / 'completion.json').exists()


@pytest.mark.parametrize('name', ['child.pid', 'launch.pid'])
def test_a_live_child_is_never_certified_by_a_job(job_run, closed_log_file, name):
    """Finding 1: job admission must apply the liveness contract to every child."""
    def mutate(job, names):
        (job / 'stage1' / name).write_text('{}\n'.format(os.getpid()))
        return names

    job, children, spec, repo = job_run(mutate=mutate)
    for owner in (None, os.getpid()):
        with pytest.raises(ValueError, match='alive'):
            exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=repo,
                                    children=children, expect='finetune', job_spec=spec,
                                    owner_pid=owner)
    assert not (job / 'completion.json').exists()


def test_the_job_owner_may_finalize_once_every_child_is_dead(job_run, closed_log_file):
    """Finding 1: the owner exception covers the job's own launch.pid, never a child's."""
    def mutate(job, names):
        (job / 'launch.pid').write_text('{}\n'.format(os.getpid()))
        (job / 'stage1/child.pid').write_text('999999999\n')
        return names

    job, children, spec, repo = job_run(mutate=mutate)
    fields = exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=repo,
                                     children=children, expect='finetune', job_spec=spec,
                                     owner_pid=os.getpid())
    assert fields['admissible_arm'] is True


@pytest.mark.parametrize('damage,cause', [
    ('receipt_redirected', 'child_exit_receipt'), ('receipt_pid', 'child_exit_receipt'),
    ('exit_time', 'child_exit_time'), ('closure_digest', 'source_closure_sha256'),
    ('closure_absent', 'source_closure_sha256')])
def test_a_child_completion_must_bind_what_the_job_validated(job_run, closed_log_file,
                                                             damage, cause):
    """Finding 2: the receipt hashed must be the receipt validated, field for field."""
    def mutate(job, names):
        record = json.loads((job / 'stage1/completion.json').read_text())
        if damage == 'receipt_redirected':  # a correct hash of the wrong file
            record['child_exit_receipt'] = dict(
                record['child_exit_receipt'], path=str(job / 'stage1/args.json'),
                sha256=provenance.sha256_file(job / 'stage1/args.json'))
            receipt = json.loads((job / 'stage1/child_exit.json').read_text())
            receipt['child_pid'] += 1
            (job / 'stage1/child_exit.json').write_text(json.dumps(receipt, sort_keys=True))
        elif damage == 'receipt_pid':
            record['child_exit_receipt']['child_pid'] += 1
        elif damage == 'exit_time':
            record['child_exit_time'] = 'never'
        elif damage == 'closure_digest':
            record['source_closure_sha256'] = 'e' * 64
        else:
            record.pop('source_closure_sha256')
        (job / 'stage1/completion.json').write_text(json.dumps(record, sort_keys=True, indent=2))
        return names

    job, children, spec, repo = job_run(mutate=mutate)
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=repo,
                                children=children, expect='finetune', job_spec=spec)
    assert not (job / 'completion.json').exists()


def test_haa_children_bind_the_registry_and_head_they_ran_under(haa_eval_run, haa_repo):
    """Finding 2: the child evidence a job compares includes its execution identity."""
    run, log, args, checkpoint = haa_eval_run
    fields = exp06_finalize.finalize(run, 'haa_eval', log, 0, repo=haa_repo)
    record = json.loads((run / 'provenance.json').read_text())
    assert fields['registry_sha256'] == exp06_finalize.registry_sha256()
    assert fields['git_head'] == record['git_state']['HEAD']
    record['registry_sha256'] = 'a' * 64
    (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    (run / 'completion.json').unlink()
    with pytest.raises(ValueError, match='registry_sha256'):
        exp06_finalize.finalize(run, 'haa_eval', log, 0, repo=haa_repo)
    assert not (run / 'completion.json').exists()


def alter_per_sample(path, damage):
    """The review's reproduction: a per-sample file from another room or protocol."""
    body = json.loads(Path(path).read_text())
    if damage == 'ir_path_room':
        body['ir_path'] = ['class_room/{}'.format(index) for index in body['index']]
    elif damage == 'ir_path_index':
        body['ir_path'] = ['hallway/{}'.format(index + 1) for index in body['index']]
    elif damage == 'ir_path_missing':
        body.pop('ir_path')
    elif damage == 'index_float':
        body['index'] = [float(index) for index in body['index']]
    elif damage == 'meta_room':
        body['meta']['room'] = 'class_room'
    else:
        body['meta'][damage] = {'split': 'val', 'num_shot': 1, 'eval_seed': 777}[damage]
    Path(path).write_text(json.dumps(body))


PROTOCOL_DAMAGE = [('ir_path_room', 'ir_path'), ('ir_path_index', 'ir_path'),
                   ('ir_path_missing', 'ir_path'), ('index_float', 'index'),
                   ('meta_room', 'room'), ('split', 'split'), ('num_shot', 'num_shot'),
                   ('eval_seed', 'eval_seed')]


@pytest.mark.parametrize('damage,cause', PROTOCOL_DAMAGE)
def test_per_sample_results_are_bound_to_the_room_and_protocol(haa_eval_run, haa_repo,
                                                               damage, cause):
    """Finding 3: wrong-room or differently sampled observations must never be admitted."""
    run, log, args, checkpoint = haa_eval_run
    alter_per_sample(run / 'per_sample_hallway.json', damage)
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(run, 'haa_eval', log, 0, repo=haa_repo)
    assert not (run / 'completion.json').exists()


@pytest.mark.parametrize('damage,cause', [('ir_path_room', 'ir_path'), ('split', 'split'),
                                          ('num_shot', 'num_shot')])
def test_a_job_refuses_a_child_evaluated_under_another_protocol(job_run, closed_log_file,
                                                                damage, cause):
    """Finding 3 on the job path: the completion carries the altered file's correct hash."""
    def mutate(job, names):
        path = job / 'eval/hallway/per_sample_hallway.json'
        alter_per_sample(path, damage)
        record = json.loads((job / 'eval/hallway/completion.json').read_text())
        record['artifacts']['per_sample_hallway.json'] = provenance.sha256_file(path)
        (job / 'eval/hallway/completion.json').write_text(
            json.dumps(record, sort_keys=True, indent=2))
        return names

    job, children, spec, repo = job_run(mutate=mutate)
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=repo,
                                children=children, expect='finetune', job_spec=spec)
    assert not (job / 'completion.json').exists()


@pytest.mark.parametrize('values,summary', [
    (['oops', {}, True], {'mean': None, 'median': None, 'n': 0}),
    ([float('nan')] * 3, {'mean': 0.5, 'median': 0.5, 'n': 0})])
def test_corrupt_per_sample_values_never_count_as_missing(tmp_path, haa_repo, heading_jsons,
                                                          data_root, values, summary):
    """Finding 4: a string or a dict is not the writer's invalid measurement."""
    checkpoint = haa_repo / 'stage2_best.pth'
    torch.save(tiny_state(), checkpoint)
    args = haa_eval_args(heading_jsons, checkpoint)
    run, log = tmp_path / 'eval' / 'hallway', tmp_path / 'child.log'
    metrics = eval_metrics(args, 'hallway', dict(PER_SAMPLE), edt_error_s=summary)
    write_haa_eval(run, args, log, haa_repo, data_root, per_sample={'edt': values},
                   meta=eval_meta(args, provenance.sha256_file(checkpoint)), metrics=metrics)
    with pytest.raises(ValueError, match='edt'):
        exp06_finalize.finalize(run, 'haa_eval', log, 0, repo=haa_repo)
    assert not (run / 'completion.json').exists()
