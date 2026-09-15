"""Completion evidence by run type: what the finalizer accepts and what it refuses."""
import copy
import functools
import json
from pathlib import Path

import pytest
import torch

from tools import exp06_finalize, exp06_recipe, exp06_train, provenance

REPO = Path(__file__).resolve().parents[1]
MARKER = 'EXP06_CHILD_EXIT 0 2026-09-15T04:05:06.070809+00:00'
STATE = {'source_network.weight': torch.arange(6.).reshape(2, 3), 'head.bias': torch.zeros(2)}


@functools.lru_cache(maxsize=None)
def provenance_record():
    return exp06_train.provenance_fields(['--backbone', 'cylindrical_oriented'], 'full')


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
def full_run(tmp_path):
    """A complete twelve-epoch attempt directory with tiny tensors and a closed log."""
    run = tmp_path / 'attempt_20260916T130000'
    run.mkdir()
    record = copy.deepcopy(provenance_record())
    args = full_args()
    record['effective_args'] = args
    (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    (run / 'args.json').write_text(json.dumps(args, indent=2))
    (run / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history_rows()))
    torch.save({'model': STATE, 'optimizer': {}, 'scheduler': {}, 'epoch': 12, 'batch_idx': 0,
                'best_test_loss': 0.1, 'args': args}, run / 'last.pth')
    torch.save(STATE, run / 'epoch_012.pth')
    log = tmp_path / 'oriented_cyl_train_full.log'
    log.write_text('train samples: 296334  test samples: 6337\n' + MARKER + '\n')
    return run, log


def test_full_completion_records_every_artifact(full_run):
    run, log = full_run
    fields = exp06_finalize.finalize(run, 'full', log, 0, repo=REPO)
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
    assert completion['source_closure_sha256'] == provenance_record()['source_closures']['training']['sha256']


def test_finalize_is_idempotent_and_refuses_a_different_completion(full_run):
    run, log = full_run
    exp06_finalize.finalize(run, 'full', log, 0, repo=REPO)
    first = (run / 'completion.json').read_bytes()
    exp06_finalize.finalize(run, 'full', log, 0, repo=REPO)
    assert (run / 'completion.json').read_bytes() == first
    (run / 'completion.json').write_text('{"run_type": "full"}\n')
    with pytest.raises(ValueError, match='completion'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=REPO)


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
def test_full_refusals_are_named_and_write_nothing(full_run, damage, cause):
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
        exp06_finalize.finalize(run, 'full', log, 0, repo=REPO)
    assert not (run / 'completion.json').exists()


@pytest.mark.parametrize('line,exit_code', [
    (MARKER, 1), ('EXP06_CHILD_EXIT 1 2026-09-15T04:05:06+00:00', 0),
    ('no marker at all', 0), ('EXP06_CHILD_EXIT 0', 0),
    ('EXP06_CHILD_EXIT 0 2026-09-15T04:05:06+00:00 extra', 0),
    ('EXP06_CHILD_EXIT 0 2026-09-15T04:05:06', 0), ('', 0)])
def test_open_or_disagreeing_logs_are_refused(full_run, line, exit_code):
    run, log = full_run
    log.write_text('train samples: 1\n' + line + ('\n' if line else ''))
    with pytest.raises(ValueError):
        exp06_finalize.finalize(run, 'full', log, exit_code, repo=REPO)
    assert not (run / 'completion.json').exists()


def test_non_zero_child_exit_is_refused_for_an_arm(full_run):
    run, log = full_run
    log.write_text('EXP06_CHILD_EXIT 7 2026-09-15T04:05:06+00:00\n')
    with pytest.raises(ValueError, match='status'):
        exp06_finalize.finalize(run, 'full', log, 7, repo=REPO)
    assert not (run / 'completion.json').exists()


@pytest.mark.parametrize('run_type', ['smoke', 'probe'])
def test_diagnostic_runs_need_no_artifacts(tmp_path, run_type):
    run = tmp_path / run_type
    run.mkdir()
    log = tmp_path / 'smoke.log'
    log.write_text('EXP06_SMOKE {"wall_s": 12.5}\nEXP06_CHILD_EXIT 3 2026-09-15T04:05:06+00:00\n')
    receipt = tmp_path / 'probe_20260915T040506.json'
    receipt.write_text(json.dumps({'diagnostic': True, 'entry': 'exp06_train'}))
    fields = exp06_finalize.finalize(run, run_type, log, 3, repo=REPO, receipt=receipt)
    assert fields['diagnostic'] is True and fields['admissible_arm'] is False
    assert fields['artifacts'] == {} and fields['child_exit'] == 3
    assert fields['receipt'] == {'path': str(receipt.resolve()),
                                 'sha256': provenance.sha256_file(receipt)}
    assert json.loads((run / 'completion.json').read_text()) == fields
    assert exp06_finalize.finalize(run, run_type, log, 3, repo=REPO, receipt=receipt) == fields


def test_unknown_run_type_and_missing_directory_are_refused(tmp_path, full_run):
    run, log = full_run
    with pytest.raises(ValueError, match='run type'):
        exp06_finalize.finalize(run, 'invented', log, 0, repo=REPO)
    with pytest.raises(ValueError, match='directory'):
        exp06_finalize.finalize(tmp_path / 'absent', 'full', log, 0, repo=REPO)
