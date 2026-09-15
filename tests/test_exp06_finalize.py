"""Completion evidence by run type: what the finalizer accepts and what it refuses."""
import copy
import functools
import json
import os
import subprocess
import sys
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


def haa_train_args(frame='heading', **overrides):
    args = dict(backbone='cylindrical_oriented', rooms=['class_room', 'hallway', 'complex_room'],
                save_dir='ckpt/exp06/sim2real/cyl_or/seed0/stage1', frame=frame, seed=0,
                epochs=1000, val_every=10, lr=1e-4, weight_decay=1e-4, eval_seed=0,
                init='ckpt/exp06/pretrain/final/epoch_012.pth', init_sha256='e' * 64,
                heading={room: {'phi_deg': -90.0, 'k': 128, 'sha256': 'f' * 64}
                         for room in ('class_room', 'hallway', 'complex_room')})
    args.update(overrides)
    return args


def write_haa_train(run, args):
    run.mkdir(parents=True, exist_ok=True)
    (run / 'args.json').write_text(json.dumps(args))
    (run / 'history.jsonl').write_text(json.dumps({'epoch': 10, 'val_loss': 0.5}) + '\n')
    (run / 'summary.json').write_text(json.dumps({'best_val_loss': 0.5, 'best_epoch': 10}))
    for name in ('best.pth', 'last.pth'):
        torch.save(STATE, run / name)
    return run


def write_haa_eval(run, args, meta):
    run.mkdir(parents=True, exist_ok=True)
    (run / 'args.json').write_text(json.dumps(args))
    room = args['rooms'][0]
    (run / 'metrics_{}.json'.format(room)).write_text(json.dumps({'edt': 0.05, 'c50': 1.1}))
    (run / 'per_sample_{}.json'.format(room)).write_text(json.dumps({'meta': meta, 'edt': [0.05]}))
    return run


@pytest.fixture
def closed_log_file(tmp_path):
    log = tmp_path / 'child.log'
    log.write_text('stage output\n' + MARKER + '\n')
    return log


def test_haa_train_completion_binds_heading_and_artifacts(tmp_path, closed_log_file):
    run = write_haa_train(tmp_path / 'stage1', haa_train_args())
    fields = exp06_finalize.finalize(run, 'haa_train', closed_log_file, 0, repo=REPO)
    assert set(fields['artifacts']) == {'args.json', 'history.jsonl', 'summary.json',
                                        'best.pth', 'last.pth'}
    assert fields['frame'] == 'heading' and fields['backbone'] == 'cylindrical_oriented'
    assert fields['heading']['hallway']['k'] == 128 and fields['init_sha256'] == 'e' * 64
    assert fields['diagnostic'] is False and fields['admissible_arm'] is True
    assert json.loads((run / 'completion.json').read_text()) == fields


def test_haa_train_in_the_room_frame_needs_no_heading(tmp_path, closed_log_file):
    args = haa_train_args(frame='room')
    args.pop('heading')
    args.pop('init_sha256')
    run = write_haa_train(tmp_path / 'stage1_room', args)
    fields = exp06_finalize.finalize(run, 'haa_train', closed_log_file, 0, repo=REPO)
    assert fields['frame'] == 'room' and fields['heading'] is None


@pytest.mark.parametrize('damage,cause', [
    ('frame', 'frame'), ('no_heading', 'heading'), ('partial_heading', 'heading'),
    ('bad_k', 'heading'), ('bad_sha', 'heading'), ('no_init', 'init_sha256'),
    ('summary', 'summary.json'), ('best', 'best.pth'), ('rooms', 'rooms')])
def test_haa_train_refusals_are_named(tmp_path, closed_log_file, damage, cause):
    args = haa_train_args()
    if damage == 'frame':
        args['frame'] = 'world'
    elif damage == 'no_heading':
        args.pop('heading')
    elif damage == 'partial_heading':
        args['heading'].pop('hallway')
    elif damage == 'bad_k':
        args['heading']['hallway']['k'] = 512
    elif damage == 'bad_sha':
        args['heading']['hallway']['sha256'] = 'zz'
    elif damage == 'no_init':
        args.pop('init_sha256')
    elif damage == 'rooms':
        args['rooms'] = []
    run = write_haa_train(tmp_path / 'stage1', args)
    if damage in ('summary', 'best'):
        (run / {'summary': 'summary.json', 'best': 'best.pth'}[damage]).unlink()
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(run, 'haa_train', closed_log_file, 0, repo=REPO)
    assert not (run / 'completion.json').exists()


def test_haa_eval_completion_records_the_single_room(tmp_path, closed_log_file):
    args = dict(backbone='cylindrical_oriented', rooms=['hallway'], frame='heading',
                checkpoint='ckpt/exp06/sim2real/cyl_or/seed0/stage2_hallway/best.pth',
                eval_seed=0, split='test',
                heading={'hallway': {'phi_deg': -90.0, 'k': 128, 'sha256': 'f' * 64}})
    meta = {'backbone': 'cylindrical_oriented', 'checkpoint_sha256': 'd' * 64,
            'frame': 'heading', 'heading': {'hallway': {'k': 128}}}
    run = write_haa_eval(tmp_path / 'eval' / 'hallway', args, meta)
    fields = exp06_finalize.finalize(run, 'haa_eval', closed_log_file, 0, repo=REPO)
    assert fields['room'] == 'hallway' and fields['frame'] == 'heading'
    assert set(fields['artifacts']) == {'args.json', 'metrics_hallway.json', 'per_sample_hallway.json'}
    assert fields['checkpoint_sha256'] == 'd' * 64


@pytest.mark.parametrize('damage,cause', [
    ('two_rooms', 'room'), ('no_metrics', 'metrics_hallway.json'),
    ('no_per_sample', 'per_sample_hallway.json'), ('no_meta_heading', 'heading'),
    ('no_meta_backbone', 'meta'), ('meta_backbone_differs', 'backbone')])
def test_haa_eval_refusals_are_named(tmp_path, closed_log_file, damage, cause):
    args = dict(backbone='cylindrical_oriented', rooms=['hallway'], frame='heading',
                checkpoint='best.pth', eval_seed=0, split='test',
                heading={'hallway': {'phi_deg': -90.0, 'k': 128, 'sha256': 'f' * 64}})
    meta = {'backbone': 'cylindrical_oriented', 'checkpoint_sha256': 'd' * 64,
            'frame': 'heading', 'heading': {'hallway': {'k': 128}}}
    if damage == 'two_rooms':
        args['rooms'] = ['hallway', 'class_room']
        args['heading']['class_room'] = {'phi_deg': -90.0, 'k': 128, 'sha256': 'f' * 64}
    elif damage == 'no_meta_heading':
        meta.pop('heading')
    elif damage == 'no_meta_backbone':
        meta.pop('backbone')
    elif damage == 'meta_backbone_differs':
        meta['backbone'] = 'cylindrical'
    run = write_haa_eval(tmp_path / 'eval' / 'hallway', args, meta)
    if damage == 'no_metrics':
        (run / 'metrics_hallway.json').unlink()
    elif damage == 'no_per_sample':
        (run / 'per_sample_hallway.json').unlink()
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(run, 'haa_eval', closed_log_file, 0, repo=REPO)
    assert not (run / 'completion.json').exists()


from sim_to_real.haa_dataset import ROOMS


def write_job(tmp_path, expect='finetune', children=None):
    job = tmp_path / 'seed0'
    names = children if children is not None else exp06_finalize.expected_children(expect)
    for name in names:
        child = job / name
        child.mkdir(parents=True)
        (child / 'completion.json').write_text(json.dumps(
            {'run_type': 'haa_eval' if name.startswith('eval/') else 'haa_train',
             'admissible_arm': True, 'run_dir': str(child)}, sort_keys=True))
    job.mkdir(parents=True, exist_ok=True)
    return job, [str(job / name) for name in names]


def test_job_completion_requires_every_child(tmp_path, closed_log_file):
    job, children = write_job(tmp_path)
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
    job, children = write_job(tmp_path, expect, names)
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
    job, children = write_job(tmp_path)
    with pytest.raises(ValueError, match='expect'):
        exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=REPO, children=children)


def test_cli_reports_refusals_with_status_two(full_run):
    run, log = full_run
    command = [sys.executable, 'tools/exp06_finalize.py', '--run-dir', str(run),
               '--run-type', 'full', '--log', str(log), '--child-exit', '0', '--repo', str(REPO)]
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
def test_malformed_inputs_raise_named_refusals(full_run, damage, cause):
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
        exp06_finalize.finalize(run, 'full', log, 0, repo=REPO)
    assert not (run / 'completion.json').exists()


def test_cli_exits_two_on_a_malformed_checkpoint(full_run):
    """The documented refusal interface: exit 2 and a named cause, nothing written."""
    run, log = full_run
    torch.save(torch.zeros(3), run / 'last.pth')
    command = [sys.executable, 'tools/exp06_finalize.py', '--run-dir', str(run),
               '--run-type', 'full', '--log', str(log), '--child-exit', '0', '--repo', str(REPO)]
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


def test_three_argument_sources_must_agree(full_run):
    """Blocker 2: startup, retained and checkpoint arguments are compared type-strictly."""
    run, log = full_run
    rewrite_sources(run)
    assert exp06_finalize.finalize(run, 'full', log, 0, repo=REPO)['admissible_arm'] is True


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
def test_argument_source_refusals_are_named(full_run, damage, cause):
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
        exp06_finalize.finalize(run, 'full', log, 0, repo=REPO)
    assert not (run / 'completion.json').exists()
