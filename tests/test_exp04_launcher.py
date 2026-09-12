"""CPU guards for the exp_04 training launcher."""
import json
import os
from pathlib import Path
import pytest
from tools import exp04_launcher as launch


@pytest.mark.parametrize('mode', ['smoke', 'full'])
def test_golden_argv_and_one_flag_refusal(mode, tmp_path):
    command = launch.command(mode, '<attempt>')
    launch.check_golden(command, mode, '<attempt>')
    command[command.index('--lr') + 1] = '0.002'
    with pytest.raises(ValueError, match='golden'):
        launch.check_golden(command, mode, '<attempt>')


def test_normalized_runtime_schema_and_mismatch():
    args = launch.effective_args(launch.command('full', 'ckpt/attempt'), '1', 9261)
    assert args['resume'] is None and args['no_save'] is False
    assert args['yaw_aug_seed'] == 0 and args['train_batches_per_epoch'] == 9261
    assert args['CUDA_VISIBLE_DEVICES'] == '1'
    runtime = {k: v for k, v in args.items() if k not in launch.ENV_KEYS}
    runtime['env'] = {k: args[k] for k in launch.ENV_KEYS}
    assert launch.normalize(runtime) == args
    launch.check_runtime(runtime, args, 'full')
    for key, value in [('lr', '0.001'), ('yaw_aug', True), ('train_batches_per_epoch', 9260)]:
        changed = dict(runtime, **{key: value})
        with pytest.raises(ValueError, match=key):
            launch.check_runtime(changed, args, 'full')
    runtime['unexpected'] = 1
    with pytest.raises(ValueError, match='unexpected'):
        launch.check_runtime(runtime, args, 'full')


def test_full_refuses_matching_but_wrong_bpe():
    args = launch.effective_args(launch.command('full', 'ckpt/attempt'), '1', 9260)
    with pytest.raises(ValueError, match='9261'):
        launch.check_runtime(args, args, 'full')


@pytest.mark.parametrize('free,apps,disk,allow,mode,refused', [
    (40, '', 50, False, 'full', False), (39.99, '', 50, False, 'full', True),
    (40, '123, train.py, 11000', 50, False, 'smoke', True),
    (30, '123, train.py, 11000', 50, True, 'smoke', False),
    (30, '123, train.py, 11000', 50, True, 'probe', False),
    (40, '', 50, True, 'full', True), (40, '', 49.99, True, 'smoke', True)])
def test_resource_thresholds(monkeypatch, tmp_path, free, apps, disk, allow, mode, refused):
    def output(argv, **kwargs):
        if argv[0] == 'df':
            return 'Avail\n' + str(int(disk * 2**30))
        if '--query-compute-apps=pid,process_name,used_gpu_memory' in argv:
            return apps
        assert '-i' in argv and argv[argv.index('-i') + 1] == '1'
        return 'GPU-test, {}, 91'.format(free * 1024)
    monkeypatch.setattr(launch.subprocess, 'check_output', output)
    if refused:
        with pytest.raises(ValueError):
            launch.resource_gate('1', tmp_path, mode, allow)
    else:
        state = launch.resource_gate('1', tmp_path, mode, allow)
        assert state['compute_apps'] == apps and state['utilization_gpu'] == 91


def test_exclusive_attempt_abort_account_and_promote(tmp_path):
    attempt = launch.create_attempt(tmp_path, 'attempt_t')
    with pytest.raises(FileExistsError):
        launch.create_attempt(tmp_path, 'attempt_t')
    (attempt / 'train_manifest.json').write_text('{}')
    aborted = launch.abort_attempt(attempt, 'child_failed', 2.5)
    assert aborted.name == 'attempt_t_ABORTED_child_failed'
    assert (aborted / 'train_manifest.json').exists()
    assert not attempt.exists()
    record = json.loads((tmp_path / 'cumulative_hours.json').read_text())
    assert record['total_hours'] == 2.5 and record['attempts'][0]['hours'] == 2.5
    launch.check_budget(tmp_path, 40.5)
    with pytest.raises(ValueError, match='43'):
        launch.check_budget(tmp_path, 40.5001)
    complete = launch.create_attempt(tmp_path, 'attempt_done')
    (complete / 'completion.json').write_text('{}')
    launch.promote(complete)
    assert os.readlink(tmp_path / 'final') == 'attempt_done'
    assert (tmp_path / 'final').resolve() == complete


@pytest.mark.parametrize('projection', [-1, float('nan'), float('inf')])
def test_invalid_budget_projection(tmp_path, projection):
    with pytest.raises(ValueError):
        launch.check_budget(tmp_path, projection)
