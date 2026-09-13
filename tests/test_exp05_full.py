"""Tier receipts drive the live, finalisation and elapsed-time training gates."""
import json

import pytest

from tools import exp04_launcher as launch, exp05_gates as gates
from test_exp05_gates import receipt


@pytest.mark.parametrize('failure', [None, 'slow', 'missing', 'changed'])
def test_full_receipt_deadline_and_abort(tmp_path, monkeypatch, receipt, failure):
    path, data = receipt
    attempt, log = tmp_path / 'arm/attempt_test', tmp_path / 'train.log'
    command = launch.command('full', str(attempt), 'S', 'simple')
    expected = launch.effective_args(command, '1', 9261)
    binding = gates.validate_receipt(path, 'a' * 40, '1', 'S', 'simple')
    fields = dict(repo=str(tmp_path), reviewed_commit='a' * 40, effective_args=expected,
                  command=command, source_closures={}, train_data_identity=launch.p._inventory([], tmp_path),
                  mutable_inputs=dict(probe_receipt=binding, control_args=dict(path=__file__, sha256=launch.p.sha256_file(__file__))))
    limits = gates.timing_limits(fields, '1')
    if failure == 'missing':
        del fields['mutable_inputs']['probe_receipt']
    if failure == 'changed':
        path.write_text(path.read_text() + '\n')
    monkeypatch.setattr(launch, 'resource_gate', lambda *a: {})
    monkeypatch.setattr(launch.time, 'monotonic', lambda: 100.)
    def runner(argv, target, gpu, guard, deadline):
        assert deadline == 100. + limits['ceiling_hours'] * 3600
        assert guard.epoch_limit_seconds == limits['epoch_seconds']
        (attempt / 'args.json').write_text(json.dumps(expected))
        duration = limits['epoch_seconds'] / 60 + (0.01 if failure == 'slow' else 0)
        history = [dict(epoch=i, train_loss=1., test_loss=.5, epoch_minutes=duration) for i in range(1, 13)]
        (attempt / 'history.jsonl').write_text('\n'.join(map(json.dumps, history)))
        for name in ['best.pth', 'last.pth'] + ['epoch_%03d.pth' % i for i in range(1, 13)]:
            (attempt / name).write_bytes(b'checkpoint')
        target.write_text('\n'.join(['XRIR_RUNTIME_ARGS ' + json.dumps(expected), 'yaw_aug DISABLED',
            'Train Epoch: 1 [0/9261] loss 1.0', 'Test set (epoch 1): Average loss: 0.5 over 199 batches',
            'epoch 1 done']) + '\n')
        guard.log_created = True
        guard.poll(target)
        return 0
    if failure:
        with pytest.raises(ValueError) as error:
            launch.execute_attempt(attempt, 'full', '1', log, lambda: fields, limits=limits, runner=runner)
        if failure == 'slow':
            assert error.value.reason == 'guard_epoch_one'
            aborted = attempt.with_name(attempt.name + '_ABORTED_slow')
            assert json.loads((aborted / 'abort.json').read_text())['reason'] == 'guard_epoch_one'
            assert log.with_name('train_ABORTED_slow.log').exists()
        assert not (attempt / 'completion.json').exists()
    else:
        result = launch.execute_attempt(attempt, 'full', '1', log, lambda: fields, limits=limits, runner=runner)
        assert result['metrics']['test_loss'] == .5
        ledger = launch.hours_record(attempt.parent)
        assert ledger['ceiling_hours'] == 1.5 * data['T_run'] / 3600
        assert ledger['probe_receipt_sha256'] == binding['sha256']
        assert json.loads((attempt / 'train_manifest.json').read_text())['mutable_inputs']['probe_receipt'] == binding


def test_live_epoch_limit_and_saved_output_limit(tmp_path, monkeypatch):
    expected = launch.effective_args(launch.command('full', str(tmp_path), 'L', 'cylindrical'), '1', 9261)
    limits = dict(epoch_seconds=100.)
    guard = launch.LogGuard(expected, 'full', tmp_path, limits=limits)
    guard.first_step_time = 1.
    log = tmp_path / 'train.log'
    log.write_text('')
    monkeypatch.setattr(launch.time, 'monotonic', lambda: 101.001)
    with pytest.raises(launch.LauncherFailure, match='epoch one') as error:
        guard.poll(log)
    assert error.value.reason == 'guard_epoch_one'
    (tmp_path / 'args.json').write_text(json.dumps(expected))
    (tmp_path / 'history.jsonl').write_text('\n'.join(json.dumps(dict(epoch=i, train_loss=1., test_loss=.5,
        epoch_minutes=100.001/60)) for i in range(1, 13)))
    with pytest.raises(launch.LauncherFailure) as error:
        launch.validate_outputs(tmp_path, 'full', expected, limits=limits)
    assert error.value.reason == 'guard_epoch_one'


@pytest.mark.parametrize('tier,backbone', [(t, b) for t in ('S', 'L') for b in ('simple', 'cylindrical')])
def test_full_cli_uses_receipt_projection(tmp_path, monkeypatch, receipt, tier, backbone):
    path, data = receipt
    data.update(tier=tier, backbone=backbone)
    path.write_text(json.dumps(data))
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    monkeypatch.setattr(launch.subprocess, 'check_output', lambda *a, **k: 'a' * 40 + '\n')
    monkeypatch.setattr(launch, 'check_golden', lambda *a: None)
    monkeypatch.setattr(launch, 'build_fields', lambda *a: dict(mutable_inputs={}))
    calls = []
    def execute(attempt, mode, gpu, log, factory, **kwargs):
        assert factory()['mutable_inputs']['probe_receipt']['sha256'] == launch.p.sha256_file(path)
        assert kwargs['projection'] == data['T_run'] / 3600
        assert kwargs['limits']['ceiling_hours'] == 1.5 * data['T_run'] / 3600
        assert kwargs['limits']['epoch_seconds'] == 1.05 * data['T_epoch']
        calls.append(attempt)
        return {'passed': True}
    monkeypatch.setattr(launch, 'execute_attempt', execute)
    launch.main(['full', '--tier', tier, '--backbone', backbone, '--gpu', '1', '--reviewed-commit', 'HEAD',
                 '--probe-json', str(path), '--projection-hours', '0.1', '--log-dir', str(tmp_path / 'logs')])
    assert len(calls) == 1 and calls[0].parent == tmp_path / 'ckpt/exp05' / (tier + '_' + backbone)
