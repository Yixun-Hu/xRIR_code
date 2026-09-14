"""Tier receipts drive the live, finalisation and elapsed-time training gates."""
import json

import pytest

from tools import exp04_launcher as launch, exp05_gates as gates
from test_exp05_gates import receipt, probe_attempt


@pytest.mark.parametrize('failure', [None, 'slow', 'missing', 'changed', 'renew'])
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
    if failure == 'renew':
        gates.set_budget(attempt.parent, dict(limits, probe_receipt_sha256='b' * 64))
    if failure == 'missing':
        del fields['mutable_inputs']['probe_receipt']
    if failure == 'changed':
        path.write_text(path.read_text() + '\n')
    monkeypatch.setattr(launch, 'resource_gate', lambda *a: dict(gpu='1', uuid='GPU-test'))
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
    if failure and failure != 'renew':
        with pytest.raises(ValueError) as error:
            launch.execute_attempt(attempt, 'full', '1', log, lambda: fields, limits=limits, runner=runner)
        if failure == 'slow':
            assert error.value.reason == 'guard_epoch_one'
            aborted = attempt.with_name(attempt.name + '_ABORTED_slow')
            assert json.loads((aborted / 'abort.json').read_text())['reason'] == 'guard_epoch_one'
            assert log.with_name('train_ABORTED_slow.log').exists()
        assert not (attempt / 'completion.json').exists()
    else:
        result = launch.execute_attempt(attempt, 'full', '1', log, lambda: fields, limits=limits, runner=runner,
            renew_ceiling='2026-09-12T23:39:54-04:00: new clean probe' if failure == 'renew' else None)
        assert result['metrics']['test_loss'] == .5
        ledger = launch.hours_record(attempt.parent)
        assert ledger['ceiling_hours'] == 1.5 * data['T_run'] / 3600
        assert ledger['probe_receipt_sha256'] == binding['sha256']
        assert len(ledger.get('renewals', [])) == (1 if failure == 'renew' else 0)
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
@pytest.mark.parametrize('renewal', [None, '2026-09-12T23:39:54-04:00: slower probe'])
def test_full_cli_uses_receipt_projection(tmp_path, monkeypatch, receipt, tier, backbone, renewal):
    path, data = receipt
    data.update(tier=tier, backbone=backbone)
    data['probe_attempt'] = probe_attempt(launch.arm_root(tier, backbone) / '_probe_new_arm', data)
    path.write_text(json.dumps(data))
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    monkeypatch.setattr(launch.subprocess, 'check_output', lambda *a, **k: 'a' * 40 + '\n')
    monkeypatch.setattr(launch, 'check_golden', lambda *a: None)
    monkeypatch.setattr(launch, 'build_fields', lambda *a: dict(mutable_inputs={}))
    calls = []
    def execute(attempt, mode, gpu, log, factory, **kwargs):
        assert kwargs['renew_ceiling'] == renewal
        assert log == tmp_path / ('worklog/worklog_yixun/exp_05_param_efficiency_claude/'
            'param_efficiency_test_train_{}_{}_full.log'.format(tier, backbone))
        assert factory()['mutable_inputs']['probe_receipt']['sha256'] == launch.p.sha256_file(path)
        assert kwargs['projection'] == data['T_run'] / 3600
        assert kwargs['limits']['ceiling_hours'] == 1.5 * data['T_run'] / 3600
        assert kwargs['limits']['epoch_seconds'] == 1.05 * data['T_epoch']
        calls.append(attempt)
        return {'passed': True}
    monkeypatch.setattr(launch, 'execute_attempt', execute)
    launch.main(['full', '--tier', tier, '--backbone', backbone, '--gpu', '1', '--reviewed-commit', 'HEAD',
                 '--probe-json', str(path), '--projection-hours', '0.1', '--timestamp', 'test'] +
                (['--renew-ceiling', renewal] if renewal else []))
    assert len(calls) == 1 and calls[0].parent == tmp_path / 'ckpt/exp05' / (tier + '_' + backbone)


@pytest.mark.parametrize('fault', ['resource', 'resource_without_renewal', 'unclean', 'stale'])
def test_refused_launch_keeps_receipt_history(receipt, tmp_path, monkeypatch, fault):
    path, data = receipt
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    monkeypatch.setattr(launch.subprocess, 'check_output', lambda *a, **k: 'a' * 40)
    monkeypatch.setattr(launch, 'check_golden', lambda *a: None)
    root = launch.arm_root('S', 'simple')
    gates.set_budget(root, dict(projection_hours=20., ceiling_hours=30., probe_receipt_sha256='b' * 64))
    if not fault.startswith('resource'):
        data.update({'PROBE_NOT_CLEAN': True} if fault == 'unclean' else {'reviewed_commit': 'c' * 40})
        path.write_text(json.dumps(data))
    def refuse(*a):
        raise ValueError('resource refused')
    monkeypatch.setattr(launch, 'resource_gate', refuse)
    with pytest.raises(ValueError, match='resource|receipt'):
        launch.main(['full', '--tier', 'S', '--reviewed-commit', 'HEAD', '--probe-json', str(path)] +
                    ([] if fault == 'resource_without_renewal' else
                     ['--renew-ceiling', '2026-09-12T23:39:54-04:00: new probe']))
    record = launch.hours_record(root)
    assert record['probe_receipt_sha256'] == 'b' * 64
    assert not record.get('probe_receipt_history') and not record.get('renewals')


@pytest.mark.parametrize('fault', ['slow', 'ceiling', 'reuse', 'historical'])
def test_ledger_refusal_creates_no_attempt(receipt, tmp_path, monkeypatch, fault):
    path, data = receipt
    root = launch.arm_root('S', 'simple')
    limits = dict(projection_hours=data['T_run'] / 3600, ceiling_hours=1.5 * data['T_run'] / 3600,
                  probe_receipt_sha256=launch.p.sha256_file(path))
    gates.set_budget(root, limits)
    if fault == 'historical':
        gates.set_budget(root, dict(limits, probe_receipt_sha256='b' * 64))
    if fault in ('slow', 'ceiling'):
        old = launch.create_attempt(root, 'attempt_old_ABORTED_' + fault)
        (old / 'abort.json').write_text(json.dumps(dict(reason='guard_epoch_one')))
        (old / 'train_manifest.json').write_text(json.dumps(dict(mutable_inputs=dict(probe_receipt=dict(
            sha256=limits['probe_receipt_sha256'])))))
        launch.account_hours(old, 1. if fault == 'slow' else limits['projection_hours'], mode='full')
    before = (root / 'cumulative_hours.json').read_bytes(), sorted(root.glob('attempt_*'))
    monkeypatch.setattr(launch.subprocess, 'check_output', lambda *a, **k: 'a' * 40)
    monkeypatch.setattr(launch, 'check_golden', lambda *a: None)
    monkeypatch.setattr(launch, 'resource_gate', lambda *a: pytest.fail('resource gate before ledger refusal'))
    with pytest.raises(ValueError, match='slow|ceiling|new clean receipt'):
        launch.main(['full', '--tier', 'S', '--reviewed-commit', 'HEAD', '--probe-json', str(path)] +
                    (['--renew-ceiling', '2026-09-13T00:05:00-04:00: reason'] if fault in ('reuse', 'historical') else []))
    assert before == ((root / 'cumulative_hours.json').read_bytes(), sorted(root.glob('attempt_*')))


@pytest.mark.parametrize('tier', ['M', 'S', 'L'])
def test_probe_prefix_must_match_tier(tier):
    expected = launch.effective_args(launch.command('full', 'attempt', tier), '1', 9261)
    with pytest.raises(ValueError, match='prefix'):
        launch.LogGuard(expected, 'probe').feed(('EXP05_' if tier == 'M' else 'EXP04_') + 'PROBE_RESULT {}')
