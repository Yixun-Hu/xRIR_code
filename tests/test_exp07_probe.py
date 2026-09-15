"""exp_07 seen fit-probe: tier-M receipts with the seen batch count and the yaw arm.

Timings are mocked and the child is stubbed, exactly as tests/test_exp05_probe*.py do;
no GPU and no trainer are involved.
"""
import json

import pytest

from tools import exp04_launcher as launch, exp05_gates as gates, exp05_probe as probe
from tools import provenance as p
from tools.exp05_params import TIERS
from test_exp05_probe import measurement

ARMS = [('seen_simple', 'simple', 0), ('seen_cyl', 'cylindrical', 0), ('seen_aug', 'simple', 1)]
SEEN_BATCHES = 9265


def seen_measurement(backbone='simple', yaw=1):
    measured = dict(measurement(), tier='M', backbone=backbone, protocol='seen', yaw_aug=yaw,
                    train_batches_per_epoch=SEEN_BATCHES, test_timing_protocol='full_test_loader',
                    test_batches_timed=195, test_batches_total=195, peak_allocated_bytes=123,
                    peak_reserved_bytes=456, train_loss=1., iteration_seconds=[1.] * 50,
                    mean_iteration_seconds=1., median_iteration_seconds=1., min_iteration_seconds=1.)
    measured.update(probe.projection(measured))
    return measured


def probe_attempt(path, data):
    """Write the attempt files the receipt binds (exp_05 round 4 rules, seen fields)."""
    path.mkdir(parents=True, exist_ok=True)
    manifest = dict(mode='probe', reviewed_commit=data['reviewed_commit'], attempt_path=str(path),
                    effective_args={key: data[key] for key in ('tier', 'backbone', 'protocol', 'yaw_aug')})
    digest = p.write_manifest(path / 'train_manifest.json', manifest)
    completion = dict(train_manifest_sha256=digest, metrics=dict(probe=data))
    return dict(path=str(path), train_manifest_sha256=digest,
                completion_sha256=p.write_manifest(path / 'completion.json', completion))


@pytest.mark.parametrize('arm,backbone,yaw', ARMS)
def test_probe_recipe_per_seen_arm(arm, backbone, yaw):
    cmd = probe.trainer_command('M', backbone, 'scratch', 'seen', yaw)
    fields = {'backbone': backbone, 'save-dir': 'scratch', 'protocol': 'seen', 'batch-size': 32,
              'accum-steps': 2, 'max-train-batches': 60, 'max-test-batches': 0, 'num-workers': 12,
              'save-every': 0, 'epoch-ckpt-every': 0}
    assert '--no-save' in cmd and '--tf32' in cmd and not any(f.startswith('--vit-') for f in cmd)
    assert all(cmd[cmd.index('--' + key) + 1] == str(value) for key, value in fields.items())
    assert (cmd[cmd.index('--yaw-aug') + 1:cmd.index('--yaw-aug') + 2] == ['1']) if yaw else (
        not any(flag.startswith('--yaw-') for flag in cmd))
    with pytest.raises(ValueError, match='tier probe'):
        probe.trainer_command('S', backbone, 'scratch', 'seen', yaw)
    with pytest.raises(ValueError, match='tier probe'):
        probe.trainer_command('M', backbone, 'scratch')


def test_projection_uses_the_protocol_batch_count():
    assert probe.TRAIN_BATCHES == launch.TRAIN_BATCHES  # kept in lockstep with the launcher
    seen = dict(measurement(), protocol='seen', train_batches_per_epoch=SEEN_BATCHES)
    assert probe.projection(seen) == dict(T_epoch=9270., T_run=111240., passed=True)
    assert probe.projection(measurement()) == dict(T_epoch=9266., T_run=111192., passed=True)
    for wrong in (dict(seen, train_batches_per_epoch=9261), dict(measurement(), protocol='seen'),
                  dict(seen, protocol='held-out')):
        with pytest.raises(ValueError, match='timing|recipe'):
            probe.projection(wrong)


@pytest.mark.parametrize('arm,backbone,yaw', ARMS)
def test_seen_probe_writes_a_bound_receipt(tmp_path, monkeypatch, arm, backbone, yaw):
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    monkeypatch.setattr(launch.subprocess, 'check_output', lambda *a, **k: 'a' * 40 + '\n')
    snapshot = dict(gpu='1', uuid='GPU-one', compute_apps='', free_gib=45.)
    monkeypatch.setattr(launch, 'gpu_snapshot', lambda gpu: snapshot)
    monkeypatch.setattr(launch, 'build_fields', lambda cmd, *a: dict(command=cmd))
    measured = seen_measurement(backbone, yaw)
    calls = []
    def execute(attempt, mode, gpu, log, factory, **kwargs):
        cmd = factory()['command']
        assert cmd[1:3] == ['-m', 'tools.exp05_probe'] and cmd[cmd.index('--tier') + 1] == 'M'
        assert cmd[cmd.index('--protocol') + 1] == 'seen'
        assert ('--yaw-aug' in cmd) is bool(yaw)
        assert factory()['trainer_command'][1:] == probe.trainer_command(
            'M', backbone, str(attempt.relative_to(tmp_path)), 'seen', yaw)
        assert log == tmp_path / 'logs' / 'seen_protocol_test_train_{}_probe.log'.format(arm)
        calls.append(attempt)
        probe_attempt(attempt, dict(measured, reviewed_commit='a' * 40))
        return dict(metrics=dict(probe=measured), resource_before=snapshot)
    monkeypatch.setattr(launch, 'execute_attempt', execute)
    launch.main(['probe', '--protocol', 'seen', '--backbone', backbone, '--reviewed-commit', 'HEAD',
                 '--timestamp', 'test', '--log-dir', str(tmp_path / 'logs')] +
                (['--yaw-aug', '1'] if yaw else []))
    root = tmp_path / 'ckpt/exp07' / arm
    path = root / ('_probe_test_' + arm + '.json')
    receipt = json.loads(path.read_text())
    assert len(calls) == 1 and calls[0].parent == root
    assert (receipt['tier'], receipt['protocol'], receipt['yaw_aug']) == ('M', 'seen', yaw)
    assert receipt['schema_version'] == 1 and receipt['gpu'] == '1' and receipt['passed'] is True
    assert receipt['PROBE_NOT_CLEAN'] is False and receipt['arms_before'] == [snapshot]
    assert receipt['T_epoch'] == measured['T_epoch'] and receipt['T_run'] == measured['T_run']
    assert receipt['probe_attempt']['path'] == str(calls[0])
    assert receipt['live_epoch_limit_seconds'] == 1.05 * receipt['T_epoch']
    assert gates.validate_receipt(path, 'a' * 40, '1', 'M', backbone, 'seen', yaw)['sha256']


@pytest.fixture
def seen_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    state = dict(gpu='1', uuid='GPU-test', compute_apps='', free_gib=45.)
    data = dict(seen_measurement(), schema_version=1, reviewed_commit='a' * 40, gpu='1',
                before=state, after=dict(state), arms_before=[dict(state)], PROBE_NOT_CLEAN=False)
    data['probe_attempt'] = probe_attempt(launch.arm_root('M', 'simple', 'seen', 1) / '_probe_t_arm', data)
    path = tmp_path / 'probe.json'
    path.write_text(json.dumps(data))
    return path, data


def test_timing_limits_for_a_seen_arm(seen_receipt):
    path, data = seen_receipt
    binding = gates.validate_receipt(path, 'a' * 40, '1', 'M', 'simple', 'seen', 1)
    fields = dict(reviewed_commit='a' * 40, mutable_inputs=dict(probe_receipt=binding),
                  effective_args=dict(backbone='simple', protocol='seen', yaw_aug=1,
                                      **{'vit_' + key: value for key, value in TIERS['M'].items()}))
    assert gates.timing_limits(fields, '1') == dict(
        epoch_seconds=1.05 * data['T_epoch'], projection_hours=data['T_run'] / 3600,
        ceiling_hours=1.5 * data['T_run'] / 3600, probe_receipt_sha256=binding['sha256'])
    fields['effective_args']['protocol'] = 'unseen'  # the exp_04 M path stays unlimited
    assert gates.timing_limits(fields, '1') is None


@pytest.mark.parametrize('arm', [('M', 'cylindrical', 'seen', 1), ('M', 'simple', 'seen', 0),
                                 ('M', 'simple', 'unseen', 0), ('S', 'simple', 'unseen', 0)])
def test_receipt_refuses_another_arm(seen_receipt, arm):
    with pytest.raises(ValueError, match='tier receipt'):
        gates.validate_receipt(seen_receipt[0], 'a' * 40, '1', *arm)


def test_log_guard_requires_the_single_arm_probe_result():
    argv = launch.command('full', 'ckpt/exp07/seen_aug/attempt_t', 'M', 'simple', 'seen', 1)
    expected = launch.effective_args(argv, '1', SEEN_BATCHES)
    guard = launch.LogGuard(expected, 'probe')
    with pytest.raises(ValueError, match='prefix'):
        guard.feed('EXP04_PROBE_RESULT {}')
    guard.feed('XRIR_RUNTIME_ARGS ' + json.dumps(expected))
    guard.feed('yaw_aug ENABLED W=512 seed=0 counter=(epoch-1)*9265+batch_idx')
    guard.feed('Train Epoch: 1 [0/9265] loss 1.0')
    measured = seen_measurement()
    guard.feed('EXP05_PROBE_RESULT ' + json.dumps(measured))
    assert guard.finish()['probe'] == measured
    guard.feed('EXP05_PROBE_RESULT ' + json.dumps(dict(measured, protocol='unseen')))
    with pytest.raises(ValueError, match='probe'):
        guard.finish()


@pytest.mark.parametrize('argv,message', [
    (['full', '--backbone', 'cylindrical'], 'probe-json'),
    (['probe', '--backbone', 'simple', '--renew-ceiling', '2026-09-15T00:00:00-04:00: reason'],
     'renew-ceiling')])
def test_cli_requires_a_receipt_and_gates_renewal(tmp_path, capsys, argv, message):
    with pytest.raises(SystemExit):
        launch.main(argv + ['--protocol', 'seen', '--reviewed-commit', 'HEAD', '--log-dir', str(tmp_path)])
    assert message in capsys.readouterr().err
    assert not (tmp_path / 'ckpt').exists()
