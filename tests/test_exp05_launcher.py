"""Tier argv, legacy control compatibility, and independent arm ledgers."""
import json
import pytest
from tools import exp04_launcher as launch
from tools.exp05_params import TIERS

def control(runtime):
    label = 'simple' if runtime['backbone'] == 'simple' else 'cyl'
    baseline = json.loads((launch.REPO / ('ckpt/xRIR_' + label + '_8_shot/args.json')).read_text())
    env = dict(XRIR_DATA_PATH=launch.DATA_ROOT, OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='1')
    return baseline, env

@pytest.mark.parametrize('tier', ['S', 'L'])
@pytest.mark.parametrize('backbone', ['simple', 'cylindrical'])
def test_tier_golden_and_control(tier, backbone):
    attempt = 'ckpt/exp05/{}_{}/attempt_test'.format(tier, backbone)
    argv = launch.command('full', attempt, tier, backbone)
    launch.check_golden(argv, 'full', attempt)
    with pytest.raises(ValueError):
        launch.check_golden(argv, 'smoke', attempt)
    assert not any(flag.startswith('--yaw-aug') for flag in argv)
    runtime = launch.effective_args(argv, '1', 9261)
    assert runtime['tier'] == tier and runtime['yaw_aug'] == 0
    assert all(runtime['vit_' + key] == value for key, value in TIERS[tier].items())
    assert all(type(value) is int for value in runtime['param_counts'].values())
    differences = launch.compare_control(runtime, *control(runtime))
    assert {key for key in differences if key.startswith('vit_')} == {'vit_' + key for key in TIERS[tier] if TIERS[tier][key] != TIERS['M'][key]}
    assert not {'param_counts', 'tier'} & differences.keys()
    baseline, env = control(runtime)
    launch.compare_control(runtime, dict(baseline, yaw_aug_seed=None), env)
    with pytest.raises(ValueError, match='control.*M'):
        launch.compare_control(runtime, runtime, env)
    guard = launch.LogGuard(runtime, 'full')
    guard.feed('yaw_aug DISABLED')
    assert guard.banner


def test_default_golden_and_legacy_m_metadata():
    argv = launch.command('full', 'attempt')
    launch.check_golden(argv, 'full', 'attempt')
    assert not any(flag.startswith('--vit-') for flag in argv)
    runtime = launch.effective_args(argv, '1', 9261)
    assert runtime['tier'] == 'M'
    assert runtime['param_counts'] == dict(encoder=19703296, full=32075965, trainable=32075945, non_encoder=12372669)
    assert not ({'param_counts', 'tier'} | {'vit_' + key for key in TIERS['M']}) & launch.compare_control(runtime, *control(runtime)).keys()


@pytest.mark.parametrize('tier,backbone,attempt', [
    ('S', 'cylindrical', 'ckpt/exp05/S_simple/attempt_test'),
    ('L', 'simple', 'ckpt/exp05/S_simple/attempt_test'),
    ('S', 'simple', 'ckpt/xRIR_simple_yawaug_8_shot/attempt_test'),
    ('M', 'simple', 'ckpt/exp05/S_simple/attempt_test'),
    ('S', 'simple', 'ckpt/exp05/S_simple/attempt_../other')])
def test_golden_refuses_wrong_arm(tier, backbone, attempt):
    argv = launch.command('full', attempt, tier, 'simple')
    argv[argv.index('--backbone') + 1] = backbone
    with pytest.raises(ValueError, match='golden.*tier'):
        launch.check_golden(argv, 'full', attempt)


def test_missing_golden_is_value_error(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    attempt = 'ckpt/exp05/S_simple/attempt_test'
    with pytest.raises(ValueError, match='golden file missing: argv_golden_S_simple.txt'):
        launch.check_golden(launch.command('full', attempt, 'S'), 'full', attempt)


def test_trailing_backbone_is_parser_error(capsys):
    attempt = 'ckpt/exp05/S_simple/attempt_test'
    argv = launch.command('full', attempt, 'S')
    index = argv.index('--backbone')
    del argv[index:index + 2]
    with pytest.raises(ValueError, match='golden tier'):
        launch.check_golden(argv + ['--backbone'], 'full', attempt)
    with pytest.raises(SystemExit) as error:
        launch.main(['full', '--tier', 'S', '--backbone'])
    assert error.value.code == 2 and 'expected one argument' in capsys.readouterr().err


def test_cli_checks_golden_before_creating_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    monkeypatch.setattr(launch.subprocess, 'check_output', lambda *a, **k: 'a' * 40)
    monkeypatch.setattr(launch.tier_gates, 'validate_receipt', lambda *a: {})
    def refuse(*args):
        assert not list(tmp_path.iterdir())
        raise ValueError('golden tier')
    monkeypatch.setattr(launch, 'check_golden', refuse)
    with pytest.raises(ValueError, match='golden tier'):
        launch.main(['full', '--tier', 'S', '--reviewed-commit', 'HEAD', '--probe-json', '/unused'])


def test_runtime_missing_backbone_converted_to_guard():
    expected = launch.effective_args(launch.command('full', 'attempt'), '1', 9261)
    payload = dict(expected)
    del payload['backbone']
    with pytest.raises(launch.LauncherFailure) as error:
        launch.LogGuard(expected, 'full').runtime_payload(json.dumps(payload))
    assert error.value.reason == 'guard_runtime_args'


@pytest.mark.parametrize('mutation', ['float_count', 'wrong_count', 'wrong_tier'])
def test_derived_metadata_refused(mutation):
    runtime = launch.effective_args(launch.command('full', 'attempt'), '1', 9261)
    baseline = control(runtime)
    if mutation == 'wrong_tier':
        runtime['tier'] = 'S'
    else:
        count = runtime['param_counts']['full']
        runtime['param_counts'] = dict(runtime['param_counts'], full=float(count) if mutation == 'float_count' else count + 1)
    with pytest.raises(ValueError):
        launch.compare_control(runtime, *baseline)
    with pytest.raises(ValueError):
        launch.check_runtime(runtime, runtime, 'full')


@pytest.mark.parametrize('key,value', [('yaw_aug', 1), ('no_save', True), ('lr', 0.002)])
def test_tier_control_forbids_unregistered_changes(key, value):
    runtime = launch.effective_args(launch.command('full', 'attempt', 'S'), '1', 9261)
    runtime[key] = value
    with pytest.raises(ValueError, match=key):
        launch.compare_control(runtime, *control(runtime))


def test_arm_ledger_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    roots = [launch.arm_root(tier, backbone) for tier in ('S', 'L') for backbone in ('simple', 'cylindrical')]
    assert len(set(roots)) == 4 and roots[0] == tmp_path / 'ckpt/exp05/S_simple'
    launch.account_hours(launch.create_attempt(roots[0], 'attempt_test'), 2)
    assert [launch.full_hours(root) for root in roots] == [2, 0, 0, 0]


@pytest.mark.parametrize('mode,tier,backbone,reason', [
    (mode, 'M', 'cylindrical', 'simple') for mode in ('full', 'smoke', 'probe')] + [
    (mode, 'S', 'simple', 'fit-probe') for mode in ('full', 'smoke')])
def test_tier_launch_requires_its_own_probe_protocol(mode, tier, backbone, reason, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    with pytest.raises(SystemExit):
        launch.main([mode, '--tier', tier, '--backbone', backbone, '--reviewed-commit', 'HEAD', '--log-dir', str(tmp_path)])
    assert reason in capsys.readouterr().err
    assert not (tmp_path / 'ckpt').exists()
