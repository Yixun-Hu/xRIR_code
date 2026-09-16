"""exp_07 seen fit-probe and gates: receipts, timing limits and the one-retry ledger.

Timings are synthetic and the child is stubbed; no GPU and no trainer are involved.
"""
import json
from pathlib import Path

import pytest

from tools import exp04_launcher as base
from tools import exp05_gates as tier_gates
from tools import exp05_probe as tier_probe
from tools import exp07_gates as gates
from tools import exp07_launcher as launch
from tools import exp07_probe as probe
from tools import provenance as p
from tools.exp05_params import TIERS
from test_exp05_gates import receipt  # noqa: F401  (the historical S/L receipt fixture)
from test_exp07_launcher_run import seen_measurement

ROOT = Path(__file__).resolve().parents[1]
ARMS = [('seen_simple', 'simple', 0), ('seen_cyl', 'cylindrical', 0), ('seen_aug', 'simple', 1)]
SEEN_BATCHES = 9265


def probe_attempt(path, data):
    """Write the attempt files the receipt binds (exp_05 round 4 rules, seen fields)."""
    path.mkdir(parents=True, exist_ok=True)
    manifest = dict(mode='probe', reviewed_commit=data['reviewed_commit'], attempt_path=str(path),
                    effective_args={key: data[key] for key in ('tier', 'backbone', 'protocol', 'yaw_aug')})
    digest = p.write_manifest(path / 'train_manifest.json', manifest)
    completion = dict(train_manifest_sha256=digest, metrics=dict(probe=data))
    return dict(path=str(path), train_manifest_sha256=digest,
                completion_sha256=p.write_manifest(path / 'completion.json', completion))


@pytest.mark.parametrize('name,backbone,yaw', ARMS)
def test_probe_recipe_per_seen_arm(name, backbone, yaw):
    cmd = probe.trainer_command('M', backbone, 'scratch', 'seen', yaw)
    fields = {'backbone': backbone, 'save-dir': 'scratch', 'protocol': 'seen', 'batch-size': 32,
              'accum-steps': 2, 'max-train-batches': 60, 'max-test-batches': 0, 'num-workers': 12,
              'save-every': 0, 'epoch-ckpt-every': 0}
    assert cmd[0] == launch.ENTRY and '--no-save' in cmd and '--tf32' in cmd
    assert not any(flag.startswith('--vit-') for flag in cmd)
    assert all(cmd[cmd.index('--' + key) + 1] == str(value) for key, value in fields.items())
    assert (cmd[cmd.index('--yaw-aug') + 1] == '1') if yaw else (
        not any(flag.startswith('--yaw-') for flag in cmd))
    for bad in (('S', backbone, 'scratch', 'seen', yaw), ('M', backbone, 'scratch', 'unseen', yaw),
                ('M', 'invented', 'scratch', 'seen', yaw)):
        with pytest.raises(ValueError, match='seen probe'):
            probe.trainer_command(*bad)
    with pytest.raises(ValueError, match='yaw_aug'):
        probe.trainer_command('M', backbone, 'scratch', 'seen', 2)


def test_projection_uses_the_seen_batch_count():
    assert probe.TRAIN_BATCHES == launch.TRAIN_BATCHES  # kept in lockstep with the launcher
    seen = seen_measurement()
    assert probe.projection(seen) == dict(T_epoch=9270., T_run=111240., passed=True)
    for wrong in (dict(seen, train_batches_per_epoch=9261), dict(seen, protocol='unseen'),
                  dict(seen, protocol='held-out')):
        with pytest.raises(ValueError, match='timing or recipe'):
            probe.projection(wrong)
    # The exp_05 projection is untouched and still reads its own 9 261 literal.
    assert tier_probe.projection(dict(seen, train_batches_per_epoch=9261))['T_epoch'] == 9266.


@pytest.mark.parametrize('name,backbone,yaw', ARMS)
def test_seen_probe_writes_a_bound_receipt(tmp_path, monkeypatch, name, backbone, yaw):
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    monkeypatch.setattr(launch, 'PYTHON', str(Path(launch.PYTHON)))
    monkeypatch.setattr(base.subprocess, 'check_output', lambda *a, **k: 'a' * 40 + '\n')
    snapshot = dict(gpu='1', uuid='GPU-one', compute_apps='', free_gib=45.)
    monkeypatch.setattr(base, 'gpu_snapshot', lambda gpu: snapshot)
    monkeypatch.setattr(launch, 'build_fields', lambda cmd, *a: dict(command=cmd))
    measured = seen_measurement(backbone, yaw)
    calls = []

    def execute(attempt, mode, gpu, log, factory, **kwargs):
        cmd = factory()['command']
        assert cmd[1:3] == ['-m', 'tools.exp07_probe'] and cmd[cmd.index('--tier') + 1] == 'M'
        assert cmd[cmd.index('--protocol') + 1] == 'seen'
        assert ('--yaw-aug' in cmd) is bool(yaw)
        assert factory()['trainer_command'][1:] == probe.trainer_command(
            'M', backbone, str(attempt.relative_to(tmp_path)), 'seen', yaw)
        assert log == tmp_path / 'logs' / 'seen_protocol_test_train_{}_probe.log'.format(name)
        calls.append(attempt)
        probe_attempt(attempt, dict(measured, reviewed_commit='a' * 40))
        return dict(metrics=dict(probe=measured), resource_before=snapshot)

    monkeypatch.setattr(base, 'execute_attempt', execute)
    launch.main(['probe', '--backbone', backbone, '--reviewed-commit', 'HEAD',
                 '--timestamp', 'test', '--log-dir', str(tmp_path / 'logs')] +
                (['--yaw-aug', '1'] if yaw else []))
    root = tmp_path / 'ckpt/exp07' / name
    path = root / ('_probe_test_' + name + '.json')
    written = json.loads(path.read_text())
    assert len(calls) == 1 and calls[0].parent == root
    assert (written['tier'], written['protocol'], written['yaw_aug']) == ('M', 'seen', yaw)
    assert written['schema_version'] == 1 and written['gpu'] == '1' and written['passed'] is True
    assert written['PROBE_NOT_CLEAN'] is False and written['arms_before'] == [snapshot]
    assert written['T_epoch'] == measured['T_epoch'] and written['T_run'] == measured['T_run']
    assert written['probe_attempt']['path'] == str(calls[0])
    assert written['live_epoch_limit_seconds'] == 1.05 * written['T_epoch']
    assert gates.validate_receipt(path, 'a' * 40, '1', 'M', backbone, 'seen', yaw)['sha256']


@pytest.fixture
def seen_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    state = dict(gpu='1', uuid='GPU-test', compute_apps='', free_gib=45.)
    data = dict(seen_measurement(), reviewed_commit='a' * 40, gpu='1', before=state,
                after=dict(state), arms_before=[dict(state)], PROBE_NOT_CLEAN=False)
    data['probe_attempt'] = probe_attempt(launch.arm_root('simple', 1) / '_probe_t_arm', data)
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
        ceiling_hours=1.5 * data['T_run'] / 3600, probe_receipt_sha256=binding['sha256'],
        protocol='seen')  # the ledger's one-retry rule reads the protocol from the limits
    fields['effective_args']['protocol'] = 'unseen'  # an unseen M arm stays unlimited
    assert gates.timing_limits(fields, '1') is None
    fields['effective_args']['protocol'] = 'seen'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='receipt'):
        gates.timing_limits(fields, '1')


@pytest.mark.parametrize('arm', [('M', 'cylindrical', 'seen', 1), ('M', 'simple', 'seen', 0),
                                 ('M', 'simple', 'unseen', 0), ('S', 'simple', 'seen', 1)])
def test_receipt_refuses_another_arm(seen_receipt, arm):
    with pytest.raises(ValueError, match='seen receipt'):
        gates.validate_receipt(seen_receipt[0], 'a' * 40, '1', *arm)


def substitute_completion(receipt_path, data, **changes):
    """Rewrite the bound completion and recompute the receipt's outer hash of it."""
    bound = data['probe_attempt']
    path = Path(bound['path']) / 'completion.json'
    completion = json.loads(path.read_text())
    completion.update({k: v for k, v in changes.items() if k == 'train_manifest_sha256'})
    completion['metrics']['probe'].update({k: v for k, v in changes.items()
                                           if k != 'train_manifest_sha256'})
    path.write_text(json.dumps(completion))
    bound['completion_sha256'] = p.sha256_file(path)
    receipt_path.write_text(json.dumps(data))


@pytest.mark.parametrize('changes', [
    dict(train_manifest_sha256='0' * 64), dict(protocol='unseen'), dict(yaw_aug=0),
    dict(tier='S'), dict(backbone='cylindrical'), dict(protocol='unseen', yaw_aug=0)])
def test_receipt_requires_a_completion_that_certifies_its_manifest(seen_receipt, changes):
    """Unrelated completion evidence is refused even with accurate outer file hashes."""
    path, data = seen_receipt
    assert gates.validate_receipt(path, 'a' * 40, '1', 'M', 'simple', 'seen', 1)['sha256']
    substitute_completion(path, data, **changes)
    with pytest.raises(ValueError, match='probe attempt arm or measurements differ'):
        gates.validate_receipt(path, 'a' * 40, '1', 'M', 'simple', 'seen', 1)


def test_unseen_receipts_keep_their_shared_behaviour(receipt):
    """The exp_05 S/L receipt validates through the untouched shared gate."""
    path, _ = receipt
    assert tier_gates.validate_receipt(path, 'a' * 40, '1', 'S', 'simple')['sha256']
    with pytest.raises(ValueError, match='seen receipt'):
        gates.validate_receipt(path, 'a' * 40, '1', 'M', 'simple', 'seen', 0)


LIMITS = dict(projection_hours=30., ceiling_hours=45., probe_receipt_sha256='a' * 64)


def ledger(root, modes, hours=1.):
    """A consistent arm ledger holding one row per attempt of the given modes."""
    root.mkdir(parents=True, exist_ok=True)
    attempts = [dict(attempt='attempt_{}'.format(i), hours=hours, mode=mode)
                for i, mode in enumerate(modes)]
    p.write_completion(root / 'cumulative_hours.json', dict(
        total_hours=hours * len(modes), attempts=attempts, ceiling_hours=45.,
        probe_receipt_sha256='a' * 64, probe_projection_hours=30.))


@pytest.mark.parametrize('protocol,modes,allowed', [
    ('seen', [], True), ('seen', ['full'], True), ('seen', ['full', 'smoke', 'probe'], True),
    ('seen', ['full', 'full'], False), ('seen', ['smoke', 'full', 'full', 'probe'], False),
    ('unseen', ['full', 'full'], True)])  # the exp_05 S/L ledgers keep their behaviour
def test_seen_arms_allow_at_most_one_retry(tmp_path, protocol, modes, allowed):
    ledger(tmp_path, modes)
    limits = dict(LIMITS, protocol=protocol)
    if allowed:  # the hours check alone admits every one of these ledgers
        assert gates.set_budget(tmp_path, limits, commit=False)['ceiling_hours'] == 45.
    else:
        with pytest.raises(ValueError, match='one retry'):
            gates.set_budget(tmp_path, limits, commit=False)


@pytest.mark.parametrize('renewal', [None, '2026-09-15T00:00:00-04:00: fresh clean probe'])
def test_neither_a_new_receipt_nor_a_renewal_resets_the_seen_count(tmp_path, renewal):
    ledger(tmp_path, ['full', 'full'])
    limits = dict(LIMITS, protocol='seen', probe_receipt_sha256='b' * 64, ceiling_hours=90.)
    with pytest.raises(ValueError, match='one retry'):
        gates.set_budget(tmp_path, limits, renewal=renewal)
    record = json.loads((tmp_path / 'cumulative_hours.json').read_text())
    assert (record['ceiling_hours'], record['probe_receipt_sha256']) == (45., 'a' * 64)
    assert not record.get('renewals') and not record.get('probe_receipt_history')


def test_third_seen_full_attempt_is_refused_before_any_directory(tmp_path, monkeypatch):
    root = tmp_path / 'ckpt/exp07/seen_simple'
    ledger(root, ['full', 'full'])
    monkeypatch.setattr(base, 'resource_gate', lambda *a: pytest.fail('resource gate reached'))
    with launch.seen_overrides(), pytest.raises(ValueError, match='one retry'):
        base.execute_attempt(root / 'attempt_new', 'full', '1', tmp_path / 'train.log',
                             lambda: pytest.fail('fields built'),
                             limits=dict(LIMITS, protocol='seen'))
    assert [f.name for f in root.iterdir()] == ['cumulative_hours.json']
