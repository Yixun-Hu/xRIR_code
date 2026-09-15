"""Tier receipts bind measured timing and resource evidence before spending."""
import copy
import json
from pathlib import Path

import pytest

from tools import exp05_gates as gates
from tools import provenance as p, exp04_launcher as launch
from tools.exp05_params import TIERS
from tools.exp05_probe import projection


def probe_attempt(path, data=None):
    path.mkdir(parents=True, exist_ok=True)
    data = data or dict(tier='S', backbone='simple', reviewed_commit='a' * 40)
    manifest = dict(mode='probe', effective_args={k: data[k] for k in ('tier', 'backbone')},
                    reviewed_commit=data['reviewed_commit'], attempt_path=str(path))
    digest = p.write_manifest(path / 'train_manifest.json', manifest)
    completion = dict(train_manifest_sha256=digest, attempt_path=str(path), metrics=dict(probe=data))
    return dict(path=str(path), train_manifest_sha256=digest,
                completion_sha256=p.write_manifest(path / 'completion.json', completion))


@pytest.fixture
def receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    state = dict(gpu='1', uuid='GPU-test', compute_apps='', free_gib=45.)
    data = dict(schema_version=1, tier='S', backbone='simple', reviewed_commit='a' * 40, gpu='1',
        before=state, after=copy.deepcopy(state), arms_before=[copy.deepcopy(state)],
        PROBE_NOT_CLEAN=False, yaw_aug=0, warmup_micro_batches=10, timed_micro_batches=50,
        batch_size=32, accum_steps=2, train_batches_per_epoch=9261, train_loss=1.,
        t_micro=dict(mean=1., median=1., min=1., values=[1.] * 50), t_test=2., t_save=3.,
        test_timing_protocol='full_test_loader', test_batches_timed=199, test_batches_total=199,
        peak_allocated_bytes=123, peak_reserved_bytes=456, iteration_seconds=[1.] * 50,
        mean_iteration_seconds=1., median_iteration_seconds=1., min_iteration_seconds=1.)
    data.update(projection(data))
    data['probe_attempt'] = probe_attempt(launch.arm_root('S', 'simple') / '_probe_test_arm', data)
    path = tmp_path / 'probe.json'
    path.write_text(json.dumps(data))
    return path, data


@pytest.mark.parametrize('fault', ['other_arm', 'mode', 'tier', 'backbone', 'commit', 'path', 'timing'])
def test_refuses_borrowed_probe_evidence(receipt, fault):
    path, data = receipt
    bound = data['probe_attempt']
    if fault == 'other_arm':
        data['probe_attempt'] = probe_attempt(launch.arm_root('L', 'cylindrical') / '_probe_other_arm', data)
    else:
        target = Path(bound['path']) / 'train_manifest.json'
        manifest = json.loads(target.read_text())
        if fault in ('tier', 'backbone'):
            manifest['effective_args'][fault] = 'L' if fault == 'tier' else 'cylindrical'
        elif fault == 'timing':
            data['t_test'] += 1
            data.update(projection(data))
        else:
            manifest[dict(mode='mode', commit='reviewed_commit', path='attempt_path')[fault]] = 'wrong'
        target.write_text(json.dumps(manifest))
        bound['train_manifest_sha256'] = p.sha256_file(target)
    path.write_text(json.dumps(data))
    message = dict(commit='commit mismatch', tier='tier/backbone mismatch',
                   backbone='tier/backbone mismatch').get(fault, 'probe-attempt binding')
    with pytest.raises(ValueError, match=message):
        gates.validate_receipt(path, 'a' * 40, '1', 'S', 'simple')


@pytest.mark.parametrize('timestamp', ['2026-09-13 00:05', '2026-09-13T00:05:00',
                                     '2026-09-13 00:05:00-04:00', '2026-09-13T00:05-04:00'])
def test_renewal_requires_notebook_timestamp(tmp_path, timestamp):
    limits = dict(projection_hours=20., ceiling_hours=30., probe_receipt_sha256='a' * 64)
    gates.set_budget(tmp_path, limits)
    with pytest.raises(ValueError, match='timestamp'):
        gates.set_budget(tmp_path, dict(limits, probe_receipt_sha256='b' * 64), timestamp + ': reason')


def test_budget_preflight_does_not_create_root(tmp_path):
    root = tmp_path / 'missing'
    limits = dict(projection_hours=20., ceiling_hours=30., probe_receipt_sha256='a' * 64)
    assert gates.set_budget(root, limits, commit=False) == limits
    assert not root.exists()


@pytest.mark.parametrize('field,value', [('tier', 'L'), ('backbone', 'cylindrical'),
    ('reviewed_commit', 'b' * 40), ('gpu', '0'), ('PROBE_NOT_CLEAN', True), ('passed', False),
    ('T_epoch', 1.), ('T_run', 1.), ('schema_version', 2), ('peak_allocated_bytes', 0),
    ('test_batches_timed', 198), ('test_batches_total', 0), ('arms_before', []),
    ('test_timing_protocol', 'fraction')])
def test_refuses_wrong_receipt(receipt, field, value):
    path, data = receipt
    data[field] = value
    path.write_text(json.dumps(data))
    message = dict(tier='tier/backbone mismatch', backbone='tier/backbone mismatch',
        reviewed_commit='commit mismatch', gpu='GPU mismatch', PROBE_NOT_CLEAN='NOT CLEAN',
        passed='T_run', T_epoch='T_run', T_run='T_run').get(field, 'receipt')
    with pytest.raises(ValueError, match=message):
        gates.validate_receipt(path, 'a' * 40, '1', 'S', 'simple')


@pytest.mark.parametrize('field,value', [('gpu', '0'), ('uuid', 'GPU-other'),
                                       ('compute_apps', 'foreign'), ('free_gib', 39.)])
def test_refuses_unclean_snapshots(receipt, field, value):
    path, data = receipt
    data['arms_before'][0][field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='GPU mismatch' if field in ('gpu', 'uuid') else 'NOT CLEAN'):
        gates.validate_receipt(path, 'a' * 40, '1', 'S', 'simple')


@pytest.mark.parametrize('name', [None, 'train_manifest', 'completion'])
@pytest.mark.parametrize('remove', [False, True])
def test_probe_attempt_binding_required(receipt, name, remove):
    path, data = receipt
    if name is None:
        del data['probe_attempt']
        path.write_text(json.dumps(data))
    else:
        target = Path(data['probe_attempt']['path']) / (name + '.json')
        target.unlink() if remove else target.write_text('changed')
    with pytest.raises(ValueError, match='probe-attempt binding'):
        gates.validate_receipt(path, 'a' * 40, '1', 'S', 'simple')


def test_limits_require_bound_receipt_and_revalidate(receipt):
    path, data = receipt
    binding = gates.validate_receipt(path, 'a' * 40, '1', 'S', 'simple')
    assert binding == dict(path=str(path), sha256=p.sha256_file(path))
    fields = dict(effective_args=dict(backbone='simple', **{'vit_' + k: v for k, v in TIERS['S'].items()}),
                  reviewed_commit='a' * 40, mutable_inputs=dict(probe_receipt=binding))
    limits = gates.timing_limits(fields, '1')
    assert limits == dict(epoch_seconds=1.05 * data['T_epoch'], projection_hours=data['T_run'] / 3600,
                         ceiling_hours=1.5 * data['T_run'] / 3600, probe_receipt_sha256=binding['sha256'])
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='receipt'):
        gates.timing_limits(fields, '1')
    del fields['mutable_inputs']['probe_receipt']
    with pytest.raises(ValueError, match='receipt'):
        gates.timing_limits(fields, '1')
    fields['effective_args'] = dict(backbone='simple', **{'vit_' + k: v for k, v in TIERS['M'].items()})
    assert gates.timing_limits(fields, '1') is None


def test_refuses_valid_projection_over_sixty_hours(receipt):
    path, data = receipt
    data.update(t_test=10000.)
    data.update(projection(data), passed=True)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='T_run'):
        gates.validate_receipt(path, 'a' * 40, '1', 'S', 'simple')


def test_receipt_change_during_validation_names_staleness(receipt, monkeypatch):
    path, _ = receipt
    original = p.sha256_file
    monkeypatch.setattr(p, 'sha256_file', lambda target: '0'*64 if Path(target) == path else original(target))
    with pytest.raises(ValueError, match='stale'):
        gates.validate_receipt(path, 'a' * 40, '1', 'S', 'simple')


def test_budget_preserves_attempts_and_reprobe_cannot_raise_ceiling(tmp_path):
    limits = dict(epoch_seconds=100., projection_hours=20., ceiling_hours=30., probe_receipt_sha256='a' * 64)
    gates.set_budget(tmp_path, limits)
    path = tmp_path / 'cumulative_hours.json'
    record = json.loads(path.read_text())
    record.update(total_hours=5., attempts=[dict(attempt='attempt_old', mode='full', hours=5.)])
    p.write_completion(path, record)
    newer = dict(limits, projection_hours=25., ceiling_hours=37.5, probe_receipt_sha256='b' * 64)
    assert gates.set_budget(tmp_path, newer)['ceiling_hours'] == 30.
    record = json.loads(path.read_text())
    assert record['attempts'][0]['hours'] == 5 and record['probe_receipt_history'] == ['a' * 64]
    with pytest.raises(ValueError, match='ceiling'):
        gates.set_budget(tmp_path, dict(newer, projection_hours=26., ceiling_hours=39., probe_receipt_sha256='c' * 64))
    assert json.loads(path.read_text()) == record
    renewal = '2026-09-12T23:39:54-04:00: slower clean re-probe'
    newer.update(projection_hours=26., ceiling_hours=39., probe_receipt_sha256='c' * 64)
    assert gates.set_budget(tmp_path, newer, renewal=renewal)['ceiling_hours'] == 39.
    assert json.loads(path.read_text())['renewals'] == [dict(timestamp='2026-09-12T23:39:54-04:00',
        reason='slower clean re-probe', old_ceiling=30., new_ceiling=39., receipt_sha256='c' * 64)]
    for reused in (newer, limits):
        with pytest.raises(ValueError, match='new.*receipt'):
            gates.set_budget(tmp_path, reused, renewal=renewal)


def test_budget_before_spending_can_replace_projection(tmp_path):
    limits = dict(epoch_seconds=100., projection_hours=20., ceiling_hours=30., probe_receipt_sha256='a' * 64)
    gates.set_budget(tmp_path, limits)
    changed = dict(limits, projection_hours=30., ceiling_hours=45.)
    with pytest.raises(ValueError, match='ceiling'):
        gates.set_budget(tmp_path, changed)
    assert gates.set_budget(tmp_path, dict(changed, probe_receipt_sha256='b' * 64))['ceiling_hours'] == 45.
