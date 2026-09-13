"""A slow arm needs new timing evidence on the same physical GPU."""
import json

import pytest

from tools import exp04_launcher as launch, exp05_gates as gates
from test_exp05_gates import receipt


def bound_fields(path, attempt):
    command = launch.command('full', str(attempt), 'S', 'simple')
    return dict(reviewed_commit='a' * 40, command=command,
        effective_args=launch.effective_args(command, '1', 9261),
        mutable_inputs=dict(probe_receipt=gates.validate_receipt(path, 'a' * 40, '1', 'S', 'simple')))


@pytest.mark.parametrize('evidence', ['same', 'new', 'missing_manifest', 'missing_abort'])
def test_slow_abort_requires_new_receipt(receipt, tmp_path, evidence):
    path, data = receipt
    root = tmp_path / 'arm'
    old = root / 'attempt_old_ABORTED_slow'
    fields = bound_fields(path, old)
    limits = gates.timing_limits(fields, '1')
    gates.set_budget(root, limits)
    old.mkdir()
    (old / 'train_manifest.json').write_text(json.dumps(fields))
    (old / 'abort.json').write_text(json.dumps(dict(reason='guard_epoch_one')))
    if evidence.startswith('missing_'):
        (old / ('train_manifest.json' if evidence == 'missing_manifest' else 'abort.json')).unlink()
    launch.account_hours(old, 1., mode='full')
    before = (root / 'cumulative_hours.json').read_bytes()
    if evidence == 'new':
        data['t_test'] += 1
        data.update(launch.tier_probe.projection(data))
        path.write_text(json.dumps(data))
        newer = gates.timing_limits(bound_fields(path, old), '1')
        assert gates.set_budget(root, newer)['probe_receipt_sha256'] != limits['probe_receipt_sha256']
    else:
        with pytest.raises(ValueError, match='probe|receipt|slow'):
            gates.set_budget(root, limits)
        assert (root / 'cumulative_hours.json').read_bytes() == before


@pytest.mark.parametrize('uuid', ['GPU-test', 'GPU-replaced', None])
def test_timing_limits_check_launch_gpu_identity(receipt, tmp_path, uuid):
    fields = bound_fields(receipt[0], tmp_path / 'attempt')
    fields['resource_before'] = dict(gpu='1', uuid=uuid)
    if uuid == 'GPU-test':
        assert gates.timing_limits(fields, '1')['projection_hours'] > 0
    else:
        with pytest.raises(ValueError, match='GPU|receipt'):
            gates.timing_limits(fields, '1')


def test_full_refuses_changed_physical_gpu_before_spawn(receipt, tmp_path, monkeypatch):
    attempt = tmp_path / 'arm/attempt_new'
    fields = bound_fields(receipt[0], attempt)
    limits = gates.timing_limits(fields, '1')
    monkeypatch.setattr(launch, 'resource_gate', lambda *a: dict(gpu='1', uuid='GPU-replaced'))
    def runner(*args):
        pytest.fail('child spawned despite different GPU from probe')
    with pytest.raises(ValueError, match='GPU|receipt'):
        launch.execute_attempt(attempt, 'full', '1', tmp_path / 'train.log', lambda: fields,
                               limits=limits, runner=runner)
