"""Tier timing receipts cannot mark resource drift or low memory as clean."""
import json

import pytest

from test_exp05_probe import measurement
from test_exp05_gates import probe_attempt
from tools import exp04_launcher as launch, exp05_probe as probe


@pytest.mark.parametrize('fault,stage', [
    ('low_memory', 'before'), ('low_memory', 'arm'), ('low_memory', 'after'),
    ('changed_gpu_uuid', 'after'), ('different_gpu_index', 'after')])
def test_resource_fault_writes_failed_receipt(tmp_path, monkeypatch, fault, stage):
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    monkeypatch.setattr(launch.subprocess, 'check_output', lambda *a, **k: 'a' * 40 + '\n')
    states = {name: dict(gpu='1', uuid='GPU-one', compute_apps='', free_gib=45.)
              for name in ('before', 'arm', 'after')}
    key, value = {'low_memory': ('free_gib', 10.), 'changed_gpu_uuid': ('uuid', 'GPU-two'),
                  'different_gpu_index': ('gpu', '0')}[fault]
    states[stage][key] = value
    snapshots = iter([states['before'], states['after']])
    monkeypatch.setattr(launch, 'gpu_snapshot', lambda gpu: next(snapshots))
    monkeypatch.setattr(launch, 'build_fields', lambda cmd, *a: dict(command=cmd))
    measured = dict(measurement(), tier='S', backbone='simple', yaw_aug=0,
                    test_timing_protocol='full_test_loader', test_batches_timed=199,
                    test_batches_total=199, peak_allocated_bytes=123, peak_reserved_bytes=456,
                    train_loss=1., iteration_seconds=[1.] * 50, mean_iteration_seconds=1.,
                    median_iteration_seconds=1., min_iteration_seconds=1.)
    measured.update(probe.projection(measured))
    probe_attempt(tmp_path / 'ckpt/exp05/S_simple/_probe_resource_arm')
    monkeypatch.setattr(launch, 'execute_attempt', lambda *a, **k:
                        dict(metrics=dict(probe=measured), resource_before=states['arm']))
    with pytest.raises(SystemExit) as error:
        launch.main(['probe', '--tier', 'S', '--backbone', 'simple', '--reviewed-commit', 'HEAD',
                     '--timestamp', 'resource', '--log-dir', str(tmp_path / 'logs'), '--allow-cotenant'])
    assert error.value.code == 1
    receipt = json.loads((tmp_path / 'ckpt/exp05/S_simple/_probe_resource_S_simple.json').read_text())
    assert receipt['PROBE_NOT_CLEAN'] is True and receipt['passed'] is False
    assert receipt['before'] == states['before'] and receipt['after'] == states['after']
    assert receipt['arms_before'] == [states['arm']]
