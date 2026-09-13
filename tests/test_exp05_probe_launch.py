"""Single-arm probe receipts and nonzero cleanliness/projection exits."""
import json

import pytest

from tools import exp04_launcher as launch, exp05_probe as probe
from test_exp05_probe import measurement


@pytest.mark.parametrize('tier,backbone', [('S', 'simple'), ('L', 'cylindrical')])
@pytest.mark.parametrize('fault', [None, 'cotenant', 'slow'])
def test_single_arm_probe_receipt(tmp_path, monkeypatch, tier, backbone, fault):
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    monkeypatch.setattr(launch.subprocess, 'check_output', lambda *a, **k: 'a' * 40 + '\n')
    snapshot = dict(gpu='1', uuid='GPU-one', compute_apps='foreign' if fault == 'cotenant' else '', free_gib=45.)
    monkeypatch.setattr(launch, 'gpu_snapshot', lambda gpu: snapshot)
    monkeypatch.setattr(launch, 'build_fields', lambda cmd, *a: dict(command=cmd))
    measured = dict(measurement(), tier=tier, backbone=backbone, yaw_aug=0,
                    test_timing_protocol='full_test_loader', test_batches_timed=199, test_batches_total=199,
                    peak_allocated_bytes=123, peak_reserved_bytes=456)
    if fault == 'slow':
        measured['t_micro'] = dict(mean=2., median=2., min=2., values=[2.] * 50)
    measured.update(probe.projection(measured))
    calls = []
    def execute(attempt, mode, gpu, log, factory, **kwargs):
        fields = factory()
        cmd = fields['command']
        assert cmd[1:3] == ['-m', 'tools.exp05_probe']
        assert cmd[cmd.index('--tier') + 1] == tier
        assert cmd[cmd.index('--backbone') + 1] == backbone
        assert fields['trainer_command'][1:] == probe.trainer_command(tier, backbone, str(attempt.relative_to(tmp_path)))
        calls.append(attempt)
        assert log == tmp_path / 'logs' / ('param_efficiency_test_train_{}_{}_probe.log'.format(tier, backbone))
        return dict(metrics=dict(probe=measured), resource_before=snapshot)
    monkeypatch.setattr(launch, 'execute_attempt', execute)
    argv = ['probe', '--tier', tier, '--backbone', backbone, '--reviewed-commit', 'HEAD',
            '--timestamp', 'test', '--log-dir', str(tmp_path / 'logs'), '--allow-cotenant']
    if fault:
        with pytest.raises(SystemExit) as error:
            launch.main(argv)
        assert error.value.code == 1
    else:
        launch.main(argv)
    root = tmp_path / 'ckpt/exp05' / (tier + '_' + backbone)
    receipt = json.loads((root / '_probe_test_{}_{}.json'.format(tier, backbone)).read_text())
    assert len(calls) == 1 and calls[0].parent == root
    assert receipt['tier'] == tier and receipt['backbone'] == backbone
    assert receipt['schema_version'] == 1 and receipt['gpu'] == '1'
    assert receipt['reviewed_commit'] == 'a' * 40 and receipt['arms_before'] == [snapshot]
    assert receipt['PROBE_NOT_CLEAN'] is (fault == 'cotenant')
    assert receipt['passed'] is (fault is None)
    assert receipt['T_epoch'] == measured['T_epoch'] and receipt['T_run'] == measured['T_run']
