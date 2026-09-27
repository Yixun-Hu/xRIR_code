"""Golden dry runs of exp_11's launcher and HAA pipeline (plan v3 items 10 and 12).

Every argv the two shells would issue is frozen in
``orientation_cue_fairness_results_assets/golden/``. A deliberate change regenerates the
files in the same commit; an accidental one -- a flag, a path, a run type, a cue -- fails
here. The refusals (unknown init, malformed job, unknown arm) are checked directly.
"""
import os
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
GOLD = (REPO / 'worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude'
        / 'orientation_cue_fairness_results_assets/golden')
PYTHON = '/home/yixunhu/miniconda3/envs/xRIR/bin/python'
COMMIT = 'c' * 40
LAUNCH = ['bash', 'tools/exp11_launch.sh']
PIPELINE = ['bash', 'tools/exp11_haa_pipeline.sh']
CASES = {
    'launch_full_H': LAUNCH + ['full', '--arm', 'H', '--gpu', '1',
                               '--reviewed-commit', COMMIT, '--dry-run'],
    'launch_full_I': LAUNCH + ['full', '--arm', 'I', '--gpu', '1',
                               '--reviewed-commit', COMMIT, '--dry-run'],
    'launch_probe_H': LAUNCH + ['probe', '--arm', 'H', '--gpu', '1',
                                '--reviewed-commit', COMMIT, '--dry-run'],
    'launch_smoke_H': LAUNCH + ['smoke', '--arm', 'H', '--gpu', '1',
                                '--reviewed-commit', COMMIT, '--dry-run'],
    'launch_finalize_I': LAUNCH + ['finalize', '--arm', 'I', '--gpu', '1',
                                   '--reviewed-commit', COMMIT, '--attempt',
                                   'ckpt/exp11/pretrain/xRIR_simpor_yawaug_8_shot/attempt_X',
                                   '--log', 'L.log', '--child-exit', '0', '--dry-run'],
    'pipeline_simple_or_seed0': PIPELINE + ['1', 'simple_or:0', '--dry-run'],
    'pipeline_simple_or_yaw_seed1': PIPELINE + ['1', 'simple_or_yaw:1', '--dry-run'],
    'pipeline_control_adapter_seed2': PIPELINE + ['1', 'control_adapter:2', '--dry-run'],
    'pipeline_yawaug_adapter_zeroshot': PIPELINE + ['1', 'yawaug_adapter:zeroshot',
                                                    '--dry-run'],
    'pipeline_mixed_queue': PIPELINE + ['0', 'simple_or:0', 'control_adapter:zeroshot',
                                        'simple_or_yaw:2', '--dry-run'],
}


def shell(command):
    """Run one shell with no GPU and normalise the machine-specific paths."""
    result = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    text = (result.stdout.replace(PYTHON, '<PYTHON>')
            .replace(str(Path.home()), '<HOME>').replace(str(REPO), '<REPO>'))
    return result.returncode, text, result.stderr


@pytest.mark.parametrize('name', sorted(CASES))
def test_golden_dry_run(name):
    status, text, stderr = shell(CASES[name])
    assert status == 0, stderr[-800:]
    golden = GOLD / (name + '.txt')
    assert golden.is_file(), 'missing golden {}'.format(golden)
    assert text == golden.read_text(), (
        'the {} dry run changed; regenerate {} in the same commit if deliberate'.format(
            name, golden.name))


def test_static_syntax():
    for script in ('tools/exp11_launch.sh', 'tools/exp11_haa_pipeline.sh'):
        assert subprocess.run(['bash', '-n', script], cwd=str(REPO)).returncode == 0


def test_the_pipeline_refuses_the_whole_queue_before_anything_runs():
    for job, message in (('cyl_or:0', 'unknown init'),          # an exp_06 arm
                         ('yawaug:0', 'unknown init'),          # an exp_09 arm
                         ('simple_or:x', 'is not a seed'),
                         ('simple_or:0:1', 'is not a seed'),
                         ('zeroshot', 'unknown job')):
        result = subprocess.run(PIPELINE + ['1', job, '--dry-run'], cwd=str(REPO),
                                capture_output=True, text=True,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        assert result.returncode == 2, job
        assert message in result.stderr, (job, result.stderr)
        assert 'MKDIR' not in result.stdout


def test_the_launcher_refuses_an_unknown_arm_and_an_incomplete_recovery():
    for extra, status in ((['--arm', 'Z'], 2), ([], 2)):
        result = subprocess.run(
            LAUNCH + ['full', '--gpu', '1', '--reviewed-commit', COMMIT, '--dry-run'] + extra,
            cwd=str(REPO), capture_output=True, text=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        if extra:
            assert result.returncode == status and 'unknown arm' in result.stderr
    result = subprocess.run(LAUNCH + ['finalize', '--arm', 'H', '--gpu', '1',
                                      '--reviewed-commit', COMMIT, '--dry-run'],
                            cwd=str(REPO), capture_output=True, text=True,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    assert result.returncode == 2 and 'usage' in result.stderr


def test_every_init_and_every_cue_appears_in_the_golden_queues():
    """Each arm's frame, cue flag and entry points are what plan section 2.2 registers."""
    heading = (GOLD / 'pipeline_simple_or_seed0.txt').read_text()
    assert 'frame=heading' in heading and '--heading-json-dir ckpt/exp06/heading' in heading
    assert '--adapter-heading-json-dir' not in heading
    adapter = (GOLD / 'pipeline_control_adapter_seed2.txt').read_text()
    assert 'frame=room' in adapter
    assert '--adapter-heading-json-dir ckpt/exp06/heading' in adapter
    assert '--heading-json-dir' not in adapter.replace('--adapter-heading-json-dir', '')
    for text in (heading, adapter):
        assert 'tools/exp11_haa_finetune.py' in text and 'tools/exp11_haa_eval.py' in text
        assert '--run-type exp11_haa_finetune' in text or 'exp11_haa_finetune' in text
        assert 'tools/exp06_haa_finetune.py' not in text
        assert 'CENSUS gpu=' in text          # the explicit exclusive-card census
        assert 'exp11_haa_job' in text
    mixed = (GOLD / 'pipeline_mixed_queue.txt').read_text()
    for init in ('simple_or', 'control_adapter', 'simple_or_yaw'):
        assert 'JOB {} '.format(init) in mixed
    assert mixed.count('CENSUS gpu=0') == 3   # one census per job, never inherited
    assert 'QUEUE_DONE gpu=0' in mixed
