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


# --- blocker 4: the 36 h ceiling may only be tightened ------------------------------

CEILING = 129600


def launch(args, env=None):
    result = subprocess.run(LAUNCH + args, cwd=str(REPO), capture_output=True, text=True,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', **(env or {})))
    return result.returncode, result.stdout, result.stderr


@pytest.mark.parametrize('value', ['0', '-1', '36h', '', '129601', '1000000'])
def test_an_override_that_disables_or_loosens_the_ceiling_is_refused(value):
    """GNU timeout treats 0 as 'no timeout', and anything above 36 h exceeds the plan."""
    status, out, err = launch(['full', '--arm', 'H', '--gpu', '1', '--reviewed-commit',
                               COMMIT, '--dry-run'], {'EXP11_FULL_CEILING_S': value})
    assert status == 2, (value, out)
    assert 'EXP11_FULL_CEILING_S' in err and 'refusing' in err
    assert 'nohup setsid' not in out


@pytest.mark.parametrize('value', ['1', '3600', str(CEILING)])
def test_an_override_that_tightens_the_ceiling_is_accepted(value):
    status, out, err = launch(['full', '--arm', 'H', '--gpu', '1', '--reviewed-commit',
                               COMMIT, '--dry-run'], {'EXP11_FULL_CEILING_S': value})
    assert status == 0, err[-500:]
    assert 'CEILING {}s'.format(value) in out
    assert 'timeout --kill-after=60 {} '.format(value) in out


def test_the_default_ceiling_is_the_registered_thirty_six_hours():
    status, out, _ = launch(['full', '--arm', 'I', '--gpu', '1', '--reviewed-commit',
                             COMMIT, '--dry-run'])
    assert status == 0 and 'CEILING {}s'.format(CEILING) in out


# --- blocker 5: recovery may not promote an attempt of another arm ------------------


def attempt_with(root, arm, profile, tmp_path):
    """One attempt directory under a pretraining root, with the args a profile selects."""
    from tools import exp11_recipe
    spec = exp11_recipe.PROFILES[profile]
    args = dict(spec['recipe'], yaw_aug_seed=0, yaw_aug_width=512)
    args.update(spec['production'])
    args.update(backbone=spec['backbone'], save_dir='x', num_workers=12, log_interval=50,
                save_every=spec['save_every'], epoch_ckpt_every=spec['epoch_ckpt_every'],
                env={}, train_batches_per_epoch=exp11_recipe.TRAIN_BATCHES_PER_EPOCH,
                tier='M', param_counts={}, exp11_run_type='full', exp11_profile=profile,
                exp11_registry_sha256='a' * 64, exp11_source_closure_sha256='b' * 64,
                exp11_git_head='c' * 40, exp11_provenance_path='p')
    directory = tmp_path / root / 'attempt_20260927T000000'
    directory.mkdir(parents=True)
    (directory / 'args.json').write_text(json.dumps(args))
    return directory


def recovery(attempt, arm, tmp_path):
    return launch(['finalize', '--arm', arm, '--gpu', '1', '--reviewed-commit', COMMIT,
                   '--attempt', str(attempt), '--log', 'L.log', '--child-exit', '0',
                   '--dry-run'], {'EXP11_PRETRAIN_ROOT': str(tmp_path)})


def test_recovery_refuses_an_attempt_of_another_arm(tmp_path):
    """A valid H attempt may not be promoted under I's root."""
    attempt = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    status, out, err = recovery(attempt, 'I', tmp_path)
    assert status == 2, out
    assert 'refusing' in err
    assert 'PROMOTE' not in out and 'EXP11_FINALIZE' not in out


def test_recovery_refuses_an_attempt_whose_profile_is_not_the_arms(tmp_path):
    """The directory can be right and the recipe still be the other arm's."""
    attempt = attempt_with('xRIR_simpor_8_shot', 'H', 'I_RECIPE', tmp_path)
    status, out, err = recovery(attempt, 'H', tmp_path)
    assert status == 2, out
    assert 'refusing' in err and 'PROMOTE' not in out


def test_recovery_accepts_the_arms_own_attempt(tmp_path):
    attempt = attempt_with('xRIR_simpor_yawaug_8_shot', 'I', 'I_RECIPE', tmp_path)
    status, out, err = recovery(attempt, 'I', tmp_path)
    assert status == 0, err[-600:]
    assert 'PROMOTE' in out and str(attempt.name) in out
