"""Golden dry runs of exp_11's launcher and HAA pipeline (plan v3 items 10 and 12).

Every argv the two shells would issue is frozen in
``orientation_cue_fairness_results_assets/golden/``. A deliberate change regenerates the
files in the same commit; an accidental one -- a flag, a path, a run type, a cue -- fails
here. The refusals (unknown init, malformed job, unknown arm) are checked directly.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

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


def attempt_with(root, arm, profile, tmp_path, name='attempt_20260927T000000'):
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
    directory = tmp_path / root / name
    directory.mkdir(parents=True)
    (directory / 'args.json').write_text(json.dumps(args))
    return directory


def recovery(attempt, arm, tmp_path, extra=(), log='L.log'):
    return launch(['finalize', '--arm', arm, '--gpu', '1', '--reviewed-commit', COMMIT,
                   '--attempt', str(attempt), '--log', log, '--child-exit', '0',
                   '--dry-run'] + list(extra),
                  {'EXP11_PRETRAIN_ROOT': str(tmp_path), 'EXP11_TEST_ROOTS': '1'})


def publish(attempt, target=None, completed=True):
    """Make ``final`` point at an attempt, as a finished run leaves the root."""
    if completed:
        (attempt / 'completion.json').write_text('{"run_type": "exp11_train"}')
    link = attempt.parent / 'final'
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to((target or attempt).name)
    return link


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
    """The accepted path also pins the recovery argv the golden set no longer carries.

    ``launch_finalize_I`` left the golden set when this check began running in dry-run:
    a golden of a *nonexistent* attempt would now be a golden of a refusal. The argv it
    used to pin is asserted here instead, on an attempt that really is arm I's.
    """
    attempt = attempt_with('xRIR_simpor_yawaug_8_shot', 'I', 'I_RECIPE', tmp_path)
    status, out, err = recovery(attempt, 'I', tmp_path)
    assert status == 0, err[-600:]
    assert 'ATTEMPT ok {} arm=I profile=I_RECIPE'.format(attempt) in out
    assert ('tools/exp11_finalize.py --run-dir {} --run-type exp11_train --log L.log '
            '--child-exit 0'.format(attempt)) in out
    assert 'preflight --mode finalize' in out
    assert 'PROMOTE {}/final -> {}'.format(attempt.parent, attempt.name) in out


# --- process item 6: the bundled smoke covers the adapter route too -----------------


def test_the_smoke_mode_carries_adapter_training_and_evaluation_commands():
    """J/K must have a reviewed smoke before their queue, not only a fixture.

    The oriented route's smokes prove nothing about ``simple_adapter``: a different
    backbone, a different cue flag, and the contextual loading mode the round-2 fix
    added. Both adapter rungs are therefore in the bundled smoke, the evaluation one
    gated on the fine-tuning one having passed.
    """
    golden = (GOLD / 'launch_smoke_H.txt').read_text()
    assert '--make-fixture ckpt/exp11/_smoke/fixture_simpadapter.pth' in golden
    assert golden.count('--kind haa_train_smoke') == 2      # oriented, then adapter
    assert golden.count('--kind haa_eval_smoke') == 2
    adapter_train = [line for line in golden.splitlines()
                     if '--kind haa_train_smoke' in line and 'simple_adapter' in line]
    adapter_eval = [line for line in golden.splitlines()
                    if '--kind haa_eval_smoke' in line and 'simple_adapter' in line]
    assert len(adapter_train) == len(adapter_eval) == 1
    assert '--adapter-heading-json-dir ckpt/exp06/heading' in adapter_train[0]
    assert '--init ckpt/exp11/_smoke/fixture_simpadapter.pth' in adapter_train[0]
    assert '--adapter-heading-json-dir ckpt/exp06/heading' in adapter_eval[0]
    assert '--heading-json-dir' not in adapter_train[0].replace(
        '--adapter-heading-json-dir', '')
    # Each rung that hands the next one a checkpoint is gated on its own completion.
    assert golden.count('passed --run-dir') == 2
    assert golden.index('haa_adapter_finetune') < golden.index('haa_adapter_eval')


# --- close review blocker 2: the ceiling check must not fail open on overflow -------


@pytest.mark.parametrize('value', ['9223372036854775808',          # 2**63
                                   '9' * 39,                       # 39 digits
                                   '0129600',                      # over six digits
                                   '129600.0', '1e5', ' 3600'])
def test_a_ceiling_outside_bashs_numeric_range_is_refused(value):
    """Bash's -gt/-le error out above 2**63-1, leaving the `if` false and the launch on.

    The contract is (0, 129600], so the check is a bounded decimal string first and a
    numeric comparison only afterwards; anything that cannot be parsed refuses.
    """
    status, out, err = launch(['full', '--arm', 'H', '--gpu', '1', '--reviewed-commit',
                               COMMIT, '--dry-run'], {'EXP11_FULL_CEILING_S': value})
    assert status == 2, (value, out[-400:])
    assert 'EXP11_FULL_CEILING_S' in err and 'refusing' in err
    assert 'timeout --kill-after' not in out
    assert 'integer expression expected' not in err


def test_the_boundary_values_are_exactly_the_contract():
    accepted, _, _ = launch(['full', '--arm', 'H', '--gpu', '1', '--reviewed-commit',
                             COMMIT, '--dry-run'], {'EXP11_FULL_CEILING_S': str(CEILING)})
    refused, _, _ = launch(['full', '--arm', 'H', '--gpu', '1', '--reviewed-commit',
                            COMMIT, '--dry-run'],
                           {'EXP11_FULL_CEILING_S': str(CEILING + 1)})
    assert (accepted, refused) == (0, 2)


# --- close review blocker 3: promotion follows the canonical attempt ----------------


def alias(target, name, tmp_path):
    link = tmp_path / name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    return link


def test_an_existing_final_alias_promotes_the_canonical_attempt(tmp_path):
    """``--attempt <root>/final`` must never promote ``final -> final``.

    The resolved attempt is what was validated, so it is also what is finalized and
    promoted: the link ends up pointing at the real ``attempt_*`` directory.
    """
    attempt = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    link = alias(attempt.name, 'xRIR_simpor_8_shot/final', tmp_path)
    status, out, err = recovery(link, 'H', tmp_path)
    assert status == 0, err[-600:]
    assert 'PROMOTE {}/final -> {}'.format(attempt.parent, attempt.name) in out
    assert 'PROMOTE {}/final -> final'.format(attempt.parent) not in out
    assert '--run-dir {} '.format(attempt) in out          # finalization is canonical too


def test_an_external_alias_promotes_the_canonical_attempt(tmp_path):
    """An alias outside the arm root may not leave a dangling target under it."""
    attempt = attempt_with('xRIR_simpor_yawaug_8_shot', 'I', 'I_RECIPE', tmp_path)
    link = alias(attempt, 'elsewhere/attempt_alias', tmp_path)
    status, out, err = recovery(link, 'I', tmp_path)
    assert status == 0, err[-600:]
    assert 'PROMOTE {}/final -> {}'.format(attempt.parent, attempt.name) in out
    assert 'attempt_alias' not in out.split('PROMOTE')[-1]


def test_an_attempt_that_does_not_resolve_into_the_arm_root_is_refused(tmp_path):
    attempt = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    outside = tmp_path / 'outside' / 'attempt_x'
    outside.mkdir(parents=True)
    (outside / 'args.json').write_text((attempt / 'args.json').read_text())
    status, out, err = recovery(outside, 'H', tmp_path)
    assert status == 2 and 'PROMOTE' not in out


def test_a_resolved_attempt_must_be_named_like_an_attempt(tmp_path):
    """``final -> some_other_directory`` is not a recoverable attempt."""
    attempt = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    other = attempt.parent / 'not_an_attempt'
    other.mkdir()
    (other / 'args.json').write_text((attempt / 'args.json').read_text())
    status, out, err = recovery(other, 'H', tmp_path)
    assert status == 2, out
    assert 'attempt_' in err and 'PROMOTE' not in out


# --- close review blocker 3b: the root override is a test-context override ----------


def test_the_pretrain_root_override_needs_the_test_context(tmp_path):
    """It must not silently move a real launch's roots or its preflight scan roots."""
    attempt = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    status, out, err = launch(['finalize', '--arm', 'H', '--gpu', '1', '--reviewed-commit',
                               COMMIT, '--attempt', str(attempt), '--log', 'L.log',
                               '--child-exit', '0', '--dry-run'],
                              {'EXP11_PRETRAIN_ROOT': str(tmp_path)})
    assert status == 2, out
    assert 'EXP11_PRETRAIN_ROOT' in err and 'refusing' in err
    assert 'PROMOTE' not in out


def test_the_override_is_honoured_only_alongside_a_dry_run(tmp_path):
    attempt = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    args = ['finalize', '--arm', 'H', '--gpu', '1', '--reviewed-commit', COMMIT,
            '--attempt', str(attempt), '--log', 'L.log', '--child-exit', '0']
    env = {'EXP11_PRETRAIN_ROOT': str(tmp_path), 'EXP11_TEST_ROOTS': '1'}
    status, out, err = launch(args, env)             # no --dry-run
    assert status == 2 and 'EXP11_PRETRAIN_ROOT' in err
    status, out, err = launch(args + ['--dry-run'], env)
    assert status == 0, err[-600:]
    assert str(tmp_path) in out


# --- close review 2: recovery safety -----------------------------------------------

LOCK = '.publish.lock'


def test_a_refused_recovery_never_aborts_a_published_attempt(tmp_path):
    """A malformed recovery does not establish that a finished run failed.

    With ``final -> attempt_A``, a refusal must leave the directory and the link exactly
    as they were. The attempt is deliberately published but *not* yet completed, so the
    idempotency path does not fire and the refusal really reaches the abort decision.
    """
    attempt = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    publish(attempt, completed=False)
    status, out, err = recovery(attempt, 'H', tmp_path, log='/nonexistent/L.log')
    assert 'PRESERVED' in out, 'a published attempt is preserved, never aborted'
    assert '_ABORTED_' not in out


def test_a_refused_recovery_never_aborts_a_completed_attempt(tmp_path):
    """Completion alone is enough: the run finished, whatever this invocation did."""
    attempt = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    (attempt / 'completion.json').write_text('{"run_type": "exp11_train"}')
    _, out, _ = recovery(attempt, 'H', tmp_path, log='/nonexistent/L.log')
    assert 'PRESERVED' in out and '_ABORTED_' not in out
def lib(script, env=None):
    """Run a snippet with the launcher sourced as a library (no mode, no preflight)."""
    body = ('set -euo pipefail\n'
            'EXP11_LAUNCH_LIB=1 source tools/exp11_launch.sh\n' + script)
    return subprocess.run(['bash', '-c', body], cwd=str(REPO), capture_output=True,
                          text=True,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES='', **(env or {})))


@pytest.mark.parametrize('published,completed', [(True, False), (False, True), (True, True)])
def test_abort_recovery_really_leaves_the_directory_alone(tmp_path, published, completed):
    """The filesystem effect, not only the announcement: nothing is renamed or removed."""
    attempt = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    log = tmp_path / 'child.log'
    log.write_text('some output\n')
    if completed:
        (attempt / 'completion.json').write_text('{"run_type": "exp11_train"}')
    if published:
        publish(attempt, completed=False)
    result = lib('ARM_ROOT={root}\nDRY=0\nabort_recovery {attempt} {log}\n'.format(
        root=attempt.parent, attempt=attempt, log=log))
    assert result.returncode == 0, result.stderr[-400:]
    assert 'PRESERVED' in result.stdout and 'ABORT' not in result.stdout
    assert attempt.is_dir() and log.is_file()
    assert not list(attempt.parent.glob('*_ABORTED_*'))
    if published:
        link = attempt.parent / 'final'
        assert link.is_symlink() and link.resolve() == attempt.resolve()


def test_abort_recovery_still_aborts_an_unowned_attempt(tmp_path):
    """Narrowing the abort path does not remove it: an orphan attempt is still cleaned."""
    attempt = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    log = tmp_path / 'child.log'
    log.write_text('some output\n')
    result = lib('ARM_ROOT={root}\nDRY=0\nabort_recovery {attempt} {log}\n'.format(
        root=attempt.parent, attempt=attempt, log=log))
    assert result.returncode == 0, result.stderr[-400:]
    assert 'ABORT' in result.stdout and 'PRESERVED' not in result.stdout
    assert not attempt.exists()
    assert (attempt.parent / (attempt.name + '_ABORTED_finalize_refused')).is_dir()


# --- plan section 11 amendment A1: the per-arm lock is the kernel's ------------------

LOCKFILE = '.publish.lock'


def flock_holder(lockfile):
    """A helper this test starts and terminates: it holds ``flock`` on the lock file.

    Only ``Popen.terminate()`` on a process this test created is used; no other process
    is ever signalled.
    """
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    lockfile.touch()
    # bash itself holds the descriptor; the keep-alive child has it closed, so
    # terminating this process really drops the lock.
    holder = subprocess.Popen(
        ['bash', '-c', 'exec 9>"$1"; flock -n 9 || exit 1; echo HELD; '
                       'while :; do sleep 1 9>&-; done', '_', str(lockfile)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for _ in range(100):                     # wait until the kernel really holds it
        probe = subprocess.run(['flock', '-n', str(lockfile), 'true'])
        if probe.returncode != 0:
            return holder
        time.sleep(0.05)
    holder.terminate()
    pytest.skip('the flock helper never took the lock')


def test_the_lock_file_is_the_arms_and_is_held_by_the_kernel(tmp_path):
    """A recovery refuses immediately while another invocation holds the arm's lock."""
    attempt = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    holder = flock_holder(attempt.parent / LOCKFILE)
    try:
        status, out, err = recovery(attempt, 'H', tmp_path)
        assert status == 2, out
        assert 'lock' in err.lower() and 'another invocation' in err.lower()
        assert 'PROMOTE' not in out and 'EXP11_FINALIZE' not in out
    finally:
        holder.terminate()
        holder.wait(timeout=30)


def test_full_mode_takes_the_same_lock(tmp_path):
    root = tmp_path / 'xRIR_simpor_8_shot'
    holder = flock_holder(root / LOCKFILE)
    try:
        status, out, err = launch(['full', '--arm', 'H', '--gpu', '1',
                                   '--reviewed-commit', COMMIT, '--dry-run'],
                                  {'EXP11_PRETRAIN_ROOT': str(tmp_path),
                                   'EXP11_TEST_ROOTS': '1'})
        assert status == 2 and 'lock' in err.lower()
        assert 'PROMOTE' not in out and 'nohup setsid' not in out
    finally:
        holder.terminate()
        holder.wait(timeout=30)


def test_full_mode_takes_the_lock_once_and_never_refuses_itself(tmp_path):
    """One protected section, one descriptor.

    flock(2) treats two descriptors on the same file as independent even inside a single
    process, so acquiring the arm lock twice would refuse the launcher its own lock.
    """
    status, out, err = launch(['full', '--arm', 'H', '--gpu', '1', '--reviewed-commit',
                               COMMIT, '--dry-run'],
                              {'EXP11_PRETRAIN_ROOT': str(tmp_path),
                               'EXP11_TEST_ROOTS': '1'})
    assert status == 0, err[-600:]
    assert 'another invocation' not in err, 'the launcher refused its own lock'
    lines = out.splitlines()
    assert sum(1 for line in lines if line.startswith('LOCK ')) == 1, out
    assert sum(1 for line in lines if line.startswith('LOCKED ')) == 1, out


def test_taking_the_lock_does_not_write_to_the_file_it_locks(tmp_path):
    """Acquisition is read-only with respect to the lock file.

    Truncating a file another invocation holds locked is never something a lock needs to
    do, and a launcher that did it would be writing inside someone else's critical
    section.
    """
    attempt = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    lockfile = attempt.parent / LOCKFILE
    lockfile.write_bytes(b'not the launcher\n')
    status, out, err = recovery(attempt, 'H', tmp_path)
    assert status == 0, err[-500:]
    assert 'PROMOTE' in out
    assert lockfile.read_bytes() == b'not the launcher\n'


def test_the_kernel_releases_the_lock_when_the_holder_ends(tmp_path):
    """No cleanup code, no stale-lock concept: the lock vanishes with its holder."""
    attempt = attempt_with('xRIR_simpor_yawaug_8_shot', 'I', 'I_RECIPE', tmp_path)
    holder = flock_holder(attempt.parent / LOCKFILE)
    assert recovery(attempt, 'I', tmp_path)[0] == 2
    holder.terminate()
    holder.wait(timeout=30)
    status, out, err = recovery(attempt, 'I', tmp_path)
    assert status == 0, err[-500:]
    assert 'PROMOTE' in out


def test_a_refusal_on_the_lock_changes_nothing(tmp_path):
    published = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    publish(published)
    lockfile = published.parent / LOCKFILE
    holder = flock_holder(lockfile)
    try:
        before = (published.is_dir(), (published / 'completion.json').read_text(),
                  os.readlink(published.parent / 'final'), lockfile.read_bytes())
        status, out, _ = recovery(published, 'H', tmp_path)
        assert status == 2 and 'PROMOTE' not in out and '_ABORTED_' not in out
        assert (published.is_dir(), (published / 'completion.json').read_text(),
                os.readlink(published.parent / 'final'), lockfile.read_bytes()) == before
    finally:
        holder.terminate()
        holder.wait(timeout=30)


# --- close review 6: the behaviour A1 still requires, restored -----------------------
# These went out with the hand-rolled lock's own tests, but none of them is about the
# lock mechanism: they are what recovery and the signal handlers must do, whoever holds
# the lock. The fixtures below take the real lock through the real holder.


def test_recovery_of_an_unpublished_incomplete_attempt_still_aborts(tmp_path):
    """The ownership rule narrows the abort path; it does not remove it."""
    attempt = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    status, out, _ = recovery(attempt, 'H', tmp_path, log='/nonexistent/L.log')
    assert '_ABORTED_finalize_refused' in out and 'PRESERVED' not in out


def test_same_target_recovery_is_idempotent(tmp_path):
    """``final`` already resolves to this attempt and it is complete: nothing to do."""
    attempt = attempt_with('xRIR_simpor_yawaug_8_shot', 'I', 'I_RECIPE', tmp_path)
    publish(attempt)
    status, out, err = recovery(attempt, 'I', tmp_path)
    assert status == 0, err[-500:]
    assert 'already published' in out.lower()
    assert 'PROMOTE' not in out and 'EXP11_FINALIZE' not in out


def test_a_different_existing_final_is_a_conflict(tmp_path):
    """Silent replacement is not a recovery policy; it must be asked for."""
    published = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    other = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path,
                         name='attempt_20260927T111111')
    publish(published)
    status, out, err = recovery(other, 'H', tmp_path)
    assert status == 2, out
    assert '--replace-final' in err and 'PROMOTE' not in out
    status, out, err = recovery(other, 'H', tmp_path, extra=['--replace-final'])
    assert status == 0, err[-500:]
    assert 'REPLACING' in out and published.name in out
    assert 'PROMOTE {}/final -> {}'.format(other.parent, other.name) in out


def handler_replay(root, signal_name):
    """Take the real lock, then replay the registered handler body. No signal is sent."""
    return ('ARM_ROOT={root}\nDRY=0\nARM=H\nhold_arm_lock finalize\n'
            'handler="$(trap -p {sig} | sed -n "s/^trap -- \'\\(.*\\)\' SIG{sig}$/\\1/p")"\n'
            '[ -n "$handler" ] || {{ echo NO_{sig}_TRAP; exit 9; }}\n'
            'eval "$handler"\necho CONTINUED_AFTER_HANDLER\n'
            'promote attempt_X\n').format(root=root, sig=signal_name)


def test_the_term_handler_terminates_instead_of_resuming(tmp_path):
    """A handler that only cleans up lets the protected work continue.

    The registered body is replayed directly -- no signal is ever sent -- and execution
    must not return to the caller: it says so and exits 128+n.
    """
    root = tmp_path / 'xRIR_simpor_8_shot'
    root.mkdir(parents=True)
    result = lib(handler_replay(root, 'TERM'), {'EXP11_TEST_ROOTS': '1'})
    assert 'CONTINUED_AFTER_HANDLER' not in result.stdout, (
        'the TERM handler returned and execution resumed')
    assert 'PROMOTE' not in result.stdout
    assert result.returncode == 143, result.stdout[-400:]
    assert 'SIGNAL TERM' in result.stdout


def test_the_int_handler_terminates_too(tmp_path):
    root = tmp_path / 'xRIR_simpor_8_shot'
    root.mkdir(parents=True)
    result = lib(handler_replay(root, 'INT'), {'EXP11_TEST_ROOTS': '1'})
    assert 'CONTINUED_AFTER_HANDLER' not in result.stdout
    assert result.returncode == 130, result.stdout[-400:]


def test_the_handler_releases_the_arm_it_was_publishing(tmp_path):
    """Whatever the handler does, the arm must be free once the launcher is gone."""
    root = tmp_path / 'xRIR_simpor_8_shot'
    root.mkdir(parents=True)
    result = lib(handler_replay(root, 'TERM'), {'EXP11_TEST_ROOTS': '1'})
    assert result.returncode == 143
    assert free(root / LOCKFILE)


def test_the_protected_steps_refuse_once_a_handler_has_fired():
    """A guard, not only an exit: nothing downstream may run after an interruption."""
    text = (REPO / 'tools/exp11_launch.sh').read_text()
    assert 'INTERRUPTED' in text, (
        'the launcher needs a guard the protected steps check, so finalization and '
        'promotion can never run after a signal handler fired')
    result = lib('INTERRUPTED=1\nif not_interrupted; then echo REACHED; else echo REFUSED; fi\n')
    assert 'REFUSED' in result.stdout and 'REACHED' not in result.stdout
    ok = lib('INTERRUPTED=0\nif not_interrupted; then echo REACHED; fi\n')
    assert 'REACHED' in ok.stdout


# --- close review 7 blocker 1: the liveness scan belongs UNDER the lock --------------


def live_stub(pidfile, seconds=30):
    """A process this test starts, registered the way the trainer registers itself."""
    stub = subprocess.Popen(['sleep', str(seconds)])
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text('{}\n'.format(stub.pid))   # one pid, the grammar
    return stub


def test_recovery_refuses_while_any_attempt_of_the_arm_is_live(tmp_path):
    """Publication is an arm-wide decision, not a directory-wide one.

    Promoting an older attempt while a trainer of the same arm is still running leaves
    ``final`` pointing at one run while another writes the next. The check that used to
    guard this looked only at the directory being recovered.
    """
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    running = old.parent / 'attempt_20260927T111111'
    stub = live_stub(running / 'train.pid')
    try:
        status, out, err = recovery(old, 'H', tmp_path)
        assert status == 2, out
        assert 'PROMOTE' not in out and 'EXP11_FINALIZE' not in out
        assert 'refusing' in err and running.name in err
    finally:
        stub.terminate()
        stub.wait(timeout=30)


def test_the_scan_happens_under_the_lock_not_before_it(tmp_path):
    """Close review 7's interleaving: a scan that ran before the lock can go stale.

    A recovery is paused at a barrier before it takes the lock. While it waits, a
    trainer of another attempt appears and the launcher that started it dies. When the
    recovery resumes it must take the lock, scan the arm **under** it, and refuse --
    a scan taken before the pause would have seen a quiet arm.
    """
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    barrier = tmp_path / 'barrier'
    barrier.write_text('wait\n')
    env = {'EXP11_PRETRAIN_ROOT': str(tmp_path), 'EXP11_TEST_ROOTS': '1',
           'EXP11_SCAN_BARRIER': str(barrier)}
    out_file, err_file = tmp_path / 'r.out', tmp_path / 'r.err'
    with open(out_file, 'w') as o, open(err_file, 'w') as e:
        proc = subprocess.Popen(
            ['bash', 'tools/exp11_launch.sh', 'finalize', '--arm', 'H', '--gpu', '1',
             '--reviewed-commit', COMMIT, '--attempt', str(old), '--log', 'L.log',
             '--child-exit', '0', '--dry-run'],
            cwd=str(REPO), stdout=o, stderr=e,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', **env))
        stub = None
        try:
            for _ in range(200):                  # the launcher tells us it is waiting
                if (tmp_path / 'barrier.at').exists():
                    break
                time.sleep(0.05)
            else:
                proc.wait(timeout=30)
                pytest.fail('the launcher never reached a scan barrier, so the '
                            'interleaving cannot be forced open: {}'.format(
                                out_file.read_text()[-400:]))
            running = old.parent / 'attempt_20260927T111111'
            stub = live_stub(running / 'train.pid')   # its launcher is already gone
            barrier.unlink()
            assert proc.wait(timeout=60) == 2, out_file.read_text()[-400:]
            assert 'PROMOTE' not in out_file.read_text()
            assert 'refusing' in err_file.read_text()
        finally:
            if stub is not None:
                stub.terminate()
                stub.wait(timeout=30)
            if proc.poll() is None:
                barrier.unlink(missing_ok=True)
                proc.wait(timeout=60)


# --- close review 7 blocker 2: the window the frozen lifecycle cannot close ----------
# run_child forks the detached trainer and writes child.pid afterwards. A launcher that
# dies in between leaves a dead launch.pid, no child.pid and a live trainer that no scan
# could see. exp_11 closes that window from both sides without touching the frozen file:
# an intent marker the launcher writes before the fork, and the trainer's own pid file.


def unregistered_launch(root, tmp_path, gate, register_child=False):
    """Replay run_child's prologue and die before child.pid.

    The stub stands in for the trainer and registers itself the way the trainer does --
    but only once ``gate`` appears, so the test can look at the arm during the window
    where the trainer is alive and nothing at all records it yet.
    """
    library = (REPO / 'tools/exp06_launch.sh').read_text()
    assert SINK_LINE in library and CHILD_LINE in library
    attempt = root / 'attempt_20260927T222222'
    stub = ('while [ ! -e "$2" ]; do sleep 0.05; done; '
            'printf "%s\\n" $$ > "$1/train.pid"; sleep 20')
    script = ('ARM_ROOT={root}\nDRY=0\nARM=H\n'
              'hold_arm_lock full || exit 2\n'
              'attempt={attempt}\nmkdir -p -- "$attempt"\n'
              'own_launch "$attempt"\n'
              # defensive: the marker is what this regression is asking for
              'type mark_launching >/dev/null 2>&1 && mark_launching "$attempt" || true\n'
              'log="$attempt/child.log"\n: > "$log"\n'
              'pipe="$attempt/child.pipe"\nmkfifo -m 600 -- "$pipe"\n'
              'cat >> "$log" < "$pipe" &\nsink=$!\n'
              'nohup setsid bash -c \'{stub}\' _ "$attempt" "{gate}" > "$pipe" 2>&1 &\n'
              'child=$!\n{child_pid}'
              'handler="$(trap -p TERM | sed -n "s/^trap -- \'\\(.*\\)\' SIGTERM$/\\1/p")"\n'
              'eval "$handler"\necho CONTINUED\n').format(
                  root=root, attempt=attempt, stub=stub, gate=gate,
                  child_pid=('printf %s\\n "$child" > "$attempt/child.pid"\n'
                             if register_child else ''))
    status, out, err = detached_lib(script, tmp_path, {'EXP11_TEST_ROOTS': '1'})
    assert status == 143, out[-400:] + err[-400:]
    return attempt


def test_a_launch_that_died_before_anything_recorded_it_blocks_the_arm(tmp_path):
    """Close review 7's boundary, at its narrowest.

    The trainer is running, the launcher is gone, ``child.pid`` was never written and
    the trainer has not yet written its own pid file. Nothing in the arm names the live
    process -- which is why the launcher must declare the *intent* to launch before the
    fork, and why that declaration must block recovery of the whole arm.
    """
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    gate = tmp_path / 'let-the-stub-register'
    attempt = unregistered_launch(old.parent, tmp_path, gate)
    try:
        assert not (attempt / 'child.pid').exists()
        assert not (attempt / 'train.pid').exists(), 'the window must still be open'
        status, out, err = recovery(old, 'H', tmp_path)
        assert status == 2, out
        assert 'PROMOTE' not in out and 'refusing' in err
        assert attempt.name in err and 'unresolved' in err
    finally:
        gate.touch()
        drain(attempt / 'child.pipe')


def test_a_launch_that_recorded_its_child_blocks_the_arm_too(tmp_path):
    """The other side of the boundary: child.pid written, the trainer still running."""
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    gate = tmp_path / 'let-the-stub-register'
    gate.touch()                                   # this stub registers immediately
    attempt = unregistered_launch(old.parent, tmp_path, gate, register_child=True)
    try:
        assert (attempt / 'child.pid').is_file()
        status, out, err = recovery(old, 'H', tmp_path)
        assert status == 2, out
        assert 'PROMOTE' not in out and 'live trainer' in err
    finally:
        drain(attempt / 'child.pipe')


# --- close review 8 blocker 1: a child.pid is a registration only when it has a pid ---


@pytest.mark.parametrize('content', ['', 'SIGNAL TERM\n', '\n', 'not-a-pid\n'])
def test_an_incomplete_child_pid_leaves_the_launch_unresolved(tmp_path, content):
    """The file existing is not the registration; the pid in it is.

    The shell creates ``child.pid`` by redirection before anything is written into it,
    so a crash in that instant leaves a file with no pid -- no liveness to find, and,
    if its mere existence counted, no marker either. The arm would read as quiet with a
    trainer still running.
    """
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    stalled = old.parent / 'attempt_20260927T333333'
    stalled.mkdir()
    (stalled / 'launching').write_text('launcher 1\nat 2026-09-27T00:00:00+00:00\n')
    (stalled / 'child.pid').write_text(content)
    status, out, err = recovery(old, 'H', tmp_path)
    assert status == 2, out
    assert 'PROMOTE' not in out
    assert 'unresolved' in err and stalled.name in err
    assert (stalled / 'launching').is_file(), 'the marker must survive'


def test_a_complete_child_pid_of_a_dead_trainer_resolves_the_launch(tmp_path):
    """The other side: a real pid that is simply gone is a finished launch."""
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    finished = old.parent / 'attempt_20260927T333333'
    finished.mkdir()
    (finished / 'child.pid').write_text('{}\n'.format(2 ** 22 - 1))   # long gone
    status, out, err = recovery(old, 'H', tmp_path)
    assert status == 0, err[-500:]
    assert 'PROMOTE' in out


@pytest.mark.parametrize('content,cleared', [('4321\n', True), ('', False),
                                             ('SIGNAL TERM\n', False)])
def test_the_marker_is_withdrawn_only_against_a_real_registration(tmp_path, content,
                                                                  cleared):
    """A complete child.pid is necessary; since close review 10 it is not sufficient,
    so the finished trainer's train.exit stands beside it here."""
    attempt = tmp_path / 'xRIR_simpor_8_shot' / 'attempt_20260927T444444'
    attempt.mkdir(parents=True)
    (attempt / 'launching').write_text('launcher 1\n')
    (attempt / 'train.exit').write_text('train.exit 0\n')
    (attempt / 'child.pid').write_text(content)
    result = lib('clear_launching {}\n'.format(attempt))
    assert result.returncode == 0, result.stderr[-300:]
    assert (attempt / 'launching').is_file() is not cleared


# --- close review 8 blocker 3: the resolution is confined to the arm it locked -------


def unresolved_attempt(root, name='attempt_20260927T555555'):
    attempt = root / name
    attempt.mkdir(parents=True)
    (attempt / 'launching').write_text('launcher 1\nat 2026-09-27T00:00:00+00:00\n')
    return attempt


def test_the_resolution_refuses_an_attempt_of_another_arm(tmp_path):
    """H holds H's lock and scanned H; it may not retire anything of I.

    The lock and the scan are per arm, so a target outside the selected arm root is
    reached with no exclusion and no liveness answer behind it at all.
    """
    theirs = unresolved_attempt(tmp_path / 'xRIR_simpor_yawaug_8_shot')
    mine = tmp_path / 'xRIR_simpor_8_shot'
    mine.mkdir(parents=True)
    result = lib('ARM_ROOT={root}\nDRY=0\nARM=H\nUNRESOLVED_GRACE_S=0\n'
                 'resolve_unregistered {target}\n'.format(root=mine, target=theirs))
    assert result.returncode != 0
    assert 'arm' in result.stderr.lower() and 'refusing' in result.stderr
    assert theirs.is_dir() and (theirs / 'launching').is_file()
    assert not list(theirs.parent.glob('*_ABORTED_*'))


@pytest.mark.parametrize('name', ['attempt', 'attempt_../escape', 'final',
                                  'attempt_20260927T000000_ABORTED_unregistered'])
def test_the_resolution_refuses_a_name_that_is_not_an_attempts(tmp_path, name):
    root = tmp_path / 'xRIR_simpor_8_shot'
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    (target / 'launching').write_text('launcher 1\n')
    result = lib('ARM_ROOT={root}\nDRY=0\nARM=H\nUNRESOLVED_GRACE_S=0\n'
                 'resolve_unregistered {target}\n'.format(root=root, target=target))
    assert result.returncode != 0 and 'refusing' in result.stderr
    assert target.is_dir()


def test_the_resolution_accepts_the_same_directory_named_differently(tmp_path):
    """An absolute target under a relative root is the same attempt; strings are not.

    ``resolve_unregistered`` is handed paths by an operator, so identity has to be the
    canonical directory, never the spelling.
    """
    root = tmp_path / 'xRIR_simpor_8_shot'
    attempt = unresolved_attempt(root)
    relative = os.path.relpath(str(root), str(REPO))
    result = lib('ARM_ROOT={root}\nDRY=0\nARM=H\nUNRESOLVED_GRACE_S=0\n'
                 'resolve_unregistered {target}\n'.format(root=relative, target=attempt))
    assert result.returncode == 0, result.stderr[-500:]
    assert not attempt.exists()
    assert (root / (attempt.name + '_ABORTED_unregistered')).is_dir()


def test_the_resolution_does_not_need_an_args_file(tmp_path):
    """A launch that died before the trainer wrote args.json is the normal case."""
    root = tmp_path / 'xRIR_simpor_8_shot'
    attempt = unresolved_attempt(root)
    assert not (attempt / 'args.json').exists()
    result = lib('ARM_ROOT={root}\nDRY=0\nARM=H\nUNRESOLVED_GRACE_S=0\n'
                 'resolve_unregistered {target}\n'.format(root=root, target=attempt))
    assert result.returncode == 0, result.stderr[-500:]


# --- close review 8 blocker 2: the resolution leaves a permanent tombstone -----------


def test_the_resolution_writes_a_tombstone_before_it_retires_the_attempt(tmp_path):
    """The directory can be renamed; the answer must outlive the name.

    A trainer that wakes after the resolution has to be able to find out that its
    launch was retired, and a rename alone tells it nothing it can read.
    """
    root = tmp_path / 'xRIR_simpor_8_shot'
    attempt = unresolved_attempt(root)
    result = lib('ARM_ROOT={root}\nDRY=0\nARM=H\nUNRESOLVED_GRACE_S=0\n'
                 'resolve_unregistered {target}\n'.format(root=root, target=attempt))
    assert result.returncode == 0, result.stderr[-500:]
    tombstone = root / (attempt.name + '.resolved')
    assert tombstone.is_file(), 'the answer must survive the rename'
    # It has to say what was actually CHECKED -- the locks, the quiet scan, the grace,
    # the absence of a receipt -- and nothing more. Not "no complete child.pid" (the
    # attempt may carry a complete, dead wrapper record) and not "no train.exit", which
    # this resolution does not require (close reviews 11 and 12, wording).
    written = tombstone.read_text()
    assert 'resolved-by' in written and 'unresolved launch' in written
    assert 'no recorded pid of this arm alive' in written
    assert 'no completion receipt' in written and 'never registered' in written
    assert 'train.exit' not in written, 'it does not check that, so it may not claim it'
    assert not attempt.exists()
    assert (root / (attempt.name + '_ABORTED_unregistered')).is_dir()


def test_the_scan_ignores_a_tombstone(tmp_path):
    """It is a record, not a claim: an arm full of answered launches is still quiet."""
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    (old.parent / 'attempt_20260927T666666.resolved').write_text('launcher 1\n')
    status, out, err = recovery(old, 'H', tmp_path)
    assert status == 0, err[-500:]
    assert 'PROMOTE' in out


def test_a_trainer_that_wakes_after_the_resolution_trains_nothing(tmp_path):
    """The whole path, end to end, with the trainer's own registration in the stub.

    The launcher dies before ``child.pid``; the trainer has not woken yet, so nothing
    names it. The operator waits out the grace and resolves the launch. Only then does
    the trainer start -- and it must find its directory gone, refuse with exit 3,
    create nothing, and leave the arm open for the next launch.
    """
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    root = old.parent
    gate = tmp_path / 'wake-the-trainer'
    attempt = root / 'attempt_20260927T777777'
    # Double quotes inside: the whole stub travels through a single-quoted shell word.
    stub = ('import os, sys, time; sys.path.insert(0, "{repo}");'
            'from tools.exp11_train import register_trainer;'
            '[time.sleep(0.05) for _ in iter(lambda: os.path.exists("{gate}"), True)];'
            'register_trainer("{attempt}"); time.sleep(20)').format(
                repo=str(REPO), gate=str(gate), attempt=str(attempt))
    script = ('ARM_ROOT={root}\nDRY=0\nARM=H\n'
              'hold_arm_lock full || exit 2\n'
              'attempt={attempt}\nmkdir -p -- "$attempt"\n'
              'own_launch "$attempt"\nmark_launching "$attempt"\n'
              'log="$attempt/child.log"\n: > "$log"\n'
              'pipe="$attempt/child.pipe"\nmkfifo -m 600 -- "$pipe"\n'
              'cat >> "$log" < "$pipe" &\n'
              'nohup setsid {python} -c \'{stub}\' > "$pipe" 2>&1 &\n'
              'handler="$(trap -p TERM | sed -n "s/^trap -- \'\\(.*\\)\' SIGTERM$/\\1/p")"\n'
              'eval "$handler"\n').format(root=root, attempt=attempt,
                                           python=PYTHON, stub=stub)
    status, out, err = detached_lib(script, tmp_path, {'EXP11_TEST_ROOTS': '1'})
    assert status == 143, out[-300:] + err[-300:]
    assert not (attempt / 'child.pid').exists() and not (attempt / 'train.pid').exists()

    assert recovery(old, 'H', tmp_path)[0] == 2, 'the arm is closed by the marker'
    old_enough = time.time() - 10_000                       # age the marker past a grace
    os.utime(str(attempt / 'launching'), (old_enough, old_enough))
    result = resolve(attempt, grace='3600')
    assert result.returncode == 0, result.stderr[-500:]
    assert (root / (attempt.name + '.resolved')).is_file()

    gate.touch()                                            # only now does it wake
    log = root / (attempt.name + '_ABORTED_unregistered/child.log')
    for _ in range(400):
        if 'refusing' in log.read_text():
            break
        time.sleep(0.05)
    assert 'refusing' in log.read_text(), (
        'the trainer ran on instead of failing closed: {!r}'.format(log.read_text()))
    assert attempt.name in log.read_text(), 'the refusal names the attempt'
    assert not attempt.exists(), 'a refused trainer may not recreate its attempt'
    assert not (root / (attempt.name + '_ABORTED_unregistered/train.pid')).exists()
    status, out, err = recovery(old, 'H', tmp_path)          # the arm is open again
    assert status == 0, err[-500:]
    assert 'PROMOTE' in out


# --- close review 9 blocker 1: one pid-record grammar, read the same way twice ------
# A pid file says exactly one thing: ^[0-9]{1,10}\n?$. Two readers that disagree about
# what the bytes mean disagree about whether a trainer exists -- which is how
# `99999991\n99999992\n` could count as a registration while nothing in it was ever
# probed for liveness.

PID_RECORDS = [
    ('1457170', True),                    # no trailing newline: a pid all the same
    ('1457170\n', True),                  # the ordinary shape
    ('\n1457170\n', False),               # a leading blank line is not a pid
    ('99999991\n99999992\n', False),      # two records are not one
    (' 1457170\n', False),                # leading space
    ('1457170\r\n', False),               # CRLF is not our line ending
    ('', False),
    ('1457170\n\n', False),               # one trailing newline at most
    ('12345678901\n', False),             # eleven digits is not a pid
    ('1234567890\n', True),               # ten is the cap
]


@pytest.mark.parametrize('content,valid', PID_RECORDS)
def test_the_launcher_reads_one_pid_record_grammar(tmp_path, content, valid):
    """`pid_record` is the single definition both readers use."""
    pidfile = tmp_path / 'child.pid'
    pidfile.write_text(content)
    result = lib('if pid_record {f}; then echo " RECORD_OK"; else echo RECORD_BAD; fi\n'
                 'if registration_complete {d}; then echo COMPLETE; else echo PARTIAL; fi\n'
                 .format(f=pidfile, d=tmp_path))
    assert result.returncode == 0, result.stderr[-400:]
    assert ('RECORD_OK' in result.stdout) is valid, (content, result.stdout)
    assert ('COMPLETE' in result.stdout) is valid, (content, result.stdout)
    if valid:
        assert result.stdout.split()[0] == content.strip()


def test_the_two_readers_never_disagree(tmp_path):
    """Liveness and completeness are asked of the same bytes, by the same grammar."""
    pidfile = tmp_path / 'child.pid'
    pidfile.write_text('99999991\n99999992\n')     # two dead pids, one file
    result = lib('if registration_complete {d}; then echo COMPLETE; else echo PARTIAL; fi\n'
                 'if pid_alive {f}; then echo ALIVE; else echo GONE; fi\n'
                 .format(d=tmp_path, f=pidfile))
    assert 'PARTIAL' in result.stdout, (
        'a file that is not one pid may not count as a registration')


def test_junk_in_child_pid_keeps_the_marker(tmp_path):
    """The arm-wide consequence: an unreadable child.pid is not a registration."""
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    stalled = old.parent / 'attempt_20260927T888888'
    stalled.mkdir()
    (stalled / 'launching').write_text('launcher 1\n')
    (stalled / 'child.pid').write_text('99999991\n99999992\n')
    status, out, err = recovery(old, 'H', tmp_path)
    assert status == 2, out
    assert 'unresolved' in err and stalled.name in err
    assert (stalled / 'launching').is_file()


# --- close review 9 blocker 2: the resolver yields to a registration in flight ------

REGISTRATION_LOCK = '.registration.lock'


def test_the_resolution_refuses_while_a_registration_holds_the_lock(tmp_path):
    """The reviewer's schedule: a registration is mid-flight, so nothing may be retired.

    A trainer that has taken the registration lock is between its checks and its write.
    If the resolver went ahead, the rename would land in that window and the write would
    reach a directory the arm no longer knows.
    """
    root = tmp_path / 'xRIR_simpor_8_shot'
    attempt = unresolved_attempt(root)
    holder = flock_holder(root / REGISTRATION_LOCK)
    try:
        result = resolve(attempt)
        assert result.returncode != 0
        assert 'registration in progress' in result.stderr, result.stderr[-400:]
        assert attempt.is_dir() and (attempt / 'launching').is_file()
        assert not (root / (attempt.name + '.resolved')).exists()
    finally:
        holder.terminate()
        holder.wait(timeout=30)
    result = resolve(attempt)                       # released: the resolution proceeds
    assert result.returncode == 0, result.stderr[-400:]


def test_a_registration_completes_in_its_own_directory(tmp_path):
    """The other half of that schedule: the registration lands where it started."""
    root = tmp_path / 'xRIR_simpor_8_shot'
    attempt = unresolved_attempt(root)
    from tools import exp11_train
    path = exp11_train.register_trainer(str(attempt))
    assert path == attempt / 'train.pid'
    assert exp11_train.pid_record(path) == os.getpid()
    assert attempt.is_dir(), 'the directory it registered in is the one it checked'


def test_a_registration_after_a_resolution_refuses_on_the_tombstone(tmp_path):
    """The mirror schedule: the resolver went first, so the trainer finds the answer."""
    root = tmp_path / 'xRIR_simpor_8_shot'
    attempt = unresolved_attempt(root)
    assert resolve(attempt).returncode == 0
    from tools import exp11_train
    with pytest.raises(SystemExit) as exit_request:
        exp11_train.register_trainer(str(attempt))
    assert exit_request.value.code == 3
    assert not (attempt.parent / (attempt.name + '_ABORTED_unregistered/train.pid')).exists()


def test_the_scans_do_not_take_the_registration_lock(tmp_path):
    """A recovery must not be blocked by a registration; the marker already covers it."""
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    holder = flock_holder(old.parent / REGISTRATION_LOCK)
    try:
        status, out, err = recovery(old, 'H', tmp_path)
        assert status == 0, err[-400:]
        assert 'PROMOTE' in out
    finally:
        holder.terminate()
        holder.wait(timeout=30)
    text = (REPO / 'tools/exp11_launch.sh').read_text()
    assert 'scans do not take it: an' in text, 'say why in the launcher'


# --- close review 10 blocker 2: the wrapper's pid is not the trainer's ---------------
# Full mode runs the trainer under GNU `timeout`, and the frozen lifecycle records the
# WRAPPER's pid in child.pid. A complete-but-dead child.pid therefore says nothing about
# the trainer; only `train.pid` does, and only `train.exit` says it finished.


def launched(root, name='attempt_20260927T999999', child_pid=None, train_pid=None,
             train_exit=None, launching=True):
    """An attempt in a named state of the launch lifecycle."""
    attempt = root / name
    attempt.mkdir(parents=True, exist_ok=True)
    if launching:
        (attempt / 'launching').write_text('launcher 1\nat 2026-09-27T00:00:00+00:00\n')
    for filename, value in (('child.pid', child_pid), ('train.pid', train_pid),
                            ('train.exit', train_exit)):
        if value is not None:
            (attempt / filename).write_text(value)
    return attempt


DEAD = '4194303\n'


def test_a_dead_wrapper_pid_does_not_account_for_the_trainer(tmp_path):
    """The reviewer's wrapper loss, in its narrowest form.

    `timeout` exited and its pid is what child.pid holds. The trainer may still be
    running and has not registered. Nothing here is alive, and the old rule called that
    quiet -- publishing over a trainer nobody could see.
    """
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    stalled = launched(old.parent, child_pid=DEAD)        # complete, dead, no train.pid
    status, out, err = recovery(old, 'H', tmp_path)
    assert status == 2, out
    assert 'PROMOTE' not in out and 'unresolved' in err
    assert stalled.name in err
    assert (stalled / 'launching').is_file()


def test_a_trainer_that_left_no_exit_leaves_the_arm_unresolved(tmp_path):
    """It registered, then vanished: registering is not finishing."""
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    stalled = launched(old.parent, child_pid=DEAD, train_pid=DEAD)
    status, out, err = recovery(old, 'H', tmp_path)
    assert status == 2, out
    assert 'unresolved' in err and stalled.name in err


def test_a_finished_trainer_leaves_the_arm_quiet(tmp_path):
    """child.pid complete, train.pid dead, train.exit written: that launch is over."""
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    launched(old.parent, child_pid=DEAD, train_pid=DEAD, train_exit='train.exit 0\n')
    status, out, err = recovery(old, 'H', tmp_path)
    assert status == 0, err[-500:]
    assert 'PROMOTE' in out


@pytest.mark.parametrize('kwargs,cleared', [
    (dict(child_pid=DEAD, train_exit='train.exit 0\n'), True),      # the ordinary run
    (dict(child_pid=DEAD), False),                                   # no trainer record
    (dict(child_pid=DEAD, train_pid=DEAD), False),                   # never finished
    (dict(child_pid='', train_exit='train.exit 0\n'), False),        # no wrapper record
])
def test_the_marker_is_withdrawn_only_when_the_trainer_finished(tmp_path, kwargs, cleared):
    attempt = launched(tmp_path / 'xRIR_simpor_8_shot', **kwargs)
    result = lib('clear_launching {}\n'.format(attempt))
    assert result.returncode == 0, result.stderr[-300:]
    assert (attempt / 'launching').is_file() is not cleared


# --- close review 10 blocker 2: the wrapper is not the trainer -----------------------
# The three schedules below run the real thing: the sourced library, the real FIFO sink,
# a real GNU `timeout` wrapper, the real publication and registration locks, and a stub
# trainer that registers through the real ``register_trainer``.

STUB_TRAINER = """import os, sys, time
sys.path.insert(0, {repo!r})
from tools.exp11_train import record_trainer_exit, register_trainer

attempt, gate, exit_gate, receipt = sys.argv[1:5]
while not os.path.exists(gate):          # paused OUTSIDE the registration
    time.sleep(0.05)
path = register_trainer(attempt)         # the real one: locks, checks, atomic write
while not os.path.exists(exit_gate):     # paused with train.pid naming a live process
    time.sleep(0.05)
if receipt == '1':
    record_trainer_exit(path, 0)
"""


def stub_trainer(tmp_path):
    path = tmp_path / 'stub_trainer.py'
    path.write_text(STUB_TRAINER.format(repo=str(REPO)))
    return path


def lifecycle_launch(attempt, tmp_path, gate, exit_gate, receipt=True, die=False,
                     where=None):
    """``run_child``'s shape around a stub trainer, under a real `timeout` wrapper.

    `die` replays the TERM handler immediately after ``child.pid`` -- the instant of the
    reviewer's schedule, where the launcher is lost with its child still running.
    Otherwise the launcher waits for the wrapper and withdraws the marker exactly where
    full mode does, after ``run_child`` returns.
    """
    tail = ('handler="$(trap -p TERM | sed -n "s/^trap -- \'\\(.*\\)\' SIGTERM$/\\1/p")"\n'
            'eval "$handler"\n' if die else
            'wait "$child" || echo "CHILD_EXIT $?"\n'
            'clear_launching "$attempt"\necho CONTINUED\n')
    script = ('ARM_ROOT={root}\nDRY=0\nARM=H\n'
              'hold_arm_lock full || exit 2\n'
              'attempt={attempt}\nmkdir -p -- "$attempt"\n'
              'own_launch "$attempt"\nmark_launching "$attempt"\n'
              'log="$attempt/child.log"\n: > "$log"\n'
              'pipe="$attempt/child.pipe"\nmkfifo -m 600 -- "$pipe"\n'
              'cat >> "$log" < "$pipe" &\n'
              'nohup setsid timeout 120 {python} {stub} "$attempt" {gate} {exit_gate}'
              ' {receipt} > "$pipe" 2>&1 &\n'
              'child=$!\nprintf "%s\\n" "$child" > "$attempt/child.pid"\n' + tail
              ).format(root=attempt.parent, attempt=attempt, python=PYTHON,
                       stub=stub_trainer(tmp_path), gate=gate, exit_gate=exit_gate,
                       receipt=1 if receipt else 0)
    status, out, err = detached_lib(script, where or tmp_path, {'EXP11_TEST_ROOTS': '1'})
    assert status == (143 if die else 0), out[-400:] + err[-400:]
    return status, out, err


def until(condition, seconds=90):
    """Wait for a real process to get somewhere -- generously: the stub imports torch."""
    for _ in range(int(seconds / 0.05)):
        if condition():
            return True
        time.sleep(0.05)
    return condition()


def scan(root):
    """The one scan both modes run, through the sourced library."""
    return lib('ARM_ROOT={root}\nDRY=0\nARM=H\nrequire_quiet_arm\n'.format(root=root),
               {'EXP11_TEST_ROOTS': '1'})


def kill_our_wrapper(pidfile, stub):
    """KILL the `timeout` this test started -- and nothing else.

    KILL, not TERM: `timeout` forwards a TERM to the process it manages, and the point
    of the schedule is the wrapper dying while the trainer lives on. The pid is checked
    against the command line this test built before anything is sent to it.
    """
    pid = pid_of(pidfile)
    assert pid is not None, 'the wrapper pid was recorded and is alive'
    cmdline = Path('/proc/{}/cmdline'.format(pid)).read_bytes()
    assert b'timeout' in cmdline and str(stub).encode() in cmdline, cmdline
    os.kill(pid, signal.SIGKILL)
    assert until(lambda: pid_of(pidfile) is None), 'the wrapper is gone'


def test_the_wrapper_loss_schedule_end_to_end(tmp_path):
    """The reviewer's schedule, end to end.

    `child.pid` holds the WRAPPER's pid. Kill the wrapper and lose the launcher while
    the trainer is still short of its registration, and the attempt used to read as a
    finished launch: a complete, dead child.pid and no train.pid at all. It must read as
    unresolved, and it must stay readable that way until the trainer itself is
    accounted for.
    """
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    root = old.parent
    attempt = root / 'attempt_20260927T121212'
    gate, exit_gate = tmp_path / 'register-now', tmp_path / 'exit-now'
    lifecycle_launch(attempt, tmp_path, gate, exit_gate, die=True)
    try:
        kill_our_wrapper(attempt / 'child.pid', stub_trainer(tmp_path))
        assert (attempt / 'child.pid').is_file() and not (attempt / 'train.pid').exists()

        result = scan(root)                       # the scan both modes run
        assert result.returncode == 1, result.stdout
        assert 'unresolved' in result.stderr and attempt.name in result.stderr
        status, out, err = recovery(old, 'H', tmp_path)      # and through finalize mode
        assert status == 2 and 'unresolved' in err and 'PROMOTE' not in out

        gate.touch()                              # the trainer registers and stays alive
        assert until(lambda: (attempt / 'train.pid').is_file())
        result = scan(root)
        assert result.returncode == 1 and 'live trainer (train.pid)' in result.stderr

        exit_gate.touch()                         # and now it finishes, saying so
        assert until(lambda: (attempt / 'train.exit').is_file())
        assert until(lambda: pid_of(attempt / 'train.pid') is None)
        old_enough = time.time() - 10_000
        os.utime(str(attempt / 'launching'), (old_enough, old_enough))
        result = resolve(attempt, grace='3600')
        assert result.returncode == 0, result.stderr[-600:]
        assert (root / (attempt.name + '.resolved')).is_file()
        assert (root / (attempt.name + '_ABORTED_unregistered')).is_dir()
        status, out, err = recovery(old, 'H', tmp_path)      # the arm is open again
        assert status == 0, err[-600:]
        assert 'PROMOTE' in out
    finally:
        gate.touch()
        exit_gate.touch()
        for pipe in (attempt / 'child.pipe',
                     root / (attempt.name + '_ABORTED_unregistered/child.pipe')):
            try:
                drain(pipe)
            except OSError:          # already drained: the sink saw EOF and left
                pass


def test_an_ordinary_run_still_withdraws_its_marker(tmp_path):
    """The other side: a trainer that registers and finishes leaves a quiet arm."""
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    attempt = old.parent / 'attempt_20260927T131313'
    gate, exit_gate = tmp_path / 'register-now', tmp_path / 'exit-now'
    gate.touch()
    exit_gate.touch()
    status, out, err = lifecycle_launch(attempt, tmp_path, gate, exit_gate)
    assert 'CONTINUED' in out and 'CHILD_EXIT' not in out, out + err[-400:]
    assert (attempt / 'train.pid').is_file() and (attempt / 'train.exit').is_file()
    assert not (attempt / 'launching').exists(), 'the marker was withdrawn'
    assert scan(old.parent).returncode == 0
    status, out, err = recovery(old, 'H', tmp_path)
    assert status == 0, err[-600:]


def test_a_trainer_that_never_reported_its_exit_keeps_the_arm_shut(tmp_path):
    """A trainer that dies without a receipt is exactly what the marker is for.

    The launcher completed, so the wrapper's pid is recorded and dead; but nothing says
    the trainer finished, so the attempt is unresolved and the next launch refuses until
    an operator answers it.
    """
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    root = old.parent
    attempt = root / 'attempt_20260927T141414'
    gate, exit_gate = tmp_path / 'register-now', tmp_path / 'exit-now'
    gate.touch()
    exit_gate.touch()
    lifecycle_launch(attempt, tmp_path, gate, exit_gate, receipt=False)
    assert (attempt / 'train.pid').is_file() and not (attempt / 'train.exit').exists()
    assert (attempt / 'launching').is_file(), 'the marker stands'
    result = scan(root)
    assert result.returncode == 1 and 'unresolved' in result.stderr
    status, out, err = recovery(old, 'H', tmp_path)
    assert status == 2 and 'unresolved' in err

    old_enough = time.time() - 10_000
    os.utime(str(attempt / 'launching'), (old_enough, old_enough))
    result = resolve(attempt, grace='3600')
    assert result.returncode == 0, result.stderr[-600:]
    assert scan(root).returncode == 0
    status, out, err = recovery(old, 'H', tmp_path)
    assert status == 0, err[-600:]


# --- close review 11: a reader that cannot run decides nothing --------------------
# Exit 1 was both "this file holds no record" and "I could not run at all". Read as the
# first, a broken reader makes every pid in the arm look dead -- and a resolution, which
# deliberately ignores its own target's marker, then tombstones and renames the
# directory of a REGISTERED, RUNNING trainer. These regressions break the reader on
# purpose and demand that nothing is decided.

needs_a_non_root_user = pytest.mark.skipif(
    os.geteuid() == 0, reason='root ignores the permission bits these cases turn off')

READERS = {
    # The reader cannot run at all: this is what a missing interpreter, an unimportable
    # module and a crash all look like from the shell -- exit 1, nothing on stdout.
    'cannot run': 'exit 1',
    # It answers, but the same way whatever it is asked: only the probes catch these.
    'always record': 'echo "record 1"; exit 0',
    # A verdict line whose pid is not a pid: the shell must not hand `kill -0` a word
    # and read the failure as "not alive".
    'nonsense pid': 'echo "record not-a-pid"; exit 0',
    'always norecord': 'echo norecord; exit 0',
}


def reader(tmp_path, kind, name='python_wrapper.sh'):
    """A ``$PYTHON`` that breaks ONLY the pid reader.

    Everything else -- the lock holder above all -- keeps the real interpreter, so the
    publication lock is taken normally and the reader fails afterwards, which is exactly
    the schedule the reviewer reproduced.
    """
    path = tmp_path / name
    if kind == 'import error':                 # a real ImportError of the real module
        fake = tmp_path / 'shadow'
        (fake / 'tools').mkdir(parents=True, exist_ok=True)
        (fake / 'tools/__init__.py').write_text('')
        (fake / 'tools/exp11_pidrecord.py').write_text(
            'raise ImportError("this reader is broken on purpose")\n')
        body = 'cd "{}" && exec "{}" "$@"'.format(fake, PYTHON)
    elif kind == 'fails on the trainer':       # healthy on the probes, blind to train.pid
        body = 'case "$*" in *train.pid*) exit 1 ;; esac\n    exec "{}" "$@"'.format(PYTHON)
    elif kind == 'nonsense on the trainer':    # healthy on the probes, a verdict of gibberish
        body = ('case "$*" in *train.pid*) echo "record not-a-pid"; exit 0 ;; esac\n'
                '    exec "{}" "$@"').format(PYTHON)
    else:
        body = READERS[kind]
    path.write_text('#!/bin/sh\ncase " $* " in\n  *" tools.exp11_pidrecord "*)\n'
                    '    {}\n    ;;\nesac\nexec "{}" "$@"\n'.format(body, PYTHON))
    path.chmod(0o755)
    return path


# --- close review 12: the bytes the reader printed, not what Bash made of them -----
# Command substitution strips trailing newlines and NUL bytes, so a reader emitting
# `norecord\n\n` or `norecord\0\n` -- exit 0, and healthy on both probes -- became a
# definite verdict and retired a live trainer's attempt.

RAW_ANSWERS = [b'norecord\n\n', b'norecord\x00\n', b'record 123\n\n',
               b'record 123\x00\n', b'record 123 \n', b'\nrecord 123\n']
RAW_IDS = [repr(payload) for payload in RAW_ANSWERS]


def byte_reader(tmp_path, payload, only_on=None, name='byte_wrapper.sh'):
    """A ``$PYTHON`` whose pid reader prints exactly these bytes.

    With ``only_on`` it answers normally everywhere else, so the health probes pass and
    what is on trial is the per-read validation; without it every read -- the probes
    included -- gets the payload.
    """
    answer = tmp_path / (name + '.bytes')
    answer.write_bytes(payload)
    body = 'cat "{}"; exit 0'.format(answer)
    if only_on is not None:
        body = 'case "$*" in *{}*) {} ;; esac\n    exec "{}" "$@"'.format(
            only_on, body, PYTHON)
    path = tmp_path / name
    path.write_text('#!/bin/sh\ncase " $* " in\n  *" tools.exp11_pidrecord "*)\n'
                    '    {}\n    ;;\nesac\nexec "{}" "$@"\n'.format(body, PYTHON))
    path.chmod(0o755)
    return path


def registered_live_attempt(tmp_path, name='attempt_20260927T161616'):
    """The reviewer's state: a marker, a complete DEAD wrapper pid, a live trainer."""
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    attempt = old.parent / name
    attempt.mkdir()
    (attempt / 'launching').write_text('launcher 1\nat 2026-09-27T00:00:00+00:00\n')
    (attempt / 'child.pid').write_text(DEAD)          # the timeout wrapper, long gone
    stub = live_stub(attempt / 'train.pid')           # the trainer, alive and registered
    stale = time.time() - 10_000
    os.utime(str(attempt / 'launching'), (stale, stale))
    return old, attempt, stub


def resolve_with_reader(attempt, broken, grace='3600'):
    """Take the publication lock with a working reader, then break it, then resolve."""
    return lib('ARM_ROOT={root}\nDRY=0\nARM=H\nUNRESOLVED_GRACE_S={grace}\n'
               'hold_arm_lock finalize || exit 9\n'
               'PYTHON={broken}\n'
               'resolve_unregistered {attempt}\n'.format(
                   root=attempt.parent, grace=grace, broken=broken, attempt=attempt),
               {'EXP11_TEST_ROOTS': '1'})


def scan_with_reader(root, broken, mode='full'):
    """The scan both modes run, with the reader broken after the lock is taken."""
    return lib('ARM_ROOT={root}\nDRY=0\nARM=H\n'
               'hold_arm_lock {mode} || exit 9\n'
               'PYTHON={broken}\n'
               'require_quiet_arm\n'.format(root=root, mode=mode, broken=broken),
               {'EXP11_TEST_ROOTS': '1'})


@pytest.mark.parametrize('kind', ['cannot run', 'import error', 'fails on the trainer',
                                  'nonsense pid', 'nonsense on the trainer'])
def test_a_broken_reader_never_retires_a_live_trainers_attempt(tmp_path, kind):
    """The reviewer's schedule: registered, running, and the reader breaks under the lock.

    The resolution skips its own target's marker by design -- the tombstone is what makes
    it safe -- so the ONLY thing standing between a running trainer and a retired
    directory is the liveness answer. A reader that cannot answer must stop the
    resolution, not be read as "nothing is alive".
    """
    old, attempt, stub = registered_live_attempt(tmp_path)
    try:
        result = resolve_with_reader(attempt, reader(tmp_path, kind))
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert 'unknown' in result.stderr or 'unhealthy' in result.stderr, result.stderr
        assert attempt.is_dir(), 'the live trainer keeps its directory'
        assert (attempt / 'launching').is_file(), 'and its marker'
        assert pid_of(attempt / 'train.pid') == stub.pid, 'and its registration'
        assert not (old.parent / (attempt.name + '.resolved')).exists(), 'no tombstone'
        assert not (old.parent / (attempt.name + '_ABORTED_unregistered')).exists()
        assert 'RESOLVED' not in result.stdout and 'ABORT' not in result.stdout
    finally:
        stub.terminate()
        stub.wait(timeout=30)


@pytest.mark.parametrize('mode', ['full', 'finalize'])
def test_a_broken_reader_stops_the_scan_of_either_mode(tmp_path, mode):
    old, attempt, stub = registered_live_attempt(tmp_path)
    try:
        result = scan_with_reader(old.parent, reader(tmp_path, 'fails on the trainer'),
                                  mode=mode)
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert 'liveness unknown' in result.stderr, result.stderr
        assert (attempt / 'launching').is_file() and attempt.is_dir()
    finally:
        stub.terminate()
        stub.wait(timeout=30)


def test_a_broken_reader_stops_a_recovery_through_the_cli(tmp_path):
    """The same through the launcher itself, which decides liveness with $PYTHON."""
    old, attempt, stub = registered_live_attempt(tmp_path)
    try:
        status, out, err = launch(
            ['finalize', '--arm', 'H', '--gpu', '1', '--reviewed-commit', COMMIT,
             '--attempt', str(old), '--log', 'L.log', '--child-exit', '0', '--dry-run'],
            {'EXP11_PRETRAIN_ROOT': str(tmp_path), 'EXP11_TEST_ROOTS': '1',
             'EXP11_PYTHON': str(reader(tmp_path, 'cannot run'))})
        assert status == 2, out
        assert 'PROMOTE' not in out
        assert 'unhealthy' in err or 'liveness unknown' in err, err
        assert attempt.is_dir() and (attempt / 'launching').is_file()
    finally:
        stub.terminate()
        stub.wait(timeout=30)


def test_the_interpreter_override_is_refused_outside_the_test_roots(tmp_path):
    """It exists for these regressions alone, and says so when anyone else tries it."""
    status, out, err = launch(
        ['finalize', '--arm', 'H', '--gpu', '1', '--reviewed-commit', COMMIT,
         '--attempt', str(tmp_path), '--log', 'L.log', '--child-exit', '0', '--dry-run'],
        {'EXP11_PYTHON': '/bin/false'})
    assert status == 2 and 'EXP11_PYTHON' in err


@pytest.mark.parametrize('kind,caught_by', [('always record', 'an empty file'),
                                            ('always norecord', 'a file holding 1')])
def test_the_health_check_catches_a_reader_that_answers_everything_the_same(
        tmp_path, kind, caught_by):
    """Two probes, because one answer is not a working reader.

    A reader stuck on `record 1` calls every pid file a registration; one stuck on
    `norecord` calls every trainer dead. Neither fails, so only asking it something
    whose answer is known catches them.
    """
    old, attempt, stub = registered_live_attempt(tmp_path)
    try:
        result = scan_with_reader(old.parent, reader(tmp_path, kind))
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert 'unhealthy' in result.stderr, result.stderr
        assert caught_by in result.stderr, result.stderr
    finally:
        stub.terminate()
        stub.wait(timeout=30)


def test_a_broken_reader_leaves_the_marker_standing(tmp_path):
    """clear_launching cannot withdraw a marker it cannot justify withdrawing."""
    attempt = tmp_path / 'xRIR_simpor_8_shot' / 'attempt_20260927T171717'
    attempt.mkdir(parents=True)
    (attempt / 'launching').write_text('launcher 1\n')
    (attempt / 'child.pid').write_text(DEAD)
    (attempt / 'train.exit').write_text('train.exit 0\n')
    result = lib('PYTHON={broken}\nclear_launching {attempt}\n'.format(
        broken=reader(tmp_path, 'cannot run'), attempt=attempt))
    assert result.returncode == 0, result.stderr[-300:]
    assert (attempt / 'launching').is_file(), 'unknown is not a reason to clear'


@pytest.mark.parametrize('payload', RAW_ANSWERS, ids=RAW_IDS)
def test_only_the_exact_verdict_bytes_are_a_verdict(tmp_path, payload):
    """Anything Bash would tidy into a verdict must be unknown instead."""
    old, attempt, stub = registered_live_attempt(tmp_path)
    try:
        broken = byte_reader(tmp_path, payload, only_on='train.pid')
        result = resolve_with_reader(attempt, broken)
        assert result.returncode == 2, (payload, result.stdout, result.stderr)
        assert 'unknown' in result.stderr or 'unhealthy' in result.stderr
        assert attempt.is_dir() and (attempt / 'launching').is_file()
        assert pid_of(attempt / 'train.pid') == stub.pid
        assert not (old.parent / (attempt.name + '.resolved')).exists()
        scan_result = scan_with_reader(old.parent, broken)
        assert scan_result.returncode == 2, scan_result.stdout
    finally:
        stub.terminate()
        stub.wait(timeout=30)


@pytest.mark.parametrize('payload', RAW_ANSWERS, ids=RAW_IDS)
def test_the_health_probes_are_byte_exact_too(tmp_path, payload):
    """The probes read the same way the real reads do, or they prove nothing."""
    old, attempt, stub = registered_live_attempt(tmp_path)
    try:
        result = scan_with_reader(old.parent, byte_reader(tmp_path, payload))
        assert result.returncode == 2, (payload, result.stdout, result.stderr)
        assert 'unhealthy' in result.stderr, result.stderr
    finally:
        stub.terminate()
        stub.wait(timeout=30)


def padding_reader(tmp_path):
    """A reader whose answers are always right and always a byte too long.

    It passes both probes under a check that compares what Bash made of the output,
    because command substitution deletes the padding -- which is the whole point.
    """
    path = tmp_path / 'padding_wrapper.sh'
    path.write_text('#!/bin/sh\ncase " $* " in\n  *" tools.exp11_pidrecord "*)\n'
                    '    answer="$("{python}" "$@")" || exit $?\n'
                    '    printf "%s\\n\\n" "$answer"\n    exit 0\n    ;;\nesac\n'
                    'exec "{python}" "$@"\n'.format(python=PYTHON))
    path.chmod(0o755)
    return path


def test_a_reader_that_pads_every_answer_is_unhealthy(tmp_path):
    """Both probe answers are correct *and* padded: only the raw bytes tell them apart."""
    old, attempt, stub = registered_live_attempt(tmp_path)
    try:
        result = scan_with_reader(old.parent, padding_reader(tmp_path))
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert 'unhealthy' in result.stderr, result.stderr
    finally:
        stub.terminate()
        stub.wait(timeout=30)


@needs_a_non_root_user
@pytest.mark.parametrize('shut_the', ['file', 'parent'])
def test_a_pid_file_the_launcher_cannot_read_is_never_dead(tmp_path, shut_the):
    """The reviewer's second schedule: chmod 000 on a registered live trainer's record.

    The file exists and may well name a living process; a reader that cannot open it has
    not established anything, least of all that the trainer is gone.
    """
    old, attempt, stub = registered_live_attempt(tmp_path)
    shut = (attempt / 'train.pid') if shut_the == 'file' else attempt
    shut.chmod(0o000)
    try:
        probe = lib('ARM_ROOT={root}\nDRY=0\nARM=H\n'
                    'status=0\npid_record {pid} || status=$?\necho "RECORD $status"\n'
                    'status=0\npid_alive {pid} || status=$?\necho "ALIVE $status"\n'
                    .format(root=old.parent, pid=attempt / 'train.pid'))
        assert 'RECORD 2' in probe.stdout and 'ALIVE 2' in probe.stdout, probe.stdout
        result = lib('ARM_ROOT={root}\nDRY=0\nARM=H\nUNRESOLVED_GRACE_S=3600\n'
                     'hold_arm_lock finalize || exit 9\n'
                     'resolve_unregistered {attempt}\n'.format(
                         root=old.parent, attempt=attempt), {'EXP11_TEST_ROOTS': '1'})
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert 'liveness unknown' in result.stderr, result.stderr
        assert not (old.parent / (attempt.name + '.resolved')).exists(), 'no tombstone'
        assert not (old.parent / (attempt.name + '_ABORTED_unregistered')).exists()
        scan_result = lib('ARM_ROOT={root}\nDRY=0\nARM=H\nrequire_quiet_arm\n'.format(
            root=old.parent), {'EXP11_TEST_ROOTS': '1'})
        assert scan_result.returncode == 2 and 'liveness unknown' in scan_result.stderr
    finally:
        shut.chmod(0o700 if shut_the == 'parent' else 0o600)
        stub.terminate()
        stub.wait(timeout=30)


def test_the_file_classification_does_not_depend_on_a_translated_word(tmp_path):
    """`stat -c %F` prints a phrase in the caller's language.

    "regular file" is `fichier régulier` under a French locale and something else again
    elsewhere, so a classification that matches that text calls a perfectly ordinary
    train.pid a directory -- and a live trainer's record then reads as no record. The
    mode bits are the same in every language.
    """
    text = (REPO / 'tools/exp11_launch.sh').read_text()
    assert '-c %F' not in text, 'the file type must be read from the mode, not a phrase'
    fifo = tmp_path / 'child.pid'
    os.mkfifo(str(fifo))
    result = lib('status=0\npid_record {fifo} || status=$?\necho "STATUS $status"\n'
                 .format(fifo=fifo))
    assert 'STATUS 1' in result.stdout, result.stdout   # definite, and it did not block


# --- close review 13: a root we cannot LIST hides every attempt in it ---------------
# A directory with mode 0300 is searchable and writable but not listable. The glob
# `"$root"/attempt_*` over it silently expands to nothing, the unmatched pattern fails
# `-d`, and the scan declared the arm quiet -- while both locks still acquired, because
# a lock file is a known pathname. A resolution then retired a registered live trainer.


def scan_under_lock(root, mode='full'):
    """``require_quiet_arm`` where each mode runs it: under the arm lock."""
    return lib('ARM_ROOT={root}\nDRY=0\nARM=H\n'
               'hold_arm_lock {mode} || exit 9\nrequire_quiet_arm\n'.format(
                   root=root, mode=mode), {'EXP11_TEST_ROOTS': '1'})


def resolve_under_lock(attempt, grace='3600'):
    return lib('ARM_ROOT={root}\nDRY=0\nARM=H\nUNRESOLVED_GRACE_S={grace}\n'
               'hold_arm_lock finalize || exit 9\n'
               'resolve_unregistered {attempt}\n'.format(
                   root=attempt.parent, grace=grace, attempt=attempt),
               {'EXP11_TEST_ROOTS': '1'})


def unlistable(root, mode):
    """Take away the right to LIST the arm, leaving the lock files reachable."""
    (root / '.publish.lock').touch()
    (root / '.registration.lock').touch()
    root.chmod(mode)


@needs_a_non_root_user
@pytest.mark.parametrize('mode,what', [(0o300, 'search and write, no read'),
                                       (0o100, 'search only')])
@pytest.mark.parametrize('scan_mode', ['full', 'finalize'])
def test_an_arm_root_that_cannot_be_listed_is_never_quiet(tmp_path, mode, what, scan_mode):
    """An attempt we cannot see is not an attempt that is not there."""
    old, attempt, stub = registered_live_attempt(tmp_path)
    root = old.parent
    unlistable(root, mode)
    try:
        result = scan_under_lock(root, mode=scan_mode)
        assert result.returncode == 2, (what, result.stdout, result.stderr)
        assert 'cannot be enumerated' in result.stderr, result.stderr
        assert 'PROMOTE' not in result.stdout
    finally:
        root.chmod(0o700)
        stub.terminate()
        stub.wait(timeout=30)


@needs_a_non_root_user
@pytest.mark.parametrize('mode', [0o300, 0o100])
def test_a_resolution_under_an_unlistable_root_retires_nothing(tmp_path, mode):
    """The reviewer's schedule: the live trainer's own attempt is the one at risk."""
    old, attempt, stub = registered_live_attempt(tmp_path)
    root = old.parent
    unlistable(root, mode)
    try:
        result = resolve_under_lock(attempt)
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert 'cannot be enumerated' in result.stderr, result.stderr
        assert 'RESOLVED' not in result.stdout and 'TOMBSTONE' not in result.stdout
    finally:
        root.chmod(0o700)
        assert attempt.is_dir() and (attempt / 'launching').is_file()
        assert pid_of(attempt / 'train.pid') == stub.pid
        assert not (root / (attempt.name + '.resolved')).exists()
        assert not (root / (attempt.name + '_ABORTED_unregistered')).exists()
        stub.terminate()
        stub.wait(timeout=30)


@needs_a_non_root_user
def test_an_unlistable_root_keeps_the_marker(tmp_path):
    """clear_launching withdraws protection on an understanding it no longer has."""
    old, attempt, stub = registered_live_attempt(tmp_path)
    (attempt / 'train.exit').write_text('train.exit 0\n')
    (attempt / 'child.pid').write_text('1\n')
    root = old.parent
    unlistable(root, 0o300)
    try:
        result = lib('ARM_ROOT={root}\nDRY=0\nARM=H\nclear_launching {attempt}\n'.format(
            root=root, attempt=attempt))
        assert result.returncode == 0, result.stderr[-300:]
    finally:
        root.chmod(0o700)
        assert (attempt / 'launching').is_file(), 'the marker stands'
        stub.terminate()
        stub.wait(timeout=30)


def test_an_empty_readable_root_is_quiet(tmp_path):
    """A listing that succeeded and found nothing is an answer: the arm IS quiet."""
    root = tmp_path / 'xRIR_simpor_8_shot'
    root.mkdir(parents=True)
    result = scan_under_lock(root)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert 'refusing' not in result.stderr


@needs_a_non_root_user
def test_an_entry_the_scan_cannot_stat_is_unknown(tmp_path):
    """The listing succeeded; one of the names in it cannot be looked at.

    A name that fails `-d` used to be skipped as "not a directory", which is a verdict
    about something nobody inspected. The arm holds nothing else, so the answer cannot
    come from another entry whichever order the listing arrives in.
    """
    root = tmp_path / 'xRIR_simpor_8_shot'
    root.mkdir(parents=True)
    shut = root / 'shut'
    (shut / 'real').mkdir(parents=True)
    (root / 'attempt_20260927T181818').symlink_to(shut / 'real')
    shut.chmod(0o000)
    try:
        result = scan_under_lock(root)
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert 'unknown' in result.stderr and 'attempt_20260927T181818' in result.stderr
    finally:
        shut.chmod(0o700)


def test_a_flood_of_verdict_bytes_is_unknown_and_never_materialised(tmp_path):
    """A rogue reader's output must never become a shell variable.

    A verdict is at most eighteen bytes; anything longer cannot match. The shell must
    therefore refuse it on the file's size and read at most a bounded prefix, rather
    than pulling megabytes through a command substitution to find that out.
    """
    text = (REPO / 'tools/exp11_launch.sh').read_text()
    assert 'head -c 64' in text, 'the digit extraction must read a bounded prefix'
    old, attempt, stub = registered_live_attempt(tmp_path)
    try:
        flood = b'record ' + b'1' * (2 * 1024 * 1024) + b'\n'
        result = resolve_with_reader(attempt, byte_reader(tmp_path, flood,
                                                         only_on='train.pid'))
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert attempt.is_dir() and (attempt / 'launching').is_file()
    finally:
        stub.terminate()
        stub.wait(timeout=30)


# --- close review 14: a root reached through a symlink, and guards that read a failed
# inspection as an absence. One family: nothing may conclude "not there" from "could not
# look".


def test_a_symlinked_arm_root_is_still_enumerated(tmp_path):
    """`find -P` does not descend a symlink given as its starting point.

    The `-d`, `-r` and `-x` checks all follow it, so the root looked fine and the
    listing came back empty and successful -- an arm with a live registered trainer in
    it, read as an arm with nothing in it.
    """
    old, attempt, stub = registered_live_attempt(tmp_path)
    link = tmp_path / 'arm-link'
    link.symlink_to(old.parent)
    try:
        for mode in ('full', 'finalize'):
            result = scan_under_lock(link, mode=mode)
            assert result.returncode == 1, (mode, result.stdout, result.stderr)
            assert 'live trainer (train.pid)' in result.stderr, result.stderr
        result = resolve_under_lock(link / attempt.name)
        assert result.returncode != 0, (result.stdout, result.stderr)
        assert 'RESOLVED' not in result.stdout and 'TOMBSTONE' not in result.stdout
        assert attempt.is_dir() and (attempt / 'launching').is_file()
        assert pid_of(attempt / 'train.pid') == stub.pid
        assert not (old.parent / (attempt.name + '.resolved')).exists()
        assert not (old.parent / (attempt.name + '_ABORTED_unregistered')).exists()
    finally:
        stub.terminate()
        stub.wait(timeout=30)


def test_a_dangling_symlink_root_is_unknown(tmp_path):
    link = tmp_path / 'arm-link'
    link.symlink_to(tmp_path / 'nothing-here')
    result = lib('ARM_ROOT={root}\nDRY=0\nARM=H\nrequire_quiet_arm\n'.format(root=link),
                 {'EXP11_TEST_ROOTS': '1'})
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert 'cannot be enumerated' in result.stderr


def behind_a_shut_door(tmp_path, name, content='x'):
    """A path that EXISTS and cannot be inspected: a symlink into a mode-000 directory."""
    shut = tmp_path / ('shut_' + name.replace('/', '_'))
    shut.mkdir()
    (shut / 'target').write_text(content)
    return shut, shut / 'target'


@needs_a_non_root_user
def test_a_marker_we_cannot_inspect_closes_the_arm(tmp_path):
    """`-f` says "no" for a marker that is there and cannot be reached.

    Every pid record of the attempt is definitely absent, so nothing else refuses: the
    marker is the only thing standing between this arm and a promotion, and it was read
    as missing because the test that looks for it failed.
    """
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    attempt = old.parent / 'attempt_20260927T191919'
    attempt.mkdir()
    shut, target = behind_a_shut_door(tmp_path, 'launching', 'launcher 1\n')
    (attempt / 'launching').symlink_to(target)
    shut.chmod(0o000)
    try:
        result = scan_under_lock(old.parent)
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert 'unknown' in result.stderr and 'launching' in result.stderr
    finally:
        shut.chmod(0o700)


@needs_a_non_root_user
def test_a_receipt_we_cannot_inspect_is_never_retired(tmp_path):
    """The resolution asserts "no completion receipt"; it may only assert what it saw."""
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    attempt = unresolved_attempt(old.parent)
    shut, target = behind_a_shut_door(tmp_path, 'completion', '{"run_type": "exp11_train"}')
    (attempt / 'completion.json').symlink_to(target)
    shut.chmod(0o000)
    try:
        result = resolve_under_lock(attempt, grace='0')
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert 'unknown' in result.stderr, result.stderr
        assert 'RESOLVED' not in result.stdout and 'TOMBSTONE' not in result.stdout
        assert attempt.is_dir() and (attempt / 'launching').is_file()
        assert not (old.parent / (attempt.name + '.resolved')).exists()
    finally:
        shut.chmod(0o700)


def unreachable_final(root, name='attempt_20260101T000000'):
    """`final` reaching an old attempt through a directory nobody may enter."""
    shut = root / 'shut'
    (shut / name).mkdir(parents=True)
    link = root / 'final'
    link.symlink_to(shut / name)
    shut.chmod(0o000)
    return shut


@needs_a_non_root_user
def test_a_final_we_cannot_resolve_is_never_replaced(tmp_path):
    """A lookup that failed is not "nothing is published"."""
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    shut = unreachable_final(old.parent)
    try:
        status, out, err = recovery(old, 'H', tmp_path)
        assert status == 2, out
        assert 'PROMOTE' not in out and 'REPLACING' not in out
        assert 'unknown' in err or 'cannot' in err, err
    finally:
        shut.chmod(0o700)


@needs_a_non_root_user
@pytest.mark.parametrize('guard', ['already_published', 'check_final_conflict',
                                   'abort_recovery'])
def test_the_publication_guards_refuse_an_unresolvable_final(tmp_path, guard):
    """None of the three may read a failed lookup as "not published"."""
    root = tmp_path / 'xRIR_simpor_8_shot'
    canonical = root / 'attempt_20260927T202020'
    canonical.mkdir(parents=True)
    shut = unreachable_final(root)
    arguments = '{} L.log'.format(canonical) if guard == 'abort_recovery' else str(canonical)
    try:
        result = lib('ARM_ROOT={root}\nDRY=1\nARM=H\nREPLACE_FINAL=0\n'
                     'status=0\n{guard} {arguments} || status=$?\necho "STATUS $status"\n'
                     .format(root=root, guard=guard, arguments=arguments),
                     {'EXP11_TEST_ROOTS': '1'})
        assert 'STATUS 2' in result.stdout, (result.stdout, result.stderr)
        assert 'ABORT' not in result.stdout and 'REPLACING' not in result.stdout
        assert canonical.is_dir()
    finally:
        shut.chmod(0o700)


def test_a_reader_that_never_returns_is_unknown(tmp_path):
    """A hanging reader is not an answer either -- and must not hang the launcher."""
    wrapper = tmp_path / 'sleeping_wrapper.sh'
    wrapper.write_text('#!/bin/sh\ncase " $* " in\n  *" tools.exp11_pidrecord "*)\n'
                       '    sleep 120\n    ;;\nesac\nexec "{}" "$@"\n'.format(PYTHON))
    wrapper.chmod(0o755)
    pidfile = tmp_path / 'train.pid'
    pidfile.write_text('1457170\n')
    body = ('set -euo pipefail\n'
            'EXP11_LAUNCH_LIB=1 source tools/exp11_launch.sh\n'
            'PYTHON={wrapper}\nstatus=0\npid_record {pidfile} || status=$?\n'
            'echo "STATUS $status"\n'.format(wrapper=wrapper, pidfile=pidfile))
    result = subprocess.run(['bash', '-c', body], cwd=str(REPO), capture_output=True,
                            text=True, timeout=30,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
                                     EXP11_READER_TIMEOUT_S='1'))
    assert 'STATUS 2' in result.stdout, (result.stdout, result.stderr)


# --- close review 15: bash's `stat` carries no errno --------------------------------
# ENAMETOOLONG, EIO and an injected failure all read as "no", and the shell called that a
# confirmed absence whenever the parent looked fine. The classification now comes from
# tools/exp11_pathprobe.py, and the shell validates its answer the way it validates the
# pid reader's -- so a probe that fails, pads or lies is unknown, not an absence.


def probe_breaker(tmp_path, only_on, name='probe_wrapper.sh', body='exit 1'):
    """A ``$PYTHON`` that breaks ONE path probe and delegates everything else."""
    path = tmp_path / name
    path.write_text('#!/bin/sh\ncase " $* " in\n  *" tools.exp11_pathprobe "*)\n'
                    '    case "$*" in *{only_on}*) {body} ;; esac\n    ;;\nesac\n'
                    'exec "{python}" "$@"\n'.format(only_on=only_on, body=body,
                                                    python=PYTHON))
    path.chmod(0o755)
    return path


def probe_states(root, path, python=None):
    """``probe_path`` and ``file_present`` on one path, through the sourced library."""
    return lib('ARM_ROOT={root}\nDRY=0\nARM=H\n{python}'
               'status=0\nprobe_path {path} || status=$?\necho "PROBE $status"\n'
               'status=0\nfile_present {path} || status=$?\necho "FILE $status"\n'.format(
                   root=root, path=path,
                   python='PYTHON={}\n'.format(python) if python else ''),
               {'EXP11_TEST_ROOTS': '1'})


def test_a_name_too_long_is_unknown_not_absent(tmp_path):
    """ENAMETOOLONG is not ENOENT, and the difference decides an arm's fate."""
    result = probe_states(tmp_path, tmp_path / ('x' * 300))
    assert 'PROBE 2' in result.stdout and 'FILE 2' in result.stdout, result.stdout


def test_a_missing_name_is_still_a_definite_absence(tmp_path):
    result = probe_states(tmp_path, tmp_path / 'nothing')
    assert 'PROBE 1' in result.stdout and 'FILE 1' in result.stdout, result.stdout


def test_a_broken_path_probe_closes_the_arm(tmp_path):
    """The marker's probe, and only it, cannot answer: the scan may not proceed."""
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    attempt = old.parent / 'attempt_20260927T212121'
    attempt.mkdir()
    (attempt / 'launching').write_text('launcher 1\n')
    result = scan_with_reader(old.parent, probe_breaker(tmp_path, '/launching'))
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert 'unknown' in result.stderr, result.stderr


def test_a_broken_receipt_probe_retires_nothing(tmp_path):
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    attempt = unresolved_attempt(old.parent)
    result = resolve_with_reader(attempt, probe_breaker(tmp_path, '/completion.json'),
                                 grace='0')
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert 'RESOLVED' not in result.stdout and attempt.is_dir()
    assert not (old.parent / (attempt.name + '.resolved')).exists()


@pytest.mark.parametrize('guard', ['already_published', 'check_final_conflict',
                                   'abort_recovery'])
def test_a_broken_final_probe_publishes_nothing(tmp_path, guard):
    root = tmp_path / 'xRIR_simpor_8_shot'
    canonical = root / 'attempt_20260927T222222'
    canonical.mkdir(parents=True)
    publish(canonical)
    arguments = '{} L.log'.format(canonical) if guard == 'abort_recovery' else str(canonical)
    result = lib('ARM_ROOT={root}\nDRY=1\nARM=H\nREPLACE_FINAL=0\n'
                 'PYTHON={python}\nstatus=0\n{guard} {arguments} || status=$?\n'
                 'echo "STATUS $status"\n'.format(
                     root=root, python=probe_breaker(tmp_path, '/final'),
                     guard=guard, arguments=arguments),
                 {'EXP11_TEST_ROOTS': '1'})
    assert 'STATUS 2' in result.stdout, (result.stdout, result.stderr)
    assert 'ABORT' not in result.stdout and 'REPLACING' not in result.stdout


@pytest.mark.parametrize('crafted', ['absent\\n\\n', 'present\\n', 'record 1\\n'])
def test_a_crafted_probe_verdict_is_not_a_verdict(tmp_path, crafted):
    """Padding, a verdict with no mode, another protocol's line: none of them decide."""
    breaker = probe_breaker(tmp_path, '/launching',
                            body="printf '{}'; exit 0".format(crafted))
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    attempt = old.parent / 'attempt_20260927T232323'
    attempt.mkdir()
    (attempt / 'launching').write_text('launcher 1\n')
    result = scan_with_reader(old.parent, breaker)
    assert result.returncode == 2, (crafted, result.stdout, result.stderr)


def test_a_dangling_train_pid_link_is_not_a_dead_trainer(tmp_path):
    """The reviewer's schedule: the record is a link, and its target has moved away.

    The resolution excludes its own target from the marker checks -- the tombstone is
    what makes that safe -- so the pid records are all that speak for the trainer. A
    reader that calls a dangling link "no record" calls a living trainer dead.
    """
    old, attempt, stub = registered_live_attempt(tmp_path)
    try:
        record = attempt / 'train.pid'
        moved = tmp_path / 'moved-away'
        record.replace(moved)
        record.symlink_to(moved)
        moved.unlink()                      # the name is there; its target is not
        result = resolve_under_lock(attempt)
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert 'RESOLVED' not in result.stdout and 'TOMBSTONE' not in result.stdout
        assert attempt.is_dir() and (attempt / 'launching').is_file()
        assert not (old.parent / (attempt.name + '.resolved')).exists()
        assert not (old.parent / (attempt.name + '_ABORTED_unregistered')).exists()
    finally:
        stub.terminate()
        stub.wait(timeout=30)


@pytest.mark.parametrize('value', ['0', '-1', '400', 'abc', '3s', ''])
def test_an_impossible_reader_timeout_is_refused(tmp_path, value):
    """It bounds every liveness question; it may not be turned off or stretched."""
    status, out, err = launch(
        ['finalize', '--arm', 'H', '--gpu', '1', '--reviewed-commit', COMMIT,
         '--attempt', str(tmp_path), '--log', 'L.log', '--child-exit', '0', '--dry-run'],
        {'EXP11_READER_TIMEOUT_S': value, 'EXP11_TEST_ROOTS': '1',
         'EXP11_PRETRAIN_ROOT': str(tmp_path)})   # never the real roots, whatever happens
    assert status == 2, out
    assert 'EXP11_READER_TIMEOUT_S' in err, err


# --- close review 16: a verdict line is not a verdict until its FIELDS are ----------
# `link garbage` passed as a published target, so a resolution retired the attempt
# `final` really pointed at; `present 41ed relative` passed as a marker's verdict and
# left the arm quiet. Every field is checked now: the mode is hex of a known file type,
# the path is absolute, normalized, printable and no longer than PATH_MAX, and a
# `present` verdict must be about the very name that was probed.

CRAFTED_VERDICTS = [
    'present',                           # a keyword with no mode
    'present 81a4 /forged/path',         # a field the protocol no longer has
    'present zzzz',                      # not hex
    'present f000',                      # hex, but no known file type
    'present ffffffff',                  # eight digits
    'present 81a4 ',                     # a trailing space is an extra field
    'absent yes',                        # an answer with something appended
    'link /forged/target',               # what `link` used to carry
    'same different',                    # two keywords
    'yes',                               # a word of another protocol
]
CRAFTED_IDS = [v[:28] for v in CRAFTED_VERDICTS]
SOME_CRAFTED = [CRAFTED_VERDICTS[1], CRAFTED_VERDICTS[3], CRAFTED_VERDICTS[8]]


def crafted_reader(tmp_path, only_on, verdict, name='crafted_wrapper.sh'):
    return probe_breaker(tmp_path, only_on, name=name,
                         body="printf '%s\\n' '{}'; exit 0".format(verdict))


@pytest.mark.parametrize('verdict', CRAFTED_VERDICTS, ids=CRAFTED_IDS)
def test_a_crafted_marker_verdict_leaves_the_arm_closed(tmp_path, verdict):
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    attempt = old.parent / 'attempt_20260928T010101'
    attempt.mkdir()
    (attempt / 'launching').write_text('launcher 1\n')
    result = scan_with_reader(old.parent, crafted_reader(tmp_path, '/launching', verdict))
    assert result.returncode == 2, (verdict, result.stdout, result.stderr)
    assert 'PROMOTE' not in result.stdout


@pytest.mark.parametrize('verdict', SOME_CRAFTED)
def test_a_crafted_receipt_verdict_retires_nothing(tmp_path, verdict):
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    attempt = unresolved_attempt(old.parent)
    result = resolve_with_reader(attempt, crafted_reader(tmp_path, '/completion.json',
                                                         verdict), grace='0')
    assert result.returncode == 2, (verdict, result.stdout, result.stderr)
    assert attempt.is_dir() and (attempt / 'launching').is_file()
    assert not (old.parent / (attempt.name + '.resolved')).exists()


def test_a_crafted_final_verdict_never_retires_the_published_attempt(tmp_path):
    """The reviewer's schedule: `final` really publishes the attempt being resolved."""
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    attempt = unresolved_attempt(old.parent)
    link = publish(attempt, completed=False)
    result = resolve_with_reader(attempt, crafted_reader(tmp_path, '/final', 'link garbage'),
                                 grace='0')
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert attempt.is_dir(), 'the published attempt is still there'
    assert os.path.realpath(str(link)) == str(attempt), 'and final still publishes it'
    assert not (old.parent / (attempt.name + '.resolved')).exists()


@pytest.mark.parametrize('verdict', SOME_CRAFTED)
@pytest.mark.parametrize('guard,replace', [('already_published', '0'),
                                           ('check_final_conflict', '0'),
                                           ('check_final_conflict', '1'),
                                           ('abort_recovery', '0')])
def test_a_crafted_final_verdict_publishes_nothing(tmp_path, verdict, guard, replace):
    root = tmp_path / 'xRIR_simpor_8_shot'
    canonical = root / 'attempt_20260928T020202'
    canonical.mkdir(parents=True)
    publish(canonical)
    arguments = '{} L.log'.format(canonical) if guard == 'abort_recovery' else str(canonical)
    result = lib('ARM_ROOT={root}\nDRY=1\nARM=H\nREPLACE_FINAL={replace}\n'
                 'PYTHON={python}\nstatus=0\n{guard} {arguments} || status=$?\n'
                 'echo "STATUS $status"\n'.format(
                     root=root, replace=replace, guard=guard, arguments=arguments,
                     python=crafted_reader(tmp_path, '/final', verdict)),
                 {'EXP11_TEST_ROOTS': '1'})
    assert 'STATUS 2' in result.stdout, (verdict, guard, result.stdout, result.stderr)
    assert 'ABORT' not in result.stdout and 'REPLACING' not in result.stdout
    assert canonical.is_dir() and (root / 'final').is_symlink()


def test_an_overflowing_reader_timeout_is_refused(tmp_path):
    """A 36-digit bound errors bash's comparison, and an errored test is false."""
    status, out, err = launch(
        ['finalize', '--arm', 'H', '--gpu', '1', '--reviewed-commit', COMMIT,
         '--attempt', str(tmp_path), '--log', 'L.log', '--child-exit', '0', '--dry-run'],
        {'EXP11_READER_TIMEOUT_S': '9' * 36, 'EXP11_TEST_ROOTS': '1',
         'EXP11_PRETRAIN_ROOT': str(tmp_path)})
    assert status == 2, out
    assert 'EXP11_READER_TIMEOUT_S' in err, err


def test_an_odd_spelling_of_an_attempt_resolves_normally(tmp_path):
    """A path is not a string to be parsed: `//`, `/./` and a trailing slash are spellings."""
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    attempt = unresolved_attempt(old.parent)
    stale = time.time() - 10_000
    os.utime(str(attempt / 'launching'), (stale, stale))
    spelling = '{}//{}/'.format(old.parent, attempt.name)
    result = lib('ARM_ROOT={root}\nDRY=0\nARM=H\nUNRESOLVED_GRACE_S=0\n'
                 'hold_arm_lock finalize || exit 9\n'
                 'resolve_unregistered {spelling}\n'.format(root=old.parent,
                                                             spelling=spelling),
                 {'EXP11_TEST_ROOTS': '1'})
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert not attempt.exists()
    assert (old.parent / (attempt.name + '_ABORTED_unregistered')).is_dir()
    assert (old.parent / (attempt.name + '.resolved')).is_file()


def test_a_published_final_is_recognised_through_any_spelling(tmp_path):
    """`already_published` asks an identity question, so the spelling cannot matter."""
    root = tmp_path / 'xRIR_simpor_8_shot'
    canonical = root / 'attempt_20260928T030303'
    canonical.mkdir(parents=True)
    publish(canonical)
    for spelling in ('{root}//{name}', '{root}/./{name}', '{root}/{name}/'):
        result = lib('ARM_ROOT={root}\nDRY=1\nARM=H\nstatus=0\n'
                     'already_published {path} || status=$?\necho "STATUS $status"\n'
                     .format(root=root,
                             path=spelling.format(root=root, name=canonical.name)),
                     {'EXP11_TEST_ROOTS': '1'})
        assert 'STATUS 0' in result.stdout, (spelling, result.stdout, result.stderr)


def pid_of(pidfile):
    """The pid in a registration file, if it still names a living process."""
    try:
        pid = int(pidfile.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    try:
        os.kill(pid, 0)          # a probe, never a signal: signal 0 delivers nothing
    except OSError:
        return None
    return pid


def resolve(attempt, grace='0', dry=0):
    """``resolve_unregistered`` through the sourced library, as the mutation tests do.

    EXP11_PRETRAIN_ROOT is dry-run-only on purpose, and this resolution really renames a
    directory, so the real roots are never involved: the library is handed the tmp root.
    """
    return lib('ARM_ROOT={root}\nDRY={dry}\nARM=H\nUNRESOLVED_GRACE_S={grace}\n'
               'resolve_unregistered {attempt}\n'.format(
                   root=attempt.parent, dry=dry, grace=grace, attempt=attempt))


def test_an_unresolved_launch_is_resolved_only_once_nothing_is_alive(tmp_path):
    """The resolution is an operator's answer, not a timeout the launcher takes itself.

    While the trainer still runs, the marker may not be cleared at any age; while it is
    younger than the grace, a trainer that has not registered yet is still possible.
    Only when both are settled does the attempt go aside -- and it goes aside *aborted*,
    because without a child.pid it has no exit receipt and can never be published.
    """
    old = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    gate = tmp_path / 'let-the-stub-register'
    attempt = unregistered_launch(old.parent, tmp_path, gate)
    try:
        gate.touch()                                   # the stub registers and runs on
        for _ in range(200):
            if (attempt / 'train.pid').exists():
                break
            time.sleep(0.05)
        result = resolve(attempt)
        assert result.returncode != 0 and 'live trainer' in result.stderr
        assert attempt.is_dir()
    finally:
        drain(attempt / 'child.pipe')
    for _ in range(600):                               # let the stub finish on its own
        if pid_of(attempt / 'train.pid') is None:
            break
        time.sleep(0.05)
    result = resolve(attempt, grace='100000')
    assert result.returncode != 0 and 's old' in result.stderr
    assert attempt.is_dir(), 'nothing moves inside the grace'
    result = resolve(attempt)
    assert result.returncode == 0, result.stderr[-500:]
    assert not attempt.exists()
    assert (attempt.parent / (attempt.name + '_ABORTED_unregistered')).is_dir()
    status, out, err = recovery(old, 'H', tmp_path)     # the arm is open again
    assert status == 0, err[-500:]
    assert 'PROMOTE' in out


def test_resolving_refuses_an_attempt_that_is_not_unresolved(tmp_path):
    """What may never be retired this way.

    No longer "child.pid is complete": that is the `timeout` wrapper's pid, and a
    wrapper lost while its trainer was unregistered is exactly the state the operator
    has to be able to answer (close review 10, blocker 2). A run that really finished is
    another matter -- a completion receipt, or the arm's published `final`.
    """
    attempt = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    result = resolve(attempt)
    assert result.returncode != 0 and 'no launching marker' in result.stderr
    (attempt / 'launching').write_text('launcher 1\n')
    (attempt / 'child.pid').write_text('1\n')
    (attempt / 'completion.json').write_text('{"run_type": "exp11_train"}')
    result = resolve(attempt)
    assert result.returncode != 0 and 'completion receipt' in result.stderr
    assert attempt.is_dir()
    (attempt / 'completion.json').unlink()
    publish(attempt, completed=False)
    result = resolve(attempt)
    assert result.returncode != 0 and 'published as' in result.stderr
    assert attempt.is_dir()


def test_the_launcher_takes_the_resolution_only_as_an_explicit_request(tmp_path):
    """The flag exists on finalize, and a dry run announces without moving anything."""
    attempt = attempt_with('xRIR_simpor_8_shot', 'H', 'H_RECIPE', tmp_path)
    (attempt / 'launching').write_text('launcher 1\nat 2026-09-27T00:00:00+00:00\n')
    status, out, err = launch(
        ['finalize', '--arm', 'H', '--gpu', '1', '--reviewed-commit', COMMIT,
         '--resolve-unregistered', str(attempt), '--dry-run'],
        {'EXP11_PRETRAIN_ROOT': str(tmp_path), 'EXP11_TEST_ROOTS': '1',
         'EXP11_UNRESOLVED_GRACE_S': '0'})
    assert status == 0, err[-500:]
    assert 'ABORT {} -> {}_ABORTED_unregistered'.format(attempt, attempt) in out
    assert 'RESOLVED' in out and 'PROMOTE' not in out
    assert attempt.is_dir(), 'a dry run moves nothing'
    assert 'LOCKED' in out, 'the resolution happens under the arm lock'


# --- close review 6: the lock may never reach the log sink or the training child ----

SINK_LINE = 'cat >> "$log" < "$pipe" &'
CHILD_LINE = 'nohup setsid "$@" > "$pipe" 2>&1 &'


def lifecycle(root, launch_child):
    """Replay the frozen ``run_child`` prologue and then take the registered TERM body.

    ``run_child`` waits for its child, so no launcher can be made to exit in the middle
    of it without sending a real signal. The two lines that matter are asserted to be
    exactly the library's, so this replay cannot drift from what production does.
    """
    library = (REPO / 'tools/exp06_launch.sh').read_text()
    assert SINK_LINE in library and CHILD_LINE in library, (
        'the frozen lifecycle changed; this replay no longer reproduces it')
    child = ('nohup setsid sleep 8 > "$pipe" 2>&1 &\nchild=$!\n'
             'printf %s\\n "$child" > "$attempt/child.pid"\n') if launch_child else ''
    return ('ARM_ROOT={root}\nDRY=0\nARM=H\n'
            'hold_arm_lock finalize || exit 2\n'
            'attempt={root}/attempt_20260927T000000\nmkdir -p -- "$attempt"\n'
            'log="$attempt/child.log"\n: > "$log"\n'
            'pipe="$attempt/child.pipe"\nmkfifo -m 600 -- "$pipe"\n'
            'cat >> "$log" < "$pipe" &\nsink=$!\n'
            'sleep 0.3\n' + child +
            'handler="$(trap -p TERM | sed -n "s/^trap -- \'\\(.*\\)\' SIGTERM$/\\1/p")"\n'
            '[ -n "$handler" ] || {{ echo NO_TERM_TRAP; exit 9; }}\n'
            'eval "$handler"\necho CONTINUED\n').format(root=root)


def free(lockfile, seconds=5.0):
    """True once an independent process can take the lock, within ``seconds``."""
    deadline = time.time() + seconds
    while True:
        if subprocess.run(['flock', '-n', str(lockfile), 'true']).returncode == 0:
            return True
        if time.time() >= deadline:
            return False
        time.sleep(0.05)


def drain(pipe):
    """Let an orphaned sink see EOF: open the FIFO for writing and close it."""
    if pipe.exists():
        os.close(os.open(str(pipe), os.O_WRONLY | os.O_NONBLOCK))


def detached_lib(script, where, env=None):
    """``lib``, but with the output on files.

    The sink and the detached child inherit the launcher's stderr, so a captured pipe
    would only reach EOF once those orphans ended -- which is the very thing under test.
    """
    out, err = where / 'launcher.out', where / 'launcher.err'
    body = ('set -euo pipefail\n'
            'EXP11_LAUNCH_LIB=1 source tools/exp11_launch.sh\n' + script)
    with open(out, 'w') as o, open(err, 'w') as e:
        status = subprocess.run(['bash', '-c', body], cwd=str(REPO), stdout=o, stderr=e,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
                                         **(env or {}))).returncode
    return status, out.read_text(), err.read_text()


@pytest.mark.parametrize('launch_child', [False, True])
def test_no_orphan_of_the_launcher_keeps_the_arm_locked(tmp_path, launch_child):
    """Close review 6's boundary: the sink must not inherit the lock.

    Interrupted after the sink started -- with the child not yet launched, and again
    with it launched and detached -- the launcher exits 143 and the arm must be free
    again at once. An orphaned sink waits on its FIFO for a writer that never comes, so
    a lock it inherited would never be released at all.
    """
    root = tmp_path / 'xRIR_simpor_8_shot'
    root.mkdir(parents=True)
    status, out, err = detached_lib(lifecycle(root, launch_child), tmp_path,
                                    {'EXP11_TEST_ROOTS': '1'})
    assert status == 143, out[-400:] + err[-400:]
    assert 'CONTINUED' not in out
    pipe = root / 'attempt_20260927T000000/child.pipe'
    try:
        assert free(root / LOCKFILE), (
            'an orphan of the launcher still holds the arm lock')
    finally:
        drain(pipe)


def test_the_launcher_signals_nothing_that_is_not_its_own_holder(tmp_path):
    """A recorded pid is not a licence to signal it.

    ``HOLDER_PID`` comes out of the holder's own announcement; if that holder has since
    died, the number can belong to anybody. Releasing the lock must therefore check that
    the process is still this launcher's child before it sends anything. Here it is a
    bystander this test started, and it has to survive.
    """
    bystander = subprocess.Popen(['sleep', '30'])
    try:
        result = lib('HOLDER_PID={}\ndrop_arm_lock\necho RELEASED\n'.format(bystander.pid))
        assert 'RELEASED' in result.stdout, result.stderr[-400:]
        time.sleep(0.5)
        assert bystander.poll() is None, (
            'the launcher signalled a process that was never its holder')
    finally:
        bystander.terminate()
        bystander.wait(timeout=30)


def test_a_holder_that_never_started_is_not_reported_as_someone_else(tmp_path):
    """A refusal has to name the real obstacle.

    "held by another invocation" sends a reader looking for a publication that is not
    happening. A holder that could not start at all -- an unwritable root, a lock path
    that is a directory, a missing interpreter -- is a different failure and says so.
    """
    root = tmp_path / 'xRIR_simpor_8_shot'
    (root / LOCKFILE).mkdir(parents=True)     # a directory where the lock file belongs
    result = lib('ARM_ROOT={}\nDRY=0\nARM=H\nhold_arm_lock finalize\n'.format(root),
                 {'EXP11_TEST_ROOTS': '1'})
    assert result.returncode != 0
    assert 'another invocation' not in result.stderr, result.stderr[-400:]
    assert 'holder' in result.stderr.lower() and 'refusing' in result.stderr


def test_the_lock_file_is_a_file_and_the_launcher_knows_no_stale_lock():
    text = (REPO / 'tools/exp11_launch.sh').read_text()
    for gone in ('break_lock', 'take_lock', 'release_lock', 'release_breaking',
                 '--break-lock', 'BREAK_LOCK', 'EXP11_LOCK_BARRIER',
                 'EXP11_ACQUIRE_BARRIER', 'LOCK_GRACE', '.breaking', 'BREAKLOCK',
                 'LOCK_NONCE', 'take_lock'):
        assert gone not in text, '{} survived the replacement'.format(gone)
    assert 'LOCK_FD' not in text, (
        'the launcher must not hold the lock on a descriptor of its own: the sink and '
        'the child inherit every one of them')
    assert 'exp11_lock_holder.py' in text
    assert 'flock' in (REPO / 'tools/exp11_lock_holder.py').read_text()
    assert 'inherited by the trainer or the log sink' in text
    usage = subprocess.run(['bash', 'tools/exp11_launch.sh'], cwd=str(REPO),
                           capture_output=True, text=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    assert 'break-lock' not in usage.stderr


def test_the_dry_run_announces_the_lock_and_takes_none_outside_the_test_roots():
    status, out, _ = launch(['full', '--arm', 'H', '--gpu', '1', '--reviewed-commit',
                             COMMIT, '--dry-run'])
    assert status == 0
    assert 'LOCK ckpt/exp11/pretrain/xRIR_simpor_8_shot/.publish.lock' in out
    assert not (REPO / 'ckpt/exp11/pretrain/xRIR_simpor_8_shot/.publish.lock').exists()
