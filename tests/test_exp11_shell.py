"""Golden dry runs of exp_11's launcher and HAA pipeline (plan v3 items 10 and 12).

Every argv the two shells would issue is frozen in
``orientation_cue_fairness_results_assets/golden/``. A deliberate change regenerates the
files in the same commit; an accidental one -- a flag, a path, a run type, a cue -- fails
here. The refusals (unknown init, malformed job, unknown arm) are checked directly.
"""
import json
import os
from pathlib import Path
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


def test_the_lock_file_is_a_file_and_the_launcher_knows_no_stale_lock():
    text = (REPO / 'tools/exp11_launch.sh').read_text()
    for gone in ('break_lock', 'take_lock', 'release_lock', 'release_breaking',
                 '--break-lock', 'BREAK_LOCK', 'EXP11_LOCK_BARRIER',
                 'EXP11_ACQUIRE_BARRIER', 'LOCK_GRACE', '.breaking', 'BREAKLOCK',
                 'LOCK_NONCE', 'take_lock'):
        assert gone not in text, '{} survived the replacement'.format(gone)
    assert 'flock' in text and 'a lock vanishes with its holder' in text
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
