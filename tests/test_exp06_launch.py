"""Launch preflight and the dry-run argv of tools/exp06_launch.sh."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import exp06_finalize

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo(tmp_path):
    """A small real repository: preflight reads git state, never the sources."""
    root = tmp_path / 'repo'
    (root / 'worklog').mkdir(parents=True)
    (root / 'tools').mkdir()
    (root / 'tools/exp06_launch.sh').write_text('#!/usr/bin/env bash\n')
    (root / 'worklog/notes.md').write_text('notebook\n')
    for command in (['init', '-q'], ['add', '-A'], ['-c', 'user.email=a@b', '-c', 'user.name=t',
                                                    'commit', '-q', '-m', 'initial']):
        subprocess.run(['git'] + command, cwd=root, check=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    return root, head


@pytest.fixture
def fake_nvidia_smi(tmp_path, monkeypatch):
    """A fake nvidia-smi on PATH reports whatever compute apps a test writes."""
    binary = tmp_path / 'bin'
    binary.mkdir()
    script = binary / 'nvidia-smi'
    script.write_text('#!/usr/bin/env bash\ncat "$(dirname "$0")/apps.txt"\n')
    script.chmod(0o755)
    monkeypatch.setenv('PATH', str(binary) + os.pathsep + os.environ['PATH'])

    def apps(text):
        (binary / 'apps.txt').write_text(text)

    apps('')
    return apps


def test_preflight_accepts_a_clean_tree_at_the_reviewed_commit(repo, fake_nvidia_smi):
    root, head = repo
    record = exp06_finalize.preflight('full', 1, head, attempt_root=root.parent / 'a', repo=root)
    assert record['reviewed_commit'] == head and record['gpu'] == 1
    assert record['gpu_compute_apps'] == [] and record['live_launches'] == []
    assert record['git_state']['HEAD'] == head and record['mode'] == 'full'
    (root / 'worklog/notes.md').write_text('edited notebook\n')
    assert exp06_finalize.preflight('smoke', 0, head, repo=root)['mode'] == 'smoke'


def test_preflight_refuses_a_wrong_commit_or_a_dirty_tree(repo, fake_nvidia_smi):
    root, head = repo
    with pytest.raises(ValueError, match='reviewed commit'):
        exp06_finalize.preflight('full', 1, 'a' * 40, repo=root)
    with pytest.raises(ValueError, match='reviewed commit'):
        exp06_finalize.preflight('full', 1, head[:12], repo=root)
    (root / 'tools/exp06_launch.sh').write_text('#!/usr/bin/env bash\n# edited\n')
    with pytest.raises(ValueError, match='dirty'):
        exp06_finalize.preflight('full', 1, head, repo=root)


def test_preflight_refuses_a_busy_gpu_only_where_it_matters(repo, fake_nvidia_smi):
    root, head = repo
    fake_nvidia_smi('3141\n')
    for mode in ('full', 'probe'):
        with pytest.raises(ValueError, match='GPU'):
            exp06_finalize.preflight(mode, 1, head, repo=root)
    assert exp06_finalize.preflight('smoke', 1, head, repo=root)['gpu_compute_apps'] is None
    fake_nvidia_smi('')
    assert exp06_finalize.preflight('full', 1, head, repo=root)['gpu_compute_apps'] == []


def test_preflight_refuses_a_live_or_unreadable_launch_pid(repo, fake_nvidia_smi):
    root, head = repo
    attempts = root.parent / 'attempts'  # ckpt/ lives outside the tracked tree
    (attempts / 'attempt_20260916T130000').mkdir(parents=True)
    (attempts / 'attempt_20260916T130000/launch.pid').write_text(str(os.getpid()) + '\n')
    with pytest.raises(ValueError, match='launch'):
        exp06_finalize.preflight('full', 1, head, attempt_root=attempts, repo=root)
    (attempts / 'attempt_20260916T130000/launch.pid').write_text('not-a-pid\n')
    with pytest.raises(ValueError, match='launch.pid'):
        exp06_finalize.preflight('full', 1, head, attempt_root=attempts, repo=root)
    (attempts / 'attempt_20260916T130000/launch.pid').write_text('999999999\n')
    assert exp06_finalize.preflight('full', 1, head, attempt_root=attempts, repo=root)


def test_preflight_cli_exits_two_on_refusal(repo, fake_nvidia_smi):
    root, head = repo
    command = [sys.executable, 'tools/exp06_finalize.py', 'preflight', '--mode', 'full',
               '--gpu', '1', '--reviewed-commit', head, '--repo', str(root)]
    env = {**os.environ, 'PYTHONPATH': str(REPO)}
    ok = subprocess.run(command, cwd=REPO, capture_output=True, text=True, env=env)
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout.split('EXP06_PREFLIGHT_OK ')[1])['mode'] == 'full'
    wrong = subprocess.run(command[:-4] + ['--reviewed-commit', 'b' * 40, '--repo', str(root)],
                           cwd=REPO, capture_output=True, text=True, env=env)
    assert wrong.returncode == 2 and 'EXP06_PREFLIGHT_REFUSED' in wrong.stderr


LAUNCHER = 'tools/exp06_launch.sh'
PYTHON = '/home/yixunhu/miniconda3/envs/xRIR/bin/python'
ROOT = 'ckpt/exp06/pretrain/xRIR_cylor_8_shot'
ATTEMPT = ROOT + '/attempt_<UTC>'
COMMIT = 'a' * 40
TRAIN_ARGV = (PYTHON + ' tools/exp06_train.py --backbone cylindrical_oriented --save-dir '
              + ATTEMPT + ' --num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4'
              ' --decay-epochs 3 --lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2'
              ' --num-workers 12 --seed 0 --tf32 --log-interval 50 --save-every 500'
              ' --epoch-ckpt-every 1 --run-type full')
SMOKE_FLAGS = ('--epochs 1 --max-train-batches 3 --max-test-batches 2 --batch-size 4'
               ' --num-workers 4 --save-every 0 --no-save')


def dry_run(mode, *extra):
    command = ['bash', LAUNCHER, mode, '--gpu', '1', '--reviewed-commit', COMMIT, '--dry-run']
    completed = subprocess.run(command + list(extra), cwd=REPO, capture_output=True, text=True,
                               env={**os.environ, 'PYTHONPATH': str(REPO)})
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.splitlines()


def test_launcher_is_valid_bash():
    assert subprocess.run(['bash', '-n', LAUNCHER], cwd=REPO).returncode == 0
    assert os.access(REPO / LAUNCHER, os.X_OK)


def test_full_dry_run_matches_the_plan_argv():
    lines = dry_run('full')
    assert ('RUN ' + PYTHON + ' tools/exp06_finalize.py preflight --mode full --gpu 1'
            ' --reviewed-commit ' + COMMIT + ' --attempt-root ' + ROOT) in lines
    assert 'MKDIR ' + ATTEMPT in lines
    assert 'ENV CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=8' in lines
    assert 'RUN nohup setsid ' + TRAIN_ARGV in lines
    log = 'worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_<UTC>_train_full.log'
    assert 'SINK cat >> ' + log in lines and 'PIDFILE ' + ATTEMPT + '/launch.pid' in lines
    assert 'MARKER EXP06_CHILD_EXIT <code> <iso> >> ' + log in lines
    assert all('tail -f' not in line for line in lines), 'no second reader of the log'
    assert ('RUN ' + PYTHON + ' tools/exp06_finalize.py --run-dir ' + ATTEMPT
            + ' --run-type full --log ' + log + ' --child-exit <code>') in lines
    assert 'PROMOTE ' + ROOT + '/final -> attempt_<UTC>' in lines
    assert 'ABORT ' + ATTEMPT + '_ABORTED_<reason>' in lines


def test_probe_dry_run_uses_the_bounded_recipe():
    lines = dry_run('probe')
    probe = [line for line in lines if 'exp06_smoke.py' in line]
    assert len(probe) == 1
    assert probe[0] == ('RUN ' + PYTHON + ' tools/exp06_smoke.py --entry exp06_train --receipt '
                        + ROOT + '/probe_<UTC>.json --alarm-seconds 2400 --max-gb 46 --'
                        ' --backbone cylindrical_oriented --save-dir ' + ROOT + '/probe_<UTC>'
                        ' --epochs 1 --max-train-batches 200 --max-test-batches 20 --no-save'
                        ' --run-type probe --batch-size 32 --accum-steps 2 --tf32'
                        ' --num-workers 12').replace(' -- ', ' -- ')
    assert 'ENV CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=8' in lines


def test_smoke_dry_run_lists_the_section_nine_commands():
    lines = dry_run('smoke')
    smokes = [line for line in lines if line.startswith('RUN ') and 'exp06_smoke.py' in line]
    assert len(smokes) == 4
    assert smokes[0].endswith('--entry trainer --receipt ckpt/exp06/_smoke/receipt_trainer_<UTC>.json'
                              ' --alarm-seconds 300 --max-gb 3 -- --backbone simple --save-dir'
                              ' ckpt/exp06/_smoke/t0 ' + SMOKE_FLAGS)
    assert smokes[1].endswith('--entry exp06_train --receipt ckpt/exp06/_smoke/receipt_exp06_train_t0_<UTC>.json'
                              ' --alarm-seconds 300 --max-gb 3 -- --backbone simple --save-dir'
                              ' ckpt/exp06/_smoke/t0 ' + SMOKE_FLAGS)
    assert smokes[2].endswith('--entry exp06_train --receipt ckpt/exp06/_smoke/receipt_exp06_train_t1_<UTC>.json'
                              ' --alarm-seconds 300 --max-gb 3 -- --backbone cylindrical_oriented'
                              ' --save-dir ckpt/exp06/_smoke/t1 ' + SMOKE_FLAGS)
    assert smokes[3].endswith('--make-fixture ckpt/exp06/_smoke/fixture_cylor.pth')
    assert all('--tf32' not in line for line in smokes)


def test_finalize_mode_and_usage_errors():
    lines = dry_run('finalize', '--attempt', ATTEMPT, '--log', 'some.log', '--child-exit', '0')
    assert ('RUN ' + PYTHON + ' tools/exp06_finalize.py --run-dir ' + ATTEMPT
            + ' --run-type full --log some.log --child-exit 0') in lines
    for extra in (['invented'], ['full'], ['full', '--gpu', '1'], ['full', '--gpu', '1',
                  '--reviewed-commit', COMMIT, '--nonsense']):
        completed = subprocess.run(['bash', LAUNCHER] + extra, cwd=REPO, capture_output=True, text=True)
        assert completed.returncode == 2, completed.stdout


HARNESS = ('set -euo pipefail\n'
           'export EXP06_LAUNCH_LIB=1\n'
           'source tools/exp06_launch.sh\n'
           'run_child {attempt} {log} {child}\n'
           'close_child {attempt} {log}\n'
           'echo "HARNESS_STATUS $CHILD_STATUS $CHILD_PID"\n')


def run_harness(tmp_path, script, child_args=''):
    """Drive the launcher's child lifecycle directly, without a mode or a preflight."""
    attempt = tmp_path / 'attempt'
    attempt.mkdir()
    log = tmp_path / 'child.log'
    log.write_text('')
    stub = tmp_path / 'stub.sh'
    stub.write_text(script)
    stub.chmod(0o755)
    completed = subprocess.run(
        ['bash', '-c', HARNESS.format(attempt=attempt, log=log,
                                      child=str(stub) + (' ' + child_args if child_args else ''))],
        cwd=REPO, capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': str(REPO)})
    return attempt, log, completed


def test_a_surviving_descendant_cannot_write_past_the_end_marker(tmp_path):
    """Blocker 4: the sink reaches EOF only when every holder of the pipe has exited."""
    attempt, log, completed = run_harness(tmp_path, '#!/usr/bin/env bash\n'
                                          'echo early\n( sleep 2; echo late ) &\nexit 0\n')
    assert completed.returncode == 0, completed.stderr
    lines = log.read_text().splitlines()
    assert 'late' in lines, 'the descendant output was lost: ' + repr(lines)
    assert lines[-1].startswith('EXP06_CHILD_EXIT 0 '), lines
    assert lines.index('late') < len(lines) - 1
    receipt = json.loads((attempt / 'child_exit.json').read_text())
    assert receipt['status'] == 0 and receipt['ended_at'] == lines[-1].split()[2]
    assert receipt['log_sha256_after_marker'] == hashlib.sha256(log.read_bytes()).hexdigest()
    assert (attempt / 'launch.pid').read_text().strip() == str(receipt['child_pid'])
    assert not (attempt / 'child.pipe').exists()


def test_a_failing_child_reports_its_status_through_the_lifecycle(tmp_path):
    attempt, log, completed = run_harness(tmp_path,
                                          '#!/usr/bin/env bash\necho boom >&2\nexit 7\n')
    assert completed.returncode == 0, completed.stderr
    assert 'HARNESS_STATUS 7 ' in completed.stdout
    lines = log.read_text().splitlines()
    assert lines[0] == 'boom' and lines[-1].startswith('EXP06_CHILD_EXIT 7 ')
    assert json.loads((attempt / 'child_exit.json').read_text())['status'] == 7
