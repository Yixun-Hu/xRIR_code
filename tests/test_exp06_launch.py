"""Launch preflight and the dry-run argv of tools/exp06_launch.sh."""
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
