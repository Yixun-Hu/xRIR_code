"""The lock holder (plan section 11 amendment A1, close review 6).

The launcher must not hold the arm lock on a descriptor of its own: the sourced
lifecycle passes every open descriptor into the background log sink and the detached
training child, so an orphaned sink would keep an arm locked with no trainer running.
A separate holder process takes the lock instead, and its life is *leased* to the
launcher's: it exits when its parent changes, so the lock vanishes within a second of
the launcher's exit or death however that happens.
"""
import fcntl
import os
from pathlib import Path
import subprocess
import time

import pytest

REPO = Path(__file__).resolve().parents[1]
HOLDER = 'tools/exp11_lock_holder.py'
PYTHON = '/home/yixunhu/miniconda3/envs/xRIR/bin/python'


def start_holder(lockfile, launcher_pid):
    """Start a holder this test owns; it is ended through its own Popen handle."""
    return subprocess.Popen([PYTHON, HOLDER, str(lockfile), str(launcher_pid)],
                            cwd=str(REPO), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def free(lockfile):
    """True when an independent process can take the lock right now."""
    return subprocess.run(['flock', '-n', str(lockfile), 'true']).returncode == 0


def wait_until_free(lockfile, seconds=5.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if free(lockfile):
            return True
        time.sleep(0.05)
    return False


def test_the_holder_announces_that_it_took_the_lock(tmp_path):
    lockfile = tmp_path / '.publish.lock'
    holder = start_holder(lockfile, os.getpid())
    try:
        line = holder.stdout.readline().strip()
        assert line.split()[0] == 'acquired', line
        assert int(line.split()[1]) == holder.pid, line
        assert not free(lockfile), 'the announcement precedes the lock'
    finally:
        holder.terminate()
        holder.wait(timeout=30)


def test_a_second_holder_is_refused_and_says_so(tmp_path):
    lockfile = tmp_path / '.publish.lock'
    first = start_holder(lockfile, os.getpid())
    try:
        assert first.stdout.readline().split()[0] == 'acquired'
        second = start_holder(lockfile, os.getpid())
        out, _ = second.communicate(timeout=30)
        assert out.strip() == 'refused'
        assert second.returncode == 1
    finally:
        first.terminate()
        first.wait(timeout=30)


def test_the_holder_exits_when_its_launcher_does(tmp_path):
    """The lease, not a cleanup path: the holder outlives nothing.

    A shell this test starts plays the launcher; it starts the holder and exits without
    releasing anything. The holder is reparented, sees the changed parent and ends.
    """
    lockfile = tmp_path / '.publish.lock'
    launcher = subprocess.run(
        ['bash', '-c', 'exec {fd}< <(exec "$1" "$2" "$3" $$); read -r a b <&$fd; '
                       'exec {fd}<&-; echo "$a $b"',
         '_', PYTHON, HOLDER, str(lockfile)],
        cwd=str(REPO), capture_output=True, text=True)
    verdict, pid = launcher.stdout.split()
    assert verdict == 'acquired', launcher.stderr[-400:]
    assert wait_until_free(lockfile), (
        'holder {} outlived the launcher that leased it'.format(pid))


def test_the_holder_ends_on_a_term_from_its_parent(tmp_path):
    lockfile = tmp_path / '.publish.lock'
    holder = start_holder(lockfile, os.getpid())
    assert holder.stdout.readline().split()[0] == 'acquired'
    holder.terminate()                      # our own child, through its handle
    assert holder.wait(timeout=30) == 0
    assert free(lockfile)


def test_the_holder_writes_nothing_into_the_file_it_locks(tmp_path):
    """A lock is a name the kernel knows, not a record anybody reads."""
    lockfile = tmp_path / '.publish.lock'
    lockfile.write_bytes(b'not the holder\n')
    holder = start_holder(lockfile, os.getpid())
    try:
        assert holder.stdout.readline().split()[0] == 'acquired'
        time.sleep(0.5)
        assert lockfile.read_bytes() == b'not the holder\n'
    finally:
        holder.terminate()
        holder.wait(timeout=30)


def test_the_lock_descriptor_is_close_on_exec():
    """Nothing the holder could ever exec may inherit the lock."""
    text = (REPO / HOLDER).read_text()
    assert 'FD_CLOEXEC' in text and 'O_CLOEXEC' in text
    probe = subprocess.run(
        [PYTHON, '-c',
         'import fcntl, os, sys; sys.path.insert(0, "tools");\n'
         'import exp11_lock_holder as h;\n'
         'fd = h.open_lock(sys.argv[1]);\n'
         'print(bool(fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC))',
         str(REPO / 'tools' / '.cloexec_probe.lock')],
        cwd=str(REPO), capture_output=True, text=True)
    try:
        assert probe.stdout.strip() == 'True', probe.stderr[-400:]
    finally:
        (REPO / 'tools' / '.cloexec_probe.lock').unlink(missing_ok=True)


def test_the_holder_refuses_a_call_it_cannot_understand(tmp_path):
    result = subprocess.run([PYTHON, HOLDER, str(tmp_path / 'x.lock')], cwd=str(REPO),
                            capture_output=True, text=True)
    assert result.returncode == 2 and 'usage' in result.stderr.lower()
