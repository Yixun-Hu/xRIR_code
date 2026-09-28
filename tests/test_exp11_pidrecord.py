"""One pid-record reader, shared by the launcher and the trainer (close review 10).

A pid file says exactly one thing. Two implementations of "what these bytes mean" is
two answers, and the shell's was the looser one: Bash command substitution drops NUL
bytes, so ``<digits>\\0`` read as a pid the kernel had never been asked about. The
grammar now lives in one module, and the shell calls that module rather than reading
the file itself.
"""
import os
from pathlib import Path
import subprocess

import pytest

from tools import exp11_pidrecord, exp11_train

REPO = Path(__file__).resolve().parents[1]
PYTHON = '/home/yixunhu/miniconda3/envs/xRIR/bin/python'
LIVE = b'1457170'                  # stands for the reviewer's P
DEAD = b'4194303'                  # stands for D: syntactically fine, long gone

# The reviewer's table, byte for byte. `valid` is "this file holds one pid record".
PID_RECORDS = [
    (LIVE,                      True),
    (LIVE + b'\n',              True),
    (b'1234567890\n',           True),
    (b'\n' + LIVE + b'\n',      False),
    (b'12\n34',                 False),
    (b' ' + LIVE + b'\n',       False),
    (LIVE + b'\r',              False),
    (LIVE + b'\r\n',            False),
    (b'',                       False),
    (b'\n\n',                   False),
    (LIVE + b'\n\n',            False),
    (b'12345678901\n',          False),
    (LIVE + b'\0',              False),          # trailing NUL
    (b'\0' + LIVE,              False),          # leading NUL
    (b'145\x007170',            False),          # NUL inside
    (LIVE + b'\0\n',            False),
    (DEAD + b'\0',              False),          # dead pid and a NUL
]
IDS = [repr(content) for content, _ in PID_RECORDS]


def write(tmp_path, content):
    path = tmp_path / 'child.pid'
    path.write_bytes(content)
    return path


@pytest.mark.parametrize('content,valid', PID_RECORDS, ids=IDS)
def test_the_module_reads_the_grammar(tmp_path, content, valid):
    record = exp11_pidrecord.pid_record(write(tmp_path, content))
    assert (record is not None) is valid, (content, record)
    if valid:
        assert record == int(content)


def cli(path):
    return subprocess.run([PYTHON, '-m', 'tools.exp11_pidrecord', str(path)],
                          cwd=str(REPO), capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=str(REPO),
                                   CUDA_VISIBLE_DEVICES=''))


@pytest.mark.parametrize('content,valid', PID_RECORDS, ids=IDS)
def test_the_cli_returns_a_definite_verdict(tmp_path, content, valid):
    """Both answers are answers (close review 11).

    Exit 1 used to mean "not a record" -- and it is also what a missing interpreter, an
    unimportable module and a crash return. The launcher could not tell a verdict from a
    failure, so a broken reader read as "nothing is alive" and a resolution retired a
    live trainer's attempt. A verdict is now a line on stdout AND exit 0; anything else
    is the reader failing, never an answer about the file.
    """
    result = cli(write(tmp_path, content))
    assert result.returncode == 0, (content, result.stdout, result.stderr)
    expected = 'record {}'.format(int(content)) if valid else 'norecord'
    assert result.stdout == expected + '\n', (content, result.stdout)
    assert result.stderr == ''


def test_a_verdict_is_one_line_and_nothing_else(tmp_path):
    """One line: the launcher reads the whole answer, so a second line is not a verdict."""
    for content, expected in ((b'1457170\n', 'record 1457170\n'), (b'', 'norecord\n')):
        assert cli(write(tmp_path, content)).stdout == expected


def test_a_misuse_is_not_a_verdict(tmp_path):
    """No file, or too many: the reader says nothing about a file it was not given."""
    for argv in ([], [str(tmp_path / 'a'), str(tmp_path / 'b')]):
        result = subprocess.run([PYTHON, '-m', 'tools.exp11_pidrecord'] + argv,
                                cwd=str(REPO), capture_output=True, text=True,
                                env=dict(os.environ, PYTHONPATH=str(REPO)))
        assert result.returncode not in (0, 1), result.stdout
        assert result.stdout == '', result.stdout


@pytest.mark.parametrize('content,valid', PID_RECORDS, ids=IDS)
def test_the_trainer_uses_that_module(tmp_path, content, valid):
    assert (exp11_train.pid_record(write(tmp_path, content)) is not None) is valid


@pytest.mark.parametrize('content,valid', PID_RECORDS, ids=IDS)
def test_the_two_callers_never_disagree(tmp_path, content, valid):
    """One table, two callers, one answer -- that is the whole point of the module."""
    path = write(tmp_path, content)
    shell = subprocess.run(
        ['bash', '-c', 'set -euo pipefail\nEXP11_LAUNCH_LIB=1 source tools/exp11_launch.sh\n'
                       'status=0\npid_record "$1" || status=$?\necho "STATUS $status"',
         '_', str(path)],
        cwd=str(REPO), capture_output=True, text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    assert shell.returncode == 0, shell.stderr[-400:]
    # 2 is "the reader failed"; no row of the table may ever produce it.
    assert 'STATUS 2' not in shell.stdout, (content, shell.stdout)
    shell_says = 'STATUS 0' in shell.stdout
    python_says = exp11_train.pid_record(path) is not None
    assert shell_says == python_says == valid, (content, shell.stdout)


# --- close review 12 blocker 1: not being able to look is not an answer -------------
# Every OSError used to become `norecord`, so a `chmod 000` train.pid -- a file that
# exists and may well hold a living pid -- read as a definite absence, and a resolution
# retired the attempt of a registered, running trainer. Only two things are definite: a
# confirmed absence, and bytes we read and found malformed.

needs_a_non_root_user = pytest.mark.skipif(
    os.geteuid() == 0, reason='root ignores the permission bits these cases turn off')


def test_an_absent_file_and_a_directory_are_definite(tmp_path):
    """The two things that really are answers about a path we could inspect."""
    assert exp11_pidrecord.pid_record(tmp_path / 'absent') is None
    assert exp11_pidrecord.pid_record(tmp_path) is None        # a directory
    for path in (tmp_path / 'absent', tmp_path):
        result = cli(path)
        assert result.returncode == 0 and result.stdout == 'norecord\n'
        assert 'Traceback' not in result.stderr


def test_a_fifo_is_definite_and_is_never_opened(tmp_path):
    """Opening a FIFO blocks forever; classifying one does not."""
    fifo = tmp_path / 'child.pid'
    os.mkfifo(str(fifo))
    result = subprocess.run([PYTHON, '-m', 'tools.exp11_pidrecord', str(fifo)],
                            cwd=str(REPO), capture_output=True, text=True, timeout=30,
                            env=dict(os.environ, PYTHONPATH=str(REPO)))
    assert result.returncode == 0 and result.stdout == 'norecord\n'


@needs_a_non_root_user
def test_a_file_we_cannot_read_is_not_an_absent_file(tmp_path):
    """The reviewer's case: chmod 000 on a train.pid that names a living trainer."""
    path = write(tmp_path, b'1457170\n')
    path.chmod(0o000)
    try:
        result = cli(path)
        assert result.returncode == 2, (result.stdout, result.stderr)
        assert result.stdout == '', 'no verdict line at all'
        assert result.stderr.strip() and 'Traceback' not in result.stderr
    finally:
        path.chmod(0o600)


@needs_a_non_root_user
def test_a_parent_we_cannot_inspect_is_not_an_absent_file(tmp_path):
    """An unreadable directory says nothing about what is or is not inside it."""
    shut = tmp_path / 'shut'
    shut.mkdir()
    (shut / 'child.pid').write_bytes(b'1457170\n')
    shut.chmod(0o000)
    try:
        for name in ('child.pid', 'absent.pid'):
            result = cli(shut / name)
            assert result.returncode == 2, (name, result.stdout, result.stderr)
            assert result.stdout == ''
    finally:
        shut.chmod(0o700)


def test_the_launcher_reads_no_file_content_itself():
    """No command substitution of file bytes anywhere: that is where the NULs went."""
    text = (REPO / 'tools/exp11_launch.sh').read_text()
    assert 'exp11_pidrecord' in text
    for lost in ('$(cat --', '$(cat -- "$1")', "tr -d '\\n'", '$(< '):
        assert lost not in text, '{} reads file content through the shell'.format(lost)
