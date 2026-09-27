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


@pytest.mark.parametrize('content,valid', PID_RECORDS, ids=IDS)
def test_the_cli_says_the_same(tmp_path, content, valid):
    """The launcher has no reader of its own; it asks this one."""
    result = subprocess.run([PYTHON, '-m', 'tools.exp11_pidrecord',
                             str(write(tmp_path, content))],
                            cwd=str(REPO), capture_output=True, text=True,
                            env=dict(os.environ, PYTHONPATH=str(REPO),
                                     CUDA_VISIBLE_DEVICES=''))
    assert (result.returncode == 0) is valid, (content, result.stdout, result.stderr)
    if valid:
        assert result.stdout.strip() == content.decode().strip()


@pytest.mark.parametrize('content,valid', PID_RECORDS, ids=IDS)
def test_the_trainer_uses_that_module(tmp_path, content, valid):
    assert (exp11_train.pid_record(write(tmp_path, content)) is not None) is valid


@pytest.mark.parametrize('content,valid', PID_RECORDS, ids=IDS)
def test_the_two_callers_never_disagree(tmp_path, content, valid):
    """One table, two callers, one answer -- that is the whole point of the module."""
    path = write(tmp_path, content)
    shell = subprocess.run(
        ['bash', '-c', 'set -euo pipefail\nEXP11_LAUNCH_LIB=1 source tools/exp11_launch.sh\n'
                       'if pid_record "$1"; then echo " RECORD_OK"; else echo RECORD_BAD; fi',
         '_', str(path)],
        cwd=str(REPO), capture_output=True, text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    assert shell.returncode == 0, shell.stderr[-400:]
    shell_says = 'RECORD_OK' in shell.stdout
    python_says = exp11_train.pid_record(path) is not None
    assert shell_says == python_says == valid, (content, shell.stdout)


def test_an_unreadable_file_is_not_a_record_and_raises_nothing(tmp_path):
    assert exp11_pidrecord.pid_record(tmp_path / 'absent') is None
    assert exp11_pidrecord.pid_record(tmp_path) is None        # a directory
    result = subprocess.run([PYTHON, '-m', 'tools.exp11_pidrecord', str(tmp_path / 'absent')],
                            cwd=str(REPO), capture_output=True, text=True,
                            env=dict(os.environ, PYTHONPATH=str(REPO)))
    assert result.returncode == 1 and 'Traceback' not in result.stderr


def test_the_launcher_reads_no_file_content_itself():
    """No command substitution of file bytes anywhere: that is where the NULs went."""
    text = (REPO / 'tools/exp11_launch.sh').read_text()
    assert 'exp11_pidrecord' in text
    for lost in ('$(cat --', '$(cat -- "$1")', "tr -d '\\n'", '$(< '):
        assert lost not in text, '{} reads file content through the shell'.format(lost)
