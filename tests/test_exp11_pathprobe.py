"""One errno-aware answer to "is this path there?" (close review 15).

`stat` in bash carries no errno: ENAMETOOLONG, EIO and an injected failure all come back
as "no", and the shell's probes read that as a confirmed absence when the parent looked
fine. A marker read as absent leaves an arm quiet, a receipt read as absent retires a
finished run, a `final` read as absent is replaced. The classification therefore lives in
one place, in Python, where the errno is available.
"""
import errno
import os
from pathlib import Path
import subprocess

import pytest

from tools import exp11_pathprobe as probe

REPO = Path(__file__).resolve().parents[1]
PYTHON = '/home/yixunhu/miniconda3/envs/xRIR/bin/python'

needs_a_non_root_user = pytest.mark.skipif(
    os.geteuid() == 0, reason='root ignores the permission bits these cases turn off')


def cli(*argv):
    return subprocess.run([PYTHON, '-m', 'tools.exp11_pathprobe'] + [str(a) for a in argv],
                          cwd=str(REPO), capture_output=True, text=True, timeout=60,
                          env=dict(os.environ, PYTHONPATH=str(REPO),
                                   CUDA_VISIBLE_DEVICES=''))


def test_a_file_that_is_there_is_present_with_its_mode(tmp_path):
    path = tmp_path / 'launching'
    path.write_text('launcher 1\n')
    state, detail = probe.probe(path)
    assert state == probe.PRESENT and detail.st_mode == path.stat().st_mode
    result = cli('probe', path)
    assert result.returncode == 0, result.stderr
    first, mode, canonical = result.stdout.rstrip('\n').split(' ', 2)
    assert first == 'present' and int(mode, 16) == path.stat().st_mode
    assert canonical == os.path.realpath(str(path))


def test_a_missing_name_with_a_searchable_parent_is_absent(tmp_path):
    state, _ = probe.probe(tmp_path / 'nothing')
    assert state == probe.ABSENT
    result = cli('probe', tmp_path / 'nothing')
    assert result.returncode == 0 and result.stdout == 'absent\n'


def test_a_dangling_symlink_is_not_an_absence(tmp_path):
    """The NAME is there; only its target is not. Nothing may call that "not there"."""
    link = tmp_path / 'train.pid'
    link.symlink_to(tmp_path / 'moved-away')
    state, _ = probe.probe(link)
    assert state == probe.UNKNOWN
    result = cli('probe', link)
    assert result.returncode == 2 and result.stdout == ''
    assert result.stderr.strip() and 'Traceback' not in result.stderr


def test_a_name_too_long_is_unknown(tmp_path):
    """ENAMETOOLONG is not ENOENT, and bash's `stat` cannot tell them apart."""
    long_name = tmp_path / ('x' * 300)
    with pytest.raises(OSError) as raised:
        os.stat(str(long_name))
    assert raised.value.errno == errno.ENAMETOOLONG
    state, _ = probe.probe(long_name)
    assert state == probe.UNKNOWN
    assert cli('probe', long_name).returncode == 2


@needs_a_non_root_user
def test_a_parent_nobody_may_search_is_unknown(tmp_path):
    shut = tmp_path / 'shut'
    shut.mkdir()
    (shut / 'launching').write_text('launcher 1\n')
    shut.chmod(0o000)
    try:
        for name in ('launching', 'absent'):
            assert probe.probe(shut / name)[0] == probe.UNKNOWN
            assert cli('probe', shut / name).returncode == 2
    finally:
        shut.chmod(0o700)


def test_a_directory_is_present_and_says_so(tmp_path):
    state, detail = probe.probe(tmp_path)
    assert state == probe.PRESENT and os.path.stat.S_ISDIR(detail.st_mode)


@pytest.mark.parametrize('kind', ['link', 'notlink', 'nolink'])
def test_the_three_link_answers(tmp_path, kind):
    path = tmp_path / 'final'
    target = tmp_path / 'attempt_20260927T000000'
    target.mkdir()
    if kind == 'link':
        path.symlink_to(target)
    elif kind == 'notlink':
        path.mkdir()
    result = cli('link', path)
    assert result.returncode == 0, result.stderr
    if kind == 'link':
        assert result.stdout == 'link {}\n'.format(os.path.realpath(str(target)))
    else:
        assert result.stdout == kind + '\n'


def test_a_link_that_cannot_be_resolved_is_unknown(tmp_path):
    loop = tmp_path / 'final'
    loop.symlink_to(loop)
    result = cli('link', loop)
    assert result.returncode == 2 and result.stdout == ''


@needs_a_non_root_user
def test_a_link_through_an_inaccessible_directory_is_unknown(tmp_path):
    shut = tmp_path / 'shut'
    (shut / 'attempt_20260101T000000').mkdir(parents=True)
    link = tmp_path / 'final'
    link.symlink_to(shut / 'attempt_20260101T000000')
    shut.chmod(0o000)
    try:
        result = cli('link', link)
        assert result.returncode == 2 and result.stdout == ''
    finally:
        shut.chmod(0o700)


def test_a_misuse_is_not_a_verdict(tmp_path):
    for argv in ([], ['probe'], ['fly', str(tmp_path)], ['probe', 'a', 'b']):
        result = cli(*argv)
        assert result.returncode == 2 and result.stdout == ''


# --- close review 16: the probe must not answer about a path it did not look up -------


def test_a_missing_component_before_a_dotdot_is_not_an_absence(tmp_path):
    """`base/missing/../leaf` does not resolve to `base/leaf`: the lookup FAILS.

    `abspath` collapses the `..` lexically and the parent check then inspects `base`,
    which is there -- so the probe called a path absent that the kernel never reached.
    """
    state, _ = probe.probe(tmp_path / 'missing' / '..' / 'leaf')
    assert state == probe.UNKNOWN
    assert cli('probe', tmp_path / 'missing' / '..' / 'leaf').returncode == 2


def test_a_dangling_component_before_a_dotdot_is_not_an_absence(tmp_path):
    dangling = tmp_path / 'dangling'
    dangling.symlink_to(tmp_path / 'not-there')
    state, _ = probe.probe(tmp_path / 'dangling' / '..' / 'leaf')
    assert state == probe.UNKNOWN
    assert cli('probe', tmp_path / 'dangling' / '..' / 'leaf').returncode == 2


def test_a_dotdot_path_that_really_resolves_is_still_answered(tmp_path):
    """The rule is "look it up", not "refuse every ..": a real path still answers."""
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'leaf').write_text('x')
    assert probe.probe(tmp_path / 'sub' / '..' / 'leaf')[0] == probe.PRESENT
    assert probe.probe(tmp_path / 'sub' / '..' / 'gone')[0] == probe.ABSENT


def test_a_trailing_slash_on_a_dangling_link_is_not_an_absence(tmp_path):
    """A trailing slash demands a directory; ENOTDIR and ENOENT are not the same news."""
    link = tmp_path / 'final'
    link.symlink_to(tmp_path / 'not-there')
    state, _ = probe.probe(str(link) + '/')
    assert state == probe.UNKNOWN
    assert cli('probe', str(link) + '/').returncode == 2
    result = cli('link', str(link) + '/')
    assert result.returncode == 2, result.stdout


def test_every_printed_field_is_validated(tmp_path):
    """A healthy module can never print a field its own readers would refuse."""
    path = tmp_path / 'launching'
    path.write_text('x')
    assert probe.valid_mode('{:x}'.format(path.stat().st_mode))
    assert probe.valid_path(os.path.realpath(str(path)))
    for bad in ('relative/x', '/abs/../x', '//abs', '/trailing/', '/dot/./x', '/' + 'x' * 5000,
                '/carriage\rreturn'):
        assert not probe.valid_path(bad), bad
    for bad in ('zzzz', '', 'ffffffff', '1', '81a4x'):
        assert not probe.valid_mode(bad), bad


def test_the_canonical_path_keeps_the_probed_name(tmp_path):
    """Only the DIRECTORY is canonicalised for a name that is not itself a symlink."""
    real = tmp_path / 'real'
    real.mkdir()
    (real / 'launching').write_text('x')
    alias = tmp_path / 'alias'
    alias.symlink_to(real)
    result = cli('probe', alias / 'launching')
    assert result.returncode == 0
    assert result.stdout.rstrip('\n').endswith('/real/launching'), result.stdout
