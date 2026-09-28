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
    """A verdict carries a MODE and nothing else -- no path ever crosses (plan A2)."""
    path = tmp_path / 'launching'
    path.write_text('launcher 1\n')
    state, detail = probe.probe(path)
    assert state == probe.PRESENT and detail.st_mode == path.stat().st_mode
    result = cli('probe', path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == 'present {:x}\n'.format(path.stat().st_mode)
    assert str(tmp_path) not in result.stdout


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
    """The classification, never the target: `link` says a link is there, no more."""
    path = tmp_path / 'final'
    target = tmp_path / 'attempt_20260927T000000'
    target.mkdir()
    if kind == 'link':
        path.symlink_to(target)
    elif kind == 'notlink':
        path.mkdir()
    result = cli('link', path)
    assert result.returncode == 0, result.stderr
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


def test_no_verdict_carries_a_path(tmp_path):
    """The grammar that three close reviews kept finding holes in is simply gone."""
    path = tmp_path / 'launching'
    path.write_text('x')
    link = tmp_path / 'final'
    link.symlink_to(path)
    for argv in (('probe', path), ('link', link), ('probe', tmp_path / 'absent'),
                 ('same', path, path), ('sameparent', path, tmp_path)):
        result = cli(*argv)
        assert result.returncode == 0, (argv, result.stderr)
        assert str(tmp_path) not in result.stdout, (argv, result.stdout)
        assert '/' not in result.stdout, (argv, result.stdout)


def test_the_mode_is_still_validated_before_it_is_printed(tmp_path):
    path = tmp_path / 'launching'
    path.write_text('x')
    assert probe.valid_mode('{:x}'.format(path.stat().st_mode))
    for bad in ('zzzz', '', 'ffffffff', '1', '81a4x', 'f000'):
        assert not probe.valid_mode(bad), bad
    assert not hasattr(probe, 'valid_path'), 'there is no path grammar any more'
    assert not hasattr(probe, 'canonical'), 'and no canonical path to hand out'


# --- identity questions, decided by device and inode (plan A2) ----------------------


def test_the_same_file_by_two_names_is_the_same(tmp_path):
    real = tmp_path / 'attempt_20260927T000000'
    real.mkdir()
    link = tmp_path / 'final'
    link.symlink_to(real)
    assert probe.same(real, link)[0] == probe.SAME
    assert cli('same', real, link).stdout == 'same\n'
    assert cli('same', link, real).stdout == 'same\n'
    # ... through a `..`, a `//` and a trailing slash, which all name one directory
    for spelling in (str(real) + '/', str(real) + '/./', str(tmp_path) + '//' + real.name,
                     str(real) + '/../' + real.name):
        assert cli('same', spelling, real).stdout == 'same\n', spelling


def test_two_different_directories_are_different(tmp_path):
    first, second = tmp_path / 'a', tmp_path / 'b'
    first.mkdir()
    second.mkdir()
    assert probe.same(first, second)[0] == probe.DIFFERENT
    assert cli('same', first, second).stdout == 'different\n'


def test_an_identity_nobody_can_establish_is_unknown(tmp_path):
    dangling = tmp_path / 'dangling'
    dangling.symlink_to(tmp_path / 'not-there')
    real = tmp_path / 'real'
    real.mkdir()
    for argv in (('same', dangling, real), ('same', real, dangling),
                 ('same', tmp_path / 'absent', real)):
        result = cli(*argv)
        assert result.returncode == 2 and result.stdout == '', argv


@needs_a_non_root_user
def test_an_identity_behind_a_shut_door_is_unknown(tmp_path):
    shut = tmp_path / 'shut'
    (shut / 'inside').mkdir(parents=True)
    real = tmp_path / 'real'
    real.mkdir()
    shut.chmod(0o000)
    try:
        assert cli('same', shut / 'inside', real).returncode == 2
    finally:
        shut.chmod(0o700)


def test_a_parent_is_asked_about_the_name_as_given(tmp_path):
    """`sameparent` takes the dirname of the UNNORMALIZED string, then stats it."""
    arm = tmp_path / 'arm'
    attempt = arm / 'attempt_20260927T000000'
    attempt.mkdir(parents=True)
    assert cli('sameparent', attempt, arm).stdout == 'same\n'
    assert cli('sameparent', attempt, tmp_path).stdout == 'different\n'
    alias = tmp_path / 'alias'
    alias.symlink_to(arm)
    assert cli('sameparent', alias / attempt.name, arm).stdout == 'same\n'
    # a parent nobody can stat is not a different parent
    assert cli('sameparent', tmp_path / 'gone' / 'x', arm).returncode == 2


def test_a_misuse_is_not_a_verdict(tmp_path):
    for argv in ([], ['probe'], ['fly', str(tmp_path)], ['probe', 'a', 'b'],
                 ['same', str(tmp_path)], ['sameparent', str(tmp_path)],
                 ['same', 'a', 'b', 'c']):
        result = cli(*argv)
        assert result.returncode == 2 and result.stdout == ''
