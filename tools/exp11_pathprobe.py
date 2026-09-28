"""Is this path there? One errno-aware answer, for the shell and for the trainer.

`test -e` and `stat` in bash carry no errno. ENAMETOOLONG, EIO, an ACL that hides a name
and a plain "no such file" all come back the same way, and the launcher read that as a
confirmed **absence** whenever the parent directory looked fine (close review 15). An
absent marker leaves an arm quiet, an absent receipt retires a finished run, an absent
`final` is replaced. So the classification lives here, in the one language where the
errno is available, and the shell asks this module the way it asks the pid reader.

**The CLI answers, or it does not answer at all.** One line on stdout, exit 0:

    probe <path>    present <mode-hex> <canonical-path>   it is there
                    absent                                it is NOT there
    link <path>     link <canonical-target>               a symlink that resolves
                    notlink                               the name is there, not a link
                    nolink                                there is no such name

Anything else -- a non-zero exit, no line, another first word, a misuse (exit 2) -- is
**unknown**: the caller must refuse to decide rather than read it as an absence.

An absence is the strongest claim here, so it is the most guarded: it needs
``FileNotFoundError`` from ``os.stat`` **and** from ``os.lstat`` (a name that lstats but
does not stat is a dangling or looping symlink -- the name IS there), and a parent
directory that could itself be stat'ed and can be searched. Every other errno is
unknown.
"""
import errno
import os
from pathlib import Path
import stat
import sys

PRESENT, ABSENT, UNKNOWN = 'present', 'absent', 'unknown'
LINK, NOTLINK, NOLINK = 'link', 'notlink', 'nolink'
USAGE_EXIT = 2          # a misuse is not a verdict about any path
UNKNOWN_EXIT = 2


def _why(error):
    name = errno.errorcode.get(getattr(error, 'errno', None), 'OSError')
    return '{}: {}'.format(name, error)


def _parent_is_inspectable(name):
    """An absence may only be claimed about a directory we could look into."""
    parent = os.path.dirname(os.path.abspath(name)) or '.'
    try:
        info = os.stat(parent)
    except OSError as error:
        return False, 'cannot inspect the directory {}: {}'.format(parent, _why(error))
    if not stat.S_ISDIR(info.st_mode):
        return False, '{} is not a directory'.format(parent)
    if not os.access(parent, os.X_OK):
        return False, '{} cannot be searched'.format(parent)
    return True, None


def probe(path):
    """``(PRESENT, stat_result)`` | ``(ABSENT, None)`` | ``(UNKNOWN, reason)``."""
    name = str(path)
    try:
        return PRESENT, os.stat(name)
    except FileNotFoundError:
        pass
    except OSError as error:               # EACCES, ELOOP, EIO, ENAMETOOLONG, ENOTDIR...
        return UNKNOWN, 'cannot inspect {}: {}'.format(name, _why(error))
    try:
        os.lstat(name)                     # the name itself, unfollowed
        return UNKNOWN, ('{} is a symlink whose target cannot be resolved; the name is '
                         'there'.format(name))
    except FileNotFoundError:
        pass
    except OSError as error:
        return UNKNOWN, 'cannot inspect {}: {}'.format(name, _why(error))
    inspectable, why = _parent_is_inspectable(name)
    return (ABSENT, None) if inspectable else (UNKNOWN, why)


def probe_link(path):
    """``(LINK, target)`` | ``(NOTLINK, None)`` | ``(NOLINK, None)`` | ``(UNKNOWN, why)``."""
    name = str(path)
    try:
        info = os.lstat(name)
    except FileNotFoundError:
        inspectable, why = _parent_is_inspectable(name)
        return (NOLINK, None) if inspectable else (UNKNOWN, why)
    except OSError as error:
        return UNKNOWN, 'cannot inspect {}: {}'.format(name, _why(error))
    if not stat.S_ISLNK(info.st_mode):
        return NOTLINK, None
    try:
        os.stat(name)                      # it must resolve to something we can see
    except OSError as error:
        return UNKNOWN, 'cannot resolve {}: {}'.format(name, _why(error))
    return LINK, os.path.realpath(name)


def file_state(path):
    """PRESENT only for a regular file; ABSENT for no such name or another kind."""
    state, detail = probe(path)
    if state == PRESENT and not stat.S_ISREG(detail.st_mode):
        return ABSENT, detail
    return state, detail


def directory_state(path):
    state, detail = probe(path)
    if state == PRESENT and not stat.S_ISDIR(detail.st_mode):
        return ABSENT, detail
    return state, detail


def main(argv):
    if len(argv) != 3 or argv[1] not in ('probe', 'link'):
        print('usage: python -m tools.exp11_pathprobe <probe|link> <path>',
              file=sys.stderr)
        return USAGE_EXIT
    if argv[1] == 'probe':
        state, detail = probe(argv[2])
        if state == UNKNOWN:
            print('cannot decide {}: {}'.format(argv[2], detail), file=sys.stderr)
            return UNKNOWN_EXIT
        if state == ABSENT:
            print(ABSENT)
        else:
            print('present {:x} {}'.format(detail.st_mode, os.path.realpath(argv[2])))
        return 0
    state, detail = probe_link(argv[2])
    if state == UNKNOWN:
        print('cannot decide {}: {}'.format(argv[2], detail), file=sys.stderr)
        return UNKNOWN_EXIT
    print('link {}'.format(detail) if state == LINK else state)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
