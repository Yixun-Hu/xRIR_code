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


# --- what a verdict field may look like (close review 16) ---------------------------
# Every field is validated before it is printed AND after it is read, so a probe that
# answers `link garbage` or `present 41ed relative` grants nothing: `garbage` as the
# published target let a resolution retire the attempt that `final` really pointed at.
PATH_MAX = 4096
FILE_TYPES = (0o010000, 0o020000, 0o040000, 0o060000, 0o100000, 0o120000, 0o140000)


def valid_mode(text):
    """A file mode in hex, three to six digits, whose type bits name a known type."""
    if not isinstance(text, str) or not 3 <= len(text) <= 6:
        return False
    if any(character not in '0123456789abcdef' for character in text):
        return False
    return (int(text, 16) & 0o170000) in FILE_TYPES


def valid_path(text):
    """An absolute, normalized, printable path of a length the kernel would accept."""
    if not isinstance(text, str) or not text.startswith('/') or len(text) > PATH_MAX:
        return False
    if any(character in text for character in ('\n', '\r', '\0')):
        return False
    if text == '/':
        return True
    if text.endswith('/') or '//' in text:
        return False
    return not any(part in ('', '.', '..') for part in text.split('/')[1:])


def _absolute(name):
    """Make a path absolute WITHOUT normalizing it: `..` is the kernel's business.

    ``abspath`` collapses `missing/../leaf` to `leaf`, and the parent check then
    inspects a directory the lookup never reached (close review 16).
    """
    return name if name.startswith('/') else os.path.join(os.getcwd(), name)


def _parent_is_inspectable(name):
    """An absence may only be claimed about a directory we could look into."""
    parent = os.path.dirname(_absolute(name)) or '/'
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
    # A trailing slash demands that the last component be a directory. ENOTDIR and
    # ENOENT are different news, and only a successful stat of a directory settles it.
    trailing_slash = name.endswith('/') and name != '/'
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
    if trailing_slash:
        return UNKNOWN, ('{} ends in a slash, so its last component must be a directory; '
                         'what failed here cannot be told apart'.format(name))
    inspectable, why = _parent_is_inspectable(name)
    return (ABSENT, None) if inspectable else (UNKNOWN, why)


def canonical(name):
    """The canonical path of ``name``, keeping the name itself when it is not a symlink.

    Only the directory is resolved for an ordinary name, so the basename a caller probed
    is the basename it gets back -- which is what lets every reader check that the
    answer is about the path it asked for.
    """
    absolute = _absolute(str(name))
    try:
        if not stat.S_ISLNK(os.lstat(absolute).st_mode):
            head, _, tail = absolute.rpartition('/')
            resolved = os.path.realpath(head or '/')
            return resolved if tail in ('', '.', '..') else os.path.join(resolved, tail)
    except OSError:
        pass
    return os.path.realpath(absolute)


def probe_link(path):
    """``(LINK, target)`` | ``(NOTLINK, None)`` | ``(NOLINK, None)`` | ``(UNKNOWN, why)``."""
    name = str(path)
    if name.endswith('/') and name != '/':
        return UNKNOWN, ('{} ends in a slash, so it names a directory, not the link '
                         'itself'.format(name))
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
    return LINK, os.path.realpath(_absolute(name))


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
            return 0
        mode, where = '{:x}'.format(detail.st_mode), canonical(argv[2])
        if not valid_mode(mode) or not valid_path(where):
            print('refusing to answer about {}: the verdict fields are not valid '
                  '({} {})'.format(argv[2], mode, where), file=sys.stderr)
            return UNKNOWN_EXIT
        print('present {} {}'.format(mode, where))
        return 0
    state, detail = probe_link(argv[2])
    if state == UNKNOWN:
        print('cannot decide {}: {}'.format(argv[2], detail), file=sys.stderr)
        return UNKNOWN_EXIT
    if state != LINK:
        print(state)
        return 0
    if not valid_path(detail):
        print('refusing to answer about {}: {} is not a valid target'.format(argv[2], detail),
              file=sys.stderr)
        return UNKNOWN_EXIT
    print('link {}'.format(detail))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
