"""What a pid file says, decided once, for everyone who asks (close review 10).

A pid record is the WHOLE file and nothing else:

    ^[0-9]{1,10}\\n?$

digits, at most one trailing newline. No leading blank, no second record, no CR, no
NUL, no other byte. Whether that pid is still alive is a different question, asked of
the number this returns.

The launcher used to read pid files itself, in Bash. Command substitution drops NUL
bytes, so ``<digits>\\0`` came back as digits and a dead pid with a NUL made an arm read
quiet while a trainer was in it. There is therefore exactly one reader now: this module,
imported by ``tools/exp11_train.py`` and executed by the launcher as

    python -m tools.exp11_pidrecord <file>

**The CLI answers, or it does not answer at all** (close review 11). Both answers are
answers, and both are exit 0 with one line on stdout:

    record <pid>     this file holds that pid record
    norecord         there is no record here, and we looked

Anything else is the reader FAILING, not a verdict about the file: a missing
interpreter, an unimportable module, a crash, a signal, an empty stdout, a different
first word, and -- since close review 12 -- **any inability to inspect the path**. Being
unable to look is not the same as looking and finding nothing:

* definite ``norecord``: the name is confirmed absent (and its directory could be
  inspected), the path exists but is not a regular file (a directory, a FIFO, a socket,
  a device -- none of them is a pid record, and a FIFO is classified without being
  opened, because opening one blocks forever), or the bytes were read and are not the
  grammar;
* **unknown** (exit 2, a reason on stderr, nothing on stdout): a permission denied, an
  I/O error, a symlink loop, a parent directory that cannot be inspected, a stat or an
  open or a read that failed. A `chmod 000` ``train.pid`` may well name a living
  trainer; reading it as "no record" is how an arm retires a running one.

The caller must treat everything that is not one of the two lines as unknown and refuse
to decide.
"""
import os
import re
import stat
import sys

PID_RECORD = re.compile(rb'[0-9]{1,10}\n?')
RECORD, NORECORD, UNKNOWN = 'record', 'norecord', 'unknown'
USAGE_EXIT = 2          # not 1: a misuse is not a verdict about any file
UNKNOWN_EXIT = 2        # neither is "I could not look"


def inspect_record(path):
    """``(verdict, detail)``: ``(RECORD, pid)``, ``(NORECORD, why)``, ``(UNKNOWN, why)``."""
    name = str(path)
    try:
        info = os.stat(name)                # follows symlinks, like open() below
    except FileNotFoundError:
        # Absent -- but only if the directory it would be in could be inspected. An
        # unreadable parent raises here too, and says nothing about what is inside it.
        parent = os.path.dirname(os.path.abspath(name)) or '.'
        try:
            os.stat(parent)
        except OSError as error:
            return UNKNOWN, 'cannot inspect the directory {}: {}'.format(parent, error)
        return NORECORD, 'no such file'
    except OSError as error:                # EACCES, ELOOP, EIO, ENOTDIR, ...
        return UNKNOWN, 'cannot inspect {}: {}'.format(name, error)
    if not stat.S_ISREG(info.st_mode):
        # A directory, FIFO, socket or device is definitely not a pid record, and this
        # is decided WITHOUT opening it: opening a FIFO waits for a writer forever.
        return NORECORD, 'not a regular file'
    try:
        with open(name, 'rb') as handle:
            data = handle.read(64)          # a record is at most 11 bytes; read a little more
    except OSError as error:                # it exists and we may not read it
        return UNKNOWN, 'cannot read {}: {}'.format(name, error)
    if PID_RECORD.fullmatch(data):
        return RECORD, int(data.decode('ascii'))
    return NORECORD, 'not a pid record'


def pid_record(path):
    """The pid ``path`` holds, or ``None``. Bytes, never text: CRLF is not a newline.

    ``None`` covers both "no record" and "could not look"; the callers that must tell
    those apart -- the launcher, through the CLI -- ask ``inspect_record``.
    """
    verdict, detail = inspect_record(path)
    return detail if verdict == RECORD else None


def main(argv):
    if len(argv) != 2:
        print('usage: python -m tools.exp11_pidrecord <file>', file=sys.stderr)
        return USAGE_EXIT
    verdict, detail = inspect_record(argv[1])
    if verdict == UNKNOWN:
        print('cannot decide {}: {}'.format(argv[1], detail), file=sys.stderr)
        return UNKNOWN_EXIT
    print(NORECORD if verdict == NORECORD else 'record {}'.format(detail))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
