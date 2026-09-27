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
    norecord         it holds no record -- missing, unreadable, a directory, bad bytes

Anything else is the reader FAILING, not a verdict about the file: a missing
interpreter, an unimportable module, a crash, a signal, an empty stdout, a different
first word. Those used to be indistinguishable from "not a record", because an
ImportError also exits 1 -- and a caller that reads a failure as "nothing is alive"
retires the attempt of a running trainer. The caller must therefore treat everything
that is not one of the two lines as **unknown** and refuse to decide.
"""
import re
import sys

PID_RECORD = re.compile(rb'[0-9]{1,10}\n?')


def pid_record(path):
    """The pid ``path`` holds, or ``None``. Bytes, never text: CRLF is not a newline."""
    try:
        with open(str(path), 'rb') as handle:
            data = handle.read(64)          # a record is at most 11 bytes; read a little more
    except OSError:
        return None
    return int(data.decode('ascii')) if PID_RECORD.fullmatch(data) else None


USAGE_EXIT = 2          # not 1: a misuse is not a verdict about any file


def main(argv):
    if len(argv) != 2:
        print('usage: python -m tools.exp11_pidrecord <file>', file=sys.stderr)
        return USAGE_EXIT
    record = pid_record(argv[1])
    print('norecord' if record is None else 'record {}'.format(record))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
