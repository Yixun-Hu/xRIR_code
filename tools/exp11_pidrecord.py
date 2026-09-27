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

which prints the pid and exits 0, or exits 1 and prints nothing. It never raises: an
absent, unreadable or impossible file is simply not a record.
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


def main(argv):
    if len(argv) != 2:
        print('usage: python -m tools.exp11_pidrecord <file>', file=sys.stderr)
        return 1
    record = pid_record(argv[1])
    if record is None:
        return 1
    print(record)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
