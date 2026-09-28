"""Hold one exp_11 arm's publication lock, for exactly as long as its launcher lives.

Plan section 11 amendment A1, close review 6. The launcher must not hold the lock on a
descriptor of its own: the sourced lifecycle (``tools/exp06_launch.sh``) hands every
open descriptor to the background log sink and to the detached training child, so an
orphaned sink -- one waiting on a FIFO writer that will never arrive -- would keep an
arm locked for good with no trainer running.

So the lock lives in this process instead, and its life is *leased* to the launcher's:

* ``flock(LOCK_EX | LOCK_NB)`` on the arm's lock file. Refused means another invocation
  is publishing this arm; this process says ``refused`` and exits 1.
* Taken means it prints ``acquired <pid>`` and then does nothing but watch its parent.
  When the launcher exits or dies -- normally, on a signal, on SIGKILL -- this process
  is reparented, sees the changed parent id and exits, and the kernel drops the lock.
  The launcher also terminates it directly at the end of the protected section, so on
  the ordinary path the release is immediate.
* The lock file is opened for append and never written to: a lock is a name the kernel
  knows, not a record anybody reads. The descriptor is close-on-exec and this process
  execs nothing, so the lock cannot escape into any other program.
"""
import fcntl
import os
import signal
import sys
import time

LEASE_POLL_S = 0.5


def open_lock(path):
    """Open the lock file without truncating it, on a close-on-exec descriptor."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC, 0o600)
    fcntl.fcntl(fd, fcntl.F_SETFD, fcntl.fcntl(fd, fcntl.F_GETFD) | fcntl.FD_CLOEXEC)
    return fd


def _leave(signum, frame):
    raise SystemExit(0)


def main(argv):
    if len(argv) != 3:
        print('usage: exp11_lock_holder.py <lock file> <launcher pid>', file=sys.stderr)
        return 2
    path, launcher = argv[1], int(argv[2])
    fd = open_lock(path)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('refused', flush=True)
        return 1
    signal.signal(signal.SIGTERM, _leave)
    signal.signal(signal.SIGINT, _leave)
    print('acquired {}'.format(os.getpid()), flush=True)
    while os.getppid() == launcher:
        time.sleep(LEASE_POLL_S)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
