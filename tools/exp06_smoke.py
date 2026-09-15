"""Bounded in-process runner for the GPU smokes of plan v4 section 9 (rung 4, R3.3).

Runs one named entry point in this process under a wall-clock alarm, reports the peak
CUDA allocation and the wall time on a single ``EXP06_SMOKE`` line, and writes a
``diagnostic`` receipt. The alarm and the memory budget abort with status 3, so a smoke
can never quietly grow past the co-tenant budget agreed with the peer session.

Budgets are checked before the entry is imported (finite and strictly positive), the
status an entry returns is recorded, and the memory ceiling is enforced *during*
execution by a one-second watchdog as well as retrospectively.

``--make-fixture`` writes a CPU-initialised ``cylindrical_oriented`` state dict (plus a
sidecar with its hash and the registry digest) for the HAA smokes: the exp_01
checkpoints cannot load into the five-channel patch embedding.

    python tools/exp06_smoke.py --entry exp06_train -- --backbone simple --no-save ...
    python tools/exp06_smoke.py --make-fixture ckpt/exp06/_smoke/fixture_cylor.pth

The receipt records a numeric ``exit_status`` (0, the integer an entry returned, 3 for an
alarm or memory abort, 1 for an exception), the reason in ``outcome`` (``ok``, ``failed``,
``aborted_alarm`` or ``aborted_memory``, with the exception type in ``outcome_detail``), and
``aborted_memory`` when the ceiling fired while the entry was still running. It also names
its runner and that runner's closure digest, the run type, the budgets, the wall-clock
window and the peak allocation, so the finalizer can require timing and memory evidence
rather than a bare ``exit_status`` (finding 6). With ``--provenance-out`` the runner writes
a ``provenance.json`` of its own -- closure, orchestration, code digests and approvals --
which the finalizer binds as it binds a full run's, minus the training artefacts.
"""
import argparse
import datetime
import importlib
import json
import math
from pathlib import Path
import signal
import sys
import time

import torch

from model.xRIR_cyl_oriented import build_xrir_exp06
from tools import exp06_profiles, exp06_train, provenance

REPO = Path(__file__).resolve().parents[1]
ENTRIES = {'trainer': 'train_xRIR_backbone', 'exp06_train': 'tools.exp06_train',
           'exp06_haa_finetune': 'tools.exp06_haa_finetune',
           'exp06_haa_eval': 'tools.exp06_haa_eval'}
GIB = 1024 ** 3
RUNNER = 'tools.exp06_smoke'
RUN_TYPES = ('smoke', 'probe')
OUTCOMES = ('ok', 'failed', 'aborted_alarm', 'aborted_memory')


class _Timeout(BaseException):
    """Raised inside the entry by SIGALRM; a BaseException so entries cannot swallow it."""


class _MemoryExceeded(BaseException):
    """Raised by the watchdog when the peak allocation passes the agreed ceiling."""


def check_budget(label, value):
    """Refuse a budget that is not a finite positive number (a 0 alarm disabled the timer)."""
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('{} must be a finite positive number, not {!r}'.format(label, value))
    return float(value)


def peak_bytes():
    """Peak CUDA allocation of this process; 0 where CUDA is unavailable."""
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


def _invoke(module, entry, argv):
    """The trainer's main takes no arguments; every exp_06 entry takes argv."""
    if entry != 'trainer':
        return module.main(list(argv))
    saved = list(sys.argv)
    sys.argv = [getattr(module, '__file__', 'train_xRIR_backbone.py')] + list(argv)
    try:
        return module.main()
    finally:
        sys.argv = saved


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def diagnostic_provenance(entry, argv, run_type, repo=REPO, approved=None,
                          reviewed_commit=None, exploratory=False):
    """Finding 6: a diagnostic binds its code the way a full run does, minus the training.

    The runner, not the entry, is the identity that matters here: ``trainer`` is the
    pinned script, which knows nothing about exp_06 provenance.
    """
    state = provenance.git_state(repo)
    commit = state['HEAD'] if reviewed_commit is None else reviewed_commit
    files, digest = exp06_profiles.closure_of('smoke', repo, commit)
    return dict(repo=str(repo), reviewed_commit=commit, run_type=run_type, entry=entry,
                source_closures={'diagnostic': {'entry_module': RUNNER, 'files': files,
                                                'sha256': digest}},
                orchestration_closures=exp06_train.orchestration_closures(repo, commit),
                code_digests=exp06_profiles.compute_code_digests(
                    repo, commit, keys=exp06_profiles.TRAINING_KEYS),
                approvals=exp06_train.approvals_binding(approved),
                exploratory=bool(exploratory), registry_sha256=exp06_train.registry_sha256(),
                git_state=state, environment=provenance.environment(),
                command=[entry] + list(argv))


def _publish(result, started, receipt):
    """One JSON line and, if asked, the receipt; both carry the same record."""
    result['wall_s'] = time.monotonic() - started
    result['peak_bytes'] = peak_bytes()
    result['ended_at'] = _now()
    if result['exit_status'] == 0 and result['peak_bytes'] > result['max_gb'] * GIB:
        result['exit_status'], result['outcome'] = 3, 'aborted_memory'  # retrospective peak
    print('EXP06_SMOKE ' + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    if receipt is not None:
        provenance.write_completion(Path(receipt), result)
    return result


def run_entry(entry, argv, receipt=None, alarm_seconds=300, max_gb=3.0, run_type='smoke',
              provenance_out=None, approved=None, reviewed_commit=None, exploratory=False):
    """Run one entry in-process; abort with status 3 on the alarm or the memory ceiling.

    ``exit_status`` is numeric: 0 for a completed entry, the integer an entry returns,
    3 for an alarm or memory abort and 1 for an exception. ``outcome`` names the reason.
    """
    if entry not in ENTRIES:
        raise ValueError('unknown smoke entry {!r}; choose from {}'.format(entry, sorted(ENTRIES)))
    if run_type not in RUN_TYPES:
        raise ValueError('a diagnostic run_type is smoke or probe, not {!r}'.format(run_type))
    alarm_seconds = check_budget('alarm_seconds', alarm_seconds)
    max_gb = check_budget('max_gb', max_gb)
    module = importlib.import_module(ENTRIES[entry])
    record = None
    if provenance_out is not None:
        record = diagnostic_provenance(entry, argv, run_type, REPO, approved,
                                       reviewed_commit, exploratory)
        provenance.write_manifest(Path(provenance_out), record)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = dict(schema_version=1, diagnostic=True, runner=RUNNER, entry=entry,
                  module=ENTRIES[entry], run_type=run_type, exploratory=bool(exploratory),
                  argv=list(argv), alarm_seconds=alarm_seconds, max_gb=max_gb,
                  entry_status=None, aborted_memory=False, outcome_detail=None,
                  runner_closure_sha256=exp06_profiles.code_digest(
                      'smoke', REPO, provenance.git_state(REPO)['HEAD']
                      if reviewed_commit is None else reviewed_commit),
                  provenance=None if record is None else {
                      'path': str(Path(provenance_out).resolve()),
                      'sha256': provenance.sha256_file(provenance_out)},
                  git_head=provenance.git_state(REPO)['HEAD'],
                  started_at=_now(), ended_at=None,
                  timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    started, ceiling = time.monotonic(), max_gb * GIB
    deadline = started + alarm_seconds

    def watchdog(signum, frame):
        """Enforce the ceiling while the entry runs, then the wall-clock deadline."""
        if peak_bytes() > ceiling:
            raise _MemoryExceeded('smoke exceeded its allocation ceiling')
        if time.monotonic() >= deadline:
            raise _Timeout('smoke exceeded its wall-clock budget')

    previous = signal.signal(signal.SIGALRM, watchdog)
    tick = min(1.0, alarm_seconds)  # periodic: sub-second alarms still fire on their deadline
    signal.setitimer(signal.ITIMER_REAL, tick, tick)
    try:
        try:
            status = _invoke(module, entry, argv)
            result['entry_status'] = status if type(status) is int else None
            result['exit_status'] = status if type(status) is int else 0
            result['outcome'] = 'failed' if result['exit_status'] else 'ok'
        finally:  # disarm before any handler runs, so nothing fires during publication
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    except _Timeout:
        result['exit_status'], result['outcome'] = 3, 'aborted_alarm'
    except _MemoryExceeded:
        result['exit_status'], result['outcome'] = 3, 'aborted_memory'
        result['aborted_memory'] = True
    except BaseException as error:
        result['exit_status'], result['outcome'] = 1, 'failed'
        result['outcome_detail'] = 'error: ' + type(error).__name__
        _publish(result, started, receipt)
        raise
    _publish(result, started, receipt)
    if result['exit_status'] != 0:
        raise SystemExit(3)
    return result


def make_fixture(path, seed=0):
    """Seeded CPU initialisation of the oriented model, with a hash/registry sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.save(build_xrir_exp06('cylindrical_oriented', 8).state_dict(), str(path))
    record = dict(schema_version=1, diagnostic=True, path=str(path.resolve()), seed=seed,
                  backbone='cylindrical_oriented', num_shot=8,
                  sha256=provenance.sha256_file(path),
                  registry_sha256=exp06_train.registry_sha256(),
                  git_head=provenance.git_state(REPO)['HEAD'],
                  timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    provenance.write_completion(Path(str(path) + '.json'), record)
    return record


def build_parser():
    parser = argparse.ArgumentParser(description='Run one bounded exp_06 smoke in-process.')
    parser.add_argument('--entry', choices=sorted(ENTRIES))
    parser.add_argument('--receipt')
    parser.add_argument('--alarm-seconds', type=float, default=300)
    parser.add_argument('--max-gb', type=float, default=3.0)
    parser.add_argument('--run-type', choices=RUN_TYPES, default='smoke',
                        help='recorded in the receipt and in provenance.json')
    parser.add_argument('--provenance-out', help='where this diagnostic records its bindings')
    parser.add_argument('--approved', help='approved_digests.json the bindings are checked against')
    parser.add_argument('--reviewed-commit', help='the commit the launcher verified HEAD against')
    parser.add_argument('--exploratory', action='store_true',
                        help='the approvals need not admit this diagnostic; recorded as such')
    parser.add_argument('--make-fixture', help='write a seeded CPU fixture instead of running')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('argv', nargs=argparse.REMAINDER, help='after --, the entry\'s own argv')
    return parser


def main(argv=None):
    """Exit 0 on a completed smoke; the runner itself raises SystemExit(3) on an abort."""
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    child = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
    if args.make_fixture:
        make_fixture(args.make_fixture, args.seed)
        return 0
    if not args.entry:
        parser.error('--entry or --make-fixture is required')
    run_entry(args.entry, child, receipt=args.receipt, alarm_seconds=args.alarm_seconds,
              max_gb=args.max_gb, run_type=args.run_type, provenance_out=args.provenance_out,
              approved=args.approved, reviewed_commit=args.reviewed_commit,
              exploratory=args.exploratory)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
