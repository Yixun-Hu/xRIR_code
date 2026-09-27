"""Bounded in-process runner for exp_11's diagnostics, with enumerated kinds.

Codex round-2 change 5: the launcher's probe and the two HAA smokes are **named** kinds,
each with an explicit budget (wall-clock alarm, allocation ceiling), an artifact contract
(the files its run directory must hold afterwards) and **permanent non-admissibility** --
every receipt carries ``diagnostic: true`` and ``admissible_arm: false``, so no diagnostic
can be mistaken for an arm however it is later re-read.

    kind              entry                          alarm    ceiling  artifacts
    ----------------  -----------------------------  -------  -------  -----------------
    probe             tools.exp11_train              900 s    40 GiB   provenance.json
    haa_train_smoke   tools.exp11_haa_finetune       1800 s   40 GiB   the stage files
    haa_eval_smoke    tools.exp11_haa_eval           1800 s   40 GiB   the room files

The bounded machinery is the pinned one: ``tools/exp06_smoke.py``'s budget check, peak
allocation, alarm/watchdog invocation and publication are imported unchanged, and only
what is exp_11's -- the kinds, the entry map, the runner identity, the approvals and the
closure digests -- is owned here. ``--make-fixture`` writes a CPU-initialised
``simple_oriented`` (or ``simple_adapter``) state dict for the HAA smokes: the exp_01
checkpoints cannot load into the five-channel patch embedding.

    python tools/exp11_smoke.py --kind probe -- --backbone simple_oriented --no-save ...
    python tools/exp11_smoke.py --make-fixture ckpt/exp11/_smoke/fixture_simpor.pth
"""
import argparse
import datetime
import importlib
import json
from pathlib import Path
import signal
import sys
import time

import torch

from model.xrir_exp11_registry import build_xrir_exp11, registry_sha256
from tools import exp06_smoke as bounded
from tools import exp11_profiles, exp11_train, provenance

REPO = Path(__file__).resolve().parents[1]
RUNNER = 'tools.exp11_smoke'
GIB = bounded.GIB
OUTCOMES = bounded.OUTCOMES
# Every exp_11 diagnostic is one of these; each names its entry, its budgets and what it
# must leave behind. They are never admissible arms.
KINDS = {
    'probe': {'entry': 'tools.exp11_train', 'run_type': 'probe',
              'alarm_seconds': 900.0, 'max_gb': 40.0,
              'artifacts': ('provenance.json',)},
    'haa_train_smoke': {'entry': 'tools.exp11_haa_finetune', 'run_type': 'haa_smoke_train',
                        'alarm_seconds': 1800.0, 'max_gb': 40.0,
                        'artifacts': ('provenance.json', 'args.json', 'history.jsonl',
                                      'summary.json', 'best.pth', 'last.pth')},
    'haa_eval_smoke': {'entry': 'tools.exp11_haa_eval', 'run_type': 'haa_smoke_eval',
                       'alarm_seconds': 1800.0, 'max_gb': 40.0,
                       'artifacts': ('provenance.json', 'args.json')},
}
KIND_NAMES = tuple(sorted(KINDS))
RUN_TYPES = tuple(sorted({spec['run_type'] for spec in KINDS.values()}))
FIXTURE_BACKBONES = ('simple_oriented', 'simple_adapter')


def kind_spec(kind):
    """One enumerated diagnostic kind; an unregistered name is refused, never guessed."""
    if kind not in KINDS:
        raise ValueError('unknown exp_11 diagnostic kind {!r}; choose from {}'.format(
            kind, ', '.join(KIND_NAMES)))
    return dict(KINDS[kind])


def diagnostic_provenance(kind, argv, repo=REPO, approved=None, reviewed_commit=None,
                          exploratory=False):
    """A diagnostic binds its code the way a full run does, minus the training artefacts."""
    spec = kind_spec(kind)
    state = provenance.git_state(repo)
    commit = state['HEAD'] if reviewed_commit is None else reviewed_commit
    files, digest = exp11_profiles.closure_of('smoke', repo, commit)
    return dict(repo=str(repo), reviewed_commit=commit, run_type=spec['run_type'],
                entry=spec['entry'], kind=kind, diagnostic=True, admissible_arm=False,
                source_closures={'diagnostic': {'entry_module': RUNNER, 'files': files,
                                                'sha256': digest}},
                orchestration_closures=exp11_train.orchestration_closures(repo, commit),
                code_digests=exp11_profiles.compute_code_digests(
                    repo, commit, keys=exp11_profiles.TRAINING_KEYS),
                approvals=exp11_train.approvals_binding(approved),
                exploratory=bool(exploratory), registry_sha256=registry_sha256(),
                git_state=state, environment=provenance.environment(),
                command=[kind] + list(argv))


def run_kind(kind, argv, receipt=None, alarm_seconds=None, max_gb=None,
             provenance_out=None, approved=None, reviewed_commit=None, exploratory=False):
    """Run one enumerated kind in-process under its own budget; abort with status 3.

    A caller may tighten a budget but never loosen it: the kind's registered ceiling is
    the maximum, so a diagnostic cannot quietly grow past what the plan allowed.
    """
    spec = kind_spec(kind)
    alarm = bounded.check_budget('alarm_seconds',
                                 spec['alarm_seconds'] if alarm_seconds is None
                                 else alarm_seconds)
    ceiling = bounded.check_budget('max_gb', spec['max_gb'] if max_gb is None else max_gb)
    if alarm > spec['alarm_seconds'] or ceiling > spec['max_gb']:
        raise ValueError('the {} budget is {} s / {} GiB; {} s / {} GiB exceeds it'.format(
            kind, spec['alarm_seconds'], spec['max_gb'], alarm, ceiling))
    module = importlib.import_module(spec['entry'])
    record = None
    if provenance_out is not None:
        record = diagnostic_provenance(kind, argv, REPO, approved, reviewed_commit,
                                       exploratory)
        provenance.write_manifest(Path(provenance_out), record)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = dict(schema_version=1, diagnostic=True, admissible_arm=False, runner=RUNNER,
                  kind=kind, entry=spec['entry'], module=spec['entry'],
                  run_type=spec['run_type'], exploratory=bool(exploratory), argv=list(argv),
                  alarm_seconds=alarm, max_gb=ceiling, artifacts=list(spec['artifacts']),
                  entry_status=None, aborted_memory=False, outcome_detail=None,
                  runner_closure_sha256=exp11_profiles.code_digest(
                      'smoke', REPO, provenance.git_state(REPO)['HEAD']
                      if reviewed_commit is None else reviewed_commit),
                  provenance=None if record is None else {
                      'path': str(Path(provenance_out).resolve()),
                      'sha256': provenance.sha256_file(provenance_out)},
                  git_head=provenance.git_state(REPO)['HEAD'],
                  started_at=bounded._now(), ended_at=None,
                  timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    started, limit = time.monotonic(), ceiling * GIB
    deadline = started + alarm

    def watchdog(signum, frame):
        if bounded.peak_bytes() > limit:
            raise bounded._MemoryExceeded('smoke exceeded its allocation ceiling')
        if time.monotonic() >= deadline:
            raise bounded._Timeout('smoke exceeded its wall-clock budget')

    previous = signal.signal(signal.SIGALRM, watchdog)
    tick = min(1.0, alarm)
    signal.setitimer(signal.ITIMER_REAL, tick, tick)
    try:
        try:
            status = module.main(list(argv))
            result['entry_status'] = status if type(status) is int else None
            result['exit_status'] = status if type(status) is int else 0
            result['outcome'] = 'failed' if result['exit_status'] else 'ok'
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    except bounded._Timeout:
        result['exit_status'], result['outcome'] = 3, 'aborted_alarm'
    except bounded._MemoryExceeded:
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


def _publish(result, started, receipt):
    """One ``EXP11_SMOKE`` line and, if asked, the receipt; both carry the same record."""
    result['wall_s'] = time.monotonic() - started
    result['peak_bytes'] = bounded.peak_bytes()
    result['ended_at'] = bounded._now()
    if result['exit_status'] == 0 and result['peak_bytes'] > result['max_gb'] * GIB:
        result['exit_status'], result['outcome'] = 3, 'aborted_memory'
    elif result['exit_status'] == 0 and result['wall_s'] > result['alarm_seconds']:
        result['exit_status'], result['outcome'] = 3, 'aborted_alarm'
    print('EXP11_SMOKE ' + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    if receipt is not None:
        provenance.write_completion(Path(receipt), result)
    return result


def make_fixture(path, backbone='simple_oriented', seed=0):
    """Seeded CPU initialisation of one exp_11 model, with a hash/registry sidecar."""
    if backbone not in FIXTURE_BACKBONES:
        raise ValueError('a fixture is one of {}, not {!r}'.format(
            ', '.join(FIXTURE_BACKBONES), backbone))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.save(build_xrir_exp11(backbone, 8).state_dict(), str(path))
    record = dict(schema_version=1, diagnostic=True, admissible_arm=False,
                  path=str(path.resolve()), seed=seed, backbone=backbone, num_shot=8,
                  sha256=provenance.sha256_file(path), registry_sha256=registry_sha256(),
                  git_head=provenance.git_state(REPO)['HEAD'],
                  timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    provenance.write_completion(Path(str(path) + '.json'), record)
    return record


def build_parser():
    parser = argparse.ArgumentParser(description='Run one bounded exp_11 diagnostic.')
    parser.add_argument('--kind', choices=KIND_NAMES)
    parser.add_argument('--receipt')
    parser.add_argument('--alarm-seconds', type=float, default=None,
                        help="at most the kind's registered budget")
    parser.add_argument('--max-gb', type=float, default=None,
                        help="at most the kind's registered ceiling")
    parser.add_argument('--provenance-out')
    parser.add_argument('--approved')
    parser.add_argument('--reviewed-commit')
    parser.add_argument('--exploratory', action='store_true')
    parser.add_argument('--make-fixture')
    parser.add_argument('--fixture-backbone', choices=FIXTURE_BACKBONES,
                        default='simple_oriented')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('argv', nargs=argparse.REMAINDER,
                        help="after --, the entry's own argv")
    return parser


def main(argv=None):
    """Exit 0 on a completed diagnostic; the runner raises SystemExit(3) on an abort."""
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    child = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
    if args.make_fixture:
        make_fixture(args.make_fixture, args.fixture_backbone, args.seed)
        return 0
    if not args.kind:
        parser.error('--kind or --make-fixture is required')
    run_kind(args.kind, child, receipt=args.receipt, alarm_seconds=args.alarm_seconds,
             max_gb=args.max_gb, provenance_out=args.provenance_out, approved=args.approved,
             reviewed_commit=args.reviewed_commit, exploratory=args.exploratory)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
