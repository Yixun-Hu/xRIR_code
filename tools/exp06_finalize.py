"""Completion evidence for exp_06 runs, dispatched by run type (plan v4 sections 5, 6.4).

The entry points write ``provenance.json``; this external finalizer decides completion
after the child has exited and writes ``completion.json``. Evidence depends on the run
type recorded on the command line (and, for ``full``, in ``provenance.json``):

``full``      twelve-epoch pretraining: closure drift, recipe/production/derived schema,
              budget completeness, ``epoch_012.pth`` equal to ``last.pth['model']``,
              child status 0 and a closed log.
``smoke``/``probe``  no artifacts; labelled ``diagnostic`` and never admissible as an arm.
``haa_train`` one fine-tuning child: its args (heading binding and init hash in the
              heading frame), history, summary and both checkpoints.
``haa_eval``  one evaluation child: its args and the metrics/per-sample files of the one
              room it names, whose ``meta`` carries the backbone, checkpoint hash, frame
              and (in the heading frame) the heading.
``haa_job``   one seed of the pipeline: the complete set of child completions
              (stage 1, four stage 2, four evaluations; four evaluations for zero shot).

    python tools/exp06_finalize.py --run-dir <dir> --run-type full \
        --log <log> --child-exit 0 [--repo <path>] [--receipt <json>] \
        [--children <dir>...] [--expect finetune|zeroshot]

The command exits 0 after writing ``completion.json`` and 2 on any refusal. The
``preflight`` subcommand is the launcher's gate before it starts a child: HEAD at the
reviewed commit, a tree clean outside ``worklog/``, no live exp_06 launch, and (for
``full``/``probe``) a GPU with no compute apps.

    python tools/exp06_finalize.py preflight --mode full --gpu 1 \
        --reviewed-commit <sha> [--attempt-root <dir>] [--repo <path>]

Every failure raises ``ValueError`` naming its cause and writes nothing; a re-run
produces byte-identical bytes, and an existing completion that differs is refused.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from sim_to_real.haa_dataset import ROOMS
from tools import exp06_recipe, provenance

REPO = Path(__file__).resolve().parents[1]
MARKER = 'EXP06_CHILD_EXIT'
DIAGNOSTIC = ('smoke', 'probe')
RUN_TYPES = ('full', 'smoke', 'probe', 'haa_train', 'haa_eval', 'haa_job')
EXPECTATIONS = ('finetune', 'zeroshot')
LAUNCH_MODES = ('smoke', 'probe', 'full')
EXCLUSIVE_GPU_MODES = ('probe', 'full')
FULL_ARTIFACTS = ('provenance.json', 'args.json', 'history.jsonl', 'last.pth', 'epoch_012.pth')
HAA_TRAIN_ARTIFACTS = ('args.json', 'history.jsonl', 'summary.json', 'best.pth', 'last.pth')
FRAMES = ('room', 'heading')
WIDTH = 512
EPOCH_CHECKPOINT = 'epoch_{:03d}.pth'.format(exp06_recipe.EXP01_RECIPE['epochs'])


def _require(ok, cause):
    if not ok:
        raise ValueError(cause)


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError('unreadable {}: {}'.format(label, error)) from error


def closure_digest(files):
    """The exp_03-compatible digest over [[path, reviewed blob], ...]."""
    return hashlib.sha256(json.dumps([[f['path'], f['reviewed_blob_sha256']] for f in files],
                                     sort_keys=True).encode()).hexdigest()


def closed_log(log, child_exit):
    """Require the launcher's end marker as the log's last line, agreeing on the status."""
    try:
        text = Path(log).read_text()
    except OSError as error:
        raise ValueError('unreadable log: {}'.format(error)) from error
    lines = text.splitlines()
    _require(lines and lines[-1].startswith(MARKER + ' '),
             'log does not end with the ' + MARKER + ' marker (child still writing?)')
    parts = lines[-1].split()
    _require(len(parts) == 3, 'malformed ' + MARKER + ' marker: ' + lines[-1])
    try:
        code, stamp = int(parts[1]), datetime.datetime.fromisoformat(parts[2])
    except ValueError as error:
        raise ValueError('malformed ' + MARKER + ' marker: {}'.format(error)) from error
    _require(stamp.tzinfo is not None, MARKER + ' marker needs a timezone-aware timestamp')
    _require(code == child_exit,
             'log marker status {} differs from the reported child status {}'.format(code, child_exit))
    return {'path': str(Path(log).resolve()), 'sha256': provenance.sha256_file(log)}, parts[2]


def artifacts(run_dir, names):
    """Hash every required artifact; a missing one is refused by name."""
    hashes = {}
    for name in names:
        path = Path(run_dir) / name
        _require(path.is_file(), 'missing artifact {} in {}'.format(name, run_dir))
        hashes[name] = provenance.sha256_file(path)
    return hashes


def full_evidence(run_dir, repo):
    """Verify the twelve-epoch pretraining contract of plan section 5."""
    run_dir = Path(run_dir)
    hashes = artifacts(run_dir, FULL_ARTIFACTS)  # every file must exist before it is parsed
    record = _read_json(run_dir / 'provenance.json', 'provenance.json')
    _require(record.get('run_type') == 'full',
             'provenance run_type is {!r}, not full'.format(record.get('run_type')))
    try:
        closure = record['source_closures']['training']
        commit = record['reviewed_commit']
        paths = [entry['path'] for entry in closure['files']]
    except (KeyError, TypeError) as error:
        raise ValueError('provenance.json has no training source closure: {}'.format(error)) from error
    _require(closure.get('sha256') == closure_digest(closure['files']),
             'recorded source closure digest does not match its own file list')
    _require(provenance.closure_record(paths, commit, repo)[1] == closure['sha256'],
             'source closure drift: the reviewed blobs differ from the recorded digest')
    args = _read_json(run_dir / 'args.json', 'args.json')
    rows = []
    for number, line in enumerate(Path(run_dir / 'history.jsonl').read_text().splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except ValueError as error:
            raise ValueError('history.jsonl line {} is not JSON: {}'.format(number, error)) from error
    last = torch.load(str(run_dir / 'last.pth'), map_location='cpu')
    meta = {key: last[key] for key in ('epoch', 'batch_idx') if key in last}
    deviations = exp06_recipe.check_all(args, history_rows=rows, last_meta=meta)
    _require(not deviations, 'schema deviations: ' + '; '.join(deviations))
    _require(last.get('args') == args, 'the args recorded in last.pth differ from args.json')
    state = torch.load(str(run_dir / EPOCH_CHECKPOINT), map_location='cpu')
    model = last['model']
    _require(set(state) == set(model), EPOCH_CHECKPOINT + ' has a different parameter set than last.pth')
    _require(all(torch.equal(state[key], model[key]) for key in state),
             EPOCH_CHECKPOINT + ' differs tensor-wise from last.pth["model"]')
    return dict(artifacts=hashes, epochs=len(rows),
                backbone=args['backbone'], source_closure_sha256=closure['sha256'],
                registry_sha256=record.get('registry_sha256'),
                git_head=record.get('git_state', {}).get('HEAD'),
                checkpoint={'path': EPOCH_CHECKPOINT, 'sha256': hashes[EPOCH_CHECKPOINT],
                            'epoch': exp06_recipe.EXP01_RECIPE['epochs']})


def diagnostic_evidence(run_dir, receipt):
    """A smoke or probe proves nothing about an arm; keep its receipt, demand no artifact."""
    fields = dict(artifacts={})
    if receipt is not None:
        path = Path(receipt)
        _require(path.is_file(), 'missing smoke receipt: {}'.format(receipt))
        fields['receipt'] = {'path': str(path.resolve()), 'sha256': provenance.sha256_file(path)}
    return fields


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


def _rooms_and_frame(args):
    """Every HAA child records the rooms it ran and the frame its geometry was in."""
    rooms = args.get('rooms')
    _require(isinstance(rooms, list) and rooms and all(isinstance(room, str) for room in rooms),
             'args.json must record a nonempty list of rooms')
    frame = args.get('frame')
    _require(frame in FRAMES, 'args.json must record frame room|heading, not {!r}'.format(frame))
    return rooms, frame


def _heading_binding(args, rooms, frame):
    """In the heading frame every room needs a bound heading record; the room frame needs none."""
    if frame != 'heading':
        return None
    heading = args.get('heading')
    _require(isinstance(heading, dict) and set(rooms) <= set(heading),
             'the heading frame requires a heading record for every room')
    for room in rooms:
        entry = heading[room]
        _require(isinstance(entry, dict) and type(entry.get('k')) is int
                 and 0 <= entry['k'] < WIDTH and _is_sha256(entry.get('sha256')),
                 'invalid heading binding for {}: {!r}'.format(room, entry))
    return {room: heading[room] for room in rooms}


def haa_train_evidence(run_dir):
    """One fine-tuning child of the HAA pipeline (plan section 6.2)."""
    hashes = artifacts(run_dir, HAA_TRAIN_ARTIFACTS)
    args = _read_json(Path(run_dir) / 'args.json', 'args.json')
    rooms, frame = _rooms_and_frame(args)
    heading = _heading_binding(args, rooms, frame)
    if frame == 'heading':
        _require(_is_sha256(args.get('init_sha256')),
                 'heading-frame fine-tuning must record init_sha256 of its initialisation')
    return dict(artifacts=hashes, rooms=rooms, frame=frame, heading=heading,
                backbone=args.get('backbone'), init_sha256=args.get('init_sha256'))


def haa_eval_evidence(run_dir):
    """One evaluation child: exactly one room, with its per-sample provenance meta."""
    args = _read_json(Path(run_dir) / 'args.json', 'args.json')
    rooms, frame = _rooms_and_frame(args)
    _require(len(rooms) == 1, 'an evaluation child covers exactly one room, not {}'.format(rooms))
    room, tag = rooms[0], args.get('tag', '')
    names = ('args.json', 'metrics_{}{}.json'.format(room, tag),
             'per_sample_{}{}.json'.format(room, tag))
    hashes = artifacts(run_dir, names)
    heading = _heading_binding(args, rooms, frame)
    meta = _read_json(Path(run_dir) / names[2], names[2]).get('meta')
    required = ('backbone', 'checkpoint_sha256', 'frame') + (('heading',) if frame == 'heading' else ())
    _require(isinstance(meta, dict) and all(key in meta for key in required),
             'per-sample meta must record ' + ', '.join(required))
    _require(meta['backbone'] == args.get('backbone') and meta['frame'] == frame,
             'per-sample meta backbone/frame differ from args.json')
    return dict(artifacts=hashes, room=room, frame=frame, heading=heading,
                backbone=args.get('backbone'), checkpoint_sha256=meta['checkpoint_sha256'])


def expected_children(expect):
    """The exclusive per-child directories one pipeline seed must complete."""
    _require(expect in EXPECTATIONS, 'a job needs --expect finetune|zeroshot, not {!r}'.format(expect))
    evaluations = tuple('eval/' + room for room in ROOMS)
    if expect == 'zeroshot':
        return evaluations
    return ('stage1',) + tuple('stage2_' + room for room in ROOMS) + evaluations


def haa_job_evidence(run_dir, children, expect):
    """Bind one seed's children: every expected directory, each with its own completion."""
    expected, job = set(expected_children(expect)), Path(run_dir).resolve()
    seen = {}
    for child in children:
        path = Path(child).resolve()
        try:
            name = path.relative_to(job).as_posix()
        except ValueError as error:
            raise ValueError('child {} lies outside the job directory'.format(child)) from error
        completion = path / 'completion.json'
        _require(completion.is_file(), 'child {} has no completion.json'.format(name))
        record = _read_json(completion, name + '/completion.json')
        _require(record.get('run_type') in ('haa_train', 'haa_eval'),
                 'child {} has run type {!r}'.format(name, record.get('run_type')))
        _require(record.get('admissible_arm') is True, 'child {} is not admissible'.format(name))
        seen[name] = provenance.sha256_file(completion)
    missing = sorted(expected - set(seen))
    _require(not missing, 'job is missing children: ' + ', '.join(missing))
    unexpected = sorted(set(seen) - expected)
    _require(not unexpected, 'unexpected children: ' + ', '.join(unexpected))
    return dict(artifacts={}, children=seen, expect=expect)


def write_completion(path, fields):
    """Publish once; a re-run must produce the same bytes, a different result is refused."""
    payload = json.dumps(fields, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    if Path(path).exists():
        _require(Path(path).read_bytes() == payload,
                 'completion.json already exists and differs from this result')
        return fields
    provenance.write_completion(path, fields)
    return fields


def finalize(run_dir, run_type, log, child_exit, repo=REPO, receipt=None,
             children=(), expect=None):
    """Verify one child's evidence for its run type and write completion.json."""
    run_dir = Path(run_dir)
    _require(run_type in RUN_TYPES, 'unknown run type: {!r}'.format(run_type))
    _require(run_dir.is_dir(), 'run directory does not exist: {}'.format(run_dir))
    _require(type(child_exit) is int, 'child status must be an integer')
    log_record, child_exit_time = closed_log(log, child_exit)
    diagnostic = run_type in DIAGNOSTIC
    if not diagnostic:
        _require(child_exit == 0, 'child exited with status {}'.format(child_exit))
    fields = dict(schema_version=1, run_type=run_type, run_dir=str(run_dir.resolve()),
                  repo=str(Path(repo).resolve()), child_exit=child_exit,
                  child_exit_time=child_exit_time, log=log_record,
                  diagnostic=diagnostic, admissible_arm=not diagnostic)
    fields.update(diagnostic_evidence(run_dir, receipt) if diagnostic
                  else full_evidence(run_dir, repo) if run_type == 'full'
                  else haa_train_evidence(run_dir) if run_type == 'haa_train'
                  else haa_eval_evidence(run_dir) if run_type == 'haa_eval'
                  else haa_job_evidence(run_dir, children, expect))
    return write_completion(run_dir / 'completion.json', fields)


def gpu_compute_apps(gpu):
    """The pids nvidia-smi reports on one card; an unqueryable card is refused."""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader', '-i', str(gpu)],
            text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError('cannot query GPU {}: {}'.format(gpu, error)) from error
    return [line.strip() for line in output.splitlines() if line.strip()]


def live_launches(attempt_root):
    """Refuse a second launch while any attempt's launch.pid still names a live process."""
    live = []
    for pid_file in sorted(Path(attempt_root).glob('*/launch.pid')):
        text = pid_file.read_text().split()
        try:
            pid = int(text[0])
        except (IndexError, ValueError) as error:
            raise ValueError('unreadable launch.pid at {}: {}'.format(pid_file, error)) from error
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            pass  # A live process owned by somebody else is still a live process.
        live.append({'path': str(pid_file), 'pid': pid})
    return live


def preflight(mode, gpu, reviewed_commit, attempt_root=None, repo=REPO):
    """Gate a launch: reviewed commit, clean tree, no live launch, and a free card."""
    _require(mode in LAUNCH_MODES, 'unknown launch mode: {!r}'.format(mode))
    state = provenance.checked_git_state(repo, confirmatory=True)
    _require(state['HEAD'] == reviewed_commit,
             'HEAD {} is not the reviewed commit {!r} (full 40-hex sha required)'.format(
                 state['HEAD'], reviewed_commit))
    running = live_launches(attempt_root) if attempt_root and Path(attempt_root).is_dir() else []
    _require(not running, 'another exp_06 launch is alive: {}'.format(running))
    apps = gpu_compute_apps(gpu) if mode in EXCLUSIVE_GPU_MODES else None
    _require(not apps, 'GPU {} is busy with compute apps {}'.format(gpu, apps))
    return dict(mode=mode, gpu=gpu, reviewed_commit=reviewed_commit, git_state=state,
                attempt_root=str(attempt_root) if attempt_root else None,
                gpu_compute_apps=apps, live_launches=running)


def preflight_main(argv):
    """Exit 0 with the record on stdout, 2 with the named cause on stderr."""
    parser = argparse.ArgumentParser(description='Gate one exp_06 launch.')
    parser.add_argument('--mode', choices=LAUNCH_MODES, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--reviewed-commit', required=True)
    parser.add_argument('--attempt-root')
    parser.add_argument('--repo', default=str(REPO))
    args = parser.parse_args(argv)
    try:
        record = preflight(args.mode, args.gpu, args.reviewed_commit, args.attempt_root, args.repo)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print('EXP06_PREFLIGHT_REFUSED ' + str(error), file=sys.stderr, flush=True)
        return 2
    print('EXP06_PREFLIGHT_OK ' + json.dumps(record, sort_keys=True), flush=True)
    return 0


def build_parser():
    """One finalization of one child or job; the launcher supplies the child's status."""
    parser = argparse.ArgumentParser(description='Write exp_06 completion evidence.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--run-type', choices=RUN_TYPES, required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--child-exit', type=int, required=True)
    parser.add_argument('--repo', default=str(REPO))
    parser.add_argument('--receipt', help='smoke/probe receipt to bind')
    parser.add_argument('--children', nargs='+', default=(), help='haa_job: the child directories')
    parser.add_argument('--expect', choices=EXPECTATIONS, help='haa_job: which child set is required')
    return parser


def main(argv=None):
    """Exit 0 after writing completion.json, 2 on any refusal (nothing written)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['preflight']:
        return preflight_main(argv[1:])
    if argv[:1] == ['finalize']:
        argv = argv[1:]
    args = build_parser().parse_args(argv)
    try:
        fields = finalize(args.run_dir, args.run_type, args.log, args.child_exit, repo=args.repo,
                          receipt=args.receipt, children=args.children, expect=args.expect)
    except (OSError, ValueError) as error:
        print('EXP06_FINALIZE_REFUSED ' + str(error), file=sys.stderr, flush=True)
        return 2
    print('EXP06_FINALIZE_OK ' + json.dumps({key: fields[key] for key in
        ('run_type', 'run_dir', 'child_exit', 'admissible_arm')}, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
