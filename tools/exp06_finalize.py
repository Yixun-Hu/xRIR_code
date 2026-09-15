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
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

from model.xRIR_cyl_oriented import build_xrir_exp06
from sim_to_real.haa_dataset import ROOMS
from tools import exp06_heading, exp06_recipe, provenance

REPO = Path(__file__).resolve().parents[1]
MARKER = 'EXP06_CHILD_EXIT'
DIAGNOSTIC = ('smoke', 'probe')
RUN_TYPES = ('full', 'smoke', 'probe', 'haa_train', 'haa_eval', 'haa_job')
EXPECTATIONS = ('finetune', 'zeroshot')
LAUNCH_MODES = ('smoke', 'probe', 'full', 'finalize')
EXCLUSIVE_GPU_MODES = ('probe', 'full')
FULL_ARTIFACTS = ('provenance.json', 'args.json', 'history.jsonl', 'last.pth', 'epoch_012.pth')
HAA_TRAIN_ARTIFACTS = ('provenance.json', 'args.json', 'history.jsonl', 'summary.json',
                       'best.pth', 'last.pth')
HAA_SUMMARY = ('best_val_loss', 'best_epoch')
HEADING_DECISIONS = ('estimated', 'override')
FRAMES = ('room', 'heading')
REQUIRED_PROVENANCE = ('run_type', 'repo', 'reviewed_commit', 'source_closures',
                       'registry_sha256', 'git_state', 'environment', 'command',
                       'effective_args')
ENTRY_MODULES = {'full': 'tools.exp06_train', 'haa_train': 'tools.exp06_haa_finetune',
                 'haa_eval': 'tools.exp06_haa_eval'}
BACKBONES = tuple(sorted(build_xrir_exp06.__globals__['BACKBONES_EXP06']))
IDENTITY_KEYS = ('train_data_identity', 'data_identity')
WIDTH = 512
EPOCH_CHECKPOINT = 'epoch_{:03d}.pth'.format(exp06_recipe.EXP01_RECIPE['epochs'])


def _require(ok, cause):
    if not ok:
        raise ValueError(cause)


def _read_json(path, label):
    """Decode one JSON object; a malformed or non-object file is a named refusal."""
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError('unreadable {}: {}'.format(label, error)) from error
    _require(isinstance(value, dict), '{} is not a JSON object'.format(label))
    return value


def _load_torch(path, label):
    """Load a checkpoint container; every decoding failure becomes a named refusal."""
    try:
        loaded = torch.load(str(path), map_location='cpu')
    except Exception as error:  # torch surfaces pickle, zip, EOF and type errors alike
        raise ValueError('unreadable {}: {}: {}'.format(label, type(error).__name__, error)) from error
    _require(isinstance(loaded, dict), '{} is not a checkpoint mapping'.format(label))
    return loaded


def _state_dict(mapping, label):
    """Require a parameter mapping of names to tensors before anything compares it."""
    _require(isinstance(mapping, dict) and mapping, '{} is not a parameter mapping'.format(label))
    bad = sorted(key for key, value in mapping.items()
                 if not isinstance(key, str) or not torch.is_tensor(value))
    _require(not bad, '{} has non-tensor entries: {}'.format(label, bad[:4]))
    return mapping


def _history_rows(path, label):
    """Parse history.jsonl into one object per line, naming the first malformed line."""
    rows = []
    try:
        text = Path(path).read_text()
    except OSError as error:
        raise ValueError('unreadable {}: {}'.format(label, error)) from error
    for number, line in enumerate(text.splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError as error:
            raise ValueError('{} line {} is not JSON: {}'.format(label, number, error)) from error
        _require(isinstance(row, dict), '{} line {} is not a JSON object'.format(label, number))
        rows.append(row)
    return rows


def closure_digest(files):
    """The exp_03-compatible digest over [[path, reviewed blob], ...]."""
    return hashlib.sha256(json.dumps([[f['path'], f['reviewed_blob_sha256']] for f in files],
                                     sort_keys=True).encode()).hexdigest()


def closed_log(log, child_exit):
    """Require the launcher's end marker as the log's last line, agreeing on the status.

    The bytes are read once and hashed here, so the receipt comparison and the recorded
    digest describe exactly the content that was validated.
    """
    try:
        data = Path(log).read_bytes()
    except OSError as error:
        raise ValueError('unreadable log: {}'.format(error)) from error
    digest = hashlib.sha256(data).hexdigest()
    lines = data.decode('utf-8', 'replace').splitlines()
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
    return {'path': str(Path(log).resolve()), 'sha256': digest}, parts[2], digest


def child_exit_receipt(run_dir, child_exit, log_digest):
    """The launcher's proof that every writer had exited when it hashed the log."""
    path = Path(run_dir) / 'child_exit.json'
    _require(path.is_file(), 'missing child_exit.json in {}'.format(run_dir))
    receipt = _read_json(path, 'child_exit.json')
    missing = [key for key in ('child_pid', 'status', 'ended_at', 'log_sha256_after_marker')
               if key not in receipt]
    _require(not missing, 'child_exit.json is incomplete: missing ' + ', '.join(missing))
    _require(type(receipt['status']) is int and receipt['status'] == child_exit,
             'child_exit.json records status {!r}, not the reported {}'.format(
                 receipt['status'], child_exit))
    _require(receipt['log_sha256_after_marker'] == log_digest,
             'child_exit.json binds a different log: {} is not the validated {}'.format(
                 receipt['log_sha256_after_marker'], log_digest))
    return {'path': str(path.resolve()), 'sha256': provenance.sha256_file(path),
            'child_pid': receipt['child_pid'], 'ended_at': receipt['ended_at']}


def refuse_live_launch(run_dir):
    """No mode, recovery included, may certify a run whose launcher is still running."""
    path = Path(run_dir) / 'launch.pid'
    if not path.is_file():
        return None
    try:
        pid = int(path.read_text().split()[0])
    except (IndexError, ValueError) as error:
        raise ValueError('unreadable launch.pid at {}: {}'.format(path, error)) from error
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return pid
    except PermissionError:
        pass  # Somebody else's live process is still a live process.
    raise ValueError('the launch pid {} of {} is still alive'.format(pid, run_dir))


@functools.lru_cache(maxsize=None)
def closure_paths(entry_module, repo):
    """Re-import the recorded entry hermetically; an unresolvable module is a refusal.

    One finalization imports one entry, so the cache only spares repeated calls inside a
    single process (tests); production calls it once.
    """
    try:
        return tuple(provenance.source_closure(entry_module, repo))
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError('cannot recompute the source closure of {}: {}'.format(
            entry_module, error)) from error


def load_provenance(run_dir, run_type):
    """The whole execution record, or a refusal naming the field that is missing."""
    record = _read_json(Path(run_dir) / 'provenance.json', 'provenance.json')
    missing = [key for key in REQUIRED_PROVENANCE if record.get(key) is None]
    _require(not missing, 'provenance.json is incomplete: missing ' + ', '.join(missing))
    _require(record['run_type'] == run_type,
             'provenance run_type is {!r}, not {}'.format(record['run_type'], run_type))
    present = [key for key in IDENTITY_KEYS if isinstance(record.get(key), dict)]
    _require(present, 'provenance.json records no train_data_identity/data_identity')
    for key in present:
        identity = record[key]
        _require(isinstance(identity.get('inventory'), list) and identity.get('data_root'),
                 'provenance.json records no {} inventory'.format(key))
    _require(isinstance(record['effective_args'], dict),
             'provenance.json records no effective_args mapping (startup arguments)')
    return record


def verify_source_closure(record, run_type, repo):
    """Three-way: the working tree now, the bytes at spawn, and the reviewed blobs."""
    closures = record['source_closures']
    _require(isinstance(closures, dict) and len(closures) == 1,
             'provenance.json must record exactly one source closure')
    name, closure = sorted(closures.items())[0]
    _require(isinstance(closure, dict), 'source closure {} is not a record'.format(name))
    entry = closure.get('entry_module')
    _require(entry == ENTRY_MODULES[run_type], 'source closure entry module is {!r}, not {!r}'
             .format(entry, ENTRY_MODULES[run_type]))
    files = closure.get('files')
    _require(isinstance(files, list) and files, 'the recorded source closure is empty')
    for item in files:
        _require(isinstance(item, dict) and isinstance(item.get('path'), str)
                 and _is_sha256(item.get('working_tree_sha256'))
                 and _is_sha256(item.get('reviewed_blob_sha256')),
                 'incomplete source closure file record: {!r}'.format(item))
    _require(closure.get('sha256') == closure_digest(files),
             'recorded source closure digest does not match its own file list')
    current = closure_paths(entry, str(Path(repo).resolve()))
    recorded = sorted(item['path'] for item in files)
    _require(sorted(current) == recorded, 'source closure membership changed: '
             + ', '.join(sorted(set(current) ^ set(recorded))))
    fresh, digest = provenance.closure_record(list(current), record['reviewed_commit'], repo)
    _require(digest == closure['sha256'],
             'source closure drift: the reviewed blobs differ from the recorded digest')
    now = {item['path']: item for item in fresh}
    for item in files:
        seen = now[item['path']]
        agreed = {seen['working_tree_sha256'], item['working_tree_sha256'],
                  item['reviewed_blob_sha256'], seen['reviewed_blob_sha256']}
        _require(len(agreed) == 1, 'source drift at {}: working tree {}, recorded {}, reviewed {}'
                 .format(item['path'], seen['working_tree_sha256'],
                         item['working_tree_sha256'], item['reviewed_blob_sha256']))
    return name, closure


def revalidate_inputs(record, repo, required=('train_data_identity', 'source_closures')):
    """Rehash the declared data inventory and mutable inputs through tools.provenance."""
    manifest = dict(record, repo=str(Path(repo).resolve()))
    mismatches = provenance.revalidate(manifest, required=required)
    _require(not mismatches, 'input revalidation failed: ' + ', '.join(sorted(mismatches)[:8]))


def artifacts(run_dir, names):
    """Hash every required artifact; a missing one is refused by name."""
    hashes = {}
    for name in names:
        path = Path(run_dir) / name
        _require(path.is_file(), 'missing artifact {} in {}'.format(name, run_dir))
        hashes[name] = provenance.sha256_file(path)
    return hashes


def check_argument_sources(record, args, last, rows, meta):
    """Blocker 2: the startup, retained and checkpoint arguments must all agree.

    Each of the three copies is validated on its own against the current-run schema
    (operational and ``exp06`` fields required), then compared field by field with
    ``exp06_recipe.compare_sources``, which never conflates ``True`` with ``1``.
    """
    effective = record.get('effective_args')
    _require(isinstance(effective, dict),
             'provenance.json records no effective_args mapping (startup arguments)')
    checkpoint = last.get('args')
    _require(isinstance(checkpoint, dict), 'last.pth records no args mapping')
    sources = {'args.json': args, 'last.pth[args]': checkpoint,
               'provenance.effective_args': effective}
    for name in sorted(sources):
        deviations = exp06_recipe.check_all(sources[name])
        _require(not deviations, '{} schema deviations: {}'.format(name, '; '.join(deviations)))
    budget = exp06_recipe.check_budget(rows, meta)
    _require(not budget, 'budget deviations: ' + '; '.join(budget))
    disagreements = exp06_recipe.compare_sources(sources)
    _require(not disagreements, 'recorded arguments disagree: ' + '; '.join(disagreements))
    return sources


def full_evidence(run_dir, repo):
    """Verify the twelve-epoch pretraining contract of plan section 5."""
    run_dir = Path(run_dir)
    hashes = artifacts(run_dir, FULL_ARTIFACTS)  # every file must exist before it is parsed
    record = load_provenance(run_dir, 'full')
    _, closure = verify_source_closure(record, 'full', repo)
    revalidate_inputs(record, repo)
    args = _read_json(run_dir / 'args.json', 'args.json')
    rows = _history_rows(run_dir / 'history.jsonl', 'history.jsonl')
    last = _load_torch(run_dir / 'last.pth', 'last.pth')
    meta = {key: last[key] for key in ('epoch', 'batch_idx') if key in last}
    check_argument_sources(record, args, last, rows, meta)
    _require('model' in last, 'last.pth records no "model" state dict')
    state = _state_dict(_load_torch(run_dir / EPOCH_CHECKPOINT, EPOCH_CHECKPOINT), EPOCH_CHECKPOINT)
    model = _state_dict(last['model'], 'last.pth["model"]')
    _require(set(state) == set(model), EPOCH_CHECKPOINT + ' has a different parameter set than last.pth')
    _require(all(torch.equal(state[key], model[key]) for key in state),
             EPOCH_CHECKPOINT + ' differs tensor-wise from last.pth["model"]')
    return dict(artifacts=hashes, epochs=len(rows),
                backbone=args['backbone'], source_closure_sha256=closure['sha256'],
                registry_sha256=record.get('registry_sha256'),
                git_head=record.get('git_state', {}).get('HEAD'),
                checkpoint={'path': EPOCH_CHECKPOINT, 'sha256': hashes[EPOCH_CHECKPOINT],
                            'epoch': exp06_recipe.EXP01_RECIPE['epochs']})


def diagnostic_evidence(run_dir, receipt, child_exit):
    """A smoke or probe proves nothing about an arm, but must produce a valid receipt.

    Validity is the receipt's own shape (diagnostic, an integer status, a ``--no-save``
    argv); ``passed`` then reports whether that diagnostic actually succeeded. A failed
    diagnostic is still recorded -- it is simply never ``admissible_arm``.
    """
    _require(receipt is not None, 'a diagnostic run needs its --receipt')
    path = Path(receipt)
    _require(path.is_file(), 'missing smoke receipt: {}'.format(receipt))
    record = _read_json(path, 'smoke receipt')
    _require(record.get('diagnostic') is True, 'the smoke receipt is not marked diagnostic')
    argv = record.get('argv')
    _require(isinstance(argv, list) and '--no-save' in argv,
             'a diagnostic must run with --no-save; its receipt records argv {!r}'.format(argv))
    status = record.get('exit_status')
    _require(type(status) is int, 'the smoke receipt records exit_status {!r}'.format(status))
    return dict(artifacts={}, passed=status == 0 and child_exit == 0,
                receipt={'path': str(path.resolve()), 'sha256': provenance.sha256_file(path),
                         'entry': record.get('entry'), 'exit_status': status,
                         'outcome': record.get('outcome')})


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


@functools.lru_cache(maxsize=None)
def expected_state_keys(backbone, num_shot):
    """The parameter names a checkpoint of this arm must carry, and nothing else."""
    with torch.random.fork_rng(devices=[]):
        return frozenset(build_xrir_exp06(backbone, num_shot).state_dict().keys())


def _finite(label, value):
    _require(type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(value),
             '{} is {!r}, not a finite number'.format(label, value))
    return value


def _resolve(path, repo):
    return Path(path) if Path(path).is_absolute() else Path(repo) / path


def _checkpoint_keys(path, label, backbone, num_shot):
    """Refuse a checkpoint that is not this arm's model, before anything hashes it."""
    state = _state_dict(_load_torch(path, label), label)
    expected = expected_state_keys(backbone, num_shot)
    _require(frozenset(state) == expected, '{}: {} parameters are not the {} of '
             'build_xrir_exp06({!r}, {})'.format(label, len(state), len(expected),
                                                 backbone, num_shot))


def _rooms_and_frame(args):
    """Every HAA child records the rooms it ran and the frame its geometry was in."""
    rooms = args.get('rooms')
    _require(isinstance(rooms, list) and rooms and all(isinstance(room, str) for room in rooms),
             'args.json must record a nonempty list of rooms')
    frame = args.get('frame')
    _require(frame in FRAMES, 'args.json must record frame room|heading, not {!r}'.format(frame))
    return rooms, frame


def _heading_binding(args, rooms, frame, repo):
    """Every room's binding names a heading JSON, and must agree with it in full."""
    if frame != 'heading':
        _require(not args.get('heading'), 'the room frame must not record a heading')
        return None
    heading = args.get('heading')
    _require(isinstance(heading, dict) and set(rooms) <= set(heading),
             'the heading frame requires a heading record for every room')
    bound = {}
    for room in rooms:
        entry = heading[room]
        _require(isinstance(entry, dict), 'heading binding for {} is not a record'.format(room))
        phi = _finite('heading binding for {}: phi_deg'.format(room), entry.get('phi_deg'))
        k = entry.get('k')
        _require(type(k) is int and 0 <= k < WIDTH,
                 'heading binding for {}: k {!r} is not a column in [0, {})'.format(room, k, WIDTH))
        _require(k == exp06_heading.heading_roll_k(phi),
                 'heading binding for {}: k {} is not the roll of {} degrees'.format(room, k, phi))
        _require(entry.get('decision') in HEADING_DECISIONS,
                 'heading binding for {} must be estimated or override, not {!r}'.format(
                     room, entry.get('decision')))
        _require(_is_sha256(entry.get('sha256')),
                 'heading binding for {} records no sha256'.format(room))
        path = entry.get('path')
        _require(isinstance(path, str) and path,
                 'heading binding for {} records no json path'.format(room))
        resolved = _resolve(path, repo)
        _require(resolved.is_file(), 'missing heading json for {}: {}'.format(room, resolved))
        _require(provenance.sha256_file(resolved) == entry['sha256'],
                 'heading json for {} does not hash to the recorded sha256'.format(room))
        try:
            record = exp06_heading.read_heading_json(str(resolved))
        except (OSError, ValueError) as error:
            raise ValueError('invalid heading json for {}: {}'.format(room, error)) from error
        _require(record['room'] == room and record['k'] == k and record['phi_deg'] == phi
                 and record['decision'] == entry['decision'],
                 'heading json for {} disagrees with the recorded binding'.format(room))
        bound[room] = dict(entry)
    return bound


def haa_child_arguments(run_dir, run_type, repo):
    """Provenance, closure and argument agreement, shared by both HAA child types."""
    record = load_provenance(run_dir, run_type)
    verify_source_closure(record, run_type, repo)
    revalidate_inputs(record, repo, required=('source_closures',))
    args = _read_json(Path(run_dir) / 'args.json', 'args.json')
    disagreements = exp06_recipe.compare_sources(
        {'args.json': args, 'provenance.effective_args': record['effective_args']})
    _require(not disagreements, 'recorded arguments disagree: ' + '; '.join(disagreements))
    backbone = args.get('backbone')
    _require(backbone in BACKBONES, 'args.json records backbone {!r}'.format(backbone))
    _require(type(args.get('num_shot')) is int and args['num_shot'] > 0,
             'args.json records no positive integer num_shot')
    return record, args


def haa_history(run_dir, args):
    """The validation cadence the arguments declare, with finite losses, and nothing else."""
    rows = _history_rows(Path(run_dir) / 'history.jsonl', 'history.jsonl')
    every, total = args.get('val_every'), args.get('epochs')
    for label, value in (('val_every', every), ('epochs', total)):
        _require(type(value) is int and value > 0,
                 'args.json records no positive integer {}'.format(label))
    expected = list(range(every, total + 1, every))
    epochs = []
    for number, row in enumerate(rows, 1):
        epoch = row.get('epoch')
        _require(type(epoch) is int, 'history.jsonl line {} records epoch {!r}'.format(number, epoch))
        _finite('history.jsonl line {} val_loss'.format(number), row.get('val_loss'))
        epochs.append(epoch)
    _require(epochs == expected, 'history.jsonl epochs {} are not the declared validation '
             'cadence {}'.format(epochs, expected))
    summary = _read_json(Path(run_dir) / 'summary.json', 'summary.json')
    missing = [key for key in HAA_SUMMARY if key not in summary]
    _require(not missing, 'summary.json is incomplete: missing ' + ', '.join(missing))
    _finite('summary.json best_val_loss', summary['best_val_loss'])
    _require(summary['best_epoch'] in epochs,
             'summary.json best_epoch {!r} is not a validated epoch'.format(summary['best_epoch']))
    return epochs, summary


def haa_train_evidence(run_dir, repo):
    """One fine-tuning child of the HAA pipeline (plan section 6.2)."""
    hashes = artifacts(run_dir, HAA_TRAIN_ARTIFACTS)
    record, args = haa_child_arguments(run_dir, 'haa_train', repo)
    rooms, frame = _rooms_and_frame(args)
    heading = _heading_binding(args, rooms, frame, repo)
    epochs, summary = haa_history(run_dir, args)
    for name in ('best.pth', 'last.pth'):
        _checkpoint_keys(Path(run_dir) / name, name, args['backbone'], args['num_shot'])
    _require(_is_sha256(args.get('init_sha256')),
             'fine-tuning must record init_sha256 of its initialisation')
    init = args.get('init')
    _require(isinstance(init, str) and init, 'args.json records no init checkpoint path')
    resolved = _resolve(init, repo)
    _require(resolved.is_file(), 'missing init checkpoint: {}'.format(resolved))
    _require(provenance.sha256_file(resolved) == args['init_sha256'],
             'init checkpoint {} does not hash to the recorded init_sha256'.format(resolved))
    return dict(artifacts=hashes, rooms=rooms, frame=frame, heading=heading,
                backbone=args['backbone'], init_sha256=args['init_sha256'],
                best_epoch=summary['best_epoch'], epochs=epochs,
                source_closure_sha256=list(record['source_closures'].values())[0]['sha256'])


def haa_eval_evidence(run_dir, repo):
    """One evaluation child: exactly one room, bound to the checkpoint it actually ran."""
    record, args = haa_child_arguments(run_dir, 'haa_eval', repo)
    rooms, frame = _rooms_and_frame(args)
    _require(len(rooms) == 1, 'an evaluation child covers exactly one room, not {}'.format(rooms))
    room, tag = rooms[0], args.get('tag', '')
    names = ('provenance.json', 'args.json', 'metrics_{}{}.json'.format(room, tag),
             'per_sample_{}{}.json'.format(room, tag))
    hashes = artifacts(run_dir, names)
    heading = _heading_binding(args, rooms, frame, repo)
    per_sample = _read_json(Path(run_dir) / names[3], names[3])
    meta = per_sample.get('meta')
    _require(isinstance(meta, dict), 'per-sample meta must be a record')
    required = ('backbone', 'checkpoint_sha256', 'frame') + (('heading',) if heading else ())
    missing = [key for key in required if key not in meta]
    _require(not missing, 'per-sample meta must record ' + ', '.join(missing))
    _require(meta['frame'] == frame and frame in FRAMES,
             'per-sample meta frame {!r} is not the {!r} of args.json'.format(meta['frame'], frame))
    _require(meta['backbone'] == args['backbone'],
             'per-sample meta backbone {!r} differs from args.json'.format(meta['backbone']))
    if heading:
        _require(exp06_recipe.strict_equal(meta['heading'], args['heading']),
                 'per-sample meta heading differs from the heading bound in args.json')
    else:
        _require(not meta.get('heading'), 'a room-frame evaluation must record no heading')
    checkpoint = args.get('checkpoint')
    _require(isinstance(checkpoint, str) and checkpoint, 'args.json records no checkpoint path')
    resolved = _resolve(checkpoint, repo)
    _require(resolved.is_file(), 'missing evaluation checkpoint: {}'.format(resolved))
    _checkpoint_keys(resolved, 'checkpoint ' + checkpoint, args['backbone'], args['num_shot'])
    digest = provenance.sha256_file(resolved)
    _require(meta['checkpoint_sha256'] == digest, 'per-sample meta checkpoint_sha256 {} is not '
             'the hash {} of {}'.format(meta['checkpoint_sha256'], digest, resolved))
    index, side = per_sample.get('index'), per_sample.get('side_label')
    _require(isinstance(index, list) and index, 'the per-sample file records no index')
    _require(isinstance(side, list) and len(side) == len(index),
             'side_label must carry one room-frame label per index ({} for {} samples)'.format(
                 len(side) if isinstance(side, list) else side, len(index)))
    bad = [value for value in side if isinstance(value, bool) or value not in (-1, 1)]
    _require(not bad, 'side_label values must be -1 or 1, not {}'.format(sorted(set(map(repr, bad)))))
    return dict(artifacts=hashes, room=room, frame=frame, heading=heading,
                backbone=args['backbone'], checkpoint_sha256=digest, samples=len(index),
                source_closure_sha256=list(record['source_closures'].values())[0]['sha256'])


def expected_children(expect):
    """The exclusive per-child directories one pipeline seed must complete."""
    _require(expect in EXPECTATIONS, 'a job needs --expect finetune|zeroshot, not {!r}'.format(expect))
    evaluations = tuple('eval/' + room for room in ROOMS)
    if expect == 'zeroshot':
        return evaluations
    return ('stage1',) + tuple('stage2_' + room for room in ROOMS) + evaluations


CHILD_COMPLETION = ('schema_version', 'run_type', 'run_dir', 'child_exit', 'child_exit_time',
                    'log', 'child_exit_receipt', 'diagnostic', 'admissible_arm', 'artifacts',
                    'backbone', 'frame', 'heading')
CHILD_EXTRA = {'haa_train': ('rooms', 'init_sha256', 'best_epoch'),
               'haa_eval': ('room', 'checkpoint_sha256', 'samples')}


def child_role(name):
    """The run type this finalizer requires at one child path of a pipeline seed."""
    if name == 'stage1' or (name.startswith('stage2_') and name[len('stage2_'):] in ROOMS):
        return 'haa_train'
    for prefix in ('eval/', 'zeroshot/eval/'):
        if name.startswith(prefix) and name[len(prefix):] in ROOMS:
            return 'haa_eval'
    raise ValueError('unexpected child path: ' + name)


def child_completion(path, name):
    """Schema, role and artefact hashes of one child, never its own admission claim."""
    completion = Path(path) / 'completion.json'
    _require(completion.is_file(), 'child {} has no completion.json'.format(name))
    record = _read_json(completion, name + '/completion.json')
    role = child_role(name)
    missing = [key for key in CHILD_COMPLETION + CHILD_EXTRA[role] if key not in record]
    _require(not missing, 'child {} completion is incomplete: missing {}'.format(
        name, ', '.join(missing)))
    _require(record['run_type'] == role,
             'child {} has run type {!r}, not the {!r} its path requires'.format(
                 name, record['run_type'], role))
    _require(record['diagnostic'] is False, 'child {} is a diagnostic run'.format(name))
    _require(type(record['child_exit']) is int and record['child_exit'] == 0,
             'child {} records child_exit {!r}'.format(name, record['child_exit']))
    hashes = record['artifacts']
    _require(isinstance(hashes, dict) and hashes, 'child {} records no artefacts'.format(name))
    for artefact, digest in sorted(hashes.items()):
        file = Path(path) / artefact
        _require(file.is_file(), 'child {} artefact {} is gone'.format(name, artefact))
        _require(provenance.sha256_file(file) == digest,
                 'child {} artefact {} no longer hashes to its recorded digest'.format(
                     name, artefact))
    return record


def job_identity(records, expect):
    """Backbone, frame, heading rolls and the init/checkpoint lineage across one seed."""
    backbones = {record['backbone'] for record in records.values()}
    _require(len(backbones) == 1, 'children disagree on the backbone: ' + repr(sorted(backbones)))
    frames = {record['frame'] for record in records.values()}
    _require(len(frames) == 1, 'children disagree on the frame: ' + repr(sorted(frames)))
    heading = {}
    for name, record in sorted(records.items()):
        for room, binding in (record['heading'] or {}).items():
            previous = heading.setdefault(room, binding)
            _require(previous.get('k') == binding.get('k'),
                     'children disagree on the heading roll of {}: {} at {}'.format(
                         room, binding.get('k'), name))
    fields = dict(backbone=sorted(backbones)[0], frame=sorted(frames)[0], heading=heading or None)
    if expect == 'zeroshot':
        digests = {record['checkpoint_sha256'] for record in records.values()}
        _require(len(digests) == 1,
                 'zero-shot lineage: the evaluations used {} different checkpoints'.format(
                     len(digests)))
        return dict(fields, checkpoint_sha256=sorted(digests)[0])
    stage1 = records['stage1']['artifacts']['best.pth']
    for name, record in sorted(records.items()):
        if name.startswith('stage2_'):
            _require(record['init_sha256'] == stage1,
                     'lineage: {} did not start from stage1/best.pth'.format(name))
        elif name.startswith('eval/'):
            room = name[len('eval/'):]
            _require(record['checkpoint_sha256'] == records['stage2_' + room]['artifacts']['best.pth'],
                     'lineage: {} did not evaluate stage2_{}/best.pth'.format(name, room))
    return dict(fields, init_sha256=records['stage1']['init_sha256'])


def haa_job_evidence(run_dir, children, expect):
    """Bind one seed's children: every expected directory, each with its own completion."""
    expected, job = set(expected_children(expect)), Path(run_dir).resolve()
    seen, records = {}, {}
    for child in children:
        path = Path(child).resolve()
        try:
            name = path.relative_to(job).as_posix()
        except ValueError as error:
            raise ValueError('child {} lies outside the job directory'.format(child)) from error
        if expect == 'zeroshot' and name.startswith('zeroshot/'):
            name = name[len('zeroshot/'):]  # <init>/zeroshot/eval/<room> and <init>/zeroshot alike
        records[name] = child_completion(path, name)
        seen[name] = provenance.sha256_file(path / 'completion.json')
    missing = sorted(expected - set(seen))
    _require(not missing, 'job is missing children: ' + ', '.join(missing))
    unexpected = sorted(set(seen) - expected)
    _require(not unexpected, 'unexpected children: ' + ', '.join(unexpected))
    return dict(job_identity(records, expect), artifacts={}, children=seen, expect=expect)


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
    refuse_live_launch(run_dir)
    log_record, child_exit_time, log_digest = closed_log(log, child_exit)
    receipt_record = child_exit_receipt(run_dir, child_exit, log_digest)
    diagnostic = run_type in DIAGNOSTIC
    if not diagnostic:
        _require(child_exit == 0, 'child exited with status {}'.format(child_exit))
    fields = dict(schema_version=1, run_type=run_type, run_dir=str(run_dir.resolve()),
                  repo=str(Path(repo).resolve()), child_exit=child_exit,
                  child_exit_time=child_exit_time, log=log_record,
                  child_exit_receipt=receipt_record,
                  diagnostic=diagnostic, admissible_arm=not diagnostic)
    fields.update(diagnostic_evidence(run_dir, receipt, child_exit) if diagnostic
                  else full_evidence(run_dir, repo) if run_type == 'full'
                  else haa_train_evidence(run_dir, repo) if run_type == 'haa_train'
                  else haa_eval_evidence(run_dir, repo) if run_type == 'haa_eval'
                  else haa_job_evidence(run_dir, children, expect))
    _require(provenance.sha256_file(log) == log_digest,
             'stale log: {} changed while its completion was being validated'.format(log))
    return write_completion(run_dir / 'completion.json', fields)


def child_exit_main(argv):
    """Close one child's log: append the end marker, then bind those bytes in a receipt."""
    parser = argparse.ArgumentParser(description='Close an exp_06 child log.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--child-pid', type=int, required=True)
    parser.add_argument('--status', type=int, required=True)
    args = parser.parse_args(argv)
    path = Path(args.run_dir) / 'child_exit.json'
    try:
        _require(not path.exists(), 'child_exit.json already exists at {}'.format(path))
        stamp = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
        with open(args.log, 'a') as stream:
            stream.write('{} {} {}\n'.format(MARKER, args.status, stamp))
            stream.flush()
            os.fsync(stream.fileno())
        provenance.write_manifest(path, dict(
            schema_version=1, child_pid=args.child_pid, status=args.status, ended_at=stamp,
            log=str(Path(args.log).resolve()),
            log_sha256_after_marker=provenance.sha256_file(args.log)))
    except (OSError, ValueError) as error:
        print('EXP06_CHILD_EXIT_REFUSED ' + str(error), file=sys.stderr, flush=True)
        return 2
    print('EXP06_CHILD_EXIT_OK ' + str(path), flush=True)
    return 0


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
    if argv[:1] == ['child-exit']:
        return child_exit_main(argv[1:])
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
