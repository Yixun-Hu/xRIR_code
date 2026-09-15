"""Completion evidence for exp_06 runs, dispatched by run type (plan v4 sections 5, 6.4).

The entry points write ``provenance.json``; this external finalizer decides completion
after the child has exited and writes ``completion.json``. Every non-job run type shares
the same execution evidence: no live ``launch.pid``, a log whose last line is the
launcher's ``EXP06_CHILD_EXIT`` marker, a ``child_exit.json`` receipt binding the hash of
exactly those bytes, and a re-hash of the log after all validation ("stale log"). For the
run types that record provenance, the closure of the entry module named in the record is
recomputed and required to agree three ways -- working tree now, bytes at spawn, reviewed
blob at the recorded commit -- and ``tools.provenance.revalidate`` rehashes the declared
data inventory and mutable inputs.

``full``      twelve-epoch pretraining: complete provenance, closure membership and the
              three-way hashes, input revalidation, the recipe/production/derived schema
              on all three recorded copies of the arguments (``args.json``,
              ``last.pth['args']``, ``provenance.effective_args``) compared type-strictly,
              budget completeness, and ``epoch_012.pth`` equal to ``last.pth['model']``.
``smoke``/``probe``  no artifacts, but a valid diagnostic receipt (``diagnostic: true``, an
              integer ``exit_status``, ``--no-save`` in its argv); ``passed`` reports
              whether the diagnostic succeeded and it is never admissible as an arm.
``haa_train`` one fine-tuning child: provenance and closure, arguments agreeing with the
              record, a heading binding per room (``phi_deg``, ``k`` = roll(phi), decision,
              sha256 and path, verified against the heading JSON), ``init_sha256`` equal to
              the hash of the recorded init, the declared validation cadence with finite
              losses, a complete ``summary.json`` and checkpoints carrying this arm's
              parameter names.
``haa_eval``  one evaluation child: provenance and closure, one room, ``meta`` whose frame,
              backbone and heading equal the arguments and whose ``checkpoint_sha256`` is
              the hash of the checkpoint the arguments name, and a ``side_label`` array of
              -1/1 with one entry per index.
``haa_job``   one seed of the pipeline: every expected child directory, each completion
              schema-validated in the role its path requires, its recorded artefact hashes
              re-verified, one backbone/frame/heading roll across children, and the
              stage1 -> stage2 -> eval checkpoint lineage. ``admissible_arm`` is derived
              here, never read from a child.

    python tools/exp06_finalize.py --run-dir <dir> --run-type full \
        --log <log> --child-exit 0 [--repo <path>] [--receipt <json>] \
        [--children <dir>...] [--expect finetune|zeroshot]

The command exits 0 after writing ``completion.json`` and 2 on any refusal. Two
subcommands serve the launcher. ``preflight`` is its gate before it starts a child: HEAD at
the reviewed commit, a tree clean outside ``worklog/``, no live exp_06 launch, and (for
``full``/``probe``) a GPU with no compute apps. ``child-exit`` closes a child's log: it
appends the end marker and exclusively writes the receipt that binds those bytes.

    python tools/exp06_finalize.py preflight --mode full --gpu 1 \
        --reviewed-commit <sha> [--attempt-root <dir>] [--repo <path>]
    python tools/exp06_finalize.py child-exit --run-dir <dir> --log <log> \
        --child-pid <pid> --status <n>

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
import tempfile

import torch

from model.xRIR_cyl_oriented import BACKBONES_EXP06, build_xrir_exp06
from sim_to_real.haa_dataset import NO_T60_ROOMS, ROOMS
from tools import exp06_heading, exp06_recipe, provenance

REPO = Path(__file__).resolve().parents[1]
MARKER = 'EXP06_CHILD_EXIT'
RECEIPT_FIELDS = ('child_pid', 'status', 'started_at', 'ended_at', 'log_sha256_after_marker')
DIAGNOSTIC = ('smoke', 'probe')
RUN_TYPES = ('full', 'smoke', 'probe', 'haa_train', 'haa_eval', 'haa_job')
EXPECTATIONS = ('finetune', 'zeroshot')
LAUNCH_MODES = ('smoke', 'probe', 'full', 'finalize')
EXCLUSIVE_GPU_MODES = ('probe', 'full')
FULL_ARTIFACTS = ('provenance.json', 'args.json', 'history.jsonl', 'last.pth', 'epoch_012.pth')
HAA_TRAIN_ARTIFACTS = ('provenance.json', 'args.json', 'history.jsonl', 'summary.json',
                       'best.pth', 'last.pth')
HAA_SUMMARY = ('best_val_loss', 'best_epoch', 'init_val_loss', 'epochs')
METRIC_SOURCE = {'edt_error_s': 'edt', 'c50_error_db': 'c50', 't60_error_pct': 't60',
                 'env_error': 'env', 'stft_log_mse': 'stft_mse', 'test_loss': 'loss'}
METRICS_REQUIRED = (('backbone', 'checkpoint', 'room', 'split', 'num_shot', 'eval_seed',
                     'n_samples', 'c50_outliers', 't60_invalid', 'edt_invalid')
                    + tuple(sorted(METRIC_SOURCE)))
HEADING_DECISIONS = ('estimated', 'override')
FRAMES = ('room', 'heading')
GIT_STATE = ('HEAD', 'dirty', 'untracked', 'dirty_outside_worklog', 'diff_sha256')
REQUIRED_PROVENANCE = ('run_type', 'repo', 'reviewed_commit', 'source_closures',
                       'registry_sha256', 'git_state', 'environment', 'command',
                       'effective_args')
ENTRY_MODULES = {'full': 'tools.exp06_train', 'haa_train': 'tools.exp06_haa_finetune',
                 'haa_eval': 'tools.exp06_haa_eval'}
BACKBONES = tuple(sorted(BACKBONES_EXP06))
IDENTITY_KEYS = ('train_data_identity', 'data_identity')
IDENTITY_FIELDS = ('data_root', 'inventory', 'inventory_files', 'inventory_bytes',
                   'inventory_sha256')
TRAIN_INVENTORY = REPO / 'ckpt/yaw_aug/train_inventory.json'  # exp_04's cache, read-only
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


def _mapping(value, label):
    """Every nested container is typed before anything traverses or sorts it."""
    _require(isinstance(value, dict), '{} is not a JSON object'.format(label))
    bad = sorted(repr(key) for key in value if not isinstance(key, str))
    _require(not bad, '{} has non-string keys: {}'.format(label, ', '.join(bad[:4])))
    return value


def _state_dict(mapping, label):
    """Require a parameter mapping of names to tensors before anything compares it."""
    _require(isinstance(mapping, dict) and mapping, '{} is not a parameter mapping'.format(label))
    bad = sorted(repr(key) for key, value in mapping.items()
                 if not isinstance(key, str) or not torch.is_tensor(value))
    _require(not bad, '{} has non-tensor entries: {}'.format(label, ', '.join(bad[:4])))
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


def inventory_digest(records):
    """The digest tools.provenance records over [[path, sha256], ...] of an inventory."""
    return hashlib.sha256(json.dumps([[r['path'], r['sha256']] for r in records],
                                     sort_keys=True).encode()).hexdigest()


def check_identity_schema(identity, key):
    """The shape ``tools.provenance.train_data_identity`` writes, with consistent totals."""
    _require(isinstance(identity, dict), '{} is not a record'.format(key))
    missing = [field for field in IDENTITY_FIELDS if field not in identity]
    _require(not missing, '{} is incomplete: missing {}'.format(key, ', '.join(missing)))
    _require(isinstance(identity['data_root'], str) and identity['data_root'],
             '{} records no data_root'.format(key))
    entries = identity['inventory']
    _require(isinstance(entries, list) and entries,
             '{}: an empty inventory identifies no training data'.format(key))
    for entry in entries:
        _require(isinstance(entry, dict) and isinstance(entry.get('path'), str)
                 and _is_sha256(entry.get('sha256')) and type(entry.get('size')) is int
                 and type(entry.get('mtime_ns')) is int,
                 '{}: incomplete inventory entry {!r}'.format(key, entry))
    _require(type(identity['inventory_files']) is int and identity['inventory_files'] == len(entries),
             '{}: inventory_files {!r} is not the {} recorded entries'.format(
                 key, identity['inventory_files'], len(entries)))
    _require(type(identity['inventory_bytes']) is int
             and identity['inventory_bytes'] == sum(entry['size'] for entry in entries),
             '{}: inventory_bytes {!r} is not the size of the recorded entries'.format(
                 key, identity['inventory_bytes']))
    _require(identity['inventory_sha256'] == inventory_digest(entries),
             '{}: inventory_sha256 is not the digest of the recorded entries'.format(key))
    return entries


def train_inventory_paths(data_root, cache_path=None):
    """The membership the pinned helper derives for one root; never read from the record.

    exp_04's stat-validated cache is reused only when it already describes this root;
    otherwise the inventory is recomputed into a temporary file, so a finalization never
    writes under ``ckpt/``.
    """
    cache = Path(TRAIN_INVENTORY if cache_path is None else cache_path)
    reuse = False
    if cache.is_file():
        try:
            reuse = json.loads(cache.read_text()).get('data_root') == str(Path(data_root).resolve())
        except (OSError, ValueError, AttributeError):
            reuse = False
    try:
        if reuse:
            identity = provenance.train_data_identity(str(data_root), cache_path=str(cache))
        else:
            with tempfile.TemporaryDirectory() as scratch:
                identity = provenance.train_data_identity(
                    str(data_root), cache_path=str(Path(scratch) / 'train_inventory.json'))
        return {entry['path'] for entry in identity['inventory']}
    except Exception as error:  # the dataset, the cache and the filesystem all refuse alike
        raise ValueError('cannot derive the training inventory of {}: {}: {}'.format(
            data_root, type(error).__name__, error)) from error


def verify_train_identity(record, key='train_data_identity'):
    """Blocker 1: the run's own root, and an inventory covering the whole training split."""
    entries = check_identity_schema(record.get(key), key)
    resolved = str(Path(record[key]['data_root']).resolve())
    declared = record.get('data_root')
    _require(isinstance(declared, str) and declared,
             'provenance.json records no resolved data_root for the run')
    _require(str(Path(declared).resolve()) == resolved,
             '{}: data_root {} is not the {} the run resolved'.format(
                 key, record[key]['data_root'], declared))
    absent = sorted(train_inventory_paths(resolved) - {entry['path'] for entry in entries})
    _require(not absent, '{}: the inventory does not cover the training split ({} files '
             'missing, e.g. {})'.format(key, len(absent), absent[:2]))
    return entries


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


def _timestamp(value, label):
    """One ISO-8601 instant with an offset; anything else is a named refusal."""
    try:
        stamp = datetime.datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError('child_exit.json {} is not an ISO-8601 timestamp: {!r}'.format(
            label, value)) from error
    _require(stamp.tzinfo is not None,
             'child_exit.json {} needs a timezone-aware timestamp: {!r}'.format(label, value))
    return stamp


def child_exit_receipt(run_dir, child_exit, log_digest, marker_time):
    """The launcher's proof that every writer had exited when it hashed the log."""
    path = Path(run_dir) / 'child_exit.json'
    _require(path.is_file(), 'missing child_exit.json in {}'.format(run_dir))
    receipt = _read_json(path, 'child_exit.json')
    missing = [key for key in RECEIPT_FIELDS if key not in receipt]
    _require(not missing, 'child_exit.json is incomplete: missing ' + ', '.join(missing))
    _require(type(receipt['status']) is int and receipt['status'] == child_exit,
             'child_exit.json records status {!r}, not the reported {}'.format(
                 receipt['status'], child_exit))
    _require(type(receipt['child_pid']) is int and receipt['child_pid'] > 0,
             'child_exit.json records child_pid {!r}, not a pid'.format(receipt['child_pid']))
    started = _timestamp(receipt['started_at'], 'started_at')
    ended = _timestamp(receipt['ended_at'], 'ended_at')
    _require(ended >= started, 'child_exit.json ended_at {} precedes the started_at {} '
             'recorded at spawn'.format(receipt['ended_at'], receipt['started_at']))
    _require(receipt['ended_at'] == marker_time,
             'child_exit.json ended_at {} is not the {} of the log marker'.format(
                 receipt['ended_at'], marker_time))
    _require(_is_sha256(receipt['log_sha256_after_marker']),
             'child_exit.json log_sha256_after_marker {!r} is not a sha256'.format(
                 receipt['log_sha256_after_marker']))
    _require(receipt['log_sha256_after_marker'] == log_digest,
             'child_exit.json binds a different log: {} is not the validated {}'.format(
                 receipt['log_sha256_after_marker'], log_digest))
    return {'path': str(path.resolve()), 'sha256': provenance.sha256_file(path),
            'child_pid': receipt['child_pid'], 'started_at': receipt['started_at'],
            'ended_at': receipt['ended_at']}


def _pid_of(path):
    """One pid file, or a named refusal; a stale file names a process that is gone."""
    try:
        pid = int(Path(path).read_text().split()[0])
    except (IndexError, ValueError) as error:
        raise ValueError('unreadable {}: {}'.format(path, error)) from error
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return pid, False
    except PermissionError:
        pass  # Somebody else's live process is still a live process.
    return pid, True


def refuse_live_launch(run_dir, owner_pid=None):
    """No mode may certify a run some other launcher or child is still writing.

    Should-fix 5: the launcher that spawned the child stays alive through draining and
    finalisation, so its own ``launch.pid`` is admissible when it identifies itself with
    ``--owner-pid``. A live ``child.pid`` is never admissible.
    """
    owner = None
    for name in ('launch.pid', 'child.pid'):
        path = Path(run_dir) / name
        if not path.is_file():
            continue
        pid, alive = _pid_of(path)
        if name == 'launch.pid':
            owner = pid
        if alive and not (name == 'launch.pid' and pid == owner_pid):
            raise ValueError('the {} {} of {} is still alive'.format(name, pid, run_dir))
    return owner


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
    _mapping(record['environment'], 'provenance.json environment')
    state = _mapping(record['git_state'], 'provenance.json git_state')
    missing = [key for key in GIT_STATE if key not in state]
    _require(not missing, 'provenance.json git_state is incomplete: missing '
             + ', '.join(missing))
    for name, entry in sorted(_mapping(record.get('mutable_inputs', {}),
                                       'provenance.json mutable_inputs').items()):
        _require(isinstance(entry, dict) and isinstance(entry.get('path'), str),
                 'provenance.json mutable_inputs[{}] records no path'.format(name))
    for key in IDENTITY_KEYS:
        if key in record:
            _mapping(record[key], 'provenance.json ' + key)
    present = [key for key in IDENTITY_KEYS if isinstance(record.get(key), dict)]
    _require(present, 'provenance.json records no train_data_identity/data_identity')
    for key in present:
        check_identity_schema(record[key], key)
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


def registry_sha256():
    """The digest of the backbone registry as it stands at finalisation."""
    mapping = {name: cls.__module__ + '.' + cls.__qualname__
               for name, cls in BACKBONES_EXP06.items()}
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()


def check_exp06_bindings(record, args, closure, run_dir, run_type, repo):
    """Blocker 2: every exp06_* field must equal the execution record it claims to bind.

    ``closure['sha256']`` has already been recomputed from the reviewed blobs by
    ``verify_source_closure``, and the registry digest is recomputed here, so agreement
    between the three recorded copies of the arguments can never establish these values.
    """
    registry = registry_sha256()
    _require(record.get('registry_sha256') == registry,
             'provenance registry_sha256 {!r} is not the {} of BACKBONES_EXP06 at '
             'finalisation'.format(record.get('registry_sha256'), registry))
    expected = {'exp06_run_type': run_type, 'exp06_git_head': record['git_state'].get('HEAD'),
                'exp06_registry_sha256': registry,
                'exp06_source_closure_sha256': closure['sha256']}
    for field in sorted(expected):
        _require(args.get(field) == expected[field], 'args {} is {!r}, not the {!r} of the '
                 'execution record'.format(field, args.get(field), expected[field]))
    path = args.get('exp06_provenance_path')
    _require(isinstance(path, str) and path, 'args exp06_provenance_path is {!r}'.format(path))
    _require(_resolve(path, repo).resolve() == (Path(run_dir) / 'provenance.json').resolve(),
             'args exp06_provenance_path {} is not the validated {}'.format(
                 path, Path(run_dir) / 'provenance.json'))


def full_evidence(run_dir, repo):
    """Verify the twelve-epoch pretraining contract of plan section 5."""
    run_dir = Path(run_dir)
    hashes = artifacts(run_dir, FULL_ARTIFACTS)  # every file must exist before it is parsed
    record = load_provenance(run_dir, 'full')
    _, closure = verify_source_closure(record, 'full', repo)
    verify_train_identity(record)
    revalidate_inputs(record, repo)
    args = _read_json(run_dir / 'args.json', 'args.json')
    rows = _history_rows(run_dir / 'history.jsonl', 'history.jsonl')
    last = _load_torch(run_dir / 'last.pth', 'last.pth')
    meta = {key: last[key] for key in ('epoch', 'batch_idx') if key in last}
    check_argument_sources(record, args, last, rows, meta)
    check_exp06_bindings(record, args, closure, run_dir, 'full', repo)
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
    """Exactly the rows ``sim_to_real/finetune_haa.py`` writes (blocker 3a).

    Epoch 0 records the initialisation's validation loss and no train loss; every epoch
    ``1..epochs`` records a finite train loss and lr, and a validation loss precisely on
    ``epoch % val_every == 0`` and on the final epoch. ``is_best`` may mark only a
    validated row, and ``summary.json`` must agree with the history it summarises --
    including a ``best_epoch`` of 0 when fine-tuning never improved on the init.
    """
    rows = _history_rows(Path(run_dir) / 'history.jsonl', 'history.jsonl')
    every, total = args.get('val_every'), args.get('epochs')
    for label, value in (('val_every', every), ('epochs', total)):
        _require(type(value) is int and value > 0,
                 'args.json records no positive integer {}'.format(label))
    _require(len(rows) == total + 1, 'history.jsonl records {} rows, not the epochs 0..{} the '
             'pipeline writes'.format(len(rows), total))
    _require(rows and rows[0].get('epoch') == 0,
             'history.jsonl does not start with the epoch 0 initialisation row')
    _require(rows[0].get('train_loss') is None,
             'history.jsonl epoch 0 records train_loss {!r}; the pipeline writes null'.format(
                 rows[0].get('train_loss')))
    validated, losses = [], {}
    for epoch, row in enumerate(rows):
        label = 'history.jsonl epoch {}'.format(epoch)
        _require(row.get('epoch') == epoch and type(row.get('epoch')) is int,
                 '{} is out of order: the row records epoch {!r}'.format(label, row.get('epoch')))
        _finite(label + ' lr', row.get('lr'))
        if epoch:
            _finite(label + ' train_loss', row.get('train_loss'))
        if epoch == 0 or epoch % every == 0 or epoch == total:
            losses[epoch] = _finite(label + ' val_loss', row.get('val_loss'))
            validated.append(epoch)
        else:
            _require('val_loss' not in row, '{} records a val_loss outside the every-{} '
                     'validation cadence'.format(label, every))
            _require('is_best' not in row, '{} records is_best without a validation'.format(label))
        _require(row.get('is_best', True) is True, label + ' records is_best false')
    summary = _read_json(Path(run_dir) / 'summary.json', 'summary.json')
    missing = [key for key in HAA_SUMMARY if key not in summary]
    _require(not missing, 'summary.json is incomplete: missing ' + ', '.join(missing))
    best = min(losses.values())
    _require(summary['best_val_loss'] == best, 'summary.json best_val_loss {!r} is not the {!r} '
             'of the history'.format(summary['best_val_loss'], best))
    _require(summary['best_epoch'] in validated and losses[summary['best_epoch']] == best,
             'summary.json best_epoch {!r} did not achieve the best validation loss'.format(
                 summary['best_epoch']))
    _require(summary['init_val_loss'] == losses[0], 'summary.json init_val_loss {!r} is not the '
             '{!r} of epoch 0'.format(summary['init_val_loss'], losses[0]))
    _require(summary['epochs'] == total,
             'summary.json epochs {!r} is not the {} of args.json'.format(summary['epochs'], total))
    return validated, summary


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


def haa_metrics(run_dir, name, room, args, per_sample):
    """Blocker 3b: the summary is parsed and cross-checked, never merely hashed."""
    metrics = _read_json(Path(run_dir) / name, name)
    missing = [key for key in METRICS_REQUIRED if key not in metrics]
    _require(not missing, '{} is incomplete: missing {}'.format(name, ', '.join(missing)))
    _require(metrics['room'] == room,
             '{} records room {!r}, not the {!r} it evaluated'.format(name, metrics['room'], room))
    for field in ('backbone', 'checkpoint', 'num_shot', 'eval_seed', 'split'):
        _require(exp06_recipe.strict_equal(metrics[field], args.get(field)),
                 '{} records {} {!r}, not the {!r} of args.json'.format(
                     name, field, metrics[field], args.get(field)))
    index = per_sample['index']
    _require(metrics['n_samples'] == len(index), '{} records n_samples {!r}, not the {} per-sample '
             'entries'.format(name, metrics['n_samples'], len(index)))
    for key, field in sorted(METRIC_SOURCE.items()):
        summary = metrics[key]
        if key == 't60_error_pct' and room in NO_T60_ROOMS:
            _require(summary is None, '{}: {} is recorded for a room the paper omits'.format(
                name, key))
            continue
        _require(isinstance(summary, dict) and set(('mean', 'median', 'n')) <= set(summary),
                 '{}: {} is not a mean/median/n summary'.format(name, key))
        values = per_sample.get(field)
        _require(isinstance(values, list) and len(values) == len(index),
                 '{}: per-sample {} needs one value per index'.format(name, field))
        finite = [value for value in values if type(value) in (int, float)
                  and not isinstance(value, bool) and math.isfinite(value)]
        _require(summary['n'] == len(finite), '{}: {} counts {!r} of {} finite per-sample '
                 'values'.format(name, key, summary['n'], len(finite)))
        mean = sum(finite) / len(finite) if finite else None
        _require(mean is None and summary['mean'] is None or mean is not None
                 and type(summary['mean']) in (int, float)
                 and math.isclose(summary['mean'], mean, rel_tol=1e-9, abs_tol=1e-9),
                 '{}: {} mean {!r} is not the {!r} of the per-sample values'.format(
                     name, key, summary['mean'], mean))
    return metrics


def haa_eval_evidence(run_dir, repo):
    """One evaluation child: exactly one room, bound to the checkpoint it actually ran."""
    record, args = haa_child_arguments(run_dir, 'haa_eval', repo)
    rooms, frame = _rooms_and_frame(args)
    _require(len(rooms) == 1, 'an evaluation child covers exactly one room, not {}'.format(rooms))
    room, tag = rooms[0], args.get('tag', '')
    _require(Path(run_dir).resolve().name == room, 'an evaluation child of {} may not claim the '
             'room {!r}'.format(Path(run_dir).resolve().name, room))
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
    haa_metrics(run_dir, names[2], room, args, per_sample)
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
CHILD_ARTIFACTS = {'haa_train': ('args.json', 'best.pth', 'last.pth'),
                   'haa_eval': ('args.json',)}


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
    absent = [artefact for artefact in CHILD_ARTIFACTS[role] if artefact not in hashes]
    _require(not absent, 'child {} records no artefact {}'.format(name, ', '.join(absent)))
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
             children=(), expect=None, owner_pid=None):
    """Verify one child's evidence for its run type and write completion.json."""
    run_dir = Path(run_dir)
    _require(run_type in RUN_TYPES, 'unknown run type: {!r}'.format(run_type))
    _require(run_dir.is_dir(), 'run directory does not exist: {}'.format(run_dir))
    _require(type(child_exit) is int, 'child status must be an integer')
    refuse_live_launch(run_dir, owner_pid)
    log_record, child_exit_time, log_digest = closed_log(log, child_exit)
    receipt_record = child_exit_receipt(run_dir, child_exit, log_digest, child_exit_time)
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
    parser.add_argument('--started-at', required=True, help='ISO-8601 instant of the spawn')
    args = parser.parse_args(argv)
    path = Path(args.run_dir) / 'child_exit.json'
    try:
        _require(not path.exists(), 'child_exit.json already exists at {}'.format(path))
        _require(args.child_pid > 0, 'child_pid {} is not a pid'.format(args.child_pid))
        started = _timestamp(args.started_at, 'started_at')
        stamp = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
        _require(datetime.datetime.fromisoformat(stamp) >= started,
                 'the clock moved backwards: {} precedes started_at {}'.format(stamp, args.started_at))
        with open(args.log, 'a') as stream:
            stream.write('{} {} {}\n'.format(MARKER, args.status, stamp))
            stream.flush()
            os.fsync(stream.fileno())
        provenance.write_manifest(path, dict(
            schema_version=1, child_pid=args.child_pid, status=args.status,
            started_at=args.started_at, ended_at=stamp,
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


def launch_roots(attempt_root):
    """Every location this launcher writes pid files into, attempts and smokes alike."""
    if attempt_root is None:
        roots = []
    elif isinstance(attempt_root, (str, Path)):
        roots = [attempt_root]
    else:
        roots = list(attempt_root)
    return [Path(root) for root in roots if Path(root).is_dir()]


def live_launches(attempt_root):
    """Refuse a second launch while any pid file under these roots names a live process."""
    live = []
    for root in launch_roots(attempt_root):
        for pid_file in sorted(list(root.glob('*/launch.pid')) + list(root.glob('*/child.pid'))):
            pid, alive = _pid_of(pid_file)
            if alive:
                live.append({'path': str(pid_file), 'pid': pid})
    return live


def preflight(mode, gpu, reviewed_commit, attempt_root=None, repo=REPO):
    """Gate a launch: reviewed commit, clean tree, no live launch, and a free card."""
    _require(mode in LAUNCH_MODES, 'unknown launch mode: {!r}'.format(mode))
    state = provenance.checked_git_state(repo, confirmatory=True)
    _require(state['HEAD'] == reviewed_commit,
             'HEAD {} is not the reviewed commit {!r} (full 40-hex sha required)'.format(
                 state['HEAD'], reviewed_commit))
    running = live_launches(attempt_root)
    _require(not running, 'another exp_06 launch is alive: {}'.format(running))
    apps = gpu_compute_apps(gpu) if mode in EXCLUSIVE_GPU_MODES else None
    _require(not apps, 'GPU {} is busy with compute apps {}'.format(gpu, apps))
    return dict(mode=mode, gpu=gpu, reviewed_commit=reviewed_commit, git_state=state,
                attempt_root=[str(root) for root in launch_roots(attempt_root)],
                gpu_compute_apps=apps, live_launches=running)


def preflight_main(argv):
    """Exit 0 with the record on stdout, 2 with the named cause on stderr."""
    parser = argparse.ArgumentParser(description='Gate one exp_06 launch.')
    parser.add_argument('--mode', choices=LAUNCH_MODES, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--reviewed-commit', required=True)
    parser.add_argument('--attempt-root', action='append', default=[],
                        help='repeatable: every root whose */launch.pid must be dead')
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
    parser.add_argument('--owner-pid', type=int,
                        help='the live launcher that owns this run dir (its launch.pid)')
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
                          receipt=args.receipt, children=args.children, expect=args.expect,
                          owner_pid=args.owner_pid)
    except (OSError, ValueError) as error:
        print('EXP06_FINALIZE_REFUSED ' + str(error), file=sys.stderr, flush=True)
        return 2
    print('EXP06_FINALIZE_OK ' + json.dumps({key: fields[key] for key in
        ('run_type', 'run_dir', 'child_exit', 'admissible_arm')}, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
