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
              three-way hashes, input revalidation, a training-data identity whose root is
              the one the run resolved and whose inventory covers the membership the pinned
              ``train_data_identity`` derives for that root, an exp_06-owned geometry
              inventory (every metadata JSON and receiver depth map of both splits, with
              membership derived again here and every file rehashed), an exp_06-owned
              inventory of the held-out waveforms the epoch test losses were measured on,
              the recipe/production/derived
              schema on all three recorded copies of the arguments (``args.json``,
              ``last.pth['args']``, ``provenance.effective_args``) compared type-strictly
              and each ``exp06_*`` field bound to the execution record (run type, HEAD,
              recomputed registry and closure digests, the validated provenance file),
              budget completeness, and ``epoch_012.pth`` equal to ``last.pth['model']``.
``smoke``/``probe``  no artifacts, but a ``provenance.json`` bound like the full case minus
              the training artefacts, and a complete diagnostic receipt: the runner's
              identity and closure digest, the entry, the argv (``--no-save``), the
              wall-clock window, the peak allocation, both budgets, an outcome in
              {ok, failed, aborted_alarm, aborted_memory} and the exploratory flag -- all
              consistent with each other, with the startup command and with the child
              window the launcher recorded. ``passed`` reports whether the diagnostic
              succeeded; never an arm.
``haa_train`` one fine-tuning child: provenance and closure, arguments agreeing with the
              record, a heading binding per room (``phi_deg``, ``k`` = roll(phi), decision,
              sha256 and path, verified against the heading JSON), ``init_sha256`` equal to
              the hash of the recorded init, exactly the rows ``finetune_haa.py`` writes
              (epoch 0 without a train loss, then every epoch, validated on the cadence and
              on the final epoch) with a ``summary.json`` that agrees with them, and
              checkpoints carrying this arm's parameter names.
``haa_eval``  one evaluation child: provenance and closure, one room equal to its own
              directory, ``meta`` whose frame, backbone and heading equal the arguments and
              whose ``checkpoint_sha256`` is the hash of the checkpoint the arguments name,
              per-sample observations bound to that room and protocol (every ``ir_path`` the
              ``<room>/<index>`` the writer records, and ``meta``'s split, num_shot and
              eval_seed the arguments'), a ``side_label`` array of -1/1 with one entry per
              index, and a parsed ``metrics_<room>.json`` whose counts, n, means and
              medians are recomputed from the numeric per-sample values -- every
              invalid-measurement counter reconciled with the observations it describes (the
              writer counts every non-finite C50, only the EDT/T60 measurements that raised,
              and none at all in a room whose T60 the paper omits, where every observation is
              its NaN).
``haa_job``   one seed of the pipeline: every expected child directory, each re-validated by
              running its own role validator again, its completion bound to that result,
              its log and receipt re-validated and rehashed, and every child checked against
              the pipeline's ``--job-spec`` (backbone, frame, seed, rooms, heading rolls)
              and the stage1 -> stage2 -> eval checkpoint lineage. ``admissible_arm`` is
              derived here, never read from a child. A job has no child process of its own
              (plan amendment A3): the shell that orchestrates it is the one that finalizes
              it, so the job closes no log, a ``child_exit.json`` at a job root is refused as
              ambiguous evidence, and the queue log -- when the pipeline passes one -- is
              recorded by path and hash as information. What binds a job instead is the job
              spec, the owner the job root's own ``launch.pid`` records -- required of every
              job, and which must be the ``--owner-pid`` declared whenever the launcher
              declares one -- and the children's re-validated completions.

    python tools/exp06_finalize.py --run-dir <dir> --run-type full \
        --log <log> --child-exit 0 [--repo <path>] [--receipt <json>] \
        [--children <dir>...] [--expect finetune|zeroshot] [--job-spec <json>] \
        [--owner-pid <pid>]

The command exits 0 after writing ``completion.json`` and 2 on any refusal. Two
subcommands serve the launcher. ``preflight`` is its gate before it starts a child: HEAD at
the reviewed commit, a tree clean outside ``worklog/``, no live launch under any root it
is given (``--attempt-root`` is repeatable: attempts and smokes alike), and (for
``full``/``probe``) a GPU with no compute apps. ``child-exit`` closes a child's log: it
appends the end marker and exclusively writes the receipt that binds those bytes. The
launcher owns ``launch.pid`` and names itself with ``--owner-pid``; the child gets
``child.pid``, and a live child is never admissible -- named by that sidecar or by the
receipt's own ``child_pid``, which must agree with it. That owner exception covers only the
directory being finalized, so a job refuses while any pid file under any of its children
is alive: a pipeline's own ``launch.pid`` belongs at the job root, never in a child.

    python tools/exp06_finalize.py preflight --mode full --gpu 1 \
        --reviewed-commit <sha> [--attempt-root <dir>]... [--repo <path>] \
        [--approved <approved_digests.json>] [--exploratory] [--min-free-gb <gib>]
    python tools/exp06_finalize.py child-exit --run-dir <dir> --log <log> \
        --child-pid <pid> --status <n> --started-at <iso>

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
import statistics
import subprocess
import sys
import tempfile
import time

import torch

from model.xRIR_cyl_oriented import BACKBONES_EXP06, build_xrir_exp06
from sim_to_real.haa_dataset import NO_T60_ROOMS, ROOMS
from tools import (exp06_heading, exp06_profiles, exp06_recipe, exp06_smoke, exp06_train,
                   provenance)

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
METRIC_COUNTS = ('c50_outliers', 't60_invalid', 'edt_invalid')
# sim_to_real/eval_haa.py:76-98 counts every non-finite C50 it reads, but counts EDT and T60
# only when the measurement raised -- a subset of their non-finite observations.
METRIC_INVALID = {'c50': ('c50_outliers', True), 'edt': ('edt_invalid', False),
                  't60': ('t60_invalid', False)}
SUMMARY_FIELDS = ('mean', 'median', 'n')  # exactly what eval_xRIR_backbone.summarize writes
METRICS_REQUIRED = (('backbone', 'checkpoint', 'room', 'split', 'num_shot', 'eval_seed',
                     'n_samples') + METRIC_COUNTS + tuple(sorted(METRIC_SOURCE)))
HEADING_DECISIONS = ('estimated', 'override')
EVAL_PROTOCOL = ('split', 'num_shot', 'eval_seed')  # the writer's own per-sample meta keys
FRAMES = ('room', 'heading')
GIT_STATE = ('HEAD', 'dirty', 'untracked', 'dirty_outside_worklog', 'diff_sha256')
REQUIRED_PROVENANCE = ('run_type', 'repo', 'reviewed_commit', 'source_closures',
                       'registry_sha256', 'git_state', 'environment', 'command',
                       'effective_args')
ENTRY_MODULES = {'full': 'tools.exp06_train', 'haa_train': 'tools.exp06_haa_finetune',
                 'haa_eval': 'tools.exp06_haa_eval', 'smoke': 'tools.exp06_smoke',
                 'probe': 'tools.exp06_smoke'}
DIAGNOSTIC_PROVENANCE = ('run_type', 'repo', 'reviewed_commit', 'source_closures',
                         'registry_sha256', 'git_state', 'environment', 'command')
# Finding 6: a receipt proves nothing without the runner's identity, its budgets and what
# the run actually cost. Every field is required and typed before `passed` is derived.
RECEIPT_NUMBERS = ('wall_s', 'alarm_seconds', 'max_gb')
DIAGNOSTIC_RECEIPT = ('runner', 'runner_closure_sha256', 'entry', 'argv', 'run_type',
                      'started_at', 'ended_at', 'exit_status', 'exploratory', 'git_head',
                      'peak_bytes', 'outcome') + RECEIPT_NUMBERS
OUTCOMES = ('ok', 'failed', 'aborted_alarm', 'aborted_memory')
SMOKE_ENTRIES = exp06_smoke.ENTRIES  # the only entries a diagnostic receipt may name
GIB = 1024 ** 3
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


def _identity_root(record, key):
    """The one root the run resolved, required to be the root the inventory describes."""
    declared = record.get('data_root')
    _require(isinstance(declared, str) and declared,
             'provenance.json records no resolved data_root for the {}'.format(key))
    resolved = str(Path(record[key]['data_root']).resolve())
    _require(str(Path(declared).resolve()) == resolved,
             '{}: data_root {} is not the {} the run resolved'.format(
                 key, record[key]['data_root'], declared))
    return resolved


def _membership(key, entries, expected):
    """Finding 3: no omission, no addition and no duplicate versus the derived membership."""
    recorded = {entry['path'] for entry in entries}
    _require(len(recorded) == len(entries),
             '{}: the inventory records {} entries for {} distinct paths'.format(
                 key, len(entries), len(recorded)))
    difference = sorted(set(expected) ^ recorded)
    _require(not difference, '{}: membership differs from the split by {} files, e.g. {}'.format(
        key, len(difference), difference[:2]))


def verify_heldout_identity(record, key='test_wav_identity'):
    """Finding 3a: the test-split waveforms every epoch's test loss was measured on.

    Membership comes from the pinned dataset's own split logic, never from the record,
    and every waveform is hashed again here.
    """
    entries = check_identity_schema(record.get(key), key)
    resolved = _identity_root(record, key)
    try:
        expected = exp06_train.heldout_wav_paths(resolved)
    except Exception as error:  # the dataset and the filesystem refuse alike
        raise ValueError('cannot derive the {} of {}: {}: {}'.format(
            key, resolved, type(error).__name__, error)) from error
    _membership(key, entries, expected)
    fresh = provenance._inventory(expected, resolved, workers=exp06_train.GEOMETRY_WORKERS)
    _require(fresh['inventory_sha256'] == record[key]['inventory_sha256'],
             '{}: the held-out waveforms changed since the run started'.format(key))
    return {'test_wav_files': len(entries), 'test_wav_bytes': fresh['inventory_bytes'],
            'test_wav_sha256': fresh['inventory_sha256']}


def verify_geometry_identity(record, key='geometry_identity'):
    """Finding 2: the metadata and depth maps the run consumed, rehashed at finalisation.

    Membership is derived here from the pinned dataset's own split logic, never read from
    the record, and every file is hashed again -- a changed source position or panorama
    is a refusal even though the IR waveforms are untouched.
    """
    entries = check_identity_schema(record.get(key), key)
    resolved = _identity_root(record, key)
    # Finding 3b: the required splits come from the full-run contract, never from the
    # record -- a correctly hashed splits=['train'] inventory left the held-out geometry
    # unbound and still certified.
    splits = record[key].get('splits')
    _require(isinstance(splits, list) and list(splits) == list(exp06_train.GEOMETRY_SPLITS),
             '{}: splits {!r} are not the {} a full run reads'.format(
                 key, splits, list(exp06_train.GEOMETRY_SPLITS)))
    try:
        expected, counts = exp06_train.geometry_paths(resolved, exp06_train.GEOMETRY_SPLITS)
    except Exception as error:  # the dataset and the filesystem refuse alike
        raise ValueError('cannot derive the {} of {}: {}: {}'.format(
            key, resolved, type(error).__name__, error)) from error
    _membership(key, entries, expected)
    started = time.monotonic()
    fresh = provenance._inventory(expected, resolved, workers=exp06_train.GEOMETRY_WORKERS)
    seconds = time.monotonic() - started
    _require(fresh['inventory_sha256'] == record[key]['inventory_sha256'],
             '{}: the geometry inputs changed since the run started'.format(key))
    # The measurement is reported, never recorded: a completion must be byte-identical on
    # a re-run, and a wall time never is.
    print('EXP06_GEOMETRY_REHASH {} files in {:.1f} s'.format(len(entries), seconds), flush=True)
    return {'geometry_files': len(entries), 'geometry_bytes': fresh['inventory_bytes'],
            'geometry_splits': dict(counts), 'geometry_sha256': fresh['inventory_sha256']}


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
    # Finding 3: the receipt's own pid was never checked, so a partially restored attempt
    # could certify while its recorded child was still writing. No owner exception here:
    # only the launcher's launch.pid may be alive at finalisation.
    _require(not _alive(receipt['child_pid']),
             'child_exit.json records child_pid {}, a process that is still alive'.format(
                 receipt['child_pid']))
    sidecar = Path(run_dir) / 'child.pid'
    if sidecar.is_file():
        recorded, _ = _pid_of(sidecar)
        _require(recorded == receipt['child_pid'],
                 'child.pid records {} but child_exit.json records child_pid {}'.format(
                     recorded, receipt['child_pid']))
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


def _alive(pid):
    """Whether one pid names a running process; somebody else's still counts."""
    _require(type(pid) is int and pid > 0, '{!r} is not a pid'.format(pid))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # Somebody else's live process is still a live process.
    return True


def _pid_of(path):
    """One pid file, or a named refusal; a stale file names a process that is gone."""
    try:
        pid = int(Path(path).read_text().split()[0])
    except (IndexError, ValueError) as error:
        raise ValueError('unreadable {}: {}'.format(path, error)) from error
    return pid, _alive(pid)


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


def load_provenance(run_dir, run_type, required=REQUIRED_PROVENANCE, identity=True):
    """The whole execution record, or a refusal naming the field that is missing."""
    record = _read_json(Path(run_dir) / 'provenance.json', 'provenance.json')
    missing = [key for key in required if record.get(key) is None]
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
    _require(present or not identity,
             'provenance.json records no train_data_identity/data_identity')
    for key in present:
        check_identity_schema(record[key], key)
    _require(not identity or isinstance(record['effective_args'], dict),
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


def _closure_files(files, label):
    """Every recorded file must carry both hashes before anything compares them."""
    _require(isinstance(files, list) and files, 'the recorded {} closure is empty'.format(label))
    for item in files:
        _require(isinstance(item, dict) and isinstance(item.get('path'), str)
                 and _is_sha256(item.get('working_tree_sha256'))
                 and _is_sha256(item.get('reviewed_blob_sha256')),
                 'incomplete {} closure file record: {!r}'.format(label, item))
    return files


def verify_orchestration(record, repo, commit, recorded_digests):
    """Finding 1: the launcher shell and the finalizer that decide this run, bound as files.

    A later round may edit either while a 31-hour training run is going, so their bytes are
    compared three ways -- the working tree now, what was recorded at spawn, and the
    reviewed blobs -- exactly as the training closure is.
    """
    closures = _mapping(record.get('orchestration_closures') or {},
                        'provenance.json orchestration_closures')
    _require(set(closures) == {'launcher', 'finalizer'},
             'provenance.json orchestration_closures must bind the launcher and the finalizer')
    for role, key in (('launcher', 'launch_sh'), ('finalizer', 'finalize')):
        closure = _mapping(closures[role], 'orchestration_closures.' + role)
        files = _closure_files(closure.get('files'), role)
        _require(closure.get('sha256') == closure_digest(files),
                 'recorded {} digest does not match its own file list'.format(role))
        _require(closure['sha256'] == recorded_digests.get(key),
                 'recorded {} digest is not the code_digests {}'.format(role, key))
        fresh, _ = exp06_profiles.closure_of(key, str(Path(repo).resolve()), commit)
        now = {item['path']: item for item in fresh}
        _require(sorted(now) == sorted(item['path'] for item in files),
                 '{} closure membership changed: {}'.format(role, ', '.join(sorted(
                     set(now) ^ {item['path'] for item in files}))))
        for item in files:
            seen = now[item['path']]
            agreed = {seen['working_tree_sha256'], item['working_tree_sha256'],
                      item['reviewed_blob_sha256'], seen['reviewed_blob_sha256']}
            _require(len(agreed) == 1, 'orchestration drift at {}: working tree {}, recorded '
                     '{}, reviewed {}'.format(item['path'], seen['working_tree_sha256'],
                                              item['working_tree_sha256'],
                                              item['reviewed_blob_sha256']))
    return {role: closures[role]['sha256'] for role in closures}


def verify_approvals(record, repo, run_type):
    """Finding 1: the training-critical code identities, three ways.

    Recomputed here, equal to what the run recorded at spawn, and equal to the approvals
    file re-read now -- whose own bytes the run bound, so a mid-run edit is a refusal.
    An exploratory diagnostic records its deviations instead, and is never an arm, so it
    alone may name approvals no reviewed commit carries.
    """
    exploratory = bool(record.get('exploratory'))
    _require(not exploratory or run_type in DIAGNOSTIC,
             'an exploratory run is never admissible as an arm')
    recorded = record.get('code_digests')
    _require(isinstance(recorded, dict)
             and set(recorded) == set(exp06_profiles.TRAINING_KEYS),
             'provenance.json records no code_digests for the training-critical keys')
    commit = record['reviewed_commit']
    current = exp06_profiles.compute_code_digests(repo, commit,
                                                  keys=exp06_profiles.TRAINING_KEYS)
    drift = sorted(key for key in exp06_profiles.TRAINING_KEYS
                   if current.get(key) != recorded.get(key))
    _require(not drift, 'code drift since the run started: ' + ', '.join(drift))
    orchestration = verify_orchestration(record, repo, commit, recorded)
    approvals = record.get('approvals')
    _require(isinstance(approvals, dict) and isinstance(approvals.get('path'), str),
             'provenance.json records no approvals binding')
    path = _resolve(approvals['path'], repo)
    _require(path.is_file(), 'missing approvals file: {}'.format(path))
    _require(provenance.sha256_file(path) == approvals.get('sha256'),
             'the approvals file {} changed since the run started'.format(path))
    # Review 2 finding 2: the recorded hash proves the file did not change during the run;
    # only its blob at the reviewed commit proves a reviewer approved those bytes.
    approved, identity = exp06_profiles.load_approved_digests(
        path, repo=None if exploratory else repo, commit=None if exploratory else commit)
    deviations = exp06_profiles.require(approved, exp06_profiles.TRAINING_KEYS, repo=repo,
                                        commit=commit, exploratory=exploratory, current=current)
    return {'approvals': dict(approvals, git_free_sha256=identity['sha256'],
                              committed_at=identity.get('committed_at')),
            'code_digests': dict(recorded), 'orchestration_digests': orchestration,
            'approval_deviations': deviations, 'exploratory': exploratory}


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
    admission = verify_approvals(record, repo, 'full')
    verify_train_identity(record)
    admission.update(verify_geometry_identity(record))
    admission.update(verify_heldout_identity(record))
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
    # Nit 8: torch.equal compares values across dtypes, so a float64 copy of a float32
    # checkpoint would pass. Identity requires the same dtype and shape as well.
    for key in sorted(state):
        _require(state[key].dtype == model[key].dtype,
                 '{} has dtype {} at {}, not the {} of last.pth["model"]'.format(
                     EPOCH_CHECKPOINT, state[key].dtype, key, model[key].dtype))
        _require(tuple(state[key].shape) == tuple(model[key].shape),
                 '{} has shape {} at {}, not the {} of last.pth["model"]'.format(
                     EPOCH_CHECKPOINT, tuple(state[key].shape), key, tuple(model[key].shape)))
    _require(all(torch.equal(state[key], model[key]) for key in state),
             EPOCH_CHECKPOINT + ' differs tensor-wise from last.pth["model"]')
    return dict(artifacts=hashes, epochs=len(rows), **admission,
                backbone=args['backbone'], source_closure_sha256=closure['sha256'],
                registry_sha256=record.get('registry_sha256'),
                git_head=record.get('git_state', {}).get('HEAD'),
                checkpoint={'path': EPOCH_CHECKPOINT, 'sha256': hashes[EPOCH_CHECKPOINT],
                            'epoch': exp06_recipe.EXP01_RECIPE['epochs']})


def diagnostic_receipt(receipt, run_type, provenance_record):
    """Finding 6: the runner's identity, budgets, timing and memory, all required."""
    _require(receipt is not None, 'a diagnostic run needs its --receipt')
    path = Path(receipt)
    _require(path.is_file(), 'missing smoke receipt: {}'.format(receipt))
    record = _read_json(path, 'smoke receipt')
    _require(record.get('diagnostic') is True, 'the smoke receipt is not marked diagnostic')
    missing = [key for key in DIAGNOSTIC_RECEIPT if key not in record]
    _require(not missing, 'the smoke receipt is incomplete: missing ' + ', '.join(missing))
    argv = record['argv']
    _require(isinstance(argv, list) and '--no-save' in argv,
             'a diagnostic must run with --no-save; its receipt records argv {!r}'.format(argv))
    _require(type(record['exit_status']) is int,
             'the smoke receipt records exit_status {!r}'.format(record['exit_status']))
    _require(record['runner'] == 'tools.exp06_smoke',
             'the smoke receipt records runner {!r}'.format(record['runner']))
    _require(_is_sha256(record['runner_closure_sha256']),
             'the smoke receipt records no runner closure digest')
    _require(isinstance(record['entry'], str) and record['entry'],
             'the smoke receipt records entry {!r}'.format(record['entry']))
    _require(record['run_type'] == run_type,
             'the smoke receipt records run_type {!r}, not {}'.format(record['run_type'], run_type))
    _require(record['outcome'] in OUTCOMES,
             'the smoke receipt records outcome {!r}, not one of {}'.format(
                 record['outcome'], list(OUTCOMES)))
    _require(type(record['exploratory']) is bool,
             'the smoke receipt records exploratory {!r}'.format(record['exploratory']))
    ended = _timestamp(record['ended_at'], 'ended_at')
    _require(ended >= _timestamp(record['started_at'], 'started_at'),
             'the smoke receipt ended_at precedes its started_at')
    for key in RECEIPT_NUMBERS:
        _require(_finite('the smoke receipt ' + key, record[key]) >= 0,
                 'the smoke receipt {} is {!r}, not a duration or budget'.format(key, record[key]))
    _require(type(record['peak_bytes']) is int and record['peak_bytes'] >= 0,
             'the smoke receipt peak_bytes is {!r}'.format(record['peak_bytes']))
    _require(record['git_head'] == provenance_record['git_state']['HEAD'],
             'the smoke receipt git_head {!r} is not the {} of its provenance'.format(
                 record['git_head'], provenance_record['git_state']['HEAD']))
    closure = list(provenance_record['source_closures'].values())[0]
    _require(record['runner_closure_sha256'] == closure['sha256'],
             'the smoke receipt runner closure is not the {} its provenance recorded'.format(
                 closure['sha256']))
    _require(record['exploratory'] == bool(provenance_record.get('exploratory')),
             'the smoke receipt and its provenance disagree about exploratory')
    return record, path


def check_receipt_consistency(fields, record, window):
    """Finding 4: field presence and two zero statuses do not make a diagnostic passed.

    Every receipt the runner writes is internally consistent, so each disagreement here is
    a forged or corrupted receipt: an entry outside the supported set or disagreeing with
    the startup command, a non-string argv, a budget that bounds nothing, an outcome that
    contradicts its own status or resource evidence, or a window outside the one the
    launcher recorded for the child. Returns whether the diagnostic actually passed.

    ``tools/exp06_smoke.py`` converts an over-budget run to ``aborted_memory`` or
    ``aborted_alarm`` before it publishes, so a real ``ok`` receipt never exceeds either
    ceiling. The child's ``ended_at`` is truncated to the second by the launcher's marker,
    which is the one second of slack allowed at the end of the window.
    """
    entry, argv = fields['entry'], fields['argv']
    _require(entry in SMOKE_ENTRIES, 'the smoke receipt records entry {!r}, not one of {}'
             .format(entry, sorted(SMOKE_ENTRIES)))
    _require(isinstance(argv, list) and all(isinstance(token, str) for token in argv),
             'the smoke receipt argv {!r} is not a list of strings'.format(argv))
    _require(record.get('entry') == entry, 'the smoke receipt ran the entry {!r}; its '
             'provenance records {!r}'.format(entry, record.get('entry')))
    _require(record.get('command') == [entry] + argv, 'the smoke receipt argv is not the '
             'child command {!r} its provenance recorded'.format(record.get('command')))
    alarm = _positive('the smoke receipt alarm_seconds', fields['alarm_seconds'])
    ceiling = _positive('the smoke receipt max_gb', fields['max_gb']) * GIB
    status, outcome, peak = fields['exit_status'], fields['outcome'], fields['peak_bytes']
    if outcome == 'ok':
        _require(status == 0, 'the smoke receipt records outcome ok with exit_status '
                 '{}'.format(status))
        _require(peak <= ceiling, 'the smoke receipt records outcome ok with a peak of {} '
                 'bytes over its {} GiB budget'.format(peak, fields['max_gb']))
        _require(fields['wall_s'] <= alarm, 'the smoke receipt records outcome ok after {} s '
                 'over its {} s budget'.format(fields['wall_s'], alarm))
    else:
        _require(status != 0, 'the smoke receipt records outcome {!r} with exit_status 0'
                 .format(outcome))
    started = _timestamp(fields['started_at'], 'started_at')
    ended = _timestamp(fields['ended_at'], 'ended_at')
    _require(_timestamp(window['started_at'], 'started_at') <= started,
             'the smoke receipt started at {}, before the child the launcher spawned at '
             '{}'.format(fields['started_at'], window['started_at']))
    _require(ended <= _timestamp(window['ended_at'], 'ended_at')
             + datetime.timedelta(seconds=1),
             'the smoke receipt ended at {}, after the child exited at {}'.format(
                 fields['ended_at'], window['ended_at']))
    return status == 0 and outcome == 'ok'


def diagnostic_evidence(run_dir, receipt, child_exit, run_type, repo, window):
    """A smoke or probe proves nothing about an arm, but must prove what it cost.

    Finding 6: the receipt carries the runner's identity, budgets, timing and peak
    allocation, and the run records a ``provenance.json`` bound the way a full run's is,
    minus the training artefacts. ``passed`` then reports whether the diagnostic actually
    succeeded; a failed one is still recorded and is simply never ``admissible_arm``.
    """
    record = load_provenance(run_dir, run_type, required=DIAGNOSTIC_PROVENANCE, identity=False)
    verify_source_closure(record, run_type, repo)
    admission = verify_approvals(record, repo, run_type)
    fields, path = diagnostic_receipt(receipt, run_type, record)
    status = fields['exit_status']
    passed = check_receipt_consistency(fields, record, window) and child_exit == 0
    return dict(artifacts={}, passed=passed, **admission,
                receipt={'path': str(path.resolve()), 'sha256': provenance.sha256_file(path),
                         'runner': fields['runner'], 'entry': fields['entry'],
                         'exit_status': status, 'outcome': fields['outcome'],
                         'wall_s': fields['wall_s'], 'peak_bytes': fields['peak_bytes'],
                         'alarm_seconds': fields['alarm_seconds'], 'max_gb': fields['max_gb']})


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


@functools.lru_cache(maxsize=None)
def expected_state_keys(backbone, num_shot):
    """The parameter names a checkpoint of this arm must carry, and nothing else."""
    with torch.random.fork_rng(devices=[]):
        return frozenset(build_xrir_exp06(backbone, num_shot).state_dict().keys())


def _isfinite(label, value):
    """Nit 3: an integer too wide to become a float is a named refusal, not an OverflowError
    escaping the numeric check."""
    try:
        return math.isfinite(value)
    except OverflowError:
        raise ValueError('{} is an integer of {} digits, too large for a finite number'.format(
            label, len(str(abs(value)))))


def _positive(label, value):
    """A budget of zero disables the thing it bounds, so it is never evidence."""
    _require(_finite(label, value) > 0, '{} is {!r}, not a positive budget'.format(label, value))
    return float(value)


def _finite(label, value):
    _require(type(value) in (int, float) and not isinstance(value, bool)
             and _isfinite(label, value), '{} is {!r}, not a finite number'.format(label, value))
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


def _validation_rooms(args, rooms):
    """The rooms whose validation loss selects ``best.pth``: --val-rooms, else the rooms.

    Round 2b finding 4: the wrapper's default is the training rooms, and a declared
    ``val_rooms`` must be a nonempty list of room names -- anything else is unusable and
    is refused before it can select a checkpoint.
    """
    declared = args.get('val_rooms')
    if declared is None:
        return list(rooms)
    _require(isinstance(declared, list) and declared
             and all(isinstance(room, str) for room in declared),
             'args.json records val_rooms {!r}, not a nonempty list of rooms'.format(declared))
    return declared


def _heading_binding(args, rooms, frame, repo):
    """Every room's binding names a heading JSON, and must agree with it in full.

    Full-train review finding 7: the record is re-read against the cache directory the
    run itself recorded (``haa_root``/``<room>``), so ``read_heading_json`` rehashes the
    four inputs the estimate was derived from -- a cache edited afterwards is refused
    here -- and only a ``confirmatory`` record, one of a clean tree whose closure equals
    its HEAD blobs, may bind a child.

    Round 2b finding 4: the rooms re-verified are the union of the training rooms and the
    effective validation rooms, because validation is what selects the checkpoint the next
    stage starts from. Training-room membership stays checked by ``_rooms_and_frame``.
    """
    if frame != 'heading':
        _require(not args.get('heading'), 'the room frame must not record a heading')
        return None
    heading = args.get('heading')
    rooms = sorted(set(rooms) | set(_validation_rooms(args, rooms)))
    _require(isinstance(heading, dict) and set(rooms) <= set(heading),
             'the heading frame requires a heading record for every room it trains or '
             'validates on: {}'.format(', '.join(sorted(set(rooms) - set(heading or ())))))
    root = args.get('haa_root')
    _require(isinstance(root, str) and root, 'the heading frame requires the resolved '
             'haa_root the run read, not {!r}'.format(root))
    cache = _resolve(root, repo)
    _require(cache.is_dir(), 'missing HAA cache root {} (args.json haa_root)'.format(cache))
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
            record = exp06_heading.read_heading_json(str(resolved), room_dir=str(cache / room))
        except (OSError, ValueError) as error:
            raise ValueError('invalid heading json for {}: {}'.format(room, error)) from error
        _require(record['admissibility'] == 'confirmatory', 'heading json for {} is {}, not '
                 'the confirmatory record a child may bind'.format(room, record['admissibility']))
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
    _require(type(args.get('seed')) is int, 'args.json records no integer seed')
    registry = registry_sha256()
    _require(record['registry_sha256'] == registry, 'provenance registry_sha256 {!r} is not '
             'the {} of BACKBONES_EXP06 at finalisation'.format(record['registry_sha256'], registry))
    return record, args


def child_identity(record):
    """Finding 2: the execution identity a job compares against its children's records."""
    return dict(source_closure_sha256=list(record['source_closures'].values())[0]['sha256'],
                registry_sha256=record['registry_sha256'],
                git_head=record['git_state']['HEAD'])


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
    return dict(child_identity(record), artifacts=hashes, rooms=rooms, frame=frame,
                heading=heading, backbone=args['backbone'], init_sha256=args['init_sha256'],
                seed=args['seed'], best_epoch=summary['best_epoch'], epochs=epochs)


def _numbers(label, values, count):
    """Finding 4: per-sample metrics are numbers -- NaN is the writer's invalid measurement,
    a string or a mapping is a corrupt observation and never a missing one."""
    _require(isinstance(values, list) and len(values) == count,
             '{} needs one value per index'.format(label))
    bad = [value for value in values if type(value) not in (int, float)]
    _require(not bad, '{} has non-numeric entries: {}'.format(label, sorted(map(repr, bad))[:4]))
    return [value for value in values if _isfinite(label, value)]


def _summary_block(label, summary, values):
    """The {mean, median, n} of eval_xRIR_backbone.summarize, recomputed from the values."""
    _require(isinstance(summary, dict) and set(summary) == set(SUMMARY_FIELDS),
             '{} is not a {} summary'.format(label, '/'.join(SUMMARY_FIELDS)))
    _require(type(summary['n']) is int and summary['n'] == len(values),
             '{} counts {!r} of {} finite per-sample values'.format(
                 label, summary['n'], len(values)))
    if not values:
        for field in ('mean', 'median'):
            _require(summary[field] is None, '{} {} is {!r}, and summarize writes null for '
                     'n == 0'.format(label, field, summary[field]))
        return
    for field, expected in (('mean', sum(values) / len(values)),
                            ('median', statistics.median(values))):
        _finite('{} {}'.format(label, field), summary[field])
        _require(math.isclose(summary[field], expected, rel_tol=1e-9, abs_tol=1e-9),
                 '{} {} {!r} is not the {!r} of the per-sample values'.format(
                     label, field, summary[field], expected))


def haa_metrics(run_dir, name, room, args, per_sample):
    """Blocker 3b and finding 4: every count, summary and observation is typed and
    recomputed here, never merely hashed."""
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
    _require(type(metrics['n_samples']) is int and metrics['n_samples'] == len(index),
             '{} records n_samples {!r}, not the {} per-sample entries'.format(
                 name, metrics['n_samples'], len(index)))
    for field in METRIC_COUNTS:
        _require(type(metrics[field]) is int and 0 <= metrics[field] <= len(index),
                 '{} records {} {!r}, not a count of at most the {} samples'.format(
                     name, field, metrics[field], len(index)))
    for key, field in sorted(METRIC_SOURCE.items()):
        label = '{}: per-sample {}'.format(name, field)
        finite = _numbers(label, per_sample.get(field), len(index))
        invalid = len(index) - len(finite)  # close-3 finding 2: what the counters describe
        counter, exact = METRIC_INVALID.get(field, (None, False))
        if key == 't60_error_pct' and room in NO_T60_ROOMS:
            _require(metrics[key] is None, '{}: {} is recorded for a room the paper omits'.format(
                name, key))
            measured = [value for value in per_sample[field] if not math.isnan(value)]
            _require(not measured, '{} records the t60 values {} in a room whose T60 the writer '
                     'never measures: every observation is its NaN'.format(
                         label, sorted(map(repr, measured))[:4]))
            _require(metrics[counter] == 0, '{} records {} {}, but nothing measures T60 in '
                     '{}'.format(name, counter, metrics[counter], room))
            continue
        if counter is not None:
            _require(metrics[counter] == invalid if exact else metrics[counter] <= invalid,
                     '{} records {} {}, which contradicts the {} non-finite per-sample {} '
                     'values'.format(name, counter, metrics[counter], invalid, field))
        _summary_block('{}: {}'.format(name, key), metrics[key], finite)
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
    required = (('backbone', 'checkpoint_sha256', 'frame') + EVAL_PROTOCOL
                + (('heading',) if heading else ()))
    missing = [key for key in required if key not in meta]
    _require(not missing, 'per-sample meta must record ' + ', '.join(missing))
    _require(meta['frame'] == frame and frame in FRAMES,
             'per-sample meta frame {!r} is not the {!r} of args.json'.format(meta['frame'], frame))
    _require(meta['backbone'] == args['backbone'],
             'per-sample meta backbone {!r} differs from args.json'.format(meta['backbone']))
    for field in EVAL_PROTOCOL:  # finding 3: observations of another protocol are not this arm's
        _require(exp06_recipe.strict_equal(meta[field], args.get(field)),
                 'per-sample meta {} {!r} is not the {!r} of args.json'.format(
                     field, meta[field], args.get(field)))
    _require(meta.get('room', room) == room, 'per-sample meta room {!r} is not the {!r} this '
             'child evaluated'.format(meta.get('room'), room))
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
    bad = [value for value in index if type(value) is not int or value < 0]
    _require(not bad, 'per-sample index entries must be the writer\'s non-negative integers, '
             'not {}'.format(sorted(map(repr, bad))[:4]))
    paths = per_sample.get('ir_path')
    _require(isinstance(paths, list) and len(paths) == len(index),
             'per-sample ir_path must record one <room>/<index> path per index, not {!r}'.format(
                 paths if not isinstance(paths, list) else len(paths)))
    wrong = [path for path, value in zip(paths, index) if path != '{}/{}'.format(room, value)]
    _require(not wrong, 'per-sample ir_path entries are not the {}/<index> this child '
             'evaluated: {}'.format(room, sorted(map(repr, wrong))[:4]))
    _require(isinstance(side, list) and len(side) == len(index),
             'side_label must carry one room-frame label per index ({} for {} samples)'.format(
                 len(side) if isinstance(side, list) else side, len(index)))
    bad = [value for value in side if isinstance(value, bool) or value not in (-1, 1)]
    _require(not bad, 'side_label values must be -1 or 1, not {}'.format(sorted(set(map(repr, bad)))))
    haa_metrics(run_dir, names[2], room, args, per_sample)
    return dict(child_identity(record), artifacts=hashes, room=room, frame=frame,
                heading=heading, seed=args['seed'], backbone=args['backbone'],
                checkpoint_sha256=digest, samples=len(index))


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
CHILD_EXTRA = {'haa_train': ('rooms', 'init_sha256', 'best_epoch', 'seed'),
               'haa_eval': ('room', 'checkpoint_sha256', 'samples', 'seed')}
JOB_SPEC = ('init', 'backbone', 'frame', 'init_sha256', 'seed', 'rooms', 'expect')


def child_role(name):
    """The run type this finalizer requires at one child path of a pipeline seed."""
    if name == 'stage1' or (name.startswith('stage2_') and name[len('stage2_'):] in ROOMS):
        return 'haa_train'
    for prefix in ('eval/', 'zeroshot/eval/'):
        if name.startswith(prefix) and name[len(prefix):] in ROOMS:
            return 'haa_eval'
    raise ValueError('unexpected child path: ' + name)


def child_completion(path, name, role):
    """Schema and role of one child's record, never its own admission claim."""
    completion = Path(path) / 'completion.json'
    _require(completion.is_file(), 'child {} has no completion.json'.format(name))
    record = _mapping(_read_json(completion, name + '/completion.json'),
                      name + '/completion.json')
    missing = [key for key in CHILD_COMPLETION + CHILD_EXTRA[role] if key not in record]
    _require(not missing, 'child {} completion is incomplete: missing {}'.format(
        name, ', '.join(missing)))
    _require(record['schema_version'] == 1,
             'child {} records schema_version {!r}'.format(name, record['schema_version']))
    _require(record['run_type'] == role,
             'child {} has run type {!r}, not the {!r} its path requires'.format(
                 name, record['run_type'], role))
    _require(isinstance(record['run_dir'], str) and record['run_dir'],
             'child {} records no run_dir path: {!r}'.format(name, record['run_dir']))
    _require(Path(record['run_dir']).resolve() == Path(path).resolve(),
             'child {} claims the run_dir {}'.format(name, record['run_dir']))
    _require(record['diagnostic'] is False, 'child {} is a diagnostic run'.format(name))
    _require(type(record['child_exit']) is int and record['child_exit'] == 0,
             'child {} records child_exit {!r}'.format(name, record['child_exit']))
    return record


def rehash_bound_evidence(record, name, path, repo):
    """Re-validate and rehash the log and receipt the child bound to its completion.

    The digests must still be those bytes, and the bytes must still satisfy the closed-log
    contract: the job never takes a child's word that its writers had gone. Finding 2: the
    receipt whose bytes are hashed must be the canonical ``<child>/child_exit.json`` that is
    validated, and the pid, timestamps and ``child_exit_time`` the completion recorded must
    be the ones that validation produced.
    """
    for key in ('log', 'child_exit_receipt'):
        bound = record[key]
        _require(isinstance(bound, dict) and isinstance(bound.get('path'), str)
                 and _is_sha256(bound.get('sha256')),
                 'child {} records no {} evidence: {!r}'.format(name, key, bound))
        file = _resolve(bound['path'], repo)
        _require(file.is_file(), 'child {} {} {} is gone'.format(name, key, bound['path']))
        _require(provenance.sha256_file(file) == bound['sha256'],
                 'child {} {} no longer hashes to its recorded digest'.format(name, key))
    log_record, marker_time, digest = closed_log(_resolve(record['log']['path'], repo),
                                                 record['child_exit'])
    _require(log_record['sha256'] == record['log']['sha256'],
             'child {} log is not the {} it bound'.format(name, record['log']['sha256']))
    receipt = child_exit_receipt(path, record['child_exit'], digest, marker_time)
    bound = record['child_exit_receipt']
    _require(_resolve(bound['path'], repo).resolve() == Path(receipt['path']),
             'child {} child_exit_receipt binds {}, not its own {}'.format(
                 name, bound['path'], receipt['path']))
    for field in ('sha256', 'child_pid', 'started_at', 'ended_at'):
        _require(exp06_recipe.strict_equal(bound.get(field), receipt[field]),
                 'child {} child_exit_receipt {} {!r} is not the {!r} of the canonical '
                 'receipt'.format(name, field, bound.get(field), receipt[field]))
    _require(record['child_exit_time'] == marker_time == receipt['ended_at'],
             'child {} records child_exit_time {!r}, not the {!r} of its validated log marker '
             'and receipt'.format(name, record['child_exit_time'], marker_time))


def verify_child(path, name, repo, spec):
    """Blocker 3c: re-run the child's own validator and bind its record to that result.

    Nothing here trusts the child: its role validator runs again over the directory, and
    every field of its completion -- artefact hashes, identity, lineage -- must equal what
    the re-run produced. The log and receipt it bound are rehashed, and the whole child is
    then checked against the job the pipeline declared.

    Finding 1: the liveness contract holds for every child as well. A live ``child.pid``
    or ``launch.pid`` under a child directory is never admissible -- the owner exception
    of ``--owner-pid`` covers the job's own launcher only, so a job can be certified only
    once every one of its children has exited.
    """
    refuse_live_launch(path)
    role = child_role(name)
    evidence = (haa_train_evidence(path, repo) if role == 'haa_train'
                else haa_eval_evidence(path, repo))
    record = child_completion(path, name, role)
    recorded = _mapping(record['artifacts'], 'child {} artifacts'.format(name))
    fresh = evidence['artifacts']
    # Close-3 finding 1: compare the whole mappings, never through .get() -- an absent key and
    # a recorded null agree there, so an invented 'nonexistent.pth': null would pass.
    _require(set(recorded) == set(fresh), 'child {} records the artefacts {}, not the {} the '
             're-run hashed'.format(name, sorted(recorded), sorted(fresh)))
    for artefact in sorted(fresh):
        _require(exp06_recipe.strict_equal(recorded[artefact], fresh[artefact]),
                 'child {} artefact {} is not the one the re-run hashed'.format(name, artefact))
    for field in sorted(set(evidence) - {'artifacts'}):
        _require(field in record, 'child {} completion records no {}'.format(name, field))
        _require(exp06_recipe.strict_equal(record[field], evidence[field]),
                 'child {} completion {} {!r} is not the {!r} of the re-run'.format(
                     name, field, record[field], evidence[field]))
    rehash_bound_evidence(record, name, path, repo)
    check_job_spec(name, role, evidence, spec,
                   _read_json(Path(path) / 'args.json', 'args.json'))
    return evidence


def check_job_identity(name, args, spec):
    """Round 2b finding 3: the child was launched under this job's own declaration.

    ``tools/exp06_haa_finetune.py`` and ``tools/exp06_haa_eval.py`` record the ``--job-spec``
    the pipeline passed them -- its digest and the initialisation it declares -- before the
    first epoch, so an initialisation label is frozen at launch and not merely asserted by
    the spec at admission. A child that recorded no declaration, or a different one, is
    never part of this job. The digest is compared last, so a contradicted field names
    itself first.
    """
    for field, declared in (('job_init', spec['init']), ('job_init_sha256', spec['init_sha256'])):
        recorded = args.get(field)
        _require(isinstance(recorded, str) and recorded, 'child {} recorded no {}: it was not '
                 'launched under a job spec'.format(name, field))
        _require(recorded == declared, 'child {} was launched under the {} {!r}, not the {!r} '
                 'of this job spec'.format(name, field, recorded, declared))
    digest = args.get('job_spec_sha256')
    _require(isinstance(digest, str) and digest,
             'child {} recorded no job_spec_sha256: it was not launched under a job '
             'spec'.format(name))
    _require(digest == spec['job_spec_sha256'], 'child {} was launched under the job spec '
             '{}, not the {} being admitted'.format(name, digest, spec['job_spec_sha256']))


def check_job_spec(name, role, evidence, spec, args):
    """Every child must have run the job the pipeline declared, not one of its own."""
    for field in ('backbone', 'frame', 'seed'):
        _require(evidence[field] == spec[field], 'child {} ran {} {!r}, not the {!r} of the '
                 'job spec'.format(name, field, evidence[field], spec[field]))
    rooms = evidence['rooms'] if role == 'haa_train' else [evidence['room']]
    outside = sorted(set(rooms) - set(spec['rooms']))
    _require(not outside, 'child {} covers rooms outside the job: {}'.format(name, outside))
    room = name.split('/')[-1] if role == 'haa_eval' else name[len('stage2_'):]
    _require(not name.startswith(('stage2_', 'eval/', 'zeroshot/')) or rooms == [room],
             'child {} records rooms {}, not the {!r} of its path'.format(name, rooms, room))
    if spec['frame'] == 'heading':
        for bound, binding in sorted((evidence['heading'] or {}).items()):
            _require(binding.get('k') == spec['heading'].get(bound),
                     'child {} rolls the {} heading to {!r}, not the {!r} of the job '
                     'spec'.format(name, bound, binding.get('k'), spec['heading'].get(bound)))
    else:
        _require(not evidence['heading'], 'child {} records a heading in the room frame'.format(name))
    check_job_identity(name, args, spec)


def load_job_spec(path, expect):
    """The pipeline's declaration of one seed: what every child of it must agree with.

    Finding 5: every nested value is typed before it is used, so a malformed spec is a
    named refusal (CLI exit 2, nothing written) and never a TypeError out of ``sorted``.
    """
    _require(path, 'a job needs the pipeline --job-spec it was run from')
    spec = _mapping(_read_json(path, 'job spec'), 'job spec')
    missing = [key for key in JOB_SPEC if key not in spec]
    _require(not missing, 'job spec is incomplete: missing ' + ', '.join(missing))
    _require(spec['expect'] == expect,
             'job spec declares {!r}, not the --expect {!r}'.format(spec['expect'], expect))
    _require(spec['backbone'] in BACKBONES, 'job spec backbone {!r}'.format(spec['backbone']))
    _require(spec['frame'] in FRAMES, 'job spec frame {!r}'.format(spec['frame']))
    _require(_is_sha256(spec['init_sha256']),
             'job spec init_sha256 {!r} is not a sha256'.format(spec['init_sha256']))
    _require(type(spec['seed']) is int, 'job spec seed {!r} is not an integer'.format(spec['seed']))
    _require(isinstance(spec['init'], str) and spec['init'],
             'job spec init {!r} is not an initialisation name'.format(spec['init']))
    _require(isinstance(spec['rooms'], list)
             and all(isinstance(room, str) for room in spec['rooms'])
             and sorted(spec['rooms']) == sorted(ROOMS),
             'job spec rooms {!r} are not the pipeline rooms'.format(spec['rooms']))
    if spec['frame'] == 'heading':
        heading = _mapping(spec.get('heading'), 'job spec heading')
        absent = [room for room in spec['rooms'] if type(heading.get(room)) is not int
                  or not 0 <= heading[room] < WIDTH]
        _require(not absent, 'job spec records no heading roll in [0, {}) for {}'.format(
            WIDTH, ', '.join(absent)))
    else:
        _require(not spec.get('heading'), 'a room-frame job spec declares no heading')
    # Finding 3: the declaration's own digest, which every child must have recorded at launch.
    spec['job_spec_sha256'] = provenance.sha256_file(path)
    return spec


def job_lineage(records, expect, spec):
    """The init -> stage1 -> stage2 -> evaluation chain the job spec declares."""
    fields = dict(backbone=spec['backbone'], frame=spec['frame'], seed=spec['seed'],
                  heading=spec['heading'] if spec['frame'] == 'heading' else None,
                  init=spec['init'], init_sha256=spec['init_sha256'])
    if expect == 'zeroshot':
        for name, record in sorted(records.items()):
            _require(record['checkpoint_sha256'] == spec['init_sha256'],
                     'lineage: {} did not evaluate the job initialisation'.format(name))
        return dict(fields, checkpoint_sha256=spec['init_sha256'])
    _require(records['stage1']['init_sha256'] == spec['init_sha256'],
             'lineage: stage1 did not start from the job initialisation')
    stage1 = records['stage1']['artifacts']['best.pth']
    for name, record in sorted(records.items()):
        if name.startswith('stage2_'):
            _require(record['init_sha256'] == stage1,
                     'lineage: {} did not start from stage1/best.pth'.format(name))
        elif name.startswith('eval/'):
            room = name[len('eval/'):]
            _require(record['checkpoint_sha256'] == records['stage2_' + room]['artifacts']['best.pth'],
                     'lineage: {} did not evaluate stage2_{}/best.pth'.format(name, room))
    return fields


def haa_job_evidence(run_dir, children, expect, repo, job_spec):
    """Bind one seed: every expected child, re-validated in the role its path requires."""
    expected, job = set(expected_children(expect)), Path(run_dir).resolve()
    spec = load_job_spec(job_spec, expect)
    seen, records = {}, {}
    for child in children:
        path = Path(child).resolve()
        try:
            name = path.relative_to(job).as_posix()
        except ValueError as error:
            raise ValueError('child {} lies outside the job directory'.format(child)) from error
        if expect == 'zeroshot' and name.startswith('zeroshot/'):
            name = name[len('zeroshot/'):]  # <init>/zeroshot/eval/<room> and <init>/zeroshot alike
        _require(name in expected, 'unexpected child: ' + name)
        records[name] = verify_child(path, name, repo, spec)
        seen[name] = provenance.sha256_file(path / 'completion.json')
    missing = sorted(expected - set(seen))
    _require(not missing, 'job is missing children: ' + ', '.join(missing))
    return dict(job_lineage(records, expect, spec), artifacts={}, children=seen, expect=expect,
                job_spec={'path': str(Path(job_spec).resolve()),
                          'sha256': provenance.sha256_file(job_spec)})


def write_completion(path, fields):
    """Publish once; a re-run must produce the same bytes, a different result is refused."""
    payload = json.dumps(fields, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    if Path(path).exists():
        _require(Path(path).read_bytes() == payload,
                 'completion.json already exists and differs from this result')
        return fields
    provenance.write_completion(path, fields)
    return fields


def job_log(log):
    """A job closes no log of its own, so the queue log is recorded, never read as proof."""
    if log is None:
        return None
    path = Path(log)
    _require(path.is_file(), 'missing job log: {}'.format(log))
    return {'path': str(path.resolve()), 'sha256': provenance.sha256_file(path)}


def haa_job_completion(run_dir, children, expect, repo, job_spec, log, child_exit, owner_pid):
    """One pipeline job: no child of its own, hence no exit receipt and no closed log (A3).

    The orchestrating shell is the process that finalizes the job, so the only execution
    evidence a job root can hold is its own still-live ``launch.pid``; a receipt there could
    only name that same shell. What binds a job is the job spec, that owner, and every
    expected child's completion, each re-validated by ``haa_job_evidence``.

    Pre-merge finding 1: the owner is bound unconditionally. ``tools/exp06_haa_pipeline.sh``
    writes the job root's ``launch.pid`` when it opens the root, so a job that records none
    -- or one that is unreadable -- is missing the very binding A3 requires, and is refused
    rather than certified with a null owner because ``--owner-pid`` happened to be omitted.
    A recovery finalisation of a job whose launcher has died stays admissible: what the
    owner may not be is absent.
    """
    _require(not (run_dir / 'child_exit.json').exists(),
             'job roots carry no child exit receipt (A3)')
    owner = refuse_live_launch(run_dir, owner_pid)
    _require(owner is not None, '{} records no launch.pid: a job binds the owner that ran '
             'it'.format(run_dir))
    _require(owner_pid is None or owner == owner_pid,
             'the job root holds launch.pid {}, not the declared owner {}'.format(
                 owner, owner_pid))
    _require(child_exit == 0, 'the job was declared with child status {}'.format(child_exit))
    fields = dict(schema_version=1, run_type='haa_job', run_dir=str(run_dir.resolve()),
                  repo=str(Path(repo).resolve()), child_exit=child_exit, log=job_log(log),
                  owner_pid=owner, diagnostic=False, admissible_arm=True)
    fields.update(haa_job_evidence(run_dir, children, expect, repo, job_spec))
    return write_completion(run_dir / 'completion.json', fields)


def finalize(run_dir, run_type, log, child_exit, repo=REPO, receipt=None,
             children=(), expect=None, owner_pid=None, job_spec=None):
    """Verify one child's evidence for its run type and write completion.json."""
    run_dir = Path(run_dir)
    _require(run_type in RUN_TYPES, 'unknown run type: {!r}'.format(run_type))
    _require(run_dir.is_dir(), 'run directory does not exist: {}'.format(run_dir))
    _require(type(child_exit) is int, 'child status must be an integer')
    if run_type == 'haa_job':  # A3: a job orchestrates children and is none itself
        return haa_job_completion(run_dir, children, expect, repo, job_spec, log, child_exit,
                                  owner_pid)
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
    fields.update(diagnostic_evidence(run_dir, receipt, child_exit, run_type, repo,
                                      receipt_record) if diagnostic
                  else full_evidence(run_dir, repo) if run_type == 'full'
                  else haa_train_evidence(run_dir, repo) if run_type == 'haa_train'
                  else haa_eval_evidence(run_dir, repo))
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


def gpu_free_gib(gpu):
    """Free memory on one card, in GiB; an unreadable or ambiguous reply is refused.

    Plan amendment A5: a rung-4 smoke shares a card with whatever else is on it, so it may
    start only where the whole budget it is allowed to allocate is still free. The query is
    separate from the compute-app census, which ``probe``/``full`` use to demand an empty
    card.
    """
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits',
             '-i', str(gpu)], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError('cannot query the free memory of GPU {}: {}'.format(gpu, error)) from error
    values = [line.strip() for line in output.splitlines() if line.strip()]
    _require(len(values) == 1,
             'GPU {} reported {!r} free, not one measurement'.format(gpu, values))
    try:
        mib = float(values[0])
    except ValueError as error:
        raise ValueError('GPU {} reported {!r} free, not a number of MiB'.format(
            gpu, values[0])) from error
    _require(math.isfinite(mib) and mib >= 0,
             'GPU {} reported {!r} free'.format(gpu, values[0]))
    return mib / 1024.0


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


def approval_gate(mode, reviewed_commit, approved, exploratory, repo):
    """Finding 1: no confirmatory launch on code the approvals do not pin.

    Null approvals refuse every mode; only a diagnostic may proceed ``--exploratory``,
    and then its deviations are recorded in the preflight record and in its receipt.

    Review 2 finding 2: every launch this gate admits reads its approvals from the blob
    committed at the reviewed commit. Only a smoke or probe declared ``--exploratory``
    may read an external or uncommitted file, and it is never admissible as an arm.
    """
    _require(not (exploratory and mode == 'full'),
             'an exploratory launch is a diagnostic; mode full must match the approvals')
    bind = not (exploratory and mode in DIAGNOSTIC)
    path = _resolve(exp06_profiles.APPROVED_RELATIVE if approved is None else approved, repo)
    _require(path.is_file(), 'missing approvals file: {}'.format(path))
    value, identity = exp06_profiles.load_approved_digests(
        path, repo=repo if bind else None, commit=reviewed_commit if bind else None)
    deviations = exp06_profiles.require(value, exp06_profiles.TRAINING_KEYS, repo=repo,
                                        commit=reviewed_commit, exploratory=exploratory)
    return {'approved': identity, 'approval_deviations': deviations,
            'exploratory': bool(exploratory)}


def preflight(mode, gpu, reviewed_commit, attempt_root=None, repo=REPO, approved=None,
              exploratory=False, min_free_gb=None):
    """Gate a launch: reviewed commit, clean tree, no live launch, approvals, a free card."""
    _require(mode in LAUNCH_MODES, 'unknown launch mode: {!r}'.format(mode))
    state = provenance.checked_git_state(repo, confirmatory=True)
    _require(state['HEAD'] == reviewed_commit,
             'HEAD {} is not the reviewed commit {!r} (full 40-hex sha required)'.format(
                 state['HEAD'], reviewed_commit))
    running = live_launches(attempt_root)
    _require(not running, 'another exp_06 launch is alive: {}'.format(running))
    admission = approval_gate(mode, reviewed_commit, approved, exploratory, repo)
    apps = gpu_compute_apps(gpu) if mode in EXCLUSIVE_GPU_MODES else None
    _require(not apps, 'GPU {} is busy with compute apps {}'.format(gpu, apps))
    free = None if min_free_gb is None else gpu_free_gib(gpu)
    _require(free is None or free >= min_free_gb,
             'GPU {} has {:.2f} GiB free, below the {} GiB this launch may allocate'.format(
                 gpu, free or 0.0, min_free_gb))
    return dict(mode=mode, gpu=gpu, reviewed_commit=reviewed_commit, git_state=state,
                attempt_root=[str(root) for root in launch_roots(attempt_root)],
                gpu_compute_apps=apps, live_launches=running, min_free_gb=min_free_gb,
                gpu_free_gib=free, **admission)


def preflight_main(argv):
    """Exit 0 with the record on stdout, 2 with the named cause on stderr."""
    parser = argparse.ArgumentParser(description='Gate one exp_06 launch.')
    parser.add_argument('--mode', choices=LAUNCH_MODES, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--reviewed-commit', required=True)
    parser.add_argument('--attempt-root', action='append', default=[],
                        help='repeatable: every root whose */launch.pid must be dead')
    parser.add_argument('--repo', default=str(REPO))
    parser.add_argument('--approved', default=None,
                        help='approved_digests.json (default: the record asset)')
    parser.add_argument('--exploratory', action='store_true',
                        help='diagnostic launch on unapproved code; never mode full')
    parser.add_argument('--min-free-gb', type=float, default=None,
                        help='A5: GiB this launch may allocate; the card must hold them free')
    args = parser.parse_args(argv)
    try:
        record = preflight(args.mode, args.gpu, args.reviewed_commit, args.attempt_root,
                           args.repo, approved=args.approved, exploratory=args.exploratory,
                           min_free_gb=args.min_free_gb)
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
    parser.add_argument('--job-spec', help='haa_job: the spec the pipeline ran this seed from')
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
                          owner_pid=args.owner_pid, job_spec=args.job_spec)
    except (OSError, ValueError) as error:
        print('EXP06_FINALIZE_REFUSED ' + str(error), file=sys.stderr, flush=True)
        return 2
    print('EXP06_FINALIZE_OK ' + json.dumps({key: fields[key] for key in
        ('run_type', 'run_dir', 'child_exit', 'admissible_arm')}, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
