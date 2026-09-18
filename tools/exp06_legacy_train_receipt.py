"""exp_06's historical-evidence branch for arm B's exp_01 training (plan 6.4, full review F2).

Arm C's simulated evaluation binds the finalised pretraining attempt -- its
``provenance.json`` and the ``completion.json`` that certified the twelve epochs. Arm B
evaluates the **exp_01** cylindrical checkpoint, whose training predates the exp_06
lifecycle entirely: no ``completion.json`` was ever written for it, and none can be
without rerunning a 31-hour job. Section 6.4 registers one answer for exactly this case --
a ``reconstructed`` receipt, written once, enumerating the retained artefacts byte for
byte -- which is what the historical HAA arms already get from
``tools.exp06_summarize_haa --write-legacy-receipt``. This module is that receipt for a
*training* directory.

    python tools/exp06_legacy_train_receipt.py --train-dir ckpt/xRIR_cyl_8_shot \
        --checkpoint ckpt/xRIR_cyl_8_shot/epoch_12.pth \
        --out ckpt/exp06/train_receipt_cyl.json

The writer is gated, not descriptive: the evaluated weights must hash to one of the
**registered** exp_01 arms of ``tools.exp04_profiles`` (so a receipt can never be written
for weights no experiment registered), the checkpoint must be a per-epoch file of the
enumerated directory, ``history.jsonl`` must record exactly the epochs ``args.json``
declares, and the checkpoint's own file name must name the last of them. The receipt is
created exclusively and never replaced.

``verify`` re-reads it and re-hashes every file it enumerates, so it is evidence only
while those artefacts are unchanged; ``tools.exp06_train_evidence`` is the caller that
binds it to an evaluation. Full-review R3: it also re-derives what the receipt *must*
enumerate from :func:`receipt_names`, re-validates the args/history/registered-arm
contract against the retained artefacts, and refuses a receipt whose producer closure
drifted or was written with ``--allow-dirty`` -- a self-consistent enumeration of two of
eighteen artefacts is not complete evidence of a training run.
"""
import argparse
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

from tools import exp04_profiles
from tools import provenance

ENTRY_MODULE = 'tools.exp06_legacy_train_receipt'
REPO = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
LABEL = 'reconstructed'
EXPERIMENT = 'exp_01'
# The retained record of an exp_01 training run: the arguments it ran with, the per-epoch
# history the budget is read from, and the run log. All three are required.
REQUIRED_FILES = ('args.json', 'history.jsonl', 'train.log')
# Enumerated when present; exp_01 kept both, but neither is evidence on its own.
OPTIONAL_FILES = ('best.pth', 'last.pth')
EPOCH_FILE = re.compile(r'epoch_(\d+)\.pth\Z')
COMMIT = re.compile(r'[0-9a-f]{40}\Z')
# `closure_record` writes git's own object hash (40 hex) or a sha256 of the bytes (64).
FILE_HASH = re.compile(r'([0-9a-f]{40}|[0-9a-f]{64})\Z')
# The arms exp_04 registers by hash; a receipt is written only for weights among them.
REGISTERED = (exp04_profiles.CONTROL, exp04_profiles.CYL)


def _require(ok, cause):
    if not ok:
        raise ValueError(cause)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError('unreadable {} ({}): {}'.format(label, path, error))


def epoch_checkpoints(train_dir):
    """Every ``epoch_NN.pth`` of the directory, ordered by the epoch it records."""
    found = {}
    for path in sorted(Path(train_dir).iterdir()):
        match = EPOCH_FILE.match(path.name)
        if match and path.is_file():
            found.setdefault(int(match.group(1)), []).append(path.name)
    return [name for epoch in sorted(found) for name in sorted(found[epoch])]


def history_epochs(path):
    """The epochs ``history.jsonl`` records: exactly ``1..N``, once each, in order."""
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise ValueError('history.jsonl line {} is not JSON: {}'.format(number, error))
        _require(isinstance(row, dict) and type(row.get('epoch')) is int
                 and not isinstance(row.get('epoch'), bool),
                 'history.jsonl line {} records no integer epoch'.format(number))
        rows.append(row['epoch'])
    _require(rows == list(range(1, len(rows) + 1)),
             'history.jsonl records the epochs {}, not 1..{}'.format(rows, len(rows)))
    return len(rows)


def receipt_names(train_dir, checkpoints=None):
    """The membership rule: which artefacts of a training directory a receipt enumerates.

    One rule, written once and applied twice -- by the writer, to enumerate, and by
    :func:`verify`, to require that a receipt enumerates all of them (R3: a receipt cut
    down to two files and re-hashed is self-consistent, and is not complete evidence).
    """
    train_dir = Path(train_dir)
    _require(train_dir.is_dir(), 'no training directory at {}'.format(train_dir))
    names = list(REQUIRED_FILES)
    for name in names:
        _require((train_dir / name).is_file(),
                 'the training directory {} has no {}'.format(train_dir, name))
    names += list(epoch_checkpoints(train_dir) if checkpoints is None else checkpoints)
    names += [name for name in OPTIONAL_FILES if (train_dir / name).is_file()]
    return names


def receipt_files(train_dir, checkpoints=None):
    """The receipt's ordered enumeration, relative to the training directory."""
    return [{'path': name, 'sha256': provenance.sha256_file(Path(train_dir) / name)}
            for name in receipt_names(train_dir, checkpoints)]


def source_identity(repo=REPO, strict=True):
    """This tool's own closure, bound to HEAD; strict refuses an edited working tree."""
    head = provenance.git_state(repo)['HEAD']
    records, digest = provenance.closure_record(
        provenance.source_closure(ENTRY_MODULE, repo), head, repo)
    drift = sorted(record['path'] for record in records
                   if record['reviewed_blob_sha256'] is None
                   or record['reviewed_blob_sha256'] != record['working_tree_sha256'])
    _require(not (strict and drift),
             'the producer closure differs from HEAD: ' + ', '.join(drift))
    return {'entry_module': ENTRY_MODULE, 'commit': head, 'sha256': digest,
            'files': records, 'drift': drift, 'strict': bool(strict)}


def _closure_digest(files):
    """``provenance.closure_record``'s own digest: [[path, reviewed blob], ...]."""
    return hashlib.sha256(json.dumps([[item['path'], item['reviewed_blob_sha256']]
                                      for item in files], sort_keys=True).encode()).hexdigest()


def _reviewed_blob(repo, commit, path):
    """The blob of ``path`` at ``commit``, hashed as ``closure_record`` hashes it."""
    blob = subprocess.run(['git', 'show', '{}:{}'.format(commit, path)],
                          cwd=str(repo), capture_output=True)
    return hashlib.sha256(blob.stdout).hexdigest() if blob.returncode == 0 else None


def check_producer(closure, repo=REPO):
    """Close-review R3: what wrote the receipt, re-derived from the closure's own records.

    ``verify`` used to read two summary flags -- ``strict`` and ``drift`` -- so a receipt
    carrying ``{"strict": true}`` and nothing else claimed a clean producer it never had.
    A producer closure is evidence only if it *is* one: this tool's own entry module, a
    real commit, the non-empty membership :func:`source_identity` records, the digest
    ``closure_record`` computes over it, records that do not differ from the reviewed
    blobs, and reviewed blobs that are the blobs of those paths **at that commit** -- a
    historical commit as readily as HEAD, since a receipt is written once and read later.

    The closure is this tool's import closure, which never contains a ``worklog/`` path,
    so 6.4's allowance for a dirty notebook needs no exception here: a dirty worklog file
    is not a source of ``tools.exp06_legacy_train_receipt`` and can never be drift.
    """
    _require(isinstance(closure, dict), 'the training receipt records no producer closure')
    _require(closure.get('strict') is True, 'the training receipt was produced with '
             '--allow-dirty, and is never confirmatory')
    _require(closure.get('entry_module') == ENTRY_MODULE, 'the training receipt was '
             'produced by {!r}, not {}'.format(closure.get('entry_module'), ENTRY_MODULE))
    commit = closure.get('commit')
    _require(isinstance(commit, str) and COMMIT.match(commit),
             'the producer closure records no commit ({!r})'.format(commit))
    files = closure.get('files')
    _require(isinstance(files, list) and files,
             'the producer closure enumerates no source file of ' + ENTRY_MODULE)
    for item in files:
        item = item if isinstance(item, dict) else {}
        _require(isinstance(item.get('path'), str)
                 and FILE_HASH.match(str(item.get('reviewed_blob_sha256')))
                 and FILE_HASH.match(str(item.get('working_tree_sha256'))),
                 'the producer closure records no reviewed and working-tree hashes for '
                 '{!r}'.format(item.get('path')))
    names = [item['path'] for item in files]
    _require(names == sorted(set(names)),
             'the producer closure enumerates its sources out of order or twice')
    declared = closure.get('drift') or []
    _require(isinstance(declared, list), 'the producer closure records a malformed drift')
    # R3: drift is what the records say, not what the receipt claims about them.
    drift = sorted(set(str(name) for name in declared)
                   | {item['path'] for item in files
                      if item['working_tree_sha256'] != item['reviewed_blob_sha256']})
    _require(not drift, 'the training receipt was produced from a tree that differs from '
             'its own sources ({}), and is never confirmatory'.format(', '.join(drift)))
    _require(_closure_digest(files) == closure.get('sha256'),
             'the producer closure does not hash to its own sha256')
    unreviewed = [item['path'] for item in files
                  if _reviewed_blob(repo, commit, item['path'])
                  != item['reviewed_blob_sha256']]
    _require(not unreviewed, 'the producer closure is not the reviewed source of {} at {}: '
             '{}'.format(ENTRY_MODULE, commit[:12], ', '.join(unreviewed)))
    return {'entry_module': ENTRY_MODULE, 'commit': commit, 'sha256': closure['sha256'],
            'files': len(files)}


def registered_arm(digest, registry=None):
    """The exp_01 arm these weights are, or a refusal naming what was offered."""
    registry = REGISTERED if registry is None else registry
    for arm in registry:
        if arm['sha256'] == digest:
            return {'role': arm['role'], 'backbone': arm['backbone'],
                    'checkpoint': arm['checkpoint'], 'epoch': arm['epoch'],
                    'sha256': arm['sha256']}
    raise ValueError('the checkpoint hashes to {}, which is not a registered exp_01 arm '
                     '({})'.format(digest, ', '.join(sorted(arm['role'] for arm in registry))))


def build_receipt(train_dir, checkpoint, repo=REPO, strict=True, registry=None,
                  identity=None):
    """The reconstructed receipt: what the exp_01 training left behind, byte for byte."""
    train_dir = Path(train_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    _require(checkpoint.is_file(), 'no checkpoint at {}'.format(checkpoint))
    _require(checkpoint.parent == train_dir, 'the checkpoint {} is not in the enumerated '
             'training directory {}'.format(checkpoint, train_dir))
    match = EPOCH_FILE.match(checkpoint.name)
    _require(match is not None, 'the evaluated checkpoint {} is not an epoch_NN.pth of the '
             'training run'.format(checkpoint.name))
    files = receipt_files(train_dir)
    enumerated = {item['path']: item['sha256'] for item in files}
    _require(checkpoint.name in enumerated,
             'the checkpoint {} is not enumerated'.format(checkpoint.name))
    args = _read_json(train_dir / 'args.json', 'args.json')
    _require(isinstance(args, dict), 'args.json is not a record')
    declared = args.get('epochs')
    _require(type(declared) is int and not isinstance(declared, bool) and declared > 0,
             'args.json records no positive integer epochs')
    epochs = history_epochs(train_dir / 'history.jsonl')
    _require(epochs == declared, 'history.jsonl records {} epochs, not the {} of args.json'
             .format(epochs, declared))
    _require(int(match.group(1)) == epochs, 'the evaluated {} is not the last of the {} '
             'epochs the history records'.format(checkpoint.name, epochs))
    arm = registered_arm(enumerated[checkpoint.name], registry)
    _require(args.get('backbone') == arm['backbone'],
             'args.json records the backbone {!r}, not the {!r} of the registered {}'.format(
                 args.get('backbone'), arm['backbone'], arm['role']))
    _require(int(arm['epoch']) == epochs, 'the registered {} is epoch {}, not the {} this '
             'directory trained'.format(arm['role'], arm['epoch'], epochs))
    return {'schema_version': SCHEMA_VERSION, 'label': LABEL, 'experiment': EXPERIMENT,
            'arm': arm['role'], 'backbone': arm['backbone'], 'registered': arm,
            'train_dir': str(train_dir), 'repo': str(Path(repo).resolve()),
            'epochs': epochs,
            'checkpoint': {'path': checkpoint.name, 'epoch': epochs,
                           'sha256': enumerated[checkpoint.name]},
            'args': {'path': 'args.json', 'sha256': enumerated['args.json'],
                     'epochs': declared, 'backbone': args.get('backbone')},
            'files': files, 'files_sha256': _digest(files),
            'git': provenance.git_state(repo),
            'source_closure': source_identity(repo, strict) if identity is None else identity,
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()}


def write_receipt(path, train_dir, checkpoint, repo=REPO, strict=True, registry=None,
                  identity=None):
    """Write once; a receipt is never silently replaced (``write_manifest`` is O_EXCL)."""
    record = build_receipt(train_dir, checkpoint, repo, strict, registry, identity)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return record, provenance.write_manifest(path, record)


def verify(path, checkpoint_sha256=None, registered_sha256=None, epochs=None,
           hasher=provenance.sha256_file, repo=None):
    """Re-read the receipt and re-hash every artefact it enumerates.

    ``hasher(path)`` lets a caller route the reads through its own input binding, so the
    same artefact hashed for several runs is read once and a contradictory second read is
    refused where every other input is. ``repo`` is the checkout the producer closure's
    reviewed blobs are read from; a receipt is written and read in the same repository,
    and its commit stays readable there once it is merged.
    """
    repo = REPO if repo is None else repo
    path = Path(path)
    _require(path.is_file(), 'missing training receipt: {}'.format(path))
    digest = provenance.sha256_file(path)
    record = _read_json(path, 'training receipt')
    _require(isinstance(record, dict), 'the training receipt is not a record')
    _require(record.get('label') == LABEL,
             'the training receipt is labelled {!r}'.format(record.get('label')))
    _require(record.get('schema_version') == SCHEMA_VERSION,
             'unknown training receipt schema {!r}'.format(record.get('schema_version')))
    _require(record.get('experiment') == EXPERIMENT,
             'the training receipt is not {}\'s'.format(EXPERIMENT))
    files = record.get('files')
    _require(isinstance(files, list) and files, 'the training receipt enumerates nothing')
    _require(_digest(files) == record.get('files_sha256'),
             'the training receipt enumeration does not hash to its own files_sha256')
    checkpoint = record.get('checkpoint')
    _require(isinstance(checkpoint, dict) and isinstance(checkpoint.get('path'), str),
             'the training receipt names no checkpoint')
    root = Path(record.get('train_dir') or '')
    _require(root.is_dir(), 'the receipt\'s training directory {} is gone'.format(root))
    # R3: a receipt written from a drifting or --allow-dirty tree is diagnostic by its own
    # contract, and no reader may promote it -- and the closure must say so with records.
    check_producer(record.get('source_closure'), repo)
    # R3: and its enumeration is the writer's membership rule, applied to the directory now
    # -- not whatever list the receipt happens to carry.
    expected = receipt_names(root)
    names = [item.get('path') for item in files if isinstance(item, dict)]
    _require(names == expected, 'the training receipt enumerates {} artefacts, not the {} '
             '{} retains (missing {}; unexpected {})'.format(
                 len(names), len(expected), root,
                 ', '.join(name for name in expected if name not in names) or 'none',
                 ', '.join(str(name) for name in names if name not in expected) or 'none'))
    inputs = {str(path.resolve()): digest}
    for item in files:
        _require(isinstance(item, dict) and isinstance(item.get('path'), str),
                 'the training receipt enumerates a malformed entry')
        artefact = root / item['path']
        actual = hasher(str(artefact))
        _require(actual == item['sha256'],
                 'training artefact changed since the receipt: ' + item['path'])
        inputs[str(artefact.resolve())] = actual
    enumerated = {item['path']: item['sha256'] for item in files}
    _require(enumerated.get(checkpoint['path']) == checkpoint.get('sha256'),
             'the receipt checkpoint {} is not the enumerated artefact'.format(
                 checkpoint['path']))
    _require(record.get('epochs') == checkpoint.get('epoch'),
             'the receipt checkpoint is epoch {!r} of a {!r}-epoch run'.format(
                 checkpoint.get('epoch'), record.get('epochs')))
    _require(enumerated.get('args.json') == (record.get('args') or {}).get('sha256'),
             'the receipt args.json is not the enumerated artefact')
    # R3: the args/history/registered-arm contract the writer applied, re-derived from the
    # retained artefacts this verification has just re-hashed.
    args = _read_json(root / 'args.json', 'args.json')
    _require(isinstance(args, dict), 'args.json is not a record')
    history = history_epochs(root / 'history.jsonl')
    _require(args.get('epochs') == history == record.get('epochs'),
             'the retained args.json and history.jsonl record {} and {} epochs, not the {} '
             'of the receipt'.format(args.get('epochs'), history, record.get('epochs')))
    match = EPOCH_FILE.match(checkpoint['path'])
    _require(match is not None and int(match.group(1)) == history,
             'the receipt certifies {}, not the last of the {} epochs the history '
             'records'.format(checkpoint['path'], history))
    _require(args.get('backbone') == record.get('backbone'),
             'the retained args.json records the backbone {!r}, not the {!r} of the '
             'receipt'.format(args.get('backbone'), record.get('backbone')))
    registered = record.get('registered')
    _require(isinstance(registered, dict)
             and registered.get('sha256') == checkpoint.get('sha256')
             and registered.get('role') == record.get('arm')
             and registered.get('backbone') == record.get('backbone')
             and registered.get('epoch') == record.get('epochs'),
             'the training receipt calls these weights the registered {!r} arm, which is '
             'not the registration it records ({!r})'.format(record.get('arm'), registered))
    if checkpoint_sha256 is not None:
        _require(checkpoint['sha256'] == checkpoint_sha256, 'the training receipt certifies '
                 '{}, not the evaluated {}'.format(checkpoint['sha256'], checkpoint_sha256))
    if registered_sha256 is not None:
        _require(checkpoint['sha256'] == registered_sha256, 'the training receipt certifies '
                 '{}, not the registered exp_01 {}'.format(checkpoint['sha256'],
                                                           registered_sha256))
    if epochs is not None:
        _require(record['epochs'] == epochs,
                 'the training receipt records {} epochs, not {}'.format(record['epochs'],
                                                                         epochs))
    return dict(record, sha256=digest, path=str(path.resolve()), inputs=inputs)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Write the reconstructed training receipt of an exp_01 arm.')
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--allow-dirty', action='store_true',
                        help='record a closure that differs from HEAD (never confirmatory)')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    record, digest = write_receipt(args.out, args.train_dir, args.checkpoint,
                                   strict=not args.allow_dirty)
    print(json.dumps({'receipt': str(Path(args.out).resolve()), 'sha256': digest,
                      'arm': record['arm'], 'epochs': record['epochs'],
                      'checkpoint': record['checkpoint'], 'files': len(record['files'])},
                     sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        raise SystemExit('refusing: ' + str(error))
