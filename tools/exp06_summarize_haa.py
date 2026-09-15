"""exp_06's HAA summariser: two admission branches, one set of tables (plan v4 6.2-7).

Arms A (``control``) and B (``cyl``) are exp_02's historical runs. They carry no
``completion.json``, no heading and no closure record, so they are admitted through
exp_02's own completeness and pairing checks -- ``sim_to_real.summarize_haa`` is
imported and never edited -- plus an exp_06-owned **hash receipt** over every retained
artifact, written once by ``--write-legacy-receipt`` and labelled ``reconstructed``.
Nothing historical is ever rewritten.

Arms C (``cyl_or``), D (``control_hf``) and F (``cyl_hf``) are exp_06's own runs under
``ckpt/exp06/sim2real``. Each of their children carries the finalizer's
``completion.json``; this module re-reads it through ``tools.exp06_finalize``'s own
schema, rehashes every artifact it bound, and requires the arm's backbone, frame,
heading roll and execution closure to be the one the table registers.

    python tools/exp06_summarize_haa.py --json ckpt/exp06/stats.json \
        --summary ckpt/exp06/summary.txt
"""
import datetime
import hashlib
import json
import subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np

from sim_to_real import summarize_haa as legacy
from tools import provenance

ENTRY_MODULE = 'tools.exp06_summarize_haa'
REPO = Path(__file__).resolve().parents[1]
ROOMS = tuple(legacy.ROOMS)
METRICS = tuple(legacy.METRICS)                      # (key, label, decimals)
METRIC_KEYS = tuple(key for key, _, _ in METRICS)
REQUIRED_METRICS = {room: tuple(keys) for room, keys in legacy.REQUIRED_METRICS.items()}
CELLS = tuple((room, key) for room in ROOMS for key in REQUIRED_METRICS[room])

SEEDS = ('seed0', 'seed1', 'seed2')
ZEROSHOT = 'zeroshot'
JOBS = SEEDS + (ZEROSHOT,)
EXPECT_OF = dict({seed: 'finetune' for seed in SEEDS}, zeroshot='zeroshot')
HEADING_K = 128                     # every HAA room's heading roll, verified against the JSON
H1_MARGIN_DB = 0.23                 # frozen in section 7; never computed from the data
H1_ROOM, H1_METRIC = 'hallway', 'c50'
ALPHA = 0.05
H2_FAMILY = len(CELLS)
N_BOOT = 10000
N_BOOT_ADJUSTED = 50000
BOOT_SEEDS = (0, 1)
CONVERGENCE_TOL = 0.10
VOID_EXCESS_FRACTION = 0.01         # excess invalid queries, as a share of the test split
VOID_COHORT_FRACTION = 0.99

ARMS = OrderedDict([
    ('control', {'label': 'A', 'branch': 'legacy', 'root': 'ckpt/sim2real/control',
                 'backbone': 'simple', 'frame': 'room', 'legacy_init': 'control'}),
    ('cyl', {'label': 'B', 'branch': 'legacy', 'root': 'ckpt/sim2real/cyl',
             'backbone': 'cylindrical', 'frame': 'room', 'legacy_init': 'cyl'}),
    ('cyl_or', {'label': 'C', 'branch': 'new', 'root': 'ckpt/exp06/sim2real/cyl_or',
                'backbone': 'cylindrical_oriented', 'frame': 'heading',
                'init_approval': 'artifacts.epoch_012.sha256'}),
    ('control_hf', {'label': 'D', 'branch': 'new', 'root': 'ckpt/exp06/sim2real/control_hf',
                    'backbone': 'simple', 'frame': 'heading',
                    'init_checkpoint': 'ckpt/xRIR_simple_8_shot/epoch_12.pth'}),
    ('cyl_hf', {'label': 'F', 'branch': 'new', 'root': 'ckpt/exp06/sim2real/cyl_hf',
                'backbone': 'cylindrical', 'frame': 'heading',
                'init_checkpoint': 'ckpt/xRIR_cyl_8_shot/epoch_12.pth'})])
LEGACY_ARMS = tuple(name for name, arm in ARMS.items() if arm['branch'] == 'legacy')
NEW_ARMS = tuple(name for name, arm in ARMS.items() if arm['branch'] == 'new')
LEGACY_ROOT = 'ckpt/sim2real'
NEW_ROOT = 'ckpt/exp06/sim2real'
CANONICAL = ('stats.json', 'summary.txt')            # exp_02's hash-bound record

# The contrasts of section 7. H1 and H1b are decision bearing; the rest describe.
H1 = ('cyl_or', 'control')
H1B = ('cyl_or', 'cyl_hf')
DESCRIPTIVE = (('cyl_or', 'control_hf'), ('control_hf', 'control'), ('cyl_hf', 'cyl'))


def _require(ok, cause):
    if not ok:
        raise ValueError(cause)


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError('unreadable {}: {}'.format(label, error))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


# --- the legacy branch: a hash receipt over exp_02's retained artifacts -------------------


def legacy_artifacts():
    """Every retained artifact of one legacy arm, relative to that arm's directory."""
    names = []
    for seed in SEEDS:
        for stage in ('stage1',) + tuple('stage2_' + room for room in ROOMS):
            names += ['{}/{}/{}'.format(seed, stage, name)
                      for name in ('args.json', 'summary.json')]
        for room in ROOMS:
            names += ['{}/eval/metrics_{}.json'.format(seed, room),
                      '{}/eval/per_sample_{}.json'.format(seed, room)]
    for room in ROOMS:
        names += ['{}/metrics_{}.json'.format(ZEROSHOT, room),
                  '{}/per_sample_{}.json'.format(ZEROSHOT, room)]
    return tuple(names)


def legacy_receipt_files(root, arms=LEGACY_ARMS):
    """The receipt's ordered enumeration: both arms, then exp_02's canonical record."""
    root, files = Path(root), []
    for arm in arms:
        for name in legacy_artifacts():
            relative = '{}/{}'.format(Path(ARMS[arm]['root']).name, name)
            path = root / relative
            _require(path.is_file(), 'the legacy arm {} has no {}'.format(arm, relative))
            files.append({'path': relative, 'sha256': provenance.sha256_file(path)})
    for name in CANONICAL:
        path = root / name
        _require(path.is_file(), 'the legacy root has no canonical ' + name)
        files.append({'path': name, 'sha256': provenance.sha256_file(path)})
    return files


def source_identity(repo=REPO, strict=True):
    """This producer's own closure, bound to HEAD; strict refuses an edited working tree."""
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo),
                                   text=True).strip()
    records, digest = provenance.closure_record(
        provenance.source_closure(ENTRY_MODULE, repo), head, repo)
    drift = [record['path'] for record in records
             if record['reviewed_blob_sha256'] is None
             or record['reviewed_blob_sha256'] != record['working_tree_sha256']]
    _require(not (strict and drift), 'the producer closure differs from HEAD: '
             + ', '.join(sorted(drift)))
    return {'entry_module': ENTRY_MODULE, 'commit': head, 'sha256': digest,
            'files': records, 'drift': sorted(drift)}


def legacy_receipt(root, arms=LEGACY_ARMS, repo=REPO, strict=True):
    """The reconstructed receipt: what the historical arms are, byte for byte, today."""
    files = legacy_receipt_files(root, arms)
    return {'schema_version': 1, 'label': 'reconstructed', 'arms': list(arms),
            'root': str(Path(root).resolve()), 'files': files, 'files_sha256': _digest(files),
            'source_closure': source_identity(repo, strict),
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()}


def write_legacy_receipt(path, root, arms=LEGACY_ARMS, repo=REPO, strict=True):
    """Write once; a receipt is never silently replaced."""
    record = legacy_receipt(root, arms, repo, strict)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return record, provenance.write_manifest(path, record)


def verify_legacy_receipt(path, root, approved=None):
    """The receipt is the approved bytes, and every artifact still hashes as enumerated."""
    path = Path(path)
    _require(path.is_file(), 'missing legacy receipt: {}'.format(path))
    digest = provenance.sha256_file(path)
    if approved is not None:
        _require(approved.get('sha256') == digest, 'the legacy receipt {} hashes to {}, not '
                 'the approved {}'.format(path, digest, approved.get('sha256')))
        _require(approved.get('path') and Path(approved['path']).name == path.name,
                 'the legacy receipt path {} is not the approved {}'.format(
                     path, approved.get('path')))
    record = _read_json(path, 'legacy receipt')
    _require(record.get('label') == 'reconstructed',
             'the legacy receipt is labelled {!r}'.format(record.get('label')))
    _require(record.get('schema_version') == 1, 'unknown legacy receipt schema')
    files = record.get('files')
    _require(isinstance(files, list) and files, 'the legacy receipt enumerates nothing')
    _require(_digest(files) == record.get('files_sha256'),
             'the legacy receipt enumeration does not hash to its own files_sha256')
    expected = [item['path'] for item in legacy_receipt_files(root, tuple(record['arms']))]
    _require([item['path'] for item in files] == expected,
             'the legacy receipt does not enumerate exactly the retained artifacts')
    for item in files:
        actual = provenance.sha256_file(Path(root) / item['path'])
        _require(actual == item['sha256'],
                 'legacy artifact changed since the receipt: ' + item['path'])
    return dict(record, sha256=digest)
