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

Arm E (``yawaug``) is exp_09's: exp_04's yaw-augmented SimpleViT, fine-tuned by the same
pipeline in the ROOM frame under ``ckpt/exp09/sim2real``. ``--experiment exp09`` publishes
that experiment's frozen configuration -- arms A, B, C and E, the contrasts E1/E2/E3 and
exp_09's own outputs -- through this one producer, so the analysis stays inside the
``code.summarize_haa`` closure 6.4 approves. exp_06's arms, contrasts, verdict semantics
and canonical record are untouched by it, and neither experiment may write the other's.

    python tools/exp06_summarize_haa.py --json ckpt/exp06/stats.json \
        --summary ckpt/exp06/summary.txt
    python tools/exp06_summarize_haa.py --experiment exp09 \
        --json ckpt/exp09/stats.json --summary ckpt/exp09/summary.txt
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

from sim_to_real import summarize_haa as legacy
from tools.exp04_profiles import CONTROL as EXP01_CONTROL, CYL as EXP01_CYL
from tools import exp06_approvals_api as approvals_api
from tools import exp06_bootstrap as bootstrap
from tools import exp06_finalize as finalizer
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
                'experiment': 'exp06', 'init_sha256': None}),   # the approved epoch_012
    ('control_hf', {'label': 'D', 'branch': 'new', 'root': 'ckpt/exp06/sim2real/control_hf',
                    'backbone': 'simple', 'frame': 'heading', 'experiment': 'exp06',
                    'init_sha256': EXP01_CONTROL['sha256']}),
    ('cyl_hf', {'label': 'F', 'branch': 'new', 'root': 'ckpt/exp06/sim2real/cyl_hf',
                'backbone': 'cylindrical', 'frame': 'heading', 'experiment': 'exp06',
                'init_sha256': EXP01_CYL['sha256']}),
    # exp_09's arm E: exp_04's yaw-augmented SimpleViT, fine-tuned in the ROOM frame, in
    # its own tree. Its initialisation is exp_04's approved checkpoints.aug, resolved by
    # expected_inits through 6.4's reused pin -- never exp_06's artifacts.epoch_012.
    ('yawaug', {'label': 'E', 'branch': 'new', 'root': 'ckpt/exp09/sim2real/yawaug',
                'backbone': 'simple', 'frame': 'room', 'experiment': 'exp09',
                'init_sha256': None}),
    # exp_11's arm G: the same approved exp_04 checkpoint as E, fine-tuned and evaluated
    # in exp_06's HEADING frame (D's frame), under exp_11's own tree. Its initialisation
    # is resolved by expected_inits exactly as E's, never by a literal here.
    ('yawaug_hf', {'label': 'G', 'branch': 'new', 'root': 'ckpt/exp11/sim2real/yawaug_hf',
                   'backbone': 'simple', 'frame': 'heading', 'experiment': 'exp11',
                   'init_sha256': None})])
LEGACY_ARMS = tuple(name for name, arm in ARMS.items() if arm['branch'] == 'legacy')
NEW_ARMS = tuple(name for name, arm in ARMS.items() if arm['branch'] == 'new')
LEGACY_ROOT = 'ckpt/sim2real'
NEW_ROOT = 'ckpt/exp06/sim2real'
EXP09_ROOT = 'ckpt/exp09/sim2real'
EXP11_ROOT = 'ckpt/exp11/sim2real'
CANONICAL = ('stats.json', 'summary.txt')            # exp_02's hash-bound record
EXP04_AUG_ARMS = ('yawaug', 'yawaug_hf')   # the arms exp_04's approved checkpoint starts

# The contrasts of section 7. H1 and H1b are decision bearing; the rest describe.
H1 = ('cyl_or', 'control')
H1B = ('cyl_or', 'cyl_hf')
DESCRIPTIVE = (('cyl_or', 'control_hf'), ('control_hf', 'control'), ('cyl_hf', 'cyl'))
# exp_09 section 3: E1 is the headline (two-sided, plus the exp_06 margin statement), E2
# the eleven-cell screen against the same comparator, E3 descriptive against arm C.
E1 = ('yawaug', 'control')
E3 = (('yawaug', 'cyl_or'),)

# --- R1: historical rows, copied from the hash-bound canonical records -------------------

EXP06_STATS = 'ckpt/exp06/stats.json'
EXP09_STATS = 'ckpt/exp09/stats.json'
# The universe of decision-bearing fields a copied row may carry. A source records some
# of them; the rest are null here and named in ``not_recorded``.
HISTORICAL_FIELDS = ('diff', 'two_way', 'nominal_two_way', 'convergence', 'verdict',
                     'category', 'non_inferior_at_margin')
DECISION_ROW = ('diff', 'two_way', 'convergence', 'verdict')
DESCRIPTIVE_ROW = ('diff', 'nominal_two_way')
# Plan section 3's source-field map: which record, which table, which row, which fields.
EXP11_HISTORICAL = (
    {'name': 'C - D', 'source': EXP06_STATS, 'table': 'D', 'kind': 'descriptive',
     'contrast': 'cyl_or - control_hf', 'fields': DESCRIPTIVE_ROW},
    {'name': 'D - A', 'source': EXP06_STATS, 'table': 'D', 'kind': 'descriptive',
     'contrast': 'control_hf - control', 'fields': DESCRIPTIVE_ROW},
    {'name': 'C - A', 'source': EXP06_STATS, 'table': 'H1', 'kind': 'decision',
     'contrast': 'cyl_or - control', 'fields': DECISION_ROW},
    {'name': 'C - F', 'source': EXP06_STATS, 'table': 'H1b', 'kind': 'decision',
     'contrast': 'cyl_or - cyl_hf', 'fields': DECISION_ROW},
    {'name': 'E - A', 'source': EXP09_STATS, 'table': 'E1', 'kind': 'decision',
     'contrast': 'yawaug - control',
     'fields': DECISION_ROW + ('category', 'non_inferior_at_margin')})


# exp_11 section 3. N1 = G - D, N1i = the interaction (G - E) - (D - A), N2 = C - G and
# N3 = G - E, each publishing exactly the statements of the notation block that it
# registers -- never a margin verdict, and never a field the plan did not ask of it.
N1 = ('yawaug_hf', 'control_hf')
N1I = ('yawaug_hf', 'yawaug', 'control_hf', 'control')
N2 = ('cyl_or', 'yawaug_hf')
N3 = ('yawaug_hf', 'yawaug')
EXP11_ARMS = ('control', 'cyl', 'cyl_or', 'control_hf', 'cyl_hf', 'yawaug', 'yawaug_hf')
EXP11_DECISIONS = (
    {'name': 'N1', 'kind': 'contrast', 'pair': N1, 'room': H1_ROOM, 'metric': H1_METRIC,
     'margin': None, 'fields': ('category',),
     'reading': 'does roll-robust pretraining change what the SimpleViT does with the '
                'heading frame?'},
    {'name': 'N1i', 'kind': 'interaction', 'arms': N1I, 'room': H1_ROOM,
     'metric': H1_METRIC, 'margin': None, 'fields': ('category',),
     'reading': 'positive = a larger heading-frame penalty after yaw pretraining, not '
                'harm by any single arm'},
    {'name': 'N2', 'kind': 'contrast', 'pair': N2, 'room': H1_ROOM, 'metric': H1_METRIC,
     'margin': H1_MARGIN_DB,
     'fields': ('category', 'y_non_inferior_at_margin', 'x_margin_advantage'),
     'reading': "G's non-inferiority to C and C's margin-sized advantage are separate "
                'statements; failing to establish one is not evidence of its negation'},
    {'name': 'N3', 'kind': 'contrast', 'pair': N3, 'room': H1_ROOM, 'metric': H1_METRIC,
     'margin': H1_MARGIN_DB, 'fields': ('category', 'equivalent_at_margin'),
     'reading': 'the prediction G = E is the equivalence statement, not the category'})
EXP11_SCREENS = (N1, N2, N3)        # three declared families; no claim across them

# One frozen configuration per experiment: which arms are loaded, which contrasts are
# computed under which names, which of them carry exp_09's two-sided classification, and
# the canonical outputs no other experiment's run may write. Nothing here is mutated at
# runtime: an exp_09 run never changes what an exp_06 run publishes.
EXPERIMENTS = OrderedDict([
    ('exp06', {'arms': ('control', 'cyl', 'cyl_or', 'control_hf', 'cyl_hf'),
               'decisions': (('H1', H1, H1_ROOM, H1_METRIC, H1_MARGIN_DB),
                             ('H1b', H1B, H1_ROOM, H1_METRIC, 0.0)),
               'screen': ('H2', H1), 'descriptive': ('D', DESCRIPTIVE), 'classified': (),
               'outputs': ('ckpt/exp06/stats.json', 'ckpt/exp06/summary.txt')}),
    ('exp09', {'arms': ('control', 'cyl', 'cyl_or', 'yawaug'),
               'decisions': (('E1', E1, H1_ROOM, H1_METRIC, H1_MARGIN_DB),),
               'screen': ('E2', E1), 'descriptive': ('E3', E3), 'classified': ('E1',),
               'outputs': ('ckpt/exp09/stats.json', 'ckpt/exp09/summary.txt')}),
    # exp_11 phase 1. A ``phase`` selects section 3's statements instead of exp_06's
    # margin verdicts, so ``decisions``/``classified`` stay empty here: the historical
    # tables are built by the historical path and are untouched by this registration.
    ('exp11', {'arms': EXP11_ARMS, 'phase': 'phase1', 'decisions': (), 'classified': (),
               'exp11_decisions': EXP11_DECISIONS, 'screens': EXP11_SCREENS,
               'historical': EXP11_HISTORICAL,
               'outputs': ('ckpt/exp11/phase1/stats.json',
                           'ckpt/exp11/phase1/summary.txt')})])


def _require(ok, cause):
    if not ok:
        raise ValueError(cause)


def bind(inputs, path, digest=None):
    """Finding 2: record the bytes of one path, and refuse a contradictory second one.

    Every reader binds what it actually read. Two readers of one path must agree, so a
    file that changed between two reads of an analysis is a contradiction here rather
    than a silent overwrite that publication would never notice.
    """
    path = str(Path(path).resolve())
    digest = provenance.sha256_file(path) if digest is None else digest
    recorded = inputs.setdefault(path, digest)
    _require(recorded == digest, 'contradictory bindings for {}: {} and {}'.format(
        path, recorded, digest))
    return digest


def _read_json(path, label, inputs=None, expected=None):
    """Parse one JSON file, binding the exact bytes that were parsed.

    Finding 2b: when the caller already knows the identity a parent record certified for
    this path, it passes it as ``expected`` and these bytes must be those bytes -- a file
    that changed between the certification and this read is refused here rather than
    rebound, so a validated identity is never replaced by a fresher one.
    """
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise ValueError('unreadable {}: {}'.format(label, error))
    digest = hashlib.sha256(raw).hexdigest()
    _require(expected is None or digest == expected, '{} is not the bytes bound for it: '
             '{} is not {}'.format(label, digest, expected))
    if inputs is not None:
        bind(inputs, path, digest)
    try:
        return json.loads(raw)
    except ValueError as error:
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


def legacy_receipt(root, repo=REPO, strict=True, identity=None):
    """The reconstructed receipt: what the historical arms are, byte for byte, today.

    Finding 6: the enumeration is derived from ``LEGACY_ARMS``, never from a caller's or
    a receipt's own list, so a receipt that omits an arm cannot be written -- nor, in
    ``verify_legacy_receipt``, accepted for a ``load_legacy`` that returns both arms.
    """
    files = legacy_receipt_files(root, LEGACY_ARMS)
    return {'schema_version': 1, 'label': 'reconstructed', 'arms': list(LEGACY_ARMS),
            'root': str(Path(root).resolve()), 'files': files, 'files_sha256': _digest(files),
            'source_closure': source_identity(repo, strict)
                              if identity is None else identity,
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()}


def write_legacy_receipt(path, root, repo=REPO, strict=True, identity=None):
    """Write once; a receipt is never silently replaced."""
    record = legacy_receipt(root, repo, strict, identity)
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
    _require(record.get('arms') == list(LEGACY_ARMS),
             'the legacy receipt declares the arms {!r}, not the registered {}'.format(
                 record.get('arms'), list(LEGACY_ARMS)))
    expected = [item['path'] for item in legacy_receipt_files(root, LEGACY_ARMS)]
    _require([item['path'] for item in files] == expected,
             'the legacy receipt does not enumerate exactly the retained artifacts')
    inputs = {str(path.resolve()): digest}
    for item in files:
        file = Path(root) / item['path']
        actual = provenance.sha256_file(file)
        _require(actual == item['sha256'],
                 'legacy artifact changed since the receipt: ' + item['path'])
        inputs[str(file.resolve())] = actual   # finding 10: hashed as it was consumed
    return dict(record, sha256=digest, inputs=inputs)


# --- the legacy branch: exp_02's own completeness gate, then this experiment's two arms --


def load_legacy(root, receipt_path=None, approved=None):
    """exp_02's loader and completeness run unchanged over the COMPLETE historical root.

    The inherited checker expects exp_02's full run set (``released``,
    ``released_repomaps``, ``control``, ``cyl``), so it is given the whole root before
    arms A and B are extracted from it. The root is given exactly as exp_02 gave it --
    repo-relative ``ckpt/sim2real`` -- because that checker compares each stage-2 ``init``
    string against ``<root>/stage1/best.pth``.
    """
    # Finding 2: the receipt is verified first, so a changed historical artifact is
    # refused before any of them is read into the tables.
    receipt = None if receipt_path is None else verify_legacy_receipt(
        receipt_path, root, approved)
    runs = legacy.load_runs(str(root))
    complete, problems = legacy.completeness(runs)
    _require(complete, 'the historical exp_02 root is incomplete: ' + '; '.join(problems))
    data = {}
    for arm in LEGACY_ARMS:
        init, jobs = ARMS[arm]['legacy_init'], {}
        for kind in ('fine-tuned', 'zero-shot'):
            for run in runs.get((init, kind), []):
                jobs[Path(run['dir']).name] = run['per']
        _require(set(jobs) == set(JOBS), 'the legacy arm {} has the jobs {}, not {}'.format(
            arm, sorted(jobs), sorted(JOBS)))
        for job, rooms in sorted(jobs.items()):
            _require(set(rooms) == set(ROOMS),
                     'the legacy arm {} job {} has the rooms {}'.format(arm, job, sorted(rooms)))
        data[arm] = {'arm': arm, 'branch': 'legacy', 'per': jobs, 'closure': None,
                     'root': str(Path(root) / Path(ARMS[arm]['root']).name)}
    return data, receipt


# --- the new branch: the finalizer's own records, re-read and rehashed -------------------

# Amendment A3, reconciled with the merged finalizer -- which is authoritative. A job
# has no child process of its own, so its completion carries no closed-log marker and no
# exit receipt; it binds the job specification, the launcher pid that owned the job, and
# the verified completion of every expected child, under the finalizer's own field names.
JOB_FIELDS = ('schema_version', 'run_type', 'run_dir', 'repo', 'child_exit', 'log',
              'owner_pid', 'diagnostic', 'admissible_arm', 'expect', 'children',
              'job_spec', 'artifacts', 'backbone', 'frame', 'seed', 'heading', 'init',
              'init_sha256')
FORBIDDEN_JOB_FIELDS = ('child_exit_time', 'child_exit_receipt')


def _is_sha256(value):
    return type(value) is str and len(value) == 64 and all(
        character in '0123456789abcdef' for character in value)


def job_completion(job_dir, expect, arm, inputs=None):
    """One job's A3 record: no receipt, a bound job spec, an owner and its children."""
    job_dir = Path(job_dir)
    path = job_dir / 'completion.json'
    _require(path.is_file(), 'job {} has no completion.json'.format(job_dir))
    _require(not (job_dir / 'child_exit.json').exists(),
             'amendment A3: the job root {} carries a child_exit.json'.format(job_dir))
    record = _read_json(path, 'job completion.json', inputs)
    present = [key for key in FORBIDDEN_JOB_FIELDS if key in record]
    _require(not present,
             'amendment A3: a job completion carries no ' + ', '.join(present))
    missing = [key for key in JOB_FIELDS if key not in record]
    _require(not missing, 'job {} completion is incomplete: missing {}'.format(
        job_dir, ', '.join(missing)))
    _require(record['schema_version'] == 1 and record['run_type'] == 'haa_job',
             'job {} records {!r}/{!r}'.format(job_dir, record['schema_version'],
                                               record['run_type']))
    _require(Path(record['run_dir']).resolve() == job_dir.resolve(),
             'job {} claims the run_dir {}'.format(job_dir, record['run_dir']))
    _require(record['expect'] == expect,
             'job {} expects {!r}, not {!r}'.format(job_dir, record['expect'], expect))
    _require(record['diagnostic'] is False and record['admissible_arm'] is True,
             'job {} is not an admissible arm'.format(job_dir))
    _require(record['child_exit'] == 0,
             'job {} was declared with child status {!r}'.format(job_dir, record['child_exit']))
    _require(type(record['owner_pid']) is int and record['owner_pid'] > 0,
             'job {} records no owning launch.pid'.format(job_dir))
    spec = record['job_spec']
    _require(isinstance(spec, dict) and isinstance(spec.get('path'), str) and spec['path']
             and _is_sha256(spec.get('sha256')),
             'job {} binds no job specification'.format(job_dir))
    expected = set(finalizer.expected_children(expect))
    _require(isinstance(record['children'], dict) and set(record['children']) == expected,
             'job {} records the children {}, not {}'.format(
                 job_dir, sorted(record['children'] or ()), sorted(expected)))
    for field in ('backbone', 'frame'):
        _require(record[field] == ARMS[arm][field], 'job {} records {} {!r}, not the {!r} of '
                 'arm {}'.format(job_dir, field, record[field], ARMS[arm][field], arm))
    return record


def job_owner(job_dir, record, inputs=None):
    """Finding 6: amendment A3 binds the launcher that owned the job, so the job root's
    own ``launch.pid`` is read and required to be the ``owner_pid`` the completion
    recorded. Liveness is deliberately not checked: a retrospective analysis runs long
    after the launcher exited and a reused pid would prove nothing either way.

    Close review 2, finding 6: the marker is read **once**. The pid is parsed from that
    buffer and the digest is taken of that same buffer, so the ownership evidence the
    record publishes is the bytes the admitted pid came from and never a second, later
    snapshot of a file that changed in between.
    """
    path = Path(job_dir) / 'launch.pid'
    _require(path.is_file(), 'job {} has no launch.pid: amendment A3 binds the launcher '
             'that owned it'.format(job_dir))
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError('job {} has an unreadable launch.pid: {}'.format(job_dir, error))
    digest = hashlib.sha256(raw).hexdigest()
    try:
        pid = int(raw.decode('utf-8', 'replace').split()[0])
    except (IndexError, ValueError) as error:
        raise ValueError('job {} has an unreadable launch.pid: {}'.format(job_dir, error))
    _require(pid == record['owner_pid'], 'job {} holds launch.pid {}, not the owner_pid {} '
             'its completion bound'.format(job_dir, pid, record['owner_pid']))
    if inputs is not None:
        bind(inputs, path, digest)
    return {'path': str(path.resolve()), 'pid': pid, 'sha256': digest}


def _heading_rolls(heading, label):
    """Every bound room rolls by the registered column and names the JSON it came from."""
    _require(isinstance(heading, dict) and heading, '{} binds no heading'.format(label))
    for room in sorted(heading):
        entry = heading[room]
        _require(isinstance(entry, dict) and entry.get('k') == HEADING_K,
                 '{} rolls {} by {!r}, not the registered {}'.format(
                     label, room, (entry or {}).get('k'), HEADING_K))
        _require(_is_sha256(entry.get('sha256')) and isinstance(entry.get('path'), str)
                 and entry['path'], '{} records no json identity for {}: the finalizer '
                 'binds the heading record a child read'.format(label, room))
    return {room: heading[room]['k'] for room in heading}


PROTOCOL = dict(legacy.PROTOCOL)      # K = 8, eval_seed 0, the DiffRIR test split
# Finding 3: plan 6.2's registered recipe. The finalizer checks that a history completes
# the budget the run *declared*; only this table says what the experiment requires, so a
# shortened training, a changed selection population or an S1 seeded-phase evaluation
# cannot enter the primary comparison against exp_02's historical arms. They are
# admissible only under ``--sensitivity``, which labels every output it produces.
S1_ROOMS = ('class_room', 'complex_room', 'hallway')   # dampened excluded, as in exp_02
STAGES = {'stage1': {'epochs': 1000, 'val_every': 10},
          'stage2': {'epochs': 200, 'val_every': 2}}
TRAIN_RECIPE = {'lr': 1e-4, 'weight_decay': 1e-4, 'decay_epochs': 50, 'lr_gamma': 0.1,
                'batch_size': 0, 'accum_steps': 1, 'tf32': True, 'max_len': 9600,
                'depth_variant': 'default', 'num_shot': 8, 'eval_seed': 0}
# The primary phase policy is unseeded Griffin-Lim, exactly as exp_02 evaluated.
EVAL_RECIPE = {'split': 'test', 'num_shot': 8, 'eval_seed': 0, 'max_len': 9600,
               'depth_variant': 'default', 'max_samples': 0, 'gl_seed_per_query': False,
               'tag': ''}
# Finding 1: the finalizer records the executed entry point's closure, and the two entry
# points are different modules with different closures. Consistency is per role, and the
# approved identity each role must match is its own.
ROLE_CODE_KEY = {'haa_train': 'haa_finetune', 'haa_eval': 'haa_eval'}


def arm_closures(children):
    """{role: digest}: every child of one entry point ran the same reviewed closure."""
    closures = {}
    for name in sorted(children):
        digest = children[name]['source_closure_sha256']
        _require(_is_sha256(digest), 'child {} records no execution closure'.format(name))
        closures.setdefault(children[name]['role'], set()).add(digest)
    for role in sorted(closures):
        _require(len(closures[role]) == 1, 'the {} children of this arm do not share one '
                 'execution closure: {}'.format(role, sorted(closures[role])))
    return {role: sorted(digests)[0] for role, digests in sorted(closures.items())}


def arm_headings(children):
    """{room: heading json sha256}: one heading record per room, across every seed."""
    headings = {}
    for name in sorted(children):
        for room, binding in sorted((children[name]['heading'] or {}).items()):
            _require(isinstance(binding, dict) and binding.get('k') == HEADING_K,
                     'child {} rolls {} by {!r}, not the registered {}'.format(
                         name, room, (binding or {}).get('k'), HEADING_K))
            _require(_is_sha256(binding.get('sha256')) and binding.get('path'),
                     'child {} records no json identity for {}'.format(name, room))
            recorded = headings.setdefault(room, binding['sha256'])
            _require(recorded == binding['sha256'], 'the arm binds two heading records for '
                     '{}: {} and {}'.format(room, recorded, binding['sha256']))
    return headings


def check_arm_identities(arm, closures, headings, approved):
    """Finding 3: what really ran, against what section 6.4 approved -- not merely null."""
    if approved is None:
        return
    code = approved['code']
    for role in sorted(closures):
        key = ROLE_CODE_KEY[role]
        _require(code.get(key) == closures[role], 'the {} children of {} ran the closure {}, '
                 'not the approved code.{} {}'.format(role, arm, closures[role], key,
                                                      code.get(key)))
    pinned = approved['artifacts']['heading']
    for room in sorted(headings):
        _require(pinned.get(room) == headings[room], 'the arm {} read the {} heading record '
                 '{}, not the approved artifacts.heading.{} {}'.format(
                     arm, room, headings[room], room, pinned.get(room)))


def child_protocol(args, name, role):
    """The protocol exp_02 froze, read from the arguments the child really ran."""
    for field in sorted(PROTOCOL):
        if role == 'haa_train' and field == 'split':
            continue                       # a fine-tuning child trains, it does not evaluate
        _require(finalizer.exp06_recipe.strict_equal(args.get(field), PROTOCOL[field]),
                 'child {} records {} {!r}, not the registered {!r}'.format(
                     name, field, args.get(field), PROTOCOL[field]))


def child_recipe(args, name, role):
    """Plan 6.2's recipe, read from the arguments the child really ran.

    Returns the deviations rather than raising: a confirmatory admission refuses them
    all, and a ``--sensitivity`` analysis reports them beside the numbers they produced.
    """
    deviations = []

    def check(field, expected):
        if not finalizer.exp06_recipe.strict_equal(args.get(field), expected):
            deviations.append('child {} records {} {!r}, not the registered {!r}'.format(
                name, field, args.get(field), expected))

    if role == 'haa_train':
        stage = 'stage1' if name == 'stage1' else 'stage2'
        for field, value in sorted(dict(TRAIN_RECIPE, **STAGES[stage]).items()):
            check(field, value)
        rooms = sorted(args.get('rooms') or [])
        expected = (sorted(S1_ROOMS) if stage == 'stage1'
                    else [name[len('stage2_'):]])
        if rooms != expected:
            deviations.append('child {} trains on the rooms {}, not the registered '
                              '{}'.format(name, rooms, expected))
        # Close review 2, finding 3: the validation loss selects ``best.pth``, and
        # ``best.pth`` initialises the next stage, so exp_02's selection population is
        # part of the recipe. The wrapper defaults ``--val-rooms`` to ``--rooms``.
        validation = sorted(args.get('val_rooms') or args.get('rooms') or [])
        if validation != expected:
            deviations.append('child {} selects best.pth on the validation rooms {}, not '
                              'the registered {}'.format(name, validation, expected))
        return deviations
    for field, value in sorted(EVAL_RECIPE.items()):
        check(field, value)
    room = name.rsplit('/', 1)[-1]
    if args.get('rooms') != [room]:
        deviations.append('child {} evaluates the rooms {!r}, not the registered '
                          '[{!r}]'.format(name, args.get('rooms'), room))
    return deviations


def check_test_indices(name, room, index):
    """Exactly the DiffRIR test split of the room, in the order exp_02 registered."""
    expected = legacy._test_indices(room)
    _require([int(value) for value in index] == expected,
             'child {} evaluated {} queries of {}, not exactly the {} DiffRIR test indices '
             'in order'.format(name, len(index), room, len(expected)))


def expected_inits(approved, arms=NEW_ARMS, inputs=None):
    """What each new arm must have started from: exp_01's weights, exp_06's approved
    epoch, or -- for arms E and G -- exp_04's approved ``checkpoints.aug``.

    Their identity is resolved through 6.4's ``reused`` exp_04 pin rather than a literal,
    so only the approvals a reviewer committed can name it; the record it was read from is
    bound into ``inputs`` and published with the analysis. It is resolved only when one of
    them is among the arms being loaded, so an exp_06 run depends on nothing further, and
    a registered arm with a null ``init_sha256`` is never left unchecked.
    """
    inits = {arm: ARMS[arm]['init_sha256'] for arm in arms
             if ARMS[arm]['branch'] == 'new'}   # an experiment's arms include the legacy two
    if not approved:
        return inits
    if 'cyl_or' in inits:
        inits['cyl_or'] = approved.get('artifacts', {}).get('epoch_012', {}).get('sha256')
    augmented = [arm for arm in EXP04_AUG_ARMS if arm in inits]
    if augmented:      # exp_09's E and exp_11's G start from the one approved checkpoint
        record = finalizer.exp04_aug_checkpoint(approved)
        for arm in augmented:
            inits[arm] = record['checkpoint']['sha256']
        if inputs is not None:
            bind(inputs, record['path'], record['sha256'])
    return inits


def child_per_sample(job_dir, name, record, arm, inputs=None):
    """One evaluation child's per-sample file, with the fields the summariser reads."""
    path = Path(job_dir) / name
    names = sorted(key for key in record['artifacts'] if key.startswith('per_sample_'))
    _require(len(names) == 1, 'child {} bound {} per-sample files'.format(name, len(names)))
    per = _read_json(path / names[0], names[0], inputs)
    meta = per.get('meta')
    _require(isinstance(meta, dict), 'child {} per-sample records no meta'.format(name))
    _require('heading' in meta, 'child {} per-sample meta records no heading'.format(name))
    _require(meta.get('frame') == ARMS[arm]['frame'], 'child {} per-sample meta records the '
             'frame {!r}, not the {!r} of arm {}'.format(name, meta.get('frame'),
                                                         ARMS[arm]['frame'], arm))
    if ARMS[arm]['frame'] == 'heading':
        _heading_rolls(meta['heading'], 'child {} per-sample meta'.format(name))
    else:      # a room-frame arm reads no heading, and the writer records that null
        _require(not meta['heading'],
                 'child {} per-sample meta records a heading in the room frame'.format(name))
    side = per.get('side_label')
    _require(isinstance(side, list) and len(side) == len(per.get('index', [])),
             'child {} per-sample records no room-frame side_label'.format(name))
    _require(all(value in (-1, 1) and not isinstance(value, bool) for value in side),
             'child {} side labels are not the room-frame signs'.format(name))
    return per


def _dependency(inputs, path, digest, label):
    """One validated dependency, refused rather than bound without an identity."""
    _require(_is_sha256(digest),
             '{} records no validated digest for {}'.format(label, path))
    bind(inputs, path, digest)


def child_dependencies(inputs, path, evidence, repo):
    """Finding 2a: the inputs the child's provenance declares and its validators verified.

    ``verify_child`` re-runs the child's own role validator, which rehashes the data
    inventory (the HAA cache files it read), every file of its source closure and every
    mutable input it recorded. Those bytes are what the certification rests on, so they
    are retained here with the digests that validation confirmed -- publication
    revalidates the whole map, and a second, different digest for any of these paths is
    a contradiction rather than an update.
    """
    label = 'child {}'.format(Path(path).name)
    record = _read_json(Path(path) / 'provenance.json', label + '/provenance.json', inputs,
                        (evidence.get('artifacts') or {}).get('provenance.json'))
    for key in ('data_identity', 'train_data_identity'):
        identity = record.get(key)
        if not isinstance(identity, dict):
            continue
        root = Path(identity['data_root'])
        for entry in identity.get('inventory') or ():
            _dependency(inputs, root / entry['path'], entry.get('sha256'), label)
    for name in sorted(record.get('source_closures') or {}):
        for item in (record['source_closures'][name] or {}).get('files') or ():
            _dependency(inputs, finalizer._resolve(item['path'], repo),
                        item.get('working_tree_sha256'), label)
    for name in sorted(record.get('mutable_inputs') or {}):
        entry = record['mutable_inputs'][name]
        _dependency(inputs, finalizer._resolve(entry['path'], repo), entry.get('sha256'),
                    label)


def bind_child(inputs, path, evidence, bound, args, repo):
    """Finding 2: every file one child's certification rests on, with the bytes the
    finalizer's own validators just re-hashed -- artefacts, log, exit receipt, the
    heading records it read, the weights it started from or evaluated, and (finding 2a)
    the data, source and mutable inputs its provenance declares."""
    if inputs is None:
        return
    for name, digest in sorted((evidence.get('artifacts') or {}).items()):
        bind(inputs, Path(path) / name, digest)
    child_dependencies(inputs, path, evidence, repo)
    for key in ('log', 'child_exit_receipt'):
        binding = bound.get(key) or {}
        bind(inputs, finalizer._resolve(binding['path'], repo), binding['sha256'])
    for room, entry in sorted((evidence.get('heading') or {}).items()):
        bind(inputs, finalizer._resolve(entry['path'], repo), entry['sha256'])
    for field, key in (('init', 'init_sha256'), ('checkpoint', 'checkpoint_sha256')):
        if key in evidence and args.get(field):
            bind(inputs, finalizer._resolve(args[field], repo), evidence[key])


def verify_job(job_dir, job, arm, repo=REPO, sensitivity=False, inputs=None):
    """Finding 2: the finalizer's own verifiers, re-run over every child of one job.

    Nothing here is taken from the completion: the job specification it bound is re-read
    and rehashed, ``verify_child`` re-runs each child's role validator over the directory
    and compares every field of its record with that result, ``job_lineage`` re-derives
    the init -> stage1 -> stage2 -> evaluation chain, and the job's own lineage fields
    must equal what the re-run derived. The frozen protocol and the DiffRIR indices are
    checked on top, because the finalizer knows nothing about exp_02's protocol.
    """
    job_dir, expect = Path(job_dir), EXPECT_OF[job]
    record = job_completion(job_dir, expect, arm, inputs)
    owner = job_owner(job_dir, record, inputs)
    recipe = []
    if inputs is not None and isinstance(record.get('log'), dict):
        bind(inputs, record['log']['path'], record['log']['sha256'])
    spec_path = Path(record['job_spec']['path'])
    _require(spec_path.is_file(), 'missing job spec {}'.format(spec_path))
    declared = record['job_spec']['sha256']
    _require(provenance.sha256_file(spec_path) == declared,
             'the job spec {} is not the bytes job {} bound'.format(spec_path, job_dir))
    if inputs is not None:      # bound before the loader runs, never after it
        bind(inputs, spec_path, declared)
    spec = finalizer.load_job_spec(str(spec_path), expect)
    # Finding 2b: the loader hashes what it read; that digest must still be the one the
    # job certified, so a spec edited while it was being loaded is refused here.
    _require(spec['job_spec_sha256'] == declared,
             'the job spec {} is not the bytes job {} bound'.format(spec_path, job_dir))
    children, rooms = {}, {}
    for name in sorted(finalizer.expected_children(expect)):
        path, role = job_dir / name, finalizer.child_role(name)
        completion = path / 'completion.json'
        _require(completion.is_file(),
                 'job {} child {} has no completion.json'.format(job_dir, name))
        certified = record['children'][name]
        _require(provenance.sha256_file(completion) == certified,
                 'job {} child {} is not the completion it bound'.format(job_dir, name))
        if inputs is not None:      # bound before the child validator is delegated to
            bind(inputs, completion, certified)
        evidence = dict(finalizer.verify_child(path, name, repo, spec), role=role)
        bound = _read_json(completion, name + '/completion.json', inputs, certified)
        _require(bound.get('admissible_arm') is True,
                 'child {} is not an admissible arm'.format(name))
        args = _read_json(path / 'args.json', name + '/args.json', inputs)
        child_protocol(args, name, role)
        recipe.extend(child_recipe(args, name, role))
        bind_child(inputs, path, evidence, bound, args, repo)
        if role == 'haa_eval':
            per = child_per_sample(job_dir, name, bound, arm, inputs)
            check_test_indices(name, evidence['room'], per['index'])
            _require(finalizer.exp06_recipe.strict_equal(per['meta'].get('heading'),
                                                         args.get('heading')),
                     'child {} per-sample meta heading is not the one its arguments '
                     'bound'.format(name))
            rooms[evidence['room']] = per
        children[name] = evidence
    lineage = finalizer.job_lineage(children, expect, spec)
    for field in sorted(lineage):
        _require(finalizer.exp06_recipe.strict_equal(record.get(field), lineage[field]),
                 'job {} records {} {!r}, not the {!r} the re-run derived'.format(
                     job_dir, field, record.get(field), lineage[field]))
    _require(set(rooms) == set(ROOMS),
             'the job {} evaluated {}'.format(job_dir, sorted(rooms)))
    _require(sensitivity or not recipe,
             'job {} did not run the registered recipe of plan 6.2: {}'.format(
                 job_dir, '; '.join(recipe)))
    return {'record': record, 'spec': spec, 'children': children, 'per': rooms,
            'owner': owner, 'closure': arm_closures(children),
            'heading': arm_headings(children), 'recipe_deviations': recipe}


def arm_directory(arm, roots):
    """One new arm's directory: its own registered root, under its own experiment's tree.

    ``ARMS[arm]['root']`` is where the arm was registered; ``roots`` maps an experiment to
    the tree its arms were actually written in (``--new-root``, ``--exp09-root``), so an
    arm is never looked for under another experiment's root.
    """
    return Path(roots[ARMS[arm]['experiment']]) / Path(ARMS[arm]['root']).name


def load_new_arm(roots, arm, init_sha256=None, repo=REPO, approved=None,
                 sensitivity=False):
    """One exp_06 or exp_09 arm: four verified jobs, one closure per role, one heading
    per room in the heading frame and none at all in the room frame."""
    base = arm_directory(arm, roots)
    _require(base.is_dir(), 'missing arm directory {}'.format(base))
    jobs, records, children, inputs, recipe = {}, {}, {}, {}, []
    for job in JOBS:
        job_dir = base / job
        _require(job_dir.is_dir(), 'the arm {} has no job {}'.format(arm, job))
        verified = verify_job(job_dir, job, arm, repo, sensitivity, inputs)
        recipe.extend('{} {}: {}'.format(arm, job, item)
                      for item in verified['recipe_deviations'])
        record = verified['record']
        _require(init_sha256 is None or record['init_sha256'] == init_sha256,
                 'the arm {} job {} did not start from the registered initialisation '
                 '{}'.format(arm, job, init_sha256))
        _require(job not in SEEDS or record['seed'] == int(job[len('seed'):]),
                 'the arm {} job {} records seed {!r}'.format(arm, job, record['seed']))
        jobs[job], records[job] = verified['per'], record
        for name in sorted(record['children']):
            children['{}/{}'.format(job, name)] = verified['children'][name]
    closures, headings = arm_closures(children), arm_headings(children)
    _require(bool(headings) == (ARMS[arm]['frame'] == 'heading'),
             'the {}-frame arm {} binds {} heading records'.format(
                 ARMS[arm]['frame'], arm, len(headings)))
    check_arm_identities(arm, closures, headings, approved)
    return {'arm': arm, 'branch': 'new', 'per': jobs, 'closure': closures, 'jobs': records,
            'heading': headings, 'root': str(base), 'inputs': inputs,
            'recipe_deviations': recipe}


# --- pairing, the finite cohort and the invalidity policy of section 7 -------------------

PAIR_META = ('eval_seed', 'num_shot')


def assert_pairing(a, b, label):
    """exp_02's own pairing assertions: the same queries, measured the same way."""
    _require(a['index'] == b['index'], 'pair {}: the query indices differ'.format(label))
    _require(a['ir_path'] == b['ir_path'], 'pair {}: the ir_path entries differ'.format(label))
    for key in PAIR_META:
        _require(a['meta'].get(key) == b['meta'].get(key) is not None,
                 'pair {}: {} {!r} != {!r}'.format(label, key, a['meta'].get(key),
                                                   b['meta'].get(key)))


def cell_rows(arms, x, y, room, metric):
    """Paired rows pooled over the three fine-tuning seeds, on the shared finite cohort."""
    columns, index = {x: [], y: []}, None
    for seed in SEEDS:
        for arm in (x, y):
            _require(seed in arms[arm]['per'] and room in arms[arm]['per'][seed],
                     'arm {} has no {} of {}'.format(arm, room, seed))
        a, b = arms[x]['per'][seed][room], arms[y]['per'][seed][room]
        assert_pairing(a, b, '{}-{} {} {}'.format(x, y, room, seed))
        index = list(a['index']) if index is None else index
        _require(list(a['index']) == index,
                 'the seeds of {}/{} do not share one query order'.format(room, metric))
        for arm, per in ((x, a), (y, b)):
            columns[arm].append(np.asarray(per[metric], dtype=float))
    stacked = {arm: np.stack(columns[arm]) for arm in (x, y)}
    finite = {arm: np.isfinite(stacked[arm]).all(axis=0) for arm in (x, y)}
    cohort = finite[x] & finite[y]
    # Finding 8: an empty cohort is a diagnosis, not an exception -- the void rule below
    # needs the exclusion counts it would otherwise never reach.
    rows = {arm: np.concatenate([stacked[arm][i][cohort] for i in range(len(SEEDS))])
            for arm in (x, y)}
    queries = np.asarray(index)[cohort]
    return {'a': rows[x], 'b': rows[y],
            'clusters': np.concatenate([queries] * len(SEEDS)),
            'seeds': np.concatenate([np.full(int(cohort.sum()), seed) for seed in SEEDS]),
            'cohort': int(cohort.sum()), 'n_test': len(index),
            'per_seed_diff': {seed: (float((stacked[x][i][cohort]
                                            - stacked[y][i][cohort]).mean())
                                     if cohort.any() else None)
                              for i, seed in enumerate(SEEDS)},
            'excluded': {arm: {'queries': int((~finite[arm]).sum()),
                               'seeds': {seed: int((~np.isfinite(stacked[arm][i])).sum())
                                         for i, seed in enumerate(SEEDS)}}
                         for arm in (x, y)}}


def void_reasons(rows, treatment, comparator):
    """Section 7's invalidity policy; a void verdict is never replaced by a subset."""
    excess = rows['excluded'][treatment]['queries'] - rows['excluded'][comparator]['queries']
    reasons = []
    if not rows['cohort']:
        reasons.append('no query of the {} test split is finite in every compared '
                       'run'.format(rows['n_test']))
    if excess > VOID_EXCESS_FRACTION * rows['n_test']:
        reasons.append('{} has {} more invalid queries than {}, above the {:.0%} of {} the '
                       'policy allows'.format(treatment, excess, comparator,
                                              VOID_EXCESS_FRACTION, rows['n_test']))
    if rows['cohort'] < VOID_COHORT_FRACTION * rows['n_test']:
        reasons.append('the cohort of {} queries is below {:.0%} of the {} in the test '
                       'split'.format(rows['cohort'], VOID_COHORT_FRACTION, rows['n_test']))
    return reasons


# --- the intervals and the verdicts ------------------------------------------------------


def intervals(rows, alpha, n_boot, seed=BOOT_SEEDS[0]):
    """The paired intervals, or ``None`` when the cohort the policy voided is empty."""
    if not rows['cohort']:
        return None
    return bootstrap.paired_intervals(rows['a'], rows['b'], rows['clusters'], rows['seeds'],
                                      alpha=alpha, n_boot=n_boot, seed=seed)


def converged_two_way(rows, alpha, n_boot):
    """The verdict-bearing interval, checked and quadrupled once by exp_06's bootstrap."""
    def factory(size):
        def interval(seed):
            return intervals(rows, alpha, size, seed)['two_way']
        return interval
    return bootstrap.converged_interval(factory, n_boot=n_boot, seed_a=BOOT_SEEDS[0],
                                        seed_b=BOOT_SEEDS[1], tol=CONVERGENCE_TOL)


def verdict_of(convergence, void, margin):
    """pass / fail / void / not_converged, in that order of precedence."""
    if void:
        return 'void'
    if convergence['status'] != 'converged':
        return 'not_converged'
    return 'pass' if convergence['interval'][1] < margin else 'fail'


def h2_label(interval):
    """A screen, never a parity claim: harm, improvement, or no detected difference."""
    low, high = interval
    if low > 0:
        return 'detected harm'
    if high < 0:
        return 'detected improvement'
    return 'no detected difference'


def decision_cell(arms, pair, room, metric, margin, n_boot=N_BOOT, alpha=ALPHA):
    """H1 or H1b: one two-way interval, the invalidity policy and the convergence gate."""
    treatment, comparator = pair
    rows = cell_rows(arms, treatment, comparator, room, metric)
    void = void_reasons(rows, treatment, comparator)
    cell = {'contrast': '{} - {}'.format(treatment, comparator), 'room': room,
            'metric': metric, 'margin': margin, 'alpha': alpha,
            'cohort': rows['cohort'], 'n_test': rows['n_test'], 'excluded': rows['excluded'],
            'per_seed_diff': rows['per_seed_diff'], 'void_reasons': void,
            'bootstrap_seeds': list(BOOT_SEEDS)}
    if void:  # Finding 8: a voided cell is reported, never bootstrapped and never raised.
        return dict(cell, diff=None, query=None, two_way=None, verdict='void',
                    convergence={'status': 'void', 'n_boot': None, 'interval': None,
                                 'attempts': []})
    convergence = converged_two_way(rows, alpha, n_boot)
    nominal = intervals(rows, alpha, convergence['n_boot'] or n_boot)
    return dict(cell, diff=nominal['diff'], query=nominal['query'],
                two_way=nominal['two_way'], convergence=convergence,
                verdict=verdict_of(convergence, void, margin))


def screen_cell_base(treatment, comparator, room, metric, alpha, adjusted_alpha, rows):
    """The fields of one screen cell that exist before anything is resampled."""
    return {'contrast': '{} - {}'.format(treatment, comparator), 'room': room,
            'metric': metric, 'alpha': alpha, 'adjusted_alpha': adjusted_alpha,
            'family': H2_FAMILY, 'cohort': rows['cohort'], 'n_test': rows['n_test'],
            'excluded': rows['excluded'], 'per_seed_diff': rows['per_seed_diff']}


def screen_cells(arms, pair, n_boot=N_BOOT, adjusted_n_boot=N_BOOT_ADJUSTED, alpha=ALPHA):
    """H2: the eleven room x metric cells, nominal and Bonferroni adjusted."""
    treatment, comparator = pair
    adjusted_alpha = bootstrap.bonferroni_alpha(alpha, H2_FAMILY)
    cells = []
    for room, metric in CELLS:
        rows = cell_rows(arms, treatment, comparator, room, metric)
        cell = screen_cell_base(treatment, comparator, room, metric, alpha,
                                adjusted_alpha, rows)
        nominal = intervals(rows, alpha, n_boot)
        if nominal is None:      # finding 8: nothing is resampled from an empty cohort
            cells.append(dict(cell, diff=None, nominal_two_way=None, query=None,
                              adjusted_two_way=None, convergence=None,
                              label='not available'))
            continue
        convergence = converged_two_way(rows, adjusted_alpha, adjusted_n_boot)
        interval = convergence['interval']
        cells.append(dict(cell, diff=nominal['diff'], nominal_two_way=nominal['two_way'],
                          query=nominal['query'], adjusted_two_way=interval,
                          convergence=convergence,
                          label='not converged' if interval is None else h2_label(interval)))
    return cells


def descriptive_cells(arms, pairs=DESCRIPTIVE, n_boot=N_BOOT, alpha=ALPHA):
    """D: the same cells for the three descriptive contrasts, nominal intervals only."""
    cells = []
    for treatment, comparator in pairs:
        if treatment not in arms or comparator not in arms:
            continue
        for room, metric in CELLS:
            rows = cell_rows(arms, treatment, comparator, room, metric)
            nominal = intervals(rows, alpha, n_boot)
            cells.append({'contrast': '{} - {}'.format(treatment, comparator), 'room': room,
                          'metric': metric,
                          'diff': None if nominal is None else nominal['diff'],
                          'nominal_two_way': None if nominal is None else nominal['two_way'],
                          'query': None if nominal is None else nominal['query'],
                          'cohort': rows['cohort'], 'n_test': rows['n_test'],
                          'per_seed_diff': rows['per_seed_diff']})
    return cells


# --- the room-frame side split -----------------------------------------------------------


CACHE_GEOMETRY = ('meta.json', 'xyzs.npy', 'speaker_xyz.npy')


def cache_side_labels(room, cache_root=None, inputs=None):
    """The room-frame sign of every test microphone's y, from the cache's own geometry."""
    root = Path(cache_root or legacy.HAA_ROOT) / room
    if inputs is not None:      # finding 10: the side split rests on these bytes too
        for name in CACHE_GEOMETRY:
            bind(inputs, root / name)
    meta = _read_json(root / 'meta.json', '{}/meta.json'.format(room))
    xyz = np.load(str(root / 'xyzs.npy')).astype(float)
    speaker = np.load(str(root / 'speaker_xyz.npy')).astype(float).reshape(-1)
    test = sorted(int(index) for index in meta['test'])
    signs = np.sign(xyz[test, 1] - speaker[1]).astype(int)
    _require(bool((signs != 0).all()),
             'a microphone of {} lies on the speaker axis: its side is undefined'.format(room))
    return dict(zip(test, (int(value) for value in signs)))


def side_labels(per, room, cache_root=None, inputs=None):
    """The writer's labels where they exist, always checked against the cache's geometry."""
    labels = cache_side_labels(room, cache_root, inputs)
    computed = [labels[int(index)] for index in per['index']]
    if 'side_label' in per:
        _require(list(per['side_label']) == computed, 'the per-sample side labels of {} are '
                 'not the room-frame signs the cache carries'.format(room))
    return computed


def side_split(arms, cache_root=None, job=ZEROSHOT, inputs=None):
    """Section 7's descriptive table: -y and +y mean error per arm, room and metric."""
    table = {}
    for arm in sorted(arms):
        for room, metric in CELLS:
            per = arms[arm]['per'][job][room]
            labels = np.asarray(side_labels(per, room, cache_root, inputs))
            values = np.asarray(per[metric], dtype=float)
            entry = {}
            for name, sign in (('minus_y', -1), ('plus_y', 1)):
                selected = values[(labels == sign) & np.isfinite(values)]
                entry[name] = float(selected.mean()) if selected.size else None
                entry['n_' + name] = int(selected.size)
            table['{}|{}|{}'.format(arm, room, metric)] = entry
    return {'job': job, 'cells': table}


# --- the tables, the approvals and the published outputs ---------------------------------


def arm_rows(arms):
    """exp_02's display rows: mean and standard deviation over the fine-tuning seeds."""
    rows = {}
    for arm in sorted(arms):
        for job_kind, jobs in (('fine-tuned', SEEDS), ('zero-shot', (ZEROSHOT,))):
            for room in ROOMS:
                for key, _, _ in METRICS:
                    per_run = []
                    for job in jobs:
                        values = np.asarray(arms[arm]['per'][job][room][key], dtype=float)
                        values = values[np.isfinite(values)]
                        per_run.append(float(values.mean()) if values.size else None)
                    finite = [value for value in per_run if value is not None]
                    rows['{}|{}|{}|{}'.format(arm, job_kind, room, key)] = {
                        'per_run': per_run, 'jobs': list(jobs),
                        'mean': float(np.mean(finite)) if finite else None,
                        'std': float(np.std(finite)) if len(finite) > 1 else None}
    return rows


def arm_inputs(arms, receipt=None, receipt_path=None):
    """Every byte this summary rests on, hashed when the bytes were consumed.

    Finding 10: nothing is rehashed here. The legacy receipt's own verification and each
    new arm's admission recorded what they read, so a file edited between the read and
    this call is a mismatch at publication, not a silently up-to-date digest.
    """
    inputs = {}
    if receipt is not None:
        for path, digest in sorted((receipt.get('inputs') or {}).items()):
            bind(inputs, path, digest)
        if receipt_path is not None:
            bind(inputs, receipt_path, receipt['sha256'])
    for arm in sorted(arms):
        for path, digest in sorted((arms[arm].get('inputs') or {}).items()):
            bind(inputs, path, digest)
    return inputs


def producer_inputs(inputs, producer=None, approvals_receipt=None):
    """Finding 2: the producer closure and the approvals record this run published."""
    for record in (producer or {}).get('files', ()):
        bind(inputs, REPO / record['path'], record['working_tree_sha256'])
    if approvals_receipt is not None:
        bind(inputs, approvals_receipt['path'], approvals_receipt['sha256'])
    return inputs


def recheck_inputs(inputs):
    """Finding 10: every bound byte is still the byte that was read (paired_compare's rule)."""
    for path in sorted(inputs):
        try:
            actual = provenance.sha256_file(path)
        except OSError as error:
            raise ValueError('input disappeared during analysis: {} ({})'.format(path, error))
        _require(actual == inputs[path], 'input changed during analysis: ' + path)


PRODUCER_CODE_KEY = {'summarize_haa': 'summarize_haa', 'legacy_receipt': 'summarize_haa'}


def check_producer_identity(approved, producer, identity, exploratory=False):
    """Finding 3: the closure that really ran, against the approved `code` digest."""
    if approved is None:
        return []
    key = PRODUCER_CODE_KEY[producer]
    pinned = approved['code'].get(key)
    if identity['sha256'] == pinned:
        return []
    deviation = 'this producer ran the closure {}, not the approved code.{} {}'.format(
        identity['sha256'], key, pinned)
    _require(exploratory, deviation)
    return [deviation]


def check_reused_identities(approved, receipt, gate_g1):
    """The exp_02 canonical record and the G1 gate artifact, as 6.4 approved them."""
    if approved is None:
        return {}
    canonical = {item['path']: item['sha256'] for item in receipt['files']}
    for name, key in (('stats.json', 'exp02_stats_sha256'),
                      ('summary.txt', 'exp02_summary_sha256')):
        _require(canonical.get(name) == approved['reused'][key],
                 'the legacy {} hashes to {}, not the approved reused.{} {}'.format(
                     name, canonical.get(name), key, approved['reused'][key]))
    pinned = approved['artifacts']['gate_g1_sha256']
    _require(gate_g1, 'the approved artifacts.gate_g1 requires --gate-g1 <path>')
    digest = provenance.sha256_file(gate_g1)
    _require(digest == pinned, 'the G1 artifact {} hashes to {}, not the approved '
             'artifacts.gate_g1 {}'.format(gate_g1, digest, pinned))
    return {str(Path(gate_g1).resolve()): digest}


def approvals(exploratory, path=None, producer='summarize_haa', commit=None, repo=REPO):
    """Section 6.4's producer gate; exploratory records what production would refuse.

    Finding 7: 6.4 approves the *commit* that fills the record, so a confirmatory run
    reads approvals that are a tracked path of this repository whose blob at the
    reviewed commit is byte-identical to the file read, and publishes that identity.
    """
    binding = {}
    if not exploratory:
        binding = {'repo': str(repo),
                   'commit': provenance.git_state(repo)['HEAD'] if commit is None
                   else commit}
    try:
        approved, receipt = approvals_api.load_approved_digests(path, **binding)
    except approvals_api.ApprovalsUnavailable as error:
        if not exploratory:
            raise
        return None, None, [str(error)]
    return approved, receipt, approvals_api.require_producer(approved, producer,
                                                             exploratory)


def classify_cell(cell):
    """exp_09's E1: the two-sided category and the +0.23 dB statement, side by side.

    Both read the one interval exp_06's margin verdict reads -- the seed-0 two-way
    interval of the size that converged -- and both use strict inequalities, so an
    endpoint exactly at 0 or at the margin establishes neither. They are independent:
    "detected harm" and "non-inferior at +0.23 dB" may hold together, and a failure to
    establish non-inferiority is not evidence of inferiority. A void or unconverged cell
    suppresses both, even though a nominal interval may still exist.
    """
    interval = (None if cell['verdict'] in ('void', 'not_converged')
                else cell['convergence']['interval'])
    if interval is None:
        return dict(cell, category=None, non_inferior_at_margin=None)
    return dict(cell, category=h2_label(interval),
                non_inferior_at_margin=bool(interval[1] < cell['margin']))


# --- exp_11's decision notation and its cells (plan section 3) ---------------------------

EXP11_FIELDS = ('category', 'y_non_inferior_at_margin', 'x_margin_advantage',
                'equivalent_at_margin')
EXP11_MARGIN_FIELDS = EXP11_FIELDS[1:]                 # the three that read the margin m
# A status that carries no interval withholds every decision-bearing field.
EXP11_WITHHELD = ('void', 'not_converged', 'unavailable', 'suppressed (draft)')


def decision_fields(interval, margin, fields):
    """Plan section 3's notation block, for one contrast X - Y with interval [L, U] at m:

    ``category``                  two-sided, exactly ``h2_label``'s semantics: harm for X
                                  if L > 0, improvement for X if U < 0, else no detected
                                  difference;
    ``y_non_inferior_at_margin``  Y is non-inferior to X at m iff -L < m;
    ``x_margin_advantage``        X has a margin-sized advantage iff U < -m;
    ``equivalent_at_margin``      95 % containment: L > -m and U < m.

    The direction is the cell's own ``contrast`` string, so X and Y are never guessed
    from a field name. Every inequality is strict: an endpoint exactly on a decision
    boundary establishes neither the statement nor its negation. Every field is ``None``
    -- withheld -- when the cell carries no interval, which is how a void, unconverged,
    degenerate or draft cell reports.
    """
    unknown = [name for name in fields if name not in EXP11_FIELDS]
    _require(not unknown, 'unknown decision field(s): ' + ', '.join(unknown))
    if [name for name in fields if name in EXP11_MARGIN_FIELDS]:
        _require(isinstance(margin, (int, float)) and not isinstance(margin, bool),
                 'the fields {} need a margin, not {!r}'.format(
                     ', '.join(name for name in fields if name in EXP11_MARGIN_FIELDS), margin))
    values = OrderedDict((name, None) for name in fields)
    if interval is None:
        return values
    low, high = float(interval[0]), float(interval[1])
    for name in values:
        if name == 'category':
            values[name] = h2_label((low, high))
        elif name == 'y_non_inferior_at_margin':
            values[name] = bool(-low < margin)
        elif name == 'x_margin_advantage':
            values[name] = bool(high < -margin)
        else:
            values[name] = bool(low > -margin and high < margin)
    return values


def zero_width(interval):
    """True only for a genuinely degenerate interval: two finite, equal endpoints.

    This is the one refusal of the frozen convergence helper exp_11 may publish as a
    defined ``unavailable`` statement. A non-finite endpoint (an overflowed mean gives
    ``inf``/``nan``) or a reversed interval is a numerical failure, not the cancellation
    the plan registers, and must propagate.
    """
    if not isinstance(interval, dict):
        return False
    low, high = interval.get('lo'), interval.get('hi')
    if not (isinstance(low, float) and isinstance(high, float)):
        return False
    return bool(math.isfinite(low) and math.isfinite(high) and low == high)


# The frozen convergence helper's own refusal, quoted so an exp_11 cell that never
# reached it records the same reason. A test raises it from the helper and compares.
ZERO_WIDTH_REASON = 'a zero-width interval cannot carry a convergence verdict'


def constant_difference(rows):
    """True when every paired difference is the same finite number.

    A resample of any size at any level then averages to that one value, so the interval
    is degenerate whatever alpha it is taken at. Reading it off the rows costs nothing
    and lets exp_11 decide *before* the frozen helper is called.
    """
    if not rows['cohort']:
        return False
    difference = np.asarray(rows['a'], dtype=float) - np.asarray(rows['b'], dtype=float)
    return bool(difference.size and np.isfinite(difference).all()
                and float(difference.max()) == float(difference.min()))


def exp11_convergence(rows, nominal, void, alpha, n_boot):
    """exp_11's convergence gate: invalidity first, the registered cancellation second,
    the frozen helper last.

    Neither an invalid cell nor an exactly cancelling one reaches ``converged_two_way``,
    because that helper refuses a zero-width interval and one such cell must withhold
    its label without aborting the phase summary. The cancellation is established two
    ways -- constant finite differences *and* a computed interval of two finite, equal
    endpoints -- so a set whose arithmetic overflowed (constant, but non-finite once
    averaged) is not mistaken for it. A refusal that still occurs is swallowed only when
    the interval really is finite and zero-width; anything else is a fault and
    propagates.
    """
    if void:
        return {'status': 'void', 'n_boot': None, 'interval': None, 'attempts': []}
    if nominal is not None and constant_difference(rows) and zero_width(nominal['two_way']):
        return {'status': 'unavailable', 'n_boot': None, 'interval': None, 'attempts': [],
                'reason': ZERO_WIDTH_REASON}
    try:
        return converged_two_way(rows, alpha, n_boot)
    except ValueError as error:
        if not zero_width(intervals(rows, alpha, n_boot)['two_way']):
            raise
        return {'status': 'unavailable', 'n_boot': None, 'interval': None, 'attempts': [],
                'reason': str(error)}


def decision_interval(cell):
    """The one interval every field of an exp_11 cell reads, or nothing at all.

    It is the interval exp_06's margin verdict reads -- the seed-0 two-way interval of
    the resample size that converged -- and a withholding status suppresses it even
    though a nominal interval may still exist.
    """
    if cell.get('status') in EXP11_WITHHELD:
        return None
    return (cell.get('convergence') or {}).get('interval')


def exp11_cell(rows, void, base, margin, fields, n_boot=N_BOOT, alpha=ALPHA):
    """One exp_11 decision cell: the invalidity policy, the convergence gate and the
    named fields of section 3.

    exp_06's ``decision_cell`` answers a margin verdict; exp_11's decisions are the
    statements above instead, so a cell carries a ``status`` and never a ``verdict``.
    A degenerate (zero-width) interval -- reachable by cancellation in an interaction --
    is a defined ``unavailable`` result: the frozen convergence helper refuses it, and
    that refusal is caught here rather than aborting the summary or being worked around
    in the helper. Only that case is caught -- ``zero_width`` requires two finite, equal
    endpoints -- so a positive width, a non-finite or reversed interval and any unrelated
    failure are faults and are re-raised.
    """
    cell = dict(base, margin=margin, alpha=alpha, fields=list(fields),
                cohort=rows['cohort'], n_test=rows['n_test'], excluded=rows['excluded'],
                per_seed_diff=rows['per_seed_diff'], void_reasons=list(void),
                bootstrap_seeds=list(BOOT_SEEDS))
    if void:   # decided before anything is resampled, exactly as decision_cell decides it
        cell.update(diff=None, query=None, two_way=None, status='void',
                    convergence={'status': 'void', 'n_boot': None, 'interval': None,
                                 'attempts': []})
        return dict(cell, **decision_fields(None, margin, fields))
    _require(rows['cohort'], 'an empty cohort is never bootstrapped: it is void')
    nominal = intervals(rows, alpha, n_boot)
    convergence = exp11_convergence(rows, nominal, void, alpha, n_boot)
    if convergence['status'] == 'converged':
        nominal = intervals(rows, alpha, convergence['n_boot'] or n_boot)
    cell.update(diff=nominal['diff'], query=nominal['query'], two_way=nominal['two_way'],
                convergence=convergence,
                status={'converged': 'reported'}.get(convergence['status'],
                                                     convergence['status']))
    return dict(cell, **decision_fields(decision_interval(cell), margin, fields))


def interaction_rows(arms, quad, room, metric):
    """The four-arm cohort of an interaction ``(X1 - Y1) - (X2 - Y2)``.

    ``cell_rows`` assembles two arms; an interaction must be estimated on the queries
    finite in **all four** arms and all three seeds, so the mask is built once here
    rather than by combining two independently assembled cohorts. Every arm is paired
    against the first with exp_02's own assertions -- the query index, the ``ir_path``
    entries and the protocol metadata (``eval_seed``, ``num_shot``) -- and each arm's
    own exclusions are reported, not only the joint ones.

    The returned rows are ``a`` = X1 - Y1 and ``b`` = X2 - Y2 on that one cohort, so the
    frozen ``paired_intervals(a, b, clusters, seeds)`` resamples the interaction jointly
    and keeps its covariance; two separately bootstrapped intervals are never subtracted.
    """
    _require(len(quad) == 4 and len(set(quad)) == 4,
             'an interaction needs four distinct arms, not {}'.format(list(quad)))
    columns, index = {arm: [] for arm in quad}, None
    for seed in SEEDS:
        for arm in quad:
            _require(seed in arms[arm]['per'] and room in arms[arm]['per'][seed],
                     'arm {} has no {} of {}'.format(arm, room, seed))
        first = arms[quad[0]]['per'][seed][room]
        for arm in quad[1:]:
            assert_pairing(first, arms[arm]['per'][seed][room],
                           '{}-{} {} {}'.format(quad[0], arm, room, seed))
        index = list(first['index']) if index is None else index
        _require(list(first['index']) == index,
                 'the seeds of {}/{} do not share one query order'.format(room, metric))
        for arm in quad:
            columns[arm].append(np.asarray(arms[arm]['per'][seed][room][metric],
                                           dtype=float))
    stacked = {arm: np.stack(columns[arm]) for arm in quad}
    finite = {arm: np.isfinite(stacked[arm]).all(axis=0) for arm in quad}
    cohort = np.ones(len(index), dtype=bool)
    for arm in quad:
        cohort = cohort & finite[arm]
    sides = [stacked[quad[0]] - stacked[quad[1]], stacked[quad[2]] - stacked[quad[3]]]
    rows = [np.concatenate([side[i][cohort] for i in range(len(SEEDS))]) for side in sides]
    queries = np.asarray(index)[cohort]
    return {'a': rows[0], 'b': rows[1],
            'clusters': np.concatenate([queries] * len(SEEDS)),
            'seeds': np.concatenate([np.full(int(cohort.sum()), seed) for seed in SEEDS]),
            'cohort': int(cohort.sum()), 'n_test': len(index),
            'per_seed_diff': {seed: (float((sides[0][i][cohort]
                                            - sides[1][i][cohort]).mean())
                                     if cohort.any() else None)
                              for i, seed in enumerate(SEEDS)},
            'excluded': {arm: {'queries': int((~finite[arm]).sum()),
                               'seeds': {seed: int((~np.isfinite(stacked[arm][i])).sum())
                                         for i, seed in enumerate(SEEDS)}}
                         for arm in quad}}


def interaction_void_reasons(arms, quad, rows, room, metric):
    """The interaction's invalidity policy: the joint cohort, plus both components.

    The joint cohort must cover at least 99 % of the room's test split, and the
    component contrasts keep exactly the checks section 7 gives them -- a component the
    two-arm policy would void cannot be rescued by the interaction's own cohort.
    """
    reasons = []
    if not rows['cohort']:
        reasons.append('no query of the {} test split is finite in all four compared '
                       'arms'.format(rows['n_test']))
    if rows['cohort'] < VOID_COHORT_FRACTION * rows['n_test']:
        reasons.append('the joint cohort of {} queries is below {:.0%} of the {} in the '
                       'test split'.format(rows['cohort'], VOID_COHORT_FRACTION,
                                           rows['n_test']))
    for treatment, comparator in ((quad[0], quad[1]), (quad[2], quad[3])):
        component = cell_rows(arms, treatment, comparator, room, metric)
        reasons.extend('{} - {}: {}'.format(treatment, comparator, reason)
                       for reason in void_reasons(component, treatment, comparator))
    return reasons


def exp11_decision(arms, spec, n_boot=N_BOOT, alpha=ALPHA):
    """One registered exp_11 decision: a two-arm contrast, or the four-arm interaction."""
    room, metric, margin = spec['room'], spec['metric'], spec.get('margin')
    base = {'name': spec['name'], 'room': room, 'metric': metric,
            'reading': spec.get('reading')}
    if spec.get('kind') == 'interaction':
        quad = tuple(spec['arms'])
        rows = interaction_rows(arms, quad, room, metric)
        void = interaction_void_reasons(arms, quad, rows, room, metric)
        base.update(kind='interaction', arms=list(quad),
                    contrast='({} - {}) - ({} - {})'.format(*quad))
    else:
        x, y = spec['pair']
        rows = cell_rows(arms, x, y, room, metric)
        void = void_reasons(rows, x, y)
        base.update(kind='contrast', x=x, y=y, contrast='{} - {}'.format(x, y))
    return exp11_cell(rows, void, base, margin, tuple(spec['fields']), n_boot, alpha)


def exp11_screen_cells(arms, pair, n_boot=N_BOOT, adjusted_n_boot=N_BOOT_ADJUSTED,
                       alpha=ALPHA):
    """One exp_11 screen family: the eleven Bonferroni-adjusted cells of exp_09's E2,
    under the universal suppression plan section 3 registers for exp_11.

    The historical screens label any cell with a non-empty cohort; exp_11 withholds the
    label of every cell the invalidity policy voids or the convergence check refuses, and
    says why. The gate is exp_11's own and it runs *first*: ``screen_cells`` resamples
    every non-empty cohort before it reports anything, and the frozen convergence helper
    refuses a zero-width interval, so a cell that exp_11 must withhold would abort the
    whole phase summary there. This builds its own cells from the same helpers and the
    same ``screen_cell_base``; ``screen_cells`` is untouched and keeps publishing exp_06's
    and exp_09's screens exactly as it always has.
    """
    treatment, comparator = pair
    adjusted_alpha = bootstrap.bonferroni_alpha(alpha, H2_FAMILY)
    cells = []
    for room, metric in CELLS:
        rows = cell_rows(arms, treatment, comparator, room, metric)
        cell = screen_cell_base(treatment, comparator, room, metric, alpha,
                                adjusted_alpha, rows)
        void = void_reasons(rows, treatment, comparator)
        nominal = intervals(rows, alpha, n_boot)
        if nominal is None:      # an empty cohort: nothing is resampled at all
            cells.append(dict(cell, diff=None, nominal_two_way=None, query=None,
                              adjusted_two_way=None, convergence=None, label=None,
                              withheld=True, void_reasons=void))
            continue
        convergence = exp11_convergence(rows, nominal, void, adjusted_alpha,
                                        adjusted_n_boot)
        interval = convergence['interval']
        withheld = bool(void) or interval is None
        cells.append(dict(cell, diff=nominal['diff'], nominal_two_way=nominal['two_way'],
                          query=nominal['query'], adjusted_two_way=interval,
                          convergence=convergence, void_reasons=void, withheld=withheld,
                          label=None if withheld else h2_label(interval)))
    return cells


def exp11_screens(arms, families, n_boot=N_BOOT, adjusted_n_boot=N_BOOT_ADJUSTED,
                  alpha=ALPHA):
    """The declared screen families, keyed by their own contrast.

    Each is its own eleven-cell Bonferroni family: the adjustment protects within a
    family and no claim is made across them (section 3, S1).
    """
    return OrderedDict(
        ('{} - {}'.format(*pair),
         exp11_screen_cells(arms, tuple(pair), n_boot, adjusted_n_boot, alpha))
        for pair in families)


def historical_source(path, label, inputs=None):
    """One canonical record, admitted only with the summary its own bytes bind."""
    path = Path(path)
    _require(path.is_file(), 'missing historical source {}'.format(path))
    digest = provenance.sha256_file(path)
    record = _read_json(path, label, inputs, digest)
    summary = path.with_name('summary.txt')
    _require(summary.is_file(),
             'the historical source {} has no {}'.format(path, summary.name))
    published = provenance.sha256_file(summary)
    _require(published == record.get('summary_sha256'),
             'the summary {} hashes to {}, not the summary_sha256 {} the record '
             'binds'.format(summary, published, record.get('summary_sha256')))
    if inputs is not None:
        bind(inputs, summary, published)
    return record, digest


def historical_row(record, spec, room, metric):
    """The row the selector names -- by contrast, room and metric, never by position."""
    table = record.get(spec['table'])
    if isinstance(table, list):
        found = [row for row in table
                 if (row.get('contrast'), row.get('room'), row.get('metric'))
                 == (spec['contrast'], room, metric)]
        _require(len(found) == 1, 'the historical table {} holds {} rows for {} {} '
                 '{}'.format(spec['table'], len(found), spec['contrast'], room, metric))
        row = found[0]
    else:
        row = table
        _require(isinstance(row, dict),
                 'the historical record has no {}'.format(spec['table']))
    for field, value in (('contrast', spec['contrast']), ('room', room),
                         ('metric', metric)):
        _require(row.get(field) == value, 'the historical {} records {} {!r}, not the '
                 'selected {!r}'.format(spec['table'], field, row.get(field), value))
    return row


def historical_rows(specs=EXP11_HISTORICAL, repo=REPO, room=H1_ROOM, metric=H1_METRIC,
                    inputs=None):
    """R1: rows re-reported from exp_06's and exp_09's canonical records.

    Extant fields are copied exactly, keeping their original names, meaning and interval
    convention (``nominal_two_way`` stays a descriptive nominal interval); a field the
    source does not record is ``null`` here and named in ``not_recorded``. Each row
    carries the source path, that file's sha256 and the selector it was found by.
    Nothing is recomputed, no convergence is manufactured, and a descriptive row is never
    reclassified as a decision.
    """
    rows, sources = [], {}
    for spec in specs:
        if spec['source'] not in sources:
            sources[spec['source']] = historical_source(
                Path(repo) / spec['source'], 'historical ' + spec['source'], inputs)
        record, digest = sources[spec['source']]
        found = historical_row(record, spec, room, metric)
        copied = OrderedDict()
        for field in HISTORICAL_FIELDS:
            if field not in spec['fields']:
                copied[field] = None
                continue
            _require(field in found, 'the historical row {} records no {}'.format(
                spec['name'], field))
            copied[field] = found[field]
        rows.append(dict(copied, name=spec['name'], kind=spec['kind'],
                         contrast=spec['contrast'], room=room, metric=metric,
                         inference='none (copied)', source=spec['source'],
                         source_sha256=digest,
                         selector={'table': spec['table'], 'contrast': spec['contrast'],
                                   'room': room, 'metric': metric},
                         not_recorded=[field for field in HISTORICAL_FIELDS
                                       if field not in spec['fields']]))
    return rows


def exp11_tables(result, arms, config, n_boot=N_BOOT,
                 adjusted_n_boot=N_BOOT_ADJUSTED, exploratory=False, sensitivity=False,
                 historical_root=REPO):
    """exp_11's phase tables: the registered decisions, the declared screen families and
    the copied historical rows, into the record ``analyse`` has already opened.

    A draft states no conclusion of any kind -- every decision-bearing field is withheld
    and the status says so -- and a relaxed admission labels every status, exactly as the
    historical path labels its verdicts.
    """
    result['phase'] = config['phase']
    result['decisions'] = [spec['name'] for spec in config['exp11_decisions']]
    for spec in config['exp11_decisions']:
        cell = exp11_decision(arms, spec, n_boot)
        if exploratory:
            cell = dict(cell, status='suppressed (draft)',
                        **decision_fields(None, spec.get('margin'),
                                          tuple(spec['fields'])))
        elif sensitivity:
            cell = dict(cell, status='sensitivity: ' + cell['status'])
        result[spec['name']] = cell
    result['screens'] = exp11_screens(arms, config['screens'], n_boot, adjusted_n_boot)
    result['historical'] = historical_rows(config['historical'], historical_root,
                                           inputs=result['inputs'])
    return result


def check_output_paths(experiment, json_path, summary_path):
    """No experiment's run may write another's canonical record."""
    targets = {str(Path(path).resolve()) for path in (json_path, summary_path)}
    for name, config in EXPERIMENTS.items():
        if name == experiment:
            continue
        for canonical in config['outputs']:
            _require(str((REPO / canonical).resolve()) not in targets,
                     'an {} run may not write {}, the canonical record of {}'.format(
                         experiment, canonical, name))


def analyse(arms, n_boot=N_BOOT, adjusted_n_boot=N_BOOT_ADJUSTED, cache_root=None,
            exploratory=False, receipt=None, receipt_path=None, approved=None,
            deviations=(), approvals_receipt=None, producer=None, extra_inputs=(),
            sensitivity=False, experiment='exp06', historical_root=REPO):
    """Every displayed number, and the evidence each rests on."""
    config = EXPERIMENTS[experiment]
    # The frozen configuration decides what is published, not what the caller happens to
    # have loaded: an experiment's tables never carry another's arm, and a missing one is
    # a refusal rather than a quietly shorter table.
    missing = [name for name in config['arms'] if name not in arms]
    _require(not missing, 'the {} analysis needs the arms {}'.format(
        experiment, ', '.join(missing)))
    arms = OrderedDict((name, arms[name]) for name in config['arms'])
    cache_inputs = {}
    # Code review round 1 finding 3: exp_06's record is the one main publishes, field for
    # field, so the experiment is named only where it is not the default. `render` reads it
    # back with exp_06 as the default, and no other field here is exp_09's.
    result = dict({'schema_version': 1},
                  **({} if experiment == 'exp06' else {'experiment': experiment}))
    result.update({'exploratory': bool(exploratory),
                  'mode': 'sensitivity' if sensitivity else 'primary',
                  'deviations': list(deviations), 'arms': {name: {
                      'branch': arms[name]['branch'], 'root': arms[name]['root'],
                      'closure': arms[name]['closure'], 'backbone': ARMS[name]['backbone'],
                      'frame': ARMS[name]['frame']} for name in sorted(arms)},
                  'margin_db': H1_MARGIN_DB, 'alpha': ALPHA, 'family': H2_FAMILY,
                  'n_boot': n_boot, 'n_boot_adjusted': adjusted_n_boot,
                  'bootstrap_seeds': list(BOOT_SEEDS), 'convergence_tolerance': CONVERGENCE_TOL,
                  'heading_k': HEADING_K, 'rows': arm_rows(arms),
                  'legacy_receipt': None if receipt is None else {
                      'path': str(receipt_path), 'sha256': receipt['sha256'],
                      'label': receipt['label'], 'files': len(receipt['files'])},
                  'approved_digests': approvals_receipt, 'producer': producer,
                  'side_split': side_split(arms, cache_root, inputs=cache_inputs)})
    result['inputs'] = arm_inputs(arms, receipt, receipt_path)
    for path, digest in sorted(dict(cache_inputs, **dict(extra_inputs)).items()):
        bind(result['inputs'], path, digest)
    producer_inputs(result['inputs'], producer, approvals_receipt)
    if 'phase' in config:   # exp_11 publishes section 3's statements, not margin verdicts
        return exp11_tables(result, arms, config, n_boot, adjusted_n_boot, exploratory,
                            sensitivity, historical_root)
    for name, pair, room, metric, margin in config['decisions']:
        cell = decision_cell(arms, pair, room, metric, margin, n_boot)
        result[name] = classify_cell(cell) if name in config['classified'] else cell
    result[config['screen'][0]] = screen_cells(arms, config['screen'][1], n_boot,
                                               adjusted_n_boot)
    result[config['descriptive'][0]] = descriptive_cells(arms, config['descriptive'][1],
                                                         n_boot)
    for name, *_ in config['decisions']:
        classified = name in config['classified']
        if exploratory:  # a draft states no conclusion of either kind
            result[name]['verdict'] = 'suppressed (draft)'
            if classified:
                result[name].update(category=None, non_inferior_at_margin=None)
        if sensitivity:  # finding 3: a relaxed admission never reads as the primary one
            result[name]['verdict'] = 'sensitivity: ' + result[name]['verdict']
            if classified and result[name]['category'] is not None:
                result[name]['category'] = 'sensitivity: ' + result[name]['category']
    return result


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return _safe(value.item())
    return value


UNAVAILABLE = 'not available'


def _number(value):
    return UNAVAILABLE if value is None else '{:+.4f}'.format(value)


def _bounds(interval):
    """An interval the invalidity policy voided has no endpoints to print."""
    if interval is None:
        return UNAVAILABLE
    return '[{:+.4f}, {:+.4f}]'.format(interval['lo'], interval['hi'])


def render_decisions(result, config):
    """exp_06's and exp_09's blocks: the decisions, the screen and the descriptive
    contrasts, printed exactly as they have always been printed."""
    lines = []
    for name, *_ in config['decisions']:
        cell = result[name]
        lines.append('\n{} {} {} {}: diff {}, two-way {}, margin {}, cohort {}/{} -> '
                     '{}'.format(name, cell['contrast'], cell['room'], cell['metric'],
                                 _number(cell['diff']), _bounds(cell['two_way']),
                                 cell['margin'], cell['cohort'], cell['n_test'],
                                 cell['verdict']))
        if name in config['classified']:
            lines.append('  category: {}; non-inferior at {} dB: {}'.format(
                UNAVAILABLE if cell['category'] is None else cell['category'],
                cell['margin'], UNAVAILABLE if cell['non_inferior_at_margin'] is None
                else cell['non_inferior_at_margin']))
        lines.extend('  void: ' + reason for reason in cell['void_reasons'])
    screen, descriptive = config['screen'][0], config['descriptive'][0]
    lines.append('\n{} screen ({} cells, adjusted at alpha/{})'.format(
        screen, len(result[screen]), H2_FAMILY))
    for cell in result[screen]:
        lines.append('  {:14s} {:4s} diff {} nominal {} adjusted {} -> {}'.format(
            cell['room'], cell['metric'], _number(cell['diff']),
            _bounds(cell['nominal_two_way']),
            'not available' if cell['adjusted_two_way'] is None else cell['adjusted_two_way'],
            cell['label']))
    lines.append('\nDescriptive contrasts')
    for cell in result[descriptive]:
        lines.append('  {:22s} {:14s} {:4s} diff {} {}'.format(
            cell['contrast'], cell['room'], cell['metric'], _number(cell['diff']),
            _bounds(cell['nominal_two_way'])))
    return lines


def render_exp11(result):
    """exp_11's blocks: each registered statement under the contrast whose direction it
    names, the three declared screen families, and the copied historical rows."""
    lines = []
    for name in result['decisions']:
        cell = result[name]
        lines.append('\n{} {} {} {}: diff {}, two-way {}, cohort {}/{} -> {}'.format(
            name, cell['contrast'], cell['room'], cell['metric'], _number(cell['diff']),
            _bounds(cell['two_way']), cell['cohort'], cell['n_test'], cell['status']))
        for field in cell['fields']:
            lines.append('  {}: {}'.format(
                field, UNAVAILABLE if cell[field] is None else cell[field]))
        if cell['margin'] is not None:
            lines.append('  margin: {} dB'.format(cell['margin']))
        lines.append('  reading: {}'.format(cell['reading']))
        lines.extend('  void: ' + reason for reason in cell['void_reasons'])
    for name in result['screens']:
        cells = result['screens'][name]
        lines.append('\nS1 screen {} ({} cells, adjusted at alpha/{})'.format(
            name, len(cells), H2_FAMILY))
        for cell in cells:
            lines.append('  {:14s} {:4s} diff {} nominal {} adjusted {} -> {}'.format(
                cell['room'], cell['metric'], _number(cell['diff']),
                _bounds(cell['nominal_two_way']),
                UNAVAILABLE if cell['adjusted_two_way'] is None
                else cell['adjusted_two_way'],
                UNAVAILABLE if cell['label'] is None else cell['label']))
    lines.append('\nR1 historical rows (copied; no new inference)')
    for row in result['historical']:
        lines.append('  {:8s} {:22s} {:4s} diff {} two-way {} nominal {} -> {}'.format(
            row['name'], row['contrast'], row['metric'], _number(row['diff']),
            _bounds(row['two_way']), _bounds(row['nominal_two_way']),
            UNAVAILABLE if row['verdict'] is None else row['verdict']))
        lines.append('    source {} ({}) not recorded: {}'.format(
            row['source'], row['source_sha256'][:12], ', '.join(row['not_recorded'])))
    return lines


def render(result):
    """The printed summary: exp_02's arm table, then the paired tables and the verdicts."""
    lines = []
    if result.get('mode') == 'sensitivity':
        lines.append('SENSITIVITY - relaxed admission; not the primary comparison of 6.2')
    if result['exploratory']:
        lines.append('DRAFT - exploratory run; verdicts are suppressed')
    if result['exploratory'] or result.get('mode') == 'sensitivity':
        lines.extend('Deviation: ' + item for item in result['deviations'])
    lines.append('{:22s}'.format('model') + ''.join('| {:39s}'.format(r) for r in ROOMS))
    lines.append('{:22s}'.format('') + ''.join(
        '| ' + ''.join('{:>13s}'.format(name) for _, name, _ in METRICS) for _ in ROOMS))
    for arm in ARMS:
        if arm not in result['arms']:
            continue
        for kind in ('zero-shot', 'fine-tuned'):
            line = '{:22s}'.format('{} {}'.format(arm, kind))
            for room in ROOMS:
                cells = []
                for key, _, nd in METRICS:
                    row = result['rows']['{}|{}|{}|{}'.format(arm, kind, room, key)]
                    cells.append(legacy.fmt(row['per_run'], nd))
                line += '| ' + ''.join(cells)
            lines.append(line)
    config = EXPERIMENTS[result.get('experiment', 'exp06')]
    lines.extend(render_exp11(result) if 'phase' in config
                 else render_decisions(result, config))
    lines.append('\nRoom-frame side split ({} job)'.format(result['side_split']['job']))
    for key in sorted(result['side_split']['cells']):
        entry = result['side_split']['cells'][key]
        lines.append('  {:30s} -y {!s:>10.10} (n={}) +y {!s:>10.10} (n={})'.format(
            key, entry['minus_y'], entry['n_minus_y'], entry['plus_y'], entry['n_plus_y']))
    return '\n'.join(lines) + '\n'


def write_outputs(result, json_path, summary_path):
    """Staged publication (finding 12): both files appear together, or neither at all.

    Finding 10: the inputs are revalidated here, after the bootstrap and immediately
    before the first byte is published, so a file edited during the analysis is refused
    rather than described by a stale digest.
    """
    paths = [Path(summary_path), Path(json_path)]
    if len({path.resolve() for path in paths}) != 2 or any(
            os.path.lexists(str(path)) for path in paths):
        raise FileExistsError('output paths must be distinct and absent')
    text = render(result)
    record = dict(result, summary_path=str(paths[0].resolve()),
                  summary_sha256=hashlib.sha256(text.encode()).hexdigest())
    data = (json.dumps(_safe(record), sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode()
    recheck_inputs(result.get('inputs') or {})
    created = []
    try:
        for path, payload in zip(paths, (text.encode(), data)):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=str(path.parent))
            try:
                with os.fdopen(handle, 'wb') as stream:
                    provenance.apply_umask(stream.fileno())
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, str(path))     # atomic, and still exclusive
                created.append(path)
            finally:
                os.unlink(temporary)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    return record, hashlib.sha256(data).hexdigest(), text


def build_parser():
    parser = argparse.ArgumentParser(description='Summarise exp_06 and exp_02 HAA arms.')
    parser.add_argument('--experiment', choices=tuple(EXPERIMENTS), default='exp06',
                        help='which frozen set of arms, contrasts and outputs to publish')
    parser.add_argument('--legacy-root', default=LEGACY_ROOT)
    parser.add_argument('--new-root', default=NEW_ROOT)
    parser.add_argument('--exp09-root', default=EXP09_ROOT)
    parser.add_argument('--exp11-root', default=EXP11_ROOT)
    parser.add_argument('--legacy-receipt', default='ckpt/exp06/legacy_receipt.json')
    parser.add_argument('--write-legacy-receipt')
    parser.add_argument('--approved')
    parser.add_argument('--approved-commit',
                        help='the reviewed commit the approvals must be committed at')
    parser.add_argument('--cache-root')
    parser.add_argument('--gate-g1')
    parser.add_argument('--json')
    parser.add_argument('--summary')
    parser.add_argument('--n-boot', type=int, default=N_BOOT)
    parser.add_argument('--n-boot-adjusted', type=int, default=N_BOOT_ADJUSTED)
    parser.add_argument('--exploratory', action='store_true')
    parser.add_argument('--sensitivity', action='store_true',
                        help='admit runs outside 6.2 recipe and label every output')
    return parser


def main(argv=None):
    """0 on success; a refusal exits 1 and an argparse usage error exits 2."""
    args = build_parser().parse_args(argv)
    producer = 'legacy_receipt' if args.write_legacy_receipt else 'summarize_haa'
    approved, approvals_receipt, deviations = approvals(args.exploratory, args.approved,
                                                        producer, args.approved_commit)
    identity = source_identity(REPO, strict=not args.exploratory)
    deviations = list(deviations) + check_producer_identity(approved, producer, identity,
                                                            args.exploratory)
    if args.write_legacy_receipt:      # finding 3: the receipt is a producer, and gated
        record, digest = write_legacy_receipt(args.write_legacy_receipt, args.legacy_root,
                                              strict=not args.exploratory, identity=identity)
        print(json.dumps({'receipt': str(args.write_legacy_receipt), 'sha256': digest,
                          'files': len(record['files']), 'label': record['label']}))
        return 0
    _require(args.json and args.summary, 'both --json and --summary are required')
    check_output_paths(args.experiment, args.json, args.summary)
    config = EXPERIMENTS[args.experiment]
    roots = {'exp06': args.new_root, 'exp09': args.exp09_root,
             'exp11': args.exp11_root}
    new_arms = tuple(arm for arm in config['arms'] if ARMS[arm]['branch'] == 'new')
    binding = approved['reused']['legacy_receipt'] if approved else None
    if binding is not None and binding.get('sha256') is None:
        binding = None
    arms, receipt = load_legacy(args.legacy_root, args.legacy_receipt, binding)
    extra = check_reused_identities(None if args.exploratory else approved, receipt,
                                    args.gate_g1)
    inits = ({} if args.exploratory
             else expected_inits(approved, new_arms, extra))
    for arm in new_arms:
        arms[arm] = load_new_arm(roots, arm, inits.get(arm), REPO,
                                 None if args.exploratory else approved, args.sensitivity)
        deviations = deviations + list(arms[arm].get('recipe_deviations') or ())
    result = analyse(arms, args.n_boot, args.n_boot_adjusted, args.cache_root,
                     args.exploratory, receipt, args.legacy_receipt, approved, deviations,
                     approvals_receipt, identity, extra, args.sensitivity, args.experiment)
    record, digest, text = write_outputs(result, args.json, args.summary)
    print(text)
    published = {'json': args.json, 'sha256': digest,
                 'summary_sha256': record['summary_sha256']}
    if args.experiment != 'exp06':     # exp_06 prints the keys it has always printed
        published['experiment'] = args.experiment
    if 'phase' in config:      # exp_11 reports a status per registered statement
        published['phase'] = config['phase']
        published.update({name: record[name]['status'] for name in record['decisions']})
    published.update({name: record[name]['verdict'] for name, *_ in config['decisions']})
    print(json.dumps(published))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit('refusing: ' + str(error))
