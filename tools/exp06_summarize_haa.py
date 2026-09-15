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
                'init_sha256': None}),      # the approved epoch_012 artifact
    ('control_hf', {'label': 'D', 'branch': 'new', 'root': 'ckpt/exp06/sim2real/control_hf',
                    'backbone': 'simple', 'frame': 'heading',
                    'init_sha256': EXP01_CONTROL['sha256']}),
    ('cyl_hf', {'label': 'F', 'branch': 'new', 'root': 'ckpt/exp06/sim2real/cyl_hf',
                'backbone': 'cylindrical', 'frame': 'heading',
                'init_sha256': EXP01_CYL['sha256']})])
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
    runs = legacy.load_runs(str(root))
    complete, problems = legacy.completeness(runs)
    _require(complete, 'the historical exp_02 root is incomplete: ' + '; '.join(problems))
    receipt = None if receipt_path is None else verify_legacy_receipt(
        receipt_path, root, approved)
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


def job_completion(job_dir, expect, arm):
    """One job's A3 record: no receipt, a bound job spec, an owner and its children."""
    job_dir = Path(job_dir)
    path = job_dir / 'completion.json'
    _require(path.is_file(), 'job {} has no completion.json'.format(job_dir))
    _require(not (job_dir / 'child_exit.json').exists(),
             'amendment A3: the job root {} carries a child_exit.json'.format(job_dir))
    record = _read_json(path, 'job completion.json')
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


def job_owner(job_dir, record):
    """Finding 6: amendment A3 binds the launcher that owned the job, so the job root's
    own ``launch.pid`` is read and required to be the ``owner_pid`` the completion
    recorded. Liveness is deliberately not checked: a retrospective analysis runs long
    after the launcher exited and a reused pid would prove nothing either way.
    """
    path = Path(job_dir) / 'launch.pid'
    _require(path.is_file(), 'job {} has no launch.pid: amendment A3 binds the launcher '
             'that owned it'.format(job_dir))
    try:
        pid = int(path.read_text().split()[0])
    except (IndexError, ValueError) as error:
        raise ValueError('job {} has an unreadable launch.pid: {}'.format(job_dir, error))
    _require(pid == record['owner_pid'], 'job {} holds launch.pid {}, not the owner_pid {} '
             'its completion bound'.format(job_dir, pid, record['owner_pid']))
    return {'path': str(path.resolve()), 'pid': pid, 'sha256': provenance.sha256_file(path)}


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
# shortened training or an S1 seeded-phase evaluation cannot enter the primary
# comparison against exp_02's historical arms. They are admissible only under
# ``--sensitivity``, which labels every output it produces.
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


def expected_inits(approved):
    """What each new arm must have started from: exp_01's weights, or the approved epoch."""
    inits = {arm: ARMS[arm]['init_sha256'] for arm in NEW_ARMS}
    if approved:
        inits['cyl_or'] = approved.get('artifacts', {}).get('epoch_012', {}).get('sha256')
    return inits


def child_per_sample(job_dir, name, record, arm):
    """One evaluation child's per-sample file, with the fields the summariser reads."""
    path = Path(job_dir) / name
    names = sorted(key for key in record['artifacts'] if key.startswith('per_sample_'))
    _require(len(names) == 1, 'child {} bound {} per-sample files'.format(name, len(names)))
    per = _read_json(path / names[0], names[0])
    meta = per.get('meta')
    _require(isinstance(meta, dict), 'child {} per-sample records no meta'.format(name))
    _require('heading' in meta, 'child {} per-sample meta records no heading'.format(name))
    if ARMS[arm]['frame'] == 'heading':
        _heading_rolls(meta['heading'], 'child {} per-sample meta'.format(name))
    side = per.get('side_label')
    _require(isinstance(side, list) and len(side) == len(per.get('index', [])),
             'child {} per-sample records no room-frame side_label'.format(name))
    _require(all(value in (-1, 1) and not isinstance(value, bool) for value in side),
             'child {} side labels are not the room-frame signs'.format(name))
    return per


def verify_job(job_dir, job, arm, repo=REPO, sensitivity=False):
    """Finding 2: the finalizer's own verifiers, re-run over every child of one job.

    Nothing here is taken from the completion: the job specification it bound is re-read
    and rehashed, ``verify_child`` re-runs each child's role validator over the directory
    and compares every field of its record with that result, ``job_lineage`` re-derives
    the init -> stage1 -> stage2 -> evaluation chain, and the job's own lineage fields
    must equal what the re-run derived. The frozen protocol and the DiffRIR indices are
    checked on top, because the finalizer knows nothing about exp_02's protocol.
    """
    job_dir, expect = Path(job_dir), EXPECT_OF[job]
    record = job_completion(job_dir, expect, arm)
    owner = job_owner(job_dir, record)
    recipe = []
    spec_path = Path(record['job_spec']['path'])
    _require(spec_path.is_file(), 'missing job spec {}'.format(spec_path))
    _require(provenance.sha256_file(spec_path) == record['job_spec']['sha256'],
             'the job spec {} is not the bytes job {} bound'.format(spec_path, job_dir))
    spec = finalizer.load_job_spec(str(spec_path), expect)
    children, rooms = {}, {}
    for name in sorted(finalizer.expected_children(expect)):
        path, role = job_dir / name, finalizer.child_role(name)
        completion = path / 'completion.json'
        _require(completion.is_file(),
                 'job {} child {} has no completion.json'.format(job_dir, name))
        _require(provenance.sha256_file(completion) == record['children'][name],
                 'job {} child {} is not the completion it bound'.format(job_dir, name))
        evidence = dict(finalizer.verify_child(path, name, repo, spec), role=role)
        bound = _read_json(completion, name + '/completion.json')
        _require(bound.get('admissible_arm') is True,
                 'child {} is not an admissible arm'.format(name))
        args = _read_json(path / 'args.json', name + '/args.json')
        child_protocol(args, name, role)
        recipe.extend(child_recipe(args, name, role))
        if role == 'haa_eval':
            per = child_per_sample(job_dir, name, bound, arm)
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


def load_new_arm(root, arm, init_sha256=None, repo=REPO, approved=None,
                 sensitivity=False):
    """One exp_06 arm: four verified jobs, one closure per role, one heading per room."""
    base = Path(root) / Path(ARMS[arm]['root']).name
    _require(base.is_dir(), 'missing arm directory {}'.format(base))
    jobs, records, children, inputs, recipe = {}, {}, {}, {}, []
    for job in JOBS:
        job_dir = base / job
        _require(job_dir.is_dir(), 'the arm {} has no job {}'.format(arm, job))
        verified = verify_job(job_dir, job, arm, repo, sensitivity)
        recipe.extend('{} {}: {}'.format(arm, job, item)
                      for item in verified['recipe_deviations'])
        record = verified['record']
        _require(init_sha256 is None or record['init_sha256'] == init_sha256,
                 'the arm {} job {} did not start from the registered initialisation '
                 '{}'.format(arm, job, init_sha256))
        _require(job not in SEEDS or record['seed'] == int(job[len('seed'):]),
                 'the arm {} job {} records seed {!r}'.format(arm, job, record['seed']))
        jobs[job], records[job] = verified['per'], record
        inputs[verified['owner']['path']] = verified['owner']['sha256']
        completion = job_dir / 'completion.json'
        inputs[str(completion.resolve())] = provenance.sha256_file(completion)
        inputs[str(Path(record['job_spec']['path']).resolve())] = record['job_spec']['sha256']
        for name, digest in sorted(record['children'].items()):
            inputs[str((job_dir / name / 'completion.json').resolve())] = digest
            children['{}/{}'.format(job, name)] = verified['children'][name]
    closures, headings = arm_closures(children), arm_headings(children)
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


def screen_cells(arms, pair, n_boot=N_BOOT, adjusted_n_boot=N_BOOT_ADJUSTED, alpha=ALPHA):
    """H2: the eleven room x metric cells, nominal and Bonferroni adjusted."""
    treatment, comparator = pair
    adjusted_alpha = bootstrap.bonferroni_alpha(alpha, H2_FAMILY)
    cells = []
    for room, metric in CELLS:
        rows = cell_rows(arms, treatment, comparator, room, metric)
        cell = {'contrast': '{} - {}'.format(treatment, comparator), 'room': room,
                'metric': metric, 'alpha': alpha, 'adjusted_alpha': adjusted_alpha,
                'family': H2_FAMILY, 'cohort': rows['cohort'], 'n_test': rows['n_test'],
                'excluded': rows['excluded'], 'per_seed_diff': rows['per_seed_diff']}
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
            inputs[str((root / name).resolve())] = provenance.sha256_file(root / name)
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
        inputs.update(receipt.get('inputs') or {})
        if receipt_path is not None:
            inputs[str(Path(receipt_path).resolve())] = receipt['sha256']
    for arm in sorted(arms):
        inputs.update(arms[arm].get('inputs') or {})
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


def analyse(arms, n_boot=N_BOOT, adjusted_n_boot=N_BOOT_ADJUSTED, cache_root=None,
            exploratory=False, receipt=None, receipt_path=None, approved=None,
            deviations=(), approvals_receipt=None, producer=None, extra_inputs=(),
            sensitivity=False):
    """Every displayed number, and the evidence each rests on."""
    cache_inputs = {}
    result = {'schema_version': 1, 'exploratory': bool(exploratory),
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
              'side_split': side_split(arms, cache_root, inputs=cache_inputs)}
    result['inputs'] = dict(arm_inputs(arms, receipt, receipt_path),
                            **dict(cache_inputs, **dict(extra_inputs)))
    result['H1'] = decision_cell(arms, H1, H1_ROOM, H1_METRIC, H1_MARGIN_DB, n_boot)
    result['H1b'] = decision_cell(arms, H1B, H1_ROOM, H1_METRIC, 0.0, n_boot)
    result['H2'] = screen_cells(arms, H1, n_boot, adjusted_n_boot)
    result['D'] = descriptive_cells(arms, DESCRIPTIVE, n_boot)
    if exploratory:
        for name in ('H1', 'H1b'):
            result[name]['verdict'] = 'suppressed (draft)'
    if sensitivity:      # finding 3: a relaxed admission never reads as the primary one
        for name in ('H1', 'H1b'):
            result[name]['verdict'] = 'sensitivity: ' + result[name]['verdict']
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
    for name in ('H1', 'H1b'):
        cell = result[name]
        lines.append('\n{} {} {} {}: diff {}, two-way {}, margin {}, cohort {}/{} -> '
                     '{}'.format(name, cell['contrast'], cell['room'], cell['metric'],
                                 _number(cell['diff']), _bounds(cell['two_way']),
                                 cell['margin'], cell['cohort'], cell['n_test'],
                                 cell['verdict']))
        lines.extend('  void: ' + reason for reason in cell['void_reasons'])
    lines.append('\nH2 screen ({} cells, adjusted at alpha/{})'.format(len(result['H2']),
                                                                      H2_FAMILY))
    for cell in result['H2']:
        lines.append('  {:14s} {:4s} diff {} nominal {} adjusted {} -> {}'.format(
            cell['room'], cell['metric'], _number(cell['diff']),
            _bounds(cell['nominal_two_way']),
            'not available' if cell['adjusted_two_way'] is None else cell['adjusted_two_way'],
            cell['label']))
    lines.append('\nDescriptive contrasts')
    for cell in result['D']:
        lines.append('  {:22s} {:14s} {:4s} diff {} {}'.format(
            cell['contrast'], cell['room'], cell['metric'], _number(cell['diff']),
            _bounds(cell['nominal_two_way'])))
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
    parser.add_argument('--legacy-root', default=LEGACY_ROOT)
    parser.add_argument('--new-root', default=NEW_ROOT)
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
    binding = approved['reused']['legacy_receipt'] if approved else None
    if binding is not None and binding.get('sha256') is None:
        binding = None
    arms, receipt = load_legacy(args.legacy_root, args.legacy_receipt, binding)
    extra = check_reused_identities(None if args.exploratory else approved, receipt,
                                    args.gate_g1)
    inits = {} if args.exploratory else expected_inits(approved)
    for arm in NEW_ARMS:
        arms[arm] = load_new_arm(args.new_root, arm, inits.get(arm), REPO,
                                 None if args.exploratory else approved, args.sensitivity)
        deviations = deviations + list(arms[arm].get('recipe_deviations') or ())
    result = analyse(arms, args.n_boot, args.n_boot_adjusted, args.cache_root,
                     args.exploratory, receipt, args.legacy_receipt, approved, deviations,
                     approvals_receipt, identity, extra, args.sensitivity)
    record, digest, text = write_outputs(result, args.json, args.summary)
    print(text)
    print(json.dumps({'json': args.json, 'sha256': digest,
                      'summary_sha256': record['summary_sha256'],
                      'H1': record['H1']['verdict'], 'H1b': record['H1b']['verdict']}))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit('refusing: ' + str(error))
