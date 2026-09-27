"""Completion evidence for exp_11 runs, dispatched by exp_11's own run types.

Plan v3 item 7 and Codex round-2 change 4. ``tools/exp06_finalize.py`` is frozen and is
never edited or monkeypatched: its validation helpers are *imported* where they fit
(log closing, exit receipts, liveness, identity schemas, metric recomputation, the
budget/type-strict comparisons), and exp_11 owns everything that depends on exp_06's
module globals -- the registry, the recipe, the entry modules, the approvals record, the
run types and the heading rules.

**Serialized run types** (what ``--run-type`` takes here) are distinguished from the
**semantic roles** the protocol checks use. An exp_06 child records ``haa_train``; an
exp_11 child records ``exp11_haa_train``, so neither finalizer can certify the other's
children:

===================== ========================= =========================
``--run-type``        provenance ``run_type``    entry module
===================== ========================= =========================
``exp11_train``       ``full``                   ``tools.exp11_train``
``exp11_smoke``       ``smoke``/``probe``/       ``tools.exp11_train`` or
                      ``haa_smoke_*``            the HAA entry points
``exp11_haa_finetune````exp11_haa_train``        ``tools.exp11_haa_finetune``
``exp11_haa_eval``    ``exp11_haa_eval``         ``tools.exp11_haa_eval``
``exp11_haa_job``     (a job runs no child)      --
===================== ========================= =========================

**The completion contract is identical to exp_06's.** Every non-job run type requires:
no live ``launch.pid``/``child.pid``; a log whose last line is the ``EXP06_CHILD_EXIT``
marker (the pinned closer is reused, so the marker text is the pinned one); a
``child_exit.json`` receipt binding exactly those bytes; ``child exit 0``; a re-hash of
the log after all validation (a "stale log" is refused); the entry module's closure
recomputed and agreeing three ways (working tree now, bytes at spawn, reviewed blob);
``tools.provenance.revalidate`` over the declared inventory and mutable inputs; and the
approvals **committed** at the child's own reviewed commit.

``exp11_train`` additionally requires: the twelve-epoch recipe of the profile the run's
own arguments select (``tools.exp11_recipe.select_profile``) on all three recorded
copies of the arguments (``args.json``, ``last.pth['args']``, ``provenance.effective_args``),
compared type-strictly; each ``exp11_*`` field bound to the execution record; the
training-data, geometry and held-out inventories; epochs 1..12 recorded exactly once
with finite losses; and ``epoch_012.pth`` equal to ``last.pth['model']`` in keys, dtypes,
shapes and values. The sha256 of every artifact is recorded, and completion is external:
the trainer never writes it, and promotion happens only after it exists.

``exp11_haa_finetune`` requires: validation-based selection (epoch 0's init loss, the
``val_every`` cadence, the final epoch, strict improvement, ``best_epoch`` achieving the
minimum) reconciled with ``summary.json``; ``init_sha256`` equal to the hash of the
recorded initialisation; checkpoints carrying this arm's parameter names; and the frame
binding below. ``exp11_haa_eval`` requires: one room equal to its own directory; a
per-sample ``meta`` whose frame, backbone, protocol and ``checkpoint_sha256`` are the
arguments'; ``ir_path`` entries ``<room>/<index>``; ``side_label`` of -1/1; and a
``metrics_<room>.json`` recomputed from the observations. ``exp11_haa_job`` requires
every expected child directory, each re-validated by running its own role validator
again, the job spec, the owner pid, and the init -> stage 1 -> stage 2 -> evaluation
lineage.

**Heading binding.** Arms H/I run in the *heading frame* and bind one heading record per
room exactly as exp_06 does. Arms J/K run in the *room frame* with the shared adapter
heading: they record ``adapter_heading`` (the same per-room records) and
``adapter_phi_deg``; every bound record must be confirmatory, must still hash the cache
it was estimated from, and **all rooms must declare the same heading** -- a mixed heading
is refused, because the adapter installs one cue for every room.

    python tools/exp11_finalize.py --run-dir <dir> --run-type exp11_train \
        --log <log> --child-exit 0 [--repo <path>] [--receipt <json>] \
        [--children <dir>...] [--expect finetune|zeroshot] [--job-spec <json>] \
        [--owner-pid <pid>]
    python tools/exp11_finalize.py preflight --mode full --gpu 1 --reviewed-commit <sha> ...
    python tools/exp11_finalize.py child-exit --run-dir <dir> --log <log> ...
    python tools/exp11_finalize.py passed --run-dir <dir> --run-type exp11_smoke

Every failure raises ``ValueError`` naming its cause and writes nothing; a re-run
produces byte-identical bytes and an existing completion that differs is refused.
"""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from model.xRIR_simple_adapter import ADAPTER_KEYS
from model.xrir_exp11_registry import BACKBONES_EXP11, build_xrir_exp11
from sim_to_real.haa_dataset import ROOMS
from tools import exp06_finalize as base
from tools import exp06_heading, exp11_profiles, exp11_recipe, exp11_smoke, provenance
from tools.exp06_finalize import (_is_sha256, _load_torch, _mapping, _read_json, _require,
                                  _resolve, _state_dict, artifacts, closure_digest,
                                  closure_paths, confined)

REPO = Path(__file__).resolve().parents[1]
RUN_TYPES = ('exp11_train', 'exp11_smoke', 'exp11_haa_finetune', 'exp11_haa_eval',
             'exp11_haa_job')
DIAGNOSTIC = ('exp11_smoke',)
HAA_CHILD_TYPES = ('exp11_haa_finetune', 'exp11_haa_eval')
# The run_type each child records in its own provenance.json: exp_11's, never exp_06's.
PROVENANCE_RUN_TYPE = {'exp11_train': 'full', 'exp11_haa_finetune': 'exp11_haa_train',
                       'exp11_haa_eval': 'exp11_haa_eval'}
ENTRY_MODULES = {'exp11_train': 'tools.exp11_train',
                 'exp11_haa_finetune': 'tools.exp11_haa_finetune',
                 'exp11_haa_eval': 'tools.exp11_haa_eval',
                 'exp11_smoke': exp11_smoke.RUNNER}
HAA_CODE_KEY = {'exp11_haa_finetune': 'haa_finetune', 'exp11_haa_eval': 'haa_eval'}
# The exp_11 run type each *semantic* role serialises to. The protocol checks keep using
# the path-derived semantic roles ('haa_train'/'haa_eval'); only the recorded run type
# and the validator are exp_11's (Codex round-2 change 4).
SERIALIZED_ROLE = {'haa_train': 'exp11_haa_finetune', 'haa_eval': 'exp11_haa_eval'}
FRAMES = ('room', 'heading')
HEADING_FIELDS = ('heading', 'adapter_heading')
WIDTH = 512
EPOCH_CHECKPOINT = 'epoch_{:03d}.pth'.format(exp11_recipe.NUMERICAL['epochs'])
FULL_ARTIFACTS = ('provenance.json', 'args.json', 'history.jsonl', 'last.pth',
                  EPOCH_CHECKPOINT)
HAA_TRAIN_ARTIFACTS = ('provenance.json', 'args.json', 'history.jsonl', 'summary.json',
                       'best.pth', 'last.pth')
EXPECTATIONS = ('finetune', 'zeroshot')
LAUNCH_MODES = ('smoke', 'probe', 'full', 'finalize')
EXCLUSIVE_GPU_MODES = ('probe', 'full')


def registry_sha256():
    """The digest of exp_11's registry as it stands at finalisation."""
    mapping = {name: cls.__module__ + '.' + cls.__qualname__
               for name, cls in BACKBONES_EXP11.items()}
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()


@functools.lru_cache(maxsize=None)
def expected_state_keys(backbone, num_shot):
    """The parameter names a checkpoint of this arm must carry, and nothing else."""
    with torch.random.fork_rng(devices=[]):
        return frozenset(build_xrir_exp11(backbone, num_shot).state_dict().keys())


def _checkpoint_keys(path, label, backbone, num_shot):
    """Refuse a checkpoint that is not this arm's model, before anything hashes it."""
    state = _state_dict(_load_torch(path, label), label)
    expected = expected_state_keys(backbone, num_shot)
    _require(frozenset(state) == expected, '{}: {} parameters are not the {} of '
             'build_xrir_exp11({!r}, {})'.format(label, len(state), len(expected),
                                                 backbone, num_shot))


# The pinned model an adapter arm's ZERO-SHOT evaluation actually loads: arms J and K
# evaluate the historical A/E checkpoint before any fine-tuning, and ``load_base_checkpoint``
# reads it as the base xRIR. Admission applies the same contract, or those jobs -- which
# Phase 1b and the final publication require -- could never complete.
BASE_BACKBONE = 'simple'


def checkpoint_contract(args, digest):
    """``'base'`` for an adapter zero-shot evaluation, ``'adapter'`` for everything else.

    The zero-shot child is the one that evaluates the very initialisation its job spec
    declared at launch, so ``checkpoint_sha256 == job_init_sha256`` identifies it -- the
    same equality ``job_lineage`` requires of every child of a ``zeroshot`` job, and the
    one ``check_job_identity`` ties to the spec. Anything else, including a fine-tuned
    evaluation whose checkpoint lost its adapter, stays on the strict adapter path.
    """
    if args.get('backbone') != ADAPTER_BACKBONE:
        return 'adapter'                       # the arm's own model, whatever it is
    declared = args.get('job_init_sha256')
    return 'base' if _is_sha256(declared) and declared == digest else 'adapter'


def verify_evaluation_checkpoint(path, label, args, digest):
    """Validate one evaluation checkpoint under the contract its context selects.

    Base mode requires **exactly** the pinned ``xRIR`` parameter set and no adapter state
    at all; adapter mode requires the complete state of the arm's own model. Returns the
    contract that was applied, which the completion records.
    """
    mode = checkpoint_contract(args, digest)
    num_shot = args['num_shot']
    if mode == 'base':
        state = _state_dict(_load_torch(path, label), label)
        expected = expected_state_keys(BASE_BACKBONE, num_shot)
        _require(frozenset(state) == expected, '{}: {} parameters are not the {} of '
                 'build_xrir_exp11({!r}, {}), the base checkpoint an adapter zero-shot '
                 'evaluation loads'.format(label, len(state), len(expected),
                                           BASE_BACKBONE, num_shot))
        carried = sorted(key for key in ADAPTER_KEYS if key in state)
        _require(not carried, '{}: a base checkpoint carries no adapter state, but this '
                 'one records {}'.format(label, ', '.join(carried)))
    else:
        _checkpoint_keys(path, label, args['backbone'], num_shot)
    return mode


def load_provenance(run_dir, run_type, identity=True):
    """exp_06's whole-record schema, required to carry exp_11's own run type."""
    return base.load_provenance(run_dir, PROVENANCE_RUN_TYPE[run_type], identity=identity)


def verify_source_closure(record, run_type, repo):
    """Three ways -- the working tree now, the bytes at spawn, the reviewed blobs.

    exp_06's function reads its own ``ENTRY_MODULES`` global, which exp_11 may neither
    extend nor monkeypatch, so the entry identity is decided here against exp_11's map
    and the file comparison below is that function's, step for step.
    """
    closures = record['source_closures']
    _require(isinstance(closures, dict) and len(closures) == 1,
             'provenance.json must record exactly one source closure')
    name, closure = sorted(closures.items())[0]
    _require(isinstance(closure, dict), 'source closure {} is not a record'.format(name))
    entry = closure.get('entry_module')
    _require(entry == ENTRY_MODULES[run_type], 'source closure entry module is {!r}, not '
             'the {!r} of {}'.format(entry, ENTRY_MODULES[run_type], run_type))
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
    _require(sorted(current) == sorted(item['path'] for item in files),
             'source closure membership changed: '
             + ', '.join(sorted(set(current) ^ {item['path'] for item in files})))
    fresh, digest = provenance.closure_record(list(current), record['reviewed_commit'], repo)
    _require(digest == closure['sha256'],
             'source closure drift: the reviewed blobs differ from the recorded digest')
    now = {item['path']: item for item in fresh}
    for item in files:
        seen = now[item['path']]
        agreed = {seen['working_tree_sha256'], item['working_tree_sha256'],
                  item['reviewed_blob_sha256'], seen['reviewed_blob_sha256']}
        _require(len(agreed) == 1, 'source drift at {}: working tree {}, recorded {}, '
                 'reviewed {}'.format(item['path'], seen['working_tree_sha256'],
                                      item['working_tree_sha256'],
                                      item['reviewed_blob_sha256']))
    return name, closure


def verify_orchestration(record, repo, commit, recorded_digests):
    """The exp_11 launcher shell and the exp_11 finalizer, bound three ways as files."""
    closures = _mapping(record.get('orchestration_closures') or {},
                        'provenance.json orchestration_closures')
    _require(set(closures) == {'launcher', 'finalizer'},
             'provenance.json orchestration_closures must bind the launcher and finalizer')
    for role, key in (('launcher', 'launch_sh'), ('finalizer', 'finalize')):
        closure = _mapping(closures[role], 'orchestration_closures.' + role)
        files = base._closure_files(closure.get('files'), role)
        _require(closure.get('sha256') == closure_digest(files),
                 'recorded {} digest does not match its own file list'.format(role))
        _require(closure['sha256'] == recorded_digests.get(key),
                 'recorded {} digest is not the code_digests {}'.format(role, key))
        fresh, _ = exp11_profiles.closure_of(key, str(Path(repo).resolve()), commit)
        now = {item['path']: item for item in fresh}
        _require(sorted(now) == sorted(item['path'] for item in files),
                 '{} closure membership changed'.format(role))
        for item in files:
            seen = now[item['path']]
            agreed = {seen['working_tree_sha256'], item['working_tree_sha256'],
                      item['reviewed_blob_sha256'], seen['reviewed_blob_sha256']}
            _require(len(agreed) == 1, 'orchestration drift at ' + item['path'])
    return {role: closures[role]['sha256'] for role in closures}


def verify_approvals(record, repo, run_type):
    """exp_11's training-critical code identities, three ways, at the reviewed commit."""
    exploratory = bool(record.get('exploratory'))
    _require(not exploratory or run_type in DIAGNOSTIC,
             'an exploratory run is never admissible as an arm')
    recorded = record.get('code_digests')
    _require(isinstance(recorded, dict)
             and set(recorded) == set(exp11_profiles.TRAINING_KEYS),
             'provenance.json records no code_digests for exp_11 training-critical keys')
    commit = record['reviewed_commit']
    current = exp11_profiles.compute_code_digests(repo, commit,
                                                  keys=exp11_profiles.TRAINING_KEYS)
    drift = sorted(key for key in exp11_profiles.TRAINING_KEYS
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
    approved, identity = exp11_profiles.load_approved_digests(
        path, repo=None if exploratory else repo, commit=None if exploratory else commit)
    deviations = exp11_profiles.require(approved, exp11_profiles.TRAINING_KEYS, repo=repo,
                                        commit=commit, exploratory=exploratory,
                                        current=current)
    return {'approvals': dict(approvals, git_free_sha256=identity['sha256'],
                              committed_at=identity.get('committed_at')),
            'code_digests': dict(recorded), 'orchestration_digests': orchestration,
            'approval_deviations': deviations, 'exploratory': exploratory}


def haa_approvals(closure, run_type, commit, repo, path=None):
    """The HAA child ran the approved entry point, at its own reviewed commit."""
    path = Path(repo) / exp11_profiles.APPROVED_RELATIVE if path is None else Path(path)
    approved, identity = exp11_profiles.approvals_at_commit(path, repo, commit)
    key = HAA_CODE_KEY[run_type]
    pinned = approved['code'][key]
    _require(pinned is not None, 'code.{} is not approved at {}'.format(key, commit))
    _require(closure['sha256'] == pinned, 'the child ran the {} closure {}, not the approved '
             'code.{} {}'.format(run_type, closure['sha256'], key, pinned))
    return {'approvals': {'path': str(path), 'sha256': identity['sha256'],
                          'committed_at': identity.get('committed_at')},
            'code_digests': {key: pinned}}


def check_argument_sources(record, args, last, rows, meta, profile):
    """The startup, retained and checkpoint arguments must all agree, type-strictly.

    Each copy is validated on its own against exp_11's schema under the profile the
    run's own arguments select, then compared field by field with
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
        deviations = exp11_recipe.check_all(sources[name], profile)
        _require(not deviations, '{} schema deviations: {}'.format(name, '; '.join(deviations)))
    budget = exp11_recipe.check_budget(rows, meta, epochs=exp11_recipe.NUMERICAL['epochs'])
    _require(not budget, 'budget deviations: ' + '; '.join(budget))
    disagreements = exp11_recipe.compare_sources(sources)
    _require(not disagreements, 'recorded arguments disagree: ' + '; '.join(disagreements))
    return sources


def check_exp11_bindings(record, args, closure, run_dir, run_type, repo, profile):
    """Every ``exp11_*`` field must equal the execution record it claims to bind."""
    registry = registry_sha256()
    _require(record.get('registry_sha256') == registry,
             'provenance registry_sha256 {!r} is not the {} of BACKBONES_EXP11 at '
             'finalisation'.format(record.get('registry_sha256'), registry))
    expected = {'exp11_run_type': PROVENANCE_RUN_TYPE[run_type], 'exp11_profile': profile,
                'exp11_git_head': record['git_state'].get('HEAD'),
                'exp11_registry_sha256': registry,
                'exp11_source_closure_sha256': closure['sha256']}
    for field in sorted(expected):
        _require(args.get(field) == expected[field], 'args {} is {!r}, not the {!r} of the '
                 'execution record'.format(field, args.get(field), expected[field]))
    path = args.get('exp11_provenance_path')
    _require(isinstance(path, str) and path, 'args exp11_provenance_path is {!r}'.format(path))
    _require(_resolve(path, repo).resolve() == (Path(run_dir) / 'provenance.json').resolve(),
             'args exp11_provenance_path {} is not the validated {}'.format(
                 path, Path(run_dir) / 'provenance.json'))


def full_evidence(run_dir, repo):
    """exp_11's twelve-epoch pretraining contract, under the profile the run selects."""
    run_dir = Path(run_dir)
    hashes = artifacts(run_dir, FULL_ARTIFACTS)   # every file exists before it is parsed
    record = load_provenance(run_dir, 'exp11_train')
    _, closure = verify_source_closure(record, 'exp11_train', repo)
    admission = verify_approvals(record, repo, 'exp11_train')
    base.verify_train_identity(record)
    admission.update(base.verify_geometry_identity(record))
    admission.update(base.verify_heldout_identity(record))
    base.revalidate_inputs(record, repo)
    args = _read_json(run_dir / 'args.json', 'args.json')
    profile = exp11_recipe.select_profile(args)
    rows = base._history_rows(run_dir / 'history.jsonl', 'history.jsonl')
    last = _load_torch(run_dir / 'last.pth', 'last.pth')
    meta = {key: last[key] for key in ('epoch', 'batch_idx') if key in last}
    check_argument_sources(record, args, last, rows, meta, profile)
    check_exp11_bindings(record, args, closure, run_dir, 'exp11_train', repo, profile)
    _require('model' in last, 'last.pth records no "model" state dict')
    state = _state_dict(_load_torch(run_dir / EPOCH_CHECKPOINT, EPOCH_CHECKPOINT),
                        EPOCH_CHECKPOINT)
    model = _state_dict(last['model'], 'last.pth["model"]')
    _require(set(state) == set(model),
             EPOCH_CHECKPOINT + ' has a different parameter set than last.pth')
    for key in sorted(state):
        _require(state[key].dtype == model[key].dtype and
                 tuple(state[key].shape) == tuple(model[key].shape),
                 '{} has {}{} at {}, not the {}{} of last.pth["model"]'.format(
                     EPOCH_CHECKPOINT, state[key].dtype, tuple(state[key].shape), key,
                     model[key].dtype, tuple(model[key].shape)))
    _require(all(torch.equal(state[key], model[key]) for key in state),
             EPOCH_CHECKPOINT + ' differs tensor-wise from last.pth["model"]')
    _checkpoint_keys(run_dir / EPOCH_CHECKPOINT, EPOCH_CHECKPOINT, args['backbone'],
                     args['num_shot'])
    return dict(artifacts=hashes, epochs=len(rows), profile=profile, **admission,
                backbone=args['backbone'], source_closure_sha256=closure['sha256'],
                registry_sha256=record.get('registry_sha256'),
                git_head=record.get('git_state', {}).get('HEAD'),
                checkpoint={'path': EPOCH_CHECKPOINT, 'sha256': hashes[EPOCH_CHECKPOINT],
                            'epoch': exp11_recipe.NUMERICAL['epochs']})


ADAPTER_BACKBONE = 'simple_adapter'
# The backbones whose cue is the frame itself; the entry point refuses them without a
# heading directory, and the finalizer refuses the same pairings from the other side.
HEADING_BACKBONES = ('cylindrical_oriented', 'simple_oriented')
HEADING_DECISIONS = base.HEADING_DECISIONS
EVAL_PROTOCOL = base.EVAL_PROTOCOL


def heading_records(args, field, rooms, repo):
    """Every room's binding under ``field``, re-read against the cache the run recorded.

    This is exp_06's ``_heading_binding`` body with the record's field name as a
    parameter, because exp_11's room-frame adapter arms bind their cue under
    ``adapter_heading`` while the heading-frame arms bind it under ``heading``. The
    record is re-read against ``haa_root/<room>``, so ``read_heading_json`` rehashes the
    four cache inputs the estimate was derived from and only a ``confirmatory`` record
    may bind a child.
    """
    heading = args.get(field)
    rooms = sorted(set(rooms) | set(base._validation_rooms(args, rooms)))
    _require(isinstance(heading, dict) and set(rooms) <= set(heading),
             'the {} requires a record for every room the child trains or validates on: '
             '{}'.format(field, ', '.join(sorted(set(rooms) - set(heading or ())))))
    root = args.get('haa_root')
    _require(isinstance(root, str) and root, 'the {} requires the resolved haa_root the '
             'run read, not {!r}'.format(field, root))
    cache = _resolve(root, repo)
    _require(cache.is_dir(), 'missing HAA cache root {} (args.json haa_root)'.format(cache))
    bound = {}
    for room in rooms:
        entry = heading[room]
        _require(isinstance(entry, dict), '{} for {} is not a record'.format(field, room))
        phi = base._finite('{} for {}: phi_deg'.format(field, room), entry.get('phi_deg'))
        k = entry.get('k')
        _require(type(k) is int and 0 <= k < WIDTH,
                 '{} for {}: k {!r} is not a column in [0, {})'.format(field, room, k, WIDTH))
        _require(k == exp06_heading.heading_roll_k(phi),
                 '{} for {}: k {} is not the roll of {} degrees'.format(field, room, k, phi))
        _require(entry.get('decision') in HEADING_DECISIONS,
                 '{} for {} must be estimated or override, not {!r}'.format(
                     field, room, entry.get('decision')))
        _require(_is_sha256(entry.get('sha256')),
                 '{} for {} records no sha256'.format(field, room))
        path = entry.get('path')
        _require(isinstance(path, str) and path,
                 '{} for {} records no json path'.format(field, room))
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


def frame_binding(args, rooms, frame, repo):
    """The cue this child ran under: a heading frame, or the room frame with an adapter.

    Plan v3 section 2.3: the adapter installs **one** cue for every room, so every bound
    record must declare the same heading and ``adapter_phi_deg`` must be that heading.
    A mixed heading is refused rather than silently reduced to one of them.
    """
    backbone = args.get('backbone')
    if frame == 'heading':
        _require(backbone != ADAPTER_BACKBONE, 'the {} backbone conditions on its adapter, '
                 'never on the frame'.format(ADAPTER_BACKBONE))
        _require(not args.get('adapter_heading'),
                 'a heading-frame child conditions on the frame, not on an adapter')
        _require(args.get('adapter_phi_deg') is None,
                 'a heading-frame child installs no adapter heading')
        return heading_records(args, 'heading', rooms, repo), None, None
    _require(not args.get('heading'), 'the room frame must not record a heading')
    _require(backbone not in HEADING_BACKBONES, 'the {} backbone reads the heading frame; '
             'a room-frame child of it delivers no cue at all'.format(backbone))
    if backbone != ADAPTER_BACKBONE:
        _require(not args.get('adapter_heading') and args.get('adapter_phi_deg') is None,
                 'only the {} backbone conditions on an adapter heading'.format(
                     ADAPTER_BACKBONE))
        return None, None, None
    bound = heading_records(args, 'adapter_heading', rooms, repo)
    declared = sorted({entry['phi_deg'] for entry in bound.values()})
    _require(len(declared) == 1, 'the adapter installs one cue for every room, but the '
             'bound records declare the headings {}'.format(declared))
    phi = base._finite('args.json adapter_phi_deg', args.get('adapter_phi_deg'))
    _require(phi == declared[0], 'args.json installs the adapter heading {}, not the {} '
             'every bound record declares'.format(phi, declared[0]))
    return None, bound, float(phi)


def haa_child_arguments(run_dir, run_type, repo):
    """Provenance, closure, approvals and argument agreement, shared by both child types."""
    record = load_provenance(run_dir, run_type)
    _, closure = verify_source_closure(record, run_type, repo)
    admission = haa_approvals(closure, run_type, record['reviewed_commit'], repo)
    base.revalidate_inputs(record, repo, required=('source_closures',))
    args = _read_json(Path(run_dir) / 'args.json', 'args.json')
    disagreements = exp11_recipe.compare_sources(
        {'args.json': args, 'provenance.effective_args': record['effective_args']})
    _require(not disagreements, 'recorded arguments disagree: ' + '; '.join(disagreements))
    backbone = args.get('backbone')
    _require(backbone in BACKBONES_EXP11, 'args.json records backbone {!r}'.format(backbone))
    _require(type(args.get('num_shot')) is int and args['num_shot'] > 0,
             'args.json records no positive integer num_shot')
    _require(type(args.get('seed')) is int, 'args.json records no integer seed')
    registry = registry_sha256()
    _require(record['registry_sha256'] == registry, 'provenance registry_sha256 {!r} is not '
             'the {} of BACKBONES_EXP11 at finalisation'.format(
                 record['registry_sha256'], registry))
    return record, args, admission


def haa_train_evidence(run_dir, repo):
    """One exp_11 fine-tuning child: validation-based selection and the frame binding."""
    hashes = artifacts(run_dir, HAA_TRAIN_ARTIFACTS)
    record, args, admission = haa_child_arguments(run_dir, 'exp11_haa_finetune', repo)
    rooms, frame = base._rooms_and_frame(args)
    heading, adapter_heading, adapter_phi = frame_binding(args, rooms, frame, repo)
    epochs, summary = base.haa_history(run_dir, args)
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
    return dict(base.child_identity(record), artifacts=hashes, rooms=rooms, frame=frame,
                heading=heading, adapter_heading=adapter_heading,
                adapter_phi_deg=adapter_phi, backbone=args['backbone'],
                init_sha256=args['init_sha256'], seed=args['seed'],
                best_epoch=summary['best_epoch'], epochs=epochs, **admission)


def haa_eval_evidence(run_dir, repo):
    """One exp_11 evaluation child: exactly one room, bound to the checkpoint it ran."""
    record, args, admission = haa_child_arguments(run_dir, 'exp11_haa_eval', repo)
    rooms, frame = base._rooms_and_frame(args)
    _require(len(rooms) == 1, 'an evaluation child covers exactly one room, not {}'.format(rooms))
    room, tag = rooms[0], args.get('tag', '')
    _require(Path(run_dir).resolve().name == room, 'an evaluation child of {} may not claim '
             'the room {!r}'.format(Path(run_dir).resolve().name, room))
    names = ('provenance.json', 'args.json', 'metrics_{}{}.json'.format(room, tag),
             'per_sample_{}{}.json'.format(room, tag))
    hashes = artifacts(run_dir, names)
    heading, adapter_heading, adapter_phi = frame_binding(args, rooms, frame, repo)
    per_sample = _read_json(Path(run_dir) / names[3], names[3])
    meta = _mapping(per_sample.get('meta'), 'per-sample meta')
    required = (('backbone', 'checkpoint_sha256', 'frame') + EVAL_PROTOCOL
                + (('heading',) if heading else ())
                + (('adapter_heading', 'adapter_phi_deg') if adapter_heading else ()))
    missing = [key for key in required if key not in meta]
    _require(not missing, 'per-sample meta must record ' + ', '.join(missing))
    _require(meta['frame'] == frame and frame in FRAMES,
             'per-sample meta frame {!r} is not the {!r} of args.json'.format(meta['frame'], frame))
    _require(meta['backbone'] == args['backbone'],
             'per-sample meta backbone {!r} differs from args.json'.format(meta['backbone']))
    for field in EVAL_PROTOCOL:
        _require(exp11_recipe.strict_equal(meta[field], args.get(field)),
                 'per-sample meta {} {!r} is not the {!r} of args.json'.format(
                     field, meta[field], args.get(field)))
    _require(meta.get('room', room) == room, 'per-sample meta room {!r} is not the {!r} this '
             'child evaluated'.format(meta.get('room'), room))
    for field, bound in (('heading', heading), ('adapter_heading', adapter_heading)):
        if bound:
            _require(exp11_recipe.strict_equal(meta[field], args[field]),
                     'per-sample meta {} differs from the one bound in args.json'.format(field))
        else:
            _require(not meta.get(field), 'this child records no {}'.format(field))
            _require(field != 'adapter_heading' or meta.get('adapter_phi_deg') is None,
                     'this child installs no adapter heading, but its meta records '
                     '{!r}'.format(meta.get('adapter_phi_deg')))
    if adapter_heading:
        _require(exp11_recipe.strict_equal(meta['adapter_phi_deg'], args['adapter_phi_deg']),
                 'per-sample meta adapter_phi_deg differs from args.json')
    checkpoint = args.get('checkpoint')
    _require(isinstance(checkpoint, str) and checkpoint, 'args.json records no checkpoint path')
    resolved = _resolve(checkpoint, repo)
    _require(resolved.is_file(), 'missing evaluation checkpoint: {}'.format(resolved))
    digest = provenance.sha256_file(resolved)
    mode = verify_evaluation_checkpoint(resolved, 'checkpoint ' + checkpoint, args, digest)
    _require(meta['checkpoint_sha256'] == digest, 'per-sample meta checkpoint_sha256 {} is '
             'not the hash {} of {}'.format(meta['checkpoint_sha256'], digest, resolved))
    index, side = per_sample.get('index'), per_sample.get('side_label')
    _require(isinstance(index, list) and index, 'the per-sample file records no index')
    bad = [value for value in index if type(value) is not int or value < 0]
    _require(not bad, 'per-sample index entries must be non-negative integers, not {}'.format(
        sorted(map(repr, bad))[:4]))
    paths = per_sample.get('ir_path')
    _require(isinstance(paths, list) and len(paths) == len(index),
             'per-sample ir_path must record one <room>/<index> path per index')
    wrong = [path for path, value in zip(paths, index) if path != '{}/{}'.format(room, value)]
    _require(not wrong, 'per-sample ir_path entries are not the {}/<index> this child '
             'evaluated: {}'.format(room, sorted(map(repr, wrong))[:4]))
    _require(isinstance(side, list) and len(side) == len(index),
             'side_label must carry one room-frame label per index')
    bad = [value for value in side if isinstance(value, bool) or value not in (-1, 1)]
    _require(not bad, 'side_label values must be -1 or 1, not {}'.format(sorted(set(map(repr, bad)))))
    base.haa_metrics(run_dir, names[2], room, args, per_sample)
    return dict(base.child_identity(record), artifacts=hashes, room=room, frame=frame,
                heading=heading, adapter_heading=adapter_heading,
                adapter_phi_deg=adapter_phi, seed=args['seed'], backbone=args['backbone'],
                checkpoint_sha256=digest, checkpoint_mode=mode, samples=len(index),
                **admission)


CHILD_COMPLETION = ('schema_version', 'run_type', 'run_dir', 'child_exit', 'child_exit_time',
                    'log', 'child_exit_receipt', 'diagnostic', 'admissible_arm', 'artifacts',
                    'backbone', 'frame', 'heading', 'adapter_heading', 'adapter_phi_deg')
CHILD_EXTRA = {'exp11_haa_finetune': ('rooms', 'init_sha256', 'best_epoch', 'seed'),
               'exp11_haa_eval': ('room', 'checkpoint_sha256', 'checkpoint_mode',
                                  'samples', 'seed')}
JOB_SPEC = ('init', 'backbone', 'frame', 'init_sha256', 'seed', 'rooms', 'expect')
EVIDENCE_OF = {'exp11_haa_finetune': haa_train_evidence, 'exp11_haa_eval': haa_eval_evidence}


def child_role(name):
    """The exp_11 run type this finalizer requires at one child path of a pipeline seed."""
    return SERIALIZED_ROLE[base.child_role(name)]


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
    _require(record['run_type'] == role, 'child {} has run type {!r}, not the {!r} its path '
             'requires'.format(name, record['run_type'], role))
    _require(isinstance(record['run_dir'], str)
             and Path(record['run_dir']).resolve() == Path(path).resolve(),
             'child {} claims the run_dir {!r}'.format(name, record['run_dir']))
    _require(record['diagnostic'] is False, 'child {} is a diagnostic run'.format(name))
    _require(type(record['child_exit']) is int and record['child_exit'] == 0,
             'child {} records child_exit {!r}'.format(name, record['child_exit']))
    return record


def check_job_spec(name, role, evidence, spec, args):
    """Every child must have run the job the pipeline declared, not one of its own."""
    for field in ('backbone', 'frame', 'seed'):
        _require(evidence[field] == spec[field], 'child {} ran {} {!r}, not the {!r} of the '
                 'job spec'.format(name, field, evidence[field], spec[field]))
    rooms = evidence['rooms'] if role == 'exp11_haa_finetune' else [evidence['room']]
    outside = sorted(set(rooms) - set(spec['rooms']))
    _require(not outside, 'child {} covers rooms outside the job: {}'.format(name, outside))
    room = name.split('/')[-1] if role == 'exp11_haa_eval' else name[len('stage2_'):]
    _require(not name.startswith(('stage2_', 'eval/', 'zeroshot/')) or rooms == [room],
             'child {} records rooms {}, not the {!r} of its path'.format(name, rooms, room))
    field = 'heading' if spec['frame'] == 'heading' else 'adapter_heading'
    declared = spec.get(field) or {}
    if declared:
        for bound, binding in sorted((evidence[field] or {}).items()):
            _require(binding.get('k') == declared.get(bound),
                     'child {} rolls the {} {} to {!r}, not the {!r} of the job spec'.format(
                         name, bound, field, binding.get('k'), declared.get(bound)))
        _require(exp11_recipe.strict_equal(evidence.get('adapter_phi_deg'),
                                           spec.get('adapter_phi_deg')),
                 'child {} installs the adapter heading {!r}, not the {!r} of the job '
                 'spec'.format(name, evidence.get('adapter_phi_deg'),
                               spec.get('adapter_phi_deg')))
    else:
        for other in HEADING_FIELDS:
            _require(not evidence[other],
                     'child {} records a {} the job spec does not declare'.format(name, other))
    base.check_job_identity(name, args, spec)


def verify_child(path, name, repo, spec):
    """Re-run the child's own role validator and bind its record to that result."""
    base.refuse_live_launch(path)
    role = child_role(name)
    evidence = EVIDENCE_OF[role](path, repo)
    record = child_completion(path, name, role)
    recorded = _mapping(record['artifacts'], 'child {} artifacts'.format(name))
    fresh = evidence['artifacts']
    _require(set(recorded) == set(fresh), 'child {} records the artefacts {}, not the {} the '
             're-run hashed'.format(name, sorted(recorded), sorted(fresh)))
    for artefact in sorted(fresh):
        _require(exp11_recipe.strict_equal(recorded[artefact], fresh[artefact]),
                 'child {} artefact {} is not the one the re-run hashed'.format(name, artefact))
    for field in sorted(set(evidence) - {'artifacts'}):
        _require(field in record, 'child {} completion records no {}'.format(name, field))
        _require(exp11_recipe.strict_equal(record[field], evidence[field]),
                 'child {} completion {} {!r} is not the {!r} of the re-run'.format(
                     name, field, record[field], evidence[field]))
    base.rehash_bound_evidence(record, name, path, repo)
    check_job_spec(name, role, evidence, spec,
                   _read_json(Path(path) / 'args.json', 'args.json'))
    return evidence


def load_job_spec(path, expect):
    """The pipeline's declaration of one seed; every nested value is typed before use."""
    _require(path, 'a job needs the pipeline --job-spec it was run from')
    try:
        data = Path(path).read_bytes()
        value = json.loads(data.decode('utf-8'))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError('unreadable job spec: {}'.format(error)) from error
    spec = _mapping(value, 'job spec')
    missing = [key for key in JOB_SPEC if key not in spec]
    _require(not missing, 'job spec is incomplete: missing ' + ', '.join(missing))
    _require(spec['expect'] == expect,
             'job spec declares {!r}, not the --expect {!r}'.format(spec['expect'], expect))
    _require(spec['backbone'] in BACKBONES_EXP11,
             'job spec backbone {!r}'.format(spec['backbone']))
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
    cue = 'heading' if spec['frame'] == 'heading' else 'adapter_heading'
    other = 'adapter_heading' if cue == 'heading' else 'heading'
    _require(not spec.get(other), 'a {}-frame job spec declares no {}'.format(
        spec['frame'], other))
    if spec['frame'] == 'heading' or spec['backbone'] == ADAPTER_BACKBONE:
        rolls = _mapping(spec.get(cue) or {}, 'job spec ' + cue)
        absent = [room for room in spec['rooms'] if type(rolls.get(room)) is not int
                  or not 0 <= rolls[room] < WIDTH]
        _require(not absent, 'job spec records no {} roll in [0, {}) for {}'.format(
            cue, WIDTH, ', '.join(absent)))
    else:
        _require(not spec.get(cue), 'a room-frame job spec without an adapter declares no cue')
    if cue == 'adapter_heading' and spec.get(cue):
        base._finite('job spec adapter_phi_deg', spec.get('adapter_phi_deg'))
    else:
        _require(spec.get('adapter_phi_deg') is None,
                 'only an adapter job spec declares adapter_phi_deg')
    spec['job_spec_sha256'] = hashlib.sha256(data).hexdigest()
    _require(provenance.sha256_file(path) == spec['job_spec_sha256'],
             'job spec {} changed while it was being validated'.format(path))
    return spec


def job_lineage(records, expect, spec):
    """The init -> stage1 -> stage2 -> evaluation chain the job spec declares."""
    fields = dict(backbone=spec['backbone'], frame=spec['frame'], seed=spec['seed'],
                  heading=spec.get('heading') if spec['frame'] == 'heading' else None,
                  adapter_heading=spec.get('adapter_heading'),
                  adapter_phi_deg=spec.get('adapter_phi_deg'),
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
            _require(record['checkpoint_sha256']
                     == records['stage2_' + room]['artifacts']['best.pth'],
                     'lineage: {} did not evaluate stage2_{}/best.pth'.format(name, room))
    return fields


def haa_job_evidence(run_dir, children, expect, repo, job_spec):
    """Bind one seed: every expected child, re-validated in the role its path requires."""
    expected, job = set(base.expected_children(expect)), Path(run_dir).resolve()
    spec = load_job_spec(job_spec, expect)
    seen, records = {}, {}
    for child in children:
        path = Path(child).resolve()
        try:
            name = path.relative_to(job).as_posix()
        except ValueError as error:
            raise ValueError('child {} lies outside the job directory'.format(child)) from error
        if expect == 'zeroshot' and name.startswith('zeroshot/'):
            name = name[len('zeroshot/'):]
        _require(name in expected, 'unexpected child: ' + name)
        records[name] = verify_child(path, name, repo, spec)
        seen[name] = provenance.sha256_file(path / 'completion.json')
    missing = sorted(expected - set(seen))
    _require(not missing, 'job is missing children: ' + ', '.join(missing))
    return dict(job_lineage(records, expect, spec), artifacts={}, children=seen, expect=expect,
                job_spec={'path': str(Path(job_spec).resolve()),
                          'sha256': spec['job_spec_sha256']})


def haa_job_completion(run_dir, children, expect, repo, job_spec, log, child_exit, owner_pid):
    """One pipeline job: no child of its own, hence no exit receipt and no closed log."""
    _require(not (run_dir / 'child_exit.json').exists(),
             'job roots carry no child exit receipt (A3)')
    owner = base.refuse_live_launch(run_dir, owner_pid)
    _require(owner is not None,
             '{} records no launch.pid: a job binds the owner that ran it'.format(run_dir))
    _require(owner_pid is None or owner == owner_pid,
             'the job root holds launch.pid {}, not the declared owner {}'.format(
                 owner, owner_pid))
    _require(child_exit == 0, 'the job was declared with child status {}'.format(child_exit))
    fields = dict(schema_version=1, run_type='exp11_haa_job', run_dir=str(run_dir.resolve()),
                  repo=str(Path(repo).resolve()), child_exit=child_exit,
                  log=base.job_log(log), owner_pid=owner, diagnostic=False,
                  admissible_arm=True)
    fields.update(haa_job_evidence(run_dir, children, expect, repo, job_spec))
    return base.write_completion(run_dir / 'completion.json', fields)


SMOKE_ROOT = 'ckpt/exp11/_smoke'    # the disposable tree a finalised diagnostic may write in
ARTIFACT_DIR = 'run'                # the launcher's --save-dir inside a diagnostic run dir
DIAGNOSTIC_PROVENANCE = ('run_type', 'repo', 'reviewed_commit', 'source_closures',
                         'git_state', 'environment', 'command', 'registry_sha256')
DIAGNOSTIC_RECEIPT = ('runner', 'runner_closure_sha256', 'kind', 'entry', 'argv', 'run_type',
                      'exit_status', 'outcome', 'exploratory', 'started_at', 'ended_at',
                      'wall_s', 'peak_bytes', 'alarm_seconds', 'max_gb', 'admissible_arm')


def smoke_run_dir(run_dir, repo):
    """A finalised diagnostic lives in exp_11's disposable smoke tree and nowhere else."""
    root = _resolve(SMOKE_ROOT, repo).resolve()
    path = Path(run_dir).resolve()
    _require(path != root and root in path.parents,
             '{} is not inside the disposable smoke tree {}'.format(path, root))
    return path


def diagnostic_receipt(receipt):
    """The runner's identity, the enumerated kind, its budgets, timing and memory."""
    _require(receipt is not None, 'a diagnostic run needs its --receipt')
    path = Path(receipt)
    _require(path.is_file(), 'missing smoke receipt: {}'.format(receipt))
    record = _read_json(path, 'smoke receipt')
    _require(record.get('diagnostic') is True, 'the smoke receipt is not marked diagnostic')
    _require(record.get('admissible_arm') is False,
             'an exp_11 diagnostic is never an admissible arm; its receipt claims {!r}'.format(
                 record.get('admissible_arm')))
    missing = [key for key in DIAGNOSTIC_RECEIPT if key not in record]
    _require(not missing, 'the smoke receipt is incomplete: missing ' + ', '.join(missing))
    spec = exp11_smoke.kind_spec(record['kind'])
    _require(record['runner'] == exp11_smoke.RUNNER,
             'the smoke receipt records runner {!r}'.format(record['runner']))
    _require(_is_sha256(record['runner_closure_sha256']),
             'the smoke receipt records no runner closure digest')
    _require(record['entry'] == spec['entry'] and record['run_type'] == spec['run_type'],
             'the {} receipt must record the entry {!r} and run_type {!r}, not {!r}/{!r}'
             .format(record['kind'], spec['entry'], spec['run_type'], record['entry'],
                     record['run_type']))
    _require(record['outcome'] in exp11_smoke.OUTCOMES,
             'the smoke receipt records outcome {!r}'.format(record['outcome']))
    _require(type(record['exit_status']) is int and type(record['exploratory']) is bool,
             'the smoke receipt records a malformed exit_status/exploratory')
    argv = record['argv']
    _require(isinstance(argv, list) and all(isinstance(token, str) for token in argv),
             'the smoke receipt argv {!r} is not a list of strings'.format(argv))
    _require(record['kind'] != 'probe' or '--no-save' in argv,
             'a probe must run with --no-save; its receipt records argv {!r}'.format(argv))
    for key in ('wall_s', 'alarm_seconds', 'max_gb'):
        _require(base._finite('the smoke receipt ' + key, record[key]) >= 0,
                 'the smoke receipt {} is {!r}'.format(key, record[key]))
    _require(base._positive('the smoke receipt alarm_seconds', record['alarm_seconds'])
             <= spec['alarm_seconds'], 'the receipt claims a {} s alarm, above the {} the '
             '{} kind registers'.format(record['alarm_seconds'], spec['alarm_seconds'],
                                        record['kind']))
    _require(base._positive('the smoke receipt max_gb', record['max_gb']) <= spec['max_gb'],
             'the receipt claims a {} GiB ceiling, above the {} the {} kind registers'.format(
                 record['max_gb'], spec['max_gb'], record['kind']))
    _require(type(record['peak_bytes']) is int and record['peak_bytes'] >= 0,
             'the smoke receipt peak_bytes is {!r}'.format(record['peak_bytes']))
    ended = base._timestamp(record['ended_at'], 'ended_at')
    _require(ended >= base._timestamp(record['started_at'], 'started_at'),
             'the smoke receipt ended_at precedes its started_at')
    return record, path, spec


def check_receipt_consistency(fields, record, window):
    """An outcome must agree with its own status, resources and the launcher's window."""
    _require(record.get('command') == [fields['kind']] + fields['argv'],
             'the smoke receipt argv is not the child command {!r} its provenance '
             'recorded'.format(record.get('command')))
    _require(record.get('kind') == fields['kind'] and record.get('entry') == fields['entry'],
             'the smoke receipt and its provenance name different diagnostics')
    _require(record.get('admissible_arm') is False,
             'an exp_11 diagnostic provenance is never an admissible arm')
    status, outcome = fields['exit_status'], fields['outcome']
    if outcome == 'ok':
        _require(status == 0, 'outcome ok with exit_status {}'.format(status))
        _require(fields['peak_bytes'] <= fields['max_gb'] * base.GIB,
                 'outcome ok with a peak of {} bytes over its {} GiB budget'.format(
                     fields['peak_bytes'], fields['max_gb']))
        _require(fields['wall_s'] <= fields['alarm_seconds'],
                 'outcome ok after {} s over its {} s budget'.format(
                     fields['wall_s'], fields['alarm_seconds']))
    else:
        _require(status != 0, 'outcome {!r} with exit_status 0'.format(outcome))
    started = base._timestamp(fields['started_at'], 'started_at')
    ended = base._timestamp(fields['ended_at'], 'ended_at')
    _require(base._timestamp(window['started_at'], 'started_at') <= started,
             'the receipt started before the child the launcher spawned')
    _require(ended <= base._timestamp(window['ended_at'], 'ended_at')
             + base.datetime.timedelta(seconds=1),
             'the receipt ended after the child exited')
    return status == 0 and outcome == 'ok'


def diagnostic_artifacts(run_dir, spec, repo):
    """The kind's artifact contract, inside the disposable tree and nowhere else."""
    root = smoke_run_dir(run_dir, repo)
    if spec['artifacts'] == ('provenance.json',):
        return {}
    directory = confined(root / ARTIFACT_DIR, root, 'artefact directory')
    _require(directory.is_dir(), 'missing artefact directory {}'.format(directory))
    confined(directory / 'args.json', root, 'artefact')
    args = _read_json(directory / 'args.json', 'args.json')
    _require(_resolve(args.get('save_dir') or '', repo).resolve() == directory,
             'args.json records the save_dir {!r}, not the {} being finalized'.format(
                 args.get('save_dir'), directory))
    names = list(spec['artifacts'])
    if spec['run_type'] == 'exp11_haa_smoke_eval':
        rooms, tag = args.get('rooms'), args.get('tag', '')
        _require(isinstance(rooms, list) and rooms and all(room in ROOMS for room in rooms),
                 'args.json records the rooms {!r}'.format(rooms))
        _require(isinstance(tag, str), 'args.json records the tag {!r}'.format(tag))
        names.append('metrics_all{}.json'.format(tag))
        for room in rooms:
            names += ['metrics_{}{}.json'.format(room, tag),
                      'per_sample_{}{}.json'.format(room, tag)]
    unregistered = sorted(set(item.name for item in directory.iterdir()) - set(names))
    _require(not unregistered, '{} holds unregistered artefacts: {}'.format(
        directory, ', '.join(unregistered)))
    return artifacts(directory, sorted(names), root=root)


def diagnostic_evidence(run_dir, receipt, child_exit, repo, window):
    """A diagnostic proves nothing about an arm, but must prove what it cost."""
    fields, path, spec = diagnostic_receipt(receipt)
    record = base.load_provenance(run_dir, spec['run_type'],
                                  required=DIAGNOSTIC_PROVENANCE, identity=False)
    verify_source_closure(record, 'exp11_smoke', repo)
    admission = verify_approvals(record, repo, 'exp11_smoke')
    passed = check_receipt_consistency(fields, record, window) and child_exit == 0
    hashes = {}
    if spec['run_type'] != 'exp11_probe':
        _require(passed, 'a finalised HAA diagnostic must have succeeded: the receipt '
                 'records outcome {!r} with exit_status {}, and the child exited {}'.format(
                     fields['outcome'], fields['exit_status'], child_exit))
        hashes = diagnostic_artifacts(run_dir, spec, repo)
    return dict(artifacts=hashes, passed=passed, kind=fields['kind'], **admission,
                receipt={'path': str(path.resolve()), 'sha256': provenance.sha256_file(path),
                         'runner': fields['runner'], 'entry': fields['entry'],
                         'kind': fields['kind'], 'exit_status': fields['exit_status'],
                         'outcome': fields['outcome'], 'wall_s': fields['wall_s'],
                         'peak_bytes': fields['peak_bytes'],
                         'alarm_seconds': fields['alarm_seconds'],
                         'max_gb': fields['max_gb']})


def finalize(run_dir, run_type, log, child_exit, repo=REPO, receipt=None, children=(),
             expect=None, owner_pid=None, job_spec=None):
    """Verify one child's evidence for its exp_11 run type and write completion.json."""
    run_dir = Path(run_dir)
    _require(run_type in RUN_TYPES, 'unknown run type: {!r}'.format(run_type))
    _require(run_dir.is_dir(), 'run directory does not exist: {}'.format(run_dir))
    _require(type(child_exit) is int, 'child status must be an integer')
    if run_type == 'exp11_haa_job':   # A3: a job orchestrates children and is none itself
        return haa_job_completion(run_dir, children, expect, repo, job_spec, log, child_exit,
                                  owner_pid)
    base.refuse_live_launch(run_dir, owner_pid)
    log_record, child_exit_time, log_digest = base.closed_log(log, child_exit)
    receipt_record = base.child_exit_receipt(run_dir, child_exit, log_digest, child_exit_time)
    diagnostic = run_type in DIAGNOSTIC
    if not diagnostic:
        _require(child_exit == 0, 'child exited with status {}'.format(child_exit))
    fields = dict(schema_version=1, run_type=run_type, run_dir=str(run_dir.resolve()),
                  repo=str(Path(repo).resolve()), child_exit=child_exit,
                  child_exit_time=child_exit_time, log=log_record,
                  child_exit_receipt=receipt_record,
                  diagnostic=diagnostic, admissible_arm=not diagnostic)
    fields.update(diagnostic_evidence(run_dir, receipt, child_exit, repo, receipt_record)
                  if diagnostic
                  else full_evidence(run_dir, repo) if run_type == 'exp11_train'
                  else haa_train_evidence(run_dir, repo) if run_type == 'exp11_haa_finetune'
                  else haa_eval_evidence(run_dir, repo))
    _require(provenance.sha256_file(log) == log_digest,
             'stale log: {} changed while its completion was being validated'.format(log))
    return base.write_completion(run_dir / 'completion.json', fields)


def approval_gate(mode, reviewed_commit, approved, exploratory, repo):
    """No confirmatory launch on code exp_11's approvals do not pin."""
    _require(not (exploratory and mode == 'full'),
             'an exploratory launch is a diagnostic; mode full must match the approvals')
    bind = not (exploratory and mode in ('smoke', 'probe'))
    path = _resolve(exp11_profiles.APPROVED_RELATIVE if approved is None else approved, repo)
    _require(path.is_file(), 'missing approvals file: {}'.format(path))
    value, identity = exp11_profiles.load_approved_digests(
        path, repo=repo if bind else None, commit=reviewed_commit if bind else None)
    deviations = exp11_profiles.require(value, exp11_profiles.TRAINING_KEYS, repo=repo,
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
    running = base.live_launches(attempt_root)
    _require(not running, 'another exp_11 launch is alive: {}'.format(running))
    admission = approval_gate(mode, reviewed_commit, approved, exploratory, repo)
    apps = base.gpu_compute_apps(gpu) if mode in EXCLUSIVE_GPU_MODES else None
    _require(not apps, 'GPU {} is busy with compute apps {}'.format(gpu, apps))
    free = None if min_free_gb is None else base.gpu_free_gib(gpu)
    _require(free is None or free >= min_free_gb,
             'GPU {} has {:.2f} GiB free, below the {} GiB this launch may allocate'.format(
                 gpu, free or 0.0, min_free_gb))
    return dict(mode=mode, gpu=gpu, reviewed_commit=reviewed_commit, git_state=state,
                attempt_root=[str(root) for root in base.launch_roots(attempt_root)],
                gpu_compute_apps=apps, live_launches=running, min_free_gb=min_free_gb,
                gpu_free_gib=free, **admission)


def preflight_main(argv):
    """Exit 0 with the record on stdout, 2 with the named cause on stderr."""
    parser = argparse.ArgumentParser(description='Gate one exp_11 launch.')
    parser.add_argument('--mode', choices=LAUNCH_MODES, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--reviewed-commit', required=True)
    parser.add_argument('--attempt-root', action='append', default=[])
    parser.add_argument('--repo', default=str(REPO))
    parser.add_argument('--approved', default=None)
    parser.add_argument('--exploratory', action='store_true')
    parser.add_argument('--min-free-gb', type=float, default=None)
    args = parser.parse_args(argv)
    try:
        record = preflight(args.mode, args.gpu, args.reviewed_commit, args.attempt_root,
                           args.repo, approved=args.approved, exploratory=args.exploratory,
                           min_free_gb=args.min_free_gb)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print('EXP11_PREFLIGHT_REFUSED ' + str(error), file=sys.stderr, flush=True)
        return 2
    print('EXP11_PREFLIGHT_OK ' + json.dumps(record, sort_keys=True), flush=True)
    return 0


def passed_main(argv):
    """Exit 0 only where a diagnostic's own completion certifies that it passed."""
    parser = argparse.ArgumentParser(description='Gate the next rung on a diagnostic.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--run-type', choices=RUN_TYPES, required=True)
    args = parser.parse_args(argv)
    try:
        fields = _read_json(Path(args.run_dir) / 'completion.json', 'completion.json')
        _require(fields.get('run_type') == args.run_type,
                 'completion.json records run_type {!r}, not {}'.format(
                     fields.get('run_type'), args.run_type))
        _require(fields.get('passed') is True,
                 '{} did not pass: its completion records passed {!r}'.format(
                     args.run_dir, fields.get('passed')))
    except (OSError, ValueError) as error:
        print('EXP11_PASSED_REFUSED ' + str(error), file=sys.stderr, flush=True)
        return 2
    print('EXP11_PASSED_OK ' + str(args.run_dir), flush=True)
    return 0


def build_parser():
    """One finalization of one child or job; the launcher supplies the child's status."""
    parser = argparse.ArgumentParser(description='Write exp_11 completion evidence.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--run-type', choices=RUN_TYPES, required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--child-exit', type=int, required=True)
    parser.add_argument('--repo', default=str(REPO))
    parser.add_argument('--receipt', help='diagnostic receipt to bind')
    parser.add_argument('--children', nargs='+', default=(),
                        help='exp11_haa_job: the child directories')
    parser.add_argument('--expect', choices=EXPECTATIONS,
                        help='exp11_haa_job: which child set is required')
    parser.add_argument('--job-spec', help='exp11_haa_job: the spec this seed was run from')
    parser.add_argument('--owner-pid', type=int,
                        help='the live launcher that owns this run dir (its launch.pid)')
    return parser


def main(argv=None):
    """Exit 0 after writing completion.json, 2 on any refusal (nothing written).

    ``child-exit`` is delegated to the pinned closer: the end marker and the receipt
    contract are shared evidence, not an exp_11 decision, and re-implementing them would
    fork the very bytes ``closed_log`` re-validates.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['preflight']:
        return preflight_main(argv[1:])
    if argv[:1] == ['child-exit']:
        return base.child_exit_main(argv[1:])
    if argv[:1] == ['passed']:
        return passed_main(argv[1:])
    if argv[:1] == ['finalize']:
        argv = argv[1:]
    args = build_parser().parse_args(argv)
    try:
        fields = finalize(args.run_dir, args.run_type, args.log, args.child_exit,
                          repo=args.repo, receipt=args.receipt, children=args.children,
                          expect=args.expect, owner_pid=args.owner_pid,
                          job_spec=args.job_spec)
    except (OSError, ValueError) as error:
        print('EXP11_FINALIZE_REFUSED ' + str(error), file=sys.stderr, flush=True)
        return 2
    print('EXP11_FINALIZE_OK ' + json.dumps({key: fields[key] for key in
        ('run_type', 'run_dir', 'child_exit', 'admissible_arm')}, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
