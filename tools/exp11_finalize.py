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

from model.xrir_exp11_registry import BACKBONES_EXP11, build_xrir_exp11
from sim_to_real.haa_dataset import ROOMS
from tools import exp06_finalize as base
from tools import exp06_heading, exp11_profiles, exp11_recipe, provenance
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
                 'exp11_haa_eval': 'tools.exp11_haa_eval'}
HAA_CODE_KEY = {'exp11_haa_finetune': 'haa_finetune', 'exp11_haa_eval': 'haa_eval'}
# The semantic role each serialized type plays in the protocol checks.
ROLE_OF = {'exp11_haa_finetune': 'haa_train', 'exp11_haa_eval': 'haa_eval'}
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
