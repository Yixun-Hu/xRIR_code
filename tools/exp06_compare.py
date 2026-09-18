"""exp_06's H3: does the oriented pipeline cost simulated accuracy at k = 0 (plan 6.3, 7)?

Three arms of five evaluation seeds each. **C** is exp_06's own
``ckpt/exp06/sim_eval/<arm>/seed<S>/`` written by ``tools.exp06_eval_launch``; **A** is
exp_04's completed control evaluations (``ckpt/yaw_aug/eval/control_k8_seed{42..46}_k0``)
and **B** is either exp_05's M-tier cylindrical evaluations or an exp_06 evaluation of
the exp_01 cylindrical checkpoint. Every run is admitted fail-closed against the
approvals and the protocol of 6.3 before a single number is read, and the statistics are
exp_04's own: ``cell_mask``, ``five_seed_mean``, ``rho_bootstrap``, ``one_sided_upper``,
``h1_verdict`` and ``convergence``, imported from ``tools.paired_compare`` and not
re-implemented.

    python tools/exp06_compare.py --runs-c <five dirs> --runs-a <five> --runs-b <five> \
        --json ckpt/exp06/h3.json --summary ckpt/exp06/h3_summary.txt
"""
import argparse
import functools
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from tools import exp06_approvals_api as approvals_api
from tools import exp06_bootstrap
from tools import exp06_train_evidence
from tools import paired_compare
from tools import provenance
from tools import exp04_profiles, exp05_profiles
from tools.exp04_profiles import CONTROL, CYL
from tools.reference_manifest import load_manifest, manifest_hash
from tools.summarize_yaw import load_run, rooms_from_paths, _check_metrics_reconciliation

ENTRY_MODULE = 'tools.exp06_compare'
REPO = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44, 45, 46)
METRICS = ('EDT', 'C50')
NUM_SHOT = 8                       # the registered K of 6.3's simulated evaluations
MARGIN = 0.03                      # exp_04's H1 rule, inherited by section 7's H3
SUPERIORITY_ALPHA = 0.025          # one-sided 97.5 % upper bound of rho
COMPANION_ALPHA = 0.05
N_BOOT = 20000
BOOT_SEEDS = (0, 1)
CONVERGENCE_TOL = 0.10
CONVERGENCE_FACTOR = 4             # section 7: the draws are quadrupled once, then withheld
# Finding 1: the registered experiment, taken from exp_04's own frozen registration --
# the split and its size, the room population, the canonical query digest, the data
# inventory every run must have read, and the seed-specific K = 8 reference manifests.
DATASET = exp04_profiles.COMMON['dataset']
SPLIT = {'split': DATASET['split'], 'n_queries': DATASET['n_queries'],
         'n_rooms': DATASET['n_rooms'], 'query_sha256': DATASET['query_sha256'],
         'inventory_sha256': DATASET['inventory_sha256'],
         'references': exp04_profiles.json_value(exp04_profiles.REFERENCES[NUM_SHOT])}
BATCH_SIZE = 16
CONDITIONS = 'P'
YAW_COLS = [0]
OUTPUTS = ('per_sample_yaw.json', 'metrics_yaw.json')
_UNBOUND = object()                # finding 4: a declared null is never "skip this check"
MUTABLE_INPUTS = frozenset({'control_args', 'train_args', 'train_manifest',
                            'train_completion', 'probe_receipt', 'heading'})
TRAINING_BINDINGS = ('train_manifest', 'train_completion')
REVALIDATE = ('repo', 'checkpoint', 'checkpoint_sha256', 'manifest_path',
              'manifest_file_sha256', 'evaluator_closure', 'source_closures',
              'mutable_inputs', 'eval_manifest')
FILES = ('eval_manifest.json', 'completion.json') + OUTPUTS
# Role -> what the run must be. C is exp_06's own arm; A and B are reused baselines whose
# checkpoints are exp_01's, hashed at admission and recorded with the result.
# Finding 7: an evaluation is written by exactly one of three entry points, and each has
# its own approved identity. B is the only role whose writer is not fixed in advance --
# it is either exp_05's M tier (exp_05's entry point, exp_04's writer) or an exp_06
# evaluation of the exp_01 cylindrical checkpoint.
ROUTES = ('exp04', 'exp05', 'exp06')
EPOCH = 12                         # every arm of 6.3 evaluates the twelfth-epoch weights
M_CYL = next(arm for arm in exp05_profiles.ARMS if arm['role'] == 'M_cyl')
# Finding 5: arm B's exp_05 route evaluates one registered tier arm, and that
# registration -- not the run's own declaration -- says which counts it must record.
EXP05_M = {'role': M_CYL['role'], 'tier': M_CYL['tier'], 'backbone': M_CYL['backbone'],
           'sha256': M_CYL['sha256'], 'epoch': M_CYL['epoch'],
           'counts': exp05_profiles.json_value(M_CYL['counts'])}
# Full-review F4: `routes` is the closed set 6.3 registers for each role. A is exp_04's
# reused control evaluation, C is exp_06's own arm, and B is *either* exp_05's M tier or an
# exp_06 evaluation of the exp_01 cylindrical checkpoint -- never a descriptive exp_04 run.
ROLES = {'C': {'arm': 'cyl_or', 'checkpoint': None, 'route': 'exp06', 'role': 'arm',
               'routes': ('exp06',),
               'backbone': 'cylindrical_oriented', 'epoch': EPOCH},
         'A': {'arm': 'control', 'checkpoint': CONTROL, 'route': 'exp04', 'role': 'arm',
               'routes': ('exp04',),
               'backbone': CONTROL['backbone'], 'epoch': CONTROL['epoch']},
         'B': {'arm': 'cyl', 'checkpoint': CYL, 'route': None, 'role': 'baseline',
               'routes': ('exp05', 'exp06'),
               'backbone': CYL['backbone'], 'epoch': CYL['epoch'], 'exp05': EXP05_M}}
CONTRASTS = (('C', 'B'), ('C', 'A'))


@functools.lru_cache(maxsize=None)
def exp06_registry():
    """Finding 4: the reviewed registry -- its digest, its classes, its metadata fields.

    ``tools.exp06_eval`` defines all three for the writer, so an admitted run is compared
    with the registry that wrote it rather than with a syntactic hex check.
    """
    from model.xRIR_cyl_oriented import BACKBONES_EXP06
    from tools import exp06_eval
    return (exp06_eval.registry_sha256(),
            {name: cls.__name__ for name, cls in BACKBONES_EXP06.items()},
            tuple(exp06_eval.METADATA_FIELDS))


@functools.lru_cache(maxsize=None)
def tier_fields():
    """The tier block exp_04's launcher copies into the completion and both outputs."""
    from tools.exp04_eval_launch import TIER_FIELDS
    return tuple(TIER_FIELDS)


def _equal(actual, expected):
    return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def _is_sha256(value):
    return type(value) is str and len(value) == 64 and all(
        character in '0123456789abcdef' for character in value)


def _closure_digest(closure):
    """exp_04's own closure digest, recomputed from the reviewed blobs it recorded."""
    return paired_compare._closure_digest(closure)


def route_of(fields, declared):
    """Which evaluator wrote this run: exp_06's, exp_05's M tier, or exp_04's."""
    if declared is not None:
        return declared
    if 'writer_exp06' in (fields.get('source_closures') or {}):
        return 'exp06'
    return 'exp05' if 'tier' in fields else 'exp04'


def exp04_approvals():
    """The identity of exp_04's own approvals record, as 6.4's `reused` section pins it."""
    path = exp04_profiles.APPROVED_DIGESTS_PATH
    return {'path': str(path), 'sha256': provenance.sha256_file(path)}


def exp05_approvals():
    """exp_05's approvals: its evaluator identity, and the record's own hash."""
    pins, identity = exp05_profiles.load_approved_digests()
    return dict(pins['closures']), dict(identity)


def check_route(route, fields, closures, approved, require):
    """Each route compares the closures that ran with that experiment's own approvals."""
    pinned = (approved or {}).get('code', {})
    reused = (approved or {}).get('reused', {})
    if route == 'exp06':
        require(closures['entrypoint']['sha256'] == pinned.get('eval'),
                'approved eval closure')
        require(closures['writer_exp06']['sha256'] == pinned.get('eval_launch'),
                'approved eval_launch closure')
        return
    if route == 'exp05':
        pins, identity = exp05_approvals()
        require(_is_sha256(identity.get('sha256'))
                and identity['sha256'] == reused.get('exp05_approved_digests_sha256'),
                'exp_05 approvals record is not the approved one')
        require(closures['entrypoint']['sha256'] == pins.get('evaluator'),
                'approved exp_05 evaluator closure')
        require(closures['writer']['sha256'] == pins.get('writer'),
                'approved exp_05 writer closure')
        return                     # the tier itself is established by check_exp05_tier
    require(exp04_approvals()['sha256'] == reused.get('exp04_approved_digests_sha256'),
            'exp_04 approvals record is not the approved one')
    require(closures['entrypoint']['sha256'] == reused.get('exp04_evaluator_closure'),
            'approved exp_04 evaluator closure')
    require(closures['writer']['sha256'] == reused.get('exp04_writer_closure'),
            'approved exp_04 writer closure')


def admit_run(run_dir, role, approved, split=SPLIT, check=None, inputs=None,
              roles=ROLES, stats=None):
    """One evaluation run of one arm, against 6.3's protocol and section 7's admission."""
    directory = Path(run_dir).resolve()
    label = '{} {}'.format(role, directory)
    inputs = {} if inputs is None else inputs
    stats = {} if stats is None else stats
    if check is None:
        def check(ok, name):
            if not ok:
                raise ValueError(name)

    def require(ok, name):
        check(ok, label + ': ' + name)

    def bind(path, expected=_UNBOUND):
        """Bind one input. A declared ``None`` is a missing hash, never a free pass."""
        path = str(Path(path).resolve())
        actual = inputs[path] if path in inputs else provenance.sha256_file(path)
        require(expected is _UNBOUND or (_is_sha256(expected) and actual == expected),
                'digest ' + path)
        inputs[path] = actual
        return actual

    require({item.name for item in directory.iterdir()} == set(FILES), 'run directory contents')
    payload = {name: json.loads((directory / name).read_text()) for name in FILES}
    fields, completion = payload['eval_manifest.json'], payload['completion.json']
    digest = bind(directory / 'eval_manifest.json', completion.get('eval_manifest_sha256'))
    require(completion.get('eval_manifest_sha256') == digest, 'completion manifest digest')
    bind(directory / 'completion.json')
    require(_equal(completion.get('schema_version'), 1)
            and type(completion.get('child_exit_status')) is int
            and completion['child_exit_status'] == 0, 'completion status/schema')
    require(completion.get('directory_listing') == sorted(
        name for name in FILES if name != 'completion.json'), 'completion directory_listing')
    for name in OUTPUTS:
        require(name in (completion.get('outputs') or {}), 'completion output ' + name)
        bind(directory / name, completion['outputs'][name])
    for key, value in (('confirmatory', True), ('allow_dirty_used', False)):
        require(_equal(completion.get(key), value) and _equal(fields.get(key), value), key)
    log = completion.get('log')
    require(isinstance(log, dict) and isinstance(log.get('path'), str),
            'completion records no child log')
    bind(log['path'], log.get('sha256'))
    expected = {'gl_seed': fields.get('manifest_seed'), 'batch_size': BATCH_SIZE,
                'batch_canonical': True, 'tf32': False, 'conditions': CONDITIONS,
                'yaw_cols': list(YAW_COLS), 'split': split['split'],
                'split_count': split['n_queries'], 'n_samples': split['n_queries'],
                'max_samples': 0, 'confirmatory': True, 'allow_dirty_used': False,
                'num_shot': NUM_SHOT}
    for key, value in sorted(expected.items()):
        require(key in fields and _equal(fields[key], value), key)
    seed = fields.get('manifest_seed')
    require(type(seed) is int and seed in SEEDS, 'manifest_seed')
    require(type(fields.get('gl_seed')) is int and fields['gl_seed'] == seed,
            'gl_seed == manifest_seed')
    checkpoint = roles[role]['checkpoint']
    epoch = roles[role]['epoch']
    # Full-review F4: which evaluator wrote this run decides what evidence it must carry, so
    # the route is established before the checkpoint block that depends on it.
    route = route_of(fields, roles[role]['route'])
    require(route in ROUTES, 'unknown evaluator route: ' + str(route))
    registered = tuple(roles[role]['routes'])
    require(route in registered, 'arm {} is admitted only through the registered route{} {},'
            ' not {}'.format(role, '' if len(registered) == 1 else 's',
                             '/'.join(registered), route))
    exp06 = route == 'exp06'
    actual = bind(fields['checkpoint'], fields.get('checkpoint_sha256'))
    require(fields.get('backbone') == roles[role]['backbone'],
            'backbone is not the registered {} of arm {}'.format(roles[role]['backbone'],
                                                                 role))
    if checkpoint is None:                      # arm C: the approved epoch_012 artifact
        pinned = (approved or {}).get('artifacts', {}).get('epoch_012', {})
        require(_is_sha256(pinned.get('sha256')), 'artifacts.epoch_012 is not approved')
        require(actual == pinned.get('sha256'), 'checkpoint is not the approved epoch_012')
        require(pinned.get('epoch') == epoch, 'the approved epoch_012 is not epoch 12')
        require(fields.get('checkpoint_epoch') == epoch, 'checkpoint_epoch')
    else:                                       # arms A and B: exp_01's published weights
        require(actual == checkpoint['sha256'], 'checkpoint is not the exp_01 ' + role)
        require(Path(fields['checkpoint']).name == Path(checkpoint['checkpoint']).name,
                'checkpoint path')
        # Finding 4: a historical manifest records no epoch; its registration does, and a
        # declared one that contradicts the registration is never published as the arm's.
        # Full-review F4: that exception is the reused exp_04/exp_05 routes'. An exp_06
        # evaluation writes `checkpoint_epoch` itself, so an absent one is a refusal.
        require(fields.get('checkpoint_epoch', None if exp06 else epoch) == epoch,
                'checkpoint_epoch')
    closures = fields.get('source_closures') or {}
    evaluator = fields.get('evaluator_closure') or {}
    require(_closure_digest(evaluator) == evaluator.get('sha256'), 'evaluator closure digest')
    pinned = (approved or {}).get('code', {})
    require(evaluator.get('sha256') == pinned.get('evaluator_exp03'),
            'evaluator closure is not the pinned exp_03 one')
    names = ('entrypoint', 'writer', 'writer_exp06') if exp06 else ('entrypoint', 'writer')
    require(set(closures) >= set(names), 'source_closures ' + ', '.join(names))
    for name in names:
        require(_closure_digest(closures[name]) == closures[name].get('sha256'),
                'closure digest ' + name)
    check_route(route, fields, closures, approved, require)
    registry, classes, metadata = exp06_registry()
    if exp06:
        require(fields.get('registry_sha256') == registry,
                'registry_sha256 is not the reviewed backbone registry')
        require(fields.get('model_class') == classes.get(fields.get('backbone')),
                'model class is not the one the reviewed registry binds to this backbone')
        require(fields.get('checkpoint_role') == roles[role]['role'], 'checkpoint_role')
        require(fields.get('frame') == 'room' and fields.get('heading') is None,
                'the simulated split carries no heading')
    run = load_run(str(directory))
    aggregate = payload['metrics_yaw.json']
    require(_equal(run.get('meta'), aggregate.get('meta')), 'output meta agreement')
    keys = ('backbone', 'checkpoint', 'manifest_hash', 'gl_seed', 'batch_size',
            'tf32', 'manifest_seed', 'yaw_cols', 'conditions', 'n_samples')
    for key in keys + (metadata if exp06 else ()):
        require(key in fields and _equal(run['meta'].get(key), fields[key]), 'meta ' + key)
    require(run['meta'].get('eval_manifest_sha256') == digest, 'meta manifest digest')
    require(set(run.get('P') or {}) == {'0'} and 'E' not in run, 'condition P at k = 0 only')
    queries, index = run.get('query'), run.get('index')
    require(isinstance(queries, list) and len(queries) == split['n_queries']
            and len(set(queries)) == len(queries), 'query count')
    require(index == list(range(split['n_queries'])) and all(type(i) is int for i in index),
            'per-sample index is the manifest order')
    for metric in METRICS:
        values = run['P']['0'].get(metric.lower())
        require(isinstance(values, list) and len(values) == split['n_queries'],
                'metric length ' + metric)
        require(all(value is None or type(value) in (int, float) for value in values),
                'metric type ' + metric)
    for failure in _check_metrics_reconciliation(label, run):
        require(False, 'reconciliation ' + failure)
    # Full-review F2: which training run this arm's weights came out of, and how it is
    # established -- the approved epoch_012 artifact for C, the registered exp_01 arm for B.
    training = None if not exp06 else {
        'role': roles[role]['role'],
        'epoch_sha256': None if checkpoint is not None else
        ((approved or {}).get('artifacts', {}).get('epoch_012', {}) or {}).get('sha256'),
        'registered_sha256': None if checkpoint is None else checkpoint['sha256']}
    check_evidence(fields, directory, digest, require, bind, stats, route,
                   training=training, split=split)
    check_reference(fields, run, split, require, bind)
    if route == 'exp05':
        run['tier'] = check_exp05_tier(fields, run, completion, actual,
                                       roles[role].get('exp05'), require, bind)
    run['role'], run['seed'], run['manifest_hash'] = role, seed, fields.get('manifest_hash')
    # Finding 5: the weights each role really evaluated, hashed here and published.
    run['checkpoint'] = {'sha256': actual, 'path': fields['checkpoint'], 'route': route,
                         'epoch': fields.get('checkpoint_epoch', epoch)}
    return run


def _stamp(path):
    status = Path(path).stat()
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns,
            status.st_ctime_ns)


def check_evidence(fields, directory, digest, require, bind, stats, route, training=None,
                   split=SPLIT):
    """Finding 4: the applicable `paired_compare.admit_run` evidence, composed here.

    The mutable inputs an arm declares, the declared inputs the launcher revalidated at
    finalisation, every file of every recorded closure, and the data inventory the run
    read -- each bound by the bytes it has now, with the stat identity that makes a
    second read of the same path a contradiction rather than a silent update.
    """
    names = set(fields.get('mutable_inputs') or {})
    require(names <= MUTABLE_INPUTS, 'mutable_inputs names')
    if route == 'exp06':
        require(set(TRAINING_BINDINGS) <= names, 'the exp_06 arm binds its training run')
    declared = dict(fields, eval_manifest={'path': str(directory / 'eval_manifest.json'),
                                           'sha256': digest})
    declared.pop('data_identity', None)      # bound below, with its inventory
    for failure in provenance.revalidate(declared, required=REVALIDATE):
        require(False, 'revalidate ' + failure)
    root = Path(fields['repo'])
    closures = dict(fields['source_closures'], frozen_evaluator=fields['evaluator_closure'])
    for name in sorted(closures):
        for record in closures[name]['files']:
            bind(root / record['path'], record.get('working_tree_sha256'))
    identity = fields.get('data_identity')
    require(isinstance(identity, dict), 'data_identity')
    require(provenance._inventory_digest(identity['inventory'])
            == identity.get('inventory_sha256'), 'data inventory digest')
    require(identity.get('inventory_sha256') == split['inventory_sha256'],
            'the registered data inventory of the unseen split')
    require(identity.get('manifest_hash') == fields['manifest_hash']
            and identity.get('manifest_file_sha256') == fields['manifest_file_sha256'],
            'dataset manifest identity')
    for record in identity['inventory']:
        path = str((Path(identity['data_root']) / record['path']).resolve())
        before = _stamp(path)
        bind(path, record.get('sha256'))
        require(before == _stamp(path) and stats.get(path, before) == before,
                'data changed ' + path)
        stats[path] = before
    for record in (fields.get('mutable_inputs') or {}).values():
        bind(root / record['path'], record.get('sha256'))
    if training is not None and set(TRAINING_BINDINGS) <= names:
        # Full-review F2: the names were satisfied by any file at all. They are now the same
        # records tools.exp06_eval_launch validated, re-checked here against this run's own
        # checkpoint digest, and every artefact the evidence enumerates is bound with it.
        bound = {name: str(root / (fields['mutable_inputs'][name] or {}).get('path', ''))
                 for name in TRAINING_BINDINGS}
        try:
            exp06_train_evidence.check(
                training['role'], bound, fields.get('checkpoint_sha256'),
                epoch_sha256=training.get('epoch_sha256'),
                registered_sha256=training.get('registered_sha256'), hasher=bind)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            require(False, 'training evidence: {}'.format(error))


def check_reference(fields, run, split, require, bind):
    """Finding 5: the registered K = 8 reference, parsed, and the order it fixes."""
    path = bind(fields['manifest_path'], fields.get('manifest_file_sha256'))
    reference = load_manifest(fields['manifest_path'])
    require(reference.get('num_shot') == NUM_SHOT
            and reference.get('seed') == fields['manifest_seed'],
            'reference seed/num_shot')
    entries = reference['entries']
    require(isinstance(entries, list) and len(entries) == split['n_queries'],
            'the reference draws one entry per query of the split')
    require(all(isinstance(entry.get('refs'), list) and len(entry['refs']) == NUM_SHOT
                for entry in entries),
            'every reference entry draws eight references (K = {})'.format(NUM_SHOT))
    require(manifest_hash(reference) == fields['manifest_hash'], 'reference semantic hash')
    require(split['references'].get(fields['manifest_seed']) == fields['manifest_hash'],
            'the registered reference manifest of seed {}'.format(fields['manifest_seed']))
    require(run['query'] == [entry['query'] for entry in entries]
            and run['index'] == [entry['index'] for entry in entries],
            'query/index reference order')
    require(paired_compare._digest(run['query']) == split['query_sha256'],
            'the canonical query digest of the registered split')
    require(len(set(rooms_from_paths(run['query']))) == split['n_rooms'], 'room count')
    return path


def check_exp05_tier(fields, run, completion, digest, spec, require, bind):
    """Finding 5: the M-tier declaration must describe the checkpoint's own args.json.

    exp_05's evaluator derives its tier block from the ``args.json`` beside the weights
    it loads, so the block is re-derived here with exp_05's own rules -- the adjacent
    file, its digest in both declarations, the registered tier configuration and the
    registered parameter counts of the arm those weights are -- and the same block must
    appear, unchanged, in the manifest, the completion and both outputs.

    Close review 2, finding 5: the file is read **once**. The record parsed and the digest
    published both come from that one buffer, which is then reconciled with any binding
    the run already carries and with both declarations, so the tier interpretation always
    describes the bytes its hash identifies.
    """
    from tools import exp05_params
    require(isinstance(spec, dict), 'this role has no registered exp_05 tier arm')
    # Full-review F4: exp_05's evaluator records no `checkpoint_epoch`; the registration it
    # pins does, so the twelfth-epoch requirement of 6.3 is established from the registry.
    require(str((spec or {}).get('epoch')) == str(EPOCH),
            'the registered exp_05 arm is not the twelfth-epoch one')
    require(digest == (spec or {}).get('sha256'),
            'the checkpoint is not the registered exp_05 {}'.format((spec or {}).get('role')))
    require(_equal(fields.get('tier'), (spec or {}).get('tier')),
            'tier is not the registered {}'.format((spec or {}).get('tier')))
    path = Path(fields['checkpoint']).resolve().parent / 'args.json'
    require(path.is_file(), 'the exp_05 checkpoint has no adjacent args.json: ' + str(path))
    raw = path.read_bytes()
    parsed = hashlib.sha256(raw).hexdigest()
    actual = bind(path, parsed)      # a binding this run already made must be these bytes
    require(actual == parsed, 'the checkpoint args.json changed while it was read')
    require(fields.get('args_json_sha256') == parsed,
            'args_json_sha256 is not the digest of the checkpoint args.json')
    binding = (fields.get('mutable_inputs') or {}).get('train_args') or {}
    declared = Path(fields['repo']) / str(binding.get('path', ''))
    require(declared.resolve() == path, 'train_args binds {}, which is not the checkpoint '
            'args.json {}'.format(binding.get('path'), path))
    require(binding.get('sha256') == parsed, 'the train_args digest is not the args.json')
    recorded = json.loads(raw)
    require(isinstance(recorded, dict), 'the checkpoint args.json is not a record')
    config = {'vit_' + key: value for key, value in exp05_params.TIERS['M'].items()}
    legacy = not any(key in recorded for key in config) and 'tier' not in recorded
    require(fields.get('legacy_M') is legacy,
            'legacy_M {!r} is not the {!r} of the checkpoint args.json'.format(
                fields.get('legacy_M'), legacy))
    if not legacy:
        require(exp05_params.tier_of(recorded) == 'M'
                and recorded.get('tier', 'M') == 'M',
                'the checkpoint args.json is not an M-tier record')
    for key, value in sorted(config.items()):
        require(_equal(fields.get(key), value), key)
    for key in ('backbone', 'param_counts'):
        require(key not in recorded or _equal(recorded[key], fields.get(key)),
                'the checkpoint args.json {} differs from the evaluated model'.format(key))
    counts = (spec or {}).get('counts')
    require(_equal(fields.get('param_counts'), counts),
            'param_counts are not the registered counts of {}'.format(
                (spec or {}).get('role')))
    require(fields.get('backbone') == (spec or {}).get('backbone'), 'tier backbone')
    for key in tier_fields():
        require(key in fields, 'the exp_05 route records no ' + key)
        require(_equal(run['meta'].get(key), fields.get(key)), 'meta ' + key)
        require(_equal(completion.get(key), fields.get(key)), 'completion ' + key)
    return {'tier': (spec or {}).get('tier'), 'legacy_M': legacy, 'param_counts': counts,
            'arm': (spec or {}).get('role'),
            'args_json': {'path': str(path), 'sha256': parsed}}


def admit_runs(groups, approved, exploratory=False, split=SPLIT, roles=ROLES):
    """Five seeds per arm, one manifest per seed, one query order across every arm."""
    deviations, inputs, admitted, stats = [], {}, {}, {}

    def check(ok, name):
        if not ok:
            deviations.append(name)

    for role in sorted(groups):
        check(role in roles, 'unknown arm role: ' + str(role))
        runs = []
        for directory in groups.get(role, ()):
            try:
                runs.append(admit_run(directory, role, approved, split, check,
                                      inputs, roles, stats))
            except (KeyError, TypeError, ValueError, OSError, IndexError) as error:
                check(False, '{}: admission {}'.format(directory, error))
        seeds = [run['seed'] for run in runs]
        check(len(runs) == len(SEEDS) and sorted(seeds) == list(SEEDS),
              '{}: the five evaluation seeds 42-46, exactly once each'.format(role))
        admitted[role] = sorted(runs, key=lambda run: run['seed'])
    flat = [run for role in sorted(admitted) for run in admitted[role]]
    if flat:
        check(all(run['query'] == flat[0]['query'] for run in flat),
              'the arms do not share one query order')
        by_seed = {}
        for run in flat:
            by_seed.setdefault(run['seed'], set()).add(run['manifest_hash'])
        check(all(len(hashes) == 1 for hashes in by_seed.values()),
              'the arms do not share one reference manifest per seed')
    if deviations and not exploratory:
        raise ValueError('admission failed: ' + '; '.join(deviations))
    return {'groups': admitted, 'inputs': inputs, 'deviations': deviations,
            'data_stats': stats}


# --- the statistics of section 7, all of them exp_04's ------------------------------------


def _arrays(runs, metric):
    return np.asarray([run['P']['0'][metric.lower()] for run in runs], dtype=float)


def convergence_attempt(draws, n_boot, tol=CONVERGENCE_TOL, alpha=COMPANION_ALPHA):
    """One attempt of section 7's rule: both companions, the endpoint ratio, and why it failed.

    Full-review F5: ``paired_compare.convergence`` lets two identical zero-width intervals
    pass, and a ratio of 0/0 is not evidence of convergence. ``exp06_bootstrap`` already
    implements the plan's stricter endpoint rule (zero width refused), so it decides here
    while both intervals are recorded either way.
    """
    intervals = {seed: tuple(paired_compare.two_sided_interval(draws[seed], alpha))
                 for seed in BOOT_SEEDS}
    record = {'n_boot': int(n_boot), 'tolerance': float(tol), 'companion_alpha': alpha,
              'seed_a': {'seed': BOOT_SEEDS[0], 'lo': float(intervals[BOOT_SEEDS[0]][0]),
                         'hi': float(intervals[BOOT_SEEDS[0]][1])},
              'seed_b': {'seed': BOOT_SEEDS[1], 'lo': float(intervals[BOOT_SEEDS[1]][0]),
                         'hi': float(intervals[BOOT_SEEDS[1]][1])}}
    try:
        check = exp06_bootstrap.convergence_endpoints(intervals.get, BOOT_SEEDS[0],
                                                      BOOT_SEEDS[1], tol)
    except ValueError as error:
        return dict(record, passed=False, movement=None, ratio=None, reason=str(error),
                    width=record['seed_a']['hi'] - record['seed_a']['lo'])
    return dict(record, passed=bool(check['passed']), movement=check['movement'],
                ratio=check['ratio'], width=check['width'], reason=None)


def converged_draws(draws_for, n_boot=N_BOOT, tol=CONVERGENCE_TOL,
                    factor=CONVERGENCE_FACTOR, alpha=COMPANION_ALPHA):
    """Section 7's policy around the inherited resampling; ``tools.paired_compare`` is pinned.

    ``draws_for(seed, n_boot)`` returns that seed's bootstrap samples. A zero-width companion
    carries no verdict and is refused; on any failure the draws are quadrupled **once**, and a
    second failure withholds the verdict. Every attempt is recorded for the published JSON.
    """
    attempts, draws, size = [], None, int(n_boot)
    for size in (int(n_boot), int(n_boot) * int(factor)):
        draws = {seed: draws_for(seed, size) for seed in BOOT_SEEDS}
        attempts.append(convergence_attempt(draws, size, tol, alpha))
        if attempts[-1]['passed']:
            break
    return draws, size, attempts


def contrast_cell(groups, x, y, metric, n_boot=N_BOOT):
    """rho = (mean(x) - mean(y)) / mean(y) on the five-seed means of the shared cohort."""
    a, b = _arrays(groups[x], metric), _arrays(groups[y], metric)
    mask, exclusions = paired_compare.cell_mask(a, e0_b=b, seeds=SEEDS)
    mean_a = paired_compare.five_seed_mean(a)[mask]
    mean_b = paired_compare.five_seed_mean(b)[mask]
    rooms = rooms_from_paths(groups[x][0]['query'])[mask]
    results = {}

    def compute(seed, size, clusters=None):
        return paired_compare.rho_bootstrap(mean_a, mean_b, n_boot=size, seed=seed,
                                            clusters=clusters)

    def draws_for(seed, size):
        results[seed] = compute(seed, size)
        return results[seed].pop('samples')

    draws, size, attempts = converged_draws(draws_for, n_boot)
    cluster = compute(BOOT_SEEDS[0], size, rooms)
    convergence = dict(attempts[-1], attempts=attempts,
                       status='converged' if attempts[-1]['passed'] else 'not_converged')
    return {'contrast': '{} - {}'.format(x, y), 'metric': metric,
            'rho': results[BOOT_SEEDS[0]]['rho'],
            'upper': paired_compare.one_sided_upper(draws[BOOT_SEEDS[0]], SUPERIORITY_ALPHA),
            'alpha': SUPERIORITY_ALPHA, 'companion_alpha': COMPANION_ALPHA,
            'companion_interval': list(paired_compare.two_sided_interval(
                draws[BOOT_SEEDS[0]], COMPANION_ALPHA)),
            'room_cluster_interval': list(paired_compare.two_sided_interval(
                cluster['samples'], COMPANION_ALPHA)),
            'n_rooms_retained': int(len(set(rooms))), 'n': int(mask.sum()),
            'exclusions': exclusions, 'n_boot': size,
            'bootstrap_seeds': list(BOOT_SEEDS), 'convergence': convergence,
            'seed_means': {role: _arrays(groups[role], metric)[:, mask].mean(axis=1).tolist()
                           for role in (x, y)}}


def arm_checkpoints(groups):
    """One checkpoint identity per arm: every seed of an arm evaluated the same weights."""
    identities = {}
    for role in sorted(groups):
        recorded = [run['checkpoint'] for run in groups[role]]
        if not recorded:
            continue
        if any(_equal(item, recorded[0]) is False for item in recorded):
            raise ValueError('arm {} evaluated more than one checkpoint'.format(role))
        identities[role] = dict(recorded[0])
    return identities


def analyse(admitted, margin=MARGIN, n_boot=N_BOOT, exploratory=False):
    """H3 is C vs B; C vs A is the same machinery, reported descriptively.

    Finding 9: a decision-bearing verdict needs the complete A/B/C run set and an
    admission with no deviations. A partial analysis names H3 `unavailable` and an
    exploratory or unapproved one names it `exploratory`; in both cases every verdict is
    suppressed, so a report can never carry a decision its evidence does not support.
    """
    groups = admitted['groups']
    complete = all(role in groups and len(groups[role]) == len(SEEDS) for role in ROLES)
    drafted = bool(exploratory or admitted['deviations'])
    status = 'unavailable' if not complete else ('exploratory' if drafted else 'available')
    result = {'schema_version': 1, 'exploratory': bool(exploratory), 'margin': margin,
              'seeds': list(SEEDS), 'metrics': list(METRICS), 'split': dict(SPLIT),
              'deviations': list(admitted['deviations']), 'inputs': admitted['inputs'],
              'cells': [], 'verdicts': {}, 'H3': status, 'complete_arms': complete,
              'checkpoints': arm_checkpoints(groups)}
    for x, y in CONTRASTS:
        if x not in groups or y not in groups:
            continue
        cells = [contrast_cell(groups, x, y, metric, n_boot) for metric in METRICS]
        result['cells'].extend(cells)
        bounds = {cell['metric']: cell['upper'] for cell in cells}
        verdict = paired_compare.h1_verdict(bounds['EDT'], bounds['C50'], margin)
        failed = [cell['metric'] for cell in cells if not cell['convergence']['passed']]
        if failed:
            verdict = 'not converged'
        if status != 'available':
            verdict = ('suppressed (draft)' if drafted
                       else 'suppressed (no H3 comparator)')
        result['verdicts']['{} - {}'.format(x, y)] = {
            'verdict': verdict, 'descriptive': (x, y) != CONTRASTS[0],
            'not_converged': failed, 'upper': bounds}
    if not exploratory:
        failed = ['{} {}'.format(cell['contrast'], cell['metric']) for cell in result['cells']
                  if not cell['convergence']['passed']]
        if failed:
            result['deviations'].append('convergence gate failed: ' + ', '.join(failed))
    return result


# --- producer identity and the published outputs ------------------------------------------


def producer_identity(strict=True):
    """Bind this producer's imported closure to HEAD, as tools.paired_compare does."""
    if strict:
        return paired_compare.producer_identity(ENTRY_MODULE)
    head = provenance.git_state(REPO)['HEAD']
    records, digest = provenance.closure_record(
        provenance.source_closure(ENTRY_MODULE, REPO), head, REPO)
    return {'sha256': digest, 'files': records, 'commit': head}


def check_producer_identity(approved, identity, exploratory=False):
    """Finding 3: the comparer's own closure, against the approved `code.compare`."""
    if approved is None:
        return []
    pinned = approved['code'].get('compare')
    if identity['sha256'] == pinned:
        return []
    deviation = 'this producer ran the closure {}, not the approved code.compare {}'.format(
        identity['sha256'], pinned)
    if not exploratory:
        raise ValueError(deviation)
    return [deviation]


def recheck_inputs(inputs, stats=None):
    """Finding 10: every bound byte is still the byte admission read."""
    stats = {} if stats is None else stats
    for path in sorted(inputs):
        expected = stats.get(path)
        try:
            unchanged = (_stamp(path) == expected if expected is not None
                         else provenance.sha256_file(path) == inputs[path])
        except OSError as error:
            raise ValueError('input disappeared during analysis: {} ({})'.format(path, error))
        if not unchanged:
            raise ValueError('input changed during analysis: ' + path)


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _safe(value.item())
    return value


def render(result):
    lines = ['{} H3 (margin {:+.3f}, one-sided {:.3f} upper bound of rho)'.format(
        'EXPLORATORY' if result['exploratory'] else 'CONFIRMATORY', result['margin'],
        result['cells'][0]['alpha'] if result['cells'] else SUPERIORITY_ALPHA)]
    for cell in result['cells']:
        lines.append('{:10s} {:4s} rho={:+.6g} upper={:+.6g} companion={} rooms={} n={}'.format(
            cell['contrast'], cell['metric'], cell['rho'], cell['upper'],
            ['{:+.6g}'.format(value) for value in cell['companion_interval']],
            ['{:+.6g}'.format(value) for value in cell['room_cluster_interval']], cell['n']))
    for name in sorted(result['verdicts']):
        entry = result['verdicts'][name]
        lines.append('{}: {}{}'.format(name, entry['verdict'],
                                       ' (descriptive)' if entry['descriptive'] else ''))
    lines.append('H3 {}{}'.format(result.get('H3', 'available'),
                                  '' if result.get('complete_arms', True)
                                  else ' (arm B absent)'))
    lines.extend('Deviation: ' + item for item in result['deviations'])
    return '\n'.join(lines) + '\n'


def write_outputs(result, json_path, summary_path, stats=None):
    """Staged publication (finding 12), after a final input recheck (finding 10)."""
    paths = [Path(summary_path), Path(json_path)]
    if len({path.resolve() for path in paths}) != 2 or any(
            os.path.lexists(str(path)) for path in paths):
        raise FileExistsError('output paths must be distinct and absent')
    text = render(result)
    record = dict(result, summary_path=str(paths[0].resolve()),
                  summary_sha256=hashlib.sha256(text.encode()).hexdigest())
    data = (json.dumps(_safe(record), sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode()
    recheck_inputs(result.get('inputs') or {}, stats)
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
                os.link(temporary, str(path))
                created.append(path)
            finally:
                os.unlink(temporary)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    return record, hashlib.sha256(data).hexdigest(), text


def build_parser():
    parser = argparse.ArgumentParser(description='exp_06 H3: simulated parity at k = 0.')
    for role in sorted(ROLES):
        parser.add_argument('--runs-' + role.lower(), nargs='+', required=(role != 'B'),
                            help='the five evaluation runs of arm ' + role)
    parser.add_argument('--approved')
    parser.add_argument('--approved-commit',
                        help='the reviewed commit the approvals must be committed at')
    parser.add_argument('--json', required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--n-boot', type=int, default=N_BOOT)
    parser.add_argument('--exploratory', action='store_true')
    return parser


def approvals(exploratory, path=None, commit=None, repo=REPO):
    """Finding 7: a confirmatory run's approvals are the bytes committed at the
    reviewed HEAD. Plan 6.4 approves a *commit*, so a well-formed record that nobody
    committed -- or one committed elsewhere -- is refused rather than trusted, and the
    identity the producer publishes names the tracked path and that commit. Exploratory
    runs keep the unbound read and record what production would have refused.
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
    return approved, receipt, approvals_api.require_producer(approved, 'compare', exploratory)


def main(argv=None):
    """0 on success; a refusal exits 1 and an argparse usage error exits 2."""
    args = build_parser().parse_args(argv)
    approved, receipt, deviations = approvals(args.exploratory, args.approved,
                                             args.approved_commit)
    groups = {role: getattr(args, 'runs_' + role.lower()) for role in sorted(ROLES)
              if getattr(args, 'runs_' + role.lower())}
    admitted = admit_runs(groups, approved, args.exploratory)
    identity = producer_identity(strict=not args.exploratory)
    admitted['deviations'] = (list(deviations)
                              + check_producer_identity(approved, identity,
                                                        args.exploratory)
                              + list(admitted['deviations']))
    # Finding 2: what H3 publishes as its own evidence -- the approvals record it read
    # and the producer closure that ran -- is revalidated with every other input.
    if receipt is not None:
        admitted['inputs'][str(Path(receipt['path']).resolve())] = receipt['sha256']
    for record in identity['files']:
        admitted['inputs'][str((REPO / record['path']).resolve())] = \
            record['working_tree_sha256']
    result = analyse(admitted, MARGIN, args.n_boot, args.exploratory)
    result['approved_digests'] = receipt
    result['producer'] = identity
    record, digest, text = write_outputs(result, args.json, args.summary,
                                         admitted.get('data_stats'))
    print(text)
    print(json.dumps({'json': args.json, 'sha256': digest,
                      'summary_sha256': record['summary_sha256'],
                      'verdicts': {name: entry['verdict']
                                   for name, entry in sorted(record['verdicts'].items())}}))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit('refusing: ' + str(error))
