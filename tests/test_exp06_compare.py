"""exp_06's H3 comparer: fail-closed admission of 6.3's runs, then exp_04's statistics."""
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from model.xRIR_cyl_oriented import BACKBONES_EXP06
from tools import exp06_approvals_api as approvals_api
from tools import exp06_compare as subject
from tools import exp05_profiles
from tools import exp06_eval
from tools import provenance

N = 24                                  # the synthetic split; production uses 6337
ROOMS = ('Cafe_idx_1', 'Hall_idx_2', 'Office_idx_3')
QUERIES = ['{}/{}/S00{}_R00{}_hybrid_IR.wav'.format(ROOMS[i % 3].split('_')[0], ROOMS[i % 3],
                                                    i % 7, i) for i in range(N)]


def reference_hash(manifest):
    from tools.reference_manifest import manifest_hash
    return manifest_hash(manifest)


SOURCES = {}          # every closure file the manifests name, with its real bytes
DATA = {'hallway/meta.json': b'{"test": [0, 1]}', 'hallway/rirs.npy': b'rirs' * 8}
INVENTORY = [{'path': name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
             for name, data in sorted(DATA.items())]


def manifest_of(seed, queries=None, n_queries=N, num_shot=None):
    """The manifest of one seed: the registered K = 8 draw for every query, in order."""
    shots = subject.NUM_SHOT if num_shot is None else num_shot
    chosen = QUERIES[:n_queries] if queries is None else queries
    return {'seed': seed, 'num_shot': shots, 'ir_root': '/ir',
            'entries': [{'index': index, 'query': query,
                         'refs': ['r/{}/{}'.format(index, k) for k in range(shots)]}
                        for index, query in enumerate(chosen)]}


# The synthetic registration: exactly the fields tools.exp04_profiles pins for the real
# unseen split, so admission compares a run against a registry rather than against itself.
SPLIT = {'split': 'unseen', 'n_queries': N, 'n_rooms': len(ROOMS),
         'query_sha256': subject.paired_compare._digest(QUERIES[:N]),
         'inventory_sha256': provenance._inventory_digest(INVENTORY),
         'references': {seed: reference_hash(manifest_of(seed)) for seed in subject.SEEDS}}


def closure(name, tag):
    files = []
    for part in ('{}.py'.format(name), '{}_support.py'.format(name)):
        content, path = (tag + part).encode(), 'tools/' + part
        SOURCES[path] = content
        digest = hashlib.sha256(content).hexdigest()
        files.append({'path': path, 'reviewed_blob_sha256': digest,
                      'working_tree_sha256': digest, 'commits_after_reviewed': []})
    digest = hashlib.sha256(json.dumps(
        [[item['path'], item['reviewed_blob_sha256']] for item in files],
        sort_keys=True).encode()).hexdigest()
    return {'files': files, 'sha256': digest}


CLOSURES = {name: closure(name, name) for name in
            ('exp06_eval', 'exp06_eval_launch', 'exp04_eval', 'exp04_eval_launch',
             'exp05_eval', 'eval_yaw_rotation')}
M_CYL = next(arm for arm in exp05_profiles.ARMS if arm['role'] == 'M_cyl')
M_COUNTS = exp05_profiles.json_value(M_CYL['counts'])
TIER_KEYS = ('tier', 'param_counts', 'args_json_sha256', 'legacy_M',
             'vit_dim', 'vit_depth', 'vit_heads', 'vit_mlp_dim')
EPOCH = 12                              # the pretraining epoch every arm of 6.3 evaluates
REGISTRY = exp06_eval.registry_sha256()
MODEL_CLASSES = {name: cls.__name__ for name, cls in BACKBONES_EXP06.items()}
META_FIELDS = tuple(exp06_eval.METADATA_FIELDS)
ROLES = {'C': {'arm': 'cyl_or', 'checkpoint': None, 'route': 'exp06', 'role': 'arm',
               'backbone': 'cylindrical_oriented', 'epoch': EPOCH, 'routes': ('exp06',)},
         'A': {'arm': 'control', 'checkpoint': {'sha256': None, 'checkpoint': 'simple.pth',
                                                'backbone': 'simple', 'epoch': 12},
               'route': 'exp04', 'role': 'arm', 'backbone': 'simple', 'epoch': 12,
               'routes': ('exp04',)},
         'B': {'arm': 'cyl', 'checkpoint': {'sha256': None, 'checkpoint': 'cyl.pth',
                                            'backbone': 'cylindrical', 'epoch': 12},
               'route': None, 'role': 'baseline', 'backbone': 'cylindrical', 'epoch': 12,
               'routes': ('exp05', 'exp06'),
               'exp05': {'role': 'M_cyl', 'tier': 'M', 'backbone': 'cylindrical',
                         'sha256': None, 'epoch': '12', 'counts': dict(M_COUNTS)}}}
EXP04_DIGEST = provenance.sha256_file(
    __import__('tools.exp04_profiles', fromlist=['x']).APPROVED_DIGESTS_PATH)
EXP05_DIGEST = 'a5' * 32


def approved_digests(epoch_012_sha):
    value = {'schema_version': 1,
             'code': dict({key: 'a' * 64 for key in approvals_api.CODE_KEYS},
                          eval=CLOSURES['exp06_eval']['sha256'],
                          eval_launch=CLOSURES['exp06_eval_launch']['sha256'],
                          evaluator_exp03=CLOSURES['eval_yaw_rotation']['sha256']),
             'reused': dict({key: 'a' * 64 for key in approvals_api.REUSED_DIGESTS},
                            exp04_evaluator_closure=CLOSURES['exp04_eval']['sha256'],
                            exp04_writer_closure=CLOSURES['exp04_eval_launch']['sha256'],
                            exp04_approved_digests_sha256=EXP04_DIGEST,
                            exp05_approved_digests_sha256=EXP05_DIGEST,
                            legacy_receipt={'path': 'r.json', 'sha256': 'a' * 64}),
             'artifacts': {'epoch_012': {'path': 'epoch_012.pth', 'epoch': 12,
                                         'sha256': epoch_012_sha},
                           'heading': {room: 'a' * 64 for room in approvals_api.ROOMS},
                           'gate_g1_sha256': 'a' * 64}}
    return approvals_api.validate(value)


def values(seed, role, metric, bad=()):
    base = {'C': 0.10, 'A': 0.101, 'B': 0.102}[role] * (1.0 if metric == 'edt' else 12.0)
    return [None if i in bad else
            base * (1.0 + 0.01 * math.sin(i * 1.3 + seed) + 0.002 * (role == 'C') * i)
            for i in range(N)]


def reference_manifest(path, seed, split=SPLIT, queries=None):
    """A real reference manifest: the entries the per-sample order must reproduce."""
    manifest = manifest_of(seed, queries, split['n_queries'])
    path.write_text(json.dumps(manifest))
    return manifest


def data_identity(repo, reference, manifest, digest):
    records = [{'path': name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
               for name, data in sorted(DATA.items())]
    return {'data_root': str(Path(repo) / 'data'), 'inventory': records,
            'inventory_files': len(records),
            'inventory_bytes': sum(len(data) for data in DATA.values()),
            'inventory_sha256': provenance._inventory_digest(records),
            'manifest_path': str(reference), 'manifest_file_sha256': digest,
            'manifest_hash': reference_hash(manifest)}


def write_run(directory, role, seed, checkpoints, split=SPLIT, manifest_hash=None,
              manifest=None, per_sample=None, completion=None, bad=(), repo=None,
              queries=None, route=None):
    """One admissible evaluation run of arm `role`, before the caller's mutations."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    route = (ROLES[role]['route'] or 'exp06') if route is None else route
    exp06 = route == 'exp06'
    checkpoint = checkpoints['C' if role == 'C' else role]
    repo = checkpoints['repo'] if repo is None else Path(repo)
    reference = directory.parent / 'reference_manifest_seed{}.json'.format(seed)
    if not reference.exists():
        reference_manifest(reference, seed, split, queries)
    declared = json.loads(reference.read_text())
    log = directory.parent / '{}_seed{}.log'.format(role, seed)
    log.write_text('evaluation output\n')
    fields = {'schema_version': 1, 'confirmatory': True, 'allow_dirty_used': False,
              'repo': str(repo), 'num_shot': subject.NUM_SHOT, 'mutable_inputs': {},
              'reviewed_commit': 'a' * 40,
              'gl_seed': seed, 'manifest_seed': seed, 'batch_size': 16,
              'batch_canonical': True, 'tf32': False, 'conditions': 'P', 'yaw_cols': [0],
              'acoustic_cols': [0], 'e_acoustic_cols': [], 'split': split['split'],
              'split_count': split['n_queries'], 'n_samples': split['n_queries'],
              'max_samples': 0, 'backbone': 'cylindrical_oriented' if role == 'C' else
              ROLES[role]['checkpoint']['backbone'],
              'checkpoint': str(checkpoint), 'checkpoint_sha256':
              provenance.sha256_file(checkpoint),
              'manifest_path': str(reference),
              'manifest_file_sha256': provenance.sha256_file(reference),
              'manifest_hash': manifest_hash or reference_hash(declared),
              'evaluator_closure': copy.deepcopy(CLOSURES['eval_yaw_rotation']),
              'source_closures': {
                  'entrypoint': copy.deepcopy(
                      CLOSURES['exp06_eval' if exp06 else
                               ('exp05_eval' if route == 'exp05' else 'exp04_eval')]),
                  'writer': copy.deepcopy(CLOSURES['exp04_eval_launch'])}}
    if exp06:
        fields['source_closures']['writer_exp06'] = copy.deepcopy(CLOSURES['exp06_eval_launch'])
        fields.update(registry_sha256=REGISTRY, checkpoint_role=ROLES[role]['role'],
                      checkpoint_epoch=12, frame='room', heading=None,
                      model_class=MODEL_CLASSES[fields['backbone']])
    fields['data_identity'] = data_identity(repo, reference, declared,
                                            fields['manifest_file_sha256'])
    binding = {'path': str(log), 'sha256': provenance.sha256_file(log)}
    if exp06:
        # Full-review F2: the names used to be satisfied by this evaluation's own log.
        fields['mutable_inputs'] = training_inputs(checkpoints, role)
    elif route == 'exp05':
        # The checkpoint-adjacent args.json exp_05's own tier rules read.
        train_args = Path(checkpoint).resolve().parent / 'args.json'
        if not train_args.exists():
            train_args.write_text(json.dumps({'backbone': fields['backbone'],
                                              'num_shot': subject.NUM_SHOT}))
        fields.update(tier='M', legacy_M=True, param_counts=dict(M_COUNTS),
                      args_json_sha256=provenance.sha256_file(train_args),
                      vit_dim=512, vit_depth=12, vit_heads=8, vit_mlp_dim=512,
                      mutable_inputs={'train_args': {
                          'path': str(train_args),
                          'sha256': provenance.sha256_file(train_args)}})
    fields.update(manifest or {})
    keys = ('backbone', 'checkpoint', 'manifest_hash', 'gl_seed', 'batch_size', 'tf32',
            'manifest_seed', 'yaw_cols', 'conditions', 'n_samples')
    extra = META_FIELDS if exp06 else (TIER_KEYS if route == 'exp05' else ())
    meta = {key: fields[key] for key in keys + extra}
    per = {'meta': meta, 'query': list(QUERIES[:split['n_queries']]),
           'index': list(range(split['n_queries'])), 'delay_flips': {'0': 0},
           'decomposition': None,
           'P': {'0': {metric: values(seed, role, metric, bad) for metric in
                       ('edt', 'c50', 't60', 'loss', 'log_mse')}}}
    for key, value in (per_sample or {}).items():
        per[key] = value
    (directory / 'eval_manifest.json').write_text(json.dumps(fields, sort_keys=True))
    meta_sha = provenance.sha256_file(directory / 'eval_manifest.json')
    per['meta']['eval_manifest_sha256'] = meta_sha
    (directory / 'per_sample_yaw.json').write_text(json.dumps(per))
    (directory / 'metrics_yaw.json').write_text(json.dumps(
        {'meta': per['meta'], 'P': {'0': {metric: summarise(values)
                                          for metric, values in per['P']['0'].items()}}}))
    record = {'schema_version': 1, 'child_exit_status': 0, 'eval_manifest_sha256': meta_sha,
              'confirmatory': True, 'allow_dirty_used': False,
              'log': {'path': str(log), 'sha256': provenance.sha256_file(log)},
              'directory_listing': sorted(n for n in subject.FILES if n != 'completion.json'),
              'outputs': {name: provenance.sha256_file(directory / name)
                          for name in subject.OUTPUTS}}
    if route == 'exp05':                     # exp_04's launcher copies them from the manifest
        record.update({key: fields[key] for key in TIER_KEYS})
    record.update(completion or {})
    (directory / 'completion.json').write_text(json.dumps(record, sort_keys=True))
    return directory


def summarise(values):
    finite = [value for value in values if value is not None]
    return {'mean': sum(finite) / len(finite) if finite else None,
            'n_valid': len(finite), 'n_nan': len(values) - len(finite)}


def pretraining_record(root, checkpoint, epochs=EPOCH):
    """Arm C's finalised attempt: the completion and provenance its evaluations bind."""
    root.mkdir(parents=True, exist_ok=True)
    commit = 'c' * 40
    (root / 'provenance.json').write_text(json.dumps(
        {'run_type': 'full', 'reviewed_commit': commit}, sort_keys=True))
    (root / 'completion.json').write_text(json.dumps(
        {'schema_version': 1, 'run_type': 'full', 'admissible_arm': True, 'epochs': epochs,
         'diagnostic': False, 'exploratory': False, 'run_dir': str(root.resolve()),
         'approvals': {'committed_at': commit},
         'artifacts': {'epoch_012.pth': provenance.sha256_file(checkpoint),
                       'provenance.json': provenance.sha256_file(root / 'provenance.json')},
         'checkpoint': {'path': 'epoch_012.pth', 'epoch': epochs,
                        'sha256': provenance.sha256_file(checkpoint)}}, sort_keys=True))
    return root


def legacy_record(root, checkpoint, epochs=EPOCH):
    """Arm B's historical evidence: the reconstructed exp_01 receipt and its args.json."""
    from tools import exp06_legacy_train_receipt as receipts
    train = root / 'xRIR_cyl_8_shot'
    train.mkdir(parents=True, exist_ok=True)
    (train / 'args.json').write_text(json.dumps({'backbone': 'cylindrical', 'epochs': epochs}))
    (train / 'history.jsonl').write_text(''.join(
        json.dumps({'epoch': epoch}) + '\n' for epoch in range(1, epochs + 1)))
    (train / 'train.log').write_text('exp_01 cylindrical\n')
    weights = train / 'epoch_{}.pth'.format(epochs)
    weights.write_bytes(Path(checkpoint).read_bytes())
    receipt = root / 'train_receipt.json'
    receipts.write_receipt(receipt, train, weights, strict=False, registry=(
        {'role': 'cyl', 'backbone': 'cylindrical', 'epoch': epochs,
         'checkpoint': 'ckpt/xRIR_cyl_8_shot/epoch_12.pth',
         'sha256': provenance.sha256_file(weights)},),
        identity={'entry_module': 'x', 'commit': 'a' * 40, 'sha256': 'b' * 64,
                  'files': [], 'drift': []})
    return train, receipt


def training_inputs(checkpoints, role):
    """The `train_manifest` / `train_completion` bindings of one exp_06-route arm."""
    if role == 'C':
        root = checkpoints['train_C']
        paths = {'train_manifest': root / 'provenance.json',
                 'train_completion': root / 'completion.json'}
    else:
        train, receipt = checkpoints['train_B']
        paths = {'train_manifest': train / 'args.json', 'train_completion': receipt}
    return {name: {'path': str(path), 'sha256': provenance.sha256_file(path)}
            for name, path in paths.items()}


@pytest.fixture(scope='module')
def checkpoints(tmp_path_factory):
    base = tmp_path_factory.mktemp('ckpt')
    for name, content in sorted(SOURCES.items()):
        (base / name).parent.mkdir(parents=True, exist_ok=True)
        (base / name).write_bytes(content)
    for name, content in sorted(DATA.items()):
        (base / 'data' / name).parent.mkdir(parents=True, exist_ok=True)
        (base / 'data' / name).write_bytes(content)
    paths = {'repo': base}
    for role, name in (('C', 'epoch_012.pth'), ('A', 'simple.pth'), ('B', 'cyl.pth')):
        path = base / name
        path.write_bytes(name.encode() * 8)
        paths[role] = path
        if role != 'C':
            ROLES[role]['checkpoint']['sha256'] = provenance.sha256_file(path)
        if role == 'B':
            ROLES[role]['exp05']['sha256'] = provenance.sha256_file(path)
    paths['train_C'] = pretraining_record(base / 'pretrain' / 'attempt', paths['C'])
    paths['train_B'] = legacy_record(base / 'exp01', paths['B'])
    return paths


@pytest.fixture
def approved(checkpoints):
    return approved_digests(provenance.sha256_file(checkpoints['C']))


@pytest.fixture
def runs(tmp_path, checkpoints):
    groups = {}
    for role in ('C', 'A', 'B'):
        groups[role] = [str(write_run(tmp_path / role / 'seed{}'.format(seed), role, seed,
                                      checkpoints)) for seed in subject.SEEDS]
    return groups


# --- admission ---------------------------------------------------------------------------


def test_the_registered_protocol_is_section_6_3s():
    assert subject.SEEDS == (42, 43, 44, 45, 46) and subject.MARGIN == 0.03
    assert subject.SPLIT['split'] == 'unseen' and subject.SPLIT['n_queries'] == 6337
    dataset = subject.exp04_profiles.COMMON['dataset']
    assert subject.SPLIT['n_rooms'] == dataset['n_rooms'] == 17
    assert subject.SPLIT['query_sha256'] == dataset['query_sha256']
    assert subject.SPLIT['inventory_sha256'] == dataset['inventory_sha256']
    assert subject.SPLIT['references'] == dict(
        subject.exp04_profiles.REFERENCES[subject.NUM_SHOT])
    assert subject.BATCH_SIZE == 16 and subject.CONDITIONS == 'P'
    assert subject.YAW_COLS == [0] and subject.SUPERIORITY_ALPHA == 0.025
    assert subject.CONTRASTS == (('C', 'B'), ('C', 'A'))
    assert sorted(subject.ROLES) == ['A', 'B', 'C']
    assert subject.ROLES['A']['checkpoint']['checkpoint'].endswith('epoch_12.pth')


def test_every_arm_is_admitted_with_its_five_seeds(runs, approved):
    admitted = subject.admit_runs(runs, approved, split=SPLIT, roles=ROLES)
    assert sorted(admitted['groups']) == ['A', 'B', 'C'] and not admitted['deviations']
    assert [run['seed'] for run in admitted['groups']['C']] == list(subject.SEEDS)
    assert all(path.endswith(('.json', '.jsonl', '.pth', '.py', '.log', '.npy'))
               for path in admitted['inputs'])
    assert len(admitted['groups']['B']) == 5


MUTATIONS = {
    'changed_output': lambda d: (d / 'per_sample_yaw.json').write_text(
        (d / 'per_sample_yaw.json').read_text() + ' '),
    'stale_completion': lambda d: (d / 'eval_manifest.json').write_text(
        (d / 'eval_manifest.json').read_text() + ' '),
    'extra_file': lambda d: (d / 'stray.json').write_text('{}'),
    'missing_output': lambda d: (d / 'metrics_yaw.json').unlink(),
}


@pytest.mark.parametrize('case', sorted(MUTATIONS))
def test_a_changed_or_incomplete_run_is_refused(runs, approved, case):
    MUTATIONS[case](Path(runs['C'][0]))
    with pytest.raises(ValueError, match='admission failed'):
        subject.admit_runs(runs, approved, split=SPLIT, roles=ROLES)


FIELD_REFUSALS = {
    'gl_seed': {'gl_seed': 7},
    'tf32': {'tf32': True},
    'conditions': {'conditions': 'E'},
    'yaw_cols': {'yaw_cols': [0, 32]},
    'split': {'split': 'seen'},
    'split_count': {'split_count': N - 1},
    'n_samples': {'n_samples': N - 1},
    'batch_size': {'batch_size': 8},
    'batch_canonical': {'batch_canonical': False},
    'epoch': {'checkpoint_epoch': 9},
    'role': {'checkpoint_role': 'diagnostic'},
    'evaluator': {'evaluator_closure': closure('other', 'other')},
    'eval_launch': {'source_closures': {
        'entrypoint': CLOSURES['exp06_eval'], 'writer': CLOSURES['exp04_eval_launch'],
        'writer_exp06': closure('other', 'other')}},
}


@pytest.mark.parametrize('case', sorted(FIELD_REFUSALS))
def test_a_run_outside_the_protocol_is_refused(tmp_path, checkpoints, approved, case):
    directory = write_run(tmp_path / 'c', 'C', 42, checkpoints, SPLIT,
                          manifest=FIELD_REFUSALS[case])
    with pytest.raises(ValueError):
        subject.admit_run(directory, 'C', approved, SPLIT, roles=ROLES)


def test_a_wrong_checkpoint_or_index_order_is_refused(tmp_path, checkpoints, approved):
    other = tmp_path / 'other.pth'
    other.write_bytes(b'not the approved weights')
    directory = write_run(tmp_path / 'wrong', 'C', 42, dict(checkpoints, C=other), SPLIT)
    with pytest.raises(ValueError, match='approved epoch_012'):
        subject.admit_run(directory, 'C', approved, SPLIT, roles=ROLES)
    shuffled = write_run(tmp_path / 'order', 'C', 43, checkpoints, SPLIT,
                         per_sample={'index': list(reversed(range(N)))})
    with pytest.raises(ValueError, match='manifest order'):
        subject.admit_run(shuffled, 'C', approved, SPLIT, roles=ROLES)


EVIDENCE_REFUSALS = {
    'null_output_hash': ({'outputs': {'per_sample_yaw.json': None,
                                      'metrics_yaw.json': None}}, None),
    'no_log': ({'log': None}, None),
    'not_confirmatory': ({'confirmatory': False}, None),
    'dirty_used': ({'allow_dirty_used': True}, None),
    'no_num_shot': (None, {'num_shot': None}),
    'wrong_num_shot': (None, {'num_shot': 1}),
    'unknown_mutable': (None, {'mutable_inputs': {'invented': {'path': 'x', 'sha256':
                                                              'a' * 64}}}),
    'no_training_binding': (None, {'mutable_inputs': {}}),
    'no_data_identity': (None, {'data_identity': None}),
}


@pytest.mark.parametrize('case', sorted(EVIDENCE_REFUSALS))
def test_a_run_without_the_evidence_h3_requires_is_refused(tmp_path, checkpoints, approved,
                                                           case):
    """Finding 4: a declared null is never permission to skip a comparison."""
    completion, manifest = EVIDENCE_REFUSALS[case]
    directory = write_run(tmp_path / case, 'C', 42, checkpoints, SPLIT,
                          manifest=manifest, completion=completion)
    with pytest.raises(ValueError):
        subject.admit_run(directory, 'C', approved, SPLIT, roles=ROLES)


def test_a_changed_source_or_data_file_is_refused(tmp_path, checkpoints, approved):
    directory = write_run(tmp_path / 'drift', 'C', 42, checkpoints, SPLIT)
    source = Path(checkpoints['repo']) / 'tools/exp06_eval.py'
    original = source.read_bytes()
    source.write_bytes(original + b'#')
    try:
        with pytest.raises(ValueError, match='(digest |revalidate source)'):
            subject.admit_run(directory, 'C', approved, SPLIT, roles=ROLES)
    finally:
        source.write_bytes(original)


def test_an_aggregate_that_contradicts_the_per_sample_arrays_is_refused(tmp_path,
                                                                       checkpoints,
                                                                       approved):
    directory = write_run(tmp_path / 'aggregate', 'C', 42, checkpoints, SPLIT)
    aggregate = json.loads((directory / 'metrics_yaw.json').read_text())
    aggregate['P']['0']['edt']['mean'] = 9999.0
    (directory / 'metrics_yaw.json').write_text(json.dumps(aggregate))
    record = json.loads((directory / 'completion.json').read_text())
    record['outputs']['metrics_yaw.json'] = provenance.sha256_file(
        directory / 'metrics_yaw.json')
    (directory / 'completion.json').write_text(json.dumps(record, sort_keys=True))
    with pytest.raises(ValueError, match='reconciliation'):
        subject.admit_run(directory, 'C', approved, SPLIT, roles=ROLES)


PROTOCOL_REFUSALS = {
    'manifest_seed': {'seed': 43},
    'manifest_num_shot': {'num_shot': 1},
}


@pytest.mark.parametrize('case', sorted(PROTOCOL_REFUSALS))
def test_a_reference_manifest_outside_the_protocol_is_refused(tmp_path, checkpoints,
                                                              approved, case):
    """Finding 5: the reference is parsed, not merely hashed."""
    directory = write_run(tmp_path / case, 'C', 42, checkpoints, SPLIT)
    fields = json.loads((directory / 'eval_manifest.json').read_text())
    reference = Path(fields['manifest_path'])
    declared = dict(json.loads(reference.read_text()), **PROTOCOL_REFUSALS[case])
    reference.write_text(json.dumps(declared))
    fields['manifest_file_sha256'] = provenance.sha256_file(reference)
    fields['manifest_hash'] = reference_hash(declared)
    rewrite(directory, fields)
    with pytest.raises(ValueError, match='reference '):
        subject.admit_run(directory, 'C', approved, SPLIT, roles=ROLES)


def test_reversed_queries_with_the_manifest_order_intact_are_refused(tmp_path, checkpoints,
                                                                     approved):
    directory = write_run(tmp_path / 'order', 'C', 42, checkpoints, SPLIT)
    per = json.loads((directory / 'per_sample_yaw.json').read_text())
    per['query'] = list(reversed(per['query']))
    (directory / 'per_sample_yaw.json').write_text(json.dumps(per))
    record = json.loads((directory / 'completion.json').read_text())
    record['outputs']['per_sample_yaw.json'] = provenance.sha256_file(
        directory / 'per_sample_yaw.json')
    (directory / 'completion.json').write_text(json.dumps(record, sort_keys=True))
    with pytest.raises(ValueError, match='reference order'):
        subject.admit_run(directory, 'C', approved, SPLIT, roles=ROLES)


def rewrite(directory, fields):
    """Re-publish a manifest and everything binding it, so one checksum masks nothing."""
    fields['data_identity'] = dict(fields['data_identity'],
                                   manifest_hash=fields['manifest_hash'],
                                   manifest_file_sha256=fields['manifest_file_sha256'])
    (directory / 'eval_manifest.json').write_text(json.dumps(fields, sort_keys=True))
    digest = provenance.sha256_file(directory / 'eval_manifest.json')
    for name in subject.OUTPUTS:
        payload = json.loads((directory / name).read_text())
        payload['meta'].update({key: fields[key] for key in payload['meta']
                                if key in fields}, eval_manifest_sha256=digest)
        (directory / name).write_text(json.dumps(payload))
    record = json.loads((directory / 'completion.json').read_text())
    record['eval_manifest_sha256'] = digest
    record['outputs'] = {name: provenance.sha256_file(directory / name)
                         for name in subject.OUTPUTS}
    (directory / 'completion.json').write_text(json.dumps(record, sort_keys=True))


def test_an_exp05_m_tier_run_is_admitted_by_its_own_route(tmp_path, checkpoints, approved,
                                                          monkeypatch):
    """Finding 7: exp05_eval delegates with its own entry point and exp_04's writer."""
    pins = {'evaluator': CLOSURES['exp05_eval']['sha256'],
            'writer': CLOSURES['exp04_eval_launch']['sha256']}
    monkeypatch.setattr(subject, 'exp05_approvals',
                        lambda: (pins, {'sha256': EXP05_DIGEST}))
    directory = write_run(tmp_path / 'm', 'B', 42, checkpoints, SPLIT, route='exp05')
    assert subject.route_of(json.loads((directory / 'eval_manifest.json').read_text()),
                            None) == 'exp05'
    run = subject.admit_run(directory, 'B', approved, SPLIT, roles=ROLES)
    assert run['role'] == 'B' and run['seed'] == 42
    for index, damage in enumerate(({'tier': 'L'}, {'legacy_M': False},
                                    {'mutable_inputs': {}})):
        broken = write_run(tmp_path / 'm{}'.format(index), 'B', 43, checkpoints, SPLIT,
                           route='exp05', manifest=damage)
        with pytest.raises(ValueError):
            subject.admit_run(broken, 'B', approved, SPLIT, roles=ROLES)


def test_an_exp05_run_whose_approvals_record_is_not_the_approved_one_is_refused(
        tmp_path, checkpoints, approved, monkeypatch):
    pins = {'evaluator': CLOSURES['exp05_eval']['sha256'],
            'writer': CLOSURES['exp04_eval_launch']['sha256']}
    monkeypatch.setattr(subject, 'exp05_approvals', lambda: (pins, {'sha256': 'f' * 64}))
    directory = write_run(tmp_path / 'wrong', 'B', 42, checkpoints, SPLIT, route='exp05')
    with pytest.raises(ValueError, match='exp_05 approvals'):
        subject.admit_run(directory, 'B', approved, SPLIT, roles=ROLES)


def test_the_exp04_route_binds_that_experiments_approvals_record(tmp_path, checkpoints,
                                                                 approved):
    assert subject.exp04_approvals()['sha256'] == EXP04_DIGEST
    directory = write_run(tmp_path / 'a', 'A', 42, checkpoints, SPLIT)
    subject.admit_run(directory, 'A', approved, SPLIT, roles=ROLES)
    wrong = copy.deepcopy(approved)
    wrong['reused']['exp04_approved_digests_sha256'] = 'c' * 64
    with pytest.raises(ValueError, match='exp_04 approvals'):
        subject.admit_run(directory, 'A', wrong, SPLIT, roles=ROLES)


def test_a_missing_or_duplicated_seed_is_refused(runs, approved):
    short = dict(runs, C=runs['C'][:4])
    with pytest.raises(ValueError, match='five evaluation seeds'):
        subject.admit_runs(short, approved, split=SPLIT, roles=ROLES)
    duplicated = dict(runs, C=runs['C'][:4] + [runs['C'][0]])
    with pytest.raises(ValueError, match='five evaluation seeds'):
        subject.admit_runs(duplicated, approved, split=SPLIT, roles=ROLES)


def test_the_arms_must_share_one_query_order_and_manifest(tmp_path, checkpoints, approved,
                                                          runs):
    odd = write_run(tmp_path / 'odd', 'A', 42, checkpoints, SPLIT,
                    manifest={'manifest_hash': 'e' * 64})
    groups = dict(runs, A=[str(odd)] + runs['A'][1:])
    with pytest.raises(ValueError, match='reference manifest'):
        subject.admit_runs(groups, approved, split=SPLIT, roles=ROLES)


def test_null_approvals_refuse_in_production_and_list_deviations(runs):
    null = approvals_api.validate({'schema_version': 1,
                                   'code': {key: None for key in approvals_api.CODE_KEYS},
                                   'reused': dict({k: None for k in approvals_api.REUSED_DIGESTS},
                                                  legacy_receipt={'path': None, 'sha256': None}),
                                   'artifacts': {'epoch_012': {'path': None, 'epoch': None,
                                                               'sha256': None},
                                                 'heading': {r: None for r in approvals_api.ROOMS},
                                                 'gate_g1_sha256': None}})
    with pytest.raises(ValueError, match='admission failed'):
        subject.admit_runs(runs, null, split=SPLIT, roles=ROLES)
    admitted = subject.admit_runs(runs, null, exploratory=True, split=SPLIT, roles=ROLES)
    assert any('approved' in item or 'pinned' in item for item in admitted['deviations'])


# --- the statistics, the verdict boundary and the published outputs -----------------------


def analysis(runs, approved, n_boot=400):
    admitted = subject.admit_runs(runs, approved, split=SPLIT, roles=ROLES)
    return subject.analyse(admitted, subject.MARGIN, n_boot)


def test_h3_is_c_against_b_with_c_against_a_descriptive(runs, approved):
    result = analysis(runs, approved)
    assert [cell['contrast'] for cell in result['cells']] == [
        'C - B', 'C - B', 'C - A', 'C - A']
    assert [cell['metric'] for cell in result['cells']] == ['EDT', 'C50'] * 2
    assert result['verdicts']['C - B']['descriptive'] is False
    assert result['verdicts']['C - A']['descriptive'] is True
    for cell in result['cells']:
        assert cell['n'] == N and cell['n_rooms_retained'] == 3
        assert len(cell['room_cluster_interval']) == 2
        # F5: the published n_boot is the size the convergence rule stopped at.
        attempts = cell['convergence']['attempts']
        assert [attempt['n_boot'] for attempt in attempts] == [400, 1600][:len(attempts)]
        assert cell['n_boot'] == attempts[-1]['n_boot'] == cell['convergence']['n_boot']
        assert cell['convergence']['passed'] is (
            cell['convergence']['status'] == 'converged')
        assert len(cell['seed_means'][cell['contrast'][0]]) == 5
        assert cell['companion_interval'][0] <= cell['upper']
    assert result['verdicts']['C - B']['verdict'] in (
        'non-inferior', 'non-inferior on EDT only', 'non-inferior on C50 only', 'not shown',
        'not converged')


def test_the_verdict_boundary_is_strict():
    assert subject.paired_compare.h1_verdict(0.0299, 0.0299, subject.MARGIN) == 'non-inferior'
    assert subject.paired_compare.h1_verdict(0.03, 0.0299, subject.MARGIN) == \
        'non-inferior on C50 only'
    assert subject.paired_compare.h1_verdict(0.03, 0.03, subject.MARGIN) == 'not shown'
    assert subject.paired_compare.h1_verdict(float('nan'), 0.0, subject.MARGIN) == \
        'non-inferior on C50 only'


def test_the_cohort_is_the_queries_finite_in_every_run(tmp_path, checkpoints, approved):
    groups = {}
    for role in ('C', 'A', 'B'):
        groups[role] = [str(write_run(tmp_path / role / str(seed), role, seed, checkpoints,
                                      SPLIT, bad=(3,) if (role == 'C' and seed == 44) else ()))
                        for seed in subject.SEEDS]
    result = analysis(groups, approved)
    for cell in result['cells']:
        assert cell['n'] == N - 1
    exclusions = result['cells'][0]['exclusions']
    assert exclusions['joint']['excluded'] == 1 and exclusions['a']['total']['excluded'] == 1


def test_the_analysis_is_deterministic_and_binds_its_inputs(runs, approved, tmp_path):
    first, second = analysis(runs, approved), analysis(runs, approved)
    assert json.dumps(subject._safe(first), sort_keys=True) == \
        json.dumps(subject._safe(second), sort_keys=True)
    assert first['inputs'] and all(len(value) == 64 for value in first['inputs'].values())
    out, summary = tmp_path / 'h3.json', tmp_path / 'h3_summary.txt'
    record, digest, text = subject.write_outputs(first, out, summary)
    assert digest == provenance.sha256_file(out) and summary.read_text() == text
    assert record['summary_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert 'C - B' in text and 'rho=' in text
    assert 'timestamp' not in json.loads(out.read_text())
    with pytest.raises(FileExistsError):
        subject.write_outputs(first, out, summary)


def test_every_arms_checkpoint_identity_is_recorded_with_the_result(runs, approved):
    """Finding 5: B's exp_01 weights are hashed at admission and published with H3."""
    admitted = subject.admit_runs(runs, approved, split=SPLIT, roles=ROLES)
    result = subject.analyse(admitted, subject.MARGIN, 400)
    assert set(result['checkpoints']) == {'A', 'B', 'C'}
    assert result['checkpoints']['B']['sha256'] == ROLES['B']['checkpoint']['sha256']
    assert result['checkpoints']['A']['sha256'] == ROLES['A']['checkpoint']['sha256']
    assert result['checkpoints']['C']['sha256'] == \
        approved['artifacts']['epoch_012']['sha256']
    assert result['checkpoints']['C']['epoch'] == 12
    assert [result['checkpoints'][role]['route'] for role in ('A', 'B', 'C')] == [
        'exp04', 'exp06', 'exp06']


def test_h3_is_unavailable_without_its_comparator(runs, approved):
    """Finding 9: the primary contrast has arm B, or the report carries no H3."""
    admitted = subject.admit_runs({role: runs[role] for role in ('C', 'A')}, approved,
                                  split=SPLIT, roles=ROLES)
    result = subject.analyse(admitted, subject.MARGIN, 400)
    assert result['H3'] == 'unavailable' and 'C - B' not in result['verdicts']
    assert result['verdicts']['C - A']['verdict'] == 'suppressed (no H3 comparator)'
    assert 'H3 unavailable' in subject.render(result)


def test_an_exploratory_admission_suppresses_every_decision_verdict(runs, approved):
    admitted = subject.admit_runs(runs, approved, exploratory=True, split=SPLIT, roles=ROLES)
    admitted['deviations'].append('not approved: code.compare')
    result = subject.analyse(admitted, subject.MARGIN, 400, exploratory=True)
    assert result['H3'] == 'exploratory'
    assert {entry['verdict'] for entry in result['verdicts'].values()} == {
        'suppressed (draft)'}
    assert 'EXPLORATORY' in subject.render(result)


def test_the_comparers_producer_closure_must_be_the_approved_one(approved):
    """Finding 3: the comparer recorded its identity without ever comparing it."""
    identity = subject.producer_identity(strict=False)
    filled = copy.deepcopy(approved)
    filled['code']['compare'] = identity['sha256']
    assert subject.check_producer_identity(filled, identity) == []
    assert subject.check_producer_identity(None, identity) == []
    with pytest.raises(ValueError, match=r'not the approved code\.compare'):
        subject.check_producer_identity(approved, identity)
    assert len(subject.check_producer_identity(approved, identity, True)) == 1


def test_an_input_changed_during_analysis_is_refused(runs, approved, tmp_path):
    """Finding 10: the inputs are rechecked after the bootstrap, before publication."""
    admitted = subject.admit_runs(runs, approved, split=SPLIT, roles=ROLES)
    result = subject.analyse(admitted, subject.MARGIN, 400)
    # One of this run's own evaluation logs: the shared training artefacts the receipt
    # enumerates belong to the module fixture and are never edited by a test.
    log = next(Path(path) for path in sorted(result['inputs'])
               if Path(path).name == 'C_seed42.log')
    log.write_text(log.read_text() + 'appended after admission\n')
    with pytest.raises(ValueError, match='input changed during analysis'):
        subject.write_outputs(result, tmp_path / 'j.json', tmp_path / 's.txt')
    assert not (tmp_path / 'j.json').exists() and not (tmp_path / 's.txt').exists()


def test_a_failed_publication_leaves_neither_h3_output_behind(runs, approved, tmp_path,
                                                              monkeypatch):
    """Finding 12: the output pair is staged, so a failure publishes nothing."""
    admitted = subject.admit_runs(runs, approved, split=SPLIT, roles=ROLES)
    result = subject.analyse(admitted, subject.MARGIN, 400)
    out, summary = tmp_path / 'pair' / 'h3.json', tmp_path / 'pair' / 'h3.txt'
    real, calls = subject.os.link, []

    def failing(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError('no space left on device')
        return real(source, target)

    monkeypatch.setattr(subject.os, 'link', failing)
    with pytest.raises(OSError):
        subject.write_outputs(result, out, summary)
    assert not out.exists() and not summary.exists()
    monkeypatch.undo()
    subject.write_outputs(result, out, summary)
    assert out.is_file() and summary.is_file()


def test_the_cli_refuses_a_production_run_on_this_branch(runs, tmp_path):
    with pytest.raises(ValueError, match='approvals incomplete'):
        subject.main(['--runs-c'] + runs['C'] + ['--runs-a'] + runs['A'] +
                     ['--json', str(tmp_path / 'j.json'), '--summary', str(tmp_path / 's.txt')])


REAL_A = Path(__file__).resolve().parents[1] / 'ckpt/yaw_aug/eval'
REAL_RUNS = [REAL_A / 'control_k8_seed{}_k0'.format(seed) for seed in subject.SEEDS]


@pytest.mark.skipif(not all(path.is_dir() for path in REAL_RUNS),
                    reason="needs exp_04's control evaluations")
def test_the_real_exp04_control_runs_are_admitted_as_arm_a():
    fields = json.loads((REAL_RUNS[0] / 'eval_manifest.json').read_text())
    approved = approvals_api.validate({
        'schema_version': 1,
        'code': dict({key: 'a' * 64 for key in approvals_api.CODE_KEYS},
                     evaluator_exp03=fields['evaluator_closure']['sha256']),
        'reused': dict({key: 'a' * 64 for key in approvals_api.REUSED_DIGESTS},
                       exp04_evaluator_closure=fields['source_closures']['entrypoint']['sha256'],
                       exp04_writer_closure=fields['source_closures']['writer']['sha256'],
                       exp04_approved_digests_sha256=EXP04_DIGEST,
                       legacy_receipt={'path': 'r.json', 'sha256': 'a' * 64}),
        'artifacts': {'epoch_012': {'path': 'p', 'epoch': 12, 'sha256': 'a' * 64},
                      'heading': {room: 'a' * 64 for room in approvals_api.ROOMS},
                      'gate_g1_sha256': 'a' * 64}})
    admitted = subject.admit_runs({'A': [str(path) for path in REAL_RUNS]}, approved)
    assert [run['seed'] for run in admitted['groups']['A']] == list(subject.SEEDS)
    assert len(admitted['groups']['A'][0]['query']) == 6337


# --- finding 1: the registered experiment, not a self-consistent one ----------------------


def test_an_unregistered_data_inventory_is_refused(tmp_path, checkpoints, approved):
    """An empty inventory with its own correct digest is not the registered split."""
    directory = write_run(tmp_path / 'inventory', 'C', 42, checkpoints, SPLIT)
    fields = json.loads((directory / 'eval_manifest.json').read_text())
    fields['data_identity'] = dict(fields['data_identity'], inventory=[], inventory_files=0,
                                   inventory_bytes=0,
                                   inventory_sha256=provenance._inventory_digest([]))
    rewrite(directory, fields)
    with pytest.raises(ValueError, match='registered data inventory'):
        subject.admit_run(directory, 'C', approved, SPLIT, roles=ROLES)


def test_a_reference_manifest_that_is_not_the_registered_one_is_refused(tmp_path, checkpoints,
                                                                        approved):
    """A different, internally consistent K = 8 draw is not this seed's registered one."""
    directory = write_run(tmp_path / 'reference', 'C', 42, checkpoints, SPLIT)
    fields = json.loads((directory / 'eval_manifest.json').read_text())
    declared = manifest_of(42)
    declared['entries'][0]['refs'] = list(reversed(declared['entries'][0]['refs']))
    reference = Path(fields['manifest_path'])
    reference.write_text(json.dumps(declared))
    fields['manifest_file_sha256'] = provenance.sha256_file(reference)
    fields['manifest_hash'] = reference_hash(declared)
    rewrite(directory, fields)
    with pytest.raises(ValueError, match='registered reference manifest'):
        subject.admit_run(directory, 'C', approved, SPLIT, roles=ROLES)


def test_invented_query_names_are_refused_even_when_consistent(tmp_path, checkpoints,
                                                               approved):
    """Every query renamed, consistently in the reference and the outputs, is refused."""
    queries = ['Cafe/Cafe_idx_1/S000_R{:03d}_invented.wav'.format(i) for i in range(N)]
    directory = write_run(tmp_path / 'names', 'C', 42, checkpoints, SPLIT, queries=queries,
                          per_sample={'query': list(queries)})
    with pytest.raises(ValueError, match='registered reference manifest'):
        subject.admit_run(directory, 'C', approved, SPLIT, roles=ROLES)
    # The canonical query digest is an independent pin: even a registration that named
    # this manifest would not make these the queries of the unseen split.
    pinned = dict(SPLIT['references'])
    pinned[42] = reference_hash(manifest_of(42, queries))
    relaxed = dict(SPLIT, references=pinned)
    with pytest.raises(ValueError, match='canonical query digest'):
        subject.admit_run(directory, 'C', approved, relaxed, roles=ROLES)


def test_a_reference_entry_without_eight_references_is_refused(tmp_path, checkpoints,
                                                               approved):
    """K = 8 is the registered conditioning; an entry that draws none is not this split."""
    directory = write_run(tmp_path / 'refs', 'C', 42, checkpoints, SPLIT)
    fields = json.loads((directory / 'eval_manifest.json').read_text())
    reference = Path(fields['manifest_path'])
    declared = json.loads(reference.read_text())
    for entry in declared['entries']:
        entry['refs'] = []
    reference.write_text(json.dumps(declared))
    fields['manifest_file_sha256'] = provenance.sha256_file(reference)
    fields['manifest_hash'] = reference_hash(declared)
    rewrite(directory, fields)
    with pytest.raises(ValueError, match='eight references'):
        subject.admit_run(directory, 'C', approved, SPLIT, roles=ROLES)


def test_the_registered_room_population_is_enforced(tmp_path, checkpoints, approved):
    """The unseen split is seventeen rooms; a run of one room is a different experiment."""
    with pytest.raises(ValueError, match='room count'):
        subject.admit_run(write_run(tmp_path / 'rooms', 'C', 42, checkpoints, SPLIT), 'C',
                          approved, dict(SPLIT, n_rooms=len(ROOMS) + 1), roles=ROLES)


# --- finding 4: role, registry, class and epoch identities --------------------------------


def test_the_reviewed_registry_and_role_identities_are_registered():
    """The comparer knows which model each role must be, not merely that a hash is hex."""
    registry, classes, fields = subject.exp06_registry()
    assert len(registry) == 64 and set(registry) <= set('0123456789abcdef')
    assert classes['cylindrical_oriented'] == 'xRIR_CylOriented'
    assert fields == ('model_class', 'registry_sha256', 'checkpoint_role',
                      'checkpoint_epoch', 'heading', 'frame')
    assert (registry, classes, fields) == (REGISTRY, MODEL_CLASSES, META_FIELDS)
    assert subject.EPOCH == EPOCH == 12
    assert subject.ROLES['C']['backbone'] == 'cylindrical_oriented'
    assert subject.ROLES['C']['epoch'] == 12
    assert subject.ROLES['A']['backbone'] == 'simple'
    assert subject.ROLES['B']['backbone'] == 'cylindrical'
    assert subject.ROLES['B']['epoch'] == subject.CYL['epoch'] == 12


IDENTITY_REFUSALS = {
    'model_class': ({'model_class': 'BogusModel'}, 'model class'),
    'registry': ({'registry_sha256': 'c' * 64}, 'reviewed backbone registry'),
    'backbone': ({'backbone': 'simple'}, 'backbone'),
}


@pytest.mark.parametrize('case', sorted(IDENTITY_REFUSALS))
def test_an_arm_that_is_not_the_registered_model_is_refused(tmp_path, checkpoints, approved,
                                                            case):
    manifest, cause = IDENTITY_REFUSALS[case]
    directory = write_run(tmp_path / case, 'C', 42, checkpoints, SPLIT, manifest=manifest)
    with pytest.raises(ValueError, match=cause):
        subject.admit_run(directory, 'C', approved, SPLIT, roles=ROLES)


def test_a_baseline_that_declares_another_epoch_is_refused(tmp_path, checkpoints, approved):
    """Finding 4: B's declared epoch used to be published unchecked."""
    directory = write_run(tmp_path / 'b9', 'B', 42, checkpoints, SPLIT, route='exp06',
                          manifest={'checkpoint_epoch': 3})
    with pytest.raises(ValueError, match='checkpoint_epoch'):
        subject.admit_run(directory, 'B', approved, SPLIT, roles=ROLES)


def test_an_exp04_baseline_publishes_its_registered_epoch(tmp_path, checkpoints, approved):
    """A's historical manifest records no epoch; its registration says twelve."""
    run = subject.admit_run(write_run(tmp_path / 'a', 'A', 42, checkpoints, SPLIT), 'A',
                            approved, SPLIT, roles=ROLES)
    assert run['checkpoint']['epoch'] == 12


def test_the_exp06_output_metadata_must_agree_with_the_manifest(tmp_path, checkpoints,
                                                                approved):
    directory = write_run(tmp_path / 'meta', 'C', 42, checkpoints, SPLIT)
    per = json.loads((directory / 'per_sample_yaw.json').read_text())
    metrics = json.loads((directory / 'metrics_yaw.json').read_text())
    for payload, name in ((per, 'per_sample_yaw.json'), (metrics, 'metrics_yaw.json')):
        payload['meta']['checkpoint_role'] = 'diagnostic'
        (directory / name).write_text(json.dumps(payload))
    record = json.loads((directory / 'completion.json').read_text())
    record['outputs'] = {name: provenance.sha256_file(directory / name)
                         for name in subject.OUTPUTS}
    (directory / 'completion.json').write_text(json.dumps(record, sort_keys=True))
    with pytest.raises(ValueError, match='meta checkpoint_role'):
        subject.admit_run(directory, 'C', approved, SPLIT, roles=ROLES)


# --- finding 5: the exp_05 route establishes what it declares ------------------------------


@pytest.fixture
def exp05(monkeypatch):
    monkeypatch.setattr(subject, 'exp05_approvals',
                        lambda: ({'evaluator': CLOSURES['exp05_eval']['sha256'],
                                  'writer': CLOSURES['exp04_eval_launch']['sha256']},
                                 {'sha256': EXP05_DIGEST}))


def test_the_exp05_route_binds_the_checkpoints_own_args_json(tmp_path, checkpoints, approved,
                                                             exp05):
    directory = write_run(tmp_path / 'm', 'B', 42, checkpoints, SPLIT, route='exp05')
    run = subject.admit_run(directory, 'B', approved, SPLIT, roles=ROLES)
    args = Path(checkpoints['B']).resolve().parent / 'args.json'
    assert run['tier'] == {'tier': 'M', 'legacy_M': True,
                           'args_json': {'path': str(args),
                                         'sha256': provenance.sha256_file(args)},
                           'param_counts': dict(M_COUNTS), 'arm': 'M_cyl'}


TIER_REFUSALS = {
    'args_digest': ({'args_json_sha256': 'b' * 64}, 'args.json'),
    'counts': ({'param_counts': {'full': 1}}, 'param_counts'),
    'dim': ({'vit_dim': 768}, 'vit_dim'),
    'legacy': ({'legacy_M': False}, 'legacy_M'),
}


@pytest.mark.parametrize('case', sorted(TIER_REFUSALS))
def test_a_tier_declaration_the_checkpoint_does_not_support_is_refused(tmp_path, checkpoints,
                                                                       approved, exp05, case):
    manifest, cause = TIER_REFUSALS[case]
    directory = write_run(tmp_path / case, 'B', 42, checkpoints, SPLIT, route='exp05',
                          manifest=manifest)
    with pytest.raises(ValueError, match=cause):
        subject.admit_run(directory, 'B', approved, SPLIT, roles=ROLES)


def test_a_train_args_binding_that_is_not_the_args_json_is_refused(tmp_path, checkpoints,
                                                                   approved, exp05):
    directory = write_run(tmp_path / 'elsewhere', 'B', 42, checkpoints, SPLIT, route='exp05')
    fields = json.loads((directory / 'eval_manifest.json').read_text())
    log = directory.parent / 'B_seed42.log'
    fields['mutable_inputs'] = {'train_args': {'path': str(log),
                                               'sha256': provenance.sha256_file(log)}}
    rewrite(directory, fields)
    with pytest.raises(ValueError, match='not the checkpoint args.json'):
        subject.admit_run(directory, 'B', approved, SPLIT, roles=ROLES)


def test_tier_fields_that_disagree_across_the_outputs_are_refused(tmp_path, checkpoints,
                                                                  approved, exp05):
    directory = write_run(tmp_path / 'disagree', 'B', 42, checkpoints, SPLIT, route='exp05')
    record = json.loads((directory / 'completion.json').read_text())
    record['tier'] = 'L'
    (directory / 'completion.json').write_text(json.dumps(record, sort_keys=True))
    with pytest.raises(ValueError, match='completion tier'):
        subject.admit_run(directory, 'B', approved, SPLIT, roles=ROLES)


def test_an_args_json_that_declares_another_tier_is_refused(tmp_path, checkpoints, approved,
                                                            exp05):
    directory = write_run(tmp_path / 'other', 'B', 43, checkpoints, SPLIT, route='exp05')
    args = Path(checkpoints['B']).resolve().parent / 'args.json'
    original = args.read_text()
    args.write_text(json.dumps({'backbone': 'cylindrical', 'vit_dim': 768, 'vit_depth': 12,
                                'vit_heads': 12, 'vit_mlp_dim': 768}))
    try:
        with pytest.raises(ValueError, match='args.json'):
            subject.admit_run(directory, 'B', approved, SPLIT, roles=ROLES)
    finally:
        args.write_text(original)


def test_the_tier_is_read_from_the_bytes_the_bound_digest_identifies(tmp_path, checkpoints,
                                                                     approved, exp05,
                                                                     monkeypatch):
    """Close review 2, finding 5: hashing one snapshot and parsing another admitted a tier
    the identified file does not support -- a transient substitution during the parsing
    read described M-tier weights that hash as the legacy record beside them."""
    directory = write_run(tmp_path / 'transient', 'B', 42, checkpoints, SPLIT, route='exp05',
                          manifest={'legacy_M': False})
    args = Path(checkpoints['B']).resolve().parent / 'args.json'
    original = args.read_bytes()
    substituted = json.dumps({'backbone': 'cylindrical', 'num_shot': subject.NUM_SHOT,
                              'tier': 'M', 'vit_dim': 512, 'vit_depth': 12, 'vit_heads': 8,
                              'vit_mlp_dim': 512})
    real = Path.read_text

    def read(self, *arguments, **named):
        return substituted if str(self) == str(args) else real(self, *arguments, **named)

    monkeypatch.setattr(Path, 'read_text', read)
    with pytest.raises(ValueError, match='legacy_M'):
        subject.admit_run(directory, 'B', approved, SPLIT, roles=ROLES)
    assert args.read_bytes() == original


# --- finding 7: production approvals are the committed, reviewed bytes --------------------


def repo_head():
    return provenance.git_state(subject.REPO)['HEAD']


def first_commit():
    import subprocess
    return subprocess.check_output(['git', 'rev-list', '--max-parents=0', 'HEAD'],
                                   cwd=str(subject.REPO), text=True).split()[0]


def test_production_approvals_are_bound_to_the_reviewed_commit():
    """A well-formed record is not an approval unless a reviewer committed those bytes."""
    approved, receipt, deviations = subject.approvals(True)
    assert approved is not None and 'committed_at' not in receipt
    with pytest.raises(ValueError, match='approvals incomplete'):
        subject.approvals(False)                      # the null template, but committed
    _, receipt, _ = subject.approvals(True, commit=repo_head())
    assert 'committed_at' not in receipt


def test_an_uncommitted_approvals_file_is_refused_in_production(tmp_path):
    outside = tmp_path / 'approved_digests.json'
    outside.write_text(Path(subject.approvals_api.approvals_module()
                            .TEMPLATE_PATH).read_text())
    with pytest.raises(ValueError, match='outside the repository'):
        subject.approvals(False, str(outside))
    assert subject.approvals(True, str(outside))[0] is not None


def test_approvals_that_were_not_tracked_at_the_reviewed_commit_are_refused():
    with pytest.raises(ValueError, match='not tracked at'):
        subject.approvals(False, commit=first_commit())


def test_the_comparer_records_the_identity_of_the_approvals_it_used(runs, approved,
                                                                    tmp_path, monkeypatch):
    template = subject.REPO / 'tools/exp06_approved_digests_template.json'
    receipt = {'path': str(template), 'sha256': provenance.sha256_file(template),
               'repo_relative': 'tools/exp06_approved_digests_template.json',
               'committed_at': repo_head()}
    monkeypatch.setattr(subject, 'approvals',
                        lambda exploratory, path=None, commit=None: (approved, receipt, []))
    out, summary = tmp_path / 'h3.json', tmp_path / 'h3.txt'
    subject.main(['--runs-c'] + runs['C'] + ['--runs-a'] + runs['A'] + ['--runs-b']
                 + runs['B'] + ['--json', str(out), '--summary', str(summary),
                                '--n-boot', '200', '--exploratory'])
    record = json.loads(out.read_text())
    assert record['approved_digests']['committed_at'] == repo_head()
    assert record['inputs'][str(template.resolve())] == receipt['sha256']


def test_the_comparers_own_closure_is_bound_and_revalidated(runs, approved, tmp_path,
                                                            monkeypatch):
    """Finding 2: the producer files H3 publishes are rechecked with every other input."""
    monkeypatch.setattr(subject, 'approvals',
                        lambda exploratory, path=None, commit=None: (approved, None, []))
    out, summary = tmp_path / 'h3.json', tmp_path / 'h3.txt'
    subject.main(['--runs-c'] + runs['C'] + ['--runs-a'] + runs['A'] + ['--runs-b']
                 + runs['B'] + ['--json', str(out), '--summary', str(summary),
                                '--n-boot', '200', '--exploratory'])
    record = json.loads(out.read_text())
    for item in record['producer']['files']:
        path = (Path(subject.REPO) / item['path']).resolve()
        assert record['inputs'][str(path)] == item['working_tree_sha256'], item['path']


# --- F5: section 7's convergence rule around the inherited H3 resampling -------------------


CONSTANT = {'edt': [0.10] * N, 'c50': [1.2] * N, 't60': [7.0] * N,
            'loss': [0.5] * N, 'log_mse': [0.3] * N}


def test_the_convergence_policy_refuses_zero_width_and_retries_once():
    """Full-review F5: identical draws are not a converged interval, they are no interval."""
    import numpy as np

    def constant(seed, n_boot):
        return np.zeros(n_boot)

    draws, size, attempts = subject.converged_draws(constant, 100)
    assert [attempt['n_boot'] for attempt in attempts] == [100, 400]
    assert not any(attempt['passed'] for attempt in attempts) and size == 400
    assert all('zero-width' in (attempt['reason'] or '') for attempt in attempts)
    assert all(attempt['seed_a']['lo'] == attempt['seed_a']['hi'] == 0.0
               for attempt in attempts)


def test_a_first_attempt_that_converges_is_not_quadrupled():
    import numpy as np

    def stable(seed, n_boot):
        return np.linspace(0.0, 1.0, n_boot) + 0.001 * seed

    draws, size, attempts = subject.converged_draws(stable, 100)
    assert len(attempts) == 1 and attempts[0]['passed'] and size == 100
    assert attempts[0]['ratio'] <= subject.CONVERGENCE_TOL
    assert len(draws[subject.BOOT_SEEDS[0]]) == 100


def test_a_failed_attempt_is_retried_once_at_four_times_the_draws():
    import numpy as np

    def improving(seed, n_boot):
        shift = 0.0 if seed == subject.BOOT_SEEDS[0] else (0.5 if n_boot == 100 else 0.001)
        return np.linspace(0.0, 1.0, n_boot) + shift

    draws, size, attempts = subject.converged_draws(improving, 100)
    assert [attempt['n_boot'] for attempt in attempts] == [100, 400]
    assert [attempt['passed'] for attempt in attempts] == [False, True]
    assert size == 400 and attempts[0]['ratio'] > subject.CONVERGENCE_TOL


def test_identical_arms_withhold_the_h3_verdict(tmp_path, checkpoints, approved):
    """Full-review F5: rho = 0 on a zero-width interval used to publish `non-inferior`."""
    groups = {}
    for role in ('C', 'A', 'B'):
        groups[role] = [str(write_run(tmp_path / role / str(seed), role, seed, checkpoints,
                                      SPLIT, per_sample={'P': {'0': dict(CONSTANT)}}))
                        for seed in subject.SEEDS]
    result = analysis(groups, approved, n_boot=100)
    assert result['verdicts']['C - B']['verdict'] == 'not converged'
    assert result['verdicts']['C - A']['verdict'] == 'not converged'
    for cell in result['cells']:
        assert cell['rho'] == 0.0 and cell['convergence']['passed'] is False
        assert cell['convergence']['status'] == 'not_converged'
        assert [attempt['n_boot'] for attempt in cell['convergence']['attempts']] == [100, 400]
    assert any('convergence gate failed' in item for item in result['deviations'])


# --- F4: arm B has exactly two registered routes ------------------------------------------


def test_each_role_declares_the_routes_section_6_3_registers_for_it():
    assert subject.ROLES['C']['routes'] == ('exp06',)
    assert subject.ROLES['A']['routes'] == ('exp04',)
    assert subject.ROLES['B']['routes'] == ('exp05', 'exp06')
    assert set(subject.ROUTES) == {'exp04', 'exp05', 'exp06'}


def test_a_descriptive_exp04_cylindrical_run_is_not_arm_b(tmp_path, checkpoints, approved):
    """Full-review F4: B's unrestricted route fell back to exp_04 and was admitted."""
    directory = write_run(tmp_path / 'descriptive', 'B', 42, checkpoints, SPLIT, route='exp04')
    fields = json.loads((directory / 'eval_manifest.json').read_text())
    assert 'checkpoint_role' not in fields and 'checkpoint_epoch' not in fields
    assert subject.route_of(fields, None) == 'exp04'
    with pytest.raises(ValueError, match='registered route'):
        subject.admit_run(directory, 'B', approved, SPLIT, roles=ROLES)


def test_an_exp06_arm_must_record_its_checkpoint_epoch(tmp_path, checkpoints, approved):
    """The historical missing-epoch exception is arm A's; an exp_06 run declares one."""
    for role in ('B', 'C'):
        directory = write_run(tmp_path / ('epoch' + role), role, 42, checkpoints, SPLIT,
                              route='exp06')
        fields = json.loads((directory / 'eval_manifest.json').read_text())
        fields.pop('checkpoint_epoch')
        rewrite(directory, fields)
        with pytest.raises(ValueError) as failure:
            subject.admit_run(directory, role, approved, SPLIT, roles=ROLES)
        # The checkpoint block refuses it, before the weaker `meta checkpoint_epoch`.
        assert str(failure.value).endswith(': checkpoint_epoch')


def test_the_exp05_route_is_the_registered_twelfth_epoch_arm(tmp_path, checkpoints, approved,
                                                             exp05):
    """B's exp_05 route takes its epoch from the registration, which must be epoch 12."""
    assert str(subject.EXP05_M['epoch']) == str(subject.EPOCH)
    directory = write_run(tmp_path / 'm12', 'B', 42, checkpoints, SPLIT, route='exp05')
    roles = copy.deepcopy(ROLES)
    roles['B']['exp05'] = dict(ROLES['B']['exp05'], epoch=9)
    with pytest.raises(ValueError, match='twelfth-epoch'):
        subject.admit_run(directory, 'B', approved, SPLIT, roles=roles)


REAL_B = Path(__file__).resolve().parents[1] / 'ckpt/yaw_aug/eval/cyl_k8_seed42_k0'


@pytest.mark.skipif(not REAL_B.is_dir(),
                    reason="needs exp_04's descriptive cylindrical evaluation")
def test_the_real_descriptive_cylindrical_run_is_refused_as_arm_b():
    """The exact counterexample of the Codex full review: 6337 queries were admitted."""
    fields = json.loads((REAL_B / 'eval_manifest.json').read_text())
    approved = approvals_api.validate({
        'schema_version': 1,
        'code': dict({key: 'a' * 64 for key in approvals_api.CODE_KEYS},
                     evaluator_exp03=fields['evaluator_closure']['sha256']),
        'reused': dict({key: 'a' * 64 for key in approvals_api.REUSED_DIGESTS},
                       exp04_evaluator_closure=fields['source_closures']['entrypoint']['sha256'],
                       exp04_writer_closure=fields['source_closures']['writer']['sha256'],
                       exp04_approved_digests_sha256=EXP04_DIGEST,
                       legacy_receipt={'path': 'r.json', 'sha256': 'a' * 64}),
        'artifacts': {'epoch_012': {'path': 'p', 'epoch': 12, 'sha256': 'a' * 64},
                      'heading': {room: 'a' * 64 for room in approvals_api.ROOMS},
                      'gate_g1_sha256': 'a' * 64}})
    with pytest.raises(ValueError, match='registered route'):
        subject.admit_runs({'B': [str(REAL_B)]}, approved)


# --- F2: the training evidence is re-validated at admission --------------------------------


def test_an_exp06_run_publishes_the_training_run_it_binds(tmp_path, checkpoints, approved):
    for role in ('C', 'B'):
        directory = write_run(tmp_path / role, role, 42, checkpoints, SPLIT, route='exp06')
        fields = json.loads((directory / 'eval_manifest.json').read_text())
        assert set(fields['mutable_inputs']) == set(subject.TRAINING_BINDINGS)
        bound = fields['mutable_inputs']['train_completion']['path']
        assert Path(bound).name == ('completion.json' if role == 'C'
                                    else 'train_receipt.json')
        subject.admit_run(directory, role, approved, SPLIT, roles=ROLES)


def test_an_evaluation_log_bound_as_training_evidence_is_refused(tmp_path, checkpoints,
                                                                  approved):
    """The fixture used to satisfy both names with this run's own log; so did the runbook."""
    directory = write_run(tmp_path / 'log', 'C', 42, checkpoints, SPLIT)
    fields = json.loads((directory / 'eval_manifest.json').read_text())
    log = directory.parent / 'C_seed42.log'
    fields['mutable_inputs'] = {name: {'path': str(log), 'sha256': provenance.sha256_file(log)}
                                for name in subject.TRAINING_BINDINGS}
    rewrite(directory, fields)
    with pytest.raises(ValueError, match='training evidence'):
        subject.admit_run(directory, 'C', approved, SPLIT, roles=ROLES)


EVIDENCE_DAMAGE = {
    'other_weights': ('C', {'artifacts': {'epoch_012.pth': 'f' * 64}}, 'epoch_012.pth'),
    'not_admissible': ('C', {'admissible_arm': False}, 'not admissible as an arm'),
    'not_full': ('C', {'run_type': 'probe'}, 'run_type'),
    'short_budget': ('C', {'epochs': 9}, 'not the registered 12'),
}


@pytest.mark.parametrize('case', sorted(EVIDENCE_DAMAGE))
def test_a_completion_that_does_not_certify_these_weights_is_refused(tmp_path, checkpoints,
                                                                     approved, case):
    role, fields, cause = EVIDENCE_DAMAGE[case]
    directory = write_run(tmp_path / case, role, 42, checkpoints, SPLIT)
    completion = Path(checkpoints['train_C']) / 'completion.json'
    original = completion.read_text()
    record = json.loads(original)
    if 'artifacts' in fields:
        fields = dict(fields, artifacts=dict(record['artifacts'], **fields['artifacts']))
    completion.write_text(json.dumps(dict(record, **fields), sort_keys=True))
    manifest = json.loads((directory / 'eval_manifest.json').read_text())
    manifest['mutable_inputs']['train_completion']['sha256'] = \
        provenance.sha256_file(completion)
    rewrite(directory, manifest)
    try:
        with pytest.raises(ValueError, match=cause):
            subject.admit_run(directory, 'C', approved, SPLIT, roles=ROLES)
    finally:
        completion.write_text(original)


def test_a_receipt_whose_artefacts_moved_is_refused(tmp_path, checkpoints, approved):
    """Arm B's evidence is the retained exp_01 artefacts, re-hashed at every admission."""
    directory = write_run(tmp_path / 'stale', 'B', 42, checkpoints, SPLIT, route='exp06')
    train, _ = checkpoints['train_B']
    original = (train / 'train.log').read_bytes()
    (train / 'train.log').write_bytes(original + b'appended after the receipt\n')
    try:
        with pytest.raises(ValueError, match='changed since the receipt'):
            subject.admit_run(directory, 'B', approved, SPLIT, roles=ROLES)
    finally:
        (train / 'train.log').write_bytes(original)


def test_arm_c_must_evaluate_the_approved_epoch_012(tmp_path, checkpoints, approved):
    """The completion, the run's own digest and the approvals must name one checkpoint."""
    directory = write_run(tmp_path / 'approved', 'C', 42, checkpoints, SPLIT)
    wrong = copy.deepcopy(approved)
    wrong['artifacts']['epoch_012']['sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='approved epoch_012'):
        subject.admit_run(directory, 'C', wrong, SPLIT, roles=ROLES)
