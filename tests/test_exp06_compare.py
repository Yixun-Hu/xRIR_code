"""exp_06's H3 comparer: fail-closed admission of 6.3's runs, then exp_04's statistics."""
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from tools import exp06_approvals_api as approvals_api
from tools import exp06_compare as subject
from tools import provenance

N = 24                                  # the synthetic split; production uses 6337
SPLIT = {'split': 'unseen', 'n_queries': N}
ROOMS = ('Cafe_idx_1', 'Hall_idx_2', 'Office_idx_3')
QUERIES = ['{}/{}/S00{}_R00{}_hybrid_IR.wav'.format(ROOMS[i % 3].split('_')[0], ROOMS[i % 3],
                                                    i % 7, i) for i in range(N)]


def closure(name, tag):
    files = [{'path': 'tools/{}.py'.format(part),
              'reviewed_blob_sha256': hashlib.sha256((tag + part).encode()).hexdigest(),
              'working_tree_sha256': hashlib.sha256((tag + part).encode()).hexdigest(),
              'commits_after_reviewed': []} for part in sorted({name, 'provenance'})]
    digest = hashlib.sha256(json.dumps(
        [[item['path'], item['reviewed_blob_sha256']] for item in files],
        sort_keys=True).encode()).hexdigest()
    return {'files': files, 'sha256': digest}


CLOSURES = {name: closure(name, name) for name in
            ('exp06_eval', 'exp06_eval_launch', 'exp04_eval', 'exp04_eval_launch',
             'eval_yaw_rotation')}
ROLES = {'C': {'arm': 'cyl_or', 'checkpoint': None, 'exp06': True, 'role': 'arm'},
         'A': {'arm': 'control', 'checkpoint': {'sha256': None, 'checkpoint': 'simple.pth',
                                                'backbone': 'simple'},
               'exp06': False, 'role': 'arm'},
         'B': {'arm': 'cyl', 'checkpoint': {'sha256': None, 'checkpoint': 'cyl.pth',
                                            'backbone': 'cylindrical'},
               'exp06': None, 'role': 'baseline'}}


def approved_digests(epoch_012_sha):
    value = {'schema_version': 1,
             'code': dict({key: 'a' * 64 for key in approvals_api.CODE_KEYS},
                          eval=CLOSURES['exp06_eval']['sha256'],
                          eval_launch=CLOSURES['exp06_eval_launch']['sha256'],
                          evaluator_exp03=CLOSURES['eval_yaw_rotation']['sha256']),
             'reused': dict({key: 'a' * 64 for key in approvals_api.REUSED_DIGESTS},
                            exp04_evaluator_closure=CLOSURES['exp04_eval']['sha256'],
                            exp04_writer_closure=CLOSURES['exp04_eval_launch']['sha256'],
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


def write_run(directory, role, seed, checkpoints, split=SPLIT, manifest_hash=None,
              manifest=None, per_sample=None, completion=None, bad=()):
    """One admissible evaluation run of arm `role`, before the caller's mutations."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    exp06 = ROLES[role]['exp06'] is not False
    checkpoint = checkpoints['C' if role == 'C' else role]
    reference = directory.parent / 'reference_manifest_seed{}.json'.format(seed)
    if not reference.exists():
        reference.write_text(json.dumps({'seed': seed}))
    fields = {'schema_version': 1, 'confirmatory': True, 'allow_dirty_used': False,
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
              'manifest_hash': manifest_hash or ('b' * 63 + str(seed - 42)),
              'evaluator_closure': copy.deepcopy(CLOSURES['eval_yaw_rotation']),
              'source_closures': {
                  'entrypoint': copy.deepcopy(CLOSURES['exp06_eval' if exp06 else 'exp04_eval']),
                  'writer': copy.deepcopy(CLOSURES['exp04_eval_launch'])}}
    if exp06:
        fields['source_closures']['writer_exp06'] = copy.deepcopy(CLOSURES['exp06_eval_launch'])
        fields.update(registry_sha256='c' * 64, checkpoint_role=ROLES[role]['role'],
                      checkpoint_epoch=12, frame='room', heading=None,
                      model_class='xRIR_CylOriented')
    fields.update(manifest or {})
    meta = {key: fields[key] for key in ('backbone', 'checkpoint', 'manifest_hash', 'gl_seed',
                                         'batch_size', 'tf32', 'manifest_seed', 'yaw_cols',
                                         'conditions', 'n_samples')}
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
    (directory / 'metrics_yaw.json').write_text(json.dumps({'meta': per['meta'], 'P': {'0': {}}}))
    record = {'schema_version': 1, 'child_exit_status': 0, 'eval_manifest_sha256': meta_sha,
              'directory_listing': sorted(n for n in subject.FILES if n != 'completion.json'),
              'outputs': {name: provenance.sha256_file(directory / name)
                          for name in subject.OUTPUTS}}
    record.update(completion or {})
    (directory / 'completion.json').write_text(json.dumps(record, sort_keys=True))
    return directory


@pytest.fixture(scope='module')
def checkpoints(tmp_path_factory):
    base = tmp_path_factory.mktemp('ckpt')
    paths = {}
    for role, name in (('C', 'epoch_012.pth'), ('A', 'simple.pth'), ('B', 'cyl.pth')):
        path = base / name
        path.write_bytes(name.encode() * 8)
        paths[role] = path
        if role != 'C':
            ROLES[role]['checkpoint']['sha256'] = provenance.sha256_file(path)
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
    assert subject.SPLIT == {'split': 'unseen', 'n_queries': 6337}
    assert subject.BATCH_SIZE == 16 and subject.CONDITIONS == 'P'
    assert subject.YAW_COLS == [0] and subject.SUPERIORITY_ALPHA == 0.025
    assert subject.CONTRASTS == (('C', 'B'), ('C', 'A'))
    assert sorted(subject.ROLES) == ['A', 'B', 'C']
    assert subject.ROLES['A']['checkpoint']['checkpoint'].endswith('epoch_12.pth')


def test_every_arm_is_admitted_with_its_five_seeds(runs, approved):
    admitted = subject.admit_runs(runs, approved, split=SPLIT, roles=ROLES)
    assert sorted(admitted['groups']) == ['A', 'B', 'C'] and not admitted['deviations']
    assert [run['seed'] for run in admitted['groups']['C']] == list(subject.SEEDS)
    assert all(path.endswith(('.json', '.pth')) for path in admitted['inputs'])
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
