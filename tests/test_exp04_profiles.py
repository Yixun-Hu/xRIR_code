import hashlib
import importlib
import json
import subprocess
from pathlib import Path
from types import MappingProxyType

import pytest

from tools import exp04_profiles as profiles

ROOT = Path(__file__).resolve().parents[1]


def test_profiles_frozen_and_typed():
    assert set(profiles.PROFILES) == {'H1_K8', 'H1_K1', 'H2_K8', 'TOST_K8', 'TABLE_V1'}
    def check(value):
        assert type(value) in (MappingProxyType, tuple, str, int, float, bool, type(None))
        if isinstance(value, MappingProxyType):
            with pytest.raises(TypeError):
                value['changed'] = True
            for item in value.values():
                check(item)
        elif isinstance(value, tuple):
            with pytest.raises(TypeError):
                value[0] = None
            for item in value:
                check(item)
    check(profiles.PROFILES)
    for name, profile in profiles.PROFILES.items():
        assert profiles.get_profile(name) is profile
        assert type(profile['schema_version']) is int
        assert profile['mode'] in ('two_arm', 'one_arm', 'table')
        assert profile['condition'] == 'P'
        assert profile['alpha'] == .05 and profile['n_boot'] == 20000
        assert profile['bootstrap_seeds'] == (0, 1)
        assert profile['convergence_tolerance'] == .10
        assert profile['batch_size'] == 16 and profile['max_samples'] == 0
        assert profile['tf32'] is False
        assert 'approved_closures' not in profile
    with pytest.raises(KeyError):
        profiles.get_profile('UNKNOWN')


def test_canonical_digest_stable_across_imports():
    before = {name: profiles.profile_digest(name) for name in profiles.PROFILES}
    importlib.reload(profiles)
    for name, digest in before.items():
        raw = json.dumps(profiles.json_value(profiles.get_profile(name)),
                         sort_keys=True, separators=(',', ':'), allow_nan=False)
        assert profiles.profile_digest(name) == hashlib.sha256(raw.encode()).hexdigest() == digest


def test_hypotheses_and_canonical_input_selection():
    for name in ('H1_K8', 'H1_K1'):
        p = profiles.get_profile(name)
        assert p['grid'] == (0,) and p['input_selection'] == 'standalone_k0'
        assert p['family'] == 2 and p['margin'] == .03
        assert p['metrics']['primary'] == ('EDT', 'C50')
    h2, tost = (profiles.get_profile(name) for name in ('H2_K8', 'TOST_K8'))
    assert h2['grid'] == (32, 64, 448, 480) and h2['family'] == 8
    assert h2['metrics']['primary'] == ('C50',) and h2['metrics']['supportive'] == ('EDT',)
    assert h2['run_grids']['aug'] == (0, 32, 64, 128, 256, 384, 448, 480)
    assert h2['run_grids']['control'] == (0, 32, 64, 448, 480)
    assert tost['grid'] == (32, 64, 128, 256, 384, 448, 480)
    assert tost['family'] == 14 and tost['margin'] == .02
    assert [a['role'] for a in tost['arms']] == ['aug']
    assert h2['input_selection'] == tost['input_selection'] == 'block'


@pytest.mark.parametrize('name', tuple(profiles.PROFILES))
def test_every_field_has_its_declared_type(name):
    p = profiles.get_profile(name)
    schema = {'schema_version': int, 'mode': str, 'arms': tuple, 'num_shot': int,
              'condition': str, 'grid': tuple, 'input_selection': str, 'run_grids': MappingProxyType,
              'metrics': MappingProxyType, 'family': int, 'margin': float, 'alpha': float,
              'tails': str, 'n_boot': int, 'bootstrap_seeds': tuple, 'convergence_tolerance': float,
              'seeds': MappingProxyType, 'gl_seed_rule': str, 'dataset': MappingProxyType,
              'max_samples': int, 'batch_size': int,
              'tf32': bool, 'companion_alpha': float, 'superiority_alpha': type(None)}
    if name.startswith('H1'):
        schema['superiority_alpha'] = float
    if name == 'TABLE_V1':
        schema.update(num_shot=tuple, margin=type(None), companion_alpha=type(None),
                      finite_count_tolerance=int)
        assert p['finite_count_tolerance'] == 2
    assert p.keys() == schema.keys()
    assert all(type(p[key]) is kind for key, kind in schema.items())
    for arm in p['arms']:
        assert set(arm) == {'label', 'backbone', 'role', 'epoch', 'checkpoint'} | (
            set() if arm['role'] == 'aug' else {'sha256'})
        assert type(arm['epoch']) is int and arm['epoch'] == 12
        assert all(type(arm[key]) is str for key in ('label', 'backbone', 'role', 'checkpoint'))
        assert 'sha256' not in arm if arm['role'] == 'aug' else len(arm['sha256']) == 64
    detached = profiles.json_value(p)
    detached['arms'][0]['label'] = 'changed'
    assert p['arms'][0]['label'] != 'changed'
    with pytest.raises(AttributeError):
        p.changed = True


@pytest.mark.parametrize('shot', (1, 8))
@pytest.mark.parametrize('seed', range(42, 47))
def test_pinned_references_and_query_identity(shot, seed):
    from tools.reference_manifest import load_manifest, manifest_hash
    from tools.provenance import sha256_file
    index = json.loads((ROOT / 'ckpt/yaw_aug/reference_manifests_index.json').read_text())
    entry = index['k{}_seed{}'.format(shot, seed)]
    path = ROOT / entry['path']
    reference = load_manifest(path)
    assert profiles.REFERENCES[shot][seed] == entry['hash'] == manifest_hash(reference)
    assert sha256_file(path) == entry['file_sha256']
    assert reference['seed'] == seed and reference['num_shot'] == shot
    queries = [row['query'] for row in reference['entries']]
    digest = hashlib.sha256(json.dumps(queries, separators=(',', ':')).encode()).hexdigest()
    dataset = profiles.COMMON['dataset']
    assert digest == dataset['query_sha256']
    assert len(queries) == dataset['n_queries'] == entry['entries']
    assert len({q.split('/')[1] for q in queries}) == dataset['n_rooms']


@pytest.mark.parametrize('arm', (profiles.CONTROL, profiles.CYL))
def test_pinned_checkpoints(arm):
    from tools.provenance import sha256_file
    path = ROOT / arm['checkpoint']
    if not path.exists():
        pytest.skip('checkpoint absent')
    assert sha256_file(path) == arm['sha256']


def test_nested_field_types_and_shared_inventory_pin():
    dataset = profiles.COMMON['dataset']
    assert type(dataset['n_queries']) is int and type(dataset['n_rooms']) is int
    assert type(dataset['split']) is str and type(dataset['query_sha256']) is str
    assert dataset['inventory_sha256'] == '23c3d8f6a0f740f54cb7d5db5766a80e82c6a78d60c05744e7542529a86a3092'
    for shot, seeds in profiles.REFERENCES.items():
        assert type(shot) is int and set(seeds) == set(range(42, 47))
        for seed, digest in seeds.items():
            assert type(seed) is int and type(digest) is str and len(digest) == 64
            reference = json.loads((ROOT / 'ckpt/yaw_aug/reference_manifest_k{}_seed{}.json'
                                    .format(shot, seed)).read_text())
            queries = {e['query'] for e in reference['entries']}
            assert {p for e in reference['entries'] for p in e['refs']} <= queries
    for p in profiles.PROFILES.values():
        assert all(type(k) is int for k in p['grid'] + p['bootstrap_seeds'])
        assert all(type(grid) is tuple and all(type(k) is int for k in grid)
                   for grid in p['run_grids'].values())
        assert all(type(names) is tuple and all(type(name) is str for name in names)
                   for names in p['metrics'].values())


def approval_repo(tmp_path, value):
    path = tmp_path / 'approved.json'
    path.write_text(json.dumps(value))
    for args in (['init', '-q'], ['add', path.name], ['-c', 'user.name=Test', '-c',
                 'user.email=test@example.com', 'commit', '-qm', 'pins', '--no-gpg-sign']):
        subprocess.check_call(['git', '-C', str(tmp_path)] + args)
    return path


def approval_template():
    return {'schema_version': None, 'closures': dict.fromkeys((
        'evaluator', 'writer', 'training_launcher', 'producer_paired_compare',
        'producer_results_table')), 'checkpoints': {'aug': dict.fromkeys(('path', 'epoch', 'sha256'))}}


@pytest.mark.parametrize('pinned', (False, True))
def test_runtime_approval_pins_frozen_and_committed(tmp_path, pinned):
    value = approval_template()
    if pinned:
        value.update(schema_version=1)
        value['closures'] = dict.fromkeys(value['closures'], 'a' * 64)
        value['checkpoints']['aug'] = {'path': 'ckpt/epoch_012.pth', 'epoch': 12, 'sha256': 'b' * 64}
    path = approval_repo(tmp_path, value)
    pins, identity = profiles.load_approved_digests(path)
    assert profiles.json_value(pins) == value
    with pytest.raises(TypeError):
        pins['checkpoints']['aug']['epoch'] = 11
    assert identity['path'] == str(path.resolve())
    assert identity['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity['git_blob'] == subprocess.check_output(
        ['git', '-C', str(tmp_path), 'rev-parse', 'HEAD:approved.json']).decode().strip()
    path.write_text(json.dumps(value) + '\n')
    with pytest.raises(ValueError, match='committed'):
        profiles.load_approved_digests(path)


@pytest.mark.parametrize('section,key,value', [('', 'extra', None), ('', 'schema_version', True),
    ('', 'schema_version', 2), ('closures', 'evaluator', 'invalid'), ('closures', 'extra', None),
    ('aug', 'epoch', True), ('aug', 'path', ''), ('aug', 'sha256', 12), ('aug', 'extra', None)])
def test_approval_schema_refuses_malformed_values(tmp_path, section, key, value):
    pins = approval_template()
    target = pins['checkpoints']['aug'] if section == 'aug' else pins.get(section, pins)
    target[key] = value
    with pytest.raises(ValueError, match='schema'):
        profiles.load_approved_digests(approval_repo(tmp_path, pins))


@pytest.mark.parametrize('key,value', [('schema_version', 1), ('evaluator', 'a' * 64), ('filled', None)])
def test_mixed_approval_schema_refused(tmp_path, key, value):
    pins = approval_template()
    if key == 'filled':
        pins['closures'] = dict.fromkeys(pins['closures'], 'a' * 64)
        pins['checkpoints']['aug'] = dict(path='ckpt/epoch_012.pth', epoch=12, sha256='b' * 64)
    else:
        (pins if key == 'schema_version' else pins['closures'])[key] = value
    with pytest.raises(ValueError, match='all-null or all-filled'):
        profiles.load_approved_digests(approval_repo(tmp_path, pins))
