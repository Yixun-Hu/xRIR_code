import hashlib
import importlib
import json
from types import MappingProxyType as MP

import pytest
from tools import exp05_profiles as p
from tools.exp05_params import TIERS, build_tier, count_parameters
from tests.test_exp04_profiles import approval_repo, ROOT
from tools.reference_manifest import load_manifest, manifest_hash
from tools.provenance import sha256_file


def test_frozen_profiles_and_stable_digests():
    def frozen(value):
        assert type(value) in (MP, tuple, str, int, float, bool, type(None))
        if isinstance(value, MP):
            with pytest.raises(TypeError):
                value['changed'] = 1
            for item in value.values():
                frozen(item)
        elif isinstance(value, tuple):
            for item in value:
                frozen(item)
    frozen(p.PROFILES)
    before = {name: p.profile_digest(name) for name in p.PROFILES}
    importlib.reload(p)
    assert before == {name: p.profile_digest(name) for name in p.PROFILES}
    for k in (8, 1):
        curve, target = p.get_profile('CURVE_K' + str(k)), p.get_profile('TARGETS_K' + str(k))
        assert curve['family'] == 6 and curve['interval_alpha'] == .05 / 6
        assert target['family'] == 4 and target['interval_alpha'] == .0125
        assert target['tost_alpha'] == .0125 and target['margin'] == .03
        assert len(target['pairings']) == 2 and curve['bootstrap_seeds'] == (0, 1)
        assert curve['n_boot'] == 20000 and curve['convergence_tolerance'] == .10
    assert p.YAW_K8_SEED42['grid'] == (0, 32, 64, 448, 480)
    assert p.YAW_K8_SEED42['eval_seeds'] == (42,)


@pytest.mark.parametrize('arm', p.ARMS)
def test_literal_counts_and_checkpoint(arm):
    assert arm['config'] == TIERS[arm['tier']]
    assert arm['counts'] == count_parameters(build_tier(arm['backbone'], arm['tier']))
    assert all(type(v) is int for v in arm['counts'].values())
    if arm['tier'] == 'M':
        assert sha256_file(ROOT / arm['checkpoint']) == arm['sha256']


@pytest.mark.parametrize('shot,seed', [(k, s) for k in (1, 8) for s in range(42, 47)])
def test_reference_pins(shot, seed):
    entry = json.loads((ROOT / 'ckpt/yaw_aug/reference_manifests_index.json').read_text())['k{}_seed{}'.format(shot, seed)]
    assert p.REFERENCES[shot][seed] == entry['hash'] == manifest_hash(load_manifest(ROOT / entry['path']))
    assert sha256_file(ROOT / entry['path']) == entry['file_sha256']


def test_approval_commit_binding_and_types(tmp_path):
    value = json.loads(p.APPROVED_DIGESTS_PATH.read_text())
    path = approval_repo(tmp_path, value)
    pins, receipt = p.load_approved_digests(path)
    assert receipt['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert pins['checkpoints']['S_cyl']['sha256'] is None
    path.write_text(json.dumps(value) + '\n')
    with pytest.raises(ValueError, match='HEAD'):
        p.load_approved_digests(path)
    value['schema_version'] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='schema'):
        p.load_approved_digests(path)


@pytest.mark.parametrize('wrong_role', [None, 'S_simple', 'S_cyl', 'L_simple', 'L_cyl'])
def test_filled_approvals_are_frozen_and_distinct_from_template(tmp_path, wrong_role):
    value = json.loads(p.APPROVED_DIGESTS_PATH.read_text())
    value['schema_version'] = 1
    value['closures'] = dict.fromkeys(value['closures'], 'a'*64)
    for role in value['checkpoints']:
        arm = next(a for a in p.ARMS if a['role'] == role)
        value['checkpoints'][role] = dict(path=arm['checkpoint'], epoch=12, sha256='b'*64)
    if wrong_role:
        value['checkpoints'][wrong_role]['path'] = 'ckpt/another/epoch_012.pth'
        with pytest.raises(ValueError, match='schema'):
            p.load_approved_digests(approval_repo(tmp_path, value))
        return
    pins, receipt = p.load_approved_digests(approval_repo(tmp_path, value))
    assert p.json_value(pins) == value and len(receipt['git_blob']) == 40
    with pytest.raises(TypeError):
        pins['checkpoints']['S_cyl']['epoch'] = 11


@pytest.mark.parametrize('key,value', [('schema_version', 1), ('evaluator', 'g'*64),
    ('evaluator', 'a'*64), ('epoch', True), ('epoch', 11), ('path', ''), ('sha256', 3), ('extra', None)])
def test_approval_partial_or_invalid_pin_refused(tmp_path, key, value):
    pins = json.loads(p.APPROVED_DIGESTS_PATH.read_text())
    target = (pins['closures'] if key == 'evaluator' else pins['checkpoints']['S_cyl']
              if key in ('epoch', 'path', 'sha256') else pins)
    target[key] = value
    with pytest.raises(ValueError, match='schema'):
        p.load_approved_digests(approval_repo(tmp_path, pins))
