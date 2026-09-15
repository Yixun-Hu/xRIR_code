"""The frozen exp_07 (seen protocol) profiles and the literals they pin."""
import hashlib
import importlib
import json
import os
import re
import shlex
from pathlib import Path
from types import MappingProxyType as MP

import pytest

from tools import exp07_profiles as profiles
from tools import provenance as prov
from tools.exp07_manifests import manifest_name
from tools.paired_compare import _digest
from tools.reference_manifest import load_manifest
from test_exp04_profiles import approval_repo

ROOT = profiles.REPO
MANIFESTS = ROOT / 'ckpt/exp07'
INDEX = MANIFESTS / 'reference_manifests_seen_index.json'
GOLDEN = str(ROOT / ('worklog/worklog_yixun/exp_07_seen_protocol_claude/'
                     'seen_protocol_results_assets/argv_golden_{}.txt'))
NEW_ARMS = [('seen_simple', 'simple', 0), ('seen_cyl', 'cylindrical', 0),
            ('seen_aug', 'simple', 1)]


def test_profiles_are_frozen_and_their_digests_are_stable():
    def check(value):
        assert type(value) in (MP, tuple, str, int, float, bool, type(None))
        if isinstance(value, MP):
            with pytest.raises(TypeError):
                value['changed'] = 1
            for item in value.values():
                check(item)
        elif isinstance(value, tuple):
            for item in value:
                check(item)
    check(profiles.PROFILES)
    assert set(profiles.PROFILES) == {'TABLE_SEEN_V1', 'PAIRS_SEEN_V1'}
    before = {name: profiles.profile_digest(name) for name in profiles.PROFILES}
    importlib.reload(profiles)
    assert before == {name: profiles.profile_digest(name) for name in profiles.PROFILES}
    with pytest.raises(KeyError):
        profiles.get_profile('TABLE_V1')


def test_the_table_profile_states_the_seen_protocol_and_its_evaluation():
    table = profiles.get_profile('TABLE_SEEN_V1')
    assert table['mode'] == 'table' and table['protocol'] == 'seen' and table['tier'] == 'M'
    assert table['num_shot'] == (1, 8) and table['grid'] == (0,) and table['condition'] == 'P'
    assert table['batch_size'] == 16 and table['tf32'] is False and table['max_samples'] == 0
    assert table['finite_count_tolerance'] == 2 and table['train_batches_per_epoch'] == 9265
    assert table['eval_seeds'] == (42, 43, 44, 45, 46) and table['seed_sd_ddof'] == 1
    dataset = table['dataset']
    assert (dataset['split'], dataset['n_queries'], dataset['n_rooms']) == ('seen', 6217, 131)
    assert dataset['seen_split_sha256'] == prov.sha256_file(ROOT / prov.SEEN_SPLIT)
    assert set(table['run_grids']) == {arm['role'] for arm in table['arms']}
    assert set(table['run_grids'].values()) == {(0,)}


def test_the_arms_are_the_three_new_trainings_and_the_released_reference():
    arms = {arm['role']: arm for arm in profiles.get_profile('TABLE_SEEN_V1')['arms']}
    assert list(arms) == ['seen_simple', 'seen_cyl', 'seen_aug', 'released_seen']
    for role, backbone, yaw in NEW_ARMS:
        arm = arms[role]
        assert arm['checkpoint'] == 'ckpt/exp07/{}/final/epoch_012.pth'.format(role)
        assert arm['epoch'] == 12 and arm['backbone'] == backbone and arm['protocol'] == 'seen'
        assert (arm['yaw_aug'], arm['yaw_aug_seed'], arm['yaw_aug_width']) == (yaw, 0, 512)
        assert arm['training'] == 'internal' and arm['reference'] is False
    released = arms['released_seen']
    assert released['checkpoint'] == 'checkpoints/xRIR_seen.pth' and released['epoch'] is None
    assert released['training'] == 'external' and released['reference'] is True
    assert released['backbone'] == 'simple' and released['protocol'] is None
    assert 'reference' in released['label'].lower()


def test_the_released_checkpoint_digest_is_the_pinned_file():
    path = ROOT / 'checkpoints/xRIR_seen.pth'
    if not path.exists():
        pytest.skip('the released checkpoint is not present in this checkout')
    released = next(arm for arm in profiles.get_profile('TABLE_SEEN_V1')['arms']
                    if arm['role'] == 'released_seen')
    assert released['sha256'] == prov.sha256_file(path)


def test_the_recipe_and_the_full_run_constraints_are_the_plans_literals():
    table = profiles.get_profile('TABLE_SEEN_V1')
    assert profiles.json_value(table['recipe']) == dict(
        num_shot=8, max_len=9600, lr=.001, weight_decay=.0001, decay_epochs=3, lr_gamma=.1,
        epochs=12, batch_size=32, accum_steps=2, num_workers=12, seed=0, tf32=True,
        log_interval=50, save_every=0, epoch_ckpt_every=1, protocol='seen')
    assert profiles.json_value(table['full_run']) == dict(
        no_save=False, resume=None, max_train_batches=0, max_test_batches=0, test_subset=0)
    assert table['recipe']['tf32'] is True and table['full_run']['no_save'] is False
    assert all(type(value) is not bool for key, value in table['recipe'].items() if key != 'tf32')
    assert type(table['recipe']['lr']) is float and type(table['recipe']['epochs']) is int


@pytest.mark.parametrize('role,backbone,yaw', NEW_ARMS)
def test_the_recipe_matches_the_launchers_golden_argv(role, backbone, yaw):
    """The profile and tools/exp04_launcher.py must describe the same training."""
    tokens, flags, index = shlex.split(Path(GOLDEN.format(role)).read_text()), {}, 0
    while index < len(tokens):
        if tokens[index].startswith('--'):
            name = tokens[index][2:].replace('-', '_')
            following = tokens[index + 1] if index + 1 < len(tokens) else '--'
            flags[name] = True if following.startswith('--') else following
            index += 1 if following.startswith('--') else 2
        else:
            index += 1
    table = profiles.get_profile('TABLE_SEEN_V1')
    assert flags['backbone'] == backbone and flags['protocol'] == 'seen'
    assert flags.get('yaw_aug', '0') == str(yaw)
    for key, expected in profiles.json_value(table['recipe']).items():
        actual = flags[key]
        if isinstance(expected, bool):
            assert actual is True
        elif isinstance(expected, float):
            assert float(actual) == expected
        else:
            assert actual == str(expected)


def test_the_pairs_profile_is_descriptive_with_absolute_and_relative_statistics():
    pairs = profiles.get_profile('PAIRS_SEEN_V1')
    table = profiles.get_profile('TABLE_SEEN_V1')
    assert pairs['mode'] == 'pairs' and pairs['decision_driving'] is False
    assert pairs['family'] == 0 and 'verdict' not in pairs and pairs['tails'] == 'descriptive'
    assert pairs['pairings'] == (('seen_cyl', 'seen_simple'), ('seen_aug', 'seen_simple'),
                                 ('released_seen', 'seen_simple'))
    assert pairs['reference_pairings'] == (('released_seen', 'seen_simple'),)
    assert pairs['num_shot'] == (8, 1) and pairs['statistics'] == ('absolute', 'relative')
    assert pairs['metrics']['descriptive'] == ('EDT', 'C50', 'T60', 'loss', 'log_mse')
    assert (pairs['metrics']['primary'], pairs['metrics']['supportive']) == ((), ())
    assert pairs['n_boot'] == 20000 and pairs['bootstrap_seeds'] == (0, 1)
    assert pairs['convergence_tolerance'] == .10 and pairs['quantiles'] == (.025, .975)
    assert pairs['interval_alpha'] == .05 and pairs['cluster'] == 'whole_room_query_weighted'
    assert pairs['dataset'] == table['dataset'] and pairs['arms'] == table['arms']


def test_identities_that_do_not_exist_yet_are_placeholders_or_digests():
    """None = supplied at pin finalisation; the agreement tests below check filled values."""
    table = profiles.get_profile('TABLE_SEEN_V1')
    pinned = [table['dataset']['query_sha256'], table['dataset']['inventory_sha256']]
    pinned += [arm['sha256'] for arm in table['arms'] if arm['role'] != 'released_seen']
    pinned += [value for shot in (8, 1) for value in table['seeds'][shot].values()]
    assert set(table['seeds']) == {8, 1}
    assert all(set(table['seeds'][shot]) == set(table['eval_seeds']) for shot in (8, 1))
    for value in pinned:
        assert value is None or re.fullmatch('[0-9a-f]{64}', value)


def test_the_manifest_pins_agree_with_the_index_when_it_exists():
    if not INDEX.exists():
        pytest.skip('the seen reference manifests are not built yet')
    index = json.loads(INDEX.read_text())
    table = profiles.get_profile('TABLE_SEEN_V1')
    assert index['seen_split'] == {'path': prov.SEEN_SPLIT,
                                   'sha256': table['dataset']['seen_split_sha256']}
    assert index['entries'] == table['dataset']['n_queries']
    for shot in (8, 1):
        for seed in table['eval_seeds']:
            record = index['manifests'][manifest_name(shot, seed)]
            assert table['seeds'][shot][seed] == record['manifest_hash']
            assert prov.sha256_file(MANIFESTS / manifest_name(shot, seed)) == record['file_sha256']


def test_the_query_digest_agrees_with_the_seed_42_manifest_when_it_exists():
    path = MANIFESTS / manifest_name(8, 42)
    if not path.exists():
        pytest.skip('the seen reference manifests are not built yet')
    table = profiles.get_profile('TABLE_SEEN_V1')
    queries = [entry['query'] for entry in load_manifest(path)['entries']]
    assert len(queries) == table['dataset']['n_queries']
    assert table['dataset']['query_sha256'] == _digest(queries)


@pytest.fixture
def template():
    """The committed template's shape, independent of the real file's fill state."""
    return dict(schema_version=None, closures=dict(
        evaluator=None, writer=None, training_launcher=[], training=None,
        producer_table=None, producer_pairs=None), checkpoints={
            role: dict.fromkeys(('path', 'epoch', 'sha256'))
            for role, _, _ in NEW_ARMS})


def filled(value):
    value['schema_version'] = 1
    value['closures'] = dict.fromkeys(value['closures'], 'a' * 64)
    value['closures']['training_launcher'] = ['b' * 64]
    for role in value['checkpoints']:
        arm = next(a for a in profiles.ARMS if a['role'] == role)
        value['checkpoints'][role] = dict(path=arm['checkpoint'], epoch=12, sha256='c' * 64)
    return value


def test_the_all_null_template_loads_and_pins_nothing(tmp_path, template):
    """The unfinalised state, read from an isolated copy of the committed schema."""
    pins, _ = profiles.load_approved_digests(approval_repo(tmp_path, template))
    assert profiles.json_value(pins) == template
    assert all(pins['closures'][key] in (None, ()) for key in profiles.CLOSURES)
    assert all(value is None for arm in pins['checkpoints'].values() for value in arm.values())


def test_the_committed_approval_file_is_a_valid_lifecycle_state(template):
    """Before finalisation the committed file is the template; afterwards it is filled.

    The loader refuses every half-filled state, so both are complete; which one is
    committed is a fact about the experiment's progress, not about the schema.
    """
    pins, receipt = profiles.load_approved_digests()
    raw = profiles.APPROVED_DIGESTS_PATH.read_bytes()
    value = json.loads(raw)
    assert profiles.json_value(pins) == value
    assert receipt['sha256'] == hashlib.sha256(raw).hexdigest() and len(receipt['git_blob']) == 40
    assert receipt['path'] == str(profiles.APPROVED_DIGESTS_PATH.resolve())
    assert set(value) == set(template) and set(value['closures']) == set(template['closures'])
    assert set(value['checkpoints']) == set(template['checkpoints'])
    pinned = [value['schema_version']] + [value['closures'][key] for key in template['closures']]
    pinned += [item for arm in value['checkpoints'].values() for item in arm.values()]
    assert value == template or all(item not in (None, [], '') for item in pinned)


@pytest.mark.parametrize('role,backbone,yaw', NEW_ARMS)
def test_the_new_arm_checkpoint_digests_come_from_the_runtime_approval(role, backbone, yaw):
    """The profile placeholder documents the path and epoch only.

    tools.exp07_table.admit replaces it with approved_digests.json's digest before any
    run is matched, so a trained arm is pinned by the approval file and by nothing else.
    """
    arm = next(item for item in profiles.ARMS if item['role'] == role)
    path = ROOT / arm['checkpoint']
    if arm['sha256'] is None:
        assert not path.exists() or prov.sha256_file(path)  # the approval pins it
    else:
        assert path.exists() and arm['sha256'] == prov.sha256_file(path)


def test_the_evaluation_inventory_pin_is_the_manifests_data_identity():
    """The pinned identity is what tools/exp04_eval_launch.py records for every run."""
    path, root = MANIFESTS / manifest_name(8, 42), os.environ.get('XRIR_DATA_PATH')
    if not path.exists() or not root or not Path(root).is_dir():
        pytest.skip('the seen manifests or the AcousticRooms root are not present')
    table = profiles.get_profile('TABLE_SEEN_V1')
    identity = prov.data_identity(path, root)
    assert identity['inventory_sha256'] == table['dataset']['inventory_sha256']
    assert prov._inventory_digest(identity['inventory']) == identity['inventory_sha256']


def test_a_filled_approval_is_accepted_and_frozen(tmp_path, template):
    value = filled(template)
    pins, receipt = profiles.load_approved_digests(approval_repo(tmp_path, value))
    assert profiles.json_value(pins) == value and len(receipt['git_blob']) == 40
    with pytest.raises(TypeError):
        pins['checkpoints']['seen_cyl']['epoch'] = 11


def test_a_loaded_approval_is_recursively_immutable(tmp_path, template):
    pins, _ = profiles.load_approved_digests(approval_repo(tmp_path, filled(template)))
    for pins in (pins, profiles.load_approved_digests()[0]):
        launchers = pins['closures']['training_launcher']
        with pytest.raises(AttributeError):
            launchers.append('f' * 64)
        with pytest.raises(TypeError):
            launchers[0:0] = ['f' * 64]


@pytest.mark.parametrize('mutate', [
    lambda v: v.update(schema_version=1),                       # partially filled
    lambda v: v['closures'].update(evaluator='a' * 64),         # partially filled
    lambda v: v.update(schema_version=True),
    lambda v: v['closures'].update(producer_pairs='not a digest'),
    lambda v: v['closures'].pop('producer_table'),
    lambda v: v['closures'].update(extra=None),
    lambda v: v['checkpoints'].pop('seen_aug'),
    lambda v: v['checkpoints']['seen_cyl'].update(path='ckpt/exp07/other/epoch_012.pth'),
    lambda v: v['checkpoints']['seen_cyl'].update(epoch=11),
    lambda v: v['closures'].update(training_launcher='a' * 64),
    lambda v: v['closures'].update(training_launcher=[None]),
])
def test_partial_or_invalid_pins_are_refused(tmp_path, template, mutate):
    mutate(template)
    with pytest.raises(ValueError, match='schema'):
        profiles.load_approved_digests(approval_repo(tmp_path, template))


@pytest.mark.parametrize('mutate', [lambda v: v['closures'].update(training_launcher=[]),
                                    lambda v: v['checkpoints']['seen_aug'].update(sha256=None)])
def test_a_filled_approval_missing_one_pin_is_refused(tmp_path, template, mutate):
    mutate(filled(template))
    with pytest.raises(ValueError, match='all-null or all-filled'):
        profiles.load_approved_digests(approval_repo(tmp_path, template))


def test_bytes_that_differ_from_the_committed_ones_are_refused(tmp_path, template):
    path = approval_repo(tmp_path, template)
    path.write_text(json.dumps(template) + '\n')
    with pytest.raises(ValueError, match='HEAD'):
        profiles.load_approved_digests(path)
