"""The frozen exp_07 (seen protocol) profiles and the literals they pin."""
import importlib
from types import MappingProxyType as MP

import pytest

from tools import exp07_profiles as profiles
from tools import provenance as prov

ROOT = profiles.REPO
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
