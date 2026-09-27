"""exp_11's two named pretraining profiles (plan v3 section 4.11 / item 5).

``H_RECIPE`` is exp_01's numerical recipe with exp_06's operational differences;
``I_RECIPE`` is exp_04's yaw-augmented one, with its saving/resume constraints and the
augmentation counter bound. Every trainer field belongs to exactly one class, unknown
fields are refused by name, comparisons are type-strict and a mixture of the two
profiles is never admitted.
"""
import json
from pathlib import Path

import pytest

from tools import exp11_recipe as recipe

REPO = Path(__file__).resolve().parents[1]
EXP01_ARGS = REPO / 'ckpt/xRIR_simple_8_shot/args.json'
EXP04_ARGS = REPO / 'ckpt/xRIR_simple_yawaug_8_shot/final/args.json'


def args_of(profile, **overrides):
    """A complete, admissible args.json for one profile."""
    name = recipe.PROFILES[profile]
    value = dict(name['recipe'], yaw_aug_seed=0, yaw_aug_width=512)
    value.update(name['production'])
    value.update(backbone=name['backbone'], save_dir='ckpt/exp11/pretrain/x/attempt_1',
                 num_workers=12, log_interval=50, save_every=name['save_every'],
                 epoch_ckpt_every=name['epoch_ckpt_every'],
                 env={'PYTHONHASHSEED': '0', 'XRIR_DATA_PATH': None},
                 train_batches_per_epoch=recipe.TRAIN_BATCHES_PER_EPOCH, tier='M',
                 param_counts=dict(recipe.expected_param_counts(name['backbone'])),
                 exp11_run_type='full', exp11_profile=profile,
                 exp11_registry_sha256='a' * 64, exp11_source_closure_sha256='b' * 64,
                 exp11_git_head='c' * 40,
                 exp11_provenance_path='ckpt/exp11/pretrain/x/attempt_1/provenance.json')
    value.update(overrides)
    return value


def test_the_numerical_recipe_is_exp01s_and_both_profiles_share_it():
    assert dict(recipe.NUMERICAL) == dict(
        num_shot=8, max_len=9600, lr=1e-3, weight_decay=1e-4, decay_epochs=3, lr_gamma=0.1,
        epochs=12, batch_size=32, accum_steps=2, seed=0, tf32=True,
        vit_dim=512, vit_depth=12, vit_heads=8, vit_mlp_dim=512)
    assert recipe.TRAIN_BATCHES_PER_EPOCH == 9261
    for name in ('H_RECIPE', 'I_RECIPE'):
        assert recipe.PROFILES[name]['recipe'] == recipe.NUMERICAL
        assert recipe.PROFILES[name]['backbone'] == 'simple_oriented'
        assert recipe.PROFILES[name]['epoch_ckpt_every'] == 1
    assert recipe.PROFILES['H_RECIPE']['save_every'] == 500
    assert recipe.PROFILES['I_RECIPE']['save_every'] == 0


def test_the_profiles_match_the_historical_records_they_are_taken_from():
    """H = exp_01's recipe; I = exp_04's, including its production constraints."""
    for path, profile in ((EXP01_ARGS, 'H_RECIPE'), (EXP04_ARGS, 'I_RECIPE')):
        if not path.is_file():                        # the checkpoints are gitignored
            pytest.skip('missing historical args: {}'.format(path))
        historical = json.loads(path.read_text())
        for field, value in recipe.PROFILES[profile]['recipe'].items():
            if field in historical:
                assert recipe.strict_equal(historical[field], value), (path, field)
        for field, value in recipe.PROFILES[profile]['production'].items():
            if field in historical:
                assert recipe.strict_equal(historical[field], value), (path, field)
    if EXP04_ARGS.is_file():
        historical = json.loads(EXP04_ARGS.read_text())
        assert historical['save_every'] == recipe.PROFILES['I_RECIPE']['save_every'] == 0
        assert historical['train_batches_per_epoch'] == recipe.TRAIN_BATCHES_PER_EPOCH


@pytest.mark.parametrize('profile', ('H_RECIPE', 'I_RECIPE'))
def test_a_complete_profile_run_has_no_deviations(profile):
    args = args_of(profile)
    assert recipe.classify(args)
    assert recipe.select_profile(args) == profile
    assert recipe.check_all(args, profile) == []
    rows = [{'epoch': e, 'train_loss': 1.0, 'test_loss': 1.0, 'epoch_minutes': 1.0}
            for e in range(1, 13)]
    assert recipe.check_all(args, profile, history_rows=rows,
                            last_meta={'epoch': 12, 'batch_idx': 0}) == []


def test_unknown_fields_are_refused_by_name():
    args = args_of('H_RECIPE', sneaky=1)
    with pytest.raises(ValueError, match='unknown args fields: sneaky'):
        recipe.classify(args)
    assert any('unknown args fields' in item for item in recipe.check_all(args, 'H_RECIPE'))


def test_mixtures_of_the_two_profiles_are_refused():
    """A run may not take H's saving cadence with I's augmentation, or vice versa."""
    mixed = args_of('I_RECIPE', save_every=500)
    assert recipe.select_profile(mixed) == 'I_RECIPE'
    assert any('save_every' in item for item in recipe.check_all(mixed, 'I_RECIPE'))
    mixed = args_of('H_RECIPE', yaw_aug=1)
    assert recipe.select_profile(mixed) == 'I_RECIPE'          # by backbone and yaw_aug
    assert any('yaw_aug' in item for item in recipe.check_all(mixed, 'H_RECIPE'))
    for field, value in (('yaw_aug_seed', 7), ('yaw_aug_width', 256), ('resume', 'last.pth')):
        assert any(field in item
                   for item in recipe.check_all(args_of('I_RECIPE', **{field: value}),
                                                'I_RECIPE'))
    assert recipe.check_all(args_of('H_RECIPE', yaw_aug_seed=7), 'H_RECIPE') == []


def test_type_strictness():
    assert any('epochs' in item for item in recipe.check_all(args_of('H_RECIPE', epochs=12.0),
                                                             'H_RECIPE'))
    assert any('tf32' in item for item in recipe.check_all(args_of('H_RECIPE', tf32=1),
                                                           'H_RECIPE'))
    assert any('epoch_ckpt_every' in item
               for item in recipe.check_all(args_of('H_RECIPE', epoch_ckpt_every=True),
                                            'H_RECIPE'))


def test_the_yaw_counter_constraint_of_exp04():
    assert recipe.check_counter('I_RECIPE', 12, recipe.TRAIN_BATCHES_PER_EPOCH) == []
    assert recipe.check_counter('H_RECIPE', 12, 2 ** 20) == []      # inert without yaw_aug
    assert recipe.check_counter('I_RECIPE', 12, 2 ** 20)


def test_parameter_counts_come_from_the_exp11_factory():
    counts = recipe.expected_param_counts('simple_oriented')
    base = recipe.expected_param_counts('simple')
    assert counts['encoder'] - base['encoder'] == 526336
    assert any('param_counts' in item
               for item in recipe.check_all(args_of('H_RECIPE',
                                                    param_counts=dict(base)), 'H_RECIPE'))


def test_operational_fields_are_validated_against_the_exp11_registry():
    for field, value in (('backbone', 'cylindrical_oriented'), ('backbone', 'nonesuch'),
                         ('save_dir', ''), ('num_workers', -1), ('log_interval', 0),
                         ('env', {'X': 1})):
        assert any(field in item
                   for item in recipe.check_all(args_of('H_RECIPE', **{field: value}),
                                                'H_RECIPE')), (field, value)


def test_presence_of_the_exp11_provenance_class_is_required():
    for field in recipe.EXP11:
        args = args_of('H_RECIPE')
        args.pop(field)
        assert any(field in item for item in recipe.check_all(args, 'H_RECIPE'))


def test_select_profile_refuses_anything_outside_the_two():
    with pytest.raises(ValueError, match='profile'):
        recipe.select_profile(args_of('H_RECIPE', backbone='simple'))
    with pytest.raises(ValueError, match='profile'):
        recipe.select_profile({'backbone': 'simple_oriented', 'yaw_aug': 2})
