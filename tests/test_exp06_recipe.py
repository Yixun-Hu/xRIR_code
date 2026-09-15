"""Type-strict classification of the trainer's args.json and budget completeness."""
import json
from pathlib import Path

import pytest

from tools.exp05_params import TIERS
from tools.exp06_recipe import (CLASSES, EXP01_RECIPE, check_production, check_recipe,
                                classify, normalize_historical)

REPO = Path(__file__).resolve().parents[1]
TRAINER_ARGS = REPO / 'ckpt/exp05/S_simple/attempt_20260913T123905/args.json'
EXP01_ARGS = ['ckpt/xRIR_simple_8_shot/args.json', 'ckpt/xRIR_cyl_8_shot/args.json']


def full_args(backbone='cylindrical_oriented', **overrides):
    args = dict(EXP01_RECIPE, backbone=backbone, save_dir='ckpt/exp06/pretrain/attempt_x',
                num_workers=12, log_interval=50, save_every=500, epoch_ckpt_every=1,
                env={'PYTHONHASHSEED': '0', 'XRIR_DATA_PATH': '/data',
                     'OMP_NUM_THREADS': '8', 'CUDA_VISIBLE_DEVICES': '1'},
                max_train_batches=0, max_test_batches=0, test_subset=0, resume=None,
                no_save=False, yaw_aug=0, yaw_aug_seed=0, yaw_aug_width=512,
                train_batches_per_epoch=9261, tier='M', param_counts={'full': 1})
    args.update(overrides)
    return args


def test_classes_are_disjoint_and_cover_the_trainer():
    fields = [field for group in CLASSES.values() for field in group]
    assert len(fields) == len(set(fields))
    assert len([f for name, group in CLASSES.items() if name != 'exp06' for f in group]) == 33
    assert set(CLASSES) == {'recipe', 'production', 'operational', 'derived', 'exp06'}
    assert set(CLASSES['recipe']) == set(EXP01_RECIPE)


def test_every_field_of_the_real_trainer_args_is_classified():
    if not TRAINER_ARGS.is_file():
        pytest.skip('exp_05 attempt args.json absent')
    args = json.loads(TRAINER_ARGS.read_text())
    assert len(args) == 33
    mapping = classify(args)
    assert set(mapping) == set(args)
    assert set(mapping.values()) == {'recipe', 'production', 'operational', 'derived'}
    assert mapping['tf32'] == 'recipe' and mapping['env'] == 'operational'
    assert mapping['yaw_aug_width'] == 'production' and mapping['param_counts'] == 'derived'


def test_exp06_provenance_fields_are_classified():
    args = full_args(exp06_run_type='full', exp06_registry_sha256='a' * 64,
                     exp06_source_closure_sha256='b' * 64, exp06_git_head='c' * 40,
                     exp06_provenance_path='ckpt/exp06/attempt/provenance.json')
    mapping = classify(args)
    assert {mapping[key] for key in args if key.startswith('exp06_')} == {'exp06'}


def test_unknown_field_is_named():
    with pytest.raises(ValueError, match='mystery'):
        classify(full_args(mystery=1))


@pytest.mark.parametrize('relative', EXP01_ARGS)
def test_historical_exp01_args_pass_the_recipe_after_normalization(relative):
    path = REPO / relative
    if not path.is_file():
        pytest.skip('exp_01 args.json absent')
    args = json.loads(path.read_text())
    assert check_recipe(args), 'historical files lack vit_* and must not pass unnormalized'
    normalized, report = normalize_historical(args)
    assert check_recipe(normalized) == []
    assert set(report['normalized']) == {'vit_dim', 'vit_depth', 'vit_heads', 'vit_mlp_dim',
                                         'yaw_aug', 'yaw_aug_seed', 'yaw_aug_width'}
    assert report['normalized']['vit_dim'] == TIERS['M']['dim']
    assert {'train_batches_per_epoch', 'tier', 'param_counts', 'env', 'no_save'} <= set(report['not_recorded'])
    assert all(field not in normalized for field in report['not_recorded'])
    assert classify(normalized)
    assert args == json.loads(path.read_text()), 'normalization must not mutate its input'


def test_registered_recipe_and_production_pass():
    args = full_args()
    assert check_recipe(args) == [] and check_production(args) == []


@pytest.mark.parametrize('overrides,named', [
    ({'resume': 'ckpt/last.pth'}, 'resume'),
    ({'no_save': True}, 'no_save'),
    ({'test_subset': 100}, 'test_subset'),
    ({'max_test_batches': 2}, 'max_test_batches'),
    ({'yaw_aug': 1}, 'yaw_aug'),
    ({'yaw_aug_seed': None}, 'yaw_aug_seed'),
    ({'yaw_aug_width': True}, 'yaw_aug_width'),
    ({'tf32': 1}, 'tf32'),
    ({'tf32': False}, 'tf32'),
    ({'seed': True}, 'seed'),
    ({'lr': 2e-3}, 'lr'),
    ({'lr': 1}, 'lr'),
    ({'epochs': 12.0}, 'epochs'),
    ({'batch_size': 64}, 'batch_size'),
    ({'vit_depth': 6}, 'vit_depth'),
    ({'num_shot': 4}, 'num_shot'),
])
def test_recipe_and_production_deviations_are_named(overrides, named):
    args = full_args(**overrides)
    deviations = check_recipe(args) + check_production(args)
    assert deviations and all(named in deviation for deviation in deviations)


@pytest.mark.parametrize('field', sorted(EXP01_RECIPE) + ['resume', 'no_save', 'yaw_aug'])
def test_missing_recipe_and_production_fields_are_refused(field):
    args = full_args()
    del args[field]
    deviations = check_recipe(args) + check_production(args)
    assert deviations and all(field in deviation for deviation in deviations)
