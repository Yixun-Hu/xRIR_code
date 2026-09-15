"""exp_06 training entry point: the trainer's flags, model and recorded arguments."""
import argparse
import sys

import pytest
import torch

import train_xRIR_backbone as trainer
from tools import exp06_recipe, exp06_train

MINIMAL = ['--backbone', 'simple', '--save-dir', 'ckpt/exp06/_smoke/t0']

RECIPE = ['--backbone', 'cylindrical_oriented', '--save-dir', 'ckpt/exp06/pretrain/attempt_x',
          '--num-shot', '8', '--max-len', '9600', '--lr', '1e-3', '--weight-decay', '1e-4',
          '--decay-epochs', '3', '--lr-gamma', '0.1', '--epochs', '12', '--batch-size', '32',
          '--accum-steps', '2', '--num-workers', '12', '--seed', '0', '--tf32',
          '--log-interval', '50', '--save-every', '500', '--epoch-ckpt-every', '1',
          '--run-type', 'full']


@pytest.fixture
def trainer_parser(monkeypatch):
    """The trainer builds its parser inside parse_args; capture it without changing the file."""
    captured = {}
    original = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        captured['parser'] = self
        return original(self, args, namespace)

    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', capture)
    monkeypatch.setattr(sys, 'argv', ['train_xRIR_backbone.py'] + MINIMAL)
    namespace = trainer.parse_args()
    return captured['parser'], namespace


def test_shared_flags_and_defaults_equal_the_trainers(trainer_parser):
    reference, reference_args = trainer_parser
    parser = exp06_train.build_parser()
    mine = {action.dest: action for action in parser._actions}
    theirs = {action.dest: action for action in reference._actions}
    assert set(theirs) - set(mine) == set()
    assert set(mine) - set(theirs) == {'run_type', 'provenance_out'}
    for dest, action in theirs.items():
        if dest in ('help', 'backbone', 'yaw_aug'):
            continue
        assert action.option_strings == mine[dest].option_strings, dest
        assert action.default == mine[dest].default and type(action.default) is type(mine[dest].default), dest
        assert action.type is mine[dest].type and action.choices == mine[dest].choices, dest
        assert action.required == mine[dest].required and action.nargs == mine[dest].nargs, dest
        assert type(action) is type(mine[dest]), dest
    args = exp06_train.parse_args(MINIMAL)
    assert {k: v for k, v in vars(args).items() if k not in ('run_type', 'provenance_out')} == vars(reference_args)
    assert (args.run_type, args.provenance_out) == ('full', None)


def test_backbone_registry_and_yaw_augmentation_are_restricted():
    from model.xRIR_cyl_oriented import BACKBONES_EXP06
    backbone = {action.dest: action for action in exp06_train.build_parser()._actions}['backbone']
    assert backbone.choices == sorted(BACKBONES_EXP06) and backbone.required
    assert exp06_train.parse_args(RECIPE).backbone == 'cylindrical_oriented'
    for argv in (MINIMAL + ['--yaw-aug', '1'], MINIMAL[:1] + ['invented'] + MINIMAL[2:],
                 ['--save-dir', 'd'], MINIMAL + ['--run-type', 'invented']):
        with pytest.raises(SystemExit) as exit:
            exp06_train.parse_args(argv)
        assert exit.value.code == 2
    assert exp06_train.parse_args(MINIMAL + ['--yaw-aug', '0']).yaw_aug == 0
    assert exp06_train.parse_args(MINIMAL + ['--seed', '7']).yaw_aug_seed == 7
    assert exp06_train.parse_args(MINIMAL + ['--no-save', '--provenance-out', 'p.json']).provenance_out == 'p.json'
    with pytest.raises(SystemExit):
        exp06_train.parse_args(MINIMAL + ['--provenance-out', 'p.json'])


def test_simple_backbone_state_is_bit_identical_to_the_trainers():
    args = exp06_train.parse_args(MINIMAL)
    torch.manual_seed(0)
    mine = exp06_train.build_model_exp06(args)
    torch.manual_seed(0)
    theirs = trainer.build_model(argparse.Namespace(**vars(args)))
    assert type(mine) is type(theirs)
    assert mine.state_dict().keys() == theirs.state_dict().keys()
    assert all(torch.equal(value, theirs.state_dict()[key]) for key, value in mine.state_dict().items())


def test_prepare_args_records_the_derived_and_provenance_fields():
    args = exp06_train.parse_args(RECIPE)
    args.train_batches_per_epoch = exp06_recipe.TRAIN_BATCHES_PER_EPOCH
    args.env = {'PYTHONHASHSEED': '0', 'XRIR_DATA_PATH': '/data',
                'OMP_NUM_THREADS': '8', 'CUDA_VISIBLE_DEVICES': '1'}
    fields = {'source_closures': {'training': {'sha256': 'a' * 64}},
              'git_state': {'HEAD': 'b' * 40}, 'registry_sha256': 'c' * 64}
    exp06_train.prepare_args(args, None, fields, 'ckpt/exp06/attempt_x/provenance.json')
    recorded = vars(args)
    assert not hasattr(args, 'run_type') and not hasattr(args, 'provenance_out')
    assert exp06_recipe.classify(recorded)
    assert exp06_recipe.check_all(recorded) == []
    assert recorded['tier'] == 'M' and recorded['param_counts']['full'] > 0
    assert recorded['exp06_run_type'] == 'full' and recorded['exp06_git_head'] == 'b' * 40
    assert recorded['exp06_source_closure_sha256'] == 'a' * 64
    assert recorded['exp06_registry_sha256'] == 'c' * 64
    assert recorded['exp06_provenance_path'] == 'ckpt/exp06/attempt_x/provenance.json'
    changed = exp06_train.parse_args(RECIPE + ['--lr', '2e-3'])
    changed.train_batches_per_epoch, changed.env = exp06_recipe.TRAIN_BATCHES_PER_EPOCH, recorded['env']
    exp06_train.prepare_args(changed, None, fields, None)
    assert any('lr' in deviation for deviation in exp06_recipe.check_all(vars(changed)))
    assert vars(changed)['exp06_provenance_path'] is None


def test_prepare_args_refuses_a_capacity_outside_the_recipe_tier():
    args = exp06_train.parse_args(RECIPE + ['--vit-depth', '6'])
    with pytest.raises(ValueError, match='tier'):
        exp06_train.prepare_args(args, None, {}, None)
