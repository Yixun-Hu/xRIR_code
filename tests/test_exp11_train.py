"""exp_11's pretraining entry point: the pinned trainer's steps on the exp_11 registry.

``train_xRIR_backbone`` and ``tools/exp06_train.py`` are frozen; this module owns a
parser over ``BACKBONES_EXP11`` that accepts ``--yaw-aug 0|1`` under exp_04's
constraints, its own ``registry_sha256`` and the ``exp11_*`` provenance class.
"""
import json
from pathlib import Path
import subprocess

import pytest
import torch

from model.xRIR_simple_oriented import xRIR_SimpleOriented
from model.xrir_exp11_registry import registry_sha256
from tools import exp06_train, exp11_recipe, exp11_train

REPO = Path(__file__).resolve().parents[1]
BASE = ['--backbone', 'simple_oriented', '--save-dir', 'ckpt/exp11/pretrain/x/attempt_1',
        '--epochs', '12', '--batch-size', '32', '--accum-steps', '2', '--lr', '1e-3',
        '--decay-epochs', '3', '--seed', '0', '--tf32', '--epoch-ckpt-every', '1',
        '--num-workers', '12', '--log-interval', '50']


def args_for(*extra):
    return exp11_train.parse_args(BASE + list(extra))


def test_the_parser_ranges_over_the_exp11_registry():
    assert exp11_train.parse_args(BASE + ['--save-every', '500']).backbone == 'simple_oriented'
    for backbone in ('simple', 'cylindrical_oriented', 'simple_adapter'):
        assert exp11_train.parse_args(
            ['--backbone', backbone, '--save-dir', 'd']).backbone == backbone
    with pytest.raises(SystemExit):
        exp11_train.parse_args(['--backbone', 'nonesuch', '--save-dir', 'd'])


def test_yaw_augmentation_carries_exp04s_constraints():
    args = args_for('--yaw-aug', '1', '--save-every', '0')
    assert args.yaw_aug == 1 and args.yaw_aug_seed == 0 and args.yaw_aug_width == 512
    for extra in (('--yaw-aug', '1', '--save-every', '500'),
                  ('--yaw-aug', '1', '--save-every', '0', '--resume', 'last.pth')):
        with pytest.raises(SystemExit):
            args_for(*extra)
    with pytest.raises(SystemExit):
        args_for('--yaw-aug', '2')
    assert args_for('--save-every', '500').yaw_aug == 0


def test_the_registry_digest_is_exp11s_own():
    assert exp11_train.registry_sha256() == registry_sha256()
    assert exp11_train.registry_sha256() != exp06_train.registry_sha256()


def test_admission_requires_the_approvals_for_a_confirmatory_run():
    with pytest.raises(ValueError, match='--approved'):
        exp11_train.check_admission(args_for('--save-every', '500'))
    smoke = args_for('--save-every', '500', '--run-type', 'smoke', '--no-save')
    assert exp11_train.check_admission(smoke) is smoke
    with pytest.raises(ValueError, match='--approved'):
        exp11_train.check_admission(args_for('--save-every', '500', '--run-type', 'smoke'))
    with pytest.raises(SystemExit):
        args_for('--run-type', 'full', '--exploratory')


def test_prepare_args_records_the_exp11_class_and_the_selected_profile():
    args = args_for('--save-every', '500')
    fields = {'registry_sha256': 'r' * 64, 'git_state': {'HEAD': 'h' * 40},
              'source_closures': {'training': {'sha256': 's' * 64}}}
    args.train_batches_per_epoch = exp11_recipe.TRAIN_BATCHES_PER_EPOCH
    args.env = {'PYTHONHASHSEED': '0'}
    with torch.random.fork_rng(devices=[]):
        exp11_train.prepare_args(args, None, fields, 'ckpt/exp11/x/provenance.json')
    recorded = vars(args)
    assert recorded['exp11_profile'] == 'H_RECIPE'
    assert recorded['exp11_registry_sha256'] == 'r' * 64
    assert recorded['exp11_source_closure_sha256'] == 's' * 64
    assert recorded['exp11_git_head'] == 'h' * 40
    assert recorded['exp11_provenance_path'] == 'ckpt/exp11/x/provenance.json'
    assert recorded['tier'] == 'M' and recorded['param_counts']['encoder'] > 0
    for flag in ('run_type', 'provenance_out', 'approved', 'reviewed_commit', 'exploratory'):
        assert flag not in recorded
    assert exp11_recipe.check_all(recorded) == []
    json.dumps(recorded, sort_keys=True, allow_nan=False)


def test_prepare_args_refuses_a_capacity_outside_the_recipe_tier():
    args = args_for('--save-every', '500', '--vit-dim', '256', '--vit-depth', '6',
                    '--vit-heads', '4', '--vit-mlp-dim', '256')
    args.train_batches_per_epoch, args.env = exp11_recipe.TRAIN_BATCHES_PER_EPOCH, {}
    with pytest.raises(ValueError, match='tier M'):
        exp11_train.prepare_args(args, None, {}, 'p')


def test_the_yaw_counter_bound_is_enforced_before_training():
    args = args_for('--yaw-aug', '1', '--save-every', '0')
    with pytest.raises(ValueError, match='2\\*\\*20'):
        exp11_train.check_counter(args, 2 ** 20)
    assert exp11_train.check_counter(args, exp11_recipe.TRAIN_BATCHES_PER_EPOCH) is None


def test_build_model_uses_the_exp11_factory():
    with torch.random.fork_rng(devices=[]):
        model = exp11_train.build_model_exp11(args_for('--vit-dim', '512'))
    assert type(model) is xRIR_SimpleOriented


def test_the_entry_module_imports_hermetically_and_its_closure_is_bounded():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO),
                                   text=True).strip()
    from tools import provenance
    files = provenance.source_closure(exp11_train.ENTRY_MODULE, REPO)
    assert 'tools/exp11_train.py' in files and 'tools/exp11_recipe.py' in files
    assert 'model/xrir_exp11_registry.py' in files
    assert 'model/simple_vit_oriented.py' in files
    records, digest = provenance.closure_record(list(files), head, REPO)
    assert len(digest) == 64 and len(records) == len(files)
