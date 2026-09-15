"""exp_06 training entry point: the trainer's flags, model and recorded arguments."""
import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import pytest
import torch

import train_xRIR_backbone as trainer
from tools import exp06_recipe, exp06_train
from tools import provenance

REPO = Path(__file__).resolve().parents[1]

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
    assert set(mine) - set(theirs) == {'run_type', 'provenance_out', 'approved',
                                       'reviewed_commit', 'exploratory'}
    for dest, action in theirs.items():
        if dest in ('help', 'backbone', 'yaw_aug'):
            continue
        assert action.option_strings == mine[dest].option_strings, dest
        assert action.default == mine[dest].default and type(action.default) is type(mine[dest].default), dest
        assert action.type is mine[dest].type and action.choices == mine[dest].choices, dest
        assert action.required == mine[dest].required and action.nargs == mine[dest].nargs, dest
        assert type(action) is type(mine[dest]), dest
    args = exp06_train.parse_args(MINIMAL)
    extra = ('run_type', 'provenance_out', 'approved', 'reviewed_commit', 'exploratory')
    assert {k: v for k, v in vars(args).items() if k not in extra} == vars(reference_args)
    assert (args.run_type, args.provenance_out) == ('full', None)
    assert (args.approved, args.reviewed_commit, args.exploratory) == (None, None, False)


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


def test_registry_digest_covers_every_registered_backbone():
    from model.xRIR_cyl_oriented import BACKBONES_EXP06
    mapping = {name: cls.__module__ + '.' + cls.__qualname__ for name, cls in BACKBONES_EXP06.items()}
    assert set(mapping) == {'simple', 'cylindrical', 'cylindrical_oriented'}
    expected = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    assert exp06_train.registry_sha256() == expected and len(expected) == 64


def test_provenance_fields_bind_code_data_and_argv():
    identity = {'data_root': '/data', 'inventory': [], 'inventory_sha256': 'd' * 64}
    fields = exp06_train.provenance_fields(RECIPE, 'full', identity=identity)
    assert fields['run_type'] == 'full' and fields['command'] == RECIPE
    assert fields['train_data_identity'] is identity and fields['repo'] == str(REPO)
    assert fields['registry_sha256'] == exp06_train.registry_sha256()
    assert fields['reviewed_commit'] == fields['git_state']['HEAD']
    assert fields['environment']['python'].startswith('3.8')
    closure = fields['source_closures']['training']
    paths = [record['path'] for record in closure['files']]
    assert {'train_xRIR_backbone.py', 'tools/exp06_train.py', 'tools/exp06_recipe.py',
            'model/xRIR_cyl_oriented.py', 'model/cylindrical_vit_oriented.py'} <= set(paths)
    assert closure['sha256'] == provenance.closure_record(paths, fields['git_state']['HEAD'], REPO)[1]
    without_data = {key: value for key, value in fields.items() if key != 'train_data_identity'}
    assert provenance.revalidate(without_data, required=('source_closures', 'run_type')) == []


def test_provenance_destination_follows_the_save_mode(tmp_path):
    saving = exp06_train.parse_args(MINIMAL)
    assert exp06_train.provenance_destination(saving) == os.path.join(saving.save_dir, 'provenance.json')
    assert exp06_train.provenance_destination(exp06_train.parse_args(MINIMAL + ['--no-save'])) is None
    out = str(tmp_path / 'p.json')
    quiet = exp06_train.parse_args(MINIMAL + ['--no-save', '--provenance-out', out])
    assert exp06_train.provenance_destination(quiet) == out


def test_data_identity_never_creates_the_shared_cache(tmp_path, monkeypatch):
    calls = {}

    def fake(data_root, cache_path=None, workers=8):
        calls['cache'] = str(cache_path)
        return {'inventory': [], 'data_root': data_root}

    monkeypatch.setattr(exp06_train.provenance, 'train_data_identity', fake)
    absent = tmp_path / 'absent.json'
    assert exp06_train.data_identity('/data', absent)['data_root'] == '/data'
    assert not absent.exists() and Path(calls['cache']).parent != tmp_path
    present = tmp_path / 'present.json'
    present.write_text('{}')
    exp06_train.data_identity('/data', present)
    assert calls['cache'] == str(present)


def test_main_composes_the_pinned_trainer_and_never_completes_a_run():
    source = inspect.getsource(exp06_train.main)
    for call in ('trainer.seed_everything(', 'trainer.seed_worker', 'trainer.train_epoch(',
                 'trainer.test_epoch(', 'trainer.save_checkpoint(', 'write_manifest('):
        assert call in source
    module = inspect.getsource(exp06_train)
    assert 'write_completion' not in module
    assert "'completion.json'" not in module and '"completion.json"' not in module
    assert 'XRIR_RUNTIME_ARGS' in source and 'history.jsonl' in source


def test_provenance_records_the_closure_entry_module():
    """Blocker 1/3: the finalizer recomputes the closure of the module named here."""
    fields = exp06_train.provenance_fields(RECIPE, 'full')
    assert fields['source_closures']['training']['entry_module'] == 'tools.exp06_train'


def test_the_data_root_is_the_dataset_modules_own_resolution():
    """Should-fix 6: the trainer and the provenance helper had different fallbacks."""
    from treble_multi_room_dataset import treble_xRIR_dataset as dataset
    assert exp06_train.DATA_ROOT == dataset.BASE_DATA_PATH
    assert exp06_train.resolve_data_root() == os.path.realpath(dataset.BASE_DATA_PATH)


def test_an_absent_data_root_is_refused(monkeypatch):
    monkeypatch.setattr(exp06_train, 'DATA_ROOT', '/nonexistent/acoustic/rooms')
    with pytest.raises(ValueError, match='data root'):
        exp06_train.resolve_data_root()


def test_provenance_records_the_resolved_data_root():
    fields = exp06_train.provenance_fields(RECIPE, 'full')
    assert fields['data_root'] == exp06_train.resolve_data_root()


@pytest.fixture(scope='module')
def approvals(tmp_path_factory):
    """A filled approvals file for this checkout, as the second reviewed commit will be."""
    from tools import exp06_profiles
    head = provenance.git_state(REPO)['HEAD']
    value = exp06_profiles.json_value(exp06_profiles.load_approved_digests()[0])
    value['code'].update(exp06_profiles.compute_code_digests(
        REPO, head, keys=exp06_profiles.TRAINING_KEYS))
    path = tmp_path_factory.mktemp('approvals') / 'approved_digests.json'
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    return path, head


def test_the_approvals_and_reviewed_commit_reach_the_parser():
    """Finding 1: the launcher hands the child what its preflight verified."""
    args = exp06_train.parse_args(RECIPE + ['--approved', 'a.json', '--reviewed-commit', 'f' * 40])
    assert args.approved == 'a.json' and args.reviewed_commit == 'f' * 40
    assert args.exploratory is False
    assert exp06_train.parse_args(MINIMAL + ['--run-type', 'probe', '--exploratory']).exploratory
    with pytest.raises(SystemExit) as exit:  # an exploratory run is never confirmatory
        exp06_train.parse_args(RECIPE + ['--exploratory'])
    assert exit.value.code == 2


def test_provenance_binds_the_orchestration_and_the_approvals(approvals):
    """Finding 1: the launcher and the finalizer bytes are recorded at spawn."""
    from tools import exp06_profiles
    path, head = approvals
    fields = exp06_train.provenance_fields(RECIPE, 'full', approved=str(path),
                                           reviewed_commit=head)
    assert fields['reviewed_commit'] == head
    orchestration = fields['orchestration_closures']
    assert set(orchestration) == {'launcher', 'finalizer'}
    assert [record['path'] for record in orchestration['launcher']['files']] == \
        ['tools/exp06_launch.sh']
    assert orchestration['finalizer']['entry_module'] == 'tools.exp06_finalize'
    assert 'tools/exp06_finalize.py' in [r['path'] for r in orchestration['finalizer']['files']]
    for role in orchestration:
        for record in orchestration[role]['files']:
            assert len(record['working_tree_sha256']) == 64
            assert len(record['reviewed_blob_sha256']) == 64
    assert fields['approvals'] == {'path': str(path), 'schema_version': 1,
                                   'sha256': provenance.sha256_file(path)}
    assert fields['exploratory'] is False
    digests = fields['code_digests']
    assert set(digests) == set(exp06_profiles.TRAINING_KEYS)
    assert digests['launch_sh'] == orchestration['launcher']['sha256']
    assert digests['finalize'] == orchestration['finalizer']['sha256']
    assert digests['trainer'] == fields['source_closures']['training']['sha256']


def test_a_full_run_without_approvals_never_starts(monkeypatch, tmp_path):
    """Finding 1: a confirmatory run is refused at spawn, not 31 hours later."""
    monkeypatch.setattr(sys, 'argv', ['tools/exp06_train.py'] + RECIPE)
    with pytest.raises(ValueError, match='approv'):
        exp06_train.main(RECIPE + ['--save-dir', str(tmp_path / 'attempt')])


def test_prepare_args_drops_every_admission_flag():
    args = exp06_train.parse_args(RECIPE + ['--approved', 'a.json', '--reviewed-commit', 'f' * 40])
    args.train_batches_per_epoch = exp06_recipe.TRAIN_BATCHES_PER_EPOCH
    args.env = {'PYTHONHASHSEED': '0', 'XRIR_DATA_PATH': '/data',
                'OMP_NUM_THREADS': '8', 'CUDA_VISIBLE_DEVICES': '1'}
    exp06_train.prepare_args(args, None, {'source_closures': {'training': {'sha256': 'a' * 64}},
                                          'git_state': {'HEAD': 'b' * 40},
                                          'registry_sha256': 'c' * 64}, 'p.json')
    for dropped in ('approved', 'reviewed_commit', 'exploratory'):
        assert not hasattr(args, dropped)
    assert exp06_recipe.check_all(vars(args)) == []
