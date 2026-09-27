"""exp_11's pretraining entry point: the pinned trainer's steps on the exp_11 registry.

``train_xRIR_backbone`` and ``tools/exp06_train.py`` are frozen; this module owns a
parser over ``BACKBONES_EXP11`` that accepts ``--yaw-aug 0|1`` under exp_04's
constraints, its own ``registry_sha256`` and the ``exp11_*`` provenance class.
"""
import json
import os
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


# --- close review 7 blocker 2: the trainer registers itself ------------------------
# The launcher records `child.pid` only after the frozen lifecycle has already forked
# this process. Between the fork and that write nothing names the trainer, so the
# trainer names itself, first thing, and says when it leaves.


def test_the_trainer_registers_its_pid_after_parsing_before_admission(tmp_path):
    """A launcher that dies from here on still leaves a trace of this process."""
    (tmp_path / 'launching').write_text('launcher 1\n')
    path = exp11_train.register_trainer(str(tmp_path))
    assert path is not None and path.is_file()
    pid, start = path.read_text().split()[:2]
    assert int(pid) == os.getpid() and float(start) > 0


def test_registration_is_skipped_for_a_diagnostic_with_no_run_directory(tmp_path):
    """A --no-save probe has nothing to register into, and must not create one."""
    missing = tmp_path / 'not-there'
    assert exp11_train.register_trainer(str(missing), no_save=True) is None
    assert not missing.exists()


@pytest.mark.parametrize('code', [0, 3, 'exception'])
def test_the_trainer_records_how_it_left(tmp_path, code):
    (tmp_path / 'launching').write_text('launcher 1\n')
    path = exp11_train.register_trainer(str(tmp_path))
    exp11_train.record_trainer_exit(path, code)
    assert (tmp_path / 'train.exit').read_text().strip() == 'train.exit {}'.format(code)


def test_main_registers_first_and_records_an_exception(tmp_path, monkeypatch):
    """The registration surrounds the whole run, including the paths that raise."""
    seen = {}

    def explode(args):
        seen['registered'] = (tmp_path / 'train.pid').is_file()
        raise RuntimeError('admission refused')

    monkeypatch.setattr(exp11_train, 'check_admission', explode)
    (tmp_path / 'launching').write_text('launcher 1\n')
    argv = ['--backbone', 'simple_oriented', '--save-dir', str(tmp_path),
            '--run-type', 'smoke']
    with pytest.raises(RuntimeError):
        exp11_train.main(argv)
    assert seen['registered'], 'the pid file must exist before anything else runs'
    assert (tmp_path / 'train.exit').read_text().strip() == 'train.exit exception'


# --- close review 8 blocker 2: the trainer fails closed after a resolution ---------
# A marker's age proves nothing about a delayed trainer. What decides is the state of
# the attempt directory when the trainer finally looks at it: gone, tombstoned, or
# carrying no launch record at all means the launch it belongs to is over.


def registered(tmp_path, launching=True, child_pid=None):
    """An attempt directory as the launcher leaves it at the fork."""
    attempt = tmp_path / 'attempt_20260927T000000'
    attempt.mkdir()
    if launching:
        (attempt / 'launching').write_text('launcher 1\n')
    if child_pid is not None:
        (attempt / 'child.pid').write_text(child_pid)
    return attempt


def test_the_trainer_refuses_a_save_dir_that_is_gone(tmp_path):
    """It never creates its own: the directory is the launcher's proof of a launch."""
    missing = tmp_path / 'attempt_20260927T000000'
    with pytest.raises(SystemExit) as exit_request:
        exp11_train.register_trainer(str(missing))
    assert exit_request.value.code == 3
    assert not missing.exists(), 'a refusal may not recreate the attempt'


def test_the_trainer_refuses_a_resolved_attempt(tmp_path):
    """The tombstone is permanent: this attempt was answered and retired."""
    attempt = registered(tmp_path)
    (tmp_path / (attempt.name + '.resolved')).write_text('launcher 1\n')
    with pytest.raises(SystemExit) as exit_request:
        exp11_train.register_trainer(str(attempt))
    assert exit_request.value.code == 3
    assert not (attempt / 'train.pid').exists()


def test_the_trainer_refuses_an_attempt_with_no_launch_record(tmp_path):
    """From the fork onward one of the two always exists; neither means neither."""
    attempt = registered(tmp_path, launching=False)
    with pytest.raises(SystemExit) as exit_request:
        exp11_train.register_trainer(str(attempt))
    assert exit_request.value.code == 3
    assert not (attempt / 'train.pid').exists()


@pytest.mark.parametrize('child_pid', ['', 'SIGNAL TERM\n'])
def test_an_incomplete_child_pid_is_not_a_launch_record(tmp_path, child_pid):
    attempt = registered(tmp_path, launching=False, child_pid=child_pid)
    with pytest.raises(SystemExit) as exit_request:
        exp11_train.register_trainer(str(attempt))
    assert exit_request.value.code == 3


@pytest.mark.parametrize('kwargs', [dict(launching=True),
                                    dict(launching=False, child_pid='4321\n')])
def test_the_trainer_registers_against_either_launch_record(tmp_path, kwargs):
    attempt = registered(tmp_path, **kwargs)
    path = exp11_train.register_trainer(str(attempt))
    assert path is not None and path.is_file()


def test_a_no_save_run_writes_no_sidecars_but_still_honours_a_tombstone(tmp_path):
    attempt = registered(tmp_path, launching=False)
    assert exp11_train.register_trainer(str(attempt), no_save=True) is None
    assert not (attempt / 'train.pid').exists()
    (tmp_path / (attempt.name + '.resolved')).write_text('launcher 1\n')
    with pytest.raises(SystemExit) as exit_request:
        exp11_train.register_trainer(str(attempt), no_save=True)
    assert exit_request.value.code == 3


def test_a_clean_exit_with_no_code_is_recorded_as_zero(tmp_path):
    """``raise SystemExit`` carries None, which is exit status 0, not 1."""
    attempt = registered(tmp_path)
    path = exp11_train.register_trainer(str(attempt))
    exp11_train.record_trainer_exit(path, exp11_train.exit_code(SystemExit()))
    assert (attempt / 'train.exit').read_text().strip() == 'train.exit 0'


# --- close review 9 blocker 1: the trainer reads the same grammar --------------------
# The launcher's `pid_record` and this one must agree byte for byte, or "is a trainer
# registered" gets two answers.

PID_RECORDS = [
    ('1457170', True),
    ('1457170\n', True),
    ('\n1457170\n', False),
    ('99999991\n99999992\n', False),
    (' 1457170\n', False),
    ('1457170\r\n', False),
    ('', False),
    ('1457170\n\n', False),
    ('12345678901\n', False),
    ('1234567890\n', True),
]


@pytest.mark.parametrize('content,valid', PID_RECORDS)
def test_the_trainer_reads_one_pid_record_grammar(tmp_path, content, valid):
    pidfile = tmp_path / 'child.pid'
    pidfile.write_text(content)
    record = exp11_train.pid_record(pidfile)
    assert (record is not None) is valid, (content, record)
    if valid:
        assert record == int(content.strip())


@pytest.mark.parametrize('content,valid', PID_RECORDS)
def test_registration_completeness_follows_that_grammar(tmp_path, content, valid):
    (tmp_path / 'child.pid').write_text(content)
    assert exp11_train.registration_complete(tmp_path) is valid


def test_a_missing_pid_file_is_simply_not_a_record(tmp_path):
    assert exp11_train.pid_record(tmp_path / 'nothing') is None
    assert exp11_train.registration_complete(tmp_path) is False
