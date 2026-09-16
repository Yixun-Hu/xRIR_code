"""exp_07 seen launcher: arms, goldens, per-arm controls and the runtime schema.

Every check is CPU-only: no trainer is spawned and no attempt directory is created
outside ``tmp_path``.  The historical comparators are the REAL ``ckpt/...args.json``
files of exp_01 and exp_04 (read-only), so a recipe drift would be caught here.
"""
import json
import subprocess
from pathlib import Path

import pytest

from tools import exp04_launcher as base
from tools import exp07_launcher as launch
from tools import exp07_provenance as e7p

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ('--num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 '
          '--lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 '
          '--seed 0 --tf32 --log-interval 50 --save-every 0 --epoch-ckpt-every 1').split()
YAW = ['--yaw-aug', '1', '--yaw-aug-seed', '0', '--yaw-aug-width', '512']
ARMS = {'seen_simple': ('simple', 0), 'seen_cyl': ('cylindrical', 0), 'seen_aug': ('simple', 1)}
SEEN_BATCHES = 9265
SMOKE = base.command('smoke', '<x>')[6:-6]  # the exp_04 smoke recipe, arm/protocol flags aside


def attempt_of(name, mode='full'):
    return 'ckpt/exp07/{}/{}test'.format(name, 'attempt_' if mode == 'full' else '_smoke_')


def argv_of(name, attempt=None, mode='full'):
    backbone, yaw = ARMS[name]
    return launch.command(mode, attempt or attempt_of(name, mode), backbone, yaw)


@pytest.mark.parametrize('mode', ['full', 'smoke'])
@pytest.mark.parametrize('name', sorted(ARMS))
def test_golden_argv_equals_the_plan_recipe(name, mode):
    backbone, yaw = ARMS[name]
    attempt = attempt_of(name, mode)
    recipe = RECIPE if mode == 'full' else SMOKE
    expected = ([launch.PYTHON, 'tools/exp07_train.py', '--backbone', backbone, '--save-dir',
                 attempt + ('' if mode == 'full' else '/smoke')] + recipe +
                ['--protocol', 'seen'] + (YAW if yaw else []))
    argv = argv_of(name, attempt, mode)
    assert argv == expected
    launch.check_golden(argv, mode, attempt)
    golden = launch.goldens() / ('argv_golden_{}{}.txt'.format(name, '' if mode == 'full' else '_smoke'))
    placeholder = attempt.replace('test', '<ts>')
    assert golden.read_text().split() == [part.replace(attempt, placeholder) for part in expected]


@pytest.mark.parametrize('flag,value', [('--lr', '0.002'), ('--epochs', '11'),
                                        ('--protocol', 'unseen'), ('--yaw-aug-width', '256')])
def test_one_flag_deviation_is_refused(flag, value):
    attempt = attempt_of('seen_aug')
    argv = argv_of('seen_aug', attempt)
    launch.check_golden(argv, 'full', attempt)
    argv[argv.index(flag) + 1] = value
    with pytest.raises(ValueError, match='golden'):
        launch.check_golden(argv, 'full', attempt)


@pytest.mark.parametrize('name,attempt', [
    ('seen_simple', 'ckpt/xRIR_simple_yawaug_8_shot/attempt_test'),
    ('seen_aug', 'ckpt/xRIR_simple_yawaug_8_shot/attempt_test'),
    ('seen_simple', 'ckpt/exp05/S_simple/attempt_test'),
    ('seen_simple', 'ckpt/exp07/seen_cyl/attempt_test'),
    ('seen_aug', 'ckpt/exp07/seen_simple/attempt_test'),
    ('seen_simple', 'ckpt/exp07/seen_simple/_smoke_test'),
    ('seen_simple', 'ckpt/exp07/seen_simple/attempt_../other')])
def test_seen_argv_refused_outside_its_own_arm(name, attempt):
    with pytest.raises(ValueError, match='golden'):
        launch.check_golden(argv_of(name, attempt), 'full', attempt)


@pytest.mark.parametrize('tier,backbone', [('M', 'simple'), ('S', 'simple'), ('L', 'cylindrical')])
def test_unseen_argv_refused_under_a_seen_root(tier, backbone):
    attempt = attempt_of('seen_simple')
    with pytest.raises(ValueError, match='golden'):
        launch.check_golden(base.command('full', attempt, tier, backbone), 'full', attempt)


@pytest.mark.parametrize('backbone,yaw', [('cylindrical', 1), ('simple', 2), ('invented', 0)])
def test_command_refuses_unregistered_seen_arms(backbone, yaw):
    with pytest.raises(ValueError, match='seen arm'):
        launch.command('full', attempt_of('seen_simple'), backbone, yaw)
    with pytest.raises(ValueError, match='seen arm'):
        launch.arm_root(backbone, yaw)


def test_full_requires_the_seen_batch_count():
    args = launch.effective_args(argv_of('seen_simple'), '1', SEEN_BATCHES)
    launch.check_runtime(args, args, 'full')
    wrong = launch.effective_args(argv_of('seen_simple'), '1', 9261)
    with pytest.raises(ValueError, match='9265'):
        launch.check_runtime(wrong, wrong, 'full')
    unseen = dict(args, protocol='unseen')
    with pytest.raises(ValueError, match='protocol'):
        launch.check_runtime(unseen, unseen, 'full')
    with pytest.raises(base.LauncherFailure, match='runtime args mismatch'):
        launch.check_runtime(dict(args, lr=0.5), args, 'full')


def test_effective_args_is_the_recorded_schema():
    args = launch.effective_args(argv_of('seen_aug'), '1', SEEN_BATCHES)
    assert (args['protocol'], args['tier'], args['yaw_aug']) == ('seen', 'M', 1)
    assert args['CUDA_VISIBLE_DEVICES'] == '1' and args['PYTHONHASHSEED'] == '0'
    assert args['train_batches_per_epoch'] == SEEN_BATCHES and args['param_counts']['full'] > 0


def test_arm_roots_and_ledger_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    roots = [launch.arm_root(backbone, yaw) for backbone, yaw in ARMS.values()]
    assert roots == [tmp_path / 'ckpt/exp07' / name for name in ARMS]
    assert base.arm_root('M', 'simple') == base.ROOT  # the exp_04 root is unchanged
    base.account_hours(base.create_attempt(roots[0], 'attempt_test'), 2, mode='full')
    assert [base.full_hours(root) for root in roots] == [2, 0, 0]


def control_of(name):
    control = json.loads((ROOT / launch.CONTROLS[name]).read_text())
    return control, control.get('env') or dict(XRIR_DATA_PATH=launch.DATA_ROOT,
                                               OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='1')


@pytest.mark.parametrize('name', sorted(ARMS))
def test_control_parity_against_the_unseen_twin(name):
    runtime = launch.effective_args(argv_of(name), '1', SEEN_BATCHES)
    assert runtime['yaw_aug'] == ARMS[name][1] and runtime['backbone'] == ARMS[name][0]
    control, env = control_of(name)
    differences = launch.compare_control(runtime, control, env)
    assert differences['protocol'] == {'treatment': 'seen', 'control': 'unseen'}
    assert set(differences) <= launch.CONTROL_EXCLUSIONS | {'protocol'}
    launch.compare_control(runtime, dict(control, yaw_aug_seed=None), env)  # exp_05 A1 default
    for other in sorted(set(ARMS) - {name}):  # only the arm's own unseen twin is admissible
        with pytest.raises(ValueError, match='control mismatch'):
            launch.compare_control(runtime, control_of(other)[0], env)
    # a seen-protocol comparator is never an unseen twin (seen_aug's recorded 9261 refuses first)
    with pytest.raises(ValueError, match='twin|train_batches_per_epoch'):
        launch.compare_control(runtime, dict(control, protocol='seen'), env)


@pytest.mark.parametrize('key,value', [('lr', 0.002), ('num_workers', '12'), ('yaw_aug', 1),
                                       ('no_save', True), ('train_batches_per_epoch', 9261)])
def test_control_parity_refuses_unregistered_changes(key, value):
    runtime = launch.effective_args(argv_of('seen_simple'), '1', SEEN_BATCHES)
    runtime[key] = value
    with pytest.raises(ValueError, match=key if key != 'train_batches_per_epoch' else 'invalid'):
        launch.compare_control(runtime, *control_of('seen_simple'))


def test_an_unseen_treatment_is_never_compared():
    runtime = launch.effective_args(argv_of('seen_simple'), '1', SEEN_BATCHES)
    with pytest.raises(ValueError, match='seen treatment'):
        launch.compare_control(dict(runtime, protocol='unseen', train_batches_per_epoch=9261),
                               *control_of('seen_simple'))


def test_the_training_minimum_adds_the_seen_module_and_the_entry_point():
    assert launch.train_minimum() == base.TRAIN_MINIMUM | {launch.SEEN_MODULE, launch.ENTRY}
    assert launch.SEEN_MODULE not in base.TRAIN_MINIMUM
    assert e7p.SEEN_SPLIT and launch.SEEN_INPUTS == {'seen_split'}


def test_the_shared_launcher_keeps_mains_bytes():
    for name in ('tools/exp04_launcher.py', 'tools/exp04_launch.sh',
                 'tools/exp05_probe.py', 'tools/exp05_gates.py'):
        assert subprocess.run(['git', 'diff', '--quiet', 'main', '--', name],
                              cwd=str(ROOT)).returncode == 0, name
    assert not hasattr(base, 'SEEN_ARMS') and not hasattr(base, 'TRAIN_FILES')
