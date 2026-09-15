"""exp_07 seen-protocol launcher guards: arms, goldens, per-arm controls, seen bindings.

Every check is CPU-only: no trainer is spawned and no attempt directory is created
outside ``tmp_path``.  The historical comparators are the REAL ``ckpt/...args.json``
files of exp_01 and exp_04 (read-only), so a recipe drift would be caught here.
"""
import json

import pytest

from tools import exp04_launcher as launch
from tools import provenance as p

RECIPE = ('--num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 '
          '--lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 '
          '--seed 0 --tf32 --log-interval 50 --save-every 0 --epoch-ckpt-every 1').split()
YAW = ['--yaw-aug', '1', '--yaw-aug-seed', '0', '--yaw-aug-width', '512']
ARMS = {'seen_simple': ('simple', 0), 'seen_cyl': ('cylindrical', 0), 'seen_aug': ('simple', 1)}
CONTROLS = {'seen_simple': 'ckpt/xRIR_simple_8_shot/args.json',
            'seen_cyl': 'ckpt/xRIR_cyl_8_shot/args.json',
            'seen_aug': 'ckpt/xRIR_simple_yawaug_8_shot/final/args.json'}
SEEN_BATCHES = 9265
SMOKE = launch.command('smoke', '<x>')[6:-6]  # the exp_04 smoke recipe, arm/protocol flags aside


def attempt_of(arm, mode='full'):
    return 'ckpt/exp07/{}/{}test'.format(arm, 'attempt_' if mode == 'full' else '_smoke_')


def argv_of(arm, attempt=None, mode='full'):
    backbone, yaw = ARMS[arm]
    return launch.command(mode, attempt or attempt_of(arm, mode), 'M', backbone, 'seen', yaw)


@pytest.mark.parametrize('mode', ['full', 'smoke'])
@pytest.mark.parametrize('arm', sorted(ARMS))
def test_golden_argv_equals_the_plan_recipe(arm, mode):
    backbone, yaw = ARMS[arm]
    attempt = attempt_of(arm, mode)
    recipe = RECIPE if mode == 'full' else SMOKE
    expected = ([launch.PYTHON, 'train_xRIR_backbone.py', '--backbone', backbone, '--save-dir',
                 attempt + ('' if mode == 'full' else '/smoke')] + recipe +
                ['--protocol', 'seen'] + (YAW if yaw else []))
    argv = argv_of(arm, attempt, mode)
    assert argv == expected
    launch.check_golden(argv, mode, attempt)
    golden = (launch.REPO / launch.EXP07_RECORD / 'seen_protocol_results_assets' /
              ('argv_golden_{}{}.txt'.format(arm, '' if mode == 'full' else '_smoke')))
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


@pytest.mark.parametrize('arm,attempt', [
    ('seen_simple', 'ckpt/xRIR_simple_yawaug_8_shot/attempt_test'),
    ('seen_aug', 'ckpt/xRIR_simple_yawaug_8_shot/attempt_test'),
    ('seen_simple', 'ckpt/exp05/S_simple/attempt_test'),
    ('seen_simple', 'ckpt/exp07/seen_cyl/attempt_test'),
    ('seen_aug', 'ckpt/exp07/seen_simple/attempt_test'),
    ('seen_simple', 'ckpt/exp07/seen_simple/_smoke_test'),
    ('seen_simple', 'ckpt/exp07/seen_simple/attempt_../other')])
def test_seen_argv_refused_outside_its_own_arm(arm, attempt):
    with pytest.raises(ValueError, match='golden'):
        launch.check_golden(argv_of(arm, attempt), 'full', attempt)


@pytest.mark.parametrize('tier,backbone', [('M', 'simple'), ('S', 'simple'), ('L', 'cylindrical')])
def test_unseen_argv_refused_under_a_seen_root(tier, backbone):
    attempt = attempt_of('seen_simple')
    with pytest.raises(ValueError, match='golden'):
        launch.check_golden(launch.command('full', attempt, tier, backbone), 'full', attempt)


@pytest.mark.parametrize('tier,backbone,yaw', [('S', 'simple', 0), ('L', 'cylindrical', 0),
                                               ('M', 'cylindrical', 1), ('M', 'simple', 2)])
def test_command_refuses_unregistered_seen_arms(tier, backbone, yaw):
    with pytest.raises(ValueError, match='seen'):
        launch.command('full', attempt_of('seen_simple'), tier, backbone, 'seen', yaw)


def test_full_requires_the_seen_batch_count():
    args = launch.effective_args(argv_of('seen_simple'), '1', SEEN_BATCHES)
    launch.check_runtime(args, args, 'full')
    wrong = launch.effective_args(argv_of('seen_simple'), '1', 9261)
    with pytest.raises(ValueError, match='9265'):
        launch.check_runtime(wrong, wrong, 'full')
    unseen = launch.effective_args(launch.command('full', 'attempt'), '1', SEEN_BATCHES)
    with pytest.raises(ValueError, match='9261'):
        launch.check_runtime(unseen, unseen, 'full')


@pytest.mark.parametrize('arm', sorted(ARMS))
def test_banner_rule_per_arm(arm):
    expected = launch.effective_args(argv_of(arm), '1', SEEN_BATCHES)
    guard = launch.LogGuard(expected, 'full')
    enabled = 'yaw_aug ENABLED W=512 seed=0 counter=(epoch-1)*{}+batch_idx'
    wanted = enabled.format(SEEN_BATCHES) if arm == 'seen_aug' else 'yaw_aug DISABLED'
    for other in (enabled.format(9261), 'yaw_aug DISABLED' if arm == 'seen_aug' else enabled.format(SEEN_BATCHES)):
        guard.feed(other)
        assert not guard.banner
    guard.feed(wanted)
    assert guard.banner


def test_arm_roots_and_ledger_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    roots = [launch.arm_root('M', backbone, 'seen', yaw) for backbone, yaw in ARMS.values()]
    assert roots == [tmp_path / 'ckpt/exp07' / arm for arm in ARMS]
    assert launch.arm_root('M', 'simple') == launch.ROOT  # the exp_04 root is unchanged
    with pytest.raises(ValueError, match='seen arm'):
        launch.arm_root('M', 'cylindrical', 'seen', 1)
    launch.account_hours(launch.create_attempt(roots[0], 'attempt_test'), 2, mode='full')
    assert [launch.full_hours(root) for root in roots] == [2, 0, 0]
