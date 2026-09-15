"""Fail-closed seen-protocol training launcher for exp_07 (plan section 2, amendment A1).

``tools/exp04_launcher.py`` keeps main's bytes.  Everything operational is reused from it
by composition -- the ledgers, the resource gate, ``run_child``, ``execute_attempt`` /
``complete_attempt`` / ``finalize_attempt``, ``normalize``, ``validate_parameters`` --
and only the arm-specific decisions are stated here: the three seen arms, their goldens
(which invoke ``tools/exp07_train.py``), the per-arm unseen-twin comparator, the seen
batch count, the ``seen_split`` binding, the seen banner/probe lines and the one-retry
ceiling of ``tools/exp07_gates.py``.

The shared orchestration looks its collaborators up as module globals of
``exp04_launcher``, so :func:`seen_overrides` rebinds exactly those names for the
duration of an exp_07 launch and restores them afterwards; ``exp04_launcher``'s own file
and its behaviour outside that context are untouched.

    tools/exp07_launch.sh smoke --backbone simple --gpu 1 --reviewed-commit <sha>
    tools/exp07_launch.sh probe --backbone simple --gpu 1 --reviewed-commit <sha>
    nohup setsid tools/exp07_launch.sh full --backbone simple --gpu 1 \
        --reviewed-commit <sha> --probe-json <receipt> > <launcher.log> 2>&1 &
    tools/exp07_launch.sh finalize <attempt> --launcher-log <launcher.log>
"""
import json
import math
import os
import re
import shlex
from pathlib import Path

from tools import exp04_launcher as base
from tools import exp07_gates as gates
from tools import exp07_probe as probe
from tools import exp07_provenance as e7p
from tools import exp07_train
from tools import provenance as p
from tools.exp05_params import TIERS, tier_of

PYTHON = base.PYTHON
REPO = base.REPO
DATA_ROOT = base.DATA_ROOT
ENV_KEYS = base.ENV_KEYS
PROTOCOL = 'seen'
ENTRY = 'tools/exp07_train.py'
ENTRY_MODULE = 'tools.exp07_train'
RECORD = 'worklog/worklog_yixun/exp_07_seen_protocol_claude'
ASSETS = 'seen_protocol_results_assets'
SEEN_MODULE = exp07_train.SEEN_MODULE
SEEN_INPUTS = {'seen_split'}  # mutable inputs every seen run must bind and revalidate
ARMS = {('simple', 0): 'seen_simple', ('cylindrical', 0): 'seen_cyl', ('simple', 1): 'seen_aug'}
CONTROLS = {'seen_simple': 'ckpt/xRIR_simple_8_shot/args.json',
            'seen_cyl': 'ckpt/xRIR_cyl_8_shot/args.json',
            'seen_aug': 'ckpt/xRIR_simple_yawaug_8_shot/final/args.json'}
TRAIN_FILES = {'unseen': 296334, 'seen': 296454}
TRAIN_BATCHES = {'unseen': 9261, 'seen': 9265}
YAW_FLAGS = '--yaw-aug 1 --yaw-aug-seed 0 --yaw-aug-width 512'
CONTROL_EXCLUSIONS = {'save_dir', 'save_every', 'epoch_ckpt_every',
                      'PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES'}


def flag_value(argv, flag, default=None):
    index = argv.index(flag) if flag in argv else -1
    return argv[index + 1] if 0 <= index < len(argv) - 1 else default


def arm(backbone, yaw_aug=0):
    """Name the arm of a backbone/yaw pair; the flag may be an int or its argv string."""
    key = (backbone, {None: 0, 0: 0, 1: 1, '0': 0, '1': 1}.get(yaw_aug, -1))
    if key not in ARMS:
        raise ValueError('no seen arm for backbone/yaw: {}/{}'.format(backbone, yaw_aug))
    return ARMS[key]


def arm_root(backbone, yaw_aug=0):
    return REPO / 'ckpt/exp07' / arm(backbone, yaw_aug)


def goldens():
    return REPO / RECORD / ASSETS


def command(mode, attempt, backbone='simple', yaw_aug=0):
    """The plan's recipe for one arm, invoking the exp_07 entry point."""
    if mode not in ('smoke', 'full'):
        raise ValueError('command requires smoke or full')
    arm(backbone, yaw_aug)  # an unregistered backbone/yaw pair has no seen arm
    smoke = mode == 'smoke'
    result = [PYTHON, ENTRY, '--backbone', backbone, '--save-dir',
              str(attempt) + ('/smoke' if smoke else '')]
    result += shlex.split('--num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 '
                          '--decay-epochs 3 --lr-gamma 0.1 --epochs ' + ('1' if smoke else '12'))
    if smoke:
        result += shlex.split('--max-train-batches 3 --max-test-batches 2')
    result += shlex.split('--batch-size 4 --accum-steps 2 --num-workers 4' if smoke else
                          '--batch-size 32 --accum-steps 2 --num-workers 12')
    result += shlex.split('--seed 0 --tf32 --log-interval ' + ('1' if smoke else '50') +
                          ' --save-every 0 --epoch-ckpt-every ' + ('0 --no-save' if smoke else '1'))
    result += ['--protocol', PROTOCOL]
    return result + (shlex.split(YAW_FLAGS) if arm(backbone, yaw_aug) == 'seen_aug' else [])


def check_golden(argv, mode, attempt):
    """Every argv the launcher spawns equals its arm's golden file, path aside."""
    if mode not in ('smoke', 'full'):
        raise ValueError('argv differs from golden: unknown mode ' + str(mode))
    try:
        name = arm(flag_value(argv, '--backbone'), flag_value(argv, '--yaw-aug', 0))
    except ValueError as error:
        raise ValueError('argv differs from golden seen arm') from error
    if list(argv[:2]) != [PYTHON, ENTRY] or flag_value(argv, '--protocol') != PROTOCOL:
        raise ValueError('argv differs from golden seen arm')
    placeholder = 'ckpt/exp07/{}/{}<ts>'.format(name, 'attempt_' if mode == 'full' else '_smoke_')
    prefix = placeholder[:-len('<ts>')]
    if (not str(attempt).startswith(prefix)
            or not re.fullmatch(r'[A-Za-z0-9_-]+', str(attempt)[len(prefix):])):
        raise ValueError('argv differs from golden seen arm')
    path = goldens() / ('argv_golden_{}{}.txt'.format(name, '' if mode == 'full' else '_smoke'))
    if not path.is_file():
        raise ValueError('golden file missing: ' + path.name)
    expected = [part.replace(placeholder, str(attempt)) for part in shlex.split(path.read_text())]
    if list(argv) != expected:
        raise ValueError('argv differs from golden ' + mode)


def child_environment(gpu):
    return dict(os.environ, XRIR_DATA_PATH=DATA_ROOT, PYTHONHASHSEED='0',
                OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES=str(gpu),
                PYTHONPATH=str(REPO), PYTHONUNBUFFERED='1')


def effective_args(argv, gpu, batches_per_epoch):
    """The recorded argument schema of an exp_07 run: the trainer's plus ``protocol``."""
    result = vars(exp07_train.parse_args(list(argv[2:])))
    result.update({key: child_environment(gpu)[key] for key in ENV_KEYS})
    result['train_batches_per_epoch'] = batches_per_epoch
    result.update(tier=tier_of(result),
                  param_counts=dict(base.tier_counts(result['backbone'], tier_of(result))))
    return result


def check_runtime(runtime, expected, mode):
    """The shared schema comparison, plus the protocol and its loader length."""
    actual = base.normalize(runtime)
    if 'tier' in actual or 'param_counts' in actual:
        base.validate_parameters(actual)
    differences = [key for key in actual.keys() | expected.keys() if key not in actual or
                   key not in expected or type(actual[key]) is not type(expected[key]) or
                   actual[key] != expected[key]]
    if differences:
        raise base.LauncherFailure('guard_runtime_args',
                                   'runtime args mismatch: ' + ', '.join(sorted(differences)))
    if actual.get('protocol') != PROTOCOL:
        raise ValueError('exp_07 runs require protocol ' + PROTOCOL)
    if mode == 'full' and actual['train_batches_per_epoch'] != TRAIN_BATCHES[PROTOCOL]:
        raise ValueError('full requires train_batches_per_epoch == ' + str(TRAIN_BATCHES[PROTOCOL]))


def compare_control(runtime, control, control_env):
    """A seen arm differs from its unseen twin in the protocol and nothing else."""
    treatment, baseline = base.normalize(runtime), base.normalize(dict(control, env=control_env))
    for values in (treatment, baseline):
        for key, value in TIERS['M'].items():
            values.setdefault('vit_' + key, value)
        base.validate_parameters(values)
        for key in ('tier', 'param_counts'):
            values.pop(key, None)
        values.setdefault('protocol', 'unseen')  # the historical comparators predate exp_07
        if values['protocol'] not in TRAIN_FILES:
            raise ValueError('unknown protocol: ' + str(values['protocol']))
        files = TRAIN_FILES[values['protocol']]
        bpe = values.pop('train_batches_per_epoch', math.ceil(files / values['batch_size']))
        if type(bpe) is not int or bpe != math.ceil(files / values['batch_size']):
            raise ValueError('invalid train_batches_per_epoch')
        for key, value in dict(yaw_aug=0, yaw_aug_seed=values['seed'],
                               yaw_aug_width=512, no_save=False).items():
            values.setdefault(key, value)
        if values['yaw_aug_seed'] is None:  # exp_05 A1: the historical default is the seed
            values['yaw_aug_seed'] = values['seed']
    if tier_of(baseline) != 'M' or tier_of(treatment) != 'M':
        raise ValueError('the seen arms and their unseen twins are tier M')
    if treatment['protocol'] != PROTOCOL:
        raise ValueError('control comparison requires a seen treatment')
    missing = object()
    differences = {key: {'treatment': treatment.get(key), 'control': baseline.get(key)}
        for key in treatment.keys() | baseline.keys()
        if type(treatment.get(key, missing)) is not type(baseline.get(key, missing))
        or treatment.get(key, missing) != baseline.get(key, missing)}
    if differences.get('protocol') != dict(treatment=PROTOCOL, control='unseen'):
        raise ValueError('control mismatch: a seen arm requires its unseen-protocol twin')
    refused = differences.keys() - (CONTROL_EXCLUSIONS | {'protocol'})
    if refused:
        raise ValueError('control mismatch: ' + ', '.join(sorted(refused)))
    return differences


def train_minimum():
    return base.TRAIN_MINIMUM | {SEEN_MODULE, ENTRY}
