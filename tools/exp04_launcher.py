"""Fail-closed training launcher for exp_04 (full runs require a reviewed commit)."""
import json
import os
from pathlib import Path
import shlex
import sys
from unittest.mock import patch

PYTHON = '/home/yixunhu/miniconda3/envs/xRIR/bin/python'
REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'ckpt/xRIR_simple_yawaug_8_shot'
GOLDENS = REPO / ('worklog/worklog_yixun/exp_04_yaw_aug_xrir_claude/'
                  'yaw_aug_xrir_results_assets')
ENV_KEYS = ('XRIR_DATA_PATH', 'PYTHONHASHSEED', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')
DATA_ROOT = '/home/yixunhu/data_cache/AcousticRooms'


def child_environment(gpu):
    return dict(os.environ, XRIR_DATA_PATH=DATA_ROOT, PYTHONHASHSEED='0',
                OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES=str(gpu),
                PYTHONPATH=str(REPO), PYTHONUNBUFFERED='1')


def command(mode, attempt):
    if mode not in ('smoke', 'full'):
        raise ValueError('command requires smoke or full')
    smoke = mode == 'smoke'
    result = [PYTHON, 'train_xRIR_backbone.py', '--backbone', 'simple', '--save-dir',
              str(attempt) + ('/smoke' if smoke else '')]
    result += shlex.split('--num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 '
                          '--decay-epochs 3 --lr-gamma 0.1 --epochs ' + ('1' if smoke else '12'))
    if smoke:
        result += shlex.split('--max-train-batches 3 --max-test-batches 2')
    result += shlex.split('--batch-size 4 --accum-steps 2 --num-workers 4' if smoke else
                          '--batch-size 32 --accum-steps 2 --num-workers 12')
    result += shlex.split('--seed 0 --tf32 --log-interval ' + ('1' if smoke else '50') +
                          ' --save-every 0 --epoch-ckpt-every ' + ('0 --no-save' if smoke else '1'))
    return result + shlex.split('--yaw-aug 1 --yaw-aug-seed 0 --yaw-aug-width 512')


def check_golden(argv, mode, attempt):
    golden = shlex.split((GOLDENS / ('argv_golden_' + mode + '.txt')).read_text())
    expected = [part.replace('<attempt>', str(attempt)) for part in golden]
    if argv != expected:
        raise ValueError('argv differs from golden ' + mode)


def effective_args(argv, gpu, batches_per_epoch):
    import train_xRIR_backbone as trainer
    with patch.object(sys, 'argv', argv[1:]):
        result = vars(trainer.parse_args())
    result.update({key: child_environment(gpu)[key] for key in ENV_KEYS})
    result['train_batches_per_epoch'] = batches_per_epoch
    return result


def normalize(runtime):
    result = dict(runtime)
    env = result.pop('env', {})
    for key, value in env.items():
        if key in result and (type(result[key]) is not type(value) or result[key] != value):
            raise ValueError('conflicting env.' + key)
        result[key] = value
    return result


def check_runtime(runtime, expected, mode):
    actual = normalize(runtime)
    differences = [key for key in actual.keys() | expected.keys() if key not in actual or
                   key not in expected or type(actual[key]) is not type(expected[key]) or
                   actual[key] != expected[key]]
    if differences:
        raise ValueError('runtime args mismatch: ' + ', '.join(sorted(differences)))
    if mode == 'full' and actual['train_batches_per_epoch'] != 9261:
        raise ValueError('full requires train_batches_per_epoch == 9261')
