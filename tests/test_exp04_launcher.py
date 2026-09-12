"""CPU guards for the exp_04 training launcher."""
import json
from pathlib import Path
import pytest
from tools import exp04_launcher as launch


@pytest.mark.parametrize('mode', ['smoke', 'full'])
def test_golden_argv_and_one_flag_refusal(mode, tmp_path):
    command = launch.command(mode, '<attempt>')
    launch.check_golden(command, mode, '<attempt>')
    command[command.index('--lr') + 1] = '0.002'
    with pytest.raises(ValueError, match='golden'):
        launch.check_golden(command, mode, '<attempt>')


def test_normalized_runtime_schema_and_mismatch():
    args = launch.effective_args(launch.command('full', 'ckpt/attempt'), '1', 9261)
    assert args['resume'] is None and args['no_save'] is False
    assert args['yaw_aug_seed'] == 0 and args['train_batches_per_epoch'] == 9261
    assert args['CUDA_VISIBLE_DEVICES'] == '1'
    runtime = {k: v for k, v in args.items() if k not in launch.ENV_KEYS}
    runtime['env'] = {k: args[k] for k in launch.ENV_KEYS}
    assert launch.normalize(runtime) == args
    launch.check_runtime(runtime, args, 'full')
    for key, value in [('lr', '0.001'), ('yaw_aug', True), ('train_batches_per_epoch', 9260)]:
        changed = dict(runtime, **{key: value})
        with pytest.raises(ValueError, match=key):
            launch.check_runtime(changed, args, 'full')
    runtime['unexpected'] = 1
    with pytest.raises(ValueError, match='unexpected'):
        launch.check_runtime(runtime, args, 'full')


def test_full_refuses_matching_but_wrong_bpe():
    args = launch.effective_args(launch.command('full', 'ckpt/attempt'), '1', 9260)
    with pytest.raises(ValueError, match='9261'):
        launch.check_runtime(args, args, 'full')
