"""Single-arm 32 x 2 tier probe; full test loader and durable scratch saves."""
import argparse
import copy
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from tools.exp04_probe import TimedLoader, trainer_command as base_command
from tools.exp05_params import TIERS


def trainer_command(tier, backbone, save_dir):
    if tier not in ('S', 'L') or backbone not in ('simple', 'cylindrical'):
        raise ValueError('tier probe requires S/L and simple/cylindrical')
    cmd = base_command(0, save_dir)
    cmd[cmd.index('--backbone') + 1] = backbone
    for key, value in TIERS[tier].items():
        cmd.extend(['--vit-' + key.replace('_', '-'), str(value)])
    return cmd + ['--max-test-batches', '0']


def projection(result):
    """Recompute the projection from its raw measurements; all times are seconds."""
    try:
        micro = result['t_micro']
        values = micro['values']
        valid = (len(values) == 50 and all(type(v) in (int, float) and math.isfinite(v) and v > 0
            for v in [*values, *[micro[k] for k in ('mean', 'median', 'min')], result['t_test'], result['t_save']])
            and (micro['mean'], micro['median'], micro['min']) == (sum(values) / 50, statistics.median(values), min(values))
            and all(type(result[k]) is int and result[k] == v for k, v in dict(warmup_micro_batches=10,
                timed_micro_batches=50, batch_size=32, accum_steps=2, train_batches_per_epoch=9261).items()))
        epoch = 9261 * micro['mean'] + result['t_test'] + result['t_save']
        if not valid or not math.isfinite(epoch * 12):
            raise ValueError('invalid measurement')
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid tier probe timing or recipe') from error
    return dict(T_epoch=epoch, T_run=12 * epoch, passed=12 * epoch <= 60 * 3600)


def run(tier, backbone, save_dir):
    import train_xRIR_backbone as trainer
    original_train, original_test, original_argv = trainer.train_epoch, trainer.test_epoch, sys.argv
    result = dict(tier=tier, backbone=backbone, yaw_aug=0, batch_size=32, accum_steps=2)
    state = {}
    def measured_train(model, loader, optimizer, scheduler, epoch, args, best):
        timed = TimedLoader(loader, trainer.torch.cuda, clock=time.perf_counter)
        loss = original_train(model, timed, optimizer, scheduler, epoch, args, best)
        if not math.isfinite(loss):
            raise ValueError('non-finite probe training loss')
        result.update(timed.result(), train_loss=loss, train_batches_per_epoch=len(loader))
        result['t_micro'] = dict(mean=result['mean_iteration_seconds'], median=result['median_iteration_seconds'],
                                min=result['min_iteration_seconds'], values=result['iteration_seconds'])
        state.update(optimizer=optimizer, scheduler=scheduler)
        return loss
    def measured_test(model, loader, epoch, args):
        trainer.torch.cuda.synchronize()
        started = time.perf_counter()
        loss = original_test(model, loader, epoch, args)
        trainer.torch.cuda.synchronize()
        result.update(t_test=time.perf_counter() - started, test_batches_timed=len(loader),
                      test_batches_total=len(loader), test_timing_protocol='full_test_loader')
        if not math.isfinite(loss):
            raise ValueError('non-finite probe test loss')
        with tempfile.TemporaryDirectory(prefix='_timed_save_', dir=save_dir) as scratch:
            save_args = copy.copy(args)
            save_args.no_save = False
            started = time.perf_counter()
            for name in ('epoch_001.pth', 'best.pth'):
                trainer.torch.save(model.state_dict(), Path(scratch) / name)
            trainer.save_checkpoint(Path(scratch) / 'last.pth', model, state['optimizer'],
                                    state['scheduler'], epoch, 0, loss, save_args)
            for path in Path(scratch).iterdir():
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
            trainer.torch.cuda.synchronize()
            result['t_save'] = time.perf_counter() - started
        return loss
    sys.argv = trainer_command(tier, backbone, save_dir)
    trainer.train_epoch, trainer.test_epoch = measured_train, measured_test
    try:
        trainer.main()
    finally:
        trainer.train_epoch, trainer.test_epoch, sys.argv = original_train, original_test, original_argv
    result.update(projection(result))
    print('EXP05_PROBE_RESULT ' + json.dumps(result, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', choices=('S', 'L'), required=True)
    parser.add_argument('--backbone', choices=('simple', 'cylindrical'), required=True)
    parser.add_argument('--save-dir', required=True)
    options = parser.parse_args()
    run(options.tier, options.backbone, options.save_dir)
