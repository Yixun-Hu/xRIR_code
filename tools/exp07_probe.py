"""exp_07 seen fit-probe: the exp_05 single-arm 32 x 2 measurement at tier M.

Amendment A1: ``tools/exp05_probe.py`` keeps main's bytes, so the seen recipe lives here.
Everything measurable is composed from the reviewed pieces -- ``exp04_probe.TimedLoader``
for the timing, ``exp04_probe.trainer_command`` for the recipe skeleton -- and the one
function that cannot be wrapped is ``run``: ``exp05_probe.run`` hard-codes ``yaw_aug=0``,
carries no protocol and prints its own ``EXP05_PROBE_RESULT`` line, all three of which a
seen arm has to change.  It is therefore restated below with the same structure, the same
warm-up/timed split and the same durable-save measurement, and prints ``EXP07_PROBE_RESULT``
so an exp_05 or exp_04 receipt can never be mistaken for a seen one.

    python -m tools.exp07_probe --tier M --backbone simple --protocol seen \
        --yaw-aug 0 --save-dir ckpt/exp07/seen_simple/_probe_<ts>_arm
"""
import argparse
import copy
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path

from tools import exp07_train
from tools.exp04_probe import TimedLoader, trainer_command as base_command

PREFIX = 'EXP07_PROBE_RESULT '
PROTOCOLS = exp07_train.PROTOCOLS
TRAIN_BATCHES = {'unseen': 9261, 'seen': 9265}  # in lockstep with exp07_launcher.TRAIN_BATCHES
ENTRY = 'tools/exp07_train.py'
YAW_FLAGS = ('--yaw-aug', '--yaw-aug-seed', '--yaw-aug-width')
RUN_HOURS_CEILING = 60


def trainer_command(tier, backbone, save_dir, protocol='seen', yaw_aug=0):
    """The measured recipe: the plan's M arms under the seen protocol, one epoch of 60."""
    if tier != 'M' or backbone not in ('simple', 'cylindrical') or protocol != 'seen':
        raise ValueError('the seen probe requires tier M, simple/cylindrical and --protocol seen')
    if yaw_aug not in (0, 1):
        raise ValueError('yaw_aug must be 0 or 1')
    cmd = base_command(1 if yaw_aug else 0, save_dir)
    cmd[0] = ENTRY
    if not yaw_aug:
        for flag in YAW_FLAGS:
            index = cmd.index(flag)
            del cmd[index:index + 2]
    cmd[cmd.index('--backbone') + 1] = backbone
    return cmd + ['--protocol', protocol, '--max-test-batches', '0']


def projection(result):
    """Recompute the projection from its raw measurements; all times are seconds."""
    try:
        micro = result['t_micro']
        values = micro['values']
        protocol = result['protocol']
        if protocol not in TRAIN_BATCHES:
            raise ValueError('unknown protocol')
        batches = TRAIN_BATCHES[protocol]
        # Exact sum/50 survives JSON round trips; hand-transcribed summaries are refused.
        valid = (len(values) == 50 and all(type(v) in (int, float) and math.isfinite(v) and v > 0
            for v in [*values, *[micro[k] for k in ('mean', 'median', 'min')], result['t_test'], result['t_save']])
            and (micro['mean'], micro['median'], micro['min']) == (sum(values) / 50, statistics.median(values), min(values))
            and all(type(result[k]) is int and result[k] == v for k, v in dict(warmup_micro_batches=10,
                timed_micro_batches=50, batch_size=32, accum_steps=2, train_batches_per_epoch=batches).items()))
        epoch = batches * micro['mean'] + result['t_test'] + result['t_save']
        if not valid or not math.isfinite(epoch * 12):
            raise ValueError('invalid measurement')
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid seen probe timing or recipe') from error
    return dict(T_epoch=epoch, T_run=12 * epoch, passed=12 * epoch <= RUN_HOURS_CEILING * 3600)


def run(tier, backbone, save_dir, protocol='seen', yaw_aug=0):
    """Measure one seen arm; the trainer's own epoch runs inside the timed loader."""
    import train_xRIR_backbone as trainer
    argv = trainer_command(tier, backbone, save_dir, protocol, yaw_aug)
    original_train, original_test = trainer.train_epoch, trainer.test_epoch
    result = dict(tier=tier, backbone=backbone, protocol=protocol, yaw_aug=yaw_aug,
                  batch_size=32, accum_steps=2)
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

    trainer.train_epoch, trainer.test_epoch = measured_train, measured_test
    try:
        exp07_train.run_trainer(argv[1:])
    finally:
        trainer.train_epoch, trainer.test_epoch = original_train, original_test
    result.update(projection(result))
    print(PREFIX + json.dumps(result, allow_nan=False), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', choices=('M',), required=True)
    parser.add_argument('--backbone', choices=('simple', 'cylindrical'), required=True)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--protocol', choices=('seen',), required=True)
    parser.add_argument('--yaw-aug', type=int, choices=(0, 1), default=0)
    options = parser.parse_args(argv)
    return run(options.tier, options.backbone, options.save_dir, options.protocol, options.yaw_aug)


if __name__ == '__main__':
    main()
