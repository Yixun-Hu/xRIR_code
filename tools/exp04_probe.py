"""Isolated, synchronized 10+50 micro-batch timing of the unchanged trainer."""
import argparse
import json
import math
import sys
import time


class TimedLoader:
    """Time fetching and consuming each batch, including optimizer and CUDA work."""
    def __init__(self, loader, cuda, clock=time.perf_counter):
        if len(loader) < 60:
            raise ValueError("probe requires at least 60 micro-batches")
        self.loader, self.cuda, self.clock = loader, cuda, clock
        self.times = []

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        iterator = iter(self.loader)
        for index in range(60):
            self.cuda.synchronize()
            started = self.clock()
            batch = next(iterator)
            yield batch
            self.cuda.synchronize()
            elapsed = self.clock() - started
            if index == 9:
                self.cuda.reset_peak_memory_stats()
            elif index >= 10:
                self.times.append(elapsed)

    def result(self):
        if len(self.times) != 50:
            raise ValueError("incomplete probe measurement")
        if any(not math.isfinite(t) or t <= 0 for t in self.times):
            raise ValueError("probe timings must be positive finite values")
        return dict(warmup_micro_batches=10, timed_micro_batches=50,
                    mean_iteration_seconds=sum(self.times) / 50,
                    peak_allocated_bytes=self.cuda.max_memory_allocated(),
                    peak_reserved_bytes=self.cuda.max_memory_reserved())


def compare_results(off, on):
    values = [r["mean_iteration_seconds"] for r in (off, on)]
    if any(not math.isfinite(t) or t <= 0 for t in values):
        raise ValueError("probe timings must be positive finite values")
    ratio = values[1] / values[0]
    return dict(overhead_ratio=ratio, overhead_fraction=ratio - 1, passed=ratio <= 1.05)


def trainer_command(yaw_aug, save_dir):
    """Expose the exact recipe for the launcher's pre-spawn arguments record."""
    return ["train_xRIR_backbone.py", "--backbone", "simple", "--save-dir", str(save_dir),
        "--num-shot", "8", "--max-len", "9600", "--lr", "1e-3", "--weight-decay", "1e-4",
        "--decay-epochs", "3", "--lr-gamma", "0.1", "--epochs", "1", "--batch-size", "32",
        "--accum-steps", "2", "--num-workers", "12", "--seed", "0", "--tf32",
        "--log-interval", "50", "--save-every", "0", "--epoch-ckpt-every", "0", "--no-save",
        "--max-train-batches", "60", "--yaw-aug", str(yaw_aug), "--yaw-aug-seed", "0",
        "--yaw-aug-width", "512"]


def run(yaw_aug, save_dir):
    """Run one arm in its own process; validation is outside throughput timing."""
    import train_xRIR_backbone as trainer
    original_train, original_test, original_argv = trainer.train_epoch, trainer.test_epoch, sys.argv
    result = {}
    def measured_train(model, loader, *args):
        timed = TimedLoader(loader, trainer.torch.cuda)
        loss = original_train(model, timed, *args)
        if not math.isfinite(loss):
            raise ValueError("non-finite probe training loss")
        result.update(timed.result(), yaw_aug=yaw_aug, train_loss=loss)
        return loss
    trainer.train_epoch = measured_train
    trainer.test_epoch = lambda *args: 0
    sys.argv = trainer_command(yaw_aug, save_dir)
    try:
        trainer.main()
    finally:
        trainer.train_epoch, trainer.test_epoch, sys.argv = original_train, original_test, original_argv
    print("EXP04_PROBE_RESULT " + json.dumps(result, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaw-aug", type=int, choices=(0, 1), required=True)
    parser.add_argument("--save-dir", required=True)
    options = parser.parse_args()
    run(options.yaw_aug, options.save_dir)
