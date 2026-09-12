"""Throughput measurements include loader fetch and completed training work."""
import math
import sys
from types import SimpleNamespace

import pytest

from tools.exp04_probe import TimedLoader, compare_results, run


class FakeCuda:
    def __init__(self):
        self.now = self.resets = 0

    def synchronize(self):
        self.now += 1

    def reset_peak_memory_stats(self):
        self.resets += 1

    def max_memory_allocated(self):
        return 123

    def max_memory_reserved(self):
        return 456


def test_timed_loader_covers_fetch_and_work(monkeypatch):
    cuda = FakeCuda()
    class Loader:
        def __len__(self):
            return 9261
        def __iter__(self):
            for i in range(100):
                cuda.now += 2
                yield i
    measured = TimedLoader(Loader(), cuda, clock=lambda: cuda.now)
    assert len(measured) == 9261
    seen = []
    for value in measured:
        seen.append(value)
        cuda.now += 5
    result = measured.result()
    assert seen == list(range(60)) and cuda.resets == 1
    assert result == dict(warmup_micro_batches=10, timed_micro_batches=50,
        mean_iteration_seconds=8, peak_allocated_bytes=123, peak_reserved_bytes=456)
    with pytest.raises(ValueError, match="60"):
        TimedLoader([1], cuda)
    with pytest.raises(ValueError, match="incomplete"):
        TimedLoader(range(60), cuda).result()


@pytest.mark.parametrize("on,passed", [(1.05, True), (1.050001, False)])
def test_comparison_threshold(on, passed):
    result = compare_results({"mean_iteration_seconds": 1}, {"mean_iteration_seconds": on})
    assert result["overhead_ratio"] == on and result["passed"] is passed


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf])
def test_comparison_refuses_invalid(value):
    with pytest.raises(ValueError, match="positive finite"):
        compare_results({"mean_iteration_seconds": value}, {"mean_iteration_seconds": 1})


@pytest.mark.parametrize("yaw", [0, 1])
def test_child_uses_trainer_path_without_saving(monkeypatch, capsys, yaw):
    cuda = FakeCuda()
    def train(model, loader, *args):
        for batch in loader:
            cuda.now += 1
        return 1.5
    def main():
        assert "--no-save" in sys.argv
        for flag, value in [("--batch-size", "32"), ("--accum-steps", "2"),
                            ("--max-train-batches", "60"), ("--yaw-aug", str(yaw))]:
            assert sys.argv[sys.argv.index(flag) + 1] == value
        assert fake.test_epoch(None) == 0
        fake.train_epoch(None, range(9261), None)
    fake = SimpleNamespace(main=main, train_epoch=train, test_epoch=None,
                           torch=SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, "train_xRIR_backbone", fake)
    run(yaw, "unused")
    assert "EXP04_PROBE_RESULT " in capsys.readouterr().out
    assert fake.train_epoch is train and fake.test_epoch is None
