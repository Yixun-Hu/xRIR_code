"""Tests for :mod:`tools.per_sample_metrics` (exp_03, yaw_rotation_degradation).

exp_03 compares a query against itself under rotation, so every metric has to be
*per sample* -- and numerically identical to the aggregate the baseline scripts report.
The loss halves therefore go through the original ``utils.spec_utils`` functions on
1-sample slices, exactly as ``train_xRIR_backbone.compute_loss`` calls them on a batch.
"""
import contextlib
import io

import pytest
import torch

from tools.per_sample_metrics import (
    acoustic_metrics,
    griffin_lim_seeded,
    per_sample_losses,
    sample_seed,
)
from utils.spec_utils import compute_spect_energy_decay_losses, stft_l1_loss


def _batch(batch_size=4, freq=63, time=310, seed=0):
    """A seeded (log-magnitude prediction, magnitude target) pair in the model's layout."""
    gen = torch.Generator().manual_seed(seed)
    tgt_spec = torch.rand(batch_size, 1, freq, time, generator=gen) * 2.0 + 1e-3   # positive
    out_spec = torch.log(torch.rand(batch_size, freq, time, 1, generator=gen) * 2.0 + 1e-3)
    return out_spec, tgt_spec


def _direct(out_spec, tgt_spec, i):
    """The exp_01 / train_xRIR_backbone loss on slice ``i``, called by hand."""
    gt = tgt_spec[i:i + 1].permute(0, 2, 3, 1)
    with contextlib.redirect_stdout(io.StringIO()):
        decay = compute_spect_energy_decay_losses(gts=gt, preds=torch.exp(out_spec[i:i + 1]) - 1e-8)
    stft = stft_l1_loss(pred_spect=out_spec[i:i + 1], gt_spect=gt)
    return float(stft + decay), float(stft), float(decay)


# --------------------------------------------------------------------------------------
# T11 -- per_sample_losses
# --------------------------------------------------------------------------------------
def test_per_sample_losses_match_the_one_sample_calls():
    out_spec, tgt_spec = _batch()
    loss, stft, decay = per_sample_losses(out_spec, tgt_spec)

    for t in (loss, stft, decay):
        assert t.shape == (4,)
        assert t.dtype == torch.float32
        assert t.device.type == "cpu"
        assert not t.requires_grad

    for i in range(4):
        want_loss, want_stft, want_decay = _direct(out_spec, tgt_spec, i)
        assert float(stft[i]) == pytest.approx(want_stft, abs=1e-7, rel=0)
        assert float(decay[i]) == pytest.approx(want_decay, abs=1e-7, rel=0)
        assert float(loss[i]) == pytest.approx(want_loss, abs=1e-7, rel=0)
    assert torch.allclose(loss, stft + decay, atol=1e-7)


def test_per_sample_stft_terms_average_to_the_batch_loss():
    """Both halves reduce with `mean`, so the per-sample mean is the batch value."""
    out_spec, tgt_spec = _batch(seed=1)
    _, stft, decay = per_sample_losses(out_spec, tgt_spec)

    gt = tgt_spec.permute(0, 2, 3, 1)
    batch_stft = float(stft_l1_loss(pred_spect=out_spec, gt_spect=gt))
    with contextlib.redirect_stdout(io.StringIO()):
        batch_decay = float(compute_spect_energy_decay_losses(
            gts=gt, preds=torch.exp(out_spec) - 1e-8))
    assert float(stft.mean()) == pytest.approx(batch_stft, abs=1e-6, rel=0)
    assert float(decay.mean()) == pytest.approx(batch_decay, abs=1e-6, rel=0)


def test_per_sample_losses_are_silent_and_grad_free():
    out_spec, tgt_spec = _batch(batch_size=2, seed=2)
    out_spec.requires_grad_(True)
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        loss, stft, decay = per_sample_losses(out_spec, tgt_spec)
    assert captured.getvalue() == "", "the energy-decay loss prints; it must be muffled"
    for t in (loss, stft, decay):
        assert not t.requires_grad and t.grad_fn is None
    assert out_spec.grad is None


def test_per_sample_losses_separate_the_samples():
    """Sample 1 is a perfect prediction, sample 0 is not: the losses must differ."""
    gen = torch.Generator().manual_seed(3)
    tgt_spec = torch.rand(2, 1, 63, 310, generator=gen) * 2.0 + 1e-3
    out_spec = torch.log(tgt_spec.permute(0, 2, 3, 1) + 1e-8)
    out_spec[0] = torch.log(torch.rand(63, 310, 1, generator=gen) * 2.0 + 1e-3)

    loss, stft, decay = per_sample_losses(out_spec, tgt_spec)
    assert float(stft[1]) < 1e-6 and float(decay[1]) < 1e-6
    assert float(stft[0]) > 1e-3
    assert float(loss[0]) > float(loss[1])


# --------------------------------------------------------------------------------------
# T12 -- sample_seed / griffin_lim_seeded
# --------------------------------------------------------------------------------------
def test_sample_seed_is_stable_distinct_and_63_bit():
    a = sample_seed(0, "Office/Office_idx_10/S001_R000_hybrid_IR.wav")
    assert isinstance(a, int)
    assert a == sample_seed(0, "Office/Office_idx_10/S001_R000_hybrid_IR.wav")
    assert 0 <= a < 2 ** 63
    assert a != sample_seed(1, "Office/Office_idx_10/S001_R000_hybrid_IR.wav")
    assert a != sample_seed(0, "Office/Office_idx_10/S002_R000_hybrid_IR.wav")

    keys = ["Room/S00{}_R000_hybrid_IR.wav".format(i) for i in range(200)]
    seeds = [sample_seed(0, k) for k in keys]
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= s < 2 ** 63 for s in seeds)


def _mag(freq=63, time=310, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(freq, time, generator=gen) * 2.0


def test_griffin_lim_seeded_is_reproducible_and_seed_sensitive():
    mag = _mag()
    first = griffin_lim_seeded(mag, 12345)
    assert first.dim() == 2 and first.shape[0] == 1
    assert first.shape[1] > 8000, "310 frames at hop 31 must invert to > 8000 samples"
    assert torch.isfinite(first).all()

    # Same seed, same magnitudes (a distinct tensor object) -> bit-identical waveform.
    assert torch.equal(griffin_lim_seeded(mag.clone(), 12345), first)
    # Re-seeding happens inside the call, so an intervening draw cannot shift it.
    torch.rand(17)
    assert torch.equal(griffin_lim_seeded(mag, 12345), first)
    # A different seed picks a different random phase.
    assert not torch.equal(griffin_lim_seeded(mag, 12346), first)


def test_griffin_lim_seeded_accepts_both_layouts_and_rejects_others():
    mag = _mag(seed=1)
    flat = griffin_lim_seeded(mag, 7)
    batched = griffin_lim_seeded(mag.unsqueeze(0), 7)
    assert batched.shape == flat.shape
    assert torch.equal(batched, flat), "[F, T] and [1, F, T] must invert identically"

    for bad in (torch.rand(2, 63, 310), torch.rand(310), torch.rand(1, 1, 63, 310)):
        with pytest.raises(ValueError):
            griffin_lim_seeded(bad, 7)


def test_griffin_lim_seeded_matches_the_unrolled_eval_path():
    """Bit for bit the same as seeding by hand and calling eval_unseen.griffin_lim."""
    from eval_unseen import griffin_lim

    mag = _mag(seed=4)
    seed = sample_seed(0, "Office/Office_idx_10/S001_R000_hybrid_IR.wav")

    got = griffin_lim_seeded(mag, seed)
    torch.manual_seed(seed)
    want = griffin_lim(mag.unsqueeze(0))
    assert torch.equal(got, want)

    # ... and it carries the slice eval_xRIR_backbone.py actually measures.
    torch.manual_seed(seed)
    eval_style = griffin_lim(mag.unsqueeze(0)).unsqueeze(0)      # [1, 1, T], as in exp_01
    assert torch.equal(got[0, :8000], eval_style[0, 0, :8000])


def test_griffin_lim_seeded_separates_different_magnitudes():
    seeded = griffin_lim_seeded(_mag(seed=2), 99)
    other = griffin_lim_seeded(_mag(seed=3), 99)
    assert not torch.equal(seeded, other)


# --------------------------------------------------------------------------------------
# T13 -- acoustic_metrics (exp_01's EDT / C50 / T60 rules, per sample, NaN when invalid)
# --------------------------------------------------------------------------------------
def _decaying_ir(seed=0, n=8000, tau=1200.0):
    import numpy as np

    rng = np.random.RandomState(seed)
    return (rng.randn(n) * np.exp(-np.arange(n) / tau)).astype(np.float64)


@pytest.fixture(scope="module")
def evaluator():
    from eval_unseen import Evaluator

    return Evaluator()


def test_acoustic_metrics_are_zero_for_a_perfect_prediction(evaluator):
    gt = _decaying_ir()
    got = acoustic_metrics(gt.copy(), gt, evaluator)
    assert set(got) == {"edt", "c50", "t60"}
    assert got["edt"] == 0.0
    assert got["c50"] == 0.0
    assert got["t60"] == 0.0


def test_acoustic_metrics_follow_the_exp_01_formulas(evaluator):
    import numpy as np

    gt, pred = _decaying_ir(seed=0, tau=1200.0), _decaying_ir(seed=1, tau=600.0)
    got = acoustic_metrics(pred, gt, evaluator)

    assert got["edt"] == pytest.approx(
        abs(evaluator.measure_edt(gt) - evaluator.measure_edt(pred)))
    assert got["c50"] == pytest.approx(
        abs(evaluator.measure_clarity(gt) - evaluator.measure_clarity(pred)))
    gt_t60, pred_t60 = evaluator.measure_rt60(gt), evaluator.measure_rt60(pred)
    assert got["t60"] == pytest.approx(abs(gt_t60 - pred_t60) / gt_t60 * 100.0)
    assert all(np.isfinite(v) for v in got.values())
    # T60 is a *relative* error, so swapping the two arguments changes it.
    assert acoustic_metrics(gt, pred, evaluator)["t60"] != got["t60"]


def test_acoustic_metrics_return_nan_for_a_degenerate_prediction(evaluator):
    import warnings

    import numpy as np

    gt = _decaying_ir()
    zeros = np.zeros(8000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # numpy's 0/0 must be suppressed locally
        got = acoustic_metrics(zeros, gt, evaluator)
    assert np.isnan(got["edt"]) and np.isnan(got["c50"]) and np.isnan(got["t60"])
    # A degenerate ground truth is just as survivable.
    assert all(np.isnan(v) for v in acoustic_metrics(gt, zeros, evaluator).values())


def test_acoustic_metrics_apply_the_8000_sample_window_themselves(evaluator):
    """A caller must not be able to widen the metric window by passing a longer IR."""
    import numpy as np

    gt, pred = _decaying_ir(seed=0), _decaying_ir(seed=1, tau=600.0)
    poison = np.full(4000, 50.0)                   # a loud tail beyond the window
    windowed = acoustic_metrics(pred, gt, evaluator)
    poisoned = acoustic_metrics(np.concatenate([pred, poison]),
                                np.concatenate([gt, poison]), evaluator)
    assert poisoned == windowed, "samples past 8000 must not reach the measurements"
    # The window is the documented default, and a narrower one really does change things.
    assert acoustic_metrics(pred, gt, evaluator, window=8000) == windowed
    assert acoustic_metrics(pred, gt, evaluator, window=4000) != windowed
    for bad in (0, -1, 8000.0, True):
        with pytest.raises(ValueError):
            acoustic_metrics(pred, gt, evaluator, window=bad)


class _StubEvaluator:
    """Returns fixed measurements, so the degenerate branches can be reached exactly."""

    def __init__(self, gt_t60, pred_t60):
        self._t60 = {0: gt_t60, 1: pred_t60}
        self.rt60_calls = 0

    def measure_edt(self, h):
        return 0.5

    def measure_clarity(self, h):
        return 3.0

    def measure_rt60(self, h):
        self.rt60_calls += 1
        return self._t60[int(h[0])]


def test_acoustic_metrics_normalise_a_non_finite_t60_to_nan():
    import numpy as np

    gt, pred = np.array([0.0, 1.0]), np.array([1.0, 1.0])
    # A zero ground-truth T60 makes the relative error infinite, not merely large.
    stub = _StubEvaluator(gt_t60=np.float64(0.0), pred_t60=np.float64(0.5))
    got = acoustic_metrics(pred, gt, stub)
    assert np.isnan(got["t60"]), "an infinite relative T60 error is invalid, not a value"
    assert got["edt"] == 0.0 and got["c50"] == 0.0

    finite = _StubEvaluator(gt_t60=np.float64(0.4), pred_t60=np.float64(0.5))
    assert acoustic_metrics(pred, gt, finite)["t60"] == pytest.approx(25.0)


def test_acoustic_metrics_never_measure_t60_when_it_is_not_wanted():
    import numpy as np

    gt, pred = np.array([0.0, 1.0]), np.array([1.0, 1.0])
    stub = _StubEvaluator(gt_t60=np.float64(0.4), pred_t60=np.float64(0.5))
    assert np.isnan(acoustic_metrics(pred, gt, stub, want_t60=False)["t60"])
    assert stub.rt60_calls == 0, "want_t60=False must not even call measure_rt60"
    acoustic_metrics(pred, gt, stub, want_t60=True)
    assert stub.rt60_calls == 2


def test_acoustic_metrics_can_skip_t60(evaluator):
    import numpy as np

    gt, pred = _decaying_ir(seed=0), _decaying_ir(seed=1, tau=600.0)
    got = acoustic_metrics(pred, gt, evaluator, want_t60=False)
    assert np.isnan(got["t60"])
    assert np.isfinite(got["edt"]) and np.isfinite(got["c50"])
    full = acoustic_metrics(pred, gt, evaluator, want_t60=True)
    assert got["edt"] == full["edt"] and got["c50"] == full["c50"]
