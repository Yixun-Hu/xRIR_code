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

from tools.per_sample_metrics import per_sample_losses
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
        assert float(stft[i]) == pytest.approx(want_stft, abs=1e-7)
        assert float(decay[i]) == pytest.approx(want_decay, abs=1e-7)
        assert float(loss[i]) == pytest.approx(want_loss, abs=1e-7)
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
    assert float(stft.mean()) == pytest.approx(batch_stft, abs=1e-6)
    assert float(decay.mean()) == pytest.approx(batch_decay, abs=1e-6)


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
