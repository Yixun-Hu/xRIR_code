"""Per-sample versions of the exp_01 evaluation metrics (exp_03).

exp_03's unit of analysis is a single query compared with *itself* under a yaw
rotation, so every metric has to exist per sample and be numerically identical to what
the baseline scripts compute:

* the loss halves go through the original ``utils.spec_utils`` functions on 1-sample
  slices, laid out exactly as ``train_xRIR_backbone.compute_loss`` calls them;
* the waveform comes from ``eval_unseen.griffin_lim``, but with its random phase
  initialisation seeded per sample so that ``k = 0`` and ``k != 0`` -- and every model,
  in any angle order -- invert the same magnitudes to the same waveform;
* the acoustic errors apply ``eval_xRIR_backbone.py``'s validity rules unchanged,
  returning NaN instead of raising or dropping a sample.
"""
from __future__ import annotations

import contextlib
import io

import torch

from utils.spec_utils import compute_spect_energy_decay_losses, stft_l1_loss


def per_sample_losses(out_spec, tgt_spec):
    """Per-sample test loss, split into its STFT-L1 and energy-decay halves.

    Calls the two loss functions on 1-sample slices, in the same layout as
    ``train_xRIR_backbone.compute_loss`` (which reduces with ``mean``), so the mean of
    the returned per-sample values is the batch loss and no batch-size effect can leak
    into a paired comparison.  ``compute_spect_energy_decay_losses`` prints its input
    shapes on every call, so stdout is redirected exactly as the training script does.

    Args:
        out_spec: predicted log-magnitude spectrograms ``[B, F, T, 1]`` (model output).
        tgt_spec: target magnitude spectrograms ``[B, 1, F, T]`` (model output).

    Returns:
        ``(loss, stft, decay)``: three detached CPU float32 tensors of shape ``[B]``,
        with ``loss = stft + decay``.
    """
    stft_terms = []
    decay_terms = []
    with torch.no_grad():
        for i in range(out_spec.shape[0]):
            gt = tgt_spec[i:i + 1].permute(0, 2, 3, 1)
            pred = out_spec[i:i + 1]
            with contextlib.redirect_stdout(io.StringIO()):
                decay_terms.append(compute_spect_energy_decay_losses(
                    gts=gt, preds=torch.exp(pred) - 1e-8))
            stft_terms.append(stft_l1_loss(pred_spect=pred, gt_spect=gt))
    stft = torch.stack(stft_terms).detach().cpu().float()
    decay = torch.stack(decay_terms).detach().cpu().float()
    return stft + decay, stft, decay
