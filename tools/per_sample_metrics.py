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
import hashlib
import io

import numpy as np
import torch

from eval_unseen import griffin_lim
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


def sample_seed(base_seed, query_key):
    """A stable 63-bit per-sample seed derived from ``(base_seed, query_key)``.

    Keyed by the query path rather than by a running counter, so the same query gets
    the same seed at every angle, in every model, whatever order the runs happen in.

    Args:
        base_seed: run-level seed (in exp_03, the manifest seed).
        query_key: the query's path relative to the IR root.

    Returns:
        A Python ``int`` in ``[0, 2**63)``.
    """
    digest = hashlib.sha256("{}:{}".format(base_seed, query_key).encode()).digest()[:8]
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def griffin_lim_seeded(mag_spec, seed):
    """``eval_unseen.griffin_lim`` with its random phase initialisation pinned.

    Griffin-Lim starts from a random phase drawn from the global torch RNG, so an
    unseeded inversion makes two evaluations of the *same* magnitudes differ -- noise
    that would be indistinguishable from a rotation effect in a paired comparison.
    Seeding immediately before the call makes the inversion a pure function of
    ``(mag_spec, seed)``.

    Note: this reseeds the global torch RNG as a side effect (that is the point); it is
    for evaluation, never inside training.

    Args:
        mag_spec: magnitude spectrogram ``[F, T]`` or ``[1, F, T]`` (the model's
            ``exp(out_spec) - 1e-8``).
        seed: the per-sample seed, e.g. from :func:`sample_seed`.

    Returns:
        The inverted waveform as ``[1, T]`` -- the batch dimension is kept, so callers
        index it exactly like ``eval_xRIR_backbone.py`` does (``wav[0, :8000]``).

    Raises:
        ValueError: if ``mag_spec`` is not ``[F, T]`` or ``[1, F, T]``.
    """
    mag = mag_spec
    if mag.dim() == 2:
        mag = mag.unsqueeze(0)
    if mag.dim() != 3 or mag.shape[0] != 1:
        raise ValueError("mag_spec must be [F, T] or [1, F, T], got shape {}".format(
            tuple(mag_spec.shape)))
    torch.manual_seed(int(seed))
    return griffin_lim(mag)


def acoustic_metrics(pred_ir, gt_ir, evaluator, want_t60=True):
    """EDT / C50 / T60 errors of one prediction, with exp_01's validity rules.

    Applies ``eval_xRIR_backbone.py``'s rules unchanged, on the arrays exactly as passed
    (the caller does the ``[:8000]`` slicing):

    * ``edt``  -- ``|EDT(gt) - EDT(pred)|`` in seconds; NaN if either measurement raises
      ``ValueError``/``IndexError`` (a silent or all-zero IR has no Schroeder decay);
    * ``c50``  -- ``|C50(gt) - C50(pred)|`` in dB; NaN when the difference is not finite
      (the exp_01 "C50 outlier" rule);
    * ``t60``  -- ``|T60(gt) - T60(pred)| / T60(gt) * 100`` as a percentage; NaN on
      ``ValueError``/``IndexError``/``ZeroDivisionError``, and NaN when ``want_t60`` is
      False.

    A NaN here means "this sample is invalid at this angle"; the paired analysis drops
    the pair and reports the invalid fraction separately, so nothing is ever silently
    imputed.  numpy's divide/invalid warnings are suppressed locally -- a degenerate IR
    is an expected outcome, not a bug.

    Args:
        pred_ir: predicted impulse response (1-D numpy array).
        gt_ir: ground-truth impulse response, same length.
        evaluator: an ``eval_unseen.Evaluator``.
        want_t60: compute T60 (it is the noisiest metric and descriptive only).

    Returns:
        ``{"edt": float, "c50": float, "t60": float}``, NaN where invalid.
    """
    nan = float("nan")
    edt = c50 = t60 = nan
    with np.errstate(divide="ignore", invalid="ignore"):
        try:
            edt = abs(evaluator.measure_edt(gt_ir) - evaluator.measure_edt(pred_ir))
        except (ValueError, IndexError):
            edt = nan
        clarity = abs(evaluator.measure_clarity(gt_ir) - evaluator.measure_clarity(pred_ir))
        c50 = float(clarity) if np.isfinite(clarity) else nan
        if want_t60:
            try:
                gt_t60 = evaluator.measure_rt60(gt_ir)
                t60 = abs(gt_t60 - evaluator.measure_rt60(pred_ir)) / gt_t60 * 100.0
            except (ValueError, IndexError, ZeroDivisionError):
                t60 = nan
    return {"edt": float(edt), "c50": float(c50), "t60": float(t60)}
