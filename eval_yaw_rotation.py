"""Yaw-rotation degradation sweep for one xRIR checkpoint (exp_03).

One pass over the pinned reference manifest (``ckpt/yaw_rotation/reference_manifest.json``)
evaluates, for every yaw angle ``k`` of the grid, two conditions of the *same* query:

* **P** (primary) -- the direct-path alignment is pinned at the ``k = 0`` coordinates
  (``tools.yaw_rotation.fixed_alignment``), so only the geometry branches of the forward
  pass see the rotation.  ``xRIR.shift_and_align`` is algebraically yaw-invariant but not
  numerically (float32 norms plus ``round()`` can flip a delay by one sample), and P
  isolates the model's geometric sensitivity from that quantisation noise.
* **E** (end-to-end) -- a plain forward on the rotated views, i.e. the alignment is
  recomputed from the rotated coordinates, which is what a deployed pipeline would do.

Both conditions collapse onto the plain forward at ``k = 0``, so every metric is a paired
comparison of a query against itself.  Per-sample metrics land in
``<out-dir>/per_sample_yaw.json`` and their aggregates in ``<out-dir>/metrics_yaw.json``;
``tools/summarize_yaw.py`` turns those into the confirmatory tables.

    python eval_yaw_rotation.py --backbone simple \\
        --checkpoint ckpt/xRIR_simple_8_shot/epoch_12.pth \\
        --manifest ckpt/yaw_rotation/reference_manifest.json --manifest-hash <sha256> \\
        --out-dir ckpt/yaw_rotation/simple_control

``model.xRIR.apply_delay`` hard-codes ``.cuda()``, so the whole evaluation runs on GPU.
"""
from __future__ import annotations

import numpy as np
import torch

from tools.per_sample_metrics import (
    acoustic_metrics,
    griffin_lim_seeded,
    per_sample_losses,
    sample_seed,
)
from tools.yaw_rotation import fixed_alignment, rotate_scene_yaw


def rotated_views(depth_coord, src_loc, ref_locs, k):
    """The scene under an active yaw of ``2*pi*k/W`` about the receiver's vertical axis.

    A thin wrapper around :func:`tools.yaw_rotation.rotate_scene_yaw` that takes the
    panorama width from the tensor itself, so a caller can never pass a ``W`` that
    disagrees with the depth map.

    Args:
        depth_coord: receiver-frame panorama coordinates ``[B, 3, H, W]``.
        src_loc: receiver-frame query-source position ``[B, 3]``.
        ref_locs: receiver-frame reference-source positions ``[B, K, 3]``.
        k: integer column roll (negative and ``>= W`` values are reduced modulo ``W``).

    Returns:
        ``(depth_coord', src_loc', ref_locs')``; at ``k = 0`` the inputs are returned
        unchanged value-for-value.
    """
    return rotate_scene_yaw(depth_coord, src_loc, ref_locs, k, W=depth_coord.shape[-1])


def forward_conditions(model, depth_coord, ref_irs, src_loc, ref_locs, tgt_wav, k, aligned0):
    """Both experimental conditions of one batch at one angle.

    Args:
        model: an ``xRIR`` (or ``xRIR_Cyl``) in ``eval()`` mode on CUDA.
        depth_coord: **unrotated** panorama coordinates ``[B, 3, H, W]``; the rotation is
            applied here so P and E cannot accidentally see different geometry.
        ref_irs: reference RIRs ``[B, K, L]`` (never rotated -- the acoustics are invariant).
        src_loc: unrotated query-source position ``[B, 3]``.
        ref_locs: unrotated reference-source positions ``[B, K, 3]``.
        tgt_wav: target RIR ``[B, 1, L]``.
        k: integer column roll.
        aligned0: the ``k = 0`` aligned reference audio, i.e.
            ``model.shift_and_align(ref_irs, src_loc, ref_locs)`` computed once per batch;
            condition P is forced to use it at every angle.

    Returns:
        ``(out_P, out_E, tgt_spec)`` -- two ``[B, F, T, 1]`` log-magnitude spectrograms and
        the shared ``[B, 1, F, T]`` target magnitude spectrogram.  At ``k = 0`` both
        outputs equal the plain forward exactly.
    """
    depth_k, src_k, ref_locs_k = rotated_views(depth_coord, src_loc, ref_locs, k)
    with torch.no_grad():
        with fixed_alignment(model, aligned0):
            out_p, tgt_spec = model(depth_k, ref_irs, src_k, ref_locs_k, tgt_wav)
        out_e, _ = model(depth_k, ref_irs, src_k, ref_locs_k, tgt_wav)
    return out_p, out_e, tgt_spec


def spectral_metrics(out_k, out_0, tgt_spec):
    """The five per-sample spectral metrics of one batch at one angle.

    Args:
        out_k: predicted log-magnitude spectrograms at the angle, ``[B, F, T, 1]``.
        out_0: the same batch's ``k = 0`` prediction, ``[B, F, T, 1]`` -- the paired
            reference for ``consistency``.
        tgt_spec: target magnitude spectrograms ``[B, 1, F, T]``.

    Returns:
        ``{"loss", "stft", "decay", "log_mse", "consistency"}``, each a detached CPU
        float32 tensor of shape ``[B]``:

        * ``loss``/``stft``/``decay`` -- ``tools.per_sample_metrics.per_sample_losses``,
          i.e. exp_01's test loss and its two halves, computed on 1-sample slices;
        * ``log_mse`` -- the per-sample form of exp_01's
          ``Evaluator.stft_loss(out_spec.squeeze(-1), log(tgt_spec + 1e-8).squeeze(1))``;
        * ``consistency`` -- ``mean |out_k - out_0|`` over the log-spectrogram, exactly
          ``0`` at ``k = 0``.
    """
    loss, stft, decay = per_sample_losses(out_k, tgt_spec)
    with torch.no_grad():
        log_tgt = torch.log(tgt_spec[:, 0] + 1e-8)
        log_mse = ((out_k[..., 0] - log_tgt) ** 2).flatten(1).mean(dim=1)
        consistency = (out_k - out_0).abs().flatten(1).mean(dim=1)
    return {"loss": loss, "stft": stft, "decay": decay,
            "log_mse": log_mse.detach().cpu().float(),
            "consistency": consistency.detach().cpu().float()}


def acoustic_metrics_batch(out_k, tgt_wav, query_keys, evaluator, gl_seed):
    """EDT / C50 / T60 errors of one batch, sample by sample.

    The magnitude spectrogram is rebuilt exactly as ``eval_xRIR_backbone.py`` does
    (``(exp(out) - 1e-8)[..., 0]``) and inverted with a Griffin-Lim whose random phase
    is seeded from ``(gl_seed, query key)``, so the same query inverts the same
    magnitudes to the same waveform at every angle, in every model and in any angle
    order.  The 8000-sample metric window is applied inside
    ``tools.per_sample_metrics.acoustic_metrics``.

    Args:
        out_k: predicted log-magnitude spectrograms ``[B, F, T, 1]`` (CPU or CUDA).
        tgt_wav: target RIRs ``[B, 1, L]``.
        query_keys: the ``B`` query paths, in batch order.
        evaluator: an ``eval_unseen.Evaluator``.
        gl_seed: run-level Griffin-Lim seed.

    Returns:
        ``{"edt", "c50", "t60"}``, each a float64 ``np.ndarray`` of shape ``[B]`` with
        NaN where the sample is invalid at this angle.

    Raises:
        ValueError: if ``query_keys`` does not have one key per row of ``out_k``.
    """
    if len(query_keys) != out_k.shape[0]:
        raise ValueError("got {} query keys for a batch of {}".format(
            len(query_keys), out_k.shape[0]))
    values = {"edt": [], "c50": [], "t60": []}
    for i in range(out_k.shape[0]):
        mag = (torch.exp(out_k[i:i + 1]) - 1e-8)[..., 0].cpu()
        wav = griffin_lim_seeded(mag, sample_seed(gl_seed, query_keys[i]))
        sample = acoustic_metrics(wav[0].numpy(), tgt_wav[i, 0].cpu().numpy(), evaluator)
        for name in values:
            values[name].append(sample[name])
    return {name: np.asarray(vals, dtype=np.float64) for name, vals in values.items()}
