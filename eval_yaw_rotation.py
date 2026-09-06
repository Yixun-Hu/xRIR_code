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
from tools.yaw_rotation import (
    fixed_alignment,
    integer_delays,
    rotate_scene_yaw,
    rotate_vectors_z,
    yaw_angle_rad,
)


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


def delay_flip_counts(src_loc, ref_locs, cols):
    """Count the integer direct-path delays a rotation moves, per angle.

    ``xRIR.shift_and_align`` is algebraically yaw-invariant, but its float32 norms and
    ``torch.round`` are not: a delay sitting on a rounding tie can move by one sample
    under an exactly norm-preserving rotation.  This audit quantifies that noise floor
    over the real geometry -- it is what condition P exists to exclude.

    Args:
        src_loc: query-source positions ``[B, 3]``.
        ref_locs: reference-source positions ``[B, K, 3]``.
        cols: iterable of integer column rolls.

    Returns:
        ``{k: count}`` -- the number of ``(sample, reference)`` pairs (out of ``B * K``)
        whose integer delay differs between the rotated and the unrotated coordinates.
        ``k = 0`` is always ``0``.
    """
    baseline = integer_delays(src_loc, ref_locs)
    counts = {}
    for k in cols:
        angle = yaw_angle_rad(k)
        rotated = integer_delays(rotate_vectors_z(src_loc, angle),
                                 rotate_vectors_z(ref_locs, angle))
        counts[int(k)] = int((rotated != baseline).sum())
    return counts


def _rel_change(rotated, base):
    """``mean|rotated - base| / mean|base|`` as a Python float."""
    denom = base.abs().mean()
    return float((rotated - base).abs().mean() / denom)


def decomposition_at(model, depth_coord, ref_irs, src_loc, ref_locs, tgt_wav, aligned0, k=32):
    """Where a patch-aligned yaw enters an ``xRIR_Cyl`` forward pass (diagnostic).

    CylindricalViT's tokens are equivariant to patch-aligned azimuth rolls, but ``xRIR``
    pools them with ``lin_proj_0`` -- a learned linear map over the 256 token *positions*
    -- and separately embeds the raw source xyz.  Neither is shift-invariant, so this
    reports the relative change of each stage for the receiver view, all under the
    pinned ``k = 0`` alignment (condition P):

    * ``tokens_rel_change`` -- ViT tokens after rolling the rotated ones back by
      ``k / patch_width`` azimuth patches (``~0`` iff the encoder is equivariant);
    * ``pooled_rel_change`` -- the pooled receiver feature ``lin_proj_0(src_proj(tokens))``,
      exactly as ``xRIR.forward`` computes ``receiver_out``;
    * ``coord_rel_change`` -- the sinusoidal source-coordinate embedding;
    * ``logspec_rel_change`` -- the final log-magnitude spectrogram.

    Args (beyond :func:`forward_conditions`'s):
        k: the patch-aligned angle to decompose (the plan's diagnostic angle is 32).

    Returns:
        ``{"k", "tokens_rel_change", "pooled_rel_change", "coord_rel_change",
        "logspec_rel_change"}`` -- one int and four floats.

    Raises:
        TypeError: if ``model.source_network`` is not tokenised on an explicit
            elevation x azimuth grid (i.e. is not a ``CylindricalViT``), so "roll the
            tokens back" would be undefined.
    """
    network = model.source_network
    if not (hasattr(network, "h_tok") and hasattr(network, "w_tok")):
        raise TypeError("decomposition_at needs a CylindricalViT source_network (an "
                        "explicit elevation x azimuth token grid), got {}".format(
                            type(network).__name__))
    h_tok, w_tok = int(network.h_tok), int(network.w_tok)
    width = depth_coord.shape[-1]
    shift = (int(k) % width) // (width // w_tok)

    depth_k, src_k, _ = rotated_views(depth_coord, src_loc, ref_locs, k)
    batch = depth_coord.shape[0]
    with torch.no_grad():
        tokens_0 = network((-depth_coord) / 5.)
        tokens_k = network((-depth_k) / 5.)
        grid_0 = tokens_0.view(batch, h_tok, w_tok, -1)
        grid_k = torch.roll(tokens_k.view(batch, h_tok, w_tok, -1), shifts=-shift, dims=2)

        pooled_0 = model.lin_proj_0(model.src_proj(tokens_0).permute(0, 2, 1))
        pooled_k = model.lin_proj_0(model.src_proj(tokens_k).permute(0, 2, 1))

        coord_0 = model.src_coord_proj(
            model.dist_embedder(src_loc.unsqueeze(1) / 5.).view(batch, -1))
        coord_k = model.src_coord_proj(
            model.dist_embedder(src_k.unsqueeze(1) / 5.).view(batch, -1))

    out_0 = forward_conditions(
        model, depth_coord, ref_irs, src_loc, ref_locs, tgt_wav, 0, aligned0)[0]
    out_k = forward_conditions(
        model, depth_coord, ref_irs, src_loc, ref_locs, tgt_wav, k, aligned0)[0]
    return {"k": int(k),
            "tokens_rel_change": _rel_change(grid_k, grid_0),
            "pooled_rel_change": _rel_change(pooled_k, pooled_0),
            "coord_rel_change": _rel_change(coord_k, coord_0),
            "logspec_rel_change": _rel_change(out_k, out_0)}
