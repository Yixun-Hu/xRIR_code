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

import torch

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
