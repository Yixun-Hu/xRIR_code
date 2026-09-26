"""Yaw-rotation *pilot* evaluator for one xRIR checkpoint (exp_10 ``yaw_pilot``).

The xRIR analogue of FLAC's exp_02 pilot.  exp_03 already asked whether a yaw rotation
of the conditioning **degrades accuracy** (it barely does); this pilot asks the other
half of the question: how far does the **prediction itself move**?  Every query is
therefore measured twice at every angle:

* **Metric 2** -- accuracy against the ground-truth RIR (``edt_err`` / ``c50_err`` /
  ``t60_err``, ``log_mse``, ``loss`` / ``stft`` / ``decay``), i.e. exp_03's numbers;
* **Metric 1** -- the *shift* of the prediction against its own ``k = 0`` prediction, with
  no ground truth involved (``wave_rel_l2`` / ``wave_mad`` on the inverted waveform,
  ``mag_rel_l2`` / ``logspec_mad`` on the model's direct output, and the acoustic gaps
  ``edt_gap`` / ``c50_gap`` / ``t60_gap`` under FLAC's "P_0 as GT" convention).

Only the **P** condition of exp_03 is run (the direct-path alignment is pinned at the
``k = 0`` coordinates via ``tools.yaw_rotation.fixed_alignment``), so only the geometry
branches of the forward pass see the rotation.

Nothing in exp_03's pinned closure is modified: the rotation, the alignment pin, the
Griffin-Lim seeding, the acoustic validity rules and the canonical batch padding are all
imported from it.  The one thing this module adds to the forward path is a device
agnostic ``apply_delay`` (``model.xRIR.apply_delay`` hard-codes ``.cuda()``), installed
by :func:`patched_apply_delay` for **every** device so the CPU and the GPU take one code
path.

    python tools/exp10_yaw_pilot.py --arm released_k8 --backbone simple \\
        --checkpoint checkpoints/xRIR_unseen.pth \\
        --manifest ckpt/yaw_rotation/reference_manifest.json --manifest-hash <sha256> \\
        --num-shot 8 --ks 0,64,128,256,384 --batches probe --controls \\
        --device cuda --out-dir ckpt/exp10/released_k8
"""
from __future__ import annotations

import numpy as np
import torch

MAG_EPS = 1e-8
REL_L2_EPS = 1e-8


def _as_waveforms(array, name):
    """Validate one ``[n, T]`` waveform block and return it as float64."""
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError("{} must be a [n, T] waveform block, got shape {}".format(
            name, values.shape))
    return values.astype(np.float64)


def waveform_gap(w0, wk):
    """FLAC's waveform distance between a rotated prediction and its ``k = 0`` twin.

    Both definitions are FLAC's ``compare_predictions.waveform_gap``:
    ``wave_rel_l2 = ||wk - w0||_2 / (||w0||_2 + 1e-8)`` and ``wave_mad = mean|wk - w0|``,
    computed over the **stored** waveform (9600 samples, the last 21 of them the zero
    padding of the 9579-sample Griffin-Lim output).  The epsilon keeps a silent
    reference finite instead of producing an infinite "gap".

    Args:
        w0: the ``k = 0`` waveforms ``[n, T]`` (the paired reference).
        wk: the rotated-angle waveforms ``[n, T]``.

    Returns:
        ``{"wave_rel_l2": [n], "wave_mad": [n]}`` as float64 ``np.ndarray``s.

    Raises:
        ValueError: if either block is not 2-D or the two shapes differ.
    """
    base = _as_waveforms(w0, "w0")
    rotated = _as_waveforms(wk, "wk")
    if base.shape != rotated.shape:
        raise ValueError("w0 {} and wk {} must have the same shape".format(
            base.shape, rotated.shape))
    diff = rotated - base
    rel_l2 = np.linalg.norm(diff, axis=1) / (np.linalg.norm(base, axis=1) + REL_L2_EPS)
    return {"wave_rel_l2": rel_l2, "wave_mad": np.abs(diff).mean(axis=1)}


def _as_logspec(tensor, name):
    """Validate one ``[n, F, T]`` / ``[n, F, T, 1]`` log-magnitude block; return ``[n, F*T]``."""
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.dim() == 4 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    if tensor.dim() != 3:
        raise ValueError("{} must be [n, F, T] or [n, F, T, 1], got shape {}".format(
            name, tuple(tensor.shape)))
    return tensor.detach().double().flatten(1)


def spectrogram_gap(out0, outk):
    """The Griffin-Lim-free shift of the prediction, on the model's direct output.

    ``logspec_mad = mean|out_k - out_0|`` over ``[F, T]`` -- exactly exp_03's
    ``consistency`` -- and ``mag_rel_l2 = ||M_k - M_0||_F / ||M_0||_F`` on the magnitude
    ``M = exp(out) - 1e-8`` that the waveform is inverted from.  Both are computed in
    float64 on the tensors' own device and returned on the CPU; unlike the waveform and
    acoustic gaps they cannot be recomputed from the stored waveforms (Griffin-Lim is a
    finite-iteration nonlinear inverse), so they are online-only quantities.

    Args:
        out0: the ``k = 0`` log-magnitude output ``[n, F, T]`` or ``[n, F, T, 1]``.
        outk: the rotated-angle output, same shape.

    Returns:
        ``{"logspec_mad": [n], "mag_rel_l2": [n]}`` as float64 ``np.ndarray``s.

    Raises:
        ValueError: if either block has the wrong rank or the two shapes differ.
    """
    base = _as_logspec(out0, "out0")
    rotated = _as_logspec(outk, "outk")
    if base.shape != rotated.shape:
        raise ValueError("out0 {} and outk {} must have the same shape".format(
            tuple(base.shape), tuple(rotated.shape)))
    with torch.no_grad():
        mad = (rotated - base).abs().mean(dim=1)
        mag_0 = torch.exp(base) - MAG_EPS
        mag_k = torch.exp(rotated) - MAG_EPS
        rel_l2 = torch.linalg.norm(mag_k - mag_0, dim=1) / torch.linalg.norm(mag_0, dim=1)
    return {"logspec_mad": mad.cpu().numpy(), "mag_rel_l2": rel_l2.cpu().numpy()}
