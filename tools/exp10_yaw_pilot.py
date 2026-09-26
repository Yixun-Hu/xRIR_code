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

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid

import numpy as np
import torch

import model.xRIR as model_xrir
from eval_unseen import Evaluator
from eval_xRIR_backbone import load_model_state
from eval_yaw_rotation import build_manifest_dataset, load_checked_manifest, pad_batch
from model.xRIR_cyl import BACKBONES, build_xrir
from tools.yaw_rotation import fixed_alignment, rotate_scene_yaw

from tools.per_sample_metrics import (
    acoustic_metrics,
    griffin_lim_seeded,
    per_sample_losses,
    sample_seed,
)

MAG_EPS = 1e-8
REL_L2_EPS = 1e-8
METRIC_WINDOW = 8000          # exp_01's acoustic window, enforced by acoustic_metrics
SAMPLE_RATE = 22050
NATIVE_LEN = 9579             # (310 - 1) * hop 31: what Griffin-Lim returns
PADDED_LEN = 9600             # exp_01's max_len, the length the waveforms are stored at
DEFAULT_BATCH_SIZE = 16       # exp_03's canonical compute shape
DEFAULT_KS = (0, 64, 128, 256, 384)

# Metric 2 -- accuracy against the ground-truth RIR (exp_03's numbers).
METRIC2_NAMES = ("edt_err", "c50_err", "t60_err", "log_mse", "loss", "stft", "decay")
# The raw measurements behind Metric 2, so any other normalisation is recoverable.
RAW_PRED_NAMES = ("edt_pred", "c50_pred", "t60_pred")
GT_NAMES = ("edt_gt", "c50_gt", "t60_gt")
# Metric 1 -- how far the prediction moved from its own k = 0 prediction.
METRIC1_NAMES = ("wave_rel_l2", "wave_mad", "mag_rel_l2", "logspec_mad",
                 "edt_gap", "c50_gap", "t60_gap")
# Same-unit supplements (seconds) for the two percentage-normalised T60 quantities.
SUPPLEMENT_NAMES = ("t60_gap_abs", "t60_err_abs")
ANGLE_METRIC_NAMES = METRIC2_NAMES + RAW_PRED_NAMES + METRIC1_NAMES + SUPPLEMENT_NAMES
# What a control additionally records: its deviation from the cell it repeats.
CONTROL_EXTRA_NAMES = ("wave_max_abs_diff", "logspec_max_abs_diff")
# (control id, angle to run, angle it must reproduce).
CONTROL_SPECS = (("ctrl_zero_repeat", 0, 0),
                 ("ctrl_k128_repeat", 128, 128),
                 ("ctrl_full_turn", 512, 0))


def _as_block(array, name):
    """Validate one ``[n, T]`` waveform block, leaving its dtype alone.

    The evaluator is sensitive to the dtype it is handed (a float32 and a float64 C50 of
    the same waveform differ at ~1e-6 dB), and the frozen
    ``tools.per_sample_metrics.acoustic_metrics`` measures whatever it is given.  Every
    function that feeds the evaluator therefore passes the **stored float32** samples
    through unchanged, so the canonical errors, the gaps and the raw measures are all
    measurements of one and the same array.
    """
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError("{} must be a [n, T] waveform block, got shape {}".format(
            name, values.shape))
    return values


def _as_waveforms(array, name):
    """Validate one ``[n, T]`` waveform block and return it as float64.

    Used only by :func:`waveform_gap`, whose norms and means are pure accumulations that
    never reach the evaluator: float64 keeps them reproducible across devices.
    """
    return _as_block(array, name).astype(np.float64)


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


def acoustic_gap(w0, wk, evaluator, want_t60=True, window=METRIC_WINDOW):
    """EDT / C50 / T60 **shift** of a rotated prediction against its ``k = 0`` twin.

    FLAC's "P_0 as ground truth" convention: the unrotated prediction takes the place of
    the reference IR, so the numbers say how far the prediction moved, never how wrong it
    is.  The measurement itself is exp_03's frozen
    ``tools.per_sample_metrics.acoustic_metrics`` -- same 8000-sample window, same
    validity rules (NaN, never imputed), same exception set.

    Args:
        w0: the ``k = 0`` waveforms ``[n, T]`` (used as ``gt``).
        wk: the rotated-angle waveforms ``[n, T]`` (used as ``pred``).
        evaluator: an ``eval_unseen.Evaluator``.
        want_t60: compute the T60 gap (NaN for every row when False).
        window: leading samples measured; exp_01's 8000.

    Returns:
        ``{"edt_gap": [n], "c50_gap": [n], "t60_gap": [n]}`` as float64 ``np.ndarray``s,
        NaN where the pair is invalid.  ``t60_gap`` is a percentage of ``T60(w0)``.

    Raises:
        ValueError: if either block is not 2-D or the two shapes differ.
    """
    base = _as_block(w0, "w0")
    rotated = _as_block(wk, "wk")
    if base.shape != rotated.shape:
        raise ValueError("w0 {} and wk {} must have the same shape".format(
            base.shape, rotated.shape))
    values = {"edt_gap": [], "c50_gap": [], "t60_gap": []}
    for i in range(base.shape[0]):
        cell = acoustic_metrics(rotated[i], base[i], evaluator, want_t60=want_t60,
                                window=window)
        values["edt_gap"].append(cell["edt"])
        values["c50_gap"].append(cell["c50"])
        values["t60_gap"].append(cell["t60"])
    return {name: np.asarray(vals, dtype=np.float64) for name, vals in values.items()}


def raw_measures(waves, evaluator, want_t60=True, window=METRIC_WINDOW):
    """The **raw** EDT / C50 / T60 of each waveform, under ``acoustic_metrics``' rules.

    The canonical metrics are differences, which fixes their normalisation forever; the
    raw measurements are stored beside them so any other normalisation (absolute seconds,
    a different reference) can be rebuilt offline without re-running the model.  The
    window, the exception set (``ValueError`` / ``IndexError`` for the Schroeder-decay
    measurements, plus ``ZeroDivisionError`` for T60) and the "non-finite becomes NaN"
    rule are exactly ``tools.per_sample_metrics.acoustic_metrics``'.

    Args:
        waves: waveforms ``[n, T]``.
        evaluator: an ``eval_unseen.Evaluator``.
        want_t60: measure T60 (NaN for every row when False).
        window: leading samples measured; exp_01's 8000.

    Returns:
        ``{"edt": [n] (s), "c50": [n] (dB), "t60": [n] (s)}`` as float64 ``np.ndarray``s,
        NaN where the measurement is invalid.

    Raises:
        ValueError: if ``waves`` is not a ``[n, T]`` block.
    """
    block = _as_block(waves, "waves")[:, :int(window)]
    nan = float("nan")
    values = {"edt": [], "c50": [], "t60": []}
    with np.errstate(divide="ignore", invalid="ignore"):
        for row in block:
            try:
                edt = evaluator.measure_edt(row)
            except (ValueError, IndexError):
                edt = nan
            clarity = evaluator.measure_clarity(row)
            t60 = nan
            if want_t60:
                try:
                    t60 = evaluator.measure_rt60(row)
                except (ValueError, IndexError, ZeroDivisionError):
                    t60 = nan
            for name, value in (("edt", edt), ("c50", clarity), ("t60", t60)):
                values[name].append(float(value) if np.isfinite(value) else nan)
    return {name: np.asarray(vals, dtype=np.float64) for name, vals in values.items()}


def device_agnostic_apply_delay(signal, delay_tensor):
    """``model.xRIR.apply_delay`` without its hard-coded ``.cuda()``.

    The shifting semantics are the model's, line for line -- positive delay shifts right
    and zero-pads the front, negative shifts left and zero-pads the tail, zero copies --
    but the output buffer is allocated on the **signal's own device**, so the same
    forward pass runs on the CPU and on the GPU.  On CUDA the two implementations are
    identical (``torch.zeros_like(x).cuda()`` is a no-op for a CUDA tensor).

    Args:
        signal: ``[batch, sequence_length]`` (or ``[batch, sequence_length, channels]``).
        delay_tensor: ``[batch]`` integer sample delays.

    Returns:
        A new tensor of the signal's shape, dtype and device.
    """
    delayed_signal = torch.zeros_like(signal)
    for i in range(signal.shape[0]):
        delay = delay_tensor[i].item()
        if delay > 0:
            delayed_signal[i, delay:] = signal[i, :-delay]
        elif delay < 0:
            delayed_signal[i, :delay] = signal[i, -delay:]
        else:
            delayed_signal[i] = signal[i]
    return delayed_signal


@contextlib.contextmanager
def patched_apply_delay():
    """Run the block with :func:`device_agnostic_apply_delay` installed in ``model.xRIR``.

    ``xRIR.shift_and_align`` calls the *module-level* ``apply_delay``, so swapping the
    module attribute redirects every model instance for the duration of the block and
    nothing else about the model is touched.  The original is restored in a ``finally``,
    including when the block raises.

    The patch is installed for **every** device (not only the CPU) so that a GPU run and
    a CPU run of this tool execute the same code; it mutates a module attribute, so it is
    for serial, eager evaluation only and is not thread-safe.

    Yields:
        The replacement function that is installed.
    """
    original = model_xrir.apply_delay
    model_xrir.apply_delay = device_agnostic_apply_delay
    try:
        yield device_agnostic_apply_delay
    finally:
        model_xrir.apply_delay = original


def _sync(device):
    """Wait for the device's queued work, so a stage's elapsed time is its own.

    CUDA launches are asynchronous: ``time.time()`` after a forward pass measures when the
    kernels were *queued*, and the first ``.cpu()`` of the next stage then pays for them.
    On the CPU there is nothing to wait for and nothing is called.
    """
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@contextlib.contextmanager
def _stage(timing, name, device):
    """Time one stage of the pipeline, synchronising at both of its boundaries.

    Args:
        timing: the accumulator dict; ``timing[name]`` is incremented.
        name: which stage (``"inference_s"`` / ``"inversion_s"`` / ``"metrics_s"`` /
            ``"writing_s"``).
        device: the run's device; only ``"cuda"`` needs the synchronisation.
    """
    _sync(device)
    clock = time.time()
    try:
        yield
    finally:
        _sync(device)
        timing[name] += time.time() - clock


def angle_logspec(model, depth, refs, src, ref_locs, tgt, k, aligned0):
    """One batch's prediction at one yaw angle, under condition **P**.

    The composition is exp_03's, unchanged: rotate the whole scene by ``k`` panorama
    columns (``tools.yaw_rotation.rotate_scene_yaw`` -- panorama roll plus ``Rz`` on the
    source and reference coordinates), then run the forward pass with the direct-path
    alignment **pinned** at the ``k = 0`` value (``tools.yaw_rotation.fixed_alignment``),
    so only the geometry branches of the model see the rotation and the float32
    quantisation of ``shift_and_align`` cannot leak into the comparison.  The reference
    audio and the target are never rotated: the acoustics are yaw-invariant.

    Args:
        model: an ``xRIR`` (or ``xRIR_Cyl``) in ``eval()`` mode on the run's device.
        depth: **unrotated** receiver-frame panorama coordinates ``[B, 3, H, W]``.
        refs: reference RIRs ``[B, K, L]``.
        src: unrotated query-source position ``[B, 3]``.
        ref_locs: unrotated reference-source positions ``[B, K, 3]``.
        tgt: target RIR ``[B, 1, L]``.
        k: integer column roll; any integer is accepted (``512`` reduces to ``0`` inside
            ``rotate_scene_yaw``, which is what the ``ctrl_full_turn`` control uses).
        aligned0: the ``k = 0`` aligned reference audio, computed once per batch as
            ``model.shift_and_align(refs, src, ref_locs)``.

    Returns:
        ``(out_k, tgt_spec)``: the ``[B, F, T, 1]`` log-magnitude prediction (the model's
        native layout, so it feeds ``per_sample_losses`` and exp_03's spectral metrics
        unchanged) and the ``[B, 1, F, T]`` target magnitude spectrogram.

    Note:
        ``model.xRIR.apply_delay`` is ``.cuda()``-only, so the caller must hold
        :func:`patched_apply_delay` (``run`` does, for every device).
    """
    depth_k, src_k, ref_locs_k = rotate_scene_yaw(depth, src, ref_locs, int(k),
                                                  W=depth.shape[-1])
    with torch.no_grad():
        with fixed_alignment(model, aligned0):
            out_k, tgt_spec = model(depth_k, refs, src_k, ref_locs_k, tgt)
    return out_k, tgt_spec


def invert(out, keys, gl_seed):
    """Griffin-Lim one batch's log-magnitude prediction into stored waveforms.

    The magnitude ``M = exp(out) - 1e-8`` is built **on the output's own device** and then
    moved to the CPU -- the op placement of exp_03's ``acoustic_metrics_batch``, kept so a
    GPU run and this tool produce the same numbers -- and inverted by
    ``tools.per_sample_metrics.griffin_lim_seeded`` with the per-query phase seed
    ``sample_seed(gl_seed, key)``, so the same query inverts the same magnitudes to the
    same waveform at every angle, in every model and in any order.

    Griffin-Lim returns 9579 samples; they are zero-padded to the stored 9600 and
    returned as float32 **exactly as they are** -- no clipping, no normalisation -- so the
    metrics, the saved ``wav_k<k>.npy`` and any offline recomputation all see one array.

    Args:
        out: log-magnitude predictions ``[n, F, T, 1]`` (the model's native layout).
        keys: the ``n`` query paths, in batch order.
        gl_seed: run-level Griffin-Lim seed.

    Returns:
        ``[n, 9600]`` float32 ``np.ndarray``.

    Raises:
        ValueError: if ``out`` is not ``[n, F, T, 1]``, if ``keys`` does not have one key
            per row, or if an inversion is longer than the stored length.
    """
    if out.dim() != 4 or out.shape[-1] != 1:
        raise ValueError("out must be [n, F, T, 1] (the model's native layout), got {}"
                         .format(tuple(out.shape)))
    if len(keys) != out.shape[0]:
        raise ValueError("got {} query keys for a batch of {}".format(
            len(keys), out.shape[0]))
    waves = np.zeros((out.shape[0], PADDED_LEN), dtype=np.float32)
    with torch.no_grad():
        for i, key in enumerate(keys):
            mag = (torch.exp(out[i:i + 1]) - MAG_EPS)[..., 0].cpu()
            samples = griffin_lim_seeded(mag, sample_seed(gl_seed, key))[0].numpy()
            if samples.shape[0] > PADDED_LEN:
                raise ValueError("Griffin-Lim returned {} samples, more than the stored "
                                 "length {}".format(samples.shape[0], PADDED_LEN))
            waves[i, :samples.shape[0]] = samples
    return waves


def _log_mse(out_k, tgt_spec):
    """exp_03's ``log_mse``: per-query mean of ``(out - log(tgt_spec + 1e-8))^2``.

    Literally ``eval_yaw_rotation.spectral_metrics``' line (``per_sample_losses`` returns
    only loss / stft / decay), kept here so the pilot's Metric 2 is the same number
    exp_03 recorded.
    """
    with torch.no_grad():
        log_tgt = torch.log(tgt_spec[:, 0] + MAG_EPS)
        return ((out_k[..., 0] - log_tgt) ** 2).flatten(1).mean(dim=1).detach().cpu()


def _angle_cell(out_k, out_0, tgt_spec, waves_k, waves_0, gt_waves, raw_0, raw_gt,
                evaluator, want_t60):
    """Every per-query metric of one angle (or one control), as float64 ``[n]`` arrays."""
    cell = {}

    # --- Metric 2: accuracy against the ground truth -------------------------------
    errors = [acoustic_metrics(waves_k[i], gt_waves[i], evaluator, want_t60=want_t60,
                               window=METRIC_WINDOW) for i in range(waves_k.shape[0])]
    for name, key in (("edt_err", "edt"), ("c50_err", "c50"), ("t60_err", "t60")):
        cell[name] = np.asarray([e[key] for e in errors], dtype=np.float64)
    loss, stft, decay = per_sample_losses(out_k, tgt_spec)
    cell["loss"] = loss.double().numpy()
    cell["stft"] = stft.double().numpy()
    cell["decay"] = decay.double().numpy()
    cell["log_mse"] = _log_mse(out_k, tgt_spec).double().numpy()

    # --- the raw measurements behind those errors -----------------------------------
    raw_k = raw_measures(waves_k, evaluator, want_t60=want_t60, window=METRIC_WINDOW)
    for name, key in (("edt_pred", "edt"), ("c50_pred", "c50"), ("t60_pred", "t60")):
        cell[name] = raw_k[key]

    # --- Metric 1: the shift of the prediction itself -------------------------------
    cell.update(waveform_gap(waves_0, waves_k))
    cell.update(spectrogram_gap(out_0, out_k))
    cell.update(acoustic_gap(waves_0, waves_k, evaluator, want_t60=want_t60,
                             window=METRIC_WINDOW))

    # --- same-unit T60 supplements (seconds), from the raw measurements -------------
    cell["t60_gap_abs"] = np.abs(raw_k["t60"] - raw_0["t60"])
    cell["t60_err_abs"] = np.abs(raw_k["t60"] - raw_gt["t60"])
    return cell


def evaluate_batch(model, batch, ks, evaluator, gl_seed=0, batch_size=DEFAULT_BATCH_SIZE,
                   device="cpu", controls=False, want_t60=True):
    """Every per-query metric of one collated batch, at every angle in ``ks``.

    The batch is padded to the canonical compute shape (``eval_yaw_rotation.pad_batch``),
    moved to ``device`` once, and the ``k = 0`` alignment and prediction are computed
    **once**: every angle reuses them, so all angles share one paired reference and the
    result does not depend on the order of ``ks``.  The padded rows are dropped before any
    metric is computed.

    Controls (``--controls``) are **fresh** inferences under their own ids -- they
    recompute the alignment and the forward pass and are stored separately, never
    overwriting the angle they repeat: ``ctrl_zero_repeat`` (k = 0 again),
    ``ctrl_k128_repeat`` (a nonzero angle again) and ``ctrl_full_turn`` (k = 512, the
    identity path through ``rotate_scene_yaw``).  Each carries, on top of the usual
    metrics, its per-query ``wave_max_abs_diff`` / ``logspec_max_abs_diff`` from the cell
    it must reproduce.

    Args:
        model: an ``xRIR`` in ``eval()`` mode, already on ``device``.
        batch: the ``ManifestDataset`` seven-tuple, collated.
        ks: integer column rolls to evaluate; ``0`` must be among them (it is the paired
            reference of every Metric-1 quantity).
        evaluator: an ``eval_unseen.Evaluator``.
        gl_seed: run-level Griffin-Lim seed.
        batch_size: canonical compute shape, or ``None`` to leave the batch alone.
        device: where the forward passes run (``"cpu"`` or ``"cuda"``).
        controls: also run the three plumbing controls.
        want_t60: measure T60 (descriptive, and the noisiest metric).

    Returns:
        ``{"keys", "n", "gt", "angles", "controls", "waveforms", "timing"}``: the ``n``
        kept query keys, the ground-truth raw measures, ``angles[k][metric] -> [n]``
        float64, ``controls[id][metric] -> [n]`` (plus the two deviation arrays),
        ``waveforms[k] -> [n, 9600]`` float32 (the very arrays the metrics were computed
        from) and the batch's inference / inversion / metric seconds -- each stage
        synchronised at both boundaries on CUDA, so the breakdown is not an artefact of
        asynchronous kernel launches.

    Raises:
        ValueError: if ``0`` is missing from ``ks``, ``ks`` repeats an angle, or a control
            has no reference angle in ``ks``.
    """
    cols = [int(k) for k in ks]
    if len(set(cols)) != len(cols):
        raise ValueError("ks repeats an angle: {}".format(cols))
    if 0 not in cols:
        raise ValueError("k = 0 is the paired reference of every Metric-1 quantity; "
                         "--ks must contain it, got {}".format(cols))
    if controls:
        missing = sorted({spec[2] for spec in CONTROL_SPECS} - set(cols))
        if missing:
            raise ValueError("the controls repeat angles {} which are not in ks {}".format(
                missing, cols))

    batch, n_real = pad_batch(batch, batch_size)
    _, src_loc, depth_coord, tgt_wav, ref_irs, ref_locs, keys = batch
    src_loc, depth_coord = src_loc.to(device), depth_coord.to(device)
    tgt_wav, ref_irs = tgt_wav.to(device), ref_irs.to(device)
    ref_locs = ref_locs.to(device)
    keys = list(keys)[:n_real]

    timing = {"inference_s": 0.0, "inversion_s": 0.0, "metrics_s": 0.0}
    with patched_apply_delay():
        with _stage(timing, "inference_s", device):
            with torch.no_grad():
                aligned0 = model.shift_and_align(ref_irs, src_loc, ref_locs)
                out_0, _ = model(depth_coord, ref_irs, src_loc, ref_locs, tgt_wav)
            out_0 = out_0[:n_real]

        with _stage(timing, "inversion_s", device):
            waves_0 = invert(out_0, keys, gl_seed)

        with _stage(timing, "metrics_s", device):
            gt_waves = tgt_wav[:n_real, 0].cpu().numpy()
            raw_0 = raw_measures(waves_0, evaluator, want_t60=want_t60,
                                 window=METRIC_WINDOW)
            raw_gt = raw_measures(gt_waves, evaluator, want_t60=want_t60,
                                  window=METRIC_WINDOW)

        angles, waveforms, outs = {}, {}, {}
        for k in cols:
            with _stage(timing, "inference_s", device):
                out_k, tgt_spec = angle_logspec(model, depth_coord, ref_irs, src_loc,
                                                ref_locs, tgt_wav, k, aligned0)
                out_k, tgt_spec = out_k[:n_real], tgt_spec[:n_real]

            with _stage(timing, "inversion_s", device):
                waves_k = invert(out_k, keys, gl_seed)

            with _stage(timing, "metrics_s", device):
                angles[k] = _angle_cell(out_k, out_0, tgt_spec, waves_k, waves_0,
                                        gt_waves, raw_0, raw_gt, evaluator, want_t60)
            waveforms[k], outs[k] = waves_k, out_k

        control_cells = {}
        for name, k, reference in (CONTROL_SPECS if controls else ()):
            with _stage(timing, "inference_s", device):
                with torch.no_grad():
                    aligned_fresh = model.shift_and_align(ref_irs, src_loc, ref_locs)
                out_c, tgt_spec = angle_logspec(model, depth_coord, ref_irs, src_loc,
                                                ref_locs, tgt_wav, k, aligned_fresh)
                out_c, tgt_spec = out_c[:n_real], tgt_spec[:n_real]

            with _stage(timing, "inversion_s", device):
                waves_c = invert(out_c, keys, gl_seed)

            with _stage(timing, "metrics_s", device):
                cell = _angle_cell(out_c, out_0, tgt_spec, waves_c, waves_0, gt_waves,
                                   raw_0, raw_gt, evaluator, want_t60)
                cell["wave_max_abs_diff"] = np.abs(
                    waves_c.astype(np.float64) - waveforms[reference].astype(np.float64)
                ).max(axis=1)
                cell["logspec_max_abs_diff"] = (out_c - outs[reference]).abs().flatten(
                    1).max(dim=1).values.double().cpu().numpy()
            control_cells[name] = cell

    gt_block = {"edt_gt": raw_gt["edt"], "c50_gt": raw_gt["c50"], "t60_gt": raw_gt["t60"]}
    return {"keys": keys, "n": int(n_real), "gt": gt_block, "angles": angles,
            "controls": control_cells, "waveforms": waveforms, "timing": timing}


def _canonical_queries(dataset):
    """The dataset's query paths in canonical (manifest) order."""
    entries = getattr(dataset, "entries", dataset)
    return [entry["query"] if isinstance(entry, dict) else str(entry) for entry in entries]


def select_probe_batches(dataset, batch_size=DEFAULT_BATCH_SIZE, per_room=1):
    """The probe: ``per_room`` canonical batches lying entirely inside each room.

    The probe has to be a *subset of the full run*, not a re-batching of it, or its
    per-sample values would be computed at a different batch composition and could not be
    compared with the full results (nor with exp_03's).  Each returned index ``b`` is
    therefore a batch of the full run -- queries ``[b * batch_size, (b + 1) * batch_size)``
    in canonical order -- chosen so that the whole window sits inside one room, which also
    makes the probe cover every room.

    Args:
        dataset: anything exposing the canonical ``entries`` (a ``ManifestDataset``), or
            the entry list itself.
        batch_size: the run's canonical batch size.
        per_room: how many intact batches to take from each room.

    Returns:
        A sorted list of batch indices (``per_room`` per room, 17 for exp_03's manifest).

    Raises:
        ValueError: if ``batch_size`` or ``per_room`` is not positive, if a room's queries
            are not contiguous in canonical order, or if a room has fewer than
            ``per_room`` intact batches.
    """
    size, wanted = int(batch_size), int(per_room)
    if size < 1 or wanted < 1:
        raise ValueError("batch_size and per_room must be positive, got {} and {}".format(
            batch_size, per_room))
    queries = _canonical_queries(dataset)
    rooms = [os.path.dirname(query) for query in queries]

    bounds = {}
    for position, room in enumerate(rooms):
        first, last = bounds.get(room, (position, position))
        bounds[room] = (min(first, position), max(last, position))

    picked = []
    for room, (first, last) in sorted(bounds.items(), key=lambda item: item[1]):
        if any(rooms[i] != room for i in range(first, last + 1)):
            raise ValueError("room {} is not contiguous in canonical order (queries {} "
                             "to {})".format(room, first, last))
        start = -(-first // size)                       # first batch starting at or after
        inside = [b for b in range(start, (last + 1) // size) if b * size >= first]
        if len(inside) < wanted:
            raise ValueError("room {} ({} queries from {}) has {} intact batches of {}, "
                             "fewer than the {} the probe needs".format(
                                 room, last - first + 1, first, len(inside), size, wanted))
        picked.extend(inside[:wanted])
    return sorted(picked)


def file_sha256(path, chunk=1 << 20):
    """sha256 of a file's bytes, so a reader can re-verify it from disk alone."""
    digest = hashlib.sha256()
    with open(path, "rb") as fin:
        for block in iter(lambda: fin.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def query_list_sha256(queries):
    """sha256 of the newline-joined query list -- pins both the set and its order."""
    return hashlib.sha256("\n".join(queries).encode()).hexdigest()


def protocol_id(fields):
    """The run's **protocol** identity (plan amendment A2).

    A sha256 over everything that decides what the numbers mean -- checkpoint, manifest,
    Griffin-Lim seed, K, device, precision flags, batch shape, library versions and the
    sha256 of this tool.  Two runs with the same ``protocol_id`` are two executions of the
    same protocol (which the summariser refuses to combine); a different id means the
    numbers are not interchangeable.

    Args:
        fields: the protocol fields; key order is irrelevant.

    Returns:
        The 64-character hex digest.
    """
    return hashlib.sha256(json.dumps(fields, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def new_execution_id():
    """A fresh **execution** identity: one uuid4, minted once per ``run`` (A2).

    Returns:
        ``(execution_id, started_at)`` -- ``"<ISO-8601 UTC>-<uuid4 hex>"`` and the
        timestamp on its own, so a per-sample file can be tied to exactly the execution
        that wrote it (not merely to a protocol that could have been run twice).
    """
    started = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    return "{}-{}".format(started, uuid.uuid4().hex), started


def json_values(values):
    """One per-sample array as strict JSON: every non-finite value becomes ``null``.

    exp_03's rule, kept identical: NaN marks a sample the angle invalidated and has to
    survive the round trip, but the JSON ``NaN`` literal is not valid JSON, so the files
    are written with ``allow_nan=False`` and invalid samples are ``null``.
    """
    return [None if not np.isfinite(value) else float(value) for value in values]


def write_json(payload, path):
    """Write ``payload`` as strict JSON atomically: the full file or no file.

    Raises:
        ValueError: if the payload still contains a non-finite float (run the arrays
            through :func:`json_values` first).
    """
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as fout:
            json.dump(payload, fout, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _git_commit(repo_root=None):
    """``HEAD`` of the tree this module lives in, plus whether it is dirty."""
    root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(["git", "-C", root, "status", "--porcelain"],
                                         stderr=subprocess.DEVNULL).decode().strip()
        return commit, bool(status)
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _parse_ks(text):
    """``"0,64,128"`` -> ``[0, 64, 128]``, refusing repeats and a grid without ``k = 0``."""
    cols = [int(token) for token in str(text).split(",") if token.strip() != ""]
    if not cols:
        raise ValueError("--ks is empty")
    if len(set(cols)) != len(cols):
        raise ValueError("--ks repeats an angle: {}".format(cols))
    if 0 not in cols:
        raise ValueError("k = 0 is the paired reference of every Metric-1 quantity; "
                         "--ks must contain it, got {}".format(cols))
    return cols


def _parse_batches(text, dataset, batch_size):
    """Resolve ``--batches`` (``all`` / ``probe`` / an explicit list) to batch indices.

    Whatever the selection, a batch index always means the same window of the canonical
    order -- queries ``[b * batch_size, (b + 1) * batch_size)`` -- so a subset run's rows
    are a subset of the full run's rows, computed at the same batch composition.
    """
    total = -(-len(dataset) // int(batch_size))
    if str(text) == "all":
        return list(range(total))
    if str(text) == "probe":
        return select_probe_batches(dataset, batch_size=batch_size)
    picked = [int(token) for token in str(text).split(",") if token.strip() != ""]
    if not picked:
        raise ValueError("--batches is empty")
    if len(set(picked)) != len(picked):
        raise ValueError("--batches repeats a batch: {}".format(picked))
    outside = [b for b in picked if not 0 <= b < total]
    if outside:
        raise ValueError("--batches {} outside the {} canonical batches".format(
            outside, total))
    return sorted(picked)


def _summarize(values):
    """``{"mean", "n_valid", "n_nan"}`` of one per-sample array (mean over finite values)."""
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    return {"mean": float(array[finite].mean()) if finite.any() else None,
            "n_valid": int(finite.sum()), "n_nan": int((~finite).sum())}


def run(args):
    """Evaluate one checkpoint over the selected canonical batches at every angle.

    The order of business is a ladder of refusals before any compute: exp_03's pinned
    closure is re-verified, the manifest must hash to ``--manifest-hash``, the manifest's
    shot count must equal ``--num-shot`` and the built model's ``num_channels``, the angle
    grid must contain ``k = 0``, the batch selection must lie inside the canonical
    batching, and the loader must return the manifest's canonical order.

    Writes, in this order: ``wav_k<k>.npy`` (float32 ``[N, 9600]``, the arrays the metrics
    were computed from), ``per_sample.json`` (strict JSON, invalid samples ``null``),
    ``metrics.json`` and finally ``meta.json``, whose ``complete`` flag is the last thing
    written -- a reader that sees it knows every other output is finished and hashed.

    Args:
        args: the parsed CLI namespace (see :func:`main`).

    Returns:
        The per-sample dict that was written.

    Raises:
        RuntimeError: if ``--device cuda`` is asked for and no GPU is visible.
        ValueError: on any of the refusals above.
    """
    started = time.time()
    execution_id, started_at = new_execution_id()
    torch.set_num_threads(int(args.threads))
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.deterministic = True
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda asked for but no GPU is visible")

    from tools.exp10_compare import verify_exp03_pins          # avoids a cyclic import
    pins = verify_exp03_pins(args.binding_report)

    manifest = load_checked_manifest(args.manifest, args.manifest_hash)
    if int(manifest["num_shot"]) != int(args.num_shot):
        raise ValueError("manifest num_shot {} != --num-shot {}".format(
            manifest["num_shot"], args.num_shot))
    cols = _parse_ks(args.ks)

    dataset = build_manifest_dataset(manifest)
    batches = _parse_batches(args.batches, dataset, args.batch_size)
    indices = [i for b in batches
               for i in range(b * args.batch_size,
                              min((b + 1) * args.batch_size, len(dataset)))]
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers,
                                         pin_memory=(args.device == "cuda"))

    model = build_xrir(args.backbone, manifest["num_shot"])
    model.load_state_dict(load_model_state(args.checkpoint), strict=True)
    if int(model.num_channels) != int(args.num_shot):
        raise ValueError("the built model has {} reference channels, expected {}".format(
            model.num_channels, args.num_shot))
    model.to(args.device).eval()
    evaluator = Evaluator()
    print("arm {}  backbone {}  device {}  queries {}  batches {}  angles {}".format(
        args.arm, args.backbone, args.device, len(indices), len(batches), cols),
        flush=True)

    queries, angle_parts, control_parts, gt_parts, wave_parts = [], {}, {}, {}, {}
    timing = {"inference_s": 0.0, "inversion_s": 0.0, "metrics_s": 0.0, "loading_s": 0.0}
    loop_started = time.time()
    with patched_apply_delay():
        for i, batch in enumerate(loader):
            batch_started = time.time()
            timing["loading_s"] += batch_started - loop_started
            result = evaluate_batch(model, batch, cols, evaluator, gl_seed=args.gl_seed,
                                    batch_size=args.batch_size, device=args.device,
                                    controls=bool(args.controls))
            for name in ("inference_s", "inversion_s", "metrics_s"):
                timing[name] += result["timing"][name]
            queries.extend(result["keys"])
            for metric, values in result["gt"].items():
                gt_parts.setdefault(metric, []).append(values)
            for k, cell in result["angles"].items():
                for metric, values in cell.items():
                    angle_parts.setdefault(k, {}).setdefault(metric, []).append(values)
                wave_parts.setdefault(k, []).append(result["waveforms"][k])
            for name, cell in result["controls"].items():
                for metric, values in cell.items():
                    control_parts.setdefault(name, {}).setdefault(metric, []).append(values)
            if (i + 1) % args.log_interval == 0 or len(queries) == len(indices):
                elapsed = time.time() - started
                rate = len(queries) / max(elapsed, 1e-9)
                print("[{}/{}] {:.2f} samples/s, eta {:.1f} min".format(
                    len(queries), len(indices), rate,
                    (len(indices) - len(queries)) / max(rate, 1e-9) / 60.0), flush=True)
            loop_started = time.time()

    entries = [dataset.entries[i] for i in indices]
    if queries != [entry["query"] for entry in entries]:
        raise ValueError("the loader did not return the manifest's canonical order")

    _sync(args.device)
    write_started = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    arrays = {}
    for k in cols:
        block = np.concatenate(wave_parts[k]).astype(np.float32)
        name = "wav_k{}.npy".format(k)
        path = os.path.join(args.out_dir, name)
        np.save(path, block)
        arrays[name] = {"sha256": file_sha256(path), "shape": list(block.shape),
                        "dtype": str(block.dtype), "k": int(k)}

    commit, dirty = _git_commit()
    tool_sha = file_sha256(os.path.abspath(__file__))
    protocol_fields = {
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest_hash": args.manifest_hash, "gl_seed": int(args.gl_seed),
        "num_shot": int(args.num_shot), "device": args.device,
        "cudnn_deterministic": True, "cudnn_allow_tf32": bool(args.tf32),
        "matmul_allow_tf32": bool(args.tf32), "batch_size": int(args.batch_size),
        "batch_canonical": True, "torch_version": torch.__version__,
        "numpy_version": np.__version__, "tool_sha256": tool_sha}
    meta_core = dict(protocol_fields)
    meta_core.update({
        "arm": args.arm, "backbone": args.backbone, "checkpoint": args.checkpoint,
        "manifest_path": args.manifest, "manifest_seed": manifest["seed"],
        "split": "unseen", "ks": cols, "controls": bool(args.controls),
        "control_specs": [{"id": name, "k": int(k), "compare_to": int(reference)}
                          for name, k, reference in CONTROL_SPECS] if args.controls else [],
        "tf32": bool(args.tf32), "batches_arg": str(args.batches), "batches": batches,
        "n_batches_total": -(-len(dataset) // args.batch_size),
        "n_queries": len(indices), "n_split": len(dataset),
        "native_len": NATIVE_LEN, "padded_len": PADDED_LEN, "sample_rate": SAMPLE_RATE,
        "metric_window": METRIC_WINDOW, "threads": int(args.threads),
        "num_workers": int(args.num_workers),
        "python_version": sys.version.split()[0], "git_commit": commit,
        "git_dirty": dirty, "verify_exp03_pins": pins,
        "run_id": protocol_id({key: protocol_fields[key] for key in (
            "checkpoint_sha256", "manifest_hash", "gl_seed", "num_shot", "device",
            "cudnn_deterministic", "cudnn_allow_tf32", "matmul_allow_tf32")}),
        "protocol_id": protocol_id(protocol_fields), "execution_id": execution_id,
        "started_at": started_at, "arrays": arrays,
        "query_list_sha256": query_list_sha256(queries)})

    def _cells(parts):
        return {name: {metric: np.concatenate(chunks)
                       for metric, chunks in metrics.items()}
                for name, metrics in parts.items()}

    angles = _cells(angle_parts)
    controls = _cells(control_parts)
    gt = {metric: np.concatenate(chunks) for metric, chunks in gt_parts.items()}

    per_sample = {"meta": meta_core, "protocol_id": meta_core["protocol_id"],
                  "execution_id": execution_id, "query": queries,
                  "index": [entry["index"] for entry in entries], "batches": batches,
                  "gt": {metric: json_values(values) for metric, values in gt.items()},
                  "angles": {str(k): {metric: json_values(values)
                                      for metric, values in cell.items()}
                             for k, cell in angles.items()},
                  "controls": {name: {metric: json_values(values)
                                      for metric, values in cell.items()}
                               for name, cell in controls.items()}}
    metrics_out = {"meta": meta_core, "n_queries": len(queries),
                   "gt": {metric: _summarize(values) for metric, values in gt.items()},
                   "angles": {str(k): {metric: _summarize(values)
                                       for metric, values in cell.items()}
                              for k, cell in angles.items()},
                   "controls": {name: {metric: _summarize(values)
                                       for metric, values in cell.items()}
                                for name, cell in controls.items()}}

    per_sample_path = os.path.join(args.out_dir, "per_sample.json")
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    write_json(per_sample, per_sample_path)
    write_json(metrics_out, metrics_path)
    _sync(args.device)
    timing["writing_s"] = time.time() - write_started
    timing["total_min"] = (time.time() - started) / 60.0
    meta = dict(meta_core)
    meta.update({"per_sample_sha256": file_sha256(per_sample_path),
                 "metrics_sha256": file_sha256(metrics_path),
                 "timing": timing, "complete": True})
    write_json(meta, os.path.join(args.out_dir, "meta.json"))
    print("done: {} queries x {} angles in {:.2f} min ({})".format(
        len(queries), len(cols), timing["total_min"], args.out_dir), flush=True)
    return per_sample


def main(argv=None):
    """Parse the CLI and run the pilot (see the module docstring for an example)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, help="the plan's arm label, e.g. released_k8")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-hash", required=True,
                        help="sha256 of the manifest's content; the run refuses to start "
                             "unless it matches, so every arm sees the same references")
    parser.add_argument("--num-shot", type=int, required=True,
                        help="must equal the manifest's num_shot and the model's channels")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS),
                        help="comma-separated column rolls (W=512); must contain 0")
    parser.add_argument("--batches", default="all",
                        help="all | probe | a comma-separated list of canonical batch "
                             "indices")
    parser.add_argument("--controls", action="store_true",
                        help="also run the three plumbing controls as fresh inferences")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--gl-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--tf32", action="store_true",
                        help="allow TF32; off by default, as in exp_03")
    parser.add_argument("--binding-report",
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "ckpt", "yaw_rotation", "binding_report.json"))
    parser.add_argument("--log-interval", type=int, default=5)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
