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

import contextlib

import numpy as np
import torch

import model.xRIR as model_xrir
from tools.yaw_rotation import fixed_alignment, rotate_scene_yaw

from tools.per_sample_metrics import (
    acoustic_metrics,
    griffin_lim_seeded,
    sample_seed,
)

MAG_EPS = 1e-8
REL_L2_EPS = 1e-8
METRIC_WINDOW = 8000          # exp_01's acoustic window, enforced by acoustic_metrics
SAMPLE_RATE = 22050
NATIVE_LEN = 9579             # (310 - 1) * hop 31: what Griffin-Lim returns
PADDED_LEN = 9600             # exp_01's max_len, the length the waveforms are stored at


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
