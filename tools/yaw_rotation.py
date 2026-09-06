"""Active yaw rotation of an xRIR scene about the receiver's vertical axis.

Experiment exp_03 (``yaw_rotation_degradation``) asks whether jointly rotating the
receiver-centred depth panorama and the source / reference-source coordinates in yaw
degrades RIR synthesis.  For an omnidirectional source and a monaural receiver this
rotation is a symmetry of the acoustics, so the ground-truth RIR is unchanged and any
change in the prediction is model sensitivity.

Convention
----------
An **active** yaw of the whole scene by ``+Delta`` about the receiver's ``z`` axis,
right-handed (counter-clockwise seen from ``+z``: ``+x -> +y``), with

    Delta = 2*pi*k / W

for an integer roll of ``k`` panorama columns (``W`` columns in total).  The panorama
convention is the dataset's ``convert_equirect_to_camera_coord``: column ``c`` looks
along ``theta_c = (c + 0.5) * 2*pi/W - pi`` with ``x = d cos(phi) cos(theta)``,
``y = d cos(phi) sin(theta)``, ``z = -d sin(phi)``.  Because ``theta_c - theta_{c-k} =
2*pi*k/W = Delta``, rolling the panorama by ``+k`` columns and then rotating the xyz
channel vectors by ``Rz(Delta)`` is exactly the same scene as re-projecting the rolled
depth map, i.e. ``Rz(Delta) . roll(convert(D), k) == convert(roll(D, k))``.

All functions are pure (no in-place modification of their arguments) and work on CPU
and CUDA tensors.
"""
from __future__ import annotations

import contextlib
import math
import numbers
from typing import Tuple

import torch


def _check_k_and_width(k, W) -> None:
    """Validate a column roll ``k`` and a panorama width ``W``.

    Raises:
        TypeError: if ``k`` or ``W`` is not an integer (floats are rejected rather than
            silently truncated, since a fractional roll has no meaning here).
        ValueError: if ``W <= 0``.
    """
    if isinstance(k, bool) or not isinstance(k, numbers.Integral):
        raise TypeError("k must be an integer number of panorama columns, got {!r}".format(k))
    if isinstance(W, bool) or not isinstance(W, numbers.Integral):
        raise TypeError("W must be an integer panorama width, got {!r}".format(W))
    if int(W) <= 0:
        raise ValueError("W must be positive, got {}".format(W))


def _require_float(t: torch.Tensor, name: str) -> None:
    """Raise ``TypeError`` unless ``t`` is a floating-point tensor."""
    if not torch.is_floating_point(t):
        raise TypeError("{} must be a floating-point tensor, got dtype {}".format(name, t.dtype))


def yaw_angle_rad(k: int, W: int = 512) -> float:
    """Yaw angle in radians corresponding to a roll of ``k`` panorama columns.

    Args:
        k: integer number of panorama columns; may be negative or ``>= W`` (not reduced
            here).  Non-integer values raise ``TypeError``.
        W: integer panorama width in columns; must be positive.

    Returns:
        ``2 * pi * k / W`` as a Python float.
    """
    _check_k_and_width(k, W)
    return 2.0 * math.pi * float(k) / float(W)


def rotate_vectors_z(v: torch.Tensor, angle: float) -> torch.Tensor:
    """Actively rotate 3-D vectors about the ``z`` axis by ``angle`` radians.

    ``x' = x cos(a) - y sin(a)``, ``y' = x sin(a) + y cos(a)``, ``z' = z``.

    Args:
        v: floating-point tensor of shape ``[..., 3]``; the last axis holds ``(x, y, z)``.
            Integer tensors are rejected (the rotation is not closed over the integers).
        angle: rotation angle in radians (right-handed about ``+z``).

    Returns:
        A new tensor of the same shape, dtype and device as ``v``.

    Raises:
        TypeError: if ``v`` is not floating point.
        ValueError: if the last axis of ``v`` is not of size 3.
    """
    if v.shape[-1] != 3:
        raise ValueError("rotate_vectors_z expects the last axis to be xyz (size 3), "
                         "got shape {}".format(tuple(v.shape)))
    _require_float(v, "v")
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack([x * cos_a - y * sin_a, x * sin_a + y * cos_a, z], dim=-1)


def rotate_scene_yaw(
    depth_coord: torch.Tensor,
    src_loc: torch.Tensor,
    ref_locs: torch.Tensor,
    k: int,
    W: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the active ``+2*pi*k/W`` yaw to a whole xRIR scene.

    ``depth_coord' = Rz(Delta) . roll(depth_coord, k, dim=-1)``,
    ``src_loc' = Rz(Delta) src_loc``, ``ref_locs' = Rz(Delta) ref_locs``.

    Args:
        depth_coord: receiver-frame panorama coordinates ``[B, 3, H, W]``, floating point.
        src_loc: receiver-frame query-source position ``[B, 3]``, floating point.
        ref_locs: receiver-frame reference-source positions ``[B, K, 3]``, floating point.
        k: integer column roll; negative values and values ``>= W`` are reduced modulo
            ``W``.  Non-integer values raise ``TypeError``.
        W: integer panorama width in columns (must match ``depth_coord.shape[-1]``).

    Returns:
        ``(depth_coord', src_loc', ref_locs')`` -- new tensors; the inputs are untouched.

    Raises:
        TypeError: if ``k``/``W`` are not integers or any tensor is not floating point.
        ValueError: if ``W <= 0`` or the shapes do not match the contract.
    """
    _check_k_and_width(k, W)
    _require_float(depth_coord, "depth_coord")
    _require_float(src_loc, "src_loc")
    _require_float(ref_locs, "ref_locs")
    if depth_coord.dim() != 4 or depth_coord.shape[1] != 3:
        raise ValueError("depth_coord must have shape [B, 3, H, W], got {}".format(
            tuple(depth_coord.shape)))
    if depth_coord.shape[-1] != W:
        raise ValueError("depth_coord has {} columns but W={}".format(
            depth_coord.shape[-1], W))

    k = int(k) % int(W)
    angle = yaw_angle_rad(k, W)

    rolled = torch.roll(depth_coord, shifts=k, dims=-1)          # [B, 3, H, W]
    depth_rot = rotate_vectors_z(rolled.permute(0, 2, 3, 1), angle).permute(0, 3, 1, 2)
    return depth_rot.contiguous(), rotate_vectors_z(src_loc, angle), rotate_vectors_z(ref_locs, angle)


def integer_delays(
    src_loc: torch.Tensor,
    ref_locs: torch.Tensor,
    sr: int = 22050,
    c: float = 343.0,
) -> torch.Tensor:
    """Integer direct-path delays exactly as ``model.xRIR.xRIR.shift_and_align`` computes them.

    The op sequence and dtypes mirror the model line for line
    (``torch.linalg.norm`` -> ``torch.round(...).int()``), so both the values and the
    dtype agree bit for bit with the model's ``delay_unit`` on the same inputs.

    Args:
        src_loc: receiver-frame query-source position ``[B, 3]``.
        ref_locs: receiver-frame reference-source positions ``[B, K, 3]``.
        sr: sample rate in Hz (the model hard-codes 22050).
        c: speed of sound in m/s (the model hard-codes 343).

    Returns:
        ``[B, K]`` **int32** tensor of sample delays -- the dtype the model itself produces
        (positive = reference arrives earlier than the query source and is therefore
        shifted right).
    """
    dist_src = torch.linalg.norm(src_loc, dim=1).unsqueeze(1)          # [B, 1]
    dist_ref = torch.linalg.norm(ref_locs, dim=-1)                     # [B, K]
    return torch.round((dist_src - dist_ref) / c * sr).int()           # as in shift_and_align


@contextlib.contextmanager
def fixed_alignment(model, aligned_refs: torch.Tensor):
    """Temporarily pin ``model.shift_and_align`` to a pre-computed alignment.

    ``xRIR.shift_and_align`` is algebraically yaw-invariant but not numerically: float32
    norms plus ``round()`` can flip a delay by one sample under rotation.  The primary
    experimental condition therefore holds the direct-path alignment at the ``k = 0``
    coordinates so that only the geometry branches see the rotation::

        aligned = model.shift_and_align(refs, src0, ref_locs0)
        with fixed_alignment(model, aligned):
            out, tgt = model(depth_k, refs, src_k, ref_locs_k, tgt)

    The original bound method is restored on exit, including when the block raises.

    Args:
        model: an ``xRIR`` (or subclass) instance.
        aligned_refs: the cached ``[B, K, T]`` aligned reference audio to return instead.

    Yields:
        ``model``, with its ``shift_and_align`` replaced.
    """
    had_own = "shift_and_align" in vars(model)
    original = vars(model).get("shift_and_align", None)

    def _fixed(*args, **kwargs):
        return aligned_refs

    model.shift_and_align = _fixed
    try:
        yield model
    finally:
        if had_own:
            model.shift_and_align = original
        elif "shift_and_align" in vars(model):
            # Guarded so a raise inside the block is never masked by an AttributeError here.
            del model.shift_and_align
