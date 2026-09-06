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
from typing import Tuple

import torch


def yaw_angle_rad(k: int, W: int = 512) -> float:
    """Yaw angle in radians corresponding to a roll of ``k`` panorama columns.

    Args:
        k: number of panorama columns; may be negative or ``>= W`` (not reduced here).
        W: panorama width in columns.

    Returns:
        ``2 * pi * k / W`` as a Python float.
    """
    return 2.0 * math.pi * float(k) / float(W)


def rotate_vectors_z(v: torch.Tensor, angle: float) -> torch.Tensor:
    """Actively rotate 3-D vectors about the ``z`` axis by ``angle`` radians.

    ``x' = x cos(a) - y sin(a)``, ``y' = x sin(a) + y cos(a)``, ``z' = z``.

    Args:
        v: tensor of shape ``[..., 3]``; the last axis holds ``(x, y, z)``.
        angle: rotation angle in radians (right-handed about ``+z``).

    Returns:
        A new tensor of the same shape, dtype and device as ``v``.
    """
    if v.shape[-1] != 3:
        raise ValueError("rotate_vectors_z expects the last axis to be xyz (size 3), "
                         "got shape {}".format(tuple(v.shape)))
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
        depth_coord: receiver-frame panorama coordinates ``[B, 3, H, W]``.
        src_loc: receiver-frame query-source position ``[B, 3]``.
        ref_locs: receiver-frame reference-source positions ``[B, K, 3]``.
        k: column roll; negative values and values ``>= W`` are reduced modulo ``W``.
        W: panorama width in columns (must match ``depth_coord.shape[-1]``).

    Returns:
        ``(depth_coord', src_loc', ref_locs')`` -- new tensors; the inputs are untouched.
    """
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
