"""Counter-seeded, geometry-only yaw augmentation for exp_04; audio is never an input.

The FLAC integer mixer is preserved, with strict argument validation before coercion.
The model/GPU-dependent alignment_audit is added in the next round.
"""
from dataclasses import dataclass
import math
import numbers

import torch

_MASK64 = (1 << 64) - 1
_SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX64_MIX1 = 0xBF58476D1CE4E5B9
_SPLITMIX64_MIX2 = 0x94D049BB133111EB
_MASK32 = (1 << 32) - 1
_YAW_AUG_RANK_BITS = 12
_YAW_AUG_MAX_RANK = 1 << _YAW_AUG_RANK_BITS
_YAW_AUG_MAX_STEP = 1 << (32 - _YAW_AUG_RANK_BITS)
_FMIX32_MUL1 = 0x85EBCA6B
_FMIX32_MUL2 = 0xC2B2AE35


def splitmix64(x: int) -> int:
    """One SplitMix64 round: avalanche a 64-bit word into a 64-bit word."""
    z = (x + _SPLITMIX64_GAMMA) & _MASK64
    z = ((z ^ (z >> 30)) * _SPLITMIX64_MIX1) & _MASK64
    z = ((z ^ (z >> 27)) * _SPLITMIX64_MIX2) & _MASK64
    return z ^ (z >> 31)


def fmix32(x: int) -> int:
    """Murmur3's bijective 32-bit finaliser."""
    x &= _MASK32
    x ^= x >> 16
    x = (x * _FMIX32_MUL1) & _MASK32
    x ^= x >> 13
    x = (x * _FMIX32_MUL2) & _MASK32
    x ^= x >> 16
    return x


def _require_ints(*values):
    if any(type(value) is not int for value in values):
        raise ValueError("arguments must be integers, not bools or coerced values")


def step_seed(seed, t, rank=0):
    """FLAC's 32-bit keyed bijection on (20-bit micro-batch counter, 12-bit rank)."""
    step = t
    _require_ints(seed, step, rank)
    seed, step, rank = int(seed), int(step), int(rank)
    if not 0 <= step < _YAW_AUG_MAX_STEP:
        raise ValueError("yaw_aug step must be in [0, 2**20)")
    if not 0 <= rank < _YAW_AUG_MAX_RANK:
        raise ValueError("yaw_aug rank must be in [0, 4096)")
    key = splitmix64(seed & _MASK64)
    k_lo = key & _MASK32
    k_hi = (key >> 32) & _MASK32
    v = ((step << _YAW_AUG_RANK_BITS) | rank) & _MASK32
    return fmix32(v ^ k_lo) ^ k_hi


def counter(epoch, batch_idx, batches_per_epoch=9261):
    """Map one-based epochs and zero-based micro-batches to the run counter."""
    _require_ints(epoch, batch_idx, batches_per_epoch)
    if epoch < 1 or batches_per_epoch <= 0 or not 0 <= batch_idx < batches_per_epoch:
        raise ValueError("invalid epoch, batch index, or batches per epoch")
    return (epoch - 1) * batches_per_epoch + batch_idx


def draw_offsets(n, W, generator):
    """Draw integer column offsets using only a dedicated CPU generator."""
    _require_ints(n, W)
    if n < 0 or W <= 0:
        raise ValueError("n must be nonnegative and W must be positive")
    if (not isinstance(generator, torch.Generator) or generator is torch.default_generator
            or generator.device.type != "cpu"):
        raise ValueError("a dedicated CPU torch.Generator is required")
    return torch.randint(0, W, (n,), generator=generator, dtype=torch.long)


def rotate_scene_yaw_batch(depth_coord, src_loc, ref_locs, ks, W=512):
    """Apply per-sample active yaw: exact column gather followed by ``Rz(2*pi*ks/W)``.

    Geometry must share a floating dtype/device and have shapes ``[B,3,H,W]``,
    ``[B,3]``, ``[B,K,3]``. Integer ``ks [B]`` must be on that device; offsets
    are reduced modulo the positive integer width W. Invalid inputs raise
    ValueError. Empty batches are legal; all three outputs are new tensors.
    """
    scene = (depth_coord, src_loc, ref_locs)
    if isinstance(W, bool) or not isinstance(W, numbers.Integral) or W <= 0:
        raise ValueError("W must be a positive integer")
    if not all(isinstance(t, torch.Tensor) for t in scene + (ks,)):
        raise ValueError("geometry and ks must be tensors")
    if not all(t.is_floating_point() and t.dtype == depth_coord.dtype for t in scene):
        raise ValueError("geometry must share a floating dtype")
    if not all(t.device == depth_coord.device for t in scene + (ks,)):
        raise ValueError("geometry and ks must share a device")
    if depth_coord.ndim != 4 or depth_coord.shape[1] != 3 or depth_coord.shape[-1] != W:
        raise ValueError("depth_coord must have shape [B, 3, H, W] matching W")
    B = depth_coord.shape[0]
    if src_loc.shape != (B, 3):
        raise ValueError("src_loc must have shape [B, 3]")
    if ref_locs.ndim != 3 or ref_locs.shape[0] != B or ref_locs.shape[-1] != 3:
        raise ValueError("ref_locs must have shape [B, K, 3]")
    if ks.shape != (B,) or ks.dtype not in (torch.uint8, torch.int8, torch.int16,
                                           torch.int32, torch.int64):
        raise ValueError("ks must be an integer tensor of shape [B]")
    ks = ks.to(dtype=torch.long).remainder(W)
    columns = (torch.arange(W, device=ks.device)[None, :] - ks[:, None]).remainder(W)
    rolled = depth_coord.gather(-1, columns[:, None, None, :].expand_as(depth_coord))
    # Evaluate trig in double precision like the scalar helper's Python math.
    angles = 2.0 * math.pi * ks.to(torch.float64) / W
    cos_a, sin_a = angles.cos().to(depth_coord.dtype), angles.sin().to(depth_coord.dtype)

    def rotate(v):
        shape = (B,) + (1,) * (v.ndim - 2)
        c, s = cos_a.reshape(shape), sin_a.reshape(shape)
        x, y, z = v.unbind(-1)
        out = torch.stack((x * c - y * s, x * s + y * c, z), dim=-1)
        # Preserve even signed zeros for identity rows, without returning aliases.
        return torch.where((ks == 0).reshape(shape + (1,)), v, out)

    depth_rot = rotate(rolled.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
    return depth_rot, rotate(src_loc), rotate(ref_locs)


def apply_yaw_aug(depth_coord, src_loc, ref_locs, ks, W=512):
    """Rotate geometry only; callers move CPU offsets to the geometry's device."""
    return rotate_scene_yaw_batch(depth_coord, src_loc, ref_locs, ks, W=W)


@dataclass(frozen=True)
class YawAug:
    """Validated configuration; enabled is consumed by the caller's training gate."""
    enabled: bool = False
    W: int = 512
    seed: int = 0

    def __post_init__(self):
        _require_ints(self.W, self.seed)
        if type(self.enabled) is not bool or self.W <= 0:
            raise ValueError("enabled must be a literal bool and W must be positive")

    def offsets_for(self, epoch, batch_idx, n):
        generator = torch.Generator(device="cpu").manual_seed(step_seed(self.seed, counter(epoch, batch_idx), 0))
        return draw_offsets(n, self.W, generator)
