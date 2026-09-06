"""Tests for :mod:`tools.yaw_rotation` (exp_03, yaw_rotation_degradation).

Convention under test: an ACTIVE yaw of the whole scene by +Delta about the
receiver's vertical (z) axis, right-handed (+x -> +y looking down the -z axis),
with Delta = 2*pi*k/W for an integer roll of ``k`` panorama columns.
"""
import math

import numpy as np
import pytest
import torch

from tools.yaw_rotation import rotate_scene_yaw, rotate_vectors_z, yaw_angle_rad


# --------------------------------------------------------------------------------------
# T1 -- yaw_angle_rad
# --------------------------------------------------------------------------------------
def test_yaw_angle_rad_values():
    assert yaw_angle_rad(0) == pytest.approx(0.0)
    assert yaw_angle_rad(512) == pytest.approx(2.0 * math.pi)
    assert yaw_angle_rad(128) == pytest.approx(math.pi / 2)
    # Explicit W is honoured.
    assert yaw_angle_rad(1, W=4) == pytest.approx(math.pi / 2)


# --------------------------------------------------------------------------------------
# T2 -- rotate_vectors_z
# --------------------------------------------------------------------------------------
def test_rotate_vectors_z_preserves_norm_and_z():
    torch.manual_seed(0)
    v = torch.randn(7, 5, 3)
    out = rotate_vectors_z(v, 0.7)
    assert out.shape == v.shape
    assert torch.allclose(torch.linalg.norm(out, dim=-1), torch.linalg.norm(v, dim=-1), atol=1e-6)
    assert torch.allclose(out[..., 2], v[..., 2], atol=1e-6)


def test_rotate_vectors_z_active_right_handed():
    axes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = rotate_vectors_z(axes, math.pi / 2)
    assert torch.allclose(out[0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert torch.allclose(out[1], torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6)


# --------------------------------------------------------------------------------------
# T3 -- rotate_scene_yaw
# --------------------------------------------------------------------------------------
def _random_scene(seed=0, B=2, K=4, H=32, W=512):
    torch.manual_seed(seed)
    return (torch.randn(B, 3, H, W), torch.randn(B, 3), torch.randn(B, K, 3))


def test_rotate_scene_yaw_identity_and_period():
    depth, src, refs = _random_scene()
    d0, s0, r0 = rotate_scene_yaw(depth, src, refs, 0)
    assert torch.equal(d0, depth) and torch.equal(s0, src) and torch.equal(r0, refs)

    dW, sW, rW = rotate_scene_yaw(depth, src, refs, 512)
    assert torch.allclose(dW, d0, atol=1e-6)
    assert torch.allclose(sW, s0, atol=1e-6)
    assert torch.allclose(rW, r0, atol=1e-6)


def test_rotate_scene_yaw_negative_k_wraps():
    depth, src, refs = _random_scene(seed=1)
    neg = rotate_scene_yaw(depth, src, refs, -16)
    pos = rotate_scene_yaw(depth, src, refs, 512 - 16)
    for a, b in zip(neg, pos):
        assert torch.allclose(a, b, atol=1e-6)


def test_rotate_scene_yaw_composition():
    depth, src, refs = _random_scene(seed=2)
    k1, k2 = 40, 500
    mid = rotate_scene_yaw(depth, src, refs, k1)
    two_step = rotate_scene_yaw(mid[0], mid[1], mid[2], k2)
    one_step = rotate_scene_yaw(depth, src, refs, (k1 + k2) % 512)
    for a, b in zip(two_step, one_step):
        assert torch.allclose(a, b, atol=1e-5)


def test_rotate_scene_yaw_does_not_mutate_inputs():
    depth, src, refs = _random_scene(seed=3)
    depth_c, src_c, refs_c = depth.clone(), src.clone(), refs.clone()
    rotate_scene_yaw(depth, src, refs, 37)
    assert torch.equal(depth, depth_c)
    assert torch.equal(src, src_c)
    assert torch.equal(refs, refs_c)


# --------------------------------------------------------------------------------------
# T4 -- oracle: Rz(Delta) . roll(convert(D), k) == convert(roll(D, k))
# --------------------------------------------------------------------------------------
def _rz_roll(t: torch.Tensor, k: int, W: int = 512) -> torch.Tensor:
    """Independent reference implementation of "roll by +k columns, then Rz(+2*pi*k/W)".

    Written out with explicit cos/sin so it does not share code with
    :func:`tools.yaw_rotation.rotate_scene_yaw`.  ``t`` is ``[B, 3, H, W]``.
    """
    k = int(k) % int(W)
    a = 2.0 * math.pi * k / W
    cos_a, sin_a = math.cos(a), math.sin(a)
    r = torch.roll(t, shifts=k, dims=-1)
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    return torch.stack([x * cos_a - y * sin_a, x * sin_a + y * cos_a, z], dim=1)


def test_rotate_scene_yaw_matches_reprojected_rolled_depth():
    from treble_multi_room_dataset.treble_xRIR_dataset import convert_equirect_to_camera_coord

    H, W = 256, 512
    rng = np.random.RandomState(0)
    depth_map = (rng.rand(H, W).astype(np.float32) * 4.0 + 1.0)          # positive depths, 1..5 m

    def convert(dm):
        return convert_equirect_to_camera_coord(
            torch.from_numpy(dm), H, W).permute(2, 0, 1).float().unsqueeze(0)

    depth_coord = convert(depth_map)                                      # [1, 3, H, W]
    zeros_src = torch.zeros(1, 3)
    zeros_ref = torch.zeros(1, 1, 3)

    for k in (0, 1, 32, 100, 511):
        got = rotate_scene_yaw(depth_coord, zeros_src, zeros_ref, k, W=W)[0]
        want = convert(np.roll(depth_map, k, axis=1))
        assert got.shape == want.shape
        # All three channels, every column.
        assert torch.allclose(got, want, atol=1e-4), "k={} max|d|={}".format(
            k, (got - want).abs().max().item())
        # Seam columns checked explicitly (wrap-around is where a convention error shows).
        for col in (0, W - 1):
            assert torch.allclose(got[..., col], want[..., col], atol=1e-4), \
                "k={} col={}".format(k, col)


# --------------------------------------------------------------------------------------
# T5 -- covariance of the three ViT views used by xRIR.forward
# --------------------------------------------------------------------------------------
def test_rotate_scene_yaw_covaries_all_three_vit_views():
    B, K, H, W, k = 2, 4, 256, 512, 64
    torch.manual_seed(7)
    depth = torch.randn(B, 3, H, W)
    src = torch.randn(B, 3)
    refs = torch.randn(B, K, 3)

    depth_r, src_r, refs_r = rotate_scene_yaw(depth, src, refs, k, W=W)

    src_view = src[:, :, None, None] - depth
    src_view_r = src_r[:, :, None, None] - depth_r
    assert torch.allclose(src_view_r, _rz_roll(src_view, k, W), atol=1e-5)

    rec_view = -depth
    rec_view_r = -depth_r
    assert torch.allclose(rec_view_r, _rz_roll(rec_view, k, W), atol=1e-5)

    for i in range(K):
        ref_view = refs[:, i, :, None, None] - depth
        ref_view_r = refs_r[:, i, :, None, None] - depth_r
        assert torch.allclose(ref_view_r, _rz_roll(ref_view, k, W), atol=1e-5), "ref {}".format(i)
