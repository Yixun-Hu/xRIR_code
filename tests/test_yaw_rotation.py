"""Tests for :mod:`tools.yaw_rotation` (exp_03, yaw_rotation_degradation).

Convention under test: an ACTIVE yaw of the whole scene by +Delta about the
receiver's vertical (z) axis, right-handed (+x -> +y looking down the -z axis),
with Delta = 2*pi*k/W for an integer roll of ``k`` panorama columns.
"""
import math

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
