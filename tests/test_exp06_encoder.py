"""Geometry and symmetry contracts for the heading-conditioned encoder."""
import inspect
import math

import pytest
import torch

from model.cylindrical_vit import CylindricalViT
from model.cylindrical_vit_oriented import CylindricalViTOriented, azimuth_channels
from tools.yaw_rotation import rotate_scene_yaw
from treble_multi_room_dataset.treble_xRIR_dataset import convert_equirect_to_camera_coord


def encoder(cls=CylindricalViTOriented):
    torch.manual_seed(6)
    return cls(image_size=(32, 512), dim=32, depth=1, heads=2, dim_head=16).eval()


def scene():
    torch.manual_seed(7)
    depth = convert_equirect_to_camera_coord(torch.rand(32, 512) + 1, 32, 512)
    return depth.permute(2, 0, 1)[None], torch.randn(1, 3), torch.randn(1, 2, 3)


def encoding(value):
    depth, src, _ = value
    return (src[:, :, None, None] - depth) / 5


def token_roll(tokens, k):
    return tokens.reshape(1, 2, 16, -1).roll(k, dims=2).reshape_as(tokens)


def test_shape_buffers_camera_and_signature():
    model = encoder()
    assert model(encoding(scene())).shape == (1, 32, 32)
    expected = torch.stack((model.cos_theta, model.sin_theta))[:, None].expand(2, 32, 512)
    assert torch.equal(model.az_channels, expected)
    assert torch.equal(azimuth_channels(32, 512), expected)
    xyz = convert_equirect_to_camera_coord(torch.ones(32, 512), 32, 512)
    horizontal = xyz[..., :2] / torch.linalg.vector_norm(xyz[..., :2], dim=-1, keepdim=True)
    torch.testing.assert_close(model.az_channels, horizontal.permute(2, 0, 1), atol=1e-6, rtol=0)
    assert 'az_channels' not in model.state_dict()
    parent = inspect.signature(CylindricalViT).parameters
    child = inspect.signature(CylindricalViTOriented).parameters
    assert [(k, v.default) for k, v in parent.items()] == [(k, v.default) for k, v in child.items()]


@pytest.mark.parametrize('kwargs', [{}, {'image_size': (32, 64), 'patch_size': (8, 16), 'dim': 32}])
def test_exact_parameter_delta(kwargs):
    counts = [sum(p.numel() for p in cls(**kwargs).parameters())
              for cls in (CylindricalViT, CylindricalViTOriented)]
    p1, p2 = kwargs.get('patch_size', (16, 32))
    delta = 2 * p1 * p2 * kwargs.get('dim', 512) + 2 * (5 - 3) * p1 * p2
    assert counts[1] - counts[0] == delta
    if not kwargs:
        assert delta == 526336


@torch.no_grad()
def test_active_scene_yaw_breaks_only_child_equivariance():
    original = scene()
    rotated = rotate_scene_yaw(*original, 32)
    for cls, equivariant in [(CylindricalViT, True), (CylindricalViTOriented, False)]:
        model = encoder(cls)
        error = (model(encoding(rotated)) - token_roll(model(encoding(original)), 1)).abs().max()
        assert error <= 1e-5 if equivariant else error > 1e-2


@pytest.mark.parametrize('j', [0, 1, 32, 128, 511])
def test_active_yaw_matches_independent_roll_and_matrix(j):
    original = scene()
    angle = j * 2 * math.pi / 512
    c, s = math.cos(angle), math.sin(angle)
    matrix = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    depth, src, refs = original
    expected = (torch.einsum('ij,bjhw->bihw', matrix, depth.roll(j, -1)),
                src @ matrix.T, refs @ matrix.T)
    for actual, independent in zip(rotate_scene_yaw(*original, j), expected):
        torch.testing.assert_close(actual, independent, atol=1e-6, rtol=0)


@pytest.mark.parametrize('phi', [0.3515625, -0.3515625, 359.8, -90.0])
@pytest.mark.parametrize('j', [0, 1, 32, 128, 511])
@torch.no_grad()
def test_joint_heading_cancellation(phi, j):
    # Independent quantisation oracle; production helper's ties are tested separately.
    k = math.floor(-phi * 512 / 360 + 0.5) % 512
    canonical = -k * 360 / 512
    shifted_k = math.floor(-(canonical + j * 360 / 512) * 512 / 360 + 0.5) % 512
    original = scene()
    first = encoding(rotate_scene_yaw(*original, k))
    second = encoding(rotate_scene_yaw(*rotate_scene_yaw(*original, j), shifted_k))
    torch.testing.assert_close(first, second, atol=1e-4, rtol=0)
    model = encoder()
    one, two = model(first), model(second)
    torch.testing.assert_close(one, two, atol=1e-4, rtol=0)
    if j == 0:
        assert torch.equal(first, second) and torch.equal(one, two)


@torch.no_grad()
def test_mirror_discrimination_in_centred_box():
    rays = convert_equirect_to_camera_coord(torch.ones(32, 512), 32, 512)
    half_extent = torch.tensor([3.0, 2.0, 1.5])
    depth = (rays / (rays.abs() / half_extent).amax(-1, keepdim=True)).permute(2, 0, 1)[None]
    src = torch.tensor([[0.6, 1.0, 0.0]])
    refs = src[:, None]
    mirrored = src * torch.tensor([-1, -1, 1])
    for cls, equivariant in [(CylindricalViT, True), (CylindricalViTOriented, False)]:
        model = encoder(cls)
        one = model(encoding((depth, src, refs)))
        two = model(encoding((depth, mirrored, refs)))
        error = (two - token_roll(one, 8)).abs().max()
        assert error <= 1e-5 if equivariant else error > 1e-2


def test_float64_cpu_and_channel_refusal():
    model = encoder().double()
    out = model(encoding(scene()).double())
    assert out.dtype == model.az_channels.dtype == torch.float64
    assert out.device == model.az_channels.device == torch.device('cpu')
    assert azimuth_channels(32, 512, dtype=torch.float64).dtype == torch.float64
    with pytest.raises(ValueError, match='3'):
        CylindricalViTOriented(in_channels=4)
