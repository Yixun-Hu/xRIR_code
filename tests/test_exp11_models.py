"""exp_11 arms H/I: the SimpleViT that receives the explicit azimuth planes.

The encoder equalities are exp_06's (``tests/test_exp06_encoder.py``) read against the
SimpleViT family: the two planes must be the normalised horizontal look direction of
``convert_equirect_to_camera_coord`` and the very channels ``CylindricalViTOriented``
appends, they must never enter a ``state_dict``, and the parameter increment is derived
from the patch embedding rather than assumed.
"""
import inspect
import math

import pytest
import torch

from model.cylindrical_vit_oriented import azimuth_channels
from model.simple_vit import SimpleViT
from model.simple_vit_oriented import SimpleViTOriented
from model.xRIR import xRIR
from model.xRIR_simple_oriented import xRIR_SimpleOriented
from treble_multi_room_dataset.treble_xRIR_dataset import convert_equirect_to_camera_coord

SMALL = dict(image_size=(32, 512), patch_size=(16, 32), dim=32, depth=1, heads=2, mlp_dim=32)


def encoder(cls=SimpleViTOriented, **kwargs):
    torch.manual_seed(11)
    return cls(**dict(SMALL, **kwargs)).eval()


def panorama(h=32, w=512, batch=2):
    torch.manual_seed(12)
    depth = convert_equirect_to_camera_coord(torch.rand(h, w) + 1, h, w)
    return depth.permute(2, 0, 1)[None].expand(batch, -1, -1, -1).contiguous()


def test_azimuth_planes_are_the_camera_horizontal_look_direction():
    model = encoder()
    expected = azimuth_channels(32, 512)
    assert torch.equal(model.az_channels, expected)
    xyz = convert_equirect_to_camera_coord(torch.ones(32, 512), 32, 512)
    horizontal = xyz[..., :2] / torch.linalg.vector_norm(xyz[..., :2], dim=-1, keepdim=True)
    torch.testing.assert_close(model.az_channels, horizontal.permute(2, 0, 1),
                               atol=1e-6, rtol=0)
    theta = (torch.arange(512, dtype=torch.float32) + 0.5) * (2 * math.pi / 512) - math.pi
    torch.testing.assert_close(model.az_channels[0, 0], theta.cos(), atol=1e-6, rtol=0)
    torch.testing.assert_close(model.az_channels[1, 0], theta.sin(), atol=1e-6, rtol=0)


def test_planes_are_not_state_and_the_signature_is_the_parents():
    model = encoder()
    assert 'az_channels' not in model.state_dict()
    assert 'az_channels' in dict(model.named_buffers())          # registered, not persisted
    assert model.state_dict().keys() == SimpleViT(**SMALL).state_dict().keys()
    parent = inspect.signature(SimpleViT).parameters
    child = inspect.signature(SimpleViTOriented).parameters
    assert [(k, v.default) for k, v in parent.items()] == [(k, v.default) for k, v in child.items()]


def test_forward_is_the_parent_on_the_concatenated_five_channel_panorama():
    model = encoder()
    image = panorama()
    assert model(image).shape == (2, 32, 32)
    azimuth = model.az_channels.expand(image.shape[0], -1, -1, -1)
    reference = SimpleViT.forward(model, torch.cat([image, azimuth], dim=1))
    assert torch.equal(model(image), reference)
    assert model.to_patch_embedding[1].normalized_shape == (5 * 16 * 32,)


@pytest.mark.parametrize('kwargs', [{}, {'patch_size': (8, 16), 'dim': 64}])
def test_exact_parameter_delta(kwargs):
    counts = [sum(p.numel() for p in encoder(cls, **kwargs).parameters())
              for cls in (SimpleViT, SimpleViTOriented)]
    p1, p2 = kwargs.get('patch_size', SMALL['patch_size'])
    dim = kwargs.get('dim', SMALL['dim'])
    assert counts[1] - counts[0] == 2 * p1 * p2 * dim + 2 * (5 - 3) * p1 * p2


def test_encoder_increment_at_the_recipe_tier():
    """The M-tier increment, derived from the patch embedding rather than assumed."""
    with torch.random.fork_rng(devices=[]):
        counts = [sum(p.numel() for p in cls(num_channels=8).source_network.parameters())
                  for cls in (xRIR, xRIR_SimpleOriented)]
    assert counts[1] - counts[0] == 2 * 16 * 32 * 512 + 2 * 2 * 16 * 32 == 526336


def test_model_swaps_only_the_source_network_and_guards_the_pool():
    model = xRIR_SimpleOriented(num_channels=2, dim=32, depth=1, heads=2, mlp_dim=32,
                                image_size=(32, 512), patch_size=(16, 32), intermediate_ch=32)
    assert isinstance(model, xRIR)
    assert isinstance(model.source_network, SimpleViTOriented)
    with pytest.raises(ValueError, match='lin_proj_0'):
        xRIR_SimpleOriented(num_channels=2, dim=32, depth=1, heads=2, mlp_dim=32,
                            image_size=(32, 512), patch_size=(16, 64), intermediate_ch=32)
