"""exp_11 arms J/K: the zero-initialised azimuth adapter around the pinned SimpleViT.

Plan v3 section 2.3 and Codex round-2 change 1: the wrapped encoder is the pinned class
itself (composition, never a modified copy); the added ``Linear(2 -> dim)`` has zero
weight *and* zero bias, so the model's function at initialisation is bit-exact to the
base checkpoint it starts from; the token angle is ``theta_j = 2 pi (j + 1/2) / 16 - pi``
and the cue is ``(cos(theta_j - phi), sin(theta_j - phi))`` broadcast over the token rows;
and both loading modes are strict -- never ``strict=False``, never a silently re-zeroed
adapter.
"""
import math

import pytest
import torch

from model.simple_vit import SimpleViT
from model.xRIR import xRIR
from model.xRIR_simple_adapter import (ADAPTER_KEYS, BASE_PREFIX, WRAPPED_PREFIX,
                                       SimpleViTAdapter, token_azimuth, xRIR_SimpleAdapter)

SMALL = dict(num_channels=2, dim=32, depth=1, heads=2, mlp_dim=32, intermediate_ch=32,
             image_size=(32, 512), patch_size=(16, 32))
PHI = -90.0


def pair_of_models():
    """One base xRIR and one adapter model carrying exactly its inherited weights."""
    torch.manual_seed(11)
    base = xRIR(**SMALL).eval()
    adapter = xRIR_SimpleAdapter(**SMALL).eval()
    adapter.load_base_checkpoint(base.state_dict())
    return base, adapter


def geometry(b=2):
    """The encoding ``xRIR.forward`` feeds its source_network: ``(loc - depth)/5``.

    The pinned ``xRIR.forward`` calls ``.cuda()`` unconditionally (``apply_delay``), so
    the CPU comparison is made at the geometry encoder -- the one component the adapter
    replaces; everything downstream is the inherited module and is compared by state.
    """
    torch.manual_seed(12)
    return (torch.randn(b, 3, 1, 1) - torch.rand(b, 3, 32, 512) * 5.0) / 5.0


def test_token_angle_convention():
    theta = token_azimuth(16)
    assert theta.shape == (16,)
    expected = torch.tensor([2 * math.pi * (j + 0.5) / 16 - math.pi for j in range(16)])
    torch.testing.assert_close(theta, expected.float(), atol=1e-6, rtol=0)


def test_the_wrapped_encoder_is_the_pinned_class_and_the_adapter_is_zero():
    model = xRIR_SimpleAdapter(**SMALL)
    assert type(model.source_network) is SimpleViTAdapter
    assert type(model.source_network.vit) is SimpleViT
    assert torch.equal(model.source_network.heading_proj.weight,
                       torch.zeros_like(model.source_network.heading_proj.weight))
    assert torch.equal(model.source_network.heading_proj.bias,
                       torch.zeros_like(model.source_network.heading_proj.bias))
    assert sorted(key for key in model.state_dict() if 'heading_proj' in key) == list(ADAPTER_KEYS)


def test_adapter_parameter_count_at_dim_512():
    encoder = SimpleViTAdapter((256, 512), (16, 32), 512, 1, 8, 512)
    added = sum(p.numel() for p in encoder.heading_proj.parameters())
    assert added == 2 * 512 + 512 == 1536
    pinned = SimpleViT((256, 512), (16, 32), 512, 1, 8, 512)
    assert (sum(p.numel() for p in encoder.parameters())
            - sum(p.numel() for p in pinned.parameters())) == 1536


def test_function_is_bit_exact_to_the_base_model_at_initialisation():
    base, adapter = pair_of_models()
    adapter.set_heading(PHI)
    image = geometry()
    with torch.no_grad():
        assert torch.equal(base.source_network(image), adapter.source_network(image))
    inherited = {key: value for key, value in adapter.state_dict().items()
                 if key not in ADAPTER_KEYS}
    base_state = base.state_dict()
    assert sorted(inherited) == sorted(
        key.replace(BASE_PREFIX, WRAPPED_PREFIX, 1) if key.startswith(BASE_PREFIX) else key
        for key in base_state)
    for key, value in base_state.items():
        moved = key.replace(BASE_PREFIX, WRAPPED_PREFIX, 1) if key.startswith(BASE_PREFIX) else key
        assert torch.equal(inherited[moved], value)


def test_the_cue_is_the_heading_relative_column_azimuth_broadcast_over_rows():
    torch.manual_seed(13)
    encoder = SimpleViTAdapter((32, 512), (16, 32), 8, 1, 2, 8).eval()
    with torch.no_grad():
        encoder.heading_proj.weight.copy_(torch.eye(8, 2))
        encoder.heading_proj.bias.fill_(0.25)
    encoder.set_heading(PHI)
    code = encoder.heading_code(torch.device('cpu'), torch.float32)
    assert code.shape == (2 * 16, 8)
    angle = token_azimuth(16) - math.radians(PHI)
    for j in range(16):
        for row in range(2):
            token = code[row * 16 + j]
            torch.testing.assert_close(token[0], angle[j].cos() + 0.25, atol=1e-6, rtol=0)
            torch.testing.assert_close(token[1], angle[j].sin() + 0.25, atol=1e-6, rtol=0)
            assert torch.equal(token[2:], torch.full((6,), 0.25))


def test_a_forward_without_a_heading_is_refused():
    model = xRIR_SimpleAdapter(**SMALL).eval()
    assert model.source_network.phi_deg is None
    with pytest.raises(ValueError, match='heading'):
        model.source_network(geometry())
    for value in (None, True, float('nan'), '0'):
        with pytest.raises(ValueError, match='heading'):
            model.set_heading(value)
    assert model.set_heading(PHI) is model and model.source_network.phi_deg == PHI


def test_base_loading_is_strict_and_leaves_the_adapter_zero():
    base, adapter = pair_of_models()
    state = base.state_dict()
    assert torch.equal(adapter.source_network.vit.to_patch_embedding[2].weight,
                       state['source_network.to_patch_embedding.2.weight'])
    assert not adapter.source_network.heading_proj.weight.any()
    with pytest.raises(RuntimeError):
        adapter.load_base_checkpoint({k: v for k, v in state.items() if 'audio_enc' not in k})
    with pytest.raises(RuntimeError):
        adapter.load_base_checkpoint(dict(state, unexpected=torch.zeros(1)))


def test_base_loading_refuses_a_checkpoint_that_already_carries_adapter_state():
    _, adapter = pair_of_models()
    with pytest.raises(ValueError, match='load_state_dict'):
        adapter.load_base_checkpoint(adapter.state_dict())


def test_stage_checkpoints_require_the_whole_adapter_state_strictly():
    _, adapter = pair_of_models()
    with torch.no_grad():
        adapter.source_network.heading_proj.weight.fill_(0.5)
    trained = {key: value.clone() for key, value in adapter.state_dict().items()}
    fresh = xRIR_SimpleAdapter(**SMALL)
    fresh.load_state_dict(trained, strict=True)
    assert torch.equal(fresh.source_network.heading_proj.weight,
                       trained['source_network.heading_proj.weight'])
    with pytest.raises(ValueError, match='strict'):
        fresh.load_state_dict(trained, strict=False)
    for key in ADAPTER_KEYS:
        with pytest.raises(ValueError, match='adapter'):
            fresh.load_state_dict({k: v for k, v in trained.items() if k != key}, strict=True)


def test_pool_guard():
    with pytest.raises(ValueError, match='lin_proj_0'):
        xRIR_SimpleAdapter(**dict(SMALL, patch_size=(16, 64)))
