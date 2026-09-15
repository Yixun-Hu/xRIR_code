"""Keep the pinned factory routes exact while adding the oriented encoder."""
import pytest
import torch

from model.cylindrical_vit_oriented import CylindricalViTOriented
from model.xRIR import xRIR
from model.xRIR_cyl import BACKBONES, build_xrir, xRIR_Cyl
from model.xRIR_cyl_oriented import BACKBONES_EXP06, build_xrir_exp06, xRIR_CylOriented


@pytest.mark.parametrize('name,cls', [('simple', xRIR), ('cylindrical', xRIR_Cyl)])
@torch.no_grad()
def test_pinned_factory_identity_and_state(name, cls):
    torch.manual_seed(60)
    baseline = build_xrir(name, 8).eval()
    torch.manual_seed(60)
    routed = build_xrir_exp06(name, 8).eval()
    assert type(routed) is cls
    assert BACKBONES_EXP06[name] is BACKBONES[name]
    left, right = baseline.state_dict(), routed.state_dict()
    assert left.keys() == right.keys()
    assert all(left[k].shape == right[k].shape and torch.equal(left[k], right[k]) for k in left)
    batch = torch.randn(1, 3, 256, 512)
    assert torch.equal(baseline.source_network(batch), routed.source_network(batch))


def test_parameter_delta_and_factory_errors():
    counts = [sum(p.numel() for p in build_xrir_exp06(name, 8).parameters())
              for name in ('cylindrical', 'cylindrical_oriented')]
    assert counts[1] - counts[0] == 526336
    with pytest.raises(ValueError, match='unknown backbone'):
        build_xrir_exp06('missing', 8)


def test_encoder_kwargs_and_pool_constraint():
    model = build_xrir_exp06('cylindrical_oriented', 8, dim=32, depth=2, heads=2, mlp_dim=48)
    assert type(model) is xRIR_CylOriented
    enc = model.source_network
    assert type(enc) is CylindricalViTOriented
    assert enc.dim == 32 and len(enc.transformer.layers) == 2
    attention, feedforward = enc.transformer.layers[0]
    assert attention.heads == 2 and feedforward.net[1].out_features == 48
    with pytest.raises(ValueError, match='tokens'):
        build_xrir_exp06('cylindrical_oriented', 8, dim=32, depth=1, image_size=(32, 512))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='full pinned forward requires CUDA')
@pytest.mark.parametrize('name', ['simple', 'cylindrical', 'cylindrical_oriented'])
@torch.no_grad()
def test_full_forward_shapes_on_gpu(name):
    model = build_xrir_exp06(name, 8, dim=32, depth=1, heads=2, mlp_dim=48).cuda().eval()
    batch = [torch.randn(*shape, device='cuda') for shape in
             [(1, 3, 256, 512), (1, 8, 9600), (1, 3), (1, 8, 3), (1, 1, 9600)]]
    prediction, target = model(*batch)
    assert prediction.shape == (1, 63, 310, 1)
    assert target.shape == (1, 1, 63, 310)
