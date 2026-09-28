"""exp_11's registry extends exp_06's without touching it (plan v3 section 2.3).

``BACKBONES_EXP06`` is part of every completed exp_06/exp_09 child's evidence: its
digest is recorded in each ``provenance.json`` and re-derived at finalisation, so it may
never be mutated. ``BACKBONES_EXP11`` is a new mapping whose inherited routes are the
very same classes and whose outputs are bit-identical.
"""
import pytest
import torch

from model.xRIR import xRIR
from model.xRIR_cyl import xRIR_Cyl
from model.xRIR_cyl_oriented import BACKBONES_EXP06, xRIR_CylOriented
from model.xRIR_simple_adapter import xRIR_SimpleAdapter
from model.xRIR_simple_oriented import xRIR_SimpleOriented
from model.xrir_exp11_registry import BACKBONES_EXP11, build_xrir_exp11, registry_sha256
from tools import exp06_finalize, exp06_haa_finetune, exp06_train

# The digest every completed exp_06 and exp_09 child recorded; exp_11 may not move it.
EXP06_REGISTRY_SHA256 = 'f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169'
SMALL = dict(dim=32, depth=1, heads=2, mlp_dim=32, intermediate_ch=32,
             image_size=(32, 512), patch_size=(16, 32))


def test_exp06_registry_and_its_digest_are_untouched():
    assert dict(BACKBONES_EXP06) == {'simple': xRIR, 'cylindrical': xRIR_Cyl,
                                     'cylindrical_oriented': xRIR_CylOriented}
    for module in (exp06_finalize, exp06_train, exp06_haa_finetune):
        assert module.registry_sha256() == EXP06_REGISTRY_SHA256


def test_registry_extends_exp06_with_exactly_the_two_new_routes():
    assert BACKBONES_EXP11 == dict(BACKBONES_EXP06, simple_oriented=xRIR_SimpleOriented,
                                   simple_adapter=xRIR_SimpleAdapter)
    assert set(BACKBONES_EXP11) - set(BACKBONES_EXP06) == {'simple_oriented', 'simple_adapter'}
    for name in BACKBONES_EXP06:
        assert BACKBONES_EXP11[name] is BACKBONES_EXP06[name]
    assert registry_sha256() != EXP06_REGISTRY_SHA256


@pytest.mark.parametrize('name', sorted(BACKBONES_EXP06))
def test_inherited_routes_build_the_pinned_class_and_agree_bit_for_bit(name):
    from model.xRIR_cyl_oriented import build_xrir_exp06
    torch.manual_seed(11)
    pinned = build_xrir_exp06(name, 2, **SMALL).eval()
    torch.manual_seed(11)
    through = build_xrir_exp11(name, 2, **SMALL).eval()
    assert type(through) is type(pinned) is BACKBONES_EXP06[name]
    torch.manual_seed(12)
    image = torch.randn(2, 3, 32, 512) / 5.0
    with torch.no_grad():
        assert torch.equal(pinned.source_network(image), through.source_network(image))


def test_new_routes_and_refusals():
    assert type(build_xrir_exp11('simple_oriented', 2, **SMALL)) is xRIR_SimpleOriented
    assert type(build_xrir_exp11('simple_adapter', 2, **SMALL)) is xRIR_SimpleAdapter
    assert build_xrir_exp11('simple_adapter', 3, **SMALL).num_channels == 3
    with pytest.raises(ValueError, match='unknown backbone'):
        build_xrir_exp11('cylindrical_adapter', 2, **SMALL)
