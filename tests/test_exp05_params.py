"""Registered capacities, measured against the frozen exp_01 model factory."""
import pytest
import torch

from tools.exp05_params import TIERS, build_tier, count_parameters, matching_ratio, tier_of


COUNTS = {
    'S': ((2766080, 15073213), (2777984, 15085117)),
    'M': ((19703296, 32075965), (19750912, 32123581)),
    'L': ((43709184, 56147389), (43780608, 56218813)),
}


@pytest.mark.parametrize('tier', TIERS)
@pytest.mark.parametrize('backbone,index', [('simple', 0), ('cylindrical', 1)])
def test_exact_counts_and_encoder_pool(tier, backbone, index):
    model = build_tier(backbone, tier)
    encoder, full = COUNTS[tier][index]
    assert count_parameters(model) == dict(encoder=encoder, full=full,
                                          trainable=full - 20, non_encoder=full - encoder)
    assert all(type(value) is int for value in count_parameters(model).values())
    assert model.num_channels == 8
    assert (256 / 16) * (512 / 32) == 256
    attention = model.source_network.transformer.layers[0][0]
    assert attention.scale == 64 ** -0.5
    assert attention.to_qkv.out_features == 3 * TIERS[tier]['heads'] * 64
    with torch.no_grad():
        tokens = model.source_network(torch.zeros(1, 3, 256, 512))
        assert tokens.shape == (1, 256, TIERS[tier]['dim'])
        pooled = model.lin_proj_0(model.src_proj(tokens).permute(0, 2, 1))
        assert pooled.shape == (1, 256, 1) and torch.isfinite(pooled).all()


@pytest.mark.parametrize('tier', TIERS)
def test_matching_and_round_trip(tier):
    simple, cylindrical = COUNTS[tier]
    assert matching_ratio(tier) == pytest.approx((cylindrical[0] - simple[0]) / simple[0])
    assert 0 <= matching_ratio(tier) <= 0.005
    assert tier_of({'vit_' + key: value for key, value in TIERS[tier].items()}) == tier


@pytest.mark.parametrize('args', [{}, {'vit_dim': 256},
    dict(vit_dim=512, vit_depth=12, vit_heads=8, vit_mlp_dim=256),
    dict(vit_dim=512.0, vit_depth=12, vit_heads=8, vit_mlp_dim=512)])
def test_unknown_or_inexact_tier_refused(args):
    with pytest.raises(ValueError):
        tier_of(args)


def test_tiers_frozen_and_num_shot_forwarded():
    with pytest.raises(TypeError):
        TIERS['S']['dim'] = 512
    with pytest.raises(TypeError):
        TIERS['XL'] = {}
    assert build_tier('simple', 'S', num_shot=1).num_channels == 1
