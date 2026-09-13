"""Frozen exp_05 capacities; the exp_03-pinned model factory remains unchanged."""
from types import MappingProxyType

from model.xRIR_cyl import build_xrir


TIERS = MappingProxyType({name: MappingProxyType(config) for name, config in {
    'S': dict(dim=256, depth=6, heads=4, mlp_dim=256),
    'M': dict(dim=512, depth=12, heads=8, mlp_dim=512),
    'L': dict(dim=768, depth=12, heads=12, mlp_dim=768),
}.items()})


def build_tier(backbone, tier, num_shot=8):
    """Use the registered kwargs, retaining the factory's dim_head=64."""
    return build_xrir(backbone, num_shot, **TIERS[tier])


def count_parameters(model):
    encoder = sum(parameter.numel() for parameter in model.source_network.parameters())
    full = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return dict(encoder=encoder, full=full, trainable=trainable, non_encoder=full - encoder)


def tier_of(args_dict):
    """Require all four recorded fields as native integers; never infer legacy M."""
    for tier, config in TIERS.items():
        if all(type(args_dict.get('vit_' + key)) is int and args_dict['vit_' + key] == value
               for key, value in config.items()):
            return tier
    raise ValueError('vit_* fields do not match a registered tier')


def matching_ratio(tier):
    """Fractional encoder overhead of cylindrical relative to simple."""
    simple = count_parameters(build_tier('simple', tier))['encoder']
    cylindrical = count_parameters(build_tier('cylindrical', tier))['encoder']
    return (cylindrical - simple) / simple
