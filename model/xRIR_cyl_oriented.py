"""Exp_06 factory; pinned backbones retain their original classes and behavior."""
from model.cylindrical_vit_oriented import CylindricalViTOriented
from model.xRIR import xRIR
from model.xRIR_cyl import BACKBONES


class xRIR_CylOriented(xRIR):
    def __init__(self, num_channels, n_bins=310, dim=512, intermediate_ch=256,
                 image_size=(256, 512), patch_size=(16, 32), depth=12, mlp_dim=512, heads=8):
        super().__init__(num_channels, n_bins=n_bins, dim=dim, intermediate_ch=intermediate_ch,
                         image_size=image_size, patch_size=patch_size, depth=depth,
                         mlp_dim=mlp_dim, heads=heads)
        self.source_network = CylindricalViTOriented(
            in_channels=3, image_size=tuple(image_size), patch_size=tuple(patch_size),
            dim=dim, depth=depth, heads=heads, dim_head=64, mlp_dim=mlp_dim,
        )
        num_tokens = self.source_network.num_tokens
        if num_tokens != intermediate_ch:
            raise ValueError(
                f'lin_proj_0 pools over the token axis and expects {intermediate_ch} tokens, '
                f'but CylindricalViTOriented produces {num_tokens} for image_size={image_size}, '
                f'patch_size={patch_size}')


BACKBONES_EXP06 = {**BACKBONES, 'cylindrical_oriented': xRIR_CylOriented}


def build_xrir_exp06(backbone, num_shot, **kwargs):
    """Build a pinned backbone or the orientation-conditioned cylindrical model."""
    if backbone not in BACKBONES_EXP06:
        raise ValueError(f'unknown backbone {backbone!r}; choose from {sorted(BACKBONES_EXP06)}')
    return BACKBONES_EXP06[backbone](num_channels=num_shot, **kwargs)
