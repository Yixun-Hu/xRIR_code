"""xRIR with the SimpleViT geometry encoder replaced by its oriented five-channel form.

Parallel to ``model/xRIR_cyl_oriented.py``: the forward pass, geometry/time projections,
audio encoder, attention over references and spectrogram head are inherited from the
pinned ``xRIR`` unchanged; only ``source_network`` is swapped, so a difference between
arm H and the exp_01 control is attributable to the azimuth planes alone.

``SimpleViT`` exposes no ``num_tokens``, so the pool guard computes the token grid from
the image and patch sizes: ``lin_proj_0`` pools over the token axis and only works while
that product equals ``intermediate_ch``.
"""
from model.simple_vit import pair
from model.simple_vit_oriented import SimpleViTOriented
from model.xRIR import xRIR


class xRIR_SimpleOriented(xRIR):
    def __init__(self, num_channels, n_bins=310, dim=512, intermediate_ch=256,
                 image_size=(256, 512), patch_size=(16, 32), depth=12, mlp_dim=512, heads=8):
        super().__init__(num_channels, n_bins=n_bins, dim=dim, intermediate_ch=intermediate_ch,
                         image_size=image_size, patch_size=patch_size, depth=depth,
                         mlp_dim=mlp_dim, heads=heads)
        self.source_network = SimpleViTOriented(
            tuple(image_size), tuple(patch_size), dim, depth, heads, mlp_dim, channels=3)
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        num_tokens = (image_height // patch_height) * (image_width // patch_width)
        if num_tokens != intermediate_ch:
            raise ValueError(
                f'lin_proj_0 pools over the token axis and expects {intermediate_ch} tokens, '
                f'but SimpleViTOriented produces {num_tokens} for image_size={image_size}, '
                f'patch_size={patch_size}')
