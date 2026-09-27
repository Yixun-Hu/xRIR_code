"""SimpleViT geometry tokens with the explicit heading-frame azimuth planes (exp_11).

The pinned :class:`~model.simple_vit.SimpleViT` is subclassed, never edited: the only
change is that it is built for five input channels and that ``forward`` concatenates the
two constant azimuth planes ``cos theta_c, sin theta_c`` of the camera-frame column
azimuth to the xyz panorama before the pinned forward runs. Those planes are exactly the
ones :class:`~model.cylindrical_vit_oriented.CylindricalViTOriented` appends (they are a
pure function of the column index and vary within a patch), so exp_11's arm H is the
SimpleViT reading of exp_06's explicit orientation cue and nothing else.

They are a non-persistent buffer: constants derived from the column grid are not
learned state and must never travel in a checkpoint.
"""
from __future__ import annotations

import torch

from model.cylindrical_vit_oriented import azimuth_channels
from model.simple_vit import SimpleViT, pair

AZIMUTH_CHANNELS = 2


class SimpleViTOriented(SimpleViT):
    """``SimpleViT(channels + 2)`` on ``[xyz, cos theta_c, sin theta_c]``."""

    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3,
                 dim_head=64):
        super().__init__(image_size, patch_size, dim, depth, heads, mlp_dim,
                         channels=channels + AZIMUTH_CHANNELS, dim_head=dim_head)
        image_height, image_width = pair(image_size)
        self.register_buffer('az_channels', azimuth_channels(image_height, image_width),
                             persistent=False)

    def forward(self, img):
        """Encode a camera-frame xyz panorama ``[B, 3, H, W]`` into ``[B, tokens, dim]``."""
        azimuth = self.az_channels.to(img.dtype).expand(img.shape[0], -1, -1, -1)
        return super().forward(torch.cat([img, azimuth], dim=1))
