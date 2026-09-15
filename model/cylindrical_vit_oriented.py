"""Cylindrical geometry tokens with azimuth in the supplied heading frame."""
from __future__ import annotations

import math

import torch
from torch import nn
from einops.layers.torch import Rearrange

from model.cylindrical_vit import CylindricalViT


def azimuth_channels(H, W, dtype=torch.float32):
    """Return column-centred (cos theta, sin theta), broadcast to [2, H, W]."""
    theta = (torch.arange(W, dtype=dtype) + 0.5) * (2.0 * math.pi / W) - math.pi
    return torch.stack((theta.cos(), theta.sin()))[:, None].expand(2, H, W)


class CylindricalViTOriented(CylindricalViT):
    """Append heading-frame azimuth after the parent's per-column gauge alignment."""

    def __init__(self, in_channels=3, image_size=(256, 512), patch_size=(16, 32),
                 dim=512, depth=12, heads=8, dim_head=64, mlp_dim=None):
        super().__init__(in_channels=in_channels, image_size=image_size, patch_size=patch_size,
                         dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim)
        p1, p2 = patch_size
        patch_dim = 5 * p1 * p2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p1, p2=p2),
            nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim),
        )
        channels = torch.stack((self.cos_theta, self.sin_theta))[:, None]
        self.register_buffer('az_channels', channels.expand(2, *image_size), persistent=False)

    def forward(self, img):
        """Encode heading-frame xyz [B, 3, H, W] into [B, num_tokens, dim]."""
        cos = self.cos_theta.to(img.dtype)
        sin = self.sin_theta.to(img.dtype)
        x, y, z = img[:, 0], img[:, 1], img[:, 2]
        x_a = x * cos + y * sin
        y_a = -x * sin + y * cos
        aligned = torch.stack([x_a, y_a, z], dim=1)
        azimuth = self.az_channels.to(img.dtype).expand(img.shape[0], -1, -1, -1)
        tokens = self.to_patch_embedding(torch.cat([aligned, azimuth], dim=1))
        tokens = tokens + self.pos_embedding.to(tokens.dtype)
        return self.transformer(tokens, self.rel_index)
