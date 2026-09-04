"""Cylindrical azimuth-equivariant ViT geometry encoder (transformer blocks).

The building blocks :class:`CylindricalViT` (part 2, row 4) is assembled from: an
elevation-only sinusoidal position embedding and a pre-norm transformer whose attention
adds a per-head *circular* relative-position bias. Bodies/signatures byte-identical to
the pinned source; only T3 (concise docstrings + provenance header) applied -- exact.
"""
import math

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


def posemb_sincos_1d(n: int, dim: int, temperature: int = 10000,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Sinusoidal 1D position embedding for ``n`` elevation rows; ``[n, dim]`` (``dim`` even)."""
    assert (dim % 2) == 0, "feature dimension must be even for sincos emb"
    pos = torch.arange(n)
    omega = torch.arange(dim // 2) / (dim // 2 - 1)
    omega = 1.0 / (temperature ** omega)
    pos = pos[:, None] * omega[None, :]
    pe = torch.cat((pos.sin(), pos.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    """Pre-norm MLP block (matches AGREE's SimpleViT feed-forward)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward block to ``x`` of shape ``[B, N, dim]``."""
        return self.net(x)


class RelPosAttention(nn.Module):
    """Multi-head self-attention; a per-head ``[heads, table_size]`` ``rel_bias`` table is
    indexed by a fixed ``[N, N]`` ``rel_index`` (signed circular azimuth distance), so the
    bias is invariant to a common azimuth shift of all tokens."""

    def __init__(self, dim: int, table_size: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.rel_bias = nn.Parameter(torch.zeros(heads, table_size))

    def forward(self, x: torch.Tensor, rel_index: torch.Tensor) -> torch.Tensor:
        """Attend over ``x`` ``[B, N, dim]`` with ``rel_index`` ``[N, N]`` -> ``[B, N, dim]``."""
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        bias = self.rel_bias[:, rel_index]                 # [heads, N, N]
        dots = dots + bias[None]
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """Pre-norm transformer with circular-relative-bias attention."""

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int,
                 table_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                RelPosAttention(dim, table_size, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim),
            ]))

    def forward(self, x: torch.Tensor, rel_index: torch.Tensor) -> torch.Tensor:
        """Run the transformer over ``x`` of shape ``[B, N, dim]``."""
        for attn, ff in self.layers:
            x = attn(x, rel_index) + x
            x = ff(x) + x
        return self.norm(x)


class CylindricalViT(nn.Module):
    """Azimuth-equivariant ViT over a source-relative geometry panorama ``[B, 3, H, W]``.

    ``forward`` gauge-aligns the horizontal channels per column (``Rz(-theta_c)``), patch-
    embeds, adds elevation-only position encoding, and runs the circular-relative-bias
    :class:`Transformer`; it returns the token sequence ``[B, num_tokens, dim]`` (the caller
    mean-pools it). Equivariant to patch-level azimuth rolls by construction, so token
    outputs permute (up to the azimuth-patch shift) under a 32-column-multiple roll.
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: tuple[int, int] = (256, 512),
        patch_size: tuple[int, int] = (16, 32),
        dim: int = 512,
        depth: int = 12,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int | None = None,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"CylindricalViT expects 3 (x, y, z) channels, got {in_channels}")
        image_height, image_width = image_size
        patch_height, patch_width = patch_size
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            "Image dimensions must be divisible by the patch size."
        mlp_dim = dim if mlp_dim is None else mlp_dim

        self.dim = dim
        self.h_tok = image_height // patch_height
        self.w_tok = image_width // patch_width
        self.num_tokens = self.h_tok * self.w_tok
        patch_dim = in_channels * patch_height * patch_width

        # (a) Per-pixel gauge alignment buffers: theta_c = (c + 0.5) * 2*pi/W - pi.
        cols = torch.arange(image_width, dtype=torch.float32)
        theta = (cols + 0.5) * (2.0 * math.pi / image_width) - math.pi
        self.register_buffer("cos_theta", torch.cos(theta), persistent=False)
        self.register_buffer("sin_theta", torch.sin(theta), persistent=False)

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # (b) Elevation-only absolute position encoding, broadcast across azimuth.
        el_pe = posemb_sincos_1d(self.h_tok, dim)                     # [H_tok, dim]
        pos_embedding = el_pe[:, None, :].expand(self.h_tok, self.w_tok, dim).reshape(self.num_tokens, dim)
        self.register_buffer("pos_embedding", pos_embedding.contiguous(), persistent=False)

        # (b) Circular relative position index into the per-head bias table.
        rel_index, table_size = self._build_rel_index(self.h_tok, self.w_tok)
        self.register_buffer("rel_index", rel_index, persistent=False)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, table_size)

    @staticmethod
    def _build_rel_index(h_tok: int, w_tok: int) -> tuple[torch.Tensor, int]:
        """Build the ``[N, N]`` circular relative-position index + bias table size
        ``(2*h_tok - 1) * w_tok`` (signed circular azimuth distance, Swin-style)."""
        el = torch.arange(h_tok).repeat_interleave(w_tok)            # token elevation, [N] (h-major)
        az = torch.arange(w_tok).repeat(h_tok)                       # token azimuth, [N]
        d_el = el[:, None] - el[None, :] + (h_tok - 1)               # [N, N] in [0, 2*h_tok-2]
        d_az = (az[:, None] - az[None, :] + w_tok // 2) % w_tok      # signed circular, in [0, w_tok-1]
        rel_index = (d_el * w_tok + d_az).long()
        table_size = (2 * h_tok - 1) * w_tok
        return rel_index, table_size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Encode geometry ``[B, 3, H, W]`` -> azimuth-equivariant tokens ``[B, num_tokens, dim]``."""
        cos = self.cos_theta.to(img.dtype)
        sin = self.sin_theta.to(img.dtype)
        x, y, z = img[:, 0], img[:, 1], img[:, 2]                    # [B, H, W] each
        x_a = x * cos + y * sin
        y_a = -x * sin + y * cos
        aligned = torch.stack([x_a, y_a, z], dim=1)                  # [B, 3, H, W]

        tokens = self.to_patch_embedding(aligned)                   # [B, N, dim]
        tokens = tokens + self.pos_embedding.to(tokens.dtype)
        return self.transformer(tokens, self.rel_index)