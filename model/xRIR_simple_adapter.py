"""xRIR with a zero-initialised circular azimuth adapter on the pinned SimpleViT.

Plan v3 section 2.3 (exp_11 arms J and K). The pinned :class:`~model.simple_vit.SimpleViT`
is **wrapped**, not edited or copied: :class:`SimpleViTAdapter` holds one as ``self.vit``
and repeats its three forward steps, adding a learned code beside the positional code --
after the patch embedding's final LayerNorm, so the geometry normalisation the round-1
review measured is untouched.

The code is ``Linear(2 -> dim)`` applied to the heading-relative azimuth
``(cos(theta_j - phi), sin(theta_j - phi))`` of each token column, broadcast over the
token rows; ``theta_j = 2 pi (j + 1/2) / W_tok - pi`` and ``phi`` is the bound heading in
radians. Weight **and** bias start at zero, so at initialisation the model's function is
bit-exact to the base checkpoint it inherits (1 536 added parameters at ``dim`` 512).

Interpretation, fixed by the plan: with one heading for every room,
``W a_j(phi) + b = W R(-phi) a_j(0) + b``, so the heading is absorbable into the learned
projection. J/K test an *added learnable circular positional representation during
fine-tuning*, not the benefit of measured heading information.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from model.simple_vit import SimpleViT, pair
from model.xRIR import xRIR

ADAPTER_INPUT = 2
BASE_PREFIX = 'source_network.'
WRAPPED_PREFIX = 'source_network.vit.'
ADAPTER_KEYS = ('source_network.heading_proj.bias', 'source_network.heading_proj.weight')


def token_azimuth(columns, dtype=torch.float32):
    """The pinned token-angle convention: ``theta_j = 2 pi (j + 1/2) / W - pi``."""
    return (torch.arange(columns, dtype=dtype) + 0.5) * (2.0 * math.pi / columns) - math.pi


class SimpleViTAdapter(nn.Module):
    """A pinned SimpleViT plus a zero-initialised heading-relative azimuth code."""

    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3,
                 dim_head=64):
        super().__init__()
        self.vit = SimpleViT(image_size, patch_size, dim, depth, heads, mlp_dim,
                             channels=channels, dim_head=dim_head)
        self.heading_proj = nn.Linear(ADAPTER_INPUT, dim)
        nn.init.zeros_(self.heading_proj.weight)
        nn.init.zeros_(self.heading_proj.bias)
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.token_rows = image_height // patch_height
        self.token_columns = image_width // patch_width
        self.register_buffer('token_theta', token_azimuth(self.token_columns),
                             persistent=False)
        self.phi_deg = None

    def set_heading(self, phi_deg):
        """Install the shared loudspeaker heading in degrees; anything else is refused."""
        if type(phi_deg) not in (int, float) or not math.isfinite(phi_deg):
            raise ValueError('the azimuth adapter needs a finite heading in degrees, '
                             'not {!r}'.format(phi_deg))
        self.phi_deg = float(phi_deg)
        return self

    def heading_code(self, device, dtype):
        """``[tokens, dim]``: one column code per token, repeated down the token rows."""
        if self.phi_deg is None:
            raise ValueError('the azimuth adapter has no heading: call set_heading(phi_deg) '
                             'with the bound record before running the model')
        angle = self.token_theta.to(device=device, dtype=dtype) - math.radians(self.phi_deg)
        cue = torch.stack((angle.cos(), angle.sin()), dim=-1)
        column = self.heading_proj(cue)                      # [W_tok, dim]
        return column[None].expand(self.token_rows, -1, -1).reshape(-1, column.shape[-1])

    def forward(self, img):
        """The pinned SimpleViT's three steps, with the cue added beside the position code."""
        x = self.vit.to_patch_embedding(img)
        code = self.heading_code(x.device, x.dtype)
        return self.vit.transformer(x + (self.vit.pos_embedding.to(x.device, dtype=x.dtype)
                                         + code))


class xRIR_SimpleAdapter(xRIR):
    """The pinned xRIR with the adapter-wrapped SimpleViT as its geometry encoder."""

    def __init__(self, num_channels, n_bins=310, dim=512, intermediate_ch=256,
                 image_size=(256, 512), patch_size=(16, 32), depth=12, mlp_dim=512, heads=8):
        super().__init__(num_channels, n_bins=n_bins, dim=dim, intermediate_ch=intermediate_ch,
                         image_size=image_size, patch_size=patch_size, depth=depth,
                         mlp_dim=mlp_dim, heads=heads)
        self.source_network = SimpleViTAdapter(
            tuple(image_size), tuple(patch_size), dim, depth, heads, mlp_dim, channels=3)
        num_tokens = self.source_network.token_rows * self.source_network.token_columns
        if num_tokens != intermediate_ch:
            raise ValueError(
                f'lin_proj_0 pools over the token axis and expects {intermediate_ch} tokens, '
                f'but SimpleViTAdapter produces {num_tokens} for image_size={image_size}, '
                f'patch_size={patch_size}')

    def set_heading(self, phi_deg):
        """Install the shared heading the entry point validated across every room."""
        self.source_network.set_heading(phi_deg)
        return self

    def zero_adapter(self):
        """Return the adapter to its initialisation, where the model is the base model."""
        with torch.no_grad():
            self.source_network.heading_proj.weight.zero_()
            self.source_network.heading_proj.bias.zero_()

    def load_base_checkpoint(self, state):
        """Mode (i): initialise from a pinned xRIR checkpoint; the adapter stays zero.

        Every inherited key is loaded strictly -- the wrapped encoder's parameters carry
        the ``source_network.vit.`` prefix here and ``source_network.`` in the base
        checkpoint, so they are remapped, and nothing else is renamed. A checkpoint that
        already carries adapter state is a later stage's, never an initialisation, and is
        refused rather than loaded with a re-zeroed adapter.
        """
        carried = sorted(key for key in state
                         if key in ADAPTER_KEYS or key.startswith(WRAPPED_PREFIX))
        if carried:
            raise ValueError('this checkpoint already carries adapter state ({}); load it '
                             'with load_state_dict(state, strict=True)'.format(
                                 ', '.join(carried[:4])))
        remapped = {(WRAPPED_PREFIX + key[len(BASE_PREFIX):] if key.startswith(BASE_PREFIX)
                     else key): value for key, value in state.items()}
        self.zero_adapter()
        current = super().state_dict()
        remapped.update({key: current[key] for key in ADAPTER_KEYS})
        return nn.Module.load_state_dict(self, remapped, strict=True)

    def load_state_dict(self, state_dict, strict=True):
        """Mode (ii): a stage or evaluation checkpoint, with its whole adapter state."""
        if strict is not True:
            raise ValueError('an exp_11 adapter checkpoint is loaded strictly; strict={!r} '
                             'would admit a model whose adapter nobody trained'.format(strict))
        missing = [key for key in ADAPTER_KEYS if key not in state_dict]
        if missing:
            raise ValueError('this checkpoint records no adapter state ({}); a base xRIR '
                             'checkpoint is loaded with load_base_checkpoint(state)'.format(
                                 ', '.join(missing)))
        return nn.Module.load_state_dict(self, state_dict, strict=True)
