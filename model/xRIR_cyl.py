"""xRIR with the SimpleViT geometry encoder replaced by CylindricalViT.

Parallel to ``model/xRIR.py``: the forward pass, geometry/time projections,
audio encoder, attention over references and spectrogram head are inherited
unchanged. Only ``source_network`` (the ViT run on every ``(loc - depth_coord)/5``
panorama) is swapped, so any difference in results is attributable to the
geometry encoder alone.

``CylindricalViT`` returns ``[B, num_tokens, dim]`` exactly like ``SimpleViT``;
for the default (256, 512) panorama with (16, 32) patches that is 256 tokens,
which is what the baseline's ``lin_proj_0`` (a 256 -> 1 linear pool over the
token axis) expects.
"""
from model.cylindrical_vit import CylindricalViT
from model.xRIR import xRIR


class xRIR_Cyl(xRIR):
    def __init__(self, num_channels, n_bins=310, dim=512, intermediate_ch=256,
                 image_size=(256, 512), patch_size=(16, 32), depth=12, mlp_dim=512, heads=8):
        super().__init__(num_channels, n_bins=n_bins, dim=dim, intermediate_ch=intermediate_ch,
                         image_size=image_size, patch_size=patch_size, depth=depth,
                         mlp_dim=mlp_dim, heads=heads)
        # Same dim / depth / heads / dim_head / mlp_dim as the SimpleViT it replaces.
        self.source_network = CylindricalViT(
            in_channels=3, image_size=tuple(image_size), patch_size=tuple(patch_size),
            dim=dim, depth=depth, heads=heads, dim_head=64, mlp_dim=mlp_dim,
        )
        num_tokens = self.source_network.num_tokens
        if num_tokens != intermediate_ch:
            raise ValueError(
                f"lin_proj_0 pools over the token axis and expects {intermediate_ch} tokens, "
                f"but CylindricalViT produces {num_tokens} for image_size={image_size}, "
                f"patch_size={patch_size}")


BACKBONES = {"simple": xRIR, "cylindrical": xRIR_Cyl}


def build_xrir(backbone, num_shot, **kwargs):
    """Construct the baseline (``simple``) or cylindrical-ViT (``cylindrical``) xRIR."""
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}")
    return BACKBONES[backbone](num_channels=num_shot, **kwargs)


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    B, K = 2, 8
    depth = torch.rand(B, 3, 256, 512) * 5.0
    refs = torch.randn(B, K, 9600) * 0.1
    src = torch.randn(B, 3)
    ref_locs = torch.randn(B, K, 3)
    tgt = torch.randn(B, 1, 9600) * 0.1
    for name in BACKBONES:
        m = build_xrir(name, K).cuda().eval()
        n_params = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            out, tgt_spec = m(depth.cuda(), refs.cuda(), src.cuda(), ref_locs.cuda(), tgt.cuda())
        print(f"{name:12s} params={n_params/1e6:.2f}M  out={tuple(out.shape)}  tgt={tuple(tgt_spec.shape)}  "
              f"finite={bool(torch.isfinite(out).all())}")
