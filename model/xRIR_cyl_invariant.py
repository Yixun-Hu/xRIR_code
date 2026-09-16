"""exp_08: CylindricalViT + azimuth-invariant readout + intrinsic coordinate conditioning.

Route 2 of ``worklog/worklog_yixun/xrir_invariant_readout_route2_handoff.md``: make a
*single* forward pass of xRIR produce a C16-rotation-invariant RIR, instead of averaging
several rotated inferences (Route 1, excluded by the user).

The physical transformation is a whole-scene yaw about the receiver's vertical axis
(``tools.yaw_rotation.rotate_scene_yaw``): the panorama is rolled by ``k`` columns and the
xyz channels, the query-source and every reference-source position are rotated by
``Rz(2*pi*k/512)``.  With a 512-column panorama and 32-column azimuth patches the
*structural* guarantee is **C16** -- ``k`` a multiple of 32, i.e. multiples of 22.5 deg.
Off-lattice angles are a diagnostic only and must never be reported as the guarantee.

Two paths in the pinned ``model/xRIR.py`` forward have no invariance (handoff section 2):

1. ``lin_proj_0`` -- a learned linear pool over the 256 token *positions*, applied to the
   query-source view, the receiver view and every reference view of the geometry encoder.
   A token-position pool is not invariant to the azimuth permutation the encoder produces.
2. ``src_coord_proj(dist_embedder(...))`` -- a Fourier embedding of the **raw** query and
   reference xyz, which rotates with the scene.

This module fixes exactly those two, and nothing else:

* :class:`InvariantReadout` replaces ``lin_proj_0`` on all three geometry branches
  (handoff section 3.2): mean over azimuth, then a learned combination over the 16
  elevation rows.  The per-token ``src_proj`` is left in place -- it commutes with a token
  permutation.
* :func:`intrinsic_scene_coords` replaces the raw xyz fed to ``dist_embedder`` for the
  query *and* every reference (handoff section 3.3).  The geometry encoder keeps its
  original ``(location - depth_coord) / 5`` input; its semantics are not swapped.

Why the composition is invariant (exact arithmetic)::

    img'(c) = Rz(D) . img(c - k)                     rotated panorama, D = 2*pi*k/512
    aligned'(c) = Rz(-theta_c) Rz(D) img(c-k)
                = Rz(-theta_{c-k}) img(c-k) = aligned(c - k)      (theta_c = theta_{c-k} + D)
    tokens'     = tokens rolled by k/32 token columns  (patch grid, elevation-only pos-emb,
                  circular relative attention bias -- all azimuth-shift equivariant)
    readout(tokens') = readout(tokens)               (mean over azimuth kills the roll)
    a' = Rz(D) a,  p' = Rz(D) p  =>  a'.p' = a.p and  a'_x p'_y - a'_y p'_x = a_x p_y - a_y p_x

and ``shift_and_align`` uses only ``||src||``, ``||ref||`` and ``||src|| - ||ref||``, which
are rotation invariants.  In float32 the residual is bounded by the rounding of the rotated
*inputs* (the model never sees the exactly-rotated scene), not by the architecture.

SimpleViT gets the same two changes in :class:`xRIR_SimpleInvariant` -- the 2x2 control arm
of handoff section 5.  It carries **no** invariance guarantee: its tokens are not azimuth
equivariant (absolute 2-D position embedding, no gauge alignment), so the invariant readout
only removes one of the two failure modes there.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from model.cylindrical_vit import CylindricalViT
from model.xRIR import xRIR


#: Horizontal radius (metres) at or below which a position counts as "on the vertical axis".
#: Explicit and testable, as the handoff requires; the basis rule below branches on it.
DEFAULT_BASIS_EPS = 1.0e-6

#: ``mode`` codes returned by :func:`horizontal_basis`.
BASIS_QUERY = 0        # basis taken from the query source's horizontal direction
BASIS_REFERENCE = 1    # query is (near-)vertical: basis from the widest reference
BASIS_DEGENERATE = 2   # every horizontal component is (near-)zero: horizontal output is 0


class InvariantReadout(nn.Module):
    """Azimuth-invariant replacement for ``xRIR.lin_proj_0`` (handoff section 3.2).

    Takes the geometry tokens ``[B, N, C]`` produced by ``src_proj(source_network(...))``,
    views them on the ``(elevation, azimuth)`` patch grid in the encoders' **h-major**
    order (``Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)")`` => ``n = h * W_tok + w``)
    and pools::

        m[b, h, c] = mean_w  F[b, h, w, c]
        p[b, c]    = sum_h   a[h] * m[b, h, c] + bias

    An azimuth roll permutes ``w`` cyclically, so ``m`` -- and therefore the output -- is
    unchanged, while the elevation structure is kept.  The output is ``[B, 1, C]``, exactly
    the shape ``lin_proj_0(...).squeeze(-1).unsqueeze(1)`` produced, so the fusion head is
    untouched.

    The azimuth mean is accumulated in float64 and cast back.  A float32 mean is
    order-dependent at the 1-ulp level, and a roll *is* a change of summation order, so a
    float32 accumulator would leak a ~1e-7 "rotation effect" that has nothing to do with the
    architecture; float64 accumulation makes the readout bitwise roll-invariant in practice.

    Args:
        h_tok: number of elevation rows in the token grid (16 for 256x512 / 16x32).
        w_tok: number of azimuth columns in the token grid (16 for the same).
    """

    def __init__(self, h_tok: int = 16, w_tok: int = 16) -> None:
        super().__init__()
        if int(h_tok) < 1 or int(w_tok) < 1:
            raise ValueError("token grid must be at least 1x1, got ({}, {})".format(h_tok, w_tok))
        self.h_tok = int(h_tok)
        self.w_tok = int(w_tok)
        self.weight = nn.Parameter(torch.empty(self.h_tok))
        self.bias = nn.Parameter(torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Uniform ``+-1/sqrt(h_tok)`` -- ``nn.Linear``'s convention with ``h_tok`` inputs.

        The pool now has ``h_tok`` effective inputs (the azimuth mean already averages the
        other axis), not ``h_tok * w_tok``; the bound keeps the pooled activation at roughly
        the scale ``lin_proj_0``'s default init produced on azimuth-correlated tokens.
        """
        bound = 1.0 / math.sqrt(self.h_tok)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool geometry tokens ``[B, N, C]`` -> ``[B, 1, C]``, invariant to azimuth rolls."""
        if tokens.dim() != 3:
            raise ValueError("expected tokens [B, N, C], got shape {}".format(tuple(tokens.shape)))
        b, n, c = tokens.shape
        if n != self.h_tok * self.w_tok:
            raise ValueError("expected {} tokens ({}x{}), got {}".format(
                self.h_tok * self.w_tok, self.h_tok, self.w_tok, n))
        grid = tokens.reshape(b, self.h_tok, self.w_tok, c)
        azim_mean = grid.mean(dim=2, dtype=torch.float64).to(tokens.dtype)   # [B, H_tok, C]
        pooled = (azim_mean * self.weight.to(tokens.dtype).view(1, self.h_tok, 1)).sum(dim=1)
        return (pooled + self.bias.to(tokens.dtype)).unsqueeze(1)            # [B, 1, C]

    @classmethod
    def from_token_pool(cls, pool, h_tok: int = 16, w_tok: int = 16) -> "InvariantReadout":
        """Warm-start from the pinned ``lin_proj_0`` (handoff section 3.2).

        With ``W`` the old ``[1, H_tok * W_tok]`` pooling weight reshaped to
        ``[H_tok, W_tok]``, set ``a[h] = sum_w W[h, w]`` and carry the bias.  Because the new
        readout *averages* over azimuth, the effective per-token weight becomes
        ``mean_w W[h, w]``: the azimuth-averaged old readout.  If the old weights happened to
        be constant along azimuth this reproduces them exactly; in general it is an
        *initialisation* only -- it does not reproduce the old model's single-angle output and
        does not promise the old accuracy (handoff section 3.2, last paragraph).

        Args:
            pool: the old ``basic_project2``/``nn.Linear`` (``weight [1, N]``, ``bias [1]``),
                or a ``(weight, bias)`` pair of tensors.
            h_tok: elevation rows of the token grid.
            w_tok: azimuth columns of the token grid.

        Returns:
            A new :class:`InvariantReadout` on CPU, in the weight's dtype.
        """
        weight, bias = _unpack_token_pool(pool)
        if weight.dim() != 2 or weight.shape[0] != 1:
            raise ValueError("lin_proj_0 weight must be [1, N], got {}".format(tuple(weight.shape)))
        if weight.shape[1] != int(h_tok) * int(w_tok):
            raise ValueError("lin_proj_0 pools {} tokens but the grid is {}x{}".format(
                weight.shape[1], h_tok, w_tok))
        module = cls(h_tok=h_tok, w_tok=w_tok)
        with torch.no_grad():
            module.weight.copy_(weight.detach().reshape(int(h_tok), int(w_tok)).sum(dim=1))
            module.bias.copy_(bias.detach().reshape(1))
        return module

    def extra_repr(self) -> str:
        return "h_tok={}, w_tok={}".format(self.h_tok, self.w_tok)


def _unpack_token_pool(pool):
    """Return ``(weight, bias)`` from a ``basic_project2`` / ``nn.Linear`` / tensor pair."""
    if isinstance(pool, (tuple, list)):
        if len(pool) != 2:
            raise ValueError("expected a (weight, bias) pair, got {} items".format(len(pool)))
        return torch.as_tensor(pool[0]), torch.as_tensor(pool[1])
    inner = getattr(pool, "proj", pool)          # basic_project2 wraps an nn.Linear in .proj
    weight = getattr(inner, "weight", None)
    bias = getattr(inner, "bias", None)
    if weight is None or bias is None:
        raise TypeError("cannot read weight/bias from {!r}".format(type(pool)))
    return weight, bias


# --------------------------------------------------------------------------------------
# Intrinsic (yaw-invariant) horizontal coordinates -- handoff section 3.3
# --------------------------------------------------------------------------------------

def horizontal_basis(src_loc: torch.Tensor, ref_locs: torch.Tensor,
                     eps: float = DEFAULT_BASIS_EPS):
    """The unit horizontal basis ``a = (a_x, a_y)`` of each scene, with the degenerate rules.

    Rules, in order (handoff section 3.3, "must cover the degenerate cases"):

    1. ``r_q = ||(src_x, src_y)|| > eps``  -> ``a = (src_x, src_y) / r_q``   (``BASIS_QUERY``)
    2. otherwise the **reference with the largest horizontal radius**, ties broken by the
       **lowest reference index** (the manifest's fixed order), provided its radius exceeds
       ``eps``                                                             (``BASIS_REFERENCE``)
    3. otherwise every horizontal component is at or below ``eps``: ``a = (0, 0)``, which
       makes the transform emit zero horizontal components    (``BASIS_DEGENERATE``)

    The radii are computed in float64 so the branch does not depend on float32 rounding of a
    rotated input; the comparison is a strict ``>`` against ``eps``, so a horizontal radius of
    exactly ``eps`` does **not** claim the query basis -- it falls through to rule 2.

    Every quantity the rule uses (horizontal radii, reference index) is yaw invariant, so the
    chosen basis rotates with the scene and the resulting coordinates do not.  Two knife
    edges remain, both measure-zero and both numerical rather than architectural: ``r_q``
    within float rounding of ``eps``, and two references whose radii agree to within float
    rounding (the arg-max could flip).

    Args:
        src_loc: query-source position ``[B, 3]`` in the receiver frame.
        ref_locs: reference-source positions ``[B, K, 3]`` in the receiver frame.
        eps: horizontal-radius threshold in metres.

    Returns:
        ``(basis, mode)`` -- ``basis`` ``[B, 2]`` in ``src_loc``'s dtype/device, ``mode``
        ``[B]`` int64 holding ``BASIS_QUERY`` / ``BASIS_REFERENCE`` / ``BASIS_DEGENERATE``.
    """
    if src_loc.dim() != 2 or src_loc.shape[-1] != 3:
        raise ValueError("src_loc must be [B, 3], got {}".format(tuple(src_loc.shape)))
    if ref_locs.dim() != 3 or ref_locs.shape[-1] != 3:
        raise ValueError("ref_locs must be [B, K, 3], got {}".format(tuple(ref_locs.shape)))
    if ref_locs.shape[0] != src_loc.shape[0]:
        raise ValueError("src_loc has {} rows but ref_locs has {}".format(
            src_loc.shape[0], ref_locs.shape[0]))
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive, got {}".format(eps))

    device = src_loc.device
    batch, n_ref = ref_locs.shape[0], ref_locs.shape[1]
    q = src_loc[:, :2].double()                                     # [B, 2]
    r_q = torch.linalg.norm(q, dim=-1)                              # [B]
    use_query = r_q > float(eps)

    if n_ref == 0:                                                  # no fallback available
        ref_unit = torch.zeros_like(q)
        use_ref = torch.zeros_like(use_query)
    else:
        ref_h = ref_locs[:, :, :2].double()                         # [B, K, 2]
        r_ref = torch.linalg.norm(ref_h, dim=-1)                    # [B, K]
        r_max = r_ref.max(dim=1).values                             # [B]
        # Lowest index among the maximal radii -- spelled out rather than left to argmax,
        # whose tie-breaking is not part of torch's contract.
        index = torch.arange(n_ref, device=device).expand(batch, n_ref)
        is_max = r_ref >= r_max.unsqueeze(1)
        pick = torch.where(is_max, index, torch.full_like(index, n_ref)).min(dim=1).values
        chosen = ref_h[torch.arange(batch, device=device), pick]    # [B, 2]
        r_chosen = r_ref[torch.arange(batch, device=device), pick]  # [B] == r_max
        ref_unit = chosen / r_chosen.clamp_min(float(eps)).unsqueeze(-1)
        use_ref = (~use_query) & (r_max > float(eps))

    q_unit = q / r_q.clamp_min(float(eps)).unsqueeze(-1)
    basis = torch.where(use_query.unsqueeze(-1), q_unit,
                        torch.where(use_ref.unsqueeze(-1), ref_unit, torch.zeros_like(q_unit)))
    mode = torch.full((batch,), BASIS_DEGENERATE, dtype=torch.long, device=device)
    mode = torch.where(use_ref, torch.full_like(mode, BASIS_REFERENCE), mode)
    mode = torch.where(use_query, torch.full_like(mode, BASIS_QUERY), mode)
    return basis.to(src_loc.dtype), mode


def to_intrinsic(positions: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Express ``positions`` in the horizontal basis ``a`` (handoff section 3.3)::

        p_invariant = (a_x p_x + a_y p_y,  a_x p_y - a_y p_x,  p_z)

    i.e. (component along ``a``, signed area with ``a``, height).  All three are invariant
    under a common yaw of ``a`` and ``p``, the relative azimuth of every reference with
    respect to the query survives, and there is no ``+-pi`` branch cut.

    Args:
        positions: ``[B, ..., 3]`` positions sharing the batch axis with ``basis``.
        basis: ``[B, 2]`` horizontal basis from :func:`horizontal_basis` (need not be unit:
            a zero basis yields zero horizontal components, which is the degenerate rule).

    Returns:
        A tensor of the same shape/dtype/device as ``positions``.  The arithmetic runs in
        float64, so the only float32 rounding is the single cast of the result.
    """
    if positions.shape[-1] != 3:
        raise ValueError("positions must end in xyz (size 3), got {}".format(
            tuple(positions.shape)))
    if basis.dim() != 2 or basis.shape[-1] != 2:
        raise ValueError("basis must be [B, 2], got {}".format(tuple(basis.shape)))
    if basis.shape[0] != positions.shape[0]:
        raise ValueError("basis has {} rows but positions has {}".format(
            basis.shape[0], positions.shape[0]))
    lead = (positions.shape[0],) + (1,) * (positions.dim() - 2)
    a_x = basis[:, 0].double().view(lead)
    a_y = basis[:, 1].double().view(lead)
    p = positions.double()
    p_x, p_y, p_z = p[..., 0], p[..., 1], p[..., 2]
    out = torch.stack([a_x * p_x + a_y * p_y, a_x * p_y - a_y * p_x, p_z], dim=-1)
    return out.to(positions.dtype)


def intrinsic_scene_coords(src_loc: torch.Tensor, ref_locs: torch.Tensor,
                           eps: float = DEFAULT_BASIS_EPS):
    """Query **and** every reference in the scene's intrinsic horizontal frame.

    Changing only the query (handoff section 3.3, last line of the rule list) would leave the
    reference embedding rotating, so both are transformed with the *same* basis.

    Args:
        src_loc: query-source position ``[B, 3]``.
        ref_locs: reference-source positions ``[B, K, 3]``.
        eps: horizontal-radius threshold (see :func:`horizontal_basis`).

    Returns:
        ``(src_intrinsic [B, 3], ref_intrinsic [B, K, 3], basis [B, 2], mode [B])``.
    """
    basis, mode = horizontal_basis(src_loc, ref_locs, eps=eps)
    return to_intrinsic(src_loc, basis), to_intrinsic(ref_locs, basis), basis, mode


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

class xRIR_InvariantBase(xRIR):
    """``xRIR`` with the invariant readout and intrinsic coordinate conditioning.

    Not a registered backbone on its own -- use :class:`xRIR_CylInvariant` (the method) or
    :class:`xRIR_SimpleInvariant` (the control).  ``lin_proj_0`` is **removed**: it is
    consumed by :attr:`invariant_readout` at warm start and would otherwise sit in the
    state dict as a dead parameter.  Everything else -- ``src_proj``, ``src_coord_proj``,
    ``dist_embedder``, ``time_proj``, ``audio_enc``, ``lin_proj_1``, ``lin_proj_2``,
    ``shift_and_align``, ``convert_ir_to_spec`` -- is the pinned module, unmodified.
    """

    def __init__(self, num_channels, n_bins=310, dim=512, intermediate_ch=256,
                 image_size=(256, 512), patch_size=(16, 32), depth=12, mlp_dim=512, heads=8,
                 basis_eps=DEFAULT_BASIS_EPS):
        super().__init__(num_channels, n_bins=n_bins, dim=dim, intermediate_ch=intermediate_ch,
                         image_size=image_size, patch_size=patch_size, depth=depth,
                         mlp_dim=mlp_dim, heads=heads)
        image_height, image_width = tuple(image_size)
        patch_height, patch_width = tuple(patch_size)
        self.h_tok = image_height // patch_height
        self.w_tok = image_width // patch_width
        self.basis_eps = float(basis_eps)
        self.invariant_readout = InvariantReadout(h_tok=self.h_tok, w_tok=self.w_tok)
        del self.lin_proj_0

    def forward(self, depth_coord, x, src_loc, ref_ir_locs, tgt_wav):
        """``xRIR.forward`` with the two non-invariant paths replaced; nothing else differs.

        Args/returns are the pinned contract: ``depth_coord [B, 3, 256, 512]``,
        ``x [B, K, L]`` reference RIRs, ``src_loc [B, 3]``, ``ref_ir_locs [B, K, 3]``,
        ``tgt_wav [B, 1, L]`` -> ``(out_log_spec [B, F, T, 1], tgt_spec [B, 1, F, T])``.
        """
        x = self.shift_and_align(x, src_loc, ref_ir_locs)
        times = self.times
        time_embed = self.time_embedder(times.unsqueeze(0).to(ref_ir_locs.device)).repeat(ref_ir_locs.shape[0], 1, 1)  # B,T,21
        time_out = self.time_proj(time_embed) # B T C

        # --- exp_08 change 1 of 2: coordinate-embedding branch sees the intrinsic frame ---
        src_intrinsic, ref_intrinsic, _, _ = intrinsic_scene_coords(
            src_loc, ref_ir_locs, eps=self.basis_eps)

        src_coord_encoding = (src_loc[:, :, None, None] - depth_coord) / 5.

        source_out = self.src_proj(self.source_network(src_coord_encoding)) # B x K x C
        # --- exp_08 change 2 of 2: azimuth-invariant pooling on all three geometry views ---
        source_out = self.invariant_readout(source_out) # B x 1 x C

        rec_coord_encoding = (-depth_coord) / 5. #self.dist_embedder(
        receiver_out = self.src_proj(self.source_network(rec_coord_encoding)) # B x K x C
        receiver_out = self.invariant_readout(receiver_out)

        ref_geo_feat_list = []

        for i in range(ref_ir_locs.shape[1]):
            ref_src_coord_encoding =  (ref_ir_locs[:, i, :, None, None] - depth_coord) / 5. #self.dist_embedder(
            ref_geo_feat = self.src_proj(self.source_network(ref_src_coord_encoding))
            ref_geo_feat = self.invariant_readout(ref_geo_feat) #b 1 C
            ref_geo_feat_list.append(ref_geo_feat)#.unsqueeze(1)
        ref_geo_feats = torch.cat(ref_geo_feat_list, dim=1) # BxNxC B x N x K x C

        fuse_geo_feats = torch.cat([receiver_out, source_out], dim=-1)
        fuse_ref_geo_feats = torch.cat([receiver_out.repeat(1, ref_geo_feats.shape[1], 1), ref_geo_feats], dim=-1) # B N 2C

        ref_src_feats = []

        for i in range(ref_ir_locs.shape[1]):
            ref_src_feat = self.src_coord_proj(self.dist_embedder((ref_intrinsic[:, i:(i+1)]) / 5.).view(ref_ir_locs.shape[0], -1)).unsqueeze(1)
            ref_src_feats.append(ref_src_feat)
        ref_src_feats = torch.cat(ref_src_feats, dim=1) # B N C
        src_feats = self.src_coord_proj(self.dist_embedder(src_intrinsic.unsqueeze(1) / 5.0).view(src_loc.shape[0], -1)).unsqueeze(1) # B 1 C

        fuse_geo_feats = torch.cat([src_feats, fuse_geo_feats], dim=-1) # B 1 3*C
        fuse_ref_geo_feats = torch.cat([ref_src_feats, fuse_ref_geo_feats], dim=-1) # B N 3*C

        a_feats_all = []
        specs_all = []
        log_specs_all = []
        for i in range(x.shape[1]):
            spec_i = self.convert_ir_to_spec(x[:, i:(i+1)])
            specs_all.append(spec_i)
            log_specs_all.append(torch.log(spec_i + 1e-8))
            a_feats_i = self.audio_enc(spec_i)
            a_feats_all.append(a_feats_i.unsqueeze(1))
        a_feats_all = torch.cat(a_feats_all, dim=1) # B N 2*C
        specs_all = torch.cat(specs_all, dim=1) # B N 63 T
        log_specs_all = torch.cat(log_specs_all, dim=1)
        fuse_ref_feats = self.lin_proj_2(torch.cat((fuse_ref_geo_feats, a_feats_all), dim=-1)) #B N C
        fuse_tgt_feats = self.lin_proj_1(fuse_geo_feats) # B 1 C

        fuse_feats = F.softmax(fuse_ref_feats @ fuse_tgt_feats.permute(0, 2, 1) / fuse_tgt_feats.shape[-1], dim=1) * fuse_ref_feats # B N C
        weights = fuse_feats @ time_out.permute(0, 2, 1) / fuse_feats.shape[-1] # B N T
        out_log_spec = torch.sum(log_specs_all * weights.unsqueeze(2), dim=1).unsqueeze(1).permute(0, 2, 3, 1)
        tgt_spec = self.convert_ir_to_spec(tgt_wav)

        return out_log_spec, tgt_spec


class xRIR_SimpleInvariant(xRIR_InvariantBase):
    """SimpleViT + the invariant readout and intrinsic coordinates -- the 2x2 control arm.

    **No invariance guarantee is expected here.**  ``SimpleViT`` uses an absolute 2-D
    sin/cos position embedding and no per-column gauge alignment, so its tokens do not
    permute under an azimuth roll; the invariant readout removes the *readout* failure mode
    and the intrinsic coordinates remove the *conditioning* failure mode, but the encoder
    itself still responds to the rotation.  Its role (handoff section 5) is attribution: it
    separates "what the readout/coordinate change buys" from "what CylindricalViT buys".
    """


class xRIR_CylInvariant(xRIR_InvariantBase):
    """The Route 2 method: CylindricalViT + invariant readout + intrinsic coordinates.

    ``source_network`` is pinned to :class:`model.cylindrical_vit.CylindricalViT` (exp_02's
    encoder, unmodified), so the only differences from ``xRIR_Cyl`` are the two paths this
    module replaces.  Together they make one forward pass invariant under the C16 scene yaw.
    """

    def __init__(self, num_channels, n_bins=310, dim=512, intermediate_ch=256,
                 image_size=(256, 512), patch_size=(16, 32), depth=12, mlp_dim=512, heads=8,
                 basis_eps=DEFAULT_BASIS_EPS):
        super().__init__(num_channels, n_bins=n_bins, dim=dim, intermediate_ch=intermediate_ch,
                         image_size=image_size, patch_size=patch_size, depth=depth,
                         mlp_dim=mlp_dim, heads=heads, basis_eps=basis_eps)
        # Same dim / depth / heads / dim_head / mlp_dim as the SimpleViT it replaces.
        self.source_network = CylindricalViT(
            in_channels=3, image_size=tuple(image_size), patch_size=tuple(patch_size),
            dim=dim, depth=depth, heads=heads, dim_head=64, mlp_dim=mlp_dim,
        )
        num_tokens = self.source_network.num_tokens
        if num_tokens != self.h_tok * self.w_tok:
            raise ValueError(
                "InvariantReadout expects a {}x{} token grid but CylindricalViT produces "
                "{} tokens for image_size={}, patch_size={}".format(
                    self.h_tok, self.w_tok, num_tokens, image_size, patch_size))
