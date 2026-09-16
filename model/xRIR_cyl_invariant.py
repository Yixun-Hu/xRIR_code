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
  original ``(location - depth_coord) / 5`` input; its semantics are not swapped.  The basis
  is computed **branch-free** -- see :func:`horizontal_basis` for why every thresholded
  version of it was a rotation hazard.

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

import model.xRIR as _xrir_module
from model.cylindrical_vit import CylindricalViT
from model.xRIR import xRIR


#: Blend band on the query's horizontal *fraction* ``||q_h|| / ||q||`` -- dimensionless, so it
#: means the same thing for a source 0.5 m away and one 40 m away.  Below ``BLEND_LO`` the query
#: direction is ignored entirely, above ``BLEND_HI`` it is used alone, and in between a quintic
#: smoothstep interpolates.  Every real AcousticRooms query sits far above ``BLEND_HI``
#: (sources and receivers share a room floor plan), so the production path is the primary one
#: and is bitwise identical to a hard ``||q_h|| > 0`` rule.
BLEND_LO = 1.0e-3
BLEND_HI = 1.0e-2


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

def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    """Quintic smoothstep ``6t^5 - 15t^4 + 10t^3`` on ``[0, 1]``, clamped outside.

    C2: value, first and second derivatives all match at both ends, so blending with it adds
    no kink anywhere.  It saturates *exactly* at 0 and 1 (not asymptotically), which is what
    lets the primary branch stay bitwise identical to an unblended unit vector.
    """
    t = t.clamp(0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def horizontal_basis(src_loc: torch.Tensor, ref_locs: torch.Tensor,
                     blend_lo: float = BLEND_LO, blend_hi: float = BLEND_HI):
    """The horizontal basis ``a = (a_x, a_y)`` of each scene -- **branch-free**::

        w = smoothstep((||q_h|| / ||q|| - lo) / (hi - lo))
        a = w * q_h / max(||q_h||, lo * ||q||)  +  (1 - w) * sum_i p_h_i / sum_i ||p_h_i||

    with ``q_h`` the query source's horizontal component and ``p_h_i`` the references'.  There
    is **no** conditional anywhere in that expression: no threshold decides which branch is
    taken, and no comparison decides which reference wins.

    Why it is built this way (exp_08 review rounds 1 and 2)
    ------------------------------------------------------
    Two earlier versions each failed on a *discontinuity*, and a discontinuity in the basis is a
    discontinuity in the model's output under rotation:

    * **round 1** -- the fallback picked the reference with the largest horizontal radius.  With
      three references at radius exactly 5, float32 rounding of the rotated coordinates chose
      the winner: coordinates jumped 4.4 m, the spectrum by ``1.5e-1``.
    * **round 2** -- the fallback became ``s / ||s||`` with ``s`` the vector sum, plus an
      ``||s|| > eps`` guard.  The sum removed the *selection*, but normalising by its own length
      is ill-conditioned exactly where the sum is small, and the guard reintroduced a hard
      branch.  For references ``(12,16,0), (20,0,0), (-32,-16,0)`` the sum is exactly zero,
      rotation rounding pushed ``||s||`` across ``eps``, the branch flipped, and the coordinates
      jumped **35.7 m** (spectrum ``2.3e-1``); for a merely near-cancelling set the branch held
      but the conditioning alone moved them **1.5 m** (spectrum ``5.2e-2``).

    Dividing by ``R = sum_i ||p_h_i||`` instead of by ``||s||`` fixes both at once:

    * ``R`` is **rotation invariant** and ``s`` is **rotation equivariant**, so ``a`` is exactly
      equivariant -- ``sum_i Rz(D) p_i = Rz(D) sum_i p_i``, and no norm moves;
    * ``R >= ||s||`` by the triangle inequality, so ``||a|| <= 1`` always and the division never
      amplifies: rounding perturbs ``s`` by ``O(1e-7 * R)``, hence ``a`` by ``O(1e-7)``
      *absolute* and the coordinates by ``O(1e-7 * ||p_h||)`` -- machine level at any scale;
    * where the reference *directions* cancel, ``||s||`` falls away while ``R`` does not, so the
      basis attenuates to zero instead of being renormalised back to unit length -- the round-2
      "all horizontal components vanish" branch is replaced by that behaviour rather than by a
      test.  A single exact ``R == 0`` guard remains, for the case where there is nothing to
      divide by at all.

    What is and is not continuous (exp_08 review round 3)
    ----------------------------------------------------
    Being branch-free removes the *rotation* hazard.  It does not make the map smooth in the
    scene geometry, and the earlier draft of this docstring overclaimed that it did:

    * **In the yaw variable -- exactly equivariant at every geometry, without exception.**
      ``sum_i Rz(D) p_i = Rz(D) sum_i p_i`` and ``R`` is built from norms, which a rotation
      preserves; the ``R == 0`` guard is rotation invariant because rotating a zero horizontal
      component leaves it zero.  So for any *fixed* scene the intrinsic coordinates are
      invariant, which is the C16 guarantee, and nothing below weakens it.
    * **In the geometry variables -- continuous except where the reference directions are
      themselves undetermined**, and only Lipschitz (not differentiable) even there:

      - ``R = sum_i ||p_h_i||`` has a **cusp** wherever an individual reference's horizontal
        component crosses zero, because the Euclidean norm is not differentiable at the origin;
      - at ``R = 0`` the map is genuinely **discontinuous**, and shrinking the references does
        not fix it: for a query at ``(5e-4, 0, 1)`` and references ``(e, 0, z_i)``,
        ``a = (1, 0)`` for *every* ``e > 0`` -- the ratio ``s / R`` depends on the references'
        directions, not their magnitudes -- while ``a = (0, 0)`` at ``e = 0`` exactly.  On the
        warm-started model that is a ``5.5e-5`` relative difference between two *neighbouring
        scenes*.

    The same directional discontinuity exists at ``q = 0`` in the primary term, for the same
    reason.  These are smoothness properties of the representation **across scenes**; they are
    not invariance defects, and they are not regularised away here -- doing so would trade an
    exact symmetry for a cosmetic one.

    The same treatment is applied to the *primary* choice, which had the identical defect: a
    hard ``||q_h|| > eps`` test switching between the query direction and the fallback.  The
    smoothstep blend replaces it, and the band is a **fraction** of ``||q||`` rather than an
    absolute length -- the round-2 threshold was ``1e-6`` m, which is meaningless for a scene
    whose references sit 20-36 m out.  The blend is C2 in that fraction; it is the fraction's
    own behaviour at ``q = 0`` that is not.

    What it means physically
    ------------------------
    When the references' directions cancel there is genuinely no horizontal direction the scene
    distinguishes, and ``||a|| < 1`` makes the representation *attenuate* its horizontal
    components in proportion to how ambiguous they are, reaching zero exactly when they cancel.
    That is the honest encoding of "this scene has no preferred azimuth", not a numerical trick:
    heights pass through untouched, and the attenuation is itself rotation invariant.

    Args:
        src_loc: query-source position ``[B, 3]`` in the receiver frame.
        ref_locs: reference-source positions ``[B, K, 3]`` in the receiver frame.
        blend_lo: query horizontal fraction below which the query direction is ignored.
        blend_hi: fraction above which it is used alone.  Must exceed ``blend_lo``.

    Returns:
        ``(basis, blend)`` -- ``basis`` ``[B, 2]`` with ``\\|basis\\| <= 1``, and ``blend``
        ``[B]`` the weight ``w`` actually used, both in ``src_loc``'s dtype/device.  ``blend``
        is a *diagnostic*: nothing in this function or its callers branches on it.
    """
    if src_loc.dim() != 2 or src_loc.shape[-1] != 3:
        raise ValueError("src_loc must be [B, 3], got {}".format(tuple(src_loc.shape)))
    if ref_locs.dim() != 3 or ref_locs.shape[-1] != 3:
        raise ValueError("ref_locs must be [B, K, 3], got {}".format(tuple(ref_locs.shape)))
    if ref_locs.shape[0] != src_loc.shape[0]:
        raise ValueError("src_loc has {} rows but ref_locs has {}".format(
            src_loc.shape[0], ref_locs.shape[0]))
    if not 0.0 <= float(blend_lo) < float(blend_hi):
        raise ValueError("need 0 <= blend_lo < blend_hi, got {} and {}".format(
            blend_lo, blend_hi))

    tiny = torch.finfo(torch.float64).tiny
    lo, hi = float(blend_lo), float(blend_hi)

    # --- primary: the query's own horizontal direction, weighted by how horizontal it is ---
    q = src_loc[:, :2].double()                                     # [B, 2]
    q_h = torch.linalg.norm(q, dim=-1)                              # [B]
    q_full = torch.linalg.norm(src_loc.double(), dim=-1)            # [B]
    fraction = q_h / q_full.clamp_min(tiny)                         # dimensionless, in [0, 1]
    blend = _smoothstep((fraction - lo) / (hi - lo))                # [B], exactly 0 or 1 outside
    q_denominator = torch.maximum(q_h, lo * q_full)
    q_direction = torch.where((q_denominator > 0).unsqueeze(-1),
                              q / q_denominator.clamp_min(tiny).unsqueeze(-1),
                              torch.zeros_like(q))

    # --- fallback: the references' vector sum over their total horizontal radius ---
    ref_h = ref_locs[:, :, :2].double()                             # [B, K, 2]
    total = ref_h.sum(dim=1)                                        # [B, 2], equivariant
    radius = torch.linalg.norm(ref_h, dim=-1).sum(dim=1)            # [B], invariant
    aggregate = torch.where((radius > 0).unsqueeze(-1),
                            total / radius.clamp_min(tiny).unsqueeze(-1),
                            torch.zeros_like(total))

    basis = blend.unsqueeze(-1) * q_direction + (1.0 - blend).unsqueeze(-1) * aggregate
    return basis.to(src_loc.dtype), blend.to(src_loc.dtype)


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
                           blend_lo: float = BLEND_LO, blend_hi: float = BLEND_HI):
    """Query **and** every reference in the scene's intrinsic horizontal frame.

    Changing only the query (handoff section 3.3, last line of the rule list) would leave the
    reference embedding rotating, so both are transformed with the *same* basis.

    Args:
        src_loc: query-source position ``[B, 3]``.
        ref_locs: reference-source positions ``[B, K, 3]``.
        blend_lo: see :func:`horizontal_basis`.
        blend_hi: see :func:`horizontal_basis`.

    Returns:
        ``(src_intrinsic [B, 3], ref_intrinsic [B, K, 3], basis [B, 2], blend [B])``; ``blend``
        is diagnostic only -- no caller branches on it.
    """
    basis, blend = horizontal_basis(src_loc, ref_locs, blend_lo=blend_lo, blend_hi=blend_hi)
    return to_intrinsic(src_loc, basis), to_intrinsic(ref_locs, basis), basis, blend


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

class xRIR_InvariantBase(xRIR):
    """``xRIR`` with the invariant readout and intrinsic coordinate conditioning.

    Not a registered backbone on its own -- use :class:`xRIR_CylInvariant` (the method) or
    :class:`xRIR_SimpleInvariant` (the control).  ``lin_proj_0`` is **removed**: it is
    consumed by :attr:`invariant_readout` at warm start and would otherwise sit in the
    state dict as a dead parameter.  :meth:`shift_and_align` is overridden to reduce the
    distances in float64 before rounding to an integer delay (see the method; it carries no
    parameters and changes no semantics beyond precision).  Everything else -- ``src_proj``,
    ``src_coord_proj``, ``dist_embedder``, ``time_proj``, ``audio_enc``, ``lin_proj_1``,
    ``lin_proj_2``, ``convert_ir_to_spec`` -- is the pinned module, unmodified.
    """

    def __init__(self, num_channels, n_bins=310, dim=512, intermediate_ch=256,
                 image_size=(256, 512), patch_size=(16, 32), depth=12, mlp_dim=512, heads=8,
                 blend_lo=BLEND_LO, blend_hi=BLEND_HI):
        super().__init__(num_channels, n_bins=n_bins, dim=dim, intermediate_ch=intermediate_ch,
                         image_size=image_size, patch_size=patch_size, depth=depth,
                         mlp_dim=mlp_dim, heads=heads)
        image_height, image_width = tuple(image_size)
        patch_height, patch_width = tuple(patch_size)
        self.h_tok = image_height // patch_height
        self.w_tok = image_width // patch_width
        self.blend_lo = float(blend_lo)
        self.blend_hi = float(blend_hi)
        self.invariant_readout = InvariantReadout(h_tok=self.h_tok, w_tok=self.w_tok)
        del self.lin_proj_0

    def shift_and_align(self, x, src_loc, ref_ir_locs):
        """``xRIR.shift_and_align`` with the distance reduction accumulated in **float64**.

        Part of the invariant arms' architecture, not a test shim (exp_08 review round 1,
        blocker B2).  The pinned line

            ``delay_unit = round((||src|| - ||ref||) / 343 * 22050)``

        is algebraically yaw invariant but discretised: a scene whose pre-round value sits
        within the float32 noise of a ``.5`` tie can have its integer delay move by one sample
        under a rotation that preserves the norms exactly, which shifts a whole reference RIR
        by one sample and breaks end-to-end invariance for that scene.  Computing the two
        norms and their difference in float64 removes the *arithmetic* half of that noise;
        what remains is the float32 rounding of the rotated coordinates themselves, which is
        irreducible because the model never receives the exactly-rotated scene.  Measured over a
        random ensemble by ``tools/exp08_validate.delay_flip_rate``, this lowers how often a
        rotation moves a delay by a factor of about 1.28 (3.6 sigma on paired per-scene counts).

        It narrows the band; it does not empty it.  A configuration engineered to sit within
        ~1e-6 samples of a tie still flips, and the validation report states that exception
        explicitly rather than claiming exactness the discretisation cannot deliver.

        The direct-path gain is reduced in float64 too and cast back, so the whole
        distance-to-audio path has one precision policy.  ``apply_delay`` is resolved from
        ``model.xRIR``'s module globals at call time, exactly as the pinned method does, so
        ``model.exp08_delay.device_preserving_delay`` still reaches it.  The pinned method's
        unused ``dist_result`` is not recomputed.

        Args:
            x: reference RIRs ``[B, K, L]``.
            src_loc: query-source position ``[B, 3]``.
            ref_ir_locs: reference-source positions ``[B, K, 3]``.

        Returns:
            ``[B, K, L]`` delayed and gain-scaled references, in ``x``'s dtype.
        """
        channel_outputs = []
        dist_src = torch.linalg.norm(src_loc.double(), dim=1).unsqueeze(1)      # [B, 1]
        dist_ref = torch.linalg.norm(ref_ir_locs.double(), dim=-1)              # [B, K]
        direct_energy_ratio = (dist_ref / (dist_src + 1e-7)).to(x.dtype)
        delay_unit = torch.round((dist_src - dist_ref) / 343. * 22050).int()

        for i in range(x.shape[1]):
            cur_delayed_x = _xrir_module.apply_delay(x[:, i, :], delay_unit[:, i]).unsqueeze(1)
            x_channel = cur_delayed_x * direct_energy_ratio[:, i:(i + 1)].unsqueeze(2)
            channel_outputs.append(x_channel)
        return torch.cat(channel_outputs, dim=1)

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
            src_loc, ref_ir_locs, blend_lo=self.blend_lo, blend_hi=self.blend_hi)

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
                 blend_lo=BLEND_LO, blend_hi=BLEND_HI):
        super().__init__(num_channels, n_bins=n_bins, dim=dim, intermediate_ch=intermediate_ch,
                         image_size=image_size, patch_size=patch_size, depth=depth,
                         mlp_dim=mlp_dim, heads=heads, blend_lo=blend_lo, blend_hi=blend_hi)
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
