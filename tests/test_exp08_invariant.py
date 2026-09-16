"""exp_08 structural validation: the handoff's section 4 checklist, on CPU, without training.

Covers, in order, the six required checks of
``worklog/worklog_yixun/xrir_invariant_readout_route2_handoff.md`` section 4:

1. azimuth-roll invariance of the readout on non-constant random tokens *and* on real
   tokens from the warm-started encoder, plus the sensitivity checks that keep the first
   assertion from being vacuous;
2. the intrinsic coordinate representation -- joint-yaw invariance, sensitivity to genuine
   relative-geometry changes, and every degenerate/fallback branch with its threshold;
3. the *whole* model at all 15 non-identity C16 angles, random-init and warm-started, on
   real manifest scenes from different rooms, asymmetric synthetic scenes, batch > 1 and a
   reduced-K case;
4. the full predicted spectrum **and** the seeded Griffin-Lim waveform, on the true
   end-to-end path (the alignment recomputed from the rotated coordinates), never on a
   pinned ``k = 0`` alignment cache alone;
5. a real forward + backward + optimizer step with finite gradients that reach the new
   readout and the coordinate branch while the pinned parameters keep training;
6. the pinned ``simple`` / ``cylindrical`` entries, unchanged bit for bit.

Plus two things the handoff insists on separately: the off-lattice angles are recorded under
their own label and never judged by the C16 criterion, and the harness is shown to be
load-bearing -- the *pinned* cylindrical model fails the same criterion by five orders of
magnitude, so a passing invariant model is evidence rather than an artefact of the test.

The heavy sweeps run once in a module-scoped fixture; the per-angle tests assert on that
cache.  Everything is CPU, FP32, TF32 off, ``eval()``, fixed seeds.
"""
from __future__ import annotations

import math

import pytest
import torch

import model.xRIR as xrir_module
import tools.exp08_validate as validate
from model.exp08_delay import apply_delay_device_preserving, device_preserving_delay
from model.exp08_factory import BACKBONES_EXP08, build_xrir_exp08
from model.xRIR import xRIR
from model.xRIR_cyl import xRIR_Cyl
from model.xRIR_cyl_invariant import (BLEND_HI, BLEND_LO, InvariantReadout, horizontal_basis,
                                      intrinsic_scene_coords, to_intrinsic, xRIR_CylInvariant,
                                      xRIR_SimpleInvariant)
from tools.exp08_warmstart import build_warm_started
from tools.yaw_rotation import (integer_delays, rotate_scene_yaw, rotate_vectors_z,
                                yaw_angle_rad)

#: The handoff's initial numerical target on the relative residual.
REL_TARGET = validate.REL_TARGET

#: Tolerance for the closed-form coordinate transform, whose residual is limited by the
#: float32 rounding of the *rotated inputs* rather than by the architecture (see
#: ``test_intrinsic_coords_are_exact_in_float64``); measured worst case ~1.2e-7.
FP32_INPUT_TOL = 1.0e-6

#: Tolerance for quantities that also pass through the 12-layer geometry encoder, whose
#: float32 accumulation order is not itself rotation invariant.  Judged by the handoff's own
#: numerical target rather than a tighter self-imposed bound; measured worst case ~8.5e-7,
#: and it drifts by tens of percent with the thread count, so a 1e-6 bound would be a coin
#: flip rather than a criterion.
ENCODER_FP32_TOL = REL_TARGET


# --------------------------------------------------------------------------------------
# Shared, expensive fixtures
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env():
    """CPU, FP32, TF32 off, 8 threads, seed 0 -- the condition the residuals are quoted in."""
    return validate.configure_cpu(threads=8, seed=0)


@pytest.fixture(scope="module")
def real(env):
    """Three real manifest queries from three different rooms (batch > 1, K = 8)."""
    batch = validate.real_batch(n_scenes=3)
    assert len(set(batch["rooms"])) == 3, "the three queries must come from three rooms"
    assert len(set(r.split("/")[0] for r in batch["rooms"])) == 3, \
        "the three rooms must also be three different room types"
    assert batch["src_loc"].shape[0] == 3 and batch["ref_locs"].shape[1] == 8
    return batch


@pytest.fixture(scope="module")
def warm_cyl(env):
    """``cylindrical_invariant`` warm-started from ``ckpt/xRIR_cyl_8_shot/epoch_12.pth``."""
    model, accounting, path = build_warm_started("cylindrical_invariant", 8, verbose=False)
    return model.eval(), accounting, path


@pytest.fixture(scope="module")
def random_cyl(env):
    """``cylindrical_invariant`` at random init -- invariance must not depend on the weights."""
    torch.manual_seed(0)
    return build_xrir_exp08("cylindrical_invariant", 8).eval()


@pytest.fixture(scope="module")
def sweeps(env, real, warm_cyl, random_cyl):
    """Every full-model angle sweep of section 4 item 3, computed once.

    Cases: {random init, warm start} x {3 real rooms (B=3, K=8), two asymmetric synthetic
    scenes (B=2, K=8), a reduced-K synthetic scene (B=2, K=3)}.  Condition ``E`` (alignment
    recomputed from the rotated coordinates) everywhere; condition ``P`` (pinned ``k = 0``
    alignment) additionally on the real batch, to separate alignment rounding from the rest.
    """
    warm_model = warm_cyl[0]
    batches = {"real": real,
               "synthA": validate.synthetic_batch(101, batch=2, n_ref=8, label="synthA"),
               "synthB": validate.synthetic_batch(202, batch=2, n_ref=8, label="synthB"),
               "synthK3": validate.synthetic_batch(303, batch=2, n_ref=3, label="synthK3")}
    out = {"batches": batches}
    with device_preserving_delay():
        for state, model in (("random_init", random_cyl), ("warm_start", warm_model)):
            for name, batch in batches.items():
                conditions = ("E", "P") if name == "real" else ("E",)
                out["{}/{}".format(state, name)] = validate.angle_sweep(
                    model, batch, validate.C16_ANGLES, conditions=conditions)
        out["off_lattice"] = validate.angle_sweep(
            warm_model, real, validate.OFF_LATTICE_ANGLES, conditions=("E",))
        torch.manual_seed(0)
        control = build_xrir_exp08("cylindrical", 8).eval()
        out["control_pinned"] = validate.angle_sweep(
            control, real, validate.C16_ANGLES, conditions=("E",))
    return out


CASES = ["{}/{}".format(state, name)
         for state in ("random_init", "warm_start")
         for name in ("real", "synthA", "synthB", "synthK3")]


# --------------------------------------------------------------------------------------
# Section 4 item 1 -- the invariant readout
# --------------------------------------------------------------------------------------

def test_readout_is_bitwise_invariant_to_azimuth_rolls():
    """Non-constant random tokens: every one of the 15 azimuth rolls leaves the output equal.

    Bitwise, not merely close: the azimuth mean is accumulated in float64, so a roll -- which
    is only a change of summation order -- cannot move the float32 result.
    """
    torch.manual_seed(0)
    readout = InvariantReadout(16, 16)
    tokens = torch.randn(3, 256, 64)
    assert tokens.std() > 0.5, "the tokens must be non-constant for this to mean anything"
    base = readout(tokens)
    for shift in range(1, 16):
        rolled = tokens.reshape(3, 16, 16, 64).roll(shift, dims=2).reshape(3, 256, 64)
        assert not torch.equal(rolled, tokens), "shift {} did not move the tokens".format(shift)
        assert torch.equal(readout(rolled), base), "readout moved under azimuth roll {}".format(shift)


def test_readout_is_invariant_on_real_warm_started_encoder_tokens(real, warm_cyl):
    """The same, on tokens the warm-started CylindricalViT produces from a real panorama.

    Runs the actual geometry branch -- ``src_proj(source_network((src - depth) / 5))`` -- at
    ``k = 0`` and at every C16 angle.  The tokens themselves must move (they permute); the
    readout must not, to the float32 input-rounding floor.
    """
    model = warm_cyl[0]
    moved, permuted, worst_rel, worst_abs = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        encoding = (real["src_loc"][:, :, None, None] - real["depth_coord"]) / 5.
        tokens0 = model.src_proj(model.source_network(encoding))
        base = model.invariant_readout(tokens0)
        denom = float(base.norm())
        for k in validate.C16_ANGLES:
            depth_k, src_k, _ = rotate_scene_yaw(real["depth_coord"], real["src_loc"],
                                                 real["ref_locs"], k)
            tokens_k = model.src_proj(model.source_network(
                (src_k[:, :, None, None] - depth_k) / 5.))
            moved = max(moved, float((tokens_k - tokens0).abs().max()))
            # Undoing the token-column roll must recover the k = 0 tokens: the encoder is
            # equivariant, which is what makes the azimuth mean invariant rather than lossy.
            unrolled = tokens_k.reshape(tokens_k.shape[0], 16, 16, -1).roll(
                -(k // 32), dims=2).reshape(tokens_k.shape)
            permuted = max(permuted, float((unrolled - tokens0).abs().max()))
            delta = model.invariant_readout(tokens_k) - base
            worst_abs = max(worst_abs, float(delta.abs().max()))
            worst_rel = max(worst_rel, float(delta.norm()) / denom)
    assert moved > 1e-3, "the encoder tokens did not actually permute; the test is vacuous"
    assert permuted <= 1e-4, \
        "the tokens are not an azimuth permutation of each other ({:.3e})".format(permuted)
    assert worst_rel <= ENCODER_FP32_TOL, \
        "readout on real tokens moved by rel {:.3e} (abs {:.3e}, ||base||_F {:.3e})".format(
            worst_rel, worst_abs, denom)


def test_readout_changes_when_token_content_changes():
    """Sensitivity: the readout must not be a constant function of its input."""
    torch.manual_seed(1)
    readout = InvariantReadout(16, 16)
    tokens = torch.randn(2, 256, 32)
    perturbed = tokens.clone()
    perturbed[:, 5] += 1.0
    assert not torch.allclose(readout(perturbed), readout(tokens), atol=1e-6)


def test_readout_changes_under_elevation_permutation():
    """Sensitivity: elevation structure is kept -- permuting the 16 rows changes the output."""
    torch.manual_seed(2)
    readout = InvariantReadout(16, 16)
    tokens = torch.randn(2, 256, 32)
    permutation = torch.randperm(16, generator=torch.Generator().manual_seed(3))
    assert not torch.equal(permutation, torch.arange(16))
    permuted = tokens.reshape(2, 16, 16, 32)[:, permutation].reshape(2, 256, 32)
    assert not torch.allclose(readout(permuted), readout(tokens), atol=1e-6)


def test_readout_warm_start_folds_the_old_pool_over_azimuth():
    """``a[h] = sum_w W[h, w]`` and the bias is carried (handoff section 3.2)."""
    torch.manual_seed(4)
    pool = torch.nn.Linear(256, 1)
    readout = InvariantReadout.from_token_pool(pool, 16, 16)
    expected = pool.weight.detach().reshape(16, 16).sum(dim=1)
    assert torch.allclose(readout.weight, expected, atol=0, rtol=0)
    assert torch.equal(readout.bias, pool.bias.detach())


def test_readout_warm_start_reproduces_an_azimuth_constant_pool():
    """When the old weights happen to be azimuth-constant, the fold is exact, not approximate."""
    torch.manual_seed(5)
    pool = torch.nn.Linear(256, 1)
    with torch.no_grad():
        pool.weight.copy_(torch.randn(16, 1).repeat(1, 16).reshape(1, 256))
    readout = InvariantReadout.from_token_pool(pool, 16, 16)
    tokens = torch.randn(2, 256, 32)
    old = pool(tokens.permute(0, 2, 1)).squeeze(-1).unsqueeze(1)
    assert torch.allclose(readout(tokens), old, atol=1e-5, rtol=1e-5)


def test_readout_rejects_a_wrong_token_count():
    """A grid/token mismatch is an error, never a silent reshape."""
    readout = InvariantReadout(16, 16)
    with pytest.raises(ValueError):
        readout(torch.randn(1, 255, 8))


# --------------------------------------------------------------------------------------
# Section 4 item 2 -- the intrinsic coordinate representation
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", list(validate.C16_ANGLES) + list(validate.OFF_LATTICE_ANGLES))
def test_intrinsic_coords_are_invariant_under_a_joint_yaw(k):
    """A common yaw of query and references leaves both intrinsic representations put.

    The transform is exactly invariant for *any* yaw (it is built from a dot product, a
    signed area and the height), so off-lattice angles are included here -- unlike the model
    test, where the C16 lattice is what the patch encoder can guarantee.
    """
    torch.manual_seed(6)
    src = torch.randn(4, 3) * 3.0
    refs = torch.randn(4, 8, 3) * 3.0
    src_0, refs_0, _, blend = intrinsic_scene_coords(src, refs)
    assert float(blend.min()) == 1.0, "generic positions must use the query's own direction"
    angle = yaw_angle_rad(k)
    src_k, refs_k, _, _ = intrinsic_scene_coords(rotate_vectors_z(src, angle),
                                                 rotate_vectors_z(refs, angle))
    scale = float(refs_0.abs().max())
    assert float((src_k - src_0).abs().max()) / scale <= FP32_INPUT_TOL
    assert float((refs_k - refs_0).abs().max()) / scale <= FP32_INPUT_TOL


def test_intrinsic_coords_are_exact_in_float64():
    """With float64 inputs the residual collapses to ~1e-15: the maths is exact.

    This is what pins the float32 residual on the rounding of the rotated *inputs* rather
    than on anything in the transform.
    """
    torch.manual_seed(7)
    src = (torch.randn(4, 3) * 3.0).double()
    refs = (torch.randn(4, 8, 3) * 3.0).double()
    src_0, refs_0, _, _ = intrinsic_scene_coords(src, refs)
    worst = 0.0
    for k in validate.C16_ANGLES:
        angle = yaw_angle_rad(k)
        src_k, refs_k, _, _ = intrinsic_scene_coords(rotate_vectors_z(src, angle),
                                                     rotate_vectors_z(refs, angle))
        worst = max(worst, float((src_k - src_0).abs().max()),
                    float((refs_k - refs_0).abs().max()))
    assert worst <= 1e-12, "float64 residual {:.3e} is too large to be pure rounding".format(worst)


def test_intrinsic_coords_track_genuine_relative_geometry():
    """Moving one reference *relative to* the query must change the representation."""
    torch.manual_seed(8)
    src = torch.tensor([[2.0, 1.0, 0.3]])
    refs = torch.tensor([[[1.0, -2.0, 0.1], [-1.5, 0.5, -0.2], [0.7, 2.5, 0.4]]])
    _, refs_0, _, _ = intrinsic_scene_coords(src, refs)
    moved = refs.clone()
    moved[0, 1, :2] = torch.tensor([0.5, -1.5])          # a genuinely different azimuth
    _, refs_1, _, _ = intrinsic_scene_coords(src, moved)
    assert float((refs_1 - refs_0).abs().max()) > 1e-2


def test_rotating_only_the_query_changes_the_representation():
    """Not a rigid motion: rotating the query alone must move the intrinsic coordinates.

    Guards against a transform that is invariant because it threw the azimuth away.
    """
    torch.manual_seed(9)
    src = torch.tensor([[2.0, 1.0, 0.3]])
    refs = torch.tensor([[[1.0, -2.0, 0.1], [-1.5, 0.5, -0.2]]])
    _, refs_0, _, _ = intrinsic_scene_coords(src, refs)
    _, refs_1, _, _ = intrinsic_scene_coords(rotate_vectors_z(src, 0.7), refs)
    assert float((refs_1 - refs_0).abs().max()) > 1e-2


def test_both_query_and_references_use_the_same_basis():
    """The query lands on ``(r_q, 0, z)``; references keep their relative azimuth."""
    src = torch.tensor([[3.0, 4.0, 1.25]])
    refs = torch.tensor([[[-4.0, 3.0, -0.5]]])
    src_i, refs_i, basis, blend = intrinsic_scene_coords(src, refs)
    assert float(blend[0]) == 1.0
    assert torch.allclose(basis, torch.tensor([[0.6, 0.8]]), atol=1e-6)
    assert torch.allclose(src_i, torch.tensor([[5.0, 0.0, 1.25]]), atol=1e-6)
    # ref is the query direction rotated by +90 deg, same radius -> (0, 5, -0.5)
    assert torch.allclose(refs_i, torch.tensor([[[0.0, 5.0, -0.5]]]), atol=1e-6)


def test_fallback_basis_is_the_sum_over_the_total_horizontal_radius():
    """Rule for a (near-)vertical query: ``a = sum_i p_h_i / sum_i ||p_h_i||``.

    Note ``||a|| < 1`` here: the references do not all point the same way, so the scene's
    horizontal direction is partly ambiguous and the representation attenuates it accordingly.
    """
    src = torch.tensor([[0.0, 0.0, 2.0]])
    refs = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 3.0, 0.5], [2.0, 0.0, -1.0]]])
    _, refs_i, basis, blend = intrinsic_scene_coords(src, refs)
    assert float(blend[0]) == 0.0, "a vertical query must not use its own horizontal direction"
    # R sums the *horizontal* norms: ||(1,0)|| + ||(0,3)|| + ||(2,0)|| = 6, not the 3-D ones.
    radius = 1.0 + 3.0 + 2.0
    assert torch.allclose(basis, torch.tensor([[3.0, 3.0]]) / radius, atol=1e-6)
    assert float(basis.norm()) == pytest.approx(math.sqrt(0.5), abs=1e-6)
    assert float(basis.norm()) < 1.0
    # Heights pass through untouched; the horizontal part is scaled by ||a||, not rotated only.
    assert torch.allclose(refs_i[..., 2], refs[..., 2], atol=0, rtol=0)
    assert torch.allclose(refs_i[..., :2].norm(dim=-1),
                          refs[..., :2].norm(dim=-1) * float(basis.norm()), atol=1e-6)


def test_fallback_basis_never_exceeds_unit_length():
    """``||a|| <= 1`` by the triangle inequality -- the property that bounds the conditioning."""
    generator = torch.Generator().manual_seed(21)
    src = torch.zeros(500, 3)
    src[:, 2] = 1.0
    refs = torch.randn(500, 6, 3, generator=generator) * 7.0
    basis, blend = horizontal_basis(src, refs)
    assert torch.equal(blend, torch.zeros(500))
    assert float(basis.norm(dim=-1).max()) <= 1.0 + 1e-7


def test_fallback_basis_is_tie_free_and_order_independent():
    """An aggregate, not a selection: no tie exists and the reference order cannot matter."""
    src = torch.tensor([[0.0, 0.0, 1.0]])
    tied = torch.tensor([[[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]])
    _, _, basis, _ = intrinsic_scene_coords(src, tied)
    for permutation in ([1, 0, 2], [2, 1, 0], [0, 2, 1]):
        _, _, permuted, _ = intrinsic_scene_coords(src, tied[:, permutation])
        assert torch.allclose(permuted, basis, atol=1e-12), \
            "reordering the references moved the basis: the fallback is still a selection"


def test_basis_has_no_branch_on_the_reference_aggregate():
    """Sweeping a cancelling reference pair across the old ``1e-6`` m guard must not jump.

    Review blocker B1(a) was exactly this crossing.  The measured contrast against the
    superseded rule is part of the assertion, so the test states what it is protecting.
    """
    sweep = validate.basis_smoothness_sweep()["cancellation"]
    assert sweep["max_step_round2"] > 0.5, "the superseded rule should visibly jump here"
    assert sweep["max_step_current"] < 1e-3, (
        "largest single step {:.3e} over {} configurations in [{:.1e}, {:.1e}]".format(
            sweep["max_step_current"], sweep["n_steps"], *sweep["range"]))


def test_basis_has_no_branch_on_the_query_horizontal_fraction():
    """The primary choice is a smooth blend, not a threshold -- same check, other branch."""
    sweep = validate.basis_smoothness_sweep()["query_blend"]
    assert sweep["blend_min"] == 0.0 and sweep["blend_max"] == 1.0, "the band was not crossed"
    assert sweep["max_step_round2"] > 0.5, "the superseded rule should visibly jump here"
    assert sweep["max_step_current"] < 0.15, (
        "largest single step {:.3e} over {} configurations".format(
            sweep["max_step_current"], sweep["n_steps"]))


@pytest.mark.parametrize("name,refs", [
    ("equal_radius_tie", validate.CODEX_TIE_REFS),
    ("sum_exactly_zero", validate.CODEX_CANCEL_REFS),
    ("near_cancellation", validate.CODEX_NEARCANCEL_REFS),
    ("all_horizontal_zero", validate.CODEX_SUM_DEGENERATE_REFS),
])
def test_adversarial_basis_coordinates_are_rotation_invariant(name, refs):
    """Every reported adversarial geometry, at the coordinate level, over all 15 C16 angles.

    The bound is *absolute* and scaled to the scene, because that is what the reviews measured:
    35.7 m, 4.4 m and 1.5 m of coordinate motion under the two superseded rules.
    """
    src = torch.tensor(validate.CODEX_TIE_SRC, dtype=torch.float32)
    refs = torch.tensor(refs, dtype=torch.float32)
    scale = float(refs.abs().max())
    src_0, refs_0, _, blend = intrinsic_scene_coords(src, refs)
    assert float(blend[0]) == 0.0, "these cases must all exercise the fallback"
    worst = 0.0
    for k in validate.C16_ANGLES:
        angle = yaw_angle_rad(k)
        src_k, refs_k, _, _ = intrinsic_scene_coords(rotate_vectors_z(src, angle),
                                                     rotate_vectors_z(refs, angle))
        worst = max(worst, float((src_k - src_0).abs().max()),
                    float((refs_k - refs_0).abs().max()))
    assert worst / scale <= FP32_INPUT_TOL, \
        "{}: coordinates moved {:.3e} m on a {:.1f} m scene".format(name, worst, scale)


@pytest.mark.parametrize("name,refs", [
    ("equal_radius_tie", validate.CODEX_TIE_REFS),
    ("sum_exactly_zero", validate.CODEX_CANCEL_REFS),
    ("near_cancellation", validate.CODEX_NEARCANCEL_REFS),
    ("all_horizontal_zero", validate.CODEX_SUM_DEGENERATE_REFS),
])
def test_adversarial_basis_full_model_is_invariant(name, refs, warm_cyl, env):
    """The same geometries through the whole warm-started model, both conditions.

    Reference values from the reviews, all at float32 on this same path: equal-radius tie
    1.478e-1, sum-exactly-zero 2.270e-1, near-cancellation 5.151e-2.
    """
    batch = validate.adversarial_batch(validate.CODEX_TIE_SRC, refs, label=name)
    with device_preserving_delay():
        sweep = validate.angle_sweep(warm_cyl[0], batch, validate.C16_ANGLES,
                                     conditions=("E", "P"))
    for condition in ("E", "P"):
        row = validate.worst(sweep["rows"], condition)
        assert row["rel_fro"] <= REL_TARGET, (
            "{} condition {} k={}: rel {:.3e} (abs {:.3e}, denom {:.3e})".format(
                name, condition, row["k"], row["rel_fro"], row["abs_fro"], row["denom_fro"]))
    print("\n{}: worst E {:.3e}, worst P {:.3e}".format(
        name, validate.worst(sweep["rows"], "E")["rel_fro"],
        validate.worst(sweep["rows"], "P")["rel_fro"]))


def test_directional_degeneracy_is_a_cross_geometry_jump_not_a_rotation_defect(warm_cyl, env):
    """Review round 3's counterexample, pinned in both directions at once.

    Query ``(5e-4, 0, 1)`` has a horizontal fraction of ``5.0e-4`` -- below the blend band, so the
    reference aggregate is in full control -- and the references ``(e, 0, z)`` all share one
    horizontal direction whose magnitude ``e`` can be made arbitrarily small.  Because ``s / R``
    follows the references' *directions* and not their magnitudes, ``a = (1, 0)`` for **every**
    ``e > 0`` while ``a = (0, 0)`` at ``e == 0``: the map is discontinuous in the geometry, and
    shrinking ``e`` does not approach the ``e = 0`` value.

    The test asserts both halves, because only together are they the honest statement:

    * **across geometries** -- neighbouring scenes really do differ, by ~``5.5e-5`` relative;
      this is recorded, not asserted away;
    * **across rotations** -- each of those geometries, held fixed, is invariant over all 15 C16
      angles in both conditions, well inside the ``1e-5`` target.  The discontinuity is therefore
      a smoothness property of the representation, not an invariance defect.
    """
    source = torch.tensor(validate.CODEX_CUSP_SRC, dtype=torch.float32)
    fraction = float(source[:, :2].norm() / source.norm())
    assert fraction < BLEND_LO, "the counterexample must sit below the blend band"

    predictions, bases = {}, {}
    for magnitude in validate.CODEX_CUSP_MAGNITUDES:
        refs = torch.tensor(validate.cusp_refs(magnitude), dtype=torch.float32)
        basis, blend = horizontal_basis(source, refs)
        bases[magnitude] = basis
        assert float(blend[0]) == 0.0, "the query direction must not participate here"

        batch = validate.adversarial_batch(validate.CODEX_CUSP_SRC,
                                           validate.cusp_refs(magnitude),
                                           label="cusp{:g}".format(magnitude))
        with device_preserving_delay():
            sweep = validate.angle_sweep(warm_cyl[0], batch, validate.C16_ANGLES,
                                         conditions=("E", "P"))
        predictions[magnitude] = sweep["out_0"]
        for condition in ("E", "P"):
            row = validate.worst(sweep["rows"], condition)
            assert row["rel_fro"] <= REL_TARGET, (
                "e={:g} condition {} k={}: rel {:.3e} -- a FIXED geometry must stay "
                "rotation-invariant".format(magnitude, condition, row["k"], row["rel_fro"]))

    # The basis does not tend to its e = 0 value as e shrinks: that is the discontinuity.
    assert torch.equal(bases[0.0], torch.zeros(1, 2))
    for magnitude in validate.CODEX_CUSP_MAGNITUDES:
        if magnitude > 0.0:
            assert torch.allclose(bases[magnitude], torch.tensor([[1.0, 0.0]]), atol=1e-6), \
                "e={:g} should give a unit basis regardless of how small e is".format(magnitude)

    baseline = predictions[0.0]
    jumps = {m: float((predictions[m] - baseline).norm() / baseline.norm())
             for m in validate.CODEX_CUSP_MAGNITUDES if m > 0.0}
    assert all(1e-5 < jump < 1e-3 for jump in jumps.values()), (
        "the documented cross-geometry jump moved: {}".format(jumps))
    print("\ncross-geometry jump at the directional degeneracy: {} "
          "(each geometry individually rotation-invariant to <= {:.0e})".format(
              {m: "{:.3e}".format(v) for m, v in jumps.items()}, REL_TARGET))


def test_reference_magnitude_does_not_set_the_fallback_direction():
    """The ratio ``s / R`` is scale-free in the references: only their directions matter.

    Stated as its own property because it is the mechanism behind the discontinuity above, and
    because it is also what makes the fallback well-conditioned when the sum is small.
    """
    source = torch.tensor([[0.0, 0.0, 1.0]])
    directions = torch.tensor([[[3.0, 4.0, 0.2], [-1.0, 2.0, -0.4]]])
    reference, _ = horizontal_basis(source, directions)
    for scale in (1e-6, 1e-3, 1.0, 1e3):
        scaled = directions.clone()
        scaled[..., :2] *= scale
        basis, _ = horizontal_basis(source, scaled)
        assert torch.allclose(basis, reference, atol=1e-6), \
            "scaling the references by {:g} moved the basis".format(scale)


def test_exactly_cancelling_references_give_exactly_zero_horizontal(warm_cyl, env):
    """The old "all horizontal components vanish" branch is now the continuous limit.

    It is still reached exactly: a reference set that cancels exactly yields ``a = 0`` and zero
    horizontal coordinates, bitwise unchanged at every angle.
    """
    src = torch.tensor(validate.CODEX_TIE_SRC, dtype=torch.float32)
    refs = torch.tensor(validate.CODEX_SUM_DEGENERATE_REFS, dtype=torch.float32)
    assert float(refs[0, :, :2].sum(dim=0).norm()) == 0.0
    src_0, refs_0, basis, _ = intrinsic_scene_coords(src, refs)
    assert torch.equal(basis, torch.zeros(1, 2))
    assert torch.equal(refs_0[..., :2], torch.zeros_like(refs_0[..., :2]))
    assert torch.equal(refs_0[..., 2], refs[..., 2])
    for k in validate.C16_ANGLES:
        angle = yaw_angle_rad(k)
        src_k, refs_k, _, _ = intrinsic_scene_coords(rotate_vectors_z(src, angle),
                                                     rotate_vectors_z(refs, angle))
        assert torch.equal(src_k, src_0) and torch.equal(refs_k, refs_0)


def test_all_references_on_the_axis_hits_the_exact_zero_guard():
    """The only remaining guard: ``R == 0`` returns the zero basis -- the rotation-
    invariant convention, NOT a continuous limit (see the directional-degeneracy
    counterexample: the ratio follows reference directions, not magnitudes)."""
    src = torch.tensor([[0.0, 0.0, 1.5]])
    refs = torch.tensor([[[0.0, 0.0, -2.0], [0.0, 0.0, 0.75]]])
    src_i, refs_i, basis, blend = intrinsic_scene_coords(src, refs)
    assert torch.equal(basis, torch.zeros(1, 2))
    assert float(blend[0]) == 0.0
    assert torch.equal(src_i, torch.tensor([[0.0, 0.0, 1.5]]))
    assert torch.equal(refs_i, torch.tensor([[[0.0, 0.0, -2.0], [0.0, 0.0, 0.75]]]))
    assert bool(torch.isfinite(src_i).all()) and bool(torch.isfinite(refs_i).all())


def test_zero_length_query_is_finite():
    """A source coincident with the receiver has no direction at all; it must not produce NaN."""
    src = torch.zeros(1, 3)
    refs = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 1.0, 0.5]]])
    src_i, refs_i, basis, blend = intrinsic_scene_coords(src, refs)
    assert float(blend[0]) == 0.0
    assert bool(torch.isfinite(basis).all())
    assert bool(torch.isfinite(src_i).all()) and bool(torch.isfinite(refs_i).all())


@pytest.mark.parametrize("fraction,expected", [
    (1.0, 1.0), (0.5, 1.0), (BLEND_HI * 2, 1.0), (BLEND_HI, 1.0),
    (BLEND_LO, 0.0), (BLEND_LO / 2, 0.0), (0.0, 0.0),
])
def test_blend_saturates_exactly_outside_the_band(fraction, expected):
    """Exactly 0 and exactly 1 outside ``[lo, hi]`` -- which is what keeps the primary path exact."""
    height = math.sqrt(max(1.0 - fraction ** 2, 0.0))
    src = torch.tensor([[fraction, 0.0, height]], dtype=torch.float64)
    refs = torch.tensor([[[1.0, 1.0, 0.0]]], dtype=torch.float64)
    _, blend = horizontal_basis(src, refs)
    assert float(blend[0]) == expected


def test_blend_band_is_scale_relative():
    """The band is a fraction of ``||q||``, so the same tilt means the same thing at any range.

    Review blocker B1 noted the round-2 threshold was an absolute ``1e-6`` m while the failing
    scene lived at 20-36 m.
    """
    refs = torch.tensor([[[1.0, 1.0, 0.0]]], dtype=torch.float64)
    for distance in (0.5, 1.0, 40.0):
        for fraction in (BLEND_LO / 2, (BLEND_LO + BLEND_HI) / 2, BLEND_HI * 2):
            height = math.sqrt(max(1.0 - fraction ** 2, 0.0))
            src = torch.tensor([[fraction * distance, 0.0, height * distance]],
                               dtype=torch.float64)
            _, blend = horizontal_basis(src, refs * distance)
            reference = torch.tensor([[fraction, 0.0, height]], dtype=torch.float64)
            _, expected = horizontal_basis(reference, refs)
            assert abs(float(blend[0]) - float(expected[0])) < 1e-9, \
                "blend depends on the absolute range: {} m".format(distance)


def test_blend_rejects_an_invalid_band():
    """``0 <= lo < hi`` is enforced rather than silently producing a division by zero."""
    src = torch.tensor([[1.0, 0.0, 1.0]])
    refs = torch.tensor([[[0.0, 4.0, 0.0]]])
    with pytest.raises(ValueError):
        horizontal_basis(src, refs, blend_lo=1e-2, blend_hi=1e-3)
    with pytest.raises(ValueError):
        horizontal_basis(src, refs, blend_lo=-1.0, blend_hi=1e-2)


def test_to_intrinsic_rejects_mismatched_shapes():
    """Shape contracts are enforced rather than broadcast into silence."""
    with pytest.raises(ValueError):
        to_intrinsic(torch.zeros(2, 4), torch.zeros(2, 2))
    with pytest.raises(ValueError):
        to_intrinsic(torch.zeros(2, 3), torch.zeros(3, 2))


def test_model_uses_the_documented_blend_band(random_cyl):
    """The model's band is the documented default and is reachable from the factory."""
    assert random_cyl.blend_lo == BLEND_LO and random_cyl.blend_hi == BLEND_HI
    other = build_xrir_exp08("cylindrical_invariant", 2, blend_lo=0.1, blend_hi=0.2)
    assert other.blend_lo == 0.1 and other.blend_hi == 0.2


def _round2_intrinsic_scene_coords(src_loc, ref_locs, **kwargs):
    """The round-2 coordinate transform, for the bitwise regression below (never used by a model)."""
    basis = validate.superseded_round2_basis(src_loc, ref_locs)
    return (to_intrinsic(src_loc, basis), to_intrinsic(ref_locs, basis), basis,
            torch.zeros(src_loc.shape[0], dtype=src_loc.dtype))


def test_battery_coordinates_are_bitwise_unchanged_by_the_blend(real):
    """Every real query sits far above the band, so the round-3 basis reproduces round 2 exactly.

    The blend must buy robustness in the degenerate corner without moving the production path by
    a single ulp.
    """
    basis, blend = horizontal_basis(real["src_loc"], real["ref_locs"])
    assert torch.equal(blend, torch.ones_like(blend)), "a real query fell inside the blend band"
    legacy = validate.superseded_round2_basis(real["src_loc"], real["ref_locs"])
    assert torch.equal(basis, legacy), "the basis moved on the battery"
    for new_out, old_out in zip(intrinsic_scene_coords(real["src_loc"], real["ref_locs"])[:2],
                                _round2_intrinsic_scene_coords(real["src_loc"],
                                                               real["ref_locs"])[:2]):
        assert torch.equal(new_out, old_out)
    fraction = (real["src_loc"][:, :2].double().norm(dim=-1)
                / real["src_loc"].double().norm(dim=-1))
    print("\nbattery horizontal fraction min {:.4f} vs band top {:.0e}".format(
        float(fraction.min()), BLEND_HI))


def test_battery_predictions_are_bitwise_unchanged_by_the_blend(real, warm_cyl, env, monkeypatch):
    """The same statement end to end: the model's output on the real battery is bit-identical.

    The round-2 coordinate rule is patched into the module the forward pass resolves it from,
    and the two predictions are compared with ``torch.equal`` -- not a tolerance.
    """
    import model.xRIR_cyl_invariant as invariant_module

    model = warm_cyl[0]
    with device_preserving_delay(), torch.no_grad():
        current, _ = model(real["depth_coord"], real["ref_irs"], real["src_loc"],
                           real["ref_locs"], real["tgt_wav"])
        monkeypatch.setattr(invariant_module, "intrinsic_scene_coords",
                            _round2_intrinsic_scene_coords)
        legacy, _ = model(real["depth_coord"], real["ref_irs"], real["src_loc"],
                          real["ref_locs"], real["tgt_wav"])
    assert torch.equal(current, legacy), "max |d| {:.3e}".format(
        float((current - legacy).abs().max()))


# --------------------------------------------------------------------------------------
# The device-preserving delay shim (brief item 4) -- proven equivalent to the pinned code
# --------------------------------------------------------------------------------------

def _delay_cases():
    torch.manual_seed(10)
    signal = torch.randn(5, 64)
    return signal, torch.tensor([0, 7, -7, 63, -63], dtype=torch.int32)


def test_device_preserving_delay_matches_the_pinned_semantics(monkeypatch):
    """Identical delays, gains and zero padding as ``model.xRIR.apply_delay``.

    ``Tensor.cuda`` is monkeypatched to the identity so the *pinned* function can run on CPU
    unmodified; the comparison is then elementwise equality, not a tolerance.
    """
    signal, delays = _delay_cases()
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self, raising=False)
    pinned = xrir_module.apply_delay(signal, delays)
    shimmed = apply_delay_device_preserving(signal, delays)
    assert torch.equal(pinned, shimmed)
    assert pinned.shape == signal.shape and shimmed.dtype == signal.dtype
    # The padding is genuinely zero on the vacated side, and the payload is the shifted signal.
    assert torch.equal(shimmed[1, :7], torch.zeros(7))
    assert torch.equal(shimmed[1, 7:], signal[1, :-7])
    assert torch.equal(shimmed[2, -7:], torch.zeros(7))
    assert torch.equal(shimmed[2, :-7], signal[2, 7:])
    assert torch.equal(shimmed[0], signal[0])


def test_device_preserving_delay_matches_pinned_on_random_delays(monkeypatch):
    """The same equality over many random delays, including the full-length extremes."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self, raising=False)
    generator = torch.Generator().manual_seed(11)
    for _ in range(20):
        signal = torch.randn(4, 96, generator=generator)
        delays = torch.randint(-95, 96, (4,), generator=generator, dtype=torch.int32)
        assert torch.equal(xrir_module.apply_delay(signal, delays),
                           apply_delay_device_preserving(signal, delays))


@pytest.mark.skipif(torch.cuda.is_available(),
                    reason="the pinned .cuda() only fails when no GPU is visible")
def test_pinned_apply_delay_cannot_run_on_cpu_but_the_shim_can():
    """Why the shim exists: the pinned function hard-codes ``.cuda()`` on its output buffer."""
    signal, delays = _delay_cases()
    with pytest.raises(RuntimeError):
        xrir_module.apply_delay(signal, delays)
    out = apply_delay_device_preserving(signal, delays)
    assert out.device == signal.device


def test_device_preserving_delay_context_manager_installs_and_restores():
    """The swap is scoped: ``model.xRIR`` is left exactly as it was found."""
    original = xrir_module.apply_delay
    with device_preserving_delay() as installed:
        assert xrir_module.apply_delay is installed is apply_delay_device_preserving
    assert xrir_module.apply_delay is original
    with pytest.raises(RuntimeError):
        with device_preserving_delay():
            raise RuntimeError("boom")
    assert xrir_module.apply_delay is original


# --------------------------------------------------------------------------------------
# Section 4 items 3 and 4 -- the whole model, spectra and waveforms
# --------------------------------------------------------------------------------------

def test_all_fifteen_c16_angles_are_covered():
    """The claim is C16: 15 non-identity angles, 22.5 deg apart, none dropped."""
    assert len(validate.C16_ANGLES) == 15
    assert validate.C16_ANGLES == tuple(range(32, 512, 32))
    degrees = [math.degrees(yaw_angle_rad(k)) for k in validate.C16_ANGLES]
    assert degrees[0] == pytest.approx(22.5) and degrees[-1] == pytest.approx(337.5)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("k", validate.C16_ANGLES)
def test_cyl_invariant_spectrum_is_c16_invariant(sweeps, case, k):
    """Condition E, every case, every angle: the full predicted spectrum is invariant.

    Condition E is the true end-to-end path -- the scene is rotated and ``shift_and_align``
    recomputes distances, direct-path gains and integer delays from the rotated coordinates.
    The absolute residual and the denominator travel with the relative number, as the handoff
    requires.
    """
    row = next(r for r in sweeps[case]["rows"] if r["k"] == k and r["condition"] == "E")
    assert row["rel_fro"] <= REL_TARGET, (
        "{} k={}: rel {:.3e} > {:.0e} (abs {:.3e}, ||out_0||_F {:.3e}, max|d| {:.3e})".format(
            case, k, row["rel_fro"], REL_TARGET, row["abs_fro"], row["denom_fro"],
            row["max_abs"]))


@pytest.mark.parametrize("case", CASES)
def test_every_row_of_every_case_is_invariant(sweeps, case):
    """Per-sample, not just per-batch: no row may hide behind the batch's Frobenius norm."""
    for row in sweeps[case]["rows"]:
        if row["condition"] != "E":
            continue
        worst = max(row["rel_fro_per_row"])
        assert worst <= REL_TARGET, "{} k={}: worst row rel {:.3e}".format(
            case, row["k"], worst)


def test_pinned_alignment_condition_also_holds(sweeps):
    """Condition P is reported too, and is at least as tight as the end-to-end condition."""
    for case in ("random_init/real", "warm_start/real"):
        rows = [r for r in sweeps[case]["rows"] if r["condition"] == "P"]
        assert len(rows) == 15
        assert max(r["rel_fro"] for r in rows) <= REL_TARGET


def test_condition_E_really_recomputes_the_alignment(real, warm_cyl):
    """The end-to-end path is exercised: the rotated alignment is recomputed, not reused.

    Guards the whole section-4-item-3 result against the failure mode the handoff calls out
    -- "zero error obtained only from a fixed, unrotated alignment cache".
    """
    model = warm_cyl[0]
    with device_preserving_delay(), torch.no_grad():
        aligned0 = model.shift_and_align(real["ref_irs"], real["src_loc"], real["ref_locs"])
        _, src_k, refs_k = rotate_scene_yaw(real["depth_coord"], real["src_loc"],
                                            real["ref_locs"], 32)
        aligned_k = model.shift_and_align(real["ref_irs"], src_k, refs_k)
    assert not torch.equal(aligned_k, aligned0), \
        "the rotated alignment is bit-identical; shift_and_align was not re-run"
    assert torch.allclose(aligned_k, aligned0, atol=1e-5, rtol=1e-4), \
        "the recomputed alignment moved further than float32 rounding"


#: Review blocker B2: the angles at which the adversarial 3.5-tie scene's integer delay moves,
#: measured with the float64 reduction the invariant arms use.  90 / 180 / 270 deg are absent
#: because ``Rz`` is exact in float32 there, so the coordinates come back unchanged.
#: If a future precision change empties this set, the test below FAILS -- update it *and* the
#: qualification in validation_report.md together, never one without the other.
CODEX_DELAY_EXPECTED_FLIP_ANGLES = (32, 64, 96, 160, 192, 224, 288, 320, 352, 416, 448, 480)


def test_invariant_alignment_matches_the_pinned_one_on_the_battery(real, warm_cyl):
    """The float64 delay reduction changes nothing measurable on non-adversarial geometry.

    Same integer delays bitwise, and the aligned reference audio within float32 rounding of the
    pinned computation -- so blocker B2's fix is a precision change, not a behaviour change.
    """
    from model.xRIR import xRIR as PinnedXRIR

    model = warm_cyl[0]
    with device_preserving_delay(), torch.no_grad():
        shimmed = model.shift_and_align(real["ref_irs"], real["src_loc"], real["ref_locs"])
        pinned = PinnedXRIR.shift_and_align(model, real["ref_irs"], real["src_loc"],
                                            real["ref_locs"])
    assert torch.equal(integer_delays(real["src_loc"], real["ref_locs"]),
                       validate.integer_delays64(real["src_loc"], real["ref_locs"])), \
        "the two reductions disagree on the real battery's delays"
    relative = float((shimmed - pinned).norm() / pinned.norm())
    assert relative <= 1e-6, "aligned audio moved by rel {:.3e}".format(relative)
    print("\nfloat64 alignment vs pinned on the real battery: rel {:.3e}, delays identical".format(
        relative))


def test_float64_delay_reduction_lowers_the_flip_rate():
    """Blocker B2's "the boundary shrinks", measured rather than asserted.

    A random ensemble is rotated through all 15 C16 angles and the integer delays compared,
    once per reduction.

    Flips **correlate within a scene** -- one scene sitting on a tie flips at most angles -- so
    the totals are not independent Poisson counts and the naive ``sqrt(n32 + n64)`` overstates
    the significance (review round 2, B2(i)).  The test uses the **paired per-scene difference**
    instead, and demands 3 sigma on that.

    It is a *narrowing*, not an elimination: float64 removes the arithmetic half of the noise,
    while the float32 rounding of the rotated input coordinates is irreducible (the model never
    receives the exactly-rotated scene).
    """
    result = validate.delay_flip_rate(n_scenes=1000000)
    f32, f64, paired = result["float32"], result["float64"], result["paired"]
    assert paired["sigma"] < paired["naive_poisson_sigma"], \
        "the paired estimate must be the conservative one"
    assert paired["sigma"] > 3.0, (
        "float32 {} flips vs float64 {}: paired difference {:.0f} +- {:.2f} is only "
        "{:.2f} sigma".format(f32["flips"], f64["flips"], paired["total_difference"],
                              paired["standard_error"], paired["sigma"]))
    print("\ndelay flip rate over {} comparisons: float32 {} ({:.3e}), float64 {} ({:.3e}), "
          "shrink {:.3f}x; paired {:.0f} +- {:.2f} = {:.2f} sigma (naive Poisson would claim "
          "{:.2f})".format(result["comparisons"], f32["flips"], f32["rate"], f64["flips"],
                           f64["rate"], result["shrink_factor"], paired["total_difference"],
                           paired["standard_error"], paired["sigma"],
                           paired["naive_poisson_sigma"]))


def test_codex_delay_boundary_case_is_either_invariant_or_a_documented_flip(warm_cyl, env):
    """Review blocker B2, the exact reported case: a pre-round delay sitting on the 3.5 tie.

    Per angle the outcome must be one of exactly two things, and which one is *pinned*:

    * no integer-delay flip  -> the end-to-end (condition E) residual must meet the C16 target;
    * a flip                 -> it must be a **one-sample** move, condition P (the alignment
      held at ``k = 0``) must still meet the target -- proving the whole effect is the integer
      delay and nothing else -- and the E residual is reported as the documented exception.

    The observed flip set is compared against :data:`CODEX_DELAY_EXPECTED_FLIP_ANGLES`, so the
    test can neither pass silently if the flips disappear nor if new ones appear.
    """
    batch = validate.adversarial_batch(validate.CODEX_DELAY_SRC, validate.CODEX_DELAY_REFS,
                                       label="codexDelay")
    preround = validate.preround_delay(batch["src_loc"], batch["ref_locs"])
    assert abs(float(preround[0, 0]) - 3.5) < 1e-5, "the case no longer sits on the 3.5 tie"

    with device_preserving_delay():
        sweep = validate.angle_sweep(warm_cyl[0], batch, validate.C16_ANGLES,
                                     conditions=("E", "P"))
    base = validate.integer_delays64(batch["src_loc"], batch["ref_locs"])

    # Whatever the integer delay does, everything else is invariant: P holds at every angle.
    worst_p = validate.worst(sweep["rows"], "P")
    assert worst_p["rel_fro"] <= REL_TARGET, (
        "condition P failed at k={} with rel {:.3e}: the effect is NOT confined to the integer "
        "delay".format(worst_p["k"], worst_p["rel_fro"]))

    observed, exceptions = [], []
    for k in validate.C16_ANGLES:
        angle = yaw_angle_rad(k)
        rotated = validate.integer_delays64(rotate_vectors_z(batch["src_loc"], angle),
                                            rotate_vectors_z(batch["ref_locs"], angle))
        n_flips = int((rotated != base).sum())
        row = next(r for r in sweep["rows"] if r["k"] == k and r["condition"] == "E")
        if n_flips == 0:
            assert row["rel_fro"] <= REL_TARGET, (
                "k={} has no delay flip but rel {:.3e} > {:.0e}".format(
                    k, row["rel_fro"], REL_TARGET))
        else:
            observed.append(k)
            shift = int((rotated - base).abs().max())
            assert shift == 1, "k={} moved a delay by {} samples, not 1".format(k, shift)
            assert n_flips == 1, "k={} flipped {} delays, not 1".format(k, n_flips)
            exceptions.append((k, row["rel_fro"]))

    assert tuple(observed) == CODEX_DELAY_EXPECTED_FLIP_ANGLES, (
        "the documented flip set changed: observed {} vs documented {}. Update both this "
        "constant and the qualification in validation_report.md.".format(
            tuple(observed), CODEX_DELAY_EXPECTED_FLIP_ANGLES))
    print("\nB2 boundary case: pre-round {:.9f} samples; {} of 15 angles flip one delay by one "
          "sample; E residual there {:.3e}..{:.3e}; condition P worst {:.3e}".format(
              float(preround[0, 0]), len(exceptions), min(e[1] for e in exceptions),
              max(e[1] for e in exceptions), worst_p["rel_fro"]))


def test_delay_flip_counter_follows_the_models_own_reduction(real, warm_cyl, env):
    """An invariant arm must be audited with float64 delays, a pinned arm with float32."""
    torch.manual_seed(0)
    pinned = build_xrir_exp08("cylindrical", 8).eval()
    boundary = validate.adversarial_batch(validate.CODEX_DELAY_SRC, validate.CODEX_DELAY_REFS,
                                          label="codexDelay")
    for model, expect64 in ((warm_cyl[0], True), (pinned, False)):
        counts = validate.delay_flip_counts_for(model, boundary["src_loc"],
                                                boundary["ref_locs"], (32,))
        reference = (validate.integer_delays64 if expect64 else integer_delays)
        base = reference(boundary["src_loc"], boundary["ref_locs"])
        rotated = reference(rotate_vectors_z(boundary["src_loc"], yaw_angle_rad(32)),
                            rotate_vectors_z(boundary["ref_locs"], yaw_angle_rad(32)))
        assert counts[32] == int((rotated != base).sum())


def test_delay_flips_are_recorded(sweeps):
    """The known float32 rounding boundary in ``shift_and_align`` is counted, not assumed away."""
    flips = sweeps["warm_start/real"]["delay_flips"]
    assert set(flips) == set(validate.C16_ANGLES)
    assert all(isinstance(v, int) and v >= 0 for v in flips.values())


def test_waveform_residuals_are_bounded_by_griffin_lims_own_conditioning(sweeps, real):
    """Section 4 item 4: the *waveform*, not only the spectrum.

    Seeded Griffin-Lim with a per-query, yaw-independent phase seed at ``k = 0`` and at every
    angle.  Griffin-Lim is an iterative non-convex inversion, so it amplifies any input
    change; the assertion is therefore that a rotation moves the waveform no more than an
    equally sized *non-rotational* perturbation does, and that the inversion is deterministic.
    """
    sweep = sweeps["warm_start/real"]
    rows = validate.waveform_residuals(sweep["out_0"], sweep["predictions"], real["keys"],
                                       gl_seed=0)
    assert len(rows) == 15 * len(real["keys"])
    rotation_worst = max(r["rel_l2"] for r in rows)
    control = validate.waveform_conditioning_control(
        sweep["out_0"], real["keys"], validate.worst(sweep["rows"], "E")["rel_fro"], gl_seed=0)
    control_worst = max(r["rel_l2"] for r in control["rows"])
    assert control["deterministic"], "Griffin-Lim is not reproducible at a fixed seed"
    assert all(r["denom_l2"] > 0 for r in rows), "a degenerate k=0 waveform would hide the test"
    assert rotation_worst <= 10.0 * max(control_worst, 1e-12), (
        "rotation moved the waveform by {:.3e} but an equally sized non-rotational "
        "perturbation only moved it by {:.3e}".format(rotation_worst, control_worst))
    assert rotation_worst < 1e-2, "waveform rel-L2 {:.3e} is not a rounding effect".format(
        rotation_worst)


def test_c16_residual_is_the_float32_input_rounding_floor(real, warm_cyl, sweeps):
    """The residual is numerical, not architectural.

    Rotating the scene and rotating it straight back is the identity in exact arithmetic, so
    what comes back differs from the original only by the float32 rounding a rotation
    introduces -- while being geometrically **unrotated**.  Pushing that through the model
    measures how far the output moves for purely numerical reasons.  The C16 residual must not
    exceed that floor by more than a small factor, or something in the architecture (not the
    arithmetic) is responsible for it.
    """
    model = warm_cyl[0]
    rotation_worst = validate.worst(sweeps["warm_start/real"]["rows"], "E")["rel_fro"]
    with device_preserving_delay():
        controls = {k: validate.input_rounding_control(model, real, k) for k in (32, 96, 224)}
    floor = max(c["output"]["rel_fro"] for c in controls.values())
    assert floor > 0.0, "the control perturbation vanished; pick an angle with a non-trivial Rz"
    assert rotation_worst <= 5.0 * floor, (
        "C16 residual {:.3e} is {:.1f}x the pure float32 input-rounding floor {:.3e}".format(
            rotation_worst, rotation_worst / floor, floor))
    print("\ninput-rounding floor {:.3e} vs worst C16 residual {:.3e}".format(
        floor, rotation_worst))
    for k, c in sorted(controls.items()):
        print("  k={:3d}: input rel depth {:.3e} src {:.3e} refs {:.3e} -> output rel {:.3e}".format(
            k, c["input_rel_depth"], c["input_rel_src"], c["input_rel_refs"],
            c["output"]["rel_fro"]))


def test_harness_detects_a_non_invariant_model(sweeps):
    """Non-vacuity: the *pinned* cylindrical model must fail the very same criterion."""
    worst = validate.worst(sweeps["control_pinned"]["rows"], "E")
    assert worst["rel_fro"] > 100 * REL_TARGET, (
        "the pinned cylindrical model scored {:.3e}; if the harness cannot see its known "
        "non-invariance it proves nothing about the new model".format(worst["rel_fro"]))


def test_off_lattice_angles_are_reported_separately(sweeps):
    """Off-lattice rolls are a diagnostic: finite, recorded, and never judged as C16."""
    rows = [r for r in sweeps["off_lattice"]["rows"] if r["condition"] == "E"]
    assert len(rows) == len(validate.OFF_LATTICE_ANGLES)
    assert all(k % 32 != 0 for k in validate.OFF_LATTICE_ANGLES)
    for row in rows:
        assert math.isfinite(row["rel_fro"]) and row["rel_fro"] >= 0.0
    print("\noff-lattice diagnostic (warm_start/real, condition E):")
    for row in rows:
        print("  k={:3d} ({:5.1f} deg)  rel {:.3e}  abs {:.3e}  max|d| {:.3e}".format(
            row["k"], row["angle_deg"], row["rel_fro"], row["abs_fro"], row["max_abs"]))


def test_simple_invariant_carries_no_guarantee(real, env):
    """The 2x2 control arm runs and is reported; no invariance is claimed for it."""
    model, _, _ = build_warm_started("simple_invariant", 8, verbose=False)
    with device_preserving_delay():
        sweep = validate.angle_sweep(model.eval(), real, (32, 256), conditions=("E",))
    values = [r["rel_fro"] for r in sweep["rows"]]
    assert all(math.isfinite(v) for v in values)
    print("\nsimple_invariant (no guarantee): rel {}".format(
        ["{:.3e}".format(v) for v in values]))


def test_invariant_classes_are_distinct_and_pin_their_encoders(random_cyl):
    """``cylindrical_invariant`` pins CylindricalViT; ``simple_invariant`` keeps SimpleViT."""
    from model.cylindrical_vit import CylindricalViT
    from model.simple_vit import SimpleViT

    assert isinstance(random_cyl, xRIR_CylInvariant)
    assert type(random_cyl.source_network) is CylindricalViT
    simple = build_xrir_exp08("simple_invariant", 2)
    assert isinstance(simple, xRIR_SimpleInvariant)
    assert type(simple.source_network) is SimpleViT
    for m in (random_cyl, simple):
        assert not hasattr(m, "lin_proj_0"), "the dead token pool must be gone"
        assert isinstance(m.invariant_readout, InvariantReadout)


# --------------------------------------------------------------------------------------
# Section 4 item 5 -- one real forward / backward / optimizer step
# --------------------------------------------------------------------------------------

def test_gradients_reach_the_new_readout_and_the_coordinate_branch(real, env):
    """Finite gradients that actually reach the new parameters, and pinned ones still move."""
    torch.manual_seed(0)
    model = build_xrir_exp08("cylindrical_invariant", 8)
    with device_preserving_delay():
        evidence = validate.grad_step_evidence(model, real)
    assert evidence["all_grads_finite"], "a non-finite gradient appeared"
    assert math.isfinite(evidence["loss"]) and evidence["loss"] > 0.0
    for name in validate.NEW_PARAMS + validate.COORD_PARAMS:
        entry = evidence["watched"][name]
        assert entry["grad_norm"] is not None and entry["grad_norm"] > 0.0, \
            "{} received no gradient".format(name)
        assert entry["max_delta"] > 0.0, "{} did not move after the optimizer step".format(name)
    for name in validate.OLD_PARAMS:
        entry = evidence["watched"][name]
        assert entry["grad_norm"] > 0.0 and entry["max_delta"] > 0.0, \
            "pinned parameter {} stopped training".format(name)
    assert evidence["tensors_changed_by_step"] == evidence["tensors_with_nonzero_grad"]
    print("\ngrad step: loss {:.4f}, {}/{} trainable tensors updated".format(
        evidence["loss"], evidence["tensors_changed_by_step"], evidence["trainable_tensors"]))


def test_only_the_pinned_dead_parameters_get_no_gradient(real, env):
    """The tensors without a gradient are xRIR's pre-existing unused heads, not exp_08's."""
    torch.manual_seed(0)
    model = build_xrir_exp08("cylindrical_invariant", 8)
    with device_preserving_delay():
        evidence = validate.grad_step_evidence(model, real)
    silent = sorted(name for name, norm in evidence["grad_norms"].items()
                    if norm is None or norm == 0.0)
    assert silent == ["audio_enc.cnn.fc_backup.bias", "audio_enc.cnn.fc_backup.weight",
                      "source_proj.proj.bias", "source_proj.proj.weight"], silent


# --------------------------------------------------------------------------------------
# Section 4 item 6 -- the pinned entries are untouched
# --------------------------------------------------------------------------------------

def test_factory_keeps_the_pinned_classes_by_identity():
    """``is``-identity, so exp_01/exp_03 results stay attributable to the code that made them."""
    assert BACKBONES_EXP08["simple"] is xRIR
    assert BACKBONES_EXP08["cylindrical"] is xRIR_Cyl
    assert BACKBONES_EXP08["cylindrical_invariant"] is xRIR_CylInvariant
    assert BACKBONES_EXP08["simple_invariant"] is xRIR_SimpleInvariant
    assert sorted(BACKBONES_EXP08) == ["cylindrical", "cylindrical_invariant", "simple",
                                       "simple_invariant"]
    with pytest.raises(ValueError):
        build_xrir_exp08("nope", 8)


def test_pinned_registry_is_not_mutated():
    """Adding the exp_08 arms must not leak back into ``model.xRIR_cyl.BACKBONES``."""
    from model.xRIR_cyl import BACKBONES
    assert sorted(BACKBONES) == ["cylindrical", "simple"]


def test_pinned_outputs_are_bit_identical_through_the_factory(real, env):
    """A fixed real batch: the factory's ``simple`` / ``cylindrical`` match the classes exactly."""
    with device_preserving_delay():
        result = validate.pinned_regression(real, seed=0, num_shot=8)
    for name, entry in result.items():
        assert entry["class_identity"], "{} is not the pinned class".format(name)
        assert entry["bitwise_equal"], "{} differs by {:.3e}".format(name, entry["max_abs_diff"])


def test_xrir_module_source_is_untouched():
    """``model/xRIR.py`` keeps its hard-coded ``.cuda()``: exp_08 shims, it does not edit."""
    import inspect
    source = inspect.getsource(xrir_module.apply_delay)
    assert "torch.zeros_like(signal).cuda()" in source


# --------------------------------------------------------------------------------------
# Warm start accounting
# --------------------------------------------------------------------------------------

def test_warm_start_accounts_for_every_parameter(warm_cyl):
    """Nothing is silently dropped: the buckets partition both key sets, and none is empty."""
    _, accounting, path = warm_cyl
    counts = accounting["counts"]
    assert counts["dropped_shape"] == 0 and counts["dropped_absent"] == 0
    assert counts["new_random"] == 0, "every new tensor must be explained"
    assert counts["consumed"] == 2 and counts["new_warm"] == 2
    assert (counts["loaded"] + counts["consumed"] == counts["checkpoint_total"])
    assert (counts["loaded"] + counts["new_warm"] == counts["model_total"])
    assert [r["key"] for r in accounting["consumed"]] == ["lin_proj_0.proj.weight",
                                                          "lin_proj_0.proj.bias"]
    assert accounting["strict_load"]["missing"] == ["invariant_readout.bias",
                                                    "invariant_readout.weight"]
    assert path.endswith("xRIR_cyl_8_shot/epoch_12.pth")


def test_warm_start_readout_matches_the_checkpoints_pool(warm_cyl):
    """``invariant_readout.weight`` really is the checkpoint's ``lin_proj_0`` folded over azimuth."""
    from tools.exp08_warmstart import load_checkpoint_state, SOURCE_CHECKPOINTS

    model, _, _ = warm_cyl
    state = load_checkpoint_state(SOURCE_CHECKPOINTS["cylindrical_invariant"])
    expected = state["lin_proj_0.proj.weight"].reshape(16, 16).sum(dim=1)
    assert torch.equal(model.invariant_readout.weight.detach(), expected)
    assert torch.equal(model.invariant_readout.bias.detach(), state["lin_proj_0.proj.bias"])


def test_warm_start_refuses_a_missing_checkpoint():
    """A missing checkpoint stops the run; it is never quietly replaced by random init."""
    with pytest.raises(FileNotFoundError):
        build_warm_started("cylindrical_invariant", 8, checkpoint="/nonexistent/epoch_12.pth")
    with pytest.raises(ValueError):
        build_warm_started("cylindrical", 8)
