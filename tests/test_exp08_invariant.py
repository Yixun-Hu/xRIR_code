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
from model.xRIR_cyl_invariant import (BASIS_DEGENERATE, BASIS_QUERY, BASIS_REFERENCE,
                                      DEFAULT_BASIS_EPS, InvariantReadout,
                                      horizontal_basis, intrinsic_scene_coords,
                                      to_intrinsic, xRIR_CylInvariant,
                                      xRIR_SimpleInvariant)
from tools.exp08_warmstart import build_warm_started
from tools.yaw_rotation import rotate_scene_yaw, rotate_vectors_z, yaw_angle_rad

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
    src_0, refs_0, _, mode = intrinsic_scene_coords(src, refs)
    assert int(mode.min()) == BASIS_QUERY, "generic positions must use the query basis"
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
    src_i, refs_i, basis, mode = intrinsic_scene_coords(src, refs)
    assert int(mode[0]) == BASIS_QUERY
    assert torch.allclose(basis, torch.tensor([[0.6, 0.8]]), atol=1e-6)
    assert torch.allclose(src_i, torch.tensor([[5.0, 0.0, 1.25]]), atol=1e-6)
    # ref is the query direction rotated by +90 deg, same radius -> (0, 5, -0.5)
    assert torch.allclose(refs_i, torch.tensor([[[0.0, 5.0, -0.5]]]), atol=1e-6)


def test_degenerate_query_on_the_vertical_axis_uses_the_widest_reference():
    """Rule 2: a (near-)vertical query falls back to the reference of largest horizontal radius."""
    src = torch.tensor([[0.0, 0.0, 2.0]])
    refs = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 3.0, 0.5], [2.0, 0.0, -1.0]]])
    _, refs_i, basis, mode = intrinsic_scene_coords(src, refs)
    assert int(mode[0]) == BASIS_REFERENCE
    assert torch.allclose(basis, torch.tensor([[0.0, 1.0]]), atol=1e-6)   # the r = 3 reference
    assert torch.allclose(refs_i[0, 1], torch.tensor([3.0, 0.0, 0.5]), atol=1e-6)


def test_degenerate_fallback_basis_is_also_yaw_invariant():
    """The fallback must not reintroduce a yaw dependence: the radii and the index are invariant."""
    src = torch.tensor([[0.0, 0.0, 2.0]])
    refs = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 3.0, 0.5], [2.0, 0.0, -1.0]]])
    src_0, refs_0, _, _ = intrinsic_scene_coords(src, refs)
    for k in validate.C16_ANGLES:
        angle = yaw_angle_rad(k)
        src_k, refs_k, _, mode = intrinsic_scene_coords(rotate_vectors_z(src, angle),
                                                        rotate_vectors_z(refs, angle))
        assert int(mode[0]) == BASIS_REFERENCE
        assert float((src_k - src_0).abs().max()) <= 1e-5
        assert float((refs_k - refs_0).abs().max()) <= 1e-5


def test_degenerate_tie_breaks_to_the_lowest_reference_index():
    """Rule 2's tie-break: equal radii pick the first reference in the manifest's order."""
    src = torch.tensor([[0.0, 0.0, 1.0]])
    tied = torch.tensor([[[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]])
    _, _, basis, mode = intrinsic_scene_coords(src, tied)
    assert int(mode[0]) == BASIS_REFERENCE
    assert torch.allclose(basis, torch.tensor([[0.0, -1.0]]), atol=1e-6)  # index 0, not 1 or 2
    # Reordering the references changes which one wins -- the rule is about the index.
    reordered = tied[:, [1, 0, 2]]
    _, _, basis_r, _ = intrinsic_scene_coords(src, reordered)
    assert torch.allclose(basis_r, torch.tensor([[1.0, 0.0]]), atol=1e-6)


def test_degenerate_all_horizontal_zero_gives_zero_horizontal_output():
    """Rule 3: nothing to point at -> zero horizontal components, no NaN."""
    src = torch.tensor([[0.0, 0.0, 1.5]])
    refs = torch.tensor([[[0.0, 0.0, -2.0], [0.0, 0.0, 0.75]]])
    src_i, refs_i, basis, mode = intrinsic_scene_coords(src, refs)
    assert int(mode[0]) == BASIS_DEGENERATE
    assert torch.equal(basis, torch.zeros(1, 2))
    assert torch.equal(src_i, torch.tensor([[0.0, 0.0, 1.5]]))
    assert torch.equal(refs_i, torch.tensor([[[0.0, 0.0, -2.0], [0.0, 0.0, 0.75]]]))
    assert bool(torch.isfinite(src_i).all()) and bool(torch.isfinite(refs_i).all())


@pytest.mark.parametrize("radius,expected", [
    (DEFAULT_BASIS_EPS * 10.0, BASIS_QUERY),        # comfortably above
    (DEFAULT_BASIS_EPS * 1.5, BASIS_QUERY),         # just above
    (DEFAULT_BASIS_EPS, BASIS_REFERENCE),           # exactly at the threshold -> not the query
    (DEFAULT_BASIS_EPS * 0.5, BASIS_REFERENCE),     # just below
    (0.0, BASIS_REFERENCE),                         # exactly on the axis
])
def test_basis_threshold_boundary(radius, expected):
    """The documented boundary: strictly greater than ``eps`` uses the query, otherwise not."""
    src = torch.tensor([[float(radius), 0.0, 1.0]], dtype=torch.float64)
    refs = torch.tensor([[[0.0, 4.0, 0.0]]], dtype=torch.float64)
    _, _, _, mode = intrinsic_scene_coords(src, refs, eps=DEFAULT_BASIS_EPS)
    assert int(mode[0]) == expected


def test_basis_threshold_is_configurable_and_validated():
    """``eps`` is an explicit parameter; a non-positive threshold is refused."""
    src = torch.tensor([[1.0e-3, 0.0, 1.0]])
    refs = torch.tensor([[[0.0, 4.0, 0.0]]])
    assert int(horizontal_basis(src, refs, eps=1e-6)[1][0]) == BASIS_QUERY
    assert int(horizontal_basis(src, refs, eps=1e-2)[1][0]) == BASIS_REFERENCE
    with pytest.raises(ValueError):
        horizontal_basis(src, refs, eps=0.0)


def test_to_intrinsic_rejects_mismatched_shapes():
    """Shape contracts are enforced rather than broadcast into silence."""
    with pytest.raises(ValueError):
        to_intrinsic(torch.zeros(2, 4), torch.zeros(2, 2))
    with pytest.raises(ValueError):
        to_intrinsic(torch.zeros(2, 3), torch.zeros(3, 2))


def test_model_uses_the_documented_basis_epsilon(random_cyl):
    """The model's threshold is the documented default and is reachable from the factory."""
    assert random_cyl.basis_eps == DEFAULT_BASIS_EPS
    other = build_xrir_exp08("cylindrical_invariant", 2, basis_eps=1e-3)
    assert other.basis_eps == 1e-3


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
