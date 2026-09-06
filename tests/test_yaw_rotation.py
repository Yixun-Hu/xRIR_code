"""Tests for :mod:`tools.yaw_rotation` (exp_03, yaw_rotation_degradation).

Convention under test: an ACTIVE yaw of the whole scene by +Delta about the
receiver's vertical (z) axis, right-handed (+x -> +y looking down the -z axis),
with Delta = 2*pi*k/W for an integer roll of ``k`` panorama columns.
"""
import math
import types

import numpy as np
import pytest
import torch

from tools.yaw_rotation import rotate_scene_yaw, rotate_vectors_z, yaw_angle_rad


# --------------------------------------------------------------------------------------
# T1 -- yaw_angle_rad
# --------------------------------------------------------------------------------------
def test_yaw_angle_rad_values():
    assert yaw_angle_rad(0) == pytest.approx(0.0)
    assert yaw_angle_rad(512) == pytest.approx(2.0 * math.pi)
    assert yaw_angle_rad(128) == pytest.approx(math.pi / 2)
    # Explicit W is honoured.
    assert yaw_angle_rad(1, W=4) == pytest.approx(math.pi / 2)


# --------------------------------------------------------------------------------------
# T2 -- rotate_vectors_z
# --------------------------------------------------------------------------------------
def test_rotate_vectors_z_preserves_norm_and_z():
    torch.manual_seed(0)
    v = torch.randn(7, 5, 3)
    out = rotate_vectors_z(v, 0.7)
    assert out.shape == v.shape
    assert torch.allclose(torch.linalg.norm(out, dim=-1), torch.linalg.norm(v, dim=-1), atol=1e-6)
    assert torch.allclose(out[..., 2], v[..., 2], atol=1e-6)


def test_rotate_vectors_z_active_right_handed():
    axes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = rotate_vectors_z(axes, math.pi / 2)
    assert torch.allclose(out[0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
    assert torch.allclose(out[1], torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6)


# --------------------------------------------------------------------------------------
# T3 -- rotate_scene_yaw
# --------------------------------------------------------------------------------------
def _random_scene(seed=0, B=2, K=4, H=32, W=512):
    torch.manual_seed(seed)
    return (torch.randn(B, 3, H, W), torch.randn(B, 3), torch.randn(B, K, 3))


def test_rotate_scene_yaw_identity_and_period():
    depth, src, refs = _random_scene()
    d0, s0, r0 = rotate_scene_yaw(depth, src, refs, 0)
    assert torch.equal(d0, depth) and torch.equal(s0, src) and torch.equal(r0, refs)

    dW, sW, rW = rotate_scene_yaw(depth, src, refs, 512)
    assert torch.allclose(dW, d0, atol=1e-6)
    assert torch.allclose(sW, s0, atol=1e-6)
    assert torch.allclose(rW, r0, atol=1e-6)


def test_rotate_scene_yaw_negative_k_wraps():
    depth, src, refs = _random_scene(seed=1)
    neg = rotate_scene_yaw(depth, src, refs, -16)
    pos = rotate_scene_yaw(depth, src, refs, 512 - 16)
    for a, b in zip(neg, pos):
        assert torch.allclose(a, b, atol=1e-6)


def test_rotate_scene_yaw_composition():
    depth, src, refs = _random_scene(seed=2)
    k1, k2 = 40, 500
    mid = rotate_scene_yaw(depth, src, refs, k1)
    two_step = rotate_scene_yaw(mid[0], mid[1], mid[2], k2)
    one_step = rotate_scene_yaw(depth, src, refs, (k1 + k2) % 512)
    for a, b in zip(two_step, one_step):
        assert torch.allclose(a, b, atol=1e-5)


def test_rotate_scene_yaw_does_not_mutate_inputs():
    depth, src, refs = _random_scene(seed=3)
    depth_c, src_c, refs_c = depth.clone(), src.clone(), refs.clone()
    rotate_scene_yaw(depth, src, refs, 37)
    assert torch.equal(depth, depth_c)
    assert torch.equal(src, src_c)
    assert torch.equal(refs, refs_c)


# --------------------------------------------------------------------------------------
# T4 -- oracle: Rz(Delta) . roll(convert(D), k) == convert(roll(D, k))
# --------------------------------------------------------------------------------------
def _rz_roll(t: torch.Tensor, k: int, W: int = 512) -> torch.Tensor:
    """Independent reference implementation of "roll by +k columns, then Rz(+2*pi*k/W)".

    Written out with explicit cos/sin so it does not share code with
    :func:`tools.yaw_rotation.rotate_scene_yaw`.  ``t`` is ``[B, 3, H, W]``.
    """
    k = int(k) % int(W)
    a = 2.0 * math.pi * k / W
    cos_a, sin_a = math.cos(a), math.sin(a)
    r = torch.roll(t, shifts=k, dims=-1)
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    return torch.stack([x * cos_a - y * sin_a, x * sin_a + y * cos_a, z], dim=1)


def test_rotate_scene_yaw_matches_reprojected_rolled_depth():
    from treble_multi_room_dataset.treble_xRIR_dataset import convert_equirect_to_camera_coord

    H, W = 256, 512
    rng = np.random.RandomState(0)
    depth_map = (rng.rand(H, W).astype(np.float32) * 4.0 + 1.0)          # positive depths, 1..5 m

    def convert(dm):
        return convert_equirect_to_camera_coord(
            torch.from_numpy(dm), H, W).permute(2, 0, 1).float().unsqueeze(0)

    depth_coord = convert(depth_map)                                      # [1, 3, H, W]
    zeros_src = torch.zeros(1, 3)
    zeros_ref = torch.zeros(1, 1, 3)

    # Negative k is compared against np.roll with a negative shift -- an independent
    # oracle, not another call of rotate_scene_yaw.
    for k in (0, 1, 32, 100, 511, -16, -100):
        got = rotate_scene_yaw(depth_coord, zeros_src, zeros_ref, k, W=W)[0]
        want = convert(np.roll(depth_map, k, axis=1))
        assert got.shape == want.shape
        # All three channels, every column.
        assert torch.allclose(got, want, atol=1e-5, rtol=0), "k={} max|d|={}".format(
            k, (got - want).abs().max().item())
        # Seam columns checked explicitly (wrap-around is where a convention error shows).
        for col in (0, W - 1):
            assert torch.allclose(got[..., col], want[..., col], atol=1e-5, rtol=0), \
                "k={} col={}".format(k, col)


# --------------------------------------------------------------------------------------
# T5 -- covariance of the three ViT views used by xRIR.forward
# --------------------------------------------------------------------------------------
def test_rotate_scene_yaw_covaries_all_three_vit_views():
    B, K, H, W, k = 2, 4, 256, 512, 64
    torch.manual_seed(7)
    depth = torch.randn(B, 3, H, W)
    src = torch.randn(B, 3)
    refs = torch.randn(B, K, 3)

    depth_r, src_r, refs_r = rotate_scene_yaw(depth, src, refs, k, W=W)

    src_view = src[:, :, None, None] - depth
    src_view_r = src_r[:, :, None, None] - depth_r
    assert torch.allclose(src_view_r, _rz_roll(src_view, k, W), atol=1e-5)

    rec_view = -depth
    rec_view_r = -depth_r
    assert torch.allclose(rec_view_r, _rz_roll(rec_view, k, W), atol=1e-5)

    for i in range(K):
        ref_view = refs[:, i, :, None, None] - depth
        ref_view_r = refs_r[:, i, :, None, None] - depth_r
        assert torch.allclose(ref_view_r, _rz_roll(ref_view, k, W), atol=1e-5), "ref {}".format(i)


# --------------------------------------------------------------------------------------
# T6 -- integer_delays reproduces xRIR.shift_and_align; fixed_alignment swaps it safely
# --------------------------------------------------------------------------------------
_NEEDS_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(scope="module")
def simple_model():
    """A CPU-resident baseline xRIR; only its (parameter-free) methods are exercised."""
    from model.xRIR_cyl import build_xrir
    return build_xrir("simple", 4).eval()


def _apply_delay_reference(signal, delays):
    """Stand-alone copy of ``model.xRIR.apply_delay`` semantics (zero-padded integer shift)."""
    out = torch.zeros_like(signal)
    for i in range(signal.shape[0]):
        d = int(delays[i].item())
        if d > 0:
            out[i, d:] = signal[i, :-d]
        elif d < 0:
            out[i, :d] = signal[i, -d:]
        else:
            out[i] = signal[i]
    return out


def _shift_and_align_reference(refs, src, ref_locs, delays):
    """Independent re-implementation of ``xRIR.shift_and_align``: zero-padded integer
    shift by ``delays`` followed by the direct-path gain ``dist_ref / (dist_src + 1e-7)``."""
    dist_src = torch.linalg.norm(src, dim=1).unsqueeze(1)
    dist_ref = torch.linalg.norm(ref_locs, dim=-1)
    ratio = dist_ref / (dist_src + 1e-7)
    return torch.cat(
        [(_apply_delay_reference(refs[:, i, :], delays[:, i]).unsqueeze(1)
          * ratio[:, i:(i + 1)].unsqueeze(2)) for i in range(refs.shape[1])], dim=1)


@_NEEDS_CUDA
def test_integer_delays_reproduces_shift_and_align(simple_model):
    from tools.yaw_rotation import integer_delays

    B, K, T = 3, 4, 2048
    torch.manual_seed(11)
    refs = (torch.randn(B, K, T) * 0.1).cuda()
    src = (torch.randn(B, 3) * 2.0).cuda()
    ref_locs = (torch.randn(B, K, 3) * 2.0).cuda()

    delays = integer_delays(src, ref_locs)
    assert delays.shape == (B, K)
    assert int(delays.abs().max()) > 0, "degenerate draw: all delays are zero"
    assert int(delays.abs().max()) < T, "delay exceeds the signal length"

    # The model's own expression, re-derived here: same value AND same dtype (int32).
    dist_src = torch.linalg.norm(src, dim=1).unsqueeze(1)
    dist_ref = torch.linalg.norm(ref_locs, dim=-1)
    delay_unit = torch.round((dist_src - dist_ref) / 343. * 22050).int()
    assert delays.dtype == torch.int32
    assert delays.dtype == delay_unit.dtype
    assert torch.equal(delays, delay_unit)
    expected = _shift_and_align_reference(refs, src, ref_locs, delays)

    got = simple_model.shift_and_align(refs, src, ref_locs)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-6)


def test_integer_delays_uses_round_half_to_even_at_exact_ties():
    """Exact float32 .5 ties must round half-to-even, exactly as the model does."""
    from tools.yaw_rotation import integer_delays

    # float32 values whose (d / 343.) * 22050 evaluates to exactly +-0.5 / +-1.5 in float32
    # (found by an ulp search; the assertion below fails loudly if that ever stops holding).
    s_half, s_three_half = 0.007777777500450611, 0.023333333432674408
    zero = [0.0, 0.0, 0.0]
    src = torch.tensor([[s_half, 0.0, 0.0], zero, [s_three_half, 0.0, 0.0], zero],
                       dtype=torch.float32)
    ref_locs = torch.tensor([[zero], [[s_half, 0.0, 0.0]], [zero], [[s_three_half, 0.0, 0.0]]],
                            dtype=torch.float32)

    dist_src = torch.linalg.norm(src, dim=1).unsqueeze(1)
    dist_ref = torch.linalg.norm(ref_locs, dim=-1)
    raw = (dist_src - dist_ref) / 343. * 22050
    assert raw.flatten().tolist() == [0.5, -0.5, 1.5, -1.5], "tie construction broke: {}".format(
        raw.flatten().tolist())

    delay_unit = torch.round(raw).int()                      # the model's own expression
    got = integer_delays(src, ref_locs)
    assert got.dtype == torch.int32 == delay_unit.dtype
    assert torch.equal(got, delay_unit)
    # Half-to-even, not half-away-from-zero: 0.5 -> 0, -0.5 -> 0, 1.5 -> 2, -1.5 -> -2.
    assert got.flatten().tolist() == [0, 0, 2, -2]


def test_fixed_alignment_swaps_and_restores(simple_model):
    from model.xRIR import xRIR
    from tools.yaw_rotation import fixed_alignment

    cached = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    assert "shift_and_align" not in vars(simple_model)

    with fixed_alignment(simple_model, cached):
        # Inside the context the replacement is still a bound method of the model.
        assert isinstance(simple_model.shift_and_align, types.MethodType)
        assert simple_model.shift_and_align.__self__ is simple_model
        out = simple_model.shift_and_align(torch.zeros(2, 3, 5), torch.zeros(2, 3), torch.zeros(2, 3, 3))
        assert out is cached
    assert simple_model.shift_and_align.__func__ is xRIR.shift_and_align
    assert "shift_and_align" not in vars(simple_model)

    with pytest.raises(RuntimeError):
        with fixed_alignment(simple_model, cached):
            assert simple_model.shift_and_align(None, None, None) is cached
            raise RuntimeError("boom")
    assert simple_model.shift_and_align.__func__ is xRIR.shift_and_align
    assert "shift_and_align" not in vars(simple_model)


def _load_real_query_coordinates(n):
    """First ``n`` (src, refs) coordinate rows of the AcousticRooms test split, or skip."""
    import os

    from treble_multi_room_dataset.treble_xRIR_dataset import BASE_DATA_PATH
    if not os.path.isdir(os.path.join(BASE_DATA_PATH, "single_channel_ir")):
        pytest.skip("AcousticRooms not available at XRIR_DATA_PATH={}".format(BASE_DATA_PATH))

    from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset

    np.random.seed(0)
    dataset = xRIR_Dataset(split="test", num_shot=8)
    n = min(n, len(dataset))
    assert n > 0
    src_rows, ref_rows = [], []
    for i in range(n):
        _, proj_source_pos, _, _, _, all_ref_src_pos = dataset[i]
        src_rows.append(proj_source_pos)
        ref_rows.append(all_ref_src_pos)
    return torch.stack(src_rows).float(), torch.stack(ref_rows).float()


@_NEEDS_CUDA
def test_integer_delays_matches_shift_and_align_on_real_coordinates(simple_model, capsys):
    """Parity against the model on REAL AcousticRooms geometry, plus the flip diagnostic."""
    from tools.yaw_rotation import integer_delays

    N, K, T = 8, 8, 2048
    src_cpu, ref_locs_cpu = _load_real_query_coordinates(N)
    N = src_cpu.shape[0]
    src, ref_locs = src_cpu.cuda(), ref_locs_cpu.cuda()
    torch.manual_seed(101)
    refs = (torch.randn(N, K, T) * 0.1).cuda()

    delays = integer_delays(src, ref_locs)
    # Re-derived with the model's own ops: same values and same dtype.
    dist_src = torch.linalg.norm(src, dim=1).unsqueeze(1)
    dist_ref = torch.linalg.norm(ref_locs, dim=-1)
    delay_unit = torch.round((dist_src - dist_ref) / 343. * 22050).int()
    assert delays.dtype == torch.int32 == delay_unit.dtype
    assert torch.equal(delays, delay_unit)
    assert int(delays.abs().max()) < T, "real delays exceed the test signal length"

    expected = _shift_and_align_reference(refs, src, ref_locs, delays)
    got = simple_model.shift_and_align(refs, src, ref_locs)
    assert got.shape == expected.shape == (N, K, T)
    assert torch.allclose(got, expected, atol=1e-6)

    # Exploratory diagnostic only (no assertion): how many integer delays move at k=4.
    angle = yaw_angle_rad(4)
    delays_rot = integer_delays(rotate_vectors_z(src, angle), rotate_vectors_z(ref_locs, angle))
    with capsys.disabled():
        print("\n[exploratory] delay flips at k=4 over {} query x {} reference pairs: {} / {}".format(
            N, K, int((delays != delays_rot).sum()), N * K))


# --------------------------------------------------------------------------------------
# T7 -- CylindricalViT tokens roll by one azimuth patch at k = 32; SimpleViT does not
# --------------------------------------------------------------------------------------
def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_cylindrical_vit_tokens_roll_by_one_azimuth_patch():
    from model.cylindrical_vit import CylindricalViT, RelPosAttention
    from model.simple_vit import SimpleViT

    device = _device()
    H, W, k = 256, 512, 32                       # k = one patch width -> a 1-patch azimuth shift
    torch.manual_seed(23)
    x = torch.randn(1, 3, H, W, device=device)
    x2 = rotate_scene_yaw(x, torch.zeros(1, 3, device=device),
                          torch.zeros(1, 1, 3, device=device), k, W=W)[0]

    cyl = CylindricalViT().to(device).eval()
    # The relative-position bias initialises to zeros; randomise it so the circular bias
    # table actually contributes to the attention logits under test.
    for mod in cyl.modules():
        if isinstance(mod, RelPosAttention):
            torch.nn.init.normal_(mod.rel_bias, std=0.5)

    with torch.no_grad():
        t1 = cyl(x).view(1, 16, 16, 512)          # [B, elevation, azimuth, dim] (h-major)
        t2 = cyl(x2).view(1, 16, 16, 512)
    diff = (torch.roll(t1, 1, dims=2) - t2).abs().max().item()
    assert diff < 1e-4, "CylindricalViT azimuth equivariance broken: max|d| = {}".format(diff)

    simple = SimpleViT((H, W), (16, 32), 512, 12, 8, 512).to(device).eval()
    with torch.no_grad():
        s1 = simple(x).view(1, 16, 16, 512)
        s2 = simple(x2).view(1, 16, 16, 512)
    s_diff = (torch.roll(s1, 1, dims=2) - s2).abs().max().item()
    assert s_diff > 0.1, "SimpleViT unexpectedly azimuth-equivariant: max|d| = {}".format(s_diff)


# --------------------------------------------------------------------------------------
# T8 -- full xRIR forward: k = W reproduces k = 0 under a pinned alignment, k = 32 does not
# --------------------------------------------------------------------------------------
@_NEEDS_CUDA
def test_full_model_period_and_sensitivity_under_fixed_alignment():
    from model.xRIR import xRIR
    from model.xRIR_cyl import build_xrir
    from tools.yaw_rotation import fixed_alignment

    B, K, T, H, W = 1, 8, 9600, 256, 512
    torch.manual_seed(31)
    model = build_xrir("cylindrical", K).cuda().eval()
    depth = (torch.rand(B, 3, H, W) * 5.0).cuda()
    refs = (torch.randn(B, K, T) * 0.1).cuda()
    src = torch.randn(B, 3).cuda()
    ref_locs = torch.randn(B, K, 3).cuda()
    tgt = (torch.randn(B, 1, T) * 0.1).cuda()

    with torch.no_grad():
        # Isolation control: pinning the k=0 alignment must not perturb the k=0 forward.
        out_plain, tgt_plain = model(depth, refs, src, ref_locs, tgt)
        aligned = model.shift_and_align(refs, src, ref_locs)

        outs = {}
        for k in (0, 512, 32):
            depth_k, src_k, ref_locs_k = rotate_scene_yaw(depth, src, ref_locs, k, W=W)
            with fixed_alignment(model, aligned):
                out_k, tgt_k = model(depth_k, refs, src_k, ref_locs_k, tgt)
            outs[k] = out_k
            if k == 0:
                tgt_fixed = tgt_k
        assert model.shift_and_align.__func__ is xRIR.shift_and_align

    assert torch.isfinite(out_plain).all()
    assert torch.equal(out_plain, outs[0]), "fixed_alignment perturbs the k=0 forward"
    assert torch.equal(tgt_plain, tgt_fixed)
    # Inputs are identical after the mod reduction, so k=W must reproduce k=0 exactly.
    assert torch.equal(outs[512], outs[0]), "k=W max|d| = {}".format(
        (outs[512] - outs[0]).abs().max().item())
    assert (outs[32] - outs[0]).abs().max().item() > 1e-4, "k=32 output is indistinguishable from k=0"

    # The alignment override is undone even when the block raises.
    with pytest.raises(RuntimeError):
        with fixed_alignment(model, aligned):
            raise RuntimeError("boom")
    assert model.shift_and_align.__func__ is xRIR.shift_and_align
    assert "shift_and_align" not in vars(model)


def test_fixed_alignment_nests(simple_model):
    """Nested use restores the outer override, then the class method (the had_own branch)."""
    from model.xRIR import xRIR
    from tools.yaw_rotation import fixed_alignment

    outer = torch.zeros(1, 1, 2)
    inner = torch.ones(1, 1, 2)
    with fixed_alignment(simple_model, outer):
        assert simple_model.shift_and_align(None, None, None) is outer
        with fixed_alignment(simple_model, inner):
            assert simple_model.shift_and_align(None, None, None) is inner
        assert simple_model.shift_and_align(None, None, None) is outer
    assert simple_model.shift_and_align.__func__ is xRIR.shift_and_align
    assert "shift_and_align" not in vars(simple_model)


# --------------------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------------------
def test_rotation_helpers_reject_integer_tensors():
    from tools.yaw_rotation import rotate_scene_yaw as _rsy

    with pytest.raises(TypeError):
        rotate_vectors_z(torch.zeros(4, 3, dtype=torch.int64), 0.3)
    with pytest.raises(TypeError):
        _rsy(torch.zeros(1, 3, 4, 512, dtype=torch.int32), torch.zeros(1, 3), torch.zeros(1, 1, 3), 8)
    with pytest.raises(TypeError):
        _rsy(torch.zeros(1, 3, 4, 512), torch.zeros(1, 3, dtype=torch.int64), torch.zeros(1, 1, 3), 8)
    with pytest.raises(TypeError):
        _rsy(torch.zeros(1, 3, 4, 512), torch.zeros(1, 3), torch.zeros(1, 1, 3, dtype=torch.int64), 8)


def test_invalid_k_and_W_are_rejected():
    depth = torch.zeros(1, 3, 4, 512)
    src, refs = torch.zeros(1, 3), torch.zeros(1, 1, 3)

    for bad_k in (1.5, 0.0, "8", None):
        with pytest.raises(TypeError):
            yaw_angle_rad(bad_k)
        with pytest.raises(TypeError):
            rotate_scene_yaw(depth, src, refs, bad_k)
    for bad_W in (0, -512):
        with pytest.raises(ValueError):
            yaw_angle_rad(8, W=bad_W)
        with pytest.raises(ValueError):
            rotate_scene_yaw(depth, src, refs, 8, W=bad_W)
    with pytest.raises(TypeError):
        yaw_angle_rad(8, W=512.0)
