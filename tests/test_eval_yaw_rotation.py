"""Tests for :mod:`eval_yaw_rotation` (exp_03, yaw_rotation_degradation).

The evaluator makes one pass per model over the pinned reference manifest and, at every
yaw angle, evaluates two conditions: **P** (primary -- the direct-path alignment pinned
at the ``k = 0`` coordinates, so only the geometry branches see the rotation) and **E**
(end-to-end -- the alignment is recomputed from the rotated coordinates, what the
pipeline would do at deployment).  Both must collapse onto the plain forward at
``k = 0``; everything downstream of that is a paired comparison against it.

``model.xRIR.apply_delay`` hard-codes ``.cuda()``, so every forward test needs a GPU.
"""
import pytest
import torch

from eval_yaw_rotation import forward_conditions, rotated_views

_NEEDS_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

B, K, T, H, W = 1, 2, 9600, 256, 512


def _synthetic_scene(seed=0, device="cpu"):
    """A small but shape-faithful scene: 256x512 panorama, 9600-sample IRs."""
    torch.manual_seed(seed)
    return (torch.rand(B, 3, H, W, device=device) * 5.0,
            torch.randn(B, K, T, device=device) * 0.1,
            torch.randn(B, 3, device=device),
            torch.randn(B, K, 3, device=device),
            torch.randn(B, 1, T, device=device) * 0.1)


# --------------------------------------------------------------------------------------
# rotated_views
# --------------------------------------------------------------------------------------
def test_rotated_views_is_the_rotate_scene_yaw_of_the_panorama_width():
    from tools.yaw_rotation import rotate_scene_yaw

    depth, _, src, ref_locs, _ = _synthetic_scene(seed=1)
    for k in (0, 32, 511, -8):
        got = rotated_views(depth, src, ref_locs, k)
        want = rotate_scene_yaw(depth, src, ref_locs, k, W=W)
        assert len(got) == 3
        for a, b in zip(got, want):
            assert torch.equal(a, b), "k={}".format(k)
    # k = 0 is the identity, bit for bit -- the whole paired design rests on it.
    d0, s0, r0 = rotated_views(depth, src, ref_locs, 0)
    assert torch.equal(d0, depth) and torch.equal(s0, src) and torch.equal(r0, ref_locs)


# --------------------------------------------------------------------------------------
# forward_conditions
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cuda_model():
    from model.xRIR_cyl import build_xrir

    torch.manual_seed(4242)
    return build_xrir("simple", K).cuda().eval()


@_NEEDS_CUDA
def test_forward_conditions_reduce_to_the_plain_forward_at_k0(cuda_model):
    from model.xRIR import xRIR

    depth, refs, src, ref_locs, tgt = _synthetic_scene(seed=2, device="cuda")
    with torch.no_grad():
        out_plain, tgt_plain = cuda_model(depth, refs, src, ref_locs, tgt)
        aligned0 = cuda_model.shift_and_align(refs, src, ref_locs)

    out_P, out_E, tgt_spec = forward_conditions(
        cuda_model, depth, refs, src, ref_locs, tgt, 0, aligned0)
    assert torch.equal(out_P, out_plain), "P at k=0 is not the plain forward"
    assert torch.equal(out_E, out_plain), "E at k=0 is not the plain forward"
    assert torch.equal(tgt_spec, tgt_plain)
    assert out_P.grad_fn is None and out_E.grad_fn is None
    # The alignment override is undone.
    assert cuda_model.shift_and_align.__func__ is xRIR.shift_and_align
    assert "shift_and_align" not in vars(cuda_model)


@_NEEDS_CUDA
def test_forward_conditions_rotate_the_geometry_and_pin_only_P(cuda_model):
    depth, refs, src, ref_locs, tgt = _synthetic_scene(seed=3, device="cuda")
    with torch.no_grad():
        out_plain, _ = cuda_model(depth, refs, src, ref_locs, tgt)
        aligned0 = cuda_model.shift_and_align(refs, src, ref_locs)

    out_P, out_E, _ = forward_conditions(
        cuda_model, depth, refs, src, ref_locs, tgt, 32, aligned0)
    assert (out_P - out_plain).abs().max().item() > 1e-5, "k=32 P is indistinguishable from k=0"
    assert (out_E - out_plain).abs().max().item() > 1e-5, "k=32 E is indistinguishable from k=0"

    # Only P reads the pinned alignment: feeding it a different cache moves P, not E.
    out_P2, out_E2, _ = forward_conditions(
        cuda_model, depth, refs, src, ref_locs, tgt, 32, torch.zeros_like(aligned0))
    assert not torch.equal(out_P2, out_P), "P ignores the pinned alignment"
    assert torch.equal(out_E2, out_E), "E is affected by the pinned alignment"


# --------------------------------------------------------------------------------------
# spectral_metrics
# --------------------------------------------------------------------------------------
def _spec_pair(batch_size=3, freq=63, time=310, seed=0):
    """A seeded (log-magnitude prediction, magnitude target) pair in the model's layout."""
    gen = torch.Generator().manual_seed(seed)
    tgt_spec = torch.rand(batch_size, 1, freq, time, generator=gen) * 2.0 + 1e-3
    out_spec = torch.log(torch.rand(batch_size, freq, time, 1, generator=gen) * 2.0 + 1e-3)
    return out_spec, tgt_spec


def test_spectral_metrics_match_the_per_sample_and_exp01_definitions():
    import numpy as np

    from eval_unseen import Evaluator
    from eval_yaw_rotation import spectral_metrics
    from tools.per_sample_metrics import per_sample_losses

    out_k, tgt_spec = _spec_pair(seed=5)
    out_0, _ = _spec_pair(seed=6)
    got = spectral_metrics(out_k, out_0, tgt_spec)

    assert sorted(got) == ["consistency", "decay", "log_mse", "loss", "stft"]
    for name, value in got.items():
        assert value.shape == (3,), name
        assert value.dtype == torch.float32 and value.device.type == "cpu", name

    loss, stft, decay = per_sample_losses(out_k, tgt_spec)
    assert torch.equal(got["loss"], loss)
    assert torch.equal(got["stft"], stft)
    assert torch.equal(got["decay"], decay)

    # log_mse is the per-sample form of exp_01's Evaluator.stft_loss call.
    evaluator = Evaluator()
    for i in range(3):
        want = evaluator.stft_loss(
            out_k[i:i + 1].squeeze(-1).numpy(),
            torch.log(tgt_spec[i:i + 1] + 1e-8).squeeze(1).numpy())
        # float32 means over 63x310 cells: torch's and numpy's reduction orders differ at
        # the last ulp, so the comparison is relative rather than bit-exact.
        assert float(got["log_mse"][i]) == pytest.approx(float(want), rel=1e-6, abs=0)
        want_consistency = float((out_k[i] - out_0[i]).abs().mean())
        assert float(got["consistency"][i]) == pytest.approx(want_consistency, abs=1e-7, rel=0)
    assert np.all(np.asarray(got["consistency"]) > 0.0)


def test_spectral_consistency_is_exactly_zero_against_itself():
    out_k, tgt_spec = _spec_pair(seed=7)
    from eval_yaw_rotation import spectral_metrics

    got = spectral_metrics(out_k, out_k, tgt_spec)
    assert torch.equal(got["consistency"], torch.zeros(3))


# --------------------------------------------------------------------------------------
# acoustic_metrics_batch
# --------------------------------------------------------------------------------------
def _acoustic_batch(batch_size=3, seed=0):
    """Log-magnitude predictions plus decaying-noise ground-truth IRs (valid metrics)."""
    gen = torch.Generator().manual_seed(seed)
    out_k = torch.log(torch.rand(batch_size, 63, 310, 1, generator=gen) * 2.0 + 1e-3)
    decay = torch.exp(-torch.arange(9600, dtype=torch.float32) / 1500.0)
    tgt_wav = (torch.randn(batch_size, 1, 9600, generator=gen) * decay) * 0.1
    keys = ["Cafe/Cafe_idx_1/S00{}_R002_hybrid_IR.wav".format(i) for i in range(batch_size)]
    return out_k, tgt_wav, keys


def _same(a, b):
    import numpy as np

    return bool((np.isnan(a) and np.isnan(b)) or a == b)


def test_acoustic_metrics_batch_matches_the_per_sample_helper():
    import numpy as np

    from eval_unseen import Evaluator
    from eval_yaw_rotation import acoustic_metrics_batch
    from tools.per_sample_metrics import acoustic_metrics, griffin_lim_seeded, sample_seed

    out_k, tgt_wav, keys = _acoustic_batch(seed=9)
    evaluator = Evaluator()
    got = acoustic_metrics_batch(out_k, tgt_wav, keys, evaluator, 0)

    assert sorted(got) == ["c50", "edt", "t60"]
    for name, value in got.items():
        assert isinstance(value, np.ndarray) and value.dtype == np.float64, name
        assert value.shape == (3,), name
        assert np.isfinite(value).all(), "{} has invalid samples: {}".format(name, value)

    for i in range(3):
        mag = (torch.exp(out_k[i:i + 1]) - 1e-8)[..., 0]
        wav = griffin_lim_seeded(mag, sample_seed(0, keys[i]))
        want = acoustic_metrics(wav[0].numpy(), tgt_wav[i, 0].numpy(), evaluator)
        for name in ("edt", "c50", "t60"):
            assert _same(got[name][i], want[name]), "{}[{}]".format(name, i)


def test_acoustic_metrics_batch_is_seeded_per_query():
    import numpy as np

    from eval_unseen import Evaluator
    from eval_yaw_rotation import acoustic_metrics_batch

    out_k, tgt_wav, keys = _acoustic_batch(seed=10)
    evaluator = Evaluator()
    a = acoustic_metrics_batch(out_k, tgt_wav, keys, evaluator, 0)
    b = acoustic_metrics_batch(out_k, tgt_wav, keys, evaluator, 0)
    c = acoustic_metrics_batch(out_k, tgt_wav, keys, evaluator, 1)
    for name in ("edt", "c50", "t60"):
        assert np.array_equal(a[name], b[name]), name
    assert not np.array_equal(a["edt"], c["edt"]), "a different Griffin-Lim seed changed nothing"


# --------------------------------------------------------------------------------------
# delay_flip_counts -- the audit of shift_and_align's numerical (non-)invariance
# --------------------------------------------------------------------------------------
def _tie_coordinates(n_max=2000):
    """Source radii whose direct-path delay sits on (or beside) a rounding tie.

    ``round()`` at an exact ``.5`` goes half-to-even, so a rotation that perturbs the
    float32 norm by an ulp flips the integer delay of about half of these pairs -- the
    effect the audit counts.  Deterministic: no RNG is involved.
    """
    steps = torch.arange(1, n_max + 1, dtype=torch.float32)
    radius = (steps + 0.5) * (343.0 / 22050.0)
    src = torch.stack([radius, torch.zeros_like(radius), torch.zeros_like(radius)], dim=1)
    return src, torch.zeros(radius.shape[0], 1, 3)


def test_delay_flip_counts_match_a_manual_recomputation():
    from eval_yaw_rotation import delay_flip_counts
    from tools.yaw_rotation import integer_delays, rotate_vectors_z, yaw_angle_rad

    src, ref_locs = _tie_coordinates()
    cols = [0, 4, 32, 128]
    got = delay_flip_counts(src, ref_locs, cols)

    assert sorted(got) == sorted(cols)
    assert all(isinstance(v, int) for v in got.values())
    assert got[0] == 0, "the unrotated coordinates cannot flip against themselves"

    d0 = integer_delays(src, ref_locs)
    n_pairs = src.shape[0] * ref_locs.shape[1]
    for k in cols:
        angle = yaw_angle_rad(k)
        dk = integer_delays(rotate_vectors_z(src, angle), rotate_vectors_z(ref_locs, angle))
        assert got[k] == int((dk != d0).sum()), "k={}".format(k)
        assert 0 <= got[k] <= n_pairs
    # The audit must be able to see flips at all: these radii are built to produce them.
    assert got[4] > 0 and got[32] > 0, got


def test_delay_flip_counts_are_zero_when_the_geometry_is_untouched():
    from eval_yaw_rotation import delay_flip_counts

    torch.manual_seed(17)
    src, ref_locs = torch.randn(32, 3) * 3.0, torch.randn(32, 4, 3) * 3.0
    assert delay_flip_counts(src, ref_locs, [0])[0] == 0


# --------------------------------------------------------------------------------------
# decomposition_at -- where the cylindrical model's rotation sensitivity enters
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cuda_cyl_model():
    from model.xRIR_cyl import build_xrir

    torch.manual_seed(99)
    return build_xrir("cylindrical", K).cuda().eval()


@_NEEDS_CUDA
def test_decomposition_at_isolates_the_pooling_and_coordinate_branches(cuda_cyl_model):
    from eval_yaw_rotation import decomposition_at

    depth, refs, src, ref_locs, tgt = _synthetic_scene(seed=21, device="cuda")
    with torch.no_grad():
        aligned0 = cuda_cyl_model.shift_and_align(refs, src, ref_locs)

    got = decomposition_at(cuda_cyl_model, depth, refs, src, ref_locs, tgt, aligned0, k=32)
    assert got["k"] == 32
    names = ("tokens_rel_change", "pooled_rel_change", "coord_rel_change", "logspec_rel_change")
    assert sorted(got) == sorted(("k",) + names)
    for name in names:
        assert isinstance(got[name], float) and got[name] == got[name], name
        assert got[name] >= 0.0, name

    # A patch-aligned roll permutes the cylindrical tokens: rolling them back recovers
    # them. The learned pooling and the raw-xyz embedding are not shift-invariant, so
    # the change survives into the output.
    assert got["tokens_rel_change"] < 1e-3, got
    assert got["coord_rel_change"] > 0.0, got
    assert got["logspec_rel_change"] > 0.0, got


@_NEEDS_CUDA
def test_decomposition_at_rejects_a_non_cylindrical_backbone(cuda_model):
    from eval_yaw_rotation import decomposition_at

    depth, refs, src, ref_locs, tgt = _synthetic_scene(seed=22, device="cuda")
    with torch.no_grad():
        aligned0 = cuda_model.shift_and_align(refs, src, ref_locs)
    with pytest.raises(TypeError):
        decomposition_at(cuda_model, depth, refs, src, ref_locs, tgt, aligned0, k=32)


# --------------------------------------------------------------------------------------
# set_precision / T14 -- REAL DATA: batch-size invariance of every per-sample metric
# --------------------------------------------------------------------------------------
def test_set_precision_pins_both_tf32_switches():
    from eval_yaw_rotation import set_precision

    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        set_precision(True)
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        set_precision(False)
        assert not torch.backends.cuda.matmul.allow_tf32
        # cuDNN defaults to True: leaving it on makes the ResNet-18 audio encoder
        # batch-size dependent at ~1e-4, which would swamp the rotation effect.
        assert not torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before


import os  # noqa: E402  (kept beside the real-data helpers it serves)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINNED_MANIFEST = os.path.join(REPO_ROOT, "ckpt", "yaw_rotation", "reference_manifest.json")
PINNED_HASH = "47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d"


def _pinned_manifest():
    """The pinned exp_03 manifest, or skip (it is a run artifact, not a repo file)."""
    from treble_multi_room_dataset.treble_xRIR_dataset import BASE_DATA_PATH
    from tools.reference_manifest import load_manifest, manifest_hash

    if not os.path.isdir(os.path.join(BASE_DATA_PATH, "single_channel_ir")):
        pytest.skip("AcousticRooms not available at XRIR_DATA_PATH={}".format(BASE_DATA_PATH))
    if not os.path.exists(PINNED_MANIFEST):
        pytest.skip("pinned manifest not found at {}".format(PINNED_MANIFEST))
    manifest = load_manifest(PINNED_MANIFEST)
    assert manifest_hash(manifest) == PINNED_HASH, "the pinned manifest changed"
    return manifest


@pytest.fixture(scope="module")
def cuda_simple8():
    """A seeded random-init baseline xRIR with the manifest's 8 shots, on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from model.xRIR_cyl import build_xrir

    torch.manual_seed(1234)
    return build_xrir("simple", 8).cuda().eval()


@pytest.fixture(scope="module")
def real_batch_of_4():
    """One collated batch of the first 4 manifest queries."""
    from torch.utils.data import DataLoader

    from eval_yaw_rotation import build_manifest_dataset, set_precision

    set_precision(False)
    dataset = build_manifest_dataset(_pinned_manifest(), max_samples=4)
    return next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)))


def _evaluate(loader, model, evaluator, cols, gl_seed=0):
    """``run``'s inner loop over a whole loader, one ``evaluate_batch`` call per batch."""
    import numpy as np

    from eval_yaw_rotation import evaluate_batch

    keys_seen, parts = [], {}
    for batch in loader:
        keys, results, _ = evaluate_batch(model, batch, evaluator, cols, cols, cols, gl_seed)
        keys_seen.extend(keys)
        for cell, metrics in results.items():
            for metric, value in metrics.items():
                parts.setdefault(cell, {}).setdefault(metric, []).append(value)
    return keys_seen, {cell: {m: np.concatenate(v) for m, v in metrics.items()}
                       for cell, metrics in parts.items()}


@_NEEDS_CUDA
def test_per_sample_metrics_are_batch_size_invariant(capsys):
    import numpy as np
    from torch.utils.data import DataLoader

    from eval_unseen import Evaluator
    from eval_yaw_rotation import build_manifest_dataset, set_precision
    from model.xRIR_cyl import build_xrir

    manifest = _pinned_manifest()
    set_precision(False)                       # exactly as ``run`` does; see T14 note below
    dataset = build_manifest_dataset(manifest, max_samples=8)
    assert len(dataset) == 8
    assert [e["query"] for e in dataset.entries] == \
        [e["query"] for e in manifest["entries"][:8]]

    torch.manual_seed(1234)
    model = build_xrir("simple", manifest["num_shot"]).cuda().eval()
    evaluator = Evaluator()
    cols = [0, 32]

    runs = {}
    for batch_size in (8, 1):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        runs[batch_size] = _evaluate(loader, model, evaluator, cols)

    keys8, big = runs[8]
    keys1, small = runs[1]
    assert keys8 == keys1 == [e["query"] for e in manifest["entries"][:8]]

    worst, scale = {}, {}
    for cell in big:
        for metric, value in big[cell].items():
            other = small[cell][metric]
            assert value.shape == other.shape == (8,), (cell, metric)
            diff = float(np.abs(value - other).max())
            worst[metric] = max(worst.get(metric, 0.0), diff)
            scale[metric] = max(scale.get(metric, 0.0), float(np.abs(value).max()))
            if metric in ("edt", "c50", "t60"):
                # Griffin-Lim phases are seeded per query, so these differ only through
                # the (matmul-order) difference in the log-spectrogram itself.
                assert np.allclose(value, other, rtol=1e-6, atol=1e-6), (cell, metric, diff)
            else:
                # atol 1e-6 plus one float32 ulp of relative slack: with an untrained
                # model log_mse is ~50, where a single ulp is already 3.8e-6, so a pure
                # absolute tolerance would test the value's magnitude, not the pipeline.
                assert np.allclose(value, other, atol=1e-6, rtol=1e-7), (cell, metric, diff)
    with capsys.disabled():
        print("\n[T14] batch 1 vs batch 8, max |difference| per metric over 8 real queries "
              "x {P,E} x k in {0,32}:")
        for metric in sorted(worst):
            print("        {:12s} {:.3e}  (relative {:.2e})".format(
                metric, worst[metric], worst[metric] / max(scale[metric], 1e-12)))


# --------------------------------------------------------------------------------------
# T15 -- REAL DATA: the angle order cannot change a per-sample value
# --------------------------------------------------------------------------------------
@_NEEDS_CUDA
def test_angle_order_invariance(real_batch_of_4, cuda_simple8):
    import numpy as np

    from eval_unseen import Evaluator
    from eval_yaw_rotation import evaluate_batch

    evaluator = Evaluator()
    forward_order = evaluate_batch(cuda_simple8, real_batch_of_4, evaluator,
                                   [0, 32], [0, 32], [32], 0)
    reverse_order = evaluate_batch(cuda_simple8, real_batch_of_4, evaluator,
                                   [32, 0], [0, 32], [32], 0)

    keys_a, res_a, flips_a = forward_order
    keys_b, res_b, flips_b = reverse_order
    assert keys_a == keys_b
    assert flips_a == flips_b == {0: 0, 32: flips_a[32]}
    assert sorted(res_a) == sorted(res_b)
    for cell in res_a:
        assert sorted(res_a[cell]) == sorted(res_b[cell]), cell
        for metric, value in res_a[cell].items():
            assert np.array_equal(value, res_b[cell][metric]), (cell, metric)

    # k = 0 is the shared paired reference: both conditions collapse onto it, acoustic
    # metrics included, and the self-consistency term is exactly zero there.
    assert sorted(res_a[("P", 0)]) == sorted(res_a[("E", 0)])
    for metric, value in res_a[("P", 0)].items():
        assert np.array_equal(value, res_a[("E", 0)][metric]), metric
    assert np.array_equal(res_a[("P", 0)]["consistency"], np.zeros(4))
    assert (res_a[("P", 32)]["consistency"] > 0).all()


# --------------------------------------------------------------------------------------
# Angle-grid validation and the per-angle summary
# --------------------------------------------------------------------------------------
def test_check_cols_rejects_a_grid_the_paired_design_cannot_use():
    from eval_yaw_rotation import _check_cols

    assert _check_cols([0, 32, 480], [0, 32], [32]) == ([0, 32, 480], [0, 32], [32])
    assert _check_cols(["0", "32"], ["32"], []) == ([0, 32], [32], [])

    with pytest.raises(ValueError):
        _check_cols([32, 64], [32], [])                 # no k = 0: nothing to pair against
    with pytest.raises(ValueError):
        _check_cols([0, 32, 32], [0], [])               # a repeated angle
    with pytest.raises(ValueError):
        _check_cols([0, 32], [0, 64], [])               # acoustic angle outside the grid
    with pytest.raises(ValueError):
        _check_cols([0, 32], [0], [64])                 # end-to-end angle outside the grid


def test_summarize_averages_over_the_finite_values_only():
    from eval_yaw_rotation import _summarize

    assert _summarize([1.0, 3.0, float("nan")]) == {"mean": 2.0, "n_valid": 2, "n_nan": 1}
    assert _summarize([float("nan")]) == {"mean": None, "n_valid": 0, "n_nan": 1}
    # An infinite metric is invalid, not a usable extreme.
    assert _summarize([float("inf"), 2.0]) == {"mean": 2.0, "n_valid": 1, "n_nan": 1}
