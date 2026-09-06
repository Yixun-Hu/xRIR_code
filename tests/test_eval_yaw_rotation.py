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
