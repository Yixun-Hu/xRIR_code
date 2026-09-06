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
