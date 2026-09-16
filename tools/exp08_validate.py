"""Structural validation of exp_08's invariant models -- CPU only, no training.

Implements the handoff's section 4 checklist ("validate the structure before spending
training time") as reusable functions, so ``tests/test_exp08_invariant.py`` asserts on the
same numbers this script reports.  Nothing here touches a GPU, writes a checkpoint, or
imports from outside this worktree; the pinned checkpoints and the exp_03 reference
manifest are read through absolute paths in the main checkout, read-only.

What it measures
----------------
* **Spectral invariance** over the 15 non-identity C16 angles (``k = 32 .. 480`` step 32)
  in two conditions:

  - ``E`` -- the true end-to-end path: the scene is rotated and the *whole* forward runs,
    so ``shift_and_align`` recomputes distances, gains and integer delays from the rotated
    coordinates.  This is the condition the guarantee is claimed in.
  - ``P`` -- the ``k = 0`` alignment pinned via ``tools.yaw_rotation.fixed_alignment``, kept
    only to separate an alignment-rounding effect from a geometry/conditioning effect.

  Reported per angle: relative Frobenius residual, absolute Frobenius residual, the
  denominator ``||out_0||_F`` and the max elementwise ``|d|`` -- the handoff requires the
  absolute error and the scale next to the relative number.
* **Waveform invariance** -- seeded Griffin-Lim on the predicted magnitudes at every angle
  with a yaw-independent per-query seed, then the relative L2 residual against ``k = 0``.
  A spectral-only claim is explicitly not sufficient (handoff sections 3.4 and 6).
* **Off-lattice diagnostic** -- a few non-multiples of 32, reported separately and never
  judged against the C16 criterion.
* **Delay flips** -- how many integer direct-path delays a rotation moves, the known float32
  rounding boundary in ``shift_and_align``.
* **A non-vacuity control** -- the same sweep on the *pinned* ``cylindrical`` model, which
  must fail the criterion by orders of magnitude; otherwise the harness proves nothing.

Usage::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$(pwd) python -m tools.exp08_validate \
        --out-json worklog/.../exp08_validation.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import time

import numpy as np
import torch

from eval_yaw_rotation import build_manifest_dataset, delay_flip_counts
from model.exp08_delay import device_preserving_delay
from model.exp08_factory import build_xrir_exp08
from model.xRIR_cyl_invariant import xRIR_InvariantBase
from tools.per_sample_metrics import griffin_lim_seeded, sample_seed
from tools.reference_manifest import load_manifest, manifest_hash
from model.xRIR_cyl_invariant import (BLEND_HI, BLEND_LO, horizontal_basis,
                                      intrinsic_scene_coords, to_intrinsic)
from tools.yaw_rotation import (fixed_alignment, rotate_scene_yaw, rotate_vectors_z,
                                yaw_angle_rad)
from utils.spec_utils import compute_spect_energy_decay_losses, stft_l1_loss

#: The 15 non-identity C16 column rolls of a 512-column panorama (22.5 deg steps).
C16_ANGLES = tuple(range(32, 512, 32))

#: Off-lattice rolls: diagnostic only, never judged against the C16 criterion.
OFF_LATTICE_ANGLES = (1, 5, 8, 16, 17, 31)

#: Read-only inputs in the main checkout.
MANIFEST_PATH = "/home/yixunhu/codespace/xRIR_code/ckpt/yaw_rotation/reference_manifest.json"
MANIFEST_HASH = "47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d"

#: The handoff's initial numerical target on the relative residual (section 4, last block).
REL_TARGET = 1.0e-5

#: exp_08 review round 1, blocker B1: a query on the vertical axis whose three references have
#: *exactly* equal horizontal radius (5, 5, 5).  Under the old max-radius selection rule float32
#: rounding decided the winner and the basis jumped between angles.
CODEX_TIE_SRC = ((0.0, 0.0, 1.0),)
CODEX_TIE_REFS = (((3.0, 4.0, 0.0), (5.0, 0.0, 0.0), (-3.0, 4.0, 0.0)),)

#: exp_08 review round 1, blocker B1, second case: a reference set whose horizontal components
#: cancel, so the aggregate itself vanishes and the basis must degenerate smoothly to zero.
CODEX_SUM_DEGENERATE_REFS = (((5.0, 0.0, 0.3), (-5.0, 0.0, -0.2), (0.0, 2.0, 0.1),
                              (0.0, -2.0, 0.4)),)

#: exp_08 review round 2, blocker B1(a): the horizontal components sum to *exactly* zero at a
#: radius of 20-36 m.  Under the round-2 ``s / ||s||`` rule with an absolute ``1e-6`` m guard,
#: rotation rounding pushed ``||s||`` across the guard, the branch flipped, and the intrinsic
#: coordinates jumped 35.7 m (spectrum 2.27e-1 at 22.5 deg, in both conditions).
CODEX_CANCEL_REFS = (((12.0, 16.0, 0.0), (20.0, 0.0, 0.0), (-32.0, -16.0, 0.0)),)

#: exp_08 review round 2, blocker B1(b): a near-cancellation that stayed on one branch, where
#: normalising by ``||s||`` alone amplified the rounding -- 1.5 m of coordinate motion, spectrum
#: 5.15e-2 at 45 deg.
CODEX_NEARCANCEL_REFS = (((3.0, 4.0, 0.0), (5.0, 0.0, 0.0),
                          (-7.999998092651367, -4.0, 0.0)),)

#: exp_08 review round 1, blocker B2: a query/reference pair whose pre-round direct-path delay
#: sits on the ``3.5`` rounding tie, so a norm-preserving rotation can move the integer delay.
CODEX_DELAY_SRC = ((1.054444432258606, 0.0, 0.0),)
CODEX_DELAY_REFS = (((1.0, 0.0, 0.0), (0.7, 1.1, 0.2), (-1.5, 0.3, -0.1)),)


def configure_cpu(threads=8, seed=0):
    """Pin the numerical environment: CPU, FP32, TF32 off, fixed thread count and seed."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(int(threads))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    return {"threads": int(torch.get_num_threads()), "seed": int(seed),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "torch": torch.__version__}


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------

def select_scene_indices(manifest, n_scenes=3):
    """Manifest indices of one query from each of ``n_scenes`` different rooms.

    Deterministic (manifest order) and spread as widely as the split allows: distinct *room
    categories* first (Apartments / Auditorium / Bathrooms / ...), falling back to distinct
    room instances if the split has fewer categories than requested.  Spreading the scenes
    matters because a single room could be accidentally symmetric and carry the result on its
    own.

    Raises:
        ValueError: if the manifest holds fewer than ``n_scenes`` distinct rooms.
    """
    by_category, by_room = [], []
    seen_category, seen_room = set(), set()
    for entry in manifest["entries"]:
        room = os.path.dirname(entry["query"])          # "<category>/<room>_idx_N"
        category = room.split("/")[0]
        if room not in seen_room:
            seen_room.add(room)
            by_room.append(int(entry["index"]))
        if category not in seen_category:
            seen_category.add(category)
            by_category.append(int(entry["index"]))
    if len(by_category) >= int(n_scenes):
        return by_category[:int(n_scenes)]
    if len(by_room) >= int(n_scenes):
        return by_room[:int(n_scenes)]
    raise ValueError("manifest has {} rooms, need {}".format(len(seen_room), n_scenes))


def real_batch(n_scenes=3, manifest_path=MANIFEST_PATH, expected_hash=MANIFEST_HASH,
               max_len=9600):
    """One collated batch of ``n_scenes`` real manifest queries, each from a different room.

    The manifest is verified against ``expected_hash`` (its *content* hash, the identity
    exp_03's sweeps were run under) before anything is loaded.

    Returns:
        A dict with ``depth_coord [B, 3, 256, 512]``, ``ref_irs [B, K, L]``, ``src_loc
        [B, 3]``, ``ref_locs [B, K, 3]``, ``tgt_wav [B, 1, L]``, ``keys`` (the query paths)
        and ``rooms``.

    Raises:
        FileNotFoundError: if the manifest is missing -- STOP and report, do not substitute.
        ValueError: if the manifest hash does not match.
    """
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError("reference manifest not found: {}".format(manifest_path))
    manifest = load_manifest(manifest_path)
    got = manifest_hash(manifest)
    if got != expected_hash:
        raise ValueError("manifest hash {} != expected {}".format(got, expected_hash))

    dataset = build_manifest_dataset(manifest, max_samples=0, max_len=max_len)
    indices = select_scene_indices(manifest, n_scenes)
    rows = [dataset[i] for i in indices]
    keys = [row[6] for row in rows]
    return {"depth_coord": torch.stack([row[2] for row in rows]),
            "ref_irs": torch.stack([row[4] for row in rows]),
            "src_loc": torch.stack([row[1] for row in rows]),
            "ref_locs": torch.stack([row[5] for row in rows]),
            "tgt_wav": torch.stack([row[3] for row in rows]),
            "keys": keys,
            "rooms": [os.path.dirname(k) for k in keys],
            "manifest_hash": got, "indices": indices}


def synthetic_batch(seed, batch=2, n_ref=8, length=9600, label="synthetic"):
    """An asymmetric random scene: no rotational symmetry to make invariance trivial.

    The panorama is per-pixel random (so no depth map is its own rotation), the source and
    every reference sit at generic off-axis positions, and the reference RIRs are random
    decaying noise.

    Args:
        seed: RNG seed; the batch is a pure function of it.
        batch: rows.
        n_ref: references per row (also exercises the reduced-K path when < 8).
        length: RIR length in samples.
        label: prefix for the synthetic "query keys" used by the Griffin-Lim seeding.

    Returns:
        The same dict shape :func:`real_batch` returns.
    """
    gen = torch.Generator().manual_seed(int(seed))

    depth = 0.5 + 6.0 * torch.rand(batch, 1, 256, 512, generator=gen)  # radial depth, metres
    phi = (torch.arange(256).float() + 0.5) * np.pi / 256 - np.pi / 2
    theta = (torch.arange(512).float() + 0.5) * 2 * np.pi / 512 - np.pi
    cos_phi, sin_phi = phi.cos()[:, None], phi.sin()[:, None]
    cos_t, sin_t = theta.cos()[None, :], theta.sin()[None, :]
    depth_coord = torch.stack([depth[:, 0] * cos_phi * cos_t,
                               depth[:, 0] * cos_phi * sin_t,
                               -depth[:, 0] * sin_phi], dim=1)        # [B, 3, 256, 512]

    src_loc = torch.stack([1.5 + 2.0 * torch.rand(batch, generator=gen),
                           -2.0 + 3.0 * torch.rand(batch, generator=gen),
                           -0.8 + 1.6 * torch.rand(batch, generator=gen)], dim=-1)
    ref_locs = torch.stack([-3.0 + 6.0 * torch.rand(batch, n_ref, generator=gen),
                            -3.0 + 6.0 * torch.rand(batch, n_ref, generator=gen),
                            -1.0 + 2.0 * torch.rand(batch, n_ref, generator=gen)], dim=-1)
    decay = torch.exp(-torch.arange(length).float() / 1200.0)
    ref_irs = 0.2 * torch.randn(batch, n_ref, length, generator=gen) * decay
    tgt_wav = 0.2 * torch.randn(batch, 1, length, generator=gen) * decay
    keys = ["{}_seed{}_K{}/row{}".format(label, seed, n_ref, i) for i in range(batch)]
    return {"depth_coord": depth_coord, "ref_irs": ref_irs, "src_loc": src_loc,
            "ref_locs": ref_locs, "tgt_wav": tgt_wav, "keys": keys,
            "rooms": ["{}#{}".format(label, seed)] * batch, "indices": list(range(batch))}


def adversarial_batch(src, refs, seed=303, label="adversarial"):
    """A synthetic scene with its geometry replaced by a hand-picked adversarial one.

    The panorama, reference RIRs and target keep :func:`synthetic_batch`'s asymmetric random
    content; only ``src_loc`` / ``ref_locs`` are pinned, which is what the two review blockers
    are about.

    Args:
        src: ``[[x, y, z]]`` query position(s).
        refs: ``[[[x, y, z], ...]]`` reference positions.
        seed: RNG seed of the surrounding synthetic scene.
        label: prefix for the Griffin-Lim query keys.
    """
    src_t = torch.tensor(src, dtype=torch.float32)
    refs_t = torch.tensor(refs, dtype=torch.float32)
    batch = synthetic_batch(seed, batch=src_t.shape[0], n_ref=refs_t.shape[1], label=label)
    batch["src_loc"] = src_t
    batch["ref_locs"] = refs_t
    return batch


def superseded_round2_basis(src_loc, ref_locs, eps=1.0e-6):
    """The **superseded** round-2 basis rule, kept only to measure what replacing it bought.

    ``a = q_h / ||q_h||`` when ``||q_h|| > eps``, else ``s / ||s||`` when ``||s|| > eps``, else
    zero -- two hard thresholds on absolute lengths, which is what review round 2 rejected.  It
    is never used by any model; :func:`basis_smoothness_sweep` calls it to quantify the jump the
    current rule does not have.
    """
    q = src_loc[:, :2].double()
    r_q = torch.linalg.norm(q, dim=-1)
    total = ref_locs[:, :, :2].double().sum(dim=1)
    r_total = torch.linalg.norm(total, dim=-1)
    use_query = r_q > float(eps)
    use_total = (~use_query) & (r_total > float(eps))
    q_unit = q / r_q.clamp_min(float(eps)).unsqueeze(-1)
    total_unit = total / r_total.clamp_min(float(eps)).unsqueeze(-1)
    return torch.where(use_query.unsqueeze(-1), q_unit,
                       torch.where(use_total.unsqueeze(-1), total_unit,
                                   torch.zeros_like(q))).to(src_loc.dtype)


def basis_smoothness_sweep(n_steps=400):
    """Walk two families of configurations across both old thresholds and the new blend band.

    A continuous basis moves a little per step; a thresholded one jumps by O(1) at one step.
    Reporting the **largest single step** therefore separates the two without needing to know
    where the threshold was.

    * ``cancellation`` -- two references ``(A, 0, 0)`` and ``(-A + d, 0, 0)`` with ``d``
      log-spaced across the round-2 ``1e-6`` m guard, at ``A = 20`` m so the guard is absurdly
      small relative to the scene (blocker B1(a)'s complaint that the threshold was absolute).
    * ``query_blend`` -- the query tilted from ``||q_h||/||q|| = 1e-8`` to ``1e-1`` at
      ``||q|| = 1`` m, so it crosses both the new band ``[1e-3, 1e-2]`` and the round-2 absolute
      ``1e-6`` m guard on the primary branch.

    Returns:
        ``{family: {"max_step_current", "max_step_round2", "n_steps", "range"}}``.
    """
    out = {}

    amplitude = 20.0
    gaps = torch.logspace(-10, -2, int(n_steps), dtype=torch.float64)
    src = torch.zeros(int(n_steps), 3, dtype=torch.float64)
    src[:, 2] = 1.0                                       # query straight up: fallback regime
    refs = torch.zeros(int(n_steps), 2, 3, dtype=torch.float64)
    refs[:, 0, 0] = amplitude
    refs[:, 1, 0] = -amplitude + gaps
    current, _ = horizontal_basis(src.float(), refs.float())
    legacy = superseded_round2_basis(src.float(), refs.float())
    out["cancellation"] = {
        "n_steps": int(n_steps), "range": [float(gaps[0]), float(gaps[-1])],
        "max_step_current": float((current[1:] - current[:-1]).norm(dim=-1).max()),
        "max_step_round2": float((legacy[1:] - legacy[:-1]).norm(dim=-1).max())}

    fractions = torch.logspace(-8, -1, int(n_steps), dtype=torch.float64)
    src = torch.zeros(int(n_steps), 3, dtype=torch.float64)
    src[:, 0] = fractions                                 # ||q_h||/||q|| ~ fractions, ||q|| ~ 1
    src[:, 2] = torch.sqrt((1.0 - fractions ** 2).clamp_min(0.0))
    refs = torch.zeros(int(n_steps), 2, 3, dtype=torch.float64)
    refs[:, 0, 1] = 4.0
    refs[:, 1, 0] = 1.0
    current, blend = horizontal_basis(src.float(), refs.float())
    legacy = superseded_round2_basis(src.float(), refs.float())
    out["query_blend"] = {
        "n_steps": int(n_steps), "range": [float(fractions[0]), float(fractions[-1])],
        "max_step_current": float((current[1:] - current[:-1]).norm(dim=-1).max()),
        "max_step_round2": float((legacy[1:] - legacy[:-1]).norm(dim=-1).max()),
        "blend_min": float(blend.min()), "blend_max": float(blend.max())}
    return out


def preround_delay(src_loc, ref_locs, dtype=torch.float64, sr=22050, c=343.0):
    """The direct-path delay *before* ``round()``, reduced in ``dtype``.

    The rotation itself always happens in float32 (that is what the model receives); only the
    norm/difference reduction takes ``dtype``, which is exactly the choice
    :meth:`model.xRIR_cyl_invariant.xRIR_InvariantBase.shift_and_align` makes.
    """
    src, refs = src_loc.to(dtype), ref_locs.to(dtype)
    return (torch.linalg.norm(src, dim=1).unsqueeze(1)
            - torch.linalg.norm(refs, dim=-1)) / float(c) * int(sr)


def integer_delays64(src_loc, ref_locs, sr=22050, c=343.0):
    """``tools.yaw_rotation.integer_delays`` with the invariant arms' float64 reduction."""
    return torch.round(preround_delay(src_loc, ref_locs, torch.float64, sr, c)).int()


def delay_flip_counts_for(model, src_loc, ref_locs, angles):
    """Delay flips counted with the reduction the *model* actually uses.

    The pinned arms reduce in float32 (``tools.yaw_rotation.integer_delays`` mirrors them line
    for line); the exp_08 invariant arms reduce in float64.  Counting an invariant arm with the
    float32 helper would report a boundary it no longer has.
    """
    if not isinstance(model, xRIR_InvariantBase):
        return delay_flip_counts(src_loc, ref_locs, angles)
    baseline = integer_delays64(src_loc, ref_locs)
    counts = {}
    for k in angles:
        angle = yaw_angle_rad(int(k))
        rotated = integer_delays64(rotate_vectors_z(src_loc, angle),
                                   rotate_vectors_z(ref_locs, angle))
        counts[int(k)] = int((rotated != baseline).sum())
    return counts


def preround_deviation(src_loc, ref_locs, angles=C16_ANGLES):
    """Largest pre-round delay deviation a C16 rotation induces, per reduction dtype.

    **Read this with care.** It is *not* a fair comparison of the two reductions: the float32
    value is itself quantised to float32, so two rotated scenes often collapse onto the same
    representable number and the measured deviation comes out optically *smaller* than the true
    one.  The float64 number is the honest deviation the float32 *inputs* imply.  What actually
    decides whether a scene flips is :func:`delay_flip_rate`, which counts flips directly.
    """
    widths = {}
    for name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        base = preround_delay(src_loc, ref_locs, dtype).double()
        worst = 0.0
        for k in angles:
            angle = yaw_angle_rad(int(k))
            rotated = preround_delay(rotate_vectors_z(src_loc, angle),
                                     rotate_vectors_z(ref_locs, angle), dtype).double()
            worst = max(worst, float((rotated - base).abs().max()))
        widths[name] = worst
    return widths


def delay_flip_rate(n_scenes=200000, n_refs=8, angles=C16_ANGLES, seed=12345, extent=6.0,
                    height_scale=0.25):
    """How often a C16 rotation moves an integer direct-path delay, float32 vs float64.

    This is the operative measure of the rounding boundary: a random ensemble of scenes is
    rotated through all 15 C16 angles and the integer delays are compared, once with the pinned
    float32 reduction and once with the invariant arms' float64 reduction.  Counting flips
    sidesteps the quantisation artefact that makes :func:`preround_deviation` unreadable.

    Flips are **correlated within a scene** -- one scene whose pre-round delay sits on a tie
    flips at most angles -- so treating the totals as independent Poisson counts overstates the
    significance (exp_08 review round 2, B2(i)).  The significance is therefore computed from
    the **paired per-scene difference**: ``d_i = flips32_i - flips64_i`` over the ensemble, with
    ``SE = sqrt(n) * sd(d)``.  The Poisson figures are still returned, labelled as the
    over-optimistic bound they are, so the two can be compared.

    Args:
        n_scenes: ensemble size.
        n_refs: references per scene.
        angles: column rolls to test.
        seed: RNG seed (the ensemble is a pure function of it).
        extent: half-width of the uniform box the positions are drawn from, in metres.
        height_scale: vertical extent as a fraction of ``extent`` (rooms are wider than tall).

    Returns:
        ``{"n_scenes", "n_refs", "n_angles", "comparisons", per-dtype
        {"flips", "rate", "poisson_sigma", "scenes_with_flip"}, "shrink_factor"}``.
    """
    generator = torch.Generator().manual_seed(int(seed))
    src = torch.rand(int(n_scenes), 3, generator=generator) * (2 * extent) - extent
    ref = torch.rand(int(n_scenes), int(n_refs), 3, generator=generator) * (2 * extent) - extent
    src[:, 2] *= float(height_scale)
    ref[:, :, 2] *= float(height_scale)

    out = {"n_scenes": int(n_scenes), "n_refs": int(n_refs), "n_angles": len(angles),
           "comparisons": int(n_scenes) * int(n_refs) * len(angles), "seed": int(seed),
           "distribution": {"positions": "uniform box", "half_width_m": float(extent),
                            "height_scale": float(height_scale)}}
    per_scene = {}
    for name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        base = torch.round(preround_delay(src, ref, dtype)).int()
        counts = torch.zeros(int(n_scenes), dtype=torch.float64)
        touched = torch.zeros(int(n_scenes), dtype=torch.bool)
        for k in angles:
            angle = yaw_angle_rad(int(k))
            rotated = torch.round(preround_delay(rotate_vectors_z(src, angle),
                                                 rotate_vectors_z(ref, angle), dtype)).int()
            differs = rotated != base
            counts += differs.sum(dim=1).double()
            touched |= differs.any(dim=1)
        per_scene[name] = counts
        flips = int(counts.sum())
        out[name] = {"flips": flips, "rate": flips / out["comparisons"],
                     "poisson_sigma_optimistic": float(flips) ** 0.5,
                     "scenes_with_flip": int(touched.sum())}
    out["shrink_factor"] = (out["float32"]["flips"] / out["float64"]["flips"]
                            if out["float64"]["flips"] else float("inf"))

    # Paired per-scene difference: this is the honest significance (B2(i)).
    difference = per_scene["float32"] - per_scene["float64"]
    total = float(difference.sum())
    standard_error = float(difference.std(unbiased=True) * math.sqrt(int(n_scenes)))
    out["paired"] = {
        "total_difference": total, "standard_error": standard_error,
        "sigma": total / standard_error if standard_error > 0 else float("inf"),
        "scenes_where_float32_flips_more": int((difference > 0).sum()),
        "scenes_where_float64_flips_more": int((difference < 0).sum())}
    naive = math.sqrt(out["float32"]["flips"] + out["float64"]["flips"])
    out["paired"]["naive_poisson_sigma"] = total / naive if naive > 0 else float("inf")
    return out


# --------------------------------------------------------------------------------------
# Forward passes and residuals
# --------------------------------------------------------------------------------------

def forward_at(model, batch, k, aligned0=None):
    """Predict the batch under a yaw of ``2*pi*k/512``.

    Args:
        model: an xRIR (or subclass) in ``eval()`` mode on CPU.
        batch: a dict from :func:`real_batch` / :func:`synthetic_batch` (**unrotated**).
        k: integer column roll.
        aligned0: if given, the ``k = 0`` aligned references are pinned for this call
            (condition P); otherwise the alignment is recomputed from the rotated
            coordinates (condition E, the true end-to-end path).

    Returns:
        ``out_log_spec`` ``[B, F, T, 1]``.
    """
    depth_k, src_k, refs_k = rotate_scene_yaw(
        batch["depth_coord"], batch["src_loc"], batch["ref_locs"], int(k),
        W=batch["depth_coord"].shape[-1])
    with torch.no_grad():
        if aligned0 is None:
            out, _ = model(depth_k, batch["ref_irs"], src_k, refs_k, batch["tgt_wav"])
        else:
            with fixed_alignment(model, aligned0):
                out, _ = model(depth_k, batch["ref_irs"], src_k, refs_k, batch["tgt_wav"])
    return out


def residual(out_k, out_0):
    """Frobenius residual of one angle against ``k = 0``, with the scale reported next to it."""
    diff = (out_k - out_0).double()
    denom = float(out_0.double().norm())
    abs_fro = float(diff.norm())
    per_row = [float((out_k[i] - out_0[i]).double().norm()
                     / max(float(out_0[i].double().norm()), 1e-300))
               for i in range(out_0.shape[0])]
    return {"rel_fro": abs_fro / denom if denom > 0 else float("nan"),
            "abs_fro": abs_fro, "denom_fro": denom,
            "max_abs": float(diff.abs().max()),
            "rel_fro_per_row": per_row}


def angle_sweep(model, batch, angles=C16_ANGLES, conditions=("E", "P")):
    """Residuals of ``model`` on ``batch`` at every angle, per condition.

    Returns:
        ``{"k0_norm", "rows": [{"k", "angle_deg", "condition", ...residual...}],
        "delay_flips": {k: count}, "seconds"}``.
    """
    started = time.time()
    aligned0 = None
    with torch.no_grad():
        aligned0 = model.shift_and_align(batch["ref_irs"], batch["src_loc"], batch["ref_locs"])
        out_0, _ = model(batch["depth_coord"], batch["ref_irs"], batch["src_loc"],
                         batch["ref_locs"], batch["tgt_wav"])
    rows, predictions = [], {}
    for k in angles:
        for condition in conditions:
            out_k = forward_at(model, batch, k, aligned0 if condition == "P" else None)
            if condition == "E":
                predictions[int(k)] = out_k
            rows.append({"k": int(k), "angle_deg": float(np.degrees(yaw_angle_rad(int(k), 512))),
                         "condition": condition, **residual(out_k, out_0)})
    return {"k0_norm": float(out_0.double().norm()), "rows": rows, "predictions": predictions,
            "out_0": out_0,
            "delay_flips": delay_flip_counts_for(model, batch["src_loc"], batch["ref_locs"],
                                                 angles),
            "seconds": time.time() - started}


def waveform_residuals(out_0, predictions, keys, gl_seed=0):
    """Seeded-Griffin-Lim waveform residuals at every angle (handoff sections 3.4 / 6).

    The magnitudes are rebuilt exactly as ``eval_xRIR_backbone.py`` does
    (``exp(out) - 1e-8``) and inverted with a phase seeded from ``(gl_seed, query key)`` --
    the **same, yaw-independent** seed at every angle, so any difference is the model's and
    not Griffin-Lim's random start.

    Returns:
        ``[{"k", "row", "key", "rel_l2", "abs_l2", "denom_l2"}]``.
    """
    base = []
    for i, key in enumerate(keys):
        mag = (torch.exp(out_0[i:i + 1]) - 1e-8)[..., 0]
        base.append(griffin_lim_seeded(mag, sample_seed(gl_seed, key))[0].double())
    rows = []
    for k in sorted(predictions):
        out_k = predictions[k]
        for i, key in enumerate(keys):
            mag = (torch.exp(out_k[i:i + 1]) - 1e-8)[..., 0]
            wav = griffin_lim_seeded(mag, sample_seed(gl_seed, key))[0].double()
            denom = float(base[i].norm())
            abs_l2 = float((wav - base[i]).norm())
            rows.append({"k": int(k), "row": i, "key": key,
                         "rel_l2": abs_l2 / denom if denom > 0 else float("nan"),
                         "abs_l2": abs_l2, "denom_l2": denom})
    return rows


def input_rounding_control(model, batch, k=32):
    """The model's float32 noise floor under a perturbation the *size* of the rotation's rounding.

    The rotated scene is rotated straight back -- ``roll(-k)`` then ``Rz(-2*pi*k/512)`` -- which
    is the identity in exact arithmetic.  What comes back therefore differs from the original
    scene only by float32 rounding, of exactly the magnitude a rotation introduces, while being
    **geometrically unrotated**.  Feeding that through the model measures how much the output
    moves for purely numerical reasons, with no rotation involved at all.

    If the C16 residual lands on the same order as this number, the residual is the noise floor
    of running a float32 network on rounded inputs -- not a property of the architecture, and not
    something more training or a different readout could remove.

    Note that at ``k = 128`` and ``k = 256`` (90 and 180 deg) the float32 ``Rz`` is exact, so the
    control degenerates to zero there and says nothing; pick angles whose sine and cosine are
    both non-trivial.  It is also a *lower* bound on the true floor: it re-rounds the coordinates
    but leaves the panorama column/gauge pairing alone, which a real roll also perturbs.

    Args:
        model: the model in ``eval()`` mode.
        batch: an unrotated batch dict.
        k: which rotation's rounding to imitate.

    Returns:
        ``{"k", "input_rel_*", "output": {...residual...}}``.
    """
    depth_k, src_k, refs_k = rotate_scene_yaw(
        batch["depth_coord"], batch["src_loc"], batch["ref_locs"], int(k),
        W=batch["depth_coord"].shape[-1])
    width = batch["depth_coord"].shape[-1]
    back_angle = -yaw_angle_rad(int(k), width)
    depth_rt = rotate_vectors_z(
        torch.roll(depth_k, shifts=-int(k), dims=-1).permute(0, 2, 3, 1),
        back_angle).permute(0, 3, 1, 2).contiguous()
    src_rt = rotate_vectors_z(src_k, back_angle)
    refs_rt = rotate_vectors_z(refs_k, back_angle)

    relative = lambda a, b: float((a - b).double().norm() / b.double().norm())
    with torch.no_grad():
        out_0, _ = model(batch["depth_coord"], batch["ref_irs"], batch["src_loc"],
                         batch["ref_locs"], batch["tgt_wav"])
        out_rt, _ = model(depth_rt, batch["ref_irs"], src_rt, refs_rt, batch["tgt_wav"])
    return {"k": int(k),
            "input_rel_depth": relative(depth_rt, batch["depth_coord"]),
            "input_rel_src": relative(src_rt, batch["src_loc"]),
            "input_rel_refs": relative(refs_rt, batch["ref_locs"]),
            "output": residual(out_rt, out_0)}


def waveform_conditioning_control(out_0, keys, rel_magnitude, gl_seed=0, noise_seed=12345):
    """How far Griffin-Lim moves under a *non-rotational* perturbation of the same size.

    Griffin-Lim is an iterative non-convex phase retrieval: even with the phase initialisation
    pinned, its fixed point is far more sensitive to its input than a linear map would be, so
    a waveform residual much larger than the spectral one is not by itself evidence of a
    rotation effect.  This control replaces the rotation with iid Gaussian noise scaled to the
    *same* relative Frobenius magnitude and inverts with the identical seeds; if the two
    residuals land on the same order, the amplification belongs to Griffin-Lim, not the model.

    Args:
        out_0: the ``k = 0`` log-magnitude prediction ``[B, F, T, 1]``.
        keys: the per-row query keys (same Griffin-Lim seeding as the real measurement).
        rel_magnitude: relative Frobenius size of the perturbation to inject.
        gl_seed: run-level Griffin-Lim seed.
        noise_seed: seed of the injected perturbation (independent of the Griffin-Lim phase).

    Returns:
        ``{"rel_magnitude", "deterministic", "rows": [{"row", "key", "rel_l2", ...}]}``;
        ``deterministic`` records that two inversions of *identical* magnitudes agree bitwise.
    """
    generator = torch.Generator().manual_seed(int(noise_seed))
    noise = torch.randn(out_0.shape, generator=generator)
    noise = noise / noise.norm() * (float(rel_magnitude) * out_0.norm())
    perturbed = out_0 + noise

    rows, deterministic = [], True
    for i, key in enumerate(keys):
        seed = sample_seed(gl_seed, key)
        base = griffin_lim_seeded((torch.exp(out_0[i:i + 1]) - 1e-8)[..., 0], seed)[0].double()
        again = griffin_lim_seeded((torch.exp(out_0[i:i + 1]) - 1e-8)[..., 0], seed)[0].double()
        deterministic = deterministic and bool(torch.equal(base, again))
        wav = griffin_lim_seeded((torch.exp(perturbed[i:i + 1]) - 1e-8)[..., 0], seed)[0].double()
        denom = float(base.norm())
        abs_l2 = float((wav - base).norm())
        rows.append({"row": i, "key": key, "abs_l2": abs_l2, "denom_l2": denom,
                     "rel_l2": abs_l2 / denom if denom > 0 else float("nan")})
    return {"rel_magnitude": float(rel_magnitude), "deterministic": bool(deterministic),
            "rows": rows}


def training_loss(out_spec, tgt_spec):
    """``train_xRIR_backbone.compute_loss``'s objective, device-preserving.

    Same two terms in the same order (STFT log-magnitude L1 + Schroeder energy-decay L1) and
    the same stdout redirection; only the ``.cuda()`` transfers of the trainer are absent.
    """
    gt = tgt_spec.permute(0, 2, 3, 1)
    with contextlib.redirect_stdout(io.StringIO()):
        decay_loss = compute_spect_energy_decay_losses(gts=gt, preds=torch.exp(out_spec) - 1e-8)
    stft_loss = stft_l1_loss(pred_spect=out_spec, gt_spect=gt)
    return stft_loss + decay_loss, stft_loss.detach(), decay_loss.detach()


#: Parameters the gradient must actually reach: the new readout and the coordinate branch.
NEW_PARAMS = ("invariant_readout.weight", "invariant_readout.bias")
COORD_PARAMS = ("src_coord_proj.proj.weight", "src_coord_proj.proj.bias")
#: A sample of pinned parameters that must keep training.
OLD_PARAMS = ("src_proj.proj.weight", "lin_proj_1.proj.weight", "lin_proj_2.proj.weight",
              "time_proj.proj.weight", "audio_enc.cnn.conv1.weight",
              "source_network.to_patch_embedding.2.weight")


def grad_step_evidence(model, batch, lr=1.0e-3, weight_decay=1.0e-4):
    """One forward + backward + AdamW step on a real batch (handoff section 4 item 5).

    Checks that the gradient is finite everywhere, that it actually *reaches* the new
    readout and the coordinate branch (a non-zero gradient, not merely a present one), and
    that the pinned parameters still move.

    Returns:
        A dict with the loss terms, the global finite-gradient verdict, per-parameter
        gradient norms and post-step parameter deltas, and the counts of trainable tensors
        that received a gradient / changed.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    out_spec, tgt_spec = model(batch["depth_coord"], batch["ref_irs"], batch["src_loc"],
                               batch["ref_locs"], batch["tgt_wav"])
    loss, stft_loss, decay_loss = training_loss(out_spec, tgt_spec)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    grads, trainable, with_grad, finite = {}, 0, 0, True
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        trainable += 1
        if p.grad is None:
            grads[name] = None
            continue
        norm = float(p.grad.detach().double().norm())
        grads[name] = norm
        finite = finite and bool(torch.isfinite(p.grad).all())
        if norm > 0.0:
            with_grad += 1

    optimizer.step()
    deltas, changed = {}, 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        delta = float((p.detach() - before[name]).double().abs().max())
        deltas[name] = delta
        if delta > 0.0:
            changed += 1

    model.eval()
    watched = list(NEW_PARAMS) + list(COORD_PARAMS) + list(OLD_PARAMS)
    return {"loss": float(loss), "stft": float(stft_loss), "decay": float(decay_loss),
            "all_grads_finite": bool(finite),
            "trainable_tensors": trainable, "tensors_with_nonzero_grad": with_grad,
            "tensors_changed_by_step": changed,
            "watched": {name: {"grad_norm": grads.get(name), "max_delta": deltas.get(name)}
                        for name in watched},
            "grad_norms": grads, "max_deltas": deltas}


def pinned_regression(batch, seed=0, num_shot=8):
    """The pinned entries must be bit-identical through the exp_08 factory.

    Builds ``xRIR`` / ``xRIR_Cyl`` directly and through ``build_xrir_exp08`` from the same
    seed, and compares the predictions with ``torch.equal`` on the given real batch.

    Returns:
        ``{backbone: {"class_identity", "bitwise_equal", "max_abs_diff"}}``.
    """
    from model.xRIR import xRIR
    from model.xRIR_cyl import xRIR_Cyl
    from model.exp08_factory import BACKBONES_EXP08

    pinned = {"simple": xRIR, "cylindrical": xRIR_Cyl}
    out = {}
    for name, cls in pinned.items():
        identity = BACKBONES_EXP08[name] is cls
        torch.manual_seed(seed)
        reference = cls(num_channels=num_shot).eval()
        torch.manual_seed(seed)
        through_factory = build_xrir_exp08(name, num_shot).eval()
        with torch.no_grad():
            a, _ = reference(batch["depth_coord"], batch["ref_irs"], batch["src_loc"],
                            batch["ref_locs"], batch["tgt_wav"])
            b, _ = through_factory(batch["depth_coord"], batch["ref_irs"], batch["src_loc"],
                                   batch["ref_locs"], batch["tgt_wav"])
        out[name] = {"class_identity": bool(identity),
                     "bitwise_equal": bool(torch.equal(a, b)),
                     "max_abs_diff": float((a - b).abs().max())}
    return out


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def worst(rows, condition="E"):
    """The row with the largest ``rel_fro`` among ``rows`` in ``condition``."""
    subset = [r for r in rows if r["condition"] == condition]
    return max(subset, key=lambda r: r["rel_fro"]) if subset else None


def format_sweep(label, sweep, condition="E"):
    """A compact per-angle table."""
    lines = ["{}  (condition {}; ||out_0||_F = {:.6g})".format(label, condition,
                                                               sweep["k0_norm"]),
             "      k   deg      rel_fro      abs_fro      max|d|   delay_flips"]
    for row in sweep["rows"]:
        if row["condition"] != condition:
            continue
        lines.append("  {:5d} {:6.1f}   {:10.3e}   {:10.3e}   {:9.3e}   {:5d}".format(
            row["k"], row["angle_deg"], row["rel_fro"], row["abs_fro"], row["max_abs"],
            sweep["delay_flips"].get(row["k"], 0)))
    return "\n".join(lines)


def strip_tensors(sweep):
    """The JSON-serialisable part of an :func:`angle_sweep` result."""
    return {key: value for key, value in sweep.items()
            if key not in ("predictions", "out_0")}


def run(args):
    env = configure_cpu(args.threads, args.seed)
    print("environment: {}".format(json.dumps(env)), flush=True)
    started = time.time()
    report = {"environment": env, "rel_target": REL_TARGET, "angles": list(C16_ANGLES),
              "off_lattice_angles": list(OFF_LATTICE_ANGLES), "cases": {}}

    from tools.exp08_warmstart import build_warm_started

    with device_preserving_delay():
        real = real_batch(args.n_scenes)
        print("real batch: rooms={} keys={}".format(real["rooms"], real["keys"]), flush=True)
        report["real_batch"] = {"keys": real["keys"], "rooms": real["rooms"],
                                "indices": real["indices"],
                                "manifest_hash": real["manifest_hash"],
                                "batch": int(real["src_loc"].shape[0]),
                                "num_shot": int(real["ref_locs"].shape[1])}

        synth_a = synthetic_batch(101, batch=2, n_ref=8, label="synthA")
        synth_b = synthetic_batch(202, batch=2, n_ref=8, label="synthB")
        synth_k3 = synthetic_batch(303, batch=2, n_ref=3, label="synthK3")

        torch.manual_seed(args.seed)
        random_model = build_xrir_exp08("cylindrical_invariant", 8).eval()
        warm_model, accounting, ckpt_path = build_warm_started(
            "cylindrical_invariant", 8, verbose=True)
        warm_model.eval()
        report["warm_start"] = {"checkpoint": ckpt_path, "accounting": accounting}

        cases = [("random_init/real", random_model, real),
                 ("random_init/synthA", random_model, synth_a),
                 ("random_init/synthB", random_model, synth_b),
                 ("random_init/synthK3", random_model, synth_k3),
                 ("warm_start/real", warm_model, real),
                 ("warm_start/synthA", warm_model, synth_a),
                 ("warm_start/synthB", warm_model, synth_b),
                 ("warm_start/synthK3", warm_model, synth_k3)]
        for label, model, batch in cases:
            sweep = angle_sweep(model, batch, C16_ANGLES)
            print(format_sweep(label, sweep), flush=True)
            entry = strip_tensors(sweep)
            entry["worst_E"] = worst(sweep["rows"], "E")
            entry["worst_P"] = worst(sweep["rows"], "P")
            if label == "warm_start/real":
                entry["waveforms"] = waveform_residuals(
                    sweep["out_0"], sweep["predictions"], batch["keys"], gl_seed=args.gl_seed)
                worst_wav = max(entry["waveforms"], key=lambda r: r["rel_l2"])
                print("  waveform worst: k={} row={} rel_l2={:.3e} abs={:.3e} denom={:.3e}".format(
                    worst_wav["k"], worst_wav["row"], worst_wav["rel_l2"],
                    worst_wav["abs_l2"], worst_wav["denom_l2"]), flush=True)
                entry["waveform_control"] = waveform_conditioning_control(
                    sweep["out_0"], batch["keys"], worst(sweep["rows"], "E")["rel_fro"],
                    gl_seed=args.gl_seed)
                print("  waveform control (same-size non-rotational perturbation, rel={:.3e}): "
                      "worst rel_l2={:.3e}, GL deterministic={}".format(
                          entry["waveform_control"]["rel_magnitude"],
                          max(r["rel_l2"] for r in entry["waveform_control"]["rows"]),
                          entry["waveform_control"]["deterministic"]), flush=True)
            report["cases"][label] = entry

        # The float32 noise floor: a rotate-and-rotate-back perturbation of the same size,
        # with no rotation left in the geometry.
        report["input_rounding_control"] = {
            str(k): input_rounding_control(warm_model, real, k) for k in (32, 96, 224)}
        for k, entry in report["input_rounding_control"].items():
            print("input-rounding control k={}: input rel (depth {:.3e}, src {:.3e}, refs {:.3e}) "
                  "-> output rel {:.3e} (abs {:.3e}, denom {:.4g})".format(
                      k, entry["input_rel_depth"], entry["input_rel_src"],
                      entry["input_rel_refs"], entry["output"]["rel_fro"],
                      entry["output"]["abs_fro"], entry["output"]["denom_fro"]), flush=True)

        # --- blocker B1: every adversarial basis configuration, through the whole model ---
        report["basis_smoothness"] = basis_smoothness_sweep()
        for family, entry in report["basis_smoothness"].items():
            print("basis smoothness ({}): largest single step {:.3e} now vs {:.3e} under the "
                  "superseded round-2 rule, over {} configurations in [{:.1e}, {:.1e}]".format(
                      family, entry["max_step_current"], entry["max_step_round2"],
                      entry["n_steps"], entry["range"][0], entry["range"][1]), flush=True)

        basis_cases = [("equal_radius_tie", CODEX_TIE_REFS, "codexTie"),
                       ("sum_exactly_zero", CODEX_CANCEL_REFS, "codexCancel"),
                       ("near_cancellation", CODEX_NEARCANCEL_REFS, "codexNearCancel"),
                       ("all_horizontal_zero", CODEX_SUM_DEGENERATE_REFS, "codexSumZero")]
        report["adversarial_basis"] = {}
        for name, refs, label in basis_cases:
            case = adversarial_batch(CODEX_TIE_SRC, refs, label=label)
            sweep = angle_sweep(warm_model, case, C16_ANGLES, conditions=("E", "P"))
            print(format_sweep("B1 adversarial: {}".format(name), sweep), flush=True)
            entry = strip_tensors(sweep)
            entry["worst_E"] = worst(sweep["rows"], "E")
            entry["worst_P"] = worst(sweep["rows"], "P")

            src_i, ref_i, basis, blend = intrinsic_scene_coords(case["src_loc"],
                                                               case["ref_locs"])
            coord_worst, legacy_worst = 0.0, 0.0
            legacy_basis = superseded_round2_basis(case["src_loc"], case["ref_locs"])
            legacy_ref = to_intrinsic(case["ref_locs"], legacy_basis)
            for k in C16_ANGLES:
                angle = yaw_angle_rad(int(k))
                src_k, ref_k, _, _ = intrinsic_scene_coords(
                    rotate_vectors_z(case["src_loc"], angle),
                    rotate_vectors_z(case["ref_locs"], angle))
                coord_worst = max(coord_worst, float((src_k - src_i).abs().max()),
                                  float((ref_k - ref_i).abs().max()))
                legacy_k = to_intrinsic(rotate_vectors_z(case["ref_locs"], angle),
                                        superseded_round2_basis(
                                            rotate_vectors_z(case["src_loc"], angle),
                                            rotate_vectors_z(case["ref_locs"], angle)))
                legacy_worst = max(legacy_worst, float((legacy_k - legacy_ref).abs().max()))
            entry.update({"blend": float(blend[0]),
                          "basis_k0": [float(v) for v in basis[0]],
                          "basis_norm": float(basis[0].norm()),
                          "coord_worst_abs": coord_worst,
                          "coord_worst_abs_round2": legacy_worst,
                          "coord_scale": float(ref_i.abs().max()),
                          "scene_scale": float(case["ref_locs"].abs().max())})
            report["adversarial_basis"][name] = entry
            print("  blend {:.3f}, |basis| {:.6f}; coords worst |d| {:.3e} m (round-2 rule: "
                  "{:.3e} m) on a {:.1f} m scene; worst E {:.3e}, worst P {:.3e}".format(
                      entry["blend"], entry["basis_norm"], coord_worst, legacy_worst,
                      entry["scene_scale"], entry["worst_E"]["rel_fro"],
                      entry["worst_P"]["rel_fro"]), flush=True)

        # The production path must be untouched by the blend: every real query is far above the
        # band, so the basis is bitwise the plain unit vector and the coordinates are unchanged.
        real_basis, real_blend = horizontal_basis(real["src_loc"], real["ref_locs"])
        horizontal = real["src_loc"][:, :2].double()
        plain_unit = (horizontal / horizontal.norm(dim=-1, keepdim=True)).to(real["src_loc"].dtype)
        report["battery_primary_path"] = {
            "blend_min": float(real_blend.min()), "blend_max": float(real_blend.max()),
            "basis_bitwise_equals_plain_unit_vector": bool(torch.equal(real_basis, plain_unit)),
            "horizontal_fraction_min": float(
                (horizontal.norm(dim=-1) / real["src_loc"].double().norm(dim=-1)).min())}
        print("battery primary path: blend in [{:.1f}, {:.1f}], basis bitwise == q_h/||q_h||: "
              "{}, min horizontal fraction {:.4f} (band top {:.0e})".format(
                  report["battery_primary_path"]["blend_min"],
                  report["battery_primary_path"]["blend_max"],
                  report["battery_primary_path"]["basis_bitwise_equals_plain_unit_vector"],
                  report["battery_primary_path"]["horizontal_fraction_min"], BLEND_HI),
              flush=True)

        # --- blocker B2: the integer-delay rounding boundary ---
        report["preround_deviation"] = {
            "real": preround_deviation(real["src_loc"], real["ref_locs"]),
            "codex_case": preround_deviation(torch.tensor(CODEX_DELAY_SRC),
                                             torch.tensor(CODEX_DELAY_REFS))}
        report["delay_flip_rate"] = delay_flip_rate(n_scenes=args.flip_ensemble)
        for name, widths in report["preround_deviation"].items():
            print("pre-round deviation ({}): float32 {:.3e} samples, float64 {:.3e} "
                  "(float32 is masked by its own quantisation -- see delay_flip_rate)".format(
                      name, widths["float32"], widths["float64"]), flush=True)
        flip = report["delay_flip_rate"]
        print("delay flip rate over {} random scenes x {} refs x {} angles = {} comparisons: "
              "float32 {} flips ({:.3e}), float64 {} flips ({:.3e}), shrink {:.2f}x".format(
                  flip["n_scenes"], flip["n_refs"], flip["n_angles"], flip["comparisons"],
                  flip["float32"]["flips"], flip["float32"]["rate"],
                  flip["float64"]["flips"], flip["float64"]["rate"],
                  flip["shrink_factor"]), flush=True)
        print("  paired per-scene difference {:.0f} +- {:.2f} -> {:.2f} sigma (the independent "
              "Poisson figure, {:.2f} sigma, overstates it: flips correlate within a scene)"
              .format(flip["paired"]["total_difference"], flip["paired"]["standard_error"],
                      flip["paired"]["sigma"], flip["paired"]["naive_poisson_sigma"]), flush=True)

        boundary = adversarial_batch(CODEX_DELAY_SRC, CODEX_DELAY_REFS, label="codexDelay")
        boundary_sweep = angle_sweep(warm_model, boundary, C16_ANGLES, conditions=("E", "P"))
        print(format_sweep("B2 adversarial: pre-round delay on the 3.5 tie", boundary_sweep),
              flush=True)
        b_entry = strip_tensors(boundary_sweep)
        b_entry["worst_E"] = worst(boundary_sweep["rows"], "E")
        b_entry["worst_P"] = worst(boundary_sweep["rows"], "P")
        base64 = integer_delays64(boundary["src_loc"], boundary["ref_locs"])
        b_entry["delays_k0"] = base64.tolist()
        b_entry["preround_k0"] = preround_delay(boundary["src_loc"],
                                                boundary["ref_locs"]).tolist()
        per_angle = {}
        for k in C16_ANGLES:
            angle = yaw_angle_rad(int(k))
            rotated = integer_delays64(rotate_vectors_z(boundary["src_loc"], angle),
                                       rotate_vectors_z(boundary["ref_locs"], angle))
            per_angle[int(k)] = {"delays": rotated.tolist(),
                                 "flips": int((rotated != base64).sum()),
                                 "max_abs_shift": int((rotated - base64).abs().max())}
        b_entry["delays_per_angle"] = per_angle
        report["adversarial_delay_boundary"] = b_entry
        print("  k=0 pre-round {} -> delays {}; flips per angle {}; worst E {:.3e}, "
              "worst P {:.3e}".format(
                  [round(v, 9) for v in b_entry["preround_k0"][0]], b_entry["delays_k0"],
                  {k: v["flips"] for k, v in per_angle.items()},
                  b_entry["worst_E"]["rel_fro"], b_entry["worst_P"]["rel_fro"]), flush=True)

        # Off-lattice diagnostic (no pass/fail).
        off = angle_sweep(warm_model, real, OFF_LATTICE_ANGLES, conditions=("E",))
        print(format_sweep("off-lattice diagnostic (warm_start/real)", off), flush=True)
        report["off_lattice"] = strip_tensors(off)

        # Non-vacuity control: the pinned cylindrical model must fail the same criterion.
        torch.manual_seed(args.seed)
        control = build_xrir_exp08("cylindrical", 8).eval()
        control_sweep = angle_sweep(control, real, C16_ANGLES, conditions=("E",))
        print(format_sweep("CONTROL pinned cylindrical (must FAIL)", control_sweep), flush=True)
        report["control_pinned_cylindrical"] = strip_tensors(control_sweep)
        report["control_pinned_cylindrical"]["worst_E"] = worst(control_sweep["rows"], "E")

        # SimpleViT control arm: no guarantee expected, reported for attribution.
        simple_warm, _, _ = build_warm_started("simple_invariant", 8, verbose=False)
        simple_sweep = angle_sweep(simple_warm.eval(), real, C16_ANGLES, conditions=("E",))
        print(format_sweep("simple_invariant (no guarantee expected)", simple_sweep), flush=True)
        report["simple_invariant"] = strip_tensors(simple_sweep)
        report["simple_invariant"]["worst_E"] = worst(simple_sweep["rows"], "E")

        # Gradient step and pinned regression, both on the real batch.
        torch.manual_seed(args.seed)
        grad_model = build_xrir_exp08("cylindrical_invariant", 8)
        report["grad_step"] = grad_step_evidence(grad_model, real)
        print("grad step: loss={:.4f} finite={} nonzero_grads={}/{} changed={}".format(
            report["grad_step"]["loss"], report["grad_step"]["all_grads_finite"],
            report["grad_step"]["tensors_with_nonzero_grad"],
            report["grad_step"]["trainable_tensors"],
            report["grad_step"]["tensors_changed_by_step"]), flush=True)

        report["regression"] = pinned_regression(real, seed=args.seed)
        print("regression: {}".format(json.dumps(report["regression"])), flush=True)

    report["seconds"] = time.time() - started
    print("total {:.1f} s".format(report["seconds"]), flush=True)

    if args.out_json:
        parent = os.path.dirname(os.path.abspath(args.out_json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out_json, "w") as fout:
            json.dump(report, fout, indent=1)
        print("report -> {}".format(args.out_json), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-scenes", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gl-seed", type=int, default=0)
    parser.add_argument("--flip-ensemble", type=int, default=1000000,
                        help="scenes in the integer-delay flip-rate ensemble")
    parser.add_argument("--out-json", default=None)
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
