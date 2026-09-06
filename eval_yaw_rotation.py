"""Yaw-rotation degradation sweep for one xRIR checkpoint (exp_03).

One pass over the pinned reference manifest (``ckpt/yaw_rotation/reference_manifest.json``)
evaluates, for every yaw angle ``k`` of the grid, two conditions of the *same* query:

* **P** (primary) -- the direct-path alignment is pinned at the ``k = 0`` coordinates
  (``tools.yaw_rotation.fixed_alignment``), so only the geometry branches of the forward
  pass see the rotation.  ``xRIR.shift_and_align`` is algebraically yaw-invariant but not
  numerically (float32 norms plus ``round()`` can flip a delay by one sample), and P
  isolates the model's geometric sensitivity from that quantisation noise.
* **E** (end-to-end) -- a plain forward on the rotated views, i.e. the alignment is
  recomputed from the rotated coordinates, which is what a deployed pipeline would do.

Both conditions collapse onto the plain forward at ``k = 0``, so every metric is a paired
comparison of a query against itself.  Per-sample metrics land in
``<out-dir>/per_sample_yaw.json`` and their aggregates in ``<out-dir>/metrics_yaw.json``;
``tools/summarize_yaw.py`` turns those into the confirmatory tables.

    python eval_yaw_rotation.py --backbone simple \\
        --checkpoint ckpt/xRIR_simple_8_shot/epoch_12.pth \\
        --manifest ckpt/yaw_rotation/reference_manifest.json --manifest-hash <sha256> \\
        --out-dir ckpt/yaw_rotation/simple_control

``model.xRIR.apply_delay`` hard-codes ``.cuda()``, so the whole evaluation runs on GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_unseen import Evaluator
from eval_xRIR_backbone import load_model_state
from model.xRIR_cyl import BACKBONES, build_xrir

from tools.reference_manifest import ManifestDataset, load_manifest, manifest_hash
from tools.per_sample_metrics import (
    acoustic_metrics,
    griffin_lim_seeded,
    per_sample_losses,
    sample_seed,
)
from tools.yaw_rotation import (
    fixed_alignment,
    integer_delays,
    rotate_scene_yaw,
    rotate_vectors_z,
    yaw_angle_rad,
)


def rotated_views(depth_coord, src_loc, ref_locs, k):
    """The scene under an active yaw of ``2*pi*k/W`` about the receiver's vertical axis.

    A thin wrapper around :func:`tools.yaw_rotation.rotate_scene_yaw` that takes the
    panorama width from the tensor itself, so a caller can never pass a ``W`` that
    disagrees with the depth map.

    Args:
        depth_coord: receiver-frame panorama coordinates ``[B, 3, H, W]``.
        src_loc: receiver-frame query-source position ``[B, 3]``.
        ref_locs: receiver-frame reference-source positions ``[B, K, 3]``.
        k: integer column roll (negative and ``>= W`` values are reduced modulo ``W``).

    Returns:
        ``(depth_coord', src_loc', ref_locs')``; at ``k = 0`` the inputs are returned
        unchanged value-for-value.
    """
    return rotate_scene_yaw(depth_coord, src_loc, ref_locs, k, W=depth_coord.shape[-1])


def forward_conditions(model, depth_coord, ref_irs, src_loc, ref_locs, tgt_wav, k, aligned0):
    """Both experimental conditions of one batch at one angle.

    Args:
        model: an ``xRIR`` (or ``xRIR_Cyl``) in ``eval()`` mode on CUDA.
        depth_coord: **unrotated** panorama coordinates ``[B, 3, H, W]``; the rotation is
            applied here so P and E cannot accidentally see different geometry.
        ref_irs: reference RIRs ``[B, K, L]`` (never rotated -- the acoustics are invariant).
        src_loc: unrotated query-source position ``[B, 3]``.
        ref_locs: unrotated reference-source positions ``[B, K, 3]``.
        tgt_wav: target RIR ``[B, 1, L]``.
        k: integer column roll.
        aligned0: the ``k = 0`` aligned reference audio, i.e.
            ``model.shift_and_align(ref_irs, src_loc, ref_locs)`` computed once per batch;
            condition P is forced to use it at every angle.

    Returns:
        ``(out_P, out_E, tgt_spec)`` -- two ``[B, F, T, 1]`` log-magnitude spectrograms and
        the shared ``[B, 1, F, T]`` target magnitude spectrogram.  At ``k = 0`` both
        outputs equal the plain forward exactly.
    """
    depth_k, src_k, ref_locs_k = rotated_views(depth_coord, src_loc, ref_locs, k)
    with torch.no_grad():
        with fixed_alignment(model, aligned0):
            out_p, tgt_spec = model(depth_k, ref_irs, src_k, ref_locs_k, tgt_wav)
        out_e, _ = model(depth_k, ref_irs, src_k, ref_locs_k, tgt_wav)
    return out_p, out_e, tgt_spec


def spectral_metrics(out_k, out_0, tgt_spec):
    """The five per-sample spectral metrics of one batch at one angle.

    Args:
        out_k: predicted log-magnitude spectrograms at the angle, ``[B, F, T, 1]``.
        out_0: the same batch's ``k = 0`` prediction, ``[B, F, T, 1]`` -- the paired
            reference for ``consistency``.
        tgt_spec: target magnitude spectrograms ``[B, 1, F, T]``.

    Returns:
        ``{"loss", "stft", "decay", "log_mse", "consistency"}``, each a detached CPU
        float32 tensor of shape ``[B]``:

        * ``loss``/``stft``/``decay`` -- ``tools.per_sample_metrics.per_sample_losses``,
          i.e. exp_01's test loss and its two halves, computed on 1-sample slices;
        * ``log_mse`` -- the per-sample form of exp_01's
          ``Evaluator.stft_loss(out_spec.squeeze(-1), log(tgt_spec + 1e-8).squeeze(1))``;
        * ``consistency`` -- ``mean |out_k - out_0|`` over the log-spectrogram, exactly
          ``0`` at ``k = 0``.
    """
    loss, stft, decay = per_sample_losses(out_k, tgt_spec)
    with torch.no_grad():
        log_tgt = torch.log(tgt_spec[:, 0] + 1e-8)
        log_mse = ((out_k[..., 0] - log_tgt) ** 2).flatten(1).mean(dim=1)
        consistency = (out_k - out_0).abs().flatten(1).mean(dim=1)
    return {"loss": loss, "stft": stft, "decay": decay,
            "log_mse": log_mse.detach().cpu().float(),
            "consistency": consistency.detach().cpu().float()}


def acoustic_metrics_batch(out_k, tgt_wav, query_keys, evaluator, gl_seed):
    """EDT / C50 / T60 errors of one batch, sample by sample.

    The magnitude spectrogram is rebuilt exactly as ``eval_xRIR_backbone.py`` does
    (``(exp(out) - 1e-8)[..., 0]``) and inverted with a Griffin-Lim whose random phase
    is seeded from ``(gl_seed, query key)``, so the same query inverts the same
    magnitudes to the same waveform at every angle, in every model and in any angle
    order.  The 8000-sample metric window is applied inside
    ``tools.per_sample_metrics.acoustic_metrics``.

    Args:
        out_k: predicted log-magnitude spectrograms ``[B, F, T, 1]`` (CPU or CUDA).
        tgt_wav: target RIRs ``[B, 1, L]``.
        query_keys: the ``B`` query paths, in batch order.
        evaluator: an ``eval_unseen.Evaluator``.
        gl_seed: run-level Griffin-Lim seed.

    Returns:
        ``{"edt", "c50", "t60"}``, each a float64 ``np.ndarray`` of shape ``[B]`` with
        NaN where the sample is invalid at this angle.

    Raises:
        ValueError: if ``query_keys`` does not have one key per row of ``out_k``.
    """
    if len(query_keys) != out_k.shape[0]:
        raise ValueError("got {} query keys for a batch of {}".format(
            len(query_keys), out_k.shape[0]))
    values = {"edt": [], "c50": [], "t60": []}
    for i in range(out_k.shape[0]):
        mag = (torch.exp(out_k[i:i + 1]) - 1e-8)[..., 0].cpu()
        wav = griffin_lim_seeded(mag, sample_seed(gl_seed, query_keys[i]))
        sample = acoustic_metrics(wav[0].numpy(), tgt_wav[i, 0].cpu().numpy(), evaluator)
        for name in values:
            values[name].append(sample[name])
    return {name: np.asarray(vals, dtype=np.float64) for name, vals in values.items()}


def delay_flip_counts(src_loc, ref_locs, cols):
    """Count the integer direct-path delays a rotation moves, per angle.

    ``xRIR.shift_and_align`` is algebraically yaw-invariant, but its float32 norms and
    ``torch.round`` are not: a delay sitting on a rounding tie can move by one sample
    under an exactly norm-preserving rotation.  This audit quantifies that noise floor
    over the real geometry -- it is what condition P exists to exclude.

    Args:
        src_loc: query-source positions ``[B, 3]``.
        ref_locs: reference-source positions ``[B, K, 3]``.
        cols: iterable of integer column rolls.

    Returns:
        ``{k: count}`` -- the number of ``(sample, reference)`` pairs (out of ``B * K``)
        whose integer delay differs between the rotated and the unrotated coordinates.
        ``k = 0`` is always ``0``.
    """
    baseline = integer_delays(src_loc, ref_locs)
    counts = {}
    for k in cols:
        angle = yaw_angle_rad(k)
        rotated = integer_delays(rotate_vectors_z(src_loc, angle),
                                 rotate_vectors_z(ref_locs, angle))
        counts[int(k)] = int((rotated != baseline).sum())
    return counts


def _rel_change(rotated, base):
    """``mean|rotated - base| / mean|base|`` as a Python float."""
    denom = base.abs().mean()
    return float((rotated - base).abs().mean() / denom)


def decomposition_at(model, depth_coord, ref_irs, src_loc, ref_locs, tgt_wav, aligned0, k=32):
    """Where a patch-aligned yaw enters an ``xRIR_Cyl`` forward pass (diagnostic).

    CylindricalViT's tokens are equivariant to patch-aligned azimuth rolls, but ``xRIR``
    pools them with ``lin_proj_0`` -- a learned linear map over the 256 token *positions*
    -- and separately embeds the raw source xyz.  Neither is shift-invariant, so this
    reports the relative change of each stage for the receiver view, all under the
    pinned ``k = 0`` alignment (condition P):

    * ``tokens_rel_change`` -- ViT tokens after rolling the rotated ones back by
      ``k / patch_width`` azimuth patches (``~0`` iff the encoder is equivariant);
    * ``pooled_rel_change`` -- the pooled receiver feature ``lin_proj_0(src_proj(tokens))``,
      exactly as ``xRIR.forward`` computes ``receiver_out``;
    * ``coord_rel_change`` -- the sinusoidal source-coordinate embedding;
    * ``logspec_rel_change`` -- the final log-magnitude spectrogram.

    Args (beyond :func:`forward_conditions`'s):
        k: the patch-aligned angle to decompose (the plan's diagnostic angle is 32).

    Returns:
        ``{"k", "tokens_rel_change", "pooled_rel_change", "coord_rel_change",
        "logspec_rel_change"}`` -- one int and four floats.

    Raises:
        TypeError: if ``model.source_network`` is not tokenised on an explicit
            elevation x azimuth grid (i.e. is not a ``CylindricalViT``), so "roll the
            tokens back" would be undefined.
    """
    network = model.source_network
    if not (hasattr(network, "h_tok") and hasattr(network, "w_tok")):
        raise TypeError("decomposition_at needs a CylindricalViT source_network (an "
                        "explicit elevation x azimuth token grid), got {}".format(
                            type(network).__name__))
    h_tok, w_tok = int(network.h_tok), int(network.w_tok)
    width = depth_coord.shape[-1]
    shift = (int(k) % width) // (width // w_tok)

    depth_k, src_k, _ = rotated_views(depth_coord, src_loc, ref_locs, k)
    batch = depth_coord.shape[0]
    with torch.no_grad():
        tokens_0 = network((-depth_coord) / 5.)
        tokens_k = network((-depth_k) / 5.)
        grid_0 = tokens_0.view(batch, h_tok, w_tok, -1)
        grid_k = torch.roll(tokens_k.view(batch, h_tok, w_tok, -1), shifts=-shift, dims=2)

        pooled_0 = model.lin_proj_0(model.src_proj(tokens_0).permute(0, 2, 1))
        pooled_k = model.lin_proj_0(model.src_proj(tokens_k).permute(0, 2, 1))

        coord_0 = model.src_coord_proj(
            model.dist_embedder(src_loc.unsqueeze(1) / 5.).view(batch, -1))
        coord_k = model.src_coord_proj(
            model.dist_embedder(src_k.unsqueeze(1) / 5.).view(batch, -1))

    out_0 = forward_conditions(
        model, depth_coord, ref_irs, src_loc, ref_locs, tgt_wav, 0, aligned0)[0]
    out_k = forward_conditions(
        model, depth_coord, ref_irs, src_loc, ref_locs, tgt_wav, k, aligned0)[0]
    return {"k": int(k),
            "tokens_rel_change": _rel_change(grid_k, grid_0),
            "pooled_rel_change": _rel_change(pooled_k, pooled_0),
            "coord_rel_change": _rel_change(coord_k, coord_0),
            "logspec_rel_change": _rel_change(out_k, out_0)}


class SubsetManifestDataset(torch.utils.data.Dataset):
    """The first ``n`` entries of an already validated :class:`ManifestDataset`.

    ``ManifestDataset`` checks the manifest's query *set* against the whole split, which
    a truncated manifest cannot satisfy.  Rather than weaken that check (round-1 code is
    frozen), the full manifest is validated first and the resulting dataset is wrapped
    here, so a ``--max-samples`` smoke run still proves the manifest describes the real
    split.  Exposes ``dataset`` / ``manifest`` / ``entries`` like the wrapped object.

    Args:
        base: a fully validated ``ManifestDataset``.
        n: how many leading entries (in the manifest's canonical order) to keep.

    Raises:
        ValueError: if ``n`` is outside ``1 .. len(base)``.
    """

    def __init__(self, base, n):
        if not 1 <= int(n) <= len(base):
            raise ValueError("cannot take the first {} of {} entries".format(n, len(base)))
        self.base = base
        self.dataset = base.dataset
        self.manifest = base.manifest
        self.entries = base.entries[:int(n)]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        if not 0 <= int(idx) < len(self.entries):
            raise IndexError("index {} is outside the truncated manifest".format(idx))
        return self.base[int(idx)]


def build_manifest_dataset(manifest, max_samples=0, max_len=9600):
    """The evaluation dataset: the unseen test split, with the manifest's references.

    Args:
        manifest: a loaded manifest (its ``num_shot`` selects the dataset's shot count).
        max_samples: keep only the first ``n`` queries in canonical order (smoke runs
            only); ``0`` keeps the whole split.
        max_len: IR length in samples (exp_01's 9600).

    Returns:
        A ``ManifestDataset`` -- or a :class:`SubsetManifestDataset` of one -- yielding
        the exp_01 six-tuple plus the query path.
    """
    from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset

    dataset = xRIR_Dataset(split="test", max_len=max_len, num_shot=manifest["num_shot"])
    full = ManifestDataset(dataset, manifest)
    if int(max_samples) > 0:
        return SubsetManifestDataset(full, min(int(max_samples), len(full)))
    return full


def evaluate_batch(model, batch, evaluator, cols, acoustic_cols=(), e_acoustic_cols=(),
                   gl_seed=0):
    """Every per-sample metric of one collated batch, at every angle in ``cols``.

    The whole batch is moved to the GPU once and the ``k = 0`` alignment and prediction
    are computed once, so the angles share exactly one paired reference.  Nothing in the
    loop depends on the order of ``cols`` or on which angles came before: the Griffin-Lim
    phase is seeded per query, so the result is a pure function of
    ``(model, batch, cols, gl_seed)``.

    Args:
        model: an ``xRIR`` on CUDA in ``eval()`` mode.
        batch: the ``ManifestDataset`` seven-tuple, collated.
        evaluator: an ``eval_unseen.Evaluator``.
        cols: integer column rolls to evaluate (spectral metrics at every one).
        acoustic_cols: the subset of ``cols`` that also gets EDT / C50 / T60 under
            condition P; ``e_acoustic_cols`` is the same for condition E.  At ``k = 0``
            the two conditions are the same prediction, so the acoustic metrics are
            computed once and stored under both.
        gl_seed: run-level Griffin-Lim seed.

    Returns:
        ``(query_keys, results, delay_flips)`` where ``results[(condition, k)]`` maps a
        metric name to a float64 ``np.ndarray`` of shape ``[B]``, and ``delay_flips``
        maps each ``k`` to the number of flipped ``(sample, reference)`` delays.
    """
    _, src_loc, depth_coord, tgt_wav, ref_irs, ref_locs, keys = batch
    src_loc, depth_coord = src_loc.cuda(), depth_coord.cuda()
    tgt_wav, ref_irs, ref_locs = tgt_wav.cuda(), ref_irs.cuda(), ref_locs.cuda()
    keys = list(keys)
    acoustic_at = {"P": set(int(k) for k in acoustic_cols),
                   "E": set(int(k) for k in e_acoustic_cols)}
    zero_wanted = bool(acoustic_at["P"] | acoustic_at["E"]) and 0 in (
        acoustic_at["P"] | acoustic_at["E"])

    with torch.no_grad():
        aligned0 = model.shift_and_align(ref_irs, src_loc, ref_locs)
        out_0, _ = model(depth_coord, ref_irs, src_loc, ref_locs, tgt_wav)

    results = {}
    acoustic_0 = None
    for k in cols:
        out_p, out_e, tgt_spec = forward_conditions(
            model, depth_coord, ref_irs, src_loc, ref_locs, tgt_wav, k, aligned0)
        for condition, pred in (("P", out_p), ("E", out_e)):
            cell = {name: value.double().numpy()
                    for name, value in spectral_metrics(pred, out_0, tgt_spec).items()}
            if int(k) == 0 and zero_wanted:
                if acoustic_0 is None:
                    acoustic_0 = acoustic_metrics_batch(pred, tgt_wav, keys, evaluator, gl_seed)
                cell.update({name: value.copy() for name, value in acoustic_0.items()})
            elif int(k) in acoustic_at[condition]:
                cell.update(acoustic_metrics_batch(pred, tgt_wav, keys, evaluator, gl_seed))
            results[(condition, int(k))] = cell
    return keys, results, delay_flip_counts(src_loc, ref_locs, cols)


def set_precision(tf32):
    """Pin the TF32 policy of both cuBLAS and cuDNN for the whole run.

    cuDNN's ``allow_tf32`` defaults to ``True``, and the reference-RIR encoder is a
    ResNet-18, so leaving it on makes the forward pass depend on the batch size at the
    1e-4 (relative) level -- measured on 8 real queries: max ``|d|`` 2.3e-4 with TF32 on
    versus 2.4e-7 with it off.  That is far above the effects this experiment measures,
    so TF32 is off by default and the choice is recorded in the run's meta block.

    Args:
        tf32: whether to allow TF32 for matmuls and convolutions.
    """
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)


def load_checked_manifest(path, expected_hash):
    """Load the manifest at ``path`` and refuse to continue unless its hash matches.

    Every model in the sweep must condition on the *same* references, so the hash is a
    required argument of the run, not an advisory check.

    Raises:
        ValueError: if the manifest's content hash is not ``expected_hash``.
    """
    manifest = load_manifest(path)
    digest = manifest_hash(manifest)
    if digest != expected_hash:
        raise ValueError("manifest {} hashes to {}, expected {}".format(
            path, digest, expected_hash))
    return manifest


def _check_cols(yaw_cols, acoustic_cols, e_acoustic_cols):
    """Reduce the three angle grids and reject a grid the paired design cannot use."""
    cols = [int(k) for k in yaw_cols]
    if 0 not in cols:
        raise ValueError("k = 0 is the paired reference of every metric; --yaw-cols must "
                         "contain it, got {}".format(cols))
    if len(set(cols)) != len(cols):
        raise ValueError("--yaw-cols contains a repeated angle: {}".format(cols))
    acoustic = [int(k) for k in acoustic_cols]
    e_acoustic = [int(k) for k in e_acoustic_cols]
    for name, subset in (("--acoustic-cols", acoustic), ("--e-acoustic-cols", e_acoustic)):
        extra = sorted(set(subset) - set(cols))
        if extra:
            raise ValueError("{} {} are not in --yaw-cols".format(name, extra))
    return cols, acoustic, e_acoustic


def _summarize(values):
    """``{"mean", "n_valid", "n_nan"}`` of one per-sample array (mean over finite values)."""
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    return {"mean": float(array[finite].mean()) if finite.any() else None,
            "n_valid": int(finite.sum()), "n_nan": int((~finite).sum())}


def run(args):
    """Evaluate one checkpoint over the pinned manifest at every angle of the grid.

    Writes ``<out-dir>/per_sample_yaw.json`` (every per-sample value, keyed by condition
    and angle, plus the query order and the run's meta block) and
    ``<out-dir>/metrics_yaw.json`` (their means and validity counts).  NaN marks an
    invalid sample and is written as the JSON ``NaN`` literal, which Python's ``json``
    reads back as ``float("nan")`` -- ``tools/summarize_yaw.py`` relies on that.

    Args:
        args: the parsed CLI namespace (see :func:`main`).

    Returns:
        The per-sample dict that was written.

    Raises:
        RuntimeError: if no GPU is visible (``model.xRIR.apply_delay`` hard-codes ``.cuda()``).
        ValueError: if the manifest hash does not match, the angle grids are inconsistent,
            or the loader does not return the manifest's queries in canonical order.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("eval_yaw_rotation needs a GPU: xRIR.apply_delay is .cuda()-only")
    started = time.time()
    torch.set_num_threads(args.threads)
    set_precision(args.tf32)

    # The hash gate comes first: a run on the wrong references is worse than no run.
    manifest = load_checked_manifest(args.manifest, args.manifest_hash)
    cols, acoustic_cols, e_acoustic_cols = _check_cols(
        args.yaw_cols, args.acoustic_cols, args.e_acoustic_cols)

    dataset = build_manifest_dataset(manifest, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    model = build_xrir(args.backbone, manifest["num_shot"])
    model.load_state_dict(load_model_state(args.checkpoint), strict=True)
    model.cuda().eval()
    evaluator = Evaluator()
    n_samples = len(dataset)
    print("backbone: {}  checkpoint: {}  samples: {}  angles: {}  batch: {}".format(
        args.backbone, args.checkpoint, n_samples, cols, args.batch_size), flush=True)

    queries, parts, flips = [], {}, {}
    for i, batch in enumerate(loader):
        keys, results, batch_flips = evaluate_batch(
            model, batch, evaluator, cols, acoustic_cols, e_acoustic_cols, args.gl_seed)
        queries.extend(keys)
        for cell, metrics in results.items():
            for metric, value in metrics.items():
                parts.setdefault(cell, {}).setdefault(metric, []).append(value)
        for k, count in batch_flips.items():
            flips[k] = flips.get(k, 0) + count
        if (i + 1) % args.log_interval == 0 or len(queries) == n_samples:
            elapsed = time.time() - started
            rate = len(queries) / elapsed
            print("[{}/{}] {:.2f} samples/s, eta {:.1f} min".format(
                len(queries), n_samples, rate, (n_samples - len(queries)) / rate / 60.0),
                flush=True)

    entries = dataset.entries
    if queries != [entry["query"] for entry in entries]:
        raise ValueError("the loader did not return the manifest's canonical order")
    merged = {cell: {metric: np.concatenate(chunks) for metric, chunks in metrics.items()}
              for cell, metrics in parts.items()}
    decomposition = None

    meta = {"backbone": args.backbone, "checkpoint": args.checkpoint,
            "manifest_path": args.manifest, "manifest_hash": args.manifest_hash,
            "manifest_seed": manifest["seed"], "gl_seed": args.gl_seed, "yaw_cols": cols,
            "acoustic_cols": acoustic_cols, "e_acoustic_cols": e_acoustic_cols,
            "batch_size": args.batch_size, "n_samples": n_samples,
            "max_samples": int(args.max_samples), "tf32": bool(args.tf32),
            "torch_version": torch.__version__,
            "elapsed_min": (time.time() - started) / 60.0}
    per_sample = {"meta": meta, "query": queries,
                  "index": [entry["index"] for entry in entries],
                  "delay_flips": {str(k): int(v) for k, v in sorted(flips.items())},
                  "decomposition": decomposition}
    metrics_out = {"meta": meta, "delay_flips": per_sample["delay_flips"],
                   "decomposition": decomposition}
    for condition in ("P", "E"):
        per_sample[condition] = {
            str(k): {metric: [float(v) for v in values]
                     for metric, values in merged[(condition, k)].items()}
            for k in cols}
        metrics_out[condition] = {
            str(k): {metric: _summarize(values)
                     for metric, values in merged[(condition, k)].items()}
            for k in cols}

    os.makedirs(args.out_dir, exist_ok=True)
    for name, payload in (("per_sample_yaw.json", per_sample),
                          ("metrics_yaw.json", metrics_out)):
        with open(os.path.join(args.out_dir, name), "w") as fout:
            json.dump(payload, fout)          # allow_nan: NaN marks an invalid sample
        print("wrote {}".format(os.path.join(args.out_dir, name)), flush=True)
    print("done: {} samples x {} angles x 2 conditions in {:.2f} min".format(
        n_samples, len(cols), meta["elapsed_min"]), flush=True)
    return per_sample


SPECTRAL_COLS = [0, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 416, 448, 480, 496,
                 504, 508]
ACOUSTIC_COLS = [0, 8, 32, 64, 128, 256, 384, 448, 480, 504]
E_ACOUSTIC_COLS = [32, 128, 384, 480]


def main(argv=None):
    """Parse the CLI and run the sweep (see the module docstring for an example)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-hash", required=True,
                        help="sha256 of the manifest's content; the run refuses to start "
                             "unless it matches, so every model sees the same references")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--yaw-cols", type=int, nargs="*", default=SPECTRAL_COLS,
                        help="column rolls for the spectral metrics (W=512); must contain 0")
    parser.add_argument("--acoustic-cols", type=int, nargs="*", default=ACOUSTIC_COLS,
                        help="subset of --yaw-cols that also gets EDT/C50/T60 under P")
    parser.add_argument("--e-acoustic-cols", type=int, nargs="*", default=E_ACOUSTIC_COLS,
                        help="the same for the end-to-end condition E")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="smoke runs only: keep the first N manifest queries")
    parser.add_argument("--gl-seed", type=int, default=0,
                        help="run-level Griffin-Lim seed (combined with each query path)")
    parser.add_argument("--threads", type=int, default=4,
                        help="torch CPU threads (Griffin-Lim runs on CPU)")
    parser.add_argument("--tf32", action="store_true",
                        help="allow TF32; off by default because cuDNN's TF32 makes the "
                             "reference encoder batch-size dependent at ~1e-4")
    parser.add_argument("--log-interval", type=int, default=50)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
