"""Confirmatory tables for the yaw-rotation degradation experiment (exp_03).

Reads the ``per_sample_yaw.json`` written by ``eval_yaw_rotation.py`` for each model and
turns the per-sample arrays into the pre-registered analysis: the relative degradation
``r_k = (mean(e_k) - mean(e_0)) / mean(e_0)`` per angle with Bonferroni-adjusted paired
bootstrap intervals, the H1 verdict, the cylindrical-vs-control difference in
differences (H2) with equivalence at the patch-aligned angles, the properly paired
``k = 0`` re-evaluation of exp_01, the delay-flip audit and the decomposition.

Every number that is printed is also written to ``--json``, together with the seeds, the
family size and the sha256 of the printed text, so a results page can bind the two.

    python tools/summarize_yaw.py --runs ckpt/yaw_rotation/{control,cyl,released} \\
        --manifest-hash <sha256> --json ckpt/yaw_rotation/summary.json \\
        --summary ckpt/yaw_rotation/summary.txt
"""
from __future__ import annotations

import json
import os

import numpy as np

from tools.paired_stats import (
    diff_in_diff_bootstrap,
    equivalence_tost,
    paired_validity,
    relative_degradation_bootstrap,
    valid_mask,
)

WIDTH = 512
CONVERGENCE_LIMIT = 0.10


def _write_text(text, path):
    """Write ``text`` atomically (temporary file + rename), leaving no partial file."""
    tmp = path + ".tmp"
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        with open(tmp, "w") as fout:
            fout.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _write_json(payload, path):
    """Write ``payload`` as JSON atomically; a reader never sees a half-written summary."""
    tmp = path + ".tmp"
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        with open(tmp, "w") as fout:
            json.dump(payload, fout, indent=1)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_run(directory):
    """Read one evaluator output directory.

    Args:
        directory: a directory containing ``per_sample_yaw.json``.

    Returns:
        The run dict, with ``"dir"`` added.

    Raises:
        IOError: if the directory holds no ``per_sample_yaw.json``.
    """
    path = os.path.join(directory, "per_sample_yaw.json")
    if not os.path.exists(path):
        raise IOError("no per_sample_yaw.json in {}".format(directory))
    with open(path, "r") as fin:
        run = json.load(fin)                  # NaN literals load as float("nan")
    run["dir"] = directory
    return run


def _as_array(values):
    """One per-sample list as a float64 array, reading strict JSON's ``null`` as NaN."""
    return np.array([np.nan if value is None else float(value) for value in values],
                    dtype=np.float64)


def rooms_from_paths(queries):
    """The room of each query: the second component of its path.

    ``Cafe/Cafe_idx_1/S001_R002_hybrid_IR.wav`` -> ``Cafe_idx_1``.  The room is the
    cluster of the secondary, population-of-rooms bootstrap.
    """
    return np.array([str(query).split("/")[1] for query in queries])


def signed_degrees(k, width=WIDTH):
    """The angle of a column roll in degrees, mapped to ``(-180, 180]``."""
    degrees = 360.0 * (int(k) % int(width)) / float(width)
    return degrees - 360.0 if degrees > 180.0 else degrees


def _ratio_or_none(e0, ek, n_boot, alpha, seed, clusters):
    """:func:`relative_degradation_bootstrap`, or ``None`` when the ratio does not exist.

    ``mean(e_0) == 0`` (``consistency``, whose baseline is identically zero) and an empty
    valid set are reported as "undefined", not as an exception: the row still carries the
    means and the validity split, which is what those metrics are read for.
    """
    try:
        return relative_degradation_bootstrap(
            e0, ek, n_boot=int(n_boot), alpha=float(alpha), seed=int(seed),
            clusters=clusters)
    except ValueError:
        return None


def degradation_rows(run, condition, metric, cols, n_boot, alpha_adj, seed, rooms=None):
    """One relative-degradation row per angle, paired against ``k = 0``.

    The pre-registered validity rule is applied first (a sample enters only if the metric
    is finite at ``k = 0`` *and* at the compared angle), and the counts of what the
    rotation invalidated are reported alongside, never imputed.

    Args:
        run: a loaded run (see :func:`load_run`).
        condition: ``"P"`` (pinned alignment) or ``"E"`` (end to end).
        metric: the per-sample metric name.
        cols: the angles to report, in the order they should be printed.
        n_boot: bootstrap resamples.
        alpha_adj: the *adjusted* two-sided level (``tools.paired_stats.bonferroni_alpha``).
        seed: bootstrap seed.
        rooms: optional room id per query; when given, each row also carries the
            room-cluster interval ``r_lo`` / ``r_hi``.

    Returns:
        A list of dicts ``{k, deg, mean0, meank, r, lo, hi, n_valid, validity}`` (plus
        ``r_lo``/``r_hi`` with ``rooms``).  ``r``/``lo``/``hi`` are ``None`` when the
        ratio does not exist -- ``mean(e_0) == 0``, which is the case for
        ``consistency``, whose baseline is identically zero.
    """
    e0 = _as_array(run[condition]["0"][metric])
    rooms = None if rooms is None else np.asarray(rooms)
    rows = []
    for k in cols:
        ek = _as_array(run[condition][str(int(k))][metric])
        mask = valid_mask(e0, ek)
        row = {"k": int(k), "deg": signed_degrees(k), "n_valid": int(mask.sum()),
               "validity": paired_validity(e0, ek),
               "mean0": float(e0[mask].mean()) if mask.any() else None,
               "meank": float(ek[mask].mean()) if mask.any() else None,
               "r": None, "lo": None, "hi": None}
        query_level = _ratio_or_none(e0[mask], ek[mask], n_boot, alpha_adj, seed, None)
        if query_level is not None:
            row.update({key: query_level[key] for key in ("r", "lo", "hi")})
        if rooms is not None:
            cluster = _ratio_or_none(e0[mask], ek[mask], n_boot, alpha_adj, seed, rooms[mask])
            row["r_lo"] = None if cluster is None else cluster["lo"]
            row["r_hi"] = None if cluster is None else cluster["hi"]
        rows.append(row)
    return rows


def h1_verdict(rows_by_metric, threshold):
    """The pre-registered H1 rule: does any acoustic angle degrade beyond the margin?

    H1 is supported if, for at least one angle ``k != 0``, the Bonferroni-adjusted
    *lower* bound of ``r_k`` exceeds ``threshold``.  The rule is one-sided and strict, so
    a lower bound sitting exactly on the margin is not a pass, and rows whose ratio does
    not exist are simply not eligible.

    Args:
        rows_by_metric: ``{metric: rows}`` from :func:`degradation_rows`; the confirmatory
            family is EDT and C50 over the non-zero acoustic angles, and the caller passes
            exactly those.
        threshold: the practical margin on ``r`` (the plan's +10 % is ``0.10``).

    Returns:
        ``(passes, passing_cells)`` where each cell is
        ``{metric, k, deg, r, lo, hi}`` -- the evidence behind the verdict.
    """
    passing = []
    for metric in sorted(rows_by_metric):
        for row in rows_by_metric[metric]:
            if int(row["k"]) == 0 or row.get("lo") is None:
                continue
            if row["lo"] > float(threshold):
                passing.append({"metric": metric, "k": int(row["k"]), "deg": row["deg"],
                                "r": row["r"], "lo": row["lo"], "hi": row["hi"]})
    return bool(passing), passing


def h1_wording(primary_pass, released_pass):
    """The verdict wording for the same-budget control and the released checkpoint.

    The two models are reported separately and never pooled: the control is the
    confirmatory model (training-matched to the cylindrical one) and the released
    checkpoint is a replication, so a disagreement is stated as such.
    """
    if primary_pass and released_pass:
        return "supported and replicated"
    if primary_pass:
        return "supported for the same-budget model only"
    if released_pass:
        return "replication only"
    return "not supported"


def h2_rows(run_cyl, run_ctrl, condition, metric, cols, n_boot, alpha_adj, seed,
            rooms=None, equiv_margin=0.02, patch_width=32):
    """H2: does the cylindrical backbone degrade less than the same-budget control?

    The statistic is the difference in relative degradations ``D_k = r_cyl - r_ctrl``,
    with both ratios recomputed on the *same* bootstrap resamples, so the interval
    reflects the cross-model difference and not the query noise the two models share.
    At the patch-aligned angles -- where CylindricalViT's tokens are equivariant by
    construction -- the plan claims "nearly unchanged", which is an equivalence claim,
    so those angles also get a TOST on the cylindrical model's own ``r_k``.

    Args:
        run_cyl, run_ctrl: the two runs, evaluated on the same queries in the same order.
        condition: ``"P"`` or ``"E"``.
        metric: the per-sample metric name.
        cols: the angles to report.
        n_boot: bootstrap resamples.
        alpha_adj: the adjusted level (two-sided for ``D_k``, per one-sided test for TOST).
        seed: bootstrap seed.
        rooms: optional room id per query, adding the cluster intervals.
        equiv_margin: the equivalence margin on ``r`` (the plan's +-2 %).
        patch_width: columns per azimuth patch; ``k`` is patch-aligned iff it is a
            non-zero multiple of it.

    Returns:
        A list of dicts ``{k, deg, d, lo, hi, c_lo, c_hi, r_cyl, r_ctrl, n_valid,
        equivalence}``; ``equivalence`` is ``None`` unless the angle is patch-aligned,
        and otherwise ``{"margin", "n", "query", "cluster"}`` with the two TOST results,
        computed on the cylindrical run's own validity mask (``n``), not on the narrower
        four-way mask the difference in differences needs (``n_valid``).

    Raises:
        ValueError: if the two runs are not query-aligned.
    """
    if run_cyl["query"] != run_ctrl["query"]:
        raise ValueError("the two runs are not query-aligned: D_k would not be paired")
    a0 = _as_array(run_cyl[condition]["0"][metric])
    b0 = _as_array(run_ctrl[condition]["0"][metric])
    rooms = None if rooms is None else np.asarray(rooms)
    rows = []
    for k in cols:
        ak = _as_array(run_cyl[condition][str(int(k))][metric])
        bk = _as_array(run_ctrl[condition][str(int(k))][metric])
        cyl_mask = valid_mask(a0, ak)
        mask = cyl_mask & valid_mask(b0, bk)
        clusters = None if rooms is None else rooms[mask]
        row = {"k": int(k), "deg": signed_degrees(k), "n_valid": int(mask.sum()),
               "d": None, "lo": None, "hi": None, "c_lo": None, "c_hi": None,
               "r_cyl": None, "r_ctrl": None, "equivalence": None}
        try:
            res = diff_in_diff_bootstrap(a0[mask], ak[mask], b0[mask], bk[mask],
                                         n_boot=int(n_boot), alpha=float(alpha_adj),
                                         seed=int(seed))
            row.update({"d": res["d"], "lo": res["lo"], "hi": res["hi"],
                        "r_cyl": res["r_a"], "r_ctrl": res["r_b"]})
            if clusters is not None:
                cluster = diff_in_diff_bootstrap(
                    a0[mask], ak[mask], b0[mask], bk[mask], n_boot=int(n_boot),
                    alpha=float(alpha_adj), seed=int(seed), clusters=clusters)
                row.update({"c_lo": cluster["lo"], "c_hi": cluster["hi"]})
        except ValueError:
            pass
        if int(k) != 0 and int(k) % int(patch_width) == 0:
            # The equivalence claim is about the cylindrical model alone, so it is read
            # through *its* validity mask; the control's invalid samples must not decide
            # which of its queries count.
            cyl_clusters = None if rooms is None else rooms[cyl_mask]
            row["equivalence"] = {"margin": float(equiv_margin), "query": None,
                                  "cluster": None, "n": int(cyl_mask.sum())}
            for name, ids in (("query", None), ("cluster", cyl_clusters)):
                if name == "cluster" and rooms is None:
                    continue
                try:
                    row["equivalence"][name] = equivalence_tost(
                        a0[cyl_mask], ak[cyl_mask], margin=float(equiv_margin),
                        n_boot=int(n_boot), alpha=float(alpha_adj), seed=int(seed),
                        clusters=ids)
                except ValueError:
                    row["equivalence"][name] = None
        rows.append(row)
    return rows


# Resamples are drawn in chunks of at most this many cells (as in tools.paired_stats),
# so a 20000 x 6337 bootstrap never materialises a 1 GB index matrix.
_CHUNK_CELLS = 2000000


def _abs_diff_bootstrap(d, n_boot, alpha, seed, clusters=None):
    """Percentile bootstrap interval of ``mean(d)`` (the *absolute* paired difference).

    ``tools.paired_stats`` only offers the relative form, and the ``k = 0`` table reports
    absolute differences in the metric's own units, exactly as exp_01's tables did.
    ``clusters`` resamples whole clusters with multiplicity weights, the same convention
    as ``paired_stats._bootstrap_ratios``.

    Args:
        d: per-pair differences (finite, non-empty).
        n_boot: resamples.
        alpha: two-sided level of the ``1 - alpha`` interval.
        seed: bootstrap seed.
        clusters: optional cluster id per pair.

    Returns:
        ``(mean, lo, hi)`` as Python floats.
    """
    d = np.asarray(d, dtype=np.float64)
    if d.size == 0:
        raise ValueError("no valid pairs to bootstrap")
    rng = np.random.default_rng(seed)
    boots = np.empty(int(n_boot), dtype=np.float64)
    done = 0
    if clusters is None:
        n = d.size
        chunk = max(1, min(int(n_boot), _CHUNK_CELLS // max(n, 1)))
        while done < int(n_boot):
            size = min(chunk, int(n_boot) - done)
            boots[done:done + size] = d[rng.integers(0, n, size=(size, n))].mean(axis=1)
            done += size
    else:
        _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
        n_cl = int(inverse.max()) + 1
        counts = np.bincount(inverse, minlength=n_cl).astype(np.float64)
        sums = np.bincount(inverse, weights=d, minlength=n_cl)
        chunk = max(1, min(int(n_boot), _CHUNK_CELLS // max(n_cl, 1)))
        while done < int(n_boot):
            size = min(chunk, int(n_boot) - done)
            draws = rng.integers(0, n_cl, size=(size, n_cl))
            weights = np.zeros((size, n_cl), dtype=np.float64)
            np.add.at(weights, (np.repeat(np.arange(size), n_cl), draws.ravel()), 1.0)
            boots[done:done + size] = (weights @ sums) / (weights @ counts)
            done += size
    lo, hi = np.percentile(boots, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(d.mean()), float(lo), float(hi)


def k0_comparison(run_cyl, run_ctrl, metrics, n_boot, alpha, seed, rooms=None):
    """The paired cylindrical-vs-control comparison at ``k = 0``.

    This is the properly paired re-evaluation of exp_01: both models are read from the
    *same* reference manifest with per-query seeded Griffin-Lim phases, so the difference
    is a genuine paired contrast rather than the reference-draw noise exp_01 measured.
    Reported at the nominal level, like exp_01's tables -- it is not part of the
    confirmatory family.

    Args:
        run_cyl, run_ctrl: the two runs, query-aligned.
        metrics: metric names to report, in print order.
        n_boot: bootstrap resamples.
        alpha: two-sided level (nominal 0.05, unadjusted).
        seed: bootstrap seed.
        rooms: optional room id per query, adding the room-cluster interval.

    Returns:
        One dict per metric: ``{metric, cyl, ctrl, diff, rel_pct, q_lo, q_hi, r_lo, r_hi,
        n_valid}``.

    Raises:
        ValueError: if the two runs are not query-aligned.
    """
    if run_cyl["query"] != run_ctrl["query"]:
        raise ValueError("the two runs are not query-aligned: the k=0 contrast would "
                         "not be paired")
    rooms = None if rooms is None else np.asarray(rooms)
    rows = []
    for metric in metrics:
        a = _as_array(run_cyl["P"]["0"][metric])
        b = _as_array(run_ctrl["P"]["0"][metric])
        mask = np.isfinite(a) & np.isfinite(b)
        diff, q_lo, q_hi = _abs_diff_bootstrap(a[mask] - b[mask], n_boot, alpha, seed)
        row = {"metric": metric, "cyl": float(a[mask].mean()), "ctrl": float(b[mask].mean()),
               "diff": diff, "rel_pct": float(100.0 * diff / b[mask].mean()),
               "q_lo": q_lo, "q_hi": q_hi, "r_lo": None, "r_hi": None,
               "n_valid": int(mask.sum())}
        if rooms is not None:
            _, r_lo, r_hi = _abs_diff_bootstrap(a[mask] - b[mask], n_boot, alpha, seed,
                                                clusters=rooms[mask])
            row.update({"r_lo": r_lo, "r_hi": r_hi})
        rows.append(row)
    return rows


def convergence_cell(fn, seed_a, seed_b):
    """Both seeds' endpoints of one decision-driving bound, and how far it moved.

    Args:
        fn: ``seed -> {"lo": ..., "hi": ...}`` (or ``None`` when the bound does not exist).
        seed_a, seed_b: the two bootstrap seeds.

    Returns:
        ``{"seed_a", "seed_b", "width", "max_change", "rel_change"}``; ``rel_change`` is
        ``0.0`` for a degenerate interval that did not move and ``inf`` for one that did.
        A missing bound yields all-``None`` endpoints and ``rel_change`` ``0.0``: there is
        no bound to be unconverged about.
    """
    first, second = fn(seed_a), fn(seed_b)
    if first is None or second is None:
        return {"seed_a": {"lo": None, "hi": None}, "seed_b": {"lo": None, "hi": None},
                "width": None, "max_change": None, "rel_change": 0.0}
    width = float(first["hi"]) - float(first["lo"])
    change = max(abs(float(first["lo"]) - float(second["lo"])),
                 abs(float(first["hi"]) - float(second["hi"])))
    if width <= 0.0:
        rel = 0.0 if change == 0.0 else float("inf")
    else:
        rel = change / width
    return {"seed_a": {"lo": float(first["lo"]), "hi": float(first["hi"])},
            "seed_b": {"lo": float(second["lo"]), "hi": float(second["hi"])},
            "width": width, "max_change": change, "rel_change": rel}


def convergence_check(fn, seed_a, seed_b):
    """How far a confirmatory interval moves when only the bootstrap seed changes.

    A percentile interval is itself a Monte-Carlo estimate; if ``n_boot`` is too small
    the reported bound is noise, and a pre-registered threshold on a noisy bound is not
    a decision rule.  Recomputing with a second seed and expressing the movement as a
    fraction of the interval's own width makes that visible.

    Args:
        fn: ``seed -> {"lo": ..., "hi": ...}``.
        seed_a, seed_b: the two bootstrap seeds.

    Returns:
        ``max(|dlo|, |dhi|) / width``; ``0.0`` if the interval is degenerate and did not
        move (perfectly proportional data), ``inf`` if it is degenerate and did move.
    """
    return convergence_cell(fn, seed_a, seed_b)["rel_change"]


def h2_verdict(h1_cells, h2_by_metric):
    """The pre-registered H2 rule, decided only where H1 actually found a degradation.

    H2 claims the cylindrical backbone degrades *less*.  That claim is only meaningful
    where there is a degradation to compare, so it is decided cell by cell at the
    (metric, angle) pairs whose H1 lower bound cleared the margin: a cell passes when the
    difference in differences is negative and its Bonferroni-adjusted **upper** bound is
    below zero.

    Args:
        h1_cells: the passing cells from :func:`h1_verdict` (the *primary* model's).
        h2_by_metric: ``{metric: rows}`` from :func:`h2_rows`.

    Returns:
        ``{"aggregate", "cells", "n_cells", "n_passing", "passing", "failing"}``.
        ``aggregate`` is "supported" (every cell), "partially supported" (some),
        "not supported" (none) or "not evaluable (H1 has no passing cell)".
    """
    rows = {(metric, int(row["k"])): row
            for metric, metric_rows in h2_by_metric.items() for row in metric_rows}
    cells = []
    for cell in h1_cells:
        row = rows.get((cell["metric"], int(cell["k"])))
        decided = row is not None and row["d"] is not None and row["hi"] is not None
        cells.append({"metric": cell["metric"], "k": int(cell["k"]), "deg": cell["deg"],
                      "d": None if row is None else row["d"],
                      "lo": None if row is None else row["lo"],
                      "hi": None if row is None else row["hi"],
                      "c_lo": None if row is None else row.get("c_lo"),
                      "c_hi": None if row is None else row.get("c_hi"),
                      "passes": bool(decided and row["d"] < 0.0 and row["hi"] < 0.0)})
    label = lambda cell: "{}@{}".format(cell["metric"], cell["k"])
    passing = [label(cell) for cell in cells if cell["passes"]]
    failing = [label(cell) for cell in cells if not cell["passes"]]
    if not cells:
        aggregate = "not evaluable (H1 has no passing cell)"
    elif not failing:
        aggregate = "supported"
    elif passing:
        aggregate = "partially supported"
    else:
        aggregate = "not supported"
    return {"aggregate": aggregate, "cells": cells, "n_cells": len(cells),
            "n_passing": len(passing), "passing": passing, "failing": failing}


SPECTRAL_METRICS = ("loss", "log_mse", "consistency")
ACOUSTIC_METRICS = ("edt", "c50", "t60")
CONFIRMATORY_METRICS = ("edt", "c50")
K0_METRICS = ("edt", "c50", "t60", "log_mse", "loss")

# The pre-registered angle grids (plan section 4). They are constants, not something a
# run's own meta may redefine: the Bonferroni family is fixed at 2 metrics x 9 non-zero
# acoustic angles, so a run on a partial grid can never buy itself a laxer correction.
PREREGISTERED_SPECTRAL_COLS = (0, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 416,
                               448, 480, 496, 504, 508)
PREREGISTERED_ACOUSTIC_COLS = (0, 8, 32, 64, 128, 256, 384, 448, 480, 504)
PREREGISTERED_E_ACOUSTIC_COLS = (32, 128, 384, 480)
FAMILY_SIZE = len(CONFIRMATORY_METRICS) * len(
    [k for k in PREREGISTERED_ACOUSTIC_COLS if k != 0])

# What a confirmatory ("full") summary is allowed to run on. Production never overrides
# these; a run that does not meet them is exploratory by definition.
FULL_EXPECTATIONS = {"n_queries": 6337, "n_rooms": 17, "gl_seed": 0, "tf32": False,
                     "batch_canonical": True,
                     "spectral_cols": PREREGISTERED_SPECTRAL_COLS,
                     "acoustic_cols": PREREGISTERED_ACOUSTIC_COLS,
                     "e_acoustic_cols": PREREGISTERED_E_ACOUSTIC_COLS}
EXPECTED_ROLE_CHECKPOINTS = {"primary": "xRIR_simple_8_shot", "cyl": "xRIR_cyl_8_shot",
                             "released": "checkpoints/xRIR_unseen.pth"}


K0_GATE_METRICS = ("edt", "c50", "t60", "loss")
RELEASED_BASELINE_METRICS = ("edt", "c50", "t60")


def _finite_mean(values):
    """Mean over the finite entries of one per-sample array (``None`` if there are none)."""
    array = _as_array(values)
    finite = np.isfinite(array)
    return float(array[finite].mean()) if finite.any() else None


def k0_gate_rows(by_label, roles, exp01_by_label, noise_runs, released_baseline,
                 metrics=K0_GATE_METRICS):
    """The k=0 parity gate: do the pinned-manifest runs reproduce exp_01's numbers?

    The manifest pins a *different* reference draw than exp_01's (which was never
    recorded), so the two cannot agree per sample -- only distributionally.  The width of
    "distributionally" is measured, not assumed: re-evaluating the control checkpoint on
    two further manifest seeds gives the reference-draw spread, and a model passes when
    its shift from exp_01 stays inside twice that spread.  The released checkpoint has no
    per-sample record, so it is compared with exp_01's reported baseline instead.

    Args:
        by_label: ``{label: run}`` -- k=0-only runs.
        roles: ``{"primary", "cyl", "released"} -> label or None``.
        exp01_by_label: ``{label: exp_01 per-sample dict}`` (eval_xRIR_backbone's layout).
        noise_runs: further k=0 runs of the *control* checkpoint on other manifest seeds.
        released_baseline: ``{"edt", "c50", "t60"}`` from exp_01's baseline reproduction,
            or ``None``.
        metrics: the metrics to gate on.

    Returns:
        ``(rows, gate_pass, reasons)``.  Each row is ``{label, role, metric, mean_k0,
        reference, source, spread, abs_diff, band, pass}``; ``pass`` is ``None`` where no
        reference exists (the released checkpoint's loss), and such rows do not decide
        the gate.
    """
    reasons = []
    control_label = roles.get("primary")
    spreads = {}
    for metric in metrics:
        control_mean = (None if control_label is None else
                        _finite_mean(by_label[control_label]["P"]["0"][metric]))
        deltas = [abs(_finite_mean(run["P"]["0"][metric]) - control_mean)
                  for run in noise_runs
                  if control_mean is not None
                  and _finite_mean(run["P"]["0"][metric]) is not None]
        spreads[metric] = max(deltas) if deltas else None
    if not noise_runs:
        reasons.append("no --noise-runs given: the reference-draw spread is unmeasured")

    rows = []
    for role in ("primary", "cyl", "released"):
        label = roles.get(role)
        if label is None or label not in by_label:
            continue
        run = by_label[label]
        reference_source = "exp_01 per-sample"
        exp01 = exp01_by_label.get(label)
        if role != "released" and exp01 is None:
            reasons.append("{}: no exp_01 per-sample file given".format(label))
        for metric in metrics:
            mean_k0 = _finite_mean(run["P"]["0"][metric])
            if role == "released":
                reference_source = "exp_01 baseline reproduction"
                reference = (None if released_baseline is None
                             else released_baseline.get(metric))
            else:
                reference = (None if exp01 is None or metric not in exp01
                             else _finite_mean(exp01[metric]))
            spread = spreads.get(metric)
            band = None if spread is None else 2.0 * spread + 1e-6
            difference = (None if reference is None or mean_k0 is None
                          else abs(mean_k0 - reference))
            passes = (None if difference is None or band is None
                      else bool(difference <= band))
            rows.append({"label": label, "role": role, "metric": metric,
                         "mean_k0": mean_k0, "reference": reference,
                         "source": reference_source, "spread": spread,
                         "abs_diff": difference, "band": band, "pass": passes})
            if passes is False:
                reasons.append("{} {}: |{:.6g} - {:.6g}| = {:.6g} > band {:.6g}".format(
                    label, metric, mean_k0, reference, difference, band))
            elif passes is None and reference is not None:
                reasons.append("{} {}: no band (the spread is unmeasured)".format(
                    label, metric))
    decided = [row["pass"] for row in rows if row["pass"] is not None]
    gate_pass = bool(decided) and all(decided) and not reasons
    return rows, gate_pass, reasons


def validate_full(by_label, roles, expected_hash):
    """Every condition the confirmatory analysis assumes, checked before it runs.

    A pre-registered decision rule is only worth anything if the evidence it reads is the
    evidence it was written for, so this refuses to *silently* summarise a partial sweep:
    it returns the reasons rather than raising, and the caller reports them and stops.

    Args:
        by_label: ``{label: run}``.
        roles: ``{"primary", "cyl", "released"} -> label or None``.
        expected_hash: the manifest hash the runs must carry (required in full mode).

    Returns:
        A list of human-readable reasons; empty means the set is confirmatory-grade.
    """
    expectations = FULL_EXPECTATIONS
    reasons = []
    if not expected_hash:
        reasons.append("--manifest-hash is required in full mode")
    for role in ("primary", "cyl", "released"):
        label = roles.get(role)
        needle = EXPECTED_ROLE_CHECKPOINTS[role]
        if label is None:
            reasons.append("no {} run".format(role))
        elif needle not in by_label[label]["meta"]["checkpoint"]:
            reasons.append("{} run {!r} has checkpoint {!r}, expected one containing {!r}"
                           .format(role, label, by_label[label]["meta"]["checkpoint"],
                                   needle))
    reference = None
    for label in sorted(by_label):
        run = by_label[label]
        meta = run["meta"]
        queries = run["query"]
        if len(queries) != expectations["n_queries"]:
            reasons.append("{}: {} queries, expected {}".format(
                label, len(queries), expectations["n_queries"]))
        if len(set(queries)) != len(queries):
            reasons.append("{}: the queries are not unique ({} of {})".format(
                label, len(set(queries)), len(queries)))
        n_rooms = int(len(np.unique(rooms_from_paths(queries))))
        if n_rooms != expectations["n_rooms"]:
            reasons.append("{}: {} rooms, expected {}".format(
                label, n_rooms, expectations["n_rooms"]))
        for key in ("gl_seed", "tf32", "batch_canonical"):
            if meta.get(key) != expectations[key]:
                reasons.append("{}: meta.{} is {!r}, expected {!r}".format(
                    label, key, meta.get(key), expectations[key]))
        for key, expected in (("yaw_cols", expectations["spectral_cols"]),
                              ("acoustic_cols", expectations["acoustic_cols"]),
                              ("e_acoustic_cols", expectations["e_acoustic_cols"])):
            if sorted(int(k) for k in meta.get(key, [])) != sorted(int(k) for k in expected):
                reasons.append("{}: meta.{} is {}, expected exactly {}".format(
                    label, key, sorted(meta.get(key, [])), sorted(expected)))
        for k in expectations["spectral_cols"]:
            if str(int(k)) not in run.get("delay_flips", {}):
                reasons.append("{}: delay_flips has no angle {}".format(label, k))
            for condition in ("P", "E"):
                cell = run[condition].get(str(int(k)))
                if cell is None:
                    reasons.append("{}: condition {} has no angle {}".format(
                        label, condition, k))
                    continue
                for metric in SPECTRAL_METRICS:
                    values = cell.get(metric)
                    if values is None or len(values) != expectations["n_queries"]:
                        reasons.append("{}: {} k={} {} has {} values, expected {}".format(
                            label, condition, k, metric,
                            "no" if values is None else len(values),
                            expectations["n_queries"]))
                    elif not np.isfinite(_as_array(values)).all():
                        reasons.append("{}: {} k={} {} has non-finite values".format(
                            label, condition, k, metric))
        for condition, angles in (("P", expectations["acoustic_cols"]),
                                  ("E", (0,) + tuple(expectations["e_acoustic_cols"]))):
            for k in angles:
                cell = run[condition].get(str(int(k)), {})
                for metric in ACOUSTIC_METRICS:
                    values = cell.get(metric)
                    if values is None or len(values) != expectations["n_queries"]:
                        reasons.append("{}: {} k={} {} has {} values, expected {}".format(
                            label, condition, k, metric,
                            "no" if values is None else len(values),
                            expectations["n_queries"]))
        if reference is None:
            reference = (label, run)
        else:
            if run["query"] != reference[1]["query"]:
                reasons.append("{} is not query-aligned with {}".format(label, reference[0]))
            if run.get("index") != reference[1].get("index"):
                reasons.append("{}: index differs from {}".format(label, reference[0]))
    return reasons


def _fmt(value, digits=4, sign=""):
    """Format a float for the tables; ``None`` (an undefined statistic) prints as ``-``."""
    return "-" if value is None else "{:{}.{}f}".format(float(value), sign, digits)


def _print_table(rows, digits, with_cluster):
    """One degradation table: the means, ``r`` and its interval(s), and the validity split."""
    header = "{:>6} {:>7} {:>11} {:>11} {:>9} {:>20}".format(
        "k", "deg", "mean0", "mean_k", "r", "query CI")
    if with_cluster:
        header += " {:>20}".format("room CI")
    print(header + " {:>6} {:>8}".format("n", "newinv"))
    for row in rows:
        line = "{:>6} {:>7.1f} {:>11} {:>11} {:>9} [{:>8}, {:>8}]".format(
            row["k"], row["deg"], _fmt(row["mean0"], digits), _fmt(row["meank"], digits),
            _fmt(row["r"], 4, "+"), _fmt(row["lo"], 4, "+"), _fmt(row["hi"], 4, "+"))
        if with_cluster:
            line += " [{:>8}, {:>8}]".format(
                _fmt(row.get("r_lo"), 4, "+"), _fmt(row.get("r_hi"), 4, "+"))
        print(line + " {:>6d} {:>8.4f}".format(
            row["n_valid"], row["validity"]["newly_invalid_frac"]))


def _resolve(explicit, predicate, by_label, what):
    """The label of a role: the explicit one if given, else the first run that matches."""
    if explicit is not None:
        if explicit not in by_label:
            raise ValueError("--{} {!r} is not one of the run labels {}".format(
                what, explicit, list(by_label)))
        return explicit
    matches = [label for label, run in by_label.items() if predicate(run)]
    return matches[0] if matches else None


def main(argv=None):
    """Print every table of the pre-registered analysis and write the canonical JSON."""
    import argparse
    import hashlib
    import io
    import sys

    from tools.paired_stats import bonferroni_alpha

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("full", "k0-gate", "exploratory"), required=True,
                        help="full: the confirmatory sweep, validated against the "
                             "pre-registered design and refused if it does not match; "
                             "k0-gate: the k=0 parity gate; exploratory: anything else, "
                             "with the verdicts and the canonical sha256 suppressed")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="directories written by eval_yaw_rotation.py")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="one label per run (default: the directory name)")
    parser.add_argument("--primary-simple", default=None,
                        help="label of the confirmatory SimpleViT model "
                             "(default: the run whose checkpoint is xRIR_simple_8_shot)")
    parser.add_argument("--cyl", default=None,
                        help="label of the cylindrical model (default: backbone == cylindrical)")
    parser.add_argument("--released", default=None,
                        help="label of the released checkpoint (default: checkpoints/xRIR_unseen.pth)")
    parser.add_argument("--manifest-hash", default=None,
                        help="assert every run used this manifest")
    parser.add_argument("--exp01-per-sample", nargs="*", default=[], metavar="LABEL=PATH",
                        help="k0-gate: exp_01's per-sample JSON for a run label")
    parser.add_argument("--noise-runs", nargs="*", default=[], metavar="DIR",
                        help="k0-gate: k=0 runs of the control checkpoint on other "
                             "manifest seeds; they measure the reference-draw spread")
    parser.add_argument("--released-baseline", default="0.0549,1.358,9.69",
                        help="k0-gate: exp_01's reported EDT,C50,T60 for the released "
                             "checkpoint (it has no per-sample record)")
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.05, help="family-wise level")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="H1's practical margin on r")
    parser.add_argument("--equiv-margin", type=float, default=0.02,
                        help="equivalence margin at the patch-aligned angles")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None,
                        help="write every displayed number plus the config here")
    parser.add_argument("--summary", default=None,
                        help="write the printed text here; its sha256 goes into --json")
    args = parser.parse_args(argv)

    if bool(args.json) != bool(args.summary):
        raise ValueError("--json and --summary go together: the JSON stores the sha256 of "
                         "the summary text, so one without the other cannot be bound")
    if args.json and os.path.abspath(args.json) == os.path.abspath(args.summary):
        raise ValueError("--json and --summary must be different files")

    runs = [load_run(directory) for directory in args.runs]
    labels = args.labels if args.labels else [
        os.path.basename(os.path.normpath(d)) for d in args.runs]
    if len(labels) != len(runs) or len(set(labels)) != len(labels):
        raise ValueError("--labels must give one distinct label per run")
    by_label = dict(zip(labels, runs))
    digests = set(run["meta"]["manifest_hash"] for run in runs)
    if len(digests) != 1:
        raise ValueError("the runs used different manifests: {}".format(sorted(digests)))
    manifest_hash = digests.pop()
    if args.manifest_hash and manifest_hash != args.manifest_hash:
        raise ValueError("runs used manifest {}, expected {}".format(
            manifest_hash, args.manifest_hash))
    for label, run in by_label.items():
        if run["query"] != runs[0]["query"]:
            raise ValueError("run {} is not query-aligned with {}".format(label, labels[0]))

    roles_note = None
    primary = _resolve(args.primary_simple,
                       lambda r: "xRIR_simple_8_shot" in r["meta"]["checkpoint"],
                       by_label, "primary-simple")
    cyl = _resolve(args.cyl, lambda r: r["meta"]["backbone"] == "cylindrical",
                   by_label, "cyl")
    released = _resolve(args.released,
                        lambda r: "checkpoints/xRIR_unseen.pth" in r["meta"]["checkpoint"],
                        by_label, "released")

    roles = {"primary": primary, "cyl": cyl, "released": released}
    exp01_by_label = {}
    for item in args.exp01_per_sample:
        if "=" not in item:
            raise ValueError("--exp01-per-sample takes LABEL=PATH, got {!r}".format(item))
        label, path = item.split("=", 1)
        if label not in by_label:
            raise ValueError("--exp01-per-sample label {!r} is not one of {}".format(
                label, list(by_label)))
        with open(path, "r") as fin:
            exp01_by_label[label] = json.load(fin)
    noise_runs = [load_run(directory) for directory in args.noise_runs]
    released_baseline = None
    if args.released_baseline:
        parts = [value.strip() for value in args.released_baseline.split(",")]
        if len(parts) != len(RELEASED_BASELINE_METRICS):
            raise ValueError("--released-baseline takes {} comma-separated numbers "
                             "({}), got {!r}".format(len(RELEASED_BASELINE_METRICS),
                                                     ",".join(RELEASED_BASELINE_METRICS),
                                                     args.released_baseline))
        released_baseline = dict(zip(RELEASED_BASELINE_METRICS,
                                     [float(value) for value in parts]))
    if args.mode == "full":
        reasons = validate_full(by_label, roles, args.manifest_hash)
        print("mode: full -- valid_for_confirmatory: {}".format(not reasons))
        if reasons:
            print("this set of runs is not confirmatory-grade; no artefacts were written:")
            for reason in reasons:
                print("   - {}".format(reason))
            raise SystemExit(1)

    queries = runs[0]["query"]
    rooms = rooms_from_paths(queries)
    cols = sorted(runs[0]["meta"]["yaw_cols"], key=signed_degrees)
    acoustic_cols = sorted(runs[0]["meta"]["acoustic_cols"], key=signed_degrees)
    e_acoustic_cols = sorted(runs[0]["meta"]["e_acoustic_cols"], key=signed_degrees)
    family = FAMILY_SIZE
    alpha_adj = bonferroni_alpha(family, args.alpha)

    buffer = io.StringIO()
    original_stdout = sys.stdout

    class _Tee(object):
        def write(self, text):
            buffer.write(text)
            original_stdout.write(text)

        def flush(self):
            original_stdout.flush()

    exploratory = args.mode == "exploratory"
    gate_mode = args.mode == "k0-gate"
    out = {"mode": args.mode, "manifest_hash": manifest_hash, "n_queries": len(queries),
           "n_rooms": int(len(np.unique(rooms))),
           "config": {"runs": list(args.runs), "labels": labels,
                      "roles": {"primary": primary, "cyl": cyl, "released": released},
                      "n_boot": args.n_boot, "alpha": args.alpha, "alpha_adj": alpha_adj,
                      "family_size": family, "threshold": args.threshold,
                      "equiv_margin": args.equiv_margin, "seed": args.seed,
                      "yaw_cols": cols, "acoustic_cols": acoustic_cols,
                      "e_acoustic_cols": e_acoustic_cols,
                      "confirmatory_metrics": list(CONFIRMATORY_METRICS),
                      "preregistered_acoustic_cols": list(PREREGISTERED_ACOUSTIC_COLS),
                      "preregistered_spectral_cols": list(PREREGISTERED_SPECTRAL_COLS),
                      "preregistered_e_acoustic_cols":
                          list(PREREGISTERED_E_ACOUSTIC_COLS)},
           "meta": {label: run["meta"] for label, run in by_label.items()},
           "spectral": {}, "acoustic": {}, "k0": [], "h1": {}, "h2": {},
           "delay_flips": {label: run["delay_flips"] for label, run in by_label.items()},
           "decomposition": {label: run["decomposition"] for label, run in by_label.items()
                             if run["decomposition"] is not None}}
    if args.mode == "full":
        out["valid_for_confirmatory"] = True
        out["validation_reasons"] = []
    if exploratory:
        out["exploratory"] = True

    sys.stdout = _Tee()
    try:
        print("exp_03 yaw rotation: {} queries in {} rooms, manifest {}".format(
            out["n_queries"], out["n_rooms"], manifest_hash[:12]))
        print("runs: " + ", ".join("{} ({}, {})".format(
            label, run["meta"]["backbone"], run["meta"]["checkpoint"]) for label, run
            in by_label.items()))
        print("roles: primary={} cyl={} released={}".format(primary, cyl, released))
        print("bootstrap: n_boot={} seed={} family={} tests -> adjusted level {:.5f} "
              "(two-sided {:.2f}% intervals)".format(
                  args.n_boot, args.seed, family, alpha_adj, 100 * (1 - alpha_adj)))

        print("\n1. Spectral metrics per angle (condition P = pinned k=0 alignment, "
              "E = end to end).\n   consistency = mean|log-spec(k) - log-spec(0)|; its "
              "k=0 baseline is 0, so its ratio does not exist.")
        for label, run in by_label.items():
            out["spectral"][label] = {}
            for condition in ("P", "E"):
                out["spectral"][label][condition] = {}
                for metric in SPECTRAL_METRICS:
                    rows = degradation_rows(run, condition, metric, cols, args.n_boot,
                                            alpha_adj, args.seed)
                    out["spectral"][label][condition][metric] = rows
                    print("\n   {} | condition {} | {}".format(label, condition, metric))
                    _print_table(rows, 5, False)

        print("\n2. Acoustic metrics per angle (Griffin-Lim; EDT s, C50 dB, T60 %). "
              "T60 is descriptive only.\n   query CI = this fixed split; room CI = the "
              "17 held-out rooms; newinv = fraction of usable baselines the angle broke.")
        for label, run in by_label.items():
            out["acoustic"][label] = {}
            for condition, angles in (("P", acoustic_cols), ("E", e_acoustic_cols)):
                out["acoustic"][label][condition] = {}
                angles = sorted(set([0] + list(angles)), key=signed_degrees)
                for metric in ACOUSTIC_METRICS:
                    if any(metric not in run[condition][str(int(k))] for k in angles):
                        print("\n   {} | condition {} | {}: not evaluated at every angle "
                              "of this grid".format(label, condition, metric))
                        continue
                    rows = degradation_rows(run, condition, metric, angles, args.n_boot,
                                            alpha_adj, args.seed, rooms=rooms)
                    out["acoustic"][label][condition][metric] = rows
                    print("\n   {} | condition {} | {}".format(label, condition, metric))
                    _print_table(rows, 4, True)

        print("\n3. Paired cylindrical - control at k = 0 (nominal {:.0f}% CIs, "
              "unadjusted).\n   Same references and same Griffin-Lim phases for both "
              "models: this supersedes exp_01's epoch-12 comparison.".format(
                  100 * (1 - args.alpha)))
        if cyl and primary:
            out["k0"] = k0_comparison(by_label[cyl], by_label[primary], list(K0_METRICS),
                                      args.n_boot, args.alpha, args.seed, rooms)
            print("   {:>10} {:>10} {:>10} {:>10} {:>8} {:>21} {:>21} {:>6}".format(
                "metric", "cyl", "control", "diff", "rel %", "query CI", "room CI", "n"))
            for row in out["k0"]:
                print("   {:>10} {:>10.4f} {:>10.4f} {:>+10.4f} {:>+8.1f} "
                      "[{:>+9.4f}, {:>+9.4f}] [{:>+9.4f}, {:>+9.4f}] {:>6d}".format(
                          row["metric"], row["cyl"], row["ctrl"], row["diff"],
                          row["rel_pct"], row["q_lo"], row["q_hi"], row["r_lo"],
                          row["r_hi"], row["n_valid"]))
        else:
            print("   skipped: needs both a cylindrical and a primary SimpleViT run")

        if gate_mode:
            extra = ["{}: contains angles beyond k=0 ({})".format(label, sorted(run["P"]))
                     for label, run in sorted(by_label.items())
                     if sorted(run["P"]) != ["0"]]
            gate_rows, gate_pass, gate_reasons = k0_gate_rows(
                by_label, roles, exp01_by_label, noise_runs, released_baseline)
            gate_reasons = extra + gate_reasons
            gate_pass = bool(gate_pass and not extra)
            out.update({"gate_rows": gate_rows, "gate_pass": gate_pass,
                        "gate_reasons": gate_reasons,
                        "noise_runs": list(args.noise_runs),
                        "released_baseline": released_baseline})
            print("\n3b. k=0 parity gate against exp_01 (band = 2 x the reference-draw "
                  "spread + 1e-6).\n    The manifest pins a different reference draw than "
                  "exp_01's, which was never recorded, so\n    the comparison is "
                  "distributional and the spread is measured on manifest seeds "
                  "{}.".format(", ".join(args.noise_runs) or "(none given)"))
            print("   {:>10} {:>8} {:>12} {:>12} {:>12} {:>12} {:>6}  {}".format(
                "run", "metric", "mean k=0", "reference", "|diff|", "band", "pass",
                "source"))
            for row in gate_rows:
                print("   {:>10} {:>8} {:>12} {:>12} {:>12} {:>12} {:>6}  {}".format(
                    row["label"], row["metric"], _fmt(row["mean_k0"], 5),
                    _fmt(row["reference"], 5), _fmt(row["abs_diff"], 6),
                    _fmt(row["band"], 6),
                    "-" if row["pass"] is None else ("yes" if row["pass"] else "NO"),
                    row["source"]))
            print("   gate_pass: {}".format(gate_pass))
            for reason in gate_reasons:
                print("      - {}".format(reason))

        print("\n4. H1 (joint yaw rotation substantially degrades the model): supported "
              "if the adjusted\n   lower bound of r exceeds {:+.0%} for EDT or C50 at any "
              "angle k != 0, condition P.".format(args.threshold))
        verdicts, primary_h1_cells = {}, []
        for role, label in (() if (exploratory or gate_mode) else
                            (("primary", primary), ("released", released))):
            if label is None:
                print("   {}: no run provided".format(role))
                continue
            rows = {metric: out["acoustic"][label]["P"][metric]
                    for metric in CONFIRMATORY_METRICS}
            passes, cells = h1_verdict(rows, args.threshold)
            verdicts[role] = passes
            if role == "primary":
                primary_h1_cells = cells
            out["h1"][label] = {"role": role, "passes": passes, "cells": cells}
            print("   {} ({}): {}".format(role, label, "PASSES" if passes else "does not pass"))
            for cell in cells:
                print("      {:>4} k={:>4} ({:>6.1f} deg)  r={:+.4f}  adjusted CI "
                      "[{:+.4f}, {:+.4f}]".format(cell["metric"], cell["k"], cell["deg"],
                                                  cell["r"], cell["lo"], cell["hi"]))
        if exploratory or gate_mode:
            out["h1"]["verdict"] = "not evaluated ({} mode)".format(args.mode)
            print("   not evaluated: a pre-registered verdict needs the full design "
                  "(--mode full)")
        else:
            out["h1"]["verdict"] = h1_wording(verdicts.get("primary", False),
                                              verdicts.get("released", False))
            missing = [role for role in ("primary", "released") if role not in verdicts]
            print("   verdict: {}{}".format(out["h1"]["verdict"], "" if not missing else
                                            "  (no {} run in this summary; it counts as "
                                            "not passing)".format(" or ".join(missing))))
        out["h1"]["roles_evaluated"] = sorted(verdicts)
        out["h1"]["absolute_margins"] = {}
        for role, label in (("primary", primary), ("released", released)):
            if label is None or label not in out["acoustic"]:
                continue
            margins = {}
            for metric, unit in (("edt", "s"), ("c50", "dB")):
                rows = out["acoustic"][label]["P"][metric]
                k0 = next((row["mean0"] for row in rows if row["k"] == 0), None)
                margins[metric] = None if k0 is None else args.threshold * k0
            out["h1"]["absolute_margins"][label] = margins
            print("   {} ({}): {:+.0%} of the k=0 error is {} s EDT and {} dB C50".format(
                role, label, args.threshold, _fmt(margins["edt"], 4),
                _fmt(margins["c50"], 3)))

        print("\n5. H2 (the cylindrical backbone degrades less): D_k = r_cyl - r_control, "
              "paired on the\n   same resamples; equivalence (+-{:.0%} TOST on r_cyl) at "
              "the patch-aligned angles.".format(args.equiv_margin))
        h2_by_metric = {}
        if cyl and primary:
            for metric in CONFIRMATORY_METRICS:
                rows = h2_rows(by_label[cyl], by_label[primary], "P", metric,
                               acoustic_cols, args.n_boot, alpha_adj, args.seed,
                               rooms=rooms, equiv_margin=args.equiv_margin)
                h2_by_metric[metric] = rows
                print("\n   metric {}".format(metric))
                print("   {:>6} {:>7} {:>9} {:>9} {:>9} {:>21} {:>21} {:>24}".format(
                    "k", "deg", "r_cyl", "r_ctrl", "D_k", "query CI", "room CI",
                    "equivalent (query / room)"))
                for row in rows:
                    equivalence = row["equivalence"]

                    def _tost(name, equivalence=equivalence):
                        if equivalence is None or equivalence.get(name) is None:
                            return "-"
                        return "yes" if equivalence[name]["equivalent"] else "no"
                    verdict = "{} / {}".format(_tost("query"), _tost("cluster"))
                    print("   {:>6} {:>7.1f} {:>9} {:>9} {:>9} [{:>9}, {:>9}] "
                          "[{:>9}, {:>9}] {:>24}".format(
                              row["k"], row["deg"], _fmt(row["r_cyl"], 4, "+"),
                              _fmt(row["r_ctrl"], 4, "+"), _fmt(row["d"], 4, "+"),
                              _fmt(row["lo"], 4, "+"), _fmt(row["hi"], 4, "+"),
                              _fmt(row["c_lo"], 4, "+"), _fmt(row["c_hi"], 4, "+"),
                              verdict))
            out["h2"]["verdict"] = ({"aggregate": "not evaluated ({} mode)".format(args.mode),
                                     "cells": [], "n_cells": 0, "n_passing": 0,
                                     "passing": [], "failing": []}
                                    if (exploratory or gate_mode) else
                                    h2_verdict(primary_h1_cells, h2_by_metric))
            verdict = out["h2"]["verdict"]
            print("\n   H2 verdict: {} ({} of {} H1-passing cells)".format(
                verdict["aggregate"], verdict["n_passing"], verdict["n_cells"]))
            if verdict["cells"]:
                print("      passing: {}".format(", ".join(verdict["passing"]) or "none"))
                print("      failing: {}".format(", ".join(verdict["failing"]) or "none"))
        else:
            out["h2"]["verdict"] = {"aggregate": "not evaluated (missing run)",
                                    "cells": [], "n_cells": 0, "n_passing": 0,
                                    "passing": [], "failing": []}
            print("   skipped: needs both a cylindrical and a primary SimpleViT run")
        out["h2"]["rows"] = h2_by_metric

        print("\n6. Delay-flip audit: (query, reference) pairs whose integer direct-path "
              "delay moves\n   under the rotation -- the numerical noise condition P "
              "excludes.")
        print("   {:>10} ".format("run") + " ".join("{:>7}".format(k) for k in cols))
        for label, run in by_label.items():
            print("   {:>10} ".format(label) + " ".join(
                "{:>7}".format(run["delay_flips"].get(str(int(k)), "-")) for k in cols))

        print("\n7. Decomposition of a patch-aligned yaw (cylindrical only): relative "
              "change of each stage.\n   Scope: receiver-view tokens and pooling plus the "
              "query-source coordinate embedding;\n   reference-coordinate features not "
              "decomposed.")
        if out["decomposition"]:
            for label, decomposition in out["decomposition"].items():
                print("   {} at k={} over {} batch(es): tokens {:.3e}  pooled {:.3e}  "
                      "coord {:.3e}  log-spec {:.3e}".format(
                          label, decomposition["k"], decomposition["n_batches"],
                          decomposition["tokens_rel_change"],
                          decomposition["pooled_rel_change"],
                          decomposition["coord_rel_change"],
                          decomposition["logspec_rel_change"]))
        else:
            print("   none written (no cylindrical run)")

        print("\n8. Bootstrap convergence: every decision-driving bound recomputed with "
              "a second seed,\n   as a fraction of its own interval width (limit "
              "{:.2f}).".format(CONVERGENCE_LIMIT))
        h2_cache = {}

        def h2_row_for(metric, k, seed):
            key = (metric, int(k), int(seed))
            if key not in h2_cache:
                h2_cache[key] = h2_rows(by_label[cyl], by_label[primary], "P", metric,
                                        [k], args.n_boot, alpha_adj, seed,
                                        equiv_margin=args.equiv_margin)[0]
            return h2_cache[key]

        cells = {}
        confirmatory_angles = [k for k in acoustic_cols if int(k) != 0]
        if not (exploratory or gate_mode):
            for label in [l for l in (primary, released) if l is not None]:
                for metric in CONFIRMATORY_METRICS:
                    for k in confirmatory_angles:
                        def h1_bound(seed, label=label, metric=metric, k=k):
                            return degradation_rows(by_label[label], "P", metric, [k],
                                                    args.n_boot, alpha_adj, seed)[0]
                        cells["h1/{}/{}/{}".format(label, metric, k)] = convergence_cell(
                            h1_bound, args.seed, args.seed + 1)
            if cyl and primary:
                for metric in CONFIRMATORY_METRICS:
                    for k in confirmatory_angles:
                        def did_bound(seed, metric=metric, k=k):
                            return h2_row_for(metric, k, seed)
                        cells["h2/{}/{}".format(metric, k)] = convergence_cell(
                            did_bound, args.seed, args.seed + 1)

                        def tost_bound(seed, metric=metric, k=k):
                            equivalence = h2_row_for(metric, k, seed)["equivalence"]
                            return None if equivalence is None else equivalence["query"]
                        cells["tost/{}/{}".format(metric, k)] = convergence_cell(
                            tost_bound, args.seed, args.seed + 1)
        finite = [cell["rel_change"] for cell in cells.values()]
        worst = max(finite) if finite else 0.0
        out["convergence"] = {"cells": cells, "max_rel_change": worst,
                              "pass": bool(worst <= CONVERGENCE_LIMIT),
                              "seeds": [args.seed, args.seed + 1],
                              "limit": CONVERGENCE_LIMIT}
        print("   {} bounds recomputed with seed {} -> {}; max movement {:.4f} "
              "(pass: {})".format(len(cells), args.seed, args.seed + 1, worst,
                                  out["convergence"]["pass"]))
        if cells:
            worst_cell = max(cells, key=lambda key: cells[key]["rel_change"])
            print("   worst bound: {} [{}, {}] -> [{}, {}]".format(
                worst_cell, _fmt(cells[worst_cell]["seed_a"]["lo"], 4),
                _fmt(cells[worst_cell]["seed_a"]["hi"], 4),
                _fmt(cells[worst_cell]["seed_b"]["lo"], 4),
                _fmt(cells[worst_cell]["seed_b"]["hi"], 4)))
        else:
            print("   no confirmatory bound in this mode")
    finally:
        sys.stdout = original_stdout

    text = buffer.getvalue()
    converged = out["convergence"]["pass"]
    if args.mode == "full" and not converged:
        print("bootstrap not converged (max movement {:.4f} > {:.2f}): raise --n-boot; "
              "no artefacts were written".format(out["convergence"]["max_rel_change"],
                                                 CONVERGENCE_LIMIT), file=sys.stderr)
        raise SystemExit(1)
    if args.summary:
        _write_text(text, args.summary)
        out["summary_path"] = args.summary
    if not exploratory:
        # An exploratory summary must not look bindable: no canonical digest.
        out["summary_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    if args.json:
        _write_json(out, args.json)
        print("wrote {}".format(args.json))
    if gate_mode and not out["gate_pass"]:
        print("k=0 parity gate FAILED; the sweep must not start", file=sys.stderr)
        raise SystemExit(1)
    if not converged:
        print("bootstrap not converged: raise --n-boot", file=sys.stderr)
        raise SystemExit(1)
    return out


if __name__ == "__main__":
    main()
