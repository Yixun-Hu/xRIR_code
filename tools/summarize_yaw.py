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
    e0 = np.asarray(run[condition]["0"][metric], dtype=np.float64)
    rooms = None if rooms is None else np.asarray(rooms)
    rows = []
    for k in cols:
        ek = np.asarray(run[condition][str(int(k))][metric], dtype=np.float64)
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
        and otherwise ``{"margin", "query", "cluster"}`` with the two TOST results.

    Raises:
        ValueError: if the two runs are not query-aligned.
    """
    if run_cyl["query"] != run_ctrl["query"]:
        raise ValueError("the two runs are not query-aligned: D_k would not be paired")
    a0 = np.asarray(run_cyl[condition]["0"][metric], dtype=np.float64)
    b0 = np.asarray(run_ctrl[condition]["0"][metric], dtype=np.float64)
    rooms = None if rooms is None else np.asarray(rooms)
    rows = []
    for k in cols:
        ak = np.asarray(run_cyl[condition][str(int(k))][metric], dtype=np.float64)
        bk = np.asarray(run_ctrl[condition][str(int(k))][metric], dtype=np.float64)
        mask = valid_mask(a0, ak) & valid_mask(b0, bk)
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
            row["equivalence"] = {"margin": float(equiv_margin), "query": None,
                                  "cluster": None}
            for name, ids in (("query", None), ("cluster", clusters)):
                if name == "cluster" and ids is None:
                    continue
                try:
                    row["equivalence"][name] = equivalence_tost(
                        a0[mask], ak[mask], margin=float(equiv_margin), n_boot=int(n_boot),
                        alpha=float(alpha_adj), seed=int(seed), clusters=ids)
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
        a = np.asarray(run_cyl["P"]["0"][metric], dtype=np.float64)
        b = np.asarray(run_ctrl["P"]["0"][metric], dtype=np.float64)
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
    first, second = fn(seed_a), fn(seed_b)
    width = float(first["hi"]) - float(first["lo"])
    change = max(abs(float(first["lo"]) - float(second["lo"])),
                 abs(float(first["hi"]) - float(second["hi"])))
    if width <= 0.0:
        return 0.0 if change == 0.0 else float("inf")
    return change / width
