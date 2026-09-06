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


SPECTRAL_METRICS = ("loss", "log_mse", "consistency")
ACOUSTIC_METRICS = ("edt", "c50", "t60")
CONFIRMATORY_METRICS = ("edt", "c50")
K0_METRICS = ("edt", "c50", "t60", "log_mse", "loss")


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

    primary = _resolve(args.primary_simple,
                       lambda r: "xRIR_simple_8_shot" in r["meta"]["checkpoint"],
                       by_label, "primary-simple")
    cyl = _resolve(args.cyl, lambda r: r["meta"]["backbone"] == "cylindrical",
                   by_label, "cyl")
    released = _resolve(args.released,
                        lambda r: "checkpoints/xRIR_unseen.pth" in r["meta"]["checkpoint"],
                        by_label, "released")

    queries = runs[0]["query"]
    rooms = rooms_from_paths(queries)
    cols = sorted(runs[0]["meta"]["yaw_cols"], key=signed_degrees)
    acoustic_cols = sorted(runs[0]["meta"]["acoustic_cols"], key=signed_degrees)
    e_acoustic_cols = sorted(runs[0]["meta"]["e_acoustic_cols"], key=signed_degrees)
    family = len(CONFIRMATORY_METRICS) * len([k for k in acoustic_cols if int(k) != 0])
    alpha_adj = bonferroni_alpha(family, args.alpha)

    buffer = io.StringIO()
    original_stdout = sys.stdout

    class _Tee(object):
        def write(self, text):
            buffer.write(text)
            original_stdout.write(text)

        def flush(self):
            original_stdout.flush()

    out = {"manifest_hash": manifest_hash, "n_queries": len(queries),
           "n_rooms": int(len(np.unique(rooms))),
           "config": {"runs": list(args.runs), "labels": labels,
                      "roles": {"primary": primary, "cyl": cyl, "released": released},
                      "n_boot": args.n_boot, "alpha": args.alpha, "alpha_adj": alpha_adj,
                      "family_size": family, "threshold": args.threshold,
                      "equiv_margin": args.equiv_margin, "seed": args.seed,
                      "yaw_cols": cols, "acoustic_cols": acoustic_cols,
                      "e_acoustic_cols": e_acoustic_cols,
                      "confirmatory_metrics": list(CONFIRMATORY_METRICS)},
           "meta": {label: run["meta"] for label, run in by_label.items()},
           "spectral": {}, "acoustic": {}, "k0": [], "h1": {}, "h2": {},
           "delay_flips": {label: run["delay_flips"] for label, run in by_label.items()},
           "decomposition": {label: run["decomposition"] for label, run in by_label.items()
                             if run["decomposition"] is not None}}

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

        print("\n4. H1 (joint yaw rotation substantially degrades the model): supported "
              "if the adjusted\n   lower bound of r exceeds {:+.0%} for EDT or C50 at any "
              "angle k != 0, condition P.".format(args.threshold))
        verdicts = {}
        for role, label in (("primary", primary), ("released", released)):
            if label is None:
                print("   {}: no run provided".format(role))
                continue
            rows = {metric: out["acoustic"][label]["P"][metric]
                    for metric in CONFIRMATORY_METRICS}
            passes, cells = h1_verdict(rows, args.threshold)
            verdicts[role] = passes
            out["h1"][label] = {"role": role, "passes": passes, "cells": cells}
            print("   {} ({}): {}".format(role, label, "PASSES" if passes else "does not pass"))
            for cell in cells:
                print("      {:>4} k={:>4} ({:>6.1f} deg)  r={:+.4f}  adjusted CI "
                      "[{:+.4f}, {:+.4f}]".format(cell["metric"], cell["k"], cell["deg"],
                                                  cell["r"], cell["lo"], cell["hi"]))
        out["h1"]["verdict"] = h1_wording(verdicts.get("primary", False),
                                          verdicts.get("released", False))
        out["h1"]["roles_evaluated"] = sorted(verdicts)
        missing = [role for role in ("primary", "released") if role not in verdicts]
        print("   verdict: {}{}".format(out["h1"]["verdict"], "" if not missing else
                                        "  (no {} run in this summary; it counts as "
                                        "not passing)".format(" or ".join(missing))))

        print("\n5. H2 (the cylindrical backbone degrades less): D_k = r_cyl - r_control, "
              "paired on the\n   same resamples; equivalence (+-{:.0%} TOST on r_cyl) at "
              "the patch-aligned angles.".format(args.equiv_margin))
        if cyl and primary:
            for metric in CONFIRMATORY_METRICS:
                rows = h2_rows(by_label[cyl], by_label[primary], "P", metric,
                               acoustic_cols, args.n_boot, alpha_adj, args.seed,
                               rooms=rooms, equiv_margin=args.equiv_margin)
                out["h2"][metric] = rows
                print("\n   metric {}".format(metric))
                print("   {:>6} {:>7} {:>9} {:>9} {:>9} {:>21} {:>21} {:>12}".format(
                    "k", "deg", "r_cyl", "r_ctrl", "D_k", "query CI", "room CI",
                    "equivalent"))
                for row in rows:
                    equivalence = row["equivalence"]
                    verdict = "-" if equivalence is None or equivalence["query"] is None \
                        else ("yes" if equivalence["query"]["equivalent"] else "no")
                    print("   {:>6} {:>7.1f} {:>9} {:>9} {:>9} [{:>9}, {:>9}] "
                          "[{:>9}, {:>9}] {:>12}".format(
                              row["k"], row["deg"], _fmt(row["r_cyl"], 4, "+"),
                              _fmt(row["r_ctrl"], 4, "+"), _fmt(row["d"], 4, "+"),
                              _fmt(row["lo"], 4, "+"), _fmt(row["hi"], 4, "+"),
                              _fmt(row["c_lo"], 4, "+"), _fmt(row["c_hi"], 4, "+"),
                              verdict))
        else:
            print("   skipped: needs both a cylindrical and a primary SimpleViT run")

        print("\n6. Delay-flip audit: (query, reference) pairs whose integer direct-path "
              "delay moves\n   under the rotation -- the numerical noise condition P "
              "excludes.")
        print("   {:>10} ".format("run") + " ".join("{:>7}".format(k) for k in cols))
        for label, run in by_label.items():
            print("   {:>10} ".format(label) + " ".join(
                "{:>7}".format(run["delay_flips"].get(str(int(k)), "-")) for k in cols))

        print("\n7. Decomposition of a patch-aligned yaw (cylindrical only): relative "
              "change of each stage.")
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

        print("\n8. Bootstrap convergence of the H1 family: how far each confirmatory "
              "bound moves\n   when only the bootstrap seed changes, as a fraction of "
              "the interval width.")
        ratios = {}
        for label in [l for l in (primary, released) if l is not None]:
            for metric in CONFIRMATORY_METRICS:
                for k in [k for k in acoustic_cols if int(k) != 0]:
                    def interval(seed, label=label, metric=metric, k=k):
                        return degradation_rows(by_label[label], "P", metric, [k],
                                                args.n_boot, alpha_adj, seed)[0]
                    ratios["{}/{}/{}".format(label, metric, k)] = convergence_check(
                        interval, args.seed, args.seed + 1)
        worst = max(ratios.values()) if ratios else 0.0
        out["convergence"] = {"per_cell": ratios, "max_ratio": worst,
                              "seeds": [args.seed, args.seed + 1], "limit": 0.10}
        print("   max over {} cells: {:.4f} (limit 0.10)".format(len(ratios), worst))
        if ratios:
            worst_cell = max(ratios, key=lambda key: ratios[key])
            print("   worst cell: {} at {:.4f}".format(worst_cell, ratios[worst_cell]))
    finally:
        sys.stdout = original_stdout

    text = buffer.getvalue()
    if args.summary:
        parent = os.path.dirname(os.path.abspath(args.summary))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.summary, "w") as fout:
            fout.write(text)
        out["summary_path"] = args.summary
    out["summary_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    if args.json:
        parent = os.path.dirname(os.path.abspath(args.json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.json, "w") as fout:
            json.dump(out, fout, indent=1)
        print("wrote {}".format(args.json))
    if out["convergence"]["max_ratio"] > out["convergence"]["limit"]:
        print("bootstrap not converged: raise --n-boot", file=sys.stderr)
        raise SystemExit(1)
    return out


if __name__ == "__main__":
    main()
