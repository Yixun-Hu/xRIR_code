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
