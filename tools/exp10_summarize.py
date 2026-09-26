"""Summariser for the exp_10 ``yaw_pilot`` runs: the two readouts and their ratio.

Per arm, angle and metric the pilot reports

* **R1** the signed accuracy change ``delta = mean(e_alpha - e_0)`` against the ground
  truth and the prediction shift ``G = mean(g_alpha)`` against the model's own ``k = 0``
  prediction, each with a 95 % paired percentile bootstrap interval on the **shared
  comparison mask**;
* **R2** their ratio ``R = G / delta`` from the **same resamples**, with an explicit
  status -- ``defined``, ``improvement``, ``denominator uncertain`` or ``undefined`` --
  so a shift beside an uncertain denominator can never be printed as a multiple;
* **R3** the Griffin-Lim-free spectral shift beside the waveform and acoustic ones.

Everything here is descriptive: intervals, statuses and counts, no hypothesis test.  The
contract the tests pin is section 4 of the plan: one draw of query indices per replicate
feeds ``delta*``, ``G*`` and ``R*`` alike; the mask is shared by the three series;
exact-zero ``delta*`` draws are counted, never dropped, and decide ``undefined`` before
any interval is read; the room-cluster bootstrap resamples rooms with their multiplicity
and re-weights queries, never averaging per room first.
"""
from __future__ import annotations

import numpy as np

N_BOOT = 10000
ALPHA = 0.05
SEEDS = (0, 1)                 # seed 0 reports, seed 1 is the convergence check
CONVERGENCE_TOLERANCE = 0.10   # a bound may move by < 10 % of the interval width
_CHUNK_CELLS = 1 << 22         # bootstrap draws are generated in chunks of this many cells


def _as_float(values, n=None):
    """A per-sample series as float64 (``None``/missing becomes NaN)."""
    if values is None:
        return None
    array = np.asarray([np.nan if value is None else float(value) for value in values],
                       dtype=np.float64)
    if n is not None and array.size != n:
        raise ValueError("expected {} values, got {}".format(n, array.size))
    return array


def shared_mask(e0, ek, gk):
    """The comparison mask of one cell, plus why each query was excluded.

    A paired readout is only paired if ``delta`` and ``G`` are computed on the *same*
    queries, so a query invalid in any of ``e_0``, ``e_alpha`` or ``g_alpha`` leaves the
    cell entirely.  A shift-only metric (no ground-truth counterpart) passes ``None`` for
    the errors and keeps every query with a finite gap.

    Args:
        e0: per-query error at ``k = 0`` (or ``None`` for a shift-only metric).
        ek: per-query error at the angle (or ``None``).
        gk: per-query gap at the angle.

    Returns:
        ``(mask, exclusions)`` -- a boolean array and
        ``{"invalid_at_0", "invalid_at_alpha", "gap_invalid", "excluded_total",
        "n_mask", "n_total"}``.
    """
    gap = np.asarray(gk, dtype=np.float64)
    finite_gap = np.isfinite(gap)
    mask = finite_gap.copy()
    counts = {"invalid_at_0": 0, "invalid_at_alpha": 0,
              "gap_invalid": int((~finite_gap).sum())}
    for name, series in (("invalid_at_0", e0), ("invalid_at_alpha", ek)):
        if series is None:
            continue
        finite = np.isfinite(np.asarray(series, dtype=np.float64))
        counts[name] = int((~finite).sum())
        mask &= finite
    counts["n_total"] = int(mask.size)
    counts["n_mask"] = int(mask.sum())
    counts["excluded_total"] = counts["n_total"] - counts["n_mask"]
    return mask, counts


def bootstrap_draws(deltas, gaps, n_boot=N_BOOT, seed=0, clusters=None):
    """Resample ``delta*``, ``G*`` and ``R*`` from **one** index draw per replicate.

    Sharing the draw is the whole point: the ratio of two independently resampled means
    would not be the ratio of the quantities the intervals describe.

    Args:
        deltas: per-query paired differences ``e_alpha - e_0`` (finite), or ``None`` for a
            shift-only metric.
        gaps: per-query gaps ``g_alpha`` (finite).
        n_boot: number of resamples.
        seed: ``np.random.default_rng`` seed.
        clusters: ``None`` to resample queries, or one cluster id per query to resample
            whole clusters with replacement and keep every query of a drawn cluster with
            its multiplicity (query-weighted, never averaged per cluster).

    Returns:
        ``{"delta", "gap", "ratio"}`` -- three ``[n_boot]`` float64 arrays (``delta`` and
        ``ratio`` are all-NaN when ``deltas`` is ``None``).
    """
    gaps = np.asarray(gaps, dtype=np.float64)
    n = gaps.size
    rng = np.random.default_rng(seed)
    out = {name: np.empty(int(n_boot), dtype=np.float64)
           for name in ("delta", "gap", "ratio")}
    if deltas is None:
        out["delta"][:] = np.nan
        out["ratio"][:] = np.nan
    series = [gaps] if deltas is None else [np.asarray(deltas, dtype=np.float64), gaps]
    names = ["gap"] if deltas is None else ["delta", "gap"]

    with np.errstate(divide="ignore", invalid="ignore"):
        if clusters is None:
            chunk = max(1, min(int(n_boot), _CHUNK_CELLS // max(n, 1)))
            done = 0
            while done < int(n_boot):
                size = min(chunk, int(n_boot) - done)
                idx = rng.integers(0, n, size=(size, n))
                for name, values in zip(names, series):
                    out[name][done:done + size] = values[idx].mean(axis=1)
                done += size
        else:
            _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
            n_cl = int(inverse.max()) + 1
            counts = np.bincount(inverse, minlength=n_cl).astype(np.float64)
            sums = [np.bincount(inverse, weights=values, minlength=n_cl)
                    for values in series]
            chunk = max(1, min(int(n_boot), _CHUNK_CELLS // max(n_cl, 1)))
            done = 0
            while done < int(n_boot):
                size = min(chunk, int(n_boot) - done)
                draws = rng.integers(0, n_cl, size=(size, n_cl))
                weights = np.zeros((size, n_cl), dtype=np.float64)
                np.add.at(weights,
                          (np.repeat(np.arange(size), n_cl), draws.ravel()), 1.0)
                denominator = weights @ counts
                for name, total in zip(names, sums):
                    out[name][done:done + size] = (weights @ total) / denominator
                done += size
        if deltas is not None:
            out["ratio"] = out["gap"] / out["delta"]
    return out


def _interval(values, alpha=ALPHA):
    """Two-sided percentile interval at level ``1 - alpha``."""
    lo, hi = np.percentile(values, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def ratio_status(delta_lo, delta_hi, zero_draws):
    """The plan's R2 status, with ``undefined`` decided before the interval is read."""
    if zero_draws > 0:
        return "undefined"
    if delta_hi < 0:
        return "improvement"
    if delta_lo > 0:
        return "defined"
    return "denominator uncertain"


def _empty_cell(exclusions, reason, unit=None):
    """A cell with nothing to estimate (empty mask, or a metric with no paired error)."""
    return {"n": int(exclusions["n_mask"]), "unit": unit,
            "mean_0": None, "mean_alpha": None, "mean_gap": None,
            "delta": {"point": None, "lo": None, "hi": None},
            "gap": {"point": None, "lo": None, "hi": None},
            "ratio": {"point": None, "lo": None, "hi": None, "status": "undefined",
                      "zero_draw_fraction": None, "reason": reason},
            "contributions": {"fraction_worse": None, "mean_positive_part": None,
                              "mean_negative_part": None},
            "exclusions": exclusions, "unit_of_resampling": None}


def bootstrap_cell(e0, ek, gk, n_boot=N_BOOT, seed=0, alpha=ALPHA, clusters=None,
                   unit=None):
    """One arm x angle x metric cell: ``delta``, ``G`` and ``R`` on the shared mask.

    Args:
        e0: per-query error at ``k = 0``, or ``None`` for a shift-only metric.
        ek: per-query error at the angle, or ``None``.
        gk: per-query gap at the angle.
        n_boot: bootstrap resamples (the plan's 10 000).
        seed: bootstrap seed (0 reports, 1 checks convergence).
        alpha: two-sided level.
        clusters: one cluster id per query for the room-cluster bootstrap.
        unit: the metric's unit, carried through to the tables.

    Returns:
        ``{"n", "unit", "mean_0", "mean_alpha", "mean_gap", "delta", "gap", "ratio",
        "contributions", "exclusions", "unit_of_resampling"}``.  The ratio carries its
        ``status``, the fraction of exact-zero ``delta*`` draws and, when it is withheld,
        the ``reason``.
    """
    n_total = len(gk)
    e0 = _as_float(e0, n_total)
    ek = _as_float(ek, n_total)
    gk = _as_float(gk, n_total)
    mask, exclusions = shared_mask(e0, ek, gk)
    if exclusions["n_mask"] == 0:
        return _empty_cell(exclusions, "empty comparison mask", unit)

    gaps = gk[mask]
    deltas = None if e0 is None or ek is None else (ek[mask] - e0[mask])
    ids = None if clusters is None else np.asarray(clusters)[mask]
    draws = bootstrap_draws(deltas, gaps, n_boot=n_boot, seed=seed, clusters=ids)

    cell = {"n": int(exclusions["n_mask"]), "unit": unit,
            "mean_0": float(e0[mask].mean()) if e0 is not None else None,
            "mean_alpha": float(ek[mask].mean()) if ek is not None else None,
            "mean_gap": float(gaps.mean()),
            "exclusions": exclusions,
            "unit_of_resampling": "query" if clusters is None else "room"}
    gap_lo, gap_hi = _interval(draws["gap"], alpha)
    cell["gap"] = {"point": float(gaps.mean()), "lo": gap_lo, "hi": gap_hi}

    if deltas is None:
        cell["delta"] = {"point": None, "lo": None, "hi": None}
        cell["ratio"] = {"point": None, "lo": None, "hi": None, "status": "undefined",
                         "zero_draw_fraction": None,
                         "reason": "no paired error for this metric"}
        cell["contributions"] = {"fraction_worse": None, "mean_positive_part": None,
                                 "mean_negative_part": None}
        return cell

    delta_point = float(deltas.mean())
    delta_lo, delta_hi = _interval(draws["delta"], alpha)
    cell["delta"] = {"point": delta_point, "lo": delta_lo, "hi": delta_hi}
    zero_draws = int(np.count_nonzero(draws["delta"] == 0.0))
    status = ratio_status(delta_lo, delta_hi, zero_draws)

    # A3: the point ratio is withheld -- with its reason -- before any division.
    point = None if delta_point == 0.0 else float(cell["gap"]["point"] / delta_point)
    ratio = {"point": point, "lo": None, "hi": None, "status": status,
             "zero_draw_fraction": float(zero_draws) / float(n_boot), "reason": None}
    if status in ("defined", "improvement"):
        ratio["lo"], ratio["hi"] = _interval(draws["ratio"], alpha)
    elif status == "undefined":
        ratio["reason"] = ("observed delta = 0" if delta_point == 0.0 else
                           "{} of {} resamples left delta exactly 0".format(
                               zero_draws, int(n_boot)))
    else:
        ratio["reason"] = "the delta interval contains 0"
    cell["ratio"] = ratio

    positive = deltas[deltas > 0]
    negative = deltas[deltas < 0]
    cell["contributions"] = {
        "fraction_worse": float(positive.size) / float(deltas.size),
        "mean_positive_part": float(positive.sum() / deltas.size),
        "mean_negative_part": float(negative.sum() / deltas.size)}
    return cell


def _bound_movement(first, second):
    """Movement of one interval's bounds between two seeds, relative to its width."""
    if first["lo"] is None or second["lo"] is None:
        return {"converged": "not applicable", "max_relative_movement": None,
                "zero_width": None}
    width = float(first["hi"]) - float(first["lo"])
    moved = max(abs(float(second["lo"]) - float(first["lo"])),
                abs(float(second["hi"]) - float(first["hi"])))
    if width == 0.0:
        # A zero-width interval that is identical under both seeds is converged (k = 0
        # is exactly that); one that moved at all is not.
        return {"converged": bool(moved == 0.0), "max_relative_movement": None,
                "zero_width": True}
    return {"converged": bool(moved < CONVERGENCE_TOLERANCE * width),
            "max_relative_movement": moved / width, "zero_width": False}


def convergence(first, second):
    """Whether every reported bound of a cell is stable between seeds 0 and 1.

    Args:
        first: the cell computed with seed 0 (the one that is reported).
        second: the same cell computed with seed 1.

    Returns:
        ``{"converged", "status_agrees", "delta", "gap", "ratio"}``; each interval entry
        carries ``converged`` (``True`` / ``False`` / ``"not applicable"``),
        ``max_relative_movement`` and ``zero_width``.
    """
    report = {name: _bound_movement(first[name], second[name])
              for name in ("delta", "gap", "ratio")}
    report["status_agrees"] = bool(first["ratio"]["status"] == second["ratio"]["status"])
    report["converged"] = all(entry["converged"] is not False
                              for entry in (report["delta"], report["gap"],
                                            report["ratio"]))
    return report


def headline_multiple(cell, convergence_report):
    """Whether this cell may be stated as "the shift is N times the change" (A3).

    A multiple needs a ``defined`` denominator, converged bounds **and** the same
    denominator status under both seeds; anything else reports why it is withheld.

    Returns:
        ``{"reportable", "point", "lower_bound", "upper_bound", "reason"}``.
    """
    ratio = cell["ratio"]
    result = {"reportable": False, "point": ratio.get("point"),
              "lower_bound": ratio.get("lo"), "upper_bound": ratio.get("hi"),
              "reason": None}
    if ratio["status"] == "denominator uncertain":
        result["reason"] = "denominator uncertain"
        return result
    if ratio["status"] == "undefined":
        result["reason"] = ratio.get("reason") or "undefined"
        return result
    if ratio["status"] == "improvement":
        result["reason"] = "predictions shift while net error improves"
        return result
    if not (convergence_report.get("converged") and
            convergence_report.get("status_agrees")):
        result["reason"] = "unresolved Monte Carlo uncertainty"
        return result
    result["reportable"] = True
    return result
