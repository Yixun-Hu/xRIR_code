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

import os

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


def room_ids(queries):
    """The room each query belongs to -- the cluster unit of the secondary bootstrap.

    A query path is ``<Category>/<Room>/S00i_R00j_hybrid_IR.wav``, so the room is its
    directory: 17 rooms over the unseen split's 6337 queries.
    """
    return [os.path.dirname(str(query)) for query in queries]


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
            "exclusions": exclusions, "unit_of_resampling": None, "n_clusters": None}


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
            "unit_of_resampling": "query" if clusters is None else "room",
            # With one cluster the cluster bootstrap is degenerate (every resample is the
            # same set), so the count travels with the interval it explains.
            "n_clusters": None if ids is None else int(np.unique(ids).size)}
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


#: The three paired readouts: (label, Metric-2 error, Metric-1 gap, unit).
METRIC_TRIPLES = (("EDT", "edt_err", "edt_gap", "s"),
                  ("C50", "c50_err", "c50_gap", "dB"),
                  ("T60", "t60_err", "t60_gap", "% of T60"))
#: Shift-only quantities: the Griffin-Lim-free pair plus the waveform distances (R3).
SHIFT_ONLY_METRICS = (("logspec_mad", "log-magnitude"), ("mag_rel_l2", "relative"),
                      ("wave_rel_l2", "relative"), ("wave_mad", "amplitude"))
#: Accuracy-only quantities (no Metric-1 counterpart exists for them).
DELTA_ONLY_METRICS = (("log_mse", "log-magnitude^2"), ("loss", "test loss"))
#: Fields that must agree between two runs on the same device (A2).
IMPLEMENTATION_FIELDS = ("batch_size", "batch_canonical", "tool_sha256", "gl_seed",
                         "cudnn_deterministic", "cudnn_allow_tf32", "matmul_allow_tf32",
                         "torch_version", "numpy_version", "native_len", "padded_len",
                         "metric_window")
CONTROL_WAVE_TOLERANCE = 1e-6
CONTROL_ACOUSTIC_TOLERANCE = 1e-9
CONTROL_ACOUSTIC_METRICS = ("edt_err", "c50_err", "t60_err",
                            "edt_gap", "c50_gap", "t60_gap")


def guard_runs(run_dirs):
    """Load the runs to be summarised, refusing any combination A2 forbids.

    Every arm's ``delta``, ``G`` and ``R`` come from **one** per-sample file whose
    ``execution_id`` matches its meta and whose waveform arrays still hash as recorded
    (``tools.exp10_compare.load_run``).  On top of that, two runs may not share a
    ``protocol_id`` -- that would be the same configuration executed twice, and nothing
    downstream could say which numbers came from which -- and two runs on the same device
    may not differ in their batch or implementation settings, because their per-query
    values are then not computed the same way.

    Args:
        run_dirs: the run directories, in the order they should be reported.

    Returns:
        A list of ``{"run_dir", "meta", "per_sample"}``.

    Raises:
        ValueError: on a failed run guard, a duplicated ``protocol_id`` or an
            implementation mismatch within one device.
    """
    from tools.exp10_compare import load_run

    loaded = []
    for run_dir in run_dirs:
        meta, per_sample = load_run(run_dir)
        loaded.append({"run_dir": run_dir, "meta": meta, "per_sample": per_sample})

    seen = {}
    for entry in loaded:
        protocol = entry["meta"]["protocol_id"]
        if protocol in seen:
            raise ValueError(
                "{} and {} have the same protocol_id {}: two executions of one "
                "configuration cannot be combined".format(
                    seen[protocol], entry["run_dir"], protocol))
        seen[protocol] = entry["run_dir"]

    by_device = {}
    for entry in loaded:
        device = entry["meta"]["device"]
        first = by_device.setdefault(device, entry)
        for field in IMPLEMENTATION_FIELDS:
            if entry["meta"].get(field) != first["meta"].get(field):
                raise ValueError(
                    "{} and {} both ran on {} but their {} differs ({!r} vs {!r}): their "
                    "per-query values are not computed the same way".format(
                        first["run_dir"], entry["run_dir"], device, field,
                        first["meta"].get(field), entry["meta"].get(field)))
    return loaded


def _series(per_sample, angle, metric):
    """One stored per-sample series of an angle cell (``None`` when the metric is absent)."""
    cell = per_sample["angles"][str(int(angle))]
    return cell[metric] if metric in cell else None


def summarize_run(run_dir, n_boot=N_BOOT, alpha=ALPHA, seeds=SEEDS):
    """Every cell of one arm: query-level and room-level, both seeds, plus convergence.

    Args:
        run_dir: the arm's run directory (one execution; see :func:`guard_runs`).
        n_boot: bootstrap resamples.
        alpha: two-sided level.
        seeds: ``(reporting seed, convergence seed)``.

    Returns:
        ``{"arm", "run_dir", "execution_id", "protocol_id", "meta", "queries", "rooms",
        "angles"}`` where ``angles[str(k)][label]`` carries ``query``, ``room``,
        ``convergence``, ``convergence_room`` and ``headline``.
    """
    from tools.exp10_compare import load_run

    meta, per_sample = load_run(run_dir)
    queries = per_sample["query"]
    clusters = room_ids(queries)

    angles = {}
    for k in sorted(int(k) for k in meta["ks"]):
        cell = {}
        specs = ([(label, error, gap, unit) for label, error, gap, unit in METRIC_TRIPLES] +
                 [(name, None, name, unit) for name, unit in SHIFT_ONLY_METRICS] +
                 [(name, name, None, unit) for name, unit in DELTA_ONLY_METRICS])
        for label, error, gap, unit in specs:
            e0 = _series(per_sample, 0, error) if error else None
            ek = _series(per_sample, k, error) if error else None
            if gap is not None:
                gk = _series(per_sample, k, gap)
            else:
                # An accuracy-only metric has no gap: its "gap" series is all-NaN, so the
                # mask still comes from the two errors and only delta is estimated.
                gk = [0.0] * len(queries)
            if (error and (e0 is None or ek is None)) or gk is None:
                continue
            entry = {}
            for name, ids in (("query", None), ("room", clusters)):
                entry[name] = bootstrap_cell(e0, ek, gk, n_boot=n_boot, seed=seeds[0],
                                             alpha=alpha, clusters=ids, unit=unit)
                second = bootstrap_cell(e0, ek, gk, n_boot=n_boot, seed=seeds[1],
                                        alpha=alpha, clusters=ids, unit=unit)
                key = "convergence" if name == "query" else "convergence_room"
                entry[key] = convergence(entry[name], second)
            if gap is None:
                for scope in ("query", "room"):
                    entry[scope]["gap"] = {"point": None, "lo": None, "hi": None}
                    entry[scope]["mean_gap"] = None
                    entry[scope]["ratio"] = {
                        "point": None, "lo": None, "hi": None, "status": "undefined",
                        "zero_draw_fraction": None,
                        "reason": "no prediction shift is defined for this metric"}
            entry["headline"] = headline_multiple(entry["query"], entry["convergence"])
            entry["room_status"] = entry["room"]["ratio"]["status"]
            cell[label] = entry
        angles[str(k)] = cell

    return {"arm": meta.get("arm"), "run_dir": run_dir, "meta": meta,
            "execution_id": meta["execution_id"], "protocol_id": meta["protocol_id"],
            "n_queries": len(queries), "rooms": sorted(set(clusters)), "angles": angles}


def _control_deviation(values, n, name, problems):
    """One per-query deviation array: present, the right length and finite throughout.

    A null deviation is a missing measurement, not a small one, so it is counted and
    reported instead of being dropped by ``nanmax`` -- a control must never pass because
    the evidence against it was unavailable.
    """
    if values is None:
        problems.append("{} is missing".format(name))
        return None, 0
    array = _as_float(values)
    if array.size != n:
        problems.append("{} has length {}, not the run's {}".format(name, array.size, n))
        return None, 0
    nonfinite = int((~np.isfinite(array)).sum())
    if nonfinite:
        problems.append("{} is not finite for {} of {} queries".format(
            name, nonfinite, n))
    finite = array[np.isfinite(array)]
    return (float(finite.max()) if finite.size else None), nonfinite


def controls_table(run_dir, wave_tol=CONTROL_WAVE_TOLERANCE,
                   acoustic_tol=CONTROL_ACOUSTIC_TOLERANCE):
    """The plumbing controls: does a repeated inference reproduce the cell it repeats?

    ``ctrl_zero_repeat`` and ``ctrl_k128_repeat`` re-run an angle from scratch and
    ``ctrl_full_turn`` runs ``k = 512``, which is the identity path through the rotation;
    all three must land on their original to 1e-6 (waveform and log-spectrogram) and 1e-9
    (acoustic metrics), and must invalidate exactly the same queries.

    The evidence is validated before any tolerance is applied: every control the run
    **declared** in ``meta.control_specs`` must be present, no undeclared control may
    appear, each cell must carry its two deviation arrays and every acoustic metric at the
    run's length, and every per-query deviation must be finite.  A control with any of
    those problems fails -- it cannot be said to have reproduced anything.

    Args:
        run_dir: the run directory (its controls must have been written).
        wave_tol: tolerance on the stored per-query waveform / log-spec deviations.
        acoustic_tol: tolerance on the acoustic metrics.

    Returns:
        ``{"ok", "run_dir", "tolerances", "declared_controls", "missing_controls",
        "undeclared_controls", "controls"}``; each control carries its maximum waveform,
        log-spec and acoustic deviation, the worst acoustic metric, the validity
        mismatches, the count of non-finite deviations, the problems found and ``ok``.
    """
    from tools.exp10_compare import load_run

    meta, per_sample = load_run(run_dir)
    specs = {entry["id"]: entry for entry in meta.get("control_specs", [])}
    stored_controls = per_sample.get("controls", {})
    n = len(per_sample["query"])
    table = {"ok": True, "run_dir": run_dir,
             "tolerances": {"waveform": float(wave_tol), "logspec": float(wave_tol),
                            "acoustic": float(acoustic_tol)},
             "declared_controls": sorted(specs),
             "missing_controls": sorted(set(specs) - set(stored_controls)),
             "undeclared_controls": sorted(set(stored_controls) - set(specs)),
             "controls": {}}
    if table["missing_controls"] or table["undeclared_controls"]:
        table["ok"] = False

    for name in sorted(set(specs) & set(stored_controls)):
        stored = stored_controls[name]
        spec = specs[name]
        problems = []
        reference = str(int(spec.get("compare_to", 0)))
        original = per_sample["angles"].get(reference)
        if original is None:
            problems.append("the reference angle {} is not in the run".format(reference))
            original = {}

        wave, wave_nonfinite = _control_deviation(
            stored.get("wave_max_abs_diff"), n, "wave_max_abs_diff", problems)
        logspec, logspec_nonfinite = _control_deviation(
            stored.get("logspec_max_abs_diff"), n, "logspec_max_abs_diff", problems)

        worst_metric, worst_value, mismatches = None, 0.0, 0
        for metric in CONTROL_ACOUSTIC_METRICS:
            if metric not in stored:
                problems.append("{} is missing from the control".format(metric))
                continue
            if metric not in original:
                problems.append("{} is missing from the angle it repeats".format(metric))
                continue
            control_values = _as_float(stored[metric])
            original_values = _as_float(original[metric])
            if control_values.size != n or original_values.size != n:
                problems.append("{} has length {}, not the run's {}".format(
                    metric, control_values.size, n))
                continue
            invalid = np.isfinite(control_values) != np.isfinite(original_values)
            mismatches += int(invalid.sum())
            both = np.isfinite(control_values) & np.isfinite(original_values)
            if both.any():
                worst = float(np.abs(control_values[both] - original_values[both]).max())
                if worst > worst_value:
                    worst_metric, worst_value = metric, worst

        within = (wave is not None and logspec is not None and
                  wave <= wave_tol and logspec <= wave_tol and
                  worst_value <= acoustic_tol)
        ok = bool(within and not problems and mismatches == 0)
        table["controls"][name] = {
            "k": spec.get("k"), "compare_to": int(reference), "n": n,
            "wave_max_abs_diff": wave, "logspec_max_abs_diff": logspec,
            "acoustic_max_abs_diff": worst_value, "worst_acoustic_metric": worst_metric,
            "validity_mismatches": mismatches,
            "nonfinite_deviations": wave_nonfinite + logspec_nonfinite,
            "problems": problems, "ok": ok}
        table["ok"] = table["ok"] and ok
    return table


#: Figure units: the stored EDT is in seconds, the figure (like FLAC's) plots milliseconds.
FIGURE_SCALE = {"EDT": ("ms", 1000.0), "C50": ("dB", 1.0), "T60": ("%", 1.0),
                "logspec_mad": ("log-magnitude", 1.0)}
FIGURE_PANELS = ("T60", "C50", "EDT", "logspec_mad")
PANEL_TITLES = {"T60": "T60", "C50": "C50", "EDT": "EDT",
                "logspec_mad": "GL-free spectral shift"}
#: exp_04's five-seed SD of the same-budget SimpleViT, in the figure's units.  It is
#: drawn for context only -- it plays no part in any status, interval or decision.
HISTORICAL_BAND = {("simple", 8): {"EDT": 0.186, "C50": 0.0053, "T60": 0.009},
                   ("simple", 1): {"EDT": 0.609, "C50": 0.0133, "T60": 0.101}}
BAND_LABEL = ("historical baseline evaluation variability "
              "(references and phases redrawn), context only")
COLOR_DEGRADATION = "#0072B2"   # Okabe-Ito blue
COLOR_SHIFT = "#E69F00"         # Okabe-Ito orange
#: Short codes for the R2 status, printed under each angle so a bar is never read as a
#: multiple when its denominator is uncertain.
STATUS_CODES = {"defined": "def", "improvement": "impr",
                "denominator uncertain": "den?", "undefined": "und"}
STATUS_FOOTNOTE = ("ratio status under each angle: def = defined, impr = net error "
                   "improves, den? = denominator uncertain (no ratio), und = undefined; "
                   "n = queries in the shared comparison mask")


def _sanitize(payload):
    """Replace every non-finite float by ``None`` so the JSON can be strict."""
    if isinstance(payload, dict):
        return {key: _sanitize(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_sanitize(value) for value in payload]
    if isinstance(payload, (float, np.floating)):
        value = float(payload)
        return value if np.isfinite(value) else None
    if isinstance(payload, (np.integer,)):
        return int(payload)
    if isinstance(payload, (np.bool_,)):
        return bool(payload)
    return payload


def historical_band(meta):
    """The context band for one arm, or ``None`` when no historical SD exists for it."""
    return HISTORICAL_BAND.get((meta.get("backbone"), int(meta.get("num_shot", 0))))


def build_summary(run_dirs, n_boot=N_BOOT, alpha=ALPHA, seeds=SEEDS):
    """Summarise every arm and bind the result to the exact inputs it came from.

    Args:
        run_dirs: one run directory per arm, in reporting order.
        n_boot: bootstrap resamples.
        alpha: two-sided level.
        seeds: ``(reporting seed, convergence seed)``.

    Returns:
        The canonical summary dict: the tool and its settings, the ``inputs`` (each run's
        path, ``protocol_id``, ``execution_id`` and the sha256 of its ``per_sample.json``
        and ``meta.json``), one entry per arm and the historical band's provenance note.
    """
    import datetime

    from tools.exp10_yaw_pilot import file_sha256

    loaded = guard_runs(run_dirs)
    labels = [entry["meta"].get("arm") for entry in loaded]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError("two runs share the arm label {}: the figures and tables are "
                         "keyed by it, so the arms must be named distinctly".format(
                             ", ".join(str(label) for label in duplicates)))
    inputs, arms = [], []
    for entry in loaded:
        run_dir, meta = entry["run_dir"], entry["meta"]
        inputs.append({"run_dir": os.path.abspath(run_dir), "arm": meta.get("arm"),
                       "protocol_id": meta["protocol_id"],
                       "execution_id": meta["execution_id"],
                       "per_sample_sha256": file_sha256(
                           os.path.join(run_dir, "per_sample.json")),
                       "meta_sha256": file_sha256(os.path.join(run_dir, "meta.json"))})
        arm = summarize_run(run_dir, n_boot=n_boot, alpha=alpha, seeds=seeds)
        arm["controls"] = controls_table(run_dir)
        arm["historical_band"] = historical_band(meta)
        arms.append(arm)
    return {"tool": "tools/exp10_summarize.py",
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "n_boot": int(n_boot), "alpha": float(alpha), "seeds": list(seeds),
            "convergence_tolerance": CONVERGENCE_TOLERANCE,
            "band_label": BAND_LABEL, "inputs": inputs, "arms": arms}


def verify_inputs(summary):
    """Refuse a summary whose inputs are no longer the bytes it was computed from.

    Two bindings are checked, not one: the files must still hash to what the summary
    recorded, **and** each run must still bind its own per-sample file
    (``meta.per_sample_sha256``) -- a run that has lost that binding cannot support the
    numbers in the summary however well the summary's own hashes match.

    Raises:
        ValueError: if a recorded ``per_sample.json`` / ``meta.json`` is missing, its
            sha256 has changed, or the run's meta no longer binds its per-sample file.
    """
    import json as _json

    from tools.exp10_yaw_pilot import file_sha256

    for entry in summary["inputs"]:
        for name in ("per_sample.json", "meta.json"):
            path = os.path.join(entry["run_dir"], name)
            key = name.replace(".json", "_sha256")
            if not os.path.isfile(path):
                raise ValueError("{} is missing; the summary cannot be verified".format(path))
            if file_sha256(path) != entry[key]:
                raise ValueError("{} no longer matches the sha256 this summary was built "
                                 "from".format(path))
        with open(os.path.join(entry["run_dir"], "meta.json")) as fin:
            meta = _json.load(fin)
        if meta.get("per_sample_sha256") != entry["per_sample_sha256"]:
            raise ValueError(
                "{}: meta.per_sample_sha256 is {!r}, not the {} this summary was built "
                "from; the run no longer binds its per-sample file".format(
                    entry["run_dir"], meta.get("per_sample_sha256"),
                    entry["per_sample_sha256"]))
    return True


def figure_data(summary):
    """The rows that the figures and the CSV both draw from (one per arm, angle, panel)."""
    rows = []
    for arm in summary["arms"]:
        band = arm.get("historical_band") or {}
        for k in sorted(int(k) for k in arm["angles"]):
            if k == 0:
                continue                      # k = 0 is the paired reference, not a bar
            for metric in FIGURE_PANELS:
                cell = arm["angles"][str(k)].get(metric)
                if cell is None:
                    continue
                unit, scale = FIGURE_SCALE[metric]
                query, ratio = cell["query"], cell["query"]["ratio"]

                def _scaled(value):
                    return None if value is None else float(value) * scale

                rows.append({
                    "arm": arm["arm"], "k": int(k),
                    "angle_deg": 360.0 * int(k) / 512.0, "metric": metric, "unit": unit,
                    "degradation": _scaled(query["delta"]["point"]),
                    "degradation_lo": _scaled(query["delta"]["lo"]),
                    "degradation_hi": _scaled(query["delta"]["hi"]),
                    "shift": _scaled(query["gap"]["point"]),
                    "shift_lo": _scaled(query["gap"]["lo"]),
                    "shift_hi": _scaled(query["gap"]["hi"]),
                    "n_mask": int(query["n"]), "ratio_status": ratio["status"],
                    "ratio_status_room": cell["room"]["ratio"]["status"],
                    "ratio_point": ratio["point"], "ratio_lo": ratio["lo"],
                    "ratio_hi": ratio["hi"],
                    "headline_reportable": cell["headline"]["reportable"],
                    "headline_reason": cell["headline"]["reason"],
                    "converged": cell["convergence"]["converged"],
                    "band": band.get(metric)})
    return rows


def _pyplot():
    """matplotlib's pyplot on a headless backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _panel(ax, rows, metric, band, annotate_n=True):
    """One panel: signed change and shift per angle, with CIs and the context band."""
    selected = [row for row in rows if row["metric"] == metric]
    positions = list(range(len(selected)))
    width = 0.38
    handles = []

    def _bars(offset, value_key, color, label):
        values = [row[value_key] or 0.0 for row in selected]
        lo = [max(0.0, (row[value_key] or 0.0) - (row[value_key + "_lo"] or 0.0))
              for row in selected]
        hi = [max(0.0, (row[value_key + "_hi"] or 0.0) - (row[value_key] or 0.0))
              for row in selected]
        return ax.bar([p + offset for p in positions], values, width, color=color,
                      label=label, yerr=[lo, hi], capsize=2,
                      error_kw={"elinewidth": 0.8, "ecolor": "#3a3a3a"})

    paired = any(row["degradation"] is not None for row in selected)
    if paired:
        handles.append(_bars(-width / 2, "degradation", COLOR_DEGRADATION,
                             "accuracy change (vs GT)"))
        handles.append(_bars(width / 2, "shift", COLOR_SHIFT,
                             r"prediction shift (vs $P_0$)"))
    else:
        handles.append(_bars(0.0, "shift", COLOR_SHIFT,
                             r"prediction shift (vs $P_0$)"))

    if band:
        band_handle = ax.axhspan(-band, band, color="0.88", zorder=0, label=BAND_LABEL)
        handles.append(band_handle)
    ax.axhline(0.0, color="0.3", linewidth=0.6, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(["{:.0f}°\n{}".format(row["angle_deg"],
                                             STATUS_CODES.get(row["ratio_status"], "")
                                             if paired else "")
                        for row in selected], fontsize=7)
    ax.set_ylabel("({})".format(selected[0]["unit"]) if selected else "", fontsize=7)
    ax.set_title(PANEL_TITLES[metric], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.margins(y=0.28)
    if annotate_n:
        counts = sorted({row["n_mask"] for row in selected})
        text = "n = {}".format(counts[0]) if len(counts) == 1 else \
            "n = {}-{}".format(counts[0], counts[-1])
        ax.annotate(text, (0.02, 0.94), xycoords="axes fraction", fontsize=6,
                    color="#555555", va="top")
    return handles


def make_figure(summary, arm_name, path):
    """The FLAC-style four-panel figure for one arm (T60, C50, EDT, GL-free shift)."""
    plt = _pyplot()
    rows = [row for row in figure_data(summary) if row["arm"] == arm_name]
    if not rows:
        raise ValueError("no figure rows for arm {!r}".format(arm_name))
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    figure, axes = plt.subplots(1, len(FIGURE_PANELS), figsize=(11.0, 3.1),
                                constrained_layout=True)
    handles = []
    for ax, metric in zip(axes, FIGURE_PANELS):
        band = next((row["band"] for row in rows if row["metric"] == metric), None)
        drawn = _panel(ax, rows, metric, band)
        handles = handles or drawn
    figure.suptitle("exp_10 yaw pilot -- {}".format(arm_name), fontsize=10)
    _figure_legend(figure, handles)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    return figure


def _figure_legend(figure, handles):
    """One legend and one status-code footnote for the whole figure."""
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                  fontsize=6.5, bbox_to_anchor=(0.5, -0.09))
    figure.text(0.5, -0.14, STATUS_FOOTNOTE, ha="center", fontsize=6, color="#555555")


def make_combined_figure(summary, path):
    """One row of panels per arm, so the arms can be read against each other."""
    plt = _pyplot()
    rows = figure_data(summary)
    arms = [arm["arm"] for arm in summary["arms"]]
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})
    figure, axes = plt.subplots(len(arms), len(FIGURE_PANELS),
                                figsize=(11.0, 3.0 * len(arms)), squeeze=False,
                                constrained_layout=True)
    handles = []
    for row_index, arm in enumerate(arms):
        arm_rows = [row for row in rows if row["arm"] == arm]
        for column, metric in enumerate(FIGURE_PANELS):
            band = next((row["band"] for row in arm_rows if row["metric"] == metric),
                        None)
            ax = axes[row_index][column]
            drawn = _panel(ax, arm_rows, metric, band)
            handles = handles or drawn
            if column == 0:
                ax.set_ylabel("{}\n{}".format(arm, ax.get_ylabel()), fontsize=7)
            if row_index:
                ax.set_title("")
    _figure_legend(figure, handles)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    return figure


CSV_COLUMNS = ("arm", "k", "angle_deg", "metric", "unit", "degradation",
               "degradation_lo", "degradation_hi", "shift", "shift_lo", "shift_hi",
               "n_mask", "ratio_status", "ratio_status_room", "ratio_point", "ratio_lo",
               "ratio_hi", "headline_reportable", "headline_reason", "converged", "band")


def _markdown_tables(summary):
    """The Markdown report: Metric 2, Metric 1, the ratio, room intervals and controls."""
    lines = ["# exp_10 `yaw_pilot` -- descriptive summary", "",
             "{} bootstrap resamples, {:.0f} % percentile intervals, seeds {} "
             "(seed {} reports, seed {} checks convergence).".format(
                 summary["n_boot"], 100 * (1 - summary["alpha"]), summary["seeds"],
                 summary["seeds"][0], summary["seeds"][1]), "",
             "The grey band in the figures is the {}.".format(summary["band_label"]),
             ""]
    for arm in summary["arms"]:
        lines.extend(["## {}".format(arm["arm"]), "",
                      "`{}` -- {} queries, {} rooms, execution `{}`.".format(
                          arm["run_dir"], arm["n_queries"], len(arm["rooms"]),
                          arm["execution_id"]), "",
                      "### Metric 2 -- accuracy against the ground truth", "",
                      "| angle | metric | mean k=0 | mean k | delta | 95 % CI | n_mask | "
                      "excluded (0 / alpha / gap) |", "|---|---|---|---|---|---|---|---|"])
        for k in sorted(int(k) for k in arm["angles"]):
            for label, _, _, _ in METRIC_TRIPLES:
                cell = arm["angles"][str(k)][label]["query"]
                exclusions = cell["exclusions"]
                lines.append("| {}° | {} | {} | {} | {} | [{}, {}] | {} | {} / {} / {} |"
                             .format(_deg(k), label, _fmt(cell["mean_0"]),
                                     _fmt(cell["mean_alpha"]), _fmt(cell["delta"]["point"]),
                                     _fmt(cell["delta"]["lo"]), _fmt(cell["delta"]["hi"]),
                                     cell["n"], exclusions["invalid_at_0"],
                                     exclusions["invalid_at_alpha"],
                                     exclusions["gap_invalid"]))
        lines.extend(["", "### Metric 1 -- how far the prediction moved", "",
                      "| angle | metric | G | 95 % CI | n |", "|---|---|---|---|---|"])
        for k in sorted(int(k) for k in arm["angles"]):
            for label in [triple[0] for triple in METRIC_TRIPLES] + \
                    [name for name, _ in SHIFT_ONLY_METRICS]:
                cell = arm["angles"][str(k)].get(label)
                if cell is None:
                    continue
                gap = cell["query"]["gap"]
                lines.append("| {}° | {} | {} | [{}, {}] | {} |".format(
                    _deg(k), label, _fmt(gap["point"]), _fmt(gap["lo"]), _fmt(gap["hi"]),
                    cell["query"]["n"]))
        lines.extend(["", "### R2 -- the ratio G / delta, with its status", "",
                      "| angle | metric | R (query) | 95 % CI | status (query) | "
                      "status (room) | converged | headline |",
                      "|---|---|---|---|---|---|---|---|"])
        for k in sorted(int(k) for k in arm["angles"]):
            for label, _, _, _ in METRIC_TRIPLES:
                cell = arm["angles"][str(k)][label]
                ratio = cell["query"]["ratio"]
                headline = cell["headline"]
                lines.append("| {}° | {} | {} | [{}, {}] | {} | {} | {} | {} |".format(
                    _deg(k), label, _fmt(ratio["point"]), _fmt(ratio["lo"]),
                    _fmt(ratio["hi"]), ratio["status"], cell["room_status"],
                    cell["convergence"]["converged"],
                    "yes" if headline["reportable"] else (headline["reason"] or "-")))
        lines.extend(["", "### Room-cluster intervals (secondary)", "",
                      "| angle | metric | delta CI (room) | G CI (room) | status | "
                      "rooms |", "|---|---|---|---|---|---|"])
        for k in sorted(int(k) for k in arm["angles"]):
            for label, _, _, _ in METRIC_TRIPLES:
                cell = arm["angles"][str(k)][label]["room"]
                lines.append("| {}° | {} | [{}, {}] | [{}, {}] | {} | {} |".format(
                    _deg(k), label, _fmt(cell["delta"]["lo"]), _fmt(cell["delta"]["hi"]),
                    _fmt(cell["gap"]["lo"]), _fmt(cell["gap"]["hi"]),
                    cell["ratio"]["status"], cell.get("n_clusters")))
        controls = arm.get("controls", {})
        lines.extend(["", "### Controls", "",
                      "| control | k | repeats | max |dwave| | max |dlogspec| | "
                      "max |dacoustic| | validity mismatches | ok |",
                      "|---|---|---|---|---|---|---|---|"])
        for name, cell in sorted(controls.get("controls", {}).items()):
            lines.append("| {} | {} | k={} | {:.3e} | {:.3e} | {:.3e} | {} | {} |".format(
                name, cell["k"], cell["compare_to"], cell["wave_max_abs_diff"],
                cell["logspec_max_abs_diff"], cell["acoustic_max_abs_diff"],
                cell["validity_mismatches"], "yes" if cell["ok"] else "NO"))
        lines.append("")
    return "\n".join(lines) + "\n"


def _deg(k):
    """A column roll as whole degrees."""
    return "{:.0f}".format(360.0 * int(k) / 512.0)


def _fmt(value, digits=6):
    """A number for a Markdown cell (``-`` when it is withheld)."""
    if value is None:
        return "-"
    return "{:.{}g}".format(float(value), digits)


def write_outputs(summary, out_dir):
    """Write the canonical JSON, the Markdown tables, the CSV and every figure.

    The inputs are re-verified first: a summary whose per-sample files have changed on
    disk since it was computed is refused rather than published.

    Returns:
        ``{"json", "markdown", "csv", "figures"}`` -- the paths written.
    """
    import csv

    verify_inputs(summary)
    os.makedirs(out_dir, exist_ok=True)
    plt = _pyplot()

    json_path = os.path.join(out_dir, "yaw_pilot_summary.json")
    import json as _json

    with open(json_path, "w") as fout:
        _json.dump(_sanitize(summary), fout, indent=1, allow_nan=False, sort_keys=True)

    markdown_path = os.path.join(out_dir, "yaw_pilot_tables.md")
    with open(markdown_path, "w") as fout:
        fout.write(_markdown_tables(summary))

    csv_path = os.path.join(out_dir, "yaw_pilot_gaps.csv")
    rows = figure_data(summary)
    with open(csv_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})

    figures = []
    for arm in summary["arms"]:
        base = os.path.join(out_dir, "yaw_pilot_gaps_{}".format(arm["arm"]))
        figure = make_figure(summary, arm["arm"], base + ".png")
        figure.savefig(base + ".pdf", bbox_inches="tight")
        plt.close(figure)
        figures.extend([base + ".png", base + ".pdf"])
    combined = os.path.join(out_dir, "yaw_pilot_gaps_all_arms")
    figure = make_combined_figure(summary, combined + ".png")
    figure.savefig(combined + ".pdf", bbox_inches="tight")
    plt.close(figure)
    figures.extend([combined + ".png", combined + ".pdf"])

    return {"json": json_path, "markdown": markdown_path, "csv": csv_path,
            "figures": figures}


def main(argv=None):
    """Summarise one or more run directories into the pilot's descriptive outputs."""
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="one exp_10 run directory per arm, in reporting order")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--seeds", type=int, nargs=2, default=list(SEEDS))
    args = parser.parse_args(argv)

    summary = build_summary(args.runs, n_boot=args.n_boot, alpha=args.alpha,
                            seeds=tuple(args.seeds))
    written = write_outputs(summary, args.out)
    for name, path in sorted(written.items()):
        print("{}: {}".format(name, path if isinstance(path, str) else len(path)))
    return summary


if __name__ == "__main__":
    main()
