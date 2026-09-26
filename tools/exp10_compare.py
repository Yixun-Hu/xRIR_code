"""Offline comparator for the exp_10 ``yaw_pilot`` runs.

Three jobs, all of them refusals first and numbers second:

* :func:`verify_exp03_pins` -- the twelve files exp_03's record pinned must still hash to
  their reviewed blobs before anything in this experiment is believed;
* :func:`check_online` -- the **waveform and acoustic** halves of Metric 1, recomputed
  from the stored ``wav_k<k>.npy`` arrays, must equal what the run recorded online (the
  spectral halves are computed from the model's direct output and are not recoverable
  from a finite-iteration Griffin-Lim waveform, so they are excluded by contract);
* :func:`parity_exp03` -- the run's rows, aligned to exp_03's historical per-sample file
  by canonical index, with the per-sample and paired differences that the plan's parity
  criteria are read off (this module reports them; it does not decide a launch).

    python tools/exp10_compare.py check-online <run-dir>
    python tools/exp10_compare.py parity-exp03 <run-dir> <exp03-per-sample.json>
    python tools/exp10_compare.py verify-pins [binding-report.json]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from eval_unseen import Evaluator
from tools.exp10_yaw_pilot import (
    acoustic_gap,
    file_sha256,
    query_list_sha256,
    waveform_gap,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BINDING_REPORT = os.path.join(REPO_ROOT, "ckpt", "yaw_rotation",
                                      "binding_report.json")
ONLINE_TOLERANCE = 1e-6
# What an offline pass over the stored waveforms can and cannot re-derive.
OFFLINE_METRICS = ("wave_rel_l2", "wave_mad", "edt_gap", "c50_gap", "t60_gap")
ONLINE_ONLY_METRICS = ("logspec_mad", "mag_rel_l2")

#: exp_10 metric -> exp_03 per-sample key, with the plan's section 7 parity criteria.
#: ``per_sample_tol`` / ``min_fraction`` are criteria (a) and (b); ``paired_abs`` with a
#: 5 % relative allowance is criterion (c).  Metrics with no thresholds are reported only.
PARITY_METRICS = {
    "logspec_mad": {"exp03": "consistency", "per_sample_tol": 1e-5, "min_fraction": 1.0,
                    "paired_abs": None, "unit": "log-magnitude"},
    "edt_err": {"exp03": "edt", "per_sample_tol": 1e-4, "min_fraction": 0.995,
                "paired_abs": 2e-5, "unit": "s"},
    "c50_err": {"exp03": "c50", "per_sample_tol": 1e-3, "min_fraction": 0.995,
                "paired_abs": 2e-4, "unit": "dB"},
    "t60_err": {"exp03": "t60", "per_sample_tol": 0.01, "min_fraction": 0.995,
                "paired_abs": 2e-3, "unit": "% of T60(GT)"},
    "log_mse": {"exp03": "log_mse"},
    "loss": {"exp03": "loss"},
    "stft": {"exp03": "stft"},
    "decay": {"exp03": "decay"},
}
PAIRED_RELATIVE_ALLOWANCE = 0.05


def verify_exp03_pins(binding_report=DEFAULT_BINDING_REPORT, repo_root=REPO_ROOT):
    """Re-verify exp_03's pinned source closure against the live tree.

    exp_03's record bound twelve files (the evaluator, the rotation and metric tools, the
    models, the dataset and the loss module) to the blobs a reviewer signed off at commit
    ``62c9107b…``.  exp_10 reuses those files unchanged, so this experiment's numbers only
    mean what they claim if the files still hash to the reviewed blobs;
    ``tests/test_exp03_record_tools.py`` exercises the *validators* on synthetic records,
    which is a different question.

    Args:
        binding_report: path to exp_03's ``binding_report.json``.
        repo_root: the tree whose files are hashed (the repository root).

    Returns:
        ``{"ok", "binding_report", "binding_report_sha256", "reviewed_commit", "n_files",
        "files"}``; each file entry carries ``path``, ``reviewed_blob_sha256``,
        ``live_sha256`` (``None`` when the file is missing) and ``match``.

    Raises:
        ValueError: if any pinned file is missing or no longer hashes to its reviewed
            blob -- the run refuses to start rather than report numbers from a changed
            closure.
    """
    with open(binding_report) as fin:
        report = json.load(fin)
    files = []
    for entry in report["source_closure"]:
        relative = entry["path"]
        path = os.path.join(repo_root, relative)
        live = file_sha256(path) if os.path.isfile(path) else None
        files.append({"path": relative,
                      "reviewed_blob_sha256": entry["reviewed_blob_sha256"],
                      "live_sha256": live,
                      "match": live == entry["reviewed_blob_sha256"]})
    broken = [entry["path"] for entry in files if not entry["match"]]
    if broken:
        raise ValueError(
            "exp_03's pinned source closure no longer matches its reviewed blobs "
            "({} of {} files): {}".format(len(broken), len(files), ", ".join(broken)))
    return {"ok": True,
            "binding_report": os.path.abspath(binding_report),
            "binding_report_sha256": file_sha256(binding_report),
            "reviewed_commit": report.get("reviewed_commit"),
            "n_files": len(files),
            "files": files}


def _as_array(values):
    """One stored per-sample list (``null`` for invalid) as a float64 array with NaN."""
    return np.asarray([np.nan if value is None else float(value) for value in values],
                      dtype=np.float64)


def load_run(run_dir, require_complete=True):
    """Load one run directory and refuse it unless its own identity holds.

    The guard is what lets everything downstream speak about "the run": the per-sample
    file must be the one this meta describes (same ``protocol_id`` **and**
    ``execution_id``, same configuration field by field), the meta must say the run
    finished, the waveform arrays on disk must still hash to what was recorded, and the
    query list must hash to ``query_list_sha256`` -- so a reordered, truncated or
    re-run-and-overwritten output cannot be read as if it were intact.

    Args:
        run_dir: the ``--out-dir`` of a ``tools/exp10_yaw_pilot.py`` run.
        require_complete: refuse a run whose ``meta.complete`` is not ``True``.

    Returns:
        ``(meta, per_sample)``.

    Raises:
        ValueError: on any inconsistency, naming the field or file that failed.
    """
    with open(os.path.join(run_dir, "meta.json")) as fin:
        meta = json.load(fin)
    with open(os.path.join(run_dir, "per_sample.json")) as fin:
        per_sample = json.load(fin)

    if require_complete and meta.get("complete") is not True:
        raise ValueError("{}: meta.complete is {!r}; the run did not finish".format(
            run_dir, meta.get("complete")))
    for field in ("protocol_id", "execution_id"):
        if per_sample.get(field) != meta.get(field):
            raise ValueError("{}: per_sample.{} {!r} != meta.{} {!r}".format(
                run_dir, field, per_sample.get(field), field, meta.get(field)))
    for field in ("checkpoint_sha256", "manifest_hash", "gl_seed", "num_shot", "device",
                  "batch_size", "batch_canonical", "ks", "arrays", "query_list_sha256"):
        if field not in meta:
            raise ValueError("{}: meta is missing the guard field {}".format(run_dir, field))
    embedded = per_sample.get("meta", {})
    for field, value in embedded.items():
        if meta.get(field) != value:
            raise ValueError("{}: per_sample.meta.{} {!r} != meta.{} {!r}".format(
                run_dir, field, value, field, meta.get(field)))

    queries = per_sample["query"]
    if query_list_sha256(queries) != meta["query_list_sha256"]:
        raise ValueError("{}: the query list does not match meta.query_list_sha256"
                         .format(run_dir))
    if len(queries) != int(meta["n_queries"]):
        raise ValueError("{}: {} queries but meta.n_queries is {}".format(
            run_dir, len(queries), meta["n_queries"]))
    for name, entry in meta["arrays"].items():
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            raise ValueError("{}: the waveform array {} is missing".format(run_dir, name))
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("{}: {} no longer matches its recorded sha256".format(
                run_dir, name))
        if list(entry["shape"]) != [len(queries), int(meta["padded_len"])]:
            raise ValueError("{}: {} has shape {} for {} queries".format(
                run_dir, name, entry["shape"], len(queries)))
    stored_angles = sorted(int(k) for k in per_sample["angles"])
    if stored_angles != sorted(int(k) for k in meta["ks"]):
        raise ValueError("{}: per_sample angles {} != meta.ks {}".format(
            run_dir, stored_angles, meta["ks"]))
    return meta, per_sample


def check_online(run_dir, tol=ONLINE_TOLERANCE, evaluator=None):
    """Recompute Metric 1's waveform and acoustic halves from the stored arrays.

    The run records its metrics while the model is still in memory; this recomputes them
    from ``wav_k<k>.npy`` alone and compares.  Agreement means the arrays that were saved
    really are the arrays the numbers describe -- the property every later offline
    analysis rests on.

    The spectral halves (``logspec_mad``, ``mag_rel_l2``) are **excluded by contract**:
    they are computed from the model's direct log-magnitude output, and Griffin-Lim is a
    finite-iteration nonlinear inverse, so no pass over the waveforms can reproduce them.
    They are retained and hashed in ``per_sample.json`` instead.

    Args:
        run_dir: the run directory (checked by :func:`load_run` first).
        tol: absolute agreement tolerance (the plan's 1e-6).
        evaluator: an ``eval_unseen.Evaluator`` (built here when omitted).

    Returns:
        ``{"ok", "run_dir", "tolerance", "excluded_by_contract", "angles"}`` where
        ``angles[str(k)][metric]`` carries ``max_abs_diff``, ``mean_abs_diff``,
        ``validity_mismatches``, ``n`` and ``ok``.

    Raises:
        ValueError: if the run fails :func:`load_run`.
    """
    meta, per_sample = load_run(run_dir)
    evaluator = evaluator or Evaluator()
    waves = {int(k): np.load(os.path.join(run_dir, "wav_k{}.npy".format(int(k))))
             for k in meta["ks"]}
    base = waves[0]

    report = {"ok": True, "run_dir": os.path.abspath(run_dir), "tolerance": float(tol),
              "excluded_by_contract": {"spectral": list(ONLINE_ONLY_METRICS),
                                       "reason": "computed from the model's direct output; "
                                                 "not recoverable from Griffin-Lim waveforms"},
              "angles": {}}
    for k in sorted(waves):
        offline = {}
        offline.update(waveform_gap(base, waves[k]))
        offline.update(acoustic_gap(base, waves[k], evaluator))
        cell = {}
        for name in OFFLINE_METRICS:
            recomputed = offline[name]
            stored = _as_array(per_sample["angles"][str(k)][name])
            invalid_offline = ~np.isfinite(recomputed)
            invalid_stored = ~np.isfinite(stored)
            mismatched = int(np.count_nonzero(invalid_offline != invalid_stored))
            both_valid = ~(invalid_offline | invalid_stored)
            diffs = np.abs(recomputed[both_valid] - stored[both_valid])
            max_diff = float(diffs.max()) if diffs.size else 0.0
            ok = bool(mismatched == 0 and max_diff <= float(tol))
            cell[name] = {"max_abs_diff": max_diff,
                          "mean_abs_diff": float(diffs.mean()) if diffs.size else 0.0,
                          "validity_mismatches": mismatched,
                          "n": int(both_valid.sum()), "ok": ok}
            report["ok"] = report["ok"] and ok
        report["angles"][str(k)] = cell
    return report


def _align_rows(meta, per_sample, history):
    """Map the run's canonical indices onto rows of exp_03's historical file.

    Refuses everything that would silently compare different queries: a different
    manifest, a run that is not in canonical order, an index the history does not carry,
    a duplicated index on either side, and any row whose query string does not match.
    """
    history_hash = history.get("meta", {}).get("manifest_hash")
    if history_hash != meta["manifest_hash"]:
        raise ValueError("the historical file's manifest {} is not the run's manifest {}"
                         .format(history_hash, meta["manifest_hash"]))

    run_index = [int(i) for i in per_sample["index"]]
    if any(b <= a for a, b in zip(run_index, run_index[1:])):
        raise ValueError("the run's indices are not in canonical (increasing) order")

    history_index = [int(i) for i in history["index"]]
    if len(set(history_index)) != len(history_index):
        raise ValueError("the historical file lists a duplicate canonical index")
    position = {index: row for row, index in enumerate(history_index)}

    rows = []
    for index in run_index:
        if index not in position:
            raise ValueError("canonical index {} is missing from the historical file"
                             .format(index))
        rows.append(position[index])
    if len(set(rows)) != len(rows):
        raise ValueError("the run maps two queries onto the same historical row "
                         "(duplicate index)")
    extracted = [history["query"][row] for row in rows]
    if extracted != list(per_sample["query"]):
        first = next(i for i, (a, b) in enumerate(zip(extracted, per_sample["query"]))
                     if a != b)
        raise ValueError("historical query {!r} at canonical index {} is not the run's "
                         "query {!r}".format(extracted[first], run_index[first],
                                             per_sample["query"][first]))
    return rows


def _parity_cell(run_k, run_0, hist_k, hist_0, rules, is_zero_angle):
    """One angle x metric parity cell: masks, per-sample spread and the paired deltas."""
    mask_run = np.isfinite(run_k) & np.isfinite(run_0)
    mask_hist = np.isfinite(hist_k) & np.isfinite(hist_0)
    mask_identical = bool(np.array_equal(mask_run, mask_hist))
    mismatches = int(np.count_nonzero(mask_run != mask_hist))
    common = mask_run & mask_hist

    diffs = np.abs(run_k[common] - hist_k[common])
    n = int(common.sum())
    quantiles = {}
    for label, q in (("p50", 50), ("p90", 90), ("p99", 99)):
        quantiles[label] = float(np.percentile(diffs, q)) if n else 0.0
    quantiles["max"] = float(diffs.max()) if n else 0.0

    paired_run = float((run_k[common] - run_0[common]).mean()) if n else float("nan")
    paired_hist = float((hist_k[common] - hist_0[common]).mean()) if n else float("nan")
    signs = (np.sign(run_k[common] - run_0[common]) ==
             np.sign(hist_k[common] - hist_0[common]))
    cell = {"n_common": n, "n_run_valid": int(mask_run.sum()),
            "n_exp03_valid": int(mask_hist.sum()), "mask_identical": mask_identical,
            "validity_mismatches": mismatches,
            "max_abs_diff": quantiles["max"],
            "mean_abs_diff": float(diffs.mean()) if n else 0.0,
            "quantiles": quantiles,
            "sign_agreement": float(signs.mean()) if n else float("nan"),
            "paired_delta_run": paired_run, "paired_delta_exp03": paired_hist,
            "paired_delta_diff": paired_run - paired_hist if n else float("nan"),
            "paired_same_sign": bool(np.sign(paired_run) == np.sign(paired_hist)) if n
                                else False}

    tol = rules.get("per_sample_tol")
    if tol is None:
        cell.update({"status": "reported", "criterion_a": None, "criterion_b": None,
                     "criterion_c": None, "fraction_within_tolerance": None})
        return cell

    within = int(np.count_nonzero(diffs <= tol))
    fraction = float(within) / n if n else 0.0
    cell["per_sample_tol"] = tol
    cell["n_within_tolerance"] = within
    cell["fraction_within_tolerance"] = fraction
    passes_per_sample = bool(n > 0 and fraction >= rules["min_fraction"])
    if rules["min_fraction"] >= 1.0:
        cell["criterion_a"], cell["criterion_b"] = passes_per_sample, None
    else:
        cell["criterion_a"], cell["criterion_b"] = None, passes_per_sample

    paired_abs = rules.get("paired_abs")
    if paired_abs is None or is_zero_angle:
        cell["criterion_c"] = None if paired_abs is None else True
    else:
        allowance = max(PAIRED_RELATIVE_ALLOWANCE * abs(paired_hist), paired_abs)
        cell["paired_allowance"] = float(allowance)
        cell["criterion_c"] = bool(abs(cell["paired_delta_diff"]) <= allowance and
                                   cell["paired_same_sign"])

    decided = [value for value in (cell["criterion_a"], cell["criterion_b"],
                                   cell["criterion_c"]) if value is not None]
    if not mask_identical:
        cell["status"] = "nonreplication (validity mismatch)"
    elif all(decided):
        cell["status"] = "replication"
    else:
        cell["status"] = "nonreplication (tolerance)"
    return cell


def parity_exp03(run_dir, exp03_per_sample, condition="P"):
    """Compare one run against exp_03's historical per-sample file, row by row.

    The run's rows are located in the historical file **by canonical index**, and the
    extracted query strings must equal the run's in order -- a subset is fine (the probe
    is 272 of 6337 rows), a permutation or a near-miss is not.  For every angle and metric
    the report carries the per-query spread (max / mean / p50 / p90 / p99), the sign
    agreement, the paired delta from *both* sources and the validity masks, plus the
    plan's criteria (a) per-query tolerance for ``logspec_mad``, (b) the 99.5 % tolerance
    for the acoustic errors and (c) the agreement of the paired delta itself.

    Amendment A1: a cell can only be called a replication if the two validity masks are
    **identical** on the compared population; otherwise it is labelled
    ``"nonreplication (validity mismatch)"`` however well the surviving numbers agree.

    This function reports; it does not decide a launch.

    Args:
        run_dir: the run directory.
        exp03_per_sample: path to exp_03's ``per_sample_yaw.json``.
        condition: which exp_03 condition to read (``"P"``; ``"E"`` exists in the file but
            exp_10 only runs P).

    Returns:
        A report dict; ``ok`` is True when every compared cell replicates.

    Raises:
        ValueError: on a manifest mismatch, a non-canonical run order, or a missing,
            duplicated or mismatched query.
    """
    meta, per_sample = load_run(run_dir)
    with open(exp03_per_sample) as fin:
        history = json.load(fin)
    rows = _align_rows(meta, per_sample, history)

    report = {"ok": True, "run_dir": os.path.abspath(run_dir),
              "exp03_path": os.path.abspath(exp03_per_sample),
              "exp03_sha256": file_sha256(exp03_per_sample),
              "manifest_hash": meta["manifest_hash"], "condition": condition,
              "n_rows": len(rows), "angles": {},
              "angles_missing_in_exp03": [], "metrics_missing_in_exp03": []}

    historical = history[condition]
    if "0" not in historical:
        raise ValueError("the historical file has no k = 0 cell to pair against")
    for k in sorted(int(k) for k in meta["ks"]):
        if str(k) not in historical:
            report["angles_missing_in_exp03"].append(k)
            continue
        cell = {}
        for metric, rules in PARITY_METRICS.items():
            key = rules["exp03"]
            if (key not in historical[str(k)] or key not in historical["0"] or
                    metric not in per_sample["angles"][str(k)]):
                report["metrics_missing_in_exp03"].append({"k": k, "metric": metric})
                continue
            run_k = _as_array(per_sample["angles"][str(k)][metric])
            run_0 = _as_array(per_sample["angles"]["0"][metric])
            hist_k = _as_array(historical[str(k)][key])[rows]
            hist_0 = _as_array(historical["0"][key])[rows]
            cell[metric] = _parity_cell(run_k, run_0, hist_k, hist_0, rules, k == 0)
            if cell[metric]["status"].startswith("nonreplication"):
                report["ok"] = False
        report["angles"][str(k)] = cell
    return report


def _print_json(report, path=None):
    """Print a report (and optionally save it) without losing NaN to strict JSON."""
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(text)
    if path:
        with open(path, "w") as fout:
            fout.write(text + "\n")
        print("wrote {}".format(path))
    return report


def main(argv=None):
    """Parse the CLI and dispatch to one of the comparator's commands."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    pins = sub.add_parser("verify-pins", help="re-verify exp_03's pinned closure")
    pins.add_argument("binding_report", nargs="?", default=DEFAULT_BINDING_REPORT)

    online = sub.add_parser("check-online",
                            help="recompute Metric 1's waveform and acoustic halves")
    online.add_argument("run_dir")
    online.add_argument("--tolerance", type=float, default=ONLINE_TOLERANCE)
    online.add_argument("--json", dest="json_path", default=None)

    parity = sub.add_parser("parity-exp03", help="compare a run with exp_03's per-sample file")
    parity.add_argument("run_dir")
    parity.add_argument("exp03_per_sample")
    parity.add_argument("--condition", default="P", choices=("P", "E"))
    parity.add_argument("--json", dest="json_path", default=None)

    args = parser.parse_args(argv)

    if args.command == "verify-pins":
        report = verify_exp03_pins(args.binding_report)
        print("exp_03 pins OK: {} files at reviewed commit {}".format(
            report["n_files"], report["reviewed_commit"]))
        return report
    if args.command == "check-online":
        report = check_online(args.run_dir, tol=args.tolerance)
        _print_json(report, args.json_path)
        print("check_online: {}".format("OK" if report["ok"] else "MISMATCH"))
        return report
    if args.command == "parity-exp03":
        report = parity_exp03(args.run_dir, args.exp03_per_sample,
                              condition=args.condition)
        _print_json(report, args.json_path)
        print("parity_exp03: {} rows, every compared cell replicates: {}".format(
            report["n_rows"], report["ok"]))
        return report
    raise ValueError("unknown command {!r}".format(args.command))


if __name__ == "__main__":
    main()
