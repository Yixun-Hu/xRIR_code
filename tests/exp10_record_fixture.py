"""Synthetic exp_10 record inputs for the Planner tooling tests (no model, no GPU, no torch).

The record tools -- ``validate_runs.py``, the two renderers, the backend table and the four
shell scripts -- read only JSON, small ``.npy`` arrays and the summariser's output files.
This builder writes exactly those, with the same *bindings* a real run carries (the
per-sample file hashes to ``meta.per_sample_sha256``, each waveform array hashes to its
``meta.arrays`` entry, the canonical summary's ``inputs`` hash to the run they came from),
so a test can break one binding and check that the tool refuses it.

Nothing here imports ``tools.exp10_*``: the fixtures describe the *file format* the record
tools consume, and a test that shared its writer with the reader could not detect a reader
that trusts fields it should verify.
"""
import hashlib
import json
import os

import numpy as np

ARMS = ("released_k8", "released_k1", "control_k8", "cyl_k8")
PARITY_ARMS = ("released_k8", "control_k8", "cyl_k8")
KS = (0, 64, 128, 256, 384)
METRICS = ("EDT", "C50", "T60", "T60_abs", "logspec_mad")
NOTES = {"band": "historical baseline evaluation variability, context only",
         "broader_population": "every query with a finite g_alpha (not the paired mask)",
         "gl_free": "logspec_mad is the Griffin-Lim-free readout of the same shift",
         "pipeline": "waveform gaps measure the model-plus-Griffin-Lim pipeline"}
METRIC_QUALIFICATIONS = {
    "C50": {"delta": "change in the C50 error against the ground truth, dB",
            "gap": "C50 distance from the k = 0 prediction, dB"},
    "EDT": {"delta": "change in the EDT error against the ground truth, seconds",
            "gap": "EDT distance from the k = 0 prediction, seconds"},
    "T60": {"delta": "change in the GT-normalised error, percentage points",
            "gap": "percent of the baseline prediction's T60"},
    "T60_abs": {"delta": "change in |T60(prediction) - T60(GT)|, seconds",
                "gap": "|T60(P_alpha) - T60(P_0)|, seconds"}}
CHECKPOINTS = {"released_k8": ("checkpoints/xRIR_unseen.pth", "simple", 8),
               "released_k1": ("checkpoints/xRIR_unseen.pth", "simple", 1),
               "control_k8": ("ckpt/xRIR_simple_8_shot/epoch_12.pth", "simple", 8),
               "cyl_k8": ("ckpt/xRIR_cyl_8_shot/epoch_12.pth", "cylindrical", 8)}
EXP03 = {"released_k8": "ckpt/yaw_rotation/sweep_released/per_sample_yaw.json",
         "control_k8": "ckpt/yaw_rotation/sweep_control/per_sample_yaw.json",
         "cyl_k8": "ckpt/yaw_rotation/sweep_cyl/per_sample_yaw.json"}


def file_sha256(path, chunk=1 << 20):
    """sha256 of a file's bytes (the same digest every exp_10 tool records)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fin:
        for block in iter(lambda: fin.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    """Write ``obj`` as JSON, creating the parent directory."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w") as fout:
        json.dump(obj, fout, indent=2, sort_keys=True)
    return path


def read_json(path):
    with open(path) as fin:
        return json.load(fin)


def _hexish(seed, tag):
    return hashlib.sha256(("%s/%s" % (seed, tag)).encode("utf-8")).hexdigest()


def write_run(run_dir, arm="control_k8", device="cuda", n_queries=4, ks=KS,
              batches_arg="all", complete=True, execution_id=None, protocol_id=None,
              padded_len=16, meta_overrides=None, per_sample_overrides=None,
              query_list=None):
    """Write a self-consistent synthetic run directory and return its meta dict.

    Args:
        run_dir: directory to create.
        arm: the arm label the run claims.
        device: ``"cuda"`` or ``"cpu"``.
        n_queries: number of queries (the record's full runs have 6337).
        ks: the angles; every one of them gets a waveform array bound in ``meta.arrays``.
        batches_arg: ``"all"`` for a full run, ``"probe"`` for a probe.
        complete: value of ``meta.complete``.
        execution_id / protocol_id: override the derived identities (to build two runs
            that share one, or a report that names the wrong one).
        padded_len: waveform length of the synthetic arrays (real runs: 9600).
        meta_overrides: fields to overwrite in ``meta.json`` *after* the bindings are
            computed, so a test can corrupt exactly one of them.
        per_sample_overrides: fields to overwrite in ``per_sample.json``.
        query_list: override the canonical query list.

    Returns:
        The meta dict as written.
    """
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    checkpoint, backbone, num_shot = CHECKPOINTS.get(arm, ("checkpoints/xRIR_unseen.pth", "simple", 8))
    tag = "%s/%s/%s" % (arm, device, batches_arg)
    execution_id = execution_id or ("20260927T%06dZ-%s" % (n_queries, _hexish(tag, "exec")[:32]))
    protocol_id = protocol_id or _hexish(tag, "protocol")
    queries = list(query_list if query_list is not None else
                   ["Cat/Room_idx_%d/S%03d_R001_hybrid_IR.wav" % (i % 3, i) for i in range(n_queries)])
    rng = np.random.RandomState(abs(hash(tag)) % (2 ** 31))
    arrays = {}
    for k in ks:
        name = "wav_k%d.npy" % int(k)
        path = os.path.join(run_dir, name)
        np.save(path, rng.randn(n_queries, padded_len).astype(np.float32))
        arrays[name] = {"sha256": file_sha256(path), "shape": [n_queries, padded_len],
                        "dtype": "float32", "k": int(k)}
    per_sample = {"protocol_id": protocol_id, "execution_id": execution_id,
                  "query": queries,
                  "angles": {str(int(k)): {"edt_err": [0.0] * n_queries} for k in ks},
                  "meta": {"arm": arm, "device": device, "gl_seed": 0}}
    if per_sample_overrides:
        per_sample.update(per_sample_overrides)
    per_sample_path = write_json(os.path.join(run_dir, "per_sample.json"), per_sample)
    metrics_path = write_json(os.path.join(run_dir, "metrics.json"),
                              {"arm": arm, "angles": {str(int(k)): {} for k in ks}})
    meta = {"arm": arm, "backbone": backbone, "checkpoint": checkpoint,
            "checkpoint_sha256": _hexish(checkpoint, "checkpoint"),
            "manifest_path": "ckpt/yaw_rotation/reference_manifest.json",
            "manifest_hash": _hexish(num_shot, "manifest"), "manifest_seed": 0,
            "gl_seed": 0, "num_shot": num_shot, "device": device,
            "cudnn_deterministic": True, "cudnn_allow_tf32": False,
            "matmul_allow_tf32": False, "tf32": False, "batch_size": 16,
            "batch_canonical": True, "torch_version": "2.0.1+cu117",
            "numpy_version": "1.23.5", "tool_sha256": _hexish("tool", "tool"),
            "split": "unseen", "ks": [int(k) for k in ks], "controls": batches_arg == "probe",
            "control_specs": [], "batches_arg": batches_arg,
            "batches": list(range(max(1, n_queries // 16))),
            "n_batches_total": 397, "n_queries": n_queries, "n_split": 6337,
            "native_len": padded_len, "padded_len": padded_len, "sample_rate": 22050,
            "metric_window": 8000, "threads": 16, "num_workers": 6,
            "python_version": "3.8.20", "git_commit": _hexish("git", "commit")[:40],
            "git_dirty": False, "run_id": _hexish(tag, "run"),
            "protocol_id": protocol_id, "execution_id": execution_id,
            "started_at": execution_id.split("-")[0], "arrays": arrays,
            "query_list_sha256": hashlib.sha256("\n".join(queries).encode("utf-8")).hexdigest(),
            "per_sample_sha256": file_sha256(per_sample_path),
            "metrics_sha256": file_sha256(metrics_path),
            "timing": {"total_min": 1.0}, "complete": complete}
    if meta_overrides:
        meta.update(meta_overrides)
    write_json(os.path.join(run_dir, "meta.json"), meta)
    return meta


def write_check_online(run_dir, meta=None, ok=True, run_dir_value=None, ks=None, extra=None):
    """Write the comparator's ``check_online.json`` next to a run (fields as in the real one)."""
    ks = ks if ks is not None else (meta or {}).get("ks", KS)
    report = {"ok": ok, "run_dir": run_dir_value or os.path.abspath(run_dir),
              "tolerance": 1e-06,
              "excluded_by_contract": {"spectral": ["logspec_mad", "mag_rel_l2"],
                                       "reason": "not recoverable from Griffin-Lim waveforms"},
              "angles": {str(int(k)): {"edt_gap": {"ok": ok, "max_abs_diff": 0.0, "n": 4}}
                         for k in ks}}
    if extra:
        report.update(extra)
    return write_json(os.path.join(run_dir, "check_online.json"), report)


def write_parity(run_dir, meta=None, ok=True, run_dir_value=None, n_rows=None,
                 manifest_hash=None, arm=None, extra=None):
    """Write the comparator's ``parity_exp03.json`` next to a run."""
    meta = meta or {}
    arm = arm or meta.get("arm", "released_k8")
    report = {"ok": ok, "run_dir": run_dir_value or os.path.abspath(run_dir),
              "condition": "P", "n_rows": n_rows if n_rows is not None else meta.get("n_queries", 4),
              "exp03_path": os.path.abspath(EXP03.get(arm, EXP03["released_k8"])),
              "exp03_sha256": _hexish(arm, "exp03"),
              "manifest_hash": manifest_hash or meta.get("manifest_hash"),
              "required_metrics": ["edt_err", "c50_err", "t60_err", "logspec_mad"],
              "optional_metrics": ["log_mse", "loss"], "missing_required": [],
              "missing_optional": [], "angles_missing_in_exp03": [],
              "metrics_missing_in_exp03": [],
              "angles": {str(int(k)): {"edt_err": {"max_abs_diff": 0.0, "status": "ok",
                                                   "fraction_within_tolerance": 1.0,
                                                   "paired_delta_run": 0.0,
                                                   "paired_delta_exp03": 0.0}}
                         for k in meta.get("ks", KS)}}
    if extra:
        report.update(extra)
    return write_json(os.path.join(run_dir, "parity_exp03.json"), report)


def controls_table(run_dir, ok=True):
    """The summariser's ``controls`` block for one arm."""
    return {"ok": ok, "run_dir": run_dir,
            "declared_controls": ["ctrl_zero_repeat"], "missing_controls": [],
            "undeclared_controls": [],
            "tolerances": {"acoustic": 1e-09, "logspec": 1e-06, "waveform": 1e-06},
            "controls": {"ctrl_zero_repeat": {"k": 0, "compare_to": 0, "n": 4, "ok": ok,
                                              "wave_max_abs_diff": 0.0,
                                              "logspec_max_abs_diff": 0.0,
                                              "acoustic_max_abs_diff": 0.0,
                                              "nonfinite_deviations": 0, "problems": []}}}


def _ci(point, lo, hi):
    return {"point": point, "lo": lo, "hi": hi}


def make_cell(reportable=True, room_status="denominator uncertain", query_status="defined"):
    """One metric cell of one angle, in the canonical summariser shape."""
    headline = ({"reportable": True, "point": 17.8, "lower_bound": 8.81,
                 "upper_bound": 95.3, "reason": None, "status_seed_0": "defined",
                 "status_seed_1": "defined"} if reportable else
                {"reportable": False, "point": None, "lower_bound": None,
                 "upper_bound": None, "reason": "denominator uncertain",
                 "status_seed_0": "undefined", "status_seed_1": "undefined"})
    ratio = ({"point": 17.8, "lo": 8.81, "hi": 95.3, "status": query_status,
              "reason": None, "point_reason": None, "zero_draw_fraction": 0.0}
             if reportable else
             {"point": None, "lo": None, "hi": None, "status": query_status,
              "reason": "denominator uncertain", "point_reason": None,
              "zero_draw_fraction": 0.4})
    query = {"mean_0": 0.045, "mean_alpha": 0.053,
             "delta": _ci(0.0008, 0.0002, 0.0015), "gap": _ci(0.0142, 0.0131, 0.0154),
             "ratio": ratio, "n": 4,
             "exclusions": {"n_total": 4, "n_mask": 4, "excluded_total": 0,
                            "invalid_at_0": 0, "invalid_at_alpha": 0, "gap_invalid": 0},
             "contributions": {"fraction_worse": 0.6, "mean_positive_part": 0.002,
                               "mean_negative_part": -0.001}}
    room = {"delta": _ci(0.0008, -0.0002, 0.0019), "gap": _ci(0.0142, 0.0120, 0.0165),
            "ratio": dict(ratio), "n": 3}
    return {"query": query, "room": room, "headline": headline,
            "room_status": room_status,
            "broader": {"population": NOTES["broader_population"], "query": query},
            "convergence": {"converged": True}, "convergence_room": {"converged": True}}


def summary_arm(run_dir, meta, controls_ok=True, reportable=True):
    """One ``arms[]`` entry of the canonical summary, for a run written by ``write_run``."""
    angles = {}
    for k in meta["ks"]:
        if int(k) == 0:
            angles["0"] = {}
            continue
        angles[str(int(k))] = {m: make_cell(reportable=reportable and m in ("EDT", "C50"))
                               for m in METRICS}
    return {"arm": meta["arm"], "run_dir": run_dir, "meta": meta,
            "execution_id": meta["execution_id"], "protocol_id": meta["protocol_id"],
            "n_queries": meta["n_queries"], "angles": angles,
            "controls": controls_table(run_dir, ok=controls_ok),
            "historical_band": {"EDT": 0.186, "C50": 0.0053, "T60": 0.009},
            "historical_band_label": "historical baseline evaluation variability, context only",
            "rooms": ["Cat/Room_idx_0", "Cat/Room_idx_1", "Cat/Room_idx_2"]}


def build_summary(runs, controls_ok=True, reportable=True, notes=None,
                  metric_qualifications=None):
    """The canonical summariser dict over ``runs`` = [(run_dir, meta), ...]."""
    inputs = []
    for run_dir, meta in runs:
        inputs.append({"arm": meta["arm"], "run_dir": os.path.abspath(run_dir),
                       "execution_id": meta["execution_id"],
                       "protocol_id": meta["protocol_id"],
                       "per_sample_sha256": file_sha256(os.path.join(run_dir, "per_sample.json")),
                       "meta_sha256": file_sha256(os.path.join(run_dir, "meta.json"))})
    return {"tool": "tools/exp10_summarize.py", "generated_at": "2026-09-27T07:33:17Z",
            "n_boot": 10000, "alpha": 0.05, "seeds": [0, 1],
            "convergence_tolerance": 0.1,
            "band_label": NOTES["band"],
            "notes": dict(NOTES if notes is None else notes),
            "metric_qualifications": dict(METRIC_QUALIFICATIONS if metric_qualifications is None
                                          else metric_qualifications),
            "inputs": inputs,
            "arms": [summary_arm(run_dir, meta, controls_ok=controls_ok, reportable=reportable)
                     for run_dir, meta in runs]}


def write_summary_dir(out_dir, runs, controls_ok=True, reportable=True, figures=True,
                      notes=None, metric_qualifications=None):
    """Write a summariser output directory (JSON + tables + CSV + the figures it names)."""
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    summary = build_summary(runs, controls_ok=controls_ok, reportable=reportable,
                           notes=notes, metric_qualifications=metric_qualifications)
    path = write_json(os.path.join(out_dir, "yaw_pilot_summary.json"), summary)
    with open(os.path.join(out_dir, "yaw_pilot_tables.md"), "w") as fout:
        fout.write("# yaw_pilot full tables\n\nevery angle, every metric\n")
    with open(os.path.join(out_dir, "yaw_pilot_gaps.csv"), "w") as fout:
        fout.write("arm,k,metric,value\n")
        for _, meta in runs:
            fout.write("%s,64,EDT,0.014\n" % meta["arm"])
    if figures:
        names = ["yaw_pilot_gaps_all_arms"] + ["yaw_pilot_gaps_%s" % meta["arm"] for _, meta in runs]
        for name in names:
            for ext in (".png", ".pdf"):
                with open(os.path.join(out_dir, name + ext), "wb") as fout:
                    fout.write(b"\x89PNG\r\n" + name.encode("utf-8"))
    return path, summary


def write_arm_tree(root, arm, device="cuda", n_queries=6337, probe_queries=272,
                   parity=True, probe_parity=None, full_ok=True, probe_controls_ok=True,
                   complete=True, padded_len=16, probe_check_ok=True):
    """Write one arm's probe + full directories with their reports and the probe summary.

    Returns:
        ``{"all": (dir, meta), "probe": (dir, meta)}``.
    """
    full_dir = os.path.join(root, "%s_all" % arm)
    probe_dir = os.path.join(root, "%s_probe" % arm)
    full_meta = write_run(full_dir, arm=arm, device=device, n_queries=n_queries,
                          batches_arg="all", complete=complete, padded_len=padded_len)
    probe_meta = write_run(probe_dir, arm=arm, device=device, n_queries=probe_queries,
                           batches_arg="probe", padded_len=padded_len)
    write_check_online(full_dir, full_meta, ok=full_ok)
    write_check_online(probe_dir, probe_meta, ok=probe_check_ok)
    if parity:
        write_parity(full_dir, full_meta, arm=arm)
    if probe_parity if probe_parity is not None else parity:
        write_parity(probe_dir, probe_meta, arm=arm)
    write_summary_dir(os.path.join(probe_dir, "summary"), [(probe_dir, probe_meta)],
                      controls_ok=probe_controls_ok)
    return {"all": (full_dir, full_meta), "probe": (probe_dir, probe_meta)}


def write_record_tree(root, arms=ARMS, n_queries=6337, padded_len=16, cpu=True):
    """The whole evidence tree the finish script validates: four GPU arms + the CPU record.

    Returns:
        ``{arm: {"all": (dir, meta), "probe": (dir, meta)}, "cpu": {...}}``.
    """
    tree = {}
    for arm in arms:
        tree[arm] = write_arm_tree(root, arm, device="cuda", n_queries=n_queries,
                                   parity=arm in PARITY_ARMS, padded_len=padded_len)
    if cpu:
        cpu_root = os.path.join(root, "cpu_protocol")
        cpu_dir = os.path.join(cpu_root, "released_k8_all")
        cpu_meta = write_run(cpu_dir, arm="released_k8", device="cpu", n_queries=n_queries,
                             batches_arg="all", padded_len=padded_len)
        write_check_online(cpu_dir, cpu_meta, ok=True)
        write_parity(cpu_dir, cpu_meta, arm="released_k8", ok=False)
        write_summary_dir(os.path.join(cpu_root, "summary"), [(cpu_dir, cpu_meta)])
        tree["cpu"] = {"all": (cpu_dir, cpu_meta)}
    return tree


STUB_SUMMARIZE = '''#!/usr/bin/env python3
"""Stand-in for tools/exp10_summarize.py in the finish-script tests.

The real summariser needs the runs' waveforms, 10 000 bootstrap replicates and matplotlib;
what the finish script needs from it is a canonical summary directory.  Run as a script,
this writes one from the fixture builders and honours a few environment flags so a test can
make one stage fail.

*Imported* as a module it is not a stub at all: the record's HTML renderer regenerates the
page's figures by calling this module's ``make_figure`` / ``make_combined_figure``, so those
come from the real summariser, copied into the scratch repository as
``exp10_summarize_approved.py``.  A stub that drew its own figures could not tell a
regenerated figure from a copied one, which is the very thing the test has to distinguish.
The same holds for ``display_label``: both renderers import the arm display names from the
summariser, so the delegation has to cover every name they import.
"""
import os
import sys

_APPROVED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "exp10_summarize_approved.py")


def _approved():
    import importlib.util

    spec = importlib.util.spec_from_file_location("exp10_summarize_approved", _APPROVED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ != "__main__":
    _real = _approved()
    figure_data = _real.figure_data
    make_figure = _real.make_figure
    make_combined_figure = _real.make_combined_figure
    display_label = _real.display_label
else:
    import argparse
    import time

    sys.path.insert(0, os.environ["EXP10_FIXTURE_DIR"])
    import exp10_record_fixture as fx

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=0)
    args = ap.parse_args()
    if os.environ.get("STUB_FAIL_ON_CPU") and any("cpu_protocol" in r for r in args.runs):
        sys.stderr.write("stub summariser: refusing the CPU run (test)\\n")
        sys.exit(9)
    runs = [(r, fx.read_json(os.path.join(r, "meta.json"))) for r in args.runs]
    path, summary = fx.write_summary_dir(args.out, runs)
    if os.environ.get("STUB_BREAK_BINDING"):
        broken = fx.read_json(path)
        broken["inputs"][0]["per_sample_sha256"] = "0" * 64
        fx.write_json(path, broken)
    if os.environ.get("STUB_DROP_CSV"):
        os.remove(os.path.join(args.out, "yaw_pilot_gaps.csv"))
    if os.environ.get("STUB_FOREIGN_FIGURE"):
        with open(os.path.join(args.out, "yaw_pilot_gaps_someone_else.png"), "wb") as fout:
            fout.write(b"\\x89PNG\\r\\nforeign")
    if os.environ.get("STUB_SUBSTITUTE_COMBINED"):
        with open(os.path.join(args.out, "yaw_pilot_gaps_all_arms.png"), "wb") as fout:
            fout.write(os.environ["STUB_SUBSTITUTE_COMBINED"].encode("utf-8"))
    pause = os.environ.get("STUB_PAUSE_UNTIL")
    if pause and args.out.rstrip("/").endswith(os.path.join("_probe", "summary")):
        # The chain's probe summary is the boundary the concurrent-chain test needs: the
        # probe directory (and its source pins) exist, the full stage has not started.
        with open(pause + ".paused", "w") as fout:
            fout.write("paused\\n")
        for _retry in range(600):
            if os.path.exists(pause):
                break
            time.sleep(0.1)
        else:
            sys.stderr.write("stub summariser: pause was never released (test)\\n")
            sys.exit(11)
    print("wrote", path, len(summary["arms"]), "arms")
'''


def make_scratch_repo(root, assets_src, arms=ARMS, n_queries=32, drop=(), cpu=True):
    """Assemble a scratch repository the launch scripts can run in.

    The real record tooling is *copied* in (the tests exercise the real generators), the
    expensive summariser is replaced by ``STUB_SUMMARIZE``, and ``ckpt/exp10`` holds a
    synthetic evidence tree.

    Args:
        root: the scratch repository root.
        assets_src: the real ``yaw_pilot_results_assets`` directory to copy the tools from.
        arms: the arms to build.
        n_queries: the full runs' query count.
        drop: paths (relative to ``ckpt/exp10``) to delete after building, so a test can
            withhold one piece of evidence.
        cpu: also build the CPU-protocol record.

    Returns:
        ``{"root", "assets", "scripts", "exp10", "tree", "record"}``.
    """
    import shutil

    record = os.path.join(root, "worklog", "worklog_yixun", "exp_10_yaw_pilot_claude")
    assets = os.path.join(record, "yaw_pilot_results_assets")
    scripts = os.path.join(assets, "scripts")
    os.makedirs(scripts)
    os.makedirs(os.path.join(root, "tools"))
    for name in ("validate_runs.py", "make_results_md.py", "make_results_html.py",
                 "make_backend_table.py"):
        shutil.copy2(os.path.join(assets_src, name), os.path.join(assets, name))
    for name in sorted(os.listdir(os.path.join(assets_src, "scripts"))):
        shutil.copy2(os.path.join(assets_src, "scripts", name), os.path.join(scripts, name))
    with open(os.path.join(root, "tools", "exp10_summarize.py"), "w") as fout:
        fout.write(STUB_SUMMARIZE)
    # The approved summariser's figure drawing, which the HTML renderer calls by import
    # (STUB_SUMMARIZE delegates to it): the stub replaces the statistics, not the figures.
    real_repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(assets_src)))))
    approved = os.path.join(real_repo, "tools", "exp10_summarize.py")
    if not os.path.isfile(approved):
        raise AssertionError("the approved summariser is not at %s" % approved)
    shutil.copy2(approved, os.path.join(root, "tools", "exp10_summarize_approved.py"))
    exp10 = os.path.join(root, "ckpt", "exp10")
    os.makedirs(exp10)
    tree = write_record_tree(exp10, arms=arms, n_queries=n_queries, cpu=cpu)
    for rel in drop:
        target = os.path.join(exp10, rel)
        if os.path.isdir(target):
            shutil.rmtree(target)
        elif os.path.isfile(target):
            os.remove(target)
    return {"root": root, "assets": assets, "scripts": scripts, "exp10": exp10,
            "tree": tree, "record": record}


STUB_YAW_PILOT = '''#!/usr/bin/env python3
"""Stand-in for tools/exp10_yaw_pilot.py in the chain-script tests (no model, no data)."""
import argparse
import os
import sys

sys.path.insert(0, os.environ["EXP10_FIXTURE_DIR"])
import exp10_record_fixture as fx

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--backbone")
ap.add_argument("--checkpoint")
ap.add_argument("--manifest")
ap.add_argument("--manifest-hash")
ap.add_argument("--num-shot")
ap.add_argument("--ks")
ap.add_argument("--batches", required=True)
ap.add_argument("--controls", action="store_true")
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--out-dir", required=True)
args = ap.parse_args()
if os.environ.get("STUB_EVAL_FAIL_STAGE") == args.batches:
    sys.stderr.write("stub evaluator: failing the %s stage (test)\\n" % args.batches)
    sys.exit(7)
fx.write_run(args.out_dir, arm=args.arm, device=args.device,
             n_queries=8 if args.batches == "probe" else 16,
             batches_arg=args.batches)
print("stub run written to", args.out_dir)
'''

STUB_COMPARE = '''#!/usr/bin/env python3
"""Stand-in for tools/exp10_compare.py in the chain-script tests.

Writes a report of the requested kind and *exits 0* even when the report says ok = false --
which is exactly the behaviour of the approved comparator that let a failed online check
through the chain (finding 7).
"""
import argparse
import os
import sys

sys.path.insert(0, os.environ["EXP10_FIXTURE_DIR"])
import exp10_record_fixture as fx

ap = argparse.ArgumentParser()
ap.add_argument("command", choices=["check-online", "parity-exp03"])
ap.add_argument("run_dir")
ap.add_argument("exp03", nargs="?", default=None)
ap.add_argument("--json", required=True)
args = ap.parse_args()
stage = "probe" if args.run_dir.rstrip("/").endswith("_probe") else "all"
meta = fx.read_json(os.path.join(args.run_dir, "meta.json"))
if args.command == "check-online":
    ok = os.environ.get("STUB_ONLINE_FAIL_STAGE") != stage
    fx.write_check_online(args.run_dir, meta, ok=ok)
else:
    if not args.exp03 or not os.path.isfile(args.exp03):
        sys.stderr.write("stub comparator: no exp_03 predecessor at %r\\n" % args.exp03)
        sys.exit(4)
    ok = os.environ.get("STUB_PARITY_FAIL_STAGE") != stage
    fx.write_parity(args.run_dir, meta, ok=ok, arm=meta["arm"],
                    extra={"exp03_path": os.path.abspath(args.exp03)})
if os.environ.get("STUB_TOUCH_TOOL") and stage == "probe":
    with open("tools/exp10_yaw_pilot.py", "a") as fout:
        fout.write("# touched after the probe (test)\\n")
print("stub", args.command, "wrote", args.json, "ok =", ok)
'''


def make_chain_repo(root, assets_src):
    """A scratch git repository the arm chain can run in, with stubs for the three tools.

    Returns:
        ``{"root", "assets", "scripts", "record"}``.
    """
    import subprocess

    built = make_scratch_repo(root, assets_src, arms=(), cpu=False)
    with open(os.path.join(root, "tools", "exp10_yaw_pilot.py"), "w") as fout:
        fout.write(STUB_YAW_PILOT)
    with open(os.path.join(root, "tools", "exp10_compare.py"), "w") as fout:
        fout.write(STUB_COMPARE)
    for name in ("sweep_released", "sweep_control", "sweep_cyl"):
        write_json(os.path.join(root, "ckpt", "yaw_rotation", name, "per_sample_yaw.json"),
                   {"angles": {}})
    write_json(os.path.join(root, "ckpt", "yaw_rotation", "reference_manifest.json"), {"k": 8})
    with open(os.path.join(root, ".gitignore"), "w") as fout:
        fout.write("ckpt/\n")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    subprocess.check_call(["git", "init", "-q", root], env=env)
    subprocess.check_call(["git", "-C", root, "add", "-A"], env=env)
    subprocess.check_call(["git", "-C", root, "commit", "-q", "-m", "scratch"], env=env)
    return built


STUB_ARM_CHAIN = '''#!/bin/bash
# Stand-in for exp10_arm_chain.sh in the queue-script tests: records its argv and exits with
# the status the test asked for (EXP10_STUB_RC_<arm>).
echo "STUB CHAIN $*"
if [ -n "${EXP10_STUB_CHAIN_LOG:-}" ]; then echo "$*" >> "$EXP10_STUB_CHAIN_LOG"; fi
eval "rc=\\${EXP10_STUB_RC_$1:-0}"
echo "stub chain for $1 exiting $rc"
exit "$rc"
'''

FAKE_NVIDIA_SMI = '''#!/bin/bash
# A GPU with plenty of free memory, so the queue's resource guard is not what a test measures.
echo 40000
'''


def make_queue_repo(root, assets_src):
    """A scratch repository for the queue wrappers: the real queues, a stub arm chain.

    Returns:
        ``{"root", "assets", "scripts", "record", "bin"}`` — ``bin`` holds a fake ``nvidia-smi``.
    """
    built = make_scratch_repo(root, assets_src, arms=(), cpu=False)
    with open(os.path.join(built["scripts"], "exp10_arm_chain.sh"), "w") as fout:
        fout.write(STUB_ARM_CHAIN)
    os.chmod(os.path.join(built["scripts"], "exp10_arm_chain.sh"), 0o755)
    bin_dir = os.path.join(root, "fakebin")
    os.makedirs(bin_dir)
    with open(os.path.join(bin_dir, "nvidia-smi"), "w") as fout:
        fout.write(FAKE_NVIDIA_SMI)
    os.chmod(os.path.join(bin_dir, "nvidia-smi"), 0o755)
    built["bin"] = bin_dir
    return built
