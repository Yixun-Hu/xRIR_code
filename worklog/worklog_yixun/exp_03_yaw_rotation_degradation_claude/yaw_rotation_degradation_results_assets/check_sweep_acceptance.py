"""Pre-summary acceptance check for the exp_03 sweep (plan §9, the pre-launch criteria, Codex rounds 2-3).

Fail-closed contract with every constant pinned here. Pure validators (return a list of failure strings) are composed by
main(); tests/test_exp03_record_tools.py exercises them with fabricated inputs.

What is required (all of it, or exit 1):
- exactly the three canonical sweep run paths;
- the k=0 gate, validated independently: the exact band-rule string, the exact ten gate runs in the gate JSON's run lists, every
  gate run's sidecar bound to the reviewed source closure with artefact/checkpoint digests matching, the expected per-run profile
  (batch, TF32, GL seed, manifest seed, k=0-only grids), finite k=0 content reconciled with its aggregates and ordered like its
  own manifest, the gate summary present, hash-bound and byte-identical to the retained decision copy in the record folder and
  to the assets copy of the gate JSON, every numeric leaf of the gate rows finite, the gate decision written before the sweep
  started (every sweep run must carry a valid start timestamp), every gate sidecar deep-checked like a sweep sidecar (closure
  records, live environment, live data identity for its manifest seed, log digest), and the decision RECOMPUTED (skipping it with
  --no-recompute-gate yields "ACCEPTANCE INCOMPLETE" and exit 2, never PASS) with the pinned producer blob (tools/summarize_yaw.py at
  the commit that took it) from the ten bound runs, the three manifests, the two exp_01 inputs and the released baseline, whose
  summary must be byte-identical to the retained copy and whose gate rows/terms/pass must equal the canonical JSON;
- the actual seed-0 manifest loaded and hashed (semantic hash pinned; file digest recorded in the sidecars), with every run's
  query list and index list equal to the manifest's in order;
- per-run meta (backbone, checkpoint, batch 16 canonical, TF32 off, GL seed 0, manifest seed 0, pre-registered grids, 6337
  samples, per-sample meta == aggregate meta);
- per-sample content: spectral arrays (loss, stft, decay, log_mse, consistency) finite numbers at all 18 angles in P and E;
  acoustic arrays (edt, c50, t60) finite-or-null at k=0 and the P (10) / E (4) angles; NaN/inf rejected everywhere;
  aggregate mean, n_valid and n_nan reconciled with every per-sample array (an all-null array requires a null mean); P == E at
  k=0; delay flips integers in [0, 6337 x 8] at all 18 angles, equal between per-sample and aggregate and across runs; the
  decomposition the producer consumes (per-sample file) identical to the aggregate's and pinned to k=32 over 1 batch for the
  cylindrical run only;
- provenance: evaluator digest and reviewed commit pinned, source-closure digest and records equal to a fresh recomputation,
  launcher HEAD (historical 5febf47... by default; --launcher-head for a future sweep), checkpoint digest equal to the file on
  disk and to the role's gate sidecar, environment and data identity equal to a live recomputation (content-hashed inventory),
  run log present with the bound digest, and every live sidecar byte-identical to the record kept in the binding report, whose
  canonical and assets copies must be byte-identical.

    python .../check_sweep_acceptance.py            # runs default to the canonical paths
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bind_provenance as bp  # noqa: E402

RECORD = "worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude"
SWEEP_RUNS = ("ckpt/yaw_rotation/sweep_control", "ckpt/yaw_rotation/sweep_cyl", "ckpt/yaw_rotation/sweep_released")
GATE_JSON = "ckpt/yaw_rotation/gate_stats.json"
GATE_JSON_COPY = os.path.join(RECORD, "yaw_rotation_degradation_results_assets", "gate_stats.json")
BAND_RULE = "v3 (2026-09-06 per-model amendment)"
REPORT = "ckpt/yaw_rotation/binding_report.json"
REPORT_COPY = os.path.join(RECORD, "yaw_rotation_degradation_results_assets", "binding_report.json")
RELEASED_BASELINE = "0.0549,1.358,9.69"
PYTHON = os.path.expanduser("~/miniconda3/envs/xRIR/bin/python")
GATE_DECISION_COPY = os.path.join(RECORD, "yaw_rotation_degradation_2026-09-06_15:40:32_gate_decision_v3_PASSED.txt")
GATE_RUNS = {"runs": ["ckpt/yaw_rotation/gate_control", "ckpt/yaw_rotation/gate_cyl", "ckpt/yaw_rotation/gate_released"],
             "noise_runs": ["ckpt/yaw_rotation/gate_noise_seed1", "ckpt/yaw_rotation/gate_noise_seed2"],
             "noise_runs_cyl": ["ckpt/yaw_rotation/gate_noise_cyl_seed1", "ckpt/yaw_rotation/gate_noise_cyl_seed2"],
             "nuisance_runs": ["ckpt/yaw_rotation/gate_phase_seed1", "ckpt/yaw_rotation/gate_tf32"],
             "shape_runs": ["ckpt/yaw_rotation/gate_shape_cyl_b1"]}
# per gate run: backbone, checkpoint, batch, tf32, gl_seed, manifest_seed
GATE_PROFILE = {"gate_control": ("simple", "ckpt/xRIR_simple_8_shot/epoch_12.pth", 16, False, 0, 0), "gate_cyl": ("cylindrical", "ckpt/xRIR_cyl_8_shot/epoch_12.pth", 16, False, 0, 0),
                "gate_released": ("simple", "checkpoints/xRIR_unseen.pth", 16, False, 0, 0), "gate_noise_seed1": ("simple", "ckpt/xRIR_simple_8_shot/epoch_12.pth", 16, False, 0, 1),
                "gate_noise_seed2": ("simple", "ckpt/xRIR_simple_8_shot/epoch_12.pth", 16, False, 0, 2), "gate_phase_seed1": ("simple", "ckpt/xRIR_simple_8_shot/epoch_12.pth", 16, False, 1, 0),
                "gate_tf32": ("simple", "ckpt/xRIR_simple_8_shot/epoch_12.pth", 16, True, 0, 0), "gate_noise_cyl_seed1": ("cylindrical", "ckpt/xRIR_cyl_8_shot/epoch_12.pth", 16, False, 0, 1),
                "gate_noise_cyl_seed2": ("cylindrical", "ckpt/xRIR_cyl_8_shot/epoch_12.pth", 16, False, 0, 2), "gate_shape_cyl_b1": ("cylindrical", "ckpt/xRIR_cyl_8_shot/epoch_12.pth", 1, False, 0, 0)}
MANIFEST = "ckpt/yaw_rotation/reference_manifest.json"
MANIFEST_HASH = "47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d"
MANIFEST_HASHES = {0: MANIFEST_HASH, 1: "6ca8164e5727d5f4db84569d5effa840b2cb6e6f77819a69bbb37bb30c6ab11e", 2: "757e5a966ee2d069a07accecbc43e46bd564ea7c788a28c219773df64e986234"}
LAUNCHER_HEAD = "5febf471318e37e60477ebe72afd48f03879c67e"
EXPECTED_N = 6337
EXPECTED_ROOMS = 17
NUM_SHOT = 8
SPECTRAL = (0, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 416, 448, 480, 496, 504, 508)
ACOUSTIC = (0, 8, 32, 64, 128, 256, 384, 448, 480, 504)
E_ACOUSTIC = (32, 128, 384, 480)
SPECTRAL_KEYS = ("loss", "stft", "decay", "log_mse", "consistency")
ACOUSTIC_KEYS = ("edt", "c50", "t60")
SWEEP_PROFILE = {"sweep_control": ("simple", "ckpt/xRIR_simple_8_shot/epoch_12.pth"), "sweep_cyl": ("cylindrical", "ckpt/xRIR_cyl_8_shot/epoch_12.pth"), "sweep_released": ("simple", "checkpoints/xRIR_unseen.pth")}
ENV_KEYS = ("python", "torch", "torchaudio", "torchvision", "einops", "numpy", "scipy", "cuda", "cudnn", "gpus", "nvidia_driver", "host")


# ---------------------------------------------------------------- pure validators (return failure strings)
def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def finite_array(arr, n):
    return isinstance(arr, list) and len(arr) == n and all(is_num(v) for v in arr)


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def finite_leaves(x):
    """True iff every numeric leaf of a nested structure is finite (bools and strings are not numbers)."""
    if isinstance(x, dict):
        return all(finite_leaves(v) for v in x.values())
    if isinstance(x, list):
        return all(finite_leaves(v) for v in x)
    if isinstance(x, float):
        return math.isfinite(x)
    return True


def finite_or_null_array(arr, n):
    return isinstance(arr, list) and len(arr) == n and all(v is None or is_num(v) for v in arr)


def check_gate_json(g, sweep_start=None, summary_sha=None, decision_copy_sha=None):
    """Structural + chronological validation of the gate JSON (no IO)."""
    f = []
    if g.get("mode") != "k0-gate":
        f.append(f"gate mode {g.get('mode')!r} != 'k0-gate'")
    if g.get("gate_pass") is not True:
        f.append(f"gate_pass is {g.get('gate_pass')!r}")
    if g.get("band_rule") != BAND_RULE:
        f.append(f"gate band rule {g.get('band_rule')!r} != {BAND_RULE!r}")
    if not finite_leaves(g.get("gate_rows")) or not finite_leaves(g.get("band_terms")) or not isinstance(g.get("gate_rows"), list) or not g.get("gate_rows"):
        f.append("gate rows/terms missing or containing non-finite numbers")
    if g.get("manifest_hash") != MANIFEST_HASH:
        f.append("gate manifest hash is not the pinned seed-0 hash")
    cfg = g.get("config") or {}
    for key, want in GATE_RUNS.items():
        have = cfg.get(key) if key == "runs" else g.get(key)
        have = [h.rstrip("/") for h in (have or [])] if isinstance(have, list) else have
        if have != want:
            f.append(f"gate {key} = {have} != required {want}")
    if not g.get("summary_path") or not g.get("summary_sha256"):
        f.append("gate JSON lacks summary_path/summary_sha256")
    if summary_sha is None:
        f.append("gate summary file missing (nothing to bind or date)")
    elif summary_sha != g.get("summary_sha256"):
        f.append("gate summary file digest != gate JSON's summary_sha256")
    if decision_copy_sha is not None and decision_copy_sha != g.get("summary_sha256"):
        f.append("retained gate decision copy is not byte-identical to the gate summary")
    if sweep_start is not None:
        if g.get("_summary_mtime") is None:
            f.append("gate chronology cannot be established (no summary mtime)")
        elif g["_summary_mtime"] >= sweep_start:
            f.append("gate summary was written after the sweep started")
    return f


def check_manifest_order(queries, indices, manifest_entries):
    f = []
    want_q = [e["query"] for e in manifest_entries]; want_i = [e["index"] for e in manifest_entries]
    if queries != want_q:
        f.append("query list differs from the manifest's queries (content or order)")
    if indices != want_i:
        f.append("index list differs from the manifest's indices")
    if len(set(queries)) != len(queries):
        f.append("duplicate queries")
    if len({"/".join(q.split("/")[:2]) for q in queries}) != EXPECTED_ROOMS:
        f.append("room count != 17")
    return f


def check_arrays(ps, m, n=EXPECTED_N):
    """Per-sample content, aggregate reconciliation, P == E at k=0."""
    f = []
    for cond in ("P", "E"):
        for k in SPECTRAL:
            rows = (ps.get(cond) or {}).get(str(k)); agg = (m.get(cond) or {}).get(str(k)) or {}
            if rows is None:
                f.append(f"{cond} k={k} missing"); continue
            for key in SPECTRAL_KEYS:
                arr = rows.get(key)
                if not finite_array(arr, n):
                    f.append(f"{cond} k={k} {key}: not {n} finite numbers"); continue
                a = agg.get(key)
                if not isinstance(a, dict) or not is_int(a.get("n_valid")) or a.get("n_valid") != n or not is_int(a.get("n_nan")) or a.get("n_nan") != 0 or not is_num(a.get("mean")) or not math.isclose(sum(arr) / n, a["mean"], rel_tol=1e-9, abs_tol=1e-12):
                    f.append(f"{cond} k={k} {key}: aggregate mean/counts not reconciled with the per-sample array (exact integer counts required)")
            want = (0,) + tuple(c for c in ACOUSTIC if c) if cond == "P" else (0,) + E_ACOUSTIC
            for key in ACOUSTIC_KEYS:
                arr = rows.get(key)
                if k not in want:
                    continue
                if not finite_or_null_array(arr, n):
                    f.append(f"{cond} k={k} {key}: not {n} finite-or-null values"); continue
                fin = [v for v in arr if v is not None]; a = agg.get(key)
                if not isinstance(a, dict) or not is_int(a.get("n_valid")) or a.get("n_valid") != len(fin) or not is_int(a.get("n_nan")) or a.get("n_nan") != n - len(fin):
                    f.append(f"{cond} k={k} {key}: n_valid/n_nan not reconciled with the per-sample array (exact integer counts required)")
                elif fin and (not is_num(a.get("mean")) or not math.isclose(sum(fin) / len(fin), a["mean"], rel_tol=1e-9, abs_tol=1e-12)):
                    f.append(f"{cond} k={k} {key}: aggregate mean != per-sample mean")
                elif not fin and a.get("mean") is not None:
                    f.append(f"{cond} k={k} {key}: no valid value but a non-null aggregate mean")
    p0, e0 = (ps.get("P") or {}).get("0") or {}, (ps.get("E") or {}).get("0") or {}
    for key in SPECTRAL_KEYS + ACOUSTIC_KEYS:
        if p0.get(key) != e0.get(key):
            f.append(f"P and E differ at k=0 for {key}")
    return f


def check_decomposition(dec, tag):
    if tag == "sweep_cyl":
        ok = isinstance(dec, dict) and dec.get("k") == 32 and dec.get("n_batches") == 1 and is_int(dec.get("k")) and is_int(dec.get("n_batches")) \
            and all(is_num(dec.get(k)) for k in ("tokens_rel_change", "pooled_rel_change", "coord_rel_change", "logspec_rel_change"))
        return [] if ok else ["decomposition missing, not at k=32 over 1 batch, or non-finite"]
    return [] if dec is None else ["unexpected decomposition on a non-cylindrical run"]


def check_delay_flips(agg_flips, ps_flips, n=EXPECTED_N, num_shot=NUM_SHOT):
    f = []
    if not isinstance(agg_flips, dict) or not all(k.isdigit() for k in agg_flips) or sorted(int(k) for k in agg_flips) != sorted(SPECTRAL):
        f.append("delay_flips not written for exactly the 18 angles"); return f
    for k, v in agg_flips.items():
        if not is_int(v) or not 0 <= v <= n * num_shot:
            f.append(f"delay_flips[{k}] = {v!r} is not an integer in [0, {n * num_shot}]")
    if ps_flips != agg_flips:
        f.append("per-sample delay_flips differ from the aggregate")
    return f


def check_meta(meta, backbone, ckpt, n=EXPECTED_N):
    f = []
    if meta.get("backbone") != backbone or meta.get("checkpoint") != ckpt:
        f.append(f"backbone/checkpoint {meta.get('backbone')} {meta.get('checkpoint')} != {backbone} {ckpt}")
    if meta.get("manifest_hash") != MANIFEST_HASH or meta.get("manifest_seed") != 0 or meta.get("manifest_path") != MANIFEST:
        f.append("manifest fields not the pinned seed-0 manifest")
    if not (is_int(meta.get("n_samples")) and meta.get("n_samples") == n and is_int(meta.get("max_samples")) and meta.get("max_samples") == 0):
        f.append("not the full split (exact integers required)")
    if not (is_int(meta.get("batch_size")) and meta.get("batch_size") == 16 and meta.get("batch_canonical") is True and meta.get("tf32") is False and is_int(meta.get("gl_seed")) and meta.get("gl_seed") == 0 and is_int(meta.get("manifest_seed"))):
        f.append("batch/TF32/GL-seed settings differ from the pre-registered ones (exact integers required)")
    for key, want in (("yaw_cols", SPECTRAL), ("acoustic_cols", ACOUSTIC), ("e_acoustic_cols", E_ACOUSTIC)):
        cols = meta.get(key)
        if not isinstance(cols, list) or not all(is_int(c) for c in cols) or sorted(cols) != sorted(want):
            f.append(f"{key} differs from the pre-registered grid (exact integers required)")
    return f


def check_provenance(pv, run, closure_rec, closure_digest, gate_sidecar, manifest_file_sha, head=LAUNCHER_HEAD, live_env=None, live_data=None, meta=None, expected_ckpt=None):
    f = []
    if meta is not None and pv.get("checkpoint") != meta.get("checkpoint"):
        f.append("sidecar checkpoint path differs from the metrics meta")
    if expected_ckpt is not None and pv.get("checkpoint") != expected_ckpt:
        f.append("sidecar checkpoint path is not the profile's checkpoint")
    start, why = bp.run_start(pv, run)
    if start is None:
        f.append(f"bound log not canonical: {why}")
    elif not isinstance(pv.get("run_start"), str) or pv["run_start"] != start.isoformat():
        f.append("bound run_start missing, empty or different from the canonical log name")
    if pv.get("evaluator_sha256") != bp.REVIEWED_EVAL_SHA or pv.get("reviewed_evaluator_commit") != bp.REVIEWED_COMMIT:
        f.append("evaluator not bound to the reviewed commit")
    if pv.get("source_closure_sha256") != closure_digest:
        f.append("source-closure digest differs from a fresh recomputation")
    have = [(r.get("path"), r.get("reviewed_blob_sha256"), r.get("working_tree_sha256")) for r in pv.get("source_closure") or []]
    want = [(r["path"], r["reviewed_blob_sha256"], r["working_tree_sha256"]) for r in closure_rec]
    if have != want:
        f.append("source-closure records differ from a fresh recomputation")
    if pv.get("git_sha") != head:
        f.append(f"launcher HEAD {str(pv.get('git_sha'))[:7]} != {head[:7]}")
    if pv.get("per_sample_sha256") != bp.sha(os.path.join(run, "per_sample_yaw.json")) or pv.get("metrics_sha256") != bp.sha(os.path.join(run, "metrics_yaw.json")):
        f.append("artefact digests differ from the files")
    if pv.get("checkpoint_sha256") != bp.sha(pv.get("checkpoint", "")) or (gate_sidecar and pv.get("checkpoint_sha256") != gate_sidecar.get("checkpoint_sha256")):
        f.append("checkpoint digest differs from the file or from the gate sidecar")
    env = pv.get("environment") or {}
    if any(k not in env for k in ENV_KEYS):
        f.append("environment record incomplete")
    if live_env is not None and any(env.get(k) != live_env.get(k) for k in ENV_KEYS):
        f.append("environment record differs from the live environment")
    d = pv.get("data_identity") or {}
    if d.get("manifest_hash") not in MANIFEST_HASHES.values() or not is_int(d.get("inventory_missing")) or d.get("inventory_missing") != 0 or not is_int(d.get("inventory_files")) or d.get("inventory_files") <= 0 \
            or d.get("manifest_file_sha256") != manifest_file_sha or not d.get("inventory_sha256"):
        f.append("data identity missing, incomplete or stale (integer inventory counts, no missing files)")
    if live_data is not None and d != live_data:
        f.append("data identity differs from a live recomputation (content-hashed inventory)")
    if not pv.get("log_sha256") or not pv.get("log") or not os.path.isfile(pv["log"]) or bp.sha(pv["log"]) != pv["log_sha256"]:
        f.append("run log missing or its digest differs from the bound one")
    return f


def check_report(report, report_copy_sha, canonical_sha, runs):
    """The retained binding report must be byte-identical in both copies and record every live sidecar's current digest."""
    f = []
    if report_copy_sha != canonical_sha:
        f.append("binding report: assets copy differs from the canonical report")
    recs = (report or {}).get("runs") or {}
    for run in runs:
        rec = recs.get(run)
        if not rec:
            f.append(f"binding report has no record for {run}"); continue
        side = os.path.join(run, "provenance.json")
        if not os.path.isfile(side) or bp.sha(side) != rec.get("sidecar_sha256"):
            f.append(f"live sidecar of {run} differs from the retained binding record")
    gi = (report or {}).get("gate_inputs") or {}
    for k in ("exp01_control", "exp01_cyl", "gate_json", "gate_summary", "decision_v1", "decision_v2", "decision_v3"):
        rec = gi.get(k) or {}
        if not rec.get("sha256") or not os.path.isfile(rec.get("path", "")) or bp.sha(rec["path"]) != rec["sha256"]:
            f.append(f"gate input {k} missing or differs from the bound digest")
    try:
        blob = subprocess.check_output(["git", "show", f"{bp.GATE_PRODUCER_COMMIT}:tools/summarize_yaw.py"])
        if gi.get("gate_producer_commit") != bp.GATE_PRODUCER_COMMIT or gi.get("gate_producer_sha256") != hashlib.sha256(blob).hexdigest():
            f.append("binding report: gate producer commit/digest differ from the pinned blob")
        want = bp.gate_producer_closure()
        if [(r["path"], r["pinned_blob_sha256"]) for r in gi.get("gate_producer_closure") or []] != [(r["path"], r["pinned_blob_sha256"]) for r in want]:
            f.append("binding report: gate producer closure differs from the pinned commit's")
        for r in gi.get("gate_producer_closure") or []:
            if not r.get("working_tree_sha256") or r["working_tree_sha256"] != (bp.sha(r["path"]) if os.path.isfile(r["path"]) else None):
                f.append(f"binding report: recorded working-tree digest of {r['path']} is stale")
    except subprocess.CalledProcessError:
        f.append("binding report: pinned gate producer blob not retrievable")
    return f


def check_k0_content(ps, m, manifest_entries, n=EXPECTED_N):
    """A k=0-only run: the sweep's exact rules at k=0 in P and E (finite / finite-or-null, mean + n_valid + n_nan reconciled,
    all-null => null mean, P == E), and the query/index order of its own manifest."""
    f = []
    for cond in ("P", "E"):
        rows = (ps.get(cond) or {}).get("0"); agg = (m.get(cond) or {}).get("0") or {}
        if rows is None:
            f.append(f"{cond} k=0 missing"); continue
        for key in SPECTRAL_KEYS:
            arr = rows.get(key); a = agg.get(key)
            if not finite_array(arr, n) or not isinstance(a, dict) or not is_int(a.get("n_valid")) or a.get("n_valid") != n or not is_int(a.get("n_nan")) or a.get("n_nan") != 0 or not is_num(a.get("mean")) or not math.isclose(sum(arr) / n, a["mean"], rel_tol=1e-9, abs_tol=1e-12):
                f.append(f"{cond} k=0 {key}: not finite or not reconciled (mean / n_valid / n_nan, exact integer counts)")
        for key in ACOUSTIC_KEYS:
            arr = rows.get(key); a = agg.get(key)
            if not finite_or_null_array(arr, n):
                f.append(f"{cond} k=0 {key}: not finite-or-null"); continue
            fin = [v for v in arr if v is not None]
            if not isinstance(a, dict) or not is_int(a.get("n_valid")) or a.get("n_valid") != len(fin) or not is_int(a.get("n_nan")) or a.get("n_nan") != n - len(fin):
                f.append(f"{cond} k=0 {key}: n_valid/n_nan not reconciled (exact integer counts)")
            elif fin and (not is_num(a.get("mean")) or not math.isclose(sum(fin) / len(fin), a["mean"], rel_tol=1e-9, abs_tol=1e-12)):
                f.append(f"{cond} k=0 {key}: aggregate mean != per-sample mean")
            elif not fin and a.get("mean") is not None:
                f.append(f"{cond} k=0 {key}: no valid value but a non-null aggregate mean")
    p0, e0 = (ps.get("P") or {}).get("0") or {}, (ps.get("E") or {}).get("0") or {}
    for key in SPECTRAL_KEYS + ACOUSTIC_KEYS:
        if p0.get(key) != e0.get(key):
            f.append(f"P and E differ at k=0 for {key}")
    f += check_manifest_order(ps.get("query"), ps.get("index"), manifest_entries)
    return f


def check_k0_meta(meta, backbone, ckpt, batch, tf32, gl, mseed, n=EXPECTED_N):
    """Exact profile of a k=0-only gate run (integer grids exactly [0], [0], [])."""
    f = []
    if not (meta.get("backbone") == backbone and meta.get("checkpoint") == ckpt and is_int(meta.get("batch_size")) and meta.get("batch_size") == batch and meta.get("tf32") is tf32
            and meta.get("batch_canonical") is True and is_int(meta.get("gl_seed")) and meta.get("gl_seed") == gl and is_int(meta.get("manifest_seed")) and meta.get("manifest_seed") == mseed):
        f.append("profile differs from the band-v3 specification (exact integers, batch_canonical True)")
    if meta.get("manifest_hash") != MANIFEST_HASHES.get(mseed) or not is_int(meta.get("n_samples")) or meta.get("n_samples") != n or not is_int(meta.get("max_samples")) or meta.get("max_samples") != 0:
        f.append("not a full-split run on the expected manifest")
    for key, want in (("yaw_cols", [0]), ("acoustic_cols", [0]), ("e_acoustic_cols", [])):
        cols = meta.get(key)
        if not isinstance(cols, list) or not all(is_int(c) for c in cols) or cols != want:
            f.append(f"{key} is not exactly {want} (exact integers required)")
    return f


def check_decomposition_pair(metrics_dec, per_sample_dec, tag):
    """The producer consumes the per-sample decomposition; it must equal the aggregate's and satisfy the pin."""
    f = []
    if metrics_dec != per_sample_dec:
        f.append("per-sample decomposition differs from the aggregate's (the producer consumes the per-sample one)")
    f += check_decomposition(per_sample_dec, tag)
    return f


def check_gate_run(run, closure_rec, closure_digest, manifests=None, live_ctx=None):
    tag = os.path.basename(run); f = []
    try:
        m = json.load(open(os.path.join(run, "metrics_yaw.json"))); pv = json.load(open(os.path.join(run, "provenance.json"))); ps = json.load(open(os.path.join(run, "per_sample_yaw.json")))
    except Exception as e:  # noqa: BLE001
        return [f"{tag}: cannot read ({e})"]
    bb, ck, bs, tf32, gl, ms = GATE_PROFILE[tag]; meta = m["meta"]
    f += [f"{tag}: {x}" for x in check_k0_meta(meta, bb, ck, bs, tf32, gl, ms)]
    if ps.get("meta") != meta:
        f.append(f"{tag}: per-sample meta != aggregate meta")
    if manifests is not None:
        f += [f"{tag}: {x}" for x in check_k0_content(ps, m, manifests[ms]["entries"])]
    live = live_ctx or {}
    man_sha = bp.sha(bp.MANIFESTS[ms]) if os.path.isfile(bp.MANIFESTS[ms]) else None
    pf = check_provenance(pv, run, closure_rec, closure_digest, None, man_sha, head=pv.get("git_sha"), live_env=live.get("env"), live_data=(live.get("data") or {}).get(ms), meta=meta, expected_ckpt=ck)
    # gate sidecars were written post hoc with git_sha pinned to the reviewed commit, so their git_sha is not a launcher HEAD (head=pv["git_sha"] above)
    f += [f"{tag}: {x}" for x in pf]
    if (pv.get("data_identity") or {}).get("manifest_hash") != MANIFEST_HASHES[ms]:
        f.append(f"{tag}: data identity is not for manifest seed {ms}")
    return f


def recompute_gate(tmpdir):
    """Re-run the k=0 gate decision with the producer's complete repo-local closure pinned at the commit that took it
    (tools/summarize_yaw.py and everything it imports, extracted with paths preserved and put first on sys.path)."""
    for f in bp.GATE_PRODUCER_FILES:
        r = subprocess.run(["git", "show", f"{bp.GATE_PRODUCER_COMMIT}:{f}"], capture_output=True)
        if r.returncode != 0:
            continue
        dst = os.path.join(tmpdir, f); os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as fh:
            fh.write(r.stdout)
    if not os.path.isfile(os.path.join(tmpdir, "tools", "__init__.py")):
        open(os.path.join(tmpdir, "tools", "__init__.py"), "w").close()   # namespace package at that commit
    prod = os.path.join(tmpdir, "tools", "summarize_yaw.py")
    js, sm = os.path.join(tmpdir, "gate_stats.json"), os.path.join(tmpdir, "gate_summary.txt")
    cmd = [PYTHON, prod, "--mode", "k0-gate", "--band-rule", "v3", "--runs", *GATE_RUNS["runs"], "--manifest-hash", MANIFEST_HASH,
           "--exp01-per-sample", f"control={bp.GATE_INPUTS['exp01_control']}", f"cyl={bp.GATE_INPUTS['exp01_cyl']}",
           "--noise-runs", *GATE_RUNS["noise_runs"], "--noise-runs-cyl", *GATE_RUNS["noise_runs_cyl"], "--nuisance-runs", *GATE_RUNS["nuisance_runs"],
           "--shape-runs", *GATE_RUNS["shape_runs"], "--released-baseline", RELEASED_BASELINE, "--json", js, "--summary", sm]
    r = subprocess.run(cmd, cwd=os.getcwd(), env={**os.environ, "PYTHONPATH": tmpdir + os.pathsep + os.getcwd()}, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(sm) or not os.path.isfile(js):
        return None, None, f"recomputation failed or produced no output (exit {r.returncode}): {r.stderr[-400:]}"
    return open(sm, "rb").read(), json.load(open(js)), f"exit {r.returncode}"


def full_gate_validation(recompute=True):
    """Everything main() checks about the gate, with live context and the retained report: for launcher preflight."""
    rec, digest = bp.closure_record(bp.source_closure(os.getcwd()), os.getcwd())
    manifests = load_manifests(); live_env = bp.environment(); live_data = {s: bp.data_identity(p) for s, p in bp.MANIFESTS.items()}
    f = gate_ok(None, (rec, digest), recompute=recompute, manifests=manifests, live_ctx={"env": live_env, "data": live_data})
    report = json.load(open(REPORT)) if os.path.isfile(REPORT) else None
    f += check_report(report, bp.sha(REPORT_COPY) if os.path.isfile(REPORT_COPY) else None, bp.sha(REPORT) if os.path.isfile(REPORT) else None, sum(GATE_RUNS.values(), []))
    return f


def load_manifests():
    from tools.reference_manifest import load_manifest, manifest_hash
    out = {}
    for seed, path in bp.MANIFESTS.items():
        raw = json.load(open(path))
        if manifest_hash(load_manifest(path)) != MANIFEST_HASHES[seed] or raw.get("num_shot") != NUM_SHOT or len(raw.get("entries", [])) != EXPECTED_N:
            raise SystemExit(f"manifest seed {seed} hash/shots/size differ from the pinned values")
        out[seed] = raw
    return out


def gate_ok(sweep_start=None, closure=None, recompute=True, manifests=None, live_ctx=None):
    """Full independent gate validation with IO; used by main() and by launcher v3 (sweep_start None before a sweep)."""
    rec, digest = closure if closure else bp.closure_record(bp.source_closure(os.getcwd()), os.getcwd())
    manifests = manifests if manifests is not None else load_manifests()
    try:
        g = json.load(open(GATE_JSON))
    except Exception as e:  # noqa: BLE001
        return [f"cannot read {GATE_JSON}: {e}"]
    sp = g.get("summary_path")
    summary_sha = bp.sha(sp) if sp and os.path.isfile(sp) else None
    copy_sha = bp.sha(GATE_DECISION_COPY) if os.path.isfile(GATE_DECISION_COPY) else None
    if sp and os.path.isfile(sp):
        g["_summary_mtime"] = datetime.datetime.fromtimestamp(os.stat(sp).st_mtime).astimezone()
    f = check_gate_json(g, sweep_start, summary_sha, copy_sha)
    if sp != "ckpt/yaw_rotation/gate_summary.txt":
        f.append(f"gate summary path {sp!r} is not the canonical one")
    if copy_sha is None:
        f.append("retained gate decision copy missing from the record folder")
    if not os.path.isfile(GATE_JSON_COPY) or bp.sha(GATE_JSON_COPY) != bp.sha(GATE_JSON):
        f.append("assets copy of the gate JSON missing or differs from the canonical one")
    for run in sum(GATE_RUNS.values(), []):
        f += check_gate_run(run, rec, digest, manifests, live_ctx)
    if recompute and not f:
        tmp = tempfile.mkdtemp(prefix="gate_recompute_")
        try:
            sm, js, note = recompute_gate(tmp)
            if sm is None:
                f.append(note)
            else:
                if sm != open(GATE_DECISION_COPY, "rb").read():
                    f.append("recomputed gate summary is not byte-identical to the retained decision copy")
                for key in ("gate_rows", "band_terms", "gate_pass", "band_rule", "gate_reasons"):
                    if js.get(key) != g.get(key):
                        f.append(f"recomputed gate {key} differs from the canonical gate JSON")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return f


def sweep_start_time(runs):
    """Earliest sweep start; every run must carry a parsable start timestamp in its bound log name, else (None, reason)."""
    starts = []
    for run in runs:
        pv = json.load(open(os.path.join(run, "provenance.json")))
        mm = re.search(r"_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})_", os.path.basename(pv.get("log", "")))
        if not mm:
            return None, f"{run}: no parsable start timestamp in the bound log name"
        starts.append(datetime.datetime.strptime(mm.group(1), "%Y-%m-%d_%H:%M:%S").astimezone())
    return min(starts), "ok"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="*", default=list(SWEEP_RUNS))
    ap.add_argument("--launcher-head", default=LAUNCHER_HEAD, help="full SHA the sweep sidecars must record (default: the historical 2026-09-06 sweep)")
    ap.add_argument("--no-recompute-gate", action="store_true", help="skip re-running the pinned gate producer (the recomputation takes a few minutes)")
    a = ap.parse_args()
    fails = []
    runs = [r.rstrip("/") for r in a.runs]
    if sorted(runs) != sorted(SWEEP_RUNS):
        print(f"FAIL runs must be exactly {SWEEP_RUNS}, got {runs}"); print("ACCEPTANCE FAIL (1)"); sys.exit(1)
    sys.path.insert(0, os.getcwd())
    from tools import summarize_yaw as sy
    from tools.reference_manifest import load_manifest, manifest_hash
    if tuple(sy.PREREGISTERED_SPECTRAL_COLS) != SPECTRAL or tuple(sy.PREREGISTERED_ACOUSTIC_COLS) != ACOUSTIC or tuple(sy.PREREGISTERED_E_ACOUSTIC_COLS) != E_ACOUSTIC:
        fails.append("summarizer grids differ from the pinned pre-registered grids")
    rec, digest = bp.closure_record(bp.source_closure(os.getcwd()), os.getcwd())
    if any(r["reviewed_blob_sha256"] != r["working_tree_sha256"] or r["commits_after_reviewed"] for r in rec):
        fails.append("source closure differs from the reviewed commit")
    print(f"source closure {len(rec)} files, digest {digest[:12]}…")
    manifests = load_manifests(); raw = manifests[0]; man_sha = bp.sha(MANIFEST)
    live_env = bp.environment(); live_data = {s: bp.data_identity(p) for s, p in bp.MANIFESTS.items()}
    print(f"live environment and data identity recomputed (inventory {live_data[0]['inventory_files']} files, {live_data[0]['inventory_bytes']} bytes, digest {live_data[0]['inventory_sha256'][:12]}…)")
    start, why = sweep_start_time(runs)
    if start is None:
        fails.append(f"sweep chronology cannot be established: {why}")
    gf = gate_ok(start, (rec, digest), recompute=not a.no_recompute_gate, manifests=manifests, live_ctx={"env": live_env, "data": live_data}); fails += gf
    print(("ok   " if not gf else "FAIL ") + f"k=0 gate independently validated (exact band rule, ten runs with content, bound, decision predates the sweep start {start}{'' if a.no_recompute_gate else ', decision recomputed with the pinned producer and byte-identical'})")
    for x in gf:
        print("     " + x)
    report = json.load(open(REPORT)) if os.path.isfile(REPORT) else None
    rf = check_report(report, bp.sha(REPORT_COPY) if os.path.isfile(REPORT_COPY) else None, bp.sha(REPORT) if os.path.isfile(REPORT) else None, list(runs) + sum(GATE_RUNS.values(), []))
    fails += rf; print(("ok   " if not rf else "FAIL ") + "binding report: canonical == assets copy; every live sidecar and gate input matches its retained digest")
    for x in rf:
        print("     " + x)
    per_run = {}
    for run in runs:
        tag = os.path.basename(run); print(f"== {run}")
        try:
            m = json.load(open(os.path.join(run, "metrics_yaw.json"))); ps = json.load(open(os.path.join(run, "per_sample_yaw.json"))); pv = json.load(open(os.path.join(run, "provenance.json")))
        except Exception as e:  # noqa: BLE001
            fails.append(f"{tag}: cannot read ({e})"); continue
        bb, ck = SWEEP_PROFILE[tag]; f = []
        f += check_meta(m["meta"], bb, ck)
        f += ["per-sample meta != aggregate meta"] if ps.get("meta") != m["meta"] else []
        f += check_manifest_order(ps.get("query"), ps.get("index"), raw["entries"])
        f += check_arrays(ps, m)
        f += check_delay_flips(m.get("delay_flips"), ps.get("delay_flips"))
        f += check_decomposition_pair(m.get("decomposition"), ps.get("decomposition"), tag)
        gate_side = json.load(open(f"ckpt/yaw_rotation/gate_{tag.split('_', 1)[1]}/provenance.json"))
        f += check_provenance(pv, run, rec, digest, gate_side, man_sha, head=a.launcher_head, live_env=live_env, live_data=live_data[0], meta=m["meta"], expected_ckpt=ck)
        for x in f:
            print("FAIL " + tag + ": " + x)
        if not f:
            print(f"ok   {tag}: meta, manifest order, content (finite, reconciled, P==E at k=0), delay flips, decomposition, provenance; elapsed {m['meta']['elapsed_min']:.1f} min")
        fails += [f"{tag}: {x}" for x in f]
        per_run[tag] = (ps.get("query"), ps.get("index"), m.get("delay_flips"))
    vals = list(per_run.values())
    if vals and not all(v == vals[0] for v in vals):
        fails.append("query lists, indices or delay-flip counts differ across the three runs")
    elif vals:
        print("ok   query lists, indices and delay-flip counts identical across the three runs")
    for x in fails:
        print("  - " + x)
    if fails:
        print(f"ACCEPTANCE FAIL ({len(fails)})"); sys.exit(1)
    if a.no_recompute_gate:
        print("ACCEPTANCE INCOMPLETE (gate decision not recomputed; run without --no-recompute-gate)"); sys.exit(2)
    print("ACCEPTANCE PASS"); sys.exit(0)


if __name__ == "__main__":
    main()
