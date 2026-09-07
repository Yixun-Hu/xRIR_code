"""Post-hoc source-closure, environment and data-identity binding for the exp_03 runs (Codex round-1 finding 8, round-2 7-8, round-3 1).

Binds the three sweep runs AND the ten k=0 gate runs (or, with --set gate, the ten gate runs alone before a sweep exists),
addressed by their exact canonical paths, to the reviewed source closure (every repo-local Python file the evaluator imports, found by importing it; each file's working-tree bytes must
equal the reviewed commit's blob, no commit after the reviewed one may touch it, and its mtime must precede every run's
start), to the software/hardware environment (post hoc, same host and conda env), and to the data identity (sha256 of the
manifest file bytes, the semantic manifest hash, and an inventory digest over (relative path, sha256 of the bytes) of every dataset
file the manifest references under the data root the runs used), and records the digests of every other decision-driving input:
the run logs, the retained gate-decision copies, the gate producer blob (tools/summarize_yaw.py at the commit that took the
decision), the two exp_01 per-sample files the gate compares against, and the canonical gate JSON/summary. Contract: all 13 runs required (--set gate: the ten gate runs, before a
decision exists; gate inputs are then recorded only if present); every check on every run passes before any sidecar is
written; **bindings are immutable**: once a sidecar carries binding fields, every previously bound field must equal the freshly
computed value or the binder refuses and reports the drift (nothing is overwritten); the retained report
ckpt/yaw_rotation/binding_report.json (copied into the record folder) carries the complete per-run binding record and is likewise
refused on drift -- only additive migrations (new keys, gate inputs that were absent before a decision existed, run_set gate->all,
append-only binding history) are accepted and printed. The whole transaction is validated before any write; the files are then
replaced one by one (each replacement atomic; the sequence is NOT crash-atomic across files -- a failure between two replacements
leaves a detectable partial transaction). Exit 1 on any failure or drift.

    python .../bind_provenance.py
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys

REVIEWED_COMMIT = "62c9107b4150e44c4ac410ff4cab359c1e71cc10"
REVIEWED_EVAL_SHA = "82a53bea3cd74b4805ea1680fb457fec27abf02e3d24a0997d21a5b2e7f21171"
EVALUATOR = "eval_yaw_rotation.py"
SWEEP_RUNS = ("ckpt/yaw_rotation/sweep_control", "ckpt/yaw_rotation/sweep_cyl", "ckpt/yaw_rotation/sweep_released")
GATE_RUNS = tuple(f"ckpt/yaw_rotation/gate_{t}" for t in ("control", "cyl", "released", "noise_seed1", "noise_seed2", "phase_seed1", "tf32", "noise_cyl_seed1", "noise_cyl_seed2", "shape_cyl_b1"))
ROLE_OF = {"sweep_control": "control", "sweep_cyl": "cyl", "sweep_released": "released"}
MANIFESTS = {0: "ckpt/yaw_rotation/reference_manifest.json", 1: "ckpt/yaw_rotation/reference_manifest_seed1.json", 2: "ckpt/yaw_rotation/reference_manifest_seed2.json"}
DATA_ROOT = "/home/yixunhu/data_cache/AcousticRooms"
RECORD = "worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude"
GATE_PRODUCER_COMMIT = "5febf471318e37e60477ebe72afd48f03879c67e"   # HEAD when the v3 gate decision was taken
GATE_PRODUCER_FILES = ("tools/summarize_yaw.py", "tools/paired_stats.py", "tools/__init__.py")   # repo-local closure of the gate producer (import graph at the pinned commit)
GATE_INPUTS = {"exp01_control": "ckpt/xRIR_simple_8_shot/per_sample_unseen_epoch12.json", "exp01_cyl": "ckpt/xRIR_cyl_8_shot/per_sample_unseen_epoch12.json",
               "gate_json": "ckpt/yaw_rotation/gate_stats.json", "gate_summary": "ckpt/yaw_rotation/gate_summary.txt",
               "decision_v1": f"{RECORD}/yaw_rotation_degradation_2026-09-06_14:53:35_gate_decision_v1_FAILED.txt",
               "decision_v2": f"{RECORD}/yaw_rotation_degradation_2026-09-06_15:07:30_gate_decision_v2_FAILED.txt",
               "decision_v3": f"{RECORD}/yaw_rotation_degradation_2026-09-06_15:40:32_gate_decision_v3_PASSED.txt"}
PYTHON = os.path.expanduser("~/miniconda3/envs/xRIR/bin/python")
REPORT = "ckpt/yaw_rotation/binding_report.json"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args]).decode()


def source_closure(repo):
    """Repo-local files imported by the evaluator, found by importing it in a subprocess (sorted, repo-relative)."""
    code = (
        "import sys, os, importlib; repo = os.getcwd(); sys.path.insert(0, repo); importlib.import_module('%s');"
        "files = sorted({os.path.relpath(os.path.abspath(m.__file__), repo) for m in list(sys.modules.values())"
        " if getattr(m, '__file__', None) and os.path.abspath(m.__file__).startswith(repo + os.sep)"
        " and 'site-packages' not in m.__file__ and os.sep + 'envs' + os.sep not in m.__file__});"
        "print('\\n'.join(files))" % EVALUATOR[:-3])
    out = subprocess.check_output([PYTHON, "-c", code], cwd=repo, env={**os.environ, "PYTHONPATH": repo}).decode().split()
    return [f for f in sorted(set(out) | {EVALUATOR}) if not f.startswith("tests" + os.sep)]


def closure_record(files, repo):
    rec = []
    for f in files:
        try:
            blob = hashlib.sha256(subprocess.check_output(["git", "show", f"{REVIEWED_COMMIT}:{f}"], stderr=subprocess.DEVNULL)).hexdigest()
        except subprocess.CalledProcessError:
            blob = None
        later = git("log", "--format=%H", f"{REVIEWED_COMMIT}..HEAD", "--", f).split()
        rec.append({"path": f, "reviewed_blob_sha256": blob, "working_tree_sha256": sha(os.path.join(repo, f)), "commits_after_reviewed": later,
                    "mtime": datetime.datetime.fromtimestamp(os.stat(os.path.join(repo, f)).st_mtime).astimezone().isoformat()})
    digest = hashlib.sha256(json.dumps([[r["path"], r["reviewed_blob_sha256"]] for r in rec], sort_keys=True).encode()).hexdigest()
    return rec, digest


def environment():
    code = ("import torch, numpy, platform, json, torchaudio, torchvision, einops, scipy;"
            "print(json.dumps({'python': platform.python_version(), 'torch': torch.__version__, 'torchaudio': torchaudio.__version__,"
            " 'torchvision': torchvision.__version__, 'einops': einops.__version__, 'numpy': numpy.__version__, 'scipy': scipy.__version__,"
            " 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),"
            " 'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))")
    env = json.loads(subprocess.check_output([PYTHON, "-c", code]).decode())
    try:
        env["nvidia_driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).decode().split()[0]
    except Exception:  # noqa: BLE001
        env["nvidia_driver"] = None
    env["host"] = platform.node(); env["python_executable"] = PYTHON
    env["note"] = "recorded post hoc by bind_provenance.py on the host and conda env that ran the runs; each run's metrics meta records torch_version independently"
    return env


def data_identity(manifest_path):
    """Semantic manifest hash, sha256 of the manifest file, and an inventory digest over (relpath, sha256 of the bytes) of every referenced dataset file."""
    sys.path.insert(0, os.getcwd())
    from tools.reference_manifest import load_manifest, manifest_hash
    m = load_manifest(manifest_path)
    raw = json.load(open(manifest_path))
    wavs = set()
    def collect(x):
        if isinstance(x, str) and x.endswith(".wav"):
            wavs.add(x)
        elif isinstance(x, dict):
            for v in x.values():
                collect(v)
        elif isinstance(x, list):
            for v in x:
                collect(v)
    collect(raw)
    files = set()
    for w in wavs:
        rel = w.split("single_channel_ir/")[-1] if "single_channel_ir/" in w else w
        cat_room, fname = rel.rsplit("/", 1)
        # same derivation as treble_xRIR_dataset.py: S<src>_R<rec>_hybrid_IR.wav -> metadata S00<src>_R00<rec>.json (ids unpadded), depth <rec>.npy
        src, rec = int(fname.split("_")[0][1:]), int(fname.split("_")[1][1:])
        files.add(f"single_channel_ir/{cat_room}/{fname}"); files.add(f"metadata/{cat_room}/S00{src}_R00{rec}.json"); files.add(f"depth_map/{cat_room}/{rec}.npy")
    inv = []
    missing = 0
    total = 0
    for f in sorted(files):
        p = os.path.join(DATA_ROOT, f)
        if os.path.isfile(p):
            inv.append([f, sha(p)]); total += os.path.getsize(p)
        else:
            missing += 1
    digest = hashlib.sha256(json.dumps(inv, sort_keys=True).encode()).hexdigest()
    return {"manifest_path": manifest_path, "manifest_file_sha256": sha(manifest_path), "manifest_hash": manifest_hash(m), "data_root": DATA_ROOT,
            "inventory_files": len(inv), "inventory_missing": missing, "inventory_bytes": total, "inventory_sha256": digest,
            "inventory_key": "(relative path, sha256 of the file bytes) of every IR wav, metadata json and depth map the manifest references"}   # bytes, not sizes


def gate_producer_closure():
    """Blob digests (at the pinned commit) and working-tree digests of every repo-local file the gate producer imports."""
    rec = []
    for f in GATE_PRODUCER_FILES:
        r = subprocess.run(["git", "show", f"{GATE_PRODUCER_COMMIT}:{f}"], capture_output=True)
        if r.returncode != 0:
            continue   # e.g. no tools/__init__.py at that commit
        rec.append({"path": f, "pinned_blob_sha256": hashlib.sha256(r.stdout).hexdigest(), "working_tree_sha256": sha(f) if os.path.isfile(f) else None})
    return rec


def gate_inputs(require=True):
    """Digests of every non-run input of the k=0 gate decision and of the decision artefacts themselves."""
    blob = subprocess.check_output(["git", "show", f"{GATE_PRODUCER_COMMIT}:tools/summarize_yaw.py"])
    out = {"gate_producer_commit": GATE_PRODUCER_COMMIT, "gate_producer_sha256": hashlib.sha256(blob).hexdigest(), "gate_producer_closure": gate_producer_closure()}
    for k, pth in GATE_INPUTS.items():
        out[k] = {"path": pth, "sha256": sha(pth) if os.path.isfile(pth) else None}
    return out


BINDING_KEYS = ("evaluator_sha256", "reviewed_evaluator_commit", "launcher_head", "git_sha_note", "source_closure_sha256", "source_closure",
                "environment", "data_identity", "run_start", "log_sha256", "checkpoint_approved_source")


def binding_drift(existing, new):
    """Sidecar drift: any binding key present before whose value differs, or any previously bound key now absent from the fresh
    computation (deletion). Keys never bound before are additive, not drift."""
    out = sorted(k for k in BINDING_KEYS if k in existing and (k not in new or existing[k] != new[k]))
    return out


def _deep_drift(prev, new, path=""):
    """Recursive comparison: every retained value must survive unchanged; removals are drift; additions are not.
    `binding_history` lists are append-only: the retained prefix must be identical, new entries may follow."""
    out = []
    if isinstance(prev, dict) and isinstance(new, dict):
        for k, v in prev.items():
            if k not in new:
                out.append(f"{path}{k} (removed)")
            elif k == "binding_history" and isinstance(v, list) and isinstance(new[k], list):
                if new[k][:len(v)] != v:
                    out.append(f"{path}{k} (retained history altered)")
            else:
                out += _deep_drift(v, new[k], f"{path}{k}.")
    elif prev != new:
        out.append(path.rstrip("."))
    return out


def project(plans, records, now):
    """The projection step of main(), separated for testing: the sidecars after binding (with bound_at set once and the
    history appended) and the per-run report records with the projected sidecar digests."""
    projected = {}
    for side, (pv, binding, unchanged) in plans.items():
        new_pv = dict(pv)
        if not unchanged:
            new_pv.update(binding); new_pv.setdefault("bound_at", now)
            new_pv["binding_history"] = list(new_pv.get("binding_history", [])) + [{"at": now, "added": sorted(set(binding) - set(pv))}]
        projected[side] = new_pv
    for run in records:
        side = os.path.join(run, "provenance.json")
        blob = json.dumps(projected[side], indent=1).encode()
        records[run]["sidecar_after_binding"] = projected[side]; records[run]["sidecar_sha256"] = hashlib.sha256(blob).hexdigest()
    return projected


def report_drift(prev, new):
    """Report drift, ignoring `written` and `binding_history`. Allowed migrations: a gate input whose retained record was
    {path, sha256: null} becoming {same path, sha256: digest}; run_set 'gate' -> 'all' with the three sweep runs added;
    additions of new keys. Everything else -- changed values, removed keys, changed paths -- is drift."""
    out = []
    for k, v in prev.items():
        if k in ("written", "binding_history", "last_migration"):
            continue
        if k not in new:
            out.append(f"{k} (removed)"); continue
        if k == "gate_inputs":
            for kk, vv in v.items():
                nv = new[k].get(kk, None) if isinstance(new[k], dict) else None
                if kk not in new[k]:
                    out.append(f"gate_inputs.{kk} (removed)")
                elif vv != nv and not (isinstance(vv, dict) and isinstance(nv, dict) and vv.get("sha256") is None and vv.get("path") == nv.get("path") and set(nv) == set(vv)):
                    out.append(f"gate_inputs.{kk}")
        elif k == "run_set":
            if not (v == new[k] or (v == "gate" and new[k] == "all")):
                out.append("run_set")
        elif k == "runs":
            for run, rec in v.items():
                if run not in new[k]:
                    out.append(f"runs.{run} (removed)"); continue
                nrec = new[k][run]
                for kk, vv in rec.items():
                    if kk == "sidecar_sha256":
                        if vv != nrec.get(kk):
                            # allowed only when both records carry the sidecar content and the change is purely additive
                            old_sc, new_sc = rec.get("sidecar_after_binding"), nrec.get("sidecar_after_binding")
                            if not isinstance(old_sc, dict) or not isinstance(new_sc, dict) or _deep_drift(old_sc, new_sc, f"runs.{run}.sidecar."):
                                out.append(f"runs.{run}.sidecar_sha256 (sidecar changed beyond an additive migration)")
                    elif kk == "sidecar_after_binding":
                        out += _deep_drift(vv, nrec.get(kk), f"runs.{run}.sidecar_after_binding.")
                    elif kk not in nrec:
                        out.append(f"runs.{run}.{kk} (removed)")
                    else:
                        out += _deep_drift(vv, nrec[kk], f"runs.{run}.{kk}.")
        else:
            out += _deep_drift(v, new[k], f"{k}.")
    return out


def commit_transaction(projected, originals, report, report_path, now):
    """Write phase, separated for testing: sidecars that changed, then the report (each file replaced atomically; the sequence is not crash-atomic). Returns the list written.
    Callers must have validated the projection (drift) before calling; this function performs no further checks."""
    written = []
    for side, new_pv in projected.items():
        if new_pv == originals[side]:
            continue
        tmp = side + ".tmp"
        with open(tmp, "w") as f:
            f.write(json.dumps(new_pv, indent=1)); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, side); written.append(side)
    prev = json.load(open(report_path)) if os.path.isfile(report_path) else None
    unchanged = bool(prev) and {k: v for k, v in prev.items() if k not in ("written", "binding_history", "last_migration")} == report
    if not unchanged:
        if prev:
            added = sorted(set(report) - set(prev)) + [f"gate_inputs.{k}" for k in report.get("gate_inputs", {}) if k not in prev.get("gate_inputs", {})] + [f"runs.{r}" for r in report["runs"] if r not in prev.get("runs", {})]
            report["binding_history"] = prev.get("binding_history", []) + [{"at": now, "migration": added or "sidecar additive migration / gate inputs now present"}]
            report["written"] = prev.get("written", now); report["last_migration"] = now
        else:
            report["written"] = now
        with open(report_path + ".tmp", "w") as f:
            json.dump(report, f, indent=1); f.flush(); os.fsync(f.fileno())
        os.replace(report_path + ".tmp", report_path); written.append(report_path)
    return written


def run_start(pv, run):
    """Start time from the bound log's canonical name: <record>/yaw_rotation_degradation_<ts>_<gate|sweep>_<tag>.log (never an aborted log)."""
    log = pv.get("log", "")
    tag = os.path.basename(run); mode, _, short = tag.partition("_")
    m = re.fullmatch(r"yaw_rotation_degradation_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})_" + re.escape(mode) + "_" + re.escape(short) + r"\.log", os.path.basename(log))
    if not m or not os.path.isfile(log) or os.path.dirname(os.path.abspath(log)) != os.path.abspath(RECORD):
        return None, "log missing, not in the record folder, or not named <ts>_<mode>_<tag>.log"
    start = datetime.datetime.strptime(m.group(1), "%Y-%m-%d_%H:%M:%S").astimezone()
    meta = json.load(open(os.path.join(run, "metrics_yaw.json")))["meta"]
    if pv.get("written"):
        implied = datetime.datetime.fromisoformat(pv["written"]) - datetime.timedelta(minutes=meta["elapsed_min"])
        if implied < start - datetime.timedelta(minutes=10):
            return None, f"log timestamp {start.isoformat()} later than written − elapsed = {implied.isoformat()}"
    if datetime.datetime.fromtimestamp(os.stat(log).st_mtime).astimezone() < start:
        return None, "log mtime precedes its own start timestamp"
    return start, "ok"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", choices=["all", "gate"], default="all", help="all = 3 sweep + 10 gate runs (default); gate = the ten gate runs only (before a sweep exists)")
    a = ap.parse_args(); repo = os.getcwd(); bad = []
    runs = list(SWEEP_RUNS + GATE_RUNS) if a.set == "all" else list(GATE_RUNS)
    missing_runs = [r for r in runs if not os.path.isfile(os.path.join(r, "provenance.json"))]
    if missing_runs:
        print(f"REFUSED: missing sidecars for {missing_runs}"); sys.exit(1)
    files = source_closure(repo); rec, digest = closure_record(files, repo)
    print(f"source closure ({len(files)} files) reviewed at {REVIEWED_COMMIT}; digest {digest}")
    for r in rec:
        ok = r["reviewed_blob_sha256"] is not None and r["reviewed_blob_sha256"] == r["working_tree_sha256"] and not r["commits_after_reviewed"]
        print(("ok   " if ok else "FAIL ") + f"{r['path']}: blob {str(r['reviewed_blob_sha256'])[:12]} == tree {r['working_tree_sha256'][:12]}; later commits {r['commits_after_reviewed'] or 'none'}; mtime {r['mtime']}")
        if not ok:
            bad.append(r["path"])
    ev = next(r for r in rec if r["path"] == EVALUATOR)
    if ev["working_tree_sha256"] != REVIEWED_EVAL_SHA:
        bad.append("evaluator sha != pinned"); print(f"FAIL evaluator sha {ev['working_tree_sha256'][:12]} != pinned")
    env = environment(); print("environment:", json.dumps(env))
    data = {str(s): data_identity(p) for s, p in MANIFESTS.items()}
    for s, d in data.items():
        print(f"data identity (manifest seed {s}): manifest_hash {d['manifest_hash'][:12]}, file sha {d['manifest_file_sha256'][:12]}, inventory {d['inventory_files']} files, missing {d['inventory_missing']}, digest {d['inventory_sha256'][:12]}")
        if d["inventory_missing"]:
            bad.append(f"data files missing for manifest seed {s}")
    ginputs = gate_inputs(); print("gate inputs:", json.dumps({k: (v["sha256"][:12] if isinstance(v, dict) and v.get("sha256") else (v if not isinstance(v, list) else f"{len(v)} files")) for k, v in ginputs.items()}))
    if a.set == "all" and any(isinstance(v, dict) and "sha256" in v and v["sha256"] is None for v in ginputs.values()):
        bad.append("gate input file missing")
    if any(r["working_tree_sha256"] != r["pinned_blob_sha256"] for r in ginputs["gate_producer_closure"]):
        print("note: the working-tree gate producer differs from the pinned commit (expected after later producer commits); the recomputation uses the pinned blobs")
    latest_mtime = max(datetime.datetime.fromisoformat(r["mtime"]) for r in rec)
    plans = {}; records = {}
    for run in runs:
        tag = os.path.basename(run); side = os.path.join(run, "provenance.json")
        pv = json.load(open(side)); meta = json.load(open(os.path.join(run, "metrics_yaw.json")))["meta"]
        start, why = run_start(pv, run)
        role = ROLE_OF.get(tag); gate = json.load(open(f"ckpt/yaw_rotation/gate_{role}/provenance.json")) if role else None
        d = data[str(meta["manifest_seed"])]
        log_sha = sha(pv["log"]) if pv.get("log") and os.path.isfile(pv["log"]) else None
        checks = {
            f"start time consistent ({why})": start is not None,
            "run log present in the record folder": log_sha is not None,
            "every closure file last modified before the run started": start is not None and latest_mtime < start,
            "checkpoint path identical in the sidecar and the metrics meta, digest equals the file on disk": pv.get("checkpoint") == meta.get("checkpoint") and pv.get("checkpoint_sha256") == sha(pv["checkpoint"]),
            "artefact digests unchanged": pv.get("per_sample_sha256") == sha(os.path.join(run, "per_sample_yaw.json")) and pv.get("metrics_sha256") == sha(os.path.join(run, "metrics_yaw.json")),
            "manifest hash equals the metrics meta and the manifest file": meta["manifest_hash"] == pv.get("manifest_hash") == d["manifest_hash"] and meta["manifest_path"] == d["manifest_path"],
            "torch version in metrics meta equals the environment's": meta.get("torch_version") == env["torch"],
            "existing binding (if any) agrees": pv.get("source_closure_sha256") in (None, digest) and pv.get("evaluator_sha256") in (None, REVIEWED_EVAL_SHA),
        }
        if gate is not None:
            checks["checkpoint digest equals the gate sidecar's (approved)"] = pv.get("checkpoint_sha256") == gate.get("checkpoint_sha256")
        for name, ok in checks.items():
            print(("ok   " if ok else "FAIL ") + f"{run}: {name}")
        if not all(checks.values()):
            bad.append(run); continue
        binding = {"evaluator_sha256": REVIEWED_EVAL_SHA, "reviewed_evaluator_commit": REVIEWED_COMMIT, "launcher_head": pv["git_sha"],
                   "git_sha_note": "git_sha is the launcher HEAD at run time (or the pinned reviewed commit for sidecars written post hoc); the evaluator and its repo-local imports are bound by source_closure to the reviewed commit (verified post hoc by bind_provenance.py)",
                   "source_closure_sha256": digest, "source_closure": rec, "environment": env, "data_identity": d, "run_start": start.isoformat(), "log_sha256": log_sha,
                   "checkpoint_approved_source": f"ckpt/yaw_rotation/gate_{role}/provenance.json" if role else "self (gate run)"}
        drift = binding_drift(pv, binding)
        if drift:
            print(f"DRIFT {run}: previously bound fields differ from the fresh computation: {drift}"); bad.append(f"drift {run}")
        unchanged = all(pv.get(k) == v for k, v in binding.items())
        plans[side] = (pv, binding, unchanged)
        records[run] = {"sidecar": side, "run_start": start.isoformat(), "elapsed_min": meta["elapsed_min"], "backbone": meta["backbone"],
                        "batch_size": meta["batch_size"], "tf32": meta["tf32"], "gl_seed": meta["gl_seed"], "manifest_seed": meta["manifest_seed"],
                        "per_sample_sha256": pv["per_sample_sha256"], "metrics_sha256": pv["metrics_sha256"], "checkpoint": pv["checkpoint"], "checkpoint_sha256": pv["checkpoint_sha256"],
                        "command": pv.get("command"), "command_note": "verbatim from the launcher" if pv.get("command") and "<0|1>" not in pv["command"] else ("launcher v1 template with a <0|1> GPU placeholder" if pv.get("command") else "sidecar written post hoc; command reconstructed in _command.md"),
                        "log": pv.get("log"), "log_sha256": log_sha, "launcher_head": pv["git_sha"]}
    if bad:
        print("BIND FAILED:", bad, "— nothing written"); sys.exit(1)
    # ---- project the whole transaction (every sidecar after binding + the report) BEFORE writing anything (validated as a whole; writes are sequential, not crash-atomic)
    now = datetime.datetime.now().astimezone().isoformat()
    projected = project(plans, records, now)
    report = {"reviewed_commit": REVIEWED_COMMIT, "evaluator_sha256": REVIEWED_EVAL_SHA, "source_closure_sha256": digest, "source_closure": rec, "environment": env,
              "data_identity": data, "gate_inputs": ginputs, "run_set": a.set, "runs": records}
    prev = json.load(open(REPORT)) if os.path.isfile(REPORT) else None
    unchanged_report = bool(prev) and {k: v for k, v in prev.items() if k not in ("written", "binding_history", "last_migration")} == report
    if prev and not unchanged_report:
        drift = report_drift(prev, report)
        if drift:
            print("REPORT DRIFT (refusing the whole transaction; nothing written):", drift); sys.exit(1)
        print("report migration (monotonic; validated before any write)")
    originals = {side: pv for side, (pv, _, _) in plans.items()}
    written = commit_transaction(projected, originals, report, REPORT, now)
    for side in projected:
        print(("bound " if side in written else "unchanged ") + side + ("" if side in written else " (binding already valid)"))
    print(("report written: " if REPORT in written else "report unchanged: ") + REPORT)
    print("BIND OK"); sys.exit(0)


if __name__ == "__main__":
    main()
