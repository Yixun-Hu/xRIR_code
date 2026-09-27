#!/usr/bin/env python3
"""Side-by-side Δ / G / status of one arm evaluated on two backends (GPU primary vs CPU protocol),
from two summariser JSONs. Descriptive; no decision. Writes Markdown + JSON (with input sha256s).

The table's claim is "only the inference device differs", so that is *checked*, not assumed
(Codex tooling review, finding 3): the GPU summary must be a CUDA run and the CPU summary a
CPU run, they must be two different executions, and everything that would otherwise explain
a difference must be identical -- the query population (``query_list_sha256`` and
``n_queries``), the checkpoint, the reference manifest, the Griffin-Lim seed, K, the batch
shape, the precision flags, the evaluator implementation (``tool_sha256``) and the torch /
numpy versions. Both summaries' ``inputs`` bindings are re-verified against the live runs,
and each summary's embedded arm meta is compared with the live ``meta.json`` of the run it
names, so the fields above are the run's own and not an edited copy.

usage: make_backend_table.py --gpu <summary.json> --cpu <summary.json> --arm released_k8
                             --out-md <md> --out-json <json>
                             [--gpu-meta <meta.json>] [--cpu-meta <meta.json>]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_runs as vr                                          # noqa: E402

KEYS = [("EDT", 1000.0, "ms"), ("C50", 1.0, "dB"), ("T60", 1.0, "pp / %"), ("T60_abs", 1000.0, "ms"),
        ("logspec_mad", 1.0, ""), ("mag_rel_l2", 1.0, ""), ("wave_rel_l2", 1.0, ""), ("wave_mad", 1.0, "")]
DEG = {"64": "45°", "128": "90°", "256": "180°", "384": "270°"}
# Everything the two runs must agree on for "only the inference device differs" to hold.
PROTOCOL_FIELDS = [("query_list_sha256", "the query population"),
                   ("n_queries", "the number of queries"),
                   ("checkpoint_sha256", "the checkpoint"),
                   ("manifest_hash", "the reference manifest"),
                   ("manifest_seed", "the manifest seed"),
                   ("gl_seed", "the Griffin-Lim seed"),
                   ("num_shot", "K"),
                   ("batch_size", "the batch size"),
                   ("batch_canonical", "canonical batching"),
                   ("ks", "the angles"),
                   ("split", "the split"),
                   ("padded_len", "the padded length"),
                   ("metric_window", "the metric window"),
                   ("sample_rate", "the sample rate"),
                   ("cudnn_deterministic", "cuDNN determinism"),
                   ("cudnn_allow_tf32", "cuDNN TF32"),
                   ("matmul_allow_tf32", "matmul TF32"),
                   ("tf32", "the TF32 flag"),
                   ("tool_sha256", "the evaluator implementation"),
                   ("torch_version", "the torch version"),
                   ("numpy_version", "the numpy version")]


def sha(p):
    return vr.file_sha256(p)


def ci(d, s):
    if not d or d.get("point") is None:
        return "–"
    if d.get("lo") is None:
        return "%.3g" % (d["point"] * s)
    return "%.3g [%.3g, %.3g]" % (d["point"] * s, d["lo"] * s, d["hi"] * s)


def arm_of(summary, label):
    arms = [a for a in summary["arms"] if a["arm"] == label]
    if len(arms) != 1:
        raise ValueError("arm %s not found exactly once" % label)
    return arms[0]


def live_meta(summary, arm, override=None):
    """The ``meta.json`` of the run this summary names for ``arm`` (or ``--*-meta``)."""
    if override:
        return override, vr.read_json(override)
    for entry in summary.get("inputs") or []:
        if entry.get("arm") == arm:
            path = os.path.join(str(entry.get("run_dir")), "meta.json")
            if not os.path.isfile(path):
                raise ValueError("%s: the run this summary names has no meta.json; pass "
                                 "--gpu-meta / --cpu-meta if the run directory moved" % path)
            return path, vr.read_json(path)
    raise ValueError("the summary has no inputs entry for arm %s" % arm)


def check_pair(gpu_arm, cpu_arm, gpu_summary, cpu_summary, gpu_meta_path, cpu_meta_path,
               gpu_live, cpu_live, arm):
    """Every requirement the "same protocol up to the backend" claim rests on.

    Returns:
        The protocol fields that were verified equal (they go into the record and the caption).

    Raises:
        ValueError: listing every failed requirement.
    """
    problems = []
    problems.extend(vr.verify_summary_inputs(gpu_summary, label="GPU summary"))
    problems.extend(vr.verify_summary_inputs(cpu_summary, label="CPU summary"))
    for role, entry, live, path in (("GPU", gpu_arm, gpu_live, gpu_meta_path),
                                    ("CPU", cpu_arm, cpu_live, cpu_meta_path)):
        meta = entry.get("meta") or {}
        for field, _label in PROTOCOL_FIELDS + [("device", "the device"),
                                               ("execution_id", "the execution"),
                                               ("protocol_id", "the protocol")]:
            if field in meta and field in live and meta[field] != live[field]:
                problems.append("the %s summary's embedded meta.%s is %r, but the run it "
                                "names (%s) has %r" % (role, field, meta[field], path,
                                                       live[field]))
    if (gpu_arm.get("meta") or {}).get("device") != "cuda":
        problems.append("the --gpu summary's device is %r, not cuda: the primary backend's "
                        "role is not filled" % (gpu_arm.get("meta") or {}).get("device"))
    if (cpu_arm.get("meta") or {}).get("device") != "cpu":
        problems.append("the --cpu summary's device is %r, not cpu"
                        % (cpu_arm.get("meta") or {}).get("device"))
    for field in ("execution_id", "protocol_id"):
        if gpu_arm.get(field) == cpu_arm.get(field):
            problems.append("both summaries report the same %s (%s): that is one execution "
                            "compared with itself, not two backends"
                            % (field, gpu_arm.get(field)))
    if int(gpu_arm.get("n_queries", -1)) != int(cpu_arm.get("n_queries", -2)):
        problems.append("the summaries cover %s and %s queries; a backend comparison needs "
                        "the same query population" % (gpu_arm.get("n_queries"),
                                                       cpu_arm.get("n_queries")))
    verified = {}
    for field, label in PROTOCOL_FIELDS:
        left = (gpu_arm.get("meta") or {}).get(field)
        right = (cpu_arm.get("meta") or {}).get(field)
        if left is None and right is None:
            problems.append("neither run records %s (%s); the protocols cannot be compared"
                            % (field, label))
            continue
        if left != right:
            problems.append("the runs differ in %s (%s): %r vs %r; only the inference "
                            "device may differ" % (field, label, left, right))
        else:
            verified[field] = left
    if problems:
        raise ValueError("refusing to build the backend table for %s:\n  - %s"
                         % (arm, "\n  - ".join(problems)))
    return verified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--gpu-meta", default=None,
                    help="the GPU run's meta.json, if its directory moved since the summary")
    ap.add_argument("--cpu-meta", default=None)
    a = ap.parse_args()
    G, C = json.load(open(a.gpu)), json.load(open(a.cpu))
    g, c = arm_of(G, a.arm), arm_of(C, a.arm)
    gpu_meta_path, gpu_live = live_meta(G, a.arm, a.gpu_meta)
    cpu_meta_path, cpu_live = live_meta(C, a.arm, a.cpu_meta)
    verified = check_pair(g, c, G, C, gpu_meta_path, cpu_meta_path, gpu_live, cpu_live, a.arm)
    rows, rec = [], {"arm": a.arm, "verified": verified,
                     "gpu": {"path": a.gpu, "sha256": sha(a.gpu), "execution_id": g["execution_id"],
                             "protocol_id": g["protocol_id"], "device": g["meta"]["device"],
                             "n": g["n_queries"], "meta_path": gpu_meta_path,
                             "meta_sha256": sha(gpu_meta_path)},
                     "cpu": {"path": a.cpu, "sha256": sha(a.cpu), "execution_id": c["execution_id"],
                             "protocol_id": c["protocol_id"], "device": c["meta"]["device"],
                             "n": c["n_queries"], "meta_path": cpu_meta_path,
                             "meta_sha256": sha(cpu_meta_path)}, "cells": []}
    for key, s, unit in KEYS:
        for k in ("64", "128", "256", "384"):
            cg, cc = g["angles"].get(k, {}).get(key), c["angles"].get(k, {}).get(key)
            if not cg or not cc:
                continue
            qg, qc = cg["query"], cc["query"]
            rows.append("| %s | %s | %s | %s | %s | %s | %s / %s |" % (key + ((" (%s)" % unit) if unit else ""), DEG[k], ci(qg.get("delta"), s), ci(qc.get("delta"), s), ci(qg.get("gap"), s), ci(qc.get("gap"), s), qg["ratio"]["status"], qc["ratio"]["status"]))
            rec["cells"].append({"metric": key, "k": int(k), "delta_gpu": qg.get("delta"), "delta_cpu": qc.get("delta"), "gap_gpu": qg.get("gap"), "gap_cpu": qc.get("gap"),
                                 "status_gpu": qg["ratio"]["status"], "status_cpu": qc["ratio"]["status"], "n_gpu": qg.get("n"), "n_cpu": qc.get("n")})
    md = ["# Backend sensitivity — %s: GPU (primary) vs CPU protocol" % a.arm, "",
          "Same checkpoint, query population (`query_list_sha256` `%s`), references (manifest "
          "`%s`), Griffin-Lim seed, K, batch shape, precision flags, evaluator implementation "
          "(`tool_sha256` `%s`) and torch / numpy versions — every one of them verified equal; only "
          "the inference device differs (GPU run: %s, execution `%s`, n = %d; CPU run: %s, "
          "execution `%s`, n = %d). Query-level bootstrap intervals; descriptive only." % (
              verified.get("query_list_sha256"), verified.get("manifest_hash"),
              verified.get("tool_sha256"), g["meta"]["device"], g["execution_id"],
              g["n_queries"], c["meta"]["device"], c["execution_id"], c["n_queries"]), "",
          "| metric | angle | Δ GPU | Δ CPU | G GPU | G CPU | status GPU / CPU |", "|---|---|---|---|---|---|---|"] + rows + ["",
          "Inputs: `%s` sha256 `%s`; `%s` sha256 `%s`; live metas `%s` sha256 `%s` and `%s` sha256 `%s`." % (
              a.gpu, rec["gpu"]["sha256"], a.cpu, rec["cpu"]["sha256"], gpu_meta_path,
              rec["gpu"]["meta_sha256"], cpu_meta_path, rec["cpu"]["meta_sha256"]), ""]
    open(a.out_md, "w").write("\n".join(md))
    json.dump(rec, open(a.out_json, "w"), indent=2, allow_nan=False, sort_keys=True)
    print("wrote", a.out_md, a.out_json, len(rows), "rows")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, vr.ValidationError) as exc:      # a refusal, not a crash
        sys.stderr.write("REFUSED: %s\n" % exc)
        sys.exit(2)
