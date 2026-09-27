#!/usr/bin/env python3
"""Side-by-side Δ / G / status of one arm evaluated on two backends (GPU primary vs CPU protocol),
from two summariser JSONs. Descriptive; no decision. Writes Markdown + JSON (with input sha256s).
usage: make_backend_table.py --gpu <summary.json> --cpu <summary.json> --arm released_k8 --out-md <md> --out-json <json>"""
import argparse
import hashlib
import json

KEYS = [("EDT", 1000.0, "ms"), ("C50", 1.0, "dB"), ("T60", 1.0, "pp / %"), ("T60_abs", 1000.0, "ms"),
        ("logspec_mad", 1.0, ""), ("mag_rel_l2", 1.0, ""), ("wave_rel_l2", 1.0, ""), ("wave_mad", 1.0, "")]
DEG = {"64": "45°", "128": "90°", "256": "180°", "384": "270°"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    G, C = json.load(open(a.gpu)), json.load(open(a.cpu))
    g, c = arm_of(G, a.arm), arm_of(C, a.arm)
    if g["meta"]["checkpoint_sha256"] != c["meta"]["checkpoint_sha256"] or g["meta"]["manifest_hash"] != c["meta"]["manifest_hash"] or g["meta"]["gl_seed"] != c["meta"]["gl_seed"]:
        raise ValueError("the two runs differ in checkpoint / manifest / gl_seed: not the same protocol up to the backend")
    rows, rec = [], {"arm": a.arm, "gpu": {"path": a.gpu, "sha256": sha(a.gpu), "execution_id": g["execution_id"], "protocol_id": g["protocol_id"], "device": g["meta"]["device"], "n": g["n_queries"]},
                     "cpu": {"path": a.cpu, "sha256": sha(a.cpu), "execution_id": c["execution_id"], "protocol_id": c["protocol_id"], "device": c["meta"]["device"], "n": c["n_queries"]}, "cells": []}
    for key, s, unit in KEYS:
        for k in ("64", "128", "256", "384"):
            cg, cc = g["angles"][k].get(key), c["angles"][k].get(key)
            if not cg or not cc:
                continue
            qg, qc = cg["query"], cc["query"]
            rows.append("| %s | %s | %s | %s | %s | %s | %s / %s |" % (key + ((" (%s)" % unit) if unit else ""), DEG[k], ci(qg.get("delta"), s), ci(qc.get("delta"), s), ci(qg.get("gap"), s), ci(qc.get("gap"), s), qg["ratio"]["status"], qc["ratio"]["status"]))
            rec["cells"].append({"metric": key, "k": int(k), "delta_gpu": qg.get("delta"), "delta_cpu": qc.get("delta"), "gap_gpu": qg.get("gap"), "gap_cpu": qc.get("gap"),
                                 "status_gpu": qg["ratio"]["status"], "status_cpu": qc["ratio"]["status"], "n_gpu": qg.get("n"), "n_cpu": qc.get("n")})
    md = ["# Backend sensitivity — %s: GPU (primary) vs CPU protocol" % a.arm, "",
          "Same checkpoint, references (manifest), Griffin-Lim seeds and settings; only the inference device differs (GPU run: %s, execution `%s`, n = %d; CPU run: %s, execution `%s`, n = %d). Query-level bootstrap intervals; descriptive only." % (
              g["meta"]["device"], g["execution_id"], g["n_queries"], c["meta"]["device"], c["execution_id"], c["n_queries"]), "",
          "| metric | angle | Δ GPU | Δ CPU | G GPU | G CPU | status GPU / CPU |", "|---|---|---|---|---|---|---|"] + rows + ["",
          "Inputs: `%s` sha256 `%s`; `%s` sha256 `%s`." % (a.gpu, rec["gpu"]["sha256"], a.cpu, rec["cpu"]["sha256"]), ""]
    open(a.out_md, "w").write("\n".join(md))
    json.dump(rec, open(a.out_json, "w"), indent=2, allow_nan=False)
    print("wrote", a.out_md, a.out_json, len(rows), "rows")


if __name__ == "__main__":
    main()
