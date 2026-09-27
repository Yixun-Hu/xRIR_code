#!/usr/bin/env python3
"""Write exp_10's `_results.md` from the canonical summariser JSON (+ optional parity / check-online JSONs).

usage: make_results_md.py --summary <yaw_pilot_summary.json> --out <yaw_pilot_results.md>
       [--parity <arm>=<parity_exp03.json> ...] [--check-online <arm>=<check_online.json> ...]
Only the headline metrics (EDT, C50, T60, T60_abs, logspec_mad) go into the compact tables; the
summariser's own `yaw_pilot_tables.md` (copied into the assets) carries everything else.
"""
import argparse
import hashlib
import json
import os

HEAD = [("EDT", "EDT", 1000.0, "ms"), ("C50", "C50", 1.0, "dB"), ("T60", "T60", 1.0, "pp (Δ) / % (G)"),
        ("T60_abs", "T60 absolute", 1000.0, "ms"), ("logspec_mad", "log-spec MAD (GL-free)", 1.0, "")]
DEG = {"0": "0°", "64": "45°", "128": "90°", "256": "180°", "384": "270°"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def n3(x, s=1.0):
    return "–" if x is None else "%.3g" % (x * s)


def ci(d, s=1.0):
    if not d or d.get("point") is None:
        return "–"
    if d.get("lo") is None or d.get("hi") is None:
        return "%s (no CI)" % n3(d["point"], s)
    return "%s [%s, %s]" % (n3(d["point"], s), n3(d["lo"], s), n3(d["hi"], s))


def multiple(cell):
    h = cell.get("headline", {})
    if h.get("reportable"):
        return "**%s×** [%s, %s]" % (n3(h["point"]), n3(h.get("lower_bound")), n3(h.get("upper_bound")))
    return h.get("reason") or cell["query"].get("ratio", {}).get("reason") or "–"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parity", nargs="*", default=[])
    ap.add_argument("--check-online", nargs="*", default=[])
    a = ap.parse_args()
    s = json.load(open(a.summary))
    parity = {kv.split("=", 1)[0]: (kv.split("=", 1)[1], json.load(open(kv.split("=", 1)[1]))) for kv in a.parity}
    online = {kv.split("=", 1)[0]: (kv.split("=", 1)[1], json.load(open(kv.split("=", 1)[1]))) for kv in a.check_online}
    L = ["# Results — exp_10 yaw_pilot", "",
         "Descriptive pilot (plan v3.1). Per arm and angle: Δ = paired mean change of the error vs ground truth (α − 0°); G = mean shift of the prediction at α relative to the prediction at 0° (no ground truth); both on the shared comparison mask, %d bootstrap replicates (seeds %s), %.0f %% percentile intervals; query-level CI first, room-cluster CI (17 rooms) second. A multiple G/Δ is printed only where the headline is reportable (Δ interval excludes 0, convergence passed, seed statuses agree); otherwise the cell shows its status. Canonical JSON: `%s` (sha256 `%s`)." % (
             s.get("n_boot", 0), s.get("seeds"), 100 * (1 - s.get("alpha", 0.05)), a.summary, sha(a.summary)), ""]
    for n in s.get("notes", []):
        L.append("> " + n)
    L.append("")
    for arm in s["arms"]:
        m = arm["meta"]
        L += ["## %s" % arm["arm"], "",
              "`%s` (sha256 `%s…`), backbone %s, K = %s, device %s, %d queries (batches `%s`), manifest `%s…`, gl_seed %s, execution `%s`." % (
                  m.get("checkpoint"), str(m.get("checkpoint_sha256"))[:12], m.get("backbone"), m.get("num_shot"), m.get("device"),
                  arm.get("n_queries", 0), m.get("batches_arg"), str(m.get("manifest_hash"))[:12], m.get("gl_seed"), arm.get("execution_id")), ""]
        for key, label, sc, unit in HEAD:
            rows = []
            for k in sorted(arm["angles"], key=int):
                if k == "0" or key not in arm["angles"][k]:
                    continue
                c = arm["angles"][k][key]; q = c["query"]; r = c.get("room", {})
                rows.append("| %s | %s | %s | %s | %s | %s | %s | %d | %s / %s | %s |" % (
                    DEG.get(k, k), n3(q.get("mean_0"), sc), n3(q.get("mean_alpha"), sc), ci(q.get("delta"), sc), ci(r.get("delta"), sc),
                    ci(q.get("gap"), sc), ci(r.get("gap"), sc), q.get("n", 0), q.get("ratio", {}).get("status", "–"), c.get("room_status", "–"), multiple(c)))
            if rows:
                L += ["### %s (%s)" % (label, unit or "dimensionless"), "",
                      "| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |",
                      "|---|---|---|---|---|---|---|---|---|---|"] + rows + [""]
        ctr = arm.get("controls", {})
        if ctr:
            L += ["**Controls:** ok = %s; " % ctr.get("ok") + "; ".join("%s (k=%s) max|Δwave| %s, max|Δlogspec| %s" % (
                n, c.get("k"), n3(c.get("wave_max_abs_diff")), n3(c.get("logspec_max_abs_diff"))) for n, c in sorted(ctr.get("controls", {}).items())), ""]
        if arm["arm"] in parity:
            path, p = parity[arm["arm"]]
            worst = {}
            for k, cells in p.get("angles", {}).items():
                for mk, c in cells.items():
                    if isinstance(c, dict) and c.get("max_abs_diff") is not None:
                        worst[mk] = max(worst.get(mk, 0.0), c["max_abs_diff"])
            L += ["**Parity vs exp_03** (`%s`, sha256 `%s…`): ok = %s, rows %s, missing required %s; largest per-query |Δ|: %s." % (
                path, sha(path)[:12], p.get("ok"), p.get("n_rows"), p.get("missing_required"), ", ".join("%s %s" % (k, n3(v)) for k, v in sorted(worst.items()))), ""]
        if arm["arm"] in online:
            path, o = online[arm["arm"]]
            L += ["**check_online** (`%s`): ok = %s." % (path, o.get("ok")), ""]
    L += ["## Provenance", ""] + ["- %s: execution `%s`, per_sample sha256 `%s`, meta sha256 `%s`, run dir `%s`" % (
        i["arm"], i["execution_id"], i["per_sample_sha256"], i["meta_sha256"], i["run_dir"]) for i in s.get("inputs", [])] + [""]
    with open(a.out, "w") as f:
        f.write("\n".join(L))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
