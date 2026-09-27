#!/usr/bin/env python3
"""Write exp_10's `_results.md` from the canonical summariser JSON (+ optional parity / check-online JSONs).

usage: make_results_md.py --summary <yaw_pilot_summary.json> --out <yaw_pilot_results.md>
       [--parity <arm>=<parity_exp03.json> ...] [--check-online <arm>=<check_online.json> ...]
Only the headline metrics (EDT, C50, T60, T60_abs, logspec_mad) go into the compact tables; the
summariser's own `yaw_pilot_tables.md` (copied into the assets) carries everything else.

As in the HTML page, nothing is written unbound (Codex tooling review, finding 4): the
summary's `inputs` hashes are re-verified against the live runs and every supplemental report
has to belong to the arm it is attached to.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_runs as vr                                          # noqa: E402

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


def note_lines(summary):
    """The canonical qualifications, rendered as their *values* (finding 5).

    ``notes`` is a dict (``band`` / ``broader_population`` / ``gl_free`` / ``pipeline``): the
    keys are labels, the values are the sentences the report has to carry.  The
    ``metric_qualifications`` block says what a Δ and a G mean per metric -- including T60's
    two different denominators -- and belongs in the Markdown as much as in the page.
    """
    lines = []
    notes = summary.get("notes") or {}
    items = (sorted(notes.items()) if isinstance(notes, dict)
             else [(None, text) for text in notes])
    for key, text in items:
        lines.append("> **%s:** %s" % (key, text) if key else "> %s" % text)
    lines.append("")
    qualifications = summary.get("metric_qualifications") or {}
    if qualifications:
        lines += ["**Metric qualifications** (canonical, from the summariser): what a Δ and a "
                  "G mean for each metric, including T60's two denominators.", "",
                  "| metric | Δ (change in the error vs ground truth) | G (shift from the 0° prediction) |",
                  "|---|---|---|"]
        for metric, entry in sorted(qualifications.items()):
            if isinstance(entry, dict):
                lines.append("| %s | %s | %s |" % (metric, entry.get("delta", "–"),
                                                   entry.get("gap", "–")))
            else:
                lines.append("| %s | %s | – |" % (metric, entry))
        lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parity", nargs="*", default=[])
    ap.add_argument("--check-online", nargs="*", default=[])
    ap.add_argument("--probe", nargs="*", default=[], help="<arm>=<probe summary JSON>: controls are rendered from the probe run")
    ap.add_argument("--assets", default=None,
                    help="the directory the summariser's tables / JSON / CSV are copied to; "
                         "the supplementary links point into it")
    ap.add_argument("--assets-href", default=None,
                    help="the published location of --assets, for the relative links")
    ap.add_argument("--backend-table", default=None,
                    help="the GPU-vs-CPU backend-sensitivity Markdown, linked and hashed here")
    ap.add_argument("--cpu-record", default=None, help="the CPU-protocol record (its tables)")
    a = ap.parse_args()
    vr.assert_ok(["%s: %s is missing" % (flag, path)
                  for flag, path in (("--backend-table", a.backend_table),
                                     ("--cpu-record", a.cpu_record))
                  if path and not os.path.exists(path)],
                 "refusing to link a supplementary file that is not there")
    if a.assets:
        # The links this report promises have to exist: make_results_html.py assembles the
        # asset directory, so it runs first (a dangling link is worse than no link).
        vr.assert_ok(["%s: the asset directory has no %s" % (a.assets, name)
                      for name in ("yaw_pilot_tables.md", "yaw_pilot_summary.json",
                                   "yaw_pilot_gaps.csv")
                      if not os.path.isfile(os.path.join(a.assets, name))],
                     "refusing to link assets that have not been assembled yet (run "
                     "make_results_html.py --assets first)")
    s = json.load(open(a.summary))
    vr.assert_ok(vr.verify_summary_inputs(s, label="the canonical summary (%s)" % a.summary),
                 "refusing to write a results page from a summary that is not bound to the "
                 "runs it came from")
    parity_pairs = vr.parse_pairs(a.parity, "--parity")
    online_pairs = vr.parse_pairs(a.check_online, "--check-online")
    probe_pairs = vr.parse_pairs(a.probe, "--probe")
    vr.assert_ok(vr.check_supplements(s, parity=parity_pairs, online=online_pairs,
                                      probes=probe_pairs),
                 "refusing to render supplemental reports that are not this arm's")
    parity = {arm: (path, json.load(open(path))) for arm, path in parity_pairs}
    online = {arm: (path, json.load(open(path))) for arm, path in online_pairs}
    probes = {arm: (path, json.load(open(path))) for arm, path in probe_pairs}
    L = ["# Results — exp_10 yaw_pilot", "",
         "Descriptive pilot (plan v3.1). Per arm and angle: Δ = paired mean change of the error vs ground truth (α − 0°); G = mean shift of the prediction at α relative to the prediction at 0° (no ground truth); both on the shared comparison mask, %d bootstrap replicates (seeds %s), %.0f %% percentile intervals; query-level CI first, room-cluster CI (17 rooms) second. A multiple G/Δ is printed only where the headline is reportable (Δ interval excludes 0, convergence passed, seed statuses agree); otherwise the cell shows its status. Canonical JSON: `%s` (sha256 `%s`)." % (
             s.get("n_boot", 0), s.get("seeds"), 100 * (1 - s.get("alpha", 0.05)), a.summary, sha(a.summary)), ""]
    L += note_lines(s)
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
        if arm["arm"] in probes:
            ppath, ps = probes[arm["arm"]]
            parm = [x for x in ps["arms"] if x["arm"] == arm["arm"]]
            if parm:
                ctr = parm[0].get("controls", {})
                L += ["Probe run: `%s` (%d queries, execution `%s`, sha256 of its summary `%s…`)." % (parm[0].get("run_dir"), parm[0].get("n_queries", 0), parm[0].get("execution_id"), sha(ppath)[:12]), ""]
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
    rel = a.assets_href or (os.path.relpath(a.assets, os.path.dirname(os.path.abspath(a.out)))
                            if a.assets else None)

    def shown(path):
        """An input inside the asset directory is named by its *published* location (the
        finish script stages the assets under another name and renames them into place)."""
        target = os.path.abspath(path)
        if a.assets and target.startswith(os.path.abspath(a.assets) + os.sep):
            return vr.link_href(path, a.assets, rel, a.out)
        return path
    links = []
    for label, path in ((("full tables (every angle, every metric, exclusions, "
                          "broader-population G)"),
                         os.path.join(a.assets, "yaw_pilot_tables.md") if a.assets else None),
                        ("canonical summary JSON (the source of every number here)",
                         os.path.join(a.assets, "yaw_pilot_summary.json") if a.assets else None),
                        ("figure data (CSV)",
                         os.path.join(a.assets, "yaw_pilot_gaps.csv") if a.assets else None),
                        ("backend sensitivity: GPU (primary) vs CPU protocol", a.backend_table),
                        ("CPU-protocol record", a.cpu_record)):
        if path and os.path.exists(path):
            links.append("- [%s](%s)" % (label, vr.link_href(path, a.assets, rel, a.out)))
    if links:
        L += ["## Supplementary results and data", ""] + links + [""]
    # Finding 11: the full sha256 of every input this report was built from, matching the
    # HTML footer -- truncated hashes cannot be re-verified.
    inputs = [("canonical summary", a.summary)]
    inputs += [("%s probe summary" % arm, path) for arm, path in sorted(probe_pairs)]
    inputs += [("%s parity_exp03" % arm, path) for arm, path in sorted(parity_pairs)]
    inputs += [("%s check_online" % arm, path) for arm, path in sorted(online_pairs)]
    if a.backend_table:
        inputs.append(("backend sensitivity table", a.backend_table))
    if a.cpu_record:
        inputs.append(("CPU-protocol record", a.cpu_record))
    L += ["## Provenance", "",
          "Every input of this report, with its full sha256 (the HTML page's footer lists the "
          "same values):", ""]
    L += ["- %s: `%s` sha256 `%s`" % (label, shown(path), sha(path))
          for label, path in inputs] + [""]
    L += ["Runs the canonical summary was computed from:", ""]
    L += ["- %s: execution `%s`, per_sample sha256 `%s`, meta sha256 `%s`, run dir `%s`" % (
        i["arm"], i["execution_id"], i["per_sample_sha256"], i["meta_sha256"], i["run_dir"])
        for i in s.get("inputs", [])] + [""]
    with open(a.out, "w") as f:
        f.write("\n".join(L))
    print("wrote", a.out)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, vr.ValidationError) as exc:      # a refusal, not a crash
        sys.stderr.write("REFUSED: %s\n" % exc)
        sys.exit(2)
