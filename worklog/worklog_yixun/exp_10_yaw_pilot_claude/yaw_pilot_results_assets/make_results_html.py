#!/usr/bin/env python3
"""Render exp_10's results page from the canonical summariser JSON (numbers appear nowhere else first).

usage: make_results_html.py --summary <yaw_pilot_summary.json> --out <page.html> --assets <dir>
       [--parity <arm>=<parity_exp03.json> ...] [--check-online <arm>=<check_online.json> ...]

Copies the summariser's figures / CSV / JSON next to the page (relative links) and records
every input's sha256 in the page footer. Statuses and wording rules follow plan v3.1 §1 / §12:
a multiple is printed only when the headline is `reportable`; otherwise the cell shows its
status (denominator uncertain / improvement / undefined / unresolved Monte Carlo uncertainty).

Nothing is rendered unbound (Codex tooling review, finding 4): the summary's own `inputs`
hashes are re-verified against the live runs first, every `--parity` / `--check-online` /
`--probe` input has to belong to the arm it is attached to, and the only files copied out of
the summariser's directory are the ones *this* summary names — a figure for another arm, or
any other neighbouring file, is left behind, because hashing arbitrary copied bytes does not
make them this summary's figures.
"""
import argparse
import hashlib
import html
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_runs as vr                                          # noqa: E402

METRICS = [("EDT", "EDT", "s", 1000.0, "ms"), ("C50", "C50", "dB", 1.0, "dB"), ("T60", "T60", "%", 1.0, "pp / %"),
           ("T60_abs", "T60 (absolute)", "s", 1000.0, "ms"), ("logspec_mad", "log-spec MAD (GL-free)", "", 1.0, ""),
           ("mag_rel_l2", "magnitude rel-L2 (GL-free)", "", 1.0, ""), ("wave_rel_l2", "waveform rel-L2", "", 1.0, ""),
           ("wave_mad", "waveform MAD", "", 1.0, ""), ("log_mse", "log-STFT MSE", "", 1.0, ""), ("loss", "test loss", "", 1.0, "")]
ANGLE_DEG = {"0": "0°", "64": "45°", "128": "90°", "256": "180°", "384": "270°"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def num(x, scale=1.0, digits=3):
    if x is None:
        return "–"
    return ("%%.%dg" % digits) % (x * scale)


def ci(d, scale=1.0, digits=3):
    if not d or d.get("point") is None:
        return "–"
    lo, hi = d.get("lo"), d.get("hi")
    if lo is None or hi is None:
        return "%s (no interval)" % num(d["point"], scale, digits)
    return "%s [%s, %s]" % (num(d["point"], scale, digits), num(lo, scale, digits), num(hi, scale, digits))


def esc(s):
    return html.escape(str(s))


def metric_rows(arm, key, scale):
    rows = []
    for k in sorted(arm["angles"], key=int):
        if k == "0":
            continue
        cell = arm["angles"][k].get(key)
        if not cell:
            continue
        q, r, h = cell["query"], cell.get("room", {}), cell.get("headline", {})
        rows.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%d</td><td>%s</td><td>%s</td></tr>" % (
            ANGLE_DEG.get(k, k), num(q.get("mean_0"), scale), num(q.get("mean_alpha"), scale), ci(q.get("delta"), scale),
            ci(r.get("delta"), scale) if r else "–", ci(q.get("gap"), scale), ci(r.get("gap"), scale) if r else "–",
            q.get("n", 0), esc(q.get("ratio", {}).get("status", "–")), ratio_text(h, q, cell)))
    return "\n".join(rows)


def ratio_text(h, q, cell):
    if h.get("reportable"):
        return "<b>%s×</b> [%s, %s]" % (num(h["point"], 1, 3), num(h.get("lower_bound"), 1, 3), num(h.get("upper_bound"), 1, 3))
    reason = h.get("reason") or q.get("ratio", {}).get("reason") or "–"
    rs = cell.get("room_status")
    return "%s%s" % (esc(reason), (" (room: %s)" % esc(rs)) if rs else "")


def arm_section(arm, parity, online, probe=None):
    out = ["<h2 id='%s'>%s</h2>" % (esc(arm["arm"]), esc(arm["arm"]))]
    m = arm["meta"]
    out.append("<p class='meta'>checkpoint <code>%s</code> (sha256 <code>%s…</code>), backbone %s, K = %s, device %s, %d queries, batches %s, manifest <code>%s…</code>, gl_seed %s, execution <code>%s</code>, protocol <code>%s…</code>.</p>" % (
        esc(m.get("checkpoint")), esc(str(m.get("checkpoint_sha256"))[:12]), esc(m.get("backbone")), esc(m.get("num_shot")), esc(m.get("device")),
        arm.get("n_queries", 0), esc(m.get("batches_arg")), esc(str(m.get("manifest_hash"))[:12]), esc(m.get("gl_seed")), esc(arm.get("execution_id")), esc(str(arm.get("protocol_id"))[:12])))
    if arm.get("historical_band_label"):
        out.append("<p class='meta'>Figure band: %s (%s).</p>" % (esc(arm["historical_band_label"]), esc(json.dumps(arm.get("historical_band")))))
    for key, label, unit, scale, shown in METRICS:
        rows = metric_rows(arm, key, scale)
        if not rows:
            continue
        out.append("<h3>%s <span class='unit'>(%s)</span></h3>" % (esc(label), esc(shown or unit or "dimensionless")))
        out.append("<div class='scroll'><table><thead><tr><th>angle</th><th>mean at 0°</th><th>mean at α</th><th>Δ (query CI)</th><th>Δ (room CI)</th><th>G (query CI)</th><th>G (room CI)</th><th>n</th><th>ratio status</th><th>multiple / reason</th></tr></thead><tbody>%s</tbody></table></div>" % rows)
    ctr = arm.get("controls", {})
    if probe:
        parm = [x for x in probe["arms"] if x["arm"] == arm["arm"]]
        if parm:
            ctr = parm[0].get("controls", {})
            out.append("<p class='meta'>Controls from the probe run <code>%s</code> (%d queries, execution <code>%s</code>).</p>" % (esc(parm[0].get("run_dir")), parm[0].get("n_queries", 0), esc(parm[0].get("execution_id"))))
    if ctr:
        rows = "".join("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            esc(n), esc(c.get("k")), num(c.get("wave_max_abs_diff"), 1, 2), num(c.get("logspec_max_abs_diff"), 1, 2), "ok" if c.get("ok") else "<b>FAIL</b>")
            for n, c in sorted(ctr.get("controls", {}).items()))
        out.append("<h3>Controls (probe)</h3><table><thead><tr><th>control</th><th>k</th><th>max |Δ wave|</th><th>max |Δ log-spec|</th><th>ok</th></tr></thead><tbody>%s</tbody></table><p class='meta'>controls table ok = %s</p>" % (rows, ctr.get("ok")))
    if parity:
        cells = []
        for k in sorted(parity.get("angles", {}), key=int):
            for mkey in ("edt_err", "c50_err", "t60_err", "logspec_mad"):
                c = parity["angles"][k].get(mkey)
                if c:
                    cells.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
                        ANGLE_DEG.get(k, k), mkey, num(c.get("max_abs_diff"), 1, 2), "%.4g" % c.get("fraction_within_tolerance", float("nan")),
                        num(c.get("paired_delta_run"), 1, 4), num(c.get("paired_delta_exp03"), 1, 4), esc(c.get("status"))))
        out.append("<h3>Parity vs exp_03 (%s)</h3><p class='meta'>ok = %s; rows %s; missing required %s; exp_03 file sha256 <code>%s…</code></p><div class='scroll'><table><thead><tr><th>angle</th><th>metric</th><th>max |Δ|</th><th>within tol.</th><th>paired Δ (run)</th><th>paired Δ (exp_03)</th><th>status</th></tr></thead><tbody>%s</tbody></table></div>" % (
            esc(os.path.basename(os.path.dirname(parity.get("exp03_path", "")))), parity.get("ok"), parity.get("n_rows"), esc(parity.get("missing_required")), esc(str(parity.get("exp03_sha256"))[:12]), "".join(cells)))
    if online is not None:
        out.append("<p class='meta'>check_online ok = %s</p>" % online.get("ok"))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--parity", nargs="*", default=[])
    ap.add_argument("--check-online", nargs="*", default=[])
    ap.add_argument("--probe", nargs="*", default=[], help="<arm>=<probe summary JSON>: controls are rendered from the probe run")
    ap.add_argument("--title", default="exp_10 yaw_pilot — results")
    args = ap.parse_args()
    s = json.load(open(args.summary))
    vr.assert_ok(vr.verify_summary_inputs(s, label="the canonical summary (%s)" % args.summary),
                 "refusing to render a summary that is not bound to the runs it came from")
    parity_pairs = vr.parse_pairs(args.parity, "--parity")
    online_pairs = vr.parse_pairs(args.check_online, "--check-online")
    probe_pairs = vr.parse_pairs(args.probe, "--probe")
    vr.assert_ok(vr.check_supplements(s, parity=parity_pairs, online=online_pairs,
                                      probes=probe_pairs),
                 "refusing to render supplemental reports that are not this arm's")
    os.makedirs(args.assets, exist_ok=True)
    sdir = os.path.dirname(os.path.abspath(args.summary))
    # Only what this summary names: its own JSON, its tables, its CSV and its figures.
    figures = vr.figure_names(s)
    wanted = ["yaw_pilot_summary.json", "yaw_pilot_tables.md", "yaw_pilot_gaps.csv"] + figures
    required = [n for n in wanted if not n.endswith(".pdf")]
    vr.assert_ok(["%s: the summariser's directory has no %s" % (sdir, n)
                  for n in required if not os.path.isfile(os.path.join(sdir, n))],
                 "refusing to render: this summary's own outputs are incomplete")
    copied = {}
    for fn in wanted:
        src = os.path.join(sdir, fn)
        if not os.path.isfile(src):
            continue                       # the .pdf companions are optional
        shutil.copy2(src, os.path.join(args.assets, fn))
        copied[fn] = sha(os.path.join(args.assets, fn))
    left = [fn for fn in sorted(os.listdir(sdir))
            if fn not in wanted and os.path.isfile(os.path.join(sdir, fn))]
    if left:
        print("not copied (not named by this summary):", ", ".join(left))
    parity = {arm: json.load(open(path)) for arm, path in parity_pairs}
    online = {arm: json.load(open(path)) for arm, path in online_pairs}
    probes = {arm: json.load(open(path)) for arm, path in probe_pairs}
    input_shas = {os.path.abspath(args.summary): sha(args.summary)}
    for _arm, p in parity_pairs + online_pairs + probe_pairs:
        input_shas[os.path.abspath(p)] = sha(p)
    rel = os.path.relpath(args.assets, os.path.dirname(os.path.abspath(args.out)))
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>%s</title><style>body{font-family:system-ui,sans-serif;max-width:1400px;margin:2em auto;padding:0 1em;color:#222}table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #ccc;padding:3px 7px;text-align:right}th{background:#f3f3f3}td:first-child,th:first-child{text-align:left}.meta{color:#555;font-size:13px}.unit{color:#777;font-weight:normal}.scroll{overflow-x:auto}.note{background:#fff8e1;border-left:4px solid #e69f00;padding:.6em 1em;margin:1em 0}img{max-width:100%%}code{font-size:12px}</style></head><body>" % esc(args.title)]
    parts.append("<h1>%s</h1>" % esc(args.title))
    parts.append("<p class='meta'>Descriptive pilot (plan v3.1): per arm and angle, Δ = paired mean change of the error vs ground truth (α minus 0°), G = mean shift of the prediction at α relative to the prediction at 0° (no ground truth), on the shared comparison mask; %d bootstrap replicates, seeds %s, %.0f %% intervals; query-level and room-cluster (17 rooms) intervals shown separately. Generated %s by %s.</p>" % (
        s.get("n_boot", 0), esc(s.get("seeds")), 100 * (1 - s.get("alpha", 0.05)), esc(s.get("generated_at")), esc(s.get("tool"))))
    for n in s.get("notes", []):
        parts.append("<div class='note'>%s</div>" % esc(n))
    mq = s.get("metric_qualifications", {})
    if mq:
        parts.append("<ul class='meta'>%s</ul>" % "".join("<li><b>%s</b>: %s</li>" % (esc(k), esc(v)) for k, v in sorted(mq.items())))
    if "yaw_pilot_gaps_all_arms.png" in copied:
        parts.append("<h2>All arms</h2><img src='%s/yaw_pilot_gaps_all_arms.png' alt='all arms'>" % esc(rel))
    for arm in s["arms"]:
        parts.append(arm_section(arm, parity.get(arm["arm"]), online.get(arm["arm"]), probes.get(arm["arm"])))
        fig = "yaw_pilot_gaps_%s.png" % arm["arm"]
        if fig in copied:
            parts.append("<img src='%s/%s' alt='%s'>" % (esc(rel), esc(fig), esc(arm["arm"])))
    parts.append("<h2>Provenance</h2><ul class='meta'>%s</ul><ul class='meta'>%s</ul>" % (
        "".join("<li><code>%s</code> sha256 <code>%s</code></li>" % (esc(p), esc(h)) for p, h in sorted(input_shas.items())),
        "".join("<li>asset <code>%s</code> sha256 <code>%s</code></li>" % (esc(p), esc(h)) for p, h in sorted(copied.items()))))
    parts.append("<ul class='meta'>%s</ul>" % "".join("<li>%s: execution <code>%s</code>, per_sample sha256 <code>%s</code>, meta sha256 <code>%s</code></li>" % (
        esc(i["arm"]), esc(i["execution_id"]), esc(i["per_sample_sha256"]), esc(i["meta_sha256"])) for i in s.get("inputs", [])))
    parts.append("</body></html>")
    with open(args.out, "w") as f:
        f.write("\n".join(parts))
    print("wrote", args.out, "assets", len(copied))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, vr.ValidationError) as exc:      # a refusal, not a crash
        sys.stderr.write("REFUSED: %s\n" % exc)
        sys.exit(2)
