"""Build yaw_rotation_degradation_01_results.html (offline, inline SVG) from the canonical summarizer JSON.

Every displayed number is read from the JSON written by `tools/summarize_yaw.py --json ... --summary ...`; the page
computes no statistics of its own and carries no locally inferred significance markers (the distance-to-threshold
diagnostics come from the producer field h1.bounds; interval levels from config.interval_levels). A page is FINAL only if the
JSON has mode == "full", valid_for_confirmatory is exactly True and exploratory is not True, and the summary file named
in the JSON matches the recorded sha256; anything else renders as a persistently watermarked DRAFT with every verdict,
equivalence result and H1/H2 cell marker suppressed.

    python worklog/.../yaw_rotation_degradation_results_assets/make_results_html.py --stats ckpt/yaw_rotation/full_stats.json
"""
import argparse
import hashlib
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
OUT_HTML = os.path.join(HERE, "..", "yaw_rotation_degradation_01_results.html")
# presentation only: colour slot per run label; the display names come from config.run_descriptions
SERIES = {"cyl": "s1", "control": "s2", "released": "s3"}
SERIES_NAME = {}   # filled from config.run_descriptions
PRECISION = {"edt": 4, "c50": 3, "t60": 2, "loss": 5, "log_mse": 4, "consistency": 4}          # display precision only
K0_PRECISION = {"edt": 5, "c50": 4, "t60": 3, "log_mse": 5, "loss": 6}
ACOUSTIC = []   # (key, name, unit, nd) filled from config.metric_names / metric_units
SPECTRAL = []
LV = {"q": "?", "r": "?", "k0": "?"}


def esc(s):
    return html.escape(str(s), quote=True)


def fmt(v, nd, unit=""):
    """Null-safe fixed-point formatter (JSON null -> en dash)."""
    return "–" if v is None else f"{v:.{nd}f}{(' ' + unit) if unit else ''}"


def pct(v, nd=1):
    return "–" if v is None else f"{100 * v:+.{nd}f} %"


def ci(lo, hi, nd=1):
    return "–" if lo is None or hi is None else f"[{pct(lo, nd)}, {pct(hi, nd)}]"


def is_final(st):
    return st.get("mode") == "full" and st.get("valid_for_confirmatory") is True and st.get("exploratory") is not True


def schema(st):
    """Producer-authored names, units, glosses and counts; refuses a JSON without them."""
    cfg = st["config"]
    need = ["labels", "roles", "metric_names", "metric_units", "condition_definitions", "angle_counts", "run_descriptions", "interval_levels", "r_definition", "bootstrap_description"]
    missing = [k for k in need if k not in cfg]
    top = [k for k in ("k0_note", "delay_flips_entity", "decomposition_stages", "decomposition_note") if k not in st]
    rules = [k for k in ("rule", "wording_rule") if k not in st.get("h1", {})] + [k for k in ("rule", "equivalence_rule") if k not in st.get("h2", {})] + ([] if "rule" in (st.get("convergence") or {}) else ["convergence.rule"])
    if missing or top or rules:
        raise SystemExit(f"producer fields missing (config {missing}, top-level {top}, rules {rules}): regenerate the JSON with the current tools/summarize_yaw.py")
    for l in cfg["labels"]:
        if l not in cfg["run_descriptions"]:
            raise SystemExit(f"run_descriptions lacks {l}")
    for m in ("edt", "c50", "t60", "loss", "log_mse", "consistency"):
        if m not in cfg["metric_names"] or m not in cfg["metric_units"]:
            raise SystemExit(f"metric_names/metric_units lack {m}")
    for k in ("P", "E"):
        if k not in cfg["condition_definitions"]:
            raise SystemExit(f"condition_definitions lacks {k}")
    for k in ("primary", "cyl"):
        if k not in cfg["roles"] or cfg["roles"][k] not in cfg["labels"]:
            raise SystemExit(f"roles lacks {k} or names an unknown label")
    SERIES_NAME.update(cfg["run_descriptions"])
    ACOUSTIC[:] = [(m, cfg["metric_names"][m], cfg["metric_units"][m], PRECISION[m]) for m in ("edt", "c50", "t60")]
    SPECTRAL[:] = [(m, cfg["metric_names"][m], cfg["metric_units"][m], PRECISION[m]) for m in ("loss", "log_mse")]
    return cfg




def check_coverage(st, cfg, final):
    """Exact nested coverage of the producer results; any missing structure is refused (no empty tables, no silent skips)."""
    labels = list(cfg["labels"]); errs = []
    for l in labels:
        for cond in ("P", "E"):
            for m in ("edt", "c50", "t60"):
                rows = ((st.get("acoustic") or {}).get(l) or {}).get(cond, {}).get(m)
                if not isinstance(rows, list) or not rows:
                    errs.append(f"acoustic[{l}][{cond}][{m}]")
            for m in ("loss", "log_mse", "consistency"):
                rows = ((st.get("spectral") or {}).get(l) or {}).get(cond, {}).get(m)
                if not isinstance(rows, list) or not rows:
                    errs.append(f"spectral[{l}][{cond}][{m}]")
        flips = (st.get("delay_flips") or {}).get(l)
        if not isinstance(flips, dict) or sorted(int(k) for k in flips) != sorted(int(k) for k in cfg["preregistered_spectral_cols"]):
            errs.append(f"delay_flips[{l}]")
        if l not in (st.get("meta") or {}):
            errs.append(f"meta[{l}]")
    if not isinstance(st.get("k0"), list) or not st["k0"]:
        errs.append("k0")
    for m in cfg["confirmatory_metrics"]:
        rows = ((st.get("h2") or {}).get("rows") or {}).get(m)
        if not isinstance(rows, list) or not rows:
            errs.append(f"h2.rows[{m}]")
        for r in rows or []:
            eq = r.get("equivalence")
            if eq is not None and ("query" not in eq or "cluster" not in eq):
                errs.append(f"h2.rows[{m}] k={r.get('k')}: equivalence lacks query/cluster")
    if cfg["roles"]["cyl"] not in (st.get("decomposition") or {}):
        errs.append("decomposition[cyl]")
    if final:
        for l in labels:
            for m in cfg["confirmatory_metrics"]:
                if m not in (((st.get("h1") or {}).get("bounds") or {}).get(l) or {}):
                    errs.append(f"h1.bounds[{l}][{m}]")
        if not isinstance((st.get("h2") or {}).get("verdict"), dict):
            errs.append("h2.verdict")
    if errs:
        raise SystemExit("producer results incomplete: " + ", ".join(errs))


def levels(st):
    """Interval levels from the producer: degradation/D_k query and room intervals (adjusted), k=0 (nominal)."""
    lv = st["config"].get("interval_levels")
    if not lv:
        raise SystemExit("config.interval_levels missing: regenerate the JSON with the current tools/summarize_yaw.py")
    f = lambda x: f"{100 * x:.2f} %".replace(".00 %", " %")
    return {"q": f(lv["degradation_query"]), "r": f(lv["degradation_room"]), "k0": f(lv["k0"])}


def verify_summary_binding(st):
    sp, want = st.get("summary_path"), st.get("summary_sha256")
    if not sp or not want:
        raise SystemExit("final JSON must name summary_path and summary_sha256")
    with open(sp, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != want:
        raise SystemExit(f"summary hash mismatch: {sp} has {got[:12]}…, JSON records {want[:12]}…")


def deg_chart(title, series, threshold=None, threshold_label="", width=480, height=260, xlabel="yaw of the whole scene, degrees"):
    """series: list of (label, cls, rows); rows carry deg, r, lo, hi and optionally r_lo, r_hi (any may be null)."""
    pts = [r for _, _, rows in series for r in rows if r.get("r") is not None]
    if not pts:
        return f'<p class="sub">{esc(title)}: no data</p>'
    vals = [100 * v for r in pts for v in (r["r"], r.get("lo"), r.get("hi"), r.get("r_lo"), r.get("r_hi")) if v is not None]
    if threshold is not None:
        vals.append(100 * threshold)
    lo, hi = min(vals + [0.0]), max(vals + [0.0]); span = hi - lo; pad = span * 0.12 if span > 0 else 1.0; lo -= pad; hi += pad
    L, R, T, B = 58, 16, 30, 40; pw, ph = width - L - R, height - T - B
    x = lambda d: L + (d + 180) / 360 * pw; y = lambda v: T + (hi - v) / (hi - lo) * ph
    s = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}"><title>{esc(title)}</title>', f'<text class="ttl" x="{L}" y="18">{esc(title)}</text>']
    step = max(1, round((hi - lo) / 5)); t = int(lo // step) * step
    while t <= hi:
        s.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/><text class="tick" x="{L - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{t:+d} %</text>'); t += step
    for d in [-180, -90, -45, 0, 45, 90, 180]:
        s.append(f'<text class="tick" x="{x(d):.1f}" y="{height - 22}" text-anchor="middle">{d}°</text>')
    s.append(f'<text class="tick" x="{L + pw / 2:.1f}" y="{height - 6}" text-anchor="middle">{esc(xlabel)}</text>')
    s.append(f'<line class="axis" x1="{L}" x2="{L + pw}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    if threshold is not None:
        s.append(f'<line class="ref" x1="{L}" x2="{L + pw}" y1="{y(100 * threshold):.1f}" y2="{y(100 * threshold):.1f}"/><text class="reflbl" x="{L + pw}" y="{y(100 * threshold) - 4:.1f}" text-anchor="end">{esc(threshold_label)}</text>')
    for label, cls, rows in series:
        rows = sorted([r for r in rows if r.get("r") is not None], key=lambda r: r["deg"])
        pts_s = " ".join("%.1f,%.1f" % (x(r["deg"]), y(100 * r["r"])) for r in rows)
        s.append(f'<polyline class="line {cls}" points="{pts_s}"/>')
        for r in rows:
            if r.get("r_lo") is not None and r.get("r_hi") is not None:
                s.append(f'<line class="ci2" x1="{x(r["deg"]):.1f}" x2="{x(r["deg"]):.1f}" y1="{y(100 * r["r_lo"]):.1f}" y2="{y(100 * r["r_hi"]):.1f}"/>')
            if r.get("lo") is not None and r.get("hi") is not None:
                s.append(f'<line class="ci {cls}" x1="{x(r["deg"]):.1f}" x2="{x(r["deg"]):.1f}" y1="{y(100 * r["lo"]):.1f}" y2="{y(100 * r["hi"]):.1f}"/>')
            tip = "%s · %+.1f°: r = %s, adjusted query-level interval (%s) %s, adjusted room-cluster interval (%s) %s, n = %s" % (
                label, r["deg"], pct(r["r"], 2), LV["q"], ci(r.get("lo"), r.get("hi"), 2), LV["r"], ci(r.get("r_lo"), r.get("r_hi"), 2), r.get("n_valid", "–"))
            s.append(f'<circle class="mk {cls}" tabindex="0" cx="{x(r["deg"]):.1f}" cy="{y(100 * r["r"]):.1f}" r="4" data-tip="{esc(tip)}"><title>{esc(tip)}</title></circle>')
    s.append("</svg>")
    return "\n".join(s)


def abs_chart(title, series, nd, unit="", width=480, height=240):
    """Absolute per-angle means (rows carry deg, meank; null allowed)."""
    vals = [r["meank"] for _, _, rows in series for r in rows if r.get("meank") is not None]
    if not vals:
        return ""
    lo, top = 0.0, max(vals); hi = top * 1.15 if top > 0 else 1.0
    L, R, T, B = 62, 16, 30, 40; pw, ph = width - L - R, height - T - B
    x = lambda d: L + (d + 180) / 360 * pw; y = lambda v: T + (hi - v) / (hi - lo) * ph
    s = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}"><title>{esc(title)}</title>', f'<text class="ttl" x="{L}" y="18">{esc(title)}</text>']
    for i in range(5):
        t = hi * i / 4; s.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/><text class="tick" x="{L - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{t:.{nd}f}</text>')
    for d in [-180, -90, -45, 0, 45, 90, 180]:
        s.append(f'<text class="tick" x="{x(d):.1f}" y="{height - 22}" text-anchor="middle">{d}°</text>')
    s.append(f'<text class="tick" x="{L + pw / 2:.1f}" y="{height - 6}" text-anchor="middle">yaw of the whole scene, degrees{(" · y in " + unit) if unit else ""}</text>')
    s.append(f'<line class="axis" x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}"/>')
    for label, cls, rows in series:
        rows = sorted([r for r in rows if r.get("meank") is not None], key=lambda r: r["deg"])
        s.append(f'<polyline class="line {cls}" points="{" ".join("%.1f,%.1f" % (x(r["deg"]), y(r["meank"])) for r in rows)}"/>')
        for r in rows:
            tip = "%s · %+.1f°: %s" % (label, r["deg"], fmt(r["meank"], nd, unit))
            s.append(f'<circle class="mk {cls}" tabindex="0" cx="{x(r["deg"]):.1f}" cy="{y(r["meank"]):.1f}" r="3.5" data-tip="{esc(tip)}"><title>{esc(tip)}</title></circle>')
    s.append("</svg>")
    return "\n".join(s)


def sci(v):
    return "–" if v is None else "%.2e" % v


def iv(lo, hi, nd=5):
    return "–" if lo is None or hi is None else "[%+.*f, %+.*f]" % (nd, lo, nd, hi)


def validity_cells(v):
    v = v or {}
    return f"<td>{v.get('paired_valid', '–')}</td><td>{v.get('newly_invalid', '–')}</td><td>{pct(v.get('newly_invalid_frac'), 2) if v.get('newly_invalid_frac') is not None else '–'}</td>"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stats", required=True); ap.add_argument("--out", default=None, help="output HTML path (default: the record's results page)"); args = ap.parse_args()
    os.chdir(REPO); out_html = os.path.abspath(args.out) if args.out else OUT_HTML
    with open(args.stats) as f:
        st = json.load(f)
    final = is_final(st); draft = not final
    if final:
        verify_summary_binding(st)
    cfg = schema(st); labels = list(cfg["labels"])
    missing_runs = [l for l in labels if l not in st.get("acoustic", {}) or l not in st.get("spectral", {}) or l not in st.get("meta", {})]
    if missing_runs:
        raise SystemExit(f"labels without acoustic/spectral/meta sections: {missing_runs}")
    if any(l not in SERIES for l in labels):
        raise SystemExit(f"no colour slot for labels {[l for l in labels if l not in SERIES]}")
    check_coverage(st, cfg, final)
    MN, MU = cfg["metric_names"], cfg["metric_units"]; roles = cfg["roles"]
    mname = lambda m: MN[m] + (f" ({MU[m]})" if MU[m] else "")
    primary, cyl_label = roles["primary"], roles["cyl"]
    n_spec = cfg["angle_counts"]["spectral"]; n_ac = cfg["angle_counts"]["acoustic"]; cond = cfg["condition_definitions"]
    lv = levels(st); LV.update(lv); thr = cfg["threshold"]; margin = cfg["equiv_margin"]
    thr_label = f'pre-registered "substantial" threshold {pct(thr, 0)}'
    # producer-authored H1-passing cells (H2 is evaluated on exactly these); nothing is inferred locally
    h1, h2 = st.get("h1", {}), st.get("h2", {"verdict": {}, "rows": {}})
    h1_cells = set() if draft else {(c["metric"], int(c["k"])) for c in (h2.get("verdict", {}).get("cells") or []) if isinstance(c, dict)}
    h2_pass = set() if draft else {(c["metric"], int(c["k"])) for c in (h2.get("verdict", {}).get("passing") or []) if isinstance(c, dict)}
    # charts
    charts, spec_charts = [], []
    for key, name, unit, nd in ACOUSTIC:
        series = [(SERIES_NAME[l], SERIES[l], st["acoustic"][l]["P"].get(key, [])) for l in labels]
        confirm = key in cfg.get("confirmatory_metrics", [])
        charts.append(deg_chart(f"{name}: relative change vs 0° (condition P, adjusted intervals)", series, threshold=thr if confirm else None, threshold_label=thr_label))
        charts.append(abs_chart(f"{name} ({unit}): mean per angle, condition P", series, nd, unit))
    for key, name, unit, nd in SPECTRAL:
        spec_charts.append(deg_chart(f"{name}: relative change vs 0° (condition P, {n_spec} angles)", [(SERIES_NAME[l], SERIES[l], st["spectral"][l]["P"].get(key, [])) for l in labels]))
    spec_charts.append(abs_chart(f"{cfg['metric_names']['consistency']}, condition P", [(SERIES_NAME[l], SERIES[l], st["spectral"][l]["P"].get("consistency", [])) for l in labels], 4))

    # tables
    def acoustic_table(l, cond):
        rows = []
        for key, name, unit, nd in ACOUSTIC:
            for r in sorted(st["acoustic"][l].get(cond, {}).get(key, []), key=lambda r: r["deg"]):
                tag = ""
                if cond == "P" and l == primary and (key, int(r["k"])) in h1_cells:
                    tag = ' <span class="tag">H1 cell</span>'
                rows.append(f"<tr><td>{name} ({unit})</td><td>{r['deg']:+.1f}°</td><td>{fmt(r.get('mean0'), nd)}</td><td>{fmt(r.get('meank'), nd)}</td><td>{pct(r.get('r'))}{tag}</td><td>{ci(r.get('lo'), r.get('hi'))}</td><td>{ci(r.get('r_lo'), r.get('r_hi'))}</td><td>{r.get('n_valid', '–')}</td>{validity_cells(r.get('validity'))}</tr>")
        if not rows:
            return ""
        return f'<h3>{esc(SERIES_NAME[l])} — acoustic metrics, condition {cond}</h3><div class="wrap"><table><tr><th>metric</th><th>yaw</th><th>mean at 0°</th><th>mean at k</th><th>r</th><th>adjusted query-level interval ({lv["q"]})</th><th>adjusted room-cluster interval ({lv["r"]})</th><th>n</th><th>paired valid</th><th>newly invalid</th><th>newly invalid, %</th></tr>{"".join(rows)}</table></div>'

    def spectral_table(l, cond):
        rows = []
        for key, name, unit, nd in SPECTRAL + [("consistency", cfg["metric_names"]["consistency"], cfg["metric_units"]["consistency"], 4)]:
            for r in sorted(st["spectral"][l].get(cond, {}).get(key, []), key=lambda r: r["deg"]):
                rows.append(f"<tr><td>{name}</td><td>{r['deg']:+.1f}°</td><td>{fmt(r.get('mean0'), nd)}</td><td>{fmt(r.get('meank'), nd)}</td><td>{pct(r.get('r'))}</td><td>{ci(r.get('lo'), r.get('hi'))}</td><td>{r.get('n_valid', '–')}</td>{validity_cells(r.get('validity'))}</tr>")
        if not rows:
            return ""
        return f'<h3>{esc(SERIES_NAME[l])} — spectral metrics, condition {cond}</h3><div class="wrap"><table><tr><th>metric</th><th>yaw</th><th>mean at 0°</th><th>mean at k</th><th>r</th><th>adjusted query-level interval ({lv["q"]})</th><th>n</th><th>paired valid</th><th>newly invalid</th><th>newly invalid, %</th></tr>{"".join(rows)}</table></div>'

    ac_tables = "".join(acoustic_table(l, c) for c in ("P", "E") for l in labels)
    sp_tables = "".join(spectral_table(l, c) for c in ("P", "E") for l in labels)
    def k0_row(r):
        nd = K0_PRECISION.get(r["metric"], 5); name = mname(r["metric"])
        return "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            esc(name), fmt(r.get("cyl"), nd), fmt(r.get("ctrl"), nd), fmt(r.get("diff"), nd),
            "" if r.get("rel_pct") is None else " (%+.1f %%)" % r["rel_pct"], iv(r.get("q_lo"), r.get("q_hi"), nd), iv(r.get("r_lo"), r.get("r_hi"), nd), r.get("n_valid", "–"))
    k0 = "".join(k0_row(r) for r in st.get("k0", []))
    h2rows = []
    for m in h2.get("rows", {}):
        for r in sorted(h2["rows"][m], key=lambda r: r["deg"]):
            eq = r.get("equivalence") or {}
            if eq and ("query" not in eq or "cluster" not in eq or "equivalent" not in eq["query"] or "equivalent" not in eq["cluster"]):
                raise SystemExit(f"h2 row k={r.get('k')}: equivalence lacks a query or cluster result")
            w = lambda d: "equivalent" if d["equivalent"] else "not established"
            eq_txt = "–" if draft or not eq else f"{w(eq['query'])} / {w(eq['cluster'])}"
            cell = (m, int(r["k"]))
            tag = "" if draft else (' <span class="tag">H1 cell · H2 pass</span>' if cell in h2_pass else (' <span class="tag">H1 cell</span>' if cell in h1_cells else ""))
            h2rows.append(f"<tr><td>{esc(mname(m))}</td><td>{r['deg']:+.1f}°</td><td>{pct(r.get('r_cyl'))}</td><td>{pct(r.get('r_ctrl'))}</td><td>{pct(r.get('d'))}{tag}</td><td>{ci(r.get('lo'), r.get('hi'))}</td><td>{ci(r.get('c_lo'), r.get('c_hi'))}</td><td>{eq_txt}</td></tr>")
    flips_cols = cfg.get("preregistered_spectral_cols", [])
    flips = "".join(f"<tr><td>{esc(SERIES_NAME[l])}</td>" + "".join(f"<td>{st['delay_flips'][l].get(str(k), '–')}</td>" for k in flips_cols) + "</tr>" for l in st.get("delay_flips", {}))
    dec = st.get("decomposition") or {}
    stages = st["decomposition_stages"]
    dec_txt = "".join("<li>%s: at k = %s over %s batch(es): %s</li>" % (
        esc(SERIES_NAME[l]), d.get("k", "–"), d.get("n_batches", "–"),
        ", ".join(f"{esc(stages[k])} {sci(d.get(k))}" for k in ("tokens_rel_change", "pooled_rel_change", "coord_rel_change", "logspec_rel_change"))) for l, d in dec.items())
    conv = st.get("convergence") or {}
    margins = "; ".join(f"{esc(SERIES_NAME[k])}: {esc(MN['edt'])} {fmt((v or {}).get('edt'), 4, MU['edt'])}, {esc(MN['c50'])} {fmt((v or {}).get('c50'), 3, MU['c50'])}" for k, v in (h1.get("absolute_margins") or {}).items())
    verdict_block = "" if draft else f'''<div class="verdicts"><div class="v {"good" if str(h1.get("verdict", "")).startswith("supported") else "warn"}"><b>H1 — {esc(h1.get("verdict", "–"))}</b><span>{esc(h1["rule"])} Wording: {esc(h1["wording_rule"])} Roles evaluated: {esc(", ".join(h1.get("roles_evaluated", [])) or "–")}; {pct(thr, 0)} absolute margins at 0°: {margins or "–"}</span></div><div class="v {"good" if str(h2.get("verdict", {}).get("aggregate", "")).startswith("supported") else "neutral"}"><b>H2 — {esc(h2.get("verdict", {}).get("aggregate", "–"))}</b><span>{esc(h2["rule"])} {h2.get("verdict", {}).get("n_passing", "–")} of {h2.get("verdict", {}).get("n_cells", "–")} H1-passing cells.</span></div><div class="v {"good" if conv.get("pass") else "warn"}"><b>Bootstrap convergence {"pass" if conv.get("pass") else "FAIL"}</b><span>{esc(conv["rule"])} Max endpoint movement {fmt(conv.get("max_rel_change"), 3)} of the interval width (limit {conv.get("limit", "–")}), seeds {esc(conv.get("seeds", "–"))}</span></div></div>'''
    bounds = (h1.get("bounds") or {}) if not draft else {}
    b_rows = []
    for l in labels:
        for m in cfg.get("confirmatory_metrics", []):
            b = (bounds.get(l) or {}).get(m)
            if not b:
                continue
            mq, mr = b.get("max_query_upper") or {}, b.get("max_room_upper") or {}
            above_r = ", ".join(f"{c['deg']:+.1f}° ({pct(c['r_hi'])})" for c in b.get("room_upper_above_threshold", [])) or "none"
            above_q = ", ".join(f"{c['deg']:+.1f}° ({pct(c['hi'])})" for c in b.get("query_upper_above_threshold", [])) or "none"
            b_rows.append(f"<tr><td>{esc(SERIES_NAME[l])}</td><td>{esc(mname(m))}</td><td>{pct(mq.get('hi'))} at {mq.get('deg', '–')}°</td><td>{above_q}</td><td>{pct(mr.get('r_hi'))} at {mr.get('deg', '–')}°</td><td>{above_r}</td></tr>")
    bounds_table = "" if not b_rows else f'<h3>Distance to the +{100 * thr:.0f} % threshold (producer field h1.bounds, condition P, non-zero angles)</h3><div class="wrap"><table><tr><th>model</th><th>metric</th><th>largest adjusted query-level upper bound</th><th>query-level upper bounds above threshold</th><th>largest adjusted room-cluster upper bound</th><th>room-cluster upper bounds above threshold</th></tr>{"".join(b_rows)}</table></div>'
    banner = '<div class="draft" role="alert">DRAFT — exploratory or not valid for confirmatory claims. Verdicts, equivalence results and H1/H2 cell markers are suppressed.</div>' if draft else ""
    watermark = '<div class="wm" aria-hidden="true">' + "".join('<span>DRAFT</span>' for _ in range(12)) + '</div>' if draft else ""
    legend = "".join(f'<span><i class="{SERIES[l]}"></i>{esc(SERIES_NAME[l])}</span>' for l in labels)
    meta_lines = "; ".join(f"{esc(SERIES_NAME[l])}: {esc(os.path.basename(str(st['meta'][l].get('checkpoint', '–'))))}, {st['meta'][l].get('n_samples', '–')} queries, batch {st['meta'][l].get('batch_size', '–')}, TF32 {st['meta'][l].get('tf32', '–')}, GL seed {st['meta'][l].get('gl_seed', '–')}" for l in labels if l in st.get("meta", {}))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>yaw_rotation_degradation — results</title>
<style>
:root{{color-scheme:light;--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#006300;--warn:#9a5a00;--card:#f4f3ef}}
@media (prefers-color-scheme:dark){{:root{{color-scheme:dark;--surface:#1a1a19;--text:#fff;--text2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#0ca30c;--warn:#e0a030;--card:#232321}}}}
body{{margin:0;background:var(--surface);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}} main{{max-width:1040px;margin:0 auto;padding:24px 20px 60px;position:relative;z-index:1}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px}} h3{{font-size:14px;margin:16px 0 4px;color:var(--text2)}} .sub{{color:var(--text2);margin:0 0 12px}}
.draft{{background:#b00020;color:#fff;font-weight:700;padding:10px 14px;border-radius:6px;margin:10px 0;font-size:15px}}
.wm{{position:fixed;inset:0;pointer-events:none;z-index:0;display:grid;grid-template-columns:repeat(3,1fr);align-content:space-around;justify-items:center}} .wm span{{font-size:80px;font-weight:800;color:rgba(176,0,32,.10);transform:rotate(-30deg)}}
.verdicts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin:12px 0}} .v{{border-left:4px solid var(--muted);background:var(--card);padding:10px 12px;border-radius:6px}} .v.good{{border-color:var(--good)}} .v.warn{{border-color:var(--warn)}} .v b{{display:block;font-size:15px}} .v span{{color:var(--text2)}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:12px}} .chart{{width:100%;height:auto}}
.ttl{{fill:var(--text);font-size:13px;font-weight:600}} .tick{{fill:var(--muted);font-size:11px}} .grid{{stroke:var(--grid)}} .axis{{stroke:var(--axis);stroke-width:1.5}}
.line{{fill:none;stroke-width:2}} .line.s1{{stroke:var(--s1)}} .line.s2{{stroke:var(--s2)}} .line.s3{{stroke:var(--s3)}}
.mk{{stroke:var(--surface);stroke-width:1.5}} .mk:focus{{outline:none;stroke:var(--text);stroke-width:2.5}} .mk.s1{{fill:var(--s1)}} .mk.s2{{fill:var(--s2)}} .mk.s3{{fill:var(--s3)}}
.ci{{stroke-width:2;opacity:.7}} .ci.s1{{stroke:var(--s1)}} .ci.s2{{stroke:var(--s2)}} .ci.s3{{stroke:var(--s3)}} .ci2{{stroke:var(--muted);stroke-width:1;opacity:.8}}
.ref{{stroke:var(--warn);stroke-width:1.5;stroke-dasharray:5 4}} .reflbl{{fill:var(--warn);font-size:11px}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;color:var(--text2);margin:6px 0}} .legend i{{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:6px}} .legend i.s1{{background:var(--s1)}} .legend i.s2{{background:var(--s2)}} .legend i.s3{{background:var(--s3)}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;margin:4px 0 8px}} th,td{{padding:4px 7px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{color:var(--text2);font-weight:600}} .wrap{{overflow-x:auto}}
.tag{{font-size:10.5px;color:var(--text2);border:1px solid var(--axis);border-radius:3px;padding:0 4px;margin-left:4px}}
#tip{{position:fixed;pointer-events:none;background:var(--text);color:var(--surface);padding:4px 8px;border-radius:4px;font-size:12px;display:none;z-index:9;max-width:460px}} .prov{{color:var(--text2)}}
</style></head><body>{watermark}<main>
<h1>Does joint yaw rotation degrade xRIR's RIR synthesis?{" — DRAFT" if draft else ""}</h1>{banner}
<p class="sub">exp_03 · AcousticRooms unseen test split, {st.get("n_queries", "–")} queries in {st.get("n_rooms", "–")} rooms, references pinned by manifest {esc(str(st.get("manifest_hash", ""))[:12])}… · the whole scene (receiver-centred depth panorama and source/reference coordinates) is rotated in yaw by an integer number of panorama columns (angles shown in degrees); condition P = {esc(cond["P"])}, condition E = {esc(cond["E"])} · {esc(cfg["r_definition"])} · {esc(cfg["bootstrap_description"])} · T60 is descriptive only</p>
{verdict_block}{bounds_table}
<div class="legend">{legend}<span><i style="background:var(--muted);height:1px"></i>thin whisker: adjusted room-cluster interval ({lv["r"]}, {st.get("n_rooms", "–")} rooms); thick whisker: adjusted query-level interval ({lv["q"]})</span></div>
<h2>1. Acoustic metrics vs yaw (condition P)</h2><div class="grid2">{"".join(charts)}</div>
<h2>2. Spectral metrics vs yaw (condition P, {n_spec} angles)</h2><div class="grid2">{"".join(spec_charts)}</div>
<h2>3. k = 0 paired comparison, {esc(SERIES_NAME[cyl_label])} − {esc(SERIES_NAME[primary])} (nominal {lv["k0"]} intervals, unadjusted, descriptive)</h2>
<p class="sub">{esc(st["k0_note"])}</p>
<div class="wrap"><table><tr><th>metric</th><th>{esc(SERIES_NAME[cyl_label])}</th><th>{esc(SERIES_NAME[primary])}</th><th>diff</th><th>query-level {lv["k0"]} interval (nominal)</th><th>room-cluster {lv["k0"]} interval (nominal)</th><th>n</th></tr>{k0}</table></div>
<h2>4. H2: difference-in-differences D = r({esc(SERIES_NAME[cyl_label])}) − r({esc(SERIES_NAME[primary])}) per angle (condition P); TOST equivalence of r({esc(SERIES_NAME[cyl_label])}) to zero at patch-aligned angles</h2>
<p class="sub">{esc(h2["equivalence_rule"])} "not established" means the interval was not inside the margin, not that inequivalence was shown.</p>
<div class="wrap"><table><tr><th>metric</th><th>yaw</th><th>r ({esc(SERIES_NAME[cyl_label])})</th><th>r ({esc(SERIES_NAME[primary])})</th><th>D</th><th>adjusted query-level interval ({lv["q"]})</th><th>adjusted room-cluster interval ({lv["r"]})</th><th>TOST r({esc(SERIES_NAME[cyl_label])}) within ±{pct(margin, 0).lstrip("+")} (query / room)</th></tr>{"".join(h2rows)}</table></div>
<h2>5. Per-angle tables — acoustic</h2>{ac_tables}
<h2>6. Per-angle tables — spectral</h2>{sp_tables}
<h2>7. Delay-flip audit: {esc(st["delay_flips_entity"])}{f" (of at most {st['delay_flips_max']})" if st.get("delay_flips_max") else ""}</h2>
<div class="wrap"><table><tr><th>run</th>{"".join(f"<th>k = {k}</th>" for k in flips_cols)}</tr>{flips}</table></div>
<h2>8. Decomposition: relative change per stage</h2><ul>{dec_txt or "<li>not available</li>"}</ul><p class="sub">{esc(st["decomposition_note"])}</p>
<p class="prov">Source: <code>{esc(args.stats)}</code> (mode {esc(st.get("mode", "–"))}{"" if draft else f", summary sha256 {esc(str(st.get('summary_sha256'))[:12])}…"}); runs {esc(", ".join(cfg.get("runs", [])))}. Selected evaluation settings (from each run's metadata): {meta_lines}. Narrative: <code>yaw_rotation_degradation_results.md</code>, <code>yaw_rotation_degradation_analysis.md</code>.</p>
</main><div id="tip"></div><script>const tip=document.getElementById('tip');document.querySelectorAll('[data-tip]').forEach(el=>{{const show=e=>{{tip.textContent=el.dataset.tip;tip.style.display='block';const r=el.getBoundingClientRect();tip.style.left=Math.min((e&&e.clientX)||r.left,window.innerWidth-480)+'px';tip.style.top=(((e&&e.clientY)||r.top)-28)+'px';}};el.addEventListener('mousemove',show);el.addEventListener('focus',()=>show(null));el.addEventListener('mouseleave',()=>tip.style.display='none');el.addEventListener('blur',()=>tip.style.display='none');}});</script></body></html>"""
    with open(out_html, "w") as f:
        f.write(page)
    data_path = os.path.join(HERE, "data.json") if not args.out else os.path.splitext(out_html)[0] + ".data.json"
    with open(data_path, "w") as f:
        json.dump({"final": final, "stats_source": args.stats, "summary_sha256": None if draft else st.get("summary_sha256"), "labels": labels, "run_descriptions": cfg["run_descriptions"],
                   "metric_names": cfg["metric_names"], "metric_units": cfg["metric_units"], "h1": None if draft else h1, "h2": None if draft else h2.get("verdict")}, f, indent=1)
    print("wrote", out_html, "| FINAL" if final else "| DRAFT")


if __name__ == "__main__":
    main()
