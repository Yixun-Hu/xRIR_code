"""Build cylvit_vs_simplevit_01_results.html (offline, inline SVG) from the exp_01 canonical statistics.

Every displayed statistic is read from ckpt/backbone_comparison_stats.json, written by
`tools/summarize_epochs.py --avg-epochs 5-12 --json ...` (the same run that writes summary.txt), so the
page cannot disagree with _results.md; this script computes no bootstrap of its own. Also reads
history.jsonl (lr) and ckpt/baseline_reproduction.json. Writes ../cylvit_vs_simplevit_01_results.html and data.json.
Run from the repo root:  python worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_results_assets/make_results_html.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_HTML = os.path.join(HERE, "..", "cylvit_vs_simplevit_01_results.html")
RUNS = {"cyl": "CylindricalViT + xRIR", "simple": "SimpleViT + xRIR (control)"}
METRICS = [("edt", "EDT error (s)", 4), ("c50", "C50 error (dB)", 3), ("t60", "T60 error (%)", 2), ("loss", "Test loss", 5)]
STATS_JSON = "ckpt/backbone_comparison_stats.json"
SUMMARY_TXT = "ckpt/backbone_comparison_summary.txt"
RESULTS_MD = os.path.join(HERE, "..", "cylvit_vs_simplevit_results.md")
RELEASED = {}  # filled from the canonical JSON in collect()


def collect():
    st = json.load(open(STATS_JSON))
    epochs = sorted(int(e) for e in st["per_epoch"])
    data = {"epochs": epochs, "n_boot": st["n_boot"], "avg_epochs": st["avg_epochs"], "stats_note": st["note"],
            "per_epoch": {k: {"cyl": [], "simple": [], "diff": [], "lo": [], "hi": [], "r_lo": [], "r_hi": []} for k, _, _ in METRICS}}
    for e in epochs:
        for k, _, _ in METRICS:
            r = st["per_epoch"][str(e)][k]; P = data["per_epoch"][k]
            P["cyl"].append(r["cyl"]); P["simple"].append(r["simple"]); P["diff"].append(r["diff"])
            P["lo"].append(r["q_lo"]); P["hi"].append(r["q_hi"]); P["r_lo"].append(r["r_lo"]); P["r_hi"].append(r["r_hi"])
    data["avg"] = {k: {"cyl": v["cyl"], "simple": v["simple"], "diff": v["diff"], "lo": v["q_lo"], "hi": v["q_hi"], "r_lo": v["r_lo"], "r_hi": v["r_hi"],
                       "rel": v["rel_pct"], "wins": v["wins"], "n": v["n_epochs"], "per_room": v["per_room_diff"]} for k, v in st["avg"].items()}
    data["best"] = st["best"]
    data["history"] = {r: [{"epoch": h["epoch"], "train": h["train_loss"], "test": h["test_loss"], "lr": h["lr"]} for h in st["history"][r]] for r in RUNS}
    data["baseline"] = st["baseline_reproduction"]
    RELEASED.update({k: v for k, v in st["released"].items() if v is not None})
    data["released"] = st["released"]; data["paper"] = st["paper"]; data["n_rooms"] = st["n_rooms"]; data["n_queries"] = st["n_queries"]
    # Binding: summary.txt must be the one written by the same producer run as the JSON (sha256 stored in the JSON),
    # and _results.md must quote its sections 2 and 3 as contiguous verbatim blocks.
    import hashlib
    summ = open(SUMMARY_TXT).read()
    assert hashlib.sha256(summ.encode()).hexdigest() == st["summary_sha256"], "summary.txt does not match the run that wrote stats.json"
    md = open(RESULTS_MD).read()
    for start, end in [("2. Paired", "3. Best"), ("3. Best", None)]:
        block = summ[summ.index(start):(summ.index(end) if end else len(summ))].strip().split("\n")[1:]
        block = "\n".join("    " + l for l in block if l.strip())
        assert block in md, f"_results.md is out of date with summary.txt (section starting {start!r} not quoted verbatim)"
    return data


# ---------- SVG helpers ----------
def nice_ticks(lo, hi, n=4):
    span = hi - lo
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min([1, 2, 2.5, 5, 10], key=lambda s: abs(s * mag - raw)) * mag
    t0 = np.floor(lo / step) * step
    return [t0 + i * step for i in range(int(np.ceil((hi - t0) / step)) + 1)]


def fmt(v, nd):
    return f"{v:.{nd}f}"


def line_chart(key, title, nd, data, width=440, height=250):
    epochs = data["epochs"]; P = data["per_epoch"][key]
    ys = P["cyl"] + P["simple"] + ([RELEASED[key]] if key in RELEASED else [])
    lo, hi = min(ys), max(ys); pad = (hi - lo) * 0.12 or 1e-3; lo -= pad; hi += pad
    ticks = nice_ticks(lo, hi); lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
    L, R, T, B = 58, 16, 30, 34; pw, ph = width - L - R, height - T - B
    x = lambda e: L + (e - 1) / 11 * pw; y = lambda v: T + (hi - v) / (hi - lo) * ph
    s = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{title} per epoch">']
    s.append(f'<text class="ttl" x="{L}" y="18">{title} <tspan class="tick">(y-axis zoomed, does not start at 0)</tspan></text>')
    for t in ticks:
        s.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/><text class="tick" x="{L - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{fmt(t, nd if nd < 4 else 4)}</text>')
    for e in epochs:
        s.append(f'<text class="tick" x="{x(e):.1f}" y="{height - 14}" text-anchor="middle">{e}</text>')
    s.append(f'<text class="tick" x="{L + pw / 2:.1f}" y="{height - 2}" text-anchor="middle">epoch</text>')
    s.append(f'<line class="axis" x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}"/>')
    if key in RELEASED:
        yr = y(RELEASED[key]); s.append(f'<line class="ref" x1="{L}" x2="{L + pw}" y1="{yr:.1f}" y2="{yr:.1f}"/><text class="reflbl" x="{L + pw}" y="{yr - 4:.1f}" text-anchor="end">released ckpt {fmt(RELEASED[key], nd)}</text>')
    for run, cls in [("simple", "s2"), ("cyl", "s1")]:
        pts = " ".join(f"{x(e):.1f},{y(v):.1f}" for e, v in zip(epochs, P[run]))
        s.append(f'<polyline class="line {cls}" points="{pts}"/>')
        for e, v, d, l, h in zip(epochs, P[run], P["diff"], P["lo"], P["hi"]):
            tip = f"{RUNS[run]} · epoch {e}: {fmt(v, nd)}"
            s.append(f'<circle class="mk {cls}" cx="{x(e):.1f}" cy="{y(v):.1f}" r="4" data-tip="{tip}"/>')
        s.append(f'<text class="dl {cls}" x="{x(12) + 5:.1f}" y="{y(P[run][-1]) + 4:.1f}">{"cyl" if run == "cyl" else "ctrl"}</text>')
    s.append("</svg>")
    return "\n".join(s)


def diff_chart(key, title, nd, data, width=440, height=230):
    epochs = data["epochs"]; P = data["per_epoch"][key]
    rel = [100 * d / b for d, b in zip(P["diff"], P["simple"])]
    rlo = [100 * l / b for l, b in zip(P["lo"], P["simple"])]; rhi = [100 * h / b for h, b in zip(P["hi"], P["simple"])]
    clo = [100 * l / b for l, b in zip(P["r_lo"], P["simple"])]; chi = [100 * h / b for h, b in zip(P["r_hi"], P["simple"])]
    lo, hi = min(clo + [0]), max(chi + [0]); pad = (hi - lo) * 0.15 or 1; lo -= pad; hi += pad
    ticks = nice_ticks(lo, hi); lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
    L, R, T, B = 58, 16, 30, 34; pw, ph = width - L - R, height - T - B
    x = lambda e: L + (e - 1) / 11 * pw; y = lambda v: T + (hi - v) / (hi - lo) * ph
    s = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{title}: paired difference per epoch">']
    s.append(f'<text class="ttl" x="{L}" y="18">{title}: cyl − ctrl, % of ctrl</text>')
    for t in ticks:
        s.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/><text class="tick" x="{L - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{t:+.0f}%</text>')
    s.append(f'<line class="axis" x1="{L}" x2="{L + pw}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    for e in epochs:
        s.append(f'<text class="tick" x="{x(e):.1f}" y="{height - 14}" text-anchor="middle">{e}</text>')
    s.append(f'<text class="tick" x="{L + pw / 2:.1f}" y="{height - 2}" text-anchor="middle">epoch</text>')
    for e, r, l, h, cl, ch in zip(epochs, rel, rlo, rhi, clo, chi):
        sig = (l > 0) or (h < 0)
        s.append(f'<line class="ci2" x1="{x(e):.1f}" x2="{x(e):.1f}" y1="{y(cl):.1f}" y2="{y(ch):.1f}"/>')
        s.append(f'<line class="ci" x1="{x(e):.1f}" x2="{x(e):.1f}" y1="{y(l):.1f}" y2="{y(h):.1f}"/>')
        s.append(f'<circle class="mk s1 {"sig" if sig else "ns"}" cx="{x(e):.1f}" cy="{y(r):.1f}" r="4.5" data-tip="epoch {e}: {r:+.1f}%; query-level CI [{l:+.1f}, {h:+.1f}]{" *" if sig else ""}; room-cluster CI [{cl:+.1f}, {ch:+.1f}]"/>')
    s.append("</svg>")
    return "\n".join(s)


def main():
    os.chdir(os.path.join(HERE, "..", "..", "..", ".."))
    data = collect()
    with open(os.path.join(HERE, "data.json"), "w") as f:
        json.dump(data, f, indent=1)
    avg = data["avg"]
    def verdict(k):  # query-level (fixed split) and room-cluster (17 rooms) read separately
        q = "cyl lower" if avg[k]["hi"] < 0 else ("control lower" if avg[k]["lo"] > 0 else "no detected difference")
        r = "cyl lower" if avg[k]["r_hi"] < 0 else ("control lower" if avg[k]["r_lo"] > 0 else "no detected difference")
        return q, r
    def vcls(k):
        q, r = verdict(k)
        return "good" if (q == "cyl lower" and r == "cyl lower") else ("goodish" if q == "cyl lower" else ("warn" if q == "control lower" else "neutral"))
    charts = "\n".join(line_chart(k, n, nd, data) for k, n, nd in METRICS)
    dcharts = "\n".join(diff_chart(k, n, nd, data) for k, n, nd in METRICS)
    avg_rows = "".join(
        f"<tr><td>{n}</td><td>{fmt(avg[k]['cyl'], nd)}</td><td>{fmt(avg[k]['simple'], nd)}</td><td>{fmt(avg[k]['diff'], nd)}</td>"
        f"<td>{avg[k]['rel']:+.1f}%</td><td>[{fmt(avg[k]['lo'], nd)}, {fmt(avg[k]['hi'], nd)}]{' *' if verdict(k)[0] != 'no detected difference' else ''}</td>"
        f"<td>[{fmt(avg[k]['r_lo'], nd)}, {fmt(avg[k]['r_hi'], nd)}]{' *' if verdict(k)[1] != 'no detected difference' else ''}</td>"
        f"<td>{avg[k]['wins']}/{avg[k]['n']}</td><td>{verdict(k)[0]}</td><td>{verdict(k)[1]}</td></tr>" for k, n, nd in METRICS)
    best = data["best"]
    best_rows = "".join(
        f"<tr><td>{n}</td><td>{fmt(best[k]['cyl'], nd)}</td><td>{fmt(best[k]['simple'], nd)}</td><td>{fmt(best[k]['diff'], nd)} ({best[k]['rel_pct']:+.1f}%)</td>"
        f"<td>[{fmt(best[k]['q_lo'], nd)}, {fmt(best[k]['q_hi'], nd)}]{' *' if (best[k]['q_hi'] < 0 or best[k]['q_lo'] > 0) else ''}</td>"
        f"<td>[{fmt(best[k]['r_lo'], nd)}, {fmt(best[k]['r_hi'], nd)}]{' *' if (best[k]['r_hi'] < 0 or best[k]['r_lo'] > 0) else ''}</td></tr>" for k, n, nd in METRICS)
    rooms = sorted(avg["edt"]["per_room"]); room_rows = "".join(f"<tr><td>{r}</td>" + "".join(f"<td>{avg[k]['per_room'][r]:+.{nd}f}</td>" for k, _, nd in METRICS) + "</tr>" for r in rooms)
    ep_rows = "".join(
        f"<tr><td>{e}</td>" + "".join(f"<td>{fmt(data['per_epoch'][k]['cyl'][i], nd)} / {fmt(data['per_epoch'][k]['simple'][i], nd)}</td>" for k, _, nd in METRICS)
        + f"<td>{data['history']['cyl'][i]['lr']:.1e}</td></tr>" for i, e in enumerate(data["epochs"]))
    bl = data["baseline"]
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>cylvit_vs_simplevit — results</title>
<style>
:root{{color-scheme:light;--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--good:#006300;--warn:#9a5a00;--card:#f4f3ef}}
@media (prefers-color-scheme:dark){{:root{{color-scheme:dark;--surface:#1a1a19;--text:#fff;--text2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--good:#0ca30c;--warn:#e0a030;--card:#232321}}}}
body{{margin:0;background:var(--surface);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
main{{max-width:1000px;margin:0 auto;padding:24px 20px 60px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px}} .sub{{color:var(--text2);margin:0 0 16px}}
.verdicts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:14px 0}}
.v{{border-left:4px solid var(--muted);background:var(--card);padding:10px 12px;border-radius:6px}} .v.good{{border-color:var(--good)}} .v.goodish{{border-color:var(--s1)}} .v.warn{{border-color:var(--warn)}}
.caveat{{background:var(--card);border-left:4px solid var(--warn);padding:10px 12px;border-radius:6px;margin:10px 0;color:var(--text2)}} .ci2{{stroke:var(--muted);stroke-width:1;opacity:.9}}
.v b{{display:block;font-size:15px}} .v span{{color:var(--text2)}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:12px}}
.chart{{width:100%;height:auto;background:var(--surface)}}
.ttl{{fill:var(--text);font-size:13px;font-weight:600}} .tick{{fill:var(--muted);font-size:11px}} .grid{{stroke:var(--grid);stroke-width:1}} .axis{{stroke:var(--axis);stroke-width:1}}
.line{{fill:none;stroke-width:2;stroke-linejoin:round}} .line.s1{{stroke:var(--s1)}} .line.s2{{stroke:var(--s2)}}
.mk{{stroke:var(--surface);stroke-width:2}} .mk.s1{{fill:var(--s1)}} .mk.s2{{fill:var(--s2)}} .mk.ns{{fill:var(--surface);stroke:var(--s1)}}
.ci{{stroke:var(--s1);stroke-width:2;opacity:.6}} .ref{{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 4}} .reflbl{{fill:var(--muted);font-size:11px}}
.dl{{font-size:11px;font-weight:600}} .dl.s1{{fill:var(--s1)}} .dl.s2{{fill:var(--s2)}}
.legend{{display:flex;gap:18px;color:var(--text2);margin:6px 0 4px}} .legend i{{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:6px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0 10px}} th,td{{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{color:var(--text2);font-weight:600}}
.wrap{{overflow-x:auto}} .pill{{padding:2px 8px;border-radius:10px;background:var(--card);color:var(--text2)}} .pill.good{{color:var(--good);font-weight:600}} .pill.warn{{color:var(--warn);font-weight:600}}
.math{{background:var(--card);padding:10px 14px;border-radius:6px;font-family:ui-serif,Georgia,serif;font-size:15px}}
#tip{{position:fixed;pointer-events:none;background:var(--text);color:var(--surface);padding:4px 8px;border-radius:4px;font-size:12px;display:none;z-index:9}}
small,.prov{{color:var(--text2)}}
</style></head><body><main>
<h1>CylindricalViT vs SimpleViT as the xRIR geometry encoder</h1>
<p class="sub">exp_01 · AcousticRooms unseen-room test split ({data["n_queries"]} queries, {data["n_rooms"]} held-out rooms, K=8) · 12-epoch same-budget training, seed 0 per model · bootstrap {data["n_boot"]} resamples, seed {json.load(open(STATS_JSON))["boot_seed"]} · generated {__import__('datetime').date.today()}</p>
<div class="caveat"><b>Pairing caveat (found in review, 2026-09-05).</b> Queries are paired, but the K=8 reference RIRs were <em>not</em> identical across the two models: the evaluator drew its loader seed from the torch RNG after model construction, and the cylindrical constructor consumes more RNG state, so about 24 % of queries (91 % of those whose receiver has 9 candidate sources) got different reference sets; Griffin-Lim phase initialisation was likewise unpaired. Both are nuisance factors whose <em>expected</em> effect on the paired differences is zero; their realised contribution in these arrays is unknown. A properly paired re-evaluation of the epoch-12 checkpoints is scheduled as the k=0 gate of exp_03. <b>Inference:</b> query-level CIs describe this fixed test split; room-cluster CIs (17 rooms) describe generalisation over rooms and are wider. All intervals are nominal 95 % with no multiplicity adjustment across the five metrics the producer evaluates (four are charted here; log-STFT MSE is in the tables of <code>cylvit_vs_simplevit_results.md</code>).</div>
<div class="verdicts">
{"".join(f'<div class="v {vcls(k)}"><b>{n}</b><span>epochs 5–12 avg: {fmt(avg[k]["cyl"], nd)} vs {fmt(avg[k]["simple"], nd)} ({avg[k]["rel"]:+.1f}%), cyl lower in {avg[k]["wins"]}/{avg[k]["n"]} epochs · fixed split: <b>{verdict(k)[0]}</b> · over rooms: <b>{verdict(k)[1]}</b></span></div>' for k, n, nd in METRICS)}
</div>
<h2>1. Per-epoch means</h2>
<div class="legend"><span><i style="background:var(--s1)"></i>CylindricalViT + xRIR</span><span><i style="background:var(--s2)"></i>SimpleViT + xRIR (control)</span><span><i style="background:var(--muted);height:0;border-top:2px dashed var(--muted)"></i>released checkpoint (reproduced)</span></div>
<div class="grid2">{charts}</div>
<h2>2. Paired difference per epoch (cylindrical − control), 95% bootstrap CIs</h2>
<p class="sub">Thick whisker: query-level CI (this fixed split); thin whisker: room-cluster CI (17 rooms). Filled marker: query-level CI excludes 0. Negative favours cylindrical (all metrics are errors). Each point pairs the same 6337 queries (references not identical, see caveat).</p>
<div class="grid2">{dcharts}</div>
<h2>3. Averaged over epochs {data["avg_epochs"]} (per-sample differences averaged across checkpoints, then bootstrapped; {data["n_boot"]} resamples)</h2>
<div class="wrap"><table><tr><th>metric</th><th>cyl</th><th>ctrl</th><th>diff</th><th>rel</th><th>query-level 95% CI</th><th>room-cluster 95% CI</th><th>epochs cyl lower</th><th>fixed split</th><th>over rooms</th></tr>{avg_rows}</table></div>
<h2>3b. Best-test-loss checkpoints (cyl epoch {best["epochs"]["cyl"]} vs control epoch {best["epochs"]["simple"]})</h2>
<div class="wrap"><table><tr><th>metric</th><th>cyl</th><th>ctrl</th><th>diff</th><th>query-level 95% CI</th><th>room-cluster 95% CI</th></tr>{best_rows}</table></div>
<h2>3c. Per-room mean difference (cyl − ctrl), epochs {data["avg_epochs"]} average</h2>
<div class="wrap"><table><tr><th>room</th>{"".join(f"<th>{n}</th>" for _, n, _ in METRICS)}</tr>{room_rows}</table></div>
<h2>4. Baseline reproduction (released checkpoints, untouched eval scripts)</h2>
<div class="wrap"><table><tr><th>split</th><th>n</th><th>EDT (s)</th><th>C50 (dB)</th><th>T60 (%)</th><th>paper Table 1</th></tr>
<tr><td>unseen</td><td>{bl['unseen']['n']}</td><td>{bl['unseen']['edt_error_s']}</td><td>{bl['unseen']['c50_error_db']}</td><td>{bl['unseen']['t60_error_pct']}</td><td>{bl['unseen']['paper_table1']['edt_error_s']} / {bl['unseen']['paper_table1']['c50_error_db']} / {bl['unseen']['paper_table1']['t60_error_pct']}</td></tr>
<tr><td>seen</td><td>{bl['seen']['n']}</td><td>{bl['seen']['edt_error_s']}</td><td>{bl['seen']['c50_error_db']}</td><td>{bl['seen']['t60_error_pct']}</td><td>{bl['seen']['paper_table1']['edt_error_s']} / {bl['seen']['paper_table1']['c50_error_db']} / {bl['seen']['paper_table1']['t60_error_pct']}</td></tr></table></div>
<h2>5. Per-epoch table (cyl / ctrl)</h2>
<div class="wrap"><table><tr><th>epoch</th>{"".join(f"<th>{n}</th>" for _, n, _ in METRICS)}<th>lr</th></tr>{ep_rows}</table></div>
<h2>6. Method</h2>
<p>Only <code>source_network</code> differs: <code>SimpleViT</code> (absolute 2-D sin/cos positional embedding) vs <code>CylindricalViT</code> (per-column gauge alignment R<sub>z</sub>(−θ<sub>c</sub>) of the xyz panorama, elevation-only positional encoding, circular relative-position attention bias); same dim 512, depth 12, 8 heads, 256 tokens. Loss for both:</p>
<div class="math">ℒ = ‖ log(Ŝ + ε) − log(S + ε) ‖<sub>1</sub> + 0.01 · ‖ EDC<sub>dB</sub>(Ŝ) − EDC<sub>dB</sub>(S) ‖<sub>1</sub>,&nbsp; ε = 10<sup>−8</sup>, EDC = Schroeder backward integration over time of the summed power spectrogram</div>
<p>Training: AdamW lr 10<sup>−3</sup>, wd 10<sup>−4</sup>, lr × 0.1 every 3 epochs (12 epochs; the authors' 200/50 compressed at the same ratio), effective batch 64 (32 × 2 accumulation), TF32, K=8 references re-drawn each step. Metrics: Griffin-Lim inversion of the predicted magnitude, EDT / C50 / T60 on the first 8000 samples exactly as <code>eval_unseen.py</code>.</p>
<p class="prov">Sources: every number on this page (statistics, learning rates, baseline reproduction, paper/released references) is read from the canonical <code>ckpt/backbone_comparison_stats.json</code> written by <code>tools/summarize_epochs.py --avg-epochs 5-12 --json</code>, the same run that writes <code>ckpt/backbone_comparison_summary.txt</code>; the generator asserts that <code>cylvit_vs_simplevit_results.md</code> quotes that summary verbatim and computes no statistics of its own. Extract: <code>cylvit_vs_simplevit_results_assets/data.json</code>. Narrative: <code>cylvit_vs_simplevit_results.md</code>, <code>cylvit_vs_simplevit_analysis.md</code>.</p>
</main><div id="tip"></div>
<script>
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{{el.addEventListener('mousemove',e=>{{tip.textContent=el.dataset.tip;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-28)+'px';}});el.addEventListener('mouseleave',()=>tip.style.display='none');}});
</script></body></html>"""
    with open(OUT_HTML, "w") as f:
        f.write(html)
    print("wrote", os.path.abspath(OUT_HTML), "and data.json;", {k: (round(v["rel"], 2), verdict(k)) for k, v in avg.items()})


if __name__ == "__main__":
    main()
