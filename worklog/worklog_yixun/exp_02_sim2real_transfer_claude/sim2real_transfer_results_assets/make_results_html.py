"""Build sim2real_transfer_01_results.html (offline, inline SVG) from ckpt/sim2real/.

Uses sim_to_real.summarize_haa.load_runs / paired for the numbers so the page and summary.txt agree.
Run from the repo root with PYTHONPATH set:
    python worklog/worklog_yixun/exp_02_sim2real_transfer_claude/sim2real_transfer_results_assets/make_results_html.py
"""
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, REPO)
from sim_to_real.summarize_haa import ROOMS, completeness, load_runs  # noqa: E402

OUT_HTML = os.path.join(HERE, "..", "sim2real_transfer_01_results.html")
STATS_JSON = os.path.join(REPO, "ckpt", "sim2real", "stats.json")  # canonical numbers: sim_to_real/summarize_haa.py --json
METRICS = [("edt", "EDT error (s)", 4), ("c50", "C50 error (dB)", 3), ("t60", "T60 error (%)", 2)]
ROOM_LABEL = {"class_room": "Classroom", "dampened_room": "Dampened room", "hallway": "Hallway", "complex_room": "Complex room"}
MODELS = [("released", "fine-tuned", "Released xRIR (SimpleViT), fine-tuned", "s3"),
          ("released_repomaps", "fine-tuned", "Released, fine-tuned, as-released depth maps (side row)", "s4"),
          ("control", "fine-tuned", "SimpleViT control (ours), fine-tuned", "s2"),
          ("cyl", "fine-tuned", "CylindricalViT (ours), fine-tuned", "s1"),
          ("released", "zero-shot", "Released, zero-shot", "z3"),
          ("control", "zero-shot", "SimpleViT control, zero-shot", "z2"),
          ("cyl", "zero-shot", "CylindricalViT, zero-shot", "z1")]


def per_run_means(stats):
    """Per-run means straight from the canonical stats.json rows (no recomputation from raw files)."""
    out = {}
    for key, row in stats["rows"].items():
        init, kind, room, k = key.split("|")
        pts = [(sd, v) for sd, v in zip(row["seeds"], row["per_run"]) if v is not None]
        if pts:
            out[(init, kind, room, k)] = pts
    return out


PAPER = {}  # filled from stats.json in main()


def dot_chart(k, name, nd, means, stats, width=470, height=240):
    groups = [(m, lbl, cls) for m, _, lbl, cls in [(x[0] + "|" + x[1], None, x[2], x[3]) for x in MODELS]]
    L, R, T, B = 52, 12, 30, 40; pw, ph = width - L - R, height - T - B
    vals = []
    for room in ROOMS:
        for init, kind, _, _ in MODELS:
            vals += [v for _, v in means.get((init, kind, room, k), [])]
        if PAPER[room][[i for i, (kk, _, _) in enumerate(METRICS) if kk == k][0]] is not None:
            vals.append(PAPER[room][[i for i, (kk, _, _) in enumerate(METRICS) if kk == k][0]])
    if not vals:
        return ""
    lo, hi = 0, max(vals) * 1.12
    nroom = len(ROOMS); gw = pw / nroom
    y = lambda v: T + (hi - v) / (hi - lo) * ph
    s = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{name} per room and model">', f'<text class="ttl" x="{L}" y="18">{name}</text>']
    step = hi / 4
    for i in range(5):
        t = i * step; s.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/><text class="tick" x="{L - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{t:.{nd if nd < 3 else 3}f}</text>')
    s.append(f'<line class="axis" x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}"/>')
    for ri, room in enumerate(ROOMS):
        x0 = L + ri * gw
        s.append(f'<text class="tick" x="{x0 + gw / 2:.1f}" y="{height - 22}" text-anchor="middle">{ROOM_LABEL[room]}</text>')
        pv = PAPER[room][[i for i, (kk, _, _) in enumerate(METRICS) if kk == k][0]]
        if pv is not None:
            s.append(f'<line class="ref" x1="{x0 + 6:.1f}" x2="{x0 + gw - 6:.1f}" y1="{y(pv):.1f}" y2="{y(pv):.1f}" data-tip="paper Table 2 xRIR K=8, {ROOM_LABEL[room]}: {pv}"/>')
        slots = [m for m in MODELS if (m[0], m[1], room, k) in means]
        for mi, (init, kind, lbl, cls) in enumerate(slots):
            xs = x0 + gw * (mi + 1) / (len(slots) + 1)
            pts = means[(init, kind, room, k)]
            mean = stats["rows"][f"{init}|{kind}|{room}|{k}"]["mean"]  # canonical seed mean, not recomputed
            for seed, v in pts:
                s.append(f'<circle class="mk {cls}" cx="{xs:.1f}" cy="{y(v):.1f}" r="4" data-tip="{lbl} · {ROOM_LABEL[room]} · {seed}: {v:.{nd}f}"/>')
            if len(pts) > 1:
                s.append(f'<line class="mean {cls}" x1="{xs - 7:.1f}" x2="{xs + 7:.1f}" y1="{y(mean):.1f}" y2="{y(mean):.1f}" data-tip="{lbl} · {ROOM_LABEL[room]} · mean of {len(pts)} seeds: {mean:.{nd}f}"/>')
    s.append("</svg>")
    return "\n".join(s)


def main():
    os.chdir(REPO)
    runs = load_runs("ckpt/sim2real")
    complete, missing = completeness(runs)
    stats = json.load(open(STATS_JSON))
    if stats["draft"] != (not complete) or sorted(stats["missing"]) != sorted(missing):
        raise SystemExit("stats.json is stale: its completeness/missing list disagrees with the run set on disk; regenerate with summarize_haa.py --json")
    import hashlib
    summ_path = os.path.join(REPO, stats["summary_path"]) if not os.path.isabs(stats["summary_path"]) else stats["summary_path"]
    assert hashlib.sha256(open(summ_path).read().encode()).hexdigest() == stats["summary_sha256"], "summary.txt does not match the run that wrote stats.json"
    # exact per-run agreement between the runs on disk and the canonical rows (check only; the page displays the JSON values)
    for (init, kind), rs in runs.items():
        for r in rs:
            sd = os.path.basename(r["dir"])
            for room in ROOMS:
                if room not in r["per"]:
                    continue
                for k, _, _ in METRICS:
                    v = np.asarray(r["per"][room][k], float); v = v[np.isfinite(v)]
                    row = stats["rows"][f"{init}|{kind}|{room}|{k}"]
                    canon = row["per_run"][row["seeds"].index(sd)]
                    assert (canon is None and v.size == 0) or (canon is not None and abs(canon - v.mean()) < 1e-9), (init, kind, sd, room, k)
    for key, row in stats["rows"].items():  # canonical mean/std must equal the aggregate of their own per_run entries
        vals = [v for v in row["per_run"] if v is not None]
        assert (row["mean"] is None and not vals) or abs(row["mean"] - float(np.mean(vals))) < 1e-12, key
        assert (row["std"] is None and len(vals) <= 1) or (len(vals) > 1 and abs(row["std"] - float(np.std(vals))) < 1e-12), key
    draft = stats["draft"]
    PAPER.update({k: tuple(v) for k, v in stats["paper"].items()})
    means = per_run_means(stats)
    for (init, kind, room, k), pts in means.items():  # seed labels must be the planned ones
        exp = stats["expected_runs"].get(f"{init}|{kind}", [])
        assert all(sd in exp for sd, _ in pts), (init, kind, room, k, [sd for sd, _ in pts], exp)
    if not draft:
        for key, exp in stats["expected_runs"].items():
            init, kind = key.split("|")
            for room in ROOMS:
                for k, _, _ in METRICS:
                    if k == "t60" and room == "dampened_room":
                        continue
                    assert sorted(sd for sd, _ in means.get((init, kind, room, k), [])) == sorted(exp), ("final page requires every planned run", init, kind, room, k)
    extract = {f"{i}|{kd}|{r}|{k}": v for (i, kd, r, k), v in means.items()}
    # summary table rows
    rows = []
    for init, kind, lbl, _ in MODELS:
        if (init, kind) not in runs:
            continue
        cells = []
        for room in ROOMS:
            for k, _, nd in METRICS:
                row = stats["rows"].get(f"{init}|{kind}|{room}|{k}")
                cells.append("–" if not row or row["mean"] is None else (f"{row['mean']:.{nd}f}" + (f" ±{row['std']:.{nd}f}" if row["std"] is not None else "")))
        rows.append(f"<tr><td>{lbl} (n={len(runs[(init, kind)])})</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    paper_row = "<tr><td>Paper Table 2, xRIR K=8</td>" + "".join(f"<td>{'–' if v is None else v}</td>" for room in ROOMS for v in stats["paper"][room]) + "</tr>"
    # paired cyl vs control, pooled over seeds: numbers come from the canonical stats.json (query-cluster bootstrap)
    prow, verdicts = [], {}
    for row in stats["paired"]:
        k = row["metric"]; nd = {kk: ndd for kk, _, ndd in METRICS}[k]; name = {kk: nn for kk, nn, _ in METRICS}[k]
        lo, hi, lo2, hi2, m, room = row["lo"], row["hi"], row["lo_two_way"], row["hi_two_way"], row["diff"], row["room"]
        v = "cyl lower" if hi2 < 0 else ("control lower" if lo2 > 0 else "no detected difference")  # verdict on the two-way (query x seed) interval
        if not draft:
            verdicts[(room, k)] = v
        pill = "good" if v == "cyl lower" else ("warn" if v == "control lower" else "")
        star = (lambda a, b: "*" if (a > 0 or b < 0) else "") if not draft else (lambda a, b: "")
        seeds_txt = ", ".join(f"{sd}: {d:+.{nd}f} (n={row['per_seed_valid'][sd]})" for sd, d in row["per_seed_diff"].items() if d is not None)
        prow.append(f"<tr><td>{ROOM_LABEL[room]}</td><td>{name}</td><td>{row['cyl']:.{nd}f}</td><td>{row['ctrl']:.{nd}f}</td>"
                    f"<td>{m:+.{nd}f} ({100 * m / row['ctrl']:+.1f}%)</td><td>[{lo:+.{nd}f}, {hi:+.{nd}f}]{star(lo, hi)}</td><td>[{lo2:+.{nd}f}, {hi2:+.{nd}f}]{star(lo2, hi2)}</td>"
                    f"<td>{row['n_valid']}/{row['n_total']} ({row['n_queries']} q × {row['n_seeds']} s)</td><td>{seeds_txt}</td><td><span class='pill {pill if not draft else ''}'>{'' if draft else v}</span></td></tr>")
    extract["paired_seeds"] = sorted({sd for row in stats["paired"] for sd in row["per_seed_diff"]}) if stats.get("paired") else []
    with open(os.path.join(HERE, "data.json"), "w") as f:
        json.dump({"draft": draft, "missing": stats["missing"], "means": extract, "paper": stats["paper"], "n_boot": stats["n_boot"], "boot_seed": stats["boot_seed"],
                   "verdicts": ({f"{r}|{k}": v for (r, k), v in verdicts.items()} if not draft else "suppressed: draft")}, f, indent=1)
    charts = "\n".join(dot_chart(k, n, nd, means, stats) for k, n, nd in METRICS)
    legend = "".join(f'<span><i class="{cls}"></i>{lbl}</span>' for init, kind, lbl, cls in MODELS if (init, kind) in runs)
    nv = {v: sum(1 for x in verdicts.values() if x == v) for v in ["cyl lower", "no detected difference", "control lower"]}
    banner = ("<div class='draft'>DRAFT — incomplete run set: " + "; ".join(stats["missing"]) + ". No verdicts are drawn from a partial set.</div>") if draft else ""
    verdict_block = "" if draft else f'<div class="verdicts"><div class="v good"><b>{nv["cyl lower"]} room×metric cells: cylindrical lower</b></div><div class="v"><b>{nv["no detected difference"]} no detected difference</b></div><div class="v warn"><b>{nv["control lower"]} cells: control lower</b></div><p class="sub" style="flex-basis:100%">Nominal 95% intervals per cell, no multiplicity adjustment across the {len(verdicts)} cells; "no detected difference" is not evidence of equivalence.</p></div>'
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>sim2real_transfer — results</title>
<style>
:root{{color-scheme:light;--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#006300;--warn:#9a5a00;--card:#f4f3ef}}
@media (prefers-color-scheme:dark){{:root{{color-scheme:dark;--surface:#1a1a19;--text:#fff;--text2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#0ca30c;--warn:#e0a030;--card:#232321}}}}
body{{margin:0;background:var(--surface);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}} main{{max-width:1000px;margin:0 auto;padding:24px 20px 60px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px}} .sub{{color:var(--text2);margin:0 0 14px}}
.verdicts{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}} .draft{{background:#b00020;color:#fff;font-weight:700;padding:10px 14px;border-radius:6px;margin:10px 0;font-size:15px}} .s4{{fill:var(--muted);stroke:var(--muted)}} .legend i.s4{{background:var(--muted);border-color:var(--muted)}}
.v{{border-left:4px solid var(--muted);background:var(--card);padding:8px 12px;border-radius:6px;min-width:180px}} .v.good{{border-color:var(--good)}} .v.warn{{border-color:var(--warn)}} .v b{{display:block}}
.chart{{width:100%;height:auto;background:var(--surface)}} .ttl{{fill:var(--text);font-size:13px;font-weight:600}} .tick{{fill:var(--muted);font-size:11px}} .grid{{stroke:var(--grid)}} .axis{{stroke:var(--axis)}}
.mk{{stroke:var(--surface);stroke-width:1.5}} .mk.s1,.mean.s1{{fill:var(--s1);stroke:var(--s1)}} .mk.s2,.mean.s2{{fill:var(--s2);stroke:var(--s2)}} .mk.s3,.mean.s3{{fill:var(--s3);stroke:var(--s3)}}
.mk.z1{{fill:var(--surface);stroke:var(--s1)}} .mk.z2{{fill:var(--surface);stroke:var(--s2)}} .mk.z3{{fill:var(--surface);stroke:var(--s3)}} .mean{{stroke-width:3}}
.ref{{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 4}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;color:var(--text2);margin:6px 0 4px;font-size:13px}} .legend i{{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle;margin-right:5px;border:2px solid}}
.legend i.s1{{background:var(--s1);border-color:var(--s1)}} .legend i.s2{{background:var(--s2);border-color:var(--s2)}} .legend i.s3{{background:var(--s3);border-color:var(--s3)}} .legend i.z1{{border-color:var(--s1)}} .legend i.z2{{border-color:var(--s2)}} .legend i.z3{{border-color:var(--s3)}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0 10px}} th,td{{padding:5px 7px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{color:var(--text2);font-weight:600}} .wrap{{overflow-x:auto}}
.pill{{padding:2px 8px;border-radius:10px;background:var(--card);color:var(--text2)}} .pill.good{{color:var(--good);font-weight:600}} .pill.warn{{color:var(--warn);font-weight:600}}
#tip{{position:fixed;pointer-events:none;background:var(--text);color:var(--surface);padding:4px 8px;border-radius:4px;font-size:12px;display:none;z-index:9}} .prov{{color:var(--text2)}}
</style></head><body><main>
<h1>Sim-to-real transfer on Hearing Anything Anywhere{" — DRAFT" if draft else ""}</h1>
{banner}
<p class="sub">exp_02 · four real rooms, DiffRIR splits (12 training RIRs per room), two-stage fine-tuning per the paper, K=8, checkpoint selection on the validation split, per-room test split; references are identical for every model and seed (deterministic per query), Griffin-Lim phase initialisation was not seeded (a nuisance whose expected effect on paired differences is zero; its realised contribution is unknown) · bootstrap {stats["n_boot"]} resamples, {stats["bootstrap"]} · generated {datetime.date.today()}</p>
{verdict_block}
<h2>1. Per-room results (dots = seeds, bar = seed mean, dashed = paper Table 2 xRIR)</h2>
<div class="legend">{legend}<span><i style="border:none;width:14px;height:0;border-top:2px dashed var(--muted);border-radius:0"></i>paper Table 2</span></div>
{charts}
<h2>2. Table (mean ± std over seeds, from the canonical stats)</h2>
<div class="wrap"><table><tr><th>model</th>{"".join(f"<th>{ROOM_LABEL[r]} {n.split(' ')[0]}</th>" for r in ROOMS for _, n, _ in METRICS)}</tr>{paper_row}{"".join(rows)}</table></div>
<h2>3. Paired cylindrical − control (fine-tuned), per-sample differences pooled over seeds</h2>
<p class="sub">Same (seed, query, references) for both models; negative favours cylindrical.{"" if draft else " * = nominal 95% bootstrap CI excludes 0."}</p>
<p class="sub">Query-cluster CI: rows of one query across training seeds resampled together (conditional on the realised seeds). Two-way CI: queries and training seeds both resampled; verdicts use the two-way interval. {"Stars and verdicts are suppressed in this draft." if draft else "* = nominal 95% CI excludes 0; per-cell, no multiplicity adjustment."}</p>
<div class="wrap"><table><tr><th>room</th><th>metric</th><th>cyl</th><th>ctrl</th><th>diff</th><th>95% CI query-cluster</th><th>95% CI two-way</th><th>valid/total</th><th>per-seed diff (valid n)</th><th>{"" if draft else "verdict"}</th></tr>{"".join(prow)}</table></div>
<h2>4. Protocol notes</h2>
<p>Speaker = model "receiver" (panorama rendered at the speaker), mics = "sources". Stage 1: classroom + hallway + complex (dampened excluded), 36 targets, 1000 full-batch epochs, AdamW lr 10<sup>−4</sup>, wd 10<sup>−4</sup>, lr × 0.1 per 50 epochs; stage 2: per room, 12 targets, 200 epochs. RIRs normalised per room by the training-set peak. T60 omitted for the dampened room (paper). Depth maps: released repo maps for classroom/hallway; for the complex and dampened rooms the released assets are unusable (complex is a byte-identical copy of the hallway map; dampened is misaligned with the mic grid), so the HAA_processed renders are used; a side row re-runs the released checkpoint on the as-released maps.</p>
<p class="prov">Sources: <code>ckpt/sim2real/&lt;init&gt;/{{zeroshot,seed*/eval}}/per_sample_&lt;room&gt;.json</code>; every displayed statistic is read from the canonical <code>ckpt/sim2real/stats.json</code> written by <code>sim_to_real/summarize_haa.py --json</code> (which also writes <code>summary.txt</code>); the page asserts exact agreement of the per-run means. Extract: <code>sim2real_transfer_results_assets/data.json</code>. Narrative (final only): <code>sim2real_transfer_results.md</code>, <code>sim2real_transfer_analysis.md</code>.</p>
</main><div id="tip"></div><script>const tip=document.getElementById('tip');document.querySelectorAll('[data-tip]').forEach(el=>{{el.addEventListener('mousemove',e=>{{tip.textContent=el.dataset.tip;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-28)+'px';}});el.addEventListener('mouseleave',()=>tip.style.display='none');}});</script></body></html>"""
    with open(OUT_HTML, "w") as f:
        f.write(html)
    print("wrote", os.path.abspath(OUT_HTML), "| DRAFT" if draft else "| FINAL", "| runs:", {f"{i}/{k}": len(v) for (i, k), v in runs.items()}, "| verdict counts:", nv if not draft else "(suppressed in draft)")


if __name__ == "__main__":
    main()
