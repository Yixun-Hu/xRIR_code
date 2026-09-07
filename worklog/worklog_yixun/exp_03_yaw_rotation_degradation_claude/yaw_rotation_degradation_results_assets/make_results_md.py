"""Write yaw_rotation_degradation_results.md from the canonical full-mode JSON: producer-authored tables and fields only.

Same FINAL predicate as make_results_html.py (mode == "full", valid_for_confirmatory is True, exploratory is not True,
summary hash binds). No statistics, no local maxima: the distance-to-threshold lines come from the producer field h1.bounds
and the interval levels from config.interval_levels.

    python .../make_results_md.py --stats ckpt/yaw_rotation/full_stats.json
"""
import argparse
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
OUT = os.path.join(HERE, "..", "yaw_rotation_degradation_results.md")
NAME = {}   # filled from config.run_descriptions
K0_PRECISION = {"edt": 5, "c50": 4, "t60": 3, "log_mse": 5, "loss": 6}   # display precision only
PRECISION = {"edt": 4, "c50": 3, "t60": 2}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def p(v, nd=2):
    return "–" if v is None else f"{100 * v:+.{nd}f}"


def ci(lo, hi, nd=2):
    return "–" if lo is None or hi is None else f"[{p(lo, nd)}, {p(hi, nd)}]"




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


def is_final(st):
    return st.get("mode") == "full" and st.get("valid_for_confirmatory") is True and st.get("exploratory") is not True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stats", required=True); ap.add_argument("--out", default=None, help="output path (default: the record's results.md)")
    ap.add_argument("--draft-ok", action="store_true", help="tests only: render a non-final JSON with a DRAFT header instead of refusing"); a = ap.parse_args(); os.chdir(REPO)
    st = json.load(open(a.stats)); out_path = os.path.abspath(a.out) if a.out else OUT
    draft = not is_final(st)
    if draft and not a.draft_ok:
        raise SystemExit("results.md is written from a valid, non-exploratory full-mode JSON only")
    sp, want = st.get("summary_path"), st.get("summary_sha256")
    if not draft and (not sp or not want or sha(sp) != want):
        raise SystemExit("summary hash mismatch or missing")
    want = want or "(draft)"
    cfg = st["config"]; labels = list(cfg["labels"]); roles = cfg["roles"]
    lv = cfg.get("interval_levels")
    need = ["labels", "roles", "metric_names", "metric_units", "condition_definitions", "angle_counts", "run_descriptions", "interval_levels", "r_definition", "bootstrap_description"]
    if any(k not in cfg for k in need) or any(k not in st for k in ("k0_note", "delay_flips_entity", "decomposition_stages", "decomposition_note")) \
            or any(k not in st["h1"] for k in ("rule", "wording_rule")) or any(k not in st["h2"] for k in ("rule", "equivalence_rule")) or "rule" not in st["convergence"]:
        raise SystemExit("producer fields missing: regenerate the JSON with the current tools/summarize_yaw.py")
    if any(l not in cfg["run_descriptions"] for l in labels) or any(m not in cfg["metric_names"] or m not in cfg["metric_units"] for m in ("edt", "c50", "t60", "loss", "log_mse", "consistency")) or any(k not in cfg["roles"] for k in ("primary", "cyl")):
        raise SystemExit("producer mappings incomplete (run_descriptions / metric_names / metric_units / roles)")
    if any(l not in st["acoustic"] or l not in st["spectral"] or l not in st["meta"] for l in labels):
        raise SystemExit("labels without acoustic/spectral/meta sections")
    check_coverage(st, cfg, not draft)
    NAME.update(cfg["run_descriptions"]); MN, MU, cond, ac = cfg["metric_names"], cfg["metric_units"], cfg["condition_definitions"], cfg["angle_counts"]
    primary, cyl_label = roles["primary"], roles["cyl"]
    LQ, LR, LK = (f"{100 * lv[k]:.2f} %" for k in ("degradation_query", "degradation_room", "k0"))
    h1, h2v = st["h1"], st["h2"]["verdict"]; bounds = h1.get("bounds")
    if not bounds:
        raise SystemExit("h1.bounds missing: regenerate the JSON with the current tools/summarize_yaw.py")
    L = ["# Results — yaw_rotation_degradation" + (" — DRAFT (non-final JSON; tests only)" if draft else "") + "\n"]
    L.append(f"AcousticRooms unseen test split, all {st['n_queries']} queries in {st['n_rooms']} rooms; references per query fixed by the hashed manifest `{st['manifest_hash']}` (identical for every model and angle; its seed and shot count are recorded in `_params_set_up.md`). Evaluation settings per run are in §6 (from the runs' own metadata). The whole scene (receiver-centred depth panorama and all source / reference-source coordinates) is rotated in yaw by k panorama columns; the true RIR and target are unchanged. Condition P = {cond['P']}; condition E = {cond['E']}. {cfg['r_definition']} {cfg['bootstrap_description']} ({LQ} query-level and {LR} room-cluster intervals, {st['n_rooms']} rooms resampled; k = 0 at nominal {LK}). T60 is descriptive only. Canonical numbers: `ckpt/yaw_rotation/full_stats.json` (sha256 `{sha(a.stats)}`, copy in `_results_assets/`) and the hash-bound `ckpt/yaw_rotation/full_summary.txt` (sha256 `{want}`, copy `yaw_rotation_degradation_full_summary.txt`); `tools/summarize_yaw.py --mode full` reported `valid_for_confirmatory: true`.\n")
    L.append("## 1. Pre-registered verdicts and distance to the threshold (producer fields h1 / h2 / h1.bounds)\n")
    L.append(f"- **H1: {h1['verdict']}** — rule: {h1['rule']} Wording: {h1['wording_rule']} Roles evaluated: {', '.join(NAME[r] if r in NAME else r for r in h1['roles_evaluated'])}."
             + f" The +{100 * cfg['threshold']:.0f} % margins at 0° are " + "; ".join(f"{NAME[k]}: {v['edt']:.4f} {MU['edt']} {MN['edt']}, {v['c50']:.3f} {MU['c50']} {MN['c50']}" for k, v in h1["absolute_margins"].items()) + ".")
    L.append(f"- **H2: {h2v['aggregate']}** — rule: {st['h2']['rule']} ({h2v['n_passing']} of {h2v['n_cells']} H1-passing cells). D_k and the TOST results are in §4.")
    conv = st["convergence"]
    L.append(f"- {conv['rule']} {len(conv['cells'])} bounds; max movement {conv['max_rel_change']:.4f} of the interval width (limit {conv['limit']}): {'pass' if conv['pass'] else 'FAIL'}.")
    L.append(f"- Distance to the +{100 * cfg['threshold']:.0f} % threshold (condition P, non-zero acoustic angles):\n")
    L.append("    model                                                   metric          largest adjusted query-level upper bound   query-level bounds above threshold   largest adjusted room-cluster upper bound   room-cluster bounds above threshold")
    for l in labels:
        for m in cfg["confirmatory_metrics"]:
            b = bounds[l][m]; mq, mr = b["max_query_upper"], b["max_room_upper"]
            aq = ", ".join(f"{c['deg']:+.1f}° ({p(c['hi'])} %)" for c in b["query_upper_above_threshold"]) or "none"
            ar = ", ".join(f"{c['deg']:+.1f}° ({p(c['r_hi'])} %)" for c in b["room_upper_above_threshold"]) or "none"
            L.append(f"    {NAME[l]:55s} {MN[m]:12s}   {p(mq['hi']):>8} % at {mq['deg']:+.1f}°{'':>19} {aq:36s} {p(mr['r_hi']):>8} % at {mr['deg']:+.1f}°{'':>19} {ar}")
    L.append("")
    L.append("## 2. Acoustic metrics per angle (" + ", ".join(f"{MN[m]} [{MU[m]}]" for m in ("edt", "c50", "t60")) + ")\n")
    L.append(f"Columns: mean at 0°, mean at k, r_k, adjusted query-level interval ({LQ}), adjusted room-cluster interval ({LR}), n paired-valid, fraction of usable baselines the angle broke (newly invalid). Condition P at the {ac['acoustic']}-angle grid, condition E at k = 0 and the {ac['e_acoustic']} E angles.\n")
    for c_ in ("P", "E"):
        for l in labels:
            for m, unit, nd in [(m, MU[m], PRECISION[m]) for m in ("edt", "c50", "t60")]:
                L.append(f"    {NAME[l]} | condition {c_} | {MN[m]} ({unit})")
                L.append("      deg      mean0     mean_k        r_k       adj. query CI          adj. room CI      n   newinv")
                for r in sorted(st["acoustic"][l][c_][m], key=lambda r: r["deg"]):
                    L.append(f"    {r['deg']:+7.1f} {r['mean0']:10.{nd}f} {r['meank']:10.{nd}f} {p(r['r']):>9} % {ci(r['lo'], r['hi']):>18} {ci(r['r_lo'], r['r_hi']):>18} {r['n_valid']:6d}   {r['validity']['newly_invalid_frac']:.4f}")
                L.append("")
    L.append(f"## 3. Spectral metrics per angle (all {ac['spectral']} angles, conditions P and E)\n")
    L.append(f"loss = {MN['loss']}; log_mse = {MN['log_mse']}; {MN['consistency']} (baseline 0, so no ratio). Intervals: adjusted query-level ({LQ}).\n")
    for c_ in ("P", "E"):
        for l in labels:
            L.append(f"    {NAME[l]} | condition {c_}")
            L.append(f"      deg   {MN['loss'][:11]:>11s}    r(loss)       adj. query CI  {MN['log_mse'][:11]:>11s}   r(logmse)       adj. query CI   {MN['consistency'][:13]:>13s}   (full names: loss = {MN['loss']}; log_mse = {MN['log_mse']}; consistency = {MN['consistency']})")
            rows = {m: {r["k"]: r for r in st["spectral"][l][c_][m]} for m in ["loss", "log_mse", "consistency"]}
            for k in sorted(rows["loss"], key=lambda k: rows["loss"][k]["deg"]):
                a_, b_, cc = rows["loss"][k], rows["log_mse"][k], rows["consistency"][k]
                L.append(f"    {a_['deg']:+7.1f} {a_['meank']:11.5f} {p(a_['r']):>8} % {ci(a_['lo'], a_['hi']):>18} {b_['meank']:11.5f} {p(b_['r']):>8} % {ci(b_['lo'], b_['hi']):>18} {cc['meank']:13.5f}")
            L.append("")
    L.append(f"## 4. Paired {NAME[cyl_label]} − {NAME[primary]} at k = 0, and H2 per angle\n")
    L.append(f"k = 0; nominal {LK} intervals, unadjusted, descriptive (two single-seed checkpoints). {st['k0_note']}\n")
    L.append(f"    metric             [A]        [B]        diff    rel %                query CI                 room CI      n   ([A] = {NAME[cyl_label]}; [B] = {NAME[primary]})")
    for r in st["k0"]:
        nd = K0_PRECISION.get(r["metric"], 5); unit = MU.get(r["metric"], ""); name = MN.get(r["metric"], r["metric"]) + (f" ({unit})" if unit else "")
        L.append(f"    {name:16s} {r['cyl']:10.{nd}f} {r['ctrl']:10.{nd}f} {r['diff']:+10.{nd}f} {r['rel_pct']:+7.1f} [{r['q_lo']:+10.{nd}f}, {r['q_hi']:+10.{nd}f}] [{r['r_lo']:+10.{nd}f}, {r['r_hi']:+10.{nd}f}] {r['n_valid']:6d}")
    L.append("")
    L.append(f"{st['h2']['rule']} Intervals: adjusted query-level {LQ} and room-cluster {LR}. {st['h2']['equivalence_rule']} Rendered as \"equivalent\" or \"not established\" (a failed TOST does not show inequivalence).\n")
    for m in cfg["confirmatory_metrics"]:
        L.append(f"    {MN[m]} ({MU[m]})")
        L.append(f"      deg      r[A]     r[B]       D_k       adj. query CI          adj. room CI   TOST r[A] (query / room)   ([A] = {NAME[cyl_label]}; [B] = {NAME[primary]})")
        for r in sorted(st["h2"]["rows"][m], key=lambda r: r["deg"]):
            eq = r.get("equivalence") or {}
            if eq and ("query" not in eq or "cluster" not in eq or "equivalent" not in eq["query"] or "equivalent" not in eq["cluster"]):
                raise SystemExit(f"h2 row k={r.get('k')}: equivalence lacks a query or cluster result")
            w = lambda d: "equivalent" if d["equivalent"] else "not established"
            eqs = "–" if not eq else f"{w(eq['query'])} / {w(eq['cluster'])}"
            L.append(f"    {r['deg']:+7.1f} {p(r['r_cyl']):>8} % {p(r['r_ctrl']):>8} % {p(r['d']):>8} % {ci(r['lo'], r['hi']):>18} {ci(r['c_lo'], r['c_hi']):>18}   {eqs}")
        L.append("")
    L.append("## 5. Delay-flip audit and decomposition\n")
    cols = cfg["preregistered_spectral_cols"]
    L.append(f"{st['delay_flips_entity']}" + (f"; at most {st['delay_flips_max']}" if st.get("delay_flips_max") else "") + " (producer field delay_flips):\n")
    L.append("    k" + " " * 56 + " ".join(f"{k:>4}" for k in cols))
    for l in labels:
        L.append(f"    {NAME[l]:56s} " + " ".join(f"{st['delay_flips'][l][str(k)]:>4}" for k in cols))
    L.append("")
    for l, d in (st.get("decomposition") or {}).items():
        stages = st["decomposition_stages"]
        L.append(f"Decomposition ({NAME[l]}, k = {d['k']}, {d['n_batches']} batch): " + ", ".join(f"{stages[k]} {d[k]:.2e}" for k in ("tokens_rel_change", "pooled_rel_change", "coord_rel_change", "logspec_rel_change")) + f". {st['decomposition_note']}\n")
    L.append("## 6. Selected evaluation settings (from each run's metrics meta)\n")
    L.append("    run                                                       backbone     checkpoint                              n    batch  canonical  tf32   gl_seed  elapsed (min)")
    for l in labels:
        m = st["meta"][l]
        L.append(f"    {NAME[l]:56s}  {m['backbone']:11s}  {m['checkpoint']:38s}  {m['n_samples']:5d}  {m['batch_size']:5d}  {str(m['batch_canonical']):9s}  {str(m['tf32']):5s}  {m['gl_seed']:7d}  {m['elapsed_min']:12.1f}")
    L.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(L))
    print("wrote", out_path, "| DRAFT" if draft else "| FINAL")


if __name__ == "__main__":
    main()
