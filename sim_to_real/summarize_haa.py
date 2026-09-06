"""Summarize sim-to-real results under ckpt/sim2real into a paper-Table-2-style table.

Rows: each init (zero-shot and fine-tuned), mean +- std over seeds; columns: room x
(EDT s, C50 dB, T60 %). Then paired cylindrical-vs-control statistics per room, pooling
per-sample differences over seeds (pairs share (seed, sample) and identical references).

    python sim_to_real/summarize_haa.py --root ckpt/sim2real
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np

ROOMS = ["class_room", "dampened_room", "hallway", "complex_room"]
PAPER = {  # xRIR (K=8), Table 2
    "class_room": (0.093, 1.628, 6.25), "dampened_room": (0.044, 3.302, None),
    "hallway": (0.062, 0.954, 3.20), "complex_room": (0.077, 1.688, 4.33)}  # complex T60 4.33 per the CVPR paper PDF (the markdown copy truncated it)
METRICS = [("edt", "EDT (s)", 4), ("c50", "C50 (dB)", 3), ("t60", "T60 (%)", 2)]


def load_runs(root):
    runs = {}  # (init, kind) -> list of {room: per_sample dict}
    for d in sorted(glob.glob(os.path.join(root, "*", "*"))):
        init, sub = d.split(os.sep)[-2:]
        if sub == "zeroshot":
            kind, evald = "zero-shot", d
        elif sub.startswith("seed") and os.path.isdir(os.path.join(d, "eval")):
            kind, evald = "fine-tuned", os.path.join(d, "eval")
        else:
            continue
        per = {}
        for room in ROOMS:
            f = os.path.join(evald, f"per_sample_{room}.json")
            if os.path.exists(f):
                per[room] = json.load(open(f))
        if per:
            runs.setdefault((init, kind), []).append({"dir": d, "per": per})
    return runs


def fmt(vals, nd):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return "-".rjust(13)
    return (f"{np.mean(vals):.{nd}f}" + (f"±{np.std(vals):.{nd}f}" if len(vals) > 1 else "")).rjust(13)


BOOT_SEED = 0


def paired(a, b, n_boot=10000, seed=BOOT_SEED, clusters=None, seeds=None):
    """Paired mean difference a-b with percentile bootstrap CIs.

    clusters: one query id per row. Rows of the same query (repeated across training seeds) are resampled
              together ("query-cluster" bootstrap: inference conditional on the realised training seeds).
    seeds:    one training-seed id per row. When given together with clusters, a second, two-way bootstrap
              resamples queries AND seeds with replacement (weights = query multiplicity x seed multiplicity),
              so the interval also reflects seed-to-seed variation of the fine-tuning outcome.
    Returns (mean, lo, hi, star, frac_a_lower, extra) with extra = {"two_way": (lo2, hi2)} when seeds is given.
    Inputs must be finite and non-empty (callers filter with np.isfinite); ValueError otherwise.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or not (np.isfinite(a).all() and np.isfinite(b).all()) or a.shape != b.shape:
        raise ValueError("paired() needs equal-length, non-empty, finite arrays")
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    extra = {}
    if clusters is None:
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    else:
        uq, qi = np.unique(np.asarray(clusters), return_inverse=True)
        boots = np.empty(n_boot)
        for t in range(n_boot):
            wq = np.bincount(rng.integers(0, len(uq), len(uq)), minlength=len(uq))[qi]
            boots[t] = np.average(d, weights=wq) if wq.sum() else np.nan
        if seeds is not None:
            us, si = np.unique(np.asarray(seeds), return_inverse=True)
            b2 = np.empty(n_boot)
            for t in range(n_boot):
                wq = np.bincount(rng.integers(0, len(uq), len(uq)), minlength=len(uq))[qi]
                ws = np.bincount(rng.integers(0, len(us), len(us)), minlength=len(us))[si]
                w = wq * ws
                b2[t] = np.average(d, weights=w) if w.sum() else np.nan
            extra["two_way"] = tuple(np.nanpercentile(b2, [2.5, 97.5]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return d.mean(), lo, hi, ("*" if lo > 0 or hi < 0 else " "), float(np.mean(np.asarray(a) < np.asarray(b))), extra


EXPECTED_RUNS = {  # exact run identities of the approved plan
    ("released", "fine-tuned"): ["seed0", "seed1", "seed2"], ("control", "fine-tuned"): ["seed0", "seed1", "seed2"],
    ("cyl", "fine-tuned"): ["seed0", "seed1", "seed2"], ("released_repomaps", "fine-tuned"): ["seed0"],
    ("released", "zero-shot"): ["zeroshot"], ("control", "zero-shot"): ["zeroshot"], ("cyl", "zero-shot"): ["zeroshot"]}
EXPECTED_N = {"class_room": 463, "dampened_room": 198, "hallway": 423, "complex_room": 198}  # DiffRIR test splits
REQUIRED_METRICS = {room: ["edt", "c50"] + ([] if room == "dampened_room" else ["t60"]) for room in ROOMS}
ALL_METRIC_KEYS = ["edt", "c50", "t60", "stft_mse", "loss", "env"]
PROTOCOL = {"num_shot": 8, "eval_seed": 0, "split": "test"}
BACKBONE_OF = {"released": "simple", "released_repomaps": "simple", "control": "simple", "cyl": "cylindrical"}
INIT_OF = {"released": "checkpoints/xRIR_unseen.pth", "released_repomaps": "checkpoints/xRIR_unseen.pth",
           "control": "ckpt/xRIR_simple_8_shot/epoch_12.pth", "cyl": "ckpt/xRIR_cyl_8_shot/epoch_12.pth"}
DEPTH_VARIANT_OF = {"released_repomaps": "repo"}  # everything else: "default"
HAA_ROOT = os.environ.get("HAA_XRIR_ROOT", os.path.expanduser("~/data_cache/HAA_xrir"))


def _test_indices(room):
    return sorted(int(i) for i in json.load(open(os.path.join(HAA_ROOT, room, "meta.json")))["test"])


def completeness(runs):
    """Strict gate against the approved run set. Returns (complete, problems). A run counts only if: its seed id is
    expected; every room is present with exactly the DiffRIR test indices (unique, in order) and every metric/ir_path
    array has that length; per-sample meta matches the protocol (K=8, eval_seed 0, split test, backbone of the init);
    every required metric has >= 1 finite value; for fine-tuned runs the stage args.json files carry the directory's
    seed, the planned init checkpoint and the planned depth variant. Unexpected runs/groups are reported and cyl and
    control must have identical seed sets."""
    problems = []
    seen_keys = set(runs)
    for key, exp_seeds in EXPECTED_RUNS.items():
        init, kind = key
        have = {os.path.basename(r["dir"]): r for r in runs.get(key, [])}
        for sd in exp_seeds:
            r = have.get(sd)
            tag = f"{init}/{kind}/{sd}"
            if r is None:
                problems.append(f"{tag}: missing"); continue
            for room in ROOMS:
                per = r["per"].get(room)
                if per is None:
                    problems.append(f"{tag}: room {room} missing"); continue
                idx = [int(i) for i in per["index"]]
                if idx != _test_indices(room):
                    problems.append(f"{tag}: {room} indices are not exactly the DiffRIR test split ({len(idx)} vs {EXPECTED_N[room]}, unique {len(set(idx))})")
                for arr in ["ir_path"] + ALL_METRIC_KEYS:
                    if len(per.get(arr, [])) != len(idx):
                        problems.append(f"{tag}: {room} array {arr} has length {len(per.get(arr, []))}, expected {len(idx)}")
                if any(not p.startswith(f"{room}/") for p in per.get("ir_path", [])):
                    problems.append(f"{tag}: {room} ir_path entries do not all belong to the room")
                meta = per.get("meta", {})
                for k, v in PROTOCOL.items():
                    if meta.get(k) != v:
                        problems.append(f"{tag}: {room} meta {k}={meta.get(k)!r}, expected {v!r}")
                if meta.get("backbone") != BACKBONE_OF[init]:
                    problems.append(f"{tag}: {room} backbone {meta.get('backbone')!r}, expected {BACKBONE_OF[init]!r}")
                for m in REQUIRED_METRICS[room]:
                    if not np.isfinite(np.asarray(per.get(m, []), float)).any():
                        problems.append(f"{tag}: {room} has no valid {m}")
            if kind == "fine-tuned":
                want_seed = int(sd[len("seed"):]); want_dv = DEPTH_VARIANT_OF.get(init, "default")
                for stage in ["stage1"] + [f"stage2_{room}" for room in ROOMS]:
                    af = os.path.join(r["dir"], stage, "args.json")
                    if not os.path.exists(af):
                        problems.append(f"{tag}: {stage}/args.json missing"); continue
                    a = json.load(open(af))
                    if a.get("seed") != want_seed:
                        problems.append(f"{tag}: {stage} trained with seed {a.get('seed')}, expected {want_seed}")
                    if "depth_variant" not in a or a["depth_variant"] != want_dv:
                        problems.append(f"{tag}: {stage} depth_variant {a.get('depth_variant')!r} (must be present), expected {want_dv!r}")
                    if a.get("backbone") != BACKBONE_OF[init]:
                        problems.append(f"{tag}: {stage} backbone {a.get('backbone')!r}")
                    want_init = INIT_OF[init] if stage == "stage1" else os.path.join(r["dir"], "stage1", "best.pth")
                    if os.path.normpath(a.get("init", "")) != os.path.normpath(want_init):
                        problems.append(f"{tag}: {stage} init {a.get('init')!r}, expected {want_init!r}")
            else:
                # zero-shot evals record no depth variant in this sweep (eval_haa.py was not changed mid-sweep); default variant by construction of run_haa_pipeline.sh
                pass
        extra = sorted(set(have) - set(exp_seeds))
        if extra:
            problems.append(f"{init}/{kind}: unexpected runs {extra}")
    for key in sorted(seen_keys - set(EXPECTED_RUNS)):
        problems.append(f"unexpected run group {key[0]}/{key[1]}")
    cyl = {os.path.basename(r["dir"]) for r in runs.get(("cyl", "fine-tuned"), [])}
    ctl = {os.path.basename(r["dir"]) for r in runs.get(("control", "fine-tuned"), [])}
    if cyl != ctl:
        problems.append(f"cyl seeds {sorted(cyl)} != control seeds {sorted(ctl)}")
    return (not problems), problems


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="ckpt/sim2real")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--json", default=None, help="also write every displayed number (means, CIs, counts, config) here")
    p.add_argument("--summary", default=None, help="write the printed summary here as well; its sha256 is stored in --json so pages can bind the two")
    p.add_argument("--allow-partial", action="store_true", help="summarise an incomplete run set (marked DRAFT)")
    args = p.parse_args()
    import contextlib, io, sys
    runs = load_runs(args.root)
    complete, missing = completeness(runs)
    if not complete and not args.allow_partial:
        raise SystemExit("incomplete run set (use --allow-partial for a draft): " + "; ".join(missing))
    buf = io.StringIO()
    class Tee:
        def write(self, t): buf.write(t); sys.__stdout__.write(t)
        def flush(self): sys.__stdout__.flush()
    sys.stdout = Tee()
    if not complete:
        print("DRAFT — incomplete run set: " + "; ".join(missing))
    out = {"draft": not complete, "missing": missing, "n_boot": args.n_boot, "boot_seed": BOOT_SEED,
           "bootstrap": {"query_cluster": "rows of one query across training seeds resampled together (conditional on the realised seeds)",
                         "two_way": "queries and training seeds both resampled with replacement"},
           "expected_runs": {f"{k[0]}|{k[1]}": v for k, v in EXPECTED_RUNS.items()}, "expected_n": EXPECTED_N,
           "paper": PAPER, "rows": {}, "paired": []}

    print(f"{'model':28s}" + "".join(f"| {r:39s}" for r in ROOMS))
    print(f"{'':28s}" + "".join("| " + "".join(f"{n:>13s}" for _, n, _ in METRICS) for _ in ROOMS))
    print(f"{'paper xRIR K=8':28s}" + "".join("| " + "".join(f"{(str(v) if v is not None else '-'):>13s}" for v in PAPER[r]) for r in ROOMS))
    order = ["released", "released_repomaps", "control", "cyl"]
    for init in order + sorted({k[0] for k in runs} - set(order)):
        for kind in ["zero-shot", "fine-tuned"]:
            if (init, kind) not in runs:
                continue
            rs = runs[(init, kind)]
            line = f"{init + ' ' + kind + f' (n={len(rs)})':28s}"
            for room in ROOMS:
                cells = []
                for key, _, nd in METRICS:
                    per_run = []
                    for r in rs:
                        if room in r["per"]:
                            v = np.asarray(r["per"][room][key], float); v = v[np.isfinite(v)]
                            per_run.append(float(v.mean()) if v.size else None)
                        else:
                            per_run.append(None)
                    cells.append(fmt(per_run, nd))
                    vals = [v for v in per_run if v is not None]
                    out["rows"][f"{init}|{kind}|{room}|{key}"] = {"per_run": per_run, "seeds": [os.path.basename(r["dir"]) for r in rs],
                                                                   "mean": float(np.mean(vals)) if vals else None, "std": float(np.std(vals)) if len(vals) > 1 else None}
                line += "| " + "".join(cells)
            print(line)

    if ("cyl", "fine-tuned") in runs and ("control", "fine-tuned") in runs:
        print("\nPaired cylindrical - control (fine-tuned), per-sample differences pooled over training seeds. Two intervals: query-cluster "
              "(conditional on the realised seeds) and two-way (queries and seeds resampled); * = nominal 95% CI excludes 0 (no multiplicity adjustment); negative favours cylindrical")
        cyl = {os.path.basename(r["dir"]): r for r in runs[("cyl", "fine-tuned")]}
        ctl = {os.path.basename(r["dir"]): r for r in runs[("control", "fine-tuned")]}
        seeds = sorted(set(cyl) & set(ctl))
        assert set(cyl) == set(ctl) or not complete, "matched seed sets are required for the final summary"
        print(f"{'room':14s} {'metric':10s} {'cyl':>9s} {'ctrl':>9s} {'diff':>9s} {'query-cluster CI':>22s} {'two-way CI':>22s} {'cyl<ctrl':>9s} {'valid':>6s}  (seeds {', '.join(seeds)})")
        for room in ROOMS:
            for key, name, nd in METRICS:
                a, b, cl, sd_ids, per_seed, per_seed_n = [], [], [], [], {}, {}
                for s in seeds:
                    if room not in cyl[s]["per"] or room not in ctl[s]["per"]:
                        continue
                    A, B = cyl[s]["per"][room], ctl[s]["per"][room]
                    assert A["index"] == B["index"] and A["ir_path"] == B["ir_path"], (room, s)
                    assert A["meta"]["eval_seed"] == B["meta"]["eval_seed"] and A["meta"]["num_shot"] == B["meta"]["num_shot"], (room, s)
                    assert A["meta"]["split"] == B["meta"]["split"] == "test", (room, s)
                    x, y = np.asarray(A[key], float), np.asarray(B[key], float)
                    okk = np.isfinite(x) & np.isfinite(y)
                    per_seed[s] = float((x[okk] - y[okk]).mean()) if okk.any() else None
                    per_seed_n[s] = int(okk.sum())
                    a.append(x); b.append(y); cl.append(np.asarray(A["index"])); sd_ids.append(np.full(len(x), s))
                if not a:
                    if key in REQUIRED_METRICS[room]:
                        raise SystemExit(f"required paired cell {room}/{key} has no runs")
                    continue
                a, b, cl, sd_ids = np.concatenate(a), np.concatenate(b), np.concatenate(cl), np.concatenate(sd_ids)
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() == 0:
                    if key in REQUIRED_METRICS[room]:
                        raise SystemExit(f"required paired cell {room}/{key} has zero valid pairs")
                    continue
                m, lo, hi, sig, frac, extra = paired(a[ok], b[ok], args.n_boot, clusters=cl[ok], seeds=sd_ids[ok])
                lo2, hi2 = extra["two_way"]; sig2 = "*" if (lo2 > 0 or hi2 < 0) else " "
                out["paired"].append({"room": room, "metric": key, "cyl": float(a[ok].mean()), "ctrl": float(b[ok].mean()), "diff": float(m),
                                      "lo": float(lo), "hi": float(hi), "lo_two_way": float(lo2), "hi_two_way": float(hi2), "frac_cyl_lower": float(frac),
                                      "n_valid": int(ok.sum()), "n_total": int(len(a)), "n_queries": int(len(np.unique(cl[ok]))), "n_seeds": int(len(np.unique(sd_ids[ok]))),
                                      "per_seed_diff": per_seed, "per_seed_valid": per_seed_n})
                print(f"{room:14s} {name:10s} {a[ok].mean():9.{nd}f} {b[ok].mean():9.{nd}f} {m:+9.{nd}f} [{lo:+9.{nd}f}, {hi:+9.{nd}f}]{sig} [{lo2:+9.{nd}f}, {hi2:+9.{nd}f}]{sig2} {frac:9.2f} {ok.sum():6d}  per-seed: "
                      + ", ".join(f"{s}:{v:+.{nd}f} (n={per_seed_n[s]})" for s, v in per_seed.items() if v is not None))


    sys.stdout = sys.__stdout__
    text = buf.getvalue()
    if args.summary:
        with open(args.summary, "w") as f:
            f.write(text)
        out["summary_path"] = args.summary
    out["summary_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    out["protocol"] = PROTOCOL; out["backbone_of"] = BACKBONE_OF; out["init_of"] = INIT_OF; out["depth_variant_of"] = {k: DEPTH_VARIANT_OF.get(k, "default") for k in BACKBONE_OF}
    out["required_metrics"] = REQUIRED_METRICS
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
