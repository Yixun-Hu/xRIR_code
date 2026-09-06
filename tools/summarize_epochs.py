"""Final summary of the backbone comparison across training epochs (full unseen split).

Reads ckpt/xRIR_{cyl,simple}_8_shot/per_sample_unseen_epochNN.json and reports:
  1. per-epoch means for both models,
  2. the paired difference averaged over --avg-epochs (per-sample diff averaged across
     epochs first, then a bootstrap over the 6337 samples),
  3. the paired comparison of each model's best-test-loss epoch.

    python tools/summarize_epochs.py --avg-epochs 5-12
"""
import argparse
import json
import os

import numpy as np

METRICS = [("edt", "EDT (s)", 4), ("c50", "C50 (dB)", 3), ("t60", "T60 (%)", 2), ("stft_mse", "log-STFT MSE", 4), ("loss", "test loss", 5)]
PAPER = {"edt": 0.055, "c50": 1.457, "t60": None}
RELEASED = {"edt": 0.0549, "c50": 1.358, "t60": 9.69}


def load(run, ep):
    f = f"ckpt/xRIR_{run}_8_shot/per_sample_unseen_epoch{ep:02d}.json"
    return json.load(open(f)) if os.path.exists(f) else None


def boot_ci(d, n_boot, rng, clusters=None):
    """Percentile bootstrap CI of the mean of d. With ``clusters`` (room id per sample) the resampling unit is
    the room (17 held-out rooms), which is the honest unit for inference about unseen rooms; without it the
    unit is the query, which is inference about this fixed test split only."""
    if clusters is None:
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    else:
        uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
        groups = [np.where(inv == g)[0] for g in range(len(uniq))]
        boots = np.array([np.concatenate([d[groups[i]] for i in rng.integers(0, len(uniq), len(uniq))]).mean() for _ in range(n_boot)])
    return np.percentile(boots, [2.5, 97.5])


def rooms_of(per_sample):
    return np.array([p.split("/")[1] for p in per_sample["ir_path"]])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--avg-epochs", default="5-12")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--json", default=None, help="write every displayed number (means, both CIs, config) here; results pages consume this")
    p.add_argument("--summary", default=None, help="write the printed summary here; its sha256 is stored in --json so pages can bind the two")
    args = p.parse_args()
    import contextlib, hashlib, io, sys
    buf = io.StringIO()
    class Tee:
        def write(self, t): buf.write(t); sys.__stdout__.write(t)
        def flush(self): sys.__stdout__.flush()
    sys.stdout = Tee()
    baseline = json.load(open("ckpt/baseline_reproduction.json")) if os.path.exists("ckpt/baseline_reproduction.json") else None
    out = {"n_boot": args.n_boot, "boot_seed": 0, "avg_epochs": args.avg_epochs, "per_epoch": {}, "avg": {}, "best": {}, "paper": PAPER, "released": RELEASED,
           "baseline_reproduction": baseline,
           "note": "references and Griffin-Lim phases were NOT identical across models in exp_01 (see the exp_01 record); "
                   "CIs: query-level = this fixed split; room-cluster = 17 held-out rooms; nominal 95%, no multiplicity adjustment"}
    lo_ep, hi_ep = map(int, args.avg_epochs.split("-"))
    rng = np.random.default_rng(0)
    hist = {r: [json.loads(l) for l in open(f"ckpt/xRIR_{r}_8_shot/history.jsonl")] for r in ["cyl", "simple"]}
    epochs = [e for e in range(1, 13) if load("cyl", e) and load("simple", e)]
    out["history"] = {r: [{"epoch": h["epoch"], "train_loss": h["train_loss"], "test_loss": h["test_loss"], "lr": h["lr"]} for h in hist[r]] for r in hist}
    out["n_rooms"] = int(len(np.unique(rooms_of(load("cyl", epochs[0]))))); out["n_queries"] = int(len(load("cyl", epochs[0])["index"]))

    print("1. Per-epoch means, cylindrical / control (full unseen split, n=6337; queries paired, references NOT identical across models, see the exp_01 record)")
    print(f"{'epoch':>5} {'lr':>8} | " + " | ".join(f"{n:>21s}" for _, n, _ in METRICS))
    for e in epochs:
        A, B = load("cyl", e), load("simple", e)
        assert A["index"] == B["index"] and A["ir_path"] == B["ir_path"], f"epoch {e}: per-sample files are not query-aligned"
        assert A["meta"]["split"] == B["meta"]["split"] == "unseen" and A["meta"]["num_shot"] == B["meta"]["num_shot"] == 8, f"epoch {e}: meta mismatch"
        cells = []
        out["per_epoch"][e] = {"lr": hist["cyl"][e - 1]["lr"]}
        for k, _, nd in METRICS:
            a, b = np.asarray(A[k], float), np.asarray(B[k], float); ok = np.isfinite(a) & np.isfinite(b)
            d = a[ok] - b[ok]; q = boot_ci(d, args.n_boot, rng); c = boot_ci(d, args.n_boot, rng, rooms_of(A)[ok])
            out["per_epoch"][e][k] = {"cyl": float(a[ok].mean()), "simple": float(b[ok].mean()), "diff": float(d.mean()), "q_lo": float(q[0]), "q_hi": float(q[1]),
                                      "r_lo": float(c[0]), "r_hi": float(c[1]), "n_valid": int(ok.sum())}
            cells.append(f"{a[ok].mean():.{nd}f} / {b[ok].mean():.{nd}f}".rjust(21))
        print(f"{e:>5} {hist['cyl'][e-1]['lr']:8.1e} | " + " | ".join(cells))
    print(f"{'paper':>5} {'':>8} | " + " | ".join(f"{(str(PAPER.get(k)) if PAPER.get(k) else '-'):>21s}" for k, _, _ in METRICS))
    print(f"{'relsd':>5} {'':>8} | " + " | ".join(f"{(str(RELEASED.get(k)) if RELEASED.get(k) else '-'):>21s}" for k, _, _ in METRICS))

    avg = [e for e in epochs if lo_ep <= e <= hi_ep]
    print(f"\n2. Paired difference cyl - control averaged over epochs {avg[0]}-{avg[-1]} ({len(avg)} checkpoint pairs); "
          "per-sample differences averaged across epochs, then bootstrapped. * = nominal 95% CI excludes 0 (no multiplicity adjustment).")
    print(f"{'metric':14s} {'cyl':>9s} {'ctrl':>9s} {'diff':>9s} {'rel %':>7s} {'query-level 95% CI':>24s} {'room-cluster 95% CI':>24s} {'epochs cyl lower':>17s}")
    rooms = rooms_of(load("cyl", avg[0]))
    for k, name, nd in METRICS:
        A = np.stack([np.asarray(load("cyl", e)[k], float) for e in avg]); B = np.stack([np.asarray(load("simple", e)[k], float) for e in avg])
        ok = np.isfinite(A).all(0) & np.isfinite(B).all(0)
        a, b = A[:, ok].mean(0), B[:, ok].mean(0)
        d = a - b; lo, hi = boot_ci(d, args.n_boot, rng); sig = "*" if lo > 0 or hi < 0 else " "
        rlo, rhi = boot_ci(d, args.n_boot, rng, rooms[ok]); rsig = "*" if rlo > 0 or rhi < 0 else " "
        wins = int(sum(np.nanmean(A[i]) < np.nanmean(B[i]) for i in range(len(avg))))
        per_room = {r: float((a - b)[rooms[ok] == r].mean()) for r in np.unique(rooms)}
        out["avg"][k] = {"cyl": float(a.mean()), "simple": float(b.mean()), "diff": float(d.mean()), "rel_pct": float(100 * d.mean() / b.mean()),
                         "q_lo": float(lo), "q_hi": float(hi), "r_lo": float(rlo), "r_hi": float(rhi), "wins": wins, "n_epochs": len(avg), "n_valid": int(ok.sum()), "per_room_diff": per_room}
        print(f"{name:14s} {a.mean():9.{nd}f} {b.mean():9.{nd}f} {d.mean():+9.{nd}f} {100*d.mean()/b.mean():+7.1f} [{lo:+9.{nd}f}, {hi:+9.{nd}f}]{sig} [{rlo:+9.{nd}f}, {rhi:+9.{nd}f}]{rsig} {wins:>8d}/{len(avg)}")

    best = {r: min(hist[r], key=lambda h: h["test_loss"])["epoch"] for r in ["cyl", "simple"]}
    print(f"\n3. Best-test-loss checkpoints: cylindrical epoch {best['cyl']} vs control epoch {best['simple']} (paired)")
    A, B = load("cyl", best["cyl"]), load("simple", best["simple"])
    print(f"{'metric':14s} {'cyl':>9s} {'ctrl':>9s} {'diff':>9s} {'rel %':>7s} {'query-level 95% CI':>24s} {'room-cluster 95% CI':>24s}")
    out["best"]["epochs"] = best
    rooms = rooms_of(A)
    for k, name, nd in METRICS:
        a, b = np.asarray(A[k], float), np.asarray(B[k], float); ok = np.isfinite(a) & np.isfinite(b)
        d = a[ok] - b[ok]; lo, hi = boot_ci(d, args.n_boot, rng); sig = "*" if lo > 0 or hi < 0 else " "
        rlo, rhi = boot_ci(d, args.n_boot, rng, rooms[ok]); rsig = "*" if rlo > 0 or rhi < 0 else " "
        out["best"][k] = {"cyl": float(a[ok].mean()), "simple": float(b[ok].mean()), "diff": float(d.mean()), "rel_pct": float(100 * d.mean() / b[ok].mean()),
                          "q_lo": float(lo), "q_hi": float(hi), "r_lo": float(rlo), "r_hi": float(rhi), "n_valid": int(ok.sum())}
        print(f"{name:14s} {a[ok].mean():9.{nd}f} {b[ok].mean():9.{nd}f} {d.mean():+9.{nd}f} {100*d.mean()/b[ok].mean():+7.1f} [{lo:+9.{nd}f}, {hi:+9.{nd}f}]{sig} [{rlo:+9.{nd}f}, {rhi:+9.{nd}f}]{rsig}")
    sys.stdout = sys.__stdout__
    text = buf.getvalue()
    if args.summary:
        with open(args.summary, "w") as f:
            f.write(text)
        out["summary_path"] = args.summary
    out["summary_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
