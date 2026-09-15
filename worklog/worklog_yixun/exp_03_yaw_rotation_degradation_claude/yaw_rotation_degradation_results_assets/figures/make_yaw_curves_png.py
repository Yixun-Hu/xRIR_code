"""Planner figure generator (2026-09-14): PNG curves of relative degradation r_k vs yaw angle,
read ONLY from the canonical exp_03 producer JSON ckpt/yaw_rotation/full_stats.json (condition P).
Usage: python make_yaw_curves_png.py --stats ckpt/yaw_rotation/full_stats.json --out-dir <dir>"""
import argparse, json, hashlib, os, subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"control": "SimpleViT control (exp_01, epoch 12)", "cyl": "CylindricalViT (exp_01, epoch 12)",
          "released": "released xRIR checkpoint"}
COLORS = {"control": "#1f77b4", "cyl": "#d62728", "released": "#7f7f7f"}
NAMES = {"edt": "EDT", "c50": "C50", "t60": "T60", "loss": "test loss (STFT L1 + decay)",
         "log_mse": "log-STFT MSE", "consistency": "consistency |logspec(k) - logspec(0)|"}


def rows(stats, section, label, metric, cond="P"):
    r = sorted([x for x in stats[section][label][cond][metric] if x.get("r") is not None], key=lambda x: x["deg"])
    pct = lambda v: 100 * v if v is not None else float("nan")
    return ([x["deg"] for x in r], [pct(x["r"]) for x in r], [pct(x.get("lo")) for x in r], [pct(x.get("hi")) for x in r])


def panel(ax, stats, section, metric, labels):
    absolute = metric == "consistency"  # consistency is 0 at k = 0 by definition, so r_k is undefined: plot the absolute mean
    for lab in labels:
        if lab not in stats[section]:
            continue
        if absolute:
            rr = sorted(stats[section][lab]["P"][metric], key=lambda x: x["deg"])
            ax.plot([x["deg"] for x in rr], [x["meank"] for x in rr], "-o", ms=3.5, lw=1.4, color=COLORS[lab], label=LABELS[lab])
            continue
        deg, r, lo, hi = rows(stats, section, lab, metric)
        ax.plot(deg, r, "-o", ms=3.5, lw=1.4, color=COLORS[lab], label=LABELS[lab])
        ax.fill_between(deg, lo, hi, color=COLORS[lab], alpha=0.15, lw=0)
    ax.axhline(0, color="k", lw=0.6)
    if not absolute:
        ax.axhline(10, color="k", lw=0.6, ls="--")
    ax.set_xlabel("yaw rotation of the whole scene (degrees)")
    ax.set_ylabel("mean |logspec(k) - logspec(0)| (absolute)" if absolute else "relative degradation r_k (%)")
    ax.set_title(NAMES[metric]); ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180]); ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stats", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--labels", nargs="*", default=["control", "cyl", "released"]); a = ap.parse_args()
    stats = json.load(open(a.stats)); os.makedirs(a.out_dir, exist_ok=True)
    digest = hashlib.sha256(open(a.stats, "rb").read()).hexdigest()[:16]
    head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    foot = "exp_03 canonical full_stats.json sha256 %s..., condition P, %d queries / %d rooms, %d-resample paired bootstrap, %s intervals, git %s" % (
        digest, stats["n_queries"], stats["n_rooms"], stats["config"]["n_boot"], "Bonferroni-adjusted (family %d)" % stats["config"]["family_size"], head)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, m in zip(axes, ("edt", "c50", "t60")):
        panel(ax, stats, "acoustic", m, a.labels)
    axes[0].legend(fontsize=8, loc="upper left"); fig.suptitle("Acoustic metrics vs yaw rotation (dashed = +10 % threshold of H1)")
    fig.text(0.01, 0.005, foot, fontsize=6.5, color="#555"); fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(a.out_dir, "yaw_acoustic_vs_angle.png"), dpi=200); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, m in zip(axes, ("loss", "log_mse", "consistency")):
        panel(ax, stats, "spectral", m, a.labels)
    axes[0].legend(fontsize=8, loc="upper left"); fig.suptitle("Spectral metrics vs yaw rotation (18 angles, condition P)")
    fig.text(0.01, 0.005, foot, fontsize=6.5, color="#555"); fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(os.path.join(a.out_dir, "yaw_spectral_vs_angle.png"), dpi=200); plt.close(fig)
    # the direct comparison: D_k = r_k(cyl) - r_k(control) for EDT/C50 from the H2 block if present
    h2 = stats.get("h2") or {}
    cells = [(m, h2["rows"][m]) for m in ("edt", "c50") if isinstance(h2.get("rows"), dict) and m in h2["rows"]]
    if cells:
        fig, axes = plt.subplots(1, len(cells), figsize=(5 * len(cells), 4.2), squeeze=False)
        for ax, (metric, rr) in zip(axes[0], cells):
            rr = sorted([x for x in rr if "deg" in x and "d" in x], key=lambda x: x["deg"])
            if not rr:
                continue
            deg = [x["deg"] for x in rr]; d = [100 * x["d"] for x in rr]
            lo = [100 * x.get("lo", x["d"]) for x in rr]; hi = [100 * x.get("hi", x["d"]) for x in rr]
            ax.errorbar(deg, d, yerr=[[v - l for v, l in zip(d, lo)], [h - v for v, h in zip(d, hi)]], fmt="o", color="#d62728", capsize=3)
            ax.axhline(0, color="k", lw=0.6); ax.set_title("H2: D_k = r_k(cyl) - r_k(control), " + NAMES[metric])
            ax.set_xlabel("yaw (degrees)"); ax.set_ylabel("D_k (percentage points)"); ax.grid(alpha=0.3)
        fig.text(0.01, 0.005, foot, fontsize=6.5, color="#555"); fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.savefig(os.path.join(a.out_dir, "yaw_h2_cyl_minus_control.png"), dpi=200); plt.close(fig)
    print("wrote", sorted(os.listdir(a.out_dir)))


if __name__ == "__main__":
    main()
