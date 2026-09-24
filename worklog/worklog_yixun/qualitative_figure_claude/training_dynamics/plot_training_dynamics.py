"""K = 8 training dynamics of xRIR (SimpleViT) and xRIR (CylindricalViT) from exp_01's per-epoch
evaluations (unseen split, 6 337 queries, epochs 1-12), in the style of the FLAC-side
checkpoint-curve figure.  x-axis: optimizer steps (k) = epoch x 296 334 / 64 (batch 32 x 2 accum).

Source JSONs: ckpt/xRIR_{simple,cyl}_8_shot/eval_unseen_epochNN.json (mean over all queries;
exp_01 protocol: per-model random references, unseeded Griffin-Lim, first 8000 samples)."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TRAIN_SAMPLES, EFFECTIVE_BATCH = 296334, 64
STEPS_PER_EPOCH = TRAIN_SAMPLES / EFFECTIVE_BATCH
ARMS = [("xRIR (SimpleViT), $K=8$", "simple", "#0072B2", "o", "--"),
        ("xRIR (CylindricalViT), $K=8$", "cyl", "#D55E00", "s", "-")]
PANELS = [("t60_error_pct", "T60 error (%) $\\downarrow$", 1.0),
          ("c50_error_db", "C50 error (dB) $\\downarrow$", 1.0),
          ("edt_error_s", "EDT error (ms) $\\downarrow$", 1000.0)]


def load(arm):
    rows = []
    for epoch in range(1, 13):
        path = os.path.join(REPO, "ckpt", "xRIR_%s_8_shot" % arm, "eval_unseen_epoch%02d.json" % epoch)
        j = json.load(open(path))
        assert j["num_shot"] == 8 and j["split"] == "unseen" and j["n_samples"] == 6337, path
        rows.append({"epoch": epoch, "steps": epoch * STEPS_PER_EPOCH, "n": j["n_samples"], "source": os.path.relpath(path, REPO),
                     **{k: j[k]["mean"] for k, _, _ in PANELS}, "stft_log_mse": j["stft_log_mse"]["mean"]})
    return rows


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix"})
    data = {arm: load(arm) for _, arm, _, _, _ in ARMS}
    with open(os.path.join(HERE, "training_dynamics_k8.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "epoch", "optimizer_steps", "t60_error_pct", "c50_error_db", "edt_error_ms", "stft_log_mse", "n_queries", "source"])
        for label, arm, _, _, _ in ARMS:
            for r in data[arm]:
                w.writerow([arm, r["epoch"], round(r["steps"]), "%.6g" % r["t60_error_pct"], "%.6g" % r["c50_error_db"],
                            "%.6g" % (r["edt_error_s"] * 1000), "%.6g" % r["stft_log_mse"], r["n"], r["source"]])
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
    for ax, (key, title, scale) in zip(axes, PANELS):
        for label, arm, color, marker, ls in ARMS:
            ax.plot([r["steps"] / 1000 for r in data[arm]], [r[key] * scale for r in data[arm]], color=color, marker=marker, ls=ls,
                    lw=1.8, ms=6, mfc=("white" if marker == "o" else color), mew=1.6, label=label)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Training steps (k)", fontsize=11)
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.set_xlim(0, 58)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "xrir_training_dynamics_k8.%s" % ext), dpi=300, bbox_inches="tight")
    print("wrote", HERE)


if __name__ == "__main__":
    main()
