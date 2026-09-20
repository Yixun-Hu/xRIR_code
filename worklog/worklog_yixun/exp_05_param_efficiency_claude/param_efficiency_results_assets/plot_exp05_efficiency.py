"""Draw a three-panel exp_05 capacity figure from admitted canonical results.

Run from the repository root with ``python -m tools.plot_exp05_efficiency``.
This presentation export is separate from the original bound record figures.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator, StrMethodFormatter

from tools import exp05_record as record


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_paper_figures"
METRICS = (
    ("EDT", "Early Decay Time", "EDT error (ms)", 1000.0),
    ("C50", "Clarity", "C50 error (dB)", 1.0),
    ("T60", "Reverberation Time", "T60 error (%)", 1.0),
)
STYLES = {
    "simple": dict(label="xRIR / SimpleViT", color="#1f77b4", marker="s", linestyle="--"),
    "cylindrical": dict(label="CylindricalViT", color="#008000", marker="o", linestyle="-"),
}


def draw(data):
    """Plot exact parameter counts and stored means/SDs, converting EDT s to ms."""
    arms = {arm["role"]: arm for arm in data["profile"]["arms"]}
    exported = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.15))
    fig.subplots_adjust(left=0.063, right=0.987, bottom=0.22, top=0.85, wspace=0.30)
    for axis, (metric, title, ylabel, scale) in zip(axes, METRICS):
        lo, hi = float("inf"), -float("inf")
        for backbone, style in STYLES.items():
            points = sorted(
                (point for point in data["curves"] if point["metric"] == metric
                 and arms[point["arm"]]["backbone"] == backbone),
                key=lambda point: point["encoder"],
            )
            x = [point["encoder"] / 1e6 for point in points]
            y = [point["mean"] * scale for point in points]
            sd = [point["sd"] * scale for point in points]
            axis.errorbar(
                x, y, yerr=sd, **style, linewidth=2.0, markersize=8,
                markeredgecolor="white", markeredgewidth=0.65,
                elinewidth=1.0, capsize=3.5, capthick=1.0, zorder=3,
            )
            lo = min(lo, min(mean - spread for mean, spread in zip(y, sd)))
            hi = max(hi, max(mean + spread for mean, spread in zip(y, sd)))
            for point, mean, spread in zip(points, y, sd):
                exported.append(dict(
                    arm=point["arm"], metric=metric,
                    encoder_parameters=point["encoder"], full_parameters=point["full"],
                    mean=mean, evaluation_seed_sd=spread,
                    unit="ms" if metric == "EDT" else "dB" if metric == "C50" else "%",
                ))
        span = hi - lo
        # Headroom for the legend, with every mean +/- SD fully visible.
        axis.set_ylim(lo - span * 0.20, hi + span * 0.48)
        axis.set_xscale("log")
        axis.set_xlim(2.25, 56)
        axis.xaxis.set_major_locator(FixedLocator([2.77, 19.73, 43.74]))
        axis.set_xticklabels(["S\n2.8", "M\n19.7", "L\n43.7"])
        axis.xaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}" if metric == "EDT" else "{x:.2f}"))
        axis.tick_params(axis="both", direction="out", length=4, width=1.0, labelsize=13)
        axis.tick_params(axis="x", pad=7)
        axis.set_title(title, fontsize=18, pad=12)
        axis.set_xlabel("Encoder parameters (M, log scale)", fontsize=14, labelpad=8)
        axis.set_ylabel(ylabel + r" $\downarrow$", fontsize=15, labelpad=9)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#333333")
    axes[1].legend(loc="upper left", fontsize=12.5, frameon=True,
                   fancybox=False, edgecolor="#dddddd", framealpha=1.0,
                   handlelength=2.8, borderpad=0.55)
    fig.suptitle(
        "Parameter Efficiency  |  AcousticRooms unseen  |  "
        + rf"$K_{{\mathrm{{ctx}}}} = {data['profile']['num_shot']}$",
        fontsize=14, y=0.985,
    )
    fig.text(
        0.5, 0.025,
        "Lower is better.  Error bars: ±1 SD over 5 evaluation seeds; "
        "one training run per model.  T60 is descriptive.",
        ha="center", va="bottom", fontsize=11, color="#444444",
    )
    return fig, exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, choices=(1, 8), default=8)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()
    profile = f"CURVE_K{args.shots}"
    source = ROOT / "ckpt/exp05/results" / f"{profile}.json"
    data, receipt = record.load(source, profile)
    record.validate(data, profile)
    outdir = args.outdir.resolve()
    record.refuse(outdir != source.parent.resolve(), "keep figure exports separate from canonical inputs")
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"parameter_efficiency_K{args.shots}"
    with plt.rc_context({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "exp05-efficiency",
    }):
        fig, rows = draw(data)
        try:
            for suffix in ("png", "pdf", "svg"):
                metadata = {"Date": None} if suffix == "svg" else {"CreationDate": None, "ModDate": None}
                fig.savefig(outdir / f"{stem}.{suffix}", dpi=300, facecolor="white", metadata=metadata)
        finally:
            plt.close(fig)
    with (outdir / f"{stem}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs = {
        f"{stem}.{suffix}": hashlib.sha256((outdir / f"{stem}.{suffix}").read_bytes()).hexdigest()
        for suffix in ("png", "pdf", "svg", "csv")
    }
    identity = dict(
        source=receipt, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matplotlib_version=matplotlib.__version__, outputs=outputs,
        uncertainty="Stored sample SD across evaluation seeds 42-46; not a training-seed CI.",
        model="Original CylindricalViT, not the oriented variant.",
        scope="Presentation export; not part of the historical binding report.",
    )
    (outdir / f"{stem}.source.json").write_text(json.dumps(identity, indent=2) + "\n")
    for name in outputs:
        print(outdir / name)


if __name__ == "__main__":
    main()
