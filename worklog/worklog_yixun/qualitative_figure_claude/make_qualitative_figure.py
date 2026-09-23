"""Qualitative RIR comparison: xRIR (SimpleViT) vs xRIR (CylindricalViT), sim + real.

Selects, from the existing per-sample evaluations, the queries with the largest
cylindrical-over-baseline improvement, re-runs both models on exactly those queries
(same references, seeded Griffin-Lim) and renders waveform + log-STFT panels in the
layout of xRIR's Fig. 4 / DALL-E's Fig. 3 (columns = queries, rows = GT / baseline /
cylindrical).

Selection rule (identical for both environments; applied to the paired per-sample
metrics of the canonical evaluations):
  * candidates: the cylindrical error is lower on every available metric
    (EDT, C50, T60; T60 is skipped for HAA dampened_room as in the paper) and the
    cylindrical EDT error is at most the split's / room's median baseline EDT error
    (so the shown cylindrical prediction is itself a decent one, not merely less bad);
  * score = 0.5 * mean relative improvement + 0.5 * tanh(mean improvement in units
    of the median baseline error / 2);
  * sim: the best-scoring query of each room category, top N categories;
    real: the best-scoring test query of each HAA room (fine-tuning seed 0).

Sources: sim = exp_04's confirmatory K=8 seed-42 P/k=0 runs of exp_01's same-budget
SimpleViT control and CylindricalViT (epoch 12), unseen split; real = exp_02's seed-0
two-stage fine-tuned checkpoints, DiffRIR test split, eval_seed 0 references.

Runs on CPU (no GPU needed for a dozen forward passes); `model.xRIR.apply_delay` is
patched at runtime with a device-agnostic copy because the original hard-codes .cuda().
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("XRIR_DATA_PATH", os.path.expanduser("~/data_cache/AcousticRooms"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402

import model.xRIR as xrir_module  # noqa: E402
from eval_unseen import Evaluator  # noqa: E402
from eval_xRIR_backbone import load_model_state  # noqa: E402
from model.xRIR_cyl import build_xrir  # noqa: E402
from sim_to_real.haa_dataset import NO_T60_ROOMS, ROOMS, HAADataset  # noqa: E402
from tools.per_sample_metrics import acoustic_metrics, griffin_lim_seeded, sample_seed  # noqa: E402
from tools.reference_manifest import ManifestDataset, load_manifest  # noqa: E402

SR = 22050
SIM_RUNS = {"simple": "ckpt/yaw_aug/eval/control_k8_seed42_k0", "cylindrical": "ckpt/yaw_aug/eval/cyl_k8_seed42_k0"}
SIM_CKPT = {"simple": "ckpt/xRIR_simple_8_shot/epoch_12.pth", "cylindrical": "ckpt/xRIR_cyl_8_shot/epoch_12.pth"}
SIM_MANIFEST = "ckpt/yaw_aug/reference_manifest_k8_seed42.json"
SIM_GL_SEED = 42
HAA_ARM = {"simple": "control", "cylindrical": "cyl"}
HAA_SEED = "seed0"
HAA_EVAL_SEED = 0
LABEL = {"simple": "xRIR (SimpleViT)", "cylindrical": "xRIR (CylindricalViT)"}
COLOR = {"gt": "#4d4d4d", "simple": "#0072B2", "cylindrical": "#D55E00"}
PRETTY_ROOM = {"class_room": "Classroom", "dampened_room": "Dampened room", "hallway": "Hallway", "complex_room": "Complex room"}


def _apply_delay_any_device(signal, delay_tensor):
    delayed = torch.zeros_like(signal)
    for i in range(signal.shape[0]):
        d = int(delay_tensor[i].item())
        if d > 0:
            delayed[i, d:] = signal[i, :-d]
        elif d < 0:
            delayed[i, :d] = signal[i, -d:]
        else:
            delayed[i] = signal[i]
    return delayed


xrir_module.apply_delay = _apply_delay_any_device


# ----------------------------------------------------------------------------- selection
def score_pairs(ctrl, cyl, metrics):
    C = {m: np.asarray(ctrl[m], float) for m in metrics}
    Y = {m: np.asarray(cyl[m], float) for m in metrics}
    med = {m: float(np.nanmedian(C[m])) for m in metrics}
    n = len(C[metrics[0]])
    ok = np.ones(n, bool)
    for m in metrics:
        ok &= np.isfinite(C[m]) & np.isfinite(Y[m]) & (Y[m] < C[m])
    ok &= Y["edt"] <= med["edt"]
    rel = np.mean([(C[m] - Y[m]) / np.maximum(C[m], 1e-9) for m in metrics], axis=0)
    absg = np.mean([(C[m] - Y[m]) / med[m] for m in metrics], axis=0)
    score = np.where(ok, 0.5 * rel + 0.5 * np.tanh(absg / 2.0), -np.inf)
    return score, C, Y, med


def select_sim(n_cols):
    c = json.load(open(os.path.join(REPO, SIM_RUNS["simple"], "per_sample_yaw.json")))
    y = json.load(open(os.path.join(REPO, SIM_RUNS["cylindrical"], "per_sample_yaw.json")))
    assert c["query"] == y["query"], "unpaired per-sample files"
    metrics = ["edt", "c50", "t60"]
    score, C, Y, med = score_pairs(c["P"]["0"], y["P"]["0"], metrics)
    order = np.argsort(-score)
    chosen, seen = [], set()
    for i in order:
        if not np.isfinite(score[i]):
            break
        cat = c["query"][i].split("/")[0]
        if cat in seen:
            continue
        seen.add(cat)
        chosen.append({"env": "sim", "query": c["query"][i], "index": int(i), "score": float(score[i]),
                       "json_metrics": {a: {m: float(D[m][i]) for m in metrics} for a, D in (("simple", C), ("cylindrical", Y))}})
        if len(chosen) >= n_cols:
            break
    return chosen, med, int(np.isfinite(score).sum()), len(score)


def select_real():
    chosen, meds = [], {}
    for room in ROOMS:
        c = json.load(open(os.path.join(REPO, "ckpt/sim2real", HAA_ARM["simple"], HAA_SEED, "eval", "per_sample_%s.json" % room)))
        y = json.load(open(os.path.join(REPO, "ckpt/sim2real", HAA_ARM["cylindrical"], HAA_SEED, "eval", "per_sample_%s.json" % room)))
        assert c["ir_path"] == y["ir_path"], room
        metrics = ["edt", "c50"] + ([] if room in NO_T60_ROOMS else ["t60"])
        score, C, Y, med = score_pairs(c, y, metrics)
        i = int(np.argmax(score))
        assert np.isfinite(score[i]), room
        meds[room] = med
        chosen.append({"env": "real", "room": room, "idx": int(c["index"][i]), "query": c["ir_path"][i], "score": float(score[i]),
                       "n_candidates": int(np.isfinite(score).sum()), "n": int(len(score)),
                       "json_metrics": {a: {m: float(D[m][i]) for m in metrics} for a, D in (("simple", C), ("cylindrical", Y))}})
    return chosen, meds


# ----------------------------------------------------------------------------- inference
def load_model(backbone, ckpt):
    m = build_xrir(backbone, 8)
    m.load_state_dict(load_model_state(os.path.join(REPO, ckpt)), strict=True)
    return m.eval()


@torch.no_grad()
def predict(model, item, gl_seed_key, gl_base_seed, want_t60, window, evaluator):
    _, src, depth, tgt, refs, ref_locs = item[:6]
    out, tgt_spec = model(depth[None], refs[None], src[None], ref_locs[None], tgt[None])
    mag = (torch.exp(out) - 1e-8)[..., 0]  # [1, F, T]
    wav = griffin_lim_seeded(mag, sample_seed(gl_base_seed, gl_seed_key))[0].numpy()
    gt = tgt[0].numpy()
    if wav.shape[0] < gt.shape[0]:  # Griffin-Lim returns (frames-1)*hop+... = 9579 samples for 310 frames
        wav = np.pad(wav, (0, gt.shape[0] - wav.shape[0]))
    met = acoustic_metrics(wav, gt, evaluator, want_t60=want_t60, window=window)
    log_pred = out[0, :, :, 0].numpy()
    log_gt = np.log(tgt_spec[0, 0].numpy() + 1e-8)
    return {"wav": wav, "gt": gt, "log_pred": log_pred, "log_gt": log_gt, "metrics": met,
            "log_mse": float(np.mean((log_pred - log_gt) ** 2))}


def run_sim(chosen, evaluator):
    from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset
    manifest = load_manifest(os.path.join(REPO, SIM_MANIFEST))
    ds = ManifestDataset(xRIR_Dataset(split="test", max_len=9600, num_shot=8), manifest)
    pos = {e["query"]: k for k, e in enumerate(ds.entries)}
    models = {b: load_model(b, SIM_CKPT[b]) for b in SIM_CKPT}
    for col in chosen:
        item = ds[pos[col["query"]]]
        assert item[6] == col["query"]
        col["panorama"] = panorama(item[2], item[1], item[5])
        col["pred"] = {b: predict(models[b], item, col["query"], SIM_GL_SEED, True, 8000, evaluator) for b in models}
        col["window"] = 8000
        col["title"] = sim_title(col["query"])


def run_real(chosen, evaluator):
    for col in chosen:
        room = col["room"]
        ds = HAADataset([room], "test", num_shot=8, max_len=9600, eval_seed=HAA_EVAL_SEED)
        item = ds[ds.items.index((room, col["idx"]))]
        col["panorama"] = panorama(item[2], item[1], item[5])
        col["pred"] = {}
        for b in SIM_CKPT:
            ckpt = os.path.join("ckpt/sim2real", HAA_ARM[b], HAA_SEED, "stage2_%s" % room, "best.pth")
            model = load_model(b, ckpt)
            col["pred"][b] = predict(model, item, "%s/%d" % (room, col["idx"]), HAA_EVAL_SEED,
                                     room not in NO_T60_ROOMS, 9600, evaluator)
            col["checkpoint_%s" % b] = ckpt
        col["window"] = 9600
        col["title"] = "%s, mic %d" % (PRETTY_ROOM[room], col["idx"])


PRETTY_CAT = {"LivingRoomsWithHallway": "Living room w/ hallway", "ListeningRoom": "Listening room", "MeetingRoom": "Meeting room"}


def sim_title(query):
    cat, room, fn = query.split("/")
    idx = room.split("_idx_")[-1]
    s, r = fn.split("_")[:2]
    return "%s %s, S%d→R%d" % (PRETTY_CAT.get(cat, cat), idx, int(s[1:]), int(r[1:]))


def panorama(depth_coord, src, ref_locs):
    xyz = depth_coord.numpy()
    rng = np.sqrt((xyz ** 2).sum(0))
    u = xyz / np.maximum(rng, 1e-6)[None]

    def pix(v):
        v = v / max(float(np.linalg.norm(v)), 1e-6)
        dot = (u * v[:, None, None]).sum(0)
        r, c = np.unravel_index(int(np.argmax(dot)), dot.shape)
        return int(r), int(c)

    return {"range": rng, "src": pix(src.numpy()), "refs": [pix(r.numpy()) for r in ref_locs]}


# ----------------------------------------------------------------------------- figure
def fmt_metrics(met, want_t60):
    s = "EDT %.1f ms\nC50 %.2f dB" % (met["edt"] * 1000.0, met["c50"])
    if want_t60:
        s += "\nT60 %.1f %%" % met["t60"]
    return s


def draw_block(fig, gs, col0, cols, header, labels=True):
    rowmap = [0, 1, 2, 3, 5, 6, 7]  # gs row 4 is an empty spacer between the waveform and STFT groups
    axes = [[fig.add_subplot(gs[rowmap[r], col0 + j]) for j in range(len(cols))] for r in range(7)]
    for j, col in enumerate(cols):
        want_t60 = not (col["env"] == "real" and col["room"] in NO_T60_ROOMS)
        gt = col["pred"]["simple"]["gt"]
        t = np.arange(len(gt)) / SR
        amp = max(np.abs(gt).max(), *[np.abs(col["pred"][b]["wav"]).max() for b in ("simple", "cylindrical")]) * 1.05
        # row 0: panorama
        ax = axes[0][j]
        pan = col["panorama"]
        ax.imshow(pan["range"], cmap="gray_r", aspect="auto", vmin=0, vmax=np.percentile(pan["range"], 99))
        for r, c in pan["refs"]:
            ax.plot(c, r, marker="o", ms=4, mfc="none", mec="#009E73", mew=1.0, ls="none")
        ax.plot(pan["src"][1], pan["src"][0], marker="*", ms=9, color="#CC0000", mec="white", mew=0.5, ls="none")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(col["title"], fontsize=7.5, pad=3)
        # rows 1-3: waveforms
        for r, key in ((1, "gt"), (2, "simple"), (3, "cylindrical")):
            ax = axes[r][j]
            if key == "gt":
                ax.plot(t, gt, color=COLOR["gt"], lw=0.35)
            else:
                ax.plot(t, gt, color="#bbbbbb", lw=0.3)
                ax.plot(t, col["pred"][key]["wav"], color=COLOR[key], lw=0.35, alpha=0.9)
                ax.text(0.98, 0.95, fmt_metrics(col["pred"][key]["metrics"], want_t60), transform=ax.transAxes,
                        ha="right", va="top", fontsize=5.6, family="DejaVu Sans Mono",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", lw=0.4, alpha=0.9))
            ax.set_xlim(0, t[-1]); ax.set_ylim(-amp, amp)
            ax.set_yticks([]); ax.tick_params(axis="x", labelsize=5.5, length=2, pad=1)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            if r != 3:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("time (s)", fontsize=6, labelpad=1)
        # rows 4-6: log-STFT (dB)
        db = 20.0 / np.log(10.0)
        gt_db = db * col["pred"]["simple"]["log_gt"]
        vmax = float(np.percentile(gt_db, 99.5)); vmin = vmax - 60.0
        for r, key in ((4, "gt"), (5, "simple"), (6, "cylindrical")):
            ax = axes[r][j]
            spec = gt_db if key == "gt" else db * col["pred"][key]["log_pred"]
            ax.imshow(spec, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,
                      extent=[0, 9600 / SR, 0, SR / 2000.0])
            if key != "gt":
                ax.text(0.98, 0.95, "log-STFT MSE %.3f" % col["pred"][key]["log_mse"], transform=ax.transAxes, ha="right", va="top",
                        fontsize=5.6, family="DejaVu Sans Mono", color="white",
                        bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.45))
            ax.tick_params(labelsize=5.5, length=2, pad=1)
            if r != 6:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("time (s)", fontsize=6, labelpad=1)
            if j != 0 or not labels:
                ax.set_yticklabels([])
    # block header
    axes[0][0].annotate(header, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 16), textcoords="offset points",
                        ha="left", va="bottom", fontsize=9, fontweight="bold")
    if not labels:
        return axes
    # row labels, left of the first column
    names = ["depth panorama\n(model input)", "ground truth", LABEL["simple"], LABEL["cylindrical"],
             "ground truth", LABEL["simple"], LABEL["cylindrical"]]
    for r, lab in enumerate(names):
        ax = axes[r][0]
        color = COLOR["gt"] if r in (1, 4) else (COLOR["simple"] if r in (2, 5) else (COLOR["cylindrical"] if r in (3, 6) else "black"))
        ax.annotate(lab, xy=(0, 0.5), xycoords="axes fraction", xytext=(-6 if r < 4 else -20, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=6.5, color=color)
    for r in (4, 5, 6):
        axes[r][0].set_ylabel("kHz", fontsize=6, labelpad=1)
    # captions of the two row groups, above the first panel of each group
    axes[1][0].annotate("waveform (RIR)", xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
                        ha="left", va="bottom", fontsize=6.5, style="italic", color="#666666")
    axes[4][0].annotate("log-magnitude STFT (dB)", xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
                        ha="left", va="bottom", fontsize=6.5, style="italic", color="#666666")
    return axes


def render(sim, real, out_base):
    ncol = len(sim) + len(real)
    fig = plt.figure(figsize=(1.9 * ncol + 1.6, 10.2))
    gs = gridspec.GridSpec(8, ncol, figure=fig, left=0.072, right=0.995, top=0.94, bottom=0.05,
                           wspace=0.12, hspace=0.32, height_ratios=[0.75, 1, 1, 1, 0.12, 1.15, 1.15, 1.15])
    draw_block(fig, gs, 0, sim, "Simulated: AcousticRooms, unseen rooms (K = 8)")
    if real:
        draw_block(fig, gs, len(sim), real, "Real: Hearing Anything Anywhere, fine-tuned (K = 8)", labels=False)
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (out_base, ext), dpi=300 if ext == "png" else None)
    plt.close(fig)


def render_single(cols, out_base, header):
    fig = plt.figure(figsize=(1.9 * len(cols) + 1.6, 10.2))
    gs = gridspec.GridSpec(8, len(cols), figure=fig, left=0.135, right=0.99, top=0.94, bottom=0.05,
                           wspace=0.12, hspace=0.32, height_ratios=[0.75, 1, 1, 1, 0.12, 1.15, 1.15, 1.15])
    draw_block(fig, gs, 0, cols, header)
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (out_base, ext), dpi=300 if ext == "png" else None)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sim", type=int, default=4)
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.deterministic = True

    sim, sim_med, n_sim_cand, n_sim = select_sim(args.n_sim)
    real, real_med = select_real()
    print("sim candidates %d / %d (median baseline errors %s)" % (n_sim_cand, n_sim, {k: round(v, 4) for k, v in sim_med.items()}))
    for c in sim:
        print("  sim  %-70s json simple %s cyl %s" % (c["query"], c["json_metrics"]["simple"], c["json_metrics"]["cylindrical"]))
    for c in real:
        print("  real %-20s cand %d/%d json simple %s cyl %s" % (c["query"], c["n_candidates"], c["n"], c["json_metrics"]["simple"], c["json_metrics"]["cylindrical"]))

    evaluator = Evaluator()
    run_sim(sim, evaluator)
    run_real(real, evaluator)
    for c in sim + real:
        for b in ("simple", "cylindrical"):
            m = c["pred"][b]["metrics"]
            print("  recomputed %-4s %-40s %-11s EDT %.4f C50 %.3f T60 %s | json EDT %.4f C50 %.3f T60 %s" % (
                c["env"], c["query"], b, m["edt"], m["c50"], "%.2f" % m["t60"] if np.isfinite(m["t60"]) else "-",
                c["json_metrics"][b]["edt"], c["json_metrics"][b]["c50"],
                "%.2f" % c["json_metrics"][b]["t60"] if "t60" in c["json_metrics"][b] else "-"))

    os.makedirs(args.out_dir, exist_ok=True)
    render(sim, real, os.path.join(args.out_dir, "qualitative_sim_real"))
    render_single(sim, os.path.join(args.out_dir, "qualitative_sim"), "Simulated: AcousticRooms, unseen rooms (K = 8)")
    render_single(real, os.path.join(args.out_dir, "qualitative_real"), "Real: Hearing Anything Anywhere, fine-tuned (K = 8)")

    record = {"selection_rule": __doc__, "sim_median_baseline_error": sim_med, "real_median_baseline_error": real_med,
              "sim_runs": SIM_RUNS, "sim_checkpoints": SIM_CKPT, "sim_manifest": SIM_MANIFEST, "sim_gl_seed": SIM_GL_SEED,
              "haa_seed": HAA_SEED, "haa_eval_seed": HAA_EVAL_SEED, "device": "cpu", "torch": torch.__version__,
              "columns": [{k: v for k, v in c.items() if k not in ("pred", "panorama")} for c in sim + real]}
    for rec, c in zip(record["columns"], sim + real):
        rec["recomputed_metrics"] = {b: {k: (None if not np.isfinite(v) else float(v)) for k, v in c["pred"][b]["metrics"].items()}
                                     for b in ("simple", "cylindrical")}
        rec["log_stft_mse"] = {b: c["pred"][b]["log_mse"] for b in ("simple", "cylindrical")}
    with open(os.path.join(args.out_dir, "qualitative_selection.json"), "w") as f:
        json.dump(record, f, indent=2)
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
