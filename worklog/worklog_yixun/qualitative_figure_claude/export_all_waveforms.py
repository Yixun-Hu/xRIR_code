"""Predicted RIR waveforms of xRIR SimpleViT and CylindricalViT for EVERY query of the
AcousticRooms unseen split (6 337, exp_04 K=8 seed-42 references, epoch-12 checkpoints) and the
HAA DiffRIR test split (1 282 over four rooms, exp_02 seed-0 fine-tuned checkpoints, eval_seed 0
references), plus the ground truth, as float32 [N, 9600] arrays at 22050 Hz in the canonical
query order, and a per-query log-STFT MSE CSV computed with the FLAC-side definition
(crop to the evaluator window, centre-padded Hann STFT n_fft 512 hop 64, ln(|X| + 1e-8), MSE over
all bins).  Requested by the cylindrical-dinov3 session for the joint figure (2026-09-23).
CPU only (batched inference; per-query seeded Griffin-Lim exactly as in the canonical evaluations)."""
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_qualitative_figure as Q  # noqa: E402  (patches model.xRIR.apply_delay for CPU)
from sim_to_real.haa_dataset import ROOMS, HAADataset  # noqa: E402
from tools.per_sample_metrics import griffin_lim_seeded, sample_seed  # noqa: E402
from tools.reference_manifest import ManifestDataset, load_manifest  # noqa: E402

OUT = os.path.join(HERE, "joint_export", "full")
FILE_ROOM = {"class_room": "classroom", "dampened_room": "dampened", "hallway": "hallway", "complex_room": "complex"}
BATCH = 16


def log_stft_mse(pred, gt, window):
    """FLAC-side definition: crop to `window`, centre pad n_fft/2, Hann n_fft 512 hop 64, ln(|X|+1e-8), MSE."""
    p = torch.from_numpy(pred[:window].astype(np.float32))
    g = torch.from_numpy(gt[:window].astype(np.float32))
    win = torch.hann_window(512)
    sp = torch.log(torch.stft(p, 512, 64, 512, win, center=True, pad_mode="constant", return_complex=True).abs() + 1e-8)
    sg = torch.log(torch.stft(g, 512, 64, 512, win, center=True, pad_mode="constant", return_complex=True).abs() + 1e-8)
    return float(((sp - sg) ** 2).mean())


@torch.no_grad()
def run(models, loader, keys_of_batch, gl_base_seed, n):
    gt = np.zeros((n, 9600), np.float32)
    preds = {b: np.zeros((n, 9600), np.float32) for b in models}
    i0 = 0
    t0 = time.time()
    for bi, batch in enumerate(loader):
        _, src, depth, tgt, refs, ref_locs = batch[:6]
        keys = keys_of_batch(batch, i0)
        bs = tgt.shape[0]
        gt[i0:i0 + bs] = tgt[:, 0].numpy()
        for b, model in models.items():
            out, _ = model(depth, refs, src, ref_locs, tgt)
            mag = (torch.exp(out) - 1e-8)[..., 0]
            for j in range(bs):
                wav = griffin_lim_seeded(mag[j:j + 1], sample_seed(gl_base_seed, keys[j]))[0].numpy()
                preds[b][i0 + j, :wav.shape[0]] = wav
        i0 += bs
        if bi % 20 == 0:
            print("  %d / %d  (%.1f min)" % (i0, n, (time.time() - t0) / 60), flush=True)
    return gt, preds


def main():
    torch.set_num_threads(24)
    os.makedirs(OUT, exist_ok=True)
    rows = []

    # ---- AcousticRooms unseen ------------------------------------------------------------
    from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset
    man = load_manifest(os.path.join(Q.REPO, Q.SIM_MANIFEST))
    ds = ManifestDataset(xRIR_Dataset(split="test", max_len=9600, num_shot=8), man)
    models = {b: Q.load_model(b, Q.SIM_CKPT[b]) for b in Q.SIM_CKPT}
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=8)
    print("sim: %d queries" % len(ds), flush=True)
    gt, preds = run(models, loader, lambda batch, i0: list(batch[6]), Q.SIM_GL_SEED, len(ds))
    ids = []
    for e in ds.entries:
        cat, room, fn = e["query"].split("/")
        s, r = fn.split("_")[:2]
        ids.append({"query": e["query"], "env": "acousticrooms_unseen", "room": room, "query_id": "S%d_R%d" % (int(s[1:]), int(r[1:])),
                    "reference_ids": e["refs"], "normalisation_constant": 1.0, "metric_window_samples": 8000})
    np.save(os.path.join(OUT, "sim_unseen_gt.npy"), gt)
    for b, tag in Q.__dict__.get("TAG", {"simple": "simplevit", "cylindrical": "cylvit"}).items():
        np.save(os.path.join(OUT, "sim_unseen_%s.npy" % tag), preds[b])
        for k, rec in enumerate(ids):
            rows.append([rec["env"], rec["room"], rec["query_id"], tag, "%.6g" % log_stft_mse(preds[b][k], gt[k], 8000)])
    json.dump({"order": "row k of every sim_unseen_*.npy is ids[k]; canonical manifest order (sorted query path), identical to per_sample_yaw.json",
               "checkpoints": Q.SIM_CKPT, "reference_manifest": Q.SIM_MANIFEST, "griffin_lim": "sample_seed(42, query)",
               "ids": ids}, open(os.path.join(OUT, "sim_unseen_ids.json"), "w"), indent=1)
    del gt, preds, models, ds, loader

    # ---- HAA test, four rooms ------------------------------------------------------------
    gts, predss, ids = [], {"simple": [], "cylindrical": []}, []
    for room in ROOMS:
        hds = HAADataset([room], "test", num_shot=8, max_len=9600, eval_seed=Q.HAA_EVAL_SEED)
        ckpts = {b: os.path.join("ckpt/sim2real", Q.HAA_ARM[b], Q.HAA_SEED, "stage2_%s" % room, "best.pth") for b in Q.SIM_CKPT}
        models = {b: Q.load_model(b, c) for b, c in ckpts.items()}
        loader = DataLoader(hds, batch_size=BATCH, shuffle=False, num_workers=4)
        print("haa %s: %d queries" % (room, len(hds)), flush=True)
        keys = ["%s/%d" % (r, i) for r, i in hds.items]
        gt, preds = run(models, loader, lambda batch, i0: keys[i0:i0 + batch[3].shape[0]], Q.HAA_EVAL_SEED, len(hds))
        scale = float(hds.data[room]["meta"]["scale"])
        for k, (r, i) in enumerate(hds.items):
            ids.append({"env": "haa", "room": room, "query_id": "mic%d" % i, "mic_index": i, "reference_ids": [int(x) for x in hds._pick_refs(room, i)],
                        "checkpoints": ckpts, "normalisation_constant": scale, "metric_window_samples": 9600, "t60_reported": room != "dampened_room"})
            for b, tag in (("simple", "simplevit"), ("cylindrical", "cylvit")):
                rows.append(["haa", room, "mic%d" % i, tag, "%.6g" % log_stft_mse(preds[b][k], gt[k], 9600)])
        gts.append(gt)
        for b in preds:
            predss[b].append(preds[b])
    np.save(os.path.join(OUT, "haa_test_gt.npy"), np.concatenate(gts))
    np.save(os.path.join(OUT, "haa_test_simplevit.npy"), np.concatenate(predss["simple"]))
    np.save(os.path.join(OUT, "haa_test_cylvit.npy"), np.concatenate(predss["cylindrical"]))
    json.dump({"order": "row k of every haa_test_*.npy is ids[k]; rooms in order class_room, dampened_room, hallway, complex_room, DiffRIR test indices ascending (= per_sample_<room>.json order)",
               "scale_note": "GT and predictions are divided by normalisation_constant (max |RIR| over the room's 12 training RIRs); multiply by it to recover the HAA scale",
               "griffin_lim": "sample_seed(0, '<room>/<mic>')", "ids": ids}, open(os.path.join(OUT, "haa_test_ids.json"), "w"), indent=1)

    with open(os.path.join(HERE, "joint_export", "per_query_log_stft_mse.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "room", "query_id", "model", "log_stft_mse"])
        w.writerows(rows)
    print("DONE", len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
