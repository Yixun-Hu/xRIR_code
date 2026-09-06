"""Evaluate an xRIR (SimpleViT) or xRIR_Cyl (CylindricalViT) checkpoint on AcousticRooms.

Parallel to ``eval_unseen.py`` / ``eval_seen.py`` and reuses their ``Evaluator`` and
``griffin_lim`` so the EDT / C50 / T60 numbers are computed identically (first 8000
samples of the Griffin-Lim inverted prediction vs. the target). Adds: argparse,
``--backbone``, ``--split unseen|seen``, deterministic order (``shuffle`` off unless
requested), DataLoader workers, a seeded ``--max-samples`` subset, and a JSON summary.

    python eval_xRIR_backbone.py --backbone simple      --split unseen --checkpoint checkpoints/xRIR_unseen.pth
    python eval_xRIR_backbone.py --backbone cylindrical --split unseen --checkpoint ckpt/xRIR_cyl_8_shot/best.pth \
        --save-metrics ckpt/xRIR_cyl_8_shot/eval_unseen.json
"""
import argparse
import contextlib
import io
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from eval_unseen import Evaluator, griffin_lim
from model.xRIR_cyl import BACKBONES, build_xrir
from utils.spec_utils import compute_spect_energy_decay_losses, stft_l1_loss


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["unseen", "seen"], default="unseen")
    p.add_argument("--num-shot", type=int, default=8)
    p.add_argument("--max-len", type=int, default=9600)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads (Griffin-Lim runs on CPU)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle", action="store_true", help="shuffle like the original eval scripts")
    p.add_argument("--max-samples", type=int, default=0, help="seeded random subset of the split (0 = all)")
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--save-metrics", default=None, help="write the summary JSON here")
    p.add_argument("--save-per-sample", default=None,
                   help="write per-sample metrics (NaN where invalid) as JSON here, for paired comparisons "
                        "with tools/compare_eval.py")
    return p.parse_args()


def seed_worker(worker_id):
    s = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(s)
    random.seed(s)


def load_model_state(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]  # train_xRIR_backbone.py last.pth
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in ckpt.items()}


def build_dataset(split, num_shot, max_len):
    if split == "unseen":
        from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset
    else:
        from treble_multi_room_dataset.treble_xRIR_seen_dataset import xRIR_Dataset
    return xRIR_Dataset(split="test", max_len=max_len, num_shot=num_shot)


def summarize(vals):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "median": None, "n": 0}
    return {"mean": float(a.mean()), "median": float(np.median(a)), "n": int(a.size)}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    base_dataset = build_dataset(args.split, args.num_shot, args.max_len)
    dataset = base_dataset
    n_total = len(dataset)
    indices = list(range(n_total))
    if args.max_samples and args.max_samples < n_total:
        idx = np.random.RandomState(args.seed).permutation(n_total)[: args.max_samples]
        indices = sorted(idx.tolist())
        dataset = Subset(dataset, indices)
    if args.shuffle and args.save_per_sample:
        raise SystemExit("--save-per-sample needs a deterministic order; drop --shuffle")
    loader = DataLoader(dataset, batch_size=1, shuffle=args.shuffle, num_workers=args.num_workers,
                        worker_init_fn=seed_worker, pin_memory=True)
    print(f"split: {args.split}  samples: {len(dataset)} / {n_total}  backbone: {args.backbone}  "
          f"checkpoint: {args.checkpoint}", flush=True)

    model = build_xrir(args.backbone, args.num_shot)
    model.load_state_dict(load_model_state(args.checkpoint), strict=True)
    model.cuda().eval()

    evaluator = Evaluator()
    edt_err, c50_err, t60_err, mag_loss, loss_all = [], [], [], [], []
    c50_outliers = t60_invalid = edt_invalid = 0
    per = {"index": [], "ir_path": [], "edt": [], "c50": [], "t60": [], "stft_mse": [], "loss": []}
    t0 = time.time()
    with torch.no_grad():
        for i, data in enumerate(loader):
            _, src_loc, depth, tgt_wav, ref_irs, ref_locs = data
            out_spec, tgt_spec = model(depth.cuda(), ref_irs.cuda(), src_loc.cuda(), ref_locs.cuda(), tgt_wav.cuda())

            gt = tgt_spec.permute(0, 2, 3, 1)
            with contextlib.redirect_stdout(io.StringIO()):
                decay = compute_spect_energy_decay_losses(gts=gt, preds=torch.exp(out_spec) - 1e-8)
            loss_all.append(float(stft_l1_loss(pred_spect=out_spec, gt_spect=gt) + decay))

            pred_mag_spec = (torch.exp(out_spec) - 1e-8)[..., 0].cpu()
            out_wav = griffin_lim(pred_mag_spec).unsqueeze(0)
            gt_ir = tgt_wav[0, 0, :8000].cpu().numpy()
            pred_ir = out_wav[0, 0, :8000].cpu().numpy()

            edt_i = c50_i = t60_i = float("nan")
            try:
                edt_i = abs(evaluator.measure_edt(gt_ir) - evaluator.measure_edt(pred_ir))
                edt_err.append(edt_i)
            except (ValueError, IndexError):
                edt_invalid += 1
            c50 = abs(evaluator.measure_clarity(gt_ir) - evaluator.measure_clarity(pred_ir))
            if np.isfinite(c50):
                c50_i = c50
                c50_err.append(c50)
            else:
                c50_outliers += 1
            try:
                gt_t60 = evaluator.measure_rt60(gt_ir)
                pred_t60 = evaluator.measure_rt60(pred_ir)
                t60_i = abs(gt_t60 - pred_t60) / gt_t60 * 100.0
                t60_err.append(t60_i)
            except (ValueError, IndexError):
                t60_invalid += 1
            mag_i = evaluator.stft_loss(out_spec.squeeze(-1).cpu().numpy(),
                                        torch.log(tgt_spec + 1e-8).squeeze(1).cpu().numpy())
            mag_loss.append(mag_i)
            per["index"].append(int(indices[i])); per["ir_path"].append(os.path.relpath(base_dataset.file_list[indices[i]], base_dataset.ir_path))
            per["edt"].append(float(edt_i)); per["c50"].append(float(c50_i)); per["t60"].append(float(t60_i))
            per["stft_mse"].append(float(mag_i)); per["loss"].append(loss_all[-1])

            if (i + 1) % args.log_interval == 0 or i + 1 == len(dataset):
                el = time.time() - t0
                print(f"[{i + 1}/{len(dataset)}] EDT {np.mean(edt_err):.4f}s  C50 {np.mean(c50_err):.4f}dB  "
                      f"T60 {np.mean(t60_err):.3f}%  STFT-MSE {np.mean(mag_loss):.4f}  loss {np.mean(loss_all):.4f}  "
                      f"({(i + 1) / el:.2f} samp/s, eta {(len(dataset) - i - 1) / ((i + 1) / el) / 60:.1f} min)",
                      flush=True)

    summary = {
        "backbone": args.backbone, "checkpoint": args.checkpoint, "split": args.split,
        "num_shot": args.num_shot, "seed": args.seed, "shuffle": args.shuffle,
        "n_samples": len(dataset), "n_split_total": n_total,
        "edt_error_s": summarize(edt_err), "c50_error_db": summarize(c50_err),
        "t60_error_pct": summarize(t60_err), "stft_log_mse": summarize(mag_loss),
        "test_loss": summarize(loss_all),
        "c50_outliers": c50_outliers, "t60_invalid": t60_invalid, "edt_invalid": edt_invalid,
        "elapsed_min": (time.time() - t0) / 60.0,
    }
    print(json.dumps(summary, indent=2))
    print(f"Average EDT error: {summary['edt_error_s']['mean']}")
    print(f"Average C50 error: {summary['c50_error_db']['mean']}")
    print(f"Average T60 error: {summary['t60_error_pct']['mean']}")
    print(f"Number of Outliers: {c50_outliers}")
    if args.save_metrics:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_metrics)), exist_ok=True)
        with open(args.save_metrics, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"saved {args.save_metrics}")
    if args.save_per_sample:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_per_sample)), exist_ok=True)
        with open(args.save_per_sample, "w") as f:
            json.dump({"meta": {k: summary[k] for k in ("backbone", "checkpoint", "split", "num_shot", "seed")}, **per}, f)
        print(f"saved {args.save_per_sample}")


if __name__ == "__main__":
    main()
