"""Evaluate an xRIR backbone checkpoint on Hearing-Anything-Anywhere rooms (paper Table 2 protocol).

Per room: DiffRIR test split (or val), K=8 references drawn deterministically from the
room's 12 training RIRs (identical across models for paired comparison), Griffin-Lim
inversion, and the repo's eval_classroom.py metrics on the full 9600-sample 22.05 kHz
waveforms: EDT (s), C50 (dB), T60 (%), plus envelope error; T60 is skipped for
dampened_room as in the paper. Writes metrics_<room>.json and per_sample_<room>.json
(compatible with tools/compare_eval.py) into --save-dir.

    python sim_to_real/eval_haa.py --backbone cylindrical --checkpoint ckpt/sim2real/cyl/seed0/stage2_class_room/best.pth \
        --rooms class_room --save-dir ckpt/sim2real/cyl/seed0/eval
"""
import argparse
import contextlib
import io
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_unseen import Evaluator, griffin_lim
from eval_xRIR_backbone import load_model_state, summarize
from model.xRIR_cyl import BACKBONES, build_xrir
from sim_to_real.haa_dataset import NO_T60_ROOMS, ROOMS, HAADataset
from utils.spec_utils import compute_spect_energy_decay_losses, stft_l1_loss


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rooms", nargs="+", choices=ROOMS, default=ROOMS)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-shot", type=int, default=8)
    p.add_argument("--max-len", type=int, default=9600)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--depth-variant", choices=["default", "repo", "local"], default="default")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=0, help="per room, first N (0 = all)")
    p.add_argument("--tag", default="", help="suffix for output files, e.g. _val")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.save_dir, exist_ok=True)
    model = build_xrir(args.backbone, args.num_shot)
    model.load_state_dict(load_model_state(args.checkpoint), strict=True)
    model.cuda().eval()
    evaluator = Evaluator()
    all_summaries = {}
    for room in args.rooms:
        ds = HAADataset([room], args.split, num_shot=args.num_shot, max_len=args.max_len, eval_seed=args.eval_seed,
                        depth_variant=args.depth_variant)
        items = ds.items[: args.max_samples] if args.max_samples else ds.items
        loader = DataLoader(torch.utils.data.Subset(ds, list(range(len(items)))), batch_size=1, shuffle=False, num_workers=0)
        per = {"index": [], "ir_path": [], "edt": [], "c50": [], "t60": [], "stft_mse": [], "loss": [], "env": []}
        counts = {"c50_outliers": 0, "t60_invalid": 0, "edt_invalid": 0}
        t0 = time.time()
        with torch.no_grad():
            for i, data in enumerate(loader):
                _, src_loc, depth, tgt_wav, ref_irs, ref_locs = data
                out_spec, tgt_spec = model(depth.cuda(), ref_irs.cuda(), src_loc.cuda(), ref_locs.cuda(), tgt_wav.cuda())
                gt = tgt_spec.permute(0, 2, 3, 1)
                with contextlib.redirect_stdout(io.StringIO()):
                    decay = compute_spect_energy_decay_losses(gts=gt, preds=torch.exp(out_spec) - 1e-8)
                loss = float(stft_l1_loss(pred_spect=out_spec, gt_spect=gt) + decay)
                out_wav = griffin_lim((torch.exp(out_spec) - 1e-8)[..., 0].cpu()).unsqueeze(0)
                gt_ir = tgt_wav[0, 0].numpy()
                pred_ir = out_wav[0, 0].numpy()
                edt = c50 = t60 = float("nan")
                try:
                    edt = abs(evaluator.measure_edt(gt_ir) - evaluator.measure_edt(pred_ir))
                except (ValueError, IndexError):
                    counts["edt_invalid"] += 1
                c = abs(evaluator.measure_clarity(gt_ir) - evaluator.measure_clarity(pred_ir))
                if np.isfinite(c):
                    c50 = c
                else:
                    counts["c50_outliers"] += 1
                if room not in NO_T60_ROOMS:
                    try:
                        g, pr = evaluator.measure_rt60(gt_ir), evaluator.measure_rt60(pred_ir)
                        t60 = abs(g - pr) / g * 100.0
                    except (ValueError, IndexError):
                        counts["t60_invalid"] += 1
                m = min(len(gt_ir), len(pred_ir))
                env = evaluator.env_loss(pred_ir[:m], gt_ir[:m])
                mag = evaluator.stft_loss(out_spec.squeeze(-1).cpu().numpy(), torch.log(tgt_spec + 1e-8).squeeze(1).cpu().numpy())
                idx = items[i][1]
                per["index"].append(int(idx)); per["ir_path"].append(f"{room}/{idx}")
                per["edt"].append(float(edt)); per["c50"].append(float(c50)); per["t60"].append(float(t60))
                per["stft_mse"].append(float(mag)); per["loss"].append(loss); per["env"].append(float(env))
        summary = {"backbone": args.backbone, "checkpoint": args.checkpoint, "room": room, "split": args.split,
                   "num_shot": args.num_shot, "eval_seed": args.eval_seed, "depth_variant": args.depth_variant,
                   "n_samples": len(items),
                   "edt_error_s": summarize([v for v in per["edt"] if np.isfinite(v)]),
                   "c50_error_db": summarize([v for v in per["c50"] if np.isfinite(v)]),
                   "t60_error_pct": summarize([v for v in per["t60"] if np.isfinite(v)]) if room not in NO_T60_ROOMS else None,
                   "env_error": summarize(per["env"]), "stft_log_mse": summarize(per["stft_mse"]),
                   "test_loss": summarize(per["loss"]), **counts, "elapsed_min": (time.time() - t0) / 60}
        all_summaries[room] = summary
        with open(os.path.join(args.save_dir, f"metrics_{room}{args.tag}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(args.save_dir, f"per_sample_{room}{args.tag}.json"), "w") as f:
            json.dump({"meta": {k: summary[k] for k in ("backbone", "checkpoint", "split", "num_shot", "eval_seed")}, **per}, f)
        t60s = f"{summary['t60_error_pct']['mean']:.2f}%" if summary["t60_error_pct"] else "-"
        print(f"{room:14s} n={len(items):4d}  EDT {summary['edt_error_s']['mean']:.4f}s  C50 {summary['c50_error_db']['mean']:.3f}dB  "
              f"T60 {t60s}  env {summary['env_error']['mean']:.2f}  loss {summary['test_loss']['mean']:.4f}  "
              f"({summary['elapsed_min']:.1f} min)", flush=True)
    with open(os.path.join(args.save_dir, f"metrics_all{args.tag}.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
