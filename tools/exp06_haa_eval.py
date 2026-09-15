"""exp_06 HAA evaluation entry point: exp_02's Table-2 protocol in a declared frame.

Plan v4 sections 6.2 and 8. Nothing of ``sim_to_real/eval_haa.py`` is modified: the
per-room loop below is that module's loop, with the pinned ``Evaluator`` /
``griffin_lim`` / ``summarize`` / ``load_model_state`` and the pinned spectrogram losses
imported. What this module owns is the exp_06 parser, the records the finalizer verifies
(``provenance.json`` and an ``args.json`` that is the same mapping), a per-sample ``meta``
carrying the frame, the heading binding and the checkpoint digest, a room-frame
``side_label`` per query, and the optional per-query Griffin-Lim seeding of the S1
sensitivity analysis (default off, so the primary comparison keeps exp_02's phases).

The heading rules are the fine-tuning wrapper's, imported: the oriented backbone may not
run without a heading directory, and every record must be decided, ``confirmatory`` and
still hash the cache files it was estimated from.

    python tools/exp06_haa_eval.py --backbone cylindrical_oriented \
        --checkpoint ckpt/exp06/sim2real/cyl_or/seed0/stage2_hallway/best.pth \
        --rooms hallway --heading-json-dir ckpt/exp06/heading \
        --save-dir ckpt/exp06/sim2real/cyl_or/seed0/eval/hallway
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from eval_unseen import Evaluator, griffin_lim
from eval_xRIR_backbone import load_model_state, summarize
from model.xRIR_cyl_oriented import BACKBONES_EXP06, build_xrir_exp06
from sim_to_real.haa_dataset import DEFAULT_ROOT, NO_T60_ROOMS, ROOMS
from tools import exp06_haa_finetune as records
from tools import provenance
from utils.spec_utils import compute_spect_energy_decay_losses, stft_l1_loss

RUN_TYPE = 'haa_eval'
ENTRY_MODULE = 'tools.exp06_haa_eval'
PINNED_META = ('backbone', 'checkpoint', 'split', 'num_shot', 'eval_seed')


def build_parser():
    """sim_to_real/eval_haa.py's flags and defaults, on the exp_06 registry."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=sorted(BACKBONES_EXP06), required=True)
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
    # exp_06 additions.
    p.add_argument("--root", default=DEFAULT_ROOT, help="HAA cache root (recorded resolved)")
    p.add_argument("--heading-json-dir", default=None,
                   help="one <room>.json per room; required for the oriented backbone")
    p.add_argument("--seed", type=int, default=0,
                   help="the pipeline seed this child belongs to; never the reference draw")
    p.add_argument("--gl-seed-per-query", action="store_true",
                   help="S1 sensitivity: seed Griffin-Lim per query instead of leaving it free")
    p.add_argument("--job-spec", default=None,
                   help="the pipeline declaration this child is launched under (recorded)")
    p.add_argument("--run-type", choices=(RUN_TYPE,), default=RUN_TYPE,
                   help="recorded in provenance.json; the finalizer dispatches on it")
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def prepare(args, command=(), repo=records.REPO):
    """Datasets, model and records, without touching a GPU or writing a file."""
    root = records.resolve_root(args)
    frame, k_by_room, heading = records.select_frame(args, root, list(args.rooms))
    if not os.path.isfile(args.checkpoint):
        raise ValueError('missing evaluation checkpoint: {}'.format(args.checkpoint))
    checkpoint_sha256 = provenance.sha256_file(args.checkpoint)
    datasets = {room: records.build_dataset([room], args.split, args, root, k_by_room,
                                            args.eval_seed) for room in args.rooms}
    model = build_xrir_exp06(args.backbone, args.num_shot)
    identity = records.data_identity(root, list(args.rooms), args.depth_variant)
    fields = records.provenance_fields(command, identity, repo=repo, run_type=RUN_TYPE,
                                       entry=ENTRY_MODULE)
    record = records.record_args(args, fields, frame=frame, heading=heading, haa_root=root,
                                 checkpoint_sha256=checkpoint_sha256,
                                 **records.job_binding(args.job_spec))
    return types.SimpleNamespace(root=root, frame=frame, heading=heading, datasets=datasets,
                                 model=model, fields=fields, args_record=record,
                                 checkpoint_sha256=checkpoint_sha256)


def per_sample_meta(args, context, room):
    """eval_haa.py's own meta keys, plus every field the finalizer binds to this arm."""
    meta = {key: getattr(args, key) for key in PINNED_META}
    meta.update(room=room, frame=context.frame, heading=context.heading,
                checkpoint_sha256=context.checkpoint_sha256,
                gl_seed_per_query=args.gl_seed_per_query, run_type=RUN_TYPE,
                registry_sha256=context.fields['registry_sha256'],
                git_head=context.fields['git_state']['HEAD'],
                exp06_source_closure_sha256=context.fields['source_closures']['child']['sha256'])
    return meta


def gl_seed(eval_seed, room, idx):
    """The frozen per-query phase seed of the S1 sensitivity analysis."""
    payload = 'gl:{}:{}:{}'.format(eval_seed, room, idx).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')


def invert(spec, args, room, idx):
    """The pinned Griffin-Lim inversion; nothing else changes when seeding is off."""
    if args.gl_seed_per_query:
        torch.manual_seed(gl_seed(args.eval_seed, room, idx))
    return griffin_lim(spec).unsqueeze(0)


def side_labels(dataset, room, indices):
    """Room-frame sign of each microphone's y relative to the speaker, before any roll."""
    labels = [int(dataset.data[room]['src_local'][idx, 1].sign().item()) for idx in indices]
    undecided = [idx for idx, label in zip(indices, labels) if label not in (-1, 1)]
    if undecided:
        raise ValueError('side_label is undefined for {} of {}: the microphone lies on the '
                         'speaker axis'.format(undecided[:4], room))
    return labels


def room_summary(args, room, per, counts, meta, elapsed_min):
    """Exactly sim_to_real/eval_haa.py's summary, with this arm's meta attached."""
    return {"backbone": args.backbone, "checkpoint": args.checkpoint, "room": room,
            "split": args.split, "num_shot": args.num_shot, "eval_seed": args.eval_seed,
            "depth_variant": args.depth_variant, "n_samples": len(per["index"]),
            "edt_error_s": summarize([v for v in per["edt"] if np.isfinite(v)]),
            "c50_error_db": summarize([v for v in per["c50"] if np.isfinite(v)]),
            "t60_error_pct": (summarize([v for v in per["t60"] if np.isfinite(v)])
                              if room not in NO_T60_ROOMS else None),
            "env_error": summarize(per["env"]), "stft_log_mse": summarize(per["stft_mse"]),
            "test_loss": summarize(per["loss"]), **counts,
            "elapsed_min": elapsed_min, "meta": meta}


def write_room_outputs(args, room, summary, per, meta):
    """The pinned file names, so the summariser's readers are unchanged."""
    with open(os.path.join(args.save_dir, "metrics_{}{}.json".format(room, args.tag)), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.save_dir, "per_sample_{}{}.json".format(room, args.tag)), "w") as f:
        json.dump(dict(meta=meta, **per), f)


def evaluate_room(model, evaluator, args, context, room):
    """sim_to_real/eval_haa.py's per-room loop, with the room-frame side label per query."""
    dataset = context.datasets[room]
    items = dataset.items[: args.max_samples] if args.max_samples else dataset.items
    loader = DataLoader(Subset(dataset, list(range(len(items)))), batch_size=1, shuffle=False,
                        num_workers=0)
    per = {"index": [], "ir_path": [], "edt": [], "c50": [], "t60": [], "stft_mse": [],
           "loss": [], "env": [], "side_label": []}
    counts = {"c50_outliers": 0, "t60_invalid": 0, "edt_invalid": 0}
    started = time.time()
    with torch.no_grad():
        for i, data in enumerate(loader):
            _, src_loc, depth, tgt_wav, ref_irs, ref_locs = data
            out_spec, tgt_spec = model(depth.cuda(), ref_irs.cuda(), src_loc.cuda(),
                                       ref_locs.cuda(), tgt_wav.cuda())
            gt = tgt_spec.permute(0, 2, 3, 1)
            with contextlib.redirect_stdout(io.StringIO()):
                decay = compute_spect_energy_decay_losses(gts=gt, preds=torch.exp(out_spec) - 1e-8)
            loss = float(stft_l1_loss(pred_spect=out_spec, gt_spect=gt) + decay)
            idx = items[i][1]
            out_wav = invert((torch.exp(out_spec) - 1e-8)[..., 0].cpu(), args, room, idx)
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
            mag = evaluator.stft_loss(out_spec.squeeze(-1).cpu().numpy(),
                                      torch.log(tgt_spec + 1e-8).squeeze(1).cpu().numpy())
            per["index"].append(int(idx))
            per["ir_path"].append("{}/{}".format(room, idx))
            per["edt"].append(float(edt))
            per["c50"].append(float(c50))
            per["t60"].append(float(t60))
            per["stft_mse"].append(float(mag))
            per["loss"].append(loss)
            per["env"].append(float(env))
    per["side_label"] = side_labels(dataset, room, per["index"])
    return per, counts, (time.time() - started) / 60


def main(argv=None):
    """sim_to_real/eval_haa.py's main, with the exp_06 records written before the loop."""
    command = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    context = prepare(args, command)
    records.write_records(args, context)
    model = context.model
    model.load_state_dict(load_model_state(args.checkpoint), strict=True)
    model.cuda().eval()
    evaluator = Evaluator()
    all_summaries = {}
    for room in args.rooms:
        meta = per_sample_meta(args, context, room)
        per, counts, elapsed = evaluate_room(model, evaluator, args, context, room)
        summary = room_summary(args, room, per, counts, meta, elapsed)
        all_summaries[room] = summary
        write_room_outputs(args, room, summary, per, meta)
        t60s = ("{:.2f}%".format(summary["t60_error_pct"]["mean"])
                if summary["t60_error_pct"] else "-")
        print("{:14s} n={:4d}  EDT {:.4f}s  C50 {:.3f}dB  T60 {}  env {:.2f}  loss {:.4f}  "
              "({:.1f} min)".format(room, summary["n_samples"], summary["edt_error_s"]["mean"],
                                    summary["c50_error_db"]["mean"], t60s,
                                    summary["env_error"]["mean"], summary["test_loss"]["mean"],
                                    elapsed), flush=True)
    with open(os.path.join(args.save_dir, "metrics_all{}.json".format(args.tag)), "w") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
