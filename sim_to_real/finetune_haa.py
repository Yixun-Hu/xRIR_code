"""Fine-tune an xRIR backbone on Hearing-Anything-Anywhere rooms (paper sec. 5.5 / suppl. 9).

Parallel to the repo's finetune_all_rooms.py (stage 1: several rooms) and
finetune_classroom.py (stage 2: one room). Same recipe: AdamW lr 1e-4, wd 1e-4,
ExponentialLR 0.1x per 50 epochs, full-batch steps over the rooms' 12 DiffRIR training
RIRs with K=8 references re-drawn every iteration. Differences: checkpoint selection on
the DiffRIR *validation* split (as the paper states) rather than the test split, argparse,
``--backbone``, seeds, and a history log. Run from the repo root with PYTHONPATH set.

    # stage 1 (3 rooms, dampened excluded as in the repo), from an AcousticRooms checkpoint
    python sim_to_real/finetune_haa.py --backbone cylindrical --init ckpt/xRIR_cyl_8_shot/epoch_12.pth \
        --rooms class_room hallway complex_room --save-dir ckpt/sim2real/cyl/seed0/stage1 --epochs 1000 --val-every 10
    # stage 2 (one room), from the stage-1 best
    python sim_to_real/finetune_haa.py --backbone cylindrical --init ckpt/sim2real/cyl/seed0/stage1/best.pth \
        --rooms class_room --save-dir ckpt/sim2real/cyl/seed0/stage2_class_room --epochs 200 --val-every 2
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from eval_xRIR_backbone import load_model_state
from model.xRIR_cyl import BACKBONES, build_xrir
from sim_to_real.haa_dataset import ROOMS, HAADataset
from train_xRIR_backbone import compute_loss, seed_everything
from utils.lr_scheduler import ExponentialLR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    p.add_argument("--init", required=True, help="checkpoint to start from (state_dict or train last.pth)")
    p.add_argument("--rooms", nargs="+", choices=ROOMS, required=True)
    p.add_argument("--val-rooms", nargs="+", choices=ROOMS, default=None, help="default: same as --rooms")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-shot", type=int, default=8)
    p.add_argument("--max-len", type=int, default=9600)
    p.add_argument("--depth-variant", choices=["default", "repo", "local"], default="default")
    # Repo recipe.
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--decay-epochs", type=int, default=50)
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=0, help="0 = all training samples in one step (repo behaviour)")
    p.add_argument("--accum-steps", type=int, default=1)
    # Runtime / selection.
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=0, help="fixed reference draw for the validation split")
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--val-batch-size", type=int, default=32)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        loss, _, _ = compute_loss(model, batch)
        b = batch[1].shape[0]
        tot += loss.item() * b
        n += b
    model.train()
    return tot / max(n, 1)


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    val_rooms = args.val_rooms or args.rooms

    train_ds = HAADataset(args.rooms, "train", num_shot=args.num_shot, max_len=args.max_len, depth_variant=args.depth_variant)
    val_ds = HAADataset(val_rooms, "val", num_shot=args.num_shot, max_len=args.max_len, eval_seed=args.eval_seed,
                        depth_variant=args.depth_variant)
    bs = args.batch_size or len(train_ds)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=0)
    print(f"rooms {args.rooms}: {len(train_ds)} train targets, {len(val_ds)} val samples from {val_rooms}; "
          f"batch {bs} x accum {args.accum_steps}; {len(train_loader)} batches/epoch", flush=True)

    model = build_xrir(args.backbone, args.num_shot)
    model.load_state_dict(load_model_state(args.init), strict=True)
    model.cuda().train()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, decay_epochs=args.decay_epochs, gamma=args.lr_gamma)

    history_path = os.path.join(args.save_dir, "history.jsonl")
    open(history_path, "w").close()
    best_val, best_epoch = float("inf"), -1
    val0 = evaluate(model, val_loader)
    with open(history_path, "a") as f:
        f.write(json.dumps({"epoch": 0, "train_loss": None, "val_loss": val0, "lr": args.lr}) + "\n")
    print(f"epoch 0 (init): val loss {val0:.5f}", flush=True)
    best_val, best_epoch = val0, 0
    torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pth"))

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tot, n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for bi, batch in enumerate(train_loader):
            loss, _, _ = compute_loss(model, batch)
            (loss / args.accum_steps).backward()
            if (bi + 1) % args.accum_steps == 0 or bi + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            tot += loss.item() * batch[1].shape[0]
            n += batch[1].shape[0]
        scheduler.step()
        train_loss = tot / max(n, 1)
        rec = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val = evaluate(model, val_loader)
            rec["val_loss"] = val
            if val < best_val:
                best_val, best_epoch = val, epoch
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pth"))
                rec["is_best"] = True
        with open(history_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch}/{args.epochs}: train {train_loss:.5f}"
                  + (f"  val {rec['val_loss']:.5f}" if "val_loss" in rec else "")
                  + f"  best val {best_val:.5f} @ {best_epoch}  lr {rec['lr']:.1e}  {(time.time() - t0) / 60:.1f} min", flush=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, "last.pth"))
    summary = {"best_epoch": best_epoch, "best_val_loss": best_val, "init_val_loss": val0, "epochs": args.epochs,
               "minutes": (time.time() - t0) / 60, "rooms": args.rooms, "val_rooms": val_rooms, "init": args.init,
               "backbone": args.backbone, "seed": args.seed}
    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("done:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
