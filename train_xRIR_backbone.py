"""Train xRIR with a selectable geometry encoder on the AcousticRooms unseen-room split.

Parallel to ``train_xRIR_unseen.py``. Same dataset, loss (STFT L1 + energy-decay),
optimizer (AdamW), LR schedule (ExponentialLR) and default hyperparameters, plus:
argparse, ``--backbone simple|cylindrical`` (SimpleViT baseline vs CylindricalViT),
``--protocol unseen|seen`` (the authors' split convention; ``seen`` must be run from the
repository root because that dataset module opens its pickle relative to the cwd),
DataLoader workers, gradient accumulation, intra-epoch checkpoints, resume, and
bounded smoke-test flags. Run from the repo root with PYTHONPATH set.

    python train_xRIR_backbone.py --backbone cylindrical --save-dir ckpt/xRIR_cyl_8_shot
    python train_xRIR_backbone.py --backbone simple      --save-dir ckpt/xRIR_simple_8_shot
    # smoke test
    python train_xRIR_backbone.py --backbone cylindrical --save-dir ckpt/_smoke \
        --epochs 1 --max-train-batches 2 --max-test-batches 2 --batch-size 2
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
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from model.xRIR_cyl import BACKBONES, build_xrir
from tools.exp05_params import TIERS, count_parameters, tier_of
from tools.yaw_aug import YawAug, apply_yaw_aug
from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset as UnseenDataset
from treble_multi_room_dataset.treble_xRIR_seen_dataset import xRIR_Dataset as SeenDataset
from utils.lr_scheduler import ExponentialLR
from utils.spec_utils import compute_spect_energy_decay_losses, stft_l1_loss

PROTOCOLS = ("unseen", "seen")
# Backwards-compatible name for the unseen dataset: tools/yaw_aug.py and the exp_04/exp_05
# tests patch ``trainer.xRIR_Dataset``, and dataset_class honours that patch.
xRIR_Dataset = UnseenDataset


def dataset_class(protocol):
    """Return the dataset class of the authors' split convention for ``protocol``.

    Both modules are imported above so that the import-derived training closure always
    contains them, whichever protocol a run selects.
    """
    if protocol not in PROTOCOLS:
        raise ValueError("unknown protocol: " + str(protocol))
    return SeenDataset if protocol == "seen" else xRIR_Dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=sorted(BACKBONES), required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-shot", type=int, default=8)
    p.add_argument("--max-len", type=int, default=9600)
    p.add_argument("--protocol", choices=PROTOCOLS, default="unseen",
                   help="split convention: unseen = the held-out rooms (default), "
                        "seen = treble_multi_room_dataset/seen_test_split.pkl")
    for key, value in TIERS['M'].items():
        p.add_argument('--vit-' + key.replace('_', '-'), type=int, default=value)
    # Baseline hyperparameters from train_xRIR_unseen.py.
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--decay-epochs", type=int, default=50)
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--accum-steps", type=int, default=1,
                   help="gradient accumulation; effective batch = batch-size * accum-steps")
    # Runtime.
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--yaw-aug", type=int, choices=(0, 1), default=0)
    p.add_argument("--yaw-aug-seed", type=int, default=None)
    p.add_argument("--yaw-aug-width", type=int, default=512)
    p.add_argument("--no-save", action="store_true", help="write nothing to save-dir (smoke runs)")
    p.add_argument("--tf32", action="store_true", help="allow TF32 matmul/cudnn (faster, slightly different numerics)")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500, help="save last.pth every N train batches (0 = only per epoch)")
    p.add_argument("--epoch-ckpt-every", type=int, default=5,
                   help="also keep a standalone epoch_NNN.pth every N epochs (0 = never); best.pth/last.pth are always written")
    p.add_argument("--resume", default=None, help="path to a last.pth to resume from")
    # Bounded runs.
    p.add_argument("--max-train-batches", type=int, default=0, help="0 = full epoch")
    p.add_argument("--max-test-batches", type=int, default=0, help="0 = full test split")
    p.add_argument("--test-subset", type=int, default=0,
                   help="evaluate on a fixed seeded subset of N test samples each epoch (0 = all)")
    args = p.parse_args()
    args.yaw_aug_seed = args.seed if args.yaw_aug_seed is None else args.yaw_aug_seed
    if args.yaw_aug and (args.resume is not None or args.save_every != 0):
        p.error("--yaw-aug 1 requires --resume to be None and --save-every exactly 0")
    return args


def build_model(args):
    model = build_xrir(args.backbone, args.num_shot,
                       **{key: getattr(args, 'vit_' + key) for key in TIERS['M']})
    args.param_counts = count_parameters(model)
    try:
        args.tier = tier_of(vars(args))
    except ValueError:
        args.tier = 'custom'
    return model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    s = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(s)
    random.seed(s)


def compute_loss(model, batch, aug=None):
    """Identical loss to train_xRIR_unseen.py: STFT log-mag L1 + Schroeder energy-decay L1."""
    _, src_loc, depth_coord, tgt_wav, ref_irs, ref_locs = batch
    if aug is not None:
        yaw, epoch, batch_idx = aug
        assert yaw.enabled, "aug requires enabled yaw augmentation"
        if yaw.W != depth_coord.shape[-1]:
            raise ValueError("--yaw-aug-width must equal the dataset depth panorama width")
        depth_coord, src_loc, ref_locs = (t.cuda(non_blocking=True) for t in (depth_coord, src_loc, ref_locs))
        ks = yaw.offsets_for(epoch, batch_idx, int(src_loc.shape[0]))
        depth_coord, src_loc, ref_locs = apply_yaw_aug(
            depth_coord, src_loc, ref_locs, ks.to(depth_coord.device), W=yaw.W)
    out_spec, tgt_spec = model(
        depth_coord.cuda(non_blocking=True), ref_irs.cuda(non_blocking=True),
        src_loc.cuda(non_blocking=True), ref_locs.cuda(non_blocking=True),
        tgt_wav.cuda(non_blocking=True))
    gt = tgt_spec.permute(0, 2, 3, 1)
    with contextlib.redirect_stdout(io.StringIO()):  # the decay loss prints tensor shapes on every call
        decay_loss = compute_spect_energy_decay_losses(gts=gt, preds=torch.exp(out_spec) - 1e-8)
    stft_loss = stft_l1_loss(pred_spect=out_spec, gt_spect=gt)
    return stft_loss + decay_loss, stft_loss.detach(), decay_loss.detach()


def save_checkpoint(path, model, optimizer, scheduler, epoch, batch_idx, best_test_loss, args):
    if args.no_save:
        return
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "batch_idx": batch_idx,
        "best_test_loss": best_test_loss,
        "args": vars(args),
    }, path)


def train_epoch(model, loader, optimizer, scheduler, epoch, args, best_test_loss):
    if type(epoch) is not int:
        raise ValueError("epoch must be a native int")
    model.train()
    aug = YawAug(True, args.yaw_aug_width, args.yaw_aug_seed, args.train_batches_per_epoch) if args.yaw_aug else None
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader) if not args.max_train_batches else min(len(loader), args.max_train_batches)
    tot = tot_stft = tot_decay = 0.0
    count = 0
    t_start = time.time()
    t_last = t_start
    for batch_idx, batch in enumerate(loader):
        if type(batch_idx) is not int:
            raise ValueError("batch_idx must be a native int")
        if args.max_train_batches and batch_idx >= args.max_train_batches:
            break
        loss, stft_l, decay_l = compute_loss(
            model, batch, (aug, epoch, batch_idx) if aug is not None else None)
        (loss / args.accum_steps).backward()
        if (batch_idx + 1) % args.accum_steps == 0 or batch_idx + 1 == n_batches:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        tot += loss.item()
        tot_stft += stft_l.item()
        tot_decay += decay_l.item()
        count += 1
        if batch_idx % args.log_interval == 0:
            now = time.time()
            rate = args.log_interval * batch[1].shape[0] / max(now - t_last, 1e-6) if batch_idx else float("nan")
            eta_min = (n_batches - batch_idx - 1) * (now - t_start) / max(batch_idx + 1, 1) / 60.0
            print(f"Train Epoch: {epoch} [{batch_idx}/{n_batches}]  loss {loss.item():.4f} "
                  f"(stft {stft_l.item():.4f}, decay {decay_l.item():.4f})  "
                  f"{rate:.1f} samp/s  eta {eta_min:.1f} min  lr {optimizer.param_groups[0]['lr']:.2e}",
                  flush=True)
            t_last = now
        if args.save_every and batch_idx and batch_idx % args.save_every == 0:
            save_checkpoint(os.path.join(args.save_dir, "last.pth"), model, optimizer, scheduler,
                            epoch, batch_idx, best_test_loss, args)
    scheduler.step()
    avg = tot / max(count, 1)
    print(f"Train Epoch: {epoch}, Avg Train Loss: {avg:.5f} (stft {tot_stft / max(count, 1):.5f}, "
          f"decay {tot_decay / max(count, 1):.5f}), {(time.time() - t_start) / 60:.1f} min", flush=True)
    return avg


@torch.no_grad()
def test_epoch(model, loader, epoch, args):
    model.eval()
    tot = 0.0
    count = 0
    for batch_idx, batch in enumerate(loader):
        if args.max_test_batches and batch_idx >= args.max_test_batches:
            break
        loss, _, _ = compute_loss(model, batch)
        tot += loss.item()
        count += 1
    avg = tot / max(count, 1)
    print(f"Test set (epoch {epoch}): Average loss: {avg:.5f} over {count} batches", flush=True)
    return avg


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    dataset = dataset_class(args.protocol)
    train_dataset = dataset(split="train", max_len=args.max_len, num_shot=args.num_shot)
    test_dataset = dataset(split="test", max_len=args.max_len, num_shot=args.num_shot)
    if args.test_subset and args.test_subset < len(test_dataset):
        idx = np.random.RandomState(args.seed).permutation(len(test_dataset))[: args.test_subset]
        test_dataset = Subset(test_dataset, sorted(idx.tolist()))
    print(f"train samples: {len(train_dataset)}  test samples: {len(test_dataset)}", flush=True)

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker,
                         persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, **loader_kwargs)
    args.train_batches_per_epoch = len(train_loader)
    if args.yaw_aug and args.epochs * args.train_batches_per_epoch >= 2**20:
        raise ValueError("--yaw-aug 1 requires epochs * train_batches_per_epoch < 2**20")
    args.env = {key: os.environ.get(key) for key in
                ("PYTHONHASHSEED", "XRIR_DATA_PATH", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}
    model = build_model(args).cuda()
    print("XRIR_RUNTIME_ARGS " + json.dumps(vars(args), sort_keys=True, allow_nan=False), flush=True)
    if not args.no_save:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    n_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"backbone: {args.backbone}  params: {n_params / 1e6:.2f}M  model size: {size_mb:.1f}MB", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, decay_epochs=args.decay_epochs, gamma=args.lr_gamma)

    start_epoch = 1
    best_test_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_test_loss = ckpt.get("best_test_loss", best_test_loss)
        # A mid-epoch checkpoint restarts that epoch; a per-epoch checkpoint starts the next one.
        start_epoch = ckpt["epoch"] if ckpt.get("batch_idx", 0) else ckpt["epoch"] + 1
        print(f"resumed from {args.resume}: epoch {ckpt['epoch']} batch {ckpt.get('batch_idx', 0)} "
              f"best_test_loss {best_test_loss:.5f} -> starting epoch {start_epoch}", flush=True)

    history_path = os.path.join(args.save_dir, "history.jsonl")
    print(f"yaw_aug ENABLED W={args.yaw_aug_width} seed={args.yaw_aug_seed} "
          f"counter=(epoch-1)*{args.train_batches_per_epoch}+batch_idx" if args.yaw_aug else "yaw_aug DISABLED", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch, args, best_test_loss)
        test_loss = test_epoch(model, test_loader, epoch, args)
        if args.no_save:
            continue
        is_best = test_loss < best_test_loss
        if is_best:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pth"))
        if args.epoch_ckpt_every and epoch % args.epoch_ckpt_every == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"epoch_{epoch:03d}.pth"))
        save_checkpoint(os.path.join(args.save_dir, "last.pth"), model, optimizer, scheduler,
                        epoch, 0, best_test_loss, args)
        with open(history_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss,
                                "best_test_loss": best_test_loss, "is_best": is_best,
                                "lr": optimizer.param_groups[0]["lr"],
                                "epoch_minutes": (time.time() - t0) / 60.0}) + "\n")
        print(f"epoch {epoch} done in {(time.time() - t0) / 60:.1f} min; best test loss {best_test_loss:.5f}"
              f"{' (new best)' if is_best else ''}", flush=True)


if __name__ == "__main__":
    main()
