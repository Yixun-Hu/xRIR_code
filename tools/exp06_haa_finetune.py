"""exp_06 HAA fine-tuning entry point: exp_02's recipe, run in a declared frame.

Plan v4 sections 6.2 and 8. Nothing of ``sim_to_real/finetune_haa.py`` is modified: its
``evaluate`` and the pinned ``compute_loss`` / ``load_model_state`` / ``ExponentialLR``
are imported, and the epoch loop below is that module's loop step for step. What this
module owns is the exp_06 parser (``--backbone`` over ``BACKBONES_EXP06``, an optional
``--heading-json-dir``, an explicit ``--root`` for the HAA cache) and the records the
external finalizer (``tools/exp06_finalize.py``) verifies: ``provenance.json`` written
once before the loop, and an ``args.json`` that is the same mapping, carrying the frame,
the per-room heading binding, ``init_sha256``, the resolved cache root and the digests of
the execution record.

Fail-closed: the oriented backbone may not run without a heading directory, every room's
heading record must be decided, ``confirmatory`` and still hash the cache files it was
estimated from, and a cache root that does not exist is refused before anything is built.

    python tools/exp06_haa_finetune.py --backbone cylindrical_oriented \
        --init ckpt/exp06/pretrain/xRIR_cylor_8_shot/final/epoch_012.pth \
        --rooms class_room hallway complex_room --heading-json-dir ckpt/exp06/heading \
        --save-dir ckpt/exp06/sim2real/cyl_or/seed0/stage1 --seed 0 \
        --epochs 1000 --val-every 10 --tf32
"""
import argparse
import hashlib
import json
import os
import sys
import time
import types

import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader

from eval_xRIR_backbone import load_model_state
from model.xRIR_cyl_oriented import BACKBONES_EXP06, build_xrir_exp06
from sim_to_real.finetune_haa import evaluate
from sim_to_real.haa_dataset import DEFAULT_ROOT, DEPTH_FILE, ROOMS, HAADataset
from tools import exp06_heading
from tools import provenance
from train_xRIR_backbone import compute_loss, seed_everything
from utils.lr_scheduler import ExponentialLR

REPO = Path(__file__).resolve().parents[1]
RUN_TYPE = 'haa_train'
ENTRY_MODULE = 'tools.exp06_haa_finetune'
ORIENTED = 'cylindrical_oriented'
USABLE_HEADINGS = ('estimated', 'override')
CACHE_FILES = ('meta.json', 'rirs.npy', 'xyzs.npy', 'speaker_xyz.npy')


def build_parser():
    """sim_to_real/finetune_haa.py's flags and defaults, on the exp_06 registry."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=sorted(BACKBONES_EXP06), required=True)
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
    # exp_06 additions.
    p.add_argument("--root", default=DEFAULT_ROOT, help="HAA cache root (recorded resolved)")
    p.add_argument("--heading-json-dir", default=None,
                   help="one <room>.json per room; required for the oriented backbone")
    p.add_argument("--run-type", choices=(RUN_TYPE,), default=RUN_TYPE,
                   help="recorded in provenance.json; the finalizer dispatches on it")
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def registry_sha256():
    """The digest of the backbone registry, defined exactly as tools.exp06_train does."""
    mapping = {name: cls.__module__ + '.' + cls.__qualname__
               for name, cls in BACKBONES_EXP06.items()}
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()


def load_headings(heading_dir, rooms, root):
    """Every room's decided, confirmatory record, re-verified against the cache it read.

    ``read_heading_json(path, room_dir)`` rehashes the four cache inputs the estimate was
    derived from, so a cache edited after estimation is refused here rather than silently
    rotating a different room.
    """
    k_by_room, records = {}, {}
    for room in sorted(set(rooms)):
        path = Path(heading_dir) / (room + '.json')
        if not path.is_file():
            raise ValueError('missing heading json for {}: {}'.format(room, path))
        try:
            record = exp06_heading.read_heading_json(str(path), room_dir=str(Path(root) / room))
        except (OSError, ValueError) as error:
            raise ValueError('unusable heading json for {}: {}'.format(room, error)) from error
        if record['room'] != room:
            raise ValueError('heading json {} records the room {!r}'.format(path, record['room']))
        if record['decision'] not in USABLE_HEADINGS:
            raise ValueError('heading json for {} records the decision {!r}'.format(
                room, record['decision']))
        if record['admissibility'] != 'confirmatory':
            raise ValueError('heading json for {} is {}, not confirmatory'.format(
                room, record['admissibility']))
        k_by_room[room] = record['k']
        records[room] = {'phi_deg': record['phi_deg'], 'k': record['k'],
                         'decision': record['decision'], 'sha256': provenance.sha256_file(path),
                         'path': str(path)}
    return k_by_room, records


def data_identity(root, rooms, depth_variant):
    """Content-hash exactly the cache files the HAA datasets read for these rooms."""
    files = [os.path.join(room, name) for room in sorted(set(rooms))
             for name in CACHE_FILES + (DEPTH_FILE[depth_variant],)]
    return provenance._inventory(files, root)


def provenance_fields(command, identity, repo=REPO, run_type=RUN_TYPE, entry=ENTRY_MODULE):
    """Bind the import closure, HEAD, environment, cache inventory and argv of one child."""
    state = provenance.git_state(repo)
    records, digest = provenance.closure_record(
        provenance.source_closure(entry, repo), state['HEAD'], repo)
    return dict(repo=str(repo), reviewed_commit=state['HEAD'], run_type=run_type,
                source_closures={'child': {'entry_module': entry, 'files': records,
                                           'sha256': digest}},
                registry_sha256=registry_sha256(), git_state=state,
                environment=provenance.environment(), command=list(command),
                data_identity=identity)


def record_args(args, fields, **derived):
    """args.json = the parsed namespace plus what binds this run to its own record."""
    record = dict(vars(args))
    record.update(derived, git_head=fields['git_state']['HEAD'],
                  exp06_registry_sha256=fields['registry_sha256'],
                  exp06_source_closure_sha256=fields['source_closures']['child']['sha256'])
    return record


def resolve_root(args):
    root = os.path.realpath(args.root)
    if not os.path.isdir(root):
        raise ValueError('HAA cache root does not exist: {} (--root)'.format(root))
    return root


def select_frame(args, root):
    """Fail-closed: the oriented backbone is meaningless without a declared heading."""
    if args.heading_json_dir is None:
        if args.backbone == ORIENTED:
            raise ValueError('--heading-json-dir is required for the {} backbone'.format(ORIENTED))
        return 'room', None, None
    rooms = list(args.rooms) + list(args.val_rooms or args.rooms)
    k_by_room, heading = load_headings(args.heading_json_dir, rooms, root)
    return 'heading', k_by_room, heading


def build_dataset(rooms, split, args, root, k_by_room, eval_seed):
    """The pinned dataset, or its heading-frame subclass; never a second implementation."""
    if k_by_room is None:
        return HAADataset(rooms, split, root=root, num_shot=args.num_shot,
                          max_len=args.max_len, eval_seed=eval_seed,
                          depth_variant=args.depth_variant)
    return exp06_heading.HeadingFrameDataset(
        rooms, split, root=root, num_shot=args.num_shot, max_len=args.max_len,
        eval_seed=eval_seed, depth_variant=args.depth_variant,
        k_by_room={room: k_by_room[room] for room in rooms})


def prepare(args, command=()):
    """Datasets, model and records, without touching a GPU or writing a file."""
    root = resolve_root(args)
    frame, k_by_room, heading = select_frame(args, root)
    val_rooms = args.val_rooms or list(args.rooms)
    if not os.path.isfile(args.init):
        raise ValueError('missing init checkpoint: {}'.format(args.init))
    init_sha256 = provenance.sha256_file(args.init)
    train_dataset = build_dataset(list(args.rooms), 'train', args, root, k_by_room, None)
    val_dataset = build_dataset(list(val_rooms), 'val', args, root, k_by_room, args.eval_seed)
    model = build_xrir_exp06(args.backbone, args.num_shot)
    identity = data_identity(root, list(args.rooms) + list(val_rooms), args.depth_variant)
    fields = provenance_fields(command, identity)
    record = record_args(args, fields, frame=frame, heading=heading, init_sha256=init_sha256,
                         haa_root=root)
    return types.SimpleNamespace(
        root=root, frame=frame, heading=heading, k_by_room=k_by_room, val_rooms=val_rooms,
        train_dataset=train_dataset, val_dataset=val_dataset, model=model, fields=fields,
        args_record=record, init_sha256=init_sha256,
        batch_size=args.batch_size or len(train_dataset))


def write_records(args, context):
    """provenance.json is written once, exclusively; args.json is the same mapping."""
    os.makedirs(args.save_dir, exist_ok=True)
    fields = dict(context.fields, effective_args=context.args_record)
    digest = provenance.write_manifest(os.path.join(args.save_dir, 'provenance.json'), fields)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as stream:
        json.dump(context.args_record, stream, indent=2)
    return digest


def main(argv=None):
    """sim_to_real/finetune_haa.py's main, step for step, with the exp_06 records."""
    command = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    context = prepare(args, command)
    write_records(args, context)
    train_ds, val_ds = context.train_dataset, context.val_dataset
    bs = context.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False, num_workers=0)
    print(f"rooms {args.rooms}: {len(train_ds)} train targets, {len(val_ds)} val samples from "
          f"{context.val_rooms}; frame {context.frame}; batch {bs} x accum {args.accum_steps}; "
          f"{len(train_loader)} batches/epoch", flush=True)

    model = context.model
    model.load_state_dict(load_model_state(args.init), strict=True)
    model.cuda().train()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, decay_epochs=args.decay_epochs, gamma=args.lr_gamma)

    history_path = os.path.join(args.save_dir, "history.jsonl")
    open(history_path, "w").close()
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
                  + f"  best val {best_val:.5f} @ {best_epoch}  lr {rec['lr']:.1e}"
                  f"  {(time.time() - t0) / 60:.1f} min", flush=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, "last.pth"))
    summary = {"best_epoch": best_epoch, "best_val_loss": best_val, "init_val_loss": val0,
               "epochs": args.epochs, "minutes": (time.time() - t0) / 60, "rooms": args.rooms,
               "val_rooms": context.val_rooms, "init": args.init, "backbone": args.backbone,
               "seed": args.seed, "frame": context.frame}
    with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("done:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
