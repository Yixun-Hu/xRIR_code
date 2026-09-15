"""exp_06 pretraining entry point: the pinned trainer's steps on the exp_06 registry.

Plan v4 sections 5 and 8. Nothing of ``train_xRIR_backbone`` is modified or monkey
patched: this module owns a parser whose shared flags and defaults equal the trainer's
(``--backbone`` ranges over ``BACKBONES_EXP06`` and ``--yaw-aug`` accepts only 0) and a
``main`` that calls the trainer's ``seed_everything``, ``seed_worker``, ``train_epoch``,
``test_epoch`` and ``save_checkpoint``, writing the same ``best.pth`` / ``epoch_NNN.pth``
/ ``last.pth`` / ``history.jsonl`` / ``args.json`` files.

It writes ``provenance.json`` once, before the epoch loop, and never writes
``completion.json`` -- completion is the external finalizer's decision
(``tools/exp06_finalize.py``).

    python tools/exp06_train.py --backbone cylindrical_oriented \
        --save-dir ckpt/exp06/pretrain/xRIR_cylor_8_shot/attempt_<UTC> --num-shot 8 \
        --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 --lr-gamma 0.1 \
        --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 --seed 0 --tf32 \
        --log-interval 50 --save-every 500 --epoch-ckpt-every 1 --run-type full
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import train_xRIR_backbone as trainer
from model.xRIR_cyl_oriented import BACKBONES_EXP06, build_xrir_exp06
from tools import exp06_profiles, exp06_recipe
from tools import provenance
from tools.exp05_params import TIERS, count_parameters, tier_of
from treble_multi_room_dataset.treble_xRIR_dataset import BASE_DATA_PATH, xRIR_Dataset
from utils.lr_scheduler import ExponentialLR

RUN_TYPES = ('full', 'smoke', 'probe')
REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = BASE_DATA_PATH  # exactly what the dataset module resolved; never a second fallback
TRAIN_INVENTORY = REPO / 'ckpt/yaw_aug/train_inventory.json'
ENV_KEYS = ('PYTHONHASHSEED', 'XRIR_DATA_PATH', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')
GEOMETRY_SPLITS = ('train', 'test')
GEOMETRY_WORKERS = 8


def build_parser():
    """The trainer's flags and defaults, restricted to the exp_06 registry and yaw_aug 0."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", choices=sorted(BACKBONES_EXP06), required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-shot", type=int, default=8)
    p.add_argument("--max-len", type=int, default=9600)
    for key, value in TIERS['M'].items():
        p.add_argument('--vit-' + key.replace('_', '-'), type=int, default=value)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--decay-epochs", type=int, default=50)
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--accum-steps", type=int, default=1,
                   help="gradient accumulation; effective batch = batch-size * accum-steps")
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--yaw-aug", type=int, choices=(0,), default=0,
                   help="exp_06 trains without yaw augmentation; recorded and inert")
    p.add_argument("--yaw-aug-seed", type=int, default=None)
    p.add_argument("--yaw-aug-width", type=int, default=512)
    p.add_argument("--no-save", action="store_true", help="write nothing to save-dir (smoke runs)")
    p.add_argument("--tf32", action="store_true", help="allow TF32 matmul/cudnn (faster, slightly different numerics)")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500, help="save last.pth every N train batches (0 = only per epoch)")
    p.add_argument("--epoch-ckpt-every", type=int, default=5,
                   help="also keep a standalone epoch_NNN.pth every N epochs (0 = never); best.pth/last.pth are always written")
    p.add_argument("--resume", default=None, help="path to a last.pth to resume from")
    p.add_argument("--max-train-batches", type=int, default=0, help="0 = full epoch")
    p.add_argument("--max-test-batches", type=int, default=0, help="0 = full test split")
    p.add_argument("--test-subset", type=int, default=0,
                   help="evaluate on a fixed seeded subset of N test samples each epoch (0 = all)")
    p.add_argument("--run-type", choices=RUN_TYPES, default='full',
                   help="recorded in provenance.json; the finalizer dispatches on it")
    p.add_argument("--provenance-out", default=None,
                   help="where a --no-save run writes provenance.json (default: nowhere)")
    p.add_argument("--approved", default=None,
                   help="approved_digests.json this run is admitted under (required for --run-type full)")
    p.add_argument("--reviewed-commit", default=None,
                   help="the commit the launcher's preflight verified HEAD against (default: HEAD)")
    p.add_argument("--exploratory", action="store_true",
                   help="diagnostic run outside the approvals; never admissible as an arm")
    return p


def parse_args(argv=None):
    """Parse and apply the trainer's post-conditions on the yaw seed."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.yaw_aug_seed = args.seed if args.yaw_aug_seed is None else args.yaw_aug_seed
    if args.provenance_out is not None and not args.no_save:
        parser.error("--provenance-out applies to --no-save runs; saving runs write it to --save-dir")
    if args.exploratory and args.run_type == 'full':
        parser.error("--exploratory is a diagnostic mode; a full run must match the approvals")
    return args


def build_model_exp06(args):
    """Construct exactly as the trainer's build_model, on the exp_06 registry."""
    return build_xrir_exp06(args.backbone, args.num_shot,
                            **{key: getattr(args, 'vit_' + key) for key in TIERS['M']})


def prepare_args(args, model, fields, destination):
    """Record the derived tier/parameter counts and the five exp_06 provenance keys.

    ``run_type`` and ``provenance_out`` leave the namespace: ``args.json`` then holds
    exactly the trainer's 33 fields plus the ``exp06_*`` class of ``tools.exp06_recipe``.
    Capacities outside the recipe's tier are refused rather than recorded as ``custom``.
    """
    args.param_counts = (count_parameters(model) if model is not None
                         else dict(exp06_recipe.expected_param_counts(args.backbone)))
    tier = tier_of(vars(args))
    if tier != exp06_recipe.TIER:
        raise ValueError('exp_06 pretrains at tier {}, not {}'.format(exp06_recipe.TIER, tier))
    args.tier = tier
    args.exp06_run_type = args.run_type
    args.exp06_registry_sha256 = fields.get('registry_sha256')
    args.exp06_source_closure_sha256 = fields.get('source_closures', {}).get('training', {}).get('sha256')
    args.exp06_git_head = fields.get('git_state', {}).get('HEAD')
    args.exp06_provenance_path = destination
    for flag in ('run_type', 'provenance_out', 'approved', 'reviewed_commit', 'exploratory'):
        if hasattr(args, flag):
            delattr(args, flag)
    return args


def resolve_data_root():
    """The one root the dataset will actually read; a missing one is refused up front."""
    root = os.path.realpath(DATA_ROOT)
    if not os.path.isdir(root):
        raise ValueError('data root does not exist: {} (set XRIR_DATA_PATH)'.format(root))
    return root


def geometry_paths(data_root, splits=GEOMETRY_SPLITS, max_len=9600, num_shot=8):
    """Every metadata JSON and depth map the pinned dataset reads for these splits.

    Finding 2: ``train_data_identity`` inventories the IR waveforms only, so a changed
    source position or panorama was not bound to the run. Membership comes from the
    dataset's own split logic -- ``xRIR_Dataset.file_list`` -- and from what
    ``__getitem__`` reads for each entry: that pair's ``metadata/.../S00s_R00r.json``
    and the receiver's ``depth_map/.../r.npy``. The references drawn by
    ``get_ir_and_location_for_other_sources`` are other sources at the *same* receiver,
    every one of them an IR of the same room directory and so already in the split's
    file list, which is why one metadata file per IR covers the reference geometry too.

    Returns ``(sorted relative paths, {split: number of IRs})``.
    """
    root = Path(data_root).resolve()
    files, counts = set(), {}
    for split in splits:
        dataset = xRIR_Dataset(split=split, max_len=max_len, num_shot=num_shot,
                               ir_path=str(root / 'single_channel_ir'),
                               pano_depth_path=str(root / 'depth_map'),
                               metadata_path=str(root / 'metadata'))
        counts[split] = len(dataset.file_list)
        for wav in dataset.file_list:
            parts = Path(wav).parts
            category, room, name = parts[-3], parts[-2], parts[-1]
            source, receiver = [int(token[1:]) for token in name.split('_')[:2]]
            files.add('metadata/{}/{}/S00{}_R00{}.json'.format(category, room, source, receiver))
            files.add('depth_map/{}/{}/{}.npy'.format(category, room, receiver))
    return sorted(files), counts


def geometry_identity(data_root, splits=GEOMETRY_SPLITS, workers=GEOMETRY_WORKERS):
    """Content-hash every geometry input of the run, in the inventory shape of provenance."""
    files, counts = geometry_paths(data_root, splits)
    record = provenance._inventory(files, data_root, workers=workers)
    return dict(record, splits=list(splits), split_files=counts,
                metadata_files=sum(1 for name in files if name.startswith('metadata/')),
                depth_maps=sum(1 for name in files if name.startswith('depth_map/')))


def registry_sha256():
    """Digest of the backbone registry as a sorted name -> class path mapping."""
    mapping = {name: cls.__module__ + '.' + cls.__qualname__ for name, cls in BACKBONES_EXP06.items()}
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()


def data_identity(data_root, cache_path=TRAIN_INVENTORY):
    """Reuse exp_04's stat-validated inventory read-only; never create it under ckpt/."""
    if Path(cache_path).is_file():
        return provenance.train_data_identity(str(data_root), cache_path=str(cache_path))
    with tempfile.TemporaryDirectory() as scratch:
        return provenance.train_data_identity(str(data_root),
                                              cache_path=str(Path(scratch) / 'train_inventory.json'))


def approvals_binding(approved):
    """Finding 1: bind the approvals file's own bytes, so a mid-run edit is detectable."""
    if approved is None:
        return None
    value, identity = exp06_profiles.load_approved_digests(approved)
    return {'path': str(approved), 'sha256': identity['sha256'],
            'schema_version': value['schema_version']}


def orchestration_closures(repo, commit):
    """Finding 1: the launcher shell and the finalizer decide this run; bind them too.

    ``tools/exp06_launch.sh`` has no import closure and is bound as a file, exp_04's
    pattern; the finalizer is bound with its whole import closure, because a later round
    may change it while a 31-hour training run is still going.
    """
    launcher, launcher_digest = exp06_profiles.closure_of('launch_sh', repo, commit)
    finalizer, finalizer_digest = exp06_profiles.closure_of('finalize', repo, commit)
    return {'launcher': {'files': launcher, 'sha256': launcher_digest},
            'finalizer': {'entry_module': 'tools.exp06_finalize',
                          'files': finalizer, 'sha256': finalizer_digest}}


def provenance_fields(argv, run_type, identity=None, repo=REPO, approved=None,
                      reviewed_commit=None, exploratory=False, geometry=None):
    """Bind the import closure, HEAD, environment, data inventory and argv of one run."""
    state = provenance.git_state(repo)
    commit = state['HEAD'] if reviewed_commit is None else reviewed_commit
    records, digest = provenance.closure_record(
        provenance.source_closure('tools.exp06_train', repo), commit, repo)
    return dict(repo=str(repo), reviewed_commit=commit, run_type=run_type,
                data_root=resolve_data_root(),
                source_closures={'training': {'entry_module': 'tools.exp06_train',
                                              'files': records, 'sha256': digest}},
                orchestration_closures=orchestration_closures(repo, commit),
                code_digests=exp06_profiles.compute_code_digests(
                    repo, commit, keys=exp06_profiles.TRAINING_KEYS),
                approvals=approvals_binding(approved), exploratory=bool(exploratory),
                geometry_identity=geometry,
                registry_sha256=registry_sha256(), git_state=state,
                environment=provenance.environment(), train_data_identity=identity,
                command=list(argv))


def provenance_destination(args):
    """Saving runs record beside the checkpoints; --no-save runs only where asked."""
    return args.provenance_out if args.no_save else os.path.join(args.save_dir, 'provenance.json')


def main(argv=None):
    """The trainer's main, step for step, with the exp_06 registry and provenance."""
    command = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.run_type == 'full' and args.approved is None:
        raise ValueError('a full run must name the approvals it is admitted under (--approved)')
    trainer.seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    train_dataset = xRIR_Dataset(split="train", max_len=args.max_len, num_shot=args.num_shot)
    test_dataset = xRIR_Dataset(split="test", max_len=args.max_len, num_shot=args.num_shot)
    if args.test_subset and args.test_subset < len(test_dataset):
        idx = np.random.RandomState(args.seed).permutation(len(test_dataset))[: args.test_subset]
        test_dataset = Subset(test_dataset, sorted(idx.tolist()))
    print(f"train samples: {len(train_dataset)}  test samples: {len(test_dataset)}", flush=True)

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True,
                         worker_init_fn=trainer.seed_worker, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, **loader_kwargs)
    args.train_batches_per_epoch = len(train_loader)
    args.env = {key: os.environ.get(key) for key in ENV_KEYS}
    model = build_model_exp06(args).cuda()

    destination = provenance_destination(args)
    root = resolve_data_root()
    fields = provenance_fields(command, args.run_type,
                               identity=data_identity(root) if destination else None,
                               geometry=geometry_identity(root) if destination else None,
                               approved=args.approved, reviewed_commit=args.reviewed_commit,
                               exploratory=args.exploratory)
    prepare_args(args, model, fields, destination)
    fields['effective_args'] = vars(args)
    if destination:
        if not args.no_save:
            os.makedirs(args.save_dir, exist_ok=True)
        provenance.write_manifest(destination, fields)  # exclusive; an attempt is written once
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
        start_epoch = ckpt["epoch"] if ckpt.get("batch_idx", 0) else ckpt["epoch"] + 1
        print(f"resumed from {args.resume}: epoch {ckpt['epoch']} batch {ckpt.get('batch_idx', 0)} "
              f"best_test_loss {best_test_loss:.5f} -> starting epoch {start_epoch}", flush=True)

    history_path = os.path.join(args.save_dir, "history.jsonl")
    print("yaw_aug DISABLED", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_epoch(model, train_loader, optimizer, scheduler, epoch, args, best_test_loss)
        test_loss = trainer.test_epoch(model, test_loader, epoch, args)
        if args.no_save:
            continue
        is_best = test_loss < best_test_loss
        if is_best:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pth"))
        if args.epoch_ckpt_every and epoch % args.epoch_ckpt_every == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"epoch_{epoch:03d}.pth"))
        trainer.save_checkpoint(os.path.join(args.save_dir, "last.pth"), model, optimizer, scheduler,
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
