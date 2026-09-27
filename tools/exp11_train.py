"""exp_11 pretraining entry point: the pinned trainer's steps on the exp_11 registry.

Plan v3 item 6. Nothing of ``train_xRIR_backbone`` or of ``tools/exp06_train.py`` is
modified or monkey patched: the pinned ``seed_everything`` / ``seed_worker`` /
``train_epoch`` / ``test_epoch`` / ``save_checkpoint`` are called, and exp_06's data,
geometry and held-out inventories are imported as they stand. What this module owns is a
parser whose ``--backbone`` ranges over ``BACKBONES_EXP11`` and whose ``--yaw-aug``
accepts 0 **and** 1 under exp_04's constraints (``--save-every 0``, no ``--resume``,
``epochs * batches < 2**20``), the ``exp11_*`` provenance class, and a
``registry_sha256`` over exp_11's own registry.

It writes ``provenance.json`` once, before the epoch loop, and never writes
``completion.json`` -- completion is the external finalizer's decision
(``tools/exp11_finalize.py``).

    python tools/exp11_train.py --backbone simple_oriented \
        --save-dir ckpt/exp11/pretrain/xRIR_simpor_8_shot/attempt_<UTC> --num-shot 8 \
        --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 --lr-gamma 0.1 \
        --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 --seed 0 --tf32 \
        --log-interval 50 --save-every 500 --epoch-ckpt-every 1 --run-type full
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import train_xRIR_backbone as trainer
from model.xrir_exp11_registry import BACKBONES_EXP11, build_xrir_exp11, registry_sha256
from tools import exp06_train, exp11_profiles, exp11_recipe, provenance
from tools.exp05_params import TIERS, count_parameters, tier_of
from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset
from utils.lr_scheduler import ExponentialLR

REPO = Path(__file__).resolve().parents[1]
ENTRY_MODULE = 'tools.exp11_train'
RUN_TYPES = ('full', 'smoke', 'probe')
DIAGNOSTIC_RUN_TYPES = ('smoke', 'probe')
ENV_KEYS = exp06_train.ENV_KEYS


def build_parser():
    """The trainer's flags and defaults, on the exp_11 registry, yaw augmentation allowed."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backbone', choices=sorted(BACKBONES_EXP11), required=True)
    p.add_argument('--save-dir', required=True)
    p.add_argument('--num-shot', type=int, default=8)
    p.add_argument('--max-len', type=int, default=9600)
    for key, value in TIERS['M'].items():
        p.add_argument('--vit-' + key.replace('_', '-'), type=int, default=value)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--decay-epochs', type=int, default=50)
    p.add_argument('--lr-gamma', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--accum-steps', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--yaw-aug', type=int, choices=(0, 1), default=0,
                   help='I_RECIPE trains with per-sample active yaw (exp_04 port)')
    p.add_argument('--yaw-aug-seed', type=int, default=None)
    p.add_argument('--yaw-aug-width', type=int, default=512)
    p.add_argument('--no-save', action='store_true', help='write nothing to save-dir')
    p.add_argument('--tf32', action='store_true')
    p.add_argument('--log-interval', type=int, default=20)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--epoch-ckpt-every', type=int, default=5)
    p.add_argument('--resume', default=None)
    p.add_argument('--max-train-batches', type=int, default=0)
    p.add_argument('--max-test-batches', type=int, default=0)
    p.add_argument('--test-subset', type=int, default=0)
    p.add_argument('--run-type', choices=RUN_TYPES, default='full')
    p.add_argument('--provenance-out', default=None)
    p.add_argument('--approved', default=None,
                   help='approved_digests.json this run is admitted under (required for full)')
    p.add_argument('--reviewed-commit', default=None)
    p.add_argument('--exploratory', action='store_true')
    return p


def parse_args(argv=None):
    """Parse and apply the trainer's post-conditions, including exp_04's saving rules."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.yaw_aug_seed = args.seed if args.yaw_aug_seed is None else args.yaw_aug_seed
    if args.yaw_aug and (args.resume is not None or args.save_every != 0):
        parser.error('--yaw-aug 1 requires --resume to be None and --save-every exactly 0')
    if args.provenance_out is not None and not args.no_save:
        parser.error('--provenance-out applies to --no-save runs')
    if args.exploratory and args.run_type == 'full':
        parser.error('--exploratory is a diagnostic mode; a full run must match the approvals')
    return args


def check_admission(args):
    """A confirmatory run always names the approvals it is admitted under."""
    if args.approved is not None:
        return args
    if args.run_type not in DIAGNOSTIC_RUN_TYPES:
        raise ValueError('a full run must name the approvals it is admitted under (--approved)')
    if not args.no_save:
        raise ValueError('a {} run that writes to its save-dir must name its approvals '
                         '(--approved), or run with --no-save'.format(args.run_type))
    return args


def check_counter(args, batches_per_epoch):
    """exp_04's bound, refused here as the pinned trainer refuses it."""
    if args.yaw_aug and args.epochs * batches_per_epoch >= exp11_recipe.YAW_COUNTER_LIMIT:
        raise ValueError('--yaw-aug 1 requires epochs * train_batches_per_epoch < 2**20')


def build_model_exp11(args):
    """Construct exactly as the trainer's build_model, on the exp_11 registry."""
    return build_xrir_exp11(args.backbone, args.num_shot,
                            **{key: getattr(args, 'vit_' + key) for key in TIERS['M']})


def prepare_args(args, model, fields, destination):
    """Record the derived fields and the six ``exp11_*`` provenance keys."""
    args.param_counts = (count_parameters(model) if model is not None
                         else dict(exp11_recipe.expected_param_counts(args.backbone)))
    tier = tier_of(vars(args))
    if tier != exp11_recipe.TIER:
        raise ValueError('exp_11 pretrains at tier {}, not {}'.format(exp11_recipe.TIER, tier))
    args.tier = tier
    args.exp11_run_type = args.run_type
    args.exp11_profile = exp11_recipe.select_profile(vars(args))
    args.exp11_registry_sha256 = fields.get('registry_sha256')
    args.exp11_source_closure_sha256 = fields.get('source_closures', {}).get(
        'training', {}).get('sha256')
    args.exp11_git_head = fields.get('git_state', {}).get('HEAD')
    args.exp11_provenance_path = destination
    for flag in ('run_type', 'provenance_out', 'approved', 'reviewed_commit', 'exploratory'):
        if hasattr(args, flag):
            delattr(args, flag)
    return args


def approvals_binding(approved):
    """Bind the approvals file's own bytes, so a mid-run edit is detectable."""
    if approved is None:
        return None
    value, identity = exp11_profiles.load_approved_digests(approved)
    return {'path': str(approved), 'sha256': identity['sha256'],
            'schema_version': value['schema_version']}


def orchestration_closures(repo, commit):
    """The launcher shell and the finalizer that decide this run, bound as exp_06 binds them."""
    launcher, launcher_digest = exp11_profiles.closure_of('launch_sh', repo, commit)
    finalizer, finalizer_digest = exp11_profiles.closure_of('finalize', repo, commit)
    return {'launcher': {'files': launcher, 'sha256': launcher_digest},
            'finalizer': {'entry_module': 'tools.exp11_finalize',
                          'files': finalizer, 'sha256': finalizer_digest}}


def provenance_fields(argv, run_type, identity=None, repo=REPO, approved=None,
                      reviewed_commit=None, exploratory=False, geometry=None, heldout=None):
    """Bind the import closure, HEAD, environment, data inventory and argv of one run."""
    state = provenance.git_state(repo)
    commit = state['HEAD'] if reviewed_commit is None else reviewed_commit
    records, digest = provenance.closure_record(
        provenance.source_closure(ENTRY_MODULE, repo), commit, repo)
    return dict(repo=str(repo), reviewed_commit=commit, run_type=run_type,
                data_root=exp06_train.resolve_data_root(),
                source_closures={'training': {'entry_module': ENTRY_MODULE,
                                              'files': records, 'sha256': digest}},
                orchestration_closures=orchestration_closures(repo, commit),
                code_digests=exp11_profiles.compute_code_digests(
                    repo, commit, keys=exp11_profiles.TRAINING_KEYS),
                approvals=approvals_binding(approved), exploratory=bool(exploratory),
                geometry_identity=geometry, test_wav_identity=heldout,
                registry_sha256=registry_sha256(), git_state=state,
                environment=provenance.environment(), train_data_identity=identity,
                command=list(argv))


def main(argv=None):
    """The trainer's main, step for step, with the exp_11 registry and provenance."""
    command = list(sys.argv[1:] if argv is None else argv)
    args = check_admission(parse_args(argv))
    trainer.seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    train_dataset = xRIR_Dataset(split='train', max_len=args.max_len, num_shot=args.num_shot)
    test_dataset = xRIR_Dataset(split='test', max_len=args.max_len, num_shot=args.num_shot)
    if args.test_subset and args.test_subset < len(test_dataset):
        idx = np.random.RandomState(args.seed).permutation(len(test_dataset))[: args.test_subset]
        test_dataset = Subset(test_dataset, sorted(idx.tolist()))
    print(f'train samples: {len(train_dataset)}  test samples: {len(test_dataset)}', flush=True)

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True,
                         worker_init_fn=trainer.seed_worker,
                         persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                             **loader_kwargs)
    args.train_batches_per_epoch = len(train_loader)
    check_counter(args, args.train_batches_per_epoch)
    args.env = {key: os.environ.get(key) for key in ENV_KEYS}
    model = build_model_exp11(args).cuda()

    destination = args.provenance_out if args.no_save else os.path.join(args.save_dir,
                                                                        'provenance.json')
    root = exp06_train.resolve_data_root()
    fields = provenance_fields(
        command, args.run_type,
        identity=exp06_train.data_identity(root) if destination else None,
        geometry=exp06_train.geometry_identity(root) if destination else None,
        heldout=exp06_train.heldout_wav_identity(root) if destination else None,
        approved=args.approved, reviewed_commit=args.reviewed_commit,
        exploratory=args.exploratory)
    prepare_args(args, model, fields, destination)
    fields['effective_args'] = vars(args)
    if destination:
        if not args.no_save:
            os.makedirs(args.save_dir, exist_ok=True)
        provenance.write_manifest(destination, fields)  # exclusive; an attempt writes once
    print('XRIR_RUNTIME_ARGS ' + json.dumps(vars(args), sort_keys=True, allow_nan=False),
          flush=True)
    if not args.no_save:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as stream:
            json.dump(vars(args), stream, indent=2)

    n_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2
    print(f'backbone: {args.backbone}  params: {n_params / 1e6:.2f}M  '
          f'model size: {size_mb:.1f}MB', flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, decay_epochs=args.decay_epochs, gamma=args.lr_gamma)
    start_epoch, best_test_loss = 1, float('inf')
    history_path = os.path.join(args.save_dir, 'history.jsonl')
    print(f'yaw_aug ENABLED W={args.yaw_aug_width} seed={args.yaw_aug_seed} '
          f'counter=(epoch-1)*{args.train_batches_per_epoch}+batch_idx'
          if args.yaw_aug else 'yaw_aug DISABLED', flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = trainer.train_epoch(model, train_loader, optimizer, scheduler, epoch,
                                         args, best_test_loss)
        test_loss = trainer.test_epoch(model, test_loader, epoch, args)
        if args.no_save:
            continue
        is_best = test_loss < best_test_loss
        if is_best:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pth'))
        if args.epoch_ckpt_every and epoch % args.epoch_ckpt_every == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'epoch_{epoch:03d}.pth'))
        trainer.save_checkpoint(os.path.join(args.save_dir, 'last.pth'), model, optimizer,
                                scheduler, epoch, 0, best_test_loss, args)
        with open(history_path, 'a') as stream:
            stream.write(json.dumps({'epoch': epoch, 'train_loss': train_loss,
                                     'test_loss': test_loss, 'best_test_loss': best_test_loss,
                                     'is_best': is_best,
                                     'lr': optimizer.param_groups[0]['lr'],
                                     'epoch_minutes': (time.time() - t0) / 60.0}) + '\n')
        print(f'epoch {epoch} done in {(time.time() - t0) / 60:.1f} min; best test loss '
              f'{best_test_loss:.5f}{" (new best)" if is_best else ""}', flush=True)


if __name__ == '__main__':
    main()
