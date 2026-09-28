"""exp_11 HAA fine-tuning entry point: exp_02's recipe, in a declared frame or with a cue.

Plan v3 item 9 and Codex round-2 change 4. ``tools/exp06_haa_finetune.py`` embeds
optimiser construction, epoch iteration, validation selection and checkpoint writing
inside its ``main``, so there is nothing importable to call; exp_11 therefore owns a
**faithful orchestration loop** (``fine_tune`` below) that is that loop step for step and
**imports the unchanged numerical helpers** -- ``train_xRIR_backbone.compute_loss``,
``sim_to_real.finetune_haa.evaluate``, ``eval_xRIR_backbone.load_model_state`` and the
pinned ``ExponentialLR``. They are the module-level defaults of the loop's keyword
arguments, so production always runs the pinned functions and only a CPU parity test
substitutes stubs. ``tests/test_exp11_haa_parity.py`` runs this loop and a transcription
of exp_06's on one tiny synthetic problem and compares their selection traces and the
bytes of the checkpoints they write, and it pins the source digest of exp_06's ``main``
so a change there fails rather than drifting silently.

Frames and cues:

* ``--heading-json-dir`` -- arms H/I: the whole scene is rotated into the heading frame by
  the pinned ``HeadingFrameDataset``, exactly as exp_06 does.
* ``--adapter-heading-json-dir`` -- arms J/K: the dataset stays in the **room** frame and
  the shared loudspeaker heading is installed in the model's azimuth adapter. Every
  participating room's bound record must declare the *same* heading; a mixed heading is
  refused, because the adapter installs one cue for every room.

    python tools/exp11_haa_finetune.py --backbone simple_oriented \
        --init ckpt/exp11/pretrain/xRIR_simpor_8_shot/final/epoch_012.pth \
        --rooms class_room hallway complex_room --heading-json-dir ckpt/exp06/heading \
        --save-dir ckpt/exp11/sim2real/simple_or/seed0/stage1 --seed 0 \
        --epochs 1000 --val-every 10 --tf32
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import types

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from eval_xRIR_backbone import load_model_state
from model.xRIR_simple_adapter import ADAPTER_KEYS
from model.xrir_exp11_registry import BACKBONES_EXP11, build_xrir_exp11, registry_sha256
from sim_to_real.finetune_haa import evaluate
from sim_to_real.haa_dataset import DEFAULT_ROOT, ROOMS
from tools import exp06_haa_finetune as legacy
from tools import provenance
from train_xRIR_backbone import compute_loss, seed_everything
from utils.lr_scheduler import ExponentialLR

REPO = Path(__file__).resolve().parents[1]
RUN_TYPE = 'exp11_haa_train'
ENTRY_MODULE = 'tools.exp11_haa_finetune'
HEADING_BACKBONES = ('cylindrical_oriented', 'simple_oriented')
ADAPTER_BACKBONE = 'simple_adapter'


def build_parser():
    """exp_06's flags and defaults, on the exp_11 registry, plus the adapter heading."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backbone', choices=sorted(BACKBONES_EXP11), required=True)
    p.add_argument('--init', required=True, help='checkpoint to start from')
    p.add_argument('--rooms', nargs='+', choices=ROOMS, required=True)
    p.add_argument('--val-rooms', nargs='+', choices=ROOMS, default=None)
    p.add_argument('--save-dir', required=True)
    p.add_argument('--num-shot', type=int, default=8)
    p.add_argument('--max-len', type=int, default=9600)
    p.add_argument('--depth-variant', choices=['default', 'repo', 'local'], default='default')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--decay-epochs', type=int, default=50)
    p.add_argument('--lr-gamma', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=0)
    p.add_argument('--accum-steps', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval-seed', type=int, default=0)
    p.add_argument('--val-every', type=int, default=5)
    p.add_argument('--val-batch-size', type=int, default=32)
    p.add_argument('--tf32', action='store_true')
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--root', default=DEFAULT_ROOT, help='HAA cache root (recorded resolved)')
    p.add_argument('--heading-json-dir', default=None,
                   help='arms H/I: rotate the scene into the heading frame')
    p.add_argument('--adapter-heading-json-dir', default=None,
                   help='arms J/K: keep the room frame and install the shared cue')
    p.add_argument('--job-spec', default=None)
    p.add_argument('--run-type', choices=(RUN_TYPE,), default=RUN_TYPE)
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def select_frame(args, root, rooms):
    """Fail-closed: every backbone that needs a cue must be given exactly one.

    Returns ``(frame, k_by_room, heading, adapter_heading, adapter_phi_deg)``. The
    heading frame rotates the scene; the adapter frame stays in the room frame and
    installs one shared heading, refusing a cohort whose records disagree.
    """
    if args.heading_json_dir and args.adapter_heading_json_dir:
        raise ValueError('a child conditions on the frame or on the adapter, never both')
    if args.adapter_heading_json_dir:
        if args.backbone != ADAPTER_BACKBONE:
            raise ValueError('--adapter-heading-json-dir applies to the {} backbone, not '
                             '{!r}'.format(ADAPTER_BACKBONE, args.backbone))
        _, records = legacy.load_headings(args.adapter_heading_json_dir, rooms, root)
        declared = sorted({record['phi_deg'] for record in records.values()})
        if len(declared) != 1:
            raise ValueError('the adapter installs one cue for every room, but these '
                             'records declare the headings {}'.format(declared))
        return 'room', None, None, records, float(declared[0])
    if args.backbone == ADAPTER_BACKBONE:
        raise ValueError('the {} backbone needs --adapter-heading-json-dir'.format(
            ADAPTER_BACKBONE))
    if args.heading_json_dir is None:
        if args.backbone in HEADING_BACKBONES:
            raise ValueError('--heading-json-dir is required for the {} backbone'.format(
                args.backbone))
        return 'room', None, None, None, None
    k_by_room, heading = legacy.load_headings(args.heading_json_dir, rooms, root)
    return 'heading', k_by_room, heading, None, None


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
                  exp11_registry_sha256=fields['registry_sha256'],
                  exp11_source_closure_sha256=fields['source_closures']['child']['sha256'])
    return record


def load_init(model, state):
    """The two strict modes of plan section 2.3, chosen by what the checkpoint carries."""
    if not isinstance(model, torch.nn.Module) or not hasattr(model, 'load_base_checkpoint'):
        model.load_state_dict(state, strict=True)
        return 'strict'
    if any(key in state for key in ADAPTER_KEYS):
        model.load_state_dict(state, strict=True)      # a later stage's adapter state
        return 'adapter'
    model.load_base_checkpoint(state)                  # arm A/E weights; adapter stays zero
    return 'base'


def prepare(args, command=(), repo=REPO):
    """Datasets, model and records, without touching a GPU or writing a file."""
    root = legacy.resolve_root(args)
    val_rooms = args.val_rooms or list(args.rooms)
    rooms = list(args.rooms) + list(val_rooms)
    frame, k_by_room, heading, adapter_heading, adapter_phi = select_frame(args, root, rooms)
    if not os.path.isfile(args.init):
        raise ValueError('missing init checkpoint: {}'.format(args.init))
    init_sha256 = provenance.sha256_file(args.init)
    train_dataset = legacy.build_dataset(list(args.rooms), 'train', args, root, k_by_room, None)
    val_dataset = legacy.build_dataset(list(val_rooms), 'val', args, root, k_by_room,
                                       args.eval_seed)
    model = build_xrir_exp11(args.backbone, args.num_shot)
    if adapter_phi is not None:
        model.set_heading(adapter_phi)
    identity = legacy.data_identity(root, rooms, args.depth_variant)
    fields = provenance_fields(command, identity, repo=repo)
    record = record_args(args, fields, frame=frame, heading=heading,
                         adapter_heading=adapter_heading, adapter_phi_deg=adapter_phi,
                         init_sha256=init_sha256, haa_root=root,
                         **legacy.job_binding(args.job_spec))
    return types.SimpleNamespace(
        root=root, frame=frame, heading=heading, adapter_heading=adapter_heading,
        adapter_phi_deg=adapter_phi, k_by_room=k_by_room, val_rooms=val_rooms,
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


def fine_tune(model, train_loader, val_loader, args, save_dir, context=None,
              loss_fn=compute_loss, evaluate_fn=evaluate, save=torch.save):
    """exp_06's fine-tuning loop, step for step, with the pinned helpers as defaults.

    The order is the one the parity test pins: epoch-0 validation and an immediate
    ``best.pth``; per batch a loss, a scaled backward and a step on the accumulation
    boundary or the last batch; ``scheduler.step()`` once per epoch after the batches;
    validation on ``epoch % val_every == 0`` or the final epoch; **strict** improvement
    selects a new ``best.pth``; ``last.pth`` and ``summary.json`` after the loop.
    """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, decay_epochs=args.decay_epochs, gamma=args.lr_gamma)
    history_path = os.path.join(save_dir, 'history.jsonl')
    open(history_path, 'w').close()
    val0 = evaluate_fn(model, val_loader)
    with open(history_path, 'a') as stream:
        stream.write(json.dumps({'epoch': 0, 'train_loss': None, 'val_loss': val0,
                                 'lr': args.lr}) + '\n')
    print(f'epoch 0 (init): val loss {val0:.5f}', flush=True)
    best_val, best_epoch = val0, 0
    save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tot, n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for bi, batch in enumerate(train_loader):
            loss, _, _ = loss_fn(model, batch)
            (loss / args.accum_steps).backward()
            if (bi + 1) % args.accum_steps == 0 or bi + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            tot += loss.item() * batch[1].shape[0]
            n += batch[1].shape[0]
        scheduler.step()
        train_loss = tot / max(n, 1)
        rec = {'epoch': epoch, 'train_loss': train_loss,
               'lr': optimizer.param_groups[0]['lr']}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val = evaluate_fn(model, val_loader)
            rec['val_loss'] = val
            if val < best_val:
                best_val, best_epoch = val, epoch
                save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
                rec['is_best'] = True
        with open(history_path, 'a') as stream:
            stream.write(json.dumps(rec) + '\n')
        if epoch % args.log_every == 0 or epoch == args.epochs:
            print(f'epoch {epoch}/{args.epochs}: train {train_loss:.5f}'
                  + (f'  val {rec["val_loss"]:.5f}' if 'val_loss' in rec else '')
                  + f'  best val {best_val:.5f} @ {best_epoch}  lr {rec["lr"]:.1e}'
                  f'  {(time.time() - t0) / 60:.1f} min', flush=True)
    save(model.state_dict(), os.path.join(save_dir, 'last.pth'))
    summary = {'best_epoch': best_epoch, 'best_val_loss': best_val, 'init_val_loss': val0,
               'epochs': args.epochs, 'minutes': (time.time() - t0) / 60,
               'rooms': list(args.rooms),
               'val_rooms': list(getattr(context, 'val_rooms', args.rooms)),
               'init': args.init, 'backbone': args.backbone, 'seed': args.seed,
               'frame': getattr(context, 'frame', 'room'),
               'adapter_phi_deg': getattr(context, 'adapter_phi_deg', None)}
    with open(os.path.join(save_dir, 'summary.json'), 'w') as stream:
        json.dump(summary, stream, indent=2)
    return summary


def main(argv=None):
    """exp_06's main, step for step, with the exp_11 registry, records and cue routing."""
    command = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    context = prepare(args, command)
    write_records(args, context)
    train_ds, val_ds = context.train_dataset, context.val_dataset
    bs = context.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0,
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=0)
    print(f'rooms {args.rooms}: {len(train_ds)} train targets, {len(val_ds)} val samples '
          f'from {context.val_rooms}; frame {context.frame}; adapter heading '
          f'{context.adapter_phi_deg}; batch {bs} x accum {args.accum_steps}; '
          f'{len(train_loader)} batches/epoch', flush=True)
    model = context.model
    mode = load_init(model, load_model_state(args.init))
    print('init loaded in {} mode'.format(mode), flush=True)
    model.cuda().train()
    summary = fine_tune(model, train_loader, val_loader, args, args.save_dir, context)
    print('done:', json.dumps(dict(summary, init_mode=mode)), flush=True)


if __name__ == '__main__':
    main()
