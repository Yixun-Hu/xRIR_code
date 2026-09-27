"""exp_11 HAA evaluation entry point: exp_02's Table-2 protocol, in a frame or with a cue.

Plan v3 item 9. The per-room evaluation loop **is importable** and is imported unchanged
from ``tools/exp06_haa_eval.py`` -- ``evaluate_room`` (the pinned ``Evaluator``,
``griffin_lim``, spectrogram losses and room-frame ``side_label``), ``room_summary`` and
``write_room_outputs`` -- so the numbers exp_11 publishes come from the very code exp_06
and exp_09 published theirs from. What this module owns is the exp_11 parser, the cue
routing (``--heading-json-dir`` for arms H/I, ``--adapter-heading-json-dir`` for J/K) and
the records the exp_11 finalizer verifies: ``provenance.json``, an ``args.json`` that is
the same mapping, and a per-sample ``meta`` carrying the frame, the bound cue and the
checkpoint digest.

    python tools/exp11_haa_eval.py --backbone simple_adapter \
        --checkpoint ckpt/exp11/sim2real/control_adapter/seed0/stage2_hallway/best.pth \
        --rooms hallway --adapter-heading-json-dir ckpt/exp06/heading \
        --save-dir ckpt/exp11/sim2real/control_adapter/seed0/eval/hallway
"""
import argparse
import json
import os
from pathlib import Path
import sys
import types

import torch

from eval_unseen import Evaluator
from eval_xRIR_backbone import load_model_state
from model.xrir_exp11_registry import BACKBONES_EXP11, build_xrir_exp11
from sim_to_real.haa_dataset import DEFAULT_ROOT, ROOMS
from tools import exp06_haa_eval as legacy_eval
from tools import exp11_haa_finetune as records
from tools import provenance

RUN_TYPE = 'exp11_haa_eval'
ENTRY_MODULE = 'tools.exp11_haa_eval'
REPO = records.REPO
PINNED_META = legacy_eval.PINNED_META


def build_parser():
    """exp_02's evaluation flags and defaults, on the exp_11 registry, plus the cue routing."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backbone', choices=sorted(BACKBONES_EXP11), required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--rooms', nargs='+', choices=ROOMS, default=list(ROOMS))
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--save-dir', required=True)
    p.add_argument('--num-shot', type=int, default=8)
    p.add_argument('--max-len', type=int, default=9600)
    p.add_argument('--eval-seed', type=int, default=0)
    p.add_argument('--depth-variant', choices=['default', 'repo', 'local'], default='default')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-samples', type=int, default=0, help='per room, first N (0 = all)')
    p.add_argument('--tag', default='', help='suffix for output files, e.g. _val')
    p.add_argument('--root', default=DEFAULT_ROOT, help='HAA cache root (recorded resolved)')
    p.add_argument('--heading-json-dir', default=None,
                   help='arms H/I: evaluate in the heading frame')
    p.add_argument('--adapter-heading-json-dir', default=None,
                   help='arms J/K: the room frame with the shared cue installed')
    p.add_argument('--seed', type=int, default=0,
                   help='the pipeline seed this child belongs to; never the reference draw')
    p.add_argument('--gl-seed-per-query', action='store_true',
                   help='S1 sensitivity: seed Griffin-Lim per query instead of leaving it free')
    p.add_argument('--job-spec', default=None,
                   help='the pipeline declaration this child is launched under (recorded)')
    p.add_argument('--run-type', choices=(RUN_TYPE,), default=RUN_TYPE)
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def prepare(args, command=(), repo=REPO):
    """Datasets, model and records, without touching a GPU or writing a file."""
    root = records.legacy.resolve_root(args)
    frame, k_by_room, heading, adapter_heading, adapter_phi = records.select_frame(
        args, root, list(args.rooms))
    if not os.path.isfile(args.checkpoint):
        raise ValueError('missing evaluation checkpoint: {}'.format(args.checkpoint))
    checkpoint_sha256 = provenance.sha256_file(args.checkpoint)
    datasets = {room: records.legacy.build_dataset([room], args.split, args, root, k_by_room,
                                                   args.eval_seed) for room in args.rooms}
    model = build_xrir_exp11(args.backbone, args.num_shot)
    if adapter_phi is not None:
        model.set_heading(adapter_phi)
    identity = records.legacy.data_identity(root, list(args.rooms), args.depth_variant)
    fields = records.provenance_fields(command, identity, repo=repo, run_type=RUN_TYPE,
                                       entry=ENTRY_MODULE)
    record = records.record_args(args, fields, frame=frame, heading=heading,
                                 adapter_heading=adapter_heading, adapter_phi_deg=adapter_phi,
                                 haa_root=root, checkpoint_sha256=checkpoint_sha256,
                                 **records.legacy.job_binding(args.job_spec))
    return types.SimpleNamespace(root=root, frame=frame, heading=heading,
                                 adapter_heading=adapter_heading, adapter_phi_deg=adapter_phi,
                                 datasets=datasets, model=model, fields=fields,
                                 args_record=record, checkpoint_sha256=checkpoint_sha256)


def per_sample_meta(args, context, room):
    """exp_06's own meta keys, plus every field the exp_11 finalizer binds to this arm."""
    meta = {key: getattr(args, key) for key in PINNED_META}
    meta.update(room=room, frame=context.frame, heading=context.heading,
                adapter_heading=context.adapter_heading,
                adapter_phi_deg=context.adapter_phi_deg,
                checkpoint_sha256=context.checkpoint_sha256,
                gl_seed_per_query=args.gl_seed_per_query, run_type=RUN_TYPE,
                registry_sha256=context.fields['registry_sha256'],
                git_head=context.fields['git_state']['HEAD'],
                exp11_source_closure_sha256=context.fields['source_closures']['child']['sha256'])
    return meta


def main(argv=None):
    """exp_06's evaluation main, with exp_11's records written before the loop."""
    command = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    context = prepare(args, command)
    records.write_records(args, context)
    model = context.model
    records.load_init(model, load_model_state(args.checkpoint))
    model.cuda().eval()
    evaluator = Evaluator()
    all_summaries = {}
    for room in args.rooms:
        meta = per_sample_meta(args, context, room)
        per, counts, elapsed = legacy_eval.evaluate_room(model, evaluator, args, context, room)
        summary = legacy_eval.room_summary(args, room, per, counts, meta, elapsed)
        all_summaries[room] = summary
        legacy_eval.write_room_outputs(args, room, summary, per, meta)
        t60s = ('{:.2f}%'.format(summary['t60_error_pct']['mean'])
                if summary['t60_error_pct'] else '-')
        print('{:14s} n={:4d}  EDT {:.4f}s  C50 {:.3f}dB  T60 {}  env {:.2f}  loss {:.4f}  '
              '({:.1f} min)'.format(room, summary['n_samples'],
                                    summary['edt_error_s']['mean'],
                                    summary['c50_error_db']['mean'], t60s,
                                    summary['env_error']['mean'],
                                    summary['test_loss']['mean'], elapsed), flush=True)
    with open(os.path.join(args.save_dir, 'metrics_all{}.json'.format(args.tag)), 'w') as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == '__main__':
    main()
