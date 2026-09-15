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
import hashlib
import json
import os
import types
from pathlib import Path

from model.xRIR_cyl_oriented import BACKBONES_EXP06, build_xrir_exp06
from sim_to_real.haa_dataset import DEFAULT_ROOT, ROOMS
from tools import exp06_haa_finetune as records
from tools import provenance

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
    p.add_argument("--run-type", choices=(RUN_TYPE,), default=RUN_TYPE,
                   help="recorded in provenance.json; the finalizer dispatches on it")
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def prepare(args, command=()):
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
    fields = records.provenance_fields(command, identity, run_type=RUN_TYPE, entry=ENTRY_MODULE)
    record = records.record_args(args, fields, frame=frame, heading=heading, haa_root=root,
                                 checkpoint_sha256=checkpoint_sha256)
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
