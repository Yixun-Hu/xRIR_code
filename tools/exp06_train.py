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

from model.xRIR_cyl_oriented import BACKBONES_EXP06, build_xrir_exp06
from tools import exp06_recipe
from tools.exp05_params import TIERS, count_parameters, tier_of

RUN_TYPES = ('full', 'smoke', 'probe')


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
    return p


def parse_args(argv=None):
    """Parse and apply the trainer's post-conditions on the yaw seed."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.yaw_aug_seed = args.seed if args.yaw_aug_seed is None else args.yaw_aug_seed
    if args.provenance_out is not None and not args.no_save:
        parser.error("--provenance-out applies to --no-save runs; saving runs write it to --save-dir")
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
    del args.run_type
    del args.provenance_out
    return args
