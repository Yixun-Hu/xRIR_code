"""exp_06 simulated-room evaluation: the frozen exp_04 loop on the exp_06 registry.

Plan v4 section 6.3. Nothing of ``tools.exp04_eval`` -- or of the exp_03 evaluator it
composes -- is modified or monkey patched. This module owns a parser whose
``--backbone`` ranges over ``BACKBONES_EXP06`` and which records the role and epoch of
the checkpoint it evaluates, an ``exp06_metadata`` block that the evaluation manifest and
both output files must carry, and a ``run_exp06`` that hands the exp_04 loop the exp_06
factory. A manifest that omits or contradicts any of those fields is refused before the
model is loaded, exactly as ``tools.exp05_eval`` refuses a tier mismatch.

    python tools/exp06_eval.py --backbone cylindrical_oriented \
        --checkpoint ckpt/exp06/pretrain/xRIR_cylor_8_shot/final/epoch_012.pth \
        --checkpoint-role arm --checkpoint-epoch 12 --conditions P \
        --manifest ckpt/yaw_aug/reference_manifest_k8_seed42.json --manifest-hash <sha> \
        --eval-manifest <out>/eval_manifest.json --out-dir <out>
"""
import hashlib
import json
import sys

from model.xRIR_cyl_oriented import BACKBONES_EXP06, build_xrir_exp06
from tools import exp04_eval as base

CHECKPOINT_ROLES = ('arm', 'baseline', 'diagnostic')
FRAME = 'room'  # the simulated split carries no heading: its geometry is the room frame
METADATA_FIELDS = ('model_class', 'registry_sha256', 'checkpoint_role', 'checkpoint_epoch',
                   'heading', 'frame')


def registry_sha256():
    """The digest of the backbone registry, defined exactly as tools.exp06_train does."""
    mapping = {name: cls.__module__ + '.' + cls.__qualname__
               for name, cls in BACKBONES_EXP06.items()}
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()


def build_parser(require_eval_manifest=True):
    """exp_04's CLI, restricted to the exp_06 registry and naming the checkpoint's role.

    The parser instance is fresh on every call, so replacing the ``--backbone`` choices
    never reaches the pinned module's own parsers.
    """
    parser = base.build_parser(require_eval_manifest)
    for action in parser._actions:
        if action.dest == 'backbone':
            action.choices = sorted(BACKBONES_EXP06)
    parser.add_argument('--checkpoint-role', choices=CHECKPOINT_ROLES, default='arm')
    parser.add_argument('--checkpoint-epoch', type=int, required=True)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def model_factory(backbone, num_shot):
    """The exp_06 registry: a pinned name still returns its pinned class, by identity."""
    return build_xrir_exp06(backbone, num_shot)


def exp06_metadata(args):
    """What the manifest and both outputs record about the arm being evaluated."""
    return {'model_class': BACKBONES_EXP06[args.backbone].__name__,
            'registry_sha256': registry_sha256(),
            'checkpoint_role': args.checkpoint_role,
            'checkpoint_epoch': args.checkpoint_epoch,
            'heading': None, 'frame': FRAME}


def validate_manifest(args, metadata=None):
    """exp_04's handshake plus the exp_06 metadata; an absent field is a mismatch."""
    fields, digest, reference = base.validate_manifest(args)
    expected = exp06_metadata(args) if metadata is None else metadata
    mismatches = [key for key, value in sorted(expected.items())
                  if key not in fields or json.dumps(fields[key], sort_keys=True)
                  != json.dumps(value, sort_keys=True)]
    if mismatches:
        raise ValueError('evaluation manifest mismatch: ' + ', '.join(mismatches))
    return fields, digest, reference


def build_fields(args, command, repo):
    """Extend the shared launcher's fields without changing exp_04's manifest schema."""
    from tools import exp04_eval_launch as launcher
    fields = launcher.build_fields(args, command, repo, module=sys.modules[__name__])
    fields.update(exp06_metadata(args))
    return fields


def run_exp06(args):
    """The frozen loop, bound to one arm: same metadata in the manifest and the outputs."""
    metadata = exp06_metadata(args)
    return base.run_exp04(args, model_factory=model_factory, metadata=metadata,
                          manifest_validator=lambda parsed: validate_manifest(parsed, metadata))


def main(argv=None):
    return run_exp06(parse_args(argv))


if __name__ == '__main__':
    main()
