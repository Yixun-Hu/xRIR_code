"""Tier-bound evaluation through the exp_04 numerical loop and manifest handshake."""
import json
import hashlib
import sys
from pathlib import Path

from tools import exp04_eval as base
from tools import provenance as p
from tools.exp05_params import TIERS, build_tier, count_parameters, tier_of


def build_parser(require_eval_manifest=True):
    parser = base.build_parser(require_eval_manifest)
    parser.add_argument('--tier', choices=tuple(TIERS), required=True)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def args_path(args):
    return Path(args.checkpoint).resolve().parent / 'args.json'


def tier_metadata(args, model=None):
    path = args_path(args)
    try:
        raw = path.read_bytes()
        recorded = json.loads(raw)
        if not isinstance(recorded, dict):
            raise ValueError('args.json must be a dictionary')
    except (OSError, ValueError, TypeError) as error:
        raise ValueError('unreadable args.json') from error
    digest = hashlib.sha256(raw).hexdigest()
    config = {'vit_' + key: value for key, value in TIERS[args.tier].items()}
    legacy = not any(key in recorded for key in config) and 'tier' not in recorded
    if legacy:
        if args.tier != 'M':
            raise ValueError('legacy checkpoint requires explicit tier M')
    elif tier_of(recorded) != args.tier or recorded.get('tier', args.tier) != args.tier:
        raise ValueError('checkpoint args.json tier differs from requested tier')
    if p.sha256_file(path) != digest:
        raise ValueError('args.json changed during admission')
    counts = count_parameters(model if model is not None else build_tier(args.backbone, args.tier))
    for key, expected in (('backbone', args.backbone), ('param_counts', counts)):
        if key in recorded and json.dumps(recorded[key], sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError('args.json ' + key + ' differs from requested model')
    return dict(config, tier=args.tier, param_counts=counts, legacy_M=legacy, args_json_sha256=digest)


def load_tier(args, num_shot):
    tier_metadata(args)
    model = build_tier(args.backbone, args.tier, num_shot)
    model.load_state_dict(base.yaw.load_model_state(args.checkpoint), strict=True)
    return model


def validate_manifest(args, metadata=None):
    fields, digest, reference = base.validate_manifest(args)
    expected = metadata or tier_metadata(args)
    mismatches = [key for key, value in expected.items()
                  if json.dumps(fields.get(key), sort_keys=True) != json.dumps(value, sort_keys=True)]
    if p.sha256_file(args_path(args)) != expected['args_json_sha256']:
        mismatches.append('args_json_sha256')
    if mismatches:
        raise ValueError('evaluation manifest mismatch: ' + ', '.join(mismatches))
    return fields, digest, reference


def build_fields(args, command, repo):
    """Extend the shared launcher's bindings without changing exp_04 manifests."""
    from tools import exp04_eval_launch as launcher
    metadata = tier_metadata(args)
    fields = launcher.build_fields(args, command, repo, module=sys.modules[__name__])
    binding = dict(path=str(args_path(args)), sha256=metadata['args_json_sha256'])
    if fields['mutable_inputs'].get('train_args', binding) != binding:
        raise ValueError('train_args must bind the checkpoint attempt args.json')
    fields['mutable_inputs']['train_args'] = binding
    fields.update(metadata)
    return fields


def run_exp05(args):
    metadata = tier_metadata(args)
    result = base.run_exp04(args, model_factory=lambda backbone, shots: build_tier(backbone, args.tier, shots),
                           metadata=metadata, manifest_validator=lambda a: validate_manifest(a, metadata))
    if p.sha256_file(args_path(args)) != metadata['args_json_sha256']:
        raise ValueError('args.json changed during evaluation')
    return result


def main(argv=None):
    return run_exp05(parse_args(argv))


if __name__ == '__main__':
    main()
