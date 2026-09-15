"""Split-bound evaluation (seen or unseen) through the exp_04 numerical loop.

The frozen ``eval_yaw_rotation.build_manifest_dataset`` hard-codes the UNSEEN test
split, so a seen reference manifest (6 217 queries) is rejected by
``ManifestDataset``'s exact query-list check.  This entry point keeps every numerical
function, the manifest handshake and the canonical batch padding of
``tools/exp04_eval.py`` and changes exactly one thing: which test split the manifest's
queries are served from.  ``--split unseen`` therefore reproduces ``tools/exp04_eval``
bit for bit; ``--split seen`` serves the authors' seen split, built with the frozen
seen dataset class at the manifest's ``num_shot``.

Run it from the repository root: the seen dataset module opens
``treble_multi_room_dataset/seen_test_split.pkl`` relative to the working directory.
That file is bound by digest in the evaluation manifest and in both output metas, and
re-checked after the run.  The seen dataset *module* is bound as well: the frozen helper
imports it inside a function, so this entry point imports it at module scope and
``source_closure('tools.exp07_eval')`` -- the digest pinned for the evaluator -- covers
the code that selects and serves the seen queries.
"""
import sys
from pathlib import Path

from eval_xRIR_backbone import build_dataset
from tools import exp04_eval as base
from tools import provenance as p
from treble_multi_room_dataset import treble_xRIR_seen_dataset as seen_module

REPO = Path(__file__).resolve().parents[1]
# Derived from the import above, so the binding cannot be dropped as an unused import.
SEEN_DATASET_SOURCE = str(Path(seen_module.__file__).resolve().relative_to(REPO))
SPLITS = ('unseen', 'seen')
SPLIT_ENTRIES = {'unseen': 6337, 'seen': 6217}  # queries of each test split
MAX_LEN = 9600


def build_parser(require_eval_manifest=True):
    parser = base.build_parser(require_eval_manifest)
    parser.add_argument('--split', choices=SPLITS, required=True)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def split_metadata(args):
    """The split identity recorded in the eval manifest and in both output metas."""
    return dict(split=args.split, seen_split_sha256=p.sha256_file(REPO / p.SEEN_SPLIT))


def build_split_dataset(args, manifest, max_samples=0):
    """The manifest's queries served from the requested split.

    ``--split unseen`` calls the frozen builder unchanged.  ``--split seen`` mirrors it
    with the seen dataset class (``eval_xRIR_backbone.build_dataset``, also frozen).
    Either way ``ManifestDataset`` compares the manifest's query list with the split's
    ``file_list``, so a manifest of the other split is refused before any model is built.
    """
    if args.split == 'unseen':
        return base.yaw.build_manifest_dataset(manifest, max_samples=max_samples, max_len=MAX_LEN)
    dataset = build_dataset(args.split, manifest['num_shot'], MAX_LEN)
    full = base.yaw.ManifestDataset(dataset, manifest)
    if int(max_samples) > 0:
        return base.yaw.SubsetManifestDataset(full, min(int(max_samples), len(full)))
    return full


def validate_manifest(args, metadata=None):
    fields, digest, reference = base.validate_manifest(args)
    expected = metadata or split_metadata(args)
    mismatches = [key for key, value in expected.items() if fields.get(key) != value]
    if mismatches:
        raise ValueError('evaluation manifest mismatch: ' + ', '.join(sorted(mismatches)))
    return fields, digest, reference


def build_fields(args, command, repo):
    """Extend the shared launcher's bindings with the split identity.

    ``seen_split`` is bound automatically (an explicit --bind-input of another file is
    refused) and revalidated at finalisation, and a seen run counts as confirmatory only
    when its manifest holds every query of the split.
    """
    from tools import exp04_eval_launch as launcher
    fields = launcher.build_fields(args, command, repo, module=sys.modules[__name__])
    fields.update(split_metadata(args))  # the shared builder records the unseen split
    binding = p.seen_split_identity(repo)
    if fields['mutable_inputs'].get('seen_split', binding) != binding:
        raise ValueError('seen_split must bind ' + p.SEEN_SPLIT)
    fields['mutable_inputs']['seen_split'] = binding
    fields['confirmatory'] = bool(fields['confirmatory']
                                  and fields['split_count'] == SPLIT_ENTRIES[args.split])
    return fields


def run_exp07(args):
    metadata = split_metadata(args)
    result = base.run_exp04(args, metadata=metadata,
                            manifest_validator=lambda parsed: validate_manifest(parsed, metadata),
                            dataset_factory=lambda manifest, max_samples=0:
                            build_split_dataset(args, manifest, max_samples))
    if p.sha256_file(REPO / p.SEEN_SPLIT) != metadata['seen_split_sha256']:
        raise ValueError('seen_test_split.pkl changed during evaluation')
    return result


def main(argv=None):
    return run_exp07(parse_args(argv))


if __name__ == '__main__':
    main()
