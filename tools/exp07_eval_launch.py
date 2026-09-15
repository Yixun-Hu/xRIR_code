"""Launch exp_07 (seen-split) evaluation through the reviewed exp_04 run owner.

Amendment A1: ``tools/exp04_eval_launch.py`` keeps main's bytes.  ``execute_run`` --
exclusive run creation, the tee'd child, digest revalidation and the atomic completion --
is reused exactly as it is; only the arguments, the child command, the bindings and the
split identity are decided here.

Where the split identity lives: the evaluation manifest carries ``split`` and
``seen_split_sha256`` and is hash-bound by the completion (``eval_manifest_sha256``);
both output metas carry the same two fields, are verified by the child itself before it
exits (``exp07_eval.check_outputs``) and are hash-bound by the completion's ``outputs``.
The reviewed ``execute_run`` therefore needs no split-specific branch.

    python tools/exp07_eval_launch.py --split seen --backbone simple \
        --checkpoint <ckpt> --num-shot 8 --manifest <seen manifest> \
        --manifest-hash <hash> --gl-seed 42 --conditions P --yaw-cols 0 ...
"""
import sys
from pathlib import Path
from unittest.mock import patch

from tools import exp04_eval_launch as base
from tools import exp07_eval as evaluator
from tools import exp07_provenance as e7p
from tools import provenance as p

OUTPUTS = base.OUTPUTS
REQUIRED_INPUTS = base.REQUIRED_INPUTS
MUTABLE_INPUTS = base.MUTABLE_INPUTS | {'seen_split'}
SPLIT_FIELDS = evaluator.SPLIT_FIELDS
ENTRY = 'tools/exp07_eval.py'
WRITER = 'tools.exp07_eval_launch'


def parse_args(argv=None):
    """exp_04's evaluation CLI plus ``--split``; ``--entry`` names this launcher only."""
    parser = evaluator.build_parser(require_eval_manifest=False)
    for name in ('run-label', 'reviewed-commit', 'data-root', 'log-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--num-shot', type=int, required=True)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--allow-dirty', action='store_true')
    parser.add_argument('--bind-input', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--entry', choices=('exp07',), default='exp07')
    args = parser.parse_args(argv)
    if args.eval_manifest is not None:
        parser.error('eval-manifest is created by the launcher')
    for key in ('out_dir', 'checkpoint', 'manifest', 'data_root', 'log_dir'):
        setattr(args, key, str(Path(getattr(args, key)).resolve()))
    args.eval_manifest = str(Path(args.out_dir) / 'eval_manifest.json')
    return args


def child_command(args, repo):
    """Serialize only evaluator arguments, retaining empty grids and bool flags."""
    command = [sys.executable, str(Path(repo) / ENTRY)]
    for action in evaluator.build_parser()._actions:
        if action.dest == 'help':
            continue
        value = getattr(args, action.dest)
        if isinstance(value, bool):
            if value:
                command.append(action.option_strings[0])
        else:
            command.append(action.option_strings[0])
            command.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return command


def is_canonical_split(explicit, binding, repo):
    """Does an explicit ``--bind-input seen_split=...`` name the authors' own file?

    Binding paths are absolutised by the shared builder, so the canonical file arrives
    spelled differently from ``seen_split_identity``'s repository-relative record.  Paths
    are therefore compared after resolution (a symlink to the file is the file) and the
    digest must match as well, so a byte-identical copy elsewhere is not accepted.
    """
    try:
        resolved = Path(explicit['path']).resolve()
        digest = explicit['sha256']
    except (AttributeError, KeyError, TypeError):
        return False
    return (resolved == (Path(repo).resolve() / binding['path']).resolve()
            and digest == binding['sha256'])


def build_fields(args, command, repo):
    """The shared bindings, with the split identity, the seen split and this writer."""
    with patch.object(base, 'MUTABLE_INPUTS', MUTABLE_INPUTS):
        fields = base.build_fields(args, command, repo, module=evaluator)
    fields.update(evaluator.split_metadata(args))  # the shared builder records "unseen"
    records, digest = p.closure_record(p.source_closure(WRITER, repo),
                                       fields['reviewed_commit'], repo)
    if not records or any(record['reviewed_blob_sha256'] is None or
            record['reviewed_blob_sha256'] != record['working_tree_sha256'] or
            record['commits_after_reviewed'] for record in records):
        raise ValueError('writer closure differs from reviewed_commit')
    fields['source_closures']['writer'] = {'files': records, 'sha256': digest}
    binding = e7p.seen_split_identity(repo)
    explicit = fields['mutable_inputs'].get('seen_split')
    if explicit is not None and not is_canonical_split(explicit, binding, repo):
        raise ValueError('seen_split must bind ' + e7p.SEEN_SPLIT)
    fields['mutable_inputs']['seen_split'] = binding
    # A run counts as confirmatory only when its manifest holds every query of its split.
    fields['confirmatory'] = bool(fields['confirmatory']
                                  and fields['split_count'] == evaluator.SPLIT_ENTRIES[args.split])
    return fields


def main(argv=None):
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    command = child_command(args, repo)
    return base.execute_run(args, command, lambda: build_fields(args, command, repo), repo)


if __name__ == '__main__':
    main()
