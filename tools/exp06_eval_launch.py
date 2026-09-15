"""Launch one exp_06 simulated evaluation; exp_04 owns the child and the completion.

Plan v4 section 6.3. ``tools.exp04_eval_launch`` accepts only exp_04's own binding
names and records ``tools.exp04_eval_launch`` as the writer, so this launcher

* parses the exp_06 evaluator's CLI plus the launcher flags,
* validates and hashes the exp_06 bindings (``heading``) **itself** and removes them
  from the argument list before delegating to ``tools.exp06_eval.build_fields``
  (which calls ``exp04_eval_launch.build_fields(module=tools.exp06_eval)``),
* adds its own closure record (``source_closures['writer_exp06']``) and the exp_06
  bindings to the fields, and re-checks the arm metadata *before* ``execute_run``
  writes the manifest those fields become,
* and runs the child through the unchanged ``execute_run``, which alone decides
  completion after the child and its log have closed.

    python tools/exp06_eval_launch.py --backbone cylindrical_oriented \
        --checkpoint <epoch_012.pth> --checkpoint-role arm --checkpoint-epoch 12 \
        --manifest ckpt/yaw_aug/reference_manifest_k8_seed42.json --manifest-hash <sha> \
        --out-dir ckpt/exp06/sim_eval/cyl_or/seed42 --log-dir <record> \
        --data-root ~/data_cache/AcousticRooms --run-label cylor_seed42 \
        --reviewed-commit <sha40> --num-shot 8 --conditions P --gpu 1
"""
import copy
import json
import sys
from pathlib import Path

from tools import exp04_eval_launch as launcher
from tools import exp06_eval as evaluator
from tools import exp06_heading
from tools import provenance as p

EXP06_INPUTS = ('heading',)
MUTABLE_INPUTS = frozenset(launcher.MUTABLE_INPUTS) | frozenset(EXP06_INPUTS)
WRITER = 'tools.exp06_eval_launch'
CLOSURES = ('entrypoint', 'writer', 'writer_exp06')
USABLE_HEADINGS = ('estimated', 'override')


def parse_args(argv=None):
    """The exp_06 evaluator's flags plus the launcher's own, resolved to absolute paths."""
    parser = evaluator.build_parser(require_eval_manifest=False)
    for name in ('run-label', 'reviewed-commit', 'data-root', 'log-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--num-shot', type=int, required=True)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--allow-dirty', action='store_true')
    parser.add_argument('--bind-input', action='append', default=[], metavar='NAME=PATH')
    args = parser.parse_args(argv)
    if args.eval_manifest is not None:
        parser.error('eval-manifest is created by the launcher')
    for key in ('out_dir', 'checkpoint', 'manifest', 'data_root', 'log_dir'):
        setattr(args, key, str(Path(getattr(args, key)).resolve()))
    args.eval_manifest = str(Path(args.out_dir) / 'eval_manifest.json')
    return args


def split_bindings(args):
    """Separate exp_06's bindings from exp_04's, validating and hashing our own.

    A heading binding must be a record ``tools.exp06_heading`` still validates and whose
    decision is usable; a refused, missing or malformed record is refused here, before
    any directory is created.
    """
    inherited, own, seen = [], {}, set()
    for binding in args.bind_input:
        name, separator, path = binding.partition('=')
        if name not in MUTABLE_INPUTS or not separator or not path or name in seen:
            raise ValueError('invalid or duplicate --bind-input: ' + binding)
        seen.add(name)
        if name not in EXP06_INPUTS:
            inherited.append(binding)
            continue
        resolved = Path(path).resolve()
        try:
            record = exp06_heading.read_heading_json(str(resolved))
        except (OSError, ValueError) as error:
            raise ValueError('unusable heading binding {}: {}'.format(path, error)) from error
        if record['decision'] not in USABLE_HEADINGS:
            raise ValueError('heading binding {} records the decision {!r}'.format(
                path, record['decision']))
        own[name] = {'path': str(resolved), 'sha256': p.sha256_file(resolved)}
    return inherited, own


def child_command(args, repo):
    """Serialize only evaluator arguments, retaining empty grids and bool flags."""
    command = [sys.executable, str(Path(repo) / (evaluator.__name__.replace('.', '/') + '.py'))]
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


def check_fields(args, fields):
    """The child's own manifest handshake, re-run before execute_run writes the manifest."""
    expected = evaluator.exp06_metadata(args)
    mismatches = [key for key, value in sorted(expected.items())
                  if key not in fields or json.dumps(fields[key], sort_keys=True)
                  != json.dumps(value, sort_keys=True)]
    mismatches += ['source_closures.' + name for name in CLOSURES
                   if name not in fields.get('source_closures', {})]
    if mismatches:
        raise ValueError('inconsistent evaluation fields: ' + ', '.join(sorted(mismatches)))
    return fields


def build_fields(args, command, repo):
    """exp_04's fields, plus this launcher's closure and the exp_06 bindings."""
    inherited, own = split_bindings(args)
    delegate = copy.copy(args)
    delegate.bind_input = inherited
    fields = evaluator.build_fields(delegate, command, repo)
    records, digest = p.closure_record(p.source_closure(WRITER, repo),
                                       fields['reviewed_commit'], repo)
    if not records or any(record['reviewed_blob_sha256'] is None
                          or record['reviewed_blob_sha256'] != record['working_tree_sha256']
                          or record['commits_after_reviewed'] for record in records):
        raise ValueError('writer_exp06 closure differs from reviewed_commit')
    fields['source_closures']['writer_exp06'] = {'files': records, 'sha256': digest}
    for name, binding in sorted(own.items()):
        if name in fields['mutable_inputs']:
            raise ValueError('duplicate mutable input: ' + name)
        fields['mutable_inputs'][name] = binding
    return check_fields(args, fields)


def main(argv=None):
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    command = child_command(args, repo)
    return launcher.execute_run(args, command, lambda: build_fields(args, command, repo), repo)


if __name__ == '__main__':
    main()
