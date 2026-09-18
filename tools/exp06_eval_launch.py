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
* runs the child through the unchanged ``execute_run``, which alone decides
  completion after the child and its log have closed,
* and then re-reads the two finished outputs itself (round 2b finding 2). exp_04's
  completion compares only its own protocol fields, so an output whose ``meta``
  contradicts the exp_06 identity in the manifest -- model class, registry digest,
  checkpoint role and epoch, heading, frame -- would be certified. This launcher
  requires both outputs to carry every one of those fields, typed and equal to the
  manifest, and to still hash to what the completion bound; anything else renames the
  run aside as ``<name>_QUARANTINED_<reason>`` (never deleting it), records the reason
  in ``quarantine.json`` and exits non-zero.

    python tools/exp06_eval_launch.py --backbone cylindrical_oriented \
        --checkpoint <epoch_012.pth> --checkpoint-role arm --checkpoint-epoch 12 \
        --manifest ckpt/yaw_aug/reference_manifest_k8_seed42.json --manifest-hash <sha> \
        --out-dir ckpt/exp06/sim_eval/cyl_or/seed42 --log-dir <record> \
        --data-root ~/data_cache/AcousticRooms --run-label cylor_seed42 \
        --reviewed-commit <sha40> --num-shot 8 --conditions P --gpu 1
"""
import copy
import datetime
import json
import sys
import uuid
from pathlib import Path

from tools import exp04_eval_launch as launcher
from tools import exp06_eval as evaluator
from tools import exp06_heading
from tools import exp06_train_evidence as evidence
from tools import provenance as p

EXP06_INPUTS = ('heading',)
# Validated by tools.exp06_train_evidence, then hashed by exp_04 like any mutable input.
TRAINING_INPUTS = evidence.TRAINING_BINDINGS
MUTABLE_INPUTS = frozenset(launcher.MUTABLE_INPUTS) | frozenset(EXP06_INPUTS)
WRITER = 'tools.exp06_eval_launch'
CLOSURES = ('entrypoint', 'writer', 'writer_exp06')
USABLE_HEADINGS = ('estimated', 'override')
# What both finished outputs must carry: this arm's identity, and exp_04's own protocol.
BOUND_META = evaluator.METADATA_FIELDS + ('conditions', 'n_samples')


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


def training_evidence(args, hasher=p.sha256_file):
    """Full-review F2: these weights are the output of the training run this run binds.

    6.3 promised the exp_06 launcher validates ``train_completion``; exp_04's launcher only
    hashes whatever file a binding names, so the semantics live here and run **before**
    ``execute_run`` creates anything. A ``diagnostic`` evaluation is never an arm and binds
    no training run.
    """
    if args.checkpoint_role not in ('arm', 'baseline'):
        return None
    bound = {}
    for binding in args.bind_input:
        name, separator, path = binding.partition('=')
        if separator and name in TRAINING_INPUTS:
            bound[name] = path
    return evidence.check(args.checkpoint_role, bound, hasher(args.checkpoint))


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


def build_fields(args, command, repo, training=None):
    """exp_04's fields, plus this launcher's closure, bindings and training evidence."""
    inherited, own = split_bindings(args)
    training = training_evidence(args) if training is None else training
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
    fields['training_evidence'] = training
    return check_fields(args, fields)


class OutputMismatch(ValueError):
    """A finished output that contradicts the manifest it would be certified against."""

    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _same(left, right):
    """Type-strict equality, as the inherited output check and exp_06's manifest use."""
    return (type(left) is type(right)
            and json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True))


def _declared(run, completion):
    """The manifest this run was certified against, still the bytes completion bound."""
    path = Path(run) / 'eval_manifest.json'
    try:
        digest = p.sha256_file(path)
        fields = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise OutputMismatch('manifest_unreadable', 'eval_manifest.json: {}'.format(error))
    if digest != completion.get('eval_manifest_sha256'):
        raise OutputMismatch('manifest_changed', 'eval_manifest.json hashes to {}, not the {} '
                             'of completion.json'.format(digest, completion.get('eval_manifest_sha256')))
    missing = [key for key in BOUND_META if key not in fields]
    if missing:
        raise OutputMismatch('manifest_incomplete',
                             'eval_manifest.json records no ' + ', '.join(missing))
    return dict({key: fields[key] for key in BOUND_META}, eval_manifest_sha256=digest)


def _persisted(run, completion):
    """The record the run retained, which must still be the one execute_run returned."""
    try:
        record = json.loads((Path(run) / 'completion.json').read_text())
    except (OSError, ValueError) as error:
        raise OutputMismatch('completion_unreadable', 'completion.json: {}'.format(error))
    try:
        matches = isinstance(record, dict) and _same(record, dict(completion))
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise OutputMismatch('completion_mismatch', 'completion.json is not the record '
                             'this run was finalised with')
    return record


def validate_outputs(run, completion):
    """Both finished outputs are this arm's, and are the bytes completion bound."""
    run = Path(run)
    completion = _persisted(run, completion)
    expected = _declared(run, completion)
    bound = completion.get('outputs') or {}
    for name in launcher.OUTPUTS:
        try:
            payload = json.loads((run / name).read_text())
        except (OSError, ValueError) as error:
            raise OutputMismatch('output_unreadable', '{}: {}'.format(name, error))
        meta = payload.get('meta') if isinstance(payload, dict) else None
        if not isinstance(meta, dict):
            raise OutputMismatch('output_meta_missing', name + ' records no meta')
        for key, value in sorted(expected.items()):
            if key not in meta:
                raise OutputMismatch('output_field_missing',
                                     '{} meta records no {}'.format(name, key))
            if not _same(meta[key], value):
                raise OutputMismatch('output_field_mismatch', '{} meta {} {!r} is not the {!r} '
                                     'of the evaluation manifest'.format(name, key, meta[key], value))
        if p.sha256_file(run / name) != bound.get(name):
            raise OutputMismatch('output_changed',
                                 '{} is not the bytes completion.json bound'.format(name))
    return expected


def quarantine(run, error):
    """Rename the run aside with the reason it was refused; nothing is ever deleted."""
    run = Path(run)
    target = run.with_name(run.name + '_QUARANTINED_' + error.reason)
    if target.exists():
        target = target.with_name(target.name + '_' + uuid.uuid4().hex)
    run.rename(target)
    p.write_completion(target / 'quarantine.json', {
        'schema_version': 1, 'reason': error.reason, 'detail': error.detail,
        'run_dir': str(run), 'quarantined_dir': str(target),
        'quarantined_at': datetime.datetime.now().astimezone().isoformat()})
    return target


def certify_outputs(run, completion):
    """exp_06's own post-run gate: the quarantine directory, or None when nothing is wrong.

    Nit 7: the hashes validated are the ones the run published in completion.json, which
    must equal the mapping execute_run returned -- a retained record that says something
    else is a refusal, not a detail.
    """
    try:
        validate_outputs(run, completion)
    except OutputMismatch as error:
        return quarantine(run, error)
    return None


def main(argv=None):
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    split_bindings(args)                     # refuse a bad binding before anything is created
    training = training_evidence(args)       # F2: and before execute_run makes the run dir
    command = child_command(args, repo)
    completion = launcher.execute_run(
        args, command, lambda: build_fields(args, command, repo, training), repo)
    quarantined = certify_outputs(Path(args.out_dir), completion)
    if quarantined is not None:
        print('EXP06_EVAL_QUARANTINED ' + str(quarantined), file=sys.stderr, flush=True)
        raise SystemExit(3)
    return completion


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit('refusing: ' + str(error))
