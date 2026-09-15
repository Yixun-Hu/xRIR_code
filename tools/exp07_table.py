"""Role-aware exp_07 (seen protocol) table producer.

Admission reuses exp_04's :func:`tools.paired_compare.admit_runs` as the collector for
everything the two experiments share and adds, in :func:`run_contract`, what exp_04
cannot express: the seen split identity in every artefact that records it, the approved
tools.exp07_eval and writer closures, and -- for the three NEW arms only -- the training
provenance (approved checkpoint, completion/manifest/args bindings tied to the
checkpoint's attempt directory, the seen protocol in the training manifest and its
identity, one shared training closure, an admissible launcher closure, and an args.json
equal to the plan's recipe with this arm's backbone and yaw flags).  The released
checkpoint is an EXTERNAL reference row: digest-only admission and no training bindings,
with every evaluation check kept.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from tools import provenance
from tools.exp07_profiles import get_profile, json_value, load_approved_digests
from tools.paired_compare import REPO, _equal, admit_runs, producer_identity
from tools import results_table
from tools.results_table import METRICS, write_outputs

TRAINING_BINDINGS = ('train_args', 'train_manifest', 'train_completion')
BINDING_FILES = {'train_args': 'args.json', 'train_manifest': 'train_manifest.json',
                 'train_completion': 'completion.json', 'control_args': 'args.json'}
CHECKPOINT_NAME = 'epoch_012.pth'


def run_contract(directory, arm, profile, pins):
    """Validate the exp_07 additions; raise on the first named failure."""
    directory = Path(directory).resolve()
    inputs = {}
    def read(path):
        raw = Path(path).read_bytes()
        inputs[str(Path(path).resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    fields, completion, sample, metrics = [read(directory / name) for name in (
        'eval_manifest.json', 'completion.json', 'per_sample_yaw.json', 'metrics_yaw.json')]
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    require(_equal(fields.get('decomposition_batches'), 0), 'decomposition_batches')
    dataset = profile['dataset']
    identity = dict(split='seen', seen_split_sha256=dataset['seen_split_sha256'])
    for payload in (fields, completion, sample['meta'], metrics['meta']):
        require(all(_equal(payload.get(key), value) for key, value in identity.items()),
                'split identity')
    require(_equal(fields.get('split_count'), dataset['n_queries']) and
            _equal(fields.get('n_samples'), dataset['n_queries']) and
            _equal(fields.get('max_samples'), profile['max_samples']), 'seen split coverage')
    bound_split = fields['mutable_inputs'].get('seen_split')
    require(bound_split is not None and bound_split['path'] == provenance.SEEN_SPLIT and
            bound_split['sha256'] == dataset['seen_split_sha256'], 'seen_split binding')
    for pin, recorded in (('evaluator', 'entrypoint'), ('writer', 'writer')):
        digest = fields['source_closures'][recorded]['sha256']
        require(digest is not None and digest == pins['closures'][pin], 'unapproved ' + pin)
    names = set(fields['mutable_inputs'])
    require(names <= set(BINDING_FILES) | {'seen_split', 'probe_receipt'},
            'unknown mutable binding')
    for name in names & set(BINDING_FILES):
        require(Path(fields['mutable_inputs'][name]['path']).name == BINDING_FILES[name],
                name + ' binding filename')
    waivers = [str(directory) + ': mutable_inputs names']
    if arm['reference']:  # external checkpoint, exempt from TRAINING provenance only
        require(not names & set(TRAINING_BINDINGS), 'released row has no training provenance')
        return dict(role=arm['role'], reference=True, training=None, waivers=waivers, inputs=inputs)
    approved = pins['checkpoints'][arm['role']]
    require(approved['path'] == arm['checkpoint'] and _equal(approved['epoch'], arm['epoch']) and
            _equal(approved['epoch'], 12), 'approved checkpoint path/epoch')
    require(approved['sha256'] == fields['checkpoint_sha256'], 'approved checkpoint digest')
    root = Path(fields['repo'])
    attempt = (root / fields['checkpoint']).resolve().parent
    training = {}
    for name in TRAINING_BINDINGS:
        require(name in names, 'missing ' + name + ' binding')
        bound = fields['mutable_inputs'][name]
        path = (root / bound['path']).resolve()
        require(path.parent == attempt, name + ' binding path')
        training[name] = read(path)
        require(inputs[str(path)] == bound['sha256'], name + ' bytes')
    manifest, finished, args = (training[name] for name in
                                ('train_manifest', 'train_completion', 'train_args'))
    require(finished.get('train_manifest_sha256') ==
            fields['mutable_inputs']['train_manifest']['sha256'],
            'train_completion binds train_manifest')
    outputs = finished.get('outputs')
    require(isinstance(outputs, dict), 'train_completion outputs')
    require(outputs.get(CHECKPOINT_NAME) == fields['checkpoint_sha256'],
            'train_completion checkpoint epoch/digest')
    require(outputs.get('args.json') == fields['mutable_inputs']['train_args']['sha256'],
            'train_completion binds args.json')
    require(_equal(manifest.get('protocol'), 'seen') and
            _equal(manifest.get('effective_args', {}).get('protocol'), 'seen') and
            _equal(manifest.get('train_data_identity', {}).get('protocol'), 'seen') and
            _equal(manifest.get('timing_limits', {}).get('protocol'), 'seen'), 'training protocol')
    trained_split = manifest.get('mutable_inputs', {}).get('seen_split')
    require(trained_split is not None and trained_split['path'] == provenance.SEEN_SPLIT and
            trained_split['sha256'] == dataset['seen_split_sha256'], 'training seen_split binding')
    launcher = manifest['source_closures']['launcher']['sha256']
    require(launcher is not None and launcher in (pins['closures']['training_launcher'] or ()),
            'training launcher closure')
    trainer = manifest['source_closures']['training']['sha256']
    require(trainer is not None and trainer == pins['closures']['training'], 'training closure')
    for key, value in dict(profile['recipe'], **profile['full_run']).items():
        require(_equal(args.get(key), value), 'args ' + key)
    for key in ('backbone', 'yaw_aug', 'yaw_aug_seed', 'yaw_aug_width'):
        require(_equal(args.get(key), arm[key]), 'args ' + key)
    require(_equal(args.get('train_batches_per_epoch'), profile['train_batches_per_epoch']),
            'args train_batches_per_epoch')
    require(_equal(args.get('tier', profile['tier']), profile['tier']), 'args tier')
    require(_equal(args.get('backbone'), fields.get('backbone')), 'evaluated backbone')
    return dict(role=arm['role'], reference=False, training=trainer, waivers=waivers, inputs=inputs)


def admit(directories, profile, approved=None, producer=None, exploratory=False):
    """Route runs to (arm, K) groups, apply the contract, then collect through exp_04.

    Only differences this module checks independently are waived from the inherited
    admission; nothing on disk and no imported producer's state is modified.
    """
    pins, receipt = load_approved_digests() if approved is None else approved
    approved = (pins, receipt)
    deviations, groups, contracts, waivers, snapshots = [], [], {}, set(), {}
    def check(ok, message):
        if not ok:
            deviations.append(message)
    check(type(pins['schema_version']) is int and pins['schema_version'] == 1,
          'approval schema_version')
    for key in ('evaluator', 'writer', 'training_launcher', 'training', 'producer_table'):
        check(bool(pins['closures'][key]), 'profile not yet approved: ' + key)
    for arm in profile['arms']:
        effective = dict(arm)
        if not arm['reference']:
            checkpoint = pins['checkpoints'][arm['role']]
            check(checkpoint['path'] == arm['checkpoint'] and
                  _equal(checkpoint['epoch'], arm['epoch']),
                  'approved checkpoint path/epoch: ' + arm['role'])
            effective['sha256'] = checkpoint['sha256']
        check(effective['sha256'] is not None,
              'profile not yet approved: ' + arm['role'] + ' checkpoint')
        for shot in sorted(profile['num_shot'], reverse=True):
            groups.append((effective, shot, []))
    if deviations and not exploratory:
        raise ValueError('admission failed: ' + '; '.join(deviations))
    directories = sorted(str(Path(item).resolve()) for item in directories)
    check(len(set(directories)) == len(directories), 'duplicate run directories')
    for directory in directories:
        try:
            fields = json.loads((Path(directory) / 'eval_manifest.json').read_text())
            matches = [(arm, paths) for arm, shot, paths in groups
                       if _equal(fields.get('num_shot'), shot) and
                       (Path(fields['repo']) / fields['checkpoint']).resolve() ==
                       (REPO / arm['checkpoint']).resolve()]
            if len(matches) != 1:
                raise ValueError('unregistered checkpoint/K')
            arm, paths = matches[0]
            paths.append(directory)
            contract = run_contract(directory, arm, profile, pins)
            contracts[directory] = {key: value for key, value in contract.items()
                                    if key not in ('waivers', 'inputs')}
            for path, digest in contract['inputs'].items():
                check(snapshots.get(path, digest) == digest, 'contract input changed: ' + path)
                snapshots[path] = digest
            waivers.update(contract['waivers'])
        except (ValueError, TypeError, KeyError, OSError, IndexError) as error:
            check(False, '{}: {}'.format(directory, error))
    trainers = {c['training'] for c in contracts.values() if c['training'] is not None}
    check(len(trainers) <= 1, 'the new arms do not share one training closure')
    producer = producer_identity('tools.exp07_table') if producer is None else producer
    admitted = admit_runs(profile, groups, approved=approved, exploratory=True,
                          producer=producer, producer_key='producer_table')
    for path, digest in snapshots.items():
        check(admitted['inputs'].get(path) == digest, 'contract input changed: ' + path)
    deviations.extend(item for item in admitted['deviations'] if item not in waivers)
    if deviations and not exploratory:
        raise ValueError('admission failed: ' + '; '.join(deviations))
    admitted.update(deviations=deviations, contracts=contracts)
    return groups, admitted


MARKDOWN = REPO / 'worklog/worklog_yixun/model_comparison_seen.md'


def build_table(directories, profile=None, approved=None, exploratory=False, producer=None):
    """Admit the runs and aggregate them under the TABLE_V1 rules; write no artefacts.

    Each seed mean uses its own finite queries and the row bounds the count differences;
    EDT is reported in ms.  The released row carries its unknown epoch as such.
    """
    profile = json_value(get_profile('TABLE_SEEN_V1')) if profile is None else profile
    groups, admitted = admit(directories, profile, approved, producer, exploratory)
    rows = []
    for (arm, shot, _), runs in zip(groups, admitted['groups']):
        if not runs:  # exploratory only: admission has already named the missing arm
            continue
        cells = [run['P'][str(profile['grid'][0])] for run in runs]
        names = set().union(*(set(cell) for cell in cells))
        if names - set(METRICS):
            raise ValueError('unknown metric names: ' + ', '.join(sorted(names - set(METRICS))))
        if any(set(cell) != names for cell in cells):
            raise ValueError('metric coverage differs across seeds')
        metrics = {}
        for source, specification in METRICS.items():
            if specification is None or source not in names:
                continue
            name, unit, scale = specification
            per_seed = {}
            for run, cell in zip(runs, cells):
                values = np.asarray(cell[source], dtype=float)
                finite = values[np.isfinite(values)]
                if not finite.size:
                    raise ValueError('zero finite queries: ' + source)
                per_seed[str(run['meta']['manifest_seed'])] = {
                    'mean': float(finite.mean() * scale), 'n_finite': int(finite.size)}
            counts = [item['n_finite'] for item in per_seed.values()]
            if max(counts) - min(counts) > profile['finite_count_tolerance']:
                raise ValueError('finite count tolerance exceeded: ' + source)
            means = np.asarray([item['mean'] for item in per_seed.values()])
            if not np.isfinite(means).all():
                raise ValueError('nonfinite seed mean: ' + source)
            metrics[name] = dict(source=source, unit=unit, mean=float(means.mean()),
                                 sd=float(means.std(ddof=profile['seed_sd_ddof'])),
                                 per_seed=per_seed)
        protocol = dict(num_shot=shot, split=profile['dataset']['split'],
            n_queries=profile['dataset']['n_queries'], seeds=sorted(profile['seeds'][shot]),
            epoch=arm['epoch'] if arm['epoch'] is not None else 'unknown (external)',
            condition=profile['condition'], k=profile['grid'][0], training=arm['training'],
            batch_size=profile['batch_size'], tf32=profile['tf32'],
            max_samples=profile['max_samples'])
        rows.append(dict(role=arm['role'], label=arm['label'], num_shot=shot, epoch=arm['epoch'],
                         backbone=arm['backbone'], reference=arm['reference'],
                         protocol=protocol, metrics=metrics))
    literal = json.loads(json.dumps(json_value(profile)))
    digest = hashlib.sha256(json.dumps(literal, sort_keys=True, separators=(',', ':'),
                                       allow_nan=False).encode()).hexdigest()
    return dict(schema_version=1, profile_name='TABLE_SEEN_V1', profile=literal,
                profile_digest=digest, exploratory=exploratory, rows=rows,
                deviations=list(admitted['deviations']), contracts=admitted['contracts'],
                inputs=admitted['inputs'],
                producer_closure_sha256=admitted['producer']['sha256']), admitted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('TABLE_SEEN_V1',), required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--json', required=True)
    parser.add_argument('--md', default=str(MARKDOWN))
    parser.add_argument('--force-md', action='store_true',
                        help='Regenerate the Markdown; the JSON is always exclusive')
    parser.add_argument('--exploratory', action='store_true',
                        help='List unapproved pins and deviations instead of refusing')
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    result, admitted = build_table(args.runs, exploratory=args.exploratory)
    command = ['exp07_table.py', '--profile', args.profile, '--runs']
    command += [str(Path(item).resolve()) for item in args.runs]
    command += ['--json', str(Path(args.json).absolute()), '--md', str(Path(args.md).absolute())]
    command += ['--force-md'] if args.force_md else []
    command += ['--exploratory'] if args.exploratory else []
    write_outputs(result, admitted, args.json, args.md, args.force_md, command)
    return result


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        raise SystemExit(str(error))
