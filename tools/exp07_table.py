"""Role-aware exp_07 (seen protocol) table producer.

Admission reuses exp_04's :func:`tools.paired_compare.admit_runs` as the collector for
everything the two experiments share and adds, in :func:`run_contract`, what exp_04
cannot express: the seen split identity in every artefact that records it, the approved
tools.exp07_eval and writer closures, and -- for the three NEW arms only -- the training
provenance (approved checkpoint, completion/manifest/args bindings tied to the
checkpoint's attempt directory, the seen protocol in the training manifest and its
identity, one shared training closure and an admissible launcher closure -- both
recomputed from their complete records -- and an args.json equal to the plan's recipe
with this arm's backbone and yaw flags).  The released
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
from tools.exp05_params import TIERS
from tools.exp07_profiles import get_profile, json_value, load_approved_digests
from tools.paired_compare import (REPO, _closure_digest, _equal, admit_runs,
                                  producer_identity)
from tools.results_table import METRICS, write_outputs

TRAINING_BINDINGS = ('train_args', 'train_manifest', 'train_completion')
BINDING_FILES = {'train_args': 'args.json', 'train_manifest': 'train_manifest.json',
                 'train_completion': 'completion.json', 'control_args': 'args.json'}
CHECKPOINT_NAME = 'epoch_012.pth'


def run_contract(directory, arm, profile, pins, cache=None):
    """Validate the exp_07 additions; raise on the first named failure.

    ``cache`` memoises the training inventory sidecars, which the thirty new-arm runs
    bind to the same three files; the bytes are still bound per run and rehashed by
    :func:`tools.results_table.write_outputs` before anything is published.
    """
    directory = Path(directory).resolve()
    cache, inputs = {} if cache is None else cache, {}
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
    require(_equal(manifest.get('mode'), 'full') and _equal(manifest.get('allow_dirty'), False),
            'training full-run mode')
    identity = manifest.get('train_data_identity', {})
    sidecar = identity.get('inventory_file') or {}
    inventory = (root / sidecar.get('path', 'absent')).resolve()
    require(inventory.name == 'train_inventory.json' and inventory.parent == attempt,
            'training inventory sidecar path')
    if cache.get(str(inventory)) is None:
        records = read(inventory)['inventory']
        cache[str(inventory)] = (inputs[str(inventory)], len(records),
                                 provenance._inventory_digest(records))
    bytes_sha256, files, digest = cache[str(inventory)]
    inputs[str(inventory)] = bytes_sha256
    require(bytes_sha256 == sidecar.get('sha256'), 'training inventory sidecar bytes')
    require(_equal(identity.get('inventory_files'), profile['train_inventory_files']) and
            files == profile['train_inventory_files'], 'training inventory file count')
    require(digest == identity.get('inventory_sha256'), 'training inventory digest')
    ledger = read(attempt.parent / 'cumulative_hours.json')
    require(any(_equal(row.get('attempt'), attempt.name) and _equal(row.get('mode'), 'full')
                for row in ledger.get('attempts', ())), 'training hours ledger')
    limits = manifest.get('timing_limits') or {}
    bound = manifest.get('mutable_inputs', {}).get('probe_receipt') or {}
    receipt_path = (root / bound.get('path', 'absent')).resolve()
    require(receipt_path.is_file(), 'probe receipt binding')
    receipt = read(receipt_path)
    require(inputs[str(receipt_path)] == bound.get('sha256') and
            limits.get('probe_receipt_sha256') == bound.get('sha256'), 'probe receipt identity')
    # The launcher derives all three limits from this receipt (tools.exp05_gates.timing_limits).
    seconds = receipt.get('T_run')
    require(type(seconds) is float and 0 < seconds / 3600 <= profile['projection_max_hours'] and
            _equal(limits.get('projection_hours'), seconds / 3600), 'training projection hours')
    require(_equal(limits.get('ceiling_hours'), profile['ceiling_factor'] * seconds / 3600),
            'training ceiling hours')
    epoch_seconds = receipt.get('T_epoch')
    require(type(epoch_seconds) is float and
            _equal(limits.get('epoch_seconds'), 1.05 * epoch_seconds), 'training epoch acceptance')
    for key, value in (('tier', profile['tier']), ('backbone', arm['backbone']),
                       ('protocol', 'seen'), ('yaw_aug', arm['yaw_aug']),
                       ('passed', True), ('PROBE_NOT_CLEAN', False)):
        require(_equal(receipt.get(key), value), 'probe receipt ' + key)
    recorded = {}
    for name in ('training', 'launcher'):
        # A7: the approval pins the REVIEWED source identity, so recompute it here from the
        # manifest's complete records instead of trusting the label they are stored beside.
        # Drift of the working tree AFTER the spawn is the completion's business, not this
        # digest's, and stays permitted.
        try:
            recorded[name] = _closure_digest(manifest['source_closures'][name])
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError('training closure records ' + name + ': ' + str(error))
        require(_equal(manifest['source_closures'][name].get('sha256'), recorded[name]),
                'training closure digest ' + name)
    require(recorded['launcher'] in (pins['closures']['training_launcher'] or ()),
            'training launcher closure')
    require(recorded['training'] == pins['closures']['training'], 'training closure')
    trainer = recorded['training']
    for key, value in dict(profile['recipe'], **profile['full_run']).items():
        require(_equal(args.get(key), value), 'args ' + key)
    for key in ('backbone', 'yaw_aug', 'yaw_aug_seed', 'yaw_aug_width'):
        require(_equal(args.get(key), arm[key]), 'args ' + key)
    require(_equal(args.get('train_batches_per_epoch'), profile['train_batches_per_epoch']),
            'args train_batches_per_epoch')
    require(_equal(args.get('tier', profile['tier']), profile['tier']), 'args tier')
    for key, value in TIERS[profile['tier']].items():  # the seen arms are trained at tier M
        require(_equal(args.get('vit_' + key, value), value), 'args tier configuration')
    require(_equal(args.get('backbone'), fields.get('backbone')), 'evaluated backbone')
    return dict(role=arm['role'], reference=False, training=trainer, waivers=waivers, inputs=inputs)


def admit(directories, profile, approved=None, producer=None, exploratory=False,
          producer_key='producer_table'):
    """Route runs to (arm, K) groups, apply the contract, then collect through exp_04.

    Only differences this module checks independently are waived from the inherited
    admission; nothing on disk and no imported producer's state is modified.
    """
    pins, receipt = load_approved_digests() if approved is None else approved
    approved = (pins, receipt)
    deviations, groups, contracts, waivers, snapshots = [], [], {}, set(), {}
    sidecars = {}  # shared across the runs of one arm; see run_contract
    def check(ok, message):
        if not ok:
            deviations.append(message)
    check(type(pins['schema_version']) is int and pins['schema_version'] == 1,
          'approval schema_version')
    for key in ('evaluator', 'writer', 'training_launcher', 'training', producer_key):
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
            contract = run_contract(directory, arm, profile, pins, sidecars)
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
    entry = 'tools.exp07_' + ('pairs' if producer_key == 'producer_pairs' else 'table')
    producer = producer_identity(entry) if producer is None else producer
    admitted = admit_runs(profile, groups, approved=approved, exploratory=True,
                          producer=producer, producer_key=producer_key)
    for path, digest in snapshots.items():
        # Evidence only this contract reads -- the training inventory sidecar, the probe
        # receipt, the hours ledger -- is bound here; what exp_04 also binds must agree.
        check(admitted['inputs'].setdefault(path, digest) == digest,
              'contract input changed: ' + path)
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
