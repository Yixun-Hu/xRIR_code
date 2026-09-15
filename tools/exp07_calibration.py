"""Released-checkpoint calibration: the pre-registered gate of plan section 2.

Before any new-arm number is read, the released ``checkpoints/xRIR_seen.pth`` is evaluated
on the seen split at K = 8 under the five registered evaluation seeds, and its five-seed
means are compared with the historical full-split reproduction (EDT 0.0389 s, C50
1.029 dB, T60 7.27 %) under the registered rule
``|mean - historical| <= 3 * sd + 0.02 * |historical|``.  This producer derives those
means and SDs (n = 5, ddof = 1) from the five evaluations themselves and binds them; a
hand-written summary is not evidence, and the record binder recomputes everything here
from the runs it binds.

The gate necessarily runs BEFORE the three trainings finish, so the committed approval
file is still all-null and cannot pin anything.  The released row needs no training
provenance (external checkpoint, digest pinned in tools/exp07_profiles.py), and the two
closure pins its admission does need -- the evaluator and the writer -- are therefore
computed from the REVIEWED code at ``--reviewed-commit`` instead of from the approval
file: the same identity the launcher itself checked when it spawned each run.  Every
other check is the one the table producer applies to a ``released_seen`` run.

    python tools/exp07_calibration.py --reviewed-commit <sha> \\
        --runs ckpt/exp07/eval/released_seen_k8_seed{42..46}_k0 \\
        --json ckpt/exp07/results/CALIBRATION_SEEN_V1.json
"""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from tools import exp07_table as table
from tools import provenance as p
from tools.exp07_profiles import (CALIBRATION, CALIBRATION_METRICS, calibration_tolerance,
                                  get_profile, json_value)
from tools.paired_compare import REPO, _equal, admit_run, producer_identity

PROFILE_NAME = 'CALIBRATION_SEEN_V1'
CLOSURE_MODULES = (('evaluator', 'tools.exp07_eval'), ('writer', 'tools.exp07_eval_launch'))
# The registered historical values, as a module attribute so a synthetic cohort can be
# calibrated against its own scale in tests; production reads the frozen profile literal.
HISTORICAL = dict(CALIBRATION['historical'])


def require(ok, message):
    if not ok:
        raise ValueError(message)


def released_arm(profile):
    arms = [arm for arm in profile['arms'] if arm['reference']]
    require(len(arms) == 1 and arms[0]['role'] == CALIBRATION['role'], 'one released arm')
    require(arms[0]['sha256'] is not None, 'the released checkpoint digest is not pinned')
    return arms[0]


def closure_pins(reviewed_commit):
    """The reviewed evaluator and writer closures, in the approval file's shape."""
    closures = {}
    for name, module in CLOSURE_MODULES:
        records, digest = p.closure_record(p.source_closure(module, REPO), reviewed_commit, REPO)
        require(records and all(record['reviewed_blob_sha256'] is not None and
                                record['reviewed_blob_sha256'] == record['working_tree_sha256'] and
                                not record['commits_after_reviewed'] for record in records),
                name + ' closure differs from the reviewed commit')
        closures[name] = digest
    return {'closures': closures, 'checkpoints': {}}


def admit(directories, profile, pins, producer):
    """Admit the five released runs exactly as the table producer admits that role."""
    arm, shot = released_arm(profile), CALIBRATION['num_shot']
    deviations, waivers, inputs, data_stats, cache = [], set(), {}, {}, {}
    runs, paths = {}, {}
    def check(ok, message):
        if not ok:
            deviations.append(message)
    directories = sorted(str(Path(item).resolve()) for item in directories)
    require(len(set(directories)) == len(directories), 'duplicate run directories')
    for directory in directories:
        try:
            fields = json.loads((Path(directory) / 'eval_manifest.json').read_text())
            require(_equal(fields.get('num_shot'), shot), 'not a K = {} run'.format(shot))
            require((Path(fields['repo']) / fields['checkpoint']).resolve() ==
                    (REPO / arm['checkpoint']).resolve(), 'not the released checkpoint')
            contract = table.run_contract(directory, arm, profile, pins, cache)
            waivers.update(contract['waivers'])
            for path, digest in contract['inputs'].items():
                check(inputs.setdefault(path, digest) == digest, 'input changed: ' + path)
            run = admit_run(directory, arm, shot, profile, pins, check, inputs, data_stats)
            seed = run['meta']['manifest_seed']
            check(seed not in runs, 'duplicate evaluation seed: ' + str(seed))
            runs[seed], paths[seed] = run, directory
        except (ValueError, TypeError, KeyError, OSError, IndexError) as error:
            check(False, '{}: {}'.format(directory, error))
    check(sorted(runs) == sorted(CALIBRATION['seeds']), 'the five registered evaluation seeds')
    check(producer['sha256'] is not None, 'producer closure')
    deviations = [item for item in deviations if item not in waivers]
    require(not deviations, 'calibration admission failed: ' + '; '.join(deviations))
    return runs, paths, inputs


def seed_means(payloads, k=None):
    """metric -> seed -> that seed's mean over ITS finite queries, in native units."""
    k = CALIBRATION['k'] if k is None else k
    result = {}
    for metric in CALIBRATION_METRICS:
        source, per_seed = CALIBRATION['sources'][metric], {}
        for seed in sorted(payloads):
            cell = payloads[seed]['P'][str(k)]
            require(source in cell, 'missing metric {} for seed {}'.format(metric, seed))
            values = np.asarray(cell[source], dtype=float)
            finite = values[np.isfinite(values)]
            require(finite.size, 'zero finite queries: {} seed {}'.format(metric, seed))
            per_seed[int(seed)] = float(finite.mean())
        result[metric] = per_seed
    return result


def summarise(per_metric):
    """Apply the registered acceptance rule to five-seed means and sample SDs."""
    metrics = {}
    for metric, per_seed in sorted(per_metric.items()):
        require(sorted(per_seed) == sorted(CALIBRATION['seeds']), 'seed coverage: ' + metric)
        means = np.asarray([per_seed[seed] for seed in sorted(per_seed)], dtype=float)
        require(np.isfinite(means).all(), 'nonfinite seed mean: ' + metric)
        mean = float(means.mean())
        sd = float(means.std(ddof=CALIBRATION['ddof']))
        historical = HISTORICAL[metric]
        require(np.isfinite([mean, sd, historical]).all(), 'nonfinite calibration: ' + metric)
        tolerance = calibration_tolerance(sd, historical)
        metrics[metric] = dict(mean=mean, sd=sd, historical=historical, tolerance=tolerance,
                               difference=mean - historical, n_seeds=len(per_seed),
                               passed=bool(abs(mean - historical) <= tolerance),
                               unit=CALIBRATION['units'][metric],
                               source=CALIBRATION['sources'][metric],
                               per_seed={str(seed): per_seed[seed] for seed in sorted(per_seed)})
    return metrics


def measure(directories, k=None):
    """The binder's route: recompute the summary from the per-sample arrays on disk."""
    payloads = {}
    for directory in sorted(str(Path(item).resolve()) for item in directories):
        sample = json.loads((Path(directory) / 'per_sample_yaw.json').read_text())
        seed = sample['meta']['manifest_seed']
        require(seed not in payloads, 'duplicate evaluation seed: ' + str(seed))
        payloads[seed] = sample
    return summarise(seed_means(payloads, k))


def build_calibration(directories, reviewed_commit, profile=None, pins=None, producer=None):
    """Admit the five released runs and derive the registered calibration; write nothing."""
    profile = json_value(get_profile('TABLE_SEEN_V1')) if profile is None else profile
    pins = closure_pins(reviewed_commit) if pins is None else pins
    producer = producer_identity('tools.exp07_calibration') if producer is None else producer
    runs, paths, inputs = admit(directories, profile, pins, producer)
    arm = released_arm(profile)
    metrics = summarise(seed_means(runs, profile['grid'][0]))
    result = dict(schema_version=1, profile_name=PROFILE_NAME, role=CALIBRATION['role'],
                  protocol=CALIBRATION['protocol'], num_shot=CALIBRATION['num_shot'],
                  k=profile['grid'][0], condition=profile['condition'],
                  passed=all(cell['passed'] for cell in metrics.values()), metrics=metrics,
                  rule=json_value(CALIBRATION), reviewed_commit=reviewed_commit,
                  checkpoint=dict(path=arm['checkpoint'], sha256=arm['sha256']),
                  manifest_hashes={str(seed): runs[seed]['meta']['manifest_hash']
                                   for seed in sorted(runs)},
                  seeds=sorted(runs), inputs=inputs,
                  run_flags={paths[seed]: seed for seed in sorted(paths)},
                  producer_closure_sha256=producer['sha256'])
    result['profile_digest'] = hashlib.sha256(json.dumps(
        result['rule'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return result, dict(inputs=inputs, producer=producer,
                        runs={str(seed): paths[seed] for seed in sorted(paths)})


def write_outputs(result, admitted, json_path, command=()):
    """Publish the canonical JSON and its sidecar exclusively, or leave nothing behind."""
    output = Path(json_path).absolute()
    sidecar = Path(str(output) + '.provenance.json')
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    for path, digest in sorted(result['inputs'].items()):
        require(p.sha256_file(path) == digest, 'input changed before publication: ' + path)
    digest = p.write_manifest(output, result)
    try:
        p.write_manifest(sidecar, dict(
            schema_version=1, profile_name=PROFILE_NAME, profile_digest=result['profile_digest'],
            inputs=result['inputs'], run_flags=result['run_flags'], producer=admitted['producer'],
            reviewed_commit=result['reviewed_commit'], generation_command=list(command),
            generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            outputs={str(output): digest}))
    except BaseException:
        output.unlink()
        raise
    require(hashlib.sha256(payload).hexdigest() == digest, 'canonical encoding')
    return digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True,
                        help='the five released_seen K = 8 seen evaluations')
    parser.add_argument('--reviewed-commit', required=True)
    parser.add_argument('--json', required=True)
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    result, admitted = build_calibration(args.runs, args.reviewed_commit)
    command = ['exp07_calibration.py', '--reviewed-commit', args.reviewed_commit, '--runs']
    command += [str(Path(item).resolve()) for item in args.runs]
    command += ['--json', str(Path(args.json).absolute())]
    write_outputs(result, admitted, args.json, command)
    for metric in CALIBRATION_METRICS:
        cell = result['metrics'][metric]
        print('{} {}: mean {:.6g} vs historical {:.6g}, sd {:.6g}, tolerance {:.6g} -> {}'.format(
            metric, cell['unit'], cell['mean'], cell['historical'], cell['sd'], cell['tolerance'],
            'PASSED' if cell['passed'] else 'FAILED'), flush=True)
    print('CALIBRATION ' + ('PASSED' if result['passed'] else
                            'FAILED: investigate before admitting any seen-arm evaluation'))
    return result


if __name__ == '__main__':
    try:
        if not main()['passed']:
            raise SystemExit(1)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        raise SystemExit(str(error))
