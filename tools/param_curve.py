"""Exp_05 paired capacity inference; all numerical primitives reuse exp_04 imports."""
import json
import argparse
import hashlib
from pathlib import Path
from types import FunctionType
import numpy as np
from tools import provenance
from tools.exp05_profiles import PROFILES, get_profile, json_value, load_approved_digests
from tools.paired_compare import (one_sided_upper, two_sided_interval, five_seed_mean,
    cell_mask, rho_bootstrap, convergence, admit_run, admit_runs, producer_identity,
    recheck_inputs, write_outputs)
from tools.paired_stats import equivalence_tost, bonferroni_alpha
from tools.summarize_yaw import rooms_from_paths
from tools.paired_compare import REPO, _digest, _equal


def matrix(runs, metric, k=0):
    return np.asarray([r['P'][str(k)][metric.lower()] for r in runs], dtype=float)


def cohort(queries, mask):
    if not mask.any():
        raise ValueError('empty cohort')
    return dict(n_queries=int(mask.sum()), n_rooms=len(set(rooms_from_paths(queries)[mask])),
                queries=np.asarray(queries)[mask].tolist())


def seed_summary(values, mask):
    means = values[:, mask].mean(axis=1)
    return dict(mean=float(means.mean()), sd=float(means.std(ddof=1)) if len(means) > 1 else None,
                per_seed=means.tolist())


def curve_point(profile, runs, arm, metric, k):
    values = matrix(runs, metric, k)
    mask = np.isfinite(values).all(axis=0)
    queries = runs[0]['query']
    retained = cohort(queries, mask)
    means = five_seed_mean(values)[mask] if len(runs) == 5 else values[0, mask]
    # A constant baseline transforms the shared ratio sampler into a mean sampler.
    scale = float(means.mean()) or 1.0
    def interval(clusters=None):
        draws = rho_bootstrap(means, np.full(means.size, scale), profile['n_boot'],
                              profile['bootstrap_seeds'][0], clusters)['samples']
        return list(two_sided_interval((draws + 1) * scale, profile['curve_alpha']))
    return dict(arm=arm['role'], metric=metric, k=k, encoder=arm['counts']['encoder'],
        full=arm['counts']['full'], counts=json_value(arm['counts']), cohort=retained,
        **seed_summary(values, mask), seed_labels=list(profile['eval_seeds']),
        interval=interval(), room_cluster_interval=interval(rooms_from_paths(queries)[mask]),
        excluded={str(s): np.asarray(queries)[~np.isfinite(v)].tolist()
                  for s, v in zip(profile['eval_seeds'], values)})


def comparison(profile, groups, pairing, metric):
    roles = [a['role'] for a in profile['arms']]
    ia, ib = [roles.index(role) for role in pairing]
    a, b = matrix(groups[ia], metric), matrix(groups[ib], metric)
    mask, exclusions = cell_mask(a, e0_b=b, seeds=profile['eval_seeds'])
    queries = groups[ia][0]['query']
    means = [five_seed_mean(v)[mask] for v in (a, b)]
    seeds = profile['bootstrap_seeds']
    draws = {s: rho_bootstrap(*means, n_boot=profile['n_boot'], seed=s) for s in seeds}
    alpha = profile['interval_alpha']
    local = bonferroni_alpha(profile['family'], profile['alpha'])
    def interval(seed, level):
        return two_sided_interval(draws[seed]['samples'], level)
    def gate(level):
        return convergence(lambda s: interval(s, level), *seeds, tol=profile['convergence_tolerance'])
    room = rho_bootstrap(*means, n_boot=profile['n_boot'], seed=seeds[0],
                         clusters=rooms_from_paths(queries)[mask])
    own_mask = np.isfinite(b).all(axis=0)
    cell = dict(pairing=list(pairing), metric=metric, k=0, estimate=draws[seeds[0]]['rho'],
        companion_interval=list(interval(seeds[0], alpha)),
        room_cluster_interval=list(two_sided_interval(room['samples'], alpha)),
        paired_cohort=cohort(queries, mask), exclusions=exclusions,
        baseline_own=dict(**seed_summary(b, own_mask), cohort=cohort(queries, own_mask)),
        paired_means={role: seed_summary(v, mask) for role, v in zip(pairing, (a, b))},
        convergence={'superiority': gate(alpha)},
        superior=bool(one_sided_upper(draws[seeds[0]]['samples'], local/2) < 0))
    if profile['mode'] == 'targets':
        # Import TOST's validation/definition, then replace its interpolated endpoints.
        tost = equivalence_tost(means[1], means[0], profile['margin'], n_boot=1,
                                alpha=profile['tost_alpha'], seed=seeds[0])
        lo, hi = interval(seeds[0], 2*profile['tost_alpha'])
        cell.update(target=float(means[1].mean()), tost_interval=[lo, hi],
                    equivalent=bool(-tost['margin'] < lo and hi < tost['margin']))
        cell['convergence']['tost'] = gate(2*profile['tost_alpha'])
        cell['reaches_target'] = cell['superior'] or cell['equivalent']
    return cell


def yaw_cell(profile, run, arm, metric, k):
    """Descriptive seed-42 yaw change, paired with k=0 within one arm."""
    base, angle = (matrix([run], metric, x)[0] for x in (0, k))
    mask = np.isfinite(base) & np.isfinite(angle)
    retained = cohort(run['query'], mask)
    def compute(clusters=None):
        return rho_bootstrap(angle[mask], base[mask], profile['n_boot'], profile['bootstrap_seeds'][0], clusters)
    query = compute()
    room = compute(rooms_from_paths(run['query'])[mask])
    return dict(arm=arm['role'], metric=metric, k=k, seed=42, estimate=query['rho'],
        pairing=['{} k={}'.format(arm['role'], k), arm['role'] + ' k=0'], paired_cohort=retained,
        paired_means={'0': float(base[mask].mean()), str(k): float(angle[mask].mean())},
        companion_interval=list(two_sided_interval(query['samples'], profile['curve_alpha'])),
        room_cluster_interval=list(two_sided_interval(room['samples'], profile['curve_alpha'])),
        excluded=np.asarray(run['query'])[~mask].tolist())


def analyze(profile, admitted, exploratory=False):
    """Compute all registered arms/cells; convergence and admission precede verdicts."""
    result = dict(schema_version=1, profile=json_value(profile), profile_digest=_digest(profile),
        exploratory=exploratory, deviations=list(admitted['deviations']),
        inputs=admitted['inputs'], run_flags=admitted['run_flags'], cells=[], curves=[])
    def failure(message):
        if not exploratory:
            raise ValueError(message)
        result['deviations'].append(message)
    if result['deviations'] and not exploratory:
        failure('admission failed: ' + '; '.join(result['deviations']))
    groups = admitted['groups']
    flat = [r for group in groups for r in group]
    paired = (len(groups) == len(profile['arms']) and flat and
        all([r['meta']['manifest_seed'] for r in group] == list(profile['eval_seeds']) for group in groups) and
        all(r['query'] == flat[0]['query'] and r['index'] == flat[0]['index'] for r in flat))
    if not paired:
        failure('analysis unavailable: registered seed/query pairing')
        return result
    for arm, runs in zip(profile['arms'], groups):
        for metric in profile['metrics']['primary'] + profile['metrics']['descriptive']:
            for k in profile['grid']:
                try:
                    result['curves'].append(curve_point(profile, runs, arm, metric, k))
                    if profile['mode'] == 'yaw' and k != 0:
                        result['cells'].append(yaw_cell(profile, runs[0], arm, metric, k))
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    failure('{} {} k={}: {}'.format(arm['role'], metric, k, exc))
    for pairing in profile.get('pairings', ()):
        for metric in profile['metrics']['primary']:
            try:
                cell = comparison(profile, groups, pairing, metric)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                failure('{} {}: {}'.format(pairing, metric, exc))
                continue
            for name, gate in cell['convergence'].items():
                if not gate['passed']:
                    failure('convergence failed: {} {} {}'.format(pairing, metric, name))
            if exploratory:
                for key in ('superior', 'equivalent', 'reaches_target'):
                    cell.pop(key, None)
            result['cells'].append(cell)
    if exploratory or profile['mode'] == 'yaw':
        return result
    if profile['mode'] == 'curve':
        result['verdicts'] = {}
        for metric in profile['metrics']['primary']:
            supported = sum(c['superior'] for c in result['cells'] if c['metric'] == metric)
            result['verdicts'][metric] = ('dominance supported on {} at the three tested tiers'.format(metric)
                if supported == 3 else 'partial' if supported else 'not supported')
    else:
        result['parameter_ratios'] = []
        arms = {a['role']: a for a in profile['arms']}
        for pairing in profile['pairings']:
            cells = [c for c in result['cells'] if c['pairing'] == list(pairing)]
            if len(cells) == 2 and all(c['reaches_target'] for c in cells):
                a, b = [arms[role]['counts']['encoder'] for role in pairing]
                result['parameter_ratios'].append(dict(pairing=list(pairing), numerator=a, denominator=b,
                    ratio=a/b, reduction_factor=b/a, scope='each metric cell paired cohort',
                    cohorts={c['metric']: c['paired_cohort'] for c in cells}))
    return result


def run_contract(directory, arm, profile, pins):
    """Validate exp05 additions before waiving any exact exp04 compatibility message."""
    directory = Path(directory).resolve()
    inputs = {}
    def read(path):
        raw = path.read_bytes()
        inputs[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    fields, completion, sample, metrics = [read(directory / name) for name in
        ('eval_manifest.json', 'completion.json', 'per_sample_yaw.json', 'metrics_yaw.json')]
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    require(_equal(fields.get('decomposition_batches'), 0), 'decomposition_batches')
    digest = fields['source_closures']['entrypoint']['sha256']
    current = digest is not None and digest == pins['closures']['evaluator']
    old = (arm['tier'] == 'M' and digest is not None and
           digest == pins['closures'].get('evaluator_exp04'))
    require(current or old, 'unapproved evaluator; re-evaluate M' if arm['tier'] == 'M' else 'unapproved evaluator')
    names = set(fields['mutable_inputs'])
    require(names <= {'control_args', 'train_args', 'train_manifest', 'train_completion', 'probe_receipt'},
            'unknown mutable binding')
    waivers = []
    if old and not current:
        require('train_args' not in names, 'legacy evaluator has unexpected train_args binding')
        waivers.append(str(directory) + ': evaluator approved closure')
        return dict(evaluator='tools.exp04_eval', sha256=digest, tier_source='pinned historical M checkpoint',
                    waivers=waivers, inputs=inputs)
    require('train_args' in names, 'missing train_args binding')
    root = Path(fields['repo'])
    path = (root / fields['checkpoint']).resolve().parent / 'args.json'
    binding = fields['mutable_inputs']['train_args']
    require((root / binding['path']).resolve() == path, 'train_args binding path')
    recorded = read(path)
    require(inputs[str(path.resolve())] == binding['sha256'], 'train_args bytes')
    config = {'vit_' + key: value for key, value in arm['config'].items()}
    legacy = not any(key in recorded for key in config) and 'tier' not in recorded
    require(not legacy or arm['tier'] == 'M', 'legacy args require tier M')
    if not legacy:
        require(all(_equal(recorded.get(key), value) for key, value in config.items()), 'args tier configuration')
        require(recorded.get('tier', arm['tier']) == arm['tier'], 'args tier')
        require(_equal(recorded.get('param_counts'), arm['counts']), 'args tier counts')
    require(recorded.get('backbone') == arm['backbone'], 'args backbone')
    for key, value in dict(num_shot=8, epochs=12, batch_size=32, accum_steps=2, seed=0).items():
        require(_equal(recorded.get(key), value), 'args ' + key)
    require(_equal(recorded.get('yaw_aug', 0), 0) and _equal(recorded.get('no_save', False), False), 'args augmentation/save')
    expected = dict(config, tier=arm['tier'], param_counts=arm['counts'], legacy_M=legacy,
                    args_json_sha256=binding['sha256'])
    for payload in (fields, completion, sample['meta'], metrics['meta']):
        require(all(_equal(payload.get(key), value) for key, value in expected.items()), 'tier metadata agreement')
    waivers.append(str(directory) + ': mutable_inputs names')
    return dict(evaluator='tools.exp05_eval', sha256=digest, tier_source='bound args.json', waivers=waivers, inputs=inputs)


def admit(directories, profile, approved=None, producer=None, exploratory=False):
    """Use exp04 admission as a collector, waiving only independently checked differences.

    Neither on-disk artifacts nor the imported producer's globals are modified.
    Single-seed yaw is checked here; no fabricated or duplicated seed runs enter it.
    """
    approved = load_approved_digests() if approved is None else approved
    pins, receipt = approved
    deviations, groups, compatibility, waivers, snapshots = [], [], {}, set(), {}
    def check(ok, message):
        if not ok:
            deviations.append(message)
    check(type(pins['schema_version']) is int and pins['schema_version'] == 1, 'approval schema_version')
    for key in ('evaluator', 'writer', 'training_launcher', 'producer_param_curve'):
        check(pins['closures'][key] is not None, 'profile not yet approved: ' + key)
    for arm in profile['arms']:
        effective = dict(arm)
        if arm['tier'] != 'M':
            checkpoint = pins['checkpoints'][arm['role']]
            check(checkpoint['path'] == arm['checkpoint'] and _equal(checkpoint['epoch'], arm['epoch']),
                  'approved checkpoint path/epoch: ' + arm['role'])
            effective['sha256'] = checkpoint['sha256']
        check(effective['sha256'] is not None, 'profile not yet approved: ' + arm['role'] + ' checkpoint')
        groups.append((effective, profile['num_shot'], []))
    if deviations and not exploratory:
        raise ValueError('admission failed: ' + '; '.join(deviations))
    directories = sorted(str(Path(d).resolve()) for d in directories)
    check(len(set(directories)) == len(directories), 'duplicate run directories')
    for directory in directories:
        arm = None
        try:
            fields = json.loads((Path(directory) / 'eval_manifest.json').read_text())
            matches = [(a, paths) for a, shot, paths in groups if _equal(fields.get('num_shot'), shot)
                and (Path(fields['repo']) / fields['checkpoint']).resolve() ==
                    (REPO / a['checkpoint']).resolve()]
            if len(matches) != 1:
                raise ValueError('unregistered checkpoint/K')
            arm, paths = matches[0]
            paths.append(directory)
            contract = run_contract(directory, arm, profile, pins)
            compatibility[directory] = {k: v for k, v in contract.items() if k not in ('waivers', 'inputs')}
            for path, digest in contract['inputs'].items():
                check(snapshots.get(path, digest) == digest, 'contract input changed: ' + path)
                snapshots[path] = digest
            waivers.update(contract['waivers'])
        except (ValueError, TypeError, KeyError, OSError, IndexError) as exc:
            check(False, '{}: {}{}'.format(directory, exc, '; re-evaluate M' if arm and arm['tier'] == 'M' else ''))
    producer = producer_identity('tools.param_curve') if producer is None else producer
    admitted = admit_runs(profile, groups, approved=approved, exploratory=True, producer=producer,
                          producer_key='producer_param_curve')
    for path, digest in snapshots.items():
        check(admitted['inputs'].get(path) == digest, 'contract input changed: ' + path)
    for (arm, _, _), runs in zip(groups, admitted['groups']):
        seeds = [r['meta']['manifest_seed'] for r in runs]
        valid = (all(type(s) is int for s in seeds) and seeds == list(profile['eval_seeds']))
        check(valid, arm['role'] + ' registered seed set' + ('; re-evaluate M' if arm['tier'] == 'M' else ''))
        if profile['mode'] == 'yaw' and valid:
            waivers.add(arm['role'] + ' seed set')
    deviations.extend(d for d in admitted['deviations'] if d not in waivers)
    if deviations and not exploratory:
        if any(a['tier'] == 'M' and any(path in d or a['role'] in d for path in paths for d in deviations)
               for a, _, paths in groups):
            deviations.append('re-evaluate M')
        raise ValueError('admission failed: ' + '; '.join(deviations))
    admitted.update(deviations=deviations, compatibility=compatibility)
    return admitted


def render_summary(result):
    label = 'EXPLORATORY' if result['exploratory'] else 'DESCRIPTIVE' if result['profile']['mode'] == 'yaw' else 'CONFIRMATORY'
    lines = [label + ' ' + result['profile_name'], 'Profile sha256: ' + result['profile_digest']]
    for point in result['curves']:
        lines.append('{arm} {metric} k={k}: mean={mean:.8g}, seed SD={sd}, CI={interval}, '
            'encoder={encoder}, full={full}, own cohort n={cohort[n_queries]}, rooms={cohort[n_rooms]}'.format(**point))
        lines.append('Excluded queries per seed: ' + str({s: len(q) for s, q in point['excluded'].items()}))
    for cell in result['cells']:
        lines.append('{} {}: rho={:.8g}, CI={}, room CI={}, paired n={}, rooms={}'.format(
            ' vs '.join(cell['pairing']), cell['metric'], cell['estimate'], cell['companion_interval'],
            cell['room_cluster_interval'], cell['paired_cohort']['n_queries'], cell['paired_cohort']['n_rooms']))
        if 'target' in cell:
            lines.append('target on paired cohort={}; baseline own cohort mean={}, n={}, rooms={}; TOST CI={}'.format(
                cell['target'], cell['baseline_own']['mean'], cell['baseline_own']['cohort']['n_queries'],
                cell['baseline_own']['cohort']['n_rooms'], cell['tost_interval']))
        if 'reaches_target' in cell:
            lines.append('Reaches target: ' + str(cell['reaches_target']))
    lines.extend('{}: {}'.format(m, v) for m, v in result.get('verdicts', {}).items())
    for ratio in result.get('parameter_ratios', []):
        lines.append('{} encoder ratio={}/{}={:.8g} (reduction {:.8g}x), {}'.format(
            ' vs '.join(ratio['pairing']), ratio['numerator'], ratio['denominator'], ratio['ratio'],
            ratio['reduction_factor'], ratio['scope']))
    lines.extend('Deviation: ' + d for d in result['deviations'])
    return '\n'.join(lines) + '\n'


def publish(result, admitted, json_path, summary_path):
    """Reuse the imported exclusive writer with an isolated renderer binding.

    exp04's writer has no renderer argument. A private globals copy keeps its
    exact publication/sidecar code and leaves all shared module globals intact.
    """
    writer = FunctionType(write_outputs.__code__, dict(write_outputs.__globals__, render_summary=render_summary),
                          write_outputs.__name__, write_outputs.__defaults__, write_outputs.__closure__)
    recheck_inputs(admitted)
    writer(result, admitted, json_path, summary_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=tuple(PROFILES), required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--json', required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--exploratory', action='store_true')
    args = parser.parse_args(argv)
    profile = get_profile(args.profile)
    admitted = admit(args.runs, profile, exploratory=args.exploratory)
    result = analyze(profile, admitted, args.exploratory)
    result.update(profile_name=args.profile, compatibility=admitted['compatibility'])
    publish(result, admitted, args.json, args.summary)
    return result


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        raise SystemExit(str(error))
