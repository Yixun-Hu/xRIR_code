"""Exp_05 paired capacity inference; all numerical primitives reuse exp_04 imports."""
import json
from pathlib import Path
import numpy as np
from tools import provenance
from tools.exp05_profiles import get_profile, json_value, load_approved_digests
from tools.paired_compare import (one_sided_upper, two_sided_interval, five_seed_mean,
    cell_mask, rho_bootstrap, convergence, admit_run, admit_runs, producer_identity,
    recheck_inputs, write_outputs)
from tools.paired_stats import equivalence_tost, bonferroni_alpha
from tools.summarize_yaw import rooms_from_paths
from tools.paired_compare import _digest, _equal


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
    fields, completion, sample, metrics = [json.loads((directory / name).read_text()) for name in
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
        return dict(evaluator='tools.exp04_eval', sha256=digest, tier_source='pinned historical M checkpoint', waivers=waivers)
    require('train_args' in names, 'missing train_args binding')
    root = Path(fields['repo'])
    path = (root / fields['checkpoint']).resolve().parent / 'args.json'
    binding = fields['mutable_inputs']['train_args']
    require((root / binding['path']).resolve() == path, 'train_args binding path')
    raw = path.read_bytes()
    require(provenance.sha256_file(path) == binding['sha256'], 'train_args bytes')
    recorded = json.loads(raw)
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
    return dict(evaluator='tools.exp05_eval', sha256=digest, tier_source='bound args.json', waivers=waivers)
