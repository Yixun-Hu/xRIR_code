"""Descriptive exp_07 paired differences on the seen split: no verdict, no decision.

One cell is a pairing at one K and one metric.  The paired cohort is the queries finite
for all five evaluation seeds in BOTH arms (an empty cohort refuses the cell); the
per-query value is each arm's plain five-seed mean; and the two reported statistics are
the ABSOLUTE difference of the arm means in the table's units (EDT in ms) and the
RELATIVE difference (delta / the baseline arm's mean).  Both are recomputed inside each
of the profile's shared query-level resamples -- one call to the shared sampler returns
both series, so the intervals describe the same draws -- and the whole-room cluster
resampling of the same query-weighted statistic is reported as the secondary interval.
Intervals are exact order statistics at the profile's quantiles.

These are checkpoint-conditional descriptive intervals: five evaluation seeds do not
estimate training-seed variability, so no cell carries a verdict, a significance marker
or a superiority/equivalence claim, and ``decision_driving`` is false throughout.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from tools.exp07_profiles import get_profile, json_value
from tools.exp07_table import admit
from tools.paired_compare import (REPO, _digest, _samples, cell_mask, convergence,
                                  five_seed_mean, two_sided_interval, write_outputs)
from tools.results_table import METRICS
from tools.summarize_yaw import rooms_from_paths

BASELINE = 'seen_simple'
# metric name -> (per-sample source name, unit, scale), as the canonical table reports it
UNITS = {spec[0]: (source, spec[1], spec[2]) for source, spec in METRICS.items() if spec}


def matrix(runs, source, k=0):
    return np.asarray([run['P'][str(k)][source] for run in runs], dtype=float)


def cohort(queries, mask):
    if not mask.any():
        raise ValueError('empty cohort')
    rooms = rooms_from_paths(queries)[mask]
    return dict(n_queries=int(mask.sum()), n_rooms=len(set(rooms)),
                retained_sha256=_digest(sorted(np.asarray(queries)[mask].tolist())),
                excluded=np.asarray(queries)[~mask].tolist())


def seed_summary(values, mask):
    means = values[:, mask].mean(axis=1)
    return dict(mean=float(means.mean()), sd=float(means.std(ddof=1)) if len(means) > 1 else None,
                per_seed=means.tolist())


def difference_draws(means_a, means_b, n_boot, seed, clusters=None):
    """Both statistics from ONE set of resamples of the shared sampler.

    The relative series is the sampler's own ratio; the absolute series uses a constant
    baseline of one, which turns the same ratio into ``mean(a - b)`` on that draw.
    """
    relative, shifted = _samples([(means_b, means_a), (np.ones_like(means_a), means_a - means_b)],
                                 n_boot, seed, clusters)
    return {'relative': relative, 'absolute': shifted + 1.0}


def paired_cell(profile, runs_a, runs_b, pairing, metric, shot):
    """One descriptive cell; raises on an empty cohort or a mis-specified profile."""
    alpha, quantiles = profile['interval_alpha'], tuple(profile['quantiles'])
    if quantiles != (alpha / 2, 1 - alpha / 2):
        raise ValueError('the profile quantiles do not match its interval alpha')
    source, unit, scale = UNITS[metric]
    a, b = (matrix(runs, source) * scale for runs in (runs_a, runs_b))
    mask, counts = cell_mask(a, e0_b=b, seeds=profile['eval_seeds'])
    queries = runs_a[0]['query']
    retained = cohort(queries, mask)
    means_a, means_b = (five_seed_mean(values)[mask] for values in (a, b))
    rooms = rooms_from_paths(queries)[mask]
    seed_a, seed_b = profile['bootstrap_seeds']
    draws = {seed: difference_draws(means_a, means_b, profile['n_boot'], seed)
             for seed in (seed_a, seed_b)}
    clustered = difference_draws(means_a, means_b, profile['n_boot'], seed_a, rooms)
    difference = float(means_a.mean() - means_b.mean())
    estimates = {'absolute': difference, 'relative': difference / float(means_b.mean())}
    statistics = {}
    for name in profile['statistics']:
        gate = convergence(lambda seed: two_sided_interval(draws[seed][name], alpha),
                           seed_a, seed_b, profile['convergence_tolerance'])
        statistics[name] = dict(estimate=estimates[name], n_boot=profile['n_boot'],
                                unit=unit if name == 'absolute' else 'fraction of ' + pairing[1],
                                interval=list(two_sided_interval(draws[seed_a][name], alpha)),
                                room_cluster_interval=list(two_sided_interval(clustered[name], alpha)),
                                convergence=gate, reconverge_required=not gate['passed'])
    reference = [list(pair) for pair in profile['reference_pairings']]
    return dict(pairing=list(pairing), metric=metric, num_shot=shot, k=profile['grid'][0],
                decision_driving=profile['decision_driving'], statistics=statistics,
                reference=list(pairing) in reference, quantiles=list(quantiles),
                paired_cohort=retained, n_rooms_retained=len(set(rooms)),
                seed_labels=list(profile['eval_seeds']), source=source,
                exclusions={pairing[0]: counts['a'], pairing[1]: counts['b'],
                            'joint': counts['joint']},
                arm_means={role: seed_summary(values, mask)
                           for role, values in zip(pairing, (a, b))})


def _role_and_shot(profile, directory):
    fields = json.loads((Path(directory) / 'eval_manifest.json').read_text())
    matches = [arm['role'] for arm in profile['arms']
               if (Path(fields['repo']) / fields['checkpoint']).resolve() ==
               (REPO / arm['checkpoint']).resolve()]
    if len(matches) != 1:
        raise ValueError('unregistered checkpoint: ' + str(directory))
    return matches[0], fields.get('num_shot')


def _side(profile, directories, label):
    identified = [_role_and_shot(profile, item) for item in directories]
    roles = {role for role, _ in identified}
    if len(roles) != 1:
        raise ValueError(label + ' must hold exactly one arm, got: ' + ', '.join(sorted(roles)))
    return roles.pop(), sorted({shot for _, shot in identified}, reverse=True)


def build_pairs(runs_a, runs_b, profile=None, approved=None, exploratory=False, producer=None):
    """Admit both arms of one registered pairing and compute its descriptive cells."""
    profile = json_value(get_profile('PAIRS_SEEN_V1')) if profile is None else profile
    role_a, shots_a = _side(profile, runs_a, '--runs-a')
    role_b, shots_b = _side(profile, runs_b, '--runs-b')
    if role_b != BASELINE:
        raise ValueError('--runs-b must be the ' + BASELINE + ' baseline, got ' + role_b)
    pairing = (role_a, role_b)
    if list(pairing) not in [list(pair) for pair in profile['pairings']]:
        raise ValueError('unregistered pairing: ' + ' - '.join(pairing))
    if shots_a != shots_b or not set(shots_a) <= set(profile['num_shot']):
        raise ValueError('both arms must be evaluated at the same registered shot counts')
    arms = [arm for arm in profile['arms'] if arm['role'] in pairing]
    restricted = dict(profile, num_shot=shots_a,
                      arms=sorted(arms, key=lambda arm: pairing.index(arm['role'])))
    groups, admitted = admit(list(runs_a) + list(runs_b), restricted, approved, producer,
                             exploratory, producer_key='producer_pairs')
    indexed = {(arm['role'], shot): runs
               for (arm, shot, _), runs in zip(groups, admitted['groups'])}
    literal = json.loads(json.dumps(json_value(profile)))
    result = dict(schema_version=1, profile_name='PAIRS_SEEN_V1', profile=literal,
                  profile_digest=hashlib.sha256(json.dumps(literal, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                  exploratory=exploratory, pairing=list(pairing), decision_driving=False,
                  verdict_scope=profile['verdict_scope'], cells=[], reconverge_required=[],
                  deviations=list(admitted['deviations']), inputs=admitted['inputs'],
                  run_flags=admitted['run_flags'], contracts=admitted['contracts'],
                  producer_closure_sha256=admitted['producer']['sha256'])
    for shot in shots_a:
        for metric in profile['metrics']['descriptive']:
            label = '{} K={}'.format(metric, shot)
            try:
                cell = paired_cell(profile, indexed[(pairing[0], shot)],
                                   indexed[(pairing[1], shot)], pairing, metric, shot)
            except (ValueError, KeyError, TypeError, IndexError) as error:
                if not exploratory:
                    raise ValueError('{}: {}'.format(label, error))
                result['deviations'].append('{}: {}'.format(label, error))
                continue
            result['cells'].append(cell)
            flagged = sorted(name for name, statistic in cell['statistics'].items()
                             if statistic['reconverge_required'])
            if flagged:
                result['reconverge_required'].append(label + ' ' + '/'.join(flagged))
    result['final'] = not result['reconverge_required'] and not result['deviations']
    return result, admitted


def render_summary(result):
    """Every number here is read from the result the canonical JSON is written from."""
    lines = ['DESCRIPTIVE {} {} - {}'.format(result['profile_name'], *result['pairing']),
             'Profile sha256: ' + result['profile_digest'],
             'No verdict. ' + result['verdict_scope']]
    for cell in result['cells']:
        absolute, relative = (cell['statistics'][name] for name in ('absolute', 'relative'))
        lines.append('{} K={}{}: absolute={:.8g} {} {}, relative={:.8g} {}, rooms {} {}, n={}'
                     .format(cell['metric'], cell['num_shot'],
                             ' [reference]' if cell['reference'] else '',
                             absolute['estimate'], absolute['unit'], absolute['interval'],
                             relative['estimate'], relative['interval'],
                             cell['n_rooms_retained'], relative['room_cluster_interval'],
                             cell['paired_cohort']['n_queries']))
        lines.append('  cohort means: ' + ', '.join(
            '{}={:.8g}'.format(role, summary['mean'])
            for role, summary in sorted(cell['arm_means'].items())))
    lines.extend('Re-run at a larger n_boot: ' + item for item in result['reconverge_required'])
    lines.extend('Deviation: ' + item for item in result['deviations'])
    if not result['final']:
        lines.append('NOT FINAL')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('PAIRS_SEEN_V1',), required=True)
    parser.add_argument('--runs-a', nargs='+', required=True, help="the pairing's first arm")
    parser.add_argument('--runs-b', nargs='+', required=True,
                        help='the ' + BASELINE + ' baseline of every pairing')
    parser.add_argument('--json', required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--exploratory', action='store_true',
                        help='List unapproved pins and refused cells instead of stopping')
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    result, admitted = build_pairs(args.runs_a, args.runs_b, exploratory=args.exploratory)
    write_outputs(result, admitted, args.json, args.summary, render_summary)
    return result


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        raise SystemExit(str(error))
