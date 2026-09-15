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
import numpy as np

from tools.exp07_profiles import get_profile, json_value
from tools.exp07_table import admit
from tools.paired_compare import (REPO, _digest, _samples, cell_mask, convergence,
                                  five_seed_mean, two_sided_interval)
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
