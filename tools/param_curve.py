"""Exp_05 paired capacity inference; all numerical primitives reuse exp_04 imports."""
import numpy as np
from tools.exp05_profiles import get_profile, json_value, load_approved_digests
from tools.paired_compare import (one_sided_upper, two_sided_interval, five_seed_mean,
    cell_mask, rho_bootstrap, convergence, admit_run, admit_runs, producer_identity,
    recheck_inputs, write_outputs)
from tools.paired_stats import equivalence_tost, bonferroni_alpha
from tools.summarize_yaw import rooms_from_paths


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
