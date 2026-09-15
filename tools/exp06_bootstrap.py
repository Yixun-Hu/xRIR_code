"""exp_06's alpha-aware paired bootstrap (plan v4 section 7).

`sim_to_real.summarize_haa.paired` is the estimator exp_02's canonical numbers came from,
and it is pinned at nominal 95 %. exp_06 needs the same two schemes at Bonferroni-adjusted
levels, so this module reproduces its arithmetic exactly -- one `np.random.default_rng`
advanced through the query-cluster draws first and the two-way draws second, float64
throughout, `np.unique` label order, multiplicity products with weighted denominators and
NumPy's linear `nanpercentile` -- and only the two tail probabilities are configurable. At
alpha = 0.05 every cell of `ckpt/sim2real/stats.json` is reproduced bit for bit.

Convergence follows `tools.paired_compare.convergence`'s endpoint rule with one deliberate
difference: a zero-width interval is refused here rather than allowed to pass, because the
plan registers zero-width intervals as inadmissible for a verdict.
"""
import numbers

import numpy as np

DEFAULT_ALPHA = 0.05
DEFAULT_N_BOOT = 10000
DEFAULT_SEED = 0
DEFAULT_TOL = 0.10
SCHEMES = ('query', 'two_way')


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(name + ' must be a positive integer')
    return int(value)


def bonferroni_alpha(alpha, m):
    """The per-comparison level of a family of `m` two-sided comparisons."""
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
        raise ValueError('alpha must be a real number')
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0, 1)')
    return alpha / _positive_int(m, 'm')


def _tails(alpha):
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
        raise ValueError('alpha must be a real number')
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0, 1)')
    return [100 * (alpha / 2), 100 * (1 - alpha / 2)]


def _summary(samples, tails):
    low, high = np.nanpercentile(samples, tails)
    return {'lo': float(low), 'hi': float(high), 'mean': float(np.nanmean(samples)),
            'sd': float(np.nanstd(samples)), 'n_nan': int(np.isnan(samples).sum())}


def paired_intervals(a, b, clusters, seeds, alpha=DEFAULT_ALPHA, n_boot=DEFAULT_N_BOOT,
                     seed=DEFAULT_SEED, return_samples=False):
    """Paired mean difference a - b with the query-cluster and two-way percentile intervals.

    `clusters` carries one query id per row and `seeds` one training-seed id per row, as in
    exp_02: the first scheme resamples queries (rows of one query move together), the second
    resamples queries and seeds with multiplicity products. Inputs must be finite, aligned
    and nonempty; everything else is a refusal.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    clusters, seeds = np.asarray(clusters), np.asarray(seeds)
    if a.size == 0 or a.shape != b.shape or a.ndim != 1:
        raise ValueError('a and b must be equal-length, nonempty 1-D arrays')
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('a and b must be finite')
    if clusters.shape != a.shape or seeds.shape != a.shape:
        raise ValueError('clusters and seeds must carry one label per row')
    tails = _tails(alpha)
    n_boot = _positive_int(n_boot, 'n_boot')
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')

    rng = np.random.default_rng(int(seed))
    d = a - b
    uq, qi = np.unique(clusters, return_inverse=True)
    query = np.empty(n_boot)
    for t in range(n_boot):
        wq = np.bincount(rng.integers(0, len(uq), len(uq)), minlength=len(uq))[qi]
        query[t] = np.average(d, weights=wq) if wq.sum() else np.nan
    us, si = np.unique(seeds, return_inverse=True)
    two_way = np.empty(n_boot)
    for t in range(n_boot):
        wq = np.bincount(rng.integers(0, len(uq), len(uq)), minlength=len(uq))[qi]
        ws = np.bincount(rng.integers(0, len(us), len(us)), minlength=len(us))[si]
        w = wq * ws
        two_way[t] = np.average(d, weights=w) if w.sum() else np.nan

    result = {'diff': float(d.mean()), 'n': int(a.size), 'n_clusters': int(len(uq)),
              'n_seeds': int(len(us)), 'alpha': float(alpha), 'tails': tails,
              'n_boot': n_boot, 'seed': int(seed),
              'frac_a_lower': float(np.mean(a < b)),
              'query': _summary(query, tails), 'two_way': _summary(two_way, tails)}
    if return_samples:
        result['samples'] = {'query': query, 'two_way': two_way}
    return result


def _endpoints(value, seed):
    if isinstance(value, dict):
        value = (value['lo'], value['hi'])
    interval = np.asarray(value, dtype=float)
    if interval.shape != (2,):
        raise ValueError('seed {} produced no pair of interval endpoints'.format(seed))
    if not np.isfinite(interval).all() or interval[0] > interval[1]:
        raise ValueError('seed {} produced a non-finite or reversed interval'.format(seed))
    return interval


def convergence_endpoints(interval_fn, seed_a=0, seed_b=1, tol=DEFAULT_TOL):
    """Maximum endpoint movement between two bootstrap seeds, over the seed-a width."""
    if not np.isfinite(tol) or tol < 0:
        raise ValueError('the convergence tolerance must be finite and nonnegative')
    try:
        first, second = [_endpoints(interval_fn(seed), seed) for seed in (seed_a, seed_b)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('the interval function did not return endpoints: ' + str(error))
    width = float(first[1] - first[0])
    if width <= 0:
        raise ValueError('a zero-width interval cannot carry a convergence verdict')
    movement = float(np.max(np.abs(first - second)))
    ratio = movement / width
    return {'passed': bool(ratio <= tol), 'movement': movement, 'width': width,
            'ratio': ratio, 'tolerance': float(tol),
            'seed_a': {'seed': seed_a, 'lo': float(first[0]), 'hi': float(first[1])},
            'seed_b': {'seed': seed_b, 'lo': float(second[0]), 'hi': float(second[1])}}


def converged_interval(fn_factory, n_boot=DEFAULT_N_BOOT, seed_a=0, seed_b=1,
                       tol=DEFAULT_TOL, factor=4):
    """Check convergence, quadruple `n_boot` once on failure, then withhold the verdict.

    `fn_factory(n_boot)` returns the interval function of that many resamples. The returned
    interval is always the seed-a interval of the size that converged; a run that never
    converges reports `not_converged` and no interval, as the plan requires.
    """
    n_boot = _positive_int(n_boot, 'n_boot')
    factor = _positive_int(factor, 'factor')
    attempts = []
    for size in (n_boot, n_boot * factor):
        interval_fn = fn_factory(size)
        check = convergence_endpoints(interval_fn, seed_a=seed_a, seed_b=seed_b, tol=tol)
        attempts.append(dict(check, n_boot=size))
        if check['passed']:
            return {'status': 'converged', 'n_boot': size, 'attempts': attempts,
                    'interval': (check['seed_a']['lo'], check['seed_a']['hi'])}
    return {'status': 'not_converged', 'n_boot': None, 'attempts': attempts,
            'interval': None}
