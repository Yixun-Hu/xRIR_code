"""Profile-driven paired inference for exp_04; pure statistical functions."""
from decimal import Decimal, ROUND_CEILING
import numbers

import numpy as np

from tools.paired_stats import (
    _bootstrap_ratios, relative_degradation_bootstrap, equivalence_tost,
    diff_in_diff_bootstrap, bonferroni_alpha, valid_mask,
)


def _quantile(samples, probability):
    """Inverse empirical CDF: rank ceil(p*n), with ranks starting at one."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("quantile samples must be a nonempty finite 1-D array")
    # Decimal ranks avoid a binary rounding step promoting rank 3 to 4
    # for alpha=.7 and n=10. The input decimal is interpreted literally.
    probability = Decimal(str(probability))
    if not probability.is_finite() or not 0 < probability < 1:
        raise ValueError("quantile probability must be in (0, 1)")
    rank = int((probability * values.size).to_integral_value(rounding=ROUND_CEILING)) - 1
    return float(np.partition(values, rank)[rank])


def one_sided_upper(samples, alpha):
    """Smallest v with >= ceil((1-alpha)*n) samples <= v; no interpolation.

    For [1, 2, 3, 4] and alpha=.25 the upper bound is 3. A bound equal
    to a practical margin fails the strict confirmatory decision rule.
    """
    return _quantile(samples, Decimal(1) - Decimal(str(alpha)))


def two_sided_interval(samples, alpha):
    """Exact [Q_(alpha/2), Q_(1-alpha/2)] order-statistic interval."""
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    tail = Decimal(str(alpha)) / 2
    return (_quantile(samples, tail), _quantile(samples, Decimal(1) - tail))


def _seed_values(values):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != 5:
        raise ValueError("expected five seed arrays with shape (5, n)")
    return matrix


def five_seed_mean(per_seed_arrays):
    """Plain five-seed mean; any non-finite seed makes the query NaN."""
    values = _seed_values(per_seed_arrays)
    with np.errstate(invalid="ignore", over="ignore"):
        means = values.mean(axis=0)
    means[~np.isfinite(values).all(axis=0)] = np.nan
    return means


def _validity_counts(baseline, paired):
    return {"n": int(baseline.size),
            "baseline_invalid": int((~baseline).sum()),
            "newly_invalid": int((baseline & ~paired).sum()),
            "excluded": int((~paired).sum()), "valid": int(paired.sum())}


def cell_mask(e0_a, ek_a=None, e0_b=None, ek_b=None, seeds=(42, 43, 44, 45, 46)):
    """H1: baseline both arms; H2: baseline/angle both; TOST: aug only.

    Per-seed exclusions condition on that seed's baseline. Arm totals
    condition on all five baseline seeds; joint totals on both arms.
    """
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("exactly five distinct seed labels are required")
    if (ek_b is not None and e0_b is None) or (
            e0_b is not None and (ek_a is None) != (ek_b is None)):
        raise ValueError("both arms require corresponding baseline/angle arrays")
    if e0_b is None and ek_a is None:
        raise ValueError("a cell requires both arms or a baseline/angle pair")
    baseline_masks, paired_masks, counts = [], [], {}
    shape = _seed_values(e0_a).shape
    arms = [("a", e0_a, ek_a)]
    if e0_b is not None:
        arms.append(("b", e0_b, ek_b))
    for arm, base, angle in arms:
        base = _seed_values(base)
        angle = base if angle is None else _seed_values(angle)
        if base.shape != shape or angle.shape != shape:
            raise ValueError("all seed arrays must have the same query shape")
        base_valid = np.isfinite(base)
        paired = np.stack([valid_mask(x, y) for x, y in zip(base, angle)])
        baseline_masks.append(base_valid.all(axis=0))
        paired_masks.append(paired.all(axis=0))
        counts[arm] = {"total": _validity_counts(baseline_masks[-1], paired_masks[-1]),
                       "seeds": {seed: _validity_counts(b, p) for seed, b, p in
                                 zip(seeds, base_valid, paired)}}
    baseline, mask = np.logical_and.reduce(baseline_masks), np.logical_and.reduce(paired_masks)
    counts["joint"] = _validity_counts(baseline, mask)
    if not mask.any():
        raise ValueError("empty cell after all-seed finite mask")
    return mask, counts


