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


def _samples(pairs, n_boot, seed, clusters):
    if isinstance(n_boot, bool) or not isinstance(n_boot, numbers.Integral):
        raise TypeError("n_boot must be an integer")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    pairs = [(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
             for a, b in pairs]
    return _bootstrap_ratios(pairs, int(n_boot), seed, clusters)


def _sample_metadata(result, samples, n_boot, seed):
    for key in ("lo", "hi", "alpha"):
        result.pop(key, None)
    result.update(samples=samples, n_boot=int(n_boot), seed=seed)
    return result


def rho_bootstrap(err_a, err_b, n_boot=20000, seed=0, clusters=None):
    """rho=(mean(a)-mean(b))/mean(b) on common query/room resamples.

    The imported public primitive validates inputs and supplies the estimate
    using one validation draw. Actual draws use its shared sampler, avoiding
    its interpolated quantiles and a second full bootstrap computation.
    """
    result = relative_degradation_bootstrap(err_b, err_a, n_boot=1, seed=seed,
                                            clusters=clusters)
    samples = _samples([(err_b, err_a)], n_boot, seed, clusters)[0]
    result["rho"] = result.pop("r")
    return _sample_metadata(result, samples, n_boot, seed)


def dk_bootstrap(e0_a, ek_a, e0_b, ek_b, n_boot=20000, seed=0, clusters=None):
    """D_k=r_k(a)-r_k(b); exp_03's definition matches §3 exactly.

    Both arm ratios share the same resample. As for rho, the public primitive
    supplies validation/estimates; exact quantiles are computed by the caller.
    """
    result = diff_in_diff_bootstrap(e0_a, ek_a, e0_b, ek_b, n_boot=1,
                                    seed=seed, clusters=clusters)
    a, b = _samples([(e0_a, ek_a), (e0_b, ek_b)], n_boot, seed, clusters)
    return _sample_metadata(result, a - b, n_boot, seed)


def tost_cell(e0, ek, margin, alpha_local, n_boot=20000, seed=0, clusters=None):
    """One-arm TOST using exact [Q_alpha_local, Q_(1-alpha_local)]."""
    result = equivalence_tost(e0, ek, margin, n_boot=1, alpha=alpha_local,
                              seed=seed, clusters=clusters)
    samples = _samples([(e0, ek)], n_boot, seed, clusters)[0]
    lo, hi = two_sided_interval(samples, 2 * alpha_local)
    result.update(lo=lo, hi=hi, n_boot=int(n_boot), samples=samples, seed=seed,
                  equivalent=bool(-margin < lo and hi < margin))
    return result


def convergence(fn, seed_a=0, seed_b=1, tol=0.10):
    """Maximum endpoint movement / seed-a width on the specified companion.

    The caller supplies the profile's decision companion (and H1 superiority
    companion separately). Identical zero-width intervals pass; changed ones
    fail. Non-finite or reversed intervals fail closed.
    """
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("convergence tolerance must be finite and nonnegative")
    intervals = []
    for seed in (seed_a, seed_b):
        value = fn(seed)
        if isinstance(value, dict):
            value = (value["lo"], value["hi"])
        interval = np.asarray(value, dtype=float)
        if interval.shape != (2,):
            raise ValueError("convergence requires two interval endpoints")
        intervals.append(interval)
    a, b = intervals
    finite = np.isfinite(intervals).all() and a[0] <= a[1] and b[0] <= b[1]
    width = float(a[1] - a[0]) if finite else float("nan")
    movement = float(np.max(np.abs(a - b))) if finite else float("inf")
    ratio = (movement / width if width > 0 else
             0.0 if finite and movement == 0 else float("inf"))
    return {"passed": bool(finite and ratio <= tol), "movement": movement,
            "width": width, "ratio": ratio, "tolerance": tol,
            "seed_a": {"seed": seed_a, "lo": float(a[0]), "hi": float(a[1])},
            "seed_b": {"seed": seed_b, "lo": float(b[0]), "hi": float(b[1])}}


def h1_verdict(edt_upper, c50_upper, margin=0.03):
    edt = np.isfinite(edt_upper) and edt_upper < margin
    c50 = np.isfinite(c50_upper) and c50_upper < margin
    if edt and c50:
        return "non-inferior"
    if edt or c50:
        return "non-inferior on {} only".format("EDT" if edt else "C50")
    return "not shown"


def superiority(upper):
    """Use the upper endpoint of H1's adjusted two-sided companion."""
    return bool(np.isfinite(upper) and upper < 0)


def h2_verdict(c50_uppers):
    if len(c50_uppers) != 4:
        raise ValueError("H2 requires the four primary C50 cells")
    passes = sum(np.isfinite(bound) and bound < 0 for bound in c50_uppers)
    return "supported" if passes == 4 else "partially supported" if passes else "not supported"


def tost_verdict(lo, hi, margin=0.02):
    equivalent = np.isfinite([lo, hi, margin]).all() and 0 < margin and -margin < lo <= hi < margin
    return "equivalent" if equivalent else "equivalence not established"


# Admission imports are at module scope so source_closure sees every dependency.
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from tools import provenance as provenance
from tools.exp04_profiles import get_profile, json_value
from tools.reference_manifest import load_manifest, manifest_hash
from tools.summarize_yaw import load_run, rooms_from_paths, signed_degrees, _check_metrics_reconciliation

REPO = Path(__file__).resolve().parents[1]
_UNBOUND = object()


def _digest(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def producer_identity():
    """Bind imported producer dependencies to committed HEAD and current bytes."""
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    records, digest = provenance.closure_record(
        provenance.source_closure('tools.paired_compare', REPO), commit, REPO)
    if any(r['reviewed_blob_sha256'] != r['working_tree_sha256'] or
           r['reviewed_blob_sha256'] is None for r in records):
        raise ValueError('producer closure differs from HEAD')
    return {'sha256': digest, 'files': records, 'commit': commit}


def _equal(actual, expected):
    """JSON comparison keeps bool, int, float, and missing values distinct."""
    return json.dumps(actual, sort_keys=True) == json.dumps(json_value(expected), sort_keys=True)


def _closure_digest(closure):
    records = closure['files']
    names = [r['path'] for r in records]
    if not records or names != sorted(set(names)) or any(
            r['working_tree_sha256'] != r['reviewed_blob_sha256'] or
            r['reviewed_blob_sha256'] is None or r['commits_after_reviewed'] for r in records):
        raise ValueError('closure records are incomplete or unreviewed')
    # Match tools.provenance.closure_record's deliberately noncompact encoding.
    return hashlib.sha256(json.dumps([[r['path'], r['reviewed_blob_sha256']]
        for r in records], sort_keys=True).encode()).hexdigest()


