"""Paired bootstrap statistics for the yaw-rotation degradation experiment (exp_03).

Every comparison in exp_03 is *paired*: the same query is evaluated at ``k = 0`` and at
a rotated angle ``k``, so the two error series share their sample index.  The reported
quantity is the **relative degradation**

    r_k = (mean(e_k) - mean(e_0)) / mean(e_0)

recomputed inside every bootstrap resample (never as a ratio of two independently
bootstrapped means), and the confirmatory decision asks whether its Bonferroni-adjusted
lower bound clears a practical margin.

Everything here is pure ``numpy`` and deterministic given ``seed``; no state is shared
between calls.  Inputs must already be masked to the valid samples (see
:func:`valid_mask`) -- a non-finite value raises rather than silently propagating.
"""
from __future__ import annotations

import numpy as np

# Resamples are drawn in chunks of at most this many cells, so a 10000 x 6337 bootstrap
# never materialises a 500 MB index matrix.
_CHUNK_CELLS = 2000000


def _as_1d(values, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("{} must be 1-D, got shape {}".format(name, array.shape))
    return array


def valid_mask(e0, ek):
    """Boolean mask of the pairs that are finite on *both* sides.

    This is the pre-registered validity rule: a sample enters the paired analysis only
    if its metric is finite at ``k = 0`` *and* at the compared angle.

    Args:
        e0: per-sample errors at ``k = 0``.
        ek: per-sample errors at the compared angle.

    Returns:
        A boolean ``np.ndarray`` of the common length.

    Raises:
        ValueError: if the two series have different lengths or are not 1-D.
    """
    a, b = _as_1d(e0, "e0"), _as_1d(ek, "ek")
    if a.shape != b.shape:
        raise ValueError("e0 and ek must have the same length, got {} and {}".format(
            a.size, b.size))
    return np.isfinite(a) & np.isfinite(b)


def invalid_fraction(ek):
    """Fraction of non-finite entries in ``ek`` (0.0 for an empty series).

    Reported per angle as a degradation indicator in its own right: a rotation that
    makes the metric undefined more often has degraded the prediction even when the
    surviving samples look healthy.
    """
    a = _as_1d(ek, "ek")
    if a.size == 0:
        return 0.0
    return float(np.count_nonzero(~np.isfinite(a)) / a.size)


def _check_pair(e0, ek, name0="e0", namek="ek"):
    """Validate one paired series and return it as float64 arrays."""
    a, b = _as_1d(e0, name0), _as_1d(ek, namek)
    if a.shape != b.shape:
        raise ValueError("{} and {} must have the same length, got {} and {}".format(
            name0, namek, a.size, b.size))
    if a.size == 0:
        raise ValueError("{}/{} are empty".format(name0, namek))
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("{}/{} must be finite; mask them with valid_mask() first".format(
            name0, namek))
    if a.mean() == 0.0:
        raise ValueError("mean({}) == 0, the relative degradation is undefined".format(name0))
    return a, b


def _check_boot_args(n_boot, alpha, clusters=None, n=None):
    if int(n_boot) < 1:
        raise ValueError("n_boot must be >= 1, got {}".format(n_boot))
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must lie in (0, 1), got {}".format(alpha))
    if clusters is not None:
        ids = np.asarray(clusters)
        if ids.ndim != 1 or ids.size != n:
            raise ValueError("clusters must give one id per pair ({} expected, got {})".format(
                n, ids.size))


def _bootstrap_ratios(pairs, n_boot, seed, clusters=None):
    """Bootstrap ``r`` for several paired series over the *same* resamples.

    Sharing the draws is what makes a difference-in-differences of two models paired:
    both models are recomputed on the identical resampled set of queries.

    Args:
        pairs: list of ``(e0, ek)`` float64 arrays, all of the same length.
        n_boot: number of resamples.
        seed: ``np.random.default_rng`` seed.
        clusters: ``None`` to resample pairs, or one cluster id per pair to resample
            whole clusters with replacement (multiplicity weights).

    Returns:
        A list of ``[n_boot]`` arrays of ratios, one per input pair.
    """
    n = pairs[0][0].size
    rng = np.random.default_rng(seed)
    out = [np.empty(n_boot, dtype=np.float64) for _ in pairs]

    with np.errstate(divide="ignore", invalid="ignore"):
        if clusters is None:
            chunk = max(1, min(n_boot, _CHUNK_CELLS // max(n, 1)))
            done = 0
            while done < n_boot:
                size = min(chunk, n_boot - done)
                idx = rng.integers(0, n, size=(size, n))
                for j, (e0, ek) in enumerate(pairs):
                    mean0 = e0[idx].mean(axis=1)
                    meank = ek[idx].mean(axis=1)
                    out[j][done:done + size] = (meank - mean0) / mean0
                done += size
        else:
            # Cluster resampling: a cluster drawn m times enters with weight m, so the
            # resampled mean is (sum_c w_c * sum(e over c)) / (sum_c w_c * n_c) -- one
            # matrix product per chunk instead of materialising the resampled pairs.
            _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
            n_cl = int(inverse.max()) + 1
            counts = np.bincount(inverse, minlength=n_cl).astype(np.float64)
            sums = [(np.bincount(inverse, weights=e0, minlength=n_cl),
                     np.bincount(inverse, weights=ek, minlength=n_cl)) for e0, ek in pairs]
            chunk = max(1, min(n_boot, _CHUNK_CELLS // max(n_cl, 1)))
            done = 0
            while done < n_boot:
                size = min(chunk, n_boot - done)
                draws = rng.integers(0, n_cl, size=(size, n_cl))
                weights = np.zeros((size, n_cl), dtype=np.float64)
                np.add.at(weights, (np.repeat(np.arange(size), n_cl), draws.ravel()), 1.0)
                denom = weights @ counts               # >= n_cl > 0, never degenerate
                for j, (sum0, sumk) in enumerate(sums):
                    mean0 = (weights @ sum0) / denom
                    meank = (weights @ sumk) / denom
                    out[j][done:done + size] = (meank - mean0) / mean0
                done += size
    return out


def _percentile_interval(boots, alpha):
    """Two-sided percentile interval at level ``1 - alpha``."""
    lo, hi = np.nanpercentile(boots, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def relative_degradation_bootstrap(e0, ek, n_boot=10000, alpha=0.05, seed=0, clusters=None):
    """Relative degradation ``r`` with a paired percentile bootstrap interval.

    Args:
        e0: per-sample errors at ``k = 0`` (finite, non-empty, ``mean != 0``).
        ek: per-sample errors at the compared angle, same length and pairing.
        n_boot: bootstrap resamples.
        alpha: two-sided level; the interval is a ``1 - alpha`` percentile interval.
        seed: bootstrap seed.
        clusters: optional cluster id per pair (e.g. the room). When given, whole
            clusters are resampled with replacement -- the secondary,
            population-of-rooms robustness check, expected to be much wider.

    Returns:
        ``{"r", "lo", "hi", "n", "n_boot", "alpha", "unit"}`` with ``unit`` either
        ``"pair"`` or ``"cluster"``.  ``n`` always counts pairs.

    Raises:
        ValueError: on non-finite, empty or mismatched inputs, ``mean(e0) == 0``, or
            invalid ``n_boot`` / ``alpha`` / ``clusters``.
    """
    a, b = _check_pair(e0, ek)
    _check_boot_args(n_boot, alpha, clusters, a.size)
    boots = _bootstrap_ratios([(a, b)], int(n_boot), seed, clusters)[0]
    lo, hi = _percentile_interval(boots, float(alpha))
    return {"r": float((b.mean() - a.mean()) / a.mean()), "lo": lo, "hi": hi,
            "n": int(a.size), "n_boot": int(n_boot), "alpha": float(alpha),
            "unit": "pair" if clusters is None else "cluster"}
