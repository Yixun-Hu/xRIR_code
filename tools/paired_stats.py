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

import numbers

import numpy as np

# Resamples are drawn in chunks of at most this many cells, so a 10000 x 6337 bootstrap
# never materialises a 500 MB index matrix.
_CHUNK_CELLS = 2000000


def _check_positive_int(value, name):
    """Reject anything but a positive Python integer (``bool`` and ``float`` included)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError("{} must be an integer, got {!r}".format(name, value))
    if int(value) < 1:
        raise ValueError("{} must be >= 1, got {}".format(name, value))
    return int(value)


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
    """Fraction of *all* non-finite entries in one series (0.0 when empty).

    This is a marginal count: it does not distinguish a sample that was already invalid
    at ``k = 0`` from one the rotation broke.  For the paired split -- which is the
    degradation indicator -- use :func:`paired_validity`.
    """
    a = _as_1d(ek, "ek")
    if a.size == 0:
        return 0.0
    return float(np.count_nonzero(~np.isfinite(a)) / a.size)


def paired_validity(e0, ek):
    """Split a paired series into the four validity categories.

    A rotation can degrade a prediction so far that the metric stops being defined at
    all; that has to be reported, not silently dropped.  Conditioning on a usable
    baseline separates "the rotation broke this sample" from "this sample was never
    measurable".

    Args:
        e0: per-sample errors at ``k = 0`` (non-finite entries allowed).
        ek: per-sample errors at the compared angle.

    Returns:
        ``{"n", "baseline_invalid", "newly_invalid", "recovered", "paired_valid",
        "newly_invalid_frac"}``; the first five are ``int`` counts, and
        ``newly_invalid_frac = newly_invalid / (n - baseline_invalid)`` is ``0.0`` when
        no sample had a usable baseline.

    Raises:
        ValueError: if the two series have different lengths or are not 1-D.
    """
    a, b = _as_1d(e0, "e0"), _as_1d(ek, "ek")
    if a.shape != b.shape:
        raise ValueError("e0 and ek must have the same length, got {} and {}".format(
            a.size, b.size))
    finite0, finitek = np.isfinite(a), np.isfinite(b)
    baseline_invalid = int(np.count_nonzero(~finite0))
    usable = int(a.size) - baseline_invalid
    newly_invalid = int(np.count_nonzero(finite0 & ~finitek))
    return {"n": int(a.size), "baseline_invalid": baseline_invalid,
            "newly_invalid": newly_invalid,
            "recovered": int(np.count_nonzero(~finite0 & finitek)),
            "paired_valid": int(np.count_nonzero(finite0 & finitek)),
            "newly_invalid_frac": (newly_invalid / usable) if usable else 0.0}


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


def _check_boot_args(n_boot, alpha, clusters=None, n=None, max_alpha=1.0):
    _check_positive_int(n_boot, "n_boot")
    if not (0.0 < float(alpha) < max_alpha):
        raise ValueError("alpha must lie in (0, {}), got {}".format(max_alpha, alpha))
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

    Raises:
        ValueError: if any resample leaves ``r`` undefined (a resampled ``mean(e0)`` of
            zero).  Dropping such draws would silently shift the percentiles.
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
    for ratios in out:
        if not np.isfinite(ratios).all():
            raise ValueError(
                "{} of {} resamples left the relative degradation undefined (a resampled "
                "mean(e0) of zero); the errors must be strictly one-signed for a ratio to "
                "be meaningful".format(int(np.count_nonzero(~np.isfinite(ratios))), n_boot))
    return out


def _percentile_interval(boots, alpha):
    """Two-sided percentile interval at level ``1 - alpha``."""
    lo, hi = np.percentile(boots, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
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
        TypeError: if ``n_boot`` is not an ``int``.
        ValueError: on non-finite, empty or mismatched inputs, ``mean(e0) == 0``,
            invalid ``n_boot`` / ``alpha`` / ``clusters``, or a resample that leaves
            ``r`` undefined.
    """
    a, b = _check_pair(e0, ek)
    _check_boot_args(n_boot, alpha, clusters, a.size)
    boots = _bootstrap_ratios([(a, b)], int(n_boot), seed, clusters)[0]
    lo, hi = _percentile_interval(boots, float(alpha))
    return {"r": float((b.mean() - a.mean()) / a.mean()), "lo": lo, "hi": hi,
            "n": int(a.size), "n_boot": int(n_boot), "alpha": float(alpha),
            "unit": "pair" if clusters is None else "cluster"}


def bonferroni_alpha(m, alpha=0.05):
    """Bonferroni-adjusted level for a family of ``m`` tests: ``alpha / m``.

    exp_03's confirmatory family is 2 metrics x 9 non-zero acoustic angles = 18 tests,
    so the pre-registered call is ``bonferroni_alpha(18)`` and every confirmatory
    interval is a ``1 - 0.05/18`` interval.

    Args:
        m: family size, a positive ``int``.
        alpha: family-wise level, in ``(0, 1)``.

    Raises:
        TypeError: if ``m`` is not an ``int`` (``bool`` and ``float`` are rejected).
        ValueError: if ``m < 1`` or ``alpha`` is outside ``(0, 1)``.
    """
    _check_positive_int(m, "the family size m")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must lie in (0, 1), got {}".format(alpha))
    return float(alpha) / int(m)


def verdict_substantial(res, threshold):
    """Whether a degradation result clears a practical margin.

    The pre-registered rule is one-sided and strict: the *lower* bound of the interval
    must exceed ``threshold`` (a lower bound sitting exactly on the margin is not a
    pass), so the claim survives the Bonferroni adjustment it was computed with.

    Args:
        res: a :func:`relative_degradation_bootstrap` result (only ``"lo"`` is read).
        threshold: the practical margin, e.g. ``0.10`` for +10 %.

    Returns:
        ``True`` if ``res["lo"] > threshold``.
    """
    return bool(res["lo"] > threshold)


def equivalence_tost(e0, ek, margin, n_boot=10000, alpha=0.05, seed=0, clusters=None):
    """Two one-sided tests for *equivalence* of ``ek`` and ``e0`` within ``+-margin``.

    The bootstrap form of TOST: the two one-sided tests at level ``alpha`` are jointly
    equivalent to asking whether the ``1 - 2*alpha`` percentile interval of ``r`` lies
    strictly inside ``[-margin, +margin]``.  Used at the patch-aligned angles, where the
    claim is "nearly unchanged" rather than "not significantly changed".

    Args:
        e0: per-sample errors at ``k = 0``.
        ek: per-sample errors at the compared angle.
        margin: finite positive equivalence margin on ``r`` (``0.02`` for +-2 %).
        n_boot: bootstrap resamples.
        alpha: level of *each* one-sided test; must be below 0.5, or ``1 - 2*alpha``
            is not an interval.
        seed: bootstrap seed.
        clusters: optional cluster id per pair, resampled exactly as in
            :func:`relative_degradation_bootstrap`.

    Returns:
        ``{"equivalent", "r", "lo", "hi", "margin", "n", "n_boot", "alpha", "unit"}``
        where ``lo``/``hi`` bound the ``1 - 2*alpha`` interval.

    Raises:
        TypeError: if ``n_boot`` is not an ``int``.
        ValueError: on degenerate input (see :func:`relative_degradation_bootstrap`),
            an ``alpha`` outside ``(0, 0.5)``, or a non-positive, infinite or NaN
            ``margin``.
    """
    a, b = _check_pair(e0, ek)
    _check_boot_args(n_boot, alpha, clusters, a.size, max_alpha=0.5)
    if not (np.isfinite(margin) and float(margin) > 0.0):
        raise ValueError("margin must be finite and positive, got {}".format(margin))
    boots = _bootstrap_ratios([(a, b)], int(n_boot), seed, clusters)[0]
    lo, hi = np.percentile(boots, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return {"equivalent": bool(-float(margin) < lo and hi < float(margin)),
            "r": float((b.mean() - a.mean()) / a.mean()), "lo": float(lo), "hi": float(hi),
            "margin": float(margin), "n": int(a.size), "n_boot": int(n_boot),
            "alpha": float(alpha), "unit": "pair" if clusters is None else "cluster"}


def diff_in_diff_bootstrap(e0_a, ek_a, e0_b, ek_b, n_boot=10000, alpha=0.05, seed=0,
                           clusters=None):
    """Difference in relative degradations of two models, ``d = r_a - r_b``.

    This is the H2 statistic: does the cylindrical backbone degrade *less* than the
    same-budget SimpleViT control?  Both models are evaluated on the same queries, so
    the two ratios are recomputed on the **same** resamples inside the bootstrap; the
    interval then reflects only the cross-model difference, not the shared query noise.
    Comparing a model with itself therefore returns exactly ``0`` with a zero-width
    interval.

    Args:
        e0_a, ek_a: model A's per-sample errors at ``k = 0`` and at the angle.
        e0_b, ek_b: model B's, for the *same* queries in the same order.
        n_boot: bootstrap resamples.
        alpha: two-sided level of the ``1 - alpha`` percentile interval.
        seed: bootstrap seed.
        clusters: optional cluster id per query (see
            :func:`relative_degradation_bootstrap`).

    Returns:
        ``{"d", "lo", "hi", "r_a", "r_b", "n", "n_boot", "alpha", "unit"}``.

    Raises:
        TypeError: if ``n_boot`` is not an ``int``.
        ValueError: on degenerate input, series of different lengths, or a resample
            that leaves either ``r`` undefined.
    """
    a0, ak = _check_pair(e0_a, ek_a, "e0_a", "ek_a")
    b0, bk = _check_pair(e0_b, ek_b, "e0_b", "ek_b")
    if a0.shape != b0.shape:
        raise ValueError("the two models must be evaluated on the same queries, got {} and {}"
                         .format(a0.size, b0.size))
    _check_boot_args(n_boot, alpha, clusters, a0.size)
    boots_a, boots_b = _bootstrap_ratios([(a0, ak), (b0, bk)], int(n_boot), seed, clusters)
    lo, hi = _percentile_interval(boots_a - boots_b, float(alpha))
    r_a = float((ak.mean() - a0.mean()) / a0.mean())
    r_b = float((bk.mean() - b0.mean()) / b0.mean())
    return {"d": r_a - r_b, "lo": lo, "hi": hi, "r_a": r_a, "r_b": r_b,
            "n": int(a0.size), "n_boot": int(n_boot), "alpha": float(alpha),
            "unit": "pair" if clusters is None else "cluster"}
