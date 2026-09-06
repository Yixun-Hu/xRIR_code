"""Tests for :mod:`tools.paired_stats` (exp_03, yaw_rotation_degradation).

The confirmatory statistic of the experiment is the relative degradation
``r_k = (mean(e_k) - mean(e_0)) / mean(e_0)`` of a per-sample error, with the ratio
recomputed *inside* every paired bootstrap resample, and a decision rule that fires
only when the (Bonferroni-adjusted) lower bound clears a practical margin.
"""
import numpy as np
import pytest

from tools.paired_stats import invalid_fraction, relative_degradation_bootstrap, valid_mask


def _paired(n=2000, ratio=1.2, noise=0.0, seed=0):
    """Positive paired errors; ``ek`` is ``ratio`` times ``e0``, optionally jittered."""
    rng = np.random.default_rng(seed)
    e0 = rng.lognormal(mean=0.0, sigma=0.5, size=n)
    ek = e0 * ratio
    if noise:
        ek = ek * np.exp(rng.normal(0.0, noise, size=n))
    return e0, ek


# --------------------------------------------------------------------------------------
# T10 -- valid_mask / invalid_fraction
# --------------------------------------------------------------------------------------
def test_valid_mask_requires_both_sides_finite():
    e0 = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.inf])
    ek = np.array([1.0, np.nan, 3.0, 4.0, -np.inf, 6.0])
    got = valid_mask(e0, ek)
    assert got.dtype == np.bool_
    assert got.tolist() == [True, False, False, True, False, False]
    assert valid_mask([1.0, 2.0], [3.0, 4.0]).tolist() == [True, True]


def test_valid_mask_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        valid_mask(np.zeros(3), np.zeros(4))


def test_invalid_fraction_counts_non_finite_entries():
    assert invalid_fraction(np.array([1.0, 2.0, 3.0, 4.0])) == 0.0
    assert invalid_fraction(np.array([1.0, np.nan, np.inf, 4.0])) == 0.5
    assert invalid_fraction(np.array([np.nan])) == 1.0


# --------------------------------------------------------------------------------------
# T10 -- relative_degradation_bootstrap
# --------------------------------------------------------------------------------------
def test_relative_degradation_is_exactly_zero_for_identical_errors():
    e0, _ = _paired(n=500)
    res = relative_degradation_bootstrap(e0, e0.copy(), n_boot=500, seed=0)
    assert set(res) == {"r", "lo", "hi", "n", "n_boot", "alpha", "unit"}
    assert res["r"] == 0.0
    assert res["lo"] <= 0.0 <= res["hi"]
    assert res["n"] == 500 and res["n_boot"] == 500 and res["alpha"] == 0.05
    assert res["unit"] == "pair"


def test_relative_degradation_recovers_a_20_percent_inflation():
    e0, ek = _paired(n=2000, ratio=1.2)
    res = relative_degradation_bootstrap(e0, ek, n_boot=2000, seed=0)
    assert res["r"] == pytest.approx(0.20, abs=1e-9)
    assert res["lo"] > 0.10, "a clean +20% degradation must clear the +10% margin"
    assert res["lo"] <= res["r"] <= res["hi"]


def test_relative_degradation_interval_brackets_a_noisy_effect():
    e0, ek = _paired(n=2000, ratio=1.2, noise=0.3, seed=1)
    res = relative_degradation_bootstrap(e0, ek, n_boot=2000, seed=0)
    assert res["lo"] < res["r"] < res["hi"], "a noisy effect must give a non-degenerate CI"
    assert res["lo"] > 0.0
    wide = relative_degradation_bootstrap(e0, ek, n_boot=2000, alpha=0.01, seed=0)
    assert wide["lo"] < res["lo"] and wide["hi"] > res["hi"], "alpha=0.01 must widen the CI"


def test_relative_degradation_is_seed_deterministic():
    e0, ek = _paired(n=1000, ratio=1.1, noise=0.4, seed=2)
    a = relative_degradation_bootstrap(e0, ek, n_boot=1000, seed=7)
    b = relative_degradation_bootstrap(e0, ek, n_boot=1000, seed=7)
    c = relative_degradation_bootstrap(e0, ek, n_boot=1000, seed=8)
    assert a == b
    assert (c["lo"], c["hi"]) != (a["lo"], a["hi"])
    assert c["r"] == a["r"], "the point estimate does not depend on the bootstrap seed"


def test_relative_degradation_rejects_degenerate_input():
    e0, ek = _paired(n=50)
    with pytest.raises(ValueError):                       # NaN in e0
        relative_degradation_bootstrap(np.array([1.0, np.nan]), np.array([1.0, 2.0]), n_boot=10)
    with pytest.raises(ValueError):                       # inf in ek
        relative_degradation_bootstrap(np.array([1.0, 2.0]), np.array([1.0, np.inf]), n_boot=10)
    with pytest.raises(ValueError):                       # empty
        relative_degradation_bootstrap(np.array([]), np.array([]), n_boot=10)
    with pytest.raises(ValueError):                       # mean(e0) == 0
        relative_degradation_bootstrap(np.array([-1.0, 1.0]), np.array([1.0, 2.0]), n_boot=10)
    with pytest.raises(ValueError):                       # mismatched lengths
        relative_degradation_bootstrap(e0, ek[:-1], n_boot=10)
    with pytest.raises(ValueError):                       # nonsense alpha / n_boot
        relative_degradation_bootstrap(e0, ek, n_boot=10, alpha=0.0)
    with pytest.raises(ValueError):
        relative_degradation_bootstrap(e0, ek, n_boot=0)
