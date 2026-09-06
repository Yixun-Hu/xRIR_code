"""Tests for :mod:`tools.paired_stats` (exp_03, yaw_rotation_degradation).

The confirmatory statistic of the experiment is the relative degradation
``r_k = (mean(e_k) - mean(e_0)) / mean(e_0)`` of a per-sample error, with the ratio
recomputed *inside* every paired bootstrap resample, and a decision rule that fires
only when the (Bonferroni-adjusted) lower bound clears a practical margin.
"""
import numpy as np
import pytest

from tools.paired_stats import (
    bonferroni_alpha,
    diff_in_diff_bootstrap,
    equivalence_tost,
    invalid_fraction,
    paired_validity,
    relative_degradation_bootstrap,
    valid_mask,
    verdict_substantial,
)


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


def test_paired_validity_counts_every_category():
    """Baseline-invalid, newly invalid, recovered and paired-valid are four different
    things, and only the second is a degradation indicator."""
    #                 both ok   e0 bad     ek bad    both bad  both ok
    e0 = np.array([1.0, np.nan, 2.0, np.inf, 3.0])
    ek = np.array([1.0, 2.0, np.nan, np.inf, 4.0])
    got = paired_validity(e0, ek)

    assert got == {"n": 5, "baseline_invalid": 2, "newly_invalid": 1, "recovered": 1,
                   "paired_valid": 2, "newly_invalid_frac": 1.0 / 3.0}
    for key in ("n", "baseline_invalid", "newly_invalid", "recovered", "paired_valid"):
        assert isinstance(got[key], int)
    assert isinstance(got["newly_invalid_frac"], float)
    # The finite-at-k=0 samples split exactly into "still valid" and "newly invalid".
    assert got["paired_valid"] + got["newly_invalid"] == got["n"] - got["baseline_invalid"]
    assert got["paired_valid"] == int(valid_mask(e0, ek).sum())


def test_paired_validity_handles_empty_and_fully_invalid_baselines():
    empty = paired_validity(np.array([]), np.array([]))
    assert empty == {"n": 0, "baseline_invalid": 0, "newly_invalid": 0, "recovered": 0,
                     "paired_valid": 0, "newly_invalid_frac": 0.0}

    # No usable baseline at all: the fraction is 0/0, reported as 0.0 rather than NaN.
    dead = paired_validity(np.array([np.nan, np.inf]), np.array([1.0, 2.0]))
    assert dead["baseline_invalid"] == 2 and dead["recovered"] == 2
    assert dead["newly_invalid"] == 0 and dead["newly_invalid_frac"] == 0.0

    with pytest.raises(ValueError):
        paired_validity(np.zeros(3), np.zeros(4))


def test_invalid_fraction_and_paired_validity_answer_different_questions():
    e0 = np.array([1.0, np.nan, 2.0, 3.0])
    ek = np.array([1.0, 2.0, np.nan, 4.0])
    # invalid_fraction looks at one series in isolation ...
    assert invalid_fraction(ek) == 0.25
    # ... paired_validity conditions on the baseline being usable.
    assert paired_validity(e0, ek)["newly_invalid_frac"] == pytest.approx(1.0 / 3.0)


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


def test_relative_degradation_is_a_ratio_of_means_not_a_mean_of_ratios():
    """The two disagree sharply on skewed errors, and only one of them is r_k."""
    e0 = np.array([1.0, 100.0])
    ek = np.array([2.0, 100.0])
    res = relative_degradation_bootstrap(e0, ek, n_boot=100, seed=0)

    assert res["r"] == (102.0 / 2.0 - 101.0 / 2.0) / (101.0 / 2.0)
    assert res["r"] == pytest.approx(1.0 / 101.0, rel=0, abs=1e-15)
    mean_of_ratios = float(np.mean(ek / e0 - 1.0))
    assert mean_of_ratios == pytest.approx(0.5)
    assert abs(res["r"] - mean_of_ratios) > 0.4, "a mean of per-sample ratios is not r_k"


def test_bootstrap_percentiles_match_an_explicit_resample_enumeration():
    """Exact oracle: same draws, same statistic, numpy's own percentile, same level."""
    e0 = np.array([1.0, 2.0, 4.0])
    ek = np.array([1.5, 2.0, 5.0])
    n_boot, seed, alpha = 1000, 5, 0.05

    res = relative_degradation_bootstrap(e0, ek, n_boot=n_boot, alpha=alpha, seed=seed)

    rng = np.random.default_rng(seed)                      # the documented draw
    idx = rng.integers(0, e0.size, size=(n_boot, e0.size))
    ratios = (ek[idx].mean(axis=1) - e0[idx].mean(axis=1)) / e0[idx].mean(axis=1)
    lo, hi = np.percentile(ratios, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    assert (res["lo"], res["hi"]) == (lo, hi)
    assert res["r"] == (ek.mean() - e0.mean()) / e0.mean()


# --------------------------------------------------------------------------------------
# T10 -- cluster (room-level) bootstrap
# --------------------------------------------------------------------------------------
def _width(res):
    return res["hi"] - res["lo"]


def test_cluster_bootstrap_with_singleton_clusters_matches_the_pair_bootstrap():
    """One cluster per pair is the pair bootstrap, so the widths must agree closely."""
    e0, ek = _paired(n=1000, ratio=1.15, noise=0.4, seed=3)
    plain = relative_degradation_bootstrap(e0, ek, n_boot=2000, seed=0)
    per_pair = relative_degradation_bootstrap(e0, ek, n_boot=2000, seed=0,
                                              clusters=np.arange(e0.size))
    assert per_pair["unit"] == "cluster"
    assert per_pair["r"] == plain["r"], "the point estimate never depends on the resampling unit"
    assert per_pair["n"] == plain["n"] == 1000
    assert abs(_width(per_pair) - _width(plain)) / _width(plain) < 0.25


def test_cluster_bootstrap_is_wider_when_clusters_disagree():
    """Five rooms with very different degradations: room-level inference must be wider."""
    rng = np.random.default_rng(11)
    n_per, ratios = 400, np.array([0.9, 1.0, 1.1, 1.3, 1.6])
    clusters = np.repeat(np.arange(5), n_per)
    e0 = rng.lognormal(mean=0.0, sigma=0.5, size=5 * n_per)
    ek = e0 * ratios[clusters] * np.exp(rng.normal(0.0, 0.1, size=e0.size))

    plain = relative_degradation_bootstrap(e0, ek, n_boot=2000, seed=0)
    clustered = relative_degradation_bootstrap(e0, ek, n_boot=2000, seed=0, clusters=clusters)
    assert clustered["r"] == plain["r"]
    assert _width(clustered) > _width(plain)
    assert _width(clustered) > 3.0 * _width(plain), "cluster width {} vs pair width {}".format(
        _width(clustered), _width(plain))


def _naive_cluster_interval(e0, ek, clusters, n_boot, seed, alpha=0.05, equal_room=False):
    """Explicit cluster bootstrap: draw clusters, concatenate their rows, take means.

    Written out the slow, obvious way -- no multiplicity weights, no per-cluster sums --
    so it shares no algebra with the implementation under test.  ``equal_room=True``
    switches to the *other* plausible convention (each drawn room contributes its own
    mean, whatever its size), which the implementation must NOT match.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    rows = [np.flatnonzero(clusters == c) for c in unique]
    room_mean0 = np.array([e0[r].mean() for r in rows])
    room_meank = np.array([ek[r].mean() for r in rows])
    ratios = np.empty(n_boot)
    for t in range(n_boot):
        draw = rng.integers(0, len(unique), len(unique))
        if equal_room:
            mean0, meank = room_mean0[draw].mean(), room_meank[draw].mean()
        else:
            idx = np.concatenate([rows[d] for d in draw])
            mean0, meank = e0[idx].mean(), ek[idx].mean()
        ratios[t] = (meank - mean0) / mean0
    return np.percentile(ratios, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def test_cluster_bootstrap_weights_unequal_rooms_like_an_explicit_resampling():
    """Rooms of 10 / 20 / 200 / 50 / 5 queries: the weighted implementation must agree
    with the literal one, and must not silently average rooms instead of queries."""
    rng = np.random.default_rng(21)
    sizes = (10, 20, 200, 50, 5)
    shifts = np.array([1.02, 1.35, 1.10, 0.95, 1.50])
    clusters = np.concatenate([np.full(n, i) for i, n in enumerate(sizes)])
    e0 = rng.lognormal(mean=0.0, sigma=0.5, size=sum(sizes))
    ek = e0 * shifts[clusters] * np.exp(rng.normal(0.0, 0.1, size=e0.size))

    res = relative_degradation_bootstrap(e0, ek, n_boot=20000, seed=0, clusters=clusters)
    for naive_seed in (123, 999):
        lo, hi = _naive_cluster_interval(e0, ek, clusters, n_boot=20000, seed=naive_seed)
        # Five rooms make the resample distribution discrete enough that 20000 draws
        # pin both percentiles to the same atom whatever the seed, so Monte-Carlo noise
        # sits far below this tolerance (verified across four independent seeds).
        assert res["lo"] == pytest.approx(lo, rel=0, abs=1e-9)
        assert res["hi"] == pytest.approx(hi, rel=0, abs=1e-9)

    # The other convention -- one vote per room regardless of size -- is measurably
    # different, so this test really does pin the weighting.
    eq_lo, eq_hi = _naive_cluster_interval(e0, ek, clusters, n_boot=20000, seed=123,
                                           equal_room=True)
    assert abs(res["lo"] - eq_lo) > 5e-3 and abs(res["hi"] - eq_hi) > 5e-3

    pooled = (ek.mean() - e0.mean()) / e0.mean()
    per_room = np.array([(ek[clusters == c].mean() - e0[clusters == c].mean())
                         / e0[clusters == c].mean() for c in range(len(sizes))])
    assert res["r"] == pooled, "queries keep their multiplicity; big rooms weigh more"
    assert abs(res["r"] - per_room.mean()) > 0.02, "this is not an equal-room average"
    assert per_room.min() <= res["lo"] and res["hi"] <= per_room.max()


def test_cluster_bootstrap_is_seed_deterministic_and_validates_its_ids():
    e0, ek = _paired(n=200, ratio=1.2, noise=0.3, seed=4)
    clusters = np.repeat(np.arange(4), 50)
    a = relative_degradation_bootstrap(e0, ek, n_boot=500, seed=1, clusters=clusters)
    b = relative_degradation_bootstrap(e0, ek, n_boot=500, seed=1, clusters=clusters)
    assert a == b
    # Cluster ids may be arbitrary labels; only the grouping matters.
    labels = np.array(["room_{}".format(c) for c in clusters])
    assert relative_degradation_bootstrap(e0, ek, n_boot=500, seed=1, clusters=labels) == a
    with pytest.raises(ValueError):
        relative_degradation_bootstrap(e0, ek, n_boot=500, clusters=clusters[:-1])


# --------------------------------------------------------------------------------------
# T10 -- Bonferroni adjustment, the substantial-degradation verdict, and TOST
# --------------------------------------------------------------------------------------
def test_bonferroni_alpha_divides_by_the_family_size():
    # The pre-registered call: family = 2 metrics x 9 non-zero acoustic angles.
    assert bonferroni_alpha(18) == 0.05 / 18
    assert bonferroni_alpha(18, 0.01) == 0.01 / 18
    assert bonferroni_alpha(1) == 0.05
    for bad_m in (0, -3):
        with pytest.raises(ValueError):
            bonferroni_alpha(bad_m)
    for bad_m in (2.0, True, "18"):
        with pytest.raises(TypeError):
            bonferroni_alpha(bad_m)
    for bad_alpha in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            bonferroni_alpha(18, bad_alpha)


def test_verdict_substantial_needs_a_strict_lower_bound():
    assert verdict_substantial({"lo": 0.1001}, 0.10) is True
    assert verdict_substantial({"lo": 0.10}, 0.10) is False, "the boundary is not a pass"
    assert verdict_substantial({"lo": 0.05}, 0.10) is False
    assert verdict_substantial({"lo": -0.2}, 0.10) is False


def test_equivalence_tost_accepts_a_negligible_change():
    e0 = _paired(n=500)[0]
    res = equivalence_tost(e0, e0.copy(), margin=0.02, n_boot=500, seed=0)
    assert {"equivalent", "lo", "hi", "r"} <= set(res)
    assert res["r"] == 0.0
    assert res["equivalent"] is True

    # A small but genuinely noisy change still fits inside +-2%.
    rng = np.random.default_rng(5)
    base = rng.lognormal(mean=0.0, sigma=0.5, size=5000)
    jittered = base * np.exp(rng.normal(0.0, 0.05, size=base.size))
    noisy = equivalence_tost(base, jittered, margin=0.02, n_boot=2000, seed=0)
    assert noisy["equivalent"] is True
    assert noisy["lo"] < noisy["r"] < noisy["hi"]
    assert -0.02 < noisy["lo"] and noisy["hi"] < 0.02


def test_equivalence_tost_rejects_a_five_percent_change():
    e0, ek = _paired(n=2000, ratio=1.05)
    res = equivalence_tost(e0, ek, margin=0.02, n_boot=2000, seed=0)
    assert res["r"] == pytest.approx(0.05, abs=1e-9)
    assert res["equivalent"] is False
    # The same data is equivalent under a margin that actually covers the effect.
    assert equivalence_tost(e0, ek, margin=0.10, n_boot=2000, seed=0)["equivalent"] is True
    with pytest.raises(ValueError):
        equivalence_tost(e0, ek, margin=0.0, n_boot=2000, seed=0)


def test_equivalence_tost_validates_its_level_and_margin():
    e0, ek = _paired(n=200, ratio=1.01, noise=0.1, seed=14)
    for bad_alpha in (0.5, 0.6, 0.0, 1.0):
        with pytest.raises(ValueError):       # 1 - 2*alpha must be a real interval
            equivalence_tost(e0, ek, margin=0.02, n_boot=100, alpha=bad_alpha)
    for bad_margin in (0.0, -0.02, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            equivalence_tost(e0, ek, margin=bad_margin, n_boot=100)
    for bad_n_boot in (0, -1):
        with pytest.raises(ValueError):
            equivalence_tost(e0, ek, margin=0.02, n_boot=bad_n_boot)
    for bad_n_boot in (100.0, True):
        with pytest.raises(TypeError):
            equivalence_tost(e0, ek, margin=0.02, n_boot=bad_n_boot)


def test_equivalence_tost_supports_the_cluster_unit():
    """Rooms that disagree widen the equivalence interval, so equivalence gets harder."""
    rng = np.random.default_rng(15)
    n_per, ratios = 200, np.array([0.98, 1.0, 1.02, 1.05])
    clusters = np.repeat(np.arange(4), n_per)
    e0 = rng.lognormal(mean=0.0, sigma=0.5, size=4 * n_per)
    ek = e0 * ratios[clusters] * np.exp(rng.normal(0.0, 0.05, size=e0.size))

    # r is about +1.7%; a 3% margin covers the query-level interval but not the
    # room-level one, which is the whole point of reporting both.
    plain = equivalence_tost(e0, ek, margin=0.03, n_boot=2000, seed=0)
    clustered = equivalence_tost(e0, ek, margin=0.03, n_boot=2000, seed=0, clusters=clusters)
    assert clustered["r"] == plain["r"]
    assert clustered["unit"] == "cluster" and plain["unit"] == "pair"
    assert clustered["hi"] - clustered["lo"] > plain["hi"] - plain["lo"]
    assert plain["equivalent"] is True and clustered["equivalent"] is False
    with pytest.raises(ValueError):
        equivalence_tost(e0, ek, margin=0.03, n_boot=100, clusters=clusters[:-1])


def test_bootstrap_rejects_a_non_finite_resample():
    """A resample whose mean(e0) is 0 makes r undefined; that must not be quietly
    dropped from the percentile."""
    e0 = np.array([-1.0, 1.0, 2.0])          # mean 2/3, but (-1, -1, 2) resamples to 0
    ek = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        relative_degradation_bootstrap(e0, ek, n_boot=200, seed=0)
    with pytest.raises(ValueError):
        equivalence_tost(e0, ek, margin=0.5, n_boot=200, seed=0)
    with pytest.raises(ValueError):
        diff_in_diff_bootstrap(e0, ek, e0, ek * 2.0, n_boot=200, seed=0)


def test_bootstrap_rejects_a_non_integer_n_boot():
    e0, ek = _paired(n=50, ratio=1.1, seed=16)
    for bad in (100.0, True, "100"):
        with pytest.raises(TypeError):
            relative_degradation_bootstrap(e0, ek, n_boot=bad)
    with pytest.raises(ValueError):
        relative_degradation_bootstrap(e0, ek, n_boot=-5)


def test_equivalence_tost_uses_the_one_minus_two_alpha_interval():
    e0, ek = _paired(n=2000, ratio=1.2, noise=0.3, seed=6)
    tost = equivalence_tost(e0, ek, margin=1.0, n_boot=2000, alpha=0.05, seed=0)
    two_sided = relative_degradation_bootstrap(e0, ek, n_boot=2000, alpha=0.05, seed=0)
    assert two_sided["lo"] < tost["lo"] and tost["hi"] < two_sided["hi"]
    assert tost["r"] == two_sided["r"]


# --------------------------------------------------------------------------------------
# T10 -- difference in differences (cross-model, on shared resamples)
# --------------------------------------------------------------------------------------
def test_diff_in_diff_of_a_model_with_itself_is_exactly_zero():
    e0, ek = _paired(n=800, ratio=1.3, noise=0.3, seed=7)
    res = diff_in_diff_bootstrap(e0, ek, e0, ek, n_boot=500, seed=0)
    assert {"d", "lo", "hi", "r_a", "r_b", "n", "n_boot", "alpha", "unit"} == set(res)
    assert res["d"] == 0.0
    assert res["lo"] == 0.0 and res["hi"] == 0.0, "shared resamples must cancel exactly"
    assert res["r_a"] == res["r_b"]
    assert res["n"] == 800 and res["unit"] == "pair"


def test_diff_in_diff_uses_the_same_resamples_for_both_models():
    """With an unchanged model B every resample of B is exactly 0, so the DiD interval
    must coincide, bound for bound, with model A's own interval at the same seed."""
    e0_a, ek_a = _paired(n=1000, ratio=1.25, noise=0.3, seed=8)
    e0_b = _paired(n=1000, ratio=1.0, noise=0.2, seed=9)[0]

    did = diff_in_diff_bootstrap(e0_a, ek_a, e0_b, e0_b.copy(), n_boot=1000, seed=3)
    alone = relative_degradation_bootstrap(e0_a, ek_a, n_boot=1000, seed=3)
    assert did["r_b"] == 0.0
    assert did["d"] == alone["r"]
    assert (did["lo"], did["hi"]) == (alone["lo"], alone["hi"])


def test_diff_in_diff_detects_a_model_that_degrades_less():
    e0 = _paired(n=3000, seed=10)[0]
    rng = np.random.default_rng(12)
    ek_a = e0 * 1.30 * np.exp(rng.normal(0.0, 0.2, size=e0.size))     # A degrades by 30%
    ek_b = e0 * 1.05 * np.exp(rng.normal(0.0, 0.2, size=e0.size))     # B by 5%

    res = diff_in_diff_bootstrap(e0, ek_b, e0, ek_a, n_boot=2000, seed=0)
    assert res["r_a"] < res["r_b"]
    assert res["d"] == pytest.approx(res["r_a"] - res["r_b"], abs=1e-12)
    assert res["hi"] < 0.0, "B degrading less than A must give a DiD upper bound below 0"
    assert res["lo"] < res["d"] < res["hi"]


def test_diff_in_diff_supports_clusters_and_validates_its_inputs():
    e0, ek = _paired(n=400, ratio=1.2, noise=0.3, seed=13)
    ek_b = e0 * 1.05
    clusters = np.repeat(np.arange(4), 100)
    res = diff_in_diff_bootstrap(e0, ek, e0, ek_b, n_boot=500, seed=0, clusters=clusters)
    assert res["unit"] == "cluster"
    assert res == diff_in_diff_bootstrap(e0, ek, e0, ek_b, n_boot=500, seed=0, clusters=clusters)

    with pytest.raises(ValueError):                       # A and B are not the same queries
        diff_in_diff_bootstrap(e0, ek, e0[:-1], ek_b[:-1], n_boot=100)
    with pytest.raises(ValueError):                       # non-finite
        diff_in_diff_bootstrap(e0, ek, e0, np.full_like(ek_b, np.nan), n_boot=100)
