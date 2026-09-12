"""Pure-statistics regression tests for the exp_04 confirmatory producer."""
import numpy as np
import pytest

from tools import paired_compare as stats


def seed_matrix(values):
    return np.tile(np.asarray(values, dtype=float), (5, 1))


@pytest.mark.parametrize("samples,alpha,expected", [
    ([4, 1, 3, 2], .25, 3), ([4, 1, 3, 2], .01, 4),
    ([1, 2, 2, 9], .5, 2), ([7], .2, 7),
    (list(range(1, 41)), .025, 39), (list(range(1, 11)), .7, 3)])
def test_one_sided_upper_exact_order_statistic(samples, alpha, expected):
    assert stats.one_sided_upper(samples, alpha) == expected


@pytest.mark.parametrize("samples,alpha", [([], .05), ([np.nan], .05),
    ([[1, 2]], .05), ([1, 2], 0), ([1, 2], 1)])
def test_quantile_rejects_invalid_inputs(samples, alpha):
    with pytest.raises(ValueError):
        stats.one_sided_upper(samples, alpha)


def test_tails_and_families():
    samples = np.arange(1, 20001)
    assert stats.bonferroni_alpha(2) == .025
    assert stats.one_sided_upper(samples, stats.bonferroni_alpha(2)) == 19500
    assert stats.one_sided_upper(samples, stats.bonferroni_alpha(8)) == 19875
    assert stats.two_sided_interval(samples, .05) == (500, 19500)
    assert stats.two_sided_interval(samples, .05 / 2) == (250, 19750)
    assert stats.two_sided_interval(samples, 2 * .05 / 8) == (125, 19875)
    assert stats.two_sided_interval(samples, 2 * .05 / 14) == (72, 19929)


def test_five_seed_mean_is_plain_and_propagates_any_invalid():
    matrix = np.arange(20, dtype=float).reshape(5, 4)
    matrix[0, 1], matrix[3, 2] = np.nan, np.inf
    np.testing.assert_allclose(stats.five_seed_mean(matrix), [8, np.nan, np.nan, 11],
                               equal_nan=True)
    with pytest.raises(ValueError, match="five"):
        stats.five_seed_mean(matrix[:4])


def test_masks_joint_and_per_arm_seed_categories():
    a0, ak, b0, bk = (seed_matrix(np.ones(6)) for _ in range(4))
    a0[0, 0], ak[1, 1], b0[2, 2], bk[3, 3] = np.nan, np.inf, np.nan, np.inf
    ak[4, 0] = np.nan  # Already-invalid baseline: not a newly invalid query.
    h1, _ = stats.cell_mask(a0, e0_b=b0)
    np.testing.assert_array_equal(h1, [False, True, False, True, True, True])
    mask, counts = stats.cell_mask(a0, ak, b0, bk)
    np.testing.assert_array_equal(mask, [False, False, False, False, True, True])
    assert counts["joint"] == {"n": 6, "baseline_invalid": 2, "newly_invalid": 2,
                                 "excluded": 4, "valid": 2}
    assert counts["a"]["total"]["baseline_invalid"] == 1
    assert counts["a"]["total"]["newly_invalid"] == 1
    assert counts["a"]["seeds"][42]["baseline_invalid"] == 1
    assert counts["a"]["seeds"][43]["newly_invalid"] == 1
    # Per-seed counts condition on that seed's baseline; all-seed totals use all seeds.
    assert counts["a"]["seeds"][46]["newly_invalid"] == 1
    tost, _ = stats.cell_mask(a0, ak)
    np.testing.assert_array_equal(tost, [False, False, True, True, True, True])
    with pytest.raises(ValueError, match="empty"):
        stats.cell_mask(seed_matrix([np.nan]), seed_matrix([1]))


def test_h1_known_shift_and_resampling_the_ratio():
    control = stats.five_seed_mean(seed_matrix([1, 2, 4, 9]))
    aug = stats.five_seed_mean(seed_matrix([2, 3, 5, 10]))
    result = stats.rho_bootstrap(aug, control, n_boot=100, seed=7)
    indices = np.random.default_rng(7).integers(0, 4, size=(100, 4))
    expected = (aug[indices].mean(1) - control[indices].mean(1)) / control[indices].mean(1)
    np.testing.assert_array_equal(result["samples"], expected)
    assert result["rho"] == .25
    np.testing.assert_array_equal(result["samples"], stats.rho_bootstrap(
        aug, control, n_boot=100, seed=7)["samples"])


def test_h2_known_shift_and_shared_query_and_room_resamples():
    a0 = stats.five_seed_mean(seed_matrix([1, 2, 4, 9]))
    ak, b0, bk = a0 + .1, 2 * a0, 2 * a0 + .8
    result = stats.dk_bootstrap(a0, ak, b0, bk, n_boot=100, seed=3)
    assert result["d"] == pytest.approx(-.075)
    expected = stats.diff_in_diff_bootstrap(a0, ak, b0, bk, n_boot=100, seed=3)
    assert result["d"] == expected["d"]
    draws = np.random.default_rng(3).integers(0, 4, size=(100, 4))
    ra = (ak[draws].mean(1) - a0[draws].mean(1)) / a0[draws].mean(1)
    rb = (bk[draws].mean(1) - b0[draws].mean(1)) / b0[draws].mean(1)
    np.testing.assert_array_equal(result["samples"], ra - rb)
    same = stats.dk_bootstrap(a0, ak, a0, ak, n_boot=100, clusters=[0, 0, 1, 1])
    assert same["unit"] == "cluster"
    assert not same["samples"].any()
    rooms = np.array([0, 0, 1, 1])
    rho = stats.rho_bootstrap(ak, a0, n_boot=20, seed=4, clusters=rooms)
    room_draws = np.random.default_rng(4).integers(0, 2, size=(20, 2))
    expected = []
    for selected in room_draws:
        idx = np.concatenate([np.flatnonzero(rooms == room) for room in selected])
        expected.append((ak[idx].mean() - a0[idx].mean()) / a0[idx].mean())
    np.testing.assert_allclose(rho["samples"], expected, atol=1e-15)


def test_invariant_tost_and_strict_margin_verdicts():
    baseline = stats.five_seed_mean(seed_matrix([1, 2, 4, 9]))
    result = stats.tost_cell(baseline, baseline, .02, .05 / 14, n_boot=100)
    assert result["equivalent"] and result["lo"] == result["hi"] == 0
    assert stats.tost_verdict(-.02, .01, .02) == "equivalence not established"
    assert stats.tost_verdict(-.01, .02, .02) == "equivalence not established"
    assert stats.tost_verdict(0, 0, .02) == "equivalent"
    upper = stats.one_sided_upper([.03] * 100, .025)
    assert stats.h1_verdict(upper, upper) == "not shown"
    assert stats.h1_verdict(.02, .04) == "non-inferior on EDT only"
    assert stats.h1_verdict(.04, .02) == "non-inferior on C50 only"
    assert stats.h1_verdict(.02, .02) == "non-inferior"
    assert not stats.superiority(0)
    assert stats.superiority(-.001)
    assert stats.h2_verdict([-.01] * 4) == "supported"
    assert stats.h2_verdict([-.01, 0, 0, 0]) == "partially supported"
    assert stats.h2_verdict([0] * 4) == "not supported"


def test_convergence_uses_seed_zero_width_and_exact_boundary():
    assert stats.convergence(lambda seed: (0, 10 + seed))["passed"]
    assert not stats.convergence(lambda seed: (0, 1 + .11 * seed))["passed"]
    assert stats.convergence(lambda seed: {"lo": 0, "hi": 0})["passed"]
    assert not stats.convergence(lambda seed: (seed, seed))["passed"]
    assert not stats.convergence(lambda seed: (0, np.nan))["passed"]
    values = np.arange(1, 11, dtype=float)
    def tiny(seed):
        draws = stats.rho_bootstrap(values + 1, values, n_boot=2, seed=seed)["samples"]
        return stats.two_sided_interval(draws, .05)
    assert not stats.convergence(tiny)["passed"]


def test_invalid_decision_bounds_and_convergence_tolerances_fail_closed():
    assert stats.h1_verdict(-np.inf, np.nan) == "not shown"
    assert stats.h2_verdict([-np.inf, np.nan, np.inf, 0]) == "not supported"
    assert not stats.superiority(-np.inf)
    assert stats.tost_verdict(.01, -.01) == "equivalence not established"
    for tolerance in (-.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="tolerance"):
            stats.convergence(lambda seed: (0, 1), tol=tolerance)
