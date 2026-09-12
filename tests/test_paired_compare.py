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


