"""Frozen training-only heading rule, records, and heading-frame dataset."""
import numpy as np
import pytest

from tools.exp06_heading import (canonical_heading_deg, compensated_levels, continuous_fit,
                                early_level, heading_roll_k, mean_direction)


@pytest.mark.parametrize('window,n', [(0.005, 110), (0.05, 1102)])
def test_early_energy_strict_onset_and_floor(window, n):
    rir = np.zeros(1200)
    rir[1], rir[2], rir[2 + n - 1], rir[2 + n] = 0.2, 1, 0.1, 0.9
    assert early_level(rir, 22050, window) == pytest.approx(1.01)
    assert early_level([0, 0, 1], 22050, window) == 1  # zero-extended tail
    assert early_level(np.zeros(1200), 22050, window) == 0


def test_distance_compensation():
    rirs = np.array([[0, 2, 0], [0, 1, 0]])
    np.testing.assert_allclose(compensated_levels(rirs, [1, 2], 1000, .005), 10 * np.log10(4))
    with pytest.raises(ValueError):
        compensated_levels(rirs, [0, 2], 1000, .005)
    with pytest.raises(ValueError):
        compensated_levels(np.zeros((2, 5)), [1, 2], 1000, .005)


@pytest.mark.parametrize('phi,k', [(-90, 128), (0, 0), (180, 256), (90, 384), (270, 128),
                                 (.35, 0), (.36, 511), (.3515625, 0),
                                 (.3515625 + .703125, 511), (-.3515625, 1)])
def test_roll_ties_and_wrap(phi, k):
    assert heading_roll_k(phi) == k
    assert canonical_heading_deg(k, 512) == -k * 360 / 512
    assert heading_roll_k(canonical_heading_deg(k, 512)) == k
    assert (heading_roll_k(phi + .703125) + 1) % 512 == k


def test_roll_general_width_and_invalid_inputs():
    assert heading_roll_k(-90, W=16) == 4
    for width in [0, -1, 2.5, True]:
        with pytest.raises(ValueError):
            heading_roll_k(0, width)
    for k in [-1, 512, .5, True]:
        with pytest.raises(ValueError):
            canonical_heading_deg(k, 512)
    with pytest.raises(ValueError):
        heading_roll_k(float('nan'))


def test_descriptive_fit_and_mean():
    theta = np.arange(-180, 180, 30)
    level = 2 + 7 * np.cos(np.deg2rad(theta + 90))
    fit = continuous_fit(theta, level)
    assert fit['phi_deg'] == -90
    assert fit['a'] == pytest.approx(2) and fit['b'] == pytest.approx(7)
    assert fit['mse'] < 1e-20
    assert continuous_fit(theta, np.ones(12))['b'] == pytest.approx(0)
    assert mean_direction([-90, 90], [2, 1]) == pytest.approx(-90)
    assert mean_direction([-90, 90], [1, 1]) is None
    with pytest.raises(ValueError):
        mean_direction([0, 90], [1, -1])


@pytest.mark.parametrize('rir,sr,w', [([], 1000, .005), ([np.nan], 1000, .005),
                                    ([1], 0, .005), ([1], 1000, .0001)])
def test_invalid_energy_inputs(rir, sr, w):
    with pytest.raises(ValueError):
        early_level(rir, sr, w)
