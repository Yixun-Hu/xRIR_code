"""Frozen training-only heading rule, records, and heading-frame dataset."""
import numpy as np
import pytest

from tools.exp06_heading import (canonical_heading_deg, compensated_levels, continuous_fit,
                                early_level, heading_roll_k, mean_direction)
from tools.exp06_heading import candidate_contrasts, decide_heading, leave_one_out_stable


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


def windows(level):
    return {'5ms': np.asarray(level), '50ms': np.asarray(level)}


@pytest.mark.parametrize('phi,axis', [(0, '+x'), (90, '+y'), (180, '-x'), (-90, '-y')])
def test_uniform_cardioid_axes(phi, axis):
    theta = np.arange(-180, 180, 15)
    amplitude = 1 + .8 * np.cos(np.deg2rad(theta - phi))
    level = 20 * np.log10(amplitude)
    assert decide_heading(theta, windows(level))[0] == axis
    assert leave_one_out_stable(theta, windows(level))


def test_hallway_axis_despite_biased_weighted_mean_and_exact_threshold():
    theta = np.array([-90] * 4 + [90] * 8)
    # These two sampled gains come from a cardioid with front/back ratio 10**(3/20).
    level = np.array([3.] * 4 + [0.] * 8)
    assert mean_direction(theta, 10 ** (level / 10)) == pytest.approx(90)
    winner, table, reason = decide_heading(theta, windows(level))
    assert winner == '-y' and reason == 'accepted'
    assert table['5ms']['runner_up_margin_db'] == 6
    assert leave_one_out_stable(theta, windows(level))


def test_singleton_and_inclusive_wrapped_sectors():
    theta = [-90] * 6 + [0, 0, 90, 90, 180, 180]
    winner, table, _ = decide_heading(theta, windows([9] * 6 + [0] * 6))
    assert winner == '-y'
    assert table['5ms']['competitor_count'] == 0
    assert table['5ms']['runner_up_margin_db'] is None
    contrast = candidate_contrasts([135, 180, -135, 0, 10, -10], [3, 3, 3, 0, 0, 0])
    assert contrast['-x'] == 3


@pytest.mark.parametrize('kind,reason', [('weak', 'contrast'), ('disagree', 'disagree'),
                                      ('margin', 'margin'), ('few', 'microphones')])
def test_decision_refusals(kind, reason):
    theta = np.repeat([0, 90, 180, -90], 3)
    level = np.repeat([0., 0., 0., 8.], 3)
    values = windows(level)
    if kind == 'weak':
        values = windows(level * .2)
    elif kind == 'disagree':
        values['50ms'] = -level
    elif kind == 'margin':
        values = windows(np.repeat([7., 0., 0., 8.], 3))
    else:
        theta, values = [0, 0, 90, 90], windows([0, 0, 8, 8])
    winner, _, why = decide_heading(theta, values)
    assert winner is None and reason in why


def test_loo_reapplies_counts_contrast_margin_and_winner():
    theta = np.repeat([0, 90, 180, -90], 3)
    level = np.array([6] * 3 + [0] * 6 + [30, 0, 0])
    assert decide_heading(theta, windows(level))[0] == '-y'
    assert decide_heading(np.delete(theta, 9), windows(np.delete(level, 9)))[0] == '+x'
    assert not leave_one_out_stable(theta, windows(level))
    theta = [-90] * 4 + [90] * 8
    assert not leave_one_out_stable(theta, windows([12, 0, 0, 0] + [0] * 8))
    theta = np.repeat([0, 90, 180, -90], 4)
    level = [4] * 4 + [0] * 8 + [14, 6, 6, 6]
    assert decide_heading(theta, windows(level))[0] == '-y'
    assert not leave_one_out_stable(theta, windows(level))  # margin collapses below 3
    with pytest.raises(ValueError):
        decide_heading(theta, {'5ms': level})
