"""The descriptive exp_07 paired producer: statistics, cohorts and refusals."""
import json

import numpy as np
import pytest

from tools import exp07_pairs as pairs
from tools.exp07_profiles import get_profile, json_value

QUERIES = ['Cat/room_{}/S00{}_R001_hybrid_IR.wav'.format(i // 4, i) for i in range(12)]
SEEDS = (42, 43, 44, 45, 46)


@pytest.fixture
def profile():
    return json_value(get_profile('PAIRS_SEEN_V1'))


def runs(values):
    """Five per-seed runs shaped like a loaded evaluation run."""
    return [dict(query=QUERIES, index=list(range(len(QUERIES))),
                 meta=dict(manifest_seed=seed), P={'0': {'edt': list(row), 'c50': list(row)}})
            for seed, row in zip(SEEDS, values)]


def base(n=12, spread=.01):
    return np.array([[.1 + i / 100 + seed * spread for i in range(n)] for seed in range(5)])


def test_the_absolute_and_relative_statistics_recover_a_known_shift(profile):
    values = base()
    shift = .004
    cell = pairs.paired_cell(profile, runs(values + shift), runs(values),
                             ('seen_cyl', 'seen_simple'), 'EDT', 8)
    absolute, relative = (cell['statistics'][name] for name in ('absolute', 'relative'))
    assert absolute['unit'] == 'ms' and absolute['estimate'] == pytest.approx(shift * 1000)
    assert relative['estimate'] == pytest.approx(shift / values.mean())
    for statistic in (absolute, relative):
        low, high = statistic['interval']
        assert low <= statistic['estimate'] <= high
        assert statistic['room_cluster_interval'][0] <= statistic['estimate']
        assert statistic['estimate'] <= statistic['room_cluster_interval'][1]
        assert statistic['convergence']['passed'] and not statistic['reconverge_required']
    assert cell['quantiles'] == [.025, .975] and cell['decision_driving'] is False
    assert cell['reference'] is False and 'verdict' not in cell
    assert cell['paired_cohort']['n_queries'] == 12 and cell['n_rooms_retained'] == 3
    assert cell['arm_means']['seen_simple']['mean'] == pytest.approx(values.mean() * 1000)
    assert cell['arm_means']['seen_cyl']['per_seed'] == pytest.approx(
        ((values + shift) * 1000).mean(axis=1).tolist())


def test_both_statistics_come_from_one_shared_resample(profile):
    """A constant shift fixes the absolute draws, so the relative draw shows its own mean."""
    values, shift = base(), .004
    means_b = values.mean(axis=0)
    draws = pairs.difference_draws(means_b + shift, means_b, 2000, 0)
    assert np.allclose(draws['absolute'], shift)
    resampled = draws['absolute'] / draws['relative']  # the denominator of the same draw
    assert resampled.min() >= means_b.min() and resampled.max() <= means_b.max()
    assert resampled.std() > 0  # a genuine per-draw mean, not the cohort mean reused
    again = pairs.difference_draws(means_b + shift, means_b, 2000, 0)
    assert np.array_equal(draws['relative'], again['relative'])


def test_the_reference_pairing_is_labelled_and_never_decision_driving(profile):
    values = base()
    cell = pairs.paired_cell(profile, runs(values * 1.3), runs(values),
                             ('released_seen', 'seen_simple'), 'C50', 1)
    assert cell['reference'] is True and cell['decision_driving'] is False
    assert cell['num_shot'] == 1 and cell['metric'] == 'C50' and cell['source'] == 'c50'
    assert cell['statistics']['absolute']['unit'] == 'dB'
    assert cell['statistics']['relative']['unit'] == 'fraction of seen_simple'


def test_a_cell_above_the_convergence_tolerance_is_flagged(profile):
    profile['n_boot'] = 7  # far too few draws for two seeds to agree
    cell = pairs.paired_cell(profile, runs(base() + .004), runs(base()),
                             ('seen_aug', 'seen_simple'), 'EDT', 8)
    assert any(cell['statistics'][name]['reconverge_required']
               for name in ('absolute', 'relative'))
    assert all(cell['statistics'][name]['n_boot'] == 7 for name in ('absolute', 'relative'))


def test_an_empty_paired_cohort_refuses_the_cell(profile):
    values = base()
    holed = values.copy()
    holed[2] = np.nan  # one seed loses every query, so no query is finite in both arms
    with pytest.raises(ValueError, match='empty'):
        pairs.paired_cell(profile, runs(holed), runs(values), ('seen_cyl', 'seen_simple'),
                          'EDT', 8)


def test_queries_missing_in_one_arm_leave_the_cohort_and_are_reported(profile):
    values = base()
    holed = values.copy()
    holed[1, :2] = np.nan
    cell = pairs.paired_cell(profile, runs(holed), runs(values), ('seen_cyl', 'seen_simple'),
                             'EDT', 8)
    assert cell['paired_cohort']['n_queries'] == 10
    assert cell['paired_cohort']['excluded'] == QUERIES[:2]
    assert cell['exclusions']['seen_cyl']['total']['excluded'] == 2
    assert cell['exclusions']['seen_simple']['total']['excluded'] == 0
    assert cell['exclusions']['joint']['valid'] == 10
    assert cell['exclusions']['seen_cyl']['seeds']['43']['excluded'] == 2


def test_the_profile_quantiles_must_match_its_interval_alpha(profile):
    profile['quantiles'] = [.05, .95]
    with pytest.raises(ValueError, match='quantiles'):
        pairs.paired_cell(profile, runs(base()), runs(base()), ('seen_cyl', 'seen_simple'),
                          'EDT', 8)


def test_cells_are_deterministic(profile):
    arguments = (profile, runs(base() + .004), runs(base()), ('seen_cyl', 'seen_simple'), 'EDT', 8)
    assert json.dumps(pairs.paired_cell(*arguments), sort_keys=True) == json.dumps(
        pairs.paired_cell(*arguments), sort_keys=True)
