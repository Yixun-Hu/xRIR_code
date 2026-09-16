"""The descriptive exp_07 paired producer: statistics, cohorts and refusals."""
import json
import pathlib

import numpy as np
import pytest

from exp07_fixture import exp07_fixture  # noqa: F401  (fixture)
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


@pytest.fixture
def built(exp07_fixture):
    return exp07_fixture('PAIRS_SEEN_V1')


def sides(built, role_a, role_b='seen_simple', shots=(8, 1)):
    return ([path for shot in shots for path in built.paths[(role_a, shot)]],
            [path for shot in shots for path in built.paths[(role_b, shot)]])


def build(built, role_a='seen_cyl', role_b='seen_simple', shots=(8, 1), **kwargs):
    runs_a, runs_b = sides(built, role_a, role_b, shots)
    return pairs.build_pairs(runs_a, runs_b, built.profile, built.approved,
                             producer=built.producer, **kwargs)


@pytest.mark.parametrize('role_a', ['seen_cyl', 'seen_aug', 'released_seen'])
def test_every_registered_pairing_is_computed_at_both_shot_counts(built, role_a):
    result, _ = build(built, role_a)
    assert result['pairing'] == [role_a, 'seen_simple'] and result['deviations'] == []
    assert [(cell['metric'], cell['num_shot']) for cell in result['cells']] == [
        (metric, shot) for shot in (8, 1)
        for metric in ('EDT', 'C50', 'T60', 'loss', 'log_mse')]
    assert result['final'] is True and result['reconverge_required'] == []
    assert 'verdict' not in result and result['decision_driving'] is False
    assert all(cell['reference'] is (role_a == 'released_seen') for cell in result['cells'])
    assert result['verdict_scope'] == built.profile['verdict_scope']
    assert result['profile_name'] == 'PAIRS_SEEN_V1'


def test_the_cells_use_the_five_seed_means_of_the_admitted_runs(built):
    result, _ = build(built, 'seen_cyl', shots=(8,))
    cell = next(c for c in result['cells'] if c['metric'] == 'C50' and c['num_shot'] == 8)
    values = {}
    for role in ('seen_cyl', 'seen_simple'):
        values[role] = np.array([built.read(pathlib.Path(directory) / 'per_sample_yaw.json')
                                 ['P']['0']['c50'] for directory in built.paths[(role, 8)]])
    difference = values['seen_cyl'].mean() - values['seen_simple'].mean()
    assert cell['statistics']['absolute']['estimate'] == pytest.approx(difference)
    assert cell['statistics']['relative']['estimate'] == pytest.approx(
        difference / values['seen_simple'].mean())
    assert cell['arm_means']['seen_cyl']['mean'] == pytest.approx(values['seen_cyl'].mean())


@pytest.mark.parametrize('role_a,role_b,message', [
    ('seen_simple', 'seen_simple', 'unregistered pairing'),
    ('seen_cyl', 'seen_aug', 'must be the seen_simple baseline')])
def test_unregistered_sides_are_refused(built, role_a, role_b, message):
    with pytest.raises(ValueError, match=message):
        build(built, role_a, role_b)


def test_the_two_sides_must_hold_one_arm_each_at_the_same_shot_counts(built):
    runs_a, runs_b = sides(built, 'seen_cyl')
    with pytest.raises(ValueError, match='--runs-a must hold exactly one arm'):
        pairs.build_pairs(runs_a + runs_b, runs_b, built.profile, built.approved,
                          producer=built.producer)
    with pytest.raises(ValueError, match='same registered shot counts'):
        pairs.build_pairs(sides(built, 'seen_cyl', shots=(8,))[0], runs_b, built.profile,
                          built.approved, producer=built.producer)


def test_an_unconverged_cell_marks_the_result_not_final(built):
    built.profile['n_boot'] = 7
    result, _ = build(built, 'seen_aug', shots=(1,))
    assert result['final'] is False
    assert any(item.startswith('EDT K=1') for item in result['reconverge_required'])


def test_the_registered_retry_reruns_only_the_flagged_statistics(built):
    """The re-run keeps the estimate and the cohort, replaces only the flagged intervals."""
    built.profile.update(n_boot=7, reconverge_n_boot=4000)
    first, _ = build(built, 'seen_aug', shots=(1,))
    retried, _ = build(built, 'seen_aug', shots=(1,), reconverge=True)
    assert first['final'] is False and first['reconverge_attempted'] is False
    assert first['reconverge_n_boot'] is None
    assert retried['reconverge_attempted'] is True and retried['reconverge_n_boot'] == 4000
    assert retried['final'] is True and retried['reconverge_required'] == []
    for before, after in zip(first['cells'], retried['cells']):
        assert (after['metric'], after['num_shot']) == (before['metric'], before['num_shot'])
        assert after['paired_cohort'] == before['paired_cohort']
        for name, statistic in sorted(after['statistics'].items()):
            estimate = before['statistics'][name]['estimate']
            assert statistic['estimate'] == estimate  # the point estimate never resamples
            if before['statistics'][name]['reconverge_required']:
                assert statistic['n_boot'] == 4000 and statistic['reconverge_attempts'] == 1
                assert statistic['reconverge_required'] is False
                assert statistic['first_attempt']['n_boot'] == 7
                assert statistic['first_attempt']['convergence']['passed'] is False
                assert (statistic['first_attempt']['interval'] ==
                        before['statistics'][name]['interval'])
            else:  # an already-converged statistic is left exactly as it was
                assert statistic == before['statistics'][name]


def test_a_retry_that_fails_again_stays_flagged_and_not_final(built):
    built.profile.update(n_boot=5, reconverge_n_boot=6)
    result, _ = build(built, 'seen_aug', shots=(1,), reconverge=True)
    assert result['reconverge_attempted'] is True and result['final'] is False
    assert result['reconverge_required']
    flagged = [cell for cell in result['cells'] if pairs.flagged_statistics(cell)]
    assert flagged
    for cell in flagged:
        for name in pairs.flagged_statistics(cell):
            statistic = cell['statistics'][name]
            assert statistic['n_boot'] == 6 and statistic['reconverge_attempts'] == 1
            assert statistic['first_attempt']['n_boot'] == 5
    assert 'Reconvergence re-run at n_boot 6' in pairs.render_summary(result)


def test_a_retry_count_that_is_not_larger_is_refused(built):
    built.profile.update(n_boot=7, reconverge_n_boot=7)
    with pytest.raises(ValueError, match='reconverge_n_boot must exceed'):
        build(built, 'seen_aug', shots=(1,), reconverge=True)


def test_the_registered_retry_count_is_the_plans_forty_thousand(profile):
    assert profile['reconverge_n_boot'] == 40000 > profile['n_boot'] == 20000


def test_a_converged_pairing_is_untouched_by_the_retry_flag(built):
    plain, _ = build(built, 'seen_cyl', shots=(8,))
    retried, _ = build(built, 'seen_cyl', shots=(8,), reconverge=True)
    assert plain['cells'] == retried['cells'] and retried['final'] is True
    assert retried['reconverge_attempted'] is True


def test_an_empty_cohort_refuses_but_is_listed_in_exploratory_mode(built):
    for directory in built.paths[('seen_cyl', 8)]:
        sample = built.read(pathlib.Path(directory) / 'per_sample_yaw.json')
        sample['P']['0']['edt'] = [None] * len(built.queries)
        metrics = built.read(pathlib.Path(directory) / 'metrics_yaw.json')
        metrics['P'] = built.summaries(sample['P'])
        built.rebind(pathlib.Path(directory), sample=sample, metrics=metrics)
    with pytest.raises(ValueError, match='EDT K=8'):
        build(built, 'seen_cyl', shots=(8,))
    result, _ = build(built, 'seen_cyl', shots=(8,), exploratory=True)
    assert any('EDT K=8' in item for item in result['deviations']) and result['final'] is False
    assert [cell['metric'] for cell in result['cells']] == ['C50', 'T60', 'loss', 'log_mse']


def publish(built, tmp_path, name, **kwargs):
    result, admitted = build(built, **kwargs)
    target = tmp_path / name
    target.mkdir()
    pairs.write_outputs(result, admitted, str(target / 'pairs.json'),
                        str(target / 'summary.txt'), pairs.render_summary)
    return target, result


def test_publication_writes_the_json_summary_and_sidecar(built, tmp_path):
    target, result = publish(built, tmp_path, 'published', shots=(8,))
    published = json.loads((target / 'pairs.json').read_text())
    assert published['pairing'] == ['seen_cyl', 'seen_simple'] and published['final'] is True
    summary = (target / 'summary.txt').read_text()
    assert 'No verdict' in summary and 'DESCRIPTIVE PAIRS_SEEN_V1' in summary
    assert published['profile_digest'] in summary and 'NOT FINAL' not in summary
    for cell in published['cells']:
        absolute = cell['statistics']['absolute']
        assert '{} K={}'.format(cell['metric'], cell['num_shot']) in summary
        assert '{:.8g} {}'.format(absolute['estimate'], absolute['unit']) in summary
    sidecar = json.loads((target / 'pairs.json.provenance.json').read_text())
    assert sidecar['profile_digest'] == published['profile_digest']
    assert sidecar['approved_digests']['sha256'] == built.approved[1]['sha256']
    assert sidecar['outputs'][str((target / 'summary.txt').resolve())]
    other, _ = publish(built, tmp_path, 'again', shots=(8,))
    assert (other / 'pairs.json').read_bytes() == (target / 'pairs.json').read_bytes()


def test_the_cli_refuses_runs_outside_the_registered_checkpoints(built, tmp_path):
    """main() uses the committed profile, whose arms are ckpt/exp07's own checkpoints."""
    runs_a, runs_b = sides(built, 'seen_aug')
    with pytest.raises(ValueError, match='unregistered checkpoint'):
        pairs.main(['--profile', 'PAIRS_SEEN_V1', '--runs-a'] + runs_a + ['--runs-b'] + runs_b +
                   ['--json', str(tmp_path / 'p.json'), '--summary', str(tmp_path / 's.txt')])
    assert not (tmp_path / 'p.json').exists() and not (tmp_path / 's.txt').exists()


@pytest.mark.parametrize('retry', [False, True])
def test_the_cli_passes_both_sides_and_the_renderer_to_the_shared_writer(monkeypatch, tmp_path,
                                                                        retry):
    captured = {}
    def fake_build(runs_a, runs_b, exploratory, reconverge):
        captured.update(a=runs_a, b=runs_b, exploratory=exploratory, reconverge=reconverge)
        return {'ok': True}, {}
    monkeypatch.setattr(pairs, 'build_pairs', fake_build)
    monkeypatch.setattr(pairs, 'write_outputs', lambda *call: captured.update(call=call))
    pairs.main(['--profile', 'PAIRS_SEEN_V1', '--runs-a', 'x', '--runs-b', 'y', '--json',
                str(tmp_path / 'p.json'), '--summary', str(tmp_path / 's.txt'), '--exploratory'] +
               (['--reconverge'] if retry else []))
    assert captured['a'] == ['x'] and captured['b'] == ['y'] and captured['exploratory'] is True
    assert captured['reconverge'] is retry
    assert captured['call'][2:] == (str(tmp_path / 'p.json'), str(tmp_path / 's.txt'),
                                    pairs.render_summary)
