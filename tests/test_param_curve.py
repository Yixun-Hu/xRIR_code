import copy
import numpy as np
import pytest
from tools import param_curve as pc
from tools.exp05_profiles import get_profile, json_value


def synthetic(name='CURVE_K8'):
    profile = json_value(get_profile(name))
    queries = ['Room/r{}/q{}'.format(i // 4, i) for i in range(12)]
    groups = []
    for arm in profile['arms']:
        offset = {'S': 4, 'M': 3, 'L': 2}[arm['tier']]
        offset -= 1 if arm['backbone'] == 'cylindrical' else 0
        groups.append([dict(query=queries, index=list(range(12)), meta={'manifest_seed': s},
            P={str(k): {m: (np.arange(12) / 16 + offset + (s-42)/100).tolist()
                for m in ('edt', 'c50', 't60')} for k in profile['grid']}) for s in profile['eval_seeds']])
    return profile, groups


def test_paired_target_counterexample_and_own_cohort():
    profile, groups = synthetic('TARGETS_K8')
    for run in groups[1]:
        run['P']['0']['edt'] = [None, None, 2.4, 2.6] + [None]*8
    for run in groups[2]:
        run['P']['0']['edt'] = [1, 1, 3, 3] + [None]*8
    cell = pc.comparison(profile, groups, ('S_cyl', 'M_simple'), 'EDT')
    assert cell['target'] == 3 and cell['baseline_own']['mean'] == 2
    assert cell['paired_cohort']['n_queries'] == 2
    assert cell['baseline_own']['cohort']['n_queries'] == 4
    assert set(cell['exclusions']) == {'S_cyl', 'M_simple', 'joint'}
    assert cell['paired_cohort']['excluded'] == groups[1][0]['query'][:2] + groups[1][0]['query'][4:]
    assert cell['estimate'] == pytest.approx(-1/6)
    assert cell['reaches_target'] and cell['convergence']['superiority']['passed']


def test_curve_seed_sd_uses_arm_cohort_and_intervals_are_deterministic():
    profile, groups = synthetic()
    groups[0][0]['P']['0']['edt'][0] = None
    point = pc.curve_point(profile, groups[0], profile['arms'][0], 'EDT', 0)
    expected = np.array([r['P']['0']['edt'][1:] for r in groups[0]]).mean(axis=1)
    assert point['mean'] == pytest.approx(expected.mean())
    assert point['sd'] == pytest.approx(expected.std(ddof=1))
    assert point['cohort']['n_queries'] == 11
    queries = groups[0][0]['query']
    assert point['cohort'] == dict(n_queries=11, n_rooms=3,
        retained_sha256=pc._digest(sorted(queries[1:])), excluded=queries[:1])
    reversed_mask = np.array([True]*11 + [False])
    assert pc.cohort(queries[::-1], reversed_mask)['retained_sha256'] == point['cohort']['retained_sha256']
    assert point['excluded']['42'] == [groups[0][0]['query'][0]]
    assert point == pc.curve_point(profile, groups[0], profile['arms'][0], 'EDT', 0)
    assert point['encoder'] == 2766080 and point['full'] == 15073213


def test_known_equivalence_and_tiny_bootstrap_failure():
    profile, groups = synthetic('TARGETS_K8')
    cell = pc.comparison(profile, groups, ('S_cyl', 'M_simple'), 'EDT')
    assert cell['equivalent'] and cell['reaches_target'] and not cell['superior']
    assert cell['tost_interval'] == [0, 0]
    profile['n_boot'] = 2
    cell = pc.comparison(profile, groups, ('M_cyl', 'L_simple'), 'EDT')
    assert cell['convergence']['superiority']['passed']
    groups[3][0]['P']['0']['edt'][0] += 1
    cell = pc.comparison(profile, groups, ('M_cyl', 'L_simple'), 'EDT')
    assert not all(gate['passed'] for gate in cell['convergence'].values())


def test_empty_cohorts_refused():
    profile, groups = synthetic()
    groups[0][0]['P']['0']['edt'] = [None]*12
    with pytest.raises(ValueError, match='empty'):
        pc.comparison(profile, groups, ('S_cyl', 'S_simple'), 'EDT')
    with pytest.raises(ValueError, match='empty'):
        pc.curve_point(profile, groups[0], profile['arms'][0], 'EDT', 0)


@pytest.mark.parametrize('name', ['CURVE_K8', 'CURVE_K1', 'TARGETS_K8', 'TARGETS_K1', 'YAW_K8_SEED42'])
def test_six_arm_analysis_and_parameter_ratio_rule(name):
    profile, groups = synthetic(name)
    admitted = dict(groups=groups, inputs={}, deviations=[], run_flags={})
    result = pc.analyze(profile, admitted)
    assert len(result['curves']) == 18 * len(profile['grid'])
    if profile['mode'] == 'curve':
        assert len(result['cells']) == 6
        assert result['verdicts']['EDT'] == 'dominance supported on EDT at the three tested tiers'
        groups[1] = copy.deepcopy(groups[0])
        assert pc.analyze(profile, admitted)['verdicts']['EDT'] == 'partial'
        for i in (1, 3, 5):
            groups[i] = copy.deepcopy(groups[i-1])
        assert pc.analyze(profile, admitted)['verdicts']['EDT'] == 'not supported'
    elif profile['mode'] == 'targets':
        assert len(result['cells']) == 4 and len(result['parameter_ratios']) == 2
        assert result['parameter_ratios'][0]['ratio'] == 2777984 / 19703296
        for run in groups[1]:
            run['P']['0']['c50'] = [100.0]*12
        assert len(pc.analyze(profile, admitted)['parameter_ratios']) == 1
    else:
        assert 'verdict' not in result and 'verdicts' not in result
        assert all(point['sd'] is None for point in result['curves'])


def test_analysis_refusals_and_exploratory_has_no_decisions():
    profile, groups = synthetic('TARGETS_K8')
    profile['n_boot'] = 2
    groups[3][0]['P']['0']['edt'][0] += 1
    admitted = dict(groups=groups, inputs={}, deviations=[], run_flags={})
    with pytest.raises(ValueError, match='convergence'):
        pc.analyze(profile, admitted)
    result = pc.analyze(profile, admitted, exploratory=True)
    assert any('convergence' in d for d in result['deviations'])
    assert not set(result) & {'verdict', 'verdicts', 'parameter_ratios'}
    assert all(not set(c) & {'superior', 'equivalent', 'reaches_target'} for c in result['cells'])
    groups[0].pop()
    with pytest.raises(ValueError, match='pairing'):
        pc.analyze(profile, admitted)
    assert pc.analyze(profile, admitted, exploratory=True)['cells'] == []


def test_yaw_uses_single_seed_paired_angle_cohort_without_verdicts():
    profile, groups = synthetic('YAW_K8_SEED42')
    groups[0][0]['P']['32']['edt'] = [None] + [5.0]*11
    result = pc.analyze(profile, dict(groups=groups, inputs={}, deviations=[], run_flags={}))
    cell = next(c for c in result['cells'] if c['arm'] == 'S_simple' and c['metric'] == 'EDT' and c['k'] == 32)
    assert cell['paired_cohort']['n_queries'] == 11
    base = np.mean(groups[0][0]['P']['0']['edt'][1:])
    assert cell['estimate'] == pytest.approx((5-base)/base)
    assert len(result['cells']) == 6*3*4
    assert all(c['descriptive'] is True and c['verdict_scope'] == profile['verdict_scope']
               for c in result['cells'])
    assert result['verdict_scope'] == profile['verdict_scope'] and 'descriptive' in result['verdict_scope']
    assert not set(cell) & {'superior', 'equivalent', 'reaches_target', 'verdict'}


@pytest.mark.parametrize('lo,hi', [(-.03, .03), (-.03, .0299), (-.0299, .03), (-.0299, .0299)])
def test_tost_sample_order_statistics_at_strict_boundaries(monkeypatch, lo, hi):
    profile, groups = synthetic('TARGETS_K8')
    samples = np.r_[np.full(250, lo), np.zeros(19499), np.full(251, hi)]
    monkeypatch.setattr(pc, 'rho_bootstrap', lambda *a, **kw: dict(rho=0., samples=samples))
    cell = pc.comparison(profile, groups, ('S_cyl', 'M_simple'), 'EDT')
    assert cell['tost_interval'] == [lo, hi]  # Exact ranks 250 and 19750 of 20000.
    assert cell['equivalent'] is (-.03 < lo and hi < .03)


@pytest.mark.parametrize('name', ['CURVE_K8', 'TARGETS_K8'])
@pytest.mark.parametrize('upper', [0., -1e-12])
def test_superiority_sample_order_statistics_at_zero(monkeypatch, name, upper):
    profile, groups = synthetic(name)
    samples = np.r_[np.full(19874, -.1), np.full(126, upper)]
    monkeypatch.setattr(pc, 'rho_bootstrap', lambda *a, **kw: dict(rho=-.1, samples=samples))
    cell = pc.comparison(profile, groups, profile['pairings'][0], 'EDT')
    assert cell['companion_interval'][1] == upper
    assert cell['superior'] is (upper < 0)


def test_dominance_requires_every_registered_pairing():
    profile, groups = synthetic()
    profile['pairings'] = profile['pairings'][:2]
    result = pc.analyze(profile, dict(groups=groups, inputs={}, deviations=[], run_flags={}))
    assert len(result['cells']) == 4
    assert result['verdicts']['EDT'].startswith('dominance supported')
