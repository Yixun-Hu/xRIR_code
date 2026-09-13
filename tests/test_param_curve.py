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
