import pytest
from exp05_fixture import exp05_fixture
from tools import param_curve as pc


@pytest.mark.parametrize('name,legacy', [('CURVE_K8', False), ('CURVE_K1', True), ('YAW_K8_SEED42', False)])
def test_shared_admission_fixture_contract(exp05_fixture, name, legacy):
    fixture = exp05_fixture(name, legacy)
    groups = [(a, fixture.profile['num_shot'], fixture.paths[a['role']]) for a in fixture.profile['arms']]
    admitted = pc.admit_runs(fixture.profile, groups, fixture.approved, exploratory=True,
                            producer=fixture.producer, producer_key='producer_param_curve')
    assert [len(g) for g in admitted['groups']] == [len(fixture.profile['eval_seeds'])]*6
    assert all(d.endswith((': mutable_inputs names', ': evaluator approved closure', ' seed set'))
               for d in admitted['deviations'])
