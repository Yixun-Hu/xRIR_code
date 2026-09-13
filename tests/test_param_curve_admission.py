import pytest
from exp05_fixture import exp05_fixture
from tools import param_curve as pc
from pathlib import Path


@pytest.mark.parametrize('name,legacy', [('CURVE_K8', False), ('CURVE_K1', True), ('YAW_K8_SEED42', False)])
def test_shared_admission_fixture_contract(exp05_fixture, name, legacy):
    fixture = exp05_fixture(name, legacy)
    groups = [(a, fixture.profile['num_shot'], fixture.paths[a['role']]) for a in fixture.profile['arms']]
    admitted = pc.admit_runs(fixture.profile, groups, fixture.approved, exploratory=True,
                            producer=fixture.producer, producer_key='producer_param_curve')
    assert [len(g) for g in admitted['groups']] == [len(fixture.profile['eval_seeds'])]*6
    assert all(d.endswith((': mutable_inputs names', ': evaluator approved closure', ' seed set'))
               for d in admitted['deviations'])


@pytest.mark.parametrize('legacy', [False, True])
def test_tier_contract_exp05_and_approved_legacy_m(exp05_fixture, legacy):
    f = exp05_fixture(m_exp04=legacy)
    for arm in f.profile['arms']:
        directory = f.paths[arm['role']][0]
        contract = pc.run_contract(directory, arm, f.profile, f.pins)
        expected = 'tools.exp04_eval' if legacy and arm['tier'] == 'M' else 'tools.exp05_eval'
        assert contract['evaluator'] == expected
        assert len(contract['waivers']) <= 1


@pytest.mark.parametrize('kind', ['tier', 'count', 'legacy', 'config_float', 'args_binding',
                                 'args_bytes', 'extra_binding', 'decomposition', 'unapproved_M'])
def test_extension_refusals(exp05_fixture, kind):
    f = exp05_fixture(m_exp04=kind == 'unapproved_M')
    arm = f.profile['arms'][2 if kind == 'unapproved_M' else 0]
    directory = Path(f.paths[arm['role']][0])
    fields = f.read(directory / 'eval_manifest.json')
    if kind in ('tier', 'legacy', 'config_float', 'count'):
        key, value = {'tier': ('tier', 'L'), 'legacy': ('legacy_M', True),
            'config_float': ('vit_dim', 256.0), 'count': ('param_counts', {})}[kind]
        fields[key] = value
    elif kind == 'args_binding':
        fields['mutable_inputs']['train_args']['path'] = str(directory / 'args.json')
    elif kind == 'args_bytes':
        Path(fields['mutable_inputs']['train_args']['path']).write_text('{}')
    elif kind == 'extra_binding':
        fields['mutable_inputs']['extra'] = fields['mutable_inputs']['train_args']
    elif kind == 'decomposition':
        fields['decomposition_batches'] = True
    else:
        f.pins['closures']['evaluator_exp04'] = None
    f.replace(directory / 'eval_manifest.json', fields)
    with pytest.raises((ValueError, OSError), match='tier|args|binding|decomposition|evaluator'):
        pc.run_contract(directory, arm, f.profile, f.pins)
