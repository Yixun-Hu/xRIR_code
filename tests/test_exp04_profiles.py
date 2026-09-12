import hashlib
import importlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from tools import exp04_profiles as profiles

ROOT = Path(__file__).resolve().parents[1]


def test_profiles_frozen_and_typed():
    assert set(profiles.PROFILES) == {'H1_K8', 'H1_K1', 'H2_K8', 'TOST_K8', 'TABLE_V1'}
    def check(value):
        assert type(value) in (MappingProxyType, tuple, str, int, float, bool, type(None))
        if isinstance(value, MappingProxyType):
            with pytest.raises(TypeError):
                value['changed'] = True
            for item in value.values():
                check(item)
        elif isinstance(value, tuple):
            with pytest.raises(TypeError):
                value[0] = None
            for item in value:
                check(item)
    check(profiles.PROFILES)
    for name, profile in profiles.PROFILES.items():
        assert profiles.get_profile(name) is profile
        assert type(profile['schema_version']) is int
        assert profile['mode'] in ('two_arm', 'one_arm', 'table')
        assert profile['condition'] == 'P'
        assert profile['alpha'] == .05 and profile['n_boot'] == 20000
        assert profile['bootstrap_seeds'] == (0, 1)
        assert profile['convergence_tolerance'] == .10
        assert profile['batch_size'] == 16 and profile['max_samples'] == 0
        assert profile['tf32'] is False
        assert set(profile['approved_closures']) == {'evaluator', 'writer', 'launcher', 'producer'}
        assert all(v is None for v in profile['approved_closures'].values())
    with pytest.raises(KeyError):
        profiles.get_profile('UNKNOWN')


def test_canonical_digest_stable_across_imports():
    before = {name: profiles.profile_digest(name) for name in profiles.PROFILES}
    importlib.reload(profiles)
    for name, digest in before.items():
        raw = json.dumps(profiles.json_value(profiles.get_profile(name)),
                         sort_keys=True, separators=(',', ':'), allow_nan=False)
        assert profiles.profile_digest(name) == hashlib.sha256(raw.encode()).hexdigest() == digest


def test_hypotheses_and_canonical_input_selection():
    for name in ('H1_K8', 'H1_K1'):
        p = profiles.get_profile(name)
        assert p['grid'] == (0,) and p['input_selection'] == 'standalone_k0'
        assert p['family'] == 2 and p['margin'] == .03
        assert p['metrics']['primary'] == ('EDT', 'C50')
    h2, tost = (profiles.get_profile(name) for name in ('H2_K8', 'TOST_K8'))
    assert h2['grid'] == (32, 64, 448, 480) and h2['family'] == 8
    assert h2['metrics']['primary'] == ('C50',) and h2['metrics']['supportive'] == ('EDT',)
    assert h2['run_grids']['aug'] == (0, 32, 64, 128, 256, 384, 448, 480)
    assert h2['run_grids']['control'] == (0, 32, 64, 448, 480)
    assert tost['grid'] == (32, 64, 128, 256, 384, 448, 480)
    assert tost['family'] == 14 and tost['margin'] == .02
    assert [a['role'] for a in tost['arms']] == ['aug']
    assert h2['input_selection'] == tost['input_selection'] == 'block'
