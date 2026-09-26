"""exp_11 phase 1 in the shared HAA summariser: arm G, the decisions of plan section 3.

The synthetic arms and the legacy tree come from ``test_exp06_summarize_haa``: exp_11 adds
one arm and a frozen configuration to the very module exp_06 and exp_09 publish through,
so every exp_11 test here runs against the same fixtures those experiments are asserted
on, and the regression test at the end compares their payloads with the module ``main``
carries.
"""
import json
from pathlib import Path

import pytest

from tools import exp06_summarize_haa as subject

from test_exp06_summarize_haa import (  # noqa: F401  (fixtures used by name)
    HEADING, NEW_OFFSETS, ROOMS, SIZE, arms, build_cache, cache_root, legacy_root,
    stub_new_arms, synthetic_arm)

G, E, D, A, C = 'yawaug_hf', 'yawaug', 'control_hf', 'control', 'cyl_or'


# --- the arm registry -------------------------------------------------------------------


def test_the_registry_carries_arm_g_under_exp11s_own_root():
    """G is E's initialisation (a null literal, resolved from exp_04's approvals) and
    D's frame, in exp_11's tree."""
    arm = subject.ARMS[G]
    assert (arm['label'], arm['backbone'], arm['frame']) == ('G', 'simple', 'heading')
    assert arm['experiment'] == 'exp11' and arm['branch'] == 'new'
    assert arm['root'] == 'ckpt/exp11/sim2real/yawaug_hf'
    assert arm['init_sha256'] is None
    assert subject.NEW_ARMS[-1] == G and G in subject.ARMS
    assert subject.EXP11_ROOT == 'ckpt/exp11/sim2real'


def test_arm_g_is_read_under_exp11s_root_and_never_another_experiments():
    roots = {'exp06': 'a/six', 'exp09': 'b/nine', 'exp11': 'c/eleven'}
    assert subject.arm_directory(G, roots) == Path('c/eleven/yawaug_hf')
    assert subject.arm_directory(E, roots) == Path('b/nine/yawaug')
    assert subject.arm_directory(D, roots) == Path('a/six/control_hf')


def test_the_cli_takes_exp11s_root_as_its_own_option():
    parsed = subject.build_parser().parse_args([])
    assert parsed.exp11_root == subject.EXP11_ROOT
    assert subject.build_parser().parse_args(['--exp11-root', 'x']).exp11_root == 'x'


# --- the initialisation identity: G is checked exactly like E ---------------------------


def test_arm_gs_initialisation_is_exp04s_approved_checkpoint(tmp_path):
    """Extending ``expected_inits``: a null ``init_sha256`` alone would leave G unchecked."""
    from tools import exp06_approvals_api as api
    approved, _ = api.load_approved_digests(api.approved_path_default())
    record = subject.finalizer.exp04_aug_checkpoint(approved)
    inputs = {}
    inits = subject.expected_inits(approved, (C, E, G), inputs)
    assert inits[G] == inits[E] == record['checkpoint']['sha256']
    assert inits[C] == approved['artifacts']['epoch_012']['sha256']
    assert inputs == {str(Path(record['path']).resolve()): record['sha256']}
    assert subject.expected_inits(approved, (G,), {})[G] == record['checkpoint']['sha256']
    assert subject.expected_inits(None, (G,)) == {G: None}


def test_a_forged_exp04_pin_refuses_arm_g():
    from tools import exp06_approvals_api as api
    approved, _ = api.load_approved_digests(api.approved_path_default())
    wrong = json.loads(json.dumps(approved))
    wrong['reused']['exp04_approved_digests_sha256'] = 'a' * 64
    with pytest.raises(ValueError, match='exp04_approved_digests_sha256'):
        subject.expected_inits(wrong, (G,), {})

