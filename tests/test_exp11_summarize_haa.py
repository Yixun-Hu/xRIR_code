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


# --- the notation block of section 3, as named fields ------------------------------------

M = subject.H1_MARGIN_DB
ALL_FIELDS = ('category', 'y_non_inferior_at_margin', 'x_margin_advantage',
              'equivalent_at_margin')
HARM, BETTER, NONE = 'detected harm', 'detected improvement', 'no detected difference'


@pytest.mark.parametrize('interval,expected', [
    ((0.1, 0.4), (HARM, True, False, False)),
    ((-0.4, -0.1), (BETTER, False, False, False)),
    ((-0.5, -0.3), (BETTER, False, True, False)),
    ((-0.1, 0.1), (NONE, True, False, True)),
    ((0.0, 0.4), (NONE, True, False, False)),
    ((-0.23, 0.0), (NONE, False, False, False)),      # -L == m establishes nothing
    ((-0.23, 0.23), (NONE, False, False, False)),     # both endpoints on the boundary
    ((-0.6, -0.23), (BETTER, False, False, False)),   # U == -m establishes no advantage
    ((-0.2, 0.22), (NONE, True, False, True))])
def test_every_decision_field_reads_the_interval_in_its_own_direction(interval, expected):
    """X - Y = [L, U] at m: category two-sided, -L < m, U < -m and (-m, m) containment.

    Every inequality is strict, so an endpoint exactly on a decision boundary
    establishes neither the statement nor its negation.
    """
    values = subject.decision_fields(interval, M, ALL_FIELDS)
    assert tuple(values[name] for name in ALL_FIELDS) == expected
    assert list(values) == list(ALL_FIELDS)
    assert all(isinstance(values[name], bool) for name in ALL_FIELDS[1:])


def test_a_cell_without_an_interval_withholds_every_field():
    assert subject.decision_fields(None, M, ALL_FIELDS) == {name: None
                                                            for name in ALL_FIELDS}
    assert subject.decision_fields(None, None, ('category',)) == {'category': None}


def test_only_the_registered_fields_are_computable_and_margins_are_required():
    with pytest.raises(ValueError, match='unknown decision field'):
        subject.decision_fields((0.0, 1.0), M, ('verdict',))
    for field in ALL_FIELDS[1:]:
        with pytest.raises(ValueError, match='margin'):
            subject.decision_fields((0.0, 1.0), None, (field,))
    assert subject.decision_fields((0.1, 0.4), None, ('category',)) == {'category': HARM}


# --- the exp_11 decision cell: status, not a margin verdict ------------------------------


def flat_rows(difference=1.0, cohort=2, n_test=2, arms_of=(G, D)):
    """Rows whose paired difference is the same in every row: a zero-width interval."""
    import numpy as np
    size = cohort * len(subject.SEEDS)
    return {'a': np.full(size, difference), 'b': np.zeros(size),
            'clusters': np.asarray(list(range(cohort)) * len(subject.SEEDS)),
            'seeds': np.asarray([seed for seed in subject.SEEDS for _ in range(cohort)]),
            'cohort': cohort, 'n_test': n_test,
            'per_seed_diff': {seed: difference for seed in subject.SEEDS},
            'excluded': {arm: {'queries': 0, 'seeds': {seed: 0 for seed in subject.SEEDS}}
                         for arm in arms_of}}


def contrast_cell(arms, x, y, room='hallway', metric='c50', fields=ALL_FIELDS,
                  margin=M, n_boot=200):
    rows = subject.cell_rows(arms, x, y, room, metric)
    base = {'name': 'N1', 'kind': 'contrast', 'x': x, 'y': y, 'room': room,
            'metric': metric, 'contrast': '{} - {}'.format(x, y)}
    return subject.exp11_cell(rows, subject.void_reasons(rows, x, y), base, margin,
                              fields, n_boot=n_boot)


def test_a_decision_cell_reports_its_interval_and_its_fields(arms):
    cell = contrast_cell(arms, G, D)
    assert cell['status'] == 'reported' and 'verdict' not in cell
    assert cell['contrast'] == 'yawaug_hf - control_hf' and cell['x'] == G and cell['y'] == D
    assert cell['kind'] == 'contrast' and cell['name'] == 'N1'
    assert cell['cohort'] == SIZE['hallway'] and cell['void_reasons'] == []
    assert cell['convergence']['status'] == 'converged'
    assert cell['category'] in (HARM, BETTER, NONE)
    assert subject.decision_fields(cell['convergence']['interval'], M, ALL_FIELDS) == {
        name: cell[name] for name in ALL_FIELDS}


def test_a_void_cell_withholds_every_field(arms):
    for job in subject.JOBS:
        per = arms[G]['per'][job]['hallway']
        per['c50'] = [float('nan')] * len(per['index'])
    cell = contrast_cell(arms, G, D)
    assert cell['status'] == 'void' and cell['void_reasons']
    assert cell['diff'] is None and cell['two_way'] is None
    assert all(cell[name] is None for name in ALL_FIELDS)


def test_a_degenerate_interval_is_a_defined_unavailable_cell_not_an_abort():
    """Cancellation can give a zero-width interval; the frozen helper refuses it, and the
    exp_11 wrapper reports that refusal instead of aborting the whole summary."""
    rows = flat_rows()
    with pytest.raises(ValueError, match='zero-width'):
        subject.converged_two_way(rows, subject.ALPHA, 200)
    cell = subject.exp11_cell(rows, [], {'name': 'N1i', 'kind': 'interaction'}, M,
                              ALL_FIELDS, n_boot=200)
    assert cell['status'] == 'unavailable'
    assert cell['convergence']['status'] == 'unavailable'
    assert 'zero-width' in cell['convergence']['reason']
    assert cell['convergence']['interval'] is None
    assert cell['diff'] == 1.0 and cell['two_way']['lo'] == cell['two_way']['hi'] == 1.0
    assert all(cell[name] is None for name in ALL_FIELDS)


# --- N1i: the four-arm interaction (G - E) - (D - A) -------------------------------------

import numpy as np                                                       # noqa: E402

QUAD = (G, E, D, A)
N1I = {'name': 'N1i', 'kind': 'interaction', 'arms': QUAD, 'room': 'hallway',
       'metric': 'c50', 'margin': None, 'fields': ('category',),
       'reading': 'positive = larger heading-frame penalty after yaw pretraining'}


def column(arms, arm, room='hallway', metric='c50'):
    return np.stack([np.asarray(arms[arm]['per'][seed][room][metric], dtype=float)
                     for seed in subject.SEEDS])


def joint(arms, room='hallway', metric='c50'):
    """What the assembler must produce: one mask, four arms, three seeds."""
    values = {arm: column(arms, arm, room, metric) for arm in QUAD}
    mask = np.ones(values[G].shape[1], dtype=bool)
    for arm in QUAD:
        mask &= np.isfinite(values[arm]).all(axis=0)
    left = np.concatenate([(values[G][i] - values[E][i])[mask] for i in range(3)])
    right = np.concatenate([(values[D][i] - values[A][i])[mask] for i in range(3)])
    index = np.asarray(arms[G]['per']['seed0'][room]['index'])[mask]
    clusters = np.concatenate([index] * 3)
    seeds = np.concatenate([np.full(int(mask.sum()), seed) for seed in subject.SEEDS])
    return left, right, clusters, seeds, mask


def test_the_interaction_is_one_jointly_resampled_estimate(arms):
    """`paired_intervals(g - e, d - a, ...)` through the frozen bootstrap -- never two
    independently bootstrapped intervals subtracted from one another."""
    left, right, clusters, seeds, mask = joint(arms)
    rows = subject.interaction_rows(arms, QUAD, 'hallway', 'c50')
    assert np.array_equal(rows['a'], left) and np.array_equal(rows['b'], right)
    assert np.array_equal(rows['clusters'], clusters)
    assert list(rows['seeds']) == list(seeds)
    assert rows['cohort'] == int(mask.sum()) == SIZE['hallway']
    assert rows['n_test'] == SIZE['hallway']
    expected = subject.bootstrap.paired_intervals(left, right, clusters, seeds,
                                                  alpha=subject.ALPHA, n_boot=200, seed=0)
    assert subject.intervals(rows, subject.ALPHA, 200) == expected
    assert rows['excluded'][A]['queries'] == 0 and sorted(rows['excluded']) == sorted(QUAD)
    cell = subject.exp11_decision(arms, N1I, n_boot=200)
    assert cell['kind'] == 'interaction' and cell['arms'] == list(QUAD)
    assert cell['contrast'] == '(yawaug_hf - yawaug) - (control_hf - control)'
    assert cell['reading'] == N1I['reading']
    assert cell['diff'] == expected['diff'] and cell['status'] == 'reported'
    assert cell['category'] in (HARM, BETTER, NONE)
    assert 'y_non_inferior_at_margin' not in cell and 'equivalent_at_margin' not in cell


def test_the_interaction_refuses_four_arms_that_are_not_the_same_queries(arms):
    """Every arm is paired against the first: index, ir_path and protocol metadata."""
    per = arms[A]['per']['seed1']['hallway']
    per['index'] = list(reversed(per['index']))
    with pytest.raises(ValueError, match='pair '):
        subject.interaction_rows(arms, QUAD, 'hallway', 'c50')
    per['index'] = list(reversed(per['index']))
    arms[E]['per']['seed0']['hallway']['meta']['eval_seed'] = 7
    with pytest.raises(ValueError, match='eval_seed'):
        subject.interaction_rows(arms, QUAD, 'hallway', 'c50')


def test_one_mask_covers_every_arm_and_seed_and_each_arms_exclusions_are_reported(arms):
    """A query invalid in one arm's one seed leaves the cohort of all four arms."""
    arms[A]['per']['seed1']['hallway']['c50'][0] = float('nan')
    rows = subject.interaction_rows(arms, QUAD, 'hallway', 'c50')
    assert rows['cohort'] == SIZE['hallway'] - 1
    assert len(rows['a']) == len(rows['b']) == 3 * rows['cohort']
    assert rows['excluded'][A] == {'queries': 1,
                                   'seeds': {'seed0': 0, 'seed1': 1, 'seed2': 0}}
    assert rows['excluded'][G]['queries'] == 0
    left, right, _, _, _ = joint(arms)
    assert np.array_equal(rows['a'], left) and np.array_equal(rows['b'], right)


def test_the_joint_cohort_must_cover_ninety_nine_percent_and_the_components_still_count(
        arms):
    """The joint cohort rule, plus the component invalidity checks of G - E and D - A."""
    for job in subject.JOBS:
        arms[G]['per'][job]['hallway']['c50'][0] = float('nan')
    rows = subject.interaction_rows(arms, QUAD, 'hallway', 'c50')
    reasons = subject.interaction_void_reasons(arms, QUAD, rows, 'hallway', 'c50')
    assert any('joint cohort' in reason for reason in reasons)
    assert any(reason.startswith('yawaug_hf - yawaug:') for reason in reasons)
    assert not any(reason.startswith('control_hf - control:') for reason in reasons)
    cell = subject.exp11_decision(arms, N1I, n_boot=200)
    assert cell['status'] == 'void' and cell['category'] is None
    assert cell['void_reasons'] == reasons


def test_an_empty_joint_cohort_is_a_void_interaction_not_an_exception(arms):
    for job in subject.JOBS:
        per = arms[E]['per'][job]['hallway']
        per['c50'] = [float('nan')] * len(per['index'])
    cell = subject.exp11_decision(arms, N1I, n_boot=200)
    assert cell['status'] == 'void' and cell['cohort'] == 0
    assert any('all four compared arms' in reason for reason in cell['void_reasons'])
    assert cell['diff'] is None and cell['category'] is None


def test_an_interaction_that_cancels_exactly_is_unavailable_not_an_abort(arms):
    """(G - E) - (D - A) == 0 in every row: the zero-width interval of the plan's
    cancellation case, reported as a defined cell."""
    base = column(arms, A)
    for arm, shift in ((A, 0.0), (D, 1.0), (E, 0.0), (G, 1.0)):
        for i, seed in enumerate(subject.SEEDS):
            arms[arm]['per'][seed]['hallway']['c50'] = list(base[i] + shift)
    cell = subject.exp11_decision(arms, N1I, n_boot=200)
    assert cell['status'] == 'unavailable' and cell['category'] is None
    assert cell['diff'] == 0.0 and 'zero-width' in cell['convergence']['reason']
