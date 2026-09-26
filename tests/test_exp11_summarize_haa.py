"""exp_11 phase 1 in the shared HAA summariser: arm G, the decisions of plan section 3.

The synthetic arms and the legacy tree come from ``test_exp06_summarize_haa``: exp_11 adds
one arm and a frozen configuration to the very module exp_06 and exp_09 publish through,
so every exp_11 test here runs against the same fixtures those experiments are asserted
on, and the regression test at the end compares their payloads with the module ``main``
carries.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
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


# --- S1: three declared screen families, under exp_11's own suppression ------------------

FAMILIES = ((G, D), (C, G), (G, E))


def test_the_three_screen_families_are_declared_separately(arms):
    """Bonferroni-11 protects inside each family; no claim is made across the three."""
    screens = subject.exp11_screens(arms, FAMILIES, n_boot=200, adjusted_n_boot=200)
    assert list(screens) == ['yawaug_hf - control_hf', 'cyl_or - yawaug_hf',
                             'yawaug_hf - yawaug']
    for name, cells in screens.items():
        assert len(cells) == 11 and {cell['contrast'] for cell in cells} == {name}
        assert {cell['family'] for cell in cells} == {11}
        assert {cell['adjusted_alpha'] for cell in cells} == {0.05 / 11}
        assert {cell['label'] for cell in cells} <= {'detected harm', 'no detected '
                                                     'difference', 'detected improvement'}
        assert not any(cell['withheld'] for cell in cells)


def test_a_void_screen_cell_is_withheld_for_exp11_and_labelled_for_exp06(arms):
    """The universal suppression is exp_11's own gate: the historical screens keep the
    behaviour exp_06 and exp_09 registered."""
    for job in subject.JOBS:
        arms[G]['per'][job]['hallway']['c50'][0] = float('nan')
    cells = {(cell['room'], cell['metric']): cell
             for cell in subject.exp11_screen_cells(arms, (G, D), n_boot=200,
                                                    adjusted_n_boot=200)}
    voided = cells[('hallway', 'c50')]
    assert voided['withheld'] is True and voided['label'] is None
    assert voided['void_reasons'] and voided['nominal_two_way'] is not None
    assert cells[('hallway', 'edt')]['withheld'] is False
    assert cells[('hallway', 'edt')]['label'] in ('detected harm', 'detected improvement',
                                                  'no detected difference')
    historical = {(cell['room'], cell['metric']): cell
                  for cell in subject.screen_cells(arms, (G, D), n_boot=200,
                                                   adjusted_n_boot=200)}
    same = historical[('hallway', 'c50')]
    assert same['label'] in ('detected harm', 'detected improvement',
                             'no detected difference')
    assert 'withheld' not in same and 'void_reasons' not in same


def test_an_unconverged_screen_cell_carries_no_exp11_label(arms, monkeypatch):
    real = subject.converged_two_way
    monkeypatch.setattr(subject, 'converged_two_way',
                        lambda rows, alpha, n_boot: dict(real(rows, alpha, n_boot),
                                                         status='not_converged',
                                                         interval=None, n_boot=None))
    cells = subject.exp11_screen_cells(arms, (G, D), n_boot=200, adjusted_n_boot=200)
    assert all(cell['withheld'] and cell['label'] is None for cell in cells)
    assert all(cell['adjusted_two_way'] is None for cell in cells)
    assert all(cell['label'] == 'not converged'
               for cell in subject.screen_cells(arms, (G, D), n_boot=200,
                                                adjusted_n_boot=200))


def test_an_empty_cohort_is_withheld_rather_than_not_available(arms):
    for job in subject.JOBS:
        per = arms[G]['per'][job]['hallway']
        per['c50'] = [float('nan')] * len(per['index'])
    cells = {(cell['room'], cell['metric']): cell
             for cell in subject.exp11_screen_cells(arms, (G, D), n_boot=200,
                                                    adjusted_n_boot=200)}
    cell = cells[('hallway', 'c50')]
    assert cell['withheld'] is True and cell['label'] is None
    assert cell['nominal_two_way'] is None and cell['cohort'] == 0


# --- R1: historical rows copied from the hash-bound canonical records --------------------

REAL_EXP06 = Path(subject.REPO) / subject.EXP06_STATS
REAL_EXP09 = Path(subject.REPO) / subject.EXP09_STATS


def write_source(directory, name, tables, summary='historical summary\n'):
    """A canonical record and the summary its own bytes bind."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'summary.txt').write_text(summary)
    record = dict(tables, summary_sha256=hashlib.sha256(summary.encode()).hexdigest())
    (directory / name).write_text(json.dumps(record))
    return directory / name


def source_tables(extra=None):
    descriptive = [
        {'contrast': 'cyl_hf - cyl', 'room': 'hallway', 'metric': 'c50', 'diff': 1.09,
         'nominal_two_way': {'lo': 0.9, 'hi': 1.2}},
        {'contrast': 'cyl_or - control_hf', 'room': 'hallway', 'metric': 'edt',
         'diff': -0.01, 'nominal_two_way': {'lo': -0.02, 'hi': 0.0}},
        {'contrast': 'cyl_or - control_hf', 'room': 'hallway', 'metric': 'c50',
         'diff': -0.6289887046267822,
         'nominal_two_way': {'lo': -0.7643895950033573, 'hi': -0.5001049541194548}},
        {'contrast': 'control_hf - control', 'room': 'hallway', 'metric': 'c50',
         'diff': 0.6307393052674443,
         'nominal_two_way': {'lo': 0.5001244137920259, 'hi': 0.7701659721472814}}]
    decision = {'contrast': 'cyl_or - control', 'room': 'hallway', 'metric': 'c50',
                'diff': 0.0017506006406620145, 'two_way': {'lo': -0.057, 'hi': 0.060},
                'convergence': {'status': 'converged', 'n_boot': 10000},
                'verdict': 'pass', 'margin': 0.23}
    tables = {'D': descriptive, 'H1': decision,
              'H1b': dict(decision, contrast='cyl_or - cyl_hf', diff=-2.0078617508011820)}
    return dict(tables, **(extra or {}))


@pytest.fixture
def sources(tmp_path):
    """exp_06's and exp_09's canonical records, as R1 must read them."""
    write_source(tmp_path / 'ckpt/exp06', 'stats.json', source_tables())
    write_source(tmp_path / 'ckpt/exp09', 'stats.json', {
        'E1': {'contrast': 'yawaug - control', 'room': 'hallway', 'metric': 'c50',
               'diff': 0.2765036091404135, 'two_way': {'lo': 0.204, 'hi': 0.352},
               'convergence': {'status': 'converged', 'n_boot': 10000},
               'verdict': 'fail', 'category': 'detected harm',
               'non_inferior_at_margin': False}}, summary='exp09 summary\n')
    return tmp_path


def test_the_historical_rows_are_the_registered_five(sources):
    rows = subject.historical_rows(repo=sources)
    assert [row['name'] for row in rows] == ['C - D', 'D - A', 'C - A', 'C - F', 'E - A']
    assert [row['contrast'] for row in rows] == [
        'cyl_or - control_hf', 'control_hf - control', 'cyl_or - control',
        'cyl_or - cyl_hf', 'yawaug - control']
    assert {row['room'] for row in rows} == {'hallway'}
    assert {row['metric'] for row in rows} == {'c50'}
    assert {row['inference'] for row in rows} == {'none (copied)'}
    assert [row['kind'] for row in rows] == ['descriptive', 'descriptive', 'decision',
                                             'decision', 'decision']


def test_a_descriptive_row_is_copied_as_descriptive_and_never_promoted(sources):
    row = subject.historical_rows(repo=sources)[0]
    assert row['diff'] == -0.6289887046267822
    assert row['nominal_two_way'] == {'lo': -0.7643895950033573, 'hi': -0.5001049541194548}
    assert row['two_way'] is None and row['convergence'] is None
    assert row['verdict'] is None and row['category'] is None
    assert row['not_recorded'] == ['two_way', 'convergence', 'verdict', 'category',
                                   'non_inferior_at_margin']
    assert row['selector'] == {'table': 'D', 'contrast': 'cyl_or - control_hf',
                               'room': 'hallway', 'metric': 'c50'}
    assert row['source'] == subject.EXP06_STATS
    assert row['source_sha256'] == subject.provenance.sha256_file(
        Path(sources) / subject.EXP06_STATS)


def test_a_decision_row_keeps_every_field_its_source_recorded(sources):
    rows = {row['name']: row for row in subject.historical_rows(repo=sources)}
    assert rows['C - A']['verdict'] == 'pass'
    assert rows['C - A']['convergence'] == {'status': 'converged', 'n_boot': 10000}
    assert rows['C - A']['two_way'] == {'lo': -0.057, 'hi': 0.060}
    assert rows['C - A']['nominal_two_way'] is None
    assert rows['E - A']['category'] == 'detected harm'
    assert rows['E - A']['non_inferior_at_margin'] is False
    assert rows['E - A']['source'] == subject.EXP09_STATS
    assert rows['E - A']['not_recorded'] == ['nominal_two_way']


def test_a_row_is_selected_by_its_contrast_and_never_by_a_list_position(sources):
    path = Path(sources) / subject.EXP06_STATS
    record = json.loads(path.read_text())
    record['D'] = list(reversed(record['D']))
    path.write_text(json.dumps(record))
    assert subject.historical_rows(repo=sources)[0]['diff'] == -0.6289887046267822
    duplicate = [row for row in record['D']
                 if row['contrast'] == 'cyl_or - control_hf' and row['metric'] == 'c50']
    record['D'] = record['D'] + [dict(duplicate[0])]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='rows for'):
        subject.historical_rows(repo=sources)


def test_a_missing_source_a_changed_summary_or_a_missing_field_is_refused(sources, tmp_path):
    path = Path(sources) / subject.EXP06_STATS
    record = json.loads(path.read_text())
    (Path(sources) / 'ckpt/exp06/summary.txt').write_text('edited\n')
    with pytest.raises(ValueError, match='summary_sha256'):
        subject.historical_rows(repo=sources)
    (Path(sources) / 'ckpt/exp06/summary.txt').write_text('historical summary\n')
    del record['H1']['verdict']
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='records no verdict'):
        subject.historical_rows(repo=sources)
    with pytest.raises(ValueError, match='missing historical source'):
        subject.historical_rows(repo=tmp_path / 'empty')


def test_the_copied_rows_bind_the_bytes_they_were_read_from(sources):
    inputs = {}
    subject.historical_rows(repo=sources, inputs=inputs)
    assert sorted(Path(path).name for path in inputs) == ['stats.json', 'stats.json',
                                                          'summary.txt', 'summary.txt']
    for path, digest in inputs.items():
        assert subject.provenance.sha256_file(path) == digest


@pytest.mark.skipif(not (REAL_EXP06.is_file() and REAL_EXP09.is_file()),
                    reason='needs the exp_06 and exp_09 canonical records')
def test_the_real_canonical_records_carry_exactly_the_mapped_rows():
    """The source-field map against the files themselves, not a fixture's idea of them."""
    rows = {row['name']: row for row in subject.historical_rows()}
    assert rows['C - D']['diff'] == -0.6289887046267822
    assert rows['C - D']['nominal_two_way']['lo'] == -0.7643895950033573
    assert rows['C - D']['verdict'] is None and rows['C - D']['convergence'] is None
    assert rows['D - A']['diff'] == 0.6307393052674443
    assert rows['C - A']['verdict'] == 'pass' and rows['C - A']['convergence']['status'] \
        == 'converged'
    assert rows['C - F']['verdict'] == 'pass' and rows['C - F']['diff'] < 0
    assert rows['E - A']['verdict'] == 'fail'
    assert rows['E - A']['category'] == 'detected harm'
    assert rows['E - A']['non_inferior_at_margin'] is False
    assert rows['E - A']['diff'] == 0.2765036091404135


# --- the frozen exp_11 configuration and its phase-1 tables ------------------------------


def test_the_experiments_freeze_exp11s_phase_one():
    eleven = subject.EXPERIMENTS['exp11']
    assert eleven['arms'] == ('control', 'cyl', 'cyl_or', 'control_hf', 'cyl_hf',
                              'yawaug', 'yawaug_hf')
    assert eleven['phase'] == 'phase1'
    assert eleven['decisions'] == () and eleven['classified'] == ()
    specs = {spec['name']: spec for spec in eleven['exp11_decisions']}
    assert [spec['name'] for spec in eleven['exp11_decisions']] == ['N1', 'N1i', 'N2', 'N3']
    assert specs['N1']['pair'] == (G, D) and specs['N1']['fields'] == ('category',)
    assert specs['N1i']['kind'] == 'interaction' and specs['N1i']['arms'] == (G, E, D, A)
    assert specs['N1i']['fields'] == ('category',)
    assert specs['N2']['pair'] == (C, G) and specs['N2']['fields'] == (
        'category', 'y_non_inferior_at_margin', 'x_margin_advantage')
    assert specs['N3']['pair'] == (G, E) and specs['N3']['fields'] == (
        'category', 'equivalent_at_margin')
    assert specs['N2']['margin'] == specs['N3']['margin'] == subject.H1_MARGIN_DB
    assert specs['N1']['margin'] is None and specs['N1i']['margin'] is None
    assert {(s['room'], s['metric']) for s in eleven['exp11_decisions']} == {('hallway',
                                                                             'c50')}
    assert eleven['screens'] == ((G, D), (C, G), (G, E))
    assert eleven['historical'] == subject.EXP11_HISTORICAL
    assert eleven['outputs'] == ('ckpt/exp11/phase1/stats.json',
                                 'ckpt/exp11/phase1/summary.txt')


@pytest.fixture
def phase1(arms, sources):
    return subject.analyse(arms, n_boot=200, adjusted_n_boot=200, experiment='exp11',
                           historical_root=sources)


def test_the_exp11_analysis_publishes_its_decisions_screens_and_copied_rows(phase1):
    assert phase1['experiment'] == 'exp11' and phase1['phase'] == 'phase1'
    assert set(phase1) >= {'N1', 'N1i', 'N2', 'N3', 'screens', 'historical'}
    assert not {'H1', 'H1b', 'H2', 'D', 'E1', 'E2', 'E3'} & set(phase1)
    assert phase1['N1']['contrast'] == 'yawaug_hf - control_hf'
    assert phase1['N1i']['contrast'] == '(yawaug_hf - yawaug) - (control_hf - control)'
    assert phase1['N2']['contrast'] == 'cyl_or - yawaug_hf'
    assert phase1['N3']['contrast'] == 'yawaug_hf - yawaug'
    assert all(phase1[name]['status'] == 'reported' for name in ('N1', 'N1i', 'N2', 'N3'))
    assert 'verdict' not in phase1['N1'] and 'verdict' not in phase1['N2']
    assert isinstance(phase1['N2']['y_non_inferior_at_margin'], bool)
    assert isinstance(phase1['N2']['x_margin_advantage'], bool)
    assert 'equivalent_at_margin' not in phase1['N2']
    assert isinstance(phase1['N3']['equivalent_at_margin'], bool)
    assert 'y_non_inferior_at_margin' not in phase1['N1']
    assert list(phase1['screens']) == ['yawaug_hf - control_hf', 'cyl_or - yawaug_hf',
                                       'yawaug_hf - yawaug']
    assert all(len(cells) == 11 for cells in phase1['screens'].values())
    assert [row['name'] for row in phase1['historical']] == ['C - D', 'D - A', 'C - A',
                                                             'C - F', 'E - A']
    assert {row['inference'] for row in phase1['historical']} == {'none (copied)'}


def test_the_exp11_tables_carry_arm_g_everywhere_the_arms_are_described(phase1):
    """Descriptive (c): G in the zero-shot room-frame side split, and per seed."""
    assert 'yawaug_hf|hallway|c50' in phase1['side_split']['cells']
    assert phase1['side_split']['job'] == subject.ZEROSHOT
    row = phase1['rows']['yawaug_hf|fine-tuned|hallway|c50']
    assert row['jobs'] == list(subject.SEEDS) and len(row['per_run']) == 3
    assert phase1['arms'][G] == {'branch': 'new', 'root': 'ckpt/exp06/sim2real/yawaug_hf',
                                 'closure': phase1['arms'][G]['closure'],
                                 'backbone': 'simple', 'frame': 'heading'}


def test_the_copied_rows_are_bound_inputs_of_the_analysis(phase1, sources):
    for name in ('ckpt/exp06/stats.json', 'ckpt/exp09/stats.json', 'ckpt/exp06/summary.txt'):
        assert str((Path(sources) / name).resolve()) in phase1['inputs']


def test_a_draft_exp11_run_states_no_conclusion(arms, sources):
    result = subject.analyse(arms, n_boot=200, adjusted_n_boot=200, exploratory=True,
                             experiment='exp11', historical_root=sources)
    for name in ('N1', 'N1i', 'N2', 'N3'):
        cell = result[name]
        assert cell['status'] == 'suppressed (draft)'
        assert all(cell[field] is None for field in cell['fields'])
    assert result['historical'][0]['diff'] is not None   # a copied row states nothing new


# --- the published phase outputs, and what the historical experiments still publish -----


def test_no_experiments_run_may_write_anothers_canonical_record(tmp_path):
    eleven = [str(subject.REPO / name) for name in subject.EXPERIMENTS['exp11']['outputs']]
    six = [str(subject.REPO / name) for name in subject.EXPERIMENTS['exp06']['outputs']]
    nine = [str(subject.REPO / name) for name in subject.EXPERIMENTS['exp09']['outputs']]
    for other in (six, nine):
        with pytest.raises(ValueError, match='canonical'):
            subject.check_output_paths('exp11', other[0], str(tmp_path / 'summary.txt'))
        with pytest.raises(ValueError, match='canonical'):
            subject.check_output_paths('exp11', str(tmp_path / 'stats.json'), other[1])
    for experiment in ('exp06', 'exp09'):
        with pytest.raises(ValueError, match='canonical'):
            subject.check_output_paths(experiment, eleven[0],
                                       str(tmp_path / 'summary.txt'))
    subject.check_output_paths('exp11', eleven[0], eleven[1])


def test_the_rendered_phase_summary_carries_every_exp11_block(phase1):
    text = subject.render(phase1)
    assert 'N1 yawaug_hf - control_hf hallway c50' in text
    assert 'N1i (yawaug_hf - yawaug) - (control_hf - control)' in text
    assert 'category:' in text and 'y_non_inferior_at_margin:' in text
    assert 'equivalent_at_margin:' in text
    for name in ('yawaug_hf - control_hf', 'cyl_or - yawaug_hf', 'yawaug_hf - yawaug'):
        assert 'S1 screen {}'.format(name) in text
    assert 'R1 historical rows (copied; no new inference)' in text
    assert 'C - D' in text and 'not recorded' in text
    assert 'Room-frame side split' in text and 'yawaug_hf|hallway|c50' in text


def test_a_withheld_statement_renders_as_not_available(arms, sources):
    for job in subject.JOBS:
        per = arms[G]['per'][job]['hallway']
        per['c50'] = [float('nan')] * len(per['index'])
    result = subject.analyse(arms, n_boot=200, adjusted_n_boot=200, experiment='exp11',
                             historical_root=sources)
    assert result['N1']['status'] == 'void' and result['N1']['category'] is None
    text = subject.render(result)
    assert 'category: not available' in text
    assert '-> void' in text and 'void: ' in text


@pytest.mark.skipif(not (REAL_EXP06.is_file() and REAL_EXP09.is_file()),
                    reason='needs the exp_06 and exp_09 canonical records')
def test_the_cli_publishes_exp11s_phase_outputs_and_nothing_elses(legacy_root,
                                                                  stub_new_arms, tmp_path):
    receipt = tmp_path / 'r.json'
    subject.write_legacy_receipt(receipt, legacy_root, strict=False)
    out, summary = tmp_path / 'phase1.json', tmp_path / 'phase1.txt'
    assert subject.main(['--experiment', 'exp11', '--legacy-root', str(legacy_root),
                         '--new-root', 'unused', '--exp09-root', 'unused',
                         '--exp11-root', 'unused', '--legacy-receipt', str(receipt),
                         '--json', str(out), '--summary', str(summary), '--n-boot', '200',
                         '--n-boot-adjusted', '200', '--exploratory']) == 0
    record = json.loads(out.read_text())
    assert record['experiment'] == 'exp11' and record['phase'] == 'phase1'
    assert sorted(record['arms']) == sorted(subject.EXPERIMENTS['exp11']['arms'])
    assert record['decisions'] == ['N1', 'N1i', 'N2', 'N3']
    assert all(record[name]['status'] == 'suppressed (draft)' for name in record['decisions'])
    assert len(record['historical']) == 5 and len(record['screens']) == 3
    assert not {'H1', 'H1b', 'H2', 'D', 'E1', 'E2', 'E3'} & set(record)
    assert record['summary_sha256'] == hashlib.sha256(summary.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):    # a phase record is never overwritten
        subject.write_outputs(record, str(out), str(summary))


def test_exp06_and_exp09_publish_exactly_what_main_publishes(arms, tmp_path):
    """The schema extension changes no historical statistic and no historical summary."""
    from test_exp06_summarize_haa import base_summariser
    base = base_summariser(tmp_path)
    settings = dict(n_boot=200, adjusted_n_boot=200, exploratory=True)
    for experiment in ('exp06', 'exp09'):
        selected = {name: arms[name] for name in base.EXPERIMENTS[experiment]['arms']}
        named = {} if experiment == 'exp06' else {'experiment': experiment}
        before = base.analyse(selected, **dict(settings, **named))
        after = subject.analyse(selected, **dict(settings, **named))
        assert list(after) == list(before), experiment
        assert json.dumps(after, sort_keys=True) == json.dumps(before, sort_keys=True)
        assert subject.render(after) == base.render(before), experiment
        assert 'phase' not in after and 'screens' not in after
        assert 'historical' not in after and 'decisions' not in after


# --- the unavailable path is the degenerate interval, and nothing else ------------------


def test_a_convergence_refusal_that_is_not_degenerate_still_aborts(arms, monkeypatch):
    """Only the zero-width case of the plan becomes a defined cell; every other refusal
    of the frozen helper is an error, not a quietly unavailable statement."""
    def refuse(rows, alpha, n_boot):
        raise ValueError('the interval function did not return endpoints: broken')
    monkeypatch.setattr(subject, 'converged_two_way', refuse)
    with pytest.raises(ValueError, match='did not return endpoints'):
        contrast_cell(arms, G, D)


def test_an_empty_cohort_is_never_bootstrapped_even_without_a_void_reason():
    rows = flat_rows()
    rows.update(cohort=0, a=rows['a'][:0], b=rows['b'][:0], clusters=rows['clusters'][:0],
                seeds=rows['seeds'][:0])
    with pytest.raises(ValueError, match='empty cohort'):
        subject.exp11_cell(rows, [], {'name': 'N1', 'kind': 'contrast'}, M, ALL_FIELDS,
                           n_boot=200)
