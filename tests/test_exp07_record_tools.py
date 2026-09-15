"""The exp_07 record generators run on real producer outputs over synthetic runs."""
import json
from pathlib import Path

import pytest

from exp07_fixture import exp07_fixture  # noqa: F401  (fixture)
from test_paired_compare import admission_fixture  # noqa: F401  (table_fixture needs it)
from test_results_table import table_fixture  # noqa: F401  (exp_04's unseen table)
from tools import exp07_pairs as pairs_producer, exp07_table as table_producer
from tools import results_table as rt
from tools.exp07_profiles import get_profile, json_value
from tools.exp07_record import load_asset

ROLES = ('seen_cyl', 'seen_aug', 'released_seen')


def sidecar(path):
    return Path(str(path) + '.provenance.json')


@pytest.fixture
def record_inputs(exp07_fixture, table_fixture, tmp_path):  # noqa: F811
    """One synthetic run layout published through both exp_07 producers, plus exp_04's."""
    md = load_asset('make_results_md')
    built = exp07_fixture()
    built.profile['n_boot'] = 2000
    results = tmp_path / 'results'
    results.mkdir()
    paths = dict(table=results / 'TABLE_SEEN_V1.json', pairs=[])
    table, admitted = table_producer.build_table(built.directories, built.profile,
                                                 built.approved, producer=built.producer)
    table_producer.write_outputs(table, admitted, str(paths['table']),
                                 str(results / 'model_comparison_seen.md'),
                                 command=['exp07_table.py'])
    profile = json_value(get_profile('PAIRS_SEEN_V1'))
    profile.update(n_boot=2000, **{key: built.profile[key] for key in
                                   ('dataset', 'arms', 'seeds', 'train_inventory_files')})
    for role in ROLES:
        sides = [[path for shot in (8, 1) for path in built.paths[(item, shot)]]
                 for item in (role, 'seen_simple')]
        result, admitted = pairs_producer.build_pairs(*sides, profile=profile,
                                                      approved=built.approved,
                                                      producer=built.producer)
        paths['pairs'].append(results / ('PAIRS_SEEN_V1_' + role + '.json'))
        pairs_producer.write_outputs(result, admitted, str(paths['pairs'][-1]),
                                     str(results / (role + '.summary.txt')),
                                     pairs_producer.render_summary)
    unseen, admitted = rt.build_table(table_fixture.directories)
    paths['unseen'] = results / 'TABLE_V1.json'
    rt.write_outputs(unseen, admitted, str(paths['unseen']), str(results / 'model_comparison.md'))
    published = json.loads(sidecar(paths['unseen']).read_text())['outputs']
    paths['binding'] = results / 'binding_report_20260915T000000000000Z.json'
    paths['binding'].write_text(json.dumps(dict(schema_version=1, git_HEAD='b' * 40, results=[
        dict(profile='TABLE_V1', producer_commit='b' * 40,
             sidecar=dict(path=str(sidecar(paths['unseen'])),
                          sha256=md.sha(sidecar(paths['unseen']).read_bytes())),
             outputs=[dict(path=path, sha256=digest)
                      for path, digest in sorted(published.items())])]), indent=1))
    paths['out'] = tmp_path / 'seen_protocol_results.md'
    return paths


def argv(paths, **changes):
    values = dict(table=paths['table'], pairs=paths['pairs'], unseen_table=paths['unseen'],
                  unseen_binding=paths['binding'], out=paths['out'])
    values.update(changes)
    return (['--table', str(values['table']), '--pairs'] + [str(p) for p in values['pairs']] +
            ['--unseen-table', str(values['unseen_table']), '--unseen-binding',
             str(values['unseen_binding']), '--out', str(values['out'])])


def test_the_generator_admits_the_complete_record(record_inputs):
    md = load_asset('make_results_md')
    args, record, head = md.arguments(argv(record_inputs))
    assert len(record['table']['rows']) == 8 and len(record['pairs']) == 3
    assert sum(len(data['cells']) for data in record['pairs']) == 30
    assert sorted(tuple(data['pairing']) for data in record['pairs']) == sorted(md.PAIRINGS)
    assert len(record['unseen']['rows']) == 6 and len(record['receipts']) == 5
    assert record['receipts'][-1]['binding_report']['path'] == str(record_inputs['binding'])
    assert len(head) == 40 and args.out == str(record_inputs['out'])


def edit(path, mutate, restamp=True):
    """Rewrite a canonical JSON, optionally repairing the digest its sidecar records."""
    md = load_asset('make_results_md')
    data = json.loads(Path(path).read_text())
    mutate(data)
    Path(path).write_text(json.dumps(data))
    if restamp:
        side = json.loads(sidecar(path).read_text())
        side['outputs'][str(Path(path).resolve())] = md.sha(Path(path).read_bytes())
        sidecar(path).write_text(json.dumps(side))


MUTATIONS = {
    'sidecar_missing': lambda paths: sidecar(paths['table']).unlink(),
    'tampered': lambda paths: edit(paths['table'], lambda d: d['rows'].pop(), restamp=False),
    'exploratory': lambda paths: edit(paths['table'], lambda d: d.update(exploratory=True)),
    'deviations': lambda paths: edit(paths['table'], lambda d: d.update(deviations=['a run'])),
    'profile_digest': lambda paths: edit(paths['table'],
                                         lambda d: d['profile'].update(n_boot=1)),
    'table_coverage': lambda paths: edit(paths['table'], lambda d: d['rows'].pop()),
    'table_metric': lambda paths: edit(paths['table'],
                                       lambda d: d['rows'][0]['metrics'].pop('log_mse')),
    'pairs_reconverge': lambda paths: edit(paths['pairs'][0],
                                           lambda d: d.update(reconverge_required=['EDT K=8'])),
    'pairs_not_final': lambda paths: edit(paths['pairs'][1], lambda d: d.update(final=False)),
    'pairs_verdict': lambda paths: edit(paths['pairs'][2],
                                        lambda d: d.update(verdict='non-inferior')),
    'pairs_decision': lambda paths: edit(paths['pairs'][0],
                                         lambda d: d.update(decision_driving=True)),
    'pairs_coverage': lambda paths: edit(paths['pairs'][1], lambda d: d['cells'].pop()),
    'pairs_exploratory': lambda paths: edit(paths['pairs'][2],
                                            lambda d: d.update(exploratory=True)),
    'unseen_coverage': lambda paths: edit(paths['unseen'], lambda d: d['rows'].pop()),
    'unseen_binding': lambda paths: paths['binding'].write_text(
        json.dumps(dict(schema_version=1, git_HEAD='b' * 40, results=[]))),
}


@pytest.mark.parametrize('mutation', sorted(MUTATIONS))
def test_the_generator_refuses_unbound_or_provisional_input(record_inputs, mutation):
    md = load_asset('make_results_md')
    MUTATIONS[mutation](record_inputs)
    with pytest.raises((ValueError, OSError, KeyError)):
        md.arguments(argv(record_inputs))


@pytest.mark.parametrize('name,message', [('duplicate', 'duplicate canonical input'),
                                          ('overlap', 'output overlaps canonical input')])
def test_the_generator_refuses_a_duplicate_input_or_an_overlapping_output(record_inputs,
                                                                          name, message):
    md = load_asset('make_results_md')
    changes = dict(pairs=record_inputs['pairs'][:2] + record_inputs['pairs'][1:2]) \
        if name == 'duplicate' else dict(out=record_inputs['table'])
    with pytest.raises(ValueError, match=message):
        md.arguments(argv(record_inputs, **changes))


def test_an_unseen_table_the_exp04_report_does_not_bind_is_refused(record_inputs):
    """Its own sidecar is repaired; only the binding report still names the old digest."""
    md = load_asset('make_results_md')
    edit(record_inputs['unseen'], lambda data: data['rows'][0].update(label='Edited'))
    with pytest.raises(ValueError, match='not the one the exp_04 report binds'):
        md.arguments(argv(record_inputs))


def test_only_one_pairing_of_a_missing_one_is_refused(record_inputs):
    md = load_asset('make_results_md')
    with pytest.raises(ValueError, match='incomplete pairing coverage'):
        md.arguments(argv(record_inputs, pairs=record_inputs['pairs'][:2]))
