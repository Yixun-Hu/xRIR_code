"""The exp_07 record generators run on real producer outputs over synthetic runs."""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from exp07_fixture import exp07_fixture  # noqa: F401  (fixture)
from test_paired_compare import admission_fixture  # noqa: F401  (table_fixture needs it)
from test_results_table import table_fixture  # noqa: F401  (exp_04's unseen table)
from tools import exp07_pairs as pairs_producer, exp07_table as table_producer
from tools import provenance as p
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
    paths['built'] = built
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


NUMBERS = r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![\w])'
VERDICT_WORDS = ('verdict', 'significant', 'non-inferior', 'superior', 'equivalent',
                 'p-value', 'better than', 'worse than', 'wins')


def test_the_markdown_is_canonical_and_byte_stable(record_inputs):
    md = load_asset('make_results_md')
    md.main(argv(record_inputs))
    first = record_inputs['out'].read_text()
    md.main(argv(record_inputs))
    assert first == record_inputs['out'].read_text()
    assert first.startswith('generated by make_results_md.py from ')
    assert md.ROUNDING in first and md.DESCRIPTIVE in first
    for title in ('Seen split', 'Descriptive paired differences', 'Seen and unseen protocols',
                  'Provenance'):
        assert '## ' + title in first
    assert first.index('Seen split') < first.index('Descriptive paired')
    assert first.index('Descriptive paired') < first.index('Seen and unseen protocols')
    assert first.index('Seen and unseen protocols') < first.index('## Provenance')
    cells = '\n'.join(line for line in first.split('\n') if line.startswith('|'))
    assert not any(word in cells.lower() for word in VERDICT_WORDS)  # the disclaimer names them
    assert '{' not in first and '[' not in first


def test_every_rendered_number_comes_from_the_canonical_json(record_inputs):
    md = load_asset('make_results_md')
    md.main(argv(record_inputs))
    rendered = record_inputs['out'].read_text().split('\n')
    sources = [record_inputs['table'], record_inputs['unseen']] + record_inputs['pairs']
    raw = ''.join(Path(path).read_text() + sidecar(path).read_text() for path in sources)
    allowed = set(re.findall(NUMBERS, raw))
    allowed |= {format(float(value) * scale, spec) for value in list(allowed)
                for scale in (1, .001, 100, 1000)
                for spec in ('.1f', '.2f', '.3f', '.6g')}
    allowed.add('95')  # the '95% interval' column labels are the only numeric literals
    assert set(re.findall(NUMBERS, '\n'.join(rendered[4:]))) <= allowed


def test_the_tables_report_the_producers_own_values(record_inputs):
    md = load_asset('make_results_md')
    _, record, head = md.arguments(argv(record_inputs))
    md.main(argv(record_inputs))
    text = record_inputs['out'].read_text()
    assert head in text
    for row in record['table']['rows']:
        edt = row['metrics']['EDT']
        assert '{} s ± {} ms'.format(format(edt['mean'] / 1000, '.3f'),
                                     md.display(edt['sd'], 'EDT')) in text
        assert '{} ± {} dB'.format(*[md.display(row['metrics']['C50'][key], 'C50')
                                     for key in ('mean', 'sd')]) in text
        assert row['label'] in text
    for data in record['pairs']:
        for cell in data['cells']:
            relative = cell['statistics']['relative']
            assert md.percent(relative['estimate']) in text
            assert md.percent(relative['convergence']['ratio']) in text
            assert md.native(cell['metric'], cell['statistics']['absolute']['estimate']) in text
    for receipt in record['receipts']:
        assert receipt['sha256'] in text and receipt['profile_digest'] in text
        assert receipt['approved_digests']['sha256'] in text
    assert record['receipts'][-1]['binding_report']['sha256'] in text


def test_the_combined_table_pairs_each_seen_arm_with_its_unseen_row(record_inputs):
    md = load_asset('make_results_md')
    _, record, head = md.arguments(argv(record_inputs))
    title, headers, rows = list(md.tables(record, head))[2]
    assert title.startswith('Seen and unseen protocols')
    assert [row[1] for row in rows] == [1] * 4 + [8] * 4
    unseen = {(row['role'], row['num_shot']): row for row in record['unseen']['rows']}
    seen = {(row['role'], row['num_shot']): row for row in record['table']['rows']}
    for index, (seen_role, unseen_role) in enumerate(md.PROTOCOL_PAIRS):
        row = rows[index]
        assert row[0] == seen[(seen_role, 1)]['label']
        assert row[2] == md.mean_sd(seen[(seen_role, 1)]['metrics'], 'EDT')
        assert row[5] == md.mean_sd(unseen[(unseen_role, 1)]['metrics'], 'EDT')
    assert rows[3][0] == seen[(md.REFERENCE_ROLE, 1)]['label'] and rows[3][5:] == ['—'] * 3


def test_the_html_page_is_offline_stable_and_shows_the_same_cells(record_inputs):
    html, md = load_asset('make_results_html'), load_asset('make_results_md')
    out = record_inputs['out'].with_suffix('.html')
    html.main(argv(record_inputs, out=out))
    first = out.read_text()
    html.main(argv(record_inputs, out=out))
    assert first == out.read_text() and first.startswith('<!doctype html>')
    assert '<script' not in first and 'http://' not in first and 'https://' not in first
    assert first.count('<table>') == 5 and first.rstrip().endswith('</html>')
    _, record, head = md.arguments(argv(record_inputs))
    for title, headers, rows in md.tables(record, head):
        assert '<h2>' + html.esc(title) + '</h2>' in first
        for row in rows:
            assert all('<td>' + html.esc(value) + '</td>' in first for value in row)


def test_the_html_page_refuses_the_same_input_the_markdown_refuses(record_inputs):
    html = load_asset('make_results_html')
    MUTATIONS['pairs_reconverge'](record_inputs)
    out = record_inputs['out'].with_suffix('.html')
    with pytest.raises(ValueError, match='not final'):
        html.main(argv(record_inputs, out=out))
    assert not out.exists()


@pytest.fixture
def bound(record_inputs, tmp_path, monkeypatch):
    """The whole record: forty runs, three attempts, four producer outputs, three documents."""
    binder = load_asset('bind_provenance')
    built = record_inputs['built']
    monkeypatch.setattr(binder, 'load_approved_digests',
                        lambda path=None: (built.pins, built.approved[1]))
    released = next(arm for arm in built.profile['arms'] if arm['reference'])
    monkeypatch.setattr(binder, 'RELEASED', released['checkpoint'])
    monkeypatch.setattr(binder, 'RELEASED_SHA256', released['sha256'])
    audit = tmp_path / 'alignment_audit_seen.json'
    audit.write_text('{"protocol": "seen"}')
    documents = [record_inputs['out']]
    load_asset('make_results_md').main(argv(record_inputs))
    for suffix, module in (('.html', 'make_results_html'), ('.tex', 'make_latex')):
        documents.append(record_inputs['out'].with_suffix(suffix))
        load_asset(module).main(argv(record_inputs, out=documents[-1]))
    reports = tmp_path / 'reports'
    reports.mkdir()
    return SimpleNamespace(binder=binder, built=built, paths=record_inputs, reports=reports,
        arguments=dict(runs=list(built.directories), audit=str(audit),
                       attempt=[str(built.attempts[role]) for role in built.pins['checkpoints']],
                       results=[str(record_inputs['table'])] + [str(item) for item in
                                                                record_inputs['pairs']],
                       rendered=[str(item) for item in documents],
                       unseen_table=str(record_inputs['unseen']),
                       unseen_binding=str(record_inputs['binding'])))


def test_the_binding_report_covers_the_whole_seen_record(bound):
    report = bound.binder.collect(**bound.arguments)
    assert len(report['runs']) == 40 and len(report['attempts']) == 3
    assert sorted(item['role'] for item in report['attempts']) == ['seen_aug', 'seen_cyl',
                                                                   'seen_simple']
    assert all(item['full_attempts'] == 1 for item in report['attempts'])
    assert all(item['inventory']['sha256'] and item['probe_receipt']['sha256'] and
               item['ledger']['sha256'] and item['seen_split']['sha256']
               for item in report['attempts'])
    assert sum(item['role'] == 'released_seen' for item in report['runs']) == 10
    assert [item['profile'] for item in report['results']].count('PAIRS_SEEN_V1') == 3
    assert [item['profile'] for item in report['results']].count('TABLE_SEEN_V1') == 1
    assert len(report['documents']) == 3 and len(report['released_checkpoint']['sha256']) == 64
    assert report['unseen']['binding_report']['path'] == str(bound.paths['binding'])
    assert report['audit']['sha256'] and len(report['git_HEAD']) == 40


def test_the_report_is_written_exclusively_and_verified_by_check_record(bound):
    checker = load_asset('check_record')
    arguments = sum(([('--' + key.replace('_', '-')), *([value] if isinstance(value, str)
                                                        else value)]
                     for key, value in bound.arguments.items()), [])
    bound.binder.main(arguments + ['--out', str(bound.reports)])
    reports = list(bound.reports.glob('binding_report_*.json'))
    assert len(reports) == 1
    checker.main([str(bound.reports)])
    modified = Path(bound.built.paths[('seen_cyl', 8)][0]) / 'metrics_yaw.json'
    modified.write_text(modified.read_text() + '\n')
    with pytest.raises((ValueError, OSError)):
        checker.main([str(bound.reports)])


def replace_ledger(bound, role, rows):
    path = bound.built.attempts[role].parent / 'cumulative_hours.json'
    ledger = json.loads(path.read_text())
    ledger['attempts'] = [dict(ledger['attempts'][0], attempt='attempt_%d' % index, hours=1.)
                          for index in range(rows)]
    ledger['total_hours'] = float(rows)
    path.unlink()
    path.write_text(json.dumps(ledger))


def bind_released(bound):
    run = Path(bound.built.paths[('released_seen', 8)][0])
    fields = bound.built.read(run / 'eval_manifest.json')
    donor = Path(bound.built.paths[('seen_simple', 8)][0])
    fields['mutable_inputs']['train_manifest'] = bound.built.read(
        donor / 'eval_manifest.json')['mutable_inputs']['train_manifest']
    bound.built.rebind(run, manifest=fields)


def relink(bound):
    run = Path(bound.built.paths[('seen_cyl', 8)][0])
    fields = bound.built.read(run / 'eval_manifest.json')
    other = bound.built.attempts['seen_aug'] / 'train_manifest.json'
    fields['mutable_inputs']['train_manifest'] = dict(
        path=str(other), sha256=p.sha256_file(other))
    bound.built.rebind(run, manifest=fields)


FORGERIES = {
    'retried_twice': (lambda b: replace_ledger(b, 'seen_simple', 3),
                      'more than one retry recorded'),
    'released_training_binding': (bind_released, 'released row has no training provenance'),
    'foreign_training_linkage': (relink, 'training linkage'),
    'missing_run': (lambda b: b.arguments.__setitem__('runs', b.arguments['runs'][:39]),
                    'the forty evaluation runs'),
    'uncited_document': (lambda b: Path(b.arguments['rendered'][0]).write_text('nothing'),
                         'does not cite every canonical digest'),
}


@pytest.mark.parametrize('forgery', sorted(FORGERIES))
def test_the_binder_refuses_a_forged_record(bound, forgery):
    mutate, message = FORGERIES[forgery]
    mutate(bound)
    with pytest.raises(ValueError, match=message):
        bound.binder.collect(**bound.arguments)


def table_rows(text):
    """The tabular's data rows, cell by cell."""
    rows = [[cell.strip() for cell in line[:-3].split('&')] for line in text.split('\n')
            if line.endswith(' \\\\') and '&' in line and not line.startswith('\\')
            and 'multicolumn' not in line]
    return [row for row in rows if row[0] != 'Method']


def latex_file(record_inputs):
    out = record_inputs['out'].with_suffix('.tex')
    load_asset('make_latex').main(argv(record_inputs, out=out))
    return out


def test_the_latex_table_is_the_paper_layout_and_byte_stable(record_inputs):
    out = latex_file(record_inputs)
    first = out.read_text()
    latex_file(record_inputs)
    assert first == out.read_text() and first.endswith('\\end{table}\n')
    assert '\\begin{tabular}{lcccccc}' in first and '\\label{tab:seen_unseen}' in first
    assert first.split('\n')[0].startswith('% generated by make_latex.py at ')
    assert first.index('$K = 1$') < first.index('$K = 8$')
    assert 'T60 (\\%)' in first and '%\\' not in first
    assert [row[0] for row in table_rows(first)] == [
        name for shot in (1, 8) for name in
        ['xRIR', 'YawAug-xRIR', 'CylindricalViT',
         'xRIR (released seen ckpt, $K = {}$)'.format(shot)]]
    md = load_asset('make_results_md')
    _, record, _ = md.arguments(argv(record_inputs))
    for receipt in record['receipts']:
        assert receipt['sha256'] in first
    assert record['receipts'][-1]['binding_report']['sha256'] in first


def test_the_latex_numbers_are_the_rounded_canonical_means(record_inputs):
    latex, md = load_asset('make_latex'), load_asset('make_results_md')
    _, record, _ = md.arguments(argv(record_inputs))
    rows = iter(table_rows(latex_file(record_inputs).read_text()))
    seen = {(row['role'], row['num_shot']): row for row in record['table']['rows']}
    unseen = {(row['role'], row['num_shot']): row for row in record['unseen']['rows']}
    plain = lambda cell: cell[len('\\textbf{'):-1] if cell.startswith('\\textbf{') else cell
    for shot in (1, 8):
        for seen_role, unseen_role, name in latex.METHODS:
            row = next(rows)
            assert row[0] == name
            for offset, metric in enumerate(latex.ACOUSTIC):
                for column, table, role in ((1, seen, seen_role), (4, unseen, unseen_role)):
                    mean = table[(role, shot)]['metrics'][metric]['mean']
                    assert plain(row[column + offset]) == latex.number(metric, mean)
                    assert plain(row[column + offset]) != str(mean)  # rounded, never raw
        row = next(rows)
        assert row[4:] == ['--'] * 3  # the released row has no unseen evaluation
        for offset, metric in enumerate(latex.ACOUSTIC):
            mean = seen[(md.REFERENCE_ROLE, shot)]['metrics'][metric]['mean']
            assert row[1 + offset] == latex.number(metric, mean)  # a reference is never bold


def test_the_smallest_mean_per_column_and_k_is_bold(record_inputs):
    latex, md = load_asset('make_latex'), load_asset('make_results_md')
    _, record, _ = md.arguments(argv(record_inputs))
    rows = [row for row in table_rows(latex_file(record_inputs).read_text())
            if not row[0].startswith('xRIR (released')]
    for index, shot in enumerate((1, 8)):
        block = rows[3 * index:3 * index + 3]
        for side, column, table in (('table', 1, record['table']), ('unseen', 4, record['unseen'])):
            values = {(row['role'], row['num_shot']): row for row in table['rows']}
            for offset, metric in enumerate(latex.ACOUSTIC):
                roles = [item[0 if side == 'table' else 1] for item in latex.METHODS]
                means = [values[(role, shot)]['metrics'][metric]['mean'] for role in roles]
                bold = [row[column + offset] for row in block
                        if row[column + offset].startswith('\\textbf{')]
                # The exp_04 fixture's three unseen arms share one mean: a tie bolds nothing.
                unique = means.count(min(means)) == 1
                assert bold == (['\\textbf{' + latex.number(metric, min(means)) + '}']
                                if unique else [])
                assert unique is (side == 'table')


def test_a_tie_leaves_the_column_unbolded(record_inputs):
    latex = load_asset('make_latex')
    def tie(data):
        rows = {(row['role'], row['num_shot']): row for row in data['rows']}
        rows[('seen_cyl', 8)]['metrics']['C50']['mean'] = \
            rows[('seen_simple', 8)]['metrics']['C50']['mean']
    edit(record_inputs['table'], tie)
    rows = [row for row in table_rows(latex_file(record_inputs).read_text())
            if not row[0].startswith('xRIR (released')][3:]
    assert not any(row[2].startswith('\\textbf{') for row in rows)  # seen C50, K = 8
    assert any(row[1].startswith('\\textbf{') for row in rows)      # seen EDT is untouched


def test_the_latex_refuses_what_the_markdown_refuses(record_inputs):
    MUTATIONS['exploratory'](record_inputs)
    with pytest.raises(ValueError, match='exploratory JSON refused'):
        latex_file(record_inputs)
    assert not record_inputs['out'].with_suffix('.tex').exists()


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
