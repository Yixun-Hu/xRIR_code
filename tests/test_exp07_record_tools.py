"""The exp_07 record generators run on real producer outputs over synthetic runs."""
import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from exp07_fixture import exp07_fixture  # noqa: F401  (fixture)
from test_exp07_parity import SUITE, case as junit_case, passing
from test_paired_compare import admission_fixture  # noqa: F401  (table_fixture needs it)
from test_results_table import table_fixture  # noqa: F401  (exp_04's unseen table)
from tools import exp07_calibration as calibration, exp07_parity as parity
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


def junit_xml(cases):
    """The JUnit XML pytest writes for these (node id, outcome) pairs."""
    return SUITE.format(n=len(cases), cases=''.join(junit_case(*item) for item in cases))


def parity_receipt(path, head, **changes):
    """A GPU parity receipt shaped exactly as tools/exp07_parity.py writes one."""
    log, junit = path.with_suffix('.log'), path.with_suffix('.xml')
    log.write_text('9 passed, 0 skipped in 1200.00s\n')
    junit.write_text(junit_xml(passing()))
    receipt = dict(schema_version=1, passed=True, pytest_exit=0, n_tests=len(parity.TESTS),
                   tests={node: 'passed' for node in parity.TESTS}, git_head=head,
                   reviewed_commit=head, cuda_device='NVIDIA RTX A6000', gpu='1',
                   allow_dirty_used=False, python='/envs/xRIR/bin/python',
                   log=dict(path=str(log), sha256=p.sha256_file(log)),
                   junit=dict(path=str(junit), sha256=p.sha256_file(junit)))
    receipt.update(changes)
    if path.exists():
        path.unlink()
    p.write_manifest(path, receipt)
    return receipt


def calibration_evidence(built, path, head, monkeypatch):
    """The real calibration producer over the fixture's five released K = 8 runs."""
    runs = list(built.paths[('released_seen', 8)])
    monkeypatch.setattr(calibration, 'closure_pins', lambda commit: dict(
        closures=dict(built.pins['closures']), checkpoints={}))
    monkeypatch.setattr(calibration, 'HISTORICAL', {name: cell['mean'] for name, cell
                                                    in calibration.measure(runs).items()})
    result, admitted = calibration.build_calibration(runs, head, profile=built.profile,
                                                     producer=built.producer)
    calibration.write_outputs(result, admitted, str(path), command=['exp07_calibration.py'])
    return result


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
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(binder.ROOT),
                                   text=True).strip()
    audit = tmp_path / 'alignment_audit_seen.json'
    audit.write_text(json.dumps(dict(pairs=768, changed_delays=0, fraction=0.0,
        cohort_sha256='a' * 64, W=512, schema_version=1, passed=True,
        args=dict(seed=0, n_batches=3, batch_size=32, protocol='seen', num_workers=12,
                  loader_batches=binder.SEEN_BATCHES, W=512, data_root='/data',
                  PYTHONHASHSEED='0', git_head=head),
        env=dict(python='3.8.0', torch='2.0.1', cuda='11.7', hostname='host'))))
    evidence = {'gpu_parity': tmp_path / 'gpu_parity.receipt.json',
                'calibration': tmp_path / 'CALIBRATION_SEEN_V1.json'}
    parity_receipt(evidence['gpu_parity'], head)
    monkeypatch.setattr(binder, 'calibration_producer', lambda: built.producer)
    calibration_evidence(built, evidence['calibration'], head, monkeypatch)
    documents = [record_inputs['out']]
    load_asset('make_results_md').main(argv(record_inputs))
    for suffix, module in (('.html', 'make_results_html'), ('.tex', 'make_latex')):
        documents.append(record_inputs['out'].with_suffix(suffix))
        load_asset(module).main(argv(record_inputs, out=documents[-1]))
    reports = tmp_path / 'reports'
    reports.mkdir()
    return SimpleNamespace(binder=binder, built=built, paths=record_inputs, reports=reports,
        evidence=evidence,
        arguments=dict(runs=list(built.directories), audit=str(audit),
                       evidence=[name + '=' + str(path) for name, path in sorted(evidence.items())],
                       attempt=[str(built.attempts[role]) for role in built.pins['checkpoints']],
                       results=[str(record_inputs['table'])] + [str(item) for item in
                                                                record_inputs['pairs']],
                       rendered=[str(item) for item in documents],
                       unseen_table=str(record_inputs['unseen']),
                       unseen_binding=str(record_inputs['binding'])))


def test_a_documented_setup_failure_is_bound_without_a_log(bound):
    """A run that aborted before the child was spawned has no log to hash; say so."""
    before = bound.binder.collect(**bound.arguments)
    setup_failure(bound)
    # The new ledger row is a real change to an artefact the products declare.
    repair_input(bound, attempt_root(bound, 'seen_cyl') / 'cumulative_hours.json')
    regenerate_documents(bound)
    report = bound.binder.collect(**bound.arguments)
    assert report != before
    rows = {item['role']: {row['attempt']: row for row in item['other_attempts']}
            for item in report['attempts']}
    failed = rows['seen_cyl']['attempt_setup_ABORTED_setup_failed']
    assert failed['state'] == 'aborted' and failed['setup_failure'] is True
    assert failed['reason'] == 'setup_failed'
    assert [item['present'] for item in failed['logs']] == [False]
    aborted = rows['seen_simple']['attempt_first_ABORTED_slow']
    assert aborted['setup_failure'] is False and aborted['reason'] == 'guard_epoch_one'
    # The original log was renamed by the abort, so only the renamed one is on disk.
    assert [item['present'] for item in aborted['logs']] == [False, True]
    assert aborted['logs'][1]['sha256'] and aborted['logs'][1]['path'].endswith('_slow.log')
    probe = rows['seen_simple']['_probe_t_arm']
    assert probe['state'] == 'certified' and probe['logs'][0]['present'] is True


def test_every_probe_attempt_is_linked_to_exactly_one_receipt(bound):
    report = bound.binder.collect(**bound.arguments)
    for attempt in report['attempts']:
        probes = [item for item in attempt['other_attempts'] if item['mode'] == 'probe']
        assert len(attempt['probe_linkage']) == len(probes) == 1
        for path, receipt in attempt['probe_linkage'].items():
            assert Path(path).name == '_probe_t_arm' and Path(receipt).name.startswith('_probe_')


def test_the_report_does_not_serialize_the_working_dependency_maps(bound):
    """One real run declares its whole data inventory: those maps stay out of the report."""
    report = bound.binder.collect(**bound.arguments)
    assert all('bound' not in item for item in report['runs'] + report['attempts'])
    assert json.dumps(report, allow_nan=False)


def test_the_binding_report_covers_the_whole_seen_record(bound):
    report = bound.binder.collect(**bound.arguments)
    assert len(report['runs']) == 40 and len(report['attempts']) == 3
    assert sorted(item['role'] for item in report['attempts']) == ['seen_aug', 'seen_cyl',
                                                                   'seen_simple']
    counted = {item['role']: item['full_attempts'] for item in report['attempts']}
    assert counted == dict(seen_simple=2, seen_cyl=1, seen_aug=1)
    # Every attempt the ledgers list is bound, aborted and probe runs included.
    others = {item['role']: sorted((row['attempt'], row['mode'])
                                   for row in item['other_attempts'])
              for item in report['attempts']}
    assert others['seen_cyl'] == others['seen_aug'] == [('_probe_t_arm', 'probe')]
    assert others['seen_simple'] == [('_probe_t_arm', 'probe'),
                                     ('attempt_first_ABORTED_slow', 'full')]
    assert all(row['files'] for item in report['attempts'] for row in item['other_attempts'])
    assert all(item['probe_receipts'] for item in report['attempts'])
    assert report['audit']['protocol'] == 'seen' and report['audit']['cohort_sha256']
    assert set(report['evidence']) == set(bound.binder.EVIDENCE)
    assert report['evidence']['calibration']['role'] == 'released_seen'
    assert sorted(report['evidence']['calibration']['metrics']) == ['C50', 'EDT', 'T60']
    assert len(report['evidence']['calibration']['runs']) == 5
    assert report['evidence']['gpu_parity']['tests'] == {
        node: 'passed' for node in parity.TESTS}
    assert report['evidence']['gpu_parity']['log']['sha256']
    assert report['evidence']['gpu_parity']['junit']['sha256']
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


def edit_json(path, mutate):
    data = json.loads(Path(path).read_text())
    mutate(data)
    Path(path).write_text(json.dumps(data))


def restamp(bound, product='table'):
    """Repair the sidecar's digest of the canonical JSON after editing both of them."""
    md = load_asset('make_results_md')
    path = Path(bound.paths[product] if product != 'pairs' else bound.paths['pairs'][0])
    side = json.loads(sidecar(path).read_text())
    side['outputs'][str(path.resolve())] = md.sha(path.read_bytes())
    sidecar(path).write_text(json.dumps(side))


def edit_both(bound, mutate, product='table'):
    """Edit a product's inputs in the JSON and its sidecar, keeping them consistent."""
    path = Path(bound.paths[product] if product != 'pairs' else bound.paths['pairs'][0])
    data, side = json.loads(path.read_text()), json.loads(sidecar(path).read_text())
    mutate(data, side)
    side['inputs'] = data['inputs']
    if 'run_flags' in data:
        data['run_flags'] = {key: value for key, value in data['run_flags'].items()
                             if key in side['run_flags']}
        side['run_flags'] = data['run_flags']
    path.write_text(json.dumps(data))
    sidecar(path).write_text(json.dumps(side))
    restamp(bound, product)


def repair_run_digests(bound, run):
    """After a run's bytes change, restate the digests every product declares for it."""
    for product in ('table', 'pairs'):
        def mutate(data, side):
            for key in list(data['inputs']):
                if Path(key).parent == Path(run):
                    data['inputs'][key] = p.sha256_file(key)
        edit_both(bound, mutate, product)


def regenerate_documents(bound):
    """Rewrite the three rendered documents after a product's digest legitimately changed."""
    load_asset('make_results_md').main(argv(bound.paths))
    for suffix, module in (('.html', 'make_results_html'), ('.tex', 'make_latex')):
        load_asset(module).main(argv(bound.paths, out=bound.paths['out'].with_suffix(suffix)))


def repair_input(bound, path):
    """Restate one declared digest in both products after a legitimate change."""
    for product in ('table', 'pairs'):
        def mutate(data, side):
            if str(path) in data['inputs']:
                data['inputs'][str(path)] = p.sha256_file(str(path))
        edit_both(bound, mutate, product)


def drop_run(bound):
    """Remove one whole evaluation run from the table's declared provenance."""
    run = Path(bound.arguments['runs'][0])
    def mutate(data, side):
        side['run_flags'].pop(str(run))
        data.get('run_flags', {}).pop(str(run), None)
        data.get('contracts', {}).pop(str(run), None)
        for key in list(data['inputs']):
            if Path(key).parent == run:
                data['inputs'].pop(key)
    edit_both(bound, mutate)


def add_dependency(bound):
    edit_both(bound, lambda data, side: data['inputs'].__setitem__(
        '/nonexistent/exp07_review_dependency.json', 'f' * 64))


def substitute_reference_manifest(bound):
    """Restate the digest of the reference manifest the first run was evaluated under."""
    run = Path(bound.arguments['runs'][0])
    reference = bound.built.read(run / 'eval_manifest.json')['manifest_path']
    edit_both(bound, lambda data, side: data['inputs'].__setitem__(reference, 'f' * 64))


def unseen_output_metas(bound):
    """Both output metas say unseen while the manifest they echo says seen."""
    run = Path(bound.arguments['runs'][0])
    sample, metrics = (bound.built.read(run / name) for name in
                       ('per_sample_yaw.json', 'metrics_yaw.json'))
    sample['meta']['split'] = metrics['meta']['split'] = 'unseen'
    bound.built.rebind(run, sample=sample, metrics=metrics)
    repair_run_digests(bound, run)


def drop_input(bound, name):
    """Remove one declared input from BOTH the canonical JSON and its sidecar."""
    def mutate(data, side):
        victim = next(item for item in data['inputs'] if Path(item).name == name)
        del data['inputs'][victim]
    edit_both(bound, mutate)


def substitute_input(bound, name, digest):
    def mutate(data, side):
        victim = next(item for item in data['inputs'] if Path(item).name == name)
        data['inputs'][victim] = digest
    edit_both(bound, mutate)


def foreign_approval(bound, replacement='/nonexistent/approved.json'):
    """Point the approval identity at another path, consistently in JSON and sidecar."""
    def mutate(data, side):
        old = side['approved_digests']['path']
        data['inputs'][replacement] = data['inputs'].pop(old)
        side['approved_digests'] = dict(side['approved_digests'], path=replacement)
    edit_both(bound, mutate)


def attempt_root(bound, role='seen_simple'):
    return bound.built.attempts[role].parent


def aborted_attempt(bound):
    return attempt_root(bound) / 'attempt_first_ABORTED_slow'


def rewrite_external_log(bound, which):
    """Change a log an attempt's own records point at, outside every attempt directory."""
    if which == 'aborted':
        record = json.loads((aborted_attempt(bound) / 'abort.json').read_text())
        path = Path(record['log']['aborted'])
    else:
        record = json.loads((attempt_root(bound) / '_probe_t_arm/completion.json').read_text())
        path = Path(record['log']['path'])
    assert path.is_file()
    path.write_text('rewritten evidence\n')


def rewrite_parity_xml(bound, text):
    """Replace the JUnit XML the parity receipt names, restating the digest it records."""
    path = bound.evidence['gpu_parity']
    data = json.loads(path.read_text())
    junit = Path(data['junit']['path'])
    junit.write_text(text)
    data['junit']['sha256'] = p.sha256_file(junit)
    path.unlink()
    p.write_manifest(path, data)


def extra_receipt(bound, name, mutate, role='seen_simple'):
    """A second probe receipt in the arm root, not the one the training manifest binds."""
    root = attempt_root(bound, role)
    data = json.loads((root / ('_probe_t_' + role + '.json')).read_text())
    mutate(data)
    p.write_manifest(root / name, data)


def add_ledger_row(bound, name, mode='full', build=None, role='seen_cyl'):
    """Register one more attempt in the arm's ledger, optionally creating its directory."""
    path = attempt_root(bound, role) / 'cumulative_hours.json'
    ledger = json.loads(path.read_text())
    ledger['attempts'].append(dict(attempt=name, hours=.2, mode=mode))
    ledger['total_hours'] = sum(row['hours'] for row in ledger['attempts'])
    path.unlink()
    path.write_text(json.dumps(ledger))
    if build is not None:
        build(attempt_root(bound, role) / name)


def setup_failure(bound, spawned=False, role='seen_cyl'):
    """The one form in which an aborted attempt legitimately has no log on disk."""
    def build(directory):
        directory.mkdir()
        p.write_completion(directory / 'abort.json', dict(
            reason='setup_failed', wall_hours=.0,
            exception_message='GPU needs no other processes and >= 40 GiB free',
            log=dict(original=str(attempt_root(bound, role) / 'never_created.log'), aborted=None)))
        if spawned:
            p.write_manifest(directory / 'execution.json',
                             dict(train_manifest_sha256='e' * 64, child_pgid=99))
    add_ledger_row(bound, 'attempt_setup_ABORTED_setup_failed', 'full', build, role)


def misnamed_setup_failure(bound, role='seen_cyl'):
    """The documented reason and log shape in a directory the abort did not name."""
    def build(directory):
        directory.mkdir()
        p.write_completion(directory / 'abort.json', dict(
            reason='setup_failed', wall_hours=.0, exception_message='GPU is not free',
            log=dict(original=str(attempt_root(bound, role) / 'never_created.log'),
                     aborted=None)))
    add_ledger_row(bound, 'attempt_setup_ABORTED_child_failed', 'full', build, role)


def stripped_abort(bound):
    """Reduce a guard_epoch_one abort to a logless one without the setup-failure shape."""
    directory = aborted_attempt(bound)
    record = json.loads((directory / 'abort.json').read_text())
    record.pop('log')
    p.write_completion(directory / 'abort.json', record)
    (directory / 'execution.json').unlink()


def certified_probe_log(bound, role='seen_simple'):
    completion = json.loads((attempt_root(bound, role) / '_probe_t_arm'
                             / 'completion.json').read_text())
    return Path(completion['log']['path'])


def aborted_probe(bound, role='seen_cyl'):
    """A probe whose child failed: tools.exp07_launcher.run_probe writes no receipt.

    The receipt is written only after ``execute_attempt`` returns, so this attempt's
    whole evidence is its abort record and the log the abort renamed.
    """
    def build(directory):
        directory.mkdir()
        renamed = attempt_root(bound, role) / 'probe_failed_train_ABORTED_child_failed.log'
        renamed.write_text('the probe child exited nonzero\n')
        p.write_manifest(directory / 'execution.json',
                         dict(train_manifest_sha256='f' * 64, child_pgid=1))
        p.write_completion(directory / 'abort.json', dict(
            reason='child_failed', wall_hours=.2, exception_message='child exited 1',
            log=dict(original=str(attempt_root(bound, role) / 'probe_failed_train.log'),
                     aborted=str(renamed))))
    add_ledger_row(bound, '_probe_failed_arm_ABORTED_child_failed', 'probe', build, role)


def test_a_legitimately_aborted_probe_is_retained_not_refused(bound):
    """Publishing the failure history is the point; only a COMPLETED probe has a receipt."""
    before = bound.binder.collect(**bound.arguments)
    aborted_probe(bound)
    repair_input(bound, attempt_root(bound, 'seen_cyl') / 'cumulative_hours.json')
    regenerate_documents(bound)
    report = bound.binder.collect(**bound.arguments)
    assert report != before
    arm = next(item for item in report['attempts'] if item['role'] == 'seen_cyl')
    rows = {row['attempt']: row for row in arm['other_attempts']}
    failed = rows['_probe_failed_arm_ABORTED_child_failed']
    assert failed['mode'] == 'probe' and failed['state'] == 'aborted'
    assert failed['reason'] == 'child_failed' and failed['setup_failure'] is False
    assert [item['present'] for item in failed['logs']] == [False, True]
    assert failed['logs'][1]['sha256'] and failed['files']
    # The one probe that completed is still the one the single receipt names.
    assert len(arm['probe_linkage']) == 1
    assert Path(next(iter(arm['probe_linkage']))).name == '_probe_t_arm'


def second_probe(bound, role='seen_cyl'):
    """A complete second probe attempt: certified, logged, and named by no receipt."""
    def build(directory):
        directory.mkdir()
        log = attempt_root(bound, role) / 'probe_second.log'
        log.write_text('a second synthetic probe\n')
        p.write_manifest(directory / 'train_manifest.json', dict(mode='probe'))
        p.write_completion(directory / 'completion.json', dict(
            metrics=dict(probe=dict(T_epoch=8400.)),
            log=dict(path=str(log), sha256=p.sha256_file(log))))
    return build


def orphan_ledger_row(bound, role='seen_cyl'):
    """Add a ledger row for an attempt directory that does not exist."""
    path = bound.built.attempts[role].parent / 'cumulative_hours.json'
    ledger = json.loads(path.read_text())
    ledger['attempts'].append(dict(attempt='attempt_vanished', hours=.5, mode='probe'))
    ledger['total_hours'] = sum(row['hours'] for row in ledger['attempts'])
    path.unlink()
    path.write_text(json.dumps(ledger))


FORGERIES = {
    'retried_twice': (lambda b: replace_ledger(b, 'seen_simple', 3),
                      'more than one retry recorded'),
    'released_training_binding': (bind_released, 'released row has no training provenance'),
    'foreign_training_linkage': (relink, 'training linkage'),
    'missing_run': (lambda b: b.arguments.__setitem__('runs', b.arguments['runs'][:39]),
                    'the forty evaluation runs'),
    'uncited_document': (lambda b: Path(b.arguments['rendered'][0]).write_text('nothing'),
                         'does not cite every canonical digest'),
    # Blocker 2: incomplete products and inputs that are not the bound evidence.
    'empty_table': (lambda b: edit(b.paths['table'], lambda d: d.update(rows=[])),
                    'incomplete seen table coverage'),
    'nonfinal_pairs': (lambda b: edit(b.paths['pairs'][0], lambda d: d.update(
        final=False, reconverge_required=['EDT K=8'])), 'pairing is not final'),
    'unconverged_cell': (lambda b: edit(b.paths['pairs'][1], lambda d: d['cells'][0][
        'statistics']['absolute'].update(reconverge_required=True)), 'larger n_boot'),
    'missing_run_input': (lambda b: drop_input(b, 'metrics_yaw.json'),
                          'incomplete run coverage'),
    'substituted_inventory_digest': (lambda b: substitute_input(
        b, 'train_inventory.json', 'f' * 64), 'input differs from the bound artefact'),
    'foreign_approval_path': (foreign_approval, 'result approval identity or pins'),
    # Blocker 3: the attempt history, the audit and the required evidence.
    'changed_abort_json': (lambda b: edit_json(
        b.built.attempts['seen_simple'].parent / 'attempt_first_ABORTED_slow/abort.json',
        lambda d: d.update(reason='rewritten')), None),
    'missing_ledger_attempt': (orphan_ledger_row, 'the ledger names a missing attempt'),
    'unseen_audit': (lambda b: edit_json(b.arguments['audit'],
                                         lambda d: d['args'].update(protocol='unseen')),
                     'not a seen-protocol audit'),
    'failed_audit': (lambda b: edit_json(b.arguments['audit'],
                                         lambda d: d.update(passed=False, changed_delays=7)),
                     'the audit did not pass'),
    'unseen_audit_loader': (lambda b: edit_json(b.arguments['audit'],
                                                lambda d: d['args'].update(loader_batches=9261)),
                            'the audit cohort is not the seen training loader'),
    'audit_without_cohort': (lambda b: edit_json(b.arguments['audit'],
                                                 lambda d: d.update(cohort_sha256=None)),
                             'the audit has no cohort digest'),
    'missing_evidence': (lambda b: b.arguments.__setitem__(
        'evidence', b.arguments['evidence'][:1]), 'every evidence artefact is required'),
    'unregistered_evidence': (lambda b: b.arguments['evidence'].append('extra=/dev/null'),
                              'invalid or duplicate --evidence'),
    'empty_parity_log': (lambda b: b.evidence['gpu_parity'].write_text(''),
                         'empty evidence artefact'),
    'calibration_outside_tolerance': (lambda b: edit_json(
        b.evidence['calibration'],
        lambda d: d['metrics']['C50'].update(mean=d['metrics']['C50']['historical'] * 2 + 1)),
        'calibration acceptance'),
    'calibration_not_the_released_row': (lambda b: edit_json(
        b.evidence['calibration'], lambda d: d.update(role='seen_simple')),
        'calibration identity'),
    # Blocker 2: a finite, self-consistent summary unrelated to any evaluation, a summary
    # that covers the wrong runs, and a bare log where a parity receipt is required.
    'calibration_fabricated_summary': (lambda b: edit_json(
        b.evidence['calibration'], lambda d: [cell.update(mean=1000000., sd=1000000.)
                                              for cell in d['metrics'].values()]),
        'differ from the bound runs'),
    'calibration_infinite_sd': (lambda b: edit_json(
        b.evidence['calibration'], lambda d: [cell.update(mean=1000000., sd=float('inf'))
                                              for cell in d['metrics'].values()]),
        'differ from the bound runs'),
    'calibration_missing_a_run': (lambda b: edit_json(
        b.evidence['calibration'],
        lambda d: d['run_flags'].pop(sorted(d['run_flags'])[0])), 'calibration run coverage'),
    'calibration_without_a_sidecar': (
        lambda b: Path(str(b.evidence['calibration']) + '.provenance.json').unlink(),
        'no provenance sidecar'),
    'parity_case_failed': (lambda b: edit_json(b.evidence['gpu_parity'], lambda d: d['tests']
                                               .update({sorted(d['tests'])[0]: 'failed'})),
                           'parity cases did not pass'),
    'parity_case_skipped': (lambda b: edit_json(b.evidence['gpu_parity'], lambda d: d['tests']
                                                .update({sorted(d['tests'])[-1]: 'skipped'})),
                            'parity cases did not pass'),
    'parity_case_missing': (lambda b: edit_json(
        b.evidence['gpu_parity'], lambda d: d['tests'].pop(sorted(d['tests'])[0])),
        'parity test coverage'),
    'parity_nonzero_exit': (lambda b: edit_json(b.evidence['gpu_parity'],
                                                lambda d: d.update(pytest_exit=1)),
                            'parity pytest exit'),
    'parity_dirty_checkout': (lambda b: edit_json(b.evidence['gpu_parity'],
                                                  lambda d: d.update(allow_dirty_used=True)),
                              'outside a clean checkout'),
    'parity_log_rewritten': (lambda b: b.evidence['gpu_parity'].with_suffix('.log')
                             .write_text('9 failed, 0 passed\n'), 'digest mismatch'),
    # Round-7 blocker 2: the receipt is checked against pytest's own XML, not only
    # against itself, so a receipt that claims nine passes over an XML that records
    # something else -- or over no test cases at all -- is refused.
    'parity_xml_case_failed': (lambda b: rewrite_parity_xml(b, junit_xml(
        [(node, 'failed' if index == 0 else 'passed')
         for index, node in enumerate(parity.TESTS)])),
        'JUnit XML does not record the nine registered passes'),
    'parity_xml_case_skipped': (lambda b: rewrite_parity_xml(b, junit_xml(
        [(node, 'skipped' if index == 8 else 'passed')
         for index, node in enumerate(parity.TESTS)])),
        'JUnit XML does not record the nine registered passes'),
    'parity_xml_without_cases': (lambda b: rewrite_parity_xml(
        b, '<testsuites><testsuite tests="9"/></testsuites>'),
        'JUnit XML does not record the nine registered passes'),
    'parity_xml_with_an_unregistered_case': (lambda b: rewrite_parity_xml(b, junit_xml(
        passing() + [('tests/test_other.py::test_x', 'passed')])),
        'JUnit XML does not record the nine registered passes'),
    'parity_xml_unparseable': (lambda b: rewrite_parity_xml(b, 'not pytest XML at all'),
                               'not parseable'),
    'parity_bare_log': (lambda b: b.arguments.__setitem__('evidence', [
        'calibration=' + str(b.evidence['calibration']),
        'gpu_parity=' + str(b.evidence['gpu_parity'].with_suffix('.log'))]),
        'not canonical JSON'),
    # Blocker 1: whole-run omissions, missing training dependencies and inputs that are
    # not the artefacts this report binds (the round-6 reviewer's four reproductions).
    'whole_run_removed_from_a_product': (drop_run, 'does not declare exactly its runs'),
    'required_training_inventory_removed': (
        lambda b: drop_input(b, 'train_inventory.json'), 'missing training dependency'),
    'reference_manifest_digest_substituted': (substitute_reference_manifest,
                                              'input differs from the bound artefact'),
    'nonexistent_dependency_added': (add_dependency,
                                     'names an artefact this report does not bind'),
    # Blocker 4: the binder's own split-agreement check.
    'unseen_output_metas_under_a_seen_manifest': (unseen_output_metas, 'output split identity'),
    # Blocker 3: every ledger-listed attempt's terminal state and external evidence.
    'aborted_attempt_without_abort_json': (
        lambda b: (aborted_attempt(b) / 'abort.json').unlink(), 'no terminal state'),
    'aborted_attempt_log_vanished': (
        lambda b: Path(json.loads((aborted_attempt(b) / 'abort.json').read_text())
                       ['log']['aborted']).unlink(), 'must bind its log'),
    'setup_failure_form_with_a_spawned_child': (
        lambda b: setup_failure(b, spawned=True), 'must bind its log'),
    'probe_attempt_without_a_receipt': (lambda b: add_ledger_row(
        b, '_probe_second_arm', 'probe', second_probe(b)), 'has no receipt'),
    'probe_receipt_naming_another_attempt': (lambda b: extra_receipt(
        b, '_probe_extra_arm.json',
        lambda d: d['probe_attempt'].update(path='/nonexistent/probe')),
        'names an attempt this arm does not list'),
    'probe_receipt_with_a_stale_digest': (lambda b: extra_receipt(
        b, '_probe_extra_arm.json',
        lambda d: d['probe_attempt'].update(completion_sha256='f' * 64)),
        'probe receipt completion digest'),
    'probe_attempt_external_log': (lambda b: rewrite_external_log(b, 'probe'),
                                   'digest mismatch'),
    # Round-7 blocker 3: a certified attempt's log is mandatory evidence that it ran, the
    # logless abort is only ever the documented setup-failure shape, and that shape is
    # the reason, the directory the abort named and the absent renamed log together.
    'certified_probe_log_deleted': (lambda b: certified_probe_log(b).unlink(),
                                    'names a log that is not on disk'),
    'logless_abort_without_the_setup_failure_shape': (stripped_abort, 'must bind its log'),
    'setup_failure_reason_in_another_abort_directory': (misnamed_setup_failure,
                                                        'must bind its log'),
    # Not a refusal: the abort receipt records the path but no digest, so a rewritten
    # aborted log must at least change the report and fail check_record.py.
    'aborted_attempt_external_log': (lambda b: rewrite_external_log(b, 'aborted'), None),
}





@pytest.mark.parametrize('forgery', sorted(name for name in FORGERIES
                                           if FORGERIES[name][1] is not None))
def test_the_binder_refuses_a_forged_record(bound, forgery):
    mutate, message = FORGERIES[forgery]
    mutate(bound)
    with pytest.raises((ValueError, KeyError), match=message):
        bound.binder.collect(**bound.arguments)


@pytest.mark.parametrize('forgery', sorted(name for name in FORGERIES
                                           if FORGERIES[name][1] is None))
def test_a_change_to_any_bound_attempt_file_changes_the_report(bound, forgery):
    """Not a refusal: the report must simply stop matching, so check_record.py fails."""
    before = bound.binder.collect(**bound.arguments)
    FORGERIES[forgery][0](bound)
    assert bound.binder.collect(**bound.arguments) != before


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


def test_every_sd_footnote_names_its_k_and_its_row(record_inputs):
    latex, md = load_asset('make_latex'), load_asset('make_results_md')
    _, record, _ = md.arguments(argv(record_inputs))
    text = latex_file(record_inputs).read_text()
    footnotes = [line for line in text.split('\n') if ' EDT ' in line and ' ms,' in line]
    assert len(footnotes) == 8  # four rows at each of the two K values
    for shot in (1, 8):
        for name in ('xRIR', 'YawAug-xRIR', 'CylindricalViT', 'xRIR (released seen ckpt)'):
            label = '{} ($K = {}$):'.format(name, shot)
            assert sum(line.startswith(label) for line in footnotes) == 1, label
    assert 'EDT in milliseconds' in text
    seen = {(row['role'], row['num_shot']): row for row in record['table']['rows']}
    row = seen[('seen_simple', 8)]
    assert 'seen EDT {} ms'.format(format(row['metrics']['EDT']['sd'], '.3f')) in text


@pytest.mark.parametrize('value,decimals,expected', [
    (0.0, 3, '0.000'), (0.000001, 3, '0.000001'), (1.2e-9, 4, '0.000000001'),
    (12.3456, 3, '12.346'), (0.00049, 3, '0.0005'), (None, 3, '--'),
    # Nit 8: below the fixed-decimal cap the widest fixed form is still zero.
    (1e-13, 3, '1.000e-13'), (4.56e-15, 4, '4.560e-15')])
def test_a_positive_sd_never_prints_as_zero(value, decimals, expected):
    """A genuine zero keeps the table's precision; a positive SD gains digits until visible."""
    latex = load_asset('make_latex')
    assert latex.sd_number(value, decimals) == expected
    assert value in (0.0, None) or float(latex.sd_number(value, decimals)) != 0


def test_a_tiny_but_positive_sd_survives_the_rendered_footnote(record_inputs):
    latex = load_asset('make_latex')
    edit(record_inputs['table'], lambda data: [row['metrics'][name].update(sd=1e-7)
                                               for row in data['rows']
                                               for name in ('EDT', 'C50', 'T60')])
    text = latex_file(record_inputs).read_text()
    footnotes = [line for line in text.split('\n') if ' EDT ' in line and ' ms,' in line]
    for line in footnotes:
        for part in line.split('seen ')[1:]:
            assert '0.000 ms' not in part and ' 0.0000 dB' not in part


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
    # A mutated CELL must be refused even though every aggregate flag still says final.
    'cell_reconverge': lambda paths: edit(paths['pairs'][0], lambda d: d['cells'][0][
        'statistics']['absolute'].update(reconverge_required=True)),
    'cell_unconverged': lambda paths: edit(paths['pairs'][1], lambda d: d['cells'][3][
        'statistics']['relative']['convergence'].update(passed=False)),
    'cell_decision': lambda paths: edit(paths['pairs'][2],
                                        lambda d: d['cells'][2].update(decision_driving=True)),
    'cell_verdict': lambda paths: edit(paths['pairs'][0],
                                       lambda d: d['cells'][1].update(verdict='better')),
    'cell_pairing': lambda paths: edit(paths['pairs'][1], lambda d: d['cells'][4].update(
        pairing=['seen_aug', 'seen_cyl'])),
    'cell_grid': lambda paths: edit(paths['pairs'][2], lambda d: d['cells'][5].update(k=32)),
    'cell_seeds': lambda paths: edit(paths['pairs'][0], lambda d: d['cells'][6].update(
        seed_labels=[42, 43, 44, 45])),
    'cell_quantiles': lambda paths: edit(paths['pairs'][1], lambda d: d['cells'][7].update(
        quantiles=[.05, .95])),
    'cell_statistics': lambda paths: edit(paths['pairs'][2],
                                          lambda d: d['cells'][8]['statistics'].pop('relative')),
    'cell_empty_cohort': lambda paths: edit(paths['pairs'][0], lambda d: d['cells'][9][
        'paired_cohort'].update(n_queries=0)),
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


def protected_outputs(paths):
    """Every input path a generator must refuse to write over, and two spellings of each."""
    names = [paths['table'], paths['unseen'], paths['binding'], paths['pairs'][0],
             sidecar(paths['table']), sidecar(paths['unseen']),
             Path(paths['built'].approved[1]['path'])]
    spelled = [Path(item).parent / '.' / Path(item).name for item in names]
    return names + spelled


@pytest.mark.parametrize('module', ['make_results_md', 'make_results_html', 'make_latex'])
def test_no_generator_can_overwrite_an_input_through_a_hardlink(record_inputs, module):
    """Should-fix 6: resolve() sees through symlinks, not hardlinks; the inode does."""
    generator = load_asset(module)
    for name in ('binding', 'table'):
        original = record_inputs[name]
        before = original.read_bytes()
        alias = original.parent / ('alias_{}_{}.out'.format(module, name))
        os.link(str(original), str(alias))
        with pytest.raises(ValueError, match='output overlaps canonical input'):
            generator.main(argv(record_inputs, out=alias))
        assert original.read_bytes() == before and alias.read_bytes() == before


@pytest.mark.parametrize('module', ['make_results_md', 'make_results_html', 'make_latex'])
def test_no_generator_can_overwrite_any_input(record_inputs, module):
    generator = load_asset(module)
    for target in protected_outputs(record_inputs):
        before = Path(target).read_bytes()
        with pytest.raises(ValueError, match='output overlaps canonical input'):
            generator.main(argv(record_inputs, out=target))
        assert Path(target).read_bytes() == before, target


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


@pytest.mark.parametrize('spelling', [
    ['--evidence', 'gpu_parity=g.log', 'calibration=c.json'],
    ['--evidence', 'gpu_parity=g.log', '--evidence', 'calibration=c.json']])
def test_both_documented_evidence_syntaxes_reach_the_binder(monkeypatch, tmp_path, spelling):
    """Should-fix 7: repeating the flag must accumulate, not replace."""
    binder = load_asset('bind_provenance')
    parsed = {}
    monkeypatch.setattr(binder, 'collect', lambda **kwargs: parsed.update(kwargs) or {})
    monkeypatch.setattr(binder.p, 'write_manifest', lambda *rest: None)
    binder.main(['--runs', 'r', '--attempt', 'a', '--results', 'j', '--rendered', 'doc',
                 '--audit', 'audit', '--unseen-table', 'u', '--unseen-binding', 'ub',
                 '--out', str(tmp_path)] + spelling)
    assert parsed['evidence'] == ['gpu_parity=g.log', 'calibration=c.json']
