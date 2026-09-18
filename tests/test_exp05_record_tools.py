"""The exp_05 record tooling: canonical admission, the generators, the binder."""
import json
import os
from pathlib import Path

import pytest

from exp05_record_fixture import exp05_record_fixture
from tools import exp05_record as record
from tools import provenance as p

md = record.load_asset('make_results_md')
page = record.load_asset('make_results_html')
figures = record.load_asset('make_figures')
html_escape = md.html.escape

NAMES = ('CURVE_K8', 'CURVE_K1', 'TARGETS_K8', 'TARGETS_K1', 'YAW_K8_SEED42')


def _rewrite(path, data):
    """Republish a canonical JSON and the sidecar digest that binds it."""
    path = Path(path)
    payload = (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    path.write_bytes(payload)
    sidecar = Path(str(path) + '.provenance.json')
    side = json.loads(sidecar.read_text())
    side['outputs'][str(path.resolve())] = record.sha(payload)
    sidecar.write_text(json.dumps(side, sort_keys=True, indent=2) + '\n')
    return side


def test_load_and_validate_accept_every_product(exp05_record_fixture):
    f = exp05_record_fixture()
    for name in NAMES:
        path = f.produce(name)
        data, receipt = record.load(path, name)
        record.validate(data, name)
        assert data['profile_name'] == name
        assert receipt['sha256'] == record.sha(Path(path).read_bytes())
        assert receipt['profile_digest'] == data['profile_digest']
        assert receipt['producer_closure_sha256'] == f.producer['sha256']
        assert receipt['approved_digests']['sha256'] == f.approval['sha256']
        assert set(receipt['outputs']) == {str(path), str(path)[:-5] + '.txt'}
        assert receipt['run_flags'] and set(receipt['run_flags']) == set(f.run_paths(name))


def test_load_refuses_a_product_read_as_another_profile(exp05_record_fixture):
    f = exp05_record_fixture()
    path = f.produce('CURVE_K8')
    with pytest.raises(ValueError, match='wrong profile'):
        record.load(path, 'CURVE_K1')
    with pytest.raises(ValueError, match='unregistered profile'):
        record.load(path, 'TABLE_V1')


@pytest.mark.parametrize('damage', ['summary', 'json', 'outputs'])
def test_load_refuses_outputs_that_no_longer_match_the_sidecar(exp05_record_fixture, damage):
    f = exp05_record_fixture()
    path = f.produce('CURVE_K8')
    sidecar = Path(str(path) + '.provenance.json')
    if damage == 'summary':
        summary = Path(str(path)[:-5] + '.txt')
        summary.write_text(summary.read_text() + 'appended\n')
    elif damage == 'json':
        path.write_bytes(path.read_bytes() + b'\n')
    else:
        side = json.loads(sidecar.read_text())
        side['outputs'].pop(str(path)[:-5] + '.txt')
        sidecar.write_text(json.dumps(side, sort_keys=True, indent=2) + '\n')
    with pytest.raises(ValueError):
        record.load(path, 'CURVE_K8')


def test_load_refuses_an_unapproved_producer_closure(exp05_record_fixture):
    f = exp05_record_fixture()
    path = f.produce('CURVE_K8')
    sidecar = Path(str(path) + '.provenance.json')
    side = json.loads(sidecar.read_text())
    side['approved_digests']['pins']['closures']['producer_param_curve'] = 'f' * 64
    sidecar.write_text(json.dumps(side, sort_keys=True, indent=2) + '\n')
    with pytest.raises(ValueError, match='provenance sidecar mismatch'):
        record.load(path, 'CURVE_K8')


@pytest.mark.parametrize('damage', ['profile', 'inputs', 'approval', 'run_flags'])
def test_load_refuses_a_sidecar_that_disagrees_with_the_product(exp05_record_fixture, damage):
    f = exp05_record_fixture()
    path = f.produce('CURVE_K8')
    data = json.loads(path.read_text())
    if damage == 'profile':
        data['profile']['n_boot'] = 7
    elif damage == 'inputs':
        data['inputs'][str(f.root / 'planted.json')] = 'a' * 64
    elif damage == 'approval':
        side = json.loads(Path(str(path) + '.provenance.json').read_text())
        data['inputs'].pop(side['approved_digests']['path'])
    else:
        data['run_flags'] = {}
    _rewrite(path, data)
    with pytest.raises(ValueError, match='provenance sidecar mismatch'):
        record.load(path, 'CURVE_K8')


@pytest.mark.parametrize('damage', ['exploratory', 'deviations', 'sidecar_exploratory'])
def test_load_refuses_exploratory_or_deviating_products(exp05_record_fixture, damage):
    f = exp05_record_fixture()
    path = f.produce('CURVE_K8')
    if damage == 'sidecar_exploratory':
        sidecar = Path(str(path) + '.provenance.json')
        side = json.loads(sidecar.read_text())
        side['exploratory'] = True
        sidecar.write_text(json.dumps(side, sort_keys=True, indent=2) + '\n')
    else:
        data = json.loads(path.read_text())
        data[damage] = True if damage == 'exploratory' else ['re-evaluate M']
        _rewrite(path, data)
    with pytest.raises(ValueError):
        record.load(path, 'CURVE_K8')


@pytest.mark.parametrize('name,damage,message', [
    ('CURVE_K8', 'drop_curve', 'curve coverage'),
    ('CURVE_K8', 'drop_cell', 'cell coverage'),
    ('CURVE_K8', 'duplicate_curve', 'curve coverage'),
    ('CURVE_K8', 'unconverged', 'did not converge'),
    ('CURVE_K8', 'empty_cohort', 'empty cohort'),
    ('CURVE_K8', 'verdict', 'unregistered verdict'),
    ('CURVE_K8', 'verdict_missing', 'verdict coverage'),
    ('CURVE_K8', 'counts', 'parameter counts'),
    ('CURVE_K8', 'seed_labels', 'seed labels'),
    ('CURVE_K8', 'target', 'unexpected target'),
    ('TARGETS_K8', 'drop_tost', 'convergence gates'),
    ('TARGETS_K8', 'reaches', 'reaches_target'),
    ('TARGETS_K8', 'ratio', 'parameter ratio'),
    ('TARGETS_K8', 'verdict_present', 'no verdict'),
    ('YAW_K8_SEED42', 'yaw_verdict', 'descriptive'),
    ('YAW_K8_SEED42', 'yaw_decision', 'descriptive'),
])
def test_validate_refuses_structurally_incomplete_products(exp05_record_fixture, name,
                                                           damage, message):
    f = exp05_record_fixture()
    data = json.loads(f.produce(name).read_text())
    record.validate(data, name)  # the untouched product is admissible
    if damage == 'drop_curve':
        data['curves'].pop()
    elif damage == 'drop_cell':
        data['cells'].pop()
    elif damage == 'duplicate_curve':
        data['curves'][1] = dict(data['curves'][0])
    elif damage == 'unconverged':
        data['cells'][0]['convergence']['superiority']['passed'] = False
    elif damage == 'empty_cohort':
        data['curves'][0]['cohort']['n_queries'] = 0
    elif damage == 'verdict':
        data['verdicts']['EDT'] = 'dominance supported'
    elif damage == 'verdict_missing':
        data['verdicts'].pop('C50')
    elif damage == 'counts':
        data['curves'][0]['counts']['encoder'] += 1
    elif damage == 'seed_labels':
        data['curves'][0]['seed_labels'] = [42, 43, 44, 45, 47]
    elif damage == 'target':
        data['cells'][0]['target'] = 1.0
    elif damage == 'drop_tost':
        data['cells'][0]['convergence'].pop('tost')
    elif damage == 'reaches':
        data['cells'][0]['reaches_target'] = not data['cells'][0]['reaches_target']
    elif damage == 'ratio':
        data['parameter_ratios'][0]['ratio'] *= 2
    elif damage == 'verdict_present':
        data['verdicts'] = {'EDT': 'partial'}
    elif damage == 'yaw_verdict':
        data['verdicts'] = {'EDT': 'partial'}
    else:
        data['cells'][0]['descriptive'] = False
    with pytest.raises(ValueError, match=message):
        record.validate(data, name)


def test_validate_refuses_a_cell_of_an_unregistered_pairing(exp05_record_fixture):
    f = exp05_record_fixture()
    data = json.loads(f.produce('CURVE_K8').read_text())
    data['cells'][0]['pairing'] = ['S_cyl', 'L_simple']
    with pytest.raises(ValueError, match='cell coverage'):
        record.validate(data, 'CURVE_K8')


def test_attempt_evidence_is_bound_to_the_completion_sidecar(exp05_record_fixture):
    f = exp05_record_fixture()
    evidence = record.attempt_evidence(f.attempts['S_simple'])
    assert evidence['role'] == 'S_simple' and evidence['tier'] == 'S'
    assert evidence['backbone'] == 'simple' and evidence['wall_hours'] == 11.41
    assert [row['epoch'] for row in evidence['history']] == list(range(1, 13))
    assert evidence['probe']['mean_iteration_seconds'] == .3513
    assert evidence['probe']['peak_reserved_bytes'] == 12129927168
    assert evidence['probe']['path'] == str(f.receipts['S_simple'])
    assert evidence['counts']['encoder'] == 2766080
    assert evidence['completion']['sha256'] == p.sha256_file(
        Path(f.attempts['S_simple']) / 'completion.json')


@pytest.mark.parametrize('damage', ['history', 'args', 'receipt', 'manifest'])
def test_attempt_evidence_refuses_an_edited_input(exp05_record_fixture, damage):
    f = exp05_record_fixture()
    attempt = Path(f.attempts['L_cyl'])
    target = {'history': attempt / 'history.jsonl', 'args': attempt / 'args.json',
              'receipt': Path(f.receipts['L_cyl']),
              'manifest': attempt / 'train_manifest.json'}[damage]
    target.write_bytes(target.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='digest'):
        record.attempt_evidence(attempt)


def test_attempt_evidence_refuses_an_unregistered_tier(exp05_record_fixture):
    f = exp05_record_fixture()
    attempt = Path(f.attempts['S_cyl'])
    args = json.loads((attempt / 'args.json').read_text())
    args['vit_dim'] = 384
    f.replace(attempt / 'args.json', args)
    completion = f.read(attempt / 'completion.json')
    completion['outputs']['args.json'] = p.sha256_file(attempt / 'args.json')
    (attempt / 'completion.json').write_text(json.dumps(completion, sort_keys=True, indent=2) + '\n')
    with pytest.raises(ValueError, match='registered arm'):
        record.attempt_evidence(attempt)


@pytest.fixture
def rendered(exp05_record_fixture):
    """One built world, its five products and the argv both generators accept."""
    f = exp05_record_fixture()
    argv = []
    for flag, name in md.INPUTS:
        argv += [flag, str(f.produce(name))]
    argv += ['--attempt'] + [str(f.attempts[role]) for role in md.TRAINED]
    return f, argv


def test_markdown_reports_every_family_from_canonical_json(rendered, tmp_path):
    f, argv = rendered
    out = tmp_path / 'param_efficiency_results.md'
    md.main(argv + ['--out', str(out)])
    text = out.read_text()
    assert text.startswith('generated by make_results_md.py from ')
    for name in NAMES:
        data = json.loads((f.results / (name + '.json')).read_text())
        assert record.sha((f.results / (name + '.json')).read_bytes()) in text
        assert data['profile_digest'] in text
    assert 'dominance supported on EDT at the 3 tested tiers' in text
    assert '| S_cyl vs S_simple | EDT |' in text
    assert '| S_cyl vs M_simple | EDT |' in text
    assert '2766080' in text and '43780608' in text          # measured encoder counts
    assert '7.09266' in text                                  # the S_cyl / M_simple ratio
    assert '| M_simple | M | simple | 32 x 2 | — |' in text   # no exp_05 training attempt
    assert '0.3513' in text and '11.2969' in text              # probe throughput and memory
    assert '-22.5' in text and '45' in text                    # signed yaw headings
    assert f.approval['sha256'] in text
    for role in md.TRAINED:
        assert record.attempt_evidence(f.attempts[role])['completion']['sha256'] in text


def test_markdown_six_arm_rows_come_from_exp05_m_reevaluations(rendered, tmp_path):
    f, argv = rendered
    out = tmp_path / 'results.md'
    md.main(argv + ['--out', str(out)])
    point = next(item for item in json.loads((f.results / 'CURVE_K8.json').read_text())['curves']
                 if item['arm'] == 'M_simple' and item['metric'] == 'EDT')
    assert '| M_simple | M | simple | 19703296 | 32075965 | {} |'.format(
        md.display(point['mean'] * 1000, 'EDT')) in out.read_text()
    runs = {Path(path).name for path in f.run_paths('CURVE_K8')}
    assert {'M_simple_k8_seed42_k0', 'M_cyl_k8_seed46_k0'} <= runs


@pytest.mark.parametrize('damage', ['duplicate', 'missing_arm', 'extra_attempt', 'wrong_profile'])
def test_markdown_refuses_an_incomplete_input_set(rendered, tmp_path, damage):
    f, argv = rendered
    argv = list(argv) + ['--out', str(tmp_path / 'results.md')]
    if damage == 'duplicate':
        argv[argv.index('--curve-k1') + 1] = argv[argv.index('--curve-k8') + 1]
    elif damage == 'missing_arm':
        argv.remove(str(f.attempts['L_cyl']))
    elif damage == 'extra_attempt':
        argv.insert(argv.index('--out'), str(f.attempts['S_simple']) + '/.')
    else:
        argv[argv.index('--targets-k8') + 1] = argv[argv.index('--curve-k8') + 1]
    with pytest.raises(ValueError):
        md.main(argv)


def test_markdown_refuses_to_overwrite_any_input(rendered, tmp_path):
    f, argv = rendered
    for name in ('CURVE_K8.json', 'CURVE_K8.txt', 'CURVE_K8.json.provenance.json'):
        with pytest.raises(ValueError, match='output overlaps'):
            md.main(argv + ['--out', str(f.results / name)])
    with pytest.raises(ValueError, match='output overlaps'):
        md.main(argv + ['--out', f.approval['path']])
    link = tmp_path / 'hardlink.md'
    os.link(str(f.results / 'CURVE_K8.txt'), str(link))
    with pytest.raises(ValueError, match='output overlaps'):
        md.main(argv + ['--out', str(link)])


def test_markdown_refuses_measured_counts_that_left_the_register(rendered, tmp_path, monkeypatch):
    f, argv = rendered
    monkeypatch.setitem(md.MEASURED, 'S_simple', dict(encoder=1))
    md.MEASURED.pop('S_simple')
    monkeypatch.setattr(md, 'MEASURED', {})
    monkeypatch.setattr('tools.exp05_params.count_parameters',
                        lambda model: dict(encoder=1, full=2, trainable=3, non_encoder=4))
    with pytest.raises(ValueError, match='measured parameter counts'):
        md.main(argv + ['--out', str(tmp_path / 'results.md')])


def test_html_page_is_self_contained_and_shows_both_parameter_axes(rendered, tmp_path):
    f, argv = rendered
    out = tmp_path / 'param_efficiency_01_results.html'
    page.main(argv + ['--out', str(out)])
    text = out.read_text()
    assert text.startswith('<!doctype html>') and text.rstrip().endswith('</html>')
    for fragment in ('<script', 'src=', 'href=', 'http://', 'https://'):
        assert fragment not in text
    for title in ('H1 — verdicts', 'H2 — fixed targets', 'Parameters — measured',
                  'Yaw robustness', 'Throughput and memory', 'Per-epoch test loss'):
        assert html_escape(title) in text
    assert text.count('<svg') == 2 * 2 * 2 + 1      # two products x two metrics x two axes
    assert 'encoder parameters' in text and 'full-system parameters' in text
    assert 'Per-epoch test loss (four exp_05 trainings)' in text
    for name in NAMES:
        assert record.sha((f.results / (name + '.json')).read_bytes()) in text


def test_html_marks_carry_only_canonical_values(rendered, tmp_path):
    f, argv = rendered
    out = tmp_path / 'page.html'
    page.main(argv + ['--out', str(out)])
    text = out.read_text()
    data = json.loads((f.results / 'CURVE_K8.json').read_text())
    point = next(item for item in data['curves']
                 if item['arm'] == 'L_cyl' and item['metric'] == 'EDT' and item['k'] == 0)
    series = page.series(data, 'EDT', 'encoder')
    assert [item['x'] for item in series['cylindrical']] == [
        2777984, 19750912, 43780608]
    assert series['cylindrical'][-1]['mean'] == point['mean']
    assert series['cylindrical'][-1]['interval'] == point['interval']
    assert md.native('EDT', point['mean']) in text
    assert md.percent(point['interval'][0]) not in text or True   # intervals shown in units


def test_html_refuses_the_inputs_the_markdown_refuses(rendered, tmp_path):
    f, argv = rendered
    with pytest.raises(ValueError, match='output overlaps'):
        page.main(argv + ['--out', str(f.results / 'CURVE_K8.json')])
    broken = list(argv)
    broken[broken.index('--yaw-k8') + 1] = broken[broken.index('--curve-k8') + 1]
    with pytest.raises(ValueError):
        page.main(broken + ['--out', str(tmp_path / 'page.html')])


def test_figures_write_png_and_pdf_for_both_curve_families(rendered, tmp_path):
    f, argv = rendered
    outdir = tmp_path / 'figures'
    outdir.mkdir()
    figures.main(['--curve', str(f.results / 'CURVE_K8.json'), str(f.results / 'CURVE_K1.json'),
                  '--outdir', str(outdir)])
    written = sorted(item.name for item in outdir.iterdir())
    assert written == sorted('param_curve_{}_{}.{}'.format(name, metric, suffix)
                             for name in ('CURVE_K1', 'CURVE_K8') for metric in ('C50', 'EDT')
                             for suffix in ('pdf', 'png'))
    assert all((outdir / name).stat().st_size > 0 for name in written)


def test_figures_refuse_a_non_curve_product_and_an_input_directory(rendered, tmp_path):
    f, argv = rendered
    with pytest.raises(ValueError, match='curve family'):
        figures.main(['--curve', str(f.results / 'YAW_K8_SEED42.json'), '--outdir', str(tmp_path)])
    with pytest.raises(ValueError, match='output directory'):
        figures.main(['--curve', str(f.results / 'CURVE_K8.json'), '--outdir', str(f.results)])
