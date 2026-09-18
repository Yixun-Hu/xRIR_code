"""The exp_05 record tooling: canonical admission, the generators, the binder."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

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
    for heading in ('-45', '-22.5', '22.5', '45'):             # signed yaw headings
        assert '| S_simple | S | EDT | {} |'.format(heading) in text
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


def test_every_reader_refuses_a_product_that_left_the_registered_profile(rendered, tmp_path):
    """A self-consistent product computed under a DIFFERENT profile is not canonical."""
    f, argv = rendered
    product = f.results / 'CURVE_K8.json'
    data = json.loads(product.read_text())
    data['profile']['num_shot'] = 123
    data['profile_digest'] = record.sha(json.dumps(
        data['profile'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    side = _rewrite(product, data)
    side['profile_digest'] = data['profile_digest']
    Path(str(product) + '.provenance.json').write_text(
        json.dumps(side, sort_keys=True, indent=2) + '\n')
    for main in (md.main, page.main):
        with pytest.raises(ValueError, match='registered profile'):
            main(argv + ['--out', str(tmp_path / 'out')])
    with pytest.raises(ValueError, match='registered profile'):
        figures.main(['--curve', str(product), '--outdir', str(tmp_path)])
    with pytest.raises(ValueError, match='registered profile'):
        record.load(product, 'CURVE_K8')


def test_markdown_reports_each_metrics_own_cohort_and_exclusions(exp05_record_fixture, tmp_path):
    """A7: EDT, C50 and T60 are three cohorts of one arm, and each is reported as its own."""
    f = exp05_record_fixture(invalid=dict(S_cyl=dict(c50=[0], t60=[0, 1, 2, 3])))
    argv = []
    for flag, name in md.INPUTS:
        argv += [flag, str(f.produce(name))]
    argv += ['--attempt'] + [str(f.attempts[role]) for role in md.TRAINED]
    out = tmp_path / 'results.md'
    md.main(argv + ['--out', str(out)])
    text = out.read_text()
    known = {(point['arm'], point['metric']): point for point
             in json.loads((f.results / 'CURVE_K8.json').read_text())['curves'] if point['k'] == 0}
    assert [known[('S_cyl', m)]['cohort']['n_queries'] for m in md.ACOUSTIC] == [12, 11, 8]
    assert [known[('S_cyl', m)]['cohort']['n_rooms'] for m in md.ACOUSTIC] == [3, 3, 2]
    row = next(line for line in text.splitlines() if line.startswith('| S_cyl | S |'))
    for queries, rooms, lost in ((12, 3, 0), (11, 3, 1), (8, 2, 4)):
        assert '| {} | {} | {} |'.format(queries, rooms, '; '.join(
            '{}: {}'.format(seed, lost) for seed in (42, 43, 44, 45, 46))) in row
    cell = next(item for item in json.loads((f.results / 'TARGETS_K8.json').read_text())['cells']
                if item['pairing'] == ['S_cyl', 'M_simple'] and item['metric'] == 'C50')
    assert cell['paired_cohort']['n_queries'] == 11          # H2 inherits the same exclusion
    assert '| {} | {} |'.format(cell['baseline_own']['cohort']['n_queries'],
                                cell['baseline_own']['cohort']['n_rooms']) in text
    assert md.percent(cell['room_cluster_interval']) in text


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


def _consumed(f):
    """Every file the generators read: products, companions, sidecars, approval, attempts."""
    attempt = Path(f.attempts['S_simple'])
    return ([f.results / name for name in
             ('CURVE_K8.json', 'CURVE_K8.txt', 'CURVE_K8.json.provenance.json')]
            + [attempt / name for name in
               ('completion.json', 'train_manifest.json', 'args.json', 'history.jsonl')]
            + [Path(f.receipts['S_simple']), Path(f.approval['path'])])


@pytest.mark.parametrize('generator', ['md', 'html'])
@pytest.mark.parametrize('kind', ['direct', 'hardlink', 'symlink'])
def test_generators_never_overwrite_a_consumed_input(rendered, tmp_path, generator, kind):
    """A destination naming an input by path, by hardlink or by symlink is refused."""
    f, argv = rendered
    main = dict(md=md.main, html=page.main)[generator]
    targets = _consumed(f)
    before = {str(item): item.read_bytes() for item in targets}
    for index, target in enumerate(targets):
        out = target
        if kind != 'direct':
            out = tmp_path / '{}_{}_{}'.format(generator, kind, index)
            os.link(str(target), str(out)) if kind == 'hardlink' else out.symlink_to(target)
        with pytest.raises(ValueError, match='output overlaps'):
            main(argv + ['--out', str(out)])
    assert {str(item): item.read_bytes() for item in targets} == before


@pytest.mark.parametrize('kind', ['hardlink', 'symlink'])
def test_figures_never_overwrite_a_consumed_input(rendered, tmp_path, kind):
    """A predicted figure destination that is a second name for an input is refused."""
    f, argv = rendered
    outdir = tmp_path / ('figures_' + kind)
    outdir.mkdir()
    target = f.results / 'CURVE_K8.json'
    before = target.read_bytes()
    destination = outdir / 'param_curve_CURVE_K8_EDT.png'
    os.link(str(target), str(destination)) if kind == 'hardlink' else destination.symlink_to(target)
    with pytest.raises(ValueError, match='output overlaps'):
        figures.main(['--curve', str(target), '--outdir', str(outdir)])
    assert target.read_bytes() == before
    assert sorted(item.name for item in outdir.iterdir()) == [destination.name]


def test_markdown_refuses_measured_counts_that_left_the_register(rendered, tmp_path, monkeypatch):
    f, argv = rendered
    monkeypatch.setattr(md, 'MEASURED', {})   # the per-process cache, restored on teardown
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
    tooltip = 'L_cyl (cylindrical, L): {} s; seed SD {}; interval {}; {} queries'.format(
        md.native('EDT', point['mean']), md.native('EDT', point['sd']),
        md.native('EDT', point['interval']), point['cohort']['n_queries'])
    assert md.html.escape(tooltip, quote=True) in text


def test_html_marks_keep_their_backbone_colour(rendered, tmp_path):
    """The imported CSS paints `.mark` blue; a series' own colour has to outrank it."""
    f, argv = rendered
    out = tmp_path / 'page.html'
    page.main(argv + ['--out', str(out)])
    text = out.read_text()
    assert '.mark{stroke:var(--blue);fill:var(--blue)}' in text     # the rule outranked
    assert 'class="mark" fill=' not in text                         # never an attribute
    orange, blue = (page.COLOUR[name] for name in ('cylindrical', 'simple'))
    curve = '<g tabindex="0" class="mark" style="fill:{0};stroke:{0}">'
    # two products x two metrics x two parameter axes x three tiers of each backbone
    assert text.count(curve.format(orange)) == text.count(curve.format(blue)) == 2 * 2 * 2 * 3
    trajectory = '<g tabindex="0" class="mark" style="fill:{};stroke:none">'
    assert text.count(trajectory.format(orange)) == 2 * 12          # S_cyl and L_cyl epochs


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


def _republish(path, data=None, side=None):
    """Rewrite a product and its sidecar so only the intended change differs."""
    path = Path(path)
    if data is not None:
        path.write_bytes((json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
    sidecar = Path(str(path) + '.provenance.json')
    side = json.loads(sidecar.read_text()) if side is None else side
    side['outputs'][str(path.resolve())] = record.sha(path.read_bytes())
    sidecar.write_text(json.dumps(side, sort_keys=True, indent=2) + '\n')


@pytest.fixture
def bound(rendered, tmp_path, monkeypatch):
    """The whole record: 66 runs, four attempts, five products, documents and figures."""
    f, argv = rendered
    binder = record.load_asset('bind_provenance')
    monkeypatch.setattr(binder, 'load_approved_digests', lambda path=None: f.approved)
    monkeypatch.setattr(binder, 'live_producer', f.producer_now)
    monkeypatch.setattr(binder, 'HISTORICAL', {
        arm['role']: dict(path=arm['checkpoint'], sha256=arm['sha256'], epoch=arm['epoch'])
        for arm in f.arms if arm['tier'] == 'M'})
    documents = [f.root / 'param_efficiency_results.md', f.root / 'param_efficiency_01_results.html']
    md.main(argv + ['--out', str(documents[0])])
    page.main(argv + ['--out', str(documents[1])])
    figdir = f.root / 'figures'
    figdir.mkdir()
    figures.main(['--curve', str(f.results / 'CURVE_K8.json'), str(f.results / 'CURVE_K1.json'),
                  '--outdir', str(figdir), '--format', 'png'])
    reports = tmp_path / 'reports'
    reports.mkdir()
    arguments = dict(runs=[str(item) for item in f.runs.values()],
                     attempt=[str(f.attempts[role]) for role in md.TRAINED],
                     results=[str(f.results / (name + '.json')) for name in NAMES],
                     rendered=[str(item) for item in documents],
                     figures=sorted(str(item) for item in figdir.iterdir()))

    def arguments_argv():
        argv = []
        for name in ('runs', 'attempt', 'results', 'rendered', 'figures'):
            argv += ['--' + name] + list(arguments[name])
        return argv + ['--out', str(reports)]

    return SimpleNamespace(binder=binder, f=f, reports=reports, documents=documents, argv=argv,
                           arguments=arguments, arguments_argv=arguments_argv)


def test_the_binding_report_covers_the_whole_param_efficiency_record(bound):
    report = bound.binder.collect(**bound.arguments)
    assert len(report['runs']) == 66 and len(report['attempts']) == 4
    assert {(item['role'], item['num_shot'], item['seed'], item['kind'])
            for item in report['runs']} == bound.binder.IDENTITIES
    assert sorted(item['role'] for item in report['attempts']) == sorted(md.TRAINED)
    assert sorted(report['historical_checkpoints']) == ['M_cyl', 'M_simple']
    assert [item['profile'] for item in report['results']] == sorted(NAMES)
    assert len(report['documents']) == 2 and len(report['figures']) == 4
    assert all('bound' not in item for item in report['runs'] + report['attempts'])
    assert json.dumps(report, allow_nan=False)
    for attempt in report['attempts']:
        assert attempt['full_attempts'] == 1 and len(attempt['probe_linkage']) == 1
        assert attempt['other_attempts'][0]['state'] == 'certified'
        assert attempt['other_attempts'][0]['logs'][0]['present'] is True


def test_each_products_dependency_map_equals_exactly_what_it_declared(bound):
    binder = bound.binder
    report = binder.collect(**bound.arguments)
    attempts = [binder.attempt_record(item, bound.f.pins) for item in bound.arguments['attempt']]
    historical = binder.historical_checkpoints()
    runs = [binder.run_record(item, attempts, historical, report['approved_digests'])
            for item in bound.arguments['runs']]
    run_bound = {item['path']: item.pop('bound') for item in runs}
    trained = {item['role']: item for item in attempts}
    assert sorted(binder.training_dependencies(trained['S_simple'])) == [
        'args.json', 'completion.json', 'train_manifest.json']
    for product in bound.arguments['results']:
        data = json.loads(Path(product).read_text())
        side = json.loads(Path(product + '.provenance.json').read_text())
        expected = set(side['run_flags'])
        roles = {run['role'] for run in runs if run['path'] in expected}
        dependencies = binder.product_dependencies(expected, roles, trained, run_bound,
                                                   report['approved_digests'])
        dependencies.update(binder.producer_declarations(side['producer']))
        assert dependencies == side['inputs'], product


def test_check_record_recomputes_the_latest_report(bound):
    checker = record.load_asset('check_record')
    bound.binder.main(bound.arguments_argv())
    reports = sorted(bound.reports.glob('binding_report_*.json'))
    assert len(reports) == 1
    checker.main([str(bound.reports)])
    report = json.loads(reports[0].read_text())
    report['runs'][0]['completion']['sha256'] = 'f' * 64
    reports[0].write_text(json.dumps(report, indent=2))
    with pytest.raises(ValueError, match='binding report mismatch'):
        checker.main([str(bound.reports)])


def test_a_drifting_producer_source_leaves_the_record_unbindable(bound):
    """A declared producer file that changes after publication is not the bound one."""
    checker = record.load_asset('check_record')
    bound.binder.main(bound.arguments_argv())
    checker.main([str(bound.reports)])
    source = bound.f.producer_source
    source.write_text(source.read_text() + '# appended after publication\n')
    with pytest.raises(ValueError, match='producer source'):
        bound.binder.collect(**bound.arguments)
    with pytest.raises(ValueError, match='producer source'):
        checker.main([str(bound.reports)])


@pytest.mark.parametrize('forgery,message', [
    ('omit_run', 'the sixty-six evaluation runs'),
    ('extra_run', 'the sixty-six evaluation runs'),
    ('substituted_digest', 'digest mismatch'),
    ('unknown_dependency', 'this report does not bind'),
    ('omitted_dependency', 'omits a required input'),
    ('run_flag_subset', 'does not declare exactly its runs'),
    ('contract_subset', 'does not record exactly its run contracts'),
    ('foreign_contract', 'the contract is not this run'),
    ('edited_attempt_log', 'digest mismatch'),
    ('edited_probe_receipt', 'manifest input mismatch'),
    ('foreign_probe_receipt', 'does not list'),
    ('stale_approval', 'approval digest mismatch'),
    ('unpinned_checkpoint', 'approved checkpoint'),
    ('relaxed_profile', 'not computed under the registered profile'),
    ('exploratory_product', 'exploratory JSON refused'),
    ('unconverged_cell', 'did not converge'),
    ('uncited_document', 'does not cite every canonical digest'),
    ('two_retries', 'more than one retry'),
    ('swapped_evaluator', 'evaluator closure is not the approved one'),
    ('swapped_writer', 'writer closure is not the approved one'),
    ('forged_closure_record', 'closure records'),
    ('missing_training_linkage', 'training linkage'),
    ('misnamed_attempt', 'does not name this attempt directory'),
])
def test_the_binder_refuses_a_forged_record(bound, forgery, message):
    f, arguments = bound.f, dict(bound.arguments)
    product = f.results / 'CURVE_K8.json'
    if forgery == 'omit_run':
        arguments['runs'] = arguments['runs'][1:]
    elif forgery == 'extra_run':
        arguments['runs'] = arguments['runs'] + [arguments['runs'][0] + '/.']
    elif forgery == 'substituted_digest':
        run = Path(arguments['runs'][0]) / 'per_sample_yaw.json'
        run.write_bytes(run.read_bytes() + b' ')
    elif forgery in ('unknown_dependency', 'omitted_dependency'):
        data = json.loads(product.read_text())
        planted = str(f.root / 'planted.json')
        Path(planted).write_text('{}\n')
        if forgery == 'unknown_dependency':
            data['inputs'][planted] = record.sha(Path(planted).read_bytes())
        else:
            data['inputs'].pop(sorted(data['inputs'])[-1])
        side = json.loads(Path(str(product) + '.provenance.json').read_text())
        side['inputs'] = data['inputs']
        _republish(product, data, side)
    elif forgery in ('run_flag_subset', 'contract_subset', 'foreign_contract'):
        data = json.loads(product.read_text())
        dropped = sorted(data['run_flags'])[0]
        if forgery == 'run_flag_subset':
            data['run_flags'].pop(dropped)
            side = json.loads(Path(str(product) + '.provenance.json').read_text())
            side['run_flags'] = data['run_flags']
            _republish(product, data, side)
        elif forgery == 'contract_subset':
            data['compatibility'].pop(dropped)
            _republish(product, data)
        else:
            data['compatibility'][dropped]['sha256'] = 'a' * 64
            _republish(product, data)
    elif forgery == 'edited_attempt_log':
        completion = f.read(Path(f.attempts['S_simple']) / 'completion.json')
        log = Path(completion['log']['path'])
        log.write_text(log.read_text() + 'appended\n')
    elif forgery == 'edited_probe_receipt':
        receipt = f.read(f.receipts['S_cyl'])
        receipt['probe_attempt']['completion_sha256'] = 'b' * 64
        f.replace(f.receipts['S_cyl'], receipt)
    elif forgery == 'foreign_probe_receipt':
        receipt = f.read(f.receipts['S_cyl'])
        receipt['probe_attempt']['path'] = str(Path(f.attempts['L_cyl']).parent / '_probe_x_arm')
        f.replace(Path(f.receipts['S_cyl']).with_name('_probe_x_S_cyl.json'), receipt)
    elif forgery == 'stale_approval':
        path = Path(f.approval['path'])
        path.write_text(path.read_text() + '\n')
    elif forgery == 'unpinned_checkpoint':
        f.pins['checkpoints']['S_simple'] = dict(f.pins['checkpoints']['S_simple'], epoch=11)
    elif forgery == 'relaxed_profile':
        data = json.loads(product.read_text())
        data['profile']['margin'] = .5
        data['profile_digest'] = record.sha(json.dumps(
            data['profile'], sort_keys=True, separators=(',', ':')).encode())
        side = json.loads(Path(str(product) + '.provenance.json').read_text())
        side['profile_digest'] = data['profile_digest']
        _republish(product, data, side)
    elif forgery in ('exploratory_product', 'unconverged_cell'):
        data = json.loads(product.read_text())
        if forgery == 'exploratory_product':
            data['exploratory'] = True
        else:
            data['cells'][0]['convergence']['superiority']['passed'] = False
        _republish(product, data)
    elif forgery == 'uncited_document':
        bound.documents[0].write_text('a record with no digests\n')
    elif forgery in ('swapped_evaluator', 'swapped_writer', 'forged_closure_record'):
        # The M pair is the only tier a legacy evaluator could ever be claimed for.
        role = 'M_simple' if forgery == 'swapped_evaluator' else 'S_simple'
        run = Path(f.runs[(role, 8, 42, 'k0')])
        manifest = f.read(run / 'eval_manifest.json')
        closures = manifest['source_closures']
        if forgery == 'swapped_evaluator':
            closures['entrypoint'] = dict(closures['writer'])
        elif forgery == 'swapped_writer':
            closures['writer'] = dict(closures['entrypoint'])
        else:
            closures['writer'] = dict(closures['writer'], sha256='a' * 64)
        f.rebind(run, manifest=manifest)
    elif forgery == 'misnamed_attempt':
        attempt = Path(f.attempts['S_cyl'])
        manifest = f.read(attempt / 'train_manifest.json')
        manifest['attempt_path'] = str(f.attempts['L_cyl'])
        digest = f.replace(attempt / 'train_manifest.json', manifest)
        completion = f.read(attempt / 'completion.json')
        completion['train_manifest_sha256'] = digest
        for key in ('outputs', 'directory_listing'):
            completion[key]['train_manifest.json'] = digest
        f.replace(attempt / 'completion.json', completion)
    elif forgery == 'two_retries':
        ledger = f.read(f.ledgers['L_simple'])
        ledger['attempts'].append(dict(attempt='attempt_first_ABORTED_slow', hours=1., mode='full'))
        ledger['attempts'].append(dict(attempt='attempt_second_ABORTED_slow', hours=1., mode='full'))
        f.replace(f.ledgers['L_simple'], ledger)
    else:
        run = Path(f.runs[('S_simple', 8, 42, 'k0')])
        manifest = f.read(run / 'eval_manifest.json')
        manifest['mutable_inputs'].pop('train_completion')
        f.rebind(run, manifest=manifest)
    with pytest.raises(ValueError, match=message):
        bound.binder.collect(**arguments)


def _relocate(directory, destination):
    """Move a bound directory away and leave a directory symlink where it was.

    Exactly the migration the Planner performs once an attempt is certified: the bytes
    move to the NAS, the arm directory keeps its ledger, probe receipts and ``final``,
    and every path the record already published still names the evidence.
    """
    directory, destination = Path(directory), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    directory.rename(destination)
    directory.symlink_to(destination, target_is_directory=True)
    return destination


@pytest.mark.parametrize('moved', ['attempt', 'arm'])
def test_a_relocated_attempt_leaves_the_record_verifiable(bound, tmp_path, moved):
    """Nothing changed but where the bytes live, so nothing about the record changes."""
    checker = record.load_asset('check_record')
    bound.binder.main(bound.arguments_argv())
    report = json.loads(sorted(bound.reports.glob('binding_report_*.json'))[0].read_text())
    attempt = Path(bound.f.attempts['S_simple'])
    directory = attempt if moved == 'attempt' else attempt.parent
    destination = _relocate(directory, tmp_path / 'nas' / directory.name)
    checker.main([str(bound.reports)])
    assert bound.binder.collect(**bound.arguments) == report
    # The approval's logical path is the identity, whichever name the binder is given.
    relocated = destination / attempt.name if moved == 'arm' else destination
    assert bound.binder.collect(**dict(bound.arguments, attempt=sorted(
        [str(relocated)] + [item for item in bound.arguments['attempt']
                            if item != str(attempt)]))) == report


def test_a_relocated_attempt_that_changed_is_still_refused(bound, tmp_path):
    checker = record.load_asset('check_record')
    bound.binder.main(bound.arguments_argv())
    attempt = Path(bound.f.attempts['S_cyl'])
    destination = _relocate(attempt, tmp_path / 'nas' / attempt.name)
    history = destination / 'history.jsonl'
    history.write_text(history.read_text() + json.dumps(dict(epoch=13)) + '\n')
    for call in (lambda: bound.binder.collect(**bound.arguments),
                 lambda: checker.main([str(bound.reports)])):
        with pytest.raises(ValueError, match='digest mismatch'):
            call()


def test_a_relocated_arm_still_refuses_a_final_symlink_naming_another_attempt(bound, tmp_path):
    """`final` is both the approval's route to the checkpoint and the attempt's seal."""
    attempt = Path(bound.f.attempts['L_simple'])
    destination = _relocate(attempt.parent, tmp_path / 'nas' / attempt.parent.name)
    final = destination / 'final'
    final.unlink()
    final.symlink_to(sorted(destination.glob('_probe_*/'))[0].name, target_is_directory=True)
    with pytest.raises(ValueError, match='approved arm|final symlink is not this attempt'):
        bound.binder.collect(**bound.arguments)


def test_the_document_names_a_relocated_attempt_where_the_record_does(rendered, tmp_path):
    """The record's own provenance table is the approval's path, not the archive's."""
    f, argv = rendered
    before, after = tmp_path / 'before.md', tmp_path / 'after.md'
    md.main(argv + ['--out', str(before)])
    attempt = Path(f.attempts['S_simple'])
    destination = _relocate(attempt, tmp_path / 'nas' / attempt.name)
    md.main(argv + ['--out', str(after)])
    assert str(attempt) in before.read_text() and str(destination) not in before.read_text()
    assert after.read_text() == before.read_text()
    # and the archived bytes are still what the generator refuses to write over
    with pytest.raises(ValueError, match='output overlaps canonical input'):
        md.main(argv + ['--out', str(destination / 'history.jsonl')])


def test_a_relocated_products_directory_leaves_the_record_verifiable(bound, tmp_path):
    """An archived ancestor takes the products with it; they keep their published names."""
    checker = record.load_asset('check_record')
    bound.binder.main(bound.arguments_argv())
    report = json.loads(sorted(bound.reports.glob('binding_report_*.json'))[0].read_text())
    _relocate(bound.f.results, tmp_path / 'nas' / 'results')
    checker.main([str(bound.reports)])
    assert bound.binder.collect(**bound.arguments) == report


def test_the_figures_of_a_relocated_products_directory_are_the_same_figures(rendered, tmp_path):
    f, argv = rendered
    outdirs = [tmp_path / 'before', tmp_path / 'after']
    curve = str(f.results / 'CURVE_K8.json')
    drawn = []
    for index, outdir in enumerate(outdirs):
        outdir.mkdir()
        if index:
            _relocate(f.results, tmp_path / 'nas' / 'results')
        drawn.append(sorted(figures.main(['--curve', curve, '--outdir', str(outdir),
                                          '--format', 'png'])))
    assert [item.name for item in drawn[0]] == [item.name for item in drawn[1]]
    assert all(one.read_bytes() == other.read_bytes() for one, other in zip(*drawn))
