"""TABLE_V1 uses the paired producer's execution-bound synthetic runs."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_paired_compare import admission_fixture, _read, _rebind, _summaries
from tools import paired_compare as pc, provenance as p, results_table as rt
from tools.exp04_profiles import get_profile, json_value


@pytest.fixture
def table_fixture(admission_fixture, monkeypatch):
    fixtures = [admission_fixture('H1_K1'), admission_fixture('H1_K8')]
    profile = json_value(get_profile('TABLE_V1'))
    profile['dataset'] = fixtures[1].profile['dataset']
    arms = {a['role']: a for a in fixtures[1].profile['arms']}
    checkpoint = fixtures[1].root / 'cyl.pth'
    checkpoint.write_bytes(b'cyl fixture checkpoint')
    arms['cyl'] = dict(profile['arms'][1], checkpoint=str(checkpoint), sha256=p.sha256_file(checkpoint))
    profile['arms'] = [arms[a['role']] for a in profile['arms']]
    directories = []
    for fixture in fixtures:
        shot = fixture.profile['num_shot']
        profile['seeds'][shot] = fixture.profile['seeds'][shot]
        for original in fixture.paths[1]:
            clone = Path(original).with_name(Path(original).name.replace('control', 'cyl'))
            shutil.copytree(original, clone)
            directories.append(str(clone))
        directories.extend(sum(fixture.paths, []))
    for directory in directories:
        run = Path(directory)
        role = run.name.split('_')[0]
        manifest, sample, metrics = [_read(run / name) for name in
            ('eval_manifest.json', 'per_sample_yaw.json', 'metrics_yaw.json')]
        manifest.update(checkpoint=arms[role]['checkpoint'], checkpoint_sha256=arms[role]['sha256'],
                        backbone=arms[role]['backbone'])
        for payload in (sample, metrics):
            payload['meta'].update(checkpoint=manifest['checkpoint'], backbone=manifest['backbone'])
        sample['P']['0'].update(stft=[.5] * 12, decay=[.5] * 12, consistency=[0.] * 12)
        metrics['P'] = _summaries(sample['P'])
        _rebind(run, manifest, sample, metrics)
    producer = pc.producer_identity()
    fixtures[1].approved['closures']['producer_results_table'] = producer['sha256']
    monkeypatch.setattr(rt, 'producer_identity', lambda _: producer)
    monkeypatch.setattr(rt, 'load_approved_digests', pc.load_approved_digests)
    monkeypatch.setattr(rt, 'get_profile', lambda _: profile)
    return SimpleNamespace(directories=directories, profile=profile)


def test_table_aggregation_and_order(table_fixture):
    result, admitted = rt.build_table(table_fixture.directories)
    reverse, _ = rt.build_table(list(reversed(table_fixture.directories)))
    assert json.dumps(result, sort_keys=True) == json.dumps(reverse, sort_keys=True)
    assert len(result['rows']) == 6 and len(admitted['groups']) == 6
    for row in result['rows']:
        assert row['protocol']['batch_size'] == 16 and row['protocol']['k'] == 0
        edt = row['metrics']['EDT']
        assert edt['mean'] == pytest.approx((1 + 5.5 / 16 + .02) * 1000)
        assert edt['sd'] == pytest.approx(np.std(np.arange(5) * 10, ddof=1))
        assert row['metrics']['T60']['mean'] == pytest.approx(edt['mean'] / 1000)
        assert [v['n_finite'] for v in edt['per_seed'].values()] == [12] * 5
        assert set(row['metrics']) == {'T60', 'C50', 'EDT', 'loss', 'log_mse'}


@pytest.mark.parametrize('mutation', ['incomplete', 'wrong_role', 'tampered', 'unknown', 'finite'])
def test_table_refuses_invalid_inputs(table_fixture, mutation):
    directories = list(table_fixture.directories)
    run = Path(directories[0])
    if mutation == 'incomplete':
        directories.pop()
    elif mutation in ('wrong_role', 'tampered'):
        path = run / ('eval_manifest.json' if mutation == 'wrong_role' else 'per_sample_yaw.json')
        value = _read(path)
        if mutation == 'wrong_role':
            value['checkpoint_sha256'] = 'f' * 64
        path.write_text(json.dumps(value) + ' ')
    else:
        sample, metrics = _read(run / 'per_sample_yaw.json'), _read(run / 'metrics_yaw.json')
        sample['P']['0']['unknown' if mutation == 'unknown' else 'edt'] = [None] * 3 + [1.] * 9
        metrics['P'] = _summaries(sample['P'])
        _rebind(run, sample=sample, metrics=metrics)
    with pytest.raises(ValueError):
        rt.build_table(directories)


@pytest.mark.parametrize('count', [2, 12])
def test_finite_count_boundary_and_empty_seed(table_fixture, count):
    run = Path(table_fixture.directories[0])
    sample, metrics = _read(run / 'per_sample_yaw.json'), _read(run / 'metrics_yaw.json')
    sample['P']['0']['edt'][:count] = [None] * count
    metrics['P'] = _summaries(sample['P'])
    _rebind(run, sample=sample, metrics=metrics)
    if count == 12:
        with pytest.raises(ValueError, match='zero finite'):
            rt.build_table(table_fixture.directories)
    else:
        result, _ = rt.build_table(table_fixture.directories)
        cell = next(row for row in result['rows'] if row['role'] == 'cyl' and row['num_shot'] == 1)
        assert cell['metrics']['EDT']['per_seed']['42']['n_finite'] == 10
        assert cell['metrics']['EDT']['per_seed']['42']['mean'] == pytest.approx((1 + 6.5 / 16) * 1000)


def table_argv(fixture, tmp_path, name='table'):
    return ['--profile', 'TABLE_V1', '--runs'] + fixture.directories + [
        '--json', str(tmp_path / (name + '.json')), '--md', str(tmp_path / (name + '.md'))]


def test_cli_outputs_protocol_and_json_only_renderer(table_fixture, tmp_path, monkeypatch):
    argv = table_argv(table_fixture, tmp_path)
    result = rt.main(argv)
    path, md = tmp_path / 'table.json', tmp_path / 'table.md'
    receipt = _read(str(path) + '.provenance.json')
    assert _read(path) == result and 'generated_at' not in result
    assert receipt['outputs'] == {str(path): p.sha256_file(path), str(md): p.sha256_file(md)}
    assert receipt['producer']['sha256'] == result['producer_closure_sha256']
    assert receipt['approved_digests']['git_blob'] and receipt['generated_at']
    assert receipt['inputs'] == result['inputs'] and len(receipt['run_flags']) == 30
    for row in result['rows']:
        for cell in row['metrics'].values():
            assert '{:.6g} ± {:.6g}'.format(cell['mean'], cell['sd']) in md.read_text()
    assert 'K = 1; unseen; 12 queries; 5 seeds; epoch 12; P; k = 0; batch 16; TF32 off' in md.read_text()
    for directory in table_fixture.directories:
        shutil.rmtree(directory)
    monkeypatch.setattr(rt, 'build_table', lambda *a: pytest.fail('renderer read runs'))
    assert rt.render_markdown(path) == md.read_text()


@pytest.mark.parametrize('existing', ['table.json', 'table.md', 'table.json.provenance.json'])
def test_exclusive_outputs(table_fixture, tmp_path, existing):
    path = tmp_path / existing
    path.write_text('preserve')
    with pytest.raises(FileExistsError):
        rt.main(table_argv(table_fixture, tmp_path))
    assert path.read_text() == 'preserve'
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == [existing]


def test_json_idempotence_and_force_md_only(table_fixture, tmp_path):
    rt.main(table_argv(table_fixture, tmp_path))
    first = (tmp_path / 'table.json').read_bytes()
    rt.main(table_argv(table_fixture, tmp_path, 'again'))
    assert (tmp_path / 'again.json').read_bytes() == first
    argv = table_argv(table_fixture, tmp_path, 'third')
    argv[-1] = str(tmp_path / 'table.md')
    rt.main(argv + ['--force-md'])
    with pytest.raises(FileExistsError):
        rt.main(argv + ['--force-md'])
    assert (tmp_path / 'table.json').read_bytes() == first


def test_no_partial_outputs_on_render_failure(table_fixture, tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError('render failed')
    monkeypatch.setattr(rt, 'render_markdown', fail)
    with pytest.raises(RuntimeError, match='render failed'):
        rt.main(table_argv(table_fixture, tmp_path))
    assert not list(tmp_path.glob('table.*'))


def test_synthetic_json_template_and_provenance(table_fixture, tmp_path):
    result, _ = rt.build_table(table_fixture.directories)
    for row in result['rows']:
        row['protocol']['n_queries'] = 6337
    path, output = tmp_path / 'synthetic.json', tmp_path / 'rendered.md'
    path.write_text(json.dumps(result))
    command = ['results_table.py', '--profile', 'TABLE_V1', '--json', str(path)]
    output.write_text(rt.render_markdown(path, command))
    text = output.read_text()
    for phrase in ('# Model comparison', 'sample SD', 'five evaluation seeds',
                   'reference manifest', 'Griffin-Lim', '## Protocol', '6337 queries',
                   'finite queries', 'tolerance: 2', '## Provenance', str(path),
                   p.sha256_file(path), result['profile_digest'], 'results_table.py --profile TABLE_V1'):
        assert phrase in text


def test_force_md_preserves_manual_content_idempotently(table_fixture, tmp_path):
    path = tmp_path / 'table.md'
    manual = 'Legacy note\n\n| Legacy model | 7.5 |\n'
    path.write_text(manual)
    argv = table_argv(table_fixture, tmp_path)
    rt.main(argv + ['--force-md'])
    first = path.read_text()
    assert first.count('Legacy note') == first.count('## Manual') == 1
    assert manual.strip() in first.split('## Manual', 1)[1]
    argv[argv.index('--json') + 1] = str(tmp_path / 'second.json')
    rt.main(argv + ['--force-md'])
    assert path.read_text().count('Legacy note') == path.read_text().count('## Manual') == 1
    assert path.read_text().count('# Model comparison') == 1


def test_renderer_refuses_tampered_canonical_json(table_fixture, tmp_path):
    rt.main(table_argv(table_fixture, tmp_path))
    path = tmp_path / 'table.json'
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='digest'):
        rt.render_markdown(path)


def test_force_md_restores_original_on_sidecar_failure(table_fixture, tmp_path, monkeypatch):
    path = tmp_path / 'table.md'
    path.write_text('Manual original')
    original = rt._publish
    def fail(path, *args, **kwargs):
        if path.name.endswith('.provenance.json'):
            raise OSError('sidecar failed')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(rt, '_publish', fail)
    with pytest.raises(OSError, match='sidecar failed'):
        rt.main(table_argv(table_fixture, tmp_path) + ['--force-md'])
    assert path.read_text() == 'Manual original'
    assert not (tmp_path / 'table.json').exists()
