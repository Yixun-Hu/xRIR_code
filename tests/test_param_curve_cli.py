import hashlib
import json
from pathlib import Path
import pytest
from exp05_fixture import exp05_fixture
from tools import param_curve as pc
from tools import paired_compare as shared


def command(monkeypatch, f, exploratory=False):
    monkeypatch.setattr(pc, 'get_profile', lambda _: f.profile)
    monkeypatch.setattr(pc, 'load_approved_digests', lambda: f.approved)
    monkeypatch.setattr(pc, 'producer_identity', lambda _: f.producer)
    name = ('YAW_K8_SEED42' if f.profile['mode'] == 'yaw' else
            ('CURVE_K' if f.profile['mode'] == 'curve' else 'TARGETS_K') + str(f.profile['num_shot']))
    return ['--profile', name,
        '--runs'] + f.directories + ['--json', str(f.root / 'result.json'),
        '--summary', str(f.root / 'summary.txt')] + (['--exploratory'] if exploratory else [])


@pytest.mark.parametrize('name', ['CURVE_K8', 'TARGETS_K8'])
def test_cli_outputs_canonical_deterministic_and_exclusive(exp05_fixture, monkeypatch, name):
    f = exp05_fixture(name)
    argv = command(monkeypatch, f)
    original = shared.render_summary
    calls, recheck = [], shared.recheck_inputs
    def checked(admitted):
        calls.append(admitted)
        return recheck(admitted)
    monkeypatch.setattr(shared, 'recheck_inputs', checked)
    monkeypatch.setattr(pc, 'recheck_inputs', checked, raising=False)
    result = pc.main(argv)
    assert len(calls) == 1
    assert shared.render_summary is original
    raw = (f.root / 'result.json').read_bytes()
    data = json.loads(raw)
    assert data['compatibility'] and data['cells'] and len(data['curves']) == 18
    receipt = json.loads((f.root / 'result.json.provenance.json').read_text())
    assert receipt['outputs'][str(f.root / 'result.json')] == hashlib.sha256(raw).hexdigest()
    assert str(f.root / 'result.json.provenance.json') not in receipt['outputs']
    assert receipt['approved_digests']['sha256'] == f.approved[1]['sha256']
    summary = (f.root / 'summary.txt').read_text()
    assert summary.startswith('CONFIRMATORY ' + name + '\n')
    assert 'S_simple EDT' in summary and 'encoder=2766080' in summary
    if name.startswith('TARGETS'):
        assert 'target on paired cohort' in summary and 'baseline own cohort' in summary
    with pytest.raises(FileExistsError):
        pc.main(argv)
    argv[argv.index('--json')+1] = str(f.root / 'second.json')
    argv[argv.index('--summary')+1] = str(f.root / 'second.txt')
    assert result == pc.main(argv)
    assert raw == (f.root / 'second.json').read_bytes()


def test_exploratory_summary_and_mutated_input_refusal(exp05_fixture, monkeypatch):
    f = exp05_fixture()
    f.pins['closures']['producer_param_curve'] = None
    argv = command(monkeypatch, f, True)
    data = pc.main(argv)
    assert data['exploratory'] and data['deviations'] and 'verdicts' not in data
    assert 'EXPLORATORY' in (f.root / 'summary.txt').read_text()
    assert 'dominance supported' not in (f.root / 'summary.txt').read_text()
    admitted = pc.admit(f.directories, f.profile, f.approved, f.producer, True)
    Path(next(iter(admitted['inputs']))).write_text('changed')
    with pytest.raises(ValueError, match='input changed'):
        pc.publish(data, admitted, f.root / 'bad.json', f.root / 'bad.txt')
    assert not (f.root / 'bad.json').exists()


def test_cli_rejects_analytic_overrides():
    with pytest.raises(SystemExit):
        pc.main(['--profile', 'CURVE_K8', '--runs', 'fake', '--json', 'a', '--summary', 'b', '--n-boot', '2'])


@pytest.mark.parametrize('name', ['CURVE_K1', 'TARGETS_K1', 'YAW_K8_SEED42'])
def test_secondary_and_yaw_cli(exp05_fixture, monkeypatch, name):
    f = exp05_fixture(name)
    result = pc.main(command(monkeypatch, f))
    assert result['profile_name'] == name
    if name.startswith('YAW'):
        assert len(result['cells']) == 72 and 'verdicts' not in result
        assert (f.root / 'summary.txt').read_text().startswith('DESCRIPTIVE')


def test_summary_rendering_cannot_race_publication_inputs(exp05_fixture, monkeypatch):
    f = exp05_fixture()
    argv = command(monkeypatch, f)
    original = pc.render_summary
    def mutate(result):
        Path(f.profile['arms'][0]['checkpoint']).parent.joinpath('args.json').write_text('{}')
        return original(result)
    monkeypatch.setattr(pc, 'render_summary', mutate)
    with pytest.raises(ValueError, match='input changed'):
        pc.main(argv)
    assert not any((f.root / name).exists() for name in ('result.json', 'summary.txt', 'result.json.provenance.json'))


def test_writer_adapter_refuses_missing_renderer_hook(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, 'write_outputs', lambda *args: None)
    with pytest.raises(RuntimeError, match='render_summary'):
        pc.publish({}, {}, tmp_path / 'result.json', tmp_path / 'summary.txt')
    assert list(tmp_path.iterdir()) == []
