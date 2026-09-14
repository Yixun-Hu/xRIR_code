"""Descriptive producers retain execution-bound admission and never claim verdicts."""
from pathlib import Path
import numpy as np
import pytest
from test_paired_compare import admission_fixture
from test_exp04_record_tools import record_inputs
from test_results_table import table_fixture
from tools import exp04_descriptive as diagnostic
from tools.exp04_record import load_asset

@pytest.mark.parametrize('name,count', [('GRID_SEED42', 66), ('EPOCH9_K8', 5)])
def test_descriptive_outputs_and_generators(admission_fixture, record_inputs, tmp_path, name, count):
    fixture = admission_fixture(name)
    argv = ['--profile', name, '--runs'] + fixture.paths[0] + ['--json', str(fixture.output), '--summary', str(fixture.summary)]
    result = diagnostic.main(argv)
    assert result == diagnostic.build_diagnostics(name, list(reversed(fixture.paths[0])))[0]
    assert len(result['cells']) == count and 'verdict' not in fixture.output.read_text()
    edt = next(c for c in result['cells'] if c['metric'] == 'EDT')
    assert edt['mean'] == pytest.approx((1 + 5.5 / 16 + (.02 if name == 'EPOCH9_K8' else 0)) * 1000)
    assert edt['sd'] == (pytest.approx(np.std(np.arange(5) * 10, ddof=1)) if name == 'EPOCH9_K8' else None)
    assert all(c['estimate'] == pytest.approx(c['k'] / 512) and c['companion_interval'] == pytest.approx((c['k'] / 512,) * 2) and not c['decision_driving'] for c in result['cells'])
    assert 'Descriptive diagnostics' in fixture.summary.read_text() and fixture.sidecar.exists()
    for asset in ('make_results_md', 'make_results_html'):
        module, output = load_asset(asset), tmp_path / (asset + '.out')
        required = sum(([flag, str(record_inputs[profile])] for flag, profile in load_asset('make_results_md').INPUTS), [])
        module.main(required + ['--diag', str(fixture.output), '--out', str(output)])
        assert 'Descriptive diagnostics' in output.read_text()


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'tamper', 'pin', 'producer'])
def test_descriptive_refusals(admission_fixture, mutation):
    fixture = admission_fixture('EPOCH9_K8')
    paths = fixture.paths[0]
    if mutation == 'missing': paths.pop()
    if mutation == 'duplicate': paths[-1] = paths[0]
    if mutation == 'pin': fixture.approved['checkpoints']['aug_epoch9'] = None
    if mutation == 'tamper': Path(paths[0], 'per_sample_yaw.json').write_text('{}')
    if mutation == 'producer': fixture.approved['closures']['producer_descriptive'] = None
    with pytest.raises(ValueError):
        diagnostic.main(['--profile', 'EPOCH9_K8', '--runs'] + paths + ['--json', str(fixture.output), '--summary', str(fixture.summary)])
    assert not fixture.output.exists() and not fixture.summary.exists() and not fixture.sidecar.exists()
