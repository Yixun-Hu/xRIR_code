"""The exp_05 record tooling: canonical admission, the generators, the binder."""
import json
from pathlib import Path

import pytest

from exp05_record_fixture import exp05_record_fixture
from tools import exp05_record as record
from tools import provenance as p

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
