"""The pre-registered released-checkpoint calibration, over the synthetic seen layout."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from exp07_fixture import exp07_fixture  # noqa: F401  (fixture)
from tools import exp07_calibration as calibration
from tools import exp07_profiles as profiles
from tools import exp07_table as table
from tools import provenance as p
from tools.paired_compare import REPO

SEEDS = (42, 43, 44, 45, 46)


@pytest.fixture
def calibrated(exp07_fixture, tmp_path, monkeypatch):  # noqa: F811
    """The five released K = 8 runs of the fixture, calibrated against their own scale."""
    built = exp07_fixture()
    runs = list(built.paths[('released_seen', 8)])
    pins = dict(closures=dict(built.pins['closures']), checkpoints={})
    monkeypatch.setattr(calibration, 'closure_pins', lambda commit: pins)
    observed = calibration.measure(runs)
    monkeypatch.setattr(calibration, 'HISTORICAL',
                        {name: cell['mean'] for name, cell in observed.items()})
    def build(directories=None, **changes):
        return calibration.build_calibration(
            runs if directories is None else directories, 'a' * 40,
            profile=changes.get('profile', built.profile), producer=built.producer)
    return SimpleNamespace(built=built, runs=runs, build=build, pins=pins,
                           json=tmp_path / 'CALIBRATION_SEEN_V1.json',
                           sidecar=tmp_path / 'CALIBRATION_SEEN_V1.json.provenance.json')


def test_the_registered_rule_and_its_operands_are_the_plans(calibrated):
    assert profiles.CALIBRATION['historical'] == {'EDT': .0389, 'C50': 1.029, 'T60': 7.27}
    assert (profiles.CALIBRATION['sd_multiple'], profiles.CALIBRATION['relative_margin'],
            profiles.CALIBRATION['ddof'], profiles.CALIBRATION['num_shot']) == (3., .02, 1, 8)
    assert profiles.CALIBRATION['seeds'] == SEEDS and profiles.CALIBRATION_METRICS == (
        'C50', 'EDT', 'T60')
    assert profiles.calibration_tolerance(.5, 2.) == 3 * .5 + .02 * 2
    # The released digest is the registered literal, and it is what the arm carries.
    released = next(arm for arm in profiles.ARMS if arm['reference'])
    assert released['sha256'] == profiles.RELEASED_SHA256 == (
        '1762c702ee23a8f584e67c4b5b052b7483237e6471bb5e58774b66d514aec2f8')
    assert released['checkpoint'] == 'checkpoints/xRIR_seen.pth'


def test_the_closure_pins_come_from_the_reviewed_evaluator_and_writer(monkeypatch):
    """The gate cannot use the all-null approval file, so it pins the reviewed code."""
    asked = []
    def record(files, commit, repo):
        asked.append((tuple(files), commit))
        return ([dict(path=name, reviewed_blob_sha256='a' * 64, working_tree_sha256='a' * 64,
                      commits_after_reviewed=[]) for name in files], files[0])
    monkeypatch.setattr(p, 'source_closure', lambda module, repo: [module])
    monkeypatch.setattr(p, 'closure_record', record)
    pins = calibration.closure_pins('b' * 40)
    assert pins == dict(closures=dict(evaluator='tools.exp07_eval',
                                      writer='tools.exp07_eval_launch'), checkpoints={})
    assert [item[1] for item in asked] == ['b' * 40] * 2


def test_a_drifted_closure_is_refused(monkeypatch):
    monkeypatch.setattr(p, 'source_closure', lambda module, repo: [module])
    monkeypatch.setattr(p, 'closure_record', lambda files, commit, repo: (
        [dict(path=name, reviewed_blob_sha256='a' * 64, working_tree_sha256='b' * 64,
              commits_after_reviewed=[]) for name in files], 'digest'))
    with pytest.raises(ValueError, match='differs from the reviewed commit'):
        calibration.closure_pins('b' * 40)


def test_the_five_seed_means_and_sds_come_from_the_runs(calibrated):
    result, _ = calibrated.build()
    assert result['profile_name'] == 'CALIBRATION_SEEN_V1' and result['passed'] is True
    assert (result['role'], result['protocol'], result['num_shot']) == ('released_seen', 'seen', 8)
    assert sorted(result['metrics']) == ['C50', 'EDT', 'T60']
    assert result['seeds'] == list(SEEDS)
    for metric, cell in result['metrics'].items():
        source = profiles.CALIBRATION['sources'][metric]
        expected = []
        for directory in sorted(calibrated.runs):
            sample = json.loads((Path(directory) / 'per_sample_yaw.json').read_text())
            values = np.asarray(sample['P']['0'][source], dtype=float)
            expected.append(float(values[np.isfinite(values)].mean()))
            assert cell['per_seed'][str(sample['meta']['manifest_seed'])] in expected
        assert cell['mean'] == float(np.mean(expected))
        assert cell['sd'] == float(np.std(expected, ddof=1))
        assert cell['tolerance'] == profiles.calibration_tolerance(cell['sd'], cell['historical'])
        assert cell['n_seeds'] == 5 and cell['unit'] in ('s', 'dB', '%')
        assert cell['passed'] is True


def test_the_binders_recomputation_route_agrees_bit_for_bit(calibrated):
    result, _ = calibrated.build()
    assert calibration.measure(calibrated.runs) == result['metrics']


def test_a_mean_outside_the_registered_tolerance_does_not_pass(calibrated, monkeypatch):
    shifted = dict(calibration.HISTORICAL)
    shifted['C50'] = shifted['C50'] * 2 + 1
    monkeypatch.setattr(calibration, 'HISTORICAL', shifted)
    result, _ = calibrated.build()
    assert result['passed'] is False and result['metrics']['C50']['passed'] is False
    assert result['metrics']['EDT']['passed'] is True


def test_the_run_set_must_be_the_five_registered_seeds(calibrated):
    with pytest.raises(ValueError, match='five registered evaluation seeds'):
        calibrated.build(calibrated.runs[:4])


def test_a_duplicate_run_directory_is_refused(calibrated):
    with pytest.raises(ValueError, match='duplicate run directories'):
        calibrated.build(calibrated.runs + calibrated.runs[:1])


def test_a_run_of_the_other_k_is_refused(calibrated):
    others = list(calibrated.built.paths[('released_seen', 1)])
    with pytest.raises(ValueError, match='K = 8'):
        calibrated.build(calibrated.runs[:4] + others[:1])


def test_a_trained_arms_run_is_refused(calibrated):
    others = list(calibrated.built.paths[('seen_simple', 8)])
    with pytest.raises(ValueError, match='released checkpoint'):
        calibrated.build(calibrated.runs[:4] + others[:1])


@pytest.mark.parametrize('fault,message', [
    ('split', 'split identity'), ('evaluator', 'unapproved evaluator'),
    ('checkpoint_digest', 'released checkpoint digest'), ('training', 'training provenance')])
def test_the_admission_is_the_table_producers(calibrated, fault, message):
    """The released row's completion/manifest/output checks are run_contract's, unchanged."""
    run = Path(calibrated.runs[0])
    built = calibrated.built
    fields = built.read(run / 'eval_manifest.json')
    if fault == 'split':
        fields['split'] = 'unseen'
    elif fault == 'evaluator':
        fields['source_closures']['entrypoint']['sha256'] = 'f' * 64
    elif fault == 'checkpoint_digest':
        fields['checkpoint_sha256'] = 'f' * 64
    else:
        donor = built.attempts['seen_simple'] / 'train_manifest.json'
        fields['mutable_inputs']['train_manifest'] = dict(path=str(donor),
                                                          sha256=p.sha256_file(donor))
    built.rebind(run, manifest=fields)
    with pytest.raises(ValueError, match=message):
        calibrated.build()


def test_publication_is_exclusive_and_binds_its_inputs(calibrated):
    result, admitted = calibrated.build()
    digest = calibration.write_outputs(result, admitted, str(calibrated.json),
                                       command=['exp07_calibration.py'])
    assert json.loads(calibrated.json.read_text()) == result
    side = json.loads(calibrated.sidecar.read_text())
    assert side['outputs'] == {str(calibrated.json): digest} == {str(calibrated.json):
                                                                 p.sha256_file(calibrated.json)}
    assert side['inputs'] == result['inputs'] and side['producer'] == calibrated.built.producer
    assert sorted(side['run_flags']) == sorted(str(Path(item).resolve())
                                               for item in calibrated.runs)
    assert side['profile_digest'] == result['profile_digest']
    with pytest.raises(FileExistsError):
        calibration.write_outputs(result, admitted, str(calibrated.json))


def test_an_input_changed_before_publication_stops_the_transaction(calibrated):
    result, admitted = calibrated.build()
    victim = Path(calibrated.runs[0]) / 'metrics_yaw.json'
    victim.write_text(victim.read_text() + '\n')
    with pytest.raises(ValueError, match='input changed before publication'):
        calibration.write_outputs(result, admitted, str(calibrated.json))
    assert not calibrated.json.exists() and not calibrated.sidecar.exists()


def test_the_producer_module_is_the_one_the_record_reads():
    assert Path(calibration.__file__).resolve() == REPO / 'tools/exp07_calibration.py'


def strip_metrics(calibrated, names=('loss', 'log_mse')):
    """Remove metric columns the table producer requires from every released run."""
    for directory in calibrated.runs:
        run = Path(directory)
        sample = calibrated.built.read(run / 'per_sample_yaw.json')
        metrics = calibrated.built.read(run / 'metrics_yaw.json')
        for name in names:
            sample['P']['0'].pop(name)
            metrics['P']['0'].pop(name)
        calibrated.built.rebind(run, sample=sample, metrics=metrics)


def test_the_gate_applies_the_tables_metric_coverage_rule(calibrated):
    """Blocker 4: a cohort the publication protocol refuses cannot pass this gate."""
    strip_metrics(calibrated)
    with pytest.raises(ValueError, match='missing metrics'):
        calibrated.build()
    with pytest.raises(ValueError, match='missing metrics'):
        calibration.measure(calibrated.runs)
    with pytest.raises(ValueError, match='missing metrics'):
        table.build_table(calibrated.built.directories, calibrated.built.profile,
                          calibrated.built.approved, producer=calibrated.built.producer)


def test_the_gate_applies_the_tables_finite_count_tolerance(calibrated):
    run = Path(calibrated.runs[0])
    sample = calibrated.built.read(run / 'per_sample_yaw.json')
    sample['P']['0']['c50'][:4] = [None] * 4
    metrics = calibrated.built.read(run / 'metrics_yaw.json')
    metrics['P'] = calibrated.built.summaries(sample['P'])
    calibrated.built.rebind(run, sample=sample, metrics=metrics)
    with pytest.raises(ValueError, match='finite count tolerance exceeded'):
        calibrated.build()
    with pytest.raises(ValueError, match='finite count tolerance exceeded'):
        calibration.measure(calibrated.runs)


def test_the_shared_checks_never_read_the_trained_arm_approval_pins(calibrated, monkeypatch):
    """The gate runs before the trainings finish, so the approval file is still all-null."""
    def never(*args, **kwargs):
        raise AssertionError('the calibration must not read the approval pins')
    monkeypatch.setattr(table, 'load_approved_digests', never)
    result, _ = calibrated.build()
    assert result['passed'] is True
    assert calibration.measure(calibrated.runs) == result['metrics']
