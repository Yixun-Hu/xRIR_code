"""The training-gate fixes of the Codex ``full_train`` review (findings 1-6, nit 8).

These cases live apart from ``tests/test_exp06_finalize.py`` so the gate work and the
concurrent round-2b HAA work touch different files. The fixtures are the ones that file
already builds; pytest resolves them by name once they are imported here.
"""
import json
import os
from pathlib import Path

import pytest
import torch

from test_exp06_finalize import (DEAD_PID, STATE, approvals, bound_args, clone,  # noqa: F401
                                 data_root, dead_pid, full_args, full_run, history_rows,
                                 provenance_record, seal)
from tools import exp06_finalize, provenance


def test_epoch_checkpoint_must_agree_in_dtype_with_last_pth(full_run, clone):
    """Nit 8: torch.equal accepts equal values across dtypes; a float64 copy must not pass."""
    run, log = full_run
    torch.save({key: value.double() for key, value in STATE.items()}, run / 'epoch_012.pth')
    with pytest.raises(ValueError, match='dtype'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


def test_epoch_checkpoint_must_agree_in_shape_with_last_pth(full_run, clone):
    """Nit 8: a reshaped tensor of the same values is a different checkpoint."""
    run, log = full_run
    torch.save({key: (value.reshape(3, 2) if value.dim() == 2 else value)
                for key, value in STATE.items()}, run / 'epoch_012.pth')
    with pytest.raises(ValueError, match='shape'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


def test_a_receipt_naming_a_live_child_is_refused(full_run, clone):
    """Finding 3: a partially restored attempt must not certify while its child runs."""
    run, log = full_run
    receipt = json.loads((run / 'child_exit.json').read_text())
    receipt['child_pid'] = os.getpid()
    (run / 'child_exit.json').write_text(json.dumps(receipt, sort_keys=True))
    with pytest.raises(ValueError, match='alive'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


def test_the_receipt_and_the_child_pid_sidecar_must_agree(full_run, clone):
    """Finding 3: the owner exception covers launch.pid only, never the child."""
    run, log = full_run
    (run / 'child.pid').write_text('{}\n'.format(dead_pid(1)))
    with pytest.raises(ValueError, match='child.pid'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()
    (run / 'child.pid').write_text('{}\n'.format(DEAD_PID))
    fields = exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert fields['child_exit_receipt']['child_pid'] == DEAD_PID


def test_a_full_completion_records_the_approvals_it_matched(full_run, clone, approvals):
    """Finding 1: the training-critical identities, approved and recomputed at the end."""
    from tools import exp06_profiles
    run, log = full_run
    fields = exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert fields['approvals']['path'] == str(approvals)
    assert fields['approvals']['sha256'] == provenance.sha256_file(approvals)
    assert set(fields['code_digests']) == set(exp06_profiles.TRAINING_KEYS)


def test_null_approvals_refuse_a_full_finalisation(full_run, clone, tmp_path):
    """Finding 1: nothing is admissible before the second reviewed commit fills them."""
    from tools import exp06_profiles
    run, log = full_run
    empty = tmp_path / 'approved_digests.json'
    empty.write_bytes(exp06_profiles.TEMPLATE_PATH.read_bytes())
    record = json.loads((run / 'provenance.json').read_text())
    record['approvals'] = {'path': str(empty), 'schema_version': 1,
                           'sha256': provenance.sha256_file(empty)}
    (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    (run / 'args.json').write_text(json.dumps(
        dict(json.loads((run / 'args.json').read_text())), indent=2))
    with pytest.raises(ValueError, match='not approved'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


@pytest.mark.parametrize('orchestrator', ['tools/exp06_launch.sh', 'tools/exp06_finalize.py'])
def test_orchestration_bytes_that_changed_after_provenance_are_refused(full_run, clone,
                                                                      orchestrator):
    """The review's reproduction: the launcher and the finalizer were never bound."""
    run, log = full_run
    path = Path(clone) / orchestrator
    original = path.read_bytes()
    path.write_bytes(original + b'\n# changed after the run started\n')
    try:
        with pytest.raises(ValueError, match='drift'):
            exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
        assert not (run / 'completion.json').exists()
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize('damage,cause', [('no_approvals', 'approvals'),
                                          ('approvals_rehashed', 'approvals'),
                                          ('no_code_digests', 'code_digests'),
                                          ('code_digest_changed', 'drift'),
                                          ('no_orchestration', 'orchestration'),
                                          ('exploratory', 'exploratory')])
def test_the_training_admission_record_must_be_complete(full_run, clone, damage, cause):
    run, log = full_run
    record = json.loads((run / 'provenance.json').read_text())
    if damage == 'no_approvals':
        record.pop('approvals')
    elif damage == 'approvals_rehashed':
        record['approvals'] = dict(record['approvals'], sha256='b' * 64)
    elif damage == 'no_code_digests':
        record.pop('code_digests')
    elif damage == 'code_digest_changed':
        record['code_digests']['recipe'] = 'c' * 64
    elif damage == 'no_orchestration':
        record.pop('orchestration_closures')
    else:
        record['exploratory'] = True
    (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


def test_geometry_membership_is_derived_from_the_datasets_own_split(data_root):
    """Finding 2: the metadata and depth maps every query and reference of a split reads."""
    from tools import exp06_train
    files, counts = exp06_train.geometry_paths(data_root)
    assert counts == {'train': 2, 'test': 1}
    assert set(files) == {
        'metadata/Apartments/Apartments_idx_1/S000_R000.json',
        'metadata/Apartments/Apartments_idx_1/S001_R000.json',
        'metadata/Bathrooms/Bathrooms_idx_18/S000_R000.json',
        'depth_map/Apartments/Apartments_idx_1/0.npy',
        'depth_map/Bathrooms/Bathrooms_idx_18/0.npy'}
    identity = exp06_train.geometry_identity(str(data_root))
    assert identity['inventory_files'] == 5 and identity['splits'] == ['train', 'test']
    assert identity['split_files'] == counts
    assert {entry['path'] for entry in identity['inventory']} == set(files)


def test_a_full_completion_rehashes_every_geometry_input(full_run, clone, data_root):
    run, log = full_run
    fields = exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert fields['geometry_files'] == 5
    assert fields['geometry_rehash_seconds'] >= 0


@pytest.mark.parametrize('changed', ['metadata/Apartments/Apartments_idx_1/S000_R000.json',
                                     'depth_map/Bathrooms/Bathrooms_idx_18/0.npy'])
def test_a_geometry_input_that_changed_after_provenance_is_refused(full_run, clone, data_root,
                                                                   changed):
    """The review's reproduction: a changed source position passed the WAV inventory."""
    run, log = full_run
    (Path(data_root) / changed).write_bytes(b'a different room')
    with pytest.raises(ValueError, match='geometry'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


@pytest.mark.parametrize('damage', ['absent', 'short', 'wrong_root'])
def test_an_incomplete_geometry_inventory_is_refused(full_run, clone, data_root, damage):
    run, log = full_run
    record = json.loads((run / 'provenance.json').read_text())
    if damage == 'absent':
        record.pop('geometry_identity')
    elif damage == 'wrong_root':
        record['geometry_identity']['data_root'] = str(Path(data_root).parent)
    else:
        identity = record['geometry_identity']
        identity['inventory'] = identity['inventory'][:-1]
        identity['inventory_files'] = len(identity['inventory'])
        identity['inventory_bytes'] = sum(entry['size'] for entry in identity['inventory'])
        identity['inventory_sha256'] = exp06_finalize.inventory_digest(identity['inventory'])
    (run / 'provenance.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    with pytest.raises(ValueError, match='geometry'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


from test_exp06_finalize import diagnostic_run  # noqa: E402,F401

DIAGNOSTIC_FIELDS = ('runner', 'runner_closure_sha256', 'entry', 'argv', 'run_type',
                     'started_at', 'ended_at', 'wall_s', 'peak_bytes', 'alarm_seconds',
                     'max_gb', 'outcome', 'exit_status', 'exploratory', 'git_head')


def test_a_diagnostic_completion_records_its_budgets_and_peak(tmp_path, clone, approvals):
    """Finding 6: timing and memory evidence, not a bare exit status."""
    run = tmp_path / 'probe'
    log = seal(run, tmp_path / 'probe.log')
    _, receipt = diagnostic_run(run, 'probe', clone, approvals)
    fields = exp06_finalize.finalize(run, 'probe', log, 0, repo=clone, receipt=receipt)
    assert fields['passed'] is True and fields['admissible_arm'] is False
    assert fields['receipt']['wall_s'] == 12.5 and fields['receipt']['peak_bytes'] == 0
    assert fields['receipt']['alarm_seconds'] == 300.0 and fields['receipt']['max_gb'] == 3.0
    assert fields['receipt']['runner'] == 'tools.exp06_smoke'
    assert fields['run_type'] == 'probe' and fields['exploratory'] is False


def test_the_reviews_minimal_receipt_is_refused(tmp_path, clone, approvals):
    """The review's reproduction: {diagnostic, argv, exit_status} produced passed: true."""
    run = tmp_path / 'smoke'
    log = seal(run, tmp_path / 'smoke.log')
    diagnostic_run(run, 'smoke', clone, approvals)
    receipt = tmp_path / 'minimal.json'
    receipt.write_text(json.dumps({'diagnostic': True, 'argv': ['--no-save'],
                                   'exit_status': 0}) + '\n')
    with pytest.raises(ValueError, match='receipt'):
        exp06_finalize.finalize(run, 'smoke', log, 0, repo=clone, receipt=receipt)
    assert not (run / 'completion.json').exists()


@pytest.mark.parametrize('field', DIAGNOSTIC_FIELDS)
def test_every_missing_receipt_field_is_refused(tmp_path, clone, approvals, field):
    run = tmp_path / 'smoke'
    log = seal(run, tmp_path / 'smoke.log')
    _, receipt = diagnostic_run(run, 'smoke', clone, approvals)
    record = json.loads(Path(receipt).read_text())
    record.pop(field)
    Path(receipt).write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    with pytest.raises(ValueError, match=field.replace('_', '.')):
        exp06_finalize.finalize(run, 'smoke', log, 0, repo=clone, receipt=receipt)
    assert not (run / 'completion.json').exists()


@pytest.mark.parametrize('damage,cause', [('runner', 'runner'), ('closure', 'runner'),
                                          ('head', 'git_head'), ('run_type', 'run_type'),
                                          ('outcome', 'outcome'), ('wall', 'wall_s'),
                                          ('peak', 'peak_bytes'), ('no_provenance', 'provenance')])
def test_a_diagnostic_receipt_must_agree_with_its_provenance(tmp_path, clone, approvals,
                                                             damage, cause):
    run = tmp_path / 'smoke'
    log = seal(run, tmp_path / 'smoke.log')
    _, receipt = diagnostic_run(run, 'smoke', clone, approvals)
    record = json.loads(Path(receipt).read_text())
    if damage == 'runner':
        record['runner'] = 'tools.exp06_train'
    elif damage == 'closure':
        record['runner_closure_sha256'] = 'e' * 64
    elif damage == 'head':
        record['git_head'] = 'f' * 40
    elif damage == 'run_type':
        record['run_type'] = 'probe'
    elif damage == 'outcome':
        record['outcome'] = 'memory'
    elif damage == 'wall':
        record['wall_s'] = -1.0
    elif damage == 'peak':
        record['peak_bytes'] = -1
    else:
        (run / 'provenance.json').unlink()
    Path(receipt).write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(run, 'smoke', log, 0, repo=clone, receipt=receipt)
