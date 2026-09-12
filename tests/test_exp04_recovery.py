"""Recovery checks operate on stub attempts and real sleeping CPU processes."""
import json
import os
from pathlib import Path
import subprocess

import pytest
from tools import exp04_launcher as launch


def test_recovery_requires_dead_group_and_closed_log(tmp_path):
    log = tmp_path / 'train.log'
    log.write_text('closed')
    child = subprocess.Popen([launch.PYTHON, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        with pytest.raises(ValueError, match='process group'):
            launch.assert_quiescent(child.pid, log)
    finally:
        child.terminate()
        child.wait()
    with log.open('r'):
        launch.assert_quiescent(child.pid, log)
    with log.open('a'):
        moved = log.with_name('renamed.log')
        log.rename(moved)
        with pytest.raises(ValueError, match='log writer'):
            launch.assert_quiescent(child.pid, moved)
    launch.assert_quiescent(child.pid, moved)
    for pgid in (None, 0, -1, True):
        with pytest.raises(ValueError, match='process group'):
            launch.assert_quiescent(pgid, moved)


@pytest.fixture
def preserved_attempt(tmp_path):
    attempt = launch.create_attempt(tmp_path, 'attempt_t')
    expected = launch.effective_args(launch.command('full', str(attempt)), '1', 9261)
    for name in ('args.json', 'effective_args.json'):
        (attempt / name).write_text(json.dumps(expected))
    for index in range(1, 13):
        (attempt / ('epoch_%03d.pth' % index)).write_bytes(b'checkpoint')
    (attempt / 'history.jsonl').write_text('\n'.join(json.dumps(dict(
        epoch=i, train_loss=1, test_loss=1, epoch_minutes=120)) for i in range(1, 13)))
    mutable = {}
    for name in ('control_args', 'probe_receipt', 'data'):
        path = tmp_path / name
        path.write_text('{}')
        mutable[name] = dict(path=str(path), sha256=launch.p.sha256_file(path))
    mutable['effective_args'] = dict(path=str(attempt / 'effective_args.json'),
                                    sha256=launch.p.sha256_file(attempt / 'effective_args.json'))
    log = tmp_path / 'train.log'
    log.write_text('XRIR_RUNTIME_ARGS ' + json.dumps(expected) + '\n'
        'yaw_aug ENABLED W=512 seed=0 counter=(epoch-1)*9261+batch_idx\n'
        'Train Epoch: 1 [0/9261] loss 1\nepoch 1 done\n')
    fields = dict(repo=str(tmp_path), mode='full', effective_args=expected, source_closures={},
        train_data_identity=launch.p._inventory(['data'], tmp_path), mutable_inputs=mutable,
        started_at='2026-09-12T00:00:00+00:00', attempt_path=str(attempt), log_path=str(log))
    digest = launch.p.write_manifest(attempt / 'train_manifest.json', fields)
    child = subprocess.Popen([launch.PYTHON, '-c', 'pass'], start_new_session=True)
    child.wait()
    launch.p.write_completion(attempt / 'execution.json', dict(train_manifest_sha256=digest,
        child_pgid=child.pid, child_exit_status=0, ended_at='2026-09-12T01:00:00+00:00',
        log_sha256=launch.p.sha256_file(log)))
    return attempt, log


@pytest.mark.parametrize('mutation', [None, 'renamed', 'effective_args', 'train_manifest',
                                    'control_args', 'probe_receipt', 'data', 'output', 'log', 'exit'])
def test_finalize_preserved_attempt(preserved_attempt, mutation):
    attempt, log = preserved_attempt
    if mutation == 'renamed':
        new_log = log.with_name('train_ABORTED_input_changed.log')
        log.rename(new_log)
        launch.p.write_completion(attempt / 'abort.json', {'log': {'aborted': str(new_log)}})
        attempt = launch.abort_attempt(attempt, 'input_changed', 1)
    elif mutation in ('effective_args', 'train_manifest'):
        (attempt / (mutation + '.json')).write_text('{}')
    elif mutation in ('control_args', 'probe_receipt', 'data'):
        (attempt.parent / mutation).write_text('changed')
    elif mutation == 'output':
        (attempt / 'epoch_012.pth').unlink()
    elif mutation == 'log':
        log.write_text('changed')
    elif mutation == 'exit':
        path = attempt / 'execution.json'
        launch.p.write_completion(path, dict(json.loads(path.read_text()), child_exit_status=3))
    if mutation not in (None, 'renamed'):
        with pytest.raises((ValueError, KeyError)):
            launch.main(['finalize', str(attempt)])
        assert not (attempt / 'completion.json').exists() and not (attempt.parent / 'final').exists()
    else:
        result = launch.main(['finalize', str(attempt)])
        assert result['wall_hours'] == 1 and len(result['outputs']) >= 14
        assert result['source_drift_after_spawn'] == []
        assert (attempt.parent / 'final').resolve() == attempt
        assert launch.full_hours(attempt.parent) == 1
        with pytest.raises(FileExistsError):
            launch.main(['finalize', str(attempt)])
