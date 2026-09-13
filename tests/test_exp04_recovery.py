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
def preserved_attempt(tmp_path, monkeypatch):
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
    identity = launch.p._inventory(['data'], tmp_path)
    sidecar = attempt / 'train_inventory.json'
    identity['inventory_file'] = dict(path=str(sidecar),
        sha256=launch.p.write_manifest(sidecar, {'inventory': identity.pop('inventory')}))
    mutable['train_inventory'] = identity['inventory_file']
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    paths = sorted(launch.TRAIN_MINIMUM | {'tools/exp04_launcher.py'})
    for name in paths:
        source = tmp_path / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('# reviewed fixture\n')
    for argv in (['init', '-q'], ['add', *paths], ['-c', 'user.name=Test', '-c',
            'user.email=test@example.com', 'commit', '-qm', 'fixture']):
        subprocess.run(['git', *argv], cwd=tmp_path, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tmp_path, text=True).strip()
    records, sha = launch.p.closure_record(paths, commit, tmp_path)
    identity['inventory_files'] = 296334
    fields = dict(repo=str(tmp_path), mode='full', effective_args=expected, source_closures={},
        train_data_identity=identity, mutable_inputs=mutable,
        started_at='2026-09-12T00:00:00+00:00', attempt_path=str(attempt), log_path=str(log))
    fields.update(reviewed_commit=commit, command=launch.command('full', str(attempt)),
        env={k: expected[k] for k in launch.ENV_KEYS}, resource_before={'gpu': '1'},
        source_closures={role: dict(files=records, sha256=sha) for role in ('training', 'launcher')})
    digest = launch.p.write_manifest(attempt / 'train_manifest.json', fields)
    child = subprocess.Popen([launch.PYTHON, '-c', 'pass'], start_new_session=True)
    child.wait()
    spawn = dict(train_manifest_sha256=digest, child_pgid=child.pid, child_exit_status=None)
    spawn['spawn_execution_sha256'] = launch.p.hashlib.sha256((
        json.dumps(spawn, sort_keys=True, indent=2) + '\n').encode()).hexdigest()
    (tmp_path / 'launcher.log').write_text('EXP04_SPAWN ' + json.dumps(spawn, sort_keys=True) + '\n')
    launch.p.write_completion(attempt / 'execution.json', dict(train_manifest_sha256=digest,
        child_pgid=child.pid, child_exit_status=0, ended_at='2026-09-12T01:00:00+00:00',
        log_sha256=launch.p.sha256_file(log), spawn_execution_sha256=spawn['spawn_execution_sha256']))
    return attempt, log


@pytest.mark.parametrize('mutation', [None, 'renamed', 'effective_args', 'train_manifest',
                                    'control_args', 'probe_receipt', 'train_inventory', 'data', 'output', 'log', 'exit'])
def test_finalize_preserved_attempt(preserved_attempt, mutation):
    attempt, log = preserved_attempt
    if mutation == 'renamed':
        new_log = log.with_name('train_ABORTED_input_changed.log')
        log.rename(new_log)
        launch.p.write_completion(attempt / 'abort.json', {'log': {'aborted': str(new_log)}})
        attempt = launch.abort_attempt(attempt, 'input_changed', 1.5)
    elif mutation in ('effective_args', 'train_manifest', 'train_inventory'):
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
            launch.main(['finalize', str(attempt), '--launcher-log', str(attempt.parent / 'launcher.log')])
        assert not (attempt / 'completion.json').exists() and not (attempt.parent / 'final').exists()
    else:
        previous_name = attempt.name
        result = launch.main(['finalize', str(attempt), '--launcher-log', str(attempt.parent / 'launcher.log')])
        if mutation == 'renamed':
            attempt = attempt.with_name('attempt_t')
            recovery = result['recovered_from']
            assert recovery['attempt'] == previous_name and recovery['recovered_at']
            assert recovery['log'] == str(new_log) and not new_log.exists()
            assert result['log'] == dict(path=str(log), sha256=launch.p.sha256_file(log))
            assert recovery['abort'] == json.loads((attempt / 'abort.json').read_text())
            assert launch.hours_record(attempt.parent)['attempts'][0]['attempt'] == 'attempt_t'
        hours = 1.5 if mutation == 'renamed' else 1
        assert result['wall_hours'] == hours and len(result['outputs']) >= 14
        assert result['directory_listing'] == {path.name: launch.p.sha256_file(path)
            for path in attempt.iterdir() if path.name != 'completion.json'}
        assert result['source_drift_after_spawn'] == []
        assert (attempt.parent / 'final').resolve() == attempt
        assert launch.full_hours(attempt.parent) == hours
        with pytest.raises(FileExistsError):
            launch.main(['finalize', str(attempt), '--launcher-log', str(attempt.parent / 'launcher.log')])


@pytest.mark.parametrize('name', ['control_args', 'probe_receipt'])
def test_full_finalization_requires_authorizing_bindings(preserved_attempt, name):
    attempt, log = preserved_attempt
    fields = json.loads((attempt / 'train_manifest.json').read_text())
    del fields['mutable_inputs'][name]
    with pytest.raises(ValueError, match=name):
        launch.complete_attempt(attempt, 'full', log, fields, 'digest', {}, 1)
    assert not (attempt / 'completion.json').exists()


@pytest.mark.parametrize('status,stdout,stderr', [(2, '', 'failed'), (0, '123', ''),
                                               (1, '', 'Permission denied')])
def test_failed_writer_inspection_refuses_recovery(preserved_attempt, monkeypatch, status, stdout, stderr):
    from types import SimpleNamespace
    attempt, log = preserved_attempt
    pgid = json.loads((attempt / 'execution.json').read_text())['child_pgid']
    monkeypatch.setattr(launch.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(returncode=status, stdout=stdout, stderr=stderr))
    with pytest.raises(ValueError, match='inspect log writers'):
        launch.assert_quiescent(pgid, log)


@pytest.mark.parametrize('key,value', [('repo', '/tmp'), ('source_closures', {}),
    ('reviewed_commit', 'fake'), ('command', []), ('env', {}), ('resource_before', None),
    ('mutable_inputs', {}), ('train_data_identity', {'inventory_files': 1}), ('child_pgid', 1)])
def test_fabricated_recovery_refused(preserved_attempt, key, value):
    attempt, _ = preserved_attempt
    fields = json.loads((attempt / 'train_manifest.json').read_text())
    execution = json.loads((attempt / 'execution.json').read_text())
    if key == 'child_pgid':
        execution[key] = value
    else:
        fields[key] = value
        (attempt / 'train_manifest.json').write_text(json.dumps(fields))
        execution['train_manifest_sha256'] = launch.p.sha256_file(attempt / 'train_manifest.json')
    if key != 'child_pgid':
        spawn = dict(execution, child_exit_status=None)
        for field in ('ended_at', 'log_sha256', 'spawn_execution_sha256'):
            spawn.pop(field)
        spawn['spawn_execution_sha256'] = launch.p.hashlib.sha256((
            json.dumps(spawn, sort_keys=True, indent=2) + '\n').encode()).hexdigest()
        execution['spawn_execution_sha256'] = spawn['spawn_execution_sha256']
        (attempt.parent / 'launcher.log').write_text('EXP04_SPAWN ' + json.dumps(spawn, sort_keys=True) + '\n')
    launch.p.write_completion(attempt / 'execution.json', execution)
    with pytest.raises(ValueError):
        launch.finalize_attempt(attempt, attempt.parent / 'launcher.log')
    assert not (attempt / 'completion.json').exists()


def test_recovery_requires_external_log_and_pinned_python(preserved_attempt, monkeypatch):
    attempt, _ = preserved_attempt
    with pytest.raises(ValueError, match='launcher-log'):
        launch.finalize_attempt(attempt)
    monkeypatch.setattr(launch.sys, 'executable', '/usr/bin/python')
    with pytest.raises(ValueError, match='pinned interpreter'):
        launch.finalize_attempt(attempt, attempt.parent / 'launcher.log')


@pytest.mark.parametrize('failure', ['occupied', 'occupied_log', 'log_rename', 'promotion', 'after_promotion', 'replace_final'])
def test_recovery_rename_failure_preserves_abort(preserved_attempt, monkeypatch, failure):
    original, log = preserved_attempt
    aborted_log = log.with_name('train_ABORTED_interrupted.log')
    log.rename(aborted_log)
    launch.p.write_completion(original / 'abort.json', {'log': {'aborted': str(aborted_log)}})
    attempt = launch.abort_attempt(original, 'interrupted', 1.5)
    if failure == 'occupied':
        original.mkdir()
        (original / 'foreign').touch()
    elif failure == 'occupied_log':
        log.write_text('foreign')
    elif failure == 'log_rename':
        rename = Path.rename
        def fail_log(path, destination):
            if path == aborted_log:
                raise OSError('log rename failed')
            return rename(path, destination)
        monkeypatch.setattr(Path, 'rename', fail_log)
    else:
        promote = launch.promote
        if failure == 'replace_final':
            (attempt.parent / 'final').symlink_to('previous')
        def fail(path):
            if failure in ('after_promotion', 'replace_final'):
                promote(path)
            raise OSError('promotion failed')
        monkeypatch.setattr(launch, 'promote', fail)
    with pytest.raises(OSError):
        launch.finalize_attempt(attempt, attempt.parent / 'launcher.log')
    assert attempt.exists() and not (attempt / 'completion.json').exists()
    assert aborted_log.exists() and (log.read_text() == 'foreign' if failure == 'occupied_log' else not log.exists())
    if failure == 'replace_final':
        assert os.readlink(attempt.parent / 'final') == 'previous'
    else:
        assert not os.path.lexists(attempt.parent / 'final')
    assert launch.hours_record(attempt.parent)['attempts'] == [dict(attempt=attempt.name, hours=1.5, mode='full')]


def test_recovery_refuses_uppercase_reviewed_commit(preserved_attempt):
    attempt, _ = preserved_attempt
    fields = json.loads((attempt / 'train_manifest.json').read_text())
    fields['reviewed_commit'] = fields['reviewed_commit'].upper()
    execution = json.loads((attempt / 'execution.json').read_text())
    with pytest.raises(ValueError, match='invalid reviewed_commit'):
        launch.recovery_evidence(fields, execution, attempt, attempt.parent / 'launcher.log')


@pytest.mark.parametrize('boundary', ['directory', 'log'])
def test_recovery_defers_signal_through_renames(preserved_attempt, monkeypatch, boundary, capsys):
    original, log = preserved_attempt
    aborted_log = log.with_name('train_ABORTED_interrupted.log')
    log.rename(aborted_log)
    launch.p.write_completion(original / 'abort.json', {'log': {'aborted': str(aborted_log)}})
    attempt = launch.abort_attempt(original, 'interrupted', 1.5)
    rename = Path.rename
    def terminate(path, destination):
        result = rename(path, destination)
        if path == (attempt if boundary == 'directory' else aborted_log):
            os.kill(os.getpid(), launch.signal.SIGTERM)
        return result
    monkeypatch.setattr(Path, 'rename', terminate)
    with pytest.raises(launch.LauncherTerminated):
        launch.finalize_attempt(attempt, attempt.parent / 'launcher.log')
    assert 'CERTIFIED ' + str(original) in capsys.readouterr().out.splitlines()
    result = json.loads((original / 'completion.json').read_text())
    assert result['recovered_from']['log'] == str(aborted_log)
    assert log.is_file() and not aborted_log.exists() and not attempt.exists()
    assert (original.parent / 'final').resolve() == original
