"""CPU guards for the exp_04 training launcher."""
import json
import os
from pathlib import Path
import pytest
from tools import exp04_launcher as launch


@pytest.mark.parametrize('mode', ['smoke', 'full'])
def test_golden_argv_and_one_flag_refusal(mode, tmp_path):
    command = launch.command(mode, '<attempt>')
    launch.check_golden(command, mode, '<attempt>')
    command[command.index('--lr') + 1] = '0.002'
    with pytest.raises(ValueError, match='golden'):
        launch.check_golden(command, mode, '<attempt>')


def test_normalized_runtime_schema_and_mismatch():
    args = launch.effective_args(launch.command('full', 'ckpt/attempt'), '1', 9261)
    assert args['resume'] is None and args['no_save'] is False
    assert args['yaw_aug_seed'] == 0 and args['train_batches_per_epoch'] == 9261
    assert args['CUDA_VISIBLE_DEVICES'] == '1'
    runtime = {k: v for k, v in args.items() if k not in launch.ENV_KEYS}
    runtime['env'] = {k: args[k] for k in launch.ENV_KEYS}
    assert launch.normalize(runtime) == args
    launch.check_runtime(runtime, args, 'full')
    for key, value in [('lr', '0.001'), ('yaw_aug', True), ('train_batches_per_epoch', 9260)]:
        changed = dict(runtime, **{key: value})
        with pytest.raises(ValueError, match=key):
            launch.check_runtime(changed, args, 'full')
    runtime['unexpected'] = 1
    with pytest.raises(ValueError, match='unexpected'):
        launch.check_runtime(runtime, args, 'full')


def test_full_refuses_matching_but_wrong_bpe():
    args = launch.effective_args(launch.command('full', 'ckpt/attempt'), '1', 9260)
    with pytest.raises(ValueError, match='9261'):
        launch.check_runtime(args, args, 'full')


@pytest.mark.parametrize('free,apps,disk,allow,mode,refused', [
    (40, '', 50, False, 'full', False), (39.99, '', 50, False, 'full', True),
    (40, '123, train.py, 11000', 50, False, 'smoke', True),
    (30, '123, train.py, 11000', 50, True, 'smoke', False),
    (30, '123, train.py, 11000', 50, True, 'probe', False),
    (40, '', 50, True, 'full', True), (40, '', 49.99, True, 'smoke', True)])
def test_resource_thresholds(monkeypatch, tmp_path, free, apps, disk, allow, mode, refused):
    def output(argv, **kwargs):
        if argv[0] == 'df':
            return 'Avail\n' + str(int(disk * 2**30))
        if '--query-compute-apps=pid,process_name,used_gpu_memory' in argv:
            return apps
        assert '-i' in argv and argv[argv.index('-i') + 1] == '1'
        return 'GPU-test, {}, 91'.format(free * 1024)
    monkeypatch.setattr(launch.subprocess, 'check_output', output)
    if refused:
        with pytest.raises(ValueError):
            launch.resource_gate('1', tmp_path, mode, allow)
    else:
        state = launch.resource_gate('1', tmp_path, mode, allow)
        assert state['compute_apps'] == apps and state['utilization_gpu'] == 91


def test_exclusive_attempt_abort_account_and_promote(tmp_path):
    attempt = launch.create_attempt(tmp_path, 'attempt_t')
    with pytest.raises(FileExistsError):
        launch.create_attempt(tmp_path, 'attempt_t')
    (attempt / 'train_manifest.json').write_text('{}')
    aborted = launch.abort_attempt(attempt, 'child_failed', 2.5)
    assert aborted.name == 'attempt_t_ABORTED_child_failed'
    assert (aborted / 'train_manifest.json').exists()
    assert not attempt.exists()
    record = json.loads((tmp_path / 'cumulative_hours.json').read_text())
    assert record['total_hours'] == 2.5 and record['attempts'][0]['hours'] == 2.5
    launch.check_budget(tmp_path, 40.5)
    with pytest.raises(ValueError, match='43'):
        launch.check_budget(tmp_path, 40.5001)
    complete = launch.create_attempt(tmp_path, 'attempt_done')
    (complete / 'completion.json').write_text('{}')
    launch.promote(complete)
    assert os.readlink(tmp_path / 'final') == 'attempt_done'
    assert (tmp_path / 'final').resolve() == complete


@pytest.mark.parametrize('projection', [-1, float('nan'), float('inf')])
def test_invalid_budget_projection(tmp_path, projection):
    with pytest.raises(ValueError):
        launch.check_budget(tmp_path, projection)


@pytest.fixture
def fake_manifest_inputs(monkeypatch):
    monkeypatch.setattr(launch.p, 'source_closure', lambda module, repo: sorted(launch.TRAIN_MINIMUM))
    monkeypatch.setattr(launch.p, 'closure_record', lambda files, commit, repo: ([
        dict(path=name, reviewed_blob_sha256='same', working_tree_sha256='same',
             commits_after_reviewed=[]) for name in files], 'closure'))
    monkeypatch.setattr(launch.p, 'environment', lambda: {'executable': launch.PYTHON})
    monkeypatch.setattr(launch.p, 'git_state', lambda repo: {'HEAD': 'commit', 'dirty_outside_worklog': False})
    def inventory(root, cache_path):
        assert cache_path == launch.REPO / 'ckpt/yaw_aug/train_inventory.json'
        return {'inventory_files': 296334}
    monkeypatch.setattr(launch.p, 'train_data_identity', inventory)


def test_manifest_binds_import_closure_data_args_environment(fake_manifest_inputs):
    command = launch.command('smoke', 'attempt')
    fields = launch.build_fields(command, '1', 'commit', 'smoke')
    assert fields['effective_args']['train_batches_per_epoch'] == 74084
    assert fields['command'] == command and fields['reviewed_commit'] == 'commit'
    assert fields['train_data_identity']['inventory_files'] == 296334
    assert fields['env']['CUDA_VISIBLE_DEVICES'] == '1'
    assert {r['path'] for r in fields['source_closures']['training']['files']} >= launch.TRAIN_MINIMUM


@pytest.mark.parametrize('mode,allow', [('full', False), ('full', True), ('smoke', False), ('probe', False)])
def test_dirty_training_gate(fake_manifest_inputs, tmp_path, monkeypatch, capsys, mode, allow):
    state = {'dirty_outside_worklog': True}
    monkeypatch.setattr(launch.p, 'git_state', lambda repo: state)
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    control = tmp_path / 'ckpt/xRIR_simple_8_shot/args.json'
    control.parent.mkdir(parents=True)
    control.write_text('{}')
    monkeypatch.setattr(launch, 'compare_control', lambda *a: {})
    command = launch.command('full' if mode == 'full' else 'smoke', 'attempt')
    if mode == 'full' and not allow:
        with pytest.raises(ValueError, match='dirty_outside_worklog'):
            launch.build_fields(command, '1', 'commit', mode, allow)
    else:
        fields = launch.build_fields(command, '1', 'commit', mode, allow)
        assert fields['git_state'] == state and fields['allow_dirty'] == allow
        assert 'WARNING' in capsys.readouterr().out


@pytest.mark.parametrize('failure', ['missing', 'modified', 'unreviewed', 'later'])
def test_training_closure_refusal(fake_manifest_inputs, monkeypatch, failure):
    if failure == 'missing':
        monkeypatch.setattr(launch.p, 'source_closure', lambda *a: ['train_xRIR_backbone.py'])
    else:
        record = dict(path='trainer', reviewed_blob_sha256='same', working_tree_sha256='same',
                      commits_after_reviewed=[])
        record[{'modified': 'working_tree_sha256', 'unreviewed': 'reviewed_blob_sha256',
                'later': 'commits_after_reviewed'}[failure]] = None if failure == 'unreviewed' else 'changed'
        monkeypatch.setattr(launch.p, 'closure_record', lambda *a: ([record], 'digest'))
    with pytest.raises(ValueError, match='closure'):
        launch.build_fields(launch.command('full', 'attempt'), '1', 'commit', 'full')


def log_lines(expected):
    return ['XRIR_RUNTIME_ARGS ' + json.dumps(expected),
            'yaw_aug ENABLED W=512 seed=0 counter=(epoch-1)*74084+batch_idx'] + [
        'Train Epoch: 1 [{}/3]  loss 1.25 (stft 0.5, decay 0.75)'.format(i) for i in range(3)] + [
        'Test set (epoch 1): Average loss: 0.25 over 2 batches']


@pytest.mark.parametrize('failure', [None, 'banner', 'late', 'args', 'nan', 'steps'])
def test_log_guard(failure):
    expected = launch.effective_args(launch.command('smoke', 'attempt'), '1', 74084)
    lines = log_lines(expected)
    if failure == 'banner':
        lines.pop(1)
    elif failure == 'late':
        lines[1], lines[2] = lines[2], lines[1]
    elif failure == 'args':
        lines[0] = lines[0].replace('74084', '1')
    elif failure == 'nan':
        lines[2] = lines[2].replace('loss 1.25', 'loss nan')
    elif failure == 'steps':
        lines.pop(4)
    guard = launch.LogGuard(expected, 'smoke')
    def consume():
        for line in lines:
            guard.feed(line)
        return guard.finish()
    if failure:
        with pytest.raises(ValueError):
            consume()
    else:
        assert consume()['train_losses'] == [1.25, 1.25, 1.25]
        assert guard.runtime == expected


def test_teed_child_closed_log_and_early_abort(tmp_path):
    import sys
    expected = launch.effective_args(launch.command('smoke', 'attempt'), '1', 74084)
    log = tmp_path / 'train.log'
    lines = log_lines(expected)
    code = 'print(' + repr('\n'.join(lines)) + ', flush=True)'
    guard = launch.LogGuard(expected, 'smoke')
    assert launch.run_child([sys.executable, '-c', code], log, '1', guard) == 0
    assert log.read_text().splitlines() == lines
    assert guard.finish()['test_loss'] == 0.25
    code = "import time; print('Train Epoch: 1 [0/3] loss 1.0', flush=True); time.sleep(30)"
    with pytest.raises(ValueError, match='banner|runtime'):
        launch.run_child([sys.executable, '-c', code], tmp_path / 'bad.log', '1',
                         launch.LogGuard(expected, 'smoke'))


@pytest.mark.parametrize('failure', [None, 'child', 'source', 'manifest', 'effective', 'unexpected'])
def test_attempt_finalization_after_closed_log(tmp_path, monkeypatch, failure):
    attempt, log = tmp_path / 'attempt_t', tmp_path / 'logs/train.log'
    argv = launch.command('smoke', str(attempt))
    expected = launch.effective_args(argv, '1', 74084)
    source = tmp_path / 'source.py'
    source.write_text('original')
    fields = dict(repo=str(tmp_path), source_closures={}, train_data_identity=launch.p._inventory([], tmp_path),
        effective_args=expected, command=argv, mutable_inputs={
        'source': {'path': str(source), 'sha256': launch.p.sha256_file(source)}})
    monkeypatch.setattr(launch, 'resource_gate', lambda *a: {})
    def runner(command, path, gpu, guard, deadline):
        assert {f.name for f in attempt.iterdir()} == {'effective_args.json', 'train_manifest.json', 'train_inventory.json'}
        saved = json.loads((attempt / 'train_manifest.json').read_text())
        assert 'inventory' not in saved['train_data_identity']
        assert saved['train_data_identity']['inventory_file']['sha256'] == launch.p.sha256_file(attempt / 'train_inventory.json')
        path.write_text('\n'.join(log_lines(expected)) + '\n')
        guard.poll(path)
        if failure == 'source':
            source.write_text('changed')
        elif failure in ('manifest', 'effective'):
            (attempt / ('train_manifest.json' if failure == 'manifest' else 'effective_args.json')).write_text('{}')
        elif failure == 'unexpected':
            (attempt / 'accidental.pth').write_text('x')
        return 1 if failure == 'child' else 0
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            launch.execute_attempt(attempt, 'smoke', '1', log, lambda: fields, runner=runner)
        aborted, = tmp_path.glob('attempt_t_ABORTED_*')
        assert not (aborted / 'completion.json').exists()
        assert json.loads((tmp_path / 'cumulative_hours.json').read_text())['total_hours'] >= 0
    else:
        result = launch.execute_attempt(attempt, 'smoke', '1', log, lambda: fields, runner=runner)
        assert {f.name for f in attempt.iterdir()} == {'effective_args.json', 'train_manifest.json', 'train_inventory.json', 'completion.json'}
        assert result['log']['sha256'] == launch.p.sha256_file(log)
        assert result['train_manifest_sha256'] == launch.p.sha256_file(attempt / 'train_manifest.json')
        assert result['metrics']['train_losses'] == [1.25] * 3


@pytest.mark.parametrize('failure', [None, 'missing', 'epoch_one', 'nonfinite'])
def test_full_output_acceptance(tmp_path, failure):
    expected = launch.effective_args(launch.command('full', str(tmp_path)), '1', 9261)
    (tmp_path / 'args.json').write_text(json.dumps(expected))
    history = [dict(epoch=i, train_loss=1.0, test_loss=0.5, epoch_minutes=120) for i in range(1, 13)]
    if failure == 'epoch_one':
        history[0]['epoch_minutes'] = 2.432 * 60
    if failure == 'nonfinite':
        history[1]['train_loss'] = float('nan')
    (tmp_path / 'history.jsonl').write_text('\n'.join(map(json.dumps, history)))
    for name in ('best.pth', 'last.pth', 'note.txt'):
        (tmp_path / name).write_bytes(b'extra')
    for i in range(1, 13):
        if failure != 'missing' or i != 12:
            (tmp_path / ('epoch_%03d.pth' % i)).write_bytes(b'checkpoint')
    if failure:
        with pytest.raises(ValueError):
            launch.validate_outputs(tmp_path, 'full', expected)
    else:
        outputs = launch.validate_outputs(tmp_path, 'full', expected)
        assert len(outputs) == 17 and outputs['epoch_012.pth'] == launch.p.sha256_file(tmp_path / 'epoch_012.pth')
        assert outputs['best.pth'] == outputs['last.pth'] == outputs['note.txt']


def test_probe_log_result_required_and_bound_to_yaw():
    expected = launch.effective_args(launch.command('full', 'attempt'), '1', 9261)
    guard = launch.LogGuard(expected, 'probe')
    for line in [log_lines(expected)[0], log_lines(expected)[1].replace('74084', '9261'),
                 'Train Epoch: 1 [0/60] loss 1.0']:
        guard.feed(line)
    with pytest.raises(ValueError, match='probe'):
        guard.finish()
    result = dict(yaw_aug=1, warmup_micro_batches=10, timed_micro_batches=50,
                  mean_iteration_seconds=1.1, peak_allocated_bytes=100, peak_reserved_bytes=200)
    guard.feed('EXP04_PROBE_RESULT ' + json.dumps(result))
    assert guard.finish()['probe'] == result
    result['yaw_aug'] = 0
    guard.feed('EXP04_PROBE_RESULT ' + json.dumps(result))
    with pytest.raises(ValueError, match='probe'):
        guard.finish()


def test_refuse_mode_never_spawns_trainer(monkeypatch):
    monkeypatch.setattr(launch.subprocess, 'Popen', lambda *a, **k: pytest.fail('spawned trainer'))
    result = launch.refusal_self_test()
    assert len(result['refusals']) >= 6 and result['passed']


@pytest.mark.parametrize('mode', ['smoke', 'probe'])
def test_cli_modes_preserve_probe_cotenant_evidence(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(launch, 'ROOT', tmp_path)
    monkeypatch.setattr(launch, 'gpu_snapshot', lambda gpu: {'compute_apps': '123, foreign.py, 11000',
                                                          'utilization_gpu': 90})
    monkeypatch.setattr(launch.subprocess, 'check_output', lambda *a, **k: 'reviewed\n')
    calls = []
    monkeypatch.setattr(launch, 'build_fields', lambda *a: {'command': [], 'allow_dirty': a[-1]})
    def execute(attempt, mode, gpu, log, factory, **kwargs):
        assert factory()['allow_dirty'] is True
        calls.append((attempt, mode, gpu, log, kwargs))
        return {'metrics': {'probe': {'mean_iteration_seconds': 1.0, 'peak_allocated_bytes': 123}},
                'resource_before': {'compute_apps': 'arm cotenant'}}
    monkeypatch.setattr(launch, 'execute_attempt', execute)
    result = launch.main([mode, '--gpu', '1', '--reviewed-commit', 'HEAD', '--timestamp', 'test',
                          '--log-dir', str(tmp_path / 'logs'), '--allow-cotenant', '--allow-dirty'])
    assert len(calls) == (1 if mode == 'smoke' else 2)
    assert all(row[2] == '1' and row[4]['allow_cotenant'] for row in calls)
    if mode == 'smoke':
        assert calls[0][0].name == '_smoke_test'
    else:
        assert result['PROBE_NOT_CLEAN'] is True and result['overhead_ratio'] == 1
        assert result['before']['compute_apps'] and result['after']['utilization_gpu'] == 90
        assert result['arms_before'] == [{'compute_apps': 'arm cotenant'}] * 2
        assert [row[0].name for row in calls] == ['_probe_test_off', '_probe_test_on']
        assert json.loads((tmp_path / '_probe_test.json').read_text()) == result


def test_cli_full_forbids_cotenant():
    with pytest.raises(SystemExit):
        launch.main(['full', '--allow-cotenant'])


@pytest.fixture
def control_parity_inputs():
    expected = launch.effective_args(launch.command('full', 'attempt'), '1', 9261)
    runtime = {key: value for key, value in expected.items() if key not in launch.ENV_KEYS}
    runtime['env'] = {key: expected[key] for key in launch.ENV_KEYS}
    control = {key: value for key, value in runtime.items() if key not in {
        'env', 'train_batches_per_epoch', 'yaw_aug', 'yaw_aug_seed', 'yaw_aug_width', 'no_save'}}
    control.update(save_dir='control', save_every=500, epoch_ckpt_every=5)
    control_env = {key: value for key, value in runtime['env'].items() if key != 'PYTHONHASHSEED'}
    return runtime, control, control_env


def test_control_parity_reconstructs_bpe_and_reports_exact_exclusions(control_parity_inputs):
    runtime, control, control_env = control_parity_inputs
    runtime['env']['CUDA_VISIBLE_DEVICES'] = '0'
    assert runtime['train_batches_per_epoch'] == (296334 + control['batch_size'] - 1) // control['batch_size']
    differences = launch.compare_control(runtime, control, control_env)
    assert set(differences) == {'save_dir', 'yaw_aug', 'yaw_aug_seed', 'yaw_aug_width',
                                'no_save', 'save_every', 'epoch_ckpt_every', 'PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES'}
    assert 'env' not in control and 'train_batches_per_epoch' not in control


@pytest.mark.parametrize('key,value', [('lr', 0.002), ('num_workers', '12'), ('tf32', 1),
    ('train_batches_per_epoch', 9260), ('unexpected', 1)])
def test_control_parity_refuses_nonexcluded_type_or_value_change(control_parity_inputs, key, value):
    runtime, control, control_env = control_parity_inputs
    (runtime['env'] if key in launch.ENV_KEYS else runtime)[key] = value
    with pytest.raises(ValueError, match=key):
        launch.compare_control(runtime, control, control_env)


def test_full_promotion_failure_accounts_elapsed_hours_once(tmp_path, monkeypatch):
    attempt, log = tmp_path / 'attempt_t', tmp_path / 'logs/train.log'
    argv = launch.command('full', str(attempt))
    fields = dict(repo=str(tmp_path), source_closures={}, train_data_identity=launch.p._inventory([], tmp_path),
                  effective_args=launch.effective_args(argv, '1', 9261), command=argv)
    clock = [100.0]
    monkeypatch.setattr(launch.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(launch, 'resource_gate', lambda *a: {})
    monkeypatch.setattr(launch.LogGuard, 'finish', lambda self: {})
    monkeypatch.setattr(launch, 'validate_outputs', lambda *a: {})
    def runner(command, path, gpu, guard, deadline):
        path.write_text('closed training log\n')
        clock[0] += 3600
        return 0
    def fail_promotion(path):
        assert (path / 'completion.json').is_file()
        raise OSError('promotion failed')
    monkeypatch.setattr(launch, 'promote', fail_promotion)
    with pytest.raises(OSError, match='promotion failed'):
        launch.execute_attempt(attempt, 'full', '1', log, lambda: fields, runner=runner)
    aborted, = tmp_path.glob('attempt_t_ABORTED_*')
    assert not attempt.exists() and not (aborted / 'completion.json').exists()
    record = launch.hours_record(tmp_path)
    assert record['total_hours'] == 1.0
    assert record['attempts'] == [{'attempt': aborted.name, 'hours': 1.0, 'mode': 'full'}]


def test_tee_preserves_redirected_launcher_output(tmp_path):
    import sys
    outer, inner = tmp_path / 'outer.log', tmp_path / 'inner.log'
    code = ('import sys; from pathlib import Path; from types import SimpleNamespace; '
            'from tools.exp04_launcher import run_child; '
            "print('launcher start', flush=True); "
            'run_child([sys.executable, "-c", "print(123, flush=True)"], Path(%r), "1", '
            'SimpleNamespace(poll=lambda path: None)); ' % str(inner) +
            "print('launcher end', flush=True)")
    with outer.open('w') as stream:
        launch.subprocess.run([sys.executable, '-c', code], cwd=launch.REPO, check=True,
                              stdout=stream, stderr=launch.subprocess.STDOUT)
    assert outer.read_text().splitlines() == ['launcher start', '123', 'launcher end']
    assert inner.read_text() == '123\n'


def test_epoch_one_acceptance_is_checked_immediately(tmp_path):
    (tmp_path / 'history.jsonl').write_text(json.dumps({'epoch': 1, 'epoch_minutes': 2.432 * 60}) + '\n')
    expected = launch.effective_args(launch.command('full', str(tmp_path)), '1', 9261)
    guard = launch.LogGuard(expected, 'full')
    with pytest.raises(ValueError, match='2.431'):
        guard.feed('epoch 1 done in 145.9 min; best test loss 0.1')


@pytest.mark.parametrize('mode', ['smoke', 'probe'])
@pytest.mark.parametrize('signum', [launch.signal.SIGTERM, launch.signal.SIGHUP])
def test_signals_abort_and_reap_sleeping_child(tmp_path, mode, signum):
    import sys
    import time
    attempt, pidfile = tmp_path / 'attempt_t', tmp_path / 'child.pid'
    expected = dict(yaw_aug=0, train_batches_per_epoch=1)
    child = ('import os, time; from pathlib import Path; '
             'print(%r, flush=True); print("yaw_aug DISABLED", flush=True); '
             'print("Train Epoch: 1 [0/1] loss 1", flush=True); '
             'Path(%r).write_text(str(os.getpid())); time.sleep(60)' %
             ('XRIR_RUNTIME_ARGS ' + json.dumps(expected), str(pidfile)))
    fields = dict(effective_args=expected, command=[sys.executable, '-c', child])
    ready = tmp_path / 'guard.ready'
    code = ('from tools import exp04_launcher as l; from pathlib import Path; '
            'l.resource_gate=lambda *a: {}; '
            'poll=l.LogGuard.poll; l.LogGuard.poll=lambda self, path: '
            '(poll(self, path), Path(%r).touch() if self.steps else None); ' % str(ready) +
            'l.execute_attempt(%r, %r, "1", %r, lambda: %r)' %
            (str(attempt), mode, str(tmp_path / 'train.log'), fields))
    process = launch.subprocess.Popen([sys.executable, '-c', code], cwd=launch.REPO)
    child_pid = None
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pidfile.exists() and ready.exists()
        child_pid = int(pidfile.read_text())
        process.send_signal(signum)
        process.wait(timeout=10)
        aborted, = tmp_path.glob('attempt_t_ABORTED_*')
        assert not attempt.exists() and not (aborted / 'completion.json').exists()
        assert launch.hours_record(tmp_path)['total_hours'] > 0
        record = json.loads((aborted / 'abort.json').read_text())
        assert record['reason'] == 'terminated_' + launch.signal.Signals(signum).name
        assert record['exception_type'] == 'LauncherTerminated'
        assert record['wall_hours'] > 0 and record['started_at'] <= record['aborted_at']
        assert record['last_guard_state']['banner'] is True
        assert not (tmp_path / 'train.log').exists()
        assert Path(record['log']['aborted']).is_file()
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        if child_pid is not None:
            try:
                os.kill(child_pid, launch.signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_termination_handlers_restore_previous_dispositions():
    signals = (launch.signal.SIGTERM, launch.signal.SIGHUP)
    before = [launch.signal.getsignal(s) for s in signals]
    with pytest.raises(launch.LauncherTerminated):
        with launch.termination_handlers():
            os.kill(os.getpid(), launch.signal.SIGHUP)
    assert [launch.signal.getsignal(s) for s in signals] == before


def test_budget_counts_only_full_including_legacy_rows(tmp_path):
    rows = [{'attempt': '_smoke_old', 'hours': 20}, {'attempt': 'attempt_t_probe_on', 'hours': 20},
            {'attempt': 'attempt_old', 'hours': 10}]
    launch.p.write_completion(tmp_path / 'cumulative_hours.json', dict(total_hours=50, attempts=rows))
    for mode in ('probe', 'smoke', 'full'):
        launch.account_hours(tmp_path / mode, 1, mode=mode)
    assert all('mode' in row for row in launch.hours_record(tmp_path)['attempts'])
    launch.check_budget(tmp_path, 32)
    with pytest.raises(ValueError, match='43'):
        launch.check_budget(tmp_path, 32.01)


@pytest.mark.parametrize('mutation', [None, 'commit', 'before', 'after', 'ratio', 'slow', 'dirty', 'passed'])
def test_probe_receipt_admission_and_binding(tmp_path, monkeypatch, mutation):
    receipt = dict(reviewed_commit='a' * 40, before={'gpu': 'GPU-free'}, after={'gpu': 'GPU-free'},
        yaw_off={'mean_iteration_seconds': 1}, yaw_on={'mean_iteration_seconds': 1.05},
        overhead_ratio=1.05, passed=True, PROBE_NOT_CLEAN=False)
    if mutation == 'commit':
        receipt['reviewed_commit'] = 'b' * 40
    elif mutation in ('before', 'after'):
        receipt[mutation]['gpu'] = '1'
    elif mutation == 'ratio':
        receipt['overhead_ratio'] = 1
    elif mutation == 'slow':
        receipt['yaw_on']['mean_iteration_seconds'] = 1.1
        receipt['overhead_ratio'] = 1.1
    elif mutation in ('dirty', 'passed'):
        receipt['PROBE_NOT_CLEAN' if mutation == 'dirty' else 'passed'] = mutation == 'dirty'
    path = tmp_path / 'probe.json'
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(launch, 'ROOT', tmp_path)
    monkeypatch.setattr(launch.subprocess, 'check_output', lambda *a, **k: 'a' * 40 + '\n')
    monkeypatch.setattr(launch, 'build_fields', lambda *a: {'mutable_inputs': {}})
    def execute(attempt, mode, gpu, log, factory, **kwargs):
        assert gpu == 'GPU-free'
        assert factory()['mutable_inputs']['probe_receipt'] == {'path': str(path), 'sha256': launch.p.sha256_file(path)}
        return {}
    monkeypatch.setattr(launch, 'execute_attempt', execute)
    argv = ['full', '--gpu', 'GPU-free', '--reviewed-commit', 'HEAD', '--probe-json', str(path), '--log-dir', str(tmp_path)]
    if mutation:
        with pytest.raises(ValueError, match='probe'):
            launch.main(argv)
    else:
        launch.main(argv)


def test_guard_uses_repo_paths_and_expected_banner(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    attempt = tmp_path / 'attempt'
    attempt.mkdir()
    expected = launch.effective_args(launch.command('full', 'attempt'), '1', 9261)
    expected.update(yaw_aug_seed=7, yaw_aug_width=256)
    (attempt / 'args.json').write_text(json.dumps(expected))
    (attempt / 'history.jsonl').write_text(json.dumps({'epoch_minutes': 120}))
    guard = launch.LogGuard(expected, 'full')
    guard.feed('XRIR_RUNTIME_ARGS ' + json.dumps(expected))
    guard.feed('yaw_aug ENABLED W=512 seed=0 counter=(epoch-1)*9261+batch_idx')
    assert not guard.banner
    guard.feed('yaw_aug ENABLED W=256 seed=7 counter=(epoch-1)*9261+batch_idx')
    guard.feed('Train Epoch: 1 [0/9261] loss 1')
    guard.feed('epoch 1 done')
    assert guard.epoch_one_done


@pytest.mark.parametrize('failure,reason', [('args', 'guard_runtime_args'), ('banner', 'guard_banner'),
    ('epoch', 'guard_epoch_one'), ('deadline', 'deadline_43h'), ('exit', 'child_exit_3'),
    ('nan', 'child_failed'), ('collision', 'child_failed')])
def test_abort_diagnostics_and_owned_log(tmp_path, monkeypatch, failure, reason):
    attempt, log = tmp_path / 'attempt', tmp_path / 'train.log'
    expected = launch.effective_args(launch.command('full', str(attempt)), '1', 9261)
    monkeypatch.setattr(launch, 'resource_gate', lambda *a: {})
    if failure == 'collision':
        log.write_text('foreign')
    def runner(argv, path, gpu, guard, deadline):
        if failure == 'collision':
            return launch.run_child([], path, gpu, guard)
        with path.open('x'):
            guard.log_created = True
        if failure == 'args':
            guard.feed('XRIR_RUNTIME_ARGS {}')
        elif failure == 'banner':
            guard.feed('Train Epoch: 1 [0/1] loss 1')
        elif failure == 'epoch':
            (attempt / 'history.jsonl').write_text('{"epoch_minutes": 146}')
            guard.feed('epoch 1 done')
        elif failure == 'deadline':
            path.unlink()
            launch.run_child([launch.PYTHON, '-c', 'import time; time.sleep(60)'], path, gpu, guard, 0)
        elif failure == 'nan':
            guard.feed('Test set (epoch 1): Average loss: nan over 2 batches')
        return 3
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        launch.execute_attempt(attempt, 'full', '1', log,
            lambda: dict(command=[], effective_args=expected), runner=runner)
    aborted, = tmp_path.glob('attempt_ABORTED_*')
    record = json.loads((aborted / 'abort.json').read_text())
    assert record['reason'] == reason and aborted.name.endswith(reason)
    assert record['log']['original'] == str(log)
    if failure == 'collision':
        assert log.read_text() == 'foreign' and record['log']['aborted'] is None
    else:
        assert not log.exists() and Path(record['log']['aborted']).is_file()


def test_child_records_group_and_closed_log_for_recovery(tmp_path):
    expected = launch.effective_args(launch.command('smoke', 'unused'), '1', 74084)
    guard = launch.LogGuard(expected, 'smoke')
    guard.execution_path = tmp_path / 'execution.json'
    guard.execution = {'train_manifest_sha256': 'spawn-digest'}
    log = tmp_path / 'train.log'
    code = 'print(%r, flush=True)' % '\n'.join(log_lines(expected))
    assert launch.run_child([launch.PYTHON, '-c', code], log, '1', guard) == 0
    record = json.loads(guard.execution_path.read_text())
    assert record['train_manifest_sha256'] == 'spawn-digest'
    assert record['child_exit_status'] == 0 and record['ended_at']
    assert record['log_sha256'] == launch.p.sha256_file(log)
    with pytest.raises(ProcessLookupError):
        os.killpg(record['child_pgid'], 0)


@pytest.mark.parametrize('key', launch.REQUIRED_INPUTS)
def test_training_required_inputs(key):
    assert 'missing.' + key in launch.p.revalidate({}, required=launch.REQUIRED_INPUTS)
