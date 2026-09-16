"""exp_07 seen launcher: pre-spawn fields, the log guard, recovery and the run modes.

CPU only: the trainer is never spawned, the GPU is stubbed and every directory is
created under ``tmp_path``.
"""
import json
import subprocess
from pathlib import Path

import pytest

from tools import exp04_launcher as base
from tools import exp07_launcher as launch
from tools import exp07_probe as probe
from tools import exp07_provenance as e7p
from tools import provenance as p
from test_exp07_launcher import ARMS, SEEN_BATCHES, argv_of, attempt_of


@pytest.fixture
def stub_inputs(monkeypatch):
    """Stub the expensive parts of build_fields; the seen bindings stay real."""
    files = sorted(launch.train_minimum())
    monkeypatch.setattr(launch.p, 'source_closure', lambda module, repo: files)
    monkeypatch.setattr(launch.p, 'closure_record', lambda paths, commit, repo: ([
        dict(path=name, reviewed_blob_sha256='same', working_tree_sha256='same',
             commits_after_reviewed=[]) for name in paths], 'closure'))
    monkeypatch.setattr(launch.p, 'environment', lambda: {'executable': launch.PYTHON})
    monkeypatch.setattr(launch.p, 'git_state', lambda repo: {'dirty_outside_worklog': False})
    calls = []

    def inventory(root, protocol='seen', cache_path=None):
        calls.append((root, protocol, str(cache_path)))
        return {'inventory_files': launch.TRAIN_FILES[protocol]}

    monkeypatch.setattr(launch.e7p, 'train_data_identity', inventory)
    return calls


@pytest.mark.parametrize('name', sorted(ARMS))
def test_seen_fields_bind_split_inventory_and_protocol(stub_inputs, name):
    fields = launch.build_fields(argv_of(name), '1', 'commit', 'full')
    assert stub_inputs == [(launch.DATA_ROOT, 'seen', str(launch.REPO / e7p.SEEN_CACHE))]
    assert fields['protocol'] == 'seen' and fields['effective_args']['protocol'] == 'seen'
    assert fields['effective_args']['train_batches_per_epoch'] == SEEN_BATCHES
    assert fields['mutable_inputs']['seen_split'] == e7p.seen_split_identity(launch.REPO)
    assert fields['mutable_inputs']['control_args']['path'] == str(launch.REPO / launch.CONTROLS[name])
    assert fields['control_excluded_differences']['protocol']['control'] == 'unseen'
    training = {r['path'] for r in fields['source_closures']['training']['files']}
    assert {launch.SEEN_MODULE, launch.ENTRY} <= training


def test_a_smoke_run_binds_no_control(stub_inputs):
    fields = launch.build_fields(argv_of('seen_simple', mode='smoke'), '1', 'commit', 'smoke')
    assert set(fields['mutable_inputs']) == {'seen_split'}
    # A smoke run is bounded to three batches of 4: its loader length is the seen count / 4.
    assert fields['effective_args']['train_batches_per_epoch'] == -(-launch.TRAIN_FILES['seen'] // 4)


@pytest.mark.parametrize('mutated', [False, True])
def test_seen_split_is_captured_before_the_inventory(stub_inputs, monkeypatch, mutated):
    """The bound split is the one the inventory was selected under; a change refuses."""
    order, real = [], e7p.seen_split_identity(launch.REPO)

    def identity(repo):
        order.append('split')
        return dict(real, sha256='b' * 64) if mutated and len(order) > 1 else dict(real)

    def inventory(root, protocol='seen', cache_path=None):
        order.append('inventory')
        return {'inventory_files': launch.TRAIN_FILES[protocol]}

    monkeypatch.setattr(launch.e7p, 'seen_split_identity', identity)
    monkeypatch.setattr(launch.e7p, 'train_data_identity', inventory)
    if mutated:
        with pytest.raises(ValueError, match='seen split changed'):
            launch.build_fields(argv_of('seen_simple'), '1', 'commit', 'full')
    else:
        assert launch.build_fields(argv_of('seen_simple'), '1', 'commit',
                                   'full')['mutable_inputs']['seen_split'] == real
    assert order == ['split', 'inventory', 'split']


def test_a_closure_without_the_entry_point_is_refused(stub_inputs, monkeypatch):
    partial = sorted(launch.train_minimum() - {launch.ENTRY})
    monkeypatch.setattr(launch.p, 'source_closure', lambda module, repo: partial)
    with pytest.raises(ValueError, match='training closure missing'):
        launch.build_fields(argv_of('seen_simple'), '1', 'commit', 'full')


def test_bound_seen_split_is_revalidated():
    record = e7p.seen_split_identity(launch.REPO)
    fields = {'repo': str(launch.REPO), 'mutable_inputs': {'seen_split': record}}
    assert p.revalidate(fields) == [] and 'seen_split' in launch.SEEN_INPUTS
    fields['mutable_inputs']['seen_split'] = dict(record, sha256='0' * 64)
    assert p.revalidate(fields) == ['seen_split']


@pytest.mark.parametrize('name', sorted(ARMS))
def test_banner_rule_per_arm(name):
    expected = launch.effective_args(argv_of(name), '1', SEEN_BATCHES)
    guard = launch.LogGuard(expected, 'full')
    enabled = 'yaw_aug ENABLED W=512 seed=0 counter=(epoch-1)*{}+batch_idx'
    wanted = enabled.format(SEEN_BATCHES) if name == 'seen_aug' else 'yaw_aug DISABLED'
    for other in (enabled.format(9261),
                  'yaw_aug DISABLED' if name == 'seen_aug' else enabled.format(SEEN_BATCHES)):
        guard.feed(other)
        assert not guard.banner
    guard.feed(wanted)
    assert guard.banner


def seen_measurement(backbone='simple', yaw=1):
    values = [1.0] * 50
    measured = dict(tier='M', backbone=backbone, protocol='seen', yaw_aug=yaw, batch_size=32,
                    accum_steps=2, warmup_micro_batches=10, timed_micro_batches=50,
                    train_batches_per_epoch=SEEN_BATCHES, t_test=3.0, t_save=2.0,
                    t_micro=dict(mean=1.0, median=1.0, min=1.0, values=values),
                    iteration_seconds=values, mean_iteration_seconds=1.0,
                    median_iteration_seconds=1.0, min_iteration_seconds=1.0,
                    test_timing_protocol='full_test_loader', test_batches_timed=195,
                    test_batches_total=195, peak_allocated_bytes=123, peak_reserved_bytes=456,
                    train_loss=1.0, schema_version=1)
    measured.update(probe.projection(measured))
    return measured


def test_log_guard_requires_the_seen_probe_line():
    argv = argv_of('seen_aug')
    expected = launch.effective_args(argv, '1', SEEN_BATCHES)
    guard = launch.LogGuard(expected, 'probe')
    for prefix in ('EXP04_PROBE_RESULT ', 'EXP05_PROBE_RESULT '):
        with pytest.raises(ValueError, match='prefix'):
            guard.feed(prefix + '{}')
    guard.feed('XRIR_RUNTIME_ARGS ' + json.dumps(expected))
    guard.feed('yaw_aug ENABLED W=512 seed=0 counter=(epoch-1)*9265+batch_idx')
    guard.feed('Train Epoch: 1 [0/9265] loss 1.0')
    measured = seen_measurement()
    guard.feed(probe.PREFIX + json.dumps(measured))
    assert guard.finish()['probe'] == measured
    guard.feed(probe.PREFIX + json.dumps(dict(measured, protocol='unseen')))
    with pytest.raises(ValueError, match='probe'):
        guard.finish()


def recovery_fields(name='seen_simple'):
    attempt = attempt_of(name)
    argv = argv_of(name, attempt)
    closure = [dict(path=part) for part in sorted(launch.train_minimum())]
    return dict(repo=str(launch.REPO), mode='full', reviewed_commit='a' * 40, command=argv,
                effective_args=launch.effective_args(argv, '1', SEEN_BATCHES),
                source_closures=dict(training=dict(files=closure), launcher=dict(files=closure)),
                resource_before={'gpu': '1'}, env={key: '' for key in launch.ENV_KEYS},
                train_data_identity=dict(inventory_files=launch.TRAIN_FILES['seen']),
                attempt_path=str((launch.REPO / attempt).resolve()),
                mutable_inputs=dict(seen_split=e7p.seen_split_identity(launch.REPO),
                                    **{part: {} for part in ('effective_args', 'train_inventory',
                                                             'control_args', 'probe_receipt')}))


@pytest.mark.parametrize('fault,message', [
    (None, 'launcher-log'), ('binding', 'required bindings'), ('digest', 'seen split'),
    ('count', 'training inventory count'), ('closure', 'closure minimum'),
    ('entry', 'closure minimum'), ('argv', 'golden')])
def test_recovery_validates_the_seen_inputs(fault, message):
    fields = recovery_fields()
    if fault == 'binding':
        del fields['mutable_inputs']['seen_split']
    elif fault == 'digest':
        fields['mutable_inputs']['seen_split']['sha256'] = '0' * 64
    elif fault == 'count':
        fields['train_data_identity']['inventory_files'] = launch.TRAIN_FILES['unseen']
    elif fault in ('closure', 'entry'):
        dropped = launch.SEEN_MODULE if fault == 'closure' else launch.ENTRY
        files = fields['source_closures']['training']['files']
        fields['source_closures']['training']['files'] = [r for r in files if r['path'] != dropped]
    elif fault == 'argv':
        fields['command'] = list(fields['command'])
        fields['command'][fields['command'].index('--lr') + 1] = '0.5'
    # A clean seen manifest reaches the launcher-log check that follows these gates.
    with pytest.raises(ValueError, match=message):
        launch.recovery_evidence(fields, {}, launch.REPO / attempt_of('seen_simple'), None)


def test_refuse_test_covers_the_seen_refusals(monkeypatch):
    monkeypatch.setattr(base.subprocess, 'Popen', lambda *a, **k: pytest.fail('spawned trainer'))
    refusals = launch.refusal_self_test()['refusals']
    assert {'seen_other_arm', 'seen_unseen_argv', 'seen_arm', 'seen_bpe', 'seen_control',
            'seen_protocol', 'probe_prefix', 'one_retry'} <= set(refusals)


def test_launch_script_passes_every_argument_through():
    script = launch.REPO / 'tools/exp07_launch.sh'
    assert subprocess.run(['bash', '-n', str(script)]).returncode == 0
    text = script.read_text()
    assert text.strip().endswith('-m tools.exp07_launcher "$@"')
    assert '--backbone <b>' in text and '--yaw-aug 1' in text


def test_seen_overrides_restore_every_shared_binding():
    before = {name: getattr(base, name) for name in
              ('REPO', 'TRAIN_MINIMUM', 'tier_gates', 'child_environment', 'check_golden',
               'check_runtime', 'compare_control', 'complete_attempt', 'recovery_evidence',
               'LogGuard')}
    with launch.seen_overrides():
        assert base.check_golden is launch.check_golden and base.LogGuard is launch.LogGuard
        assert base.tier_gates is launch.gates and launch.SEEN_MODULE in base.TRAIN_MINIMUM
    assert all(getattr(base, name) is value for name, value in before.items())


@pytest.fixture
def seen_repo(tmp_path, monkeypatch):
    """A throwaway repository with the launcher's inputs; no GPU and no trainer."""
    real = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    for name in sorted(launch.train_minimum() | {e7p.SEEN_SPLIT, 'tools/exp07_launcher.py',
                                                 'tools/exp07_probe.py', 'tools/exp07_launch.sh'}):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text('# fixture ' + name + '\n')
    assets = launch.ASSETS + '/argv_golden_seen_simple_smoke.txt'
    (tmp_path / launch.RECORD / assets).parent.mkdir(parents=True)
    (tmp_path / launch.RECORD / assets).write_text((real / launch.RECORD / assets).read_text())
    for argv in (['init', '-q'], ['add', '-A'], ['-c', 'user.name=T', '-c', 'user.email=t@e',
                                                 'commit', '-qm', 'fixture']):
        subprocess.run(['git'] + argv, cwd=str(tmp_path), check=True)
    monkeypatch.setattr(launch.p, 'source_closure', lambda module, repo:
                        sorted(launch.train_minimum()) if module == launch.ENTRY_MODULE
                        else ['tools/exp07_launcher.py'])
    monkeypatch.setattr(launch.p, 'environment', lambda: {'executable': launch.PYTHON})
    monkeypatch.setattr(launch.e7p, 'train_data_identity', lambda root, protocol='seen', cache_path=None:
                        dict(p._inventory([e7p.SEEN_SPLIT], tmp_path), protocol=protocol,
                             split='train', cache_key=str(cache_path)))
    monkeypatch.setattr(base, 'resource_gate', lambda *a: {'gpu': '1'})
    return tmp_path


def test_seen_smoke_writes_the_pre_spawn_files(seen_repo, monkeypatch):
    attempt = seen_repo / 'ckpt/exp07/seen_simple/_smoke_test'

    def runner(argv, path, gpu, guard, deadline=None):
        assert argv[:2] == [launch.PYTHON, launch.ENTRY]
        assert {f.name for f in attempt.iterdir()} == {'effective_args.json',
                                                       'train_manifest.json', 'train_inventory.json'}
        expected = json.loads((attempt / 'effective_args.json').read_text())
        path.write_text('XRIR_RUNTIME_ARGS ' + json.dumps(expected) + '\nyaw_aug DISABLED\n' +
                        ''.join('Train Epoch: 1 [{}/3] loss 1.25\n'.format(i) for i in range(3)) +
                        'Test set (epoch 1): Average loss: 0.25 over 2 batches\n')
        guard.log_created = True
        guard.poll(path)
        return 0

    execute = base.execute_attempt
    monkeypatch.setattr(base, 'execute_attempt', lambda *a, **k: execute(*a, runner=runner, **k))
    launch.main(['smoke', '--backbone', 'simple', '--reviewed-commit', 'HEAD',
                 '--timestamp', 'test', '--log-dir', str(seen_repo / 'logs')])
    manifest = json.loads((attempt / 'train_manifest.json').read_text())
    assert manifest['protocol'] == 'seen' and manifest['effective_args']['protocol'] == 'seen'
    assert manifest['command'] == launch.command('smoke', 'ckpt/exp07/seen_simple/_smoke_test',
                                                 'simple', 0)
    assert manifest['mutable_inputs']['seen_split'] == e7p.seen_split_identity(seen_repo)
    assert manifest['train_data_identity']['protocol'] == 'seen'
    assert 'inventory' not in manifest['train_data_identity']
    assert (manifest['train_data_identity']['inventory_file']['sha256'] ==
            p.sha256_file(attempt / 'train_inventory.json'))
    training = {r['path'] for r in manifest['source_closures']['training']['files']}
    assert {launch.SEEN_MODULE, launch.ENTRY} <= training
    assert json.loads((attempt / 'effective_args.json').read_text()) == manifest['effective_args']
    assert (attempt / 'completion.json').is_file()
    assert not list(seen_repo.glob('ckpt/exp07/*/*_ABORTED_*'))
    log = seen_repo / 'logs/seen_protocol_test_train_seen_simple_smoke.log'
    assert json.loads((attempt / 'completion.json').read_text())['log']['path'] == str(log)


def test_finalize_runs_the_shared_recovery_under_the_seen_overrides(tmp_path, monkeypatch):
    """The shared finaliser is reused as it is; exp_07 only supplies its collaborators."""
    seen = {}

    def finalize(attempt, launcher_log=None):
        seen.update(attempt=attempt, log=launcher_log, golden=base.check_golden,
                    guard=base.LogGuard, gates=base.tier_gates,
                    recovery=base.recovery_evidence)
        return {'ok': True}

    monkeypatch.setattr(base, 'finalize_attempt', finalize)
    attempt = tmp_path / 'ckpt/exp07/seen_simple/attempt_test'
    result = launch.main(['finalize', str(attempt), '--launcher-log', str(tmp_path / 'l.log')])
    assert result == {'ok': True} and seen['attempt'] == str(attempt)
    assert seen['log'] == str(tmp_path / 'l.log')
    assert seen['golden'] is launch.check_golden and seen['guard'] is launch.LogGuard
    assert seen['gates'] is launch.gates and seen['recovery'] is launch.recovery_evidence
    assert base.check_golden is not launch.check_golden  # restored on the way out


def test_finalize_without_an_attempt_directory_is_refused(tmp_path, capsys):
    with pytest.raises(SystemExit):
        launch.main(['finalize', '--launcher-log', str(tmp_path / 'l.log')])
    assert 'attempt directory' in capsys.readouterr().err


@pytest.mark.parametrize('argv,message', [
    (['full', '--backbone', 'cylindrical'], 'probe-json'),
    (['probe', '--backbone', 'simple', '--renew-ceiling', '2026-09-15T00:00:00-04:00: r'],
     'renew-ceiling'),
    (['probe', '--backbone', 'cylindrical', '--yaw-aug', '1'], 'seen arm'),
    (['smoke', '--backbone', 'simple', '--timestamp', 'bad/stamp'], 'timestamp'),
    (['full', '--backbone', 'simple', '--probe-json', 'r.json', '--allow-cotenant'], 'cotenant')])
def test_cli_refusals_create_nothing(tmp_path, capsys, argv, message):
    with pytest.raises(SystemExit):
        launch.main(argv + ['--reviewed-commit', 'HEAD', '--log-dir', str(tmp_path)])
    assert message in capsys.readouterr().err
    assert not (tmp_path / 'ckpt').exists()
