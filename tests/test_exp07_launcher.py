"""exp_07 seen-protocol launcher guards: arms, goldens, per-arm controls, seen bindings.

Every check is CPU-only: no trainer is spawned and no attempt directory is created
outside ``tmp_path``.  The historical comparators are the REAL ``ckpt/...args.json``
files of exp_01 and exp_04 (read-only), so a recipe drift would be caught here.
"""
import json
import subprocess
from pathlib import Path

import pytest

from tools import exp04_launcher as launch
from tools import provenance as p

RECIPE = ('--num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 --decay-epochs 3 '
          '--lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2 --num-workers 12 '
          '--seed 0 --tf32 --log-interval 50 --save-every 0 --epoch-ckpt-every 1').split()
YAW = ['--yaw-aug', '1', '--yaw-aug-seed', '0', '--yaw-aug-width', '512']
ARMS = {'seen_simple': ('simple', 0), 'seen_cyl': ('cylindrical', 0), 'seen_aug': ('simple', 1)}
CONTROLS = {'seen_simple': 'ckpt/xRIR_simple_8_shot/args.json',
            'seen_cyl': 'ckpt/xRIR_cyl_8_shot/args.json',
            'seen_aug': 'ckpt/xRIR_simple_yawaug_8_shot/final/args.json'}
SEEN_BATCHES = 9265
SMOKE = launch.command('smoke', '<x>')[6:-6]  # the exp_04 smoke recipe, arm/protocol flags aside


def attempt_of(arm, mode='full'):
    return 'ckpt/exp07/{}/{}test'.format(arm, 'attempt_' if mode == 'full' else '_smoke_')


def argv_of(arm, attempt=None, mode='full'):
    backbone, yaw = ARMS[arm]
    return launch.command(mode, attempt or attempt_of(arm, mode), 'M', backbone, 'seen', yaw)


@pytest.mark.parametrize('mode', ['full', 'smoke'])
@pytest.mark.parametrize('arm', sorted(ARMS))
def test_golden_argv_equals_the_plan_recipe(arm, mode):
    backbone, yaw = ARMS[arm]
    attempt = attempt_of(arm, mode)
    recipe = RECIPE if mode == 'full' else SMOKE
    expected = ([launch.PYTHON, 'train_xRIR_backbone.py', '--backbone', backbone, '--save-dir',
                 attempt + ('' if mode == 'full' else '/smoke')] + recipe +
                ['--protocol', 'seen'] + (YAW if yaw else []))
    argv = argv_of(arm, attempt, mode)
    assert argv == expected
    launch.check_golden(argv, mode, attempt)
    golden = (launch.REPO / launch.EXP07_RECORD / 'seen_protocol_results_assets' /
              ('argv_golden_{}{}.txt'.format(arm, '' if mode == 'full' else '_smoke')))
    placeholder = attempt.replace('test', '<ts>')
    assert golden.read_text().split() == [part.replace(attempt, placeholder) for part in expected]


@pytest.mark.parametrize('flag,value', [('--lr', '0.002'), ('--epochs', '11'),
                                        ('--protocol', 'unseen'), ('--yaw-aug-width', '256')])
def test_one_flag_deviation_is_refused(flag, value):
    attempt = attempt_of('seen_aug')
    argv = argv_of('seen_aug', attempt)
    launch.check_golden(argv, 'full', attempt)
    argv[argv.index(flag) + 1] = value
    with pytest.raises(ValueError, match='golden'):
        launch.check_golden(argv, 'full', attempt)


@pytest.mark.parametrize('arm,attempt', [
    ('seen_simple', 'ckpt/xRIR_simple_yawaug_8_shot/attempt_test'),
    ('seen_aug', 'ckpt/xRIR_simple_yawaug_8_shot/attempt_test'),
    ('seen_simple', 'ckpt/exp05/S_simple/attempt_test'),
    ('seen_simple', 'ckpt/exp07/seen_cyl/attempt_test'),
    ('seen_aug', 'ckpt/exp07/seen_simple/attempt_test'),
    ('seen_simple', 'ckpt/exp07/seen_simple/_smoke_test'),
    ('seen_simple', 'ckpt/exp07/seen_simple/attempt_../other')])
def test_seen_argv_refused_outside_its_own_arm(arm, attempt):
    with pytest.raises(ValueError, match='golden'):
        launch.check_golden(argv_of(arm, attempt), 'full', attempt)


@pytest.mark.parametrize('tier,backbone', [('M', 'simple'), ('S', 'simple'), ('L', 'cylindrical')])
def test_unseen_argv_refused_under_a_seen_root(tier, backbone):
    attempt = attempt_of('seen_simple')
    with pytest.raises(ValueError, match='golden'):
        launch.check_golden(launch.command('full', attempt, tier, backbone), 'full', attempt)


@pytest.mark.parametrize('tier,backbone,yaw', [('S', 'simple', 0), ('L', 'cylindrical', 0),
                                               ('M', 'cylindrical', 1), ('M', 'simple', 2)])
def test_command_refuses_unregistered_seen_arms(tier, backbone, yaw):
    with pytest.raises(ValueError, match='seen'):
        launch.command('full', attempt_of('seen_simple'), tier, backbone, 'seen', yaw)


def test_full_requires_the_seen_batch_count():
    args = launch.effective_args(argv_of('seen_simple'), '1', SEEN_BATCHES)
    launch.check_runtime(args, args, 'full')
    wrong = launch.effective_args(argv_of('seen_simple'), '1', 9261)
    with pytest.raises(ValueError, match='9265'):
        launch.check_runtime(wrong, wrong, 'full')
    unseen = launch.effective_args(launch.command('full', 'attempt'), '1', SEEN_BATCHES)
    with pytest.raises(ValueError, match='9261'):
        launch.check_runtime(unseen, unseen, 'full')


@pytest.mark.parametrize('arm', sorted(ARMS))
def test_banner_rule_per_arm(arm):
    expected = launch.effective_args(argv_of(arm), '1', SEEN_BATCHES)
    guard = launch.LogGuard(expected, 'full')
    enabled = 'yaw_aug ENABLED W=512 seed=0 counter=(epoch-1)*{}+batch_idx'
    wanted = enabled.format(SEEN_BATCHES) if arm == 'seen_aug' else 'yaw_aug DISABLED'
    for other in (enabled.format(9261), 'yaw_aug DISABLED' if arm == 'seen_aug' else enabled.format(SEEN_BATCHES)):
        guard.feed(other)
        assert not guard.banner
    guard.feed(wanted)
    assert guard.banner


def test_arm_roots_and_ledger_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    roots = [launch.arm_root('M', backbone, 'seen', yaw) for backbone, yaw in ARMS.values()]
    assert roots == [tmp_path / 'ckpt/exp07' / arm for arm in ARMS]
    assert launch.arm_root('M', 'simple') == launch.ROOT  # the exp_04 root is unchanged
    with pytest.raises(ValueError, match='seen arm'):
        launch.arm_root('M', 'cylindrical', 'seen', 1)
    launch.account_hours(launch.create_attempt(roots[0], 'attempt_test'), 2, mode='full')
    assert [launch.full_hours(root) for root in roots] == [2, 0, 0]
def control_of(arm):
    control = json.loads((launch.REPO / CONTROLS[arm]).read_text())
    return control, control.get('env') or dict(XRIR_DATA_PATH=launch.DATA_ROOT,
                                               OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='1')


@pytest.mark.parametrize('arm', sorted(ARMS))
def test_control_parity_against_the_unseen_twin(arm):
    runtime = launch.effective_args(argv_of(arm), '1', SEEN_BATCHES)
    assert runtime['protocol'] == 'seen' and runtime['tier'] == 'M'
    assert runtime['yaw_aug'] == ARMS[arm][1] and runtime['backbone'] == ARMS[arm][0]
    control, env = control_of(arm)
    differences = launch.compare_control(runtime, control, env)
    assert differences['protocol'] == {'treatment': 'seen', 'control': 'unseen'}
    assert set(differences) <= {'protocol', 'save_dir', 'save_every', 'epoch_ckpt_every',
                                'PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES'}
    launch.compare_control(runtime, dict(control, yaw_aug_seed=None), env)  # exp_05 A1 default
    for other in sorted(set(ARMS) - {arm}):  # only the arm's own unseen twin is admissible
        with pytest.raises(ValueError, match='control mismatch'):
            launch.compare_control(runtime, control_of(other)[0], env)
    # a seen-protocol comparator is never an unseen twin (seen_aug's recorded 9261 refuses first)
    with pytest.raises(ValueError, match='twin|train_batches_per_epoch'):
        launch.compare_control(runtime, dict(control, protocol='seen'), env)


@pytest.mark.parametrize('key,value', [('lr', 0.002), ('num_workers', '12'), ('yaw_aug', 1),
                                       ('no_save', True), ('train_batches_per_epoch', 9261)])
def test_control_parity_refuses_unregistered_changes(key, value):
    runtime = launch.effective_args(argv_of('seen_simple'), '1', SEEN_BATCHES)
    runtime[key] = value
    with pytest.raises(ValueError, match=key if key != 'train_batches_per_epoch' else 'invalid'):
        launch.compare_control(runtime, *control_of('seen_simple'))


@pytest.fixture
def stub_inputs(monkeypatch):
    """Stub the expensive parts of build_fields; the seen bindings stay real."""
    files = sorted(launch.TRAIN_MINIMUM | {launch.SEEN_MODULE})
    monkeypatch.setattr(launch.p, 'source_closure', lambda module, repo: files)
    monkeypatch.setattr(launch.p, 'closure_record', lambda paths, commit, repo: ([
        dict(path=name, reviewed_blob_sha256='same', working_tree_sha256='same',
             commits_after_reviewed=[]) for name in paths], 'closure'))
    monkeypatch.setattr(launch.p, 'environment', lambda: {'executable': launch.PYTHON})
    monkeypatch.setattr(launch.p, 'git_state', lambda repo: {'dirty_outside_worklog': False})
    calls = []
    def inventory(root, protocol='unseen', cache_path=None):
        calls.append((root, protocol, str(cache_path)))
        return {'inventory_files': launch.TRAIN_FILES[protocol]}
    monkeypatch.setattr(launch.p, 'train_data_identity', inventory)
    return calls


@pytest.mark.parametrize('arm', sorted(ARMS))
def test_seen_fields_bind_split_inventory_and_protocol(stub_inputs, arm):
    fields = launch.build_fields(argv_of(arm), '1', 'commit', 'full')
    assert stub_inputs == [(launch.DATA_ROOT, 'seen',
                            str(launch.REPO / 'ckpt/exp07/train_inventory_seen.json'))]
    assert fields['protocol'] == 'seen' and fields['effective_args']['protocol'] == 'seen'
    assert fields['effective_args']['train_batches_per_epoch'] == SEEN_BATCHES
    assert fields['mutable_inputs']['seen_split'] == p.seen_split_identity(launch.REPO)
    assert fields['mutable_inputs']['control_args']['path'] == str(launch.REPO / CONTROLS[arm])
    assert fields['control_excluded_differences']['protocol']['control'] == 'unseen'
    assert launch.SEEN_MODULE in {r['path'] for r in fields['source_closures']['training']['files']}


def test_unseen_fields_keep_the_exp04_inventory_and_no_seen_binding(stub_inputs):
    fields = launch.build_fields(launch.command('smoke', 'attempt'), '1', 'commit', 'smoke')
    assert stub_inputs == [(launch.DATA_ROOT, 'unseen',
                            str(launch.REPO / 'ckpt/yaw_aug/train_inventory.json'))]
    assert 'protocol' not in fields and fields['effective_args']['protocol'] == 'unseen'
    assert 'seen_split' not in fields.get('mutable_inputs', {})


def test_bound_seen_split_is_revalidated(tmp_path):
    record = p.seen_split_identity(launch.REPO)
    fields = {'repo': str(launch.REPO), 'mutable_inputs': {'seen_split': record}}
    assert p.revalidate(fields) == [] and 'seen_split' in launch.SEEN_INPUTS
    fields['mutable_inputs']['seen_split'] = dict(record, sha256='0' * 64)
    assert p.revalidate(fields) == ['seen_split']
    from tools import exp04_eval_launch
    assert 'seen_split' in exp04_eval_launch.MUTABLE_INPUTS


def recovery_fields(arm='seen_simple'):
    attempt = attempt_of(arm)
    argv = argv_of(arm, attempt)
    closure = [dict(path=name) for name in sorted(launch.TRAIN_MINIMUM | {launch.SEEN_MODULE})]
    return dict(repo=str(launch.REPO), mode='full', reviewed_commit='a' * 40, command=argv,
                effective_args=launch.effective_args(argv, '1', SEEN_BATCHES),
                source_closures=dict(training=dict(files=closure), launcher=dict(files=closure)),
                resource_before={'gpu': '1'}, env={key: '' for key in launch.ENV_KEYS},
                train_data_identity=dict(inventory_files=launch.TRAIN_FILES['seen']),
                attempt_path=str((launch.REPO / attempt).resolve()),
                mutable_inputs=dict(seen_split=p.seen_split_identity(launch.REPO),
                                    **{name: {} for name in ('effective_args', 'train_inventory',
                                                             'control_args', 'probe_receipt')}))


@pytest.mark.parametrize('fault,message', [
    (None, 'launcher-log'), ('binding', 'required bindings'), ('digest', 'seen split'),
    ('count', 'training inventory count'), ('closure', 'closure minimum')])
def test_recovery_validates_the_seen_inputs(fault, message):
    fields = recovery_fields()
    if fault == 'binding':
        del fields['mutable_inputs']['seen_split']
    elif fault == 'digest':
        fields['mutable_inputs']['seen_split']['sha256'] = '0' * 64
    elif fault == 'count':
        fields['train_data_identity']['inventory_files'] = launch.TRAIN_FILES['unseen']
    elif fault == 'closure':
        files = fields['source_closures']['training']['files']
        fields['source_closures']['training']['files'] = [
            r for r in files if r['path'] != launch.SEEN_MODULE]
    # A clean seen manifest reaches the launcher-log check that follows these gates.
    with pytest.raises(ValueError, match=message):
        launch.recovery_evidence(fields, {}, launch.REPO / attempt_of('seen_simple'), None)


def test_refuse_test_covers_the_seen_refusals(monkeypatch):
    monkeypatch.setattr(launch.subprocess, 'Popen', lambda *a, **k: pytest.fail('spawned trainer'))
    refusals = launch.refusal_self_test()['refusals']
    assert {'seen_other_arm', 'seen_unseen_argv', 'seen_arm', 'seen_bpe', 'seen_control'} <= set(refusals)


def test_launch_script_passes_every_argument_through():
    script = launch.REPO / 'tools/exp04_launch.sh'
    assert subprocess.run(['bash', '-n', str(script)]).returncode == 0
    text = script.read_text()
    assert text.strip().endswith('-m tools.exp04_launcher "$@"')
    assert '--protocol seen' in text  # the documented seen invocations


@pytest.fixture
def seen_repo(tmp_path, monkeypatch):
    """A throwaway repository with the launcher's inputs; no GPU and no trainer."""
    real = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    for name in sorted(launch.TRAIN_MINIMUM | {launch.SEEN_MODULE, p.SEEN_SPLIT,
                       'tools/exp04_launcher.py', 'tools/exp04_probe.py', 'tools/exp04_launch.sh'}):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text('# fixture ' + name + '\n')
    assets = 'seen_protocol_results_assets/argv_golden_seen_simple_smoke.txt'
    (tmp_path / launch.EXP07_RECORD / assets).parent.mkdir(parents=True)
    (tmp_path / launch.EXP07_RECORD / assets).write_text(
        (real / launch.EXP07_RECORD / assets).read_text())
    for argv in (['init', '-q'], ['add', '-A'], ['-c', 'user.name=T', '-c', 'user.email=t@e',
                                                 'commit', '-qm', 'fixture']):
        subprocess.run(['git'] + argv, cwd=str(tmp_path), check=True)
    monkeypatch.setattr(launch.p, 'source_closure', lambda module, repo:
                        sorted(launch.TRAIN_MINIMUM | {launch.SEEN_MODULE})
                        if module == 'train_xRIR_backbone' else ['tools/exp04_launcher.py'])
    monkeypatch.setattr(launch.p, 'environment', lambda: {'executable': launch.PYTHON})
    monkeypatch.setattr(launch.p, 'train_data_identity', lambda root, protocol='unseen', cache_path=None:
                        dict(p._inventory([p.SEEN_SPLIT], tmp_path), protocol=protocol, split='train',
                             cache_key=str(cache_path)))
    monkeypatch.setattr(launch, 'resource_gate', lambda *a: {'gpu': '1'})
    return tmp_path


def test_seen_smoke_writes_the_pre_spawn_files(seen_repo, monkeypatch):
    attempt = seen_repo / 'ckpt/exp07/seen_simple/_smoke_test'
    def runner(argv, path, gpu, guard, deadline=None):
        assert argv[:2] == [launch.PYTHON, 'train_xRIR_backbone.py']
        assert {f.name for f in attempt.iterdir()} == {'effective_args.json', 'train_manifest.json',
                                                       'train_inventory.json'}
        expected = json.loads((attempt / 'effective_args.json').read_text())
        path.write_text('XRIR_RUNTIME_ARGS ' + json.dumps(expected) + '\nyaw_aug DISABLED\n' +
                        ''.join('Train Epoch: 1 [{}/3] loss 1.25\n'.format(i) for i in range(3)) +
                        'Test set (epoch 1): Average loss: 0.25 over 2 batches\n')
        guard.log_created = True
        guard.poll(path)
        return 0
    execute = launch.execute_attempt
    monkeypatch.setattr(launch, 'execute_attempt', lambda *a, **k: execute(*a, runner=runner, **k))
    launch.main(['smoke', '--protocol', 'seen', '--backbone', 'simple', '--reviewed-commit', 'HEAD',
                 '--timestamp', 'test', '--log-dir', str(seen_repo / 'logs')])
    manifest = json.loads((attempt / 'train_manifest.json').read_text())
    assert manifest['protocol'] == 'seen' and manifest['effective_args']['protocol'] == 'seen'
    assert manifest['command'] == launch.command('smoke', 'ckpt/exp07/seen_simple/_smoke_test',
                                                 'M', 'simple', 'seen', 0)
    assert manifest['mutable_inputs']['seen_split'] == p.seen_split_identity(seen_repo)
    assert manifest['train_data_identity']['protocol'] == 'seen'
    assert 'inventory' not in manifest['train_data_identity']
    assert (manifest['train_data_identity']['inventory_file']['sha256'] ==
            p.sha256_file(attempt / 'train_inventory.json'))
    assert launch.SEEN_MODULE in {r['path'] for r in manifest['source_closures']['training']['files']}
    assert json.loads((attempt / 'effective_args.json').read_text()) == manifest['effective_args']
    assert (attempt / 'completion.json').is_file() and not list(seen_repo.glob('ckpt/exp07/*/*_ABORTED_*'))
