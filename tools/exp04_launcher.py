"""Fail-closed training launcher for exp_04 (full runs require a reviewed commit)."""
import json
import datetime
import os
from pathlib import Path
import shlex
import sys
from unittest.mock import patch

PYTHON = '/home/yixunhu/miniconda3/envs/xRIR/bin/python'
REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'ckpt/xRIR_simple_yawaug_8_shot'
GOLDENS = REPO / ('worklog/worklog_yixun/exp_04_yaw_aug_xrir_claude/'
                  'yaw_aug_xrir_results_assets')
ENV_KEYS = ('XRIR_DATA_PATH', 'PYTHONHASHSEED', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')
DATA_ROOT = '/home/yixunhu/data_cache/AcousticRooms'
REQUIRED_INPUTS = ('repo', 'source_closures', 'train_data_identity', 'effective_args', 'mutable_inputs')


def child_environment(gpu):
    return dict(os.environ, XRIR_DATA_PATH=DATA_ROOT, PYTHONHASHSEED='0',
                OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES=str(gpu),
                PYTHONPATH=str(REPO), PYTHONUNBUFFERED='1')


def command(mode, attempt):
    if mode not in ('smoke', 'full'):
        raise ValueError('command requires smoke or full')
    smoke = mode == 'smoke'
    result = [PYTHON, 'train_xRIR_backbone.py', '--backbone', 'simple', '--save-dir',
              str(attempt) + ('/smoke' if smoke else '')]
    result += shlex.split('--num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 '
                          '--decay-epochs 3 --lr-gamma 0.1 --epochs ' + ('1' if smoke else '12'))
    if smoke:
        result += shlex.split('--max-train-batches 3 --max-test-batches 2')
    result += shlex.split('--batch-size 4 --accum-steps 2 --num-workers 4' if smoke else
                          '--batch-size 32 --accum-steps 2 --num-workers 12')
    result += shlex.split('--seed 0 --tf32 --log-interval ' + ('1' if smoke else '50') +
                          ' --save-every 0 --epoch-ckpt-every ' + ('0 --no-save' if smoke else '1'))
    return result + shlex.split('--yaw-aug 1 --yaw-aug-seed 0 --yaw-aug-width 512')


def check_golden(argv, mode, attempt):
    golden = shlex.split((GOLDENS / ('argv_golden_' + mode + '.txt')).read_text())
    expected = [part.replace('<attempt>', str(attempt)) for part in golden]
    if argv != expected:
        raise ValueError('argv differs from golden ' + mode)


def effective_args(argv, gpu, batches_per_epoch):
    import train_xRIR_backbone as trainer
    with patch.object(sys, 'argv', argv[1:]):
        result = vars(trainer.parse_args())
    result.update({key: child_environment(gpu)[key] for key in ENV_KEYS})
    result['train_batches_per_epoch'] = batches_per_epoch
    return result


def normalize(runtime):
    result = dict(runtime)
    env = result.pop('env', {})
    for key, value in env.items():
        if key in result and (type(result[key]) is not type(value) or result[key] != value):
            raise ValueError('conflicting env.' + key)
        result[key] = value
    return result


def check_runtime(runtime, expected, mode):
    actual = normalize(runtime)
    differences = [key for key in actual.keys() | expected.keys() if key not in actual or
                   key not in expected or type(actual[key]) is not type(expected[key]) or
                   actual[key] != expected[key]]
    if differences:
        raise LauncherFailure('guard_runtime_args', 'runtime args mismatch: ' + ', '.join(sorted(differences)))
    if mode == 'full' and actual['train_batches_per_epoch'] != 9261:
        raise ValueError('full requires train_batches_per_epoch == 9261')


import fcntl
import math
import subprocess
import uuid
from tools import provenance as p


def gpu_snapshot(gpu):
    options = ['nvidia-smi', '-i', str(gpu), '--format=csv,noheader,nounits']
    raw = subprocess.check_output(options + ['--query-gpu=uuid,memory.free,utilization.gpu'], text=True)
    identifier, free, utilization = raw.strip().split(',')
    apps = subprocess.check_output(options +
        ['--query-compute-apps=pid,process_name,used_gpu_memory'], text=True).strip()
    return dict(gpu=str(gpu), uuid=identifier.strip(), free_gib=float(free) / 1024,
                utilization_gpu=float(utilization), compute_apps=apps, query_gpu=raw.strip())


def resource_gate(gpu, volume, mode, allow_cotenant=False):
    if allow_cotenant and mode == 'full':
        raise ValueError('full forbids --allow-cotenant')
    state = gpu_snapshot(gpu)
    free = int(subprocess.check_output(['df', '-B1', '--output=avail', str(volume)],
                                      text=True).splitlines()[-1]) / 2**30
    state['disk_free_gib'] = free
    if free < 50:
        raise ValueError('checkpoint volume needs >= 50 GiB')
    if not allow_cotenant and (state['compute_apps'] or state['free_gib'] < 40):
        raise ValueError('GPU needs no other processes and >= 40 GiB free')
    return state


def create_attempt(root, name):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    attempt = root / name
    attempt.mkdir()
    return attempt


def hours_record(root):
    path = Path(root) / 'cumulative_hours.json'
    result = json.loads(path.read_text()) if path.exists() else {'total_hours': 0, 'attempts': []}
    hours = result['total_hours']
    if not math.isfinite(hours) or hours < 0 or hours != sum(r['hours'] for r in result['attempts']):
        raise ValueError('invalid cumulative hours')
    for row in result['attempts']:
        name = row['attempt']
        row.setdefault('mode', 'probe' if '_probe_' in name else 'smoke' if name.startswith('_smoke_') else 'full')
        if row['mode'] not in ('full', 'smoke', 'probe') or not math.isfinite(row['hours']) or row['hours'] < 0:
            raise ValueError('invalid cumulative hours row')
    return result


def full_hours(root):
    return sum(row['hours'] for row in hours_record(root)['attempts'] if row['mode'] == 'full')


def check_budget(root, projection):
    if not math.isfinite(projection) or projection <= 0:
        raise ValueError('projection must be finite and positive')
    if full_hours(root) + projection > 43:
        raise ValueError('cumulative hours + projection exceeds 43 h')


def account_hours(attempt, hours, previous_name=None, mode='full'):
    with (attempt.parent / '.hours.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = hours_record(attempt.parent)
        result['attempts'] = [row for row in result['attempts']
                              if row['attempt'] not in (attempt.name, previous_name)]
        result['attempts'].append({'attempt': attempt.name, 'hours': hours, 'mode': mode})
        result['total_hours'] = sum(row['hours'] for row in result['attempts'])
        p.write_completion(attempt.parent / 'cumulative_hours.json', result)


def abort_attempt(attempt, reason, hours, mode='full'):
    aborted = attempt.with_name(attempt.name + '_ABORTED_' + reason)
    if aborted.exists():
        aborted = aborted.with_name(aborted.name + '_' + uuid.uuid4().hex)
    attempt.rename(aborted)
    account_hours(aborted, hours, previous_name=attempt.name, mode=mode)
    return aborted


def promote(attempt):
    if not (attempt / 'completion.json').is_file():
        raise ValueError('promotion requires completion')
    temporary = attempt.parent / ('.final_' + uuid.uuid4().hex)
    try:
        temporary.symlink_to(attempt.name)
        os.replace(temporary, attempt.parent / 'final')
    finally:
        if temporary.is_symlink():
            temporary.unlink()


TRAIN_MINIMUM = {'train_xRIR_backbone.py', 'treble_multi_room_dataset/treble_xRIR_dataset.py',
    'model/xRIR.py', 'model/xRIR_cyl.py', 'model/simple_vit.py', 'model/cylindrical_vit.py',
    'utils/spec_utils.py', 'utils/lr_scheduler.py', 'tools/yaw_aug.py', 'tools/yaw_rotation.py'}



CONTROL_EXCLUSIONS = {'save_dir', 'yaw_aug', 'yaw_aug_seed', 'yaw_aug_width', 'no_save',
                      'save_every', 'epoch_ckpt_every', 'PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES'}


def compare_control(runtime, control, control_env):
    treatment, baseline = normalize(runtime), normalize(dict(control, env=control_env))
    baseline.setdefault('train_batches_per_epoch', math.ceil(296334 / baseline['batch_size']))
    missing = object()
    differences = {key: {'treatment': treatment.get(key), 'control': baseline.get(key)}
        for key in treatment.keys() | baseline.keys()
        if type(treatment.get(key, missing)) is not type(baseline.get(key, missing))
        or treatment.get(key, missing) != baseline.get(key, missing)}
    refused = differences.keys() - CONTROL_EXCLUSIONS
    if refused:
        raise ValueError('control mismatch: ' + ', '.join(sorted(refused)))
    return differences


def build_fields(argv, gpu, reviewed_commit, mode, allow_dirty=False):
    state = p.checked_git_state(REPO, mode == 'full', allow_dirty)
    files = p.source_closure('train_xRIR_backbone', REPO)
    if not TRAIN_MINIMUM <= set(files):
        raise ValueError('training closure missing required files')
    closures = {}
    for role, paths in [('training', files), ('launcher',
            p.source_closure('tools.exp04_launcher', REPO) + ['tools/exp04_probe.py', 'tools/exp04_launch.sh'])]:
        records, digest = p.closure_record(paths, reviewed_commit, REPO)
        if not records or any(r['reviewed_blob_sha256'] is None or
                r['reviewed_blob_sha256'] != r['working_tree_sha256'] or
                r['commits_after_reviewed'] for r in records):
            raise ValueError(role + ' closure differs from reviewed commit')
        closures[role] = {'files': records, 'sha256': digest}
    print('Hashing training data identity...', flush=True)
    data = p.train_data_identity(DATA_ROOT, cache_path=REPO / 'ckpt/yaw_aug/train_inventory.json')
    provisional = effective_args(argv, gpu, 1)
    bpe = math.ceil(data['inventory_files'] / provisional['batch_size'])
    effective = effective_args(argv, gpu, bpe)
    check_runtime(effective, effective, mode)
    fields = dict(repo=str(REPO), reviewed_commit=reviewed_commit, mode=mode,
        source_closures=closures, train_data_identity=data, effective_args=effective,
        command=argv, environment=p.environment(), git_state=state, allow_dirty=allow_dirty,
        env={key: child_environment(gpu)[key] for key in ENV_KEYS})
    if mode == 'full':
        control_path = REPO / 'ckpt/xRIR_simple_8_shot/args.json'
        control_env = dict(XRIR_DATA_PATH=DATA_ROOT, OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='1')
        fields['control_excluded_differences'] = compare_control(effective, json.loads(control_path.read_text()), control_env)
        fields['control_env_reconstructed'] = control_env
        fields['mutable_inputs'] = {'control_args': {'path': str(control_path), 'sha256': p.sha256_file(control_path)}}
    return fields


import re
import signal
import time
from contextlib import contextmanager


class LauncherTerminated(BaseException):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(signal.Signals(signum).name)


class LauncherFailure(ValueError):
    def __init__(self, reason, message):
        self.reason = reason
        super().__init__(message)


@contextmanager
def termination_handlers():
    def terminate(signum, frame):
        raise LauncherTerminated(signum)
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        for signum in previous:
            signal.signal(signum, terminate)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


class LogGuard:
    def __init__(self, expected, mode):
        self.expected, self.mode = expected, mode
        self.runtime = None
        self.banner = False
        self.steps, self.losses = [], []
        self.test_loss = None
        self.position = 0
        self.first_step_time = None
        self.epoch_one_done = False
        self.probe = None
        self.log_created = False

    def feed(self, line):
        if line.startswith('EXP04_PROBE_RESULT '):
            self.probe = json.loads(line[len('EXP04_PROBE_RESULT '):])
        if line.startswith('XRIR_RUNTIME_ARGS '):
            self.runtime = json.loads(line[len('XRIR_RUNTIME_ARGS '):])
            check_runtime(self.runtime, self.expected, self.mode)
        wanted = ('yaw_aug ENABLED W={yaw_aug_width} seed={yaw_aug_seed} '
                  'counter=(epoch-1)*{train_batches_per_epoch}+batch_idx'.format(**self.expected)
                  if self.expected['yaw_aug'] else 'yaw_aug DISABLED')
        if line.strip() == wanted:
            if self.banner:
                raise LauncherFailure('guard_banner', 'duplicate banner')
            self.banner = True
        step = re.search(r'Train Epoch: (\d+) \[(\d+)/(\d+)\].*?loss (\S+)', line)
        if step:
            if not self.banner or self.runtime is None:
                raise LauncherFailure('guard_banner', 'first step without banner or runtime args')
            if self.mode == 'full':
                runtime = REPO / self.expected['save_dir'] / 'args.json'
                check_runtime(json.loads(runtime.read_text()), self.expected, self.mode)
            values = re.findall(r'(?:loss |stft |decay )([^\s,)]+)', line)
            if not values or not all(math.isfinite(float(value)) for value in values):
                raise ValueError('non-finite training loss')
            self.first_step_time = self.first_step_time or time.monotonic()
            self.steps.append((int(step[1]), int(step[2])))
            self.losses.append(float(step[4]))
        if line.startswith('Test set (epoch 1):'):
            match = re.search(r'Average loss: (\S+) over (\d+) batches', line)
            self.test_loss = float(match[1])
            if not math.isfinite(self.test_loss) or (self.mode == 'smoke' and int(match[2]) != 2):
                raise ValueError('invalid test loss/count')
        if line.startswith('epoch 1 done'):
            self.epoch_one_done = True
            if self.mode == 'full':
                history = REPO / self.expected['save_dir'] / 'history.jsonl'
                first = json.loads(history.read_text().splitlines()[0])
                if not math.isfinite(first['epoch_minutes']) or first['epoch_minutes'] > 2.431 * 60:
                    raise LauncherFailure('guard_epoch_one', 'epoch one exceeds 2.431 h acceptance')

    def poll(self, path):
        with path.open() as stream:
            stream.seek(self.position)
            while True:
                line = stream.readline()
                if not line.endswith('\n'):
                    break
                self.position = stream.tell()
                self.feed(line)
        if (self.mode == 'full' and self.first_step_time and not self.epoch_one_done
                and time.monotonic() - self.first_step_time > 2.6 * 3600):
            raise LauncherFailure('guard_epoch_one', 'epoch one exceeds 2.6 h operational ceiling')

    def finish(self):
        if not self.banner or not self.runtime or not self.steps:
            raise LauncherFailure('guard_banner', 'missing runtime args, banner or steps')
        if self.mode == 'smoke' and (self.steps != [(1, 0), (1, 1), (1, 2)] or self.test_loss is None):
            raise ValueError('smoke requires three finite steps and test loss')
        if self.mode == 'probe' and (not self.probe or self.probe.get('yaw_aug') != self.expected['yaw_aug']
                or self.probe.get('warmup_micro_batches') != 10 or self.probe.get('timed_micro_batches') != 50
                or not math.isfinite(self.probe.get('mean_iteration_seconds', float('nan')))
                or self.probe['mean_iteration_seconds'] <= 0):
            raise ValueError('missing or invalid probe result')
        return dict(train_losses=self.losses, test_loss=self.test_loss, banner=self.banner, probe=self.probe)


def run_child(argv, log_path, gpu, guard, deadline=None):
    child = sink = None
    with log_path.open('xb') as log:
        guard.log_created = True
        try:
            child = subprocess.Popen(argv, cwd=REPO, env=child_environment(gpu),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            sink = subprocess.Popen(['tee', '-a', str(log_path)], stdin=child.stdout, stdout=2,
                                    start_new_session=True)
            child.stdout.close()
            while child.poll() is None or sink.poll() is None:
                guard.poll(log_path)
                if deadline is not None and time.monotonic() > deadline:
                    raise LauncherFailure('deadline_43h', '43 h cumulative ceiling reached')
                if sink.poll() not in (None, 0):
                    raise RuntimeError('log sink failed')
                time.sleep(0.05)
            log.flush()
            os.fsync(log.fileno())
            guard.poll(log_path)
            if sink.returncode:
                raise RuntimeError('log sink failed')
            return child.returncode
        finally:
            for process in (child, sink):
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()


def validate_outputs(attempt, mode, expected):
    if mode != 'full':
        if {f.name for f in attempt.iterdir()} != {'effective_args.json', 'train_manifest.json'}:
            raise ValueError('no-save run wrote unexpected outputs')
        return {}
    check_runtime(json.loads((attempt / 'args.json').read_text()), expected, mode)
    history = [json.loads(line) for line in (attempt / 'history.jsonl').read_text().splitlines()]
    if [r['epoch'] for r in history] != list(range(1, 13)) or any(
            not math.isfinite(r[key]) for r in history for key in ('train_loss', 'test_loss', 'epoch_minutes')):
        raise ValueError('incomplete or non-finite history')
    if history[0]['epoch_minutes'] > 2.431 * 60:
        raise LauncherFailure('guard_epoch_one', 'epoch one exceeds 2.431 h acceptance')
    epochs = sorted(attempt.glob('epoch_*.pth'))
    if [f.name for f in epochs] != ['epoch_%03d.pth' % i for i in range(1, 13)]:
        raise ValueError('missing or unexpected epoch checkpoints')
    return {f.name: p.sha256_file(f) for f in epochs + [attempt / 'history.jsonl', attempt / 'args.json']}


def abort_log(log_path, reason, created):
    if not created:
        return None
    destination = log_path.with_name(log_path.stem + '_ABORTED_' + reason + '.log')
    if destination.exists():
        destination = destination.with_name(destination.stem + '_' + uuid.uuid4().hex + '.log')
    try:
        log_path.rename(destination)
        return str(destination.resolve())
    except OSError:
        return None


@termination_handlers()
def execute_attempt(attempt, mode, gpu, log_path, fields_factory, allow_cotenant=False,
                    projection=30.0, runner=run_child):
    attempt, log_path = Path(attempt), Path(log_path)
    if mode == 'full':
        check_budget(attempt.parent, projection)
    create_attempt(attempt.parent, attempt.name)  # An existing directory is never renamed.
    started, reason = time.monotonic(), 'setup_failed'
    started_at, guard = datetime.datetime.now().astimezone().isoformat(), None
    try:
        if attempt == log_path or attempt in log_path.parents:
            raise ValueError('log must be outside attempt')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        before = resource_gate(gpu, attempt.parent, mode, allow_cotenant)
        fields = fields_factory()
        expected = fields['effective_args']
        effective_path = attempt / 'effective_args.json'
        effective_digest = p.write_manifest(effective_path, expected)
        fields.setdefault('mutable_inputs', {})['effective_args'] = {
            'path': str(effective_path.resolve()), 'sha256': effective_digest}
        fields.update(resource_before=before, allow_cotenant=allow_cotenant)
        manifest_path = attempt / 'train_manifest.json'
        digest = p.write_manifest(manifest_path, fields)
        if {f.name for f in attempt.iterdir()} != {'effective_args.json', 'train_manifest.json'}:
            raise ValueError('unexpected pre-spawn files')
        guard = LogGuard(expected, mode)
        reason = 'child_failed'
        deadline = started + (43 - full_hours(attempt.parent)) * 3600 if mode == 'full' else None
        status = runner(fields['command'], log_path, gpu, guard, deadline)
        if status:
            raise LauncherFailure('child_exit_' + str(status), 'child exited nonzero: ' + str(status))
        metrics = guard.finish()
        reason = 'invalid_outputs'
        outputs = validate_outputs(attempt, mode, expected)
        reason = 'input_changed'
        fields['mutable_inputs']['train_manifest'] = {'path': str(manifest_path.resolve()), 'sha256': digest}
        print('Revalidating training inputs after log close...', flush=True)
        drift = []
        mismatches = p.revalidate(fields, required=REQUIRED_INPUTS, source_drift=drift)
        if mismatches:
            raise ValueError('mutable input mismatch: ' + ', '.join(mismatches))
        completion = dict(train_manifest_sha256=digest, source_drift_after_spawn=drift,
            log={'path': str(log_path.resolve()), 'sha256': p.sha256_file(log_path)},
            outputs=outputs, metrics=metrics, wall_hours=(time.monotonic() - started) / 3600)
        reason = 'completion_failed'
        p.write_completion(attempt / 'completion.json', completion)
        account_hours(attempt, completion['wall_hours'], mode=mode)
        if mode == 'full':
            promote(attempt)
        return completion
    except BaseException as error:
        reason = ('terminated_' + signal.Signals(error.signum).name if isinstance(error, LauncherTerminated)
                  else getattr(error, 'reason', reason))
        hours = (time.monotonic() - started) / 3600
        renamed = abort_log(log_path, reason, guard is not None and guard.log_created)
        p.write_completion(attempt / 'abort.json', dict(reason=reason,
            exception_type=type(error).__name__, exception_message=str(error), started_at=started_at,
            aborted_at=datetime.datetime.now().astimezone().isoformat(), wall_hours=hours,
            last_guard_state={k: v for k, v in vars(guard).items() if k != 'expected'} if guard else None,
            log={'original': str(log_path.resolve()), 'aborted': renamed}))
        completion_path = attempt / 'completion.json'
        if completion_path.exists():
            completion_path.unlink()
        aborted = abort_attempt(attempt, reason, hours, mode=mode)
        print('ABORTED ' + str(aborted), flush=True)
        raise


def refusal_self_test():
    """Exercise operational refusals in scratch storage, with no trainer spawn."""
    import tempfile
    refusals = []
    def refuses(name, action):
        try:
            action()
        except (ValueError, FileExistsError):
            refusals.append(name)
        else:
            raise AssertionError('refusal not enforced: ' + name)
    with tempfile.TemporaryDirectory(prefix='exp04_refuse_') as temporary:
        root = Path(temporary)
        attempt = create_attempt(root, 'attempt_test')
        refuses('exclusive_directory', lambda: create_attempt(root, attempt.name))
        argv = command('full', str(attempt))
        argv[argv.index('--lr') + 1] = '0.5'
        refuses('golden_argv', lambda: check_golden(argv, 'full', str(attempt)))
        expected = effective_args(command('full', str(attempt)), '1', 9261)
        refuses('runtime_schema', lambda: check_runtime(dict(expected, lr='0.001'), expected, 'full'))
        invalid = dict(expected, train_batches_per_epoch=9260)
        refuses('full_bpe', lambda: check_runtime(invalid, invalid, 'full'))
        refuses('banner', lambda: LogGuard(expected, 'full').feed('Train Epoch: 1 [0/9261] loss 1'))
        refuses('budget', lambda: check_budget(root, 44))
        refuses('full_cotenant', lambda: resource_gate('1', root, 'full', True))
        with patch(__name__ + '.gpu_snapshot', return_value={'compute_apps': 'foreign', 'free_gib': 39}), \
                patch.object(subprocess, 'check_output', return_value='Avail\n' + str(50 * 2**30)):
            refuses('gpu', lambda: resource_gate('1', root, 'smoke'))
        abort_attempt(attempt, 'test', 1)
        refuses('cumulative_budget', lambda: check_budget(root, 42.1))
    return {'passed': True, 'refusals': refusals}


def probe_receipt(path, commit, gpu):
    from tools.exp04_probe import compare_results
    path = Path(path).resolve()
    digest = p.sha256_file(path)
    receipt = json.loads(path.read_text())
    try:
        measured = compare_results(receipt['yaw_off'], receipt['yaw_on'])
        valid = (receipt['reviewed_commit'] == commit and receipt['before']['gpu'] == gpu
            and receipt['after']['gpu'] == gpu and receipt['PROBE_NOT_CLEAN'] is False
            and receipt['passed'] is True and measured['passed']
            and receipt['overhead_ratio'] == measured['overhead_ratio'])
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid or p.sha256_file(path) != digest:
        raise ValueError('full requires a clean passing probe bound to reviewed commit and GPU')
    return {'path': str(path), 'sha256': digest}


@termination_handlers()
def main(argv=None):
    import argparse
    import datetime
    from tools.exp04_probe import trainer_command, compare_results
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'probe', 'full', 'refuse-test'))
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--reviewed-commit')
    parser.add_argument('--log-dir')
    parser.add_argument('--timestamp', default=datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f'))
    parser.add_argument('--allow-cotenant', action='store_true')
    parser.add_argument('--allow-dirty', action='store_true')
    parser.add_argument('--projection-hours', type=float, default=30.0)
    parser.add_argument('--probe-json', help='full requires a clean passing probe receipt')
    args = parser.parse_args(argv)
    if args.mode == 'refuse-test':
        result = refusal_self_test()
        print(json.dumps(result), flush=True)
        return result
    if not args.reviewed_commit or not args.log_dir:
        parser.error('--reviewed-commit and --log-dir are required')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.timestamp):
        parser.error('timestamp must be a safe filename component')
    if Path(sys.executable).resolve() != Path(PYTHON).resolve():
        parser.error('launcher requires pinned interpreter ' + PYTHON)
    commit = subprocess.check_output(['git', 'rev-parse', args.reviewed_commit + '^{commit}'],
                                     cwd=REPO, text=True).strip()
    if args.mode == 'full':
        if args.allow_cotenant:
            parser.error('full forbids --allow-cotenant')
        if not args.probe_json:
            parser.error('full requires --probe-json (clean passing probe)')
        receipt = probe_receipt(args.probe_json, commit, args.gpu)
        check_budget(ROOT, args.projection_hours)
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / '.launch.lock').open('a') as lock, patch.dict(os.environ, child_environment(args.gpu)):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp, mode = args.timestamp, args.mode
        log_dir = Path(args.log_dir).resolve()
        if mode in ('smoke', 'full'):
            attempt = ROOT / (('_smoke_' if mode == 'smoke' else 'attempt_') + stamp)
            relative = os.path.relpath(attempt, REPO)
            cmd = command(mode, relative)
            check_golden(cmd, mode, relative)
            def fields_factory():
                fields = build_fields(cmd, args.gpu, commit, mode, args.allow_dirty)
                if mode == 'full':
                    fields['mutable_inputs']['probe_receipt'] = receipt
                return fields
            result = execute_attempt(attempt, mode, args.gpu,
                log_dir / ('yaw_aug_xrir_' + stamp + '_train_' + mode + '.log'),
                fields_factory,
                allow_cotenant=args.allow_cotenant, projection=args.projection_hours)
        else:
            output = ROOT / ('_probe_' + stamp + '.json')
            if output.exists():
                raise FileExistsError(str(output))
            before = gpu_snapshot(args.gpu)
            measurements, attempts = [], []
            for yaw, label in ((0, 'off'), (1, 'on')):
                attempt = ROOT / ('attempt_' + stamp + '_probe_' + label)
                relative = os.path.relpath(attempt, REPO)
                cmd = [PYTHON, '-m', 'tools.exp04_probe', '--yaw-aug', str(yaw), '--save-dir', relative]
                def fields_factory():
                    fields = build_fields([PYTHON] + trainer_command(yaw, relative), args.gpu, commit, 'probe', args.allow_dirty)
                    fields['trainer_command'], fields['command'] = fields['command'], cmd
                    return fields
                completed = execute_attempt(attempt, 'probe', args.gpu,
                    log_dir / ('yaw_aug_xrir_' + stamp + '_' + label + '_train_probe.log'),
                    fields_factory, allow_cotenant=args.allow_cotenant)
                measurements.append(completed['metrics']['probe'])
                attempts.append(str(attempt))
            after = gpu_snapshot(args.gpu)
            result = dict(compare_results(*measurements), yaw_off=measurements[0], yaw_on=measurements[1],
                before=before, after=after, PROBE_NOT_CLEAN=bool(before['compute_apps'] or after['compute_apps']),
                attempts=attempts, reviewed_commit=commit)
            p.write_manifest(output, result)
            if result['PROBE_NOT_CLEAN']:
                print('PROBE_NOT_CLEAN: another process was present; timing gate needs Planner judgment.', flush=True)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        if mode == 'probe' and not result['passed']:
            raise SystemExit(1)
        return result


if __name__ == '__main__':
    main()
