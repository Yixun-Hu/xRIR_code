"""Fail-closed training launcher for exp_04 (full runs require a reviewed commit)."""
import json
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
        raise ValueError('runtime args mismatch: ' + ', '.join(sorted(differences)))
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
    return result


def check_budget(root, projection):
    if not math.isfinite(projection) or projection <= 0:
        raise ValueError('projection must be finite and positive')
    if hours_record(root)['total_hours'] + projection > 43:
        raise ValueError('cumulative hours + projection exceeds 43 h')


def account_hours(attempt, hours):
    with (attempt.parent / '.hours.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = hours_record(attempt.parent)
        if any(row['attempt'] == attempt.name for row in result['attempts']):
            return
        result['attempts'].append({'attempt': attempt.name, 'hours': hours})
        result['total_hours'] = sum(row['hours'] for row in result['attempts'])
        p.write_completion(attempt.parent / 'cumulative_hours.json', result)


def abort_attempt(attempt, reason, hours):
    aborted = attempt.with_name(attempt.name + '_ABORTED_' + reason)
    if aborted.exists():
        aborted = aborted.with_name(aborted.name + '_' + uuid.uuid4().hex)
    attempt.rename(aborted)
    account_hours(aborted, hours)
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


def build_fields(argv, gpu, reviewed_commit, mode):
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
    data = p.train_data_identity(DATA_ROOT)
    provisional = effective_args(argv, gpu, 1)
    bpe = math.ceil(data['inventory_files'] / provisional['batch_size'])
    effective = effective_args(argv, gpu, bpe)
    check_runtime(effective, effective, mode)
    return dict(repo=str(REPO), reviewed_commit=reviewed_commit, mode=mode,
        source_closures=closures, train_data_identity=data, effective_args=effective,
        command=argv, environment=p.environment(), git_state=p.git_state(REPO),
        env={key: child_environment(gpu)[key] for key in ENV_KEYS})


import re
import signal
import time


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

    def feed(self, line):
        if line.startswith('XRIR_RUNTIME_ARGS '):
            self.runtime = json.loads(line[len('XRIR_RUNTIME_ARGS '):])
            check_runtime(self.runtime, self.expected, self.mode)
        wanted = ('yaw_aug ENABLED W=512 seed=0 counter=(epoch-1)*{}+batch_idx'.format(
            self.expected['train_batches_per_epoch']) if self.expected['yaw_aug'] else 'yaw_aug DISABLED')
        if line.strip() == wanted:
            if self.banner:
                raise ValueError('duplicate banner')
            self.banner = True
        step = re.search(r'Train Epoch: (\d+) \[(\d+)/(\d+)\].*?loss (\S+)', line)
        if step:
            if not self.banner or self.runtime is None:
                raise ValueError('first step without banner or runtime args')
            if self.mode == 'full':
                runtime = Path(self.expected['save_dir']) / 'args.json'
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
            raise ValueError('epoch one exceeds 2.6 h operational ceiling')

    def finish(self):
        if not self.banner or not self.runtime or not self.steps:
            raise ValueError('missing runtime args, banner or steps')
        if self.mode == 'smoke' and (self.steps != [(1, 0), (1, 1), (1, 2)] or self.test_loss is None):
            raise ValueError('smoke requires three finite steps and test loss')
        return dict(train_losses=self.losses, test_loss=self.test_loss, banner=self.banner)


def run_child(argv, log_path, gpu, guard, deadline=None):
    child = sink = None
    with log_path.open('xb') as log:
        try:
            child = subprocess.Popen(argv, cwd=REPO, env=child_environment(gpu),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            sink = subprocess.Popen(['tee', '/dev/stderr'], stdin=child.stdout, stdout=log,
                                    start_new_session=True)
            child.stdout.close()
            while child.poll() is None or sink.poll() is None:
                guard.poll(log_path)
                if deadline is not None and time.monotonic() > deadline:
                    raise ValueError('43 h cumulative ceiling reached')
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
