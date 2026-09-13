"""Fail-closed training launcher for exp_04 (full runs require a reviewed commit)."""
import json
import datetime
import os
from pathlib import Path
import shlex
import sys
from unittest.mock import patch
from functools import lru_cache
from tools.exp05_params import TIERS, build_tier, count_parameters, tier_of
from tools import exp05_probe as tier_probe
from tools import exp05_gates as tier_gates

PYTHON = '/home/yixunhu/miniconda3/envs/xRIR/bin/python'
REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'ckpt/xRIR_simple_yawaug_8_shot'
GOLDENS = REPO / ('worklog/worklog_yixun/exp_04_yaw_aug_xrir_claude/'
                  'yaw_aug_xrir_results_assets')
ENV_KEYS = ('XRIR_DATA_PATH', 'PYTHONHASHSEED', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')
DATA_ROOT = '/home/yixunhu/data_cache/AcousticRooms'
TIER_RECORD = 'worklog/worklog_yixun/exp_05_param_efficiency_claude'
REQUIRED_INPUTS = ('repo', 'source_closures', 'train_data_identity', 'effective_args', 'mutable_inputs')


def child_environment(gpu):
    return dict(os.environ, XRIR_DATA_PATH=DATA_ROOT, PYTHONHASHSEED='0',
                OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES=str(gpu),
                PYTHONPATH=str(REPO), PYTHONUNBUFFERED='1')


def arm_root(tier, backbone):
    return ROOT if tier == 'M' else REPO / 'ckpt/exp05' / (tier + '_' + backbone)


def command(mode, attempt, tier='M', backbone='simple'):
    if mode not in ('smoke', 'full'):
        raise ValueError('command requires smoke or full')
    smoke = mode == 'smoke'
    if (tier == 'M' and backbone != 'simple') or (tier != 'M' and smoke):
        raise ValueError('M uses exp04 simple; S/L currently provide full argv only')
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
    if tier != 'M':
        result[result.index('--backbone') + 1] = backbone
        return result + [part for key, value in TIERS[tier].items()
                         for part in ('--vit-' + key.replace('_', '-'), str(value))]
    return result + shlex.split('--yaw-aug 1 --yaw-aug-seed 0 --yaw-aug-width 512')


def check_golden(argv, mode, attempt):
    path, placeholder = GOLDENS / ('argv_golden_' + mode + '.txt'), '<attempt>'
    if '--vit-dim' in argv:
        if mode != 'full':
            raise ValueError('tier golden requires full mode')
        try:
            tier = tier_of({'vit_' + key: int(argv[argv.index('--vit-' + key.replace('_', '-')) + 1])
                            for key in TIERS['M']})
        except (ValueError, IndexError) as error:
            raise ValueError('argv differs from golden tier') from error
        backbone = argv[argv.index('--backbone') + 1]
        prefix = 'ckpt/exp05/{}_{}/attempt_'.format(tier, backbone)
        if not str(attempt).startswith(prefix) or not re.fullmatch(r'[A-Za-z0-9_-]+', str(attempt)[len(prefix):]):
            raise ValueError('argv differs from golden tier arm')
        path = REPO / TIER_RECORD / 'param_efficiency_results_assets' / ('argv_golden_{}_{}.txt'.format(tier, backbone))
        placeholder = 'ckpt/exp05/{}_{}/attempt_<ts>'.format(tier, backbone)
    elif (REPO / 'ckpt/exp05').resolve() in (REPO / attempt).resolve().parents:
        raise ValueError('argv differs from golden tier arm')
    if not path.is_file():
        raise ValueError('golden file missing: ' + path.name)
    golden = shlex.split(path.read_text())
    expected = [part.replace(placeholder, str(attempt)) for part in golden]
    if argv != expected:
        raise ValueError('argv differs from golden ' + mode)


def effective_args(argv, gpu, batches_per_epoch):
    import train_xRIR_backbone as trainer
    with patch.object(sys, 'argv', argv[1:]):
        result = vars(trainer.parse_args())
    result.update({key: child_environment(gpu)[key] for key in ENV_KEYS})
    result['train_batches_per_epoch'] = batches_per_epoch
    result.update(tier=tier_of(result), param_counts=dict(tier_counts(result['backbone'], tier_of(result))))
    return result


@lru_cache(None)
def tier_counts(backbone, tier):
    import torch
    with torch.random.fork_rng(devices=[]):
        return count_parameters(build_tier(backbone, tier))


def validate_parameters(values):
    tier = tier_of(values)
    if 'tier' in values and (type(values['tier']) is not str or values['tier'] != tier):
        raise ValueError('tier disagrees with vit_* fields')
    if 'param_counts' in values:
        counts = values['param_counts']
        if (type(counts) is not dict or counts != tier_counts(values['backbone'], tier)
                or any(type(value) is not int for value in counts.values())):
            raise ValueError('param_counts disagree with model')
    return tier


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
    if 'tier' in actual or 'param_counts' in actual:
        validate_parameters(actual)
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
        if 'mode' not in row:
            raise ValueError('legacy cumulative hours row missing mode; regenerate from attempt records')
        if row['mode'] not in ('full', 'smoke', 'probe') or not math.isfinite(row['hours']) or row['hours'] < 0:
            raise ValueError('invalid cumulative hours row')
    return result


def full_hours(root):
    return sum(row['hours'] for row in hours_record(root)['attempts'] if row['mode'] == 'full')


def check_budget(root, projection, ceiling=43):
    if not math.isfinite(projection) or projection <= 0:
        raise ValueError('projection must be finite and positive')
    if full_hours(root) + projection > ceiling:
        raise ValueError('cumulative hours + projection exceeds {} h'.format(ceiling))


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
    if '_ABORTED_' in attempt.name:
        raise ValueError('promotion requires a recovered attempt name')
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
    'tools/exp05_params.py',
    'model/xRIR.py', 'model/xRIR_cyl.py', 'model/simple_vit.py', 'model/cylindrical_vit.py',
    'utils/spec_utils.py', 'utils/lr_scheduler.py', 'tools/yaw_aug.py', 'tools/yaw_rotation.py'}



CONTROL_EXCLUSIONS = {'save_dir', 'yaw_aug', 'yaw_aug_seed', 'yaw_aug_width', 'no_save',
                      'save_every', 'epoch_ckpt_every', 'PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES'}


def compare_control(runtime, control, control_env):
    treatment, baseline = normalize(runtime), normalize(dict(control, env=control_env))
    for values in (treatment, baseline):
        for key, value in TIERS['M'].items():
            values.setdefault('vit_' + key, value)
        validate_parameters(values)
        for key in ('tier', 'param_counts'):
            values.pop(key, None)
        bpe = values.pop('train_batches_per_epoch', math.ceil(296334 / values['batch_size']))
        if type(bpe) is not int or bpe != math.ceil(296334 / values['batch_size']):
            raise ValueError('invalid train_batches_per_epoch')
    if tier_of(baseline) != 'M':
        raise ValueError('control requires tier M')
    exclusions = CONTROL_EXCLUSIONS
    if tier_of(treatment) != 'M':
        for values in (treatment, baseline):
            for key, value in dict(yaw_aug=0, yaw_aug_seed=values['seed'], yaw_aug_width=512, no_save=False).items():
                values.setdefault(key, value)
            if values['yaw_aug_seed'] is None:
                values['yaw_aug_seed'] = values['seed']
            if type(values['yaw_aug']) is not int or values['yaw_aug'] != 0:
                raise ValueError('exp05 requires yaw_aug=0')
        exclusions = (exclusions - {'yaw_aug', 'yaw_aug_seed', 'yaw_aug_width', 'no_save'}) | {'vit_' + key for key in TIERS['M']}
    missing = object()
    differences = {key: {'treatment': treatment.get(key), 'control': baseline.get(key)}
        for key in treatment.keys() | baseline.keys()
        if type(treatment.get(key, missing)) is not type(baseline.get(key, missing))
        or treatment.get(key, missing) != baseline.get(key, missing)}
    refused = differences.keys() - exclusions
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
        label = 'cyl' if effective['backbone'] == 'cylindrical' else 'simple'
        control_path = REPO / ('ckpt/xRIR_' + label + '_8_shot/args.json')
        control_env = dict(XRIR_DATA_PATH=DATA_ROOT, OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='1')
        fields['control_excluded_differences'] = compare_control(effective, json.loads(control_path.read_text()), control_env)
        fields['control_env_reconstructed'] = control_env
        fields['mutable_inputs'] = {'control_args': {'path': str(control_path), 'sha256': p.sha256_file(control_path)}}
    return fields


import re
import signal
import time
from tools.provenance import LauncherTerminated, termination_handlers


class LauncherFailure(ValueError):
    def __init__(self, reason, message):
        self.reason = reason
        super().__init__(message)


class LogGuard:
    def __init__(self, expected, mode, attempt=None, limits=None):
        self.expected, self.mode = expected, mode
        self.output_dir = Path(attempt) if attempt else REPO / expected.get('save_dir', '.')
        self.runtime = None
        self.banner = False
        self.steps, self.losses = [], []
        self.test_loss = None
        self.position = 0
        self.first_step_time = None
        self.epoch_one_done = False
        self.probe = None
        self.log_created = False
        self.epoch_limit_seconds = limits['epoch_seconds'] if limits else 2.431 * 3600
        self.live_epoch_limit_seconds = limits['epoch_seconds'] if limits else 2.6 * 3600
        self.tier_limits = limits

    def runtime_payload(self, payload):
        try:
            actual = json.loads(payload.read_text() if isinstance(payload, Path) else payload)
            check_runtime(actual, self.expected, self.mode)
            return actual
        except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
            raise LauncherFailure('guard_runtime_args', str(error)) from error

    def feed(self, line):
        prefix = 'EXP04_PROBE_RESULT ' if self.expected.get('tier', 'M') == 'M' else 'EXP05_PROBE_RESULT '
        if line.startswith(('EXP04_PROBE_RESULT ', 'EXP05_PROBE_RESULT ')):
            if not line.startswith(prefix):
                raise ValueError('probe result prefix differs from tier')
            self.probe = json.loads(line[len(prefix):])
        if line.startswith('XRIR_RUNTIME_ARGS '):
            self.runtime = self.runtime_payload(line[len('XRIR_RUNTIME_ARGS '):])
        wanted = ('yaw_aug ENABLED W={yaw_aug_width} seed={yaw_aug_seed} '
                  'counter=(epoch-1)*{train_batches_per_epoch}+batch_idx'.format(**self.expected)
                  if self.expected['yaw_aug'] else 'yaw_aug DISABLED')
        if line.strip() == wanted:
            if self.banner:
                raise LauncherFailure('guard_banner', 'duplicate banner')
            self.banner = True
            if self.tier_limits and self.mode == 'full':
                self.first_step_time = time.monotonic()
        step = re.search(r'Train Epoch: (\d+) \[(\d+)/(\d+)\].*?loss (\S+)', line)
        if step:
            if not self.banner or self.runtime is None:
                raise LauncherFailure('guard_banner', 'first step without banner or runtime args')
            if self.mode == 'full':
                runtime = self.output_dir / 'args.json'
                self.runtime_payload(runtime)
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
                history = self.output_dir / 'history.jsonl'
                first = json.loads(history.read_text().splitlines()[0])
                limit = self.epoch_limit_seconds / 60 if self.tier_limits else 2.431 * 60
                if not math.isfinite(first['epoch_minutes']) or first['epoch_minutes'] > limit:
                    raise LauncherFailure('guard_epoch_one', 'epoch one exceeds timing acceptance' if self.tier_limits
                                          else 'epoch one exceeds 2.431 h acceptance')

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
                and time.monotonic() - self.first_step_time > self.live_epoch_limit_seconds):
            raise LauncherFailure('guard_epoch_one', 'epoch one exceeds probe acceptance' if self.tier_limits
                                  else 'epoch one exceeds 2.6 h operational ceiling')

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
        if self.mode == 'probe' and self.expected.get('tier', 'M') != 'M':
            tier_probe.projection(self.probe)
            if any(self.probe.get(k) != self.expected[k] for k in ('tier', 'backbone')):
                raise ValueError('probe tier/backbone mismatch')
        return dict(train_losses=self.losses, test_loss=self.test_loss, banner=self.banner, probe=self.probe)


def run_child(argv, log_path, gpu, guard, deadline=None):
    child = sink = None
    with log_path.open('xb') as log:
        guard.log_created = True
        try:
            child = subprocess.Popen(argv, cwd=REPO, env=child_environment(gpu),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            if getattr(guard, 'execution_path', None):
                guard.execution.update(child_pgid=child.pid, child_exit_status=None)
                p.write_completion(guard.execution_path, guard.execution)
                guard.execution['spawn_execution_sha256'] = p.sha256_file(guard.execution_path)
                print('EXP04_SPAWN ' + json.dumps(guard.execution, sort_keys=True), flush=True)
            sink = subprocess.Popen(['tee', '-a', str(log_path)], stdin=child.stdout, stdout=2,
                                    start_new_session=True)
            child.stdout.close()
            while child.poll() is None or sink.poll() is None:
                guard.poll(log_path)
                if deadline is not None and time.monotonic() > deadline:
                    raise LauncherFailure('deadline_tier' if guard.tier_limits else 'deadline_43h',
                                          'tier cumulative ceiling reached' if guard.tier_limits else '43 h cumulative ceiling reached')
                if sink.poll() not in (None, 0):
                    raise RuntimeError('log sink failed')
                time.sleep(0.05)
            log.flush()
            os.fsync(log.fileno())
            guard.poll(log_path)
            if sink.returncode:
                raise RuntimeError('log sink failed')
            if getattr(guard, 'execution_path', None):
                guard.execution.update(child_exit_status=child.returncode,
                    ended_at=datetime.datetime.now().astimezone().isoformat(), log_sha256=p.sha256_file(log_path))
                p.write_completion(guard.execution_path, guard.execution)
            return child.returncode
        finally:
            p.reap_process_groups(child, sink)


def validate_outputs(attempt, mode, expected, limits=None):
    if mode != 'full':
        if {f.name for f in attempt.iterdir()} - {'execution.json', 'abort.json', 'train_inventory.json'} != {'effective_args.json', 'train_manifest.json'}:
            raise ValueError('no-save run wrote unexpected outputs')
        return {}
    check_runtime(json.loads((attempt / 'args.json').read_text()), expected, mode)
    history = [json.loads(line) for line in (attempt / 'history.jsonl').read_text().splitlines()]
    if [r['epoch'] for r in history] != list(range(1, 13)) or any(
            not math.isfinite(r[key]) for r in history for key in ('train_loss', 'test_loss', 'epoch_minutes')):
        raise ValueError('incomplete or non-finite history')
    if history[0]['epoch_minutes'] > (limits['epoch_seconds'] / 60 if limits else 2.431 * 60):
        raise LauncherFailure('guard_epoch_one', 'epoch one exceeds probe acceptance' if limits
                              else 'epoch one exceeds 2.431 h acceptance')
    epochs = sorted(attempt.glob('epoch_*.pth'))
    if [f.name for f in epochs] != ['epoch_%03d.pth' % i for i in range(1, 13)]:
        raise ValueError('missing or unexpected epoch checkpoints')
    return directory_listing(attempt)


def directory_listing(attempt):
    listing = {}
    for path in sorted(attempt.rglob('*')):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('unexpected attempt entry: ' + str(path))
        listing[str(path.relative_to(attempt))] = p.sha256_file(path) if path.is_file() else None
    return listing


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


def diagnostic_value(value):
    if isinstance(value, dict):
        return {k: diagnostic_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_value(v) for v in value]
    if isinstance(value, Path) or isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def assert_quiescent(pgid, log_path):
    """Require a gone group and no fuser-visible writers (ordinary user permissions).

    The execution receipt separately requires successful child and tee completion;
    fuser, like /proc inspection, cannot inspect inaccessible users' descriptors.
    """
    if type(pgid) is not int or pgid <= 0:
        raise ValueError('missing or invalid recorded process group')
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('child process group is still present')
    log_path.stat()
    result = subprocess.run(['fuser', '-I', '-v', str(log_path.resolve())],
        capture_output=True, text=True, env=dict(os.environ, LC_ALL='C'))
    access = re.findall(r'^\s+\S+\s+([.cefrmF]{5})\s+\S+', result.stderr, re.M)
    pids = result.stdout.split()
    if (result.returncode not in (0, 1) or
            result.returncode == 0 and (not access or len(access) != len(pids) or
                                       not all(pid.isdigit() for pid in pids)) or
            result.returncode == 1 and (result.stdout.strip() or result.stderr.strip())):
        raise ValueError('cannot inspect log writers with fuser')
    if any('F' in flags for flags in access):
        raise ValueError('log writer still open: ' + result.stdout.strip())


def complete_attempt(attempt, mode, log_path, fields, digest, metrics, hours, transaction=None):
    """Defer stop signals through validation and commit, including the caller handoff."""
    with p.deferred_termination():
        result = _complete_attempt(attempt, mode, log_path, fields, digest, metrics, hours)
        if transaction is not None:
            transaction['committed'] = True
        return result


def _complete_attempt(attempt, mode, log_path, fields, digest, metrics, hours):
    limits = tier_gates.timing_limits(fields, fields['effective_args']['CUDA_VISIBLE_DEVICES']) if mode == 'full' else None
    outputs = validate_outputs(attempt, mode, fields['effective_args'], limits)
    fields['mutable_inputs']['train_manifest'] = {
        'path': str((attempt / 'train_manifest.json').resolve()), 'sha256': digest}
    required = {'effective_args', 'train_manifest'} | ({'control_args', 'probe_receipt'} if mode == 'full' else set())
    missing = required - fields['mutable_inputs'].keys()
    if missing:
        raise LauncherFailure('input_changed', 'missing mutable inputs: ' + ', '.join(sorted(missing)))
    print('Revalidating training inputs after log close...', flush=True)
    drift = []
    mismatches = p.revalidate(fields, required=REQUIRED_INPUTS, source_drift=drift)
    if mismatches:
        raise LauncherFailure('input_changed', 'mutable input mismatch: ' + ', '.join(mismatches))
    completion = dict(train_manifest_sha256=digest, source_drift_after_spawn=drift,
        log={'path': str(log_path.resolve()), 'sha256': p.sha256_file(log_path)},
        outputs=outputs, directory_listing=outputs if mode == 'full' else directory_listing(attempt),
        resource_before=fields.get('resource_before'),
        metrics=metrics, wall_hours=hours() if callable(hours) else hours)
    preserved = attempt
    preserved_log = log_path
    recovered_log = Path(fields.get('log_path', str(log_path)))
    recovered = Path(fields.get('attempt_path', str(attempt)))
    if recovered != attempt:
        if recovered.parent != attempt.parent or not attempt.name.startswith(recovered.name + '_ABORTED_'):
            raise ValueError('invalid recovered attempt path')
        if os.path.lexists(recovered):
            raise FileExistsError('recovery destination already exists: ' + str(recovered))
        if recovered_log != log_path and os.path.lexists(recovered_log):
            raise FileExistsError('recovery log destination already exists: ' + str(recovered_log))
        completion['recovered_from'] = dict(attempt=attempt.name, log=str(log_path),
            abort=json.loads((attempt / 'abort.json').read_text()),
            recovered_at=datetime.datetime.now().astimezone().isoformat())
        completion['log']['path'] = str(recovered_log)
    final = attempt.parent / 'final'
    previous_final = os.readlink(final) if final.is_symlink() else None
    try:
        if recovered != attempt:
            attempt.rename(recovered)
            attempt = recovered
            if recovered_log != log_path:
                log_path.rename(recovered_log)
                log_path = recovered_log
        p.write_completion(attempt / 'completion.json', completion)
        account_hours(attempt, completion['wall_hours'], previous_name=preserved.name, mode=mode)
        if mode == 'full':
            promote(attempt)
        return completion
    except BaseException:
        if log_path != preserved_log:
            log_path.rename(preserved_log)
        if attempt != preserved:
            if final.is_symlink() and os.readlink(final) == attempt.name:
                if previous_final is None:
                    final.unlink()
                else:
                    temporary = final.with_name('.final_' + uuid.uuid4().hex)
                    temporary.symlink_to(previous_final)
                    os.replace(temporary, final)
            attempt.rename(preserved)
            account_hours(preserved, completion['wall_hours'], previous_name=attempt.name, mode=mode)
        raise


def recovery_evidence(fields, execution, attempt, launcher_log):
    if fields.get('repo') != str(REPO) or fields.get('mode') not in ('smoke', 'probe', 'full'):
        raise ValueError('invalid recovery repo or mode')
    if not re.fullmatch('[0-9a-f]{40}', fields.get('reviewed_commit', '')):
        raise ValueError('invalid reviewed_commit')
    closures = fields.get('source_closures', {})
    if (not TRAIN_MINIMUM <= {r['path'] for r in closures.get('training', {}).get('files', [])}
            or not closures.get('launcher', {}).get('files')):
        raise ValueError('recovery closure minimum missing')
    if not fields.get('resource_before') or not set(ENV_KEYS) <= set(fields.get('env', {})):
        raise ValueError('recovery resource_before or env missing')
    mode = fields['mode']
    if mode in ('smoke', 'full'):
        save_dir = fields['effective_args']['save_dir']
        original = str(Path(save_dir).parent) if mode == 'smoke' else save_dir
        if (REPO / original).resolve() != Path(fields['attempt_path']):
            raise ValueError('recovery command attempt path mismatch')
        check_golden(fields['command'], mode, original)
    required = {'effective_args', 'train_inventory'} | ({'control_args', 'probe_receipt'} if mode == 'full' else set())
    if not required <= set(fields.get('mutable_inputs', {})):
        raise ValueError('recovery required bindings missing')
    if mode == 'full' and fields['train_data_identity'].get('inventory_files') != 296334:
        raise ValueError('recovery training inventory count')
    if launcher_log is None:
        raise ValueError('recovery requires --launcher-log from nohup setsid invocation')
    path = Path(launcher_log).resolve()
    spawn = dict(train_manifest_sha256=execution['train_manifest_sha256'],
                 child_pgid=execution['child_pgid'], child_exit_status=None)
    spawn['spawn_execution_sha256'] = p.hashlib.sha256((
        json.dumps(spawn, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()).hexdigest()
    if execution.get('spawn_execution_sha256') != spawn['spawn_execution_sha256']:
        raise ValueError('spawn execution digest mismatch')
    line = 'EXP04_SPAWN ' + json.dumps(spawn, sort_keys=True)
    if path == attempt or attempt in path.parents or path.read_text().splitlines().count(line) != 1:
        raise ValueError('execution does not match external launcher log')


@termination_handlers()
def finalize_attempt(attempt, launcher_log=None):
    if Path(sys.executable).resolve() != Path(PYTHON).resolve():
        raise ValueError('launcher requires pinned interpreter ' + PYTHON)
    attempt = Path(attempt).resolve()
    with (attempt.parent / '.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (attempt / 'completion.json').exists():
            raise FileExistsError('attempt already has completion')
        execution = json.loads((attempt / 'execution.json').read_text())
        manifest_path = attempt / 'train_manifest.json'
        digest = execution['train_manifest_sha256']
        if p.sha256_file(manifest_path) != digest:
            raise ValueError('mutable input mismatch: train_manifest')
        fields = json.loads(manifest_path.read_text())
        recovery_evidence(fields, execution, attempt, launcher_log)
        log_path = Path(fields['log_path'])
        if (attempt / 'abort.json').exists():
            aborted_log = json.loads((attempt / 'abort.json').read_text())['log']['aborted']
            if aborted_log:
                log_path = Path(aborted_log)
        assert_quiescent(execution.get('child_pgid'), log_path)
        if type(execution.get('child_exit_status')) is not int or execution['child_exit_status'] != 0:
            raise ValueError('recovery requires recorded successful child exit')
        if p.sha256_file(log_path) != execution['log_sha256']:
            raise ValueError('closed log changed')
        original = Path(fields['attempt_path'])
        effective = fields['mutable_inputs']['effective_args']
        if Path(effective['path']) != original / 'effective_args.json':
            raise ValueError('unexpected effective_args path')
        effective['path'] = str(attempt / 'effective_args.json')
        identity = fields['train_data_identity']
        if 'inventory_file' in identity:
            for record in (identity['inventory_file'], fields['mutable_inputs']['train_inventory']):
                if Path(record['path']) != original / 'train_inventory.json':
                    raise ValueError('unexpected train_inventory path')
                record['path'] = str(attempt / 'train_inventory.json')
        limits = tier_gates.timing_limits(fields, fields['effective_args']['CUDA_VISIBLE_DEVICES']) if fields['mode'] == 'full' else None
        guard = LogGuard(fields['effective_args'], fields['mode'], attempt, limits)
        guard.poll(log_path)
        hours = (datetime.datetime.fromisoformat(execution['ended_at']) -
                 datetime.datetime.fromisoformat(fields['started_at'])).total_seconds() / 3600
        if not math.isfinite(hours) or hours < 0:
            raise ValueError('invalid recorded execution duration')
        hours = max([hours] + [row['hours'] for row in hours_record(attempt.parent)['attempts']
                              if row['attempt'] == attempt.name])
        transaction = {}
        try:
            return complete_attempt(attempt, fields['mode'], log_path, fields, digest, guard.finish(), hours, transaction)
        except BaseException:
            if transaction.get('committed'):
                print('CERTIFIED ' + str(original), flush=True)
                raise
            if (attempt / 'completion.json').exists():
                (attempt / 'completion.json').unlink()
            raise


@termination_handlers()
def execute_attempt(attempt, mode, gpu, log_path, fields_factory, allow_cotenant=False,
                    projection=30.0, runner=run_child, limits=None, renew_ceiling=None):
    attempt, log_path = Path(attempt), Path(log_path)
    if mode == 'full':
        check_budget(attempt.parent, projection, limits['ceiling_hours'] if limits else 43)
    create_attempt(attempt.parent, attempt.name)  # An existing directory is never renamed.
    started, reason = time.monotonic(), 'setup_failed'
    started_at, guard = datetime.datetime.now().astimezone().isoformat(), None
    transaction = {}
    try:
        if attempt == log_path or attempt in log_path.parents:
            raise ValueError('log must be outside attempt')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        before = resource_gate(gpu, attempt.parent, mode, allow_cotenant)
        fields = fields_factory()
        expected = fields['effective_args']
        fields['resource_before'] = before
        validated = tier_gates.timing_limits(fields, gpu) if mode == 'full' else None
        if validated:
            limits = tier_gates.set_budget(attempt.parent, validated, renewal=renew_ceiling)
            fields['timing_limits'] = limits
        elif limits:
            raise ValueError('tier timing limits require an S/L full run')
        effective_path = attempt / 'effective_args.json'
        effective_digest = p.write_manifest(effective_path, expected)
        fields.setdefault('mutable_inputs', {})['effective_args'] = {
            'path': str(effective_path.resolve()), 'sha256': effective_digest}
        identity = fields.get('train_data_identity', {})
        if 'inventory' in identity:
            inventory_path = attempt / 'train_inventory.json'
            inventory_digest = p.write_manifest(inventory_path, {'inventory': identity.pop('inventory')})
            identity['inventory_file'] = {'path': str(inventory_path.resolve()), 'sha256': inventory_digest}
            fields['mutable_inputs']['train_inventory'] = dict(identity['inventory_file'])
        fields.update(resource_before=before, allow_cotenant=allow_cotenant, mode=mode,
                      started_at=started_at, attempt_path=str(attempt.resolve()), log_path=str(log_path.resolve()))
        manifest_path = attempt / 'train_manifest.json'
        digest = p.write_manifest(manifest_path, fields)
        if {f.name for f in attempt.iterdir()} - {'train_inventory.json'} != {'effective_args.json', 'train_manifest.json'}:
            raise ValueError('unexpected pre-spawn files')
        guard = LogGuard(expected, mode, limits=limits)
        guard.execution_path = attempt / 'execution.json'
        guard.execution = {'train_manifest_sha256': digest}
        reason = 'child_failed'
        ceiling = limits['ceiling_hours'] if limits else 43
        deadline = started + (ceiling - full_hours(attempt.parent)) * 3600 if mode == 'full' else None
        status = runner(fields['command'], log_path, gpu, guard, deadline)
        if status:
            raise LauncherFailure('child_exit_' + str(status), 'child exited nonzero: ' + str(status))
        metrics = guard.finish()
        reason = 'output_invalid'
        return complete_attempt(attempt, mode, log_path, fields, digest, metrics,
                                lambda: (time.monotonic() - started) / 3600, transaction)
    except BaseException as error:
        if transaction.get('committed'):
            print('CERTIFIED ' + str(attempt), flush=True)
            raise
        reason = p.masked_abort_reason(error, reason)
        hours = (time.monotonic() - started) / 3600
        suffix = 'slow' if reason == 'guard_epoch_one' and limits else reason
        renamed = abort_log(log_path, suffix, guard is not None and guard.log_created)
        p.write_completion(attempt / 'abort.json', dict(reason=reason,
            exception_type=type(error).__name__, exception_message=str(error), started_at=started_at,
            aborted_at=datetime.datetime.now().astimezone().isoformat(), wall_hours=hours,
            last_guard_state=diagnostic_value({k: v for k, v in vars(guard).items() if k != 'expected'}) if guard else None,
            log={'original': str(log_path.resolve()), 'aborted': renamed}))
        completion_path = attempt / 'completion.json'
        if completion_path.exists():
            completion_path.unlink()
        aborted = abort_attempt(attempt, suffix, hours, mode=mode)
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
    parser.add_argument('mode', choices=('smoke', 'probe', 'full', 'finalize', 'refuse-test'))
    parser.add_argument('attempt_dir', nargs='?')
    parser.add_argument('--gpu', default='1', choices=('0', '1'))
    parser.add_argument('--tier', choices=tuple(TIERS), default='M')
    parser.add_argument('--backbone', choices=('simple', 'cylindrical'), default='simple')
    parser.add_argument('--reviewed-commit')
    parser.add_argument('--log-dir')
    parser.add_argument('--launcher-log', help='finalize: external nohup setsid launcher stdout log')
    parser.add_argument('--timestamp', default=datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f'))
    parser.add_argument('--allow-cotenant', action='store_true')
    parser.add_argument('--allow-dirty', action='store_true')
    parser.add_argument('--projection-hours', type=float, default=30.0)
    parser.add_argument('--probe-json', help='full requires a clean passing probe receipt')
    parser.add_argument('--renew-ceiling', help='S/L full only: notebook timestamp: reason')
    args = parser.parse_args(argv)
    tiered = args.tier != 'M'
    if args.renew_ceiling is not None and (not tiered or args.mode != 'full'):
        parser.error('--renew-ceiling requires full --tier S|L')
    if tiered and not args.log_dir:
        args.log_dir = str(REPO / TIER_RECORD)
    if args.mode == 'finalize':
        if not args.attempt_dir:
            parser.error('finalize requires an attempt directory')
        result = finalize_attempt(args.attempt_dir, args.launcher_log)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return result
    if args.attempt_dir:
        parser.error('attempt directory is only valid for finalize')
    if args.mode == 'refuse-test':
        result = refusal_self_test()
        print(json.dumps(result), flush=True)
        return result
    if not args.reviewed_commit or not args.log_dir:
        parser.error('--reviewed-commit and --log-dir are required')
    if args.tier != 'M' and args.mode == 'smoke':
        parser.error('exp05 requires the fit-probe protocol before full')
    if args.tier != 'M' and args.mode == 'full' and not args.probe_json:
        parser.error('full requires --probe-json (clean passing fit-probe)')
    if args.tier == 'M' and args.backbone != 'simple':
        parser.error('exp04 M launches require backbone simple')
    root = arm_root(args.tier, args.backbone)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.timestamp) or '_ABORTED_' in args.timestamp:
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
        receipt = (probe_receipt(args.probe_json, commit, args.gpu) if args.tier == 'M' else
                   tier_gates.validate_receipt(args.probe_json, commit, args.gpu, args.tier, args.backbone))
        if args.tier == 'M':
            check_budget(root, args.projection_hours)
    stamp, mode = args.timestamp, args.mode
    if mode in ('smoke', 'full'):
        attempt = root / (('_smoke_' if mode == 'smoke' else 'attempt_') + stamp)
        relative = os.path.relpath(attempt, REPO)
        cmd = command(mode, relative, args.tier, args.backbone)
        check_golden(cmd, mode, relative)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.launch.lock').open('a') as lock, patch.dict(os.environ, child_environment(args.gpu)):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        log_dir = Path(args.log_dir).resolve()
        train_log = ('param_efficiency_{}_train_{}_{}_{}.log'.format(stamp, args.tier, args.backbone, mode)
                     if tiered else 'yaw_aug_xrir_' + stamp + '_train_' + mode + '.log')
        if mode in ('smoke', 'full'):
            limits = None
            if mode == 'full' and args.tier != 'M':
                limits = tier_gates.timing_limits(dict(reviewed_commit=commit,
                    effective_args=effective_args(cmd, args.gpu, 9261), mutable_inputs=dict(probe_receipt=receipt)), args.gpu)
                args.projection_hours = limits['projection_hours']
            def fields_factory():
                fields = build_fields(cmd, args.gpu, commit, mode, args.allow_dirty)
                if mode == 'full':
                    fields['mutable_inputs']['probe_receipt'] = receipt
                return fields
            result = execute_attempt(attempt, mode, args.gpu,
                log_dir / train_log,
                fields_factory,
                allow_cotenant=args.allow_cotenant, projection=args.projection_hours,
                **({'limits': limits, 'renew_ceiling': args.renew_ceiling} if limits else {}))
        else:
            output = root / ('_probe_' + stamp + ('_' + args.tier + '_' + args.backbone if tiered else '') + '.json')
            if output.exists():
                raise FileExistsError(str(output))
            before = gpu_snapshot(args.gpu)
            measurements, attempts, arms_before = [], [], []
            for yaw, label in (((0, 'arm'),) if tiered else ((0, 'off'), (1, 'on'))):
                attempt = root / ('_probe_' + stamp + '_' + label)
                relative = os.path.relpath(attempt, REPO)
                cmd = [PYTHON, '-m', 'tools.exp04_probe', '--yaw-aug', str(yaw), '--save-dir', relative]
                trainer_argv = trainer_command(yaw, relative)
                if tiered:
                    cmd = [PYTHON, '-m', 'tools.exp05_probe', '--tier', args.tier,
                           '--backbone', args.backbone, '--save-dir', relative]
                    trainer_argv = tier_probe.trainer_command(args.tier, args.backbone, relative)
                def fields_factory():
                    fields = build_fields([PYTHON] + trainer_argv, args.gpu, commit, 'probe', args.allow_dirty)
                    fields['trainer_command'], fields['command'] = fields['command'], cmd
                    return fields
                completed = execute_attempt(attempt, 'probe', args.gpu,
                    log_dir / (train_log if tiered else 'yaw_aug_xrir_' + stamp + '_' + label + '_train_probe.log'),
                    fields_factory, allow_cotenant=args.allow_cotenant)
                measurements.append(completed['metrics']['probe'])
                arms_before.append(completed['resource_before'])
                attempts.append(str(attempt))
            after = gpu_snapshot(args.gpu)
            measured = (dict(measurements[0], schema_version=1, gpu=args.gpu) if tiered else
                        dict(compare_results(*measurements), yaw_off=measurements[0], yaw_on=measurements[1]))
            result = dict(measured,
                before=before, after=after, arms_before=arms_before,
                PROBE_NOT_CLEAN=any(bool(state['compute_apps']) for state in [before, *arms_before, after]),
                attempts=attempts, reviewed_commit=commit)
            if tiered:
                result.update(probe_attempt=dict(path=str(attempt.resolve()), **{
                    name + '_sha256': p.sha256_file(attempt / (name + '.json'))
                    for name in ('train_manifest', 'completion')}),
                    live_epoch_limit_seconds=1.05 * result['T_epoch'], live_epoch_limit_start='banner')
                result['PROBE_NOT_CLEAN'] |= any(state['gpu'] != args.gpu or
                    not before['uuid'] or state['uuid'] != before['uuid'] or
                    not math.isfinite(state['free_gib']) or state['free_gib'] < 40
                    for state in [before, *arms_before, after])
                if result['PROBE_NOT_CLEAN']:
                    result['passed'] = False
            p.write_manifest(output, result)
            if result['PROBE_NOT_CLEAN']:
                print('PROBE_NOT_CLEAN: GPU resource checks failed; re-probe on a clean GPU.' if tiered else
                      'PROBE_NOT_CLEAN: another process was present; timing gate needs Planner judgment.', flush=True)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        if mode == 'probe' and not result['passed']:
            raise SystemExit(1)
        return result


if __name__ == '__main__':
    main()
