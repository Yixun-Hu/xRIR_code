"""Fail-closed seen-protocol training launcher for exp_07 (plan section 2, amendment A1).

``tools/exp04_launcher.py`` keeps main's bytes.  Everything operational is reused from it
by composition -- the ledgers, the resource gate, ``run_child``, ``execute_attempt`` /
``complete_attempt`` / ``finalize_attempt``, ``normalize``, ``validate_parameters`` --
and only the arm-specific decisions are stated here: the three seen arms, their goldens
(which invoke ``tools/exp07_train.py``), the per-arm unseen-twin comparator, the seen
batch count, the ``seen_split`` binding, the seen banner/probe lines and the one-retry
ceiling of ``tools/exp07_gates.py``.

The shared orchestration looks its collaborators up as module globals of
``exp04_launcher``, so :func:`seen_overrides` rebinds exactly those names for the
duration of an exp_07 launch and restores them afterwards; ``exp04_launcher``'s own file
and its behaviour outside that context are untouched.

    tools/exp07_launch.sh smoke --backbone simple --gpu 1 --reviewed-commit <sha>
    tools/exp07_launch.sh probe --backbone simple --gpu 1 --reviewed-commit <sha>
    nohup setsid tools/exp07_launch.sh full --backbone simple --gpu 1 \
        --reviewed-commit <sha> --probe-json <receipt> > <launcher.log> 2>&1 &
    tools/exp07_launch.sh finalize <attempt> --launcher-log <launcher.log>
"""
import json
import math
import os
import re
import shlex
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from tools import exp04_launcher as base
from tools import exp07_gates as gates
from tools import exp07_probe as probe
from tools import exp07_provenance as e7p
from tools import exp07_train
from tools import provenance as p
from tools.exp05_params import TIERS, tier_of

PYTHON = base.PYTHON
REPO = base.REPO
DATA_ROOT = base.DATA_ROOT
ENV_KEYS = base.ENV_KEYS
PROTOCOL = 'seen'
ENTRY = 'tools/exp07_train.py'
ENTRY_MODULE = 'tools.exp07_train'
RECORD = 'worklog/worklog_yixun/exp_07_seen_protocol_claude'
ASSETS = 'seen_protocol_results_assets'
SEEN_MODULE = exp07_train.SEEN_MODULE
SEEN_INPUTS = {'seen_split'}  # mutable inputs every seen run must bind and revalidate
ARMS = {('simple', 0): 'seen_simple', ('cylindrical', 0): 'seen_cyl', ('simple', 1): 'seen_aug'}
CONTROLS = {'seen_simple': 'ckpt/xRIR_simple_8_shot/args.json',
            'seen_cyl': 'ckpt/xRIR_cyl_8_shot/args.json',
            'seen_aug': 'ckpt/xRIR_simple_yawaug_8_shot/final/args.json'}
TRAIN_FILES = {'unseen': 296334, 'seen': 296454}
TRAIN_BATCHES = {'unseen': 9261, 'seen': 9265}
YAW_FLAGS = '--yaw-aug 1 --yaw-aug-seed 0 --yaw-aug-width 512'
CONTROL_EXCLUSIONS = {'save_dir', 'save_every', 'epoch_ckpt_every',
                      'PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES'}


def flag_value(argv, flag, default=None):
    index = argv.index(flag) if flag in argv else -1
    return argv[index + 1] if 0 <= index < len(argv) - 1 else default


def arm(backbone, yaw_aug=0):
    """Name the arm of a backbone/yaw pair; the flag may be an int or its argv string."""
    key = (backbone, {None: 0, 0: 0, 1: 1, '0': 0, '1': 1}.get(yaw_aug, -1))
    if key not in ARMS:
        raise ValueError('no seen arm for backbone/yaw: {}/{}'.format(backbone, yaw_aug))
    return ARMS[key]


def arm_root(backbone, yaw_aug=0):
    return REPO / 'ckpt/exp07' / arm(backbone, yaw_aug)


def goldens():
    return REPO / RECORD / ASSETS


def command(mode, attempt, backbone='simple', yaw_aug=0):
    """The plan's recipe for one arm, invoking the exp_07 entry point."""
    if mode not in ('smoke', 'full'):
        raise ValueError('command requires smoke or full')
    arm(backbone, yaw_aug)  # an unregistered backbone/yaw pair has no seen arm
    smoke = mode == 'smoke'
    result = [PYTHON, ENTRY, '--backbone', backbone, '--save-dir',
              str(attempt) + ('/smoke' if smoke else '')]
    result += shlex.split('--num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4 '
                          '--decay-epochs 3 --lr-gamma 0.1 --epochs ' + ('1' if smoke else '12'))
    if smoke:
        result += shlex.split('--max-train-batches 3 --max-test-batches 2')
    result += shlex.split('--batch-size 4 --accum-steps 2 --num-workers 4' if smoke else
                          '--batch-size 32 --accum-steps 2 --num-workers 12')
    result += shlex.split('--seed 0 --tf32 --log-interval ' + ('1' if smoke else '50') +
                          ' --save-every 0 --epoch-ckpt-every ' + ('0 --no-save' if smoke else '1'))
    result += ['--protocol', PROTOCOL]
    return result + (shlex.split(YAW_FLAGS) if arm(backbone, yaw_aug) == 'seen_aug' else [])


def check_golden(argv, mode, attempt):
    """Every argv the launcher spawns equals its arm's golden file, path aside."""
    if mode not in ('smoke', 'full'):
        raise ValueError('argv differs from golden: unknown mode ' + str(mode))
    try:
        name = arm(flag_value(argv, '--backbone'), flag_value(argv, '--yaw-aug', 0))
    except ValueError as error:
        raise ValueError('argv differs from golden seen arm') from error
    if list(argv[:2]) != [PYTHON, ENTRY] or flag_value(argv, '--protocol') != PROTOCOL:
        raise ValueError('argv differs from golden seen arm')
    placeholder = 'ckpt/exp07/{}/{}<ts>'.format(name, 'attempt_' if mode == 'full' else '_smoke_')
    prefix = placeholder[:-len('<ts>')]
    if (not str(attempt).startswith(prefix)
            or not re.fullmatch(r'[A-Za-z0-9_-]+', str(attempt)[len(prefix):])):
        raise ValueError('argv differs from golden seen arm')
    path = goldens() / ('argv_golden_{}{}.txt'.format(name, '' if mode == 'full' else '_smoke'))
    if not path.is_file():
        raise ValueError('golden file missing: ' + path.name)
    expected = [part.replace(placeholder, str(attempt)) for part in shlex.split(path.read_text())]
    if list(argv) != expected:
        raise ValueError('argv differs from golden ' + mode)


def child_environment(gpu):
    return dict(os.environ, XRIR_DATA_PATH=DATA_ROOT, PYTHONHASHSEED='0',
                OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES=str(gpu),
                PYTHONPATH=str(REPO), PYTHONUNBUFFERED='1')


def effective_args(argv, gpu, batches_per_epoch):
    """The recorded argument schema of an exp_07 run: the trainer's plus ``protocol``."""
    result = vars(exp07_train.parse_args(list(argv[2:])))
    result.update({key: child_environment(gpu)[key] for key in ENV_KEYS})
    result['train_batches_per_epoch'] = batches_per_epoch
    result.update(tier=tier_of(result),
                  param_counts=dict(base.tier_counts(result['backbone'], tier_of(result))))
    return result


def check_runtime(runtime, expected, mode):
    """The shared schema comparison, plus the protocol and its loader length."""
    actual = base.normalize(runtime)
    if 'tier' in actual or 'param_counts' in actual:
        base.validate_parameters(actual)
    differences = [key for key in actual.keys() | expected.keys() if key not in actual or
                   key not in expected or type(actual[key]) is not type(expected[key]) or
                   actual[key] != expected[key]]
    if differences:
        raise base.LauncherFailure('guard_runtime_args',
                                   'runtime args mismatch: ' + ', '.join(sorted(differences)))
    if actual.get('protocol') != PROTOCOL:
        raise ValueError('exp_07 runs require protocol ' + PROTOCOL)
    if mode == 'full' and actual['train_batches_per_epoch'] != TRAIN_BATCHES[PROTOCOL]:
        raise ValueError('full requires train_batches_per_epoch == ' + str(TRAIN_BATCHES[PROTOCOL]))


def compare_control(runtime, control, control_env):
    """A seen arm differs from its unseen twin in the protocol and nothing else."""
    treatment, baseline = base.normalize(runtime), base.normalize(dict(control, env=control_env))
    for values in (treatment, baseline):
        for key, value in TIERS['M'].items():
            values.setdefault('vit_' + key, value)
        base.validate_parameters(values)
        for key in ('tier', 'param_counts'):
            values.pop(key, None)
        values.setdefault('protocol', 'unseen')  # the historical comparators predate exp_07
        if values['protocol'] not in TRAIN_FILES:
            raise ValueError('unknown protocol: ' + str(values['protocol']))
        files = TRAIN_FILES[values['protocol']]
        bpe = values.pop('train_batches_per_epoch', math.ceil(files / values['batch_size']))
        if type(bpe) is not int or bpe != math.ceil(files / values['batch_size']):
            raise ValueError('invalid train_batches_per_epoch')
        for key, value in dict(yaw_aug=0, yaw_aug_seed=values['seed'],
                               yaw_aug_width=512, no_save=False).items():
            values.setdefault(key, value)
        if values['yaw_aug_seed'] is None:  # exp_05 A1: the historical default is the seed
            values['yaw_aug_seed'] = values['seed']
    if tier_of(baseline) != 'M' or tier_of(treatment) != 'M':
        raise ValueError('the seen arms and their unseen twins are tier M')
    if treatment['protocol'] != PROTOCOL:
        raise ValueError('control comparison requires a seen treatment')
    missing = object()
    differences = {key: {'treatment': treatment.get(key), 'control': baseline.get(key)}
        for key in treatment.keys() | baseline.keys()
        if type(treatment.get(key, missing)) is not type(baseline.get(key, missing))
        or treatment.get(key, missing) != baseline.get(key, missing)}
    if differences.get('protocol') != dict(treatment=PROTOCOL, control='unseen'):
        raise ValueError('control mismatch: a seen arm requires its unseen-protocol twin')
    refused = differences.keys() - (CONTROL_EXCLUSIONS | {'protocol'})
    if refused:
        raise ValueError('control mismatch: ' + ', '.join(sorted(refused)))
    return differences


def train_minimum():
    return base.TRAIN_MINIMUM | {SEEN_MODULE, ENTRY}


def build_fields(argv, gpu, reviewed_commit, mode, allow_dirty=False):
    """The pre-spawn record: closures, the seen inventory, the split and the twin diff."""
    state = p.checked_git_state(REPO, mode == 'full', allow_dirty)
    files = p.source_closure(ENTRY_MODULE, REPO)
    if not train_minimum() <= set(files):
        raise ValueError('training closure missing required files')
    closures = {}
    for role, paths in [('training', files), ('launcher',
            p.source_closure('tools.exp07_launcher', REPO) +
            ['tools/exp07_probe.py', 'tools/exp07_launch.sh'])]:
        records, digest = p.closure_record(paths, reviewed_commit, REPO)
        if not records or any(r['reviewed_blob_sha256'] is None or
                r['reviewed_blob_sha256'] != r['working_tree_sha256'] or
                r['commits_after_reviewed'] for r in records):
            raise ValueError(role + ' closure differs from reviewed commit')
        closures[role] = {'files': records, 'sha256': digest}
    # The split file selects the training inventory: capture its identity BEFORE the
    # inventory is built, re-verify it afterwards and bind the one it was selected under.
    split = e7p.seen_split_identity(REPO)
    print('Hashing training data identity...', flush=True)
    data = e7p.train_data_identity(DATA_ROOT, protocol=PROTOCOL,
                                   cache_path=REPO / e7p.SEEN_CACHE)
    if e7p.seen_split_identity(REPO) != split:
        raise ValueError('seen split changed during training inventory construction')
    provisional = effective_args(argv, gpu, 1)
    bpe = math.ceil(data['inventory_files'] / provisional['batch_size'])
    effective = effective_args(argv, gpu, bpe)
    check_runtime(effective, effective, mode)
    fields = dict(repo=str(REPO), reviewed_commit=reviewed_commit, mode=mode, protocol=PROTOCOL,
        source_closures=closures, train_data_identity=data, effective_args=effective,
        command=list(argv), environment=p.environment(), git_state=state, allow_dirty=allow_dirty,
        env={key: child_environment(gpu)[key] for key in ENV_KEYS},
        mutable_inputs={'seen_split': split})
    if mode == 'full':
        control_path = REPO / CONTROLS[arm(effective['backbone'], effective['yaw_aug'])]
        control = json.loads(control_path.read_text())
        control_env = control.get('env') or dict(XRIR_DATA_PATH=DATA_ROOT, OMP_NUM_THREADS='2',
                                                 CUDA_VISIBLE_DEVICES='1')
        fields['control_excluded_differences'] = compare_control(effective, control, control_env)
        fields['control_env_reconstructed'] = control_env
        fields['mutable_inputs']['control_args'] = {'path': str(control_path),
                                                    'sha256': p.sha256_file(control_path)}
    return fields


class LogGuard(base.LogGuard):
    """The shared guard, reading the exp_07 probe line and refusing the other two."""

    def feed(self, line):
        if line.startswith(('EXP04_PROBE_RESULT ', 'EXP05_PROBE_RESULT ')):
            raise ValueError('probe result prefix differs from the seen probe')
        if line.startswith(probe.PREFIX):
            self.probe = json.loads(line[len(probe.PREFIX):])
        super().feed(line)

    def finish(self):
        result = super().finish()
        if self.mode == 'probe':
            probe.projection(self.probe)
            if any(self.probe.get(key) != self.expected.get(key)
                   for key in ('tier', 'backbone', 'protocol', 'yaw_aug')):
                raise ValueError('probe tier/backbone/protocol mismatch')
        return result


_BASE_COMPLETE = base.complete_attempt


def complete_attempt(attempt, mode, log_path, fields, digest, metrics, hours, transaction=None):
    """Require the seen bindings before the shared completion commits anything."""
    missing = SEEN_INPUTS - set(fields.get('mutable_inputs', {}))
    if missing:
        raise base.LauncherFailure('input_changed',
                                   'missing mutable inputs: ' + ', '.join(sorted(missing)))
    return _BASE_COMPLETE(attempt, mode, log_path, fields, digest, metrics, hours, transaction)


def recovery_evidence(fields, execution, attempt, launcher_log):
    """Finalisation evidence for a seen attempt: closures, bindings, split and log."""
    if fields.get('repo') != str(REPO) or fields.get('mode') not in ('smoke', 'probe', 'full'):
        raise ValueError('invalid recovery repo or mode')
    if not re.fullmatch('[0-9a-f]{40}', fields.get('reviewed_commit', '')):
        raise ValueError('invalid reviewed_commit')
    closures = fields.get('source_closures', {})
    if (not train_minimum() <= {r['path'] for r in closures.get('training', {}).get('files', [])}
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
    required = ({'effective_args', 'train_inventory'} | SEEN_INPUTS |
                ({'control_args', 'probe_receipt'} if mode == 'full' else set()))
    if not required <= set(fields.get('mutable_inputs', {})):
        raise ValueError('recovery required bindings missing')
    if fields['mutable_inputs']['seen_split'] != e7p.seen_split_identity(REPO):
        raise ValueError('recovery seen split changed')
    if mode == 'full' and fields['train_data_identity'].get('inventory_files') != TRAIN_FILES[PROTOCOL]:
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


@contextmanager
def seen_overrides():
    """Rebind exp04_launcher's collaborators to exp_07's for the duration of a run.

    The shared orchestration resolves ``check_golden``, ``check_runtime``, ``LogGuard``,
    ``complete_attempt``, ``recovery_evidence``, ``tier_gates``, ``TRAIN_MINIMUM``,
    ``child_environment`` and ``REPO`` as module globals of ``exp04_launcher``.  Binding
    them here is how exp_07 supplies its own decisions without editing that file; outside
    this context ``exp04_launcher`` behaves exactly as main's bytes describe.
    """
    with patch.multiple(base, REPO=REPO, TRAIN_MINIMUM=train_minimum(), tier_gates=gates,
                        child_environment=child_environment, check_golden=check_golden,
                        check_runtime=check_runtime, compare_control=compare_control,
                        complete_attempt=complete_attempt, recovery_evidence=recovery_evidence,
                        LogGuard=LogGuard):
        yield


def refusal_self_test():
    """Exercise the seen refusals in scratch storage, with no trainer spawn."""
    import tempfile
    refusals = []

    def refuses(name, action):
        try:
            action()
        except (ValueError, FileExistsError):
            refusals.append(name)
        else:
            raise AssertionError('refusal not enforced: ' + name)

    with tempfile.TemporaryDirectory(prefix='exp07_refuse_') as temporary:
        root = Path(temporary)
        attempt = base.create_attempt(root, 'attempt_test')
        refuses('exclusive_directory', lambda: base.create_attempt(root, attempt.name))
        seen = command('full', 'ckpt/exp07/seen_aug/attempt_test', 'simple', 1)
        argv = list(seen)
        argv[argv.index('--lr') + 1] = '0.5'
        refuses('golden_argv', lambda: check_golden(argv, 'full', 'ckpt/exp07/seen_aug/attempt_test'))
        refuses('seen_other_arm', lambda: check_golden(seen, 'full', 'ckpt/exp07/seen_simple/attempt_test'))
        refuses('seen_unseen_argv', lambda: check_golden(
            base.command('full', 'ckpt/exp07/seen_aug/attempt_test'), 'full',
            'ckpt/exp07/seen_aug/attempt_test'))
        refuses('seen_arm', lambda: arm('cylindrical', 1))
        expected = effective_args(seen, '1', TRAIN_BATCHES[PROTOCOL])
        refuses('runtime_schema', lambda: check_runtime(dict(expected, lr='0.001'), expected, 'full'))
        wrong = effective_args(seen, '1', TRAIN_BATCHES['unseen'])
        refuses('seen_bpe', lambda: check_runtime(wrong, wrong, 'full'))
        refuses('seen_protocol', lambda: check_runtime(dict(expected, protocol='unseen'),
                                                       dict(expected, protocol='unseen'), 'full'))
        control = json.loads((REPO / CONTROLS['seen_simple']).read_text())
        refuses('seen_control', lambda: compare_control(expected, control,
            dict(XRIR_DATA_PATH=DATA_ROOT, OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='1')))
        refuses('banner', lambda: LogGuard(expected, 'full').feed('Train Epoch: 1 [0/9265] loss 1'))
        refuses('probe_prefix', lambda: LogGuard(expected, 'probe').feed('EXP05_PROBE_RESULT {}'))
        refuses('full_cotenant', lambda: base.resource_gate('1', root, 'full', True))
        base.abort_attempt(attempt, 'test', 1)
        refuses('one_retry', lambda: gates.set_budget(root, dict(protocol=PROTOCOL,
            ceiling_hours=45., projection_hours=1., probe_receipt_sha256='a' * 64), commit=False))
    return {'passed': True, 'refusals': refusals}


@p.termination_handlers()
def main(argv=None):
    import argparse
    import datetime
    import fcntl
    import subprocess
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'probe', 'full', 'finalize', 'refuse-test'))
    parser.add_argument('attempt_dir', nargs='?')
    parser.add_argument('--gpu', default='1', choices=('0', '1'))
    parser.add_argument('--backbone', choices=('simple', 'cylindrical'), default='simple')
    parser.add_argument('--yaw-aug', type=int, choices=(0, 1), default=0,
                        help='1 selects the seen_aug arm (simple backbone only)')
    parser.add_argument('--reviewed-commit')
    parser.add_argument('--log-dir')
    parser.add_argument('--launcher-log', help='finalize: external nohup setsid launcher stdout log')
    parser.add_argument('--timestamp', default=datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f'))
    parser.add_argument('--allow-cotenant', action='store_true')
    parser.add_argument('--allow-dirty', action='store_true')
    parser.add_argument('--probe-json', help='full requires a clean passing seen probe receipt')
    parser.add_argument('--renew-ceiling', help='full only: notebook timestamp: reason')
    args = parser.parse_args(argv)
    if args.renew_ceiling is not None and args.mode != 'full':
        parser.error('--renew-ceiling requires a full run')
    try:
        name = arm(args.backbone, args.yaw_aug)
    except ValueError as error:
        parser.error(str(error))
    args.log_dir = args.log_dir or str(REPO / RECORD)
    with seen_overrides():
        if args.mode == 'finalize':
            if not args.attempt_dir:
                parser.error('finalize requires an attempt directory')
            result = base.finalize_attempt(args.attempt_dir, args.launcher_log)
            print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
            return result
        if args.attempt_dir:
            parser.error('attempt directory is only valid for finalize')
        if args.mode == 'refuse-test':
            result = refusal_self_test()
            print(json.dumps(result), flush=True)
            return result
        if not args.reviewed_commit:
            parser.error('--reviewed-commit is required')
        if args.mode == 'full' and not args.probe_json:
            parser.error('full requires --probe-json (clean passing fit-probe)')
        if args.mode == 'full' and args.allow_cotenant:
            parser.error('full forbids --allow-cotenant')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', args.timestamp) or '_ABORTED_' in args.timestamp:
            parser.error('timestamp must be a safe filename component')
        if Path(sys.executable).resolve() != Path(PYTHON).resolve():
            parser.error('launcher requires pinned interpreter ' + PYTHON)
        root, stamp, mode = arm_root(args.backbone, args.yaw_aug), args.timestamp, args.mode
        commit = subprocess.check_output(['git', 'rev-parse', args.reviewed_commit + '^{commit}'],
                                         cwd=str(REPO), text=True).strip()
        receipt = (gates.validate_receipt(args.probe_json, commit, args.gpu, 'M', args.backbone,
                                          PROTOCOL, args.yaw_aug) if mode == 'full' else None)
        if mode in ('smoke', 'full'):
            attempt = root / (('_smoke_' if mode == 'smoke' else 'attempt_') + stamp)
            relative = os.path.relpath(attempt, REPO)
            cmd = command(mode, relative, args.backbone, args.yaw_aug)
            check_golden(cmd, mode, relative)
        root.mkdir(parents=True, exist_ok=True)
        with (root / '.launch.lock').open('a') as lock, \
                patch.dict(os.environ, child_environment(args.gpu)):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            log_dir = Path(args.log_dir).resolve()
            train_log = 'seen_protocol_{}_train_{}_{}.log'.format(stamp, name, mode)
            if mode in ('smoke', 'full'):
                limits, projection = None, 30.0
                if mode == 'full':
                    limits = gates.timing_limits(dict(reviewed_commit=commit,
                        effective_args=effective_args(cmd, args.gpu, TRAIN_BATCHES[PROTOCOL]),
                        mutable_inputs=dict(probe_receipt=receipt)), args.gpu)
                    projection = limits['projection_hours']

                def fields_factory():
                    fields = build_fields(cmd, args.gpu, commit, mode, args.allow_dirty)
                    if mode == 'full':
                        fields['mutable_inputs']['probe_receipt'] = receipt
                    return fields

                result = base.execute_attempt(attempt, mode, args.gpu, log_dir / train_log,
                    fields_factory, allow_cotenant=args.allow_cotenant, projection=projection,
                    **({'limits': limits, 'renew_ceiling': args.renew_ceiling} if limits else {}))
            else:
                result = run_probe(root, stamp, name, commit, log_dir / train_log, args)
            print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
            if mode == 'probe' and not result['passed']:
                raise SystemExit(1)
            return result


def run_probe(root, stamp, name, commit, log_path, args):
    """One single-arm fit probe; the receipt binds its own attempt and GPU snapshots."""
    output = root / ('_probe_' + stamp + '_' + name + '.json')
    if output.exists():
        raise FileExistsError(str(output))
    before = base.gpu_snapshot(args.gpu)
    attempt = root / ('_probe_' + stamp + '_arm')
    relative = os.path.relpath(attempt, REPO)
    cmd = ([PYTHON, '-m', 'tools.exp07_probe', '--tier', 'M', '--backbone', args.backbone,
            '--protocol', PROTOCOL, '--save-dir', relative] +
           (['--yaw-aug', str(args.yaw_aug)] if args.yaw_aug else []))
    trainer_argv = probe.trainer_command('M', args.backbone, relative, PROTOCOL, args.yaw_aug)

    def fields_factory():
        fields = build_fields([PYTHON] + trainer_argv, args.gpu, commit, 'probe', args.allow_dirty)
        fields['trainer_command'], fields['command'] = fields['command'], cmd
        return fields

    completed = base.execute_attempt(attempt, 'probe', args.gpu, log_path, fields_factory,
                                     allow_cotenant=args.allow_cotenant)
    after = base.gpu_snapshot(args.gpu)
    arms_before = [completed['resource_before']]
    states = [before, *arms_before, after]
    result = dict(completed['metrics']['probe'], schema_version=1, gpu=args.gpu, before=before,
                  after=after, arms_before=arms_before, attempts=[str(attempt)],
                  reviewed_commit=commit,
                  PROBE_NOT_CLEAN=any(bool(state['compute_apps']) for state in states))
    result.update(probe_attempt=dict(path=str(attempt.resolve()), **{
        part + '_sha256': p.sha256_file(attempt / (part + '.json'))
        for part in ('train_manifest', 'completion')}),
        live_epoch_limit_seconds=1.05 * result['T_epoch'], live_epoch_limit_start='banner')
    result['PROBE_NOT_CLEAN'] |= any(
        state['gpu'] != args.gpu or not before['uuid'] or state['uuid'] != before['uuid']
        or not math.isfinite(state['free_gib']) or state['free_gib'] < 40 for state in states)
    if result['PROBE_NOT_CLEAN']:
        result['passed'] = False
    p.write_manifest(output, result)
    if result['PROBE_NOT_CLEAN']:
        print('PROBE_NOT_CLEAN: GPU resource checks failed; re-probe on a clean GPU.', flush=True)
    return result


if __name__ == '__main__':
    main()
