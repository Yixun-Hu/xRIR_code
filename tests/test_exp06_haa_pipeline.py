"""The exp_06 HAA pipeline: one queue of jobs, exclusive children, finalized in place."""
import datetime
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/exp06_haa_pipeline.sh'
PYTHON = '/home/yixunhu/miniconda3/envs/xRIR/bin/python'
RECORD = 'worklog/worklog_yixun/exp_06_oriented_cyl_claude'
OUT = 'ckpt/exp06/sim2real'
HEADING = 'ckpt/exp06/heading'
HAA_ROOT = '/tmp/exp06_pipeline_haa_root'
ROOMS = ('class_room', 'dampened_room', 'hallway', 'complex_room')
S1_ROOMS = 'class_room hallway complex_room'
CYLOR = 'ckpt/exp06/pretrain/xRIR_cylor_8_shot/final/epoch_012.pth'
INITS = {'cyl_or': ('cylindrical_oriented', CYLOR),
         'control_hf': ('simple', 'ckpt/xRIR_simple_8_shot/epoch_12.pth'),
         'cyl_hf': ('cylindrical', 'ckpt/xRIR_cyl_8_shot/epoch_12.pth')}


def run(*argv, **environment):
    env = dict(HAA_XRIR_ROOT=HAA_ROOT, **environment)
    result = subprocess.run(['bash', str(SCRIPT), *argv], cwd=str(ROOT), text=True,
                            capture_output=True, env={**_base_env(), **env})
    return result


def _base_env():
    import os
    return {key: os.environ[key] for key in ('PATH', 'HOME', 'USER', 'LANG')
            if key in os.environ}


def log_of(name, tag, stage):
    return '{}/oriented_cyl_<UTC>_haa_{}_{}_{}.log'.format(RECORD, name, tag, stage)


def child_block(directory, log, command, run_type):
    return ['MKDIR ' + directory,
            'SINK cat >> ' + log,
            'PIDFILE ' + directory + '/child.pid',
            'RUN nohup setsid ' + command,
            'MARKER EXP06_CHILD_EXIT <code> <iso> >> ' + log,
            'RUN {} tools/exp06_finalize.py --run-dir {} --run-type {} --log {} '
            '--child-exit <code>'.format(PYTHON, directory, run_type, log)]


def train_command(backbone, init, rooms, save_dir, seed, recipe, root):
    return ('{} tools/exp06_haa_finetune.py --backbone {} --init {} --rooms {} '
            '--heading-json-dir {} --save-dir {} --seed {} --job-spec {}/job_spec.json '
            '{}'.format(PYTHON, backbone, init, rooms, HEADING, save_dir, seed, root, recipe))


def eval_command(backbone, checkpoint, room, save_dir, seed, root):
    return ('{} tools/exp06_haa_eval.py --backbone {} --checkpoint {} --rooms {} '
            '--heading-json-dir {} --save-dir {} --seed {} --job-spec {}/job_spec.json'.format(
                PYTHON, backbone, checkpoint, room, HEADING, save_dir, seed, root))


def job_finalize(root, log, expect, children):
    return ('RUN {} tools/exp06_finalize.py --run-dir {} --run-type haa_job --log {} '
            '--child-exit 0 --owner-pid <pid> --children {} --expect {} '
            '--job-spec {}/job_spec.json'.format(PYTHON, root, log, ' '.join(children),
                                                 expect, root))


def header(gpu, jobs):
    return ['EXP06_HAA_PIPELINE gpu={} jobs={} stamp=<UTC> dry_run=1 heading={} out={}'.format(
                gpu, ' '.join(jobs), HEADING, OUT),
            'ENV CUDA_VISIBLE_DEVICES={} PYTHONHASHSEED=0 OMP_NUM_THREADS=8 '
            'HAA_XRIR_ROOT={}'.format(gpu, HAA_ROOT)]


def finetune_job(name, seed):
    backbone, init = INITS[name]
    root = '{}/{}/seed{}'.format(OUT, name, seed)
    tag = 'seed{}'.format(seed)
    lines = ['JOB {} {} backbone={} init={} expect=finetune'.format(name, tag, backbone, init),
             'MKDIR ' + root, 'PIDFILE ' + root + '/launch.pid',
             'JOBSPEC {}/job_spec.json init={} checkpoint={} seed={} expect=finetune '
             'backbone={} frame=heading heading={}'.format(root, name, init, seed, backbone,
                                                           HEADING),
             'CHECKSPEC {}/job_spec.json expect=finetune'.format(root)]
    lines += child_block(root + '/stage1', log_of(name, tag, 'stage1'),
                         train_command(backbone, init, S1_ROOMS, root + '/stage1', seed,
                                       '--epochs 1000 --val-every 10 --tf32', root),
                         'haa_train')
    children = [root + '/stage1']
    for room in ROOMS:
        stage2 = '{}/stage2_{}'.format(root, room)
        lines += child_block(stage2, log_of(name, tag, 'stage2_' + room),
                             train_command(backbone, root + '/stage1/best.pth', room, stage2,
                                           seed, '--epochs 200 --val-every 2 --tf32', root),
                             'haa_train')
        evaluation = '{}/eval/{}'.format(root, room)
        lines += child_block(evaluation, log_of(name, tag, 'eval_' + room),
                             eval_command(backbone, stage2 + '/best.pth', room, evaluation,
                                          seed, root), 'haa_eval')
        children += [stage2, evaluation]
    joblog = log_of(name, tag, 'job')
    lines += ['MARKER EXP06_CHILD_EXIT 0 <iso> >> ' + joblog,
              job_finalize(root, joblog, 'finetune', children)]
    return lines


def zeroshot_job(name):
    backbone, init = INITS[name]
    root = '{}/{}/zeroshot'.format(OUT, name)
    lines = ['JOB {} zeroshot backbone={} init={} expect=zeroshot'.format(name, backbone, init),
             'MKDIR ' + root, 'PIDFILE ' + root + '/launch.pid',
             'JOBSPEC {}/job_spec.json init={} checkpoint={} seed=0 expect=zeroshot '
             'backbone={} frame=heading heading={}'.format(root, name, init, backbone, HEADING),
             'CHECKSPEC {}/job_spec.json expect=zeroshot'.format(root)]
    children = []
    for room in ROOMS:
        evaluation = '{}/eval/{}'.format(root, room)
        lines += child_block(evaluation, log_of(name, 'zeroshot', 'eval_' + room),
                             eval_command(backbone, init, room, evaluation, 0, root),
                             'haa_eval')
        children.append(evaluation)
    joblog = log_of(name, 'zeroshot', 'job')
    lines += ['MARKER EXP06_CHILD_EXIT 0 <iso> >> ' + joblog,
              job_finalize(root, joblog, 'zeroshot', children)]
    return lines


def test_the_script_parses():
    assert subprocess.run(['bash', '-n', str(SCRIPT)]).returncode == 0


def test_the_dry_run_of_one_finetune_seed_is_the_golden_queue():
    result = run('7', 'cyl_or:0', '--dry-run')
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == (header('7', ['cyl_or:0'])
                                          + finetune_job('cyl_or', 0) + ['QUEUE_DONE gpu=7'])


def test_the_dry_run_of_the_zeroshot_job_covers_every_init():
    result = run('1', 'zeroshot', '--dry-run')
    assert result.returncode == 0, result.stderr
    expected = header('1', ['zeroshot'])
    for name in ('cyl_or', 'control_hf', 'cyl_hf'):
        expected += zeroshot_job(name)
    assert result.stdout.splitlines() == expected + ['QUEUE_DONE gpu=1']


@pytest.mark.parametrize('job', ['invented:0', 'cyl_or', 'cyl_or:x', 'cyl_or:', ':0',
                                 'zeroshot:0', 'cyl_or:anything:0', 'cyl_or:0:1',
                                 'cyl_or:-1'])
def test_an_unknown_job_is_refused_before_anything_runs(job):
    result = run('1', job, '--dry-run')
    assert result.returncode == 2 and 'refus' in (result.stderr + result.stdout).lower()
    assert 'MKDIR' not in result.stdout


def test_usage_is_refused_without_a_job():
    assert run('1').returncode == 2
    assert run().returncode == 2


def test_the_heading_directory_and_pretrained_checkpoint_are_overridable():
    result = run('0', 'cyl_or:2', '--dry-run', EXP06_HEADING_DIR='alt/heading',
                 EXP06_CYLOR_CKPT='alt/epoch_012.pth')
    assert result.returncode == 0, result.stderr
    assert 'alt/heading' in result.stdout and 'alt/epoch_012.pth' in result.stdout
    assert HEADING not in result.stdout and CYLOR not in result.stdout
    assert OUT + '/cyl_or/seed2/stage1' in result.stdout


# --- preparation is a gate: a failed job never reaches a child (round 2b finding 1) ----

ADAPTER = '''EXP06_PIPELINE_LIB=1 source tools/exp06_haa_pipeline.sh
DRY=0; STAMP=faults; GPU=7; RECORD="$WORK/record"; OUT="$WORK/out"
mkdir -p -- "$RECORD" "$OUT"
: > "$WORK/events"
child() { printf 'CHILD %s\\n' "$2" >> "$WORK/events"; }
finalize_job() { printf 'FINALIZE %s\\n' "$1" >> "$WORK/events"; }
'''
INVOKE = 'if {}; then echo "STATUS 0"; else echo "STATUS $?"; fi\n'


def run_lib(script, work, **environment):
    """Exercise the pipeline's own functions with launching and finalisation stubbed out."""
    env = {**_base_env(), 'WORK': str(work), 'HAA_XRIR_ROOT': HAA_ROOT, **environment}
    result = subprocess.run(['bash', '-c', ADAPTER + script], cwd=str(ROOT), text=True,
                            capture_output=True, env=env)
    return result, Path(work, 'events').read_text().splitlines()


@pytest.fixture(scope='module')
def flat_dampened(tmp_path_factory):
    """A private HAA cache whose dampened_room has no acoustic axis, and its headings."""
    from tools import exp06_heading
    from test_exp06_haa import confirmatory, write_room
    base = tmp_path_factory.mktemp('flat')
    root, headings = base / 'HAA_xrir', base / 'heading'
    headings.mkdir(parents=True)
    decisions = {}
    for room in ROOMS:
        levels = [0] * 12 if room == 'dampened_room' else None
        record = exp06_heading.estimate_room_heading(write_room(root / room, levels))
        exp06_heading.write_heading_json(headings / (room + '.json'), confirmatory(record))
        decisions[room] = record['decision']
    assert decisions == {'class_room': 'estimated', 'hallway': 'estimated',
                         'complex_room': 'estimated', 'dampened_room': 'refused'}
    init = base / 'init.pth'
    init.write_bytes(b'initialisation')
    return str(root), str(headings), str(init)


@pytest.mark.parametrize('fault,cause', [
    ('open_job() { return 3; }', 'REFUSED open_job'),
    ('job_spec() { return 2; }', 'REFUSED job_spec'),
    ('job_spec() { return 0; }\ncheck_spec() { return 2; }', 'REFUSED check_spec')])
def test_a_failed_preparation_launches_no_child(tmp_path, fault, cause):
    """Called through `||`, these helpers suppress errexit: each status is checked."""
    result, events = run_lib(fault + '\n' + INVOKE.format('run_finetune cyl_or 0'), tmp_path)
    assert events == [] and cause in result.stdout
    assert 'STATUS 0' not in result.stdout and 'STATUS 1' in result.stdout


@pytest.mark.parametrize('fault,reason', [
    ('job_spec() { return 2; }', 'job_spec'),
    ('job_spec() { return 0; }\ncheck_spec() { return 2; }', 'check_spec')])
def test_a_failed_preparation_leaves_a_structured_receipt(tmp_path, fault, reason):
    """Nit 8: the cause outlives the terminal, and the root this attempt created is
    named _ABORTED_."""
    result, events = run_lib(fault + '\n' + INVOKE.format('run_finetune cyl_or 0'), tmp_path)
    assert events == [] and 'STATUS 1' in result.stdout
    root = tmp_path / 'out/cyl_or/seed0'
    aborted = root.with_name(root.name + '_ABORTED_prepare_' + reason)
    assert not root.exists() and aborted.is_dir()
    record = json.loads((aborted / 'preparation_failure.json').read_text())
    assert record['reason'] == reason and record['job_root'] == str(root)
    assert record['aborted_dir'] == str(aborted) and record['owned_root'] is True
    assert record['git']['HEAD'] == subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True,
        capture_output=True).stdout.strip()
    datetime.datetime.fromisoformat(record['failed_at'])


# --- round-3a finding 3: a root this attempt did not acquire is never touched ----------


def occupied_root(tmp_path):
    """A populated job root some earlier attempt left behind."""
    root = tmp_path / 'out/cyl_or/seed0'
    root.mkdir(parents=True)
    (root / 'completion.json').write_text('{"already": "finalized"}')
    return root


def failure_files(root):
    return sorted(root.parent.glob(root.name + '_PREPARE_FAILED_*.json'))


@pytest.mark.parametrize('fault,reason,extra', [
    ('open_job() { return 3; }', 'open_job', []),
    ('job_spec() { return 2; }', 'job_spec', ['launch.pid'])])
def test_a_failed_preparation_never_renames_a_root_it_did_not_create(tmp_path, fault,
                                                                     reason, extra):
    """Finding 3: an unowned root keeps its contents, its name and its recorded paths."""
    root = occupied_root(tmp_path)
    result, events = run_lib(fault + '\n' + INVOKE.format('run_finetune cyl_or 0'), tmp_path)
    assert events == [] and 'STATUS 1' in result.stdout
    assert sorted(p.name for p in root.iterdir()) == sorted(['completion.json'] + extra)
    assert not sorted(root.parent.glob(root.name + '_ABORTED_*'))
    record = json.loads(failure_files(root)[0].read_text())
    assert record['reason'] == reason and record['job_root'] == str(root)
    assert record['owned_root'] is False and record['aborted_dir'] is None
    assert record['git']['HEAD'] and datetime.datetime.fromisoformat(record['failed_at'])


def test_a_root_whose_pid_file_cannot_be_written_is_left_alone(tmp_path):
    """A failed acquisition is not ownership: launch.pid cannot be written here."""
    root = occupied_root(tmp_path)
    (root / 'launch.pid').mkdir()
    result, events = run_lib(INVOKE.format('run_finetune cyl_or 0'), tmp_path)
    assert events == [] and 'STATUS 1' in result.stdout
    assert 'REFUSED open_job' in result.stdout
    assert sorted(p.name for p in root.iterdir()) == ['completion.json', 'launch.pid']
    assert (root / 'launch.pid').is_dir()
    assert not sorted(root.parent.glob(root.name + '_ABORTED_*'))
    assert json.loads(failure_files(root)[0].read_text())['owned_root'] is False


def test_a_second_refused_preparation_never_overwrites_the_first(tmp_path):
    script = 'job_spec() { return 2; }\n' + INVOKE.format('run_finetune cyl_or 0') * 2
    result, events = run_lib(script, tmp_path)
    assert events == [] and result.stdout.count('STATUS 1') == 2
    aborted = sorted(Path(tmp_path, 'out/cyl_or').glob('seed0_ABORTED_prepare_job_spec*'))
    assert len(aborted) == 2
    assert all((directory / 'preparation_failure.json').is_file() for directory in aborted)


def test_a_refused_heading_launches_no_child(tmp_path, flat_dampened):
    """A refused dampened_room stops stage 1 too, which never trains on that room."""
    root, headings, init = flat_dampened
    result, events = run_lib(INVOKE.format('run_finetune cyl_or 0'), tmp_path,
                             EXP06_HEADING_DIR=headings, HAA_XRIR_ROOT=root,
                             EXP06_CYLOR_CKPT=init)
    assert events == [] and 'STATUS 1' in result.stdout
    assert 'REFUSED job_spec' in result.stdout
    assert 'dampened_room' in result.stderr and 'refus' in result.stderr
    assert not (tmp_path / 'out/cyl_or/seed0/job_spec.json').exists()
    aborted = tmp_path / 'out/cyl_or/seed0_ABORTED_prepare_job_spec'
    assert json.loads((aborted / 'preparation_failure.json').read_text())['reason'] == 'job_spec'
    assert not (aborted / 'job_spec.json').exists()


def test_a_fully_decided_job_declares_itself_and_launches_every_child(tmp_path, cache):
    """The positive control: the same queue, with every room decided and bound."""
    result, events = run_lib(INVOKE.format('run_finetune cyl_or 0'), tmp_path,
                             EXP06_HEADING_DIR=cache['heading'], HAA_XRIR_ROOT=cache['root'],
                             EXP06_CYLOR_CKPT=cache['init'])
    assert 'STATUS 0' in result.stdout, result.stderr
    assert len([line for line in events if line.startswith('CHILD ')]) == 9
    assert events[-1].startswith('FINALIZE ')
    spec = json.loads((tmp_path / 'out/cyl_or/seed0/job_spec.json').read_text())
    assert spec['heading'] == {room: 128 for room in ROOMS}
    assert spec['expect'] == 'finetune' and spec['frame'] == 'heading' and spec['seed'] == 0
    assert spec['init'] == 'cyl_or' and spec['rooms'] == sorted(ROOMS)


def test_a_failed_child_stops_its_own_job(tmp_path):
    result, events = run_lib(
        'prepare_job() { return 0; }\n'
        "child() { printf 'CHILD %s\\n' \"$2\" >> \"$WORK/events\"; return 1; }\n"
        + INVOKE.format('run_finetune cyl_or 0'), tmp_path)
    assert events == ['CHILD ' + str(tmp_path / 'out/cyl_or/seed0/stage1')]
    assert 'STATUS 1' in result.stdout


def test_the_queue_counts_every_failure_and_never_reports_done(tmp_path):
    """A finalizer failure on one job must not be reported as a completed queue."""
    result, events = run_lib(
        'prepare_job() { return 0; }\n'
        "finalize_job() { printf 'FINALIZE %s\\n' \"$1\" >> \"$WORK/events\"; return 1; }\n"
        + INVOKE.format('run_queue cyl_or:0 cyl_or:1'), tmp_path)
    assert 'QUEUE_FAILED 2' in result.stdout and 'QUEUE_DONE' not in result.stdout
    assert 'STATUS 1' in result.stdout
    assert len([line for line in events if line.startswith('FINALIZE ')]) == 2


def test_a_checkpoint_path_with_spaces_is_never_split(tmp_path):
    """Nit 5: the override is one path, not two arguments."""
    override = str(tmp_path / 'path with spaces' / 'epoch_012.pth')
    result = run('0', 'cyl_or:2', '--dry-run', EXP06_CYLOR_CKPT=override)
    assert result.returncode == 0, result.stderr
    assert 'init=' + override in result.stdout
    assert '--init ' + override + ' --rooms' in result.stdout


# --- end-to-end: the finalizer on artefacts these wrappers actually write -------------
import json
import os
import shutil

import torch

from sim_to_real.haa_dataset import NO_T60_ROOMS
from tools import exp06_finalize
from tools import exp06_haa_eval as evaluator
from tools import exp06_haa_finetune as trainer
from test_exp06_finalize import pipeline_history, state_keys
from test_exp06_haa import MAX_LEN, cache  # noqa: F401  (session fixture)

PRETRAIN = 'pretrain_epoch_012'
ENTRY_FILES = ('tools/exp06_haa_finetune.py', 'tools/exp06_haa_eval.py',
               'tools/exp06_haa_pipeline.sh', 'tools/exp06_finalize.py',
               'tools/exp06_heading.py', 'tools/exp06_eval.py', 'tools/exp06_eval_launch.py')


@pytest.fixture(scope='module')
def clone(tmp_path_factory):
    """A clone whose HEAD carries this working tree's entry points.

    The finalizer's three-way closure check compares the working tree, the bytes at
    spawn and the reviewed blobs, so the repository a child records must be one whose
    HEAD is these files -- true in production, and made true here by committing them.
    """
    root = tmp_path_factory.mktemp('exp06pipe') / 'repo'
    subprocess.run(['git', 'clone', '--local', '--quiet', '--single-branch', str(ROOT),
                    str(root)], check=True)
    for name in ENTRY_FILES:
        shutil.copy2(str(ROOT / name), str(root / name))
    subprocess.run(['git', 'add', '-A'], cwd=str(root), check=True)
    subprocess.run(['git', '-c', 'user.email=a@b', '-c', 'user.name=t', 'commit', '-q',
                    '--allow-empty', '-m', 'round 2b working tree'], cwd=str(root), check=True)
    assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=str(root),
                                   text=True) == ''
    return root


def tiny_state(tag=0.0):
    keys = state_keys()
    return {key: (torch.full((1,), tag) if index == 0 else torch.zeros(1))
            for index, key in enumerate(keys)}


def dead_pid():
    """A pid no process can hold: the finalizer refuses a receipt whose child is alive."""
    return int(Path('/proc/sys/kernel/pid_max').read_text().strip()) + 1


def close_child(run_dir, log, text='child output\n'):
    Path(log).write_text(text)
    assert exp06_finalize.child_exit_main([
        '--run-dir', str(run_dir), '--log', str(log), '--child-pid', str(dead_pid()),
        '--status', '0', '--started-at', '2026-09-15T00:00:00+00:00']) == 0


def launched_under(root):
    """Every child of a seed is launched under the job spec written before the first one."""
    spec = Path(root) / 'job_spec.json'
    return ['--job-spec', str(spec)] if spec.is_file() else []


def make_train_child(clone, cache, root, name, rooms, init, tag, seed=0):
    save = root / name
    argv = ['--backbone', 'cylindrical_oriented', '--init', str(init), '--rooms', *rooms,
            '--heading-json-dir', cache['heading'], '--save-dir', str(save),
            '--root', cache['root'], '--max-len', str(MAX_LEN), '--seed', str(seed),
            '--epochs', '4', '--val-every', '2', *launched_under(root)]
    args = trainer.build_parser().parse_args(argv)
    context = trainer.prepare(args, command=argv, repo=clone)
    trainer.write_records(args, context)
    rows, summary = pipeline_history(args.epochs, args.val_every)
    (save / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (save / 'summary.json').write_text(json.dumps(summary))
    for index, checkpoint in enumerate(('best.pth', 'last.pth')):
        torch.save(tiny_state(tag + index / 10.0), str(save / checkpoint))
    log = root / (name.replace('/', '_') + '.log')
    close_child(save, log)
    exp06_finalize.finalize(save, 'haa_train', log, 0, repo=clone)
    return save


def make_eval_child(clone, cache, root, room, checkpoint, seed=0):
    save = root / 'eval' / room
    argv = ['--backbone', 'cylindrical_oriented', '--checkpoint', str(checkpoint),
            '--rooms', room, '--heading-json-dir', cache['heading'], '--save-dir', str(save),
            '--root', cache['root'], '--max-len', str(MAX_LEN), '--seed', str(seed),
            *launched_under(root)]
    args = evaluator.build_parser().parse_args(argv)
    context = evaluator.prepare(args, command=argv, repo=clone)
    trainer.write_records(args, context)
    dataset = context.datasets[room]
    indices = [idx for _, idx in dataset.items]
    meta = evaluator.per_sample_meta(args, context, room)
    per = {'index': indices, 'ir_path': ['{}/{}'.format(room, idx) for idx in indices],
           'edt': [0.05, 0.07], 'c50': [1.1, 1.3],
           't60': [float('nan')] * 2 if room in NO_T60_ROOMS else [4.0, 6.0],
           'stft_mse': [0.2, 0.4], 'loss': [0.03, 0.05], 'env': [1.0, 3.0],
           'side_label': evaluator.side_labels(dataset, room, indices)}
    counts = {'c50_outliers': 0, 't60_invalid': 0, 'edt_invalid': 0}
    summary = evaluator.room_summary(args, room, per, counts, meta, 1.5)
    evaluator.write_room_outputs(args, room, summary, per, meta)
    log = root / ('eval_' + room + '.log')
    close_child(save, log)
    exp06_finalize.finalize(save, 'haa_eval', log, 0, repo=clone)
    return save


def write_job_spec(clone, cache, root, init, seed=0, expect='finetune'):
    script = ('EXP06_PIPELINE_LIB=1 source tools/exp06_haa_pipeline.sh; DRY=0; '
              'job_spec "$1" "$2" "$3" "$4" "$5" "$6"')
    environment = dict(_base_env(), PYTHONPATH=str(clone), EXP06_HEADING_DIR=cache['heading'],
                       HAA_XRIR_ROOT=cache['root'])
    subprocess.run(['bash', '-c', script, 'bash', str(root / 'job_spec.json'), PRETRAIN,
                    str(init), str(seed), expect, 'cylindrical_oriented'],
                   cwd=str(clone), check=True, env=environment)
    return root / 'job_spec.json'


@pytest.fixture(scope='module')
def finetune_seed(clone, cache, tmp_path_factory):
    """One complete seed: nine children written and certified exactly as the queue does."""
    root = tmp_path_factory.mktemp('seed') / 'seed0'
    root.mkdir(parents=True)
    init = root.parent / (PRETRAIN + '.pth')
    torch.save(tiny_state(99.0), str(init))
    spec = write_job_spec(clone, cache, root, init)  # written before the first child starts
    children = ['stage1']
    make_train_child(clone, cache, root, 'stage1', ['class_room', 'hallway', 'complex_room'],
                     init, 1.0)
    for tag, room in enumerate(ROOMS, 2):
        make_train_child(clone, cache, root, 'stage2_' + room, [room],
                         root / 'stage1/best.pth', float(tag))
        make_eval_child(clone, cache, root, room, root / ('stage2_' + room) / 'best.pth')
        children += ['stage2_' + room, 'eval/' + room]
    joblog = root / 'job.log'
    close_child(root, joblog, text='job output\n')
    (root / 'launch.pid').write_text(str(os.getpid()) + '\n')
    return root, [str(root / name) for name in children], str(spec), joblog


def test_a_complete_seed_is_an_admissible_arm(finetune_seed, clone, cache):
    root, children, spec, joblog = finetune_seed
    fields = exp06_finalize.finalize(root, 'haa_job', joblog, 0, repo=clone, children=children,
                                     expect='finetune', job_spec=spec, owner_pid=os.getpid())
    assert fields['admissible_arm'] is True and fields['diagnostic'] is False
    assert set(fields['children']) == set(exp06_finalize.expected_children('finetune'))
    assert fields['backbone'] == 'cylindrical_oriented' and fields['frame'] == 'heading'
    assert fields['init'] == PRETRAIN and fields['seed'] == 0
    declared = json.loads(Path(spec).read_text())
    assert fields['init_sha256'] == declared['init_sha256']
    assert fields['heading'] == declared['heading'] == {room: 128 for room in ROOMS}
    assert (root / 'completion.json').is_file()


@pytest.mark.parametrize('field,value,cause', [
    ('backbone', 'cylindrical', 'backbone'), ('frame', 'room', 'frame'), ('seed', 1, 'seed'),
    ('expect', 'zeroshot', 'expect'), ('init_sha256', 'a' * 64, 'init'),
    ('rooms', ['hallway'], 'rooms'), ('init', 'somebody_elses_pretrain', 'init'),
    ('heading', {room: 0 for room in ROOMS}, 'rolls')])
def test_a_flipped_frozen_field_refuses_the_job(finetune_seed, clone, field, value, cause):
    """Round 2b finding 3: every mutation is refused at first admission, and by name.

    A job that has already published a completion would refuse a changed result merely
    because it differs from the published bytes, so each mutation is tried against a job
    with no completion at all and must name the field it contradicts.
    """
    root, children, spec, joblog = finetune_seed
    completion = Path(root) / 'completion.json'
    if completion.exists():
        completion.unlink()
    original = Path(spec).read_text()
    Path(spec).write_text(json.dumps(dict(json.loads(original), **{field: value})))
    try:
        with pytest.raises(ValueError, match=cause):
            exp06_finalize.finalize(root, 'haa_job', joblog, 0, repo=clone, children=children,
                                    expect='finetune', job_spec=spec, owner_pid=os.getpid())
        assert not completion.exists()
    finally:
        Path(spec).write_text(original)


def test_a_child_of_another_job_spec_is_refused(finetune_seed, clone, tmp_path):
    """The declaration each child recorded at launch is the one the job may admit."""
    root, children, spec, joblog = finetune_seed
    completion = Path(root) / 'completion.json'
    if completion.exists():
        completion.unlink()
    other = tmp_path / 'other_job_spec.json'  # the same declaration, not the same bytes
    other.write_text(json.dumps(json.loads(Path(spec).read_text()), sort_keys=True))
    with pytest.raises(ValueError, match='job spec'):
        exp06_finalize.finalize(root, 'haa_job', joblog, 0, repo=clone, children=children,
                                expect='finetune', job_spec=str(other), owner_pid=os.getpid())
    assert not completion.exists()


def test_a_live_pid_in_a_child_refuses_the_job(finetune_seed, clone):
    root, children, spec, joblog = finetune_seed
    marker = Path(children[0]) / 'launch.pid'
    marker.write_text(str(os.getpid()) + '\n')
    try:
        with pytest.raises(ValueError, match='alive'):
            exp06_finalize.finalize(root, 'haa_job', joblog, 0, repo=clone, children=children,
                                    expect='finetune', job_spec=spec, owner_pid=os.getpid())
    finally:
        marker.unlink()
