"""The exp_06 HAA pipeline: one queue of jobs, exclusive children, finalized in place."""
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


def train_command(backbone, init, rooms, save_dir, seed, recipe):
    return ('{} tools/exp06_haa_finetune.py --backbone {} --init {} --rooms {} '
            '--heading-json-dir {} --save-dir {} --seed {} {}'.format(
                PYTHON, backbone, init, rooms, HEADING, save_dir, seed, recipe))


def eval_command(backbone, checkpoint, room, save_dir, seed):
    return ('{} tools/exp06_haa_eval.py --backbone {} --checkpoint {} --rooms {} '
            '--heading-json-dir {} --save-dir {} --seed {}'.format(
                PYTHON, backbone, checkpoint, room, HEADING, save_dir, seed))


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
                                                           HEADING)]
    lines += child_block(root + '/stage1', log_of(name, tag, 'stage1'),
                         train_command(backbone, init, S1_ROOMS, root + '/stage1', seed,
                                       '--epochs 1000 --val-every 10 --tf32'), 'haa_train')
    children = [root + '/stage1']
    for room in ROOMS:
        stage2 = '{}/stage2_{}'.format(root, room)
        lines += child_block(stage2, log_of(name, tag, 'stage2_' + room),
                             train_command(backbone, root + '/stage1/best.pth', room, stage2,
                                           seed, '--epochs 200 --val-every 2 --tf32'),
                             'haa_train')
        evaluation = '{}/eval/{}'.format(root, room)
        lines += child_block(evaluation, log_of(name, tag, 'eval_' + room),
                             eval_command(backbone, stage2 + '/best.pth', room, evaluation,
                                          seed), 'haa_eval')
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
             'backbone={} frame=heading heading={}'.format(root, name, init, backbone, HEADING)]
    children = []
    for room in ROOMS:
        evaluation = '{}/eval/{}'.format(root, room)
        lines += child_block(evaluation, log_of(name, 'zeroshot', 'eval_' + room),
                             eval_command(backbone, init, room, evaluation, 0), 'haa_eval')
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
                                 'zeroshot:0'])
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


def close_child(run_dir, log, text='child output\n'):
    Path(log).write_text(text)
    assert exp06_finalize.child_exit_main([
        '--run-dir', str(run_dir), '--log', str(log), '--child-pid', str(os.getpid()),
        '--status', '0', '--started-at', '2026-09-15T00:00:00+00:00']) == 0


def make_train_child(clone, cache, root, name, rooms, init, tag, seed=0):
    save = root / name
    argv = ['--backbone', 'cylindrical_oriented', '--init', str(init), '--rooms', *rooms,
            '--heading-json-dir', cache['heading'], '--save-dir', str(save),
            '--root', cache['root'], '--max-len', str(MAX_LEN), '--seed', str(seed),
            '--epochs', '4', '--val-every', '2']
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
            '--root', cache['root'], '--max-len', str(MAX_LEN), '--seed', str(seed)]
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
    children = ['stage1']
    make_train_child(clone, cache, root, 'stage1', ['class_room', 'hallway', 'complex_room'],
                     init, 1.0)
    for tag, room in enumerate(ROOMS, 2):
        make_train_child(clone, cache, root, 'stage2_' + room, [room],
                         root / 'stage1/best.pth', float(tag))
        make_eval_child(clone, cache, root, room, root / ('stage2_' + room) / 'best.pth')
        children += ['stage2_' + room, 'eval/' + room]
    spec = write_job_spec(clone, cache, root, init)
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


@pytest.mark.parametrize('field,value', [
    ('backbone', 'cylindrical'), ('frame', 'room'), ('seed', 1), ('expect', 'zeroshot'),
    ('init_sha256', 'a' * 64), ('rooms', ['hallway']), ('init', 'somebody_elses_pretrain'),
    ('heading', {room: 0 for room in ROOMS})])
def test_a_flipped_frozen_field_refuses_the_job(finetune_seed, clone, field, value):
    root, children, spec, joblog = finetune_seed
    original = Path(spec).read_text()
    Path(spec).write_text(json.dumps(dict(json.loads(original), **{field: value})))
    try:
        with pytest.raises(ValueError):
            exp06_finalize.finalize(root, 'haa_job', joblog, 0, repo=clone, children=children,
                                    expect='finetune', job_spec=spec, owner_pid=os.getpid())
    finally:
        Path(spec).write_text(original)


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
