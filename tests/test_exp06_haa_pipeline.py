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
