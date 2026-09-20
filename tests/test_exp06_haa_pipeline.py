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
YAWAUG = 'ckpt/xRIR_simple_yawaug_8_shot/final/epoch_012.pth'
# exp_09's arm E runs the same pipeline in the room frame, from exp_04's checkpoint.
INITS = {'cyl_or': ('cylindrical_oriented', CYLOR, 'heading'),
         'control_hf': ('simple', 'ckpt/xRIR_simple_8_shot/epoch_12.pth', 'heading'),
         'cyl_hf': ('cylindrical', 'ckpt/xRIR_cyl_8_shot/epoch_12.pth', 'heading'),
         'yawaug': ('simple', YAWAUG, 'room')}
EXP06_ARMS = ('cyl_or', 'control_hf', 'cyl_hf')
EXP09_RECORD = 'worklog/worklog_yixun/exp_09_yawaug_haa_claude'
EXP09_OUT = 'ckpt/exp09/sim2real'


def run(*argv, **environment):
    env = dict(HAA_XRIR_ROOT=HAA_ROOT, **environment)
    result = subprocess.run(['bash', str(SCRIPT), *argv], cwd=str(ROOT), text=True,
                            capture_output=True, env={**_base_env(), **env})
    return result


def _base_env():
    import os
    return {key: os.environ[key] for key in ('PATH', 'HOME', 'USER', 'LANG')
            if key in os.environ}


def log_of(name, tag, stage, record=RECORD, prefix='oriented_cyl'):
    return '{}/{}_<UTC>_haa_{}_{}_{}.log'.format(record, prefix, name, tag, stage)


def child_block(directory, log, command, run_type):
    return ['MKDIR ' + directory,
            'SINK cat >> ' + log,
            'PIDFILE ' + directory + '/child.pid',
            'RUN nohup setsid ' + command,
            'MARKER EXP06_CHILD_EXIT <code> <iso> >> ' + log,
            'RUN {} tools/exp06_finalize.py --run-dir {} --run-type {} --log {} '
            '--child-exit <code>'.format(PYTHON, directory, run_type, log)]


def heading_flag(frame):
    """A room-frame child is launched with no heading directory at all."""
    return '--heading-json-dir {} '.format(HEADING) if frame == 'heading' else ''


def train_command(backbone, init, rooms, save_dir, seed, recipe, root, frame='heading'):
    return ('{} tools/exp06_haa_finetune.py --backbone {} --init {} --rooms {} '
            '{}--save-dir {} --seed {} --job-spec {}/job_spec.json '
            '{}'.format(PYTHON, backbone, init, rooms, heading_flag(frame), save_dir, seed,
                        root, recipe))


def eval_command(backbone, checkpoint, room, save_dir, seed, root, frame='heading'):
    return ('{} tools/exp06_haa_eval.py --backbone {} --checkpoint {} --rooms {} '
            '{}--save-dir {} --seed {} --job-spec {}/job_spec.json'.format(
                PYTHON, backbone, checkpoint, room, heading_flag(frame), save_dir, seed, root))


def job_spec_line(root, name, init, seed, expect, backbone, frame):
    """A room-frame job declares no heading directory and no heading rolls."""
    return ('JOBSPEC {}/job_spec.json init={} checkpoint={} seed={} expect={} backbone={} '
            'frame={} heading={}'.format(root, name, init, seed, expect, backbone, frame,
                                         HEADING if frame == 'heading' else 'none'))


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


def approvals_line(name, checkpoint):
    """F3: 6.4's producer gate, printed before the job root is acquired."""
    return 'APPROVALS {} checkpoint={} heading={} producer=haa_children'.format(
        name, checkpoint, HEADING)


def finetune_job(name, seed, out=OUT, record=RECORD, prefix='oriented_cyl'):
    backbone, init, frame = INITS[name]
    root = '{}/{}/seed{}'.format(out, name, seed)
    tag = 'seed{}'.format(seed)
    lines = ['JOB {} {} backbone={} init={} expect=finetune'.format(name, tag, backbone, init),
             approvals_line(name, init),
             'MKDIR ' + root, 'PIDFILE ' + root + '/launch.pid',
             job_spec_line(root, name, init, seed, 'finetune', backbone, frame),
             'CHECKSPEC {}/job_spec.json expect=finetune'.format(root)]
    lines += child_block(root + '/stage1', log_of(name, tag, 'stage1', record, prefix),
                         train_command(backbone, init, S1_ROOMS, root + '/stage1', seed,
                                       '--epochs 1000 --val-every 10 --tf32', root, frame),
                         'haa_train')
    children = [root + '/stage1']
    for room in ROOMS:
        stage2 = '{}/stage2_{}'.format(root, room)
        lines += child_block(stage2, log_of(name, tag, 'stage2_' + room, record, prefix),
                             train_command(backbone, root + '/stage1/best.pth', room, stage2,
                                           seed, '--epochs 200 --val-every 2 --tf32', root,
                                           frame),
                             'haa_train')
        evaluation = '{}/eval/{}'.format(root, room)
        lines += child_block(evaluation, log_of(name, tag, 'eval_' + room, record, prefix),
                             eval_command(backbone, stage2 + '/best.pth', room, evaluation,
                                          seed, root, frame), 'haa_eval')
        children += [stage2, evaluation]
    joblog = log_of(name, tag, 'job', record, prefix)
    lines.append(job_finalize(root, joblog, 'finetune', children))  # A3: no end marker
    return lines


def zeroshot_job(name, out=OUT, record=RECORD, prefix='oriented_cyl'):
    backbone, init, frame = INITS[name]
    root = '{}/{}/zeroshot'.format(out, name)
    lines = ['JOB {} zeroshot backbone={} init={} expect=zeroshot'.format(name, backbone, init),
             approvals_line(name, init),
             'MKDIR ' + root, 'PIDFILE ' + root + '/launch.pid',
             job_spec_line(root, name, init, 0, 'zeroshot', backbone, frame),
             'CHECKSPEC {}/job_spec.json expect=zeroshot'.format(root)]
    children = []
    for room in ROOMS:
        evaluation = '{}/eval/{}'.format(root, room)
        lines += child_block(evaluation, log_of(name, 'zeroshot', 'eval_' + room, record,
                                                prefix),
                             eval_command(backbone, init, room, evaluation, 0, root, frame),
                             'haa_eval')
        children.append(evaluation)
    joblog = log_of(name, 'zeroshot', 'job', record, prefix)
    lines.append(job_finalize(root, joblog, 'zeroshot', children))  # A3: no end marker
    return lines


def test_the_script_parses():
    assert subprocess.run(['bash', '-n', str(SCRIPT)]).returncode == 0


# --- the golden queue, printed against an output root the test owns -------------------
#
# The CLI path's OUT is the live ckpt/exp06/sim2real. Now that confirmatory runs exist
# there, child() finds their completion.json and prints one `SKIP <dir>` in place of the
# six lines that child's block records, so a golden comparison against the CLI asserts
# the state of one machine's tree instead of what the queue prints. EXP06_PIPELINE_LIB=1
# defines the same functions and returns before the CLI path, so the queue below runs the
# very function that path calls -- run_queue -- with the same dry-run placeholders over a
# temporary root, and the printed paths are substituted back: the golden lines stay
# exactly what a first run of this queue prints, wherever it is run.

DRY_ADAPTER = '''EXP06_PIPELINE_LIB=1 source tools/exp06_haa_pipeline.sh
DRY=1; STAMP='<UTC>'; OWNER='<pid>'      # the placeholders the CLI dry-run path sets
GPU="$QUEUE_GPU"; OUT="$WORK/sim2real"; RECORD="$QUEUE_RECORD"
run_queue $QUEUE_JOBS
'''


def run_dry(work, *jobs, gpu='7', out=OUT, record=RECORD, private='record', **environment):
    """One queue's dry run, with the private output and record roots printed as the real ones.

    ``private`` is the directory name the record is written under, because the child-log
    prefix is derived from the record root: an exp_09 record prints ``yawaug_haa_`` logs.
    """
    env = {**_base_env(), 'WORK': str(work), 'QUEUE_GPU': gpu, 'QUEUE_JOBS': ' '.join(jobs),
           'QUEUE_RECORD': '{}/{}'.format(work, private), 'HAA_XRIR_ROOT': HAA_ROOT,
           **environment}
    result = subprocess.run(['bash', '-c', DRY_ADAPTER], cwd=str(ROOT), text=True,
                            capture_output=True, env=env)
    assert result.returncode == 0, result.stderr
    return result.stdout.replace('{}/sim2real'.format(work), out).replace(
        '{}/{}'.format(work, private), record).splitlines()


def test_the_dry_run_of_one_finetune_seed_is_the_golden_queue(tmp_path):
    assert run_dry(tmp_path, 'cyl_or:0') == finetune_job('cyl_or', 0) + ['QUEUE_DONE gpu=7']


def test_a_completed_child_is_skipped_and_the_rest_of_the_queue_is_unchanged(tmp_path):
    """A resumed queue prints one SKIP line where that child's six lines stood.

    This is what a live output root does to the comparison above, asserted on a tree the
    test owns rather than on whichever confirmatory runs happen to have finished.
    """
    completed = tmp_path / 'sim2real/cyl_or/seed0/stage1'
    completed.mkdir(parents=True)
    (completed / 'completion.json').write_text('{"already": "finalized"}')
    golden = finetune_job('cyl_or', 0) + ['QUEUE_DONE gpu=7']
    stage1 = '{}/cyl_or/seed0/stage1'.format(OUT)
    start = golden.index('MKDIR ' + stage1)
    width = len(child_block(stage1, 'log', 'command', 'haa_train'))
    assert run_dry(tmp_path, 'cyl_or:0') == (golden[:start] + ['SKIP ' + stage1]
                                             + golden[start + width:])


def test_the_dry_run_of_the_zeroshot_job_covers_every_init(tmp_path):
    expected = []
    for name in EXP06_ARMS:
        expected += zeroshot_job(name)
    assert run_dry(tmp_path, 'zeroshot', gpu='1') == expected + ['QUEUE_DONE gpu=1']


# --- exp_09: the same queue in the room frame, under its own roots --------------------


def test_the_dry_run_of_a_yawaug_seed_is_the_room_frame_golden(tmp_path):
    """Arm E runs every child without a heading directory and declares frame=room."""
    printed = run_dry(tmp_path, 'yawaug:0')
    assert printed == finetune_job('yawaug', 0) + ['QUEUE_DONE gpu=7']
    assert not [line for line in printed if '--heading-json-dir' in line]
    assert 'frame=room heading=none' in printed[4]


def test_the_yawaug_zeroshot_set_is_its_own_job_token(tmp_path):
    """Bare `zeroshot` keeps its exp_06 meaning, so arm E asks for its own."""
    assert run_dry(tmp_path, 'yawaug:zeroshot') == zeroshot_job('yawaug') + ['QUEUE_DONE gpu=7']
    printed = '\n'.join(run_dry(tmp_path, 'zeroshot', gpu='1'))
    assert 'yawaug' not in printed and 'cyl_or' in printed


def test_the_exp09_queue_is_three_seeds_and_one_zero_shot_set(tmp_path):
    """31 children and four jobs, as the plan registers them."""
    printed = run_dry(tmp_path, 'yawaug:0', 'yawaug:1', 'yawaug:2', 'yawaug:zeroshot',
                      out=EXP09_OUT, record=EXP09_RECORD, private='exp_09_record',
                      EXP06_HAA_OUT=str(tmp_path / 'sim2real'))
    assert len([line for line in printed if line.startswith('RUN nohup setsid ')]) == 31
    assert len([line for line in printed if line.startswith('JOBSPEC ')]) == 4
    assert printed[-1] == 'QUEUE_DONE gpu=7'


def test_a_mixed_queue_gives_every_job_its_own_frame(tmp_path):
    """The frame is a property of the initialisation, not of the queue that ran first."""
    printed = run_dry(tmp_path, 'yawaug:0', 'cyl_or:0')
    assert printed == (finetune_job('yawaug', 0) + finetune_job('cyl_or', 0)
                       + ['QUEUE_DONE gpu=7'])
    assert run_dry(tmp_path, 'cyl_or:0', 'yawaug:0') == (
        finetune_job('cyl_or', 0) + finetune_job('yawaug', 0) + ['QUEUE_DONE gpu=7'])


def test_the_output_and_record_roots_are_overridable(tmp_path):
    """exp_09 writes under its own roots; the child logs take the record's own prefix."""
    lines = run_dry(tmp_path, 'yawaug:0', out=EXP09_OUT, record=EXP09_RECORD,
                    private='exp_09_record', EXP06_HAA_OUT=str(tmp_path / 'sim2real'))
    assert lines == finetune_job('yawaug', 0, out=EXP09_OUT, record=EXP09_RECORD,
                                 prefix='yawaug_haa') + ['QUEUE_DONE gpu=7']
    printed = '\n'.join(lines)
    assert OUT not in printed and 'oriented_cyl_' not in printed


def test_the_cli_takes_the_output_root_from_the_environment(tmp_path):
    result = run('7', 'yawaug:0', '--dry-run', EXP06_HAA_OUT=str(tmp_path / 'exp09'))
    assert result.returncode == 0, result.stderr
    assert 'out={}/exp09'.format(tmp_path) in result.stdout.splitlines()[0]


def test_the_yawaug_checkpoint_is_overridable(tmp_path):
    printed = '\n'.join(run_dry(tmp_path, 'yawaug:2', EXP09_YAWAUG_CKPT='alt/aug.pth'))
    assert 'init=alt/aug.pth' in printed and YAWAUG not in printed


def test_the_cli_announces_the_gpu_the_jobs_and_the_output_root():
    """The header the CLI prints before the queue: the part of its output that no state
    under the output root can change, and the only golden line still asserted of the
    script as the runbook invokes it."""
    result = run('7', 'cyl_or:0', '--dry-run')
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[:2] == header('7', ['cyl_or:0'])


@pytest.mark.parametrize('job', ['invented:0', 'cyl_or', 'cyl_or:x', 'cyl_or:', ':0',
                                 'zeroshot:0', 'cyl_or:anything:0', 'cyl_or:0:1',
                                 'cyl_or:-1', 'yawaug:zero', 'yawaug:zeroshot:0'])
def test_an_unknown_job_is_refused_before_anything_runs(job):
    result = run('1', job, '--dry-run')
    assert result.returncode == 2 and 'refus' in (result.stderr + result.stdout).lower()
    assert 'MKDIR' not in result.stdout


def test_usage_is_refused_without_a_job():
    assert run('1').returncode == 2
    assert run().returncode == 2


def test_the_heading_directory_and_pretrained_checkpoint_are_overridable(tmp_path):
    printed = '\n'.join(run_dry(tmp_path, 'cyl_or:2', gpu='0',
                                EXP06_HEADING_DIR='alt/heading',
                                EXP06_CYLOR_CKPT='alt/epoch_012.pth'))
    assert 'alt/heading' in printed and 'alt/epoch_012.pth' in printed
    assert HEADING not in printed and CYLOR not in printed
    assert OUT + '/cyl_or/seed2/stage1' in printed


# --- preparation is a gate: a failed job never reaches a child (round 2b finding 1) ----

ADAPTER = '''EXP06_PIPELINE_LIB=1 source tools/exp06_haa_pipeline.sh
DRY=0; STAMP=faults; GPU=7; RECORD="$WORK/record"; OUT="$WORK/out"
mkdir -p -- "$RECORD" "$OUT"
: > "$WORK/events"
child() { printf 'CHILD %s\\n' "$2" >> "$WORK/events"; }
finalize_job() { printf 'FINALIZE %s\\n' "$1" >> "$WORK/events"; }
approvals_ok() { return 0; }          # F3: the gate has its own tests below
'''
INVOKE = 'if {}; then echo "STATUS 0"; else echo "STATUS $?"; fi\n'


def run_lib(script, work, adapter=ADAPTER, **environment):
    """Exercise the pipeline's own functions with launching and finalisation stubbed out."""
    env = {**_base_env(), 'WORK': str(work), 'HAA_XRIR_ROOT': HAA_ROOT, **environment}
    result = subprocess.run(['bash', '-c', adapter + script], cwd=str(ROOT), text=True,
                            capture_output=True, env=env)
    return result, Path(work, 'events').read_text().splitlines()


JOB_ADAPTER = '''EXP06_PIPELINE_LIB=1 source tools/exp06_haa_pipeline.sh
DRY=0; OWNER=$$; RECORD="$WORK/record"; OUT="$WORK/out"
mkdir -p -- "$RECORD" "$OUT/cyl_or/seed0"
: > "$WORK/events"
run() { printf 'FINALIZE %s\\n' "$*" >> "$WORK/events"; }
'''


def test_a_real_job_finalisation_leaves_no_receipt_at_the_job_root(tmp_path):
    """A3: finalize_job appends its queue line and finalizes; it closes no log."""
    root, joblog = tmp_path / 'out/cyl_or/seed0', tmp_path / 'job.log'
    result, events = run_lib(
        INVOKE.format('finalize_job "$WORK/out/cyl_or/seed0" "$WORK/job.log" finetune a b'),
        tmp_path, adapter=JOB_ADAPTER)
    assert 'STATUS 0' in result.stdout, result.stderr
    assert not (root / 'child_exit.json').exists()
    assert joblog.read_text() == 'job {} expect finetune children 2\n'.format(root)
    assert len(events) == 1 and '--run-type haa_job' in events[0]
    assert '--children a b --expect finetune' in events[0]
    assert events[0].endswith('--job-spec {}/job_spec.json'.format(root))


def test_open_job_records_the_owner_pid_at_the_job_root(tmp_path):
    """Pre-merge finding 1: the owner a job completion binds is written when the root opens.

    ``own_launch`` records the pipeline shell's own ``$$`` -- the process that stays alive
    through every child and finalizes the job -- so a job root always carries the
    ``launch.pid`` the finalizer now requires of it, whether it created or adopted the root.
    """
    root = tmp_path / 'out/cyl_or/seed0'
    script = INVOKE.format('open_job "' + str(root) + '"') + 'echo "PID $$"\n'
    result, _ = run_lib(script, tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'STATUS 0' in result.stdout
    assert 'PIDFILE ' + str(root) + '/launch.pid' in result.stdout
    pid = [line for line in result.stdout.splitlines() if line.startswith('PID ')][0].split()[1]
    assert (root / 'launch.pid').read_text().strip() == pid
    result, _ = run_lib(script, tmp_path)  # the queue re-opens an existing root
    pid = [line for line in result.stdout.splitlines() if line.startswith('PID ')][0].split()[1]
    assert (root / 'launch.pid').read_text().strip() == pid


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
    ('approvals_ok() { return 5; }', 'REFUSED approvals'),
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
    """Nit 5: the override is one path, not two arguments.

    Asserted of the stage-1 command line, which a completed stage 1 in a live output
    root would replace with a single SKIP -- so this queue, too, runs against a root
    the test owns.
    """
    override = str(tmp_path / 'path with spaces' / 'epoch_012.pth')
    printed = '\n'.join(run_dry(tmp_path, 'cyl_or:2', gpu='0', EXP06_CYLOR_CKPT=override))
    assert 'init=' + override in printed
    assert '--init ' + override + ' --rooms' in printed


# --- end-to-end: the finalizer on artefacts these wrappers actually write -------------
import json
import os
import shutil

import torch

from sim_to_real.haa_dataset import NO_T60_ROOMS
from tools import exp06_finalize, provenance
from tools import exp06_haa_eval as evaluator
from tools import exp06_haa_finetune as trainer
from test_exp06_finalize import DEAD_PID, pipeline_history, state_keys
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
    """What the launcher leaves behind once its child is gone.

    The receipt names the child, and the finalizer refuses one whose pid is still alive,
    so the fixture records a pid that can never be running -- never the test process's
    own. Only the job root's ``launch.pid`` (the launcher, still draining) may be live.
    """
    Path(log).write_text(text)
    assert exp06_finalize.child_exit_main([
        '--run-dir', str(run_dir), '--log', str(log), '--child-pid', str(DEAD_PID),
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
    joblog.write_text('job output\n')  # A3: a job closes no log and writes no receipt
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


def test_the_job_the_pipeline_really_writes_is_admissible(finetune_seed, clone):
    """A3: the pipeline shell that finalizes a job is alive and leaves no receipt of its own.

    ``tools/exp06_haa_pipeline.sh::finalize_job`` runs in the very shell whose pid the job
    root's ``launch.pid`` names, so the owner exception is the only way any job is ever
    admitted. It no longer closes the job log, and the job root carries no
    ``child_exit.json``: the nine children below it hold the exit evidence, and their pids
    really are dead by the time the launcher finalizes them.
    """
    root, children, spec, joblog = finetune_seed
    completion = Path(root) / 'completion.json'
    if completion.exists():
        completion.unlink()
    assert not (Path(root) / 'child_exit.json').exists()
    assert (Path(root) / 'launch.pid').read_text().strip() == str(os.getpid())
    fields = exp06_finalize.finalize(root, 'haa_job', joblog, 0, repo=clone, children=children,
                                     expect='finetune', job_spec=spec, owner_pid=os.getpid())
    assert fields['admissible_arm'] is True and fields['owner_pid'] == os.getpid()
    assert fields['log'] == {'path': str(Path(joblog).resolve()),
                             'sha256': provenance.sha256_file(joblog)}
    assert 'child_exit_receipt' not in fields and 'child_exit_time' not in fields
    assert completion.is_file()


def test_a_receipt_at_the_job_root_is_refused(finetune_seed, clone):
    """A3: the job root carries no child exit receipt, so one found there is refused."""
    root, children, spec, joblog = finetune_seed
    completion, receipt = Path(root) / 'completion.json', Path(root) / 'child_exit.json'
    if completion.exists():
        completion.unlink()
    receipt.write_text(json.dumps({'child_pid': os.getpid(), 'status': 0}))
    try:
        with pytest.raises(ValueError, match='no child exit receipt'):
            exp06_finalize.finalize(root, 'haa_job', joblog, 0, repo=clone, children=children,
                                    expect='finetune', job_spec=spec, owner_pid=os.getpid())
        assert not completion.exists()
    finally:
        receipt.unlink()


# --- F3: 6.4's producer gate refuses before the job root is acquired -----------------------


REAL_ADAPTER = ''.join(line + '\n' for line in ADAPTER.splitlines()
                       if not line.startswith('approvals_ok()'))


def test_the_real_approvals_gate_refuses_an_unapproved_init(tmp_path):
    """An initialisation the approvals do not name stops the job before anything is made."""
    init = tmp_path / 'not_the_approved_epoch_012.pth'
    init.write_bytes(b'weights nobody approved')
    result, events = run_lib(INVOKE.format('run_finetune cyl_or 0'), tmp_path,
                             adapter=REAL_ADAPTER, EXP06_CYLOR_CKPT=str(init))
    assert events == [] and 'STATUS 1' in result.stdout
    assert 'REFUSED approvals' in result.stdout
    assert 'refusing:' in result.stderr and 'artifacts.epoch_012' in result.stderr
    assert not (tmp_path / 'out/cyl_or/seed0').exists()
    assert not sorted(Path(tmp_path, 'out/cyl_or').glob('seed0_ABORTED_*'))
    failure = sorted(Path(tmp_path, 'out/cyl_or').glob('seed0_PREPARE_FAILED_*.json'))
    assert len(failure) == 1
    assert json.loads(failure[0].read_text())['reason'] == 'approvals'


def test_the_real_approvals_gate_pins_the_historical_initialisations(tmp_path):
    """control_hf and cyl_hf start from exp_01's published weights, pinned by their sha."""
    other = tmp_path / 'epoch_12.pth'
    other.write_bytes(b'not the exp_01 control')
    script = ('init_of() { INIT_BACKBONE=simple; INIT_CKPT=%s; }\n' % other
              + INVOKE.format('run_finetune control_hf 0'))
    result, events = run_lib(script, tmp_path, adapter=REAL_ADAPTER)
    assert events == [] and 'REFUSED approvals' in result.stdout
    assert 'not the registered exp_01' in result.stderr


def test_the_gate_runs_before_the_job_root_is_opened(tmp_path):
    """A refusal must not acquire, adopt or rename anything: order is the guarantee."""
    script = ('approvals_ok() { echo GATE; return 4; }\n'
              'open_job() { echo OPENED; return 0; }\n'
              + INVOKE.format('run_finetune cyl_or 0'))
    result, events = run_lib(script, tmp_path)
    assert 'GATE' in result.stdout and 'OPENED' not in result.stdout
    assert events == [] and 'REFUSED approvals' in result.stdout
