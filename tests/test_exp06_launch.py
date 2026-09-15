"""Launch preflight and the dry-run argv of tools/exp06_launch.sh."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import exp06_finalize, exp06_profiles, provenance

REPO = Path(__file__).resolve().parents[1]
APPROVED = ('worklog/worklog_yixun/exp_06_oriented_cyl_claude/'
            'oriented_cyl_results_assets/approved_digests.json')


@pytest.fixture
def repo(tmp_path):
    """A small real repository: preflight reads git state, never the sources."""
    root = tmp_path / 'repo'
    (root / 'worklog').mkdir(parents=True)
    (root / 'tools').mkdir()
    (root / 'tools/exp06_launch.sh').write_text('#!/usr/bin/env bash\n')
    (root / 'worklog/notes.md').write_text('notebook\n')
    write_approvals(root, {key: '{:064x}'.format(index)
                           for index, key in enumerate(exp06_profiles.TRAINING_KEYS)})
    for command in (['init', '-q'], ['add', '-A'], ['-c', 'user.email=a@b', '-c', 'user.name=t',
                                                    'commit', '-q', '-m', 'initial']):
        subprocess.run(['git'] + command, cwd=root, check=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    return root, head


@pytest.fixture(autouse=True)
def approved_code(monkeypatch, tmp_path_factory, request):
    """Most preflight cases gate git, pids and GPUs; the approvals have their own cases.

    The fixture repository carries a filled approvals file at the registered relative
    path and the digests it pins are what this checkout is made to report.
    """
    digests = {key: '{:064x}'.format(index)
               for index, key in enumerate(exp06_profiles.TRAINING_KEYS)}
    monkeypatch.setattr(exp06_profiles, 'compute_code_digests', lambda *a, **k: dict(digests))
    return digests


def write_approvals(root, digests=None):
    path = Path(root) / exp06_profiles.APPROVED_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    value = json.loads(exp06_profiles.TEMPLATE_PATH.read_text())
    value['code'].update(digests or {})
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    return path


@pytest.fixture
def fake_nvidia_smi(tmp_path, monkeypatch):
    """A fake nvidia-smi on PATH reports whatever compute apps a test writes."""
    binary = tmp_path / 'bin'
    binary.mkdir()
    script = binary / 'nvidia-smi'
    script.write_text('#!/usr/bin/env bash\ncat "$(dirname "$0")/apps.txt"\n')
    script.chmod(0o755)
    monkeypatch.setenv('PATH', str(binary) + os.pathsep + os.environ['PATH'])

    def apps(text):
        (binary / 'apps.txt').write_text(text)

    apps('')
    return apps


def test_preflight_accepts_a_clean_tree_at_the_reviewed_commit(repo, fake_nvidia_smi):
    root, head = repo
    record = exp06_finalize.preflight('full', 1, head, attempt_root=root.parent / 'a', repo=root)
    assert record['reviewed_commit'] == head and record['gpu'] == 1
    assert record['gpu_compute_apps'] == [] and record['live_launches'] == []
    assert record['git_state']['HEAD'] == head and record['mode'] == 'full'
    (root / 'worklog/notes.md').write_text('edited notebook\n')
    assert exp06_finalize.preflight('smoke', 0, head, repo=root)['mode'] == 'smoke'


def test_preflight_refuses_a_wrong_commit_or_a_dirty_tree(repo, fake_nvidia_smi):
    root, head = repo
    with pytest.raises(ValueError, match='reviewed commit'):
        exp06_finalize.preflight('full', 1, 'a' * 40, repo=root)
    with pytest.raises(ValueError, match='reviewed commit'):
        exp06_finalize.preflight('full', 1, head[:12], repo=root)
    (root / 'tools/exp06_launch.sh').write_text('#!/usr/bin/env bash\n# edited\n')
    with pytest.raises(ValueError, match='dirty'):
        exp06_finalize.preflight('full', 1, head, repo=root)


def test_preflight_refuses_a_busy_gpu_only_where_it_matters(repo, fake_nvidia_smi):
    root, head = repo
    fake_nvidia_smi('3141\n')
    for mode in ('full', 'probe'):
        with pytest.raises(ValueError, match='GPU'):
            exp06_finalize.preflight(mode, 1, head, repo=root)
    assert exp06_finalize.preflight('smoke', 1, head, repo=root)['gpu_compute_apps'] is None
    fake_nvidia_smi('')
    assert exp06_finalize.preflight('full', 1, head, repo=root)['gpu_compute_apps'] == []


def test_preflight_refuses_a_live_or_unreadable_launch_pid(repo, fake_nvidia_smi):
    root, head = repo
    attempts = root.parent / 'attempts'  # ckpt/ lives outside the tracked tree
    (attempts / 'attempt_20260916T130000').mkdir(parents=True)
    (attempts / 'attempt_20260916T130000/launch.pid').write_text(str(os.getpid()) + '\n')
    with pytest.raises(ValueError, match='launch'):
        exp06_finalize.preflight('full', 1, head, attempt_root=attempts, repo=root)
    (attempts / 'attempt_20260916T130000/launch.pid').write_text('not-a-pid\n')
    with pytest.raises(ValueError, match='launch.pid'):
        exp06_finalize.preflight('full', 1, head, attempt_root=attempts, repo=root)
    (attempts / 'attempt_20260916T130000/launch.pid').write_text('999999999\n')
    assert exp06_finalize.preflight('full', 1, head, attempt_root=attempts, repo=root)


def test_preflight_scans_every_launch_location(repo, fake_nvidia_smi, tmp_path):
    """Should-fix 5: smoke pid files live outside the attempt root and count too."""
    root, head = repo
    attempts, smoke = tmp_path / 'attempts', tmp_path / '_smoke'
    (attempts / 'attempt_x').mkdir(parents=True)
    (smoke / 'exp06_train_t1_x').mkdir(parents=True)
    (smoke / 'exp06_train_t1_x/launch.pid').write_text(str(os.getpid()) + '\n')
    assert exp06_finalize.preflight('full', 1, head, attempt_root=attempts, repo=root)
    with pytest.raises(ValueError, match='launch'):
        exp06_finalize.preflight('full', 1, head, attempt_root=[attempts, smoke], repo=root)
    (smoke / 'exp06_train_t1_x/launch.pid').unlink()
    (smoke / 'exp06_train_t1_x/child.pid').write_text(str(os.getpid()) + '\n')
    with pytest.raises(ValueError, match='launch'):
        exp06_finalize.preflight('smoke', 1, head, attempt_root=[attempts, smoke], repo=root)


def test_a_live_owner_may_finalize_its_own_attempt(tmp_path):
    """Should-fix 5: the launcher stays alive through finalisation, others may not."""
    run = tmp_path / 'attempt'
    run.mkdir()
    (run / 'launch.pid').write_text('{}\n'.format(os.getpid()))
    assert exp06_finalize.refuse_live_launch(run, owner_pid=os.getpid()) == os.getpid()
    with pytest.raises(ValueError, match='alive'):
        exp06_finalize.refuse_live_launch(run)
    (run / 'child.pid').write_text('{}\n'.format(os.getpid()))
    with pytest.raises(ValueError, match='alive'):
        exp06_finalize.refuse_live_launch(run, owner_pid=os.getpid())


def test_preflight_cli_exits_two_on_refusal(repo, fake_nvidia_smi):
    root, head = repo
    command = [sys.executable, 'tools/exp06_finalize.py', 'preflight', '--mode', 'smoke',
               '--gpu', '1', '--reviewed-commit', head, '--repo', str(root), '--exploratory']
    env = {**os.environ, 'PYTHONPATH': str(REPO)}
    ok = subprocess.run(command, cwd=REPO, capture_output=True, text=True, env=env)
    assert ok.returncode == 0, ok.stderr  # the child process computes real digests
    assert json.loads(ok.stdout.split('EXP06_PREFLIGHT_OK ')[1])['mode'] == 'smoke'
    wrong = subprocess.run(command[:-5] + ['--reviewed-commit', 'b' * 40, '--repo', str(root)],
                           cwd=REPO, capture_output=True, text=True, env=env)
    assert wrong.returncode == 2 and 'EXP06_PREFLIGHT_REFUSED' in wrong.stderr


LAUNCHER = 'tools/exp06_launch.sh'
DATA_ROOT = '/home/yixunhu/data_cache/AcousticRooms'
ENV_LINE = ('ENV CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=8'
            ' XRIR_DATA_PATH=' + DATA_ROOT)
PYTHON = '/home/yixunhu/miniconda3/envs/xRIR/bin/python'
ROOT = 'ckpt/exp06/pretrain/xRIR_cylor_8_shot'
ATTEMPT = ROOT + '/attempt_<UTC>'
COMMIT = 'a' * 40
TRAIN_ARGV = (PYTHON + ' tools/exp06_train.py --backbone cylindrical_oriented --save-dir '
              + ATTEMPT + ' --num-shot 8 --max-len 9600 --lr 1e-3 --weight-decay 1e-4'
              ' --decay-epochs 3 --lr-gamma 0.1 --epochs 12 --batch-size 32 --accum-steps 2'
              ' --num-workers 12 --seed 0 --tf32 --log-interval 50 --save-every 500'
              ' --epoch-ckpt-every 1 --run-type full --approved ' + APPROVED
              + ' --reviewed-commit ' + COMMIT)
SMOKE_FLAGS = ('--epochs 1 --max-train-batches 3 --max-test-batches 2 --batch-size 4'
               ' --num-workers 4 --save-every 0 --no-save')


def dry_run(mode, *extra):
    command = ['bash', LAUNCHER, mode, '--gpu', '1', '--reviewed-commit', COMMIT, '--dry-run']
    completed = subprocess.run(command + list(extra), cwd=REPO, capture_output=True, text=True,
                               env={**os.environ, 'PYTHONPATH': str(REPO),
                                    'XRIR_DATA_PATH': DATA_ROOT})
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.splitlines()


def test_launcher_is_valid_bash():
    assert subprocess.run(['bash', '-n', LAUNCHER], cwd=REPO).returncode == 0
    assert os.access(REPO / LAUNCHER, os.X_OK)


PREFLIGHT_ROOTS = (' --attempt-root ' + ROOT + ' --attempt-root ckpt/exp06/_smoke'
                   + ' --approved ' + APPROVED)


def test_full_dry_run_matches_the_plan_argv():
    lines = dry_run('full')
    assert ('RUN ' + PYTHON + ' tools/exp06_finalize.py preflight --mode full --gpu 1'
            ' --reviewed-commit ' + COMMIT + PREFLIGHT_ROOTS) in lines
    assert 'MKDIR ' + ATTEMPT in lines
    assert ENV_LINE in lines
    assert 'RUN nohup setsid ' + TRAIN_ARGV in lines
    log = 'worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_<UTC>_train_full.log'
    assert 'SINK cat >> ' + log in lines and 'PIDFILE ' + ATTEMPT + '/launch.pid' in lines
    assert 'MARKER EXP06_CHILD_EXIT <code> <iso> >> ' + log in lines
    assert all('tail -f' not in line for line in lines), 'no second reader of the log'
    assert ('RUN ' + PYTHON + ' tools/exp06_finalize.py --run-dir ' + ATTEMPT
            + ' --run-type full --log ' + log + ' --child-exit <code>'
            + ' --owner-pid <pid>') in lines
    assert 'PROMOTE ' + ROOT + '/final -> attempt_<UTC>' in lines
    assert 'ABORT ' + ATTEMPT + '_ABORTED_child_exit_<code>' in lines
    assert all('<reason>' not in line for line in lines), 'no placeholder abort reason'


def test_probe_dry_run_uses_the_bounded_recipe():
    lines = dry_run('probe')
    probe = [line for line in lines if 'exp06_smoke.py' in line]
    assert len(probe) == 1
    assert probe[0] == ('RUN nohup setsid ' + PYTHON + ' tools/exp06_smoke.py --receipt '
                        + ROOT + '/probe_<UTC>.json --run-type probe --provenance-out '
                        + ROOT + '/probe_<UTC>/provenance.json --approved ' + APPROVED
                        + ' --reviewed-commit ' + COMMIT
                        + ' --entry exp06_train --alarm-seconds 2400 --max-gb 46 --'
                        ' --backbone cylindrical_oriented --save-dir ' + ROOT + '/probe_<UTC>'
                        ' --epochs 1 --max-train-batches 200 --max-test-batches 20 --no-save'
                        ' --run-type probe --batch-size 32 --accum-steps 2 --tf32'
                        ' --num-workers 12 --decay-epochs 3 --log-interval 50')
    assert ENV_LINE in lines


def test_smoke_dry_run_lists_the_section_nine_commands():
    lines = dry_run('smoke')
    smokes = [line for line in lines if line.startswith('RUN ') and 'exp06_smoke.py' in line]
    assert len(smokes) == 4
    for index, (entry, name, backbone, target) in enumerate([
            ('trainer', 'trainer', 'simple', 't0'),
            ('exp06_train', 'exp06_train_t0', 'simple', 't0'),
            ('exp06_train', 'exp06_train_t1', 'cylindrical_oriented', 't1')]):
        assert smokes[index].endswith(
            '--receipt ckpt/exp06/_smoke/receipt_{}_<UTC>.json --run-type smoke'
            ' --provenance-out ckpt/exp06/_smoke/{}_<UTC>/provenance.json --approved {}'
            ' --reviewed-commit {} --entry {} --alarm-seconds 300 --max-gb 3 --'
            ' --backbone {} --save-dir ckpt/exp06/_smoke/{} {}'.format(
                name, name, APPROVED, COMMIT, entry, backbone, target, SMOKE_FLAGS)), smokes[index]
    assert smokes[3].endswith('--make-fixture ckpt/exp06/_smoke/fixture_cylor.pth')
    assert all('--tf32' not in line for line in smokes)


def test_finalize_mode_and_usage_errors():
    lines = dry_run('finalize', '--attempt', ATTEMPT, '--log', 'some.log', '--child-exit', '0')
    assert ('RUN ' + PYTHON + ' tools/exp06_finalize.py --run-dir ' + ATTEMPT
            + ' --run-type full --log some.log --child-exit 0') in lines
    assert all('--owner-pid' not in line for line in lines), 'recovery owns no live launch'
    for extra in (['invented'], ['full'], ['full', '--gpu', '1'], ['full', '--gpu', '1',
                  '--reviewed-commit', COMMIT, '--nonsense']):
        completed = subprocess.run(['bash', LAUNCHER] + extra, cwd=REPO, capture_output=True, text=True)
        assert completed.returncode == 2, completed.stdout


HARNESS = ('set -euo pipefail\n'
           'export EXP06_LAUNCH_LIB=1\n'
           'source tools/exp06_launch.sh\n'
           'own_launch {attempt}\n'
           'run_child {attempt} {log} {child}\n'
           'close_child {attempt} {log}\n'
           'echo "HARNESS_STATUS $CHILD_STATUS $CHILD_PID $$"\n')


def run_harness(tmp_path, script, child_args=''):
    """Drive the launcher's child lifecycle directly, without a mode or a preflight."""
    attempt = tmp_path / 'attempt'
    attempt.mkdir()
    log = tmp_path / 'child.log'
    log.write_text('')
    stub = tmp_path / 'stub.sh'
    stub.write_text(script)
    stub.chmod(0o755)
    completed = subprocess.run(
        ['bash', '-c', HARNESS.format(attempt=attempt, log=log,
                                      child=str(stub) + (' ' + child_args if child_args else ''))],
        cwd=REPO, capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': str(REPO)})
    return attempt, log, completed


def test_a_surviving_descendant_cannot_write_past_the_end_marker(tmp_path):
    """Blocker 4: the sink reaches EOF only when every holder of the pipe has exited."""
    attempt, log, completed = run_harness(tmp_path, '#!/usr/bin/env bash\n'
                                          'echo early\n( sleep 2; echo late ) &\nexit 0\n')
    assert completed.returncode == 0, completed.stderr
    lines = log.read_text().splitlines()
    assert 'late' in lines, 'the descendant output was lost: ' + repr(lines)
    assert lines[-1].startswith('EXP06_CHILD_EXIT 0 '), lines
    assert lines.index('late') < len(lines) - 1
    receipt = json.loads((attempt / 'child_exit.json').read_text())
    assert receipt['status'] == 0 and receipt['ended_at'] == lines[-1].split()[2]
    assert receipt['log_sha256_after_marker'] == hashlib.sha256(log.read_bytes()).hexdigest()
    assert not (attempt / 'child.pipe').exists()


def test_the_launcher_owns_launch_pid_while_the_child_has_its_own(tmp_path):
    """Should-fix 5: launch.pid names the launcher that drains and finalizes."""
    attempt, log, completed = run_harness(tmp_path, '#!/usr/bin/env bash\necho done\n')
    assert completed.returncode == 0, completed.stderr
    status, child, launcher = completed.stdout.split('HARNESS_STATUS ')[1].split()
    receipt = json.loads((attempt / 'child_exit.json').read_text())
    assert (attempt / 'child.pid').read_text().strip() == child == str(receipt['child_pid'])
    assert (attempt / 'launch.pid').read_text().strip() == launcher != child


SINK_HARNESS = ('set -euo pipefail\n'
                'export EXP06_LAUNCH_LIB=1\n'
                'source tools/exp06_launch.sh\n'
                'ulimit -c 0\n'
                'ulimit -f 1\n'          # the sink dies on SIGXFSZ after consuming output
                'run_child {attempt} {log} {child}\n'
                'close_child {attempt} {log}\n')


def test_a_failing_log_sink_aborts_the_launch(tmp_path):
    """Blocker 4: a sink that dies on a write error must publish nothing."""
    attempt = tmp_path / 'attempt'
    attempt.mkdir()
    log = tmp_path / 'child.log'
    log.write_text('')
    stub = tmp_path / 'stub.sh'
    stub.write_text('#!/usr/bin/env bash\nhead -c 8192 /dev/zero | tr "\\0" "x"\necho\nexit 0\n')
    stub.chmod(0o755)
    completed = subprocess.run(
        ['bash', '-c', SINK_HARNESS.format(attempt=attempt, log=log, child=stub)],
        cwd=REPO, capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': str(REPO)})
    assert completed.returncode == 3, completed.stdout + completed.stderr
    assert 'ABORT {}_ABORTED_sink_failed'.format(attempt) in completed.stdout
    aborted = Path(str(log) + '_ABORTED_sink_failed')
    assert aborted.is_file() and (tmp_path / 'attempt_ABORTED_sink_failed').is_dir()
    assert not log.exists() and not attempt.exists()
    assert 'EXP06_CHILD_EXIT' not in aborted.read_text()
    assert not (tmp_path / 'attempt_ABORTED_sink_failed/child_exit.json').exists()


def test_a_failing_child_reports_its_status_through_the_lifecycle(tmp_path):
    attempt, log, completed = run_harness(tmp_path,
                                          '#!/usr/bin/env bash\necho boom >&2\nexit 7\n')
    assert completed.returncode == 0, completed.stderr
    assert 'HARNESS_STATUS 7 ' in completed.stdout
    lines = log.read_text().splitlines()
    assert lines[0] == 'boom' and lines[-1].startswith('EXP06_CHILD_EXIT 7 ')
    assert json.loads((attempt / 'child_exit.json').read_text())['status'] == 7


def test_preflight_supports_the_recovery_mode_without_a_gpu_query(repo, fake_nvidia_smi):
    """Should-fix 5: recovery finalization is gated by the same commit and pid checks."""
    root, head = repo
    fake_nvidia_smi('3141\n')
    record = exp06_finalize.preflight('finalize', 1, head, repo=root)
    assert record['mode'] == 'finalize' and record['gpu_compute_apps'] is None
    with pytest.raises(ValueError, match='reviewed commit'):
        exp06_finalize.preflight('finalize', 1, 'a' * 40, repo=root)


def test_diagnostic_modes_run_the_same_child_lifecycle():
    """Should-fix 5: smoke and probe get a pid file, a closed log and a completion."""
    for mode, kind in (('smoke', 'smoke'), ('probe', 'probe')):
        lines = dry_run(mode)
        assert any(line.startswith('MKDIR ') for line in lines), mode
        assert any(line.startswith('SINK cat >> ') for line in lines), mode
        assert any(line.endswith('/launch.pid') and line.startswith('PIDFILE ') for line in lines)
        assert any(line.startswith('MARKER EXP06_CHILD_EXIT <code> <iso> >> ') for line in lines)
        assert all('nohup setsid' in line for line in lines
                   if line.startswith('RUN ') and '--entry' in line), mode
        finals = [line for line in lines if '--run-type ' + kind in line]
        assert finals and all('--receipt ' in line for line in finals), mode


def test_recovery_finalize_gates_promotes_and_aborts():
    lines = dry_run('finalize', '--attempt', ATTEMPT, '--log', 'some.log', '--child-exit', '0')
    assert ('RUN ' + PYTHON + ' tools/exp06_finalize.py preflight --mode finalize --gpu 1'
            ' --reviewed-commit ' + COMMIT + PREFLIGHT_ROOTS) in lines
    assert ('RUN ' + PYTHON + ' tools/exp06_finalize.py --run-dir ' + ATTEMPT
            + ' --run-type full --log some.log --child-exit 0') in lines
    assert 'PROMOTE ' + ROOT + '/final -> attempt_<UTC>' in lines
    assert 'ABORT ' + ATTEMPT + '_ABORTED_finalize_refused' in lines
    assert all('<reason>' not in line for line in lines), 'the recovery reason is recorded'


def test_a_diagnostic_directory_is_created_exclusively(tmp_path):
    """Should-fix 5: a repeated stamp must never reuse a diagnostic directory."""
    existing = tmp_path / 'probe_20260916T130000'
    existing.mkdir()
    script = ('set -euo pipefail\nexport EXP06_LAUNCH_LIB=1\nsource tools/exp06_launch.sh\n'
              'DRY=0\ndiagnostic probe {dir} {log} {receipt} --entry exp06_train\n').format(
                  dir=existing, log=tmp_path / 'probe.log', receipt=tmp_path / 'probe.json')
    completed = subprocess.run(['bash', '-c', script], cwd=REPO, capture_output=True, text=True,
                               env={**os.environ, 'PYTHONPATH': str(REPO)})
    assert completed.returncode != 0, completed.stdout
    assert 'File exists' in completed.stderr, completed.stderr
    assert not (existing / 'child.pid').exists() and not (existing / 'child_exit.json').exists()


def test_abort_renames_the_attempt_and_its_log(tmp_path):
    attempt = tmp_path / 'attempt_20260916T130000'
    attempt.mkdir()
    log = tmp_path / 'train.log'
    log.write_text('output\n')
    script = ('set -euo pipefail\nexport EXP06_LAUNCH_LIB=1\nsource tools/exp06_launch.sh\n'
              'DRY=0\nabort {a} {l} child_exit_7\n'.format(a=attempt, l=log))
    completed = subprocess.run(['bash', '-c', script], cwd=REPO, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / 'attempt_20260916T130000_ABORTED_child_exit_7').is_dir()
    assert (tmp_path / 'train.log_ABORTED_child_exit_7').is_file()
    assert not attempt.exists() and not log.exists()


def test_preflight_gates_on_the_approved_code_digests(repo, fake_nvidia_smi, tmp_path,
                                                      monkeypatch):
    """Finding 1: null approvals refuse a full launch; only a diagnostic may go exploratory."""
    root, head = repo
    approvals = tmp_path / 'approved_digests.json'
    approvals.write_bytes(exp06_profiles.TEMPLATE_PATH.read_bytes())
    with pytest.raises(ValueError, match='not approved'):
        exp06_finalize.preflight('full', 1, head, repo=root, approved=approvals)
    record = exp06_finalize.preflight('smoke', 0, head, repo=root, approved=approvals,
                                      exploratory=True)
    assert record['exploratory'] is True and record['approval_deviations']
    assert record['approved']['sha256'] == provenance.sha256_file(approvals)
    with pytest.raises(ValueError, match='exploratory'):
        exp06_finalize.preflight('full', 1, head, repo=root, approved=approvals,
                                 exploratory=True)
    with pytest.raises(ValueError, match='approvals'):
        exp06_finalize.preflight('full', 1, head, repo=root, approved=tmp_path / 'absent.json')


def test_preflight_admits_a_full_launch_whose_digests_match(repo, fake_nvidia_smi, tmp_path,
                                                            monkeypatch):
    root, head = repo
    approvals = tmp_path / 'approved_digests.json'
    value = json.loads(exp06_profiles.TEMPLATE_PATH.read_text())
    digests = {key: '{:064x}'.format(index)
               for index, key in enumerate(exp06_profiles.TRAINING_KEYS)}
    value['code'].update(digests)
    approvals.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    monkeypatch.setattr(exp06_profiles, 'compute_code_digests', lambda *a, **k: dict(digests))
    record = exp06_finalize.preflight('full', 1, head, repo=root, approved=approvals)
    assert record['approval_deviations'] == [] and record['exploratory'] is False
    assert record['approved'] == {'path': str(approvals.resolve()),
                                  'sha256': provenance.sha256_file(approvals)}
    digests['launch_sh'] = 'f' * 64
    with pytest.raises(ValueError, match='launch_sh'):
        exp06_finalize.preflight('full', 1, head, repo=root, approved=approvals)


def test_the_preflight_cli_takes_the_approvals_path(repo, fake_nvidia_smi, tmp_path):
    root, head = repo
    approvals = tmp_path / 'approved_digests.json'
    approvals.write_bytes(exp06_profiles.TEMPLATE_PATH.read_bytes())
    command = [sys.executable, 'tools/exp06_finalize.py', 'preflight', '--mode', 'full',
               '--gpu', '1', '--reviewed-commit', head, '--repo', str(root),
               '--approved', str(approvals)]
    env = {**os.environ, 'PYTHONPATH': str(REPO)}
    refused = subprocess.run(command, cwd=REPO, capture_output=True, text=True, env=env)
    assert refused.returncode == 2 and 'not approved' in refused.stderr
    diagnostic = subprocess.run(command[:4] + ['smoke'] + command[5:] + ['--exploratory'],
                                cwd=REPO, capture_output=True, text=True, env=env)
    assert diagnostic.returncode == 0, diagnostic.stderr
    record = json.loads(diagnostic.stdout.split('EXP06_PREFLIGHT_OK ')[1])
    assert record['exploratory'] is True and record['approval_deviations']


STUB = '''#!/usr/bin/env bash
case "$1" in
  tools/exp06_smoke.py) echo "smoke output"; exit "${SMOKE_STATUS:-0}" ;;
  tools/exp06_finalize.py)
    if [ "$2" = child-exit ]; then exec "$REAL_PYTHON" "$@"; fi
    echo "FINALIZE $*"; exit "${FINALIZE_STATUS:-0}" ;;
esac
exit 0
'''

DIAGNOSTIC_HARNESS = ('set -euo pipefail\n'
                      'export EXP06_LAUNCH_LIB=1\n'
                      'source tools/exp06_launch.sh\n'
                      'DRY=0\nPYTHON={stub}\nAPPROVED=a.json\nCOMMIT={commit}\nOWNER=$$\n'
                      'EXPLORATORY=0\n'
                      'diagnostic smoke {dir} {log} {receipt} --entry exp06_train'
                      ' --alarm-seconds 300 --max-gb 3 -- --no-save\n'
                      'echo LAUNCHER_CONTINUED\n')


def run_diagnostic(tmp_path, **env):
    stub = tmp_path / 'stub.sh'
    stub.write_text(STUB)
    stub.chmod(0o755)
    run = tmp_path / 'smoke_20260916T130000'
    log = tmp_path / 'smoke.log'
    script = DIAGNOSTIC_HARNESS.format(stub=stub, commit=COMMIT, dir=run, log=log,
                                       receipt=tmp_path / 'receipt.json')
    completed = subprocess.run(['bash', '-c', script], cwd=REPO, capture_output=True, text=True,
                               env={**os.environ, 'PYTHONPATH': str(REPO),
                                    'REAL_PYTHON': sys.executable, **env})
    return run, log, completed


def test_a_failed_diagnostic_child_aborts_and_propagates(tmp_path):
    """Finding 5: the launcher returned the finalizer's status, so a failed rung passed."""
    run, log, completed = run_diagnostic(tmp_path, SMOKE_STATUS='3')
    assert completed.returncode == 3, completed.stdout + completed.stderr
    assert 'LAUNCHER_CONTINUED' not in completed.stdout
    assert '--child-exit 3' in completed.stdout, 'the failure is still recorded'
    assert Path(str(run) + '_ABORTED_child_failed_3').is_dir()
    assert Path(str(log) + '_ABORTED_child_failed_3').is_file()
    assert not run.exists() and not log.exists()


def test_a_refused_diagnostic_finalisation_aborts_with_two(tmp_path):
    run, log, completed = run_diagnostic(tmp_path, FINALIZE_STATUS='2')
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert 'LAUNCHER_CONTINUED' not in completed.stdout
    assert Path(str(run) + '_ABORTED_finalize_refused').is_dir()
    assert Path(str(log) + '_ABORTED_finalize_refused').is_file()


def test_a_passing_diagnostic_leaves_the_launcher_running(tmp_path):
    run, log, completed = run_diagnostic(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'LAUNCHER_CONTINUED' in completed.stdout
    assert run.is_dir() and log.is_file()
    assert log.read_text().splitlines()[-1].startswith('EXP06_CHILD_EXIT 0 ')
