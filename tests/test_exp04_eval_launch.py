"""Execution-bound evaluation finalisation, using tiny CPU child processes."""
import hashlib
import datetime
import json
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import exp04_eval_launch as launcher
from tools import provenance as p


@pytest.fixture
def attempt(tmp_path):
    args = SimpleNamespace(out_dir=str(tmp_path / "run"), log_dir=str(tmp_path / "logs"),
                           run_label="tiny", data_root=str(tmp_path), gpu='1')
    paths = {key: tmp_path / key for key in ("checkpoint", "reference", "source", "data")}
    for path in paths.values():
        path.write_text("original")
    fields = {"repo": str(tmp_path), "conditions": "P", "n_samples": 1,
              "confirmatory": True, "allow_dirty_used": False,
              "source_closures": {}, "mutable_inputs": {},
              "checkpoint": str(paths["checkpoint"]),
              "checkpoint_sha256": p.sha256_file(paths["checkpoint"]),
              "manifest_path": str(paths["reference"]),
              "manifest_file_sha256": p.sha256_file(paths["reference"]),
              "evaluator_closure": {"files": [{"path": "source",
                  "working_tree_sha256": p.sha256_file(paths["source"])}]},
              "data_identity": {"data_root": str(tmp_path), "inventory": [
                  {"path": "data", "sha256": p.sha256_file(paths["data"])}]}}
    fields["data_identity"]["inventory_sha256"] = hashlib.sha256(json.dumps(
        [["data", p.sha256_file(paths["data"])]]).encode()).hexdigest()
    return args, paths, fields


def child_code(out_dir, extra="", outputs=True):
    return [sys.executable, "-c", "from pathlib import Path; import os, json, hashlib; " +
            "assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'; " +
            "assert os.environ['XRIR_DATA_PATH'] == %r; " % str(Path(out_dir).parent) +
            "manifest = Path(%r).read_bytes(); " % str(Path(out_dir) / 'eval_manifest.json') +
            "meta = dict(json.loads(manifest), eval_manifest_sha256=hashlib.sha256(manifest).hexdigest()); " +
            ("; ".join("Path(%r).write_text(json.dumps(dict(meta=meta, query=['q'])))" % str(Path(out_dir) / name)
                       for name in launcher.OUTPUTS) + "; " if outputs else "") +
            "print('closed log tail', flush=True); " + extra]


def test_existing_directory_refused_before_collecting_inputs(attempt):
    args, _, _ = attempt
    Path(args.out_dir).mkdir()
    with pytest.raises(FileExistsError):
        launcher.execute_run(args, [], lambda: pytest.fail("input work ran"), args.data_root)
    assert not Path(args.log_dir).exists()


@pytest.mark.parametrize("failure", ["child", "missing", "checkpoint", "reference",
                                      "source", "data", "eval_manifest"])
def test_failed_or_mutated_run_has_no_completion(attempt, failure):
    args, paths, fields = attempt
    extra = "raise SystemExit(3)" if failure == "child" else ""
    if failure in paths or failure == "eval_manifest":
        changed = paths.get(failure, Path(args.out_dir) / "eval_manifest.json")
        extra = "Path(%r).write_text('changed')" % str(changed)
    command = child_code(args.out_dir, extra, outputs=failure != "missing")
    with pytest.raises((RuntimeError, ValueError)):
        launcher.execute_run(args, command, lambda: fields, args.data_root)
    assert not Path(args.out_dir).exists()
    aborted, = Path(args.data_root).glob("run_ABORTED_*")
    assert not (aborted / "completion.json").exists()
    assert (aborted / "eval_manifest.json").exists()
    log, = Path(args.log_dir).glob('*.log')
    assert log.stem.endswith(aborted.name[len('run'):])


@pytest.mark.parametrize('allow_dirty', [False, True])
def test_completion_binds_manifest_closed_log_and_outputs(attempt, allow_dirty):
    args, _, fields = attempt
    fields.update(confirmatory=not allow_dirty, allow_dirty_used=allow_dirty)
    completion = launcher.execute_run(args, child_code(args.out_dir), lambda: fields,
                                      args.data_root)
    run = Path(args.out_dir)
    assert json.loads((run / "completion.json").read_text()) == completion
    assert completion["eval_manifest_sha256"] == p.sha256_file(run / "eval_manifest.json")
    log = Path(completion["log"]["path"])
    assert log.parent == Path(args.log_dir)
    assert log.read_text() == "closed log tail\n"
    assert completion["log"]["sha256"] == p.sha256_file(log)
    assert completion["outputs"] == {name: p.sha256_file(run / name)
                                     for name in launcher.OUTPUTS}
    assert completion['schema_version'] == json.loads((run / 'eval_manifest.json').read_text())['schema_version'] == 1
    assert completion['directory_listing'] == sorted(['eval_manifest.json', *launcher.OUTPUTS])
    assert completion['child_exit_status'] == 0
    assert completion['confirmatory'] is (not allow_dirty)
    assert completion['allow_dirty_used'] is allow_dirty
    assert datetime.datetime.fromisoformat(completion['started_at']) <= datetime.datetime.fromisoformat(completion['ended_at'])


@pytest.mark.parametrize('boundary', ['revalidate', 'write_completion'])
def test_signal_during_tail_certifies_then_exits_one(attempt, boundary):
    args, _, fields = attempt
    code = '''
import os, signal
from types import SimpleNamespace
from tools import exp04_eval_launch as launch, provenance as p
signal.signal(signal.SIGTERM, signal.SIG_DFL)
original = getattr(p, %r)
def terminate(*a, **kw):
    result = original(*a, **kw)
    os.kill(os.getpid(), signal.SIGTERM)
    return result
setattr(p, %r, terminate)
launch.execute_run(SimpleNamespace(**%r), %r, lambda: %r, %r)
''' % (boundary, boundary, vars(args), child_code(args.out_dir), fields, args.data_root)
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and 'LauncherTerminated: SIGTERM' in result.stderr
    run = Path(args.out_dir)
    assert 'CERTIFIED ' + str(run) in result.stdout.splitlines()
    assert (run / 'completion.json').is_file() and not (run / 'abort.json').exists()
    assert not list(run.parent.glob('*_ABORTED_*'))


@pytest.mark.parametrize('name,fault', [(name, fault) for name in launcher.OUTPUTS
    for fault in ['extra', 'directory', 'json', 'digest', 'conditions', 'n_samples', 'query', 'meta']
    if fault != 'query' or name == 'per_sample_yaw.json'])
def test_invalid_outputs_are_not_finalized(attempt, name, fault):
    args, _, fields = attempt
    target = str(Path(args.out_dir) / name)
    if fault in ('extra', 'directory'):
        extra = "Path(%r).%s" % (str(Path(args.out_dir) / '.rogue'),
                                  "mkdir()" if fault == 'directory' else "write_text('x')")
    elif fault == 'json':
        extra = "Path(%r).write_text('{')" % target
    else:
        extra = "payload = json.loads(Path(%r).read_text()); " % target
        extra += {"digest": "payload['meta']['eval_manifest_sha256'] = 'wrong'; ",
                  "conditions": "payload['meta']['conditions'] = 'PE'; ",
                  "n_samples": "payload['meta']['n_samples'] = 2; ",
                  "query": "payload['query'] = []; ", "meta": "payload.pop('meta'); "}[fault]
        extra += "Path(%r).write_text(json.dumps(payload))" % target
    with pytest.raises((RuntimeError, ValueError, KeyError)):
        launcher.execute_run(args, child_code(args.out_dir, extra), lambda: fields, args.data_root)
    aborted = Path(args.out_dir + '_ABORTED_output_invalid')
    assert aborted.is_dir() and not (aborted / 'completion.json').exists()


@pytest.fixture
def launch_args(attempt, monkeypatch):
    args, paths, _ = attempt
    from tools.reference_manifest import manifest_hash
    reference = {"seed": 42, "num_shot": 8, "entries": [{"query": "room/q.wav"}]}
    paths["reference"].write_text(json.dumps(reference))
    args = launcher.parse_args(["--checkpoint", str(paths["checkpoint"]), "--backbone", "simple",
        "--manifest", str(paths["reference"]), "--manifest-hash", manifest_hash(reference),
        "--out-dir", args.out_dir, "--log-dir", args.log_dir, "--data-root", args.data_root,
        "--run-label", "fixture", "--reviewed-commit", "HEAD", "--num-shot", "8",
        "--conditions", "P", "--yaw-cols", "0", "32", "--acoustic-cols", "0",
        "--e-acoustic-cols", "--max-samples", "16"])
    monkeypatch.setattr(p, "source_closure", lambda module, repo: [module + ".py"])
    monkeypatch.setattr(p, "closure_record", lambda files, commit, repo: (
        [{"path": path, "reviewed_blob_sha256": "sha", "working_tree_sha256": "sha",
          "commits_after_reviewed": []} for path in files], "closure-sha"))
    monkeypatch.setattr(p, "data_identity", lambda path, root: {"inventory_sha256": "data-sha"})
    monkeypatch.setattr(p, "git_state", lambda repo: {"dirty_outside_worklog": False})
    monkeypatch.setattr(launcher, "_capture_environment", lambda *a: {"python": "3.8"})
    return args


def test_manifest_builder_binds_exact_child_cli_and_all_closures(launch_args):
    args = launch_args
    repo = Path(launcher.__file__).resolve().parents[1]
    command = launcher.child_command(args, repo)
    fields = launcher.build_fields(args, command, repo)
    from tools.exp04_eval import parse_args
    child = parse_args(command[2:])
    assert fields["command"] == command
    assert fields['schema_version'] == 1
    assert fields['reviewed_commit'] == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    assert fields["num_shot"] == 8 and fields["manifest_seed"] == 42
    assert fields["conditions"] == child.conditions == "P"
    assert fields["checkpoint_sha256"] == p.sha256_file(args.checkpoint)
    assert fields["manifest_file_sha256"] == p.sha256_file(args.manifest)
    assert fields["max_samples"] == 16
    assert fields["n_samples"] == 1 and fields["batch_canonical"] is True
    assert fields["data_root"] == str(Path(args.data_root).resolve())
    assert set(fields["source_closures"]) == {"entrypoint", "writer"}
    assert fields["evaluator_closure"]["files"][0]["path"] == "eval_yaw_rotation.py"
    assert child.eval_manifest == str(Path(args.out_dir) / "eval_manifest.json")
    assert "--reviewed-commit" not in command
    assert args.gpu == '1' and args.bind_input == [] and args.allow_dirty is False
    for key, value in vars(child).items():
        if key not in ("eval_manifest", "out_dir"):
            assert fields["manifest_path" if key == "manifest" else key] == value


@pytest.mark.parametrize('changed', [None, 'control_args', 'train_manifest', 'train_completion', 'probe_receipt'])
def test_bound_training_inputs_are_revalidated(launch_args, attempt, changed):
    args, paths, fields = attempt
    fields['manifest_file_sha256'] = p.sha256_file(paths['reference'])
    repo = Path(launcher.__file__).resolve().parents[1]
    cli = launcher.child_command(launch_args, repo)[2:]
    cli = cli[:cli.index('--eval-manifest')] + cli[cli.index('--eval-manifest') + 2:]
    bindings = [item for name in ('control_args', 'train_manifest', 'train_completion', 'probe_receipt')
                for item in ('--bind-input', name + '=' + str(paths['data']))]
    parsed = launcher.parse_args(cli + ['--log-dir', args.log_dir, '--data-root', args.data_root,
        '--run-label', 'binding', '--reviewed-commit', 'HEAD', '--num-shot', '8', *bindings])
    fields['mutable_inputs'] = launcher.build_fields(parsed, launcher.child_command(parsed, repo), repo)['mutable_inputs']
    assert set(fields['mutable_inputs']) == {'control_args', 'train_manifest', 'train_completion', 'probe_receipt'}
    extra = "Path(%r).write_text('changed')" % fields['mutable_inputs'][changed]['path'] if changed else ''
    if changed:
        with pytest.raises(ValueError, match=changed):
            launcher.execute_run(args, child_code(args.out_dir, extra), lambda: fields, args.data_root)
    else:
        launcher.execute_run(args, child_code(args.out_dir), lambda: fields, args.data_root)


@pytest.mark.parametrize('key', ['checkpoint_sha256', 'data_identity', 'source_closures', 'evaluator_closure'])
def test_finalization_requires_inputs(attempt, key):
    args, _, fields = attempt
    fields.pop(key)
    with pytest.raises(ValueError, match=key):
        launcher.execute_run(args, child_code(args.out_dir), lambda: fields, args.data_root)


@pytest.mark.parametrize('samples,allow', [(0, False), (0, True), (16, False), (6336, False),
                                        (6337, False), (100000, False)])
def test_dirty_evaluation_gate(launch_args, monkeypatch, capsys, samples, allow):
    launch_args.max_samples, launch_args.allow_dirty = samples, allow
    monkeypatch.setattr(launcher.evaluator.yaw, 'load_checked_manifest',
                        lambda *a: {'num_shot': 8, 'seed': 42, 'entries': [None] * 6337})
    monkeypatch.setattr(p, 'git_state', lambda repo: {'dirty_outside_worklog': True})
    repo = Path(launcher.__file__).resolve().parents[1]
    if (samples == 0 or samples >= 6337) and not allow:
        with pytest.raises(ValueError, match='dirty_outside_worklog'):
            launcher.build_fields(launch_args, launcher.child_command(launch_args, repo), repo)
    else:
        fields = launcher.build_fields(launch_args, launcher.child_command(launch_args, repo), repo)
        assert fields['allow_dirty_used'] is allow and fields['confirmatory'] is False
        assert 'WARNING' in capsys.readouterr().out


def test_child_gpu_and_pythonpath(attempt, monkeypatch):
    args, _, fields = attempt
    args.gpu = '0'
    monkeypatch.setenv('PYTHONPATH', ':old::path:repo:old:')
    assert launcher.child_environment('repo', 'data', '0')['PYTHONPATH'] == 'repo:old:path'
    command = child_code(args.out_dir)
    command[-1] = command[-1].replace("== '1'", "== '0'")
    launcher.execute_run(args, command, lambda: fields, args.data_root)


@pytest.mark.parametrize('samples,allow,confirmatory', [(0, False, True), (0, True, False), (1, False, False)])
def test_confirmatory_manifest_flags(launch_args, samples, allow, confirmatory):
    launch_args.max_samples, launch_args.allow_dirty = samples, allow
    repo = Path(launcher.__file__).resolve().parents[1]
    fields = launcher.build_fields(launch_args, launcher.child_command(launch_args, repo), repo)
    assert fields['confirmatory'] is confirmatory and fields['allow_dirty_used'] is False
    assert 'allow_dirty' not in fields


@pytest.mark.parametrize('name', ['unknown', 'control-args', ''])
def test_unknown_binding_identifier_refused(launch_args, name):
    launch_args.bind_input = [name + '=' + launch_args.checkpoint]
    repo = Path(launcher.__file__).resolve().parents[1]
    with pytest.raises(ValueError, match='bind-input'):
        launcher.build_fields(launch_args, launcher.child_command(launch_args, repo), repo)


def test_redirected_launcher_lines_survive_child_tee(attempt, tmp_path):
    args, _, fields = attempt
    code = ("from types import SimpleNamespace; from tools import exp04_eval_launch as l; "
            "print('launcher before', flush=True); l.execute_run(SimpleNamespace(**%r), %r, "
            "lambda: %r, %r); print('launcher after', flush=True)" %
            (vars(args), child_code(args.out_dir), fields, args.data_root))
    console = tmp_path / 'console.log'
    with console.open('w') as stream:
        subprocess.run([sys.executable, '-c', code], stdout=stream, stderr=subprocess.STDOUT, check=True)
    assert console.read_text().splitlines() == ['launcher before', 'closed log tail', 'launcher after']


@pytest.mark.parametrize('collision', ['original', 'aborted'])
def test_foreign_log_is_preserved(attempt, monkeypatch, collision):
    args, _, fields = attempt
    now = datetime.datetime.now().astimezone()
    monkeypatch.setattr(launcher, 'datetime', SimpleNamespace(datetime=SimpleNamespace(now=lambda: now)))
    logs = Path(args.log_dir)
    logs.mkdir()
    stem = args.run_label + '_' + now.strftime('%Y-%m-%d_%H:%M:%S.%f')
    foreign = logs / (stem + ('_ABORTED_child_failed' if collision == 'aborted' else '') + '.log')
    foreign.write_text('foreign log')
    with pytest.raises((FileExistsError, RuntimeError)):
        launcher.execute_run(args, child_code(args.out_dir, 'raise SystemExit(3)'), lambda: fields, args.data_root)
    assert foreign.read_text() == 'foreign log'
    if collision == 'original':
        assert Path(args.out_dir + '_ABORTED_log_exists').is_dir()
        assert list(logs.iterdir()) == [foreign]
    else:
        assert len(list(logs.glob('*_ABORTED_child_failed*.log'))) == 2


@pytest.mark.parametrize('ref', ['HEAD', 'main', '07f3bc9', 'no-such-commit'])
def test_reviewed_ref_resolution(launch_args, ref):
    repo = Path(launcher.__file__).resolve().parents[1]
    launch_args.reviewed_commit = ref
    if ref == 'no-such-commit':
        with pytest.raises(subprocess.CalledProcessError):
            launcher.build_fields(launch_args, launcher.child_command(launch_args, repo), repo)
    else:
        fields = launcher.build_fields(launch_args, launcher.child_command(launch_args, repo), repo)
        assert fields['reviewed_commit'] == subprocess.check_output(
            ['git', 'rev-parse', '--verify', ref + '^{commit}'], cwd=repo, text=True).strip()


@pytest.mark.parametrize("failure", ["num_shot", "unreviewed", "changed", "later_commit"])
def test_manifest_builder_refuses_unbound_input(launch_args, monkeypatch, failure):
    args = launch_args
    if failure == "num_shot":
        args.num_shot = 1
    else:
        record = {"path": "source", "reviewed_blob_sha256": "sha",
                  "working_tree_sha256": "sha", "commits_after_reviewed": []}
        record[{"unreviewed": "reviewed_blob_sha256", "changed": "working_tree_sha256",
                "later_commit": "commits_after_reviewed"}[failure]] = None if failure == "unreviewed" else "changed"
        monkeypatch.setattr(p, "closure_record", lambda *args: ([record], "digest"))
    repo = Path(launcher.__file__).resolve().parents[1]
    with pytest.raises(ValueError):
        launcher.build_fields(args, launcher.child_command(args, repo), repo)


def test_completion_hash_failure_is_output_invalid(attempt, monkeypatch):
    args, _, fields = attempt
    sha256 = p.sha256_file
    def fail(path):
        if Path(path).suffix == '.log':
            raise OSError('log hash failed')
        return sha256(path)
    monkeypatch.setattr(p, 'sha256_file', fail)
    with pytest.raises(OSError, match='log hash failed'):
        launcher.execute_run(args, child_code(args.out_dir), lambda: fields, args.data_root)
    assert Path(args.out_dir + '_ABORTED_output_invalid').is_dir()
