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
                           run_label="tiny", data_root=str(tmp_path))
    paths = {key: tmp_path / key for key in ("checkpoint", "reference", "source", "data")}
    for path in paths.values():
        path.write_text("original")
    fields = {"repo": str(tmp_path), "conditions": "P", "n_samples": 1,
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


def test_completion_binds_manifest_closed_log_and_outputs(attempt):
    args, _, fields = attempt
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
    assert datetime.datetime.fromisoformat(completion['started_at']) <= datetime.datetime.fromisoformat(completion['ended_at'])


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
    monkeypatch.setattr(p, "git_state", lambda repo: {"HEAD": "head", "dirty": False})
    monkeypatch.setattr(launcher, "_capture_environment", lambda repo, root: {"python": "3.8"})
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
    for key, value in vars(child).items():
        if key not in ("eval_manifest", "out_dir"):
            assert fields["manifest_path" if key == "manifest" else key] == value


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
