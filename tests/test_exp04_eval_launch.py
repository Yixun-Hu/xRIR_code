"""Execution-bound evaluation finalisation, using tiny CPU child processes."""
import hashlib
import json
import sys
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
    fields = {"repo": str(tmp_path), "checkpoint": str(paths["checkpoint"]),
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
    return [sys.executable, "-c", "from pathlib import Path; " +
            ("; ".join("Path(%r).write_text('{}')" % str(Path(out_dir) / name)
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
