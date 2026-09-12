"""Execution-bound evaluation gates and parity with the frozen exp_03 evaluator."""
import json
from pathlib import Path

import pytest

from tools import exp04_eval as subject
from tools import provenance
from tools.reference_manifest import manifest_hash


@pytest.fixture
def bound_run(tmp_path, monkeypatch):
    checkpoint, reference = tmp_path / "checkpoint.pth", tmp_path / "reference.json"
    checkpoint.write_bytes(b"weights")
    refs = {"seed": 42, "num_shot": 8, "entries": []}
    reference.write_text(json.dumps(refs))
    output = tmp_path / "run"
    output.mkdir()
    path = output / "eval_manifest.json"
    args = subject.parse_args([
        "--backbone", "simple", "--checkpoint", str(checkpoint), "--manifest",
        str(reference), "--manifest-hash", manifest_hash(refs), "--out-dir", str(output),
        "--eval-manifest", str(path)])
    record = [{"path": "eval_yaw_rotation.py", "reviewed_blob_sha256": "source",
               "working_tree_sha256": "source", "commits_after_reviewed": []}]
    monkeypatch.setattr(provenance, "source_closure", lambda *a: ["eval_yaw_rotation.py"])
    monkeypatch.setattr(provenance, "closure_record", lambda *a: (record, "closure"))
    fields = {key: value for key, value in vars(args).items()
              if key not in ("eval_manifest", "out_dir", "manifest")}
    fields.update(checkpoint_sha256=provenance.sha256_file(checkpoint),
                  manifest_path=str(reference), manifest_file_sha256=provenance.sha256_file(reference),
                  manifest_seed=42, num_shot=8, batch_canonical=True,
                  data_root=subject.BASE_DATA_PATH, repo=str(Path(__file__).resolve().parents[1]),
                  reviewed_commit="reviewed", evaluator_closure={"files": record, "sha256": "closure"})
    path.write_text(json.dumps(fields))
    return args, fields, path


def test_manifest_roundtrip_and_digest(bound_run):
    args, fields, path = bound_run
    got, digest, reference = subject.validate_manifest(args)
    assert got == fields and digest == provenance.sha256_file(path)
    assert reference["num_shot"] == 8


@pytest.mark.parametrize("field", [
    "backbone", "checkpoint", "checkpoint_sha256", "manifest_path", "manifest_hash",
    "manifest_file_sha256", "gl_seed", "num_shot", "manifest_seed", "yaw_cols",
    "acoustic_cols", "e_acoustic_cols", "conditions", "batch_size", "batch_canonical",
    "tf32", "max_samples", "data_root", "num_workers", "threads", "log_interval",
    "decomposition_batches", "evaluator_closure"])
def test_each_manifest_field_mismatch_is_refused(bound_run, field):
    args, fields, path = bound_run
    fields[field] = {"files": [], "sha256": "different"} if field == "evaluator_closure" else "different"
    path.write_text(json.dumps(fields))
    with pytest.raises(ValueError, match=field):
        subject.validate_manifest(args)


def test_output_directory_must_contain_only_the_manifest(bound_run):
    args, _, path = bound_run
    (path.parent / "stale.json").write_text("{}")
    with pytest.raises(ValueError, match="out_dir"):
        subject.validate_manifest(args)


def test_p_mode_rejects_e_acoustic_columns(bound_run):
    args, fields, path = bound_run
    args.conditions = fields["conditions"] = "P"
    path.write_text(json.dumps(fields))
    with pytest.raises(ValueError, match="e_acoustic_cols"):
        subject.validate_manifest(args)


@pytest.mark.parametrize("key,value", [("working_tree_sha256", "changed"), ("commits_after_reviewed", ["later"])])
def test_modified_evaluator_source_is_refused(bound_run, key, value):
    args, fields, _ = bound_run
    fields["evaluator_closure"]["files"][0][key] = value
    with pytest.raises(ValueError, match="evaluator_closure"):
        subject.validate_manifest(args)
