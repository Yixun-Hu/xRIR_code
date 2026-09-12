"""Execution-bound evaluation gates and parity with the frozen exp_03 evaluator."""
import json
from pathlib import Path

import pytest

from tools import exp04_eval as subject
from tools import provenance
from tools.reference_manifest import manifest_hash


@pytest.fixture
def protocol_run(tmp_path):
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
    fields = {key: value for key, value in vars(args).items()
              if key not in ("eval_manifest", "out_dir", "manifest")}
    fields.update(schema_version=1, checkpoint_sha256=provenance.sha256_file(checkpoint),
                  manifest_path=str(reference), manifest_file_sha256=provenance.sha256_file(reference),
                  manifest_seed=42, num_shot=8, batch_canonical=True,
                  data_root=subject.BASE_DATA_PATH, repo=str(Path(__file__).resolve().parents[1]),
                  reviewed_commit="a" * 40, evaluator_closure={"files": record, "sha256": "closure"})
    path.write_text(json.dumps(fields))
    return args, fields, path


@pytest.fixture
def bound_run(protocol_run, monkeypatch):
    record = protocol_run[1]['evaluator_closure']['files']
    monkeypatch.setattr(provenance, "source_closure", lambda *a: ["eval_yaw_rotation.py"])
    monkeypatch.setattr(provenance, "closure_record", lambda *a: (record, "closure"))
    return protocol_run


def test_cpu_manifest_uses_real_closure_and_git(protocol_run, tmp_path, monkeypatch):
    import subprocess
    args, fields, path = protocol_run
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'eval_yaw_rotation.py').write_text('import helper\n')
    (repo / 'helper.py').write_text('VALUE = 1\n')
    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=repo, text=True).strip()
    git('init', '-q')
    git('add', 'eval_yaw_rotation.py', 'helper.py')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'fixture')
    fields.update(repo=str(repo), reviewed_commit=git('rev-parse', 'HEAD'))
    monkeypatch.setattr(subject, '__file__', str(repo / 'tools/exp04_eval.py'))
    records, digest = provenance.closure_record(['eval_yaw_rotation.py', 'helper.py'], fields['reviewed_commit'], repo)
    fields['evaluator_closure'] = {'files': records, 'sha256': digest}
    path.write_text(json.dumps(fields))
    assert subject.validate_manifest(args)[0] == fields
    for declared in ({'files': records, 'sha256': 'tampered'},
                     {'files': records[:1], 'sha256': digest}):
        path.write_text(json.dumps(dict(fields, evaluator_closure=declared)))
        with pytest.raises(ValueError, match='evaluator_closure'):
            subject.validate_manifest(args)
    path.write_text(json.dumps(fields))
    (repo / 'helper.py').write_text('VALUE = 2\n')
    with pytest.raises(ValueError, match='evaluator_closure'):
        subject.validate_manifest(args)


def test_manifest_roundtrip_and_digest(bound_run):
    args, fields, path = bound_run
    got, digest, reference = subject.validate_manifest(args)
    assert got == fields and digest == provenance.sha256_file(path)
    assert reference["num_shot"] == 8


@pytest.mark.parametrize('commit', ['HEAD', 'main', 'fce0035', 'A' * 40, 'a' * 39, None, 123])
def test_reviewed_commit_must_be_full_lowercase_sha(bound_run, commit):
    args, fields, path = bound_run
    fields['reviewed_commit'] = commit
    path.write_text(json.dumps(fields))
    with pytest.raises(ValueError, match='reviewed_commit'):
        subject.validate_manifest(args)


@pytest.mark.parametrize("field", [
    "backbone", "checkpoint", "checkpoint_sha256", "manifest_path", "manifest_hash",
    "manifest_file_sha256", "gl_seed", "num_shot", "manifest_seed", "yaw_cols",
    "acoustic_cols", "e_acoustic_cols", "conditions", "batch_size", "batch_canonical",
    "tf32", "max_samples", "data_root", "num_workers", "threads", "log_interval",
    "decomposition_batches", "evaluator_closure", "repo", "schema_version"])
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


@pytest.mark.parametrize("key", ["reviewed_blob_sha256", "working_tree_sha256", "commits_after_reviewed"])
def test_declared_source_record_cannot_be_tampered(bound_run, key):
    args, _, path = bound_run
    fields = json.loads(path.read_text())
    fields["evaluator_closure"]["files"][0][key] = "tampered"
    path.write_text(json.dumps(fields))
    with pytest.raises(ValueError, match="evaluator_closure"):
        subject.validate_manifest(args)


def test_invalid_reviewed_commit_names_the_field(bound_run, monkeypatch):
    import subprocess

    def invalid(*args):
        raise subprocess.CalledProcessError(128, ["git", "log", "invalid..HEAD"])

    args, _, _ = bound_run
    monkeypatch.setattr(provenance, "closure_record", invalid)
    with pytest.raises(ValueError, match="reviewed_commit"):
        subject.validate_manifest(args)


def test_angle_element_types_are_bound(bound_run):
    args, fields, path = bound_run
    fields["yaw_cols"] = [False] + fields["yaw_cols"][1:]
    path.write_text(json.dumps(fields))
    with pytest.raises(ValueError, match="yaw_cols"):
        subject.validate_manifest(args)


def test_eval_manifest_must_be_inside_the_existing_output_directory(bound_run):
    args, _, path = bound_run
    for name in ("empty", "absent"):
        out = path.parent.parent / name
        if name == "empty":
            out.mkdir()
        args.out_dir = str(out)
        with pytest.raises(ValueError, match="out_dir"):
            subject.validate_manifest(args)


def test_p_only_reuses_p_numerics_and_omits_e_forwards(monkeypatch):
    import numpy as np
    import torch
    from eval_yaw_rotation import evaluate_batch

    class Model(torch.nn.Module):
        calls = 0

        def shift_and_align(self, refs, src, locs):
            return refs + src[:, 0, None, None]

        def forward(self, depth, refs, src, locs, target):
            self.calls += 1
            value = self.shift_and_align(refs, src, locs).mean((1, 2)) + src[:, 0]
            out = value[:, None, None, None].expand(-1, 3, 5, 1)
            return out, torch.ones(out.shape[0], 1, 3, 5)

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(subject.yaw, "acoustic_metrics_batch", lambda out, *a:
                        {"edt": out[:, 0, 0, 0].numpy().astype(np.float64)})
    batch = (torch.zeros(2, 3), torch.ones(2, 3), torch.ones(2, 3, 1, 512),
             torch.ones(2, 1, 15), torch.ones(2, 8, 15), torch.ones(2, 8, 3), ["a", "b"])
    model = Model()
    keys, expected, flips = evaluate_batch(model, batch, None, [0, 32], [0, 32], batch_size=4)
    assert model.calls == 5
    model.calls = 0
    got_keys, actual, got_flips = subject.evaluate_p_batch(
        model, batch, None, [0, 32], [0, 32], batch_size=4)
    assert model.calls == 3 and got_keys == keys and got_flips == flips
    assert set(actual) == {("P", 0), ("P", 32)}
    for cell, metrics in actual.items():
        for name, values in metrics.items():
            np.testing.assert_array_equal(values, expected[cell][name])


def test_run_validates_before_model_construction(bound_run, monkeypatch):
    args, fields, path = bound_run
    fields["batch_size"] = 999
    path.write_text(json.dumps(fields))
    monkeypatch.setattr(subject.yaw, "build_xrir", lambda *a: pytest.fail("model built before gate"))
    with pytest.raises(ValueError, match="batch_size"):
        subject.run_exp04(args)


def test_real_16_query_k8_original_pe_and_p_parity(tmp_path):
    import os
    import subprocess
    import sys
    import torch

    repo = Path(__file__).resolve().parents[1]
    checkpoint = repo / "ckpt/xRIR_simple_8_shot/epoch_12.pth"
    reference = repo / "ckpt/yaw_rotation/reference_manifest.json"
    if not torch.cuda.is_available() or not checkpoint.exists() or not reference.exists():
        pytest.skip("requires CUDA and the local K8 checkpoint/reference assets")
    refs = json.loads(reference.read_text())
    common = ["--backbone", "simple", "--checkpoint", str(checkpoint), "--manifest",
              str(reference), "--manifest-hash", manifest_hash(refs), "--max-samples", "16",
              "--yaw-cols", "0", "32", "--acoustic-cols", "0", "32", "--num-workers", "0",
              "--threads", "2", "--decomposition-batches", "0"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", PYTHONPATH=str(repo))
    baseline = tmp_path / "original"
    subprocess.run([sys.executable, str(repo / "eval_yaw_rotation.py"), *common,
                    "--e-acoustic-cols", "32", "--out-dir", str(baseline)],
                   cwd=str(repo), env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo)).decode().strip()
    records, digest = provenance.closure_record(provenance.source_closure("eval_yaw_rotation", repo), commit, repo)
    for condition in ("PE", "P"):
        output = tmp_path / condition
        output.mkdir()
        path = output / "eval_manifest.json"
        argv = common + ["--e-acoustic-cols"] + (["32"] if condition == "PE" else [])
        argv += ["--out-dir", str(output), "--eval-manifest", str(path), "--conditions", condition]
        args = subject.parse_args(argv)
        fields = {k: v for k, v in vars(args).items() if k not in ("eval_manifest", "manifest", "out_dir")}
        fields.update(schema_version=1, checkpoint_sha256=provenance.sha256_file(checkpoint), manifest_path=str(reference),
                      manifest_file_sha256=provenance.sha256_file(reference), manifest_seed=refs["seed"],
                      num_shot=8, batch_canonical=True, data_root=subject.BASE_DATA_PATH, repo=str(repo),
                      reviewed_commit=commit, evaluator_closure={"files": records, "sha256": digest})
        provenance.write_manifest(path, fields)
        subprocess.run([sys.executable, str(repo / "tools/exp04_eval.py"), *argv], cwd=str(repo),
                       env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for name in ("per_sample_yaw.json", "metrics_yaw.json"):
            actual, expected = json.loads((output / name).read_text()), json.loads((baseline / name).read_text())
            assert actual["meta"].pop("eval_manifest_sha256") == provenance.sha256_file(path)
            assert actual["meta"].pop("conditions") == condition
            assert actual["meta"].pop("evaluator_closure_sha256") == digest
            assert actual["meta"].pop("reviewed_commit") == commit
            actual["meta"].pop("elapsed_min")
            expected["meta"].pop("elapsed_min")
            if condition == "P":
                assert "E" not in actual
                expected.pop("E")
                expected["meta"]["e_acoustic_cols"] = []
            assert actual == expected
