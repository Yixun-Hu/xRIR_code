"""Real GPU-1 evaluation parity and smoke-checkpoint round trips for every tier."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

import eval_yaw_rotation as frozen
from test_exp05_trainer import real_batch, step
from tools import exp04_eval, exp05_eval, provenance
from tools.exp05_params import TIERS, build_tier, count_parameters
from train_xRIR_backbone import compute_loss

ROOT = Path(__file__).resolve().parents[1]
EXTRA_META = {"elapsed_min", "conditions", "eval_manifest_sha256", "reviewed_commit",
              "evaluator_closure_sha256", "tier", "param_counts", "legacy_M",
              "args_json_sha256", *("vit_" + key for key in TIERS["M"])}


def evaluate(output, module, backbone, tier, checkpoint, shots=8, samples=16):
    reference = ROOT / "ckpt/yaw_aug/reference_manifest_k{}_seed42.json".format(shots)
    manifest = frozen.load_manifest(str(reference))
    argv = ["--backbone", backbone, "--checkpoint", str(checkpoint), "--manifest",
            str(reference), "--manifest-hash", frozen.manifest_hash(manifest),
            "--out-dir", str(output), "--max-samples", str(samples), "--gl-seed", "42",
            "--yaw-cols", "0", "--acoustic-cols", "0", "--e-acoustic-cols",
            "--decomposition-batches", "0", "--num-workers", "0", "--threads", "2"]
    if module is frozen:
        frozen.main(argv)
    else:
        output.mkdir()
        path = output / "eval_manifest.json"
        argv += ["--eval-manifest", str(path), "--conditions", "P"]
        if module is exp05_eval:
            argv += ["--tier", tier]
        args = module.parse_args(argv)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
        records, digest = provenance.closure_record(
            provenance.source_closure("eval_yaw_rotation", ROOT), commit, ROOT)
        fields = {key: value for key, value in vars(args).items()
                  if key not in ("out_dir", "eval_manifest", "manifest")}
        fields.update(schema_version=1, repo=str(ROOT), reviewed_commit=commit,
                      manifest_path=str(reference), manifest_seed=42, num_shot=shots,
                      manifest_file_sha256=provenance.sha256_file(reference),
                      checkpoint_sha256=provenance.sha256_file(checkpoint),
                      batch_canonical=True, data_root=exp04_eval.BASE_DATA_PATH,
                      evaluator_closure={"files": records, "sha256": digest})
        if module is exp05_eval:
            fields.update(exp05_eval.tier_metadata(args))
        provenance.write_manifest(path, fields)
        (exp05_eval.run_exp05 if module is exp05_eval else exp04_eval.run_exp04)(args)
    return [json.loads((output / name).read_text())
            for name in ("per_sample_yaw.json", "metrics_yaw.json")]


@pytest.mark.parametrize("backbone,folder", [("simple", "simple"), ("cylindrical", "cyl")])
def test_m_32_query_seed42_k8_matches_both_evaluators(tmp_path, real_batch, backbone, folder):
    checkpoint = ROOT / "ckpt" / ("xRIR_" + folder + "_8_shot") / "epoch_12.pth"
    outputs = [evaluate(tmp_path / name, module, backbone, "M", checkpoint, samples=32)
               for name, module in (("frozen", frozen), ("exp04", exp04_eval), ("exp05", exp05_eval))]
    for original, previous, actual in zip(*outputs):
        assert actual["meta"]["legacy_M"] is True
        assert actual["meta"]["tier"] == "M"
        assert actual["meta"]["args_json_sha256"] == provenance.sha256_file(checkpoint.parent / "args.json")
        for key, value in TIERS["M"].items():
            assert actual["meta"]["vit_" + key] == value
        assert all(type(value) is int for value in actual["meta"]["param_counts"].values())
        for payload in (original, previous, actual):
            payload["meta"] = {key: value for key, value in payload["meta"].items() if key not in EXTRA_META}
        original.pop("E")
        assert actual == previous == original  # Includes query/index order, null masks and aggregates.


@pytest.mark.parametrize("backbone", ["simple", "cylindrical"])
@pytest.mark.parametrize("tier", ["S", "L"])
def test_one_step_checkpoint_evaluates_k8_and_k1(tmp_path, real_batch, backbone, tier):
    result = step(lambda: build_tier(backbone, tier), compute_loss, real_batch)
    state = {key[6:]: value for key, value in result.items() if key.startswith("state:")}
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    checkpoint = attempt / "one_step.pth"
    torch.save(state, str(checkpoint))
    del result, state
    config = {"vit_" + key: value for key, value in TIERS[tier].items()}
    counts = count_parameters(build_tier(backbone, tier))
    (attempt / "args.json").write_text(json.dumps(dict(config, tier=tier, backbone=backbone,
                                                       no_save=True, param_counts=counts)))
    for shots in (8, 1):
        per_sample, metrics = evaluate(tmp_path / ("k" + str(shots)), exp05_eval, backbone, tier, checkpoint, shots)
        assert per_sample["meta"]["tier"] == tier and per_sample["meta"]["legacy_M"] is False
        assert per_sample["meta"]["param_counts"] == metrics["meta"]["param_counts"] == counts
        assert len(per_sample["query"]) == len(per_sample["index"]) == 16
        for values in per_sample["P"]["0"].values():
            assert len(values) == 16 and np.isfinite(np.asarray(values, dtype=float)).all()
        for aggregate in metrics["P"]["0"].values():
            assert aggregate["n_valid"] == 16 and aggregate["n_nan"] == 0
            assert np.isfinite(aggregate["mean"])
