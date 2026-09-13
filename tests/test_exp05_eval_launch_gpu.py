"""A bounded real tier launcher run must equal the direct evaluator."""
import json

import torch

from test_exp05_eval_gpu import ROOT, evaluate, real_batch
from tools import exp04_eval_launch as launcher, exp05_eval, provenance as p


def test_real_launcher_32_queries_equals_direct(tmp_path, real_batch):
    checkpoint = ROOT / "ckpt/xRIR_simple_8_shot/epoch_12.pth"
    reference = ROOT / "ckpt/yaw_aug/reference_manifest_k8_seed42.json"
    torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = False
    expected = evaluate(tmp_path / "direct", exp05_eval, "simple", "M", checkpoint, samples=32)
    run = tmp_path / "launched"
    completion = launcher.main([
        "--tier", "M", "--backbone", "simple", "--checkpoint", str(checkpoint),
        "--manifest", str(reference), "--manifest-hash", expected[0]["meta"]["manifest_hash"],
        "--out-dir", str(run), "--log-dir", str(tmp_path / "logs"), "--run-label", "tier",
        "--data-root", launcher.evaluator.BASE_DATA_PATH, "--reviewed-commit", "HEAD",
        "--gpu", "1", "--num-shot", "8", "--max-samples", "32", "--gl-seed", "42",
        "--conditions", "P", "--yaw-cols", "0", "--acoustic-cols", "0", "--e-acoustic-cols",
        "--decomposition-batches", "0", "--num-workers", "0", "--threads", "2"])
    manifest = json.loads((run / "eval_manifest.json").read_text())
    assert manifest["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert manifest["mutable_inputs"]["train_args"]["sha256"] == p.sha256_file(checkpoint.parent / "args.json")
    assert completion["confirmatory"] is False
    for key in ("tier", "param_counts", "args_json_sha256", "legacy_M"):
        assert completion[key] == manifest[key] == expected[0]["meta"][key]
    for name, baseline in zip(launcher.OUTPUTS, expected):
        actual = json.loads((run / name).read_text())
        for payload in (actual, baseline):
            for key in ("elapsed_min", "eval_manifest_sha256"):
                payload["meta"].pop(key)
        assert actual == baseline
