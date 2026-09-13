"""Tier recording, historical M checkpoints, and deterministic real-batch steps."""
import json
import pickle
import random
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

import train_xRIR_backbone as trainer
from tools.exp05_params import TIERS, count_parameters

ROOT = Path(__file__).resolve().parents[1]
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="full xRIR steps require CUDA")


def arguments(monkeypatch, backbone="simple", tier="M", extra=()):
    flags = [] if tier == "M" else [item for key, value in TIERS[tier].items()
                                    for item in ("--vit-" + key.replace("_", "-"), str(value))]
    monkeypatch.setattr(sys, "argv", ["trainer", "--backbone", backbone, "--save-dir", "/unused",
                        "--epochs", "0", "--num-workers", "0", "--save-every", "0"] + flags + list(extra))
    return trainer.parse_args()


@pytest.mark.parametrize("backbone,folder", [("simple", "simple"), ("cylindrical", "cyl")])
def test_default_checkpoint_shapes(monkeypatch, backbone, folder):
    args = arguments(monkeypatch, backbone)
    assert {key: getattr(args, "vit_" + key) for key in TIERS["M"]} == dict(TIERS["M"])
    model = trainer.build_model(args)
    checkpoint = torch.load(str(ROOT / "ckpt" / ("xRIR_" + folder + "_8_shot") / "epoch_12.pth"),
                            map_location="cpu")
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == {
        key: tuple(value.shape) for key, value in checkpoint.items()}
    assert args.tier == "M" and args.param_counts == count_parameters(model)


@pytest.mark.parametrize("tier,custom", [("S", False), ("L", False), ("S", True)])
def test_main_records_tier_counts_and_independent_yaw(monkeypatch, tmp_path, tier, custom):
    extra = ["--save-dir", str(tmp_path), "--yaw-aug", "1"]
    if custom:
        extra += ["--vit-depth", "5"]
    args = arguments(monkeypatch, tier=tier, extra=extra)
    monkeypatch.setattr(trainer, "parse_args", lambda: args)
    monkeypatch.setattr(trainer, "xRIR_Dataset", lambda **kwargs: [0])
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **k: self)
    models = []
    original = trainer.build_model
    def build(values):
        models.append(original(values))
        return models[-1]
    monkeypatch.setattr(trainer, "build_model", build)
    trainer.main()
    recorded = json.loads((tmp_path / "args.json").read_text())
    assert recorded["tier"] == ("custom" if custom else tier)
    assert recorded["param_counts"] == count_parameters(models[0])
    assert all(type(value) is int for value in recorded["param_counts"].values())
    assert recorded["yaw_aug"] == 1
    assert {key: recorded["vit_" + key] for key in TIERS[tier]} == dict(
        TIERS[tier], **({"depth": 5} if custom else {}))


@pytest.fixture
def real_batch():
    if not torch.cuda.is_available():
        pytest.skip("full xRIR steps require CUDA")
    previous_device = torch.cuda.current_device()
    torch.cuda.set_device(1 if torch.cuda.device_count() > 1 else 0)
    old = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
           torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    trainer.seed_everything(17)
    try:
        dataset = trainer.xRIR_Dataset(split="train", max_len=9600, num_shot=8)
        yield default_collate([dataset[0]])
    finally:
        (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
         torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) = old
        torch.cuda.empty_cache()
        torch.cuda.set_device(previous_device)


def step(builder, loss_function, batch):
    trainer.seed_everything(42)
    model = builder().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses = loss_function(model, batch)
    assert all(torch.isfinite(loss).all() for loss in losses)
    losses[0].backward()
    gradients = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                 for name, p in model.named_parameters()}
    assert all(value is None or torch.isfinite(value).all() for value in gradients.values())
    assert any(value is not None and value.count_nonzero() > 0 for name, value in gradients.items()
               if name.startswith("source_network."))
    optimizer.step()
    result = {"grad:" + name: value for name, value in gradients.items()}
    result.update({"state:" + name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
    result.update({"loss:" + str(i): value.detach().cpu() for i, value in enumerate(losses)})
    result["python_numpy_rng"] = pickle.dumps((random.getstate(), np.random.get_state()))
    result["cpu_rng"] = torch.get_rng_state()
    result.update({"cuda_rng:" + str(i): state for i, state in enumerate(torch.cuda.get_rng_state_all())})
    return result


@GPU
@pytest.mark.parametrize("backbone", ["simple", "cylindrical"])
def test_m_step_bit_identical_to_prechange(monkeypatch, real_batch, backbone):
    source = subprocess.check_output(["git", "show", "ed5f2a1:train_xRIR_backbone.py"], cwd=str(ROOT))
    before = types.ModuleType("exp05_prechange_trainer")
    exec(compile(source, "ed5f2a1:train_xRIR_backbone.py", "exec"), before.__dict__)
    args = arguments(monkeypatch, backbone)
    expected = step(lambda: before.build_xrir(backbone, 8), before.compute_loss, real_batch)
    actual = step(lambda: trainer.build_model(args), trainer.compute_loss, real_batch)
    assert actual.keys() == expected.keys()
    for key in expected:
        assert (torch.equal(actual[key], expected[key]) if torch.is_tensor(expected[key])
                else actual[key] == expected[key]), key


@GPU
@pytest.mark.parametrize("backbone", ["simple", "cylindrical"])
@pytest.mark.parametrize("tier", ["S", "L"])
def test_non_m_real_optimizer_step(monkeypatch, real_batch, backbone, tier):
    args = arguments(monkeypatch, backbone, tier)
    result = step(lambda: trainer.build_model(args), trainer.compute_loss, real_batch)
    assert args.tier == tier
    assert all(torch.isfinite(value).all() for key, value in result.items() if key.startswith("state:"))
