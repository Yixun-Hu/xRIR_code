"""Trainer contract, including literal loss parity against commit 8cb87d1."""
import contextlib
import io
import json
import pickle
import random
import sys

import numpy as np
import pytest
import torch

import train_xRIR_backbone as trainer
from tools.yaw_aug import YawAug
from utils.spec_utils import compute_spect_energy_decay_losses, stft_l1_loss


def _reference_compute_loss(model, batch):
    """Identical loss to train_xRIR_unseen.py: STFT log-mag L1 + Schroeder energy-decay L1."""
    _, src_loc, depth_coord, tgt_wav, ref_irs, ref_locs = batch
    out_spec, tgt_spec = model(
        depth_coord.cuda(non_blocking=True), ref_irs.cuda(non_blocking=True),
        src_loc.cuda(non_blocking=True), ref_locs.cuda(non_blocking=True),
        tgt_wav.cuda(non_blocking=True))
    gt = tgt_spec.permute(0, 2, 3, 1)
    with contextlib.redirect_stdout(io.StringIO()):  # the decay loss prints tensor shapes on every call
        decay_loss = compute_spect_energy_decay_losses(gts=gt, preds=torch.exp(out_spec) - 1e-8)
    stft_loss = stft_l1_loss(pred_spect=out_spec, gt_spect=gt)
    return stft_loss + decay_loss, stft_loss.detach(), decay_loss.detach()


@pytest.fixture
def batch(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **k: self)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    yield (torch.zeros(2, 3), torch.randn(2, 3), torch.randn(2, 3, 256, 512),
           torch.randn(2, 1, 9600), torch.randn(2, 2, 9600), torch.randn(2, 2, 3))
    torch.set_num_threads(threads)


def test_real_model_parity_and_augmentation(batch, monkeypatch):
    model = trainer.build_xrir("simple", num_shot=2).train()
    buffers = [b.clone() for b in model.buffers()]
    rng = lambda: pickle.dumps((random.getstate(), np.random.get_state(), torch.get_rng_state().numpy()))
    state = rng()
    reference = _reference_compute_loss(model, batch)
    reference[0].backward()
    gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    updated = [b.clone() for b in model.buffers()]
    for b, saved in zip(model.buffers(), buffers):
        b.copy_(saved)
    actual = trainer.compute_loss(model, batch)
    assert all(torch.equal(a, b) for a, b in zip(actual, reference)) and rng() == state
    assert all(torch.equal(a, b) for a, b in zip(model.buffers(), updated))
    assert all(p.grad is None if g is None else torch.equal(p.grad, g)
               for p, g in zip(model.parameters(), gradients))
    model.zero_grad(set_to_none=True)
    actual[0].backward()
    assert all(p.grad is None if g is None else torch.equal(p.grad, g)
               for p, g in zip(model.parameters(), gradients))
    audio = [batch[i].clone() for i in (3, 4)]
    model.zero_grad(set_to_none=True)
    rotated = trainer.compute_loss(model, batch, (YawAug(True), np.int64(1), np.int64(0)))[0]
    assert torch.isfinite(rotated) and not torch.equal(actual[0], rotated)
    rotated.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        monkeypatch.setattr(YawAug, "offsets_for", lambda self, epoch, idx, n: torch.zeros(n, dtype=torch.long))
        assert torch.equal(actual[0], trainer.compute_loss(model, batch, (YawAug(True), 1, 0))[0])
    assert all(torch.equal(batch[i], old) for i, old in zip((3, 4), audio))


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, depth, refs, src, locs, target):
        return self.weight.expand(len(src), 2, 2, 1), torch.ones(len(src), 1, 2, 2)


@pytest.fixture
def run_main(monkeypatch, tmp_path, batch):
    monkeypatch.setattr(trainer, "build_xrir", lambda *a, **k: _Tiny())
    samples = [tuple(component[i] for component in batch) for i in range(2)]
    monkeypatch.setattr(trainer, "xRIR_Dataset", lambda **k: samples)
    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["trainer", "--backbone", "simple", "--save-dir", str(tmp_path / "run"),
            "--epochs", "1", "--num-workers", "0", "--num-shot", "2", "--batch-size", "2", "--save-every", "0",
            "--epoch-ckpt-every", "1", "--max-train-batches", "1", "--max-test-batches", "1"] + list(extra))
        trainer.main()
    return run


@pytest.mark.parametrize("enabled,no_save", [(0, True), (1, True), (1, False)])
def test_main_saves_banner_and_eval_gate(run_main, monkeypatch, tmp_path, capsys, enabled, no_save, batch):
    calls = []
    original = YawAug.offsets_for
    def offsets(self, epoch, idx, n):
        assert type(epoch) is int and type(idx) is int
        calls.append((epoch, idx))
        return original(self, epoch, idx, n)
    monkeypatch.setattr(YawAug, "offsets_for", offsets)
    monkeypatch.setenv("PYTHONHASHSEED", "17")
    run_main("--yaw-aug", str(enabled), "--seed", "9", *(["--no-save"] if no_save else []))
    text = capsys.readouterr().out
    banner = "yaw_aug ENABLED W=512 seed=9 counter=(epoch-1)*9261+batch_idx" if enabled else "yaw_aug DISABLED"
    assert text.count("yaw_aug ") == 1 and banner in text
    assert calls == ([(1, 0)] if enabled else [])
    monkeypatch.setattr(trainer, "apply_yaw_aug", lambda *a, **k: pytest.fail("evaluation augmented"))
    trainer.test_epoch(_Tiny(), [batch], 1, trainer.parse_args())
    if no_save:
        assert not list(tmp_path.rglob("*"))
    else:
        args = json.loads((tmp_path / "run" / "args.json").read_text())
        assert (args["yaw_aug"], args["yaw_aug_seed"], args["yaw_aug_width"], args["no_save"]) == (1, 9, 512, False)
        assert args["train_batches_per_epoch"] == 1 and args["env"]["PYTHONHASHSEED"] == "17"
        assert set(args["env"]) == {"PYTHONHASHSEED", "XRIR_DATA_PATH", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES"}


@pytest.mark.parametrize("extra,match", [(["--resume", "x"], "resume"), (["--save-every", "500"], "save-every"),
    (["--save-every", "-1"], "save-every"), (["--yaw-aug", "2"], "invalid choice")])
def test_invalid_flags(run_main, capsys, extra, match):
    with pytest.raises(SystemExit):
        run_main("--yaw-aug", "1", *extra)
    assert match in capsys.readouterr().err.splitlines()[-1]


@pytest.mark.parametrize("oversized", [False, True])
def test_width_and_loader_bound(run_main, monkeypatch, oversized):
    if oversized:
        monkeypatch.setattr(trainer.DataLoader, "__len__", lambda self: 9262)
    with pytest.raises(ValueError, match="9261" if oversized else "width|W"):
        run_main("--yaw-aug", "1", "--no-save", "--yaw-aug-width", "256")
