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
from tools.yaw_rotation import rotate_scene_yaw
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
    rotated = trainer.compute_loss(model, batch, (YawAug(True), 1, 0))[0]
    assert torch.isfinite(rotated) and not torch.equal(actual[0], rotated)
    rotated.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        monkeypatch.setattr(YawAug, "offsets_for", lambda self, epoch, idx, n: torch.zeros(n, dtype=torch.long))
        assert torch.equal(actual[0], trainer.compute_loss(model, batch, (YawAug(True), 1, 0))[0])
        ks = torch.tensor([37, 400])
        monkeypatch.setattr(YawAug, "offsets_for", lambda self, epoch, idx, n: ks)
        scenes = [rotate_scene_yaw(*(batch[i][b:b + 1] for i in (2, 1, 5)), int(ks[b]))
                  for b in range(len(ks))]
        batch_rot = list(batch)
        for index, rows in zip((2, 1, 5), zip(*scenes)):
            batch_rot[index] = torch.cat(rows)
        expected = _reference_compute_loss(model, batch_rot)
        actual_rot = trainer.compute_loss(model, batch, (YawAug(True), 1, 0))
        assert all(torch.equal(a, b) for a, b in zip(actual_rot, expected))
    assert all(torch.equal(batch[i], old) for i, old in zip((3, 4), audio))


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.source_network = torch.nn.Identity()

    def forward(self, depth, refs, src, locs, target):
        return self.weight.expand(len(src), 2, 2, 1), torch.ones(len(src), 1, 2, 2)


@pytest.fixture
def run_main(monkeypatch, tmp_path, batch):
    monkeypatch.setattr(trainer, "build_xrir", lambda *a, **k: _Tiny())
    samples = [tuple(component[i] for component in batch) for i in range(2)]
    monkeypatch.setattr(trainer, "xRIR_Dataset", lambda **k: samples if k["split"] == "train" else samples * 2)
    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["trainer", "--backbone", "simple", "--save-dir", str(tmp_path / "run"),
            "--epochs", "1", "--num-workers", "0", "--num-shot", "2", "--batch-size", "2", "--save-every", "0",
            "--epoch-ckpt-every", "1", "--max-train-batches", "1", "--max-test-batches", "1"] + list(extra))
        trainer.main()
    return run


@pytest.mark.parametrize("enabled,no_save,epochs,aug_seed", [
    (0, True, 1, None), (1, True, 2, None), (1, False, 1, None), (1, False, 1, 3)])
def test_main_saves_banner_and_eval_gate(run_main, monkeypatch, tmp_path, capsys, enabled, no_save, epochs, aug_seed, batch):
    calls = []
    original = YawAug.offsets_for
    def offsets(self, epoch, idx, n):
        assert type(epoch) is int and type(idx) is int
        assert self.batches_per_epoch == 1
        calls.append((epoch, idx))
        return original(self, epoch, idx, n)
    monkeypatch.setattr(YawAug, "offsets_for", offsets)
    monkeypatch.setenv("PYTHONHASHSEED", "17")
    seed = 9 if aug_seed is None else aug_seed
    extra = [] if aug_seed is None else ["--yaw-aug-seed", str(aug_seed)]
    run_main("--yaw-aug", str(enabled), "--seed", "9" if aug_seed is None else "0",
             "--epochs", str(epochs), *extra, *(["--no-save"] if no_save else []))
    text = capsys.readouterr().out
    banner = f"yaw_aug ENABLED W=512 seed={seed} counter=(epoch-1)*1+batch_idx" if enabled else "yaw_aug DISABLED"
    assert text.count("yaw_aug ") == 1 and banner in text
    assert text.index("yaw_aug ") < text.index("Train Epoch")
    records = [line[len("XRIR_RUNTIME_ARGS "):] for line in text.splitlines()
               if line.startswith("XRIR_RUNTIME_ARGS ")]
    assert len(records) == 1
    runtime = json.loads(records[0])
    assert text.index("XRIR_RUNTIME_ARGS ") < text.index("backbone:")
    assert set(runtime) == set(vars(trainer.parse_args())) | {"train_batches_per_epoch", "env", "tier", "param_counts"}
    assert (runtime["yaw_aug"], runtime["yaw_aug_seed"], runtime["yaw_aug_width"], runtime["no_save"]) == (
        enabled, seed, 512, no_save)
    assert runtime["train_batches_per_epoch"] == 1 and runtime["env"]["PYTHONHASHSEED"] == "17"
    assert set(runtime["env"]) == {"PYTHONHASHSEED", "XRIR_DATA_PATH", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES"}
    assert calls == ([(epoch, 0) for epoch in range(1, epochs + 1)] if enabled else [])
    monkeypatch.setattr(trainer, "apply_yaw_aug", lambda *a, **k: pytest.fail("evaluation augmented"))
    trainer.test_epoch(_Tiny(), [batch], 1, trainer.parse_args())
    if no_save:
        assert not list(tmp_path.rglob("*"))
    else:
        args = json.loads((tmp_path / "run" / "args.json").read_text())
        assert args == runtime
        assert (args["yaw_aug"], args["yaw_aug_seed"], args["yaw_aug_width"], args["no_save"]) == (1, seed, 512, False)
        assert args["train_batches_per_epoch"] == 1 and args["env"]["PYTHONHASHSEED"] == "17"
        assert set(args["env"]) == {"PYTHONHASHSEED", "XRIR_DATA_PATH", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES"}


@pytest.mark.parametrize("extra,match", [(["--resume", "x"], "resume"), (["--resume", ""], "resume"),
    (["--save-every", "500"], "save-every"),
    (["--save-every", "-1"], "save-every"), (["--yaw-aug", "2"], "invalid choice")])
def test_invalid_flags(run_main, capsys, extra, match):
    with pytest.raises(SystemExit):
        run_main("--yaw-aug", "1", *extra)
    assert match in capsys.readouterr().err.splitlines()[-1]


def test_no_save_suppresses_mid_epoch_checkpoint(run_main, tmp_path, monkeypatch):
    calls = []
    original = trainer.save_checkpoint
    def save(*args):
        calls.append(args[5])
        return original(*args)
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    run_main("--yaw-aug", "0", "--no-save", "--save-every", "1", "--max-train-batches", "2", "--batch-size", "1")
    assert calls == [1] and not list(tmp_path.rglob("*"))


_INVALID_COUNTERS = [(1.7, 0), (True, 0), (np.int64(1), 0),
                     (1, 0.2), (1, False), (1, np.int64(0))]


@pytest.mark.parametrize("epoch,idx", _INVALID_COUNTERS)
def test_train_epoch_requires_native_counters(run_main, monkeypatch, epoch, idx):
    original = trainer.train_epoch
    def train(*args):
        values = list(args)
        values[4] = epoch
        return original(*values)
    monkeypatch.setattr(trainer, "train_epoch", train)
    monkeypatch.setattr(trainer, "enumerate", lambda loader: iter([(idx, next(iter(loader)))]), raising=False)
    with pytest.raises(ValueError, match="native int"):
        run_main("--yaw-aug", "1", "--no-save")


@pytest.mark.parametrize("epoch,idx", _INVALID_COUNTERS)
def test_compute_loss_does_not_coerce_counters(batch, epoch, idx):
    with pytest.raises(ValueError, match="integers"):
        trainer.compute_loss(_Tiny(), batch, (YawAug(True), epoch, idx))


def test_compute_loss_requires_enabled_augmentation(batch):
    with pytest.raises(AssertionError):
        trainer.compute_loss(_Tiny(), batch, (YawAug(False), 1, 0))


def test_width_guard(run_main):
    with pytest.raises(ValueError, match="width|W"):
        run_main("--yaw-aug", "1", "--no-save", "--yaw-aug-width", "256")


@pytest.mark.parametrize("length,epochs,refused", [
    (9262, 1, False), (2**20 - 1, 1, False), (2**20, 1, True),
    (2**19, 2, True), (2**19 - 1, 2, False), (9261, 114, True)])
def test_loader_counter_overflow_boundary(run_main, monkeypatch, length, epochs, refused):
    monkeypatch.setattr(trainer.DataLoader, "__len__", lambda self: length)
    calls = []
    original = YawAug.offsets_for
    def offsets(self, epoch, idx, n):
        assert self.batches_per_epoch == length
        calls.append((epoch, idx))
        return original(self, epoch, idx, n)
    monkeypatch.setattr(YawAug, "offsets_for", offsets)
    if refused:
        with pytest.raises(ValueError, match=r"epochs.*train_batches_per_epoch.*< 2\*\*20"):
            run_main("--yaw-aug", "1", "--no-save", "--epochs", str(epochs))
        assert calls == []
    else:
        run_main("--yaw-aug", "1", "--no-save", "--epochs", str(epochs))
        assert calls == [(epoch, 0) for epoch in range(1, epochs + 1)]
