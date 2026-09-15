"""exp_07 protocol selection: the unseen default is unchanged, the seen split loads.

The bit-identity gate against the pre-change trainer (commit f492613) needs a real
forward/backward pass, and ``model.xRIR.apply_delay`` allocates with ``.cuda()``; that
test is therefore GPU-only and skips when no CUDA device is visible.
"""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, default_collate

import train_xRIR_backbone as trainer
# real_batch is a fixture; step/GPU are exp_05's reviewed CUDA parity harness, reused verbatim.
from test_exp05_trainer import GPU, real_batch, step
from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset as UnseenDataset
from treble_multi_room_dataset.treble_xRIR_seen_dataset import xRIR_Dataset as SeenDataset

ROOT = Path(__file__).resolve().parents[1]
BASE = 'f492613'  # the trainer immediately before --protocol
SEEN_TRAIN, SEEN_TEST = 296454, 6217
BATCH, SEEN_BATCHES, SEEN_LAST = 32, 9265, 6


def parse(monkeypatch, *extra):
    monkeypatch.setattr(sys, 'argv', ['trainer', '--backbone', 'simple', '--save-dir', '/unused',
                        '--epochs', '0', '--num-workers', '0', '--save-every', '0'] + list(extra))
    return trainer.parse_args()


@pytest.fixture(scope='module')
def prechange():
    source = subprocess.check_output(['git', 'show', BASE + ':train_xRIR_backbone.py'], cwd=str(ROOT))
    module = types.ModuleType('exp07_prechange_trainer')
    exec(compile(source, BASE + ':train_xRIR_backbone.py', 'exec'), module.__dict__)
    return module


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.source_network = torch.nn.Identity()

    def forward(self, depth, refs, src, locs, target):
        return self.weight.expand(len(src), 2, 2, 1), torch.ones(len(src), 1, 2, 2)


def run_main(module, monkeypatch, save_dir, extra=()):
    """Run ``main`` on stub data/model; return the recorded args.json and the dataset calls."""
    samples = [(torch.zeros(3), torch.randn(3), torch.randn(3, 4, 8), torch.randn(1, 16),
                torch.randn(2, 16), torch.randn(2, 3)) for _ in range(2)]
    calls = []

    def stub(label):
        def factory(**kwargs):
            calls.append((label, kwargs['split']))
            return samples
        return factory

    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self, *a, **k: self)
    monkeypatch.setattr(module, 'build_xrir', lambda *a, **k: _Tiny())
    monkeypatch.setattr(module, 'xRIR_Dataset', stub('unseen'))
    if hasattr(module, 'SeenDataset'):
        monkeypatch.setattr(module, 'SeenDataset', stub('seen'))
    monkeypatch.setattr(sys, 'argv', ['trainer', '--backbone', 'simple', '--save-dir', str(save_dir),
        '--epochs', '1', '--num-workers', '0', '--num-shot', '2', '--batch-size', '2',
        '--save-every', '0', '--max-train-batches', '1', '--max-test-batches', '1'] + list(extra))
    module.main()
    return json.loads((Path(save_dir) / 'args.json').read_text()), calls


def test_protocol_default_and_dataset_dispatch(monkeypatch):
    assert parse(monkeypatch).protocol == 'unseen'
    assert parse(monkeypatch, '--protocol', 'seen').protocol == 'seen'
    assert parse(monkeypatch, '--protocol', 'unseen').protocol == 'unseen'
    with pytest.raises(SystemExit):
        parse(monkeypatch, '--protocol', 'held-out')
    assert trainer.dataset_class('unseen') is UnseenDataset is trainer.xRIR_Dataset
    assert trainer.dataset_class('seen') is SeenDataset
    assert UnseenDataset is not SeenDataset
    with pytest.raises(ValueError, match='protocol'):
        trainer.dataset_class('Seen')


def test_args_schema_gains_exactly_protocol(prechange, monkeypatch, tmp_path):
    before, before_calls = run_main(prechange, monkeypatch, tmp_path / 'before')
    after, after_calls = run_main(trainer, monkeypatch, tmp_path / 'after')
    assert set(after) - set(before) == {'protocol'} and set(before) - set(after) == set()
    assert after['protocol'] == 'unseen'
    ignored = {'protocol', 'save_dir'}
    assert ({key: value for key, value in after.items() if key not in ignored} ==
            {key: value for key, value in before.items() if key not in ignored})
    assert after_calls == before_calls == [('unseen', 'train'), ('unseen', 'test')]


@pytest.mark.parametrize('protocol', ['unseen', 'seen'])
def test_main_selects_both_loaders_by_protocol(monkeypatch, tmp_path, protocol):
    recorded, calls = run_main(trainer, monkeypatch, tmp_path, ('--protocol', protocol))
    assert calls == [(protocol, 'train'), (protocol, 'test')]
    assert recorded['protocol'] == protocol


@GPU
@pytest.mark.parametrize('real_batch', [BATCH], indirect=True)
@pytest.mark.parametrize('backbone', ['simple', 'cylindrical'])
@pytest.mark.parametrize('yaw', [False, True])
def test_unseen_step_bit_identical_to_prechange(prechange, monkeypatch, real_batch, backbone, yaw):
    args = parse(monkeypatch, '--backbone', backbone)
    assert args.protocol == 'unseen'
    aug = (trainer.YawAug(True, 512, 0, 9261), 1, 0) if yaw else None
    loss_of = lambda module: (lambda model, batch: module.compute_loss(model, batch, aug))
    expected = step(lambda: prechange.build_xrir(backbone, 8), loss_of(prechange), real_batch)
    actual = step(lambda: trainer.build_model(args), loss_of(trainer), real_batch)
    assert actual.keys() == expected.keys()
    for key in expected:
        assert (torch.equal(actual[key], expected[key]) if torch.is_tensor(expected[key])
                else actual[key] == expected[key]), key


@pytest.fixture(scope='module')
def seen_split():
    """The frozen seen module opens seen_test_split.pkl relative to the process cwd."""
    previous = os.getcwd()
    os.chdir(str(ROOT))
    try:
        seen = trainer.dataset_class('seen')
        return (seen(split='train', max_len=9600, num_shot=8),
                seen(split='test', max_len=9600, num_shot=8))
    finally:
        os.chdir(previous)


def test_seen_split_sizes(seen_split):
    train, test = seen_split
    assert type(train) is SeenDataset and type(test) is SeenDataset
    assert (len(train), len(test)) == (SEEN_TRAIN, SEEN_TEST)
    assert not set(train.file_list) & set(test.file_list)


def test_seen_loader_length_and_final_batch(seen_split):
    train, _ = seen_split
    loader = DataLoader(train, shuffle=True, batch_size=BATCH, num_workers=0)
    assert len(loader) == SEEN_BATCHES
    assert len(train) - BATCH * (SEEN_BATCHES - 1) == SEEN_LAST


@pytest.mark.parametrize('split', [0, 1])
def test_seen_real_batch_loads_on_cpu(seen_split, split):
    dataset = seen_split[split]
    tensors = default_collate([dataset[0], dataset[len(dataset) - 1]])
    assert [tuple(t.shape) for t in tensors] == [(2, 3), (2, 3), (2, 3, 256, 512),
                                                 (2, 1, 9600), (2, 8, 9600), (2, 8, 3)]
    assert all(t.dtype is torch.float32 and torch.isfinite(t).all() for t in tensors)
    assert torch.equal(tensors[0], torch.zeros(2, 3))
