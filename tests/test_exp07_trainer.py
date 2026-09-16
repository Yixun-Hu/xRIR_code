"""exp_07 protocol selection: the trainer is unchanged, the entry point selects the split.

The bit-identity gate against the pinned trainer needs a real forward/backward pass, and
``model.xRIR.apply_delay`` allocates with ``.cuda()``; those tests are GPU-only and skip
when no CUDA device is visible.
"""
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, default_collate

import train_xRIR_backbone as trainer
from tools import exp07_train
# GPU is exp_05's reviewed CUDA skip marker, reused verbatim.
from test_exp05_trainer import GPU
from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset as UnseenDataset
from treble_multi_room_dataset.treble_xRIR_seen_dataset import xRIR_Dataset as SeenDataset

ROOT = Path(__file__).resolve().parents[1]
# The pristine seams, captured at import: a second bounded_run in one test must wrap the
# trainer's own functions, never the previous run's recorders.
TRAINER_BUILD_MODEL, TRAINER_COMPUTE_LOSS = trainer.build_model, trainer.compute_loss
ADAMW_STEP = torch.optim.AdamW.step
SEEN_TRAIN, SEEN_TEST = 296454, 6217
BATCH, SEEN_BATCHES, SEEN_LAST = 32, 9265, 6
MINIMAL = ['--backbone', 'simple', '--save-dir', '/unused', '--epochs', '0',
           '--num-workers', '0', '--save-every', '0']


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))
        self.source_network = torch.nn.Identity()

    def forward(self, depth, refs, src, locs, target):
        return self.weight.expand(len(src), 2, 2, 1), torch.ones(len(src), 1, 2, 2)


def stub_data_and_model(monkeypatch):
    """Tiny CPU stand-ins for the dataset, the model and .cuda(); returns the loader calls."""
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
    monkeypatch.setattr(trainer, 'build_xrir', lambda *a, **k: _Tiny())
    monkeypatch.setattr(exp07_train, 'UnseenDataset', stub('unseen'))
    monkeypatch.setattr(exp07_train, 'SeenDataset', stub('seen'))
    monkeypatch.setattr(trainer, 'xRIR_Dataset', stub('unseen'))
    return calls


def run_main(monkeypatch, save_dir, extra=(), entry=None):
    """Run the entry point on stub data/model; return ``args.json`` and the dataset calls."""
    calls = stub_data_and_model(monkeypatch)
    argv = ['--backbone', 'simple', '--save-dir', str(save_dir), '--epochs', '1',
            '--num-workers', '0', '--num-shot', '2', '--batch-size', '2', '--save-every', '0',
            '--max-train-batches', '1', '--max-test-batches', '1'] + list(extra)
    monkeypatch.setattr(sys, 'argv', ['entry'] + argv)
    (entry or exp07_train.run_trainer)()
    return json.loads((Path(save_dir) / 'args.json').read_text()), calls


def test_split_protocol_removes_the_flag_and_defaults_to_unseen():
    assert exp07_train.split_protocol(MINIMAL) == ('unseen', MINIMAL)
    assert exp07_train.split_protocol(MINIMAL + ['--protocol', 'seen']) == ('seen', MINIMAL)
    assert exp07_train.split_protocol(['--protocol=seen'] + MINIMAL) == ('seen', MINIMAL)
    assert exp07_train.split_protocol(MINIMAL + ['--protocol', 'unseen']) == ('unseen', MINIMAL)
    for argv, message in ((['--protocol', 'held-out'], 'unknown protocol'),
                          (['--protocol=Seen'], 'unknown protocol'),
                          (['--protocol'], 'requires a value'),
                          (['--protocol', 'seen', '--protocol', 'seen'], 'more than once')):
        with pytest.raises(ValueError, match=message):
            exp07_train.split_protocol(argv)
    # The flag is matched exactly: an abbreviation is left for the trainer to refuse.
    assert exp07_train.split_protocol(['--proto', 'seen']) == ('unseen', ['--proto', 'seen'])


def test_dataset_dispatch_and_the_untouched_trainer():
    assert exp07_train.dataset_class('unseen') is UnseenDataset is trainer.xRIR_Dataset
    assert exp07_train.dataset_class('seen') is SeenDataset
    assert UnseenDataset is not SeenDataset
    with pytest.raises(ValueError, match='protocol'):
        exp07_train.dataset_class('Seen')
    # The trainer itself never learned about protocols: its bytes are main's.
    assert subprocess.run(['git', 'diff', '--quiet', 'main', '--', 'train_xRIR_backbone.py'],
                          cwd=str(ROOT)).returncode == 0


def test_parse_args_is_the_trainers_namespace_plus_protocol(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['trainer'] + MINIMAL)
    reference = vars(trainer.parse_args())
    assert 'protocol' not in reference
    for protocol in ('unseen', 'seen'):
        mine = vars(exp07_train.parse_args(MINIMAL + ['--protocol', protocol]))
        assert set(mine) - set(reference) == {'protocol'} and set(reference) - set(mine) == set()
        assert mine['protocol'] == protocol
        assert {k: v for k, v in mine.items() if k != 'protocol'} == reference
    monkeypatch.setattr(sys, 'argv', ['entry'] + MINIMAL + ['--protocol', 'seen'])
    assert exp07_train.parse_args().protocol == 'seen'


def test_args_json_and_runtime_line_gain_exactly_protocol(monkeypatch, tmp_path, capsys):
    before, before_calls = run_main(monkeypatch, tmp_path / 'trainer', entry=trainer.main)
    capsys.readouterr()
    after, after_calls = run_main(monkeypatch, tmp_path / 'entry')
    lines = [line for line in capsys.readouterr().out.splitlines()
             if line.startswith('XRIR_RUNTIME_ARGS ')]
    runtime = json.loads(lines[0][len('XRIR_RUNTIME_ARGS '):])
    assert set(after) - set(before) == {'protocol'} and set(before) - set(after) == set()
    assert after['protocol'] == runtime['protocol'] == 'unseen' and runtime == after
    ignored = {'protocol', 'save_dir'}
    assert ({key: value for key, value in after.items() if key not in ignored} ==
            {key: value for key, value in before.items() if key not in ignored})
    assert after_calls == before_calls == [('unseen', 'train'), ('unseen', 'test')]


@pytest.mark.parametrize('protocol', ['unseen', 'seen'])
def test_main_selects_both_loaders_by_protocol(monkeypatch, tmp_path, protocol):
    recorded, calls = run_main(monkeypatch, tmp_path, ('--protocol', protocol))
    assert calls == [(protocol, 'train'), (protocol, 'test')]
    assert recorded['protocol'] == protocol
    assert trainer.xRIR_Dataset is not exp07_train.SeenDataset  # every binding restored


# One bounded real-batch epoch of the pinned recipe: the smallest route through main()
# that still builds the model, takes a real optimizer step and runs a test batch.
BOUNDED = ['--num-shot', '8', '--max-len', '9600', '--epochs', '1', '--max-train-batches', '1',
           '--max-test-batches', '1', '--batch-size', str(BATCH), '--accum-steps', '1',
           '--num-workers', '0', '--seed', '0', '--log-interval', '1', '--save-every', '0',
           '--epoch-ckpt-every', '0', '--no-save']


def tensors(mapping):
    return {name: value.detach().cpu().clone() for name, value in mapping.items()}


def rng_state():
    """Every generator a training step can consume, after the run."""
    state = {'python': pickle.dumps(random.getstate()),
             'numpy': pickle.dumps(np.random.get_state()), 'cpu': torch.get_rng_state()}
    state.update({'cuda:' + str(index): value
                  for index, value in enumerate(torch.cuda.get_rng_state_all())})
    return state


def bounded_run(monkeypatch, argv, protocol=None):
    """Run ONE bounded epoch through the real entry point and capture what must agree.

    Both routes reach ``train_xRIR_backbone.main``; the recording seams are the trainer's
    own ``build_model`` and ``compute_loss`` and the optimizer's ``step``, so the
    initialisation, the losses, the gradients at the update, the parameters after it and
    the RNG states afterwards are captured identically for the trainer and for
    ``tools.exp07_train.run_trainer``.
    """
    captured, model_box = {}, []
    build_model, compute_loss, update = TRAINER_BUILD_MODEL, TRAINER_COMPUTE_LOSS, ADAMW_STEP

    def record_model(args):
        model = build_model(args)
        model_box.append(model)
        captured['init'] = tensors(model.state_dict())
        captured['args'] = {key: value for key, value in vars(args).items()
                            if key not in ('save_dir', 'protocol')}
        return model

    def record_loss(model, batch, aug=None):
        losses = compute_loss(model, batch, aug)
        captured.setdefault('losses', []).append([float(item) for item in losses])
        captured.setdefault('aug', []).append(
            None if aug is None else [vars(aug[0]), aug[1], aug[2]])
        return losses

    def record_update(self, *args, **kwargs):
        captured.setdefault('grads', {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in model_box[0].named_parameters()})
        return update(self, *args, **kwargs)

    monkeypatch.setattr(trainer, 'build_model', record_model)
    monkeypatch.setattr(trainer, 'compute_loss', record_loss)
    monkeypatch.setattr(torch.optim.AdamW, 'step', record_update)
    if protocol is None:
        monkeypatch.setattr(sys, 'argv', ['train_xRIR_backbone.py'] + list(argv))
        trainer.main()
    else:
        assert exp07_train.run_trainer(list(argv) + ['--protocol', protocol]) == protocol
    captured['final'] = tensors(model_box[0].state_dict())
    captured['rng'] = rng_state()
    model_box.clear()
    torch.cuda.empty_cache()
    return captured


def compare(actual, expected, where=''):
    assert actual.keys() == expected.keys(), where
    for key, value in expected.items():
        place = where + '/' + str(key)
        if isinstance(value, dict):
            compare(actual[key], value, place)
        elif torch.is_tensor(value):
            assert torch.equal(actual[key], value), place
        else:
            assert actual[key] == value, place


def test_the_bounded_parity_harness_agrees_without_a_gpu(monkeypatch, tmp_path):
    """The GPU harness itself, exercised on stubs: both routes reach the same seams."""
    stub_data_and_model(monkeypatch)
    argv = ['--backbone', 'simple', '--save-dir', str(tmp_path / 'unused'), '--epochs', '1',
            '--num-workers', '0', '--num-shot', '2', '--batch-size', '2', '--save-every', '0',
            '--max-train-batches', '1', '--max-test-batches', '1', '--no-save']
    expected = bounded_run(monkeypatch, argv)
    actual = bounded_run(monkeypatch, argv, protocol='unseen')
    assert set(expected) == {'init', 'args', 'losses', 'aug', 'grads', 'final', 'rng'}
    assert len(expected['losses']) == 2 and expected['aug'] == [None, None]
    assert set(expected['grads']) == {'weight'} and expected['args']['epochs'] == 1
    assert not torch.equal(expected['init']['weight'], expected['final']['weight'])
    compare(actual, expected)


@GPU
@pytest.mark.parametrize('batch_size', [BATCH])
@pytest.mark.parametrize('backbone', ['simple', 'cylindrical'])
@pytest.mark.parametrize('yaw', [False, True])
def test_unseen_step_bit_identical_to_the_trainer(monkeypatch, tmp_path, batch_size, backbone,
                                                  yaw):
    """The entry point runs the unchanged trainer's bounded epoch, bit for bit.

    Both routes are compared on the initialisation, every loss, the gradients at the
    update, the parameters after it and the Python/NumPy/CUDA RNG states, over a real
    AcousticRooms batch of the unseen training split.  cuDNN is pinned to deterministic
    kernels for both, so a difference can only come from the entry point.
    """
    previous = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    argv = ['--backbone', backbone, '--save-dir', str(tmp_path / 'unused')] + BOUNDED
    argv += ['--yaw-aug', '1', '--yaw-aug-seed', '0', '--yaw-aug-width', '512'] if yaw else []
    try:
        expected = bounded_run(monkeypatch, argv)
        actual = bounded_run(monkeypatch, argv, protocol='unseen')
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous
        torch.cuda.empty_cache()
    assert len(expected['losses']) == 2 and expected['args']['train_batches_per_epoch'] == 9261
    assert expected['aug'][0] == (None if not yaw else
                                  [dict(enabled=True, W=512, seed=0, batches_per_epoch=9261), 1, 0])
    assert any(value is not None and value.count_nonzero() > 0
               for name, value in expected['grads'].items() if name.startswith('source_network.'))
    assert any(not torch.equal(expected['init'][name], value)
               for name, value in expected['final'].items())
    compare(actual, expected)
    assert not (tmp_path / 'unused').exists()  # --no-save writes nothing


@pytest.fixture(scope='module')
def seen_split():
    """The frozen seen module opens seen_test_split.pkl relative to the process cwd."""
    previous = os.getcwd()
    os.chdir(str(ROOT))
    try:
        seen = exp07_train.dataset_class('seen')
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


def test_the_entry_points_closure_contains_the_seen_module_and_the_trainer():
    from tools import provenance
    files = set(provenance.source_closure('tools.exp07_train', ROOT))
    assert {'tools/exp07_train.py', 'train_xRIR_backbone.py', exp07_train.SEEN_MODULE,
            'treble_multi_room_dataset/treble_xRIR_dataset.py'} <= files
    assert set(provenance.source_closure('train_xRIR_backbone', ROOT)) < files
