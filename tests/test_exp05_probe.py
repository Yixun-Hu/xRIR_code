"""The tier probe times the real trainer recipe, validation, and three saves."""
import itertools
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import exp05_probe as probe
from tools.exp05_params import TIERS


@pytest.mark.parametrize('tier,backbone', [(t, b) for t in ('S', 'L') for b in ('simple', 'cylindrical')])
def test_recipe(tier, backbone):
    cmd = probe.trainer_command(tier, backbone, 'scratch')
    fields = {'backbone': backbone, 'save-dir': 'scratch', 'batch-size': 32,
              'accum-steps': 2, 'max-train-batches': 60, 'max-test-batches': 0,
              'num-workers': 12, 'yaw-aug': 0, 'save-every': 0, 'epoch-ckpt-every': 0}
    fields.update({'vit-' + k.replace('_', '-'): v for k, v in TIERS[tier].items()})
    assert '--no-save' in cmd and '--tf32' in cmd
    assert all(cmd[cmd.index('--' + k) + 1] == str(v) for k, v in fields.items())


def measurement():
    return dict(t_micro=dict(mean=1., median=1., min=1., values=[1.] * 50), t_test=2., t_save=3.,
                warmup_micro_batches=10, timed_micro_batches=50, batch_size=32,
                accum_steps=2, train_batches_per_epoch=9261)


@pytest.mark.parametrize('field,value', [('t_test', 0), ('t_save', -1), ('t_test', math.nan), ('t_save', math.inf)])
def test_projection_refuses_invalid(field, value):
    result = measurement()
    result[field] = value
    with pytest.raises(ValueError, match='timing'):
        probe.projection(result)


def test_projection_recomputes_and_refuses_false_summary():
    result = measurement()
    assert probe.projection(result) == dict(T_epoch=9266., T_run=111192., passed=True)
    result['t_micro'] = dict(mean=2., median=2., min=2., values=[2.] * 50)
    assert probe.projection(result)['passed'] is False
    result['t_micro']['mean'] = 1.
    with pytest.raises(ValueError, match='timing'):
        probe.projection(result)


@pytest.mark.parametrize('bad_stage', [None, 'train', 'test'])
def test_child_measures_full_test_and_scratch_saves(monkeypatch, tmp_path, capsys, bad_stage):
    clock = itertools.count()
    monkeypatch.setattr(probe.time, 'perf_counter', lambda: next(clock))
    cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                           max_memory_allocated=lambda: 123, max_memory_reserved=lambda: 456)
    saved = []
    def save(state, path):
        path = Path(path)
        path.write_bytes(b'checkpoint')
        saved.append(path)
    def checkpoint(path, model, optimizer, scheduler, epoch, batch, loss, args):
        assert not args.no_save and epoch == 1 and batch == 0 and loss == 1.25
        save({'optimizer': optimizer, 'scheduler': scheduler}, path)
    def train(model, loader, *args):
        assert list(loader) == list(range(60))
        return math.nan if bad_stage == 'train' else 1.5
    def test(model, loader, epoch, args):
        assert len(loader) == 199 and args.max_test_batches == 0
        return math.nan if bad_stage == 'test' else 1.25
    def main():
        assert sys.argv == probe.trainer_command('L', 'cylindrical', str(tmp_path))
        args = SimpleNamespace(no_save=True, max_test_batches=0)
        model = SimpleNamespace(state_dict=lambda: {'weight': 1})
        fake.train_epoch(model, range(9261), 'optimizer', 'scheduler', 1, args, math.inf)
        fake.test_epoch(model, range(199), 1, args)
        assert args.no_save
    fake = SimpleNamespace(main=main, train_epoch=train, test_epoch=test, save_checkpoint=checkpoint,
                           torch=SimpleNamespace(cuda=cuda, save=save))
    monkeypatch.setitem(sys.modules, 'train_xRIR_backbone', fake)
    if bad_stage:
        with pytest.raises(ValueError, match='non-finite'):
            probe.run('L', 'cylindrical', str(tmp_path))
    else:
        result = probe.run('L', 'cylindrical', str(tmp_path))
        assert result['t_micro'] == dict(mean=1., median=1., min=1., values=[1.] * 50)
        assert result['t_test'] == result['t_save'] == 1
        assert result['test_batches_timed'] == result['test_batches_total'] == 199
        assert result['test_timing_protocol'] == 'full_test_loader'
        assert result['peak_allocated_bytes'] == 123 and result['peak_reserved_bytes'] == 456
        assert {p.name for p in saved} == {'epoch_001.pth', 'best.pth', 'last.pth'}
        assert 'EXP05_PROBE_RESULT ' in capsys.readouterr().out
    assert all(not p.exists() for p in saved) and not list(tmp_path.iterdir())
    assert fake.train_epoch is train and fake.test_epoch is test
