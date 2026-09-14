"""Keep the pinned exp_03 acceptance test intact on CPU-only hosts."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('devices', (0, 1, 2, 3))
def test_pinned_acceptance_gpu_requirement(monkeypatch, devices):
    path = Path(__file__).with_name('conftest.py')
    spec = importlib.util.spec_from_file_location('exp04_collection_config', path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: devices)
    marks = [[], [], []]
    target = 'test_skipping_the_gate_recomputation_is_not_acceptance'
    items = [SimpleNamespace(fspath=path.with_name(filename), nodeid=filename + '::' + name,
                             add_marker=markers.append)
             for filename, name, markers in (
                 ('test_exp03_record_tools.py', target, marks[0]),
                 ('test_exp03_record_tools.py', 'test_other', marks[1]),
                 ('test_other.py', target, marks[2]))]
    config.pytest_collection_modifyitems(items)
    assert len(marks[0]) == int(devices < 2)
    assert not marks[1] and not marks[2]
    if marks[0]:
        assert marks[0][0].name == 'skip'
        assert 'two visible GPUs' in marks[0][0].kwargs['reason']
