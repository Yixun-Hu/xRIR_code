"""The training-gate fixes of the Codex ``full_train`` review (findings 1-6, nit 8).

These cases live apart from ``tests/test_exp06_finalize.py`` so the gate work and the
concurrent round-2b HAA work touch different files. The fixtures are the ones that file
already builds; pytest resolves them by name once they are imported here.
"""
import json
import os

import pytest
import torch

from test_exp06_finalize import (DEAD_PID, STATE, bound_args, clone, data_root,  # noqa: F401
                                 dead_pid, full_args, full_run, history_rows,
                                 provenance_record, seal)
from tools import exp06_finalize


def test_epoch_checkpoint_must_agree_in_dtype_with_last_pth(full_run, clone):
    """Nit 8: torch.equal accepts equal values across dtypes; a float64 copy must not pass."""
    run, log = full_run
    torch.save({key: value.double() for key, value in STATE.items()}, run / 'epoch_012.pth')
    with pytest.raises(ValueError, match='dtype'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


def test_epoch_checkpoint_must_agree_in_shape_with_last_pth(full_run, clone):
    """Nit 8: a reshaped tensor of the same values is a different checkpoint."""
    run, log = full_run
    torch.save({key: (value.reshape(3, 2) if value.dim() == 2 else value)
                for key, value in STATE.items()}, run / 'epoch_012.pth')
    with pytest.raises(ValueError, match='shape'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


def test_a_receipt_naming_a_live_child_is_refused(full_run, clone):
    """Finding 3: a partially restored attempt must not certify while its child runs."""
    run, log = full_run
    receipt = json.loads((run / 'child_exit.json').read_text())
    receipt['child_pid'] = os.getpid()
    (run / 'child_exit.json').write_text(json.dumps(receipt, sort_keys=True))
    with pytest.raises(ValueError, match='alive'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()


def test_the_receipt_and_the_child_pid_sidecar_must_agree(full_run, clone):
    """Finding 3: the owner exception covers launch.pid only, never the child."""
    run, log = full_run
    (run / 'child.pid').write_text('{}\n'.format(dead_pid(1)))
    with pytest.raises(ValueError, match='child.pid'):
        exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert not (run / 'completion.json').exists()
    (run / 'child.pid').write_text('{}\n'.format(DEAD_PID))
    fields = exp06_finalize.finalize(run, 'full', log, 0, repo=clone)
    assert fields['child_exit_receipt']['child_pid'] == DEAD_PID
