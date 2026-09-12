"""Recovery checks operate on stub attempts and real sleeping CPU processes."""
import json
import os
from pathlib import Path
import subprocess

import pytest
from tools import exp04_launcher as launch


def test_recovery_requires_dead_group_and_closed_log(tmp_path):
    log = tmp_path / 'train.log'
    log.write_text('closed')
    child = subprocess.Popen([launch.PYTHON, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        with pytest.raises(ValueError, match='process group'):
            launch.assert_quiescent(child.pid, log)
    finally:
        child.terminate()
        child.wait()
    with log.open('r'):
        launch.assert_quiescent(child.pid, log)
    with log.open('a'):
        moved = log.with_name('renamed.log')
        log.rename(moved)
        with pytest.raises(ValueError, match='log writer'):
            launch.assert_quiescent(child.pid, moved)
    launch.assert_quiescent(child.pid, moved)
    for pgid in (None, 0, -1, True):
        with pytest.raises(ValueError, match='process group'):
            launch.assert_quiescent(pgid, moved)
