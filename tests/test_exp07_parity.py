"""The deferred GPU parity gate: run exactly the nine cases, receipt what they did.

CPU-only: the pytest subprocess is replaced by a runner that writes the JUnit XML the
real one would, so every refusal is exercised without a GPU.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import exp07_parity as parity
from tools import provenance as p

REPO = Path(parity.__file__).resolve().parents[1]
DEVICE = 'NVIDIA RTX A6000'
SUITE = ('<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" '
         'errors="0" failures="0" skipped="0" tests="{n}">{cases}</testsuite></testsuites>')
BODIES = {'passed': '', 'failed': '<failure message="m">boom</failure>',
          'error': '<error message="m">boom</error>', 'skipped': '<skipped message="no cuda"/>'}


def case(node, outcome):
    path, _, name = node.partition('::')
    return ('<testcase classname="{}" file="{}" name="{}" time="1.0">{}</testcase>'
            .format(path[:-len('.py')].replace('/', '.'), path, name, BODIES[outcome]))


def passing():
    return [(node, 'passed') for node in parity.TESTS]


def fake_pytest(cases=None, status=0, output=b'9 passed, 0 skipped\n'):
    """A stand-in for the pytest subprocess: writes the JUnit XML and returns its status."""
    cases = passing() if cases is None else cases
    def runner(command, **kwargs):
        junit = next(item for item in command if item.startswith('--junitxml='))
        Path(junit[len('--junitxml='):]).write_text(
            SUITE.format(n=len(cases), cases=''.join(case(*item) for item in cases)))
        return SimpleNamespace(returncode=status, stdout=output)
    return runner


@pytest.fixture
def parity_run(tmp_path, monkeypatch):
    monkeypatch.setattr(parity, 'cuda_device', lambda: DEVICE)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO), text=True).strip()
    log = tmp_path / ('gpu_parity_' + head + '.log')
    def run(runner=None, commit=head, extra=('--allow-dirty',)):
        # --allow-dirty: this worktree always carries untracked review artefacts.
        argv = ['--log', str(log), '--reviewed-commit', commit] + list(extra)
        return parity.main(argv, runner=fake_pytest() if runner is None else runner)
    return SimpleNamespace(log=log, head=head, run=run,
                           receipt=log.with_suffix('.receipt.json'),
                           junit=log.with_suffix('.junit.xml'))


def test_the_registered_node_ids_are_the_collected_gpu_cases():
    """The nine ids are real: pytest collects each of them from the two GPU files."""
    output = subprocess.check_output(
        [sys.executable, '-m', 'pytest', 'tests/test_exp07_trainer.py',
         'tests/test_exp07_eval_gpu.py', '--collect-only', '-q', '-p', 'no:cacheprovider'],
        cwd=str(REPO), text=True,
        env=dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES=''))
    collected = {line.strip() for line in output.splitlines() if '::' in line}
    assert len(parity.TESTS) == 9 and len(set(parity.TESTS)) == 9
    assert set(parity.TESTS) <= collected
    everything = {item for item in collected if item.startswith('tests/test_exp07_eval_gpu.py')}
    assert everything == {item for item in parity.TESTS if 'eval_gpu' in item}
    assert sum('test_exp07_trainer.py' in item for item in parity.TESTS) == 4


def test_a_clean_nine_case_run_writes_a_bound_receipt(parity_run):
    receipt = parity_run.run()
    assert json.loads(parity_run.receipt.read_text()) == receipt
    assert receipt['passed'] is True and receipt['pytest_exit'] == 0
    assert receipt['tests'] == {node: 'passed' for node in parity.TESTS}
    assert receipt['git_head'] == receipt['reviewed_commit'] == parity_run.head
    assert receipt['cuda_device'] == DEVICE and receipt['schema_version'] == 1
    for key, path in (('log', parity_run.log), ('junit', parity_run.junit)):
        assert receipt[key]['path'] == str(path.resolve())
        assert receipt[key]['sha256'] == p.sha256_file(path)
    assert parity_run.log.read_bytes() == b'9 passed, 0 skipped\n'
    assert all(node in receipt['command'] for node in parity.TESTS)


@pytest.mark.parametrize('outcome', ['skipped', 'failed', 'error'])
def test_one_case_that_did_not_pass_is_refused(parity_run, outcome):
    cases = [(node, outcome if index == 3 else 'passed')
             for index, node in enumerate(parity.TESTS)]
    with pytest.raises(ValueError, match='did not pass'):
        parity_run.run(fake_pytest(cases))
    assert not parity_run.receipt.exists()


def test_a_missing_case_is_refused(parity_run):
    with pytest.raises(ValueError, match='missing'):
        parity_run.run(fake_pytest(passing()[:-1]))
    assert not parity_run.receipt.exists()


def test_an_unregistered_case_is_refused(parity_run):
    with pytest.raises(ValueError, match='unregistered'):
        parity_run.run(fake_pytest(passing() + [('tests/test_other.py::test_x', 'passed')]))
    assert not parity_run.receipt.exists()


def test_a_nonzero_pytest_exit_is_refused(parity_run):
    with pytest.raises(ValueError, match='pytest exited'):
        parity_run.run(fake_pytest(status=2))
    assert not parity_run.receipt.exists()


def test_the_log_and_the_receipt_are_never_overwritten(parity_run):
    parity_run.log.write_text('an older run')
    with pytest.raises(FileExistsError):
        parity_run.run()
    assert parity_run.log.read_text() == 'an older run' and not parity_run.receipt.exists()


def test_parity_must_run_at_the_reviewed_commit(parity_run):
    with pytest.raises(ValueError, match='reviewed commit'):
        parity_run.run(commit=parity_run.head + '~1')
    assert not parity_run.receipt.exists() and not parity_run.log.exists()


def test_without_a_visible_cuda_device_nothing_runs(parity_run, monkeypatch):
    def refuse():
        raise ValueError('no CUDA device is visible')
    monkeypatch.setattr(parity, 'cuda_device', refuse)
    def never(*args, **kwargs):
        raise AssertionError('pytest must not run without a GPU')
    with pytest.raises(ValueError, match='CUDA'):
        parity_run.run(never)
    assert not parity_run.log.exists() and not parity_run.receipt.exists()


def test_a_dirty_checkout_is_refused_without_the_override(parity_run):
    with pytest.raises(ValueError, match='allow-dirty'):
        parity_run.run(extra=())
    assert not parity_run.log.exists() and not parity_run.receipt.exists()


def test_the_receipt_records_whether_the_override_was_used(parity_run):
    assert parity_run.run()['allow_dirty_used'] is True


def test_a_preexisting_junit_xml_is_refused(parity_run):
    """Blocker 2: the run owns its evidence, so a planted XML is never reused."""
    parity_run.junit.write_text(SUITE.format(n=9, cases=''.join(case(*v) for v in passing())))
    before = parity_run.junit.read_bytes()
    def never(*args, **kwargs):
        raise AssertionError('pytest must not run over someone else\'s JUnit XML')
    with pytest.raises(FileExistsError):
        parity_run.run(never)
    assert parity_run.junit.read_bytes() == before
    assert not parity_run.log.exists() and not parity_run.receipt.exists()


def test_the_child_runs_with_no_ambient_pytest_options(parity_run, monkeypatch):
    """PYTEST_ADDOPTS and any ini addopts are cleared: only these options select cases."""
    monkeypatch.setenv('PYTEST_ADDOPTS', '--help')
    seen = {}
    inner = fake_pytest()
    def runner(command, **kwargs):
        seen.update(command=command, env=kwargs['env'])
        return inner(command, **kwargs)
    parity_run.run(runner)
    assert seen['env']['PYTEST_ADDOPTS'] == ''
    assert seen['command'].count('-o') == 1
    assert seen['command'][seen['command'].index('-o') + 1] == 'addopts='
    assert any(item.startswith('--junitxml=') for item in seen['command'])
    assert seen['command'][-len(parity.TESTS):] == list(parity.TESTS)


def test_a_run_that_writes_no_junit_xml_is_refused(parity_run):
    """The reserved file is still empty: nothing ran, whatever the child's exit status."""
    def silent(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=b'usage: __main__.py [options]\n')
    with pytest.raises(ValueError, match='no JUnit XML'):
        parity_run.run(silent)
    assert not parity_run.receipt.exists()


def test_an_unparseable_junit_xml_is_refused(parity_run):
    def garbage(command, **kwargs):
        junit = next(item for item in command if item.startswith('--junitxml='))
        Path(junit[len('--junitxml='):]).write_text('this is not pytest XML')
        return SimpleNamespace(returncode=0, stdout=b'')
    with pytest.raises(ValueError, match='not parseable'):
        parity_run.run(garbage)
    assert not parity_run.receipt.exists()


def test_an_xml_without_test_cases_is_refused(parity_run):
    def empty(command, **kwargs):
        junit = next(item for item in command if item.startswith('--junitxml='))
        Path(junit[len('--junitxml='):]).write_text('<testsuites><testsuite tests="9"/></testsuites>')
        return SimpleNamespace(returncode=0, stdout=b'')
    with pytest.raises(ValueError, match='missing parity cases'):
        parity_run.run(empty)
    assert not parity_run.receipt.exists()
