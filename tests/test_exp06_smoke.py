"""Bounded in-process smoke runner: alarm, peak memory, receipts and the CPU fixture."""
import json
import sys
import time
import types
from pathlib import Path

import pytest
import torch

from model.xRIR_cyl_oriented import build_xrir_exp06
from tools import exp06_smoke, exp06_train, provenance

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def stub_entry(monkeypatch):
    """Round 2b's HAA entries do not exist yet; dispatch is by module name."""
    calls = []

    def main(argv, sleep=0.0):
        calls.append(list(argv))
        time.sleep(sleep)
        return 0

    module = types.ModuleType('tools.exp06_haa_finetune')
    module.main = main
    module.calls = calls
    monkeypatch.setitem(sys.modules, 'tools.exp06_haa_finetune', module)
    return module


def test_entry_runs_in_process_and_writes_a_receipt(tmp_path, stub_entry):
    receipt = tmp_path / 'smoke.json'
    argv = ['--backbone', 'cylindrical_oriented', '--rooms', 'class_room']
    result = exp06_smoke.run_entry('exp06_haa_finetune', argv, receipt=receipt, alarm_seconds=30)
    assert stub_entry.calls == [argv]
    assert result['exit_status'] == 0 and result['outcome'] == 'ok'
    assert result['diagnostic'] is True
    assert result['entry'] == 'exp06_haa_finetune' and result['argv'] == argv
    assert result['peak_bytes'] == 0 and result['wall_s'] >= 0
    assert result['git_head'] == provenance.git_state(REPO)['HEAD']
    assert json.loads(receipt.read_text()) == result


def test_alarm_aborts_with_status_three(tmp_path, stub_entry, monkeypatch):
    monkeypatch.setattr(stub_entry, 'main', lambda argv: time.sleep(30))
    receipt = tmp_path / 'alarm.json'
    with pytest.raises(SystemExit) as exit:
        exp06_smoke.run_entry('exp06_haa_finetune', [], receipt=receipt, alarm_seconds=0.2)
    assert exit.value.code == 3
    record = json.loads(receipt.read_text())
    assert record['outcome'] == 'aborted_alarm' and record['exit_status'] == 3
    assert record['wall_s'] < 5
    assert record['alarm_seconds'] == 0.2 and record['diagnostic'] is True


def test_peak_memory_over_the_budget_aborts(tmp_path, stub_entry, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 4 * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    receipt = tmp_path / 'memory.json'
    with pytest.raises(SystemExit) as exit:
        exp06_smoke.run_entry('exp06_haa_finetune', [], receipt=receipt, max_gb=3)
    assert exit.value.code == 3
    record = json.loads(receipt.read_text())
    assert record['outcome'] == 'aborted_memory' and record['peak_bytes'] == 4 * 1024 ** 3
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 2 * 1024 ** 3)
    ok = exp06_smoke.run_entry('exp06_haa_finetune', [], max_gb=3)
    assert ok['exit_status'] == 0 and ok['peak_bytes'] == 2 * 1024 ** 3


def test_entry_failure_is_recorded_and_reraised(tmp_path, stub_entry, monkeypatch):
    def explode(argv):
        raise RuntimeError('boom')

    monkeypatch.setattr(stub_entry, 'main', explode)
    receipt = tmp_path / 'error.json'
    with pytest.raises(RuntimeError, match='boom'):
        exp06_smoke.run_entry('exp06_haa_finetune', [], receipt=receipt)
    recorded = json.loads(receipt.read_text())
    assert recorded['outcome'] == 'failed'
    assert recorded['outcome_detail'] == 'error: RuntimeError'


def test_trainer_entry_is_driven_through_sys_argv(monkeypatch):
    calls = {}
    module = types.ModuleType('train_xRIR_backbone')
    module.__file__ = 'train_xRIR_backbone.py'
    module.main = lambda: calls.setdefault('argv', list(sys.argv))
    monkeypatch.setitem(sys.modules, 'train_xRIR_backbone', module)
    before = list(sys.argv)
    result = exp06_smoke.run_entry('trainer', ['--backbone', 'simple', '--no-save'])
    assert calls['argv'][1:] == ['--backbone', 'simple', '--no-save']
    assert sys.argv == before and result['module'] == 'train_xRIR_backbone'


def test_unknown_entry_is_refused():
    with pytest.raises(ValueError, match='entry'):
        exp06_smoke.run_entry('invented', [])
    assert set(exp06_smoke.ENTRIES) == {'trainer', 'exp06_train', 'exp06_haa_finetune',
                                        'exp06_haa_eval'}
    assert exp06_smoke.ENTRIES['exp06_train'] == 'tools.exp06_train'


def test_fixture_loads_strictly_into_the_oriented_model(tmp_path):
    path = tmp_path / 'fixture_cylor.pth'
    record = exp06_smoke.make_fixture(path, seed=0)
    state = torch.load(str(path), map_location='cpu')
    build_xrir_exp06('cylindrical_oriented', 8).load_state_dict(state, strict=True)
    assert record['sha256'] == provenance.sha256_file(path)
    assert record['registry_sha256'] == exp06_train.registry_sha256()
    assert record == json.loads(Path(str(path) + '.json').read_text())
    assert record['backbone'] == 'cylindrical_oriented' and record['diagnostic'] is True
    torch.manual_seed(0)
    expected = build_xrir_exp06('cylindrical_oriented', 8).state_dict()
    assert all(torch.equal(value, expected[key]) for key, value in state.items())
    with pytest.raises(RuntimeError):
        build_xrir_exp06('cylindrical', 8).load_state_dict(state, strict=True)


def test_cli_prints_one_smoke_line_and_supports_the_fixture(tmp_path, stub_entry, capsys):
    argv = ['--entry', 'exp06_haa_finetune', '--receipt', str(tmp_path / 'r.json'),
            '--alarm-seconds', '30', '--', '--rooms', 'hallway']
    assert exp06_smoke.main(argv) == 0
    printed = [line for line in capsys.readouterr().out.splitlines() if line.startswith('EXP06_SMOKE ')]
    assert len(printed) == 1
    assert json.loads(printed[0][len('EXP06_SMOKE '):])['argv'] == ['--rooms', 'hallway']
    assert stub_entry.calls == [['--rooms', 'hallway']]
    assert exp06_smoke.main(['--make-fixture', str(tmp_path / 'f.pth')]) == 0
    assert (tmp_path / 'f.pth').is_file() and (tmp_path / 'f.pth.json').is_file()
    with pytest.raises(SystemExit):
        exp06_smoke.main(['--', '--rooms', 'hallway'])


@pytest.mark.parametrize('budget,value', [
    ('alarm_seconds', 0), ('alarm_seconds', -1), ('alarm_seconds', float('inf')),
    ('alarm_seconds', float('nan')), ('alarm_seconds', True), ('alarm_seconds', '300'),
    ('max_gb', 0), ('max_gb', -3), ('max_gb', float('inf')), ('max_gb', float('nan')),
    ('max_gb', None)])
def test_non_positive_or_non_finite_budgets_are_refused(tmp_path, stub_entry, budget, value):
    """Should-fix 7: a zero alarm silently disabled the timer; every budget is now checked."""
    receipt = tmp_path / 'budget.json'
    with pytest.raises(ValueError, match=budget):
        exp06_smoke.run_entry('exp06_haa_finetune', [], receipt=receipt, **{budget: value})
    assert not receipt.exists() and stub_entry.calls == []


def test_entry_returning_a_nonzero_status_fails_the_smoke(tmp_path, stub_entry, monkeypatch):
    """Should-fix 7: a status the entry returns must not be recorded as a success."""
    monkeypatch.setattr(stub_entry, 'main', lambda argv: 7)
    receipt = tmp_path / 'status.json'
    with pytest.raises(SystemExit) as exit:
        exp06_smoke.run_entry('exp06_haa_finetune', [], receipt=receipt, alarm_seconds=30)
    assert exit.value.code == 3
    record = json.loads(receipt.read_text())
    assert record['exit_status'] == 7 and record['outcome'] == 'failed'
    assert record['entry_status'] == 7 and record['aborted_memory'] is False


def test_memory_ceiling_aborts_during_execution(tmp_path, stub_entry, monkeypatch):
    """Should-fix 7: the ceiling is enforced while the entry runs, not only afterwards."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 4 * 1024 ** 3)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(stub_entry, 'main', lambda argv: time.sleep(60))
    receipt = tmp_path / 'ceiling.json'
    with pytest.raises(SystemExit) as exit:
        exp06_smoke.run_entry('exp06_haa_finetune', [], receipt=receipt, alarm_seconds=60, max_gb=3)
    assert exit.value.code == 3
    record = json.loads(receipt.read_text())
    assert record['outcome'] == 'aborted_memory' and record['aborted_memory'] is True
    assert record['exit_status'] == 3 and record['wall_s'] < 30, 'the entry ran to its alarm'


def test_a_successful_smoke_records_a_zero_status(tmp_path, stub_entry):
    """Should-fix 5 reads exit_status == 0 off the receipt, so success must be numeric."""
    record = exp06_smoke.run_entry('exp06_haa_finetune', ['--no-save'],
                                   receipt=tmp_path / 'ok.json', alarm_seconds=30)
    assert record['exit_status'] == 0 and record['outcome'] == 'ok'
    assert record['aborted_memory'] is False and record['diagnostic'] is True


REQUIRED = ('runner', 'runner_closure_sha256', 'entry', 'module', 'argv', 'run_type',
            'started_at', 'ended_at', 'wall_s', 'peak_bytes', 'alarm_seconds', 'max_gb',
            'outcome', 'exit_status', 'exploratory', 'git_head', 'diagnostic')


def test_the_receipt_carries_its_runner_budgets_and_timing(tmp_path, stub_entry):
    """Finding 6: a diagnostic must prove who ran what, for how long, and how big."""
    import datetime
    record = exp06_smoke.run_entry('exp06_haa_finetune', ['--no-save'],
                                   receipt=tmp_path / 'ok.json', alarm_seconds=30,
                                   run_type='probe')
    assert set(REQUIRED) <= set(record)
    assert record['runner'] == 'tools.exp06_smoke'
    assert len(record['runner_closure_sha256']) == 64
    assert record['run_type'] == 'probe' and record['exploratory'] is False
    assert record['outcome'] == 'ok' and record['alarm_seconds'] == 30.0
    assert isinstance(record['peak_bytes'], int) and record['peak_bytes'] >= 0
    started = datetime.datetime.fromisoformat(record['started_at'])
    ended = datetime.datetime.fromisoformat(record['ended_at'])
    assert started.tzinfo is not None and ended >= started
    assert record['wall_s'] >= 0


def test_a_diagnostic_writes_its_own_provenance(tmp_path, stub_entry):
    """Finding 6: smoke and probe bind their code the way the full run does."""
    from tools import exp06_profiles
    out = tmp_path / 'run' / 'provenance.json'
    out.parent.mkdir()
    record = exp06_smoke.run_entry(
        'exp06_haa_finetune', ['--no-save'], receipt=tmp_path / 'r.json', alarm_seconds=30,
        run_type='smoke', provenance_out=str(out), exploratory=True)
    written = json.loads(out.read_text())
    assert written['run_type'] == 'smoke' and written['exploratory'] is True
    assert written['source_closures']['diagnostic']['entry_module'] == 'tools.exp06_smoke'
    assert set(written['code_digests']) == set(exp06_profiles.TRAINING_KEYS)
    assert set(written['orchestration_closures']) == {'launcher', 'finalizer'}
    assert written['git_state']['HEAD'] == record['git_head']
    assert record['runner_closure_sha256'] == written['source_closures']['diagnostic']['sha256']
    assert record['provenance']['sha256'] == provenance.sha256_file(out)
    assert record['exploratory'] is True


def test_the_cli_accepts_the_diagnostic_admission_flags(tmp_path, stub_entry, capsys):
    argv = ['--entry', 'exp06_haa_finetune', '--receipt', str(tmp_path / 'r.json'),
            '--alarm-seconds', '30', '--run-type', 'probe', '--exploratory', '--',
            '--no-save']
    assert exp06_smoke.main(argv) == 0
    printed = [line for line in capsys.readouterr().out.splitlines()
               if line.startswith('EXP06_SMOKE ')]
    record = json.loads(printed[0][len('EXP06_SMOKE '):])
    assert record['run_type'] == 'probe' and record['exploratory'] is True
    with pytest.raises(SystemExit):
        exp06_smoke.main(['--entry', 'exp06_haa_finetune', '--run-type', 'full', '--',
                          '--no-save'])


def test_a_run_that_outlasted_its_alarm_is_published_as_an_abort(tmp_path):
    """Finding 4: the watchdog ticks once a second, so a late return must still abort.

    The finalizer refuses an ``ok`` receipt whose wall time is over its budget; the
    runner and the receipt therefore have to agree on that retrospectively, exactly as
    they already do for the memory ceiling.
    """
    result = dict(exit_status=0, outcome='ok', alarm_seconds=0.001, max_gb=3.0)
    published = exp06_smoke._publish(result, time.monotonic() - 1.0, tmp_path / 'late.json')
    assert published['outcome'] == 'aborted_alarm' and published['exit_status'] == 3
    assert published['wall_s'] >= 1.0
    assert json.loads((tmp_path / 'late.json').read_text())['outcome'] == 'aborted_alarm'
