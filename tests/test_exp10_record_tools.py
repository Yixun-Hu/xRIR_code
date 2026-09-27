"""exp_10 yaw_pilot — tests for the Planner's record tooling (Codex tooling review, findings 1-13).

The tools under test are the record generators and launch scripts under
``worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/``.  They are
loaded from their real paths (they are not importable modules of the package), and every
test runs on a synthetic evidence tree from ``tests/exp10_record_fixture.py``: no GPU, no
model, no checkpoint and nothing read from the live ``ckpt/`` tree.

Each test names the review finding it reproduces, so a regression points at the finding.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys

import pytest

from tests import exp10_record_fixture as fx

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(REPO, "worklog", "worklog_yixun", "exp_10_yaw_pilot_claude",
                      "yaw_pilot_results_assets")
SCRIPTS = os.path.join(ASSETS, "scripts")


def load_tool(name):
    """Import one of the record generators from its path in the record directory."""
    path = os.path.join(ASSETS, name + ".py")
    spec = importlib.util.spec_from_file_location("exp10_record_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def validate():
    return load_tool("validate_runs")


def run_python(path, *args, **kwargs):
    """Run one of the generators as a subprocess; returns the CompletedProcess."""
    env = dict(os.environ)
    env.update(kwargs.pop("env", {}) or {})
    return subprocess.run([sys.executable, path] + [str(a) for a in args],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          env=env, cwd=kwargs.pop("cwd", None) or REPO)


def out(completed):
    return completed.stdout.decode("utf-8", "replace")


# --------------------------------------------------------------------------------------
# Finding 1 -- the finish preflight has to require, validate and bind the run evidence.
# --------------------------------------------------------------------------------------

@pytest.fixture
def tree(tmp_path):
    """A complete four-arm record tree plus the CPU-protocol record (small arrays)."""
    root = str(tmp_path / "exp10")
    os.makedirs(root)
    built = fx.write_record_tree(root, n_queries=32)
    return root, built


def validate_tree(validate, root, **kwargs):
    kwargs.setdefault("expect_n", 32)
    return validate.validate_record(root, **kwargs)


def test_complete_record_validates_and_binds_every_arm(validate, tree):
    """Finding 1: the happy path passes and records the identities it bound."""
    root, built = tree
    report = validate_tree(validate, root)
    assert report["ok"] is True, report["problems"]
    assert sorted(report["arms"]) == sorted(fx.ARMS)
    for arm in fx.ARMS:
        entry = report["arms"][arm]
        assert entry["full"]["execution_id"] == built[arm]["all"][1]["execution_id"]
        assert entry["full"]["protocol_id"] == built[arm]["all"][1]["protocol_id"]
        assert entry["full"]["per_sample_sha256"] == built[arm]["all"][1]["per_sample_sha256"]
        assert entry["probe"]["execution_id"] == built[arm]["probe"][1]["execution_id"]
        assert entry["check_online"]["ok"] is True
        assert entry["check_online"]["bound_by"]
        assert entry["probe_summary"]["controls_ok"] is True
        if arm in fx.PARITY_ARMS:
            assert entry["parity_all"]["ok"] is True
            assert entry["parity_probe"]["ok"] is True
        else:
            assert entry["parity_all"] is None
    assert report["cpu"]["full"]["device"] == "cpu"
    assert report["cpu"]["parity_all"]["ok"] is False


def test_missing_full_run_is_refused(validate, tree):
    """Finding 1: the blocker -- a missing full arm must not reach FINISH DONE."""
    root, built = tree
    os.rename(os.path.join(root, "released_k8_all"), os.path.join(root, "released_k8_all.moved"))
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("released_k8_all" in p and "meta.json" in p for p in report["problems"])


def test_incomplete_full_run_is_refused(validate, tree):
    root, built = tree
    run_dir, meta = built["cyl_k8"]["all"]
    meta["complete"] = False
    fx.write_json(os.path.join(run_dir, "meta.json"), meta)
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("complete" in p for p in report["problems"])


def test_wrong_query_count_is_refused(validate, tree):
    root, _ = tree
    report = validate_tree(validate, root, expect_n=6337)
    assert report["ok"] is False
    assert any("n_queries" in p for p in report["problems"])


def test_cpu_device_in_a_gpu_arm_is_refused(validate, tree):
    root, built = tree
    run_dir, meta = built["control_k8"]["all"]
    meta["device"] = "cpu"
    fx.write_json(os.path.join(run_dir, "meta.json"), meta)
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("device" in p for p in report["problems"])


def test_unbound_per_sample_file_is_refused(validate, tree):
    """Finding 1: evidence must be bound -- an edited per_sample.json breaks the run."""
    root, built = tree
    run_dir, _ = built["control_k8"]["all"]
    with open(os.path.join(run_dir, "per_sample.json"), "a") as fout:
        fout.write("\n")
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("per_sample.json" in p for p in report["problems"])


def test_failed_check_online_is_refused(validate, tree):
    root, built = tree
    run_dir, meta = built["cyl_k8"]["all"]
    fx.write_check_online(run_dir, meta, ok=False)
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("check_online" in p and "ok" in p for p in report["problems"])


def test_check_online_naming_another_run_is_refused(validate, tree):
    """Finding 1: the reproduction reached FINISH DONE with online reports naming another run."""
    root, built = tree
    run_dir, meta = built["cyl_k8"]["all"]
    other = built["control_k8"]["all"][0]
    fx.write_check_online(run_dir, meta, ok=True, run_dir_value=os.path.abspath(other))
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("check_online" in p and "run_dir" in p for p in report["problems"])


def test_missing_check_online_is_refused(validate, tree):
    root, built = tree
    os.remove(os.path.join(built["released_k1"]["all"][0], "check_online.json"))
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("check_online.json" in p and "missing" in p for p in report["problems"])


def test_failed_probe_controls_are_refused(validate, tree):
    """Finding 1: probe controls were never checked; a failing probe must block the record."""
    root, built = tree
    probe_dir, probe_meta = built["released_k1"]["probe"]
    fx.write_summary_dir(os.path.join(probe_dir, "summary"), [(probe_dir, probe_meta)],
                         controls_ok=False)
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("controls" in p for p in report["problems"])


def test_probe_summary_of_another_execution_is_refused(validate, tree):
    root, built = tree
    probe_dir, _ = built["cyl_k8"]["probe"]
    other_dir, other_meta = built["control_k8"]["probe"]
    fx.write_summary_dir(os.path.join(probe_dir, "summary"), [(other_dir, other_meta)])
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("probe" in p.lower() for p in report["problems"])


def test_missing_probe_meta_is_refused(validate, tree):
    root, built = tree
    os.remove(os.path.join(built["control_k8"]["probe"][0], "meta.json"))
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("control_k8_probe" in p for p in report["problems"])


def test_missing_full_parity_is_refused_for_the_k8_arms(validate, tree):
    """Finding 1: full parity was optional; for K = 8 both probe and full parity are required."""
    root, built = tree
    os.remove(os.path.join(built["control_k8"]["all"][0], "parity_exp03.json"))
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("parity_exp03.json" in p for p in report["problems"])


def test_missing_probe_parity_is_refused_for_the_k8_arms(validate, tree):
    root, built = tree
    os.remove(os.path.join(built["cyl_k8"]["probe"][0], "parity_exp03.json"))
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("parity_exp03.json" in p for p in report["problems"])


def test_failed_parity_is_refused_for_the_k8_arms(validate, tree):
    root, built = tree
    run_dir, meta = built["released_k8"]["all"]
    fx.write_parity(run_dir, meta, arm="released_k8", ok=False)
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("parity" in p and "ok" in p for p in report["problems"])


def test_parity_naming_another_run_is_refused(validate, tree):
    root, built = tree
    run_dir, meta = built["released_k8"]["all"]
    fx.write_parity(run_dir, meta, arm="released_k8",
                    run_dir_value=os.path.abspath(built["cyl_k8"]["all"][0]))
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("parity" in p and "run_dir" in p for p in report["problems"])


def test_released_k1_needs_no_parity(validate, tree):
    """Finding 1: released_k1 has no exp_03 predecessor -- parity is not required there."""
    root, built = tree
    assert not os.path.exists(os.path.join(built["released_k1"]["all"][0], "parity_exp03.json"))
    report = validate_tree(validate, root)
    assert report["ok"] is True, report["problems"]


def test_released_k1_parity_is_still_validated_when_present(validate, tree):
    root, built = tree
    run_dir, meta = built["released_k1"]["all"]
    fx.write_parity(run_dir, meta, arm="released_k8", ok=False)
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("released_k1" in p and "parity" in p for p in report["problems"])


def test_two_arms_sharing_one_execution_are_refused(validate, tree):
    """Finding 1: the same execution presented twice is not four arms of evidence."""
    root, built = tree
    run_dir, meta = built["cyl_k8"]["all"]
    meta["execution_id"] = built["control_k8"]["all"][1]["execution_id"]
    fx.write_json(os.path.join(run_dir, "meta.json"), meta)
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("execution_id" in p for p in report["problems"])


def test_cpu_record_must_be_a_cpu_run(validate, tree):
    root, built = tree
    run_dir, meta = built["cpu"]["all"]
    meta["device"] = "cuda"
    fx.write_json(os.path.join(run_dir, "meta.json"), meta)
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("cpu" in p.lower() and "device" in p for p in report["problems"])


def test_cpu_record_keeps_its_documented_parity_failure(validate, tree):
    """Finding 1: the CPU record's parity is recorded, not required to be ok."""
    root, built = tree
    report = validate_tree(validate, root)
    assert report["ok"] is True, report["problems"]
    assert report["cpu"]["parity_all"]["ok"] is False


def test_cpu_record_without_parity_is_refused(validate, tree):
    root, built = tree
    os.remove(os.path.join(built["cpu"]["all"][0], "parity_exp03.json"))
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("cpu" in p.lower() and "parity" in p for p in report["problems"])


def test_cpu_record_failed_check_online_is_refused(validate, tree):
    root, built = tree
    run_dir, meta = built["cpu"]["all"]
    fx.write_check_online(run_dir, meta, ok=False)
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("check_online" in p for p in report["problems"])


def test_truncated_waveform_array_is_refused(validate, tree):
    """Finding 1: meta.arrays is a binding -- a short array is not the run's output."""
    root, built = tree
    run_dir, _ = built["control_k8"]["all"]
    path = os.path.join(run_dir, "wav_k64.npy")
    with open(path, "r+b") as fout:
        fout.truncate(64)
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("wav_k64.npy" in p for p in report["problems"])


def test_missing_array_binding_is_refused(validate, tree):
    root, built = tree
    run_dir, meta = built["control_k8"]["all"]
    meta["arrays"].pop("wav_k128.npy")
    fx.write_json(os.path.join(run_dir, "meta.json"), meta)
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("wav_k128.npy" in p for p in report["problems"])


def test_full_array_check_detects_an_edited_array_of_the_same_size(validate, tree):
    """The default check is by size; ``--array-check full`` rehashes the bytes."""
    root, built = tree
    run_dir, _ = built["cyl_k8"]["all"]
    path = os.path.join(run_dir, "wav_k256.npy")
    with open(path, "r+b") as fout:
        fout.seek(os.path.getsize(path) - 4)
        fout.write(b"\x00\x00\x00\x00")
    assert validate_tree(validate, root, array_check="size")["ok"] is True
    report = validate_tree(validate, root, array_check="full")
    assert report["ok"] is False
    assert any("wav_k256.npy" in p and "sha256" in p for p in report["problems"])


def test_cli_reports_and_exits_nonzero_on_a_broken_record(validate, tree):
    root, built = tree
    report_path = os.path.join(root, "validation.json")
    ok = run_python(os.path.join(ASSETS, "validate_runs.py"), "--root", root,
                    "--expect-n", 32, "--json", report_path)
    assert ok.returncode == 0, out(ok)
    assert json.load(open(report_path))["ok"] is True
    os.remove(os.path.join(built["cyl_k8"]["all"][0], "check_online.json"))
    bad = run_python(os.path.join(ASSETS, "validate_runs.py"), "--root", root,
                     "--expect-n", 32, "--json", report_path)
    assert bad.returncode != 0
    assert "check_online.json" in out(bad)
    assert json.load(open(report_path))["ok"] is False


def test_relocated_cpu_reports_are_refused_without_a_declaration(validate, tree):
    """Finding 1: exp_10's CPU record was moved into cpu_protocol/ after its reports were
    written, so they name the pre-move path -- silently accepting that would also accept a
    report naming any other directory."""
    root, built = tree
    run_dir, meta = built["cpu"]["all"]
    moved_from = os.path.join(root, "released_k8_all_cpu_original")
    fx.write_check_online(run_dir, meta, ok=True, run_dir_value=moved_from)
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("cpu" in p.lower() and "run_dir" in p for p in report["problems"])


def test_declared_cpu_relocation_is_accepted_and_recorded(validate, tree):
    root, built = tree
    run_dir, meta = built["cpu"]["all"]
    moved_from = os.path.join(root, "released_k8_all")      # same basename: the same run
    fx.write_check_online(run_dir, meta, ok=True, run_dir_value=moved_from)
    fx.write_parity(run_dir, meta, arm="released_k8", ok=False, run_dir_value=moved_from)
    report = validate_tree(validate, root, cpu_recorded_run_dir=moved_from)
    assert report["ok"] is True, report["problems"]
    assert report["cpu"]["check_online"]["relocated_from"] == moved_from
    assert "run_dir (declared relocation)" in report["cpu"]["check_online"]["bound_by"]


def test_a_declaration_does_not_admit_another_runs_report(validate, tree):
    """The declared alternative only covers this run: a different basename is still refused."""
    root, built = tree
    run_dir, meta = built["cpu"]["all"]
    other = os.path.join(root, "control_k8_all")
    fx.write_check_online(run_dir, meta, ok=True, run_dir_value=other)
    report = validate_tree(validate, root, cpu_recorded_run_dir=other)
    assert report["ok"] is False
    assert any("run_dir" in p for p in report["problems"])
