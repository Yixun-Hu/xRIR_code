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
import shlex
import shutil
import subprocess
import sys
import tempfile

import pytest

from tests import exp10_record_fixture as fx

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(REPO, "worklog", "worklog_yixun", "exp_10_yaw_pilot_claude",
                      "yaw_pilot_results_assets")


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


def set_in(entry, path, value):
    """Set ``entry[path[0]][path[1]]... = value`` (the tests edit one nested field)."""
    target = entry
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


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


@pytest.mark.parametrize("field,value", [
    (("protocol_id",), "not-this-protocol"),
    (("n_queries",), 7),
    (("meta", "per_sample_sha256"), "0" * 64),
    (("meta", "arrays", "wav_k64.npy", "sha256"), "1" * 64),
])
def test_a_probe_summary_payload_contradicting_the_probe_run_is_refused(validate, tree,
                                                                       field, value):
    """Round-5 finding 2: the preflight bound the probe summary's inputs but never compared
    the *rendered* arm payload with them or with the live probe meta."""
    root, built = tree
    probe_dir, _ = built["control_k8"]["probe"]
    path = os.path.join(probe_dir, "summary", "yaw_pilot_summary.json")
    summary = fx.read_json(path)
    set_in(summary["arms"][0], field, value)
    fx.write_json(path, summary)
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any(field[-1] in p for p in report["problems"]), report["problems"]


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


def test_cpu_reports_naming_another_directory_are_refused(validate, tree):
    """Finding 1 (round 4): every report must name its own run directory."""
    root, built = tree
    run_dir, meta = built["cpu"]["all"]
    elsewhere = os.path.join(root, "released_k8_all_cpu_original")
    fx.write_check_online(run_dir, meta, ok=True, run_dir_value=elsewhere)
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("cpu" in p.lower() and "run_dir" in p for p in report["problems"])


def test_cpu_reports_naming_the_gpu_arms_directory_are_refused(validate, tree):
    """Round-4 finding 1, the blocker: the relocation alias accepted a report naming
    ``<root>/released_k8_all``, which is now the *GPU* execution -- so the GPU arm's
    validation reports could be filed as the CPU record's and reach FINISH DONE."""
    root, built = tree
    run_dir, meta = built["cpu"]["all"]
    gpu_dir = built["released_k8"]["all"][0]         # same basename, another execution
    assert os.path.basename(gpu_dir) == os.path.basename(run_dir)
    fx.write_check_online(run_dir, meta, ok=True, run_dir_value=os.path.abspath(gpu_dir))
    fx.write_parity(run_dir, meta, arm="released_k8", ok=False,
                    run_dir_value=os.path.abspath(gpu_dir))
    report = validate_tree(validate, root)
    assert report["ok"] is False
    assert any("cpu" in p.lower() and "run_dir" in p for p in report["problems"])


def test_the_relocation_exception_is_gone(validate, tree):
    """Round-4 finding 1: no argument may waive the run_dir binding any more."""
    import inspect

    for function in (validate.validate_record, validate.validate_cpu_record,
                     validate.bind_report):
        names = list(inspect.signature(function).parameters)
        assert not [n for n in names if "reloc" in n or n == "also_accept"], names
    assert not hasattr(validate, "_declared_relocation")
    root, _ = tree
    done = run_python(os.path.join(ASSETS, "validate_runs.py"), "--root", root,
                      "--expect-n", 32, "--cpu-recorded-run-dir", root)
    assert done.returncode != 0
    assert "cpu-recorded-run-dir" in out(done)


# --------------------------------------------------------------------------------------
# Finding 3 -- the backend comparison must enforce matching protocols and populations.
# --------------------------------------------------------------------------------------

@pytest.fixture
def backend_pair(tmp_path):
    """A GPU and a CPU full run of released_k8 with identical protocol and query population."""
    root = str(tmp_path / "exp10")
    gpu_dir = os.path.join(root, "released_k8_all")
    cpu_dir = os.path.join(root, "cpu_protocol", "released_k8_all")
    gpu_meta = fx.write_run(gpu_dir, arm="released_k8", device="cuda", n_queries=32)
    cpu_meta = fx.write_run(cpu_dir, arm="released_k8", device="cpu", n_queries=32)
    gpu_summary, _ = fx.write_summary_dir(os.path.join(root, "summary"), [(gpu_dir, gpu_meta)])
    cpu_summary, _ = fx.write_summary_dir(os.path.join(root, "cpu_protocol", "summary"),
                                          [(cpu_dir, cpu_meta)])
    return {"root": root, "gpu": (gpu_dir, gpu_meta, gpu_summary),
            "cpu": (cpu_dir, cpu_meta, cpu_summary)}


def backend_table(pair, tmp_path, gpu=None, cpu=None):
    return run_python(os.path.join(ASSETS, "make_backend_table.py"),
                      "--gpu", gpu or pair["gpu"][2], "--cpu", cpu or pair["cpu"][2],
                      "--arm", "released_k8",
                      "--out-md", str(tmp_path / "backend.md"),
                      "--out-json", str(tmp_path / "backend.json"))


def test_backend_table_accepts_a_matched_gpu_cpu_pair(backend_pair, tmp_path):
    done = backend_table(backend_pair, tmp_path)
    assert done.returncode == 0, out(done)
    record = json.load(open(str(tmp_path / "backend.json")))
    assert record["gpu"]["device"] == "cuda" and record["cpu"]["device"] == "cpu"
    assert record["gpu"]["execution_id"] != record["cpu"]["execution_id"]
    assert record["verified"]["query_list_sha256"] == backend_pair["gpu"][1]["query_list_sha256"]
    assert "GPU (primary) vs CPU protocol" in open(str(tmp_path / "backend.md")).read()


def test_backend_table_refuses_the_same_execution_twice(backend_pair, tmp_path):
    """Finding 3: supplying the same CPU execution twice was accepted."""
    done = backend_table(backend_pair, tmp_path, gpu=backend_pair["cpu"][2])
    assert done.returncode != 0
    assert "device" in out(done) or "execution" in out(done)


def test_backend_table_refuses_a_probe_against_a_full_run(backend_pair, tmp_path):
    """Finding 3: a 272-query probe was compared against a 6337-query full run."""
    root = backend_pair["root"]
    probe_dir = os.path.join(root, "released_k8_probe")
    probe_meta = fx.write_run(probe_dir, arm="released_k8", device="cuda", n_queries=8,
                              batches_arg="probe")
    probe_summary, _ = fx.write_summary_dir(os.path.join(probe_dir, "summary"),
                                            [(probe_dir, probe_meta)])
    done = backend_table(backend_pair, tmp_path, gpu=probe_summary)
    assert done.returncode != 0
    assert "n_queries" in out(done) or "query" in out(done)


def test_backend_table_refuses_a_batch_size_mismatch(backend_pair, tmp_path):
    cpu_dir, cpu_meta, cpu_summary = backend_pair["cpu"]
    summary = fx.read_json(cpu_summary)
    summary["arms"][0]["meta"]["batch_size"] = 8
    fx.write_json(cpu_summary, summary)
    done = backend_table(backend_pair, tmp_path)
    assert done.returncode != 0
    assert "batch_size" in out(done)


def test_backend_table_refuses_a_different_implementation(backend_pair, tmp_path):
    cpu_dir, cpu_meta, cpu_summary = backend_pair["cpu"]
    summary = fx.read_json(cpu_summary)
    summary["arms"][0]["meta"]["tool_sha256"] = "f" * 64
    fx.write_json(cpu_summary, summary)
    done = backend_table(backend_pair, tmp_path)
    assert done.returncode != 0
    assert "tool_sha256" in out(done)


def test_backend_table_refuses_a_different_query_population(backend_pair, tmp_path):
    cpu_dir, cpu_meta, cpu_summary = backend_pair["cpu"]
    summary = fx.read_json(cpu_summary)
    summary["arms"][0]["meta"]["query_list_sha256"] = "a" * 64
    fx.write_json(cpu_summary, summary)
    done = backend_table(backend_pair, tmp_path)
    assert done.returncode != 0
    assert "query_list_sha256" in out(done)


def test_backend_table_refuses_an_unbound_summary(backend_pair, tmp_path):
    """Finding 3: each summary's inputs[].per_sample_sha256 must match the live run."""
    cpu_dir = backend_pair["cpu"][0]
    with open(os.path.join(cpu_dir, "per_sample.json"), "a") as fout:
        fout.write("\n")
    done = backend_table(backend_pair, tmp_path)
    assert done.returncode != 0
    assert "per_sample.json" in out(done)


@pytest.mark.parametrize("field,value", [("alpha", 0.5), ("n_boot", 200),
                                         ("seeds", [0, 7]),
                                         ("convergence_tolerance", 0.9)])
def test_backend_table_refuses_different_statistical_settings(backend_pair, tmp_path,
                                                              field, value):
    """Round-4 finding 5: the runs' metadata was compared but the *summaries'* statistical
    settings were not.  On identical observations the canonical summariser gave "denominator
    uncertain" at alpha .05 and "defined" at alpha .5, and the table printed both while
    claiming only the inference device differed."""
    cpu_summary = backend_pair["cpu"][2]
    summary = fx.read_json(cpu_summary)
    summary[field] = value
    fx.write_json(cpu_summary, summary)
    done = backend_table(backend_pair, tmp_path)
    assert done.returncode != 0
    assert field in out(done)


def test_backend_table_records_the_statistical_settings_it_verified(backend_pair, tmp_path):
    """Finding 5: they are part of what the table certifies, so they are in the record."""
    done = backend_table(backend_pair, tmp_path)
    assert done.returncode == 0, out(done)
    record = json.load(open(str(tmp_path / "backend.json")))
    gpu = fx.read_json(backend_pair["gpu"][2])
    for field in ("alpha", "n_boot", "seeds", "convergence_tolerance"):
        assert record["verified"][field] == gpu[field], field
    caption = open(str(tmp_path / "backend.md")).read()
    assert "alpha %s" % gpu["alpha"] in caption
    assert "convergence tolerance %s" % gpu["convergence_tolerance"] in caption


def test_backend_table_refuses_summaries_that_record_no_statistics(backend_pair, tmp_path):
    for role in ("gpu", "cpu"):
        path = backend_pair[role][2]
        summary = fx.read_json(path)
        summary.pop("alpha")
        fx.write_json(path, summary)
    done = backend_table(backend_pair, tmp_path)
    assert done.returncode != 0
    assert "alpha" in out(done)


def test_backend_table_refuses_an_embedded_meta_that_is_not_the_runs(backend_pair, tmp_path):
    """The summary's embedded meta is compared to the live meta.json of the run it names."""
    cpu_summary = backend_pair["cpu"][2]
    summary = fx.read_json(cpu_summary)
    summary["arms"][0]["meta"]["gl_seed"] = 7
    fx.write_json(cpu_summary, summary)
    done = backend_table(backend_pair, tmp_path)
    assert done.returncode != 0
    assert "gl_seed" in out(done)


# --------------------------------------------------------------------------------------
# Finding 4 -- the renderers must bind their summary, supplements and figures.
# --------------------------------------------------------------------------------------

RENDER_ARMS = ("control_k8", "cyl_k8")


@pytest.fixture
def render_case(tmp_path):
    """A two-arm canonical summary with each arm's probe summary, parity and online report."""
    root = str(tmp_path / "exp10")
    tree = fx.write_record_tree(root, arms=RENDER_ARMS, n_queries=32, cpu=False)
    runs = [tree[arm]["all"] for arm in RENDER_ARMS]
    summary_dir = os.path.join(root, "summary")
    summary_path, summary = fx.write_summary_dir(summary_dir, runs)
    case = {"root": root, "tree": tree, "summary": summary_path, "summary_dir": summary_dir,
            "out_dir": str(tmp_path / "page"), "assets": str(tmp_path / "page" / "generated")}
    os.makedirs(case["out_dir"])
    case["parity"] = ["%s=%s" % (arm, os.path.join(tree[arm]["all"][0], "parity_exp03.json"))
                      for arm in RENDER_ARMS]
    case["online"] = ["%s=%s" % (arm, os.path.join(tree[arm]["all"][0], "check_online.json"))
                      for arm in RENDER_ARMS]
    case["probe"] = ["%s=%s" % (arm, os.path.join(tree[arm]["probe"][0], "summary",
                                                  "yaw_pilot_summary.json"))
                     for arm in RENDER_ARMS]
    return case


def render_html(case, parity=None, online=None, probe=None, extra=()):
    args = ["--summary", case["summary"], "--out", os.path.join(case["out_dir"], "page.html"),
            "--assets", case["assets"]]
    for flag, values in (("--parity", parity if parity is not None else case["parity"]),
                         ("--check-online", online if online is not None else case["online"]),
                         ("--probe", probe if probe is not None else case["probe"])):
        if values:
            args += [flag] + list(values)
    return run_python(os.path.join(ASSETS, "make_results_html.py"), *(args + list(extra)))


def render_md(case, parity=None, online=None, probe=None, extra=()):
    args = ["--summary", case["summary"], "--out", os.path.join(case["out_dir"], "page.md")]
    for flag, values in (("--parity", parity if parity is not None else case["parity"]),
                         ("--check-online", online if online is not None else case["online"]),
                         ("--probe", probe if probe is not None else case["probe"])):
        if values:
            args += [flag] + list(values)
    return run_python(os.path.join(ASSETS, "make_results_md.py"), *(args + list(extra)))


def test_both_renderers_render_the_happy_path(render_case):
    assert render_html(render_case).returncode == 0
    assert render_md(render_case).returncode == 0
    assert os.path.isfile(os.path.join(render_case["out_dir"], "page.html"))
    assert os.path.isfile(os.path.join(render_case["out_dir"], "page.md"))


@pytest.mark.parametrize("renderer", [render_html, render_md])
def test_a_summary_whose_input_hash_fails_is_refused(render_case, renderer):
    """Finding 4: both renderers accepted a deliberately incorrect summary input hash."""
    summary = fx.read_json(render_case["summary"])
    summary["inputs"][0]["per_sample_sha256"] = "0" * 64
    fx.write_json(render_case["summary"], summary)
    done = renderer(render_case)
    assert done.returncode != 0
    assert "per_sample.json" in out(done)


@pytest.mark.parametrize("renderer", [render_html, render_md])
def test_parity_attached_to_the_wrong_arm_is_refused(render_case, renderer):
    """Finding 4: attaching the cylindrical parity report to the control arm succeeded."""
    wrong = ["control_k8=%s" % os.path.join(render_case["tree"]["cyl_k8"]["all"][0],
                                            "parity_exp03.json")]
    done = renderer(render_case, parity=wrong)
    assert done.returncode != 0
    assert "parity" in out(done) and "control_k8" in out(done)


@pytest.mark.parametrize("renderer", [render_html, render_md])
def test_check_online_attached_to_the_wrong_arm_is_refused(render_case, renderer):
    wrong = ["cyl_k8=%s" % os.path.join(render_case["tree"]["control_k8"]["all"][0],
                                        "check_online.json")]
    done = renderer(render_case, online=wrong)
    assert done.returncode != 0
    assert "check_online" in out(done)


@pytest.mark.parametrize("renderer", [render_html, render_md])
def test_a_supplement_for_an_arm_outside_the_summary_is_refused(render_case, renderer):
    wrong = ["released_k8=%s" % os.path.join(render_case["tree"]["cyl_k8"]["all"][0],
                                             "check_online.json")]
    done = renderer(render_case, online=wrong)
    assert done.returncode != 0
    assert "released_k8" in out(done)


@pytest.mark.parametrize("renderer", [render_html, render_md])
def test_a_full_run_summary_passed_as_a_probe_is_refused(render_case, renderer):
    """Finding 4: the probe controls must come from a probe run of that arm."""
    done = renderer(render_case, probe=["control_k8=%s" % render_case["summary"]])
    assert done.returncode != 0
    assert "probe" in out(done)


@pytest.mark.parametrize("renderer", [render_html, render_md])
def test_another_arms_probe_summary_is_refused(render_case, renderer):
    wrong = ["control_k8=%s" % os.path.join(render_case["tree"]["cyl_k8"]["probe"][0],
                                            "summary", "yaw_pilot_summary.json")]
    done = renderer(render_case, probe=wrong)
    assert done.returncode != 0
    assert "control_k8" in out(done)


@pytest.mark.parametrize("renderer", [render_html, render_md])
@pytest.mark.parametrize("kind", ["parity_exp03", "check_online"])
@pytest.mark.parametrize("field", list(("execution_id", "protocol_id", "per_sample_sha256")))
def test_a_supplement_contradicting_its_runs_identity_is_refused(render_case, renderer,
                                                                 kind, field):
    """Round-4 finding 4: a report with the correct path but a deliberately wrong
    execution_id / protocol_id / per_sample_sha256 was rendered.  The supplements were
    bound by ``run_dir`` alone; every identity the report carries is now compared with the
    meta of the run it names (the reports carry ``run_dir`` only, so that is where the
    identity comes from)."""
    path = os.path.join(render_case["tree"]["cyl_k8"]["all"][0], "%s.json" % kind)
    report = fx.read_json(path)
    report[field] = "wrong-execution-or-hash"
    fx.write_json(path, report)
    done = renderer(render_case)
    assert done.returncode != 0
    assert field in out(done)


@pytest.mark.parametrize("renderer", [render_html, render_md])
def test_a_summary_missing_an_input_for_a_rendered_arm_is_refused(render_case, renderer):
    """Round-4 finding 4: keeping only the control's input binding while still rendering
    every arm succeeded in both renderers."""
    summary = fx.read_json(render_case["summary"])
    summary["inputs"] = [i for i in summary["inputs"] if i["arm"] == "control_k8"]
    fx.write_json(render_case["summary"], summary)
    done = renderer(render_case)
    assert done.returncode != 0
    assert "cyl_k8" in out(done)


@pytest.mark.parametrize("renderer", [render_html, render_md])
def test_a_summary_binding_one_arm_twice_is_refused(render_case, renderer):
    """The other half of "exactly one": two entries claiming an arm bind none of them."""
    summary = fx.read_json(render_case["summary"])
    summary["inputs"].append(dict(summary["inputs"][0]))
    fx.write_json(render_case["summary"], summary)
    done = renderer(render_case)
    assert done.returncode != 0
    assert "control_k8" in out(done)


@pytest.mark.parametrize("renderer", [render_html, render_md])
def test_a_probe_summary_of_another_execution_is_refused(render_case, renderer):
    """Finding 4: the probe controls are rendered as that probe *run's* -- so the probe
    summary's own arm entry has to bind the meta of the run it names."""
    arm = "control_k8"
    path = os.path.join(render_case["tree"][arm]["probe"][0], "summary",
                        "yaw_pilot_summary.json")
    probe = fx.read_json(path)
    probe["arms"][0]["execution_id"] = "20260927T000000Z-not-the-probe-run"
    fx.write_json(path, probe)
    done = renderer(render_case)
    assert done.returncode != 0
    assert "execution" in out(done)


@pytest.mark.parametrize("renderer", [render_html, render_md])
def test_a_probe_summary_naming_a_full_run_is_refused(render_case, renderer):
    """Finding 4: the embedded ``meta.batches_arg`` is part of the summary, so it can say
    ``probe`` about a full run.  The run it names has to be a probe run of that arm."""
    arm = "control_k8"
    full_dir, full_meta = render_case["tree"][arm]["all"]
    forged = fx.build_summary([(full_dir, full_meta)])
    forged["arms"][0]["meta"]["batches_arg"] = "probe"
    path = os.path.join(render_case["out_dir"], "forged_probe_summary.json")
    fx.write_json(path, forged)
    done = renderer(render_case, probe=["%s=%s" % (arm, path)])
    assert done.returncode != 0
    assert "probe" in out(done)


def execution_b(case, arm, device="cpu"):
    """A second, independently bound run and summary of ``arm`` (another execution)."""
    other_dir = os.path.join(case["root"], "%s_all_execution_b" % arm)
    other_meta = fx.write_run(other_dir, arm=arm, device=device, n_queries=32)
    return other_dir, other_meta, fx.build_summary([(other_dir, other_meta)])


@pytest.mark.parametrize("renderer", [render_html, render_md])
def test_a_transplanted_arm_payload_is_refused(render_case, renderer):
    """Round-5 finding 2, the blocker: a *complete* ``arms[]`` payload transplanted from
    another independently bound summary rendered in both renderers, exit 0.  The page showed
    CPU execution B while the provenance and the online / parity evidence identified CUDA
    execution A: the verified ``inputs`` and the rendered payloads were never compared."""
    arm = "control_k8"
    _dir, _meta, foreign = execution_b(render_case, arm)
    summary = fx.read_json(render_case["summary"])
    bound = [i for i in summary["inputs"] if i["arm"] == arm][0]
    assert foreign["arms"][0]["execution_id"] != bound["execution_id"]
    summary["arms"] = [foreign["arms"][0] if a["arm"] == arm else a for a in summary["arms"]]
    fx.write_json(render_case["summary"], summary)
    done = renderer(render_case)
    assert done.returncode == 2, out(done)
    assert "execution_id" in out(done) and arm in out(done)


@pytest.mark.parametrize("renderer", [render_html, render_md])
@pytest.mark.parametrize("field,value", [
    (("execution_id",), "20260927T000000Z-not-this-execution"),
    (("protocol_id",), "not-this-protocol"),
    (("arm",), "released_k8"),
    (("n_queries",), 7),
    (("meta", "execution_id"), "20260927T000000Z-not-this-execution"),
    (("meta", "protocol_id"), "not-this-protocol"),
    (("meta", "batches_arg"), "probe"),
    (("meta", "n_queries"), 7),
    (("meta", "device"), "cpu"),
    (("meta", "per_sample_sha256"), "0" * 64),
    (("meta", "query_list_sha256"), "0" * 64),
    (("meta", "arrays", "wav_k64.npy", "sha256"), "1" * 64),
    (("meta", "arrays", "wav_k64.npy", "shape"), [7, 16]),
])
def test_a_rendered_arm_payload_that_contradicts_its_run_is_refused(render_case, renderer,
                                                                   field, value):
    """Every identity and digest the transplant carries, one at a time: the rendered payload
    has to be the one its unique ``inputs`` entry and the hash-verified live ``meta.json``
    describe.  ``arm`` renames the entry, so the payload then binds no input at all."""
    summary = fx.read_json(render_case["summary"])
    set_in(summary["arms"][0], field, value)
    fx.write_json(render_case["summary"], summary)
    done = renderer(render_case)
    assert done.returncode == 2, out(done)
    assert field[-1] in out(done), out(done)


@pytest.mark.parametrize("renderer", [render_html, render_md])
def test_a_rendered_arm_payload_missing_an_array_binding_is_refused(render_case, renderer):
    """The waveform digests are part of the payload: a summary that drops one no longer says
    which arrays its numbers came from."""
    summary = fx.read_json(render_case["summary"])
    del summary["arms"][0]["meta"]["arrays"]["wav_k128.npy"]
    fx.write_json(render_case["summary"], summary)
    done = renderer(render_case)
    assert done.returncode == 2, out(done)
    assert "wav_k128.npy" in out(done)


@pytest.mark.parametrize("renderer", [render_html, render_md])
def test_a_rendered_arm_payload_with_a_foreign_array_binding_is_refused(render_case, renderer):
    """... and one that binds an array the run does not have describes another run."""
    summary = fx.read_json(render_case["summary"])
    arrays = summary["arms"][0]["meta"]["arrays"]
    arrays["wav_k512.npy"] = dict(arrays["wav_k64.npy"], k=512)
    fx.write_json(render_case["summary"], summary)
    done = renderer(render_case)
    assert done.returncode == 2, out(done)
    assert "wav_k512.npy" in out(done)


@pytest.mark.parametrize("renderer", [render_html, render_md])
def test_a_probe_summary_whose_arm_payload_contradicts_its_input_is_refused(render_case,
                                                                           renderer):
    """Round-5 finding 2: probe validation accepted contradictory arm protocol / payload
    identities.  A probe summary's rendered arm entry has to be the run its own input binding
    -- and the live probe meta -- describe, exactly like the canonical summary's."""
    arm = "control_k8"
    path = os.path.join(render_case["tree"][arm]["probe"][0], "summary",
                        "yaw_pilot_summary.json")
    probe = fx.read_json(path)
    probe["arms"][0]["protocol_id"] = "contradictory-protocol"
    probe["arms"][0]["meta"]["per_sample_sha256"] = "0" * 64
    fx.write_json(path, probe)
    done = renderer(render_case)
    assert done.returncode == 2, out(done)
    assert "protocol_id" in out(done) and "per_sample_sha256" in out(done)


@pytest.mark.parametrize("renderer", [render_html, render_md])
def test_a_probe_summary_carrying_a_transplanted_arm_payload_is_refused(render_case, renderer):
    """The same transplant against a probe summary: another execution's complete payload."""
    arm = "control_k8"
    path = os.path.join(render_case["tree"][arm]["probe"][0], "summary",
                        "yaw_pilot_summary.json")
    probe = fx.read_json(path)
    _dir, _meta, foreign = execution_b(render_case, arm)
    probe["arms"] = [foreign["arms"][0]]
    fx.write_json(path, probe)
    done = renderer(render_case)
    assert done.returncode == 2, out(done)
    assert "execution_id" in out(done)


def test_html_does_not_copy_a_foreign_figure(render_case):
    """Finding 4: neighbouring figures were copied without binding them to the summary.

    A figure named for an arm that is not in this summary belongs to another run: replacing
    the combined figure with the CPU one produced GPU tables under a CPU plot.
    """
    foreign = os.path.join(render_case["summary_dir"], "yaw_pilot_gaps_released_k8.png")
    with open(foreign, "wb") as fout:
        fout.write(b"\x89PNG\r\nforeign")
    stale = os.path.join(render_case["summary_dir"], "stale_summary_interim.json")
    with open(stale, "w") as fout:
        fout.write("{}")
    done = render_html(render_case)
    assert done.returncode == 0, out(done)
    copied = sorted(os.listdir(render_case["assets"]))
    assert "yaw_pilot_gaps_released_k8.png" not in copied
    assert "stale_summary_interim.json" not in copied
    assert "yaw_pilot_gaps_control_k8.png" in copied
    assert "yaw_pilot_gaps_all_arms.png" in copied
    page = open(os.path.join(render_case["out_dir"], "page.html")).read()
    assert "yaw_pilot_gaps_released_k8.png" not in page


FIGURES = ("yaw_pilot_gaps_all_arms.png", "yaw_pilot_gaps_control_k8.png",
           "yaw_pilot_gaps_cyl_k8.png")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_html_regenerates_the_figures_from_the_summary(render_case):
    """Round-4 finding 3, the original substitution: filtering the *names* of the files
    copied out of the summariser's directory does not bind their *contents* to this
    summary, so replacing the combined PNG with another run's published it under "All
    arms" beside this summary's tables.  The figures are now drawn from the validated
    summary by the approved summariser's own functions; nothing is copied."""
    substituted = {}
    for name in FIGURES:
        path = os.path.join(render_case["summary_dir"], name)
        with open(path, "wb") as fout:
            fout.write(b"\x89PNG\r\nanother run's figure " + name.encode("utf-8"))
        substituted[name] = fx.file_sha256(path)
    done = render_html(render_case)
    assert done.returncode == 0, out(done)
    for name in FIGURES:
        published = os.path.join(render_case["assets"], name)
        assert os.path.isfile(published), name
        assert fx.file_sha256(published) != substituted[name], name
        with open(published, "rb") as fin:
            assert fin.read(8) == PNG_MAGIC, name
    page = rendered(render_case, "page.html")
    assert "regenerated" in page
    assert fx.file_sha256(render_case["summary"]) in page


def test_the_regenerated_figures_are_deterministic(render_case):
    """Finding 3: a regenerated figure is only evidence if it is reproducible byte for byte
    from the same summary."""
    assert render_html(render_case).returncode == 0
    first = {name: fx.file_sha256(os.path.join(render_case["assets"], name))
             for name in FIGURES}
    for name in FIGURES:
        os.remove(os.path.join(render_case["assets"], name))
    assert render_html(render_case).returncode == 0
    second = {name: fx.file_sha256(os.path.join(render_case["assets"], name))
              for name in FIGURES}
    assert first == second


def test_html_regenerates_a_figure_the_summary_directory_does_not_have(render_case):
    """The summariser's own PNGs are no longer an input, so a directory without them still
    renders -- from the summary, which is the binding that matters."""
    for name in FIGURES + ("yaw_pilot_gaps_all_arms.pdf", "yaw_pilot_gaps_cyl_k8.pdf",
                           "yaw_pilot_gaps_control_k8.pdf"):
        path = os.path.join(render_case["summary_dir"], name)
        if os.path.isfile(path):
            os.remove(path)
    done = render_html(render_case)
    assert done.returncode == 0, out(done)
    for name in FIGURES:
        assert os.path.isfile(os.path.join(render_case["assets"], name)), name


def test_html_refuses_a_summary_whose_figures_cannot_be_drawn(render_case):
    """A summary the approved figure code cannot plot is refused, not published without
    the figure the page promises."""
    summary = fx.read_json(render_case["summary"])
    for angles in (arm["angles"] for arm in summary["arms"]):
        for cells in angles.values():
            cells.clear()
    fx.write_json(render_case["summary"], summary)
    done = render_html(render_case)
    assert done.returncode != 0
    assert "figure" in out(done).lower()


# --------------------------------------------------------------------------------------
# Finding 5 -- the canonical notes and metric qualifications must appear, as values.
# Finding 6 -- the HTML page must always show query *and* room status.
# --------------------------------------------------------------------------------------

def rendered(case, name):
    return open(os.path.join(case["out_dir"], name)).read()


def test_md_renders_the_note_values_not_their_keys(render_case):
    """Finding 5: the renderers iterated the notes dict and printed band / gl_free / pipeline."""
    assert render_md(render_case).returncode == 0
    page = rendered(render_case, "page.md")
    for text in fx.NOTES.values():
        assert text in page
    assert "\n> band\n" not in page and "\n> pipeline\n" not in page


def test_md_renders_the_metric_qualifications_with_both_t60_denominators(render_case):
    """The values are read back through a pipe-table split: T60_abs quotes absolute values,
    so its cells only carry their definition when the pipes are escaped (finding 7)."""
    assert render_md(render_case).returncode == 0
    rows = {cells[0]: cells for cells in md_rows(rendered(render_case, "page.md"))
            if cells and cells[0] in fx.METRIC_QUALIFICATIONS}
    assert rows["T60"][1] == fx.METRIC_QUALIFICATIONS["T60"]["delta"]
    assert rows["T60"][2] == fx.METRIC_QUALIFICATIONS["T60"]["gap"]
    assert rows["T60_abs"][2] == fx.METRIC_QUALIFICATIONS["T60_abs"]["gap"]
    assert rows["EDT"][1] == fx.METRIC_QUALIFICATIONS["EDT"]["delta"]


def split_md_row(line):
    """Split one Markdown pipe-table row into its cells, honouring ``\\|`` escapes.

    This is what a Markdown renderer does, and it is the only way to tell whether a cell
    that *contains* a pipe survived: reading the raw line cannot.
    """
    body = line.strip()
    assert body.startswith("|") and body.endswith("|"), body
    cells, current, i = [], "", 1
    while i < len(body) - 1:
        if body[i] == "\\" and body[i + 1] == "|":
            current += "|"
            i += 2
        elif body[i] == "|":
            cells.append(current.strip())
            current = ""
            i += 1
        else:
            current += body[i]
            i += 1
    cells.append(current.strip())
    return cells


def md_rows(page):
    """Every pipe-table row of a Markdown page, split into cells."""
    return [split_md_row(line) for line in page.splitlines() if line.strip().startswith("|")]


def test_md_qualification_cells_survive_a_pipe_table_split(render_case):
    """Round-4 finding 7: T60_abs quotes absolute values (``|T60(P_alpha) - T60(P_0)|``) and
    the unescaped pipes split the cell, so the rendered table showed the delta as "change in"
    and the gap as "T60(prediction) - T60(GT)" -- the definition and its units disappeared."""
    assert render_md(render_case).returncode == 0
    rows = {cells[0]: cells for cells in md_rows(rendered(render_case, "page.md"))
            if cells and cells[0] in fx.METRIC_QUALIFICATIONS}
    assert sorted(rows) == sorted(fx.METRIC_QUALIFICATIONS)
    for metric, entry in sorted(fx.METRIC_QUALIFICATIONS.items()):
        cells = rows[metric]
        assert len(cells) == 3, cells
        assert cells[1] == entry["delta"], metric
        assert cells[2] == entry["gap"], metric


def test_md_metric_rows_survive_a_pipe_in_a_status_cell(render_case):
    """Every cell is escaped, not only the qualifications: a reason or status that quotes an
    absolute value must not split its row either."""
    summary = fx.read_json(render_case["summary"])
    cell = summary["arms"][0]["angles"]["64"]["EDT"]
    cell["headline"]["reportable"] = False
    cell["headline"]["reason"] = "|T60(P) - T60(GT)| moved"
    fx.write_json(render_case["summary"], summary)
    assert render_md(render_case).returncode == 0
    rows = [cells for cells in md_rows(rendered(render_case, "page.md"))
            if cells and cells[-1] == "|T60(P) - T60(GT)| moved"]
    assert rows, rendered(render_case, "page.md")
    for cells in rows:
        assert len(cells) == 10, cells


def test_html_renders_the_note_values_and_qualifications(render_case):
    assert render_html(render_case).returncode == 0
    page = rendered(render_case, "page.html")
    for text in fx.NOTES.values():
        assert html_escaped(text) in page
    assert html_escaped(fx.METRIC_QUALIFICATIONS["T60"]["delta"]) in page
    assert html_escaped(fx.METRIC_QUALIFICATIONS["T60"]["gap"]) in page
    assert "{'delta'" not in page and "&#x27;delta&#x27;" not in page


def html_escaped(text):
    import html as _html
    return _html.escape(text)


def test_html_shows_room_status_beside_a_reportable_multiple(render_case):
    """Finding 6: control EDT at 90° showed 17.8x and dropped the room status entirely."""
    assert render_html(render_case).returncode == 0
    page = rendered(render_case, "page.html")
    assert "status (query)" in page and "status (room)" in page
    rows = [r for r in page.split("<tr>") if "17.8" in r]
    assert rows, page[:2000]
    for row in rows:
        assert "denominator uncertain" in row, row
        assert row.count("<td>") >= 11, row


def test_html_shows_both_statuses_for_a_non_reportable_cell(render_case):
    assert render_html(render_case).returncode == 0
    page = rendered(render_case, "page.html")
    body = page[page.index("log-spec MAD"):]
    row = [r for r in body.split("<tr>") if "denominator uncertain" in r][0]
    assert row.count("denominator uncertain") >= 2      # query status, room status (and reason)


def test_md_keeps_both_statuses_in_its_status_column(render_case):
    assert render_md(render_case).returncode == 0
    page = rendered(render_case, "page.md")
    assert "status query / room" in page
    assert "defined / denominator uncertain" in page


# --------------------------------------------------------------------------------------
# Finding 11 -- the Markdown needs full sha256 provenance for every supplied input.
# Finding 12 -- both outputs must link the copied tables, data, CPU record and backend table.
# --------------------------------------------------------------------------------------

def supplementary(case):
    """The assembled asset directory: the summariser's copies, a backend table, a CPU record.

    ``make_results_html.py`` assembles these in the record; the Markdown renderer links them
    and refuses to promise a link to a file that is not there, so the fixture writes them.
    """
    os.makedirs(os.path.join(case["assets"], "cpu_protocol"), exist_ok=True)
    for name in ("yaw_pilot_tables.md", "yaw_pilot_summary.json", "yaw_pilot_gaps.csv"):
        import shutil
        shutil.copy2(os.path.join(case["summary_dir"], name),
                     os.path.join(case["assets"], name))
    backend = os.path.join(case["assets"], "backend_sensitivity_released_k8.md")
    with open(backend, "w") as fout:
        fout.write("# Backend sensitivity\n")
    cpu = os.path.join(case["assets"], "cpu_protocol", "yaw_pilot_tables.md")
    with open(cpu, "w") as fout:
        fout.write("# CPU protocol tables\n")
    return ["--backend-table", backend, "--cpu-record", cpu]


def test_md_records_full_sha256_for_every_supplied_input(render_case):
    """Finding 11: probe/parity hashes were truncated and check-online had none."""
    extra = supplementary(render_case)
    done = render_md(render_case, extra=["--assets", render_case["assets"]] + extra)
    assert done.returncode == 0, out(done)
    page = rendered(render_case, "page.md")
    inputs = [render_case["summary"], extra[1], extra[3]]
    inputs += [p for _a, p in [kv.split("=", 1) for kv in
                               render_case["parity"] + render_case["online"] + render_case["probe"]]]
    for path in inputs:
        assert fx.file_sha256(path) in page, path


def test_md_and_html_provenance_hashes_agree(render_case):
    extra = supplementary(render_case)
    assert render_md(render_case, extra=["--assets", render_case["assets"]] + extra).returncode == 0
    assert render_html(render_case, extra=extra).returncode == 0
    md, page = rendered(render_case, "page.md"), rendered(render_case, "page.html")
    md_hashes = set(re.findall(r"[0-9a-f]{64}", md))
    html_hashes = set(re.findall(r"[0-9a-f]{64}", page))
    inputs = [render_case["summary"], extra[1], extra[3]]
    inputs += [p for _a, p in [kv.split("=", 1) for kv in
                               render_case["parity"] + render_case["online"] + render_case["probe"]]]
    for path in inputs:
        digest = fx.file_sha256(path)
        assert digest in md_hashes and digest in html_hashes, path


@pytest.mark.parametrize("name,renderer", [("page.html", render_html), ("page.md", render_md)])
def test_both_outputs_link_the_supplementary_files(render_case, name, renderer):
    """Finding 12: copied tables / JSON / CSV were named but not linked, and the backend
    comparison was absent from the page's navigation."""
    extra = supplementary(render_case)
    if renderer is render_md:
        extra = ["--assets", render_case["assets"]] + extra
    assert renderer(render_case, extra=extra).returncode == 0
    page = rendered(render_case, name)
    for target in ("generated/yaw_pilot_tables.md", "generated/yaw_pilot_summary.json",
                   "generated/yaw_pilot_gaps.csv",
                   "generated/backend_sensitivity_released_k8.md",
                   "generated/cpu_protocol/yaw_pilot_tables.md"):
        assert target in page, (target, name)


@pytest.mark.parametrize("name,renderer", [("page.html", render_html), ("page.md", render_md)])
def test_assets_href_overrides_the_computed_relative_path(render_case, name, renderer):
    """The finish script stages the assets elsewhere and publishes them under generated/."""
    extra = supplementary(render_case) + ["--assets-href", "yaw_pilot_results_assets/generated"]
    if renderer is render_md:
        extra = ["--assets", render_case["assets"]] + extra
    assert renderer(render_case, extra=extra).returncode == 0
    page = rendered(render_case, name)
    assert "yaw_pilot_results_assets/generated/yaw_pilot_tables.md" in page
    assert "yaw_pilot_results_assets/generated/backend_sensitivity_released_k8.md" in page


@pytest.mark.parametrize("renderer", [render_html, render_md])
def test_a_missing_supplementary_input_is_refused(render_case, renderer):
    extra = ["--backend-table", os.path.join(render_case["assets"], "nope.md")]
    if renderer is render_md:
        extra = ["--assets", render_case["assets"]] + extra
    done = renderer(render_case, extra=extra)
    assert done.returncode != 0
    assert "nope.md" in out(done)


def test_md_refuses_to_link_assets_that_are_not_assembled(render_case):
    """Finding 12: the links have to resolve -- the HTML renderer assembles the assets first."""
    os.makedirs(render_case["assets"], exist_ok=True)
    done = render_md(render_case, extra=["--assets", render_case["assets"]])
    assert done.returncode != 0
    assert "yaw_pilot_tables.md" in out(done)


# --------------------------------------------------------------------------------------
# Findings 1 + 2 -- the finish script: validated evidence, staged assembly, atomic publish.
# --------------------------------------------------------------------------------------

def shim_bin(request):
    """A directory for the PATH shims that is free of ``os.pathsep``, wherever pytest runs.

    Round-9: the fault-injection helpers below prepend this directory to ``PATH``, and the
    shell splits ``PATH`` on ``os.pathsep``.  Under a basetemp that contains a colon
    (``--basetemp=/tmp/pytest-exp11-2026-09-28_07:55:15``) a shim inside ``tmp_path`` is
    therefore never found: the real ``mv``/``cp`` runs, the finish script publishes, and the
    twelve fault-injection tests fail for a reason that has nothing to do with the script.
    The shims live in ``/tmp`` (or ``$TMPDIR`` when that is itself usable as a ``PATH``
    entry) instead, and the test is failed loudly rather than silently if even that is not.

    Round-10: ``$TMPDIR`` is made absolute *before* it is inspected.  A ``PATH`` entry has to
    survive ``run_finish``'s change of directory, and Python 3.8's ``mkdtemp(dir=...)`` keeps
    a relative directory relative -- so a relative ``$TMPDIR`` passed the separator check and
    still produced a shim the finish script could not see.
    """
    base = os.path.abspath(os.environ.get("TMPDIR") or "/tmp")
    if os.pathsep in base or any(ch.isspace() for ch in base):
        base = "/tmp"
    bin_dir = tempfile.mkdtemp(prefix="exp10-shim-", dir=base)
    request.addfinalizer(lambda: shutil.rmtree(bin_dir, ignore_errors=True))
    if os.pathsep in bin_dir or any(ch.isspace() for ch in bin_dir):
        pytest.fail("the PATH shims need a directory free of %r and of whitespace, not %r"
                    % (os.pathsep, bin_dir))
    return bin_dir


def build_scratch(root, request):
    """The scratch repository the finish-script tests run on, built at an explicit ``root``.

    The root is a parameter only so that the round-9 regression test below can put one at a
    path that contains a colon; every other test takes it from the ``scratch`` fixture.
    ``request`` is what the shim directory registers its cleanup on.
    """
    os.makedirs(root)
    built = fx.make_scratch_repo(root, ASSETS, n_queries=32)
    generated = os.path.join(built["assets"], "generated")
    os.makedirs(generated)
    with open(os.path.join(generated, "stale_summary_interim.json"), "w") as fout:
        fout.write("{}")                       # a previously published asset set
    with open(os.path.join(built["record"], "yaw_pilot_results.md"), "w") as fout:
        fout.write("PREVIOUS MARKDOWN\n")
    with open(os.path.join(built["record"], "yaw_pilot_01_results.html"), "w") as fout:
        fout.write("PREVIOUS PAGE\n")
    built["generated"] = generated
    built["shim_bin"] = shim_bin(request)
    return built


@pytest.fixture
def scratch(tmp_path, request):
    """A scratch repository with the real record tooling, a stub summariser and a full tree."""
    return build_scratch(str(tmp_path / "repo"), request)


def run_finish(scratch, env=None):
    e = dict(os.environ)
    e.update({"EXP10_REPO_ROOT": scratch["root"], "EXP10_EXPECT_N": "32",
              "EXP10_FIXTURE_DIR": os.path.join(REPO, "tests")})
    e.update(env or {})
    return subprocess.run(["bash", os.path.join(scratch["scripts"], "exp10_finish.sh")],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=e,
                          cwd=scratch["root"])


def staging_dirs(scratch):
    return [n for n in os.listdir(scratch["assets"])
            if n.startswith("generated.staging") or n.startswith("generated.previous")
            or n.startswith(".finish_tmp") or n.startswith(".finish_prev")]


def assert_generated_untouched(scratch):
    assert os.path.isfile(os.path.join(scratch["generated"], "stale_summary_interim.json"))
    assert not os.path.isfile(os.path.join(scratch["generated"], "SHA256SUMS"))
    assert staging_dirs(scratch) == []
    assert open(os.path.join(scratch["record"], "yaw_pilot_results.md")).read() == \
        "PREVIOUS MARKDOWN\n"
    assert open(os.path.join(scratch["record"], "yaw_pilot_01_results.html")).read() == \
        "PREVIOUS PAGE\n"


def test_finish_refuses_when_released_k8_is_absent(tmp_path):
    """Findings 1 + 2: FINISH DONE was reached without the headline arm's evidence."""
    root = str(tmp_path / "repo")
    os.makedirs(root)
    built = fx.make_scratch_repo(root, ASSETS, n_queries=32, drop=("released_k8_all",))
    built["generated"] = os.path.join(built["assets"], "generated")
    os.makedirs(built["generated"])
    with open(os.path.join(built["generated"], "stale_summary_interim.json"), "w") as fout:
        fout.write("{}")
    with open(os.path.join(built["record"], "yaw_pilot_results.md"), "w") as fout:
        fout.write("PREVIOUS MARKDOWN\n")
    with open(os.path.join(built["record"], "yaw_pilot_01_results.html"), "w") as fout:
        fout.write("PREVIOUS PAGE\n")
    done = run_finish(built)
    assert done.returncode != 0
    assert "released_k8_all" in out(done)
    assert "FINISH DONE" not in out(done)
    assert_generated_untouched(built)


def test_finish_refuses_an_incomplete_released_k8(scratch):
    run_dir, meta = scratch["tree"]["released_k8"]["all"]
    meta["complete"] = False
    fx.write_json(os.path.join(run_dir, "meta.json"), meta)
    done = run_finish(scratch)
    assert done.returncode != 0
    assert "complete" in out(done) and "FINISH DONE" not in out(done)
    assert_generated_untouched(scratch)


def test_finish_refuses_a_failed_probe_control(scratch):
    probe_dir, probe_meta = scratch["tree"]["cyl_k8"]["probe"]
    fx.write_summary_dir(os.path.join(probe_dir, "summary"), [(probe_dir, probe_meta)],
                         controls_ok=False)
    done = run_finish(scratch)
    assert done.returncode != 0
    assert "controls" in out(done)
    assert_generated_untouched(scratch)


def test_finish_refuses_the_gpu_arms_reports_filed_as_the_cpu_record(scratch):
    """Round-4 finding 1: the finish script passed a relocation alias, so the GPU arm's
    check_online / parity reports -- copied into the CPU record -- were accepted and
    published as the CPU protocol's evidence."""
    cpu_dir = scratch["tree"]["cpu"]["all"][0]
    gpu_dir = scratch["tree"]["released_k8"]["all"][0]
    for name in ("check_online.json", "parity_exp03.json"):
        import shutil
        shutil.copy2(os.path.join(gpu_dir, name), os.path.join(cpu_dir, name))
    done = run_finish(scratch)
    assert done.returncode != 0
    assert "FINISH DONE" not in out(done)
    assert "run_dir" in out(done)
    assert_generated_untouched(scratch)


def test_finish_publishes_a_verified_asset_set(scratch):
    done = run_finish(scratch)
    assert done.returncode == 0, out(done)
    assert "FINISH DONE" in out(done)
    generated = scratch["generated"]
    sums = os.path.join(generated, "SHA256SUMS")
    assert os.path.isfile(sums)
    check = subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=generated,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert check.returncode == 0, check.stdout.decode()
    listed = [line.split(None, 1)[1].strip() for line in open(sums) if line.strip()]
    for expected in ("validation.json", "backend_sensitivity_released_k8.md",
                     "backend_sensitivity_released_k8.json", "yaw_pilot_summary.json",
                     "yaw_pilot_tables.md", "yaw_pilot_gaps.csv",
                     "yaw_pilot_gaps_all_arms.png", "runs/released_k8/meta_all.json",
                     "runs/released_k8/parity_exp03_probe.json",
                     "runs/released_k1/probe_summary.json",
                     "cpu_protocol/parity_exp03_all.json", "cpu_protocol/yaw_pilot_tables.md"):
        assert any(name.lstrip("./") == expected for name in listed), expected
    # the previous asset set is gone, not reused
    assert not os.path.isfile(os.path.join(generated, "stale_summary_interim.json"))
    assert not any(name.lstrip("./") == "stale_summary_interim.json" for name in listed)
    assert staging_dirs(scratch) == []
    # released_k1 has no parity report, so none is certified for it
    assert not any("released_k1/parity" in name for name in listed)
    page = open(os.path.join(scratch["record"], "yaw_pilot_01_results.html")).read()
    assert "yaw_pilot_results_assets/generated/backend_sensitivity_released_k8.md" in page
    md = open(os.path.join(scratch["record"], "yaw_pilot_results.md")).read()
    assert "yaw_pilot_results_assets/generated/yaw_pilot_tables.md" in md


def publication_snapshot(scratch):
    """Every byte of the three published artefacts: the assets, the HTML and the Markdown."""
    snapshot = {}
    for root, _dirs, files in os.walk(scratch["generated"]):
        for name in files:
            path = os.path.join(root, name)
            snapshot[os.path.relpath(path, scratch["assets"])] = fx.file_sha256(path)
    for name in ("yaw_pilot_01_results.html", "yaw_pilot_results.md"):
        path = os.path.join(scratch["record"], name)
        snapshot[name] = fx.file_sha256(path) if os.path.isfile(path) else None
    return snapshot


def shim(scratch, command, subject, cases):
    """A ``command`` on PATH that runs a body instead of acting, as the reviews injected it.

    ``subject`` is the shell expansion every case is matched against (``"$1"`` for a rename's
    source, ``"${@: -1}"`` for a copy's destination) and ``cases`` is a list of
    ``(pattern, body)`` pairs, so one shim can intercept two different calls -- the round-6
    review's rollback injection needs a failing rename *and* a signal from inside the
    rollback that follows it.  A body that does not exit falls through to the real command.
    """
    bin_dir = scratch["shim_bin"]          # never under tmp_path: see ``shim_bin`` above
    path = os.path.join(bin_dir, command)
    arms = "".join("  %s)\n    %s\n    ;;\n" % (pattern, body) for pattern, body in cases)
    with open(path, "w") as fout:
        fout.write('#!/bin/bash\ncase %s in\n%sesac\nexec /usr/bin/%s "$@"\n'
                   % (subject, arms, command))
    os.chmod(path, 0o755)
    return {"PATH": bin_dir + os.pathsep + os.environ.get("PATH", "")}


def mv_shim(scratch, pattern, body):
    """A ``mv`` on PATH that runs ``body`` instead of moving one source, as the reviews did.

    ``pattern`` is a shell ``case`` pattern matched against the *source* of the rename, so a
    test can pick exactly one of the finish script's three publications.
    """
    return shim(scratch, "mv", '"$1"', [(pattern, body)])


def cp_shim(scratch, pattern, body):
    """A ``cp`` on PATH that runs ``body`` instead of copying to one destination.

    ``pattern`` is matched against the *destination* (the last argument), so a test can pick
    exactly one of the rollback copies the finish script makes before it publishes anything.
    """
    return shim(scratch, "cp", '"${@: -1}"', [(pattern, body)])


def failing_mv(scratch, pattern):
    """A ``mv`` on PATH that fails for one destination, as the round-4 review injected it."""
    return mv_shim(scratch, pattern,
                   'echo "simulated mv I/O error on $1" >&2\n    exit 43')


def terminating_mv(scratch, pattern, after="exit 43"):
    """A ``mv`` that sends SIGTERM to the finish script immediately before one rename.

    This is the round-5 review's own injection: ``kill -TERM "$PPID"`` from the shim, which
    the finish script's shell only acts on once the shim has exited.  ``after`` decides
    whether the rename itself then happens (``exec /usr/bin/mv "$@"``) or not.
    """
    return mv_shim(scratch, pattern,
                   'echo "the test is terminating the script before $1" >&2\n'
                   '    kill -TERM "$PPID"\n    %s' % after)


TRUNCATE_COPY = '/usr/bin/head -c 4 "${@: -2:1}" > "${@: -1}"'   # a *partial* rollback copy


def truncating_cp(scratch, pattern, tail="exit 44"):
    """A ``cp`` that leaves a partial rollback copy behind, as the round-6 review did.

    The body writes the first four bytes of the source to the destination and then runs
    ``tail``: the review's own injection failed afterwards (``exit 44``), and a copy that
    truncates while *reporting success* is the same defect seen from the other side -- the
    existence of a backup file is never proof that it holds the whole original.
    """
    return cp_shim(scratch, pattern,
                   'echo "the test truncates the rollback copy ${@: -1}" >&2\n'
                   '    %s\n    %s' % (TRUNCATE_COPY, tail))


@pytest.mark.parametrize("pattern,stage", [
    ("*/.finish_tmp.*/yaw_pilot_results.md", "the Markdown report"),
    ("*/.finish_tmp.*/yaw_pilot_01_results.html", "the HTML page")])
def test_finish_restores_all_three_artefacts_when_a_publication_fails(scratch, pattern, stage):
    """Round-4 finding 2: the previous asset set was deleted before both reports were
    published, so a failing final rename left new assets, a new HTML page and the *old*
    Markdown -- three generations mixed -- while the cleanup claimed nothing had changed."""
    assert run_finish(scratch).returncode == 0          # a real published set to roll back to
    before = publication_snapshot(scratch)
    # regenerate from the same evidence: the publication must fail at `stage` this time
    done = run_finish(scratch, env=failing_mv(scratch, pattern))
    assert done.returncode != 0
    assert "FINISH DONE" not in out(done)
    assert "restored" in out(done)
    assert publication_snapshot(scratch) == before, stage
    assert staging_dirs(scratch) == []


def test_finish_removes_a_publication_it_created_when_the_next_one_fails(scratch):
    """The same rollback when there was nothing published before: the artefacts this run
    created are removed, so no half-published generation is left certified."""
    import shutil
    shutil.rmtree(scratch["generated"])
    for name in ("yaw_pilot_01_results.html", "yaw_pilot_results.md"):
        os.remove(os.path.join(scratch["record"], name))
    done = run_finish(scratch, env=failing_mv(scratch, "*/.finish_tmp.*/yaw_pilot_results.md"))
    assert done.returncode != 0
    assert not os.path.exists(scratch["generated"])
    for name in ("yaw_pilot_01_results.html", "yaw_pilot_results.md"):
        assert not os.path.exists(os.path.join(scratch["record"], name))
    assert staging_dirs(scratch) == []


def test_finish_rolls_back_all_three_artefacts_when_it_is_terminated(scratch):
    """Round-5 finding 1, the blocker: SIGTERM immediately before the final rename left new
    assets, a new HTML page and the *old* Markdown -- and the EXIT cleanup then deleted both
    backups, because it only restored what the destination no longer had.  The previous
    publication was gone for good.  A terminated publication has to roll all three back."""
    assert run_finish(scratch).returncode == 0          # a real published set to roll back to
    before = publication_snapshot(scratch)
    done = run_finish(scratch,
                      env=terminating_mv(scratch, "*/.finish_tmp.*/yaw_pilot_results.md"))
    assert done.returncode == 143, out(done)     # the TERM trap ran; the shell was not killed
    assert "SIGTERM" in out(done)                # ... and it said so
    assert "FINISH DONE" not in out(done)
    assert publication_snapshot(scratch) == before
    assert staging_dirs(scratch) == []


def test_finish_keeps_an_artefacts_absence_when_it_is_terminated(scratch):
    """The rollback has to preserve original *absence* too: a report that did not exist
    before the run must not exist after a terminated one.  Here the shim lets the Markdown
    rename through and terminates the script immediately afterwards, so the rollback has to
    remove an artefact this run created rather than put an earlier one back."""
    os.remove(os.path.join(scratch["record"], "yaw_pilot_results.md"))
    before = publication_snapshot(scratch)
    done = run_finish(scratch,
                      env=terminating_mv(scratch, "*/.finish_tmp.*/yaw_pilot_results.md",
                                         after='exec /usr/bin/mv "$@"'))
    assert done.returncode == 143, out(done)
    assert not os.path.exists(os.path.join(scratch["record"], "yaw_pilot_results.md"))
    assert publication_snapshot(scratch) == before
    assert staging_dirs(scratch) == []


def test_finish_keeps_the_backups_when_the_rollback_itself_fails(scratch):
    """If restoration cannot put an artefact back, the backup is the only copy of the
    previous publication left: it is kept and named, never deleted by the EXIT cleanup."""
    assert run_finish(scratch).returncode == 0
    record = scratch["record"]
    env = mv_shim(scratch, "*/.finish_tmp.*/yaw_pilot_results.md",
                  'chmod a-w %s\n    echo "simulated mv I/O error on $1" >&2\n    exit 43'
                  % shlex.quote(record))
    try:
        done = run_finish(scratch, env=env)
        assert done.returncode != 0
        assert "FINISH DONE" not in out(done)
        kept = [n for n in staging_dirs(scratch) if n.startswith(".finish_prev")]
        assert kept, (staging_dirs(scratch), out(done))
        assert kept[0] in out(done)              # what remains is named
        for name in ("yaw_pilot_01_results.html", "yaw_pilot_results.md"):
            assert os.path.isfile(os.path.join(scratch["assets"], kept[0], name)), name
        # ... and the closing line must not claim the previous publication is back in place
        closing = [line for line in out(done).splitlines() if "FINISH FAILED" in line]
        assert closing and "could NOT be put back" in closing[-1], closing
    finally:
        os.chmod(record, 0o755)


@pytest.mark.parametrize("tail,how", [
    ("exit 44", "the copy fails after writing part of the file"),
    ("exit 0", "the copy writes part of the file and reports success")])
def test_finish_refuses_an_incomplete_rollback_copy(scratch, tail, how):
    """Round-6 finding 1, the blocker: ``PUB_STATE=started`` preceded the report backups, so a
    ``cp`` that failed half-way left a truncated backup -- and restore() took its existence
    for proof of completeness, deleted the *intact* published HTML page and installed the
    truncation (``OLD HTML`` became ``OLD ``), with no backup left anywhere afterwards.  The
    backups are now made, verified and marked complete while nothing has been published yet."""
    before = publication_snapshot(scratch)              # the publication to protect
    done = run_finish(scratch, env=truncating_cp(
        scratch, "*/.finish_prev.*/yaw_pilot_01_results.html", tail=tail))
    assert done.returncode != 0, out(done)
    assert "FINISH DONE" not in out(done)
    assert "rollback copy" in out(done), out(done)      # ... and it names what it refused on
    assert "could NOT be put back" not in out(done)     # nothing was published: nothing to put back
    assert publication_snapshot(scratch) == before, how
    assert_generated_untouched(scratch)                 # no partial backup is installed or kept


@pytest.mark.parametrize("body,how", [
    ('kill -TERM "$PPID"\n    ' + TRUNCATE_COPY + '\n    exit 44',
     "the copy is cut in half, as the round-6 review injected it"),
    ('kill -TERM "$PPID"\n    exec /usr/bin/cp "$@"',
     "the copy itself completes and the signal ends the run")])
def test_finish_keeps_the_publication_when_it_is_terminated_while_backing_it_up(scratch, body,
                                                                               how):
    """Round-6 finding 1, the interruption half: the same truncated backup was installed over
    the intact HTML page when SIGTERM arrived during backup creation, because the publication
    was already ``started``.  Every backup is now made while it is still ``idle``, so a signal
    there can only end the run -- there is nothing published for it to leave half-replaced."""
    before = publication_snapshot(scratch)
    done = run_finish(scratch, env=cp_shim(
        scratch, "*/.finish_prev.*/yaw_pilot_01_results.html",
        'echo "the test is terminating the script while it copies ${@: -1}" >&2\n    ' + body))
    assert done.returncode == 143, out(done)            # the TERM trap ran; nothing was killed
    assert "SIGTERM" in out(done)
    assert "FINISH DONE" not in out(done)
    assert "could NOT be put back" not in out(done)
    assert publication_snapshot(scratch) == before, how
    assert_generated_untouched(scratch)


BACKUP_MARKER = ".backup_complete"   # exp10_finish.sh writes it beside a *verified* copy


def test_finish_never_installs_a_rollback_copy_that_lost_its_marker(scratch):
    """Round-6 finding 1, the other half: a copy is installed only while it is *known* whole.

    The HTML page has already been published when its rollback copy is truncated behind the
    script's back and its completion marker removed, so the rollback may not put those bytes
    anywhere: it leaves the published page as it is, keeps the copy, names it as INCOMPLETE
    and says the publication could not be put back in full -- while the two artefacts whose
    copies are still marked complete do go back.
    """
    before = publication_snapshot(scratch)
    env = shim(scratch, "mv", '"$1"', [
        ("*/.finish_tmp.*/yaw_pilot_results.md",
         'A=$(dirname "$(dirname "$1")")\n'
         '    for b in "$A"/.finish_prev.*/yaw_pilot_01_results.html; do\n'
         '      /usr/bin/head -c 4 "$b" > "$b.part"; /usr/bin/mv "$b.part" "$b"\n'
         '      /usr/bin/rm -f "$b%s"\n'
         '    done\n'
         '    echo "the test truncated the HTML rollback copy and unmarked it" >&2\n'
         '    exit 43' % BACKUP_MARKER)])
    done = run_finish(scratch, env=env)
    assert done.returncode == 9, out(done)
    assert "INCOMPLETE" in out(done), out(done)
    kept = [n for n in staging_dirs(scratch) if n.startswith(".finish_prev")]
    assert kept and kept[0] in out(done), (staging_dirs(scratch), out(done))
    corrupt = os.path.join(scratch["assets"], kept[0], "yaw_pilot_01_results.html")
    assert os.path.getsize(corrupt) == 4            # the copy it refused to install is kept
    page = open(os.path.join(scratch["record"], "yaw_pilot_01_results.html")).read()
    assert page != "PREVIOUS PAGE\n" and len(page) > 4   # ... and never written over the page
    closing = [line for line in out(done).splitlines() if "FINISH FAILED" in line]
    assert closing and "could NOT be put back" in closing[-1], closing
    # the Markdown report and the asset set, whose copies are marked complete, are back
    assert open(os.path.join(scratch["record"], "yaw_pilot_results.md")).read() == \
        "PREVIOUS MARKDOWN\n"
    assert publication_snapshot(scratch)["generated/stale_summary_interim.json"] == \
        before["generated/stale_summary_interim.json"]


def test_finish_completes_the_rollback_when_it_is_terminated_during_it(scratch):
    """Round-6 finding 2: the explicit rollback marked itself ``rolled_back`` before it had put
    anything back and left TERM / INT / HUP armed, so SIGTERM during the asset restoration
    exited 143 with the old assets, the *new* HTML page and the old Markdown -- three
    generations mixed -- while the EXIT trap, seeing ``rolled_back``, did no more."""
    before = publication_snapshot(scratch)
    env = shim(scratch, "mv", '"$1"', [
        ("*/.finish_tmp.*/yaw_pilot_results.md",        # the last publication fails ...
         'echo "simulated mv I/O error on $1" >&2\n    exit 43'),
        ("*/generated.previous.*",                      # ... and a signal arrives inside the
         'echo "the test terminates the script inside the rollback" >&2\n'
         '    kill -TERM "$PPID"')])                    # rollback; the rename still happens
    done = run_finish(scratch, env=env)
    # A failure triggered this rollback, so the failure's status is what is reported: TERM was
    # ignored (SIG_IGN) for the whole rollback, so the signal is discarded rather than queued.
    assert done.returncode == 9, out(done)
    assert "FINISH DONE" not in out(done)
    assert "restored" in out(done)
    assert "could NOT be put back" not in out(done)
    assert publication_snapshot(scratch) == before      # all three artefacts, byte for byte
    assert_generated_untouched(scratch)                 # ... and the backups are then removed


def test_finish_leaves_the_published_assets_alone_when_a_stage_fails(scratch):
    """Finding 2: a failed stage used to leave stale destination contents certified."""
    done = run_finish(scratch, env={"STUB_FAIL_ON_CPU": "1"})
    assert done.returncode != 0
    assert "FINISH DONE" not in out(done)
    assert_generated_untouched(scratch)


def test_finish_refuses_when_the_summariser_output_is_incomplete(scratch):
    done = run_finish(scratch, env={"STUB_DROP_CSV": "1"})
    assert done.returncode != 0
    assert "yaw_pilot_gaps.csv" in out(done)
    assert_generated_untouched(scratch)


def test_finish_does_not_certify_a_foreign_figure(scratch):
    """Finding 2/4: only this summary's own files may enter the published set."""
    done = run_finish(scratch, env={"STUB_FOREIGN_FIGURE": "1"})
    assert done.returncode == 0, out(done)
    listed = open(os.path.join(scratch["generated"], "SHA256SUMS")).read()
    assert "someone_else" not in listed


def test_finish_publishes_regenerated_figures_not_the_summarisers_files(scratch):
    """Round-4 finding 3, end to end: a PNG substituted in the summariser's output
    directory under the *right* name never reaches the published asset set."""
    marker = "another run's combined figure"
    done = run_finish(scratch, env={"STUB_SUBSTITUTE_COMBINED": marker})
    assert done.returncode == 0, out(done)
    published = os.path.join(scratch["generated"], "yaw_pilot_gaps_all_arms.png")
    with open(published, "rb") as fin:
        head = fin.read(8)
    assert head == PNG_MAGIC
    assert marker.encode("utf-8") != open(published, "rb").read()
    page = open(os.path.join(scratch["record"], "yaw_pilot_01_results.html")).read()
    assert "regenerated" in page
    check = subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"],
                           cwd=scratch["generated"], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    assert check.returncode == 0, check.stdout.decode()


def test_finish_fault_injection_fires_from_a_scratch_path_with_a_colon(tmp_path, request):
    """Round-9 regression: the fault injection itself must not depend on the scratch path.

    ``PATH`` is split on ``os.pathsep``, so a shim directory under a basetemp that contains a
    colon (another session ran this suite with ``--basetemp=/tmp/pytest-...2026-09-28_07:55:15``)
    is invisible to the shell: the real ``mv`` runs, the finish script publishes successfully
    and twelve fault-injection tests fail for a reason that has nothing to do with the script.
    Here the whole scratch repository deliberately lives under a colon, and the shim must
    still fire.
    """
    built = build_scratch(str(tmp_path / "with:colon" / "repo"), request)
    assert os.pathsep in built["root"]
    done = run_finish(built, env=failing_mv(built, "*/.finish_tmp.*/yaw_pilot_results.md"))
    assert done.returncode != 0, out(done)             # the shim fired ...
    assert "simulated mv I/O error" in out(done)       # ... and it was ours, not a real mv
    assert "FINISH DONE" not in out(done)
    assert "restored" in out(done)
    assert staging_dirs(built) == []
    assert_generated_untouched(built)


def test_finish_fault_injection_fires_when_tmpdir_is_relative(tmp_path, request, monkeypatch):
    """Round-9 review, finding 1: a relative ``TMPDIR`` must not move the shims out of reach.

    Python 3.8's ``mkdtemp(dir=...)`` keeps a relative directory relative, and ``run_finish``
    runs the finish script with ``cwd`` set to the scratch repository.  A ``PATH`` entry taken
    straight from a relative ``$TMPDIR`` therefore names a directory that does not exist from
    the script's point of view: the real ``mv`` runs, the script publishes, and the
    fault-injection tests fail for a reason that has nothing to do with the script -- the same
    silent failure the colon regression above guards against.  ``shim_bin`` makes the base
    absolute *before* it inspects it, so the ``PATH`` entry survives the change of directory.
    """
    relative_tmpdir = str(tmp_path / "reltmp")
    os.makedirs(relative_tmpdir)
    relative = os.path.relpath(relative_tmpdir, os.getcwd())
    assert not os.path.isabs(relative)
    monkeypatch.setenv("TMPDIR", relative)
    built = build_scratch(str(tmp_path / "repo"), request)
    done = run_finish(built, env=failing_mv(built, "*/.finish_tmp.*/yaw_pilot_results.md"))
    assert done.returncode != 0, out(done)             # the shim fired ...
    assert "simulated mv I/O error" in out(done)       # ... and it was ours, not a real mv
    assert "FINISH DONE" not in out(done)
    assert "restored" in out(done)
    assert staging_dirs(built) == []
    assert_generated_untouched(built)
    assert os.path.isabs(built["shim_bin"])            # ... and not "<relative TMPDIR>/exp10-shim-*"


# --------------------------------------------------------------------------------------
# Findings 7-10 -- the arm chain: ok-gates, the predecessor table, atomic reservation, pins.
# --------------------------------------------------------------------------------------

@pytest.fixture
def chain(tmp_path):
    """A scratch git repository with stubs for the evaluator, comparator and summariser."""
    root = str(tmp_path / "repo")
    os.makedirs(root)
    return fx.make_chain_repo(root, ASSETS)


def chain_command(chain, arm="control_k8", args=None, env=None):
    """``(argv, env)`` for one arm-chain invocation in the scratch repository."""
    script = os.path.join(chain["scripts"], "exp10_arm_chain.sh")
    argv = ["bash", script] + list(args if args is not None else
                                  [arm, "simple", "ckpt/xRIR_simple_8_shot/epoch_12.pth",
                                   "ckpt/yaw_rotation/reference_manifest.json", "47637a55",
                                   "8", "cpu", "0"])
    e = dict(os.environ)
    e.update({"EXP10_REPO_ROOT": chain["root"], "EXP10_FIXTURE_DIR": os.path.join(REPO, "tests")})
    e.update(env or {})
    return argv, e


def run_chain(chain, arm="control_k8", args=None, env=None):
    argv, e = chain_command(chain, arm=arm, args=args, env=env)
    return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=e,
                          cwd=chain["root"])


def chain_commit(chain, message):
    """Commit the scratch repository's tools, so the chain's clean-tree check passes."""
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    subprocess.check_call(["git", "-C", chain["root"], "add", "-A", "tools"], env=env)
    subprocess.check_call(["git", "-C", chain["root"], "commit", "-q", "-m", message],
                          env=env)


def run_path(chain, *parts):
    return os.path.join(chain["root"], "ckpt", "exp10", *parts)


def test_chain_runs_the_probe_then_the_full_stage(chain):
    done = run_chain(chain)
    assert done.returncode == 0, out(done)
    assert "ARM DONE" in out(done)
    assert os.path.isfile(run_path(chain, "control_k8_probe", "meta.json"))
    assert os.path.isfile(run_path(chain, "control_k8_all", "meta.json"))
    assert os.path.isfile(run_path(chain, "control_k8_probe", "parity_exp03.json"))


def test_chain_fails_a_stage_whose_online_check_reports_not_ok(chain):
    """Finding 7: the comparator exits 0 with ok=false, so the chain launched full anyway."""
    done = run_chain(chain, env={"STUB_ONLINE_FAIL_STAGE": "probe"})
    assert done.returncode != 0
    assert "ARM DONE" not in out(done)
    assert "check-online" in out(done)
    assert not os.path.exists(run_path(chain, "control_k8_all"))


def test_chain_fails_when_the_full_runs_online_check_is_not_ok(chain):
    done = run_chain(chain, env={"STUB_ONLINE_FAIL_STAGE": "all"})
    assert done.returncode != 0
    assert "ARM DONE" not in out(done)


def test_chain_fails_a_stage_whose_parity_reports_not_ok(chain):
    done = run_chain(chain, env={"STUB_PARITY_FAIL_STAGE": "probe"})
    assert done.returncode != 0
    assert not os.path.exists(run_path(chain, "control_k8_all"))


def test_chain_refuses_an_unknown_arm(chain):
    """Finding 8: the predecessor came from an unrestricted argument."""
    done = run_chain(chain, args=["made_up_arm", "simple", "ck", "man", "hash", "8", "cpu", "0"])
    assert done.returncode != 0
    assert "made_up_arm" in out(done)


def test_chain_uses_the_table_predecessor_when_the_argument_is_omitted(chain):
    done = run_chain(chain)
    assert done.returncode == 0, out(done)
    parity = fx.read_json(run_path(chain, "control_k8_all", "parity_exp03.json"))
    assert parity["exp03_path"].endswith("sweep_control/per_sample_yaw.json")


def test_chain_refuses_none_for_a_replicated_arm(chain):
    """Finding 8: omitting the predecessor for control_k8 yielded parity_ok=n/a."""
    done = run_chain(chain, args=["control_k8", "simple", "ck", "man", "hash", "8", "cpu", "0",
                                  "none"])
    assert done.returncode != 0
    assert "none" in out(done)
    assert not os.path.exists(run_path(chain, "control_k8_probe"))


def test_chain_refuses_a_predecessor_that_is_not_the_tables(chain):
    done = run_chain(chain, args=["control_k8", "simple", "ck", "man", "hash", "8", "cpu", "0",
                                  "ckpt/yaw_rotation/sweep_cyl/per_sample_yaw.json"])
    assert done.returncode != 0
    assert "sweep_control" in out(done)


def test_chain_runs_released_k1_without_a_predecessor(chain):
    done = run_chain(chain, arm="released_k1")
    assert done.returncode == 0, out(done)
    assert not os.path.exists(run_path(chain, "released_k1_probe", "parity_exp03.json"))


def test_chain_refuses_an_existing_output_directory_without_metadata(chain):
    """Finding 9: a directory with a waveform but no meta.json was overwritten."""
    partial = run_path(chain, "control_k8_probe")
    os.makedirs(partial)
    with open(os.path.join(partial, "wav_k0.npy"), "wb") as fout:
        fout.write(b"partial")
    done = run_chain(chain)
    assert done.returncode != 0
    assert "control_k8_probe" in out(done)
    assert not os.path.isfile(os.path.join(partial, "meta.json"))


def test_chain_tolerates_narrative_files_under_worklog(chain):
    """Finding 10: only worklog markdown and logs are exempt from the clean-tree check."""
    with open(os.path.join(chain["record"], "scratch_notes.md"), "w") as fout:
        fout.write("notes\n")
    with open(os.path.join(chain["record"], "scratch.log"), "w") as fout:
        fout.write("log\n")
    assert run_chain(chain).returncode == 0


def test_chain_refuses_a_dirty_script_under_worklog(chain):
    with open(os.path.join(chain["scripts"], "sneaky.py"), "w") as fout:
        fout.write("print('hi')\n")
    done = run_chain(chain)
    assert done.returncode != 0
    assert "dirty" in out(done)
    assert not os.path.exists(run_path(chain, "control_k8_probe"))


def test_chain_refuses_when_a_tool_changes_between_the_probe_and_the_full_stage(chain):
    """Finding 10: cleanliness was checked once, so an evaluator edited after the probe ran."""
    done = run_chain(chain, env={"STUB_TOUCH_TOOL": "1"})
    assert done.returncode != 0
    assert "ARM DONE" not in out(done)
    assert not os.path.exists(run_path(chain, "control_k8_all"))
    assert "exp10_yaw_pilot.py" in out(done)


def test_the_chains_source_pins_live_in_the_probe_it_reserved(chain):
    """Round-4 finding 6: the pins were written to a shared per-arm filename *before* the
    probe directory was reserved, which is what let a second chain replace them."""
    done = run_chain(chain)
    assert done.returncode == 0, out(done)
    pins = run_path(chain, "control_k8_probe", "source_pins.sha256")
    assert os.path.isfile(pins)
    assert not os.path.exists(run_path(chain, "control_k8_source_pins.sha256"))
    assert not (os.stat(pins).st_mode & 0o222)          # immutable for this invocation
    assert "tools/exp10_yaw_pilot.py" in open(pins).read()


def test_a_rejected_concurrent_chain_cannot_replace_an_active_chains_pins(chain):
    """Round-4 finding 6: chain A paused after its probe; a committed evaluator change and
    chain B for the same arm followed.  B overwrote the shared pin file and *then* refused
    the existing probe directory with 2 -- so A accepted the replacement pins as its own
    baseline and launched the full stage with a changed evaluator, finishing 0."""
    import time

    outside = os.path.dirname(chain["root"])            # never inside the repo: the chain's
    release = os.path.join(outside, "release_probe_summary")   # clean-tree check would refuse
    log_path = os.path.join(outside, "chain_a.log")
    argv, env = chain_command(chain, env={"STUB_PAUSE_UNTIL": release})
    with open(log_path, "wb") as log:
        first = subprocess.Popen(argv, cwd=chain["root"], env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
        deadline = time.time() + 60
        while not os.path.exists(release + ".paused"):
            assert first.poll() is None, open(log_path).read()
            assert time.time() < deadline, "chain A never reached its probe summary"
            time.sleep(0.1)
        pins = run_path(chain, "control_k8_probe", "source_pins.sha256")
        assert os.path.isfile(pins)
        before = open(pins).read()
        evaluator = os.path.join(chain["root"], "tools", "exp10_yaw_pilot.py")
        with open(evaluator, "a") as fout:
            fout.write("\n# a new evaluator implementation while the first chain waits\n")
        chain_commit(chain, "new evaluator implementation")
        second = run_chain(chain, env={"STUB_PAUSE_UNTIL": release})
        assert second.returncode == 2, out(second)
        assert "control_k8_probe" in out(second)
        assert open(pins).read() == before          # chain A's baseline is untouched
        with open(release, "w") as fout:
            fout.write("go\n")
        assert first.wait(timeout=120) != 0
    log_text = open(log_path).read()
    assert "exp10_yaw_pilot.py" in log_text
    assert "ARM DONE" not in log_text
    assert not os.path.exists(run_path(chain, "control_k8_all"))


# --------------------------------------------------------------------------------------
# Finding 13 -- the queue wrappers must not hide a child's failure.
# --------------------------------------------------------------------------------------

@pytest.fixture
def queue(tmp_path):
    root = str(tmp_path / "repo")
    os.makedirs(root)
    return fx.make_queue_repo(root, ASSETS)


def run_script(queue, name, args=(), env=None):
    e = dict(os.environ)
    e.update({"EXP10_REPO_ROOT": queue["root"],
              "PATH": queue["bin"] + os.pathsep + os.environ.get("PATH", ""),
              "EXP10_STUB_CHAIN_LOG": os.path.join(queue["root"], "chain_calls.txt")})
    e.update(env or {})
    return subprocess.run(["bash", os.path.join(queue["scripts"], name)] + [str(a) for a in args],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=e, cwd=queue["root"])


def test_queue_exits_zero_when_every_arm_succeeds(queue):
    done = run_script(queue, "exp10_gpu_queue.sh", ["1"])
    assert done.returncode == 0, out(done)
    assert "QUEUE DONE" in out(done)
    calls = open(os.path.join(queue["root"], "chain_calls.txt")).read()
    for arm in ("released_k1", "control_k8", "cyl_k8"):
        assert arm in calls


def test_queue_runs_every_arm_but_exits_nonzero_when_one_fails(queue):
    """Finding 13: the queue returned success after all its mocked arms had failed."""
    done = run_script(queue, "exp10_gpu_queue.sh", ["1"], env={"EXP10_STUB_RC_control_k8": "7"})
    assert done.returncode != 0
    assert "QUEUE DONE" in out(done)               # the remaining arms still ran
    calls = open(os.path.join(queue["root"], "chain_calls.txt")).read()
    assert "cyl_k8" in calls
    assert "failure" in out(done).lower()


def test_queue_exits_nonzero_when_all_arms_fail(queue):
    done = run_script(queue, "exp10_gpu_queue.sh", ["1"],
                      env={"EXP10_STUB_RC_released_k1": "3", "EXP10_STUB_RC_control_k8": "3",
                           "EXP10_STUB_RC_cyl_k8": "3"})
    assert done.returncode != 0


def test_after_queue_wrapper_preserves_the_chain_status(queue):
    """Finding 13: the wrapper logged the failure and returned success."""
    qlog = os.path.join(queue["root"], "queue.log")
    with open(qlog, "w") as fout:
        fout.write("[00:00:00] QUEUE DONE (arms attempted 3, failures 0)\n")
    ok = run_script(queue, "exp10_released_k1_after_queue.sh", ["1", qlog])
    assert ok.returncode == 0, out(ok)
    assert "RELEASED_K1_EXIT=0" in open(qlog).read()
    bad = run_script(queue, "exp10_released_k1_after_queue.sh", ["1", qlog],
                     env={"EXP10_STUB_RC_released_k1": "5"})
    assert bad.returncode == 5
    assert "RELEASED_K1_EXIT=5" in open(qlog).read()


@pytest.mark.parametrize("name,renderer", [("page.html", render_html), ("page.md", render_md)])
def test_provenance_names_the_published_location_of_a_staged_asset(render_case, name, renderer):
    """The finish script stages the assets under another name: provenance must not record a
    path that stops existing when the staging directory is renamed into place."""
    extra = supplementary(render_case) + ["--assets-href", "yaw_pilot_results_assets/generated"]
    if renderer is render_md:
        extra = ["--assets", render_case["assets"]] + extra
    assert renderer(render_case, extra=extra).returncode == 0
    page = rendered(render_case, name)
    digest = fx.file_sha256(os.path.join(render_case["assets"],
                                         "backend_sensitivity_released_k8.md"))
    assert digest in page
    assert render_case["assets"] not in page.split("Provenance")[-1]
    assert "yaw_pilot_results_assets/generated/backend_sensitivity_released_k8.md" in page


# --------------------------------------------------------------------------------------
# Record review, finding 5 -- the K = 1 arm is the K = 8 checkpoint evaluated at K = 1,
# and a section header carrying the bare key does not say so.
# --------------------------------------------------------------------------------------

K1_ARMS = ("control_k8", "released_k1")


@pytest.fixture
def render_case_k1(tmp_path):
    """A two-arm summary that includes ``released_k1`` (which has, and needs, no parity)."""
    root = str(tmp_path / "exp10k1")
    tree = fx.write_record_tree(root, arms=K1_ARMS, n_queries=32, cpu=False)
    runs = [tree[arm]["all"] for arm in K1_ARMS]
    summary_dir = os.path.join(root, "summary")
    summary_path, _summary = fx.write_summary_dir(summary_dir, runs)
    case = {"root": root, "tree": tree, "summary": summary_path,
            "summary_dir": summary_dir, "out_dir": str(tmp_path / "page"),
            "assets": str(tmp_path / "page" / "generated")}
    os.makedirs(case["out_dir"])
    case["parity"] = ["control_k8=%s" % os.path.join(tree["control_k8"]["all"][0],
                                                     "parity_exp03.json")]
    case["online"] = ["%s=%s" % (arm, os.path.join(tree[arm]["all"][0],
                                                   "check_online.json"))
                      for arm in K1_ARMS]
    case["probe"] = ["%s=%s" % (arm, os.path.join(tree[arm]["probe"][0], "summary",
                                                  "yaw_pilot_summary.json"))
                     for arm in K1_ARMS]
    return case


def test_both_reports_qualify_the_k1_arm_and_keep_its_key(render_case_k1):
    """Finding 5: read on its own, a section headed ``released_k1`` looks like a model
    trained at K = 1; the arm *key* it is filed under must not change with the wording."""
    done = render_md(render_case_k1)
    assert done.returncode == 0, out(done)
    done = render_html(render_case_k1)
    assert done.returncode == 0, out(done)
    markdown = rendered(render_case_k1, "page.md")
    page = rendered(render_case_k1, "page.html")

    assert "## released_k1 (trained K = 8, evaluated K = 1)" in markdown
    assert "## control_k8\n" in markdown
    assert "## control_k8 (" not in markdown
    assert "<h2 id='released_k1'>released_k1 (trained K = 8, evaluated K = 1)</h2>" in page
    assert "<h2 id='control_k8'>control_k8</h2>" in page
    assert "alt='released_k1 (trained K = 8, evaluated K = 1)'" in page

    # The keys the evidence is bound by, and the file names, are untouched.
    assert "- released_k1: execution `" in markdown
    assert "yaw_pilot_gaps_released_k1.png" in page
    assert os.path.isfile(os.path.join(render_case_k1["assets"],
                                       "yaw_pilot_gaps_released_k1.png"))
    copied = fx.read_json(os.path.join(render_case_k1["assets"], "yaw_pilot_summary.json"))
    assert [arm["arm"] for arm in copied["arms"]] == list(K1_ARMS)


def test_the_renderers_take_the_arm_label_from_the_summariser():
    """One home for the wording: both renderers import the summariser's mapping."""
    from tools import exp10_summarize as summarize

    for name in ("make_results_md", "make_results_html"):
        module = load_tool(name)
        assert module.display_label is summarize.display_label, name
    assert summarize.display_label("released_k1") == \
        "released_k1 (trained K = 8, evaluated K = 1)"
