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


def test_html_refuses_a_summary_whose_figures_are_missing(render_case):
    os.remove(os.path.join(render_case["summary_dir"], "yaw_pilot_gaps_cyl_k8.png"))
    done = render_html(render_case)
    assert done.returncode != 0
    assert "yaw_pilot_gaps_cyl_k8.png" in out(done)


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
    assert render_md(render_case).returncode == 0
    page = rendered(render_case, "page.md")
    assert fx.METRIC_QUALIFICATIONS["T60"]["delta"] in page
    assert fx.METRIC_QUALIFICATIONS["T60"]["gap"] in page
    assert fx.METRIC_QUALIFICATIONS["T60_abs"]["gap"] in page
    assert fx.METRIC_QUALIFICATIONS["EDT"]["delta"] in page


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
