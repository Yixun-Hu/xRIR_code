"""Adversarial tests for the exp_03 record tools (Planner one-offs in the record's assets folder): the acceptance
checker's pure validators must reject fabricated, stale or malformed inputs (Codex round-3 findings 2-3)."""
import copy
import datetime
import math
import os
import sys

import pytest

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worklog", "worklog_yixun",
                      "exp_03_yaw_rotation_degradation_claude", "yaw_rotation_degradation_results_assets")
sys.path.insert(0, ASSETS)
import check_sweep_acceptance as c  # noqa: E402

N = 4


def good_gate():
    return {"mode": "k0-gate", "gate_pass": True, "band_rule": "v3 (2026-09-06 per-model amendment)", "manifest_hash": c.MANIFEST_HASH,
            "config": {"runs": list(c.GATE_RUNS["runs"])}, "noise_runs": list(c.GATE_RUNS["noise_runs"]), "noise_runs_cyl": list(c.GATE_RUNS["noise_runs_cyl"]),
            "nuisance_runs": list(c.GATE_RUNS["nuisance_runs"]), "shape_runs": list(c.GATE_RUNS["shape_runs"]), "summary_path": "x.txt", "summary_sha256": "abc",
            "gate_rows": [{"metric": "edt", "diff": 0.1}], "band_terms": {"control": {"s_ref": 0.1}}}


def test_a_minimal_fabricated_gate_is_rejected():
    fake = {"mode": "k0-gate", "gate_pass": True, "manifest_hash": c.MANIFEST_HASH, "summary_path": "x", "summary_sha256": "y"}
    f = c.check_gate_json(fake, summary_sha="y")
    assert any("band rule" in x for x in f) and any("runs" in x for x in f)


def test_a_complete_gate_passes_and_each_requirement_is_enforced():
    assert c.check_gate_json(good_gate(), summary_sha="abc", decision_copy_sha="abc") == []
    for mutate, needle in [(lambda g: g.update(band_rule="v2"), "band rule"), (lambda g: g.update(gate_pass=False), "gate_pass"),
                           (lambda g: g.update(shape_runs=[]), "shape_runs"), (lambda g: g["config"].update(runs=g["config"]["runs"][:2]), "runs"),
                           (lambda g: g.update(manifest_hash="0" * 64), "manifest")]:
        g = good_gate(); mutate(g)
        assert any(needle in x for x in c.check_gate_json(g, summary_sha="abc", decision_copy_sha="abc")), needle
    assert any("decision copy" in x for x in c.check_gate_json(good_gate(), summary_sha="abc", decision_copy_sha="other"))
    assert any("digest" in x for x in c.check_gate_json(good_gate(), summary_sha="zzz", decision_copy_sha="abc"))


def test_a_gate_written_after_the_sweep_started_is_rejected():
    g = good_gate(); t = datetime.datetime(2026, 9, 6, 15, 41, tzinfo=datetime.timezone.utc)
    g["_summary_mtime"] = t + datetime.timedelta(minutes=1)
    assert any("after the sweep" in x for x in c.check_gate_json(g, sweep_start=t, summary_sha="abc", decision_copy_sha="abc"))
    g["_summary_mtime"] = t - datetime.timedelta(minutes=1)
    assert c.check_gate_json(g, sweep_start=t, summary_sha="abc", decision_copy_sha="abc") == []


def manifest_entries():
    rooms = [f"Cat/Room_{i}" for i in range(c.EXPECTED_ROOMS)]
    return [{"index": i, "query": f"{rooms[i % len(rooms)]}/S00{i}_R001_hybrid_IR.wav"} for i in range(c.EXPECTED_ROOMS)]


def test_query_lists_must_equal_the_manifest_in_order():
    ents = manifest_entries(); q = [e["query"] for e in ents]; i = [e["index"] for e in ents]
    assert c.check_manifest_order(q, i, ents) == []
    assert c.check_manifest_order(list(reversed(q)), i, ents)
    assert c.check_manifest_order(q, [0] * len(i), ents)
    invented = [f"Cat/Room_{k}/S00{k}_R009_hybrid_IR.wav" for k in range(c.EXPECTED_ROOMS)]
    assert c.check_manifest_order(invented, i, ents)


def synthetic_run(n=N):
    ps = {"P": {}, "E": {}}; m = {"P": {}, "E": {}}
    for cond in ("P", "E"):
        for k in c.SPECTRAL:
            rows = {key: [0.1 * (j + 1) for j in range(n)] for key in c.SPECTRAL_KEYS}
            for key in c.ACOUSTIC_KEYS:
                rows[key] = [0.5 * (j + 1) for j in range(n)]
            ps[cond][str(k)] = rows
            m[cond][str(k)] = {key: {"mean": sum(rows[key]) / n, "n_valid": n, "n_nan": 0} for key in c.SPECTRAL_KEYS}
            for key in c.ACOUSTIC_KEYS:
                m[cond][str(k)][key] = {"mean": sum(rows[key]) / n, "n_valid": n, "n_nan": 0}
    return ps, m


def test_synthetic_content_passes_then_every_hole_is_caught():
    ps, m = synthetic_run(); assert c.check_arrays(ps, m, n=N) == []
    bad = copy.deepcopy(ps); bad["P"]["32"]["loss"][0] = float("nan"); assert any("loss" in x for x in c.check_arrays(bad, m, n=N))
    bad = copy.deepcopy(ps); bad["P"]["32"]["edt"][0] = float("inf"); assert any("edt" in x for x in c.check_arrays(bad, m, n=N))
    bad = copy.deepcopy(ps); bad["P"]["32"]["edt"][0] = "0.5"; assert any("edt" in x for x in c.check_arrays(bad, m, n=N))
    bad = copy.deepcopy(ps); bad["P"]["32"]["c50"][0] = None; assert any("n_valid" in x for x in c.check_arrays(bad, m, n=N))
    bad = copy.deepcopy(ps); bad["P"]["32"]["decay"] = bad["P"]["32"]["decay"][:-1]; assert any("decay" in x for x in c.check_arrays(bad, m, n=N))
    badm = copy.deepcopy(m); badm["P"]["32"]["log_mse"]["mean"] += 1e-6; assert any("log_mse" in x for x in c.check_arrays(ps, badm, n=N))
    bad = copy.deepcopy(ps); bad["E"]["0"]["t60"][1] += 1.0; badm = copy.deepcopy(m); badm["E"]["0"]["t60"]["mean"] = sum(bad["E"]["0"]["t60"]) / N
    assert any("P and E differ at k=0" in x for x in c.check_arrays(bad, badm, n=N))
    bad = copy.deepcopy(ps); del bad["E"]["508"]; assert any("E k=508 missing" in x for x in c.check_arrays(bad, m, n=N))


def test_delay_flips_are_bounded_integers_equal_everywhere():
    good = {str(k): 0 for k in c.SPECTRAL}; good["480"] = 4
    assert c.check_delay_flips(good, dict(good)) == []
    assert c.check_delay_flips({str(k): True for k in c.SPECTRAL}, {str(k): True for k in c.SPECTRAL})
    assert c.check_delay_flips({**good, "32": -1}, {**good, "32": -1})
    assert c.check_delay_flips({**good, "32": c.EXPECTED_N * c.NUM_SHOT + 1}, {**good, "32": c.EXPECTED_N * c.NUM_SHOT + 1})
    assert c.check_delay_flips(good, {**good, "32": 1})
    assert c.check_delay_flips({k: v for k, v in good.items() if k != "4"}, good)


def test_meta_must_match_the_pre_registered_settings():
    meta = {"backbone": "simple", "checkpoint": "ckpt/xRIR_simple_8_shot/epoch_12.pth", "manifest_hash": c.MANIFEST_HASH, "manifest_seed": 0, "manifest_path": c.MANIFEST,
            "n_samples": c.EXPECTED_N, "max_samples": 0, "batch_size": 16, "batch_canonical": True, "tf32": False, "gl_seed": 0,
            "yaw_cols": list(c.SPECTRAL), "acoustic_cols": list(c.ACOUSTIC), "e_acoustic_cols": list(c.E_ACOUSTIC)}
    assert c.check_meta(meta, "simple", "ckpt/xRIR_simple_8_shot/epoch_12.pth") == []
    for key, val in [("tf32", True), ("batch_size", 1), ("gl_seed", 1), ("manifest_seed", 1), ("max_samples", 32), ("yaw_cols", [0, 32])]:
        bad = dict(meta); bad[key] = val
        assert c.check_meta(bad, "simple", "ckpt/xRIR_simple_8_shot/epoch_12.pth"), key


def test_provenance_requires_closure_environment_and_data_identity(tmp_path, monkeypatch):
    run = tmp_path / "sweep_x"; run.mkdir(); (run / "per_sample_yaw.json").write_text("{}"); (run / "metrics_yaw.json").write_text('{"meta": {"elapsed_min": 1}}')
    ck = tmp_path / "ck.pth"; ck.write_bytes(b"weights"); record = tmp_path / "record"; record.mkdir(); monkeypatch.setattr(c.bp, "RECORD", str(record))
    log = record / "yaw_rotation_degradation_2026-09-06_15:41:19_sweep_x.log"; log.write_text("log")
    rec = [{"path": "eval_yaw_rotation.py", "reviewed_blob_sha256": "a", "working_tree_sha256": "a"}]
    pv = {"evaluator_sha256": c.bp.REVIEWED_EVAL_SHA, "reviewed_evaluator_commit": c.bp.REVIEWED_COMMIT, "source_closure_sha256": "dig", "source_closure": rec, "git_sha": c.LAUNCHER_HEAD,
          "per_sample_sha256": c.bp.sha(str(run / "per_sample_yaw.json")), "metrics_sha256": c.bp.sha(str(run / "metrics_yaw.json")), "checkpoint": str(ck), "checkpoint_sha256": c.bp.sha(str(ck)),
          "environment": {k: "x" for k in c.ENV_KEYS}, "data_identity": {"manifest_hash": c.MANIFEST_HASH, "inventory_missing": 0, "inventory_files": 3, "manifest_file_sha256": "mf", "inventory_sha256": "inv"},
          "log": str(log), "log_sha256": c.bp.sha(str(log)), "run_start": c.bp.run_start({"log": str(log)}, str(run))[0].isoformat()}
    gate = {"checkpoint_sha256": c.bp.sha(str(ck))}
    assert c.check_provenance(pv, str(run), rec, "dig", gate, "mf") == []
    for key, val, needle in [("source_closure_sha256", "other", "closure digest"), ("git_sha", "0" * 40, "HEAD"), ("evaluator_sha256", "0" * 64, "reviewed commit"),
                             ("environment", {}, "environment"), ("data_identity", {"manifest_hash": c.MANIFEST_HASH, "inventory_missing": 3, "manifest_file_sha256": "mf", "inventory_sha256": "inv"}, "data identity")]:
        bad = dict(pv); bad[key] = val
        assert any(needle in x for x in c.check_provenance(bad, str(run), rec, "dig", gate, "mf")), key
    assert any("gate sidecar" in x or "checkpoint" in x for x in c.check_provenance(pv, str(run), rec, "dig", {"checkpoint_sha256": "zzz"}, "mf"))
    assert any("stale" in x or "data identity" in x for x in c.check_provenance(pv, str(run), rec, "dig", gate, "changed"))


def test_the_pinned_grids_match_the_summarizer_constants():
    sys.path.insert(0, ASSETS.split("/worklog/")[0])
    from tools import summarize_yaw as sy
    assert tuple(sy.PREREGISTERED_SPECTRAL_COLS) == c.SPECTRAL and tuple(sy.PREREGISTERED_ACOUSTIC_COLS) == c.ACOUSTIC and tuple(sy.PREREGISTERED_E_ACOUSTIC_COLS) == c.E_ACOUSTIC
    assert not math.isfinite(float("nan")) and c.is_num(1) and not c.is_num(True) and not c.is_num("1")


# ---------------------------------------------------------------- round-4 additions
def test_gate_requires_the_exact_band_rule_an_existing_summary_and_finite_rows():
    g = good_gate(); g["gate_rows"] = [{"metric": "edt", "diff": 0.1}]; g["band_terms"] = {"control": {"s_ref": 0.1}}
    assert c.check_gate_json(g, summary_sha="abc", decision_copy_sha="abc") == []
    g2 = dict(g); g2["band_rule"] = "v3 (something else)"
    assert any("band rule" in x for x in c.check_gate_json(g2, summary_sha="abc", decision_copy_sha="abc"))
    assert any("summary file missing" in x for x in c.check_gate_json(g, summary_sha=None, decision_copy_sha="abc"))
    g3 = dict(g); g3["gate_rows"] = [{"metric": "edt", "diff": float("nan")}]
    assert any("non-finite" in x for x in c.check_gate_json(g3, summary_sha="abc", decision_copy_sha="abc"))
    g4 = dict(g); g4["gate_rows"] = []
    assert any("gate rows" in x for x in c.check_gate_json(g4, summary_sha="abc", decision_copy_sha="abc"))
    t = datetime.datetime(2026, 9, 6, 15, 41, tzinfo=datetime.timezone.utc)
    assert any("chronology" in x for x in c.check_gate_json(g, sweep_start=t, summary_sha="abc", decision_copy_sha="abc"))


def test_spectral_counts_all_null_means_and_exact_grids():
    ps, m = synthetic_run()
    badm = copy.deepcopy(m); badm["P"]["32"]["loss"]["n_valid"] = N - 1
    assert any("loss" in x for x in c.check_arrays(ps, badm, n=N))
    bad = copy.deepcopy(ps); badm = copy.deepcopy(m); bad["P"]["32"]["t60"] = [None] * N; badm["P"]["32"]["t60"] = {"mean": float("inf"), "n_valid": 0, "n_nan": N}
    assert any("non-null aggregate mean" in x for x in c.check_arrays(bad, badm, n=N))
    badm["P"]["32"]["t60"]["mean"] = None
    assert not any("t60" in x for x in c.check_arrays(bad, badm, n=N))
    meta = {"backbone": "simple", "checkpoint": "ckpt/xRIR_simple_8_shot/epoch_12.pth", "manifest_hash": c.MANIFEST_HASH, "manifest_seed": 0, "manifest_path": c.MANIFEST,
            "n_samples": c.EXPECTED_N, "max_samples": 0, "batch_size": 16, "batch_canonical": True, "tf32": False, "gl_seed": 0,
            "yaw_cols": list(c.SPECTRAL), "acoustic_cols": list(c.ACOUSTIC), "e_acoustic_cols": list(c.E_ACOUSTIC)}
    assert c.check_meta(meta, "simple", "ckpt/xRIR_simple_8_shot/epoch_12.pth") == []
    frac = dict(meta); frac["yaw_cols"] = [float(k) for k in c.SPECTRAL]
    assert any("exact integers" in x for x in c.check_meta(frac, "simple", "ckpt/xRIR_simple_8_shot/epoch_12.pth"))
    boolish = dict(meta); boolish["acoustic_cols"] = [True] + list(c.ACOUSTIC)[1:]
    assert c.check_meta(boolish, "simple", "ckpt/xRIR_simple_8_shot/epoch_12.pth")


def test_decomposition_is_pinned_to_the_preregistered_probe():
    good = {"k": 32, "n_batches": 1, "tokens_rel_change": 1e-7, "pooled_rel_change": 0.01, "coord_rel_change": 0.7, "logspec_rel_change": 0.003}
    assert c.check_decomposition(good, "sweep_cyl") == []
    assert c.check_decomposition({**good, "k": 64}, "sweep_cyl")
    assert c.check_decomposition({**good, "n_batches": 2}, "sweep_cyl")
    assert c.check_decomposition({**good, "coord_rel_change": float("nan")}, "sweep_cyl")
    assert c.check_decomposition(None, "sweep_cyl")
    assert c.check_decomposition(good, "sweep_control")
    assert c.check_decomposition(None, "sweep_control") == []


def test_delay_flips_reject_non_digit_keys():
    good = {str(k): 0 for k in c.SPECTRAL}
    assert c.check_delay_flips({**{k: v for k, v in good.items() if k != "32"}, "32.0": 0}, good)


def test_binding_report_must_match_live_sidecars_and_gate_inputs(tmp_path, monkeypatch):
    run = tmp_path / "sweep_x"; run.mkdir(); side = run / "provenance.json"; side.write_text("{}")
    inp = tmp_path / "gate.json"; inp.write_text("{}")
    import hashlib, subprocess
    blob = subprocess.check_output(["git", "show", f"{c.bp.GATE_PRODUCER_COMMIT}:tools/summarize_yaw.py"], cwd=ASSETS.split("/worklog/")[0])
    gi = {k: {"path": str(inp), "sha256": c.bp.sha(str(inp))} for k in ("exp01_control", "exp01_cyl", "gate_json", "gate_summary", "decision_v1", "decision_v2", "decision_v3")}
    cwd = os.getcwd(); os.chdir(ASSETS.split("/worklog/")[0])
    try:
        gi.update({"gate_producer_commit": c.bp.GATE_PRODUCER_COMMIT, "gate_producer_sha256": hashlib.sha256(blob).hexdigest(), "gate_producer_closure": c.bp.gate_producer_closure()})
    finally:
        os.chdir(cwd)
    report = {"runs": {str(run): {"sidecar_sha256": c.bp.sha(str(side))}}, "gate_inputs": gi}
    cwd = os.getcwd(); os.chdir(ASSETS.split("/worklog/")[0])
    try:
        assert c.check_report(report, "same", "same", [str(run)]) == []
        assert any("assets copy" in x for x in c.check_report(report, "other", "same", [str(run)]))
        side.write_text('{"changed": 1}')
        assert any("live sidecar" in x for x in c.check_report(report, "same", "same", [str(run)]))
        assert any("no record" in x for x in c.check_report(report, "same", "same", [str(tmp_path / "sweep_y")]))
        inp.write_text("tampered")
        assert any("gate input" in x for x in c.check_report(report, "same", "same", [str(run)]))
        bad = copy.deepcopy(report); bad["gate_inputs"]["gate_producer_sha256"] = "0" * 64; side.write_text("{}")
        assert any("gate producer commit/digest" in x for x in c.check_report(bad, "same", "same", [str(run)]))
    finally:
        os.chdir(cwd)


def test_provenance_requires_live_environment_data_identity_and_log(tmp_path, monkeypatch):
    run = tmp_path / "sweep_x"; run.mkdir(); (run / "per_sample_yaw.json").write_text("{}"); (run / "metrics_yaw.json").write_text('{"meta": {"elapsed_min": 1}}')
    ck = tmp_path / "ck.pth"; ck.write_bytes(b"weights"); record = tmp_path / "record"; record.mkdir(); monkeypatch.setattr(c.bp, "RECORD", str(record))
    log = record / "yaw_rotation_degradation_2026-09-06_15:41:19_sweep_x.log"; log.write_text("log")
    rec = [{"path": "eval_yaw_rotation.py", "reviewed_blob_sha256": "a", "working_tree_sha256": "a"}]
    env = {k: "x" for k in c.ENV_KEYS}; data = {"manifest_hash": c.MANIFEST_HASH, "inventory_missing": 0, "inventory_files": 3, "manifest_file_sha256": "mf", "inventory_sha256": "inv"}
    pv = {"evaluator_sha256": c.bp.REVIEWED_EVAL_SHA, "reviewed_evaluator_commit": c.bp.REVIEWED_COMMIT, "source_closure_sha256": "dig", "source_closure": rec, "git_sha": c.LAUNCHER_HEAD,
          "per_sample_sha256": c.bp.sha(str(run / "per_sample_yaw.json")), "metrics_sha256": c.bp.sha(str(run / "metrics_yaw.json")), "checkpoint": str(ck), "checkpoint_sha256": c.bp.sha(str(ck)),
          "environment": env, "data_identity": data, "log": str(log), "log_sha256": c.bp.sha(str(log)), "run_start": c.bp.run_start({"log": str(log)}, str(run))[0].isoformat()}
    gate = {"checkpoint_sha256": c.bp.sha(str(ck))}
    assert c.check_provenance(pv, str(run), rec, "dig", gate, "mf", live_env=env, live_data=data) == []
    assert any("live environment" in x for x in c.check_provenance(pv, str(run), rec, "dig", gate, "mf", live_env={**env, "torch": "9.9"}, live_data=data))
    assert any("live recomputation" in x for x in c.check_provenance(pv, str(run), rec, "dig", gate, "mf", live_env=env, live_data={**data, "inventory_sha256": "zzz"}))
    log.write_text("edited")
    assert any("run log" in x for x in c.check_provenance(pv, str(run), rec, "dig", gate, "mf", live_env=env, live_data=data))
    pv2 = dict(pv); pv2["git_sha"] = "f" * 40
    assert not any("HEAD" in x for x in c.check_provenance(pv2, str(run), rec, "dig", gate, "mf", head="f" * 40, live_env=env, live_data=data))


def test_k0_content_of_a_gate_run_is_checked():
    ps, m = synthetic_run(); ents = manifest_entries()[:N]
    ps["query"] = [e["query"] for e in ents]; ps["index"] = [e["index"] for e in ents]
    ok = c.check_k0_content(ps, m, ents, n=N); assert [x for x in ok if "room count" not in x] == []
    bad = copy.deepcopy(ps); bad["P"]["0"]["loss"][0] = float("nan")
    assert any("loss" in x for x in c.check_k0_content(bad, m, ents, n=N))
    bad = copy.deepcopy(ps); bad["query"] = list(reversed(bad["query"]))
    assert any("manifest" in x for x in c.check_k0_content(bad, m, ents, n=N))


def test_changed_producer_labels_propagate_to_the_page(tmp_path):
    """The page must render whatever names/units the producer serialises (Codex round-4 finding 4): alter them and regenerate."""
    import json, subprocess
    src = os.path.join(ASSETS.split("/worklog/")[0], "ckpt", "yaw_rotation", "_explore32_v4_stats.json")
    if not os.path.isfile(src):
        pytest.skip("exploratory v4 sample not present")
    st = json.load(open(src))
    st["config"]["run_descriptions"]["cyl"] = "ZZ_CYL_LABEL_ZZ"; st["config"]["metric_names"]["c50"] = "ZZ_C50_NAME_ZZ"; st["config"]["metric_units"]["edt"] = "ZZunitZZ"
    st["k0_note"] = "ZZ_K0_NOTE_ZZ"; st["h2"]["equivalence_rule"] = "ZZ_TOST_RULE_ZZ"
    alt = tmp_path / "alt.json"; alt.write_text(json.dumps(st)); out = tmp_path / "page.html"
    py = os.path.expanduser("~/miniconda3/envs/xRIR/bin/python")
    r = subprocess.run([py, os.path.join(ASSETS, "make_results_html.py"), "--stats", str(alt), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    html = out.read_text()
    for needle in ("ZZ_CYL_LABEL_ZZ", "ZZ_C50_NAME_ZZ", "ZZunitZZ", "ZZ_K0_NOTE_ZZ", "ZZ_TOST_RULE_ZZ"):
        assert needle in html, needle
    assert "CylindricalViT + xRIR" not in html and "SimpleViT control" not in html
    del st["h1"]["rule"]; alt.write_text(json.dumps(st))
    r = subprocess.run([py, os.path.join(ASSETS, "make_results_html.py"), "--stats", str(alt), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode != 0 and "producer fields missing" in (r.stdout + r.stderr)


# ---------------------------------------------------------------- round-5 additions
def test_gate_run_k0_content_applies_the_sweep_rules():
    ps, m = synthetic_run(); ents = manifest_entries()[:N]
    ps["query"] = [e["query"] for e in ents]; ps["index"] = [e["index"] for e in ents]
    base = [x for x in c.check_k0_content(ps, m, ents, n=N) if "room count" not in x]; assert base == []
    badm = copy.deepcopy(m); badm["P"]["0"]["loss"]["n_nan"] = 777
    assert any("n_nan" in x for x in c.check_k0_content(ps, badm, ents, n=N))
    bad = copy.deepcopy(ps); badm = copy.deepcopy(m); bad["P"]["0"]["t60"] = [None] * N; bad["E"]["0"]["t60"] = [None] * N
    badm["P"]["0"]["t60"] = {"mean": 123.0, "n_valid": 0, "n_nan": N}; badm["E"]["0"]["t60"] = {"mean": None, "n_valid": 0, "n_nan": N}
    assert any("non-null aggregate mean" in x for x in c.check_k0_content(bad, badm, ents, n=N))
    bad = copy.deepcopy(ps); badm = copy.deepcopy(m); bad["E"]["0"]["edt"][0] += 1.0; badm["E"]["0"]["edt"]["mean"] = sum(bad["E"]["0"]["edt"]) / N
    assert any("P and E differ" in x for x in c.check_k0_content(bad, badm, ents, n=N))


def test_gate_run_meta_requires_exact_integer_grids():
    meta = {"backbone": "simple", "checkpoint": "ckpt/xRIR_simple_8_shot/epoch_12.pth", "batch_size": 16, "tf32": False, "batch_canonical": True, "gl_seed": 0, "manifest_seed": 0,
            "manifest_hash": c.MANIFEST_HASH, "n_samples": c.EXPECTED_N, "max_samples": 0, "yaw_cols": [0], "acoustic_cols": [0], "e_acoustic_cols": []}
    assert c.check_k0_meta(meta, "simple", "ckpt/xRIR_simple_8_shot/epoch_12.pth", 16, False, 0, 0) == []
    for key, val in [("yaw_cols", [0.25]), ("acoustic_cols", [False]), ("e_acoustic_cols", [32]), ("yaw_cols", [0, 32])]:
        bad = dict(meta); bad[key] = val
        assert any("exact" in x for x in c.check_k0_meta(bad, "simple", "ckpt/xRIR_simple_8_shot/epoch_12.pth", 16, False, 0, 0)), (key, val)
    assert c.check_k0_meta(meta, "simple", "ckpt/xRIR_simple_8_shot/epoch_12.pth", 1, False, 0, 0)


def test_sweep_chronology_requires_a_timestamp_for_every_run(tmp_path):
    import json
    good = tmp_path / "sweep_a"; good.mkdir(); (good / "provenance.json").write_text(json.dumps({"log": "x/yaw_rotation_degradation_2026-09-06_15:41:19_sweep_a.log"}))
    bad = tmp_path / "sweep_b"; bad.mkdir(); (bad / "provenance.json").write_text(json.dumps({"log": "x/nolog.log"}))
    t, why = c.sweep_start_time([str(good)]); assert t is not None and why == "ok"
    t, why = c.sweep_start_time([str(good), str(bad)]); assert t is None and "no parsable start timestamp" in why


def test_decomposition_consumed_by_the_producer_must_equal_the_aggregate():
    good = {"k": 32, "n_batches": 1, "tokens_rel_change": 1e-7, "pooled_rel_change": 0.01, "coord_rel_change": 0.7, "logspec_rel_change": 0.003}
    assert c.check_decomposition_pair(good, dict(good), "sweep_cyl") == []
    assert any("per-sample decomposition differs" in x for x in c.check_decomposition_pair(good, {**good, "coord_rel_change": 0.1}, "sweep_cyl"))
    assert any("per-sample decomposition differs" in x for x in c.check_decomposition_pair(good, None, "sweep_cyl"))
    assert c.check_decomposition_pair(None, None, "sweep_control") == []


def test_binder_refuses_drift_and_accepts_additive_migration():
    bp = c.bp
    existing = {"environment": {"torch": "2.0.1"}, "data_identity": {"inventory_sha256": "a"}, "log_sha256": "l1"}
    same = dict(existing); assert bp.binding_drift(existing, same) == []
    assert bp.binding_drift(existing, {**existing, "log_sha256": "l2"}) == ["log_sha256"]
    assert bp.binding_drift(existing, {**existing, "new_field": 1}) == []
    prev = {"written": "t0", "source_closure_sha256": "d", "runs": {"r1": {"sidecar_sha256": "s1"}}, "gate_inputs": {"gate_json": {"path": "p", "sha256": None}, "exp01_cyl": {"path": "q", "sha256": "e"}}}
    new = {"source_closure_sha256": "d", "runs": {"r1": {"sidecar_sha256": "s1"}}, "gate_inputs": {"gate_json": {"path": "p", "sha256": "now"}, "exp01_cyl": {"path": "q", "sha256": "e"}}}
    assert bp.report_drift(prev, new) == []                                    # gate input that was absent may appear
    assert any(x.startswith("runs.r1.sidecar_sha256") for x in bp.report_drift(prev, {**new, "runs": {"r1": {"sidecar_sha256": "s2"}}}))
    assert bp.report_drift(prev, {**new, "gate_inputs": {**new["gate_inputs"], "exp01_cyl": {"path": "q", "sha256": "changed"}}}) == ["gate_inputs.exp01_cyl"]


def test_skipping_the_gate_recomputation_is_not_acceptance():
    import subprocess
    py = os.path.expanduser("~/miniconda3/envs/xRIR/bin/python")
    repo = ASSETS.split("/worklog/")[0]
    if not os.path.isfile(os.path.join(repo, "ckpt", "yaw_rotation", "sweep_control", "provenance.json")):
        pytest.skip("canonical runs not present")
    r = subprocess.run([py, os.path.join(ASSETS, "check_sweep_acceptance.py"), "--no-recompute-gate"], cwd=repo, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": repo})
    assert r.returncode in (1, 2) and "ACCEPTANCE PASS" not in r.stdout
    assert ("ACCEPTANCE FAIL" if r.returncode == 1 else "ACCEPTANCE INCOMPLETE") in r.stdout


def test_changed_producer_labels_propagate_to_every_consumer_and_missing_mappings_are_refused(tmp_path):
    """Round-5 finding 6: HTML, Markdown and data.json must follow every producer mapping; nested missing keys are refused."""
    import json, subprocess
    repo = ASSETS.split("/worklog/")[0]; src = os.path.join(repo, "ckpt", "yaw_rotation", "_explore32_v4_stats.json")
    if not os.path.isfile(src):
        pytest.skip("exploratory v4 sample not present")
    py = os.path.expanduser("~/miniconda3/envs/xRIR/bin/python"); st = json.load(open(src))
    for l in st["config"]["labels"]:
        st["config"]["run_descriptions"][l] = f"ZZ_{l.upper()}_ZZ"
    for m in st["config"]["metric_names"]:
        st["config"]["metric_names"][m] = f"ZZ_{m}_NAME_ZZ"; st["config"]["metric_units"][m] = f"u{m}"
    st["config"]["condition_definitions"] = {"P": "ZZ_P_ZZ", "E": "ZZ_E_ZZ"}
    st["k0_note"] = "ZZ_K0_ZZ"; st["delay_flips_entity"] = "ZZ_DELAY_ZZ"; st["decomposition_note"] = "ZZ_DEC_ZZ"
    st["h1"]["rule"] = "ZZ_H1_ZZ"; st["h2"]["rule"] = "ZZ_H2_ZZ"; st["h2"]["equivalence_rule"] = "ZZ_TOST_ZZ"; st["convergence"]["rule"] = "ZZ_CONV_ZZ"
    needles = ["ZZ_CYL_ZZ", "ZZ_CONTROL_ZZ", "ZZ_edt_NAME_ZZ", "ZZ_c50_NAME_ZZ", "uedt", "ZZ_P_ZZ", "ZZ_E_ZZ", "ZZ_K0_ZZ", "ZZ_DELAY_ZZ", "ZZ_DEC_ZZ", "ZZ_TOST_ZZ"]
    md_only = ["ZZ_H1_ZZ", "ZZ_H2_ZZ", "ZZ_CONV_ZZ"]   # the HTML draft suppresses the verdict block (rules included) by design
    alt = tmp_path / "alt.json"; alt.write_text(json.dumps(st))
    html_out = tmp_path / "page.html"; md_out = tmp_path / "results.md"
    r = subprocess.run([py, os.path.join(ASSETS, "make_results_html.py"), "--stats", str(alt), "--out", str(html_out)], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-400:]
    r = subprocess.run([py, os.path.join(ASSETS, "make_results_md.py"), "--stats", str(alt), "--out", str(md_out), "--draft-ok"], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-400:]
    html, md = html_out.read_text(), md_out.read_text()
    for n in needles:
        assert n in html, ("html", n)
        assert n in md, ("md", n)
    for n in md_only:
        assert n in md, ("md", n)
    assert "ZZ_H2_ZZ" not in html   # draft: verdicts suppressed
    for leaked in ("EDT error", "C50 error", "cyl (cylindrical", "control (simple", "CylindricalViT"):
        assert leaked not in html and leaked not in md, leaked
    # draft pages do not carry verdict rules, but the markdown draft does; the HTML draft must still not silently drop a label
    for mutate, msg in [(lambda d: d["config"]["run_descriptions"].pop("cyl"), "run_descriptions"), (lambda d: d["config"]["metric_units"].pop("t60"), "metric"),
                        (lambda d: d["config"]["roles"].pop("primary"), "roles"), (lambda d: d["config"]["condition_definitions"].pop("E"), "condition"),
                        (lambda d: d["config"]["labels"].append("ghost"), "ghost")]:
        d = json.loads(alt.read_text()); mutate(d); bad = tmp_path / "bad.json"; bad.write_text(json.dumps(d))
        r = subprocess.run([py, os.path.join(ASSETS, "make_results_html.py"), "--stats", str(bad), "--out", str(html_out)], capture_output=True, text=True)
        assert r.returncode != 0, ("html accepted", msg)
        r = subprocess.run([py, os.path.join(ASSETS, "make_results_md.py"), "--stats", str(bad), "--out", str(md_out), "--draft-ok"], capture_output=True, text=True)
        assert r.returncode != 0, ("md accepted", msg)


# ---------------------------------------------------------------- round-6 additions
def test_binder_drift_covers_deletions_paths_and_monotonic_migrations():
    bp = c.bp
    existing = {"environment": {"torch": "2.0.1"}, "log_sha256": "l1", "run_start": "t"}
    assert bp.binding_drift(existing, {"environment": {"torch": "2.0.1"}, "log_sha256": "l1", "run_start": "t"}) == []
    assert "log_sha256" in bp.binding_drift(existing, {"environment": {"torch": "2.0.1"}, "run_start": "t"})          # deletion is drift
    prev = {"written": "t0", "run_set": "gate", "environment": {"torch": "2.0.1", "gpus": ["a"]}, "runs": {"g1": {"sidecar_sha256": "s1"}},
            "gate_inputs": {"gate_json": {"path": "p", "sha256": None}, "exp01_cyl": {"path": "q", "sha256": "e"}}}
    new = {"run_set": "all", "environment": {"torch": "2.0.1", "gpus": ["a"]}, "runs": {"g1": {"sidecar_sha256": "s1"}, "s1": {"sidecar_sha256": "x"}},
           "gate_inputs": {"gate_json": {"path": "p", "sha256": "now"}, "exp01_cyl": {"path": "q", "sha256": "e"}}}
    assert bp.report_drift(prev, new) == []                                                              # gate -> all, null -> digest, added run
    assert any("removed" in x for x in bp.report_drift(prev, {**new, "runs": {"s1": {"sidecar_sha256": "x"}}}))    # removed run
    assert any("gate_inputs.gate_json" in x for x in bp.report_drift(prev, {**new, "gate_inputs": {**new["gate_inputs"], "gate_json": {"path": "OTHER", "sha256": "now"}}}))   # path change
    assert any("environment" in x for x in bp.report_drift(prev, {**new, "environment": {"torch": "2.0.1"}}))    # nested deletion
    assert bp.report_drift({**prev, "run_set": "all"}, {**new, "run_set": "gate"}) == ["run_set"]


def test_scalar_counts_and_profiles_reject_booleans():
    ps, m = synthetic_run()
    badm = copy.deepcopy(m); badm["P"]["32"]["loss"]["n_nan"] = False
    assert any("loss" in x for x in c.check_arrays(ps, badm, n=N))
    meta = {"backbone": "simple", "checkpoint": "ckpt/xRIR_simple_8_shot/epoch_12.pth", "manifest_hash": c.MANIFEST_HASH, "manifest_seed": 0, "manifest_path": c.MANIFEST,
            "n_samples": c.EXPECTED_N, "max_samples": 0, "batch_size": 16, "batch_canonical": True, "tf32": False, "gl_seed": 0,
            "yaw_cols": list(c.SPECTRAL), "acoustic_cols": list(c.ACOUSTIC), "e_acoustic_cols": list(c.E_ACOUSTIC)}
    for key in ("max_samples", "gl_seed"):
        bad = dict(meta); bad[key] = False
        assert c.check_meta(bad, "simple", "ckpt/xRIR_simple_8_shot/epoch_12.pth"), key
    k0 = {"backbone": "cylindrical", "checkpoint": "ckpt/xRIR_cyl_8_shot/epoch_12.pth", "batch_size": True, "tf32": False, "batch_canonical": True, "gl_seed": 0, "manifest_seed": 0,
          "manifest_hash": c.MANIFEST_HASH, "n_samples": c.EXPECTED_N, "max_samples": 0, "yaw_cols": [0], "acoustic_cols": [0], "e_acoustic_cols": []}
    assert c.check_k0_meta(k0, "cylindrical", "ckpt/xRIR_cyl_8_shot/epoch_12.pth", 1, False, 0, 0)


def test_provenance_ties_checkpoint_identities_and_canonical_log_names(tmp_path, monkeypatch):
    run = tmp_path / "sweep_control"; run.mkdir(); (run / "per_sample_yaw.json").write_text("{}"); (run / "metrics_yaw.json").write_text('{"meta": {"elapsed_min": 1}}')
    ck = tmp_path / "ck.pth"; ck.write_bytes(b"w"); rec = [{"path": "eval_yaw_rotation.py", "reviewed_blob_sha256": "a", "working_tree_sha256": "a"}]
    record = tmp_path / "record"; record.mkdir(); monkeypatch.setattr(c.bp, "RECORD", str(record))
    log = record / "yaw_rotation_degradation_2026-09-06_15:41:19_sweep_control.log"; log.write_text("x")
    env = {k: "x" for k in c.ENV_KEYS}; data = {"manifest_hash": c.MANIFEST_HASH, "inventory_missing": 0, "inventory_files": 3, "manifest_file_sha256": "mf", "inventory_sha256": "inv"}
    pv = {"evaluator_sha256": c.bp.REVIEWED_EVAL_SHA, "reviewed_evaluator_commit": c.bp.REVIEWED_COMMIT, "source_closure_sha256": "dig", "source_closure": rec, "git_sha": c.LAUNCHER_HEAD,
          "per_sample_sha256": c.bp.sha(str(run / "per_sample_yaw.json")), "metrics_sha256": c.bp.sha(str(run / "metrics_yaw.json")), "checkpoint": str(ck), "checkpoint_sha256": c.bp.sha(str(ck)),
          "environment": env, "data_identity": data, "log": str(log), "log_sha256": c.bp.sha(str(log)), "run_start": c.bp.run_start({"log": str(log)}, str(run))[0].isoformat()}
    gate = {"checkpoint_sha256": c.bp.sha(str(ck))}; meta = {"checkpoint": str(ck)}
    assert c.check_provenance(pv, str(run), rec, "dig", gate, "mf", live_env=env, live_data=data, meta=meta, expected_ckpt=str(ck)) == []
    assert any("metrics meta" in x for x in c.check_provenance(pv, str(run), rec, "dig", gate, "mf", live_env=env, live_data=data, meta={"checkpoint": "other.pth"}, expected_ckpt=str(ck)))
    assert any("profile" in x for x in c.check_provenance(pv, str(run), rec, "dig", gate, "mf", live_env=env, live_data=data, meta=meta, expected_ckpt="other.pth"))
    txt = record / "yaw_rotation_degradation_2026-09-06_15:40:32_gate_decision_v3_PASSED.txt"; txt.write_text("d")
    bad = dict(pv); bad["log"] = str(txt); bad["log_sha256"] = c.bp.sha(str(txt))
    assert any("not canonical" in x for x in c.check_provenance(bad, str(run), rec, "dig", gate, "mf", live_env=env, live_data=data, meta=meta, expected_ckpt=str(ck)))
    aborted = record / "yaw_rotation_degradation_2026-09-06_15:41:19_sweep_control_ABORTED_evaluator_exit7.log"; aborted.write_text("x")
    bad = dict(pv); bad["log"] = str(aborted); bad["log_sha256"] = c.bp.sha(str(aborted))
    assert any("not canonical" in x for x in c.check_provenance(bad, str(run), rec, "dig", gate, "mf", live_env=env, live_data=data, meta=meta, expected_ckpt=str(ck)))


def test_report_check_requires_every_decision_copy_and_fresh_closure_digests(tmp_path):
    import hashlib, subprocess
    run = tmp_path / "sweep_x"; run.mkdir(); side = run / "provenance.json"; side.write_text("{}"); inp = tmp_path / "gate.json"; inp.write_text("{}")
    repo = ASSETS.split("/worklog/")[0]
    blob = subprocess.check_output(["git", "show", f"{c.bp.GATE_PRODUCER_COMMIT}:tools/summarize_yaw.py"], cwd=repo)
    gi = {k: {"path": str(inp), "sha256": c.bp.sha(str(inp))} for k in ("exp01_control", "exp01_cyl", "gate_json", "gate_summary", "decision_v1", "decision_v2", "decision_v3")}
    cwd = os.getcwd(); os.chdir(repo)
    try:
        gi.update({"gate_producer_commit": c.bp.GATE_PRODUCER_COMMIT, "gate_producer_sha256": hashlib.sha256(blob).hexdigest(), "gate_producer_closure": c.bp.gate_producer_closure()})
        report = {"runs": {str(run): {"sidecar_sha256": c.bp.sha(str(side))}}, "gate_inputs": gi}
        assert c.check_report(report, "same", "same", [str(run)]) == []
        bad = copy.deepcopy(report); del bad["gate_inputs"]["decision_v2"]
        assert any("decision_v2" in x for x in c.check_report(bad, "same", "same", [str(run)]))
        bad = copy.deepcopy(report); bad["gate_inputs"]["gate_producer_closure"][0]["working_tree_sha256"] = "0" * 64
        assert any("stale" in x for x in c.check_report(bad, "same", "same", [str(run)]))
    finally:
        os.chdir(cwd)


def test_consumers_refuse_missing_result_structures_and_data_json_follows_labels(tmp_path):
    import json, subprocess
    repo = ASSETS.split("/worklog/")[0]; src = os.path.join(repo, "ckpt", "yaw_rotation", "_explore32_v4_stats.json")
    if not os.path.isfile(src):
        pytest.skip("exploratory v4 sample not present")
    py = os.path.expanduser("~/miniconda3/envs/xRIR/bin/python"); base = json.load(open(src))
    def run(d, which):
        alt = tmp_path / "alt.json"; alt.write_text(json.dumps(d)); out = tmp_path / ("p.html" if which == "html" else "r.md")
        args = [py, os.path.join(ASSETS, f"make_results_{which}.py"), "--stats", str(alt), "--out", str(out)] + ([] if which == "html" else ["--draft-ok"])
        return subprocess.run(args, capture_output=True, text=True), out
    for mutate, msg in [(lambda d: d["acoustic"]["cyl"]["P"].pop("c50"), "acoustic section"), (lambda d: d["spectral"]["control"]["E"].pop("consistency"), "spectral section"),
                        (lambda d: d["delay_flips"].pop("cyl"), "delay flips"), (lambda d: d["h2"]["rows"].pop("edt"), "h2 rows"), (lambda d: d["k0"].clear(), "k0"),
                        (lambda d: d["decomposition"].pop("cyl"), "decomposition"),
                        (lambda d: [r["equivalence"].pop("cluster") for r in d["h2"]["rows"]["c50"] if r.get("equivalence")], "cluster TOST")]:
        d = copy.deepcopy(base); mutate(d)
        for which in ("html", "md"):
            r, _ = run(d, which)
            assert r.returncode != 0 and ("incomplete" in r.stdout + r.stderr or "lacks" in r.stdout + r.stderr), (which, msg, (r.stdout + r.stderr)[-200:])
    d = copy.deepcopy(base); d["config"]["run_descriptions"]["control"] = "ZZ_CTRL_ZZ"; d["config"]["metric_names"]["t60"] = "ZZ_T60_ZZ"
    r, out = run(d, "html"); assert r.returncode == 0, r.stderr[-300:]
    dj = json.load(open(os.path.splitext(str(out))[0] + ".data.json"))
    assert dj["run_descriptions"]["control"] == "ZZ_CTRL_ZZ" and dj["metric_names"]["t60"] == "ZZ_T60_ZZ" and dj["labels"] == d["config"]["labels"]


# ---------------------------------------------------------------- round-7 additions
def test_binder_transaction_is_idempotent_after_migration_and_preserves_timestamps(tmp_path):
    """Filesystem-level: bind, migrate additively, rerun identically (no writes), and refuse drift with nothing written."""
    import json
    bp = c.bp
    side = tmp_path / "run" / "provenance.json"; side.parent.mkdir(); report = tmp_path / "report.json"
    original = {"git_sha": "h", "checkpoint": "ck", "checkpoint_sha256": "c", "per_sample_sha256": "p", "metrics_sha256": "m", "log": "l"}
    side.write_text(json.dumps(original, indent=1))
    binding = {k: "v" for k in bp.BINDING_KEYS}
    new_pv = dict(original); new_pv.update(binding); new_pv["bound_at"] = "T1"; new_pv["binding_history"] = [{"at": "T1", "added": sorted(binding)}]
    rep = {"run_set": "gate", "runs": {str(side.parent): {"sidecar": str(side), "sidecar_sha256": "x1", "sidecar_after_binding": new_pv}}, "gate_inputs": {"g": {"path": "p", "sha256": None}}}
    written = bp.commit_transaction({str(side): new_pv}, {str(side): original}, dict(rep), str(report), "T1")
    assert str(side) in written and str(report) in written
    r1 = json.load(open(report)); assert r1["written"] == "T1" and "last_migration" not in r1
    # identical rerun: no drift, nothing written
    assert bp.report_drift(r1, dict(rep)) == []
    written = bp.commit_transaction({str(side): new_pv}, {str(side): new_pv}, dict(rep), str(report), "T2"); assert written == []
    # additive migration (gate -> all, gate input now present, extra sidecar key): allowed, timestamps preserved, history appended
    pv2 = dict(new_pv); pv2["extra"] = 1
    rep2 = {"run_set": "all", "runs": {str(side.parent): {"sidecar": str(side), "sidecar_sha256": "x2", "sidecar_after_binding": pv2}, "s1": {"sidecar_sha256": "y", "sidecar_after_binding": {}}},
            "gate_inputs": {"g": {"path": "p", "sha256": "now"}}}
    assert bp.report_drift(json.load(open(report)), dict(rep2)) == []
    bp.commit_transaction({str(side): pv2}, {str(side): new_pv}, dict(rep2), str(report), "T3")
    r3 = json.load(open(report)); assert r3["written"] == "T1" and r3["last_migration"] == "T3" and len(r3["binding_history"]) == 1
    assert json.load(open(side))["bound_at"] == "T1"
    # identical rerun after the migration: no drift (last_migration ignored), nothing written
    assert bp.report_drift(r3, dict(rep2)) == []
    assert bp.commit_transaction({str(side): pv2}, {str(side): pv2}, dict(rep2), str(report), "T4") == []
    # drift (a bound value changed): refused before any write -> files untouched
    pv_bad = dict(pv2); pv_bad["log_sha256"] = "changed"
    rep_bad = copy.deepcopy(rep2); rep_bad["runs"][str(side.parent)]["sidecar_after_binding"] = pv_bad; rep_bad["runs"][str(side.parent)]["sidecar_sha256"] = "x3"
    assert bp.report_drift(r3, rep_bad) and bp.binding_drift(pv2, pv_bad) == ["log_sha256"]
    before = (side.read_text(), report.read_text())
    # main() would stop here; assert the guard fires and that not calling commit leaves the files unchanged
    assert (side.read_text(), report.read_text()) == before


def test_acceptance_rejects_empty_run_start_boolean_inventory_and_non_canonical_gate_batches(tmp_path, monkeypatch):
    run = tmp_path / "sweep_control"; run.mkdir(); (run / "per_sample_yaw.json").write_text("{}"); (run / "metrics_yaw.json").write_text('{"meta": {"elapsed_min": 1}}')
    ck = tmp_path / "ck.pth"; ck.write_bytes(b"w"); rec = [{"path": "eval_yaw_rotation.py", "reviewed_blob_sha256": "a", "working_tree_sha256": "a"}]
    record = tmp_path / "record"; record.mkdir(); monkeypatch.setattr(c.bp, "RECORD", str(record))
    log = record / "yaw_rotation_degradation_2026-09-06_15:41:19_sweep_control.log"; log.write_text("x")
    env = {k: "x" for k in c.ENV_KEYS}; data = {"manifest_hash": c.MANIFEST_HASH, "inventory_missing": 0, "inventory_files": 3, "manifest_file_sha256": "mf", "inventory_sha256": "inv"}
    start = c.bp.run_start({"log": str(log)}, str(run))[0].isoformat()
    pv = {"evaluator_sha256": c.bp.REVIEWED_EVAL_SHA, "reviewed_evaluator_commit": c.bp.REVIEWED_COMMIT, "source_closure_sha256": "dig", "source_closure": rec, "git_sha": c.LAUNCHER_HEAD,
          "per_sample_sha256": c.bp.sha(str(run / "per_sample_yaw.json")), "metrics_sha256": c.bp.sha(str(run / "metrics_yaw.json")), "checkpoint": str(ck), "checkpoint_sha256": c.bp.sha(str(ck)),
          "environment": env, "data_identity": data, "log": str(log), "log_sha256": c.bp.sha(str(log)), "run_start": start}
    gate = {"checkpoint_sha256": c.bp.sha(str(ck))}; meta = {"checkpoint": str(ck)}
    ok = lambda p: c.check_provenance(p, str(run), rec, "dig", gate, "mf", live_env=env, live_data=data, meta=meta, expected_ckpt=str(ck))
    assert ok(pv) == []
    for val in (None, "", False):
        bad = dict(pv); bad["run_start"] = val
        assert any("run_start" in x for x in ok(bad)), val
    bad = dict(pv); del bad["run_start"]; assert any("run_start" in x for x in ok(bad))
    for key, val in (("inventory_missing", False), ("inventory_missing", 1), ("inventory_files", 0), ("inventory_files", True)):
        bad = dict(pv); bad["data_identity"] = {**data, key: val}
        assert any("data identity" in x for x in c.check_provenance(bad, str(run), rec, "dig", gate, "mf", live_env=env, live_data={**data, key: val}, meta=meta, expected_ckpt=str(ck))), (key, val)
    k0 = {"backbone": "cylindrical", "checkpoint": "ckpt/xRIR_cyl_8_shot/epoch_12.pth", "batch_size": 1, "tf32": False, "batch_canonical": True, "gl_seed": 0, "manifest_seed": 0,
          "manifest_hash": c.MANIFEST_HASH, "n_samples": c.EXPECTED_N, "max_samples": 0, "yaw_cols": [0], "acoustic_cols": [0], "e_acoustic_cols": []}
    assert c.check_k0_meta(k0, "cylindrical", "ckpt/xRIR_cyl_8_shot/epoch_12.pth", 1, False, 0, 0) == []
    for val in (False, None, 1):
        bad = dict(k0); bad["batch_canonical"] = val
        assert c.check_k0_meta(bad, "cylindrical", "ckpt/xRIR_cyl_8_shot/epoch_12.pth", 1, False, 0, 0), val


# ---------------------------------------------------------------- round-8 additions
def test_binder_projection_then_migration_then_identical_rerun_through_the_real_projection_path(tmp_path):
    """Integrated: project() -> report_drift() -> commit_transaction() twice, with the history append main() performs."""
    import json
    bp = c.bp
    run = tmp_path / "gate_x"; run.mkdir(); side = run / "provenance.json"; report = tmp_path / "report.json"
    original = {"git_sha": "h", "checkpoint": "ck", "checkpoint_sha256": "c", "per_sample_sha256": "p", "metrics_sha256": "m", "log": "l"}
    side.write_text(json.dumps(original, indent=1))
    binding = {k: "v" for k in bp.BINDING_KEYS}
    # first binding (pre-decision, gate set)
    plans = {str(side): (dict(original), dict(binding), False)}; records = {str(run): {"sidecar": str(side)}}
    projected = bp.project(plans, records, "T1")
    rep1 = {"run_set": "gate", "runs": records, "gate_inputs": {"g": {"path": "p", "sha256": None}}}
    assert bp.commit_transaction(projected, {str(side): original}, dict(rep1), str(report), "T1")
    pv1 = json.load(open(side)); assert pv1["bound_at"] == "T1" and len(pv1["binding_history"]) == 1
    # second binding: an additive sidecar key appears (e.g. a new binding field) and the gate input is now present -> migration
    binding2 = dict(binding); binding2["new_field"] = "x"
    drift = bp.binding_drift(pv1, binding2); assert drift == []
    unchanged = all(pv1.get(k) == v for k, v in binding2.items()); assert not unchanged
    plans2 = {str(side): (dict(pv1), binding2, False)}; records2 = {str(run): {"sidecar": str(side)}}
    projected2 = bp.project(plans2, records2, "T2")
    rep2 = {"run_set": "gate", "runs": records2, "gate_inputs": {"g": {"path": "p", "sha256": "now"}}}
    prev = json.load(open(report)); assert bp.report_drift(prev, dict(rep2)) == [], bp.report_drift(prev, dict(rep2))   # history append + additive key accepted
    bp.commit_transaction(projected2, {str(side): pv1}, dict(rep2), str(report), "T2")
    pv2 = json.load(open(side)); assert pv2["bound_at"] == "T1" and len(pv2["binding_history"]) == 2 and pv2["new_field"] == "x"
    r2 = json.load(open(report)); assert r2["written"] == "T1" and r2["last_migration"] == "T2"
    # identical rerun through the same path: unchanged sidecar, no drift, nothing written
    plans3 = {str(side): (dict(pv2), binding2, True)}; records3 = {str(run): {"sidecar": str(side)}}
    projected3 = bp.project(plans3, records3, "T3")
    assert bp.report_drift(r2, dict(rep2)) == [] and bp.commit_transaction(projected3, {str(side): pv2}, dict(rep2), str(report), "T3") == []
    # refusal through the same path: a changed bound value -> drift reported by both guards, and main() would not reach commit
    pv_bad = dict(pv2); pv_bad["log_sha256"] = "changed"
    plans4 = {str(side): (pv_bad, binding2, False)}; records4 = {str(run): {"sidecar": str(side)}}
    projected4 = bp.project(plans4, records4, "T4")
    # main() computes binding_drift(existing sidecar, fresh binding) before projecting; here the fresh binding says "v" while the
    # sidecar says "changed", so that guard fires and the transaction never reaches project()/commit_transaction()
    assert bp.binding_drift(pv_bad, binding2) == ["log_sha256"]
    assert json.load(open(side)) == pv2 and json.load(open(report)) == r2   # untouched
    # and if a tampered sidecar were somehow projected, the report guard catches the changed bound value
    tampered = copy.deepcopy(r2); tampered["runs"][str(run)]["sidecar_after_binding"]["log_sha256"] = "changed"; tampered["runs"][str(run)]["sidecar_sha256"] = "zzz"
    assert any("sidecar" in x for x in bp.report_drift(r2, tampered))
    # tampering with the retained history prefix is drift
    r_t = copy.deepcopy(r2); r_t["runs"][str(run)]["sidecar_after_binding"]["binding_history"][0]["at"] = "T0"
    assert any("history" in x for x in bp.report_drift(r2, r_t))
