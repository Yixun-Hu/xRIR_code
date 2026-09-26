"""Tests for ``tools/exp10_summarize.py`` (exp_10 ``yaw_pilot``, plan v3.1 tests 12-16).

The summariser turns per-sample files into the pilot's two readouts -- the signed
accuracy change and the prediction shift -- and their ratio.  The plan's paired-bootstrap
contract is what these tests pin: one shared draw per replicate, one shared mask, statuses
that never quietly become a multiple, and a refusal to mix two executions.
"""
import json
import os

import numpy as np
import pytest

from tools import exp10_summarize as summarize


# ------------------------------------------------- test 12: the paired-bootstrap contract

def test_shared_mask_drops_a_query_invalid_in_any_of_the_three_series():
    nan = float("nan")
    e0 = np.array([1.0, nan, 3.0, 4.0, 5.0])
    ek = np.array([1.5, 2.5, nan, 4.5, 5.5])
    gk = np.array([0.5, 0.5, 0.5, nan, 0.5])
    mask, exclusions = summarize.shared_mask(e0, ek, gk)

    np.testing.assert_array_equal(mask, [True, False, False, False, True])
    assert exclusions == {"invalid_at_0": 1, "invalid_at_alpha": 1, "gap_invalid": 1,
                          "excluded_total": 3, "n_mask": 2, "n_total": 5}


def test_shared_mask_of_a_shift_only_metric_keeps_every_query():
    gk = np.array([0.1, 0.2, 0.3])
    mask, exclusions = summarize.shared_mask(None, None, gk)
    np.testing.assert_array_equal(mask, [True, True, True])
    assert exclusions["n_mask"] == 3 and exclusions["excluded_total"] == 0


def test_bootstrap_draws_use_one_index_draw_per_replicate():
    rng = np.random.RandomState(0)
    deltas = rng.randn(40)
    gaps = np.abs(rng.randn(40)) + 0.5
    draws = summarize.bootstrap_draws(deltas, gaps, n_boot=200, seed=0)

    assert draws["delta"].shape == (200,) and draws["gap"].shape == (200,)
    # Delta*, G* and R* come from the SAME resample, so the ratio is exactly their quotient.
    np.testing.assert_allclose(draws["ratio"], draws["gap"] / draws["delta"], rtol=0,
                               atol=0)
    repeat = summarize.bootstrap_draws(deltas, gaps, n_boot=200, seed=0)
    np.testing.assert_array_equal(draws["delta"], repeat["delta"])
    assert not np.array_equal(
        draws["delta"], summarize.bootstrap_draws(deltas, gaps, n_boot=200, seed=1)["delta"])


def test_bootstrap_cell_recovers_a_synthetic_shift_inside_its_interval():
    rng = np.random.RandomState(1)
    e0 = np.abs(rng.randn(400)) + 1.0
    ek = e0 + 0.10 + 0.01 * rng.randn(400)
    gk = np.abs(rng.randn(400)) * 0.05 + 0.50
    cell = summarize.bootstrap_cell(e0, ek, gk, n_boot=2000, seed=0)

    assert cell["n"] == 400
    assert cell["delta"]["point"] == pytest.approx(0.10, abs=0.01)
    assert cell["delta"]["lo"] < 0.10 < cell["delta"]["hi"]
    assert cell["gap"]["lo"] < cell["gap"]["point"] < cell["gap"]["hi"]
    assert cell["ratio"]["status"] == "defined"
    assert cell["ratio"]["point"] == pytest.approx(cell["gap"]["point"] /
                                                   cell["delta"]["point"], rel=1e-12)
    assert cell["ratio"]["lo"] is not None and cell["ratio"]["hi"] is not None
    assert cell["mean_0"] == pytest.approx(float(e0.mean()))
    assert cell["mean_alpha"] == pytest.approx(float(ek.mean()))


def test_bootstrap_cell_excluding_one_query_moves_delta_gap_and_ratio_together():
    nan = float("nan")
    e0 = np.array([1.0, 1.0, 1.0, 1.0])
    ek = np.array([1.1, 1.1, 1.1, 9.0])
    gk = np.array([0.2, 0.2, 0.2, 9.0])
    full = summarize.bootstrap_cell(e0, ek, gk, n_boot=200, seed=0)
    dropped = summarize.bootstrap_cell(e0, ek, np.array([0.2, 0.2, 0.2, nan]),
                                       n_boot=200, seed=0)
    assert dropped["n"] == 3
    assert dropped["delta"]["point"] != full["delta"]["point"]
    assert dropped["gap"]["point"] != full["gap"]["point"]
    assert dropped["ratio"]["point"] != full["ratio"]["point"]
    assert dropped["exclusions"]["gap_invalid"] == 1


@pytest.mark.parametrize("delta_shift,expected", [
    (0.20, "defined"),
    (-0.20, "improvement"),
    (0.00005, "denominator uncertain"),
])
def test_ratio_status_covers_every_branch(delta_shift, expected):
    rng = np.random.RandomState(2)
    e0 = np.abs(rng.randn(300)) + 1.0
    ek = e0 + delta_shift + 0.05 * rng.randn(300)
    gk = np.abs(rng.randn(300)) * 0.02 + 0.40
    cell = summarize.bootstrap_cell(e0, ek, gk, n_boot=2000, seed=0)
    assert cell["ratio"]["status"] == expected
    if expected == "denominator uncertain":
        assert cell["ratio"]["lo"] is None and cell["ratio"]["hi"] is None
    else:
        assert cell["ratio"]["lo"] is not None


def test_undefined_status_counts_the_exact_zero_draws_and_never_divides():
    e0 = np.array([1.0, 2.0, 3.0, 4.0])
    ek = np.array([1.0, 2.0, 3.0, 4.0])      # every paired difference is exactly zero
    gk = np.array([0.1, 0.2, 0.3, 0.4])
    cell = summarize.bootstrap_cell(e0, ek, gk, n_boot=500, seed=0)

    assert cell["ratio"]["status"] == "undefined"
    assert cell["ratio"]["zero_draw_fraction"] == 1.0
    assert cell["ratio"]["point"] is None
    assert cell["ratio"]["reason"] == "observed delta = 0"
    assert cell["ratio"]["lo"] is None and cell["ratio"]["hi"] is None
    assert cell["delta"] == {"point": 0.0, "lo": 0.0, "hi": 0.0}


def test_a_single_zero_draw_is_enough_for_undefined():
    # Half the queries have a zero paired difference and half a positive one, so some
    # resamples are all-zero: the plan counts those draws, it never drops them.
    e0 = np.array([1.0, 1.0, 1.0])
    ek = np.array([1.0, 1.0, 1.2])
    gk = np.array([0.3, 0.3, 0.3])
    cell = summarize.bootstrap_cell(e0, ek, gk, n_boot=400, seed=0)
    assert cell["ratio"]["status"] == "undefined"
    assert 0.0 < cell["ratio"]["zero_draw_fraction"] < 1.0
    assert cell["ratio"]["point"] is not None       # the observed delta is not zero


def test_bootstrap_cell_reports_the_paired_contributions():
    e0 = np.array([1.0, 1.0, 1.0, 1.0])
    ek = np.array([1.4, 1.2, 0.8, 0.6])
    gk = np.array([0.4, 0.2, 0.2, 0.4])
    cell = summarize.bootstrap_cell(e0, ek, gk, n_boot=100, seed=0)
    contributions = cell["contributions"]
    assert contributions["fraction_worse"] == 0.5
    assert contributions["mean_positive_part"] == pytest.approx(0.15)
    assert contributions["mean_negative_part"] == pytest.approx(-0.15)


def test_bootstrap_cell_on_an_empty_mask_reports_n_zero_without_raising():
    nan = float("nan")
    cell = summarize.bootstrap_cell(np.array([nan, nan]), np.array([1.0, 2.0]),
                                    np.array([0.1, 0.2]), n_boot=50, seed=0)
    assert cell["n"] == 0
    assert cell["delta"] == {"point": None, "lo": None, "hi": None}
    assert cell["gap"] == {"point": None, "lo": None, "hi": None}
    assert cell["ratio"]["status"] == "undefined"
    assert cell["ratio"]["reason"] == "empty comparison mask"


def test_shift_only_cell_reports_the_gap_alone():
    gk = np.abs(np.random.RandomState(3).randn(50)) * 0.1
    cell = summarize.bootstrap_cell(None, None, gk, n_boot=500, seed=0)
    assert cell["n"] == 50
    assert cell["delta"] == {"point": None, "lo": None, "hi": None}
    assert cell["gap"]["lo"] < cell["gap"]["point"] < cell["gap"]["hi"]
    assert cell["ratio"]["status"] == "undefined"
    assert cell["ratio"]["reason"] == "no paired error for this metric"


def test_convergence_compares_the_bounds_of_two_seeds():
    rng = np.random.RandomState(4)
    e0 = np.abs(rng.randn(300)) + 1.0
    ek = e0 + 0.2 + 0.05 * rng.randn(300)
    gk = np.abs(rng.randn(300)) * 0.02 + 0.4
    first = summarize.bootstrap_cell(e0, ek, gk, n_boot=4000, seed=0)
    second = summarize.bootstrap_cell(e0, ek, gk, n_boot=4000, seed=1)
    report = summarize.convergence(first, second)

    assert report["converged"] is True
    assert report["delta"]["converged"] is True
    assert report["ratio"]["converged"] is True
    assert report["status_agrees"] is True
    assert report["ratio"]["max_relative_movement"] < 0.10


def test_convergence_treats_an_identical_zero_width_interval_as_converged():
    e0 = np.array([1.0, 2.0, 3.0])
    ek = np.array([1.0, 2.0, 3.0])
    gk = np.zeros(3)
    first = summarize.bootstrap_cell(e0, ek, gk, n_boot=200, seed=0)
    second = summarize.bootstrap_cell(e0, ek, gk, n_boot=200, seed=1)
    report = summarize.convergence(first, second)

    assert report["converged"] is True
    assert report["delta"]["zero_width"] is True
    assert report["ratio"]["converged"] == "not applicable"


def test_convergence_fails_when_a_zero_width_interval_moved():
    first = {"delta": {"point": 0.0, "lo": 0.0, "hi": 0.0},
             "gap": {"point": 0.0, "lo": 0.0, "hi": 0.0},
             "ratio": {"status": "undefined", "lo": None, "hi": None}}
    second = {"delta": {"point": 0.0, "lo": 0.0, "hi": 0.1},
              "gap": {"point": 0.0, "lo": 0.0, "hi": 0.0},
              "ratio": {"status": "undefined", "lo": None, "hi": None}}
    report = summarize.convergence(first, second)
    assert report["converged"] is False
    assert report["delta"]["converged"] is False


def test_headline_multiple_needs_convergence_and_an_agreeing_status():
    cell = {"ratio": {"status": "defined", "point": 5.0, "lo": 3.0, "hi": 8.0}}
    good = summarize.headline_multiple(cell, {"converged": True, "status_agrees": True})
    assert good["reportable"] is True and good["lower_bound"] == 3.0

    for report in ({"converged": False, "status_agrees": True},
                   {"converged": True, "status_agrees": False}):
        blocked = summarize.headline_multiple(cell, report)
        assert blocked["reportable"] is False
        assert blocked["reason"] == "unresolved Monte Carlo uncertainty"

    uncertain = summarize.headline_multiple(
        {"ratio": {"status": "denominator uncertain", "point": 5.0, "lo": None,
                   "hi": None}},
        {"converged": True, "status_agrees": True})
    assert uncertain["reportable"] is False
    assert uncertain["reason"] == "denominator uncertain"


# ------------------------------------------------ test 13: the room-cluster bootstrap

def test_room_ids_are_the_room_directories_of_the_queries():
    queries = ["Cat/Room_idx_1/S001_R001_hybrid_IR.wav",
               "Cat/Room_idx_1/S002_R001_hybrid_IR.wav",
               "Other/Room_idx_2/S001_R003_hybrid_IR.wav"]
    assert summarize.room_ids(queries) == ["Cat/Room_idx_1", "Cat/Room_idx_1",
                                           "Other/Room_idx_2"]


def test_room_bootstrap_keeps_every_query_of_a_drawn_room_and_weights_by_size():
    # Room A: 90 queries at 1.0, room B: 10 queries at 5.0.  Query-weighted, the three
    # possible draws give 1.0, 1.4 and 5.0; averaging per room first would give 3.0.
    deltas = np.concatenate([np.ones(90), np.full(10, 5.0)])
    gaps = deltas.copy()
    clusters = ["A"] * 90 + ["B"] * 10
    draws = summarize.bootstrap_draws(deltas, gaps, n_boot=400, seed=0,
                                      clusters=clusters)
    values = np.unique(np.round(draws["delta"], 10))
    np.testing.assert_allclose(sorted(values), [1.0, 1.4, 5.0])
    assert not np.any(np.isclose(draws["delta"], 3.0))


def test_room_bootstrap_counts_a_twice_drawn_room_twice():
    # One query at 0.0 in room A, two queries at 3.0 in room B.
    deltas = np.array([0.0, 3.0, 3.0])
    clusters = ["A", "B", "B"]
    draws = summarize.bootstrap_draws(deltas, deltas, n_boot=300, seed=1,
                                      clusters=clusters)
    values = np.unique(np.round(draws["delta"], 10))
    np.testing.assert_allclose(sorted(values), [0.0, 2.0, 3.0])


def test_room_level_cell_is_labelled_and_wider_than_the_query_level_one():
    rng = np.random.RandomState(5)
    per_room = []
    clusters = []
    for room in range(6):
        offset = 0.4 * room
        per_room.append(offset + 0.01 * rng.randn(40))
        clusters.extend(["Room_{}".format(room)] * 40)
    deltas = np.concatenate(per_room)
    e0 = np.abs(rng.randn(deltas.size)) + 2.0
    ek = e0 + deltas
    gk = np.abs(deltas) + 0.1

    query_cell = summarize.bootstrap_cell(e0, ek, gk, n_boot=2000, seed=0)
    room_cell = summarize.bootstrap_cell(e0, ek, gk, n_boot=2000, seed=0,
                                         clusters=clusters)
    assert query_cell["unit_of_resampling"] == "query"
    assert room_cell["unit_of_resampling"] == "room"
    assert room_cell["delta"]["point"] == pytest.approx(query_cell["delta"]["point"])
    query_width = query_cell["delta"]["hi"] - query_cell["delta"]["lo"]
    room_width = room_cell["delta"]["hi"] - room_cell["delta"]["lo"]
    assert room_width > 3 * query_width


def test_room_level_cell_masks_before_it_clusters():
    nan = float("nan")
    e0 = np.array([1.0, 1.0, 1.0, 1.0])
    ek = np.array([1.2, 1.2, 1.4, 1.4])
    gk = np.array([0.2, nan, 0.4, 0.4])
    clusters = ["A", "A", "B", "B"]
    cell = summarize.bootstrap_cell(e0, ek, gk, n_boot=200, seed=0, clusters=clusters)
    assert cell["n"] == 3
    assert cell["exclusions"]["gap_invalid"] == 1
    # Room A now holds one query and room B two: the draws are the query-weighted means.
    assert cell["delta"]["point"] == pytest.approx((0.2 + 0.4 + 0.4) / 3.0)


# ------------------------------------------- test 16: the run-identity guard (A2)

def _fixture_run(tmp_path, name, **kwargs):
    from tests.exp10_fixture import make_run

    out_dir = str(tmp_path / name)
    make_run(out_dir, **kwargs)
    return out_dir


def test_guard_runs_accepts_two_arms_of_one_protocol_family(tmp_path):
    first = _fixture_run(tmp_path, "released_k8")
    second = _fixture_run(tmp_path, "control_k8",
                          protocol_overrides={"checkpoint_sha256": "d" * 64})
    loaded = summarize.guard_runs([first, second])
    assert [entry["run_dir"] for entry in loaded] == [first, second]
    assert loaded[0]["meta"]["protocol_id"] != loaded[1]["meta"]["protocol_id"]


def test_guard_runs_refuses_two_executions_of_the_same_configuration(tmp_path):
    first = _fixture_run(tmp_path, "run_a")
    second = _fixture_run(tmp_path, "run_b")
    with pytest.raises(ValueError) as excinfo:
        summarize.guard_runs([first, second])
    assert "same protocol_id" in str(excinfo.value)


def test_guard_runs_refuses_same_device_runs_with_different_batch_settings(tmp_path):
    first = _fixture_run(tmp_path, "batch16")
    second = _fixture_run(tmp_path, "batch8",
                          protocol_overrides={"batch_size": 8,
                                              "checkpoint_sha256": "e" * 64})
    with pytest.raises(ValueError) as excinfo:
        summarize.guard_runs([first, second])
    assert "batch_size" in str(excinfo.value)


def test_guard_runs_refuses_same_device_runs_built_by_different_tool_versions(tmp_path):
    first = _fixture_run(tmp_path, "tool_a")
    second = _fixture_run(tmp_path, "tool_b",
                          protocol_overrides={"tool_sha256": "f" * 64,
                                              "checkpoint_sha256": "e" * 64})
    with pytest.raises(ValueError) as excinfo:
        summarize.guard_runs([first, second])
    assert "tool_sha256" in str(excinfo.value)


def test_guard_runs_refuses_a_per_sample_file_from_another_execution(tmp_path):
    run_dir = _fixture_run(tmp_path, "mixed",
                           per_sample_meta_overrides={"execution_id": "someone-else"})
    with pytest.raises(ValueError):
        summarize.guard_runs([run_dir])


def test_summarize_run_reads_one_execution_and_labels_it(tmp_path):
    run_dir = _fixture_run(tmp_path, "arm", n=24)
    summary = summarize.summarize_run(run_dir, n_boot=200)
    assert summary["arm"] == "fixture"
    assert summary["execution_id"] == summary["meta"]["execution_id"]
    assert summary["protocol_id"] == summary["meta"]["protocol_id"]
    assert sorted(summary["angles"]) == ["0", "128"]

    cell = summary["angles"]["128"]["EDT"]
    assert cell["query"]["n"] > 0
    assert cell["room"]["unit_of_resampling"] == "room"
    assert cell["convergence"]["status_agrees"] in (True, False)
    assert "headline" in cell
    zero = summary["angles"]["0"]["EDT"]
    assert zero["query"]["delta"]["point"] == 0.0
    assert zero["query"]["ratio"]["status"] == "undefined"
    shift_only = summary["angles"]["128"]["logspec_mad"]
    assert shift_only["query"]["gap"]["point"] > 0
    assert shift_only["query"]["delta"]["point"] is None


# ------------------------------------------------------- test 14: the controls table

def test_controls_table_passes_on_exact_repeats(tmp_path):
    run_dir = _fixture_run(tmp_path, "controls_ok")
    table = summarize.controls_table(run_dir)
    assert table["ok"] is True
    assert sorted(table["controls"]) == ["ctrl_full_turn", "ctrl_k128_repeat",
                                         "ctrl_zero_repeat"]
    for cell in table["controls"].values():
        assert cell["ok"] is True
        assert cell["wave_max_abs_diff"] == 0.0
        assert cell["logspec_max_abs_diff"] == 0.0
        assert cell["acoustic_max_abs_diff"] == 0.0


@pytest.mark.parametrize("metric,value,field", [
    ("wave_max_abs_diff", 1e-3, "wave_max_abs_diff"),
    ("logspec_max_abs_diff", 1e-5, "logspec_max_abs_diff"),
])
def test_controls_table_flags_a_waveform_or_logspec_mismatch(tmp_path, metric, value,
                                                             field):
    from tests.exp10_fixture import edit_json

    run_dir = _fixture_run(tmp_path, "controls_" + metric)
    edit_json(os.path.join(run_dir, "per_sample.json"),
              lambda payload: payload["controls"]["ctrl_k128_repeat"][metric]
              .__setitem__(1, value), rehash_meta=True)
    table = summarize.controls_table(run_dir)
    assert table["ok"] is False
    assert table["controls"]["ctrl_k128_repeat"]["ok"] is False
    assert table["controls"]["ctrl_k128_repeat"][field] == pytest.approx(value)
    assert table["controls"]["ctrl_zero_repeat"]["ok"] is True


def test_controls_table_flags_an_acoustic_mismatch_above_1e_9(tmp_path):
    from tests.exp10_fixture import edit_json

    run_dir = _fixture_run(tmp_path, "controls_acoustic")

    def nudge(payload):
        cell = payload["controls"]["ctrl_zero_repeat"]
        cell["c50_err"][0] = cell["c50_err"][0] + 1e-8

    edit_json(os.path.join(run_dir, "per_sample.json"), nudge, rehash_meta=True)
    table = summarize.controls_table(run_dir)
    assert table["ok"] is False
    cell = table["controls"]["ctrl_zero_repeat"]
    assert cell["ok"] is False
    assert cell["acoustic_max_abs_diff"] == pytest.approx(1e-8, rel=1e-3)
    assert cell["worst_acoustic_metric"] == "c50_err"


def test_controls_table_flags_a_validity_mismatch(tmp_path):
    from tests.exp10_fixture import edit_json

    run_dir = _fixture_run(tmp_path, "controls_validity")
    edit_json(os.path.join(run_dir, "per_sample.json"),
              lambda payload: payload["controls"]["ctrl_full_turn"]["edt_err"]
              .__setitem__(0, None), rehash_meta=True)
    table = summarize.controls_table(run_dir)
    assert table["ok"] is False
    assert table["controls"]["ctrl_full_turn"]["validity_mismatches"] == 1


# ---------------------------------------------------- test 15: the canonical outputs

@pytest.fixture
def two_arm_summary(tmp_path):
    """A summary over two arms, one of which improves on EDT (a signed negative bar)."""
    first = _fixture_run(tmp_path, "released_k8", n=24)
    second = _fixture_run(tmp_path, "control_k8", n=24, seed=11,
                          protocol_overrides={"checkpoint_sha256": "d" * 64},
                          meta_overrides={"arm": "control_k8"},
                          error_shift={"edt_err": -0.004})
    return summarize.build_summary([first, second], n_boot=200), [first, second]


def test_build_summary_records_its_inputs_and_their_hashes(two_arm_summary):
    summary, run_dirs = two_arm_summary
    assert [arm["arm"] for arm in summary["arms"]] == ["fixture", "control_k8"]
    assert summary["n_boot"] == 200 and summary["alpha"] == 0.05
    assert summary["seeds"] == [0, 1]
    for run_dir, entry in zip(run_dirs, summary["inputs"]):
        assert entry["run_dir"] == os.path.abspath(run_dir)
        assert entry["per_sample_sha256"] and entry["meta_sha256"]
        assert entry["protocol_id"] and entry["execution_id"]
    summarize.verify_inputs(summary)


def test_verify_inputs_refuses_a_summary_whose_inputs_moved(two_arm_summary, tmp_path):
    summary, run_dirs = two_arm_summary
    with open(os.path.join(run_dirs[0], "per_sample.json")) as fin:
        payload = json.load(fin)
    payload["query"][0] = payload["query"][0]
    with open(os.path.join(run_dirs[0], "per_sample.json"), "w") as fout:
        json.dump(payload, fout, indent=1)          # same content, different bytes
    with pytest.raises(ValueError) as excinfo:
        summarize.verify_inputs(summary)
    assert "per_sample.json" in str(excinfo.value)


def test_figure_data_carries_the_signed_change_and_the_shift(two_arm_summary):
    summary, _ = two_arm_summary
    rows = summarize.figure_data(summary)
    assert rows
    for row in rows:
        assert set(row) >= {"arm", "k", "angle_deg", "metric", "unit", "degradation",
                            "degradation_lo", "degradation_hi", "shift", "shift_lo",
                            "shift_hi", "n_mask", "ratio_status", "band"}
        assert row["k"] != 0                      # k = 0 is the reference, not a bar
    negatives = [row for row in rows
                 if row["arm"] == "control_k8" and row["metric"] == "EDT"
                 and row["degradation"] < 0]
    assert negatives, "the improving arm must produce a signed negative bar"
    edt = next(row for row in rows if row["metric"] == "EDT")
    assert edt["unit"] == "ms"                     # seconds are plotted in milliseconds


def test_write_outputs_writes_every_artefact_and_a_csv_that_matches_the_json(
        two_arm_summary, tmp_path):
    summary, _ = two_arm_summary
    out_dir = str(tmp_path / "summary")
    written = summarize.write_outputs(summary, out_dir)

    for name in ("yaw_pilot_summary.json", "yaw_pilot_tables.md", "yaw_pilot_gaps.csv",
                 "yaw_pilot_gaps_all_arms.png"):
        assert os.path.exists(os.path.join(out_dir, name)), name
    for arm in summary["arms"]:
        for extension in ("png", "pdf"):
            assert os.path.exists(os.path.join(
                out_dir, "yaw_pilot_gaps_{}.{}".format(arm["arm"], extension)))
    assert written["csv"].endswith("yaw_pilot_gaps.csv")

    import csv

    with open(os.path.join(out_dir, "yaw_pilot_gaps.csv")) as fin:
        rows = list(csv.DictReader(fin))
    expected = summarize.figure_data(summary)
    assert len(rows) == len(expected)
    def _maybe(text):
        return None if text == "" else float(text)

    for row, source in zip(rows, expected):
        assert row["arm"] == source["arm"]
        assert int(row["k"]) == source["k"]
        assert row["metric"] == source["metric"]
        # A shift-only panel has no accuracy change: the cell is empty in both.
        assert _maybe(row["degradation"]) == pytest.approx(source["degradation"])
        assert _maybe(row["shift"]) == pytest.approx(source["shift"])
        assert int(row["n_mask"]) == source["n_mask"]
        assert row["ratio_status"] == source["ratio_status"]
    assert any(row["degradation"] == "" for row in rows)        # logspec_mad
    assert all(row["shift"] != "" for row in rows)

    with open(os.path.join(out_dir, "yaw_pilot_summary.json")) as fin:
        canonical = json.load(fin)
    assert canonical["inputs"] == summary["inputs"]
    with open(os.path.join(out_dir, "yaw_pilot_tables.md")) as fin:
        markdown = fin.read()
    assert "### Metric 1 -- how far the prediction moved" in markdown
    statuses = {arm["angles"][str(k)][label]["query"]["ratio"]["status"]
                for arm in summary["arms"] for k in (0, 128)
                for label, _, _, _ in summarize.METRIC_TRIPLES}
    for status in statuses:
        assert status in markdown, status
    assert "historical baseline evaluation variability" in markdown
    assert "context only" in markdown


def test_the_figure_axes_leave_room_for_a_negative_bar(two_arm_summary, tmp_path):
    summary, _ = two_arm_summary
    rows = [row for row in summarize.figure_data(summary) if row["arm"] == "control_k8"]
    worst = min(row["degradation"] for row in rows if row["metric"] == "EDT")
    assert worst < 0
    figure = summarize.make_figure(summary, "control_k8",
                                   str(tmp_path / "fig.png"))
    try:
        axes = [ax for ax in figure.axes if ax.get_title().startswith("EDT")]
        assert axes, [ax.get_title() for ax in figure.axes]
        assert axes[0].get_ylim()[0] <= worst
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_write_outputs_refuses_inputs_that_no_longer_match(two_arm_summary, tmp_path):
    summary, run_dirs = two_arm_summary
    with open(os.path.join(run_dirs[1], "meta.json"), "a") as fout:
        fout.write("\n")
    with pytest.raises(ValueError):
        summarize.write_outputs(summary, str(tmp_path / "refused"))


def test_build_summary_refuses_two_arms_with_the_same_label(tmp_path):
    first = _fixture_run(tmp_path, "one")
    second = _fixture_run(tmp_path, "two",
                          protocol_overrides={"checkpoint_sha256": "d" * 64})
    with pytest.raises(ValueError) as excinfo:
        summarize.build_summary([first, second], n_boot=50)
    assert "arm label" in str(excinfo.value)


def test_room_cell_records_how_many_clusters_it_resampled():
    # A one-room subset makes the cluster bootstrap degenerate (every resample is the
    # same room), so the count has to be visible beside the interval.
    e0 = np.array([1.0, 1.0, 1.0, 1.0])
    ek = np.array([1.2, 1.3, 1.9, 2.0])      # the two rooms differ, so the draws differ
    gk = np.array([0.2, 0.3, 0.9, 1.0])
    one_room = summarize.bootstrap_cell(e0, ek, gk, n_boot=100, seed=0,
                                        clusters=["A"] * 4)
    two_rooms = summarize.bootstrap_cell(e0, ek, gk, n_boot=100, seed=0,
                                         clusters=["A", "A", "B", "B"])
    query = summarize.bootstrap_cell(e0, ek, gk, n_boot=100, seed=0)

    assert one_room["n_clusters"] == 1
    assert two_rooms["n_clusters"] == 2
    assert query["n_clusters"] is None
    assert one_room["delta"]["lo"] == one_room["delta"]["hi"]   # degenerate by construction
    assert two_rooms["delta"]["lo"] < two_rooms["delta"]["hi"]


# ------------------- round 2, finding 1: the per-sample binding, all the way to the summary

def test_build_summary_refuses_a_tampered_per_sample_file(tmp_path):
    from tests.exp10_fixture import edit_json

    run_dir = _fixture_run(tmp_path, "tampered", n=12)
    edit_json(os.path.join(run_dir, "per_sample.json"),
              lambda payload: payload["angles"]["128"]["logspec_mad"].__setitem__(0, 999.0))
    with pytest.raises(ValueError) as excinfo:
        summarize.build_summary([run_dir], n_boot=50)
    assert "per_sample_sha256" in str(excinfo.value)


def test_verify_inputs_refuses_a_run_that_no_longer_binds_its_per_sample_file(tmp_path):
    from tests.exp10_fixture import edit_json
    from tools.exp10_yaw_pilot import file_sha256

    run_dir = _fixture_run(tmp_path, "unbound", n=12)
    edit_json(os.path.join(run_dir, "meta.json"),
              lambda payload: payload.__setitem__("per_sample_sha256", "0" * 64))
    # A summary whose recorded hashes match the files on disk, but whose run no longer
    # binds its own per-sample file: the verification has to catch that too.
    summary = {"inputs": [{
        "run_dir": os.path.abspath(run_dir),
        "per_sample_sha256": file_sha256(os.path.join(run_dir, "per_sample.json")),
        "meta_sha256": file_sha256(os.path.join(run_dir, "meta.json"))}]}
    with pytest.raises(ValueError) as excinfo:
        summarize.verify_inputs(summary)
    assert "per_sample_sha256" in str(excinfo.value)


# ------------------- round 2, finding 3: the controls table validates its own evidence

def test_controls_table_refuses_a_declared_control_that_is_missing(tmp_path):
    from tests.exp10_fixture import edit_json

    run_dir = _fixture_run(tmp_path, "controls_absent")
    edit_json(os.path.join(run_dir, "per_sample.json"),
              lambda payload: payload["controls"].pop("ctrl_full_turn"),
              rehash_meta=True)
    table = summarize.controls_table(run_dir)
    assert table["ok"] is False
    assert table["missing_controls"] == ["ctrl_full_turn"]
    assert "ctrl_full_turn" not in table["controls"]
    assert table["controls"]["ctrl_zero_repeat"]["ok"] is True


def test_controls_table_refuses_a_control_no_spec_declared(tmp_path):
    from tests.exp10_fixture import edit_json

    run_dir = _fixture_run(tmp_path, "controls_extra")

    def add(payload):
        payload["controls"]["ctrl_mystery"] = payload["controls"]["ctrl_zero_repeat"]

    edit_json(os.path.join(run_dir, "per_sample.json"), add, rehash_meta=True)
    table = summarize.controls_table(run_dir)
    assert table["ok"] is False
    assert table["undeclared_controls"] == ["ctrl_mystery"]


def test_controls_table_refuses_a_null_deviation_instead_of_dropping_it(tmp_path):
    from tests.exp10_fixture import edit_json

    run_dir = _fixture_run(tmp_path, "controls_null")
    edit_json(os.path.join(run_dir, "per_sample.json"),
              lambda payload: payload["controls"]["ctrl_k128_repeat"]
              ["wave_max_abs_diff"].__setitem__(1, None), rehash_meta=True)
    table = summarize.controls_table(run_dir)
    cell = table["controls"]["ctrl_k128_repeat"]
    assert table["ok"] is False
    assert cell["ok"] is False
    assert cell["nonfinite_deviations"] == 1
    assert any("finite" in problem for problem in cell["problems"])


def test_controls_table_refuses_a_deviation_array_of_the_wrong_length(tmp_path):
    from tests.exp10_fixture import edit_json

    run_dir = _fixture_run(tmp_path, "controls_short")
    edit_json(os.path.join(run_dir, "per_sample.json"),
              lambda payload: payload["controls"]["ctrl_zero_repeat"].__setitem__(
                  "logspec_max_abs_diff",
                  payload["controls"]["ctrl_zero_repeat"]["logspec_max_abs_diff"][:-1]),
              rehash_meta=True)
    table = summarize.controls_table(run_dir)
    cell = table["controls"]["ctrl_zero_repeat"]
    assert table["ok"] is False and cell["ok"] is False
    assert any("length" in problem for problem in cell["problems"])


def test_controls_table_refuses_a_control_missing_a_required_array(tmp_path):
    from tests.exp10_fixture import edit_json

    run_dir = _fixture_run(tmp_path, "controls_incomplete")
    edit_json(os.path.join(run_dir, "per_sample.json"),
              lambda payload: payload["controls"]["ctrl_zero_repeat"].pop("c50_err"),
              rehash_meta=True)
    table = summarize.controls_table(run_dir)
    cell = table["controls"]["ctrl_zero_repeat"]
    assert table["ok"] is False and cell["ok"] is False
    assert any("c50_err" in problem for problem in cell["problems"])


# ---------------- round 2, finding 4: A3's uncertainty qualification comes first

def test_point_ratio_withholds_its_value_with_a_reason_before_dividing():
    assert summarize.point_ratio(0.5, 0.1) == (5.0, None)
    assert summarize.point_ratio(0.5, 0.0) == (None, "observed delta = 0")


@pytest.mark.parametrize("e0,ek,gk,expected", [
    (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]),
     "observed delta = 0"),
    (None, None, np.array([0.1, 0.2, 0.3]), "no paired error for this metric"),
])
def test_a_null_point_ratio_always_carries_its_reason(e0, ek, gk, expected):
    cell = summarize.bootstrap_cell(e0, ek, gk, n_boot=100, seed=0)
    assert cell["ratio"]["point"] is None
    assert cell["ratio"]["point_reason"] == expected


def test_an_empty_mask_records_why_its_point_ratio_is_null():
    nan = float("nan")
    cell = summarize.bootstrap_cell(np.array([nan, nan]), np.array([1.0, 2.0]),
                                    np.array([0.1, 0.2]), n_boot=50, seed=0)
    assert cell["ratio"]["point_reason"] == "empty comparison mask"


def test_convergence_records_the_status_of_both_seeds():
    first = {"delta": {"point": -0.3, "lo": -0.5, "hi": -0.1},
             "gap": {"point": 0.4, "lo": 0.3, "hi": 0.5},
             "ratio": {"status": "improvement", "lo": -4.0, "hi": -0.8}}
    second = {"delta": {"point": -0.1, "lo": -0.4, "hi": 0.2},
              "gap": {"point": 0.4, "lo": 0.3, "hi": 0.5},
              "ratio": {"status": "denominator uncertain", "lo": None, "hi": None}}
    report = summarize.convergence(first, second)
    assert report["status_agrees"] is False
    assert report["status_seed_0"] == "improvement"
    assert report["status_seed_1"] == "denominator uncertain"


def test_a_disagreeing_denominator_status_outranks_the_improvement_wording():
    # Seed 0's delta interval is entirely negative, seed 1's crosses zero: which of the
    # two the pilot would have reported is itself uncertain, so A3's qualification has to
    # come before any substantive wording about improving or worsening.
    first = {"delta": {"point": -0.3, "lo": -0.5, "hi": -0.1},
             "gap": {"point": 0.4, "lo": 0.3, "hi": 0.5},
             "ratio": {"status": "improvement", "point": -1.33, "lo": -4.0, "hi": -0.8}}
    second = {"delta": {"point": -0.1, "lo": -0.4, "hi": 0.2},
              "gap": {"point": 0.4, "lo": 0.3, "hi": 0.5},
              "ratio": {"status": "denominator uncertain", "point": -4.0, "lo": None,
                        "hi": None}}
    report = summarize.convergence(first, second)
    headline = summarize.headline_multiple(first, report)

    assert headline["reportable"] is False
    assert headline["reason"] == "unresolved Monte Carlo uncertainty"
    assert headline["status_seed_0"] == "improvement"
    assert headline["status_seed_1"] == "denominator uncertain"


def test_a_failed_convergence_outranks_the_uncertain_denominator_wording():
    cell = {"ratio": {"status": "denominator uncertain", "point": 5.0, "lo": None,
                      "hi": None}}
    headline = summarize.headline_multiple(
        cell, {"converged": False, "status_agrees": True,
               "status_seed_0": "denominator uncertain",
               "status_seed_1": "denominator uncertain"})
    assert headline["reason"] == "unresolved Monte Carlo uncertainty"


# ------------- round 2, finding 5: the standalone G on the broader gap population

def test_the_broader_gap_population_is_estimated_beside_the_paired_one():
    # The first query has no valid baseline error, so the paired readout must drop it --
    # and with it the largest gap.  Section 4 also asks for G on every query whose gap is
    # finite, reported separately: n = 4 and G = 26.5, not n = 3 and G = 2.
    nan = float("nan")
    e0 = np.array([nan, 1.0, 1.0, 1.0])
    ek = np.array([1.0, 1.1, 1.2, 1.3])
    gk = np.array([100.0, 1.0, 2.0, 3.0])

    paired = summarize.bootstrap_cell(e0, ek, gk, n_boot=200, seed=0)
    broader = summarize.gap_only_cell(gk, n_boot=200, seed=0)

    assert paired["n"] == 3
    assert paired["gap"]["point"] == pytest.approx(2.0)
    assert broader["n"] == 4
    assert broader["gap"]["point"] == pytest.approx(26.5)
    assert broader["population"] == summarize.BROADER_POPULATION
    assert broader["delta"]["point"] is None      # it is a shift, not a paired change
    assert broader["gap"]["lo"] <= broader["gap"]["point"] <= broader["gap"]["hi"]


def test_summarize_run_carries_the_broader_population_without_moving_the_paired_one(
        tmp_path):
    run_dir = _fixture_run(tmp_path, "broader", n=24)
    summary = summarize.summarize_run(run_dir, n_boot=100)
    entry = summary["angles"]["128"]["EDT"]

    assert entry["broader"]["population"] == summarize.BROADER_POPULATION
    assert entry["broader"]["query"]["n"] >= entry["query"]["n"]
    assert entry["broader"]["room"]["unit_of_resampling"] == "room"
    assert entry["broader"]["convergence"]["gap"]["converged"] in (True, False,
                                                                   "not applicable")
    # The paired readouts are untouched by its presence.
    assert entry["query"]["delta"]["point"] is not None
    assert entry["query"]["gap"]["point"] == pytest.approx(
        summarize.bootstrap_cell(
            [None if v is None else v
             for v in json.load(open(os.path.join(run_dir, "per_sample.json")))
             ["angles"]["0"]["edt_err"]],
            json.load(open(os.path.join(run_dir, "per_sample.json")))
            ["angles"]["128"]["edt_err"],
            json.load(open(os.path.join(run_dir, "per_sample.json")))
            ["angles"]["128"]["edt_gap"], n_boot=100, seed=0)["gap"]["point"])


def test_the_markdown_reports_the_broader_population_as_its_own_table(two_arm_summary,
                                                                      tmp_path):
    summary, _ = two_arm_summary
    out_dir = str(tmp_path / "broader_md")
    summarize.write_outputs(summary, out_dir)
    with open(os.path.join(out_dir, "yaw_pilot_tables.md")) as fin:
        markdown = fin.read()
    assert summarize.BROADER_POPULATION in markdown
    assert "Standalone G" in markdown


# ------------- round 2, finding 6: T60's denominators and the pipeline qualification

def test_the_absolute_seconds_t60_pair_is_summarised_like_the_other_metrics(tmp_path):
    run_dir = _fixture_run(tmp_path, "t60_abs", n=24)
    summary = summarize.summarize_run(run_dir, n_boot=100)
    entry = summary["angles"]["128"]["T60_abs"]

    assert entry["query"]["unit"] == "s"
    assert entry["query"]["delta"]["point"] is not None
    assert entry["query"]["gap"]["point"] is not None
    assert entry["query"]["gap"]["lo"] <= entry["query"]["gap"]["point"] <= \
        entry["query"]["gap"]["hi"]
    assert entry["room"]["unit_of_resampling"] == "room"
    assert "headline" in entry


def test_every_metric_carries_the_denominator_of_its_change_and_its_shift():
    for label in ("EDT", "C50", "T60", "T60_abs"):
        qualification = summarize.METRIC_QUALIFICATIONS[label]
        assert qualification["delta"] and qualification["gap"]
    assert "percentage point" in summarize.METRIC_QUALIFICATIONS["T60"]["delta"]
    assert "baseline prediction" in summarize.METRIC_QUALIFICATIONS["T60"]["gap"]
    assert "second" in summarize.METRIC_QUALIFICATIONS["T60_abs"]["gap"]


def test_the_canonical_json_and_markdown_carry_the_pipeline_qualification(
        two_arm_summary, tmp_path):
    summary, _ = two_arm_summary
    out_dir = str(tmp_path / "qualified")
    summarize.write_outputs(summary, out_dir)

    with open(os.path.join(out_dir, "yaw_pilot_summary.json")) as fin:
        canonical = json.load(fin)
    assert canonical["notes"]["pipeline"] == summarize.PIPELINE_NOTE
    assert canonical["notes"]["gl_free"] == summarize.GL_FREE_NOTE
    assert canonical["metric_qualifications"]["T60"]["gap"] == \
        summarize.METRIC_QUALIFICATIONS["T60"]["gap"]

    with open(os.path.join(out_dir, "yaw_pilot_tables.md")) as fin:
        markdown = fin.read()
    for text in (summarize.PIPELINE_NOTE, summarize.GL_FREE_NOTE,
                 summarize.METRIC_QUALIFICATIONS["T60"]["delta"],
                 summarize.METRIC_QUALIFICATIONS["T60"]["gap"]):
        assert text in markdown, text


def test_the_figure_footer_states_the_pipeline_qualification(two_arm_summary, tmp_path):
    summary, _ = two_arm_summary
    figure = summarize.make_figure(summary, "fixture", str(tmp_path / "footer.png"))
    try:
        footer = " ".join(text.get_text() for text in figure.texts)
        assert "Griffin-Lim" in footer
        assert "GL-free" in footer or "Griffin-Lim-free" in footer
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


# ---------------- round 2, finding 7: an unavailable estimate is not a measured zero

def _edt_axis(figure):
    axes = [ax for ax in figure.axes if ax.get_title().startswith("EDT")]
    assert axes, [ax.get_title() for ax in figure.axes]
    return axes[0]


def _bar_heights(ax):
    """The heights of the bars actually drawn (the context band is a polygon, not a bar)."""
    from matplotlib.container import BarContainer

    return [patch.get_height() for container in ax.containers
            if isinstance(container, BarContainer) for patch in container.patches]


def test_a_figure_omits_an_unavailable_bar_instead_of_drawing_it_at_zero(tmp_path):
    run_dir = _fixture_run(tmp_path, "unavailable", n=12, ks=(0, 128, 256))
    summary = summarize.build_summary([run_dir], n_boot=100)
    blank = summary["arms"][0]["angles"]["128"]["EDT"]
    for scope in ("query", "room"):
        blank[scope]["n"] = 0
        for key in ("delta", "gap"):
            blank[scope][key] = {"point": None, "lo": None, "hi": None}
    # An estimate that really is zero must still be drawn.
    summary["arms"][0]["angles"]["256"]["EDT"]["query"]["delta"]["point"] = 0.0

    rows = [row for row in summarize.figure_data(summary) if row["metric"] == "EDT"]
    assert [row["k"] for row in rows] == [128, 256]
    assert rows[0]["degradation"] is None and rows[0]["shift"] is None
    assert rows[1]["degradation"] == 0.0

    figure = summarize.make_figure(summary, "fixture", str(tmp_path / "unavailable.png"))
    try:
        ax = _edt_axis(figure)
        heights = _bar_heights(ax)
        assert len(heights) == 2, heights        # only k = 256: its change and its shift
        assert 0.0 in heights                    # the genuine zero survived
        annotations = " ".join(text.get_text() for text in ax.texts)
        assert "unavailable" in annotations
        assert "n=0" in annotations.replace(" ", "")
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_a_figure_draws_a_bar_without_error_bars_when_only_the_interval_is_missing(
        tmp_path):
    run_dir = _fixture_run(tmp_path, "no_interval", n=12, ks=(0, 128))
    summary = summarize.build_summary([run_dir], n_boot=100)
    cell = summary["arms"][0]["angles"]["128"]["EDT"]["query"]
    cell["delta"]["lo"] = cell["delta"]["hi"] = None

    figure = summarize.make_figure(summary, "fixture", str(tmp_path / "no_interval.png"))
    try:
        ax = _edt_axis(figure)
        heights = _bar_heights(ax)
        assert len(heights) == 2                 # the change bar is still drawn
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


# ------------- round 2, finding 9: the Markdown must show what the JSON already holds

def _markdown(summary, tmp_path, name):
    out_dir = str(tmp_path / name)
    summarize.write_outputs(summary, out_dir)
    with open(os.path.join(out_dir, "yaw_pilot_tables.md")) as fin:
        return fin.read()


def _table_rows(markdown, heading):
    """The data rows of the Markdown table under one heading.

    Cells are split on *unescaped* pipes, which is what a Markdown renderer does -- so a
    header that forgets to escape the pipes of ``|dwave|`` shows up here as more cells
    than the rows have.
    """
    import re

    lines = markdown.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(heading))
    rows = []
    for line in lines[start:]:
        if line.startswith("###") and not line.startswith(heading):
            break
        if line.startswith("|") and not set(line) <= set("|- "):
            cells = re.split(r"(?<!\\)\|", line)[1:-1]
            rows.append([cell.strip() for cell in cells])
    return rows[0], rows[1:]


def test_the_markdown_reports_log_mse_and_loss_with_their_units(two_arm_summary,
                                                                tmp_path):
    summary, _ = two_arm_summary
    markdown = _markdown(summary, tmp_path, "metric2")
    header, rows = _table_rows(markdown, "### Metric 2")

    assert "unit" in header
    metrics = {row[1] for row in rows}
    assert {"log_mse", "loss"} <= metrics
    units = {row[1]: row[2] for row in rows}
    assert units["log_mse"] == "log-magnitude^2"
    assert units["loss"] == "test loss"
    for row in rows:
        assert len(row) == len(header)


def test_the_markdown_room_table_carries_the_ratio_and_its_qualification(two_arm_summary,
                                                                         tmp_path):
    summary, _ = two_arm_summary
    markdown = _markdown(summary, tmp_path, "rooms")
    assert summarize.ROOM_QUALIFICATION in markdown
    header, rows = _table_rows(markdown, "### Room-cluster intervals")

    assert "R (room)" in header
    assert any("converged" in cell for cell in header)
    for row in rows:
        assert len(row) == len(header)
    # The room ratio of at least one cell is actually printed, not left blank.
    ratio_column = header.index("R (room)")
    assert any(row[ratio_column] not in ("", "-") for row in rows)


# --------------------- round 2, finding 11: a Markdown header with the right cell count

def test_the_controls_table_header_has_one_cell_per_column(two_arm_summary, tmp_path):
    summary, _ = two_arm_summary
    markdown = _markdown(summary, tmp_path, "controls_md")
    header, rows = _table_rows(markdown, "### Controls")

    assert rows, "the controls table has no data rows"
    for row in rows:
        assert len(row) == len(header), (header, row)
    assert any("dwave" in cell for cell in header)
    assert "problems" in header


# --------- round 2, finding 12: the cylindrical context band, keyed by backbone and K

def test_the_cylindrical_band_is_exp_04s_five_seed_sd_of_cylindrical_vit():
    # model_comparison.md, CylindricalViT K = 8: EDT 45.0629 +- 0.253312 ms,
    # C50 1.23094 +- 0.00593518 dB, T60 9.44256 +- 0.0135858 %.
    assert summarize.HISTORICAL_BAND[("cylindrical", 8)] == {"EDT": 0.253, "C50": 0.0059,
                                                             "T60": 0.014}
    label = summarize.band_label({"backbone": "cylindrical", "num_shot": 8})
    assert "CylindricalViT" in label and "exp_04" in label and "context only" in label
    assert "SimpleViT" in summarize.band_label({"backbone": "simple", "num_shot": 8})
    assert summarize.band_label({"backbone": "cylindrical", "num_shot": 1}) is None


def test_the_band_is_chosen_by_backbone_and_k_not_by_the_arm_label(tmp_path):
    # An arm may be called anything; what decides the band is the model that produced it.
    cyl = _fixture_run(tmp_path, "cyl", n=12,
                       meta_overrides={"arm": "released_k8", "backbone": "cylindrical"})
    simple = _fixture_run(tmp_path, "simple", n=12,
                          protocol_overrides={"checkpoint_sha256": "d" * 64},
                          meta_overrides={"arm": "cyl_k8", "backbone": "simple"})
    summary = summarize.build_summary([cyl, simple], n_boot=50)

    assert summary["arms"][0]["historical_band"] == summarize.HISTORICAL_BAND[
        ("cylindrical", 8)]
    assert "CylindricalViT" in summary["arms"][0]["historical_band_label"]
    assert summary["arms"][1]["historical_band"] == summarize.HISTORICAL_BAND[
        ("simple", 8)]
    assert "SimpleViT" in summary["arms"][1]["historical_band_label"]

    rows = summarize.figure_data(summary)
    cyl_edt = next(row for row in rows if row["arm"] == "released_k8"
                   and row["metric"] == "EDT")
    assert cyl_edt["band"] == 0.253
    assert "CylindricalViT" in cyl_edt["band_label"]


def test_the_markdown_names_each_arms_context_band(tmp_path):
    run_dir = _fixture_run(tmp_path, "cyl_md", n=12,
                           meta_overrides={"arm": "cyl_k8", "backbone": "cylindrical"})
    summary = summarize.build_summary([run_dir], n_boot=50)
    markdown = _markdown(summary, tmp_path, "band_md")
    assert summarize.band_label(summary["arms"][0]["meta"]) in markdown
