"""Tests for :mod:`tools.summarize_yaw` (exp_03, yaw_rotation_degradation).

Every test runs on synthetic ``per_sample_yaw.json`` files built here, whose relative
degradations are planted exactly: ``e_k = e_0 * (1 + shift)`` elementwise, so the
statistic the summarizer must recover is known in closed form and any bookkeeping error
(a wrong pairing, a dropped mask, a swapped condition) shows up as a wrong number rather
than as noise.
"""
import json
import math
import os

import numpy as np
import pytest

MANIFEST_HASH = "47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d"
ROOMS = [("Cafe", "Cafe_idx_1"), ("Office", "Office_idx_10"), ("Bedrooms", "Bedrooms_idx_18"),
         ("MeetingRoom", "MeetingRoom_idx_20"), ("Apartments", "Apartments_idx_42")]
METRICS = ("edt", "c50", "t60", "loss", "log_mse", "consistency")
COLS = (0, 32, 64)


def _queries(n):
    """Realistic AcousticRooms query paths spread over five rooms."""
    return ["{}/{}/S{:03d}_R{:03d}_hybrid_IR.wav".format(
        ROOMS[i % len(ROOMS)][0], ROOMS[i % len(ROOMS)][1], i // 7, i % 7) for i in range(n)]


def write_run(directory, shifts, n=300, seed=0, backbone="simple",
              checkpoint="ckpt/xRIR_simple_8_shot/epoch_12.pth", cols=COLS,
              nan_rows=(), baseline_nan_rows=(), manifest_hash=MANIFEST_HASH,
              noise=0.02, noise_seed=None, base_offset=0.0):
    """Write one synthetic ``per_sample_yaw.json`` with planted relative degradations.

    Args:
        shifts: ``{(condition, k): shift}`` or ``{(condition, k, metric): shift}``;
            ``e_k = e_0 * (1 + shift + eps)``, so the true ``r_k`` is ``shift`` up to the
            per-sample noise ``eps`` (which is what gives the bootstrap a real width).
        nan_rows: sample indices that the rotation invalidates (NaN at every ``k != 0``).
        baseline_nan_rows: sample indices invalid at every angle, ``k = 0`` included.
        seed: seeds the ``k = 0`` baseline, so two runs sharing it are paired exactly
            there; ``noise_seed`` (default ``seed + 1``) seeds the per-angle noise.
        base_offset: added to every value, so a second run can differ from the first by
            a known constant at ``k = 0``.
    """
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 1 if noise_seed is None else noise_seed)
    queries = _queries(n)
    base = {metric: rng.gamma(4.0, 0.25, size=n) + 0.05 + base_offset for metric in METRICS}
    run = {"query": queries, "index": list(range(n)),
           "delay_flips": {str(int(k)): (0 if int(k) == 0 else 7 * int(k)) for k in cols},
           "decomposition": ({"k": 32, "n_batches": 1, "tokens_rel_change": 3.2e-07,
                              "pooled_rel_change": 1.6e-02, "coord_rel_change": 0.68,
                              "logspec_rel_change": 2.0e-03}
                             if backbone == "cylindrical" else None),
           "meta": {"backbone": backbone, "checkpoint": checkpoint,
                    "manifest_path": "ckpt/yaw_rotation/reference_manifest.json",
                    "manifest_hash": manifest_hash, "manifest_seed": 0, "gl_seed": 0,
                    "yaw_cols": [int(k) for k in cols],
                    "acoustic_cols": [int(k) for k in cols],
                    "e_acoustic_cols": [int(k) for k in cols if int(k) != 0],
                    "batch_size": 16, "n_samples": n, "max_samples": 0, "tf32": False,
                    "torch_version": "2.0.1", "elapsed_min": 12.5}}
    for condition in ("P", "E"):
        run[condition] = {}
        for k in cols:
            k = int(k)
            cell = {}
            for metric in METRICS:
                if metric == "consistency":
                    values = np.zeros(n) if k == 0 else np.full(n, 0.1)
                else:
                    shift = shifts.get((condition, k, metric),
                                       shifts.get((condition, k), 0.0)) if k else 0.0
                    eps = noise_rng.normal(0.0, noise, size=n) if k else np.zeros(n)
                    values = base[metric] * (1.0 + shift + eps)
                    if k:
                        values[list(nan_rows)] = np.nan
                    values[list(baseline_nan_rows)] = np.nan
                cell[metric] = [float(v) for v in values]
            run[condition][str(k)] = cell
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "per_sample_yaw.json"), "w") as fout:
        # Strict JSON, exactly as eval_yaw_rotation.py writes it: invalid samples are null.
        json.dump(_nulled(run), fout, allow_nan=False)
    return directory


def _nulled(run):
    """Replace every non-finite per-sample value with ``None`` (strict-JSON encoding)."""
    for condition in ("P", "E"):
        for cell in run[condition].values():
            for metric, values in cell.items():
                cell[metric] = [None if not np.isfinite(v) else float(v) for v in values]
    return run


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    """A control run that degrades and a cylindrical run that (mostly) does not."""
    root = tmp_path_factory.mktemp("runs")
    control = write_run(str(root / "control"),
                        {("P", 32): 0.15, ("P", 64): 0.30, ("E", 32): 0.18, ("E", 64): 0.34},
                        nan_rows=(3, 17, 88), baseline_nan_rows=(5,), noise_seed=1)
    cyl = write_run(str(root / "cyl"),
                    {("P", 32): 0.0, ("P", 64): 0.05, ("E", 32): 0.01, ("E", 64): 0.06},
                    backbone="cylindrical", checkpoint="ckpt/xRIR_cyl_8_shot/epoch_12.pth",
                    nan_rows=(3, 17, 88), baseline_nan_rows=(5,), noise_seed=2)
    return control, cyl


# --------------------------------------------------------------------------------------
# load_run / rooms_from_paths
# --------------------------------------------------------------------------------------
def test_rooms_from_paths_takes_the_second_path_component():
    from tools.summarize_yaw import rooms_from_paths

    got = rooms_from_paths(["Cafe/Cafe_idx_1/S001_R002_hybrid_IR.wav",
                            "Office/Office_idx_10/S003_R000_hybrid_IR.wav"])
    assert list(got) == ["Cafe_idx_1", "Office_idx_10"]
    assert len(np.unique(rooms_from_paths(_queries(300)))) == 5


def test_load_run_reads_the_evaluator_output(two_runs):
    from tools.summarize_yaw import load_run

    control, _ = two_runs
    run = load_run(control)
    assert run["dir"] == control
    assert run["meta"]["manifest_hash"] == MANIFEST_HASH
    assert len(run["query"]) == 300 and sorted(run["P"]) == ["0", "32", "64"]
    assert len(run["P"]["32"]["edt"]) == 300

    with pytest.raises(IOError):
        load_run(os.path.join(control, "does_not_exist"))


# --------------------------------------------------------------------------------------
# degradation_rows
# --------------------------------------------------------------------------------------
def test_degradation_rows_recover_the_planted_shift_and_the_validity_split(two_runs):
    from tools.summarize_yaw import degradation_rows, load_run, rooms_from_paths

    control = load_run(two_runs[0])
    rooms = rooms_from_paths(control["query"])
    rows = degradation_rows(control, "P", "edt", [0, 32, 64], 2000, 0.05, 0, rooms=rooms)
    by_k = {row["k"]: row for row in rows}
    assert sorted(by_k) == [0, 32, 64]

    assert by_k[0]["r"] == pytest.approx(0.0, abs=1e-12)
    assert by_k[32]["r"] == pytest.approx(0.15, abs=0.01)
    assert by_k[64]["r"] == pytest.approx(0.30, abs=0.01)
    assert by_k[0]["deg"] == 0.0 and by_k[32]["deg"] == pytest.approx(22.5)
    assert by_k[64]["deg"] == pytest.approx(45.0)

    for k in (32, 64):
        row = by_k[k]
        assert row["lo"] < row["r"] < row["hi"]
        assert row["r_lo"] < row["r"] < row["r_hi"]          # room-cluster interval
        assert row["meank"] == pytest.approx(row["mean0"] * (1.0 + row["r"]), rel=1e-9)
        # 300 samples, one invalid baseline, three the rotation broke.
        assert row["n_valid"] == 296
        assert row["validity"]["baseline_invalid"] == 1
        assert row["validity"]["newly_invalid"] == 3
        assert row["validity"]["newly_invalid_frac"] == pytest.approx(3.0 / 299.0)
    assert by_k[0]["validity"]["newly_invalid"] == 0


def test_degradation_rows_map_the_angle_to_a_signed_degree(two_runs):
    from tools.summarize_yaw import degradation_rows, load_run

    control = load_run(two_runs[0])
    control["P"]["384"] = control["P"]["32"]
    control["P"]["256"] = control["P"]["64"]
    rows = {row["k"]: row["deg"] for row in
            degradation_rows(control, "P", "edt", [256, 384], 200, 0.05, 0)}
    assert rows[256] == pytest.approx(180.0)                 # 180 stays positive
    assert rows[384] == pytest.approx(-90.0)                 # 270 -> -90


def test_degradation_rows_report_an_undefined_ratio_instead_of_raising(two_runs):
    """``consistency`` is 0 at k = 0, so its relative degradation does not exist."""
    from tools.summarize_yaw import degradation_rows, load_run

    control = load_run(two_runs[0])
    rows = degradation_rows(control, "P", "consistency", [0, 32], 200, 0.05, 0)
    for row in rows:
        assert row["r"] is None and row["lo"] is None and row["hi"] is None
        assert row["mean0"] == 0.0
    assert rows[1]["meank"] == pytest.approx(0.1)


# --------------------------------------------------------------------------------------
# h1_verdict / h1_wording
# --------------------------------------------------------------------------------------
def _h1_rows(run_dir, threshold_metrics=("edt", "c50"), n_boot=2000, alpha=0.05):
    from tools.summarize_yaw import degradation_rows, load_run

    run = load_run(run_dir)
    return {metric: degradation_rows(run, "P", metric, [0, 32, 64], n_boot, alpha, 0)
            for metric in threshold_metrics}


def test_h1_verdict_needs_the_adjusted_lower_bound_to_clear_the_margin(two_runs):
    from tools.summarize_yaw import h1_verdict

    rows = _h1_rows(two_runs[0])                      # planted r = 0.15 (k=32), 0.30 (k=64)
    passes, cells = h1_verdict(rows, 0.10)
    assert passes is True
    assert {(cell["metric"], cell["k"]) for cell in cells} == {
        ("edt", 32), ("edt", 64), ("c50", 32), ("c50", 64)}
    assert all(cell["lo"] > 0.10 for cell in cells)
    assert all(cell["r"] == pytest.approx(0.15 if cell["k"] == 32 else 0.30, abs=0.01)
               for cell in cells)

    # A margin above the largest degradation: nothing clears it.
    assert h1_verdict(rows, 0.35) == (False, [])
    # Between the two planted shifts: only k = 64 clears it.
    passes, cells = h1_verdict(rows, 0.20)
    assert passes is True and {cell["k"] for cell in cells} == {64}


def test_h1_verdict_ignores_k0_and_undefined_rows(two_runs):
    from tools.summarize_yaw import degradation_rows, h1_verdict, load_run

    control = load_run(two_runs[0])
    rows = {"consistency": degradation_rows(control, "P", "consistency", [0, 32], 200, 0.05, 0),
            "edt": degradation_rows(control, "P", "edt", [0], 200, 0.05, 0)}
    assert h1_verdict(rows, -1.0) == (False, []), "k=0 or an undefined ratio cannot pass"


def test_h1_wording_covers_the_four_outcomes():
    from tools.summarize_yaw import h1_wording

    assert h1_wording(True, True) == "supported and replicated"
    assert h1_wording(True, False) == "supported for the same-budget model only"
    assert h1_wording(False, True) == "replication only"
    assert h1_wording(False, False) == "not supported"


# --------------------------------------------------------------------------------------
# h2_rows -- difference in differences and patch-aligned equivalence
# --------------------------------------------------------------------------------------
def test_h2_rows_have_the_right_sign_and_test_equivalence_where_it_is_claimed(two_runs):
    from tools.summarize_yaw import h2_rows, load_run, rooms_from_paths

    control, cyl = load_run(two_runs[0]), load_run(two_runs[1])
    rooms = rooms_from_paths(cyl["query"])
    rows = {row["k"]: row for row in
            h2_rows(cyl, control, "P", "edt", [0, 32, 64], 2000, 0.05, 0, rooms=rooms)}
    assert sorted(rows) == [0, 32, 64]

    # planted: r_cyl - r_ctrl = 0.00 - 0.15 at k=32 and 0.05 - 0.30 at k=64.
    assert rows[0]["d"] == pytest.approx(0.0, abs=1e-9)
    assert rows[32]["d"] == pytest.approx(-0.15, abs=0.01)
    assert rows[64]["d"] == pytest.approx(-0.25, abs=0.01)
    for k in (32, 64):
        assert rows[k]["hi"] < 0.0, "the cylindrical model degrades less: D_k < 0"
        assert rows[k]["lo"] < rows[k]["d"] < rows[k]["hi"]
        assert rows[k]["c_lo"] is not None and rows[k]["c_hi"] is not None
        assert rows[k]["r_cyl"] == pytest.approx(0.0 if k == 32 else 0.05, abs=0.01)
        assert rows[k]["r_ctrl"] == pytest.approx(0.15 if k == 32 else 0.30, abs=0.01)
        assert rows[k]["n_valid"] == 296

    # Equivalence is claimed only at the patch-aligned angles, and only for the
    # cylindrical model: it holds for a zero shift and fails for a 5 % one at +-2 %.
    assert rows[0]["equivalence"] is None
    assert rows[32]["equivalence"]["query"]["equivalent"] is True
    assert rows[64]["equivalence"]["query"]["equivalent"] is False
    assert rows[64]["equivalence"]["query"]["r"] == pytest.approx(0.05, abs=0.01)
    for k in (32, 64):
        assert rows[k]["equivalence"]["margin"] == 0.02
        assert "equivalent" in rows[k]["equivalence"]["cluster"]


def test_h2_rows_refuse_two_runs_that_are_not_query_aligned(two_runs, tmp_path):
    from tools.summarize_yaw import h2_rows, load_run

    control, cyl = load_run(two_runs[0]), load_run(two_runs[1])
    shuffled = dict(cyl)
    shuffled["query"] = list(reversed(cyl["query"]))
    with pytest.raises(ValueError):
        h2_rows(shuffled, control, "P", "edt", [32], 200, 0.05, 0)


def test_h2_rows_skip_equivalence_at_an_angle_that_is_not_patch_aligned(two_runs):
    from tools.summarize_yaw import h2_rows, load_run

    control, cyl = load_run(two_runs[0]), load_run(two_runs[1])
    cyl["P"]["48"] = cyl["P"]["32"]
    control["P"]["48"] = control["P"]["32"]
    row = h2_rows(cyl, control, "P", "edt", [48], 200, 0.05, 0)[0]
    assert row["equivalence"] is None and row["d"] == pytest.approx(-0.15, abs=0.01)


# --------------------------------------------------------------------------------------
# k0_comparison -- the properly paired re-evaluation of exp_01 at k = 0
# --------------------------------------------------------------------------------------
def test_k0_comparison_recovers_a_known_absolute_difference(tmp_path):
    from tools.summarize_yaw import k0_comparison, load_run, rooms_from_paths

    # Same baseline seed, so the two runs are paired exactly at k = 0 and differ there
    # by a planted constant.
    ctrl = load_run(write_run(str(tmp_path / "ctrl"), {}, seed=3, noise_seed=1))
    cyl = load_run(write_run(str(tmp_path / "cyl"), {}, seed=3, noise_seed=2,
                             backbone="cylindrical", base_offset=0.01))
    rooms = rooms_from_paths(ctrl["query"])
    rows = {row["metric"]: row for row in
            k0_comparison(cyl, ctrl, ["edt", "c50"], 500, 0.05, 0, rooms)}

    assert sorted(rows) == ["c50", "edt"]
    for metric, row in rows.items():
        assert row["diff"] == pytest.approx(0.01, abs=1e-9), metric
        assert row["cyl"] - row["ctrl"] == pytest.approx(0.01, abs=1e-9)
        assert row["n_valid"] == 300
        # Every resample sees the same constant difference, so both intervals collapse.
        assert row["q_lo"] == pytest.approx(0.01, abs=1e-9)
        assert row["q_hi"] == pytest.approx(0.01, abs=1e-9)
        assert row["r_lo"] == pytest.approx(0.01, abs=1e-9)
        assert row["r_hi"] == pytest.approx(0.01, abs=1e-9)
        assert row["rel_pct"] == pytest.approx(100.0 * 0.01 / row["ctrl"], rel=1e-9)


def test_k0_comparison_brackets_a_noisy_difference(tmp_path):
    from tools.summarize_yaw import k0_comparison, load_run, rooms_from_paths

    ctrl = load_run(write_run(str(tmp_path / "ctrl"), {}, seed=3))
    cyl = load_run(write_run(str(tmp_path / "cyl"), {}, seed=11, backbone="cylindrical"))
    rooms = rooms_from_paths(ctrl["query"])
    row = k0_comparison(cyl, ctrl, ["edt"], 2000, 0.05, 0, rooms)[0]

    assert row["q_lo"] < row["diff"] < row["q_hi"]
    assert row["r_lo"] < row["diff"] < row["r_hi"]
    assert row["q_hi"] - row["q_lo"] > 0.0 and row["r_hi"] - row["r_lo"] > 0.0
    assert row["diff"] == pytest.approx(row["cyl"] - row["ctrl"], rel=1e-12)


# --------------------------------------------------------------------------------------
# convergence_check -- is the bootstrap itself converged?
# --------------------------------------------------------------------------------------
def test_convergence_check_is_small_for_a_converged_bootstrap(two_runs):
    from tools.summarize_yaw import convergence_check, degradation_rows, load_run

    control = load_run(two_runs[0])

    def interval(seed):
        return degradation_rows(control, "P", "edt", [32], 4000, 0.05, seed)[0]

    ratio = convergence_check(interval, 0, 1)
    assert 0.0 <= ratio < 0.10, ratio


def test_convergence_check_reports_a_zero_width_interval_honestly():
    from tools.summarize_yaw import convergence_check

    assert convergence_check(lambda seed: {"lo": 1.0, "hi": 1.0}, 0, 1) == 0.0
    assert math.isinf(convergence_check(
        lambda seed: {"lo": 1.0 + seed, "hi": 1.0 + seed}, 0, 1))
    assert convergence_check(lambda seed: {"lo": 0.0, "hi": 1.0 + 0.05 * seed}, 0, 1) \
        == pytest.approx(0.05)


# --------------------------------------------------------------------------------------
# main -- the printed summary and the canonical JSON that binds to it
# --------------------------------------------------------------------------------------
def test_main_writes_a_summary_and_a_json_that_matches_it(two_runs, tmp_path, capsys):
    import hashlib

    from tools.summarize_yaw import main

    control, cyl = two_runs
    json_path = str(tmp_path / "summary.json")
    summary_path = str(tmp_path / "summary.txt")
    out = main(["--runs", control, cyl, "--labels", "control", "cyl",
                "--primary-simple", "control", "--cyl", "cyl",
                "--manifest-hash", MANIFEST_HASH, "--n-boot", "4000",
                "--json", json_path, "--summary", summary_path])

    assert os.path.exists(json_path) and os.path.exists(summary_path)
    written = json.load(open(json_path))
    text = open(summary_path).read()
    assert written["summary_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert written == out

    assert written["manifest_hash"] == MANIFEST_HASH
    assert written["n_queries"] == 300 and written["n_rooms"] == 5
    config = written["config"]
    assert config["n_boot"] == 4000 and config["alpha"] == 0.05
    assert config["threshold"] == 0.10 and config["equiv_margin"] == 0.02
    # 2 metrics x the non-zero acoustic angles of the run (k = 32, 64).
    assert config["family_size"] == 4
    assert config["alpha_adj"] == pytest.approx(0.05 / 4)
    assert config["labels"] == ["control", "cyl"]
    assert config["roles"] == {"primary": "control", "cyl": "cyl", "released": None}

    # H1: the control's planted degradation clears +10 %; there is no released run.
    assert written["h1"]["control"] == {"role": "primary", "passes": True,
                                        "cells": written["h1"]["control"]["cells"]}
    assert "cyl" not in written["h1"], "H1 is only asked of the primary and the released model"
    assert written["h1"]["verdict"] == "supported for the same-budget model only"
    assert {(c["metric"], c["k"]) for c in written["h1"]["control"]["cells"]} == {
        ("edt", 32), ("edt", 64), ("c50", 32), ("c50", 64)}

    # H2: the cylindrical model degrades less at both angles.
    for row in written["h2"]["edt"]:
        if row["k"]:
            assert row["d"] < 0 and row["hi"] < 0
    assert written["k0"][0]["diff"] == pytest.approx(0.0, abs=1e-9)
    assert written["delay_flips"]["control"]["32"] == 7 * 32
    assert written["decomposition"]["cyl"]["tokens_rel_change"] == pytest.approx(3.2e-07)
    assert 0.0 <= written["convergence"]["max_ratio"] < 0.10
    # The angles are printed in signed-degree order, so k = 0 sits in the middle of a
    # full grid and first in this one ({0, 32, 64} -> {0, 22.5, 45} degrees).
    assert [row["k"] for row in written["spectral"]["control"]["P"]["loss"]] == [0, 32, 64]

    # Everything printed is in the summary file.
    for needle in ("1. Spectral", "2. Acoustic", "3. Paired cylindrical", "4. H1",
                   "5. H2", "6. Delay-flip", "7. Decomposition", "8. Bootstrap"):
        assert needle in text, needle


def test_main_rejects_runs_with_different_manifests(two_runs, tmp_path):
    from tools.summarize_yaw import main

    other = write_run(str(tmp_path / "other"), {}, manifest_hash="0" * 64)
    with pytest.raises(ValueError):
        main(["--runs", two_runs[0], other, "--n-boot", "200"])
    with pytest.raises(ValueError):
        main(["--runs", two_runs[0], "--manifest-hash", "0" * 64, "--n-boot", "200"])


def test_main_skips_a_metric_the_run_did_not_evaluate_at_every_angle(two_runs, tmp_path):
    """The E condition is only measured acoustically at a few angles by design."""
    from tools.summarize_yaw import main

    partial = str(tmp_path / "partial")
    os.makedirs(partial, exist_ok=True)
    run = json.load(open(os.path.join(two_runs[0], "per_sample_yaw.json")))
    for angle in ("32", "64"):
        del run["E"][angle]["edt"]
    run["meta"]["e_acoustic_cols"] = [32, 64]
    with open(os.path.join(partial, "per_sample_yaw.json"), "w") as fout:
        json.dump(run, fout)

    out = main(["--runs", partial, "--labels", "partial", "--n-boot", "500"])
    assert "edt" not in out["acoustic"]["partial"]["E"]
    assert "c50" in out["acoustic"]["partial"]["E"]
    assert "edt" in out["acoustic"]["partial"]["P"]


def test_the_summarizer_reads_null_and_nan_encodings_identically(two_runs, tmp_path):
    """The evaluator writes strict JSON (null); an older file may carry NaN literals."""
    from tools.summarize_yaw import degradation_rows, load_run

    strict = load_run(two_runs[0])
    legacy_dir = str(tmp_path / "legacy")
    os.makedirs(legacy_dir, exist_ok=True)
    legacy = json.load(open(os.path.join(two_runs[0], "per_sample_yaw.json")))
    for condition in ("P", "E"):
        for cell in legacy[condition].values():
            for metric, values in cell.items():
                cell[metric] = [float("nan") if v is None else v for v in values]
    with open(os.path.join(legacy_dir, "per_sample_yaw.json"), "w") as fout:
        json.dump(legacy, fout)                      # allow_nan default: NaN literals

    assert None in strict["P"]["32"]["edt"], "the fixture must contain an invalid sample"
    rows_strict = degradation_rows(strict, "P", "edt", [0, 32], 500, 0.05, 0)
    rows_legacy = degradation_rows(load_run(legacy_dir), "P", "edt", [0, 32], 500, 0.05, 0)
    assert rows_strict == rows_legacy
    assert rows_strict[1]["validity"]["newly_invalid"] == 3
