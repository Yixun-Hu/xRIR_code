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
import shutil

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
              noise=0.02, noise_seed=None, base_offset=0.0, no_shift_rows=(),
              gl_seed=0, tf32=False, batch_size=16):
    """Write one synthetic ``per_sample_yaw.json`` with planted relative degradations.

    Args:
        shifts: ``{(condition, k): shift}`` or ``{(condition, k, metric): shift}``;
            ``e_k = e_0 * (1 + shift + eps)``, so the true ``r_k`` is ``shift`` up to the
            per-sample noise ``eps`` (which is what gives the bootstrap a real width).
        nan_rows: sample indices that the rotation invalidates (NaN at every ``k != 0``).
        no_shift_rows: sample indices that keep their baseline value at every ``k`` (used
            to make a run's own degradation differ from the one seen through another
            run's validity mask).
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
                    "manifest_hash": manifest_hash, "manifest_seed": 0,
                    "gl_seed": gl_seed,
                    "yaw_cols": [int(k) for k in cols],
                    "acoustic_cols": [int(k) for k in cols],
                    "e_acoustic_cols": [int(k) for k in cols if int(k) != 0],
                    "batch_size": batch_size, "batch_canonical": True, "n_samples": n,
                    "max_samples": 0, "tf32": tf32,
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
                    per_sample_shift = np.full(n, shift, dtype=np.float64)
                    per_sample_shift[list(no_shift_rows)] = 0.0
                    values = base[metric] * (1.0 + per_sample_shift + eps)
                    if metric in ("edt", "c50", "t60"):
                        # Only the acoustic metrics can be undefined; the spectral ones
                        # are plain arithmetic and are always finite, as in a real run.
                        if k:
                            values[list(nan_rows)] = np.nan
                        values[list(baseline_nan_rows)] = np.nan
                cell[metric] = [float(v) for v in values]
            run[condition][str(k)] = cell
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "per_sample_yaw.json"), "w") as fout:
        # Strict JSON, exactly as eval_yaw_rotation.py writes it: invalid samples are null.
        json.dump(_nulled(run), fout, allow_nan=False)
    write_provenance(directory, checkpoint, manifest_hash)
    write_metrics(directory, run)
    return directory


def write_metrics(directory, run):
    """``metrics_yaw.json`` as eval_yaw_rotation.py writes it: the per-angle summaries."""
    payload = {"meta": run["meta"], "delay_flips": run["delay_flips"],
               "decomposition": run["decomposition"]}
    for condition in ("P", "E"):
        payload[condition] = {}
        for angle, cell in run[condition].items():
            payload[condition][angle] = {}
            for metric, values in cell.items():
                finite = [v for v in values if v is not None and np.isfinite(v)]
                payload[condition][angle][metric] = {
                    "mean": float(np.mean(finite)) if finite else None,
                    "n_valid": len(finite), "n_nan": len(values) - len(finite)}
    with open(os.path.join(directory, "metrics_yaw.json"), "w") as fout:
        json.dump(payload, fout, allow_nan=False)
    return directory


def write_provenance(directory, checkpoint, manifest_hash, git_sha="0" * 40):
    """The sidecar the launcher is expected to drop next to every confirmatory run."""
    with open(os.path.join(directory, "provenance.json"), "w") as fout:
        json.dump({"git_sha": git_sha, "manifest_hash": manifest_hash,
                   "checkpoint_sha256": sha256_of(checkpoint)}, fout)
    return directory


def sha256_of(path):
    """sha256 of a file, computed independently of the implementation under test."""
    import hashlib

    if not os.path.exists(path):
        return "0" * 64
    if path not in _DIGESTS:
        digest = hashlib.sha256()
        with open(path, "rb") as fin:
            for block in iter(lambda: fin.read(1 << 20), b""):
                digest.update(block)
        _DIGESTS[path] = digest.hexdigest()
    return _DIGESTS[path]


_DIGESTS = {}


def _nulled(run):
    """Replace every non-finite per-sample value with ``None`` (strict-JSON encoding)."""
    for condition in ("P", "E"):
        for cell in run[condition].values():
            for metric, values in cell.items():
                cell[metric] = [None if not np.isfinite(v) else float(v) for v in values]
    return run


@pytest.fixture(scope="module")
def released_run(tmp_path_factory):
    """A third run standing in for the released checkpoint."""
    root = tmp_path_factory.mktemp("released")
    return write_run(str(root / "released"),
                     {("P", 32): 0.12, ("P", 64): 0.26, ("E", 32): 0.14, ("E", 64): 0.29},
                     checkpoint="checkpoints/xRIR_unseen.pth",
                     nan_rows=(3, 17, 88), baseline_nan_rows=(5,), noise_seed=3)


def relax_full_expectations(monkeypatch, n_queries=300, n_rooms=5, cols=(0, 32, 64),
                            min_n_boot=500):
    """Point full mode's size and grid expectations at the synthetic fixtures.

    Only the *expectations* move; every rule they feed stays exactly as production runs
    it, and production never passes anything but the pre-registered constants.
    """
    import tools.summarize_yaw as summarize_yaw

    monkeypatch.setattr(summarize_yaw, "FULL_EXPECTATIONS",
                        dict(summarize_yaw.FULL_EXPECTATIONS, n_queries=n_queries,
                             n_rooms=n_rooms, spectral_cols=tuple(cols),
                             acoustic_cols=tuple(cols),
                             e_acoustic_cols=tuple(k for k in cols if k),
                             min_n_boot=min_n_boot))


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
def test_main_writes_a_summary_and_a_json_that_matches_it(two_runs, released_run,
                                                          tmp_path, monkeypatch, capsys):
    import hashlib

    from tools.summarize_yaw import main

    control, cyl = two_runs
    relax_full_expectations(monkeypatch)
    json_path = str(tmp_path / "summary.json")
    summary_path = str(tmp_path / "summary.txt")
    out = main(["--mode", "full", "--runs", control, cyl, released_run,
                "--labels", "control", "cyl", "released",
                "--primary-simple", "control", "--cyl", "cyl", "--released", "released",
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
    # The family is pre-registered at 2 metrics x 9 non-zero acoustic angles and is
    # NEVER derived from the grid a particular run happened to use.
    assert config["family_size"] == 18
    assert config["alpha_adj"] == pytest.approx(0.05 / 18)
    assert config["preregistered_acoustic_cols"] == [0, 8, 32, 64, 128, 256, 384, 448,
                                                     480, 504]
    assert config["labels"] == ["control", "cyl", "released"]
    assert config["roles"] == {"primary": "control", "cyl": "cyl", "released": "released"}
    assert written["mode"] == "full" and written["valid_for_confirmatory"] is True
    assert written["validation_reasons"] == []

    # H1: the control's planted degradation clears +10 %; there is no released run.
    assert written["h1"]["control"] == {"role": "primary", "passes": True,
                                        "cells": written["h1"]["control"]["cells"]}
    assert "cyl" not in written["h1"], "H1 is only asked of the primary and the released model"
    assert written["h1"]["released"]["passes"] is True
    assert written["h1"]["verdict"] == "supported and replicated"
    # The margin is relative; its absolute size at k=0 is what a reader can judge.
    margins = written["h1"]["absolute_margins"]
    k0_edt = written["acoustic"]["control"]["P"]["edt"][0]["mean0"]
    assert margins["control"]["edt"] == pytest.approx(0.10 * k0_edt)
    assert set(margins) == {"control", "released"}
    assert {(c["metric"], c["k"]) for c in written["h1"]["control"]["cells"]} == {
        ("edt", 32), ("edt", 64), ("c50", 32), ("c50", 64)}

    # H2: the cylindrical model degrades less at both angles, so every H1-passing cell
    # passes and the aggregate verdict is "supported".
    for row in written["h2"]["rows"]["edt"]:
        if row["k"]:
            assert row["d"] < 0 and row["hi"] < 0
    assert written["h2"]["verdict"]["aggregate"] == "supported"
    assert written["h2"]["verdict"]["n_cells"] == 4
    assert sorted(written["h2"]["verdict"]["passing"]) == [
        "c50@32", "c50@64", "edt@32", "edt@64"]
    assert written["k0"][0]["diff"] == pytest.approx(0.0, abs=1e-9)
    assert written["delay_flips"]["control"]["32"] == 7 * 32
    assert written["decomposition"]["cyl"]["tokens_rel_change"] == pytest.approx(3.2e-07)
    convergence = written["convergence"]
    assert convergence["pass"] is True and 0.0 <= convergence["max_rel_change"] < 0.10
    assert convergence["seeds"] == [0, 1] and convergence["limit"] == 0.10
    # Every decision-driving bound: both H1 models, the difference in differences and
    # the equivalence bounds, each with both seeds' endpoints.
    names = set(convergence["cells"])
    assert {"h1/control/edt/32", "h1/released/c50/64", "h2/edt/32", "tost/c50/64"} <= names
    for cell in convergence["cells"].values():
        assert set(cell["seed_a"]) == {"lo", "hi"} and set(cell["seed_b"]) == {"lo", "hi"}
    # The angles are printed in signed-degree order, so k = 0 sits in the middle of a
    # full grid and first in this one ({0, 32, 64} -> {0, 22.5, 45} degrees).
    assert [row["k"] for row in written["spectral"]["control"]["P"]["loss"]] == [0, 32, 64]

    # Everything printed is in the summary file.
    for needle in ("1. Spectral", "2. Acoustic", "3. Paired cylindrical", "4. H1",
                   "5. H2", "H2 verdict: supported", "6. Delay-flip", "7. Decomposition",
                   "8. Bootstrap convergence", "bounds recomputed with seed 0 -> 1",
                   "+10% of the k=0 error is", "equivalent (query / room)",
                   "reference-coordinate features not decomposed"):
        assert needle in text, needle


def test_main_rejects_runs_with_different_manifests(two_runs, tmp_path):
    from tools.summarize_yaw import main

    other = write_run(str(tmp_path / "other"), {}, manifest_hash="0" * 64)
    with pytest.raises(ValueError) as excinfo:
        main(["--mode", "exploratory", "--runs", two_runs[0], other,
              "--labels", "control", "other", "--n-boot", "200"])
    assert "different manifests" in str(excinfo.value)
    with pytest.raises(ValueError):
        main(["--mode", "exploratory", "--runs", two_runs[0],
              "--manifest-hash", "0" * 64, "--n-boot", "200"])


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

    out = main(["--mode", "exploratory", "--runs", partial, "--labels", "partial",
                "--n-boot", "500"])
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


def test_h2_equivalence_uses_the_cylindrical_runs_own_valid_mask(tmp_path):
    """The equivalence claim is about the cylindrical model, not about the pair.

    Here the control is unscorable on all but 12 samples -- exactly the 12 where the
    cylindrical model happens not to degrade. Read through the four-way difference mask,
    the cylindrical model would look unchanged (r = 0, "equivalent"); its own mask shows
    the +5 % it actually has.
    """
    from tools.summarize_yaw import h2_rows, load_run, rooms_from_paths

    keep = tuple(range(0, 300, 25))
    invalid = tuple(i for i in range(300) if i not in keep)
    cyl = load_run(write_run(str(tmp_path / "cyl"), {("P", 32): 0.05}, no_shift_rows=keep,
                             backbone="cylindrical",
                             checkpoint="ckpt/xRIR_cyl_8_shot/epoch_12.pth", noise_seed=2))
    ctrl = load_run(write_run(str(tmp_path / "ctrl"), {("P", 32): 0.05}, nan_rows=invalid,
                              noise_seed=1))
    rooms = rooms_from_paths(cyl["query"])

    row = h2_rows(cyl, ctrl, "P", "edt", [32], 2000, 0.05, 0, rooms=rooms)[0]
    assert row["n_valid"] == len(keep), "the difference in differences keeps the pair mask"
    equivalence = row["equivalence"]["query"]
    assert equivalence["n"] == 300, "the TOST must use the cylindrical run's own mask"
    assert equivalence["r"] == pytest.approx(0.048, abs=0.005)
    assert equivalence["equivalent"] is False
    assert row["equivalence"]["cluster"]["n"] == 300
    assert row["equivalence"]["n"] == 300


# --------------------------------------------------------------------------------------
# h2_verdict -- the pre-registered cross-model rule
# --------------------------------------------------------------------------------------
def _h1_cell(metric, k):
    return {"metric": metric, "k": k, "deg": 22.5, "r": 0.15, "lo": 0.12, "hi": 0.18}


def _h2_row(k, d, hi, lo=None):
    if lo is None and d is not None:
        lo = d - 0.05
    return {"k": k, "deg": 22.5, "d": d, "lo": lo,
            "hi": hi, "c_lo": None, "c_hi": None, "r_cyl": 0.0, "r_ctrl": 0.15,
            "n_valid": 296, "equivalence": None}


def test_h2_verdict_is_supported_only_when_every_h1_cell_passes():
    from tools.summarize_yaw import h2_verdict

    h1 = [_h1_cell("edt", 32), _h1_cell("c50", 32)]
    rows = {"edt": [_h2_row(32, -0.15, -0.10)], "c50": [_h2_row(32, -0.12, -0.08)]}
    got = h2_verdict(h1, rows)
    assert got["aggregate"] == "supported"
    assert got["n_cells"] == 2 and got["n_passing"] == 2
    assert got["failing"] == [] and sorted(got["passing"]) == ["c50@32", "edt@32"]
    assert all(cell["passes"] for cell in got["cells"])


def test_h2_verdict_is_partial_when_only_some_cells_pass():
    from tools.summarize_yaw import h2_verdict

    h1 = [_h1_cell("edt", 32), _h1_cell("c50", 32)]
    rows = {"edt": [_h2_row(32, -0.15, -0.10)],
            "c50": [_h2_row(32, -0.02, +0.03)]}      # interval straddles zero
    got = h2_verdict(h1, rows)
    assert got["aggregate"] == "partially supported"
    assert got["passing"] == ["edt@32"] and got["failing"] == ["c50@32"]


def test_h2_verdict_is_not_supported_when_no_cell_passes():
    from tools.summarize_yaw import h2_verdict

    h1 = [_h1_cell("edt", 32), _h1_cell("c50", 64)]
    rows = {"edt": [_h2_row(32, +0.05, +0.09)],      # the wrong sign
            "c50": [_h2_row(64, -0.02, +0.03)]}      # right sign, bound above zero
    got = h2_verdict(h1, rows)
    assert got["aggregate"] == "not supported"
    assert got["n_passing"] == 0 and len(got["cells"]) == 2


def test_h2_verdict_is_not_evaluable_without_a_passing_h1_cell():
    from tools.summarize_yaw import h2_verdict

    got = h2_verdict([], {"edt": [_h2_row(32, -0.15, -0.10)]})
    assert got["aggregate"] == "not evaluable (H1 has no passing cell)"
    assert got["cells"] == [] and got["n_cells"] == 0


def test_h2_verdict_fails_a_cell_whose_difference_is_missing():
    from tools.summarize_yaw import h2_verdict

    h1 = [_h1_cell("edt", 32), _h1_cell("edt", 64)]
    rows = {"edt": [_h2_row(32, None, None), _h2_row(64, -0.2, -0.1)]}
    got = h2_verdict(h1, rows)
    assert got["aggregate"] == "partially supported"
    assert got["failing"] == ["edt@32"]


def test_the_confirmatory_family_is_pre_registered_at_eighteen_tests():
    from tools.summarize_yaw import (
        CONFIRMATORY_METRICS,
        FAMILY_SIZE,
        PREREGISTERED_ACOUSTIC_COLS,
        PREREGISTERED_E_ACOUSTIC_COLS,
        PREREGISTERED_SPECTRAL_COLS,
    )

    assert tuple(CONFIRMATORY_METRICS) == ("edt", "c50")
    assert PREREGISTERED_ACOUSTIC_COLS == (0, 8, 32, 64, 128, 256, 384, 448, 480, 504)
    assert PREREGISTERED_E_ACOUSTIC_COLS == (32, 128, 384, 480)
    assert PREREGISTERED_SPECTRAL_COLS == (0, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320,
                                           384, 416, 448, 480, 496, 504, 508)
    assert len(PREREGISTERED_SPECTRAL_COLS) == 18
    assert FAMILY_SIZE == 2 * 9 == 18


# --------------------------------------------------------------------------------------
# validate_full -- what the confirmatory analysis is allowed to run on
# --------------------------------------------------------------------------------------
def _roles(control, cyl, released):
    return {"primary": control, "cyl": cyl, "released": released}


def _by_label(two_runs, released_run):
    from tools.summarize_yaw import load_run

    return {"control": load_run(two_runs[0]), "cyl": load_run(two_runs[1]),
            "released": load_run(released_run)}


PRE_REGISTERED_SETTINGS = {"alpha": 0.05, "threshold": 0.10, "equiv_margin": 0.02,
                           "seed": 0, "n_boot": 20000}


def test_validate_full_accepts_a_complete_set_of_runs(two_runs, released_run, monkeypatch):
    from tools.summarize_yaw import validate_full

    relax_full_expectations(monkeypatch)
    reasons = validate_full(_by_label(two_runs, released_run),
                            _roles("control", "cyl", "released"), MANIFEST_HASH,
                            PRE_REGISTERED_SETTINGS)
    assert reasons == []


def test_validate_full_names_every_violation(two_runs, released_run, monkeypatch):
    import copy

    from tools.summarize_yaw import validate_full

    relax_full_expectations(monkeypatch)
    roles = _roles("control", "cyl", "released")

    def reasons_for(mutate=None, roles_override=None, digest=MANIFEST_HASH):
        runs = copy.deepcopy(_by_label(two_runs, released_run))
        if mutate is not None:
            mutate(runs)
        return validate_full(runs, roles_override or roles, digest,
                             PRE_REGISTERED_SETTINGS)

    assert any("manifest-hash" in r for r in reasons_for(digest=None))
    assert any("no released run" in r for r in
               reasons_for(roles_override=_roles("control", "cyl", None)))
    assert any("checkpoint" in r for r in reasons_for(
        lambda runs: runs["control"]["meta"].update(checkpoint="ckpt/other/epoch_12.pth")))
    assert any("queries" in r for r in reasons_for(
        lambda runs: runs["cyl"].update(query=runs["cyl"]["query"][:-1])))
    assert any("unique" in r for r in reasons_for(
        lambda runs: runs["cyl"]["query"].__setitem__(1, runs["cyl"]["query"][0])))
    assert any("rooms" in r for r in reasons_for(
        lambda runs: runs["cyl"].update(
            query=["Cafe/Cafe_idx_1/S{:03d}_R000_hybrid_IR.wav".format(i)
                   for i in range(300)])))
    assert any("gl_seed" in r for r in reasons_for(
        lambda runs: runs["control"]["meta"].update(gl_seed=7)))
    assert any("tf32" in r for r in reasons_for(
        lambda runs: runs["control"]["meta"].update(tf32=True)))
    assert any("batch_canonical" in r for r in reasons_for(
        lambda runs: runs["control"]["meta"].update(batch_canonical=False)))
    assert any("yaw_cols" in r for r in reasons_for(
        lambda runs: runs["control"]["meta"].update(yaw_cols=[0, 32])))
    assert any("acoustic_cols" in r for r in reasons_for(
        lambda runs: runs["control"]["meta"].update(acoustic_cols=[0, 32])))
    assert any("angle 64" in r for r in reasons_for(
        lambda runs: runs["control"]["P"].pop("64")))
    assert any("loss" in r for r in reasons_for(
        lambda runs: runs["control"]["P"]["32"].__setitem__("loss", [0.0] * 299)))
    assert any("non-finite" in r for r in reasons_for(
        lambda runs: runs["control"]["P"]["32"]["loss"].__setitem__(0, None)))
    assert any("edt" in r for r in reasons_for(
        lambda runs: runs["control"]["P"]["32"].pop("edt")))
    assert any("delay_flips" in r for r in reasons_for(
        lambda runs: runs["control"]["delay_flips"].pop("64")))
    assert any("index" in r for r in reasons_for(
        lambda runs: runs["cyl"].update(index=list(range(1, 301)))))


def test_main_in_full_mode_refuses_incomplete_runs_and_writes_nothing(two_runs, tmp_path):
    from tools.summarize_yaw import main

    json_path = str(tmp_path / "summary.json")
    summary_path = str(tmp_path / "summary.txt")
    with pytest.raises(SystemExit) as excinfo:
        main(["--mode", "full", "--runs", two_runs[0], two_runs[1],
              "--labels", "control", "cyl", "--manifest-hash", MANIFEST_HASH,
              "--n-boot", "200", "--json", json_path, "--summary", summary_path])
    assert excinfo.value.code != 0
    assert not os.path.exists(json_path) and not os.path.exists(summary_path)


def test_main_in_exploratory_mode_suppresses_the_verdicts(two_runs, tmp_path):
    from tools.summarize_yaw import main

    json_path = str(tmp_path / "summary.json")
    summary_path = str(tmp_path / "summary.txt")
    out = main(["--mode", "exploratory", "--runs", two_runs[0], two_runs[1],
                "--labels", "control", "cyl", "--n-boot", "500",
                "--json", json_path, "--summary", summary_path])
    assert out["exploratory"] is True and out["mode"] == "exploratory"
    assert "summary_sha256" not in out
    assert out["h1"]["verdict"] == "not evaluated (exploratory mode)"
    assert out["h2"]["verdict"]["aggregate"] == "not evaluated (exploratory mode)"
    written = json.load(open(json_path))
    assert written["exploratory"] is True and "summary_sha256" not in written
    assert "exploratory" in open(summary_path).read()


# --------------------------------------------------------------------------------------
# k0-gate: the parity check against exp_01 before any rotated angle is read
# --------------------------------------------------------------------------------------
GATE_METRICS = ("edt", "c50", "t60", "loss")


def _k0_run(directory, seed=0, scale=1.0, **kwargs):
    """A k = 0 only run, optionally with every value scaled by a known factor."""
    from tools.summarize_yaw import load_run

    run = load_run(write_run(directory, {}, cols=(0,), seed=seed, **kwargs))
    if scale != 1.0:
        for metric in METRICS:
            run["P"]["0"][metric] = [None if v is None else v * scale
                                     for v in run["P"]["0"][metric]]
            run["E"]["0"][metric] = list(run["P"]["0"][metric])
    return run


def _exp01_file(path, run, scale=1.0):
    """An exp_01 per-sample JSON (eval_xRIR_backbone.py's --save-per-sample layout)."""
    payload = {"meta": {"split": "unseen", "num_shot": 8}, "index": run["index"],
               "ir_path": run["query"], "stft_mse": run["P"]["0"]["log_mse"]}
    for metric in GATE_METRICS:
        payload[metric] = [None if v is None else v * scale for v in run["P"]["0"][metric]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fout:
        json.dump(payload, fout, allow_nan=False)
    return path


def test_k0_gate_rows_pass_inside_the_reference_draw_noise(tmp_path):
    from tools.summarize_yaw import k0_gate_rows

    control = _k0_run(str(tmp_path / "control"))
    cyl = _k0_run(str(tmp_path / "cyl"), backbone="cylindrical",
                  checkpoint="ckpt/xRIR_cyl_8_shot/epoch_12.pth")
    released = _k0_run(str(tmp_path / "released"),
                       checkpoint="checkpoints/xRIR_unseen.pth")
    by_label = {"control": control, "cyl": cyl, "released": released}
    roles = {"primary": "control", "cyl": "cyl", "released": "released"}
    exp01 = {"control": json.load(open(_exp01_file(str(tmp_path / "e/c.json"), control, 1.001))),
             "cyl": json.load(open(_exp01_file(str(tmp_path / "e/y.json"), cyl, 0.999)))}
    noise = [_k0_run(str(tmp_path / "n1"), scale=1.002),
             _k0_run(str(tmp_path / "n2"), scale=0.998)]
    baseline = {metric: _mean(released["P"]["0"][metric]) for metric in ("edt", "c50", "t60")}

    rows, gate_pass, reasons = k0_gate_rows(by_label, roles, exp01, noise, baseline,
                                            band_rule="v1")
    assert gate_pass is True and reasons == []
    by_cell = {(row["label"], row["metric"]): row for row in rows}
    assert set(row["label"] for row in rows) == {"control", "cyl", "released"}
    for metric in GATE_METRICS:
        row = by_cell[("control", metric)]
        assert row["spread"] == pytest.approx(0.002 * row["mean_k0"], rel=1e-6)
        assert row["band"] == pytest.approx(2 * row["spread"] + 1e-6)
        assert row["abs_diff"] == pytest.approx(0.001 * row["mean_k0"], rel=1e-6)
        assert row["pass"] is True and row["source"] == "exp_01 per-sample"
    # The released checkpoint is compared against exp_01's reported baseline instead,
    # and its loss has no published counterpart.
    assert by_cell[("released", "edt")]["source"] == "exp_01 baseline reproduction"
    assert by_cell[("released", "loss")]["reference"] is None
    assert by_cell[("released", "loss")]["pass"] is None


def _mean(values):
    return float(np.mean([v for v in values if v is not None]))


def test_k0_gate_rows_fail_a_model_outside_the_band(tmp_path):
    from tools.summarize_yaw import k0_gate_rows

    control = _k0_run(str(tmp_path / "control"))
    cyl = _k0_run(str(tmp_path / "cyl"), backbone="cylindrical",
                  checkpoint="ckpt/xRIR_cyl_8_shot/epoch_12.pth")
    exp01 = {"control": json.load(open(_exp01_file(str(tmp_path / "e/c.json"), control, 1.01))),
             "cyl": json.load(open(_exp01_file(str(tmp_path / "e/y.json"), cyl, 1.0)))}
    noise = [_k0_run(str(tmp_path / "n1"), scale=1.002)]
    rows, gate_pass, reasons = k0_gate_rows(
        {"control": control, "cyl": cyl},
        {"primary": "control", "cyl": "cyl", "released": None}, exp01, noise, None,
        band_rule="v1")

    assert gate_pass is False
    assert len(reasons) == len(GATE_METRICS)
    assert all("control" in reason for reason in reasons)
    failing = [row for row in rows if row["pass"] is False]
    assert {row["metric"] for row in failing} == set(GATE_METRICS)


def test_k0_gate_rows_report_missing_inputs_as_reasons(tmp_path):
    from tools.summarize_yaw import k0_gate_rows

    control = _k0_run(str(tmp_path / "control"))
    rows, gate_pass, reasons = k0_gate_rows(
        {"control": control}, {"primary": "control", "cyl": None, "released": None},
        {}, [], None, band_rule="v1")
    assert gate_pass is False
    assert any("no --noise-runs" in reason for reason in reasons)
    assert any("no exp_01 per-sample file" in reason for reason in reasons)


def test_main_in_k0_gate_mode_prints_the_gate_and_skips_the_hypotheses(tmp_path,
                                                                       monkeypatch):
    argv, _ = _gate_setup(tmp_path, monkeypatch)
    out, code = _run_gate(argv, tmp_path, "printed")

    assert code == 0 and out["mode"] == "k0-gate"
    assert out["gate_pass"] is True and out["gate_reasons"] == []
    assert len(out["gate_rows"]) == 3 * len(GATE_METRICS)
    assert out["h1"]["verdict"] == "not evaluated (k0-gate mode)"
    assert out["h2"]["verdict"]["aggregate"] == "not evaluated (k0-gate mode)"
    assert out["k0"] and out["k0"][0]["metric"] == "edt"
    text = open(str(tmp_path / "printed.txt")).read()
    assert "k=0 parity gate" in text and "3. Paired cylindrical" in text
    assert "gate_pass: True" in text


# --------------------------------------------------------------------------------------
# validate_full -- what the confirmatory analysis is allowed to run on
# --------------------------------------------------------------------------------------
def _roles(control, cyl, released):
    return {"primary": control, "cyl": cyl, "released": released}


def _by_label(two_runs, released_run):
    from tools.summarize_yaw import load_run

    return {"control": load_run(two_runs[0]), "cyl": load_run(two_runs[1]),
            "released": load_run(released_run)}


PRE_REGISTERED_SETTINGS = {"alpha": 0.05, "threshold": 0.10, "equiv_margin": 0.02,
                           "seed": 0, "n_boot": 20000}


def test_validate_full_accepts_a_complete_set_of_runs(two_runs, released_run, monkeypatch):
    from tools.summarize_yaw import validate_full

    relax_full_expectations(monkeypatch)
    reasons = validate_full(_by_label(two_runs, released_run),
                            _roles("control", "cyl", "released"), MANIFEST_HASH,
                            PRE_REGISTERED_SETTINGS)
    assert reasons == []


def test_validate_full_names_every_violation(two_runs, released_run, monkeypatch):
    import copy

    from tools.summarize_yaw import validate_full

    relax_full_expectations(monkeypatch)
    roles = _roles("control", "cyl", "released")

    def reasons_for(mutate=None, roles_override=None, digest=MANIFEST_HASH):
        runs = copy.deepcopy(_by_label(two_runs, released_run))
        if mutate is not None:
            mutate(runs)
        return validate_full(runs, roles_override or roles, digest,
                             PRE_REGISTERED_SETTINGS)

    assert any("manifest-hash" in r for r in reasons_for(digest=None))
    assert any("no released run" in r for r in
               reasons_for(roles_override=_roles("control", "cyl", None)))
    assert any("checkpoint" in r for r in reasons_for(
        lambda runs: runs["control"]["meta"].update(checkpoint="ckpt/other/epoch_12.pth")))
    assert any("queries" in r for r in reasons_for(
        lambda runs: runs["cyl"].update(query=runs["cyl"]["query"][:-1])))
    assert any("unique" in r for r in reasons_for(
        lambda runs: runs["cyl"]["query"].__setitem__(1, runs["cyl"]["query"][0])))
    assert any("rooms" in r for r in reasons_for(
        lambda runs: runs["cyl"].update(
            query=["Cafe/Cafe_idx_1/S{:03d}_R000_hybrid_IR.wav".format(i)
                   for i in range(300)])))
    assert any("gl_seed" in r for r in reasons_for(
        lambda runs: runs["control"]["meta"].update(gl_seed=7)))
    assert any("tf32" in r for r in reasons_for(
        lambda runs: runs["control"]["meta"].update(tf32=True)))
    assert any("batch_canonical" in r for r in reasons_for(
        lambda runs: runs["control"]["meta"].update(batch_canonical=False)))
    assert any("yaw_cols" in r for r in reasons_for(
        lambda runs: runs["control"]["meta"].update(yaw_cols=[0, 32])))
    assert any("acoustic_cols" in r for r in reasons_for(
        lambda runs: runs["control"]["meta"].update(acoustic_cols=[0, 32])))
    assert any("angle 64" in r for r in reasons_for(
        lambda runs: runs["control"]["P"].pop("64")))
    assert any("loss" in r for r in reasons_for(
        lambda runs: runs["control"]["P"]["32"].__setitem__("loss", [0.0] * 299)))
    assert any("non-finite" in r for r in reasons_for(
        lambda runs: runs["control"]["P"]["32"]["loss"].__setitem__(0, None)))
    assert any("edt" in r for r in reasons_for(
        lambda runs: runs["control"]["P"]["32"].pop("edt")))
    assert any("delay_flips" in r for r in reasons_for(
        lambda runs: runs["control"]["delay_flips"].pop("64")))
    assert any("index" in r for r in reasons_for(
        lambda runs: runs["cyl"].update(index=list(range(1, 301)))))


def test_main_in_full_mode_refuses_incomplete_runs_and_writes_nothing(two_runs, tmp_path):
    from tools.summarize_yaw import main

    json_path = str(tmp_path / "summary.json")
    summary_path = str(tmp_path / "summary.txt")
    with pytest.raises(SystemExit) as excinfo:
        main(["--mode", "full", "--runs", two_runs[0], two_runs[1],
              "--labels", "control", "cyl", "--manifest-hash", MANIFEST_HASH,
              "--n-boot", "200", "--json", json_path, "--summary", summary_path])
    assert excinfo.value.code != 0
    assert not os.path.exists(json_path) and not os.path.exists(summary_path)


def test_main_in_exploratory_mode_suppresses_the_verdicts(two_runs, tmp_path):
    from tools.summarize_yaw import main

    json_path = str(tmp_path / "summary.json")
    summary_path = str(tmp_path / "summary.txt")
    out = main(["--mode", "exploratory", "--runs", two_runs[0], two_runs[1],
                "--labels", "control", "cyl", "--n-boot", "500",
                "--json", json_path, "--summary", summary_path])
    assert out["exploratory"] is True and out["mode"] == "exploratory"
    assert "summary_sha256" not in out
    assert out["h1"]["verdict"] == "not evaluated (exploratory mode)"
    assert out["h2"]["verdict"]["aggregate"] == "not evaluated (exploratory mode)"
    written = json.load(open(json_path))
    assert written["exploratory"] is True and "summary_sha256" not in written
    assert "exploratory" in open(summary_path).read()


# --------------------------------------------------------------------------------------
# k0-gate: the parity check against exp_01 before any rotated angle is read
# --------------------------------------------------------------------------------------
GATE_METRICS = ("edt", "c50", "t60", "loss")


def _k0_run(directory, seed=0, scale=1.0, **kwargs):
    """A k = 0 only run, optionally with every value scaled by a known factor."""
    from tools.summarize_yaw import load_run

    run = load_run(write_run(directory, {}, cols=(0,), seed=seed, **kwargs))
    if scale != 1.0:
        for metric in METRICS:
            run["P"]["0"][metric] = [None if v is None else v * scale
                                     for v in run["P"]["0"][metric]]
            run["E"]["0"][metric] = list(run["P"]["0"][metric])
    return run


def _exp01_file(path, run, scale=1.0):
    """An exp_01 per-sample JSON (eval_xRIR_backbone.py's --save-per-sample layout)."""
    payload = {"meta": {"split": "unseen", "num_shot": 8}, "index": run["index"],
               "ir_path": run["query"], "stft_mse": run["P"]["0"]["log_mse"]}
    for metric in GATE_METRICS:
        payload[metric] = [None if v is None else v * scale for v in run["P"]["0"][metric]]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fout:
        json.dump(payload, fout, allow_nan=False)
    return path


def test_k0_gate_rows_pass_inside_the_reference_draw_noise(tmp_path):
    from tools.summarize_yaw import k0_gate_rows

    control = _k0_run(str(tmp_path / "control"))
    cyl = _k0_run(str(tmp_path / "cyl"), backbone="cylindrical",
                  checkpoint="ckpt/xRIR_cyl_8_shot/epoch_12.pth")
    released = _k0_run(str(tmp_path / "released"),
                       checkpoint="checkpoints/xRIR_unseen.pth")
    by_label = {"control": control, "cyl": cyl, "released": released}
    roles = {"primary": "control", "cyl": "cyl", "released": "released"}
    exp01 = {"control": json.load(open(_exp01_file(str(tmp_path / "e/c.json"), control, 1.001))),
             "cyl": json.load(open(_exp01_file(str(tmp_path / "e/y.json"), cyl, 0.999)))}
    noise = [_k0_run(str(tmp_path / "n1"), scale=1.002),
             _k0_run(str(tmp_path / "n2"), scale=0.998)]
    baseline = {metric: _mean(released["P"]["0"][metric]) for metric in ("edt", "c50", "t60")}

    rows, gate_pass, reasons = k0_gate_rows(by_label, roles, exp01, noise, baseline,
                                            band_rule="v1")
    assert gate_pass is True and reasons == []
    by_cell = {(row["label"], row["metric"]): row for row in rows}
    assert set(row["label"] for row in rows) == {"control", "cyl", "released"}
    for metric in GATE_METRICS:
        row = by_cell[("control", metric)]
        assert row["spread"] == pytest.approx(0.002 * row["mean_k0"], rel=1e-6)
        assert row["band"] == pytest.approx(2 * row["spread"] + 1e-6)
        assert row["abs_diff"] == pytest.approx(0.001 * row["mean_k0"], rel=1e-6)
        assert row["pass"] is True and row["source"] == "exp_01 per-sample"
    # The released checkpoint is compared against exp_01's reported baseline instead,
    # and its loss has no published counterpart.
    assert by_cell[("released", "edt")]["source"] == "exp_01 baseline reproduction"
    assert by_cell[("released", "loss")]["reference"] is None
    assert by_cell[("released", "loss")]["pass"] is None


def _mean(values):
    return float(np.mean([v for v in values if v is not None]))


def test_k0_gate_rows_fail_a_model_outside_the_band(tmp_path):
    from tools.summarize_yaw import k0_gate_rows

    control = _k0_run(str(tmp_path / "control"))
    cyl = _k0_run(str(tmp_path / "cyl"), backbone="cylindrical",
                  checkpoint="ckpt/xRIR_cyl_8_shot/epoch_12.pth")
    exp01 = {"control": json.load(open(_exp01_file(str(tmp_path / "e/c.json"), control, 1.01))),
             "cyl": json.load(open(_exp01_file(str(tmp_path / "e/y.json"), cyl, 1.0)))}
    noise = [_k0_run(str(tmp_path / "n1"), scale=1.002)]
    rows, gate_pass, reasons = k0_gate_rows(
        {"control": control, "cyl": cyl},
        {"primary": "control", "cyl": "cyl", "released": None}, exp01, noise, None,
        band_rule="v1")

    assert gate_pass is False
    assert len(reasons) == len(GATE_METRICS)
    assert all("control" in reason for reason in reasons)
    failing = [row for row in rows if row["pass"] is False]
    assert {row["metric"] for row in failing} == set(GATE_METRICS)


def test_k0_gate_rows_report_missing_inputs_as_reasons(tmp_path):
    from tools.summarize_yaw import k0_gate_rows

    control = _k0_run(str(tmp_path / "control"))
    rows, gate_pass, reasons = k0_gate_rows(
        {"control": control}, {"primary": "control", "cyl": None, "released": None},
        {}, [], None, band_rule="v1")
    assert gate_pass is False
    assert any("no --noise-runs" in reason for reason in reasons)
    assert any("no exp_01 per-sample file" in reason for reason in reasons)


def test_main_in_full_mode_writes_nothing_when_the_bootstrap_has_not_converged(
        two_runs, released_run, tmp_path, monkeypatch):
    """A pre-registered threshold read off an unconverged bound is not a decision rule."""
    from tools.summarize_yaw import main

    relax_full_expectations(monkeypatch, min_n_boot=30)
    json_path = str(tmp_path / "summary.json")
    summary_path = str(tmp_path / "summary.txt")
    argv = ["--mode", "full", "--runs", two_runs[0], two_runs[1], released_run,
            "--labels", "control", "cyl", "released",
            "--manifest-hash", MANIFEST_HASH, "--n-boot", "30",
            "--json", json_path, "--summary", summary_path]
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code != 0
    assert not os.path.exists(json_path) and not os.path.exists(summary_path)
    assert not os.path.exists(json_path + ".tmp")


def test_main_requires_the_json_and_the_summary_together(two_runs, tmp_path):
    from tools.summarize_yaw import main

    base = ["--mode", "exploratory", "--runs", two_runs[0], "--n-boot", "200"]
    path = str(tmp_path / "only.json")
    with pytest.raises(ValueError):
        main(base + ["--json", path])
    with pytest.raises(ValueError):
        main(base + ["--summary", str(tmp_path / "only.txt")])
    with pytest.raises(ValueError):
        main(base + ["--json", path, "--summary", path])


# --------------------------------------------------------------------------------------
# derive_labels -- the documented command line names its runs by directory
# --------------------------------------------------------------------------------------
def test_labels_are_derived_from_the_checkpoints(tmp_path):
    from tools.summarize_yaw import derive_labels, load_run

    directories = [str(tmp_path / "gate_control"), str(tmp_path / "gate_cyl"),
                   str(tmp_path / "gate_released")]
    runs = [load_run(write_run(directories[0], {}, cols=(0,))),
            load_run(write_run(directories[1], {}, cols=(0,), backbone="cylindrical",
                               checkpoint="ckpt/xRIR_cyl_8_shot/epoch_12.pth")),
            load_run(write_run(directories[2], {}, cols=(0,),
                               checkpoint="checkpoints/xRIR_unseen.pth"))]
    assert derive_labels(runs, directories) == ["control", "cyl", "released"]

    # An absolute path to the same checkpoint is the same run.
    runs[0]["meta"]["checkpoint"] = "/home/x/repo/ckpt/xRIR_simple_8_shot/epoch_12.pth"
    assert derive_labels(runs, directories)[0] == "control"

    # An unrecognised checkpoint keeps the directory name.
    other = str(tmp_path / "something_else")
    extra = load_run(write_run(other, {}, cols=(0,), checkpoint="ckpt/other/epoch_09.pth"))
    assert derive_labels([extra], [other]) == ["something_else"]

    # Two runs of the same checkpoint cannot be told apart: say so rather than guess.
    with pytest.raises(ValueError):
        derive_labels([runs[0], runs[0]], directories[:2])


# --------------------------------------------------------------------------------------
# validate_gate -- the k=0 gate fails closed
# --------------------------------------------------------------------------------------
SEED1_HASH = "6ca8164e5727d5f4db84569d5effa840b2cb6e6f77819a69bbb37bb30c6ab11e"
SEED2_HASH = "757e5a966ee2d069a07accecbc43e46bd564ea7c788a28c219773df64e986234"


def _scaled_run(directory, scale=1.0, **kwargs):
    """A k=0-only run whose values are the shared baseline times ``scale``."""
    write_run(directory, {}, cols=(0,), **kwargs)
    if scale != 1.0:
        _edit_run(directory, lambda run: [
            cell.__setitem__(metric, [None if v is None else v * scale for v in values])
            for condition in ("P", "E") for cell in [run[condition]["0"]]
            for metric, values in list(cell.items())])
    return directory


def _edit_run(directory, mutate):
    """Load, mutate and rewrite one run's per_sample_yaw.json in place."""
    path = os.path.join(directory, "per_sample_yaw.json")
    run = json.load(open(path))
    mutate(run)
    with open(path, "w") as fout:
        json.dump(run, fout, allow_nan=False)
    return directory


def _gate_setup(tmp_path, monkeypatch):
    """The three gate runs, two noise runs and the two exp_01 files, all consistent."""
    relax_full_expectations(monkeypatch)
    control = _scaled_run(str(tmp_path / "gate_control"))
    cyl = _scaled_run(str(tmp_path / "gate_cyl"), backbone="cylindrical",
                      checkpoint="ckpt/xRIR_cyl_8_shot/epoch_12.pth")
    released = _scaled_run(str(tmp_path / "gate_released"),
                           checkpoint="checkpoints/xRIR_unseen.pth")
    noise1 = _scaled_run(str(tmp_path / "noise_seed1"), scale=1.002,
                         manifest_hash=SEED1_HASH)
    noise2 = _scaled_run(str(tmp_path / "noise_seed2"), scale=0.998,
                         manifest_hash=SEED2_HASH)
    phase = _scaled_run(str(tmp_path / "gate_phase_seed1"), scale=1.001, gl_seed=1)
    tf32 = _scaled_run(str(tmp_path / "gate_tf32"), scale=0.9995, tf32=True)
    cyl_kwargs = {"backbone": "cylindrical",
                  "checkpoint": "ckpt/xRIR_cyl_8_shot/epoch_12.pth"}
    noise_cyl1 = _scaled_run(str(tmp_path / "gate_noise_cyl_seed1"), scale=1.003,
                             manifest_hash=SEED1_HASH, **cyl_kwargs)
    noise_cyl2 = _scaled_run(str(tmp_path / "gate_noise_cyl_seed2"), scale=0.997,
                             manifest_hash=SEED2_HASH, **cyl_kwargs)
    shape_control = _scaled_run(str(tmp_path / "gate_shape_b1"), scale=1.0005,
                                batch_size=1)
    shape_cyl = _scaled_run(str(tmp_path / "gate_shape_cyl_b1"), scale=0.999,
                            batch_size=1, **cyl_kwargs)
    loaded = {name: json.load(open(os.path.join(d, "per_sample_yaw.json")))
              for name, d in (("control", control), ("cyl", cyl), ("released", released))}
    exp01 = {name: _exp01_file(str(tmp_path / "exp01" / "{}.json".format(name)),
                               loaded[name], 1.001) for name in ("control", "cyl")}
    baseline = ",".join("{:.10g}".format(_mean(loaded["released"]["P"]["0"][metric]))
                        for metric in ("edt", "c50", "t60"))
    argv = ["--mode", "k0-gate", "--runs", control, cyl, released,
            "--manifest-hash", MANIFEST_HASH,
            "--exp01-per-sample", "control=" + exp01["control"], "cyl=" + exp01["cyl"],
            "--noise-runs", noise1, noise2,
            "--noise-runs-cyl", noise_cyl1, noise_cyl2,
            "--nuisance-runs", phase, tf32,
            "--shape-runs", shape_control, shape_cyl,
            "--released-baseline", baseline, "--n-boot", "200"]
    return argv, {"control": control, "cyl": cyl, "released": released,
                  "noise1": noise1, "noise2": noise2, "phase": phase, "tf32": tf32,
                  "noise_cyl1": noise_cyl1, "noise_cyl2": noise_cyl2,
                  "shape_control": shape_control, "shape_cyl": shape_cyl}


def _run_gate(argv, tmp_path, name):
    """Run the documented gate command line; return (out or None, exit code)."""
    from tools.summarize_yaw import main

    full = argv + ["--json", str(tmp_path / (name + ".json")),
                   "--summary", str(tmp_path / (name + ".txt"))]
    try:
        return main(full), 0
    except SystemExit as exit_signal:
        return json.load(open(str(tmp_path / (name + ".json")))), exit_signal.code


def test_the_documented_gate_command_passes_when_everything_is_present(tmp_path,
                                                                       monkeypatch):
    argv, _ = _gate_setup(tmp_path, monkeypatch)
    out, code = _run_gate(argv, tmp_path, "ok")
    assert code == 0
    assert out["gate_pass"] is True and out["gate_reasons"] == []
    # The labels came from the checkpoints, not from the directory names.
    assert out["config"]["labels"] == ["control", "cyl", "released"]
    assert out["config"]["roles"] == {"primary": "control", "cyl": "cyl",
                                      "released": "released"}
    assert len(out["gate_rows"]) == 3 * len(GATE_METRICS)


def test_the_gate_fails_closed_on_every_missing_or_wrong_input(tmp_path, monkeypatch):
    argv, paths = _gate_setup(tmp_path, monkeypatch)

    def fails(mutate_argv, name, needle):
        changed = mutate_argv(list(argv))
        out, code = _run_gate(changed, tmp_path, name)
        assert code != 0, name
        assert out["gate_pass"] is False, name
        assert any(needle in reason for reason in out["gate_reasons"]), \
            (name, needle, out["gate_reasons"])

    def drop_a_noise_run(a):
        return a[:a.index("--noise-runs") + 2] + a[a.index("--noise-runs-cyl"):]

    def drop_the_released_run(a):
        del a[a.index("--runs") + 3]
        return a

    def drop_an_exp01_file(a):
        del a[a.index("--exp01-per-sample") + 2]
        return a

    fails(drop_a_noise_run, "one_noise", "exactly 2")
    fails(drop_the_released_run, "no_released", "no released run")
    fails(drop_an_exp01_file, "no_exp01", "exp_01 per-sample")

    # A noise run of the wrong checkpoint, and one on the wrong manifest seed.
    _edit_run(paths["noise1"],
              lambda run: run["meta"].update(checkpoint="ckpt/xRIR_cyl_8_shot/epoch_12.pth"))
    fails(lambda a: a, "wrong_ckpt", "checkpoint")
    _edit_run(paths["noise1"],
              lambda run: run["meta"].update(checkpoint="ckpt/xRIR_simple_8_shot/epoch_12.pth",
                                             manifest_hash="0" * 64))
    fails(lambda a: a, "wrong_hash", "manifest")
    _edit_run(paths["noise1"], lambda run: run["meta"].update(manifest_hash=SEED1_HASH))

    # P and E must be the same forward at k = 0.
    _edit_run(paths["cyl"], lambda run: run["E"]["0"].__setitem__(
        "edt", [v if v is None else v + 1.0 for v in run["E"]["0"]["edt"]]))
    fails(lambda a: a, "p_ne_e", "P and E differ at k=0")


# --------------------------------------------------------------------------------------
# full mode pins the pre-registration itself, not just the shape of the runs
# --------------------------------------------------------------------------------------
def _full_trio(tmp_path, prefix=""):
    """Three complete, mutually consistent runs for the confirmatory summary."""
    shifts = {("P", 32): 0.15, ("P", 64): 0.30, ("E", 32): 0.18, ("E", 64): 0.34}
    control = write_run(str(tmp_path / (prefix + "control")), shifts, noise_seed=1)
    cyl = write_run(str(tmp_path / (prefix + "cyl")), {("P", 32): 0.0, ("P", 64): 0.05},
                    backbone="cylindrical",
                    checkpoint="ckpt/xRIR_cyl_8_shot/epoch_12.pth", noise_seed=2)
    released = write_run(str(tmp_path / (prefix + "released")), shifts,
                         checkpoint="checkpoints/xRIR_unseen.pth", noise_seed=3)
    return [control, cyl, released]


def test_full_mode_lists_every_pre_registration_violation(tmp_path, monkeypatch, capsys):
    from tools.summarize_yaw import main

    relax_full_expectations(monkeypatch, min_n_boot=20000)

    def check(name, needle, mutate=None, argv_extra=(), digest=MANIFEST_HASH):
        runs = _full_trio(tmp_path / name)
        if mutate is not None:
            for directory in runs:
                _edit_run(directory, mutate)
        argv = ["--mode", "full", "--runs"] + runs + [
            "--manifest-hash", digest, "--n-boot", "20000",
            "--json", str(tmp_path / (name + ".json")),
            "--summary", str(tmp_path / (name + ".txt"))] + list(argv_extra)
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code != 0, name
        printed = capsys.readouterr().out
        assert needle in printed, (name, needle, printed[-2000:])
        assert not os.path.exists(str(tmp_path / (name + ".json"))), name

    check("hash", "not the pinned seed-0 manifest", digest="0" * 64,
          mutate=lambda run: run["meta"].update(manifest_hash="0" * 64))
    check("seed", "meta.manifest_seed", mutate=lambda run: run["meta"].update(manifest_seed=9))
    check("batch", "meta.batch_size", mutate=lambda run: run["meta"].update(batch_size=8))
    check("epoch11", "epoch_12.pth",
          mutate=lambda run: run["meta"].update(
              checkpoint=run["meta"]["checkpoint"].replace("epoch_12", "epoch_11")))
    check("alpha", "--alpha", argv_extra=["--alpha", "0.1"])
    check("nboot", "--n-boot", argv_extra=["--n-boot", "5000"])
    check("threshold", "--threshold", argv_extra=["--threshold", "0.2"])
    check("margin", "--equiv-margin", argv_extra=["--equiv-margin", "0.05"])
    check("bootseed", "--seed", argv_extra=["--seed", "3"])


def test_full_mode_requires_one_delay_audit_across_the_three_runs(tmp_path, monkeypatch,
                                                                 capsys):
    from tools.summarize_yaw import main

    relax_full_expectations(monkeypatch)
    runs = _full_trio(tmp_path / "audit")
    _edit_run(runs[1], lambda run: run["delay_flips"].update({"32": 999}))
    argv = ["--mode", "full", "--runs"] + runs + [
        "--manifest-hash", MANIFEST_HASH, "--n-boot", "500",
        "--json", str(tmp_path / "a.json"), "--summary", str(tmp_path / "a.txt")]
    with pytest.raises(SystemExit):
        main(argv)
    printed = capsys.readouterr().out
    assert "delay_flips differ" in printed

    # The audit is geometry only, so a non-zero count at k=0 is impossible.
    runs = _full_trio(tmp_path / "audit0")
    for directory in runs:
        _edit_run(directory, lambda run: run["delay_flips"].update({"0": 3}))
    argv[argv.index("--runs") + 1:argv.index("--runs") + 4] = runs
    with pytest.raises(SystemExit):
        main(argv)
    assert "delay_flips[0]" in capsys.readouterr().out


def test_full_mode_requires_a_provenance_sidecar(tmp_path, monkeypatch, capsys):
    from tools.summarize_yaw import main

    relax_full_expectations(monkeypatch)
    runs = _full_trio(tmp_path / "prov")
    os.remove(os.path.join(runs[1], "provenance.json"))
    argv = ["--mode", "full", "--runs"] + runs + [
        "--manifest-hash", MANIFEST_HASH, "--n-boot", "500",
        "--json", str(tmp_path / "p.json"), "--summary", str(tmp_path / "p.txt")]
    with pytest.raises(SystemExit):
        main(argv)
    assert "provenance sidecar missing" in capsys.readouterr().out
    assert not os.path.exists(str(tmp_path / "p.json"))


def test_full_mode_checks_what_the_sidecar_records(tmp_path, monkeypatch, capsys):
    from tools.summarize_yaw import main, validate_full, load_run

    relax_full_expectations(monkeypatch)

    def reasons_for(mutate):
        runs = _full_trio(tmp_path / "sc{}".format(len(capsys.readouterr().out)))
        path = os.path.join(runs[0], "provenance.json")
        sidecar = json.load(open(path))
        mutate(sidecar)
        with open(path, "w") as fout:
            json.dump(sidecar, fout)
        by_label = {label: load_run(directory) for label, directory in
                    zip(("control", "cyl", "released"), runs)}
        return validate_full(by_label, _roles("control", "cyl", "released"),
                             MANIFEST_HASH, PRE_REGISTERED_SETTINGS)

    assert reasons_for(lambda side: None) == []
    assert any("manifest_hash" in reason for reason in
               reasons_for(lambda side: side.update(manifest_hash="0" * 64)))
    assert any("epoch_12.pth" in reason and "provenance" in reason for reason in
               reasons_for(lambda side: side.update(checkpoint_sha256="1" * 64)))
    assert any("git_sha" in reason for reason in
               reasons_for(lambda side: side.update(git_sha="")))
    assert any("no checkpoint_sha256" in reason for reason in
               reasons_for(lambda side: side.pop("checkpoint_sha256")))


def test_the_sidecar_digest_is_the_real_file_digest(tmp_path):
    from tools.summarize_yaw import _file_sha256

    path = str(tmp_path / "blob.bin")
    with open(path, "wb") as fout:
        fout.write(b"xRIR" * 100000)
    assert _file_sha256(path) == sha256_of(path)
    assert _file_sha256("ckpt/xRIR_simple_8_shot/epoch_12.pth") == \
        sha256_of("ckpt/xRIR_simple_8_shot/epoch_12.pth")


def test_the_summary_json_is_strict_json(tmp_path):
    import tools.summarize_yaw as summarize_yaw

    path = str(tmp_path / "out.json")
    summarize_yaw._write_json({"a": float("inf"), "b": [1.0, float("nan")],
                               "c": {"d": -float("inf")}, "e": "text"}, path)
    raw = open(path).read()
    assert "NaN" not in raw and "Infinity" not in raw
    assert json.loads(raw) == {"a": None, "b": [1.0, None], "c": {"d": None}, "e": "text"}


def test_main_writes_only_strict_json(two_runs, tmp_path):
    from tools.summarize_yaw import main

    json_path = str(tmp_path / "s.json")
    main(["--mode", "exploratory", "--runs", two_runs[0], two_runs[1],
          "--labels", "control", "cyl", "--n-boot", "500",
          "--json", json_path, "--summary", str(tmp_path / "s.txt")])
    raw = open(json_path).read()
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw)


def test_full_mode_reconciles_the_metrics_file_with_the_per_sample_arrays(tmp_path,
                                                                         monkeypatch):
    from tools.summarize_yaw import load_run, validate_full

    relax_full_expectations(monkeypatch)

    def reasons_for(mutate=None, remove=False):
        runs = _full_trio(tmp_path / ("rec{}".format(id(mutate) + int(remove))))
        path = os.path.join(runs[0], "metrics_yaw.json")
        if remove:
            os.remove(path)
        elif mutate is not None:
            payload = json.load(open(path))
            mutate(payload)
            with open(path, "w") as fout:
                json.dump(payload, fout)
        by_label = {label: load_run(directory) for label, directory in
                    zip(("control", "cyl", "released"), runs)}
        return validate_full(by_label, _roles("control", "cyl", "released"),
                             MANIFEST_HASH, PRE_REGISTERED_SETTINGS)

    assert reasons_for() == []
    assert any("metrics_yaw.json missing" in reason for reason in reasons_for(remove=True))
    assert any("k=32 edt" in reason and "mean" in reason for reason in reasons_for(
        lambda payload: payload["P"]["32"]["edt"].update(mean=0.5)))
    assert any("n_valid" in reason for reason in reasons_for(
        lambda payload: payload["P"]["32"]["edt"].update(n_valid=1)))


# --------------------------------------------------------------------------------------
# The amended band (v2): the nuisance runs that measure what exp_01's numbers carry
# --------------------------------------------------------------------------------------
def test_the_gate_requires_the_two_nuisance_runs(tmp_path, monkeypatch):
    """exp_01's means also carry unseeded Griffin-Lim phase and TF32 arithmetic.

    The v1 band measured only the reference draw, which is why two cells failed it; the
    default rule now needs both nuisance runs and cannot silently fall back.
    """
    argv, _ = _gate_setup(tmp_path, monkeypatch)
    without = argv[:argv.index("--nuisance-runs")] + argv[argv.index("--released-baseline"):]
    out, code = _run_gate(without, tmp_path, "no_nuisance")
    assert code != 0 and out["gate_pass"] is False
    assert any("--nuisance-runs" in reason for reason in out["gate_reasons"])


def test_the_gate_fails_closed_on_a_bad_nuisance_run(tmp_path, monkeypatch):
    argv, paths = _gate_setup(tmp_path, monkeypatch)

    def fails(name, needle, mutate_argv=lambda a: a):
        out, code = _run_gate(mutate_argv(list(argv)), tmp_path, name)
        assert code != 0, name
        assert out["gate_pass"] is False, name
        assert any(needle in reason for reason in out["gate_reasons"]), \
            (name, needle, out["gate_reasons"])

    def one_only(a):
        del a[a.index("--nuisance-runs") + 2]
        return a

    def three(a):
        a.insert(a.index("--nuisance-runs") + 3, paths["phase"])
        return a

    fails("one_nuisance", "exactly 2 --nuisance-runs", one_only)
    fails("three_nuisance", "exactly 2 --nuisance-runs", three)

    # Two phase runs: the TF32 term would silently be missing.
    _edit_run(paths["tf32"], lambda run: run["meta"].update(gl_seed=1, tf32=False))
    fails("two_phase", "one --gl-seed 1 phase run and one --tf32 run")
    # Neither profile at all.
    _edit_run(paths["tf32"], lambda run: run["meta"].update(gl_seed=7, tf32=True))
    fails("odd_profile", "gl_seed")
    _edit_run(paths["tf32"], lambda run: run["meta"].update(gl_seed=0, tf32=True))

    _edit_run(paths["phase"], lambda run: run["meta"].update(manifest_hash=SEED1_HASH))
    fails("nuisance_hash", "manifest_hash")
    _edit_run(paths["phase"], lambda run: run["meta"].update(manifest_hash=MANIFEST_HASH,
                                                             checkpoint="ckpt/xRIR_cyl_8_shot/epoch_12.pth"))
    fails("nuisance_ckpt", "checkpoint")


def test_the_v2_band_is_twice_the_sum_of_the_three_spread_terms(tmp_path):
    from tools.summarize_yaw import k0_gate_rows, load_run

    control = _k0_run(str(tmp_path / "control"))
    cyl = _k0_run(str(tmp_path / "cyl"), backbone="cylindrical",
                  checkpoint="ckpt/xRIR_cyl_8_shot/epoch_12.pth")
    exp01 = {"control": json.load(open(_exp01_file(str(tmp_path / "e/c.json"), control, 1.001))),
             "cyl": json.load(open(_exp01_file(str(tmp_path / "e/y.json"), cyl, 0.999)))}
    noise = [_k0_run(str(tmp_path / "n1"), scale=1.002),
             _k0_run(str(tmp_path / "n2"), scale=0.998)]
    nuisance = [load_run(_scaled_run(str(tmp_path / "phase"), scale=1.001, gl_seed=1)),
                load_run(_scaled_run(str(tmp_path / "tf32"), scale=0.9995, tf32=True))]

    rows, gate_pass, reasons = k0_gate_rows(
        {"control": control, "cyl": cyl},
        {"primary": "control", "cyl": "cyl", "released": None}, exp01, noise, None,
        nuisance_runs=nuisance, band_rule="v2")
    assert gate_pass is True and reasons == []

    for row in rows:
        mean = row["mean_k0"]
        terms = row["band_terms"]
        assert terms["s_ref"] == pytest.approx(0.002 * mean, rel=1e-6)
        assert terms["s_phase"] == pytest.approx(0.001 * mean, rel=1e-6)
        assert terms["s_tf32"] == pytest.approx(0.0005 * mean, rel=1e-6)
        expected = 2.0 * (terms["s_ref"] + terms["s_phase"] + terms["s_tf32"]) + 1e-6
        assert row["band"] == pytest.approx(expected, rel=1e-12)
        assert row["spread"] == pytest.approx(terms["s_ref"] + terms["s_phase"]
                                              + terms["s_tf32"], rel=1e-12)
        # The amended band is wide enough for the two cells the v1 band failed.
        assert row["abs_diff"] == pytest.approx(0.001 * mean, rel=1e-6)
        assert row["pass"] is True


def test_the_v1_band_is_only_available_when_it_is_asked_for(tmp_path, monkeypatch):
    argv, paths = _gate_setup(tmp_path, monkeypatch)

    # v1 with the amended band's inputs still on the command line is a contradiction.
    out, code = _run_gate(argv + ["--band-rule", "v1"], tmp_path, "v1_with")
    assert code != 0 and out["gate_pass"] is False
    assert any("must not be given --nuisance-runs" in reason
               for reason in out["gate_reasons"])

    # v1 without them reproduces the superseded, narrower band.
    without = _drop_flag(_drop_flag(_drop_flag(argv, "--nuisance-runs"),
                                    "--noise-runs-cyl"), "--shape-runs")
    out, code = _run_gate(without + ["--band-rule", "v1"], tmp_path, "v1_only")
    assert code == 0 and out["gate_pass"] is True
    assert out["band_rule"] == "v1 (reference draw only)"
    for row in out["gate_rows"]:
        assert row["band_terms"]["s_phase"] is None
        assert row["band_terms"]["s_tf32"] is None
        assert row["band"] == pytest.approx(2.0 * row["band_terms"]["s_ref"] + 1e-6)


def test_the_gate_reports_the_band_terms_it_used(tmp_path, monkeypatch):
    argv, _ = _gate_setup(tmp_path, monkeypatch)
    argv = _drop_flag(_drop_flag(argv, "--noise-runs-cyl"), "--shape-runs")
    out, code = _run_gate(argv + ["--band-rule", "v2"], tmp_path, "terms")

    assert code == 0
    assert out["band_rule"] == "v2 (2026-09-06 amendment)"
    assert sorted(out["band_terms"]) == ["control", "cyl", "released"]
    for label, per_metric in out["band_terms"].items():
        assert sorted(per_metric) == sorted(GATE_METRICS), label
        for metric, terms in per_metric.items():
            assert sorted(terms) == ["s_phase", "s_ref", "s_shape", "s_shape_measured",
                                     "s_tf32"], metric
            assert terms["s_shape"] is None, "v2 has no shape term"
            assert terms["s_shape_measured"] is False
            assert all(terms[key] > 0 for key in ("s_ref", "s_phase", "s_tf32")), metric
    text = open(str(tmp_path / "terms.txt")).read()
    assert "S_ref" in text and "S_phase" in text and "S_tf32" in text
    assert "v2 (2026-09-06 amendment)" in text


# --------------------------------------------------------------------------------------
# Band v3: every term measured on the model it is applied to
# --------------------------------------------------------------------------------------
def _drop_flag(argv, flag, keep=0):
    """Remove a flag and its values from an argv, optionally keeping the first ``keep``."""
    start = argv.index(flag)
    end = start + 1
    while end < len(argv) and not argv[end].startswith("--"):
        end += 1
    return argv[:start] + ([flag] + argv[start + 1:start + 1 + keep] if keep else []) \
        + argv[end:]


def test_the_v3_band_needs_the_cylindrical_reference_draw(tmp_path, monkeypatch):
    """The cylindrical model's reference-draw spread is not the control's."""
    argv, _ = _gate_setup(tmp_path, monkeypatch)
    out, code = _run_gate(_drop_flag(argv, "--noise-runs-cyl"), tmp_path, "no_cyl_noise")
    assert code != 0 and out["gate_pass"] is False
    assert any("--noise-runs-cyl" in reason for reason in out["gate_reasons"])


def test_the_gate_fails_closed_on_a_bad_cylindrical_noise_run(tmp_path, monkeypatch):
    argv, paths = _gate_setup(tmp_path, monkeypatch)

    def fails(name, needle, changed=None):
        out, code = _run_gate(changed or argv, tmp_path, name)
        assert code != 0, name
        assert out["gate_pass"] is False, name
        assert any(needle in reason for reason in out["gate_reasons"]), \
            (name, needle, out["gate_reasons"])

    fails("one_cyl_noise", "exactly 2 --noise-runs-cyl",
          _drop_flag(argv, "--noise-runs-cyl", keep=1))
    _edit_run(paths["noise_cyl1"],
              lambda run: run["meta"].update(checkpoint="ckpt/xRIR_simple_8_shot/epoch_12.pth",
                                             backbone="simple"))
    fails("cyl_noise_ckpt", "checkpoint")
    _edit_run(paths["noise_cyl1"],
              lambda run: run["meta"].update(checkpoint="ckpt/xRIR_cyl_8_shot/epoch_12.pth",
                                             backbone="cylindrical",
                                             manifest_hash=MANIFEST_HASH))
    fails("cyl_noise_hash", "manifest")


def test_the_v3_band_uses_each_model_own_reference_draw(tmp_path, monkeypatch):
    argv, _ = _gate_setup(tmp_path, monkeypatch)
    out, code = _run_gate(argv, tmp_path, "v3")

    assert code == 0 and out["gate_pass"] is True
    assert out["band_rule"] == "v3 (2026-09-06 per-model amendment)"
    rows = dict(((row["label"], row["metric"]), row) for row in out["gate_rows"])
    for metric in GATE_METRICS:
        control, cyl = rows[("control", metric)], rows[("cyl", metric)]
        mean = control["mean_k0"]
        assert control["band_terms"]["s_ref"] == pytest.approx(0.002 * mean, rel=1e-6)
        assert cyl["band_terms"]["s_ref"] == pytest.approx(0.003 * mean, rel=1e-6)
        # The phase and TF32 terms are measured on the control and shared.
        for row in (control, cyl):
            assert row["band_terms"]["s_phase"] == pytest.approx(0.001 * mean, rel=1e-6)
            assert row["band_terms"]["s_tf32"] == pytest.approx(0.0005 * mean, rel=1e-6)
            terms = row["band_terms"]
            expected = 2.0 * (terms["s_ref"] + terms["s_phase"] + terms["s_tf32"]
                              + (terms["s_shape"] or 0.0)) + 1e-6
            assert row["band"] == pytest.approx(expected, rel=1e-12)
        assert cyl["band"] > control["band"]
        # The released checkpoint is judged with the control's terms.
        assert rows[("released", metric)]["band"] == pytest.approx(control["band"])


def test_the_older_band_rules_stay_available_only_when_asked_for(tmp_path, monkeypatch):
    argv, _ = _gate_setup(tmp_path, monkeypatch)

    # v2 does not know about the cylindrical reference draw.
    out, code = _run_gate(argv + ["--band-rule", "v2"], tmp_path, "v2_with_cyl")
    assert code != 0
    assert any("--noise-runs-cyl" in reason and "v2" in reason
               for reason in out["gate_reasons"])

    v2_argv = _drop_flag(_drop_flag(argv, "--noise-runs-cyl"), "--shape-runs")
    out, code = _run_gate(v2_argv + ["--band-rule", "v2"], tmp_path, "v2_only")
    assert code == 0 and out["gate_pass"] is True
    assert out["band_rule"] == "v2 (2026-09-06 amendment)"
    for row in out["gate_rows"]:
        assert row["band_terms"]["s_shape"] is None
        assert row["band"] == pytest.approx(2.0 * (row["band_terms"]["s_ref"]
                                                   + row["band_terms"]["s_phase"]
                                                   + row["band_terms"]["s_tf32"]) + 1e-6)


def test_the_shape_term_is_measured_per_model_and_may_be_absent(tmp_path, monkeypatch):
    """exp_01 ran at batch 1; the gate runs at 16, and the shapes do not agree bit for bit."""
    argv, _ = _gate_setup(tmp_path, monkeypatch)
    out, code = _run_gate(argv, tmp_path, "shape")
    assert code == 0 and out["gate_pass"] is True

    rows = dict(((row["label"], row["metric"]), row) for row in out["gate_rows"])
    for metric in GATE_METRICS:
        control, cyl = rows[("control", metric)], rows[("cyl", metric)]
        mean = control["mean_k0"]
        assert control["band_terms"]["s_shape"] == pytest.approx(0.0005 * mean, rel=1e-6)
        assert cyl["band_terms"]["s_shape"] == pytest.approx(0.001 * mean, rel=1e-6)
        assert control["band_terms"]["s_shape_measured"] is True
        assert cyl["band_terms"]["s_shape_measured"] is True
        for row in (control, cyl, rows[("released", metric)]):
            terms = row["band_terms"]
            assert row["band"] == pytest.approx(
                2.0 * (terms["s_ref"] + terms["s_phase"] + terms["s_tf32"]
                       + terms["s_shape"]) + 1e-6, rel=1e-12)
        # The released checkpoint has no shape run; it carries the control's term.
        assert rows[("released", metric)]["band_terms"]["s_shape"] == \
            control["band_terms"]["s_shape"]

    # Without any shape run the term is zero and the table says so.
    out, code = _run_gate(_drop_flag(argv, "--shape-runs"), tmp_path, "no_shape")
    assert code == 0 and out["gate_pass"] is True
    for row in out["gate_rows"]:
        assert row["band_terms"]["s_shape"] == 0.0
        assert row["band_terms"]["s_shape_measured"] is False
    assert "not measured" in open(str(tmp_path / "no_shape.txt")).read()


def test_the_gate_fails_closed_on_a_bad_shape_run(tmp_path, monkeypatch):
    argv, paths = _gate_setup(tmp_path, monkeypatch)

    def fails(name, needle, changed=None):
        out, code = _run_gate(changed or argv, tmp_path, name)
        assert code != 0, name
        assert out["gate_pass"] is False, name
        assert any(needle in reason for reason in out["gate_reasons"]), \
            (name, needle, out["gate_reasons"])

    fails("three_shape", "at most 2 --shape-runs",
          argv + [paths["shape_control"]] if False else
          argv[:argv.index("--shape-runs") + 3] + [paths["shape_control"]]
          + argv[argv.index("--released-baseline"):])

    _edit_run(paths["shape_cyl"], lambda run: run["meta"].update(batch_size=16))
    fails("shape_batch", "meta.batch_size")
    _edit_run(paths["shape_cyl"], lambda run: run["meta"].update(batch_size=1,
                                                                 manifest_hash=SEED1_HASH))
    fails("shape_manifest", "manifest_hash")
    _edit_run(paths["shape_cyl"], lambda run: run["meta"].update(
        manifest_hash=MANIFEST_HASH, checkpoint="checkpoints/xRIR_unseen.pth",
        backbone="simple"))
    fails("shape_unknown", "is neither the control's nor the cylindrical")
    # Two shape runs of the same model leave the other one unmeasured without saying so.
    _edit_run(paths["shape_cyl"], lambda run: run["meta"].update(
        checkpoint="ckpt/xRIR_simple_8_shot/epoch_12.pth"))
    fails("shape_duplicate", "two --shape-runs of the same checkpoint")


def test_the_older_band_rules_reject_the_shape_runs(tmp_path, monkeypatch):
    argv, _ = _gate_setup(tmp_path, monkeypatch)
    without_cyl = _drop_flag(argv, "--noise-runs-cyl")
    out, code = _run_gate(without_cyl + ["--band-rule", "v2"], tmp_path, "v2_shape")
    assert code != 0
    assert any("--shape-runs" in reason for reason in out["gate_reasons"])


# --------------------------------------------------------------------------------------
# Interval labelling: what level each printed interval is actually at
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def exploratory_summary(two_runs, tmp_path_factory):
    """One exploratory summary of the two synthetic runs: ``(json payload, text)``."""
    from tools.summarize_yaw import main

    root = tmp_path_factory.mktemp("labelling")
    json_path = str(root / "summary.json")
    summary_path = str(root / "summary.txt")
    out = main(["--mode", "exploratory", "--runs", two_runs[0], two_runs[1],
                "--labels", "control", "cyl", "--n-boot", "500",
                "--json", json_path, "--summary", summary_path])
    return out, open(summary_path).read()


def _section(text, opener, closer):
    """The block of the summary between two section headings."""
    return text.split(opener)[1].split(closer)[0]


def test_the_config_records_the_level_of_every_reported_interval(exploratory_summary):
    """A consumer must not have to guess which intervals carry the Bonferroni level."""
    out, _ = exploratory_summary

    levels = out["config"]["interval_levels"]
    assert set(levels) == {"degradation_query", "degradation_room", "k0"}
    # The room-cluster bootstrap is called with alpha_adj, exactly like the query-level
    # one, so both are 1 - 0.05/18 intervals; only the k=0 table is nominal.
    assert levels["degradation_query"] == pytest.approx(1.0 - 0.05 / 18)
    assert levels["degradation_room"] == pytest.approx(1.0 - out["config"]["alpha_adj"])
    assert levels["degradation_room"] == levels["degradation_query"]
    assert levels["k0"] == pytest.approx(1.0 - out["config"]["alpha"]) == pytest.approx(0.95)


def test_the_summary_labels_the_room_intervals_as_adjusted(exploratory_summary):
    """The room columns are at the adjusted level too; the text used to imply otherwise."""
    _, text = exploratory_summary

    flat = " ".join(text.split())
    assert "query-level and room-cluster intervals on r_k and D_k are both two-sided" in flat
    assert "99.72%" in flat
    for opener, closer in (("\n2. Acoustic", "\n3. Paired"), ("\n5. H2", "\n6. Delay")):
        section = _section(text, opener, closer)
        assert "adj. room CI" in section, opener
        # ... and no bare "room CI" left to be read as an unadjusted interval.
        assert "room CI" not in section.replace("adj. room CI", ""), opener
    # Section 3 is the one place that really is nominal, and it still says so.
    section3 = _section(text, "\n3. Paired", "\n4. H1")
    assert "nominal 95% CIs, unadjusted" in section3
    assert "adj. room CI" not in section3


# --------------------------------------------------------------------------------------
# Equivalence wording: a failed TOST is "not established", never "inequivalent"
# --------------------------------------------------------------------------------------
def test_the_equivalence_column_never_reads_as_a_yes_no_verdict(exploratory_summary):
    """A TOST that does not clear the margin says nothing; it must not print as "no"."""
    import re

    out, text = exploratory_summary
    section5 = _section(text, "\n5. H2", "\n6. Delay")

    # The cylindrical run is planted at r = 0.00 at k = 32 (inside the +-2 % margin) and
    # at r = 0.05 at k = 64 (outside it), so both renderings appear in this one table.
    assert "equivalent / equivalent" in section5
    assert "not established" in section5
    assert re.search(r"\b(?:yes|no) / (?:yes|no)\b", section5) is None, section5
    # The section says what is being tested against what, so "equivalent" cannot be read
    # as a claim about the two models.
    assert "TOST equivalence of r_cyl to zero within +-2%" in " ".join(text.split())
    assert "at the adjusted level (query / room)" in " ".join(text.split())
    # The JSON keeps the booleans; only the rendering changed.
    equivalences = [row["equivalence"] for row in out["h2"]["rows"]["edt"]
                    if row["equivalence"] is not None]
    assert equivalences and all(isinstance(item["query"]["equivalent"], bool)
                                for item in equivalences)


# --------------------------------------------------------------------------------------
# h1_bounds -- how far the data are from the +10 % margin, computed by the producer
# --------------------------------------------------------------------------------------
def _bound_rows():
    """One synthetic metric's rows: two angles above the margin, one null in each column."""
    return {"edt": [
        # k = 0 is the baseline of the ratio and is never a bound, however large.
        {"k": 0, "deg": 0.0, "hi": 9.0, "r_hi": 9.0},
        {"k": 256, "deg": 180.0, "hi": 0.08, "r_hi": 0.11},
        {"k": 32, "deg": 22.5, "hi": 0.12, "r_hi": None},
        {"k": 64, "deg": 45.0, "hi": None, "r_hi": 0.30},
        {"k": 448, "deg": -45.0, "hi": 0.04, "r_hi": 0.06},
    ]}


def test_h1_bounds_report_the_largest_upper_bound_and_the_cells_above_the_margin():
    from tools.summarize_yaw import h1_bounds

    bounds = h1_bounds(_bound_rows(), 0.10)["edt"]

    assert bounds["max_query_upper"] == {"k": 32, "deg": 22.5, "hi": 0.12}
    assert bounds["max_room_upper"] == {"k": 64, "deg": 45.0, "r_hi": 0.30}
    # Sorted by k, and a row whose bound does not exist is skipped rather than imputed.
    assert bounds["query_upper_above_threshold"] == [{"k": 32, "deg": 22.5, "hi": 0.12}]
    assert bounds["room_upper_above_threshold"] == [{"k": 64, "deg": 45.0, "r_hi": 0.30},
                                                    {"k": 256, "deg": 180.0, "r_hi": 0.11}]
    assert set(bounds) == {"max_query_upper", "max_room_upper",
                           "query_upper_above_threshold", "room_upper_above_threshold"}


def test_h1_bounds_are_empty_when_no_interval_exists():
    from tools.summarize_yaw import h1_bounds

    rows = {"c50": [{"k": 32, "deg": 22.5, "hi": None, "r_hi": None},
                    {"k": 64, "deg": 45.0, "hi": None}]}
    assert h1_bounds(rows, 0.10)["c50"] == {
        "max_query_upper": None, "max_room_upper": None,
        "query_upper_above_threshold": [], "room_upper_above_threshold": []}


def test_h1_bounds_are_strict_at_the_margin():
    from tools.summarize_yaw import h1_bounds

    rows = {"edt": [{"k": 32, "deg": 22.5, "hi": 0.10, "r_hi": 0.1000001}]}
    bounds = h1_bounds(rows, 0.10)["edt"]
    assert bounds["query_upper_above_threshold"] == []
    assert [cell["k"] for cell in bounds["room_upper_above_threshold"]] == [32]


def test_the_json_carries_the_bounds_of_every_run(exploratory_summary):
    """Every run, not only the H1 roles: the page reports the cylindrical one too."""
    out, _ = exploratory_summary

    bounds = out["h1"]["bounds"]
    assert sorted(bounds) == ["control", "cyl"]
    for label in bounds:
        assert sorted(bounds[label]) == ["c50", "edt"]
        for metric, cell in bounds[label].items():
            rows = {row["k"]: row for row in out["acoustic"][label]["P"][metric]}
            top = cell["max_query_upper"]
            assert top["hi"] == rows[top["k"]]["hi"] == max(
                row["hi"] for row in out["acoustic"][label]["P"][metric] if row["k"])
            assert top["deg"] == rows[top["k"]]["deg"]
            for above in cell["query_upper_above_threshold"]:
                assert above["hi"] > out["config"]["threshold"]
                assert rows[above["k"]]["hi"] == above["hi"]
            for above in cell["room_upper_above_threshold"]:
                assert rows[above["k"]]["r_hi"] == above["r_hi"] > out["config"]["threshold"]
    # The control's planted +15 % / +30 % degradations are far above the margin; the
    # cylindrical run's are not.
    assert bounds["control"]["edt"]["room_upper_above_threshold"]
    assert bounds["cyl"]["edt"]["room_upper_above_threshold"] == []


def test_the_summary_prints_the_distance_to_the_margin(exploratory_summary):
    out, text = exploratory_summary
    section4 = _section(text, "\n4. H1", "\n5. H2")

    assert "distance to the +10% margin" in section4
    best = max((cell["max_query_upper"]
                for per_label in out["h1"]["bounds"].values()
                for cell in per_label.values()), key=lambda cell: cell["hi"])
    assert "largest adjusted query-level upper bound: {:+.4f}".format(best["hi"]) in section4
    above = ["{} {} {:.1f} deg".format(label, metric, cell["deg"])
             for label, per_label in out["h1"]["bounds"].items()
             for metric, bound in per_label.items()
             for cell in bound["room_upper_above_threshold"]]
    assert above, "the control's planted degradation must show up here"
    for needle in above:
        assert needle in section4, needle
    assert "room-cluster cells with an adjusted upper bound above +10%" in section4


def test_the_bounds_block_is_printed_in_full_mode_too(two_runs, released_run, tmp_path,
                                                      monkeypatch):
    from tools.summarize_yaw import main

    relax_full_expectations(monkeypatch)
    out = main(["--mode", "full", "--runs", two_runs[0], two_runs[1], released_run,
                "--labels", "control", "cyl", "released",
                "--manifest-hash", MANIFEST_HASH, "--n-boot", "4000",
                "--json", str(tmp_path / "full.json"),
                "--summary", str(tmp_path / "full.txt")])
    text = open(str(tmp_path / "full.txt")).read()
    section4 = _section(text, "\n4. H1", "\n5. H2")

    assert sorted(out["h1"]["bounds"]) == ["control", "cyl", "released"]
    # The block sits after the verdict, so a reader gets the decision first.
    assert section4.index("verdict:") < section4.index("distance to the +10% margin")
    assert "largest adjusted query-level upper bound:" in section4


def test_the_bounds_block_is_omitted_by_the_k0_gate(tmp_path, monkeypatch):
    """A gate run has no rotated angle, so there is no bound to be far from the margin."""
    argv, _ = _gate_setup(tmp_path, monkeypatch)
    out, code = _run_gate(argv, tmp_path, "bounds")

    assert code == 0
    assert "bounds" not in out["h1"]
    assert "distance to the" not in open(str(tmp_path / "bounds.txt")).read()


# --------------------------------------------------------------------------------------
# metric_names / metric_units -- the label and the unit of every emitted metric
# --------------------------------------------------------------------------------------
def _emitted_metric_keys(out):
    """Every metric key that appears as a metric in the JSON, wherever it appears."""
    keys = set(row["metric"] for row in out.get("k0", []))
    keys.update(row["metric"] for row in out.get("gate_rows", []))
    for block in ("spectral", "acoustic"):
        for per_label in out.get(block, {}).values():
            for per_condition in per_label.values():
                keys.update(per_condition)
    for per_label in out.get("h1", {}).get("bounds", {}).values():
        keys.update(per_label)
    keys.update(out.get("h2", {}).get("rows", {}))
    return keys


def test_the_config_names_and_gives_a_unit_for_every_metric_the_json_emits(
        exploratory_summary):
    """A results page must never have to invent a metric label or a unit of its own."""
    out, _ = exploratory_summary

    names, units = out["config"]["metric_names"], out["config"]["metric_units"]
    emitted = _emitted_metric_keys(out)
    assert emitted == {"edt", "c50", "t60", "loss", "log_mse", "consistency"}
    assert emitted <= set(names) and emitted <= set(units)
    assert set(names) == set(units)
    assert names == {"edt": "EDT error", "c50": "C50 error", "t60": "T60 error",
                     "loss": "test loss = STFT L1 + 0.01 x decay",
                     "log_mse": "log-STFT MSE",
                     "consistency": "consistency = mean|log-spec(k) - log-spec(0)|"}
    # "" is the unit of a unitless metric, so a page can print "name [unit]" or drop the
    # bracket without a special case per metric.
    assert units == {"edt": "s", "c50": "dB", "t60": "%",
                     "loss": "", "log_mse": "", "consistency": ""}


def test_the_metric_names_and_units_are_the_ones_the_summary_text_prints(
        exploratory_summary):
    out, text = exploratory_summary
    names, units = out["config"]["metric_names"], out["config"]["metric_units"]

    section1 = _section(text, "\n1. Spectral", "\n2. Acoustic")
    assert names["consistency"] in section1
    section2 = _section(text, "\n2. Acoustic", "\n3. Paired")
    assert "EDT {}, C50 {}, T60 {}".format(units["edt"], units["c50"],
                                           units["t60"]) in section2


def test_the_k0_gate_json_names_the_metrics_it_gates_on(tmp_path, monkeypatch):
    argv, _ = _gate_setup(tmp_path, monkeypatch)
    out, code = _run_gate(argv, tmp_path, "names")

    assert code == 0
    names = out["config"]["metric_names"]
    assert set(row["metric"] for row in out["gate_rows"]) <= set(names)
    assert names["edt"] == "EDT error"
    assert out["config"]["metric_units"]["c50"] == "dB"


# --------------------------------------------------------------------------------------
# condition_definitions / angle_counts / run_descriptions
# --------------------------------------------------------------------------------------
def test_the_config_defines_the_two_conditions_the_way_section_1_does(
        exploratory_summary):
    """"P" and "E" are opaque letters; the page must not have to gloss them itself."""
    out, text = exploratory_summary

    definitions = out["config"]["condition_definitions"]
    assert definitions == {"P": "pinned k=0 alignment", "E": "end to end"}
    section1 = _section(text, "\n1. Spectral", "\n2. Acoustic")
    assert "condition P = {}, E = {}".format(definitions["P"],
                                             definitions["E"]) in section1
    # Exactly the conditions the JSON is keyed by.
    assert set(definitions) == set(out["spectral"]["control"])


def test_the_config_counts_the_three_preregistered_angle_grids(exploratory_summary):
    """The page says "18 angles" / "9 rotated acoustic angles"; the counts come from here."""
    from tools.summarize_yaw import (PREREGISTERED_ACOUSTIC_COLS,
                                     PREREGISTERED_E_ACOUSTIC_COLS,
                                     PREREGISTERED_SPECTRAL_COLS)
    out, _ = exploratory_summary

    counts = out["config"]["angle_counts"]
    assert counts == {"spectral": 18, "acoustic": 10, "e_acoustic": 4}
    assert counts["spectral"] == len(PREREGISTERED_SPECTRAL_COLS)
    assert counts["acoustic"] == len(PREREGISTERED_ACOUSTIC_COLS)
    assert counts["e_acoustic"] == len(PREREGISTERED_E_ACOUSTIC_COLS)
    # They count the pre-registered grids, not this run's, which may be a subset.
    for key, cols in (("spectral", "preregistered_spectral_cols"),
                      ("acoustic", "preregistered_acoustic_cols"),
                      ("e_acoustic", "preregistered_e_acoustic_cols")):
        assert counts[key] == len(out["config"][cols])


def test_the_config_describes_each_run_the_way_the_header_line_does(exploratory_summary):
    out, text = exploratory_summary

    descriptions = out["config"]["run_descriptions"]
    assert descriptions == {
        "control": "control (simple, ckpt/xRIR_simple_8_shot/epoch_12.pth)",
        "cyl": "cyl (cylindrical, ckpt/xRIR_cyl_8_shot/epoch_12.pth)"}
    assert set(descriptions) == set(out["config"]["labels"])
    assert "runs: " + ", ".join(descriptions[label]
                                for label in out["config"]["labels"]) in text
    for label, description in descriptions.items():
        meta = out["meta"][label]
        assert description == "{} ({}, {})".format(label, meta["backbone"],
                                                   meta["checkpoint"])


def test_the_k0_gate_json_carries_the_same_three_descriptions(tmp_path, monkeypatch):
    argv, _ = _gate_setup(tmp_path, monkeypatch)
    out, code = _run_gate(argv, tmp_path, "described")

    assert code == 0
    config = out["config"]
    assert config["condition_definitions"] == {"P": "pinned k=0 alignment",
                                               "E": "end to end"}
    assert config["angle_counts"] == {"spectral": 18, "acoustic": 10, "e_acoustic": 4}
    assert set(config["run_descriptions"]) == {"control", "cyl", "released"}
    assert config["run_descriptions"]["released"] == \
        "released (simple, checkpoints/xRIR_unseen.pth)"


# --------------------------------------------------------------------------------------
# delay_flips_entity / delay_flips_max -- what the delay-flip counts count
# --------------------------------------------------------------------------------------
def _flat(text):
    """The text with every run of whitespace collapsed, so a wrapped sentence matches."""
    return " ".join(text.split())


def test_the_json_says_what_a_delay_flip_counts(exploratory_summary):
    """A bare integer per angle is meaningless without the entity it counts."""
    out, text = exploratory_summary

    entity = out["delay_flips_entity"]
    assert entity == ("(query, reference) pairs whose integer direct-path delay moves "
                      "under the rotation -- the numerical noise condition P excludes")
    section6 = _section(text, "\n6. Delay-flip", "\n7. Decomposition")
    assert entity + "." in _flat(section6)


def test_delay_flips_max_is_omitted_when_no_run_records_a_shot_count(
        exploratory_summary):
    """eval_yaw_rotation.py's meta has no num_shot; guessing 8 would be inventing it."""
    out, _ = exploratory_summary

    assert "delay_flips_max" not in out
    assert all("num_shot" not in meta for meta in out["meta"].values())


def test_delay_flips_max_is_the_number_of_pairs_when_the_runs_record_the_shot_count(
        two_runs, tmp_path):
    from tools.summarize_yaw import main

    directories = []
    for index, source in enumerate(two_runs):
        directory = str(tmp_path / "shot{}".format(index))
        shutil.copytree(source, directory)
        _edit_run(directory, lambda run: run["meta"].update(num_shot=8))
        directories.append(directory)
    out = main(["--mode", "exploratory", "--runs", directories[0], directories[1],
                "--labels", "control", "cyl", "--n-boot", "200",
                "--json", str(tmp_path / "shot.json"),
                "--summary", str(tmp_path / "shot.txt")])

    assert out["delay_flips_max"] == out["n_queries"] * 8
    assert all(counts <= out["delay_flips_max"]
               for per_label in out["delay_flips"].values()
               for counts in per_label.values())
    # Two runs that disagree cannot name one denominator, so none is written.
    _edit_run(directories[1], lambda run: run["meta"].update(num_shot=4))
    out = main(["--mode", "exploratory", "--runs", directories[0], directories[1],
                "--labels", "control", "cyl", "--n-boot", "200",
                "--json", str(tmp_path / "shot2.json"),
                "--summary", str(tmp_path / "shot2.txt")])
    assert "delay_flips_max" not in out


def test_the_k0_gate_json_also_says_what_a_delay_flip_counts(tmp_path, monkeypatch):
    argv, _ = _gate_setup(tmp_path, monkeypatch)
    out, code = _run_gate(argv, tmp_path, "flips")

    assert code == 0
    assert out["delay_flips_entity"].startswith("(query, reference) pairs")
    assert set(out["delay_flips"]) == {"control", "cyl", "released"}


# --------------------------------------------------------------------------------------
# decomposition_stages / decomposition_note / k0_note
# --------------------------------------------------------------------------------------
def test_the_json_names_every_decomposition_stage(exploratory_summary):
    """The four keys are field names, not axis labels; the page needs the labels."""
    out, _ = exploratory_summary

    stages = out["decomposition_stages"]
    assert stages == {"tokens_rel_change": "receiver-view tokens",
                      "pooled_rel_change": "pooled receiver feature",
                      "coord_rel_change": "query-source coordinate embedding",
                      "logspec_rel_change": "output log-spectrogram"}
    # Exactly the numeric fields of a decomposition record, in its own order.
    decomposition = out["decomposition"]["cyl"]
    assert [key for key in decomposition if key not in ("k", "n_batches")] == list(stages)


def test_the_decomposition_note_carries_the_scope_sentence_section_7_prints(
        exploratory_summary):
    out, text = exploratory_summary

    note = out["decomposition_note"]
    scope = ("Scope: receiver-view tokens and pooling plus the query-source coordinate "
             "embedding; reference-coordinate features not decomposed.")
    assert note == scope + (" Relative changes of different representation spaces; "
                            "not additive.")
    section7 = _section(text, "\n7. Decomposition", "\n8. Bootstrap")
    assert scope in _flat(section7)


def test_the_json_carries_the_k0_note_the_summary_prints(exploratory_summary):
    out, text = exploratory_summary

    note = out["k0_note"]
    assert note == ("Same references and same Griffin-Lim phases for both models: this "
                    "supersedes exp_01's epoch-12 comparison.")
    assert note in _section(text, "\n3. Paired", "\n4. H1")


def test_the_k0_gate_json_carries_the_notes_of_the_sections_it_prints(tmp_path,
                                                                     monkeypatch):
    argv, _ = _gate_setup(tmp_path, monkeypatch)
    out, code = _run_gate(argv, tmp_path, "notes")
    text = open(str(tmp_path / "notes.txt")).read()

    assert code == 0
    assert out["k0_note"] in text
    assert out["decomposition_note"].split(".")[0] in _flat(text)
    # The stage labels describe section 7 whether or not this mode fills it in.
    assert sorted(out["decomposition_stages"]) == ["coord_rel_change",
                                                   "logspec_rel_change",
                                                   "pooled_rel_change",
                                                   "tokens_rel_change"]


def test_full_mode_serializes_every_label_the_consumers_read(two_runs, released_run,
                                                             tmp_path, monkeypatch):
    """One check that the confirmatory artefact carries the whole descriptive block."""
    from tools.summarize_yaw import main

    relax_full_expectations(monkeypatch)
    out = main(["--mode", "full", "--runs", two_runs[0], two_runs[1], released_run,
                "--labels", "control", "cyl", "released",
                "--manifest-hash", MANIFEST_HASH, "--n-boot", "4000",
                "--json", str(tmp_path / "full.json"),
                "--summary", str(tmp_path / "full.txt")])
    text = open(str(tmp_path / "full.txt")).read()

    config = out["config"]
    assert set(config["metric_names"]) == set(config["metric_units"]) == {
        "edt", "c50", "t60", "loss", "log_mse", "consistency"}
    assert config["condition_definitions"] == {"P": "pinned k=0 alignment",
                                               "E": "end to end"}
    assert config["angle_counts"] == {"spectral": 18, "acoustic": 10, "e_acoustic": 4}
    assert set(config["run_descriptions"]) == {"control", "cyl", "released"}
    assert "runs: " + ", ".join(config["run_descriptions"][label]
                                for label in config["labels"]) in text
    assert out["delay_flips_entity"] + "." in _flat(text)
    assert "delay_flips_max" not in out
    assert out["k0_note"] in text
    assert out["decomposition_note"].split(".")[0] in _flat(text)
    assert len(out["decomposition_stages"]) == 4
    # The scientific rules, in the confirmatory artefact and printed there too.
    assert _header(text, "\n4. ", "\n5. H2", 2) == out["h1"]["rule"]
    assert out["h1"]["verdict"] in out["h1"]["wording_rule"]


# --------------------------------------------------------------------------------------
# h1.rule / h1.wording_rule -- the pre-registered H1 rule as a producer string
# --------------------------------------------------------------------------------------
def _header(text, opener, closer, lines):
    """A section's first ``lines`` physical lines, unwrapped into one sentence."""
    return _flat("\n".join(_section(text, opener, closer).split("\n")[:lines]))


def test_the_json_states_the_h1_rule_section_4_prints(exploratory_summary):
    """A results page must quote the decision rule, never compose one of its own."""
    out, text = exploratory_summary

    rule = out["h1"]["rule"]
    assert rule == ("H1 (joint yaw rotation substantially degrades the model): supported "
                    "if the adjusted lower bound of r exceeds +10% for EDT or C50 at any "
                    "angle k != 0, condition P.")
    # Word for word section 4's header, only unwrapped: the two cannot drift.
    assert _header(text, "\n4. ", "\n5. H2", 2) == rule
    # And the margin in the sentence is the one this run was configured with.
    assert "{:+.0%}".format(out["config"]["threshold"]) in rule


def test_the_json_states_the_verdict_wording_rule_the_code_applies(exploratory_summary):
    """The five-way verdict wording is a rule of the analysis, not of the page."""
    from tools.summarize_yaw import H1_WORDING, h1_wording

    out, _ = exploratory_summary
    wording_rule = out["h1"]["wording_rule"]

    assert set(H1_WORDING) == {(True, True), (True, False), (False, True), (False, False)}
    for (primary, released), verdict in H1_WORDING.items():
        # The sentence is built from the mapping h1_wording itself reads.
        assert h1_wording(primary, released) == verdict
        assert '"{}"'.format(verdict) in wording_rule
    assert "counts as not passing" in wording_rule
    # This mode issues no verdict at all, and the rule says which modes do.
    assert out["h1"]["verdict"] == "not evaluated (exploratory mode)"
    assert '"not evaluated (<mode> mode)"' in wording_rule


def test_the_k0_gate_json_carries_the_rules_of_the_sections_it_prints(tmp_path,
                                                                     monkeypatch):
    """The gate prints section 4's header too, so it serialises its rule too."""
    argv, _ = _gate_setup(tmp_path, monkeypatch)
    out, code = _run_gate(argv, tmp_path, "rules")
    text = open(str(tmp_path / "rules.txt")).read()

    assert code == 0
    assert _header(text, "\n4. ", "\n5. H2", 2) == out["h1"]["rule"]
    assert out["h1"]["verdict"] == "not evaluated (k0-gate mode)"
    assert out["h1"]["wording_rule"].startswith("Verdict wording (--mode full only")
