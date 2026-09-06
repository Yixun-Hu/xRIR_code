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
        json.dump(run, fout)
    return directory


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
