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
