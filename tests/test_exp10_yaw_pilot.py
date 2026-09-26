"""Tests for ``tools/exp10_yaw_pilot.py`` (exp_10 ``yaw_pilot``, plan v3.1 section 6).

The pilot measures two families per query: how far the *prediction* moves when the
conditioning is yaw-rotated (Metric 1, no ground truth) and how the *accuracy vs GT*
changes (Metric 2).  Both are built out of the frozen exp_03 helpers, so most of these
tests pin the composition -- "this call is exactly that call" -- rather than re-deriving
the numerics the exp_03 record already certified.
"""
import numpy as np
import pytest
import torch

from tools import exp10_yaw_pilot as pilot


# ---------------------------------------------------------------- test 1: waveform_gap

def test_waveform_gap_of_a_waveform_with_itself_is_exactly_zero():
    waves = np.random.RandomState(0).randn(4, 32).astype(np.float32)
    gap = pilot.waveform_gap(waves, waves)
    assert sorted(gap) == ["wave_mad", "wave_rel_l2"]
    for name in gap:
        assert gap[name].shape == (4,)
        np.testing.assert_array_equal(gap[name], np.zeros(4))


def test_waveform_gap_rel_l2_of_a_doubled_waveform_is_one():
    waves = np.random.RandomState(1).randn(3, 64).astype(np.float32)
    gap = pilot.waveform_gap(waves, 2.0 * waves)
    np.testing.assert_allclose(gap["wave_rel_l2"], np.ones(3), rtol=1e-6)
    np.testing.assert_allclose(gap["wave_mad"], np.abs(waves).mean(axis=1), rtol=1e-6)


def test_waveform_gap_with_a_zero_reference_stays_finite():
    zeros = np.zeros((2, 8), dtype=np.float32)
    other = np.ones((2, 8), dtype=np.float32)
    assert np.isfinite(pilot.waveform_gap(zeros, zeros)["wave_rel_l2"]).all()
    gap = pilot.waveform_gap(zeros, other)
    assert np.isfinite(gap["wave_rel_l2"]).all()
    np.testing.assert_allclose(gap["wave_mad"], np.ones(2))


def test_waveform_gap_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        pilot.waveform_gap(np.zeros((2, 8)), np.zeros((3, 8)))
    with pytest.raises(ValueError):
        pilot.waveform_gap(np.zeros(8), np.zeros(8))


# ------------------------------------------------------------- test 2: spectrogram_gap

def test_spectrogram_gap_of_a_spectrogram_with_itself_is_exactly_zero():
    out = torch.randn(3, 5, 7, 1)
    gap = pilot.spectrogram_gap(out, out)
    assert sorted(gap) == ["logspec_mad", "mag_rel_l2"]
    for name in gap:
        assert gap[name].shape == (3,)
        np.testing.assert_array_equal(gap[name], np.zeros(3))


def test_spectrogram_gap_matches_a_hand_computation():
    torch.manual_seed(0)
    out_0 = torch.randn(2, 4, 6, 1)
    out_k = out_0 + 0.25 * torch.randn(2, 4, 6, 1)
    gap = pilot.spectrogram_gap(out_0, out_k)

    a = out_0[..., 0].double().numpy()
    b = out_k[..., 0].double().numpy()
    expected_mad = np.abs(b - a).reshape(2, -1).mean(axis=1)
    mag_0 = np.exp(a) - 1e-8
    mag_k = np.exp(b) - 1e-8
    expected_rel = (np.linalg.norm((mag_k - mag_0).reshape(2, -1), axis=1) /
                    np.linalg.norm(mag_0.reshape(2, -1), axis=1))
    np.testing.assert_allclose(gap["logspec_mad"], expected_mad, rtol=1e-12)
    np.testing.assert_allclose(gap["mag_rel_l2"], expected_rel, rtol=1e-12)


def test_spectrogram_gap_uses_the_magnitude_offset_not_the_bare_exponential():
    # exp(out) - 1e-8 differs from exp(out) exactly where the magnitude is tiny.
    out_0 = torch.full((1, 2, 2, 1), -18.0)
    out_k = torch.full((1, 2, 2, 1), -18.5)
    gap = pilot.spectrogram_gap(out_0, out_k)
    with_offset = (abs(np.exp(-18.5) - np.exp(-18.0)) /
                   abs(np.exp(-18.0) - 1e-8))
    np.testing.assert_allclose(gap["mag_rel_l2"], [with_offset], rtol=1e-9)


def test_spectrogram_gap_accepts_three_dimensional_outputs_and_rejects_mismatches():
    out = torch.randn(2, 3, 4)
    np.testing.assert_array_equal(pilot.spectrogram_gap(out, out)["logspec_mad"],
                                  np.zeros(2))
    with pytest.raises(ValueError):
        pilot.spectrogram_gap(torch.randn(2, 3, 4), torch.randn(3, 3, 4))
