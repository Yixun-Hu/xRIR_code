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


# ------------------------------------------------- test 3: acoustic_gap / raw_measures

def _decaying_ir(seed, n=9600, tau=1200.0, scale=1.0):
    """A synthetic, well-behaved impulse response (noise under an exponential decay)."""
    rng = np.random.RandomState(seed)
    envelope = np.exp(-np.arange(n) / tau)
    return (scale * rng.randn(n) * envelope).astype(np.float32)


@pytest.fixture(scope="module")
def evaluator():
    from eval_unseen import Evaluator
    return Evaluator()


def test_acoustic_gap_of_a_waveform_with_itself_is_exactly_zero(evaluator):
    waves = np.stack([_decaying_ir(0), _decaying_ir(1, tau=800.0)])
    gap = pilot.acoustic_gap(waves, waves, evaluator)
    assert sorted(gap) == ["c50_gap", "edt_gap", "t60_gap"]
    for name in gap:
        assert gap[name].shape == (2,)
        np.testing.assert_array_equal(gap[name], np.zeros(2))


def test_acoustic_gap_skips_t60_when_it_is_not_wanted(evaluator):
    waves = np.stack([_decaying_ir(2)])
    gap = pilot.acoustic_gap(waves, waves + 1e-3, evaluator, want_t60=False)
    assert np.isnan(gap["t60_gap"]).all()
    assert np.isfinite(gap["edt_gap"]).all()


def test_acoustic_gap_and_raw_measures_return_nan_on_a_silent_waveform(evaluator):
    zeros = np.zeros((1, 9600), dtype=np.float32)
    gap = pilot.acoustic_gap(zeros, zeros, evaluator)
    for name in gap:
        assert np.isnan(gap[name]).all()
    raw = pilot.raw_measures(zeros, evaluator)
    for name in raw:
        assert np.isnan(raw[name]).all()


def test_raw_measures_reproduce_the_canonical_errors_where_both_are_finite(evaluator):
    from tools.per_sample_metrics import acoustic_metrics

    preds = np.stack([_decaying_ir(3), _decaying_ir(4, tau=2000.0)])
    gts = np.stack([_decaying_ir(5, tau=900.0), _decaying_ir(6, tau=1500.0)])
    raw_pred = pilot.raw_measures(preds, evaluator)
    raw_gt = pilot.raw_measures(gts, evaluator)
    canonical = [acoustic_metrics(preds[i], gts[i], evaluator) for i in range(2)]

    for i, cell in enumerate(canonical):
        assert np.isfinite(cell["edt"]) and np.isfinite(cell["c50"]) and np.isfinite(cell["t60"])
        np.testing.assert_allclose(cell["edt"], abs(raw_gt["edt"][i] - raw_pred["edt"][i]),
                                   rtol=1e-12)
        np.testing.assert_allclose(cell["c50"], abs(raw_gt["c50"][i] - raw_pred["c50"][i]),
                                   rtol=1e-12)
        np.testing.assert_allclose(
            cell["t60"],
            abs(raw_gt["t60"][i] - raw_pred["t60"][i]) / raw_gt["t60"][i] * 100.0,
            rtol=1e-12)


def test_raw_measures_only_look_at_the_metric_window(evaluator):
    wave = _decaying_ir(7)
    tail_changed = wave.copy()
    tail_changed[pilot.METRIC_WINDOW:] = 3.0
    a = pilot.raw_measures(wave[None, :], evaluator)
    b = pilot.raw_measures(tail_changed[None, :], evaluator)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])


def test_acoustic_gap_rejects_mismatched_blocks(evaluator):
    with pytest.raises(ValueError):
        pilot.acoustic_gap(np.zeros((2, 16)), np.zeros((3, 16)), evaluator)
    with pytest.raises(ValueError):
        pilot.raw_measures(np.zeros(16), evaluator)


# ------------------------------- test 4: device-agnostic apply_delay and its patch point

def test_device_agnostic_apply_delay_shifts_like_the_model(tmp_path):
    signal = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    delays = torch.tensor([2, -1, 0], dtype=torch.int32)
    out = pilot.device_agnostic_apply_delay(signal, delays)

    expected = torch.zeros_like(signal)
    expected[0, 2:] = signal[0, :-2]      # positive: shift right, zero-pad the front
    expected[1, :-1] = signal[1, 1:]      # negative: shift left, zero-pad the tail
    expected[2] = signal[2]               # zero: copy
    torch.testing.assert_close(out, expected)
    assert out.device == signal.device and out.dtype == signal.dtype
    torch.testing.assert_close(signal, torch.arange(12, dtype=torch.float32).reshape(3, 4))


def test_device_agnostic_apply_delay_works_where_the_model_version_cannot():
    import model.xRIR as model_xrir

    signal = torch.randn(2, 8)
    delays = torch.tensor([3, -2], dtype=torch.int32)
    if torch.cuda.is_available():
        reference = model_xrir.apply_delay(signal.cuda(), delays.cuda()).cpu()
        torch.testing.assert_close(pilot.device_agnostic_apply_delay(signal, delays),
                                   reference)
    else:
        with pytest.raises(Exception):
            model_xrir.apply_delay(signal, delays)
        assert pilot.device_agnostic_apply_delay(signal, delays).shape == signal.shape


def test_patched_apply_delay_installs_and_restores_the_module_attribute():
    import model.xRIR as model_xrir

    original = model_xrir.apply_delay
    with pilot.patched_apply_delay():
        assert model_xrir.apply_delay is pilot.device_agnostic_apply_delay
    assert model_xrir.apply_delay is original


def test_patched_apply_delay_restores_even_when_the_block_raises():
    import model.xRIR as model_xrir

    original = model_xrir.apply_delay
    with pytest.raises(RuntimeError):
        with pilot.patched_apply_delay():
            raise RuntimeError("boom")
    assert model_xrir.apply_delay is original


def test_patched_apply_delay_lets_the_model_run_on_the_cpu():
    from model.xRIR_cyl import build_xrir

    torch.manual_seed(0)
    model = build_xrir("simple", 2).eval()
    refs = torch.randn(2, 2, 512)
    src = torch.randn(2, 3)
    ref_locs = torch.randn(2, 2, 3)
    with pytest.raises(Exception):
        with torch.no_grad():
            model.shift_and_align(refs, src, ref_locs)
    with pilot.patched_apply_delay():
        with torch.no_grad():
            aligned = model.shift_and_align(refs, src, ref_locs)
    assert aligned.shape == refs.shape
