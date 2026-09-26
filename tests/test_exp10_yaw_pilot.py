"""Tests for ``tools/exp10_yaw_pilot.py`` (exp_10 ``yaw_pilot``, plan v3.1 section 6).

The pilot measures two families per query: how far the *prediction* moves when the
conditioning is yaw-rotated (Metric 1, no ground truth) and how the *accuracy vs GT*
changes (Metric 2).  Both are built out of the frozen exp_03 helpers, so most of these
tests pin the composition -- "this call is exactly that call" -- rather than re-deriving
the numerics the exp_03 record already certified.
"""
import json
import os

import numpy as np
import pytest
import torch

from tools import exp10_yaw_pilot as pilot

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_MANIFEST = os.path.join(REPO_ROOT, "ckpt", "yaw_rotation", "reference_manifest.json")


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


# ------------------------------------------------------------- test 5: angle_logspec

class _CountingAlign:
    """An instance-level ``shift_and_align`` that counts its calls and then delegates."""

    def __init__(self, model):
        self.model = model
        self.calls = 0

    def __call__(self, x, src_loc, ref_ir_locs):
        self.calls += 1
        return type(self.model).shift_and_align(self.model, x, src_loc, ref_ir_locs)


@pytest.fixture(scope="module")
def tiny_scene():
    """A random-init two-shot xRIR plus one synthetic two-query batch (CPU)."""
    from model.xRIR_cyl import build_xrir

    torch.manual_seed(0)
    model = build_xrir("simple", 2).eval()
    batch = {
        "depth": torch.rand(2, 3, 256, 512) * 5.0,
        "refs": torch.randn(2, 2, 9600) * 0.1,
        "src": torch.randn(2, 3),
        "ref_locs": torch.randn(2, 2, 3),
        "tgt": torch.randn(2, 1, 9600) * 0.1,
    }
    with pilot.patched_apply_delay():
        with torch.no_grad():
            batch["aligned0"] = model.shift_and_align(batch["refs"], batch["src"],
                                                      batch["ref_locs"])
    return model, batch


def _angle_logspec(model, batch, k):
    with pilot.patched_apply_delay():
        return pilot.angle_logspec(model, batch["depth"], batch["refs"], batch["src"],
                                   batch["ref_locs"], batch["tgt"], k, batch["aligned0"])


def test_angle_logspec_at_zero_equals_the_plain_forward(tiny_scene):
    model, batch = tiny_scene
    with pilot.patched_apply_delay():
        with torch.no_grad():
            plain, plain_tgt = model(batch["depth"], batch["refs"], batch["src"],
                                     batch["ref_locs"], batch["tgt"])
    out, tgt_spec = _angle_logspec(model, batch, 0)
    assert torch.equal(out, plain)
    assert torch.equal(tgt_spec, plain_tgt)


def test_angle_logspec_is_the_trusted_rotation_composition(tiny_scene):
    from tools.yaw_rotation import fixed_alignment, rotate_scene_yaw

    model, batch = tiny_scene
    out, tgt_spec = _angle_logspec(model, batch, 128)

    depth_k, src_k, ref_locs_k = rotate_scene_yaw(
        batch["depth"], batch["src"], batch["ref_locs"], 128, W=batch["depth"].shape[-1])
    with pilot.patched_apply_delay():
        with torch.no_grad():
            with fixed_alignment(model, batch["aligned0"]):
                expected, expected_tgt = model(depth_k, batch["refs"], src_k,
                                               ref_locs_k, batch["tgt"])
    assert torch.equal(out, expected)
    assert torch.equal(tgt_spec, expected_tgt)
    assert not torch.equal(out, _angle_logspec(model, batch, 0)[0])


def test_angle_logspec_leaves_the_reference_and_target_audio_untouched(tiny_scene):
    model, batch = tiny_scene
    refs_before = batch["refs"].clone()
    tgt_before = batch["tgt"].clone()
    depth_before = batch["depth"].clone()
    _angle_logspec(model, batch, 256)
    assert torch.equal(batch["refs"], refs_before)
    assert torch.equal(batch["tgt"], tgt_before)
    assert torch.equal(batch["depth"], depth_before)


def test_angle_logspec_never_recomputes_the_alignment(tiny_scene):
    model, batch = tiny_scene
    counter = _CountingAlign(model)
    model.shift_and_align = counter
    try:
        _angle_logspec(model, batch, 64)
        assert counter.calls == 0
        assert model.shift_and_align is counter       # the pin was restored
    finally:
        del model.shift_and_align


def test_angle_logspec_does_not_depend_on_the_order_of_the_angles(tiny_scene):
    model, batch = tiny_scene
    forward = [_angle_logspec(model, batch, k)[0] for k in (0, 64, 384)]
    backward = {k: _angle_logspec(model, batch, k)[0] for k in (384, 64, 0)}
    for out, k in zip(forward, (0, 64, 384)):
        assert torch.equal(out, backward[k])


def test_angle_logspec_restores_the_alignment_after_an_exception(tiny_scene):
    model, batch = tiny_scene
    with pytest.raises(Exception):
        with pilot.patched_apply_delay():
            pilot.angle_logspec(model, batch["depth"], batch["refs"], batch["src"],
                                batch["ref_locs"], None, 64, batch["aligned0"])
    assert "shift_and_align" not in vars(model)


# -------------------------------------------------------------------- test 6: invert

def _fake_logspec(seed, n=2, freqs=63, frames=310):
    torch.manual_seed(seed)
    return (torch.randn(n, freqs, frames, 1) * 0.5 - 2.0)


def test_invert_is_the_frozen_seeded_griffin_lim_composition():
    from tools.per_sample_metrics import griffin_lim_seeded, sample_seed

    out = _fake_logspec(0)
    keys = ["Room/a_hybrid_IR.wav", "Room/b_hybrid_IR.wav"]
    waves = pilot.invert(out, keys, 0)

    assert waves.shape == (2, pilot.PADDED_LEN)
    assert waves.dtype == np.float32
    for i, key in enumerate(keys):
        mag = (torch.exp(out[i:i + 1]) - 1e-8)[..., 0]
        expected = griffin_lim_seeded(mag, sample_seed(0, key))[0].numpy()
        assert expected.shape == (pilot.NATIVE_LEN,)
        np.testing.assert_array_equal(waves[i, :pilot.NATIVE_LEN], expected)
        np.testing.assert_array_equal(waves[i, pilot.NATIVE_LEN:], np.zeros(21))


def test_invert_is_a_pure_function_of_output_key_and_seed():
    out = _fake_logspec(1)
    keys = ["Room/a_hybrid_IR.wav", "Room/b_hybrid_IR.wav"]
    first = pilot.invert(out, keys, 0)
    np.testing.assert_array_equal(first, pilot.invert(out, keys, 0))
    other_key = pilot.invert(out, ["Room/zzz_hybrid_IR.wav", keys[1]], 0)
    assert not np.array_equal(first[0], other_key[0])
    np.testing.assert_array_equal(first[1], other_key[1])
    other_seed = pilot.invert(out, keys, 1)
    assert not np.array_equal(first[0], other_seed[0])


def test_invert_rejects_a_key_count_or_layout_it_cannot_pair():
    out = _fake_logspec(2)
    with pytest.raises(ValueError):
        pilot.invert(out, ["only-one"], 0)
    with pytest.raises(ValueError):
        pilot.invert(out[..., 0], ["a", "b"], 0)


# ------------------------------------------------------------ test 7: evaluate_batch

def _collated_batch(n, num_shot=2, seed=3):
    """A synthetic ``ManifestDataset`` seven-tuple, collated, with ``n`` real rows."""
    torch.manual_seed(seed)
    keys = ["Room/Room_idx_1/S00{}_R001_hybrid_IR.wav".format(i) for i in range(n)]
    envelope = torch.exp(-torch.arange(9600, dtype=torch.float32) / 1200.0)
    return (torch.zeros(n, 3),
            torch.randn(n, 3),
            torch.rand(n, 3, 256, 512) * 5.0,
            (torch.randn(n, 1, 9600) * envelope),
            (torch.randn(n, num_shot, 9600) * envelope),
            torch.randn(n, num_shot, 3),
            keys)


@pytest.fixture(scope="module")
def tiny_model():
    from model.xRIR_cyl import build_xrir

    torch.manual_seed(0)
    return build_xrir("simple", 2).eval()


@pytest.fixture(scope="module")
def batch_of_three():
    return _collated_batch(3)


def test_evaluate_batch_drops_the_padded_rows_and_fills_every_metric_family(
        tiny_model, batch_of_three, evaluator):
    result = pilot.evaluate_batch(tiny_model, batch_of_three, [0, 128], evaluator,
                                  gl_seed=0, batch_size=4, device="cpu")
    assert result["n"] == 3
    assert result["keys"] == list(batch_of_three[-1])
    for name in pilot.GT_NAMES:
        assert result["gt"][name].shape == (3,)
    for k in (0, 128):
        cell = result["angles"][k]
        assert sorted(cell) == sorted(pilot.ANGLE_METRIC_NAMES)
        for name in pilot.ANGLE_METRIC_NAMES:
            assert cell[name].shape == (3,), name
        assert result["waveforms"][k].shape == (3, pilot.PADDED_LEN)
        assert result["waveforms"][k].dtype == np.float32


def test_evaluate_batch_reports_exactly_zero_shift_at_k_zero(
        tiny_model, batch_of_three, evaluator):
    result = pilot.evaluate_batch(tiny_model, batch_of_three, [0, 128], evaluator,
                                  gl_seed=0, batch_size=4, device="cpu")
    zero = result["angles"][0]
    for name in ("wave_rel_l2", "wave_mad", "logspec_mad", "mag_rel_l2"):
        np.testing.assert_array_equal(zero[name], np.zeros(3))
    for name in ("edt_gap", "c50_gap", "t60_gap", "t60_gap_abs"):
        values = zero[name]
        assert np.all(np.isnan(values) | (values == 0.0)), name
    assert np.abs(result["angles"][128]["logspec_mad"]).max() > 0


def test_evaluate_batch_keeps_the_real_rows_free_of_the_padding(
        tiny_model, batch_of_three, evaluator):
    alone = pilot.evaluate_batch(tiny_model, batch_of_three, [0, 128], evaluator,
                                 gl_seed=0, batch_size=None, device="cpu")
    padded = pilot.evaluate_batch(tiny_model, batch_of_three, [0, 128], evaluator,
                                  gl_seed=0, batch_size=4, device="cpu")
    # Not bit-identical: CPU GEMM kernels depend on the batch shape (measured 1.9e-8 on
    # the log-spectrogram), which is exactly why pad_batch pins a canonical shape.  What
    # the padding must not do is leak a padded row into a kept row.
    for name in ("wave_rel_l2", "wave_mad", "logspec_mad", "mag_rel_l2", "log_mse"):
        np.testing.assert_allclose(alone["angles"][128][name],
                                   padded["angles"][128][name], rtol=0, atol=1e-6)
    for name in ("edt_err", "c50_err", "edt_gap", "c50_gap"):
        np.testing.assert_allclose(alone["angles"][128][name],
                                   padded["angles"][128][name], rtol=0, atol=1e-3)
    # Griffin-Lim amplifies that 1.9e-8 into ~5e-5 on the waveform (the plan's R3 note).
    np.testing.assert_allclose(alone["waveforms"][128], padded["waveforms"][128],
                               rtol=0, atol=1e-3)


def test_evaluate_batch_controls_are_fresh_inferences_under_their_own_ids(
        tiny_model, batch_of_three, evaluator):
    counter = _CountingAlign(tiny_model)
    tiny_model.shift_and_align = counter
    try:
        plain = pilot.evaluate_batch(tiny_model, batch_of_three, [0, 128], evaluator,
                                     gl_seed=0, batch_size=4, device="cpu")
        without_controls = counter.calls
        # The explicit k = 0 alignment plus the plain k = 0 forward that computes the
        # paired reference -- exp_03's op sequence; every angle after that is pinned.
        assert without_controls == 2
        assert plain["controls"] == {}

        counter.calls = 0
        with_controls = pilot.evaluate_batch(tiny_model, batch_of_three, [0, 128],
                                             evaluator, gl_seed=0, batch_size=4,
                                             device="cpu", controls=True)
        control_calls = counter.calls
    finally:
        del tiny_model.shift_and_align

    assert control_calls == without_controls + len(pilot.CONTROL_SPECS)
    assert sorted(with_controls["controls"]) == sorted(spec[0] for spec in pilot.CONTROL_SPECS)
    for name, cell in with_controls["controls"].items():
        for metric in pilot.ANGLE_METRIC_NAMES:
            assert cell[metric].shape == (3,), (name, metric)
        assert cell["wave_max_abs_diff"].max() < 1e-6
        assert cell["logspec_max_abs_diff"].max() < 1e-6


def test_evaluate_batch_refuses_a_control_without_its_reference_angle(
        tiny_model, batch_of_three, evaluator):
    with pytest.raises(ValueError):
        pilot.evaluate_batch(tiny_model, batch_of_three, [0], evaluator, gl_seed=0,
                             batch_size=4, device="cpu", controls=True)
    with pytest.raises(ValueError):
        pilot.evaluate_batch(tiny_model, batch_of_three, [64, 128], evaluator, gl_seed=0,
                             batch_size=4, device="cpu")


# ------------------------------------------------------ test 8: select_probe_batches

class _FakeManifestDataset:
    """The only thing :func:`select_probe_batches` needs: canonical-order entries."""

    def __init__(self, sizes):
        self.entries = []
        for room, size in enumerate(sizes):
            for i in range(size):
                self.entries.append({
                    "index": len(self.entries),
                    "query": "Cat/Room_idx_{}/S{:03d}_R001_hybrid_IR.wav".format(room, i)})

    def __len__(self):
        return len(self.entries)


def test_select_probe_batches_returns_one_intact_batch_inside_every_room():
    dataset = _FakeManifestDataset([40, 35, 48])
    picked = pilot.select_probe_batches(dataset, batch_size=16)
    assert picked == [0, 3, 5]
    rooms = [entry["query"].rsplit("/", 1)[0] for entry in dataset.entries]
    for b in picked:
        window = rooms[16 * b:16 * b + 16]
        assert len(window) == 16 and len(set(window)) == 1


def test_select_probe_batches_can_take_several_batches_per_room():
    dataset = _FakeManifestDataset([64, 32])
    assert pilot.select_probe_batches(dataset, batch_size=16, per_room=2) == [0, 1, 4, 5]


def test_select_probe_batches_refuses_a_room_with_no_intact_batch():
    with pytest.raises(ValueError):
        pilot.select_probe_batches(_FakeManifestDataset([8, 40]), batch_size=16)
    with pytest.raises(ValueError):
        pilot.select_probe_batches(_FakeManifestDataset([64, 32]), batch_size=16,
                                   per_room=3)


def test_select_probe_batches_refuses_a_room_that_is_not_contiguous():
    dataset = _FakeManifestDataset([32, 32])
    dataset.entries[0], dataset.entries[40] = dataset.entries[40], dataset.entries[0]
    with pytest.raises(ValueError):
        pilot.select_probe_batches(dataset, batch_size=16)


@pytest.mark.skipif(not os.path.exists(REAL_MANIFEST),
                    reason="exp_03's pinned reference manifest is not available")
def test_select_probe_batches_reproduces_the_planned_probe_on_the_real_manifest():
    import types

    with open(REAL_MANIFEST) as fin:
        manifest = json.load(fin)
    dataset = types.SimpleNamespace(entries=manifest["entries"])
    picked = pilot.select_probe_batches(dataset, batch_size=16)
    assert picked == [0, 16, 32, 94, 110, 125, 141, 157, 214, 277, 292, 307, 322, 338,
                      353, 369, 381]
    assert len(picked) == 17


# -------------------------------------------- identity and serialisation helpers (A2)

def test_protocol_id_hashes_the_configuration_not_its_key_order():
    fields = {"checkpoint_sha256": "a" * 64, "manifest_hash": "b" * 64, "gl_seed": 0,
              "num_shot": 8, "device": "cpu", "batch_size": 16}
    digest = pilot.protocol_id(fields)
    assert len(digest) == 64 and int(digest, 16) >= 0
    assert digest == pilot.protocol_id(dict(reversed(list(fields.items()))))
    assert digest != pilot.protocol_id(dict(fields, gl_seed=1))
    assert digest != pilot.protocol_id(dict(fields, device="cuda"))


def test_new_execution_id_is_unique_and_carries_its_start_timestamp():
    first, started = pilot.new_execution_id()
    second, _ = pilot.new_execution_id()
    assert first != second
    assert started in first
    assert started.endswith("Z")


def test_query_list_sha256_pins_the_order_of_the_queries():
    a = pilot.query_list_sha256(["x/a.wav", "x/b.wav"])
    assert a != pilot.query_list_sha256(["x/b.wav", "x/a.wav"])
    assert a == pilot.query_list_sha256(["x/a.wav", "x/b.wav"])


def test_file_sha256_reads_the_bytes_on_disk(tmp_path):
    import hashlib

    path = tmp_path / "blob.bin"
    path.write_bytes(b"exp_10")
    assert pilot.file_sha256(str(path)) == hashlib.sha256(b"exp_10").hexdigest()


def test_write_json_is_strict_atomic_and_maps_nan_to_null(tmp_path):
    path = str(tmp_path / "per_sample.json")
    payload = {"values": pilot.json_values([1.0, float("nan"), float("inf"), 2.5])}
    pilot.write_json(payload, path)
    assert not os.path.exists(path + ".tmp")
    with open(path) as fin:
        text = fin.read()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"values": [1.0, None, None, 2.5]}

    with pytest.raises(ValueError):
        pilot.write_json({"bad": float("nan")}, str(tmp_path / "bad.json"))
    assert not os.path.exists(str(tmp_path / "bad.json"))


# ----------------------------------------------------------------- test 9: run smoke

MANIFEST_HASH = "47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d"
CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "xRIR_unseen.pth")
DATA_ROOT = os.environ.get("XRIR_DATA_PATH", os.path.join(REPO_ROOT, "data"))
HAVE_RUN_INPUTS = (os.path.exists(REAL_MANIFEST) and os.path.exists(CHECKPOINT) and
                   os.path.isdir(os.path.join(DATA_ROOT, "single_channel_ir")))
needs_run_inputs = pytest.mark.skipif(
    not HAVE_RUN_INPUTS, reason="the AcousticRooms cache, the manifest or the released "
                                "checkpoint is not available")


def _run_argv(out_dir, **overrides):
    argv = {"--arm": "smoke_test", "--backbone": "simple", "--checkpoint": CHECKPOINT,
            "--manifest": REAL_MANIFEST, "--manifest-hash": MANIFEST_HASH,
            "--num-shot": "8", "--ks": "0,128", "--batches": "0", "--device": "cpu",
            "--threads": "8", "--num-workers": "2", "--out-dir": out_dir}
    argv.update(overrides)
    return [token for pair in argv.items() for token in pair]


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory):
    """Two identically configured CPU runs of the first canonical batch."""
    if not HAVE_RUN_INPUTS:
        pytest.skip("the AcousticRooms cache, the manifest or the checkpoint is missing")
    directories = []
    for name in ("first", "second"):
        out_dir = str(tmp_path_factory.mktemp(name))
        pilot.main(_run_argv(out_dir))
        directories.append(out_dir)
    return directories


@needs_run_inputs
def test_run_writes_every_planned_output_with_a_complete_meta(smoke_runs):
    out_dir = smoke_runs[0]
    for name in ("per_sample.json", "metrics.json", "meta.json", "wav_k0.npy",
                 "wav_k128.npy"):
        assert os.path.exists(os.path.join(out_dir, name)), name

    with open(os.path.join(out_dir, "meta.json")) as fin:
        meta = json.load(fin)
    for field in ("arm", "backbone", "checkpoint", "checkpoint_sha256", "manifest_path",
                  "manifest_hash", "manifest_seed", "num_shot", "gl_seed", "ks",
                  "batches", "batches_arg", "batch_size", "batch_canonical", "device",
                  "tf32", "cudnn_deterministic", "n_queries", "native_len", "padded_len",
                  "sample_rate", "metric_window", "torch_version", "numpy_version",
                  "git_commit", "tool_sha256", "run_id", "protocol_id", "execution_id",
                  "started_at", "verify_exp03_pins", "arrays", "query_list_sha256",
                  "per_sample_sha256", "timing", "complete"):
        assert field in meta, field
    assert meta["complete"] is True
    assert meta["batch_canonical"] is True and meta["batch_size"] == 16
    assert meta["native_len"] == 9579 and meta["padded_len"] == 9600
    assert meta["metric_window"] == 8000 and meta["sample_rate"] == 22050
    assert meta["verify_exp03_pins"]["ok"] is True
    assert meta["n_queries"] == 16 and meta["batches"] == [0]
    assert set(meta["timing"]) >= {"inference_s", "inversion_s", "metrics_s",
                                   "writing_s", "loading_s", "total_min"}

    for name, entry in meta["arrays"].items():
        path = os.path.join(out_dir, name)
        assert pilot.file_sha256(path) == entry["sha256"]
        assert entry["shape"] == [16, 9600] and entry["dtype"] == "float32"
    assert meta["per_sample_sha256"] == pilot.file_sha256(
        os.path.join(out_dir, "per_sample.json"))


@needs_run_inputs
def test_run_ties_the_per_sample_file_to_its_meta(smoke_runs):
    out_dir = smoke_runs[0]
    with open(os.path.join(out_dir, "meta.json")) as fin:
        meta = json.load(fin)
    with open(os.path.join(out_dir, "per_sample.json")) as fin:
        per_sample = json.load(fin)

    assert per_sample["protocol_id"] == meta["protocol_id"]
    assert per_sample["execution_id"] == meta["execution_id"]
    for field, value in per_sample["meta"].items():
        assert meta[field] == value, field
    assert len(per_sample["query"]) == 16
    assert per_sample["index"] == list(range(16))
    assert pilot.query_list_sha256(per_sample["query"]) == meta["query_list_sha256"]
    assert sorted(per_sample["angles"]) == ["0", "128"]
    for name in pilot.GT_NAMES:
        assert len(per_sample["gt"][name]) == 16
    for k in ("0", "128"):
        for name in pilot.ANGLE_METRIC_NAMES:
            assert len(per_sample["angles"][k][name]) == 16, (k, name)
    for name in ("wave_rel_l2", "wave_mad", "logspec_mad", "mag_rel_l2"):
        assert per_sample["angles"]["0"][name] == [0.0] * 16
    assert max(abs(v) for v in per_sample["angles"]["128"]["logspec_mad"]) > 0


@needs_run_inputs
def test_run_is_reproducible_bit_for_bit_on_the_cpu(smoke_runs):
    first, second = smoke_runs
    for name in ("wav_k0.npy", "wav_k128.npy"):
        a = np.load(os.path.join(first, name))
        b = np.load(os.path.join(second, name))
        assert a.dtype == np.float32 and a.shape == (16, 9600)
        np.testing.assert_array_equal(a, b)          # save/load round trip is exact too

    payloads = []
    for out_dir in (first, second):
        with open(os.path.join(out_dir, "per_sample.json")) as fin:
            payload = json.load(fin)
        payload.pop("meta"), payload.pop("execution_id")
        payloads.append(payload)
    assert payloads[0] == payloads[1]

    metas = []
    for out_dir in (first, second):
        with open(os.path.join(out_dir, "meta.json")) as fin:
            metas.append(json.load(fin))
    assert metas[0]["protocol_id"] == metas[1]["protocol_id"]
    assert metas[0]["execution_id"] != metas[1]["execution_id"]
    assert (metas[0]["arrays"]["wav_k128.npy"]["sha256"] ==
            metas[1]["arrays"]["wav_k128.npy"]["sha256"])


@needs_run_inputs
def test_run_refuses_a_wrong_manifest_hash_or_shot_count(tmp_path):
    with pytest.raises(ValueError):
        pilot.main(_run_argv(str(tmp_path / "bad_hash"), **{"--manifest-hash": "0" * 64}))
    with pytest.raises(ValueError):
        pilot.main(_run_argv(str(tmp_path / "bad_shots"), **{"--num-shot": "4"}))


@needs_run_inputs
def test_run_refuses_an_out_of_range_batch_selection(tmp_path):
    with pytest.raises(ValueError):
        pilot.main(_run_argv(str(tmp_path / "bad_batch"), **{"--batches": "99999"}))
