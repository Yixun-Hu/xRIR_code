"""Gate G1's mirror probe: the cohort rule, the recomposed forward and the decision."""
import numpy as np
import pytest
import torch

from model.xRIR_cyl_oriented import build_xrir_exp06
from tools import exp06_mirror_probe as subject
from tools.exp06_probe_align import shift_and_align_device

MAX_LEN = 1829          # 1829 // 31 + 1 = 60 frames, so the late C50 window is not empty
N_BINS = 60


@pytest.fixture(scope='module')
def tiny():
    """A two-shot xRIR small enough to run on a CPU, with the pinned pooling contract."""
    torch.manual_seed(6)
    return build_xrir_exp06('simple', 2, n_bins=N_BINS, dim=32, depth=1, heads=1,
                            mlp_dim=32).eval()


def batch_of(shots=2, size=2, seed=1):
    generator = torch.Generator().manual_seed(seed)
    depth = torch.randn(size, 3, 256, 512, generator=generator)
    src = torch.rand(size, 3, generator=generator) * 3 + 1
    refs = torch.rand(size, shots, 3, generator=generator) * 3 + 1
    ref_irs = torch.randn(size, shots, MAX_LEN, generator=generator) * 0.1
    target = torch.randn(size, 1, MAX_LEN, generator=generator) * 0.1
    return depth, ref_irs, src, refs, target


# --- the 2026-09-14 cohort rule -------------------------------------------------------


def test_the_cohort_rule_reproduces_a_hand_computed_selection():
    """Sort each side by room-frame y, then take evenly spaced ranks of that order."""
    y = np.array([3.0, -1.0, 1.0, -3.0, 2.0, -2.0], dtype=np.float32)
    test_indices = np.array([10, 11, 12, 13, 14, 15])
    cohort = subject.legacy_cohort(test_indices, y, n=3)
    assert cohort['+y']['positions'] == [2, 4, 0]         # y 1, 2, 3 -> positions 2, 4, 0
    assert cohort['+y']['mic_ids'] == [12, 14, 10]
    assert cohort['-y']['positions'] == [3, 5, 1]         # y -3, -2, -1
    assert cohort['-y']['mic_ids'] == [13, 15, 11]


def test_the_cohort_rule_spaces_ranks_over_the_whole_side():
    y = np.arange(1.0, 11.0, dtype=np.float32)
    cohort = subject.legacy_cohort(np.arange(100, 110), y, n=4)
    assert cohort['+y']['positions'] == [0, 3, 6, 9]
    assert cohort['-y'] == {'positions': [], 'mic_ids': []}


@pytest.mark.parametrize('case', ['length', 'dims', 'too_few', 'n', 'nonfinite'])
def test_the_cohort_rule_refuses_malformed_input(case):
    y = np.array([1.0, -1.0, 2.0, -2.0])
    ids = np.arange(4)
    kwargs = {'n': 2}
    if case == 'length':
        ids = np.arange(3)
    elif case == 'dims':
        y = y.reshape(2, 2)
        ids = ids.reshape(2, 2)
    elif case == 'too_few':
        kwargs['n'] = 3
    elif case == 'n':
        kwargs['n'] = 0
    else:
        y = np.array([1.0, np.nan, 2.0, -2.0])
    with pytest.raises(ValueError):
        subject.legacy_cohort(ids, y, **kwargs)


# --- the spectral C50 of the diagnostic -----------------------------------------------


def test_spectral_c50_is_the_early_to_late_energy_ratio():
    """Ten units of energy per early frame, one per late frame, from the onset on."""
    frames = 200
    magnitude = torch.zeros(2, 63, frames)
    onset = torch.tensor([4, 10])
    n50 = subject.C50_FRAMES
    for b in range(2):
        start = int(onset[b])
        magnitude[b, 0, start:start + n50] = np.sqrt(10.0)
        magnitude[b, 0, start + n50:] = 1.0
    late = [frames - int(onset[b]) - n50 for b in range(2)]
    expected = [10 * np.log10(10.0 * n50 / (late[b] + 1e-12)) for b in range(2)]
    got = subject.spectral_c50(magnitude, onset)
    assert got.shape == (2,)
    assert np.allclose(got.numpy(), expected, rtol=1e-5)
    assert subject.C50_FRAMES == round(0.05 * 22050 / 31)


def test_spectral_c50_refuses_a_bad_shape_or_onset():
    with pytest.raises(ValueError):
        subject.spectral_c50(torch.zeros(2, 63), torch.tensor([0, 0]))
    with pytest.raises(ValueError):
        subject.spectral_c50(torch.zeros(2, 63, 100), torch.tensor([0]))
    with pytest.raises(ValueError):
        subject.spectral_c50(torch.zeros(2, 63, 100), torch.tensor([0, -1]))


def test_onset_frames_are_the_direct_path_in_hops():
    src = torch.tensor([[3.43, 0.0, 0.0], [0.0, 6.86, 0.0]])
    assert subject.onset_frames(src).tolist() == [
        round(3.43 / 343.0 * 22050 / 31), round(6.86 / 343.0 * 22050 / 31)]


# --- the geometry branch and the recomposed forward -----------------------------------


@torch.no_grad()
def test_geometry_feature_is_the_pinned_pooled_source_branch(tiny):
    depth = torch.randn(3, 256, 512, generator=torch.Generator().manual_seed(4))
    positions = torch.tensor([[1.0, 2.0, 0.5], [-1.0, -2.0, 0.5]])
    feature = subject.geometry_feature(tiny, positions, depth)
    encoded = (positions[:, :, None, None] - depth[None]) / 5.
    expected = tiny.lin_proj_0(tiny.src_proj(tiny.source_network(encoded)).permute(0, 2, 1))
    assert feature.shape == (2, 256)
    assert torch.equal(feature, expected.squeeze(-1))


@torch.no_grad()
def test_the_recomposed_forward_mixes_the_reference_log_spectrograms(tiny):
    depth, ref_irs, src, refs, target = batch_of()
    out_log, tgt, weights, source_out = subject.composed_forward(
        tiny, depth, ref_irs, src, refs, target)
    assert out_log.shape == (2, 63, N_BINS) and tgt.shape == (2, 63, N_BINS)
    assert weights.shape == (2, 2, N_BINS) and source_out.shape == (2, 256)
    aligned = shift_and_align_device(tiny, ref_irs, src, refs)
    logs = torch.cat([torch.log(tiny.convert_ir_to_spec(aligned[:, i:i + 1]) + 1e-8)
                      for i in range(refs.shape[1])], dim=1)
    assert torch.equal(out_log, torch.sum(logs * weights.unsqueeze(2), dim=1))
    assert torch.equal(tgt, tiny.convert_ir_to_spec(target)[:, 0])


@torch.no_grad()
def test_the_recomposed_forward_puts_the_time_grid_on_the_inputs_device(tiny):
    """The pinned forward transfers `model.times`; so does this one (plan v4 R3.2)."""
    depth, ref_irs, src, refs, target = batch_of()
    seen = {}
    original = tiny.time_embedder.forward

    def spy(x):
        seen['device'] = x.device
        return original(x)

    tiny.time_embedder.forward = spy
    try:
        subject.composed_forward(tiny, depth, ref_irs, src, refs, target)
    finally:
        tiny.time_embedder.forward = original
    assert seen['device'] == refs.device


@torch.no_grad()
def test_the_recomposed_forward_uses_the_injected_alignment(tiny):
    depth, ref_irs, src, refs, target = batch_of()
    calls = []

    def align(model, x, src_loc, ref_locs):
        calls.append(model)
        return shift_and_align_device(model, x, src_loc, ref_locs)

    subject.composed_forward(tiny, depth, ref_irs, src, refs, target, align=align)
    assert calls == [tiny]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
@torch.no_grad()
def test_cuda_parity_with_the_pinned_forward():
    model = build_xrir_exp06('simple', 2, n_bins=310, dim=32, depth=1, heads=1,
                             mlp_dim=32).cuda().eval()
    generator = torch.Generator().manual_seed(9)
    depth = torch.randn(2, 3, 256, 512, generator=generator).cuda()
    src = (torch.rand(2, 3, generator=generator) * 3 + 1).cuda()
    refs = (torch.rand(2, 2, 3, generator=generator) * 3 + 1).cuda()
    ref_irs = (torch.randn(2, 2, 9600, generator=generator) * 0.1).cuda()
    target = (torch.randn(2, 1, 9600, generator=generator) * 0.1).cuda()
    out_log, tgt, _, _ = subject.composed_forward(model, depth, ref_irs, src, refs, target)
    pinned_log, pinned_tgt = model(depth, ref_irs, src, refs, target)
    assert torch.allclose(out_log.unsqueeze(-1), pinned_log, atol=1e-5, rtol=1e-5)
    assert torch.allclose(tgt, pinned_tgt[:, 0], atol=1e-5, rtol=1e-5)


# --- the decision ---------------------------------------------------------------------


def stats(cyl_cos=0.97, cyl_share=0.85, control_cos=0.44, control_share=0.24,
          cylor_cos=0.5, cylor_share=0.3):
    def cell(cosine, share):
        return {'mirror_cosine': cosine, 'opposite_side_weight_share': share}
    return {'cyl': cell(cyl_cos, cyl_share), 'control': cell(control_cos, control_share),
            'cyl_or': cell(cylor_cos, cylor_share)}


def test_the_gate_passes_when_the_anchors_bracket_and_the_oriented_model_is_below():
    decision = subject.g1_decision(stats())
    assert decision['outcome'] == 'pass' and decision['reasons'] == []
    assert decision['thresholds'] == subject.G1_THRESHOLDS


@pytest.mark.parametrize('override', [{'cylor_cos': 0.85}, {'cylor_share': 0.6},
                                      {'cylor_cos': 0.80}, {'cylor_share': 0.50}])
def test_the_gate_fails_when_the_oriented_model_still_mirrors(override):
    decision = subject.g1_decision(stats(**override))
    assert decision['outcome'] == 'fail' and decision['reasons']


@pytest.mark.parametrize('override', [{'cyl_cos': 0.80}, {'cyl_cos': 0.90},
                                      {'control_cos': 0.70}, {'control_cos': 0.60},
                                      {'cyl_share': 0.65}, {'cyl_share': 0.70},
                                      {'control_share': 0.40}, {'control_share': 0.35}])
def test_the_gate_is_inconclusive_when_the_anchors_do_not_bracket(override):
    decision = subject.g1_decision(stats(**override))
    assert decision['outcome'] == 'inconclusive' and decision['reasons']


def test_the_gate_refuses_incomplete_statistics():
    incomplete = stats()
    del incomplete['cyl_or']
    with pytest.raises(ValueError):
        subject.g1_decision(incomplete)
    with pytest.raises(ValueError):
        subject.g1_decision(stats(cyl_cos=float('nan')))
