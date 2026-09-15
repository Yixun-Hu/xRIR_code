"""The probe's alignment: the pinned arithmetic, allocated where the caller already is."""
import numpy as np
import pytest
import torch

from model.xRIR_cyl_oriented import build_xrir_exp06
from tools import exp06_probe_align as subject

SAMPLE_RATE = 22050
SPEED_OF_SOUND = 343.0


def reference(x, src_loc, ref_locs):
    """Pure NumPy: integer delay, zero padding, direct-path gain -- written from the spec."""
    x = np.asarray(x, dtype=np.float64)
    src, refs = np.asarray(src_loc, np.float64), np.asarray(ref_locs, np.float64)
    batch, shots, length = x.shape
    out = np.zeros_like(x)
    for b in range(batch):
        d_src = float(np.sqrt((src[b] ** 2).sum()))
        for k in range(shots):
            d_ref = float(np.sqrt((refs[b, k] ** 2).sum()))
            delay = int(np.round((d_src - d_ref) / SPEED_OF_SOUND * SAMPLE_RATE))
            shifted = np.zeros(length)
            if delay > 0:
                shifted[delay:] = x[b, k, :length - delay]
            elif delay < 0:
                shifted[:length + delay] = x[b, k, -delay:]
            else:
                shifted[:] = x[b, k]
            out[b, k] = shifted * (d_ref / (d_src + 1e-7))
    return out


def geometry(batch, shots, generator, scale=3.0):
    src = torch.rand(batch, 3, generator=generator) * scale + 0.5
    refs = torch.rand(batch, shots, 3, generator=generator) * scale + 0.5
    return src, refs


@pytest.mark.parametrize('dtype,tol', [(torch.float32, 2e-6), (torch.float64, 1e-12)])
@pytest.mark.parametrize('batch,shots,length', [(1, 1, 64), (3, 8, 240), (2, 4, 31)])
def test_the_helper_reproduces_the_numpy_reference(dtype, tol, batch, shots, length):
    generator = torch.Generator().manual_seed(6 + batch * 10 + shots)
    src, refs = geometry(batch, shots, generator)
    x = torch.randn(batch, shots, length, generator=generator)
    out = subject.shift_and_align_device(None, x.to(dtype), src.to(dtype), refs.to(dtype))
    assert out.dtype is dtype and out.shape == (batch, shots, length)
    expected = reference(x.numpy(), src.numpy(), refs.numpy())
    assert np.allclose(out.double().numpy(), expected, rtol=tol, atol=tol)


@pytest.mark.parametrize('delay', [0, 1, 7, -1, -9, 240, -240, 400, -400])
def test_every_sign_of_delay_moves_the_impulse_and_zero_pads(delay):
    """One impulse in the middle: the shift, the clipping and the padding are exact."""
    length, position = 240, 120
    metres = -delay * SPEED_OF_SOUND / SAMPLE_RATE
    src = torch.tensor([[20.0, 0.0, 0.0]])
    refs = torch.tensor([[[20.0 + metres, 0.0, 0.0]]])
    x = torch.zeros(1, 1, length)
    x[0, 0, position] = 1.0
    out = subject.shift_and_align_device(None, x, src, refs)[0, 0]
    gain = float(refs.norm(dim=-1)[0, 0] / (src.norm(dim=1)[0] + 1e-7))
    landed = position + delay
    if 0 <= landed < length:
        assert out[landed].item() == pytest.approx(gain, rel=1e-6)
        assert torch.count_nonzero(out).item() == 1
    else:
        assert torch.count_nonzero(out).item() == 0
    assert torch.equal(out, subject.shift_and_align_device(None, x, src, refs)[0, 0])


def test_the_delay_is_the_pinned_rounded_sample_count():
    src = torch.tensor([[5.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    refs = torch.tensor([[[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                         [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    delays, gains = subject.integer_delays_and_gains(src, refs)
    assert delays.dtype is torch.int32
    assert delays.tolist() == [[int(round(3.0 / SPEED_OF_SOUND * SAMPLE_RATE)), 0],
                               [0, int(round(-2.0 / SPEED_OF_SOUND * SAMPLE_RATE))]]
    assert gains.shape == (2, 2) and gains[0, 0].item() == pytest.approx(2.0 / 5.0, rel=1e-6)


def test_the_output_is_allocated_on_the_inputs_device_and_nothing_is_mutated():
    generator = torch.Generator().manual_seed(11)
    src, refs = geometry(2, 3, generator)
    x = torch.randn(2, 3, 96, generator=generator)
    before = x.clone()
    out = subject.shift_and_align_device(None, x, src, refs)
    assert out.device == x.device and torch.equal(x, before)
    assert out.data_ptr() != x.data_ptr()


@pytest.mark.parametrize('case', ['x_dim', 'src_dim', 'ref_dim', 'shots', 'batch',
                                  'dtype', 'integer', 'model'])
def test_malformed_arguments_are_refused(case):
    src, refs = geometry(2, 3, torch.Generator().manual_seed(3))
    x = torch.randn(2, 3, 64)
    model = None
    if case == 'x_dim':
        x = x[0]
    elif case == 'src_dim':
        src = src[:, :2]
    elif case == 'ref_dim':
        refs = refs[..., :2]
    elif case == 'shots':
        refs = refs[:, :2]
    elif case == 'batch':
        src = src[:1]
    elif case == 'dtype':
        src = src.double()
    elif case == 'integer':
        x = x.long()
    else:
        model = 'not a module'
    with pytest.raises(ValueError):
        subject.shift_and_align_device(model, x, src, refs)


def test_the_helper_never_reaches_for_a_gpu():
    source = subject.__file__.replace('.pyc', '.py')
    with open(source) as handle:
        body = handle.read()
    assert '.cuda(' not in body and "'cuda'" not in body and '"cuda"' not in body


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
@torch.no_grad()
def test_cuda_parity_with_the_pinned_method():
    """Bit-exact with `xRIR.shift_and_align`, which only runs where `.cuda()` works."""
    model = build_xrir_exp06('simple', 8, dim=32, depth=1, heads=1, mlp_dim=32).cuda().eval()
    generator = torch.Generator().manual_seed(77)
    src, refs = geometry(3, 8, generator)
    x = torch.randn(3, 8, 9600, generator=generator)
    x, src, refs = x.cuda(), src.cuda(), refs.cuda()
    pinned = model.shift_and_align(x, src, refs)
    ours = subject.shift_and_align_device(model, x, src, refs)
    assert ours.device == pinned.device and torch.equal(ours, pinned)
