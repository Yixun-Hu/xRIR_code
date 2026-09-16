"""Device-preserving re-implementation of the pinned alignment arithmetic (plan v4 6.1).

``model.xRIR.shift_and_align`` delegates the shift to ``model.xRIR.apply_delay``, which
allocates its output on the default GPU unconditionally. G1's legacy reproduction runs
on CPU, so exp_06 owns this copy of the same arithmetic -- the same integer delays, the
same direct-path gains, the same zero-padded shift semantics and clipping -- allocating
where the caller's tensors already are. The pinned method reads nothing from ``self``; the
``model`` argument is kept so the two are interchangeable at a call site (and so the GPU
parity test can hand the same object to both), and is only type-checked, never read.
"""
import torch

SAMPLE_RATE = 22050          # the pinned method's hard-coded constants, not the model's
SPEED_OF_SOUND = 343.0
EPS = 1e-7                   # the pinned denominator guard on the query distance


def _check(model, x, src_loc, ref_ir_locs):
    """Fail closed: an argument in the wrong position must never align anything."""
    if model is not None and not isinstance(model, torch.nn.Module):
        raise ValueError('model must be a torch.nn.Module or None')
    tensors = (x, src_loc, ref_ir_locs)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise ValueError('x, src_loc and ref_ir_locs must be tensors')
    if not all(t.is_floating_point() for t in tensors):
        raise ValueError('x, src_loc and ref_ir_locs must be floating point')
    if len({t.dtype for t in tensors}) != 1:
        raise ValueError('x, src_loc and ref_ir_locs must share one floating dtype')
    if len({t.device for t in tensors}) != 1:
        raise ValueError('x, src_loc and ref_ir_locs must share one device')
    if x.dim() != 3:
        raise ValueError('x must have shape [B, K, L], got {}'.format(tuple(x.shape)))
    batch, shots = x.shape[0], x.shape[1]
    if src_loc.shape != (batch, 3):
        raise ValueError('src_loc must have shape [B, 3], got {}'.format(tuple(src_loc.shape)))
    if ref_ir_locs.shape != (batch, shots, 3):
        raise ValueError('ref_ir_locs must have shape [B, K, 3], got {}'.format(
            tuple(ref_ir_locs.shape)))


def integer_delays_and_gains(src_loc, ref_ir_locs):
    """The pinned ``shift_and_align`` arithmetic: ``round((|s|-|r|)/c*fs)`` and ``|r|/(|s|+eps)``."""
    dist_src = torch.linalg.norm(src_loc, dim=1).unsqueeze(1)     # [B, 1]
    dist_ref = torch.linalg.norm(ref_ir_locs, dim=-1)             # [B, K]
    direct_energy_ratio = dist_ref / (dist_src + EPS)
    delay_unit = torch.round((dist_src - dist_ref) / SPEED_OF_SOUND * SAMPLE_RATE).int()
    return delay_unit, direct_energy_ratio


def apply_delay_device(signal, delay_tensor):
    """``model.xRIR.apply_delay`` with the output allocated where the signal already is.

    Positive delays shift right and zero-pad the front, negative delays shift left and
    zero-pad the tail, and a delay at least as long as the signal leaves it all zero --
    exactly the slicing of the pinned function, which clips by slicing to nothing.
    """
    delayed_signal = torch.zeros_like(signal)
    for i in range(signal.shape[0]):
        delay = int(delay_tensor[i].item())
        if delay > 0:
            delayed_signal[i, delay:] = signal[i, :-delay]
        elif delay < 0:
            delayed_signal[i, :delay] = signal[i, -delay:]
        else:
            delayed_signal[i] = signal[i]
    return delayed_signal


def shift_and_align_device(model, x, src_loc, ref_ir_locs):
    """``xRIR.shift_and_align`` on the caller's device: ``[B, K, L]`` in, ``[B, K, L]`` out."""
    _check(model, x, src_loc, ref_ir_locs)
    delay_unit, direct_energy_ratio = integer_delays_and_gains(src_loc, ref_ir_locs)
    channel_outputs = []
    for i in range(x.shape[1]):
        cur_delayed_x = apply_delay_device(x[:, i, :], delay_unit[:, i]).unsqueeze(1)
        channel_outputs.append(cur_delayed_x * direct_energy_ratio[:, i:(i + 1)].unsqueeze(2))
    return torch.cat(channel_outputs, dim=1)
