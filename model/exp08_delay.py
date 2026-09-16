"""Device-preserving ``apply_delay`` so exp_08 can be validated on CPU.

``model.xRIR.apply_delay`` allocates its output with ``torch.zeros_like(signal).cuda()``.
On a GPU that ``.cuda()`` is a no-op (``signal`` is already there); with
``CUDA_VISIBLE_DEVICES=""`` it raises, so the whole ``shift_and_align`` path -- the one the
handoff insists must be exercised end to end (section 3.4: "must not rely on a fixed
unrotated alignment cache") -- would be untestable on CPU.

``model/xRIR.py`` is **not** edited (it is shared with the pinned ``simple`` /
``cylindrical`` entries).  Instead this module provides the same function with the
``.cuda()`` removed, plus :func:`device_preserving_delay`, a context manager that swaps it
into ``model.xRIR`` for the duration of a block.  ``shift_and_align`` resolves
``apply_delay`` from the module globals at call time, so the swap reaches the pinned models
as well as the exp_08 ones.

``tests/test_exp08_invariant.py`` proves the two are equivalent -- identical delays, gains
and zero padding -- by monkeypatching ``torch.Tensor.cuda`` to the identity and comparing
the pinned function's output elementwise with this one's.
"""
from __future__ import annotations

import contextlib

import torch

import model.xRIR as _xrir


def apply_delay_device_preserving(signal, delay_tensor):
    """``model.xRIR.apply_delay`` with the hard-coded ``.cuda()`` dropped.

    Line for line the pinned body otherwise: integer shift per batch row, zero padding on
    the vacated side, ``delay > 0`` shifts right, ``delay < 0`` shifts left, ``0`` copies.
    The output lands on ``signal``'s device instead of on cuda:0.

    Args:
        signal: ``[B, L]`` (or ``[B, L, C]``) reference audio.
        delay_tensor: ``[B]`` integer delays, one per row.

    Returns:
        The delayed signal, same shape/dtype/device as ``signal``.
    """
    batch_size, sequence_length = signal.shape[:2]

    # Output tensor initialized with zeros (same shape *and device* as the input signal)
    delayed_signal = torch.zeros_like(signal)

    for i in range(batch_size):
        delay = delay_tensor[i].item()  # Get the delay value for this batch item

        if delay > 0:
            # Positive delay: shift right and pad with zeros at the beginning
            delayed_signal[i, delay:] = signal[i, :-delay]
        elif delay < 0:
            # Negative delay: shift left and pad with zeros at the end
            delayed_signal[i, :delay] = signal[i, -delay:]
        else:
            # No delay, just copy the signal
            delayed_signal[i] = signal[i]

    return delayed_signal


@contextlib.contextmanager
def device_preserving_delay():
    """Swap :func:`apply_delay_device_preserving` into ``model.xRIR`` for the block.

    Mutates the ``model.xRIR`` module global, so it is for **serial, eager** use only (the
    same caveat ``tools.yaw_rotation.fixed_alignment`` carries) and is not thread-safe.  The
    original function is restored on exit, including when the block raises.

    Yields:
        The installed function, so a caller can assert on identity.
    """
    original = _xrir.apply_delay
    _xrir.apply_delay = apply_delay_device_preserving
    try:
        yield apply_delay_device_preserving
    finally:
        _xrir.apply_delay = original
