"""Training-only acoustic heading inference and heading-frame HAA geometry."""
import math
import numbers

import numpy as np


def _vector(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError('expected a nonempty finite vector')
    return value


def _paired(theta, values):
    theta, values = _vector(theta), _vector(values)
    if theta.shape != values.shape:
        raise ValueError('angles and values must have matching shapes')
    return theta, values


def early_level(rir, sr, window_s):
    """Energy in floor(window_s * sr) samples, from the first >20%-peak onset.

    Samples beyond the RIR are implicitly zero; a silent RIR has zero energy.
    """
    rir = _vector(rir)
    if not np.isfinite([sr, window_s]).all() or sr <= 0 or window_s <= 0:
        raise ValueError('sample rate and window must be positive and finite')
    n = int(window_s * sr)
    if n < 1:
        raise ValueError('window must contain at least one sample')
    onset = np.flatnonzero(np.abs(rir) > 0.2 * np.max(np.abs(rir)))
    return float(np.sum(rir[onset[0]:onset[0] + n] ** 2)) if onset.size else 0.0


def compensated_levels(rirs, dist, sr, window_s):
    """Return 10 log10(d^2 E); zero energy/distance cannot define a heading."""
    dist = _vector(dist)
    if len(rirs) != len(dist) or np.any(dist <= 0):
        raise ValueError('one positive horizontal distance is required per RIR')
    energy = np.array([early_level(rir, sr, window_s) for rir in rirs])
    if np.any(energy <= 0):
        raise ValueError('heading requires positive early energy')
    return 10 * np.log10(dist ** 2 * energy)


def _width(W):
    if isinstance(W, bool) or not isinstance(W, numbers.Integral) or W <= 0:
        raise ValueError('width must be a positive integer')


def heading_roll_k(phi_deg, W=512):
    """Round HALF UP in column units, preserving integer-shift cancellation."""
    _width(W)
    if not math.isfinite(phi_deg):
        raise ValueError('heading must be finite')
    return math.floor(-phi_deg * W / 360 + 0.5) % W


def canonical_heading_deg(k, W):
    """Heading represented exactly by a validated integer column roll."""
    _width(W)
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or not 0 <= k < W:
        raise ValueError('k must be an integer in [0, W)')
    return -int(k) * 360 / W


def continuous_fit(theta_deg, level):
    """Descriptive 0.5-degree grid least squares: level = a + b*cos(theta-phi), b>=0."""
    theta, level = _paired(theta_deg, level)
    grid = np.arange(-180.0, 180.0, 0.5)
    cosine = np.cos(np.deg2rad(theta[:, None] - grid[None]))
    centered = cosine - cosine.mean(axis=0)
    denom = np.sum(centered ** 2, axis=0)
    slope = np.divide(centered.T @ (level - level.mean()), denom,
                      out=np.zeros_like(denom), where=denom > 1e-12).clip(0)
    intercept = level.mean() - slope * cosine.mean(axis=0)
    mse = np.mean((level[:, None] - intercept - cosine * slope) ** 2, axis=0)
    best = int(np.argmin(mse))
    return dict(phi_deg=float(grid[best]), a=float(intercept[best]),
                b=float(slope[best]), mse=float(mse[best]))


def mean_direction(theta_deg, weights):
    """Weighted circular mean in degrees; None denotes an undefined resultant."""
    theta, weights = _paired(theta_deg, weights)
    if np.any(weights < 0):
        raise ValueError('weights must be nonnegative')
    resultant = np.sum(weights * np.exp(1j * np.deg2rad(theta)))
    if abs(resultant) <= 1e-12 * max(1, weights.sum()):
        return None
    return float(np.rad2deg(np.angle(resultant)))
