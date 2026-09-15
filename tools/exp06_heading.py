"""Training-only acoustic heading inference and heading-frame HAA geometry."""
import math
import numbers

import numpy as np

from sim_to_real.haa_dataset import DEFAULT_ROOT, HAADataset
from tools.yaw_rotation import rotate_scene_yaw


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


CANDIDATES = {'+x': 0, '+y': 90, '-x': 180, '-y': -90}


def candidate_contrasts(theta_deg, levels, candidates=CANDIDATES,
                        half_width_deg=45, min_in=3, min_out=3):
    """Mean inside minus outside each inclusive circular sector; None if undersampled."""
    theta, levels = _paired(theta_deg, levels)
    result = {}
    for name, phi in candidates.items():
        inside = np.abs((theta - phi + 180) % 360 - 180) <= half_width_deg
        result[name] = (float(levels[inside].mean() - levels[~inside].mean())
                        if inside.sum() >= min_in and (~inside).sum() >= min_out else None)
    return result


def decide_heading(theta_deg, levels_by_window):
    """Apply both-window agreement, 3 dB contrast, and 3 dB competitor margins.

    Leave-one-out stability is a separate gate, applied by estimate_room_heading.
    The table retains raw winners even when a decision threshold refuses them.
    """
    if len(levels_by_window) != 2:
        raise ValueError('exactly two energy windows are required')
    table = {}
    for window, levels in levels_by_window.items():
        contrasts = candidate_contrasts(theta_deg, levels)
        ranked = sorted(((v, k) for k, v in contrasts.items() if v is not None), reverse=True)
        table[window] = dict(contrasts_db=contrasts, winner=ranked[0][1] if ranked else None,
                             competitor_count=max(0, len(ranked) - 1),
                             runner_up_margin_db=ranked[0][0] - ranked[1][0] if len(ranked) > 1 else None)
    rows = list(table.values())
    winners = [row['winner'] for row in rows]
    if None in winners:
        return None, table, 'insufficient microphones inside/outside candidate sectors'
    if winners[0] != winners[1]:
        return None, table, 'windows disagree'
    winner = winners[0]
    if any(row['contrasts_db'][winner] < 3 for row in rows):
        return None, table, 'contrast below 3 dB'
    if any(row['runner_up_margin_db'] is not None and row['runner_up_margin_db'] < 3 for row in rows):
        return None, table, 'runner-up margin below 3 dB'
    return winner, table, 'accepted'


def _leave_one_out_winners(theta_deg, levels_by_window):
    return [decide_heading(np.delete(theta_deg, i),
            {w: np.delete(level, i) for w, level in levels_by_window.items()})[0]
            for i in range(len(theta_deg))]


def leave_one_out_stable(theta_deg, levels_by_window):
    """Require every refit to pass all decision thresholds with the original winner."""
    winner = decide_heading(theta_deg, levels_by_window)[0]
    if winner is None or len(theta_deg) < 2:
        return False
    return all(refit == winner for refit in _leave_one_out_winners(theta_deg, levels_by_window))


class HeadingFrameDataset(HAADataset):
    """Rotate geometry after the pinned dataset's draw; audio and room data stay intact.

    The parent retains its eager RIR loading behavior. Training-only mmap access
    belongs to the estimator, not to this compatibility wrapper.
    """

    def __init__(self, rooms, split, root=DEFAULT_ROOT, num_shot=8, max_len=9600,
                 eval_seed=None, depth_variant='default', *, k_by_room):
        rooms = list(rooms)
        if any(room not in k_by_room for room in rooms):
            raise ValueError('a heading roll is required for every room')
        self.k_by_room = dict(k_by_room)
        for room in rooms:
            canonical_heading_deg(self.k_by_room[room], 512)
        super().__init__(rooms, split, root=root, num_shot=num_shot, max_len=max_len,
                         eval_seed=eval_seed, depth_variant=depth_variant)

    def __getitem__(self, i):
        item = super().__getitem__(i)
        k = self.k_by_room[self.items[i][0]]
        if k == 0:
            return item
        listener, src, depth, target, audio, refs = item
        depth, src, refs = rotate_scene_yaw(depth.unsqueeze(0), src.unsqueeze(0), refs.unsqueeze(0), k)
        return listener, src.squeeze(0), depth.squeeze(0), target, audio, refs.squeeze(0)

    def side_label(self, room, idx):
        """Original room-frame sign of microphone y relative to the speaker."""
        return int(self.data[room]['src_local'][idx, 1].sign().item())
