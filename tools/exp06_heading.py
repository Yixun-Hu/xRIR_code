"""Training-only acoustic heading inference and heading-frame HAA geometry.

The CLI exits 0 when a heading is estimated or overridden, 3 when the estimator
refuses (evidence JSON written), and 2 for argparse or input errors (nothing written).
A record is `confirmatory` only when its source closure matches HEAD and the tree is
clean outside worklog/; any other record is `diagnostic` and never binds a run.
"""
import math
import numbers
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from sim_to_real.haa_dataset import DEFAULT_ROOT, HAADataset
from tools.yaw_rotation import rotate_scene_yaw
from tools.provenance import closure_record, git_state, sha256_file, source_closure


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

GIT_FIELDS = ('HEAD', 'dirty', 'dirty_outside_worklog', 'diff_sha256')


def _is_hex(value, length):
    return (isinstance(value, str) and len(value) == length
            and all(c in '0123456789abcdef' for c in value))


def _is_sha256(value):
    return _is_hex(value, 64)


def _closure_digest(files, key):
    """Digest the closure as [[path, hash], ...], as exp_03's closure records do."""
    return hashlib.sha256(json.dumps([[f['path'], f[key]] for f in files],
                                     sort_keys=True).encode()).hexdigest()


def _admissibility(files, git):
    """Only a clean tree whose closure equals its HEAD blobs can bind a confirmatory run."""
    return ('confirmatory' if not git['dirty_outside_worklog']
            and all(f['head_blob_sha256'] == f['sha256'] for f in files) else 'diagnostic')


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


def estimate_room_heading(room_dir, override_deg=None, override_reason=None):
    """Infer an acoustic axis from exactly the 12 training RIRs, retaining audit evidence.

    Full input files are streamed only for SHA256 identity. Signal calculations
    select training rows from a read-only mmap. The source digest binds current
    file bytes, including uncommitted code, using provenance's closure discovery.
    """
    if override_deg is None and override_reason is not None:
        raise ValueError('override reason requires an override heading')
    if override_deg is not None:
        heading_roll_k(override_deg)
        if not isinstance(override_reason, str) or not override_reason.strip():
            raise ValueError('override heading requires a nonblank reason')
    room = Path(room_dir)
    meta = json.loads((room / 'meta.json').read_text())
    train = meta['train']
    if (len(train) != 12 or len(set(train)) != 12 or
            any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in train)):
        raise ValueError('expected 12 distinct nonnegative training indices')
    train = np.array(train, dtype=np.int64)
    xyz = np.load(room / 'xyzs.npy')[train] - np.load(room / 'speaker_xyz.npy')
    theta = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0]))
    dist = np.linalg.norm(xyz[:, :2], axis=1)
    rirs = np.load(room / 'rirs.npy', mmap_mode='r')[train]
    levels = {name: compensated_levels(rirs, dist, meta['sr'], duration)
              for name, duration in [('5ms', .005), ('50ms', .05)]}
    winner, table, reason = decide_heading(theta, levels)
    loo = _leave_one_out_winners(theta, levels)
    stable = winner is not None and all(value == winner for value in loo)
    accepted = winner is not None and stable
    if winner is not None and not stable:
        reason = 'leave-one-out decision is unstable'
    phi = CANDIDATES[winner] if accepted else None
    decision = 'estimated' if accepted else 'refused'
    if override_deg is not None:
        phi, decision = override_deg, 'override'
    descriptive = {}
    for window, level in levels.items():
        fits = [continuous_fit(np.delete(theta, i), np.delete(level, i))['phi_deg']
                for i in range(len(train))]
        descriptive[window] = dict(continuous_fit=continuous_fit(theta, level),
            loo_phi_deg=fits, loo_range_deg=[min(fits), max(fits)],
            mean_direction_deg=mean_direction(theta, 10 ** ((level - level.max()) / 10)))
    repo = Path(__file__).resolve().parents[1]
    state = git_state(repo)
    records, head_digest = closure_record(source_closure('tools.exp06_heading', repo),
                                          state['HEAD'], repo)
    files = [dict(path=r['path'], sha256=r['working_tree_sha256'],
                  head_blob_sha256=r['reviewed_blob_sha256']) for r in records]
    digest = _closure_digest(files, 'sha256')
    git = {key: state[key] for key in GIT_FIELDS}
    return dict(schema_version=1, room=room.name, phi_deg=phi,
        admissibility=_admissibility(files, git),
        k=heading_roll_k(phi) if phi is not None else None, decision=decision,
        reason=override_reason if decision == 'override' else reason,
        override_deg=override_deg, override_reason=override_reason,
        estimator=dict(winner=winner, table=table, reason=reason,
                       leave_one_out_winners=loo, leave_one_out_stable=stable),
        descriptive=descriptive,
        per_mic=[dict(index=int(idx), theta_deg=float(theta[i]), distance_m=float(dist[i]),
                      levels_db={w: float(level[i]) for w, level in levels.items()})
                 for i, idx in enumerate(train)],
        input_sha256={name: sha256_file(room / name) for name in
                      ['meta.json', 'xyzs.npy', 'speaker_xyz.npy', 'rirs.npy']},
        source_closure=dict(entry_module='tools.exp06_heading', repo=str(repo), files=files,
                            sha256=digest, head_sha256=head_digest, git=git, basis='working_tree'),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())


def _validate_heading_record(record):
    """Validate schema and decision evidence; refused records have no usable rotation."""
    required = {'schema_version', 'room', 'phi_deg', 'k', 'decision', 'reason', 'override_deg',
                'override_reason', 'estimator', 'descriptive', 'per_mic', 'input_sha256',
                'source_closure', 'timestamp', 'admissibility'}
    try:
        if not isinstance(record, dict) or not required <= record.keys():
            raise ValueError('missing heading fields')
        json.dumps(record, allow_nan=False)
        if type(record['schema_version']) is not int or record['schema_version'] != 1:
            raise ValueError('unsupported heading schema')
        decision, phi, k = record['decision'], record['phi_deg'], record['k']
        if decision not in {'estimated', 'override', 'refused'}:
            raise ValueError('invalid heading decision')
        if decision == 'refused':
            if phi is not None or k is not None:
                raise ValueError('refused records cannot specify a rotation')
        else:
            canonical_heading_deg(k, 512)
            if type(phi) not in (int, float) or k != heading_roll_k(phi):
                raise ValueError('heading and roll disagree')
        if decision == 'override':
            if (phi != record['override_deg'] or not isinstance(record['override_reason'], str)
                    or not record['override_reason'].strip()):
                raise ValueError('override heading and reason required')
        elif record['override_deg'] is not None or record['override_reason'] is not None:
            raise ValueError('unexpected override fields')
        if not all(isinstance(record[key], str) and record[key] for key in ['room', 'reason']):
            raise ValueError('room and reason required')
        if datetime.datetime.fromisoformat(record['timestamp']).tzinfo is None:
            raise ValueError('timestamp must include its timezone')
        inputs = record['input_sha256']
        if set(inputs) != {'meta.json', 'rirs.npy', 'xyzs.npy', 'speaker_xyz.npy'}:
            raise ValueError('four input hashes required')
        source = record['source_closure']
        if (source['entry_module'] != 'tools.exp06_heading' or source['basis'] != 'working_tree'
                or not Path(source['repo']).is_absolute() or not source['files']):
            raise ValueError('invalid source closure')
        pairs = [[f['path'], f['sha256']] for f in source['files']]
        hashes = (list(inputs.values()) + [source['sha256'], source['head_sha256']]
                  + [h for _, h in pairs]
                  + [f['head_blob_sha256'] for f in source['files'] if f['head_blob_sha256'] is not None])
        if any(not _is_sha256(h) for h in hashes):
            raise ValueError('invalid SHA256')
        if source['sha256'] != _closure_digest(source['files'], 'sha256'):
            raise ValueError('source closure digest mismatch')
        git = source['git']
        if (set(git) != set(GIT_FIELDS) or not _is_hex(git['HEAD'], 40)
                or any(type(git[key]) is not bool for key in ('dirty', 'dirty_outside_worklog'))
                or (git['diff_sha256'] is not None) != git['dirty']
                or (git['dirty'] and not _is_sha256(git['diff_sha256']))):
            raise ValueError('invalid git identity')
        if source['head_sha256'] != _closure_digest(source['files'], 'head_blob_sha256'):
            raise ValueError('HEAD closure digest mismatch')
        if record['admissibility'] != _admissibility(source['files'], git):
            raise ValueError('admissibility disagrees with the recorded git identity')
        mics = record['per_mic']
        if (len(mics) != 12 or len({m['index'] for m in mics}) != 12
                or any(type(m['index']) is not int or m['index'] < 0 or m['distance_m'] <= 0 for m in mics)):
            raise ValueError('12 distinct training microphone records required')
        theta = [m['theta_deg'] for m in mics]
        levels = {w: [m['levels_db'][w] for m in mics] for w in ['5ms', '50ms']}
        winner, table, reason = decide_heading(theta, levels)
        loo = _leave_one_out_winners(theta, levels)
        stable = winner is not None and all(w == winner for w in loo)
        if winner is not None and not stable:
            reason = 'leave-one-out decision is unstable'
        expected = dict(winner=winner, table=table, reason=reason,
                        leave_one_out_winners=loo, leave_one_out_stable=stable)
        if record['estimator'] != expected:
            raise ValueError('estimator evidence mismatch')
        if decision != 'override':
            if (decision == 'estimated') != stable or (stable and phi != CANDIDATES[winner]):
                raise ValueError('decision disagrees with estimator')
        if set(record['descriptive']) != set(levels):
            raise ValueError('descriptive results required for both windows')
        for result in record['descriptive'].values():
            fit = result['continuous_fit']
            values = ([fit[key] for key in ['phi_deg', 'a', 'b', 'mse']]
                      + result['loo_phi_deg'] + result['loo_range_deg'])
            if result['mean_direction_deg'] is not None:
                values.append(result['mean_direction_deg'])
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
                raise ValueError('descriptive fields must be finite numbers')
            if (not {'phi_deg', 'a', 'b', 'mse'} <= fit.keys() or fit['b'] < 0 or fit['mse'] < 0
                    or len(result['loo_phi_deg']) != 12 or len(result['loo_range_deg']) != 2
                    or 'mean_direction_deg' not in result):
                raise ValueError('invalid descriptive fit')
    except (KeyError, TypeError, OverflowError, AttributeError) as error:
        raise ValueError('malformed heading record') from error
    return record


def write_heading_json(path, record):
    """Write a validated, finite heading record (including explicit refusal records)."""
    _validate_heading_record(record)
    Path(path).write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + '\n')


def verify_heading_inputs(record, room_dir):
    """Refuse a validated record whose recorded cache inputs no longer hash the same."""
    _validate_heading_record(record)
    for name, digest in sorted(record['input_sha256'].items()):
        try:
            actual = sha256_file(Path(room_dir) / name)
        except OSError as error:
            raise ValueError('unreadable heading input: ' + name) from error
        if actual != digest:
            raise ValueError('heading input changed since estimation: ' + name)
    return record


def read_heading_json(path, room_dir=None):
    """Read and validate a record; with a room directory, re-verify its input hashes."""
    record = _validate_heading_record(json.loads(Path(path).read_text()))
    return record if room_dir is None else verify_heading_inputs(record, room_dir)


def main(argv=None):
    """Write the inferred acoustic axis; a refusal writes evidence and exits 3, input errors 2."""
    parser = argparse.ArgumentParser(description='Infer an acoustic axis from HAA training RIRs.')
    parser.add_argument('--room-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--heading-deg', type=float)
    parser.add_argument('--override-reason')
    args = parser.parse_args(argv)
    try:
        record = estimate_room_heading(args.room_dir, args.heading_deg, args.override_reason)
        write_heading_json(args.out, record)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({key: record[key] for key in ['room', 'decision', 'phi_deg', 'k', 'reason']}))
    return 3 if record['decision'] == 'refused' else 0


if __name__ == '__main__':
    raise SystemExit(main())
