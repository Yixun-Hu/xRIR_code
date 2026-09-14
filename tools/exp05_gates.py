"""Admission and cumulative time limits for exp_05 S/L training arms."""
import fcntl
import datetime
import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path

from tools import provenance as p
from tools.exp05_params import tier_of
from tools.exp05_probe import projection


def validate_receipt(path, commit, gpu, tier, backbone):
    from tools.exp04_launcher import arm_root
    def require(ok, cause):
        if not ok:
            raise ValueError(cause)
    context = 'validation'
    try:
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        require(data['reviewed_commit'] == commit, 'commit mismatch')
        require(tier in ('S', 'L') and backbone in ('simple', 'cylindrical')
                and data['tier'] == tier and data['backbone'] == backbone, 'tier/backbone mismatch')
        require(data['gpu'] == gpu, 'GPU mismatch')
        require(data['PROBE_NOT_CLEAN'] is False, 'NOT CLEAN')
        projected = projection(data)
        require(data['passed'] is True and projected['passed']
                and all(data[k] == projected[k] for k in ('T_epoch', 'T_run')), 'T_run or timing mismatch')
        context = 'probe-attempt binding'
        attempt = data['probe_attempt']
        if any(p.sha256_file(Path(attempt['path']) / (name + '.json')) != attempt[name + '_sha256']
               for name in ('train_manifest', 'completion')):
            raise ValueError('probe attempt changed')
        attempt_path = Path(attempt['path']).resolve()
        manifest = json.loads((attempt_path / 'train_manifest.json').read_bytes())
        completion = json.loads((attempt_path / 'completion.json').read_bytes())
        require(manifest['reviewed_commit'] == commit, 'commit mismatch')
        require(manifest['effective_args']['tier'] == tier and manifest['effective_args']['backbone'] == backbone,
                'tier/backbone mismatch')
        if (attempt_path.parent != arm_root(tier, backbone).resolve()
                or manifest['mode'] != 'probe'
                or Path(manifest['attempt_path']).resolve() != attempt_path
                or any(completion['metrics']['probe'][k] != data[k] for k in
                       ('t_micro', 't_test', 't_save', 'iteration_seconds', 'peak_allocated_bytes',
                        'peak_reserved_bytes', 'T_epoch', 'T_run'))):
            raise ValueError('probe attempt arm or measurements differ')
        context = 'validation'
        snapshots = [data['before'], *data['arms_before'], data['after']]
        require(all(s['gpu'] == gpu and s['uuid'] == data['before']['uuid'] for s in snapshots), 'GPU mismatch')
        require(len(data['arms_before']) == 1 and type(data['before']['uuid']) is str and bool(data['before']['uuid'])
                and all(s['compute_apps'] == '' and math.isfinite(s['free_gib']) and s['free_gib'] >= 40
                        for s in snapshots), 'NOT CLEAN')
        valid = (type(data['schema_version']) is int and data['schema_version'] == 1
            and data['test_timing_protocol'] == 'full_test_loader'
            and type(data['test_batches_total']) is int and data['test_batches_total'] > 0
            and type(data['test_batches_timed']) is int and data['test_batches_timed'] == data['test_batches_total']
            and all(type(data[k]) is int and data[k] > 0 for k in ('peak_allocated_bytes', 'peak_reserved_bytes'))
            and data['peak_reserved_bytes'] >= data['peak_allocated_bytes'] and data['yaw_aug'] == 0
            and math.isfinite(data['train_loss']) and data['iteration_seconds'] == data['t_micro']['values']
            and all(data[k + '_iteration_seconds'] == data['t_micro'][k] for k in ('mean', 'median', 'min')))
        require(valid, 'invalid measurements or schema')
        require(p.sha256_file(path) == digest, 'stale receipt: bytes changed during validation')
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('full requires a clean passing tier receipt ({}): {}'.format(context, error)) from error
    return dict(path=str(path), sha256=digest)


def timing_limits(fields, gpu):
    effective = fields['effective_args']
    tier = tier_of(effective)
    if tier == 'M':
        return None
    try:
        bound = fields['mutable_inputs']['probe_receipt']
        actual = validate_receipt(bound['path'], fields['reviewed_commit'], gpu, tier, effective['backbone'])
        raw = Path(actual['path']).read_bytes()
        if bound['sha256'] != actual['sha256'] or hashlib.sha256(raw).hexdigest() != actual['sha256']:
            raise ValueError('changed receipt')
        data = json.loads(raw)
        if 'resource_before' in fields and (fields['resource_before'].get('uuid') != data['before']['uuid']
                or fields['resource_before'].get('gpu') != gpu):
            raise ValueError('launch GPU differs from probe GPU')
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError('missing or changed tier probe receipt') from error
    return dict(epoch_seconds=1.05 * data['T_epoch'], projection_hours=data['T_run'] / 3600,
                ceiling_hours=1.5 * data['T_run'] / 3600, probe_receipt_sha256=actual['sha256'])


def set_budget(root, limits, renewal=None, commit=True):
    """Validate the arm ceiling; commit=False performs a read-only preflight."""
    from tools.exp04_launcher import hours_record
    root = Path(root)
    if any(not math.isfinite(limits[k]) or limits[k] <= 0 for k in ('ceiling_hours', 'projection_hours')):
        raise ValueError('invalid cumulative ceiling or projection')
    if commit:
        root.mkdir(parents=True, exist_ok=True)
    with (root / '.hours.lock').open('a') if commit else nullcontext() as lock:
        if commit:
            fcntl.flock(lock, fcntl.LOCK_EX)
        record = hours_record(root)
        for attempt in root.glob('attempt_*_ABORTED_slow*'):
            try:
                abort = json.loads((attempt / 'abort.json').read_text())
                fields = json.loads((attempt / 'train_manifest.json').read_text())
                digest = fields['mutable_inputs']['probe_receipt']['sha256']
                if abort['reason'] != 'guard_epoch_one' or digest == limits['probe_receipt_sha256']:
                    raise ValueError('slow abort requires a new probe receipt')
            except (OSError, KeyError, TypeError, ValueError) as error:
                raise ValueError('slow abort requires intact evidence and a new probe receipt') from error
        used = sum(r['hours'] for r in record['attempts'] if r['mode'] == 'full')
        ceiling = limits['ceiling_hours']
        old = record.get('probe_receipt_sha256')
        if old == limits['probe_receipt_sha256'] and record.get('probe_projection_hours') != limits['projection_hours']:
            raise ValueError('receipt projection disagrees with cumulative ceiling')
        if used or renewal is not None:
            if not old or not math.isfinite(record.get('ceiling_hours', 0)) or record.get('ceiling_hours', 0) <= 0:
                raise ValueError('missing or invalid cumulative ceiling')
            if renewal is None:
                ceiling = min(ceiling, record['ceiling_hours'])
        if renewal is not None:
            timestamp, separator, reason = renewal.partition(': ')
            if not separator or not reason.strip():
                raise ValueError('renewal requires notebook timestamp: reason')
            parsed = datetime.datetime.fromisoformat(timestamp)
            if parsed.tzinfo is None or timestamp != parsed.isoformat():
                raise ValueError('renewal requires ISO-8601 notebook timestamp')
            if limits['probe_receipt_sha256'] in [old, *record.get('probe_receipt_history', [])]:
                raise ValueError('renewal requires a new clean receipt')
            record.setdefault('renewals', []).append(dict(timestamp=timestamp, reason=reason.strip(),
                old_ceiling=record['ceiling_hours'], new_ceiling=ceiling,
                receipt_sha256=limits['probe_receipt_sha256']))
        if used + limits['projection_hours'] > ceiling:
            raise ValueError('cumulative hours + projection exceeds tier ceiling')
        if old and old != limits['probe_receipt_sha256']:
            record.setdefault('probe_receipt_history', []).append(old)
        record.update(ceiling_hours=ceiling, probe_receipt_sha256=limits['probe_receipt_sha256'],
                      probe_projection_hours=limits['projection_hours'])
        if commit:
            p.write_completion(root / 'cumulative_hours.json', record)
    return dict(limits, ceiling_hours=ceiling)
