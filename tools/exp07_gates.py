"""Admission and cumulative time limits for the exp_07 seen training arms.

Amendment A1: ``tools/exp05_gates.py`` keeps main's bytes.  ``set_budget`` here is the
reviewed exp_05 ledger with one added rule -- a seen arm gets at most one retry, and
neither a new receipt nor a ``--renew-ceiling`` renewal resets that count -- and
``timing_limits`` hands an unseen (tier S/L or M) field set straight back to exp_05.
``validate_receipt`` cannot be wrapped: the exp_05 function pins ``tier in ('S', 'L')``
and ``yaw_aug == 0``, so the seen identity checks are stated here, against
``exp07_probe.projection`` and the exp_07 arm roots.
"""
import hashlib
import json
import math
from pathlib import Path

from tools import exp05_gates as tier_gates
from tools import provenance as p
from tools.exp05_params import tier_of
from tools.exp07_probe import projection

PROTOCOL = 'seen'
MAX_FULL_ATTEMPTS = 2  # one launch and at most one retry


def validate_receipt(path, commit, gpu, tier, backbone, protocol=PROTOCOL, yaw_aug=0):
    """The seen counterpart of exp05_gates.validate_receipt, bound to its probe attempt."""
    from tools.exp07_launcher import arm_root

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
        require(tier == 'M' and protocol == PROTOCOL and yaw_aug in (0, 1)
                and backbone in ('simple', 'cylindrical') and data['tier'] == tier
                and data['backbone'] == backbone and data['protocol'] == protocol,
                'tier/backbone/protocol mismatch')
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
        effective = manifest['effective_args']
        require(effective['tier'] == tier and effective['backbone'] == backbone
                and effective['protocol'] == protocol
                and effective.get('yaw_aug', 0) == yaw_aug, 'tier/backbone/protocol mismatch')
        measured = completion['metrics']['probe']
        # The receipt's own identity is required to equal the requested arm above, so tying
        # the completion to the request also ties the completion to the receipt.
        requested = dict(tier=tier, backbone=backbone, protocol=protocol, yaw_aug=yaw_aug)
        if (attempt_path.parent != arm_root(backbone, yaw_aug).resolve()
                or manifest['mode'] != 'probe'
                or Path(manifest['attempt_path']).resolve() != attempt_path
                or completion['train_manifest_sha256'] != attempt['train_manifest_sha256']
                or any(measured.get(key) != value for key, value in requested.items())
                or any(measured[k] != data[k] for k in
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
            and data['peak_reserved_bytes'] >= data['peak_allocated_bytes'] and data['yaw_aug'] == yaw_aug
            and math.isfinite(data['train_loss']) and data['iteration_seconds'] == data['t_micro']['values']
            and all(data[k + '_iteration_seconds'] == data['t_micro'][k] for k in ('mean', 'median', 'min')))
        require(valid, 'invalid measurements or schema')
        require(p.sha256_file(path) == digest, 'stale receipt: bytes changed during validation')
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('full requires a clean passing seen receipt ({}): {}'.format(context, error)) from error
    return dict(path=str(path), sha256=digest)


def timing_limits(fields, gpu):
    """Seen limits from the bound receipt; anything else is exp_05's decision."""
    effective = fields['effective_args']
    if effective.get('protocol') != PROTOCOL:
        return tier_gates.timing_limits(fields, gpu)
    if tier_of(effective) != 'M':
        raise ValueError('the seen protocol runs at tier M only')
    try:
        bound = fields['mutable_inputs']['probe_receipt']
        actual = validate_receipt(bound['path'], fields['reviewed_commit'], gpu, 'M',
                                  effective['backbone'], PROTOCOL, effective.get('yaw_aug', 0))
        raw = Path(actual['path']).read_bytes()
        if bound['sha256'] != actual['sha256'] or hashlib.sha256(raw).hexdigest() != actual['sha256']:
            raise ValueError('changed receipt')
        data = json.loads(raw)
        if 'resource_before' in fields and (fields['resource_before'].get('uuid') != data['before']['uuid']
                or fields['resource_before'].get('gpu') != gpu):
            raise ValueError('launch GPU differs from probe GPU')
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError('missing or changed seen probe receipt') from error
    return dict(epoch_seconds=1.05 * data['T_epoch'], projection_hours=data['T_run'] / 3600,
                ceiling_hours=1.5 * data['T_run'] / 3600, probe_receipt_sha256=actual['sha256'],
                protocol=PROTOCOL)


def set_budget(root, limits, renewal=None, commit=True):
    """exp_05's ledger plus the seen one-retry rule, checked before anything is written."""
    from tools.exp04_launcher import hours_record
    if limits.get('protocol') == PROTOCOL:
        full = [row for row in hours_record(root)['attempts'] if row['mode'] == 'full']
        if len(full) >= MAX_FULL_ATTEMPTS:
            raise ValueError('seen arm allows at most one retry: two full attempts recorded')
    return tier_gates.set_budget(root, limits, renewal=renewal, commit=commit)
