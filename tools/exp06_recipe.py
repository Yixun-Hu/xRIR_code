"""Type-strict schema for every field the xRIR trainer records in ``args.json``.

Plan v4 section 5. Each of the 33 fields written by ``train_xRIR_backbone`` belongs to
exactly one class, and the five ``exp06_*`` provenance keys written by
``tools.exp06_train`` form a fifth:

* ``recipe`` -- must equal exp_01's pretraining recipe (``EXP01_RECIPE``);
* ``production`` -- constraints an arm must satisfy (no truncation, no resume, saving on,
  yaw augmentation off; ``yaw_aug_seed``/``yaw_aug_width`` are recorded but inert);
* ``operational`` -- declared differences (backbone, directories, workers, cadences, env);
* ``derived`` -- computed from the data and the model (loader length, tier, parameters);
* ``exp06`` -- run type, registry digest, closure digest, HEAD, provenance path.

Comparisons are type-strict (``type(value) is int/float/bool``), so a recorded ``1``
never passes for ``True`` and ``12.0`` never passes for ``12``. Every check returns a
list of deviation strings naming the offending field; an empty list is a pass.
"""
import functools
import math
from types import MappingProxyType

import torch

from model.xRIR_cyl_oriented import BACKBONES_EXP06, build_xrir_exp06
from tools.exp05_params import TIERS, count_parameters

EXP01_RECIPE = MappingProxyType(dict(
    num_shot=8, max_len=9600, lr=1e-3, weight_decay=1e-4, decay_epochs=3, lr_gamma=0.1,
    epochs=12, batch_size=32, accum_steps=2, seed=0, tf32=True,
    vit_dim=512, vit_depth=12, vit_heads=8, vit_mlp_dim=512))

PRODUCTION = MappingProxyType(dict(max_train_batches=0, max_test_batches=0, test_subset=0,
                                   resume=None, no_save=False, yaw_aug=0))

INERT = ('yaw_aug_seed', 'yaw_aug_width')

OPERATIONAL = ('backbone', 'save_dir', 'num_workers', 'log_interval', 'save_every',
               'epoch_ckpt_every', 'env')

DERIVED = ('train_batches_per_epoch', 'tier', 'param_counts')

EXP06 = ('exp06_run_type', 'exp06_registry_sha256', 'exp06_source_closure_sha256',
         'exp06_git_head', 'exp06_provenance_path')

CLASSES = MappingProxyType({'recipe': tuple(EXP01_RECIPE), 'production': tuple(PRODUCTION) + INERT,
                            'operational': OPERATIONAL, 'derived': DERIVED, 'exp06': EXP06})

TRAIN_BATCHES_PER_EPOCH = 9261  # loader length at batch 32 on the 296 334-sample train split
TIER = 'M'

MISSING = object()


def classify(args_dict):
    """Map every recorded field to exactly one class; unknown fields are refused by name."""
    lookup = {field: name for name, group in CLASSES.items() for field in group}
    unknown = sorted(set(args_dict) - set(lookup))
    if unknown:
        raise ValueError('unknown args fields: ' + ', '.join(unknown))
    return {field: lookup[field] for field in args_dict}


def _compare(label, actual, expected):
    """Type-strict equality; a missing field is a deviation, never a default."""
    if actual is MISSING:
        return '{}: not recorded'.format(label)
    if type(actual) is not type(expected) or actual != expected:
        return '{}: {!r} is not {!r}'.format(label, actual, expected)
    return None


def _integer(label, value):
    if value is MISSING:
        return '{}: not recorded'.format(label)
    if type(value) is not int:
        return '{}: {!r} is not an integer'.format(label, value)
    return None


def check_recipe(args_dict, expected=EXP01_RECIPE):
    """Deviations from exp_01's frozen pretraining recipe."""
    deviations = [_compare('recipe.' + field, args_dict.get(field, MISSING), expected[field])
                  for field in sorted(expected)]
    return [deviation for deviation in deviations if deviation]


def check_production(args_dict):
    """Deviations from the constraints an admissible training arm must satisfy."""
    deviations = [_compare('production.' + field, args_dict.get(field, MISSING), value)
                  for field, value in sorted(PRODUCTION.items())]
    deviations += [_integer('production.' + field, args_dict.get(field, MISSING))
                   for field in INERT]
    return [deviation for deviation in deviations if deviation]


def normalize_historical(args_dict):
    """Fill the fields exp_01's trainer never wrote; report filled and absent fields.

    Returns ``(normalized, report)``. Only the tier defaults and the inert yaw fields are
    filled: derived and production fields stay absent and are reported as ``not_recorded``
    so no check can mistake an assumption for a record.
    """
    normalized = dict(args_dict)
    filled = {}
    defaults = {'vit_' + key: value for key, value in TIERS[TIER].items()}
    defaults.update(yaw_aug=0, yaw_aug_seed=0, yaw_aug_width=512)
    for field, value in defaults.items():
        if field not in normalized:
            normalized[field] = filled[field] = value
    known = [field for name, group in CLASSES.items() if name != 'exp06' for field in group]
    return normalized, {'normalized': filled,
                        'not_recorded': sorted(f for f in known if f not in normalized)}


@functools.lru_cache(maxsize=None)
def expected_param_counts(backbone):
    """Parameter counts of the registered backbone at the recipe's tier.

    Nit 8: the counting model is randomly initialised, so it is built inside
    ``torch.random.fork_rng`` -- a cold call must leave the caller's global generator
    exactly where a warm (cached) call leaves it.
    """
    with torch.random.fork_rng(devices=[]):
        model = build_xrir_exp06(backbone, EXP01_RECIPE['num_shot'], **TIERS[TIER])
        return MappingProxyType(count_parameters(model))


def check_derived(args_dict, backbone):
    """Deviations of the fields the trainer derives from the data and the model."""
    deviations = []
    recorded = args_dict.get('backbone', MISSING)
    if recorded is MISSING or recorded != backbone:
        deviations.append('derived.backbone: {!r} is not {!r}'.format(
            None if recorded is MISSING else recorded, backbone))
    for deviation in (_compare('derived.train_batches_per_epoch',
                               args_dict.get('train_batches_per_epoch', MISSING),
                               TRAIN_BATCHES_PER_EPOCH),
                      _compare('derived.tier', args_dict.get('tier', MISSING), TIER)):
        if deviation:
            deviations.append(deviation)
    counts = args_dict.get('param_counts', MISSING)
    if backbone not in BACKBONES_EXP06:
        deviations.append('derived.param_counts: unknown backbone {!r}'.format(backbone))
    elif counts is MISSING:
        deviations.append('derived.param_counts: not recorded')
    elif type(counts) is not dict or counts != dict(expected_param_counts(backbone)):
        deviations.append('derived.param_counts: {!r} is not {!r}'.format(
            counts, dict(expected_param_counts(backbone))))
    return deviations


def check_budget(history_rows, last_meta, epochs=EXP01_RECIPE['epochs']):
    """Completeness of ``history.jsonl`` and of the epoch/batch recorded in ``last.pth``."""
    deviations = []
    rows = list(history_rows)
    recorded = [row.get('epoch', MISSING) for row in rows]
    if any(type(value) is not int for value in recorded):
        deviations.append('budget.epoch: every history row needs a native integer epoch')
    values = [value for value in recorded if type(value) is int]
    missing = [epoch for epoch in range(1, epochs + 1) if values.count(epoch) != 1]
    if missing:
        deviations.append('budget.epoch: epochs not recorded exactly once: {}'.format(missing))
    unexpected = sorted({value for value in values if not 1 <= value <= epochs})
    if unexpected:
        deviations.append('budget.epoch: unexpected epochs {}'.format(unexpected))
    if values != sorted(set(values)):
        deviations.append('budget.order: epochs must increase once each in file order')
    for row, epoch in zip(rows, recorded):
        for field in ('train_loss', 'test_loss', 'epoch_minutes'):
            value = row.get(field, MISSING)
            if type(value) not in (int, float) or not math.isfinite(value):
                deviations.append('budget.{}: epoch {!r} recorded {!r}'.format(
                    field, None if epoch is MISSING else epoch,
                    None if value is MISSING else value))
    for field, expected in (('epoch', epochs), ('batch_idx', 0)):
        deviation = _compare('budget.last.' + field, last_meta.get(field, MISSING), expected)
        if deviation:
            deviations.append(deviation)
    return deviations


def check_all(args_dict, backbone=None, history_rows=None, last_meta=None, expected=EXP01_RECIPE):
    """Every schema deviation of one run; budget checks run when history is supplied."""
    deviations = []
    try:
        classify(args_dict)
    except ValueError as error:
        deviations.append('schema: ' + str(error))
    deviations += check_recipe(args_dict, expected)
    deviations += check_production(args_dict)
    deviations += check_derived(args_dict, args_dict.get('backbone') if backbone is None else backbone)
    if history_rows is not None or last_meta is not None:
        deviations += check_budget(history_rows or [], last_meta or {})
    return deviations
