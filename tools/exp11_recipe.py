"""Type-strict schema for exp_11's two named pretraining profiles (plan v3 item 5).

``H_RECIPE`` is exp_01's numerical recipe -- the one arm H's control was trained under --
with the operational differences exp_06 declared (``epoch_ckpt_every 1``,
``save_every 500``, an exclusive attempt directory, recorded ``env`` and
``param_counts``). ``I_RECIPE`` is exp_04's yaw-augmented recipe: the same numbers and
the same loader length, plus exp_04's production constraints (``yaw_aug 1``,
``yaw_aug_seed 0``, ``yaw_aug_width 512``, ``save_every 0``, ``resume None``) and the
counter bound ``epochs * train_batches_per_epoch < 2**20``.

Field classes, exactly as ``tools.exp06_recipe`` defines them for exp_06:

* ``recipe``       -- the shared numerical recipe (``NUMERICAL``);
* ``production``   -- the constraints of the *selected* profile (they differ);
* ``operational``  -- declared differences, validated against the exp_11 registry;
* ``derived``      -- loader length, tier and parameter counts from the new factory;
* ``exp11``        -- run type, profile name, registry/closure digests, HEAD, provenance.

A profile is selected from what the run recorded (``select_profile``: backbone and
``yaw_aug``), never from a caller's claim, so a mixture -- H's saving cadence with I's
augmentation, or the reverse -- is a named deviation rather than a silent admission.
The generic comparison helpers (``strict_equal``, ``compare_sources``, ``check_budget``)
are imported from ``tools.exp06_recipe`` unchanged; that module is never edited.
"""
import functools
from types import MappingProxyType as MP

import torch

from model.xrir_exp11_registry import BACKBONES_EXP11, build_xrir_exp11
from tools.exp05_params import TIERS, count_parameters
from tools.exp06_recipe import (MISSING, _bounded_int, _compare, _environment, _integer,
                                check_budget, compare_sources, strict_equal)

NUMERICAL = MP(dict(num_shot=8, max_len=9600, lr=1e-3, weight_decay=1e-4, decay_epochs=3,
                    lr_gamma=0.1, epochs=12, batch_size=32, accum_steps=2, seed=0, tf32=True,
                    vit_dim=512, vit_depth=12, vit_heads=8, vit_mlp_dim=512))
TRAIN_BATCHES_PER_EPOCH = 9261    # loader length at batch 32 on the 296 334-sample split
TIER = 'M'
BACKBONE = 'simple_oriented'
YAW_COUNTER_LIMIT = 2 ** 20       # train_xRIR_backbone's own bound on the yaw counter

_COMMON = dict(max_train_batches=0, max_test_batches=0, test_subset=0, resume=None,
               no_save=False)
PROFILES = MP({
    'H_RECIPE': MP(dict(
        name='H_RECIPE', arm='simple_or', backbone=BACKBONE, recipe=NUMERICAL,
        production=MP(dict(_COMMON, yaw_aug=0)), inert=('yaw_aug_seed', 'yaw_aug_width'),
        save_every=500, epoch_ckpt_every=1)),
    'I_RECIPE': MP(dict(
        name='I_RECIPE', arm='simple_or_yaw', backbone=BACKBONE, recipe=NUMERICAL,
        production=MP(dict(_COMMON, yaw_aug=1, yaw_aug_seed=0, yaw_aug_width=512)),
        inert=(), save_every=0, epoch_ckpt_every=1)),
})
PROFILE_NAMES = tuple(PROFILES)

OPERATIONAL = ('backbone', 'save_dir', 'num_workers', 'log_interval', 'save_every',
               'epoch_ckpt_every', 'env')
DERIVED = ('train_batches_per_epoch', 'tier', 'param_counts')
EXP11 = ('exp11_run_type', 'exp11_profile', 'exp11_registry_sha256',
         'exp11_source_closure_sha256', 'exp11_git_head', 'exp11_provenance_path')
PRODUCTION_FIELDS = tuple(sorted(set(PROFILES['H_RECIPE']['production'])
                                 | set(PROFILES['I_RECIPE']['production'])
                                 | {'yaw_aug_seed', 'yaw_aug_width'}))
CLASSES = MP({'recipe': tuple(NUMERICAL), 'production': PRODUCTION_FIELDS,
              'operational': OPERATIONAL, 'derived': DERIVED, 'exp11': EXP11})


def classify(args_dict):
    """Map every recorded field to exactly one class; unknown fields are refused by name."""
    lookup = {field: name for name, group in CLASSES.items() for field in group}
    unknown = sorted(set(args_dict) - set(lookup))
    if unknown:
        raise ValueError('unknown args fields: ' + ', '.join(unknown))
    return {field: lookup[field] for field in args_dict}


def select_profile(args_dict):
    """The profile a run's own record puts it in; anything else is refused."""
    backbone, yaw = args_dict.get('backbone', MISSING), args_dict.get('yaw_aug', MISSING)
    for name, profile in PROFILES.items():
        if backbone == profile['backbone'] and strict_equal(
                yaw, profile['production']['yaw_aug']):
            return name
    raise ValueError('no exp_11 profile covers backbone {!r} with yaw_aug {!r}; the '
                     'registered profiles are {}'.format(
                         None if backbone is MISSING else backbone,
                         None if yaw is MISSING else yaw, ', '.join(PROFILE_NAMES)))


@functools.lru_cache(maxsize=None)
def expected_param_counts(backbone):
    """Parameter counts of one exp_11 backbone at the recipe's tier, from the factory.

    The counting model is randomly initialised, so it is built inside
    ``torch.random.fork_rng``: a cold call leaves the caller's generator where a warm one
    does.
    """
    with torch.random.fork_rng(devices=[]):
        model = build_xrir_exp11(backbone, NUMERICAL['num_shot'], **TIERS[TIER])
        return MP(count_parameters(model))


def check_recipe(args_dict, profile):
    """Deviations from the shared numerical recipe."""
    expected = PROFILES[profile]['recipe']
    return [item for item in (_compare('recipe.' + field, args_dict.get(field, MISSING),
                                       expected[field]) for field in sorted(expected)) if item]


def check_production(args_dict, profile):
    """Deviations from the selected profile's constraints; its inert fields stay integers."""
    production = PROFILES[profile]['production']
    deviations = [_compare('production.' + field, args_dict.get(field, MISSING), value)
                  for field, value in sorted(production.items())]
    deviations += [_integer('production.' + field, args_dict.get(field, MISSING))
                   for field in PROFILES[profile]['inert']]
    return [item for item in deviations if item]


def check_operational(args_dict, profile):
    """The declared operational differences must be valid, not merely present."""
    backbone = args_dict.get('backbone', MISSING)
    save_dir = args_dict.get('save_dir', MISSING)
    deviations = [
        None if backbone in BACKBONES_EXP11 else
        'operational.backbone: {!r} is not one of {}'.format(
            None if backbone is MISSING else backbone, sorted(BACKBONES_EXP11)),
        _compare('operational.backbone', backbone, PROFILES[profile]['backbone'])
        if backbone in BACKBONES_EXP11 else None,
        None if type(save_dir) is str and save_dir else
        'operational.save_dir: {!r} is not a directory path'.format(
            None if save_dir is MISSING else save_dir),
        _bounded_int('operational.num_workers', args_dict.get('num_workers', MISSING), 0),
        _bounded_int('operational.log_interval', args_dict.get('log_interval', MISSING), 1),
        _compare('operational.save_every', args_dict.get('save_every', MISSING),
                 PROFILES[profile]['save_every']),
        _compare('operational.epoch_ckpt_every', args_dict.get('epoch_ckpt_every', MISSING),
                 PROFILES[profile]['epoch_ckpt_every']),
        _environment('operational.env', args_dict.get('env', MISSING)),
    ]
    return [item for item in deviations if item]


def check_derived(args_dict, backbone):
    """Deviations of the fields the trainer derives from the data and the model."""
    deviations = []
    recorded = args_dict.get('backbone', MISSING)
    if recorded is MISSING or recorded != backbone:
        deviations.append('derived.backbone: {!r} is not {!r}'.format(
            None if recorded is MISSING else recorded, backbone))
    for item in (_compare('derived.train_batches_per_epoch',
                          args_dict.get('train_batches_per_epoch', MISSING),
                          TRAIN_BATCHES_PER_EPOCH),
                 _compare('derived.tier', args_dict.get('tier', MISSING), TIER)):
        if item:
            deviations.append(item)
    counts = args_dict.get('param_counts', MISSING)
    if backbone not in BACKBONES_EXP11:
        deviations.append('derived.param_counts: unknown backbone {!r}'.format(backbone))
    elif counts is MISSING:
        deviations.append('derived.param_counts: not recorded')
    elif type(counts) is not dict:
        deviations.append('derived.param_counts: {!r} is not a mapping'.format(counts))
    elif any(type(value) is not int for value in counts.values()):
        deviations.append('derived.param_counts: values must be native integers')
    elif counts != dict(expected_param_counts(backbone)):
        deviations.append('derived.param_counts: {!r} is not {!r}'.format(
            counts, dict(expected_param_counts(backbone))))
    return deviations


def check_presence(args_dict):
    """A current run records every operational and exp_11 provenance field."""
    return ['{}.{}: not recorded'.format(name, field)
            for name, group in (('operational', OPERATIONAL), ('exp11', EXP11))
            for field in group if field not in args_dict]


def check_counter(profile, epochs, batches_per_epoch):
    """exp_04's constraint: the yaw counter must not wrap inside the budget."""
    if not PROFILES[profile]['production'].get('yaw_aug'):
        return []
    if epochs * batches_per_epoch >= YAW_COUNTER_LIMIT:
        return ['production.yaw_aug: {} epochs x {} batches reaches the 2**20 counter '
                'limit'.format(epochs, batches_per_epoch)]
    return []


def check_all(args_dict, profile=None, history_rows=None, last_meta=None):
    """Every schema deviation of one run under the profile its own record selects."""
    deviations = []
    try:
        classify(args_dict)
    except ValueError as error:
        deviations.append('schema: ' + str(error))
    profile = select_profile(args_dict) if profile is None else profile
    if profile not in PROFILES:
        raise ValueError('unknown exp_11 profile: {!r}'.format(profile))
    claimed = args_dict.get('exp11_profile', MISSING)
    if claimed is not MISSING and claimed != profile:
        deviations.append('exp11.exp11_profile: {!r} is not the {!r} its arguments '
                          'select'.format(claimed, profile))
    deviations += check_presence(args_dict)
    deviations += check_operational(args_dict, profile)
    deviations += check_recipe(args_dict, profile)
    deviations += check_production(args_dict, profile)
    deviations += check_derived(args_dict, args_dict.get('backbone'))
    deviations += check_counter(profile, args_dict.get('epochs') or 0,
                                args_dict.get('train_batches_per_epoch') or 0)
    if history_rows is not None or last_meta is not None:
        deviations += check_budget(history_rows or [], last_meta or {},
                                   epochs=NUMERICAL['epochs'])
    return deviations
