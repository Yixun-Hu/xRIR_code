"""Literal exp_07 (seen protocol) analysis specification; unbuilt identities stay None.

The seen reference manifests and the query list they fix do not exist until the Planner
builds them with tools/exp07_manifests.py, and the three seen checkpoints do not exist
until the trainings finish.  ``dataset.query_sha256``, ``dataset.inventory_sha256``, the
ten manifest hashes of ``seeds`` and the new arms' ``sha256`` are therefore documented
placeholders, filled in the plan's pin-finalisation step (plan section 3) and checked
against the artefacts by tests/test_exp07_profiles.py as soon as those exist.  The
released checkpoint is external and pinned here and now.

Digests use the shared canonical JSON encoding.  No checkpoint, reference manifest or
source file is read while importing profiles.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from types import MappingProxyType as MP

from tools.exp04_profiles import json_value

REPO = Path(__file__).resolve().parents[1]
EVAL_SEEDS = (42, 43, 44, 45, 46)
# The authors' seen split (plan section 2); the pickle itself is a revalidated input.
SEEN_SPLIT_SHA256 = 'bc97850b090198cf956dc7975a2da611dd058f8b0f83adee7f93f7b7fad7720f'
RELEASED_SHA256 = '1762c702ee23a8f584e67c4b5b052b7483237e6471bb5e58774b66d514aec2f8'


def freeze(value):
    return MP({key: freeze(item) for key, item in value.items()}) if isinstance(value, dict) else value


def _arm(role, label, backbone, yaw_aug):
    return freeze(dict(role=role, label=label, backbone=backbone, epoch=12, sha256=None,
                       checkpoint='ckpt/exp07/{}/final/epoch_012.pth'.format(role),
                       protocol='seen', yaw_aug=yaw_aug, yaw_aug_seed=0, yaw_aug_width=512,
                       training='internal', reference=False))


ARMS = (_arm('seen_simple', 'SimpleViT (seen)', 'simple', 0),
        _arm('seen_cyl', 'CylindricalViT (seen)', 'cylindrical', 0),
        _arm('seen_aug', 'YawAugxRIR (seen)', 'simple', 1),
        freeze(dict(role='released_seen', label='xRIR released (reference)', backbone='simple',
                    checkpoint='checkpoints/xRIR_seen.pth', sha256=RELEASED_SHA256,
                    epoch=None, protocol=None, yaw_aug=None, yaw_aug_seed=None,
                    yaw_aug_width=None, training='external', reference=True)))
# The plan section 2 command line, compared field for field against the bound args.json.
RECIPE = freeze(dict(num_shot=8, max_len=9600, lr=.001, weight_decay=.0001, decay_epochs=3,
                     lr_gamma=.1, epochs=12, batch_size=32, accum_steps=2, num_workers=12,
                     seed=0, tf32=True, log_interval=50, save_every=0, epoch_ckpt_every=1,
                     protocol='seen'))
FULL_RUN = freeze(dict(no_save=False, resume=None, max_train_batches=0, max_test_batches=0,
                       test_subset=0))
DATASET = freeze(dict(split='seen', n_queries=6217, n_rooms=131, query_sha256=None,
                      inventory_sha256=None, seen_split_sha256=SEEN_SPLIT_SHA256))
SEEDS = MP({shot: MP({seed: None for seed in EVAL_SEEDS}) for shot in (8, 1)})
COMMON = MP({'schema_version': 1, 'protocol': 'seen', 'tier': 'M', 'condition': 'P',
             'alpha': .05, 'n_boot': 20000, 'bootstrap_seeds': (0, 1), 'seed_sd_ddof': 1,
             'convergence_tolerance': .10, 'seeds': SEEDS, 'eval_seeds': EVAL_SEEDS,
             'gl_seed_rule': 'gl_seed == eval seed', 'dataset': DATASET, 'arms': ARMS,
             'recipe': RECIPE, 'full_run': FULL_RUN, 'train_batches_per_epoch': 9265,
             'train_inventory_files': 296454,  # the seen training split (plan section 2)
             'projection_max_hours': 60., 'ceiling_factor': 1.5,  # plan section 3, round 2
             'grid': (0,), 'input_selection': 'standalone_k0', 'max_samples': 0,
             'batch_size': 16, 'tf32': False,
             'run_grids': MP({arm['role']: (0,) for arm in ARMS})})
TABLE_SEEN_V1 = MP({**COMMON, 'mode': 'table', 'num_shot': (1, 8),
                    'finite_count_tolerance': 2, 'family': 0, 'margin': None,
                    'tails': 'descriptive', 'companion_alpha': None, 'superiority_alpha': None,
                    'metrics': MP({'primary': (), 'supportive': (),
                                   'descriptive': ('T60', 'C50', 'EDT', 'loss', 'log_mse')})})
PAIRS_SEEN_V1 = MP({**COMMON, 'mode': 'pairs', 'num_shot': (8, 1), 'family': 0,
                    'margin': None, 'tails': 'descriptive', 'decision_driving': False,
                    'pairings': (('seen_cyl', 'seen_simple'), ('seen_aug', 'seen_simple'),
                                 ('released_seen', 'seen_simple')),
                    'reference_pairings': (('released_seen', 'seen_simple'),),
                    'statistics': ('absolute', 'relative'), 'quantiles': (.025, .975),
                    'interval_alpha': .05, 'cluster': 'whole_room_query_weighted',
                    'statistic': 'difference of arm means in native units and relative to '
                                 'the baseline mean, both recomputed inside shared resamples',
                    'verdict_scope': 'descriptive, checkpoint-conditional intervals for these '
                                     'checkpoints only; no verdict, no superiority claim',
                    'metrics': MP({'primary': (), 'supportive': (),
                                   'descriptive': ('EDT', 'C50', 'T60', 'loss', 'log_mse')})})
PROFILES = MP(dict(TABLE_SEEN_V1=TABLE_SEEN_V1, PAIRS_SEEN_V1=PAIRS_SEEN_V1))


def get_profile(name):
    """Return the recursively immutable registered profile (unknown names raise)."""
    return PROFILES[name]


def profile_digest(name):
    """Hash the complete analysis specification, including unapproved placeholders."""
    payload = json.dumps(json_value(get_profile(name)), sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


APPROVED_DIGESTS_PATH = REPO / ('worklog/worklog_yixun/exp_07_seen_protocol_claude/'
                                'seen_protocol_results_assets/approved_digests.json')
CLOSURES = ('evaluator', 'writer', 'training_launcher', 'training', 'producer_table',
            'producer_pairs')
NEW_ROLES = tuple(arm['role'] for arm in ARMS if arm['training'] == 'internal')


def load_approved_digests(path=None):
    """Read the committed exp_07 pins; return a frozen view and the file's identity.

    ``evaluator`` pins tools.exp07_eval's closure and ``writer`` tools.exp04_eval_launch's;
    ``training_launcher`` is the list of admissible training-launcher closures and
    ``training`` the single closure all three arms must share.  The committed all-null
    template loads -- the schema is reviewable before anything is approved -- but every
    producer refuses those nulls in production; only --exploratory lists them as
    deviations.  Pins are all-null or all-filled: a half-filled file is refused.
    """
    path = Path(path or APPROVED_DIGESTS_PATH).resolve()
    raw = path.read_bytes()
    value = json.loads(raw)
    def shape(obj, keys):
        if type(obj) is not dict or set(obj) != set(keys):
            raise ValueError('approved digests schema: expected ' + ', '.join(keys))
    shape(value, ('schema_version', 'closures', 'checkpoints'))
    shape(value['closures'], CLOSURES)
    shape(value['checkpoints'], NEW_ROLES)
    def digest(item):
        return type(item) is str and bool(re.fullmatch('[0-9a-f]{64}', item))
    checks = [(value['schema_version'], lambda v: type(v) is int and v == 1)]
    checks += [(value['closures'][key], digest) for key in CLOSURES if key != 'training_launcher']
    launchers = value['closures']['training_launcher']  # [] is this pin's unfilled form
    checks += [(None if launchers == [] else launchers,
                lambda v: type(v) is list and len(v) >= 1 and all(digest(item) for item in v))]
    for role, arm in value['checkpoints'].items():
        shape(arm, ('path', 'epoch', 'sha256'))
        expected = next(a['checkpoint'] for a in ARMS if a['role'] == role)
        checks += [(arm['path'], lambda v, expected=expected: type(v) is str and v == expected),
                   (arm['epoch'], lambda v: type(v) is int and v == 12), (arm['sha256'], digest)]
    if any(item is not None and not valid(item) for item, valid in checks):
        raise ValueError('approved digests schema: invalid pin type or value')
    if any(item is None for item, _ in checks) and any(item is not None for item, _ in checks):
        raise ValueError('approved digests schema: requires all-null or all-filled pins')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(path.parent)] + list(args),
                                       stderr=subprocess.PIPE)
    try:
        repo = Path(git('rev-parse', '--show-toplevel').decode().strip())
        blob = git('rev-parse', 'HEAD:' + path.relative_to(repo).as_posix()).decode().strip()
        committed = git('cat-file', 'blob', blob)
    except subprocess.CalledProcessError as exc:
        raise ValueError('approved digests must be committed at HEAD') from exc
    if raw != committed:
        raise ValueError('approved digests differ from committed HEAD bytes')
    return freeze(value), dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                               git_blob=blob)
