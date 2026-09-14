"""Literal exp_04 analysis specification. Unapproved identities stay None.

Digests use UTF-8 canonical JSON (sorted keys, compact separators, ASCII escapes).
The query digest hashes the ordered list of query strings with that same encoding.
No checkpoint, reference manifest, or source file is read while importing profiles.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from types import MappingProxyType as MP

APPROVED_DIGESTS_PATH = Path(__file__).resolve().parents[1] / (
    'worklog/worklog_yixun/exp_04_yaw_aug_xrir_claude/'
    'yaw_aug_xrir_results_assets/approved_digests.json')


def load_approved_digests(path=None):
    """Read committed A9 pins; return a frozen view and sha256/git-blob identity.

    schema_version=None is the all-null pre-approval template; approved schemas use 1.
    """
    path = Path(path or APPROVED_DIGESTS_PATH).resolve()
    raw = path.read_bytes()
    value = json.loads(raw)
    closures = ('evaluator', 'writer', 'training_launcher',
                'producer_paired_compare', 'producer_results_table', 'producer_descriptive')
    def shape(obj, keys):
        if type(obj) is not dict or set(obj) != set(keys):
            raise ValueError('approved digests schema: expected ' + ', '.join(keys))
    shape(value, ('schema_version', 'closures', 'checkpoints'))
    shape(value['closures'], closures)
    shape(value['checkpoints'], ('aug', 'aug_epoch9'))
    aug = value['checkpoints']['aug']
    shape(aug, ('path', 'epoch', 'sha256'))
    checks = [(value['schema_version'], lambda v: type(v) is int and v == 1),
              (aug['path'], lambda v: type(v) is str and bool(v)),
              (aug['epoch'], lambda v: type(v) is int and v > 0)]
    checks += [(v, lambda s: type(s) is str and re.fullmatch('[0-9a-f]{64}', s))
               for v in [value['closures'][k] for k in closures if k != 'producer_descriptive'] + [aug['sha256']]]
    if any(v is not None and not valid(v) for v, valid in checks):
        raise ValueError('approved digests schema: invalid pin type or value')
    if any(v is None for v, _ in checks) and not all(v is None for v, _ in checks):
        raise ValueError('approved digests schema: requires all-null or all-filled pins')
    for pin in (value['closures']['producer_descriptive'], value['checkpoints']['aug_epoch9']):
        if pin is not None and (type(pin) is not str or not re.fullmatch('[0-9a-f]{64}', pin)):
            raise ValueError('approved digests schema: invalid diagnostic pin')
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
    def freeze(obj):
        return MP({key: freeze(item) for key, item in obj.items()}) if isinstance(obj, dict) else obj
    return freeze(value), {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                           'git_blob': blob}


def json_value(value):
    """Make a detached JSON value; modifying it never modifies the specification."""
    if isinstance(value, MP):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    return value


REFERENCES = MP({
    8: MP({42: '9cf5c8a236afa4e38aa111d8658769767543389b169d6b16f7042c092d3faa10',
           43: '6f0a81fcfa1970eba40c5ebfc13ede58c23f387855e2ca0513da4b5f84456f6b',
           44: '42e4662a5964d8b7542a342eae0d650f21d47d4356f9ad988ecc4c0fd7a2f680',
           45: '3fb941d8c99a5ce1f00d9dc3cc523487f076040b0b76cc6c5166658f3f837f76',
           46: '03f2ad94a9827c01bec46d60ef8d938d31ca837bf0c538c5bb867f279669f71e'}),
    1: MP({42: 'b2955bddf62c7b9929c53e422bc8fd2725a4bc0be4fa24cdf2fcb28434f1d859',
           43: '0d9e033a7e11989b01266c730543fea1827611671db86c4259ee14425d5d6521',
           44: '9f7238cd36cfc05bb0d047ddd4152ed702b1515ff8f0b4679caa26d57a28a4d2',
           45: 'f2727209c214b26b478744d2a6e7060078e2f7084738b07130a34896c90b9013',
           46: '61fdcba41edfd647902e602539578dbc71d04c70028b47f6a3b3d9eea454d34b'})})
CONTROL = MP({'label': 'SimpleViT', 'backbone': 'simple', 'role': 'control', 'epoch': 12,
              'checkpoint': 'ckpt/xRIR_simple_8_shot/epoch_12.pth',
              'sha256': '651e3a377e110022fbd89332ff1f1c2648ef398bd83840bf369b05eb5decbedf'})
CYL = MP({'label': 'CylindricalViT', 'backbone': 'cylindrical', 'role': 'cyl', 'epoch': 12,
          'checkpoint': 'ckpt/xRIR_cyl_8_shot/epoch_12.pth',
          'sha256': '8ba344ad25d4a68f2ed7f22b4fb90e72b354fafe78b4d28d3eeeb6c1f8eab48e'})
AUG = MP({'label': 'YawAugxRIR', 'backbone': 'simple', 'role': 'aug', 'epoch': 12,
          'checkpoint': 'ckpt/xRIR_simple_yawaug_8_shot/final/epoch_012.pth'})
COMMON = MP({'schema_version': 1, 'condition': 'P', 'alpha': .05, 'n_boot': 20000,
             'bootstrap_seeds': (0, 1), 'convergence_tolerance': .10,
             'seeds': REFERENCES, 'gl_seed_rule': 'gl_seed == eval seed',
             'dataset': MP({'split': 'unseen', 'n_queries': 6337, 'n_rooms': 17,
                 'query_sha256': 'c94225ce7d67a97f311e22aa351deb193ef27547ba6222a907120388ce06b170',
                 'inventory_sha256': '23c3d8f6a0f740f54cb7d5db5766a80e82c6a78d60c05744e7542529a86a3092'}),
             'max_samples': 0, 'batch_size': 16, 'tf32': False})
H1 = MP({**COMMON, 'mode': 'two_arm', 'arms': (AUG, CONTROL), 'grid': (0,),
         'input_selection': 'standalone_k0', 'run_grids': MP({'aug': (0,), 'control': (0,)}),
         'metrics': MP({'primary': ('EDT', 'C50'), 'supportive': (), 'descriptive': ('T60',)}),
         'family': 2, 'margin': .03, 'tails': 'one_sided_upper',
         'companion_alpha': .05, 'superiority_alpha': .025})
BLOCKS = MP({'aug': (0, 32, 64, 128, 256, 384, 448, 480),
             'control': (0, 32, 64, 448, 480)})
PROFILES = MP({
    'H1_K8': MP({**H1, 'num_shot': 8}),
    'H1_K1': MP({**H1, 'num_shot': 1}),
    'H2_K8': MP({**COMMON, 'mode': 'two_arm', 'arms': (AUG, CONTROL), 'num_shot': 8,
        'grid': (32, 64, 448, 480), 'input_selection': 'block', 'run_grids': BLOCKS,
        'metrics': MP({'primary': ('C50',), 'supportive': ('EDT',), 'descriptive': ('T60',)}),
        'family': 8, 'margin': 0.0, 'tails': 'one_sided_upper',
        'companion_alpha': .0125, 'superiority_alpha': None}),
    'TOST_K8': MP({**COMMON, 'mode': 'one_arm', 'arms': (AUG,), 'num_shot': 8,
        'grid': (32, 64, 128, 256, 384, 448, 480), 'input_selection': 'block',
        'run_grids': MP({'aug': BLOCKS['aug']}),
        'metrics': MP({'primary': ('C50', 'EDT'), 'supportive': (), 'descriptive': ('T60',)}),
        'family': 14, 'margin': .02, 'tails': 'two_one_sided',
        'companion_alpha': 2 * .05 / 14, 'superiority_alpha': None}),
    'TABLE_V1': MP({**COMMON, 'mode': 'table', 'arms': (CONTROL, CYL, AUG), 'num_shot': (1, 8),
        'finite_count_tolerance': 2,
        'grid': (0,), 'input_selection': 'standalone_k0',
        'run_grids': MP({'control': (0,), 'cyl': (0,), 'aug': (0,)}),
        'metrics': MP({'primary': (), 'supportive': (),
                       'descriptive': ('T60', 'C50', 'EDT', 'loss', 'log_mse')}),
        'family': 0, 'margin': None, 'tails': 'descriptive',
        'companion_alpha': None, 'superiority_alpha': None})})


SPECTRAL_GRID = (0, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 416, 448, 480, 496, 504, 508)
ACOUSTIC_GRID = (0, 8, 32, 64, 128, 256, 384, 448, 480, 504)
DESCRIPTIVE = MP({**H1, 'mode': 'descriptive', 'arms': (AUG,), 'num_shot': 8, 'input_selection': 'diagnostic',
    'metrics': MP({'primary': (), 'supportive': (), 'descriptive': ('T60', 'C50', 'EDT', 'loss', 'log_mse')}),
    'family': 0, 'margin': None, 'tails': 'descriptive', 'companion_alpha': None, 'superiority_alpha': None})
PROFILES = MP({**PROFILES,
    'GRID_SEED42': MP({**DESCRIPTIVE, 'grid': SPECTRAL_GRID, 'acoustic_grid': ACOUSTIC_GRID,
        'run_grids': MP({'aug': SPECTRAL_GRID}), 'seeds': MP({8: MP({42: REFERENCES[8][42]})})}),
    'EPOCH9_K8': MP({**DESCRIPTIVE, 'grid': (0,), 'acoustic_grid': (0,), 'checkpoint_key': 'aug_epoch9',
        'arms': (MP({**AUG, 'epoch': 9, 'checkpoint': 'ckpt/xRIR_simple_yawaug_8_shot/final/epoch_009.pth'}),),
        'run_grids': MP({'aug': (0,)})})})


def get_profile(name):
    """Return the recursively immutable registered profile (unknown names raise)."""
    return PROFILES[name]


def profile_digest(name):
    """Hash the complete analysis specification, including unapproved placeholders."""
    payload = json.dumps(json_value(get_profile(name)), sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()
