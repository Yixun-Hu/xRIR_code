"""Frozen exp_05 registration; shared exp_04 literals are imported without I/O."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from types import MappingProxyType as MP
from tools.exp04_profiles import COMMON, REFERENCES, CONTROL, CYL, json_value


def freeze(value):
    return MP({k: freeze(v) for k, v in value.items()}) if isinstance(value, dict) else value


ARMS = tuple(freeze(dict(role=role, tier=tier, backbone=backbone, label=role, epoch=12,
    config=dict(dim=dim, depth=depth, heads=heads, mlp_dim=dim), dim_head=64, tokens=256,
    counts=dict(encoder=encoder, full=full, trainable=full-20, non_encoder=full-encoder),
    checkpoint=checkpoint, sha256=digest)) for role, tier, backbone, dim, depth, heads,
    encoder, full, checkpoint, digest in (
    ('S_simple', 'S', 'simple', 256, 6, 4, 2766080, 15073213, 'ckpt/exp05/S_simple/final/epoch_012.pth', None),
    ('S_cyl', 'S', 'cylindrical', 256, 6, 4, 2777984, 15085117, 'ckpt/exp05/S_cylindrical/final/epoch_012.pth', None),
    ('M_simple', 'M', 'simple', 512, 12, 8, 19703296, 32075965, CONTROL['checkpoint'], CONTROL['sha256']),
    ('M_cyl', 'M', 'cylindrical', 512, 12, 8, 19750912, 32123581, CYL['checkpoint'], CYL['sha256']),
    ('L_simple', 'L', 'simple', 768, 12, 12, 43709184, 56147389, 'ckpt/exp05/L_simple/final/epoch_012.pth', None),
    ('L_cyl', 'L', 'cylindrical', 768, 12, 12, 43780608, 56218813, 'ckpt/exp05/L_cylindrical/final/epoch_012.pth', None)))
COHORTS = MP({'arm': 'finite for every evaluation seed of this arm and metric',
    'paired': 'finite for every evaluation seed in both cell arms; H2 target uses this cohort',
    'target_display': 'baseline own-cohort curve mean shown separately; report both query/room sizes',
    'exclusions': 'query identities per arm/seed/metric; empty cohorts refuse',
    'inference': 'query bootstrap primary on fixed split; room bootstrap secondary'})
BASE = MP({**COMMON, 'arms': ARMS, 'eval_seeds': (42, 43, 44, 45, 46), 'cohorts': COHORTS,
    'grid': (0,), 'input_selection': 'standalone_k0',
    'run_grids': MP({a['role']: (0,) for a in ARMS}),
    'metrics': MP({'primary': ('EDT', 'C50'), 'descriptive': ('T60',)}),
    'curve_alpha': .05, 'axes': ('encoder', 'full'), 'seed_sd_ddof': 1})
H1 = MP({**BASE, 'mode': 'curve', 'family': 6, 'interval_alpha': .05/6,
    'pairings': (('S_cyl', 'S_simple'), ('M_cyl', 'M_simple'), ('L_cyl', 'L_simple')),
    'statistic': '(mean(cyl)-mean(simple))/mean(simple), recomputed inside shared resamples',
    'verdict_scope': 'per metric at the three tested tiers only; never extrapolate'})
H2 = MP({**BASE, 'mode': 'targets', 'family': 4, 'interval_alpha': .0125,
    'tost_alpha': .0125, 'margin': .03, 'pairings': (('S_cyl', 'M_simple'), ('M_cyl', 'L_simple')),
    'ratio_scope': 'encoder cyl/base ratio only when both EDT and C50 reach their paired-cohort targets',
    'target_rule': 'superiority upper < 0 OR TOST interval strictly inside (-0.03, +0.03)'})
CURVE_K8, CURVE_K1 = (MP({**H1, 'num_shot': k}) for k in (8, 1))
TARGETS_K8, TARGETS_K1 = (MP({**H2, 'num_shot': k}) for k in (8, 1))
YAW_K8_SEED42 = MP({**BASE, 'mode': 'yaw', 'num_shot': 8, 'family': 0,
    'eval_seeds': (42,), 'grid': (0, 32, 64, 448, 480), 'input_selection': 'block',
    'verdict_scope': 'descriptive only at the tested tiers and yaw angles; no confirmatory verdict',
    'run_grids': MP({a['role']: (0, 32, 64, 448, 480) for a in ARMS})})
PROFILES = MP(dict(CURVE_K8=CURVE_K8, CURVE_K1=CURVE_K1, TARGETS_K8=TARGETS_K8,
                   TARGETS_K1=TARGETS_K1, YAW_K8_SEED42=YAW_K8_SEED42))


def get_profile(name):
    return PROFILES[name]


def profile_digest(name):
    return hashlib.sha256(json.dumps(json_value(get_profile(name)), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


APPROVED_DIGESTS_PATH = Path(__file__).resolve().parents[1] / (
    'worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets/approved_digests.json')


def load_approved_digests(path=None):
    """Require all-null or complete committed pins; optional evaluator_exp04 enables M reuse."""
    path = Path(path or APPROVED_DIGESTS_PATH).resolve()
    raw = path.read_bytes()
    value = json.loads(raw)
    def shape(obj, keys):
        if type(obj) is not dict or set(obj) != set(keys):
            raise ValueError('approved digests schema: unexpected keys')
    shape(value, ('schema_version', 'closures', 'checkpoints'))
    closure_keys = ('evaluator', 'writer', 'training_launcher', 'training', 'producer_param_curve')
    shape(value['closures'], closure_keys + (('evaluator_exp04',) if 'evaluator_exp04' in value['closures'] else ()))
    shape(value['checkpoints'], ('S_simple', 'S_cyl', 'L_simple', 'L_cyl'))
    checks = [(value['schema_version'], lambda v: type(v) is int and v == 1)]
    digest = lambda v: type(v) is str and re.fullmatch('[0-9a-f]{64}', v)
    checks += [(value['closures'][k], digest) for k in closure_keys if k != 'training_launcher']
    launchers = value['closures']['training_launcher']
    checks += [(None if launchers == [] else launchers,
                lambda v: type(v) is list and len(v) >= 1 and all(digest(d) for d in v))]
    for role, arm in value['checkpoints'].items():
        shape(arm, ('path', 'epoch', 'sha256'))
        expected = next(a['checkpoint'] for a in ARMS if a['role'] == role)
        checks += [(arm['path'], lambda v, expected=expected: type(v) is str and v == expected),
                   (arm['epoch'], lambda v: type(v) is int and v == 12), (arm['sha256'], digest)]
    optional = value['closures'].get('evaluator_exp04')
    if any(v is not None and not valid(v) for v, valid in checks) or (optional is not None and not digest(optional)):
        raise ValueError('approved digests schema: invalid pin type or value')
    if any(v is None for v, _ in checks) and (any(v is not None for v, _ in checks) or optional is not None):
        raise ValueError('approved digests schema: requires all-null or all-filled pins')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(path.parent)] + list(args), stderr=subprocess.PIPE)
    try:
        repo = Path(git('rev-parse', '--show-toplevel').decode().strip())
        blob = git('rev-parse', 'HEAD:' + path.relative_to(repo).as_posix()).decode().strip()
        committed = git('cat-file', 'blob', blob)
    except subprocess.CalledProcessError as exc:
        raise ValueError('approved digests must be committed at HEAD') from exc
    if raw != committed:
        raise ValueError('approved digests differ from committed HEAD bytes')
    return freeze(value), dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), git_blob=blob)
