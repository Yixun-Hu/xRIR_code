"""exp_11 approvals: the code, reused and artifact identities every producer must match.

Plan v3 section 4.7. exp_11 owns a **separate** record,
``orientation_cue_fairness_results_assets/approved_digests.json``, with the same three
sections and fill times as exp_06's: ``code`` (source-closure digests, filled at the
second reviewed commit), ``reused`` (the exp_04 and exp_06 approvals records and exp_02's
legacy receipt) and ``artifacts`` (the two pretraining checkpoints, which exist only
after the runs). Until a leaf is filled it is ``null`` and every producer that needs it
refuses. ``tools/exp06_profiles.py`` and ``tools/exp06_approvals_api.py`` are imported
read-only and never edited; only ``committed_bytes`` is reused, because what "the bytes a
reviewer committed" means may not be re-implemented per experiment.

Codex round-2 change 5: the two shell orchestrators are outside every Python import
closure and get their own keys (``launch_sh``, ``haa_pipeline_sh``), and the
producer/arm **requirement matrix** consumes only available inputs -- pretraining never
requires its own output pin, Phase 1b and the adapter arms run while both H/I pins are
null, H runs while I's pin is null, and only the final publication requires every
planned arm.

    approved, identity = load_approved_digests(APPROVED_DIGESTS_PATH, repo, commit)
    require_producer(approved, 'haa_simple_or')          # raises on any missing leaf
    require(approved, ('train',), repo=REPO, commit=head)  # raises on code drift
"""
import functools
import hashlib
import json
from pathlib import Path
import re
import tempfile
from types import MappingProxyType as MP

from tools import exp06_profiles
from tools import provenance

REPO = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
TEMPLATE_PATH = REPO / 'tools/exp11_approved_digests_template.json'
APPROVED_RELATIVE = ('worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/'
                     'orientation_cue_fairness_results_assets/approved_digests.json')
APPROVED_DIGESTS_PATH = REPO / APPROVED_RELATIVE
DIGEST = re.compile('[0-9a-f]{64}')

CODE_SPECS = MP({
    'train': ('tools.exp11_train', ()),
    'finalize': ('tools.exp11_finalize', ()),
    'haa_finetune': ('tools.exp11_haa_finetune', ()),
    'haa_eval': ('tools.exp11_haa_eval', ()),
    'haa_pipeline_sh': (None, ('tools/exp11_haa_pipeline.sh',)),
    # The launcher sources the pinned exp_06 lifecycle library, so both files are
    # bound: what decides an exp_11 launch is the pair, not the new file alone.
    'launch_sh': (None, ('tools/exp11_launch.sh', 'tools/exp06_launch.sh')),
    'smoke': ('tools.exp11_smoke', ()),
    # The summariser is the one shared producer; exp_11 pins the closure it runs itself.
    'summarize_haa': ('tools.exp06_summarize_haa', ()),
})
CODE_KEYS = tuple(CODE_SPECS)
REUSED_KEYS = ('approved_digests_exp04', 'approved_digests_exp06', 'legacy_receipt')
ARTIFACT_KEYS = ('simpor_epoch_012', 'simpor_yaw_epoch_012')
ARTIFACT_LEAVES = ('epoch', 'path', 'sha256')
TRAINING_KEYS = ('train', 'finalize', 'launch_sh', 'smoke')
# The HAA queue runs the two entry points, the pipeline shell, the finalizer that decides
# every child, the diagnostics -- and the launcher library the pipeline sources for the
# child lifecycle, which is bound by ``launch_sh``.
HAA_KEYS = ('haa_finetune', 'haa_eval', 'haa_pipeline_sh', 'launch_sh', 'finalize',
            'smoke')
ARM_ARTIFACT = MP({'simple_or': 'simpor_epoch_012', 'simple_or_yaw': 'simpor_yaw_epoch_012',
                   'control_adapter': None, 'yawaug_adapter': None})

_CODE = tuple('code.' + key for key in CODE_KEYS)
_REUSED = ('reused.approved_digests_exp04', 'reused.approved_digests_exp06',
           'reused.legacy_receipt.path', 'reused.legacy_receipt.sha256')
_ARTIFACT = MP({key: tuple('artifacts.{}.{}'.format(key, leaf) for leaf in ARTIFACT_LEAVES)
                for key in ARTIFACT_KEYS})


def _haa(arm):
    """One HAA queue's leaves: its entry points, the reused pins, and its own init."""
    artifact = ARM_ARTIFACT[arm]
    return (tuple('code.' + key for key in HAA_KEYS) + _REUSED
            + (_ARTIFACT[artifact] if artifact else ()))


# The matrix. Each producer requires the listed leaves and never its own outputs.
PRODUCER_REQUIREMENTS = MP({
    'pretrain_simple_or': tuple('code.' + key for key in TRAINING_KEYS),
    'pretrain_simple_or_yaw': tuple('code.' + key for key in TRAINING_KEYS),
    'haa_control_adapter': _haa('control_adapter'),
    'haa_yawaug_adapter': _haa('yawaug_adapter'),
    'haa_simple_or': _haa('simple_or'),
    'haa_simple_or_yaw': _haa('simple_or_yaw'),
    'summarize_phase1b': ('code.summarize_haa',) + tuple('code.' + k for k in HAA_KEYS) + _REUSED,
    'summarize_final': _CODE + _REUSED + _ARTIFACT['simpor_epoch_012']
                       + _ARTIFACT['simpor_yaw_epoch_012'],
})
PRODUCER_OUTPUTS = MP({
    'pretrain_simple_or': _ARTIFACT['simpor_epoch_012'],
    'pretrain_simple_or_yaw': _ARTIFACT['simpor_yaw_epoch_012'],
    'haa_control_adapter': (), 'haa_yawaug_adapter': (),
    'haa_simple_or': (), 'haa_simple_or_yaw': (),
    'summarize_phase1b': (), 'summarize_final': (),
})
PRODUCER_CODE_KEYS = MP({
    'pretrain_simple_or': TRAINING_KEYS, 'pretrain_simple_or_yaw': TRAINING_KEYS,
    'haa_control_adapter': HAA_KEYS, 'haa_yawaug_adapter': HAA_KEYS,
    'haa_simple_or': HAA_KEYS, 'haa_simple_or_yaw': HAA_KEYS,
    'summarize_phase1b': ('summarize_haa',), 'summarize_final': ('summarize_haa',),
})


def _require(ok, cause):
    if not ok:
        raise ValueError(cause)


def _digest_or_null(value, label):
    _require(value is None or (type(value) is str and DIGEST.fullmatch(value)),
             'approved digests schema: {} is {!r}, not a sha256 or null'.format(label, value))


def _shape(value, keys, label):
    _require(type(value) is dict and set(value) == set(keys),
             'approved digests schema: {} must have exactly {}'.format(
                 label, ', '.join(sorted(keys))))


def _freeze(value):
    return MP({key: _freeze(item) for key, item in value.items()}) if isinstance(value, dict) else value


def validate(value):
    """The whole schema; an unknown key or a malformed leaf is refused by name."""
    _shape(value, ('schema_version', 'code', 'reused', 'artifacts'), 'the file')
    _require(type(value['schema_version']) is int and value['schema_version'] == SCHEMA_VERSION,
             'approved digests schema: schema_version must be {}'.format(SCHEMA_VERSION))
    _shape(value['code'], CODE_KEYS, 'code')
    _shape(value['reused'], REUSED_KEYS, 'reused')
    _shape(value['artifacts'], ARTIFACT_KEYS, 'artifacts')
    for key in CODE_KEYS:
        _digest_or_null(value['code'][key], 'code.' + key)
    for key in ('approved_digests_exp04', 'approved_digests_exp06'):
        _digest_or_null(value['reused'][key], 'reused.' + key)
    _shape(value['reused']['legacy_receipt'], ('path', 'sha256'), 'reused.legacy_receipt')
    _digest_or_null(value['reused']['legacy_receipt']['sha256'], 'reused.legacy_receipt.sha256')
    for key in ARTIFACT_KEYS:
        record = value['artifacts'][key]
        _shape(record, ARTIFACT_LEAVES, 'artifacts.' + key)
        _digest_or_null(record['sha256'], 'artifacts.{}.sha256'.format(key))
        _require(record['epoch'] is None or (type(record['epoch']) is int
                                             and record['epoch'] > 0),
                 'approved digests schema: artifacts.{}.epoch is {!r}'.format(
                     key, record['epoch']))
    for label, path in [('reused.legacy_receipt.path',
                         value['reused']['legacy_receipt']['path'])] + [
                        ('artifacts.{}.path'.format(key), value['artifacts'][key]['path'])
                        for key in ARTIFACT_KEYS]:
        _require(path is None or (type(path) is str and path),
                 'approved digests schema: {} is {!r}, not a path or null'.format(label, path))
    return value


def load_approved_digests(path=None, repo=None, commit=None):
    """Read and validate the approvals; with ``repo``/``commit`` bind the committed blob."""
    _require((repo is None) == (commit is None),
             'binding approvals needs both the repository and the commit')
    path = Path(TEMPLATE_PATH if path is None else path).resolve()
    try:
        raw = path.read_bytes()
        value = validate(json.loads(raw))
    except (OSError, ValueError) as error:
        raise ValueError('approved digests are unreadable: {}'.format(error)) from error
    identity = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    if repo is not None:
        relative, blob = exp06_profiles.committed_bytes(path, repo, commit)
        _require(blob == raw, 'approvals {} differ from the bytes committed at {}'.format(
            relative, commit))
        identity.update(repo_relative=relative, committed_at=commit)
    return _freeze(value), identity


def approvals_at_commit(path, repo, commit):
    """The approvals a reviewer committed, read from that immutable blob itself.

    exp_06's rule, for exp_11's record: binding a *working* file to a child's reviewed
    commit makes every certified child unverifiable the moment the approvals are re-filled
    for a later producer. The blob at the child's own commit is the approval it ran under,
    so it is parsed on a private copy of exactly those bytes.
    """
    relative, blob = exp06_profiles.committed_bytes(path, repo, commit)
    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / Path(path).name
        copy.write_bytes(blob)
        approved, _ = load_approved_digests(copy)
    return approved, {'path': str(path), 'repo_relative': relative,
                      'sha256': hashlib.sha256(blob).hexdigest(), 'committed_at': commit}


@functools.lru_cache(maxsize=None)
def _paths_of(name, repo):
    """The file list one key's digest is taken over; a missing module is reported."""
    module, extra = CODE_SPECS[name]
    files = list(provenance.source_closure(module, repo)) if module else []
    return tuple(files) + tuple(extra)


@functools.lru_cache(maxsize=None)
def _closure_of(name, repo, commit, stamp):
    return provenance.closure_record(list(_paths_of(name, repo)), commit, repo)


def _stamp(files, repo):
    """Stat validation of the cache: a file edited mid-process invalidates its answer."""
    return tuple((name, (Path(repo) / name).stat().st_size,
                  (Path(repo) / name).stat().st_mtime_ns) for name in files)


def closure_of(name, repo, commit):
    """``(file records, digest)`` for one code key: what a producer records at spawn."""
    _require(name in CODE_SPECS, 'unknown approval key: {!r}'.format(name))
    repo = str(Path(repo).resolve())
    files = _paths_of(name, repo)
    records, digest = _closure_of(name, repo, commit, _stamp(files, repo))
    return [dict(record) for record in records], digest


def code_digest(name, repo, commit):
    """The approved identity of one code key at one reviewed commit."""
    return closure_of(name, repo, commit)[1]


def present_keys(repo=REPO):
    """The keys whose module and shell files exist in this checkout."""
    repo, names = Path(repo), []
    for name, (module, extra) in CODE_SPECS.items():
        relative = (module.replace('.', '/') + '.py') if module else None
        if relative is not None and not (repo / relative).is_file():
            continue
        if any(not (repo / path).is_file() for path in extra):
            continue
        names.append(name)
    return tuple(names)


def compute_code_digests(repo=REPO, commit=None, keys=None, notes=None):
    """Recompute ``{key: digest}`` for every requested key that exists in this checkout."""
    repo = Path(repo).resolve()
    commit = provenance.git_state(repo)['HEAD'] if commit is None else commit
    available = set(present_keys(repo))
    digests = {}
    for name in (CODE_KEYS if keys is None else keys):
        _require(name in CODE_SPECS, 'unknown approval key: {!r}'.format(name))
        if name not in available:
            if notes is not None:
                notes.append('{}: not in this checkout yet'.format(name))
            continue
        digests[name] = code_digest(name, str(repo), commit)
    return digests


def _leaf(approved, path):
    value = approved
    for part in path.split('.'):
        _require(isinstance(value, (dict, MP)) and part in value,
                 'approved digests record no ' + path)
        value = value[part]
    return value


def producer_leaves(producer):
    """The approvals leaves one producer consumes; an unknown producer is refused."""
    _require(producer in PRODUCER_REQUIREMENTS, 'unknown producer: {}'.format(producer))
    return PRODUCER_REQUIREMENTS[producer]


def require_producer(approved, producer, exploratory=False):
    """The requirement matrix for one producer; production raises, exploratory reports."""
    deviations = ['not approved: ' + path for path in producer_leaves(producer)
                  if _leaf(approved, path) is None]
    _require(not deviations or exploratory,
             'approvals incomplete: ' + '; '.join(deviations))
    return deviations


def require(approved, keys, repo=REPO, commit=None, exploratory=False, current=None):
    """Fail-closed admission: every named code key is approved and is what is here now."""
    code = approved['code'] if 'code' in approved else approved
    unknown = sorted(key for key in keys if key not in CODE_SPECS)
    _require(not unknown, 'unknown approval key: ' + ', '.join(unknown))
    if current is None:
        current = compute_code_digests(repo, commit, keys=keys)
    deviations = []
    for key in keys:
        if code.get(key) is None:
            deviations.append('code.{}: not approved (null in approvals)'.format(key))
        elif key not in current:
            deviations.append('code.{}: approved but absent from this checkout'.format(key))
        elif current[key] != code[key]:
            deviations.append('code.{}: {} is not the approved {}'.format(
                key, current[key], code[key]))
    _require(not deviations or exploratory,
             'approved code digests do not admit this run: ' + '; '.join(deviations))
    return deviations
