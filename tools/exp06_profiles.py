"""exp_06 approvals: the code, reused and artifact identities every producer must match.

Plan v4 section 6.4. ``oriented_cyl_results_assets/approved_digests.json`` has three
sections with separate fill times -- ``code`` (source-closure digests, filled before any
confirmatory execution), ``reused`` (the exp_04/05 approvals and the exp_02 record
identity) and ``artifacts`` (hashes that exist only after the runs). Until a section is
approved its entries are ``null``, and every producer that needs them refuses.

This module implements the **training-critical subset** of finding 1 of the Codex
``full_train`` review: ``TRAINING_KEYS`` are the identities the confirmatory pretraining
depends on, including the shell launcher and the finalizer that decides its completion.
Round 3b extends the file with the evaluation and statistics producers, whose keys are
already present as ``null`` placeholders.

A key's digest is ``closure_record(source_closure(module) + extra files)[1]``, the
exp_03-compatible hash over ``[[path, reviewed blob sha256], ...]``. The two shell
orchestrators have no import closure, so they are bound as files alone -- exp_04's
launcher pattern (``tools/exp04_launcher.py``), which appends shell files to a Python
closure, has no Python entry point here.

    approved, identity = load_approved_digests()
    require(approved, TRAINING_KEYS, repo=REPO, commit=head)   # raises on any drift
"""
import functools
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType as MP

from tools import provenance

REPO = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
TEMPLATE_PATH = REPO / 'tools/exp06_approved_digests_template.json'
APPROVED_DIGESTS_PATH = REPO / ('worklog/worklog_yixun/exp_06_oriented_cyl_claude/'
                                'oriented_cyl_results_assets/approved_digests.json')

# The exp_03 evaluator is pinned by tests/test_exp03_record_tools.py, so recomputing its
# closure at any later commit reproduces this digest; it is the one value filled up front.
EVALUATOR_EXP03 = '5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48'

CODE_SPECS = MP({
    'trainer': ('tools.exp06_train', ()),
    'finalize': ('tools.exp06_finalize', ()),
    'recipe': ('tools.exp06_recipe', ()),
    'smoke': ('tools.exp06_smoke', ()),
    'encoder': ('model.cylindrical_vit_oriented', ()),
    'factory': ('model.xRIR_cyl_oriented', ()),
    'heading': ('tools.exp06_heading', ()),
    'profiles': ('tools.exp06_profiles', ()),
    'launch_sh': (None, ('tools/exp06_launch.sh',)),
    'haa_pipeline_sh': (None, ('tools/exp06_haa_pipeline.sh',)),
    'evaluator_exp03': ('eval_yaw_rotation', ()),
    'eval': ('tools.exp06_eval', ()),
    'eval_launch': ('tools.exp06_eval_launch', ()),
    'haa_finetune': ('tools.exp06_haa_finetune', ()),
    'haa_eval': ('tools.exp06_haa_eval', ()),
    'mirror_probe': ('tools.exp06_mirror_probe', ()),
    'summarize_haa': ('tools.exp06_summarize_haa', ()),
    'bootstrap': ('tools.exp06_bootstrap', ()),
    'compare': ('tools.exp06_compare', ()),
})
CODE_KEYS = tuple(CODE_SPECS)
TRAINING_KEYS = ('trainer', 'finalize', 'recipe', 'smoke', 'encoder', 'factory',
                 'launch_sh', 'profiles')
REUSED_KEYS = ('evaluator_exp04', 'writer_exp04', 'approved_digests_exp04',
               'approved_digests_exp05', 'exp02_stats_sha256', 'exp02_summary_sha256',
               'legacy_receipt')
ARTIFACT_KEYS = ('epoch_012', 'heading', 'gate_g1')
HEADING_ROOMS = ('class_room', 'complex_room', 'dampened_room', 'hallway')
NESTED = MP({'reused.legacy_receipt': ('path', 'sha256'),
             'artifacts.epoch_012': ('epoch', 'path', 'sha256'),
             'artifacts.heading': HEADING_ROOMS})
DIGEST = re.compile('[0-9a-f]{64}')


@functools.lru_cache(maxsize=None)
def _paths_of(name, repo):
    """The file list a key's digest is taken over; a missing module is reported, never guessed.

    Cached per process: one import subprocess per module, however many producers ask.
    """
    module, extra = CODE_SPECS[name]
    files = list(provenance.source_closure(module, repo)) if module else []
    return tuple(files) + tuple(extra)


def json_value(value):
    """A detached JSON value; modifying it never modifies the loaded approvals."""
    if isinstance(value, (MP, dict)):
        return {key: json_value(item) for key, item in value.items()}
    return value


def _freeze(value):
    return MP({key: _freeze(item) for key, item in value.items()}) if isinstance(value, dict) else value


def _shape(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError('approved digests schema: {} must have exactly {}'.format(
            label, ', '.join(sorted(keys))))


def _digest_or_null(value, label):
    if value is not None and not (type(value) is str and DIGEST.fullmatch(value)):
        raise ValueError('approved digests schema: {} is {!r}, not a sha256 or null'.format(
            label, value))


def load_approved_digests(path=None):
    """Read the approvals; unknown or missing keys and malformed values are refused.

    Returns ``(frozen approvals, {'path', 'sha256'})``. The hash is what a producer
    records so the finalizer can re-read the file and prove it did not change mid-run.
    """
    path = Path(TEMPLATE_PATH if path is None else path).resolve()
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError) as error:
        raise ValueError('approved digests are unreadable: {}'.format(error)) from error
    _shape(value, ('schema_version', 'code', 'reused', 'artifacts'), 'the file')
    if value['schema_version'] != SCHEMA_VERSION or type(value['schema_version']) is not int:
        raise ValueError('approved digests schema: schema_version must be {}'.format(SCHEMA_VERSION))
    _shape(value['code'], CODE_KEYS, 'code')
    _shape(value['reused'], REUSED_KEYS, 'reused')
    _shape(value['artifacts'], ARTIFACT_KEYS, 'artifacts')
    for name, keys in NESTED.items():
        section, key = name.split('.')
        _shape(value[section][key], keys, name)
    for key, item in sorted(value['code'].items()):
        _digest_or_null(item, 'code.' + key)
    for key in ('exp02_stats_sha256', 'exp02_summary_sha256', 'evaluator_exp04',
                'writer_exp04', 'approved_digests_exp04', 'approved_digests_exp05'):
        _digest_or_null(value['reused'][key], 'reused.' + key)
    _digest_or_null(value['reused']['legacy_receipt']['sha256'], 'reused.legacy_receipt.sha256')
    _digest_or_null(value['artifacts']['epoch_012']['sha256'], 'artifacts.epoch_012.sha256')
    _digest_or_null(value['artifacts']['gate_g1'], 'artifacts.gate_g1')
    for room in HEADING_ROOMS:
        _digest_or_null(value['artifacts']['heading'][room], 'artifacts.heading.' + room)
    return _freeze(value), {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def file_digest(files, repo, commit):
    """The closure digest over an explicit file list (the shell orchestrators)."""
    return provenance.closure_record(list(files), commit, repo)[1]


@functools.lru_cache(maxsize=None)
def _closure_of(name, repo, commit, stamp):
    return provenance.closure_record(list(_paths_of(name, repo)), commit, repo)


def _stamp(files, repo):
    """Stat validation of the cache, as ``provenance.train_data_identity`` uses for data.

    The records carry working-tree hashes, so a cached answer must be invalidated when a
    file is edited; size and mtime do that. A producer process computes each key once.
    """
    stamps = []
    for name in files:
        status = (Path(repo) / name).stat()
        stamps.append((name, status.st_size, status.st_mtime_ns))
    return tuple(stamps)


def closure_of(name, repo, commit):
    """``(file records, digest)`` for one code key: what a producer records at spawn."""
    if name not in CODE_SPECS:
        raise ValueError('unknown approval key: {!r}'.format(name))
    repo = str(Path(repo).resolve())
    files = _paths_of(name, repo)
    records, digest = _closure_of(name, repo, commit, _stamp(files, repo))
    return [dict(record) for record in records], digest


def code_digest(name, repo, commit):
    """The approved identity of one code key at one reviewed commit."""
    return closure_of(name, repo, commit)[1]


def present_keys(repo=REPO):
    """The keys whose module and files exist in this checkout; the rest await later rounds."""
    repo = Path(repo)
    names = []
    for name, (module, extra) in CODE_SPECS.items():
        relative = (module.replace('.', '/') + '.py') if module else None
        if relative is not None and not (repo / relative).is_file():
            continue
        if any(not (repo / path).is_file() for path in extra):
            continue
        names.append(name)
    return tuple(names)


PRESENT_KEYS_NOW = present_keys()


def compute_code_digests(repo=REPO, commit=None, keys=None, notes=None):
    """Recompute {key: digest} for every requested key that exists in this checkout.

    A key whose module or shell file is absent (rounds 2b and 3) is skipped and named in
    ``notes`` rather than silently defaulting to anything.
    """
    repo = Path(repo).resolve()
    commit = provenance.git_state(repo)['HEAD'] if commit is None else commit
    available = set(present_keys(repo))
    digests = {}
    for name in (CODE_KEYS if keys is None else keys):
        if name not in CODE_SPECS:
            raise ValueError('unknown approval key: {!r}'.format(name))
        if name not in available:
            if notes is not None:
                notes.append('{}: not in this checkout yet'.format(name))
            continue
        digests[name] = code_digest(name, str(repo), commit)
    return digests


def require(approved, keys, repo=REPO, commit=None, exploratory=False, current=None):
    """Fail-closed admission: every named key must be approved and match what is here now.

    ``exploratory=True`` returns the deviations instead of raising, for the diagnostic
    runs that record them in their receipt; it never admits a confirmatory run.
    """
    code = approved['code'] if 'code' in approved else approved
    unknown = sorted(key for key in keys if key not in CODE_SPECS)
    if unknown:
        raise ValueError('unknown approval key: ' + ', '.join(unknown))
    if current is None:
        current = compute_code_digests(repo, commit, keys=keys)
    deviations = []
    for key in keys:
        if code.get(key) is None:
            deviations.append('code.{}: not approved (null in {})'.format(key, 'approvals'))
        elif key not in current:
            deviations.append('code.{}: approved but absent from this checkout'.format(key))
        elif current[key] != code[key]:
            deviations.append('code.{}: {} is not the approved {}'.format(
                key, current[key], code[key]))
    if deviations and not exploratory:
        raise ValueError('approved code digests do not admit this run: '
                         + '; '.join(deviations))
    return deviations
