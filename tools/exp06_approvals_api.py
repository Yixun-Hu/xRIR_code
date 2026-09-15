"""Round 3b's view of exp_06's approvals (plan v4 6.4), behind a lazy adapter.

``tools.exp06_profiles`` and its ``approved_digests.json`` were created on the training
branch ``exp06-window``; they are not part of this branch and the two merge after this
round. So the round-3b producers reach the approvals only through this module, which

* imports that one **lazily, inside functions**, and raises a named
  :class:`ApprovalsUnavailable` when it is absent, so a branch without it refuses rather
  than importing a parallel implementation of the same file, and
* owns what round 3b adds on top of the training-critical subset there (``code`` keys,
  ``load_approved_digests``, ``code_digest``, ``compute_code_digests``, ``require``,
  ``TRAINING_KEYS``): the full ``reused`` and ``artifacts`` semantics, the remaining
  ``code`` keys, and 6.4's per-producer requirement matrix.

Two documented differences from the template committed in the record folder (reported,
never edited here): it carries no ``code.probe_align`` -- the §6.4 helper the mirror
probe's alignment comes from, which must be **added** -- and it spells four ``reused``
and one ``artifacts`` key differently. The spellings are accepted through the alias
tables below and normalised, so no committed key has to be renamed; the missing
``probe_align`` key is a real gap and is refused.
"""
import importlib
import re

MODULE = 'tools.exp06_profiles'
SCHEMA_VERSION = 1
HEX64 = re.compile('[0-9a-f]{64}')
ROOMS = ('class_room', 'complex_room', 'dampened_room', 'hallway')

# key -> (importable module, extra files bound as files). Shell orchestration has no
# Python entry, so it is bound as its own file alone (exp_04's launcher pattern).
CODE_SOURCES = {
    'trainer': ('tools.exp06_train', ()),
    'finalize': ('tools.exp06_finalize', ()),
    'encoder': ('model.cylindrical_vit_oriented', ()),
    'factory': ('model.xRIR_cyl_oriented', ()),
    'heading': ('tools.exp06_heading', ()),
    'recipe': ('tools.exp06_recipe', ()),
    'probe_align': ('tools.exp06_probe_align', ()),
    'eval': ('tools.exp06_eval', ()),
    'eval_launch': ('tools.exp06_eval_launch', ()),
    'haa_finetune': ('tools.exp06_haa_finetune', ()),
    'haa_eval': ('tools.exp06_haa_eval', ()),
    'mirror_probe': ('tools.exp06_mirror_probe', ()),
    'summarize_haa': ('tools.exp06_summarize_haa', ()),
    'bootstrap': ('tools.exp06_bootstrap', ()),
    'compare': ('tools.exp06_compare', ()),
    'smoke': ('tools.exp06_smoke', ()),
    'launch_sh': (None, ('tools/exp06_launch.sh',)),
    'haa_pipeline_sh': (None, ('tools/exp06_haa_pipeline.sh',)),
    'evaluator_exp03': ('eval_yaw_rotation', ()),
}
CODE_KEYS = tuple(sorted(CODE_SOURCES))
# The approvals module's own closure: present in the committed template, not a round-3b
# requirement, and never a producer's input here.
OPTIONAL_CODE_KEYS = ('profiles',)

REUSED_DIGESTS = ('exp02_stats_sha256', 'exp02_summary_sha256',
                  'exp04_approved_digests_sha256', 'exp04_evaluator_closure',
                  'exp04_writer_closure', 'exp05_approved_digests_sha256')
REUSED_KEYS = REUSED_DIGESTS + ('legacy_receipt',)
REUSED_ALIASES = {'approved_digests_exp04': 'exp04_approved_digests_sha256',
                  'approved_digests_exp05': 'exp05_approved_digests_sha256',
                  'evaluator_exp04': 'exp04_evaluator_closure',
                  'writer_exp04': 'exp04_writer_closure'}
ARTIFACT_KEYS = ('epoch_012', 'gate_g1_sha256', 'heading')
ARTIFACT_ALIASES = {'gate_g1': 'gate_g1_sha256'}
SECTIONS = ('code', 'reused', 'artifacts')

# 6.4's matrix. Each producer requires the listed approvals and never its own outputs.
PRODUCER_REQUIREMENTS = {
    'heading': ('code',),
    'legacy_receipt': ('code',),
    'mirror_probe': ('code', 'artifacts.epoch_012', 'artifacts.heading'),
    'haa_children': ('code', 'artifacts.epoch_012', 'artifacts.heading'),
    'summarize_haa': ('code', 'reused', 'artifacts'),
    'sim_eval': ('code', 'artifacts.epoch_012'),
    'compare': ('code', 'reused'),
    'pages': (),
}
# What each producer writes; nothing here may appear in its own requirements.
PRODUCER_OUTPUTS = {
    'heading': ('artifacts.heading',),
    'legacy_receipt': ('reused.legacy_receipt',),
    'mirror_probe': ('artifacts.gate_g1_sha256',),
    'haa_children': (),
    'summarize_haa': (),
    'sim_eval': (),
    'compare': (),
    'pages': (),
}
# Evidence beyond the approvals each producer binds as well (run directories, not pins).
PRODUCER_EVIDENCE = {
    'summarize_haa': ('haa job completion.json of every new-arm child',),
    'compare': ('completion.json of every simulated evaluation run',),
    'pages': ("the summariser's and comparer's hash-bound outputs",),
}


class ApprovalsUnavailable(ValueError):
    """The approvals module is not importable from this branch."""


def approvals_module():
    """Import ``tools.exp06_profiles`` lazily; its absence is a named refusal."""
    try:
        return importlib.import_module(MODULE)
    except ImportError as error:
        raise ApprovalsUnavailable(
            'approvals module not available on this branch: {} ({})'.format(MODULE, error))


def _hex_or_null(value):
    return value is None or (type(value) is str and bool(HEX64.fullmatch(value)))


def _path_or_null(value):
    return value is None or (type(value) is str and bool(value))


def _epoch_or_null(value):
    return value is None or (type(value) is int and not isinstance(value, bool) and value > 0)


LEAF_RULES = {'sha256': _hex_or_null, 'path': _path_or_null, 'epoch': _epoch_or_null}


def _section(value, name, keys, aliases, optional=()):
    """One normalised section: known keys only, aliases resolved, nothing dropped."""
    if type(value) is not dict:
        raise ValueError('approved digests: section {} is not a record'.format(name))
    resolved = {}
    for key, item in sorted(value.items()):
        canonical = aliases.get(key, key)
        if canonical not in keys and canonical not in optional:
            raise ValueError('approved digests: unknown key {}.{}'.format(name, key))
        if canonical in resolved:
            raise ValueError('approved digests: {}.{} is given twice'.format(name, canonical))
        resolved[canonical] = item
    missing = [key for key in keys if key not in resolved]
    if missing:
        raise ValueError('approved digests: {} is missing {}'.format(name, ', '.join(missing)))
    return resolved


def _record(value, label, rules):
    if type(value) is not dict or set(value) != set(rules):
        raise ValueError('approved digests: {} must record {}'.format(
            label, ', '.join(sorted(rules))))
    for key, rule in sorted(rules.items()):
        if not rule(value[key]):
            raise ValueError('approved digests: {}.{} is {!r}'.format(label, key, value[key]))
    return dict(value)


def validate(value):
    """Return the canonical mapping; unknown keys and malformed leaves are refused."""
    if type(value) is not dict or set(value) != set(SECTIONS) | {'schema_version'}:
        raise ValueError('approved digests: expected schema_version and '
                         + ', '.join(SECTIONS))
    if value['schema_version'] != SCHEMA_VERSION:
        raise ValueError('approved digests: schema_version {!r}'.format(value['schema_version']))
    code = _section(value['code'], 'code', CODE_KEYS, {}, OPTIONAL_CODE_KEYS)
    for key, item in sorted(code.items()):
        if not _hex_or_null(item):
            raise ValueError('approved digests: code.{} is {!r}'.format(key, item))
    reused = _section(value['reused'], 'reused', REUSED_KEYS, REUSED_ALIASES)
    for key in REUSED_DIGESTS:
        if not _hex_or_null(reused[key]):
            raise ValueError('approved digests: reused.{} is {!r}'.format(key, reused[key]))
    reused['legacy_receipt'] = _record(reused['legacy_receipt'], 'reused.legacy_receipt',
                                      {'path': _path_or_null, 'sha256': _hex_or_null})
    artifacts = _section(value['artifacts'], 'artifacts', ARTIFACT_KEYS, ARTIFACT_ALIASES)
    if not _hex_or_null(artifacts['gate_g1_sha256']):
        raise ValueError('approved digests: artifacts.gate_g1_sha256 is {!r}'.format(
            artifacts['gate_g1_sha256']))
    artifacts['epoch_012'] = _record(artifacts['epoch_012'], 'artifacts.epoch_012', LEAF_RULES)
    artifacts['heading'] = _record(artifacts['heading'], 'artifacts.heading',
                                  {room: _hex_or_null for room in ROOMS})
    return {'schema_version': SCHEMA_VERSION, 'code': code, 'reused': reused,
            'artifacts': artifacts}


def load_approved_digests(path=None, module=None):
    """Delegate to the approvals module, then apply round 3b's own schema."""
    module = approvals_module() if module is None else module
    approved, receipt = module.load_approved_digests(path)
    return validate({key: {name: item for name, item in value.items()}
                     if isinstance(value, dict) else value
                     for key, value in dict(approved).items()}), receipt

