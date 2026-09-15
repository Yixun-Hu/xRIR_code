"""Round 3b's approvals adapter: the schema of 6.4 and the per-producer matrix.

``tools.exp06_profiles`` lives on the training branch and is not importable here, so
``tools.exp06_approvals_api`` is exercised against stub approvals objects and the named
refusal it raises when the real module is absent.
"""
import copy
import json
import re

import pytest

from tools import exp06_approvals_api as subject

HEX = 'a' * 64


def template():
    """The all-null template of 6.4, in this round's canonical spelling."""
    return {'schema_version': 1,
            'code': {key: None for key in subject.CODE_KEYS},
            'reused': dict({key: None for key in subject.REUSED_DIGESTS},
                           legacy_receipt={'path': None, 'sha256': None}),
            'artifacts': {'epoch_012': {'path': None, 'epoch': None, 'sha256': None},
                          'heading': {room: None for room in subject.ROOMS},
                          'gate_g1_sha256': None}}


def filled(digest=HEX):
    value = template()
    value['code'] = {key: digest for key in subject.CODE_KEYS}
    value['reused'] = dict({key: digest for key in subject.REUSED_DIGESTS},
                           legacy_receipt={'path': 'ckpt/exp06/legacy_receipt.json',
                                           'sha256': digest})
    value['artifacts'] = {'epoch_012': {'path': 'ckpt/exp06/final/epoch_012.pth',
                                        'epoch': 12, 'sha256': digest},
                          'heading': {room: digest for room in subject.ROOMS},
                          'gate_g1_sha256': digest}
    return value


class StubApprovals(object):
    """What ``tools.exp06_profiles.load_approved_digests`` returns, without the file."""

    def __init__(self, value):
        self.value = value

    def load_approved_digests(self, path=None):
        return copy.deepcopy(self.value), {'path': str(path), 'sha256': HEX}


# --- the schema of 6.4 -----------------------------------------------------------------


def test_the_null_template_loads_and_keeps_every_section():
    value = subject.validate(template())
    assert set(value) == {'schema_version', 'code', 'reused', 'artifacts'}
    assert set(value['code']) == set(subject.CODE_KEYS)
    assert set(value['reused']) == set(subject.REUSED_KEYS)
    assert set(value['artifacts']) == set(subject.ARTIFACT_KEYS)
    assert all(item is None for item in value['code'].values())


def test_a_filled_template_validates():
    value = subject.validate(filled())
    assert value['artifacts']['epoch_012']['epoch'] == 12
    assert value['reused']['legacy_receipt']['sha256'] == HEX
    assert all(re.fullmatch('[0-9a-f]{64}', item) for item in value['code'].values())


def test_probe_align_is_a_required_code_key():
    """§6.4 lists the alignment helper the mirror probe's forward is recomposed from."""
    assert 'probe_align' in subject.CODE_KEYS
    assert subject.CODE_SOURCES['probe_align'][0] == 'tools.exp06_probe_align'
    value = template()
    del value['code']['probe_align']
    with pytest.raises(ValueError, match='missing probe_align'):
        subject.validate(value)


def test_the_committed_spellings_are_accepted_and_normalised():
    """The record template's own key names resolve to this round's canonical ones."""
    value = template()
    for committed, canonical in subject.REUSED_ALIASES.items():
        value['reused'][committed] = value['reused'].pop(canonical)
    value['artifacts']['gate_g1'] = value['artifacts'].pop('gate_g1_sha256')
    value['code']['profiles'] = None
    got = subject.validate(value)
    assert set(got['reused']) == set(subject.REUSED_KEYS)
    assert 'gate_g1_sha256' in got['artifacts'] and 'gate_g1' not in got['artifacts']
    assert got['code']['profiles'] is None


MALFORMED = {
    'code_not_hex': lambda v: v['code'].update(bootstrap='not a digest'),
    'code_short': lambda v: v['code'].update(bootstrap='ab' * 20),
    'code_unknown': lambda v: v['code'].update(mystery=None),
    'reused_not_hex': lambda v: v['reused'].update(exp02_stats_sha256=7),
    'reused_unknown': lambda v: v['reused'].update(mystery=None),
    'receipt_not_record': lambda v: v['reused'].update(legacy_receipt=HEX),
    'receipt_missing_key': lambda v: v['reused']['legacy_receipt'].pop('path'),
    'receipt_empty_path': lambda v: v['reused']['legacy_receipt'].update(path=''),
    'epoch_zero': lambda v: v['artifacts']['epoch_012'].update(epoch=0),
    'epoch_bool': lambda v: v['artifacts']['epoch_012'].update(epoch=True),
    'epoch_extra': lambda v: v['artifacts']['epoch_012'].update(role='arm'),
    'heading_room': lambda v: v['artifacts']['heading'].update(kitchen=None),
    'heading_missing': lambda v: v['artifacts']['heading'].pop('hallway'),
    'gate_not_hex': lambda v: v['artifacts'].update(gate_g1_sha256='x' * 64),
    'section_missing': lambda v: v.pop('reused'),
    'section_extra': lambda v: v.update(checkpoints={}),
    'schema_version': lambda v: v.update(schema_version=2),
}


@pytest.mark.parametrize('case', sorted(MALFORMED))
def test_a_malformed_leaf_or_section_is_refused(case):
    value = filled()
    MALFORMED[case](value)
    with pytest.raises(ValueError):
        subject.validate(value)


# --- the lazy import -------------------------------------------------------------------


def test_the_absent_approvals_module_is_a_named_refusal(monkeypatch):
    """The merge brought the module; a checkout without it still refuses by name."""
    monkeypatch.setattr(subject, 'MODULE', 'tools.exp06_profiles_absent_here')
    with pytest.raises(subject.ApprovalsUnavailable, match='not available on this branch'):
        subject.approvals_module()
    with pytest.raises(subject.ApprovalsUnavailable):
        subject.load_approved_digests()
    assert issubclass(subject.ApprovalsUnavailable, ValueError)


def test_the_merged_module_is_the_approvals_this_round_reads():
    """The real committed template loads through the adapter's own schema."""
    from tools import exp06_profiles, provenance
    approved, receipt = subject.load_approved_digests()
    assert set(approved['code']) == set(subject.CODE_KEYS) | {'profiles'}
    assert approved['code']['evaluator_exp03'] == exp06_profiles.EVALUATOR_EXP03
    assert set(approved['reused']) == set(subject.REUSED_KEYS)
    assert set(approved['artifacts']) == set(subject.ARTIFACT_KEYS)
    assert receipt['sha256'] == provenance.sha256_file(exp06_profiles.TEMPLATE_PATH)
    with pytest.raises(ValueError, match='approvals incomplete'):
        subject.require_producer(approved, 'summarize_haa')


def test_a_stub_module_is_delegated_to_and_revalidated(tmp_path):
    path = tmp_path / 'approved_digests.json'
    path.write_text(json.dumps(filled()))
    approved, receipt = subject.load_approved_digests(path, module=StubApprovals(filled()))
    assert approved == subject.validate(filled()) and receipt['sha256'] == HEX
    with pytest.raises(ValueError):
        subject.load_approved_digests(path, module=StubApprovals({'schema_version': 1}))


# --- the per-producer matrix and the digests that fill the code section ------------------


def test_require_lists_every_unapproved_leaf_and_raises_in_production():
    null = subject.validate(template())
    deviations = subject.require(null, ('code',), exploratory=True)
    assert deviations == ['not approved: code.' + key for key in subject.CODE_KEYS]
    with pytest.raises(ValueError, match='approvals incomplete'):
        subject.require(null, ('code',))
    assert subject.require(subject.validate(filled()), subject.SECTIONS) == []
    with pytest.raises(ValueError, match='unknown approvals section'):
        subject.require(null, ('checkpoints',))


def test_require_reaches_the_leaves_of_a_named_subsection():
    value = subject.validate(filled())
    value['artifacts']['heading']['hallway'] = None
    assert subject.require(value, ('artifacts.epoch_012',)) == []
    assert subject.require(value, ('artifacts.heading',), exploratory=True) == [
        'not approved: artifacts.heading.hallway']
    assert subject.require(value, ('artifacts.heading.class_room',)) == []


@pytest.mark.parametrize('producer', sorted(subject.PRODUCER_REQUIREMENTS))
def test_no_producer_requires_its_own_outputs(producer):
    required = set(subject.leaf_paths(subject.producer_sections(producer)))
    for output in subject.PRODUCER_OUTPUTS[producer]:
        assert not any(path == output or path.startswith(output + '.') for path in required)


def test_the_matrix_is_the_one_section_6_4_registers():
    assert subject.producer_sections('mirror_probe') == (
        'code', 'artifacts.epoch_012', 'artifacts.heading')
    assert subject.producer_sections('summarize_haa') == ('code', 'reused', 'artifacts')
    assert subject.producer_sections('sim_eval') == ('code', 'artifacts.epoch_012')
    assert subject.producer_sections('compare') == ('code', 'reused')
    assert subject.producer_sections('heading') == ('code',)
    with pytest.raises(ValueError, match='unknown producer'):
        subject.producer_sections('nobody')
    assert set(subject.PRODUCER_OUTPUTS) == set(subject.PRODUCER_REQUIREMENTS)
    assert set(subject.PRODUCER_EVIDENCE) <= set(subject.PRODUCER_REQUIREMENTS)


def test_require_producer_refuses_the_null_template_for_every_producer():
    null = subject.validate(template())
    for producer in sorted(subject.PRODUCER_REQUIREMENTS):
        if subject.producer_sections(producer):
            with pytest.raises(ValueError):
                subject.require_producer(null, producer)
        assert subject.require_producer(subject.validate(filled()), producer) == []


def test_every_code_key_hashes_to_a_closure_digest_of_this_worktree():
    """The helper the fill commit runs: one 64-hex digest per registered code key."""
    import subprocess
    repo = subject.__file__.rsplit('/tools/', 1)[0]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
                                     text=True).strip()
    digests = subject.compute_code_digests(repo, commit, keys=('probe_align', 'launch_sh'))
    assert set(digests) == {'probe_align', 'launch_sh'}
    assert all(re.fullmatch('[0-9a-f]{64}', value) for value in digests.values())
    assert digests['probe_align'] != digests['launch_sh']
    with pytest.raises(ValueError, match='unknown code key'):
        subject.code_digest('mystery', repo, commit)


def test_a_shell_key_binds_its_own_file_and_no_python_closure():
    assert subject.CODE_SOURCES['launch_sh'] == (None, ('tools/exp06_launch.sh',))
    assert subject.CODE_SOURCES['haa_pipeline_sh'] == (None, ('tools/exp06_haa_pipeline.sh',))
    assert subject.CODE_SOURCES['evaluator_exp03'][0] == 'eval_yaw_rotation'
