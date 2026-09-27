"""exp_11's per-arm admission adapter, the adapter cue, and phases 1b and 2.

Codex round-2 change 4: a per-arm code-key mapping is not enough -- job admission also
invokes a specification loader, a child verifier, a role resolver and a lineage
function. Historical arms keep exp_06's; H/I/J/K use exp_11's. The serialized run types
differ (``haa_job`` vs ``exp11_haa_job``) while the semantic roles the protocol checks
use are the same path-derived ``haa_train``/``haa_eval``.
"""
import json
from pathlib import Path

import pytest

from tools import exp06_finalize as legacy_finalizer
from tools import exp06_summarize_haa as subject
from tools import exp11_finalize as exp11_finalizer

from test_exp06_summarize_haa import (  # noqa: F401  (fixtures used by name)
    HEADING, NEW_OFFSETS, ROOMS, SIZE, arms, build_cache, cache_root, legacy_root,
    room_frame_child, stub_new_arms, synthetic_arm)

H, I, J, K = 'simple_or', 'simple_or_yaw', 'control_adapter', 'yawaug_adapter'
A, C, E, G = 'control', 'cyl_or', 'yawaug', 'yawaug_hf'


# --- the registry -----------------------------------------------------------------


def test_the_four_new_arms_are_registered_with_their_frames_and_cues():
    expected = {H: ('H', 'simple_oriented', 'heading', 'planes'),
                I: ('I', 'simple_oriented', 'heading', 'planes'),
                J: ('J', 'simple_adapter', 'room', 'adapter'),
                K: ('K', 'simple_adapter', 'room', 'adapter')}
    for name, (label, backbone, frame, cue) in expected.items():
        arm = subject.ARMS[name]
        assert (arm['label'], arm['backbone'], arm['frame']) == (label, backbone, frame)
        assert arm['cue'] == cue and arm['admission'] == 'exp11'
        assert arm['experiment'] == 'exp11' and arm['branch'] == 'new'
        assert arm['root'] == 'ckpt/exp11/sim2real/' + name
    assert subject.ADAPTER_ARMS == (J, K)
    assert subject.EXP04_AUG_ARMS == ('yawaug', 'yawaug_hf', K)
    assert subject.ARMS[J]['init_sha256'] is not None      # exp_01's control, a literal
    assert subject.ARMS[K]['init_sha256'] is None          # exp_04's, resolved from pins
    for name in ('control', 'cyl', C, 'control_hf', 'cyl_hf', E, G):
        assert 'admission' not in subject.ARMS[name] and 'cue' not in subject.ARMS[name]


def test_the_arms_are_read_under_exp11s_root():
    roots = {'exp06': 'a', 'exp09': 'b', 'exp11': 'c'}
    for name in (H, I, J, K):
        assert subject.arm_directory(name, roots) == Path('c') / name


# --- the admission adapter ---------------------------------------------------------


def test_each_arm_names_the_validator_family_that_finalised_it():
    for name in ('control', C, 'control_hf', 'cyl_hf', E, G):
        admission = subject.arm_admission(name)
        assert admission['name'] == 'exp06'
        assert admission['finalizer'] is legacy_finalizer
        assert admission['job_run_type'] == 'haa_job' and admission['extra_job_fields'] == ()
    for name in (H, I, J, K):
        admission = subject.arm_admission(name)
        assert admission['name'] == 'exp11'
        assert admission['finalizer'] is exp11_finalizer
        assert admission['job_run_type'] == 'exp11_haa_job'
        assert admission['extra_job_fields'] == ('adapter_heading', 'adapter_phi_deg')
    assert subject.ROLE_CODE_KEY == {'haa_train': 'haa_finetune', 'haa_eval': 'haa_eval'}
    # The semantic roles the protocol checks use are the same for both families.
    assert legacy_finalizer.child_role('stage1') == 'haa_train'
    assert exp11_finalizer.child_role('stage1') == 'exp11_haa_finetune'


def job_record(run_type='exp11_haa_job', **overrides):
    record = {key: 'x' for key in subject.JOB_FIELDS}
    record.update(schema_version=1, run_type=run_type, child_exit=0, owner_pid=1,
                  diagnostic=False, admissible_arm=True, expect='finetune',
                  job_spec={'path': 'j', 'sha256': 'a' * 64},
                  children={name: 'b' * 64
                            for name in legacy_finalizer.expected_children('finetune')},
                  backbone='simple_adapter', frame='room',
                  adapter_heading=HEADING, adapter_phi_deg=-90.0)
    record.update(overrides)
    return record


def write_job(tmp_path, record):
    (tmp_path / 'completion.json').write_text(json.dumps(record))
    return tmp_path


def test_the_job_run_type_and_extra_fields_are_the_arms_own(tmp_path):
    directory = write_job(tmp_path, job_record(run_dir=str(tmp_path.resolve())))
    record = subject.job_completion(directory, 'finetune', J)
    assert record['run_type'] == 'exp11_haa_job'
    # An exp_06 job record is refused for an exp_11 arm ...
    write_job(tmp_path, job_record(run_type='haa_job', run_dir=str(tmp_path.resolve())))
    with pytest.raises(ValueError, match="not the 'exp11_haa_job' of arm"):
        subject.job_completion(tmp_path, 'finetune', J)
    # ... and an exp_11 job record is refused for a historical arm.
    write_job(tmp_path, job_record(run_dir=str(tmp_path.resolve()), backbone='simple',
                                   frame='heading'))
    with pytest.raises(ValueError, match="not the 'haa_job' of arm"):
        subject.job_completion(tmp_path, 'finetune', G)
    # The adapter fields are required of an exp_11 arm and of nothing else.
    missing = job_record(run_dir=str(tmp_path.resolve()))
    missing.pop('adapter_phi_deg')
    write_job(tmp_path, missing)
    with pytest.raises(ValueError, match='missing adapter_phi_deg'):
        subject.job_completion(tmp_path, 'finetune', J)


# --- the adapter cue in the per-sample record --------------------------------------


def test_the_adapter_cue_is_admitted_only_for_the_arms_that_install_one(arms, tmp_path):
    per = arms[J]['per']['seed0']['hallway']
    assert per['meta']['frame'] == 'room' and per['meta']['heading'] is None
    assert per['meta']['adapter_heading'] == HEADING
    record = room_frame_child(tmp_path, per)
    assert subject.child_per_sample(tmp_path, 'eval/hallway', record, J)['index']
    # A historical room-frame arm may not carry one.
    with pytest.raises(ValueError, match='records an adapter cue'):
        subject.child_per_sample(tmp_path, 'eval/hallway', record, E)


def test_an_adapter_arm_without_a_cue_or_with_a_contradictory_one_is_refused(arms, tmp_path):
    per = arms[J]['per']['seed0']['hallway']
    record = room_frame_child(tmp_path, per, adapter_heading=None, adapter_phi_deg=None)
    with pytest.raises(ValueError, match='binds no heading'):
        subject.child_per_sample(tmp_path, 'eval/hallway', record, J)
    record = room_frame_child(tmp_path, per, adapter_phi_deg=0.0)
    with pytest.raises(ValueError, match='its bound records declare'):
        subject.child_per_sample(tmp_path, 'eval/hallway', record, J)
    record = room_frame_child(tmp_path, per, adapter_phi_deg='south')
    with pytest.raises(ValueError, match='installs the adapter heading'):
        subject.child_per_sample(tmp_path, 'eval/hallway', record, J)
    # The frame heading stays refused in the room frame, cue or no cue.
    record = room_frame_child(tmp_path, per, heading=HEADING)
    with pytest.raises(ValueError, match='records a heading in the room frame'):
        subject.child_per_sample(tmp_path, 'eval/hallway', record, J)


def test_the_arm_level_cue_must_agree_across_every_child():
    children = {'seed0/eval/hallway': {'adapter_heading': HEADING, 'adapter_phi_deg': -90.0},
                'seed1/eval/hallway': {'adapter_heading': HEADING, 'adapter_phi_deg': -90.0}}
    assert set(subject.arm_adapter_headings(children)) == set(ROOMS)
    children['seed1/eval/hallway']['adapter_phi_deg'] = 0.0
    with pytest.raises(ValueError, match='one heading for every room'):
        subject.arm_adapter_headings(children)
    assert subject.arm_adapter_headings({'a': {'adapter_heading': None}}) == {}


# --- the phases --------------------------------------------------------------------


def test_the_three_exp11_phases_are_registered_with_their_own_outputs():
    assert subject.PHASES == {'phase1': 'exp11', 'phase1b': 'exp11_phase1b',
                              'final': 'exp11_final'}
    assert subject.PUBLISHED_EXPERIMENTS == ('exp06', 'exp09', 'exp11')
    phase1b = subject.EXPERIMENTS['exp11_phase1b']
    assert phase1b['phase'] == 'phase1b'
    assert set(phase1b['arms']) == set(subject.EXP11_ARMS) | {J, K}
    assert [spec['name'] for spec in phase1b['exp11_decisions']] == ['Q1', 'Q2', 'Q3', 'Q4']
    assert phase1b['screens'] == ((J, A), (C, J), (K, E), (C, K))
    assert phase1b['outputs'] == ('ckpt/exp11/phase1b/stats.json',
                                  'ckpt/exp11/phase1b/summary.txt')
    final = subject.EXPERIMENTS['exp11_final']
    assert set(final['arms']) == set(subject.ARMS)
    assert [spec['name'] for spec in final['exp11_decisions']] == [
        'N1', 'N1i', 'N2', 'N3', 'Q1', 'Q2', 'Q3', 'Q4', 'P3', "P3'", 'P1', 'P2',
        'P4', "P4'"]
    assert len(final['screens']) == len(set(final['screens'])) == 13
    assert final['outputs'] == ('ckpt/exp11/stats.json', 'ckpt/exp11/summary.txt')


def test_every_registered_decision_names_a_registered_arm_and_only_allowed_fields():
    for key in ('exp11', 'exp11_phase1b', 'exp11_final'):
        config = subject.EXPERIMENTS[key]
        for spec in config['exp11_decisions']:
            members = spec['arms'] if spec.get('kind') == 'interaction' else spec['pair']
            assert set(members) <= set(config['arms']), (key, spec['name'])
            assert set(spec['fields']) <= set(subject.EXP11_FIELDS)
            assert spec['room'] == subject.H1_ROOM and spec['metric'] == subject.H1_METRIC
            assert spec['reading']
        for pair in config['screens']:
            assert set(pair) <= set(config['arms']), (key, pair)


def test_the_phase_resolver_and_the_output_guard():
    assert subject.experiment_key('exp11', 'phase1') == 'exp11'
    assert subject.experiment_key('exp11', 'final') == 'exp11_final'
    assert subject.experiment_key('exp06') == 'exp06'
    with pytest.raises(ValueError, match='--phase applies to exp_11'):
        subject.experiment_key('exp09', 'final')
    with pytest.raises(ValueError, match='unknown exp_11 phase'):
        subject.experiment_key('exp11', 'phase3')
    with pytest.raises(ValueError, match='canonical record of exp11_final'):
        subject.check_output_paths('exp11_phase1b', 'ckpt/exp11/stats.json', 'x.txt')
    with pytest.raises(ValueError, match='canonical record of exp11_phase1b'):
        subject.check_output_paths('exp11_final', 'a.json',
                                   'ckpt/exp11/phase1b/summary.txt')
    subject.check_output_paths('exp11_final', 'ckpt/exp11/stats.json',
                               'ckpt/exp11/summary.txt')


def test_the_parser_takes_the_phase_and_exp11s_approvals():
    parsed = subject.build_parser().parse_args([])
    assert parsed.phase == 'phase1' and parsed.exp11_approved is None
    assert subject.build_parser().parse_args(['--phase', 'final']).phase == 'final'
    with pytest.raises(SystemExit):
        subject.build_parser().parse_args(['--phase', 'phase2'])
    with pytest.raises(SystemExit):
        subject.build_parser().parse_args(['--experiment', 'exp11_final'])


# --- the external reference row ----------------------------------------------------


def test_the_external_row_is_copied_from_exp02s_canonical_record():
    inputs = {}
    rows = subject.external_rows(inputs=inputs)
    assert len(rows) == 1
    row = rows[0]
    assert row['name'] == "A'" and row['arm'] == 'released' and row['paired'] is False
    assert row['inference'] == 'none (external reference row)'
    assert row['source'] == 'ckpt/sim2real/stats.json' and len(row['source_sha256']) == 64
    assert set(row['cells']) == {'{}|{}'.format(room, key) for room, key in subject.CELLS}
    cell = row['cells']['hallway|c50']
    assert cell['selector'] == 'released|fine-tuned|hallway|c50'
    record = json.loads((subject.REPO / 'ckpt/sim2real/stats.json').read_text())
    assert cell['mean'] == record['rows'][cell['selector']]['mean']
    assert cell['std'] == record['rows'][cell['selector']]['std']
    assert str((subject.REPO / 'ckpt/sim2real/stats.json').resolve()) in inputs
    assert str((subject.REPO / 'ckpt/sim2real/summary.txt').resolve()) in inputs


def test_the_external_row_is_never_an_arm_and_never_paired():
    assert 'released' not in subject.ARMS
    for key in ('exp11_phase1b', 'exp11_final'):
        config = subject.EXPERIMENTS[key]
        assert config['external'] == subject.EXTERNAL_ROWS
        for spec in config['exp11_decisions']:
            members = spec['arms'] if spec.get('kind') == 'interaction' else spec['pair']
            assert 'released' not in members
    assert 'external' not in subject.EXPERIMENTS['exp11']
    assert 'external' not in subject.EXPERIMENTS['exp06']


def test_a_missing_or_changed_external_source_is_refused(tmp_path):
    with pytest.raises(ValueError, match='missing historical source'):
        subject.external_rows(repo=tmp_path)
