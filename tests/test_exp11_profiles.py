"""exp_11's own approvals: keys, schema, committed-blob binding, requirement matrix.

Plan v3 section 4.7 and Codex round-2 change 5. exp_11 owns a separate approvals record;
``tools/exp06_profiles.py`` and ``tools/exp06_approvals_api.py`` are read-only here and
must stay byte-identical. The requirement matrix consumes only available inputs:
pretraining never requires its own output pin, Phase 1b runs while both H/I pins are
null, H runs while I's pin is null, and only the final publication requires every arm.
"""
import hashlib
import json
import re
from pathlib import Path
import subprocess

import pytest

from tools import exp11_profiles as profiles

REPO = Path(__file__).resolve().parents[1]


def template():
    return json.loads(profiles.TEMPLATE_PATH.read_text())


def filled(**sections):
    """A template with every requested leaf set to a distinct plausible digest."""
    value = template()
    for name, keys in sections.items():
        for key in keys:
            digest = hashlib.sha256('{}.{}'.format(name, key).encode()).hexdigest()
            if name == 'artifacts':
                value[name][key] = {'epoch': 12, 'path': 'ckpt/exp11/' + key,
                                    'sha256': digest}
            elif key == 'legacy_receipt':
                value[name][key] = {'path': 'ckpt/exp06/legacy_receipt.json',
                                    'sha256': digest}
            else:
                value[name][key] = digest
    return value


def write(tmp_path, value, name='approved_digests.json'):
    path = tmp_path / name
    path.write_text(json.dumps(value, indent=2) + '\n')
    return path


def test_the_committed_template_is_all_null_and_carries_every_key():
    value = template()
    assert value['schema_version'] == profiles.SCHEMA_VERSION
    assert sorted(value['code']) == sorted(profiles.CODE_KEYS)
    assert sorted(value['reused']) == sorted(profiles.REUSED_KEYS)
    assert sorted(value['artifacts']) == sorted(profiles.ARTIFACT_KEYS)
    assert all(item is None for item in value['code'].values())
    assert all(item is None for item in value['reused'].values()
               if not isinstance(item, dict))
    for key in profiles.ARTIFACT_KEYS:
        assert value['artifacts'][key] == {'epoch': None, 'path': None, 'sha256': None}
    approved, identity = profiles.load_approved_digests(profiles.TEMPLATE_PATH)
    assert identity['sha256'] == hashlib.sha256(
        profiles.TEMPLATE_PATH.read_bytes()).hexdigest()
    assert approved['code']['train'] is None


def test_code_keys_are_exactly_the_eight_the_plan_registers():
    assert sorted(profiles.CODE_KEYS) == sorted(
        ('train', 'finalize', 'haa_finetune', 'haa_eval', 'haa_pipeline_sh', 'launch_sh',
         'smoke', 'summarize_haa'))
    # The launcher sources the pinned exp_06 lifecycle library and starts the lock
    # holder that owns the arm's publication lock; all three files are bound.
    assert profiles.CODE_SPECS['launch_sh'] == (
        None, ('tools/exp11_launch.sh', 'tools/exp06_launch.sh',
               'tools/exp11_lock_holder.py', 'tools/exp11_pidrecord.py',
               'tools/exp11_pathprobe.py'))
    assert profiles.CODE_SPECS['haa_pipeline_sh'] == (None, ('tools/exp11_haa_pipeline.sh',))
    assert profiles.CODE_SPECS['summarize_haa'] == ('tools.exp06_summarize_haa', ())


def test_the_launcher_binds_every_module_it_runs():
    """A shell closure is only as honest as the list of files the shell actually runs.

    The launcher has no imports for a closure walker to follow: what it runs, it runs as
    a subprocess. Every ``tools/exp11_*`` module named in its text must therefore be in
    ``launch_sh``'s spec, or a change to that module's behaviour -- the pid-record
    grammar, say -- would leave the launcher's approved digest standing (close review
    10, blocker 1).
    """
    text = (REPO / 'tools/exp11_launch.sh').read_text()
    named = {'tools/{}.py'.format(module.replace('.', '/').split('/')[-1])
             for module in re.findall(r'tools\.(exp11_[a-z_0-9]+)', text)}
    named |= set(re.findall(r'tools/exp11_[a-z_0-9]+\.py', text))
    bound = set()
    for module, extra in profiles.CODE_SPECS.values():
        if module:                      # a key of its own: bound through its closure
            bound.add(module.replace('.', '/') + '.py')
        bound.update(extra)
    assert named, 'the launcher runs at least one module of its own'
    assert named <= bound, 'no approval key covers: {}'.format(sorted(named - bound))
    # The two helpers with no key of their own are the launcher's, and only
    # `launch_sh` can bind them.
    for helper in ('tools/exp11_lock_holder.py', 'tools/exp11_pidrecord.py',
                   'tools/exp11_pathprobe.py'):
        assert helper in profiles.CODE_SPECS['launch_sh'][1]


def test_schema_refuses_unknown_keys_and_malformed_leaves(tmp_path):
    for mutate in (lambda v: v['code'].update(mirror_probe=None),
                   lambda v: v['code'].pop('train'),
                   lambda v: v.update(schema_version=2),
                   lambda v: v['code'].update(train='not-a-digest'),
                   lambda v: v['artifacts'].update(simpor_epoch_012=None),
                   lambda v: v['reused'].update(legacy_receipt='deadbeef')):
        value = template()
        mutate(value)
        with pytest.raises(ValueError):
            profiles.load_approved_digests(write(tmp_path, value))


def test_committed_blob_binding(tmp_path):
    """Only the bytes a reviewer committed at that commit admit a confirmatory run."""
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO),
                                   text=True).strip()
    approved, identity = profiles.load_approved_digests(profiles.APPROVED_DIGESTS_PATH,
                                                        repo=REPO, commit=head)
    assert identity['committed_at'] == head
    assert identity['repo_relative'] == profiles.APPROVED_RELATIVE
    with pytest.raises(ValueError, match='outside the repository'):
        profiles.load_approved_digests(write(tmp_path, template()), repo=REPO, commit=head)
    with pytest.raises(ValueError, match='both the repository and the commit'):
        profiles.load_approved_digests(profiles.APPROVED_DIGESTS_PATH, repo=REPO)
    blob, at_commit = profiles.approvals_at_commit(profiles.APPROVED_DIGESTS_PATH, REPO, head)
    assert blob['code']['train'] is None and at_commit['committed_at'] == head
    assert at_commit['sha256'] == identity['sha256']


def test_code_digests_are_computed_only_for_present_keys():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO),
                                   text=True).strip()
    notes = []
    digests = profiles.compute_code_digests(REPO, head, notes=notes)
    assert 'summarize_haa' in digests and len(digests['summarize_haa']) == 64
    assert set(digests) | {note.split(':')[0] for note in notes} == set(profiles.CODE_KEYS)
    assert digests['summarize_haa'] == profiles.code_digest('summarize_haa', str(REPO), head)
    with pytest.raises(ValueError, match='unknown approval key'):
        profiles.code_digest('mirror_probe', str(REPO), head)


@pytest.mark.parametrize('producer', sorted(profiles.PRODUCER_REQUIREMENTS))
def test_no_producer_requires_its_own_output(producer):
    required = set(profiles.producer_leaves(producer))
    produced = set(profiles.PRODUCER_OUTPUTS[producer])
    assert not (required & produced), producer
    assert set(profiles.PRODUCER_CODE_KEYS[producer]) <= set(profiles.CODE_KEYS)


def test_phase_1b_and_h_run_while_the_later_pins_are_null():
    code = filled(code=profiles.CODE_KEYS, reused=profiles.REUSED_KEYS)
    for producer in ('haa_control_adapter', 'haa_yawaug_adapter', 'summarize_phase1b'):
        assert profiles.require_producer(code, producer) == []
    with_h = json.loads(json.dumps(code))
    with_h['artifacts']['simpor_epoch_012'] = {
        'epoch': 12, 'path': 'p', 'sha256': 'a' * 64}
    assert profiles.require_producer(with_h, 'haa_simple_or') == []
    assert profiles.require_producer(code, 'pretrain_simple_or') == []
    assert profiles.require_producer(code, 'pretrain_simple_or_yaw') == []
    for producer in ('haa_simple_or', 'haa_simple_or_yaw', 'summarize_final'):
        deviations = profiles.require_producer(code, producer, exploratory=True)
        assert deviations, producer
        with pytest.raises(ValueError, match='approvals incomplete'):
            profiles.require_producer(code, producer)


def test_the_final_publication_requires_every_planned_arm():
    everything = filled(code=profiles.CODE_KEYS, reused=profiles.REUSED_KEYS,
                        artifacts=profiles.ARTIFACT_KEYS)
    assert profiles.require_producer(everything, 'summarize_final') == []
    for key in profiles.ARTIFACT_KEYS:
        missing = json.loads(json.dumps(everything))
        missing['artifacts'][key]['sha256'] = None
        assert profiles.require_producer(missing, 'summarize_final', exploratory=True) == [
            'not approved: artifacts.{}.sha256'.format(key)]
    with pytest.raises(ValueError, match='unknown producer'):
        profiles.require_producer(everything, 'pages')


def test_require_reports_code_drift_against_this_checkout():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO),
                                   text=True).strip()
    current = profiles.compute_code_digests(REPO, head, keys=('summarize_haa',))
    approved = {'code': {'summarize_haa': current['summarize_haa']}}
    assert profiles.require(approved, ('summarize_haa',), repo=REPO, commit=head,
                            current=current) == []
    wrong = {'code': {'summarize_haa': 'b' * 64}}
    assert profiles.require(wrong, ('summarize_haa',), repo=REPO, commit=head,
                            exploratory=True, current=current)
    with pytest.raises(ValueError, match='do not admit this run'):
        profiles.require(wrong, ('summarize_haa',), repo=REPO, commit=head, current=current)


def test_exp06_approvals_modules_are_untouched():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO), text=True).strip()
    for name in ('tools/exp06_profiles.py', 'tools/exp06_approvals_api.py'):
        blob = subprocess.check_output(['git', 'cat-file', '-p', '{}:{}'.format(head, name)],
                                       cwd=str(REPO))
        assert hashlib.sha256(blob).hexdigest() == hashlib.sha256(
            (REPO / name).read_bytes()).hexdigest()
