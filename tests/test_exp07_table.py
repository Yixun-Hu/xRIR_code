"""The exp_07 seen table producer: admission, refusals and canonical outputs."""
import json
from pathlib import Path

import pytest

from exp07_fixture import exp07_approval_template, exp07_fixture  # noqa: F401  (fixtures)
from tools import exp07_table as table
from tools import provenance as p


def test_the_fixture_is_the_launchers_run_layout(exp07_fixture):
    built = exp07_fixture()
    assert len(built.directories) == 40 == 4 * 2 * 5  # roles x K x seeds
    assert len(set(built.directories)) == len(built.directories)
    for directory in built.directories:
        assert sorted(item.name for item in Path(directory).iterdir()) == [
            'completion.json', 'eval_manifest.json', 'metrics_yaw.json', 'per_sample_yaw.json']
        fields = json.loads((Path(directory) / 'eval_manifest.json').read_text())
        assert fields['split'] == 'seen' and fields['confirmatory'] is True
        assert fields['mutable_inputs']['seen_split'] == dict(
            path=p.SEEN_SPLIT, sha256=p.sha256_file(built.split))
    released = [d for d in built.directories if 'released_seen' in d]
    assert len(released) == 10
    for directory in released:
        fields = json.loads((Path(directory) / 'eval_manifest.json').read_text())
        assert set(fields['mutable_inputs']) == {'seen_split'}  # no training provenance


@pytest.fixture
def built(exp07_fixture):
    return exp07_fixture()


def test_the_contract_accepts_a_trained_arm_and_the_reference_row(built):
    for role, reference in (('seen_simple', False), ('released_seen', True)):
        directory = built.paths[(role, 8)][0]
        arm = next(a for a in built.profile['arms'] if a['role'] == role)
        contract = table.run_contract(directory, arm, built.profile, built.pins)
        assert contract['role'] == role and contract['reference'] is reference
        assert (contract['training'] is None) is reference
        assert contract['waivers'] == [directory + ': mutable_inputs names']
        assert str(Path(directory) / 'eval_manifest.json') in contract['inputs']
        assert (str(built.attempts[role] / 'args.json') in contract['inputs']) is not reference


def admit(built, **kwargs):
    return table.admit(built.directories, built.profile, built.approved,
                       producer=built.producer, **kwargs)


def test_the_complete_fixture_is_admitted_role_by_role(built):
    groups, admitted = admit(built)
    assert admitted['deviations'] == []
    assert [(arm['role'], shot) for arm, shot, _ in groups] == [
        (role, shot) for role in ('seen_simple', 'seen_cyl', 'seen_aug', 'released_seen')
        for shot in (8, 1)]
    assert [len(runs) for runs in admitted['groups']] == [5] * 8
    contracts = admitted['contracts']
    assert len(contracts) == 40 and sum(c['reference'] for c in contracts.values()) == 10
    trained = {c['training'] for c in contracts.values() if not c['reference']}
    assert trained == {built.pins['closures']['training']}  # one closure for all three arms
    assert str(built.split.resolve()) in admitted['inputs']


def test_null_pins_refuse_but_are_listed_in_exploratory_mode(built, exp07_approval_template):
    built.approved = (exp07_approval_template, built.approved[1])
    with pytest.raises(ValueError, match='not yet approved'):
        admit(built)
    _, admitted = admit(built, exploratory=True)
    assert any('profile not yet approved: evaluator' in item for item in admitted['deviations'])
    assert any('seen_simple checkpoint' in item for item in admitted['deviations'])


def first(built, role='seen_simple', shot=8):
    return Path(built.paths[(role, shot)][0])


def rebind_training(built, role, args=None, manifest=None, completion=None):
    """Rewrite an arm's bound training artefacts and repair every digest that binds them."""
    attempt, digests = built.attempts[role], {}
    for name, filename, payload in (('train_args', 'args.json', args),
                                    ('train_manifest', 'train_manifest.json', manifest)):
        if payload is not None:
            digests[name] = built.replace(attempt / filename, payload)
    finished = completion if completion is not None else built.read(attempt / 'completion.json')
    finished['train_manifest_sha256'] = p.sha256_file(attempt / 'train_manifest.json')
    finished['outputs']['args.json'] = p.sha256_file(attempt / 'args.json')
    digests['train_completion'] = built.replace(attempt / 'completion.json', finished)
    for shot in (8, 1):
        for directory in built.paths[(role, shot)]:
            fields = built.read(Path(directory) / 'eval_manifest.json')
            for name, digest in digests.items():
                fields['mutable_inputs'][name]['sha256'] = digest
            built.rebind(Path(directory), manifest=fields)


def evaluated(built, role='seen_simple', shot=8, **changes):
    run = first(built, role, shot)
    fields = built.read(run / 'eval_manifest.json')
    fields.update(changes)
    built.rebind(run, manifest=fields)


def trained(built, role='seen_simple', **changes):
    args = dict(built.read(built.attempts[role] / 'args.json'), **changes)
    rebind_training(built, role, args=args)


def an_evaluated_split_that_is_not_seen(built):
    evaluated(built, split='unseen')
    return 'split identity'


def an_output_meta_from_another_split(built):
    run = first(built)
    sample = built.read(run / 'per_sample_yaw.json')
    sample['meta']['seen_split_sha256'] = 'f' * 64
    built.rebind(run, sample=sample)
    return 'split identity'


def a_binding_of_another_split_file(built):
    evaluated(built, mutable_inputs=dict(
        built.read(first(built) / 'eval_manifest.json')['mutable_inputs'],
        seen_split=dict(path='data/query.dat', sha256=p.sha256_file(built.root / 'data/query.dat'))))
    return 'seen_split binding'


def a_split_pickle_changed_after_the_run(built):
    built.split.write_bytes(b'a different seen_test_split.pkl')
    return 'revalidate seen_split'


def an_unapproved_evaluator_closure(built):
    built.pins['closures']['evaluator'] = 'f' * 64
    return 'unapproved evaluator'


def an_unapproved_writer_closure(built):
    built.pins['closures']['writer'] = 'f' * 64
    return 'unapproved writer'


def a_missing_training_binding(built):
    bindings = dict(built.read(first(built) / 'eval_manifest.json')['mutable_inputs'])
    bindings.pop('train_args')
    evaluated(built, mutable_inputs=bindings)
    return 'missing train_args binding'


def a_completion_that_certifies_another_checkpoint(built):
    completion = built.read(built.attempts['seen_simple'] / 'completion.json')
    completion['outputs']['epoch_012.pth'] = 'f' * 64
    rebind_training(built, 'seen_simple', completion=completion)
    return 'train_completion checkpoint epoch/digest'


def a_training_manifest_of_the_unseen_protocol(built):
    manifest = built.read(built.attempts['seen_cyl'] / 'train_manifest.json')
    manifest['train_data_identity']['protocol'] = 'unseen'
    rebind_training(built, 'seen_cyl', manifest=manifest)
    return 'training protocol'


def a_training_closure_that_is_not_the_pin(built):
    built.pins['closures']['training'] = 'f' * 64
    return 'training closure'


def a_launcher_outside_the_approved_list(built):
    built.pins['closures']['training_launcher'] = ['f' * 64]
    return 'training launcher closure'


def an_args_file_with_another_epoch_budget(built):
    trained(built, epochs=11)
    return 'args epochs'


def an_args_file_with_the_wrong_yaw_flag(built):
    trained(built, yaw_aug=1)
    return 'args yaw_aug'


def an_args_file_with_the_unseen_batch_count(built):
    trained(built, train_batches_per_epoch=9261)
    return 'args train_batches_per_epoch'


def a_released_row_claiming_training_provenance(built):
    binding = built.read(first(built) / 'eval_manifest.json')['mutable_inputs']['train_args']
    evaluated(built, role='released_seen', mutable_inputs=dict(
        built.read(first(built, 'released_seen') / 'eval_manifest.json')['mutable_inputs'],
        train_args=binding))
    return 'released row has no training provenance'


def a_training_binding_outside_the_attempt_directory(built):
    stray = built.root / 'train_manifest.json'
    stray.write_bytes((built.attempts['seen_aug'] / 'train_manifest.json').read_bytes())
    evaluated(built, role='seen_aug', mutable_inputs=dict(
        built.read(first(built, 'seen_aug') / 'eval_manifest.json')['mutable_inputs'],
        train_manifest=dict(path=str(stray), sha256=p.sha256_file(stray))))
    return 'train_manifest binding path'


def a_bounded_evaluation(built):
    evaluated(built, max_samples=16)
    return 'seen split coverage'


def a_checkpoint_and_shot_count_no_role_registers(built):
    evaluated(built, num_shot=3)
    return 'unregistered checkpoint/K'


REFUSALS = [an_evaluated_split_that_is_not_seen, an_output_meta_from_another_split,
            a_binding_of_another_split_file, a_split_pickle_changed_after_the_run,
            an_unapproved_evaluator_closure, an_unapproved_writer_closure,
            a_missing_training_binding, a_completion_that_certifies_another_checkpoint,
            a_training_manifest_of_the_unseen_protocol, a_training_closure_that_is_not_the_pin,
            a_launcher_outside_the_approved_list, an_args_file_with_another_epoch_budget,
            an_args_file_with_the_wrong_yaw_flag, an_args_file_with_the_unseen_batch_count,
            a_released_row_claiming_training_provenance,
            a_training_binding_outside_the_attempt_directory, a_bounded_evaluation,
            a_checkpoint_and_shot_count_no_role_registers]


@pytest.mark.parametrize('mutation', REFUSALS, ids=[item.__name__ for item in REFUSALS])
def test_admission_refuses_and_names_the_deviation(built, mutation):
    expected = mutation(built)
    with pytest.raises(ValueError, match='admission failed'):
        admit(built)
    _, admitted = admit(built, exploratory=True)
    assert any(expected in item for item in admitted['deviations']), admitted['deviations']
