"""The exp_07 seen table producer: admission, refusals and canonical outputs."""
import json
from pathlib import Path

import numpy as np
import pytest

from exp07_fixture import exp07_approval_template, exp07_fixture  # noqa: F401  (fixtures)
from tools import exp07_table as table
from tools import provenance as p
from tools.results_table import STAMP


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


def record(name, reviewed='e' * 64):
    return dict(path=name, reviewed_blob_sha256=reviewed, working_tree_sha256='e' * 64,
                commits_after_reviewed=[], mtime='2026-09-15T00:00:00+00:00')


def a_training_closure_with_no_file_records(built):
    manifest = built.read(built.attempts['seen_simple'] / 'train_manifest.json')
    manifest['source_closures']['training']['files'] = []
    rebind_training(built, 'seen_simple', manifest=manifest)
    return 'training closure records training'


def a_launcher_closure_record_that_was_never_reviewed(built):
    manifest = built.read(built.attempts['seen_cyl'] / 'train_manifest.json')
    manifest['source_closures']['launcher']['files'] = [record('tools/wrong.py', None)]
    rebind_training(built, 'seen_cyl', manifest=manifest)
    return 'training closure records launcher'


def a_training_closure_label_that_is_not_its_records(built):
    manifest = built.read(built.attempts['seen_aug'] / 'train_manifest.json')
    manifest['source_closures']['training']['files'].append(record('wrong.py'))
    rebind_training(built, 'seen_aug', manifest=manifest)
    return 'training closure digest training'


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


def an_args_file_from_another_capacity_tier(built):
    trained(built, vit_dim=256, vit_heads=4)
    return 'args tier configuration'


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
            a_training_closure_with_no_file_records, a_training_closure_label_that_is_not_its_records,
            a_launcher_closure_record_that_was_never_reviewed,
            a_launcher_outside_the_approved_list, an_args_file_with_another_epoch_budget,
            an_args_file_with_the_wrong_yaw_flag, an_args_file_with_the_unseen_batch_count,
            an_args_file_from_another_capacity_tier,
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


def test_source_drift_recorded_after_the_spawn_is_still_admitted(built):
    """A7: the pin is the reviewed identity; the launcher records later working-tree drift."""
    attempt = built.attempts['seen_simple']
    completion = built.read(attempt / 'completion.json')
    completion['source_drift_after_spawn'] = [dict(path='train_xRIR_backbone.py',
                                                   spawn_sha256='a' * 64, now_sha256='b' * 64)]
    rebind_training(built, 'seen_simple', completion=completion)
    _, admitted = admit(built)
    assert admitted['deviations'] == []


def build(built, **kwargs):
    return table.build_table(built.directories, built.profile, built.approved,
                             producer=built.producer, **kwargs)[0]


def test_the_table_reports_five_seed_means_per_role_and_shot_count(built):
    result = build(built)
    assert [(row['role'], row['num_shot']) for row in result['rows']] == [
        (role, shot) for role in ('seen_simple', 'seen_cyl', 'seen_aug', 'released_seen')
        for shot in (8, 1)]
    assert result['profile_name'] == 'TABLE_SEEN_V1' and result['deviations'] == []
    assert 'generated_at' not in json.dumps(result)  # the canonical JSON carries no timestamp
    row = next(r for r in result['rows'] if (r['role'], r['num_shot']) == ('seen_simple', 8))
    edt = row['metrics']['EDT']
    assert edt['unit'] == 'ms' and edt['source'] == 'edt'
    expected = []
    for directory in built.paths[('seen_simple', 8)]:
        values = built.read(Path(directory) / 'per_sample_yaw.json')['P']['0']['edt']
        seed = str(built.read(Path(directory) / 'eval_manifest.json')['manifest_seed'])
        assert edt['per_seed'][seed]['n_finite'] == 12
        assert edt['per_seed'][seed]['mean'] == pytest.approx(np.mean(values) * 1000)
        expected.append(np.mean(values) * 1000)
    assert edt['mean'] == pytest.approx(np.mean(expected))
    assert edt['sd'] == pytest.approx(np.std(expected, ddof=1))
    assert row['protocol']['n_queries'] == 12 and row['protocol']['split'] == 'seen'
    assert row['protocol']['seeds'] == [42, 43, 44, 45, 46] and row['protocol']['k'] == 0


def test_the_released_row_is_marked_a_reference_with_an_unknown_epoch(built):
    rows = [row for row in build(built)['rows'] if row['role'] == 'released_seen']
    assert len(rows) == 2 and all(row['reference'] and row['epoch'] is None for row in rows)
    assert all(row['protocol']['epoch'] == 'unknown (external)' for row in rows)
    assert all(row['protocol']['training'] == 'external' for row in rows)
    assert all('reference' in row['label'].lower() for row in rows)
    trained = [row for row in build(built)['rows'] if row['role'] != 'released_seen']
    assert all(row['epoch'] == 12 and row['protocol']['training'] == 'internal' for row in trained)


def test_a_seed_losing_too_many_queries_refuses_the_cell(built):
    directory = Path(built.paths[('seen_cyl', 1)][2])
    sample = built.read(directory / 'per_sample_yaw.json')
    sample['P']['0']['c50'][:3] = [None, None, None]
    metrics = built.read(directory / 'metrics_yaw.json')
    metrics['P'] = built.summaries(sample['P'])
    built.rebind(directory, sample=sample, metrics=metrics)
    with pytest.raises(ValueError, match='finite count tolerance exceeded'):
        build(built)


def publish(built, tmp_path, name, **kwargs):
    result, admitted = table.build_table(built.directories, built.profile, built.approved,
                                         producer=built.producer, **kwargs)
    target = tmp_path / name
    target.mkdir()
    table.write_outputs(result, admitted, str(target / 'table.json'),
                        str(target / 'model_comparison_seen.md'), command=['exp07_table.py'])
    return target


def test_publication_is_byte_stable_and_the_markdown_comes_from_the_json(built, tmp_path):
    first_run, second_run = (publish(built, tmp_path, name) for name in ('one', 'two'))
    assert (first_run / 'table.json').read_bytes() == (second_run / 'table.json').read_bytes()
    markdown = (first_run / 'model_comparison_seen.md').read_text()
    other = (second_run / 'model_comparison_seen.md').read_text()
    def body(text, run):  # the block names the canonical JSON it was rendered from
        return text.split(STAMP)[0].replace(str(run), 'RUN')
    assert body(markdown, first_run) == body(other, second_run)
    result = json.loads((first_run / 'table.json').read_text())
    for row in result['rows']:
        for name, metric in row['metrics'].items():
            assert '{:.6g} ± {:.6g}'.format(metric['mean'], metric['sd']) in markdown
        assert row['label'] in markdown
    assert 'seen' in markdown and '12 queries' in markdown
    sidecar = json.loads((first_run / 'table.json.provenance.json').read_text())
    assert sidecar['profile_digest'] == result['profile_digest'] and sidecar['generated_at']
    assert sidecar['approved_digests']['sha256'] == built.approved[1]['sha256']
    assert sidecar['outputs'][str((first_run / 'table.json').resolve())]


def test_the_cli_refuses_the_unapproved_committed_profile_and_writes_nothing(tmp_path, built):
    with pytest.raises(ValueError, match='not yet approved'):
        table.main(['--profile', 'TABLE_SEEN_V1', '--runs'] + built.directories +
                   ['--json', str(tmp_path / 'table.json'), '--md', str(tmp_path / 'seen.md')])
    assert not (tmp_path / 'table.json').exists() and not (tmp_path / 'seen.md').exists()


def test_the_cli_defaults_the_markdown_to_the_living_seen_table(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(table, 'build_table', lambda runs, exploratory: ({'ok': True}, {}))
    monkeypatch.setattr(table, 'write_outputs', lambda *a: captured.update(call=a))
    table.main(['--profile', 'TABLE_SEEN_V1', '--runs', 'a', '--json', str(tmp_path / 't.json')])
    assert table.MARKDOWN == table.REPO / 'worklog/worklog_yixun/model_comparison_seen.md'
    assert captured['call'][3] == str(table.MARKDOWN) and captured['call'][4] is False
    assert captured['call'][5][:4] == ['exp07_table.py', '--profile', 'TABLE_SEEN_V1', '--runs']
