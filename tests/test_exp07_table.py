"""The exp_07 seen table producer: admission, refusals and canonical outputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from exp07_fixture import exp07_approval_template, exp07_fixture  # noqa: F401  (fixtures)
from test_exp04_profiles import approval_repo
from tools import exp07_table as table
from tools import exp07_provenance as e7p
from tools import provenance as p
from tools.exp07_profiles import load_approved_digests
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
            path=e7p.SEEN_SPLIT, sha256=p.sha256_file(built.split))
    released = [d for d in built.directories if 'released_seen' in d]
    assert len(released) == 10
    for directory in released:
        fields = json.loads((Path(directory) / 'eval_manifest.json').read_text())
        assert set(fields['mutable_inputs']) == {'seen_split'}  # no training provenance


def test_the_fixture_carries_the_full_runs_training_evidence(exp07_fixture):
    built = exp07_fixture()
    for role in ('seen_simple', 'seen_cyl', 'seen_aug'):
        attempt = built.attempts[role]
        manifest = built.read(attempt / 'train_manifest.json')
        assert manifest['mode'] == 'full' and manifest['allow_dirty'] is False
        identity, sidecar = manifest['train_data_identity'], attempt / 'train_inventory.json'
        assert identity['inventory_file'] == dict(path=str(sidecar), sha256=p.sha256_file(sidecar))
        records = built.read(sidecar)['inventory']
        assert p._inventory_digest(records) == identity['inventory_sha256']
        assert identity['inventory_files'] == len(records) == built.profile['train_inventory_files']
        receipt = built.read(receipt_path(attempt))
        limits = manifest['timing_limits']
        assert limits['probe_receipt_sha256'] == p.sha256_file(receipt_path(attempt))
        assert limits['projection_hours'] == receipt['T_run'] / 3600 <= 60
        assert limits['ceiling_hours'] == 1.5 * limits['projection_hours']
        assert limits['epoch_seconds'] == 1.05 * receipt['T_epoch']
        ledger = built.read(attempt.parent / 'cumulative_hours.json')
        # Every arm probed once; seen_simple also retried after a slow first epoch.
        expected = ['full', 'probe'] + (['full'] if role == 'seen_simple' else [])
        assert sorted(row['mode'] for row in ledger['attempts']) == sorted(expected)
        assert any(row['attempt'] == attempt.name and row['mode'] == 'full'
                   for row in ledger['attempts'])
        args = built.read(attempt / 'args.json')
        assert manifest['effective_args'] != args  # the launcher normalises the env keys
        assert manifest['effective_args']['PYTHONHASHSEED'] == args['env']['PYTHONHASHSEED']
        listing = built.read(attempt / 'completion.json')['directory_listing']
        assert {'epoch_%03d.pth' % epoch for epoch in range(1, 13)} <= set(listing)


def test_the_fixtures_attempts_revalidate_as_the_binder_reads_them(exp07_fixture):
    """The reviewed blobs resolve, so tools.provenance can ask for post-spawn drift."""
    built = exp07_fixture()
    for role in ('seen_simple', 'seen_cyl', 'seen_aug'):
        fields = built.read(built.attempts[role] / 'train_manifest.json')
        drift = []
        assert p.revalidate(fields, required=('repo', 'source_closures', 'mutable_inputs',
                                              'train_data_identity', 'effective_args'),
                            source_drift=drift) == []
        assert drift == [] and len(fields['reviewed_commit']) == 40


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
    for name in ('args.json', 'train_manifest.json'):
        if name in finished['outputs']:  # the entries this rewrite invalidated
            finished['outputs'][name] = p.sha256_file(attempt / name)
    finished['directory_listing'] = dict(finished['outputs'])
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


def a_completion_that_binds_another_evaluation_manifest(built):
    run = first(built)
    completion = built.read(run / 'completion.json')
    completion['eval_manifest_sha256'] = 'f' * 64
    (run / 'completion.json').write_text(json.dumps(completion, indent=2, sort_keys=True) + '\n')
    return 'completion binds the evaluation manifest'


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


def a_training_run_that_was_not_the_full_recipe(built):
    manifest = built.read(built.attempts['seen_simple'] / 'train_manifest.json')
    manifest['mode'] = 'smoke'
    rebind_training(built, 'seen_simple', manifest=manifest)
    return 'training full-run mode'


def a_training_run_that_allowed_a_dirty_tree(built):
    manifest = built.read(built.attempts['seen_cyl'] / 'train_manifest.json')
    manifest['allow_dirty'] = True
    rebind_training(built, 'seen_cyl', manifest=manifest)
    return 'training full-run mode'


def an_inventory_sidecar_outside_the_attempt(built):
    manifest = built.read(built.attempts['seen_aug'] / 'train_manifest.json')
    manifest['train_data_identity']['inventory_file']['path'] = str(built.root / 'elsewhere.json')
    rebind_training(built, 'seen_aug', manifest=manifest)
    return 'training inventory sidecar path'


def an_inventory_sidecar_rewritten_after_the_run(built):
    sidecar = built.attempts['seen_simple'] / 'train_inventory.json'
    built.replace(sidecar, dict(inventory=[]))
    return 'training inventory sidecar bytes'


def an_inventory_count_that_is_not_the_seen_training_split(built):
    manifest = built.read(built.attempts['seen_cyl'] / 'train_manifest.json')
    manifest['train_data_identity']['inventory_files'] = 1
    rebind_training(built, 'seen_cyl', manifest=manifest)
    return 'training inventory file count'


def a_substituted_training_inventory_digest(built):
    manifest = built.read(built.attempts['seen_aug'] / 'train_manifest.json')
    manifest['train_data_identity']['inventory_sha256'] = 'd' * 64
    rebind_training(built, 'seen_aug', manifest=manifest)
    return 'training inventory digest'


def an_hours_ledger_without_this_full_attempt(built):
    attempt = built.attempts['seen_simple']
    ledger = built.read(attempt.parent / 'cumulative_hours.json')
    ledger['attempts'][0]['mode'] = 'probe'
    built.replace(attempt.parent / 'cumulative_hours.json', ledger)
    return 'training hours ledger'


def retimed(built, role, **changes):
    manifest = built.read(built.attempts[role] / 'train_manifest.json')
    manifest['timing_limits'].update(**changes)
    rebind_training(built, role, manifest=manifest)


def receipt_path(attempt):
    """The arm's probe receipt, named as tools/exp07_launcher.py writes it."""
    return next(attempt.parent.glob('_probe_*.json'))


def rereceipted(built, role, **changes):
    """Rewrite the arm's probe receipt and the limits the launcher derives from it."""
    attempt = built.attempts[role]
    receipt = dict(built.read(receipt_path(attempt)), **changes)
    digest = built.replace(receipt_path(attempt), receipt)
    manifest = built.read(attempt / 'train_manifest.json')
    manifest['mutable_inputs']['probe_receipt']['sha256'] = digest
    manifest['timing_limits'].update(probe_receipt_sha256=digest, epoch_seconds=1.05 * receipt['T_epoch'],
        projection_hours=receipt['T_run'] / 3600, ceiling_hours=1.5 * receipt['T_run'] / 3600)
    rebind_training(built, role, manifest=manifest)


def a_projection_that_disagrees_with_the_receipt(built):
    retimed(built, 'seen_cyl', projection_hours=1000.)
    return 'training projection hours'


def a_projection_above_the_sixty_hour_budget_rule(built):
    rereceipted(built, 'seen_aug', T_run=61 * 3600.)
    return 'training projection hours'


def a_ceiling_that_is_not_one_and_a_half_projections(built):
    retimed(built, 'seen_simple', ceiling_hours=60.)
    return 'training ceiling hours'


def a_limit_derived_from_another_receipt(built):
    retimed(built, 'seen_cyl', probe_receipt_sha256='f' * 64)
    return 'probe receipt identity'


def a_probe_receipt_for_another_backbone(built):
    rereceipted(built, 'seen_simple', backbone='cylindrical')
    return 'probe receipt backbone'


def a_missing_probe_receipt_binding(built):
    manifest = built.read(built.attempts['seen_aug'] / 'train_manifest.json')
    manifest['mutable_inputs'].pop('probe_receipt')
    rebind_training(built, 'seen_aug', manifest=manifest)
    return 'probe receipt binding'


def a_manifest_epoch_budget_that_is_not_the_trainers(built):
    manifest = built.read(built.attempts['seen_simple'] / 'train_manifest.json')
    manifest['effective_args']['epochs'] = 1
    rebind_training(built, 'seen_simple', manifest=manifest)
    return 'effective_args disagree with args.json'


def a_manifest_backbone_that_is_not_the_trainers(built):
    manifest = built.read(built.attempts['seen_cyl'] / 'train_manifest.json')
    manifest['effective_args']['backbone'] = 'simple'
    rebind_training(built, 'seen_cyl', manifest=manifest)
    return 'effective_args disagree with args.json'


def a_manifest_yaw_flag_that_is_not_the_trainers(built):
    manifest = built.read(built.attempts['seen_aug'] / 'train_manifest.json')
    manifest['effective_args']['yaw_aug'] = 0
    rebind_training(built, 'seen_aug', manifest=manifest)
    return 'effective_args disagree with args.json'


def a_save_dir_that_is_not_the_attempt(built):
    attempt, elsewhere = built.attempts['seen_aug'], str(built.root / 'elsewhere')
    manifest = built.read(attempt / 'train_manifest.json')
    manifest['effective_args']['save_dir'] = elsewhere
    rebind_training(built, 'seen_aug', manifest=manifest,
                    args=dict(built.read(attempt / 'args.json'), save_dir=elsewhere))
    return 'effective_args save_dir'


def a_completion_missing_an_epoch_checkpoint(built):
    completion = built.read(built.attempts['seen_cyl'] / 'completion.json')
    completion['outputs'].pop('epoch_005.pth')
    rebind_training(built, 'seen_cyl', completion=completion)
    return 'training epoch checkpoints'


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
            a_completion_that_binds_another_evaluation_manifest, a_binding_of_another_split_file, a_split_pickle_changed_after_the_run,
            an_unapproved_evaluator_closure, an_unapproved_writer_closure,
            a_missing_training_binding, a_completion_that_certifies_another_checkpoint,
            a_training_manifest_of_the_unseen_protocol, a_training_closure_that_is_not_the_pin,
            a_training_closure_with_no_file_records, a_training_closure_label_that_is_not_its_records,
            a_launcher_closure_record_that_was_never_reviewed,
            a_training_run_that_was_not_the_full_recipe, a_training_run_that_allowed_a_dirty_tree,
            an_inventory_sidecar_outside_the_attempt, an_inventory_sidecar_rewritten_after_the_run,
            an_inventory_count_that_is_not_the_seen_training_split,
            a_substituted_training_inventory_digest, an_hours_ledger_without_this_full_attempt,
            a_projection_that_disagrees_with_the_receipt, a_missing_probe_receipt_binding,
            a_projection_above_the_sixty_hour_budget_rule, a_limit_derived_from_another_receipt,
            a_ceiling_that_is_not_one_and_a_half_projections, a_probe_receipt_for_another_backbone,
            a_manifest_epoch_budget_that_is_not_the_trainers, a_save_dir_that_is_not_the_attempt,
            a_manifest_backbone_that_is_not_the_trainers, a_completion_missing_an_epoch_checkpoint,
            a_manifest_yaw_flag_that_is_not_the_trainers,
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


def test_the_validated_training_evidence_is_bound_into_the_producer_inputs(built):
    _, admitted = admit(built)
    for role in ('seen_simple', 'seen_cyl', 'seen_aug'):
        attempt = built.attempts[role]
        for path in (attempt / 'train_inventory.json', attempt.parent / 'cumulative_hours.json',
                     receipt_path(attempt)):
            assert admitted['inputs'][str(path)] == p.sha256_file(path)


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


def test_a_group_missing_a_registered_metric_refuses_the_table(built):
    for directory in built.paths[('seen_aug', 1)]:
        directory = Path(directory)
        sample = built.read(directory / 'per_sample_yaw.json')
        for cell in sample['P'].values():
            cell.pop('log_mse')
        metrics = built.read(directory / 'metrics_yaw.json')
        metrics['P'] = built.summaries(sample['P'])
        built.rebind(directory, sample=sample, metrics=metrics)
    admit(built)  # the evaluation-level admission does not require the spectral diagnostics
    with pytest.raises(ValueError, match='missing metrics: log_mse'):
        build(built)


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


def test_the_confirmatory_markdown_carries_no_exploratory_banner(built, tmp_path):
    text = (publish(built, tmp_path, 'clean') / 'model_comparison_seen.md').read_text()
    assert 'EXPLORATORY' not in text and 'Admission deviations' not in text


def test_the_exploratory_markdown_states_it_and_lists_every_deviation(built, tmp_path):
    expected = an_evaluated_split_that_is_not_seen(built)
    target = publish(built, tmp_path, 'exploratory', exploratory=True)
    result = json.loads((target / 'table.json').read_text())
    text = (target / 'model_comparison_seen.md').read_text()
    assert result['exploratory'] is True and result['rows'] and result['deviations']
    assert any(expected in item for item in result['deviations'])
    assert 'EXPLORATORY - NOT a confirmatory exp_07 result' in text.split('# Model comparison')[0]
    for item in result['deviations']:
        assert '- ' + item in text
    body, _, rest = text.partition(STAMP)  # the generated block stays self-verifying
    assert rest.split(' -->')[0] == hashlib.sha256(body.encode()).hexdigest()


def cli_argv(tmp_path, built):
    """The producer's argv for this fixture's runs, writing into an empty directory."""
    return ['--profile', 'TABLE_SEEN_V1', '--runs'] + built.directories + [
        '--json', str(tmp_path / 'table.json'), '--md', str(tmp_path / 'seen.md')]


def unapproved(tmp_path, template):
    """Commit the all-null template in an isolated repository and load it for real."""
    directory = tmp_path / 'pins'
    directory.mkdir()
    return load_approved_digests(approval_repo(directory, template))


def test_the_cli_refuses_an_unapproved_profile_and_writes_nothing(
        tmp_path, built, monkeypatch, exp07_approval_template):  # noqa: F811  (fixture)
    """The refusal is a fact about the runtime approval, not about the committed pins.

    tools.exp07_table.admit reads the pins through this module's load_approved_digests
    and through nothing else, so an all-null record routed there refuses whatever state
    the committed file is in -- the template before the pins are filled, and the filled
    file afterwards.
    """
    pins = unapproved(tmp_path, exp07_approval_template)
    monkeypatch.setattr(table, 'load_approved_digests', lambda: pins)
    with pytest.raises(ValueError, match='not yet approved'):
        table.main(cli_argv(tmp_path, built))
    assert not (tmp_path / 'table.json').exists() and not (tmp_path / 'seen.md').exists()


def canonical_profile_digest(profile):
    """Hash a profile the way the producer does, without reading anything it produced.

    tools/exp07_table.py:374 detaches the registered profile with this module's own
    ``json_value``; lines 405-407 round-trip that value through JSON and hash the
    canonical text with exactly these ``json.dumps`` keywords.  Recomputing the digest
    from the fixture's profile is what makes a digest the producer invents (an all-zero
    string, say) fail, where comparing the written digest with the returned one lets both
    be wrong together.
    """
    detached = table.json_value(profile)                            # exp07_table.py:374
    literal = json.loads(json.dumps(table.json_value(detached)))    # exp07_table.py:405
    return hashlib.sha256(json.dumps(literal, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()  # :406-407


def test_the_cli_publishes_the_same_runs_once_the_approval_is_filled(tmp_path, built, monkeypatch):
    """The symmetric case: the identical argv proceeds under the fixture's filled pins."""
    monkeypatch.setattr(table, 'get_profile', lambda _: built.profile)
    monkeypatch.setattr(table, 'load_approved_digests', lambda: built.approved)
    monkeypatch.setattr(table, 'producer_identity', lambda _: built.producer)
    expected_digest = canonical_profile_digest(built.profile)
    result = table.main(cli_argv(tmp_path, built))
    assert result['deviations'] == [] and result['exploratory'] is False
    assert result['profile_name'] == 'TABLE_SEEN_V1' and len(result['rows']) == 8
    written = json.loads((tmp_path / 'table.json').read_text())
    assert written['rows'] == result['rows']
    assert result['profile_digest'] == expected_digest
    assert written['profile_digest'] == expected_digest
    assert built.profile['arms'][0]['label'] in (tmp_path / 'seen.md').read_text()


def test_the_cli_defaults_the_markdown_to_the_living_seen_table(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(table, 'build_table', lambda runs, exploratory: ({'ok': True}, {}))
    monkeypatch.setattr(table, 'write_outputs', lambda *a: captured.update(call=a))
    table.main(['--profile', 'TABLE_SEEN_V1', '--runs', 'a', '--json', str(tmp_path / 't.json')])
    assert table.MARKDOWN == table.REPO / 'worklog/worklog_yixun/model_comparison_seen.md'
    assert captured['call'][3] == str(table.MARKDOWN) and captured['call'][4] is False
    assert captured['call'][5][:4] == ['exp07_table.py', '--profile', 'TABLE_SEEN_V1', '--runs']
