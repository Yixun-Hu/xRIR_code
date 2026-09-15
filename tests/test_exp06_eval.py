"""Registry routing, arm metadata and the manifest handshake of the exp_06 evaluator."""
import json
from pathlib import Path

import pytest

from model.xRIR_cyl_oriented import BACKBONES_EXP06
from tools import exp04_eval as base
from tools import exp06_eval as subject
from tools import provenance
from tools.reference_manifest import manifest_hash

ROOT = Path(__file__).resolve().parents[1]


def backbone_choices(parser):
    return [action.choices for action in parser._actions if action.dest == 'backbone']


@pytest.fixture
def exp06_run(tmp_path, monkeypatch):
    """An exp_04 protocol manifest carrying this arm's exp_06 metadata."""
    checkpoint, reference = tmp_path / 'checkpoint.pth', tmp_path / 'reference.json'
    checkpoint.write_bytes(b'weights')
    refs = {'seed': 42, 'num_shot': 8, 'entries': []}
    reference.write_text(json.dumps(refs))
    output = tmp_path / 'run'
    output.mkdir()
    path = output / 'eval_manifest.json'
    args = subject.parse_args([
        '--backbone', 'cylindrical_oriented', '--checkpoint', str(checkpoint),
        '--manifest', str(reference), '--manifest-hash', manifest_hash(refs),
        '--out-dir', str(output), '--eval-manifest', str(path),
        '--checkpoint-role', 'arm', '--checkpoint-epoch', '12'])
    record = [{'path': 'eval_yaw_rotation.py', 'reviewed_blob_sha256': 'source',
               'working_tree_sha256': 'source', 'commits_after_reviewed': []}]
    fields = {key: value for key, value in vars(args).items()
              if key not in ('eval_manifest', 'out_dir', 'manifest')}
    fields.update(schema_version=1, checkpoint_sha256=provenance.sha256_file(checkpoint),
                  manifest_path=str(reference),
                  manifest_file_sha256=provenance.sha256_file(reference),
                  manifest_seed=42, num_shot=8, batch_canonical=True,
                  data_root=base.BASE_DATA_PATH, repo=str(ROOT), reviewed_commit='a' * 40,
                  evaluator_closure={'files': record, 'sha256': 'closure'})
    fields.update(subject.exp06_metadata(args))
    path.write_text(json.dumps(fields))
    monkeypatch.setattr(provenance, 'source_closure', lambda *a: ['eval_yaw_rotation.py'])
    monkeypatch.setattr(provenance, 'closure_record', lambda *a: (record, 'closure'))
    return args, fields, path


def test_the_parser_ranges_over_the_exp06_registry_without_changing_exp04():
    assert backbone_choices(subject.build_parser()) == [sorted(BACKBONES_EXP06)]
    assert 'cylindrical_oriented' in sorted(BACKBONES_EXP06)
    assert backbone_choices(base.build_parser()) == [sorted(base.yaw.BACKBONES)]
    assert 'cylindrical_oriented' not in base.yaw.BACKBONES


def test_the_parser_is_child_command_compatible():
    """tools.exp04_eval_launch.child_command serialises option strings, never positionals."""
    for require in (True, False):
        for action in subject.build_parser(require)._actions:
            assert action.option_strings, action.dest
        dests = {action.dest for action in subject.build_parser(require)._actions}
        assert {'checkpoint_role', 'checkpoint_epoch'} <= dests
    parsed = subject.parse_args(['--backbone', 'simple', '--checkpoint', 'c', '--manifest', 'm',
                                 '--manifest-hash', 'h', '--out-dir', 'o', '--eval-manifest', 'e',
                                 '--checkpoint-epoch', '12'])
    assert parsed.checkpoint_role == 'arm' and parsed.checkpoint_epoch == 12


@pytest.mark.parametrize('argv', [
    ['--backbone', 'simple', '--checkpoint', 'c', '--manifest', 'm', '--manifest-hash', 'h',
     '--out-dir', 'o', '--eval-manifest', 'e'],
    ['--backbone', 'unknown', '--checkpoint', 'c', '--manifest', 'm', '--manifest-hash', 'h',
     '--out-dir', 'o', '--eval-manifest', 'e', '--checkpoint-epoch', '12'],
    ['--backbone', 'simple', '--checkpoint', 'c', '--manifest', 'm', '--manifest-hash', 'h',
     '--out-dir', 'o', '--eval-manifest', 'e', '--checkpoint-epoch', '12',
     '--checkpoint-role', 'invented']])
def test_missing_epoch_and_unregistered_names_are_refused(argv):
    with pytest.raises(SystemExit):
        subject.parse_args(argv)


@pytest.mark.parametrize('backbone', sorted(BACKBONES_EXP06))
def test_the_factory_routes_every_registered_name_to_its_class(backbone):
    assert type(subject.model_factory(backbone, 2)) is BACKBONES_EXP06[backbone]


def test_the_factory_refuses_an_unregistered_name():
    with pytest.raises(ValueError, match='unknown backbone'):
        subject.model_factory('invented', 2)


def test_metadata_records_the_arm_identity(exp06_run):
    args, _, _ = exp06_run
    metadata = subject.exp06_metadata(args)
    assert set(metadata) == {'model_class', 'registry_sha256', 'checkpoint_role',
                             'checkpoint_epoch', 'heading', 'frame'}
    assert metadata['model_class'] == BACKBONES_EXP06['cylindrical_oriented'].__name__
    assert metadata['checkpoint_role'] == 'arm' and metadata['checkpoint_epoch'] == 12
    assert metadata['heading'] is None and metadata['frame'] == 'room'
    from tools import exp06_finalize
    assert metadata['registry_sha256'] == exp06_finalize.registry_sha256()


def test_a_manifest_carrying_the_metadata_round_trips(exp06_run):
    args, fields, path = exp06_run
    got, digest, reference = subject.validate_manifest(args)
    assert got == fields and digest == provenance.sha256_file(path)
    assert reference['num_shot'] == 8
    assert subject.validate_manifest(args, subject.exp06_metadata(args))[0] == fields


@pytest.mark.parametrize('field', sorted(['model_class', 'registry_sha256', 'checkpoint_role',
                                          'checkpoint_epoch', 'heading', 'frame']))
def test_each_metadata_mismatch_is_refused(exp06_run, field):
    args, fields, path = exp06_run
    path.write_text(json.dumps(dict(fields, **{field: 'wrong'})))
    with pytest.raises(ValueError, match=field):
        subject.validate_manifest(args)
    path.write_text(json.dumps({key: value for key, value in fields.items() if key != field}))
    with pytest.raises(ValueError, match=field):
        subject.validate_manifest(args)


@pytest.mark.parametrize('role', ['baseline', 'diagnostic'])
def test_a_manifest_of_another_checkpoint_role_is_refused(exp06_run, role):
    args, fields, path = exp06_run
    args.checkpoint_role = role
    with pytest.raises(ValueError, match='checkpoint_role'):
        subject.validate_manifest(args)
    args.checkpoint_role = 'arm'
    args.checkpoint_epoch = 11
    with pytest.raises(ValueError, match='checkpoint_epoch'):
        subject.validate_manifest(args)
