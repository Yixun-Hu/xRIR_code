"""The exp_07 evaluation launcher: entry routing, seen bindings and cross-split refusals.

CPU-only, with the tiny stub children of tests/test_exp04_eval_launch.py.
"""
import json
from pathlib import Path

import pytest

from tools import exp04_eval_launch as launcher
from tools import exp07_eval
from tools import provenance as p
from test_exp04_eval_launch import attempt, child_code, launch_args  # noqa: F401  (fixtures)

REPO = Path(launcher.__file__).resolve().parents[1]


def cli(*extra):
    return ['--checkpoint', 'c', '--backbone', 'simple', '--manifest', 'm', '--manifest-hash',
            'h', '--out-dir', 'o', '--log-dir', 'l', '--data-root', 'd', '--run-label', 'r',
            '--reviewed-commit', 'HEAD', '--num-shot', '8'] + list(extra)


@pytest.fixture
def seen_args(launch_args):
    launch_args.entry, launch_args.split = 'exp07', 'seen'
    return launch_args


def test_the_seen_entry_routes_to_exp07_eval():
    args = launcher.parse_args(cli('--entry', 'exp07', '--split', 'seen'))
    assert (args.entry, args.split) == ('exp07', 'seen')
    assert launcher.entrypoint(args) is exp07_eval
    command = launcher.child_command(args, REPO)
    assert command[1] == str(REPO / 'tools/exp07_eval.py')
    assert command[command.index('--split') + 1] == 'seen'


@pytest.mark.parametrize('extra', [['--split', 'seen'], ['--split', 'seen', '--entry', 'exp05'],
                                   ['--split', 'seen', '--tier', 'S'], ['--entry', 'exp07'],
                                   ['--entry', 'exp07', '--split', 'unseen'],
                                   ['--entry', 'exp07', '--tier', 'S']])
def test_cross_entry_and_cross_split_invocations_are_refused(extra, capsys):
    with pytest.raises(SystemExit):
        launcher.parse_args(cli(*extra))
    assert 'exp07' in capsys.readouterr().err


@pytest.mark.parametrize('extra', [[], ['--split', 'unseen'], ['--tier', 'S']])
def test_the_unseen_entries_are_unchanged(extra):
    args = launcher.parse_args(cli(*extra))
    assert args.split == 'unseen' and launcher.entrypoint(args) is not exp07_eval
    assert '--split' not in launcher.child_command(args, REPO)


def test_seen_fields_record_the_split_and_bind_the_split_file(seen_args):
    fields = launcher.build_fields(seen_args, launcher.child_command(seen_args, REPO), REPO)
    assert fields['split'] == 'seen' and fields['split_count'] == 1
    assert fields['seen_split_sha256'] == p.sha256_file(REPO / p.SEEN_SPLIT)
    assert fields['mutable_inputs']['seen_split'] == p.seen_split_identity(REPO)
    assert p.revalidate({'repo': str(REPO), 'mutable_inputs': fields['mutable_inputs']}) == []
    assert fields['command'][1] == str(REPO / 'tools/exp07_eval.py')


def test_an_explicit_seen_split_binding_must_be_the_authors_file(seen_args, tmp_path):
    other = tmp_path / 'other.pkl'
    other.write_text('not the split')
    seen_args.bind_input = ['seen_split=' + str(other)]
    with pytest.raises(ValueError, match='seen_split'):
        launcher.build_fields(seen_args, launcher.child_command(seen_args, REPO), REPO)


@pytest.mark.parametrize('entries,samples,confirmatory', [
    (6217, 0, True), (6216, 0, False), (6218, 0, False), (6217, 16, False)])
def test_confirmatory_requires_every_seen_query(seen_args, monkeypatch, entries, samples,
                                                confirmatory):
    seen_args.max_samples = samples
    monkeypatch.setattr(launcher.evaluator.yaw, 'load_checked_manifest',
                        lambda *a: {'num_shot': 8, 'seed': 42, 'entries': [None] * entries})
    fields = launcher.build_fields(seen_args, launcher.child_command(seen_args, REPO), REPO)
    assert fields['confirmatory'] is confirmatory and fields['split_count'] == entries
    assert exp07_eval.SPLIT_ENTRIES['seen'] == 6217


def test_completion_records_the_split_of_a_seen_run(attempt):
    args, _, fields = attempt
    fields.update(split='seen', seen_split_sha256='a' * 64)
    completion = launcher.execute_run(args, child_code(args.out_dir), lambda: fields,
                                      args.data_root)
    assert completion['split'] == 'seen' and completion['seen_split_sha256'] == 'a' * 64
    assert json.loads((Path(args.out_dir) / 'completion.json').read_text()) == completion


def test_an_unseen_run_records_no_split_fields(attempt):
    args, _, fields = attempt
    completion = launcher.execute_run(args, child_code(args.out_dir), lambda: fields,
                                      args.data_root)
    assert not {'split', 'seen_split_sha256'} & set(completion)


@pytest.mark.parametrize('field', ['split', 'seen_split_sha256'])
def test_a_child_that_evaluated_another_split_is_not_finalized(attempt, field):
    args, _, fields = attempt
    fields.update(split='seen', seen_split_sha256='a' * 64)
    target = str(Path(args.out_dir) / 'per_sample_yaw.json')
    extra = ("payload = json.loads(Path(%r).read_text()); payload['meta'][%r] = 'other'; "
             "Path(%r).write_text(json.dumps(payload))" % (target, field, target))
    with pytest.raises(ValueError, match='output meta mismatch'):
        launcher.execute_run(args, child_code(args.out_dir, extra), lambda: fields,
                             args.data_root)
    assert Path(args.out_dir + '_ABORTED_output_invalid').is_dir()
