"""The exp_07 evaluation launcher: its own CLI, seen bindings and the shared run owner.

CPU-only, with the tiny stub children of tests/test_exp04_eval_launch.py.
"""
import json
import subprocess
from pathlib import Path

import pytest

from tools import exp04_eval_launch as base
from tools import exp07_eval
from tools import exp07_eval_launch as launcher
from tools import exp07_provenance as e7p
from tools import provenance as p
from test_exp04_eval_launch import attempt, child_code, launch_args  # noqa: F401  (fixtures)

REPO = Path(base.__file__).resolve().parents[1]


def cli(*extra):
    return ['--checkpoint', 'c', '--backbone', 'simple', '--manifest', 'm', '--manifest-hash',
            'h', '--out-dir', 'o', '--log-dir', 'l', '--data-root', 'd', '--run-label', 'r',
            '--reviewed-commit', 'HEAD', '--num-shot', '8'] + list(extra)


@pytest.fixture
def seen_args(launch_args):
    seen = launcher.parse_args(cli('--split', 'seen'))
    for key, value in vars(launch_args).items():
        if key not in ('split', 'entry'):
            setattr(seen, key, value)
    return seen


def test_the_launcher_spawns_the_exp07_entry_point():
    args = launcher.parse_args(cli('--split', 'seen'))
    # Nit 9: this launcher has no --entry at all; it spawns tools/exp07_eval.py and nothing else.
    assert args.split == 'seen' and not hasattr(args, 'entry')
    assert '--entry' not in Path(launcher.__file__).read_text()
    command = launcher.child_command(args, REPO)
    assert command[1] == str(REPO / launcher.ENTRY)
    assert command[command.index('--split') + 1] == 'seen'
    assert '--tier' not in command and '--entry' not in command


@pytest.mark.parametrize('extra', [[], ['--split', 'held-out'], ['--tier', 'S'],
                                   ['--entry', 'exp07'], ['--entry', 'exp04']])
def test_missing_or_foreign_arguments_are_refused(extra):
    with pytest.raises(SystemExit):
        launcher.parse_args(cli(*extra))


def test_the_shared_launcher_is_unchanged_and_still_refuses_the_seen_names():
    assert subprocess.run(['git', 'diff', '--quiet', 'main', '--', 'tools/exp04_eval_launch.py'],
                          cwd=str(REPO)).returncode == 0
    assert 'seen_split' not in base.MUTABLE_INPUTS
    assert 'seen_split' in launcher.MUTABLE_INPUTS
    with pytest.raises(SystemExit):  # the exp_04 CLI has no --split
        base.parse_args(cli('--split', 'seen'))


@pytest.mark.parametrize('split', ['seen', 'unseen'])
def test_fields_record_the_split_and_bind_the_split_file(seen_args, split):
    seen_args.split = split
    fields = launcher.build_fields(seen_args, launcher.child_command(seen_args, REPO), REPO)
    assert fields['split'] == split and fields['split_count'] == 1
    assert fields['seen_split_sha256'] == p.sha256_file(REPO / e7p.SEEN_SPLIT)
    assert fields['mutable_inputs']['seen_split'] == e7p.seen_split_identity(REPO)
    assert p.revalidate({'repo': str(REPO), 'mutable_inputs': fields['mutable_inputs']}) == []
    assert fields['command'][1] == str(REPO / launcher.ENTRY)


def test_the_writer_closure_is_this_launcher(seen_args, monkeypatch):
    asked = []
    monkeypatch.setattr(p, 'source_closure',
                        lambda module, repo: asked.append(module) or [module + '.py'])
    fields = launcher.build_fields(seen_args, launcher.child_command(seen_args, REPO), REPO)
    assert asked == ['eval_yaw_rotation', 'tools.exp07_eval', 'tools.exp04_eval_launch',
                     launcher.WRITER]
    assert [r['path'] for r in fields['source_closures']['writer']['files']] == [launcher.WRITER + '.py']
    assert [r['path'] for r in fields['source_closures']['entrypoint']['files']] == ['tools.exp07_eval.py']


def test_a_drifted_writer_is_refused(seen_args, monkeypatch):
    monkeypatch.setattr(p, 'closure_record', lambda files, commit, repo: (
        [{'path': name, 'reviewed_blob_sha256': 'a', 'working_tree_sha256': 'b',
          'commits_after_reviewed': []} for name in files], 'digest')
        if files == [launcher.WRITER + '.py'] else
        ([{'path': name, 'reviewed_blob_sha256': 'a', 'working_tree_sha256': 'a',
           'commits_after_reviewed': []} for name in files], 'digest'))
    monkeypatch.setattr(p, 'source_closure', lambda module, repo: [module + '.py'])
    with pytest.raises(ValueError, match='writer closure'):
        launcher.build_fields(seen_args, launcher.child_command(seen_args, REPO), REPO)


def test_the_real_writer_closure_contains_both_launchers():
    files = set(p.source_closure(launcher.WRITER, REPO))
    assert {'tools/exp07_eval_launch.py', 'tools/exp04_eval_launch.py', 'tools/exp07_eval.py',
            exp07_eval.SEEN_DATASET_SOURCE} <= files


def test_an_explicit_seen_split_binding_must_be_the_authors_file(seen_args, tmp_path):
    other = tmp_path / 'other.pkl'
    other.write_text('not the split')
    seen_args.bind_input = ['seen_split=' + str(other)]
    with pytest.raises(ValueError, match='seen_split'):
        launcher.build_fields(seen_args, launcher.child_command(seen_args, REPO), REPO)


@pytest.mark.parametrize('spelling', ['absolute', 'symlink', 'indirect'])
def test_an_explicit_binding_of_the_canonical_split_is_accepted(seen_args, tmp_path, spelling):
    """The shared builder resolves --bind-input paths; the canonical file is the file."""
    canonical = REPO / e7p.SEEN_SPLIT
    named = {'absolute': canonical,
             'indirect': canonical.parent / '..' / canonical.parent.name / canonical.name,
             'symlink': tmp_path / 'link.pkl'}[spelling]
    if spelling == 'symlink':
        named.symlink_to(canonical)
    seen_args.bind_input = ['seen_split=' + str(named)]
    fields = launcher.build_fields(seen_args, launcher.child_command(seen_args, REPO), REPO)
    assert fields['mutable_inputs']['seen_split'] == e7p.seen_split_identity(REPO)  # canonical
    assert p.revalidate({'repo': str(REPO), 'mutable_inputs': fields['mutable_inputs']}) == []


def test_a_byte_identical_copy_of_the_split_elsewhere_is_refused(seen_args, tmp_path):
    copy = tmp_path / 'seen_test_split.pkl'
    copy.write_bytes((REPO / e7p.SEEN_SPLIT).read_bytes())
    seen_args.bind_input = ['seen_split=' + str(copy)]
    with pytest.raises(ValueError, match='seen_split'):
        launcher.build_fields(seen_args, launcher.child_command(seen_args, REPO), REPO)


def test_an_unknown_binding_name_is_still_refused(seen_args):
    seen_args.bind_input = ['invented=' + str(REPO / e7p.SEEN_SPLIT)]
    with pytest.raises(ValueError, match='bind-input'):
        launcher.build_fields(seen_args, launcher.child_command(seen_args, REPO), REPO)


@pytest.mark.parametrize('entries,samples,confirmatory', [
    (6217, 0, True), (6216, 0, False), (6218, 0, False), (6217, 16, False)])
def test_confirmatory_requires_every_seen_query(seen_args, monkeypatch, entries, samples,
                                                confirmatory):
    seen_args.max_samples = samples
    monkeypatch.setattr(base.evaluator.yaw, 'load_checked_manifest',
                        lambda *a: {'num_shot': 8, 'seed': 42, 'entries': [None] * entries})
    fields = launcher.build_fields(seen_args, launcher.child_command(seen_args, REPO), REPO)
    assert fields['confirmatory'] is confirmatory and fields['split_count'] == entries
    assert exp07_eval.SPLIT_ENTRIES['seen'] == 6217


def test_the_completion_hash_binds_the_split_through_the_eval_manifest(attempt):
    """The split lives in the eval manifest and in both output metas, both hash-bound."""
    args, _, fields = attempt
    fields.update(split='seen', seen_split_sha256='a' * 64)
    completion = base.execute_run(args, child_code(args.out_dir), lambda: fields, args.data_root)
    run = Path(args.out_dir)
    manifest = json.loads((run / 'eval_manifest.json').read_text())
    assert (manifest['split'], manifest['seen_split_sha256']) == ('seen', 'a' * 64)
    assert completion['eval_manifest_sha256'] == p.sha256_file(run / 'eval_manifest.json')
    assert completion['outputs'] == {name: p.sha256_file(run / name) for name in launcher.OUTPUTS}
    assert json.loads((run / 'completion.json').read_text()) == completion
