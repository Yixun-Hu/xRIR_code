"""The seen/unseen evaluation entry point: handshake, split dispatch and refusals.

CPU-only: both split dataset classes are stubbed, so the real frozen builders run
without AcousticRooms and without a GPU.  The bit-parity and final-batch checks that
need a GPU live in tests/test_exp07_eval_gpu.py.
"""
import json
import os
from pathlib import Path

import pytest

from tools import exp04_eval as base
from tools import exp07_eval as subject
from tools import exp07_provenance as e7p
from tools import provenance as p
from test_exp04_eval import protocol_run, bound_run  # noqa: F401  (fixtures)

SEEN = ('Cat/RoomS/S000_R000_hybrid_IR.wav', 'Cat/RoomS/S001_R000_hybrid_IR.wav')
UNSEEN = ('Cat/RoomU/S000_R000_hybrid_IR.wav', 'Cat/RoomU/S001_R000_hybrid_IR.wav')
BASE = ['--backbone', 'simple', '--checkpoint', 'ckpt.pth', '--manifest', 'm.json',
        '--manifest-hash', 'hash', '--out-dir', 'out', '--eval-manifest', 'out/eval.json']


class StubDataset:
    """The split classes as the frozen ManifestDataset uses them."""

    def __init__(self, queries, num_shot):
        self.ir_path, self.num_shot, self.max_len = '/root', num_shot, 9600
        self.file_list = [os.path.join(self.ir_path, query) for query in reversed(queries)]


def manifest_of(queries, num_shot=1):
    other = {queries[0]: queries[1], queries[1]: queries[0]}
    return {'seed': 42, 'num_shot': num_shot, 'ir_root': '/root',
            'entries': [{'index': i, 'query': query, 'refs': [other[query]] * num_shot}
                        for i, query in enumerate(queries)]}


@pytest.fixture
def split_classes(monkeypatch):
    """Stub both dataset classes; the frozen builders and validators stay real."""
    from treble_multi_room_dataset import treble_xRIR_dataset as unseen
    from treble_multi_room_dataset import treble_xRIR_seen_dataset as seen
    calls = []
    def stub(label, queries):
        def factory(split='train', max_len=9600, num_shot=4, **kwargs):
            calls.append((label, split, max_len, num_shot))
            return StubDataset(queries, num_shot)
        return factory
    monkeypatch.setattr(unseen, 'xRIR_Dataset', stub('unseen', UNSEEN))
    monkeypatch.setattr(seen, 'xRIR_Dataset', stub('seen', SEEN))
    return calls


@pytest.fixture
def split_run(bound_run):
    """The exp_04 handshake fixture, re-expressed for the split-bound entry point."""
    args, fields, path = bound_run
    def rebuild(split):
        rebuilt = subject.parse_args(
            ['--backbone', args.backbone, '--checkpoint', args.checkpoint, '--manifest',
             args.manifest, '--manifest-hash', args.manifest_hash, '--out-dir', args.out_dir,
             '--eval-manifest', args.eval_manifest, '--split', split])
        record = dict(fields, **subject.split_metadata(rebuilt))
        path.write_text(json.dumps(record))
        return rebuilt, record
    return rebuild


def args_for(split):
    return subject.parse_args(BASE + ['--split', split])


@pytest.mark.parametrize('extra', [[], ['--split', 'held-out'], ['--split']])
def test_the_split_is_required_and_constrained(extra):
    with pytest.raises(SystemExit):
        subject.parse_args(BASE + extra)
    assert subject.SPLITS == ('unseen', 'seen')


@pytest.mark.parametrize('split', ['seen', 'unseen'])
def test_handshake_binds_the_split_and_the_split_file(split_run, split):
    args, fields = split_run(split)
    got, digest, reference = subject.validate_manifest(args)
    assert got == fields and fields['split'] == split and reference['num_shot'] == 8
    assert fields['seen_split_sha256'] == p.sha256_file(subject.REPO / e7p.SEEN_SPLIT)
    assert digest == p.sha256_file(args.eval_manifest)


@pytest.mark.parametrize('field', ['split', 'seen_split_sha256'])
def test_handshake_refuses_a_changed_split_binding(split_run, field):
    args, fields = split_run('seen')
    Path(args.eval_manifest).write_text(json.dumps(dict(fields, **{field: 'different'})))
    with pytest.raises(ValueError, match=field):
        subject.validate_manifest(args)


@pytest.mark.parametrize('split,queries', [('seen', SEEN), ('unseen', UNSEEN)])
def test_the_dataset_of_each_split_is_built_from_its_own_class(split_classes, split, queries):
    dataset = subject.build_split_dataset(args_for(split), manifest_of(queries))
    assert split_classes == [(split, 'test', subject.MAX_LEN, 1)]
    assert [entry['query'] for entry in dataset.entries] == list(queries)
    assert len(dataset) == len(queries)


@pytest.mark.parametrize('split,queries', [('seen', UNSEEN), ('unseen', SEEN)])
def test_a_manifest_of_the_other_split_is_refused(split_classes, split, queries):
    with pytest.raises(ValueError, match='do not match dataset.file_list'):
        subject.build_split_dataset(args_for(split), manifest_of(queries))
@pytest.mark.parametrize('split,queries', [('seen', SEEN), ('unseen', UNSEEN)])
def test_max_samples_keeps_the_frozen_subset_behaviour(split_classes, split, queries):
    dataset = subject.build_split_dataset(args_for(split), manifest_of(queries), max_samples=1)
    assert isinstance(dataset, base.yaw.SubsetManifestDataset) and len(dataset) == 1
    assert dataset.entries == manifest_of(queries)['entries'][:1]


def test_num_shot_comes_from_the_manifest(split_classes):
    subject.build_split_dataset(args_for('seen'), manifest_of(SEEN, num_shot=8))
    assert split_classes == [('seen', 'test', subject.MAX_LEN, 8)]


def test_run_exp07_needs_a_gpu_after_the_handshake(split_run, monkeypatch):
    """The handshake runs first; the restated loop then refuses a CPU-only host."""
    args, fields = split_run('seen')
    monkeypatch.setattr(subject.yaw.torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='needs a GPU'):
        subject.run_exp07(args)
    Path(args.eval_manifest).write_text(json.dumps(dict(fields, split='unseen')))
    with pytest.raises(ValueError, match='split'):
        subject.run_exp07(args)


@pytest.mark.parametrize('fault', ['split', 'seen_split_sha256', 'pickle'])
def test_check_outputs_reads_back_the_written_split(split_run, tmp_path, monkeypatch, fault):
    args, _ = split_run('seen')
    args.out_dir = str(tmp_path)
    metadata = subject.split_metadata(args)
    for name in ('per_sample_yaw.json', 'metrics_yaw.json'):
        (tmp_path / name).write_text(json.dumps({'meta': dict(metadata)}))
    subject.check_outputs(args, metadata)
    if fault == 'pickle':
        monkeypatch.setattr(subject.provenance, 'sha256_file', lambda path: 'c' * 64)
        message = 'changed during evaluation'
    else:
        (tmp_path / 'metrics_yaw.json').write_text(
            json.dumps({'meta': dict(metadata, **{fault: 'different'})}))
        message = 'output split metadata mismatch'
    with pytest.raises(ValueError, match=message):
        subject.check_outputs(args, metadata)


def test_the_shared_evaluator_keeps_mains_signature():
    """exp_04's loop is untouched: it has no dataset factory and no split argument."""
    import inspect
    import subprocess
    assert subprocess.run(['git', 'diff', '--quiet', 'main', '--', 'tools/exp04_eval.py'],
                          cwd=str(subject.REPO)).returncode == 0
    assert list(inspect.signature(base.run_exp04).parameters) == [
        'args', 'model_factory', 'metadata', 'manifest_validator']
    assert 'split' not in {a.dest for a in base.build_parser()._actions}
    assert 'split' in {a.dest for a in subject.build_parser()._actions}


@pytest.fixture(scope='module')
def entry_closure():
    """The entry point's import-derived closure (a subprocess import; computed once)."""
    return p.source_closure('tools.exp07_eval', subject.REPO)


def test_the_seen_dataset_module_is_bound_by_the_entry_points_closure(entry_closure):
    """The frozen helper imports it inside a function, so the entry point imports it."""
    assert subject.SEEN_DATASET_SOURCE == 'treble_multi_room_dataset/treble_xRIR_seen_dataset.py'
    assert subject.SEEN_DATASET_SOURCE in entry_closure
    assert 'treble_multi_room_dataset/treble_xRIR_dataset.py' in entry_closure


@pytest.mark.parametrize('module', ['tools.exp04_eval', 'tools.exp04_eval_launch'])
def test_the_unseen_entry_and_the_writer_do_not_gain_the_seen_module(module):
    """Binding the seen code must not move the exp_04 entry or writer digests."""
    assert subject.SEEN_DATASET_SOURCE not in p.source_closure(module, subject.REPO)


def test_a_changed_seen_dataset_module_is_refused_by_revalidate(tmp_path, entry_closure):
    """A run bound at the old closure refuses once the seen split's code changes."""
    files = []
    for name in entry_closure:
        copy = tmp_path / name
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes((subject.REPO / name).read_bytes())
        files.append(dict(path=name, working_tree_sha256=p.sha256_file(copy)))
    manifest = dict(repo=str(tmp_path), source_closures=dict(entrypoint=dict(files=files)))
    assert p.revalidate(manifest) == []
    drifted = tmp_path / subject.SEEN_DATASET_SOURCE
    drifted.write_bytes(drifted.read_bytes() + b'\n# a change to the seen split code\n')
    assert p.revalidate(manifest) == ['source.entrypoint.' + subject.SEEN_DATASET_SOURCE]
