"""The seen-split reference manifests: exclusive creation and a bound index.

CPU-only and data-free: the builder is driven with a stub dataset exposing the seen
class's interface (``file_list`` and ``ir_path``) over a throwaway RIR tree, so no
AcousticRooms read and no write into ``ckpt/`` happens here.
"""
import json

import pytest

from tools import exp07_manifests as subject
from tools import provenance as p
from tools.reference_manifest import manifest_hash


class StubSeenDataset:
    """``treble_xRIR_seen_dataset.xRIR_Dataset`` as ``build_manifest`` uses it."""

    def __init__(self, root, files, num_shot):
        self.ir_path, self.num_shot, self.max_len = str(root), num_shot, 9600
        self.file_list = list(reversed(files))  # os.listdir order is not canonical


@pytest.fixture
def tree(tmp_path):
    root, files = tmp_path / 'single_channel_ir', []
    for room in range(2):
        directory = root / 'Cat' / 'Room{}'.format(room)
        directory.mkdir(parents=True)
        for src in range(4):
            for rec in range(2):
                path = directory / 'S00{}_R00{}_hybrid_IR.wav'.format(src, rec)
                path.write_bytes(b'')
                files.append(str(path))
    return root, files


@pytest.fixture
def factory(tree):
    root, files = tree
    return lambda num_shot: StubSeenDataset(root, files, num_shot)


def test_builds_one_manifest_per_k_and_seed_with_a_bound_index(tmp_path, factory, tree):
    out = tmp_path / 'exp07'
    index = subject.build(out, seeds=(42, 43), num_shots=(3, 1), entries=len(tree[1]),
                          factory=factory)
    names = [subject.manifest_name(k, s) for k in (3, 1) for s in (42, 43)]
    assert sorted(item.name for item in out.iterdir()) == sorted(names + [subject.INDEX])
    assert index == json.loads((out / subject.INDEX).read_text())
    assert index['seen_split'] == p.seen_split_identity(subject.REPO)
    assert index['seeds'] == [42, 43] and index['num_shots'] == [3, 1]
    for name in names:
        manifest = json.loads((out / name).read_text())
        record = index['manifests'][name]
        assert record['manifest_hash'] == manifest_hash(manifest)
        assert record['file_sha256'] == p.sha256_file(out / name)
        assert record['entries'] == len(manifest['entries']) == len(tree[1])
        assert (record['num_shot'], record['seed']) == (manifest['num_shot'], manifest['seed'])
        assert all(len(entry['refs']) == record['num_shot'] for entry in manifest['entries'])
    hashes = {index['manifests'][name]['manifest_hash'] for name in names}
    assert len(hashes) == len(names)  # every seed and K draws its own references


@pytest.mark.parametrize('existing', [None, 'manifest', 'index'])
def test_refuses_to_overwrite(tmp_path, factory, tree, existing):
    out = tmp_path / 'exp07'
    out.mkdir()
    keywords = dict(seeds=(42,), num_shots=(3,), entries=len(tree[1]), factory=factory)
    if existing == 'manifest':
        (out / subject.manifest_name(3, 42)).write_text('{}')
    elif existing == 'index':
        (out / subject.INDEX).write_text('{}')
    else:
        subject.build(out, **keywords)
    with pytest.raises(FileExistsError):
        subject.build(out, **keywords)
    if existing:  # a refused run leaves the pre-existing bytes alone
        assert (out / (subject.INDEX if existing == 'index'
                       else subject.manifest_name(3, 42))).read_text() == '{}'


def test_an_unexpected_entry_count_refuses_and_writes_no_index(tmp_path, factory):
    out = tmp_path / 'exp07'
    with pytest.raises(ValueError, match='6217'):
        subject.build(out, seeds=(42,), num_shots=(3,), factory=factory)
    assert not (out / subject.INDEX).exists() and subject.ENTRIES == 6217
