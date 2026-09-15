"""exp_07 provenance: protocol-isolated training inventories and the seen-split binding."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from tools import exp07_provenance as e7
from tools import provenance as p
from test_provenance import data  # the synthetic AcousticRooms root, reused verbatim

ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_ROOT = os.environ.get('XRIR_DATA_PATH', '')
REAL_CACHE = ROOT / e7.SEEN_CACHE


@pytest.fixture
def protocol_root(data, monkeypatch):
    """The synthetic root with every category and one room the unseen split holds out."""
    monkeypatch.chdir(ROOT)  # the frozen seen module opens its pickle relative to the cwd
    root = data[0]
    for category in ['Bathrooms', 'Cafe', 'LivingRoomsWithHallway', 'Office',
                     'Auditorium', 'Bedrooms', 'ListeningRoom', 'MeetingRoom', 'Restaurants']:
        (root / 'single_channel_ir' / category).mkdir()
    held_out = root / 'single_channel_ir/Apartments/Apartments_idx_50'
    held_out.mkdir()
    (held_out / 'excluded.wav').write_bytes(b'test split')
    return root


def test_train_data_identity_is_protocol_isolated(protocol_root, tmp_path):
    caches = {name: tmp_path / (name + '.json') for name in e7.PROTOCOLS}
    records = {name: e7.train_data_identity(protocol_root, name, cache_path=path, workers=2)
               for name, path in caches.items()}
    # The seen protocol also trains on the room the unseen protocol holds out.
    assert [records[name]['inventory_files'] for name in ('unseen', 'seen')] == [2, 3]
    assert all(records[name]['protocol'] == name and records[name]['split'] == 'train'
               for name in records)
    assert records['unseen']['cache_key'] != records['seen']['cache_key']
    assert records['unseen']['inventory_sha256'] != records['seen']['inventory_sha256']
    for name, other in (('unseen', 'seen'), ('seen', 'unseen')):
        with pytest.raises(ValueError, match='protocol'):
            e7.train_data_identity(protocol_root, name, cache_path=caches[other])
    with pytest.raises(ValueError, match='unknown protocol'):
        e7.train_data_identity(protocol_root, 'Seen', cache_path=caches['unseen'])
    (protocol_root / 'single_channel_ir/Apartments/room/S003_R002_hybrid_IR.wav').write_bytes(b'x')
    for name, cache in caches.items():
        with pytest.raises(ValueError, match='stale'):
            e7.train_data_identity(protocol_root, name, cache_path=cache)


def test_a_legacy_protocol_less_cache_is_read_as_unseen(protocol_root, tmp_path):
    cache = tmp_path / 'legacy.json'
    record = e7.train_data_identity(protocol_root, 'unseen', cache_path=cache, workers=2)
    legacy = json.loads(cache.read_text())  # exp_04/exp_05 caches predate the field
    del legacy['protocol']
    cache.write_text(json.dumps(legacy))
    stored = cache.read_bytes()
    # A legacy hit is normalised in memory only: the caller always sees 'protocol'.
    assert (e7.train_data_identity(protocol_root, 'unseen', cache_path=cache) ==
            dict(legacy, protocol='unseen') == record)
    assert cache.read_bytes() == stored and 'protocol' not in json.loads(stored)
    with pytest.raises(ValueError, match='protocol'):
        e7.train_data_identity(protocol_root, 'seen', cache_path=cache)


def test_the_unseen_cache_key_still_matches_the_shared_helper(protocol_root, tmp_path):
    """exp_07 must never invalidate exp_04's or exp_05's caches."""
    shared = tmp_path / 'shared.json'
    mine = tmp_path / 'mine.json'
    theirs = p.train_data_identity(protocol_root, cache_path=shared, workers=2)
    record = e7.train_data_identity(protocol_root, 'unseen', cache_path=mine, workers=2)
    assert record == dict(theirs, protocol='unseen')
    assert e7.cache_key(Path(protocol_root).resolve(), 'unseen', 2) == theirs['cache_key']
    assert e7.cache_key(Path(protocol_root).resolve(), 'seen', 2) != theirs['cache_key']
    # The shared helper is main's: it takes no protocol and hashes only the unseen split.
    with pytest.raises(TypeError):
        p.train_data_identity(protocol_root, protocol='seen', cache_path=tmp_path / 'x.json')


@pytest.mark.skipif(not REAL_CACHE.is_file(), reason='needs ckpt/exp07/train_inventory_seen.json')
def test_the_planners_seen_inventory_cache_is_still_readable(monkeypatch, tmp_path):
    """The real cache, read-only: its key, count and protocol are this module's."""
    stored = REAL_CACHE.read_bytes()
    record = json.loads(stored)
    assert record['protocol'] == 'seen' and record['split'] == 'train'
    assert record['inventory_files'] == e7.TRAIN_FILES['seen'] == len(record['inventory'])
    assert record['cache_key'] == e7.cache_key(record['data_root'], 'seen', record['inventory_files'])
    assert p._inventory_digest(record['inventory']) == record['inventory_sha256']
    # A hit is decided by the same predicate the module applies; stat the real files.
    root = Path(record['data_root'])
    if root.is_dir():
        stamps = [(r['path'], (root / r['path']).stat().st_size,
                   (root / r['path']).stat().st_mtime_ns) for r in record['inventory'][:64]]
        assert stamps == [(r['path'], r['size'], r['mtime_ns']) for r in record['inventory'][:64]]
    assert REAL_CACHE.read_bytes() == stored  # never written by this test


@pytest.mark.skipif(not os.path.isdir(REAL_DATA_ROOT), reason='needs the AcousticRooms cache')
def test_train_data_identity_counts_the_real_protocol_splits(tmp_path, monkeypatch):
    """Real file selection; hashing is stubbed so the suite never rereads 33 GB twice."""
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(p, 'sha256_file', lambda path: hashlib.sha256(str(path).encode()).hexdigest())
    listed = {}
    for protocol, expected in e7.TRAIN_FILES.items():
        cache = tmp_path / (protocol + '.json')
        record = e7.train_data_identity(REAL_DATA_ROOT, protocol, cache_path=cache)
        assert (record['inventory_files'], record['protocol']) == (expected, protocol)
        assert all(r['path'].startswith('single_channel_ir/') for r in record['inventory'])
        listed[protocol] = {r['path'] for r in record['inventory']}
        cache.unlink()  # a stubbed-digest inventory must never outlive the test
    assert len(listed['seen'] - listed['unseen']) == 5244  # unseen test rooms minus 1093 seen-test pairs
    assert len(listed['unseen'] - listed['seen']) == 5124  # seen-test pairs inside the unseen train rooms


def test_seen_split_identity_binds_the_authors_pickle(tmp_path):
    record = e7.seen_split_identity(ROOT)
    assert record == {'path': e7.SEEN_SPLIT, 'sha256': p.sha256_file(ROOT / record['path'])}
    assert record['sha256'] == subprocess.check_output(
        ['sha256sum', record['path']], cwd=str(ROOT), text=True).split()[0]
    copy = tmp_path / record['path']
    copy.parent.mkdir(parents=True)
    copy.write_bytes((ROOT / record['path']).read_bytes())
    assert e7.seen_split_identity(tmp_path) == record
    fields = {'repo': str(tmp_path), 'mutable_inputs': {'seen_split': record}}
    assert p.revalidate(fields) == []
    copy.write_bytes(b'tampered split')
    assert p.revalidate(fields) == ['seen_split']
    copy.unlink()
    assert p.revalidate(fields) == ['seen_split']


def test_the_shared_module_keeps_mains_bytes():
    assert subprocess.run(['git', 'diff', '--quiet', 'main', '--', 'tools/provenance.py'],
                          cwd=str(ROOT)).returncode == 0
    assert not hasattr(p, 'seen_split_identity') and not hasattr(p, 'PROTOCOLS')
