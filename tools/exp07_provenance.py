"""exp_07 provenance: the protocol-aware training inventory and the seen-split binding.

Amendment A1: ``tools/provenance.py`` keeps main's bytes -- exp_04's bound record
re-hashes its recorded source paths against the live tree -- so the protocol-aware
inventory lives here and reuses ``provenance``'s own hashing helpers unchanged.

Two facts about the inventory are inherited from exp_04/exp_05 and disclosed again:
only the training WAVs are hashed (the metadata and depth-map files of the training
rooms are NOT part of it), and a stale cache raises instead of being silently reused.

The cache is keyed by root, protocol and file count, and every record carries its
``protocol``, so a cache built for one protocol can never satisfy the other.  The unseen
key keeps ``provenance.train_data_identity``'s pre-exp_07 encoding, so exp_07 never
invalidates exp_04's or exp_05's caches; exp_07 itself only ever uses the seen path.
"""
import hashlib
import json
import tempfile
from pathlib import Path

from tools import exp07_train
from tools import provenance as p

PROTOCOLS = exp07_train.PROTOCOLS
SEEN_SPLIT = 'treble_multi_room_dataset/seen_test_split.pkl'
SEEN_CACHE = 'ckpt/exp07/train_inventory_seen.json'
TRAIN_FILES = {'unseen': 296334, 'seen': 296454}


def train_files(data_root, protocol):
    """List one protocol's training WAVs, relative to the resolved data root."""
    if protocol not in PROTOCOLS:
        raise ValueError('unknown protocol: ' + str(protocol))
    # The seen dataset module opens SEEN_SPLIT relative to the cwd: run from the repo root.
    dataset_class = exp07_train.dataset_class(protocol)
    root = Path(data_root).resolve()
    dataset = dataset_class(split='train', ir_path=str(root / 'single_channel_ir'))
    return root, sorted(str(Path(f).resolve().relative_to(root)) for f in dataset.file_list)


def cache_key(root, protocol, count):
    payload = [str(root), count] if protocol == 'unseen' else [str(root), protocol, count]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def train_data_identity(data_root, protocol='seen', cache_path=None, workers=8):
    """Content-hash the training WAVs of ``protocol``, with a stat-validated cache."""
    root, files = train_files(data_root, protocol)
    key = cache_key(root, protocol, len(files))
    cache = (Path(cache_path) if cache_path else
             Path(tempfile.gettempdir()) / 'xrir-provenance' / (key + '.json'))
    if cache.exists():
        wrong_protocol = False
        try:
            record = json.loads(cache.read_text())
            # Records written before exp_07 carry no protocol and are unseen inventories;
            # the key, count and per-file stamps below bind them to the split regardless.
            wrong_protocol = record.get('protocol', 'unseen') != protocol
            stamps = []
            for name in files:
                stat = (root / name).stat()
                stamps.append((name, stat.st_size, stat.st_mtime_ns))
            cached = [(r['path'], r['size'], r['mtime_ns']) for r in record['inventory']]
            if (not wrong_protocol and record['data_root'] == str(root)
                    and record['cache_key'] == key and cached == stamps
                    and record['inventory_files'] == len(files)
                    and p._inventory_digest(record['inventory']) == record['inventory_sha256']):
                # Normalise a legacy (protocol-less) hit in memory; the file stays as written.
                return record if 'protocol' in record else dict(record, protocol='unseen')
        except (OSError, ValueError, KeyError, TypeError):
            pass
        raise ValueError(('training inventory cache holds another protocol: ' if wrong_protocol
                          else 'stale training inventory cache: ') + str(cache))
    record = dict(p._inventory(files, root, workers), split='train', protocol=protocol,
                  cache_key=key)
    cache.parent.mkdir(parents=True, exist_ok=True)
    p.write_completion(cache, record)
    return record


def seen_split_identity(repo):
    """Bind the authors' seen-split pickle as ``mutable_inputs['seen_split']``.

    'seen_split' is an accepted binding name in tools/exp07_eval_launch.MUTABLE_INPUTS and
    in tools/exp07_launcher.SEEN_INPUTS, which requires it on every --protocol seen run.
    ``provenance.revalidate`` rehashes it from ``manifest['repo']`` like every other
    repository-relative mutable input.
    """
    return {'path': SEEN_SPLIT, 'sha256': p.sha256_file(Path(repo) / SEEN_SPLIT)}
