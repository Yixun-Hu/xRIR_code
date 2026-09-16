"""Build and index the exp_07 seen-split reference manifests (one per K and seed).

The evaluation of exp_07 pins its conditioning sets exactly as exp_03 does, but over
the authors' SEEN test split: every manifest is
``reference_manifest.build_manifest(seen_test_dataset, seed, num_shot)``, written once
and never overwritten, and the index records each manifest's semantic hash, its file
digest and its entry count (6 217 queries) together with the digest of the split file
those queries come from.

Run it from the repository root: the frozen seen dataset module opens
``treble_multi_room_dataset/seen_test_split.pkl`` relative to the working directory.
"""
import argparse
import json
import os
from pathlib import Path

from tools import exp07_provenance as e7p
from tools import provenance as p
from tools.reference_manifest import build_manifest, manifest_hash

REPO = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44, 45, 46)
NUM_SHOTS = (8, 1)
ENTRIES = 6217  # the seen test split (plan section 2)
MAX_LEN = 9600
INDEX = 'reference_manifests_seen_index.json'


def manifest_name(num_shot, seed):
    return 'reference_manifest_seen_k{}_seed{}.json'.format(num_shot, seed)


def seen_dataset(num_shot, data_root=None, max_len=MAX_LEN):
    """The frozen seen test split at ``num_shot`` references, from the repository root."""
    if Path.cwd().resolve() != REPO:
        raise ValueError('build the seen manifests from the repository root: ' + str(REPO))
    if data_root is not None:
        os.environ['XRIR_DATA_PATH'] = str(data_root)
    from treble_multi_room_dataset import treble_xRIR_seen_dataset as seen
    if data_root is not None and Path(seen.BASE_DATA_PATH) != Path(data_root):
        raise ValueError('the seen dataset module was imported with data root '
                         + seen.BASE_DATA_PATH)
    return seen.xRIR_Dataset(split='test', num_shot=num_shot, max_len=max_len)


def build(out_dir, seeds=SEEDS, num_shots=NUM_SHOTS, entries=None, factory=seen_dataset):
    """Write one manifest per (K, seed) and the index binding them; never overwrite.

    The split identity is captured BEFORE any dataset is constructed and re-checked
    before the index is published: a pickle replaced while the grid is being built would
    otherwise be recorded as the source of manifests drawn from the old split (or of a
    mixed grid).  The captured binding is the one stored.
    """
    expected = ENTRIES if entries is None else entries
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / INDEX
    if index_path.exists():
        raise FileExistsError(str(index_path))
    split = e7p.seen_split_identity(REPO)  # before the first dataset reads the pickle
    manifests = {}
    for num_shot in num_shots:
        dataset = factory(num_shot)  # one dataset per K, reused across the seeds
        for seed in seeds:
            path = out / manifest_name(num_shot, seed)
            if path.exists():
                raise FileExistsError(str(path))
            manifest = build_manifest(dataset, seed=seed, num_shot=num_shot)
            if len(manifest['entries']) != expected:
                raise ValueError('{} has {} entries, expected {}'.format(
                    path.name, len(manifest['entries']), expected))
            with path.open('x') as stream:  # exclusive creation
                json.dump(manifest, stream)
            manifests[path.name] = dict(num_shot=num_shot, seed=seed,
                                        entries=len(manifest['entries']),
                                        manifest_hash=manifest_hash(manifest),
                                        file_sha256=p.sha256_file(path))
    if e7p.seen_split_identity(REPO) != split:
        raise ValueError('seen_test_split.pkl changed while the manifests were built')
    index = dict(schema_version=1, entries=expected, num_shots=list(num_shots),
                 seeds=list(seeds), seen_split=split, manifests=manifests)
    p.write_manifest(index_path, index)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out-dir', default=str(REPO / 'ckpt/exp07'))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--num-shots', type=int, nargs='+', default=list(NUM_SHOTS))
    args = parser.parse_args(argv)
    index = build(args.out_dir, args.seeds, args.num_shots,
                  factory=lambda num_shot: seen_dataset(num_shot, data_root=args.data_root))
    print(json.dumps(index, sort_keys=True), flush=True)
    return index


if __name__ == '__main__':
    main()
