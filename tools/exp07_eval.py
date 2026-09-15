"""Split-bound evaluation (seen or unseen) with exp_04's numerical path, restated.

Amendment A1: ``tools/exp04_eval.py`` keeps main's bytes, so the dataset choice cannot be
a parameter of ``run_exp04``.  Every numerical function used below is imported unchanged
-- the frozen ``eval_yaw_rotation`` ones through ``tools.exp04_eval``, whose
``build_parser``, ``validate_manifest`` and ``evaluate_p_batch`` are reused as they are --
and :func:`run_exp07` restates ``run_exp04``'s loop with exactly one difference: which
test split the manifest's queries are served from.  ``--split unseen`` therefore
reproduces ``tools/exp04_eval`` bit for bit, the metadata exceptions being ``elapsed_min``
(wall time), ``split`` and ``seen_split_sha256``.

The frozen ``eval_yaw_rotation.build_manifest_dataset`` hard-codes the UNSEEN test split,
so a seen reference manifest (6 217 queries) would be rejected by ``ManifestDataset``'s
exact query-list check; ``--split seen`` builds the same wrapper around the frozen seen
dataset (``eval_xRIR_backbone.build_dataset``, also pinned).

Run it from the repository root: the seen dataset module opens
``treble_multi_room_dataset/seen_test_split.pkl`` relative to the working directory.
That file is bound by digest in the evaluation manifest and in both output metas, and
re-checked after the run.  The seen dataset *module* is bound as well: the frozen helper
imports it inside a function, so this entry point imports it at module scope and
``source_closure('tools.exp07_eval')`` -- the digest pinned for the evaluator -- covers
the code that selects and serves the seen queries.
"""
import json
import time
from pathlib import Path

from eval_xRIR_backbone import build_dataset
from tools import exp04_eval as base
from tools import exp07_provenance as e7p
from tools import provenance
from treble_multi_room_dataset import treble_xRIR_seen_dataset as seen_module

yaw = base.yaw
REPO = Path(__file__).resolve().parents[1]
# Derived from the import above, so the binding cannot be dropped as an unused import.
SEEN_DATASET_SOURCE = str(Path(seen_module.__file__).resolve().relative_to(REPO))
SPLITS = ('unseen', 'seen')
SPLIT_ENTRIES = {'unseen': 6337, 'seen': 6217}  # queries of each test split
SPLIT_FIELDS = ('split', 'seen_split_sha256')
MAX_LEN = 9600


def build_parser(require_eval_manifest=True):
    parser = base.build_parser(require_eval_manifest)
    parser.add_argument('--split', choices=SPLITS, required=True)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def split_metadata(args):
    """The split identity recorded in the eval manifest and in both output metas."""
    return dict(split=args.split, seen_split_sha256=provenance.sha256_file(REPO / e7p.SEEN_SPLIT))


def build_split_dataset(args, manifest, max_samples=0):
    """The manifest's queries served from the requested split.

    ``--split unseen`` calls the frozen builder unchanged.  ``--split seen`` mirrors it
    with the seen dataset class.  Either way ``ManifestDataset`` compares the manifest's
    query list with the split's ``file_list``, so a manifest of the other split is
    refused before any model is built.
    """
    if args.split == 'unseen':
        return yaw.build_manifest_dataset(manifest, max_samples=max_samples, max_len=MAX_LEN)
    dataset = build_dataset(args.split, manifest['num_shot'], MAX_LEN)
    full = yaw.ManifestDataset(dataset, manifest)
    if int(max_samples) > 0:
        return yaw.SubsetManifestDataset(full, min(int(max_samples), len(full)))
    return full


def validate_manifest(args, metadata=None):
    """The frozen preflight, plus the split identity the eval manifest must declare."""
    fields, digest, reference = base.validate_manifest(args)
    expected = metadata or split_metadata(args)
    mismatches = [key for key, value in expected.items() if fields.get(key) != value]
    if mismatches:
        raise ValueError('evaluation manifest mismatch: ' + ', '.join(sorted(mismatches)))
    return fields, digest, reference


def run_exp07(args):
    """``run_exp04``'s loop with the split-bound dataset and the split recorded."""
    metadata = split_metadata(args)
    fields, digest, manifest = validate_manifest(args, metadata)
    if not yaw.torch.cuda.is_available():
        raise RuntimeError('exp07_eval needs a GPU: xRIR.apply_delay is .cuda()-only')
    started = time.time()
    yaw.torch.set_num_threads(args.threads)
    yaw.set_precision(args.tf32)
    manifest = yaw.load_checked_manifest(args.manifest, args.manifest_hash)
    cols, acoustic, e_acoustic = yaw._check_cols(args.yaw_cols, args.acoustic_cols, args.e_acoustic_cols)
    dataset = build_split_dataset(args, manifest, max_samples=args.max_samples)
    loader = yaw.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    model = yaw.build_xrir(args.backbone, manifest['num_shot'])
    model.load_state_dict(yaw.load_model_state(args.checkpoint), strict=True)
    model.cuda().eval()
    evaluator = yaw.Evaluator()
    n_samples = len(dataset)
    print('backbone: {}  checkpoint: {}  samples: {}  angles: {}  batch: {}'.format(
        args.backbone, args.checkpoint, n_samples, cols, args.batch_size), flush=True)
    queries, parts, flips, decompositions = [], {}, {}, []
    evaluate = yaw.evaluate_batch if args.conditions == 'PE' else base.evaluate_p_batch
    for i, batch in enumerate(loader):
        keys, results, batch_flips = evaluate(model, batch, evaluator, cols, acoustic,
                                              e_acoustic, args.gl_seed, batch_size=args.batch_size)
        queries.extend(keys)
        for cell, metrics in results.items():
            for metric, values in metrics.items():
                parts.setdefault(cell, {}).setdefault(metric, []).append(values)
        for k, count in batch_flips.items():
            flips[k] = flips.get(k, 0) + count
        if args.backbone == 'cylindrical' and i < args.decomposition_batches:
            # The unchanged diagnostic internally computes P and E forwards.
            decompositions.append(yaw._decomposition_for_batch(model, batch))
        if (i + 1) % args.log_interval == 0 or len(queries) == n_samples:
            rate = len(queries) / (time.time() - started)
            print('[{}/{}] {:.2f} samples/s, eta {:.1f} min'.format(
                len(queries), n_samples, rate, (n_samples - len(queries)) / rate / 60), flush=True)
    entries = dataset.entries
    if queries != [entry['query'] for entry in entries]:
        raise ValueError('the loader did not return the manifest\'s canonical order')
    merged = {cell: {metric: yaw.np.concatenate(chunks) for metric, chunks in metrics.items()}
              for cell, metrics in parts.items()}
    decomposition = None
    if decompositions:
        names = ('tokens_rel_change', 'pooled_rel_change', 'coord_rel_change', 'logspec_rel_change')
        decomposition = {'k': decompositions[0]['k'], 'n_batches': len(decompositions)}
        decomposition.update({name: float(yaw.np.mean([d[name] for d in decompositions])) for name in names})
    meta = {key: getattr(args, key) for key in ('backbone', 'checkpoint', 'manifest_hash',
            'gl_seed', 'batch_size', 'max_samples', 'tf32')}
    meta.update(manifest_path=args.manifest, manifest_seed=manifest['seed'], yaw_cols=cols,
                acoustic_cols=acoustic, e_acoustic_cols=e_acoustic, batch_canonical=True,
                n_samples=n_samples, torch_version=yaw.torch.__version__,
                elapsed_min=(time.time() - started) / 60, eval_manifest_sha256=digest,
                conditions=args.conditions, evaluator_closure_sha256=fields['evaluator_closure']['sha256'],
                reviewed_commit=fields['reviewed_commit'])
    meta.update(metadata)
    per_sample = {'meta': meta, 'query': queries, 'index': [entry['index'] for entry in entries],
                  'delay_flips': {str(k): int(v) for k, v in sorted(flips.items())},
                  'decomposition': decomposition}
    metrics_out = {'meta': meta, 'delay_flips': per_sample['delay_flips'], 'decomposition': decomposition}
    for condition in args.conditions:
        per_sample[condition] = {str(k): {name: yaw._json_values(values) for name, values in
                                merged[(condition, k)].items()} for k in cols}
        metrics_out[condition] = {str(k): {name: yaw._summarize(values) for name, values in
                                 merged[(condition, k)].items()} for k in cols}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for name, payload in (('per_sample_yaw.json', per_sample), ('metrics_yaw.json', metrics_out)):
        path = str(Path(args.out_dir) / name)
        yaw._write_json(payload, path)
        print('wrote {}'.format(path), flush=True)
    for name, path, expected in (('checkpoint', args.checkpoint, fields['checkpoint_sha256']),
                                 ('manifest', args.manifest, fields['manifest_file_sha256']),
                                 ('eval_manifest', args.eval_manifest, digest)):
        if provenance.sha256_file(path) != expected:
            raise ValueError('{} changed during evaluation'.format(name))
    check_outputs(args, metadata)
    print('done: {} samples x {} angles x {} conditions in {:.2f} min'.format(
        n_samples, len(cols), len(args.conditions), meta['elapsed_min']), flush=True)
    return per_sample


def check_outputs(args, metadata):
    """Re-read what was written: both metas carry the split the run was admitted under.

    The split pickle is re-hashed here as well, so a file swapped mid-run is refused
    before the launcher can bind the outputs.
    """
    if split_metadata(args) != metadata:
        raise ValueError(e7p.SEEN_SPLIT + ' changed during evaluation')
    for name in ('per_sample_yaw.json', 'metrics_yaw.json'):
        meta = json.loads((Path(args.out_dir) / name).read_text())['meta']
        if any(meta.get(key) != metadata[key] for key in SPLIT_FIELDS):
            raise ValueError('output split metadata mismatch: ' + name)


def main(argv=None):
    return run_exp07(parse_args(argv))


if __name__ == '__main__':
    main()
