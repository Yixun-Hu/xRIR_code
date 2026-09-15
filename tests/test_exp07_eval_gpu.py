"""GPU parity of the split-bound evaluator: unseen bit parity and the seen final batch.

DEFERRED while both GPUs run trainings; every case skips without CUDA and must pass
before any exp_07 evaluation is admitted.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

import eval_yaw_rotation as frozen
from tools import exp04_eval, exp07_eval, exp07_eval_launch as launcher
from tools import exp07_provenance as e7p
from tools import provenance
from tools.reference_manifest import build_manifest, manifest_hash, save_manifest

ROOT = Path(__file__).resolve().parents[1]
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='the evaluator needs CUDA')
CHECKPOINT = ROOT / 'ckpt/xRIR_simple_8_shot/epoch_12.pth'
SEEN_MANIFESTS = ROOT / 'ckpt/exp07'  # written by tools/exp07_manifests.py
# The only meta keys tools/exp07_eval.py may add or change (plan section 2).
EXTRA_META = {'elapsed_min', 'split', 'seen_split_sha256', 'eval_manifest_sha256'}


def run(output, module, reference, split=None, samples=0, shots=8):
    """One evaluation through an entry point, with the handshake it validates."""
    output.mkdir()
    manifest = frozen.load_manifest(str(reference))
    argv = ['--backbone', 'simple', '--checkpoint', str(CHECKPOINT), '--manifest', str(reference),
            '--manifest-hash', manifest_hash(manifest), '--out-dir', str(output),
            '--eval-manifest', str(output / 'eval_manifest.json'), '--conditions', 'P',
            '--max-samples', str(samples), '--gl-seed', '42', '--yaw-cols', '0',
            '--acoustic-cols', '0', '--e-acoustic-cols', '--decomposition-batches', '0',
            '--num-workers', '0', '--threads', '2'] + (['--split', split] if split else [])
    args = module.parse_args(argv)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
    records, digest = provenance.closure_record(
        provenance.source_closure('eval_yaw_rotation', ROOT), commit, ROOT)
    fields = {key: value for key, value in vars(args).items()
              if key not in ('out_dir', 'eval_manifest', 'manifest')}
    fields.update(schema_version=1, repo=str(ROOT), reviewed_commit=commit,
                  manifest_path=str(reference), manifest_seed=manifest['seed'],
                  num_shot=shots, batch_canonical=True, data_root=exp04_eval.BASE_DATA_PATH,
                  manifest_file_sha256=provenance.sha256_file(reference),
                  checkpoint_sha256=provenance.sha256_file(CHECKPOINT),
                  evaluator_closure={'files': records, 'sha256': digest})
    if module is exp07_eval:
        fields.update(exp07_eval.split_metadata(args))
    provenance.write_manifest(Path(args.eval_manifest), fields)
    (exp07_eval.run_exp07 if module is exp07_eval else exp04_eval.run_exp04)(args)
    return [json.loads((output / name).read_text())
            for name in ('per_sample_yaw.json', 'metrics_yaw.json')]


@GPU
def test_unseen_32_query_run_is_bit_identical_to_exp04(tmp_path):
    reference = ROOT / 'ckpt/yaw_aug/reference_manifest_k8_seed42.json'
    if not reference.exists() or not CHECKPOINT.exists():
        pytest.skip('requires the exp_04 reference manifest and the K8 checkpoint')
    previous = run(tmp_path / 'exp04', exp04_eval, reference, samples=32)
    actual = run(tmp_path / 'exp07', exp07_eval, reference, split='unseen', samples=32)
    for old, new in zip(previous, actual):
        assert new['meta']['split'] == 'unseen'
        assert new['meta']['seen_split_sha256'] == provenance.sha256_file(
            ROOT / e7p.SEEN_SPLIT)
        differing = {key for key in set(old['meta']) | set(new['meta'])
                     if old['meta'].get(key) != new['meta'].get(key)}
        assert differing <= EXTRA_META and {'split', 'eval_manifest_sha256'} <= differing
        assert set(new['meta']) - set(old['meta']) == {'split', 'seen_split_sha256'}
        for payload in (old, new):
            payload['meta'] = {k: v for k, v in payload['meta'].items() if k not in EXTRA_META}
        assert new == old  # query order, per-sample arrays, null masks and aggregates


def direct_reference(dataset, reference, shots, batch_size=16, gl_seed=42, max_samples=0):
    """The frozen functions called directly on the seen dataset, without the entry point.

    The reference is ``eval_yaw_rotation.evaluate_batch`` -- exp_03's pinned code -- and
    not ``exp04_eval.evaluate_p_batch``, which is the implementation under test: at k = 0
    condition P is the same fixed-alignment forward pass, so a common error in the exp_04
    copy cannot pass both sides.
    """
    frozen.set_precision(False)
    frozen.torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = False
    data = frozen.ManifestDataset(dataset, frozen.load_manifest(str(reference)))
    if max_samples:
        data = frozen.SubsetManifestDataset(data, max_samples)
    loader = frozen.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    model = frozen.build_xrir('simple', shots)
    model.load_state_dict(frozen.load_model_state(str(CHECKPOINT)), strict=True)
    model.cuda().eval()
    evaluator, queries, parts = frozen.Evaluator(), [], {}
    for batch in loader:
        keys, results, _ = frozen.evaluate_batch(model, batch, evaluator, [0], [0], (),
                                                 gl_seed, batch_size=batch_size)
        queries.extend(keys)
        for metric, values in results[('P', 0)].items():
            parts.setdefault(metric, []).append(values)
    return queries, {metric: frozen.np.concatenate(chunks) for metric, chunks in parts.items()}


def visible_gpu():
    """The physical index this process evaluates on, so the child uses the same device."""
    visible = [item for item in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if item]
    return visible[torch.cuda.current_device()] if visible else str(torch.cuda.current_device())


@GPU
@pytest.mark.parametrize('shots', [8, 1])
def test_seen_final_batch_matches_a_direct_frozen_call(tmp_path, monkeypatch, shots):
    """25 queries = one canonical batch plus the seen split's own final batch of 9."""
    if not CHECKPOINT.exists():
        pytest.skip('requires the K8 checkpoint')
    from treble_multi_room_dataset import treble_xRIR_seen_dataset as seen
    dataset = seen.xRIR_Dataset(split='test', num_shot=shots, max_len=exp07_eval.MAX_LEN)
    assert len(dataset.file_list) == exp07_eval.SPLIT_ENTRIES['seen']  # 6217 = 388 * 16 + 9
    dataset.file_list = sorted(dataset.file_list,
                               key=lambda path: os.path.relpath(path, dataset.ir_path))[-25:]
    monkeypatch.setattr(seen, 'xRIR_Dataset', lambda **kwargs: dataset)
    reference = tmp_path / 'reference_seen.json'
    save_manifest(build_manifest(dataset, seed=42, num_shot=shots), str(reference))
    per_sample, metrics = run(tmp_path / 'seen', exp07_eval, reference, split='seen', shots=shots)
    queries, expected = direct_reference(dataset, reference, shots)
    assert per_sample['query'] == queries and len(queries) % 16 == 9
    assert per_sample['meta']['batch_canonical'] is True and per_sample['meta']['n_samples'] == 25
    assert per_sample['meta']['split'] == 'seen' and per_sample['meta']['tf32'] is False
    for metric, values in expected.items():
        assert per_sample['P']['0'][metric] == frozen._json_values(values)
        assert metrics['P']['0'][metric] == frozen._summarize(values)


@GPU
@pytest.mark.parametrize('shots', [8, 1])
def test_the_launcher_runs_the_seen_split_and_matches_the_frozen_functions(tmp_path, shots):
    """The planned invocation: --split seen, bounded to one batch of 16 plus a final 9.

    Unlike the parity cases above this goes through tools/exp07_eval_launch.py, so the
    real handshake, the split bindings and the finalisation gates are exercised.  It
    needs the seen reference manifests of tools/exp07_manifests.py (the launcher hashes
    every file the manifest references), so it skips until the Planner has built them.
    """
    reference = SEEN_MANIFESTS / 'reference_manifest_seen_k{}_seed42.json'.format(shots)
    if not reference.exists() or not CHECKPOINT.exists():
        pytest.skip('requires the seen reference manifests and the K8 checkpoint')
    from treble_multi_room_dataset import treble_xRIR_seen_dataset as seen
    manifest = frozen.load_manifest(str(reference))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
    run = tmp_path / 'launched'
    completion = launcher.main([
        '--split', 'seen', '--backbone', 'simple', '--checkpoint',
        str(CHECKPOINT), '--manifest', str(reference), '--manifest-hash', manifest_hash(manifest),
        '--out-dir', str(run), '--log-dir', str(tmp_path / 'logs'), '--run-label', 'seen',
        '--data-root', exp04_eval.BASE_DATA_PATH, '--reviewed-commit', commit,
        '--gpu', visible_gpu(), '--num-shot', str(shots), '--max-samples', '25',
        '--gl-seed', '42', '--conditions', 'P', '--yaw-cols', '0', '--acoustic-cols', '0',
        '--e-acoustic-cols', '--decomposition-batches', '0', '--num-workers', '0',
        '--threads', '2'])
    digest = provenance.sha256_file(ROOT / e7p.SEEN_SPLIT)
    fields = json.loads((run / 'eval_manifest.json').read_text())
    assert fields['split'] == 'seen' and fields['seen_split_sha256'] == digest
    assert fields['mutable_inputs']['seen_split'] == e7p.seen_split_identity(ROOT)
    assert fields['split_count'] == exp07_eval.SPLIT_ENTRIES['seen'] and fields['n_samples'] == 25
    assert exp07_eval.SEEN_DATASET_SOURCE in [record['path'] for record
                                              in fields['source_closures']['entrypoint']['files']]
    assert completion['eval_manifest_sha256'] == provenance.sha256_file(run / 'eval_manifest.json')
    assert completion['confirmatory'] is False  # 25 of 6 217 queries
    dataset = seen.xRIR_Dataset(split='test', num_shot=shots, max_len=exp07_eval.MAX_LEN)
    queries, expected = direct_reference(dataset, reference, shots, max_samples=25)
    per_sample, metrics = [json.loads((run / name).read_text()) for name in launcher.OUTPUTS]
    assert per_sample['query'] == queries and len(queries) % 16 == 9  # 16 + a short final batch
    assert per_sample['meta']['split'] == 'seen' and per_sample['meta']['tf32'] is False
    for metric, values in expected.items():
        assert per_sample['P']['0'][metric] == frozen._json_values(values)
        assert metrics['P']['0'][metric] == frozen._summarize(values)
