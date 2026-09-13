"""Reusable byte-bound exp05 producer fixtures; import from the tests directory."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import provenance as p
from tools.exp05_profiles import get_profile, json_value
from tools.reference_manifest import manifest_hash
from test_paired_compare import _canonical_digest, _read, _replace, _rebind, _summaries


def _source(root, name):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('# synthetic fixture ' + name + '\n')
    sha = p.sha256_file(path)
    files = [dict(path=name, reviewed_blob_sha256=sha, working_tree_sha256=sha,
                  commits_after_reviewed=[], mtime='2026-09-12T00:00:00+00:00')]
    return dict(files=files, sha256=hashlib.sha256(json.dumps([[name, sha]], sort_keys=True).encode()).hexdigest())


@pytest.fixture
def exp05_fixture(tmp_path):
    def build(name='CURVE_K8', m_exp04=False):
        root = tmp_path / (name + ('_legacy' if m_exp04 else ''))
        root.mkdir()
        profile = json_value(get_profile(name))
        queries = sorted('Room/room_{}/S00{}_R001_hybrid_IR.wav'.format(i // 4, i + 1) for i in range(12))
        profile['dataset'].update(n_queries=12, n_rooms=3, query_sha256=_canonical_digest(queries))
        frozen, entry, old_entry, writer, training = [_source(root, path) for path in (
            'eval_yaw_rotation.py', 'tools/exp05_eval.py', 'tools/exp04_eval.py',
            'tools/exp04_eval_launch.py', 'tools/exp04_launcher.py')]
        producer = dict(sha256='a' * 64, files=[], commit='b' * 40)
        pins = dict(schema_version=1, closures=dict(evaluator=entry['sha256'],
            evaluator_exp04=old_entry['sha256'], writer=writer['sha256'],
            training_launcher=training['sha256'], producer_param_curve=producer['sha256']), checkpoints={})
        data_root = root / 'data'
        data_root.mkdir()
        (data_root / 'query.dat').write_bytes(b'synthetic dataset bytes')
        identity = p._inventory(['query.dat'], data_root)
        profile['dataset']['inventory_sha256'] = identity['inventory_sha256']
        references = {}
        for seed in profile['eval_seeds']:
            reference = dict(seed=seed, num_shot=profile['num_shot'], ir_root=str(data_root),
                entries=[dict(index=i, query=query, refs=[query.rsplit('/', 1)[0] +
                    '/S099_R001_hybrid_IR.wav'] * profile['num_shot']) for i, query in enumerate(queries)])
            path = root / ('reference_{}.json'.format(seed))
            p.write_manifest(path, reference)
            profile['seeds'][profile['num_shot']][seed] = manifest_hash(reference)
            references[seed] = (path, reference)
        paths = {}
        scales = dict(S_simple=1.2, S_cyl=1.01, M_simple=1.0, M_cyl=.75, L_simple=.85, L_cyl=.8)
        for arm in profile['arms']:
            role, legacy = arm['role'], arm['tier'] == 'M'
            attempt = root / role / 'final'
            attempt.mkdir(parents=True)
            checkpoint = attempt / ('epoch_12.pth' if legacy else 'epoch_012.pth')
            checkpoint.write_bytes(('synthetic checkpoint ' + role).encode())
            arm.update(checkpoint=str(checkpoint), sha256=p.sha256_file(checkpoint))
            if not legacy:
                pins['checkpoints'][role] = dict(path=str(checkpoint), epoch=12, sha256=arm['sha256'])
            args = dict(backbone=arm['backbone'], save_dir=str(attempt), num_shot=8, max_len=9600,
                lr=.001, weight_decay=.0001, decay_epochs=3, lr_gamma=.1, epochs=12, batch_size=32,
                accum_steps=2, num_workers=12, seed=0, tf32=True, log_interval=50,
                save_every=500 if legacy else 0, epoch_ckpt_every=5 if legacy else 1,
                resume=None, max_train_batches=0, max_test_batches=0, test_subset=0)
            config = {'vit_' + key: value for key, value in arm['config'].items()}
            if not legacy:
                args.update(config, tier=arm['tier'], param_counts=arm['counts'], yaw_aug=0,
                    yaw_aug_seed=0, yaw_aug_width=512, no_save=False, train_batches_per_epoch=9261,
                    env=dict(XRIR_DATA_PATH=str(data_root), PYTHONHASHSEED='0', OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='0'))
            args_path = attempt / 'args.json'
            p.write_manifest(args_path, args)
            binding = dict(path=str(args_path), sha256=p.sha256_file(args_path))
            tier = dict(config, tier=arm['tier'], param_counts=arm['counts'], legacy_M=legacy,
                        args_json_sha256=binding['sha256'])
            bindings = {'control_args' if legacy and m_exp04 else 'train_args': binding}
            if not legacy:
                train_manifest = attempt / 'train_manifest.json'
                p.write_manifest(train_manifest, dict(repo=str(root), reviewed_commit='b' * 40,
                    mode='full', effective_args=args, source_closures=dict(launcher=training), mutable_inputs={}))
                train_log = attempt / 'train.log'
                train_log.write_text('synthetic training completed\n')
                train_completion = attempt / 'completion.json'
                p.write_completion(train_completion, dict(train_manifest_sha256=p.sha256_file(train_manifest),
                    source_drift_after_spawn=[], log=dict(path=str(train_log), sha256=p.sha256_file(train_log)),
                    outputs={checkpoint.name: arm['sha256'], 'args.json': binding['sha256']}))
                for key, path in (('train_manifest', train_manifest), ('train_completion', train_completion)):
                    bindings[key] = dict(path=str(path), sha256=p.sha256_file(path))
            paths[role] = []
            for seed in profile['eval_seeds']:
                directory = root / '{}_{}'.format(role, seed)
                directory.mkdir()
                paths[role].append(str(directory))
                ref_path, reference = references[seed]
                grid = profile['run_grids'][role]
                chosen_entry = old_entry if legacy and m_exp04 else entry
                fields = dict(schema_version=1, repo=str(root), reviewed_commit='b' * 40,
                    checkpoint=str(checkpoint), checkpoint_sha256=arm['sha256'],
                    manifest_path=str(ref_path), manifest_file_sha256=p.sha256_file(ref_path),
                    manifest_hash=manifest_hash(reference), manifest_seed=seed, gl_seed=seed,
                    num_shot=profile['num_shot'], backbone=arm['backbone'], batch_size=16, batch_canonical=True,
                    max_samples=0, tf32=False, conditions='P', yaw_cols=grid, acoustic_cols=grid,
                    e_acoustic_cols=[], decomposition_batches=0, n_samples=12, split_count=12,
                    split='unseen', no_tta=True, data_root=str(data_root), mutable_inputs=bindings,
                    confirmatory=True, allow_dirty_used=False, evaluator_closure=frozen,
                    source_closures=dict(entrypoint=chosen_entry, writer=writer),
                    data_identity=dict(identity, manifest_path=str(ref_path), manifest_hash=manifest_hash(reference),
                        manifest_file_sha256=p.sha256_file(ref_path)))
                if not (legacy and m_exp04):
                    fields.update(tier)
                digest = p.write_manifest(directory / 'eval_manifest.json', fields)
                keys = ('backbone', 'checkpoint', 'manifest_hash', 'gl_seed', 'batch_size', 'max_samples',
                    'tf32', 'manifest_path', 'manifest_seed', 'yaw_cols', 'acoustic_cols', 'e_acoustic_cols',
                    'batch_canonical', 'n_samples', 'conditions', 'reviewed_commit')
                meta = {key: fields[key] for key in keys}
                meta.update(eval_manifest_sha256=digest, evaluator_closure_sha256=frozen['sha256'], elapsed_min=.1)
                if not (legacy and m_exp04):
                    meta.update(tier)
                cells = {str(k): {metric: [(1 + i / 16 + (seed - 42) / 100) * scales[role]
                    + .0001 * np.sin(i + seed + k) for i in range(12)]
                    for metric in ('edt', 'c50', 't60', 'loss', 'log_mse')} for k in grid}
                sample = dict(meta=meta, query=queries, index=list(range(12)), P=cells,
                    delay_flips={str(k): 0 for k in grid}, decomposition=None)
                p.write_manifest(directory / 'per_sample_yaw.json', sample)
                p.write_manifest(directory / 'metrics_yaw.json', dict(meta=meta, P=_summaries(cells),
                    delay_flips=sample['delay_flips'], decomposition=None))
                log = root / '{}_{}.log'.format(role, seed)
                log.write_text('synthetic evaluation completed\n')
                completion = dict(schema_version=1, confirmatory=True, allow_dirty_used=False,
                    eval_manifest_sha256=digest, child_exit_status=0,
                    directory_listing=['eval_manifest.json', 'metrics_yaw.json', 'per_sample_yaw.json'],
                    log=dict(path=str(log), sha256=p.sha256_file(log)),
                    outputs={key: p.sha256_file(directory / key) for key in ('per_sample_yaw.json', 'metrics_yaw.json')})
                if not (legacy and m_exp04):
                    completion.update(tier)
                p.write_completion(directory / 'completion.json', completion)
        approval_path = root / 'approved.json'
        p.write_manifest(approval_path, pins)
        receipt = dict(path=str(approval_path), sha256=p.sha256_file(approval_path), git_blob='c' * 40)
        return SimpleNamespace(root=root, profile=profile, paths=paths, producer=producer, pins=pins,
            approved=(pins, receipt), directories=[d for group in paths.values() for d in group],
            queries=queries, read=_read, replace=_replace, rebind=_rebind, summaries=_summaries)
    return build
