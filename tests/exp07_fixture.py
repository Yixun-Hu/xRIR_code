"""A synthetic exp_07 run layout: four roles x two K x five seeds, byte-bound.

Shaped exactly like tools/exp07_eval_launch.py --split seen writes it
(eval manifest, completion, the two outputs, the bound training artefacts and the seen
split pickle), so the producers can be exercised without a GPU or AcousticRooms.
"""
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import exp07_provenance as e7p
from tools import provenance as p
from tools.exp07_profiles import get_profile, json_value
from tools.reference_manifest import manifest_hash
from tools.paired_compare import _closure_digest
from test_paired_compare import _canonical_digest, _read, _replace, _rebind, _summaries

METRICS = ('edt', 'c50', 't60', 'loss', 'log_mse')
SCALES = dict(seen_simple=1.0, seen_cyl=.94, seen_aug=1.08, released_seen=1.21)


def _git(root, *args):
    return subprocess.check_output(['git', '-C', str(root)] + list(args), text=True).strip()


def _commit_sources(root, names):
    """Make the fixture root a repository so reviewed blobs can be resolved.

    tools.provenance.revalidate reads every closure record's reviewed blob with `git show`
    when a binder asks for post-spawn drift, exactly as it does for the live repository.
    """
    _git(root, 'init', '-q')
    _git(root, 'add', *names)
    _git(root, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.com',
         'commit', '-qm', 'fixture sources', '--no-gpg-sign')
    return _git(root, 'rev-parse', 'HEAD')


def _source(root, name):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('# synthetic fixture ' + name + '\n')
    sha = p.sha256_file(path)
    files = [dict(path=name, reviewed_blob_sha256=sha, working_tree_sha256=sha,
                  commits_after_reviewed=[], mtime='2026-09-15T00:00:00+00:00')]
    return dict(files=files, sha256=hashlib.sha256(
        json.dumps([[name, sha]], sort_keys=True).encode()).hexdigest())


@pytest.fixture
def exp07_approval_template():
    """The approval shape, independent of the committed file's fill state."""
    return dict(schema_version=None, closures=dict(
        evaluator=None, writer=None, training_launcher=[], training=None,
        producer_table=None, producer_pairs=None), checkpoints={
            role: dict.fromkeys(('path', 'epoch', 'sha256'))
            for role in ('seen_simple', 'seen_cyl', 'seen_aug')})


@pytest.fixture
def exp07_fixture(tmp_path):
    def build(name='TABLE_SEEN_V1'):
        root = tmp_path / name
        root.mkdir()
        profile = json_value(get_profile(name))
        queries = sorted('Cat/room_{}/S00{}_R001_hybrid_IR.wav'.format(i // 4, i + 1)
                         for i in range(12))
        split = root / e7p.SEEN_SPLIT
        split.parent.mkdir(parents=True)
        split.write_bytes(b'synthetic seen_test_split.pkl')
        profile['dataset'].update(n_queries=12, n_rooms=3, query_sha256=_canonical_digest(queries),
                                  seen_split_sha256=p.sha256_file(split))
        split_binding = dict(path=e7p.SEEN_SPLIT, sha256=p.sha256_file(split))
        names = ('tools/exp07_eval.py', 'tools/exp07_eval_launch.py', 'tools/exp07_launcher.py',
                 'tools/exp07_train.py', 'eval_yaw_rotation.py')
        entry, writer, launcher, training, frozen = [_source(root, name) for name in names]
        commit = _commit_sources(root, names)
        # A producer closure of one REAL repository file: the producers declare their own
        # source as an input and the record binder validates it against this pin, so the
        # records must be self-consistent and the bytes must be the ones on disk.
        repo = Path(__file__).resolve().parents[1]
        own = 'tools/exp07_table.py'
        digest = p.sha256_file(repo / own)
        files = [dict(path=own, reviewed_blob_sha256=digest, working_tree_sha256=digest,
                      commits_after_reviewed=[], mtime='2026-09-15T00:00:00+00:00')]
        producer = dict(sha256=_closure_digest(dict(files=files)), files=files,
                        commit=_git(repo, 'rev-parse', 'HEAD'))
        pins = dict(schema_version=1, closures=dict(
            evaluator=entry['sha256'], writer=writer['sha256'],
            training_launcher=[launcher['sha256']], training=training['sha256'],
            producer_table=producer['sha256'], producer_pairs=producer['sha256']), checkpoints={})
        data_root = root / 'data'
        data_root.mkdir()
        (data_root / 'query.dat').write_bytes(b'synthetic dataset bytes')
        (data_root / 'train.dat').write_bytes(b'synthetic training bytes')
        identity = p._inventory(['query.dat'], data_root)
        profile['dataset']['inventory_sha256'] = identity['inventory_sha256']
        # The training inventory is a different file set from the evaluation one.
        trained_data = p._inventory(['query.dat', 'train.dat'], data_root)
        profile['train_inventory_files'] = trained_data['inventory_files']
        references = {}
        for shot in profile['num_shot']:
            for seed in profile['eval_seeds']:
                reference = dict(seed=seed, num_shot=shot, ir_root=str(data_root),
                    entries=[dict(index=i, query=query, refs=[query.rsplit('/', 1)[0] +
                        '/S099_R001_hybrid_IR.wav'] * shot) for i, query in enumerate(queries)])
                path = root / 'reference_k{}_seed{}.json'.format(shot, seed)
                p.write_manifest(path, reference)
                profile['seeds'][shot][seed] = manifest_hash(reference)
                references[(shot, seed)] = (path, reference)
        paths, attempts = {}, {}
        for arm in profile['arms']:
            role = arm['role']
            attempt = root / role / 'final'
            attempt.mkdir(parents=True)
            attempts[role] = attempt
            checkpoint = attempt / 'epoch_012.pth'
            checkpoint.write_bytes(('synthetic checkpoint ' + role).encode())
            arm.update(checkpoint=str(checkpoint), sha256=p.sha256_file(checkpoint))
            bindings = {'seen_split': dict(split_binding)}
            if arm['reference']:
                p.write_manifest(attempt / 'release_note.json', dict(source='authors'))
            else:
                pins['checkpoints'][role] = dict(path=str(checkpoint), epoch=12, sha256=arm['sha256'])
                for epoch in range(1, 12):  # the full run's twelve epoch checkpoints
                    (attempt / ('epoch_%03d.pth' % epoch)).write_bytes(
                        'synthetic {} epoch {}'.format(role, epoch).encode())
                (attempt / 'history.jsonl').write_text(''.join(json.dumps(dict(
                    epoch=epoch, train_loss=1., test_loss=1., epoch_minutes=140.)) + '\n'
                    for epoch in range(1, 13)))
                args = dict(profile['recipe'], **profile['full_run'])
                args.update(backbone=arm['backbone'], save_dir=str(attempt), yaw_aug=arm['yaw_aug'],
                    yaw_aug_seed=arm['yaw_aug_seed'], yaw_aug_width=arm['yaw_aug_width'],
                    vit_dim=512, vit_depth=12, vit_heads=8, vit_mlp_dim=512, tier='M',
                    train_batches_per_epoch=profile['train_batches_per_epoch'],
                    param_counts=dict(encoder=19703296, full=32075965),
                    env=dict(XRIR_DATA_PATH=str(data_root), PYTHONHASHSEED='0',
                             OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='1'))
                args_path = attempt / 'args.json'
                p.write_manifest(args_path, args)
                # tools.exp04_launcher.normalize: the launcher's effective args carry the
                # environment keys at the top level, the trainer's args.json nests them.
                effective = {key: value for key, value in args.items() if key != 'env'}
                effective.update(args['env'])
                effective_path = attempt / 'effective_args.json'
                effective_digest = p.write_manifest(effective_path, effective)
                receipt = dict(schema_version=1, tier='M', backbone=arm['backbone'],
                               protocol='seen', yaw_aug=arm['yaw_aug'], passed=True,
                               PROBE_NOT_CLEAN=False, T_epoch=8400., T_run=100800.,
                               reviewed_commit=commit, gpu='1')
                receipt_path = attempt.parent / ('_probe_t_' + role + '.json')
                receipt_digest = p.write_manifest(receipt_path, receipt)
                # Every arm probed once; seen_simple also retried after a slow epoch one.
                history = [dict(attempt=attempt.name, hours=27.5, mode='full'),
                           dict(attempt='_probe_t_arm', hours=.4, mode='probe')]
                probe = attempt.parent / '_probe_t_arm'
                probe.mkdir()
                p.write_manifest(probe / 'train_manifest.json', dict(mode='probe', role=role))
                p.write_completion(probe / 'completion.json',
                                   dict(metrics=dict(probe=dict(T_epoch=receipt['T_epoch']))))
                if role == 'seen_simple':
                    aborted = attempt.parent / 'attempt_first_ABORTED_slow'
                    aborted.mkdir()
                    p.write_completion(aborted / 'abort.json', dict(
                        reason='guard_epoch_one', wall_hours=3.1,
                        exception_message='epoch one exceeds probe acceptance'))
                    p.write_manifest(aborted / 'train_manifest.json',
                                     dict(mode='full', role=role, attempt_path=str(aborted)))
                    history.append(dict(attempt=aborted.name, hours=3.1, mode='full'))
                p.write_manifest(attempt.parent / 'cumulative_hours.json', dict(
                    total_hours=sum(row['hours'] for row in history),
                    ceiling_hours=1.5 * receipt['T_run'] / 3600,
                    probe_projection_hours=receipt['T_run'] / 3600,
                    probe_receipt_sha256=receipt_digest, attempts=history))
                inventory_path = attempt / 'train_inventory.json'
                inventory_digest = p.write_manifest(
                    inventory_path, dict(inventory=trained_data['inventory']))
                trained_identity = dict(
                    {key: value for key, value in trained_data.items() if key != 'inventory'},
                    split='train', protocol='seen', cache_key='e' * 64,
                    inventory_file=dict(path=str(inventory_path), sha256=inventory_digest))
                train_bindings = dict(seen_split=dict(split_binding),
                    effective_args=dict(path=str(effective_path), sha256=effective_digest),
                    train_inventory=dict(trained_identity['inventory_file']),
                    probe_receipt=dict(path=str(receipt_path), sha256=receipt_digest))
                train_manifest = attempt / 'train_manifest.json'
                p.write_manifest(train_manifest, dict(repo=str(root), reviewed_commit=commit,
                    mode='full', protocol='seen', effective_args=effective, allow_dirty=False,
                    attempt_path=str(attempt), train_data_identity=trained_identity,
                    timing_limits=dict(protocol='seen', epoch_seconds=1.05 * receipt['T_epoch'],
                                       projection_hours=receipt['T_run'] / 3600,
                                       ceiling_hours=1.5 * receipt['T_run'] / 3600,
                                       probe_receipt_sha256=receipt_digest),
                    source_closures=dict(launcher=launcher, training=training),
                    mutable_inputs=train_bindings))
                log = attempt.parent / 'train.log'  # the launcher keeps it outside the attempt
                log.write_text('synthetic training completed\n')
                completion = attempt / 'completion.json'
                outputs = {item.name: p.sha256_file(item) for item in sorted(attempt.iterdir())}
                p.write_completion(completion, dict(
                    train_manifest_sha256=p.sha256_file(train_manifest), source_drift_after_spawn=[],
                    log=dict(path=str(log), sha256=p.sha256_file(log)), wall_hours=27.5,
                    outputs=outputs, directory_listing=dict(outputs)))
                for key, path in (('train_args', args_path), ('train_manifest', train_manifest),
                                  ('train_completion', completion)):
                    bindings[key] = dict(path=str(path), sha256=p.sha256_file(path))
            for shot in profile['num_shot']:
                paths[(role, shot)] = []
                for seed in profile['eval_seeds']:
                    directory = root / 'runs' / '{}_k{}_{}'.format(role, shot, seed)
                    directory.mkdir(parents=True)
                    paths[(role, shot)].append(str(directory))
                    reference_path, reference = references[(shot, seed)]
                    fields = dict(schema_version=1, repo=str(root), reviewed_commit=commit,
                        checkpoint=str(checkpoint), checkpoint_sha256=arm['sha256'],
                        manifest_path=str(reference_path), manifest_seed=seed, gl_seed=seed,
                        manifest_file_sha256=p.sha256_file(reference_path),
                        manifest_hash=manifest_hash(reference), num_shot=shot,
                        backbone=arm['backbone'], batch_size=16, batch_canonical=True,
                        max_samples=0, tf32=False, conditions='P', yaw_cols=[0], acoustic_cols=[0],
                        e_acoustic_cols=[], decomposition_batches=0, n_samples=12, split_count=12,
                        split='seen', seen_split_sha256=split_binding['sha256'], no_tta=True,
                        data_root=str(data_root), mutable_inputs=bindings, confirmatory=True,
                        allow_dirty_used=False, evaluator_closure=frozen,
                        source_closures=dict(entrypoint=entry, writer=writer),
                        data_identity=dict(identity, manifest_path=str(reference_path),
                            manifest_hash=manifest_hash(reference),
                            manifest_file_sha256=p.sha256_file(reference_path)))
                    digest = p.write_manifest(directory / 'eval_manifest.json', fields)
                    meta = {key: fields[key] for key in (
                        'backbone', 'checkpoint', 'manifest_hash', 'gl_seed', 'batch_size',
                        'max_samples', 'tf32', 'manifest_path', 'manifest_seed', 'yaw_cols',
                        'acoustic_cols', 'e_acoustic_cols', 'batch_canonical', 'n_samples',
                        'conditions', 'reviewed_commit', 'split', 'seen_split_sha256')}
                    meta.update(eval_manifest_sha256=digest, elapsed_min=.1,
                                evaluator_closure_sha256=frozen['sha256'])
                    cells = {'0': {metric: [(1 + i / 16 + (seed - 42) / 100 + shot / 32)
                        * SCALES[role] + .0001 * np.sin(i + seed) for i in range(12)]
                        for metric in METRICS}}
                    sample = dict(meta=meta, query=queries, index=list(range(12)), P=cells,
                                  delay_flips={'0': 0}, decomposition=None)
                    p.write_manifest(directory / 'per_sample_yaw.json', sample)
                    p.write_manifest(directory / 'metrics_yaw.json', dict(
                        meta=meta, P=_summaries(cells), delay_flips=sample['delay_flips'],
                        decomposition=None))
                    run_log = root / 'runs' / '{}_k{}_{}.log'.format(role, shot, seed)
                    run_log.write_text('synthetic evaluation completed\n')
                    p.write_completion(directory / 'completion.json', dict(
                        schema_version=1, confirmatory=True, allow_dirty_used=False,
                        eval_manifest_sha256=digest, child_exit_status=0,
                        directory_listing=['eval_manifest.json', 'metrics_yaw.json',
                                           'per_sample_yaw.json'],
                        log=dict(path=str(run_log), sha256=p.sha256_file(run_log)),
                        outputs={key: p.sha256_file(directory / key)
                                 for key in ('per_sample_yaw.json', 'metrics_yaw.json')}))
        approval_path = root / 'approved.json'
        p.write_manifest(approval_path, pins)
        receipt = dict(path=str(approval_path), sha256=p.sha256_file(approval_path),
                       git_blob='c' * 40)
        return SimpleNamespace(root=root, profile=profile, pins=pins, producer=producer,
            approved=(pins, receipt), paths=paths, attempts=attempts, split=split,
            directories=[d for group in paths.values() for d in group], queries=queries,
            read=_read, replace=_replace, rebind=_rebind, summaries=_summaries)
    return build
