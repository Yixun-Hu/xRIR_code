"""A synthetic exp_05 record world: six arms x two K x five seeds plus six yaw blocks.

Shaped exactly like ``tools/exp05_eval.py`` and the exp_04 launcher write it, and
carrying the complete training evidence the record binds -- attempt, hours ledger, probe
attempt and its receipt, the external logs the launcher keeps outside the attempt -- so
the record tooling can be exercised without a GPU, AcousticRooms or the real 66 runs.

``tests/exp05_fixture.py`` builds one profile's admission inputs in its own root; the
record needs ONE world whose 66 run directories are shared between the five products,
so this builds that world and publishes the canonical products from it.
"""
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import param_curve as pc
from tools import provenance as p
from tools.exp05_profiles import get_profile, json_value
from tools.paired_compare import _closure_digest
from tools.reference_manifest import manifest_hash
from test_paired_compare import _canonical_digest, _read, _replace, _rebind, _summaries

METRICS = ('edt', 'c50', 't60', 'loss', 'log_mse')
# Relative error levels: every cylindrical arm below its pair, S_cyl equivalent to M_simple.
SCALES = dict(S_simple=1.2, S_cyl=1.01, M_simple=1.0, M_cyl=.75, L_simple=.85, L_cyl=.8)
YAW_GRID = (0, 32, 64, 448, 480)
# (num_shot, run kind) of the runs each registered profile consumes.
PROFILE_RUNS = dict(CURVE_K8=(8, 'k0'), TARGETS_K8=(8, 'k0'), CURVE_K1=(1, 'k0'),
                    TARGETS_K1=(1, 'k0'), YAW_K8_SEED42=(8, 'yaw'))
M_ROOTS = dict(M_simple='ckpt/xRIR_simple_8_shot', M_cyl='ckpt/xRIR_cyl_8_shot')
RECIPE = dict(num_shot=8, max_len=9600, lr=.001, weight_decay=.0001, decay_epochs=3,
              lr_gamma=.1, epochs=12, batch_size=32, accum_steps=2, seed=0, tf32=True)
SOURCES = ('tools/exp05_eval.py', 'tools/exp04_eval_launch.py', 'tools/exp04_launcher.py',
           'train_xRIR_backbone.py', 'eval_yaw_rotation.py')


def _git(root, *args):
    return subprocess.check_output(['git', '-C', str(root)] + list(args), text=True).strip()


def _commit_sources(root, names):
    """Make the world a repository: revalidate resolves reviewed blobs with `git show`."""
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
                  commits_after_reviewed=[], mtime='2026-09-18T00:00:00+00:00')]
    return dict(files=files, sha256=hashlib.sha256(
        json.dumps([[name, sha]], sort_keys=True).encode()).hexdigest())


def _values(role, seed, angle, n_queries):
    """The exp05 admission fixture's value shape: one scale per arm, tiny per-query noise."""
    return [(1 + i / 16 + (seed - 42) / 100) * SCALES[role] + .0001 * np.sin(i + seed + angle)
            for i in range(n_queries)]


@pytest.fixture
def exp05_record_fixture(tmp_path):
    def build(n_queries=12):
        root = tmp_path / 'world'
        root.mkdir()
        queries = sorted('Rooms/room_{}/S00{}_R001_hybrid_IR.wav'.format(i // 4, i + 1)
                         for i in range(n_queries))
        entry, writer, launcher, training, frozen = [_source(root, name) for name in SOURCES]
        commit = _commit_sources(root, SOURCES)
        # One REAL repository file as the producer closure: the binder validates the
        # product's declared producer sources against the bytes on disk at HEAD.
        repo = Path(__file__).resolve().parents[1]
        own = 'tools/param_curve.py'
        own_digest = p.sha256_file(repo / own)
        files = [dict(path=own, reviewed_blob_sha256=own_digest, working_tree_sha256=own_digest,
                      commits_after_reviewed=[], mtime='2026-09-18T00:00:00+00:00')]
        producer = dict(sha256=_closure_digest(dict(files=files)), files=files,
                        commit=_git(repo, 'rev-parse', 'HEAD'))
        pins = dict(schema_version=1, closures=dict(
            evaluator=entry['sha256'], writer=writer['sha256'],
            training_launcher=[launcher['sha256']], training=training['sha256'],
            producer_param_curve=producer['sha256']), checkpoints={})
        data_root = root / 'data'
        data_root.mkdir()
        (data_root / 'query.dat').write_bytes(b'synthetic dataset bytes')
        (data_root / 'train.dat').write_bytes(b'synthetic training bytes')
        identity = p._inventory(['query.dat'], data_root)
        trained_data = p._inventory(['query.dat', 'train.dat'], data_root)
        references, hashes = {}, {8: {}, 1: {}}
        for shot in (8, 1):
            for seed in (42, 43, 44, 45, 46):
                reference = dict(seed=seed, num_shot=shot, ir_root=str(data_root),
                    entries=[dict(index=i, query=query, refs=[query.rsplit('/', 1)[0] +
                        '/S099_R001_hybrid_IR.wav'] * shot) for i, query in enumerate(queries)])
                path = root / 'reference_k{}_seed{}.json'.format(shot, seed)
                p.write_manifest(path, reference)
                hashes[shot][seed] = manifest_hash(reference)
                references[(shot, seed)] = (path, reference)
        arms = json_value(get_profile('CURVE_K8'))['arms']
        runs, attempts, ledgers, receipts = {}, {}, {}, {}
        for arm in arms:
            role, legacy = arm['role'], arm['tier'] == 'M'
            config = {'vit_' + key: value for key, value in arm['config'].items()}
            args = dict(RECIPE, backbone=arm['backbone'], num_workers=12, log_interval=50,
                        resume=None, max_train_batches=0, max_test_batches=0, test_subset=0)
            if legacy:
                attempt = root / M_ROOTS[role]
                attempt.mkdir(parents=True)
                checkpoint = attempt / 'epoch_12.pth'
                args.update(save_dir=str(attempt), save_every=500, epoch_ckpt_every=5)
            else:
                attempt = root / 'ckpt/exp05' / role / 'attempt_20260918T000000'
                attempt.mkdir(parents=True)
                checkpoint = attempt / 'epoch_012.pth'
                args.update(config, save_dir=str(attempt), save_every=0, epoch_ckpt_every=1,
                    tier=arm['tier'], param_counts=arm['counts'], yaw_aug=0, yaw_aug_seed=0,
                    yaw_aug_width=512, no_save=False, train_batches_per_epoch=9261,
                    env=dict(XRIR_DATA_PATH=str(data_root), PYTHONHASHSEED='0',
                             OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='1'))
            checkpoint.write_bytes(('synthetic checkpoint ' + role).encode())
            arm.update(checkpoint=str(checkpoint), sha256=p.sha256_file(checkpoint))
            args_path = attempt / 'args.json'
            p.write_manifest(args_path, args)
            binding = dict(path=str(args_path), sha256=p.sha256_file(args_path))
            bindings = {'train_args': binding}
            tier = dict(config, tier=arm['tier'], param_counts=arm['counts'], legacy_M=legacy,
                        args_json_sha256=binding['sha256'])
            if legacy:
                attempts[role] = None
            else:
                pins['checkpoints'][role] = dict(path=str(checkpoint), epoch=12,
                                                 sha256=arm['sha256'])
                attempts[role] = attempt
                bindings.update(_train_attempt(root, attempt, arm, args, commit, launcher,
                                               training, trained_data, ledgers, receipts))
            for shot in (8, 1):
                for seed in (42, 43, 44, 45, 46):
                    for kind, grid in (('k0', (0,)), ('yaw', YAW_GRID)):
                        if kind == 'yaw' and (shot != 8 or seed != 42):
                            continue
                        runs[(role, shot, seed, kind)] = _eval_run(
                            root, role, arm, shot, seed, kind, grid, queries, commit,
                            references, bindings, tier, dict(entrypoint=entry, writer=writer),
                            frozen, identity, data_root)
        approval_path = root / 'approved.json'
        p.write_manifest(approval_path, pins)
        receipt = dict(path=str(approval_path), sha256=p.sha256_file(approval_path),
                       git_blob='c' * 40)
        results = root / 'results'
        results.mkdir()

        def profile_for(name):
            profile = json_value(get_profile(name))
            profile['dataset'].update(n_queries=n_queries, n_rooms=n_queries // 4,
                query_sha256=_canonical_digest(queries),
                inventory_sha256=identity['inventory_sha256'])
            profile['seeds'] = {8: dict(hashes[8]), 1: dict(hashes[1])}
            profile['arms'] = arms
            return profile

        def run_paths(name):
            shot, kind = PROFILE_RUNS[name]
            return [str(path) for key, path in sorted(runs.items())
                    if key[1] == shot and key[3] == kind]

        def produce(name, directory=None):
            directory = Path(directory) if directory else results
            profile = profile_for(name)
            admitted = pc.admit(run_paths(name), profile, approved=(pins, receipt),
                                producer=producer)
            result = pc.analyze(profile, admitted)
            result.update(profile_name=name, compatibility=admitted['compatibility'])
            json_path = directory / (name + '.json')
            pc.publish(result, admitted, str(json_path), str(directory / (name + '.txt')))
            return json_path

        return SimpleNamespace(root=root, arms=arms, queries=queries, runs=runs,
            attempts=attempts, ledgers=ledgers, receipts=receipts, pins=pins, producer=producer,
            approved=(pins, receipt), approval=receipt, results=results, references=references,
            data_root=data_root, commit=commit, profile_for=profile_for, run_paths=run_paths,
            produce=produce, directories=[str(path) for path in runs.values()],
            read=_read, replace=_replace, rebind=_rebind, summaries=_summaries)
    return build


def _train_attempt(root, attempt, arm, args, commit, launcher, training, trained_data,
                   ledgers, receipts):
    """The launcher's full-run evidence: probe attempt, receipt, ledger, logs, completion."""
    role = arm['role']
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    effective = {key: value for key, value in args.items() if key != 'env'}
    effective.update(args['env'])
    effective_path = attempt / 'effective_args.json'
    effective_digest = p.write_manifest(effective_path, effective)
    probe = attempt.parent / '_probe_20260913T000000_arm'
    probe.mkdir()
    probe_log = logs / ('probe_' + role + '.log')
    probe_log.write_text('synthetic probe completed\n')
    p.write_manifest(probe / 'train_manifest.json', dict(mode='probe', role=role))
    p.write_completion(probe / 'completion.json', dict(
        metrics=dict(probe=dict(T_epoch=3292.)), wall_hours=.04,
        log=dict(path=str(probe_log), sha256=p.sha256_file(probe_log))))
    receipt = dict(schema_version=1, tier=arm['tier'], backbone=arm['backbone'], passed=True,
        PROBE_NOT_CLEAN=False, T_epoch=3292.1, T_run=39505.5, reviewed_commit=commit, gpu='1',
        batch_size=32, accum_steps=2, yaw_aug=0, timed_micro_batches=50,
        warmup_micro_batches=10, train_batches_per_epoch=9261, t_test=38.2, t_save=.43,
        mean_iteration_seconds=.3513, median_iteration_seconds=.3509, min_iteration_seconds=.3430,
        peak_allocated_bytes=11214637568, peak_reserved_bytes=12129927168,
        probe_attempt=dict(path=str(probe), **{part + '_sha256': p.sha256_file(probe / (part + '.json'))
                                               for part in ('train_manifest', 'completion')}))
    receipt_path = attempt.parent / ('_probe_20260913T000000_' + role + '.json')
    receipt_digest = p.write_manifest(receipt_path, receipt)
    receipts[role] = receipt_path
    history = [dict(attempt=probe.name, hours=.04, mode='probe'),
               dict(attempt=attempt.name, hours=11.41, mode='full')]
    ledger_path = attempt.parent / 'cumulative_hours.json'
    p.write_manifest(ledger_path, dict(total_hours=sum(row['hours'] for row in history),
        ceiling_hours=1.5 * receipt['T_run'] / 3600, probe_projection_hours=receipt['T_run'] / 3600,
        probe_receipt_sha256=receipt_digest, attempts=history))
    ledgers[role] = ledger_path
    inventory_path = attempt / 'train_inventory.json'
    inventory_digest = p.write_manifest(inventory_path, dict(inventory=trained_data['inventory']))
    train_identity = dict({key: value for key, value in trained_data.items() if key != 'inventory'},
        split='train', cache_key='e' * 64,
        inventory_file=dict(path=str(inventory_path), sha256=inventory_digest))
    (attempt / 'history.jsonl').write_text(''.join(json.dumps(dict(epoch=epoch, train_loss=.02,
        test_loss=.0180 - .0001 * epoch, epoch_minutes=56.)) + '\n' for epoch in range(1, 13)))
    p.write_manifest(attempt / 'execution.json', dict(train_manifest_sha256='d' * 64, child_pgid=42))
    for name in ['best.pth', 'last.pth'] + ['epoch_%03d.pth' % epoch for epoch in range(1, 12)]:
        (attempt / name).write_bytes('synthetic {} {}'.format(role, name).encode())
    bound = dict(control_args=dict(path=str(attempt / 'args.json'),
                                   sha256=p.sha256_file(attempt / 'args.json')),
                 effective_args=dict(path=str(effective_path), sha256=effective_digest),
                 probe_receipt=dict(path=str(receipt_path), sha256=receipt_digest),
                 train_inventory=dict(train_identity['inventory_file']))
    train_manifest = attempt / 'train_manifest.json'
    p.write_manifest(train_manifest, dict(repo=str(root), reviewed_commit=commit, mode='full',
        allow_dirty=False, allow_cotenant=False, attempt_path=str(attempt), effective_args=effective,
        train_data_identity=train_identity, mutable_inputs=bound,
        timing_limits=dict(epoch_seconds=1.05 * receipt['T_epoch'],
                           projection_hours=receipt['T_run'] / 3600,
                           ceiling_hours=1.5 * receipt['T_run'] / 3600,
                           probe_receipt_sha256=receipt_digest),
        source_closures=dict(launcher=launcher, training=training)))
    log = logs / ('train_' + role + '.log')
    log.write_text('synthetic training completed\n')
    outputs = {item.name: p.sha256_file(item) for item in sorted(attempt.iterdir())}
    p.write_completion(attempt / 'completion.json', dict(wall_hours=history[-1]['hours'],
        train_manifest_sha256=p.sha256_file(train_manifest), source_drift_after_spawn=[],
        log=dict(path=str(log), sha256=p.sha256_file(log)), outputs=outputs,
        directory_listing=dict(outputs), metrics=dict(banner=True, probe=None, test_loss=.0168,
                                                      train_losses=[.02])))
    (attempt.parent / 'final').symlink_to(attempt.name)
    return {key: dict(path=str(path), sha256=p.sha256_file(path)) for key, path in
            (('train_manifest', train_manifest), ('train_completion', attempt / 'completion.json'))}


def _eval_run(root, role, arm, shot, seed, kind, grid, queries, commit, references, bindings,
              tier, closures, frozen, identity, data_root):
    """One evaluation run directory, exactly as tools/exp05_eval.py's launcher writes it."""
    directory = root / 'eval' / '{}_k{}_seed{}_{}'.format(role, shot, seed, kind)
    directory.mkdir(parents=True)
    reference_path, reference = references[(shot, seed)]
    fields = dict(schema_version=1, repo=str(root), reviewed_commit=commit,
        checkpoint=arm['checkpoint'], checkpoint_sha256=arm['sha256'],
        manifest_path=str(reference_path), manifest_file_sha256=p.sha256_file(reference_path),
        manifest_hash=manifest_hash(reference), manifest_seed=seed, gl_seed=seed, num_shot=shot,
        backbone=arm['backbone'], batch_size=16, batch_canonical=True, max_samples=0, tf32=False,
        conditions='P', yaw_cols=list(grid), acoustic_cols=list(grid), e_acoustic_cols=[],
        decomposition_batches=0, n_samples=len(queries), split_count=len(queries), split='unseen',
        no_tta=True, data_root=str(data_root), mutable_inputs=bindings, confirmatory=True,
        allow_dirty_used=False, evaluator_closure=frozen, source_closures=dict(closures),
        data_identity=dict(identity, manifest_path=str(reference_path),
                           manifest_hash=manifest_hash(reference),
                           manifest_file_sha256=p.sha256_file(reference_path)), **tier)
    digest = p.write_manifest(directory / 'eval_manifest.json', fields)
    meta = {key: fields[key] for key in (
        'backbone', 'checkpoint', 'manifest_hash', 'gl_seed', 'batch_size', 'max_samples', 'tf32',
        'manifest_path', 'manifest_seed', 'yaw_cols', 'acoustic_cols', 'e_acoustic_cols',
        'batch_canonical', 'n_samples', 'conditions', 'reviewed_commit')}
    meta.update(eval_manifest_sha256=digest, evaluator_closure_sha256=frozen['sha256'],
                elapsed_min=.1, **tier)
    cells = {str(angle): {metric: _values(role, seed, angle, len(queries)) for metric in METRICS}
             for angle in grid}
    sample = dict(meta=meta, query=queries, index=list(range(len(queries))), P=cells,
                  delay_flips={str(angle): 0 for angle in grid}, decomposition=None)
    p.write_manifest(directory / 'per_sample_yaw.json', sample)
    p.write_manifest(directory / 'metrics_yaw.json', dict(meta=meta, P=_summaries(cells),
                     delay_flips=sample['delay_flips'], decomposition=None))
    log = root / 'logs' / '{}_k{}_seed{}_{}.log'.format(role, shot, seed, kind)
    log.parent.mkdir(exist_ok=True)
    log.write_text('synthetic evaluation completed\n')
    p.write_completion(directory / 'completion.json', dict(schema_version=1, confirmatory=True,
        allow_dirty_used=False, eval_manifest_sha256=digest, child_exit_status=0,
        directory_listing=['eval_manifest.json', 'metrics_yaw.json', 'per_sample_yaw.json'],
        log=dict(path=str(log), sha256=p.sha256_file(log)),
        outputs={name: p.sha256_file(directory / name)
                 for name in ('per_sample_yaw.json', 'metrics_yaw.json')}, **tier))
    return directory
