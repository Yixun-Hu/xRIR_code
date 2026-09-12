"""Pure-statistics regression tests for the exp_04 confirmatory producer."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


from tools import paired_compare as pc
from tools import provenance as p
from tools.exp04_profiles import get_profile, json_value
from tools.reference_manifest import manifest_hash

import numpy as np
import pytest

from tools import paired_compare as stats


def seed_matrix(values):
    return np.tile(np.asarray(values, dtype=float), (5, 1))


@pytest.mark.parametrize("samples,alpha,expected", [
    ([4, 1, 3, 2], .25, 3), ([4, 1, 3, 2], .01, 4),
    ([1, 2, 2, 9], .5, 2), ([7], .2, 7),
    (list(range(1, 41)), .025, 39), (list(range(1, 11)), .7, 3)])
def test_one_sided_upper_exact_order_statistic(samples, alpha, expected):
    assert stats.one_sided_upper(samples, alpha) == expected


@pytest.mark.parametrize("samples,alpha", [([], .05), ([np.nan], .05),
    ([[1, 2]], .05), ([1, 2], 0), ([1, 2], 1)])
def test_quantile_rejects_invalid_inputs(samples, alpha):
    with pytest.raises(ValueError):
        stats.one_sided_upper(samples, alpha)


def test_tails_and_families():
    samples = np.arange(1, 20001)
    assert stats.bonferroni_alpha(2) == .025
    assert stats.one_sided_upper(samples, stats.bonferroni_alpha(2)) == 19500
    assert stats.one_sided_upper(samples, stats.bonferroni_alpha(8)) == 19875
    assert stats.two_sided_interval(samples, .05) == (500, 19500)
    assert stats.two_sided_interval(samples, .05 / 2) == (250, 19750)
    assert stats.two_sided_interval(samples, 2 * .05 / 8) == (125, 19875)
    assert stats.two_sided_interval(samples, 2 * .05 / 14) == (72, 19929)


def test_five_seed_mean_is_plain_and_propagates_any_invalid():
    matrix = np.arange(20, dtype=float).reshape(5, 4)
    matrix[0, 1], matrix[3, 2] = np.nan, np.inf
    np.testing.assert_allclose(stats.five_seed_mean(matrix), [8, np.nan, np.nan, 11],
                               equal_nan=True)
    with pytest.raises(ValueError, match="five"):
        stats.five_seed_mean(matrix[:4])


def test_masks_joint_and_per_arm_seed_categories():
    a0, ak, b0, bk = (seed_matrix(np.ones(6)) for _ in range(4))
    a0[0, 0], ak[1, 1], b0[2, 2], bk[3, 3] = np.nan, np.inf, np.nan, np.inf
    ak[4, 0] = np.nan  # Already-invalid baseline: not a newly invalid query.
    h1, _ = stats.cell_mask(a0, e0_b=b0)
    np.testing.assert_array_equal(h1, [False, True, False, True, True, True])
    mask, counts = stats.cell_mask(a0, ak, b0, bk)
    np.testing.assert_array_equal(mask, [False, False, False, False, True, True])
    assert counts["joint"] == {"n": 6, "baseline_invalid": 2, "newly_invalid": 2,
                                 "excluded": 4, "valid": 2}
    assert counts["a"]["total"]["baseline_invalid"] == 1
    assert counts["a"]["total"]["newly_invalid"] == 1
    assert counts["a"]["seeds"][42]["baseline_invalid"] == 1
    assert counts["a"]["seeds"][43]["newly_invalid"] == 1
    # Per-seed counts condition on that seed's baseline; all-seed totals use all seeds.
    assert counts["a"]["seeds"][46]["newly_invalid"] == 1
    tost, _ = stats.cell_mask(a0, ak)
    np.testing.assert_array_equal(tost, [False, False, True, True, True, True])
    with pytest.raises(ValueError, match="empty"):
        stats.cell_mask(seed_matrix([np.nan]), seed_matrix([1]))


def test_h1_known_shift_and_resampling_the_ratio():
    control = stats.five_seed_mean(seed_matrix([1, 2, 4, 9]))
    aug = stats.five_seed_mean(seed_matrix([2, 3, 5, 10]))
    result = stats.rho_bootstrap(aug, control, n_boot=100, seed=7)
    indices = np.random.default_rng(7).integers(0, 4, size=(100, 4))
    expected = (aug[indices].mean(1) - control[indices].mean(1)) / control[indices].mean(1)
    np.testing.assert_array_equal(result["samples"], expected)
    assert result["rho"] == .25
    np.testing.assert_array_equal(result["samples"], stats.rho_bootstrap(
        aug, control, n_boot=100, seed=7)["samples"])


def test_h2_known_shift_and_shared_query_and_room_resamples():
    a0 = stats.five_seed_mean(seed_matrix([1, 2, 4, 9]))
    ak, b0, bk = a0 + .1, 2 * a0, 2 * a0 + .8
    result = stats.dk_bootstrap(a0, ak, b0, bk, n_boot=100, seed=3)
    assert result["d"] == pytest.approx(-.075)
    expected = stats.diff_in_diff_bootstrap(a0, ak, b0, bk, n_boot=100, seed=3)
    assert result["d"] == expected["d"]
    draws = np.random.default_rng(3).integers(0, 4, size=(100, 4))
    ra = (ak[draws].mean(1) - a0[draws].mean(1)) / a0[draws].mean(1)
    rb = (bk[draws].mean(1) - b0[draws].mean(1)) / b0[draws].mean(1)
    np.testing.assert_array_equal(result["samples"], ra - rb)
    same = stats.dk_bootstrap(a0, ak, a0, ak, n_boot=100, clusters=[0, 0, 1, 1])
    assert same["unit"] == "cluster"
    assert not same["samples"].any()
    rooms = np.array([0, 0, 1, 1])
    rho = stats.rho_bootstrap(ak, a0, n_boot=20, seed=4, clusters=rooms)
    room_draws = np.random.default_rng(4).integers(0, 2, size=(20, 2))
    expected = []
    for selected in room_draws:
        idx = np.concatenate([np.flatnonzero(rooms == room) for room in selected])
        expected.append((ak[idx].mean() - a0[idx].mean()) / a0[idx].mean())
    np.testing.assert_allclose(rho["samples"], expected, atol=1e-15)


def test_invariant_tost_and_strict_margin_verdicts():
    baseline = stats.five_seed_mean(seed_matrix([1, 2, 4, 9]))
    result = stats.tost_cell(baseline, baseline, .02, .05 / 14, n_boot=100)
    assert result["equivalent"] and result["lo"] == result["hi"] == 0
    assert stats.tost_verdict(-.02, .01, .02) == "equivalence not established"
    assert stats.tost_verdict(-.01, .02, .02) == "equivalence not established"
    assert stats.tost_verdict(0, 0, .02) == "equivalent"
    upper = stats.one_sided_upper([.03] * 100, .025)
    assert stats.h1_verdict(upper, upper) == "not shown"
    assert stats.h1_verdict(.02, .04) == "non-inferior on EDT only"
    assert stats.h1_verdict(.04, .02) == "non-inferior on C50 only"
    assert stats.h1_verdict(.02, .02) == "non-inferior"
    assert not stats.superiority(0)
    assert stats.superiority(-.001)
    assert stats.h2_verdict([-.01] * 4) == "supported"
    assert stats.h2_verdict([-.01, 0, 0, 0]) == "partially supported"
    assert stats.h2_verdict([0] * 4) == "not supported"


def test_convergence_uses_seed_zero_width_and_exact_boundary():
    assert stats.convergence(lambda seed: (0, 10 + seed))["passed"]
    assert not stats.convergence(lambda seed: (0, 1 + .11 * seed))["passed"]
    assert stats.convergence(lambda seed: {"lo": 0, "hi": 0})["passed"]
    assert not stats.convergence(lambda seed: (seed, seed))["passed"]
    assert not stats.convergence(lambda seed: (0, np.nan))["passed"]
    values = np.arange(1, 11, dtype=float)
    def tiny(seed):
        draws = stats.rho_bootstrap(values + 1, values, n_boot=2, seed=seed)["samples"]
        return stats.two_sided_interval(draws, .05)
    assert not stats.convergence(tiny)["passed"]


def test_invalid_decision_bounds_and_convergence_tolerances_fail_closed():
    assert stats.h1_verdict(-np.inf, np.nan) == "not shown"
    assert stats.h2_verdict([-np.inf, np.nan, np.inf, 0]) == "not supported"
    assert not stats.superiority(-np.inf)
    assert stats.tost_verdict(.01, -.01) == "equivalence not established"
    for tolerance in (-.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="tolerance"):
            stats.convergence(lambda seed: (0, 1), tol=tolerance)




def _canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                   separators=(',', ':')).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _replace(path, value):
    path = Path(path)
    if path.exists():
        path.unlink()
    return p.write_manifest(path, value)


def _closure(root, name):
    path = root / (name + '.py')
    path.write_text('# fixture ' + name + '\n')
    digest = p.sha256_file(path)
    files = [{'path': path.name, 'reviewed_blob_sha256': digest,
              'working_tree_sha256': digest, 'commits_after_reviewed': [],
              'mtime': '2026-09-12T00:00:00+00:00'}]
    identity = hashlib.sha256(json.dumps([[path.name, digest]],
                                        sort_keys=True).encode()).hexdigest()
    return {'files': files, 'sha256': identity}


def _summaries(cells):
    result = {}
    for angle, metrics in cells.items():
        result[angle] = {}
        for name, values in metrics.items():
            array = np.asarray(values, dtype=float)
            finite = np.isfinite(array)
            result[angle][name] = {'mean': float(array[finite].mean()) if finite.any() else None,
                                  'n_valid': int(finite.sum()), 'n_nan': int((~finite).sum())}
    return result


def _rebind(run, manifest=None, sample=None, metrics=None):
    """Refresh byte bindings after a deliberate, otherwise valid protocol mutation."""
    manifest = manifest if manifest is not None else _read(run / 'eval_manifest.json')
    digest = _replace(run / 'eval_manifest.json', manifest)
    sample = sample if sample is not None else _read(run / 'per_sample_yaw.json')
    sample['meta']['eval_manifest_sha256'] = digest
    _replace(run / 'per_sample_yaw.json', sample)
    metrics = metrics if metrics is not None else _read(run / 'metrics_yaw.json')
    metrics['meta']['eval_manifest_sha256'] = digest
    _replace(run / 'metrics_yaw.json', metrics)
    completion = _read(run / 'completion.json')
    completion['eval_manifest_sha256'] = digest
    completion['outputs'] = {name: p.sha256_file(run / name)
                             for name in ('per_sample_yaw.json', 'metrics_yaw.json')}
    p.write_completion(run / 'completion.json', completion)


@pytest.fixture
def admission_fixture(tmp_path, monkeypatch):
    def build(name='H1_K8'):
        root = tmp_path / name
        root.mkdir()
        profile = json_value(get_profile(name))
        profile['n_boot'] = 2000
        queries = sorted('Room/room_{}/S00{}_R001_hybrid_IR.wav'.format(i // 4, i + 1)
                         for i in range(12))
        profile['dataset'].update(n_queries=12, n_rooms=3,
                                  query_sha256=_canonical_digest(queries))
        frozen, entry, writer = [_closure(root, item) for item in ('frozen', 'entry', 'writer')]
        producer = {'sha256': 'a' * 64, 'files': [], 'commit': 'b' * 40}
        approved = dict(schema_version=1, closures=dict(evaluator=entry['sha256'],
            writer=writer['sha256'], training_launcher=None,
            producer_paired_compare=producer['sha256'], producer_results_table=None), checkpoints={})
        data_root = root / 'data'
        data_root.mkdir()
        (data_root / 'query.dat').write_bytes(b'fixture dataset bytes')
        identity = p._inventory(['query.dat'], data_root)
        profile['dataset']['inventory_sha256'] = identity['inventory_sha256']
        paths = [[], []]
        for arm_index, arm in enumerate(profile['arms']):
            checkpoint = root / (arm['role'] + '.pth')
            checkpoint.write_bytes(('fixture checkpoint ' + arm['role']).encode())
            arm.update(checkpoint=str(checkpoint), sha256=p.sha256_file(checkpoint))
        approved['checkpoints']['aug'] = dict(path=profile['arms'][0]['checkpoint'],
            epoch=12, sha256=profile['arms'][0]['sha256'])
        approval_path = root / 'approved.json'
        def approval():
            _replace(approval_path, approved)
            return approved, dict(path=str(approval_path), sha256=p.sha256_file(approval_path), git_blob='c' * 40)
        monkeypatch.setattr(pc, 'load_approved_digests', approval)
        references = {}
        for seed in range(42, 47):
            reference = {'seed': seed, 'num_shot': profile['num_shot'], 'ir_root': str(data_root),
                         'entries': [{'index': i, 'query': query,
                                      'refs': [query.rsplit('/', 1)[0] + '/S099_R001_hybrid_IR.wav']
                                      * profile['num_shot']}
                                     for i, query in enumerate(queries)]}
            path = root / ('reference_{}.json'.format(seed))
            p.write_manifest(path, reference)
            profile['seeds'][profile['num_shot']][seed] = manifest_hash(reference)
            references[seed] = (path, reference)
        for arm_index, arm in enumerate(profile['arms']):
            for seed in range(42, 47):
                run = root / '{}_{}'.format(arm['role'], seed)
                run.mkdir()
                paths[arm_index].append(str(run))
                ref_path, reference = references[seed]
                grid = profile['run_grids'][arm['role']]
                manifest = dict(schema_version=1, repo=str(root), reviewed_commit='b' * 40,
                    checkpoint=arm['checkpoint'], checkpoint_sha256=arm['sha256'],
                    manifest_path=str(ref_path), manifest_file_sha256=p.sha256_file(ref_path),
                    manifest_hash=manifest_hash(reference), manifest_seed=seed, gl_seed=seed,
                    num_shot=profile['num_shot'], backbone=arm['backbone'], batch_size=16,
                    batch_canonical=True, max_samples=0, tf32=False, conditions='P', yaw_cols=grid,
                    acoustic_cols=grid, e_acoustic_cols=[], n_samples=12, split_count=12,
                    split='unseen', no_tta=True, data_root=str(data_root), mutable_inputs={},
                    data_identity=dict(identity, manifest_path=str(ref_path),
                        manifest_file_sha256=p.sha256_file(ref_path), manifest_hash=manifest_hash(reference)),
                    evaluator_closure=frozen, source_closures={'entrypoint': entry, 'writer': writer})
                digest = p.write_manifest(run / 'eval_manifest.json', manifest)
                keys = ('backbone', 'checkpoint', 'manifest_hash', 'gl_seed', 'batch_size',
                        'max_samples', 'tf32', 'manifest_path', 'manifest_seed', 'yaw_cols',
                        'acoustic_cols', 'e_acoustic_cols', 'batch_canonical', 'n_samples',
                        'conditions', 'reviewed_commit')
                meta = {key: manifest[key] for key in keys}
                meta.update(eval_manifest_sha256=digest, evaluator_closure_sha256=frozen['sha256'],
                            elapsed_min=0.1, torch_version='fixture')
                cells = {str(k): {metric: [1 + i / 16 + (seed - 42) / 100 for i in range(12)]
                                  for metric in ('edt', 'c50', 't60', 'loss', 'log_mse')}
                         for k in grid}
                sample = dict(meta=meta, query=queries, index=list(range(12)), P=cells,
                              delay_flips={str(k): 0 for k in grid}, decomposition=None)
                metrics = dict(meta=meta, P=_summaries(cells),
                               delay_flips=sample['delay_flips'], decomposition=None)
                p.write_manifest(run / 'per_sample_yaw.json', sample)
                p.write_manifest(run / 'metrics_yaw.json', metrics)
                log = root / '{}_{}.log'.format(arm['role'], seed)
                log.write_text('fixture child completed\n')
                p.write_completion(run / 'completion.json', dict(schema_version=1,
                    eval_manifest_sha256=digest, directory_listing=['eval_manifest.json',
                        'metrics_yaw.json', 'per_sample_yaw.json'], child_exit_status=0,
                    started_at='2026-09-12T00:00:00+00:00', ended_at='2026-09-12T00:00:01+00:00',
                    log={'path': str(log), 'sha256': p.sha256_file(log)},
                    outputs={item: p.sha256_file(run / item)
                             for item in ('per_sample_yaw.json', 'metrics_yaw.json')}))
        monkeypatch.setattr(pc, 'get_profile', lambda _: profile)
        monkeypatch.setattr(pc, 'producer_identity', lambda: producer)
        output, summary = root / 'result.json', root / 'summary.txt'
        argv = ['--profile', name, '--runs-a'] + paths[0]
        if len(profile['arms']) == 2:
            argv += ['--runs-b'] + paths[1]
        argv += ['--json', str(output), '--summary', str(summary)]
        return SimpleNamespace(root=root, profile=profile, paths=paths, argv=argv, output=output,
                               summary=summary, approved=approved, sidecar=Path(str(output) + '.provenance.json'))
    return build


def _assert_no_outputs(fixture):
    assert not fixture.output.exists()
    assert not fixture.summary.exists()
    assert not fixture.sidecar.exists()


def _all_strings(value):
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(*[_all_strings(item) for item in value.values()]) if value else set()
    if isinstance(value, list):
        return set().union(*[_all_strings(item) for item in value]) if value else set()
    return set()


def test_admission_happy_path(admission_fixture):
    fixture = admission_fixture()
    pc.main(fixture.argv)
    payload = _read(fixture.output)
    assert payload['exploratory'] is False
    assert 'non-inferior' in _all_strings(payload)
    assert fixture.summary.is_file() and fixture.sidecar.is_file()
    sidecar = _read(fixture.sidecar)
    strings = _all_strings(sidecar)
    assert p.sha256_file(fixture.output) in strings
    assert p.sha256_file(fixture.summary) in strings
    assert _canonical_digest(fixture.profile) in strings
    assert fixture.approved['closures']['producer_paired_compare'] in strings
    for arm in fixture.paths:
        for run in arm:
            for filename in ('eval_manifest.json', 'per_sample_yaw.json', 'metrics_yaw.json',
                             'completion.json'):
                assert p.sha256_file(Path(run) / filename) in strings
    before = {path: path.read_bytes() for path in (fixture.output, fixture.summary, fixture.sidecar)}
    with pytest.raises((ValueError, FileExistsError)):
        pc.main(fixture.argv)
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize('kind', ['missing_seed', 'wrong_seed', 'queries', 'num_shot',
    'grid', 'truncated', 'manifest_sidecar', 'checkpoint', 'missing_completion',
    'stale_output', 'missing_metrics_cell', 'reconciliation', 'forged_closure'])
def test_admission_refuses_invalid_run(admission_fixture, kind):
    fixture = admission_fixture()
    run = Path(fixture.paths[0][0])
    manifest = _read(run / 'eval_manifest.json')
    sample = _read(run / 'per_sample_yaw.json')
    metrics = _read(run / 'metrics_yaw.json')
    match = None
    if kind == 'missing_seed':
        fixture.argv.remove(str(run))
        match = 'seed|five|5'
    elif kind == 'wrong_seed':
        manifest['manifest_seed'] = 47
        sample['meta']['manifest_seed'] = metrics['meta']['manifest_seed'] = 47
        match = 'seed'
    elif kind == 'queries':
        sample['query'][0], sample['query'][1] = sample['query'][1], sample['query'][0]
        match = 'quer|order'
    elif kind == 'num_shot':
        manifest['num_shot'] = 1
        match = 'num_shot|shot| K'
    elif kind == 'grid':
        manifest['yaw_cols'] = [0, 32]
        sample['meta']['yaw_cols'] = metrics['meta']['yaw_cols'] = [0, 32]
        match = 'grid|yaw_cols|cols'
    elif kind == 'truncated':
        sample['P']['0']['edt'].pop()
        metrics['P'] = _summaries(sample['P'])
        match = 'length|count|shape|samples|queries'
    elif kind == 'checkpoint':
        manifest['checkpoint_sha256'] = 'f' * 64
        match = 'checkpoint'
    elif kind == 'missing_metrics_cell':
        del metrics['P']['0']['edt']
        match = 'metric|edt|cell'
    elif kind == 'reconciliation':
        metrics['P']['0']['edt']['mean'] += 1
        match = 'mean|reconcil'
    elif kind == 'forged_closure':
        manifest['source_closures']['entrypoint']['sha256'] = 'f' * 64
        match = 'closure|evaluator'
    _rebind(run, manifest, sample, metrics)
    if kind == 'missing_completion':
        (run / 'completion.json').unlink()
        match = 'completion'
    elif kind == 'manifest_sidecar':
        manifest['gl_seed'] = 99
        _replace(run / 'eval_manifest.json', manifest)
        match = 'manifest|digest'
    elif kind == 'stale_output':
        path = run / 'per_sample_yaw.json'
        path.write_text(path.read_text() + ' ')
        match = 'per_sample|digest|output'
    with pytest.raises(ValueError, match=match):
        pc.main(fixture.argv)
    _assert_no_outputs(fixture)


def test_tost_refuses_runs_b(admission_fixture):
    fixture = admission_fixture('TOST_K8')
    with pytest.raises((ValueError, SystemExit)):
        pc.main(fixture.argv + ['--runs-b', fixture.paths[0][0]])
    _assert_no_outputs(fixture)


def test_profile_cli_refuses_analytic_overrides(admission_fixture):
    fixture = admission_fixture()
    with pytest.raises(SystemExit):
        pc.main(fixture.argv + ['--n-boot', '10'])
    _assert_no_outputs(fixture)


@pytest.mark.parametrize('unapproved', ['evaluator', 'writer', 'producer_paired_compare', 'checkpoint'])
def test_production_refuses_unapproved_identity(admission_fixture, unapproved):
    fixture = admission_fixture()
    if unapproved == 'checkpoint':
        fixture.approved['checkpoints']['aug']['sha256'] = None
    else:
        fixture.approved['closures'][unapproved] = None
    with pytest.raises(ValueError, match='approved|None|digest|sha256'):
        pc.main(fixture.argv)
    _assert_no_outputs(fixture)


def test_exploratory_records_deviation_and_omits_all_verdicts(admission_fixture):
    fixture = admission_fixture()
    run = Path(fixture.paths[0][0])
    manifest = _read(run / 'eval_manifest.json')
    manifest['batch_size'] = 8
    sample = _read(run / 'per_sample_yaw.json')
    metrics = _read(run / 'metrics_yaw.json')
    sample['meta']['batch_size'] = metrics['meta']['batch_size'] = 8
    _rebind(run, manifest=manifest, sample=sample, metrics=metrics)
    pc.main(fixture.argv + ['--exploratory'])
    payload = _read(fixture.output)
    assert payload['exploratory'] is True
    assert 'batch_size' in fixture.output.read_text()
    assert 'verdict' not in fixture.output.read_text().lower()
    assert 'non-inferior' not in fixture.summary.read_text()
    assert 'exploratory' in fixture.summary.read_text().lower()
    assert _read(fixture.sidecar)['exploratory'] is True


@pytest.mark.parametrize('name,count,word', [('H1_K1', 3, 'non-inferior'),
    ('H2_K8', 12, 'not supported'), ('TOST_K8', 21, 'equivalent')])
def test_other_profiles_use_block_baselines_and_families(admission_fixture, name, count, word):
    fixture = admission_fixture(name)
    result = pc.main(fixture.argv)
    assert len(result['cells']) == count
    assert word in _all_strings(result)
    for cell in result['cells']:
        assert cell['estimate'] == 0
        assert cell['alpha_local'] == .05 / fixture.profile['family']
        assert len(cell['mask']) == 12
        assert cell['room_cluster_interval'] == [0, 0]


@pytest.mark.parametrize('digest', [None, False, '', 'f' * 64])
def test_invalid_completion_output_digest_never_reaches_statistics(admission_fixture, monkeypatch, digest):
    fixture = admission_fixture()
    run = Path(fixture.paths[0][0])
    completion = _read(run / 'completion.json')
    completion['outputs']['per_sample_yaw.json'] = digest
    p.write_completion(run / 'completion.json', completion)
    monkeypatch.setattr(pc, 'analyze', lambda *a: pytest.fail('statistics before admission'))
    with pytest.raises(ValueError, match='digest'):
        pc.main(fixture.argv)
    _assert_no_outputs(fixture)


@pytest.mark.parametrize('target', [.05, .025])
def test_h1_convergence_gate_includes_superiority(admission_fixture, monkeypatch, target):
    fixture = admission_fixture()
    original = pc.two_sided_interval
    calls = []
    def unstable(samples, alpha):
        calls.append(alpha)
        # compute(seed0), cluster, then convergence evaluates seed0 and seed1.
        if alpha == target and calls.count(alpha) % (4 if target == .05 else 3) == 0:
            return (0, 100)
        return original(samples, alpha)
    monkeypatch.setattr(pc, 'two_sided_interval', unstable)
    with pytest.raises(ValueError, match='convergence'):
        pc.main(fixture.argv)
    _assert_no_outputs(fixture)


def test_input_mutation_during_analysis_refuses_all_outputs(admission_fixture, monkeypatch):
    fixture = admission_fixture()
    original = pc.analyze
    def changed(*args):
        result = original(*args)
        Path(fixture.paths[0][0], 'per_sample_yaw.json').write_text('{}')
        return result
    monkeypatch.setattr(pc, 'analyze', changed)
    with pytest.raises(ValueError, match='input changed'):
        pc.main(fixture.argv)
    _assert_no_outputs(fixture)


def test_nonfinite_aggregate_mean_is_not_a_reconciled_metric(admission_fixture):
    fixture = admission_fixture()
    run = Path(fixture.paths[0][0])
    metrics = _read(run / 'metrics_yaw.json')
    metrics['P']['0']['edt']['mean'] = float('nan')
    (run / 'metrics_yaw.json').write_text(json.dumps(metrics))
    completion = _read(run / 'completion.json')
    completion['outputs']['metrics_yaw.json'] = p.sha256_file(run / 'metrics_yaw.json')
    p.write_completion(run / 'completion.json', completion)
    with pytest.raises(ValueError, match='metric|reconcil|mean'):
        pc.main(fixture.argv)
    _assert_no_outputs(fixture)


def test_named_seed_means_and_descriptive_cells(admission_fixture):
    fixture = admission_fixture()
    result = pc.main(fixture.argv)
    for cell in result['cells']:
        assert cell['seed_labels'] == list(range(42, 47))
        assert set(cell['seed_means']) == {'aug', 'control'}
        assert cell['seed_means']['aug']['0']['sd'] == pytest.approx(np.std(np.arange(5) / 100, ddof=1))
        if cell['metric'] == 'T60':
            assert not cell['decision_driving'] and 'decision_bound' not in cell


@pytest.mark.parametrize('role', ['evaluator', 'writer', 'producer_paired_compare'])
def test_run_claim_cannot_replace_an_approved_closure_pin(admission_fixture, role):
    fixture = admission_fixture()
    fixture.approved['closures'][role] = 'f' * 64
    with pytest.raises(ValueError, match=role + ' approved closure'):
        pc.main(fixture.argv)
    _assert_no_outputs(fixture)


def test_existing_summary_never_leaves_a_partial_json(admission_fixture):
    fixture = admission_fixture()
    fixture.summary.write_text('preserve me')
    with pytest.raises(FileExistsError):
        pc.main(fixture.argv)
    assert fixture.summary.read_text() == 'preserve me'
    assert not fixture.output.exists() and not fixture.sidecar.exists()


def test_public_group_admission_and_null_exploratory(admission_fixture):
    fixture = admission_fixture()
    groups = [(arm, 8, paths) for arm, paths in zip(fixture.profile['arms'], fixture.paths)]
    approved, receipt = pc.load_approved_digests()
    run = pc.admit_run(groups[0][2][0], groups[0][0], 8, fixture.profile, approved)
    assert run['meta']['manifest_seed'] == 42
    admitted = pc.admit_runs(fixture.profile, groups)
    assert admitted['approved_digests']['sha256'] == receipt['sha256']
    fixture.approved['closures']['producer_paired_compare'] = None
    fixture.approved['checkpoints']['aug']['sha256'] = None
    result = pc.main(fixture.argv + ['--exploratory'])
    assert any('producer_paired_compare' in item for item in result['deviations'])
    assert any('aug checkpoint' in item for item in result['deviations'])
    assert 'verdict' not in result
    assert _read(fixture.sidecar)['approved_digests']['git_blob'] == receipt['git_blob']
