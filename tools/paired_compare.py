"""Pure statistics and fail-closed, profile-driven paired inference for exp_04."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from tools import provenance
from tools.exp04_profiles import get_profile, json_value, load_approved_digests
from tools.reference_manifest import load_manifest, manifest_hash
from tools.summarize_yaw import load_run, rooms_from_paths, signed_degrees, _check_metrics_reconciliation

from decimal import Decimal, ROUND_CEILING
import numbers

import numpy as np

from tools.paired_stats import (
    _bootstrap_ratios, relative_degradation_bootstrap, equivalence_tost,
    diff_in_diff_bootstrap, bonferroni_alpha, valid_mask,
)


def _quantile(samples, probability, upper=False):
    """Inverse empirical CDF: rank ceil(p*n); upper=True accepts the upper tail."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("quantile samples must be a nonempty finite 1-D array")
    # Decimal ranks avoid a binary rounding step promoting rank 3 to 4
    # for alpha=.7 and n=10. The input decimal is interpreted literally.
    probability = Decimal(str(probability))
    if not probability.is_finite() or not 0 < probability < 1:
        raise ValueError("quantile probability must be in (0, 1)")
    probability = Decimal(1) - probability if upper else probability
    rank = int((probability * values.size).to_integral_value(rounding=ROUND_CEILING)) - 1
    return float(np.partition(values, rank)[rank])


def one_sided_upper(samples, alpha):
    """Smallest v with >= ceil((1-alpha)*n) samples <= v; no interpolation.

    For [1, 2, 3, 4] and alpha=.25 the upper bound is 3. A bound equal
    to a practical margin fails the strict confirmatory decision rule.
    """
    return _quantile(samples, alpha, upper=True)


def two_sided_interval(samples, alpha):
    """Exact [Q_(alpha/2), Q_(1-alpha/2)] order-statistic interval."""
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    tail = Decimal(str(alpha)) / 2
    return (_quantile(samples, tail), _quantile(samples, tail, upper=True))


def _seed_values(values):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != 5:
        raise ValueError("expected five seed arrays with shape (5, n)")
    return matrix


def five_seed_mean(per_seed_arrays):
    """Plain five-seed mean; any non-finite seed makes the query NaN."""
    values = _seed_values(per_seed_arrays)
    with np.errstate(invalid="ignore", over="ignore"):
        means = values.mean(axis=0)
    means[~np.isfinite(values).all(axis=0)] = np.nan
    return means


def _validity_counts(baseline, paired):
    return {"n": int(baseline.size),
            "baseline_invalid": int((~baseline).sum()),
            "newly_invalid": int((baseline & ~paired).sum()),
            "excluded": int((~paired).sum()), "valid": int(paired.sum())}


def cell_mask(e0_a, ek_a=None, e0_b=None, ek_b=None, seeds=(42, 43, 44, 45, 46)):
    """H1: baseline both arms; H2: baseline/angle both; TOST: aug only.

    Per-seed exclusions condition on that seed's baseline. Arm totals
    condition on all five baseline seeds; joint totals on both arms.
    """
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("exactly five distinct seed labels are required")
    if (ek_b is not None and e0_b is None) or (
            e0_b is not None and (ek_a is None) != (ek_b is None)):
        raise ValueError("both arms require corresponding baseline/angle arrays")
    if e0_b is None and ek_a is None:
        raise ValueError("a cell requires both arms or a baseline/angle pair")
    baseline_masks, paired_masks, counts = [], [], {}
    shape = _seed_values(e0_a).shape
    arms = [("a", e0_a, ek_a)]
    if e0_b is not None:
        arms.append(("b", e0_b, ek_b))
    for arm, base, angle in arms:
        base = _seed_values(base)
        angle = base if angle is None else _seed_values(angle)
        if base.shape != shape or angle.shape != shape:
            raise ValueError("all seed arrays must have the same query shape")
        base_valid = np.isfinite(base)
        paired = np.stack([valid_mask(x, y) for x, y in zip(base, angle)])
        baseline_masks.append(base_valid.all(axis=0))
        paired_masks.append(paired.all(axis=0))
        counts[arm] = {"total": _validity_counts(baseline_masks[-1], paired_masks[-1]),
                       "seeds": {str(seed): _validity_counts(b, p) for seed, b, p in
                                 zip(seeds, base_valid, paired)}}
    baseline, mask = np.logical_and.reduce(baseline_masks), np.logical_and.reduce(paired_masks)
    counts["joint"] = _validity_counts(baseline, mask)
    if not mask.any():
        raise ValueError("empty cell after all-seed finite mask")
    return mask, counts


def _samples(pairs, n_boot, seed, clusters):
    if isinstance(n_boot, bool) or not isinstance(n_boot, numbers.Integral):
        raise TypeError("n_boot must be an integer")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    pairs = [(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
             for a, b in pairs]
    return _bootstrap_ratios(pairs, int(n_boot), seed, clusters)


def _sample_metadata(result, samples, n_boot, seed):
    for key in ("lo", "hi", "alpha"):
        result.pop(key, None)
    result.update(samples=samples, n_boot=int(n_boot), seed=seed)
    return result


def rho_bootstrap(err_a, err_b, n_boot=20000, seed=0, clusters=None):
    """rho=(mean(a)-mean(b))/mean(b) on common query/room resamples.

    The imported public primitive validates inputs and supplies the estimate
    using one validation draw. Actual draws use its shared sampler, avoiding
    its interpolated quantiles and a second full bootstrap computation.
    """
    result = relative_degradation_bootstrap(err_b, err_a, n_boot=1, seed=seed,
                                            clusters=clusters)
    samples = _samples([(err_b, err_a)], n_boot, seed, clusters)[0]
    result["rho"] = result.pop("r")
    return _sample_metadata(result, samples, n_boot, seed)


def dk_bootstrap(e0_a, ek_a, e0_b, ek_b, n_boot=20000, seed=0, clusters=None):
    """D_k=r_k(a)-r_k(b); exp_03's definition matches §3 exactly.

    Both arm ratios share the same resample. As for rho, the public primitive
    supplies validation/estimates; exact quantiles are computed by the caller.
    """
    result = diff_in_diff_bootstrap(e0_a, ek_a, e0_b, ek_b, n_boot=1,
                                    seed=seed, clusters=clusters)
    a, b = _samples([(e0_a, ek_a), (e0_b, ek_b)], n_boot, seed, clusters)
    return _sample_metadata(result, a - b, n_boot, seed)


def tost_cell(e0, ek, margin, alpha_local, n_boot=20000, seed=0, clusters=None):
    """One-arm TOST using exact [Q_alpha_local, Q_(1-alpha_local)]."""
    result = equivalence_tost(e0, ek, margin, n_boot=1, alpha=alpha_local,
                              seed=seed, clusters=clusters)
    samples = _samples([(e0, ek)], n_boot, seed, clusters)[0]
    lo, hi = _quantile(samples, alpha_local), _quantile(samples, alpha_local, upper=True)
    result.update(lo=lo, hi=hi, n_boot=int(n_boot), samples=samples, seed=seed,
                  equivalent=bool(-margin < lo and hi < margin))
    return result


def convergence(fn, seed_a=0, seed_b=1, tol=0.10):
    """Maximum endpoint movement / seed-a width on the specified companion.

    The caller supplies the profile's decision companion (and H1 superiority
    companion separately). Identical zero-width intervals pass; changed ones
    fail. Non-finite or reversed intervals fail closed.
    """
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("convergence tolerance must be finite and nonnegative")
    intervals = []
    for seed in (seed_a, seed_b):
        value = fn(seed)
        if isinstance(value, dict):
            value = (value["lo"], value["hi"])
        interval = np.asarray(value, dtype=float)
        if interval.shape != (2,):
            raise ValueError("convergence requires two interval endpoints")
        intervals.append(interval)
    a, b = intervals
    finite = np.isfinite(intervals).all() and a[0] <= a[1] and b[0] <= b[1]
    width = float(a[1] - a[0]) if finite else float("nan")
    movement = float(np.max(np.abs(a - b))) if finite else float("inf")
    ratio = (movement / width if width > 0 else
             0.0 if finite and movement == 0 else float("inf"))
    return {"passed": bool(finite and ratio <= tol), "movement": movement,
            "width": width, "ratio": ratio, "tolerance": tol,
            "seed_a": {"seed": seed_a, "lo": float(a[0]), "hi": float(a[1])},
            "seed_b": {"seed": seed_b, "lo": float(b[0]), "hi": float(b[1])}}


def h1_verdict(edt_upper, c50_upper, margin=0.03):
    edt = np.isfinite(edt_upper) and edt_upper < margin
    c50 = np.isfinite(c50_upper) and c50_upper < margin
    if edt and c50:
        return "non-inferior"
    if edt or c50:
        return "non-inferior on {} only".format("EDT" if edt else "C50")
    return "not shown"


def superiority(upper):
    """Use the upper endpoint of H1's adjusted two-sided companion."""
    return bool(np.isfinite(upper) and upper < 0)


def h2_verdict(c50_uppers):
    if len(c50_uppers) != 4:
        raise ValueError("H2 requires the four primary C50 cells")
    passes = sum(np.isfinite(bound) and bound < 0 for bound in c50_uppers)
    return "supported" if passes == 4 else "partially supported" if passes else "not supported"


def tost_verdict(lo, hi, margin=0.02):
    equivalent = np.isfinite([lo, hi, margin]).all() and 0 < margin and -margin < lo <= hi < margin
    return "equivalent" if equivalent else "equivalence not established"



REPO = Path(__file__).resolve().parents[1]
_UNBOUND = object()


def _digest(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def producer_identity(entry_module='tools.paired_compare'):
    """Bind imported producer dependencies to committed HEAD and current bytes."""
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    records, digest = provenance.closure_record(
        provenance.source_closure(entry_module, REPO), commit, REPO)
    if any(r['reviewed_blob_sha256'] != r['working_tree_sha256'] or
           r['reviewed_blob_sha256'] is None for r in records):
        raise ValueError('producer closure differs from HEAD')
    return {'sha256': digest, 'files': records, 'commit': commit}


def _equal(actual, expected):
    """JSON comparison keeps bool, int, float, and missing values distinct."""
    return json.dumps(actual, sort_keys=True) == json.dumps(json_value(expected), sort_keys=True)


def _closure_digest(closure):
    records = closure['files']
    names = [r['path'] for r in records]
    if not records or names != sorted(set(names)) or any(
            r['working_tree_sha256'] != r['reviewed_blob_sha256'] or
            r['reviewed_blob_sha256'] is None or r['commits_after_reviewed'] for r in records):
        raise ValueError('closure records are incomplete or unreviewed')
    # Match tools.provenance.closure_record's deliberately noncompact encoding.
    return hashlib.sha256(json.dumps([[r['path'], r['reviewed_blob_sha256']]
        for r in records], sort_keys=True).encode()).hexdigest()


def admit_run(run_dir, arm, num_shot, profile, approved, check=None, inputs=None, data_stats=None):
    """Admit one run. evaluator=exp04_eval; writer=exp04_eval_launch.

    training_launcher is recorded, not compared; its evidence is checkpoint-bound.
    Aug evaluation launches must pass --bind-input train_manifest=<attempt>/train_manifest.json
    and --bind-input train_completion=<attempt>/completion.json.
    approved is the loader's frozen pins mapping, without its identity receipt.
    Optional check/inputs share deviations and byte bindings across a grouped analysis.
    """
    if check is None:
        def check(ok, name):
            if not ok:
                raise ValueError(name)
    inputs = {} if inputs is None else inputs
    data_stats = {} if data_stats is None else data_stats
    directory = Path(run_dir).resolve()
    label = str(directory)
    def require(ok, name):
        check(ok, label + ': ' + name)
    def bind(path, expected=_UNBOUND):
        path = str(Path(path).resolve())
        actual = inputs[path] if path in inputs else provenance.sha256_file(path)
        require(expected is _UNBOUND or actual == expected, 'digest ' + path)
        inputs[path] = actual
        return actual
    files = {name: directory / name for name in
             ('eval_manifest.json', 'completion.json', 'per_sample_yaw.json', 'metrics_yaw.json')}
    require({p.name for p in directory.iterdir()} == set(files), 'run directory contents')
    payloads = {name: json.loads(path.read_text()) for name, path in files.items()}
    fields, completion = (payloads[name] for name in ('eval_manifest.json', 'completion.json'))
    digest = bind(files['eval_manifest.json'], completion.get('eval_manifest_sha256'))
    require(completion.get('eval_manifest_sha256') == digest, 'completion manifest digest')
    bind(files['completion.json'])
    for name in ('per_sample_yaw.json', 'metrics_yaw.json'):
        require(name in completion['outputs'], 'completion output ' + name)
        bind(files[name], completion['outputs'][name])
    require(_equal(completion.get('schema_version'), 1) and
            type(completion.get('child_exit_status')) is int and completion['child_exit_status'] == 0,
            'completion status/schema')
    require(completion.get('directory_listing') == sorted(n for n in files if n != 'completion.json'),
            'completion directory_listing')
    bind(completion['log']['path'], completion['log']['sha256'])
    require(_equal(fields.get('schema_version'), 1), 'eval manifest schema')
    flags = {'confirmatory': True, 'allow_dirty_used': False}
    for key, value in flags.items():
        require(_equal(fields.get(key), value) and _equal(completion.get(key), value), key)
    require(set(fields['mutable_inputs']) <=
            {'control_args', 'train_manifest', 'train_completion', 'probe_receipt'}, 'mutable_inputs names')
    if arm['role'] == 'aug':
        require({'train_manifest', 'train_completion'} <= set(fields['mutable_inputs']), 'aug train provenance binding')
    root = Path(fields['repo'])
    declarations = dict(fields, eval_manifest={'path': str(files['eval_manifest.json']), 'sha256': digest})
    # The shared data cache below validates inventory bytes and stat identity once.
    declarations.pop('data_identity')
    for failure in provenance.revalidate(declarations, required=(
            'repo', 'checkpoint', 'checkpoint_sha256', 'manifest_path', 'manifest_file_sha256',
            'evaluator_closure', 'source_closures', 'mutable_inputs', 'eval_manifest')):
        require(False, 'revalidate ' + failure)
    closures = dict(fields['source_closures'], frozen_evaluator=fields['evaluator_closure'])
    for name, closure in closures.items():
        require(_closure_digest(closure) == closure['sha256'], 'closure digest ' + name)
        for record in closure['files']:
            bind(root / record['path'], record['working_tree_sha256'])
    for pin, recorded in (('evaluator', 'entrypoint'), ('writer', 'writer')):
        require(closures[recorded]['sha256'] == approved['closures'][pin], pin + ' approved closure')
    reference = load_manifest(root / fields['manifest_path'])
    seed, shot = fields['manifest_seed'], num_shot
    checkpoint = approved['checkpoints']['aug'] if arm['role'] == 'aug' else arm
    if profile.get('checkpoint_key') == 'aug_epoch9':
        checkpoint = dict(path=arm['checkpoint'], epoch=arm['epoch'], sha256=approved['checkpoints']['aug_epoch9'])
    if arm['role'] == 'aug':
        require(checkpoint['path'] == arm['checkpoint'] and checkpoint['epoch'] == arm['epoch'],
                'aug checkpoint path/epoch approval')
    require(type(seed) is int, 'seed type')
    expected = {'num_shot': shot, 'gl_seed': seed, 'split': profile['dataset']['split'],
        'split_count': profile['dataset']['n_queries'], 'n_samples': profile['dataset']['n_queries'],
        'conditions': profile['condition'], 'batch_size': profile['batch_size'],
        'max_samples': profile['max_samples'], 'tf32': profile['tf32'], 'batch_canonical': True,
        'no_tta': True, 'backbone': arm['backbone'], 'checkpoint_sha256': checkpoint['sha256'],
        'manifest_hash': profile['seeds'][shot].get(seed), 'e_acoustic_cols': []}
    for key, value in expected.items():
        require(value is not None and _equal(fields.get(key), value), key)
    require((root / fields['checkpoint']).resolve() == (REPO / arm['checkpoint']).resolve(), 'checkpoint path/epoch')
    require(reference['seed'] == seed and reference['num_shot'] == shot, 'reference seed/num_shot')
    require(manifest_hash(reference) == fields['manifest_hash'], 'reference semantic hash')
    for path_key, hash_key in (('checkpoint', 'checkpoint_sha256'), ('manifest_path', 'manifest_file_sha256')):
        bind(root / fields[path_key], fields[hash_key])
    identity = fields['data_identity']
    require(identity['inventory_sha256'] == profile['dataset']['inventory_sha256'], 'dataset inventory')
    require(identity.get('manifest_hash') == fields['manifest_hash'] and
            identity.get('manifest_file_sha256') == fields['manifest_file_sha256'], 'dataset manifest identity')
    require(provenance._inventory_digest(identity['inventory']) == identity['inventory_sha256'],
            'data inventory digest')
    if 'manifest_path' in identity:
        bind(identity['manifest_path'], identity['manifest_file_sha256'])
    for record in identity['inventory']:
        path = str((Path(identity['data_root']) / record['path']).resolve())
        before = _file_stamp(path)
        bind(path, record['sha256'])
        require(before == _file_stamp(path) and data_stats.get(path, before) == before, 'data changed ' + path)
        data_stats[path] = before
    for record in fields['mutable_inputs'].values():
        bind(root / record['path'], record['sha256'])
    run = load_run(str(directory))
    aggregate = payloads['metrics_yaw.json']
    require(_equal(run['meta'], aggregate['meta']), 'output meta agreement')
    for key in ('backbone', 'checkpoint', 'manifest_hash', 'gl_seed', 'batch_size', 'max_samples',
                'tf32', 'manifest_path', 'manifest_seed', 'yaw_cols', 'acoustic_cols',
                'e_acoustic_cols', 'batch_canonical', 'n_samples', 'conditions', 'reviewed_commit'):
        require(key in fields and _equal(run['meta'].get(key), fields[key]), 'meta ' + key)
    require(run['meta'].get('eval_manifest_sha256') == digest, 'meta manifest digest')
    require(run['meta'].get('evaluator_closure_sha256') == closures['frozen_evaluator']['sha256'],
            'meta frozen evaluator closure')
    queries, indices = run['query'], run['index']
    require(queries == [e['query'] for e in reference['entries']] and
            indices == [e['index'] for e in reference['entries']], 'query/index reference order')
    require(len(queries) == profile['dataset']['n_queries'] and len(set(queries)) == len(queries), 'query count')
    require(_digest(queries) == profile['dataset']['query_sha256'], 'query digest')
    require(indices == list(range(len(queries))) and all(type(i) is int for i in indices), 'index order')
    require(len(set(rooms_from_paths(queries))) == profile['dataset']['n_rooms'], 'room count')
    grid = profile['run_grids'][arm['role']]
    for key in ('yaw_cols', 'acoustic_cols'):
        require(_equal(fields.get(key), profile.get('acoustic_grid', grid) if key == 'acoustic_cols' else grid), 'grid ' + key)
    for payload in (run, aggregate):
        require('E' not in payload, 'condition P only')
        require(set(payload['P']) == {str(k) for k in grid}, 'grid P cells')
    for angle, cell in run['P'].items():
        require(set(cell) == set(aggregate['P'].get(angle, {})), 'metrics cell coverage')
        for name in ('edt', 'c50', 't60'):
            require((name in cell) == (int(angle) in profile.get('acoustic_grid', grid)), 'metric coverage ' + name)
        for metric, values in cell.items():
            require(isinstance(values, list) and len(values) == len(queries), 'truncated metric length ' + metric)
            require(all(v is None or type(v) in (float, int) for v in values), 'metric type ' + metric)
            summary = aggregate['P'][angle].get(metric, {})
            mean = summary.get('mean')
            require('mean' in summary and (mean is None or
                type(mean) in (float, int) and math.isfinite(mean)), 'metric mean ' + metric)
            require(all(type(summary.get(key)) is int for key in ('n_valid', 'n_nan')),
                    'metric validity counts ' + metric)
    for failure in _check_metrics_reconciliation(label, run):
        require(False, 'reconciliation ' + failure)
    run['admission_flags'] = {key: fields[key] for key in flags}
    return run


def admit_runs(profile, groups, approved=None, exploratory=False, producer=None,
               producer_key='producer_paired_compare'):
    """Admit (arm, num_shot, directories) groups before computing statistics.

    approved is the loader's (pins, identity) pair; only this producer's pin is compared.
    """
    approved, receipt = load_approved_digests() if approved is None else approved
    required = [(key, approved['closures'][key]) for key in ('evaluator', 'writer', producer_key)]
    required += [(a['role'] + ' checkpoint', approved['checkpoints']['aug']['sha256']
                  if a['role'] == 'aug' else a['sha256']) for a, _, _ in groups]
    if any(arm['role'] == 'aug' for arm, _, _ in groups):
        required += [('aug checkpoint ' + key, approved['checkpoints']['aug'][key])
                     for key in ('path', 'epoch')]
    if profile.get('checkpoint_key') == 'aug_epoch9':
        required += [('aug_epoch9 checkpoint', approved['checkpoints']['aug_epoch9'])]
    required += [('dataset inventory', profile['dataset']['inventory_sha256']),
                 ('approval schema_version', approved['schema_version'])]
    deviations, inputs, run_flags, data_stats = [], {}, {}, {}
    def check(ok, name):
        if not ok:
            deviations.append(name)
    for key, value in required:
        check(value is not None, 'profile not yet approved: ' + key)
    producer = producer_identity() if producer is None else producer
    check(producer['sha256'] == approved['closures'][producer_key], producer_key + ' approved closure')
    inputs[receipt['path']] = receipt['sha256']
    for record in producer['files']:
        inputs[str((REPO / record['path']).resolve())] = record['working_tree_sha256']
    admitted_groups = []
    for arm, shot, directories in groups:
        runs = []
        for directory in directories:
            try:
                runs.append(admit_run(directory, arm, shot, profile, approved, check, inputs, data_stats))
                run_flags[str(Path(directory).resolve())] = runs[-1]['admission_flags']
            except (KeyError, TypeError, ValueError, OSError, IndexError) as error:
                check(False, '{}: admission {}'.format(directory, error))
        seeds = [r['meta'].get('manifest_seed') for r in runs]
        check(len(runs) == len(profile['seeds'][shot]) and all(type(s) is int for s in seeds) and
              set(seeds) == set(profile['seeds'][shot]), arm['role'] + ' seed set')
        admitted_groups.append(sorted(runs, key=lambda r: int(r['meta']['manifest_seed'])))
    flat = [r for group in admitted_groups for r in group]
    if flat:
        check(all(r['query'] == flat[0]['query'] and r['index'] == flat[0]['index'] for r in flat),
              'query/index order across runs')
    if deviations and not exploratory:
        raise ValueError('admission failed: ' + '; '.join(deviations))
    return {'groups': admitted_groups, 'approved_digests': dict(receipt, pins=json_value(approved)),
            'deviations': deviations, 'inputs': inputs, 'producer': producer,
            'run_flags': run_flags, 'data_stats': data_stats}


def _seed_summary(values, mask):
    means = values[:, mask].mean(axis=1)
    return {'per_seed': means.tolist(), 'mean': float(means.mean()),
            'sd': float(means.std(ddof=1))}


def _analyze_cell(profile, groups, metric, k):
    h1 = profile['input_selection'] == 'standalone_k0'
    one_arm = profile['mode'] == 'one_arm'
    arrays = [np.asarray([r['P'][str(angle)][metric.lower()] for r in group], dtype=float)
              for group in groups for angle in ((0,) if h1 else (0, k))]
    labels = tuple(sorted(profile['seeds'][profile['num_shot']]))
    mask, exclusions = (cell_mask(arrays[0], e0_b=arrays[1], seeds=labels) if h1 else
                       cell_mask(*arrays, seeds=labels))
    roles = dict(zip(('a', 'b'), (arm['role'] for arm in profile['arms'])))
    exclusions = {roles.get(key, key): value for key, value in exclusions.items()}
    means = [five_seed_mean(array)[mask] for array in arrays]
    rooms = rooms_from_paths(groups[0][0]['query'])[mask]
    local = bonferroni_alpha(profile['family'], profile['alpha'])
    def compute(seed, clusters=None):
        kwargs = dict(n_boot=profile['n_boot'], seed=seed, clusters=clusters)
        if h1:
            return rho_bootstrap(*means, **kwargs)
        if one_arm:
            return tost_cell(*means, margin=profile['margin'], alpha_local=local, **kwargs)
        return dk_bootstrap(*means, **kwargs)
    seed0, seed1 = profile['bootstrap_seeds']
    result = compute(seed0)
    draws = {seed0: result.pop('samples')}
    companion = two_sided_interval(draws[seed0], profile['companion_alpha'])
    primary = metric in profile['metrics']['primary']
    decision = primary or metric in profile['metrics']['supportive']
    stride = 1 if h1 else 2
    seed_means = {arm['role']: {str(angle): _seed_summary(arrays[i * stride + j], mask)
                  for j, angle in enumerate((0,) if h1 else (0, k))}
                  for i, arm in enumerate(profile['arms'])}
    cell = {'metric': metric, 'k': k, 'degrees': signed_degrees(k), 'primary': primary,
            'decision_driving': decision, 'mask': mask.tolist(), 'exclusions': exclusions,
            'seed_labels': list(labels), 'seed_means': seed_means,
            'estimate': result['rho' if h1 else 'r' if one_arm else 'd'],
            'alpha_local': local, 'companion_interval': list(companion)}
    cluster = compute(seed0, rooms)
    cell['room_cluster_interval'] = list(two_sided_interval(cluster['samples'], profile['companion_alpha']))
    cell['n_rooms_retained'] = len(set(rooms))
    cell['convergence'] = {}
    if decision:
        cell['decision_bound'] = one_sided_upper(draws[seed0], local)
        draws[seed1] = compute(seed1)['samples']
        def gate(alpha):
            return convergence(lambda seed: two_sided_interval(draws[seed], alpha),
                               seed0, seed1, profile['convergence_tolerance'])
        cell['convergence']['decision'] = gate(profile['companion_alpha'])
        if h1:
            cell['superiority_interval'] = list(two_sided_interval(draws[seed0], profile['superiority_alpha']))
            cell['convergence']['superiority'] = gate(profile['superiority_alpha'])
    if not h1 and not one_arm:
        cell.update(r_a=result['r_a'], r_b=result['r_b'])
    return cell


def analyze(profile, admitted, exploratory=False):
    """Use profile-only analytic settings; every canonical decision must converge."""
    groups = admitted['groups']
    result = {'schema_version': 1, 'exploratory': exploratory, 'profile': json_value(profile),
              'profile_digest': _digest(profile), 'inputs': admitted['inputs'],
              'run_flags': admitted['run_flags'],
              'deviations': list(admitted['deviations']), 'cells': []}
    flat = [r for group in groups for r in group]
    paired = all(len(group) == 5 for group in groups) and flat and all(
        r['query'] == flat[0]['query'] and r['index'] == flat[0]['index'] for r in flat)
    if not paired:
        if not exploratory:
            raise ValueError('five-seed pairing unavailable')
        result['deviations'].append('analysis unavailable: five-seed query pairing')
        return result
    metrics = sum((tuple(profile['metrics'][kind]) for kind in ('primary', 'supportive', 'descriptive')), ())
    for metric in metrics:
        for k in profile['grid']:
            try:
                result['cells'].append(_analyze_cell(profile, groups, metric, k))
            except (KeyError, ValueError, TypeError) as error:
                if not exploratory:
                    raise ValueError('{} k={}: {}'.format(metric, k, error)) from error
                result['deviations'].append('{} k={}: {}'.format(metric, k, error))
    failed = ['{} k={} {}'.format(cell['metric'], cell['k'], name)
              for cell in result['cells'] for name, gate in cell['convergence'].items()
              if not gate['passed']]
    if failed:
        if not exploratory:
            raise ValueError('convergence gate failed: ' + ', '.join(failed))
        result['deviations'].append('convergence gate failed: ' + ', '.join(failed))
    if not exploratory:
        cells = [cell for cell in result['cells'] if cell['decision_driving']]
        if profile['input_selection'] == 'standalone_k0':
            bounds = {c['metric']: c['decision_bound'] for c in cells}
            result['verdict'] = h1_verdict(bounds['EDT'], bounds['C50'], profile['margin'])
            for cell in cells:
                cell['superiority'] = superiority(cell['superiority_interval'][1])
        elif profile['mode'] == 'two_arm':
            result['verdict'] = h2_verdict([c['decision_bound'] for c in cells if c['primary']])
        else:
            for cell in cells:
                cell['verdict'] = tost_verdict(*cell['companion_interval'], margin=profile['margin'])
    return result


def _safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(item) for item in value]
    return value


def render_summary(result):
    label = 'EXPLORATORY' if result['exploratory'] else 'CONFIRMATORY'
    lines = [label + ' ' + result['profile_name'], 'Profile sha256: ' + result['profile_digest']]
    for cell in result['cells']:
        text = '{} k={}: estimate={:.8g}, upper={:.8g}, companion={}, rooms={}, n={}'.format(
            cell['metric'], cell['k'], cell['estimate'], cell['companion_interval'][1],
            cell['companion_interval'], cell['room_cluster_interval'], cell['exclusions']['joint']['valid'])
        lines.append(text + ('; ' + cell['verdict'] if 'verdict' in cell else ''))
    if 'verdict' in result:
        lines.append(result['verdict'])
    lines.extend('Deviation: ' + item for item in result['deviations'])
    return '\n'.join(lines) + '\n'


def _file_stamp(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def recheck_inputs(admitted):
    """Rehash mutable artifacts; data bytes are hashed once with stat checks until publish."""
    for path, digest in admitted['inputs'].items():
        expected = admitted['data_stats'].get(path)
        unchanged = (_file_stamp(path) == expected if expected is not None
                     else provenance.sha256_file(path) == digest)
        if not unchanged:
            raise ValueError('input changed during analysis: ' + path)


def write_outputs(result, admitted, json_path, summary_path, renderer=None):
    """Exclusive creation; the last sidecar binds inputs and both completed outputs."""
    paths = [Path(json_path), Path(summary_path), Path(str(json_path) + '.provenance.json')]
    if len({p.resolve() for p in paths}) != 3 or any(os.path.lexists(p) for p in paths):
        raise FileExistsError('output paths must be distinct and absent')
    text = (renderer or render_summary)(result).encode()
    data = (json.dumps(_safe_json(result), sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    sidecar = {'schema_version': 1, 'exploratory': result['exploratory'], 'inputs': admitted['inputs'],
               'profile_digest': result['profile_digest'], 'producer': admitted['producer'],
               'approved_digests': admitted['approved_digests'], 'run_flags': admitted['run_flags'],
               'outputs': {str(paths[0].resolve()): hashlib.sha256(data).hexdigest(),
                           str(paths[1].resolve()): hashlib.sha256(text).hexdigest()}}
    receipt = (json.dumps(sidecar, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    recheck_inputs(admitted)
    created = []
    try:
        for path, payload in zip(paths, (data, text, receipt)):
            fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    provenance.apply_umask(stream.fileno())
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, path)  # Atomic publication, preserving exclusive creation.
                created.append(path)
            finally:
                os.unlink(temporary)
    except BaseException:
        for path in created:
            path.unlink()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='Profile-bound exp_04 paired comparisons')
    parser.add_argument('--profile', choices=('H1_K8', 'H1_K1', 'H2_K8', 'TOST_K8'), required=True)
    parser.add_argument('--runs-a', nargs='+', required=True, help='First profile arm: aug')
    parser.add_argument('--runs-b', nargs='+', help='Second profile arm: control (H1/H2 only)')
    parser.add_argument('--json', required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--exploratory', action='store_true')
    args = parser.parse_args(argv)
    profile = get_profile(args.profile)
    if profile['mode'] == 'one_arm' and args.runs_b is not None:
        raise ValueError('one-arm TOST refuses --runs-b')
    if profile['mode'] == 'two_arm' and not args.runs_b:
        raise ValueError('two-arm profile requires --runs-b')
    groups = [(arm, profile['num_shot'], paths) for arm, paths in
              zip(profile['arms'], (args.runs_a, args.runs_b))]
    admitted = admit_runs(profile, groups, exploratory=args.exploratory)
    result = analyze(profile, admitted, args.exploratory)
    result['profile_name'] = args.profile
    write_outputs(result, admitted, args.json, args.summary)
    return result


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        raise SystemExit(str(error))
