"""Profile-driven paired inference for exp_04; pure statistical functions."""
from decimal import Decimal, ROUND_CEILING
import numbers

import numpy as np

from tools.paired_stats import (
    _bootstrap_ratios, relative_degradation_bootstrap, equivalence_tost,
    diff_in_diff_bootstrap, bonferroni_alpha, valid_mask,
)


def _quantile(samples, probability):
    """Inverse empirical CDF: rank ceil(p*n), with ranks starting at one."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("quantile samples must be a nonempty finite 1-D array")
    # Decimal ranks avoid a binary rounding step promoting rank 3 to 4
    # for alpha=.7 and n=10. The input decimal is interpreted literally.
    probability = Decimal(str(probability))
    if not probability.is_finite() or not 0 < probability < 1:
        raise ValueError("quantile probability must be in (0, 1)")
    rank = int((probability * values.size).to_integral_value(rounding=ROUND_CEILING)) - 1
    return float(np.partition(values, rank)[rank])


def one_sided_upper(samples, alpha):
    """Smallest v with >= ceil((1-alpha)*n) samples <= v; no interpolation.

    For [1, 2, 3, 4] and alpha=.25 the upper bound is 3. A bound equal
    to a practical margin fails the strict confirmatory decision rule.
    """
    return _quantile(samples, Decimal(1) - Decimal(str(alpha)))


def two_sided_interval(samples, alpha):
    """Exact [Q_(alpha/2), Q_(1-alpha/2)] order-statistic interval."""
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    tail = Decimal(str(alpha)) / 2
    return (_quantile(samples, tail), _quantile(samples, Decimal(1) - tail))


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
                       "seeds": {seed: _validity_counts(b, p) for seed, b, p in
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
    lo, hi = two_sided_interval(samples, 2 * alpha_local)
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


# Admission imports are at module scope so source_closure sees every dependency.
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from tools import provenance as provenance
from tools.exp04_profiles import get_profile, json_value
from tools.reference_manifest import load_manifest, manifest_hash
from tools.summarize_yaw import load_run, rooms_from_paths, signed_degrees, _check_metrics_reconciliation

REPO = Path(__file__).resolve().parents[1]
_UNBOUND = object()


def _digest(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def producer_identity():
    """Bind imported producer dependencies to committed HEAD and current bytes."""
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    records, digest = provenance.closure_record(
        provenance.source_closure('tools.paired_compare', REPO), commit, REPO)
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


def _admit_run(directory, arm, profile, check, inputs):
    directory = Path(directory).resolve()
    label = str(directory)
    def require(ok, name):
        check(ok, label + ': ' + name)
    def bind(path, expected=_UNBOUND):
        path = str(Path(path).resolve())
        actual = provenance.sha256_file(path)
        require(expected is _UNBOUND or actual == expected, 'digest ' + path)
        require(path not in inputs or inputs[path] == actual, 'input changed ' + path)
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
    root = Path(fields['repo'])
    declarations = dict(fields, eval_manifest={'path': str(files['eval_manifest.json']), 'sha256': digest})
    for failure in provenance.revalidate(declarations, required=(
            'repo', 'checkpoint', 'checkpoint_sha256', 'manifest_path', 'manifest_file_sha256',
            'data_identity', 'evaluator_closure', 'source_closures', 'mutable_inputs', 'eval_manifest')):
        require(False, 'revalidate ' + failure)
    closures = dict(fields['source_closures'], frozen_evaluator=fields['evaluator_closure'])
    for name, closure in closures.items():
        require(_closure_digest(closure) == closure['sha256'], 'closure digest ' + name)
        for record in closure['files']:
            bind(root / record['path'], record['working_tree_sha256'])
    for approved, recorded in (('evaluator', 'entrypoint'), ('writer', 'writer'), ('launcher', 'writer')):
        require(closures[recorded]['sha256'] == profile['approved_closures'][approved],
                approved + ' approved closure')
    reference = load_manifest(root / fields['manifest_path'])
    seed, shot = fields['manifest_seed'], profile['num_shot']
    require(type(seed) is int, 'seed type')
    expected = {'num_shot': shot, 'gl_seed': seed, 'split': profile['dataset']['split'],
        'split_count': profile['dataset']['n_queries'], 'n_samples': profile['dataset']['n_queries'],
        'conditions': profile['condition'], 'batch_size': profile['batch_size'],
        'max_samples': profile['max_samples'], 'tf32': profile['tf32'], 'batch_canonical': True,
        'no_tta': True, 'backbone': arm['backbone'], 'checkpoint_sha256': arm['sha256'],
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
    for record in identity['inventory']:
        bind(Path(identity['data_root']) / record['path'], record['sha256'])
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
        require(_equal(fields.get(key), grid), 'grid ' + key)
    for payload in (run, aggregate):
        require('E' not in payload, 'condition P only')
        require(set(payload['P']) == {str(k) for k in grid}, 'grid P cells')
    for angle, cell in run['P'].items():
        require(set(cell) == set(aggregate['P'].get(angle, {})), 'metrics cell coverage')
        for name in ('edt', 'c50', 't60'):
            require(name in cell, 'metric missing ' + name)
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
    return run


def admit_runs(profile, runs_a, runs_b=None, exploratory=False):
    """Finish every admission check before computing any statistic or writing files."""
    if profile['mode'] == 'one_arm' and runs_b is not None:
        raise ValueError('one-arm TOST refuses --runs-b')
    if profile['mode'] == 'two_arm' and not runs_b:
        raise ValueError('two-arm profile requires --runs-b')
    required = list(profile['approved_closures'].items())
    required += [(a['role'] + ' checkpoint', a['sha256']) for a in profile['arms']]
    required += [('dataset inventory', profile['dataset']['inventory_sha256'])]
    for key, value in required:
        if value is None:
            raise ValueError('profile not yet approved: ' + key)
    deviations, inputs = [], {}
    def check(ok, name):
        if not ok:
            deviations.append(name)
    producer = producer_identity()
    check(producer['sha256'] == profile['approved_closures']['producer'], 'producer approved closure')
    for record in producer['files']:
        inputs[str((REPO / record['path']).resolve())] = record['working_tree_sha256']
    groups = []
    for arm, directories in zip(profile['arms'], (runs_a, runs_b)):
        runs = []
        for directory in directories:
            try:
                runs.append(_admit_run(directory, arm, profile, check, inputs))
            except (KeyError, TypeError, ValueError, OSError, IndexError) as error:
                check(False, '{}: admission {}'.format(directory, error))
        seeds = [r['meta'].get('manifest_seed') for r in runs]
        check(len(runs) == 5 and all(type(s) is int for s in seeds) and
              set(seeds) == set(profile['seeds'][profile['num_shot']]), arm['role'] + ' seed set')
        groups.append(sorted(runs, key=lambda r: str(r['meta'].get('manifest_seed'))))
    flat = [r for group in groups for r in group]
    if flat:
        check(all(r['query'] == flat[0]['query'] and r['index'] == flat[0]['index'] for r in flat),
              'query/index order across runs')
    if deviations and not exploratory:
        raise ValueError('admission failed: ' + '; '.join(deviations))
    return {'groups': groups, 'deviations': deviations, 'inputs': inputs, 'producer': producer}


