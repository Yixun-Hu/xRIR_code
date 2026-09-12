"""Canonical five-evaluation-seed TABLE_V1 aggregation for exp_04."""
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.exp04_profiles import get_profile, json_value, load_approved_digests
from tools.paired_compare import admit_runs, producer_identity

# Source names are evaluator-owned; acoustic T60 is already a percentage.
METRICS = {'t60': ('T60', '%', 1), 'c50': ('C50', 'dB', 1),
           'edt': ('EDT', 'ms', 1000), 'loss': ('loss', 'objective', 1),
           'log_mse': ('log_mse', 'log-STFT MSE', 1),
           'stft': None, 'decay': None, 'consistency': None}


def build_table(directories, profile=None, approved=None):
    """Return canonical JSON data and admitted provenance; write no artefacts.

    Each seed mean uses its own finite queries; TABLE_V1 bounds count differences.
    The three recognized spectral diagnostics are validated but not table columns.
    """
    profile = get_profile('TABLE_V1') if profile is None else profile
    approved = load_approved_digests() if approved is None else approved
    groups = [(arm, shot, []) for arm in profile['arms'] for shot in sorted(profile['num_shot'])]
    for directory in sorted(str(Path(d).resolve()) for d in directories):
        fields = json.loads((Path(directory) / 'eval_manifest.json').read_text())
        matching = [(arm, shot, paths) for arm, shot, paths in groups
            if type(fields.get('num_shot')) is int and fields['num_shot'] == shot
            and fields.get('checkpoint_sha256') is not None
            and fields['checkpoint_sha256'] == (approved[0]['checkpoints']['aug']['sha256']
                if arm['role'] == 'aug' else arm['sha256'])]
        if len(matching) != 1:
            raise ValueError('checkpoint/num_shot does not map to a TABLE_V1 role: ' + directory)
        matching[0][2].append(directory)
    producer = producer_identity('tools.results_table')
    admitted = admit_runs(profile, groups, approved=approved, producer=producer,
                          producer_key='producer_results_table')
    rows = []
    for (arm, shot, _), runs in zip(groups, admitted['groups']):
        cells = [run['P']['0'] for run in runs]
        names = set().union(*(set(cell) for cell in cells))
        if names - set(METRICS):
            raise ValueError('unknown metric names: ' + ', '.join(sorted(names - set(METRICS))))
        if any(set(cell) != names for cell in cells):
            raise ValueError('metric coverage differs across seeds')
        metrics = {}
        for source, specification in METRICS.items():
            if specification is None or source not in names:
                continue
            name, unit, scale = specification
            per_seed = {}
            for run, cell in zip(runs, cells):
                values = np.asarray(cell[source], dtype=float)
                finite = values[np.isfinite(values)]
                if not finite.size:
                    raise ValueError('zero finite queries: ' + source)
                per_seed[str(run['meta']['manifest_seed'])] = {
                    'mean': float(finite.mean() * scale), 'n_finite': int(finite.size)}
            counts = [v['n_finite'] for v in per_seed.values()]
            if max(counts) - min(counts) > profile['finite_count_tolerance']:
                raise ValueError('finite count tolerance exceeded: ' + source)
            means = np.asarray([v['mean'] for v in per_seed.values()])
            if not np.isfinite(means).all():
                raise ValueError('nonfinite seed mean: ' + source)
            metrics[name] = dict(source=source, unit=unit, mean=float(means.mean()),
                                 sd=float(means.std(ddof=1)), per_seed=per_seed)
        protocol = dict(num_shot=shot, split=profile['dataset']['split'],
            n_queries=profile['dataset']['n_queries'], seeds=sorted(profile['seeds'][shot]),
            epoch=arm['epoch'], condition=profile['condition'], k=0,
            batch_size=profile['batch_size'], tf32=profile['tf32'], max_samples=profile['max_samples'])
        rows.append(dict(role=arm['role'], label=arm['label'], num_shot=shot,
                         epoch=arm['epoch'], protocol=protocol, metrics=metrics))
    literal = json_value(profile)
    digest = hashlib.sha256(json.dumps(literal, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return dict(schema_version=1, profile_name='TABLE_V1', profile=literal,
                profile_digest=digest, inputs=admitted['inputs'],
                producer_closure_sha256=producer['sha256'], rows=rows), admitted
