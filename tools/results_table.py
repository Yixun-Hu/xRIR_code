"""Canonical five-evaluation-seed TABLE_V1 aggregation for exp_04."""
import argparse
import datetime
import os
import sys
import tempfile
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.exp04_profiles import get_profile, json_value, load_approved_digests
from tools.paired_compare import admit_runs, producer_identity, recheck_inputs

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
    literal = json.loads(json.dumps(json_value(profile)))
    digest = hashlib.sha256(json.dumps(literal, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return dict(schema_version=1, profile_name='TABLE_V1', profile=literal,
                profile_digest=digest, inputs=admitted['inputs'],
                producer_closure_sha256=producer['sha256'], rows=rows), admitted


def render_markdown(json_path):
    """Render metric cells and protocol exclusively from the saved canonical JSON."""
    result = json.loads(Path(json_path).read_text())
    columns = ('T60', 'C50', 'EDT', 'loss', 'log_mse')
    lines = ['| Model | K | T60 (%) | C50 (dB) | EDT (ms) | Loss (objective) | Log-STFT MSE | Protocol |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in result['rows']:
        cells = ['{:.6g} ± {:.6g}'.format(row['metrics'][name]['mean'], row['metrics'][name]['sd'])
                 if name in row['metrics'] else '—' for name in columns]
        protocol = row['protocol']
        description = ('K = {num_shot}; {split}; {n_queries} queries; ' +
            '{} seeds; epoch {{epoch}}; {{condition}}; k = {{k}}; batch {{batch_size}}; TF32 {{precision}}'
            .format(len(protocol['seeds']))).format(**protocol, precision='on' if protocol['tf32'] else 'off')
        lines.append('| ' + ' | '.join([row['label'], str(row['num_shot'])] + cells + [description]) + ' |')
    return '\n'.join(lines) + '\n'


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _publish(path, payload, overwrite=False):
    """Atomic complete-file publication; hard-link creation refuses an existing name."""
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_outputs(result, admitted, json_path, md_path, force_md=False, command=()):
    paths = [Path(json_path).absolute(), Path(md_path).absolute(),
             Path(str(json_path) + '.provenance.json').absolute()]
    if (len({p.resolve() for p in paths}) != 3 or any(p.is_symlink() for p in paths)
            or any(os.path.lexists(paths[i]) for i in (0, 2))
            or (os.path.lexists(paths[1]) and not force_md)):
        raise FileExistsError('output paths must be distinct and absent; --force-md permits Markdown only')
    if any(str(path.resolve()) in admitted['inputs'] for path in paths):
        raise ValueError('output overlaps an input')
    previous = paths[1].read_bytes() if paths[1].exists() else None
    data, created = _json_bytes(result), []
    recheck_inputs(admitted)
    try:
        _publish(paths[0], data)
        created.append(paths[0])
        markdown = render_markdown(paths[0]).encode()
        receipt = dict(schema_version=1, profile_digest=result['profile_digest'],
            inputs=admitted['inputs'], producer=admitted['producer'],
            approved_digests=admitted['approved_digests'], run_flags=admitted['run_flags'],
            generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            generation_command=list(command), outputs={str(path): hashlib.sha256(payload).hexdigest()
                for path, payload in zip(paths, (data, markdown))})
        recheck_inputs(admitted)
        _publish(paths[1], markdown, overwrite=force_md)
        created.append(paths[1])
        _publish(paths[2], _json_bytes(receipt))
    except BaseException:
        for path in reversed(created):
            if path == paths[1] and previous is not None:
                _publish(path, previous, overwrite=True)
            else:
                path.unlink()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('TABLE_V1',), required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--json', required=True)
    parser.add_argument('--md', required=True)
    parser.add_argument('--force-md', action='store_true', help='Regenerate Markdown; JSON is always exclusive')
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    result, admitted = build_table(args.runs)
    write_outputs(result, admitted, args.json, args.md, args.force_md, ['results_table.py'] + list(argv))
    return result


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error))
