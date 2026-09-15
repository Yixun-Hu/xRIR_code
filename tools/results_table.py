"""Canonical five-evaluation-seed TABLE_V1 aggregation for exp_04."""
import argparse
import datetime
import os
import sys
import shlex
import re
import tempfile
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.provenance import apply_umask
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
    pins = [(key, approved[0]['closures'][key]) for key in ('evaluator', 'writer', 'producer_results_table')]
    pins += [('aug checkpoint ' + key, value) for key, value in approved[0]['checkpoints']['aug'].items()]
    pins += [(a['role'] + ' checkpoint', a['sha256']) for a in profile['arms'] if a['role'] != 'aug']
    pins += [('approval schema_version', approved[0]['schema_version']),
             ('dataset inventory', profile['dataset']['inventory_sha256'])]
    for key, value in pins:
        if value is None:
            raise ValueError('profile not yet approved: ' + key)
    groups = [(arm, shot, []) for arm in profile['arms'] for shot in sorted(profile['num_shot'], reverse=True)]
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


BEGIN, END = '<!-- results_table:begin -->', '<!-- results_table:end -->'
STAMP = '<!-- results_table:sha256:'


def render_markdown(json_path, generation_command=None):
    """Read all table content from JSON; the sidecar supplies only command provenance.

    During publication the writer supplies that command before finalizing the sidecar.
    No run directories are opened, including when regenerating an existing table.
    """
    path = Path(json_path).absolute()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    result = json.loads(raw)
    sidecar = Path(str(path) + '.provenance.json')
    if sidecar.exists():
        receipt = json.loads(sidecar.read_text())
        if receipt['outputs'].get(str(path)) != digest:
            raise ValueError('canonical JSON digest mismatch')
        generation_command = receipt['generation_command']
    lines = [BEGIN, '# Model comparison', '',
        'Mean ± sample SD over five evaluation seeds (42–46). Each seed selects the K-specific '
        'reference manifest and the Griffin-Lim phase. Each seed mean uses its finite queries; '
        'per-seed finite counts are recorded in the canonical JSON. '
        'Values are rounded to 6 significant digits; the JSON is canonical.', '']
    columns = ('T60', 'C50', 'EDT', 'loss', 'log_mse')
    lines += ['| Model | K | T60 (%) | C50 (dB) | EDT (ms) | Loss (objective) | Log-STFT MSE | Protocol |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in result['rows']:
        cells = ['{:.6g} ± {:.6g}'.format(row['metrics'][name]['mean'], row['metrics'][name]['sd'])
                 if name in row['metrics'] else '—' for name in columns]
        protocol = row['protocol']
        description = ('K = {num_shot}; {split}; {n_queries} queries; ' +
            '{} seeds; epoch {{epoch}}; {{condition}}; k = {{k}}; batch {{batch_size}}; TF32 {{precision}}'
            .format(len(protocol['seeds']))).format(**protocol, precision='on' if protocol['tf32'] else 'off')
        lines.append('| ' + ' | '.join([row['label'], str(row['num_shot'])] + cells + [description]) + ' |')
    lines += ['', '## Protocol', '',
        'Rows report K, {dataset[split]} split query count, seed count, epoch, condition {condition}, '
        'standalone k = {grid[0]}, canonical batch {batch_size}, and TF32 {precision}. '
        'max_samples = {max_samples}; no test-time augmentation.'.format(
            **result['profile'], precision='on' if result['profile']['tf32'] else 'off'),
        'Finite-count tolerance: {} queries between seeds in each row/metric; empty seeds are refused.'
        .format(result['profile']['finite_count_tolerance']), '', '## Provenance', '',
        'Canonical JSON: `{}`; sha256: `{}`.'.format(path, digest),
        'Profile digest: `{}`.'.format(result['profile_digest']), '', 'Generation command:',
        '```sh', shlex.join(generation_command or []), '```']
    body = '\n'.join(lines) + '\n'
    return body + STAMP + hashlib.sha256(body.encode()).hexdigest() + ' -->\n' + END + '\n'


def _preserve_manual(generated, previous):
    if previous is None:
        return generated
    text = previous.decode('utf-8')
    blocks = list(re.finditer(re.escape(BEGIN) + r'.*?' + re.escape(END), text, re.S))
    if len(blocks) > 1 or text.count(BEGIN) != len(blocks) or text.count(END) != len(blocks):
        raise ValueError('ambiguous generated block markers; keep exactly one generated block')
    for block in blocks:
        body, separator, claimed = block.group()[:-len(END)].rpartition(STAMP)
        expected = hashlib.sha256(body.encode()).hexdigest() + ' -->\n'
        if not separator or claimed != expected:
            raise ValueError('generated block was edited by hand; move your notes below the end marker and rerun')
        text = text[:block.start()] + text[block.end():]
    text = text.strip()
    if text.startswith('## Manual\n'):
        text = text[len('## Manual\n'):].strip()
    return generated + ('\n## Manual\n\n' + text + '\n' if text else '')


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _publish(path, payload, overwrite=False):
    """Atomic complete-file publication; hard-link creation refuses an existing name."""
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            apply_umask(stream.fileno())
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
        markdown = _preserve_manual(render_markdown(paths[0], command), previous).encode()
        receipt = dict(schema_version=1, profile_digest=result['profile_digest'],
            inputs=admitted['inputs'], producer=admitted['producer'],
            approved_digests=admitted['approved_digests'], run_flags=admitted['run_flags'],
            generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            generation_command=list(command), outputs={str(path): hashlib.sha256(payload).hexdigest()
                for path, payload in zip(paths, (data, markdown))})
        recheck_inputs(admitted)
        current = paths[1].read_bytes() if paths[1].exists() else None
        if current != previous:
            raise ValueError('Markdown changed during rendering')
        _publish(paths[1], markdown, overwrite=previous is not None)
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
    command = ['results_table.py', '--profile', args.profile, '--runs'] + [str(Path(d).resolve()) for d in args.runs]
    command += ['--json', str(Path(args.json).absolute()), '--md', str(Path(args.md).absolute())]
    write_outputs(result, admitted, args.json, args.md, args.force_md, command + (['--force-md'] if args.force_md else []))
    return result


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        raise SystemExit(str(error))
