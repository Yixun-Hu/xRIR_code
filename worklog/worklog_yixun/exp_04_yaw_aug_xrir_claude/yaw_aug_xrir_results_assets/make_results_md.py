"""Canonical-only exp_04 tables. Ratios retain producer units; no statistics here."""
import argparse
import hashlib
import html
import json
import subprocess
from pathlib import Path

INPUTS = tuple(zip(('--h1-k8', '--h1-k1', '--h2-k8', '--tost-k8', '--table'),
                   ('H1_K8', 'H1_K1', 'H2_K8', 'TOST_K8', 'TABLE_V1')))
REPO = Path(__file__).resolve().parents[4]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def display(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def load(path, name=None, draft_ok=False):
    path = Path(path).resolve()
    raw = path.read_bytes()
    data, side = json.loads(raw), json.loads(Path(str(path) + '.provenance.json').read_text())
    digest = sha(raw)
    if len(side['outputs']) != 2 or str(path) not in side['outputs']:
        raise ValueError('JSON and summary output coverage')
    for output, expected in side['outputs'].items():
        if sha(Path(output).read_bytes()) != expected:
            raise ValueError('JSON or summary digest mismatch: ' + output)
    profile_digest = sha(json.dumps(data['profile'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    producer = side['producer']['sha256']
    key = 'producer_results_table' if data['profile_name'] == 'TABLE_V1' else 'producer_paired_compare'
    if (side['outputs'].get(str(path)) != digest or side['profile_digest'] != profile_digest
            or data['profile_digest'] != profile_digest or side['inputs'] != data['inputs']
            or side['approved_digests']['sha256'] != data['inputs'].get(side['approved_digests']['path'])
            or side['run_flags'] != data.get('run_flags', side['run_flags'])
            or producer != side['approved_digests']['pins']['closures'][key]
            or data.get('producer_closure_sha256', producer) != producer):
        raise ValueError('provenance sidecar mismatch: ' + str(path))
    if data.get('exploratory', False) is not False or side.get('exploratory', False) is not False or data.get('deviations'):
        raise ValueError('exploratory JSON refused')
    if name is not None and data['profile_name'] != name:
        raise ValueError('wrong profile')
    profile, draft = data['profile'], False
    if key == 'producer_results_table':
        expected = {(a['role'], k) for a in profile['arms'] for k in profile['num_shot']}
        actual = [(r['role'], r['num_shot']) for r in data['rows']]
    else:
        expected = {(m, k) for group in profile['metrics'].values() for m in group for k in profile['grid']}
        actual = [(c['metric'], c['k']) for c in data['cells']]
        driving = [c for c in data['cells'] if c['decision_driving']]
        verdicts = [c.get('verdict') for c in driving] if profile['mode'] == 'one_arm' else [data.get('verdict')]
        draft = not verdicts or any(not isinstance(v, str) or not v for v in verdicts)
        for cell in driving:
            gates = cell['convergence']
            required = {'decision', 'superiority'} if profile['input_selection'] == 'standalone_k0' else {'decision'}
            if set(gates) != required or any(g['passed'] is not True for g in gates.values()):
                raise ValueError('convergence gate missing or failed')
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('incomplete cell coverage')
    if draft and not draft_ok:
        raise ValueError('verdict absent')
    return data, dict(path=str(path), sha256=digest, profile_digest=profile_digest,
                      producer_closure_sha256=producer, approved_digests=side['approved_digests'], outputs=side['outputs']), draft


def tables(data):
    """Shared presentation model: titles, headers and producer-valued cells."""
    name = data['profile_name']
    if name == 'TABLE_V1':
        metrics = ('T60', 'C50', 'EDT', 'loss', 'log_mse')
        rows = [[r['label'], r['num_shot']] + ['{} ± {} {}'.format(
            display(r['metrics'][m]['mean']), display(r['metrics'][m]['sd']), r['metrics'][m]['unit'])
            for m in metrics] + [r['protocol']] for r in data['rows']]
        yield name + ' — mean ± seed SD', ['Arm', 'K'] + list(metrics) + ['Protocol'], rows
    else:
        estimate = 'ρ' if name.startswith('H1') else 'D_k' if name.startswith('H2') else 'r_k'
        headers = ['Metric', 'Role', 'Degrees', estimate + ' (ratio)', 'One-sided upper', 'Companion interval',
                   'Room-cluster interval', 'Queries', 'Rooms', 'Verdict', 'superiority', 'Superiority interval']
        rows = [[c['metric'], 'primary' if c['primary'] else 'supportive' if c['decision_driving'] else 'descriptive',
                 c['degrees'], c['estimate'], c.get('decision_bound'), c['companion_interval'], c['room_cluster_interval'],
                 c['exclusions']['joint']['valid'], c['n_rooms_retained'], c.get('verdict', '—'),
                 c.get('superiority'), c.get('superiority_interval')] for c in data['cells']]
        yield name + (' — aggregate verdict: ' + data['verdict'] if 'verdict' in data else ''), headers, rows
        for field in ('exclusions', 'seed_means', 'convergence'):
            yield name + ' — ' + field, ['Metric', 'k', field], [[c['metric'], c['k'], c[field]] for c in data['cells']]
    yield name + ' — protocol / margins', ['Field', 'Value'], list(data['profile'].items())


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, _ in INPUTS:
        parser.add_argument(flag, required=True)
    parser.add_argument('--diag', nargs='+', action='extend', default=[])
    parser.add_argument('--out', required=True)
    parser.add_argument('--draft-ok', action='store_true', help='Tests only: permit absent verdicts')
    args = parser.parse_args(argv)
    sources = [(getattr(args, flag[2:].replace('-', '_')), name) for flag, name in INPUTS]
    sources += [(path, None) for path in args.diag]
    records = [load(path, name, args.draft_ok) for path, name in sources]
    if Path(args.out).resolve() in {Path(p).resolve() for _, receipt, _ in records for p in receipt['outputs']} | {Path(str(Path(p).resolve()) + '.provenance.json') for p, _ in sources}:
        raise ValueError('output overlaps canonical input')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return args, records, head


def main(argv=None):
    args, records, head = arguments(argv)
    lines = ['generated by make_results_md.py from ' + ', '.join(r[1]['sha256'] for r in records) + ' at ' + head, '']
    if any(r[2] for r in records):
        lines += ['DRAFT — missing verdict; tests only', '']
    blocks = [block for data, _, _ in records for block in tables(data)]
    blocks.append(('Provenance', ['Input JSON', 'sha256', 'Profile digest', 'Producer closure digest', 'Approval'],
                   [[p['path'], p['sha256'], p['profile_digest'], p['producer_closure_sha256'], p['approved_digests']] for _, p, _ in records]))
    for title, headers, rows in blocks:
        lines += ['## ' + title, '', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |']
        lines += ['| ' + ' | '.join(html.escape(display(v)).replace('|', '&#124;').replace('\n', '<br>') for v in row) + ' |' for row in rows]
        lines.append('')
    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
