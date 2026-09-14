"""Canonical-only exp_04 tables; rounding and unit display, no statistics here."""
import argparse
import hashlib
import html
import json
import subprocess
from pathlib import Path

INPUTS = tuple(zip(('--h1-k8', '--h1-k1', '--h2-k8', '--tost-k8', '--table'),
                   ('H1_K8', 'H1_K1', 'H2_K8', 'TOST_K8', 'TABLE_V1')))
REPO = Path(__file__).resolve().parents[4]
ROUNDING = 'Display rounding: EDT ms 1 decimal; C50 dB 3; T60 %, ratios and bounds % 2; other values 6 significant figures. JSON remains canonical.'
VERDICTS = {'H1': {'non-inferior', 'non-inferior on EDT only', 'non-inferior on C50 only', 'not shown'},
            'H2': {'supported', 'partially supported', 'not supported'},
            'TOST': {'equivalent', 'equivalence not established'}}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def display(value, metric=None, scale=1):
    if value is None: return '—'
    if isinstance(value, bool): return 'yes' if value else 'no'
    if isinstance(value, dict):
        return '; '.join(str(k) + ': ' + display(v, metric, scale) for k, v in value.items()) or '—'
    if isinstance(value, (tuple, list)):
        return ', '.join(display(v, metric, scale) for v in value) or '—'
    if isinstance(value, float):
        return format(value * scale, '.' + str({'EDT': 1, 'C50': 3, 'T60': 2, 'ratio': 2}[metric]) + 'f') if metric else format(value, '.6g')
    return str(value)


def percent(value):
    return display(float(value) if type(value) is int else value, 'ratio', 100)


def summaries(data):
    name = data['profile_name']
    if name.startswith(('H1', 'H2')):
        yield name + ' verdict', ['Family', 'Verdict'], [[name, data.get('verdict', '—')]]
    elif name == 'TOST_K8':
        yield name + ' verdicts', ['Metric', 'Degrees', 'Verdict'], [
            [c['metric'], c['degrees'], c.get('verdict', '—')] for c in data['cells'] if c['decision_driving']]


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
        family = data['profile_name'].split('_')[0]
        if family not in VERDICTS or any(v and v not in VERDICTS[family] for v in verdicts):
            raise ValueError('unregistered verdict wording')
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
            display(r['metrics'][m]['mean'], m if m in ('EDT', 'C50', 'T60') else None), display(r['metrics'][m]['sd'], m if m in ('EDT', 'C50', 'T60') else None), r['metrics'][m]['unit'])
            for m in metrics] + [r['protocol']] for r in data['rows']]
        yield name + ' — mean ± seed SD', ['Arm', 'K'] + list(metrics) + ['Protocol'], rows
    else:
        estimate = 'ρ' if name.startswith('H1') else 'D_k' if name.startswith('H2') else 'r_k'
        headers = ['Metric', 'Role', 'Degrees', estimate + ' (%)', 'One-sided upper (%)', 'Companion interval (%)',
                   'Room-cluster interval (%)', 'Queries', 'Rooms', 'Verdict', 'superiority', 'Superiority interval (%)']
        rows = [[c['metric'], 'primary' if c['primary'] else 'supportive' if c['decision_driving'] else 'descriptive',
                 c['degrees'], percent(c['estimate']), percent(c.get('decision_bound')), percent(c['companion_interval']), percent(c['room_cluster_interval']),
                 c['exclusions']['joint']['valid'], c['n_rooms_retained'], c.get('verdict', '—'),
                 c.get('superiority'), percent(c.get('superiority_interval'))] for c in data['cells']]
        yield name + (' — aggregate verdict: ' + data['verdict'] if 'verdict' in data else ''), headers, rows
        for field in ('exclusions', 'seed_means', 'convergence'):
            yield name + ' — ' + field + (' (%, except seeds)' if field == 'convergence' else ''), ['Metric', 'k', field], [[c['metric'], c['k'],
                display(c[field], c['metric'], 1000 if c['metric'] == 'EDT' else 1) + (' ms' if c['metric'] == 'EDT' else ' dB' if c['metric'] == 'C50' else ' %')
                if field == 'seed_means' else percent(c[field]) if field == 'convergence' else c[field]] for c in data['cells']]
    yield name + ' — protocol / margins', ['Field', 'Value'], [(k + ' (%)', percent(v)) if k in ('margin', 'convergence_tolerance') else (k, v) for k, v in data['profile'].items()]


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
    if len({Path(p).resolve() for p, _ in sources}) != len(sources):
        raise ValueError('duplicate canonical input or diagnostic')
    records = [load(path, name, args.draft_ok) for path, name in sources]
    if any(data['profile'].get('mode') != 'descriptive' for data, _, _ in records[5:]):
        raise ValueError('diagnostic requires a descriptive producer profile')
    if Path(args.out).resolve() in {Path(p).resolve() for _, receipt, _ in records for p in receipt['outputs']} | {Path(str(Path(p).resolve()) + '.provenance.json') for p, _ in sources}:
        raise ValueError('output overlaps canonical input')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return args, records, head


def main(argv=None):
    args, records, head = arguments(argv)
    lines = ['generated by make_results_md.py from ' + ', '.join(r[1]['sha256'] for r in records) + ' at ' + head, ROUNDING, '']
    if any(r[2] for r in records):
        lines += ['DRAFT — missing verdict; tests only', '']
    blocks = [block for data, _, _ in records for block in summaries(data)]
    blocks += [block for data, _, _ in records for block in tables(data)]
    blocks.append(('Provenance', ['Input JSON', 'sha256', 'Profile digest', 'Producer closure digest', 'Approval'],
                   [[p['path'], p['sha256'], p['profile_digest'], p['producer_closure_sha256'], p['approved_digests']] for _, p, _ in records]))
    for title, headers, rows in blocks:
        lines += ['## ' + title, '', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |']
        lines += ['| ' + ' | '.join(html.escape(display(v)).replace('|', '&#124;').replace('\n', '<br>') for v in row) + ' |' for row in rows]
        lines.append('')
    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
