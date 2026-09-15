"""Canonical-only exp_07 seen tables; rounding and unit display, no statistics here.

Every number comes from a canonical producer JSON that carries a valid sidecar: the seen
table (TABLE_SEEN_V1), the three descriptive pairings (PAIRS_SEEN_V1) and, for the
combined table, exp_04's unseen TABLE_V1 read through its own sidecar AND through the
digest its binding report recorded.  Exploratory or deviating JSON, a pairing that still
needs a larger n_boot, and an unseen table the exp_04 record does not bind are refused.
"""
import argparse
import html
import json
import subprocess
from pathlib import Path

from tools.exp04_record import load_asset

exp04 = load_asset('make_results_md')
display, percent, sha = exp04.display, exp04.percent, exp04.sha

REPO = Path(__file__).resolve().parents[4]
ROUNDING = ('Display rounding: EDT seconds 3 decimals with its SD in ms (1 decimal); C50 dB 3; '
            'T60 %, relative differences and interval bounds % 2; other values 6 significant '
            'figures. The canonical JSON keeps full precision and reports EDT in milliseconds.')
DESCRIPTIVE = ('Descriptive, checkpoint-conditional intervals for these checkpoints only: five '
               'evaluation seeds do not estimate training-seed variability. No verdict, no '
               'significance marker and no superiority or equivalence claim is made or implied.')
COLUMNS = ('EDT', 'C50', 'T60', 'loss', 'log_mse')
ACOUSTIC = ('EDT', 'C50', 'T60')
PAIRINGS = (('seen_cyl', 'seen_simple'), ('seen_aug', 'seen_simple'),
            ('released_seen', 'seen_simple'))
# Each seen arm and the exp_04 row that trained the same recipe under the other protocol.
PROTOCOL_PAIRS = (('seen_simple', 'control'), ('seen_aug', 'aug'), ('seen_cyl', 'cyl'))
REFERENCE_ROLE = 'released_seen'
KEYS = {'TABLE_SEEN_V1': 'producer_table', 'PAIRS_SEEN_V1': 'producer_pairs'}


def load(path, name):
    """Return the canonical data and its bound identity, or refuse it."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    data, digest = json.loads(raw), sha(raw)
    side = json.loads(Path(str(path) + '.provenance.json').read_text())
    if data.get('profile_name') != name:
        raise ValueError('wrong profile: ' + str(path))
    if len(side['outputs']) != 2 or side['outputs'].get(str(path)) != digest:
        raise ValueError('JSON and companion output coverage: ' + str(path))
    for output, expected in side['outputs'].items():
        if sha(Path(output).read_bytes()) != expected:
            raise ValueError('output digest mismatch: ' + output)
    profile_digest = sha(json.dumps(data['profile'], sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode())
    producer = side['producer']['sha256']
    if (side['profile_digest'] != profile_digest or data['profile_digest'] != profile_digest
            or side['inputs'] != data['inputs']
            or side['approved_digests']['sha256'] != data['inputs'].get(side['approved_digests']['path'])
            or side['run_flags'] != data.get('run_flags', side['run_flags'])
            or producer != side['approved_digests']['pins']['closures'][KEYS[name]]
            or data.get('producer_closure_sha256') != producer):
        raise ValueError('provenance sidecar mismatch: ' + str(path))
    if data.get('exploratory', False) is not False or side.get('exploratory', False) is not False:
        raise ValueError('exploratory JSON refused: ' + str(path))
    if data.get('deviations'):
        raise ValueError('admission deviations refused: ' + str(path))
    return data, dict(path=str(path), sha256=digest, profile_digest=profile_digest,
                      producer_closure_sha256=producer, approved_digests=side['approved_digests'],
                      outputs=side['outputs'])


def check_table(data):
    roles = [arm['role'] for arm in data['profile']['arms']]
    expected = {(role, shot) for role in roles for shot in data['profile']['num_shot']}
    actual = [(row['role'], row['num_shot']) for row in data['rows']]
    if len(actual) != 8 or set(actual) != expected or len(set(actual)) != len(actual):
        raise ValueError('incomplete seen table coverage')
    for row in data['rows']:
        missing = set(COLUMNS) - set(row['metrics'])
        if missing:
            raise ValueError('missing metrics: ' + ', '.join(sorted(missing)))


def check_pairs(records):
    """Three registered pairings, each one final, descriptive and completely covered."""
    if sorted(tuple(data['pairing']) for data in records) != sorted(PAIRINGS):
        raise ValueError('incomplete pairing coverage')
    for data in records:
        check_pairing(data)


def check_pairing(data):
    """One pairing: ten final descriptive cells, no verdict, every cell converged."""
    label = ' - '.join(data['pairing'])
    if data.get('final') is not True or data.get('reconverge_required'):
        raise ValueError('pairing is not final: ' + label)
    if (data.get('decision_driving') is not False or 'verdict' in data
            or any('verdict' in cell for cell in data['cells'])):
        raise ValueError('descriptive pairs cannot carry a verdict: ' + label)
    expected = {(metric, shot) for metric in data['profile']['metrics']['descriptive']
                for shot in data['profile']['num_shot']}
    actual = [(cell['metric'], cell['num_shot']) for cell in data['cells']]
    if len(actual) != 10 or set(actual) != expected or len(set(actual)) != len(actual):
        raise ValueError('incomplete pair cell coverage: ' + label)
    for cell in data['cells']:
        check_cell(data, cell, label)


def check_cell(data, cell, label):
    """A cell is publishable only if BOTH its statistics converged and it drives nothing.

    The aggregate flags are a summary; a mutated cell must not pass because the summary
    still says "final".  Every statistic the profile registers must be present, carry a
    passed convergence diagnostic and no outstanding reconvergence, and the cell's own
    pairing, grid and seed labels must be the parent's.
    """
    where = '{} {} K={}'.format(label, cell.get('metric'), cell.get('num_shot'))
    if cell.get('decision_driving') is not False:
        raise ValueError('a descriptive cell cannot drive a decision: ' + where)
    if list(cell.get('pairing') or []) != list(data['pairing']):
        raise ValueError('cell pairing differs from its pairing: ' + where)
    profile = data['profile']
    if cell.get('k') != profile['grid'][0] or list(cell.get('seed_labels') or []) != list(
            profile['eval_seeds']):
        raise ValueError('cell grid or seed labels differ from the profile: ' + where)
    if list(cell.get('quantiles') or []) != list(profile['quantiles']):
        raise ValueError('cell quantiles differ from the profile: ' + where)
    statistics = cell.get('statistics') or {}
    if set(statistics) != set(profile['statistics']):
        raise ValueError('cell statistics differ from the profile: ' + where)
    for name, statistic in sorted(statistics.items()):
        if statistic.get('reconverge_required') is not False:
            raise ValueError('cell {} still needs a larger n_boot: {}'.format(name, where))
        if (statistic.get('convergence') or {}).get('passed') is not True:
            raise ValueError('cell {} did not converge: {}'.format(name, where))
    if not (cell.get('paired_cohort') or {}).get('n_queries'):
        raise ValueError('empty paired cohort: ' + where)


def validate(data, name):
    """The canonical validation both the generators and the binder apply."""
    if name == 'TABLE_SEEN_V1':
        return check_table(data)
    if name == 'PAIRS_SEEN_V1':
        return check_pairing(data)
    raise ValueError('unregistered profile: ' + str(name))


def load_unseen(table, binding):
    """exp_04's own admission, plus the digest its binding report recorded for that table."""
    data, receipt, draft = exp04.load(table, 'TABLE_V1')
    if draft:
        raise ValueError('draft unseen table refused')
    report = json.loads(Path(binding).read_bytes())
    records = [item for item in report['results'] if item['profile'] == 'TABLE_V1']
    if len(records) != 1:
        raise ValueError('the exp_04 binding report does not bind TABLE_V1')
    bound = {item['path']: item['sha256'] for item in records[0]['outputs']}
    sidecar = Path(str(Path(table).resolve()) + '.provenance.json')
    if (bound.get(receipt['path']) != receipt['sha256']
            or records[0]['sidecar']['sha256'] != sha(sidecar.read_bytes())):
        raise ValueError('the unseen table is not the one the exp_04 report binds')
    receipt = dict(receipt, binding_report=dict(path=str(Path(binding).resolve()),
                                                sha256=sha(Path(binding).read_bytes())))
    return data, receipt


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', required=True, help='canonical TABLE_SEEN_V1.json')
    parser.add_argument('--pairs', nargs='+', required=True,
                        help='the three canonical PAIRS_SEEN_V1.json outputs')
    parser.add_argument('--unseen-table', required=True, help="exp_04's bound TABLE_V1.json")
    parser.add_argument('--unseen-binding', required=True, help="exp_04's binding report")
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    sources = [args.table] + list(args.pairs) + [args.unseen_table]
    if len({Path(item).resolve() for item in sources}) != len(sources):
        raise ValueError('duplicate canonical input')
    table, table_receipt = load(args.table, 'TABLE_SEEN_V1')
    validate(table, 'TABLE_SEEN_V1')
    pairs = [load(path, 'PAIRS_SEEN_V1') for path in sorted(args.pairs)]
    for data, _ in pairs:
        validate(data, 'PAIRS_SEEN_V1')
    check_pairs([data for data, _ in pairs])
    pairs.sort(key=lambda item: PAIRINGS.index(tuple(item[0]['pairing'])))
    unseen, unseen_receipt = load_unseen(args.unseen_table, args.unseen_binding)
    available = {(row['role'], row['num_shot']) for row in unseen['rows']}
    for seen_role, unseen_role in PROTOCOL_PAIRS:
        for shot in table['profile']['num_shot']:
            if (unseen_role, shot) not in available:
                raise ValueError('the unseen table has no {} row at K = {}'.format(unseen_role, shot))
    receipts = [table_receipt] + [receipt for _, receipt in pairs] + [unseen_receipt]
    published = {output for receipt in receipts for output in receipt['outputs']}
    published |= {str(Path(item).resolve()) + '.provenance.json' for item in sources}
    published |= {str(Path(item).resolve()) for item in sources}
    # The exp_04 binding report and every approval blob are inputs too: refuse to write
    # a document over any of them, whatever spelling named them on the command line.
    published |= {str(Path(args.unseen_binding).resolve())}
    published |= {str(Path(receipt['approved_digests']['path']).resolve())
                  for receipt in receipts if receipt.get('approved_digests')}
    published |= {str(Path(unseen_receipt['binding_report']['path']).resolve())}
    if str(Path(args.out).resolve()) in published:
        raise ValueError('output overlaps canonical input')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    record = dict(table=table, pairs=[data for data, _ in pairs], unseen=unseen, receipts=receipts)
    return args, record, head


def native(metric, value):
    return display(value, metric if metric in ACOUSTIC else None)


def mean_sd(metrics, name):
    """EDT in seconds with its SD in milliseconds, so no printed SD rounds to zero."""
    metric = metrics[name]
    if name == 'EDT':
        return '{} s ± {} ms'.format(format(metric['mean'] / 1000, '.3f'),
                                     display(metric['sd'], 'EDT'))
    return '{} ± {} {}'.format(native(name, metric['mean']), native(name, metric['sd']),
                               metric['unit'])


def protocol_line(row):
    values = dict(row['protocol'], tf32='on' if row['protocol']['tf32'] else 'off',
                  seeds=len(row['protocol']['seeds']))
    return ('K = {num_shot}; {split}; {n_queries} queries; {seeds} seeds; epoch {epoch}; '
            'condition {condition}; k = {k}; batch {batch_size}; TF32 {tf32}; '
            'training {training}').format(**values)


def tables(record, head):
    """Shared presentation model: titles, headers and producer-valued cells."""
    yield ('Seen split — mean ± seed SD (TABLE_SEEN_V1)',
           ['Arm', 'K', 'EDT (s)', 'EDT (ms)', 'C50 (dB)', 'T60 (%)', 'Loss', 'log-STFT MSE',
            'Protocol'],
           [[row['label'], row['num_shot'], mean_sd(row['metrics'], 'EDT'),
             '{} ± {}'.format(display(row['metrics']['EDT']['mean'], 'EDT'),
                              display(row['metrics']['EDT']['sd'], 'EDT'))] +
            [mean_sd(row['metrics'], name) for name in ('C50', 'T60', 'loss', 'log_mse')] +
            [protocol_line(row)] for row in record['table']['rows']])
    rows = []
    for data in record['pairs']:
        arm, baseline = data['pairing']
        for cell in data['cells']:
            absolute, relative = (cell['statistics'][key] for key in ('absolute', 'relative'))
            rows.append([' minus '.join(data['pairing']), cell['metric'], cell['num_shot'],
                         native(cell['metric'], absolute['estimate']), absolute['unit'],
                         native(cell['metric'], absolute['interval']),
                         native(cell['metric'], absolute['room_cluster_interval']),
                         percent(relative['estimate']), percent(relative['interval']),
                         percent(relative['room_cluster_interval']),
                         cell['paired_cohort']['n_queries'], cell['n_rooms_retained'],
                         percent(relative['convergence']['ratio']),
                         native(cell['metric'], cell['arm_means'][arm]['mean']),
                         native(cell['metric'], cell['arm_means'][baseline]['mean'])])
    yield ('Descriptive paired differences (PAIRS_SEEN_V1) — no verdict',
           ['Pairing', 'Metric', 'K', 'Absolute difference', 'Unit', 'Query 95% interval',
            'Room 95% interval', 'Relative difference (%)', 'Relative query 95% (%)',
            'Relative room 95% (%)', 'Queries', 'Rooms', 'Convergence ratio (%)',
            'Cohort mean (arm)', 'Cohort mean (baseline)'], rows)
    seen = {(row['role'], row['num_shot']): row for row in record['table']['rows']}
    unseen = {(row['role'], row['num_shot']): row for row in record['unseen']['rows']}
    rows = []
    for shot in (1, 8):
        for seen_role, unseen_role in PROTOCOL_PAIRS + ((REFERENCE_ROLE, None),):
            row = seen[(seen_role, shot)]
            other = unseen[(unseen_role, shot)] if unseen_role else None
            rows.append([row['label'], shot] +
                        [mean_sd(row['metrics'], name) for name in ACOUSTIC] +
                        ([mean_sd(other['metrics'], name) for name in ACOUSTIC]
                         if other else ['—'] * 3))
    yield ('Seen and unseen protocols — mean ± seed SD',
           ['Method', 'K', 'Seen EDT (s)', 'Seen C50 (dB)', 'Seen T60 (%)', 'Unseen EDT (s)',
            'Unseen C50 (dB)', 'Unseen T60 (%)'], rows)
    yield ('Provenance', ['Canonical input', 'sha256', 'Profile digest', 'Producer closure',
                          'Approval file', 'Approval sha256'],
           [[receipt['path'], receipt['sha256'], receipt['profile_digest'],
             receipt['producer_closure_sha256'], receipt['approved_digests']['path'],
             receipt['approved_digests']['sha256']] for receipt in record['receipts']])
    report = record['receipts'][-1]['binding_report']
    yield ('Provenance — git HEAD and the exp_04 record', ['Field', 'Value'],
           [['git HEAD', head], ['exp_04 binding report', report['path']],
            ['exp_04 binding report sha256', report['sha256']]])


def main(argv=None):
    args, record, head = arguments(argv)
    lines = ['generated by make_results_md.py from ' +
             ', '.join(receipt['sha256'] for receipt in record['receipts']) + ' at ' + head,
             ROUNDING, DESCRIPTIVE, '']
    for title, headers, rows in tables(record, head):
        lines += ['## ' + title, '', '| ' + ' | '.join(headers) + ' |',
                  '| ' + ' | '.join('---' for _ in headers) + ' |']
        lines += ['| ' + ' | '.join(html.escape(display(value)).replace('|', '&#124;')
                                    for value in row) + ' |' for row in rows]
        lines.append('')
    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
