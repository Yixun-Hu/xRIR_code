"""Canonical-only exp_05 tables; rounding and unit display, no statistics here.

Every number comes from one of the five canonical `tools/param_curve.py` products read
through `tools.exp05_record` -- H1's two curve families, H2's two fixed-target families
and the descriptive yaw block -- except the throughput, memory and per-epoch trajectory
columns, which no canonical JSON carries: those are read from the four certified
training attempts, every file at the digest the attempt's own completion sidecar records.

The six-arm table's M rows come from exp_05's OWN M re-evaluations inside CURVE_K8 and
CURVE_K1 (the twenty `M_*` runs through `tools/exp05_eval.py`), never from exp_04's
TABLE_V1: the protocols differ and the two must not be mixed in one row.

The parameter table is measured here with `tools.exp05_params.count_parameters` and must
agree exactly with the counts the profile registered and the products carry.
"""
import argparse
import html
import json
import os
import subprocess
from pathlib import Path

from tools import exp05_record as record
from tools.exp04_record import load_asset
from tools.exp05_profiles import ARMS
from tools.summarize_yaw import signed_degrees

exp04 = load_asset('make_results_md')
display, percent, sha = exp04.display, exp04.percent, exp04.sha

REPO = Path(__file__).resolve().parents[4]
INPUTS = tuple(zip(('--curve-k8', '--curve-k1', '--targets-k8', '--targets-k1', '--yaw-k8'),
                   ('CURVE_K8', 'CURVE_K1', 'TARGETS_K8', 'TARGETS_K1', 'YAW_K8_SEED42')))
CURVES = ('CURVE_K8', 'CURVE_K1')
TARGETS = ('TARGETS_K8', 'TARGETS_K1')
YAW = 'YAW_K8_SEED42'
ACOUSTIC = ('EDT', 'C50', 'T60')
UNITS = dict(EDT='s', C50='dB', T60='%')
ROLES = {arm['role']: arm for arm in ARMS}
TRAINED = tuple(arm['role'] for arm in ARMS if arm['tier'] != 'M')
ROUNDING = ('Display rounding: EDT seconds 4 decimals with its seed SD in ms (1 decimal); '
            'C50 dB 3; T60 %, relative differences and interval bounds % 2; parameter counts '
            'are exact integers; other values 6 significant figures. The canonical JSON keeps '
            'full precision and reports EDT in seconds.')
SCOPE = ('H1 verdicts are stated per metric at the three tested tiers only and are never '
         'extrapolated between capacities. H2 is one four-cell family: each target and each '
         '"reaches the target" claim is scoped to that cell\'s paired cohort, with the '
         'baseline\'s own-cohort mean shown separately. The yaw block is descriptive. The M '
         'tier is exp_01\'s pair re-evaluated under exp_05\'s protocol; its training predates '
         'this experiment (different day, launcher cadence, unrecorded PYTHONHASHSEED) and no '
         'tier pair shares a training realization (one seed per arm, unshared data order and '
         'reference draws).')
MEASURED = {}


def measured_counts():
    """Measure all six capacities once per process; refuse any drift from the register."""
    if not MEASURED:
        from tools.exp05_params import build_tier, count_parameters
        for role, arm in sorted(ROLES.items()):
            counts = count_parameters(build_tier(arm['backbone'], arm['tier']))
            record.refuse(counts == dict(arm['counts']),
                          'measured parameter counts differ from the register: ' + role)
            MEASURED[role] = counts
    return MEASURED


def native(metric, value, scale=1):
    """Acoustic metrics in their own units; everything else at six significant figures."""
    if metric == 'EDT' and not isinstance(value, str):
        if isinstance(value, (list, tuple)):
            return ', '.join(native(metric, item, scale) for item in value)
        return '—' if value is None else format(value * scale, '.4f')
    return display(value, metric if metric in ACOUSTIC else None)


def mean_sd(point):
    """A curve point's five-seed mean with its seed SD, EDT's in milliseconds."""
    metric = point['metric']
    if metric == 'EDT':
        return '{} s ± {} ms'.format(native(metric, point['mean']),
                                     display(point['sd'], 'EDT', 1000) if point['sd'] else '—')
    return '{} ± {} {}'.format(native(metric, point['mean']),
                               native(metric, point['sd']) if point['sd'] else '—', UNITS[metric])


def points(data):
    return {(point['arm'], point['metric']): point for point in data['curves'] if point['k'] == 0}


def gate(cell, name='superiority'):
    return percent(cell['convergence'][name]['ratio'])


def curve_rows(record_data, name):
    """H1: one row per tier and metric, with the adjusted interval that decides it."""
    data = record_data['products'][name]
    shot = data['profile']['num_shot']
    known = points(data)
    rows = []
    for pairing in data['profile']['pairings']:
        for metric in data['profile']['metrics']['primary']:
            cell = next(item for item in data['cells'] if item['pairing'] == list(pairing)
                        and item['metric'] == metric)
            arm, base = [ROLES[role] for role in pairing]
            means = cell['paired_means']
            rows.append(['K = {}'.format(shot), arm['tier'], ' vs '.join(pairing), metric,
                         percent(cell['estimate']), percent(cell['companion_interval']),
                         percent(cell['room_cluster_interval']), cell['paired_cohort']['n_queries'],
                         cell['paired_cohort']['n_rooms'], cell['superior'], gate(cell),
                         native(metric, means[pairing[0]]['mean']),
                         native(metric, means[pairing[1]]['mean']),
                         '{} / {}'.format(arm['counts']['encoder'], base['counts']['encoder']),
                         known[(pairing[0], metric)]['cohort']['n_queries']])
    return rows


def target_rows(record_data, name):
    """H2: the two fixed-target cells, with both decision routes shown separately."""
    data = record_data['products'][name]
    shot = data['profile']['num_shot']
    rows = []
    for pairing in data['profile']['pairings']:
        for metric in data['profile']['metrics']['primary']:
            cell = next(item for item in data['cells'] if item['pairing'] == list(pairing)
                        and item['metric'] == metric)
            own = cell['baseline_own']
            rows.append(['K = {}'.format(shot), ' vs '.join(pairing), metric,
                         native(metric, cell['target']),
                         native(metric, cell['paired_means'][pairing[0]]['mean']),
                         percent(cell['estimate']), percent(cell['companion_interval']),
                         cell['superior'], percent(cell['tost_interval']), cell['equivalent'],
                         cell['reaches_target'], cell['paired_cohort']['n_queries'],
                         cell['paired_cohort']['n_rooms'], native(metric, own['mean']),
                         own['cohort']['n_queries'], gate(cell), gate(cell, 'tost')])
    return rows


def arm_rows(data):
    """The six-arm table of one K: each arm's own cohort, in the metric's own units."""
    known = points(data)
    rows = []
    for role, arm in sorted(ROLES.items(), key=lambda item: item[1]['counts']['encoder']):
        cohort = known[(role, 'EDT')]['cohort']
        rows.append([role, arm['tier'], arm['backbone'], arm['counts']['encoder'],
                     arm['counts']['full'],
                     display(known[(role, 'EDT')]['mean'] * 1000, 'EDT')] +
                    [mean_sd(known[(role, metric)]) for metric in ACOUSTIC] +
                    [cohort['n_queries'], cohort['n_rooms']])
    return rows


def yaw_rows(data):
    """The descriptive per-tier yaw block: r_k against that arm's own k = 0."""
    rows = []
    for cell in data['cells']:
        arm, metric, k = ROLES[cell['arm']], cell['metric'], cell['k']
        rows.append([cell['arm'], arm['tier'], metric, signed_degrees(k, 512),
                     percent(cell['estimate']), percent(cell['companion_interval']),
                     percent(cell['room_cluster_interval']),
                     native(metric, cell['paired_means']['0']),
                     native(metric, cell['paired_means'][str(k)]),
                     cell['paired_cohort']['n_queries'], cell['paired_cohort']['n_rooms']])
    return sorted(rows, key=lambda row: (row[1], row[0], row[2], row[3]))


def attempt_rows(attempts):
    """Throughput and memory per tier, from the probe receipts and certified attempts."""
    rows = []
    for item in attempts:
        probe = item['probe']
        rows.append([item['role'], item['tier'], item['backbone'],
                     '{} x {}'.format(probe['batch_size'], probe['accum_steps']),
                     display(probe['mean_iteration_seconds']),
                     display(probe['T_epoch'] / 3600), display(probe['T_run'] / 3600),
                     display(probe['peak_allocated_bytes'] / 2 ** 30),
                     display(probe['peak_reserved_bytes'] / 2 ** 30),
                     display(item['wall_hours']), len(item['history']),
                     probe['train_batches_per_epoch'], probe['sha256']])
    for role in ('M_simple', 'M_cyl'):
        rows.append([role, 'M', ROLES[role]['backbone'], '32 x 2'] + ['—'] * 9)
    return rows


def tables(data, head):
    """Shared presentation model: titles, headers and producer-valued cells."""
    yield ('H1 — verdicts (dominance of the curve)', ['Family', 'Metric', 'Verdict', 'Scope'],
           [['{} (K = {})'.format(name, data['products'][name]['profile']['num_shot']), metric,
             verdict, data['products'][name]['verdict_scope']]
            for name in CURVES for metric, verdict in sorted(data['products'][name]['verdicts'].items())])
    yield ('H1 — per tier and metric (adjusted intervals, family m = 6)',
           ['Family', 'Tier', 'Pairing', 'Metric', 'ρ (%)', 'Adjusted query interval (%)',
            'Room-cluster interval (%)', 'Paired queries', 'Rooms', 'Superior',
            'Convergence ratio (%)', 'Cyl paired-cohort mean', 'Base paired-cohort mean',
            'Encoder parameters (cyl / base)', 'Own-cohort queries (cyl)'],
           [row for name in CURVES for row in curve_rows(data, name)])
    yield ('H2 — fixed targets (one four-cell family per K; cohort-scoped)',
           ['Family', 'Pairing', 'Metric', 'Target (paired cohort)', 'Cyl paired-cohort mean',
            'ρ (%)', 'Superiority interval (%)', 'Superior', 'TOST interval (%)', 'Equivalent',
            'Reaches target', 'Paired queries', 'Rooms', 'Baseline own-cohort mean',
            'Baseline own-cohort queries', 'Convergence ratio (%)', 'TOST convergence ratio (%)'],
           [row for name in TARGETS for row in target_rows(data, name)])
    yield ('H2 — encoder parameter ratios (published only where both metric cells reach)',
           ['Family', 'Pairing', 'Cyl encoder', 'Base encoder', 'Ratio', 'Reduction factor', 'Scope'],
           [['{} (K = {})'.format(name, data['products'][name]['profile']['num_shot']),
             ' vs '.join(item['pairing']), item['numerator'], item['denominator'],
             display(item['ratio']), display(item['reduction_factor']), item['scope']]
            for name in TARGETS for item in data['products'][name]['parameter_ratios']])
    for name in CURVES:
        product = data['products'][name]
        yield ('Six arms at K = {} — mean ± seed SD on each arm\'s own cohort'.format(
                   product['profile']['num_shot']),
               ['Arm', 'Tier', 'Backbone', 'Encoder parameters', 'Full parameters', 'EDT (ms)',
                'EDT (s)', 'C50 (dB)', 'T60 (%)', 'Queries', 'Rooms'], arm_rows(product))
    yield ('Parameters — measured with tools/exp05_params.count_parameters',
           ['Arm', 'Tier', 'Backbone', 'dim / depth / heads / mlp', 'Encoder', 'Full',
            'Trainable', 'Non-encoder', 'Cylindrical encoder overhead (%)'],
           [[role, arm['tier'], arm['backbone'],
             ' / '.join(str(arm['config'][key]) for key in ('dim', 'depth', 'heads', 'mlp_dim'))]
            + [measured_counts()[role][key] for key in ('encoder', 'full', 'trainable', 'non_encoder')]
            + [percent((arm['counts']['encoder'] - ROLES[role.replace('_cyl', '_simple')]
                        ['counts']['encoder']) / ROLES[role.replace('_cyl', '_simple')]
                       ['counts']['encoder']) if role.endswith('_cyl') else '—']
            for role, arm in sorted(ROLES.items(), key=lambda item: item[1]['counts']['encoder'])])
    yield ('Yaw robustness at K = 8, seed 42 — descriptive, r_k against the arm\'s own k = 0',
           ['Arm', 'Tier', 'Metric', 'Yaw (degrees)', 'r_k (%)', 'Query interval (%)',
            'Room interval (%)', 'k = 0 mean', 'Rotated mean', 'Queries', 'Rooms'],
           yaw_rows(data['products'][YAW]))
    yield ('Throughput and memory — probe receipts and certified attempts',
           ['Arm', 'Tier', 'Backbone', 'Micro-batch x accumulation', 'Mean iteration (s)',
            'T_epoch (h)', 'T_run (h)', 'Peak allocated (GiB)', 'Peak reserved (GiB)',
            'Wall (h)', 'Epochs', 'Micro-batches per epoch', 'Probe receipt sha256'],
           attempt_rows(data['attempts']))
    yield ('Per-epoch test loss — the four exp_05 trainings',
           ['Arm'] + ['Epoch {}'.format(epoch) for epoch in range(1, 13)],
           [[item['role']] + [display(row['test_loss']) for row in item['history']]
            for item in data['attempts']])
    yield ('Provenance — canonical products', ['Canonical input', 'sha256', 'Profile digest',
           'Producer closure', 'Approval file', 'Approval sha256'],
           [[receipt['path'], receipt['sha256'], receipt['profile_digest'],
             receipt['producer_closure_sha256'], receipt['approved_digests']['path'],
             receipt['approved_digests']['sha256']] for receipt in data['receipts']])
    yield ('Provenance — training attempts and git HEAD',
           ['Arm', 'Attempt', 'completion.json sha256', 'Probe receipt sha256'],
           [[item['role'], item['path'], item['completion']['sha256'], item['probe']['sha256']]
            for item in data['attempts']] + [['git HEAD', head, '—', '—']])


def same_file(target, published):
    """Filesystem identity: resolve() sees through symlinks but not hardlinks.

    A destination that is a second name for an input's inode would be truncated open, so
    an existing output is compared with every protected input by device and inode.
    """
    try:
        identity = os.stat(str(target))
    except OSError:
        return False
    for item in published:
        try:
            other = os.stat(item)
        except OSError:
            continue
        if (other.st_dev, other.st_ino) == (identity.st_dev, identity.st_ino):
            return True
    return False


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, name in INPUTS:
        parser.add_argument(flag, required=True, help='canonical ' + name + '.json')
    parser.add_argument('--attempt', nargs='+', required=True,
                        help='the four certified exp_05 training attempts')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    sources = [getattr(args, flag[2:].replace('-', '_')) for flag, _ in INPUTS]
    if len({Path(item).resolve() for item in sources}) != len(sources):
        raise ValueError('duplicate canonical input')
    products, receipts = {}, []
    for path, (_, name) in zip(sources, INPUTS):
        data, receipt = record.load(path, name)
        record.validate(data, name)
        products[name] = data
        receipts.append(receipt)
    attempts = [record.attempt_evidence(item) for item in sorted(set(args.attempt))]
    roles = sorted(item['role'] for item in attempts)
    if roles != sorted(TRAINED):
        raise ValueError('every trained arm must be bound exactly once: ' + ', '.join(roles))
    protected = {str(Path(item).resolve()) for item in sources}
    protected |= {item + '.provenance.json' for item in protected}
    protected |= {output for receipt in receipts for output in receipt['outputs']}
    protected |= {str(Path(receipt['approved_digests']['path']).resolve()) for receipt in receipts}
    for item in attempts:
        protected |= {item['completion']['path'], item['probe']['path']}
    if str(Path(args.out).resolve()) in protected or same_file(args.out, protected):
        raise ValueError('output overlaps canonical input')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return args, dict(products=products, receipts=receipts, attempts=attempts), head


def main(argv=None):
    args, data, head = arguments(argv)
    lines = ['generated by make_results_md.py from ' +
             ', '.join(receipt['sha256'] for receipt in data['receipts']) + ' at ' + head,
             ROUNDING, SCOPE, '']
    for title, headers, rows in tables(data, head):
        lines += ['## ' + title, '', '| ' + ' | '.join(headers) + ' |',
                  '| ' + ' | '.join('---' for _ in headers) + ' |']
        lines += ['| ' + ' | '.join(html.escape(display(value)).replace('|', '&#124;')
                                    for value in row) + ' |' for row in rows]
        lines.append('')
    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
