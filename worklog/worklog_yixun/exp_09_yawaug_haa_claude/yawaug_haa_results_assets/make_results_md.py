"""Write yawaug_haa_results.md from the canonical exp_06 producer outputs only.

Inputs (all hash-bound, producer-authored): ckpt/exp06/stats.json (+ its summary.txt),
ckpt/exp06/h3.json (+ h3.txt), ckpt/exp06/gate_g1.json. Nothing is recomputed here: every
number is copied from a producer field; any missing structure is refused (no empty tables).

    python make_results_md.py --stats ckpt/exp06/stats.json --h3 ckpt/exp06/h3.json \
        --gate ckpt/exp06/gate_g1.json [--out yawaug_haa_results.md]
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
OUT = os.path.join(HERE, '..', 'yawaug_haa_results.md')
ROOMS = ('class_room', 'dampened_room', 'hallway', 'complex_room')
ROOM_LABEL = {'class_room': 'Classroom', 'dampened_room': 'Dampened', 'hallway': 'Hallway',
              'complex_room': 'Complex'}
ARM_LABEL = {'control': 'A control (SimpleViT, room frame, exp_02)',
             'cyl': 'B cylindrical (room frame, exp_02)',
             'cyl_or': 'C oriented cylindrical (heading frame, exp_06)',
             'control_hf': 'D SimpleViT in the heading frame (exp_06)',
             'cyl_hf': 'F cylindrical in the heading frame (exp_06)'}
METRIC_ORDER = ('edt', 'c50', 't60')
METRIC_LABEL = {'edt': 'EDT (s)', 'c50': 'C50 (dB)', 't60': 'T60 (%)'}
PRECISION = {'edt': 4, 'c50': 3, 't60': 2}


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def refuse(message):
    raise SystemExit('refusing: ' + message)


def load(path, label):
    if not os.path.isfile(path):
        refuse('missing ' + label + ': ' + path)
    return json.load(open(path))


def fmt(value, nd):
    if value is None:
        return '–'
    return '{:.{nd}f}'.format(value, nd=nd)


def signed(value, nd):
    if value is None:
        return 'n/a'
    return '{:+.{nd}f}'.format(value, nd=nd)


def interval(item, nd):
    """A producer interval: {'lo','hi'} (summariser) or a two-element list (adjusted tails)."""
    if isinstance(item, (list, tuple)) and len(item) == 2:
        item = {'lo': item[0], 'hi': item[1]}
    if not isinstance(item, dict) or item.get('lo') is None or item.get('hi') is None:
        return 'n/a'
    return '[{}, {}]'.format(signed(item['lo'], nd), signed(item['hi'], nd))


def metric_key(name):
    """Producer metric keys vary in case ('C50' / 'c50'); tables use the lower-case stem."""
    return name.lower().replace('_error_s', '').replace('_error_db', '').replace('_error_pct', '')


def row_key(rows, arm, kind, room, metric):
    """Find the summariser row for one cell regardless of how the metric was spelled."""
    for key, row in rows.items():
        parts = key.split('|')
        if len(parts) == 4 and parts[0] == arm and parts[1] == kind and parts[2] == room \
                and metric_key(parts[3]) == metric:
            return row
    return None


def per_run_text(row, metric):
    """The producer's mean and standard deviation over seeds (arm_rows), never recomputed."""
    nd = PRECISION[metric]
    mean, std = row.get('mean'), row.get('std')
    if mean is None:
        return '–'
    if std is None:
        return fmt(mean, nd)
    return '{}±{}'.format(fmt(mean, nd), fmt(std, nd))


def arm_table(stats):
    rows = stats['rows']
    lines = ['| Arm | ' + ' | '.join(
        '{} {}'.format(ROOM_LABEL[r], '/'.join(METRIC_LABEL[m].split(' ')[0] for m in METRIC_ORDER))
        for r in ROOMS) + ' |', '|---|' + '---|' * len(ROOMS)]
    for arm in ('control', 'cyl', 'cyl_or', 'control_hf', 'cyl_hf'):
        if arm not in stats['arms']:
            continue
        for kind in ('zero-shot', 'fine-tuned'):
            cells = []
            for room in ROOMS:
                parts = []
                for metric in METRIC_ORDER:
                    row = row_key(rows, arm, kind, room, metric)
                    if row is None:
                        if metric == 't60' and room == 'dampened_room':
                            parts.append('–')
                            continue
                        refuse('no row for {} {} {} {}'.format(arm, kind, room, metric))
                    parts.append(per_run_text(row, metric))
                cells.append(' / '.join(parts))
            lines.append('| {} {} | {} |'.format(ARM_LABEL[arm], kind, ' | '.join(cells)))
    return '\n'.join(lines)


def decision_lines(stats):
    out = []
    for name in ('H1', 'H1b'):
        cell = stats.get(name)
        if not isinstance(cell, dict):
            refuse('missing ' + name)
        nd = PRECISION[metric_key(cell['metric'])]
        out.append('- **{}** ({} on {} {}): diff {}, two-way 95 % {}, margin {}, cohort {}/{} → **{}**{}'.format(
            name, cell['contrast'], ROOM_LABEL.get(cell['room'], cell['room']), cell['metric'],
            signed(cell['diff'], nd), interval(cell.get('two_way'), nd), cell['margin'],
            cell['cohort'], cell['n_test'], cell['verdict'],
            '' if not cell.get('void_reasons') else ' (void: ' + '; '.join(cell['void_reasons']) + ')'))
    return '\n'.join(out)


def h2_table(stats):
    cells = stats.get('H2')
    if not isinstance(cells, list) or not cells:
        refuse('missing H2 screen')
    lines = ['| Room | Metric | diff C − A | nominal two-way | adjusted two-way | label |', '|---|---|---|---|---|---|']
    for cell in cells:
        nd = PRECISION[metric_key(cell['metric'])]
        adj_text = interval(cell.get('adjusted_two_way'), nd)
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(
            ROOM_LABEL.get(cell['room'], cell['room']), cell['metric'], signed(cell['diff'], nd),
            interval(cell.get('nominal_two_way'), nd), adj_text, cell['label']))
    return '\n'.join(lines)


def d_table(stats):
    cells = stats.get('D')
    if not isinstance(cells, list) or not cells:
        refuse('missing descriptive contrasts')
    lines = ['| Contrast | Room | Metric | diff | nominal two-way |', '|---|---|---|---|---|']
    for cell in cells:
        nd = PRECISION[metric_key(cell['metric'])]
        lines.append('| {} | {} | {} | {} | {} |'.format(
            cell['contrast'], ROOM_LABEL.get(cell['room'], cell['room']), cell['metric'],
            signed(cell['diff'], nd), interval(cell.get('nominal_two_way'), nd)))
    return '\n'.join(lines)


def side_table(stats):
    split = stats.get('side_split')
    if not isinstance(split, dict) or not split.get('cells'):
        refuse('missing side split')
    lines = ['| Arm / room / metric | −y side (n) | +y side (n) |', '|---|---|---|']
    for key in sorted(split['cells']):
        entry = split['cells'][key]
        lines.append('| {} | {} ({}) | {} ({}) |'.format(
            key, entry['minus_y'] if entry['minus_y'] is not None else '–', entry['n_minus_y'],
            entry['plus_y'] if entry['plus_y'] is not None else '–', entry['n_plus_y']))
    return '\n'.join(lines)


def gate_lines(gate):
    dec = gate.get('decision') or {}
    vals = dec.get('values') or {}
    if dec.get('outcome') is None or not vals:
        refuse('gate_g1.json has no decision values')
    lines = ['| Model | mirror cosine | wrong-side weight share |', '|---|---|---|']
    for name in ('control', 'cyl', 'cyl_or'):
        v = vals.get(name)
        if not isinstance(v, dict):
            refuse('gate value missing for ' + name)
        lines.append('| {} | {:.3f} | {:.3f} |'.format(name, v['mirror_cosine'], v['opposite_side_weight_share']))
    thr = dec.get('thresholds') or {}
    return ('Outcome **{}** (legacy anchors reproduced: {}); bracket control cosine < {} and share < {}, '
            'cylindrical cosine > {} and share > {}; pass rule cyl_or cosine < {} and share < {}.\n\n{}'.format(
                dec['outcome'], gate.get('legacy_reproduction', {}).get('reproduced'),
                thr.get('control_cosine_max'), thr.get('control_share_max'), thr.get('cyl_cosine_min'),
                thr.get('cyl_share_min'), thr.get('cylor_cosine_max'), thr.get('cylor_share_max'),
                '\n'.join(lines)))


def h3_section(h3):
    """The comparer's per-metric cells (rho, one-sided upper bound, companion interval) and verdicts."""
    cells, verdicts = h3.get('cells'), h3.get('verdicts')
    if not isinstance(cells, list) or not cells or not isinstance(verdicts, dict) or not verdicts:
        refuse('h3.json has no cells/verdicts')
    lines = ['| Contrast | Metric | ρ | one-sided upper ({:g}) | two-sided companion ({:g}) | room-cluster | n queries | convergence |'.format(
        cells[0]['alpha'], cells[0]['companion_alpha']), '|---|---|---|---|---|---|---|---|']
    for cell in cells:
        ci = cell.get('companion_interval') or [None, None]
        rc = cell.get('room_cluster_interval') or [None, None]
        lines.append('| {} | {} | {:+.2%} | {:+.2%} | [{:+.2%}, {:+.2%}] | [{:+.2%}, {:+.2%}] | {} | {} ({} draws) |'.format(
            cell['contrast'], cell['metric'], cell['rho'], cell['upper'], ci[0], ci[1], rc[0], rc[1],
            cell['n'], cell['convergence']['status'], cell['n_boot']))
    lines.append('')
    for name in sorted(verdicts):
        v = verdicts[name]
        lines.append('- **{}**: {}{} (margin {:+.0%}; upper bounds {})'.format(
            name, v['verdict'], ' — descriptive' if v.get('descriptive') else ' — confirmatory H3',
            h3['margin'], ', '.join('{} {:+.2%}'.format(k, u) for k, u in sorted((v.get('upper') or {}).items()) if u is not None)))
    lines.append('')
    lines.append('H3 status: {}; complete arms: {}; deviations: {}'.format(h3.get('H3'), h3.get('complete_arms'), h3.get('deviations') or 'none'))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', default=os.path.join(REPO, 'ckpt/exp06/stats.json'))
    parser.add_argument('--h3', default=os.path.join(REPO, 'ckpt/exp06/h3.json'))
    parser.add_argument('--gate', default=os.path.join(REPO, 'ckpt/exp06/gate_g1.json'))
    parser.add_argument('--out', default=OUT)
    args = parser.parse_args(argv)
    stats = load(args.stats, 'stats.json')
    if stats.get('exploratory') or stats.get('mode') != 'primary':
        refuse('stats.json is not a primary, non-exploratory record')
    summary = stats.get('summary_path')
    if not summary or not os.path.isfile(summary) or sha(summary) != stats.get('summary_sha256'):
        refuse('summary.txt does not bind to stats.json')
    h3 = load(args.h3, 'h3.json')
    if h3.get('exploratory'):
        refuse('h3.json is exploratory')
    gate = load(args.gate, 'gate_g1.json')
    text = '\n'.join([
        '# Results — yawaug_haa (exp_09)',
        '',
        'Every number below is copied from the hash-bound producer outputs: `ckpt/exp06/stats.json` '
        '(sha256 {}), `ckpt/exp06/h3.json` (sha256 {}), `ckpt/exp06/gate_g1.json` (sha256 {}). '
        'Generated by `yawaug_haa_results_assets/make_results_md.py`; nothing is recomputed here.'.format(
            sha(args.stats), sha(args.h3), sha(args.gate)),
        '',
        '## 1. Gate G1 (hallway mirror probe, pretrained checkpoints, zero-shot)',
        '',
        gate_lines(gate),
        '',
        '## 2. HAA arms (DiffRIR test splits; mean ± sd over fine-tuning seeds; EDT s / C50 dB / T60 %)',
        '',
        arm_table(stats),
        '',
        'Protocol: {} bootstrap draws (adjusted tails {}), seeds {}, convergence tolerance {}, heading roll k = {}, '
        'H1 margin {} dB, alpha {}, H2 family {}. Phase policy: primary (unseeded Griffin-Lim, as in exp_02).'.format(
            stats['n_boot'], stats['n_boot_adjusted'], stats['bootstrap_seeds'], stats['convergence_tolerance'],
            stats['heading_k'], stats['margin_db'], stats['alpha'], stats['family']),
        '',
        '## 3. Pre-registered decisions',
        '',
        decision_lines(stats),
        '',
        '## 4. H2 screen (C − A, all room × metric cells)',
        '',
        h2_table(stats),
        '',
        '## 5. Descriptive contrasts',
        '',
        d_table(stats),
        '',
        '## 6. Room-frame side split (zero-shot job)',
        '',
        side_table(stats),
        '',
        '## 7. H3 — simulated parity at k = 0 (K = 8, unseen split, five evaluation seeds)',
        '',
        h3_section(h3),
        '',
    ])
    with open(args.out, 'w') as handle:
        handle.write(text)
    print('wrote', args.out, len(text), 'bytes')
    return 0


if __name__ == '__main__':
    sys.exit(main())
