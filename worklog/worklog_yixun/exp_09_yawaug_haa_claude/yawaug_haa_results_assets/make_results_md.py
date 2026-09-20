"""Write yawaug_haa_results.md from the canonical exp_09 producer output only.

Input: ckpt/exp09/stats.json (+ its summary.txt, hash-bound), produced by
``tools/exp06_summarize_haa.py --experiment exp09``. Nothing is recomputed here: every number
is copied from a producer field; any missing structure is refused (no empty tables).

    python make_results_md.py --stats ckpt/exp09/stats.json [--out yawaug_haa_results.md]
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
ARM_ORDER = ('control', 'cyl', 'cyl_or', 'yawaug')
ARM_LABEL = {'control': 'A control (SimpleViT, room frame, exp_02)',
             'cyl': 'B cylindrical (room frame, exp_02)',
             'cyl_or': 'C oriented cylindrical (heading frame, exp_06)',
             'yawaug': 'E yaw-augmented SimpleViT (room frame, exp_04 init, exp_09)'}
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
    return '–' if value is None else '{:.{nd}f}'.format(value, nd=nd)


def signed(value, nd):
    return 'n/a' if value is None else '{:+.{nd}f}'.format(value, nd=nd)


def interval(item, nd):
    """A producer interval: {'lo','hi'} (summariser) or a two-element list (adjusted tails)."""
    if isinstance(item, (list, tuple)) and len(item) == 2:
        item = {'lo': item[0], 'hi': item[1]}
    if not isinstance(item, dict) or item.get('lo') is None or item.get('hi') is None:
        return 'n/a'
    return '[{}, {}]'.format(signed(item['lo'], nd), signed(item['hi'], nd))


def metric_key(name):
    return name.lower().replace('_error_s', '').replace('_error_db', '').replace('_error_pct', '')


def row_key(rows, arm, kind, room, metric):
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
    lines = ['| Arm | ' + ' | '.join('{} EDT/C50/T60'.format(ROOM_LABEL[r]) for r in ROOMS) + ' |',
             '|---|' + '---|' * len(ROOMS)]
    for arm in ARM_ORDER:
        if arm not in stats['arms']:
            refuse('arm {} missing from stats.json'.format(arm))
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


def e1_lines(stats):
    cell = stats.get('E1')
    if not isinstance(cell, dict):
        refuse('missing E1')
    nd = PRECISION[metric_key(cell['metric'])]
    cat = cell.get('category')
    nim = cell.get('non_inferior_at_margin')
    lines = ['- **E1** ({} on {} {}): diff {}, two-way 95 % {}, cohort {}/{}'.format(
        cell['contrast'], ROOM_LABEL.get(cell['room'], cell['room']), cell['metric'],
        signed(cell['diff'], nd), interval(cell.get('two_way'), nd), cell['cohort'], cell['n_test'])]
    lines.append('- **Category:** **{}**; **non-inferior at the +{} dB margin:** {}'.format(
        'withheld' if cat is None else cat, cell['margin'],
        'withheld' if nim is None else ('yes' if nim else 'no')))
    if cell.get('void_reasons'):
        lines.append('- Void: ' + '; '.join(cell['void_reasons']))
    conv = cell.get('convergence') or {}
    lines.append('- Convergence: {} ({} draws); margin verdict field: {}'.format(
        conv.get('status'), conv.get('n_boot'), cell.get('verdict')))
    return '\n'.join(lines)


def screen_table(cells, contrast_label):
    if not isinstance(cells, list) or not cells:
        refuse('missing screen cells')
    lines = ['| Room | Metric | diff {} | nominal two-way | adjusted two-way | label |'.format(contrast_label),
             '|---|---|---|---|---|---|']
    for cell in cells:
        nd = PRECISION[metric_key(cell['metric'])]
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(
            ROOM_LABEL.get(cell['room'], cell['room']), cell['metric'], signed(cell['diff'], nd),
            interval(cell.get('nominal_two_way'), nd), interval(cell.get('adjusted_two_way'), nd),
            cell['label']))
    return '\n'.join(lines)


def d_table(cells):
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
            key, '–' if entry['minus_y'] is None else entry['minus_y'], entry['n_minus_y'],
            '–' if entry['plus_y'] is None else entry['plus_y'], entry['n_plus_y']))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', default=os.path.join(REPO, 'ckpt/exp09/stats.json'))
    parser.add_argument('--out', default=OUT)
    args = parser.parse_args(argv)
    stats = load(args.stats, 'stats.json')
    if stats.get('exploratory') or stats.get('mode') != 'primary':
        refuse('stats.json is not a primary, non-exploratory record')
    if stats.get('experiment', 'exp09') != 'exp09' and stats.get('E1') is None:
        refuse('stats.json is not an exp_09 record')
    summary = stats.get('summary_path')
    if not summary or not os.path.isfile(summary) or sha(summary) != stats.get('summary_sha256'):
        refuse('summary.txt does not bind to stats.json')
    text = '\n'.join([
        '# Results — yawaug_haa (exp_09)',
        '',
        'Every number below is copied from the hash-bound producer output `ckpt/exp09/stats.json` '
        '(sha256 {}), written by `tools/exp06_summarize_haa.py --experiment exp09`. Generated by '
        '`yawaug_haa_results_assets/make_results_md.py`; nothing is recomputed here.'.format(sha(args.stats)),
        '',
        '## 1. HAA arms (DiffRIR test splits; mean ± sd over fine-tuning seeds; EDT s / C50 dB / T60 %)',
        '',
        arm_table(stats),
        '',
        'Protocol: {} bootstrap draws (adjusted tails {}), seeds {}, convergence tolerance {}, '
        'E1 margin {} dB, alpha {}, screen family {}. Phase policy: primary (unseeded Griffin-Lim, as in exp_02).'.format(
            stats['n_boot'], stats['n_boot_adjusted'], stats['bootstrap_seeds'], stats['convergence_tolerance'],
            stats['margin_db'], stats['alpha'], stats['family']),
        '',
        '## 2. E1 — pre-registered primary cell (hallway C50, yaw-augmented − control)',
        '',
        e1_lines(stats),
        '',
        '## 3. E2 — screen (yaw-augmented − control, all room × metric cells, Bonferroni-11)',
        '',
        screen_table(stats.get('E2'), 'E − A'),
        '',
        '## 4. E3 — descriptive contrasts',
        '',
        d_table(stats.get('E3')),
        '',
        '## 5. Room-frame side split (zero-shot job)',
        '',
        side_table(stats),
        '',
    ])
    with open(args.out, 'w') as handle:
        handle.write(text)
    print('wrote', args.out, len(text), 'bytes')
    return 0


if __name__ == '__main__':
    sys.exit(main())
