"""Write orientation_cue_fairness_results.md from a canonical exp_11 producer output only.

Input: an exp_11 stats.json (Phase 1: ckpt/exp11/phase1/stats.json) + its summary.txt (hash-bound), produced by
``tools/exp06_summarize_haa.py --experiment exp11``. Nothing is recomputed here: every number is copied
from a producer field; any missing structure is refused (no empty tables).

    python make_results_md.py --stats ckpt/exp11/phase1/stats.json [--out orientation_cue_fairness_results.md]
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
OUT = os.path.join(HERE, '..', 'orientation_cue_fairness_results.md')
ROOMS = ('class_room', 'dampened_room', 'hallway', 'complex_room')
ROOM_LABEL = {'class_room': 'Classroom', 'dampened_room': 'Dampened', 'hallway': 'Hallway',
              'complex_room': 'Complex'}
ARM_ORDER = ('control', 'control_hf', 'yawaug', 'yawaug_hf', 'cyl', 'cyl_hf', 'cyl_or',
             'control_adapter', 'yawaug_adapter', 'simple_or', 'simple_or_yaw')
ARM_LABEL = {'control': 'A xRIR control (SimpleViT, room frame, exp_02)',
             'control_hf': 'D SimpleViT in the heading frame — implicit conditioning (exp_06)',
             'yawaug': 'E yaw-augmented SimpleViT (room frame, exp_09)',
             'yawaug_hf': 'G yaw-augmented SimpleViT in the heading frame (exp_11)',
             'cyl': 'B CylindricalViT (room frame, exp_02)',
             'cyl_hf': 'F CylindricalViT in the heading frame (exp_06)',
             'cyl_or': 'C oriented CylindricalViT + cue (heading frame, exp_06)',
             'control_adapter': 'J xRIR + explicit azimuth adapter (room frame, exp_11)',
             'yawaug_adapter': 'K yaw-augmented xRIR + explicit azimuth adapter (room frame, exp_11)',
             'simple_or': 'H xRIR + orientation cue, literal simple_oriented pretraining (heading frame, exp_11)',
             'simple_or_yaw': 'I yaw-augmented xRIR + orientation cue, literal simple_oriented pretraining (heading frame, exp_11)'}
METRIC_ORDER = ('edt', 'c50', 't60')
PRECISION = {'edt': 4, 'c50': 3, 't60': 2}


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def refuse(message):
    raise SystemExit('refusing: ' + message)


def fmt(value, nd):
    return '–' if value is None else '{:.{nd}f}'.format(value, nd=nd)


def signed(value, nd):
    return 'n/a' if value is None else '{:+.{nd}f}'.format(value, nd=nd)


def interval(item, nd):
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
    present = [arm for arm in ARM_ORDER if arm in stats['arms']]
    present += sorted(arm for arm in stats['arms'] if arm not in ARM_ORDER)
    if not present:
        refuse('no arms in stats.json')
    for arm in present:
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
            lines.append('| {} {} | {} |'.format(ARM_LABEL.get(arm, arm), kind, ' | '.join(cells)))
    return '\n'.join(lines)


def field_text(value):
    if value is None:
        return 'withheld'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    return str(value)


def decision_lines(stats):
    names = stats.get('decisions')
    if not isinstance(names, list) or not names:
        refuse('missing decisions')
    lines = []
    for name in names:
        cell = stats.get(name)
        if not isinstance(cell, dict):
            refuse('missing decision cell ' + name)
        nd = PRECISION[metric_key(cell['metric'])]
        lines.append('- **{}** ({} on {} {}): diff {}, two-way 95 % {}, cohort {}/{}, status **{}**'.format(
            name, cell['contrast'], ROOM_LABEL.get(cell['room'], cell['room']), cell['metric'],
            signed(cell['diff'], nd), interval(cell.get('two_way'), nd), cell['cohort'], cell['n_test'],
            cell['status']))
        for field in cell['fields']:
            lines.append('  - {}: **{}**'.format(field, field_text(cell.get(field))))
        if cell.get('margin') is not None:
            lines.append('  - margin: {} dB'.format(cell['margin']))
        lines.append('  - reading: {}'.format(cell['reading']))
        for reason in cell.get('void_reasons') or []:
            lines.append('  - void: ' + reason)
    return '\n'.join(lines)


def screen_tables(stats):
    screens = stats.get('screens')
    if not isinstance(screens, dict) or not screens:
        refuse('missing screens')
    blocks = []
    for name in screens:
        cells = screens[name]
        if not isinstance(cells, list) or not cells:
            refuse('empty screen family ' + name)
        lines = ['**Family {}** (Bonferroni-{} within the family; labels withheld on void/unconverged cells)'.format(
            name, stats['family']), '',
            '| Room | Metric | diff | nominal two-way | adjusted two-way | label |', '|---|---|---|---|---|---|']
        for cell in cells:
            nd = PRECISION[metric_key(cell['metric'])]
            lines.append('| {} | {} | {} | {} | {} | {} |'.format(
                ROOM_LABEL.get(cell['room'], cell['room']), cell['metric'], signed(cell['diff'], nd),
                interval(cell.get('nominal_two_way'), nd), interval(cell.get('adjusted_two_way'), nd),
                'withheld' if cell.get('label') is None else cell['label']))
        blocks.append('\n'.join(lines))
    return '\n\n'.join(blocks)


def historical_table(stats):
    rows = stats.get('historical')
    if not isinstance(rows, list) or not rows:
        refuse('missing historical rows')
    lines = ['| Row | Contrast | Metric | diff | two-way | nominal two-way | verdict / label (as recorded) | source (sha256) | not recorded |',
             '|---|---|---|---|---|---|---|---|---|']
    for row in rows:
        nd = PRECISION[metric_key(row['metric'])]
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} ({}) | {} |'.format(
            row['name'], row['contrast'], row['metric'], signed(row['diff'], nd),
            interval(row.get('two_way'), nd), interval(row.get('nominal_two_way'), nd),
            'not recorded' if row.get('verdict') is None else row['verdict'],
            row['source'], row['source_sha256'][:12], ', '.join(row.get('not_recorded') or []) or '–'))
    return '\n'.join(lines)


def external_table(stats):
    """The external reference rows (A', copied from exp_02's canonical JSON; never paired)."""
    rows = stats.get('external')
    if not rows:
        return None
    lines = []
    for row in rows:
        cells = row['cells']
        parts = []
        for room in ROOMS:
            vals = []
            for metric in METRIC_ORDER:
                key = next((k for k in cells if k.split('|')[0] == room and metric_key(k.split('|')[1]) == metric), None)
                if key is None:
                    vals.append('–' if (metric == 't60' and room == 'dampened_room') else 'n/a')
                    continue
                vals.append(per_run_text(cells[key], metric))
            parts.append(' / '.join(vals))
        lines.append('| {} — {} ({}) | {} |'.format(row['label'], row['description'], row['job_kind'], ' | '.join(parts)))
        lines.append('')
        lines.append('Source `{}` (sha256 {}); paired: {}; inference: {}.'.format(
            row['source'], row['source_sha256'][:16], 'yes' if row.get('paired') else 'no', row['inference']))
    header = ['| External row | ' + ' | '.join('{} EDT/C50/T60'.format(ROOM_LABEL[r]) for r in ROOMS) + ' |',
              '|---|' + '---|' * len(ROOMS)]
    return '\n'.join(header + lines)


def side_table(stats):
    split = stats.get('side_split')
    if not isinstance(split, dict) or not split.get('cells'):
        refuse('missing side split')
    lines = ['| Arm / room / metric | −y side (n) | +y side (n) |', '|---|---|---|']
    for key in sorted(split['cells']):
        entry = split['cells'][key]
        lines.append('| {} | {} ({}) | {} ({}) |'.format(
            key, '–' if entry['minus_y'] is None else round(entry['minus_y'], 4), entry['n_minus_y'],
            '–' if entry['plus_y'] is None else round(entry['plus_y'], 4), entry['n_plus_y']))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', default=os.path.join(REPO, 'ckpt/exp11/phase1/stats.json'))
    parser.add_argument('--out', default=OUT)
    args = parser.parse_args(argv)
    if not os.path.isfile(args.stats):
        refuse('missing stats.json: ' + args.stats)
    stats = json.load(open(args.stats))
    if stats.get('exploratory') or stats.get('mode') != 'primary':
        refuse('stats.json is not a primary, non-exploratory record')
    if stats.get('experiment') != 'exp11':
        refuse('stats.json is not an exp_11 record')
    summary = stats.get('summary_path')
    if not summary or not os.path.isfile(summary) or sha(summary) != stats.get('summary_sha256'):
        refuse('summary.txt does not bind to stats.json')
    phase = stats.get('phase', 'unknown')
    text = '\n'.join([
        '# Results — orientation_cue_fairness (exp_11), {}'.format(phase),
        '',
        'Every number below is copied from the hash-bound producer output `{}` (sha256 {}), written by '
        '`tools/exp06_summarize_haa.py --experiment exp11`. Generated by '
        '`orientation_cue_fairness_results_assets/make_results_md.py`; nothing is recomputed here.'.format(
            os.path.relpath(args.stats, REPO), sha(args.stats)),
        '',
        '## 1. HAA arms (DiffRIR test splits; mean ± sd over fine-tuning seeds; EDT s / C50 dB / T60 %)',
        '',
        arm_table(stats),
        '',
        'Protocol: {} bootstrap draws (adjusted tails {}), seeds {}, convergence tolerance {}, '
        'margin {} dB (C50 cells only), alpha {}, screen family {}. Phase policy: primary (unseeded Griffin-Lim, as in exp_02). '
        'Every decision-bearing field is withheld on void, unconverged or unavailable cells.'.format(
            stats['n_boot'], stats['n_boot_adjusted'], stats['bootstrap_seeds'], stats['convergence_tolerance'],
            stats['margin_db'], stats['alpha'], stats['family']),
        '',
        '## 2. Pre-registered statements (plan §3)',
        '',
        decision_lines(stats),
        '',
        '## 3. Screens (plan §3 S1; descriptive families)',
        '',
        screen_tables(stats),
        '',
        '## 4. Historical rows (R1; copied from the exp_06/exp_09 canonical JSONs, no new inference)',
        '',
        historical_table(stats),
        '',
        '## 5. External reference rows (not paired; copied from exp_02)',
        '',
        external_table(stats) or '(none in this phase)',
        '',
        '## 6. Room-frame side split (zero-shot job)',
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
