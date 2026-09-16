"""The paper's combined seen/unseen table, rendered from canonical JSON only.

Same admission as make_results_md.py (write it to table_seen_unseen.tex).  Means are
printed to three decimals in seconds and dB and two in per cent; the five-seed sample
SDs follow the table, EDT's in milliseconds so that none of them prints as zero.  The
best mean per column and K is bold, computed from the canonical means over the three
recipe arms only -- the released checkpoint is an external reference row with unknown
training budget and is never part of that comparison -- and a tie leaves both plain.
"""
from pathlib import Path

from tools.exp07_record import load_asset

md = load_asset('make_results_md')

ACOUSTIC = ('EDT', 'C50', 'T60')
DECIMALS = {'EDT': 3, 'C50': 3, 'T60': 2}
SCALES = {'EDT': .001, 'C50': 1, 'T60': 1}  # EDT is milliseconds in the canonical JSON
# seen role, exp_04 unseen role, the paper's method name; the table's row order.
METHODS = (('seen_simple', 'control', 'xRIR'), ('seen_aug', 'aug', 'YawAug-xRIR'),
           ('seen_cyl', 'cyl', 'CylindricalViT'))
HEADER = ['\\begin{table}[t]', '\\centering', '\\begin{tabular}{lcccccc}', '\\toprule',
          '& \\multicolumn{3}{c}{Seen} & \\multicolumn{3}{c}{Unseen} \\\\',
          '\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}',
          'Method & EDT (s) & C50 (dB) & T60 (\\%) & EDT (s) & C50 (dB) & T60 (\\%) \\\\']


def number(metric, mean):
    return format(mean * SCALES[metric], '.' + str(DECIMALS[metric]) + 'f')


def best(rows, metric):
    """The unique smallest error among the recipe arms; a tie bolds nothing."""
    means = [row['metrics'][metric]['mean'] for row in rows if row is not None]
    smallest = min(means) if means else None
    return smallest if means.count(smallest) == 1 else None


def cells(row, targets):
    return [('\\textbf{' + number(metric, row['metrics'][metric]['mean']) + '}'
             if row['metrics'][metric]['mean'] == targets[metric]
             else number(metric, row['metrics'][metric]['mean'])) for metric in ACOUSTIC]


def body(record):
    seen = {(row['role'], row['num_shot']): row for row in record['table']['rows']}
    unseen = {(row['role'], row['num_shot']): row for row in record['unseen']['rows']}
    lines, notes = [], []
    for shot in (1, 8):
        targets = {side: {metric: best([table[(role, shot)] for role, _, _ in pairs], metric)
                          for metric in ACOUSTIC}
                   for side, table, pairs in (('seen', seen, METHODS),
                                              ('unseen', unseen, [(u, u, n) for _, u, n in METHODS]))}
        lines += ['\\midrule', '\\multicolumn{7}{l}{$K = ' + str(shot) + '$} \\\\']
        for seen_role, unseen_role, name in METHODS:
            rows = (seen[(seen_role, shot)], unseen[(unseen_role, shot)])
            lines.append(' & '.join([name] + cells(rows[0], targets['seen']) +
                                    cells(rows[1], targets['unseen'])) + ' \\\\')
            notes.append(note(name, shot, rows))
        row = seen[(md.REFERENCE_ROLE, shot)]
        label = 'xRIR (released seen ckpt, $K = {}$)'.format(shot)
        lines.append(' & '.join([label] + cells(row, dict.fromkeys(ACOUSTIC)) +
                                ['--'] * 3) + ' \\\\')
        notes.append(note('xRIR (released seen ckpt)', shot, (row, None)))
    return lines, notes


SD_DECIMALS = {'EDT': 3, 'C50': 4, 'T60': 3}
SD_MAX_DECIMALS = 12
SD_SCIENTIFIC = 3


def sd_number(value, decimals):
    """Never print a positive SD as zero; a genuine zero keeps the table's precision.

    Below the fixed-decimal cap even the widest fixed form rounds to zero, so such a
    value is printed in scientific notation rather than contradicting the invariant.
    """
    if value is None:
        return '--'
    if value == 0:
        return format(0.0, '.' + str(decimals) + 'f')
    places = decimals
    while places < SD_MAX_DECIMALS and float(format(value, '.' + str(places) + 'f')) == 0:
        places += 1
    text = format(value, '.' + str(places) + 'f')
    return text if float(text) != 0 else format(value, '.' + str(SD_SCIENTIFIC) + 'e')


def note(name, shot, rows):
    """One SD line per table row, labelled with its K; EDT in ms, others in table units."""
    parts = []
    for side, row in zip(('seen', 'unseen'), rows):
        if row is None:
            continue
        metrics = row['metrics']
        parts.append('{} EDT {} ms, C50 {} dB, T60 {} \\%'.format(
            side, sd_number(metrics['EDT']['sd'], SD_DECIMALS['EDT']),
            sd_number(metrics['C50']['sd'], SD_DECIMALS['C50']),
            sd_number(metrics['T60']['sd'], SD_DECIMALS['T60'])))
    return '{} ($K = {}$): {}.'.format(name, shot, '; '.join(parts))


def caption(record):
    digests = [receipt['sha256'] for receipt in record['receipts']]
    return ('\\caption{Seen and unseen protocols, identical recipe and epoch budget; each value '
            'is the mean over five evaluation seeds of that seed\'s mean over its finite '
            'queries, and the sample SDs follow the table. Bold marks the smallest mean error '
            'per column and $K$ among the three recipe arms; the released checkpoint is an '
            'external reference with unknown training budget and is not compared. Canonical '
            'JSON: seen \\texttt{' + digests[0] + '}, unseen \\texttt{' + digests[-1] + '}; the '
            'file header lists every input digest.}')


def main(argv=None):
    args, record, head = md.arguments(argv)
    lines = ['% generated by make_latex.py at ' + head]
    lines += ['% ' + receipt['path'] + ' sha256 ' + receipt['sha256']
              for receipt in record['receipts']]
    report = record['receipts'][-1]['binding_report']
    lines += ['% ' + report['path'] + ' sha256 ' + report['sha256']]
    lines += HEADER
    rows, notes = body(record)
    lines += rows + ['\\bottomrule', '\\end{tabular}', '\\\\[2pt]',
                     '{\\footnotesize Five-seed sample SDs (ddof 1), EDT in milliseconds:']
    lines += [note + ('' if index == len(notes) - 1 else ' \\\\')
              for index, note in enumerate(notes)]
    lines += ['}', caption(record), '\\label{tab:seen_unseen}', '\\end{table}']
    Path(args.out).write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
