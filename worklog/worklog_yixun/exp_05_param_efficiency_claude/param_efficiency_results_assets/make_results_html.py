"""Offline exp_05 page: the Markdown generator's admission and tables, plus the curves.

The figures are inline SVG whose only coordinate arithmetic scales canonical values: the
two backbone curves of a product on a logarithmic parameter axis (one panel for encoder
parameters, a second for full-system parameters), each point carrying its five-seed mean,
its seed SD and its bootstrap interval, and the per-epoch test-loss trajectories of the
four exp_05 trainings.  No script, no external stylesheet, no fetched asset.
"""
import math
from pathlib import Path

from tools import exp05_record as record
from tools.exp04_record import load_asset as exp04_asset

md = record.load_asset('make_results_md')
exp04 = exp04_asset('make_results_html')
CSS, esc, table_block = exp04.CSS, exp04.esc, exp04.table_block
ROLES, CURVES, ACOUSTIC, UNITS = md.ROLES, md.CURVES, md.ACOUSTIC, md.UNITS
AXES = (('encoder', 'encoder parameters'), ('full', 'full-system parameters'))
BACKBONES = ('simple', 'cylindrical')
COLOUR = dict(simple='var(--blue)', cylindrical='var(--orange)')
TITLE = 'Parameter-efficiency curves (exp_05)'
LEFT, RIGHT, TOP, BOTTOM = 95., 810., 25., 250.


def series(data, metric, axis):
    """Both backbone curves of one product: value, seed SD and interval per tier."""
    known = {(point['arm'], point['metric']): point for point in data['curves']
             if point['k'] == data['profile']['grid'][0]}
    curves = {}
    for backbone in BACKBONES:
        items = []
        for role, arm in sorted(ROLES.items(), key=lambda item: item[1]['counts']['encoder']):
            if arm['backbone'] != backbone:
                continue
            point = known[(role, metric)]
            items.append(dict(role=role, tier=arm['tier'], x=point[axis], mean=point['mean'],
                              sd=point['sd'] or 0., interval=list(point['interval']),
                              cohort=point['cohort']['n_queries']))
        curves[backbone] = items
    return curves


def scales(curves):
    """One shared frame per panel: log parameters across, metric units up."""
    xs = [math.log10(item['x']) for items in curves.values() for item in items]
    values = [value for items in curves.values() for item in items
              for value in item['interval'] + [item['mean'] - item['sd'], item['mean'] + item['sd']]]
    low, high = min(xs), max(xs)
    bottom, top = min(values), max(values)
    pad = (high - low) * .08 or 1.
    span = (top - bottom) * .08 or abs(top) * .08 or 1.
    def x_of(value):
        return LEFT + (RIGHT - LEFT) * (math.log10(value) - low + pad) / (high - low + 2 * pad)
    def y_of(value):
        return BOTTOM - (BOTTOM - TOP) * (value - bottom + span) / (top - bottom + 2 * span)
    return x_of, y_of, (bottom, top)


def panel(data, metric, axis, label):
    """One SVG panel: two polylines, a seed-SD bar and a bootstrap interval per point."""
    curves = series(data, metric, axis)
    x_of, y_of, (bottom, top) = scales(curves)
    name = data['profile_name']
    title = '{} — {} against {} (log scale)'.format(name, metric, label)
    parts = ['<svg viewBox="0 0 850 300" role="img" aria-label="{}"><title>{}</title>'.format(
        esc(title), esc(title))]
    parts.append('<line class="axis" x1="{0}" x2="{0}" y1="{1}" y2="{2}"/>'.format(
        LEFT - 15, TOP, BOTTOM))
    parts.append('<line class="axis" x1="{}" x2="{}" y1="{}" y2="{}"/>'.format(
        LEFT - 15, RIGHT, BOTTOM + 15, BOTTOM + 15))
    for value in (bottom, top):
        parts.append('<text x="5" y="{}">{}</text>'.format(
            y_of(value) + 4, esc(md.native(metric, value))))
    parts.append('<text x="5" y="{}">{} ({})</text>'.format(TOP - 8, esc(metric), esc(UNITS[metric])))
    parts.append('<text x="{}" y="292" text-anchor="end">{}</text>'.format(RIGHT, esc(label)))
    for index, backbone in enumerate(BACKBONES):
        items = curves[backbone]
        colour = COLOUR[backbone]
        points = ' '.join('{},{}'.format(x_of(item['x']), y_of(item['mean'])) for item in items)
        parts.append('<polyline fill="none" stroke="{}" points="{}"/>'.format(colour, points))
        for item in items:
            x, y = x_of(item['x']), y_of(item['mean'])
            tooltip = ('{} ({}, {}): {} {}; seed SD {}; interval {}; {} queries'.format(
                item['role'], backbone, item['tier'], md.native(metric, item['mean']),
                UNITS[metric], md.native(metric, item['sd']),
                md.native(metric, item['interval']), item['cohort']))
            parts.append('<g tabindex="0" class="mark" fill="{0}" stroke="{0}"><title>{1}</title>'
                         .format(colour, esc(tooltip)))
            parts.append('<line x1="{0}" x2="{0}" y1="{1}" y2="{2}"/>'.format(
                x, y_of(item['interval'][0]), y_of(item['interval'][1])))
            for bound in item['interval']:
                parts.append('<line x1="{}" x2="{}" y1="{}" y2="{}"/>'.format(
                    x - 5, x + 5, y_of(bound), y_of(bound)))
            parts.append('<line x1="{0}" x2="{0}" y1="{1}" y2="{2}" stroke-width="4"/>'.format(
                x, y_of(item['mean'] - item['sd']), y_of(item['mean'] + item['sd'])))
            parts.append('<circle cx="{}" cy="{}" r="4"/></g>'.format(x, y))
            if index == 0:
                parts.append('<text x="{}" y="272" text-anchor="middle">{} {}</text>'.format(
                    x, esc(item['tier']), esc(format(item['x'] / 1e6, '.2f') + 'M')))
        parts.append('<text x="{}" y="{}" fill="{}">{}</text>'.format(
            LEFT, TOP + 14 * index, colour, esc(backbone)))
    return ''.join(parts) + '</svg>'


def curve_figure(data, metric):
    """One product and metric: the encoder panel and the full-system panel."""
    caption = ('{} — {}: point = five-seed mean on the arm\'s own cohort, thick bar = seed SD, '
               'thin bar = bootstrap interval; x is logarithmic.'.format(data['profile_name'], metric))
    return ('<figure>' + ''.join(panel(data, metric, axis, label) for axis, label in AXES)
            + '<figcaption>' + esc(caption) + '</figcaption></figure>')


def trajectories(attempts):
    """The per-epoch test loss of the four exp_05 trainings, from their bound history."""
    title = 'Per-epoch test loss (four exp_05 trainings)'
    rows = [(item['role'], [(row['epoch'], row['test_loss']) for row in item['history']])
            for item in attempts]
    values = [value for _, history in rows for _, value in history]
    epochs = [epoch for _, history in rows for epoch, _ in history]
    low, high, first, last = min(values), max(values), min(epochs), max(epochs)
    def x_of(epoch):
        return LEFT + (RIGHT - LEFT) * (epoch - first) / float(last - first or 1)
    def y_of(value):
        return BOTTOM - (BOTTOM - TOP) * (value - low) / ((high - low) or 1.)
    parts = ['<figure><svg viewBox="0 0 850 300" role="img" aria-label="{}"><title>{}</title>'
             .format(esc(title), esc(title))]
    parts.append('<line class="axis" x1="{0}" x2="{0}" y1="{1}" y2="{2}"/>'.format(
        LEFT - 15, TOP, BOTTOM))
    for value in (low, high):
        parts.append('<text x="5" y="{}">{}</text>'.format(y_of(value) + 4, esc(md.display(value))))
    for index, (role, history) in enumerate(sorted(rows)):
        colour = COLOUR[ROLES[role]['backbone']]
        dash = ' stroke-dasharray="6 3"' if ROLES[role]['tier'] == 'L' else ''
        points = ' '.join('{},{}'.format(x_of(epoch), y_of(value)) for epoch, value in history)
        parts.append('<polyline fill="none" stroke="{}"{} points="{}"/>'.format(colour, dash, points))
        for epoch, value in history:
            parts.append('<g tabindex="0" class="mark" fill="{}" stroke="none"><title>{}</title>'
                         '<circle cx="{}" cy="{}" r="3"/></g>'.format(
                             colour, esc('{} epoch {}: {}'.format(role, epoch, md.display(value))),
                             x_of(epoch), y_of(value)))
        parts.append('<text x="{}" y="{}" fill="{}">{}</text>'.format(
            RIGHT - 120, TOP + 14 * index, colour, esc(role)))
    for epoch in range(first, last + 1):
        parts.append('<text x="{}" y="272" text-anchor="middle">{}</text>'.format(x_of(epoch), epoch))
    caption = 'Test loss per epoch, read from each attempt\'s bound history.jsonl.'
    return ''.join(parts) + '</svg><figcaption>' + esc(caption) + '</figcaption></figure>'


def main(argv=None):
    args, data, head = md.arguments(argv)
    identity = ('generated by make_results_html.py from ' +
                ', '.join(receipt['sha256'] for receipt in data['receipts']) + ' at ' + head)
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>' + esc(TITLE) + '</title><style>' + CSS + '</style><body><main>',
             '<p class="identity">' + esc(identity) + '</p>',
             '<h1>' + esc(TITLE) + '</h1>',
             '<p>' + esc(md.ROUNDING) + '</p>', '<p>' + esc(md.SCOPE) + '</p>']
    for name in CURVES:
        product = data['products'][name]
        for metric in product['profile']['metrics']['primary']:
            parts.append(curve_figure(product, metric))
    parts.append(trajectories(data['attempts']))
    blocks = list(md.tables(data, head))
    parts.extend(table_block(*block) for block in blocks[:-2])
    parts += ['<footer>'] + [table_block(*block) for block in blocks[-2:]] + ['</footer>']
    parts.append('</main></body></html>')
    Path(args.out).write_text('\n'.join(parts), encoding='utf-8')


if __name__ == '__main__':
    main()
