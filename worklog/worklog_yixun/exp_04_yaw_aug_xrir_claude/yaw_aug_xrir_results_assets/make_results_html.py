"""Offline exp_04 page; shared canonical admission and tables, SVG layout only."""
import html
from pathlib import Path
from make_results_md import arguments, display, tables

CSS = '''
:root{color-scheme:light;--surface:#fcfcfb;--text:#0b0b0b;--muted:#555;--grid:#ddd;--blue:#2a78d6;--orange:#c34c20;--card:#f2f2ef}
@media(prefers-color-scheme:dark){:root{color-scheme:dark;--surface:#1a1a19;--text:#fff;--muted:#c3c2b7;--grid:#444;--blue:#3987e5;--orange:#eb885f;--card:#232321}}
body{margin:0;background:var(--surface);color:var(--text);font:14px/1.5 system-ui,sans-serif}
main{max-width:1120px;margin:auto;padding:28px}h1{font-size:26px}h2{font-size:19px;margin-top:30px}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12px}
th,td{border-bottom:1px solid var(--grid);padding:7px;text-align:left;vertical-align:top}td{min-width:65px;max-width:480px;overflow-wrap:anywhere}
th{color:var(--muted)}figure{margin:16px 0;padding:12px;background:var(--card);border-radius:8px}
svg{width:100%;height:auto}svg text{fill:var(--text);font:12px system-ui,sans-serif}
.axis{stroke:var(--muted)}.bound{stroke:var(--orange);stroke-dasharray:5 4}.mark{stroke:var(--blue);fill:var(--blue)}
.mark:focus{outline:2px solid var(--orange)}.draft{padding:14px;border:3px solid var(--orange);font-weight:bold;position:sticky;top:0;background:var(--surface)}
footer{margin-top:35px;border-top:2px solid var(--grid)}.identity,figcaption{color:var(--muted);overflow-wrap:anywhere}
'''


def esc(value):
    return html.escape(display(value), quote=True)


def table_block(title, headers, rows):
    return '<section><h2>' + esc(title) + '</h2><div class="wrap"><table><thead><tr>' + ''.join(
        '<th scope="col">' + esc(v) + '</th>' for v in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div></section>'


def strip(data, equivalence=False):
    """Coordinates scale canonical values; endpoints and labels are never inferred."""
    cells = [c for c in data['cells'] if c['decision_driving']]
    intervals = [c['companion_interval'] if equivalence else [c['estimate'], c['decision_bound']] for c in cells]
    margin = data['profile']['margin'] if equivalence else 0
    low, high = min([0, -margin] + [v for pair in intervals for v in pair]), max([0, margin] + [v for pair in intervals for v in pair])
    x = lambda v: 220 + 500 * (v - low) / (high - low or 1)
    height = 65 + 30 * len(cells)
    title = data['profile_name'] + (' — equivalence intervals (ratio)' if equivalence else ' — D_k and one-sided bound (ratio)')
    parts = ['<figure><svg viewBox="0 0 850 {}" role="img" aria-label="{}"><title>{}</title>'.format(height, esc(title), esc(title))]
    for value in ([0, -margin, margin] if equivalence else [0]):
        parts.append('<line class="{}" x1="{}" x2="{}" y1="20" y2="{}"/>'.format('axis' if value == 0 else 'bound', x(value), x(value), height - 30))
    for index, (cell, pair) in enumerate(zip(cells, intervals)):
        y = 35 + index * 30
        label = '{} / {}°'.format(cell['metric'], display(cell['degrees']))
        tooltip = '{}; estimate {}; interval/bound {}; {}'.format(label, display(cell['estimate']), display(pair), cell.get('verdict', data.get('verdict', '')))
        parts.append('<text x="10" y="{}">{}</text><g class="mark" tabindex="0"><title>{}</title>'.format(y + 4, esc(label), esc(tooltip)))
        parts.append('<line x1="{}" x2="{}" y1="{}" y2="{}" stroke-width="3"/>'.format(x(pair[0]), x(pair[1]), y, y))
        for endpoint in pair:
            parts.append('<line x1="{}" x2="{}" y1="{}" y2="{}"/>'.format(x(endpoint), x(endpoint), y - 5, y + 5))
        parts.append('<circle cx="{}" cy="{}" r="4"/></g>'.format(x(cell['estimate']), y))
    for value in (low, high):
        parts.append('<text x="{}" y="{}" text-anchor="middle">{}</text>'.format(x(value), height - 10, esc(value)))
    caption = 'Point = estimate; segment = companion interval; dashed lines = ±' + display(margin) if equivalence else 'Point = D_k; segment ends at the one-sided upper bound. Vertical line = zero.'
    return ''.join(parts) + '</svg><figcaption>' + esc(caption) + '</figcaption></figure>'


def curves(data):
    """Optional one-arm canonical cells; preserves their profile and seed labels."""
    if data['profile'].get('mode') != 'one_arm':
        return ''
    figures = []
    for metric in sorted({c['metric'] for c in data['cells']}):
        cells = sorted((c for c in data['cells'] if c['metric'] == metric), key=lambda c: c['degrees'])
        low, high = min([0] + [c['estimate'] for c in cells]), max([0] + [c['estimate'] for c in cells])
        left, right = cells[0]['degrees'], cells[-1]['degrees']
        x = lambda v: 80 + 650 * (v - left) / (right - left or 1)
        y = lambda v: 180 - 140 * (v - low) / (high - low or 1)
        title = 'Diagnostic ' + data['profile_name'] + ' — ' + metric + ' r_k (ratio)'
        parts = ['<figure><svg viewBox="0 0 850 230" role="img"><title>' + esc(title) + '</title>']
        points = ' '.join('{},{}'.format(x(c['degrees']), y(c['estimate'])) for c in cells)
        parts += ['<line class="axis" x1="80" x2="730" y1="{0}" y2="{0}"/>'.format(y(0)), '<polyline stroke="var(--blue)" fill="none" points="' + points + '"/>']
        for cell in cells:
            label = '{}°: {}'.format(display(cell['degrees']), display(cell['estimate']))
            parts += ['<circle class="mark" tabindex="0" cx="{}" cy="{}" r="4"><title>{}</title></circle>'.format(x(cell['degrees']), y(cell['estimate']), esc(label)),
                      '<text x="{}" y="210" text-anchor="middle">{}°</text>'.format(x(cell['degrees']), esc(cell['degrees']))]
        for value in (low, high):
            parts.append('<text x="5" y="{}">{}</text>'.format(y(value), esc(value)))
        figures.append(''.join(parts) + '</svg><figcaption>' + esc(title) + '</figcaption></figure>')
    return ''.join(figures)


def main(argv=None):
    args, records, head = arguments(argv)
    identity = 'generated by make_results_html.py from ' + ', '.join(r[1]['sha256'] for r in records) + ' at ' + head
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>Yaw augmentation results</title><style>' + CSS + '</style><body><main>',
             '<p class="identity">' + esc(identity) + '</p><h1>Yaw augmentation results</h1>']
    if any(r[2] for r in records):
        parts.append('<div class="draft">DRAFT — missing verdict; tests only</div>')
    for index, (data, _, _) in enumerate(records):
        parts.extend(table_block(*block) for block in tables(data))
        if data['profile_name'] in ('H2_K8', 'TOST_K8'):
            parts.append(strip(data, data['profile_name'] == 'TOST_K8'))
        if index >= 5:
            parts.append(curves(data))
    rows = [[p['path'], p['sha256'], p['profile_digest'], p['producer_closure_sha256'], p['approved_digests']] for _, p, _ in records]
    parts += ['<footer>', table_block('Provenance', ['Input JSON', 'sha256', 'Profile digest', 'Producer closure digest', 'Approval'], rows), '</footer></main></body></html>']
    Path(args.out).write_text('\n'.join(parts), encoding='utf-8')


if __name__ == '__main__':
    main()
