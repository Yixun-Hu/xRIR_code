"""PNG and PDF versions of the two capacity curves, for the paper.

The plot model is the page's: :func:`make_results_html.series` reads the canonical curve
points of an admitted CURVE product, so a figure and the page can only ever show the same
five-seed means, seed SDs and bootstrap intervals.  Nothing is recomputed here.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools import exp05_record as record

page = record.load_asset('make_results_html')
md = page.md
STYLE = dict(simple=dict(colour='#2a78d6', marker='o'),
             cylindrical=dict(colour='#c34c20', marker='s'))
# A figure is evidence, so nothing about the run that drew it may reach the file.  The PDF
# backend stamps ``CreationDate`` with the wall clock unless it is removed; ``ModDate`` is
# removed with it in case a future matplotlib predefines one.  What is left -- ``Creator``
# and ``Producer`` -- is the matplotlib version string, which is stable for an environment.
# The PNG backend drops a ``None`` metadata value, so one dict serves both formats.
METADATA = {'CreationDate': None, 'ModDate': None}


def load_curve(path):
    """Admit one canonical product and refuse anything but a registered curve family."""
    name = json.loads(Path(path).read_bytes()).get('profile_name')
    record.refuse(name in page.CURVES, 'not a registered curve family: ' + str(path))
    data, receipt = record.load(path, name)
    record.validate(data, name)
    return data, receipt


def draw(data, metric, axes):
    """Both backbone curves on both parameter axes of one product and metric."""
    for axis, (field, label) in zip(axes, page.AXES):
        curves = page.series(data, metric, field)
        for backbone, items in sorted(curves.items()):
            style = STYLE[backbone]
            x = [item['x'] for item in items]
            y = [item['mean'] for item in items]
            spread = [[item['mean'] - item['interval'][0] for item in items],
                      [item['interval'][1] - item['mean'] for item in items]]
            axis.errorbar(x, y, yerr=spread, color=style['colour'], marker=style['marker'],
                          capsize=4, elinewidth=1, linewidth=1.4, label=backbone + ' (interval)')
            axis.errorbar(x, y, yerr=[item['sd'] for item in items], color=style['colour'],
                          linestyle='none', elinewidth=4, capsize=0, alpha=.55,
                          label=backbone + ' (seed SD)')
            for item in items:
                axis.annotate(item['tier'], (item['x'], item['mean']), textcoords='offset points',
                              xytext=(6, 6), fontsize=8, color=style['colour'])
        axis.set_xscale('log')
        axis.set_xlabel(label)
        axis.set_ylabel('{} ({})'.format(metric, page.UNITS[metric]))
        axis.grid(True, which='both', alpha=.25, linewidth=.5)
    axes[0].legend(fontsize=7, loc='best')


def destination(data, metric, outdir, suffix):
    """The one concrete file a panel is written to; preflighted before anything is drawn."""
    return Path(outdir) / 'param_curve_{}_{}.{}'.format(data['profile_name'], metric, suffix)


def figure(data, metric, outdir, formats):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    try:
        draw(data, metric, axes)
        fig.suptitle('{} — {} against capacity (K = {}, five seeds, own cohorts)'.format(
            data['profile_name'], metric, data['profile']['num_shot']), fontsize=11)
        written = []
        for suffix in formats:
            path = destination(data, metric, outdir, suffix)
            fig.savefig(str(path), dpi=200, metadata=METADATA)
            written.append(path)
        return written
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curve', nargs='+', required=True, help='canonical CURVE_K*.json')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--format', nargs='+', default=['png', 'pdf'], choices=('png', 'pdf'))
    args = parser.parse_args(argv)
    sources = [record.logical(item) for item in args.curve]
    record.refuse(len({item.resolve() for item in sources}) == len(sources),
                  'duplicate canonical input')
    outdir = record.logical(args.outdir)
    record.refuse(outdir.is_dir(), 'output directory does not exist: ' + str(outdir))
    record.refuse(all(item.parent != outdir for item in sources),
                  'the output directory holds a canonical input: ' + str(outdir))
    # Admit every product, then preflight EVERY predicted file: a destination that is a
    # second name for an input -- by path, hardlink or symlink -- would truncate it open.
    curves = [load_curve(path) for path in sources]
    protected = md.protected_inputs([receipt for _, receipt in curves])
    md.refuse_overlap([destination(data, metric, outdir, suffix) for data, _ in curves
                       for metric in data['profile']['metrics']['primary']
                       for suffix in args.format], protected)
    written = []
    for data, _ in curves:
        for metric in data['profile']['metrics']['primary']:
            written.extend(figure(data, metric, outdir, args.format))
    return written


if __name__ == '__main__':
    main()
