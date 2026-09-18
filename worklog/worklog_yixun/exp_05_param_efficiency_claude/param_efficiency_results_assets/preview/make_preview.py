"""Descriptive preview from user-supplied rounded means; no confirmatory inference.

Run: MPLCONFIGDIR=/tmp/exp05_mpl python path/to/make_preview.py
S is a low-confidence flat-capacity scenario: midpoint of each backbone's M/L
means, NOT an evaluated result or a statistically fitted prediction.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).resolve().parent
PARAMS = {'SimpleViT': [2766080, 19703296, 43709184],
          'CylindricalViT': [2777984, 19750912, 43780608]}
# Columns: EDT (ms), C50 error (dB), T60 error (%); rows: M, L.
MEANS = {
    8: {'SimpleViT': [[47.02, 1.238, 9.58], [47.13, 1.275, 9.54]],
        'CylindricalViT': [[45.06, 1.231, 9.44], [46.08, 1.243, 9.37]]},
    1: {'SimpleViT': [[68.28, 1.890, 13.38], [67.72, 1.903, 13.17]],
        'CylindricalViT': [[69.76, 1.942, 13.58], [68.11, 1.918, 13.25]]},
}
COLORS = {'SimpleViT': '#2878B5', 'CylindricalViT': '#D86624'}
MARKERS = {'SimpleViT': 'o', 'CylindricalViT': 's'}
METRICS = ['EDT error (ms)', 'C50 error (dB)', 'T60 error (%)']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def draw(include_forecast):
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.3))
    fig.subplots_adjust(left=.075, right=.98, bottom=.17, top=.81,
                        hspace=.55, wspace=.28)
    fig.suptitle('xRIR | Parameter efficiency', fontsize=21, fontweight='bold', y=.975)
    subtitle = ('M/L reported means + hypothetical S forecast' if include_forecast
                else 'M/L reported means | S evaluation pending')
    fig.text(.5, .925, subtitle, ha='center', color='#555555', fontsize=12)
    handles = [Line2D([], [], color=COLORS[b], marker=MARKERS[b], label=b)
               for b in PARAMS]
    if include_forecast:
        handles.append(Line2D([], [], color='#666666', marker='o',
                              markerfacecolor='white', linestyle='--',
                              label='S forecast (not measured)'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .905),
               ncol=len(handles), frameon=False)
    for row, k in enumerate([8, 1]):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            ax.set_title(f'K = {k}  |  {metric}  ↓', loc='left', fontsize=12, pad=12)
            for b in PARAMS:
                x = np.array(PARAMS[b]) / 1e6
                values = np.array(MEANS[k][b])[:, col]
                ax.plot(x[1:], values, color=COLORS[b], marker=MARKERS[b],
                        linewidth=1.8, markersize=7, zorder=3)
                if include_forecast:
                    forecast = values.mean()
                    ax.plot(x[:2], [forecast, values[0]], color=COLORS[b],
                            linestyle='--', alpha=.65, linewidth=1.3)
                    ax.plot(x[0], forecast, color=COLORS[b], marker=MARKERS[b],
                            markerfacecolor='white', markeredgewidth=1.8,
                            markersize=8, zorder=4)
                for idx, val in enumerate(values):
                    fmt = '.3f' if col == 1 else '.2f'
                    other = 'CylindricalViT' if b == 'SimpleViT' else 'SimpleViT'
                    above = val >= MEANS[k][other][idx][col]
                    ax.annotate(format(val, fmt), (x[idx+1], val),
                                xytext=(0, 10 if above else -17),
                                textcoords='offset points', ha='center',
                                color=COLORS[b], fontsize=9)
            # Separate row scales make small differences legible; raw units retained.
            all_values = np.array([MEANS[k][b] for b in PARAMS])[:, :, col]
            lo, hi = all_values.min(), all_values.max()
            pad = max((hi-lo)*.55, [.3, .004, .04][col])
            ax.set_ylim(lo-pad, hi+pad)
            ax.set_xscale('log')
            ax.set_xlim(2.15, 59)
            ax.set_xticks([2.77, 19.7, 43.7], ['S\n2.77', 'M\n19.7', 'L\n43.7'])
            ax.minorticks_off()
            ax.grid(axis='y', alpha=.18)
            ax.set_xlabel('Encoder parameters (M, log scale)', fontsize=10)
            if not include_forecast:
                ax.text(2.77, (lo+hi)/2, 'Pending', ha='center',
                        color='#888888', rotation=90, fontsize=10)
    foot = ('Source: user-supplied rounded five-evaluation-seed means; M reused from exp_04, L from exp_05.\n'
            'One training run per arm. No CI available in this preview. Row scales differ; lower is better.')
    if include_forecast:
        foot += '\nS forecast = same-backbone M/L midpoint (flat-capacity assumption); no prediction interval or equivalence claim.'
    fig.text(.075, .025, foot, fontsize=9, color='#555555', linespacing=1.7)
    name = 'parameter_efficiency_with_S_forecast' if include_forecast else 'parameter_efficiency_M_L'
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=220, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    with (OUT / 'preview_values.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['K', 'backbone', 'tier', 'encoder_parameters', 'status',
                         'EDT_ms', 'C50_dB', 'T60_percent', 'source'])
        for k in [8, 1]:
            for b in PARAMS:
                values = np.array(MEANS[k][b])
                for i, tier in enumerate(['S', 'M', 'L']):
                    v = values.mean(axis=0) if i == 0 else values[i-1]
                    writer.writerow([k, b, tier, PARAMS[b][i],
                                     'hypothetical_forecast' if i == 0 else 'reported_preview_mean',
                                     *[f'{n:.6f}' for n in v],
                                     'flat-capacity M/L midpoint' if i == 0 else 'user-pasted summary'])
    draw(False)
    draw(True)
    print(f'Wrote two figures (PNG/PDF/SVG) and preview_values.csv to {OUT}')
