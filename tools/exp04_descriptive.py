"""Profile-bound descriptive diagnostics; no decision or verdict is produced."""
import argparse
import numpy as np
from tools import paired_compare as pc
from tools.results_table import METRICS


def build_diagnostics(name, directories):
    profile = pc.get_profile(name)
    if profile['mode'] != 'descriptive':
        raise ValueError('descriptive profile required')
    producer = pc.producer_identity('tools.exp04_descriptive')
    admitted = pc.admit_runs(profile, [(profile['arms'][0], profile['num_shot'], sorted(directories))],
                            producer=producer, producer_key='producer_descriptive')
    runs, cells = admitted['groups'][0], []
    if any(set(cell) - set(METRICS) for run in runs for cell in run['P'].values()):
        raise ValueError('unknown diagnostic metric')
    for source, specification in METRICS.items():
        if specification is None:
            continue
        metric, unit, scale = specification
        grid = profile['acoustic_grid'] if source in ('edt', 'c50', 't60') else profile['grid']
        for k in grid:
            baseline, angle = [np.asarray([r['P'][str(j)][source] for r in runs], dtype=float)
                               for j in (0, k)]
            mask = np.isfinite(baseline).all(axis=0) & np.isfinite(angle).all(axis=0)
            if not mask.any():
                raise ValueError('empty paired finite cohort')
            means = angle[:, mask].mean(axis=1) * scale
            estimate = pc.rho_bootstrap(angle[:, mask].mean(axis=0), baseline[:, mask].mean(axis=0),
                                       profile['n_boot'], profile['bootstrap_seeds'][0])
            cells.append(dict(metric=metric, k=k, degrees=pc.signed_degrees(k), unit=unit,
                mean=float(means.mean()), sd=float(means.std(ddof=1)) if len(runs) > 1 else None,
                per_seed={str(r['meta']['manifest_seed']): float(v) for r, v in zip(runs, means)},
                n_finite=int(mask.sum()), estimate=estimate['rho'], decision_driving=False,
                companion_interval=pc.two_sided_interval(estimate['samples'], profile['alpha'])))
    return dict(schema_version=1, profile_name=name, profile=pc.json_value(profile),
        profile_digest=pc._digest(profile), inputs=admitted['inputs'], run_flags=admitted['run_flags'],
        producer_closure_sha256=producer['sha256'], exploratory=False, decision_driving=False,
        bootstrap='descriptive query-level paired bootstrap; plain seed means; 95% interval',
        cohort='finite baseline and angle in every seed; means and sample SD use this paired cohort',
        cells=cells), admitted


def render_summary(result):
    return 'Descriptive diagnostics — ' + result['profile_name'] + '\n' + result['bootstrap'] + '\n' + '\n'.join(
        '{} k={}: mean={} {}, SD={}, r_k={}, interval={}'.format(c['metric'], c['k'], c['mean'],
        c['unit'], c['sd'], c['estimate'], c['companion_interval']) for c in result['cells']) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('GRID_SEED42', 'EPOCH9_K8'), required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--json', required=True)
    parser.add_argument('--summary', required=True)
    args = parser.parse_args(argv)
    result, admitted = build_diagnostics(args.profile, args.runs)
    pc.write_outputs(result, admitted, args.json, args.summary, renderer=render_summary)
    return result


if __name__ == '__main__':
    main()
