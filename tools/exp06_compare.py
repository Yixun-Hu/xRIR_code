"""exp_06's H3: does the oriented pipeline cost simulated accuracy at k = 0 (plan 6.3, 7)?

Three arms of five evaluation seeds each. **C** is exp_06's own
``ckpt/exp06/sim_eval/<arm>/seed<S>/`` written by ``tools.exp06_eval_launch``; **A** is
exp_04's completed control evaluations (``ckpt/yaw_aug/eval/control_k8_seed{42..46}_k0``)
and **B** is either exp_05's M-tier cylindrical evaluations or an exp_06 evaluation of
the exp_01 cylindrical checkpoint. Every run is admitted fail-closed against the
approvals and the protocol of 6.3 before a single number is read, and the statistics are
exp_04's own: ``cell_mask``, ``five_seed_mean``, ``rho_bootstrap``, ``one_sided_upper``,
``h1_verdict`` and ``convergence``, imported from ``tools.paired_compare`` and not
re-implemented.

    python tools/exp06_compare.py --runs-c <five dirs> --runs-a <five> --runs-b <five> \
        --json ckpt/exp06/h3.json --summary ckpt/exp06/h3_summary.txt
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools import exp06_approvals_api as approvals_api
from tools import paired_compare
from tools import provenance
from tools.exp04_profiles import CONTROL, CYL
from tools.summarize_yaw import load_run, rooms_from_paths

ENTRY_MODULE = 'tools.exp06_compare'
REPO = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44, 45, 46)
METRICS = ('EDT', 'C50')
NUM_SHOT = 8                       # the registered K of 6.3's simulated evaluations
MARGIN = 0.03                      # exp_04's H1 rule, inherited by section 7's H3
SUPERIORITY_ALPHA = 0.025          # one-sided 97.5 % upper bound of rho
COMPANION_ALPHA = 0.05
N_BOOT = 20000
BOOT_SEEDS = (0, 1)
CONVERGENCE_TOL = 0.10
SPLIT = {'split': 'unseen', 'n_queries': 6337}
BATCH_SIZE = 16
CONDITIONS = 'P'
YAW_COLS = [0]
OUTPUTS = ('per_sample_yaw.json', 'metrics_yaw.json')
FILES = ('eval_manifest.json', 'completion.json') + OUTPUTS
# Role -> what the run must be. C is exp_06's own arm; A and B are reused baselines whose
# checkpoints are exp_01's, hashed at admission and recorded with the result.
ROLES = {'C': {'arm': 'cyl_or', 'checkpoint': None, 'exp06': True, 'role': 'arm'},
         'A': {'arm': 'control', 'checkpoint': CONTROL, 'exp06': False, 'role': 'arm'},
         'B': {'arm': 'cyl', 'checkpoint': CYL, 'exp06': None, 'role': 'baseline'}}
CONTRASTS = (('C', 'B'), ('C', 'A'))


def _equal(actual, expected):
    return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def _is_sha256(value):
    return type(value) is str and len(value) == 64 and all(
        character in '0123456789abcdef' for character in value)


def _closure_digest(closure):
    """exp_04's own closure digest, recomputed from the reviewed blobs it recorded."""
    return paired_compare._closure_digest(closure)


def admit_run(run_dir, role, approved, split=SPLIT, check=None, inputs=None,
              roles=ROLES):
    """One evaluation run of one arm, against 6.3's protocol and section 7's admission."""
    directory = Path(run_dir).resolve()
    label = '{} {}'.format(role, directory)
    inputs = {} if inputs is None else inputs
    if check is None:
        def check(ok, name):
            if not ok:
                raise ValueError(name)

    def require(ok, name):
        check(ok, label + ': ' + name)

    def bind(path, expected=None):
        path = str(Path(path).resolve())
        actual = inputs[path] if path in inputs else provenance.sha256_file(path)
        require(expected is None or actual == expected, 'digest ' + path)
        inputs[path] = actual
        return actual

    require({item.name for item in directory.iterdir()} == set(FILES), 'run directory contents')
    payload = {name: json.loads((directory / name).read_text()) for name in FILES}
    fields, completion = payload['eval_manifest.json'], payload['completion.json']
    digest = bind(directory / 'eval_manifest.json', completion.get('eval_manifest_sha256'))
    require(completion.get('eval_manifest_sha256') == digest, 'completion manifest digest')
    bind(directory / 'completion.json')
    require(_equal(completion.get('schema_version'), 1)
            and type(completion.get('child_exit_status')) is int
            and completion['child_exit_status'] == 0, 'completion status/schema')
    require(completion.get('directory_listing') == sorted(
        name for name in FILES if name != 'completion.json'), 'completion directory_listing')
    for name in OUTPUTS:
        require(name in (completion.get('outputs') or {}), 'completion output ' + name)
        bind(directory / name, completion['outputs'][name])
    expected = {'gl_seed': fields.get('manifest_seed'), 'batch_size': BATCH_SIZE,
                'batch_canonical': True, 'tf32': False, 'conditions': CONDITIONS,
                'yaw_cols': list(YAW_COLS), 'split': split['split'],
                'split_count': split['n_queries'], 'n_samples': split['n_queries'],
                'max_samples': 0, 'confirmatory': True, 'allow_dirty_used': False}
    for key, value in sorted(expected.items()):
        require(key in fields and _equal(fields[key], value), key)
    seed = fields.get('manifest_seed')
    require(type(seed) is int and seed in SEEDS, 'manifest_seed')
    require(type(fields.get('gl_seed')) is int and fields['gl_seed'] == seed,
            'gl_seed == manifest_seed')
    checkpoint = roles[role]['checkpoint']
    actual = bind(fields['checkpoint'], fields.get('checkpoint_sha256'))
    if checkpoint is None:                      # arm C: the approved epoch_012 artifact
        pinned = (approved or {}).get('artifacts', {}).get('epoch_012', {})
        require(_is_sha256(pinned.get('sha256')), 'artifacts.epoch_012 is not approved')
        require(actual == pinned.get('sha256'), 'checkpoint is not the approved epoch_012')
        require(pinned.get('epoch') == fields.get('checkpoint_epoch'), 'checkpoint_epoch')
    else:                                       # arms A and B: exp_01's published weights
        require(actual == checkpoint['sha256'], 'checkpoint is not the exp_01 ' + role)
        require(Path(fields['checkpoint']).name == Path(checkpoint['checkpoint']).name,
                'checkpoint path')
        require(fields.get('backbone') == checkpoint['backbone'], 'backbone')
    closures = fields.get('source_closures') or {}
    evaluator = fields.get('evaluator_closure') or {}
    require(_closure_digest(evaluator) == evaluator.get('sha256'), 'evaluator closure digest')
    pinned = (approved or {}).get('code', {})
    require(evaluator.get('sha256') == pinned.get('evaluator_exp03'),
            'evaluator closure is not the pinned exp_03 one')
    declared = roles[role]['exp06']
    exp06 = ('writer_exp06' in closures) if declared is None else declared
    names = ('entrypoint', 'writer', 'writer_exp06') if exp06 else ('entrypoint', 'writer')
    require(set(closures) >= set(names), 'source_closures ' + ', '.join(names))
    for name in names:
        require(_closure_digest(closures[name]) == closures[name].get('sha256'),
                'closure digest ' + name)
    if exp06:
        require(closures['entrypoint']['sha256'] == pinned.get('eval'), 'approved eval closure')
        require(closures['writer_exp06']['sha256'] == pinned.get('eval_launch'),
                'approved eval_launch closure')
        require(_is_sha256(fields.get('registry_sha256')), 'registry_sha256')
        require(fields.get('checkpoint_role') == roles[role]['role'], 'checkpoint_role')
        require(fields.get('frame') == 'room' and fields.get('heading') is None,
                'the simulated split carries no heading')
    else:
        reused = (approved or {}).get('reused', {})
        require(closures['entrypoint']['sha256'] == reused.get('exp04_evaluator_closure'),
                'approved exp_04 evaluator closure')
        require(closures['writer']['sha256'] == reused.get('exp04_writer_closure'),
                'approved exp_04 writer closure')
    run = load_run(str(directory))
    aggregate = payload['metrics_yaw.json']
    require(_equal(run.get('meta'), aggregate.get('meta')), 'output meta agreement')
    for key in ('backbone', 'checkpoint', 'manifest_hash', 'gl_seed', 'batch_size',
                'tf32', 'manifest_seed', 'yaw_cols', 'conditions', 'n_samples'):
        require(key in fields and _equal(run['meta'].get(key), fields[key]), 'meta ' + key)
    require(run['meta'].get('eval_manifest_sha256') == digest, 'meta manifest digest')
    require(set(run.get('P') or {}) == {'0'} and 'E' not in run, 'condition P at k = 0 only')
    queries, index = run.get('query'), run.get('index')
    require(isinstance(queries, list) and len(queries) == split['n_queries']
            and len(set(queries)) == len(queries), 'query count')
    require(index == list(range(split['n_queries'])) and all(type(i) is int for i in index),
            'per-sample index is the manifest order')
    for metric in METRICS:
        values = run['P']['0'].get(metric.lower())
        require(isinstance(values, list) and len(values) == split['n_queries'],
                'metric length ' + metric)
        require(all(value is None or type(value) in (int, float) for value in values),
                'metric type ' + metric)
    bind(fields['manifest_path'], fields.get('manifest_file_sha256'))
    run['role'], run['seed'], run['manifest_hash'] = role, seed, fields.get('manifest_hash')
    return run


def admit_runs(groups, approved, exploratory=False, split=SPLIT, roles=ROLES):
    """Five seeds per arm, one manifest per seed, one query order across every arm."""
    deviations, inputs, admitted = [], {}, {}

    def check(ok, name):
        if not ok:
            deviations.append(name)

    for role in sorted(groups):
        check(role in roles, 'unknown arm role: ' + str(role))
        runs = []
        for directory in groups.get(role, ()):
            try:
                runs.append(admit_run(directory, role, approved, split, check,
                                      inputs, roles))
            except (KeyError, TypeError, ValueError, OSError, IndexError) as error:
                check(False, '{}: admission {}'.format(directory, error))
        seeds = [run['seed'] for run in runs]
        check(len(runs) == len(SEEDS) and sorted(seeds) == list(SEEDS),
              '{}: the five evaluation seeds 42-46, exactly once each'.format(role))
        admitted[role] = sorted(runs, key=lambda run: run['seed'])
    flat = [run for role in sorted(admitted) for run in admitted[role]]
    if flat:
        check(all(run['query'] == flat[0]['query'] for run in flat),
              'the arms do not share one query order')
        by_seed = {}
        for run in flat:
            by_seed.setdefault(run['seed'], set()).add(run['manifest_hash'])
        check(all(len(hashes) == 1 for hashes in by_seed.values()),
              'the arms do not share one reference manifest per seed')
    if deviations and not exploratory:
        raise ValueError('admission failed: ' + '; '.join(deviations))
    return {'groups': admitted, 'inputs': inputs, 'deviations': deviations}


# --- the statistics of section 7, all of them exp_04's ------------------------------------


def _arrays(runs, metric):
    return np.asarray([run['P']['0'][metric.lower()] for run in runs], dtype=float)


def contrast_cell(groups, x, y, metric, n_boot=N_BOOT):
    """rho = (mean(x) - mean(y)) / mean(y) on the five-seed means of the shared cohort."""
    a, b = _arrays(groups[x], metric), _arrays(groups[y], metric)
    mask, exclusions = paired_compare.cell_mask(a, e0_b=b, seeds=SEEDS)
    mean_a = paired_compare.five_seed_mean(a)[mask]
    mean_b = paired_compare.five_seed_mean(b)[mask]
    rooms = rooms_from_paths(groups[x][0]['query'])[mask]

    def compute(seed, clusters=None):
        return paired_compare.rho_bootstrap(mean_a, mean_b, n_boot=n_boot, seed=seed,
                                            clusters=clusters)

    result = compute(BOOT_SEEDS[0])
    draws = {BOOT_SEEDS[0]: result.pop('samples')}
    draws[BOOT_SEEDS[1]] = compute(BOOT_SEEDS[1])['samples']
    cluster = compute(BOOT_SEEDS[0], rooms)
    convergence = paired_compare.convergence(
        lambda seed: paired_compare.two_sided_interval(draws[seed], COMPANION_ALPHA),
        BOOT_SEEDS[0], BOOT_SEEDS[1], CONVERGENCE_TOL)
    return {'contrast': '{} - {}'.format(x, y), 'metric': metric, 'rho': result['rho'],
            'upper': paired_compare.one_sided_upper(draws[BOOT_SEEDS[0]], SUPERIORITY_ALPHA),
            'alpha': SUPERIORITY_ALPHA, 'companion_alpha': COMPANION_ALPHA,
            'companion_interval': list(paired_compare.two_sided_interval(
                draws[BOOT_SEEDS[0]], COMPANION_ALPHA)),
            'room_cluster_interval': list(paired_compare.two_sided_interval(
                cluster['samples'], COMPANION_ALPHA)),
            'n_rooms_retained': int(len(set(rooms))), 'n': int(mask.sum()),
            'exclusions': exclusions, 'n_boot': n_boot,
            'bootstrap_seeds': list(BOOT_SEEDS), 'convergence': convergence,
            'seed_means': {role: _arrays(groups[role], metric)[:, mask].mean(axis=1).tolist()
                           for role in (x, y)}}


def analyse(admitted, margin=MARGIN, n_boot=N_BOOT, exploratory=False):
    """H3 is C vs B; C vs A is the same machinery, reported descriptively."""
    groups = admitted['groups']
    result = {'schema_version': 1, 'exploratory': bool(exploratory), 'margin': margin,
              'seeds': list(SEEDS), 'metrics': list(METRICS), 'split': dict(SPLIT),
              'deviations': list(admitted['deviations']), 'inputs': admitted['inputs'],
              'cells': [], 'verdicts': {}}
    for x, y in CONTRASTS:
        if x not in groups or y not in groups:
            continue
        cells = [contrast_cell(groups, x, y, metric, n_boot) for metric in METRICS]
        result['cells'].extend(cells)
        bounds = {cell['metric']: cell['upper'] for cell in cells}
        verdict = paired_compare.h1_verdict(bounds['EDT'], bounds['C50'], margin)
        failed = [cell['metric'] for cell in cells if not cell['convergence']['passed']]
        result['verdicts']['{} - {}'.format(x, y)] = {
            'verdict': 'not converged' if failed else verdict,
            'descriptive': (x, y) != CONTRASTS[0], 'not_converged': failed,
            'upper': bounds}
    if not exploratory:
        failed = ['{} {}'.format(cell['contrast'], cell['metric']) for cell in result['cells']
                  if not cell['convergence']['passed']]
        if failed:
            result['deviations'].append('convergence gate failed: ' + ', '.join(failed))
    return result


# --- producer identity and the published outputs ------------------------------------------


def producer_identity(strict=True):
    """Bind this producer's imported closure to HEAD, as tools.paired_compare does."""
    if strict:
        return paired_compare.producer_identity(ENTRY_MODULE)
    head = provenance.git_state(REPO)['HEAD']
    records, digest = provenance.closure_record(
        provenance.source_closure(ENTRY_MODULE, REPO), head, REPO)
    return {'sha256': digest, 'files': records, 'commit': head}


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _safe(value.item())
    return value


def render(result):
    lines = ['{} H3 (margin {:+.3f}, one-sided {:.3f} upper bound of rho)'.format(
        'EXPLORATORY' if result['exploratory'] else 'CONFIRMATORY', result['margin'],
        result['cells'][0]['alpha'] if result['cells'] else SUPERIORITY_ALPHA)]
    for cell in result['cells']:
        lines.append('{:10s} {:4s} rho={:+.6g} upper={:+.6g} companion={} rooms={} n={}'.format(
            cell['contrast'], cell['metric'], cell['rho'], cell['upper'],
            ['{:+.6g}'.format(value) for value in cell['companion_interval']],
            ['{:+.6g}'.format(value) for value in cell['room_cluster_interval']], cell['n']))
    for name in sorted(result['verdicts']):
        entry = result['verdicts'][name]
        lines.append('{}: {}{}'.format(name, entry['verdict'],
                                       ' (descriptive)' if entry['descriptive'] else ''))
    lines.extend('Deviation: ' + item for item in result['deviations'])
    return '\n'.join(lines) + '\n'


def write_outputs(result, json_path, summary_path):
    """Exclusive creation; the JSON carries the summary's digest, and no timestamp."""
    text = render(result)
    for path in (json_path, summary_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(summary_path), 'x') as stream:
        stream.write(text)
    record = dict(result, summary_path=str(Path(summary_path).resolve()),
                  summary_sha256=hashlib.sha256(text.encode()).hexdigest())
    return record, provenance.write_manifest(json_path, _safe(record)), text


def build_parser():
    parser = argparse.ArgumentParser(description='exp_06 H3: simulated parity at k = 0.')
    for role in sorted(ROLES):
        parser.add_argument('--runs-' + role.lower(), nargs='+', required=(role != 'B'),
                            help='the five evaluation runs of arm ' + role)
    parser.add_argument('--approved')
    parser.add_argument('--json', required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--n-boot', type=int, default=N_BOOT)
    parser.add_argument('--exploratory', action='store_true')
    return parser


def approvals(exploratory, path=None):
    try:
        approved, receipt = approvals_api.load_approved_digests(path)
    except approvals_api.ApprovalsUnavailable as error:
        if not exploratory:
            raise
        return None, None, [str(error)]
    return approved, receipt, approvals_api.require_producer(approved, 'compare', exploratory)


def main(argv=None):
    """0 on success; a refusal exits 1 and an argparse usage error exits 2."""
    args = build_parser().parse_args(argv)
    approved, receipt, deviations = approvals(args.exploratory, args.approved)
    groups = {role: getattr(args, 'runs_' + role.lower()) for role in sorted(ROLES)
              if getattr(args, 'runs_' + role.lower())}
    admitted = admit_runs(groups, approved, args.exploratory)
    admitted['deviations'] = list(deviations) + list(admitted['deviations'])
    result = analyse(admitted, MARGIN, args.n_boot, args.exploratory)
    result['approved_digests'] = receipt
    result['producer'] = producer_identity(strict=not args.exploratory)
    record, digest, text = write_outputs(result, args.json, args.summary)
    print(text)
    print(json.dumps({'json': args.json, 'sha256': digest,
                      'summary_sha256': record['summary_sha256'],
                      'verdicts': {name: entry['verdict']
                                   for name, entry in sorted(record['verdicts'].items())}}))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit('refusing: ' + str(error))
