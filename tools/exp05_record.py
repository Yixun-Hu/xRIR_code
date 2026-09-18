"""exp_05 record helpers: load its assets by path, and admit its canonical products.

The five products of ``tools/param_curve.py`` are the only source of a published number:
:func:`load` binds one to its provenance sidecar (both outputs at their digests, the
profile digest, the approval identity and the approved producer closure) and
:func:`validate` is the structural admission the generators and the record binder share
-- complete curve/cell coverage for the profile's registered arms, metrics, grid and
pairings, a non-empty cohort everywhere, a passed convergence gate on every decision
interval, the section 5 verdict wording, and the mode's own obligations (a fixed-target
cell's TOST and parameter ratio; a yaw cell's descriptive scope).

:func:`attempt_evidence` is the other half: the throughput, memory and per-epoch
trajectory of a training attempt are not in any canonical JSON, so every file they are
read from is checked against the digest the attempt's own completion sidecar records.
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from tools.exp05_profiles import ARMS, PROFILES, get_profile, profile_digest

ASSETS = Path(__file__).resolve().parents[1] / ('worklog/worklog_yixun/'
    'exp_05_param_efficiency_claude/param_efficiency_results_assets')
# One producer writes every exp_05 product; its pin is the only one a product may claim.
PRODUCER_KEY = 'producer_param_curve'
KEYS = {name: PRODUCER_KEY for name in PROFILES}
ACOUSTIC = ('EDT', 'C50', 'T60')
UNITS = dict(EDT='s', C50='dB', T60='%')
TARGET_FIELDS = ('target', 'tost_interval', 'equivalent', 'reaches_target')
ROLES = {arm['role']: arm for arm in ARMS}
TRAINED = tuple(arm['role'] for arm in ARMS if arm['tier'] != 'M')
# The registered specification every product must have been computed under: a product
# that embeds a consistent but relaxed profile passes its own validation, not this one.
REGISTERED = {name: profile_digest(name) for name in PROFILES}


def load_asset(name):
    """Import one record asset by path; exp_03/04/07 use these same file names."""
    key = 'exp05_record_' + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ASSETS / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[key]
            raise
    return sys.modules[key]


def logical(path):
    """A path as the record spells it: absolute and normalised, never resolved.

    A certified attempt is archived behind a directory symlink once the record is
    published, so the name the approval, the manifests and the binding report all give
    it stays its name here too, and the document this evidence is written into keeps
    saying what the report says.  The bytes are read, and digest-checked, through it.
    """
    return Path(os.path.abspath(str(path)))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def refuse(ok, message):
    if not ok:
        raise ValueError(message)


def load(path, name):
    """Return the canonical data and its bound identity, or refuse it."""
    refuse(name in KEYS, 'unregistered profile: ' + str(name))
    path = logical(path)
    raw = path.read_bytes()
    data, digest = json.loads(raw), sha(raw)
    side = json.loads(Path(str(path) + '.provenance.json').read_text())
    refuse(data.get('profile_name') == name, 'wrong profile: ' + str(path))
    refuse(len(side['outputs']) == 2 and side['outputs'].get(str(path)) == digest,
           'JSON and companion output coverage: ' + str(path))
    for output, expected in side['outputs'].items():
        refuse(sha(Path(output).read_bytes()) == expected, 'output digest mismatch: ' + output)
    profile_digest = sha(json.dumps(data['profile'], sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode())
    producer = side['producer']['sha256']
    approval = side['approved_digests']
    refuse(side['profile_digest'] == profile_digest and data['profile_digest'] == profile_digest
           and side['inputs'] == data['inputs'] and side['run_flags'] == data['run_flags']
           and approval['sha256'] == data['inputs'].get(approval['path'])
           and producer == approval['pins']['closures'][KEYS[name]],
           'provenance sidecar mismatch: ' + str(path))
    refuse(profile_digest == REGISTERED[name],
           'the product was not computed under the registered profile: ' + str(path))
    refuse(data.get('exploratory', False) is False and side.get('exploratory', False) is False,
           'exploratory JSON refused: ' + str(path))
    refuse(not data.get('deviations'), 'admission deviations refused: ' + str(path))
    refuse(data.get('schema_version') == 1, 'canonical schema_version: ' + str(path))
    return data, dict(path=str(path), sha256=digest, profile_digest=profile_digest,
                      producer_closure_sha256=producer, approved_digests=approval,
                      run_flags=side['run_flags'], outputs=side['outputs'])


def wordings(profile, metric):
    """Section 5's verdict wording, scoped to the tiers this family actually tested."""
    return {'dominance supported on {} at the {} tested tiers'.format(
        metric, len(profile['pairings'])), 'partial', 'not supported'}


def check_cohort(cohort, where):
    refuse(cohort.get('n_queries') and cohort.get('n_rooms'), 'empty cohort: ' + where)
    refuse(isinstance(cohort.get('retained_sha256'), str), 'cohort identity: ' + where)


def check_interval(cell, key, where):
    value = cell.get(key)
    refuse(isinstance(value, list) and len(value) == 2 and value[0] <= value[1],
           '{}: {}'.format(key, where))


def check_curve(profile, point):
    """One plotted point: the arm's own cohort, its seed set and its registered capacity."""
    where = '{arm} {metric} k={k}'.format(**point)
    arm = ROLES[point['arm']]
    refuse(point['counts'] == dict(arm['counts']), 'parameter counts: ' + where)
    refuse(point['encoder'] == arm['counts']['encoder'] and point['full'] == arm['counts']['full'],
           'parameter counts: ' + where)
    seeds = list(profile['eval_seeds'])
    refuse(list(point['seed_labels']) == seeds, 'seed labels: ' + where)
    refuse(len(point['per_seed']) == len(seeds), 'per-seed coverage: ' + where)
    refuse(sorted(point['excluded']) == sorted(str(seed) for seed in seeds),
           'per-seed exclusions: ' + where)
    check_cohort(point['cohort'], where)
    for key in ('interval', 'room_cluster_interval'):
        check_interval(point, key, where)


def check_cell(profile, cell):
    """One comparison or yaw cell: cohort, convergence and the mode's own obligations."""
    where = '{} {}'.format(' vs '.join(cell['pairing']), cell['metric'])
    check_cohort(cell['paired_cohort'], where)
    for key in ('companion_interval', 'room_cluster_interval'):
        check_interval(cell, key, where)
    if profile['mode'] == 'yaw':
        refuse(cell.get('descriptive') is True and 'convergence' not in cell
               and cell.get('verdict_scope') == profile['verdict_scope']
               and not any(key in cell for key in TARGET_FIELDS + ('superior',)),
               'a yaw cell is descriptive only: ' + where)
        return
    expected = {'superiority', 'tost'} if profile['mode'] == 'targets' else {'superiority'}
    gates = cell.get('convergence') or {}
    refuse(set(gates) == expected, 'convergence gates: ' + where)
    for gate, diagnostic in sorted(gates.items()):
        refuse(diagnostic.get('passed') is True, 'cell {} did not converge: {}'.format(gate, where))
    refuse(isinstance(cell.get('superior'), bool), 'superiority decision: ' + where)
    if profile['mode'] != 'targets':
        refuse(not any(key in cell for key in TARGET_FIELDS), 'unexpected target: ' + where)
        return
    refuse(all(key in cell for key in TARGET_FIELDS), 'fixed-target fields: ' + where)
    check_interval(cell, 'tost_interval', where)
    refuse(cell['reaches_target'] == (cell['superior'] or cell['equivalent']),
           'reaches_target disagrees with its rule: ' + where)
    refuse(cell['baseline_own']['cohort']['n_queries'] > 0, 'empty cohort: ' + where)


def check_ratios(data):
    """A ratio is published only where both metric cells of its pairing reach the target."""
    profile = data['profile']
    reached = {tuple(pairing): [cell for cell in data['cells']
                                if cell['pairing'] == list(pairing) and cell['reaches_target']]
               for pairing in profile['pairings']}
    expected = sorted(pairing for pairing, cells in reached.items()
                      if len(cells) == len(profile['metrics']['primary']))
    ratios = data.get('parameter_ratios') or []
    refuse(sorted(tuple(item['pairing']) for item in ratios) == expected,
           'parameter ratio coverage')
    for item in ratios:
        counts = [ROLES[role]['counts']['encoder'] for role in item['pairing']]
        refuse([item['numerator'], item['denominator']] == counts
               and item['ratio'] == counts[0] / counts[1]
               and item['reduction_factor'] == counts[1] / counts[0],
               'parameter ratio: ' + ' vs '.join(item['pairing']))


def validate(data, name):
    """The canonical validation both the generators and the binder apply."""
    refuse(name in KEYS and data.get('profile_name') == name, 'unregistered profile: ' + str(name))
    profile = get_profile(name)
    metrics = list(profile['metrics']['primary']) + list(profile['metrics']['descriptive'])
    expected = {(arm['role'], metric, k) for arm in profile['arms']
                for metric in metrics for k in profile['grid']}
    actual = [(point['arm'], point['metric'], point['k']) for point in data['curves']]
    refuse(len(actual) == len(expected) and set(actual) == expected, 'curve coverage: ' + name)
    for point in data['curves']:
        check_curve(profile, point)
    if profile['mode'] == 'yaw':
        cells = {(cell['arm'], cell['metric'], cell['k']) for cell in data['cells']}
        expected = {(arm['role'], metric, k) for arm in profile['arms'] for metric in metrics
                    for k in profile['grid'] if k != 0}
        refuse(len(data['cells']) == len(expected) and cells == expected, 'cell coverage: ' + name)
        refuse('verdicts' not in data and not data.get('parameter_ratios'),
               'a descriptive yaw product carries no verdicts: ' + name)
    else:
        expected = {(tuple(pairing), metric) for pairing in profile['pairings']
                    for metric in profile['metrics']['primary']}
        actual = [(tuple(cell['pairing']), cell['metric']) for cell in data['cells']]
        refuse(len(actual) == len(expected) and set(actual) == expected, 'cell coverage: ' + name)
    for cell in data['cells']:
        check_cell(profile, cell)
    if profile['mode'] == 'curve':
        verdicts = data.get('verdicts') or {}
        refuse(set(verdicts) == set(profile['metrics']['primary']), 'verdict coverage: ' + name)
        for metric, verdict in sorted(verdicts.items()):
            refuse(verdict in wordings(profile, metric),
                   'unregistered verdict wording: {} {}'.format(metric, verdict))
        refuse(not data.get('parameter_ratios'), 'unexpected parameter ratio: ' + name)
    elif profile['mode'] == 'targets':
        refuse('verdicts' not in data, 'a fixed-target family publishes no verdict block: ' + name)
        check_ratios(data)
    return data


def attempt_evidence(directory):
    """One training attempt's timing, memory and trajectory, bound to its own completion.

    The completion sidecar hash-binds every file the attempt produced and the training
    manifest; the manifest names the probe receipt the arm's limits came from, at its
    digest.  Nothing below is read without checking those bytes, so no published number
    can come from a file that changed after the launcher certified the run.

    ``consumed`` names every file this evidence was read from, so a caller protects
    exactly what it consumed rather than a hand-kept list that can fall behind.
    """
    directory = logical(directory)
    completion_path = directory / 'completion.json'
    completion = json.loads(completion_path.read_text())
    outputs = completion['outputs']
    consumed = [str(completion_path)]

    def bound(path, expected, label):
        path = logical(path)
        raw = path.read_bytes()
        refuse(sha(raw) == expected, '{} digest: {}'.format(label, path))
        consumed.append(str(path))
        return raw

    manifest = json.loads(bound(directory / 'train_manifest.json',
                                completion['train_manifest_sha256'], 'train_manifest.json'))
    args = json.loads(bound(directory / 'args.json', outputs['args.json'], 'args.json'))
    lines = bound(directory / 'history.jsonl', outputs['history.jsonl'], 'history.jsonl').decode()
    history = [json.loads(line) for line in lines.splitlines() if line.strip()]
    binding = manifest['mutable_inputs']['probe_receipt']
    receipt = json.loads(bound(binding['path'], binding['sha256'], 'probe receipt'))
    roles = [role for role in TRAINED if ROLES[role]['backbone'] == args.get('backbone')
             and all(type(args.get('vit_' + key)) is int and args['vit_' + key] == value
                     for key, value in ROLES[role]['config'].items())]
    refuse(len(roles) == 1 and args.get('tier') == ROLES[roles[0]]['tier'],
           'the attempt is not exactly one registered arm: ' + str(directory))
    refuse(args.get('param_counts') == dict(ROLES[roles[0]]['counts']),
           'attempt parameter counts: ' + roles[0])
    return dict(role=roles[0], tier=args['tier'], backbone=args['backbone'], path=str(directory),
                counts=args['param_counts'], wall_hours=completion['wall_hours'], history=history,
                consumed=sorted(set(consumed)),
                completion=dict(path=str(completion_path), sha256=sha(completion_path.read_bytes())),
                probe=dict(binding, **{key: receipt[key] for key in (
                    'mean_iteration_seconds', 'median_iteration_seconds', 'T_epoch', 'T_run',
                    'peak_allocated_bytes', 'peak_reserved_bytes', 'batch_size', 'accum_steps',
                    'train_batches_per_epoch', 'passed', 'PROBE_NOT_CLEAN')}))
