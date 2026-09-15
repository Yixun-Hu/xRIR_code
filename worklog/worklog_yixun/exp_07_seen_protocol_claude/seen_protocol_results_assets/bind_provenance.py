"""Read execution evidence and exclusively publish the exp_07 seen-protocol binding.

Binds the three certified training attempts (completion, manifest, inventory sidecar, probe
receipt, hours ledger and the approved checkpoint) AND every other attempt each arm's
ledger lists -- aborted full runs with their abort.json, probe attempts, and the arm's
probe receipts -- so a later change to any of them changes the report.  It also binds the
released reference checkpoint, the forty evaluation runs with their seen-split bindings,
the seen alignment audit (protocol, cohort digest, passed, commit), the required
``--evidence`` artefacts (the GPU parity receipt, whose nine registered cases must all
have passed, and the released-checkpoint calibration, whose pre-registered acceptance rule
AND its five-seed operands are recomputed here from the bound runs), the four canonical producer
outputs revalidated through the generators' own checks with exact per-run input coverage,
their sidecars and companions, the rendered Markdown/HTML/LaTeX, the exp_04 inputs the
combined table reuses, the approval blob and git HEAD.  Each arm's ledger must show at
most one retry.  Run directories are read and never modified.
"""
import argparse
import json
from pathlib import Path

from tools import exp07_calibration as calibration
from tools import exp07_launcher as launcher
from tools import exp07_parity as parity
from tools import exp07_profiles as profiles
from tools import exp07_provenance as e7p
from tools import provenance as p
from tools.exp04_record import load_asset as exp04_asset
from tools.exp07_eval_launch import OUTPUTS as EVAL_OUTPUTS, SPLIT_FIELDS
from tools.exp07_profiles import (ARMS, EVAL_SEEDS, RELEASED_SHA256, get_profile,
                                  load_approved_digests)
from tools.exp07_record import load_asset
from tools.paired_compare import _closure_digest, producer_identity

binder = exp04_asset('bind_provenance')
md = load_asset('make_results_md')
ROOT, require, stamp, snapshot = binder.ROOT, binder.require, binder.stamp, binder.snapshot
check_ancestor, report_path = binder.check_ancestor, binder.report_path
TRAINING = ('train_args', 'train_manifest', 'train_completion')
RUN_FILES = ('eval_manifest.json', 'completion.json', 'metrics_yaw.json', 'per_sample_yaw.json')
RELEASED = next(arm['checkpoint'] for arm in ARMS if arm['reference'])
ROLES = tuple(arm['role'] for arm in ARMS)
NUM_SHOT = tuple(get_profile('TABLE_SEEN_V1')['num_shot'])
# The complete evaluation set the plan registers: four roles x two K x five seeds.
IDENTITIES = frozenset((role, shot, seed) for role in ROLES
                       for shot in NUM_SHOT for seed in EVAL_SEEDS)
# Every product that used a trained role must declare that arm's training evidence.
TRAINING_DEPENDENCIES = ('args.json', 'train_manifest.json', 'train_inventory.json')
EVIDENCE = ('gpu_parity', 'calibration')  # --evidence NAME=PATH, both required
CALIBRATION = profiles.CALIBRATION
CALIBRATION_METRICS = profiles.CALIBRATION_METRICS
SEEN_BATCHES = launcher.TRAIN_BATCHES['seen']


def external_log(reference, digest=None):
    """Bind a log an attempt's own records point at, wherever the launcher put it.

    The launcher deliberately keeps logs OUTSIDE the attempt directory, so hashing the
    directory does not cover them; the path comes from the attempt's own completion or
    abort receipt, and a log that is not on disk is recorded as absent, never assumed.
    """
    path = Path(reference)
    if not path.is_file():
        return dict(path=str(path), present=False)
    return dict(stamp(path, digest) if digest else stamp(path), present=True)


def terminal_state(directory, row):
    """A ledger-listed attempt ended certified or aborted; bind that state's evidence.

    Setup failure is the documented third shape of an abort: ``execute_attempt`` raises
    before the child is spawned, so ``guard.log_created`` is false, ``abort_log`` renames
    nothing (``log.aborted`` is null), no ``execution.json`` was ever written and there is
    no log on disk to hash.  Any other abort must bind a log.
    """
    completion, abort = directory / 'completion.json', directory / 'abort.json'
    if completion.is_file():  # a recovered attempt keeps its abort.json as well
        record = json.loads(completion.read_text()).get('log') or {}
        require(record.get('path'), 'a certified attempt records no log: ' + row['attempt'])
        return dict(state='certified', setup_failure=False,
                    logs=[external_log(record['path'], record.get('sha256'))])
    require(abort.is_file(),
            'the attempt has no terminal state (completion.json or abort.json): ' + row['attempt'])
    record = json.loads(abort.read_text())
    reason = record.get('reason')
    require(isinstance(reason, str) and reason, 'an aborted attempt records no reason: '
            + row['attempt'])
    log = record.get('log') or {}
    logs = [external_log(log[key]) for key in ('original', 'aborted') if log.get(key)]
    failure = not log.get('aborted') and not (directory / 'execution.json').is_file()
    require(failure or any(item['present'] for item in logs),
            'an aborted attempt that spawned a child must bind its log: ' + row['attempt'])
    return dict(state='aborted', setup_failure=failure, reason=reason, logs=logs)


def other_attempts(certified, ledger, role):
    """Bind every OTHER attempt the arm's ledger lists, file by file.

    Plan section 3 requires the complete attempt history: an aborted full run (with its
    abort.json), each probe attempt and the external logs all of them reference are bound
    here at their current bytes, so a later change to any of them changes the report and
    check_record.py fails.
    """
    root = Path(certified).parent
    records = []
    for row in sorted(ledger['attempts'], key=lambda item: item['attempt']):
        directory = (root / row['attempt']).resolve()
        if directory == Path(certified).resolve():
            continue
        require(directory.is_dir(), 'the ledger names a missing attempt: ' + row['attempt'])
        files = sorted(item for item in directory.rglob('*') if item.is_file())
        require(files, 'a bound attempt holds no files: ' + row['attempt'])
        records.append(dict(role=role, attempt=row['attempt'], mode=row['mode'],
                            hours=row['hours'], path=str(directory),
                            files=[stamp(item) for item in files],
                            **terminal_state(directory, row)))
    return records


def probe_linkage(records, receipts):
    """Every probe attempt is named by exactly one receipt, at the bytes it recorded."""
    probes = {item['path'] for item in records if item['mode'] == 'probe'}
    named = {}
    for receipt in receipts:
        bound = (json.loads(Path(receipt['path']).read_text()).get('probe_attempt') or {})
        path = str(Path(bound.get('path', 'absent')).resolve())
        require(path in probes,
                'a probe receipt names an attempt this arm does not list: ' + receipt['path'])
        require(path not in named, 'two probe receipts name one attempt: ' + path)
        for part in ('train_manifest', 'completion'):
            item = stamp(Path(path) / (part + '.json'))
            require(bound.get(part + '_sha256') == item['sha256'],
                    'probe receipt {} digest: {}'.format(part, receipt['path']))
        named[path] = receipt['path']
    missing = sorted(probes - set(named))
    require(not missing, 'a probe attempt has no receipt: ' + ', '.join(missing))
    return named


def evidence_record(bindings, runs, head):
    """--evidence NAME=PATH for the parity receipt and the released-checkpoint calibration."""
    parsed = {}
    for binding in bindings or []:
        name, separator, path = binding.partition('=')
        require(separator and path and name in EVIDENCE and name not in parsed,
                'invalid or duplicate --evidence: ' + binding)
        parsed[name] = path
    require(set(parsed) == set(EVIDENCE), 'every evidence artefact is required: ' +
            ', '.join(sorted(set(EVIDENCE) - set(parsed))))
    record = {}
    for name in EVIDENCE:
        item = stamp(parsed[name])
        require(Path(item['path']).stat().st_size > 0, 'empty evidence artefact: ' + name)
        record[name] = item
    record['calibration'] = calibration_record(record['calibration'], runs, head)
    record['gpu_parity'] = parity_record(record['gpu_parity'], head)
    return record


def read_evidence(item, label):
    """Evidence is a receipt, not a log: a file that is not canonical JSON is refused."""
    try:
        return json.loads(Path(item['path']).read_text())
    except ValueError as error:
        raise ValueError('the ' + label + ' evidence is not canonical JSON: ' + str(error))


def parity_record(item, head):
    """The deferred GPU parity: exactly the nine registered cases, all passed, at HEAD."""
    data = read_evidence(item, 'GPU parity')
    require(data.get('schema_version') == 1 and data.get('passed') is True,
            'parity receipt identity')
    tests = data.get('tests') or {}
    require(sorted(tests) == sorted(parity.TESTS), 'parity test coverage')
    refused = sorted(node for node, outcome in tests.items() if outcome != 'passed')
    require(not refused, 'parity cases did not pass: ' + ', '.join(refused))
    require(data.get('pytest_exit') == 0, 'parity pytest exit: ' + str(data.get('pytest_exit')))
    require(data.get('allow_dirty_used') is False, 'parity ran outside a clean checkout')
    require(isinstance(data.get('cuda_device'), str) and data['cuda_device'],
            'the parity receipt names no CUDA device')
    check_ancestor(data.get('git_head'), head)
    require(data.get('reviewed_commit') == data.get('git_head'),
            'parity did not run at the reviewed commit')
    files = {name: stamp(data[name]['path'], data[name]['sha256']) for name in ('log', 'junit')}
    return dict(item, tests=dict(tests), pytest_exit=data['pytest_exit'],
                git_head=data['git_head'], cuda_device=data['cuda_device'], **files)


def calibration_producer():
    """The calibration producer's identity; this gate precedes every approval pin."""
    return producer_identity('tools.exp07_calibration')


def calibration_record(item, runs, head):
    """The released row's five-seed means must meet the pre-registered acceptance rule.

    The rule is recomputed here, and so are its OPERANDS: the five-seed means and sample
    SDs are derived again from the per-sample arrays of the five released K = 8 runs this
    report binds, through the producer's own function, so a hand-written or fabricated
    summary -- finite, self-consistent, and unrelated to any evaluation -- is refused.
    """
    data = read_evidence(item, 'calibration')
    require(data.get('schema_version') == 1 and data.get('passed') is True
            and data.get('profile_name') == calibration.PROFILE_NAME
            and data.get('role') == CALIBRATION['role']
            and data.get('protocol') == CALIBRATION['protocol']
            and data.get('num_shot') == CALIBRATION['num_shot'], 'calibration identity')
    metrics = data.get('metrics') or {}
    require(sorted(metrics) == sorted(CALIBRATION_METRICS), 'calibration metrics')
    accepted = {}
    for name in CALIBRATION_METRICS:
        cell = metrics[name]
        historical = calibration.HISTORICAL[name]
        tolerance = profiles.calibration_tolerance(cell['sd'], historical)
        require(cell.get('historical') == historical, 'calibration historical: ' + name)
        require(cell.get('passed') is True and abs(cell['mean'] - historical) <= tolerance,
                'calibration acceptance: ' + name)
        accepted[name] = dict(mean=cell['mean'], sd=cell['sd'], historical=historical,
                              tolerance=tolerance)
    expected = {run['path'] for run in runs if run['role'] == CALIBRATION['role']
                and run['num_shot'] == CALIBRATION['num_shot']}
    require(len(expected) == CALIBRATION['n_seeds'], 'the five released calibration runs')
    require(set(data.get('run_flags') or {}) == expected, 'calibration run coverage')
    coverage = run_coverage([run for run in runs if run['path'] in expected])
    for path in sorted(expected):
        for name, digest in sorted(coverage[path].items()):
            require((data.get('inputs') or {}).get(name) == digest,
                    'incomplete calibration run coverage: ' + name)
    require(calibration.measure(sorted(expected)) == metrics,
            'the calibration metrics differ from the bound runs')
    require((data.get('checkpoint') or {}).get('sha256') == RELEASED_SHA256,
            'calibration checkpoint')
    sidecar = Path(item['path'] + '.provenance.json')
    require(sidecar.is_file(), 'the calibration has no provenance sidecar: ' + item['path'])
    side = json.loads(sidecar.read_text())
    require(side.get('outputs') == {item['path']: item['sha256']}, 'calibration sidecar outputs')
    require(side.get('inputs') == data.get('inputs') and
            side.get('run_flags') == data.get('run_flags'), 'calibration sidecar bindings')
    producer = calibration_producer()
    require(side['producer']['sha256'] == producer['sha256'] ==
            data.get('producer_closure_sha256'), 'calibration producer closure')
    check_ancestor(side['producer']['commit'], head)
    check_ancestor(data.get('reviewed_commit'), head)
    return dict(item, role=data['role'], num_shot=data['num_shot'], metrics=accepted,
                runs=sorted(expected), producer_closure_sha256=producer['sha256'])


def audit_record(path, head):
    """The seen alignment audit: this experiment's artefact, passed, with its cohort."""
    item = stamp(path)
    data = json.loads(Path(item['path']).read_text())
    args = data.get('args') or {}
    require(args.get('protocol') == 'seen', 'the audit is not a seen-protocol audit')
    require(data.get('passed') is True and data.get('changed_delays') == 0
            and isinstance(data.get('pairs'), int) and data['pairs'] > 0, 'the audit did not pass')
    require(args.get('loader_batches') == SEEN_BATCHES and args.get('W') == 512
            and isinstance(args.get('n_batches'), int) and args['n_batches'] > 0,
            'the audit cohort is not the seen training loader')
    digest = data.get('cohort_sha256')
    require(isinstance(digest, str) and len(digest) == 64, 'the audit has no cohort digest')
    check_ancestor(args.get('git_head'), head)
    return dict(item, protocol='seen', cohort_sha256=digest, n_batches=args['n_batches'],
                pairs=data['pairs'], git_head=args['git_head'])


def attempt_record(attempt, pins):
    """One arm's full run, the evidence its limits came from, and its hours ledger."""
    record, fields = snapshot(attempt, 'train')
    directory = Path(record['path'])
    roles = [role for role, pin in pins['checkpoints'].items()
             if (ROOT / pin['path']).resolve().parent == directory]
    require(len(roles) == 1, 'attempt is not exactly one approved arm: ' + str(directory))
    role = roles[0]
    checkpoint = stamp(ROOT / pins['checkpoints'][role]['path'], pins['checkpoints'][role]['sha256'])
    require(record['outputs'].get(Path(checkpoint['path']).name) == checkpoint,
            'approved checkpoint is not this attempt\'s output: ' + role)
    require(fields.get('protocol') == 'seen' and
            fields['effective_args'].get('protocol') == 'seen', 'not a seen training: ' + role)
    bound = fields['mutable_inputs']
    identity = fields['train_data_identity']['inventory_file']
    ledger_path = directory.parent / 'cumulative_hours.json'
    ledger = json.loads(ledger_path.read_text())
    full = [row for row in ledger['attempts'] if row['mode'] == 'full']
    require(len(full) <= 2, 'more than one retry recorded: ' + role)
    require(any(row['attempt'] == directory.name for row in full),
            'the ledger does not hold this attempt: ' + role)
    record = dict(record, role=role, checkpoint=checkpoint, ledger=stamp(ledger_path),
                  full_attempts=len(full),
                  other_attempts=other_attempts(directory, ledger, role),
                  probe_receipts=[stamp(item) for item in
                                  sorted(directory.parent.glob('_probe_*.json'))],
                  inventory=stamp(identity['path'], identity['sha256']),
                  probe_receipt=stamp(Path(fields['repo']) / bound['probe_receipt']['path'],
                                      bound['probe_receipt']['sha256']),
                  seen_split=stamp(Path(fields['repo']) / bound['seen_split']['path'],
                                   bound['seen_split']['sha256']))
    record['probe_linkage'] = probe_linkage(record['other_attempts'], record['probe_receipts'])
    return dict(record, bound=attempt_declarations(record))


def attempt_declarations(record):
    """Every artefact this attempt binds, by absolute path and bound digest."""
    items = [record[key] for key in ('manifest', 'completion', 'log', 'checkpoint', 'ledger',
                                     'inventory', 'probe_receipt', 'seen_split')]
    items += list(record['probe_receipts'])
    items += [item for item in record['outputs'].values() if item]
    for other in record['other_attempts']:
        items += list(other['files'])
        items += [item for item in other['logs'] if item['present']]
    return {item['path']: item['sha256'] for item in items if item}


def split_agreement(record, fields):
    """Both output metas must declare the split identity the evaluation manifest does.

    The completion hash-binds the manifest and both outputs, which makes them immutable,
    not consistent.  The launcher compares them before it certifies a run; this is the
    same comparison, made again over the bytes the report binds.
    """
    identity = {key: fields.get(key) for key in SPLIT_FIELDS}
    require(all(value is not None for value in identity.values()),
            'the evaluation manifest declares no split identity: ' + record['path'])
    for name in EVAL_OUTPUTS:
        meta = json.loads(Path(record['path']).joinpath(name).read_text()).get('meta') or {}
        require(all(meta.get(key) == value for key, value in identity.items()),
                'output split identity: {} in {}'.format(name, record['path']))


def run_declarations(record, fields):
    """Every artefact this run may declare, at the digest its hash-bound records give it.

    Data and source files are bound by the digests the evaluation manifest records -- the
    manifest itself is bound by the completion -- so nothing here is re-hashed off disk.
    """
    root = Path(fields['repo'])
    bound = {item['path']: item['sha256'] for item in
             [record['manifest'], record['completion'], record['log']] +
             [item for item in record['outputs'].values() if item]}
    for path_key, digest_key in (('checkpoint', 'checkpoint_sha256'),
                                 ('manifest_path', 'manifest_file_sha256')):
        bound[str((root / fields[path_key]).resolve())] = fields[digest_key]
    identity = fields['data_identity']
    if 'manifest_path' in identity:
        bound[str(Path(identity['manifest_path']).resolve())] = identity['manifest_file_sha256']
    for item in identity['inventory']:
        bound[str((Path(identity['data_root']) / item['path']).resolve())] = item['sha256']
    for closure in [fields['evaluator_closure']] + list(fields['source_closures'].values()):
        for item in closure['files']:
            bound[str((root / item['path']).resolve())] = item['working_tree_sha256']
    for item in fields['mutable_inputs'].values():
        bound[str((root / item['path']).resolve())] = item['sha256']
    return bound


def run_record(run, attempts, released):
    """One evaluation run: the seen split it used and the training it is bound to."""
    record, fields = snapshot(run, 'eval')
    require(fields.get('split') == 'seen', 'not a seen evaluation: ' + str(run))
    split_agreement(record, fields)
    split = fields['mutable_inputs']['seen_split']
    require(split['path'] == e7p.SEEN_SPLIT, 'seen_split binding path: ' + str(run))
    split = stamp(Path(fields['repo']) / split['path'], split['sha256'])
    seed, shot = fields.get('manifest_seed'), fields.get('num_shot')
    require(type(seed) is int and type(shot) is int, 'run K/seed identity: ' + str(run))
    identity = dict(seen_split=split, num_shot=shot, seed=seed,
                    bound=run_declarations(record, fields))
    checkpoint = (Path(fields['repo']) / fields['checkpoint']).resolve()
    owners = [item for item in attempts if Path(item['checkpoint']['path']) == checkpoint]
    if not owners:
        require(Path(released['path']) == checkpoint and
                fields['checkpoint_sha256'] == released['sha256'], 'unregistered checkpoint')
        require(not set(TRAINING) & set(fields['mutable_inputs']),
                'the released row has no training provenance')
        return dict(record, role='released_seen', **identity)
    owner, bound = owners[0], fields['mutable_inputs']
    require(set(TRAINING) <= set(bound), 'missing training linkage: ' + str(run))
    for key, expected in (('train_manifest', owner['manifest']),
                          ('train_completion', owner['completion'])):
        require(stamp(Path(fields['repo']) / bound[key]['path'], bound[key]['sha256']) == expected,
                'training linkage: ' + str(run))
    require(Path(bound['train_args']['path']).resolve().parent == Path(owner['path']),
            'train_args linkage: ' + str(run))
    return dict(record, role=owner['role'], **identity)


def known_digests(declarations, approval, released):
    """Every artefact this report binds, by absolute path, with the digest it was bound at.

    These maps are working evidence, not report content: one run declares its whole data
    inventory, so they are merged here and never serialized into the binding report.
    """
    known = {approval['path']: approval['sha256'], released['path']: released['sha256']}
    for item in declarations:
        known.update(item)
    return known


def producer_declarations(producer):
    """The producer's own source files; the approved closure digest pins the list."""
    require(_closure_digest(producer) == producer['sha256'], 'producer closure records')
    return {str((ROOT / item['path']).resolve()): item['working_tree_sha256']
            for item in producer['files']}


def training_dependencies(attempt):
    """The training evidence any product that used this arm must declare for itself."""
    required = dict(attempt['outputs'])
    missing = [name for name in TRAINING_DEPENDENCIES if not required.get(name)]
    require(not missing, 'the attempt has no ' + ', '.join(missing) + ': ' + attempt['role'])
    items = {name: required[name] for name in TRAINING_DEPENDENCIES}
    items['completion.json'] = attempt['completion']
    return items


def run_coverage(runs):
    """Per run directory, every file of it a producer that used the run must declare."""
    coverage = {}
    for run in runs:
        files = [run['manifest'], run['completion']] + list(run['outputs'].values())
        coverage[run['path']] = {item['path']: item['sha256'] for item in files if item}
    return coverage


def bind_results(paths, head, approval, runs, attempts, released, declarations):
    """Every canonical producer output, revalidated, with exact coverage of its inputs."""
    known = known_digests(declarations, approval, released)
    coverage = run_coverage(runs)
    by_role, trained = {}, {item['role']: item for item in attempts}
    for run in runs:
        by_role.setdefault(run['role'], set()).add(run['path'])
    records = []
    for path in sorted(str(Path(item).resolve()) for item in paths):
        name = json.loads(Path(path).read_bytes()).get('profile_name')
        require(name in md.KEYS, 'unregistered producer output: ' + path)
        data, receipt = md.load(path, name)
        md.validate(data, name)  # the same canonical validation the generators apply
        side = json.loads(Path(path + '.provenance.json').read_text())
        require(all(side['approved_digests'][key] == approval[key] for key in
                    ('path', 'sha256', 'git_blob'))
                and side['approved_digests']['pins'] == approval['blob'],
                'result approval identity or pins')
        require(side['producer']['sha256'] == approval['blob']['closures'][md.KEYS[name]],
                'result producer closure is not the approved one: ' + path)
        # The expected run set is derived from the bound identities, not from the product:
        # the table is every role, a pairing is its two arms, and both are exact.
        roles = set(ROLES) if name == 'TABLE_SEEN_V1' else set(data.get('pairing') or ())
        require(roles and roles <= set(by_role), 'unregistered product roles: ' + path)
        expected = set().union(*(by_role[role] for role in sorted(roles)))
        flags = set(side['run_flags'])
        require(flags == expected,
                'the product does not declare exactly its runs: ' + path)
        for run_path in sorted(flags):  # exact coverage: no subset of a used run's files
            for item, digest in sorted(coverage[run_path].items()):
                require(side['inputs'].get(item) == digest,
                        'incomplete run coverage: {} in {}'.format(item, path))
        declared = dict(known, **producer_declarations(side['producer']))
        for item, digest in sorted(side['inputs'].items()):
            require(item in declared,
                    'input names an artefact this report does not bind: {} in {}'.format(item, path))
            require(declared[item] == digest,
                    'input differs from the bound artefact: {} in {}'.format(item, path))
        for role in sorted(roles & set(trained)):
            for required, item in sorted(training_dependencies(trained[role]).items()):
                require(side['inputs'].get(item['path']) == item['sha256'],
                        'missing training dependency: {} of {} in {}'.format(required, role, path))
        check_ancestor(side['producer']['commit'], head)
        records.append(dict(profile=name, pairing=data.get('pairing'), sha256=receipt['sha256'],
                            producer_commit=side['producer']['commit'],
                            producer_closure_sha256=side['producer']['sha256'],
                            sidecar=stamp(path + '.provenance.json'),
                            outputs=[stamp(item, digest)
                                     for item, digest in sorted(side['outputs'].items())]))
    require([item['profile'] for item in records].count('TABLE_SEEN_V1') == 1, 'one seen table')
    require(sorted(tuple(item['pairing']) for item in records
                   if item['profile'] == 'PAIRS_SEEN_V1') == sorted(md.PAIRINGS),
            'the three registered pairings')
    return records


def collect(runs, attempt, audit, evidence, results, rendered, unseen_table, unseen_binding,
            approved=None, head=None):
    pins, identity = load_approved_digests(approved)
    require(pins['schema_version'] == 1, 'approval pins are not final')
    approval = dict(identity, blob=json.loads(Path(identity['path']).read_text()))
    require(stamp(identity['path'])['sha256'] == identity['sha256'], 'approval digest mismatch')
    released = stamp((ROOT / RELEASED).resolve(), RELEASED_SHA256)
    attempts = sorted((attempt_record(item, pins) for item in attempt), key=lambda a: a['role'])
    require(sorted(a['role'] for a in attempts) == sorted(pins['checkpoints']),
            'every approved arm must be bound exactly once')
    declarations = [item.pop('bound') for item in attempts]
    paths = sorted(str(Path(item).resolve()) for item in runs)
    require(len(paths) == len(set(paths)) and len(paths) == len(IDENTITIES),
            'the forty evaluation runs')
    records = [run_record(item, attempts, released) for item in paths]
    declarations += [item.pop('bound') for item in records]
    require({(item['role'], item['num_shot'], item['seed']) for item in records} == IDENTITIES,
            'the forty evaluation runs are the registered role/K/seed set')
    head = head or binder.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                  text=True).strip()
    require(binder.subprocess.run(['git', 'cat-file', '-e', head + '^{commit}'], cwd=ROOT,
                                  stdout=binder.subprocess.PIPE,
                                  stderr=binder.subprocess.PIPE).returncode == 0,
            'invalid binding HEAD')
    published = bind_results(results, head, approval, records, attempts, released, declarations)
    audited = audit_record(audit, head)
    evidenced = evidence_record(evidence, records, head)
    unseen, unseen_receipt = md.load_unseen(unseen_table, unseen_binding)
    cited = [item['sha256'] for item in published] + [unseen_receipt['sha256']]
    documents = []
    for path in sorted(str(Path(item).resolve()) for item in rendered):
        text = Path(path).read_text(errors='replace')
        require(all(digest in text for digest in cited),
                'a rendered document does not cite every canonical digest: ' + path)
        documents.append(stamp(path))
    require(documents, 'no rendered documents')
    return dict(schema_version=1, git_HEAD=head, results=published, runs=records,
                attempts=attempts, released_checkpoint=released, audit=audited,
                evidence=evidenced, documents=documents, approved_digests=approval,
                unseen=dict(table=stamp(unseen_table, unseen_receipt['sha256']),
                            sidecar=stamp(unseen_table + '.provenance.json'),
                            binding_report=stamp(unseen_binding,
                                                 unseen_receipt['binding_report']['sha256'])),
                inputs=dict(runs=paths, attempt=[item['path'] for item in attempts],
                            audit=str(Path(audit).resolve()), evidence=sorted(evidence),
                            approved=identity['path'],
                            unseen_table=str(Path(unseen_table).resolve()),
                            unseen_binding=str(Path(unseen_binding).resolve()),
                            rendered=sorted(str(Path(item).resolve()) for item in rendered),
                            results=sorted(str(Path(item).resolve()) for item in results)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'attempt', 'results', 'rendered'):
        parser.add_argument('--' + name, nargs='+', required=True)
    # Both documented spellings: one flag with several values, or the flag repeated.
    parser.add_argument('--evidence', action='append', nargs='+', required=True,
                        metavar='NAME=PATH')
    for name in ('audit', 'unseen-table', 'unseen-binding', 'out', 'approved'):
        parser.add_argument('--' + name, required=name != 'approved')
    args = vars(parser.parse_args(argv))
    args['evidence'] = [item for group in args['evidence'] for item in group]
    output = report_path(args.pop('out'))
    p.write_manifest(output, collect(**args))


if __name__ == '__main__':
    main()
