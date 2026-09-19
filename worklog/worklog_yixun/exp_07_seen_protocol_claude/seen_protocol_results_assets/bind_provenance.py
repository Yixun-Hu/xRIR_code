"""Read execution evidence and exclusively publish the exp_07 seen-protocol binding.

Binds the three certified training attempts (completion, manifest, inventory sidecar, probe
receipt, hours ledger and the approved checkpoint) AND every other attempt each arm's
ledger lists: each must have ended certified or aborted, and the external logs its own
records name -- the launcher keeps them outside the attempt directory -- are bound too,
so a later change to any of them changes the report.  A certified attempt's log is
mandatory; the one abort without a log is the documented setup failure, in the exact
shape that produces it.  Every probe attempt that COMPLETED has exactly one receipt; a
probe whose child failed has none to have, and is published through its abort evidence.

It also binds the released reference checkpoint, the forty evaluation runs (the registered
role/K/seed set, each with its seen-split binding, training linkage and an output split
identity agreeing with its manifest), the seen alignment audit, and the two required
``--evidence`` receipts: the GPU parity receipt, whose nine registered cases must all have
passed at the reviewed commit, and the released-checkpoint calibration, whose
pre-registered acceptance rule AND its five-seed operands are recomputed here from the
bound runs, with its provenance sidecar bound at its own digest.  The four canonical
producer outputs are revalidated through the generators' own checks; each must cover
exactly its registered run set in both its run flags and its contracts, each contract
being the one the run it is filed under would produce, and declare EXACTLY its own
dependencies at the bound digests: every artefact those runs declare, everything the
contract of each trained arm it used reads, the approval and its producer closure, and
nothing else -- an arm's unused files are bound in the attempt history below, but they
are inputs of nothing.  Finally the rendered Markdown/HTML/LaTeX, the exp_04
inputs the combined table reuses, the approval blob and git HEAD.  Each arm's ledger
must show at most one retry.  Run directories are read and never modified.
"""
import argparse
import json
import re
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
record_paths = load_asset('record_paths')
identical, logical = record_paths.identical, record_paths.logical
ROOT, require = binder.ROOT, binder.require
check_ancestor, report_path = binder.check_ancestor, binder.report_path
TRAINING = ('train_args', 'train_manifest', 'train_completion')
RELEASED = next(arm['checkpoint'] for arm in ARMS if arm['reference'])
ROLES = tuple(arm['role'] for arm in ARMS)
REFERENCE_ROLE = next(arm['role'] for arm in ARMS if arm['reference'])
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


def stamp(path, *expected):
    """The inherited digest check, recorded at the logical path."""
    return dict(binder.stamp(path, *expected), path=str(logical(path)))


def snapshot(directory, kind):
    """The inherited snapshot -- listing, outputs, revalidation -- recorded logically.

    Everything it validates (a directory listing that is exactly the completion's, no
    output entry resolving outside the directory, every declared input at its recorded
    digest) is unchanged; only the names it hands back are, so a relocated run keeps the
    paths its own manifest, completion and consumers already name.  A training attempt is
    reached through the arm's `final` symlink and, once archived, through a directory
    symlink as well, so neither route is its name: the launcher wrote that name into the
    manifest, the completion hash-binds the manifest, and it must still be this very
    directory -- which is a question about inodes, not about spelling.
    """
    bound_record, fields = binder.snapshot(directory, kind)
    named = fields.get('attempt_path') if kind == 'train' else None
    directory = logical(named or directory)
    require(named is None or identical(directory, bound_record['path']),
            'the manifest does not name this attempt directory: ' + str(directory))
    log = json.loads(Path(bound_record['completion']['path']).read_text())['log']
    return dict(bound_record, path=str(directory),
                manifest=dict(bound_record['manifest'],
                              path=str(directory / (kind + '_manifest.json'))),
                completion=dict(bound_record['completion'],
                                path=str(directory / 'completion.json')),
                outputs={name: item and dict(item, path=str(directory / name))
                         for name, item in bound_record['outputs'].items()},
                log=stamp(Path(fields['repo']) / log['path'], log['sha256'])), fields


def external_log(reference, digest=None, mandatory=False):
    """Bind a log an attempt's own records point at, wherever the launcher put it.

    The launcher deliberately keeps logs OUTSIDE the attempt directory, so hashing the
    directory does not cover them; the path comes from the attempt's own completion or
    abort receipt.  A CERTIFIED attempt's log is the evidence that it ran, so it must be
    on disk at the digest that attempt recorded; only the two paths an abort receipt
    names may be absent, because the abort itself renames one of them away.
    """
    path = Path(reference)
    if mandatory:
        require(path.is_file(),
                'a certified attempt names a log that is not on disk: ' + str(path))
    elif not path.is_file():
        return dict(path=str(path), present=False)
    return dict(stamp(path, digest) if digest else stamp(path), present=True)


SETUP_FAILURE = re.compile(r'.*_ABORTED_setup_failed(_[0-9a-f]+)?\Z')


def setup_failure(directory, record, logs):
    """The one documented abort with no log to hash, in the exact shape that produces it.

    ``execute_attempt`` raises before the child is spawned: ``reason`` is still its
    initial ``setup_failed``, ``guard`` is None so ``abort_log`` renames nothing and
    ``log.aborted`` is null, ``abort_attempt`` names the directory after that reason, no
    ``execution.json`` was ever written and neither named log is on disk.  A logless
    abort that is not all of that is an abort with its evidence removed.
    """
    log = record.get('log') or {}
    return (record.get('reason') == 'setup_failed'
            and SETUP_FAILURE.match(directory.name) is not None
            and sorted(log) == ['aborted', 'original'] and log['aborted'] is None
            and isinstance(log['original'], str) and bool(log['original'])
            and not (directory / 'execution.json').is_file()
            and not any(item['present'] for item in logs))


def terminal_state(directory, row):
    """A ledger-listed attempt ended certified or aborted; bind that state's evidence."""
    completion, abort = directory / 'completion.json', directory / 'abort.json'
    if completion.is_file():  # a recovered attempt keeps its abort.json as well
        record = json.loads(completion.read_text()).get('log') or {}
        # complete_attempt always writes both; a path without a digest binds nothing.
        require(record.get('path') and record.get('sha256'),
                'a certified attempt records no log at a digest: ' + row['attempt'])
        return dict(state='certified', setup_failure=False,
                    logs=[external_log(record['path'], record['sha256'], mandatory=True)])
    require(abort.is_file(),
            'the attempt has no terminal state (completion.json or abort.json): ' + row['attempt'])
    record = json.loads(abort.read_text())
    reason = record.get('reason')
    require(isinstance(reason, str) and reason, 'an aborted attempt records no reason: '
            + row['attempt'])
    log = record.get('log') or {}
    logs = [external_log(log[key]) for key in ('original', 'aborted') if log.get(key)]
    failure = setup_failure(directory, record, logs)
    require(failure or any(item['present'] for item in logs),
            'an aborted attempt must bind its log unless it is the documented setup '
            'failure: ' + row['attempt'])
    return dict(state='aborted', setup_failure=failure, reason=reason, logs=logs)


def other_attempts(certified, ledger, role):
    """Bind every OTHER attempt the arm's ledger lists, file by file.

    Plan section 3 requires the complete attempt history: an aborted full run (with its
    abort.json), each probe attempt and the external logs all of them reference are bound
    here at their current bytes, so a later change to any of them changes the report and
    check_record.py fails.
    """
    root = logical(certified).parent
    records = []
    for row in sorted(ledger['attempts'], key=lambda item: item['attempt']):
        directory = logical(root / row['attempt'])
        if directory == logical(certified):
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
    """Every COMPLETED probe is named by exactly one receipt, at the bytes it recorded.

    ``tools.exp07_launcher.run_probe`` writes the receipt only after ``execute_attempt``
    returns, so a probe whose child failed has abort evidence and no receipt at all;
    demanding one would make the arm's real failure history unpublishable.  That attempt
    is bound through :func:`terminal_state` instead, and every receipt that does exist
    must still name a probe attempt of this arm, one each.
    """
    probes = {item['path'] for item in records if item['mode'] == 'probe'}
    completed = {item['path'] for item in records
                 if item['mode'] == 'probe' and item['state'] == 'certified'}
    named = {}
    for receipt in receipts:
        bound = (json.loads(Path(receipt['path']).read_text()).get('probe_attempt') or {})
        path = str(logical(bound.get('path', 'absent')))
        require(path in probes,
                'a probe receipt names an attempt this arm does not list: ' + receipt['path'])
        require(path not in named, 'two probe receipts name one attempt: ' + path)
        for part in ('train_manifest', 'completion'):
            item = stamp(Path(path) / (part + '.json'))
            require(bound.get(part + '_sha256') == item['sha256'],
                    'probe receipt {} digest: {}'.format(part, receipt['path']))
        named[path] = receipt['path']
    missing = sorted(completed - set(named))
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
    """The deferred GPU parity: exactly the nine registered cases, all passed, at HEAD.

    The receipt is the producer's claim; pytest's own JUnit XML is the evidence for it.
    Both are read: the XML must itself record exactly the nine registered node ids as
    passed -- no skip, no error, no extra case, and not an empty or unparseable file --
    and the receipt must say what the XML says.
    """
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
    recorded = parity.parsed_outcomes(files['junit']['path'])
    try:
        parity.check(recorded, data['pytest_exit'])
    except ValueError as error:
        raise ValueError('the parity JUnit XML does not record the nine registered passes: '
                         + str(error))
    require(recorded == dict(tests), 'the parity receipt disagrees with its JUnit XML')
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
    # The sidecar's own closure records must produce the digest it claims, as a product
    # sidecar's must; a sidecar with its records removed is not evidence of a producer.
    producer_declarations(side['producer'])
    require(side['producer']['sha256'] == producer['sha256'] ==
            data.get('producer_closure_sha256'), 'calibration producer closure')
    check_ancestor(side['producer']['commit'], head)
    check_ancestor(data.get('reviewed_commit'), head)
    return dict(item, role=data['role'], num_shot=data['num_shot'], metrics=accepted,
                runs=sorted(expected), sidecar=stamp(str(sidecar)),
                producer_closure_sha256=producer['sha256'])


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
    """One arm's full run, the evidence its limits came from, and its hours ledger.

    tools/exp07_profiles.py pins each checkpoint through the arm's `final` symlink while
    this report binds the attempt the launcher named: whether the pin and the attempt's
    own output are one file is the filesystem's question, asked by inode, and the path
    recorded is the attempt's.  A pin spelled any other way is honoured the same way.

    That inode says which FILE the pin names, never which DIRECTORY the arm promoted: a
    hardlink of the approved checkpoint under any other directory is the same file, so
    the pin's route has to be checked as well.  `final` is the launcher's promotion of
    this attempt -- a symlink naming this very directory -- and an ordinary directory of
    that name is not one, whatever it holds.
    """
    record, fields = snapshot(attempt, 'train')
    directory = Path(record['path'])
    roles = [role for role, pin in sorted(pins['checkpoints'].items())
             if identical(ROOT / pin['path'], directory / Path(pin['path']).name)]
    require(len(roles) == 1, 'attempt is not exactly one approved arm: ' + str(directory))
    role = roles[0]
    final = directory.parent / 'final'
    require(final.is_symlink() and identical(final, directory),
            '`final` does not name this attempt: ' + str(directory))
    checkpoint = stamp(directory / Path(pins['checkpoints'][role]['path']).name,
                       pins['checkpoints'][role]['sha256'])
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
    return record


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
        bound[str(logical(root / fields[path_key]))] = fields[digest_key]
    identity = fields['data_identity']
    if 'manifest_path' in identity:
        bound[str(logical(identity['manifest_path']))] = identity['manifest_file_sha256']
    for item in identity['inventory']:
        bound[str(logical(Path(identity['data_root']) / item['path']))] = item['sha256']
    for closure in [fields['evaluator_closure']] + list(fields['source_closures'].values()):
        for item in closure['files']:
            bound[str(logical(root / item['path']))] = item['working_tree_sha256']
    for item in fields['mutable_inputs'].values():
        bound[str(logical(root / item['path']))] = item['sha256']
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
    checkpoint = logical(Path(fields['repo']) / fields['checkpoint'])
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
    require(logical(bound['train_args']['path']).parent == Path(owner['path']),
            'train_args linkage: ' + str(run))
    return dict(record, role=owner['role'], **identity)


def product_dependencies(expected, roles, trained, run_bound, approval):
    """EXACTLY the inputs of a product built over exactly these runs.

    Everything each of those runs declares for itself -- its manifest, completion, both
    outputs, log, checkpoint, reference manifest, data inventory, evaluator and writer
    closures and mutable bindings -- plus the training evidence every trained arm it used
    makes it consume (:func:`training_dependencies`) and the approval blob.  Nothing
    else: neither an artefact of a run this product did not use nor a file of an arm it
    did use that no contract reads, however well its digest matches.  This map is working
    evidence, not report content: one run declares its whole data inventory, so it is
    never serialized.
    """
    dependencies = {approval['path']: approval['sha256']}
    for path in sorted(expected):
        dependencies.update(run_bound[path])
    for role in sorted(roles & set(trained)):
        for item in training_dependencies(trained[role]).values():
            dependencies[item['path']] = item['sha256']
    return dependencies


def producer_declarations(producer):
    """The producer's own source files; the approved closure digest pins the list."""
    require(_closure_digest(producer) == producer['sha256'], 'producer closure records')
    return {str(logical(ROOT / item['path'])): item['working_tree_sha256']
            for item in producer['files']}


def training_dependencies(attempt):
    """Exactly the training evidence a product that used this arm consumes.

    ``tools.exp07_table.run_contract`` reads all of it UNCONDITIONALLY for a trained
    arm -- args.json, train_manifest.json and completion.json, the inventory sidecar the
    training manifest names, the hours ledger beside the attempt and the probe receipt
    the timing limits came from -- so a product that used the arm must declare every one
    of them.  The arm's remaining files (the eleven unused epoch checkpoints,
    history.jsonl, an aborted attempt's records and logs) are bound in this report's
    attempt history and are inputs of nothing.
    """
    outputs = dict(attempt['outputs'])
    missing = [name for name in TRAINING_DEPENDENCIES if not outputs.get(name)]
    require(not missing, 'the attempt has no ' + ', '.join(missing) + ': ' + attempt['role'])
    items = {name: outputs[name] for name in TRAINING_DEPENDENCIES}
    items['completion.json'] = attempt['completion']
    items['inventory sidecar'] = attempt['inventory']
    items['cumulative_hours.json'] = attempt['ledger']
    items['probe receipt'] = attempt['probe_receipt']
    return items


def run_coverage(runs):
    """Per run directory, every file of it a producer that used the run must declare."""
    coverage = {}
    for run in runs:
        files = [run['manifest'], run['completion']] + list(run['outputs'].values())
        coverage[run['path']] = {item['path']: item['sha256'] for item in files if item}
    return coverage


def bind_results(paths, head, approval, runs, attempts, run_bound):
    """Every canonical producer output, revalidated, with exact coverage of its inputs."""
    coverage = run_coverage(runs)
    by_role, trained = {}, {item['role']: item for item in attempts}
    by_path = {run['path']: run for run in runs}
    for run in runs:
        by_role.setdefault(run['role'], set()).add(run['path'])
    records = []
    for path in sorted(str(logical(item)) for item in paths):
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
        contracts = data.get('contracts') or {}
        require(set(contracts) == expected,
                'the product does not record exactly its run contracts: ' + path)
        for run_path in sorted(expected):  # and each contract is THIS run's contract
            run, contract = by_path[run_path], contracts[run_path]
            require(contract.get('role') == run['role'] and
                    contract.get('reference') is (run['role'] == REFERENCE_ROLE) and
                    contract.get('num_shot') == run['num_shot'] and
                    contract.get('seed') == run['seed'] and
                    contract.get('eval_manifest_sha256') == run['manifest']['sha256'],
                    'the contract is not this run: {} in {}'.format(run_path, path))
        for run_path in sorted(flags):  # exact coverage: no subset of a used run's files
            for item, digest in sorted(coverage[run_path].items()):
                require(side['inputs'].get(item) == digest,
                        'incomplete run coverage: {} in {}'.format(item, path))
        # The product's inputs must equal this map exactly: every dependency declared at
        # the bound digest, and nothing else -- not even another artefact of this report.
        dependencies = product_dependencies(expected, roles, trained, run_bound, approval)
        dependencies.update(producer_declarations(side['producer']))
        for item, digest in sorted(side['inputs'].items()):
            require(item in dependencies, 'input names an artefact this report does not '
                    'bind as this product\'s dependency: {} in {}'.format(item, path))
            require(dependencies[item] == digest,
                    'input differs from the bound artefact: {} in {}'.format(item, path))
        for role in sorted(roles & set(trained)):  # named, before the general omission
            for dependency, item in sorted(training_dependencies(trained[role]).items()):
                require(side['inputs'].get(item['path']) == item['sha256'],
                        'missing training dependency: {} of {} in {}'.format(
                            dependency, role, path))
        for item, digest in sorted(dependencies.items()):
            require(side['inputs'].get(item) == digest,
                    'the product omits a required input: {} in {}'.format(item, path))
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
    released = stamp(ROOT / RELEASED, RELEASED_SHA256)
    attempts = sorted((attempt_record(item, pins) for item in attempt), key=lambda a: a['role'])
    require(sorted(a['role'] for a in attempts) == sorted(pins['checkpoints']),
            'every approved arm must be bound exactly once')
    paths = sorted(str(logical(item)) for item in runs)
    require(len(paths) == len(set(paths)) and len(paths) == len(IDENTITIES),
            'the forty evaluation runs')
    records = [run_record(item, attempts, released) for item in paths]
    run_bound = {item['path']: item.pop('bound') for item in records}
    require({(item['role'], item['num_shot'], item['seed']) for item in records} == IDENTITIES,
            'the forty evaluation runs are the registered role/K/seed set')
    head = head or binder.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                  text=True).strip()
    require(binder.subprocess.run(['git', 'cat-file', '-e', head + '^{commit}'], cwd=ROOT,
                                  stdout=binder.subprocess.PIPE,
                                  stderr=binder.subprocess.PIPE).returncode == 0,
            'invalid binding HEAD')
    published = bind_results(results, head, approval, records, attempts, run_bound)
    audited = audit_record(audit, head)
    evidenced = evidence_record(evidence, records, head)
    unseen, unseen_receipt = md.load_unseen(unseen_table, unseen_binding)
    cited = [item['sha256'] for item in published] + [unseen_receipt['sha256']]
    documents = []
    for path in sorted(str(logical(item)) for item in rendered):
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
                            audit=str(logical(audit)), evidence=sorted(evidence),
                            approved=identity['path'],
                            unseen_table=str(logical(unseen_table)),
                            unseen_binding=str(logical(unseen_binding)),
                            rendered=sorted(str(logical(item)) for item in rendered),
                            results=sorted(str(logical(item)) for item in results)))


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
