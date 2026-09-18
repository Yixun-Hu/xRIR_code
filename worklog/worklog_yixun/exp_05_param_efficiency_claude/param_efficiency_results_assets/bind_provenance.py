"""Read execution evidence and exclusively publish the exp_05 param-efficiency binding.

Binds the four certified training attempts (completion, manifest, inventory sidecar, hours
ledger, probe receipt and the approved epoch-12 checkpoint) AND every other attempt each
arm's ledger lists: each must have ended certified or aborted, and the external logs its
own records name -- the launcher keeps them outside the attempt directory -- are bound
too, so a later change to any of them changes the report.  Every probe attempt that
COMPLETED has exactly one receipt; a probe whose child failed has none to have, and is
published through its abort evidence.  The four arms must share one training closure and
each launcher closure must be one the approval lists: that is section 10a's single-delta
contract, recomputed here from the bound manifests.

It also binds the M pair's historical exp_01 checkpoints at the digests the profile pins
(their training predates this experiment and has no attempt to bind) and the sixty-six
evaluation runs -- six arms x two K x five seeds at k = 0, plus one seed-42 yaw block each
-- every one with its registered arm/K/seed identity, its tier metadata agreeing across
the manifest, the completion and both output metas, and its training or historical
checkpoint linkage.

The five canonical producer outputs are revalidated through ``tools.exp05_record``; each
must cover exactly its registered run set in its run flags AND in its per-run contracts,
each contract being the one the run it is filed under would produce, and declare EXACTLY
its own dependencies at the bound digests: every artefact those runs declare, the training
evidence of each trained arm it used, the approval and its producer closure, and nothing
else.  Finally the rendered documents -- which must cite every canonical digest and every
bound attempt completion -- the paper figures, the approval blob and git HEAD.  Run
directories are read and never modified.
"""
import argparse
import json
import re
from pathlib import Path

from tools import exp05_record as record
from tools import provenance as p
from tools.exp04_record import load_asset as exp04_asset
from tools.exp05_profiles import ARMS, get_profile, load_approved_digests
from tools.paired_compare import _closure_digest, producer_identity

binder = exp04_asset('bind_provenance')
ROOT, require, stamp, snapshot = binder.ROOT, binder.require, binder.stamp, binder.snapshot
check_ancestor, report_path = binder.check_ancestor, binder.report_path

ROLES = {arm['role']: arm for arm in ARMS}
TRAINED = tuple(sorted(role for role, arm in ROLES.items() if arm['tier'] != 'M'))
# The M pair is exp_01's: pinned by digest in the profile, with no exp_05 attempt to bind.
HISTORICAL = {role: dict(path=arm['checkpoint'], sha256=arm['sha256'], epoch=arm['epoch'])
              for role, arm in ROLES.items() if arm['tier'] == 'M'}
EVAL_SEEDS = tuple(get_profile('CURVE_K8')['eval_seeds'])
NUM_SHOT = (8, 1)
YAW_GRID = tuple(get_profile('YAW_K8_SEED42')['grid'])
YAW_SEED = 42
# The complete evaluation set the plan registers: six arms x two K x five seeds, plus the
# six seed-42 yaw blocks at K = 8.
IDENTITIES = frozenset([(role, shot, seed, 'k0') for role in ROLES for shot in NUM_SHOT
                        for seed in EVAL_SEEDS]
                       + [(role, 8, YAW_SEED, 'yaw') for role in ROLES])
PRODUCT_RUNS = dict(CURVE_K8=(8, 'k0'), TARGETS_K8=(8, 'k0'), CURVE_K1=(1, 'k0'),
                    TARGETS_K1=(1, 'k0'), YAW_K8_SEED42=(8, 'yaw'))
TRAINING = ('train_manifest', 'train_completion')
# Exactly what tools.param_curve.run_contract reads for a trained arm.
TRAINING_DEPENDENCIES = ('args.json', 'train_manifest.json')
CONTRACTS = {True: ('tools.exp05_eval', 'bound args.json'),
             False: ('tools.exp04_eval', 'pinned historical M checkpoint')}
SETUP_FAILURE = re.compile(r'.*_ABORTED_setup_failed(_[0-9a-f]+)?\Z')


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


def setup_failure(directory, record_, logs):
    """The one documented abort with no log to hash, in the exact shape that produces it.

    ``execute_attempt`` raises before the child is spawned: ``reason`` is still its
    initial ``setup_failed``, ``guard`` is None so ``abort_log`` renames nothing and
    ``log.aborted`` is null, ``abort_attempt`` names the directory after that reason, no
    ``execution.json`` was ever written and neither named log is on disk.  A logless
    abort that is not all of that is an abort with its evidence removed.
    """
    log = record_.get('log') or {}
    return (record_.get('reason') == 'setup_failed'
            and SETUP_FAILURE.match(directory.name) is not None
            and sorted(log) == ['aborted', 'original'] and log['aborted'] is None
            and isinstance(log['original'], str) and bool(log['original'])
            and not (directory / 'execution.json').is_file()
            and not any(item['present'] for item in logs))


def terminal_state(directory, row):
    """A ledger-listed attempt ended certified or aborted; bind that state's evidence."""
    completion, abort = directory / 'completion.json', directory / 'abort.json'
    if completion.is_file():  # a recovered attempt keeps its abort.json as well
        recorded = json.loads(completion.read_text()).get('log') or {}
        require(recorded.get('path') and recorded.get('sha256'),
                'a certified attempt records no log at a digest: ' + row['attempt'])
        return dict(state='certified', setup_failure=False,
                    logs=[external_log(recorded['path'], recorded['sha256'], mandatory=True)])
    require(abort.is_file(),
            'the attempt has no terminal state (completion.json or abort.json): '
            + row['attempt'])
    recorded = json.loads(abort.read_text())
    reason = recorded.get('reason')
    require(isinstance(reason, str) and reason,
            'an aborted attempt records no reason: ' + row['attempt'])
    log = recorded.get('log') or {}
    logs = [external_log(log[key]) for key in ('original', 'aborted') if log.get(key)]
    failure = setup_failure(directory, recorded, logs)
    require(failure or any(item['present'] for item in logs),
            'an aborted attempt must bind its log unless it is the documented setup '
            'failure: ' + row['attempt'])
    return dict(state='aborted', setup_failure=failure, reason=reason, logs=logs)


def other_attempts(certified, ledger, role):
    """Bind every OTHER attempt the arm's ledger lists, file by file.

    Section 3 requires the complete attempt history: each probe attempt, any aborted full
    run with its abort.json, and the external logs all of them reference are bound here at
    their current bytes, so a later change to any of them changes the report.
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
    """Every COMPLETED probe is named by exactly one receipt, at the bytes it recorded.

    The launcher writes a receipt only after the probe attempt returns, so a probe whose
    child failed has abort evidence and no receipt at all; demanding one would make the
    arm's real failure history unpublishable.  That attempt is bound through
    :func:`terminal_state` instead, and every receipt that does exist must still name a
    probe attempt of this arm, one each.
    """
    probes = {item['path'] for item in records if item['mode'] == 'probe'}
    completed = {item['path'] for item in records
                 if item['mode'] == 'probe' and item['state'] == 'certified'}
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
    missing = sorted(completed - set(named))
    require(not missing, 'a probe attempt has no receipt: ' + ', '.join(missing))
    return named


def attempt_record(attempt, pins):
    """One arm's full run, the evidence its limits came from, and its hours ledger."""
    bound_record, fields = snapshot(attempt, 'train')
    directory = Path(bound_record['path'])
    roles = [role for role, pin in pins['checkpoints'].items()
             if (ROOT / pin['path']).resolve().parent == directory]
    require(len(roles) == 1, 'attempt is not exactly one approved arm: ' + str(directory))
    role, pin = roles[0], pins['checkpoints'][roles[0]]
    checkpoint = stamp(ROOT / pin['path'], pin['sha256'])
    require(bound_record['outputs'].get(Path(checkpoint['path']).name) == checkpoint,
            'approved checkpoint is not this attempt\'s output: ' + role)
    require(pin['epoch'] == ROLES[role]['epoch']
            and Path(checkpoint['path']).name == 'epoch_%03d.pth' % pin['epoch'],
            'approved checkpoint epoch: ' + role)
    bound = fields['mutable_inputs']
    closures = fields['source_closures']
    require(closures['training']['sha256'] == pins['closures']['training'],
            'training closure is not the approved one: ' + role)
    require(closures['launcher']['sha256'] in list(pins['closures']['training_launcher']),
            'training launcher closure is not approved: ' + role)
    identity = fields['train_data_identity']['inventory_file']
    ledger_path = directory.parent / 'cumulative_hours.json'
    ledger = json.loads(ledger_path.read_text())
    full = [row for row in ledger['attempts'] if row['mode'] == 'full']
    require(len(full) <= 2, 'more than one retry recorded: ' + role)
    require(any(row['attempt'] == directory.name for row in full),
            'the ledger does not hold this attempt: ' + role)
    require(ledger['probe_receipt_sha256'] == bound['probe_receipt']['sha256'],
            'the ledger and the attempt name different probe receipts: ' + role)
    bound_record = dict(bound_record, role=role, checkpoint=checkpoint, ledger=stamp(ledger_path),
        full_attempts=len(full), other_attempts=other_attempts(directory, ledger, role),
        probe_receipts=[stamp(item) for item in sorted(directory.parent.glob('_probe_*.json'))],
        inventory=stamp(identity['path'], identity['sha256']),
        training_closure=closures['training']['sha256'],
        launcher_closure=closures['launcher']['sha256'],
        **{name: stamp(Path(fields['repo']) / bound[name]['path'], bound[name]['sha256'])
           for name in ('probe_receipt', 'control_args', 'effective_args')})
    final = directory.parent / 'final'
    require(final.is_symlink() and final.resolve() == directory,
            'the arm\'s final symlink is not this attempt: ' + role)
    bound_record['probe_linkage'] = probe_linkage(bound_record['other_attempts'],
                                                  bound_record['probe_receipts'])
    return bound_record


def historical_checkpoints():
    """The M pair's exp_01 checkpoints, at the digests the profile pins."""
    return {role: dict(stamp(ROOT / pin['path'], pin['sha256']), role=role, epoch=pin['epoch'])
            for role, pin in sorted(HISTORICAL.items())}


def tier_agreement(bound_record, fields, role):
    """The tier block the evaluator recorded must be this arm's, in all four records.

    The completion hash-binds the manifest and both outputs, which makes them immutable,
    not consistent; this is the launcher's own comparison made again over the bound bytes.
    """
    arm = ROLES[role]
    expected = {'vit_' + key: value for key, value in arm['config'].items()}
    expected.update(tier=arm['tier'], param_counts=dict(arm['counts']),
                    legacy_M=arm['tier'] == 'M',
                    args_json_sha256=fields['mutable_inputs']['train_args']['sha256'])
    payloads = [fields, json.loads(Path(bound_record['completion']['path']).read_text())]
    payloads += [json.loads(Path(bound_record['path']).joinpath(name).read_text()).get('meta') or {}
                 for name in ('per_sample_yaw.json', 'metrics_yaw.json')]
    for payload in payloads:
        require(all(payload.get(key) == value for key, value in expected.items()),
                'tier metadata agreement: ' + bound_record['path'])


def run_declarations(bound_record, fields):
    """Every artefact this run may declare, at the digest its hash-bound records give it.

    Data and source files are bound by the digests the evaluation manifest records -- the
    manifest itself is bound by the completion -- so nothing here is re-hashed off disk.
    """
    root = Path(fields['repo'])
    bound = {item['path']: item['sha256'] for item in
             [bound_record['manifest'], bound_record['completion'], bound_record['log']]
             + [item for item in bound_record['outputs'].values() if item]}
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


def run_closures(fields, role, approval, where):
    """One run's evaluator and writer closures: self-consistent, and both approved.

    A closure record that does not hash to its own digest is a claim, not evidence, and
    an evaluator or writer the approval does not pin is not this experiment's.  exp_04's
    frozen evaluator is admissible for the historical M pair ONLY against an explicit
    ``evaluator_exp04`` approval -- the live approvals carry none -- so a substituted
    closure cannot pass itself off as the legacy one.
    """
    pins = approval['blob']['closures']
    closures = fields['source_closures']
    for key in ('entrypoint', 'writer'):
        require(_closure_digest(closures[key]) == closures[key]['sha256'],
                'closure records: {} of {}'.format(key, where))
    require(closures['writer']['sha256'] == pins['writer'],
            'the writer closure is not the approved one: ' + where)
    entrypoint = closures['entrypoint']['sha256']
    require(entrypoint == pins['evaluator'] or (ROLES[role]['tier'] == 'M'
                                                and entrypoint == pins.get('evaluator_exp04')),
            'the evaluator closure is not the approved one: ' + where)


def run_record(run, attempts, historical, approval):
    """One evaluation run: its registered identity and the training it is bound to."""
    bound_record, fields = snapshot(run, 'eval')
    require(fields.get('split') == 'unseen' and fields.get('conditions') == 'P',
            'not an unseen condition-P evaluation: ' + str(run))
    seed, shot = fields.get('manifest_seed'), fields.get('num_shot')
    require(type(seed) is int and seed in EVAL_SEEDS and type(shot) is int and shot in NUM_SHOT,
            'run K/seed identity: ' + str(run))
    grid = tuple(fields['yaw_cols'])
    require(grid in ((0,), YAW_GRID), 'unregistered yaw grid: ' + str(run))
    kind = 'k0' if grid == (0,) else 'yaw'
    require(tuple(fields['acoustic_cols']) == grid and fields['e_acoustic_cols'] == []
            and fields.get('decomposition_batches') == 0, 'grid disagreement: ' + str(run))
    checkpoint = (Path(fields['repo']) / fields['checkpoint']).resolve()
    bound = fields['mutable_inputs']
    owners = [item for item in attempts if Path(item['checkpoint']['path']) == checkpoint]
    if owners:
        owner, role = owners[0], owners[0]['role']
        require(set(TRAINING) <= set(bound), 'missing training linkage: ' + str(run))
        for key, expected in (('train_manifest', owner['manifest']),
                              ('train_completion', owner['completion'])):
            require(stamp(Path(fields['repo']) / bound[key]['path'], bound[key]['sha256'])
                    == expected, 'training linkage: ' + str(run))
        require(Path(bound['train_args']['path']).resolve().parent == Path(owner['path']),
                'train_args linkage: ' + str(run))
    else:
        roles = [item['role'] for item in historical.values()
                 if Path(item['path']) == checkpoint
                 and fields['checkpoint_sha256'] == item['sha256']]
        require(len(roles) == 1, 'unregistered checkpoint: ' + str(run))
        role = roles[0]
        require(not set(TRAINING) & set(bound),
                'a historical M row carries no exp_05 training provenance: ' + str(run))
        require(Path(bound['train_args']['path']).resolve().parent == checkpoint.parent,
                'train_args linkage: ' + str(run))
    run_closures(fields, role, approval, str(run))
    tier_agreement(bound_record, fields, role)
    return dict(bound_record, role=role, num_shot=shot, seed=seed, kind=kind,
                entrypoint=fields['source_closures']['entrypoint']['sha256'],
                bound=run_declarations(bound_record, fields))


def training_dependencies(attempt):
    """Exactly the training evidence a product that used this arm consumes.

    ``tools.param_curve.run_contract`` reads the bound args.json, train_manifest.json and
    completion.json of a trained arm and nothing else of it; the arm's remaining files
    (the eleven unused epoch checkpoints, best/last, history.jsonl, the inventory sidecar,
    the hours ledger, the probe receipt and every other attempt) are bound in this
    report's attempt history and are inputs of nothing.
    """
    outputs = dict(attempt['outputs'])
    missing = [name for name in TRAINING_DEPENDENCIES if not outputs.get(name)]
    require(not missing, 'the attempt has no ' + ', '.join(missing) + ': ' + attempt['role'])
    items = {name: outputs[name] for name in TRAINING_DEPENDENCIES}
    items['completion.json'] = attempt['completion']
    return items


def run_coverage(runs):
    """Per run directory, every file of it a producer that used the run must declare."""
    coverage = {}
    for run in runs:
        files = [run['manifest'], run['completion']] + list(run['outputs'].values())
        coverage[run['path']] = {item['path']: item['sha256'] for item in files if item}
    return coverage


def live_producer():
    """Recompute the producer's identity from the working tree at this commit.

    ``producer_identity`` reads every file of the import closure and refuses any that
    differs from its reviewed blob, so a producer edited after publication cannot agree
    with the digest its own products recorded.
    """
    return producer_identity('tools.param_curve')


def producer_declarations(producer, approved=None):
    """The producer's own source files, recomputed live and verified byte for byte.

    A sidecar's closure records are the producer's claim about itself; the list they pin
    is evidence only while the sources they name still hash to the digests they gave
    them, so the closure is recomputed here, required to agree with the approval pin and
    with the sidecar, and every declared file is stamped at its declared digest.
    """
    require(_closure_digest(producer) == producer['sha256'], 'producer closure records')
    require(live_producer()['sha256'] == producer['sha256'],
            'the producer source is not the one this product was written by')
    require(approved is None or approved == producer['sha256'],
            'the producer closure is not the approved one')
    return {stamp(ROOT / item['path'], item['working_tree_sha256'])['path']:
            item['working_tree_sha256'] for item in producer['files']}


def product_dependencies(expected, roles, trained, run_bound, approval):
    """EXACTLY the inputs of a product built over exactly these runs.

    Everything each of those runs declares for itself -- its manifest, completion, both
    outputs, log, checkpoint, reference manifest, data inventory, evaluator and writer
    closures and mutable bindings -- plus the training evidence every trained arm it used
    makes it consume, and the approval blob.  Nothing else: neither an artefact of a run
    this product did not use nor a file of an arm it did use that no contract reads.  This
    map is working evidence, not report content: one run declares its whole data
    inventory, so it is never serialized.
    """
    dependencies = {approval['path']: approval['sha256']}
    for path in sorted(expected):
        dependencies.update(run_bound[path])
    for role in sorted(roles & set(trained)):
        for item in training_dependencies(trained[role]).values():
            dependencies[item['path']] = item['sha256']
    return dependencies


def bind_results(paths, head, approval, runs, attempts, run_bound):
    """Every canonical producer output, revalidated, with exact coverage of its inputs."""
    coverage = run_coverage(runs)
    by_path = {run['path']: run for run in runs}
    trained = {item['role']: item for item in attempts}
    records = []
    for path in sorted(str(Path(item).resolve()) for item in paths):
        name = json.loads(Path(path).read_bytes()).get('profile_name')
        require(name in record.KEYS, 'unregistered producer output: ' + path)
        data, receipt = record.load(path, name)
        record.validate(data, name)  # the same canonical validation the generators apply
        side = json.loads(Path(path + '.provenance.json').read_text())
        require(all(side['approved_digests'][key] == approval[key]
                    for key in ('path', 'sha256', 'git_blob'))
                and side['approved_digests']['pins'] == approval['blob'],
                'result approval identity or pins')
        require(side['producer']['sha256'] == approval['blob']['closures'][record.KEYS[name]],
                'result producer closure is not the approved one: ' + path)
        # The expected run set is derived from the bound identities, not from the product.
        shot, kind = PRODUCT_RUNS[name]
        expected = {run['path'] for run in runs
                    if run['num_shot'] == shot and run['kind'] == kind}
        require(len(expected) == (len(ROLES) if kind == 'yaw' else len(ROLES) * len(EVAL_SEEDS)),
                'the registered run set of this product: ' + path)
        require(set(side['run_flags']) == expected,
                'the product does not declare exactly its runs: ' + path)
        contracts = data.get('compatibility') or {}
        require(set(contracts) == expected,
                'the product does not record exactly its run contracts: ' + path)
        for run_path in sorted(expected):  # and each contract is THIS run's contract
            run, contract = by_path[run_path], contracts[run_path]
            current = run['entrypoint'] == approval['blob']['closures']['evaluator']
            evaluator, source = CONTRACTS[current]
            require(contract.get('sha256') == run['entrypoint']
                    and contract.get('evaluator') == evaluator
                    and contract.get('tier_source') == source
                    and (current or ROLES[run['role']]['tier'] == 'M'),
                    'the contract is not this run: {} in {}'.format(run_path, path))
        for run_path in sorted(expected):  # exact coverage: no subset of a used run's files
            for item, digest in sorted(coverage[run_path].items()):
                require(side['inputs'].get(item) == digest,
                        'incomplete run coverage: {} in {}'.format(item, path))
        roles = {by_path[item]['role'] for item in expected}
        # The product's inputs must equal this map exactly: every dependency declared at
        # the bound digest, and nothing else -- not even another artefact of this report.
        dependencies = product_dependencies(expected, roles, trained, run_bound, approval)
        dependencies.update(producer_declarations(
            side['producer'], approval['blob']['closures'][record.KEYS[name]]))
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
        records.append(dict(profile=name, sha256=receipt['sha256'],
                            producer_commit=side['producer']['commit'],
                            producer_closure_sha256=side['producer']['sha256'],
                            profile_digest=receipt['profile_digest'],
                            sidecar=stamp(path + '.provenance.json'),
                            outputs=[stamp(item, digest)
                                     for item, digest in sorted(side['outputs'].items())]))
    require(sorted(item['profile'] for item in records) == sorted(record.KEYS),
            'the five registered products, one each')
    return records


def collect(runs, attempt, results, rendered, figures=(), approved=None, head=None):
    pins, identity = load_approved_digests(approved)
    require(pins['schema_version'] == 1, 'approval pins are not final')
    approval = dict(identity, blob=json.loads(Path(identity['path']).read_text()))
    require(stamp(identity['path'])['sha256'] == identity['sha256'], 'approval digest mismatch')
    attempts = sorted((attempt_record(item, pins) for item in attempt),
                      key=lambda item: item['role'])
    require(sorted(item['role'] for item in attempts) == sorted(pins['checkpoints'])
            and sorted(item['role'] for item in attempts) == list(TRAINED),
            'every trained arm must be bound exactly once')
    require(len({item['training_closure'] for item in attempts}) == 1,
            'the four arms do not share one training closure')
    historical = historical_checkpoints()
    paths = sorted(str(Path(item).resolve()) for item in runs)
    require(len(paths) == len(set(paths)) and len(paths) == len(IDENTITIES),
            'the sixty-six evaluation runs')
    records = [run_record(item, attempts, historical, approval) for item in paths]
    run_bound = {item['path']: item.pop('bound') for item in records}
    require({(item['role'], item['num_shot'], item['seed'], item['kind'])
             for item in records} == IDENTITIES,
            'the sixty-six evaluation runs are the registered arm/K/seed set')
    head = head or binder.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                  text=True).strip()
    require(binder.subprocess.run(['git', 'cat-file', '-e', head + '^{commit}'], cwd=ROOT,
                                  stdout=binder.subprocess.PIPE,
                                  stderr=binder.subprocess.PIPE).returncode == 0,
            'invalid binding HEAD')
    published = bind_results(results, head, approval, records, attempts, run_bound)
    cited = ([item['sha256'] for item in published]
             + [item['completion']['sha256'] for item in attempts])
    documents = []
    for path in sorted(str(Path(item).resolve()) for item in rendered):
        text = Path(path).read_text(errors='replace')
        require(all(digest in text for digest in cited),
                'a rendered document does not cite every canonical digest: ' + path)
        documents.append(stamp(path))
    require(documents, 'no rendered documents')
    return dict(schema_version=1, git_HEAD=head, results=published, runs=records,
                attempts=attempts, historical_checkpoints=historical, documents=documents,
                figures=[stamp(item) for item in sorted(str(Path(name).resolve())
                                                        for name in figures)],
                approved_digests=approval,
                inputs=dict(runs=paths, attempt=[item['path'] for item in attempts],
                            results=sorted(str(Path(item).resolve()) for item in results),
                            rendered=sorted(str(Path(item).resolve()) for item in rendered),
                            figures=sorted(str(Path(item).resolve()) for item in figures),
                            approved=identity['path']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'attempt', 'results', 'rendered'):
        parser.add_argument('--' + name, nargs='+', required=True)
    parser.add_argument('--figures', nargs='+', default=[], help='paper PNG/PDF figures')
    for name in ('out', 'approved'):
        parser.add_argument('--' + name, required=name != 'approved')
    args = vars(parser.parse_args(argv))
    output = report_path(args.pop('out'))
    p.write_manifest(output, collect(**args))


if __name__ == '__main__':
    main()
