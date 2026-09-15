"""Read execution evidence and exclusively publish the exp_07 seen-protocol binding.

Binds the three certified training attempts (completion, manifest, inventory sidecar, probe
receipt, hours ledger and the approved checkpoint) AND every other attempt each arm's
ledger lists -- aborted full runs with their abort.json, probe attempts, and the arm's
probe receipts -- so a later change to any of them changes the report.  It also binds the
released reference checkpoint, the forty evaluation runs with their seen-split bindings,
the seen alignment audit (protocol, cohort digest, passed, commit), the required
``--evidence`` artefacts (the GPU parity log and the released-checkpoint calibration,
whose pre-registered acceptance rule is recomputed here), the four canonical producer
outputs revalidated through the generators' own checks with exact per-run input coverage,
their sidecars and companions, the rendered Markdown/HTML/LaTeX, the exp_04 inputs the
combined table reuses, the approval blob and git HEAD.  Each arm's ledger must show at
most one retry.  Run directories are read and never modified.
"""
import argparse
import json
from pathlib import Path

from tools import exp07_launcher as launcher
from tools import exp07_provenance as e7p
from tools import provenance as p
from tools.exp04_record import load_asset as exp04_asset
from tools.exp07_profiles import ARMS, RELEASED_SHA256, load_approved_digests
from tools.exp07_record import load_asset

binder = exp04_asset('bind_provenance')
md = load_asset('make_results_md')
ROOT, require, stamp, snapshot = binder.ROOT, binder.require, binder.stamp, binder.snapshot
check_ancestor, report_path = binder.check_ancestor, binder.report_path
TRAINING = ('train_args', 'train_manifest', 'train_completion')
RUN_FILES = ('eval_manifest.json', 'completion.json', 'metrics_yaw.json', 'per_sample_yaw.json')
RELEASED = next(arm['checkpoint'] for arm in ARMS if arm['reference'])
EVIDENCE = ('gpu_parity', 'calibration')  # --evidence NAME=PATH, both required
CALIBRATION_METRICS = ('EDT', 'C50', 'T60')
# The historical full-split reproduction the released row is calibrated against
# (plan section 2; exp_01 results.md): EDT seconds, C50 dB, T60 per cent.
HISTORICAL = {'EDT': .0389, 'C50': 1.029, 'T60': 7.27}
SEEN_BATCHES = launcher.TRAIN_BATCHES['seen']


def other_attempts(certified, ledger, role):
    """Bind every OTHER attempt the arm's ledger lists, file by file.

    Plan section 3 requires the complete attempt history: an aborted full run (with its
    abort.json) and each probe attempt are bound here at their current bytes, so a later
    change to any of them changes the report and check_record.py fails.
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
                            files=[stamp(item) for item in files]))
    return records


def evidence_record(bindings):
    """--evidence NAME=PATH for the parity log and the released-checkpoint calibration."""
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
    record['calibration'] = calibration_record(record['calibration'])
    return record


def calibration_record(item):
    """The released row's five-seed means must meet the pre-registered acceptance rule."""
    data = json.loads(Path(item['path']).read_text())
    require(data.get('passed') is True and data.get('role') == 'released_seen'
            and data.get('protocol') == 'seen' and data.get('num_shot') == 8,
            'calibration identity')
    metrics = data.get('metrics') or {}
    require(sorted(metrics) == sorted(CALIBRATION_METRICS), 'calibration metrics')
    accepted = {}
    for name in CALIBRATION_METRICS:
        cell = metrics[name]
        require(cell.get('historical') == HISTORICAL[name], 'calibration historical: ' + name)
        tolerance = 3 * cell['sd'] + .02 * abs(HISTORICAL[name])
        require(cell.get('passed') is True
                and abs(cell['mean'] - HISTORICAL[name]) <= tolerance,
                'calibration acceptance: ' + name)
        accepted[name] = dict(mean=cell['mean'], sd=cell['sd'], historical=HISTORICAL[name],
                              tolerance=tolerance)
    return dict(item, role=data['role'], num_shot=data['num_shot'], metrics=accepted)


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
    return dict(record, role=role, checkpoint=checkpoint, ledger=stamp(ledger_path),
                full_attempts=len(full),
                other_attempts=other_attempts(directory, ledger, role),
                probe_receipts=[stamp(item) for item in
                                sorted(directory.parent.glob('_probe_*.json'))],
                inventory=stamp(identity['path'], identity['sha256']),
                probe_receipt=stamp(Path(fields['repo']) / bound['probe_receipt']['path'],
                                    bound['probe_receipt']['sha256']),
                seen_split=stamp(Path(fields['repo']) / bound['seen_split']['path'],
                                 bound['seen_split']['sha256']))


def run_record(run, attempts, released):
    """One evaluation run: the seen split it used and the training it is bound to."""
    record, fields = snapshot(run, 'eval')
    require(fields.get('split') == 'seen', 'not a seen evaluation: ' + str(run))
    split = fields['mutable_inputs']['seen_split']
    require(split['path'] == e7p.SEEN_SPLIT, 'seen_split binding path: ' + str(run))
    split = stamp(Path(fields['repo']) / split['path'], split['sha256'])
    checkpoint = (Path(fields['repo']) / fields['checkpoint']).resolve()
    owners = [item for item in attempts if Path(item['checkpoint']['path']) == checkpoint]
    if not owners:
        require(Path(released['path']) == checkpoint and
                fields['checkpoint_sha256'] == released['sha256'], 'unregistered checkpoint')
        require(not set(TRAINING) & set(fields['mutable_inputs']),
                'the released row has no training provenance')
        return dict(record, role='released_seen', seen_split=split)
    owner, bound = owners[0], fields['mutable_inputs']
    require(set(TRAINING) <= set(bound), 'missing training linkage: ' + str(run))
    for key, expected in (('train_manifest', owner['manifest']),
                          ('train_completion', owner['completion'])):
        require(stamp(Path(fields['repo']) / bound[key]['path'], bound[key]['sha256']) == expected,
                'training linkage: ' + str(run))
    require(Path(bound['train_args']['path']).resolve().parent == Path(owner['path']),
            'train_args linkage: ' + str(run))
    return dict(record, role=owner['role'], seen_split=split)


def known_digests(runs, attempts, approval, released):
    """Every artefact this report binds, by absolute path, with the digest it was bound at."""
    known = {approval['path']: approval['sha256'], released['path']: released['sha256']}
    for run in runs:
        for item in [run['manifest'], run['completion']] + list(run['outputs'].values()):
            if item:
                known[item['path']] = item['sha256']
    for attempt in attempts:
        items = [attempt[key] for key in
                 ('manifest', 'completion', 'checkpoint', 'ledger', 'inventory',
                  'probe_receipt', 'seen_split')]
        items += [item for item in attempt['outputs'].values() if item]
        for item in items:
            known[item['path']] = item['sha256']
    return known


def run_coverage(runs):
    """Per run directory, every file of it a producer that used the run must declare."""
    coverage = {}
    for run in runs:
        files = [run['manifest'], run['completion']] + list(run['outputs'].values())
        coverage[run['path']] = {item['path']: item['sha256'] for item in files if item}
    return coverage


def bind_results(paths, head, approval, runs, attempts, released):
    """Every canonical producer output, revalidated, with exact coverage of its inputs."""
    known = known_digests(runs, attempts, approval, released)
    coverage = run_coverage(runs)
    run_paths = set(coverage)
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
        flags = set(side['run_flags'])
        require(flags and flags <= run_paths, 'result inputs are not the bound runs')
        for run_path in sorted(flags):  # exact coverage: no subset of a used run's files
            for item, digest in sorted(coverage[run_path].items()):
                require(side['inputs'].get(item) == digest,
                        'incomplete run coverage: {} in {}'.format(item, path))
        for item, digest in sorted(side['inputs'].items()):
            if item in known:
                require(known[item] == digest,
                        'input differs from the bound artefact: {} in {}'.format(item, path))
            else:
                require(Path(item).name not in RUN_FILES or str(Path(item).parent) in flags,
                        'input names a run this report does not bind: {} in {}'.format(item, path))
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
    paths = sorted(str(Path(item).resolve()) for item in runs)
    require(len(paths) == len(set(paths)) and len(paths) == 40, 'the forty evaluation runs')
    records = [run_record(item, attempts, released) for item in paths]
    head = head or binder.subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                  text=True).strip()
    require(binder.subprocess.run(['git', 'cat-file', '-e', head + '^{commit}'], cwd=ROOT,
                                  stdout=binder.subprocess.PIPE,
                                  stderr=binder.subprocess.PIPE).returncode == 0,
            'invalid binding HEAD')
    published = bind_results(results, head, approval, records, attempts, released)
    audited, evidenced = audit_record(audit, head), evidence_record(evidence)
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
