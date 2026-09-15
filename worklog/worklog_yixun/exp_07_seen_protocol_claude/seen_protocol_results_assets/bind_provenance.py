"""Read execution evidence and exclusively publish the exp_07 seen-protocol binding.

Binds the three training attempts (completion, manifest, inventory sidecar, probe receipt,
hours ledger and the approved checkpoint), the released reference checkpoint, the forty
evaluation runs with their seen-split bindings, the alignment audit, the four canonical
producer outputs with their sidecars and companions, the rendered Markdown/HTML/LaTeX,
the exp_04 inputs the combined table reuses, the approval blob and git HEAD.  Each arm's
ledger must show at most one retry.  Run directories are read and never modified.
"""
import argparse
import json
from pathlib import Path

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


def bind_results(paths, head, approval, runs, attempts):
    """Every canonical producer output, its sidecar and the runs its inputs name."""
    bound = {item['path']: item['sha256'] for run in runs for item in
             [run['manifest'], run['completion']] + [o for o in run['outputs'].values() if o]}
    directories = {item['path'] for item in attempts}
    records = []
    for path in sorted(str(Path(item).resolve()) for item in paths):
        name = json.loads(Path(path).read_bytes()).get('profile_name')
        require(name in md.KEYS, 'unregistered producer output: ' + path)
        data, receipt = md.load(path, name)
        side = json.loads(Path(path + '.provenance.json').read_text())
        require(all(side['approved_digests'][key] == approval[key] for key in ('sha256', 'git_blob'))
                and side['approved_digests']['pins'] == approval['blob'],
                'result approval identity or pins')
        inputs = {item: digest for item, digest in side['inputs'].items()
                  if Path(item).name in RUN_FILES and str(Path(item).parent) not in directories}
        require(inputs and inputs.items() <= bound.items() and side['run_flags']
                and set(side['run_flags']) <= {run['path'] for run in runs},
                'result inputs are not the bound runs')
        check_ancestor(side['producer']['commit'], head)
        records.append(dict(profile=name, pairing=data.get('pairing'), sha256=receipt['sha256'],
                            producer_commit=side['producer']['commit'],
                            sidecar=stamp(path + '.provenance.json'),
                            outputs=[stamp(item, digest)
                                     for item, digest in sorted(side['outputs'].items())]))
    require([item['profile'] for item in records].count('TABLE_SEEN_V1') == 1, 'one seen table')
    require(sorted(tuple(item['pairing']) for item in records
                   if item['profile'] == 'PAIRS_SEEN_V1') == sorted(md.PAIRINGS),
            'the three registered pairings')
    return records


def collect(runs, attempt, audit, results, rendered, unseen_table, unseen_binding,
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
    published = bind_results(results, head, approval, records, attempts)
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
                attempts=attempts, released_checkpoint=released, audit=stamp(audit),
                documents=documents, approved_digests=approval,
                unseen=dict(table=stamp(unseen_table, unseen_receipt['sha256']),
                            sidecar=stamp(unseen_table + '.provenance.json'),
                            binding_report=stamp(unseen_binding,
                                                 unseen_receipt['binding_report']['sha256'])),
                inputs=dict(runs=paths, attempt=[item['path'] for item in attempts],
                            audit=str(Path(audit).resolve()), approved=identity['path'],
                            unseen_table=str(Path(unseen_table).resolve()),
                            unseen_binding=str(Path(unseen_binding).resolve()),
                            rendered=sorted(str(Path(item).resolve()) for item in rendered),
                            results=sorted(str(Path(item).resolve()) for item in results)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'attempt', 'results', 'rendered'):
        parser.add_argument('--' + name, nargs='+', required=True)
    for name in ('audit', 'unseen-table', 'unseen-binding', 'out', 'approved'):
        parser.add_argument('--' + name, required=name != 'approved')
    args = vars(parser.parse_args(argv))
    output = report_path(args.pop('out'))
    p.write_manifest(output, collect(**args))


if __name__ == '__main__':
    main()
