"""Read execution evidence and exclusively publish the exp_04 run binding."""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tools import provenance as p
from tools.exp04_profiles import load_approved_digests
from tools.exp04_record import load_asset

ROOT = Path(__file__).resolve().parents[4]
UNBOUND = object()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stamp(path, expected=UNBOUND):
    path = Path(path).resolve()
    digest = p.sha256_file(path)
    require(expected is UNBOUND or expected == digest, 'digest mismatch: ' + str(path))
    return dict(path=str(path), sha256=digest)


def snapshot(directory, kind):
    directory = Path(directory).resolve()
    manifest_path, completion_path = directory / (kind + '_manifest.json'), directory / 'completion.json'
    fields, completion = [json.loads(path.read_text()) for path in (manifest_path, completion_path)]
    require(sorted(p.name for p in directory.iterdir() if p.name != 'completion.json') ==
            sorted(completion['directory_listing']), 'actual directory listing mismatch')
    manifest = stamp(manifest_path, completion[kind + '_manifest_sha256'])
    outputs = {}
    for name, digest in completion['outputs'].items():
        path = directory / name
        path.resolve().relative_to(directory)
        require(digest is not None or path.is_dir(), 'missing output directory: ' + name)
        outputs[name] = stamp(path, digest) if digest is not None else None
    required = ('repo', 'source_closures', 'mutable_inputs')
    if kind == 'eval':
        require(set(outputs) == {'metrics_yaw.json', 'per_sample_yaw.json'}, 'eval output coverage')
        require(completion['directory_listing'] == sorted(['eval_manifest.json'] + list(outputs)), 'eval listing')
        require(type(completion['child_exit_status']) is int and completion['child_exit_status'] == 0, 'eval status')
        for value in (fields, completion):
            require(value['schema_version'] == 1 and value['confirmatory'] is True
                    and value['allow_dirty_used'] is False, 'non-final evaluation')
        for name in outputs:
            require(json.loads((directory / name).read_text())['meta']['eval_manifest_sha256'] == manifest['sha256'], 'manifest echo')
        required += ('checkpoint', 'manifest_path', 'evaluator_closure', 'data_identity')
    else:
        require(fields['mode'] == 'full' and fields['allow_dirty'] is False, 'non-final training')
        require(completion['directory_listing'] == completion['outputs'], 'training listing')
        required += ('train_data_identity', 'effective_args')
        require({'epoch_%03d.pth' % i for i in range(1, 13)} | {'history.jsonl', 'args.json',
                'effective_args.json', 'train_manifest.json'} <= set(outputs), 'training output coverage')
    require(not p.revalidate(fields, required=required, source_drift=[] if kind == 'train' else None), 'manifest input mismatch')
    log = completion['log']
    return dict(path=str(directory), manifest=manifest, completion=stamp(completion_path), outputs=outputs,
                log=stamp(Path(fields['repo']) / log['path'], log['sha256'])), fields


def check_ancestor(commit, head):
    require(isinstance(commit, str) and len(commit) == 40 and
            all(c in '0123456789abcdef' for c in commit), 'invalid producer commit')
    require(subprocess.run(['git', 'merge-base', '--is-ancestor', commit, head], cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0,
            'producer commit is not an ancestor of binding HEAD')


def bind_results(paths, head, approval, runs, training):
    records = []
    md = load_asset('make_results_md')
    bound = {f['path']: f['sha256'] for r in runs for f in
             [r['manifest'], r['completion']] + [o for o in r['outputs'].values() if o]}
    for path in sorted(str(Path(path).resolve()) for path in paths):
        data, _, draft = md.load(path)
        require(not draft, 'draft result refused')
        sidecar = Path(path + '.provenance.json')
        side = json.loads(sidecar.read_text())
        require(all(side['approved_digests'][key] == approval[key] for key in ('sha256', 'git_blob'))
                and side['approved_digests']['pins'] == approval['blob'], 'result approval identity or pins')
        inputs = {path: digest for path, digest in side['inputs'].items()
                  if Path(path).name in ('eval_manifest.json', 'completion.json', 'metrics_yaw.json', 'per_sample_yaw.json')
                  and path != training['completion']['path']}
        require(inputs and inputs.items() <= bound.items() and side['run_flags']
                and set(side['run_flags']) <= {r['path'] for r in runs}
                and all(side['inputs'].get(path) == digest for path, digest in bound.items()
                        if str(Path(path).parent) in side['run_flags']), 'result inputs are not the bound runs')
        check_ancestor(side['producer']['commit'], head)
        records.append(dict(profile=data['profile_name'], producer_commit=side['producer']['commit'],
                            sidecar=stamp(sidecar), outputs=[stamp(p, d) for p, d in sorted(side['outputs'].items())]))
    require(len({r['sidecar']['path'] for r in records}) == len(records), 'duplicate results')
    names, required = [r['profile'] for r in records], {name for _, name in md.INPUTS}
    require(len(names) == len(set(names)) and required <= set(names) <= required | {'GRID_SEED42', 'EPOCH9_K8'}, 'result profile coverage')
    return records


def collect(runs, attempt, probe_receipt, audit, results, approved=None, head=None):
    pins, identity = load_approved_digests(approved)
    require(pins['schema_version'] == 1, 'approval pins are not final')
    approval = dict(identity, blob=json.loads(Path(identity['path']).read_text()))
    require(stamp(identity['path'])['sha256'] == identity['sha256'], 'approval digest mismatch')
    training, fields = snapshot(attempt, 'train')
    receipt, audit_record = stamp(probe_receipt), stamp(audit)
    bound_probe = fields['mutable_inputs']['probe_receipt']
    require(stamp(Path(fields['repo']) / bound_probe['path'], bound_probe['sha256']) == receipt, 'probe receipt linkage')
    checkpoint = pins['checkpoints']['aug']
    selected = stamp(ROOT / checkpoint['path'], checkpoint['sha256'])
    require(Path(selected['path']).parent == Path(training['path'])
            and training['outputs'].get(Path(selected['path']).name) == selected, 'approved training checkpoint')
    paths = sorted(str(Path(run).resolve()) for run in runs)
    require(paths and len(paths) == len(set(paths)), 'empty or duplicate run set')
    records = []
    for path in paths:
        record, manifest = snapshot(path, 'eval')
        bindings = manifest['mutable_inputs']
        if (Path(manifest['repo']) / manifest['checkpoint']).resolve() == Path(selected['path']):
            require({'train_manifest', 'train_completion'} <= set(bindings), 'missing training linkage')
        for key, expected in (('train_manifest', training['manifest']), ('train_completion', training['completion'])):
            if key in bindings:
                value = bindings[key]
                require(stamp(Path(manifest['repo']) / value['path'], value['sha256']) == expected, 'training linkage')
        records.append(record)
    head = head or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.check_call(['git', 'cat-file', '-e', head + '^{commit}'], cwd=ROOT)
    return dict(schema_version=1, git_HEAD=head, results=bind_results(results, head, approval, records, training), runs=records, training=training, probe_receipt=receipt,
        audit=audit_record, approved_digests=approval, inputs=dict(runs=paths, attempt=training['path'],
        probe_receipt=receipt['path'], audit=audit_record['path'], approved=identity['path'],
        results=sorted(str(Path(path).resolve()) for path in results)))


def report_path(directory):
    return Path(directory) / ('binding_report_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--results', nargs='+', required=True, help='Canonical producer JSONs')
    for name in ('attempt', 'probe-receipt', 'audit', 'out', 'approved'):
        parser.add_argument('--' + name, required=name != 'approved')
    args = vars(parser.parse_args(argv))
    output = report_path(args.pop('out'))
    p.write_manifest(output, collect(**args))


if __name__ == '__main__':
    main()
