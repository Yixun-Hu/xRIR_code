"""Completion evidence for exp_06 runs, dispatched by run type (plan v4 sections 5, 6.4).

The entry points write ``provenance.json``; this external finalizer decides completion
after the child has exited and writes ``completion.json``. Evidence depends on the run
type recorded on the command line (and, for ``full``, in ``provenance.json``):

``full``      twelve-epoch pretraining: closure drift, recipe/production/derived schema,
              budget completeness, ``epoch_012.pth`` equal to ``last.pth['model']``,
              child status 0 and a closed log.
``smoke``/``probe``  no artifacts; labelled ``diagnostic`` and never admissible as an arm.

Every failure raises ``ValueError`` naming its cause and writes nothing; a re-run
produces byte-identical bytes, and an existing completion that differs is refused.
"""
import datetime
import hashlib
import json
from pathlib import Path

import torch

from tools import exp06_recipe, provenance

REPO = Path(__file__).resolve().parents[1]
MARKER = 'EXP06_CHILD_EXIT'
DIAGNOSTIC = ('smoke', 'probe')
FULL_ARTIFACTS = ('provenance.json', 'args.json', 'history.jsonl', 'last.pth', 'epoch_012.pth')
EPOCH_CHECKPOINT = 'epoch_{:03d}.pth'.format(exp06_recipe.EXP01_RECIPE['epochs'])


def _require(ok, cause):
    if not ok:
        raise ValueError(cause)


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError('unreadable {}: {}'.format(label, error)) from error


def closure_digest(files):
    """The exp_03-compatible digest over [[path, reviewed blob], ...]."""
    return hashlib.sha256(json.dumps([[f['path'], f['reviewed_blob_sha256']] for f in files],
                                     sort_keys=True).encode()).hexdigest()


def closed_log(log, child_exit):
    """Require the launcher's end marker as the log's last line, agreeing on the status."""
    try:
        text = Path(log).read_text()
    except OSError as error:
        raise ValueError('unreadable log: {}'.format(error)) from error
    lines = text.splitlines()
    _require(lines and lines[-1].startswith(MARKER + ' '),
             'log does not end with the ' + MARKER + ' marker (child still writing?)')
    parts = lines[-1].split()
    _require(len(parts) == 3, 'malformed ' + MARKER + ' marker: ' + lines[-1])
    try:
        code, stamp = int(parts[1]), datetime.datetime.fromisoformat(parts[2])
    except ValueError as error:
        raise ValueError('malformed ' + MARKER + ' marker: {}'.format(error)) from error
    _require(stamp.tzinfo is not None, MARKER + ' marker needs a timezone-aware timestamp')
    _require(code == child_exit,
             'log marker status {} differs from the reported child status {}'.format(code, child_exit))
    return {'path': str(Path(log).resolve()), 'sha256': provenance.sha256_file(log)}, parts[2]


def artifacts(run_dir, names):
    """Hash every required artifact; a missing one is refused by name."""
    hashes = {}
    for name in names:
        path = Path(run_dir) / name
        _require(path.is_file(), 'missing artifact {} in {}'.format(name, run_dir))
        hashes[name] = provenance.sha256_file(path)
    return hashes


def full_evidence(run_dir, repo):
    """Verify the twelve-epoch pretraining contract of plan section 5."""
    run_dir = Path(run_dir)
    hashes = artifacts(run_dir, FULL_ARTIFACTS)  # every file must exist before it is parsed
    record = _read_json(run_dir / 'provenance.json', 'provenance.json')
    _require(record.get('run_type') == 'full',
             'provenance run_type is {!r}, not full'.format(record.get('run_type')))
    try:
        closure = record['source_closures']['training']
        commit = record['reviewed_commit']
        paths = [entry['path'] for entry in closure['files']]
    except (KeyError, TypeError) as error:
        raise ValueError('provenance.json has no training source closure: {}'.format(error)) from error
    _require(closure.get('sha256') == closure_digest(closure['files']),
             'recorded source closure digest does not match its own file list')
    _require(provenance.closure_record(paths, commit, repo)[1] == closure['sha256'],
             'source closure drift: the reviewed blobs differ from the recorded digest')
    args = _read_json(run_dir / 'args.json', 'args.json')
    rows = []
    for number, line in enumerate(Path(run_dir / 'history.jsonl').read_text().splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except ValueError as error:
            raise ValueError('history.jsonl line {} is not JSON: {}'.format(number, error)) from error
    last = torch.load(str(run_dir / 'last.pth'), map_location='cpu')
    meta = {key: last[key] for key in ('epoch', 'batch_idx') if key in last}
    deviations = exp06_recipe.check_all(args, history_rows=rows, last_meta=meta)
    _require(not deviations, 'schema deviations: ' + '; '.join(deviations))
    _require(last.get('args') == args, 'the args recorded in last.pth differ from args.json')
    state = torch.load(str(run_dir / EPOCH_CHECKPOINT), map_location='cpu')
    model = last['model']
    _require(set(state) == set(model), EPOCH_CHECKPOINT + ' has a different parameter set than last.pth')
    _require(all(torch.equal(state[key], model[key]) for key in state),
             EPOCH_CHECKPOINT + ' differs tensor-wise from last.pth["model"]')
    return dict(artifacts=hashes, epochs=len(rows),
                backbone=args['backbone'], source_closure_sha256=closure['sha256'],
                registry_sha256=record.get('registry_sha256'),
                git_head=record.get('git_state', {}).get('HEAD'),
                checkpoint={'path': EPOCH_CHECKPOINT, 'sha256': hashes[EPOCH_CHECKPOINT],
                            'epoch': exp06_recipe.EXP01_RECIPE['epochs']})


def diagnostic_evidence(run_dir, receipt):
    """A smoke or probe proves nothing about an arm; keep its receipt, demand no artifact."""
    fields = dict(artifacts={})
    if receipt is not None:
        path = Path(receipt)
        _require(path.is_file(), 'missing smoke receipt: {}'.format(receipt))
        fields['receipt'] = {'path': str(path.resolve()), 'sha256': provenance.sha256_file(path)}
    return fields


def write_completion(path, fields):
    """Publish once; a re-run must produce the same bytes, a different result is refused."""
    payload = json.dumps(fields, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    if Path(path).exists():
        _require(Path(path).read_bytes() == payload,
                 'completion.json already exists and differs from this result')
        return fields
    provenance.write_completion(path, fields)
    return fields


def finalize(run_dir, run_type, log, child_exit, repo=REPO, receipt=None):
    """Verify one child's evidence for its run type and write completion.json."""
    run_dir = Path(run_dir)
    _require(run_type in ('full',) + DIAGNOSTIC, 'unknown run type: {!r}'.format(run_type))
    _require(run_dir.is_dir(), 'run directory does not exist: {}'.format(run_dir))
    _require(type(child_exit) is int, 'child status must be an integer')
    log_record, child_exit_time = closed_log(log, child_exit)
    diagnostic = run_type in DIAGNOSTIC
    if not diagnostic:
        _require(child_exit == 0, 'child exited with status {}'.format(child_exit))
    fields = dict(schema_version=1, run_type=run_type, run_dir=str(run_dir.resolve()),
                  repo=str(Path(repo).resolve()), child_exit=child_exit,
                  child_exit_time=child_exit_time, log=log_record,
                  diagnostic=diagnostic, admissible_arm=not diagnostic)
    fields.update(diagnostic_evidence(run_dir, receipt) if diagnostic
                  else full_evidence(run_dir, repo))
    return write_completion(run_dir / 'completion.json', fields)
