"""exp_06 launcher: its own bindings, its own writer closure, exp_04's finalisation."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools import exp04_eval_launch as inherited
from tools import exp06_eval as evaluator
from tools import exp06_eval_launch as subject
from tools import exp06_heading
from tools import provenance as p
from tools.reference_manifest import manifest_hash

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = {'seed': 42, 'num_shot': 8, 'entries': [{'query': 'room/q.wav'}]}


def write_cache(room, levels):
    """A twelve-microphone HAA room cache the heading estimator can really read."""
    room.mkdir(parents=True)
    room.joinpath('meta.json').write_text(json.dumps(
        dict(train=list(range(12)), valid=[12], test=[12, 13], sr=22050)))
    theta = np.deg2rad([-90] * 4 + [90] * 8)
    dist = np.arange(1, 13) / 3
    xyz = np.zeros((14, 3))
    xyz[:12, :2] = np.stack((np.cos(theta), np.sin(theta)), axis=1) * dist[:, None]
    xyz[:, 2] = 50
    np.save(room / 'xyzs.npy', xyz)
    np.save(room / 'speaker_xyz.npy', np.zeros(3))
    rirs = np.zeros((14, 1200))
    rirs[:12, 10] = 10 ** (np.asarray(levels) / 20) / dist
    np.save(room / 'rirs.npy', rirs)
    return room


@pytest.fixture(scope='session')
def headings(tmp_path_factory):
    """One decided and one refused heading record, both schema-valid on disk."""
    root = tmp_path_factory.mktemp('headings')
    out = {}
    for name, levels in (('decided', [9] * 4 + [0] * 8), ('refused', [0] * 12)):
        record = exp06_heading.estimate_room_heading(write_cache(root / name / 'hallway', levels))
        path = root / (name + '.json')
        exp06_heading.write_heading_json(path, record)
        out[name] = path
    assert exp06_heading.read_heading_json(out['decided'])['decision'] == 'estimated'
    assert exp06_heading.read_heading_json(out['refused'])['decision'] == 'refused'
    return out


@pytest.fixture
def launch_args(tmp_path):
    reference = tmp_path / 'reference.json'
    reference.write_text(json.dumps(REFERENCE))
    checkpoint = tmp_path / 'epoch_012.pth'
    checkpoint.write_bytes(b'weights')
    def build(*extra):
        return subject.parse_args([
            '--backbone', 'cylindrical_oriented', '--checkpoint', str(checkpoint),
            '--manifest', str(reference), '--manifest-hash', manifest_hash(REFERENCE),
            '--out-dir', str(tmp_path / 'run'), '--log-dir', str(tmp_path / 'logs'),
            '--data-root', str(tmp_path), '--run-label', 'cylor_seed42',
            '--reviewed-commit', 'HEAD', '--num-shot', '8', '--conditions', 'P',
            '--yaw-cols', '0', '--acoustic-cols', '0', '--e-acoustic-cols',
            '--max-samples', '16', '--checkpoint-epoch', '12', *extra])
    return build


@pytest.fixture
def bound(monkeypatch):
    """exp_04's own fixture pattern: real field assembly, stubbed git and inventory."""
    monkeypatch.setattr(p, 'source_closure', lambda module, repo: [module + '.py'])
    monkeypatch.setattr(p, 'closure_record', lambda files, commit, repo: (
        [{'path': path, 'reviewed_blob_sha256': 'sha', 'working_tree_sha256': 'sha',
          'commits_after_reviewed': []} for path in files], 'closure-sha'))
    monkeypatch.setattr(p, 'data_identity', lambda path, root: {'inventory_sha256': 'data-sha'})
    monkeypatch.setattr(p, 'git_state', lambda repo: {'dirty_outside_worklog': False})
    monkeypatch.setattr(inherited, '_capture_environment', lambda *a: {'python': '3.8'})


def test_child_command_is_the_exp06_evaluator_argv(launch_args):
    args = launch_args()
    command = subject.child_command(args, ROOT)
    assert command[:2] == [sys.executable, str(ROOT / 'tools/exp06_eval.py')]
    assert command[2:] == [
        '--backbone', 'cylindrical_oriented', '--checkpoint', args.checkpoint,
        '--manifest', args.manifest, '--manifest-hash', args.manifest_hash,
        '--out-dir', args.out_dir, '--eval-manifest', args.eval_manifest,
        '--conditions', 'P', '--yaw-cols', '0', '--acoustic-cols', '0', '--e-acoustic-cols',
        '--batch-size', '16', '--num-workers', '6', '--max-samples', '16', '--gl-seed', '0',
        '--threads', '4', '--log-interval', '10', '--decomposition-batches', '1',
        '--checkpoint-role', 'arm', '--checkpoint-epoch', '12']
    child = evaluator.parse_args(command[2:])
    assert child.out_dir == args.out_dir and child.eval_manifest == args.eval_manifest
    assert child.checkpoint_epoch == 12 and child.tf32 is False


def test_the_launcher_creates_the_eval_manifest_path(launch_args):
    args = launch_args()
    assert args.eval_manifest == str(Path(args.out_dir) / 'eval_manifest.json')
    with pytest.raises(SystemExit):
        launch_args('--eval-manifest', args.eval_manifest)


@pytest.mark.parametrize('binding', ['invented=/x', 'heading', 'heading=', '=/x',
                                     'train_args'])
def test_unlisted_or_malformed_bindings_are_refused(launch_args, binding):
    with pytest.raises(ValueError, match='bind-input'):
        subject.split_bindings(launch_args('--bind-input', binding))


def test_a_duplicate_binding_is_refused(launch_args, headings):
    args = launch_args('--bind-input', 'heading=' + str(headings['decided']),
                       '--bind-input', 'heading=' + str(headings['decided']))
    with pytest.raises(ValueError, match='bind-input'):
        subject.split_bindings(args)


def test_exp06_bindings_are_validated_here_and_stripped_for_exp04(launch_args, headings):
    args = launch_args('--bind-input', 'heading=' + str(headings['decided']),
                       '--bind-input', 'train_completion=' + str(headings['decided']))
    exp04_bindings, exp06_bindings = subject.split_bindings(args)
    assert exp04_bindings == ['train_completion=' + str(headings['decided'])]
    assert set(exp06_bindings) == {'heading'}
    assert exp06_bindings['heading'] == {'path': str(headings['decided'].resolve()),
                                         'sha256': p.sha256_file(headings['decided'])}
    assert 'heading' not in inherited.MUTABLE_INPUTS  # exp_04's own list is unchanged


@pytest.mark.parametrize('name', ['refused', 'missing', 'malformed'])
def test_an_unusable_heading_binding_is_refused(launch_args, headings, tmp_path, name):
    path = {'refused': headings['refused'], 'missing': tmp_path / 'absent.json',
            'malformed': tmp_path / 'malformed.json'}[name]
    if name == 'malformed':
        path.write_text('{"decision": "estimated"}')
    with pytest.raises(ValueError,
                       match='(unusable heading binding|records the decision)'):
        subject.split_bindings(launch_args('--bind-input', 'heading=' + str(path)))


def test_both_writer_closures_and_the_metadata_reach_the_manifest(launch_args, headings, bound):
    args = launch_args('--bind-input', 'heading=' + str(headings['decided']),
                       '--bind-input', 'train_completion=' + str(headings['decided']))
    command = subject.child_command(args, ROOT)
    fields = subject.build_fields(args, command, ROOT)
    assert set(fields['source_closures']) == {'entrypoint', 'writer', 'writer_exp06'}
    assert fields['source_closures']['entrypoint']['files'][0]['path'] == 'tools.exp06_eval.py'
    assert fields['source_closures']['writer']['files'][0]['path'] == 'tools.exp04_eval_launch.py'
    assert fields['source_closures']['writer_exp06']['files'][0]['path'] == 'tools.exp06_eval_launch.py'
    assert set(fields['mutable_inputs']) == {'heading', 'train_completion'}
    assert fields['mutable_inputs']['heading']['sha256'] == p.sha256_file(headings['decided'])
    for key, value in evaluator.exp06_metadata(args).items():
        assert fields[key] == value
    assert fields['command'] == command and fields['schema_version'] != 0
    assert fields['reviewed_commit'] == subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def test_a_drifted_writer_closure_is_refused(launch_args, bound, monkeypatch):
    args = launch_args()
    command = subject.child_command(args, ROOT)
    def drifted(files, commit, repo):
        return ([{'path': path, 'reviewed_blob_sha256': 'sha', 'working_tree_sha256': 'other',
                  'commits_after_reviewed': []} for path in files], 'closure-sha')
    monkeypatch.setattr(p, 'closure_record', drifted)
    with pytest.raises(ValueError, match='closure'):
        subject.build_fields(args, command, ROOT)


def test_a_dirty_tree_refuses_a_confirmatory_launch(tmp_path, launch_args):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'tracked.txt').write_text('original\n')
    subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    subprocess.run(['git', 'add', 'tracked.txt'], cwd=repo, check=True)
    subprocess.run(['git', '-c', 'user.email=a@b', '-c', 'user.name=t', 'commit', '-qm', 'x'],
                   cwd=repo, check=True)
    (repo / 'tracked.txt').write_text('changed\n')
    args = launch_args()
    args.max_samples = 0
    with pytest.raises(ValueError, match='dirty_outside_worklog'):
        subject.build_fields(args, subject.child_command(args, repo), repo)


def test_main_runs_the_child_through_the_shared_finaliser(launch_args, monkeypatch):
    seen = {}
    monkeypatch.setattr(subject.launcher, 'execute_run',
                        lambda args, command, factory, repo: seen.update(
                            args=args, command=command, repo=repo) or 'completion')
    monkeypatch.setattr(subject, 'certify_outputs',
                        lambda run, completion: seen.update(certified=(run, completion)))
    argv = ['--backbone', 'simple', '--checkpoint', 'c', '--manifest', 'm',
            '--manifest-hash', 'h', '--out-dir', 'o', '--log-dir', 'l', '--data-root', 'd',
            '--run-label', 'r', '--reviewed-commit', 'HEAD', '--num-shot', '8',
            '--checkpoint-epoch', '12']
    assert subject.main(argv) == 'completion'
    assert seen['command'][1] == str(ROOT / 'tools/exp06_eval.py')
    assert seen['repo'] == ROOT and seen['args'].num_shot == 8
    assert seen['certified'] == (Path(seen['args'].out_dir), 'completion')


# --- the finished outputs must be this arm's, or nothing stands (round 2b finding 2) ---

OUTPUT_FIELDS = ('model_class', 'registry_sha256', 'checkpoint_role', 'checkpoint_epoch',
                 'heading', 'frame')


@pytest.fixture
def certified(tmp_path):
    """A finished run: the manifest, both outputs, and the hashes completion captured."""
    run = tmp_path / 'run'
    run.mkdir()
    fields = {'schema_version': 1, 'conditions': ['P'], 'n_samples': 16,
              'model_class': 'xRIR_CylOriented', 'registry_sha256': 'r' * 64,
              'checkpoint_role': 'arm', 'checkpoint_epoch': 12, 'heading': None,
              'frame': 'room'}
    digest = p.write_manifest(run / 'eval_manifest.json', fields)
    meta = dict({key: fields[key] for key in OUTPUT_FIELDS}, conditions=fields['conditions'],
                n_samples=fields['n_samples'], eval_manifest_sha256=digest)
    for name in inherited.OUTPUTS:
        (run / name).write_text(json.dumps({'meta': meta, 'query': []}))
    completion = {'schema_version': 1, 'eval_manifest_sha256': digest,
                  'outputs': {name: p.sha256_file(run / name) for name in inherited.OUTPUTS}}
    p.write_completion(run / 'completion.json', completion)
    return run, completion


def persist(run, completion):
    """Republish the on-disk completion after a test has changed the returned mapping."""
    (run / 'completion.json').unlink()
    p.write_completion(run / 'completion.json', completion)


def test_intact_outputs_pass_the_post_run_check(certified):
    run, completion = certified
    assert subject.certify_outputs(run, completion) is None
    assert run.is_dir() and not list(run.parent.glob('*_QUARANTINED_*'))


@pytest.mark.parametrize('name', list(inherited.OUTPUTS))
@pytest.mark.parametrize('field', OUTPUT_FIELDS)
@pytest.mark.parametrize('damage', ['corrupt', 'remove'])
def test_an_output_contradicting_the_manifest_is_quarantined(certified, name, field, damage):
    """Corruption before completion captured the hashes: the inherited check cannot see it."""
    run, completion = certified
    body = json.loads((run / name).read_text())
    if damage == 'remove':
        body['meta'].pop(field)
    else:
        body['meta'][field] = 'not what the manifest declares'
    (run / name).write_text(json.dumps(body))
    completion['outputs'][name] = p.sha256_file(run / name)
    persist(run, completion)
    quarantined = subject.certify_outputs(run, completion)
    assert quarantined is not None and not run.exists()
    assert quarantined.name.startswith(run.name + '_QUARANTINED_')
    record = json.loads((quarantined / 'quarantine.json').read_text())
    assert record['reason'] in quarantined.name
    assert name in record['detail'] and field in record['detail']
    assert record['run_dir'] == str(run) and (quarantined / name).is_file()


@pytest.mark.parametrize('damage', ['output_bytes', 'manifest_bytes', 'output_gone',
                                    'no_meta'])
def test_a_run_whose_bound_bytes_moved_is_quarantined(certified, damage):
    """The completion's own hashes are re-checked here, over the same two files."""
    run, completion = certified
    name = inherited.OUTPUTS[0]
    if damage == 'output_bytes':
        (run / name).write_text(json.dumps({'meta': {}, 'appended': True}))
    elif damage == 'manifest_bytes':
        (run / 'eval_manifest.json').write_text('{"schema_version": 1}')
    elif damage == 'output_gone':
        (run / name).unlink()
    else:
        (run / name).write_text(json.dumps({'query': []}))
        completion['outputs'][name] = p.sha256_file(run / name)
        persist(run, completion)
    quarantined = subject.certify_outputs(run, completion)
    assert quarantined is not None and not run.exists()
    assert json.loads((quarantined / 'quarantine.json').read_text())['reason']


# --- the gate reads the record the run published, not the caller's dict (nit 7) ---


@pytest.mark.parametrize('name', list(inherited.OUTPUTS))
def test_a_tampered_persisted_completion_is_quarantined(certified, name):
    """The returned mapping still binds the right bytes; only the retained record moved."""
    run, completion = certified
    persist(run, dict(completion, outputs=dict(completion['outputs'], **{name: 'f' * 64})))
    quarantined = subject.certify_outputs(run, completion)
    assert quarantined is not None and not run.exists()
    record = json.loads((quarantined / 'quarantine.json').read_text())
    assert record['reason'] == 'completion_mismatch' and record['reason'] in quarantined.name
    assert (quarantined / 'completion.json').is_file()


def test_an_unserialisable_returned_record_is_quarantined(certified):
    """Nothing the caller hands over can make the comparison itself raise."""
    run, completion = certified
    quarantined = subject.certify_outputs(run, dict(completion, outputs={'x'}))
    assert quarantined is not None and not run.exists()
    assert json.loads((quarantined / 'quarantine.json').read_text())['reason'] \
        == 'completion_mismatch'


@pytest.mark.parametrize('damage', ['missing', 'unparsable', 'not_an_object'])
def test_an_unreadable_persisted_completion_is_quarantined(certified, damage):
    run, completion = certified
    if damage == 'missing':
        (run / 'completion.json').unlink()
    else:
        (run / 'completion.json').write_text('[]' if damage == 'not_an_object' else '{nope')
    quarantined = subject.certify_outputs(run, completion)
    assert quarantined is not None and not run.exists()
    record = json.loads((quarantined / 'quarantine.json').read_text())
    assert record['reason'] in ('completion_unreadable', 'completion_mismatch')
    assert record['reason'] in quarantined.name


def test_the_persisted_record_is_the_one_whose_hashes_are_checked(certified):
    """A stale hash in the caller's dict cannot excuse the bytes the run published."""
    run, completion = certified
    name = inherited.OUTPUTS[0]
    (run / name).write_text(json.dumps({'meta': {}, 'rewritten': True}))
    persist(run, dict(completion, outputs=dict(completion['outputs'],
                                               **{name: p.sha256_file(run / name)})))
    quarantined = subject.certify_outputs(run, completion)
    assert quarantined is not None and not run.exists()
    assert json.loads((quarantined / 'quarantine.json').read_text())['reason'] \
        == 'completion_mismatch'


def test_main_fails_when_the_outputs_are_quarantined(launch_args, monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(subject.launcher, 'execute_run',
                        lambda *arguments: {'outputs': {}, 'eval_manifest_sha256': 'x'})
    monkeypatch.setattr(subject, 'certify_outputs',
                        lambda run, completion: seen.setdefault('run', run))
    argv = ['--backbone', 'simple', '--checkpoint', 'c', '--manifest', 'm',
            '--manifest-hash', 'h', '--out-dir', str(tmp_path / 'o'), '--log-dir', 'l',
            '--data-root', 'd', '--run-label', 'r', '--reviewed-commit', 'HEAD',
            '--num-shot', '8', '--checkpoint-epoch', '12']
    with pytest.raises(SystemExit) as failure:
        subject.main(argv)
    assert failure.value.code not in (0, None)
    assert seen['run'] == Path(str(tmp_path / 'o'))
