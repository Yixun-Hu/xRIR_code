"""The runbook's own simulated-evaluation argv, end to end (Codex full review F2(d)).

The review's blocker was a composition failure, not a unit one: every flag the runbook
generates parses, and every check the comparer runs is implemented, but the generated argv
bound no training evidence, so each of the ten registered runs would have been produced and
then refused. This test builds the argv the runbook's `sim` step writes -- both arms, with
the two `--bind-input` flags the post-review round adds -- pushes it through
``exp06_eval_launch.parse_args / split_bindings / training_evidence / build_fields /
check_fields`` with no GPU and no child, then writes the run directory the way the comparer
fixtures do and requires ``exp06_compare.admit_run`` to admit it.
"""
import json
import subprocess
from pathlib import Path

import pytest

from tools import exp06_compare as comparer
from tools import exp06_eval as evaluator
from tools import exp06_eval_launch as launcher
from tools import provenance as p

from test_exp06_compare import (ROLES, SPLIT, approved_digests, checkpoints,  # noqa: F401
                                training_inputs, write_run)

# tools/exp06_eval_launch.py ... (the runbook's `sim` step, one line per arm and seed)
ARMS = {'C': ('cyl_or', 'cylindrical_oriented', 'arm'),
        'B': ('cyl', 'cylindrical', 'baseline')}
REPO = Path(__file__).resolve().parents[1]
HEAD = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO), text=True).strip()


def runbook_argv(role, seed, checkpoint, manifest, manifest_hash, out_dir, log_dir,
                 data_root, commit, bindings):
    """Exactly the flags `posttrain_runbook.sh` step `sim` writes, plus F2's bindings."""
    name, backbone, checkpoint_role = ARMS[role]
    argv = ['--backbone', backbone, '--checkpoint', str(checkpoint),
            '--checkpoint-role', checkpoint_role, '--checkpoint-epoch', '12',
            '--manifest', str(manifest), '--manifest-hash', manifest_hash,
            '--out-dir', str(out_dir), '--log-dir', str(log_dir),
            '--data-root', str(data_root), '--run-label', '{}_seed{}'.format(name, seed),
            '--reviewed-commit', commit, '--num-shot', '8', '--conditions', 'P',
            '--yaw-cols', '0', '--acoustic-cols', '0', '--e-acoustic-cols',
            '--batch-size', '16', '--num-workers', '6', '--max-samples', '0',
            '--gl-seed', str(seed), '--threads', '4', '--log-interval', '10',
            '--decomposition-batches', '0', '--gpu', '1']
    for name, path in sorted(bindings.items()):
        argv += ['--bind-input', '{}={}'.format(name, path)]
    return argv


@pytest.fixture
def approved(checkpoints):                                          # noqa: F811
    return approved_digests(p.sha256_file(checkpoints['C']))


@pytest.fixture(autouse=True)
def registered_baseline(checkpoints, monkeypatch):                  # noqa: F811
    """R2: arm B's registration, on the synthetic exp_01 weights this fixture stands for."""
    from tools import exp04_profiles
    from tools import exp06_train_evidence as evidence
    monkeypatch.setattr(evidence, 'REGISTERED', {'baseline': dict(
        exp04_profiles.CYL, sha256=p.sha256_file(checkpoints['B']))})


@pytest.fixture(autouse=True)
def approvals_gate(monkeypatch):
    """F3's producer gate has its own tests; here the runbook's argv is under test."""
    monkeypatch.setattr(launcher, 'approvals_receipt', lambda args, repo: {
        'producer': 'sim_eval', 'keys_checked': [], 'deviations': [], 'artifacts': {},
        'approvals': {'path': 'approved_digests.json', 'sha256': 'a' * 64,
                      'committed_at': 'c' * 40},
        'exploratory': False, 'admissibility': 'confirmatory'})


@pytest.fixture
def bound(monkeypatch):
    """exp_04's fixture pattern: real field assembly, stubbed git, closures and inventory."""
    monkeypatch.setattr(p, 'source_closure', lambda module, repo: [module + '.py'])
    monkeypatch.setattr(p, 'closure_record', lambda files, commit, repo: (
        [{'path': path, 'reviewed_blob_sha256': 'sha', 'working_tree_sha256': 'sha',
          'commits_after_reviewed': []} for path in files], 'closure-sha'))
    monkeypatch.setattr(p, 'data_identity', lambda path, root: {'inventory_sha256': 'data-sha'})
    monkeypatch.setattr(p, 'git_state', lambda repo: {'dirty_outside_worklog': False})
    from tools import exp04_eval_launch as inherited
    monkeypatch.setattr(inherited, '_capture_environment', lambda *a: {'python': '3.8'})


def bindings_for(checkpoints, role):                                # noqa: F811
    return {name: record['path']
            for name, record in training_inputs(checkpoints, role).items()}


def argv_for(role, checkpoints, tmp_path, seed=42, bindings=None):  # noqa: F811
    reference = tmp_path / 'reference_manifest_k8_seed{}.json'.format(seed)
    from test_exp06_compare import reference_manifest, reference_hash
    manifest = reference_manifest(reference, seed, SPLIT)
    return runbook_argv(role, seed, checkpoints['C' if role == 'C' else 'B'], reference,
                        reference_hash(manifest), tmp_path / 'run', tmp_path / 'logs',
                        checkpoints['repo'], HEAD,
                        bindings_for(checkpoints, role) if bindings is None else bindings)


@pytest.mark.parametrize('role', ['C', 'B'])
def test_the_runbook_argv_parses_to_the_registered_evaluation(role, checkpoints,  # noqa: F811
                                                              tmp_path):
    args = launcher.parse_args(argv_for(role, checkpoints, tmp_path))
    assert args.backbone == ARMS[role][1] and args.checkpoint_role == ARMS[role][2]
    assert args.checkpoint_epoch == 12 and args.num_shot == 8
    assert args.conditions == 'P' and args.yaw_cols == [0] and args.e_acoustic_cols == []
    assert args.batch_size == 16 and args.tf32 is False and args.max_samples == 0
    assert args.gl_seed == 42 and args.decomposition_batches == 0
    assert evaluator.exp06_metadata(args)['frame'] == 'room'
    assert evaluator.exp06_metadata(args)['heading'] is None
    inherited, own = launcher.split_bindings(args)
    assert own == {} and len(inherited) == 2          # both are exp_04's binding names
    assert sorted(item.split('=')[0] for item in inherited) == \
        sorted(comparer.TRAINING_BINDINGS)


@pytest.mark.parametrize('role', ['C', 'B'])
def test_the_runbook_argv_builds_fields_that_the_comparer_admits(role, checkpoints,  # noqa: F811
                                                                 tmp_path, approved, bound):
    args = launcher.parse_args(argv_for(role, checkpoints, tmp_path))
    training = launcher.training_evidence(args)
    assert training['route'] == ('exp06_full' if role == 'C' else 'reconstructed')
    assert training['epochs'] == 12
    # The fields are built against the real repository (the reviewed commit must exist);
    # the run directory below is the comparer fixture's synthetic tree.
    fields = launcher.build_fields(args, launcher.child_command(args, REPO), REPO, training)
    assert launcher.check_fields(args, fields) is fields
    assert fields['training_evidence'] == training
    assert set(fields['mutable_inputs']) == set(comparer.TRAINING_BINDINGS)
    assert fields['checkpoint_role'] == ARMS[role][2] and fields['checkpoint_epoch'] == 12
    # The same evidence, in a run directory shaped like the ones the evaluator writes.
    directory = write_run(tmp_path / 'admit' / role, role, 42, checkpoints, SPLIT,
                          route='exp06')
    recorded = json.loads((directory / 'eval_manifest.json').read_text())
    assert recorded['mutable_inputs'] == fields['mutable_inputs']
    run = comparer.admit_run(directory, role, approved, SPLIT, roles=ROLES)
    assert run['role'] == role and run['checkpoint']['route'] == 'exp06'
    assert run['checkpoint']['epoch'] == 12


@pytest.mark.parametrize('role', ['C', 'B'])
@pytest.mark.parametrize('name', ['train_manifest', 'train_completion'])
def test_the_runbooks_original_argv_is_refused(role, name, checkpoints, tmp_path):  # noqa: F811
    """The generated commands bound nothing at all; one missing name is already a refusal."""
    bindings = bindings_for(checkpoints, role)
    bindings.pop(name)
    args = launcher.parse_args(argv_for(role, checkpoints, tmp_path, bindings=bindings))
    with pytest.raises(ValueError, match='binds its training run'):
        launcher.training_evidence(args)
    empty = launcher.parse_args(argv_for(role, checkpoints, tmp_path, bindings={}))
    with pytest.raises(ValueError, match='binds its training run'):
        launcher.training_evidence(empty)


def test_the_arms_may_not_swap_their_training_evidence(checkpoints, tmp_path):  # noqa: F811
    """C's completion does not certify the exp_01 weights, and B's receipt is not C's."""
    args = launcher.parse_args(argv_for('C', checkpoints, tmp_path,
                                        bindings=bindings_for(checkpoints, 'B')))
    with pytest.raises(ValueError, match='train_completion records run_type'):
        launcher.training_evidence(args)
    other = launcher.parse_args(argv_for('B', checkpoints, tmp_path,
                                         bindings=bindings_for(checkpoints, 'C')))
    with pytest.raises(ValueError, match='labelled'):
        launcher.training_evidence(other)


def test_a_non_admissible_completion_never_launches_an_arm(checkpoints, tmp_path):  # noqa: F811
    completion = Path(checkpoints['train_C']) / 'completion.json'
    original = completion.read_text()
    completion.write_text(json.dumps(dict(json.loads(original), admissible_arm=False)))
    try:
        args = launcher.parse_args(argv_for('C', checkpoints, tmp_path))
        with pytest.raises(ValueError, match='not admissible as an arm'):
            launcher.training_evidence(args)
    finally:
        completion.write_text(original)


def test_a_receipt_with_a_stale_artefact_never_launches_arm_b(checkpoints, tmp_path):  # noqa: F811
    train, _ = checkpoints['train_B']
    original = (train / 'history.jsonl').read_bytes()
    (train / 'history.jsonl').write_bytes(original + b'{"epoch": 13}\n')
    try:
        args = launcher.parse_args(argv_for('B', checkpoints, tmp_path))
        with pytest.raises(ValueError, match='changed since the receipt'):
            launcher.training_evidence(args)
    finally:
        (train / 'history.jsonl').write_bytes(original)
