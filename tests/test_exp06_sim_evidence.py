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
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools import exp06_approvals_api as approvals_api
from tools import exp06_compare as comparer
from tools import exp06_eval as evaluator
from tools import exp06_eval_launch as launcher
from tools import exp06_legacy_train_receipt as receipt_tool
from tools import provenance as p

from test_exp06_compare import (ROLES, SPLIT, approved_digests, checkpoints,  # noqa: F401
                                training_inputs, write_run)
from test_exp06_legacy_train_receipt import (PRODUCER, altered, closure_digest,  # noqa: F401
                                            git_object, rewritten)

# tools/exp06_eval_launch.py ... (the runbook's `sim` step, one line per arm and seed)
ARMS = {'C': ('cyl_or', 'cylindrical_oriented', 'arm'),
        'B': ('cyl', 'cylindrical', 'baseline')}
REPO = Path(__file__).resolve().parents[1]
HEAD = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO), text=True).strip()


def runbook_argv(role, seed, checkpoint, manifest, manifest_hash, out_dir, log_dir,
                 data_root, commit, bindings, approved=None):
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
    # R4: the runbook names the approvals and the commit they are committed at; the
    # defaults were what the helper exercised.
    return argv + ['--approved', approved or approvals_api.approved_path_default(),
                   '--approved-commit', commit]


ENTRY_FILES = {'eval_yaw_rotation': 'eval_yaw_rotation.py',
               'tools.exp06_eval': 'tools/exp06_eval.py',
               'tools.exp04_eval_launch': 'tools/exp04_eval_launch.py',
               'tools.exp06_eval_launch': 'tools/exp06_eval_launch.py'}


def closure_of(files, commit, repo):
    """`closure_record`'s shape over the real bytes of the files a closure names.

    R4: the manifest the launcher builds is the one admitted, so its closures must be
    self-consistent and re-bindable -- the comparer re-hashes every file they record.
    """
    records = [{'path': name, 'reviewed_blob_sha256': p.sha256_file(Path(repo) / name),
                'working_tree_sha256': p.sha256_file(Path(repo) / name),
                'commits_after_reviewed': []} for name in sorted(set(files))]
    return records, hashlib.sha256(json.dumps(
        [[item['path'], item['reviewed_blob_sha256']] for item in records],
        sort_keys=True).encode()).hexdigest()


@pytest.fixture
def approved(checkpoints):                                          # noqa: F811
    """The approvals of 6.4, filled with the closures this launcher really records."""
    value = approved_digests(p.sha256_file(checkpoints['C']))
    value['code'] = dict(value['code'], **{
        key: closure_of([ENTRY_FILES[module]], None, REPO)[1] for key, module in
        (('eval', 'tools.exp06_eval'), ('eval_launch', 'tools.exp06_eval_launch'),
         ('evaluator_exp03', 'eval_yaw_rotation'))})
    return value


REAL_APPROVALS_RECEIPT = launcher.approvals_receipt


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
def bound(monkeypatch, checkpoints):                                # noqa: F811
    """Real field assembly; only git state and the environment capture are stubbed.

    The closures are the real files of this repository and the data identity is the
    fixture split's, so the fields the launcher builds are admissible evidence rather
    than placeholders the comparer would have to be told to ignore.
    """
    from test_exp06_compare import data_identity
    monkeypatch.setattr(p, 'source_closure', lambda module, repo: [ENTRY_FILES[module]])
    monkeypatch.setattr(p, 'closure_record', closure_of)
    monkeypatch.setattr(p, 'data_identity', lambda path, root: data_identity(
        checkpoints['repo'], path, json.loads(Path(path).read_text()), p.sha256_file(path)))
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
    # R4: these fields are the evaluation manifest, and the outputs and completion are
    # written around them, so the comparer admits what the launcher would have written.
    directory = write_run(tmp_path / 'admit' / role, role, 42, checkpoints, SPLIT,
                          route='exp06', launched=fields)
    recorded = json.loads((directory / 'eval_manifest.json').read_text())
    assert recorded == json.loads(json.dumps(fields, sort_keys=True))
    run = comparer.admit_run(directory, role, approved, SPLIT, roles=ROLES)
    assert run['role'] == role and run['checkpoint']['route'] == 'exp06'
    assert run['checkpoint']['epoch'] == 12
    assert run['checkpoint']['sha256'] == fields['checkpoint_sha256']


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


def test_the_runbook_names_the_approvals_the_producer_gate_reads(checkpoints,  # noqa: F811
                                                                 tmp_path, monkeypatch):
    """R4: the runbook supplies --approved and --approved-commit, and both reach 6.4."""
    seen = {}
    monkeypatch.setattr(launcher.approvals_api, 'enforce_producer',
                        lambda *a, **k: seen.update(a=a, k=k) or {'deviations': []})
    args = launcher.parse_args(argv_for('C', checkpoints, tmp_path))
    assert args.approved == str(Path(approvals_api.approved_path_default()).resolve())
    assert args.approved_commit == HEAD
    assert REAL_APPROVALS_RECEIPT(args, REPO) == {'deviations': []}
    assert seen['a'] == ('sim_eval', REPO, HEAD)
    assert seen['k'] == {'approved_path': args.approved, 'checkpoint': args.checkpoint}
    # Arm B's weights are not the approved epoch_012, and its approvals may be named at
    # any reviewed commit the runbook passes.
    other = launcher.parse_args(argv_for('B', checkpoints, tmp_path))
    other.approved_commit = 'b' * 40
    REAL_APPROVALS_RECEIPT(other, REPO)
    assert seen['a'][2] == 'b' * 40 and seen['k']['checkpoint'] is None


# --- close review R3: arm B's producer evidence, at the launcher and at the comparer --------


REAL_TRAIN = REPO / 'ckpt/xRIR_cyl_8_shot'
REAL_CHECKPOINT = REAL_TRAIN / 'epoch_12.pth'
# Every producer closure Codex got past `verify`, and so past `build_fields` and `admit_run`.
CLOSURE_DAMAGE = {
    'strict_only': ({'strict': True}, 'produced by'),
    'wrong_producer': (dict(PRODUCER, entry_module='tools.exp06_finalize'), 'produced by'),
    'empty_membership': (dict(PRODUCER, files=[], sha256=closure_digest([])),
                         'no source file'),
    'corrupt_digest': (dict(PRODUCER, sha256='f' * 64), 'does not hash'),
    'contradictory': (altered(PRODUCER, working_tree_sha256='c' * 64),
                      'differs from its own sources'),
    'missing_reviewed': (altered(PRODUCER, reviewed_blob_sha256=None),
                         'no reviewed and working-tree hashes'),
    'unreviewed_blob': (altered(PRODUCER, reviewed_blob_sha256='c' * 64,
                                working_tree_sha256='c' * 64), 'reviewed source'),
    # Close review 2: the recorded commit was forty hex digits and nothing more, so the
    # repository's own tree object -- whose blobs are HEAD's -- was admitted as the producer.
    'tree_object': (dict(PRODUCER, commit=git_object(PRODUCER['commit'] + '^{tree}')),
                    'not a commit'),
    'blob_object': (dict(PRODUCER, commit=git_object(
        PRODUCER['commit'] + ':tools/exp06_legacy_train_receipt.py')), 'not a commit'),
}


def rebind(fields, name, path):
    """The launcher's own fields, with one training binding pointing at `path`."""
    record = {'path': str(path), 'sha256': p.sha256_file(path)}
    fields['mutable_inputs'] = dict(fields['mutable_inputs'], **{name: record})
    fields['training_evidence'] = dict(fields['training_evidence'], **{name: record})
    return fields


@pytest.mark.parametrize('case', sorted(CLOSURE_DAMAGE))
def test_a_receipt_without_producer_evidence_neither_launches_nor_admits_arm_b(
        case, checkpoints, tmp_path, approved, bound):                  # noqa: F811
    """R3: the `{'strict': true}` receipt passed `build_fields()` and `admit_run()` as B."""
    closure, cause = CLOSURE_DAMAGE[case]
    _, receipt = checkpoints['train_B']
    damaged = rewritten(json.loads(Path(receipt).read_text()), tmp_path / 'damaged.json',
                        source_closure=closure)
    bindings = dict(bindings_for(checkpoints, 'B'), train_completion=str(damaged))
    args = launcher.parse_args(argv_for('B', checkpoints, tmp_path, bindings=bindings))
    with pytest.raises(ValueError, match=cause):
        launcher.training_evidence(args)
    good = launcher.parse_args(argv_for('B', checkpoints, tmp_path))
    fields = launcher.build_fields(good, launcher.child_command(good, REPO), REPO,
                                   launcher.training_evidence(good))
    directory = write_run(tmp_path / 'admit', 'B', 42, checkpoints, SPLIT, route='exp06',
                          launched=rebind(fields, 'train_completion', damaged))
    with pytest.raises(ValueError, match='training evidence'):
        comparer.admit_run(directory, 'B', approved, SPLIT, roles=ROLES)


@pytest.fixture(scope='module')
def real_receipt(tmp_path_factory):
    """The writer's own default path -- strict, no stubs -- on the retained exp_01 run.

    Module scoped so that it is written before any test's `monkeypatch` stands in for
    `tools.provenance`: this is the receipt the runbook's `breceipt` step writes.
    """
    if not REAL_CHECKPOINT.is_file():
        pytest.skip("needs exp_01's retained cylindrical training directory")
    out = tmp_path_factory.mktemp('breceipt') / 'train_receipt_cyl.json'
    receipt_tool.write_receipt(out, REAL_TRAIN, REAL_CHECKPOINT)
    return out


def test_the_real_exp01_receipt_still_launches_and_admits_arm_b(real_receipt, checkpoints,
                                                                tmp_path, approved, bound,
                                                                monkeypatch):  # noqa: F811
    """R3 keeps the one receipt that is evidence: exp_01's, of its registered arm."""
    from tools import exp04_profiles
    from tools import exp06_train_evidence as evidence
    from test_exp06_compare import reference_manifest, reference_hash
    monkeypatch.setattr(evidence, 'REGISTERED', {'baseline': exp04_profiles.CYL})
    reference = tmp_path / 'reference_manifest_k8_seed42.json'
    declared = reference_manifest(reference, 42, SPLIT)
    args = launcher.parse_args(runbook_argv(
        'B', 42, REAL_CHECKPOINT, reference, reference_hash(declared), tmp_path / 'run',
        tmp_path / 'logs', checkpoints['repo'], HEAD,
        {'train_manifest': REAL_TRAIN / 'args.json', 'train_completion': real_receipt}))
    training = launcher.training_evidence(args)
    assert training['route'] == 'reconstructed' and training['epochs'] == 12
    assert training['checkpoint_sha256'] == exp04_profiles.CYL['sha256']
    fields = launcher.build_fields(args, launcher.child_command(args, REPO), REPO, training)
    directory = write_run(tmp_path / 'admit', 'B', 42, checkpoints, SPLIT, route='exp06',
                          launched=fields)
    roles = copy.deepcopy(ROLES)          # arm B is exp_01's registered cylindrical run
    roles['B']['checkpoint'] = dict(exp04_profiles.CYL)
    run = comparer.admit_run(directory, 'B', approved, SPLIT, roles=roles)
    assert run['role'] == 'B' and run['checkpoint']['route'] == 'exp06'
    assert run['checkpoint']['sha256'] == exp04_profiles.CYL['sha256']
