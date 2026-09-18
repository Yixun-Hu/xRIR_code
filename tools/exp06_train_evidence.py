"""The training run behind an evaluated checkpoint (plan 6.3/6.4, Codex full review F2).

``tools.exp06_compare`` requires every exp_06-route evaluation to bind ``train_manifest``
and ``train_completion``, and 6.3 promises the launcher validates them. It did not: exp_04's
launcher merely hashes whatever file a ``--bind-input`` names, so any file at all satisfied
the names, and the comparer's own fixture bound an evaluation log to them.

This module is the semantics, shared by the launcher (before the child runs) and the
comparer (at admission), so one rule decides both:

* ``arm`` -- the exp_06 pretraining. ``train_completion`` is the finalizer's
  ``completion.json``: ``run_type == 'full'``, ``admissible_arm``, not diagnostic or
  exploratory, twelve epochs, and an ``epoch_012.pth`` artifact whose hash **is** the
  evaluated checkpoint. ``train_manifest`` is that run's ``provenance.json``, the bytes the
  completion bound, recording the same ``run_type`` and the reviewed commit the completion
  was approved at. A ``best.pth`` or any other epoch offered as the checkpoint fails on the
  artifact hash.
* ``baseline`` -- the exp_01 checkpoint. ``train_completion`` is the ``reconstructed``
  receipt of ``tools.exp06_legacy_train_receipt`` and ``train_manifest`` the ``args.json``
  that receipt enumerates: the historical run's own startup record, so the two names keep
  the meaning they have for arm C (what the run declared, and what certified it).
* ``diagnostic`` -- never admissible as an arm, and binds nothing.
"""
import json
from pathlib import Path

from tools import exp06_legacy_train_receipt as receipt_tool
from tools import provenance

TRAINING_BINDINGS = ('train_manifest', 'train_completion')
EPOCHS = 12                        # section 5's twelve-epoch contract, shared by both arms
EPOCH_ARTIFACT = 'epoch_012.pth'
FULL_RUN_TYPE = 'full'
ARM_ROUTE, BASELINE_ROUTE = 'exp06_full', 'reconstructed'
ROLES = ('arm', 'baseline', 'diagnostic')


def _require(ok, cause):
    if not ok:
        raise ValueError(cause)


def _read_json(path, label):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError('unreadable {} ({}): {}'.format(label, path, error))
    _require(isinstance(value, dict), '{} ({}) is not a record'.format(label, path))
    return value


def _bound(bindings, role):
    """Both names are required, and each must name a readable file."""
    missing = [name for name in TRAINING_BINDINGS if not (bindings or {}).get(name)]
    _require(not missing, 'the {} checkpoint binds its training run: --bind-input {}'.format(
        role, ', '.join('{}=<path>'.format(name) for name in missing)))
    resolved = {}
    for name in TRAINING_BINDINGS:
        path = Path(bindings[name]).resolve()
        _require(path.is_file(), 'the {} binding names no file: {}'.format(name, path))
        resolved[name] = path
    return resolved


def check_arm(bindings, checkpoint_sha256, epoch_sha256=None, epochs=EPOCHS,
              hasher=provenance.sha256_file):
    """Arm C: the finalised full pretraining attempt whose twelfth epoch is these weights."""
    paths = _bound(bindings, 'arm')
    completion = _read_json(paths['train_completion'], 'train_completion')
    _require(completion.get('run_type') == FULL_RUN_TYPE,
             'train_completion records run_type {!r}, not {!r}'.format(
                 completion.get('run_type'), FULL_RUN_TYPE))
    _require(completion.get('admissible_arm') is True,
             'train_completion is not admissible as an arm')
    _require(completion.get('diagnostic') is False and completion.get('exploratory') is False,
             'train_completion is a diagnostic or exploratory run')
    _require(completion.get('epochs') == epochs,
             'train_completion records {!r} epochs, not the registered {}'.format(
                 completion.get('epochs'), epochs))
    artifacts = completion.get('artifacts')
    _require(isinstance(artifacts, dict) and EPOCH_ARTIFACT in artifacts,
             'train_completion binds no ' + EPOCH_ARTIFACT)
    _require(artifacts[EPOCH_ARTIFACT] == checkpoint_sha256,
             'the evaluated checkpoint hashes to {}, not the {} of the training run\'s '
             '{}'.format(checkpoint_sha256, artifacts[EPOCH_ARTIFACT], EPOCH_ARTIFACT))
    checkpoint = completion.get('checkpoint')
    _require(isinstance(checkpoint, dict) and checkpoint.get('sha256') == checkpoint_sha256
             and checkpoint.get('path') == EPOCH_ARTIFACT
             and checkpoint.get('epoch') == epochs,
             'train_completion certifies the checkpoint {!r}, not the twelfth epoch these '
             'weights are'.format(checkpoint))
    if epoch_sha256 is not None:
        _require(checkpoint_sha256 == epoch_sha256, 'the evaluated checkpoint is not the '
                 'approved artifacts.epoch_012 {}'.format(epoch_sha256))
    run_dir = Path(completion.get('run_dir') or '')
    _require(paths['train_completion'].parent == run_dir.resolve(),
             'train_completion {} is not the completion of the run directory it names '
             '({})'.format(paths['train_completion'], run_dir))
    _require(paths['train_manifest'] == (run_dir / 'provenance.json').resolve(),
             'train_manifest {} is not the provenance.json of the training run {}'.format(
                 paths['train_manifest'], run_dir))
    digest = hasher(str(paths['train_manifest']))
    _require(digest == artifacts.get('provenance.json'),
             'train_manifest hashes to {}, not the {} the completion bound'.format(
                 digest, artifacts.get('provenance.json')))
    record = _read_json(paths['train_manifest'], 'train_manifest')
    _require(record.get('run_type') == completion['run_type'],
             'train_manifest records run_type {!r}, not the {!r} of the completion'.format(
                 record.get('run_type'), completion['run_type']))
    reviewed = (completion.get('approvals') or {}).get('committed_at')
    _require(isinstance(reviewed, str) and reviewed,
             'train_completion records no approvals commit the training was reviewed at')
    _require(record.get('reviewed_commit') == reviewed,
             'train_manifest was reviewed at {!r}, not the {!r} of the completion'.format(
                 record.get('reviewed_commit'), reviewed))
    return {'route': ARM_ROUTE, 'role': 'arm', 'epochs': epochs, 'run_dir': str(run_dir),
            'reviewed_commit': reviewed, 'checkpoint_sha256': checkpoint_sha256,
            'train_manifest': {'path': str(paths['train_manifest']), 'sha256': digest},
            'train_completion': {'path': str(paths['train_completion']),
                                 'sha256': hasher(str(paths['train_completion']))}}


def check_baseline(bindings, checkpoint_sha256, registered_sha256=None, epochs=EPOCHS,
                   hasher=provenance.sha256_file):
    """Arm B: the reconstructed receipt of the exp_01 training, and its own args.json."""
    paths = _bound(bindings, 'baseline')
    record = receipt_tool.verify(paths['train_completion'],
                                 checkpoint_sha256=checkpoint_sha256,
                                 registered_sha256=registered_sha256, epochs=epochs,
                                 hasher=hasher)
    train_dir = Path(record['train_dir'])
    _require(paths['train_manifest'] == (train_dir / 'args.json').resolve(),
             'train_manifest {} is not the args.json the receipt enumerates ({})'.format(
                 paths['train_manifest'], train_dir / 'args.json'))
    digest = hasher(str(paths['train_manifest']))
    _require(digest == record['args']['sha256'],
             'train_manifest hashes to {}, not the {} the receipt enumerates'.format(
                 digest, record['args']['sha256']))
    return {'route': BASELINE_ROUTE, 'role': 'baseline', 'epochs': record['epochs'],
            'run_dir': str(train_dir), 'arm': record['arm'],
            'checkpoint_sha256': record['checkpoint']['sha256'],
            'train_manifest': {'path': str(paths['train_manifest']), 'sha256': digest},
            'train_completion': {'path': record['path'], 'sha256': record['sha256']},
            'inputs': record['inputs']}


def check(role, bindings, checkpoint_sha256, epoch_sha256=None, registered_sha256=None,
          epochs=EPOCHS, hasher=provenance.sha256_file):
    """6.3's two admissible training-evidence branches, by the checkpoint's declared role."""
    _require(role in ROLES, 'unknown checkpoint role: {!r}'.format(role))
    if role == 'arm':
        return check_arm(bindings, checkpoint_sha256, epoch_sha256, epochs, hasher)
    if role == 'baseline':
        return check_baseline(bindings, checkpoint_sha256, registered_sha256, epochs, hasher)
    return None                      # a diagnostic evaluation is never admissible as an arm
