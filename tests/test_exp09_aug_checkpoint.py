"""Arm E starts from the checkpoint exp_04 approved, and from nothing else.

``tools/exp06_finalize.exp04_aug_checkpoint`` is the one place exp_09 resolves its
initialisation. It lives in the finalizer rather than in the shared approvals module
(code review round 1, finding 1) because that module is inside the closures the ten
completed exp_06 simulated evaluations are verified against; ``tests/
test_exp09_sim_eval_closures.py`` guards that. What it must establish is here: exp_04's
own approvals record is the authority for a digest ``tools/exp04_profiles.py`` does not
carry, and only the record whose bytes are exp_06's approved
``reused.exp04_approved_digests_sha256`` may name it.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from tools import exp06_approvals_api as approvals
from tools import exp06_finalize as subject

HEX = 'a' * 64


def filled(digest=HEX):
    """A complete exp_06 approvals record: only its reused exp_04 pin matters here."""
    return {'schema_version': 1,
            'code': {key: digest for key in approvals.CODE_KEYS},
            'reused': dict({key: digest for key in approvals.REUSED_DIGESTS},
                           legacy_receipt={'path': 'ckpt/exp06/legacy_receipt.json',
                                           'sha256': digest}),
            'artifacts': {'epoch_012': {'path': 'ckpt/exp06/final/epoch_012.pth',
                                        'epoch': 12, 'sha256': digest},
                          'heading': {room: digest for room in approvals.ROOMS},
                          'gate_g1_sha256': digest}}


EXP04_CLOSURES = ('evaluator', 'writer', 'training_launcher', 'producer_paired_compare',
                  'producer_results_table', 'producer_descriptive')
AUG_PATH = 'ckpt/xRIR_simple_yawaug_8_shot/final/epoch_012.pth'


def exp04_record(tmp_path, sha256='f' * 64, epoch=12, path=AUG_PATH):
    """An exp_04-shaped approvals record, committed as exp_04's own record is.

    exp_04's loader admits an all-null template or an all-filled record, never a mixture,
    so a record without the aug digest is the template it approves nothing with.
    """
    root = tmp_path / 'exp04'
    (root / 'assets').mkdir(parents=True)
    file = root / 'assets/approved_digests.json'
    filled_in = sha256 is not None
    file.write_text(json.dumps(
        {'schema_version': 1 if filled_in else None,
         'closures': {key: ('d' * 64 if filled_in else None) for key in EXP04_CLOSURES},
         'checkpoints': {'aug_epoch9': 'e' * 64 if filled_in else None,
                         'aug': {'path': path if filled_in else None,
                                 'epoch': epoch if filled_in else None,
                                 'sha256': sha256}}},
        sort_keys=True, indent=2) + '\n')
    for command in (['init', '-q'], ['add', '-A'],
                    ['-c', 'user.email=a@b', '-c', 'user.name=t', 'commit', '-q', '-m', 'x']):
        subprocess.run(['git'] + command, cwd=str(root), check=True)
    return file


def approvals_pinning(file, digest='the record itself'):
    """exp_06 approvals whose reused exp_04 pin is that file -- or something else."""
    value = filled()
    value['reused']['exp04_approved_digests_sha256'] = (
        hashlib.sha256(Path(file).read_bytes()).hexdigest()
        if digest == 'the record itself' else digest)
    return approvals.validate(value)


def test_the_yawaug_initialisation_comes_from_exp04s_approved_checkpoint(tmp_path):
    """exp_04's profile carries no digest for it, so its approvals record is the authority."""
    file = exp04_record(tmp_path)
    record = subject.exp04_aug_checkpoint(approvals_pinning(file), file)
    assert record['path'] == str(file)
    assert record['sha256'] == approvals_pinning(file)['reused']['exp04_approved_digests_sha256']
    assert record['checkpoint'] == {'path': AUG_PATH, 'epoch': 12, 'sha256': 'f' * 64}


@pytest.mark.parametrize('digest', [HEX, None])
def test_exp04_approvals_that_are_not_the_reused_identity_are_refused(tmp_path, digest):
    """The reused pin is what makes an unreviewed file unable to name a checkpoint."""
    file = exp04_record(tmp_path)
    with pytest.raises(ValueError, match=r'reused\.exp04_approved_digests_sha256'):
        subject.exp04_aug_checkpoint(approvals_pinning(file, digest), file)


@pytest.mark.parametrize('epoch,sha256', [(9, 'f' * 64), (12, None)])
def test_an_aug_record_of_another_epoch_or_without_a_digest_is_refused(tmp_path, epoch,
                                                                       sha256):
    """Only the twelfth epoch exp_04 certified may initialise arm E."""
    file = exp04_record(tmp_path, sha256=sha256, epoch=epoch)
    with pytest.raises(ValueError, match=r'checkpoints\.aug'):
        subject.exp04_aug_checkpoint(approvals_pinning(file), file)


def test_the_real_exp04_record_names_the_yaw_augmented_checkpoint():
    """The production path: exp_06's approved reused pin admits exp_04's own record."""
    from tools import exp04_profiles
    file = exp04_profiles.APPROVED_DIGESTS_PATH
    record = subject.exp04_aug_checkpoint(approvals_pinning(file), file)
    assert record['checkpoint']['path'] == AUG_PATH and record['checkpoint']['epoch'] == 12
    assert re.fullmatch('[0-9a-f]{64}', record['checkpoint']['sha256'])
