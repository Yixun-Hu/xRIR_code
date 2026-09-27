"""Blocker 2: every exp_11 approval value must be ENFORCED at its consuming gate.

Verifying that a record's bytes are the committed ones and that its fields are non-null
says nothing about whether those values describe what actually ran. Each value is
compared here with the artefact it names:

* ``code.summarize_haa`` with the producer closure this run is executing;
* ``reused.approved_digests_exp04`` with exp_04's approvals record, which
  ``exp04_aug_checkpoint`` reads to resolve arms E, G and K's initialisation;
* ``reused.approved_digests_exp06`` with exp_06's approvals record, which supplies the
  heading artefacts and arm C's epoch;
* ``reused.legacy_receipt`` (path **and** sha256) with the receipt the legacy branch
  actually consumed.

Every test here uses an **incorrect but populated** value: a null field is a different
failure and does not cover this one.
"""
import hashlib
import json
from pathlib import Path

import pytest

from tools import exp04_profiles
from tools import exp06_approvals_api as api
from tools import exp06_summarize_haa as subject
from tools import exp11_profiles

REPO = subject.REPO
WRONG = 'f' * 64


def consumed():
    """The three records exp_11's ``reused`` section names, as this checkout holds them."""
    return {'exp04': Path(exp04_profiles.APPROVED_DIGESTS_PATH),
            'exp06': Path(api.approved_path_default()),
            'receipt': REPO / 'ckpt/exp06/legacy_receipt.json'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def approvals(**overrides):
    """An exp_11 approvals value whose reused identities are the true ones by default."""
    paths = consumed()
    for name, path in paths.items():
        if not path.is_file():
            pytest.skip('missing consumed record {}'.format(path))
    value = {'schema_version': 1,
             'code': {key: WRONG for key in exp11_profiles.CODE_KEYS},
             'reused': {'approved_digests_exp04': sha(paths['exp04']),
                        'approved_digests_exp06': sha(paths['exp06']),
                        'legacy_receipt': {'path': 'ckpt/exp06/legacy_receipt.json',
                                           'sha256': sha(paths['receipt'])}},
             'artifacts': {key: {'epoch': 12, 'path': 'p', 'sha256': WRONG}
                           for key in exp11_profiles.ARTIFACT_KEYS}}
    value['code']['summarize_haa'] = subject.source_identity(REPO, strict=False)['sha256']
    for section, changes in overrides.items():
        value[section].update(changes)
    return value


def reused(approvals_value, receipt, receipt_path, exp06_receipt, **kwargs):
    """Call the gate with the exp_06 record ``main()`` actually loaded from ``--approved``.

    The gate hashed ``approved_path_default()`` instead, so exp_11's
    ``reused.approved_digests_exp06`` pin could match the default file while the
    summariser consumed a different committed record -- and every identity downstream
    (arm C's epoch, the heading artefacts) came from the record nobody pinned.
    """
    import inspect
    assert 'exp06_receipt' in inspect.signature(subject.check_exp11_reused).parameters, (
        'check_exp11_reused must be given the exp_06 approvals record main() loaded from '
        '--approved; it hashes approved_path_default() instead, so an alternate '
        'committed record passes the pin of the default one')
    return subject.check_exp11_reused(approvals_value, receipt, receipt_path,
                                      exp06_receipt=exp06_receipt, **kwargs)


def exp06_receipt_for(path):
    """The identity shape ``approvals_api.load_approved_digests`` returns for a record."""
    return {'path': str(path), 'sha256': sha(path)}


# --- code.summarize_haa against the producer that is running ------------------------


def test_the_producer_closure_is_checked_against_exp11s_own_key():
    identity = subject.source_identity(REPO, strict=False)
    assert subject.check_exp11_producer(approvals(), identity) == []
    wrong = approvals(code={'summarize_haa': WRONG})
    with pytest.raises(ValueError, match='not the approved exp_11 code.summarize_haa'):
        subject.check_exp11_producer(wrong, identity)
    assert subject.check_exp11_producer(wrong, identity, exploratory=True)
    assert subject.check_exp11_producer(None, identity) == []


def test_the_exp06_producer_check_is_unchanged():
    """The historical gate keeps comparing exp_06's own record; nothing is replaced."""
    identity = subject.source_identity(REPO, strict=False)
    approved = {'code': {'summarize_haa': identity['sha256']}}
    assert subject.check_producer_identity(approved, 'summarize_haa', identity) == []
    with pytest.raises(ValueError, match='not the approved code.summarize_haa'):
        subject.check_producer_identity({'code': {'summarize_haa': WRONG}},
                                        'summarize_haa', identity)


# --- the three reused identities against the records actually consumed --------------


def test_the_reused_identities_are_compared_with_the_consumed_records():
    paths = consumed()
    receipt = {'sha256': sha(paths['receipt'])}
    inputs = {}
    reused(approvals(), receipt, str(paths['receipt']),
           exp06_receipt_for(paths['exp06']), inputs=inputs)
    assert str(paths['exp04'].resolve()) in inputs
    assert str(paths['exp06'].resolve()) in inputs


@pytest.mark.parametrize('key', ['approved_digests_exp04', 'approved_digests_exp06'])
def test_an_incorrect_but_populated_reused_digest_refuses(key):
    paths = consumed()
    receipt = {'sha256': sha(paths['receipt'])}
    with pytest.raises(ValueError, match=key):
        reused(approvals(reused={key: WRONG}), receipt, str(paths['receipt']),
               exp06_receipt_for(paths['exp06']))


def test_an_incorrect_legacy_receipt_digest_or_a_nonexistent_path_refuses(tmp_path):
    paths = consumed()
    receipt = {'sha256': sha(paths['receipt'])}
    wrong_digest = approvals(reused={'legacy_receipt': {
        'path': 'ckpt/exp06/legacy_receipt.json', 'sha256': WRONG}})
    with pytest.raises(ValueError, match='legacy_receipt'):
        reused(wrong_digest, receipt, str(paths['receipt']),
               exp06_receipt_for(paths['exp06']))
    missing = approvals(reused={'legacy_receipt': {
        'path': 'ckpt/exp06/no_such_receipt.json', 'sha256': sha(paths['receipt'])}})
    with pytest.raises(ValueError, match='no_such_receipt'):
        reused(missing, receipt, str(paths['receipt']), exp06_receipt_for(paths['exp06']))
    elsewhere = approvals(reused={'legacy_receipt': {
        'path': 'ckpt/exp06/legacy_receipt.json', 'sha256': sha(paths['receipt'])}})
    with pytest.raises(ValueError, match='legacy_receipt'):
        reused(elsewhere, receipt, str(tmp_path / 'other.json'),
               exp06_receipt_for(paths['exp06']))


def test_the_reused_check_is_skipped_where_exp11s_record_was_not_read():
    subject.check_exp11_reused(None, {'sha256': WRONG}, 'anywhere', exp06_receipt=None)


def test_the_gate_refuses_to_re_resolve_the_exp06_record_itself():
    """A caller that supplies no receipt is refused, not quietly served the default."""
    paths = consumed()
    with pytest.raises(ValueError, match='must be passed in as its own receipt'):
        subject.check_exp11_reused(approvals(), {'sha256': sha(paths['receipt'])},
                                   str(paths['receipt']))


# --- the pipeline gate resolves K's initialisation under exp_11's exp_04 pin ---------


def pipeline_snippet(name):
    text = (REPO / 'tools/exp11_haa_pipeline.sh').read_text()
    return text.split(name + "='", 1)[1].split("\n'", 1)[0]


def test_the_pipeline_checks_exp11s_own_exp04_pin_before_resolving_K():
    body = pipeline_snippet('APPROVALS_PY')
    compile(body, 'APPROVALS_PY', 'exec')
    assert 'approved_digests_exp04' in body, (
        "K's initialisation is resolved from exp_04's approvals record, so exp_11's own "
        'pin for it must be checked at the queue gate')
    assert 'approved_digests_exp06' in body
    # The exp_04 pin is checked before the record is read, not after.
    assert body.index('approved_digests_exp04') < body.index('exp04_aug_checkpoint')


# --- close review blocker 1: the CONSUMED exp_06 record, not the default one --------


def test_the_consumed_exp06_record_is_the_one_the_pin_is_compared_with():
    paths = consumed()
    receipt = {'sha256': sha(paths['receipt'])}
    inputs = {}
    reused(approvals(), receipt, str(paths['receipt']),
           exp06_receipt_for(paths['exp06']), inputs=inputs)
    assert inputs[str(Path(paths['exp06']).resolve())] == sha(paths['exp06'])


def test_an_alternate_committed_exp06_record_is_refused(tmp_path):
    """The reviewer's fixture: the pin matches the default file, the run consumed another.

    The alternate record differs only in bytes -- a changed artefact digest is enough --
    and the run that loaded it must be refused, whatever the default file happens to hash
    to.
    """
    paths = consumed()
    alternate = tmp_path / 'approved_digests.json'
    value = json.loads(Path(paths['exp06']).read_text())
    value['artifacts']['epoch_012']['sha256'] = WRONG
    alternate.write_text(json.dumps(value, indent=2) + '\n')
    assert sha(alternate) != sha(paths['exp06'])
    receipt = {'sha256': sha(paths['receipt'])}
    with pytest.raises(ValueError, match='approved_digests_exp06'):
        reused(approvals(), receipt, str(paths['receipt']),
               exp06_receipt_for(alternate))


def test_the_matching_record_is_admitted_whichever_path_it_was_loaded_from(tmp_path):
    """A copy of the approved bytes at another path is the approved record."""
    paths = consumed()
    copy = tmp_path / 'approved_digests.json'
    copy.write_bytes(Path(paths['exp06']).read_bytes())
    receipt = {'sha256': sha(paths['receipt'])}
    bound = reused(approvals(), receipt, str(paths['receipt']), exp06_receipt_for(copy))
    assert bound[str(copy.resolve())] == sha(paths['exp06'])


def test_main_passes_the_loaded_receipt_to_the_gate():
    """The call site must hand over the receipt, not let the gate re-resolve a path."""
    import inspect
    source = inspect.getsource(subject.main)
    assert 'check_exp11_reused(' in source
    call = source.split('check_exp11_reused(', 1)[1].split(')', 1)[0]
    assert 'approvals_receipt' in call, (
        'main() must pass the approvals receipt it loaded from --approved into '
        'check_exp11_reused, so the pin is compared with the consumed record')
