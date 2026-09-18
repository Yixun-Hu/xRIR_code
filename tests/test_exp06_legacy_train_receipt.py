"""exp_06's reconstructed training receipt for arm B (plan 6.4, Codex full review F2)."""
import hashlib
import json
from pathlib import Path

import pytest

from tools import exp04_profiles
from tools import exp06_legacy_train_receipt as subject
from tools import provenance

REPO = Path(__file__).resolve().parents[1]
EPOCHS = 12


def write_train_dir(root, epochs=EPOCHS, backbone='cylindrical', extra=(), missing=()):
    """A retained exp_01 training directory: args, history, log and per-epoch weights."""
    root.mkdir(parents=True, exist_ok=True)
    if 'args.json' not in missing:
        (root / 'args.json').write_text(json.dumps(
            {'backbone': backbone, 'epochs': epochs, 'num_shot': 8, 'batch_size': 32}))
    if 'history.jsonl' not in missing:
        (root / 'history.jsonl').write_text(''.join(
            json.dumps({'epoch': epoch, 'train_loss': 0.01, 'test_loss': 0.02}) + '\n'
            for epoch in range(1, epochs + 1)))
    if 'train.log' not in missing:
        (root / 'train.log').write_text('epoch 1..{}\n'.format(epochs))
    for epoch in range(2, epochs + 1):
        (root / 'epoch_{:02d}.pth'.format(epoch)).write_bytes(
            ('weights of epoch %d' % epoch).encode())
    for name in ('best.pth', 'last.pth') + tuple(extra):
        (root / name).write_bytes(name.encode() * 4)
    return root


def registry_for(path, role='cyl', backbone='cylindrical', epoch=EPOCHS):
    """The exp_01 registration these synthetic weights stand in for."""
    return ({'role': role, 'backbone': backbone, 'epoch': epoch,
             'checkpoint': 'ckpt/xRIR_cyl_8_shot/epoch_12.pth',
             'sha256': provenance.sha256_file(path)},)


@pytest.fixture
def identity():
    """The producer closure, stubbed: these tests are about the receipt, not about git."""
    return {'entry_module': subject.ENTRY_MODULE, 'commit': 'a' * 40,
            'sha256': 'b' * 64, 'files': [], 'drift': [], 'strict': True}


@pytest.fixture
def train(tmp_path):
    return write_train_dir(tmp_path / 'xRIR_cyl_8_shot')


def build(train, out, identity, **named):
    checkpoint = train / named.pop('checkpoint', 'epoch_12.pth')
    registry = named.pop('registry', None) or registry_for(checkpoint)
    return subject.write_receipt(out, train, checkpoint, repo=REPO, strict=False,
                                 registry=registry, identity=identity, **named)


def test_the_receipt_enumerates_every_retained_artefact(train, tmp_path, identity):
    record, digest = build(train, tmp_path / 'receipt.json', identity)
    names = [item['path'] for item in record['files']]
    assert names[:3] == ['args.json', 'history.jsonl', 'train.log']
    assert names[3:14] == ['epoch_{:02d}.pth'.format(e) for e in range(2, 13)]
    assert names[14:] == ['best.pth', 'last.pth']
    assert record['label'] == 'reconstructed' and record['experiment'] == 'exp_01'
    assert record['epochs'] == EPOCHS and record['arm'] == 'cyl'
    assert record['checkpoint'] == {'path': 'epoch_12.pth', 'epoch': 12,
                                    'sha256': provenance.sha256_file(train / 'epoch_12.pth')}
    assert record['args']['sha256'] == provenance.sha256_file(train / 'args.json')
    assert record['source_closure']['entry_module'] == subject.ENTRY_MODULE
    assert record['files_sha256'] == subject._digest(record['files'])
    assert digest == provenance.sha256_file(tmp_path / 'receipt.json')


def test_the_receipt_is_written_once(train, tmp_path, identity):
    out = tmp_path / 'receipt.json'
    build(train, out, identity)
    with pytest.raises(FileExistsError):
        build(train, out, identity)


def test_zero_padded_epoch_checkpoints_are_ordered_by_their_epoch(tmp_path, identity):
    """exp_01 wrote both `epoch_05.pth` and `epoch_005.pth`; both are enumerated."""
    train = write_train_dir(tmp_path / 'cyl', extra=('epoch_005.pth', 'epoch_010.pth'))
    record, _ = build(train, tmp_path / 'r.json', identity)
    names = [item['path'] for item in record['files'] if item['path'].startswith('epoch_')]
    assert names.index('epoch_005.pth') < names.index('epoch_06.pth')
    assert names.index('epoch_005.pth') < names.index('epoch_05.pth')   # one epoch, sorted
    assert names.index('epoch_010.pth') < names.index('epoch_11.pth')


REFUSALS = {
    'unregistered': ({'registry': ({'role': 'cyl', 'backbone': 'cylindrical', 'epoch': 12,
                                    'checkpoint': 'x', 'sha256': 'c' * 64},)},
                     'not a registered exp_01 arm'),
    'not_the_last_epoch': ({'checkpoint': 'epoch_11.pth'}, 'not the last of the'),
    'not_an_epoch_file': ({'checkpoint': 'best.pth'}, 'not an epoch_NN.pth'),
}


@pytest.mark.parametrize('case', sorted(REFUSALS))
def test_a_checkpoint_the_receipt_may_not_certify_is_refused(train, tmp_path, identity, case):
    named, cause = REFUSALS[case]
    if 'registry' not in named:
        named = dict(named, registry=registry_for(train / named['checkpoint']))
    with pytest.raises(ValueError, match=cause):
        build(train, tmp_path / 'r.json', identity, **named)
    assert not (tmp_path / 'r.json').exists()


@pytest.mark.parametrize('name', sorted(subject.REQUIRED_FILES))
def test_an_incomplete_training_directory_is_refused(tmp_path, identity, name):
    train = write_train_dir(tmp_path / 'cyl', missing=(name,))
    with pytest.raises(ValueError, match='has no ' + name):
        build(train, tmp_path / 'r.json', identity)


def test_a_history_that_does_not_match_args_json_is_refused(tmp_path, identity):
    train = write_train_dir(tmp_path / 'cyl')
    rows = (train / 'history.jsonl').read_text().splitlines()
    (train / 'history.jsonl').write_text('\n'.join(rows[:-1]) + '\n')
    with pytest.raises(ValueError, match='not the 12 of args.json'):
        build(train, tmp_path / 'r.json', identity)
    (train / 'history.jsonl').write_text('\n'.join(reversed(rows)) + '\n')
    with pytest.raises(ValueError, match='not 1..12'):
        build(train, tmp_path / 'r2.json', identity)


def test_a_backbone_that_is_not_the_registered_arms_is_refused(tmp_path, identity):
    train = write_train_dir(tmp_path / 'cyl', backbone='simple')
    with pytest.raises(ValueError, match='not the .cylindrical. of the registered'):
        build(train, tmp_path / 'r.json', identity)


# --- verification --------------------------------------------------------------------------


def test_verify_rehashes_every_enumerated_artefact(train, tmp_path, identity):
    out = tmp_path / 'receipt.json'
    record, digest = build(train, out, identity)
    checked = subject.verify(out, checkpoint_sha256=record['checkpoint']['sha256'],
                             registered_sha256=record['checkpoint']['sha256'], epochs=12)
    assert checked['sha256'] == digest and checked['epochs'] == 12
    assert str(out.resolve()) in checked['inputs']
    assert checked['inputs'][str((train / 'epoch_12.pth').resolve())] == \
        record['checkpoint']['sha256']
    assert len(checked['inputs']) == len(record['files']) + 1


def test_verify_routes_its_reads_through_a_callers_hasher(train, tmp_path, identity):
    out = tmp_path / 'receipt.json'
    build(train, out, identity)
    seen = []

    def hasher(path):
        seen.append(path)
        return provenance.sha256_file(path)

    subject.verify(out, hasher=hasher)
    assert len(seen) == 16 and all(Path(item).is_file() for item in seen)


def test_a_stale_or_tampered_receipt_is_refused(train, tmp_path, identity):
    out = tmp_path / 'receipt.json'
    record, _ = build(train, out, identity)
    subject.verify(out)
    (train / 'epoch_07.pth').write_bytes(b'rewritten after the receipt')
    with pytest.raises(ValueError, match='changed since the receipt'):
        subject.verify(out)


DAMAGE = {
    'label': ({'label': 'confirmatory'}, 'labelled'),
    'schema': ({'schema_version': 2}, 'schema'),
    'experiment': ({'experiment': 'exp_05'}, "not exp_01's"),
    'enumeration': ({'files_sha256': 'd' * 64}, 'own files_sha256'),
    'no_files': ({'files': []}, 'enumerates nothing'),
}


@pytest.mark.parametrize('case', sorted(DAMAGE))
def test_a_receipt_that_contradicts_itself_is_refused(train, tmp_path, identity, case):
    fields, cause = DAMAGE[case]
    record, _ = build(train, tmp_path / 'receipt.json', identity)
    damaged = tmp_path / 'damaged.json'
    damaged.write_text(json.dumps(dict(record, **fields)))
    with pytest.raises(ValueError, match=cause):
        subject.verify(damaged)


def test_the_expected_identities_are_compared(train, tmp_path, identity):
    out = tmp_path / 'receipt.json'
    record, _ = build(train, out, identity)
    with pytest.raises(ValueError, match='not the evaluated'):
        subject.verify(out, checkpoint_sha256='e' * 64)
    with pytest.raises(ValueError, match='not the registered exp_01'):
        subject.verify(out, registered_sha256='e' * 64)
    with pytest.raises(ValueError, match='not 9'):
        subject.verify(out, epochs=9)


def test_the_cli_writes_the_receipt_and_reports_it(train, tmp_path, monkeypatch, capsys,
                                                   identity):
    monkeypatch.setattr(subject, 'REGISTERED', registry_for(train / 'epoch_12.pth'))
    monkeypatch.setattr(subject, 'source_identity', lambda repo=REPO, strict=True: identity)
    out = tmp_path / 'out' / 'receipt.json'
    assert subject.main(['--train-dir', str(train), '--checkpoint',
                         str(train / 'epoch_12.pth'), '--out', str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['sha256'] == provenance.sha256_file(out) and printed['epochs'] == 12
    assert printed['arm'] == 'cyl' and printed['files'] == 16


# --- the real exp_01 cylindrical training directory -----------------------------------------


REAL_TRAIN = REPO / 'ckpt/xRIR_cyl_8_shot'
REAL_CHECKPOINT = REAL_TRAIN / 'epoch_12.pth'


@pytest.mark.skipif(not REAL_CHECKPOINT.is_file(),
                    reason="needs exp_01's cylindrical training directory")
def test_the_real_exp01_cylindrical_training_gets_a_receipt(tmp_path, identity):
    out = tmp_path / 'receipt.json'
    record, digest = subject.write_receipt(out, REAL_TRAIN, REAL_CHECKPOINT, repo=REPO,
                                           strict=False, identity=identity)
    assert record['arm'] == exp04_profiles.CYL['role'] == 'cyl'
    assert record['checkpoint']['sha256'] == exp04_profiles.CYL['sha256']
    assert record['epochs'] == 12 and record['backbone'] == 'cylindrical'
    names = [item['path'] for item in record['files']]
    assert set(subject.REQUIRED_FILES) <= set(names) and 'epoch_12.pth' in names
    checked = subject.verify(out, checkpoint_sha256=exp04_profiles.CYL['sha256'],
                             registered_sha256=exp04_profiles.CYL['sha256'], epochs=12)
    assert checked['sha256'] == digest
    assert hashlib.sha256(out.read_bytes()).hexdigest() == digest


# --- R3: the receipt's own enumeration is not evidence of complete retention ----------------


def rewritten(record, out, **fields):
    """The receipt with `fields` replaced, its enumeration re-hashed as a forger would."""
    record = dict(record, **fields)
    record['files_sha256'] = subject._digest(record['files'])
    Path(out).write_text(json.dumps(record))
    return out


def test_a_receipt_that_omits_retained_artefacts_is_refused(train, tmp_path, identity):
    """R3: Codex kept two of eighteen files, re-hashed the list, and was admitted."""
    record, _ = build(train, tmp_path / 'receipt.json', identity)
    kept = [item for item in record['files']
            if item['path'] in ('args.json', 'epoch_12.pth')]
    two = rewritten(record, tmp_path / 'two.json', files=kept)
    with pytest.raises(ValueError, match='history.jsonl'):
        subject.verify(two)
    assert subject.receipt_names(train) == [item['path'] for item in record['files']]
    # An artefact the directory never held is the same refusal, from the other side.
    more = rewritten(record, tmp_path / 'more.json',
                     files=record['files'] + [{'path': 'epoch_13.pth', 'sha256': 'c' * 64}])
    with pytest.raises(ValueError, match='epoch_13.pth'):
        subject.verify(more)


def test_a_receipt_produced_from_a_drifting_or_dirty_tree_is_refused(train, tmp_path,
                                                                     identity):
    """`--allow-dirty` says the receipt is never confirmatory; the reader now agrees."""
    record, _ = build(train, tmp_path / 'receipt.json', identity)
    drifted = rewritten(record, tmp_path / 'drift.json',
                        source_closure=dict(identity, drift=['tools/edited.py']))
    with pytest.raises(ValueError, match='tools/edited.py'):
        subject.verify(drifted)
    dirty = rewritten(record, tmp_path / 'dirty.json',
                      source_closure=dict(identity, strict=False))
    with pytest.raises(ValueError, match='allow-dirty'):
        subject.verify(dirty)
    none = rewritten(record, tmp_path / 'none.json', source_closure=None)
    with pytest.raises(ValueError, match='no producer closure'):
        subject.verify(none)


CONTRADICTIONS = {
    'epochs': ({'epochs': 9, 'checkpoint': {'path': 'epoch_12.pth', 'epoch': 9}},
               'not the 9 of the receipt'),
    'backbone': ({'backbone': 'simple'}, 'backbone'),
    'arm': ({'arm': 'control'}, 'registered'),
    'registered_weights': ({'registered': {'role': 'cyl', 'backbone': 'cylindrical',
                                           'epoch': 12, 'checkpoint': 'x',
                                           'sha256': 'c' * 64}}, 'registered'),
}


@pytest.mark.parametrize('case', sorted(CONTRADICTIONS))
def test_a_receipt_that_contradicts_the_retained_training_is_refused(train, tmp_path,
                                                                     identity, case):
    """R3: the args/history/registered-arm contract is re-checked, not taken on trust."""
    fields, cause = CONTRADICTIONS[case]
    record, _ = build(train, tmp_path / 'receipt.json', identity)
    if 'checkpoint' in fields:
        fields = dict(fields, checkpoint=dict(record['checkpoint'], **fields['checkpoint']))
    damaged = rewritten(record, tmp_path / (case + '.json'), **fields)
    with pytest.raises(ValueError, match=cause):
        subject.verify(damaged)


def test_the_membership_rule_is_the_writers_own(train, tmp_path, identity):
    """A file retained after the receipt was written is a refusal, not a silent extra."""
    build(train, tmp_path / 'receipt.json', identity)
    (train / 'epoch_13.pth').write_bytes(b'a thirteenth epoch, retained later')
    try:
        with pytest.raises(ValueError, match='epoch_13.pth'):
            subject.verify(tmp_path / 'receipt.json')
    finally:
        (train / 'epoch_13.pth').unlink()
    assert subject.verify(tmp_path / 'receipt.json')['epochs'] == EPOCHS
