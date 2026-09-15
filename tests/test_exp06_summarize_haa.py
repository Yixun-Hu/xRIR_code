"""exp_06's HAA summariser: the legacy receipt, the new-arm branch and the verdicts.

The synthetic trees below are exp_02-shaped (arms A and B) and exp_06-shaped (C, D, F);
the exp_06 children are finalised by ``tools.exp06_finalize`` itself, so their
``completion.json`` files are the real thing rather than a fixture's idea of one.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sim_to_real import summarize_haa as legacy
from tools import exp06_summarize_haa as subject

ROOMS = tuple(legacy.ROOMS)
SIZE = {'class_room': 5, 'dampened_room': 4, 'hallway': 6, 'complex_room': 4}
LEGACY_INITS = ('released', 'released_repomaps', 'control', 'cyl')


def indices(room):
    return list(range(SIZE[room]))


def per_sample(room, backbone, checkpoint, offset=0.0, invalid=()):
    """One exp_02 per-sample file: the writer's arrays, plus this arm's offset."""
    idx = indices(room)
    def column(base):
        return [None if False else float('nan') if i in invalid else base + offset + 0.05 * i
                for i in idx]
    t60 = [float('nan')] * len(idx) if room == 'dampened_room' else column(3.0)
    return {'index': idx, 'ir_path': ['{}/{}'.format(room, i) for i in idx],
            'meta': {'backbone': backbone, 'checkpoint': checkpoint, 'split': 'test',
                     'num_shot': 8, 'eval_seed': 0},
            'edt': column(0.06), 'c50': column(1.0), 't60': t60,
            'stft_mse': column(0.4), 'loss': column(0.5), 'env': column(5.0)}


def metrics_file(room, backbone, checkpoint, per):
    finite = {key: [v for v in per[key] if not math.isnan(v)] for key in ('edt', 'c50', 't60')}
    def block(values):
        return {'mean': float(np.mean(values)) if values else None,
                'median': float(np.median(values)) if values else None, 'n': len(values)}
    return {'backbone': backbone, 'checkpoint': checkpoint, 'room': room, 'split': 'test',
            'num_shot': 8, 'eval_seed': 0, 'depth_variant': 'default',
            'n_samples': len(per['index']), 'edt_error_s': block(finite['edt']),
            'c50_error_db': block(finite['c50']),
            't60_error_pct': None if room == 'dampened_room' else block(finite['t60']),
            'edt_invalid': 0, 'c50_outliers': 0, 't60_invalid': 0,
            'env_error': block(finite['edt']), 'stft_log_mse': block(finite['edt']),
            'test_loss': 0.5, 'elapsed_min': 1.0}


def write_eval(directory, room, backbone, checkpoint, offset=0.0, invalid=()):
    directory.mkdir(parents=True, exist_ok=True)
    per = per_sample(room, backbone, checkpoint, offset, invalid)
    (directory / 'per_sample_{}.json'.format(room)).write_text(json.dumps(per))
    (directory / 'metrics_{}.json'.format(room)).write_text(
        json.dumps(metrics_file(room, backbone, checkpoint, per)))
    return per


def write_stage(directory, backbone, seed, init, rooms, depth_variant='default'):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'args.json').write_text(json.dumps(
        {'backbone': backbone, 'seed': seed, 'init': init, 'rooms': list(rooms),
         'depth_variant': depth_variant, 'num_shot': 8, 'eval_seed': 0, 'epochs': 4}))
    (directory / 'summary.json').write_text(json.dumps(
        {'backbone': backbone, 'seed': seed, 'init': init, 'rooms': list(rooms),
         'best_epoch': 2, 'best_val_loss': 0.4, 'init_val_loss': 0.6, 'epochs': 4,
         'minutes': 1.0, 'val_rooms': list(rooms)}))


OFFSETS = {'released': 0.2, 'released_repomaps': 0.25, 'control': 0.0, 'cyl': 0.1}


def build_legacy_root(root, offsets=None):
    """A faithful exp_02 tree: the four inits, three seeds each, and the canonical record."""
    offsets = OFFSETS if offsets is None else offsets
    root = Path(root)
    for init in LEGACY_INITS:
        backbone = legacy.BACKBONE_OF[init]
        seeds = ('seed0',) if init == 'released_repomaps' else subject.SEEDS
        variant = legacy.DEPTH_VARIANT_OF.get(init, 'default')
        for seed in seeds:
            number = int(seed[len('seed'):])
            base = root / init / seed
            write_stage(base / 'stage1', backbone, number, legacy.INIT_OF[init], ROOMS, variant)
            for room in ROOMS:
                write_stage(base / ('stage2_' + room), backbone, number,
                            str(base / 'stage1' / 'best.pth'), [room], variant)
                write_eval(base / 'eval', room, backbone,
                           str(base / ('stage2_' + room) / 'best.pth'),
                           offsets[init] + 0.01 * number)
        if init != 'released_repomaps':
            for room in ROOMS:
                write_eval(root / init / 'zeroshot', room, backbone,
                           legacy.INIT_OF[init], offsets[init] + 0.5)
    (root / 'stats.json').write_text(json.dumps({'paired': []}))
    (root / 'summary.txt').write_text('exp_02 canonical summary\n')
    return root


def build_cache(root):
    """The four rooms' meta.json, so exp_02's completeness reads these test indices."""
    root = Path(root)
    for room in ROOMS:
        (root / room).mkdir(parents=True, exist_ok=True)
        (root / room / 'meta.json').write_text(json.dumps(
            {'train': [], 'test': indices(room)}))
        xyz = np.zeros((SIZE[room], 3), dtype='float32')
        xyz[:, 1] = [1.0 if i % 2 else -1.0 for i in indices(room)]
        np.save(root / room / 'xyzs.npy', xyz)
        np.save(root / room / 'speaker_xyz.npy', np.zeros(3, dtype='float32'))
    return root


@pytest.fixture
def legacy_root(tmp_path, monkeypatch):
    cache = build_cache(tmp_path / 'HAA_xrir')
    monkeypatch.setattr(legacy, 'HAA_ROOT', str(cache))
    return build_legacy_root(tmp_path / 'sim2real')


# --- the tables of 6.2 ------------------------------------------------------------------


def test_the_arm_table_is_the_five_arms_of_section_2_3():
    assert tuple(subject.ARMS) == ('control', 'cyl', 'cyl_or', 'control_hf', 'cyl_hf')
    assert subject.LEGACY_ARMS == ('control', 'cyl')
    assert subject.NEW_ARMS == ('cyl_or', 'control_hf', 'cyl_hf')
    assert [subject.ARMS[a]['backbone'] for a in subject.ARMS] == [
        'simple', 'cylindrical', 'cylindrical_oriented', 'simple', 'cylindrical']
    assert [subject.ARMS[a]['frame'] for a in subject.ARMS] == [
        'room', 'room', 'heading', 'heading', 'heading']
    assert subject.ARMS['cyl_or']['root'] == 'ckpt/exp06/sim2real/cyl_or'


def test_the_registered_constants_are_frozen():
    assert subject.H1_MARGIN_DB == 0.23 and (subject.H1_ROOM, subject.H1_METRIC) == (
        'hallway', 'c50')
    assert subject.HEADING_K == 128
    assert subject.SEEDS == ('seed0', 'seed1', 'seed2') and subject.ZEROSHOT == 'zeroshot'
    assert len(subject.CELLS) == 11 and subject.H2_FAMILY == 11
    assert subject.N_BOOT == 10000 and subject.N_BOOT_ADJUSTED == 50000


# --- the legacy receipt ------------------------------------------------------------------


def test_the_receipt_enumerates_every_retained_artifact_of_both_arms(legacy_root):
    names = subject.legacy_artifacts()
    assert len(names) == len(subject.SEEDS) * (2 * 5 + 2 * 4) + 2 * 4
    assert 'seed0/stage1/args.json' in names and 'seed2/stage2_hallway/summary.json' in names
    assert 'seed1/eval/per_sample_class_room.json' in names
    assert 'zeroshot/metrics_complex_room.json' in names
    files = subject.legacy_receipt_files(legacy_root)
    paths = [item['path'] for item in files]
    assert paths[-2:] == ['stats.json', 'summary.txt']
    assert len(paths) == 2 * len(names) + 2 and len(set(paths)) == len(paths)
    assert all(len(item['sha256']) == 64 for item in files)


def test_the_receipt_is_written_once_labelled_reconstructed(legacy_root, tmp_path):
    out = tmp_path / 'receipt' / 'legacy_receipt.json'
    record, digest = subject.write_legacy_receipt(out, legacy_root, strict=False)
    assert record['label'] == 'reconstructed' and record['arms'] == ['control', 'cyl']
    assert record['timestamp'] and len(record['source_closure']['sha256']) == 64
    assert record['source_closure']['entry_module'] == 'tools.exp06_summarize_haa'
    assert digest == subject.provenance.sha256_file(out)
    with pytest.raises(FileExistsError):
        subject.write_legacy_receipt(out, legacy_root, strict=False)


def test_a_faithful_tree_verifies_against_its_receipt(legacy_root, tmp_path):
    out = tmp_path / 'legacy_receipt.json'
    _, digest = subject.write_legacy_receipt(out, legacy_root, strict=False)
    record = subject.verify_legacy_receipt(
        out, legacy_root, {'path': str(out), 'sha256': digest})
    assert record['sha256'] == digest and len(record['files']) > 40


def test_a_tampered_receipt_or_a_changed_artifact_is_refused(legacy_root, tmp_path):
    out = tmp_path / 'legacy_receipt.json'
    _, digest = subject.write_legacy_receipt(out, legacy_root, strict=False)
    approved = {'path': str(out), 'sha256': digest}
    changed = json.loads(out.read_text())
    changed['files'][3]['sha256'] = 'f' * 64
    (tmp_path / 'tampered.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='files_sha256'):
        subject.verify_legacy_receipt(tmp_path / 'tampered.json', legacy_root)
    with pytest.raises(ValueError, match='approved'):
        subject.verify_legacy_receipt(tmp_path / 'tampered.json', legacy_root, approved)
    target = Path(legacy_root) / 'control/seed1/eval/per_sample_hallway.json'
    target.write_text(target.read_text() + ' ')
    with pytest.raises(ValueError, match='changed since the receipt'):
        subject.verify_legacy_receipt(out, legacy_root, approved)


def test_a_receipt_of_an_incomplete_arm_is_never_written(legacy_root, tmp_path):
    (Path(legacy_root) / 'cyl/seed2/eval/metrics_hallway.json').unlink()
    with pytest.raises(ValueError, match='cyl has no cyl/seed2/eval/metrics_hallway'):
        subject.write_legacy_receipt(tmp_path / 'r.json', legacy_root, strict=False)
