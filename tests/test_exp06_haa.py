"""exp_06 HAA entry points: fail-closed headings, pinned-path parity and the records."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sim_to_real import finetune_haa
from sim_to_real.haa_dataset import ROOMS, HAADataset
from tools import exp06_haa_finetune as trainer
from tools import exp06_heading
from tools import provenance

ROOT = Path(__file__).resolve().parents[1]
MAX_LEN = 64
ORIENTED = 'cylindrical_oriented'


def closure_digest(files, key):
    return hashlib.sha256(json.dumps([[f['path'], f[key]] for f in files],
                                     sort_keys=True).encode()).hexdigest()


def confirmatory(record):
    """The record round 3 writes: a clean tree whose closure equals its HEAD blobs."""
    files = [{'path': 'tools/exp06_heading.py', 'sha256': 'a' * 64, 'head_blob_sha256': 'a' * 64}]
    git = {'HEAD': 'b' * 40, 'dirty': False, 'dirty_outside_worklog': False,
           'diff_sha256': None}
    source = dict(record['source_closure'], files=files, git=git,
                  sha256=closure_digest(files, 'sha256'),
                  head_sha256=closure_digest(files, 'head_blob_sha256'))
    return dict(record, source_closure=source, admissibility='confirmatory')


def write_room(room, levels=None):
    """One HAA cache room: twelve training microphones, two held-out measurements."""
    room.mkdir(parents=True)
    room.joinpath('meta.json').write_text(json.dumps(
        dict(train=list(range(12)), valid=[12], test=[12, 13], sr=22050)))
    theta = np.deg2rad([-90] * 4 + [90] * 8)
    dist = np.arange(1, 13) / 3
    xyz = np.zeros((14, 3))
    xyz[:12, :2] = np.stack((np.cos(theta), np.sin(theta)), axis=1) * dist[:, None]
    xyz[12:, :2] = [[1.0, -2.0], [-1.5, 3.0]]
    xyz[:, 2] = 50
    np.save(room / 'xyzs.npy', xyz)
    np.save(room / 'speaker_xyz.npy', np.zeros(3))
    rirs = np.zeros((14, MAX_LEN))
    rirs[:12, 10] = 10 ** (np.asarray([9] * 4 + [0] * 8 if levels is None else levels) / 20) / dist
    rirs[12:, 5] = [0.5, 0.25]
    np.save(room / 'rirs.npy', rirs)
    np.save(room / 'depth.npy', np.ones((256, 512), dtype='float32'))
    return room


@pytest.fixture(scope='session')
def cache(tmp_path_factory):
    """A four-room synthetic HAA cache with a confirmatory heading record per room."""
    root = tmp_path_factory.mktemp('haa') / 'HAA_xrir'
    headings = root.parent / 'heading'
    headings.mkdir(parents=True)
    for room in ROOMS:
        record = exp06_heading.estimate_room_heading(write_room(root / room))
        assert record['decision'] == 'estimated' and record['room'] == room
        exp06_heading.write_heading_json(headings / (room + '.json'), confirmatory(record))
    init = root.parent / 'init.pth'
    init.write_bytes(b'initialisation')
    return {'root': str(root), 'heading': str(headings), 'init': str(init)}


def train_argv(cache, *extra, backbone=ORIENTED, rooms=('hallway',)):
    return ['--backbone', backbone, '--init', cache['init'], '--rooms', *rooms,
            '--save-dir', str(Path(cache['root']).parent / 'stage'), '--root', cache['root'],
            '--max-len', str(MAX_LEN), '--epochs', '4', '--val-every', '2', *extra]


def test_shared_flag_defaults_equal_the_pinned_trainer(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['finetune_haa.py', '--backbone', 'simple', '--init', 'i',
                                      '--rooms', 'class_room', '--save-dir', 'd'])
    pinned = vars(finetune_haa.parse_args())
    mine = vars(trainer.build_parser().parse_args(
        ['--backbone', 'simple', '--init', 'i', '--rooms', 'class_room', '--save-dir', 'd']))
    assert set(pinned) <= set(mine)
    for key, value in pinned.items():
        assert type(mine[key]) is type(value) and mine[key] == value, key
    assert set(mine) - set(pinned) == {'root', 'heading_json_dir', 'run_type',
                                      'job_spec'}
    assert mine['run_type'] == 'haa_train'
    assert sorted(action.choices for action in trainer.build_parser()._actions
                  if action.dest == 'backbone') == [sorted(trainer.BACKBONES_EXP06)]


def test_the_oriented_backbone_refuses_to_run_without_a_heading(cache):
    args = trainer.build_parser().parse_args(train_argv(cache))
    with pytest.raises(ValueError, match='heading'):
        trainer.prepare(args)


def test_a_pinned_backbone_without_a_heading_is_the_pinned_dataset(cache):
    args = trainer.build_parser().parse_args(train_argv(cache, backbone='simple'))
    context = trainer.prepare(args)
    assert context.frame == 'room' and context.heading is None
    assert type(context.train_dataset) is HAADataset
    assert type(context.val_dataset) is HAADataset
    pinned = HAADataset(['hallway'], 'val', root=cache['root'], num_shot=args.num_shot,
                        max_len=MAX_LEN, eval_seed=args.eval_seed)
    assert context.val_dataset.items == pinned.items
    for index in range(len(pinned)):
        assert all(torch.equal(left, right)
                   for left, right in zip(pinned[index], context.val_dataset[index]))


def test_a_heading_directory_selects_the_heading_frame(cache):
    args = trainer.build_parser().parse_args(
        train_argv(cache, '--heading-json-dir', cache['heading']))
    context = trainer.prepare(args)
    assert context.frame == 'heading'
    assert type(context.train_dataset) is exp06_heading.HeadingFrameDataset
    assert context.train_dataset.k_by_room == {'hallway': 128}
    assert context.heading['hallway']['k'] == 128
    assert context.heading['hallway']['phi_deg'] == -90
    assert context.heading['hallway']['decision'] == 'estimated'
    path = Path(cache['heading']) / 'hallway.json'
    assert context.heading['hallway']['path'] == str(path)
    assert context.heading['hallway']['sha256'] == provenance.sha256_file(path)


@pytest.mark.parametrize('damage', ['missing', 'refused', 'diagnostic', 'changed_input',
                                    'wrong_room'])
def test_unusable_heading_records_are_refused(cache, tmp_path, damage):
    directory = tmp_path / 'heading'
    directory.mkdir()
    room = 'hallway'
    record = exp06_heading.read_heading_json(Path(cache['heading']) / (room + '.json'))
    if damage == 'refused':
        flat = exp06_heading.estimate_room_heading(write_room(tmp_path / 'flat' / room, [0] * 12))
        assert flat['decision'] == 'refused'
        record = confirmatory(flat)
    elif damage == 'diagnostic':
        record = dict(record, admissibility='diagnostic',
                      source_closure=dict(record['source_closure'], git=dict(
                          record['source_closure']['git'], dirty=True,
                          dirty_outside_worklog=True, diff_sha256='c' * 64)))
    elif damage == 'wrong_room':
        record = dict(record, room='class_room')
    if damage != 'missing':
        exp06_heading.write_heading_json(directory / (room + '.json'), record)
    if damage == 'changed_input':
        Path(cache['root'], room, 'meta.json').write_text(
            json.dumps(dict(train=list(range(12)), valid=[12], test=[12, 13], sr=22050,
                            note='edited after estimation')))
    args = trainer.build_parser().parse_args(
        train_argv(cache, '--heading-json-dir', str(directory)))
    try:
        with pytest.raises(ValueError, match='heading'):
            trainer.prepare(args)
    finally:
        if damage == 'changed_input':
            Path(cache['root'], room, 'meta.json').write_text(json.dumps(
                dict(train=list(range(12)), valid=[12], test=[12, 13], sr=22050)))


def test_prepare_records_every_field_the_finalizer_binds(cache, tmp_path):
    args = trainer.build_parser().parse_args(
        train_argv(cache, '--heading-json-dir', cache['heading'], '--seed', '3'))
    context = trainer.prepare(args, command=['--backbone', ORIENTED])
    record = context.args_record
    assert record['frame'] == 'heading' and record['rooms'] == ['hallway']
    assert record['seed'] == 3 and record['eval_seed'] == 0 and record['num_shot'] == 8
    assert record['heading']['hallway']['k'] == 128
    assert record['init_sha256'] == provenance.sha256_file(cache['init'])
    assert record['haa_root'] == str(Path(cache['root']).resolve())
    assert record['run_type'] == 'haa_train'
    assert record['git_head'] == context.fields['git_state']['HEAD']
    assert record['exp06_registry_sha256'] == context.fields['registry_sha256']
    assert record['exp06_source_closure_sha256'] == \
        context.fields['source_closures']['child']['sha256']
    fields = context.fields
    assert fields['run_type'] == 'haa_train'
    assert fields['source_closures']['child']['entry_module'] == 'tools.exp06_haa_finetune'
    assert set(fields['data_identity']) >= {'data_root', 'inventory', 'inventory_files',
                                            'inventory_bytes', 'inventory_sha256'}
    assert {entry['path'] for entry in fields['data_identity']['inventory']} == {
        'hallway/' + name for name in ('meta.json', 'rirs.npy', 'xyzs.npy',
                                       'speaker_xyz.npy', 'depth.npy')}
    assert json.dumps(record, allow_nan=False) and json.dumps(fields, allow_nan=False)
    assert type(context.model) is trainer.BACKBONES_EXP06[ORIENTED]


def test_a_missing_cache_root_is_refused(cache, tmp_path):
    args = trainer.build_parser().parse_args(train_argv(cache, backbone='simple'))
    args.root = str(tmp_path / 'absent')
    with pytest.raises(ValueError, match='root'):
        trainer.prepare(args)


def eval_argv(cache, *extra, backbone=ORIENTED, rooms=('hallway',)):
    return ['--backbone', backbone, '--checkpoint', cache['init'], '--rooms', *rooms,
            '--save-dir', str(Path(cache['root']).parent / 'eval' / rooms[0]),
            '--root', cache['root'], '--max-len', str(MAX_LEN), *extra]


def test_eval_shared_flag_defaults_equal_the_pinned_evaluator(monkeypatch):
    from sim_to_real import eval_haa
    from tools import exp06_haa_eval as evaluator
    monkeypatch.setattr(sys, 'argv', ['eval_haa.py', '--backbone', 'simple',
                                      '--checkpoint', 'c', '--save-dir', 'd'])
    pinned = vars(eval_haa.parse_args())
    mine = vars(evaluator.build_parser().parse_args(
        ['--backbone', 'simple', '--checkpoint', 'c', '--save-dir', 'd']))
    assert set(pinned) <= set(mine)
    for key, value in pinned.items():
        assert type(mine[key]) is type(value) and mine[key] == value, key
    assert set(mine) - set(pinned) == {'root', 'heading_json_dir', 'seed', 'run_type',
                                       'gl_seed_per_query', 'job_spec'}
    assert mine['run_type'] == 'haa_eval' and mine['gl_seed_per_query'] is False
    assert mine['seed'] == 0 and mine['eval_seed'] == 0


def test_eval_refuses_the_oriented_backbone_without_a_heading(cache):
    from tools import exp06_haa_eval as evaluator
    args = evaluator.build_parser().parse_args(eval_argv(cache))
    with pytest.raises(ValueError, match='heading'):
        evaluator.prepare(args)


def test_eval_preserves_the_pinned_item_order_and_reference_draw(cache):
    from tools import exp06_haa_eval as evaluator
    args = evaluator.build_parser().parse_args(eval_argv(cache, backbone='simple'))
    context = evaluator.prepare(args)
    dataset = context.datasets['hallway']
    pinned = HAADataset(['hallway'], 'test', root=cache['root'], num_shot=args.num_shot,
                        max_len=MAX_LEN, eval_seed=args.eval_seed)
    assert type(dataset) is HAADataset and dataset.items == pinned.items
    assert [item[1] for item in dataset.items] == [12, 13]
    for index, (room, idx) in enumerate(pinned.items):
        np.testing.assert_array_equal(dataset._pick_refs(room, idx), pinned._pick_refs(room, idx))
        assert all(torch.equal(left, right) for left, right in zip(pinned[index], dataset[index]))


def test_eval_records_every_field_the_finalizer_binds(cache):
    from tools import exp06_haa_eval as evaluator
    args = evaluator.build_parser().parse_args(
        eval_argv(cache, '--heading-json-dir', cache['heading'], '--gl-seed-per-query'))
    context = evaluator.prepare(args, command=['--backbone', ORIENTED])
    record = context.args_record
    assert record['frame'] == 'heading' and record['rooms'] == ['hallway']
    assert record['seed'] == 0 and record['run_type'] == 'haa_eval'
    assert record['heading']['hallway']['k'] == 128
    assert record['checkpoint_sha256'] == provenance.sha256_file(cache['init'])
    assert record['haa_root'] == str(Path(cache['root']).resolve())
    assert context.fields['run_type'] == 'haa_eval'
    assert context.fields['source_closures']['child']['entry_module'] == 'tools.exp06_haa_eval'
    assert type(context.datasets['hallway']) is exp06_heading.HeadingFrameDataset
    meta = evaluator.per_sample_meta(args, context, 'hallway')
    assert meta['frame'] == 'heading' and meta['room'] == 'hallway'
    assert meta['checkpoint_sha256'] == record['checkpoint_sha256']
    assert meta['gl_seed_per_query'] is True and meta['run_type'] == 'haa_eval'
    assert meta['heading'] == record['heading']
    for key in ('backbone', 'checkpoint', 'split', 'num_shot', 'eval_seed'):
        assert meta[key] == record[key]
    assert meta['registry_sha256'] == context.fields['registry_sha256']
    assert meta['git_head'] == context.fields['git_state']['HEAD']
    assert meta['exp06_source_closure_sha256'] == \
        context.fields['source_closures']['child']['sha256']


def test_the_per_query_griffin_lim_seed_is_the_frozen_formula():
    from tools import exp06_haa_eval as evaluator
    for eval_seed in (0, 3):
        for room in ('hallway', 'class_room'):
            for idx in (0, 1, 12):
                expected = int.from_bytes(hashlib.sha256(
                    'gl:{}:{}:{}'.format(eval_seed, room, idx).encode()).digest()[:8], 'little')
                assert evaluator.gl_seed(eval_seed, room, idx) == expected
    assert evaluator.gl_seed(0, 'hallway', 0) != evaluator.gl_seed(0, 'hallway', 1)
    assert evaluator.gl_seed(0, 'hallway', 0) != evaluator.gl_seed(1, 'hallway', 0)
    assert evaluator.gl_seed(0, 'hallway', 0) != evaluator.gl_seed(0, 'class_room', 0)


@pytest.mark.parametrize('enabled', [False, True])
def test_the_rng_is_touched_only_when_per_query_seeding_is_asked_for(cache, monkeypatch, enabled):
    from tools import exp06_haa_eval as evaluator
    extra = ['--gl-seed-per-query'] if enabled else []
    args = evaluator.build_parser().parse_args(eval_argv(cache, *extra, backbone='simple'))
    seeds = []
    monkeypatch.setattr(evaluator.torch, 'manual_seed', lambda seed: seeds.append(seed))
    waveform = evaluator.invert(torch.rand(1, 63, 20), args, 'hallway', 7)
    assert waveform.shape[0] == 1 and waveform.dim() == 3
    assert seeds == ([evaluator.gl_seed(0, 'hallway', 7)] if enabled else [])


def test_per_query_seeding_makes_the_inversion_reproducible(cache):
    from tools import exp06_haa_eval as evaluator
    args = evaluator.build_parser().parse_args(
        eval_argv(cache, '--gl-seed-per-query', backbone='simple'))
    spec = torch.rand(1, 63, 20)
    first = evaluator.invert(spec, args, 'hallway', 7)
    assert torch.equal(first, evaluator.invert(spec, args, 'hallway', 7))
    assert not torch.equal(first, evaluator.invert(spec, args, 'hallway', 8))


def test_side_labels_are_the_room_frame_in_both_frames(cache):
    from tools import exp06_haa_eval as evaluator
    room_args = evaluator.build_parser().parse_args(eval_argv(cache, backbone='simple'))
    head_args = evaluator.build_parser().parse_args(
        eval_argv(cache, '--heading-json-dir', cache['heading']))
    room_ds = evaluator.prepare(room_args).datasets['hallway']
    head_ds = evaluator.prepare(head_args).datasets['hallway']
    indices = [idx for _, idx in room_ds.items]
    labels = evaluator.side_labels(room_ds, 'hallway', indices)
    assert labels == [-1, 1]
    assert evaluator.side_labels(head_ds, 'hallway', indices) == labels
    room_ds.data['hallway']['src_local'][indices[0], 1] = 0.0
    with pytest.raises(ValueError, match='side_label'):
        evaluator.side_labels(room_ds, 'hallway', indices)


@pytest.mark.parametrize('room', ['hallway', 'dampened_room'])
def test_the_room_summary_is_the_pinned_writers(cache, room):
    from tools import exp06_finalize
    from tools import exp06_haa_eval as evaluator
    args = evaluator.build_parser().parse_args(
        eval_argv(cache, '--heading-json-dir', cache['heading'], rooms=(room,)))
    context = evaluator.prepare(args)
    meta = evaluator.per_sample_meta(args, context, room)
    per = {'index': [12, 13], 'ir_path': [room + '/12', room + '/13'],
           'edt': [0.05, 0.07], 'c50': [1.1, 1.3], 't60': [4.0, 6.0],
           'stft_mse': [0.2, 0.4], 'loss': [0.03, 0.05], 'env': [1.0, 3.0],
           'side_label': [-1, 1]}
    counts = {'c50_outliers': 0, 't60_invalid': 0, 'edt_invalid': 0}
    summary = evaluator.room_summary(args, room, per, counts, meta, 2.5)
    assert not [key for key in exp06_finalize.METRICS_REQUIRED if key not in summary]
    assert summary['n_samples'] == 2 and summary['meta'] == meta
    assert summary['edt_error_s']['n'] == 2
    assert summary['edt_error_s']['mean'] == pytest.approx(0.06)
    assert summary['edt_error_s']['median'] == pytest.approx(0.06)
    assert (summary['t60_error_pct'] is None) is (room in evaluator.NO_T60_ROOMS)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)  # main's makedirs
    evaluator.write_room_outputs(args, room, summary, per, meta)
    written = json.loads(Path(args.save_dir, 'metrics_{}.json'.format(room)).read_text())
    body = json.loads(Path(args.save_dir, 'per_sample_{}.json'.format(room)).read_text())
    assert written == summary and body['meta'] == meta and body['index'] == per['index']
    assert body['side_label'] == per['side_label']


def test_eval_main_refuses_before_it_reaches_the_gpu(cache, tmp_path):
    from tools import exp06_haa_eval as evaluator
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='heading'):
        evaluator.main(eval_argv(cache) + ['--save-dir', str(out)])
    assert not out.exists()


# --- the pre-launch job declaration a child is bound to (round 2b finding 3) ----------

JOB_INIT = 'pretrain_epoch_012'


def job_spec_file(path, **overrides):
    """What the pipeline writes before the first child of a seed starts."""
    spec = {'init': JOB_INIT, 'backbone': ORIENTED, 'frame': 'heading', 'seed': 0,
            'init_sha256': 'a' * 64, 'rooms': sorted(ROOMS), 'expect': 'finetune',
            'heading': {room: 128 for room in ROOMS}}
    spec.update(overrides)
    Path(path).write_text(json.dumps(spec, sort_keys=True, indent=2) + '\n')
    return path


def test_a_child_records_the_job_spec_it_was_launched_under(cache, tmp_path):
    spec = job_spec_file(tmp_path / 'job_spec.json')
    args = trainer.build_parser().parse_args(
        train_argv(cache, '--heading-json-dir', cache['heading'], '--job-spec', str(spec)))
    record = trainer.prepare(args).args_record
    assert record['job_spec'] == str(spec)
    assert record['job_spec_sha256'] == provenance.sha256_file(spec)
    assert record['job_init'] == JOB_INIT and record['job_init_sha256'] == 'a' * 64


def test_an_evaluation_child_records_the_same_job_identity(cache, tmp_path):
    from tools import exp06_haa_eval as evaluator
    spec = job_spec_file(tmp_path / 'job_spec.json', expect='zeroshot')
    args = evaluator.build_parser().parse_args(
        eval_argv(cache, '--heading-json-dir', cache['heading'], '--job-spec', str(spec)))
    record = evaluator.prepare(args).args_record
    assert record['job_spec_sha256'] == provenance.sha256_file(spec)
    assert record['job_init'] == JOB_INIT and record['job_init_sha256'] == 'a' * 64


def test_a_child_launched_by_hand_records_no_job_identity(cache):
    args = trainer.build_parser().parse_args(
        train_argv(cache, '--heading-json-dir', cache['heading']))
    record = trainer.prepare(args).args_record
    assert record['job_spec'] is None and record['job_spec_sha256'] is None
    assert record['job_init'] is None and record['job_init_sha256'] is None


@pytest.mark.parametrize('damage', ['missing', 'not_json', 'not_a_record', 'no_init',
                                    'empty_init', 'bad_init_sha256'])
def test_an_unusable_job_spec_refuses_the_child(cache, tmp_path, damage):
    """Fail-closed: a declaration that names no initialisation binds nothing."""
    path = tmp_path / (damage + '.json')
    if damage == 'not_json':
        path.write_text('nonsense')
    elif damage == 'not_a_record':
        path.write_text('["a job spec is a record"]')
    elif damage == 'no_init':
        job_spec_file(path, init=None)
    elif damage == 'empty_init':
        job_spec_file(path, init='')
    elif damage == 'bad_init_sha256':
        job_spec_file(path, init_sha256='not a digest')
    args = trainer.build_parser().parse_args(
        train_argv(cache, '--heading-json-dir', cache['heading'], '--job-spec', str(path)))
    with pytest.raises(ValueError, match='job spec'):
        trainer.prepare(args)
