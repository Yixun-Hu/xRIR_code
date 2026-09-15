"""What the finalizer requires of the heading records and observations a HAA child cites.

Round 2b's consumer contract (full-train review finding 7 and close-4 finding 1): the
heading JSON a child bound is re-read against the HAA cache directory the child recorded,
so a cache edited after estimation is refused; only a ``confirmatory`` record may bind a
child; and in a room whose T60 the paper omits every observation must be the writer's NaN,
never an infinity.
"""
import json
from pathlib import Path

import pytest
import torch

from sim_to_real.haa_dataset import ROOMS
from tools import exp06_finalize, provenance
from test_exp06_finalize import (closed_log_file, data_root, haa_eval_args,  # noqa: F401
                                 haa_repo, haa_train_args, heading_jsons, job_run,
                                 tiny_state, write_haa_eval, write_haa_train)

INFINITIES = ([float('inf'), float('nan'), float('nan')], [float('-inf')] * 3)


def eval_child(tmp_path, haa_repo, heading_jsons, data_root, room='hallway', **overrides):
    checkpoint = haa_repo / 'stage2_best.pth'
    torch.save(tiny_state(), checkpoint)
    args = haa_eval_args(heading_jsons, checkpoint, room=room, **overrides)
    run, log = tmp_path / 'eval' / room, tmp_path / 'child.log'
    from test_exp06_finalize import eval_meta
    return args, run, log, checkpoint


def write_eval(args, run, log, haa_repo, data_root, checkpoint, **kwargs):
    from test_exp06_finalize import eval_meta
    write_haa_eval(run, args, log, haa_repo, data_root,
                   meta=eval_meta(args, provenance.sha256_file(checkpoint)), **kwargs)


# --- an omitted room measures no T60 at all: every observation is its NaN --------------

@pytest.mark.parametrize('t60', INFINITIES)
def test_an_omitted_room_measures_only_nan_t60(tmp_path, haa_repo, heading_jsons, data_root,
                                               t60):
    """Close-4 finding 1: infinity is not the writer's skipped-measurement sentinel."""
    args, run, log, checkpoint = eval_child(tmp_path, haa_repo, heading_jsons, data_root,
                                            room='dampened_room')
    write_eval(args, run, log, haa_repo, data_root, checkpoint, per_sample={'t60': t60})
    with pytest.raises(ValueError, match='t60'):
        exp06_finalize.finalize(run, 'haa_eval', log, 0, repo=haa_repo)
    assert not (run / 'completion.json').exists()


@pytest.mark.parametrize('t60', INFINITIES)
def test_a_job_refuses_an_omitted_room_whose_t60_is_infinite(job_run, closed_log_file, t60):
    """The same sentinel rule at job admission, with the child's hashes refreshed."""
    def mutate(job, names):
        path = job / 'eval/dampened_room'
        name = 'per_sample_dampened_room.json'
        body = json.loads((path / name).read_text())
        body['t60'] = list(t60)[: len(body['index'])]
        (path / name).write_text(json.dumps(body))
        record = json.loads((path / 'completion.json').read_text())
        record['artifacts'][name] = provenance.sha256_file(path / name)
        (path / 'completion.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
        return names

    job, children, spec, repo = job_run(mutate=mutate)
    with pytest.raises(ValueError, match='t60'):
        exp06_finalize.finalize(job, 'haa_job', closed_log_file, 0, repo=repo,
                                children=children, expect='finetune', job_spec=spec)
    assert not (job / 'completion.json').exists()


# --- the heading record must still describe the cache the run read --------------------

def test_a_cache_edited_after_estimation_refuses_the_child(tmp_path, haa_repo, heading_jsons,
                                                           data_root):
    """Finding 7: the record's input hashes are re-verified against <haa_root>/<room>."""
    init = haa_repo / 'init.pth'
    torch.save(tiny_state(), init)
    args = haa_train_args(heading_jsons, provenance.sha256_file(init), rooms=['hallway'],
                          init=str(init))
    run, log = tmp_path / 'stage1', tmp_path / 'child.log'
    write_haa_train(run, args, log, haa_repo, data_root)
    meta = Path(heading_jsons['root'], 'hallway', 'meta.json')
    original = meta.read_text()
    meta.write_text(original.rstrip() + ' ')
    try:
        with pytest.raises(ValueError, match='heading'):
            exp06_finalize.finalize(run, 'haa_train', log, 0, repo=haa_repo)
    finally:
        meta.write_text(original)
    assert not (run / 'completion.json').exists()
    assert exp06_finalize.finalize(run, 'haa_train', log, 0, repo=haa_repo)['frame'] == 'heading'


def test_a_diagnostic_heading_record_may_not_bind_a_child(tmp_path, haa_repo, heading_jsons,
                                                          data_root):
    """Only a record of a clean tree whose closure equals HEAD binds a confirmatory run."""
    args, run, log, checkpoint = eval_child(tmp_path, haa_repo, heading_jsons, data_root)
    path = Path(heading_jsons['hallway']['path'])
    original = path.read_text()
    record = json.loads(original)
    source = dict(record['source_closure'],
                  git=dict(record['source_closure']['git'], dirty=True,
                           dirty_outside_worklog=True, diff_sha256='c' * 64))
    path.write_text(json.dumps(dict(record, admissibility='diagnostic', source_closure=source),
                               sort_keys=True, indent=2) + '\n')
    args['heading']['hallway']['sha256'] = provenance.sha256_file(path)
    write_eval(args, run, log, haa_repo, data_root, checkpoint)
    try:
        with pytest.raises(ValueError, match='confirmatory'):
            exp06_finalize.finalize(run, 'haa_eval', log, 0, repo=haa_repo)
    finally:
        path.write_text(original)
    assert not (run / 'completion.json').exists()


@pytest.mark.parametrize('damage,cause', [('absent', 'haa_root'), ('empty', 'haa_root'),
                                          ('missing', 'HAA cache'), ('wrong_type', 'haa_root')])
def test_the_recorded_cache_root_is_required_by_the_heading_frame(tmp_path, haa_repo,
                                                                  heading_jsons, data_root,
                                                                  damage, cause):
    """A heading binding that names no readable cache root cannot be re-verified."""
    args, run, log, checkpoint = eval_child(tmp_path, haa_repo, heading_jsons, data_root)
    if damage == 'absent':
        args.pop('haa_root')
    else:
        args['haa_root'] = {'empty': '', 'wrong_type': 12,
                            'missing': str(tmp_path / 'absent')}[damage]
    write_eval(args, run, log, haa_repo, data_root, checkpoint)
    with pytest.raises(ValueError, match=cause):
        exp06_finalize.finalize(run, 'haa_eval', log, 0, repo=haa_repo)
    assert not (run / 'completion.json').exists()


def test_the_room_frame_needs_no_cache_root(tmp_path, haa_repo, heading_jsons, data_root):
    """The room frame binds no heading, so it cites no cache directory either."""
    args, run, log, checkpoint = eval_child(tmp_path, haa_repo, heading_jsons, data_root,
                                            frame='room')
    args.pop('haa_root', None)
    write_eval(args, run, log, haa_repo, data_root, checkpoint)
    fields = exp06_finalize.finalize(run, 'haa_eval', log, 0, repo=haa_repo)
    assert fields['frame'] == 'room' and fields['heading'] is None


def test_every_room_of_a_stage_one_child_is_re_verified(tmp_path, haa_repo, heading_jsons,
                                                        data_root):
    """Stage 1 covers three rooms, and each record is checked against its own cache."""
    init = haa_repo / 'init.pth'
    torch.save(tiny_state(), init)
    args = haa_train_args(heading_jsons, provenance.sha256_file(init), init=str(init))
    run, log = tmp_path / 'stage1', tmp_path / 'child.log'
    write_haa_train(run, args, log, haa_repo, data_root)
    assert sorted(exp06_finalize.finalize(run, 'haa_train', log, 0, repo=haa_repo)['heading']) \
        == sorted(['class_room', 'hallway', 'complex_room'])
    for room in ('class_room', 'hallway', 'complex_room'):
        assert room in ROOMS
