"""Tier/checkpoint admission and the extra manifest handshake."""
import json
from types import SimpleNamespace

import pytest
import torch

from tools import exp05_eval as subject
from tools.exp05_params import TIERS, build_tier, count_parameters
from test_exp04_eval import protocol_run, bound_run


@pytest.fixture
def checkpoint(tmp_path):
    args = SimpleNamespace(checkpoint=str(tmp_path / 'epoch_001.pth'), backbone='simple', tier='S')
    model = build_tier('simple', 'S')
    torch.save(model.state_dict(), args.checkpoint)
    (tmp_path / 'args.json').write_text(json.dumps({'vit_' + k: v for k, v in TIERS['S'].items()}))
    return args


@pytest.mark.parametrize('claim,requested', [('L', 'S'), ('S', 'L'), ('L', 'L')])
def test_tier_and_state_shapes_refused(checkpoint, claim, requested):
    checkpoint.tier = requested
    subject.args_path(checkpoint).write_text(json.dumps({'vit_' + k: v for k, v in TIERS[claim].items()}))
    with pytest.raises((ValueError, RuntimeError), match='tier|size mismatch'):
        subject.load_tier(checkpoint, 8)


@pytest.mark.parametrize('tier', ['S', 'M', 'L'])
def test_legacy_only_m(checkpoint, tier):
    subject.args_path(checkpoint).write_text('{}')
    checkpoint.tier = tier
    if tier != 'M':
        with pytest.raises(ValueError, match='legacy'):
            subject.tier_metadata(checkpoint)
    else:
        meta = subject.tier_metadata(checkpoint)
        assert meta['legacy_M'] is True and meta['vit_dim'] == 512


def test_partial_or_contradictory_record_refused(checkpoint):
    for record in ({'vit_dim': 512}, dict(tier='L', **{'vit_' + k: v for k, v in TIERS['S'].items()})):
        subject.args_path(checkpoint).write_text(json.dumps(record))
        with pytest.raises(ValueError, match='tier|vit_'):
            subject.tier_metadata(checkpoint)


@pytest.mark.parametrize('raw', [None, 'null', '[]', '"hello"', '{',
    '{"backbone":"cylindrical"}', '{"param_counts":{}}'])
def test_malformed_or_mismatched_args_refused(checkpoint, raw):
    checkpoint.tier = 'M'
    path = subject.args_path(checkpoint)
    path.unlink() if raw is None else path.write_text(raw)
    with pytest.raises(ValueError):
        subject.tier_metadata(checkpoint)


@pytest.mark.parametrize('counts_tier', ['L', 'float_S'])
def test_recorded_counts_must_match_registered_tier(checkpoint, counts_tier):
    path = subject.args_path(checkpoint)
    record = json.loads(path.read_text())
    counts = count_parameters(build_tier('simple', 'L' if counts_tier == 'L' else 'S'))
    if counts_tier == 'float_S':
        counts['encoder'] = float(counts['encoder'])
    path.write_text(json.dumps(dict(record, backbone='simple', param_counts=counts)))
    with pytest.raises(ValueError, match='param_counts'):
        subject.tier_metadata(checkpoint)


def test_tier_metadata_and_manifest(checkpoint, bound_run):
    args, fields, path = bound_run
    args.checkpoint, args.tier = checkpoint.checkpoint, 'S'
    from tools.provenance import sha256_file
    meta = subject.tier_metadata(args)
    assert meta['param_counts'] == count_parameters(subject.load_tier(args, 8))
    assert all(type(value) is int for value in meta['param_counts'].values())
    fields.update(meta, checkpoint=args.checkpoint, checkpoint_sha256=sha256_file(args.checkpoint))
    path.write_text(json.dumps(fields))
    assert subject.validate_manifest(args, meta)[0] == fields
    for key in meta:
        path.write_text(json.dumps(dict(fields, **{key: 'wrong'})))
        with pytest.raises(ValueError, match=key):
            subject.validate_manifest(args, meta)
