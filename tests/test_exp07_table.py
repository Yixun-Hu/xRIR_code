"""The exp_07 seen table producer: admission, refusals and canonical outputs."""
import json
from pathlib import Path

import pytest

from exp07_fixture import exp07_approval_template, exp07_fixture  # noqa: F401  (fixtures)
from tools import provenance as p


def test_the_fixture_is_the_launchers_run_layout(exp07_fixture):
    built = exp07_fixture()
    assert len(built.directories) == 40 == 4 * 2 * 5  # roles x K x seeds
    assert len(set(built.directories)) == len(built.directories)
    for directory in built.directories:
        assert sorted(item.name for item in Path(directory).iterdir()) == [
            'completion.json', 'eval_manifest.json', 'metrics_yaw.json', 'per_sample_yaw.json']
        fields = json.loads((Path(directory) / 'eval_manifest.json').read_text())
        assert fields['split'] == 'seen' and fields['confirmatory'] is True
        assert fields['mutable_inputs']['seen_split'] == dict(
            path=p.SEEN_SPLIT, sha256=p.sha256_file(built.split))
    released = [d for d in built.directories if 'released_seen' in d]
    assert len(released) == 10
    for directory in released:
        fields = json.loads((Path(directory) / 'eval_manifest.json').read_text())
        assert set(fields['mutable_inputs']) == {'seen_split'}  # no training provenance
