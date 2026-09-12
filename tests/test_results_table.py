"""TABLE_V1 uses the paired producer's execution-bound synthetic runs."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_paired_compare import admission_fixture, _read, _rebind, _summaries
from tools import paired_compare as pc, provenance as p, results_table as rt
from tools.exp04_profiles import get_profile, json_value


@pytest.fixture
def table_fixture(admission_fixture, monkeypatch):
    fixtures = [admission_fixture('H1_K1'), admission_fixture('H1_K8')]
    profile = json_value(get_profile('TABLE_V1'))
    profile['dataset'] = fixtures[1].profile['dataset']
    arms = {a['role']: a for a in fixtures[1].profile['arms']}
    checkpoint = fixtures[1].root / 'cyl.pth'
    checkpoint.write_bytes(b'cyl fixture checkpoint')
    arms['cyl'] = dict(profile['arms'][1], checkpoint=str(checkpoint), sha256=p.sha256_file(checkpoint))
    profile['arms'] = [arms[a['role']] for a in profile['arms']]
    directories = []
    for fixture in fixtures:
        shot = fixture.profile['num_shot']
        profile['seeds'][shot] = fixture.profile['seeds'][shot]
        for original in fixture.paths[1]:
            clone = Path(original).with_name(Path(original).name.replace('control', 'cyl'))
            shutil.copytree(original, clone)
            directories.append(str(clone))
        directories.extend(sum(fixture.paths, []))
    for directory in directories:
        run = Path(directory)
        role = run.name.split('_')[0]
        manifest, sample, metrics = [_read(run / name) for name in
            ('eval_manifest.json', 'per_sample_yaw.json', 'metrics_yaw.json')]
        manifest.update(checkpoint=arms[role]['checkpoint'], checkpoint_sha256=arms[role]['sha256'],
                        backbone=arms[role]['backbone'])
        for payload in (sample, metrics):
            payload['meta'].update(checkpoint=manifest['checkpoint'], backbone=manifest['backbone'])
        sample['P']['0'].update(stft=[.5] * 12, decay=[.5] * 12, consistency=[0.] * 12)
        metrics['P'] = _summaries(sample['P'])
        _rebind(run, manifest, sample, metrics)
    producer = pc.producer_identity()
    fixtures[1].approved['closures']['producer_results_table'] = producer['sha256']
    monkeypatch.setattr(rt, 'producer_identity', lambda _: producer)
    monkeypatch.setattr(rt, 'load_approved_digests', pc.load_approved_digests)
    monkeypatch.setattr(rt, 'get_profile', lambda _: profile)
    return SimpleNamespace(directories=directories, profile=profile)


def test_table_aggregation_and_order(table_fixture):
    result, admitted = rt.build_table(table_fixture.directories)
    reverse, _ = rt.build_table(list(reversed(table_fixture.directories)))
    assert json.dumps(result, sort_keys=True) == json.dumps(reverse, sort_keys=True)
    assert len(result['rows']) == 6 and len(admitted['groups']) == 6
    for row in result['rows']:
        assert row['protocol']['batch_size'] == 16 and row['protocol']['k'] == 0
        edt = row['metrics']['EDT']
        assert edt['mean'] == pytest.approx((1 + 5.5 / 16 + .02) * 1000)
        assert edt['sd'] == pytest.approx(np.std(np.arange(5) * 10, ddof=1))
        assert row['metrics']['T60']['mean'] == pytest.approx(edt['mean'] / 1000)
        assert [v['n_finite'] for v in edt['per_seed'].values()] == [12] * 5
        assert set(row['metrics']) == {'T60', 'C50', 'EDT', 'loss', 'log_mse'}


@pytest.mark.parametrize('mutation', ['incomplete', 'wrong_role', 'tampered', 'unknown', 'finite'])
def test_table_refuses_invalid_inputs(table_fixture, mutation):
    directories = list(table_fixture.directories)
    run = Path(directories[0])
    if mutation == 'incomplete':
        directories.pop()
    elif mutation in ('wrong_role', 'tampered'):
        path = run / ('eval_manifest.json' if mutation == 'wrong_role' else 'per_sample_yaw.json')
        value = _read(path)
        if mutation == 'wrong_role':
            value['checkpoint_sha256'] = 'f' * 64
        path.write_text(json.dumps(value) + ' ')
    else:
        sample, metrics = _read(run / 'per_sample_yaw.json'), _read(run / 'metrics_yaw.json')
        sample['P']['0']['unknown' if mutation == 'unknown' else 'edt'] = [None] * 3 + [1.] * 9
        metrics['P'] = _summaries(sample['P'])
        _rebind(run, sample=sample, metrics=metrics)
    with pytest.raises(ValueError):
        rt.build_table(directories)


@pytest.mark.parametrize('count', [2, 12])
def test_finite_count_boundary_and_empty_seed(table_fixture, count):
    run = Path(table_fixture.directories[0])
    sample, metrics = _read(run / 'per_sample_yaw.json'), _read(run / 'metrics_yaw.json')
    sample['P']['0']['edt'][:count] = [None] * count
    metrics['P'] = _summaries(sample['P'])
    _rebind(run, sample=sample, metrics=metrics)
    if count == 12:
        with pytest.raises(ValueError, match='zero finite'):
            rt.build_table(table_fixture.directories)
    else:
        result, _ = rt.build_table(table_fixture.directories)
        cell = next(row for row in result['rows'] if row['role'] == 'cyl' and row['num_shot'] == 1)
        assert cell['metrics']['EDT']['per_seed']['42']['n_finite'] == 10
        assert cell['metrics']['EDT']['per_seed']['42']['mean'] == pytest.approx((1 + 6.5 / 16) * 1000)
