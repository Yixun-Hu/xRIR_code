"""exp_07 alignment audit: the reviewed measurement on the protocol's own training loader."""
import json
import platform
import subprocess
from pathlib import Path

import pytest
import torch

import train_xRIR_backbone as trainer
from tools import exp07_audit, exp07_train, yaw_aug

ROOT = Path(__file__).resolve().parents[1]


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, split='train', max_len=9600, num_shot=8):
        assert (split, max_len, num_shot) == ('train', 9600, 8)
        self.file_list = ['query_%d.wav' % i for i in range(64)]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return (torch.zeros(3), torch.zeros(3), torch.zeros(3, 1, 512),
                torch.zeros(1, 8), torch.zeros(8, 8), torch.ones(8, 3))


@pytest.fixture
def stubs(monkeypatch):
    """Stub both protocol datasets, the model and CUDA so the audit runs on the CPU."""
    built = []

    def stub(label):
        return lambda **kwargs: (built.append(label), _Dataset(**kwargs))[1]

    monkeypatch.setattr(exp07_train, 'UnseenDataset', stub('unseen'))
    monkeypatch.setattr(exp07_train, 'SeenDataset', stub('seen'))
    monkeypatch.setattr(trainer, 'build_xrir', lambda backbone, num_shot: torch.nn.Linear(3, 3))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    return built


def argv_for(out, protocol='seen', **extra):
    argv = ['--protocol', protocol, '--n-batches', '2', '--batch-size', '16',
            '--num-workers', '0', '--out', str(out)]
    for key, value in extra.items():
        argv += ['--' + key.replace('_', '-'), str(value)]
    return argv


@pytest.mark.parametrize('protocol', ['seen', 'unseen'])
def test_audit_dispatches_and_records_the_protocol(stubs, tmp_path, protocol):
    out = tmp_path / 'audit.json'
    result = exp07_audit.main(argv_for(out, protocol))
    assert stubs == [protocol]
    assert json.loads(out.read_text()) == result
    assert set(result) == {'pairs', 'changed_delays', 'fraction', 'cohort_sha256', 'W',
                           'schema_version', 'passed', 'args', 'env'}
    assert (result['pairs'], result['W'], result['passed']) == (2 * 16 * 8, 512, True)
    assert result['args']['protocol'] == protocol and result['args']['loader_batches'] == 4
    assert result['args']['git_head'] == subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
    assert result['env']['python'] == platform.python_version()
    assert result['cohort_sha256'] and result['schema_version'] == exp07_audit.SCHEMA_VERSION


def test_out_is_required_and_never_overwritten(stubs, tmp_path):
    out = tmp_path / 'audit.json'
    with pytest.raises(SystemExit):
        exp07_audit.parse_args(['--protocol', 'seen', '--num-workers', '0'])
    assert stubs == []
    exp07_audit.main(argv_for(out))
    with pytest.raises(SystemExit):
        exp07_audit.parse_args(argv_for(out))
    for bad in (dict(n_batches=0), dict(batch_size=0), dict(num_workers=-1)):
        with pytest.raises(SystemExit):
            exp07_audit.parse_args(argv_for(tmp_path / 'other.json', **bad))


def test_a_cohort_larger_than_the_loader_is_refused(stubs, tmp_path):
    _, args = exp07_audit.parse_args(argv_for(tmp_path / 'a.json') + ['--n-batches', '9'])
    with pytest.raises(ValueError, match='cohort exceeds'):
        exp07_audit.audit(args)


def test_the_offsets_are_the_training_counters_of_the_protocol(stubs, tmp_path, monkeypatch):
    """The audit draws epoch-1 offsets with the loader length of the selected protocol."""
    seen = []
    real = yaw_aug.YawAug
    monkeypatch.setattr(yaw_aug, 'YawAug', lambda *a: (seen.append(a), real(*a))[1])
    exp07_audit.main(argv_for(tmp_path / 'audit.json'))
    assert seen == [(True, 512, 0, 4)]  # enabled, W, seed, loader length


def test_a_changed_delay_fails_the_audit(stubs, tmp_path, monkeypatch):
    monkeypatch.setattr(exp07_audit.yaw_aug, 'alignment_audit',
                        lambda *a, **k: dict(pairs=8, changed_delays=1, fraction=0.125,
                                             cohort_sha256='f' * 64, W=512))
    with pytest.raises(SystemExit):
        exp07_audit.main(argv_for(tmp_path / 'audit.json'))
    assert json.loads((tmp_path / 'audit.json').read_text())['passed'] is False


def test_the_shared_audit_cli_keeps_mains_behaviour():
    """tools/yaw_aug.py is untouched: no --protocol, and exp_04's default artefact path."""
    assert subprocess.run(['git', 'diff', '--quiet', 'main', '--', 'tools/yaw_aug.py'],
                          cwd=str(ROOT)).returncode == 0
    with pytest.raises(SystemExit):
        yaw_aug._audit_main(['--audit', '--protocol', 'seen'])
