import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import provenance as p


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root).decode().strip()
    git('init', '-q')
    git('config', 'user.email', 'test@example.com')
    git('config', 'user.name', 'Test')
    (root / 'entry.py').write_text('import helper\nprint("import noise")\n')
    (root / 'helper.py').write_text('VALUE = 1\n')
    git('add', 'entry.py', 'helper.py')
    git('commit', '-qm', 'initial')
    return root, git


def test_manifest_round_trip_and_exclusive_create(tmp_path):
    path = tmp_path / 'manifest.json'
    fields = {'checkpoint': 'model.pth', 'seed': 42}
    digest = p.write_manifest(path, fields)
    assert json.loads(path.read_text()) == fields
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        p.write_manifest(path, {'overwritten': True})
    assert p.sha256_file(path) == digest
    p.write_completion(tmp_path / 'completion.json', {'manifest_sha256': digest})
    assert json.loads((tmp_path / 'completion.json').read_text())['manifest_sha256'] == digest
    assert sorted(x.name for x in tmp_path.iterdir()) == ['completion.json', 'manifest.json']


def test_atomic_write_failure_preserves_previous(tmp_path, monkeypatch):
    path = tmp_path / 'completion.json'
    path.write_text('original')
    def fail(*args):
        raise OSError('rename failed')
    monkeypatch.setattr(p.os, 'replace', fail)
    with pytest.raises(OSError):
        p.write_completion(path, {'complete': True})
    assert path.read_text() == 'original'
    assert list(tmp_path.iterdir()) == [path]


def test_closure_and_dirty_state(repo):
    root, git = repo
    head = git('rev-parse', 'HEAD')
    assert p.git_state(root) == {'HEAD': head, 'dirty': False, 'diff_sha256': None}
    files = p.source_closure('entry', root)
    assert files == ['entry.py', 'helper.py']
    records, digest = p.closure_record(files, head, root)
    assert all(r['reviewed_blob_sha256'] == r['working_tree_sha256'] for r in records)
    assert all(r['mtime'] and r['commits_after_reviewed'] == [] for r in records)
    assert digest == hashlib.sha256(json.dumps([
        [r['path'], r['reviewed_blob_sha256']] for r in records], sort_keys=True).encode()).hexdigest()
    (root / 'helper.py').write_text('VALUE = 2\n')
    state = p.git_state(root)
    assert state['dirty'] and state['diff_sha256'] == hashlib.sha256(
        subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root)).hexdigest()
    changed, same_digest = p.closure_record(files, head, root)
    assert same_digest == digest and changed[1]['working_tree_sha256'] != records[1]['working_tree_sha256']
    git('commit', '-qam', 'changed')
    assert p.closure_record(files, head, root)[0][1]['commits_after_reviewed'] == [git('rev-parse', 'HEAD')]
    (root / 'untracked').touch()
    assert p.git_state(root)['dirty']


def test_environment():
    record = p.environment()
    assert set(record) == {'python', 'torch', 'torchaudio', 'torchvision', 'einops',
        'numpy', 'scipy', 'cuda', 'cudnn', 'driver', 'gpus', 'host', 'executable'}
    assert record['executable'] == sys.executable
    assert record['python'].startswith('3.8.')
