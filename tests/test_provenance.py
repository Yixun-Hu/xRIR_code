import hashlib
import json
import os
import shutil
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
    assert p.git_state(root) == {'HEAD': head, 'dirty': False, 'dirty_outside_worklog': False, 'diff_sha256': None}
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
    assert p.git_state(root)['dirty'] and p.git_state(root)['dirty_outside_worklog']


def test_worklog_only_and_rename_status(repo):
    root, git = repo
    (root / 'worklog').mkdir()
    (root / 'worklog/note with\nnewline').write_text('notes')
    assert p.git_state(root)['dirty'] and not p.git_state(root)['dirty_outside_worklog']
    git('mv', 'helper.py', 'worklog/helper.py')
    assert p.git_state(root)['dirty_outside_worklog']


def test_completion_respects_umask(tmp_path):
    previous = os.umask(0o027)
    try:
        p.write_completion(tmp_path / 'done.json', {})
    finally:
        os.umask(previous)
    assert (tmp_path / 'done.json').stat().st_mode & 0o777 == 0o640


@pytest.mark.parametrize('key', ['checkpoint_sha256', 'data_identity', 'source_closures'])
def test_required_inputs_cannot_be_omitted(key):
    assert p.revalidate({}, required=(key,)) == ['missing.' + key]


@pytest.mark.parametrize('symlink', [False, True])
def test_closure_refuses_foreign_dependencies(repo, tmp_path, symlink):
    root, _ = repo
    foreign = tmp_path / 'foreign.py'
    foreign.write_text('VALUE = 3\n')
    if symlink:
        (root / 'helper.py').unlink()
        (root / 'helper.py').symlink_to(foreign)
    else:
        (root / 'entry.py').write_text('import sys\nsys.path.append(%r)\nimport foreign\n' % str(tmp_path))
    with pytest.raises(RuntimeError, match=str(foreign)):
        p.source_closure('entry', root)


def test_closure_does_not_fall_back_to_another_checkout(tmp_path, monkeypatch):
    original = Path(__file__).resolve().parents[1]
    copy = tmp_path / 'copy'
    for name in subprocess.check_output(['git', 'ls-files', '*.py'], cwd=original, text=True).splitlines():
        if name == 'tools/provenance.py':
            continue
        target = copy / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, target)
    monkeypatch.setenv('PYTHONPATH', str(original))
    with pytest.raises(RuntimeError, match='provenance'):
        p.source_closure('tools.exp04_eval', copy)


def test_exp03_closure_and_reviewed_digest_are_unchanged():
    repo = Path(__file__).resolve().parents[1]
    expected = ['eval_unseen.py', 'eval_xRIR_backbone.py', 'eval_yaw_rotation.py',
        'model/cylindrical_vit.py', 'model/simple_vit.py', 'model/xRIR.py', 'model/xRIR_cyl.py',
        'tools/per_sample_metrics.py', 'tools/reference_manifest.py', 'tools/yaw_rotation.py',
        'treble_multi_room_dataset/treble_xRIR_dataset.py', 'utils/spec_utils.py']
    assert p.source_closure('eval_yaw_rotation', repo) == expected
    records, digest = p.closure_record(expected, '62c9107b4150e44c4ac410ff4cab359c1e71cc10', repo)
    assert digest == '5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48'
    assert all(r['working_tree_sha256'] == r['reviewed_blob_sha256'] and
               not r['commits_after_reviewed'] for r in records)


def test_environment():
    record = p.environment()
    assert set(record) == {'python', 'torch', 'torchaudio', 'torchvision', 'einops',
        'numpy', 'scipy', 'cuda', 'cudnn', 'driver', 'gpus', 'host', 'executable'}
    assert record['executable'] == sys.executable
    assert record['python'].startswith('3.8.')


@pytest.fixture
def data(tmp_path):
    root = tmp_path / 'data'
    wavs = ['Apartments/room/S0001_R0002_hybrid_IR.wav',
            'Apartments/room/S003_R002_hybrid_IR.wav']
    names = ['single_channel_ir/' + w for w in wavs] + [
        'metadata/Apartments/room/S001_R002.json',
        'metadata/Apartments/room/S003_R002.json', 'depth_map/Apartments/room/2.npy']
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    manifest = tmp_path / 'references.json'
    manifest.write_text(json.dumps({'seed': 42, 'num_shot': 1,
        'ir_root': str(root / 'single_channel_ir'), 'entries': [
            {'index': 0, 'query': wavs[0], 'refs': [wavs[1]]}]}))
    return root, manifest, names


def test_data_inventory_hashes_every_referenced_file(data):
    from tools.reference_manifest import load_manifest, manifest_hash
    root, manifest, names = data
    record = p.data_identity(manifest, root)
    assert record['manifest_hash'] == manifest_hash(load_manifest(manifest))
    assert record['manifest_file_sha256'] == p.sha256_file(manifest)
    assert record['inventory_files'] == 5 and record['inventory_missing'] == 0
    assert [r['path'] for r in record['inventory']] == sorted(names)
    assert record['inventory_bytes'] == sum((root / n).stat().st_size for n in names)
    assert all(r['sha256'] == p.sha256_file(root / r['path']) for r in record['inventory'])
    (root / names[-1]).write_bytes(b'changed depth')
    assert p.data_identity(manifest, root)['inventory_sha256'] != record['inventory_sha256']
    (root / names[0]).unlink()
    with pytest.raises(FileNotFoundError):
        p.data_identity(manifest, root)


@pytest.mark.parametrize('field', ['checkpoint', 'manifest_path', 'source', 'data', 'eval_manifest'])
def test_revalidate_detects_mutable_inputs(data, tmp_path, field):
    root, manifest, names = data
    checkpoint, source = tmp_path / 'checkpoint', tmp_path / 'source.py'
    checkpoint.write_bytes(b'weights')
    source.write_text('old source')
    fields = {'repo': str(tmp_path), 'checkpoint': str(checkpoint),
        'checkpoint_sha256': p.sha256_file(checkpoint), 'manifest_path': str(manifest),
        'manifest_file_sha256': p.sha256_file(manifest),
        'data_identity': p.data_identity(manifest, root), 'evaluator_closure': {'files': [
            {'path': source.name, 'working_tree_sha256': p.sha256_file(source)}]}}
    eval_manifest = tmp_path / 'eval_manifest.json'
    fields['eval_manifest'] = {'path': str(eval_manifest),
                               'sha256': p.write_manifest(eval_manifest, fields)}
    assert p.revalidate(fields) == []
    target = {'checkpoint': checkpoint, 'manifest_path': manifest, 'source': source,
              'data': root / names[-1], 'eval_manifest': eval_manifest}[field]
    target.write_bytes(b'changed')
    assert any(field in mismatch for mismatch in p.revalidate(fields))
    target.unlink()
    assert p.revalidate(fields)


@pytest.mark.parametrize('change', ['bytes', 'mtime', 'name', 'missing'])
def test_training_inventory_cache_is_checked(data, tmp_path, monkeypatch, change):
    import os
    from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset
    root, _, names = data
    for category in ['Bathrooms', 'Cafe', 'LivingRoomsWithHallway', 'Office',
                     'Auditorium', 'Bedrooms', 'ListeningRoom', 'MeetingRoom', 'Restaurants']:
        (root / 'single_channel_ir' / category).mkdir()
    held_out = root / 'single_channel_ir/Apartments/Apartments_idx_50'
    held_out.mkdir()
    (held_out / 'excluded.wav').write_bytes(b'test split')
    cache = tmp_path / 'train_cache.json'
    result = p.train_data_identity(root, cache_path=cache, workers=2)
    listed = xRIR_Dataset(split='train', ir_path=str(root / 'single_channel_ir')).file_list
    assert result['inventory_files'] == len(listed) == 2
    assert all(r['path'].startswith('single_channel_ir/') for r in result['inventory'])
    assert cache.is_file()
    assert cache.stat().st_mode & 0o777 == (root / names[0]).stat().st_mode & 0o777
    def no_hash(*args):
        pytest.fail('valid cache should avoid content rereads')
    with monkeypatch.context() as patch:
        patch.setattr(p, 'sha256_file', no_hash)
        assert p.train_data_identity(root, cache_path=cache) == result
    path = root / names[0]
    if change == 'bytes':
        path.write_bytes(b'changed')
    elif change == 'mtime':
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1000000000))
    elif change == 'name':
        path.rename(path.with_name('renamed.wav'))
    else:
        path.unlink()
    with pytest.raises(ValueError, match='stale'):
        p.train_data_identity(root, cache_path=cache)
    cache.unlink()
    updated = p.train_data_identity(root, cache_path=cache)
    assert (updated['inventory_sha256'] == result['inventory_sha256']) == (change == 'mtime')


def test_revalidate_checks_writer_and_extra_inputs(tmp_path):
    file = tmp_path / 'writer.py'
    file.write_bytes(b'original')
    record = {'path': str(file), 'sha256': p.sha256_file(file)}
    fields = {'mutable_inputs': {'training_args': record}, 'source_closures': {'writer': {
        'files': [dict(record, working_tree_sha256=record['sha256'])]}}}
    assert p.revalidate(fields) == []
    file.write_bytes(b'mutated')
    assert p.revalidate(fields) == ['training_args', 'source.writer.' + str(file)]
