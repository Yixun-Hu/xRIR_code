"""Execution-bound provenance helpers; paths and reviewed commits are explicit."""
import datetime
import importlib
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_closure(entry_module, repo):
    """Import hermetically; refuse files outside the repo and interpreter prefixes."""
    repo = Path(repo).resolve()
    code = '''import importlib, json, os, sys
repo = os.path.realpath(sys.argv[1])
sys.path = [repo] + [p for p in sys.path if p]
importlib.import_module(sys.argv[2])
files = {os.path.realpath(m.__file__) for m in list(sys.modules.values())
         if getattr(m, '__file__', None)}
prefixes = [repo, os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix)]
foreign = sorted(f for f in files if not any(f.startswith(p + os.sep) for p in prefixes))
if foreign:
    raise RuntimeError('FOREIGN dependencies: ' + ', '.join(foreign))
print('CLOSURE_JSON:' + json.dumps(sorted(os.path.relpath(f, repo) for f in files
    if f.startswith(repo + os.sep))))
'''
    try:
        output = subprocess.check_output([sys.executable, '-c', code, str(repo), entry_module],
            cwd=repo, text=True, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH=str(repo)))
    except subprocess.CalledProcessError as error:
        raise RuntimeError('source_closure failed: ' + error.output) from error
    return json.loads(next(line[len('CLOSURE_JSON:'):] for line in reversed(output.splitlines())
                           if line.startswith('CLOSURE_JSON:')))


def closure_record(files, reviewed_commit, repo):
    """Return file records and the exp_03-compatible digest of reviewed blobs."""
    records = []
    for name in sorted(set(files)):
        path = Path(repo) / name
        path.resolve().relative_to(Path(repo).resolve())
        blob = subprocess.run(['git', 'show', '{}:{}'.format(reviewed_commit, name)],
                              cwd=repo, capture_output=True)
        later = subprocess.check_output(['git', 'log', '--format=%H',
            '{}..HEAD'.format(reviewed_commit), '--', name], cwd=repo, text=True).split()
        records.append({'path': name, 'reviewed_blob_sha256':
            hashlib.sha256(blob.stdout).hexdigest() if blob.returncode == 0 else None,
            'working_tree_sha256': sha256_file(path), 'commits_after_reviewed': later,
            'mtime': datetime.datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()})
    digest = hashlib.sha256(json.dumps([[r['path'], r['reviewed_blob_sha256']]
        for r in records], sort_keys=True).encode()).hexdigest()
    return records, digest


def git_state(repo):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain', '-z'], cwd=repo, text=True)
    paths, entries = [], iter(status.split('\0'))
    for entry in entries:
        if entry:
            paths.append(entry[3:])
            if 'R' in entry[:2] or 'C' in entry[:2]:
                paths.append(next(entries))  # Include the source path of a rename/copy.
    dirty = bool(status)
    diff = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=repo) if dirty else None
    return {'HEAD': head, 'dirty': dirty,
            'dirty_outside_worklog': any(not path.startswith('worklog/') for path in paths),
            'diff_sha256': hashlib.sha256(diff).hexdigest() if dirty else None}


def checked_git_state(repo, confirmatory, allow_dirty=False):
    """Gate confirmatory launches, allowing notebook-only edits without an override."""
    state = git_state(repo)
    if state['dirty_outside_worklog']:
        if confirmatory and not allow_dirty:
            raise ValueError('dirty_outside_worklog requires --allow-dirty')
        print('WARNING: dirty_outside_worklog', flush=True)
    return state


def environment():
    versions = {name: importlib.import_module(name).__version__ for name in
                ('torch', 'torchaudio', 'torchvision', 'einops', 'numpy', 'scipy')}
    torch = importlib.import_module('torch')
    try:
        driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
            '--format=csv,noheader'], stderr=subprocess.DEVNULL, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        driver = None
    return dict(versions, python=platform.python_version(), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), driver=driver, host=platform.node(),
        executable=sys.executable,
        gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


def write_manifest(path, fields):
    """Exclusively create strict JSON; return the digest of exactly those bytes."""
    payload = json.dumps(fields, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    with open(path, 'xb') as stream:  # Python's x mode uses O_CREAT | O_EXCL.
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def write_completion(path, fields):
    """Publish the complete record with an atomic rename on the same filesystem."""
    payload = json.dumps(fields, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    fd, temporary = tempfile.mkstemp(prefix='.' + Path(path).name, dir=Path(path).parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            mask = os.umask(0)
            os.umask(mask)
            os.fchmod(stream.fileno(), 0o666 & ~mask)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _inventory_digest(records):
    return hashlib.sha256(json.dumps([[r['path'], r['sha256']] for r in records],
                                    sort_keys=True).encode()).hexdigest()


def _inventory(files, data_root, workers=8):
    root = Path(data_root).resolve()
    def record(name):
        path = root / name
        path.resolve().relative_to(root)
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('data changed while hashing: ' + name)
        return {'path': name, 'sha256': digest, 'size': after.st_size,
                'mtime_ns': after.st_mtime_ns}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(record, sorted(set(files))))
    # Missing files raise above; retain the zero field for exp_03 schema compatibility.
    return {'data_root': str(root), 'inventory': records, 'inventory_files': len(records),
        'inventory_missing': 0, 'inventory_bytes': sum(r['size'] for r in records),
        'inventory_sha256': _inventory_digest(records)}


def data_identity(manifest_path, data_root):
    """Hash every query/reference IR, its unpadded-id metadata, and receiver depth."""
    from tools.reference_manifest import load_manifest, manifest_hash
    manifest = load_manifest(manifest_path)
    files = set()
    for entry in manifest['entries']:
        for wav in [entry['query']] + entry['refs']:
            relative = wav.split('single_channel_ir/')[-1]
            room, name = relative.rsplit('/', 1)
            src, rec = [int(token[1:]) for token in name.split('_')[:2]]
            files.update(('single_channel_ir/' + relative,
                'metadata/{}/S00{}_R00{}.json'.format(room, src, rec),
                'depth_map/{}/{}.npy'.format(room, rec)))
    return dict(_inventory(files, data_root), manifest_path=str(Path(manifest_path).resolve()),
        manifest_file_sha256=sha256_file(manifest_path), manifest_hash=manifest_hash(manifest))


def train_data_identity(data_root, cache_path=None, workers=8):
    """Content-hash train IRs using the actual split, with a stat-validated cache.

    A stale cache raises instead of silently reusing it. Remove that cache explicitly
    to rebuild. The default cache is outside the dataset, keyed by root and count.
    """
    from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset
    root = Path(data_root).resolve()
    dataset = xRIR_Dataset(split='train', ir_path=str(root / 'single_channel_ir'))
    files = sorted(str(Path(f).resolve().relative_to(root)) for f in dataset.file_list)
    key = hashlib.sha256(json.dumps([str(root), len(files)]).encode()).hexdigest()
    cache = Path(cache_path) if cache_path else Path(tempfile.gettempdir()) / 'xrir-provenance' / (key + '.json')
    if cache.exists():
        try:
            record = json.loads(cache.read_text())
            stamps = []
            for name in files:
                stat = (root / name).stat()
                stamps.append((name, stat.st_size, stat.st_mtime_ns))
            cached = [(r['path'], r['size'], r['mtime_ns']) for r in record['inventory']]
            if (record['data_root'] == str(root) and record['cache_key'] == key
                    and cached == stamps and record['inventory_files'] == len(files)
                    and _inventory_digest(record['inventory']) == record['inventory_sha256']):
                return record
        except (OSError, ValueError, KeyError, TypeError):
            pass
        raise ValueError('stale training inventory cache: ' + str(cache))
    record = dict(_inventory(files, root, workers), split='train', cache_key=key)
    cache.parent.mkdir(parents=True, exist_ok=True)
    write_completion(cache, record)
    return record


def revalidate(manifest, required=()):
    """Rehash declared inputs; missing required keys and missing files are mismatches.

    A launcher adds eval_manifest={path, sha256} in memory after exclusive creation,
    avoiding a self-referential digest in the serialized manifest itself.
    """
    mismatches = ['missing.' + key for key in required if key not in manifest]
    repo = Path(manifest.get('repo', '.'))
    def check(label, path, expected):
        try:
            actual = sha256_file(path)
        except OSError:
            actual = None
        if expected is None or actual != expected:
            mismatches.append(label)
    for key, digest in [('checkpoint', 'checkpoint_sha256'),
                         ('manifest_path', 'manifest_file_sha256')]:
        if key in manifest:
            check(key, repo / manifest[key], manifest.get(digest))
    for name, record in manifest.get('mutable_inputs', {}).items():
        check(name, repo / record['path'], record.get('sha256'))
    if 'eval_manifest' in manifest:
        record = manifest['eval_manifest']
        check('eval_manifest', record['path'], record.get('sha256'))
    closures = dict(manifest.get('source_closures', {}))
    if 'evaluator_closure' in manifest:
        closures['evaluator'] = manifest['evaluator_closure']
    for name, closure in closures.items():
        for record in closure['files']:
            check('source.{}.{}'.format(name, record['path']), repo / record['path'],
                  record.get('working_tree_sha256'))
    for key in ('data_identity', 'train_data_identity'):
        if key not in manifest:
            continue
        identity = manifest[key]
        for record in identity['inventory']:
            check(key + '.' + record['path'], Path(identity['data_root']) / record['path'],
                  record.get('sha256'))
        if 'manifest_path' in identity:
            check(key + '.manifest_path', identity['manifest_path'], identity.get('manifest_file_sha256'))
        if _inventory_digest(identity['inventory']) != identity.get('inventory_sha256'):
            mismatches.append(key + '.inventory_sha256')
    return mismatches
