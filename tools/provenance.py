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


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_closure(entry_module, repo):
    """Import in a fresh interpreter, keeping only files under this repository."""
    code = '''import importlib, json, os, sys
repo = os.path.realpath(sys.argv[1])
sys.path.insert(0, repo)
importlib.import_module(sys.argv[2])
files = {os.path.realpath(m.__file__) for m in list(sys.modules.values())
         if getattr(m, '__file__', None)}
print('CLOSURE_JSON:' + json.dumps(sorted(os.path.relpath(f, repo) for f in files
    if f.startswith(repo + os.sep) and 'site-packages' not in f)))
'''
    output = subprocess.check_output([sys.executable, '-c', code, str(Path(repo).resolve()),
                                      entry_module], cwd=repo, text=True)
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
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo))
    diff = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=repo) if dirty else None
    return {'HEAD': head, 'dirty': dirty,
            'diff_sha256': hashlib.sha256(diff).hexdigest() if dirty else None}


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
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
