"""exp_07 record helpers: load its assets by path, and publish its canonical products."""
import datetime
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from tools.paired_compare import recheck_inputs
from tools.results_table import _json_bytes, _preserve_manual, _publish

ASSETS = Path(__file__).resolve().parents[1] / ('worklog/worklog_yixun/'
    'exp_07_seen_protocol_claude/seen_protocol_results_assets')


def load_asset(name):
    key = 'exp07_record_' + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ASSETS / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[key]
            raise
    return sys.modules[key]


def write_outputs(result, admitted, json_path, md_path, renderer, force_md=False, command=()):
    """exp_04's publication transaction with the document renderer supplied by the caller.

    Amendment A1: ``tools.results_table.write_outputs`` keeps main's bytes -- exp_04's
    bound record re-hashes it -- and it renders with its own ``render_markdown``, so the
    exp_07 producers need this copy.  It is the reviewed transaction verbatim apart from
    ``renderer``: distinct, absent, non-symlink outputs that never overlap an input; the
    inputs rechecked before and after rendering; JSON published first, then the Markdown
    (with any manual section preserved), then the provenance sidecar; and every created
    file rolled back, the Markdown restored to its previous bytes, on any failure.
    """
    paths = [Path(json_path).absolute(), Path(md_path).absolute(),
             Path(str(json_path) + '.provenance.json').absolute()]
    if (len({item.resolve() for item in paths}) != 3 or any(item.is_symlink() for item in paths)
            or any(os.path.lexists(paths[i]) for i in (0, 2))
            or (os.path.lexists(paths[1]) and not force_md)):
        raise FileExistsError('output paths must be distinct and absent; '
                              '--force-md permits Markdown only')
    if any(str(item.resolve()) in admitted['inputs'] for item in paths):
        raise ValueError('output overlaps an input')
    previous = paths[1].read_bytes() if paths[1].exists() else None
    data, created = _json_bytes(result), []
    recheck_inputs(admitted)
    try:
        _publish(paths[0], data)
        created.append(paths[0])
        markdown = _preserve_manual(renderer(paths[0], command), previous).encode()
        receipt = dict(schema_version=1, profile_digest=result['profile_digest'],
            inputs=admitted['inputs'], producer=admitted['producer'],
            approved_digests=admitted['approved_digests'], run_flags=admitted['run_flags'],
            generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            generation_command=list(command),
            outputs={str(path): hashlib.sha256(payload).hexdigest()
                     for path, payload in zip(paths, (data, markdown))})
        recheck_inputs(admitted)
        current = paths[1].read_bytes() if paths[1].exists() else None
        if current != previous:
            raise ValueError('Markdown changed during rendering')
        _publish(paths[1], markdown, overwrite=previous is not None)
        created.append(paths[1])
        _publish(paths[2], _json_bytes(receipt))
    except BaseException:
        for path in reversed(created):
            if path == paths[1] and previous is not None:
                _publish(path, previous, overwrite=True)
            else:
                path.unlink()
        raise
