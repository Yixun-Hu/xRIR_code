"""Load exp_07 record assets by path without colliding with exp_03's or exp_04's modules."""
import importlib.util
import sys
from pathlib import Path

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
