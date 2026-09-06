"""Pytest configuration for the xRIR test suite.

The project is used as a plain source tree (no installed package), so the repository
root is prepended to ``sys.path``.  ``XRIR_DATA_PATH`` is defaulted to the local
AcousticRooms cache: ``treble_multi_room_dataset.treble_xRIR_dataset`` reads that
variable *at import time*, so it has to be set before any test imports the dataset.
An externally provided value always wins, and the fallback is applied only when that
directory actually exists, so the suite stays portable to machines without the cache.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_DATA_CACHE_FALLBACK = "/home/yixunhu/data_cache/AcousticRooms"
if "XRIR_DATA_PATH" not in os.environ and os.path.isdir(_DATA_CACHE_FALLBACK):
    os.environ["XRIR_DATA_PATH"] = _DATA_CACHE_FALLBACK
