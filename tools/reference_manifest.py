"""Deterministic reference-RIR manifest for the AcousticRooms unseen test split.

``xRIR_Dataset`` picks the ``num_shot`` reference RIRs of a query with a worker-local
``np.random.choice`` whose draw is never recorded, so two evaluation runs -- and two
models -- do not see the same conditioning set.  Experiment exp_03 compares a query
against *itself* under a yaw rotation, which only works if the references are pinned.

This module re-derives the same *kind* of draw as a pure function of
``(manifest seed, query path)``: the selection is independent of batch size, worker
count, model and directory listing order, and the resulting manifest (query path +
reference paths) is written once, hashed, and asserted by every run.

The candidate set and the with-replacement fallback mirror
``xRIR_Dataset.get_ir_and_location_for_other_sources`` exactly; only the *choice* is
made deterministic.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np


def select_references(candidates, num_shot, seed, query_key):
    """Deterministically pick ``num_shot`` references for one query.

    The candidates are sorted first, so the result depends only on the *set* of
    candidate paths, never on the order ``os.listdir`` happened to return them in.
    Sampling is without replacement when there are enough candidates and with
    replacement otherwise -- the same fallback ``xRIR_Dataset`` takes.

    Args:
        candidates: candidate reference paths (any iterable of str); not modified.
        num_shot: number of references to draw.
        seed: manifest seed (an int; combined with ``query_key`` into the rng seed).
        query_key: per-query string, normally the query path relative to the IR root.

    Returns:
        A ``list`` of ``num_shot`` paths taken from ``candidates`` (repeats only in the
        with-replacement case).

    Raises:
        ValueError: if ``candidates`` is empty (propagated from ``numpy``).
    """
    ordered = sorted(candidates)
    digest = hashlib.sha256("{}:{}".format(seed, query_key).encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    idx = rng.choice(len(ordered), size=num_shot, replace=len(ordered) < num_shot)
    return [ordered[int(i)] for i in idx]


def candidate_references(ir_path, listdir_cache=None):
    """The candidate reference paths ``xRIR_Dataset`` would consider for one query.

    Reproduces ``xRIR_Dataset.get_ir_and_location_for_other_sources``: every *other*
    source node that appears in the query's room directory, kept when the file
    ``S00<node>_<receiver token>_hybrid_IR.wav`` (the query's own receiver) exists on
    disk.  The naming logic is re-implemented rather than borrowed so building a
    manifest needs no dataset instance and reads no audio.

    Args:
        ir_path: path of the query RIR (``.../<Category>/<Room>/S00i_R00j_hybrid_IR.wav``).
        listdir_cache: optional ``dict`` reused across calls to keep one ``os.listdir``
            per room directory (``build_manifest`` passes one); it is filled in place.

    Returns:
        Sorted list of absolute candidate paths (possibly empty).
    """
    dir_name = os.path.abspath(os.path.dirname(ir_path))
    ir_file_name = os.path.basename(ir_path)
    src_node = int(ir_file_name.split("_")[0][1:])
    rec_token = ir_file_name.split("_")[1]

    if listdir_cache is None:
        listing = os.listdir(dir_name)
    else:
        if dir_name not in listdir_cache:
            listdir_cache[dir_name] = os.listdir(dir_name)
        listing = listdir_cache[dir_name]

    all_src_node = set(int(fn.split("_")[0][1:]) for fn in listing)
    paths = []
    for node in all_src_node.difference({src_node}):
        candidate = os.path.join(dir_name, "S00{}_{}_hybrid_IR.wav".format(node, rec_token))
        if os.path.exists(candidate):
            paths.append(candidate)
    return sorted(paths)
