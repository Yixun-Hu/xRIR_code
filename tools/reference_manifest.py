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
