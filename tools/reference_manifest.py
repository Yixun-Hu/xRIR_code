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
import json
import os

import numpy as np
import torch
import torchaudio

from treble_multi_room_dataset.treble_xRIR_dataset import (
    convert_equirect_to_camera_coord,
    get_3d_point_camera_coord,
)


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


def build_manifest(dataset, seed, num_shot=8):
    """Build the reference manifest for every query of ``dataset``.

    Reads directory listings only (no audio, no metadata, no depth maps), caching one
    ``os.listdir`` per room directory, so the full 6337-query test split takes a couple
    of seconds.

    Args:
        dataset: an ``xRIR_Dataset`` (only ``file_list`` and ``ir_path`` are used).
        seed: manifest seed; part of every per-query rng seed.
        num_shot: references per query.

    Returns:
        ``{"seed", "num_shot", "ir_root", "entries"}`` where ``entries`` is one dict per
        query -- ``{"index", "query", "refs"}`` -- in ``dataset.file_list`` order, with
        ``query`` / ``refs`` relative to ``ir_root``.  ``query`` is also the key the
        reference draw is seeded with.
    """
    ir_root = dataset.ir_path
    listdir_cache = {}
    entries = []
    for index, ir_path in enumerate(dataset.file_list):
        query = os.path.relpath(ir_path, ir_root)
        refs = select_references(
            candidate_references(ir_path, listdir_cache=listdir_cache), num_shot, seed, query)
        entries.append({"index": index, "query": query,
                        "refs": [os.path.relpath(ref, ir_root) for ref in refs]})
    return {"seed": seed, "num_shot": num_shot, "ir_root": ir_root, "entries": entries}


def manifest_hash(manifest):
    """Hex sha256 of the manifest's *content* -- seed, num_shot and entries.

    ``ir_root`` is deliberately excluded so the same selection hashes identically no
    matter where the dataset is mounted.
    """
    payload = {"seed": manifest["seed"], "num_shot": manifest["num_shot"],
               "entries": manifest["entries"]}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_manifest(manifest, path):
    """Write ``manifest`` to ``path`` as JSON, creating the parent directory."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as fout:
        json.dump(manifest, fout)
    return path


def load_manifest(path):
    """Read a manifest written by :func:`save_manifest`."""
    with open(path, "r") as fin:
        return json.load(fin)


class ManifestDataset(torch.utils.data.Dataset):
    """``xRIR_Dataset`` with its reference draw replaced by a manifest.

    Yields the exp_01 six-tuple -- ``(listener_pos[3], src_local[3],
    depth_coord[3, 256, 512], tgt_wav[1, L], ref_irs[K, L], ref_src_local[K, 3])`` --
    built exactly like ``xRIR_Dataset.__getitem__``, plus a seventh element: the query
    path relative to the IR root, which is the key every per-sample metric is stored
    under (and the key the Griffin-Lim phase seed is derived from).

    Paths are resolved against ``dataset.ir_path``, not against the manifest's recorded
    ``ir_root``, so a manifest stays usable when the cache is mounted elsewhere.

    Args:
        dataset: the ``xRIR_Dataset`` the manifest was built from.
        manifest: the manifest dict; its query list and ``num_shot`` must match.

    Raises:
        AssertionError: if the manifest was not built from this dataset/split, or was
            built for a different ``num_shot``.
    """

    def __init__(self, dataset, manifest):
        entries = manifest["entries"]
        assert [entry["query"] for entry in entries] == \
            [os.path.relpath(path, dataset.ir_path) for path in dataset.file_list], \
            "manifest queries do not match dataset.file_list (wrong split, order or root)"
        assert manifest["num_shot"] == dataset.num_shot, \
            "manifest num_shot {} != dataset num_shot {}".format(
                manifest["num_shot"], dataset.num_shot)
        self.dataset = dataset
        self.manifest = manifest
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def _load_ir(self, ir_file_path):
        """Load one RIR and zero-pad / truncate it to ``max_len``, as the dataset does."""
        wav, rate = torchaudio.load(ir_file_path)
        assert rate == 22050, "IR sampling rate must be 22050!"
        max_len = self.dataset.max_len
        if wav.shape[1] < max_len:
            wav = torch.cat([wav, torch.zeros(wav.shape[0], max_len - wav.shape[1])], dim=1)
        else:
            wav = wav[:, :max_len]
        return wav

    def __getitem__(self, idx):
        entry = self.entries[idx]
        dataset = self.dataset
        ir_file_path = os.path.join(dataset.ir_path, entry["query"])
        ir_file_name = os.path.basename(ir_file_path).split("_hybrid_IR")[0]
        scene_name = ir_file_path.split("/")[-3]
        scene_id = ir_file_path.split("/")[-2]
        receiver_idx = int(ir_file_name.split("_")[1][1:])

        source_pos, listener_pos = dataset.get_receiver_source_location(ir_file_path)
        proj_source_pos = get_3d_point_camera_coord(0, listener_pos, source_pos)
        proj_listener_pos = np.array([0., 0., 0.])

        pano_depth = np.load(os.path.join(dataset.pano_depth_path, scene_name, scene_id,
                                          "{}.npy".format(receiver_idx)))
        depth_coord = convert_equirect_to_camera_coord(torch.from_numpy(pano_depth), 256, 512)
        tgt_wav = self._load_ir(ir_file_path)

        all_ref_irs = []
        all_ref_src_pos = []
        for ref in entry["refs"]:
            ref_file_path = os.path.join(dataset.ir_path, ref)
            all_ref_irs.append(self._load_ir(ref_file_path))
            src_loc, rec_loc = dataset.get_receiver_source_location(ref_file_path)
            all_ref_src_pos.append(
                torch.Tensor(get_3d_point_camera_coord(0, rec_loc, src_loc)).float())

        return (torch.Tensor(proj_listener_pos).float(),
                torch.Tensor(proj_source_pos).float(),
                torch.Tensor(depth_coord).permute(2, 0, 1).float(),
                tgt_wav,
                torch.cat(all_ref_irs, dim=0),
                torch.vstack(all_ref_src_pos),
                entry["query"])
