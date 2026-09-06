"""Tests for :mod:`tools.reference_manifest` (exp_03, yaw_rotation_degradation).

The manifest pins, once and for all, which 8 reference RIRs every query of the
AcousticRooms unseen test split is evaluated with, so that k=0 and k!=0 -- and every
model -- see exactly the same conditioning set.  ``xRIR_Dataset`` itself draws the
references with a worker-local ``np.random`` call that is never recorded, which is why
the draw is re-implemented here as a pure function of (seed, query path).
"""
import os

import pytest

from tools.reference_manifest import (
    build_manifest,
    candidate_references,
    load_manifest,
    manifest_hash,
    save_manifest,
    select_references,
)


# --------------------------------------------------------------------------------------
# T9a -- select_references
# --------------------------------------------------------------------------------------
def _candidates(n, prefix="/ir/Office/Office_idx_777/S00"):
    return ["{}{}_R000_hybrid_IR.wav".format(prefix, i) for i in range(1, n + 1)]


def test_select_references_is_deterministic_in_seed_and_key():
    cands = _candidates(9)
    a = select_references(cands, 8, 0, "Office/Office_idx_777/S001_R000_hybrid_IR.wav")
    b = select_references(cands, 8, 0, "Office/Office_idx_777/S001_R000_hybrid_IR.wav")
    assert a == b
    assert isinstance(a, list) and all(isinstance(p, str) for p in a)
    assert len(a) == 8
    assert set(a).issubset(set(cands))


def test_select_references_changes_with_seed_and_with_key():
    cands = _candidates(9)
    key = "Office/Office_idx_777/S001_R000_hybrid_IR.wav"
    base = select_references(cands, 8, 0, key)
    assert select_references(cands, 8, 1, key) != base
    assert select_references(cands, 8, 0, key.replace("S001", "S002")) != base


def test_select_references_returns_the_full_set_when_counts_match():
    """8 candidates, 8 shots -> sampling without replacement must return all of them."""
    cands = _candidates(8)
    for seed in (0, 1, 7):
        got = select_references(cands, 8, seed, "q")
        assert len(got) == 8
        assert sorted(got) == sorted(cands)


def test_select_references_falls_back_to_replacement_when_short():
    """6 candidates, 8 shots -> the dataset's with-replacement fallback."""
    cands = _candidates(6)
    got = select_references(cands, 8, 0, "q")
    assert len(got) == 8
    assert set(got).issubset(set(cands))
    assert len(set(got)) < 8, "a with-replacement draw of 8 from 6 must repeat"


def test_select_references_is_independent_of_candidate_order():
    cands = _candidates(9)
    shuffled = list(reversed(cands))
    assert shuffled != cands
    assert select_references(shuffled, 8, 0, "q") == select_references(cands, 8, 0, "q")
    # Absolute paths differing only in directory order are sorted the same way.
    assert select_references(sorted(cands), 8, 3, "q") == select_references(cands, 8, 3, "q")


def test_select_references_does_not_mutate_the_candidate_list():
    cands = _candidates(9)
    original = list(cands)
    select_references(cands, 8, 0, "q")
    assert cands == original


def test_select_references_matches_the_documented_hash_construction():
    """The rng seed is sha256("<seed>:<query key>")[:8] read little-endian."""
    import hashlib

    import numpy as np

    cands = _candidates(9)
    seed, key = 0, "Office/Office_idx_777/S001_R000_hybrid_IR.wav"
    digest = hashlib.sha256("{}:{}".format(seed, key).encode()).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    expected = [sorted(cands)[i] for i in rng.choice(len(cands), size=8, replace=False)]
    assert select_references(cands, 8, seed, key) == expected


# --------------------------------------------------------------------------------------
# A synthetic AcousticRooms-like tree, used by every test below.
#
# ``xRIR_Dataset`` is instantiated for real against it (its ``BASE_DATA_PATH`` module
# global is monkeypatched), rather than faked with a stand-in object, so the file naming,
# the ``glob``-derived ``file_list`` and ``get_receiver_source_location`` under test are
# the production ones.  ``__init__`` lists every scene category, so all ten must exist.
# --------------------------------------------------------------------------------------
CATEGORIES = ["Apartments", "Bathrooms", "Cafe", "LivingRoomsWithHallway", "Office",
              "Auditorium", "Bedrooms", "ListeningRoom", "MeetingRoom", "Restaurants"]
# (category, room, sources, receivers, missing "S<src>_R<rec>" pairs); no room name may
# contain a TEST_ROOMS name, so all three land in the "train" split.
ROOM_A = ("Office", "Office_idx_777", list(range(1, 11)), [0, 1], [])      # 9 candidates
ROOM_B = ("Bedrooms", "Bedrooms_idx_701", list(range(1, 8)), [0], [])      # 6 -> replacement
ROOM_C = ("Cafe", "Cafe_idx_900", [1, 2, 3, 4], [0, 1], [(3, 1)])         # a hole on disk
ROOMS = [ROOM_A, ROOM_B, ROOM_C]
WAV_LEN = 400
DEPTH_H, DEPTH_W = 256, 512


def _build_tree(root):
    import json

    import numpy as np
    import torch
    import torchaudio

    for cat in CATEGORIES:
        os.makedirs(os.path.join(root, "single_channel_ir", cat), exist_ok=True)
    rng = np.random.RandomState(0)
    for cat, room, sources, receivers, missing in ROOMS:
        ir_dir = os.path.join(root, "single_channel_ir", cat, room)
        meta_dir = os.path.join(root, "metadata", cat, room)
        depth_dir = os.path.join(root, "depth_map", cat, room)
        for d in (ir_dir, meta_dir, depth_dir):
            os.makedirs(d, exist_ok=True)
        for rec in receivers:
            np.save(os.path.join(depth_dir, "{}.npy".format(rec)),
                    (rng.rand(DEPTH_H, DEPTH_W) * 4.0 + 1.0).astype(np.float32))
        for src in sources:
            for rec in receivers:
                if (src, rec) in missing:
                    continue
                stem = "S00{}_R00{}".format(src, rec)
                wav = torch.from_numpy((rng.rand(1, WAV_LEN) * 2.0 - 1.0).astype(np.float32))
                torchaudio.save(os.path.join(ir_dir, stem + "_hybrid_IR.wav"), wav, 22050)
                with open(os.path.join(meta_dir, stem + ".json"), "w") as fout:
                    json.dump({"src_loc": [float(src), 0.5 * src, 1.2],
                               "rec_loc": [-1.0 * rec, 2.0 + rec, 1.0]}, fout)
    return root


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    return _build_tree(str(tmp_path_factory.mktemp("acoustic_rooms")))


@pytest.fixture
def synthetic_dataset(synthetic_root, monkeypatch):
    from treble_multi_room_dataset import treble_xRIR_dataset as ds_mod

    monkeypatch.setattr(ds_mod, "BASE_DATA_PATH", synthetic_root)
    dataset = ds_mod.xRIR_Dataset(split="train", num_shot=8, max_len=512)
    assert len(dataset) == 20 + 7 + 7
    return dataset


def _query(dataset, name):
    """The absolute path of one query, by ``<Room>/<stem>_hybrid_IR.wav`` suffix."""
    matches = [p for p in dataset.file_list if p.endswith(name)]
    assert len(matches) == 1, "{} -> {}".format(name, matches)
    return matches[0]


# --------------------------------------------------------------------------------------
# T9b (part 1) -- candidate_references reproduces the dataset's candidate set
# --------------------------------------------------------------------------------------
def test_candidate_references_matches_the_dataset_rule(synthetic_dataset):
    query = _query(synthetic_dataset, "Office_idx_777/S001_R000_hybrid_IR.wav")
    got = candidate_references(query)

    assert got == sorted(got), "candidates must be returned sorted"
    assert all(os.path.isabs(p) and os.path.exists(p) for p in got)
    assert len(got) == 9
    assert query not in got
    assert all(os.path.basename(p).split("_")[0] != "S001" for p in got), "own source leaked in"
    assert all(os.path.basename(p).split("_")[1] == "R000" for p in got), "wrong receiver"
    # Independently re-derived from the directory listing.
    expected = sorted(os.path.join(os.path.dirname(query), "S00{}_R000_hybrid_IR.wav".format(s))
                      for s in range(2, 11))
    assert got == expected


def test_candidate_references_counts_per_room(synthetic_dataset):
    # 7 sources, 1 receiver -> 6 candidates (the with-replacement case).
    assert len(candidate_references(_query(synthetic_dataset,
                                          "Bedrooms_idx_701/S001_R000_hybrid_IR.wav"))) == 6
    # 4 sources, but S003_R001 is missing on disk -> that pair is skipped for R001 only.
    got_r1 = candidate_references(_query(synthetic_dataset, "Cafe_idx_900/S001_R001_hybrid_IR.wav"))
    assert [os.path.basename(p) for p in got_r1] == ["S002_R001_hybrid_IR.wav",
                                                     "S004_R001_hybrid_IR.wav"]
    got_r0 = candidate_references(_query(synthetic_dataset, "Cafe_idx_900/S001_R000_hybrid_IR.wav"))
    assert len(got_r0) == 3


def test_candidate_references_agrees_with_the_dataset_on_every_query(synthetic_dataset):
    """Same candidate sets as ``get_ir_and_location_for_other_sources`` builds internally."""
    for query in synthetic_dataset.file_list:
        dir_name = os.path.dirname(query)
        ir_file_name = os.path.basename(query)
        src_node = int(ir_file_name.split("_")[0][1:])
        rec_n = ir_file_name.split("_")[1]
        all_src_node = set(int(fn.split("_")[0][1:]) for fn in os.listdir(dir_name))
        expected = sorted(
            p for p in (os.path.join(dir_name, "S00{}_{}_hybrid_IR.wav".format(n, rec_n))
                        for n in all_src_node.difference({src_node}))
            if os.path.exists(p))
        assert candidate_references(query) == expected, query


# --------------------------------------------------------------------------------------
# T9b (part 2) -- build_manifest / manifest_hash / save_manifest / load_manifest
# --------------------------------------------------------------------------------------
def test_build_manifest_follows_file_list_and_respects_the_dataset_rules(synthetic_dataset):
    manifest = build_manifest(synthetic_dataset, seed=0, num_shot=8)

    assert set(manifest) == {"seed", "num_shot", "ir_root", "entries"}
    assert manifest["seed"] == 0 and manifest["num_shot"] == 8
    assert manifest["ir_root"] == synthetic_dataset.ir_path
    entries = manifest["entries"]
    assert len(entries) == len(synthetic_dataset.file_list)
    for i, (entry, path) in enumerate(zip(entries, synthetic_dataset.file_list)):
        assert entry["index"] == i
        assert entry["query"] == os.path.relpath(path, synthetic_dataset.ir_path)
        assert not os.path.isabs(entry["query"])
        assert len(entry["refs"]) == 8
        q_src, q_rec = os.path.basename(entry["query"]).split("_")[:2]
        for ref in entry["refs"]:
            assert not os.path.isabs(ref)
            assert os.path.dirname(ref) == os.path.dirname(entry["query"]), "left the room"
            r_src, r_rec = os.path.basename(ref).split("_")[:2]
            assert r_src != q_src, "the query's own source is not a valid reference"
            assert r_rec == q_rec, "references must share the query's receiver"
            assert os.path.exists(os.path.join(manifest["ir_root"], ref))


def test_build_manifest_repeats_references_only_where_candidates_are_short(synthetic_dataset):
    manifest = build_manifest(synthetic_dataset, seed=0, num_shot=8)
    for entry in manifest["entries"]:
        n_candidates = len(candidate_references(
            os.path.join(manifest["ir_root"], entry["query"])))
        distinct = len(set(entry["refs"]))
        if n_candidates >= 8:
            assert distinct == 8, entry["query"]          # without replacement
        else:
            # With replacement: 8 draws from fewer candidates, so repeats are certain
            # but hitting every candidate is not.
            assert distinct < 8 and distinct <= n_candidates, entry["query"]
    # Bedrooms_idx_701 (6 candidates) and Cafe_idx_900 (3 / 2) exercise the short path,
    # Office_idx_777 (9 candidates) the without-replacement one.
    short = [e for e in manifest["entries"] if len(set(e["refs"])) < 8]
    assert len(short) == 7 + 7
    assert len(manifest["entries"]) - len(short) == 20


def test_build_manifest_reads_no_audio(synthetic_dataset, monkeypatch):
    import torchaudio

    def _fail(*a, **kw):
        raise AssertionError("build_manifest must not read audio")

    monkeypatch.setattr(torchaudio, "load", _fail)
    assert len(build_manifest(synthetic_dataset, seed=0)["entries"]) == len(synthetic_dataset)


def test_manifest_hash_is_stable_and_sensitive(synthetic_dataset):
    import copy

    m1 = build_manifest(synthetic_dataset, seed=0, num_shot=8)
    m2 = build_manifest(synthetic_dataset, seed=0, num_shot=8)
    h = manifest_hash(m1)
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
    assert manifest_hash(m2) == h

    # A different manifest seed selects different references, hence a different hash.
    assert manifest_hash(build_manifest(synthetic_dataset, seed=1, num_shot=8)) != h

    altered = copy.deepcopy(m1)
    altered["entries"][3]["refs"][2] = altered["entries"][3]["refs"][2].replace("S00", "S99")
    assert manifest_hash(altered) != h

    # The hash covers the selection only, not where the data happens to be mounted.
    moved = copy.deepcopy(m1)
    moved["ir_root"] = "/somewhere/else/single_channel_ir"
    assert manifest_hash(moved) == h


def test_save_and_load_manifest_round_trip(synthetic_dataset, tmp_path):
    manifest = build_manifest(synthetic_dataset, seed=0, num_shot=8)
    path = str(tmp_path / "sub" / "reference_manifest.json")
    save_manifest(manifest, path)
    loaded = load_manifest(path)
    assert loaded == manifest
    assert manifest_hash(loaded) == manifest_hash(manifest)
