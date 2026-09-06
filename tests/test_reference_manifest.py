"""Tests for :mod:`tools.reference_manifest` (exp_03, yaw_rotation_degradation).

The manifest pins, once and for all, which 8 reference RIRs every query of the
AcousticRooms unseen test split is evaluated with, so that k=0 and k!=0 -- and every
model -- see exactly the same conditioning set.  ``xRIR_Dataset`` itself draws the
references with a worker-local ``np.random`` call that is never recorded, which is why
the draw is re-implemented here as a pure function of (seed, query path).
"""
from tools.reference_manifest import select_references


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
