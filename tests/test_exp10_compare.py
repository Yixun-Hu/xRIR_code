"""Tests for ``tools/exp10_compare.py`` (exp_10 ``yaw_pilot``, plan v3.1 tests 10-11).

The comparator is the part of the pilot that is allowed to say "these two runs agree":
it re-derives Metric 1 offline from the stored waveforms, and it aligns the run against
exp_03's historical per-sample file.  Both jobs are only meaningful if the run's identity
holds, so most of these tests are about the refusals.
"""
import json
import os

import pytest

from tools import exp10_compare as compare

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINDING_REPORT = os.path.join(REPO_ROOT, "ckpt", "yaw_rotation", "binding_report.json")


# ----------------------------------------------------- test 11a: verify_exp03_pins

@pytest.mark.skipif(not os.path.exists(BINDING_REPORT),
                    reason="exp_03's binding report is not available")
def test_verify_exp03_pins_passes_on_the_untouched_tree():
    report = compare.verify_exp03_pins(BINDING_REPORT, repo_root=REPO_ROOT)
    assert report["ok"] is True
    assert report["n_files"] == 12
    assert report["reviewed_commit"].startswith("62c9107b")
    assert {entry["path"] for entry in report["files"]} >= {
        "eval_yaw_rotation.py", "tools/yaw_rotation.py", "tools/per_sample_metrics.py",
        "model/xRIR.py", "utils/spec_utils.py"}
    for entry in report["files"]:
        assert entry["match"] is True
        assert entry["live_sha256"] == entry["reviewed_blob_sha256"]


def _pin_fixture(tmp_path, contents):
    """A miniature tree plus the binding report that pins it."""
    root = tmp_path / "tree"
    (root / "tools").mkdir(parents=True)
    import hashlib

    closure = []
    for relative, text in contents.items():
        path = root / relative
        path.write_text(text)
        closure.append({"path": relative,
                        "reviewed_blob_sha256":
                            hashlib.sha256(text.encode()).hexdigest()})
    report = tmp_path / "binding_report.json"
    report.write_text(json.dumps({"reviewed_commit": "0" * 40,
                                  "source_closure": closure}))
    return str(root), str(report)


def test_verify_exp03_pins_refuses_an_altered_pinned_file(tmp_path):
    root, report = _pin_fixture(tmp_path, {"a.py": "original\n",
                                           "tools/b.py": "also original\n"})
    assert compare.verify_exp03_pins(report, repo_root=root)["ok"] is True

    with open(os.path.join(root, "tools", "b.py"), "w") as fout:
        fout.write("tampered\n")
    with pytest.raises(ValueError) as excinfo:
        compare.verify_exp03_pins(report, repo_root=root)
    assert "tools/b.py" in str(excinfo.value)


def test_verify_exp03_pins_refuses_a_missing_pinned_file(tmp_path):
    root, report = _pin_fixture(tmp_path, {"a.py": "original\n",
                                           "tools/b.py": "also original\n"})
    os.remove(os.path.join(root, "a.py"))
    with pytest.raises(ValueError):
        compare.verify_exp03_pins(report, repo_root=root)


# ----------------------------------------------- test 10: the meta guard and check_online

def _run(tmp_path, name="run", **kwargs):
    from tests.exp10_fixture import make_run

    out_dir = str(tmp_path / name)
    make_run(out_dir, **kwargs)
    return out_dir


def test_load_run_accepts_a_self_consistent_run(tmp_path):
    out_dir = _run(tmp_path)
    meta, per_sample = compare.load_run(out_dir)
    assert meta["complete"] is True
    assert per_sample["execution_id"] == meta["execution_id"]
    assert per_sample["protocol_id"] == meta["protocol_id"]
    assert len(per_sample["query"]) == meta["n_queries"]


@pytest.mark.parametrize("field,value", [
    ("gl_seed", 1),
    ("manifest_hash", "z" * 64),
    ("checkpoint_sha256", "z" * 64),
    ("num_shot", 1),
    ("protocol_id", "z" * 64),
    ("execution_id", "another-execution"),
])
def test_load_run_refuses_a_per_sample_file_that_disagrees_with_its_meta(
        tmp_path, field, value):
    out_dir = _run(tmp_path, per_sample_meta_overrides={field: value})
    with pytest.raises(ValueError) as excinfo:
        compare.load_run(out_dir)
    assert field in str(excinfo.value)


def test_load_run_refuses_an_incomplete_run(tmp_path):
    out_dir = _run(tmp_path, complete=False)
    with pytest.raises(ValueError):
        compare.load_run(out_dir)


def test_load_run_refuses_a_waveform_array_that_does_not_match_its_hash(tmp_path):
    import numpy as np

    out_dir = _run(tmp_path)
    path = os.path.join(out_dir, "wav_k128.npy")
    block = np.load(path)
    block[0, 0] += np.float32(0.5)
    np.save(path, block)
    with pytest.raises(ValueError) as excinfo:
        compare.load_run(out_dir)
    assert "wav_k128.npy" in str(excinfo.value)


def test_load_run_refuses_a_reordered_query_list(tmp_path):
    from tests.exp10_fixture import edit_json

    out_dir = _run(tmp_path)

    def swap(payload):
        payload["query"][0], payload["query"][1] = payload["query"][1], payload["query"][0]

    edit_json(os.path.join(out_dir, "per_sample.json"), swap, rehash_meta=True)
    with pytest.raises(ValueError) as excinfo:
        compare.load_run(out_dir)
    assert "query_list_sha256" in str(excinfo.value)


def test_check_online_reproduces_the_waveform_and_acoustic_gaps(tmp_path):
    out_dir = _run(tmp_path)
    report = compare.check_online(out_dir)
    assert report["ok"] is True
    assert sorted(report["angles"]) == ["0", "128"]
    for angle in report["angles"].values():
        assert sorted(angle) == ["c50_gap", "edt_gap", "t60_gap", "wave_mad",
                                 "wave_rel_l2"]
        for cell in angle.values():
            assert cell["ok"] is True
            assert cell["max_abs_diff"] <= 1e-6
            assert cell["validity_mismatches"] == 0
    assert "logspec_mad" not in report["angles"]["128"]
    assert "spectral" in report["excluded_by_contract"]


def test_check_online_flags_a_stored_value_that_the_arrays_do_not_support(tmp_path):
    from tests.exp10_fixture import edit_json

    out_dir = _run(tmp_path)
    edit_json(os.path.join(out_dir, "per_sample.json"),
              lambda payload: payload["angles"]["128"]["wave_mad"].__setitem__(2, 0.5),
              rehash_meta=True)
    report = compare.check_online(out_dir)
    assert report["ok"] is False
    assert report["angles"]["128"]["wave_mad"]["ok"] is False
    assert report["angles"]["128"]["wave_mad"]["max_abs_diff"] > 1e-6
    assert report["angles"]["128"]["wave_rel_l2"]["ok"] is True


def test_check_online_flags_a_validity_mismatch(tmp_path):
    from tests.exp10_fixture import edit_json

    out_dir = _run(tmp_path)
    edit_json(os.path.join(out_dir, "per_sample.json"),
              lambda payload: payload["angles"]["128"]["edt_gap"].__setitem__(1, None),
              rehash_meta=True)
    report = compare.check_online(out_dir)
    assert report["ok"] is False
    assert report["angles"]["128"]["edt_gap"]["validity_mismatches"] == 1


# ------------------------------------------------------------ test 11: parity_exp03

def _probe_like_run(tmp_path, name="probe_run", **kwargs):
    """A run over a non-contiguous subset of a larger canonical population."""
    from tests.exp10_fixture import make_run

    indices = list(range(0, 8)) + list(range(16, 24))
    queries = ["Cat/Room_idx_{}/S{:03d}_R001_hybrid_IR.wav".format(i // 8, i)
               for i in indices]
    out_dir = str(tmp_path / name)
    make_run(out_dir, n=len(indices), queries=queries, indices=indices, batches=(0, 2),
             **kwargs)
    return out_dir


def _history(tmp_path, run_dir, name="per_sample_yaw.json", **kwargs):
    from tests.exp10_fixture import make_exp03_file

    with open(os.path.join(run_dir, "per_sample.json")) as fin:
        per_sample = json.load(fin)
    path = str(tmp_path / name)
    make_exp03_file(path, per_sample, **kwargs)
    return path


def test_parity_exp03_accepts_a_non_contiguous_subset_and_replicates_it(tmp_path):
    run_dir = _probe_like_run(tmp_path)
    history = _history(tmp_path, run_dir)
    report = compare.parity_exp03(run_dir, history)

    assert report["ok"] is True
    assert report["n_rows"] == 16
    assert report["manifest_hash"] == "m" * 64
    for angle in report["angles"].values():
        for metric in ("edt_err", "c50_err", "t60_err", "logspec_mad"):
            cell = angle[metric]
            assert cell["status"] == "replication", (metric, cell)
            assert cell["mask_identical"] is True
            assert cell["max_abs_diff"] == 0.0
            assert cell["sign_agreement"] == 1.0


def test_parity_exp03_refuses_a_different_manifest(tmp_path):
    run_dir = _probe_like_run(tmp_path)
    history = _history(tmp_path, run_dir, manifest_hash="z" * 64)
    with pytest.raises(ValueError) as excinfo:
        compare.parity_exp03(run_dir, history)
    assert "manifest" in str(excinfo.value)


def test_parity_exp03_refuses_a_missing_or_duplicated_historical_query(tmp_path):
    run_dir = _probe_like_run(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        compare.parity_exp03(run_dir, _history(tmp_path, run_dir, name="dropped.json",
                                               drop_index=3))
    assert "missing" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        compare.parity_exp03(run_dir, _history(tmp_path, run_dir, name="dup.json",
                                               duplicate_index=3))
    assert "duplicate" in str(excinfo.value)


def test_parity_exp03_refuses_a_run_that_is_not_in_canonical_order(tmp_path):
    from tests.exp10_fixture import make_run

    indices = [5, 1, 2, 3]
    queries = ["Cat/Room_idx_0/S{:03d}_R001_hybrid_IR.wav".format(i) for i in indices]
    out_dir = str(tmp_path / "shuffled")
    make_run(out_dir, n=4, queries=queries, indices=indices)
    history = _history(tmp_path, out_dir)
    with pytest.raises(ValueError) as excinfo:
        compare.parity_exp03(out_dir, history)
    assert "canonical" in str(excinfo.value)


def test_parity_exp03_refuses_a_historical_row_whose_query_does_not_match(tmp_path):
    from tests.exp10_fixture import edit_json

    run_dir = _probe_like_run(tmp_path)
    history = _history(tmp_path, run_dir)
    edit_json(history, lambda payload: payload["query"].__setitem__(
        2, "Cat/Room_idx_9/S999_R001_hybrid_IR.wav"))
    with pytest.raises(ValueError) as excinfo:
        compare.parity_exp03(run_dir, history)
    assert "query" in str(excinfo.value)


def test_parity_exp03_reports_an_injected_offset_exactly(tmp_path):
    run_dir = _probe_like_run(tmp_path)
    history = _history(tmp_path, run_dir, offsets={(128, "edt_err"): 0.002})
    report = compare.parity_exp03(run_dir, history)

    cell = report["angles"]["128"]["edt_err"]
    assert report["ok"] is False
    assert cell["status"] == "nonreplication (tolerance)"
    assert cell["mask_identical"] is True
    assert cell["max_abs_diff"] == pytest.approx(0.002, abs=1e-12)
    assert cell["mean_abs_diff"] == pytest.approx(0.002, abs=1e-12)
    assert cell["fraction_within_tolerance"] == 0.0
    # exp_03's every row moved by the same constant, so the paired delta moved with it.
    assert cell["paired_delta_exp03"] - cell["paired_delta_run"] == pytest.approx(
        0.002, abs=1e-12)
    assert report["angles"]["128"]["c50_err"]["status"] == "replication"


def test_parity_exp03_flags_a_validity_mismatch_even_when_the_numbers_agree(tmp_path):
    run_dir = _probe_like_run(tmp_path)
    history = _history(tmp_path, run_dir, nan_cells={(128, "edt_err"): [4]})
    report = compare.parity_exp03(run_dir, history)

    cell = report["angles"]["128"]["edt_err"]
    assert report["ok"] is False
    assert cell["status"] == "nonreplication (validity mismatch)"
    assert cell["mask_identical"] is False
    assert cell["validity_mismatches"] == 1
    assert cell["n_common"] == 15
    assert cell["max_abs_diff"] == 0.0      # the surviving rows still agree exactly
    assert report["angles"]["128"]["c50_err"]["status"] == "replication"


# ------------------------------- round 2, finding 1: the per-sample file's own binding

def test_load_run_refuses_a_per_sample_file_whose_recorded_hash_no_longer_holds(tmp_path):
    from tests.exp10_fixture import edit_json

    out_dir = _run(tmp_path)
    edit_json(os.path.join(out_dir, "per_sample.json"),
              lambda payload: payload["angles"]["128"]["logspec_mad"].__setitem__(0, 999.0))
    with pytest.raises(ValueError) as excinfo:
        compare.load_run(out_dir)
    assert "per_sample_sha256" in str(excinfo.value)


def test_load_run_refuses_a_meta_that_never_recorded_the_per_sample_hash(tmp_path):
    from tests.exp10_fixture import edit_json

    out_dir = _run(tmp_path)
    edit_json(os.path.join(out_dir, "meta.json"),
              lambda payload: payload.pop("per_sample_sha256"))
    with pytest.raises(ValueError) as excinfo:
        compare.load_run(out_dir)
    assert "per_sample_sha256" in str(excinfo.value)


def test_check_online_refuses_a_tampered_per_sample_file(tmp_path):
    from tests.exp10_fixture import edit_json

    out_dir = _run(tmp_path)
    edit_json(os.path.join(out_dir, "per_sample.json"),
              lambda payload: payload["angles"]["128"]["logspec_mad"].__setitem__(0, 999.0))
    with pytest.raises(ValueError) as excinfo:
        compare.check_online(out_dir)
    assert "per_sample_sha256" in str(excinfo.value)


def test_parity_exp03_refuses_a_tampered_per_sample_file(tmp_path):
    from tests.exp10_fixture import edit_json

    run_dir = _probe_like_run(tmp_path)
    history = _history(tmp_path, run_dir)
    edit_json(os.path.join(run_dir, "per_sample.json"),
              lambda payload: payload["angles"]["128"]["edt_err"].__setitem__(0, 999.0))
    with pytest.raises(ValueError) as excinfo:
        compare.parity_exp03(run_dir, history)
    assert "per_sample_sha256" in str(excinfo.value)


def _edit_run_meta(out_dir, mutate):
    """Apply one edit to ``meta.json`` *and* to ``per_sample.json``'s embedded copy.

    The two copies must agree (that is its own guard), so a test about a third rule has to
    change both and re-bind the per-sample hash, or it would trip the wrong refusal.
    """
    from tests.exp10_fixture import edit_json
    from tools.exp10_yaw_pilot import file_sha256

    per_sample = os.path.join(out_dir, "per_sample.json")
    edit_json(per_sample, lambda payload: mutate(payload["meta"]))

    def _apply(meta):
        mutate(meta)
        meta["per_sample_sha256"] = file_sha256(per_sample)

    edit_json(os.path.join(out_dir, "meta.json"), _apply)


def test_load_run_refuses_a_waveform_array_whose_binding_was_deleted(tmp_path):
    out_dir = _run(tmp_path)
    _edit_run_meta(out_dir, lambda meta: meta["arrays"].pop("wav_k128.npy"))
    with pytest.raises(ValueError) as excinfo:
        compare.load_run(out_dir)
    message = str(excinfo.value)
    assert "wav_k128.npy" in message and "binding" in message


def test_load_run_refuses_a_binding_whose_recorded_shape_is_wrong(tmp_path):
    out_dir = _run(tmp_path)
    _edit_run_meta(out_dir,
                   lambda meta: meta["arrays"]["wav_k128.npy"].__setitem__("shape",
                                                                           [3, 9600]))
    with pytest.raises(ValueError) as excinfo:
        compare.load_run(out_dir)
    assert "wav_k128.npy" in str(excinfo.value)
