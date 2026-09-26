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

    edit_json(os.path.join(out_dir, "per_sample.json"), swap)
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
              lambda payload: payload["angles"]["128"]["wave_mad"].__setitem__(2, 0.5))
    report = compare.check_online(out_dir)
    assert report["ok"] is False
    assert report["angles"]["128"]["wave_mad"]["ok"] is False
    assert report["angles"]["128"]["wave_mad"]["max_abs_diff"] > 1e-6
    assert report["angles"]["128"]["wave_rel_l2"]["ok"] is True


def test_check_online_flags_a_validity_mismatch(tmp_path):
    from tests.exp10_fixture import edit_json

    out_dir = _run(tmp_path)
    edit_json(os.path.join(out_dir, "per_sample.json"),
              lambda payload: payload["angles"]["128"]["edt_gap"].__setitem__(1, None))
    report = compare.check_online(out_dir)
    assert report["ok"] is False
    assert report["angles"]["128"]["edt_gap"]["validity_mismatches"] == 1
