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
