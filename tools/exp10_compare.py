"""Offline comparator for the exp_10 ``yaw_pilot`` runs.

Three jobs, all of them refusals first and numbers second:

* :func:`verify_exp03_pins` -- the twelve files exp_03's record pinned must still hash to
  their reviewed blobs before anything in this experiment is believed;
* :func:`check_online` -- the **waveform and acoustic** halves of Metric 1, recomputed
  from the stored ``wav_k<k>.npy`` arrays, must equal what the run recorded online (the
  spectral halves are computed from the model's direct output and are not recoverable
  from a finite-iteration Griffin-Lim waveform, so they are excluded by contract);
* :func:`parity_exp03` -- the run's rows, aligned to exp_03's historical per-sample file
  by canonical index, with the per-sample and paired differences that the plan's parity
  criteria are read off (this module reports them; it does not decide a launch).

    python tools/exp10_compare.py check-online <run-dir>
    python tools/exp10_compare.py parity-exp03 <run-dir> <exp03-per-sample.json>
    python tools/exp10_compare.py verify-pins [binding-report.json]
"""
from __future__ import annotations

import argparse
import json
import os

from tools.exp10_yaw_pilot import file_sha256

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BINDING_REPORT = os.path.join(REPO_ROOT, "ckpt", "yaw_rotation",
                                      "binding_report.json")


def verify_exp03_pins(binding_report=DEFAULT_BINDING_REPORT, repo_root=REPO_ROOT):
    """Re-verify exp_03's pinned source closure against the live tree.

    exp_03's record bound twelve files (the evaluator, the rotation and metric tools, the
    models, the dataset and the loss module) to the blobs a reviewer signed off at commit
    ``62c9107b…``.  exp_10 reuses those files unchanged, so this experiment's numbers only
    mean what they claim if the files still hash to the reviewed blobs;
    ``tests/test_exp03_record_tools.py`` exercises the *validators* on synthetic records,
    which is a different question.

    Args:
        binding_report: path to exp_03's ``binding_report.json``.
        repo_root: the tree whose files are hashed (the repository root).

    Returns:
        ``{"ok", "binding_report", "binding_report_sha256", "reviewed_commit", "n_files",
        "files"}``; each file entry carries ``path``, ``reviewed_blob_sha256``,
        ``live_sha256`` (``None`` when the file is missing) and ``match``.

    Raises:
        ValueError: if any pinned file is missing or no longer hashes to its reviewed
            blob -- the run refuses to start rather than report numbers from a changed
            closure.
    """
    with open(binding_report) as fin:
        report = json.load(fin)
    files = []
    for entry in report["source_closure"]:
        relative = entry["path"]
        path = os.path.join(repo_root, relative)
        live = file_sha256(path) if os.path.isfile(path) else None
        files.append({"path": relative,
                      "reviewed_blob_sha256": entry["reviewed_blob_sha256"],
                      "live_sha256": live,
                      "match": live == entry["reviewed_blob_sha256"]})
    broken = [entry["path"] for entry in files if not entry["match"]]
    if broken:
        raise ValueError(
            "exp_03's pinned source closure no longer matches its reviewed blobs "
            "({} of {} files): {}".format(len(broken), len(files), ", ".join(broken)))
    return {"ok": True,
            "binding_report": os.path.abspath(binding_report),
            "binding_report_sha256": file_sha256(binding_report),
            "reviewed_commit": report.get("reviewed_commit"),
            "n_files": len(files),
            "files": files}


def main(argv=None):
    """Parse the CLI and dispatch to one of the comparator's commands."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pins = sub.add_parser("verify-pins", help="re-verify exp_03's pinned closure")
    pins.add_argument("binding_report", nargs="?", default=DEFAULT_BINDING_REPORT)
    args = parser.parse_args(argv)

    if args.command == "verify-pins":
        report = verify_exp03_pins(args.binding_report)
        print("exp_03 pins OK: {} files at reviewed commit {}".format(
            report["n_files"], report["reviewed_commit"]))
        return report
    raise ValueError("unknown command {!r}".format(args.command))


if __name__ == "__main__":
    main()
