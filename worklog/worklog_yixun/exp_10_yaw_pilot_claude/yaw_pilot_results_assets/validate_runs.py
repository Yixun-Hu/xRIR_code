#!/usr/bin/env python3
"""Validate — and bind — the exp_10 run evidence the record is built from.

The finish script calls this first: it is the preflight that decides whether the four GPU
arms, their probes and the CPU-protocol record are the evidence the plan prescribes, and
whether every validation report on the table actually *belongs* to the run it is filed
under.  Codex's tooling review reached ``FINISH DONE`` with missing full parity reports,
failing probe controls and online reports naming another run; nothing here is satisfied by
the mere presence of a file.

What is required per arm (``--root``, default ``ckpt/exp10``):

* ``<arm>_all/meta.json``  — ``complete``, ``n_queries == --expect-n``, ``device == cuda``,
  a non-empty ``execution_id`` / ``protocol_id``, ``batches_arg == "all"``, its
  ``per_sample.json`` hashing to ``meta.per_sample_sha256`` and every angle in ``meta.ks``
  carrying a waveform array that still matches its ``meta.arrays`` binding.
* ``<arm>_all/check_online.json`` — ``ok is True`` **and** bound to that run: its
  ``run_dir`` has to resolve to the run directory, and any identity field it carries
  (``execution_id`` / ``protocol_id`` / ``per_sample_sha256``) has to match the meta.  The
  comparator's report carries none of those today, so the binding is ``run_dir`` plus the
  run's own re-verified ``per_sample_sha256`` — which is why the per-sample hash is checked
  before any report is believed.
* ``<arm>_probe/meta.json`` (``batches_arg == "probe"``) and
  ``<arm>_probe/summary/yaw_pilot_summary.json`` — exactly that arm, the probe's
  ``execution_id``, ``controls.ok is True``, and the summary's own ``inputs`` bindings.
* ``released_k8`` / ``control_k8`` / ``cyl_k8``: **both** ``<arm>_probe/parity_exp03.json``
  and ``<arm>_all/parity_exp03.json`` with ``ok is True`` and their ``run_dir`` bound.
  ``released_k1`` has no exp_03 predecessor, so parity is not required — but a parity
  report found there is still validated rather than ignored.
* The four arms must be four *executions*: duplicated ``execution_id`` or ``protocol_id``
  is refused.

The CPU-protocol record (``--cpu-run``, default ``<root>/cpu_protocol/released_k8_all``) is
validated separately: complete, ``device == cpu``, ``check_online`` ok, and a parity report
that must be *present* and is recorded with whatever ``ok`` it has — exp_10's CPU run fails
criterion (a) by design (plan §7) and that failure is part of the record.

Waveform arrays are checked by size against their recorded shape and dtype by default
(``--array-check size``); ``full`` re-hashes them (``none`` skips them).  The canonical
summariser re-hashes every array through ``tools/exp10_compare.load_run`` when it builds
the summary a moment later, so the cheap check here is a preflight, not the only binding.

usage: validate_runs.py [--root ckpt/exp10] [--arms released_k8 ...] [--expect-n 6337]
                        [--device cuda] [--cpu-run <dir>|--no-cpu]
                        [--cpu-recorded-run-dir <pre-move path>]
                        [--array-check size|full|none] [--json <report.json>]
exit status: 0 = every requirement met, 2 = at least one failure (all of them are printed).
"""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import sys

ARMS = ("released_k8", "released_k1", "control_k8", "cyl_k8")
PARITY_ARMS = ("released_k8", "control_k8", "cyl_k8")
EXPECTED_N_QUERIES = 6337
DTYPE_BYTES = {"float16": 2, "float32": 4, "float64": 8, "int16": 2, "int32": 4, "int64": 8}
NPY_HEADER_SLACK = 4096          # a .npy header is 128 bytes today; never a whole block
IDENTITY_FIELDS = ("execution_id", "protocol_id", "per_sample_sha256")


class ValidationError(Exception):
    """A record input that cannot be used: the message lists every failed requirement."""


def file_sha256(path, chunk=1 << 20):
    """sha256 of a file's bytes — the digest every exp_10 tool records."""
    digest = hashlib.sha256()
    with open(path, "rb") as fin:
        for block in iter(lambda: fin.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path) as fin:
        return json.load(fin)


def same_path(left, right):
    """True if two paths name the same directory or file after resolving symlinks."""
    return os.path.realpath(str(left)) == os.path.realpath(str(right))


def assert_ok(problems, label):
    """Raise ``ValidationError`` listing ``problems`` (a no-op when there are none)."""
    if problems:
        raise ValidationError("%s:\n  - %s" % (label, "\n  - ".join(problems)))
    return True


def _expected_array_bytes(entry):
    shape = [int(x) for x in entry.get("shape") or []]
    item = DTYPE_BYTES.get(entry.get("dtype"))
    if not shape or item is None:
        return None
    count = 1
    for dim in shape:
        count *= dim
    return count * item


def check_arrays(run_dir, meta, array_check="size", label=None):
    """The waveform arrays every angle in ``meta.ks`` must still bind to."""
    label = label or run_dir
    problems = []
    if array_check == "none":
        return problems
    arrays = meta.get("arrays") or {}
    for k in meta.get("ks") or []:
        name = "wav_k%d.npy" % int(k)
        entry = arrays.get(name)
        if not entry:
            problems.append("%s: meta.arrays has no binding for %s (angle %s of meta.ks)"
                            % (label, name, k))
            continue
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            problems.append("%s: the waveform array %s is missing" % (label, name))
            continue
        shape = [int(x) for x in entry.get("shape") or []]
        if shape[:1] != [int(meta.get("n_queries", -1))]:
            problems.append("%s: %s has shape %s for %s queries"
                            % (label, name, entry.get("shape"), meta.get("n_queries")))
        expected = _expected_array_bytes(entry)
        size = os.path.getsize(path)
        if expected is None:
            problems.append("%s: %s has no usable shape/dtype binding (%r / %r)"
                            % (label, name, entry.get("shape"), entry.get("dtype")))
        elif not expected <= size <= expected + NPY_HEADER_SLACK:
            problems.append("%s: %s is %d bytes, not the %d its recorded shape %s of %s "
                            "needs" % (label, name, size, expected, entry.get("shape"),
                                       entry.get("dtype")))
        if array_check == "full" and os.path.isfile(path):
            live = file_sha256(path)
            if live != entry.get("sha256"):
                problems.append("%s: %s hashes to %s, not the recorded sha256 %s"
                                % (label, name, live, entry.get("sha256")))
    return problems


def check_run(run_dir, arm=None, expect_device=None, expect_n=None,
              expect_batches_arg=None, array_check="size", label=None):
    """Validate one run directory and return ``(meta, problems, binding)``.

    Args:
        run_dir: the evaluator's ``--out-dir``.
        arm: the arm label the run has to claim.
        expect_device: ``"cuda"`` or ``"cpu"``.
        expect_n: required ``meta.n_queries``.
        expect_batches_arg: ``"all"`` for a full run, ``"probe"`` for a probe.
        array_check: ``size`` (default), ``full`` or ``none``.
        label: how to name this run in the messages (default: the directory).

    Returns:
        ``(meta or None, problems, binding)``; ``binding`` carries the identities that were
        verified, so the caller can publish what it bound.
    """
    label = label or run_dir
    problems, binding = [], {"run_dir": os.path.abspath(run_dir)}
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return None, ["%s: meta.json is missing (%s)" % (label, meta_path)], binding
    try:
        meta = read_json(meta_path)
    except ValueError as exc:
        return None, ["%s: meta.json is not readable JSON (%s)" % (label, exc)], binding
    binding["meta_sha256"] = file_sha256(meta_path)
    for field in IDENTITY_FIELDS:
        binding[field] = meta.get(field)
    binding["device"] = meta.get("device")
    binding["n_queries"] = meta.get("n_queries")
    binding["batches_arg"] = meta.get("batches_arg")
    binding["arm"] = meta.get("arm")

    if arm is not None and meta.get("arm") != arm:
        problems.append("%s: meta.arm is %r, not %r" % (label, meta.get("arm"), arm))
    if meta.get("complete") is not True:
        problems.append("%s: meta.complete is %r; the run did not finish"
                        % (label, meta.get("complete")))
    if expect_n is not None and int(meta.get("n_queries", -1)) != int(expect_n):
        problems.append("%s: meta.n_queries is %r, not the %s the protocol prescribes"
                        % (label, meta.get("n_queries"), expect_n))
    if expect_device is not None and meta.get("device") != expect_device:
        problems.append("%s: meta.device is %r, not %r" % (label, meta.get("device"),
                                                           expect_device))
    if expect_batches_arg is not None and meta.get("batches_arg") != expect_batches_arg:
        problems.append("%s: meta.batches_arg is %r, not %r"
                        % (label, meta.get("batches_arg"), expect_batches_arg))
    for field in ("execution_id", "protocol_id", "query_list_sha256", "per_sample_sha256"):
        if not meta.get(field):
            problems.append("%s: meta.%s is %r; the run is unbound"
                            % (label, field, meta.get(field)))

    per_sample = os.path.join(run_dir, "per_sample.json")
    if not os.path.isfile(per_sample):
        problems.append("%s: per_sample.json is missing" % label)
    elif meta.get("per_sample_sha256"):
        live = file_sha256(per_sample)
        if live != meta["per_sample_sha256"]:
            problems.append("%s: per_sample.json hashes to %s but meta.per_sample_sha256 "
                            "is %s; the file is not the one the run recorded"
                            % (label, live, meta["per_sample_sha256"]))
    problems.extend(check_arrays(run_dir, meta, array_check=array_check, label=label))
    return meta, problems, binding


def _declared_relocation(recorded, run_dir, also_accept):
    """True if ``recorded`` is a declared former location of this very run directory."""
    if not also_accept:
        return False
    name = os.path.basename(os.path.abspath(str(run_dir)).rstrip("/"))
    if os.path.basename(str(recorded).rstrip("/")) != name:
        return False
    candidates = [also_accept] if isinstance(also_accept, str) else list(also_accept)
    return any(same_path(recorded, c) or
               os.path.abspath(str(recorded)) == os.path.abspath(str(c))
               for c in candidates)


def bind_report(path, run_dir, meta, kind, require_ok=True, label=None, also_accept=None):
    """Validate a per-run validation report (``check_online.json`` / ``parity_exp03.json``).

    The report must exist, be readable, name *this* run in ``run_dir``, agree with the meta
    on every identity field it happens to carry, and — unless ``require_ok`` is False —
    report ``ok is True``.  ``ok`` is compared to the Boolean: the comparator writes
    ``false`` and exits 0, which is exactly how a failed check slipped through.

    ``also_accept`` declares paths the report may name *instead* of ``run_dir`` because the
    directory was moved after the report was written (exp_10's CPU record was relocated to
    ``cpu_protocol/`` when the headline arm was re-run on the GPU, and its two reports still
    name the pre-move path).  A declared alternative is accepted only when its last path
    component is the run's own — so it cannot silently admit another arm's report — and the
    relocation is recorded in the binding rather than hidden.
    """
    label = label or "%s %s" % (os.path.basename(str(run_dir).rstrip("/")), kind)
    problems, binding = [], {"path": os.path.abspath(path), "kind": kind, "bound_by": []}
    if not os.path.isfile(path):
        return None, ["%s: %s is missing (%s)" % (label, os.path.basename(path), path)], binding
    try:
        report = read_json(path)
    except ValueError as exc:
        return None, ["%s: %s is not readable JSON (%s)"
                      % (label, os.path.basename(path), exc)], binding
    binding["sha256"] = file_sha256(path)
    binding["ok"] = report.get("ok")
    if report.get("run_dir") is None:
        problems.append("%s: the report has no run_dir; it cannot be bound to a run" % label)
    elif same_path(report["run_dir"], run_dir):
        binding["bound_by"].append("run_dir")
    elif _declared_relocation(report["run_dir"], run_dir, also_accept):
        binding["bound_by"].append("run_dir (declared relocation)")
        binding["relocated_from"] = report["run_dir"]
    else:
        problems.append("%s: the report's run_dir %r is not this run (%s)"
                        % (label, report.get("run_dir"), os.path.abspath(run_dir)))
    for field in IDENTITY_FIELDS:
        if field in report:
            if report[field] != (meta or {}).get(field):
                problems.append("%s: the report's %s is %r, not the run's %r"
                                % (label, field, report[field], (meta or {}).get(field)))
            else:
                binding["bound_by"].append(field)
    if meta and meta.get("per_sample_sha256"):
        # No identity field in today's reports: the run's own re-verified per-sample hash
        # (checked in check_run) is what the report's run_dir is bound through.
        binding["per_sample_sha256"] = meta["per_sample_sha256"]
        binding["bound_by"].append("meta.per_sample_sha256")
    if require_ok and report.get("ok") is not True:
        problems.append("%s: the report's ok is %r, not true" % (label, report.get("ok")))
    if kind == "parity_exp03" and meta:
        if report.get("manifest_hash") and report["manifest_hash"] != meta.get("manifest_hash"):
            problems.append("%s: the report's manifest_hash %r is not the run's %r"
                            % (label, report.get("manifest_hash"), meta.get("manifest_hash")))
        if report.get("n_rows") is not None and meta.get("n_queries") is not None:
            if int(report["n_rows"]) != int(meta["n_queries"]):
                problems.append("%s: the report compares %s rows, the run has %s queries"
                                % (label, report.get("n_rows"), meta.get("n_queries")))
        binding["exp03_path"] = report.get("exp03_path")
        binding["exp03_sha256"] = report.get("exp03_sha256")
        binding["n_rows"] = report.get("n_rows")
    if kind == "check_online" and meta:
        angles = sorted(int(k) for k in (report.get("angles") or {}))
        if angles and angles != sorted(int(k) for k in (meta.get("ks") or [])):
            problems.append("%s: the report covers angles %s, the run's ks are %s"
                            % (label, angles, meta.get("ks")))
    return report, problems, binding


def verify_summary_inputs(summary, arms=None, label="summary"):
    """Re-verify a summariser JSON's ``inputs`` against the live files it names.

    Both bindings the summariser itself records are checked: the per-sample and meta files
    must still hash to what the summary was built from, and each run's meta must still bind
    its own per-sample file.  A summary that fails this is not evidence for its numbers.
    """
    problems = []
    entries = summary.get("inputs")
    if not entries:
        return ["%s: has no inputs block; its numbers are unbound" % label]
    for entry in entries:
        run_dir = entry.get("run_dir")
        if arms is not None and entry.get("arm") not in arms:
            problems.append("%s: input arm %r is not one of %s"
                            % (label, entry.get("arm"), ", ".join(sorted(arms))))
        for name, key in (("per_sample.json", "per_sample_sha256"), ("meta.json", "meta_sha256")):
            path = os.path.join(str(run_dir), name)
            if not os.path.isfile(path):
                problems.append("%s: %s is missing; the summary cannot be verified" % (label, path))
                continue
            live = file_sha256(path)
            if live != entry.get(key):
                problems.append("%s: %s hashes to %s, not the %s the summary was built from"
                                % (label, path, live, entry.get(key)))
        meta_path = os.path.join(str(run_dir), "meta.json")
        if os.path.isfile(meta_path):
            meta = read_json(meta_path)
            if meta.get("per_sample_sha256") != entry.get("per_sample_sha256"):
                problems.append("%s: %s no longer binds its per-sample file (meta %r vs "
                                "summary %r)" % (label, run_dir, meta.get("per_sample_sha256"),
                                                 entry.get("per_sample_sha256")))
            for field in ("execution_id", "protocol_id"):
                if meta.get(field) != entry.get(field):
                    problems.append("%s: %s has %s %r, the summary recorded %r"
                                    % (label, run_dir, field, meta.get(field), entry.get(field)))
    return problems


def check_probe_summary(path, arm, probe_meta, probe_dir=None, require_controls_ok=True):
    """Validate a probe run's own summary: exactly that arm, that execution, controls ok."""
    label = "%s probe summary" % arm
    problems, binding = [], {"path": os.path.abspath(path)}
    if not os.path.isfile(path):
        return None, ["%s: %s is missing" % (label, path)], binding
    try:
        summary = read_json(path)
    except ValueError as exc:
        return None, ["%s: not readable JSON (%s)" % (label, exc)], binding
    binding["sha256"] = file_sha256(path)
    arms = summary.get("arms") or []
    if len(arms) != 1 or arms[0].get("arm") != arm:
        problems.append("%s: the summary covers %r, not exactly the one arm %r"
                        % (label, [a.get("arm") for a in arms], arm))
        return summary, problems, binding
    entry = arms[0]
    binding["execution_id"] = entry.get("execution_id")
    binding["n_queries"] = entry.get("n_queries")
    binding["controls_ok"] = (entry.get("controls") or {}).get("ok")
    if probe_meta and entry.get("execution_id") != probe_meta.get("execution_id"):
        problems.append("%s: the summary's execution %r is not the probe run's %r"
                        % (label, entry.get("execution_id"), probe_meta.get("execution_id")))
    if (entry.get("meta") or {}).get("batches_arg") != "probe":
        problems.append("%s: the summarised run's batches_arg is %r, not 'probe'"
                        % (label, (entry.get("meta") or {}).get("batches_arg")))
    if probe_dir is not None and not same_path(entry.get("run_dir") or "", probe_dir):
        # run_dir is recorded relative to the repository root, so resolve both ways.
        inputs = [i.get("run_dir") for i in summary.get("inputs") or []]
        if not any(same_path(p or "", probe_dir) for p in inputs):
            problems.append("%s: the summary's run %r is not the probe directory %s"
                            % (label, entry.get("run_dir"), os.path.abspath(probe_dir)))
    if require_controls_ok and binding["controls_ok"] is not True:
        problems.append("%s: controls.ok is %r, not true" % (label, binding["controls_ok"]))
    problems.extend(verify_summary_inputs(summary, label=label))
    return summary, problems, binding


def link_href(path, assets_dir, assets_href, out_path):
    """The relative href both renderers use for a copied asset (finding 12).

    A file inside the assets directory is linked through ``assets_href`` (the *published*
    location of that directory, which the finish script stages elsewhere and renames into
    place); anything else is linked relative to the page itself.
    """
    target = os.path.abspath(str(path))
    assets = os.path.abspath(str(assets_dir)) if assets_dir else None
    if assets and (target == assets or target.startswith(assets + os.sep)):
        inside = os.path.relpath(target, assets).replace(os.sep, "/")
        return "%s/%s" % (str(assets_href).rstrip("/"), inside) if inside != "." else str(assets_href)
    return os.path.relpath(target, os.path.dirname(os.path.abspath(str(out_path)))).replace(os.sep, "/")


def parse_pairs(items, flag):
    """``["<arm>=<path>", ...]`` → ``[(arm, path), ...]``, refusing a malformed entry."""
    pairs, problems = [], []
    for item in items or []:
        if "=" not in item:
            problems.append("%s %r is not <arm>=<path>" % (flag, item))
            continue
        arm, path = item.split("=", 1)
        if not arm or not path:
            problems.append("%s %r is not <arm>=<path>" % (flag, item))
            continue
        pairs.append((arm, path))
    assert_ok(problems, "bad %s argument" % flag)
    return pairs


def check_supplements(summary, parity=(), online=(), probes=(), stage="_all"):
    """Check that every supplemental report belongs to the arm it is attached to.

    A report is accepted for ``arm`` only when the arm is in this summary, the report's
    ``run_dir`` resolves to the very run directory the summary was built from for that arm,
    and its last path component is that arm's stage directory.  A probe summary must cover
    exactly that arm, have ``batches_arg == "probe"`` and bind its own inputs.  (Codex's
    review attached the cylindrical parity report to the control arm and it was rendered.)
    """
    problems = []
    arms = [a.get("arm") for a in summary.get("arms") or []]
    run_dirs = {i.get("arm"): i.get("run_dir") for i in summary.get("inputs") or []}
    for kind, items in (("parity_exp03", parity), ("check_online", online)):
        for arm, path in items or []:
            label = "%s %s (%s)" % (arm, kind, path)
            if arm not in arms:
                problems.append("%s: %r is not an arm of this summary (%s)"
                                % (label, arm, ", ".join(str(a) for a in arms)))
                continue
            if not os.path.isfile(path):
                problems.append("%s: the report is missing" % label)
                continue
            try:
                report = read_json(path)
            except ValueError as exc:
                problems.append("%s: not readable JSON (%s)" % (label, exc))
                continue
            recorded = report.get("run_dir")
            if recorded is None:
                problems.append("%s: the report has no run_dir; it cannot be bound to an arm"
                                % label)
                continue
            expected_name = "%s%s" % (arm, stage)
            if os.path.basename(str(recorded).rstrip("/")) != expected_name:
                problems.append("%s: the report was written for %r, not this arm's %s"
                                % (label, recorded, expected_name))
            if arm in run_dirs and not same_path(recorded, run_dirs[arm]):
                problems.append("%s: the report's run %r is not the run this summary "
                                "summarised for %s (%s)"
                                % (label, recorded, arm, run_dirs[arm]))
    for arm, path in probes or []:
        label = "%s probe summary (%s)" % (arm, path)
        if arm not in arms:
            problems.append("%s: %r is not an arm of this summary (%s)"
                            % (label, arm, ", ".join(str(a) for a in arms)))
            continue
        _summary, probs, _binding = check_probe_summary(path, arm, None,
                                                       require_controls_ok=False)
        problems.extend(probs)
    return problems


def figure_names(summary):
    """The figure basenames a summariser output directory may contribute to the page.

    Only the combined figure and one per arm *in this summary*: a figure named for another
    arm in the same directory belongs to another run and is not copied (finding 4).
    """
    names = ["yaw_pilot_gaps_all_arms.png", "yaw_pilot_gaps_all_arms.pdf"]
    for arm in summary.get("arms") or []:
        names.append("yaw_pilot_gaps_%s.png" % arm.get("arm"))
        names.append("yaw_pilot_gaps_%s.pdf" % arm.get("arm"))
    return names


def validate_arm(root, arm, expect_n=EXPECTED_N_QUERIES, device="cuda",
                 require_parity=True, array_check="size"):
    """Validate one arm's full run, probe, reports and probe summary."""
    entry = {"arm": arm, "problems": [], "full": None, "probe": None,
             "check_online": None, "parity_all": None, "parity_probe": None,
             "probe_summary": None}
    full_dir = os.path.join(root, "%s_all" % arm)
    probe_dir = os.path.join(root, "%s_probe" % arm)
    full_meta, problems, binding = check_run(
        full_dir, arm=arm, expect_device=device, expect_n=expect_n,
        expect_batches_arg="all", array_check=array_check,
        label="%s_all" % arm)
    entry["full"] = binding
    entry["problems"].extend(problems)

    report, problems, binding = bind_report(
        os.path.join(full_dir, "check_online.json"), full_dir, full_meta, "check_online",
        require_ok=True, label="%s_all check_online" % arm)
    entry["check_online"] = binding
    entry["problems"].extend(problems)

    probe_meta, problems, binding = check_run(
        probe_dir, arm=arm, expect_device=device, expect_batches_arg="probe",
        array_check=array_check, label="%s_probe" % arm)
    entry["probe"] = binding
    entry["problems"].extend(problems)

    report, problems, binding = bind_report(
        os.path.join(probe_dir, "check_online.json"), probe_dir, probe_meta, "check_online",
        require_ok=True, label="%s_probe check_online" % arm)
    entry["probe_check_online"] = binding
    entry["problems"].extend(problems)

    summary, problems, binding = check_probe_summary(
        os.path.join(probe_dir, "summary", "yaw_pilot_summary.json"), arm, probe_meta,
        probe_dir=probe_dir)
    entry["probe_summary"] = binding
    entry["problems"].extend(problems)

    for key, stage_dir, stage_meta, label in (
            ("parity_all", full_dir, full_meta, "%s_all parity_exp03" % arm),
            ("parity_probe", probe_dir, probe_meta, "%s_probe parity_exp03" % arm)):
        path = os.path.join(stage_dir, "parity_exp03.json")
        if not require_parity and not os.path.isfile(path):
            entry[key] = None            # released_k1: no exp_03 predecessor to compare to
            continue
        report, problems, binding = bind_report(path, stage_dir, stage_meta, "parity_exp03",
                                                require_ok=True, label=label)
        entry[key] = binding
        entry["problems"].extend(problems)
    return entry


def validate_cpu_record(run_dir, expect_n=EXPECTED_N_QUERIES, arm="released_k8",
                        array_check="size", recorded_run_dir=None):
    """Validate the CPU-protocol record, preserving its documented parity failure.

    ``recorded_run_dir`` declares the path this run's reports were written under, for the
    documented relocation into ``cpu_protocol/`` (see ``bind_report``).
    """
    entry = {"arm": arm, "run_dir": os.path.abspath(run_dir), "problems": [],
             "recorded_run_dir": recorded_run_dir}
    meta, problems, binding = check_run(run_dir, arm=arm, expect_device="cpu",
                                        expect_n=expect_n, expect_batches_arg="all",
                                        array_check=array_check,
                                        label="cpu_protocol/%s_all" % arm)
    entry["full"] = binding
    entry["problems"].extend(problems)
    report, problems, binding = bind_report(
        os.path.join(run_dir, "check_online.json"), run_dir, meta, "check_online",
        require_ok=True, label="cpu_protocol/%s_all check_online" % arm,
        also_accept=recorded_run_dir)
    entry["check_online"] = binding
    entry["problems"].extend(problems)
    # Parity must be *present* — the plan's CPU-protocol decision rests on it — but its
    # ok is recorded, not required: exp_10's CPU run fails criterion (a) by design.
    report, problems, binding = bind_report(
        os.path.join(run_dir, "parity_exp03.json"), run_dir, meta, "parity_exp03",
        require_ok=False, label="cpu_protocol/%s_all parity_exp03" % arm,
        also_accept=recorded_run_dir)
    entry["parity_all"] = binding
    entry["problems"].extend(problems)
    return entry


def validate_record(root, arms=ARMS, expect_n=EXPECTED_N_QUERIES, device="cuda",
                    cpu_run=None, array_check="size", parity_arms=PARITY_ARMS,
                    cpu_recorded_run_dir=None):
    """Validate the whole evidence tree; returns a report whose ``ok`` decides the finish.

    Args:
        root: the directory holding ``<arm>_all`` / ``<arm>_probe`` (``ckpt/exp10``).
        arms: the arms the record prescribes.
        expect_n: the full runs' query count.
        device: the device the GPU arms must have run on.
        cpu_run: the CPU-protocol run directory; ``None`` → ``<root>/cpu_protocol/released_k8_all``;
            ``False`` → do not validate a CPU record.
        cpu_recorded_run_dir: the path the CPU record's reports name, if the directory was
            moved after they were written (a declared relocation, recorded in the report).
        array_check: ``size`` (default), ``full`` or ``none``.
        parity_arms: the arms with an exp_03 predecessor (parity required).

    Returns:
        ``{"ok", "root", "arms", "cpu", "problems"}``.
    """
    report = {"ok": False, "root": os.path.abspath(root), "expect_n": int(expect_n),
              "device": device, "array_check": array_check,
              "arms": {}, "cpu": None, "problems": []}
    for arm in arms:
        entry = validate_arm(root, arm, expect_n=expect_n, device=device,
                             require_parity=arm in parity_arms, array_check=array_check)
        report["arms"][arm] = entry
        report["problems"].extend(entry["problems"])
    for field in ("execution_id", "protocol_id"):
        seen = {}
        for arm, entry in sorted(report["arms"].items()):
            value = (entry.get("full") or {}).get(field)
            if not value:
                continue
            if value in seen:
                report["problems"].append(
                    "%s and %s share one %s (%s): that is one execution presented twice, "
                    "not two arms" % (seen[value], arm, field, value))
            seen[value] = arm
    if cpu_run is not False:
        cpu_run = cpu_run or os.path.join(root, "cpu_protocol", "released_k8_all")
        entry = validate_cpu_record(cpu_run, expect_n=expect_n, array_check=array_check,
                                    recorded_run_dir=cpu_recorded_run_dir)
        report["cpu"] = entry
        report["problems"].extend(entry["problems"])
    report["ok"] = not report["problems"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="ckpt/exp10")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--expect-n", type=int, default=EXPECTED_N_QUERIES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-run", default=None)
    parser.add_argument("--cpu-recorded-run-dir", default=None,
                        help="the run_dir the CPU record's reports name, if its directory "
                             "was moved after they were written (declared relocation)")
    parser.add_argument("--no-cpu", action="store_true",
                        help="do not validate a CPU-protocol record")
    parser.add_argument("--array-check", choices=("none", "size", "full"), default="size")
    parser.add_argument("--json", default=None, help="write the validation report here")
    args = parser.parse_args(argv)
    report = validate_record(args.root, arms=tuple(args.arms), expect_n=args.expect_n,
                             device=args.device,
                             cpu_run=False if args.no_cpu else args.cpu_run,
                             array_check=args.array_check,
                             cpu_recorded_run_dir=args.cpu_recorded_run_dir)
    if args.json:
        parent = os.path.dirname(os.path.abspath(args.json))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        with open(args.json, "w") as fout:
            json.dump(report, fout, indent=2, sort_keys=True)
    for problem in report["problems"]:
        print("REFUSED: %s" % problem)
    if report["ok"]:
        print("validate_runs: %d arms + %s validated and bound under %s"
              % (len(args.arms), "the CPU record" if report["cpu"] else "no CPU record",
                 report["root"]))
        return 0
    print("validate_runs: %d requirement(s) failed" % len(report["problems"]))
    return 2


if __name__ == "__main__":
    sys.exit(main())
