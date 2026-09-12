"""Launch exp_04 evaluation and bind completion only after the child and log close."""
import datetime
import os
import subprocess
import uuid
from pathlib import Path

from tools import provenance as p

OUTPUTS = ("per_sample_yaw.json", "metrics_yaw.json")


def _run_child(command, log_path, repo, data_root):
    """Run a literal argv through tee; both processes finish before the log closes."""
    env = dict(os.environ, XRIR_DATA_PATH=str(data_root), CUDA_VISIBLE_DEVICES="1")
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    child = sink = None
    with log_path.open("xb") as log:
        try:
            child = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
            sink = subprocess.Popen(["tee", "/dev/stderr"], stdin=child.stdout, stdout=log)
            child.stdout.close()
            status = child.wait()
            if sink.wait():
                raise RuntimeError("log_sink")
            log.flush()
            os.fsync(log.fileno())
            return status
        finally:
            for process in (child, sink):
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait()


def execute_run(args, command, fields_factory, repo):
    """Own exclusive creation, spawn, digest revalidation and atomic finalisation."""
    run = Path(args.out_dir).resolve()
    run.mkdir()  # Deliberately before input hashing and outside abort handling.
    reason = "setup_failed"
    try:
        log_dir = Path(args.log_dir).resolve()
        if run == log_dir or run in log_dir.parents:
            raise ValueError("log-dir must be outside out-dir")
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d_%H:%M:%S.%f")
        label = args.run_label
        if not label or Path(label).name != label or label in (".", ".."):
            raise ValueError("run-label must be a filename component")
        log_path = log_dir / (label + "_" + stamp + ".log")
        fields = fields_factory()
        manifest_path = run / "eval_manifest.json"
        digest = p.write_manifest(manifest_path, fields)
        reason = "child_failed"
        if _run_child(command, log_path, repo, args.data_root):
            raise RuntimeError(reason)
        reason = "missing_output"
        if any(not (run / name).is_file() for name in OUTPUTS):
            raise RuntimeError(reason)
        reason = "input_changed"
        declared = dict(fields, eval_manifest={"path": str(manifest_path), "sha256": digest})
        mismatches = p.revalidate(declared)
        if mismatches:
            raise ValueError("mutable input mismatch: " + ", ".join(mismatches))
        completion = {"eval_manifest_sha256": digest,
                      "log": {"path": str(log_path), "sha256": p.sha256_file(log_path)},
                      "outputs": {name: p.sha256_file(run / name) for name in OUTPUTS}}
        reason = "completion_failed"
        p.write_completion(run / "completion.json", completion)
        return completion
    except BaseException:
        completion_path = run / "completion.json"
        if completion_path.exists():
            completion_path.unlink()
        aborted = run.with_name(run.name + "_ABORTED_" + reason)
        if aborted.exists():
            aborted = aborted.with_name(aborted.name + "_" + uuid.uuid4().hex)
        run.rename(aborted)
        raise
