"""Launch exp_04 evaluation and bind completion only after the child and log close."""
import datetime
import inspect
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from tools import provenance as p
from tools import exp04_eval as evaluator
from eval_unseen import griffin_lim

OUTPUTS = ("per_sample_yaw.json", "metrics_yaw.json")


def child_environment(repo, data_root):
    env = dict(os.environ, XRIR_DATA_PATH=str(data_root), CUDA_VISIBLE_DEVICES="1")
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_child(command, log_path, repo, data_root):
    """Run a literal argv through tee; both processes finish before the log closes."""
    env = child_environment(repo, data_root)
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
    log_path = None
    started = datetime.datetime.now().astimezone()
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
        fields = dict(fields_factory(), schema_version=1)
        manifest_path = run / "eval_manifest.json"
        digest = p.write_manifest(manifest_path, fields)
        reason = "child_failed"
        status = _run_child(command, log_path, repo, args.data_root)
        if status:
            raise RuntimeError(reason)
        reason = "output_invalid"
        listing = sorted(item.name for item in run.iterdir())
        if listing != sorted(["eval_manifest.json", *OUTPUTS]):
            raise RuntimeError(reason)
        expected = dict(eval_manifest_sha256=digest, conditions=fields["conditions"],
                        n_samples=fields["n_samples"])
        for name in OUTPUTS:
            payload = json.loads((run / name).read_text())
            meta = payload.get("meta", {})
            if any(type(meta.get(key)) is not type(value) or meta.get(key) != value
                   for key, value in expected.items()):
                raise ValueError("output meta mismatch: " + name)
            if name == "per_sample_yaw.json" and (not isinstance(payload.get("query"), list)
                    or len(payload["query"]) != fields["n_samples"]):
                raise ValueError("output query count mismatch")
        reason = "input_changed"
        declared = dict(fields, eval_manifest={"path": str(manifest_path), "sha256": digest})
        mismatches = p.revalidate(declared)
        if mismatches:
            raise ValueError("mutable input mismatch: " + ", ".join(mismatches))
        completion = {"schema_version": 1, "eval_manifest_sha256": digest,
                      "directory_listing": listing, "child_exit_status": status,
                      "started_at": started.isoformat(),
                      "ended_at": datetime.datetime.now().astimezone().isoformat(),
                      "log": {"path": str(log_path), "sha256": p.sha256_file(log_path)},
                      "outputs": {name: p.sha256_file(run / name) for name in OUTPUTS}}
        reason = "completion_failed"
        p.write_completion(run / "completion.json", completion)
        return completion
    except BaseException:
        if log_path is not None:
            try:
                log_path.rename(log_path.with_name(log_path.stem + "_ABORTED_" + reason + ".log"))
            except OSError:
                pass  # Preserve the original failure if the log cannot be renamed.
        completion_path = run / "completion.json"
        if completion_path.exists():
            completion_path.unlink()
        aborted = run.with_name(run.name + "_ABORTED_" + reason)
        if aborted.exists():
            aborted = aborted.with_name(aborted.name + "_" + uuid.uuid4().hex)
        run.rename(aborted)
        raise


def parse_args(argv=None):
    parser = evaluator.build_parser(require_eval_manifest=False)
    for name in ("run-label", "reviewed-commit", "data-root", "log-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--num-shot", type=int, required=True)
    args = parser.parse_args(argv)
    if args.eval_manifest is not None:
        parser.error("eval-manifest is created by the launcher")
    for key in ("out_dir", "checkpoint", "manifest", "data_root", "log_dir"):
        setattr(args, key, str(Path(getattr(args, key)).resolve()))
    args.eval_manifest = str(Path(args.out_dir) / "eval_manifest.json")
    return args


def child_command(args, repo):
    """Serialize only evaluator arguments, retaining empty grids and bool flags."""
    command = [sys.executable, str(Path(repo) / "tools/exp04_eval.py")]
    for action in evaluator.build_parser()._actions:
        if action.dest == "help":
            continue
        value = getattr(args, action.dest)
        if isinstance(value, bool):
            if value:
                command.append(action.option_strings[0])
        else:
            command.append(action.option_strings[0])
            command.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return command


def _capture_environment(repo, data_root):
    code = "from tools.provenance import environment; import json; print(json.dumps(environment()))"
    output = subprocess.check_output([sys.executable, "-c", code], cwd=repo, text=True,
                                     env=child_environment(repo, data_root))
    return json.loads(output.splitlines()[-1])


def build_fields(args, command, repo):
    """Bind protocol, reviewed code and full referenced data before spawning."""
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', args.reviewed_commit + '^{commit}'],
                                     cwd=repo, text=True).strip()
    reference = evaluator.yaw.load_checked_manifest(args.manifest, args.manifest_hash)
    if args.num_shot != reference["num_shot"]:
        raise ValueError("num_shot differs from reference manifest")
    evaluator.yaw._check_cols(args.yaw_cols, args.acoustic_cols, args.e_acoustic_cols)
    if args.conditions == "P" and args.e_acoustic_cols:
        raise ValueError("e_acoustic_cols must be empty for P")
    closures = {}
    for role, module in (("evaluator", "eval_yaw_rotation"), ("entrypoint", "tools.exp04_eval"),
                         ("writer", "tools.exp04_eval_launch")):
        records, digest = p.closure_record(p.source_closure(module, repo), commit, repo)
        if not records or any(record["reviewed_blob_sha256"] is None or
                record["reviewed_blob_sha256"] != record["working_tree_sha256"] or
                record["commits_after_reviewed"] for record in records):
            raise ValueError(role + " closure differs from reviewed_commit")
        closures[role] = {"files": records, "sha256": digest}
    fields = vars(evaluator.parse_args(command[2:]))
    fields.pop("eval_manifest")
    fields.pop("out_dir")
    fields["manifest_path"] = fields.pop("manifest")
    count = len(reference["entries"])
    fields.update(schema_version=1, repo=str(Path(repo).resolve()), reviewed_commit=commit,
        checkpoint_sha256=p.sha256_file(args.checkpoint),
        manifest_file_sha256=p.sha256_file(args.manifest), manifest_seed=reference["seed"],
        num_shot=args.num_shot, batch_canonical=True, data_root=args.data_root,
        data_identity=p.data_identity(args.manifest, args.data_root),
        evaluator_closure=closures.pop("evaluator"), source_closures=closures,
        environment=_capture_environment(repo, args.data_root), git_state=p.git_state(repo),
        run_label=args.run_label, command=command, split="unseen", split_count=count,
        n_samples=min(count, args.max_samples) if args.max_samples else count,
        no_tta=True,
        stft_gl={name: parameter.default for name, parameter in
                 inspect.signature(griffin_lim).parameters.items() if name != "spec"})
    fields["env"] = {key: child_environment(repo, args.data_root).get(key) for key in
                     ("CUDA_VISIBLE_DEVICES", "XRIR_DATA_PATH", "PYTHONPATH", "PYTHONHASHSEED")}
    return fields


def main(argv=None):
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    command = child_command(args, repo)
    return execute_run(args, command, lambda: build_fields(args, command, repo), repo)


if __name__ == "__main__":
    main()
