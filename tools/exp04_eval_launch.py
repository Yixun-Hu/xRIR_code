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
REQUIRED_INPUTS = ('repo', 'checkpoint', 'checkpoint_sha256', 'manifest_path', 'manifest_file_sha256',
                   'data_identity', 'evaluator_closure', 'source_closures', 'mutable_inputs', 'eval_manifest')
MUTABLE_INPUTS = {'control_args', 'train_manifest', 'train_completion', 'probe_receipt'}


def child_environment(repo, data_root, gpu='1'):
    env = dict(os.environ, XRIR_DATA_PATH=str(data_root), CUDA_VISIBLE_DEVICES=str(gpu))
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(
        [str(repo)] + [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p]))
    return env


def _run_child(command, log_path, repo, data_root, gpu='1'):
    """Run a literal argv through tee; both processes finish before the log closes."""
    env = child_environment(repo, data_root, gpu)
    child = sink = None
    with log_path.open("ab") as log:
        try:
            child = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            sink = subprocess.Popen(["tee", "-a", str(log_path)], stdin=child.stdout, stdout=2,
                                    start_new_session=True)
            child.stdout.close()
            status = child.wait()
            if sink.wait():
                raise RuntimeError("log_sink")
            log.flush()
            os.fsync(log.fileno())
            return status
        finally:
            p.reap_process_groups(child, sink)


@p.termination_handlers()
def execute_run(args, command, fields_factory, repo):
    """Own exclusive creation, spawn, digest revalidation and atomic finalisation."""
    run = Path(args.out_dir).resolve()
    run.mkdir()  # Deliberately before input hashing and outside abort handling.
    reason = "setup_failed"
    log_path = None
    log_created = False
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
        reason = "log_exists"
        with log_path.open("xb"):
            log_created = True
        reason = "child_failed"
        status = _run_child(command, log_path, repo, args.data_root, args.gpu)
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
        mismatches = p.revalidate(declared, required=REQUIRED_INPUTS)
        if mismatches:
            raise ValueError("mutable input mismatch: " + ", ".join(mismatches))
        reason = "output_invalid"
        completion = {"schema_version": 1, "eval_manifest_sha256": digest,
                      "confirmatory": fields['confirmatory'], "allow_dirty_used": fields['allow_dirty_used'],
                      "directory_listing": listing, "child_exit_status": status,
                      "started_at": started.isoformat(),
                      "ended_at": datetime.datetime.now().astimezone().isoformat(),
                      "log": {"path": str(log_path), "sha256": p.sha256_file(log_path)},
                      "outputs": {name: p.sha256_file(run / name) for name in OUTPUTS}}
        p.write_completion(run / "completion.json", completion)
        return completion
    except BaseException as error:
        reason = p.masked_abort_reason(error, reason)
        renamed = None
        if log_created:
            try:
                aborted_log = log_path.with_name(log_path.stem + "_ABORTED_" + reason + ".log")
                if aborted_log.exists():
                    aborted_log = aborted_log.with_name(aborted_log.stem + "_" + uuid.uuid4().hex + ".log")
                log_path.rename(aborted_log)
                renamed = str(aborted_log)
            except OSError:
                pass  # Preserve the original failure if the log cannot be renamed.
        ended = datetime.datetime.now().astimezone()
        p.write_completion(run / 'abort.json', dict(reason=reason, exception_type=type(error).__name__,
            exception_message=str(error), started_at=started.isoformat(), aborted_at=ended.isoformat(),
            wall_hours=(ended - started).total_seconds() / 3600,
            log={'original': str(log_path) if log_path else None, 'aborted': renamed}))
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
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--allow-dirty', action='store_true')
    parser.add_argument('--bind-input', action='append', default=[], metavar='NAME=PATH')
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


def _capture_environment(repo, data_root, gpu='1'):
    code = "from tools.provenance import environment; import json; print(json.dumps(environment()))"
    output = subprocess.check_output([sys.executable, "-c", code], cwd=repo, text=True,
                                     env=child_environment(repo, data_root, gpu))
    return json.loads(output.splitlines()[-1])


def build_fields(args, command, repo):
    """Bind protocol, reviewed code and full referenced data before spawning.

    Round-4 admission: dirty_outside_worklog is admissible only when the closure
    digests equal the profile's approved digests; per-file bindings cover evaluated
    code. mutable_inputs identifiers are control_args, train_manifest,
    train_completion and probe_receipt; all other identifiers are refused.
    """
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', args.reviewed_commit + '^{commit}'],
                                     cwd=repo, text=True).strip()
    reference = evaluator.yaw.load_checked_manifest(args.manifest, args.manifest_hash)
    count = len(reference["entries"])
    n_samples = min(count, args.max_samples) if args.max_samples else count
    state = p.checked_git_state(repo, n_samples == count, args.allow_dirty)
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
    fields.update(schema_version=1, repo=str(Path(repo).resolve()), reviewed_commit=commit,
        checkpoint_sha256=p.sha256_file(args.checkpoint),
        manifest_file_sha256=p.sha256_file(args.manifest), manifest_seed=reference["seed"],
        num_shot=args.num_shot, batch_canonical=True, data_root=args.data_root,
        data_identity=p.data_identity(args.manifest, args.data_root),
        evaluator_closure=closures.pop("evaluator"), source_closures=closures,
        environment=_capture_environment(repo, args.data_root, args.gpu), git_state=state,
        allow_dirty_used=bool(args.allow_dirty and state['dirty_outside_worklog']),
        confirmatory=args.max_samples == 0 and n_samples == count and not args.allow_dirty,
        run_label=args.run_label, command=command, split="unseen", split_count=count,
        n_samples=n_samples,
        no_tta=True,
        stft_gl={name: parameter.default for name, parameter in
                 inspect.signature(griffin_lim).parameters.items() if name != "spec"})
    fields['mutable_inputs'] = {}
    for binding in args.bind_input:
        name, separator, path = binding.partition('=')
        if name not in MUTABLE_INPUTS or not separator or not path or name in fields['mutable_inputs']:
            raise ValueError('invalid or duplicate --bind-input: ' + binding)
        path = str(Path(path).resolve())
        fields['mutable_inputs'][name] = {'path': path, 'sha256': p.sha256_file(path)}
    fields["env"] = {key: child_environment(repo, args.data_root, args.gpu).get(key) for key in
                     ("CUDA_VISIBLE_DEVICES", "XRIR_DATA_PATH", "PYTHONPATH", "PYTHONHASHSEED")}
    return fields


def main(argv=None):
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    command = child_command(args, repo)
    return execute_run(args, command, lambda: build_fields(args, command, repo), repo)


if __name__ == "__main__":
    main()
