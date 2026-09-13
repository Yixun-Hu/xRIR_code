"""Tier selection, args binding and fail-closed launcher completion."""
import json
from pathlib import Path

import pytest

from test_exp04_eval_launch import attempt, child_code, launch_args
from tools import exp04_eval_launch as launcher, provenance as p
from tools.exp05_params import TIERS

ROOT = Path(launcher.__file__).resolve().parents[1]
TIER_FIELDS = ("tier", "param_counts", "args_json_sha256", "legacy_M",
               *("vit_" + key for key in TIERS["M"]))


def select(args, extra=()):
    cli = launcher.child_command(args, ROOT)[2:]
    index = cli.index("--eval-manifest")
    return launcher.parse_args(cli[:index] + cli[index + 2:] + [
        "--log-dir", args.log_dir, "--data-root", args.data_root, "--run-label", "tier",
        "--reviewed-commit", "HEAD", "--num-shot", "8", *extra])


@pytest.mark.parametrize("extra,tier,entry", [([], "M", "exp04"), (["--entry", "exp05"], "M", "exp05"),
    *[(["--tier", tier], tier, "exp05") for tier in TIERS]])
def test_child_selection_preserves_default(launch_args, extra, tier, entry):
    args = select(launch_args, extra)
    command = launcher.child_command(args, ROOT)
    assert args.tier == tier and args.entry == entry
    assert Path(command[1]).name == entry + "_eval.py"
    assert ("--tier" in command) is (entry == "exp05")
    if entry == "exp05":
        assert command[command.index("--tier") + 1] == tier


@pytest.mark.parametrize("fault", [None, "args_mismatch", "train_args", *TIER_FIELDS])
def test_tier_binding_and_completion(launch_args, attempt, fault):
    args = select(launch_args, ["--tier", "S"])
    train_args = Path(args.checkpoint).parent / "args.json"
    config = {"vit_" + key: value for key, value in TIERS["L" if fault == "args_mismatch" else "S"].items()}
    train_args.write_text(json.dumps(config))
    if fault == "args_mismatch":
        with pytest.raises(ValueError, match="tier"):
            launcher.build_fields(args, launcher.child_command(args, ROOT), ROOT)
        return
    built = launcher.build_fields(args, launcher.child_command(args, ROOT), ROOT)
    binding = {"path": str(train_args), "sha256": p.sha256_file(train_args)}
    assert built["mutable_inputs"]["train_args"] == binding
    assert built["args_json_sha256"] == binding["sha256"] and built["legacy_M"] is False
    assert built["source_closures"]["entrypoint"]["files"][0]["path"] == "tools.exp05_eval.py"
    _, paths, fields = attempt
    fields.update({key: built[key] for key in TIER_FIELDS})
    fields.update(mutable_inputs={"train_args": binding}, manifest_file_sha256=p.sha256_file(paths["reference"]))
    extra = "Path(%r).write_text('changed')" % str(train_args) if fault == "train_args" else ""
    if fault in TIER_FIELDS:
        target = str(Path(args.out_dir) / launcher.OUTPUTS[0])
        extra = ("payload=json.loads(Path(%r).read_text()); payload['meta'][%r]='tampered'; "
                 "Path(%r).write_text(json.dumps(payload))") % (target, fault, target)
    command = child_code(args.out_dir, extra)
    if fault:
        with pytest.raises(ValueError):
            launcher.execute_run(args, command, lambda: fields, args.data_root)
        aborted, = Path(args.data_root).glob("run_ABORTED_*")
        assert not (aborted / "completion.json").exists()
    else:
        completion = launcher.execute_run(args, command, lambda: fields, args.data_root)
        assert {key: completion[key] for key in TIER_FIELDS} == {key: built[key] for key in TIER_FIELDS}
