"""Warm-start an exp_08 invariant model from a pinned ``epoch_12.pth``, with full accounting.

Handoff section 5 ("fast fine-tuning trial"): the compatible parameters of the existing
12-epoch checkpoints may be transferred into the new structure and fine-tuned.  Two things
change shape or meaning, so this is an **initialisation with a recorded provenance**, not a
resumption:

* ``lin_proj_0.proj.{weight,bias}`` -- ``[1, 256]`` token pool -> ``invariant_readout.weight``
  ``[16]`` + ``bias`` ``[1]`` via ``a[h] = sum_w W[h, w]`` (handoff section 3.2).  The old key
  is *consumed*, not loaded, and the new model has no ``lin_proj_0``.
* ``src_coord_proj`` keeps its shape but its **input semantics change**: it now sees the
  intrinsic horizontal coordinates instead of raw xyz (handoff section 3.3, closing note --
  "shape compatibility does not imply the old accuracy is preserved").

Nothing about the old optimizer state transfers either (the parameter set differs), so a
warm-started run is a new run with its own identity.

Source checkpoints live in the **main checkout** and are read-only here::

    cyl    -> /home/yixunhu/codespace/xRIR_code/ckpt/xRIR_cyl_8_shot/epoch_12.pth     -> cylindrical_invariant
    simple -> /home/yixunhu/codespace/xRIR_code/ckpt/xRIR_simple_8_shot/epoch_12.pth  -> simple_invariant

Usage::

    python -m tools.exp08_warmstart --backbone cylindrical_invariant
    python -m tools.exp08_warmstart --backbone simple_invariant --num-shot 8
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from model.exp08_factory import build_xrir_exp08
from model.xRIR_cyl_invariant import InvariantReadout, xRIR_InvariantBase

#: Read-only provenance: which pinned checkpoint warm-starts which exp_08 arm.
SOURCE_CHECKPOINTS = {
    "cylindrical_invariant": "/home/yixunhu/codespace/xRIR_code/ckpt/xRIR_cyl_8_shot/epoch_12.pth",
    "simple_invariant": "/home/yixunhu/codespace/xRIR_code/ckpt/xRIR_simple_8_shot/epoch_12.pth",
}

#: The old key that is consumed by (rather than copied into) the new structure.
CONSUMED_PREFIX = "lin_proj_0."

#: Where it goes.
READOUT_PREFIX = "invariant_readout."


def load_checkpoint_state(path):
    """Read ``path`` and normalise its keys exactly as ``eval_xRIR_backbone`` does."""
    from eval_xRIR_backbone import load_model_state  # single source of truth for the keying
    return load_model_state(path)


def warm_start(model, state_dict, verbose=True):
    """Transfer ``state_dict`` into ``model`` and return an explicit parameter accounting.

    Every checkpoint key lands in exactly one bucket and every new-model key is explained;
    the totals are asserted, so a silently dropped tensor cannot hide in the summary.

    Args:
        model: an :class:`model.xRIR_cyl_invariant.xRIR_InvariantBase` subclass.
        state_dict: a checkpoint state dict (already key-normalised, e.g. by
            :func:`load_checkpoint_state`).
        verbose: print the summary table.

    Returns:
        A dict with, per bucket, a list of ``{"key", "shape", "numel"}`` records:

        * ``loaded`` -- copied verbatim (same name, same shape);
        * ``consumed`` -- ``lin_proj_0.*``, folded into the invariant readout;
        * ``dropped_shape`` -- name matches but the shape does not (nothing is coerced);
        * ``dropped_absent`` -- present in the checkpoint, absent from the new model;
        * ``new_warm`` -- new-model keys initialised from ``consumed``;
        * ``new_random`` -- new-model keys the checkpoint could not fill (still at init).

        plus ``counts``/``numel`` summaries and ``strict_load`` (the ``missing``/
        ``unexpected`` lists that ``load_state_dict`` reported for the verbatim part).

    Raises:
        TypeError: if ``model`` is not an exp_08 invariant model.
        KeyError: if the checkpoint has no ``lin_proj_0`` pool to warm-start the readout from.
    """
    if not isinstance(model, xRIR_InvariantBase):
        raise TypeError("warm_start targets an exp_08 invariant model, got {}".format(
            type(model).__name__))

    target = model.state_dict()
    record = lambda k, t: {"key": k, "shape": list(t.shape), "numel": int(t.numel())}

    loaded, consumed, dropped_shape, dropped_absent = [], [], [], []
    transfer = {}
    for key, tensor in state_dict.items():
        if key.startswith(CONSUMED_PREFIX):
            consumed.append(record(key, tensor))
        elif key not in target:
            dropped_absent.append(record(key, tensor))
        elif tuple(target[key].shape) != tuple(tensor.shape):
            dropped_shape.append({**record(key, tensor),
                                  "model_shape": list(target[key].shape)})
        else:
            transfer[key] = tensor
            loaded.append(record(key, tensor))

    report = model.load_state_dict(transfer, strict=False)
    unexpected = list(report.unexpected_keys)
    if unexpected:  # cannot happen -- every key in `transfer` was checked against `target`
        raise RuntimeError("load_state_dict rejected keys it should accept: {}".format(unexpected))

    # The readout: a[h] = sum_w W[h, w], bias carried (handoff section 3.2).
    weight_key, bias_key = CONSUMED_PREFIX + "proj.weight", CONSUMED_PREFIX + "proj.bias"
    if weight_key not in state_dict or bias_key not in state_dict:
        raise KeyError("checkpoint has no {} / {} to warm-start the invariant readout".format(
            weight_key, bias_key))
    seeded = InvariantReadout.from_token_pool(
        (state_dict[weight_key], state_dict[bias_key]), h_tok=model.h_tok, w_tok=model.w_tok)
    with torch.no_grad():
        model.invariant_readout.weight.copy_(seeded.weight)
        model.invariant_readout.bias.copy_(seeded.bias)

    filled = set(transfer) | {READOUT_PREFIX + "weight", READOUT_PREFIX + "bias"}
    new_warm = [record(k, target[k]) for k in target if k.startswith(READOUT_PREFIX)]
    new_random = [record(k, target[k]) for k in target if k not in filled]

    accounting = {
        "loaded": loaded, "consumed": consumed, "dropped_shape": dropped_shape,
        "dropped_absent": dropped_absent, "new_warm": new_warm, "new_random": new_random,
        "strict_load": {"missing": sorted(report.missing_keys), "unexpected": unexpected},
    }
    accounting["counts"] = {name: len(accounting[name]) for name in
                            ("loaded", "consumed", "dropped_shape", "dropped_absent",
                             "new_warm", "new_random")}
    accounting["counts"]["checkpoint_total"] = len(state_dict)
    accounting["counts"]["model_total"] = len(target)
    accounting["numel"] = {name: int(sum(r["numel"] for r in accounting[name])) for name in
                           ("loaded", "consumed", "dropped_shape", "dropped_absent",
                            "new_warm", "new_random")}
    accounting["numel"]["checkpoint_total"] = int(sum(t.numel() for t in state_dict.values()))
    accounting["numel"]["model_total"] = int(sum(t.numel() for t in target.values()))

    # Conservation: every checkpoint key and every model key is in exactly one bucket.
    assert (accounting["counts"]["loaded"] + accounting["counts"]["consumed"]
            + accounting["counts"]["dropped_shape"] + accounting["counts"]["dropped_absent"]
            == accounting["counts"]["checkpoint_total"]), "checkpoint keys are not conserved"
    assert (accounting["counts"]["loaded"] + accounting["counts"]["new_warm"]
            + accounting["counts"]["new_random"] == accounting["counts"]["model_total"]), \
        "model keys are not conserved"

    if verbose:
        print(format_accounting(accounting), flush=True)
    return accounting


def format_accounting(accounting, max_examples=6):
    """Render :func:`warm_start`'s accounting as a short human-readable table."""
    counts, numel = accounting["counts"], accounting["numel"]
    lines = ["exp_08 warm-start accounting",
             "  checkpoint: {} tensors / {} params".format(
                 counts["checkpoint_total"], numel["checkpoint_total"]),
             "  new model : {} tensors / {} params".format(
                 counts["model_total"], numel["model_total"])]
    for name, label in (("loaded", "loaded verbatim"),
                        ("consumed", "consumed -> invariant_readout"),
                        ("dropped_shape", "dropped (shape mismatch)"),
                        ("dropped_absent", "dropped (absent from new model)"),
                        ("new_warm", "new, warm-started"),
                        ("new_random", "new, left at random init")):
        entries = accounting[name]
        lines.append("  {:32s} {:4d} tensors / {:10d} params".format(
            label, counts[name], numel[name]))
        for item in entries[:max_examples]:
            lines.append("      {} {}".format(item["key"], tuple(item["shape"])))
        if len(entries) > max_examples:
            lines.append("      ... and {} more".format(len(entries) - max_examples))
    strict = accounting["strict_load"]
    lines.append("  load_state_dict(strict=False): {} missing, {} unexpected".format(
        len(strict["missing"]), len(strict["unexpected"])))
    return "\n".join(lines)


def build_warm_started(backbone, num_shot=8, checkpoint=None, verbose=True, **kwargs):
    """Build ``backbone`` and warm-start it from its pinned checkpoint.

    Args:
        backbone: ``"cylindrical_invariant"`` or ``"simple_invariant"``.
        num_shot: reference count the model is built for.
        checkpoint: override the :data:`SOURCE_CHECKPOINTS` path.
        verbose: print the accounting.
        **kwargs: forwarded to :func:`model.exp08_factory.build_xrir_exp08`.

    Returns:
        ``(model, accounting, checkpoint_path)``.

    Raises:
        ValueError: if ``backbone`` has no pinned source checkpoint.
        FileNotFoundError: if the checkpoint is missing (the caller should STOP, not fake it).
    """
    path = checkpoint or SOURCE_CHECKPOINTS.get(backbone)
    if path is None:
        raise ValueError("no pinned warm-start checkpoint for backbone {!r}; known: {}".format(
            backbone, sorted(SOURCE_CHECKPOINTS)))
    if not os.path.isfile(path):
        raise FileNotFoundError("warm-start checkpoint not found: {}".format(path))
    model = build_xrir_exp08(backbone, num_shot, **kwargs)
    accounting = warm_start(model, load_checkpoint_state(path), verbose=verbose)
    return model, accounting, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", choices=sorted(SOURCE_CHECKPOINTS), required=True)
    parser.add_argument("--num-shot", type=int, default=8)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save-json", default=None, help="write the accounting here")
    args = parser.parse_args(argv)

    _, accounting, path = build_warm_started(args.backbone, args.num_shot, args.checkpoint)
    print("source checkpoint: {}".format(path))
    if args.save_json:
        parent = os.path.dirname(os.path.abspath(args.save_json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.save_json, "w") as fout:
            json.dump({"backbone": args.backbone, "checkpoint": path,
                       "accounting": accounting}, fout, indent=1)
        print("accounting -> {}".format(args.save_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
