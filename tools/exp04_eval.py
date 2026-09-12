"""Manifest-bound exp_04 evaluation using the unchanged exp_03 numerical functions."""
import argparse
import json
from pathlib import Path

import eval_yaw_rotation as yaw
from tools import provenance
from treble_multi_room_dataset.treble_xRIR_dataset import BASE_DATA_PATH


def build_parser(require_eval_manifest=True):
    """Share the frozen evaluator's CLI, with the exp_04 manifest and condition gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=sorted(yaw.BACKBONES), required=True)
    for name in ("checkpoint", "manifest", "manifest-hash", "out-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--eval-manifest", required=require_eval_manifest)
    parser.add_argument("--conditions", choices=("P", "PE"), default="PE")
    for name, default in (("yaw-cols", yaw.SPECTRAL_COLS), ("acoustic-cols", yaw.ACOUSTIC_COLS),
                          ("e-acoustic-cols", yaw.E_ACOUSTIC_COLS)):
        parser.add_argument("--" + name, type=int, nargs="*", default=default)
    for name, default in (("batch-size", 16), ("num-workers", 6), ("max-samples", 0),
                          ("gl-seed", 0), ("threads", 4), ("log-interval", yaw.LOG_INTERVAL_DEFAULT),
                          ("decomposition-batches", 1)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--tf32", action="store_true")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def validate_manifest(args):
    """Refuse changed inputs, protocol, or reviewed evaluator bytes before model loading."""
    path = Path(args.eval_manifest)
    digest = provenance.sha256_file(path)
    fields = json.loads(path.read_text())
    out = Path(args.out_dir)
    if out.exists() and any(item.absolute() != path.absolute() for item in out.iterdir()):
        raise ValueError("out_dir contains files other than eval_manifest")
    reference = yaw.load_manifest(args.manifest)
    expected = {key: value for key, value in vars(args).items()
                if key not in ("eval_manifest", "out_dir", "manifest")}
    expected.update(manifest_path=args.manifest,
                    checkpoint_sha256=provenance.sha256_file(args.checkpoint),
                    manifest_file_sha256=provenance.sha256_file(args.manifest),
                    manifest_hash=yaw.manifest_hash(reference), num_shot=reference["num_shot"],
                    manifest_seed=reference["seed"], batch_canonical=True, data_root=BASE_DATA_PATH)
    mismatches = [key for key, value in expected.items()
                  if fields.get(key) != value or type(fields.get(key)) is not type(value)]
    if args.manifest_hash != expected["manifest_hash"]:
        mismatches.append("manifest_hash")
    if args.conditions == "P" and args.e_acoustic_cols:
        mismatches.append("e_acoustic_cols")
    repo = str(Path(__file__).resolve().parents[1])
    if fields.get("repo") != repo:
        mismatches.append("repo")
    try:
        files, closure_digest = provenance.closure_record(
            provenance.source_closure("eval_yaw_rotation", repo), fields["reviewed_commit"], repo)
        declared = fields["evaluator_closure"]
        if (declared["sha256"] != closure_digest or
                [r["path"] for r in declared["files"]] != [r["path"] for r in files] or
                any(r["working_tree_sha256"] != r["reviewed_blob_sha256"] or
                    r["commits_after_reviewed"] for r in files)):
            mismatches.append("evaluator_closure")
    except (KeyError, OSError, ValueError) as error:
        raise ValueError("reviewed_commit/evaluator_closure: {}".format(error)) from error
    if mismatches:
        raise ValueError("evaluation manifest mismatch: " + ", ".join(sorted(set(mismatches))))
    yaw._check_cols(args.yaw_cols, args.acoustic_cols, args.e_acoustic_cols)
    return fields, digest, reference


def evaluate_p_batch(model, batch, evaluator, cols, acoustic_cols=(), e_acoustic_cols=(),
                     gl_seed=0, batch_size=None):
    """The frozen P numerical path, including padding, without the E forward passes."""
    if e_acoustic_cols:
        raise ValueError("e_acoustic_cols must be empty for P")
    batch, n_real = yaw.pad_batch(batch, batch_size)
    _, src, depth, target, refs, locations, keys = batch
    src, depth = src.cuda(), depth.cuda()
    target, refs, locations = target.cuda(), refs.cuda(), locations.cuda()
    keys = list(keys)[:n_real]
    with yaw.torch.no_grad():
        aligned0 = model.shift_and_align(refs, src, locations)
        baseline, _ = model(depth, refs, src, locations, target)
    baseline, target_real = baseline[:n_real], target[:n_real]
    results = {}
    for k in cols:
        depth_k, src_k, locations_k = yaw.rotated_views(depth, src, locations, k)
        with yaw.torch.no_grad(), yaw.fixed_alignment(model, aligned0):
            prediction, spectrum = model(depth_k, refs, src_k, locations_k, target)
        prediction, spectrum = prediction[:n_real], spectrum[:n_real]
        cell = {name: values.double().numpy() for name, values in
                yaw.spectral_metrics(prediction, baseline, spectrum).items()}
        if int(k) in acoustic_cols:
            cell.update(yaw.acoustic_metrics_batch(prediction, target_real, keys, evaluator, gl_seed))
        results[("P", int(k))] = cell
    return keys, results, yaw.delay_flip_counts(src[:n_real], locations[:n_real], cols)
