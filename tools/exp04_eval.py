"""Manifest-bound exp_04 evaluation using the unchanged exp_03 numerical functions."""
import argparse
import json
import re
import subprocess
import time
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
    if not out.is_dir() or path.resolve().parent != out.resolve():
        raise ValueError("out_dir must exist and contain the eval_manifest")
    if any(item.absolute() != path.absolute() for item in out.iterdir()):
        raise ValueError("out_dir contains files other than eval_manifest")
    reference = yaw.load_manifest(args.manifest)
    expected = {key: value for key, value in vars(args).items()
                if key not in ("eval_manifest", "out_dir", "manifest")}
    expected.update(schema_version=1, manifest_path=args.manifest,
                    checkpoint_sha256=provenance.sha256_file(args.checkpoint),
                    manifest_file_sha256=provenance.sha256_file(args.manifest),
                    manifest_hash=yaw.manifest_hash(reference), num_shot=reference["num_shot"],
                    manifest_seed=reference["seed"], batch_canonical=True, data_root=BASE_DATA_PATH)
    mismatches = [key for key, value in expected.items()
                  if json.dumps(fields.get(key), sort_keys=True) != json.dumps(value, sort_keys=True)]
    if args.manifest_hash != expected["manifest_hash"]:
        mismatches.append("manifest_hash")
    if args.conditions == "P" and args.e_acoustic_cols:
        mismatches.append("e_acoustic_cols")
    repo = str(Path(__file__).resolve().parents[1])
    if fields.get("repo") != repo:
        mismatches.append("repo")
    try:
        commit = fields.get("reviewed_commit")
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise ValueError("reviewed_commit must be 40 lowercase hex characters")
        files, closure_digest = provenance.closure_record(
            provenance.source_closure("eval_yaw_rotation", repo), fields["reviewed_commit"], repo)
        declared = fields["evaluator_closure"]
        identity_keys = ("path", "reviewed_blob_sha256", "working_tree_sha256", "commits_after_reviewed")
        if (declared["sha256"] != closure_digest or
                [[r[key] for key in identity_keys] for r in declared["files"]] !=
                [[r[key] for key in identity_keys] for r in files] or
                any(r["working_tree_sha256"] != r["reviewed_blob_sha256"] or
                    r["commits_after_reviewed"] for r in files)):
            mismatches.append("evaluator_closure")
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as error:
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


def run_exp04(args, model_factory=None, metadata=None, manifest_validator=None):
    """Mirror the frozen run, binding its outputs to preflight-validated inputs."""
    fields, digest, manifest = (manifest_validator or validate_manifest)(args)
    if not yaw.torch.cuda.is_available():
        raise RuntimeError("exp04_eval needs a GPU: xRIR.apply_delay is .cuda()-only")
    started = time.time()
    yaw.torch.set_num_threads(args.threads)
    yaw.set_precision(args.tf32)
    manifest = yaw.load_checked_manifest(args.manifest, args.manifest_hash)
    cols, acoustic, e_acoustic = yaw._check_cols(args.yaw_cols, args.acoustic_cols, args.e_acoustic_cols)
    dataset = yaw.build_manifest_dataset(manifest, max_samples=args.max_samples)
    loader = yaw.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    model = (model_factory or yaw.build_xrir)(args.backbone, manifest["num_shot"])
    model.load_state_dict(yaw.load_model_state(args.checkpoint), strict=True)
    model.cuda().eval()
    evaluator = yaw.Evaluator()
    n_samples = len(dataset)
    print("backbone: {}  checkpoint: {}  samples: {}  angles: {}  batch: {}".format(
        args.backbone, args.checkpoint, n_samples, cols, args.batch_size), flush=True)
    queries, parts, flips, decompositions = [], {}, {}, []
    evaluate = yaw.evaluate_batch if args.conditions == "PE" else evaluate_p_batch
    for i, batch in enumerate(loader):
        keys, results, batch_flips = evaluate(model, batch, evaluator, cols, acoustic,
                                              e_acoustic, args.gl_seed, batch_size=args.batch_size)
        queries.extend(keys)
        for cell, metrics in results.items():
            for metric, values in metrics.items():
                parts.setdefault(cell, {}).setdefault(metric, []).append(values)
        for k, count in batch_flips.items():
            flips[k] = flips.get(k, 0) + count
        if args.backbone == "cylindrical" and i < args.decomposition_batches:
            # The unchanged diagnostic internally computes P and E forwards.
            decompositions.append(yaw._decomposition_for_batch(model, batch))
        if (i + 1) % args.log_interval == 0 or len(queries) == n_samples:
            rate = len(queries) / (time.time() - started)
            print("[{}/{}] {:.2f} samples/s, eta {:.1f} min".format(
                len(queries), n_samples, rate, (n_samples - len(queries)) / rate / 60), flush=True)
    entries = dataset.entries
    if queries != [entry["query"] for entry in entries]:
        raise ValueError("the loader did not return the manifest's canonical order")
    merged = {cell: {metric: yaw.np.concatenate(chunks) for metric, chunks in metrics.items()}
              for cell, metrics in parts.items()}
    decomposition = None
    if decompositions:
        names = ("tokens_rel_change", "pooled_rel_change", "coord_rel_change", "logspec_rel_change")
        decomposition = {"k": decompositions[0]["k"], "n_batches": len(decompositions)}
        decomposition.update({name: float(yaw.np.mean([d[name] for d in decompositions])) for name in names})
    meta = {key: getattr(args, key) for key in ("backbone", "checkpoint", "manifest_hash",
            "gl_seed", "batch_size", "max_samples", "tf32")}
    meta.update(manifest_path=args.manifest, manifest_seed=manifest["seed"], yaw_cols=cols,
                acoustic_cols=acoustic, e_acoustic_cols=e_acoustic, batch_canonical=True,
                n_samples=n_samples, torch_version=yaw.torch.__version__,
                elapsed_min=(time.time() - started) / 60, eval_manifest_sha256=digest,
                conditions=args.conditions, evaluator_closure_sha256=fields["evaluator_closure"]["sha256"],
                reviewed_commit=fields["reviewed_commit"])
    meta.update(metadata or {})
    per_sample = {"meta": meta, "query": queries, "index": [entry["index"] for entry in entries],
                  "delay_flips": {str(k): int(v) for k, v in sorted(flips.items())},
                  "decomposition": decomposition}
    metrics_out = {"meta": meta, "delay_flips": per_sample["delay_flips"], "decomposition": decomposition}
    for condition in args.conditions:
        per_sample[condition] = {str(k): {name: yaw._json_values(values) for name, values in
                                merged[(condition, k)].items()} for k in cols}
        metrics_out[condition] = {str(k): {name: yaw._summarize(values) for name, values in
                                 merged[(condition, k)].items()} for k in cols}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for name, payload in (("per_sample_yaw.json", per_sample), ("metrics_yaw.json", metrics_out)):
        path = str(Path(args.out_dir) / name)
        yaw._write_json(payload, path)
        print("wrote {}".format(path), flush=True)
    for name, path, expected in (("checkpoint", args.checkpoint, fields["checkpoint_sha256"]),
                                 ("manifest", args.manifest, fields["manifest_file_sha256"]),
                                 ("eval_manifest", args.eval_manifest, digest)):
        if provenance.sha256_file(path) != expected:
            raise ValueError("{} changed during evaluation".format(name))
    print("done: {} samples x {} angles x {} conditions in {:.2f} min".format(
        n_samples, len(cols), len(args.conditions), meta["elapsed_min"]), flush=True)
    return per_sample


def main(argv=None):
    return run_exp04(parse_args(argv))


if __name__ == "__main__":
    main()
