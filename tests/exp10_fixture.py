"""A synthetic exp_10 ``yaw_pilot`` run directory, for the comparator and summariser tests.

Writing a real run costs a model, the AcousticRooms cache and a minute of CPU; what the
offline tools need is only the *shape* of a run: waveform arrays, a per-sample file whose
Metric-1 values really do describe those arrays, and a meta block that binds the two.
This builder produces exactly that -- with the pilot's own gap functions, so the stored
values are the ones an honest run would have stored -- and lets a test corrupt any one
piece to check that the guard refuses it.
"""
import json
import os

import numpy as np

from tools import exp10_yaw_pilot as pilot
from tools.per_sample_metrics import acoustic_metrics


def decaying_noise(n, length=pilot.PADDED_LEN, seed=0, tau=1200.0):
    """``n`` well-behaved synthetic impulse responses (noise under an exponential decay)."""
    rng = np.random.RandomState(seed)
    envelope = np.exp(-np.arange(length) / tau)
    waves = (rng.randn(n, length) * envelope).astype(np.float32)
    waves[:, pilot.NATIVE_LEN:] = 0.0
    return waves


def make_run(out_dir, ks=(0, 128), n=8, seed=0, controls=True, shift=0.01,
             meta_overrides=None, per_sample_meta_overrides=None, complete=True,
             queries=None, indices=None, batches=(0,), execution_id=None):
    """Write a complete, self-consistent synthetic run directory and return its meta.

    Args:
        out_dir: directory to create the run in.
        ks: the angles; ``0`` must be first (it is the paired reference).
        n: number of queries.
        seed: seed of the synthetic waveforms.
        controls: also write the three control cells (as exact repeats of their angles).
        shift: how far the rotated waveforms are moved away from the ``k = 0`` ones.
        meta_overrides: fields to overwrite in ``meta.json`` after it is built.
        per_sample_meta_overrides: fields to overwrite in ``per_sample.json``'s meta copy
            (this is what the guard compares against ``meta.json``).
        complete: value of ``meta.complete``.
        queries / indices: override the canonical query list and their canonical indices.
        batches: the canonical batch indices the run claims to cover.

    Returns:
        The meta dict that was written.
    """
    from eval_unseen import Evaluator

    os.makedirs(out_dir, exist_ok=True)
    evaluator = Evaluator()
    queries = list(queries if queries is not None else
                   ["Cat/Room_idx_0/S{:03d}_R001_hybrid_IR.wav".format(i) for i in range(n)])
    indices = list(indices if indices is not None else range(n))
    gt_waves = decaying_noise(n, seed=seed + 100, tau=900.0)

    waves = {}
    rng = np.random.RandomState(seed + 1)
    for k in ks:
        if int(k) == 0:
            waves[int(k)] = decaying_noise(n, seed=seed)
        else:
            moved = waves[0] + (shift * rng.randn(n, pilot.PADDED_LEN)).astype(np.float32)
            moved[:, pilot.NATIVE_LEN:] = 0.0
            waves[int(k)] = moved

    raw_0 = pilot.raw_measures(waves[0], evaluator)
    raw_gt = pilot.raw_measures(gt_waves, evaluator)
    angles = {}
    for k in ks:
        block = waves[int(k)]
        cell = {}
        cell.update(pilot.waveform_gap(waves[0], block))
        cell.update(pilot.acoustic_gap(waves[0], block, evaluator))
        errors = [acoustic_metrics(block[i], gt_waves[i], evaluator) for i in range(n)]
        for name, key in (("edt_err", "edt"), ("c50_err", "c50"), ("t60_err", "t60")):
            cell[name] = np.asarray([e[key] for e in errors], dtype=np.float64)
        raw_k = pilot.raw_measures(block, evaluator)
        for name, key in (("edt_pred", "edt"), ("c50_pred", "c50"), ("t60_pred", "t60")):
            cell[name] = raw_k[key]
        magnitude = 0.0 if int(k) == 0 else 0.02 + 0.001 * int(k) / 128.0
        cell["logspec_mad"] = np.full(n, magnitude) + 0.001 * np.arange(n)
        cell["mag_rel_l2"] = np.full(n, magnitude * 2.0)
        cell["log_mse"] = 0.5 + 0.01 * np.arange(n) + magnitude
        cell["loss"] = 1.0 + magnitude + 0.01 * np.arange(n)
        cell["stft"] = 0.6 + magnitude
        cell["decay"] = 0.4 + 0.01 * np.arange(n)
        cell["t60_gap_abs"] = np.abs(raw_k["t60"] - raw_0["t60"])
        cell["t60_err_abs"] = np.abs(raw_k["t60"] - raw_gt["t60"])
        for name in cell:
            cell[name] = np.asarray(cell[name], dtype=np.float64) * np.ones(n)
        angles[int(k)] = cell

    control_cells = {}
    if controls:
        for name, k, reference in pilot.CONTROL_SPECS:
            if int(reference) not in angles:
                continue
            cell = {metric: values.copy() for metric, values in angles[int(reference)].items()}
            cell["wave_max_abs_diff"] = np.zeros(n)
            cell["logspec_max_abs_diff"] = np.zeros(n)
            control_cells[name] = cell

    arrays = {}
    for k in ks:
        name = "wav_k{}.npy".format(int(k))
        path = os.path.join(out_dir, name)
        np.save(path, waves[int(k)])
        arrays[name] = {"sha256": pilot.file_sha256(path),
                        "shape": list(waves[int(k)].shape), "dtype": "float32",
                        "k": int(k)}

    protocol_fields = {"checkpoint_sha256": "c" * 64, "manifest_hash": "m" * 64,
                       "gl_seed": 0, "num_shot": 8, "device": "cpu",
                       "cudnn_deterministic": True, "cudnn_allow_tf32": False,
                       "matmul_allow_tf32": False, "batch_size": 16,
                       "batch_canonical": True, "torch_version": "2.0.1+cu117",
                       "numpy_version": "1.23.5", "tool_sha256": "t" * 64}
    execution, started = pilot.new_execution_id()
    meta_core = dict(protocol_fields)
    meta_core.update({
        "arm": "fixture", "backbone": "simple", "checkpoint": "checkpoints/fixture.pth",
        "manifest_path": "ckpt/yaw_rotation/reference_manifest.json",
        "manifest_seed": 0, "split": "unseen", "ks": [int(k) for k in ks],
        "controls": bool(controls),
        "control_specs": [{"id": cid, "k": int(ck), "compare_to": int(cr)}
                          for cid, ck, cr in pilot.CONTROL_SPECS] if controls else [],
        "tf32": False, "batches_arg": ",".join(str(b) for b in batches),
        "batches": list(batches), "n_batches_total": 397, "n_queries": n,
        "n_split": 6337, "native_len": pilot.NATIVE_LEN, "padded_len": pilot.PADDED_LEN,
        "sample_rate": pilot.SAMPLE_RATE, "metric_window": pilot.METRIC_WINDOW,
        "threads": 8, "num_workers": 2, "python_version": "3.8.20",
        "git_commit": "0" * 40, "git_dirty": False,
        "verify_exp03_pins": {"ok": True, "n_files": 12, "files": []},
        "run_id": pilot.protocol_id({"device": "cpu"}),
        "protocol_id": pilot.protocol_id(protocol_fields),
        "execution_id": execution_id or execution, "started_at": started,
        "arrays": arrays, "query_list_sha256": pilot.query_list_sha256(queries)})
    if meta_overrides:
        meta_core.update(meta_overrides)

    per_sample_meta = dict(meta_core)
    if per_sample_meta_overrides:
        per_sample_meta.update(per_sample_meta_overrides)
    per_sample = {
        "meta": per_sample_meta, "protocol_id": per_sample_meta["protocol_id"],
        "execution_id": per_sample_meta["execution_id"], "query": queries,
        "index": indices, "batches": list(batches),
        "gt": {"edt_gt": pilot.json_values(raw_gt["edt"]),
               "c50_gt": pilot.json_values(raw_gt["c50"]),
               "t60_gt": pilot.json_values(raw_gt["t60"])},
        "angles": {str(int(k)): {metric: pilot.json_values(values)
                                 for metric, values in cell.items()}
                   for k, cell in angles.items()},
        "controls": {name: {metric: pilot.json_values(values)
                            for metric, values in cell.items()}
                     for name, cell in control_cells.items()}}
    pilot.write_json(per_sample, os.path.join(out_dir, "per_sample.json"))
    metrics_out = {"meta": meta_core, "n_queries": n,
                   "angles": {str(int(k)): {metric: {"mean": float(np.nanmean(values)),
                                                     "n_valid": int(np.isfinite(values).sum()),
                                                     "n_nan": int((~np.isfinite(values)).sum())}
                                            for metric, values in cell.items()}
                              for k, cell in angles.items()}}
    pilot.write_json(metrics_out, os.path.join(out_dir, "metrics.json"))

    meta = dict(meta_core)
    meta.update({"per_sample_sha256": pilot.file_sha256(
                     os.path.join(out_dir, "per_sample.json")),
                 "metrics_sha256": pilot.file_sha256(os.path.join(out_dir, "metrics.json")),
                 "timing": {"inference_s": 1.0, "inversion_s": 1.0, "metrics_s": 1.0,
                            "loading_s": 0.5, "writing_s": 0.1, "total_min": 0.1},
                 "complete": complete})
    pilot.write_json(meta, os.path.join(out_dir, "meta.json"))
    return meta


def edit_json(path, mutate):
    """Load a JSON file, hand it to ``mutate`` and write it back (for corruption tests)."""
    with open(path) as fin:
        payload = json.load(fin)
    mutate(payload)
    pilot.write_json(payload, path)
    return payload
