"""Raw predictions for the joint FLAC + xRIR qualitative figure (requested by the
cylindrical-dinov3 session, 2026-09-23): float32 waveforms at 22050 Hz + manifest.

Same checkpoints / references / seeds / Griffin-Lim as make_qualitative_figure.py."""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_qualitative_figure as Q  # noqa: E402  (patches model.xRIR.apply_delay for CPU)
from eval_unseen import Evaluator  # noqa: E402
from sim_to_real.haa_dataset import NO_T60_ROOMS, HAADataset  # noqa: E402
from tools.reference_manifest import ManifestDataset, load_manifest  # noqa: E402

OUT = os.path.join(HERE, "joint_export")
DEFAULT_SIM = ["LivingRoomsWithHallway/LivingRoomsWithHallway_idx_25/S003_R015_hybrid_IR.wav",
               "Apartments/Apartments_idx_42/S002_R013_hybrid_IR.wav"]
DEFAULT_HAA = ["class_room:174", "complex_room:300"]
TAG = {"simple": "simplevit", "cylindrical": "cylvit"}
HAA_FILE_ROOM = {"class_room": "classroom", "dampened_room": "dampened", "hallway": "hallway", "complex_room": "complex"}


def clean(met):
    return {k: (None if not np.isfinite(v) else float(v)) for k, v in met.items()}


def export_sim(sim_query, evaluator, manifest):
    from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset
    man = load_manifest(os.path.join(Q.REPO, Q.SIM_MANIFEST))
    ds = ManifestDataset(xRIR_Dataset(split="test", max_len=9600, num_shot=8), man)
    pos = {e["query"]: k for k, e in enumerate(ds.entries)}
    item = ds[pos[sim_query]]
    entry = ds.entries[pos[sim_query]]
    s, r = sim_query.split("/")[-1].split("_")[:2]
    key = "sim_%s_%s" % (s, r)
    rec = {"env": "sim (AcousticRooms, unseen split)", "query": sim_query, "receiver_frame_source_xyz_m": item[1].tolist(),
           "reference_ids": entry["refs"], "reference_manifest": Q.SIM_MANIFEST, "manifest_seed": man["seed"],
           "griffin_lim_seed": "tools.per_sample_metrics.sample_seed(42, query)", "metric_window_samples": 8000,
           "gt_normalisation": "none (torchaudio.load of the hybrid IR wav, 22050 Hz, truncated to 9600)",
           "normalisation_constant": 1.0, "models": {}}
    np.save(os.path.join(OUT, "%s_gt.npy" % key), item[3][0].numpy().astype(np.float32))
    for b in ("simple", "cylindrical"):
        model = Q.load_model(b, Q.SIM_CKPT[b])
        p = Q.predict(model, item, sim_query, Q.SIM_GL_SEED, True, 8000, evaluator)
        np.save(os.path.join(OUT, "%s_%s.npy" % (key, TAG[b])), p["wav"].astype(np.float32))
        rec["models"][b] = {"checkpoint": Q.SIM_CKPT[b], "file": "%s_%s.npy" % (key, TAG[b]),
                            "errors_as_in_figure": clean(p["metrics"]), "log_stft_mse": p["log_mse"]}
        print(key, b, clean(p["metrics"]))
    manifest["queries"][key] = rec


def export_haa(room, idx, evaluator, manifest):
    hds = HAADataset([room], "test", num_shot=8, max_len=9600, eval_seed=Q.HAA_EVAL_SEED)
    item = hds[hds.items.index((room, idx))]
    refs = [int(i) for i in hds._pick_refs(room, idx)]
    meta = hds.data[room]["meta"]
    key = "haa_%s_mic%d" % (HAA_FILE_ROOM[room], idx)
    rec = {"env": "real (HAA %s, DiffRIR test split, two-stage fine-tuned)" % meta["scene"], "room": room, "mic_index": idx,
           "speaker_xyz_m": meta["speaker_xyz"],
           "mic_xyz_m": np.load(os.path.join(os.path.expanduser("~/data_cache/HAA_xrir"), room, "xyzs.npy"))[idx].tolist(),
           "reference_ids": refs, "reference_rule": "8 of the 12 DiffRIR training mics, sha256(eval_seed:room:idx)-seeded draw",
           "eval_seed": Q.HAA_EVAL_SEED, "fine_tuning_seed": Q.HAA_SEED,
           "griffin_lim_seed": "tools.per_sample_metrics.sample_seed(0, '%s/%d')" % (room, idx), "metric_window_samples": 9600,
           "gt_normalisation": "HAA mono_rirs_22050Hz/%d.wav divided by normalisation_constant (max |RIR| over the 12 training RIRs), first 9600 samples, no offset; predictions are on the same scale" % idx,
           "normalisation_constant": float(meta["scale"]),
           "t60_reported": room not in NO_T60_ROOMS, "models": {}}
    np.save(os.path.join(OUT, "%s_gt.npy" % key), item[3][0].numpy().astype(np.float32))
    for b in ("simple", "cylindrical"):
        ckpt = os.path.join("ckpt/sim2real", Q.HAA_ARM[b], Q.HAA_SEED, "stage2_%s" % room, "best.pth")
        model = Q.load_model(b, ckpt)
        p = Q.predict(model, item, "%s/%d" % (room, idx), Q.HAA_EVAL_SEED, room not in NO_T60_ROOMS, 9600, evaluator)
        np.save(os.path.join(OUT, "%s_%s.npy" % (key, TAG[b])), p["wav"].astype(np.float32))
        rec["models"][b] = {"checkpoint": ckpt, "file": "%s_%s.npy" % (key, TAG[b]),
                            "errors_recomputed": clean(p["metrics"]), "log_stft_mse": p["log_mse"]}
        ps = json.load(open(os.path.join(Q.REPO, "ckpt/sim2real", Q.HAA_ARM[b], Q.HAA_SEED, "eval", "per_sample_%s.json" % room)))
        j = ps["index"].index(idx)
        rec["models"][b]["errors_canonical_exp02"] = {m: ps[m][j] for m in ("edt", "c50", "t60")}
        print(key, b, clean(p["metrics"]), "canonical", rec["models"][b]["errors_canonical_exp02"])
    manifest["queries"][key] = rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", nargs="*", default=DEFAULT_SIM, help="AcousticRooms query paths")
    ap.add_argument("--haa", nargs="*", default=DEFAULT_HAA, help="<room>:<mic index>")
    args = ap.parse_args()
    torch.set_num_threads(16)
    os.makedirs(OUT, exist_ok=True)
    evaluator = Evaluator()
    mpath = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    manifest.update({"sample_rate": Q.SR, "length": 9600, "device": "cpu", "torch": torch.__version__,
                     "waveform_convention": ("GT: dataset waveform as scored (no time offset; first 9600 samples). Prediction: "
                                             "Griffin-Lim (32 iter, n_fft 124, hop 31, win 62; per-query seeded random phase) of the "
                                             "predicted magnitude, 9579 samples zero-padded to 9600; no offset, no normalisation.")})
    manifest.setdefault("queries", {})
    for q in args.sim:
        export_sim(q, evaluator, manifest)
    for spec in args.haa:
        room, idx = spec.split(":")
        export_haa(room, int(idx), evaluator, manifest)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
