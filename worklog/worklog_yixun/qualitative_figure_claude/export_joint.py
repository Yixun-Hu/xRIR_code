"""Raw predictions for the joint FLAC + xRIR qualitative figure (requested by the
cylindrical-dinov3 session, 2026-09-23): float32 waveforms at 22050 Hz + manifest.

Same checkpoints / references / seeds / Griffin-Lim as make_qualitative_figure.py."""
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
SIM_QUERY = "LivingRoomsWithHallway/LivingRoomsWithHallway_idx_25/S003_R015_hybrid_IR.wav"
HAA_ROOM, HAA_IDX = "class_room", 174
TAG = {"simple": "simplevit", "cylindrical": "cylvit"}


def clean(met):
    return {k: (None if not np.isfinite(v) else float(v)) for k, v in met.items()}


def main():
    torch.set_num_threads(16)
    os.makedirs(OUT, exist_ok=True)
    evaluator = Evaluator()
    manifest = {"sample_rate": Q.SR, "length": 9600, "device": "cpu", "torch": torch.__version__,
                "waveform_convention": ("GT: dataset waveform as scored (no time offset; first 9600 samples). Prediction: "
                                        "Griffin-Lim (32 iter, n_fft 124, hop 31, win 62; per-query seeded random phase) of the "
                                        "predicted magnitude, 9579 samples zero-padded to 9600; no offset, no normalisation."),
                "queries": {}}

    # ---- sim -----------------------------------------------------------------------------------
    from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset
    man = load_manifest(os.path.join(Q.REPO, Q.SIM_MANIFEST))
    ds = ManifestDataset(xRIR_Dataset(split="test", max_len=9600, num_shot=8), man)
    pos = {e["query"]: k for k, e in enumerate(ds.entries)}
    item = ds[pos[SIM_QUERY]]
    entry = ds.entries[pos[SIM_QUERY]]
    rec = {"env": "sim (AcousticRooms, unseen split)", "query": SIM_QUERY, "receiver_frame_source_xyz_m": item[1].tolist(),
           "reference_ids": entry["refs"], "reference_manifest": Q.SIM_MANIFEST, "manifest_seed": man["seed"],
           "griffin_lim_seed": "tools.per_sample_metrics.sample_seed(42, query)", "metric_window_samples": 8000,
           "gt_normalisation": "none (torchaudio.load of the hybrid IR wav, 22050 Hz, truncated to 9600)", "models": {}}
    np.save(os.path.join(OUT, "sim_S003_R015_gt.npy"), item[3][0].numpy().astype(np.float32))
    for b in ("simple", "cylindrical"):
        model = Q.load_model(b, Q.SIM_CKPT[b])
        p = Q.predict(model, item, SIM_QUERY, Q.SIM_GL_SEED, True, 8000, evaluator)
        np.save(os.path.join(OUT, "sim_S003_R015_%s.npy" % TAG[b]), p["wav"].astype(np.float32))
        rec["models"][b] = {"checkpoint": Q.SIM_CKPT[b], "file": "sim_S003_R015_%s.npy" % TAG[b],
                            "errors_as_in_figure": clean(p["metrics"]), "log_stft_mse": p["log_mse"]}
        print("sim", b, clean(p["metrics"]))
    manifest["queries"]["sim_S003_R015"] = rec

    # ---- HAA ----------------------------------------------------------------------------------
    hds = HAADataset([HAA_ROOM], "test", num_shot=8, max_len=9600, eval_seed=Q.HAA_EVAL_SEED)
    k = hds.items.index((HAA_ROOM, HAA_IDX))
    item = hds[k]
    refs = [int(i) for i in hds._pick_refs(HAA_ROOM, HAA_IDX)]
    meta = hds.data[HAA_ROOM]["meta"]
    rec = {"env": "real (HAA classroomBase, DiffRIR test split, two-stage fine-tuned)", "room": HAA_ROOM, "mic_index": HAA_IDX,
           "speaker_xyz_m": meta["speaker_xyz"], "mic_xyz_m": np.load(os.path.join(os.path.expanduser("~/data_cache/HAA_xrir"), HAA_ROOM, "xyzs.npy"))[HAA_IDX].tolist(),
           "reference_ids": refs, "reference_rule": "8 of the 12 DiffRIR training mics, sha256(eval_seed:room:idx)-seeded draw",
           "eval_seed": Q.HAA_EVAL_SEED, "fine_tuning_seed": Q.HAA_SEED,
           "griffin_lim_seed": "tools.per_sample_metrics.sample_seed(0, 'class_room/174')", "metric_window_samples": 9600,
           "gt_normalisation": "HAA mono_rirs_22050Hz/174.wav divided by %.8f (max |RIR| over the 12 training RIRs), first 9600 samples, no offset" % meta["scale"],
           "t60_reported": HAA_ROOM not in NO_T60_ROOMS, "models": {}}
    np.save(os.path.join(OUT, "haa_classroom_mic174_gt.npy"), item[3][0].numpy().astype(np.float32))
    for b in ("simple", "cylindrical"):
        ckpt = os.path.join("ckpt/sim2real", Q.HAA_ARM[b], Q.HAA_SEED, "stage2_%s" % HAA_ROOM, "best.pth")
        model = Q.load_model(b, ckpt)
        p = Q.predict(model, item, "%s/%d" % (HAA_ROOM, HAA_IDX), Q.HAA_EVAL_SEED, HAA_ROOM not in NO_T60_ROOMS, 9600, evaluator)
        np.save(os.path.join(OUT, "haa_classroom_mic174_%s.npy" % TAG[b]), p["wav"].astype(np.float32))
        rec["models"][b] = {"checkpoint": ckpt, "file": "haa_classroom_mic174_%s.npy" % TAG[b],
                            "errors_recomputed": clean(p["metrics"]), "log_stft_mse": p["log_mse"]}
        # canonical per-sample value from exp_02's evaluation (unseeded Griffin-Lim there)
        ps = json.load(open(os.path.join(Q.REPO, "ckpt/sim2real", Q.HAA_ARM[b], Q.HAA_SEED, "eval", "per_sample_%s.json" % HAA_ROOM)))
        j = ps["index"].index(HAA_IDX)
        rec["models"][b]["errors_canonical_exp02"] = {m: ps[m][j] for m in ("edt", "c50", "t60")}
        print("haa", b, clean(p["metrics"]), "canonical", rec["models"][b]["errors_canonical_exp02"])
    manifest["queries"]["haa_classroom_mic174"] = rec
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
