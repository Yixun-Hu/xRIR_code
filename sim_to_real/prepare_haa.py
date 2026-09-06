"""Build the compact Hearing-Anything-Anywhere (HAA) cache used by haa_dataset.py.

Per room it writes ``<out>/<room>/``:
  rirs.npy         [N, L] float32, 22.05 kHz mono, divided by the max |amplitude| over the
                   room's 12 DiffRIR training RIRs (README preprocessing), first L samples
  xyzs.npy         [N, 3] microphone positions (the model's "sources")
  speaker_xyz.npy  [3]    loudspeaker position (the model's "receiver"; panorama centre)
  depth_repo.npy   [256, 512] float32 panorama at the speaker from the repo's sim_to_real/depth_map
  depth_local.npy  [256, 512] float32 panorama from HAA_processed/<scene>/depth_images
  depth.npy        the default used for experiments: repo for class_room / hallway, local for
                   complex_room / dampened_room. The released complex_room.npy is byte-identical to
                   hallway.npy and the released dampened_room.npy is misaligned with the pose frame
                   (walls closer than the mic grid extends); the local renders are consistent.
  meta.json        splits (DiffRIR train / valid / test), scale, sr, L, depth_default

    python sim_to_real/prepare_haa.py --out ~/data_cache/HAA_xrir
"""
import argparse
import json
import os

import numpy as np
import torchaudio

ROOMS = {  # HAA scene name -> repo depth-map name
    "classroomBase": "class_room",
    "dampenedBase": "dampened_room",
    "hallwayBase": "hallway",
    "complexBase": "complex_room",
}
DEPTH_DEFAULT = {"class_room": "repo", "hallway": "repo", "complex_room": "local", "dampened_room": "local"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--haa-root", default="/media/diskstation/yixunhu/HAA_processed")
    p.add_argument("--depth-dir", default="sim_to_real/depth_map")
    p.add_argument("--out", default=os.path.expanduser("~/data_cache/HAA_xrir"))
    p.add_argument("--length", type=int, default=22050, help="samples kept per RIR (model uses the first 9600)")
    p.add_argument("--sr", type=int, default=22050)
    args = p.parse_args()

    scenes = json.load(open(os.path.join(args.haa_root, "metadata", "scenes_metadata.json")))
    poses = json.load(open(os.path.join(args.haa_root, "metadata", "poses_metadata.json")))
    test_json = json.load(open(os.path.join(args.haa_root, "test_base.json")))
    for scene, local in ROOMS.items():
        s = scenes[scene]
        n = len(poses[scene])
        xyzs = np.array([poses[scene][str(i)] for i in range(n)], dtype=np.float32)
        train = sorted(int(i) for i in s["train_indices"])
        valid = sorted(int(i) for i in s["valid_indices"])
        test = sorted(i for i in range(n) if i not in set(train) | set(valid))
        test_from_json = sorted(int(os.path.splitext(f)[0]) for f in test_json[scene])
        assert test == test_from_json, f"{scene}: test split differs from test_base.json"
        assert not (set(train) & set(valid)) and not (set(train) & set(test)) and not (set(valid) & set(test))

        rirs = np.zeros((n, args.length), dtype=np.float32)
        for i in range(n):
            w, sr = torchaudio.load(os.path.join(args.haa_root, scene, "mono_rirs_22050Hz", f"{i}.wav"))
            assert sr == args.sr and w.shape[0] == 1, (scene, i, sr, w.shape)
            w = w[0].numpy()
            m = min(args.length, len(w))
            rirs[i, :m] = w[:m]
        # README: divide by the training set's largest magnitude (12 samples in that room).
        scale = float(np.abs(rirs[train]).max())
        rirs /= scale
        depth_repo = np.load(os.path.join(args.depth_dir, f"{local}.npy")).astype(np.float32)
        depth_local = np.load(os.path.join(args.haa_root, scene, "depth_images", f"{scene}_depth_image.npy")).astype(np.float32)
        assert depth_repo.shape == depth_local.shape == (256, 512), (depth_repo.shape, depth_local.shape)
        depth = {"repo": depth_repo, "local": depth_local}[DEPTH_DEFAULT[local]]

        out = os.path.join(args.out, local)
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "rirs.npy"), rirs)
        np.save(os.path.join(out, "xyzs.npy"), xyzs)
        np.save(os.path.join(out, "speaker_xyz.npy"), np.array(s["speaker_xyz"], dtype=np.float32))
        np.save(os.path.join(out, "depth.npy"), depth)
        np.save(os.path.join(out, "depth_repo.npy"), depth_repo)
        np.save(os.path.join(out, "depth_local.npy"), depth_local)
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump({"scene": scene, "sr": args.sr, "length": args.length, "scale": scale,
                       "depth_default": DEPTH_DEFAULT[local],
                       "speaker_xyz": s["speaker_xyz"], "train": train, "valid": valid, "test": test}, f)
        print(f"{local:14s} ({scene}): n={n} train={len(train)} valid={len(valid)} test={len(test)} "
              f"scale={scale:.4f} depth={DEPTH_DEFAULT[local]} -> {out}")


if __name__ == "__main__":
    main()
