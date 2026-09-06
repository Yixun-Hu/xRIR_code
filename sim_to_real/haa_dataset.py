"""Hearing-Anything-Anywhere dataset for xRIR fine-tuning / evaluation.

Reads the cache built by ``sim_to_real/prepare_haa.py`` and returns the same tuple as
``treble_multi_room_dataset.xRIR_Dataset``:

    (listener_pos [3] (zeros), src_local [3], depth_coord [3, 256, 512],
     tgt_wav [1, L], ref_irs [K, L], ref_src_local [K, 3])

Role swap as in the paper's sim-to-real setup: the fixed loudspeaker is the model's
"receiver" (the panorama is rendered there) and the microphone positions are the
"sources". References are K of the room's 12 DiffRIR training RIRs, excluding the
target itself; drawn with the global numpy RNG for the train split (fresh every
call, as in the repo's HAA datasets) and deterministically per (eval_seed, room, idx)
when ``eval_seed`` is given, so different models see identical (query, references).
"""
import hashlib
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from treble_multi_room_dataset.treble_xRIR_dataset import convert_equirect_to_camera_coord

ROOMS = ["class_room", "dampened_room", "hallway", "complex_room"]
NO_T60_ROOMS = ["dampened_room"]  # paper omits T60 for the dampened room (low SNR)
DEFAULT_ROOT = os.environ.get("HAA_XRIR_ROOT", os.path.expanduser("~/data_cache/HAA_xrir"))
SPLIT_KEY = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}
DEPTH_FILE = {"default": "depth.npy", "repo": "depth_repo.npy", "local": "depth_local.npy"}


class HAADataset(Dataset):
    def __init__(self, rooms, split, root=DEFAULT_ROOT, num_shot=8, max_len=9600, eval_seed=None,
                 depth_variant="default"):
        self.rooms = list(rooms)
        self.split = split
        self.num_shot = num_shot
        self.max_len = max_len
        self.eval_seed = eval_seed
        self.data = {}
        self.items = []
        for room in self.rooms:
            d = os.path.join(root, room)
            meta = json.load(open(os.path.join(d, "meta.json")))
            rirs = np.load(os.path.join(d, "rirs.npy"))
            if rirs.shape[1] < max_len:
                rirs = np.pad(rirs, ((0, 0), (0, max_len - rirs.shape[1])))
            rirs = np.ascontiguousarray(rirs[:, :max_len], dtype=np.float32)
            depth = torch.from_numpy(np.load(os.path.join(d, DEPTH_FILE[depth_variant])).astype(np.float32))
            depth_coord = convert_equirect_to_camera_coord(depth, 256, 512).permute(2, 0, 1).float().contiguous()
            xyzs = np.load(os.path.join(d, "xyzs.npy")).astype(np.float32)
            spk = np.load(os.path.join(d, "speaker_xyz.npy")).astype(np.float32)
            self.data[room] = {
                "rirs": torch.from_numpy(rirs),
                "depth_coord": depth_coord,
                "src_local": torch.from_numpy(xyzs - spk[None]).float(),  # rotation 0: plain subtraction
                "train": [int(i) for i in meta["train"]],
                "meta": meta,
            }
            assert len(self.data[room]["train"]) > num_shot, room
            self.items += [(room, int(i)) for i in meta[SPLIT_KEY[split]]]

    def __len__(self):
        return len(self.items)

    def _pick_refs(self, room, idx):
        cands = [i for i in self.data[room]["train"] if i != idx]
        if self.eval_seed is None:
            return np.random.choice(cands, self.num_shot, replace=False)
        seed = int.from_bytes(hashlib.sha256(f"{self.eval_seed}:{room}:{idx}".encode()).digest()[:8], "little")
        return np.random.default_rng(seed).choice(cands, self.num_shot, replace=False)

    def __getitem__(self, i):
        room, idx = self.items[i]
        D = self.data[room]
        refs = torch.as_tensor(self._pick_refs(room, idx), dtype=torch.long)
        return (torch.zeros(3), D["src_local"][idx], D["depth_coord"], D["rirs"][idx:idx + 1],
                D["rirs"][refs], D["src_local"][refs])


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        ds = HAADataset(ROOMS, split, eval_seed=0)
        print(split, len(ds), {r: sum(1 for rr, _ in ds.items if rr == r) for r in ROOMS})
    b = ds[0]
    print("shapes:", [tuple(x.shape) for x in b], "src_local", b[1].tolist())
    print("deterministic refs:", torch.equal(ds[0][4], ds[0][4]))
