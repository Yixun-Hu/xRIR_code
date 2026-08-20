"""Cross-receiver dataset for the xRIR baseline receiver-frame ablation.

This evaluation-only dataset uses the full AcousticRooms unseen test split.
For each query RIR it deterministically selects K reference RIRs from the same
room whose receivers differ from the query receiver. It returns both geometry
representations required by the controlled comparison:

``ref_pair_local  = source_i - receiver_i``
``ref_query_local = source_i - query_receiver``

The same returned reference audio is therefore usable for both the mismatched
and coordinate-aligned baseline conditions.
"""

import hashlib
import os

import numpy as np
import torch
import torchaudio

from treble_multi_room_dataset.treble_xRIR_dataset import (
    convert_equirect_to_camera_coord,
    get_3d_point_camera_coord,
    xRIR_Dataset,
)


class xRIRReceiverFrameDataset(xRIR_Dataset):
    """Full unseen-test dataset with deterministic cross-receiver references."""

    def __init__(
        self,
        max_len=9600,
        num_shot=8,
        pano_depth_path="depth_map",
        ir_path="single_channel_ir",
        metadata_path="metadata",
        reference_seed=0,
    ):
        super().__init__(
            split="test",
            max_len=max_len,
            num_shot=num_shot,
            pano_depth_path=pano_depth_path,
            ir_path=ir_path,
            metadata_path=metadata_path,
        )
        self.reference_seed = reference_seed

    def __getitem__(self, idx):
        ir_file_path = self.file_list[idx]
        ir_file_name = os.path.basename(ir_file_path).split("_hybrid_IR")[0]
        scene_name = ir_file_path.split("/")[-3]
        scene_id = ir_file_path.split("/")[-2]
        receiver_idx = int(ir_file_name.split("_")[1][1:])
        source_pos, query_receiver_pos = self.get_receiver_source_location(
            ir_file_path
        )

        query_source_local = get_3d_point_camera_coord(
            0, query_receiver_pos, source_pos
        )
        query_receiver_local = np.zeros(3, dtype=np.float32)

        pano_depth = np.load(
            os.path.join(
                self.pano_depth_path, scene_name, scene_id, f"{receiver_idx}.npy"
            )
        )
        depth_coord = convert_equirect_to_camera_coord(
            torch.from_numpy(pano_depth), 256, 512
        )

        target_rir = self._load_rir(ir_file_path)
        reference_rirs, ref_pair_local, ref_query_local = self._load_references(
            ir_file_path, query_receiver_pos
        )

        return (
            torch.as_tensor(query_receiver_local, dtype=torch.float32),
            torch.as_tensor(query_source_local, dtype=torch.float32),
            torch.as_tensor(depth_coord, dtype=torch.float32).permute(2, 0, 1),
            target_rir,
            reference_rirs,
            ref_pair_local,
            ref_query_local,
        )

    def _load_rir(self, path):
        rir, sample_rate = torchaudio.load(path)
        assert sample_rate == 22050, "IR sampling rate must be 22050!"
        if rir.shape[1] < self.max_len:
            rir = torch.cat(
                [rir, torch.zeros(rir.shape[0], self.max_len - rir.shape[1])],
                dim=1,
            )
        return rir[:, : self.max_len]

    def _load_references(self, query_path, query_receiver_pos):
        reference_paths = self._select_reference_paths(query_path)
        reference_rirs = []
        ref_pair_local = []
        ref_query_local = []

        for reference_path in reference_paths:
            reference_rirs.append(self._load_rir(reference_path))
            source_pos, receiver_pos = self.get_receiver_source_location(
                reference_path
            )
            ref_pair_local.append(
                torch.as_tensor(
                    get_3d_point_camera_coord(0, receiver_pos, source_pos),
                    dtype=torch.float32,
                )
            )
            ref_query_local.append(
                torch.as_tensor(
                    get_3d_point_camera_coord(0, query_receiver_pos, source_pos),
                    dtype=torch.float32,
                )
            )

        ref_pair_local = torch.stack(ref_pair_local)
        ref_query_local = torch.stack(ref_query_local)
        receiver_offset = ref_query_local - ref_pair_local
        assert torch.all(torch.linalg.norm(receiver_offset, dim=-1) > 1e-6)

        return (
            torch.cat(reference_rirs, dim=0),
            ref_pair_local,
            ref_query_local,
        )

    def _select_reference_paths(self, query_path):
        room_dir = os.path.dirname(query_path)
        query_receiver = self._receiver_token(query_path)
        candidates = sorted(
            os.path.join(room_dir, filename)
            for filename in os.listdir(room_dir)
            if filename.endswith("_hybrid_IR.wav")
            and self._receiver_token(filename) != query_receiver
        )
        if not candidates:
            raise RuntimeError(
                f"No cross-receiver reference candidates for {query_path}"
            )

        relative_query_path = os.path.relpath(query_path, self.ir_path)
        seed_material = f"{self.reference_seed}:{relative_query_path}".encode()
        query_seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8], "little"
        )
        rng = np.random.default_rng(query_seed)
        selected_indices = rng.choice(
            len(candidates),
            size=self.num_shot,
            replace=len(candidates) < self.num_shot,
        )
        return [candidates[index] for index in selected_indices]

    @staticmethod
    def _receiver_token(path):
        return os.path.basename(path).split("_")[1]


if __name__ == "__main__":
    dataset = xRIRReceiverFrameDataset()
    print("n_test:", len(dataset))
