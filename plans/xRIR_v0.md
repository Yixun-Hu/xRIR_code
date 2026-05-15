# xRIR v0 (receiver-origin) — train & eval plan

## Context

The user is introducing a new coordinate convention for the **xRIR** model on the AcousticRooms dataset (`data/single_channel_ir/`, `data/depth_map/`, `data/metadata/`). The current xRIR pipeline already places the **query receiver at the origin** for the query side (depth map and query source are in the query receiver's camera frame), but the **context (reference) RIRs are not**: today each reference's source is expressed in *its own* receiver's camera frame, and the reference receiver position is loaded from JSON and then discarded. That mismatch makes the ViT's per-reference input `(ref_src - depth_coord)/5.` geometrically inconsistent (it subtracts a vector in one receiver's frame from a depth map in another). The v0 spec fixes this by:

- Defining the **query receiver as the local origin**: `T_q(x) = x - r_q^g`, so query target is `h(\tilde{s}_q, 0)` with `\tilde{s}_q = s_q^g - r_q^g`, and the query depth map `G_{r_q}` is already in that frame.
- For each context RIR `h_i = h_m(s_i^g, r_i^g)`, expressing **both** endpoints in the query-receiver frame: `\tilde{s}_i^{(q)} = s_i^g - r_q^g`, `\tilde{r}_i^{(q)} = r_i^g - r_q^g`. The query receiver is the only point set to zero — context receivers stay explicit.
- Preserving each context RIR's pair-local geometry `d_i = s_i^g - r_i^g` (direct-path distance/direction of the context pair itself).

Resulting tokens:
- Context: `c_i = φ_ctx(E(h_i), d_i, s_i^g - r_q^g, r_i^g - r_q^g)`
- Query:   `q = φ_q(s_q^g - r_q^g, G_{r_q})`

Goal: train and evaluate xRIR under this convention on AcousticRooms (both unseen and seen splits) without breaking the existing baseline.

## Decisions confirmed with user

1. Add v0 as **new parallel files** — keep baseline reproducible.
2. Cover **both unseen and seen** splits.
3. Encode `d_i` as a **full 3-vector** through the shared sinusoidal embedding (matches how `s_i - r_q` and `r_i - r_q` are encoded; preserves direction).
4. Final plan lives only at this path during Plan mode. After `ExitPlanMode`, the implementation step copies this file to `./plans/xRIR_v0.md` inside the repo.

## File-level changes

### A. Dataset

**New** `treble_multi_room_dataset/treble_xRIR_v0_dataset.py` — class `xRIR_v0_Dataset`. Fork of `treble_xRIR_dataset.py:xRIR_Dataset`. Identical `__init__` signature, same metadata/file-list construction, same `convert_equirect_to_camera_coord` / `get_3d_point_camera_coord` helpers.

Replace `__getitem__` (model on lines 132–155 of the original):
- Keep query side unchanged through line 150: `source_pos`, `listener_pos` (i.e. `r_q^g`), `proj_source_pos` (= `q_src_local = s_q^g - r_q^g`), `depth_coord`, `tgt_wav`.
- Replace the old `get_ir_and_location_for_other_sources` call with a new helper `get_ir_and_geom_v0(self, ir_file_path, listener_pos, num_ref_sources)`. For each sampled reference `i` it loads `src_loc_i, rec_loc_i` from JSON (same JSON the helper already opens) and emits three 3-D vectors:
  - `d_i = src_loc_i - rec_loc_i`                      (pair-local; equivalent to `get_3d_point_camera_coord(0, rec_loc_i, src_loc_i)`)
  - `s_minus_rq_i = src_loc_i - listener_pos`         (context source in query-receiver frame; = `get_3d_point_camera_coord(0, listener_pos, src_loc_i)`)
  - `r_minus_rq_i = rec_loc_i - listener_pos`         (context receiver in query-receiver frame; = `get_3d_point_camera_coord(0, listener_pos, rec_loc_i)`)
- All four geometry tensors share the same units/scale as the existing `proj_source_pos` (meters).

Return tuple (8 items):
```
(proj_listener_pos, q_src_local, depth_coord, tgt_wav,
 all_ref_irs, all_ref_d, all_ref_s_minus_rq, all_ref_r_minus_rq)
```

Per-sample shapes:
| name | shape | dtype |
|---|---|---|
| `proj_listener_pos` | `(3,)` | float32 (always `[0,0,0]`) |
| `q_src_local` | `(3,)` | float32 |
| `depth_coord` | `(3, 256, 512)` | float32 |
| `tgt_wav` | `(1, 9600)` | float32 |
| `all_ref_irs` | `(N, 9600)` | float32 |
| `all_ref_d`, `all_ref_s_minus_rq`, `all_ref_r_minus_rq` | `(N, 3)` each | float32 |

**New** `treble_multi_room_dataset/treble_xRIR_v0_seen_dataset.py` — class `xRIR_v0_Seen_Dataset`. Same fork-relationship to `treble_xRIR_seen_dataset.py`; identical edits as above. Reuses the existing `seen_test_split.pkl` selector.

### B. Model

**New** `model/xRIR_v0.py` — class `xRIR_v0(nn.Module)`. Import helpers from `model.xRIR` (`AudioEnc`, `embedding_module_log`, `basic_project2`, `apply_delay`). Do **not** subclass `xRIR` — the forward signature and `lin_proj_2` input dim both change.

Constructor delta vs. `xRIR.__init__` (lines 152–175):
- `source_network` (SimpleViT, `image_size=(256,512)`, `patch_size=(16,32)`, `dim=512`, `depth=12`, `heads=8`, `mlp_dim=512`, `channels=3`), `src_proj`, `dist_embedder` (`num_freqs=10, max_freq=7`), `lin_proj_0`, `audio_enc`, `time_embedder`, `time_proj`, `times` buffer — **unchanged**.
- `src_coord_proj`: **unchanged** — input `21*3 = 63` (1 + 2 × 10 = 21 channels per scalar × 3 dims), output `intermediate_ch = 256`. Shared by `q_src_local`, `d_i`, `s_minus_rq_i`, `r_minus_rq_i`.
- `lin_proj_1` (query token): **unchanged**, `3*intermediate_ch → intermediate_ch`. Query fusion remains `[src_feats, receiver_out, source_out]`.
- `lin_proj_2` (context token): **grow input from `5*C` to `7*C`** (`1792 → 256`). Context token fusion becomes
  `[receiver_out, ref_geo_feats, d_i_feats, s_minus_rq_i_feats, r_minus_rq_i_feats, a_feats_all]`
  with sizes `(C, C, C, C, C, 2C)` summing to `7C = 1792`. Here `a_feats_all` is the ResNet-18 output of `AudioEnc` per reference (512 = 2C).

New forward signature:
```
xRIR_v0.forward(depth_coord, x, q_src_local,
                ref_d, ref_s_minus_rq, ref_r_minus_rq, tgt_wav)
```

Body, mirroring `xRIR.forward` (lines 180–239) with these substitutions:
1. `x = self.shift_and_align(x, q_src_local, ref_d)`. The math at lines 254–266 is unchanged: `dist_src = ||q_src_local||`, `dist_ref[:,i] = ||ref_d[:,i]||` — exactly the source-to-receiver distance for each pair, which is what the direct-path delay needs. Hard-coded constants `343.` and `22050` stay.
2. Query ViT views: `(q_src_local[:,:,None,None] - depth_coord) / 5.` and `(-depth_coord) / 5.` → `source_out`, `receiver_out` ∈ `[B,1,C]` (identical to baseline lines 187–194).
3. **Reference ViT views (the v0 fix)**: loop `i ∈ [0, N)` and compute
   `ref_view_i = (ref_s_minus_rq[:, i, :, None, None] - depth_coord) / 5.`
   Both terms are now in the query receiver's camera frame, so the subtraction is geometrically consistent. Stack → `ref_geo_feats ∈ [B,N,C]`.
4. Reference geometry projections (three per ref), each in a loop matching baseline line 211:
   - `d_feats[:,i] = src_coord_proj(dist_embedder(ref_d[:, i:i+1] / 5.).view(B,-1))`
   - `s_minus_rq_feats[:,i] = src_coord_proj(dist_embedder(ref_s_minus_rq[:, i:i+1] / 5.).view(B,-1))`
   - `r_minus_rq_feats[:,i] = src_coord_proj(dist_embedder(ref_r_minus_rq[:, i:i+1] / 5.).view(B,-1))`
   Each → `[B,N,C]` after stacking.
5. Query geometry: `src_feats = src_coord_proj(dist_embedder(q_src_local.unsqueeze(1)/5.).view(B,-1)).unsqueeze(1)` → `[B,1,C]` (= baseline line 214).
6. Token assembly:
   - Query: `fuse_geo = cat([src_feats, receiver_out, source_out], dim=-1)`, `[B,1,3C]`, then `lin_proj_1` → `[B,1,C]`.
   - Context: `fuse_ref = cat([receiver_out.repeat(1,N,1), ref_geo_feats, d_feats, s_minus_rq_feats, r_minus_rq_feats, a_feats_all], dim=-1)`, `[B,N,7C]`, then `lin_proj_2` → `[B,N,C]`.
7. Cross-attention / time decoding / spectrogram head: **unchanged** from baseline lines 234–239. Reuse `convert_ir_to_spec` verbatim.

### C. Train scripts

**New** `train_xRIR_v0_unseen.py` — fork of `train_xRIR_unseen.py`. Diffs:
- `from treble_multi_room_dataset.treble_xRIR_v0_dataset import xRIR_v0_Dataset`
- `from model.xRIR_v0 import xRIR_v0`
- Tuple unpacking in `train()` and `test()` becomes the 8-item form. The current 6-item unpacking at lines 17–19 and 45–47 must change to:
  ```
  (_, q_src_local, depth_coord, tgt_wav,
   all_ref_irs, all_ref_d, all_ref_s_minus_rq, all_ref_r_minus_rq) = data
  ```
- Forward call (was line 21 / 48):
  ```
  out_spec, tgt_spec = model(
      depth_coord.cuda(), all_ref_irs.cuda(), q_src_local.cuda(),
      all_ref_d.cuda(), all_ref_s_minus_rq.cuda(), all_ref_r_minus_rq.cuda(),
      tgt_wav.cuda())
  ```
- Checkpoint dir: `./ckpt/xRIR_v0_{num_shot}_shot/` (was `./ckpt/xRIR_{num_shot}_shot/` on lines 114, 116). Loss (`stft_l1_loss` + `compute_spect_energy_decay_losses`) and all hyperparameters (lr 1e-3, batch 64, num_shot 8, max_len 9600, 200 epochs, decay_epochs 50, lr_gamma 0.1, weight_decay 1e-4) unchanged.

There is **no** existing `train_xRIR_seen.py`; the seen split remains eval-only.

### D. Eval scripts

**New** `eval_xRIR_v0_unseen.py` — fork of `eval_unseen.py`. Modify only the `if __name__ == "__main__":` block (line 355+):
- Same dataset / model import swaps as the train script.
- `test_dataset = xRIR_v0_Dataset(num_shot=8, split="test")`.
- 8-item tuple unpacking and matching forward call.
- Default checkpoint path: `./checkpoints/xRIR_v0_unseen.pth` (user produces this via the new train script).
- Leave the `Evaluator` class and `measure_*` helpers (lines 1–354) untouched — they are reused as-is.

**New** `eval_xRIR_v0_seen.py` — fork of `eval_seen.py`, same pattern, uses `xRIR_v0_Seen_Dataset`. Default checkpoint path: `./checkpoints/xRIR_v0_seen.pth`.

### E. Plan-file copy

After `ExitPlanMode` (i.e. during implementation), copy this plan to `./plans/xRIR_v0.md` in the repo, creating the `plans/` directory if needed. Plan mode forbids editing repo files, so the copy is deferred to the implementation step.

## Constants kept in lockstep

Must be byte-identical across the new dataset, new model, baseline, and `utils/spec_utils.py`:
- Sample rate `22050` (dataset asserts at lines 146, 192; `shift_and_align` line 260; `Evaluator.measure_*` defaults in `eval_unseen.py`).
- `max_len = 9600` (dataset default + zero-pad logic lines 147–150 and 193–196; train script lines 81/84).
- Depth panorama `(256, 512)` (dataset line 144; `xRIR.__init__` defaults `image_size=(256,512)`, `patch_size=(16,32)`).
- STFT `fft_size=124, hop_size=31, win_length=62, window=hann_window(62)` (`xRIR.convert_ir_to_spec` lines 244–250; `n_bins=310` matches this STFT config).
- Coord scale `5.` (model lines 187, 192, 199, 211, 214). All four geometry vectors (`q_src_local`, `d_i`, `s_minus_rq`, `r_minus_rq`) divide by `5.` before the sinusoidal embedding and before subtraction from `depth_coord`.
- Sinusoidal embedding `num_freqs=10, max_freq=7` → 21 channels per scalar → `21*3 = 63` per 3-vector → `src_coord_proj` input dim.
- `intermediate_ch = 256`, `dim = 512`, `audio_enc` output = 512 = `2*intermediate_ch`. New `lin_proj_2` input dim = `7 * 256 = 1792`.
- `speed_of_sound = 343.` in `shift_and_align` line 260 — unchanged.

## Reused functions / utilities

- `convert_equirect_to_camera_coord`, `get_3d_point_camera_coord` (`treble_multi_room_dataset/treble_xRIR_dataset.py`) — used unchanged inside the v0 dataset.
- `AudioEnc`, `embedding_module_log`, `basic_project2`, `apply_delay`, `compute_energy_db` (`model/xRIR.py`) — imported by `model/xRIR_v0.py`.
- `SimpleViT` (`model/simple_vit.py`) — used as-is.
- `stft`, `stft_l1_loss`, `compute_spect_energy_decay_losses` (`utils/spec_utils.py`) — used as-is.
- `Evaluator`, `griffin_lim`, `measure_edt` / `measure_clarity` / `measure_rt60` (`eval_unseen.py`) — used as-is by both v0 eval scripts.
- `ExponentialLR` (`utils/lr_scheduler.py`) — used as-is.

## Verification

1. **Static shape assertions** inside `xRIR_v0.forward` (cheap `assert` calls, no jit hook needed):
   - After `shift_and_align`: `x.shape == [B, N, 9600]`.
   - `source_out.shape == receiver_out.shape == [B, 1, 256]`.
   - `ref_geo_feats.shape == [B, N, 256]`.
   - `d_feats.shape == s_minus_rq_feats.shape == r_minus_rq_feats.shape == [B, N, 256]`.
   - Pre-`lin_proj_2`: `[B, N, 1792]`. Pre-`lin_proj_1`: `[B, 1, 768]`.
   - `out_log_spec.shape == [B, 63, 310, 1]` matching the baseline.

2. **Frame-consistency invariant** check (in a standalone verify script, e.g. `treble_multi_room_dataset/_verify_xRIR_v0_dataset.py`): for one batch,
   - `all_ref_s_minus_rq - all_ref_r_minus_rq ≈ all_ref_d` within float tolerance (algebraic identity `(s - r_q) − (r − r_q) = s − r`).
   - Print `||all_ref_d - all_ref_s_minus_rq||` over a batch — non-zero values prove `r_q^g ≠ 0` and the new convention actually moved the reference coordinates (negative control against a silent identity-mapping bug).

3. **Smoke train**: in `train_xRIR_v0_unseen.py`, temporarily slice `train_dataset.file_list = train_dataset.file_list[:8]` and run with `batch_size=2, num_epoch=1, num_shot=8`. Confirm forward + backward complete, loss is finite, a checkpoint is written, and `eval_xRIR_v0_unseen.py` loads it without a `state_dict` mismatch.

4. **Negative control**: run `eval_xRIR_v0_unseen.py` against the baseline `checkpoints/xRIR_unseen.pth` — should fail with a `state_dict` size mismatch on `lin_proj_2` (input 5C vs. 7C). Confirms v0 and baseline checkpoints are not silently interchangeable.

5. **End-to-end run**: train v0 to convergence (200 epochs on full unseen-train split) using `bash`:
   ```
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   python train_xRIR_v0_unseen.py
   python eval_xRIR_v0_unseen.py  # after copying best ckpt to checkpoints/xRIR_v0_unseen.pth
   python eval_xRIR_v0_seen.py    # after a separate v0 seen training (or reuse the unseen ckpt as a baseline reference)
   ```
   Confirm EDT / C50 / T60 are reported and in a similar order of magnitude to the baseline numbers in `eval_unseen.py` output.

## Ambiguities (not blocking implementation)

- **Shared vs. per-call ViT**: today the ViT runs `2 + N` times per sample (query source view, query receiver view, each reference view). The v0 spec doesn't require changing this; plan keeps the per-call structure for minimal diff and direct comparability with the baseline.
- **`φ_q` content**: the spec writes `q = φ_q(s_q − r_q, G_{r_q})`. The current query-side fusion `[src_feats, receiver_out, source_out]` (which feeds both a "with-source" ViT view and a "depth-only" ViT view) realizes this. Plan keeps it. If a strict reading is preferred (depth-only ViT, no `source_out`), drop `source_out` and shrink `lin_proj_1` to `2C → C`.
- **Rotation augmentation**: baseline hard-codes `rotation = 0` in `__getitem__`. The new geometry vectors are computed by plain world-frame subtraction, which is correct only when `rotation = 0`. If rotation augmentation is later enabled, all four geometry vectors must be rotated by the query receiver's camera matrix from `get_3d_point_camera_coord(rotation, listener_pos, ...)`. Document this in a comment in the new dataset but do not implement now.

## Critical files

- `treble_multi_room_dataset/treble_xRIR_dataset.py` — source of the fork; reference for `__getitem__` structure (lines 132–207) and helpers.
- `treble_multi_room_dataset/treble_xRIR_seen_dataset.py` — same role for the seen split.
- `model/xRIR.py` — source of the model fork; reference for forward (lines 180–266) and projection dims (lines 152–175).
- `model/simple_vit.py` — ViT reused as-is.
- `train_xRIR_unseen.py` — source of the train-script fork.
- `eval_unseen.py`, `eval_seen.py` — source of the eval-script forks; `Evaluator` reused as-is.
- `utils/spec_utils.py` — losses and STFT helper reused as-is.
- `utils/lr_scheduler.py` — `ExponentialLR` reused as-is.
