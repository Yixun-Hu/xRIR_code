# Plan — yaw_rotation_degradation

**Status:** draft v1 for Codex plan review, then user approval. No code has been written.

## 1. Question

Does **jointly rotating the receiver-centred depth panorama and the source / reference-source coordinates in yaw** (a symmetry of the acoustics: the true RIR is unchanged) substantially degrade RIR synthesis quality in the xRIR pipeline?

## 2. Background and mechanism (why the answer is not obvious)

- In `xRIR.forward`, room geometry enters only through the ViT run on `(loc − depth_coord)/5` for the query source, the receiver (`−depth_coord`) and each reference source. A yaw rotation of the scene by Δ about the receiver's vertical axis maps the panorama column `c` to `c+k` (k = Δ·W/2π, W=512) and rotates every 3-vector by `Rz(Δ)`: `depth_coord' = Rz(Δ)·roll(depth_coord, k)`, `src' = Rz(Δ)·src`, `ref_locs' = Rz(Δ)·ref_locs`. Reference RIRs and the target are unchanged. Distances are preserved, so `shift_and_align` (direct-path delay/gain) is exactly invariant; only the ViT branch and the sinusoidal coordinate embedding (`dist_embedder` on the raw xyz) can change.
- **SimpleViT** has an absolute 2-D sin/cos positional embedding: rolled patches get different position codes, so its tokens are *not* equivariant (measured earlier: max token change 2.0 under a one-patch roll).
- **CylindricalViT** is azimuth-equivariant *at the token level* for patch-aligned rolls (k multiple of 32; measured token error 2e-6). But xRIR pools tokens with `lin_proj_0`, a *learned linear map over the 256 token positions* (16 elevation × 16 azimuth), which is not permutation-invariant, and the coordinate embedding of `src`/`ref_locs` is not rotation-invariant either. So **neither full model is yaw-invariant by construction**; the experiment measures how much each one *actually* changes.
- Training used a fixed canonical frame (`rotation = 0`, panoramas axis-aligned with the room), so rotated inputs are also out-of-distribution in wall orientation. This is part of the phenomenon being measured (a deployed system sees arbitrary headings), not a confound to remove; it is stated so the result is not over-attributed to positional encodings alone.

## 3. Hypotheses (stated before evidence)

- **H1 (user's claim):** for the SimpleViT models (released checkpoint and the exp_01 control), joint yaw rotation degrades quality *substantially* at some angle.
- **H2:** the CylindricalViT model degrades less than SimpleViT at every angle, and is nearly unchanged at patch-aligned angles (k ∈ 32ℤ), with residual change coming only from `lin_proj_0` and the coordinate embedding.
- **H0:** the pipeline relies on the panorama weakly (its output is a re-weighting of reference log-spectrograms), so rotation changes predictions little for either backbone.

**Pre-registered definition of "substantial":** at some tested angle, the paired relative increase over Δ=0 is ≥ 10 % in EDT error *or* C50 error with the 95 % bootstrap CI excluding 0; secondary: test-loss increase ≥ 5 %. For scale, the exp_01 backbone gap is 3.7 % EDT; the adjacent-epoch noise floor is ≈2 % EDT.

## 4. Design

- **Models (3):** released `checkpoints/xRIR_unseen.pth` (SimpleViT, authors' training); `ckpt/xRIR_simple_8_shot/epoch_12.pth` (SimpleViT control, exp_01); `ckpt/xRIR_cyl_8_shot/epoch_12.pth` (CylindricalViT, exp_01). No training.
- **Data:** full AcousticRooms unseen test split (6337 queries, K=8), deterministic references (seed 0) identical across models and angles → every comparison is paired.
- **Angles:** integer column rolls only (no interpolation): k ∈ {0, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 448, 480, 496, 504, 508} = {0°, 2.8°, 5.6°, 11.25°, 22.5°, 45°, 67.5°, 90°, 135°, 180°, 225°, 270°, 315°, 337.5°, 348.8°, 354.4°, 357.2°}; patch-aligned set = multiples of 32.
- **Per (model, angle, sample) — cheap "spectral" metrics** (no Griffin-Lim): test loss vs ground truth (STFT L1 + decay), log-STFT MSE vs GT, and **consistency** = mean |log-spec(Δ) − log-spec(0)|. All 17 angles.
- **Per (model, angle, sample) — full acoustic metrics** (Griffin-Lim, EDT / C50 / T60 as in `eval_unseen.py`): angles {0°, 5.6°, 22.5°, 45°, 90°, 180°} (k = 0, 8, 32, 64, 128, 256) — the same code path as exp_01 so numbers are comparable.
- **Statistics:** paired difference vs Δ=0 per model/angle/metric with 10k-bootstrap 95 % CI and relative change; verdict per model against the pre-registered threshold; cross-model comparison of degradation (cyl minus simple) at each angle, paired.
- **Controls / sanity:** (a) k=0 must reproduce exp_01 epoch-12 per-sample numbers exactly (same eval path, same seeds) — parity check; (b) k=512 ≡ k=0 (assert); (c) shift-and-align outputs at k≠0 equal those at k=0 to float tolerance (assert in a test); (d) CylindricalViT *token* equivariance at k=32 re-asserted in a test.

## 5. Planned code

New, no baseline file touched.

| File | Purpose |
|---|---|
| `tools/yaw_rotation.py` (new) | pure functions: `yaw_angle_rad(k, W=512)`; `rotate_vectors_z(v, angle)` for `[..., 3]`; `rotate_scene_yaw(depth_coord, src_loc, ref_locs, k, W=512)` → rotated `(depth_coord', src', ref_locs')` for batched tensors (`[B,3,H,W]`, `[B,3]`, `[B,K,3]`) |
| `eval_yaw_rotation.py` (new) | one pass over the split per model; for each batch loops over `--yaw-cols`; computes spectral metrics for all angles and full acoustic metrics for `--full-metric-cols`; writes `per_sample_yaw.json` (angle → per-sample arrays) and `metrics_yaw.json`; `--max-samples` for smoke only |
| `tools/summarize_yaw.py` (new) | tables (per model × angle: means, paired diff vs 0, CI, rel %), verdicts, cross-model degradation; emits a JSON data extract for the results page |
| `tests/test_yaw_rotation.py` (new, TDD) | see §6 |
| `tests/test_summarize_yaw.py` (new, TDD) | paired-stat helper on synthetic arrays |

Reuse: `eval_xRIR_backbone.load_model_state`, `eval_unseen.Evaluator/griffin_lim`, `treble_multi_room_dataset.xRIR_Dataset`, `tools/compare_eval` bootstrap logic (factor the paired-CI helper into a small importable function if the Coder finds duplication).

## 6. Per-function test list (write each test first; red → green)

`tools/yaw_rotation.py`
1. `yaw_angle_rad(0)==0`, `yaw_angle_rad(512)==2π`, `yaw_angle_rad(128)==π/2`.
2. `rotate_vectors_z` preserves norms and z; rotating by π/2 maps (1,0,0)→(0,1,0) (right-handed, counter-clockwise; the sign convention must match `convert_equirect_to_camera_coord`, where column θ=(c+0.5)·2π/W−π and x=d·cosφ·cosθ, y=d·cosφ·sinθ).
3. `rotate_scene_yaw` with k=0 is the identity; k=W equals k=0 (float tolerance).
4. **Convention test:** build a synthetic `depth_coord` from a random depth panorama via `convert_equirect_to_camera_coord`; after `rotate_scene_yaw(k)`, the vector at column `c` must point along θ_c (its xy direction matches (cosθ_c, sinθ_c)) and its norm equals the original depth at column `c−k`.
5. Composition: rotate by k1 then k2 equals rotate by k1+k2 (mod W).
6. Direct-path invariance: `xRIR.shift_and_align(refs, src', ref_locs')` equals `shift_and_align(refs, src, ref_locs)` (delays are rounded, so require exact equality of the integer delay tensor and allclose on outputs).
7. CylindricalViT token equivariance: tokens of the rotated panorama at k=32 equal the k=0 tokens rolled by one patch along azimuth (atol 1e-4); SimpleViT tokens do not (sanity, not a correctness requirement).

`tools/summarize_yaw.py`
8. `paired_stats(a, b)` returns diff mean, CI containing 0 for identical arrays, negative CI for `b = a + const`.
9. `verdict(rel_increase, ci_lo, threshold)` implements the pre-registered rule.

`eval_yaw_rotation.py` — no unit test (integration); validated by the ladder below.

## 7. Validation ladder

1. Static: `py_compile`, `bash -n`, `git diff --check`.
2. Tests: `pytest tests/test_yaw_rotation.py tests/test_summarize_yaw.py`.
3. Tiny synthetic forward: `rotate_scene_yaw` on random tensors + one `xRIR` forward at k=0 and k=32 on GPU (shapes, finiteness).
4. Small real-data readback: 4 real test samples; assert k=0 per-sample loss equals exp_01 epoch-12 values for those indices.
5. Smoke: `eval_yaw_rotation.py --max-samples 8` with all angles, full metrics for 2 angles.
6. Fit/batch probe: forward-only batch 16 × 17 angles fits beside the sim-to-real jobs (expected ≈3 GB).
7. Full run: 3 models × full split; parity check (k=0 vs exp_01 epoch-12 files) before reading any other angle.

## 8. Parity audit (vs exp_01 evaluation path)

Same dataset object, `seed 0`, `num_workers` count and worker seeding → identical (query, reference) pairs; same `Evaluator`, same Griffin-Lim parameters, same 8000-sample metric window; same `compute_loss`. Verified by rung 4/7 equality at k=0.

## 9. Runs, cost, acceptance

- 3 runs (one per model), each: 6337 samples × 17 forward angles at batch 16 (≈20 min) + Griffin-Lim metrics for 6 angles (≈95 min) ≈ 2 h; two GPUs → ≈3.5 h wall-clock. Launch after the exp_02 queues finish (shared-resource etiquette).
- Acceptance: every run completes all angles with no NaN; k=0 reproduces exp_01 epoch-12 per-sample metrics to 1e-6 (released checkpoint: reproduces its exp_01 baseline numbers within Griffin-Lim nondeterminism); per-sample files contain 6337 entries per angle.
- Outputs: `ckpt/yaw_rotation/<model>/{per_sample_yaw.json, metrics_yaw.json}`, `ckpt/yaw_rotation/summary.txt`, results page `yaw_rotation_degradation_01_results.html` with degradation-vs-angle charts.

## 10. Decisions requested from the user at approval

1. **Tests folder location** (first tests in this repo): proposal `tests/` at the repo root.
2. **Base commit:** the working tree holds uncommitted exp_01/exp_02 code and the retrospective worklogs. Proposal: one commit "exp_01/exp_02: backbone comparison and sim-to-real pipeline + retrospective SOP records" as the known-good base before exp_03's first TDD commit.
3. Angle set and the 10 % / 5 % "substantial" thresholds above.
4. Whether to also include the exp_02 fine-tuned real-room models (HAA rooms) as a second data domain — proposed **no** for this experiment (keep one clean domain); can be a follow-up.
