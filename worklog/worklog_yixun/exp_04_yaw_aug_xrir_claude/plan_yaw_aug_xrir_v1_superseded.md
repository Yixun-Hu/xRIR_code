# Plan — yaw_aug_xrir (exp_04)

**Status:** v1, written 2026-09-12 by the Planner. Awaiting the Codex plan review and the user's approval. No code has been written.

## 1. Question

Does training-time random yaw augmentation — one physically consistent yaw per training sample, exactly as the user's FLAC exp_17 arm (`/home/yixunhu/codespace/exp-17-yawaug-a6000`, `training.yaw_aug = {enabled, img_w: 512, seed}`) — change xRIR's few-shot RIR prediction on AcousticRooms (a) at the canonical heading and (b) under scene yaw, relative to the same-budget un-augmented SimpleViT control of exp_01? The deliverable is the \YawAugxRIR{} row of the user's table (K = 1 and K = 8: T60, C50, EDT on the unseen split) plus the yaw-sensitivity contrast measured with exp_03's protocol.

## 2. Treatment and single-delta contract (mirrors exp_17 §2)

- **Augmentation semantics (port of FLAC exp_15/17, `src/data/yaw_rotation.py` + `src/training/diffusion.py`):** for every training sample, draw one integer panorama-column offset `d ~ Uniform{0, …, W−1}` (W = 512; every sample is rotated, no skip probability), convert to the exact angle `Δ = 2πd/W`, and apply the active yaw to the whole scene: `depth_coord' = Rz(Δ)·roll(depth_coord, d, dim=W)`, `src_loc' = Rz(Δ)·src_loc`, `ref_locs' = Rz(Δ)·ref_locs`. The target RIR and the reference RIRs are untouched (a rigid yaw of the whole scene leaves every impulse response unchanged; the `shift_and_align` step is a function of distances, which are invariant). This is exactly `tools/yaw_rotation.rotate_scene_yaw` (exp_03, reviewed), applied per sample with its own `d`.
- **Seeding (port of `_yaw_aug_step_seed`):** counter-based, no stream state: the per-batch `torch.Generator` seed is a keyed 32-bit bijection of `(epoch, batch_idx)` under the run's `--yaw-aug-seed` (splitmix64 key, fmix32 mix, packed counters — the same construction as FLAC, single rank so the rank field is 0), so a resumed run reproduces the draws of an uninterrupted one and the global RNG (dropout-free model, but the loader's worker seeding) is never consumed by the augmentation. Draw from a dedicated CPU generator only.
- **Where:** `train_xRIR_backbone.py::compute_loss` (training step only); `test_epoch` and every evaluator remain un-augmented. A one-line banner (`yaw_aug ENABLED W=512 seed=…`) is printed before the first step; its absence in the log fails the launch gate.
- **Single delta:** the new flags default to off (`--yaw-aug 0`); with `--yaw-aug 0` the training path is behaviourally identical to the exp_01 script (parity test: same seed, same first batch → bit-identical loss and gradients). Everything else — backbone `simple`, `--batch-size 32 --accum-steps 2 --tf32`, 12 epochs, `--decay-epochs 3 --lr-gamma 0.1`, lr 1e-3, weight decay 1e-4, seed 0, 12 workers, per-epoch checkpoints — is the exp_01 control's recipe (`ckpt/xRIR_simple_8_shot/args.json`), so the comparison is control (exp_01) vs control + augmentation.

## 3. Hypotheses and pre-registered decision rules

Statistics as in exp_03 (`tools/paired_stats.py`, `tools/summarize_yaw.py`): per-query paired bootstrap, 20 000 resamples, the ratio or difference recomputed inside each resample, query-level intervals primary (this fixed split) and room-cluster intervals (17 rooms) secondary, Bonferroni within each family, T60 descriptive.

- **H1 (accuracy at the canonical heading, K = 8, unseen, epoch-matched):** the augmented model is *non-inferior* to the control on EDT and C50 error: the adjusted upper bound of the relative difference (aug − control)/control is below **+3 %** for both metrics (two one-sided tests, family m = 2, α = 0.05). Verdicts: "non-inferior" (both), "non-inferior on {metric}" (one), "not shown" (neither); a *superior* result (adjusted upper bound below 0) is reported as such. The same table is produced at K = 1 (secondary, same rule, separate family).
- **H2 (yaw robustness, K = 8, condition P of exp_03):** the augmented model degrades less than the control under scene yaw at the angles where the control's degradation was largest in exp_03: C50 and EDT at k ∈ {32, 64, 448, 480} (±22.5°, ±45°). Rule: D_k = r_k(aug) − r_k(control) on shared resamples; a cell passes if the adjusted upper bound of D_k is below 0 (family m = 2 metrics × 4 angles = 8); H2 "supported" if every C50 cell passes, "partially supported" if some, "not supported" if none. D_k is reported at all ten acoustic angles and the 18 spectral angles. The augmented model's own r_k is also tested for near-invariance (TOST ±2 % at the patch-aligned angles, as in exp_03).
- **Descriptive:** the augmented model's absolute yaw sensitivity curve; the epoch-9 checkpoint (≈41.7k updates, the "40k" convention of exp_17) alongside epoch 12; the seen split at the epoch of best test loss.

## 4. Data, models, budget

- Training data: AcousticRooms training split as in exp_01 (296 352 samples/epoch, 9 261 batches of 32, accumulation 2 → 4 630 updates/epoch); local NVMe mirror `XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`.
- Budget: **12 epochs (55.6k updates), one RTX A6000, ≈43 h** at the control's 1.4 s/iteration plus the augmentation overhead (a roll and two small matmuls per sample, expected < 3 %; measured in the smoke). Per-epoch checkpoints `epoch_NN.pth`, `best.pth`, `last.pth` + `--resume`. Output `ckpt/xRIR_simple_yawaug_8_shot/`.
- Comparators (no retraining): `ckpt/xRIR_simple_8_shot/epoch_12.pth` (control) and `ckpt/xRIR_cyl_8_shot/epoch_12.pth` (cylindrical) — both already evaluated under the paired protocol at K = 8 (exp_03 sweeps) and K = 1 (2026-09-12 k = 0 runs, manifest `reference_manifest_k1.json`).
- Evaluation protocol (declared, per FLAC directive 05): `eval_yaw_rotation.py`, batch 16 canonical, TF32 off, `--gl-seed 0`, reference manifests seed 0 (K = 8: `47637a55…`; K = 1: `f6d71f86…`), no test-time augmentation, no rotation except the pre-registered grids; the K = 8 sweep (18 spectral + 10 acoustic P angles + 4 E angles, ≈2 h) and the K = 1 k = 0 run (≈6 min) for the augmented checkpoint; the same manifests as the comparators so every comparison is paired per query.

## 5. Code (Coder, TDD, one small commit per red→green cycle; Codex review per round)

| file | change | tests (`tests/test_yaw_aug.py`) |
|---|---|---|
| `tools/yaw_rotation.py` | `rotate_scene_yaw_batch(depth_coord, src, ref_locs, ks, W)` — per-sample integer offsets (`ks: LongTensor [B]`), exact roll via gather and `Rz` per sample; returns the three rotated tensors | equals `rotate_scene_yaw` applied per sample for random ks; `ks = 0` is the identity (bit-exact); `ks = W` ≡ 0; the direct-path distances (norms) of src and refs are preserved to 1e-6; dtype/device preserved |
| `tools/yaw_aug.py` (new) | `splitmix64`, `fmix32`, `step_seed(seed, epoch, batch_idx)` (keyed 32-bit bijection, injective on epoch < 2^8, batch_idx < 2^20 — asserted), `draw_offsets(n, W, generator)` (dedicated generator, `randint(0, W)`), `apply_yaw_aug(batch_tensors, seed, epoch, batch_idx, W)` returning rotated `(depth_coord, src_loc, ref_locs)` and the offsets | distinct (epoch, batch) → distinct seeds on a sampled domain and a full small domain; determinism (same inputs → same offsets); the global torch RNG state is bit-identical before/after a draw; offsets uniform-ish over a large draw (chi-square sanity) and all in [0, W); `apply_yaw_aug` leaves `tgt_wav`/`ref_irs` untouched and equals `rotate_scene_yaw_batch` with the drawn offsets |
| `train_xRIR_backbone.py` | flags `--yaw-aug {0,1}` (default 0), `--yaw-aug-seed` (default: `--seed`), `--yaw-aug-width 512`; `compute_loss(model, batch, aug=None)` applies `apply_yaw_aug` when `aug` is given (training only); `train_epoch` passes `(epoch, batch_idx)`; banner before the first step; `args.json` records the flags | parity: with `--yaw-aug 0` the loss on a fixed synthetic batch is bit-identical to the current function (guarded by a saved reference value); with `--yaw-aug 1` and forced offsets `ks = 0` it equals the un-augmented loss; the banner text appears exactly once in a smoke log |
| `tools/paired_compare.py` (new) | paired comparison of two k = 0 runs from their `per_sample_yaw.json` (any two labels): per-metric means, difference and relative difference with query-level and room-cluster bootstrap intervals (adjusted α given a family size), non-inferiority verdict against a margin; JSON + text output with sha256 binding; refuses runs whose query lists differ | synthetic per-sample inputs with a known shift; margin rule; refusal on mismatched queries; reuses `tools/paired_stats` |
| `tools/summarize_yaw.py` | no change; `--mode exploratory --runs sweep_control sweep_yawaug --labels control yawaug` gives r_k per run and D_k needs the H2 family — add `--h2-family` and role names? **Decision:** keep the summariser untouched and compute H2's D_k in `tools/paired_compare.py --angles …` from the two sweeps' per-sample arrays (same resamples, adjusted family 8) | D_k on synthetic sweeps with a known effect |

Estimated size: ≈350 lines of code + tests; 4 Coder rounds.

## 6. Validation ladder (each rung logged)

1. static (`py_compile`, `bash -n`, `git diff --check`); 2. `pytest tests/` (existing 237 + new); 3. tiny synthetic forward: `compute_loss` with `--yaw-aug 1` on random tensors, finite loss, gradients flow; 4. real-data readback: one real batch rotated by drawn offsets — depth roll exact (column d of the rotated map equals column 0 of the original after `Rz`), source distance unchanged, target unchanged; 5. smoke: `--epochs 1 --max-train-batches 3 --max-test-batches 2 --batch-size 4 --save-every 0 --yaw-aug 1` → banner, three finite steps; 6. throughput probe: 50 batches at batch 32 × 2 with and without augmentation → overhead < 5 %; 7. parity audit: `--yaw-aug 0` loss on the first real batch equals the exp_01 script's loss (same seed) bit for bit; only then the full run.

## 7. Launch, acceptance criteria, monitoring

Written in `_worklog.md` before launch: commit SHA; GPU index; batch 32 × accumulation 2; 12 epochs; seed 0; `--yaw-aug 1 --yaw-aug-seed 0 --yaw-aug-width 512`; the banner in the log; ≥ 1 optimizer step with finite loss; first-epoch time ≤ 4.0 h; `history.jsonl` grows one line per epoch; `epoch_NN.pth` for every epoch. Launched with `nohup`, timestamped log in the record folder, `_command.md` entry at launch, `_params_set_up.md` at launch. Failure triage infrastructure vs bug; resume from `last.pth`.

## 8. Analysis and deliverables

`_results.md` (generated from the canonical JSON of `tools/paired_compare.py` and the exp_03 summariser's exploratory output — tables only), `_analysis.md` (Planner), `yaw_aug_xrir_01_results.html` (+ assets; page reads producer JSON only, as exp_03's), the table row values with their intervals, `commits_yaw_aug_xrir.md`. Provenance: the exp_03 binder/acceptance pattern is reused for the new runs (sidecars with closure digest, environment, data identity), scoped to the new run directories.

## 9. Cost and risks

- Cost: ≈43 h training on one GPU (the other GPU stays free for the K = 1 and K = 8 evaluations of the comparators and for the user); evaluations ≈2.5 h; summaries minutes.
- Risks: (i) augmentation may raise k = 0 error on the axis-aligned test rooms (that is a finding, covered by H1's margin); (ii) single training seed for both arms (as exp_01/exp_17) — run-to-run training noise is not estimated; stated as a residual; (iii) the position pool `lin_proj_0` and the absolute-coordinate embedding now see all headings — the expected mechanism, but 12 epochs may be insufficient to converge under the harder distribution (the epoch curve is reported); (iv) throughput overhead (bounded by the probe).

## 10. Decisions requested from the user

1. Budget: 12 epochs to match exp_01's control (recommended; the epoch-9 checkpoint gives the "40k" convention), or stop at epoch 9 (≈41.7k updates, ≈32 h).
2. Only the SimpleViT backbone is augmented (the table has no YawAug-CylindricalViT row); a cylindrical + augmentation arm would cost another ≈45 h and is not planned.
3. Tests go in `tests/` (as exp_03).
4. R@1/5/10 stay undefined in this pipeline (no retrieval task exists); the row keeps "--" there.
