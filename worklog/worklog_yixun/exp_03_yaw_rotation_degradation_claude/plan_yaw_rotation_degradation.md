# Plan — yaw_rotation_degradation

**Status:** v2 (revised after the Codex plan review of v1, `yaw_rotation_degradation_codex_plan_review.md`). Awaiting user approval. No code has been written. v1 is kept as `plan_yaw_rotation_degradation_v1_superseded.md`.

## 1. Question

Does **jointly rotating the receiver-centred depth panorama and the source / reference-source coordinates in yaw** (an *active* rotation of the whole scene about the receiver's vertical axis by +Δ; a symmetry of single-channel acoustics with omnidirectional source and monaural receiver, so the true RIR is unchanged) substantially degrade RIR synthesis quality in the xRIR pipeline?

## 2. Mechanism (what can and cannot change)

- Geometry enters `xRIR.forward` only through the ViT views `(loc − depth_coord)/5` (query source, receiver `−depth_coord`, each reference source) and the sinusoidal embedding of the raw xyz of `src_loc` / `ref_ir_locs`. Under the active yaw +Δ with k = ΔW/2π integer columns: `depth_coord' = Rz(Δ)·roll(depth_coord, k, axis=W)`, `src' = Rz(Δ)·src`, `ref_locs' = Rz(Δ)·ref_locs`; reference RIRs and the target are unchanged. Convention: column θ_c = (c+0.5)·2π/W − π with x = d cosφ cosθ, y = d cosφ sinθ (`convert_equirect_to_camera_coord`), so a +k roll followed by Rz(+2πk/W) maps the ray of column c−k onto column c (confirmed by the reviewer). The dataset's unused `rotation ∈ {90,180,270}` branches rotate about y and are **not** used.
- `shift_and_align` is invariant algebraically but **not numerically**: float32 norms and `round()` to integer samples can flip a delay by one sample under rotation (reviewer's example: delay 80.5 → 80 unrotated, 80.500015 → 81 at k=4). Therefore the **primary** condition holds the direct-path alignment at the k=0 coordinates (cached aligned reference audio), so only the geometry branches see the rotation; a **secondary end-to-end** condition lets alignment use the rotated coordinates (what the pipeline would do at deployment), and the number of flipped integer delays per angle is audited over all real (query, reference) pairs.
- Neither full model is yaw-invariant by construction: SimpleViT has absolute 2-D positional codes and an unconstrained flattened-xyz patch projection; CylindricalViT's tokens are equivariant for patch-aligned rolls, but xRIR pools tokens with `lin_proj_0` (a learned linear map over the 256 token *positions*) and embeds raw xyz. The experiment measures actual sensitivity and decomposes it (token-level vs pooled-level change for the cylindrical model).
- Training used a fixed canonical heading, so rotated inputs are also out-of-distribution in wall orientation; this is part of the deployment-relevant phenomenon, stated so the effect is not over-attributed to positional codes.

## 3. Hypotheses and pre-registered decision rules

Primary SimpleViT model: the exp_01 **same-budget control** (`ckpt/xRIR_simple_8_shot/epoch_12.pth`, training-matched to the cylindrical model). The released checkpoint is a **replication** model. Cylindrical: `ckpt/xRIR_cyl_8_shot/epoch_12.pth`. Conclusions are scoped to these epoch-12, single-seed checkpoints.

- **H1 (user's claim, confirmatory):** joint yaw rotation substantially degrades the primary SimpleViT model. **Rule:** define the relative degradation r_k = (ē_k − ē_0)/ē_0 for EDT error and C50 error, computed *inside* each paired bootstrap resample. H1 is supported if, for at least one angle in the acoustic angle set, the **Bonferroni-adjusted** (family = 2 metrics × 9 non-zero acoustic angles = 18 tests, i.e. 1 − 0.05/18 two-sided percentile intervals) lower bound of r_k exceeds **+10 %**. Verdict wording: "supported" (primary passes) / "supported and replicated" (released also passes) / "supported for the same-budget model only" or "replication only" (they disagree; both reported, no pooling) / "not supported". Absolute equivalents of the 10 % margin at k=0 are reported (≈0.005 s EDT, ≈0.12 dB C50 for the control).
- **H2 (cross-model, confirmatory for the difference):** the cylindrical model degrades less than the primary SimpleViT. **Rule:** difference-in-differences D_k = r_k(cyl) − r_k(control) per angle and metric, paired per sample, Bonferroni-adjusted over the same family; H2 supported if D_k < 0 with the adjusted upper bound below 0 at the angles where H1 holds (and reported at all angles). **Patch-aligned near-invariance** (k ∈ 32ℤ): equivalence test with margin ±2 % (two one-sided tests on r_k at the adjusted level); "nearly unchanged" only if the interval lies within ±2 %.
- **H0:** neither model degrades beyond the margins.
- **Inference target (reviewer Q2):** primary inference is about **this fixed published split** (6337 queries, 17 rooms) via query-level paired bootstrap; a **room-cluster bootstrap** (resample the 17 rooms with replacement, keeping all queries of a drawn room, paired) is reported as a secondary, population-of-rooms robustness check and is expected to be wide.
- **Invalid metrics (pre-registered mask):** per model and metric, a sample enters the paired analysis only if the metric is finite at k=0 *and* at the compared angle; the count and fraction of samples that become invalid at each angle are reported as a separate degradation indicator. T60 is reported descriptively only (exp_01 showed it is the noisiest metric; it is not in the confirmatory family).

## 4. Design

- **Data:** full AcousticRooms unseen test split (published config), K=8. **Reference manifest:** references are chosen deterministically per query from a sha256 hash of (manifest seed 0, query path), independent of batch size, worker count and model; the manifest (query path + 8 reference paths) is written once (`ckpt/yaw_rotation/reference_manifest.json`) and every run asserts its hash. Consequently k=0 is *not* per-sample identical to exp_01 (which used worker-local random draws that were never recorded); parity with exp_01 is checked distributionally (§8).
- **Angles (integer column rolls, W=512):** spectral grid k ∈ {0, ±4, ±8, ±16, ±32, ±64, ±96, ±128, ±192, 256} (19 angles: 0°, ±2.8°, ±5.6°, ±11.25°, ±22.5°, ±45°, ±67.5°, ±90°, ±135°, 180°; negative k ≡ W−k). **Acoustic grid** (Griffin-Lim metrics): k ∈ {0, ±8, ±32, ±64, ±128, 256} (10 angles).
- **Conditions:** (P) primary, alignment fixed at k=0; (E) end-to-end, alignment from rotated coordinates — spectral metrics only for E at all angles, acoustic metrics for E at {±32, ±128} to bound its effect.
- **Metrics per (model, condition, angle, sample):** spectral — test loss (STFT L1 + 0.01·decay, per-sample reduction by calling the existing loss functions on 1-sample slices so numerics equal the batch-1 path), log-STFT MSE vs GT, consistency = mean |log-spec(k) − log-spec(0)|; acoustic (acoustic grid only) — EDT, C50, T60 via `Evaluator` on the first 8000 samples exactly as exp_01, with **Griffin-Lim phase initialisation seeded per sample** (`torch.manual_seed(hash(manifest seed, query path))` immediately before each inversion) so k=0 and k≠0 use identical random phases for the same sample and results are angle-order- and model-order-invariant.
- **Decomposition (cylindrical only, diagnostic):** at k=32, report the relative change of (a) ViT tokens after azimuth roll-back (expected ~0), (b) the pooled `lin_proj_0` output, (c) the coordinate-embedding features, (d) the final log-spectrogram — a trained `lin_proj_0` azimuth-shift audit.
- **Delay-flip audit:** for every (query, reference) in the manifest and every angle, count integer-delay changes between rotated and unrotated coordinates (no model needed).

## 5. Planned code (new files only; no baseline file is modified)

| File | Purpose |
|---|---|
| `tools/yaw_rotation.py` | `yaw_angle_rad(k, W)`, `rotate_vectors_z(v, angle)`, `rotate_scene_yaw(depth_coord, src_loc, ref_locs, k, W)`; `fixed_alignment(model, aligned_refs)` context manager that makes `model.shift_and_align` return cached aligned audio (restores the original method on exit); `integer_delays(src, ref_locs)` reproducing `shift_and_align`'s rounding for the audit |
| `tools/reference_manifest.py` | `build_manifest(dataset, seed)` (deterministic per-query reference paths), `manifest_hash`, `ManifestDataset` wrapper returning the exp_01 tuple plus the query path; loads audio/depth via the existing dataset helpers |
| `tools/paired_stats.py` | `relative_degradation_bootstrap(e0, ek, n_boot, alpha, rng)` (r_k inside each resample), `bonferroni_alpha(m)`, `diff_in_diff_bootstrap`, `equivalence_tost`, `cluster_bootstrap(rooms, ...)`, `valid_mask(e0, ek)`; refactors the bootstrap now duplicated in `tools/compare_eval.py` (that file is left as is for exp_01 provenance) |
| `tools/per_sample_metrics.py` | `per_sample_losses(out_spec, tgt_spec)` (1-sample slices through the original loss functions), `griffin_lim_seeded(mag, seed)`, `acoustic_metrics(pred_ir, gt_ir, evaluator, want_t60)` returning NaN for invalid |
| `eval_yaw_rotation.py` | one pass per model over the manifest at batch 16: cache k=0 aligned refs per batch; loop angles; compute spectral metrics for P and E, acoustic metrics on the acoustic grid; write `ckpt/yaw_rotation/<model>/per_sample_yaw.json` + `metrics_yaw.json`; `--max-samples` for smoke only; `--batch-size 1` supported for the equality test |
| `tools/summarize_yaw.py` | tables per model × angle (means, r_k with adjusted CI, verdicts), H1/H2/equivalence verdicts, cluster-bootstrap check, decomposition and delay-flip tables, JSON extract for the results page |
| `tests/test_yaw_rotation.py`, `tests/test_reference_manifest.py`, `tests/test_paired_stats.py`, `tests/test_per_sample_metrics.py`, `tests/test_eval_yaw_rotation.py` | TDD, §6 |

## 6. Per-function test list (test first; red → green; one small commit per cycle)

`tools/yaw_rotation.py`
1. `yaw_angle_rad`: k=0 → 0, k=W → 2π, k=128 → π/2.
2. `rotate_vectors_z`: preserves norms and z; +π/2 maps (1,0,0) → (0,1,0) (active, right-handed).
3. `rotate_scene_yaw`: k=0 identity; k=W equals k=0 (atol 1e-6); composition k1 then k2 == k1+k2 mod W.
4. **Oracle:** for a random depth panorama, `Rz(Δ)·roll(convert(depth), k) == convert(roll(depth, k))` for all channels incl. z, checked at every column including the seam (c=0, W−1).
5. **Covariance of all three views:** `(src' − depth')`, `(−depth')`, `(ref'_i − depth')` equal `Rz(Δ)·roll(·)` of the unrotated views.
6. `integer_delays` equals the delays computed inside `shift_and_align` (monkeypatch-free re-implementation checked against the model on real coordinates); `fixed_alignment` returns the cached audio inside the context and restores `shift_and_align` after (also on exception); with k=0 cached audio the forward equals the plain forward exactly.
7. CylindricalViT tokens reshaped to `[B,16,16,C]` and rolled by one **azimuth** patch at k=32 equal the k=0 tokens (atol 1e-4); SimpleViT tokens do not (sanity).
8. **Full-model k=W vs k=0** equality under fixed alignment and seeded Griffin-Lim (acoustic metrics identical).

`tools/reference_manifest.py`
9. Same seed → identical manifest across two dataset instances and across `num_workers ∈ {0, 4}`; different seed → different; hash changes if one reference changes; references exclude the query's own source and come from the query's receiver.

`tools/paired_stats.py`
10. `relative_degradation_bootstrap`: ek = e0 → interval contains 0; ek = 1.2·e0 → point estimate 0.20 and the lower bound > 0.10 for n large; `bonferroni_alpha(18) == 0.05/18`; `equivalence_tost` passes for |r| < margin with tight data and fails otherwise; `valid_mask` drops NaN in either array and reports counts; `cluster_bootstrap` with one cluster per sample equals the query-level bootstrap in expectation (loose tolerance) and returns wider intervals when clusters are correlated (synthetic).

`tools/per_sample_metrics.py`
11. `per_sample_losses` on a batch of 4 equals the batch-1 loss values sample by sample (atol 1e-7).
12. `griffin_lim_seeded` is deterministic for the same seed and differs for a different seed; identical magnitudes → identical waveforms.
13. `acoustic_metrics` returns NaN (not an exception) for a degenerate IR, and `want_t60=False` skips T60.

`eval_yaw_rotation.py`
14. Batch-1 vs batch-16 equality of every per-sample metric on materialised inputs for 8 real queries (atol 1e-6 spectral, exact for acoustic given the seeded phases).
15. Angle-order invariance: evaluating angles [0, 32] vs [32, 0] gives identical per-sample outputs.
16. Result indexing: per-sample arrays keyed by query path match the manifest order and length.

## 7. Validation ladder

1. Static: `py_compile`, `bash -n`, `git diff --check`. 2. `pytest tests/`. 3. Tiny synthetic forward (rotate random tensors; one `xRIR` forward at k=0/32 on GPU). 4. Small real-data readback (8 manifest queries; shapes, dtypes, min/max/std of depth and audio vs the dataset). 5. Smoke `eval_yaw_rotation.py --max-samples 8` with all angles and acoustic grid. 6. Batch probe (batch 16 × 19 angles forward-only beside the running jobs; expected ≈3 GB). 7. **k=0 parity gate on the full split, before any rotated angle is read** (§8). 8. Full runs.

## 8. Parity audit (before the full run)

- **Numeric recipe:** same `Evaluator`, Griffin-Lim parameters (n_fft 124, win 62, hop 31, 32 iters), 8000-sample metric window, loss functions and `compute_loss` semantics as exp_01; per-sample losses verified against batch-1 (test 11/14).
- **Data parity:** the manifest's query set equals exp_01's per-sample index set (same 6337 paths); one query's depth/audio tensors byte-identical to the dataset's.
- **The k=0 gate doubles as the properly paired re-evaluation of exp_01** (the exp_01 record was corrected on 2026-09-05: its references were not identical across models and its Griffin-Lim phases were unseeded). At k=0 the three models are evaluated on the hashed manifest with seeded phases; `tools/summarize_yaw.py` reports the cyl-vs-control paired comparison at k=0 with query-level and room-cluster CIs as a stand-alone table, which supersedes exp_01's epoch-12 comparison.
- **k=0 gate (distributional, because references differ):** for each model, k=0 means of EDT / C50 / T60 / loss must lie within the exp_01 epoch-12 values ± the reference-draw noise measured by re-evaluating exp_01's epoch-12 checkpoint on two extra manifest seeds (expected ≈1–2 % for EDT); the released checkpoint must match its exp_01 baseline reproduction the same way. Failure blocks the sweep.

## 9. Runs, cost, acceptance

- 3 models × [19 spectral angles × 2 conditions forward-only ≈ 25 min + acoustic: 10 angles (P) + 4 (E) × 6337 × ≈0.15 s ≈ 3.7 h] ≈ 4 h per model → ≈6 h wall-clock on two GPUs (CPU Griffin-Lim contention may add), after the parity gate (≈20 min per model) and after exp_02's queues finish.
- Acceptance: all angles complete, no NaN in spectral metrics, manifest hash matches in every output, k=0 gate passed and logged, per-sample arrays have 6337 entries per angle, delay-flip audit written.
- Outputs: `ckpt/yaw_rotation/{reference_manifest.json, <model>/per_sample_yaw.json, <model>/metrics_yaw.json, summary.txt}`; results page `yaw_rotation_degradation_01_results.html` (degradation vs angle with adjusted CIs, cross-model D_k, decomposition, delay-flip audit).

## 10. Decisions requested from the user at approval

1. **Tests folder:** `tests/` at the repo root (first tests in this repo).
2. **Base commit:** commit the uncommitted exp_01/exp_02 code and the retrospective records as one known-good base before exp_03's first TDD commit.
3. Angle grids, the 10 % / ±2 % margins and the Bonferroni family (18 tests) above; T60 descriptive only.
4. Inference target: fixed split primary, room-cluster bootstrap secondary.
5. Scope: AcousticRooms only (no HAA rooms in this experiment).

## 11. v2 changelog — responses to the Codex plan review (v1)

| Finding | Response in v2 |
|---|---|
| B1 `shift_and_align` not numerically invariant | Primary condition caches k=0 aligned audio via `fixed_alignment`; end-to-end condition kept as secondary; delay-flip audit over all real pairs and angles; tests 6, 8. |
| B2 parity/pairing at batch 16; unrecorded references; batch loss reduction | Deterministic reference manifest with hash, written and asserted; per-sample losses through 1-sample slices; batch-1 vs batch-16 equality test (14); k=0 parity gate run *before* the sweep, distributional because exp_01 references were never recorded (§8). |
| B3 Griffin-Lim randomness | Per-sample seeded phase init shared across angles/models; angle-order-invariance test (15); k=W==k=0 full-model test (8). |
| B4 statistics | r_k inside the bootstrap; Bonferroni family declared (18); lower bound > 10 % required for the confirmatory claim; fixed-split inference primary, room-cluster bootstrap secondary; pre-registered invalid-metric mask and invalid-fraction reporting; T60 descriptive. |
| B5 H1/H2 decidability, angle grid | Control = primary, released = replication with explicit disagreement wording; H2 as difference-in-differences with adjusted CI; equivalence margin ±2 % at patch-aligned angles; k=±96 (±67.5°) both included; signed acoustic grid (±8, ±32, ±64, ±128, 180). |
| B6 evaluator TDD | Evaluator factored into `reference_manifest`, `paired_stats`, `per_sample_metrics`; tests 9–16; per-round Codex reviews and a `full` review before launch. |
| Non-blocking | "Active +Δ yaw" naming; y-axis rotation branches excluded; SimpleViT patch-projection non-commutation noted; thresholds given absolute equivalents and labelled practical margins; cost restated (≈6 GPU-h, ≈6 h wall-clock); conclusions scoped to epoch-12 single-seed checkpoints. |
| Missing tests | Oracle incl. seam (4), three-view covariance (5), azimuth-only token roll on [B,16,16,C] (7), full-model k=W (8), manifest equality (9), batch equality (14), threshold/equivalence/cluster/invalid tests (10), `lin_proj_0` shift audit (§4 decomposition). |
| Q1 | 10 % is a confirmatory lower-bound threshold, not a point-estimate label. |
| Q2 | Fixed 17-room split (primary); population-of-rooms only via the secondary cluster bootstrap. |
| Q3 | Reported separately with the wording in §3; no pooling. |
| Q4 | Cached k=0 alignment; hashed manifest; per-sample seeded Griffin-Lim. |
| Q5 | Both added (k=±96; signed acoustic grid). |
