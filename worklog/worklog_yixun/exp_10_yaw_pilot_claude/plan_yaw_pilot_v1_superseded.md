# Plan — exp_10 `yaw_pilot` (xRIR analogue of FLAC exp_02 `yaw_noninvariance`)

**Planner:** Claude Fable 5.1 (session xrir-code-25). **Coder:** Claude Opus 5 (Agent, worktree `xRIR_code_wt10`, branch `exp10-yaw-pilot`). **Reviewer:** OpenAI Codex `gpt-6-astra` at xhigh. **Date:** 2026-09-26. **Status:** v1, for Codex plan review.

## 1. Question and stance

**Question (Yixun, 2026-09-26).** Mimicking FLAC's exp_02 pilot: when the *conditioning* of the released xRIR checkpoint is yaw-rotated on the AcousticRooms unseen split, how much does the **prediction itself move** (Metric 1, P_α vs P_0, no GT) compared with how much **accuracy vs GT degrades** (Metric 2)? FLAC's finding was "predictions move ~5× more than net accuracy loses" (T60 gap 3.3 % vs degradation 0.4–0.7 %; EDT gap 19–20 ms vs 3.5–6.3 ms). The pilot section needs the same two numbers, per angle, for xRIR.

**Stance.** This is a **descriptive pilot**, like FLAC's: means with paired query-level bootstrap intervals, no confirmatory hypotheses (exp_03 already ran the Bonferroni-adjusted H1/H2 tests on Metric 2 and found degradation small, peaking at diagonal headings). Pre-registered readouts: (R1) per angle and metric, degradation Δ(α) = mean error(α) − mean error(0) vs GT and shift G(α) = mean gap of P_α w.r.t. P_0, each with a 95 % query-level bootstrap CI; (R2) the ratio G/Δ (CI by the same resamples; reported as "undefined" when the Δ interval contains 0) — the FLAC-style statement is "shift ≫ degradation" only if the ratio's lower bound exceeds 2; (R3) the GL-free spectrogram shift beside the waveform shift, because xRIR's waveform is a Griffin-Lim inversion of a predicted magnitude and GL's nonlinearity inflates waveform distances (FLAC decodes waveforms directly, so its rel-L2 has no such stage) — the caption must say which quantity is FLAC-comparable.

## 2. Mechanism recap (from exp_03; unchanged)

Geometry enters `xRIR.forward` through the ViT views `(loc − depth_coord)/5` and the sinusoidal embedding of raw xyz. Active yaw by k columns (W = 512): `depth' = Rz(2πk/W)·roll(depth, k)`, `src' = Rz·src`, `ref_locs' = Rz·ref_locs`; audio and target unchanged (`tools/yaw_rotation.rotate_scene_yaw`, oracle-tested in exp_03). Condition **P** pins the direct-path alignment at k = 0 (`fixed_alignment`) so only the geometry branches see the rotation; exp_03 showed E (end-to-end) = P to 2e-5, so E is not run here. The prediction is a log-magnitude spectrogram; waveforms come from Griffin-Lim with the phase seeded per query (`tools/per_sample_metrics.griffin_lim_seeded`, `sample_seed(gl_seed, query)`), so P_α and P_0 of one query start from the same random phase and differ only through the magnitudes.

## 3. Arms, references, angles

| Arm | Checkpoint | K | Manifest (exp_03's, hashed) | Role |
|---|---|---|---|---|
| `released_k8` | `checkpoints/xRIR_unseen.pth` (authors', frozen; sha256 `1762c702…`) | 8 | `ckpt/yaw_rotation/reference_manifest.json` (`47637a55…`) | **primary** — the FLAC analogue (released checkpoint), xRIR's paper setting |
| `released_k1` | same | 1 | `ckpt/yaw_rotation/reference_manifest_k1.json` (`f6d71f86…`) | FLAC's K |
| `control_k8` | `ckpt/xRIR_simple_8_shot/epoch_12.pth` | 8 | `47637a55…` | same-budget SimpleViT (exp_01) — context |
| `cyl_k8` | `ckpt/xRIR_cyl_8_shot/epoch_12.pth` | 8 | `47637a55…` | CylindricalViT — the fix's starting point |

Execution order = table order (the headline lands first). Settings identical to exp_03's sweeps: batch 16 with canonical padding, TF32 off, `gl_seed 0`, metrics on the first 8000 samples, unseen split, 6 337 queries / 17 rooms. Angles **k ∈ {0, 64, 128, 256, 384}** = 0°, 45°, 90°, 180°, 270° (FLAC's three axis angles plus 45°, where exp_03 found the peak; all five are on exp_03's acoustic grid, so every Metric-2 number here is a replication of exp_03's). Controls: **k = 512** (a full turn; must equal k = 0) on the first 256 queries, and a **repeated k = 0** pass on the same 256 queries (run-to-run determinism) — FLAC's `rot0 == baseline` check.

## 4. Metrics (per query, then mean over queries; NaN = invalid, never imputed)

**Metric 2 — accuracy vs GT at angle α (condition P):** EDT error (s), C50 error (dB), T60 error (%) via `tools.per_sample_metrics.acoustic_metrics` (exp_03's validity rules, 8000-sample window); log-STFT MSE and the test loss via `per_sample_losses`. Degradation Δ(α) = mean(e_α − e_0) over queries valid at both angles.

**Metric 1 — prediction shift, P_α vs P_0, no GT:**
- `wave_rel_l2` = ‖w_α − w_0‖₂ / (‖w_0‖₂ + 1e-8) and `wave_mad` = mean|w_α − w_0| over the 9 600-sample waveforms (FLAC's definitions, `compare_predictions.waveform_gap`);
- `mag_rel_l2` = ‖M_α − M_0‖_F / ‖M_0‖_F on the predicted magnitude spectrograms (GL-free) and `logspec_mad` = mean|log M_α − log M_0| (= exp_03's `consistency`, replicated);
- acoustic gaps `edt_gap`, `c50_gap`, `t60_gap` = `acoustic_metrics(pred = w_α, gt = w_0)` (T60 gap as % of T60(w_0)), FLAC's "P_ref as GT" convention.

**Uncertainty:** paired query-level percentile bootstrap, 10 000 resamples, seed 0, 95 % two-sided, for Δ, G and G/Δ; convergence check with a second seed (bounds move < 10 % of the interval width). **Noise-floor band** for the figure (FLAC used exp_01's single-eval σ): exp_04's five-seed SDs of the same-budget SimpleViT at K = 8 (EDT 0.186 ms, C50 0.0053 dB, T60 0.009 %) and K = 1 (0.609 ms, 0.0133 dB, 0.101 %) — reference-draw + phase noise — drawn as 2σ; the released checkpoint has no five-seed set, stated in the caption. Room-cluster CIs are reported as a secondary robustness line (17 rooms).

## 5. Planned code (new files only — exp_03's pinned closure is not touched; the frozen `eval_yaw_rotation.py` refuses to run without CUDA, so the CPU/GPU-agnostic path lives in the new tool, which reuses `pad_batch`, `rotate_scene_yaw`, `fixed_alignment`, `per_sample_losses`, `griffin_lim_seeded`, `sample_seed`, `acoustic_metrics`, `ManifestDataset`, `load_checked_manifest`)

| File | Purpose |
|---|---|
| `tools/exp10_yaw_pilot.py` | `device_agnostic_apply_delay` (runtime replacement of `model.xRIR.apply_delay`, which hard-codes `.cuda()`; installed by `run` only, never edits the module); `angle_logspec(model, tensors, k, aligned0)` → log-magnitude `[n, F, T]` under `fixed_alignment`; `invert(logspec, keys, gl_seed)` → `[n, 9600]` float32 (9 579 GL samples zero-padded); `waveform_gap`, `spectrogram_gap`, `acoustic_gap`; `evaluate_batch(model, batch, ks, evaluator, gl_seed, batch_size, device)` → per-query metrics per k + waveforms; `run(args)`: hash-checked manifest, canonical order check, writes `ckpt/exp10/<arm>/per_sample.json`, `metrics.json`, `wav_k<k>.npy` (float32 `[N, 9600]`, ~243 MB each), `meta` (checkpoint sha256, manifest hash, gl_seed, device, torch, git commit, elapsed); CLI `--arm --backbone --checkpoint --manifest --manifest-hash --num-shot --ks --device --batch-size --max-samples --out-dir --controls` |
| `tools/exp10_compare.py` | FLAC-style offline comparator: Metric 1 from two stored `wav_k*.npy` (+ meta guard: same arm/manifest/gl_seed/device); `check_online(per_sample.json)` asserts equality with the online values (1e-6); `parity_exp03(per_sample.json, <exp_03 per_sample_yaw.json>)`: same query list, per-angle mean |Δ| and max per-sample |Δ| of EDT/C50/T60/consistency at the shared angles (tolerances in §7) |
| `tools/exp10_summarize.py` | tables (Markdown + canonical JSON with input sha256s): Metric 2 per angle (mean_0, mean_α, Δ, CI), Metric 1 per angle (each gap, CI), ratio G/Δ; controls (k = 512 vs 0, repeated 0: max |Δ| waveform/log-spec, must be 0 up to 1e-6); figure `yaw_pilot_gaps_<arm>.{pdf,png}` in FLAC's style (panels T60 / C50 / EDT; bars = accuracy degradation vs prediction shift per angle, 2σ band, values annotated) + a combined four-arm figure; CSV of plotted values |
| `tests/test_exp10_yaw_pilot.py`, `tests/test_exp10_compare.py`, `tests/test_exp10_summarize.py` | §6 |

## 6. Per-function tests (TDD; red → green; one small commit per cycle)

`tools/exp10_yaw_pilot.py`
1. `waveform_gap`: identity → (0, 0); `rel_l2(w, 2w) = 1`; zero reference → finite; shapes `[n, T]` → `[n]`.
2. `spectrogram_gap`: identity → 0; `logspec_mad` equals `mean|a − b|` on a random pair; `mag_rel_l2` scale check.
3. `acoustic_gap`: identity → EDT 0, C50 0, T60 0; `want_t60=False` → NaN T60; invalid (all-zero) input → NaN not exception.
4. `device_agnostic_apply_delay`: equals the module's own `apply_delay` on CUDA when available (skip otherwise) and shifts correctly for positive/negative/zero delays on CPU; installation is scoped (restored after `run`).
5. `angle_logspec`: on a random-init `build_xrir("simple", 2)` and a synthetic batch (CPU): k = 0 equals the plain forward exactly; k = 512 equals k = 0 (atol 1e-6); the pinned alignment is used (a counter on `shift_and_align` stays at 1).
6. `invert`: same `(logspec, key, gl_seed)` twice → identical; different key → different; output `[n, 9600]`, last 21 samples zero.
7. `evaluate_batch`: synthetic batch of 3 padded to 4 → results have `n = 3`; every k has both metric families; at k = 0 all gaps are exactly 0; padded rows do not change the kept rows (compare batch of 3 vs the same 3 inside a batch of 4).
8. `run` smoke (`--max-samples 4`, real manifest; skipped when the data root is absent): outputs exist with the right shapes; meta carries the checkpoint sha256 and manifest hash; a second run is bit-identical on CPU; refuses a wrong `--manifest-hash`.

`tools/exp10_compare.py`
9. Offline Metric 1 from stored arrays equals the online per-sample values (1e-6); meta guard refuses different `gl_seed` / manifest / arm.
10. `parity_exp03`: refuses a different query list; on a fixture with an injected per-sample offset reports the max |Δ| and the mean |Δ| exactly.

`tools/exp10_summarize.py`
11. Bootstrap: seeded reproducibility; a shifted synthetic sample yields Δ's CI containing the true shift; ratio undefined when Δ's CI contains 0.
12. Controls table: flags a k = 512 mismatch > 1e-6.
13. Figure/CSV: files written; CSV rows equal the JSON table values; refuses a JSON whose input sha256s do not match the files on disk.

## 7. Validation ladder and acceptance

Static checks → tests (new + the exp_03/04/07 record suites unchanged) → smoke (`--max-samples 4`, CPU) → probe (256 queries, CPU, timing) → full. **Parity gate before the full runs:** on the 256-query probe of `released_k8`, Metric 2 at k ∈ {0, 64, 128, 256, 384} vs exp_03's `sweep_released/per_sample_yaw.json` (GPU, TF32 off): mean |Δ| ≤ 1e-3 s EDT / 1e-2 dB C50 / 0.1 % T60 and `logspec_mad` mean within 1e-3 (CPU vs GPU float differences are expected at the 1e-5 level; a larger gap means a wrong reference/phase/alignment and blocks the launch). Controls (k = 512 ≡ 0; repeated 0) must be exact to 1e-6. Full-run acceptance: 6 337 queries in canonical order, every angle present, finite counts = 6 337 at k = 0 for EDT/C50 (T60 descriptive), meta complete, `check_online` passes.

## 8. Compute and outputs

CPU: ≈ 40 min per angle-pass per arm (measured on the qualitative export, 24 threads) → ≈ 3.5 h per arm, 4 arms ≈ 14 h serial, ≈ 7 h as two processes. GPU (if a card frees): ≈ 20 min per arm. Disk: 20 waveform arrays × 243 MB ≈ 4.9 GB under `ckpt/exp10/` (git-ignored; 51 GB free). Deliverables per SOP: `_params_set_up.md`, `_command.md`, teed logs, `_results.md`, `_analysis.md`, `yaw_pilot_01_results.html` + assets (figures, CSV, canonical JSON with sha256s), `commits_yaw_pilot.md`.

## 9. Out of scope

Training; condition E; angles beyond the five; retrieval metrics (xRIR has none); any change to pinned exp_03/exp_04 modules; confirmatory hypothesis tests (exp_03 holds them).
