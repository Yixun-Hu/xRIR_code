# exp_10 yaw_pilot — Codex plan review (round 1, plan v1)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="xhigh"' -s read-only -C xRIR_code --skip-git-repo-check`, stdin closed) · **Prompt:** `review_prompts/plan_prompt.md` · **Raw log:** `yaw_pilot_2026-09-26_17:24:59_codex_plan_review.log` · **Date:** 2026-09-26. Note: the SOP mandates `ultra`; the shared `~/.codex/config.toml` had drifted to `xhigh` and this round used the explicit `xhigh` override (finding 11); round 2 onward runs at `ultra` (verified accepted by codex-cli 0.154.0).

**Reviewer:** OpenAI Codex (`codex-cli 0.154.0`; configured `gpt-6-astra`, `xhigh`; `codex exec`, read-only) · **Date:** 2026-09-26

Reviewed [plan v1](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/plan_yaw_pilot.md), the requested experiment records and implementations, and exp_04’s five-seed table.

The core design is sound: condition P consistently rotates the geometric conditioning while preserving reference audio, target audio, and baseline alignment. This is an appropriate analogue of FLAC’s conditioning rotation. Keeping exp_03’s modules frozen and excluding training, condition E, and confirmatory tests is appropriate.

1. **Blocker — Specify the complete paired-bootstrap contract (§§1, 4, 6).**  
   The intended ratio of means is correct, but Δ uses paired-valid errors while G’s validity mask is unspecified. Independently dropping invalid values can make the bars describe different queries and break ratio pairing.

   **Minimal fix:** For each acoustic metric and angle, define a comparison mask where `e₀`, `eα`, and `gα` are all finite. Resample those query tuples once per replicate and compute `Δ*=mean(eα−e₀)`, `G*=mean(gα)`, and `R*=G*/Δ*`. Report the comparison count and exclusions; distinguish any standalone G computed on a broader population. Restrict ratios to metrics with corresponding error/gap definitions.

   Keep the zero-crossing rule, but distinguish “denominator uncertain” from mathematical division by zero. Handle negative degradation explicitly: an entirely negative Δ interval means improvement, not a positive degradation multiple. Never take `abs(Δ)`, add an epsilon, or silently discard inconvenient bootstrap draws. Add tests for shared masks/resamples, negative and zero denominators, empty masks, and unequal room sizes. Room bootstraps must preserve query weighting and drawn-room multiplicities.

2. **Blocker — The CPU/GPU parity gate can admit discrepancies comparable to the finding (§7).**  
   The proposed limits are 1 ms EDT and 0.01 dB C50 per angle. In the released exp_03 records, ΔEDT at 45° is approximately **−0.585 ms**, while ΔC50 at 90° and 180° is approximately **−0.00277 and +0.00845 dB**. Passing the gate therefore does not establish preservation of the small effects or their signs. Per-sample maxima are reported but have no acceptance limits.

   The probe is also geographically narrow: the first 256 queries contain **250 from one apartment and six from another**.

   **Minimal fix:** Select intact canonical batches spanning all 17 rooms. Calibrate backend discrepancies through aligned references, predicted log-spectra, magnitudes, and acoustic metrics; gate the discrepancy in **paired Δ itself**, alongside per-sample diagnostics. Qualify each backbone/K execution path. Preserve exp_03’s operation placement: magnitude construction occurs before transfer to CPU, and Griffin–Lim runs on CPU. Pin the relevant library versions and precision settings. If backend equivalence cannot be established, report a separate CPU protocol rather than claiming numerical replication.

3. **Should-fix — FLAC comparability and Griffin–Lim attribution need precise limits (§§1, 4).**  
   The acoustic gaps follow the same reference convention and both pipelines use the first 8,000 samples at 22,050 Hz. However, xRIR’s waveform distances use 9,600 samples, including 21 padding samples; FLAC’s stored predictions have 10,240 samples. Their waveform distances are not directly matched measurements.

   Shared phase initialization correctly removes independent initialization noise. It does **not** remove Griffin–Lim’s nonlinear response to magnitude changes. That affects acoustic gaps as well as waveform L2. The plan’s categorical claim that Griffin–Lim “inflates” distances is stronger than this experiment establishes.

   **Minimal fix:** Label waveform and acoustic results as sensitivity of the **model-plus-Griffin–Lim pipeline**, conditional on the chosen seed. Say Griffin–Lim *can amplify* differences; retain GL-free spectral shift as separate evidence. Avoid direct FLAC/xRIR waveform-magnitude rankings unless evaluated on a common window. Define `M=exp(out)−1e−8` and `logspec_mad=mean|outα−out₀|` explicitly—the latter is exp_03’s consistency, not literally a fresh `log(M)` calculation.

4. **Should-fix — T60’s two percentages have different denominators (§4).**  
   T60 accuracy error is normalized by `T60(GT)`; prediction shift is normalized by `T60(P₀)`. Consequently, G/Δ is a valid descriptive convention, but its magnitude also reflects different normalizations. It is not a common-normalization measure of displacement versus accuracy loss.

   **Minimal fix:** Label ΔT60 as **percentage-point change in GT-normalized error** and G as **percent of baseline-prediction T60**. Qualify any T60 multiple accordingly. EDT and C50 support the cleaner same-unit comparison; an additional absolute-T60 analysis is optional.

5. **Should-fix — Controls need independent execution and a nonzero-angle wrapper test (§§3, 6).**  
   `rotate_scene_yaw` reduces k modulo 512, so k=512 exercises the identity path. It checks plumbing but cannot establish correct nonzero-angle handling. Likewise, a repeated-zero control is ineffective if it reuses cached predictions. The current new-wrapper tests emphasize identity cases.

   **Minimal fix:** Require fresh inference for repeated-zero controls, stored under distinct control identifiers. On the representative probe, repeat one nonzero angle on the selected backend and verify both waveforms and acoustic metrics. Add a nonzero-angle test against the trusted rotation/fixed-alignment composition, checking unchanged reference/target audio; add angle-order invariance and restoration-after-exception tests. These remain bounded diagnostics, without condition E or additional scientific angles.

6. **Should-fix — Released K=1 is a new evaluation condition, not an exp_03 replication (§3).**  
   “Every Metric-2 number here is a replication of exp_03” contradicts the notebook: there is no released K=1 predecessor. Evaluating a K=8-trained checkpoint with one reference is feasible, but matching FLAC’s reference count does not match its training/evaluation protocol. The K=1 manifest is also not simply the first reference from each K=8 entry.

   **Minimal fix:** Label this arm **“released checkpoint, trained K=8, evaluated K=1”**, retain K=8 as primary, and mark released-K1 parity against exp_03 as unavailable. Validate strict checkpoint loading and agreement among CLI, manifest, and model K. Do not attribute K=1-versus-K=8 differences solely to reference count. Scope the cited P/E agreement to exp_03’s measured K=8 cells.

7. **Should-fix — The borrowed band is not the noise floor of Δ or G (§4).**  
   Exp_04’s SD describes variability of baseline error means when **both references and Griffin–Lim phases change**. This pilot fixes both across angles. That SD does not estimate the variability of paired Δ, prediction shift, or released-checkpoint results. Merely noting the missing released five-seed set does not resolve the statistical mismatch.

   **Minimal fix:** Remove the band or label it **historical baseline evaluation variability, for context only**. Do not use it for significance or acceptance decisions. Use the available cylindrical SD for cylindrical context. Keep paired CIs and measured repeatability controls distinct from this historical scale.

8. **Should-fix — Make float32 storage and artifact identity an explicit contract (§§5–7).**  
   Float32 `.npy` storage is appropriate for the existing fp32 inversion path; increasing precision is unnecessary. The important requirement is that online and offline metrics consume the same waveform representation. Arm/manifest/seed/device labels alone do not establish checkpoint identity or array ordering.

   **Minimal fix:** Compute online waveform/acoustic metrics from the exact float32 arrays being saved, without clipping or normalization. Record native/padded lengths, sample rate, metric window, ordered query IDs, checkpoint hash, backbone/K, batch/precision settings, and relevant software versions. Hash waveform artifacts and require matching metadata during comparison. Reject incomplete runs and mismatched identities; test the actual save/load round trip and missing-value masks. Verify the pinned source files remain unchanged before launch.

9. **Should-fix — Deliverables must support negative and uncertain degradation, and visibly include the GL-free readout (§§1, 5, 8).**  
   The figure specification lists acoustic panels, despite R3 requiring GL-free spectral shift beside waveform shift. It also does not explicitly require CI error bars, invalid counts, or negative-bar handling. These matter because the released checkpoint already shows improvements at several selected angles.

   **Minimal fix:** Include the spectral readout, signed degradation bars, interval endpoints, paired-valid counts, and explicit ratio status in the canonical JSON/CSV and presentation. Permit wording such as “predictions shift while net error improves” or “net degradation is unresolved.” Do not translate a zero-crossing CI into equivalence or invariance. Keep claims descriptive, per angle, checkpoint, reference draw, and seed; disclose room-cluster disagreements. If claiming cancellation of improvements and deteriorations, show their paired contributions rather than inferring cancellation from G/Δ alone.

10. **Should-fix — Runtime estimates need measurement of this evaluator (§8).**  
    The approximately 4.9 GB waveform estimate is consistent. The runtime estimates are less secure: Griffin–Lim remains CPU work even with GPU inference, K=1 and K=8 have different forward costs, and two 24-thread processes need not halve elapsed time.

    **Minimal fix:** Add an explicit thread/worker budget and time inference, inversion, metrics, and writing separately in the probe. Forecast each arm/backend from those measurements, including controls and offline comparison. Qualify concurrent CPU execution against available cores and ongoing jobs before assuming a twofold speedup.

11. **Should-fix — Review configuration conflicts with the SOP (§0/header).**  
    The plan and recorded invocation use `xhigh`; the SOP specifies `gpt-6-astra` at `ultra` for reviews.

    **Minimal fix:** Run the revised-plan approval review at the mandated setting, or record an explicit user-authorized exception. Preserve the required per-round and final code reviews before full runs.

**Verdict: REQUEST CHANGES.**
