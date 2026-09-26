# exp_10 yaw_pilot — Codex plan review (round 2, plan v2)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="ultra"' -s read-only -C xRIR_code --skip-git-repo-check`, stdin closed) · **Prompt:** `review_prompts/plan_round2_prompt.md` · **Raw log:** `yaw_pilot_2026-09-26_17:32:32_codex_plan_review_round2.log` · **Date:** 2026-09-26

**Reviewer:** OpenAI Codex `gpt-6-astra`, `ultra` (`codex-cli 0.154.0`, `codex exec`, read-only) · **Date:** 2026-09-26

Reviewed all requested records and source files against [plan v2](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/plan_yaw_pilot.md).

The core design is sound. Condition P is an appropriate analogue of FLAC’s conditioning rotation. The shared masks and resampling indices correctly pair Δ, G, and G/Δ; room resampling preserves query weighting. Fresh zero/nonzero controls, float32 waveform storage, the K=8-trained/K=1-evaluated qualification, Griffin–Lim attribution, and the contextual historical variability band adequately address their round-1 concerns.

1. **Blocker — The CPU fallback mixes the quantities being compared (§7).**

   After failed CPU parity, the plan permits proceeding “with Metric 2 quoted from exp_03.” That would compare new CPU prediction shifts with historical GPU accuracy changes precisely when backend equivalence failed. Matching query IDs cannot make these measurements describe the same predictions. Released K=1 additionally has no historical counterpart.

   **Minimal fix:** Compute Δ, G, and their paired ratio entirely from the current CPU run. Show exp_03’s GPU values separately as historical comparisons. Add a rejection test for combining metrics from different run/protocol identities.

2. **Should-fix — The parity comparator’s contract rejects the planned probe (§§5–7).**

   `parity_exp03` requires identical query lists, but the probe contains 272 queries and the historical file contains 6,337. The launch rule also explicitly blocks failure of criterion (c), leaving enforcement of (a)/(b) ambiguous.

   **Minimal fix:** Specify validated subset alignment: verify manifest identity and unique canonical query indices, extract exactly the probe’s historical rows, then require ordered equality. Test a valid noncontiguous probe and missing/duplicate/reordered queries. Make all three parity criteria binding and report validity-mask mismatches explicitly. Repeat the inexpensive offline comparison over the full results before calling an arm an exp_03 replication.

3. **Should-fix — The named tests do not verify that the live pinned modules remain unchanged (§5).**

   `tests/test_exp03_record_tools.py` tests provenance validators using [synthetic closure records](/home/yixunhu/codespace/xRIR_code/tests/test_exp03_record_tools.py:112). Passing it does not establish byte equality between the worktree’s evaluator dependencies and exp_03’s reviewed source.

   **Minimal fix:** Add an explicit prelaunch comparison of the pinned files’ live hashes against reviewed commit `62c9107b4150e44c4ac410ff4cab359c1e71cc10`, fail on mismatch, and retain the comparison in exp_10 metadata. This fits inside the authorized new files.

4. **Should-fix — Saved waveforms cannot reproduce every Metric-1 quantity (§§5–6).**

   The comparator promises offline “Metric 1 == online,” but only waveforms and scalar metrics are retained. Native `logspec_mad` and `mag_rel_l2` cannot be reconstructed from finite-iteration Griffin–Lim waveforms. Taking another STFT would measure a different quantity.

   **Minimal fix:** Explicitly restrict offline recomputation to waveform/acoustic gaps; retain and hash the online spectral measurements and verify their direct-output calculations in tests. Alternatively, retain native spectra and update storage estimates.

5. **Should-fix — Acoustic comparability needs an estimator/validity audit, beyond matching windows (§§1, 4).**

   Both evaluators use the first 8,000 samples at 22,050 Hz. However, FLAC substitutes `−1` for failed T60 measurements and [includes those values in AcousticRooms errors](/home/yixunhu/codespace/FLAC/src/metrics/modules/RT60.py:44); xRIR [returns NaN and excludes invalid measurements](/home/yixunhu/codespace/xRIR_code/tools/per_sample_metrics.py:165). FLAC also adds a small constant inside the C50 logarithm.

   **Minimal fix:** Record estimator and invalid-policy differences alongside window verification. Qualify comparisons where matching validity has not been established. Keep the pinned xRIR evaluator unchanged. The common “P₀ as GT” convention establishes conceptual comparability, not automatically identical numerical protocols.

6. **Should-fix — Paper wording must expose query/room disagreement (§§1, 4–5).**

   The plan provides one ratio status and secondary room intervals without requiring disagreements to accompany headline multiples. This matters: exp_03’s released T60 results at +45°/+90° have positive query-level intervals while room intervals cross zero.

   **Minimal fix:** Emit separate query- and room-bootstrap denominator/ratio statuses. Scope headline statements to the checkpoint, reference manifest, phase seed, and fixed split; disclose room-level uncertainty beside any multiple. Also retain round-1’s restriction: G/Δ alone does not establish that improvements and deteriorations cancel—such wording requires their paired contributions.

7. **Nit — The convergence rule cannot pass deterministic zero-width intervals (§4).**

   At k=0, Δ and G have intervals `[0, 0]`. Requiring endpoint movement strictly below 10% of width evaluates to `0 < 0`, which fails.

   **Minimal fix:** Treat identical zero-width intervals as converged, changed zero-width intervals as failed, and absent ratio intervals as not applicable. Include a k=0 fixture; make exact-zero-denominator status precedence explicit.

8. **Nit — The resource window and runtime forecast disagree (§§3, 8).**

   The recorded GPU agreement covers approximately two hours, whereas the forecast is four to five hours plus comparison work.

   **Minimal fix:** Reconcile the launch window with the measured per-arm forecast before scheduling the full sweep. The proposed probe timing breakdown and approximately 4.9 GB waveform estimate are otherwise reasonable.

**Verdict: REQUEST CHANGES.**
