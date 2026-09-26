# exp_10 yaw_pilot — Codex plan review (round 3, plan v3)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="ultra"' -s read-only -C xRIR_code --skip-git-repo-check`, stdin closed) · **Prompt:** `review_prompts/plan_round3_prompt.md` · **Raw log:** `yaw_pilot_2026-09-26_17:37:52_codex_plan_review_round3.log` · **Date:** 2026-09-26 · **Verdict:** APPROVE WITH CHANGES (3 should-fixes + 1 nit → plan §12 amendments A1–A4, enforced in the Coder round and its code review)

**Reviewer:** OpenAI Codex `gpt-6-astra`, `ultra` (`codex-cli 0.154.0`, `codex exec`, read-only) · **Date:** 2026-09-26

Reviewed all requested records and code against plan v3. **No methodological blocker remains.** Condition P is an appropriate conditioning-rotation analogue. Metric 1 now establishes conceptual FLAC comparability with explicit estimator, window, and T60-normalization qualifications. Griffin–Lim attribution is honest; float32 storage, fresh controls, the released checkpoint’s K=1 qualification, and the contextual historical variability band are appropriate.

The shared-mask bootstrap correctly computes paired Δ, G, and ratios of means; room resampling preserves query weighting. The revised backend gates address the small historical effects, and CPU fallback keeps comparisons within its own predictions. Storage is approximately **4.87 GB**; runtime remains provisional until the per-arm probes. All 12 live pinned files currently match their reviewed hashes.

1. **Should-fix — Validity mismatches need a binding parity rule.**  
   [§7](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/plan_yaw_pilot.md:82) reports mask mismatches but does not make them fail replication. Its 99.5% tolerance permits one newly invalid probe value; paired Δ could still pass when calculated on the remaining common-finite rows. Full acceptance checks only baseline EDT/C50 counts. Historical records are finite at every selected acoustic angle.

   **Minimal fix:** Require matching per-query validity masks at every replicated angle/metric, or explicitly label affected cells as nonreplications. Define the population used for historical Δ comparison. Add a fixture with one nonzero-angle NaN whose remaining numerical differences pass.

2. **Should-fix — `run_id` does not uniquely identify the predictions being paired.**  
   [§5](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/plan_yaw_pilot.md:51) hashes configuration fields, so separate executions receive the same ID. It also omits batch/padding configuration, implementation identity, and software versions. Test 16 therefore cannot fully enforce the promised same-run comparison.

   **Minimal fix:** Require Δ/G/R to come from one verified `per_sample.json` and its bound waveform artifacts. Distinguish a complete protocol hash from a unique execution ID. Test rejection of separate executions with identical configuration and of same-device runs with different batch or implementation settings.

3. **Should-fix — Finish the ratio’s zero-denominator and convergence reporting rules.**  
   [R2 and the bootstrap contract](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/plan_yaw_pilot.md:12) check exact-zero **bootstrap** denominators, but the observed Δ itself can equal zero even when none of the finite bootstrap draws does. Suppressing the interval alone does not safely define the point ratio. Also, endpoint movement below 10% of interval width does not guarantee that both seeds give the same denominator status.

   **Minimal fix:** Guard observed `Δ == 0` before division and emit a null point ratio with its reason. Permit headline multiples only when bound convergence passes and denominator statuses agree across seeds; otherwise report unresolved Monte Carlo uncertainty. Add fixtures for both cases. No additional model evaluations are needed.

4. **Nit — Correct the `log_mse` implementation contract.**  
   [§4](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/plan_yaw_pilot.md:34) attributes `log_mse` to `per_sample_losses`, which returns only loss, STFT, and decay terms.

   **Minimal fix:** Specify or reuse [exp_03’s separate calculation](/home/yixunhu/codespace/xRIR_code/eval_yaw_rotation.py:128): mean squared difference between predicted log-magnitude and `log(tgt_spec + 1e-8)`, reduced per query.

With these bounded corrections, the deliverables support descriptive, per-angle statements about **prediction shift versus signed accuracy change**, with room-level uncertainty disclosed. All fixes fit within the authorized new files and tests.

**Verdict: APPROVE WITH CHANGES.**
