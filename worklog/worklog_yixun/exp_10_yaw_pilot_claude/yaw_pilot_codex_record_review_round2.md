# exp_10 yaw_pilot — Codex record review (round 2)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="ultra"' -s read-only -C /home/yixunhu/codespace/xRIR_code --skip-git-repo-check`, stdin closed) · **Prompt:** `review_prompts/record_review_round2_prompt.md` · **Record at:** `a384c9f` · **Raw log:** `yaw_pilot_2026-09-27_08:49:56_codex_record_review_round2.log` · **Date:** 2026-09-27 · **Verdict:** APPROVED WITH CORRECTIONS (0 blockers; 3 should-fixes: two ratio wordings, stale summary hashes, command-register completeness; 2 nits: K1 EDT placement, HEAD roles)

**Reviewer:** OpenAI Codex `gpt-6-astra`, ultra; codex-cli 0.154.0, read-only `codex exec` · **Date:** 2026-09-27 · **Record:** `a384c9f`

**Verdict: APPROVED WITH CORRECTIONS — 0 blockers, 3 should-fixes, 2 nits.** The canonical statistics are unchanged and the regenerated numerical record passes. Remaining corrections concern wording and provenance.

1. **Should-fix — two ratio statements remain overgeneralized.**  
   [yaw_pilot_analysis.md:29](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_analysis.md:29), also line 53.

   “Where a multiple is reportable … an order of magnitude” includes reportable examples of **5.2×** and **6.0×**. The proposed paper sentence “FLAC’s single-run degradations are 3–5× smaller than its shifts” applies approximately to **EDT only**: FLAC’s T60 multiples span **4.56–8.56×**, and C50 spans **3.87–20.85×**.

   **Minimal fix:** use “Reportable cells show shifts several times larger than signed error changes”; scope the FLAC sentence to “EDT shifts are 3.2–5.4× its single-run degradations.”

2. **Should-fix — the analysis still cites round-1 summary hashes.**  
   [yaw_pilot_analysis.md:3](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_analysis.md:3).

   Both canonical files changed hash because regeneration updated `generated_at`. Their current SHA256 values are:

   - GPU: `826498f58780dfe42a4e6228075b0fcb2be845a18a0986d0e835302faee85b32`
   - CPU: `3526094b796a9c1f8bd4d8bae82f8373a7df063ff68cfec6fecdc601f1b8ced1`

   **Minimal fix:** replace the two stale pins. No statistical correction is needed.

3. **Should-fix — round-1 finding 8 remains partially open: reconstructed commands are incomplete.**  
   [yaw_pilot_command.md:38](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_command.md:38), especially lines 50–54.

   Smoke commands still contain `47637a55…`, `<scratchpad>` and `<dir>`. The common “unchanged for every run” implementation statement also incorrectly includes developmental smokes: the first smoke records evaluator hash `1618de6d…`, versus production `e727cba0…`; both early smoke records disclose dirty working trees.

   Preview/interim commands and logs remain recoverable as `run_e10_preview.sh`, `e10_sum_preview.log`, `e10_sum_cpu.log` and `e10_sum_interim.log` in the session scratchpad. The regeneration log’s actual timestamp is **04:38:18**, not `04:3x`.

   **Minimal fix:** finish the reconstructed register with executable commands, full hashes, concrete paths/logs, and per-execution revision qualifications. Preserve reconstruction labels and the honest disclosure that the first CPU attempt’s output is unrecoverable. Its failure reproduction must leave `PYTHONPATH` unset, unlike the stated common environment.

4. **Nit — K1’s EDT placement is still described incorrectly.**  
   [yaw_pilot_analysis.md:25](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_analysis.md:25).

   K1 does not “lie between” control and cylindrical EDT shifts: it is below both at 45° (**12.515 versus 14.250/16.391 ms**) and above both at the other angles.

   **Minimal fix:** say simply “has EDT gaps of 10.6–12.5 ms.”

5. **Nit — repository HEAD and reviewed implementation remain slightly conflated.**  
   [yaw_pilot_params_set_up.md:45](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_params_set_up.md:45), [yaw_pilot_command.md:56](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_command.md:56), [yaw_pilot_worklog.md:146](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_worklog.md:146).

   Chain HEADs differ in unrelated exp_06 code as well as bookkeeping; the **exp_10 evaluation bytes** remain unchanged. The second finish log records process HEAD **`f01116f`**; **`9327a4d`** is the reviewed merge.

   **Minimal fix:** narrow the unchanged-code statement to exp_10 and distinguish these two commit roles.

**Round-1 finding disposition**

- **1:** blocking defect repaired; named xRIR multiples now obey reportability and room-status rules, and the must-not-claim list is adequate. Residual wording is finding 1 above.
- **2:** closed. Both MC failure mechanisms are disclosed and withholding is preserved. The plotted cylindrical T60 cell shows `mc?`; control T60-absolute is outside the standard figure panels and is withheld in both reports.
- **3:** the specified peak, ordering, interval-width and reduction errors are corrected; the remaining EDT description is nit 4 above.
- **4, 5, 6:** closed—unchanged queries counted separately, K1 qualification present in reports/figures, backend equality claim removed.
- **7:** substantively closed—completion HEAD and chain-start HEAD are distinguished; nit 5 remains.
- **8:** partially closed—failed-launch disclosure and aborted-log rename corrected; command completeness remains finding 3.
- **9:** closed—substantive experiment commits, including scaffold, original publication and republication, are indexed; unrelated history is separated.

**Verified**

- Read the SOP, plan/amendments, three plan reviews, all eight code reviews, tooling review, experiment records, chain/queue/finish logs, and specified FLAC sources.
- Compared both canonical summaries recursively with round-1 copies at **`8009347`**: **only `/generated_at` changed**. Every number, interval, count, status, convergence field and input identity is identical.
- All **80 Markdown metric rows**, **160 HTML metric rows** and **32 backend-table rows** match canonical values, intervals, counts, query/room statuses and permitted multiples.
- Independently recomputed **250 point cells** from per-sample data—200 GPU and 50 CPU, including angle zero—with **zero discrepancy**.
- Independently reproduced **18 cells’ query and room bootstraps**, each with 10,000 seed-0 draws: all **36 sets** of bounds/statuses matched, with maximum floating-point difference **4.27×10⁻¹⁴**.

Selected recomputed cells follow. EDT/T60-absolute use ms; C50 uses dB; T60 uses pp for Δ and baseline-prediction % for G. D = defined, U = denominator uncertain, I = improvement, X = undefined.

| Arm | Angle / metric | Δ | G | Reportable multiple | Query / room |
|---|---|---:|---:|---:|---|
| released_k8 | 45° EDT | −0.584839 | 26.2470 | — | U / U |
| released_k8 | 90° C50 | −0.002775 | 0.606292 | — | U / U |
| released_k8 | 180° T60 | −0.235516 | 3.28893 | — | I / U |
| released_k8 | 270° T60 abs | 0.464844 | 18.0057 | — | U / U |
| released_k1 | 45° EDT | −0.124310 | 12.5152 | — | U / U |
| released_k1 | 90° C50 | 0.031451 | 0.226879 | 7.214× | D / U |
| released_k1 | 180° T60 | 0.052176 | 1.43724 | — | U / U |
| released_k1 | 270° T60 abs | 0.117354 | 7.39138 | — | U / U |
| control_k8 | 45° EDT | 0.630398 | 16.3912 | — | U / U |
| control_k8 | 90° C50 | 0.024037 | 0.299991 | 12.480× | D / U |
| control_k8 | 180° T60 | −0.341186 | 1.53514 | — | I / I |
| control_k8 | 270° T60 abs | −0.906865 | 9.58584 | — | I / I |
| control_k8 | 45° T60 abs | 0.065848 | 14.6337 | MC withheld | X / U |
| cyl_k8 | 45° EDT | 1.147091 | 14.2501 | 12.423× | D / D |
| cyl_k8 | 90° C50 | 0.032011 | 0.251351 | 7.852× | D / U |
| cyl_k8 | 180° T60 | −0.304588 | 1.22736 | — | I / I |
| cyl_k8 | 270° T60 abs | −0.464865 | 8.33133 | — | I / U |
| cyl_k8 | 270° T60 | −0.071539 | 1.42346 | MC withheld | I / U |

- Independently reproduced both MC mechanisms: control T60-absolute45 has **1/10,000 exact-zero draws** under seed 0 and none under seed 1; cylindrical T60 270° ratio-bound movement is **0.1088600626 > 0.10**.
- Released K8’s primary cells are **5 uncertain / 3 improvement / 4 defined**. EDT45 counts reproduce **3,195 worse / 2,998 better / 144 unchanged**, with mean contributions **+10.8546/−11.4394 ms**.
- **Replication-qualified arms: GPU `released_k8`, `control_k8`, `cyl_k8`.** All **120 required probe/full parity cells** pass with identical validity masks. K1 has no predecessor. CPU released K8 fails full spectral parity and correctly remains a **separate CPU protocol**.
- All **12 GPU probe controls** match their original acoustic arrays and have zero recorded waveform/log-spectrum deviations; all **10 online-check reports** pass. Backend maximum point difference is **0.462555%**, for EDT45 Δ.
- All **46 asset checksums**, **70 run-bound files**, **27 copied JSONs**, **27 HTML provenance hashes**, and **12 source pins** match. Page links resolve; ten execution identities agree across their bound records.
- The must-not-claim list now covers the required restrictions. T60 denominators, context-only historical bands, Griffin-Lim qualification and GL-free readouts are present. FLAC qualifications match its code: **8,000 samples at 22,050 Hz**, AcousticRooms T60 sentinel inclusion, and C50’s **`+1e-10` inside the logarithm**.
- Required record artifacts and review closures are present. Historical SOP deviations remain disclosed: late command reconstruction, the running-queue edit, and unrecoverable first CPU-attempt output. They cannot be represented as complete launch-time logging.
