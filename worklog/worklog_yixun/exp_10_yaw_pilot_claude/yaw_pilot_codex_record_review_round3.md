# exp_10 yaw_pilot — Codex record review (round 3)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="ultra"' -s read-only -C /home/yixunhu/codespace/xRIR_code --skip-git-repo-check`, stdin closed) · **Prompt:** `review_prompts/record_review_round3_prompt.md` · **Record at:** `0c1b20d` · **Raw log:** `yaw_pilot_2026-09-27_08:57:46_codex_record_review_round3.log` · **Date:** 2026-09-27 · **Verdict:** APPROVED WITH CORRECTIONS (0 blockers; 1 should-fix: smoke commands in the register; 1 nit: HEAD label residue)

**Verdict: APPROVED WITH CORRECTIONS — 0 blockers, 1 should-fix, 1 nit.**

**Reviewer:** OpenAI Codex · **Date:** 2026-09-27 · **Record:** `0c1b20d`

The numerical record and scientific claims pass. Three round-2 items are closed; command completeness and HEAD labeling remain partially corrected.

1. **Should-fix — smoke reproduction commands remain incomplete and inaccurate.**  
   [yaw_pilot_command.md:50](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_command.md:50).

   Entry 5 retains literal `<out-dir>` arguments and ellipsized command paths. Its reconstructed round-2 summarizer command would use **10,000 draws** and write `exp10_smoke_r2_summary`; the retained artifact is actually `scratchpad/exp10_smoke_r2/summary/yaw_pilot_summary.json`, with **`n_boot: 2000`**.

   The common environment exports the main repository in `PYTHONPATH`, while the smoke command subsequently changes into `xRIR_code_wt10`. The reconstruction needs the worktree’s import path explicitly.

   **Minimal fix:** spell out each smoke, comparator and summarizer command, including actual output/report paths and bootstrap counts; explicitly set the worktree environment. Preserve the reconstruction labels, dirty-tree disclosures and distinct evaluator hashes. Entries **1, 6, 8, 9 and 10** are substantively corrected; entry **11** has the labeling residue below.

2. **Nit — the obsolete finish HEAD label survives beside its correction.**  
   [yaw_pilot_command.md:56](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_command.md:56), [yaw_pilot_worklog.md:146](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_worklog.md:146).

   The command register correctly introduces process HEAD `f01116f` and reviewed tooling merge `9327a4d`, but still says `(HEAD 9327a4d)` later in the same entry. The historical worklog command retains that label too. The finish log records process HEAD **`f01116f1768687a04e0cdae458f831f90120b2bf`**.

   **Minimal fix:** relabel `9327a4d` consistently as the reviewed merge and append an explicit notebook correction. The params document now distinguishes the roles correctly.

**Round-2 and round-1 disposition**

- **Ratio wordings: closed.** “Several times larger” and the metric-specific FLAC ranges are accurate.
- **Summary hashes: closed.** Both analysis pins match the current canonical files.
- **Command register: partially closed**, finding 1 above.
- **K1 EDT sentence: closed.** Its 10.6–12.5 ms range and angle-specific ordering are correct.
- **HEAD roles: partially closed**, finding 2 above.
- Round-1 scientific corrections remain resolved: reportability, room caveats, cancellation contributions, unchanged-query counts, K1 qualification, backend wording and must-not-claim restrictions. The aborted-log rename and substantive commit index remain corrected. Round-1 command completeness remains partially open.

**Verified**

- Read the SOP, plan/amendments, plan reviews, all eight code reviews, tooling review, record documents, production logs and specified FLAC sources.
- Compared **both canonical summaries recursively against round-1 copies at `8009347`**: **only `/generated_at` changed**. Every number, interval, count, status, convergence field and input identity is identical. Published summary copies match canonical files byte-for-byte.
- Independently reconstructed all **80 Markdown metric rows**, **160 HTML metric rows** and **32 backend comparison rows**. Values, intervals, counts, query/room statuses and permitted multiples match.
- Independently recomputed **250 point cells** from per-sample data—200 GPU and 50 CPU, including zero angle—with **zero discrepancy**.
- Reproduced **19 cells’ query and room bootstraps**, each with 10,000 seed-0 draws: **38 sets** of bounds/statuses match, maximum floating-point discrepancy **1.42×10⁻¹⁴**. Coverage includes EDT45, C50 90, T60 180 and T60-absolute270 for every GPU arm, both MC exceptions, and CPU EDT45.

Selected recomputed cells follow. EDT/T60-absolute use ms; C50 uses dB; T60 uses pp for Δ and baseline-prediction % for G. D/U/I = defined/denominator uncertain/improvement.

| Arm | Angle / metric | Δ | G | Permitted multiple | Query / room |
|---|---|---:|---:|---:|---|
| released_k8 | 45° EDT | −0.584839 | 26.2470 | — | U / U |
| released_k8 | 90° C50 | −0.002775 | 0.606292 | — | U / U |
| released_k8 | 270° T60 abs | 0.464844 | 18.0057 | — | U / U |
| released_k1 | 45° EDT | −0.124310 | 12.5152 | — | U / U |
| released_k1 | 90° C50 | 0.031451 | 0.226879 | 7.214× | D / U |
| released_k1 | 180° T60 | 0.052176 | 1.43724 | — | U / U |
| control_k8 | 45° EDT | 0.630398 | 16.3912 | — | U / U |
| control_k8 | 90° C50 | 0.024037 | 0.299991 | 12.480× | D / U |
| control_k8 | 270° T60 abs | −0.906865 | 9.58584 | — | I / I |
| cyl_k8 | 45° EDT | 1.147091 | 14.2501 | 12.423× | D / D |
| cyl_k8 | 90° C50 | 0.032011 | 0.251351 | 7.852× | D / U |
| cyl_k8 | 180° T60 | −0.304588 | 1.22736 | — | I / I |

- Independently reproduced both MC failures: control T60-absolute45 has **1/10,000 exact-zero draws** under seed 0 versus none under seed 1; cylindrical T60 270 has ratio-bound movement **0.1088600626 > 0.10**. Withholding remains intact. The plotted cylindrical cell shows `mc?`; control T60-absolute is outside the standard panels and withheld in both reports.
- **Replication-qualified arms:** GPU `released_k8`, `control_k8`, `cyl_k8`. All **120 required probe/full parity cells** pass with identical masks. K1 has no predecessor. CPU released K8 fails full spectral parity and remains correctly labeled a **separate CPU protocol**.
- All **12 GPU probe controls** match; all **10 online-check reports** pass. Backend maximum relative point difference is **0.462555%**, and all 16 acoustic denominator statuses agree.
- Released-K8 primary statuses reproduce **5 uncertain / 3 improvement / 4 defined**. EDT45 contributions reproduce **3,195 worse / 2,998 better / 144 unchanged**, supporting the qualified cancellation statement.
- All **46 asset checksums**, **70 run-bound files**, **27 copied JSONs**, **27 HTML provenance hashes**, **10 relative page links** and **12 source pins** verify; ten execution identities agree.
- Figures retain the **“trained K = 8, evaluated K = 1”** label, readable qualifications and context-only bands. T60 denominators and model-plus-Griffin-Lim scope are present. FLAC qualifications match its code: 8,000-sample window at 22,050 Hz, AcousticRooms T60 sentinel inclusion, and C50’s `+1e-10` inside `log10`. The must-not-claim list is adequate.

Required record artifacts are present. Historical SOP deviations remain disclosed: reconstructed commands, editing the running queue, and unrecoverable output from the first CPU launch. These cannot retrospectively constitute complete launch-time logging.
