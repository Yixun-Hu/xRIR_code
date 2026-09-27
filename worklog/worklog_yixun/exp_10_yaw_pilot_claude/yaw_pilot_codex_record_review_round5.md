# exp_10 yaw_pilot — Codex record review (round 5)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="ultra"' -s read-only -C /home/yixunhu/codespace/xRIR_code --skip-git-repo-check`, stdin closed) · **Prompt:** `review_prompts/record_review_round5_prompt.md` · **Record at:** `7fc25a0` · **Raw log:** `yaw_pilot_2026-09-27_09:10:01_codex_record_review_round5.log` · **Date:** 2026-09-27 · **Verdict:** RECORD APPROVED (no open blockers, should-fixes or nits)

**Verdict: RECORD APPROVED — no open blockers, should-fixes, or new nits.**

**Reviewer:** OpenAI Codex `gpt-6-astra`, ultra; codex-cli 0.154.0, read-only `codex exec` · **Date:** 2026-09-27 · **Record:** `7fc25a0`

1. **Should-fix from round 4 — closed.**  
   [yaw_pilot_command.md:52](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_command.md:52) supplies the worktree environment and executable `EXP10_SMOKE_SCRATCH` assignment. Round 2’s evaluator command is fully written at line 62. All nine Python commands pass shell syntax checks; expanded inputs, output directories and report paths resolve to the retained artifacts. Execution IDs, evaluator hashes, dirty-tree disclosures and bootstrap counts **10,000 / 2,000 / 2,000** match those artifacts. Commands were checked without rerunning or overwriting experiments. **Minimal fix: none remaining.**

2. **Rounds 1–3 — remain resolved.**  
   The analysis, must-not-claim restrictions, reportability and room caveats, cancellation contributions, unchanged-query counts, backend wording, summary hashes, params’ HEAD distinctions, aborted-log rename and substantive commit index remain corrected. Round-8’s figure annotation and K1-label fixes survive republication. **Minimal fix: none remaining.**

**Verified**

- Compared both canonical summaries recursively against the round-1 published copies at **`8009347`**: **only `/generated_at` differs**. Every number, interval, count, status, convergence field and input identity is identical. Published summary copies match current canonical files byte-for-byte.
- Independently reconstructed **80 Markdown metric rows, 160 HTML metric rows, 32 backend rows and 64 figure CSV rows**. Displayed values, intervals, counts, query/room statuses and permitted multiples match.
- Recomputed **250 point cells** from per-sample data, including masks, exclusions, means, Δ/G, paired contributions and ratios. Maximum discrepancy: **3.33×10⁻¹⁶**.
- Independently reproduced **20 cells’ query and room bootstraps**, each with 10,000 seed-0 draws: **40 sets** of bounds/statuses match; maximum floating-point discrepancy **1.42×10⁻¹⁴**.

Representative checked cells follow. EDT/T60-absolute use ms; C50 uses dB; T60 uses GT-normalized pp for Δ and baseline-prediction % for G. D/U/I/X mean defined/denominator uncertain/improvement/undefined.

| Arm | Angle / metric | Δ | G | Permitted multiple | Query / room |
|---|---|---:|---:|---:|---|
| released_k8 | 45° EDT | −0.584839 | 26.246974 | — | U / U |
| released_k8 | 45° T60 | 0.472875 | 3.782135 | 7.998× | D / D |
| released_k8 | 180° T60 | −0.235516 | 3.288928 | — | I / U |
| released_k1 | 90° C50 | 0.031451 | 0.226879 | 7.214× | D / U |
| released_k1 | 180° T60 | 0.052176 | 1.437240 | — | U / U |
| control_k8 | 90° C50 | 0.024037 | 0.299991 | 12.480× | D / U |
| control_k8 | 270° T60 abs | −0.906865 | 9.585836 | — | I / I |
| control_k8 | 45° T60 abs | 0.065848 | 14.633698 | MC withheld | X / U |
| cyl_k8 | 45° EDT | 1.147091 | 14.250139 | 12.423× | D / D |
| cyl_k8 | 90° C50 | 0.032011 | 0.251351 | 7.852× | D / U |
| cyl_k8 | 180° T60 | −0.304588 | 1.227356 | — | I / I |
| cyl_k8 | 270° T60 | −0.071539 | 1.423461 | MC withheld | I / U |

- Fresh seed-1 recomputation also confirms both MC exceptions: control T60-absolute45 has **1/10,000 exact-zero draws** under seed 0 versus none under seed 1; cylindrical T60 270 has ratio-bound movement **0.1088600626 > 0.10**. Both remain withheld. Cylindrical T60 270 displays `mc?`; control T60-absolute45 is outside the standard figure panels and is withheld in the reports.
- **Replication-qualified arms: GPU `released_k8`, `control_k8`, and `cyl_k8`.** All **120 required probe/full parity cells** pass with identical masks. K1 has no predecessor. CPU released K8 fails four full-split spectral cells and remains correctly labeled a **separate CPU protocol**.
- All **12 GPU probe controls** have zero recorded waveform/log-spectrum deviations and matching acoustic arrays; all **10 online-check reports** pass. Backend statuses agree across all 16 acoustic cells; maximum relative point difference is **0.462555%**.
- Released-K8 primary statuses reproduce **5 uncertain / 3 improvement / 4 defined**. EDT45 cancellation counts reproduce **3,195 worse / 2,998 better / 144 unchanged**, with contributions **+10.8545965 / −11.4394352 ms**.
- All **46 asset checksums, 70 run-bound file hashes, 27 published JSON copies, 27 HTML provenance hashes, 10 relative links and 12 source pins** verify. All ten execution identities and params rows agree with metadata.
- K1’s **“trained K = 8, evaluated K = 1”** label, context-only historical bands, T60 denominators and model-plus-Griffin-Lim qualifications remain present. FLAC qualifications match its code: **8,000 samples at 22,050 Hz**, AcousticRooms inclusion of failed-T60 sentinels, and C50’s **`+1e-10` inside `log10`**. No waveform ranking is claimed.
- The **must-not-claim list is adequate** for the scoped pilot section.

Required artifact categories and review closures are present. Approval retains the documented historical SOP exceptions: reconstructed commands, editing the running queue, and unrecoverable first-CPU-attempt output. These disclosures do not retrospectively make launch-time logging complete.
