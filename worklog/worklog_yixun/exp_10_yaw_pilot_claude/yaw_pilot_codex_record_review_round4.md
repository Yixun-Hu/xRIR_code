# exp_10 yaw_pilot — Codex record review (round 4)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="ultra"' -s read-only -C /home/yixunhu/codespace/xRIR_code --skip-git-repo-check`, stdin closed) · **Prompt:** `review_prompts/record_review_round4_prompt.md` · **Record at:** `6040bfc` · **Raw log:** `yaw_pilot_2026-09-27_09:04:14_codex_record_review_round4.log` · **Date:** 2026-09-27 · **Verdict:** APPROVED WITH CORRECTIONS (0 blockers; 1 should-fix: executable scratch-path assignment in command entry 5)

**Verdict: APPROVED WITH CORRECTIONS — 0 blockers, 1 should-fix, 0 new nits.**

**Reviewer:** OpenAI Codex `gpt-6-astra`, ultra; codex-cli 0.154.0, read-only `codex exec` · **Date:** 2026-09-27 · **Record:** `6040bfc`

The numerical record and scientific claims pass. The round-3 HEAD nit is resolved; the smoke-command correction remains partially open.

1. **Should-fix — entry 5 still contains non-executable scratch-path substitutions.**  
   [yaw_pilot_command.md:50](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_command.md:50), commands at lines 52–61.

   Commands literally pass `SCRATCH/exp10_smoke`, etc. The prose definition of `SCRATCH` neither assigns a shell variable nor expands these arguments. Copied commands would write beneath the worktree’s relative `SCRATCH/` directory or fail to find the retained artifacts.

   **Minimal fix:** add an executable assignment:

   ```bash
   EXP10_SMOKE_SCRATCH='/tmp/claude-1013/-home-yixunhu-codespace-xRIR-code/279323f7-0bed-4b41-8631-99735eadc6f1/scratchpad'
   ```

   Replace every command argument beginning `SCRATCH/` with `"${EXP10_SMOKE_SCRATCH}/..."`, preserving the actual suffix, and spell out round 2’s evaluator command.

   The substantive corrections **do verify**: worktree `PYTHONPATH`, evaluator hashes and dirty-tree disclosures, exact report destinations, round-1 **10,000** draws, and round-2/3 **2,000** draws. This remaining issue requires only a documentation correction.

The **round-3 HEAD nit is closed**: [command entry 11](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_command.md:68) distinguishes reviewed merge `9327a4d` from process HEAD `f01116f`; the [appended notebook correction](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_worklog.md:158) explicitly corrects the preserved historical entry.

Rounds 1–2’s scientific corrections remain resolved: scoped multiples and room caveats, cancellation contributions, unchanged-query counts, K1 qualification, backend wording, summary hashes and adequate must-not-claim restrictions. The params correction, aborted-log rename and substantive commit index remain intact.

**Verified**

- Read the SOP, plan/amendments, three plan reviews, all eight code reviews, tooling review, record documents, production logs and specified FLAC sources.
- Compared **both canonical summaries recursively against round-1 copies at `8009347`**: **only `/generated_at` differs**. Every number, interval, count, status, convergence field and input identity is identical. Published summary copies match canonical files byte-for-byte.
- Independently reconstructed **80 Markdown metric rows, 160 HTML metric rows and 32 backend rows**. Values, intervals, counts, query/room statuses and permitted multiples match.
- Recomputed **250 point cells** from per-sample data—200 GPU and 50 CPU—including masks, exclusions, means, Δ/G, contributions and ratios: **zero discrepancy**.
- Independently reproduced **19 cells’ query and room bootstraps**, each with 10,000 seed-0 draws: all **38 sets** of bounds/statuses match with **zero discrepancy**.

Representative checked cells follow. EDT/T60-absolute use ms; C50 uses dB; T60 uses GT-normalized pp for Δ and baseline-prediction % for G. D/U/I/X = defined/denominator uncertain/improvement/undefined.

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

- Both MC failures reproduce: control T60-absolute45 has **1/10,000 exact-zero draws** under seed 0 versus none under seed 1; cylindrical T60 270 has ratio-bound movement **0.1088600626 > 0.10**. Withholding remains intact. Cylindrical T60 270 displays `mc?`; control T60-absolute45 is outside the standard figure panels and withheld in the tables.
- **Replication-qualified arms: GPU `released_k8`, `control_k8`, `cyl_k8`.** Their reports contain **120 passing required probe/full parity cells**, with identical validity masks. K1 has no predecessor. CPU released K8 fails four full-split spectral cells and correctly remains a **separate CPU protocol**.
- All **12 GPU probe controls** pass, with zero recorded waveform/log-spectrum deviations and independently matching acoustic arrays. All **10 online-check reports** pass. Backend statuses agree across all 16 acoustic cells; maximum relative point difference is **0.462555%**.
- All **46 asset checksums, 70 run-bound file hashes, 27 copied JSONs, 27 HTML provenance hashes, 10 relative links and 12 source pins** verify. Execution identities and all ten params rows agree with metadata.
- K1 labels, context-only bands, T60 denominators and model-plus-Griffin-Lim qualifications remain present. FLAC qualifications match its code: **8,000 samples at 22,050 Hz**, AcousticRooms T60 sentinel inclusion, and C50’s **`+1e-10` inside `log10`**. The corrected metric-specific FLAC ranges are accurate; no waveform ranking is claimed.
- The must-not-claim list is adequate for the scoped pilot section.

Required record artifacts and review closures are present. Historical SOP deviations remain honestly disclosed: reconstructed commands, editing the running queue, and unrecoverable first CPU-attempt output. They cannot retrospectively constitute complete launch-time logging.
