# exp_10 yaw_pilot — Codex record review (round 1)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="ultra"' -s read-only -C /home/yixunhu/codespace/xRIR_code --skip-git-repo-check`, stdin closed) · **Prompt:** `review_prompts/record_review_prompt.md` · **Raw log:** `yaw_pilot_2026-09-27_07:57:01_codex_record_review.log` · **Date:** 2026-09-27 · **Verdict:** NOT APPROVED (1 blocker on the paper statement / must-not-claim list; 8 should-fixes: MC withholding in figures, over-generalisations, unchanged queries, K1 label, backend wording, params column, command/log completeness, commits index)

**Reviewer:** OpenAI Codex `gpt-6-astra`, ultra; codex-cli 0.154.0, read-only `codex exec` · **Date:** 2026-09-27

**Verdict: NOT APPROVED.** The canonical data and generated numeric tables are consistent. The proposed paper statement, figure wording, and parts of the experiment history need correction.

1. **Blocker — the proposed paper statement exceeds the preregistered ratio evidence.**  
   [yaw_pilot_analysis.md:53](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_analysis.md:53), recommendation at line 63.

   “In both families … an order of magnitude more” generalizes a positive multiple to xRIR cells whose denominators are uncertain or whose errors improve. FLAC also does not support a universal tenfold statement: its EDT at 180° gives `20.11 / 6.28 = 3.20×`.

   **Minimal fix:** describe prediction shifts and mixed signed-error outcomes; give multiples only for named `headline.reportable` cells, with room status alongside. Make the scope of “mostly unresolved” explicit: across all four angles and three primary acoustic metrics, released K8 has five uncertain, three improvement, and four defined query-level cells.

   The **must-not-claim list at line 59 is incomplete**. Add explicit prohibitions on multiples when `headline.reportable=false`, bypassing Monte Carlo withholding, backend equivalence, attributing Griffin-Lim-inclusive gaps solely to the model, causal reference-count explanations, and FLAC/xRIR waveform rankings. Retain the fixed checkpoint/manifest/phase-seed/split scope.

2. **Should-fix — Monte Carlo withholding is incomplete in the analysis and bypassed by the figures.**  
   [yaw_pilot_analysis.md:20](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_analysis.md:20); [tools/exp10_summarize.py:990](/home/yixunhu/codespace/xRIR_code/tools/exp10_summarize.py:990).

   There are **two** unresolved headline cells:
   - `cyl_k8 / angles["384"] / T60`: ratio-bound movement `0.108860 > 0.1`.
   - `control_k8 / angles["64"] / T60_abs`: seed statuses disagree—`undefined` versus `denominator uncertain`; seed-0 exact-zero draw fraction is `0.0001`.

   The Markdown/HTML tables correctly disclose both. However, the cylindrical and combined figures label T60 at 270° `impr`, expanded in their footer as “net error improves,” because the renderer ignores `headline.reason`.

   **Minimal fix:** give “unresolved Monte Carlo uncertainty” precedence in figure annotations; name both failure mechanisms in the analysis. Correct the metric count: ten metrics, hence 160 nonzero-angle cells. Regenerate affected assets and hashes.

3. **Should-fix — several numerical generalizations are false or insufficiently scoped.**  
   [yaw_pilot_analysis.md:25](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_analysis.md:25), also line 49.

   Released K1 C50 shift peaks at **180°: 0.2450 dB**, exceeding **0.2275 dB at 45°**. K1 is not uniformly lowest on C50/T60: at 180°, cylindrical gaps are **0.2348 dB / 1.2274%**, versus K1 **0.2450 dB / 1.4372%**.

   “All intervals are narrow (±5%)” excludes the room intervals: released K8 EDT45 is **26.247 ms [10.627, 43.683]** at room level. The general “10–15% less” cylindrical-shift statement also exceeds its examples; acoustic reductions span approximately **6.5–23.9%**.

   **Minimal fix:** state the exceptions, restrict the narrow-interval statement to approximate query-level gap uncertainty, and scope percentage reductions to named metrics/angles.

4. **Should-fix — unchanged queries are counted as improvements.**  
   [yaw_pilot_analysis.md:47](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_analysis.md:47).

   Released K8 EDT45 has **3,195 worse, 2,998 better, and 144 unchanged** queries. Thus “49.6% better” is incorrect.

   **Minimal fix:** report **50.4% worse, 47.3% better, 2.3% unchanged**. The positive/negative mean contributions themselves recompute correctly and support the cancellation discussion.

5. **Should-fix — standalone outputs omit the required K1 training qualification.**  
   [yaw_pilot_results.md:78](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results.md:78); [yaw_pilot_01_results.html:70](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_01_results.html:70), and figure labels/captions.

   These identify only “K = 1.” The analysis contains the qualification, but the standalone results do not.

   **Minimal fix:** label the arm **“trained K = 8, evaluated K = 1”** in the reports and figures. Preserve the restriction against attributing K1/K8 differences solely to reference count.

6. **Should-fix — backend wording claims equality despite measured differences.**  
   [yaw_pilot_analysis.md:19](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_analysis.md:19).

   “No effect on any statistic” contradicts the comparison. EDT45 Δ is **−0.000584838638 s GPU** versus **−0.000582133437 s CPU**, a **0.462555%** relative difference.

   **Minimal fix:** say that point differences are small and all **16 acoustic query-level denominator statuses agree**. Keep the CPU arm explicitly separate; describe its passing acoustic tolerances without implying that the complete CPU protocol replicated exp_03.

7. **Should-fix — “launch commit” is actually completion-time HEAD.**  
   [yaw_pilot_params_set_up.md:20](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_params_set_up.md:20); [tools/exp10_yaw_pilot.py:901](/home/yixunhu/codespace/xRIR_code/tools/exp10_yaw_pilot.py:901).

   Git metadata is captured after inference. The CPU full run started **September 26, 20:40:09 EDT**, but its listed “launch commit,” `f31d8e6`, was created **September 27, 02:26:19 EDT**.

   **Minimal fix:** rename the column “HEAD recorded at completion,” separately record chain-start HEAD from logs, and identify the reviewed implementation commit/hash. Do not rewrite canonical metadata. The evaluator bytes match across these commits.

8. **Should-fix — command and failed-launch records are incomplete under the SOP.**  
   [yaw_pilot_command.md:19](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_command.md:19), line 29; [yaw_pilot_worklog.md:62](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_worklog.md:62).

   Retry/regeneration commands contain `…` or `<queue log>` placeholders. The 20:21 CPU retry, reported smokes, interim summaries, and referenced preview lack complete individual command entries. The initial CPU attempt’s output was discarded before logging; **no retained first-attempt log exists**. The K1 failure log survives but lacks the SOP’s aborted designation.

   **Minimal fix:** append executable commands with environment, script revision, outputs and log paths; register both failed attempts and retries. Explicitly identify reconstructed entries and unrecoverable terminal output. Do not backdate reconstruction or fabricate the missing log.

9. **Should-fix — the commit index omits essential experiment commits.**  
   [commits_yaw_pilot.md:3](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_yaw_pilot_claude/commits_yaw_pilot.md:3), line 73.

   “From the scaffold” starts after the scaffold/plan-review history, omitting `ac06bca` and subsequent early record commits. It also omits **`8009347`**, which committed the submitted results, HTML, assets, checksums and analysis. Unrelated exp_11 commits appear under the exp_10 Coder heading.

   **Minimal fix:** complete the experiment-specific index through the submitted record and identify interleaved unrelated history separately.

**Verified**

- Read the SOP, plan/amendments, three plan reviews, all **seven** code reviews, tooling review, experiment records, and chain/queue/finish logs. Implementation and tooling review closures preceded their respective production use. Previously disclosed process deviations remain relevant disclosures.
- **80 Markdown metric rows and 160 HTML metric rows** match canonical values, intervals, counts, query/room statuses and permitted multiples at displayed precision.
- Independently recomputed **200 GPU summary cells**, including angle zero, from per-sample data: means, Δ/G, masks/counts, paired contributions and reportable point ratios. Maximum absolute discrepancy: **3.33×10⁻¹⁶**.
- Independently reproduced **16 cells’ query and room bootstraps**, each with 10,000 seed-0 draws: all **32** sets of Δ/G/R bounds and statuses matched. Representative points follow; EDT and absolute T60 use ms, C50 uses dB, and T60 uses pp for Δ and baseline-prediction % for G. D = defined, U = denominator uncertain, I = improvement; “—” means no reportable multiple.

| Arm | Angle / metric | Δ | G | Multiple | Query / room |
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
| cyl_k8 | 45° EDT | 1.14709 | 14.2501 | 12.423× | D / D |
| cyl_k8 | 90° C50 | 0.032011 | 0.251351 | 7.852× | D / U |
| cyl_k8 | 180° T60 | −0.304588 | 1.22736 | — | I / I |
| cyl_k8 | 270° T60 abs | −0.464865 | 8.33133 | — | I / U |

- All **32 backend-table rows** match the GPU/CPU canonical summaries. All **12 GPU probe controls** have zero stored waveform/log-spectrum deviations; acoustic control arrays also match their corresponding originals exactly.
- **Replication-qualified arms: GPU `released_k8`, `control_k8`, and `cyl_k8`.** All 120 required probe/full parity cells pass with identical validity masks. Released K1 has no predecessor. CPU released K8 passes its probe but fails full spectral parity and remains a **separate CPU protocol**.
- All **46 SHA256SUMS entries**, **70 run-bound files**, **27 copied JSON artifacts**, and **27 HTML provenance hashes** match. Relative page links resolve; execution/protocol identities agree across all ten executions; all 12 exp_03 source pins match.
- T60 denominators, model-plus-Griffin-Lim qualification, GL-free readouts, and context-only historical bands are present. FLAC qualifications agree with its code: 8,000 samples at 22,050 Hz, AcousticRooms T60 sentinel inclusion, and C50’s `+1e-10` inside the logarithm. No cross-family waveform ranking was found.
