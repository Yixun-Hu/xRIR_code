# exp_10 yaw_pilot — Codex code review, round 1 (full)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="ultra"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt10 --skip-git-repo-check`, stdin closed) · **Reviewed:** commits `023071f..26562f0` (19, Claude Opus 5) on `exp10-yaw-pilot` · **Prompt:** `review_prompts/code_round1_full_prompt.md` · **Raw log:** `yaw_pilot_2026-09-26_18:44:55_codex_code_round1_full.log` · **Date:** 2026-09-26 · **Verdict:** request changes; not approved for merge (blocking: findings 1–10)

**Round-1 verdict: request changes. Merge verdict: not approved.** Reviewed all 19 commits `023071f..26562f0`. The numerical smoke claims reproduce, but integrity checks, acceptance reporting, and several required outputs need fixes.

**Findings**

1. **Blocker — the recorded per-sample hash is never enforced.** [tools/exp10_compare.py:139](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_compare.py:139). Changing one stored `logspec_mad` value to `999.0` still passes `load_run`, `check_online`, summary generation, and summary-input verification. The summariser simply hashes the altered file anew. **Minimal fix:** require and validate `meta.per_sample_sha256` before consuming the file. Also require the complete expected waveform-binding set; deleting an array’s binding currently bypasses its validation.

2. **Should-fix — parity can succeed with required comparisons missing.** [tools/exp10_compare.py:397](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_compare.py:397). Removing historical angle 128, or its C50 metric, produces `ok=True`; missing cells are listed and skipped. This does not satisfy §7’s criteria for every requested angle and required metric. **Minimal fix:** refuse incomplete required coverage or return `ok=False`, distinguishing optional diagnostics.

3. **Should-fix — controls can pass with missing or invalid evidence.** [tools/exp10_summarize.py:502](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_summarize.py:502). One null waveform/log-spectrum deviation is silently discarded by `nanmax`; the control still passes. Removing declared `ctrl_full_turn` also leaves overall `ok=True`. Both were reproduced with otherwise consistent fixture metadata. **Minimal fix:** validate the requested control set, required arrays, lengths, and finite per-query waveform/log-spectrum deviations before applying tolerances.

4. **Should-fix — A3’s uncertainty qualification loses precedence.** [tools/exp10_summarize.py:317](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_summarize.py:317). A 10,000-draw fixture gives an entirely negative seed-0 Δ interval and a seed-1 interval crossing zero. Although `status_agrees=False`, the headline says “predictions shift while net error improves.” **Minimal fix:** check convergence and status agreement before substantive status wording, returning “unresolved Monte Carlo uncertainty.” Also record the observed-Δ-zero reason independently of bootstrap status; currently a null point ratio can lack A3’s required reason.

5. **Should-fix — the canonical summary omits standalone G on the broader population.** [tools/exp10_summarize.py:439](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_summarize.py:439). With gaps `[100,1,2,3]` and an invalid baseline error for the first query, the summary reports only paired `n=3, G=2`; §4’s separately labelled gap-finite result is `n=4, G=26.5`. Broader means exist in the pilot’s `metrics.json`, but disappear from the canonical summary and Markdown. **Minimal fix:** add the separate population, count, estimate, intervals, and convergence without changing paired Δ/G/R.

6. **Should-fix — T60 and pipeline qualifications are missing from the report.** [tools/exp10_summarize.py:335](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_summarize.py:335), [tools/exp10_summarize.py:532](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_summarize.py:532). T60 bars use `%`, and ratios omit the distinction between GT-normalised error change and baseline-prediction-normalised shift. The stored absolute-seconds T60 pair is not summarised. Generated reports also omit R3’s model-plus-Griffin-Lim qualification. **Minimal fix:** include the absolute-seconds comparison, explicit denominators, and the statement that waveform/acoustic gaps measure pipeline sensitivity and GL can amplify magnitude changes.

7. **Should-fix — figures turn unavailable estimates into measured zeros.** [tools/exp10_summarize.py:697](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_summarize.py:697). `value or 0.0` converts null estimates and bounds into zero-height bars. An empty EDT cell remained null in JSON/CSV but appeared as zero change and zero shift in the figure. **Minimal fix:** omit unavailable bars/error bars and identify the affected angle as unavailable with its count; preserve actual zero estimates.

8. **Should-fix — comparator output is not strict JSON.** [tools/exp10_compare.py:421](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_compare.py:421). An all-invalid common population produces literal `NaN` in the written parity report; a strict parser rejects it. **Minimal fix:** convert nonfinite values to null and serialize with `allow_nan=False`.

9. **Should-fix — Markdown omits required results already computed in JSON.** [tools/exp10_summarize.py:816](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_summarize.py:816), [tools/exp10_summarize.py:856](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_summarize.py:856). Metric 2 omits `log_mse` and `loss`; the room table omits room ratio estimates/intervals and convergence. **Minimal fix:** render those stored results, including their units and separate room-level qualifications.

10. **Should-fix — GPU timing assigns unfinished inference to inversion.** [tools/exp10_yaw_pilot.py:538](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_yaw_pilot.py:538). Inference timing stops before CUDA synchronization; the subsequent `.cpu()` in `invert` waits for outstanding work inside the inversion timer. The required GPU stage breakdown is therefore unreliable, although total elapsed time remains meaningful. **Minimal fix:** synchronize at stage boundaries or use synchronized CUDA events. This finding is from source inspection; no GPU was used.

11. **Nit — malformed Markdown controls header and ineffective assertion.** [tools/exp10_summarize.py:865](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_summarize.py:865) contains unescaped absolute-value pipes, creating fourteen header cells for eight data columns. [tests/test_exp10_summarize.py:528](/home/yixunhu/codespace/xRIR_code_wt10/tests/test_exp10_summarize.py:528) ends its content assertion with `or True`. **Minimal fix:** escape/relabel the header and remove the unconditional bypass.

12. **Nit — the declared cylindrical historical band remains deferred.** [tools/exp10_summarize.py:539](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_summarize.py:539). Omitting a context-only band does not invalidate the statistics, but the cylindrical figure is incomplete against the plan. **Minimal fix:** add the Planner-supplied, verified cylindrical SDs before producing final cylindrical figures.

**Verified**

- **Scope:** exactly 19 commits and **seven new files**, including `tests/exp10_fixture.py`. The Coder prompt explicitly permits fixtures; the six-file claim is inaccurate, but no pre-existing tracked file changed.
- **Static checks:** Python **3.8.20** compilation of all seven files passed; both range and working-tree `git diff --check` passed. Live `verify_exp03_pins` matched **12/12** reviewed files. Altering a copied pinned file correctly caused refusal.
- **Suites:** `tests/test_exp10_*.py`: **112 passed**. Exp_03/exp_04 record suites: **80 passed, 1 skipped**.
- **Inference and metrics:** executed composition/identity, audio-preservation, restoration, inversion/seeding/padding, and canonical-batching tests. Fresh saved float32 waveforms recomputed online waveform/acoustic gaps exactly. Raw measurements reconstructed the canonical acoustic errors and gaps exactly. A4 `log_mse` and loss-component paths match the pinned definitions.
- **Bootstrap:** independent query and room oracles verified shared masks/draws, unequal room sizes, multiplicities, and query weighting. All-zero Δ, a single exact-zero draw, undefined precedence, and zero-width convergence rules passed. Duplicate executions and same-device batch/implementation mismatches were refused.
- **Parity guards:** missing/duplicate/reordered queries and manifest mismatch were refused. Injecting one nonzero-angle NaN correctly produced **nonreplication (validity mismatch)**.
- **Outputs:** fresh run JSON was strict, required metadata was present, hashes matched, and completion metadata was written last. Generated JSON/Markdown/CSV/PNG/PDF; all fields of 16 plotted CSV rows matched canonical JSON. Inspected signed bars, GL-free panel, and “context only” band label. Changed inputs after summary construction were refused.

Two independent batch-0 CPU smokes covered 16 queries, five angles, and three fresh controls. All five waveform arrays and all per-query metrics/controls were identical between executions, with matching protocol IDs and distinct execution IDs.

| Check | Observed |
|---|---:|
| k=0 gaps | Exactly 0 |
| Control waveform/log-spectrum/acoustic differences | Exactly 0 |
| Offline waveform/acoustic gap differences | Exactly 0 |
| Exp_03 EDT error maximum difference | 0 |
| Exp_03 T60 error maximum difference | 0 |
| Exp_03 C50 error maximum difference | `4.6491623e-5 dB` |
| Exp_03 logspec_mad maximum difference | `6.2820358e-6` |

All compared validity masks matched; paired signs and tolerances passed. The isolated repeat took **65.0 s**, averaging **6.225 s per batch-16 forward** and **47.2 ms per query/inversion**. Thus the numerical and GL-timing claims reproduce; the claimed 4.4-second forward timing was not reproduced.

Evidence is retained under [.review-round1/](/home/yixunhu/codespace/xRIR_code_wt10/.review-round1), including the [smoke audit](/home/yixunhu/codespace/xRIR_code_wt10/.review-round1/main/smoke-audit.log) and [adversarial guard results](/home/yixunhu/codespace/xRIR_code_wt10/.review-round1/compare/adversarial_results.json). Tracked files and checkpoint symlinks were left untouched.

Declared deviations 1–6 and 9–11 are reasonable implementation choices; the control representation still needs finding 3’s validation. Item 7 is safely deferred context, not completed functionality. Commit size is a process concern rather than a correctness blocker; additional commits also exceed the guideline, so item 8 understates that deviation.

**Exact merge-blocking list: findings 1–10. Findings 11–12 are nonblocking nits. Round 1 remains open; not approved for merge.**
