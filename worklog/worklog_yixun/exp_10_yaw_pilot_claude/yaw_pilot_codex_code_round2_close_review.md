# exp_10 yaw_pilot — Codex code review, round 2 (close)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="ultra"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt10 --skip-git-repo-check`, stdin closed) · **Reviewed:** commits `02b567a..d3e927b` (12 fixes, Claude Opus 5) · **Prompt:** `review_prompts/code_round2_close_prompt.md` · **Raw log:** `yaw_pilot_2026-09-26_19:41:48_codex_code_round2_close.log` · **Date:** 2026-09-26 · **Verdict:** request changes; not approved for merge (blocking: figure footer overlap = round-1 finding 6 residual; nits 2–3)

**Round-2 verdict: request changes. Merge verdict: not approved.**

Reviewed the 12 fixes in `02b567a..d3e927b` on `exp10-yaw-pilot`. The numerical and integrity fixes pass independent verification. One required figure qualification remains unreadable.

**Remaining findings**

1. **Should-fix — round-1 finding 6 remains open: overlapping figure footers obscure the qualification.** [tools/exp10_summarize.py:999](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_summarize.py:999). The status footnote overlaps the new pipeline/T60 caption. The renderer audit measured overlap across the entire status block; this also occurs in the fresh [CPU-smoke figure](/home/yixunhu/codespace/xRIR_code_wt10/.review-round2/main/summary/yaw_pilot_gaps_released_k8_smoke.png). JSON and Markdown contain the correct qualifications, but the figure does not communicate them legibly. **Minimal fix:** reserve sufficient caption space and position the legend and text blocks without overlap; verify rendered bounds and inspect the PNG/PDF.

2. **Nit — round-1 finding 7’s missing-bound case still draws error-bar caps.** [tools/exp10_summarize.py:916](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_summarize.py:916). A point of `1` with null bounds draws a zero-length interval at `1`. The [test](/home/yixunhu/codespace/xRIR_code_wt10/tests/test_exp10_summarize.py:927) checks only bar count. **Minimal fix:** add error bars only where both bounds exist, and assert absence of error-bar artists otherwise. Null estimates are correctly omitted and actual zeros preserved. Ordinary generated cells did not exhibit finite points with missing bounds, so this residual is nonblocking.

3. **Nit — round-1 finding 12’s combined figure mislabels cylindrical bands.** [tools/exp10_summarize.py:1025](/home/yixunhu/codespace/xRIR_code_wt10/tools/exp10_summarize.py:1025). `handles = handles or drawn` retains the first arm’s legend. A SimpleViT+CylindricalViT figure therefore labels all gray bands “SimpleViT, exp_04,” despite plotting the correct cylindrical values. **Minimal fix:** use a generic shared legend and model-specific labels beside each arm’s row. Individual cylindrical figures are correctly labelled.

**Round-1 closure audit**

| Finding | Fix commit | Independently verified result |
|---|---|---|
| 1 | `6bfd393` | **Closed.** Changing `logspec_mad` to `999` is refused by all five consumers. Deleting a waveform binding is independently refused, including with otherwise consistent metadata. |
| 2 | `6685344` | **Closed.** Missing required angle/metric yields `ok=False` and the missing list; optional diagnostics remain optional. |
| 3 | `e19a6f5` | **Closed.** Null deviations, missing declared controls, undeclared controls, missing arrays, and wrong lengths fail. |
| 4 | `8202d22` | **Closed.** Original 10,000-draw reproduction now reports “unresolved Monte Carlo uncertainty.” Seed-0 Δ CI is entirely negative; seed-1 crosses zero. Null point ratios carry reasons. |
| 5 | `08ece38` | **Closed.** Paired `n=3, G=2` appears beside broader `n=4, G=26.5`, with query/room intervals and convergence. |
| 6 | `c5c349e` | **Partially closed.** Denominators, `T60_abs`, and GL qualifications are present; figure layout remains blocking as above. |
| 7 | `3781f7a` | **Main reproduction closed.** Null produces no bar; actual zero produces a bar. Missing-bound residual is nit 2 above. |
| 8 | `4ac265b` | **Closed.** Original and stronger all-invalid populations serialize as strict JSON with nulls. |
| 9 | `a4d0584` | **Closed.** Markdown includes `log_mse`/`loss` with units and room ratios, intervals, convergence, and qualification. |
| 10 | `941ad2f` | **Closed.** Source and stub tests verify synchronization at stage boundaries, including exception cleanup; CPU synchronization remains absent. |
| 11 | `4b33da1` | **Closed.** Controls header/separator/data each have ten cells; `or True` is gone. |
| 12 | `d3e927b` | **Numerical fix verified.** Cylindrical SDs and `(backbone, K)` selection are correct; combined-legend residual is nit 3 above. |

**Verified**

- **Suites:** all `tests/test_exp10_*.py`: **156 passed**. Exp_03/exp_04 record suites: **80 passed, 1 skipped**.
- **Static/scope:** Python **3.8.20** compilation passed for all seven files. Range and working-tree `git diff --check` passed. Exactly the seven permitted files changed.
- **Regressions:** composition, inversion/seeding/padding, canonical batching, raw acoustic reconstruction, shared bootstrap draws/masks, unequal-room multiplicities/query weighting, identity guards, parity subset alignment, and A1 validity-mask rejection passed. Live exp_03 pins matched **12/12**; copied-source tampering was refused.
- **Reports:** generated JSON/Markdown/CSV/PNG/PDF; all fields of 16 independently audited plotted CSV rows matched canonical JSON. Verified cylindrical SDs against the canonical five-seed record.
- **CPU smoke:** batch 0, 16 queries, five angles, three fresh controls; `check-online`, `parity-exp03`, and the default 10,000-draw summariser completed. All stored waveforms and numerical per-sample results exactly matched the prior isolated round-1 smoke.

| Smoke check | Observed |
|---|---:|
| k=0 gaps; control deviations; offline gap differences | Exactly 0 |
| Exp_03 EDT/T60 maximum differences | 0 |
| Exp_03 C50 maximum difference | `4.6491623e-5 dB` |
| Exp_03 logspec maximum difference | `6.2820358e-6` |
| Validity mismatches | 0 |
| Required comparisons missing | `[]` |
| CPU batch-16 forward | **5.979 s** |
| Inversion per query/angle | **45.0 ms** |

The declared methodological/testing deviations are sound. The original stochastic A3 reproduction also passes, supporting the deterministic regression fixture.

The disposable clone’s **`--no-ff --no-commit` merge onto main `9c7dd1c`**, a descendant of `e755792`, completed without conflicts and added exactly the seven permitted files. The clone was deleted.

Evidence is retained under [.review-round2/](/home/yixunhu/codespace/xRIR_code_wt10/.review-round2), including the [smoke audit](/home/yixunhu/codespace/xRIR_code_wt10/.review-round2/main/smoke-audit.log), [adversarial results](/home/yixunhu/codespace/xRIR_code_wt10/.review-round2/compare/adversarial_results.json), and [merge rehearsal](/home/yixunhu/codespace/xRIR_code_wt10/.review-round2/main/merge.log). Tracked files and checkpoints remained untouched; no GPU was used.

**Exact blocking list: remaining finding 1 above—round-1 finding 6’s overlapping figure qualification. Findings 2–3 are nonblocking nits. Not approved for merge.**
