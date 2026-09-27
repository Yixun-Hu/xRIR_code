# exp_10 yaw_pilot — Codex code review, round 5 (close; Planner tooling)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="ultra"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt10 --skip-git-repo-check`, stdin closed) · **Reviewed:** `4ad0652..f53b172` (8 commits, Claude Opus 5) · **Prompt:** `review_prompts/code_round5_close_prompt.md` · **Raw log:** `yaw_pilot_2026-09-27_05:32:33_codex_code_round5_close.log` · **Date:** 2026-09-27 · **Verdict:** request changes; not approved (blocking: termination-time rollback; rendered-arm payload binding)

**Reviewer:** OpenAI Codex `gpt-6-astra` (codex-cli 0.154.0, read-only review) · **Date:** 2026-09-27  
**Reviewed:** eight commits `4ad0652..f53b172`, after `68ddf1a`, on `exp10-yaw-pilot`.

**Round-5 verdict: request changes. Merge verdict: not approved. The finish script must not yet build the final record.**

Two blocking findings remain. Paths below are relative to `worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/`.

1. **Blocker — termination during publication still leaves mixed generations and deletes the backups.**  
   [`scripts/exp10_finish.sh:49`](/home/yixunhu/codespace/xRIR_code_wt10/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/scripts/exp10_finish.sh:49). The original final-rename failure now restores all three artifacts. However, sending SIGTERM immediately before that rename leaves **new assets, new HTML, old Markdown**. EXIT cleanup deletes both backups because the destinations exist; it never calls `restore()`. Independently reproduced in scratch; process terminated with SIGTERM and no backups remained.  
   **Minimal fix:** track publication-started/committed state and invoke all-three rollback on EXIT/TERM/INT whenever publication is uncommitted, preserving original absence too. Retain backups if restoration fails. This is residual round-4 finding 2. [Evidence](/home/yixunhu/codespace/xRIR_code_wt10/.review-round5/finish/results.json).

2. **Should-fix, merge-blocking — rendered arm payloads are not bound to the verified inputs.**  
   [`validate_runs.py:292`](/home/yixunhu/codespace/xRIR_code_wt10/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/validate_runs.py:292), also `:351` and `:497`. Input entries now bind to live files, but their identities are never compared with the corresponding rendered `arms` entries. Transplanting another independently bound summary’s complete arm payload succeeds in **both renderers, exit 0**: the page displays CPU execution B while provenance and successful online/parity evidence identify CUDA execution A. Probe validation also accepts contradictory arm protocol/payload identities.  
   **Minimal fix:** compare every rendered arm’s execution/protocol identity and embedded metadata—including arm and per-sample digest—with its unique input and hash-verified live metadata. Apply the same check to probe arm entries. Add transplant and contradictory-probe regressions. This is residual round-4 finding 4. [Evidence and rendered outputs](/home/yixunhu/codespace/xRIR_code_wt10/.review-round5/shell/arm_binding_consistent/results.json).

**Verified**

All execution used **Python 3.8.20**, with GPUs disabled or mocked. Tracked files remain unchanged; writes stayed under `.review-round5/`. Live `ckpt/exp10/released_k8_all` was not accessed.

Each requested round-4 reproduction was rerun:

| Prior finding | Observed result |
|---|---|
| 1 — GPU reports in CPU record | Refused, **exit 2**; existing publication preserved. |
| 2 — final-rename failure | **Exit 9**, all three artifacts restored byte-for-byte. Termination variant remains finding 1 above. |
| 3 — foreign combined PNG | Foreign image ignored; regenerated PNG and PDF match a second regeneration byte-for-byte. |
| 4 — incomplete inputs / contradictory supplements | Missing/duplicate inputs, bad hashes, and contradictory execution/protocol/payload identities refused by both renderers, **exit 2**. Arm-payload gap remains finding 2 above. |
| 5 — alpha `.05` versus `.5` | Canonical reproduction refused, **exit 2**. Matching `.05` control passes and records all four statistical settings. |
| 6 — concurrent-chain pin race | B refuses with **2**; A refuses changed evaluator with **8**. Pins unchanged, mode `0444`; full stage never launches. |
| 7 — T60 qualification pipes | Splitter recovers exactly three cells, including complete Δ/G definitions and units. |

Additional verification:

- **132 record-tool tests passed**; exp_10/exp_03 regression **193 passed, 1 skipped**.
- All four scripts passed `bash -n`; all six scoped Python files passed `py_compile`; round, working-tree and merge `git diff --check` passed.
- Regenerated and inspected interim [HTML](/home/yixunhu/codespace/xRIR_code_wt10/.review-round5/root/render/interim/page.html), [Markdown](/home/yixunhu/codespace/xRIR_code_wt10/.review-round5/root/render/interim/page.md), and combined figure. Qualifications, room status beside multiples, full hashes and links verified.
- Round-3 footer, missing-bound error-bar and legend closures passed. Approved summarizer unchanged; exp_03 source pins **12/12** matched. Twenty prior shell gate/wrapper reproductions passed.
- Finish scratch rehearsal at **6,337 queries** published **46 checksum-verified assets**, removed stale assets and retained CPU parity `false`. Missing/incomplete evidence and intermediate failures preserved the previous publication. Statistics/evaluation were stubbed; approved figure functions were exercised.
- Disposable `--no-ff --no-commit` merge onto MAIN **`b924cd693d29f96d8bdfb2b38037e50ddbf97f98`** succeeded; exactly ten authorized files changed. Round 5 itself changes eight scoped files. Clone deleted. [Static/merge evidence](/home/yixunhu/codespace/xRIR_code_wt10/.review-round5/checks/static_merge.json).

**Exact blocking list: findings 1 and 2 above. Round 5 remains open; not approved for merge or final-record construction.**
