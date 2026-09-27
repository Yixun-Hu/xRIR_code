# exp_10 yaw_pilot — Codex code review, round 6 (close; Planner tooling)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="ultra"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt10 --skip-git-repo-check`, stdin closed) · **Reviewed:** `40cf3fb, 1fadba3, 6ebccea, ca2dfc8` (Claude Opus 5) · **Prompt:** `review_prompts/code_round6_close_prompt.md` · **Raw log:** `yaw_pilot_2026-09-27_06:40:47_codex_code_round6_close.log` · **Date:** 2026-09-27 · **Verdict:** request changes; payload binding closed; two rollback defects in the finish script remain blocking

**Reviewer:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec`, read-only review) · **Date:** 2026-09-27  
**Reviewed:** `40cf3fb`, `1fadba3`, `6ebccea`, `ca2dfc8`, after `6dbb0be`, on `exp10-yaw-pilot`.

**Round-6 verdict: request changes. Merge verdict: not approved. The finish script must not yet build the final record.**

The payload-binding blocker is closed. The requested final-rename rollback now works, but two rollback defects remain.

1. **Blocker — failed or interrupted backup creation destroys the intact original.**  
   [`scripts/exp10_finish.sh:239`](/home/yixunhu/codespace/xRIR_code_wt10/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/scripts/exp10_finish.sh:239), also lines 242 and 77–79. `PUB_STATE=started` precedes the report-backup copies. If `cp` leaves a partial backup, `restore()` treats its existence as proof of completeness, deletes the intact published report, and installs the partial copy.

   Independently reproduced with both copy failure and SIGTERM: HTML changed from `OLD HTML\n` to `OLD `; exits **44/143**; no backups remained. The closing message incorrectly claimed the previous publication was restored. [Evidence](/home/yixunhu/codespace/xRIR_code_wt10/.review-round6/finish/verification.json).

   **Minimal fix:** complete both report backups while publication remains `idle`; enter `started` immediately before the first publication mutation. Never restore an incomplete copy over an unchanged original. Add partial-copy failure and interruption regressions.

2. **Should-fix, merge-blocking — SIGTERM interrupts an explicit rollback before all three artifacts are restored.**  
   [`scripts/exp10_finish.sh:65`](/home/yixunhu/codespace/xRIR_code_wt10/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/scripts/exp10_finish.sh:65), also lines 98, 103 and 250. Direct `restore()` calls mark `rolled_back` immediately, while signal handlers remain enabled. After an injected final-rename failure, SIGTERM during successful asset restoration exits **143**; EXIT cleanup skips the remaining restoration, leaving **old assets, new HTML, old Markdown**. Original report backups survive and diagnostics correctly identify incomplete restoration. [Evidence](/home/yixunhu/codespace/xRIR_code_wt10/.review-round6/finish/13_sigterm_during_restore/result.json).

   **Minimal fix:** suppress TERM/INT/HUP throughout every rollback entry path, and mark rollback complete only after processing all three artifacts. Add this interruption regression.

**Verified**

Execution used **Python 3.8.20**, with GPUs disabled. Tracked files remain unchanged; writes stayed under `.review-round6/`. Live `ckpt/exp10/released_k8_all` was not accessed.

| Round-6 target | Commits | Independent result |
|---|---|---|
| Termination rollback | `40cf3fb`, `6ebccea`, `ca2dfc8` | Requested final-rename, absence and failed-restore cases pass; findings 1–2 above remain. |
| Rendered arm/probe binding | `1fadba3` | Complete execution-B arm transplant and contradictory probe payload refused by **both renderers, exit 2**. Untouched control renders. |

Additional adversarial checks:

| Check | Observed result |
|---|---|
| SIGTERM immediately before final rename | **143**; all three artifacts byte-identical; no backups left. |
| Markdown originally absent; all three originally absent | Original absence preserved, including when rename completes before signal handling. |
| Restoration destination unwritable | Original reports retained byte-identically in named backups; accurate failure message. |
| Final-rename failure | **9**; complete restoration. |
| SIGINT / SIGHUP before final rename | **130 / 129**; complete restoration. HUP tested after resetting the harness’s inherited `SIG_IGN`. |
| GPU reports filed as CPU evidence | **2**; publication preserved. |
| Foreign PNG | Ignored; regenerated PNG/PDF match independent regeneration. |
| Contradictory supplements / missing or duplicate inputs | Both renderers refuse, **2**. |
| Alpha `.05` versus `.5` | Refused, **2**. |
| Concurrent pin race | B exits **2**, A exits **8**; pins unchanged, mode `0444`; no full launch. |
| T60 qualification pipes | Exactly three complete Markdown cells. |

- **175 record-tool tests passed**; exp_10/exp_03 regression **193 passed, 1 skipped**. The skip requires two GPUs. [Test evidence](/home/yixunhu/codespace/xRIR_code_wt10/.review-round6/checks/test_results.json).
- All four scripts passed `bash -n`; six Python files passed `py_compile`; round, working-tree and merge `git diff --check` passed. Exp_03 pins **12/12** matched.
- Regenerated and inspected interim [HTML](/home/yixunhu/codespace/xRIR_code_wt10/.review-round6/root/render/interim/page.html), [Markdown](/home/yixunhu/codespace/xRIR_code_wt10/.review-round6/root/render/interim/page.md), and combined figure. Qualifications, room statuses, full hashes and links passed.
- Scratch finish at **6,337 queries** published **46 checksum-verified assets**, removed stale assets and preserved CPU parity `false`. Evaluation/statistics were stubbed; actual validators, renderers and approved figure functions ran.
- Disposable `--no-ff --no-commit` merge onto MAIN **`80276a6074f6211eecabf7fbe7cb064590ae605d`** succeeded with exactly ten authorized files changed. Round 6 itself changes three authorized files. Clone deleted. [Static/merge evidence](/home/yixunhu/codespace/xRIR_code_wt10/.review-round6/checks/static_merge.json).

**Exact blocking list: findings 1 and 2 above. Round 6 remains open; not approved for merge or final-record construction.**
