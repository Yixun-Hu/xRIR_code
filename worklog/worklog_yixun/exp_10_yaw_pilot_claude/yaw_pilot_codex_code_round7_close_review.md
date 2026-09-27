# exp_10 yaw_pilot — Codex code review, round 7 (close; finish-script rollback)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="ultra"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt10 --skip-git-repo-check`, stdin closed) · **Reviewed:** `2122f66, be6a192, 3b0f28c` (Claude Opus 5) · **Prompt:** `review_prompts/code_round7_close_prompt.md` · **Raw log:** `yaw_pilot_2026-09-27_07:39:36_codex_code_round7_close.log` · **Date:** 2026-09-27 · **Verdict:** APPROVED for merge; finish script may build the final record; one nonblocking nit (committed-state closing message)

**Reviewed:** `2122f66`, `be6a192`, `3b0f28c` after `058726b`, on `exp10-yaw-pilot` · **Date:** 2026-09-27.

**Round-7 verdict: approved, with one nonblocking nit. Merge verdict: approved for merge. Exact blocking list: empty.** The reviewed `scripts/exp10_finish.sh` **may now build the final record after merging onto MAIN**.

Both round-6 defects are closed:

- `2122f66` / `3b0f28c`: all backups are completed, verified and marked while `idle`; `started` immediately precedes the first publication mutation. Restoration requires a verified-copy marker.
- `be6a192`: rollback ignores TERM/INT/HUP throughout and marks `rolled_back` only after processing all three artifacts.

1. **Nit — misleading closing message after committed publication.**  
   [`scripts/exp10_finish.sh:167`](/home/yixunhu/codespace/xRIR_code_wt10/worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/scripts/exp10_finish.sh:167). SIGTERM during backup deletion after `committed` leaves the complete **new** publication, but cleanup claims the previous publication remains. Independently reproduced: exit **143**, all **46** asset checksums verify, no backups remain. This pre-existing diagnostic issue does not compromise artifacts.  
   **Minimal fix:** use a committed-state-specific closing message. [Evidence](/home/yixunhu/codespace/xRIR_code_wt10/.review-round7/finish/29_sigterm_after_commit/result.json).

**Verified**

All **28 independent finish cases passed**:

| Case | Observed result |
|---|---|
| Partial backup `cp` failure; truncated copy reporting success | **12**; originals byte-identical; no leftovers |
| SIGTERM during partial/completed report backup or asset backup | **143**; originals byte-identical |
| Final-rename failure | **9**; all three originals restored |
| TERM / INT / HUP before final rename | **143 / 130 / 129**; complete restoration |
| TERM / INT / HUP during explicit rollback | **9** each; complete restoration; backups removed |
| Markdown originally absent; all three originally absent | Original absence preserved |
| Unwritable restoration destination | Assets restored; both original reports preserved byte-identically in named backups; accurate failure warning |
| Missing/incomplete runs across all four arms; GPU reports filed as CPU evidence; failed probe controls | **2**; publication unchanged |
| Successful finish; foreign PNG input | **46 checksum-verified assets**; stale assets removed; CPU parity remains `false` |

Signal delivery was confirmed; explicit rollback’s signal mask independently showed all three signals ignored. Exit **9** preserves the triggering publication failure. These were synthetic **6,337-query** runs: evaluation/statistics were stubbed; actual validators, renderers and approved figure functions executed. [Finish evidence](/home/yixunhu/codespace/xRIR_code_wt10/.review-round7/finish/results.json).

- **181 record-tools tests passed.** Requested exp_10 + exp_03 regression: **193 passed, 1 skipped**; the skip requires two GPUs. [Test evidence](/home/yixunhu/codespace/xRIR_code_wt10/.review-round7/checks/test_results.json).
- Four scripts passed `bash -n`; six Python files passed `py_compile`; round, working-tree and merge `git diff --check` passed. Exp_03 pins **12/12** matched.
- Disposable `--no-ff --no-commit` merge onto MAIN **`91d5b22e0e9ab0dc781b0929003940c5158f54a6`** succeeded with exactly ten authorized tooling files changed. **Clone deleted.** Round 7 itself changes only the finish script and record-tools tests; the fixture and interim-generation code are unchanged. [Static/merge evidence](/home/yixunhu/codespace/xRIR_code_wt10/.review-round7/checks/static_merge.json).
- Python **3.8.20**, no GPU; tracked files remained unchanged. All review writes stayed under `.review-round7/`.
