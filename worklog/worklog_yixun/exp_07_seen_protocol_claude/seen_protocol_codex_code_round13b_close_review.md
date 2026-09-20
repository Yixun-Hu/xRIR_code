# exp_07 seen_protocol — Codex code review, round 13b (close: independent profile-digest assertion)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` (`codex exec -c 'model_reasoning_effort="xhigh"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt07 --skip-git-repo-check`, stdin closed) · **Reviewed commit:** `981312d` on `exp07-window` (parent `d7280bd`) · **Prompt:** `review_prompts/code_round13b_close_prompt.md` · **Raw log:** `seen_protocol_2026-09-20_05:53:01_codex_code_round13b_close.log` · **Date:** 2026-09-20

**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-20

**Reviewed:** `981312d8dcb448cca1bf670dc70e604b4a7b8b3c`, parent `d7280bd`.

**Round-13b verdict:** approve.  
**Merge verdict:** **approved for merge**.  
**Exact blocking list:** **none (`[]`)**.

1. **Should-fix from round 13 — CLOSED.** [tests/test_exp07_table.py:665](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_exp07_table.py:665) computes the expected digest from `built.profile` before production. Lines 671–672 independently compare the returned and written digests against it. The requested minimal fix is implemented; no further change is required.

No new blockers, should-fixes, or nits.

**Verified**

- **Scope:** `git diff d7280bd..981312d` changes only `tests/test_exp07_table.py`, **+19/−1**. The production diff against `main`, excluding tests/worklog, is empty.
- **Canonicalization:** [the helper](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_exp07_table.py:644) matches `tools/exp07_table.py:374,405–407`: `json_value`, JSON round-trip, sorted keys, compact separators, `allow_nan=False`, encoding and SHA-256. Captured producer/helper payloads were **byte-identical**; the hashing expressions also match structurally.
- **Helper reuse:** `exp07_profiles.profile_digest(name)` hashes the registered profile, not the mutated fixture. `exp04_profiles.profile_digest(name)` has the same limitation. `paired_compare._digest(value)` accepts a value but omits the round-trip; integer keys `{2,10}` demonstrate different serialization. It could hash an already-normalized literal, but reuse is unnecessary here.
- **Future refactors:** duplicating this serialization is acceptable for the focused regression. Changes affecting the fixture’s digest will fail the test. Unexercised inputs or changes to the shared `json_value` converter can escape detection; this test does not independently validate that converter.

The disposable-clone sabotage checks established:

| Producer behavior | Result |
|---|---|
| Return and write all-zero digest | Fails at line **671** |
| Same sabotage with parent’s test | **Passes**, reproducing the original gap |
| Hash a copy with a changed arm label | Fails at line **671** |
| Corrupt only the written digest | Fails at line **672** |
| Corrupt only the returned digest | Fails at line **671** |

Thus neither assertion is tautological.

All execution used **Python 3.8.20**, hidden GPUs, `PYTHONDONTWRITEBYTECODE=1`, pytest `-p no:cacheprovider`, and scratch-confined temporary paths.

| Check | Result |
|---|---|
| `python -m pytest tests/test_exp07_table.py -q -p no:cacheprovider` | **59 passed** |
| Record suites `07 → 04 → 03`, then reverse | **235 passed, 1 skipped each** |
| `bash …/seen_protocol_results_assets/static_checks.sh` | **Exit 0:** 334 passed; four-suite orders **330 passed, 1 skipped each** |

The skip requires two visible GPUs.

- **Preservation:** all 30 inspected source closures exclude tests. All six exp_07 approval hashes and the live calibration producer hash match. No closure, pin, bound-record artifact, or exp_04/exp_06 checker was changed.
- **Hygiene:** Python 3.8 compilation and both working-tree/commit `git diff --check` pass. All **1,552 tracked file hashes** remain unchanged; no status changes occurred outside `.review-round13b/`.
- **Merge:** `git merge --no-ff --no-commit 981312d` onto main `4ac17481aa112c761f93f777c557b70af61a0a61`—a descendant of `35631a2`—was conflict-free and staged only the reviewed test file. The merged positive test passed. The clone and generated temporary fixtures were deleted.

Evidence: [verification summary](/home/yixunhu/codespace/xRIR_code_wt07/.review-round13b/summary.json), [sabotage and merge log](/home/yixunhu/codespace/xRIR_code_wt07/.review-round13b/logs/clone-checks.log), [static checks](/home/yixunhu/codespace/xRIR_code_wt07/.review-round13b/logs/static.log), [pin audit](/home/yixunhu/codespace/xRIR_code_wt07/.review-round13b/logs/pins.log).
