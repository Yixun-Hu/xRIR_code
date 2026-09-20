# exp_07 seen_protocol — Codex code review, round 13 (close: hermetic unapproved-profile test)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` (`codex exec -c 'model_reasoning_effort="xhigh"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt07 --skip-git-repo-check`, stdin closed) · **Reviewed commit:** `c0155d4` on `exp07-window` (parent `46bb84d` = main) · **Prompt:** `review_prompts/code_round13_close_prompt.md` · **Raw log:** `seen_protocol_2026-09-20_04:18:39_codex_code_round13_close.log` · **Date:** 2026-09-20

**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-20

**Round-13 verdict:** approve with changes (nonblocking).  
**Merge verdict:** **approved for merge**.  
**Blocking findings:** **none**.

1. **Should-fix, nonblocking — [tests/test_exp07_table.py:654](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_exp07_table.py:654): independently validate the profile digest.**  
   The assertion compares the written digest with the returned digest, so both can be wrong together. In the isolated clone, changing the producer to return and write an all-zero digest still let the new positive test pass. Minimal fix: independently compute the canonical SHA-256 of `built.profile` and compare both digests against it. This is an assertion gap; the refusal fix works, and existing downstream record validation independently checks profile digests.

**Verified**

- **Scope:** `c0155d44e23c8ddb5de2d6dec1d62e6d0536c170` has parent `46bb84d`. Exactly one file changed: `tests/test_exp07_table.py`, **+41/−3**.
- **Approval seam:** `main → build_table → admit` resolves approval through `tools.exp07_table.load_approved_digests` at line 234. Shared admission receives that approval explicitly; it does not reload real pins.
- **Non-vacuous refusal:** the isolated all-null record passes through the real loader. Tracing identifies the exception as `admit`’s refusal at line 258. JSON, Markdown, and provenance sidecar remain absent. Removing the monkeypatch makes the test fail with `unregistered checkpoint/K`; the original parent test fails likewise.
- **Positive execution:** tracing confirms real `main → build_table → admit → write_outputs → exp07_record.write_outputs`. JSON, Markdown, and sidecar are written. Suppressing publication makes the test fail; supplying a wrong producer identity is refused. The digest limitation is finding 1.
- **Collection safety:** targeted collections and both relevant orders load one `test_exp04_profiles` module object. Full-suite collection contains two pre-existing bare/package-qualified aliases, already introduced by exp_05/exp_07 imports in the parent; this commit does not introduce that duplication.

All execution used **Python 3.8.20**, hidden GPUs, disabled pytest caching, and scratch-confined temporary files:

| Verification | Result |
|---|---:|
| Table + record files | **214 passed**: 59 + 155 |
| Exact three-record-suite orders, both directions | **235 passed, 1 skipped each** |
| `static_checks.sh` | **exit 0**: 334 passed; both four-suite orders 330 passed / 1 skipped |
| Clean clone at `c0155d4`, filled committed pins | **59 passed** |
| Clone with historical all-null approval committed | **59 passed** |

The skip requires two visible GPUs. Neither clone run used `--allow-dirty`.

- **Closure preservation:** 30 source closures inspected; none contains tests. All six exp_07 approval closure hashes and the live calibration hash match. Bound records, approval files, and exp_04/exp_06 checker code are unchanged.
- **Hygiene and merge:** Python 3.8 compilation and `git diff --check` pass. Tracked working-tree/index diffs remain empty. The `--no-ff --no-commit` merge onto main `6a5384b` was conflict-free and changed only the reviewed test file. The clone was deleted; reviewer writes stayed under `.review-round13/`.

Evidence: [static checks](/home/yixunhu/codespace/xRIR_code_wt07/.review-round13/static.log), [clone and merge checks](/home/yixunhu/codespace/xRIR_code_wt07/.review-round13/clone-checks.log), [behavior traces](/home/yixunhu/codespace/xRIR_code_wt07/.review-round13/behavior-probe.log).
