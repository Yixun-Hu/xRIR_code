# exp_07 seen_protocol — Codex code review, round 10 close (test-only): APPROVED

**Reviewer:** OpenAI Codex `gpt-6-astra` (Ultra), `codex exec -s read-only -C /home/yixunhu/codespace/xRIR_code_wt07`, prompt `review_prompts/code_round10_close_prompt.md`, log `seen_protocol_2026-09-16_02:33:01_codex_code_round10_close.log`. **Scope:** `e6b70e0..cf46275`. **Date:** 2026-09-16 02:33–02:47 EDT.

**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-16

**Verdict: approve. Blocking findings: none.**

**Verified:**

- `git diff e6b70e0..cf46275`: one commit, 21 insertions / 2 deletions, only `tests/test_exp07_parity.py`. No production or pinned files changed; `git diff --check` passes.
- All 16 existing test functions are AST-identical to the base version. No existing assertion was weakened.
- The fixture patches `tools.provenance.git_state`, preserving the production `checked_git_state` refusal. Temporarily replacing its `raise` with `pass` in the clone produced **1 failed, 18 passed**: the dirty-refusal test failed with `DID NOT RAISE`. Original bytes were restored; the clone is clean.
- Python **3.8.20** compilation passes. `python -m pytest tests/test_exp07_parity.py -q -p no:cacheprovider` produced **19 passed** in both the worktree and a clean clone at `cf46275` with local `main`.
- `static_checks.sh`: **exit 0**, **319 passed** publication cases; **221 passed / 1 skipped** in each collection order. All runs were CPU-only.
