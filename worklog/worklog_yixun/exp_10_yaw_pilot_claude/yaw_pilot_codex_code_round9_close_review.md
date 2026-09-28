# exp_10 yaw_pilot — Codex code review, round 9 (close; post-closure test maintenance)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="ultra"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt10 --skip-git-repo-check`, stdin closed) · **Reviewed:** `1b90b21, 67e88b6` (Claude Opus 5) · **Prompt:** `review_prompts/code_round9_close_prompt.md` · **Raw log:** `yaw_pilot_2026-09-28_09:42:51_codex_code_round9_close.log` · **Date:** 2026-09-28 · **Verdict:** request changes (blocking: relative TMPDIR regression); nonblocking follow-up: the queue fixture's fake nvidia-smi is not immune

**Reviewer:** OpenAI Codex · **Date:** 2026-09-28  
Reviewed `1b90b21` and `67e88b6` against merge base `9e09599`.

1. **Should-fix — merge-blocking: relative `TMPDIR` breaks the new helper.**  
   [tests/test_exp10_record_tools.py:1210](/home/yixunhu/codespace/xRIR_code_wt10/tests/test_exp10_record_tools.py:1210). Python 3.8’s `mkdtemp(dir=...)` preserves a relative directory. `run_finish` changes cwd, so the returned PATH entry points somewhere else and the shim never fires. With `TMPDIR=.review-round9/relative/tmp` and a normal basetemp, an existing rollback test passes at `9e09599` but fails at HEAD because finish returns 0. The new colon regression also fails under that environment.  
   **Minimal fix:** normalize the base with `os.path.abspath(...)` **before** checking separators/whitespace, and cover relative `TMPDIR`. That one-line change in a scratch copy made both reproductions pass.

2. **Should-fix — pre-existing, nonblocking follow-up: the queue sibling is not immune.**  
   [tests/exp10_record_fixture.py:609](/home/yixunhu/codespace/xRIR_code_wt10/tests/exp10_record_fixture.py:609), [tests/test_exp10_record_tools.py:1918](/home/yixunhu/codespace/xRIR_code_wt10/tests/test_exp10_record_tools.py:1918). A colon splits `<root>/fakebin`, bypassing fake `nvidia-smi`. Tests can nevertheless pass when the ambient query reports sufficient memory—or fails, because the script continues after the resulting numeric-comparison error. A safe fallback reporting zero memory reproduced **two spurious failures**. The after-queue wrapper never calls `nvidia-smi` and is immune.  
   **Minimal fix:** give the queue fake an absolute, colon-free directory with finalizer cleanup; add a colon-root regression proving that fake actually ran. Handle this unchanged sibling in a separate test-only follow-up.

**Verified**

- **Scope:** only `tests/test_exp10_record_tools.py` differs from the merge base, **+60/−7**. No tool, script, fixture-module, or record changes.
- **Python 3.8.20:** complete record-tools file passed with both `.review-round9/normal/pt_normal` and `.review-round9/colon/pt_colon_09:00:00`: **184 passed each**. A logging `nvidia-smi` stand-in prevented hardware queries; the colon run reached it three times, confirming finding 2.
- All exp_10 suites plus the exp_03 record suite, in forward and reverse file order: **385 passed, 1 skipped each**. CUDA was disabled; the skip requires two visible GPUs.
- The regression explicitly exercises `with:colon`. Running the red commit’s test/helper from a scratch copy failed as expected: `FINISH DONE`, rc 0.
- **Cleanup:** all **88** recorded temporary shim directories across verification runs were removed, including failed tests. The unsafe-path guard reported the offending path and registered cleanup.
- `git diff --check`, Python 3.8 `py_compile`, and all **12 exp_03 source pins** passed.
- Disposable-clone `--no-ff --no-commit` merge onto main **`c0cc29a0d579e301a2e13bd39df8c277ae1f0d6f`** succeeded; only the intended test file changed. Clone deleted. Tracked files remained untouched; writes stayed under `.review-round9/`.

[Verification commands and results](/home/yixunhu/codespace/xRIR_code_wt10/.review-round9/verification.json).

**Round-9 verdict: request changes; not closed.**  
**Merge verdict: not approved.**  
**Exact blocking list: finding 1 — relative-`TMPDIR` regression.**
