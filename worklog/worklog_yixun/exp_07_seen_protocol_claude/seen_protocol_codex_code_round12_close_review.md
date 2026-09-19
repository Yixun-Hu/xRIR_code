**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-19

Reviewed `087dbc5`, `fdac647`, `d96a516`, `4ff000c`, `68eb435`; tested tip `68eb4351911deedc37247aa33fe60e892cac227a`, following merge `068d84a` of MAIN `6449754`.

**Round-12 verdict: approve. Merge verdict: approved for merge. Exact blocking list: none.** Round-11 blockers 1–2 are closed. The generator scope of should-fix 3 is implemented; the frozen producer residual is outside the documented sequence.

1. **Nit — the README understates the residual writer gap.** [README.md:196](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/README.md:196), [tools/exp07_record.py:48](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_record.py:48).

   After admission and relocation, forced Markdown publication overwrites synthetic `args.json` through **either** its published spelling or its archive spelling. The restored writer resolves both destinations before comparing them with the pre-move input map; neither matches. Thus “protected under one of its two names only” is inaccurate.

   **Minimal fix:** say that forced Markdown can overwrite through either spelling after an admission-to-publication move. Keep the documented **producers → generators → bind → check → archive** order. This reproduction is **not reachable within that sequence**; no additional Planner restriction is needed when it is followed. Existing JSON destinations and Markdown without `--force-md` are refused. Without relocation, direct/directory-symlink aliases are refused; replacing a separate hardlink leaves the admitted source bytes intact. This documentation correction is nonblocking and does not require changing the frozen producer closure.

**Verified**

- **Blocker 1 closed:** [bind_provenance.py:382](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:382) checks `final.is_symlink()` and directory identity after arm identification. All eight preserved round-11 relocation/identity cases pass. Both hardlink-holding `final` negatives are refused by fresh collection **and** checking a previously saved report. Attempt, arm, evaluation-run and complete-root moves preserve reports and byte-identical Markdown/HTML/LaTeX; changed bytes and misnamed attempts remain refused.
- **Blocker 2 closed:** a fresh process, with no fixture or monkeypatch, called the branch binder's `run_record` for the five actual released runs and `calibration_record` on the real `ckpt/exp07/results/CALIBRATION_SEEN_V1.json`: **accepted**. Computed and recorded producer digests both equal `8734a835961ade84b5f0f6b50cbe0a498b7bd57cf185ca33362874acd1fe07a1`. The pre-restore digest is `aedbf15c…`; `tools/exp07_record.py` is its sole changed closure member.
- **Frozen bytes preserved:** `tools/exp07_record.py` matches MAIN's committed and working bytes, SHA-256 `04a417f894e9195adca8f86816f4ef395ef22beeb9f40b8895c2616b01348d68`. It is absent from the MAIN diff. All four import-derived closures match MAIN: training `36d57a84…` (13 files), launcher `ee2d2845…` (22), evaluation `1c9eaf10…` (22), evaluation launcher `ee84bb44…` (24). The seven changed files intersect none of these or the calibration producer closure; path-loaded record assets stay outside them. [Live/closure evidence](/home/yixunhu/codespace/xRIR_code_wt07/.review-round12/live_and_closures.log).
- **Should-fix 3, generators:** [make_results_md.py:213](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/make_results_md.py:213) protects every declared product dependency. Independent probes produced **45 refusals** across all three generators, published/resolved/hardlink spellings, relocated training/run inputs and an unseen-product dependency, preserving every source byte. The frozen writer residual above was separately reproduced with the actual table renderer.

| Command/check | Result |
| --- | --- |
| `python -m pytest tests/test_exp07_record_tools.py -q -p no:cacheprovider` | **155 passed** |
| `bash …/seen_protocol_results_assets/static_checks.sh` — exp_07 subset | **332 passed / 1 failed**; script exit **1**, solely the known seen_simple approval guard |
| Record suites exp_07 → exp_05 → exp_04 → exp_03 | **330 passed / 1 skipped** |
| Reverse collection order | **330 passed / 1 skipped** |
| `python -m pytest tests -q -p no:cacheprovider --basetemp=<external scratch>` | **3,215 passed / 49 skipped / 1 failed**, 3,028.21 s |
| Preserved rounds 6–9 forgery suites plus eight round-11 relocation cases | **85 passed initially; 2 missing-clone setup errors; corrected rerun 2 passed** |
| Reviewer generator/writer probes | **4 passed initially; 2 corrected scratch-assertion reruns passed** |
| Python 3.8 parsing/compilation, shell syntax, `git diff --check` | Passed |

The sole full-suite/static failure is `tests/test_exp07_profiles.py:253`: the seen_simple checkpoint exists while its runtime approval digest is still null. This is the documented guard awaiting the all-arm pin fill; neither invocation was zero-failure. [Full-suite log](/home/yixunhu/codespace/xRIR_code_wt07/.review-round12/full_suite.log), [static checks and both orders](/home/yixunhu/codespace/xRIR_code_wt07/.review-round12/static_checks.log).

The preserved suites retain **880 dependency mutations with zero unexpected outcomes**, **74 end-to-end refusals**, **70 detected attempt-file/log edits**, and **75 calibration boundary checks**, plus foreign-role/producer, parity XML/receipt, split, coverage and terminal-state refusals. The round-10 parity suite is also covered by the static checks and full run. The two parity setup errors were repaired by supplying their expected clean clone; both tests passed and that clone was deleted. The publication probe correction changed only my assertion of the renderer's Markdown marker. Logs preserve both initial and corrected runs.

**Scope and merge:** every changed line reviewed. The five round-12 commits total **+186/−34**; the cumulative branch change relative to MAIN's merge base is exactly **seven files, +439/−55**: the fixture, record tests, README, binder, Markdown generator, new `record_paths.py`, and static-check script. Direct `git diff main` also shows seven MAIN-only worklog/log updates from newer `d7d94e4`; these are not branch edits. A disposable clone of MAIN **`d7d94e4944489e05050ca320151615bf300f127b`**, a descendant of `6449754`, accepted `git merge --no-ff --no-commit 68eb435` without conflicts. Its staged diff was exactly the seven reviewed files; whitespace checks passed and merged file bytes matched the reviewed tip. **Clone deleted.** [Merge log](/home/yixunhu/codespace/xRIR_code_wt07/.review-round12/merge.log).

Python **3.8.20**, `CUDA_VISIBLE_DEVICES=''`, disabled bytecode/pytest caching, external full-suite scratch. Reviewed source and live checkpoints/data were only read; no GPU work or process signals. Writes were confined to review scratch and the requested MAIN review record. Evidence and reproduction scripts: [.review-round12](/home/yixunhu/codespace/xRIR_code_wt07/.review-round12).

Saved on MAIN: [seen_protocol_codex_code_round12_close_review.md](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_codex_code_round12_close_review.md). Both disposable clones and all external review temporary directories were deleted; logs and reproduction scripts remain.
