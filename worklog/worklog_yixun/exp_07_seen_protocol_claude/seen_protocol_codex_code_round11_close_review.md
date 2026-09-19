**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-18

Reviewed `4ae57f6`, `6d77a02`, `dbf42fd`, including merges `8655523` and `a1c5b9b`; tested worktree HEAD `a1c5b9b`.

**Round-11 verdict: request changes. Merge verdict: not approved. Blocking findings: 1 and 2.**

1. **Blocker — checkpoint identity does not establish the identity of `final`.** [bind_provenance.py:370](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:370)

   After saving a valid report and relocating an attempt, I created another directory containing a hardlink to its checkpoint and repointed `final` there. **Both `collect` and `check_record` accepted it; collection reproduced the unchanged report.** Replacing `final` with an ordinary directory containing that hardlink also passed. The new check compares checkpoint inodes but never checks that `final` names the certified attempt directory. This violates the requested repointing refusal and omits exp_05's reviewed directory check.

   **Minimal fix:** require `final.is_symlink() and identical(final, directory)` for `final = directory.parent / 'final'`, retaining checkpoint identity/digest checks. Add both negative cases. An in-memory application of that guard passed all eight reviewer relocation/identity cases without changing source.

2. **Blocker — the change invalidates the existing live calibration receipt.** [tools/exp07_record.py:30](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_record.py:30), [bind_provenance.py:333](/home/yixunhu/codespace/xRIR_code_wt07/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/bind_provenance.py:333)

   `tools.exp07_calibration` imports `exp07_table`, which imports `exp07_record`. The latter is therefore in the already-published calibration producer closure. Using the actual `ckpt/exp07/results/CALIBRATION_SEEN_V1.json` and its five released runs, **main accepts `calibration_record`; round 11 refuses it with `calibration producer closure`**. Its recorded digest is `8734a835…`; this branch computes `aedbf15c…`, with `tools/exp07_record.py` the only changed member. Filling training pins will not repair this mismatch. Synthetic fixtures replace the producer identity, so their success does not cover this live compatibility requirement.

   **Minimal fix:** preserve that imported module's bytes and move the new path helpers into record assets. If its publication changes are retained, provide a reviewed CPU-only calibration migration: retain the original gate receipt, regenerate from the same five runs, verify identical metrics/decision, and update the eventual bind's evidence path. Add a regression using a receipt produced before round 11.

3. **Should-fix, nonblocking for this close — writer overlap protection misses an input relocated after admission.** [tools/exp07_record.py:70](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_record.py:70)

   After `build_table` admitted a synthetic input, I moved its attempt behind a directory symlink and passed the archive's `args.json` as the forced Markdown destination. `write_outputs` succeeded and changed the admitted input. Both destination spellings miss the input map's logical key; this writer has no inode check. This requires relocation between admission and publication, outside the documented archive-after-binding sequence; it is not one of the two merge blockers. The document generators correctly refuse equivalent aliases.

   **Minimal fix:** compare existing output destinations against admitted inputs with `identical`/device-and-inode before publication. The scratch guard makes the reproduction pass.

**Verified**

Validation used Python **3.8.20**, hidden CUDA and disabled bytecode/pytest caching. Test fixtures and review artifacts were kept in worktree scratch. CUDA remained hidden; live artifacts and tracked source files were only read. No live training/evaluation processes were signalled.

| Check | Result |
|---|---|
| `git diff 9b25738..a1c5b9b` | Exactly the seven claimed files, **+286/−54**; every changed line reviewed |
| Python 3.8 parsing/compilation, shell syntax, `git diff --check` | Passed |
| `bash worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_results_assets/static_checks.sh` | **Exit 1; 327 passed / 1 failed** — the known seen_simple approval guard |
| exp_07 → exp_05 → exp_04 → exp_03 record tools | **325 passed / 1 skipped** |
| Reverse collection order | **325 passed / 1 skipped** |
| `python -m pytest tests -q -p no:cacheprovider` | **3,208 passed / 49 skipped / 3 failed**, 3,597.51 s; corrected reruns below |
| Prior reviewer suites, copied from `.review-round9` | **79 passed** |
| Independent complete-root relocation suite | **6 passed** |
| Dry-run `git merge --no-ff --no-commit a1c5b9b` onto main `dfd71cf` (descendant of `615579a`) | No conflicts; staged diff remains seven files; clone deleted |

The full invocation's two exp_06 repository-boundary failures were caused by my in-repository `--basetemp`: both files were correctly refused as **untracked**, while the tests expected **outside the repository**. Re-running those two cases in a clone with scratch outside the clone produced **2 passed**. The remaining failure is the known seen_simple approval guard. The full invocation itself was not zero-failure. See [full-suite log](/home/yixunhu/codespace/xRIR_code_wt07/.review-round11/full_suite.log) and [corrected rerun](/home/yixunhu/codespace/xRIR_code_wt07/.review-round11/outside-case-rerun.log).

- **Relocation:** attempt, arm, evaluation run, and complete synthetic root including products, audit, calibration and report preserve unchanged-report checking and fresh collection. Regenerated Markdown, HTML and LaTeX are byte-identical and retain logical names. Changed checkpoint bytes, ordinary `final` repointing, and a manifest naming another attempt are refused; finding 1 records the remaining hardlink exception. Publication through a directory symlink records logical output names.
- **Earlier refusals:** all **880 dependency mutations** have zero unexpected outcomes; **74 end-to-end mutations**, plus foreign-role inventory and foreign-producer checks, remain refused. All **70 attempt-file/log edits** are detected. **75 calibration boundary checks** preserve five-metric finite-cohort rules and approval independence. Parity receipts/XML, repeated `--evidence`, split agreement, exact run/product coverage and terminal-state checks remain effective.
- **Live paths:** `seen_simple/final` names `attempt_20260917T034540full`; its manifest's `attempt_path` names that same directory. All **10** seen_simple evaluation checkpoint paths name that attempt's `epoch_012.pth`, share the final-routed file's inode, and agree with its completion digest. Approval pins remain all-null; seen_cyl and seen_aug have no terminal receipts, so a full live bind is not yet possible.
- **Running-training protection:** intersections of the seven changed files with `source_closure('tools.exp07_train')`, `('tools.exp07_launcher')`, `('tools.exp07_eval')`, and `('tools.exp07_eval_launch')` are all **empty** (13/22/22/24 files). The launcher's extra probe/shell files are unchanged. Finding 2 concerns the separate, already-used calibration producer closure.

Evidence and reproducible scratch checks: [.review-round11](/home/yixunhu/codespace/xRIR_code_wt07/.review-round11). **Exact blocking list: 1 — missing final-directory identity check; 2 — existing calibration evidence invalidated without preservation or migration.**

Saved on main: [seen_protocol_codex_code_round11_close_review.md](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_codex_code_round11_close_review.md).
