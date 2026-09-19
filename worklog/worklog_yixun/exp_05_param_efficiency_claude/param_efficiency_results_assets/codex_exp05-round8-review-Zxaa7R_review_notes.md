Round-8 close and integrative full review evidence
Reviewed branch exp05-record, tip 377cc2cbe16be76b42c7548095823dabcd5c0196.
Round-8 commits 82e0f56 078548f 0d2bebd 377cc2c above aee12d1 (merge of main 9ff4b3e).
Merge rehearsal: current main d0b9f00ec1b4315f238589b92b974e14a4e70caa.

Environment: xRIR Python 3.8.20, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE=1,
PYTHONPATH=scratch clone (plus clone/tests for synthetic harnesses),
XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms,
NUMBA_CACHE_DIR, MPLCONFIGDIR, TMPDIR all within this scratch root;
OMP_NUM_THREADS=2, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=2 where specified.
All live sources, checkpoints, canonical outputs and data read only. No GPU work or process signals.

Executed:
- clone: bash worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets/static_checks.sh
  See static-checks.log. Includes compilation, shell syntax, whitespace, 94 record tests,
  and both orders of exp05/exp04/exp03/exp07 record test files.
- main clone: git merge --no-ff --no-commit 377cc2c onto d0b9f00. No conflicts;
  exactly ten staged additions, no existing main file changed; cached diff --check passes.
  python -m pytest tests/test_exp05_record_tools.py -q -p no:cacheprovider
  See merged-record-tests.log.
- git diff --check aee12d1..377cc2c; merge-base-to-tip has exactly the ten allowed additions.
  All twelve exp03 pinned files have empty git log 62c9107..377cc2c.
- adversarial.py: original round7 independent 216 document output refusals (2 generators,
  36 concrete inputs, direct/hardlink/symlink); 112 figure refusals (8 destinations,
  7 concrete consumed inputs, hardlink/symlink) before any other figure write.
  All bytes unchanged. Repaired evaluator/writer swaps and forged closure record refused;
  producer drift fails collect and check_record; explicit legacy-M helper positive succeeds
  only with explicit pin and S refuses it. No end-to-end positive legacy-M claim.
- moved_adversarial.py: all 216 + 112 refusals replayed again after all four attempts AND
  products move behind directory symlinks. All bytes unchanged.
- unequal_audit.py: 12 HTML/Markdown tables equal, S_cyl EDT/C50/T60 = 12/11/8 queries,
  3/3/2 rooms, 0/1/4 exclusions per seed for BOTH K families. H2 baseline rooms and
  room intervals equal canonical cells. Eight direct figure collisions refused.
- regressions.py: K123 profile, parameter-count corruption, bound throughput/loss corruption,
  foreign-K and missing-run coverage refused. Deliberate canonical numeric mutations change
  presentation (correct: generators do not recompute statistics). Aborted attempt and external
  log bound; changing log changes record; missing log refused.
- producer_drift.py in a separate source clone: actual producer identity 10484795...,
  ten files. Appending to tools/param_curve.py fails live identity; independently holding
  identity fixed fails declared-source digest. Original source restored.
- relocation.py: unchanged checker + fresh collect + final-routed collect reproduce the same
  report for one attempt, a whole arm, ckpt/exp05 WITH products, products alone, one eval,
  all evals, and all four attempts copied to new inodes. Reports/documents untouched;
  regenerated Markdown identical. Direct archive path as CLI attempt also canonicalises.
  Changed history bytes refused by both collect/check; copied attempt with hardlinked checkpoint
  refused; final repointed to that decoy refused by both; escaping output-entry symlink refused.
- inherited_relocation.py: provenance.revalidate(train, source_drift=[]) and eval both []
  before/after relocation; launcher.directory_listing identical and rejects internal symlink.
  Exp07 attempt_record helper passes the synthetic fixture with seen-only protocol fields adapted
  in memory, then fails looking for archive-parent/cumulative_hours.json after the move.
  Exp04 dependency-map defect additionally confirmed by unchanged source inspection.
- real_record.py: fresh documents and eight figures generated from MAIN canonical products;
  binder ROOT, inherited ROOT, approval path and paired_compare.REPO routed to MAIN;
  bind round7's original unchanged documents/figures to compare full report field-for-field
  except git_HEAD; check_record independently recomputes this report. See real-record.log.
- live_final_routes.py: all four actual MAIN attempts loaded via final and their full evidence
  records compared exactly to round7's launcher-named attempts. No writes to live data.
- all_tables_audit.py and presentation_audit.py: all 12 tables agree; all numerical table fields
  trace to canonical products or hash-bound attempts, both documents cite 9 digests;
  live H1 K8 partial/partial, K1 not supported/not supported; H2 decisions match intervals.
  All four PNGs byte-identical to round7 (therefore round6).
- baseline clone WITHOUT branch, main d0b9f00: the two known test IDs replayed.
  Exp06 filled-digest test now PASSES after main's approval update; exp07 seen_simple null pin
  remains the same failure. main-failures.log: 1 failed, 1 passed, 54.23 seconds.
- Coder full suite was NOT rerun by reviewer. Inspected the actual tool-result transcript;
  original Coder scratch logs had been deleted in cleanup. Preserved exact completed output in
  coder-verification-excerpt.log: 3199 passed,49 skipped,2 failed in3495.73s, known IDs;
  definitive second static run 94,315/1,315/1 and STATIC_EXIT=0. Earlier mid-edit run excluded.
  Transcript also records original relocation red: attempt and arm cases failed, 2 passed,
  86 deselected. Round7's disclosed missing separate red for item5 remains a process deviation,
  not a new runtime blocker.

Completed live bind 429.8997s and independent check 429.7828s, both exit0. Report exactly
round7 except git_HEAD=64f3ad7ba2d2d52edff6ca2e4c8c9726da6606c8. Four final-route attempts
independently equal round7,175.732s. Merged record tests94passed in341.11s.
Refreshed no-ff/no-commit merge onto64f3ad7: clean, exactlyten additions, diff--check passed;
only2 record-only files changed from tested d0b9f00.
Initial static script forward order:312passed/4skipped in614.88s; three exp03 sample tests
skipped because the scratch clone lacked ckpt. Added read-only ckpt route, rerunning full
forward order separately; reverse order from original script also gets the sample.

FINAL completed verification: static_checks.sh exit0.
94 passed, 6 warnings in 339.70s (0:05:39)
312 passed, 4 skipped, 7 warnings in 614.88s (0:10:14)
315 passed, 1 skipped, 7 warnings in 612.77s (0:10:12)
Corrected complete forward order:315 passed,1 skipped,7 warnings in633.22s; exit0.
Reverse order uses the read-only historical sample and has only the expected GPU skip.
Reviewed worktree remains377cc2c with only pre-existing untracked checkpoints; staged merge10 additions/2889 lines.
Final verdict: round8 approve; integrative full approved for merge and record use; blocking list empty.
