Round-7 review scratch evidence. Reviewed tip 8535fae; seven fix commits 19bbca7 d3304b4 ede2260 cf0a22b a5671a4 e3dda86 f3884ec. Main clone base 8ce27e74afe1ca4f4e615bd23bc44e9c0a292b11.
All executions: xRIR Python 3.8.20, CUDA_VISIBLE_DEVICES empty, PYTHONDONTWRITEBYTECODE=1; test/cache/output files in this scratch root. Original source worktree untouched (pre-existing untracked checkpoints only). No live checkpoint writes, no GPU work, no process signals.

New blocking finding: NAS directory symlinks invalidate the record.
- adversarial.py starts with passing collect/check_record. Move synthetic S_simple attempt to nas/S_simple/attempt_..., leave original path as a directory symlink; no evidence bytes change.
- Both check_record and fresh collect raise FileNotFoundError for nas/S_simple/cumulative_hours.json. attempt_record derives metadata siblings from snapshot's resolved directory (bind_provenance.py:183-184,202-218).
- symlink_layers.py aliases logical siblings into the NAS parent to get beyond the first defect. Both calls then refuse the original args.json input as not a product dependency (line463), because current dependencies carry resolved NAS paths while product sidecars retain the published original paths.
- Restoring the directory makes the same original report pass again.
- Fix requires retaining logical declared paths in persisted bindings/dependency maps, using resolution/filesystem identity for checks, and locating arm metadata from the logical approved arm root. Regression must bind/check, relocate an attempt behind a directory symlink, then successfully check the UNCHANGED report without rewriting published evidence.

Round-6 findings 1-7 otherwise closed:
- adversarial.log: 216 document output refusals = md/html x direct/hardlink/symlink x 36 independent concrete inputs. All bytes unchanged.
- 112 figure refusals = 8 predicted PNG/PDF destinations x 7 consumed inputs x hardlink/symlink; no preceding figure created. unequal.log adds 8 direct destination refusals using a valid renamed companion (load permits arbitrary companion paths).
- Full repaired-enclosing-binding evaluator/writer swaps and forged closure record refused. Helper positive legacy M admission requires explicit pin; identical S rejected; current producer's shared admission independently excludes legacy, so no end-to-end positive claim.
- Synthetic producer drift causes collect and check_record refusals. producer-drift.log: actual producer identity from scratch clone, 10 files, pin10484795..., actual param_curve.py appended -> producer closure differs from HEAD. Holding live_producer fixed also fails declared-file stamp at expected digest. File restored.
- unequal.log: independent HTML parsing matches all12 Markdown tables, two K families; S_cyl EDT/C50/T60 query sizes12/11/8, rooms3/3/2, per-seed exclusions0/1/4, H2 baseline rooms and secondary interval agree with producer JSON.
- static suite contains K123 shared-reader refusals and CSS inline specificity regressions; own regressions.py repeats K123, source values vs presentation, parameter corruption, throughput/loss corruption, foreignK/omission and aborted-log binding.
- all-tables.log: every numeric table field independently reconstructed from MAIN canonical products and hash-bound attempts. Both documents cite all9 required digests. All12 HTML/Markdown tables match.
- presentation.log: live H1 K8 partial/partial, K1 not-supported/not-supported; H2 reaches flags and ratios agree with intervals.
- live-pins.log: all66 evaluation evaluator/writer/canonical-batch/TF32/seed identities and four training/launcher/epoch12 pins agree.
- All four PNGs are byte-identical to round6 fresh generation.

Integration and hygiene:
- main-failures.log: requested two tests run WITHOUT branch on clean main8ce27e7, both fail as claimed. exp06 mismatches compare and eval_launch; exp07 seen_simple pin is null with checkpoint present.
- coder-full-suite.log is a COPY OF CODER'S completed run, not a fresh reviewer rerun:3191 passed,49 skipped,2failed in3103.61s. Both failing test IDs match independent main replay.
- merge.log: git merge --no-ff --no-commit8535fae onto main8ce27e7, clean; ten additions exactly; no merge commit created.
- git diff --check; git diff --check main; git diff --check d45469c..8535fae all pass.
- literal git diff main --name-only has13 entries: the10 additions plus3 upstream review-record differences (main moved after final branch merge). Actual staged merge delta is exactly10 added files; existing main files are preserved.
- The exact12 exp03 pinned files have empty history62c9107..8535fae.
- Read-only stat/readlink observation: supplied exp04 attempt path appears as a directory (not a symlink) in this environment; no modification made. The independent exp05 relocation reproduction above does not rely on this observation.

Completed live MAIN bind:443.54784477176145 seconds. Independent check_record:434.21096284594387 seconds, exact recomputation, exit0. The runs, attempts, historical_checkpoints, results and approved_digests sections are identical to the round6 report. All5 probes and8 figures are included.
Merged-tree record tests:86 passed in291.61s, exit0. static_checks.sh: record86 passed in287.97s; first combined collection order307 passed/1skipped in542.65s. Reverse order:307 passed/1skipped in522.84s. Entire static_checks.sh completed with exit0 (compilation, shell syntax, diff check and all three pytest invocations).

Final verdict: round7 request changes; integrative full not approved. Exact blocking list: the single NAS directory-symlink relocation finding above. Round6 findings1-7 are closed. The disclosed lack of a separately observed red phase for item5 remains a process deviation, not an additional runtime blocker.
