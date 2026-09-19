Review: exp_05 round 6, d9e9759..840e345, plus integrative merge into main ecbcc5f.
Reviewer: OpenAI Codex (GPT-6, API agent). Date: 2026-09-18.
Original worktree unchanged: only the pre-existing untracked checkpoints symlink.
All review writes are in scratch; all executions use CUDA_VISIBLE_DEVICES empty and xRIR Python 3.8.

Verdict established: request changes / not approved. Blocking findings 1-4:
1. make_results_md.py:290-297 omits history.jsonl, args.json, train_manifest.json from output protection. Actual md/page invocations overwrite these synthetic inputs. make_figures.py:84-90 permits predicted PNG destinations hardlinked/symlinked to a canonical JSON; both overwrite it. Protect every input and concrete output by filesystem identity before writes.
2. bind_provenance.py:347-351 validates producer closure claims without verifying current producer source bytes/identity. A scratch clone edit to tools/param_curve.py leaves collect() identical and check_record passes. Recompute producer_identity(tools.param_curve), compare to pin and sidecar, hash declared producer files.
3. bind_provenance.py:405-413 accepts any non-current evaluator as legacy M without a matching evaluator_exp04 pin; the writer approval is not enforced. Fresh contracts.py replays M-entrypoint/writer closure swaps, both admitted after binding repair. Enforce closure-record consistency and evaluator/writer pins, allow legacy M only against explicit approval.
4. make_results_md.py:143-149 uses EDT own cohort for all metrics. Through the real producer, S_cyl C50 n=11 versus EDT/T60 n=12; row claims n=12. H2 baseline-own room count, per-arm/seed/metric exclusions, H2 room-cluster intervals also omitted. Render these canonical fields with metric-specific cohorts.
Nonblocking:
5. Shared record.load/validate does not require registered profile_digest(name); num_shot=123 with repaired profile/sidecar is rendered as K=123. Binder catches this separately (line385); move it into shared admission.
6. README.md:99-100 --approved MAIN/path alone cannot route producer dependencies to MAIN. Replayed product-binding stage rejects MAIN/tools/exp04_profiles.py. Remove workaround or implement complete routing.
Additional static presentation issue: make_results_html.py:88 and :138 put backbone colors in presentation attributes on class mark; imported exp04.CSS has .mark{stroke:var(--blue);fill:var(--blue)} at exp04 HTML line18, which overrides those attributes. Cylindrical dots and uncertainty marks inherit blue. Fix with explicit backbone CSS or inline style. jsdom computed-style API is broken in this environment; conclusion is from CSS/source inspection, not a browser test.

Fresh results:
- adversarial.py / adversarial.log: all six original findings reproduced; curve/interval/verdict mutate respective display; count/throughput/epoch-loss changes refused; foreign K and omission refused; aborted attempt and external log bound, changed log changes report, missing log refused.
- contracts.py / contracts.log: both unapproved closure substitutions accepted.
- all_tables_audit.py: every displayed numeric table field independently checked, 12 HTML/Markdown tables equal.
- presentation_audit.py: H1 K8 EDT M/L supported, C50 L supported -> partial both. K1 none -> not supported both. H2 K8 four cells reach, K1 only M_cyl vs L_simple C50 reaches; ratios correct.
- live_pins_audit.py: all 66 evaluator/writer/canonical precision/seed pins and four training/launcher/checkpoint pins agree.
- git diff checks pass; all 10 new paths absent from main; exact 12 exp03 pinned files have empty post-62c9107 history.
- merge.log: git merge --no-ff --no-commit 840e345 into main ecbcc5f: no conflicts, not committed.
- merged-record-tests.log / XML: 71 passed in 224.67s on that rehearsed merge.
- Fresh real_record.py: started from MAIN CWD, loads 840e345 tools from clone, routes md.REPO and binder roots to MAIN; writes only scratch real/. BIND PASS 429.7087935702875 seconds. Independent check_record PASSED in 425.5738285719417 seconds; report identical. real-record.log ends COMPLETE.
- New and earlier bindings have identical runs, attempts, historical_checkpoints, results and approved_digests. All four PNGs byte-identical. Both documents cite all five products and four completion digests. inputs records all eight figures.
- The original detached process completed normally; its PID was 1239773, outside the read-only sandbox PID namespace. Do not infer exit from a sandbox ps. A duplicate foreground retry was stopped with SIGINT after identifying exact argv and unique TMPDIR; do not count live_retry/ as completed evidence.

Recovered completed verification from the interrupted invocation, inspected and clone HEAD/status checked:
/tmp/exp05-round6-review-pq1uu5v4/clone HEAD 840e345, local main2642e4e, only untracked ckpt/checkpoints.
static-checks.log: 71 passed; both 4-experiment collection orders 292 passed / 1 skipped each; py_compile/bash-n/diff checks passed.
full-suite.log and full-suite.exit: 3050 passed,49 skipped,2 failed in2524.17s,exit1. Independent failures: exp07 seen_simple runtime approval still null despite live checkpoint; exp06 HAA dry-run golden sees live completed stage and prints SKIP instead of MKDIR. Both reproduced on main without this branch; exp06 passes with baseline clone ckpt link temporarily absent (then restored). These are not new round6 blockers.
Earlier real-record.log: bind453.44s/check437.65s passed; fresh completed run supersedes this.

Markdown test/implementation co-landed in 9b956b1. The disclosed not-red-first workflow is an SOP process deviation; green tests do not establish test-first ordering.
