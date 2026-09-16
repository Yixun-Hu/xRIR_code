**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-16.

Reviewed `exp06-prelaunch`, commits `515136f..8bc7cf8`, against the required SOP, plan, reviews, experiment records, Coder prompt/report, and composed tooling.

**Verdict: request changes.** One should-fix finding blocks closure.

1. **Should-fix — Allow-listed artifacts can resolve outside the disposable tree.**  
   [tools/exp06_finalize.py:887](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:887), especially lines 894–897; delegated hashing at [line 691](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:691).

   `smoke_run_dir()` checks the attempt’s resolved location, but the artifact allow-list checks only filenames. `artifacts()` then uses `is_file()` and hashes through symlinks without checking each file’s resolved location.

   I reproduced passing finalization for both new run types with an otherwise valid attempt and an allow-listed artifact linked outside `_smoke`: `best.pth` for training and `per_sample_hallway.json` for evaluation. Both produced `passed: true`, `diagnostic: true`, and `admissible_arm: false` in the in-memory completion capture.

   This violates the prompt’s explicit “anything outside `_smoke` refused” contract. A passing training diagnostic can consequently direct the next rung to an external checkpoint.

   **Minimal fix:** validate every registered artifact’s resolved location before accepting it, or require regular, non-symlink artifact files. Include `args.json` and `provenance.json`. Add genuine red-first regressions for external checkpoint and evaluation-artifact symlinks; both must refuse without publishing completion.

The Coder’s three design decisions are acceptable:

- Refusing failed HAA diagnostics follows the prompt’s explicit non-`ok` refusal requirement. Existing smoke/probe failure recording remains intact.
- Allow-listing `metrics_all<tag>.json` is necessary because the evaluation wrapper writes it unconditionally.
- Cycle 4 preserves the snapshot guarantee: parsed JSON and its recorded digest come from the same buffer; the second read only rejects staleness. `haa_job_evidence` retains the loader-bound digest, and the summarizer compares against it. Replacing the blanket prohibition on rehashing with the evidence-binding regression is justified.

The approvals-drift failure is expected pending the Planner’s refill. I independently found exactly the four reported changed keys. This is not an additional Coder finding.

Verification performed:

- Read the round’s changes, including intermediate test revisions. `git diff main...exp06-prelaunch --name-status` lists exactly:
  ```
  tests/test_exp06_finalize.py
  tests/test_exp06_launch.py
  tools/exp06_finalize.py
  tools/exp06_launch.sh
  tools/exp06_smoke.py
  ```
  No other exp_06 module or another experiment’s file changed.
- Compared **41 distinct pinned files byte-for-byte with `main`**, covering exp_03’s 12 files, the pinned trainer’s 11-file closure, `sim_to_real/*.py`, the specified exp_04/05 tools, and shared reporting/provenance tools: all identical.
- `git diff --check 515136f..8bc7cf8`: passed.  
  `bash -n tools/exp06_launch.sh tools/exp06_haa_pipeline.sh`: passed.  
  In-memory `compile()` of all four changed Python files: passed.
- Native pytest, using the specified interpreter/environment and `--capture=sys -p no:cacheprovider`:
  - `tests/test_exp06_bootstrap.py -q`: **53 passed**, including the canonical exp_02 reproduction checks.
  - `tests/test_exp06_profiles.py -q`, selecting template round-trip, pinned-template identity, training keys, record schema, and committed-record binding: **5 passed, 19 deselected**.
- Executed **11 existing launcher test bodies** through an in-memory harness: passed, including full/probe/smoke dry-run goldens, budget overrides, HAA ordering, and recovery output. Both HAA parser checks and both actual-shell `require_passed` outcomes also passed.
- Exercised preflight through a fake shell `nvidia-smi`: **14 scenarios passed**. At a 6 GiB ceiling, 6143 MiB refused and 6144/6145 MiB passed; the 8 GiB override behaved correctly. Empty, malformed, multiple-line, negative, nonfinite, and failed queries refused. No real GPU query occurred.
- Exercised **20 in-memory finalization scenarios**. Both valid HAA types passed as diagnostics. All **14 expected refusals** worked: unregistered artifact, missing artifact, unsuccessful receipt, failed child, missing marker, non-`_smoke` directory, and wrong entry, for each type. The external-symlink cases exposed finding 1.
- Snapshot probes confirmed that unchanged bytes retain their buffer-derived digest and substitution during validation refuses.

The supplemental harnesses executed extracted production functions/test bodies without filesystem writes. Finalization’s filesystem, publication, source/approval gates, and process-liveness dependencies were simulated; these checks do not replace full integration replay.

**Replay limitation:** the requested full suite could not run in this strictly read-only sandbox. The initial targeted pytest invocation failed before collection because no writable temporary directory was available; normal finalizer imports also encountered unwritable dependency caches. I therefore cannot independently certify the Coder’s **1,247 passed / 7 skipped / 1 expected failure** totals.

Commit hygiene checks confirmed **11/11 commits carry both trailers**, with a maximum of **154 changed lines**. Implementation commits follow their corresponding red commits. Supplemental commit `7954508` is test-after with a disclosed retrospective red against `c956a3a`; it should not be described as genuine red-first evidence.

The independently computed digests at **8bc7cf8** are:

| Key | Classification | SHA-256 |
|---|---|---|
| `finalize` | TRAINING_KEYS | `9e3a000ba089aa0f4cb40372d07f5e4b7a4e3b851b4884c95dbf5a04aa470b80` |
| `smoke` | TRAINING_KEYS | `00e3b91b8289cc0f1fda8c2ab2ce3b151ca897f4a4c746f93863e3d338091d00` |
| `launch_sh` | TRAINING_KEYS | `c64b60df4632b83617c1343f0d0a56c6ac7cbab24db95719a2aa8734f3977773` |
| `summarize_haa` | Non-training | `c2b56e66f6aa32c7732080219437aa6fb0f71c78298be304ff7fdc9d365f228a` |

These were computed with static import-closure membership and the repository’s `provenance.closure_record` hashing. All 16 other code keys match the committed approvals. Recompute after the fix; these values identify the reviewed tip.

**Exact blocking list for the Coder: finding 1 only—enforce artifact confinement and add refusal regressions for both HAA diagnostic types.**

After closure, preserve A6: merge only after the currently running pretraining has completed and finalized using its original code. The Planner must then refill approvals and obtain the required green suite/integrative review before HAA execution. No files were modified, no processes were signalled, and no GPU work was started.
