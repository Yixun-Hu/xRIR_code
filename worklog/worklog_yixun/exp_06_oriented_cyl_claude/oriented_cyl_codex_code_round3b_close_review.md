**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

## Verdict: request changes

Reviewed `0ca806c..df7ea24` in `/home/yixunhu/codespace/xRIR_code_wt2b`.

**The merge preserves the approved training implementation. Round 3b cannot close yet:** several original admission and revalidation findings remain partially resolved. The counterexamples below use CPU checks and RAM-only fixtures; no files were modified, GPU work started, or processes signaled.

## Blocking findings

### 1. **Blocker — H3 verifies self-consistency without enforcing the registered dataset and references**

[tools/exp06_compare.py:284](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:284), [tools/exp06_compare.py:303](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:303)

`check_evidence` verifies the supplied inventory’s hash but never compares it with the registered inventory. `check_reference` verifies the supplied manifest’s semantic hash and ordering but never requires the registered K=8 seed-specific manifest hash or canonical query identities.

**Reproduction:** Starting from RAM copies of the real exp_04 seed-42 control run, admission accepted each independently:

- Empty inventory with its correctly recomputed digest.
- `num_shot: 8` with every entry’s `refs` replaced by `[]`.
- All 6,337 query names replaced with invented names, consistently in the reference and output.

All dependent hashes were updated; the actual checkpoint and source files remained unchanged.

**Why it matters:** These can be admitted as the registered unseen-split experiment despite evaluating different—or absent—inputs.

**Minimal fix:** Compose the applicable `paired_compare.admit_run` checks against `exp04_profiles`: registered inventory, seed-specific reference semantic hash, canonical query digest and room count. Validate reference entries, including eight references per query. Retain the real A positive regression.

### 2. **Blocker — final revalidation omits the HAA child evidence and still separates reads from their bindings**

[tools/exp06_summarize_haa.py:483](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:483), [tools/exp06_summarize_haa.py:742](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:742)

`load_new_arm` retains only job completions, job specifications and child completions in `inputs`. It discards the bindings for the actual child per-sample files, metrics, arguments, provenance, checkpoints, logs, receipts and heading evidence. Rehashing an unchanged completion does not revalidate the files it names.

**Reproduction:** Exercising the production binding code with an explicit verified-job delivery seam produced **39 bindings for 31 children**. Changing a child per-sample file afterwards was accepted by `recheck_inputs`; that file was absent from the map.

Other gaps remain:

- Job completions are hashed after `verify_job` consumes them.
- Legacy runs are loaded before receipt verification.
- Repeated side-cache reads overwrite earlier bindings.
- Producer source records and approvals receipts are published but omitted from final input revalidation.

**Why it matters:** Publication can describe evidence that changed during analysis, contrary to original finding 10.

**Minimal fix:** Carry the complete verified evidence bindings forward; associate parsed values with the exact bytes hashed; reject contradictory bindings for the same path; revalidate the complete set before publication. Add a child-artifact mutation test through admission and publication, beyond the existing cache/log tests.

### 3. **Blocker — HAA admission does not enforce the registered training recipe or primary phase policy**

[tools/exp06_summarize_haa.py:372](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:372)

`child_protocol` checks only `num_shot`, `eval_seed` and evaluation `split`. The finalizer checks that a history completes its *declared* budget; it does not establish the experiment’s required budget.

**Reproduction:** The finalizer-generated nine-child fixture passes `verify_job` with **four epochs per training stage**, rather than §6.2’s 1,000/200 epochs. Direct protocol checks also accept changed learning rate, depth variant and `gl_seed_per_query=True`.

**Why it matters:** Shortened training or S1’s seeded-phase evaluations can enter the primary comparison against historical exp_02 arms.

**Minimal fix:** Add an exp_06 admission profile for §6.2: stage-specific rooms, epochs and validation cadence; training hyperparameters, batching and TF32; waveform length and depth variant; and primary unseeded phase policy. Keep diagnostic fixtures explicitly separate from confirmatory admission.

### 4. **Should-fix — H3 model, registry and epoch identities remain incompletely checked**

[tools/exp06_compare.py:195](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:195), [tools/exp06_compare.py:222](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:222)

The registry digest is checked only for hexadecimal syntax. `model_class` is unchecked; C’s backbone is unchecked; B’s declared epoch is unchecked. The output-meta comparison omits the exp_06-specific identity fields.

**Reproduction:** Internally consistent synthetic runs were admitted with:

- C: `model_class="BogusModel"` and an arbitrary registry digest.
- C: `backbone="simple"`.
- B: `checkpoint_epoch=3`, which was then published as its epoch.

Historical A currently publishes `epoch: null` despite its registered epoch-12 identity.

**Minimal fix:** Validate role-specific backbone/class, the reviewed registry identity, epoch 12 and all applicable exp_06 output metadata. Derive historical baseline epochs from their registration when the historical manifest lacks that field.

### 5. **Should-fix — the exp_05 route does not establish its claimed args/tier binding**

[tools/exp06_compare.py:125](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:125)

The route requires `tier=="M"`, `legacy_M is True`, a `train_args` key and a syntactically valid `args_json_sha256`. It does not establish that these describe the checkpoint’s actual `args.json`, or reconcile tier metadata across the outputs and completion.

**Reproduction:** Using the same approved-record seam as the Coder’s positive test, admission accepted `train_args` pointing to the evaluation log and an unrelated `args_json_sha256`. The positive fixture itself has this inconsistency.

**Minimal fix:** Require the checkpoint-adjacent `args.json`; compare its actual digest with both declarations; derive and validate the applicable metadata using exp_05’s existing rules; reconcile the tier fields in the manifest, outputs and completion.

### 6. **Should-fix — A3 owner presence is checked, but its binding is not**

[tools/exp06_summarize_haa.py:289](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:289)

Requiring a positive `owner_pid` does not prove that it is the owner recorded by the job.

**Reproduction:** After successfully finalizing and verifying the RAM nine-child fixture, I deleted its `launch.pid`. `verify_job` still admitted it. The authoritative finalizer explicitly refuses a missing owner file.

**Minimal fix:** Read and validate the job-root `launch.pid`, require equality with `owner_pid`, and retain its binding for publication.

Avoiding a **job-root liveness check** during retrospective analysis is reasonable because of PID reuse. It does not justify omitting the owner-file binding. Preserve exception-free child liveness checks.

### 7. **Should-fix — production producers accept approvals that were never committed**

[tools/exp06_summarize_haa.py:804](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:804), [tools/exp06_compare.py:568](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:568)

Both producers call the approvals loader without its supported `repo`/`commit` arguments.

**Reproduction:** A fully populated, correctly shaped approvals file outside the repository passed the production approvals gate with **zero deviations**.

**Why it matters:** Plan §6.4 already requires the reviewed approval commits. This is an existing contract, rather than a new plan-level decision.

**Minimal fix:** Bind non-exploratory approvals to the committed bytes at the selected reviewed HEAD, record that identity and revalidate it before publication. Exploratory handling may retain its explicitly labeled deviations.

## Original findings 1–12: disposition

| Original finding | Status | Evidence |
|---|---|---|
| 1 — role-specific closures | **Resolved** | Distinct actual training/evaluation closures admitted through the real finalizer-generated fixture; within-role consistency and approved-digest checks pass. |
| 2 — shallow HAA admission | **Partial** | Full child verification and lineage are composed. Artifact-less 31-child tree and `admissible_arm=False` child now refused. Findings **3 and 6** remain. |
| 3 — actual producer approvals | **Partial** | Wrong producer/child/heading/G1/legacy/reused identities are checked; receipt production is gated. Findings **2 and 7** remain. |
| 4 — H3 evidence | **Partial** | Null hashes, missing log, false confirmatory flags, missing training bindings, source/data drift and contradictory aggregates are refused. Finding **1** remains. |
| 5 — H3 protocol/identity | **Partial** | K=1, reversed output query order and an invented semantic hash with consistent surrounding bindings are refused. Findings **1 and 4** remain. |
| 6 — partial legacy receipt | **Resolved** | Registered two-arm enumeration enforced; partial receipts and either-arm tampering refused. |
| 7 — exp_05 route | **Partial** | Explicit route and approved-record checks work; shaped positive admitted. Finding **5** remains. |
| 8 — empty HAA cohort | **Resolved** | Reportable void cells and unavailable intervals; zero-width bootstrap refusal preserved. |
| 9 — H3 eligibility | **Resolved** | C/A-only reports H3 unavailable; exploratory decisions suppressed. |
| 10 — final revalidation | **Partial** | Existing cache/log mutation tests pass; finding **2** remains. |
| 11 — weak refusal regexes | **Resolved** | Six assertions now use diagnostic phrases. |
| 12 — partial output publication | **Resolved** | Distinct/absent-path preflight and cleanup pass injected publication failures. This is cleanup on failure, not crash-atomic publication of two files. |

## Merge and commit hygiene

- The requested `git diff 5091f6c HEAD -- …` shows **only** the `probe_align` registration in `exp06_profiles.py` and its template key. Finalizer, trainer, recipe, smoke and launcher are byte-identical.
- `open_job` retains both exclusive ownership tracking and `own_launch`.
- The shared dead-PID helper preserves the fixture contract.
- The adapter tests moved **byte-for-byte** to `tests/test_exp06_approvals_api.py` in the merge. Training-branch profiles tests remain unchanged.
- No training-parent test names were lost. The later fix cycle deliberately removed four old synthetic admission test functions and renamed the approvals-refusal test. Removing invalid fixtures was justified, but equivalent whole-arm coverage was not fully restored: current summary positives verify one nine-child job, while CLI tests stub `load_new_arm`. Restore missing-job, registered-init and whole-arm binding coverage with the fixes.
- **25 commits follow `90ce3ba`, including integration commit `76e8f6d`; 26 including the merge.** All have both required trailers. Maximum post-merge change size: **171 lines**. The report’s “26 after merge” count should be corrected.
- The Planner’s pending template is byte-identical to the worktree template. The restored main-tree copy is the documented temporary schema accommodation.

## Verification performed

Read the required SOP, plan/reviews/notebook, role-change entry, historical records, fix prompt/report, composed tooling, changed source/tests and merge resolutions. The excluded worktree was not accessed.

All Python checks used the specified xRIR interpreter/environment, CPU-only settings and disabled bytecode. Pytest used `-p no:cacheprovider`.

| Check | Observed result |
|---|---|
| `git diff --check 0ca806c..HEAD` | Clean |
| `git diff main...HEAD --diff-filter=M --name-only` | Empty |
| Protected-source comparison against `main` | **41 paths identical**, including exp_03’s 12-file closure and the trainer’s 11-file closure |
| In-memory compilation of changed Python files | **21 passed** |
| `bash -n tools/exp06_haa_pipeline.sh tools/exp06_launch.sh` | Passed |
| `pytest.main` over `tests/test_exp06_*.py`, `tests/test_provenance.py`, `tests/test_exp03_record_tools.py`, excluding writable-fixture cases | **429 passed, 8 skipped, 740 deselected** |
| RAM-fixture replay of compare, summarize-HAA and approvals-adapter modules | **128 passed, 16 deselected** |
| Four remaining finalizer-backed summary assertion cases, with RAM artifacts and unchanged verification logic | **4 passed** |
| Real exp_02 canonical regression | **All 11 cells exact**: difference, both query-bootstrap endpoints, both two-way endpoints and cohort counts |
| Real exp_04 A admission | **All five retained control runs admitted; 6,337 queries each**, using a constructed mapping of their approved identities |
| RNG isolation | Python, NumPy global and Torch states unchanged by the exercised admission/statistics paths |
| Repeated H3 analysis | Identical serialized results |
| Encoder/factory regression within selected suite | Active-yaw symmetry, heading cancellation, k=0 exactness and legacy factory/state/encoder-output parity passed |
| Final repository status | Clean, HEAD `df7ea24` |

Normal collection encountered Numba’s writable-cache requirement. Adapted runs disabled Numba JIT and used the existing Matplotlib cache. Temporary artifacts existed only in a RAM filesystem overlay; initial overlay setup failures were corrected before the reported green replay. The four finalizer-backed cases used the current repository instead of creating a Git clone and executed the shell’s unchanged job-spec Python in-process.

**I did not independently reproduce the Coder’s complete 1,169-pass suite or historical red-run counts.**

## Wiring decisions and closure requirement

The new `--gate-g1` flag is appropriate: production must receive and verify the approved gate artifact. The repo-relative `manifest_path` limitation fails closed for the documented use; absolute paths should remain explicit in Planner commands.

After fixes and a successful close review, Planner wiring must restore the pending schema, recompute code approvals at the final merged revision, commit the reviewed approval records, produce and approve the legacy receipt, fill artifact identities, and pass explicit `--approved` and `--gate-g1` paths.

**Exact blocking list: findings 1–7 above. Round 3b may not close at `df7ea24`.** This verdict does not reopen the separately approved training-path merge.
