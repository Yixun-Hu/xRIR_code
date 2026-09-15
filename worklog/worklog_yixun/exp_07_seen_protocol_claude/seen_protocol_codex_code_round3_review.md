**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15 · log `seen_protocol_2026-09-15_12:27:17_codex_code_round3.log`

**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15

## Verdict: request changes

**Blocking findings: 1 — the seen dataset module is absent from the evaluator’s recorded source closure.**

Round-2 blockers 1–2 and should-fix 3 are closed by cycle 0. Round 3 has the new blocker below.

Reviewed every changed line in `a035d7f..ab585d0`. No repository files were modified, no GPU was used, and no processes were signalled.

## Findings

1. **Blocker — the evaluator does not bind the seen dataset code.**  
   [tools/exp07_eval.py:20](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_eval.py:20), [tools/exp07_eval.py:55](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_eval.py:55)

   The frozen `build_dataset` helper imports `treble_xRIR_seen_dataset` inside its function. Importing `tools.exp07_eval` therefore does not load that module, and `source_closure()` omits it.

   **Reproduced:** `source_closure('tools.exp07_eval', REPO)` returns 15 files, excluding `treble_multi_room_dataset/treble_xRIR_seen_dataset.py`. Neither the frozen evaluator closure nor the writer closure covers it. With the recorded bindings intact, an in-memory simulated change to that module produces `revalidate(...) == []`.

   Consequently, the approved evaluator digest does not certify all code used to select and serve seen queries. The pickle binding protects the split file, but does not cover its implementing module.

   **Minimal fix:** import the seen dataset module at module scope in the new entry point, while retaining the frozen helper if desired. Add closure-membership coverage and a regression proving changes to that module refuse. Leave the pinned module itself untouched.

2. **Should-fix — the seen GPU parity reference shares the implementation under test and bypasses the launcher.**  
   [tests/test_exp07_eval_gpu.py:86](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_exp07_eval_gpu.py:86), [tests/test_exp07_eval_gpu.py:108](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_exp07_eval_gpu.py:108)

   `direct_reference()` calls `exp04_eval.evaluate_p_batch`, which is also the evaluated P path. A common error there can pass both sides. The test’s `run()` helper also fabricates a handshake and invokes the evaluator directly; it does not exercise the planned seen run through the launcher.

   **Minimal fix:** obtain the reference P results from frozen `eval_yaw_rotation.evaluate_batch`, and add the planned launcher integration case. Retain K=8/K=1 and the final nine samples.

   The **25-query selection is correct**: it contains one complete canonical batch followed by the actual final nine queries.

3. **Should-fix — manifest construction records the split identity too late.**  
   [tools/exp07_manifests.py:56](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_manifests.py:56), [tools/exp07_manifests.py:72](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_manifests.py:72)

   The builder first constructs datasets and writes manifests, then obtains `seen_split_identity`. A replacement during construction can make the index attribute old-split manifests—or a mixed grid—to the replacement pickle.

   **Reproduced with a stub dataset and in-memory outputs:** dataset selection occurred before replacement; the build succeeded and recorded only the new identity. This repeats the timing pattern repaired in the training launcher.

   **Minimal fix:** capture the split binding before constructing either dataset, recheck it before publishing the index, and store the captured binding. Add mutation-between-K coverage. Downstream query-list checking limits the damage, but the index itself must describe its inputs correctly.

4. **Nit — explicitly binding the correct split file is rejected.**  
   [tools/exp07_eval.py:82](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_eval.py:82)

   The shared launcher stores explicit binding paths as absolute paths; `seen_split_identity()` returns a repository-relative path. Direct dictionary comparison rejects even the canonical file.

   **Reproduced:** `--bind-input seen_split=<REPO>/treble_multi_room_dataset/seen_test_split.pkl` raises `seen_split must bind ...`.

   **Minimal fix:** compare resolved paths and digests, then store the canonical binding. Automatic binding works.

## Verified

### Cycle 0: round-2 findings closed

Using base/head modules and in-memory receipt bytes with recomputed outer hashes:

- The valid seen receipt passes.
- All six substituted-completion cases pass at `a035d7f` and refuse at `ab585d0`: wrong manifest digest, protocol, yaw, tier, backbone, and the combined original forgery.
- Two one-hour full attempts permit another 30-hour attempt at the base and refuse at the head.
- A fresh receipt and renewal cannot reset the seen count.
- Smoke/probe rows do not count; zero or one full attempt remains admissible.
- The third attempt refuses before directory creation, resource checks, or field construction.
- The equivalent unseen ledger remains admissible.
- **Five real exp_05 receipts** have identical successful base/head validation results.
- The training split-before-inventory regression passes, including mutation refusal.

The receipt now establishes both the completion-to-manifest digest link and the requested arm identity.

### Manifests

The stub build produced four K/seed manifests. All four index records matched the saved manifests’ semantic hashes, file hashes, and entry counts.

Verified overwrite refusal with original bytes preserved, the default 6,217-entry gate, wrong-cwd refusal, and the planned defaults `{42…46} × {8,1}`. Finding 3 concerns split identity capture.

### Evaluator and historical paths

- After removing only the new optional parameter, docstring change, and factory-selection expression, **`run_exp04`’s AST is identical to the base**. Default exp_04/exp_05 construction, loader configuration, numerical operations, padding, aggregation, and serialization remain unchanged.
- **Five historical launcher command vectors** match base/head exactly: default exp04, explicit exp05, and tiers S/M/L.
- Real seen dataset readback gives **6,217 queries / 131 rooms**.
- Real cross-split checks refuse **6,337 unseen queries under seen** and **6,217 seen queries under unseen**, before model construction.
- `ManifestDataset` validates the full query list before applying `max_samples`.
- Canonical padding remains the frozen implementation.

This supports unchanged default numerical execution. Actual GPU output parity remains deferred; literal JSON bytes also contain elapsed-time and provenance metadata.

The unseen GPU test compares complete payloads after excluding exactly:

`elapsed_min`, `split`, `seen_split_sha256`, `eval_manifest_sha256`.

These are within plan §2’s permitted exceptions. Other provenance fields remain equal in this fixture and are still compared. No numerical fields are exempted.

### Evaluation launcher

In-memory runs through the actual finalisation function verified:

- Successful completion records `split` and `seen_split_sha256`.
- Wrong split metadata in the outputs refuses completion.
- A split pickle changed after child completion refuses with `mutable input mismatch: seen_split`.
- Failed cases leave no completion.

Field construction gives:

| Manifest entries | `max_samples` | `confirmatory` |
|---:|---:|---|
| 6217 | 0 | true |
| 6216 | 0 | false |
| 6218 | 0 | false |
| 6217 | 16 | false |

The paired entry/split routing and requested CLI refusals pass. A short standalone manifest still fails the real dataset’s query-list check; `confirmatory=False` does not make it a valid full-split manifest.

Legacy eval manifests already contained `split='unseen'`. Their metas and completions retain their previous field sets.

### Tests and hygiene

Environment: pinned Python 3.8 interpreter, worktree `PYTHONPATH`, local AcousticRooms root, `CUDA_VISIBLE_DEVICES=''`, and `PYTHONDONTWRITEBYTECODE=1`.

| Check | Result |
|---|---|
| `python -m pytest tests -q -p no:cacheprovider` | Blocked before collection: no writable temporary directory |
| Whole-suite replay with `--capture=sys`, cache accommodations, and explicit unwritable-fixture skips | **1473 collected; 487 passed, 986 skipped, 0 failed** |
| Skip breakdown | **953 sandbox-related; 33 CUDA/environment** |
| `py_compile.compile(..., doraise=True)` with in-memory bytecode sink | **13/13 changed Python files passed** |
| `git diff --check a035d7f..ab585d0` | Exit 0 |
| Pinned-path diff | Empty |
| Scope | Reported 13 files; **857 insertions / 11 deletions** |
| Co-author trailers | Correct on all eight commits |

The replay **does not independently certify 1,432 passed / 41 skipped**. Blocked fixtures include filesystem-backed manifest creation, receipt/ledger writes, temporary Git repositories, and real child/finalisation tests. Critical cases were additionally exercised in memory as described above.

| Commit | Changed lines |
|---|---:|
| `9e8cdf7` | 109 |
| `b5cf876` | 28 |
| `f80a119` | 179 |
| `28a876a` | 29 |
| `71c5101` | **206** |
| `23bdddf` | 51 |
| `2e7703f` | 115 |
| `ab585d0` | 151 |

**Process deviation:** `71c5101` exceeds the round’s limit. Its content is coherent and reviewable; retain the recorded deviation and split future changes before committing.

The pre-existing untracked `checkpoints` symlink remains unchanged.

## Coder decisions

The supplied records do not contain the numbered decisions 2–6 list, so I cannot reliably assign those numbers. Assessments of the stated choices:

| Choice | Assessment |
|---|---|
| Reuse frozen `build_dataset` and `ManifestDataset` | Accept the construction path; require closure repair in finding 1 |
| Optional `dataset_factory=None` hook | Accept; preserves the existing default execution |
| Add `protocol` to timing limits and update two equality assertions | Accept; supplies the discriminator without changing unseen numerical limits |
| Mark non-6,217 manifests non-confirmatory | Accept; retain independent query-list and producer checks |
| Use 25 queries for final-batch parity at both K | Accept the cohort; repair the reference and launcher coverage in finding 2 |
| Exclusive manifest creation and repository-root requirement | Accept; repair split capture in finding 3 |

## Fitness for round 4

**Resolve finding 1 before closing round 3 and pinning its evaluator closure.** Correct findings 2–3 before using the corresponding parity evidence or manifest index.

Round-4 admission must consume these exact locations:

| Artifact | Required fields |
|---|---|
| Reference index | `seen_split.path`, `seen_split.sha256`; `manifests[filename].manifest_hash`, `.file_sha256`, `.entries`, `.num_shot`, `.seed`; grid fields `seeds`, `num_shots` |
| Eval manifest | `split`, `seen_split_sha256`, `mutable_inputs.seen_split.{path,sha256}`, `split_count`, `n_samples`, `max_samples`, `confirmatory`, `allow_dirty_used` |
| Eval recipe/data | `num_shot`, `manifest_seed`, `gl_seed`, `manifest_hash`, `manifest_file_sha256`, `conditions`, `yaw_cols`, `acoustic_cols`, `e_acoustic_cols`, `batch_size`, `batch_canonical`, `tf32`, `data_identity` |
| Both output metas | `split`, `seen_split_sha256`, `eval_manifest_sha256`, plus recipe/sample metadata |
| Eval completion | `split`, `seen_split_sha256`, `eval_manifest_sha256`, `confirmatory`, `allow_dirty_used`, `outputs`, `child_exit_status` |
| Eval code pins | **`source_closures.entrypoint.sha256`** for `tools.exp07_eval`; `source_closures.writer.sha256`; separately `evaluator_closure` for frozen `eval_yaw_rotation` |
| Training bindings in eval manifest | `mutable_inputs.train_manifest`, `.train_completion`, `.train_args`, each with `path` and `sha256`; these three require explicit launcher bindings |
| Bound training manifest | `protocol`, `effective_args.protocol`, backbone/yaw/recipe fields, `train_data_identity.protocol`, `mutable_inputs.seen_split`, `timing_limits.protocol`, `source_closures.training`, `source_closures.launcher` |
| Bound training completion | `train_manifest_sha256`; `outputs['epoch_012.pth']` tied to eval `checkpoint_sha256`; `outputs['args.json']` tied to bound training args |
| Bound training `args.json` | `protocol`, backbone/yaw/recipe fields, tier/configuration, and `train_batches_per_epoch` |

**Evaluation `split` and training `protocol` are distinct.** Exp07 eval artifacts do not acquire exp05’s `tier` or `args_json_sha256` metadata automatically.

The inherited producer allowlists reject `seen_split`—and the base paired producer also rejects `train_args`. Round 4 needs an exp07 admission wrapper that explicitly validates those bindings. Preserve every evaluation check for `released_seen`; exempt it only from training provenance.

The four trainer parity cases, three evaluator parity cases, calibration, and final integrative review remain outstanding gates.
158,883
**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15

## Verdict: request changes

**Blocking findings: 1 — the seen dataset module is absent from the evaluator’s recorded source closure.**

Round-2 blockers 1–2 and should-fix 3 are closed by cycle 0. Round 3 has the new blocker below.

Reviewed every changed line in `a035d7f..ab585d0`. No repository files were modified, no GPU was used, and no processes were signalled.

## Findings

1. **Blocker — the evaluator does not bind the seen dataset code.**  
   [tools/exp07_eval.py:20](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_eval.py:20), [tools/exp07_eval.py:55](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_eval.py:55)

   The frozen `build_dataset` helper imports `treble_xRIR_seen_dataset` inside its function. Importing `tools.exp07_eval` therefore does not load that module, and `source_closure()` omits it.

   **Reproduced:** `source_closure('tools.exp07_eval', REPO)` returns 15 files, excluding `treble_multi_room_dataset/treble_xRIR_seen_dataset.py`. Neither the frozen evaluator closure nor the writer closure covers it. With the recorded bindings intact, an in-memory simulated change to that module produces `revalidate(...) == []`.

   Consequently, the approved evaluator digest does not certify all code used to select and serve seen queries. The pickle binding protects the split file, but does not cover its implementing module.

   **Minimal fix:** import the seen dataset module at module scope in the new entry point, while retaining the frozen helper if desired. Add closure-membership coverage and a regression proving changes to that module refuse. Leave the pinned module itself untouched.

2. **Should-fix — the seen GPU parity reference shares the implementation under test and bypasses the launcher.**  
   [tests/test_exp07_eval_gpu.py:86](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_exp07_eval_gpu.py:86), [tests/test_exp07_eval_gpu.py:108](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_exp07_eval_gpu.py:108)

   `direct_reference()` calls `exp04_eval.evaluate_p_batch`, which is also the evaluated P path. A common error there can pass both sides. The test’s `run()` helper also fabricates a handshake and invokes the evaluator directly; it does not exercise the planned seen run through the launcher.

   **Minimal fix:** obtain the reference P results from frozen `eval_yaw_rotation.evaluate_batch`, and add the planned launcher integration case. Retain K=8/K=1 and the final nine samples.

   The **25-query selection is correct**: it contains one complete canonical batch followed by the actual final nine queries.

3. **Should-fix — manifest construction records the split identity too late.**  
   [tools/exp07_manifests.py:56](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_manifests.py:56), [tools/exp07_manifests.py:72](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_manifests.py:72)

   The builder first constructs datasets and writes manifests, then obtains `seen_split_identity`. A replacement during construction can make the index attribute old-split manifests—or a mixed grid—to the replacement pickle.

   **Reproduced with a stub dataset and in-memory outputs:** dataset selection occurred before replacement; the build succeeded and recorded only the new identity. This repeats the timing pattern repaired in the training launcher.

   **Minimal fix:** capture the split binding before constructing either dataset, recheck it before publishing the index, and store the captured binding. Add mutation-between-K coverage. Downstream query-list checking limits the damage, but the index itself must describe its inputs correctly.

4. **Nit — explicitly binding the correct split file is rejected.**  
   [tools/exp07_eval.py:82](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_eval.py:82)

   The shared launcher stores explicit binding paths as absolute paths; `seen_split_identity()` returns a repository-relative path. Direct dictionary comparison rejects even the canonical file.

   **Reproduced:** `--bind-input seen_split=<REPO>/treble_multi_room_dataset/seen_test_split.pkl` raises `seen_split must bind ...`.

   **Minimal fix:** compare resolved paths and digests, then store the canonical binding. Automatic binding works.

## Verified

### Cycle 0: round-2 findings closed

Using base/head modules and in-memory receipt bytes with recomputed outer hashes:

- The valid seen receipt passes.
- All six substituted-completion cases pass at `a035d7f` and refuse at `ab585d0`: wrong manifest digest, protocol, yaw, tier, backbone, and the combined original forgery.
- Two one-hour full attempts permit another 30-hour attempt at the base and refuse at the head.
- A fresh receipt and renewal cannot reset the seen count.
- Smoke/probe rows do not count; zero or one full attempt remains admissible.
- The third attempt refuses before directory creation, resource checks, or field construction.
- The equivalent unseen ledger remains admissible.
- **Five real exp_05 receipts** have identical successful base/head validation results.
- The training split-before-inventory regression passes, including mutation refusal.

The receipt now establishes both the completion-to-manifest digest link and the requested arm identity.

### Manifests

The stub build produced four K/seed manifests. All four index records matched the saved manifests’ semantic hashes, file hashes, and entry counts.

Verified overwrite refusal with original bytes preserved, the default 6,217-entry gate, wrong-cwd refusal, and the planned defaults `{42…46} × {8,1}`. Finding 3 concerns split identity capture.

### Evaluator and historical paths

- After removing only the new optional parameter, docstring change, and factory-selection expression, **`run_exp04`’s AST is identical to the base**. Default exp_04/exp_05 construction, loader configuration, numerical operations, padding, aggregation, and serialization remain unchanged.
- **Five historical launcher command vectors** match base/head exactly: default exp04, explicit exp05, and tiers S/M/L.
- Real seen dataset readback gives **6,217 queries / 131 rooms**.
- Real cross-split checks refuse **6,337 unseen queries under seen** and **6,217 seen queries under unseen**, before model construction.
- `ManifestDataset` validates the full query list before applying `max_samples`.
- Canonical padding remains the frozen implementation.

This supports unchanged default numerical execution. Actual GPU output parity remains deferred; literal JSON bytes also contain elapsed-time and provenance metadata.

The unseen GPU test compares complete payloads after excluding exactly:

`elapsed_min`, `split`, `seen_split_sha256`, `eval_manifest_sha256`.

These are within plan §2’s permitted exceptions. Other provenance fields remain equal in this fixture and are still compared. No numerical fields are exempted.

### Evaluation launcher

In-memory runs through the actual finalisation function verified:

- Successful completion records `split` and `seen_split_sha256`.
- Wrong split metadata in the outputs refuses completion.
- A split pickle changed after child completion refuses with `mutable input mismatch: seen_split`.
- Failed cases leave no completion.

Field construction gives:

| Manifest entries | `max_samples` | `confirmatory` |
|---:|---:|---|
| 6217 | 0 | true |
| 6216 | 0 | false |
| 6218 | 0 | false |
| 6217 | 16 | false |

The paired entry/split routing and requested CLI refusals pass. A short standalone manifest still fails the real dataset’s query-list check; `confirmatory=False` does not make it a valid full-split manifest.

Legacy eval manifests already contained `split='unseen'`. Their metas and completions retain their previous field sets.

### Tests and hygiene

Environment: pinned Python 3.8 interpreter, worktree `PYTHONPATH`, local AcousticRooms root, `CUDA_VISIBLE_DEVICES=''`, and `PYTHONDONTWRITEBYTECODE=1`.

| Check | Result |
|---|---|
| `python -m pytest tests -q -p no:cacheprovider` | Blocked before collection: no writable temporary directory |
| Whole-suite replay with `--capture=sys`, cache accommodations, and explicit unwritable-fixture skips | **1473 collected; 487 passed, 986 skipped, 0 failed** |
| Skip breakdown | **953 sandbox-related; 33 CUDA/environment** |
| `py_compile.compile(..., doraise=True)` with in-memory bytecode sink | **13/13 changed Python files passed** |
| `git diff --check a035d7f..ab585d0` | Exit 0 |
| Pinned-path diff | Empty |
| Scope | Reported 13 files; **857 insertions / 11 deletions** |
| Co-author trailers | Correct on all eight commits |

The replay **does not independently certify 1,432 passed / 41 skipped**. Blocked fixtures include filesystem-backed manifest creation, receipt/ledger writes, temporary Git repositories, and real child/finalisation tests. Critical cases were additionally exercised in memory as described above.

| Commit | Changed lines |
|---|---:|
| `9e8cdf7` | 109 |
| `b5cf876` | 28 |
| `f80a119` | 179 |
| `28a876a` | 29 |
| `71c5101` | **206** |
| `23bdddf` | 51 |
| `2e7703f` | 115 |
| `ab585d0` | 151 |

**Process deviation:** `71c5101` exceeds the round’s limit. Its content is coherent and reviewable; retain the recorded deviation and split future changes before committing.

The pre-existing untracked `checkpoints` symlink remains unchanged.

## Coder decisions

The supplied records do not contain the numbered decisions 2–6 list, so I cannot reliably assign those numbers. Assessments of the stated choices:

| Choice | Assessment |
|---|---|
| Reuse frozen `build_dataset` and `ManifestDataset` | Accept the construction path; require closure repair in finding 1 |
| Optional `dataset_factory=None` hook | Accept; preserves the existing default execution |
| Add `protocol` to timing limits and update two equality assertions | Accept; supplies the discriminator without changing unseen numerical limits |
| Mark non-6,217 manifests non-confirmatory | Accept; retain independent query-list and producer checks |
| Use 25 queries for final-batch parity at both K | Accept the cohort; repair the reference and launcher coverage in finding 2 |
| Exclusive manifest creation and repository-root requirement | Accept; repair split capture in finding 3 |

## Fitness for round 4

**Resolve finding 1 before closing round 3 and pinning its evaluator closure.** Correct findings 2–3 before using the corresponding parity evidence or manifest index.

Round-4 admission must consume these exact locations:

| Artifact | Required fields |
|---|---|
| Reference index | `seen_split.path`, `seen_split.sha256`; `manifests[filename].manifest_hash`, `.file_sha256`, `.entries`, `.num_shot`, `.seed`; grid fields `seeds`, `num_shots` |
| Eval manifest | `split`, `seen_split_sha256`, `mutable_inputs.seen_split.{path,sha256}`, `split_count`, `n_samples`, `max_samples`, `confirmatory`, `allow_dirty_used` |
| Eval recipe/data | `num_shot`, `manifest_seed`, `gl_seed`, `manifest_hash`, `manifest_file_sha256`, `conditions`, `yaw_cols`, `acoustic_cols`, `e_acoustic_cols`, `batch_size`, `batch_canonical`, `tf32`, `data_identity` |
| Both output metas | `split`, `seen_split_sha256`, `eval_manifest_sha256`, plus recipe/sample metadata |
| Eval completion | `split`, `seen_split_sha256`, `eval_manifest_sha256`, `confirmatory`, `allow_dirty_used`, `outputs`, `child_exit_status` |
| Eval code pins | **`source_closures.entrypoint.sha256`** for `tools.exp07_eval`; `source_closures.writer.sha256`; separately `evaluator_closure` for frozen `eval_yaw_rotation` |
| Training bindings in eval manifest | `mutable_inputs.train_manifest`, `.train_completion`, `.train_args`, each with `path` and `sha256`; these three require explicit launcher bindings |
| Bound training manifest | `protocol`, `effective_args.protocol`, backbone/yaw/recipe fields, `train_data_identity.protocol`, `mutable_inputs.seen_split`, `timing_limits.protocol`, `source_closures.training`, `source_closures.launcher` |
| Bound training completion | `train_manifest_sha256`; `outputs['epoch_012.pth']` tied to eval `checkpoint_sha256`; `outputs['args.json']` tied to bound training args |
| Bound training `args.json` | `protocol`, backbone/yaw/recipe fields, tier/configuration, and `train_batches_per_epoch` |

**Evaluation `split` and training `protocol` are distinct.** Exp07 eval artifacts do not acquire exp05’s `tier` or `args_json_sha256` metadata automatically.

The inherited producer allowlists reject `seen_split`—and the base paired producer also rejects `train_args`. Round 4 needs an exp07 admission wrapper that explicitly validates those bindings. Preserve every evaluation check for `released_seen`; exempt it only from training provenance.

The four trainer parity cases, three evaluator parity cases, calibration, and final integrative review remain outstanding gates.
