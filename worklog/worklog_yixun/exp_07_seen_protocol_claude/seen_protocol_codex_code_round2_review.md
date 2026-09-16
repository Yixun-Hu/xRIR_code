**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15 · log `seen_protocol_2026-09-15_11:42:33_codex_code_round2.log`

**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15

## Verdict: request changes

**Blocking findings: 1 — missing one-retry enforcement; 2 — incomplete probe receipt binding.**

Reviewed every changed line in `git diff d7511fb..a035d7f`. No repository files were changed, no GPU was used, and no processes were signalled.

## Findings

1. **Blocker — a third full attempt is allowed.**  
   [tools/exp04_launcher.py:792](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp04_launcher.py:792), [tools/exp05_gates.py:120](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp05_gates.py:120)

   The shared budget code sums hours but never limits the number of full attempts. Routing seen runs through it therefore leaves the plan’s one-retry rule unenforced.

   **Reproduced:** a seen-arm ledger containing two one-hour aborted full attempts accepts another full attempt projected at 30 hours under a 45-hour ceiling. Both budget checks pass.

   **Minimal fix:** before creating a seen full attempt, refuse when two full attempts already exist in the arm’s ledger. Smoke/probe attempts must not count; a new receipt or `--renew-ceiling` must not reset this count. Add third-attempt refusal coverage while preserving historical unseen behavior.

2. **Blocker — the receipt does not establish that its completion certifies its manifest.**  
   [tools/exp05_gates.py:42](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp05_gates.py:42)

   Validation hashes the manifest and completion separately but never checks:
   `completion.train_manifest_sha256 == probe_attempt.train_manifest_sha256`.
   Its comparison with `completion.metrics.probe` also omits protocol, yaw, tier and backbone.

   **Reproduced with in-memory files:** a correctly located `seen_aug` manifest was accepted alongside a completion whose manifest digest was `'0' * 64` and whose probe metadata declared `protocol='unseen', yaw_aug=0`. Matching timing fields and accurate outer file hashes sufficed.

   This weakness is inherited from exp_05 and now affects seen admission.

   **Minimal fix:** require the completion-to-manifest digest link and matching probe identity/recipe fields. Add a regression that recomputes the outer file hashes after substituting unrelated completion evidence.

3. **Should-fix — the split is first bound after inventory construction.**  
   [tools/exp04_launcher.py:354](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp04_launcher.py:354)

   `train_data_identity()` selects and hashes training files before `seen_split_identity()` records the pickle digest. A same-count pickle replacement during that interval can produce an old-partition inventory bound to a new pickle.

   **Reproduced with stubbed inventory construction:** `build_fields()` accepted an inventory selected under the old split, recorded the replacement split, and retained B=9,265. Subsequent hashing of the listed WAVs does not establish that they belong to the recorded partition.

   **Minimal fix:** capture the split identity before inventory construction, verify it afterward, and bind that original identity. Add a mutation-during-construction refusal test.

4. **Nit — correct the aggregate change report.**  
   [MAIN-tree worklog:23](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_worklog.md:23)

   The specified range contains **699 insertions / 82 deletions across 15 files**, rather than 697/80. `tests/test_exp05_launcher.py` is unchanged; production code repairs its failures. Update the record accordingly.

## Verified

### Historical exp_04 / exp_05 paths

Executed the original five parity tests against launcher modules loaded from the respective Git revisions, using the real historical `args.json` files:

| Revision | Result |
|---|---|
| `d7511fb` | All five fail: `control mismatch: protocol` |
| `36e385d` | All five pass |
| `a035d7f` | All five pass |

Thus the five tests **were red at round 1’s endpoint**. The repair is correct and minimal: `setdefault('protocol', 'unseen')` normalizes historical records without excluding protocol from ordinary unseen comparisons. The tests themselves were not weakened.

Additional base/head comparisons established:

- **6/6** historical command vectors, effective-argument dictionaries and golden checks match: exp_04 smoke/full plus four exp_05 full arms.
- **4/4** exp_05 probe command vectors match.
- **7/7 real historical receipts** produce identical admission outcomes: six accepted, one old exp_04 receipt refused.
- Historical ledger, completion and recovery branches retain their existing behavior by inspection. Their filesystem-dependent regression cases could not all execute here.

### Seen arms, arguments and provenance

- All six goldens have the required recipe tokens; smoke vectors retain exp_04’s three-micro-batch, batch-4, two-test-batch, `--no-save` form.
- Root/backbone/yaw/protocol cross-refusals pass, including cylindrical+yaw rejection.
- All three real historical comparators pass. `seen_aug` uses exp_04’s recorded environment. Returned differences contain only protocol and permitted operational exclusions; wrong comparators and recipe/type changes refuse.
- Full seen runs require **9,265** batches; unseen runs retain **9,261**. Banner checks distinguish the three arms correctly.
- Top-level manifest `protocol` is added only for seen runs.
- Seen fields require `seen_split`; completion revalidation checks its bytes, and recovery checks its canonical binding, closure membership and file count. Simulated changed split bytes refuse. Finding 3 concerns the earlier binding window.
- Both live caches validated read-only, with unchanged file bytes:

| Cache | Files | Returned protocol |
|---|---:|---|
| Historical unseen | 296,334 | `unseen`, although absent on disk |
| Separate seen | 296,454 | `seen` |

- The training closure contains the seen dataset module. The exp_03 closure/digest regression passed with dependency-cache accommodations; pinned files are untouched.

### Probe, timing and renewal

Verified the single-arm seen probe commands, yaw flag and projection:

`T_epoch = 9265 × mean(t_micro) + t_test + t_save`; `T_run = 12 × T_epoch`.

Adversarial checks refused wrong commit/GPU/protocol/yaw/batch count, unclean snapshots and excessive duration. Exactly **60 hours** passes; exceeding it refuses.

All three seen arms enforce the live and recorded epoch-one limit of **1.05 × T_epoch**. The shared abort path maps that failure to `_ABORTED_slow`. Renewal checks preserve consumed hours, prevent an ordinary replacement receipt from raising the recorded ceiling, and accept a properly timestamped renewal with a new receipt. Findings 1 and 2 remain.

### Smoke dry run

The stub-child test at [tests/test_exp07_launcher.py:285](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_exp07_launcher.py:285) correctly checks these files before child execution:

- `effective_args.json`
- `train_manifest.json`
- `train_inventory.json`

It also checks protocol, split binding, inventory-sidecar digest, closure membership and completion. **Its temporary Git-repository fixture was blocked by the sandbox**, so this review verified its implementation by reading rather than independently replaying its filesystem writes.

### Tests and hygiene

The requested ordinary pytest invocation stopped before collection because no writable temporary directory exists. Dependency imports also required NumPy/Numba/Matplotlib cache accommodations.

A whole-suite replay using in-memory capture, dependency-cache accommodations and explicit skips for unwritable fixtures produced:

**463 passed, 0 failed, 942 skipped** — 912 sandbox-related skips and 30 GPU/environment skips.

All **1,405 tests** collected, including the new **50 launcher + 15 probe** cases. This does **not independently certify** the Coder’s 1,367 passed / 38 skipped report. Blocked coverage includes filesystem smoke, receipt creation, ledger writes, finalization/recovery and provenance mutation fixtures.

| Check | Result |
|---|---|
| `bash -n tools/exp04_launch.sh` | Exit 0 |
| `git diff --check d7511fb..a035d7f` | Exit 0 |
| Python 3.8 `py_compile.compile(..., doraise=True)` with an in-memory bytecode sink | All 8 changed Python files pass |
| Pinned-path diff | Empty |
| Intended tracked-file scope | Confirmed |
| Opus co-author trailers | Present on all nine commits |

| Commit | Changed lines |
|---|---:|
| `2fe4395` | 9 |
| `36e385d` | 193 |
| `68fb737` | 119 |
| `6d9578d` | 83 |
| `57c3a36` | **283** |
| `8215702` | 89 |
| `9a0fbe0` | 7 |
| `e4832f2` | 2 |
| `a035d7f` | 4 |

**Process deviation:** `57c3a36` exceeds the round’s 200-line limit. Its probe → receipt → timing-gate integration is coherent and reviewable as one unit. Keep the recorded deviation; do not amend/rebase to conceal it.

## First-launch preparation

**Yes: the Planner should pre-build the inventory before reserving a GPU for the probe.** Inventory construction occurs after the resource check and delays child launch.

This has now happened: the MAIN-tree notebook records completion at **11:44:23**, taking **73 seconds**. I independently validated the resulting cache, whose inventory digest begins `b18e9ec3488c…`.

Exact pinned-interpreter command:

```bash
cd /home/yixunhu/codespace/xRIR_code_wt07
PYTHONPATH="$PWD" \
XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms \
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home/yixunhu/miniconda3/envs/xRIR/bin/python -c \
"from tools.provenance import train_data_identity; r = train_data_identity('/home/yixunhu/data_cache/AcousticRooms', protocol='seen', cache_path='ckpt/exp07/train_inventory_seen.json'); print(r['inventory_files'], r['inventory_sha256'])"
```

Cache hits still enumerate/stat the files; completion still rehashes the bound inventory. Training inventory coverage remains WAV-only.

The pre-existing untracked `checkpoints` symlink currently makes the full-launch dirty-tree gate refuse. Resolve its local ignore status before launching. The four deferred GPU parity cases, seen finite-step/alignment checks, real smokes and final integrative review remain required.

## Round 3 requirements

- Implement the seen entry point and dataset-factory hook in `tools/exp04_eval.py`; preserve the unseen default and every frozen file.
- Construct the seen dataset with `num_shot=K` from repository-root cwd. Bind/revalidate the split pickle and record its digest in manifest and output metadata.
- Enforce cross-split refusals and all **6,217 queries**; preserve batch-16 padding, including the final **9 real samples**, at K=1 and K=8.
- Prove unseen 32-query bit parity and seen parity against direct frozen-function calls. Limit metadata exceptions to those listed in plan §2.
- Build/index the ten K/seed reference manifests before evaluation; retain condition P, k=0 and TF32 off.
- Preserve training bindings for the three new arms. The released checkpoint’s exemption covers training provenance only. Complete the pre-registered calibration before admitting seen-arm evaluations.

**Round 2 should remain open until blocking findings 1 and 2 are fixed and re-verified.**
202,060
**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15

## Verdict: request changes

**Blocking findings: 1 — missing one-retry enforcement; 2 — incomplete probe receipt binding.**

Reviewed every changed line in `git diff d7511fb..a035d7f`. No repository files were changed, no GPU was used, and no processes were signalled.

## Findings

1. **Blocker — a third full attempt is allowed.**  
   [tools/exp04_launcher.py:792](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp04_launcher.py:792), [tools/exp05_gates.py:120](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp05_gates.py:120)

   The shared budget code sums hours but never limits the number of full attempts. Routing seen runs through it therefore leaves the plan’s one-retry rule unenforced.

   **Reproduced:** a seen-arm ledger containing two one-hour aborted full attempts accepts another full attempt projected at 30 hours under a 45-hour ceiling. Both budget checks pass.

   **Minimal fix:** before creating a seen full attempt, refuse when two full attempts already exist in the arm’s ledger. Smoke/probe attempts must not count; a new receipt or `--renew-ceiling` must not reset this count. Add third-attempt refusal coverage while preserving historical unseen behavior.

2. **Blocker — the receipt does not establish that its completion certifies its manifest.**  
   [tools/exp05_gates.py:42](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp05_gates.py:42)

   Validation hashes the manifest and completion separately but never checks:
   `completion.train_manifest_sha256 == probe_attempt.train_manifest_sha256`.
   Its comparison with `completion.metrics.probe` also omits protocol, yaw, tier and backbone.

   **Reproduced with in-memory files:** a correctly located `seen_aug` manifest was accepted alongside a completion whose manifest digest was `'0' * 64` and whose probe metadata declared `protocol='unseen', yaw_aug=0`. Matching timing fields and accurate outer file hashes sufficed.

   This weakness is inherited from exp_05 and now affects seen admission.

   **Minimal fix:** require the completion-to-manifest digest link and matching probe identity/recipe fields. Add a regression that recomputes the outer file hashes after substituting unrelated completion evidence.

3. **Should-fix — the split is first bound after inventory construction.**  
   [tools/exp04_launcher.py:354](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp04_launcher.py:354)

   `train_data_identity()` selects and hashes training files before `seen_split_identity()` records the pickle digest. A same-count pickle replacement during that interval can produce an old-partition inventory bound to a new pickle.

   **Reproduced with stubbed inventory construction:** `build_fields()` accepted an inventory selected under the old split, recorded the replacement split, and retained B=9,265. Subsequent hashing of the listed WAVs does not establish that they belong to the recorded partition.

   **Minimal fix:** capture the split identity before inventory construction, verify it afterward, and bind that original identity. Add a mutation-during-construction refusal test.

4. **Nit — correct the aggregate change report.**  
   [MAIN-tree worklog:23](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_07_seen_protocol_claude/seen_protocol_worklog.md:23)

   The specified range contains **699 insertions / 82 deletions across 15 files**, rather than 697/80. `tests/test_exp05_launcher.py` is unchanged; production code repairs its failures. Update the record accordingly.

## Verified

### Historical exp_04 / exp_05 paths

Executed the original five parity tests against launcher modules loaded from the respective Git revisions, using the real historical `args.json` files:

| Revision | Result |
|---|---|
| `d7511fb` | All five fail: `control mismatch: protocol` |
| `36e385d` | All five pass |
| `a035d7f` | All five pass |

Thus the five tests **were red at round 1’s endpoint**. The repair is correct and minimal: `setdefault('protocol', 'unseen')` normalizes historical records without excluding protocol from ordinary unseen comparisons. The tests themselves were not weakened.

Additional base/head comparisons established:

- **6/6** historical command vectors, effective-argument dictionaries and golden checks match: exp_04 smoke/full plus four exp_05 full arms.
- **4/4** exp_05 probe command vectors match.
- **7/7 real historical receipts** produce identical admission outcomes: six accepted, one old exp_04 receipt refused.
- Historical ledger, completion and recovery branches retain their existing behavior by inspection. Their filesystem-dependent regression cases could not all execute here.

### Seen arms, arguments and provenance

- All six goldens have the required recipe tokens; smoke vectors retain exp_04’s three-micro-batch, batch-4, two-test-batch, `--no-save` form.
- Root/backbone/yaw/protocol cross-refusals pass, including cylindrical+yaw rejection.
- All three real historical comparators pass. `seen_aug` uses exp_04’s recorded environment. Returned differences contain only protocol and permitted operational exclusions; wrong comparators and recipe/type changes refuse.
- Full seen runs require **9,265** batches; unseen runs retain **9,261**. Banner checks distinguish the three arms correctly.
- Top-level manifest `protocol` is added only for seen runs.
- Seen fields require `seen_split`; completion revalidation checks its bytes, and recovery checks its canonical binding, closure membership and file count. Simulated changed split bytes refuse. Finding 3 concerns the earlier binding window.
- Both live caches validated read-only, with unchanged file bytes:

| Cache | Files | Returned protocol |
|---|---:|---|
| Historical unseen | 296,334 | `unseen`, although absent on disk |
| Separate seen | 296,454 | `seen` |

- The training closure contains the seen dataset module. The exp_03 closure/digest regression passed with dependency-cache accommodations; pinned files are untouched.

### Probe, timing and renewal

Verified the single-arm seen probe commands, yaw flag and projection:

`T_epoch = 9265 × mean(t_micro) + t_test + t_save`; `T_run = 12 × T_epoch`.

Adversarial checks refused wrong commit/GPU/protocol/yaw/batch count, unclean snapshots and excessive duration. Exactly **60 hours** passes; exceeding it refuses.

All three seen arms enforce the live and recorded epoch-one limit of **1.05 × T_epoch**. The shared abort path maps that failure to `_ABORTED_slow`. Renewal checks preserve consumed hours, prevent an ordinary replacement receipt from raising the recorded ceiling, and accept a properly timestamped renewal with a new receipt. Findings 1 and 2 remain.

### Smoke dry run

The stub-child test at [tests/test_exp07_launcher.py:285](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_exp07_launcher.py:285) correctly checks these files before child execution:

- `effective_args.json`
- `train_manifest.json`
- `train_inventory.json`

It also checks protocol, split binding, inventory-sidecar digest, closure membership and completion. **Its temporary Git-repository fixture was blocked by the sandbox**, so this review verified its implementation by reading rather than independently replaying its filesystem writes.

### Tests and hygiene

The requested ordinary pytest invocation stopped before collection because no writable temporary directory exists. Dependency imports also required NumPy/Numba/Matplotlib cache accommodations.

A whole-suite replay using in-memory capture, dependency-cache accommodations and explicit skips for unwritable fixtures produced:

**463 passed, 0 failed, 942 skipped** — 912 sandbox-related skips and 30 GPU/environment skips.

All **1,405 tests** collected, including the new **50 launcher + 15 probe** cases. This does **not independently certify** the Coder’s 1,367 passed / 38 skipped report. Blocked coverage includes filesystem smoke, receipt creation, ledger writes, finalization/recovery and provenance mutation fixtures.

| Check | Result |
|---|---|
| `bash -n tools/exp04_launch.sh` | Exit 0 |
| `git diff --check d7511fb..a035d7f` | Exit 0 |
| Python 3.8 `py_compile.compile(..., doraise=True)` with an in-memory bytecode sink | All 8 changed Python files pass |
| Pinned-path diff | Empty |
| Intended tracked-file scope | Confirmed |
| Opus co-author trailers | Present on all nine commits |

| Commit | Changed lines |
|---|---:|
| `2fe4395` | 9 |
| `36e385d` | 193 |
| `68fb737` | 119 |
| `6d9578d` | 83 |
| `57c3a36` | **283** |
| `8215702` | 89 |
| `9a0fbe0` | 7 |
| `e4832f2` | 2 |
| `a035d7f` | 4 |

**Process deviation:** `57c3a36` exceeds the round’s 200-line limit. Its probe → receipt → timing-gate integration is coherent and reviewable as one unit. Keep the recorded deviation; do not amend/rebase to conceal it.

## First-launch preparation

**Yes: the Planner should pre-build the inventory before reserving a GPU for the probe.** Inventory construction occurs after the resource check and delays child launch.

This has now happened: the MAIN-tree notebook records completion at **11:44:23**, taking **73 seconds**. I independently validated the resulting cache, whose inventory digest begins `b18e9ec3488c…`.

Exact pinned-interpreter command:

```bash
cd /home/yixunhu/codespace/xRIR_code_wt07
PYTHONPATH="$PWD" \
XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms \
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home/yixunhu/miniconda3/envs/xRIR/bin/python -c \
"from tools.provenance import train_data_identity; r = train_data_identity('/home/yixunhu/data_cache/AcousticRooms', protocol='seen', cache_path='ckpt/exp07/train_inventory_seen.json'); print(r['inventory_files'], r['inventory_sha256'])"
```

Cache hits still enumerate/stat the files; completion still rehashes the bound inventory. Training inventory coverage remains WAV-only.

The pre-existing untracked `checkpoints` symlink currently makes the full-launch dirty-tree gate refuse. Resolve its local ignore status before launching. The four deferred GPU parity cases, seen finite-step/alignment checks, real smokes and final integrative review remain required.

## Round 3 requirements

- Implement the seen entry point and dataset-factory hook in `tools/exp04_eval.py`; preserve the unseen default and every frozen file.
- Construct the seen dataset with `num_shot=K` from repository-root cwd. Bind/revalidate the split pickle and record its digest in manifest and output metadata.
- Enforce cross-split refusals and all **6,217 queries**; preserve batch-16 padding, including the final **9 real samples**, at K=1 and K=8.
- Prove unseen 32-query bit parity and seen parity against direct frozen-function calls. Limit metadata exceptions to those listed in plan §2.
- Build/index the ten K/seed reference manifests before evaluation; retain condition P, k=0 and TF32 off.
- Preserve training bindings for the three new arms. The released checkpoint’s exemption covers training provenance only. Complete the pre-registered calibration before admitting seen-arm evaluations.

**Round 2 should remain open until blocking findings 1 and 2 are fixed and re-verified.**
