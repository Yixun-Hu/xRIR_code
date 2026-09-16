**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15 · log `seen_protocol_2026-09-15_13:32:42_codex_code_round4.log`

**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15

## Verdict: request changes

**Blocking findings: 1 and 2 below.** Round-3 items are closed. The paired statistics and table aggregation check out, but training-provenance admission remains incomplete.

Reviewed every changed line in `ab585d0..7b3ad6c`. No repository or live-data files were modified; no GPU was used.

## Findings

### 1. Blocker — training closure pins certify unchecked digest labels

[tools/exp07_table.py:104](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_table.py:104)

The contract compares `source_closures.training.sha256` and `.launcher.sha256` with approval pins without recomputing those digests from their `files` records. The inherited collector validates the **evaluation** closures; it does not recursively validate closures inside the bound training manifest.

**Reproduced:** independently replacing either training or launcher `files` with `[]`, or replacing training records with an unreviewed `wrong.py` record, still produces eight table rows with `deviations == []`. Outer manifest/completion bindings were refreshed; approval pins remained unchanged.

Thus the equal-training-closures check proves equality of supplied labels, not the recorded source identities.

**Minimal fix:** validate each training closure’s complete records and recompute its digest before comparing with the pin. Preserve A7’s distinction between immutable reviewed source identity and permitted, recorded post-spawn working-tree drift. Add these mutations to the refusal tests.

### 2. Blocker — the training manifest’s data and execution contract are not validated

[tools/exp07_table.py:97](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_table.py:97)

Beyond the split binding, the manifest checks mostly establish that several `protocol` strings equal `"seen"`. The recipe check applies to bound `args.json`, without establishing consistency with the manifest’s actual execution specification.

**Each of these independent mutations was admitted with no deviations:**

- `train_data_identity.inventory_files = 1`
- A substituted `train_data_identity.inventory_sha256`
- `mode = "smoke"`
- `allow_dirty = true`
- `effective_args.epochs = 1`
- Wrong `effective_args.backbone` or `effective_args.yaw_aug`
- `effective_args.save_dir` pointing elsewhere
- `timing_limits.projection_hours = 1000`

The existing fixture itself supplies a training inventory digest without an inventory or bound inventory sidecar, so its successful admission does not exercise the planned data-identity contract.

**Minimal fix:** validate the real training inventory/sidecar and digest, the 296,454-file seen identity, full-run status, relevant timing constraints, and normalized consistency between `effective_args`, bound runtime args, and the attempt directory. Bind the additional validated evidence into producer provenance. Extend the fixture to represent those real artifacts.

### 3. Should-fix — required spectral columns can disappear silently

[tools/exp07_table.py:212](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_table.py:212)

Removing `loss` and `log_mse` from both outputs for all five seeds of one group still produces an admitted table without deviations. The collector requires the acoustic metrics, while aggregation silently skips absent recognized metrics.

Plan §2 requires both spectral metrics in canonical JSON.

**Minimal fix:** require all five registered output metrics for every group, with named missing-metric refusals. Keep the existing allowance for recognized optional diagnostics.

### 4. Should-fix — pin-finalisation tests do not support the intended lifecycle completely

[tests/test_exp07_profiles.py:142](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_exp07_profiles.py:142), [tests/test_exp07_profiles.py:199](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_exp07_profiles.py:199)

The committed-file test permanently asserts that approval equals the all-null template. It will fail after legitimate pin finalisation. Also, the inventory placeholder receives a format check but no artifact-agreement test; the new-arm profile digest placeholders likewise lack the advertised agreement check.

**Minimal fix:** test all-null semantics against an isolated template, allow either valid committed lifecycle state, and test available inventory/checkpoint identities against their authoritative artifacts. Clarify that runtime approval supplies new-arm checkpoint digests—the admission code currently replaces the profile placeholders with those values.

The two current manifest/query agreement failures are **expected pending-pin failures**, described below, rather than defects in those agreement tests.

### 5. Should-fix — exploratory Markdown omits the deviations

[tools/exp07_table.py:270](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_table.py:270)

The shared Markdown renderer ignores `exploratory` and `deviations`. A synthetic table with 120 admission deviations renders the ordinary “Model comparison” table without displaying those deviations or a prominent exploratory status. A command flag or filename is insufficient disclosure.

**Minimal fix:** provide an exp07 rendering adapter that reads these fields from JSON and includes them before the output hashes and sidecar are finalized.

### 6. Nit — loaded approval is not recursively immutable

[tools/exp07_profiles.py:30](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_profiles.py:30)

The registered profiles are immutable, but `freeze()` leaves lists mutable. Consequently:

```python
pins["closures"]["training_launcher"].append("f" * 64)
```

succeeds, including on the returned committed template.

**Minimal fix:** freeze sequences recursively and add a nested-list mutation test.

### 7. Nit — two commits exceed the round’s size limit

- `8772121`: **210 changed lines**; fixture plus its layout test.
- `eff1b2a`: **224 changed lines**; paired-cell implementation/tests plus producer-key admission plumbing.

Both are coherent, reviewable units. Accept the recorded process deviations; split future commits before crossing the limit.

## Verified

### Round-3 closure

- Base evaluator closure: **15 files**, seen dataset module absent.
- Head evaluator closure: **16 files**, with precisely that module added.
- Exp04 evaluator/writer closures retain **14/15 files**, excluding the seen module.
- Simulated changed seen-module bytes:
  - Old membership: `revalidate(...) == []`
  - Head: `source.entrypoint.treble_multi_room_dataset/treble_xRIR_seen_dataset.py`
- Base manifest builder accepts replacement between K values and publishes an index.
- Head records call order `split → dataset8 → dataset1 → split` and refuses replacement without publishing the index.
- Canonical absolute and indirect split paths are accepted; another location or wrong digest is refused.
- GPU parity now uses frozen `eval_yaw_rotation.evaluate_batch`. Direct tests retain the actual final nine queries; launcher tests exercise a 16+9 bounded batch at both K values. **Execution remains deferred.**

### Profiles and admission

Both literal digests equal their files:

| Artifact | SHA-256 |
|---|---|
| Released checkpoint | `1762c702ee23a8f584e67c4b5b052b7483237e6471bb5e58774b66d514aec2f8` |
| Seen split pickle | `bc97850b090198cf956dc7975a2da611dd058f8b0f83adee7f93f7b7fad7720f` |

Verified immutable registered profiles, stable profile digests, and recipe agreement with all three committed full-run goldens.

The approval loader accepts valid null/filled states and rejects partial filling, malformed digests, wrong keys/path/epoch/types, invalid launcher lists, and bytes differing from HEAD.

Using the actual fixture construction with filesystem operations redirected into memory:

- Valid fixture: **40 runs, eight groups, five seeds each, zero deviations**.
- All **19 supplied refusal mutations** produce their named deviations.
- **31 additional recipe/full-run/arm/tier field mutations** produce named refusals.
- Wrong inner completion-to-manifest and completion-to-args hashes refuse despite repaired outer bindings.
- Eight evaluation mutations also refuse on the released row.
- K, grid, P-only, batch, TF32, canonical padding, GL seed, bounded evaluation, and confirmatory flags refuse as expected.
- Finding 1–2 mutations establish the remaining training-provenance gaps.

### Numerical results and publication

- Hand-computed fixture EDT: **1613.7508401599655 ms**, sample SD **15.808442658968 ms**; producer agrees.
- Finite-count difference **2 accepted; 3 refused**.
- Canonical table JSON is byte-stable and has no generated timestamp.
- Markdown numerical values come from JSON; independent rendering required only JSON and its sidecar.
- Table serialization/sidecar assembly was replayed in memory. Actual filesystem publication transactions remain untested here.

Independent paired-statistic checks:

| Check | Result |
|---|---|
| Shared query indices, relative draws | Exactly equal |
| Shared query indices, absolute draws | Maximum error `2.0e-15` |
| Explicit whole-room sampling, unequal room sizes | Maximum absolute/relative errors `8.9e-16` / `1.7e-16` |
| Exact quantiles on `1…20000` | **500, 19500** |
| Synthetic `+0.004 s` shift | Absolute estimate **4 ms**, relative **0.022857142857…**, both inside query and room intervals |
| Tiny `n_boot=7` | Relative convergence flagged |
| Three registered pairings, both K | Ten cells each; deterministic |
| Decision fields | `decision_driving: false`; no inferential verdict |

The constant-baseline transformation is mathematically and numerically correct. Room sampling retains all queries from each sampled room and computes the query-weighted statistic.

### CPU suite and hygiene

Environment used the prescribed Python 3.8 interpreter, worktree `PYTHONPATH`, local AcousticRooms root, empty `CUDA_VISIBLE_DEVICES`, and `PYTHONDONTWRITEBYTECODE=1`.

| Command/check | Result |
|---|---|
| `python -m pytest tests -q -p no:cacheprovider` | Blocked before collection: no writable temporary directory |
| Adapted whole-suite replay | **1,562 collected; 509 passed, 1,051 skipped, 2 failed** |
| Skip breakdown | **1,016 filesystem-related; 35 CUDA/environment** |
| `py_compile.compile(..., doraise=True)` with in-memory bytecode sink | **13/13 passed** |
| `git diff --check ab585d0..7b3ad6c` | Exit 0 |
| Pinned/forbidden-path diff | Empty |
| Scope | Intended 14 files; **1,876 insertions / 13 deletions** |
| Co-author trailers | Correct on all 18 commits |

The adapted replay used system capture, disabled Numba disk caching, reused the existing Matplotlib cache, and applied corresponding cache accommodations to closure-import subprocesses.

The two failures are:

- `test_the_manifest_pins_agree_with_the_index_when_it_exists`
- `test_the_query_digest_agrees_with_the_seed_42_manifest_when_it_exists`

The Planner built the manifests at 13:33; their profile pins remain `None`. Thus **1517 passed / 45 skipped is not independently certified in the current environment**.

Filesystem-backed publication, temporary Git repositories, and child-launch/finalisation tests could not run normally. The pre-existing untracked `checkpoints` symlink remains unchanged.

## Coder decisions

The supplied records do not contain the original numbered decisions 1–8. I requested that report; I cannot reliably assign its numbering. Assessments of the documented choices are:

| Choice | Assessment |
|---|---|
| Frozen construction helper plus module-scope seen import | Accept |
| Frozen parity reference plus launcher coverage | Accept; GPU evidence pending |
| Literal profiles and runtime approval JSON | Accept architecture; findings 4 and 6 |
| Evaluation inventory placeholder | Accept; fill and add agreement coverage |
| `admit(producer_key=…)` | Accept |
| Shared sampler with constant-baseline absolute series | Accept |
| One pairing per invocation | Accept; binder must cover three outputs |
| Explicit `train_args/train_manifest/train_completion` bindings | Accept and require on every new-arm evaluation |

## Fitness for round 5

**Resolve blockers 1–2 before closing round 4.** The renderers can then consume the following interface.

### Exact JSON fields

| Purpose | Fields |
|---|---|
| Common provenance | `schema_version`, `profile_name`, `profile`, `profile_digest`, `exploratory`, `deviations`, `inputs`, `contracts`, `producer_closure_sha256` |
| Table row identity | `rows[].role`, `.label`, `.num_shot`, `.epoch`, `.backbone`, `.reference`, `.protocol` |
| Table numbers | `rows[].metrics.{EDT,C50,T60,loss,log_mse}.{source,unit,mean,sd,per_seed}` |
| Seed table values | `per_seed["42"…"46"].{mean,n_finite}` |
| Pair publication state | `pairing`, `decision_driving`, `verdict_scope`, `final`, `reconverge_required`, `run_flags` |
| Pair cell identity | `cells[].pairing`, `.metric`, `.source`, `.num_shot`, `.k`, `.reference`, `.decision_driving` |
| Pair statistics | `cells[].statistics.{absolute,relative}.{estimate,unit,n_boot,interval,room_cluster_interval,convergence,reconverge_required}` |
| Pair cohort | `paired_cohort.{n_queries,n_rooms,retained_sha256,excluded}`, `n_rooms_retained`, `exclusions`, `seed_labels` |
| Cohort means | `arm_means[role].{mean,sd,per_seed}` |

Renderer/binder requirements:

- EDT means, SDs, absolute differences and endpoints are **ms**; divide by 1000 for seconds.
- Relative statistics are **fractions**; multiply by 100 for percentages.
- Keep paired-cohort means distinct from table means.
- Reject exploratory/deviating publication inputs; additionally require pair `final == true` and no reconvergence flags.
- Require eight table rows and all **30 paired cells across three pairing outputs**. `final` alone does not establish complete K coverage.
- Bind each output and sidecar, all 40 runs, three training attempts, released checkpoint, and the validated exp04 inputs.
- Provide a reproducible **40,000-resample rerun path**. The current CLI only uses the fixed profile count; the flag correctly identifies failures but does not itself execute the planned rerun.
- Preserve nine deferred exp07 GPU parity tests, calibration, and final integrative review as outstanding gates.

### Pin-finalisation values

The ten newly available manifests independently match their index’s semantic hashes and file hashes. They share the same ordered **6,217 queries / 131 rooms**; every reference belongs to that query set.

Fill these profile values:

| Field | Value |
|---|---|
| `dataset.query_sha256` | `be5095a8501211be43cbab6b0f11658b261c54376743ed36fa617a6b1d0e1266` |
| `dataset.inventory_sha256` | `251b7d7b3ec7b4462ca9c99ef18f088f338596047d637513d9ec3a2a9814a9e1` |

The evaluation inventory was freshly hashed: **13,064 files, 1,044,560,337 bytes**. It is distinct from the training inventory.

Fill `seeds[K][seed]` with these **semantic manifest hashes**:

| K | Seed | Hash |
|---:|---:|---|
| 8 | 42 | `2aaa459f52369bb8eb4a2092b7c6ecb65b95164d54d87d32a7703fa082a884e5` |
| 8 | 43 | `92c7eee3fa86ae6ec64a6cb2d043ff38978bf01d800d70ba1d5e82d00046d0f4` |
| 8 | 44 | `1b1f9860ce553266fb8c2d866fa4b2628243113b584a8dd21b502c13387838c8` |
| 8 | 45 | `cc1e71c300fb75eb17becb6dd586e81401f7b75f3eda9f1a8a71193ac82ae3fd` |
| 8 | 46 | `e0f33d8565c98f23c1d19e00c541013a471b71de0ba4af45e6675d51551871a9` |
| 1 | 42 | `939163752323ca225216c2106ce69cc60f4a94f876bb66706b773491a8b3f059` |
| 1 | 43 | `d552a5d63f7b1ea9d5bc3106f4ddda9bb0549cc3ce29a33993b2fdf3d604a5b4` |
| 1 | 44 | `b199203a1c502b6e10c9ff7a1fdaa351822ee46c4e6729e224b37d846c014020` |
| 1 | 45 | `5fc6cdc59acac960d7dd520ae941642d32829efc14b9c60a165bf8c6c5734dc1` |
| 1 | 46 | `91ac1117b395ac585e2550554c314d04b54500d197166d770fe79784169bffea` |

After fixes and training, fill the approval JSON atomically:

- `schema_version: 1`
- `closures.evaluator`: reviewed `tools.exp07_eval` closure
- `closures.writer`: reviewed evaluation-writer closure
- `closures.training_launcher`: list of actual approved launcher closure digests
- `closures.training`: validated shared training closure digest
- `closures.producer_table`, `closures.producer_pairs`: final producer closure digests
- `checkpoints.{seen_simple,seen_cyl,seen_aug}`:
  - `path: ckpt/exp07/<role>/final/epoch_012.pth`
  - `epoch: 12`
  - `sha256`: actual completed checkpoint digest

Fill available profile identities before computing final producer closures; commit the completed runtime approval file before production analysis.

**Exact blocking list: finding 1, unchecked training closure records; finding 2, incomplete training data/execution provenance validation.**
191,542
**Reviewer:** OpenAI Codex gpt-6-astra (codex exec, read-only sandbox, worktree exp07-window) · **Date:** 2026-09-15

## Verdict: request changes

**Blocking findings: 1 and 2 below.** Round-3 items are closed. The paired statistics and table aggregation check out, but training-provenance admission remains incomplete.

Reviewed every changed line in `ab585d0..7b3ad6c`. No repository or live-data files were modified; no GPU was used.

## Findings

### 1. Blocker — training closure pins certify unchecked digest labels

[tools/exp07_table.py:104](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_table.py:104)

The contract compares `source_closures.training.sha256` and `.launcher.sha256` with approval pins without recomputing those digests from their `files` records. The inherited collector validates the **evaluation** closures; it does not recursively validate closures inside the bound training manifest.

**Reproduced:** independently replacing either training or launcher `files` with `[]`, or replacing training records with an unreviewed `wrong.py` record, still produces eight table rows with `deviations == []`. Outer manifest/completion bindings were refreshed; approval pins remained unchanged.

Thus the equal-training-closures check proves equality of supplied labels, not the recorded source identities.

**Minimal fix:** validate each training closure’s complete records and recompute its digest before comparing with the pin. Preserve A7’s distinction between immutable reviewed source identity and permitted, recorded post-spawn working-tree drift. Add these mutations to the refusal tests.

### 2. Blocker — the training manifest’s data and execution contract are not validated

[tools/exp07_table.py:97](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_table.py:97)

Beyond the split binding, the manifest checks mostly establish that several `protocol` strings equal `"seen"`. The recipe check applies to bound `args.json`, without establishing consistency with the manifest’s actual execution specification.

**Each of these independent mutations was admitted with no deviations:**

- `train_data_identity.inventory_files = 1`
- A substituted `train_data_identity.inventory_sha256`
- `mode = "smoke"`
- `allow_dirty = true`
- `effective_args.epochs = 1`
- Wrong `effective_args.backbone` or `effective_args.yaw_aug`
- `effective_args.save_dir` pointing elsewhere
- `timing_limits.projection_hours = 1000`

The existing fixture itself supplies a training inventory digest without an inventory or bound inventory sidecar, so its successful admission does not exercise the planned data-identity contract.

**Minimal fix:** validate the real training inventory/sidecar and digest, the 296,454-file seen identity, full-run status, relevant timing constraints, and normalized consistency between `effective_args`, bound runtime args, and the attempt directory. Bind the additional validated evidence into producer provenance. Extend the fixture to represent those real artifacts.

### 3. Should-fix — required spectral columns can disappear silently

[tools/exp07_table.py:212](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_table.py:212)

Removing `loss` and `log_mse` from both outputs for all five seeds of one group still produces an admitted table without deviations. The collector requires the acoustic metrics, while aggregation silently skips absent recognized metrics.

Plan §2 requires both spectral metrics in canonical JSON.

**Minimal fix:** require all five registered output metrics for every group, with named missing-metric refusals. Keep the existing allowance for recognized optional diagnostics.

### 4. Should-fix — pin-finalisation tests do not support the intended lifecycle completely

[tests/test_exp07_profiles.py:142](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_exp07_profiles.py:142), [tests/test_exp07_profiles.py:199](/home/yixunhu/codespace/xRIR_code_wt07/tests/test_exp07_profiles.py:199)

The committed-file test permanently asserts that approval equals the all-null template. It will fail after legitimate pin finalisation. Also, the inventory placeholder receives a format check but no artifact-agreement test; the new-arm profile digest placeholders likewise lack the advertised agreement check.

**Minimal fix:** test all-null semantics against an isolated template, allow either valid committed lifecycle state, and test available inventory/checkpoint identities against their authoritative artifacts. Clarify that runtime approval supplies new-arm checkpoint digests—the admission code currently replaces the profile placeholders with those values.

The two current manifest/query agreement failures are **expected pending-pin failures**, described below, rather than defects in those agreement tests.

### 5. Should-fix — exploratory Markdown omits the deviations

[tools/exp07_table.py:270](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_table.py:270)

The shared Markdown renderer ignores `exploratory` and `deviations`. A synthetic table with 120 admission deviations renders the ordinary “Model comparison” table without displaying those deviations or a prominent exploratory status. A command flag or filename is insufficient disclosure.

**Minimal fix:** provide an exp07 rendering adapter that reads these fields from JSON and includes them before the output hashes and sidecar are finalized.

### 6. Nit — loaded approval is not recursively immutable

[tools/exp07_profiles.py:30](/home/yixunhu/codespace/xRIR_code_wt07/tools/exp07_profiles.py:30)

The registered profiles are immutable, but `freeze()` leaves lists mutable. Consequently:

```python
pins["closures"]["training_launcher"].append("f" * 64)
```

succeeds, including on the returned committed template.

**Minimal fix:** freeze sequences recursively and add a nested-list mutation test.

### 7. Nit — two commits exceed the round’s size limit

- `8772121`: **210 changed lines**; fixture plus its layout test.
- `eff1b2a`: **224 changed lines**; paired-cell implementation/tests plus producer-key admission plumbing.

Both are coherent, reviewable units. Accept the recorded process deviations; split future commits before crossing the limit.

## Verified

### Round-3 closure

- Base evaluator closure: **15 files**, seen dataset module absent.
- Head evaluator closure: **16 files**, with precisely that module added.
- Exp04 evaluator/writer closures retain **14/15 files**, excluding the seen module.
- Simulated changed seen-module bytes:
  - Old membership: `revalidate(...) == []`
  - Head: `source.entrypoint.treble_multi_room_dataset/treble_xRIR_seen_dataset.py`
- Base manifest builder accepts replacement between K values and publishes an index.
- Head records call order `split → dataset8 → dataset1 → split` and refuses replacement without publishing the index.
- Canonical absolute and indirect split paths are accepted; another location or wrong digest is refused.
- GPU parity now uses frozen `eval_yaw_rotation.evaluate_batch`. Direct tests retain the actual final nine queries; launcher tests exercise a 16+9 bounded batch at both K values. **Execution remains deferred.**

### Profiles and admission

Both literal digests equal their files:

| Artifact | SHA-256 |
|---|---|
| Released checkpoint | `1762c702ee23a8f584e67c4b5b052b7483237e6471bb5e58774b66d514aec2f8` |
| Seen split pickle | `bc97850b090198cf956dc7975a2da611dd058f8b0f83adee7f93f7b7fad7720f` |

Verified immutable registered profiles, stable profile digests, and recipe agreement with all three committed full-run goldens.

The approval loader accepts valid null/filled states and rejects partial filling, malformed digests, wrong keys/path/epoch/types, invalid launcher lists, and bytes differing from HEAD.

Using the actual fixture construction with filesystem operations redirected into memory:

- Valid fixture: **40 runs, eight groups, five seeds each, zero deviations**.
- All **19 supplied refusal mutations** produce their named deviations.
- **31 additional recipe/full-run/arm/tier field mutations** produce named refusals.
- Wrong inner completion-to-manifest and completion-to-args hashes refuse despite repaired outer bindings.
- Eight evaluation mutations also refuse on the released row.
- K, grid, P-only, batch, TF32, canonical padding, GL seed, bounded evaluation, and confirmatory flags refuse as expected.
- Finding 1–2 mutations establish the remaining training-provenance gaps.

### Numerical results and publication

- Hand-computed fixture EDT: **1613.7508401599655 ms**, sample SD **15.808442658968 ms**; producer agrees.
- Finite-count difference **2 accepted; 3 refused**.
- Canonical table JSON is byte-stable and has no generated timestamp.
- Markdown numerical values come from JSON; independent rendering required only JSON and its sidecar.
- Table serialization/sidecar assembly was replayed in memory. Actual filesystem publication transactions remain untested here.

Independent paired-statistic checks:

| Check | Result |
|---|---|
| Shared query indices, relative draws | Exactly equal |
| Shared query indices, absolute draws | Maximum error `2.0e-15` |
| Explicit whole-room sampling, unequal room sizes | Maximum absolute/relative errors `8.9e-16` / `1.7e-16` |
| Exact quantiles on `1…20000` | **500, 19500** |
| Synthetic `+0.004 s` shift | Absolute estimate **4 ms**, relative **0.022857142857…**, both inside query and room intervals |
| Tiny `n_boot=7` | Relative convergence flagged |
| Three registered pairings, both K | Ten cells each; deterministic |
| Decision fields | `decision_driving: false`; no inferential verdict |

The constant-baseline transformation is mathematically and numerically correct. Room sampling retains all queries from each sampled room and computes the query-weighted statistic.

### CPU suite and hygiene

Environment used the prescribed Python 3.8 interpreter, worktree `PYTHONPATH`, local AcousticRooms root, empty `CUDA_VISIBLE_DEVICES`, and `PYTHONDONTWRITEBYTECODE=1`.

| Command/check | Result |
|---|---|
| `python -m pytest tests -q -p no:cacheprovider` | Blocked before collection: no writable temporary directory |
| Adapted whole-suite replay | **1,562 collected; 509 passed, 1,051 skipped, 2 failed** |
| Skip breakdown | **1,016 filesystem-related; 35 CUDA/environment** |
| `py_compile.compile(..., doraise=True)` with in-memory bytecode sink | **13/13 passed** |
| `git diff --check ab585d0..7b3ad6c` | Exit 0 |
| Pinned/forbidden-path diff | Empty |
| Scope | Intended 14 files; **1,876 insertions / 13 deletions** |
| Co-author trailers | Correct on all 18 commits |

The adapted replay used system capture, disabled Numba disk caching, reused the existing Matplotlib cache, and applied corresponding cache accommodations to closure-import subprocesses.

The two failures are:

- `test_the_manifest_pins_agree_with_the_index_when_it_exists`
- `test_the_query_digest_agrees_with_the_seed_42_manifest_when_it_exists`

The Planner built the manifests at 13:33; their profile pins remain `None`. Thus **1517 passed / 45 skipped is not independently certified in the current environment**.

Filesystem-backed publication, temporary Git repositories, and child-launch/finalisation tests could not run normally. The pre-existing untracked `checkpoints` symlink remains unchanged.

## Coder decisions

The supplied records do not contain the original numbered decisions 1–8. I requested that report; I cannot reliably assign its numbering. Assessments of the documented choices are:

| Choice | Assessment |
|---|---|
| Frozen construction helper plus module-scope seen import | Accept |
| Frozen parity reference plus launcher coverage | Accept; GPU evidence pending |
| Literal profiles and runtime approval JSON | Accept architecture; findings 4 and 6 |
| Evaluation inventory placeholder | Accept; fill and add agreement coverage |
| `admit(producer_key=…)` | Accept |
| Shared sampler with constant-baseline absolute series | Accept |
| One pairing per invocation | Accept; binder must cover three outputs |
| Explicit `train_args/train_manifest/train_completion` bindings | Accept and require on every new-arm evaluation |

## Fitness for round 5

**Resolve blockers 1–2 before closing round 4.** The renderers can then consume the following interface.

### Exact JSON fields

| Purpose | Fields |
|---|---|
| Common provenance | `schema_version`, `profile_name`, `profile`, `profile_digest`, `exploratory`, `deviations`, `inputs`, `contracts`, `producer_closure_sha256` |
| Table row identity | `rows[].role`, `.label`, `.num_shot`, `.epoch`, `.backbone`, `.reference`, `.protocol` |
| Table numbers | `rows[].metrics.{EDT,C50,T60,loss,log_mse}.{source,unit,mean,sd,per_seed}` |
| Seed table values | `per_seed["42"…"46"].{mean,n_finite}` |
| Pair publication state | `pairing`, `decision_driving`, `verdict_scope`, `final`, `reconverge_required`, `run_flags` |
| Pair cell identity | `cells[].pairing`, `.metric`, `.source`, `.num_shot`, `.k`, `.reference`, `.decision_driving` |
| Pair statistics | `cells[].statistics.{absolute,relative}.{estimate,unit,n_boot,interval,room_cluster_interval,convergence,reconverge_required}` |
| Pair cohort | `paired_cohort.{n_queries,n_rooms,retained_sha256,excluded}`, `n_rooms_retained`, `exclusions`, `seed_labels` |
| Cohort means | `arm_means[role].{mean,sd,per_seed}` |

Renderer/binder requirements:

- EDT means, SDs, absolute differences and endpoints are **ms**; divide by 1000 for seconds.
- Relative statistics are **fractions**; multiply by 100 for percentages.
- Keep paired-cohort means distinct from table means.
- Reject exploratory/deviating publication inputs; additionally require pair `final == true` and no reconvergence flags.
- Require eight table rows and all **30 paired cells across three pairing outputs**. `final` alone does not establish complete K coverage.
- Bind each output and sidecar, all 40 runs, three training attempts, released checkpoint, and the validated exp04 inputs.
- Provide a reproducible **40,000-resample rerun path**. The current CLI only uses the fixed profile count; the flag correctly identifies failures but does not itself execute the planned rerun.
- Preserve nine deferred exp07 GPU parity tests, calibration, and final integrative review as outstanding gates.

### Pin-finalisation values

The ten newly available manifests independently match their index’s semantic hashes and file hashes. They share the same ordered **6,217 queries / 131 rooms**; every reference belongs to that query set.

Fill these profile values:

| Field | Value |
|---|---|
| `dataset.query_sha256` | `be5095a8501211be43cbab6b0f11658b261c54376743ed36fa617a6b1d0e1266` |
| `dataset.inventory_sha256` | `251b7d7b3ec7b4462ca9c99ef18f088f338596047d637513d9ec3a2a9814a9e1` |

The evaluation inventory was freshly hashed: **13,064 files, 1,044,560,337 bytes**. It is distinct from the training inventory.

Fill `seeds[K][seed]` with these **semantic manifest hashes**:

| K | Seed | Hash |
|---:|---:|---|
| 8 | 42 | `2aaa459f52369bb8eb4a2092b7c6ecb65b95164d54d87d32a7703fa082a884e5` |
| 8 | 43 | `92c7eee3fa86ae6ec64a6cb2d043ff38978bf01d800d70ba1d5e82d00046d0f4` |
| 8 | 44 | `1b1f9860ce553266fb8c2d866fa4b2628243113b584a8dd21b502c13387838c8` |
| 8 | 45 | `cc1e71c300fb75eb17becb6dd586e81401f7b75f3eda9f1a8a71193ac82ae3fd` |
| 8 | 46 | `e0f33d8565c98f23c1d19e00c541013a471b71de0ba4af45e6675d51551871a9` |
| 1 | 42 | `939163752323ca225216c2106ce69cc60f4a94f876bb66706b773491a8b3f059` |
| 1 | 43 | `d552a5d63f7b1ea9d5bc3106f4ddda9bb0549cc3ce29a33993b2fdf3d604a5b4` |
| 1 | 44 | `b199203a1c502b6e10c9ff7a1fdaa351822ee46c4e6729e224b37d846c014020` |
| 1 | 45 | `5fc6cdc59acac960d7dd520ae941642d32829efc14b9c60a165bf8c6c5734dc1` |
| 1 | 46 | `91ac1117b395ac585e2550554c314d04b54500d197166d770fe79784169bffea` |

After fixes and training, fill the approval JSON atomically:

- `schema_version: 1`
- `closures.evaluator`: reviewed `tools.exp07_eval` closure
- `closures.writer`: reviewed evaluation-writer closure
- `closures.training_launcher`: list of actual approved launcher closure digests
- `closures.training`: validated shared training closure digest
- `closures.producer_table`, `closures.producer_pairs`: final producer closure digests
- `checkpoints.{seen_simple,seen_cyl,seen_aug}`:
  - `path: ckpt/exp07/<role>/final/epoch_012.pth`
  - `epoch: 12`
  - `sha256`: actual completed checkpoint digest

Fill available profile identities before computing final producer closures; commit the completed runtime approval file before production analysis.

**Exact blocking list: finding 1, unchecked training closure records; finding 2, incomplete training data/execution provenance validation.**
