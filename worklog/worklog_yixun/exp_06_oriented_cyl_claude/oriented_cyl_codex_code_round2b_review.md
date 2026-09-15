**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

# Round 2b review — request changes

Reviewed `12e06fc..d6a4f42` on `exp06-round2b`, including every changed line, the SOP, approved v4 plan, three plan reviews, Fable round-1 review, notebook and role change, prior experiment records, composed tooling, and this round’s prompt/report.

The numerical composition is sound by inspection and the CPU checks described below. The remaining mandatory findings concern launch ordering, output validation, and heading/job-spec bindings.

No files were modified, GPU work started, or processes signaled. I did not access the concurrent worktree.

## Findings

### 1. **Blocker — Job-spec failure does not prevent children from launching**

**Locations:** [exp06_haa_pipeline.sh:181](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_haa_pipeline.sh:181), also lines 59, 162 and 203–212.

`run_finetune` and `run_zeroshot` are called through `||`. Bash therefore suppresses `errexit` within those functions. Neither checks the return status of `open_job` or `job_spec`.

**Reproduction:** Executing the actual `run_finetune` function with filesystem/process operations replaced by in-memory stubs, and making `job_spec` return 2, reached **all nine children**. A subsequent finalizer failure still produced `QUEUE_DONE` and process status **0**.

Separately, the job-spec writer accepts a refused heading and writes its `k: null`. A refused **dampened-room** heading need not stop stage 1, which uses the other three rooms; substantial training can occur before that refusal is encountered.

**Why it matters:** The mandatory declaration is not an effective pre-launch gate. The queue can spend compute on a job already known to be inadmissible and report success after failures.

**Minimal fix:**

- Explicitly propagate failure from `open_job` and `job_spec`.
- Validate all four headings as decided, confirmatory, and bound to the selected cache before launching any child.
- Validate the resulting job specification before use.
- Accumulate queue failures and return nonzero.
- Add fault tests requiring **zero child launches** after failed preparation, including a refused dampened-room heading.

### 2. **Should-fix — Simulated completion does not validate exp_06 output metadata**

**Location:** [exp06_eval_launch.py:139](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_eval_launch.py:139).

The adapter returns directly from `execute_run`. Its preceding `check_fields` validates the intended manifest, not the finished output files. The inherited completion check does not compare `model_class`, `registry_sha256`, `checkpoint_role`, `checkpoint_epoch`, `heading`, or `frame`.

**Reproduction:** The actual inherited `execute_run` function, with an in-memory filesystem and child, certified **24/24** cases: corrupting or removing each of those six fields in either output.

This is corruption **before completion hashes are captured**. Ordinary changes after certification remain detectable through those hashes.

**Why it matters:** Completion can certify outputs contradicting the experiment identity in their manifest.

**Minimal fix:** Add an exp_06-owned post-run validator comparing both outputs against the manifest, with presence and type checks, and verifying their completion hashes. On failure, invalidate/quarantine the completion and fail the launch. Add corresponding corruption/omission tests.

The report’s claim that this requires editing the pinned launcher is incorrect; an adapter-owned check can close the gap without changing it.

### 3. **Should-fix — Frozen-field tests conflate admission refusal with completion idempotence**

**Location:** [test_exp06_haa_pipeline.py:316](/home/yixunhu/codespace/xRIR_code_wt2b/tests/test_exp06_haa_pipeline.py:316); related writer at [exp06_haa_pipeline.sh:57](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_haa_pipeline.sh:57).

The module-scoped fixture is shared with the preceding positive test, which writes the job’s `completion.json`. Mutation tests accept any `ValueError`, including “completion.json already exists and differs.”

**Reproduction:** The actual `load_job_spec`, `check_job_spec`, and `job_lineage` functions accept both `init: cyl_or` and `init: somebody_elses_pretrain` against identical child evidence. The existing mutation test can nevertheless pass because publishing the changed completion conflicts with the earlier completion.

**Why it matters:** At least the `init` test does not establish refusal before first admission. The job-spec writer records the initialization label, but children do not freeze that declaration.

**Minimal fix:** Run each mutation against a job with **no prior job completion**, assert the intended refusal reason, and require that no completion appears. Bind the pre-launch job-spec identity into child provenance and verify it during job admission, or provide an equivalent explicit initialization-identity binding.

### 4. **Should-fix — Validation-only headings escape finalizer revalidation**

**Locations:** [exp06_haa_finetune.py:193](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_haa_finetune.py:193), [exp06_finalize.py:771](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_finalize.py:771).

The wrapper correctly loads and records headings for the union of training and validation rooms. The finalizer calls `_heading_binding` with only `args["rooms"]`.

**Reproduction:** With training room `hallway` and validation room `class_room`, replacing the validation-only heading JSON with invalid text was accepted by the heading-consumer path; only `hallway` was checked.

**Why it matters:** Validation determines `best.pth`. Every heading used for checkpoint selection belongs to the input-revalidation contract. The default pipeline avoids this case, but the supported `--val-rooms` interface does not.

**Minimal fix:** Revalidate the union of training and effective validation rooms, while preserving training-room membership separately. Add a validation-only heading mutation regression.

### 5. **Nit — Shell parsing truncates checkpoint overrides and accepts malformed jobs**

**Locations:** [exp06_haa_pipeline.sh:176](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_haa_pipeline.sh:176), also lines 143 and 157.

`set -- $(init_of "$name")` splits checkpoint paths on whitespace.

**Reproductions:**

- Override `/tmp/path with spaces/epoch_012.pth` becomes checkpoint `/tmp/path`.
- Job `cyl_or:anything:0` is accepted and schedules `cyl_or:0`.

**Minimal fix:** Assign backbone/checkpoint through quoted variables or an array, and validate exactly `<allowed-init>:<integer-seed>`.

### 6. **Nit — Commit-size accounting omits an oversized test commit**

**Locations:** [exp06_haa_finetune.py:1](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_haa_finetune.py:1), [exp06_haa_pipeline.sh:1](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_haa_pipeline.sh:1), [test_exp06_finalize_haa_consumer.py:1](/home/yixunhu/codespace/xRIR_code_wt2b/tests/test_exp06_finalize_haa_consumer.py:1).

Three commits exceed 200 additions plus deletions:

| Commit | Changed lines |
|---|---:|
| `aa61fc4` | 294 |
| `140c3c9` | 212 |
| `4b30735` | 227 |

The report identifies two oversized implementation commits but omits the third from its deviation count. Tests also count as code.

**Minimal fix:** Correct the record and keep subsequent commits smaller. No history rewrite is warranted. All 18 commits have the required trailers, and the red/implementation ordering is visible.

## Disposition of all 21 reported discrepancies

| Report item | Assessment |
|---|---|
| 1 — Required checkpoint epoch | Accept. Avoids inventing an epoch. |
| 2 — Named model factory | Accept; equivalent routing. |
| 3 — Repeated registry digest helper | Accept; identical definition and useful closure isolation. |
| 4 — Missing output metadata check | Finding **2**; locally fixable. |
| 5 — Evaluation closure includes trainer | Accept; conservative over-binding. |
| 6 — Private `_inventory` helper | Accept against the explicitly pinned implementation. |
| 7 — Added `--root` | Accept; necessary resolved cache identity. |
| 8 — Added evaluation `--seed` | Accept; distinct from reference/phase seed. |
| 9 — Heading filename convention | Accept; consistent across producer and consumers. |
| 10 — Producer requires confirmatory headings | Accept; necessary fail-closed behavior. |
| 11 — Oversized commits | Finding **6**, nonblocking. |
| 12 — Existing finalizer fixture edits | Accept; necessary adaptation, confined to fixture regions. |
| 13 — Local job lifecycle helpers | Reasonable composition; implementation has finding **1**. |
| 14 — Job receipt uses orchestrator PID | Compatible with this reviewed tip; combined replay after the training-gate merge remains necessary. |
| 15 — Job-spec writer accepts unusable headings | Finding **1**. |
| 16 — Delegated `train_completion` binding | Accept; already supported and rehashed by exp_04. |
| 17 — `prepare(repo=…)` | Accept. |
| 18 — Throwaway clone for integration test | Reasonable approach; mutation evidence needs finding **3**. |
| 19 — Added metrics metadata/aggregate file | Accept; consumers must use the bound artifacts. |
| 20 — Training/validation heading union | Producer is correct; consumer gap is finding **4**. |
| 21 — GPU loops not executed | Appropriate for this round; GPU parity remains a later ladder check. |

## Verification performed

All Python checks used the specified interpreter and environment:

```text
PYTHONPATH=/home/yixunhu/codespace/xRIR_code_wt2b
XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
CUDA_VISIBLE_DEVICES=''
OMP_NUM_THREADS=4
PYTHONDONTWRITEBYTECODE=1
NUMBA_CACHE_DIR=/tmp/exp06_codex_review_numba
/home/yixunhu/miniconda3/envs/xRIR/bin/python
```

### Source and static checks

- Read the complete round diff, including all six changed test files.
- `git diff --check 12e06fc..d6a4f42`: clean.
- `bash -n tools/exp06_haa_pipeline.sh`: passed.
- In-memory `compile()` on all **11 changed Python files**: passed without writing bytecode.
- `git diff main...d6a4f42 --diff-filter=M --name-only`: empty.
- **39 protected source files** compared directly with `main`: zero mismatches; separately verified every member of the trainer’s static local import closure.
- Exp_03’s **12 enumerated files**, reviewed bytes and history: unchanged; digest `5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`.
- Trainer’s **11-file enumerated closure**: unchanged; digest `5b2da2504e220b63bfc932cf0d83c143261e117edea1838e6907159c3b12eb13`.
- Worktree remained clean at `d6a4f42e4db43ce7fbd1ae972116a0e36dd43650`.

### CPU checks completed

| Check | Result |
|---|---|
| Pipeline’s unchanged read-only test functions, invoked directly | **11 passed**, including exact `cyl_or:0` and `zeroshot` dry-run goldens |
| Encoder test bodies with extracted heading/camera helpers to avoid cache-dependent imports | **31 passed**: symmetry, heading cancellation, parameter delta, dtype and zero-shift exactness |
| Fine-tuning epoch calculations, updates and history | Exact AST equality with pinned loop, excluding progress-print formatting |
| HAA evaluation metric loop | Compared line by line; no unintended numerical calculation difference found |
| Actual CPU Griffin–Lim with extracted wrapper | Same-query bit-exact; different-query phases differ; unseeded output **and RNG advancement** match pinned call exactly |
| Side-label helper | Room-frame `[-1, 1]`; zero-y refused |
| Training-only real heading readback | All four rooms: **−90°, k=128, 12 stable refits** |
| Cycle-6 consumer probes with JSON/cache changes confined to memory | Changed `meta.json` refused for each room; diagnostic heading refused |
| Cycle-0 numeric probes | NaN accepted; positive/negative infinity, finite T60 and overflowing integer refused |
| Adversarial launch/admission checks | Findings **1–4** reproduced as described |

The heading readback used lazy RIR loading and selected training rows only. Import-closure discovery was replaced by the enumerated closure for this adapted readback.

### Replay limitations

The requested seven-file replay was attempted:

```text
-m pytest tests/test_exp06_eval.py tests/test_exp06_eval_launch.py
tests/test_exp06_haa.py tests/test_exp06_haa_pipeline.py
tests/test_exp06_finalize_haa_consumer.py tests/test_exp06_finalize.py
tests/test_provenance.py -q -p no:cacheprovider
```

It failed before collection because pytest capture could not create a temporary file anywhere permitted by the sandbox. Retrying `test_exp06_eval.py` with `-s`, and `test_exp06_encoder.py` with `--capture=sys`, reached collection but failed on Numba’s unwritable import cache.

Therefore:

- **No unmodified pytest test completed.**
- The **42 direct test-body passes** and additional probes above are adapted checks, not a full-suite replay.
- The filesystem-backed nine-child integration test, atomic publication, and real process/log lifecycle were **not independently replayed**.
- The Coder’s **631 passed / 3 skipped** remains a reported result, not my reproduced total.

## Exact blocking list for closing round 2b

Fix and re-verify **findings 1–4**:

1. Enforce job preparation and all-room heading validation before children; propagate queue failure.
2. Validate exp_06 metadata in both completed simulated outputs.
3. Test first admission independently of existing completion, and freeze initialization/job-spec identity.
4. Revalidate validation-only heading inputs.

Findings **5–6** are nonblocking nits.

**Final verdict: request changes.**
