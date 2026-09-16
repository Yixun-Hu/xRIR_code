**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

# exp_06 oriented_cyl — round 2a CLOSE review

**Verdict: request changes. Round 2a may not close.**

Reviewed all changed lines in `4f92f54..2fe25bf`, ending at `2fe25bfc1b5e314a8209c8bebb7802b76ac4c035`. The fixes address many original counterexamples, but completion still admits incomplete or contradictory evidence. HAA history validation also rejects the actual exp_02 history format.

No files were modified, GPU work started, or processes signalled. Mutation tests used in-memory file overlays.

## Disposition of original findings 1–9

| Finding | Disposition |
|---|---|
| **1 — Input revalidation** | **Partial.** Changed source/data bytes, missing identity, and empty source closure are refused. An empty data inventory is accepted. |
| **2 — Three argument sources** | **Partial.** Recursive type checks, required fields, and native-integer counts work. Identical false provenance bindings remain accepted. |
| **3 — HAA evidence** | **Unresolved.** Original examples are refused, but history handling is incompatible with the pipeline, and evaluation/job admission still has material gaps. |
| **4 — Closed log** | **Partial.** Normal descendant draining, live-PID refusal, receipt hashes, and stale-log checks work. Sink failures are discarded. |
| **5 — Shared lifecycle/recovery** | **Partial.** Diagnostic receipts and recovery gating/promotion were added. Ownership, smoke discovery, and failure handling remain incomplete. |
| **6 — Data-root resolution** | **Resolved.** Training and provenance use the dataset module’s resolution; launcher export/default/refusal and goldens are present. |
| **7 — Smoke bounds/status** | **Resolved for the requested CPU-testable behavior.** Invalid budgets and returned failures are handled; the periodic memory callback is implemented. Actual GPU enforcement was not tested. |
| **8 — RNG isolation** | **Resolved.** Cold parameter counting preserves Torch’s global RNG state. |
| **9 — Malformed-input refusals** | **Partial.** Common decoding failures are normalized, but malformed nested structures still escape as uncaught exceptions. |

## Outstanding findings

### 1. Blocker — an empty data inventory still supports admissible completion

**Where:** [tools/exp06_finalize.py:240](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:240), [tools/exp06_finalize.py:284](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:284)

The identity check requires an inventory *list*, but permits an empty one. Revalidation then checks only the declared entries.

**Verified:** Replacing the full-run identity with `inventory=[]`, zero file/byte counts, and the correct empty-list digest produced `admissible_arm: true`.

This establishes no training-data identity, despite the full completion contract.

**Minimal fix:** Validate the identity schema and required training inventory membership, including agreement with the recorded resolved root. Refuse empty or incomplete inventories. Add adversarial tests for both; keep the pinned provenance helper unchanged.

### 2. Blocker — agreement between argument copies does not establish correct provenance bindings

**Where:** [tools/exp06_finalize.py:301](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:301)

All three argument sources now pass schema checks and strict equality. Their exp_06 bindings are still not checked against the execution record.

**Verified:** I changed all three copies identically to contain:

- `exp06_run_type='probe'` for a full run;
- false Git, registry, and source-closure hashes;
- `exp06_provenance_path='/wrong/provenance.json'`.

Full finalization still returned `admissible_arm: true`.

**Minimal fix:** Bind those fields to the validated run type, recorded execution HEAD, registry/closure identities, and actual provenance location. Retain the existing strict comparisons and add a regression where all three copies agree on false bindings.

### 3. Blocker — HAA validation rejects real histories and still admits invalid evidence

#### 3a. The frozen history rule contradicts the immutable pipeline

**Where:** [tools/exp06_finalize.py:472](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:472)

`haa_history` expects only epochs `range(val_every, epochs+1, val_every)` and requires validation loss on every row.

Actual [sim_to_real/finetune_haa.py:104](/home/yixunhu/codespace/xRIR_code_wt/sim_to_real/finetune_haa.py:104) writes:

- an initial validation row at epoch **0**;
- a training row for **every** subsequent epoch;
- validation loss at cadence boundaries **and the final epoch**;
- a potentially valid `best_epoch=0`.

**Verified:** Passing the retained exp_02 control/seed0/stage1 history and arguments to `haa_history` fails with:

```text
history.jsonl line 2 val_loss is None, not a finite number
```

Conversely, synthetic cadence-only histories with nonfinite training losses are accepted. A summary claiming `best_val_loss=-999` also passes despite disagreeing with its history.

**Minimal fix:** Validate rows `0..epochs`, the actual validation schedule, finite training losses, and summary consistency. Permit epoch 0 as the best checkpoint. Add a faithful pipeline-history fixture, including an off-cadence final epoch and an initialization that remains best.

#### 3b. Evaluation metrics are hashed without being parsed

**Where:** [tools/exp06_finalize.py:526](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:526)

**Verified:** Replacing `metrics_hallway.json` with `not JSON` still produces admissible HAA evaluation completion.

**Minimal fix:** Parse and validate required metrics fields and their applicable consistency with per-sample evidence before admission.

#### 3c. Job admission still trusts insufficiently validated child claims

**Where:** [tools/exp06_finalize.py:574](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:574), [tools/exp06_finalize.py:621](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:621)

Required child artifacts are only `args.json`, `best.pth`, and `last.pth` for training, and **only `args.json` for evaluation**. Most completion fields are checked for presence rather than validity. Identity agreement across children does not establish agreement with the job specification.

**Verified admissible counterexamples:**

- An evaluation child with its metrics/per-sample files and corresponding artifact entries removed.
- Children retaining the expected schema keys but carrying stale log/child-exit evidence hashes.
- Children claiming the wrong room for their directory.
- Children agreeing on an invented backbone and frame, with invalid schema version/run-directory claims and empty log/receipt records.

A changed, explicitly recorded checkpoint artifact is correctly refused; that does not cover omitted artifacts or separate evidence records.

**Minimal fix:** Validate complete per-role child schemas and required artifacts; rehash logs, receipts, and other bound evidence; enforce room/path and job identity, including the declared seed and initialization. Reuse child validation where practical. Keep deriving admission from verified evidence rather than child admission flags.

### 4. Blocker — a failed logging sink is treated as successfully drained

**Where:** [tools/exp06_launch.sh:98](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_launch.sh:98)

```bash
wait "$sink" || true
```

A sink can terminate because of a write error rather than EOF. Its failure is discarded, leaving a successful direct-child status eligible for marker, receipt, and completion publication. The claimed closed-log guarantee therefore has an unchecked failure path.

The normal descendant case is repaired: waiting for a successful sink captures output written after the direct child exits.

**Minimal fix:** Capture and enforce sink status. A logging failure must prevent successful completion and enter the abort path. Add a CPU fault test where the sink fails after consuming child output.

The receipt schema also remains weak: nonsensical `child_pid` and `ended_at` values were accepted. Validate these fields and bind the timestamp to the validated marker.

### 5. Should-fix — lifecycle ownership and recovery failure handling remain incomplete

**Where:** [tools/exp06_launch.sh:28](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_launch.sh:28), [tools/exp06_launch.sh:96](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_launch.sh:96), [tools/exp06_launch.sh:224](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_launch.sh:224)

Three gaps remain:

- Preflight scans `ATTEMPT_ROOT`, while smoke PID files live under `ckpt/exp06/_smoke`.
- `launch.pid` names the direct child. It stops identifying a live owner while the launcher drains descendants or finalizes.
- Recovery failure passes the literal string `'<reason>'` to `abort`, producing a placeholder suffix rather than a recorded cause.

Diagnostic directories also use nonexclusive `mkdir -p`, unlike full attempts.

**Minimal fix:** Retain launcher ownership through finalization, separately track the child, scan all applicable launch locations, create attempts exclusively, and use an actual recovery failure reason. Preserve the successful diagnostic-receipt and recovery-promotion changes.

### 9. Should-fix — malformed structures still escape the named-refusal interface

**Where:** [tools/exp06_finalize.py:124](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:124), [tools/exp06_finalize.py:346](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:346)

**Verified:**

- A model mapping containing invalid integer and string keys raises `TypeError` while sorting them.
- `git_state=[]` raises `AttributeError`.
- `mutable_inputs=None` raises `AttributeError` during revalidation.

These remain fail-closed, but bypass the promised named `ValueError`/CLI exit 2.

**Minimal fix:** Validate nested container and key types before traversing or sorting them. Normalize expected schema failures at the input boundary and add these cases to the refusal tests.

### 10. Nit — commit-size and red-first requirements were not fully met

**Where:** fix commit history and Coder report §§3–4.

All 15 commits have both required trailers. The reported insertion counts **230/330/207** are correct, but the prompt limits **changed lines**. Five commits exceed 200:

| Commit | Insertions + deletions |
|---|---:|
| `e9dd0a5` | 203 |
| `6d1e870` | 273 |
| `0ce5d13` | 217 |
| `3721cc8` | 406 |
| `e9e6e6f` | 246 |

The report explicitly identifies `4c66184`’s red run as retrospective. It therefore does **not** satisfy genuine red-before-implementation. The remaining red counts are reported, but commit contents alone cannot independently establish their chronology; `62655ae` is an import-only refactor without a separate reported red test.

**Minimal fix:** Preserve these disclosures. Use smaller subsequent commits and record genuine failing tests before implementation. No history rewrite is requested.

## Judgment on the frozen choices

| Choice | Judgment |
|---|---|
| Cadence-only HAA history rows | **Reject.** Validation cadence differs from history-writing cadence; finding 3a. |
| Required heading binding `path` | **Accept.** Necessary to locate and verify the bound heading JSON. |
| Stage 2 from `stage1/best.pth`; evaluation from `stage2_<room>/best.pth` | **Accept.** Confirmed against `run_haa_pipeline.sh`. Job verification remains incomplete. |
| Numeric `exit_status` plus string `outcome` | **Accept.** Correctly separates process status from diagnostic reason. |
| `child_exit.json` for every run type | **Accept as the frozen contract**, including jobs. Validation gaps remain in findings 3–4. |
| Commit sizes 230/330/207 insertions | Factually correct; **five** commits violate the changed-line limit. |

The subsecond watchdog interval and recording failed diagnostics with `passed:false` are reasonable resolutions of the prompt’s wording.

## Verification performed

Read the required SOP, approved v4 plan, plan/code reviews, notebook and role-change entry, prior experiment records, composed tooling, Coder prompts, and fix report before judging.

### Scope and immutable files

- Read the complete `git diff 4f92f54..2fe25bf`: **10 files, 1,907 insertions, 316 deletions**, all within round 2a.
- `git diff main...exp06-window --diff-filter=M --name-only`: **empty**.
- Compared requested pinned files and dynamically recomputed trainer/exp_03 import closures with `main`: **no byte mismatches**.
- `git status --short`: **clean**.
- `git diff --check 4f92f54..2fe25bf`: **passed**.
- `bash -n tools/exp06_launch.sh`: **passed**.
- In-memory `compile(...)` of nine changed Python files: **passed**, without writing bytecode.

### Test replay

Used the specified interpreter, data path, CPU-only environment, four OpenMP threads, disabled bytecode, and pytest cache provider disabled.

| Check | Result |
|---|---|
| Requested `python -m pytest tests/test_exp06_*.py tests/test_provenance.py -q -p no:cacheprovider` | Stopped before collection because pytest capture needed a writable temporary directory. |
| Read-only subset through `pytest.main`, `--capture=sys -q -rs -p no:cacheprovider`, excluding temporary-file fixtures | **183 passed, 3 skipped, 221 deselected**, 39.63 s. |
| Existing finalizer regression functions/parameterizations replayed with in-memory file I/O | **138 passed**. |
| Smoke regression cases replayed with in-memory receipts and mocked timer/memory callbacks | **20 passed**. |

The adapted runs are **not an unmodified full-suite replay**, and their counts overlap. They do not independently reproduce the Coder’s reported **404 passed, 3 skipped**.

Imports used `NUMBA_DISABLE_JIT=1` and a process-local shim to read existing Matplotlib caches. Closure subprocesses received the same accommodation. Finalizer replays used retained fixture repositories read-only; filesystem mutations/publication were redirected into memory.

### Contract checks

- Replayed source/data-byte drift, missing/empty closure, three-source bool/int drift, parameter-count typing, malformed checkpoints, heading/hash mismatches, child-role/lineage failures, stale logs, and live-PID refusal branches.
- Verified full checkpoint tensor equality, completion idempotence, and refusal when an existing completion differs.
- Verified zero/negative/nonfinite smoke-budget rejection, returned entry status `7`, failed outcomes, deadline/memory callbacks, and RNG isolation.
- Checked encoder symmetry/model-factory regressions in the read-only subset.
- Read actual exp_02 history and reproduced finding 3a.
- Ran the additional admission counterexamples documented above.
- Inspected retained descendant-test logs/receipts and independently exercised anonymous-pipe draining: both `early` and descendant `late` output arrived before sink completion.

Actual FIFO creation, abort renames, symlink promotion, timer signal delivery, and the full launcher lifecycle were not independently executed because this review prohibits file modification and signalling.

## Closure decision

**Severity blockers: 1, 2, 3, and 4.**

**Exact mandatory fix list before round 2a closes: 1, 2, 3, 4, 5, and 9.** Findings 5 and 9 remain mandatory under the fix-cycle prompt despite their should-fix severity. Finding 10 is a disclosed process nit.

**Round 2a may not close.**
