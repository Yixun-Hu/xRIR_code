**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

# Round 2a close review 2

**Verdict: request changes. Round 2a may not close.**

Reviewed `2fe25bf..bf51484` on `exp06-window`, including the required briefing, approved plan, previous reviews, experiment records, Coder prompt/report, and composed tooling. The fixes resolve several major problems, but job admission still accepts contradictory evidence and live children.

## Findings

### 1. Blocker — Job admission bypasses child liveness checks

**Location:** [tools/exp06_finalize.py:895](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:895), `verify_child`.

The job branch calls `haa_train_evidence` or `haa_eval_evidence` directly. These calls omit `refuse_live_launch`, which runs only for the directory passed to top-level `finalize`.

**Reproduction:** Starting from a valid nine-child job, adding either `stage1/child.pid` or `stage1/launch.pid` containing my live process ID still produced `admissible_arm=True`. Process existence was checked through `/proc`; no signals were sent.

**Why it matters:** A job can certify a child whose process remains alive and may change the evidence being certified.

**Minimal fix:** Apply the liveness contract to every child during job admission. Always reject a live `child.pid`; handle any permitted launcher ownership explicitly rather than bypassing child checks. Add job-level regressions.

### 2. Blocker — The receipt being hashed need not be the receipt being validated

**Location:** [tools/exp06_finalize.py:875](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:875), `rehash_bound_evidence`; comparison loop at line 911.

The code hashes the path declared in `completion.child_exit_receipt`, then separately validates `<child>/child_exit.json`. It never establishes that these are the same file or compares the returned receipt fields with the completion.

**Reproductions accepted by job admission:**

- Redirect the recorded receipt binding to `stage1/args.json`, using that file’s correct hash; change the canonical receipt’s `child_pid`; retain otherwise valid canonical receipt contents.
- Set the completion’s `child_exit_time` to `"never"`.
- Supply a false `source_closure_sha256`, which is omitted from the comparison with freshly derived child evidence.

The redirected receipt case also accepted nonsense receipt metadata in the completion.

**Why it matters:** The completion can claim to bind receipt bytes and execution metadata that the job did not actually validate. This leaves the mandatory stale-evidence fix incomplete.

**Minimal fix:** Compare the canonical receipt’s resolved path, digest, PID and timestamps against the recorded binding; compare `child_exit_time` against the validated marker. Require agreement for all generated child evidence fields, including the source-closure digest.

### 3. Blocker — Per-sample results are not bound to the room or evaluation protocol

**Location:** [tools/exp06_finalize.py:789](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:789), `haa_eval_evidence`.

The validator checks selected per-sample metadata but ignores `ir_path` and the per-sample `split`, `num_shot`, and `eval_seed`.

**Reproduction:** In `eval/hallway`, I changed the per-sample paths to `class_room/<index>` and metadata to `split="val"`, `num_shot=1`, `eval_seed=777`. Hallway arguments and aggregate metrics remained unchanged. Both standalone evaluation admission and job admission accepted the result. For the job probe, the child completion contained the correct hash of the altered per-sample file.

**Why it matters:** Wrong-room or differently sampled observations can enter a nominally valid job, undermining the paired comparisons. The fix prompt explicitly requires the per-sample room to match the directory.

**Minimal fix:** Validate the writer’s `ir_path` representation against the room and corresponding indices. Require the protocol metadata written by the real evaluator to agree with arguments and aggregate metrics. Update the success fixture—which currently uses `x/0`, `x/1`, `x/2`—to represent real writer output, and test both admission paths.

### 4. Should-fix — Malformed metric values are silently accepted

**Location:** [tools/exp06_finalize.py:739](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:739), `haa_metrics`.

Required count fields are checked only for presence. Summary medians are not validated. Per-sample elements of invalid types are silently discarded when constructing the finite-value list.

**Independent accepted examples:**

- `c50_outliers: "nonsense"`
- `edt_invalid: -5`
- `t60_invalid: {}`
- `edt_error_s.median: {"nonsense": true}`
- Per-sample EDT values `["oops", {}, true]`, paired with `n=0`, `mean=null`, `median=null`

**Why it matters:** Corrupt observations can be treated as missing numerical measurements and certified successfully.

**Minimal fix:** Validate count and summary types, including strict integer counts and sensible bounds. Reject nonnumeric per-sample elements before filtering legitimate numeric nonfinite values. Validate median/null semantics against the real writer. Preserve its supported invalid-measurement representation.

### 5. Should-fix — Nested job schema failures still escape the refusal interface

**Locations:** [tools/exp06_finalize.py:940](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:940), `load_job_spec`; [tools/exp06_finalize.py:867](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:867), child `run_dir` validation.

**Reproductions:**

- Job `rooms=["hallway", 1, null, {}]` raises an uncaught sorting `TypeError`.
- Child completion `run_dir=null` raises an uncaught `Path` constructor `TypeError`.
- Job `init=null` is accepted and propagated into an admissible job completion.

Calling `main()` with the first two inputs confirmed uncaught exceptions rather than the documented named refusal and exit 2.

**Why it matters:** The new job-spec interface remains incompletely typed, and mandatory finding 9’s refusal contract is not satisfied throughout the finalizer.

**Minimal fix:** Validate list element types before sorting, validate paths before constructing `Path`, and require a valid nonempty initialization name. Add CLI-level regressions producing named `ValueError` refusals and no completion.

## Disposition of previous mandatory findings

| Previous finding | Status | Evidence |
|---|---|---|
| **1 — Data identity** | **Resolved** | Empty inventory with its correct digest, missing membership, inconsistent totals, and root mismatch are rejected. Expected membership comes from the pinned inventory helper. |
| **2 — Execution bindings** | **Resolved** | Agreement across all three argument copies no longer establishes false bindings. Run type, HEAD, registry, closure and provenance path are checked against execution evidence. |
| **3a — HAA history** | **Resolved** | Retained exp_02 control/seed0/stage1 passes: **1,001 rows, 101 validated epochs, best epoch 930**. Cadence-only history is correctly rejected. Off-cadence final validation and an unimproved epoch-0 initialization pass. |
| **3b — Metrics** | **Partial** | `not JSON`, wrong sample count and mismatched means are rejected. Malformed values still pass: finding **4** above. |
| **3c — Job admission** | **Partial** | Removed artifacts, ordinary stale hashes, wrong room claims in arguments, invented identities and wrong child seeds are rejected. Findings **1–3** above remain. |
| **4 — Sink/receipt contract** | **Resolved** | Sink failure reaches exit 3 before closure; canonical receipt PID, status, timestamp and digest checks reject the supplied nonsense cases. Job rebinding remains separately defective under finding **2**. |
| **5 — Ownership/recovery** | **Resolved for the launcher/recovery path** | Separate launcher/child PIDs, both launch roots, exclusive diagnostic directories and concrete abort reasons are implemented. Live root recovery is rejected; the declared owner can finalize only after its child is dead. |
| **9 — Malformed nested inputs** | **Partial** | Original malformed model-state, `git_state`, and `mutable_inputs` cases now produce named refusals. New job-boundary failures remain: finding **5** above. |

## Assessment of the Coder’s interface choices

- **Mandatory `--job-spec`:** Appropriate. Defining its consumer schema now is within round 2a. Round 2b must write it before execution and preserve its binding. Finding 5 must be fixed first.
- **Integer `seed` in every HAA child’s `args.json`:** Appropriate for binding children to their pipeline seed. Round 2b must record it in effective provenance arguments too, keeping its meaning distinct from `eval_seed`.
- **Room-named evaluation directories:** Appropriate and consistent with the planned layout. Directory naming alone does not resolve finding 3.
- **Sink failure exit 3:** Appropriate; it distinguishes log-transport failure from successful closure.
- **`--owner-pid`:** Appropriate for the owning launcher’s finalization. Recovery correctly omits it. Child admission still needs the liveness checks described above.

## Verification performed

All Python checks used the requested interpreter and CPU environment. No files were modified, no GPU work ran, and no process was signalled. RIR readback used lazy training-row access.

### Source, commits and pins

- Read the complete four-file round diff, including removed lines, and inspected the test/implementation commit sequence.
- `git diff --check 2fe25bf..bf51484` — **passed**.
- `git diff main...exp06-window --diff-filter=M --name-only` — **empty**.
- `bash -n tools/exp06_launch.sh` — **passed**.
- In-memory `compile()` of the three changed Python files — **passed**, without generating bytecode.
- Checked all **20 commits**: each is at most 200 inserted-plus-deleted lines; maximum **182**. All carry the required Claude coauthor and session trailers.
- Compared **42 distinct pinned/composed files** against `main`: **zero byte mismatches**. This included exp_03’s 12-file closure and the trainer’s 11-file closure.
- Worktree remained clean at `bf514843eb0740a0295c54ead15c8f645a6397de`.

The commit sequence supports test-first development. I inspected the pairs and replayed HEAD tests; I did not independently reproduce historical red-run execution.

### Tests and adversarial checks

The requested command:

```text
python -m pytest tests/test_exp06_*.py tests/test_provenance.py -q -p no:cacheprovider
```

stopped **before collection** because pytest’s default capture required a writable temporary directory.

Read-only adaptations produced:

- Write-free pytest subset with `--capture=sys` and cache provider disabled: **183 passed, 3 CUDA skips, 305 deselected**. This covered CPU symmetry, bit-exactness, recipe/RNG isolation, heading readback and provenance checks.
- Finalizer test-function replay with in-memory filesystem/checkpoint publication: **220 cases passed**, including three CLI tests routed through the real `main()` in-process. Harness defects encountered initially were corrected and affected cases rerun.
- Three focused launcher ownership/preflight tests with in-memory PID files and simulated GPU-query output: **passed**.
- Adapted shell sink probe using anonymous pipes: consumed output, reported `SINK_FAILED status=1`, selected `_ABORTED_sink_failed`, and **exited 3 before the close step**.
- Independent adversarial probes produced the acceptances and refusals reported above.

These counts overlap and are **not** a combined full-suite result. Cache-related imports used process-local accommodations and `NUMBA_DISABLE_JIT=1`. Actual FIFO creation, filesystem renames, atomic publication and the unmodified shell lifecycle were not independently replayed under this sandbox. The Coder’s reported **488 passed, 3 skipped** therefore remains a reported full-suite result, not my independently reproduced count.

## Exact blocking list before round 2a closes

The Coder must fix **findings 1, 2, 3, 4 and 5 in this review**, with failing regressions followed by implementation:

1. Enforce child liveness during job admission.
2. Bind canonical receipt and completion evidence to what was validated.
3. Bind per-sample room and evaluation protocol.
4. Reject malformed metric values and counts.
5. Complete nested job-schema validation and named CLI refusals.

These are remaining round-2a contract defects; they must not be deferred to round 2b.
