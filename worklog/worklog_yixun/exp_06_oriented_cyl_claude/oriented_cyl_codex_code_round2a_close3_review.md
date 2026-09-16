**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

# Round 2a close review 3

**Verdict: request changes. Round 2a may not close.**

Reviewed `bf51484..a977a68`, including every changed line, the required SOP/plan/reviews/experiment records, composed tooling, and this cycle’s Coder prompt and report. The original reproductions are fixed, but two new adversarial cases expose remaining evidence-validation gaps.

## Findings

### 1. Should-fix — Extra null artifact entries bypass full-set evidence comparison

**Location:** [tools/exp06_finalize.py:989](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:989), `verify_child`.

The artifact comparison iterates over both mappings’ keys but compares values using `.get()`. An absent key and an explicitly recorded `null` therefore compare equal.

**Reproduction:** Starting with a valid nine-child job, add this entry to `stage1/completion.json`:

```json
"artifacts": {
  "...existing entries...": "...",
  "nonexistent.pth": null
}
```

The file does not exist. Job finalization nevertheless returns **0**, prints `EXP06_FINALIZE_OK`, and produces `admissible_arm: true` in the memory-backed publication. An extra string-valued artifact is rejected; the null value bypasses that check.

**Why it matters:** The completion’s artifact mapping can differ from the mapping actually validated and hashed. This leaves the required full-set evidence comparison incomplete.

**Minimal fix:** Require identical artifact key sets, then compare values by direct indexing, or compare the complete mappings with strict equality. Add a job-level regression for an extra null artifact.

### 2. Should-fix — Invalid-measurement counters can contradict the observations

**Location:** [tools/exp06_finalize.py:802](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:802), `haa_metrics`.

The counters are now correctly typed and bounded by `n_samples`, but their relationship to the per-sample values remains unchecked.

**Independently accepted examples with three samples:**

- C50 values `[1.1, 1.3, NaN]`, correctly recomputed summary `n=2`, but `c50_outliers=0`.
- Three finite EDT values, but `edt_invalid=3`.
- Three finite T60 values, but `t60_invalid=2`.
- An omitted-T60 room with `t60_invalid=2`.

The first three pass standalone evaluation admission. All four also pass job admission with correctly updated metric-file hashes, returning **0** and `admissible_arm: true`.

**Why it matters:** A correctly hashed completion can certify mutually inconsistent measurement evidence.

The report’s concern about nonfinite results occurring without exceptions does not justify accepting these contradictions. The pinned writer in [sim_to_real/eval_haa.py:76](/home/yixunhu/codespace/xRIR_code_wt/sim_to_real/eval_haa.py:76) supports narrower, safe checks:

- `c50_outliers` equals the number of nonfinite C50 observations.
- EDT/T60 exception counters cannot exceed their respective nonfinite-observation counts.
- `t60_invalid` is zero in omitted-T60 rooms.

**Minimal fix:** Enforce those relationships while preserving legitimate NaN measurements and omitted-T60 summaries. Add standalone and job-level regressions with refreshed artifact hashes.

### 3. Nit — Extremely large JSON integers escape the named-refusal interface

**Locations:** [tools/exp06_finalize.py:763](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:763), `_numbers`; [tools/exp06_finalize.py:585](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:585), `_finite`.

A per-sample value or summary mean of `10**400` passes the integer type check, then raises an uncaught `OverflowError` during finite-number validation. `main()` does not return the documented named refusal and status 2.

**Impact:** No completion is published, so this remains fail-closed.

**Minimal fix:** Translate numeric-conversion overflow into a field-specific `ValueError`. This nit may be batched into round 2b.

## Disposition of the five close-2 findings

| Close-2 finding | Status | Evidence |
|---|---|---|
| **1 — Child liveness** | **Resolved** | Live child `child.pid` and `launch.pid` are rejected, including when `--owner-pid` equals that live child PID. A live launcher at the job root with dead children remains accepted. |
| **2 — Receipt/evidence identity** | **Partial** | Original receipt redirection, false timestamps and false/missing identity fields are rejected. A byte-identical receipt copy at another path is also rejected. Extra null artifact evidence still passes: finding **1** above. |
| **3 — Per-sample room/protocol** | **Resolved** | Original wrong-room/protocol cases are rejected, including job admission with refreshed hashes. Matching-room paths with cyclically permuted indices are rejected. |
| **4 — Malformed metric values** | **Partial** | Original malformed types, false medians and corrupt observations are rejected. Summary `n=2` against three finite observations is rejected despite being within bounds. Contradictory invalid counters remain accepted: finding **2** above. |
| **5 — Nested job-schema typing** | **Resolved** | Original failures and additional duplicate/unknown/empty rooms, boolean or floating seeds, malformed initialization/backbone/frame fields, invalid heading values and non-string child paths produce named CLI refusals, status 2, and no completion. |

## Verification performed

All checks were read-only and CPU-only. No files were modified, no GPU work ran, and no process was signalled.

### Source, pins and commit discipline

- `git diff bf51484..a977a68` — read completely; changes are confined to `tools/exp06_finalize.py` and its tests.
- `git diff --check bf51484..a977a68` — **passed**.
- `git diff main...exp06-window --diff-filter=M --name-only` — **empty**.
- Compared **42 distinct pinned/composed files** against `main`, including exp_03’s 12-file closure and the trainer’s import closure — **zero byte mismatches**.
- `bash -n tools/exp06_launch.sh` — **passed**.
- In-memory `compile()` of both changed Python files — **passed**, without writing bytecode.
- Worktree remained clean at `a977a68d67e83207114a0ba3f6e09de113196ca5`.

All **11 commits** satisfy the ≤200 inserted-plus-deleted-line limit and carry the required Claude coauthor and session trailers:

| Finding | Red commit / changed lines | Implementation / changed lines | Independently reproduced red failures |
|---|---|---|---:|
| 1 | `7c5b74f` / 30 | `77946c8` / 6 | 2 |
| 2 | `1be0997` / 52 | `2741f37` / 51 | 6 |
| 3 | `d8005a9` / 63 | `961d7e0` / 27 | 11 |
| 4 | `32db42a` / 29 | `d94bf1b` / 76 | 9 |
| 5 | `4a9f40c` / 30 | `e7fb365` / 20 | 4 |

Documentation commit `a977a68` changes **4 lines**. Maximum commit size is **76 lines**.

Historical replay loaded each red commit’s tests and then-current implementation using `git show`, without checkout. The **32 reproduced failures** match the report; every implementation follows its corresponding red-test commit.

### Tests and adversarial checks

The requested command was attempted:

```bash
python -m pytest tests/test_exp06_*.py tests/test_provenance.py -q -p no:cacheprovider
```

It failed before collection because pytest’s default capture required a writable temporary directory.

Read-only adaptations produced:

- Pytest with `--capture=sys` and selection excluding write-dependent tests: **183 passed, 3 CUDA skips, 340 deselected**. This included CPU symmetry/bit-exactness, factory parity, RNG isolation, recipe and provenance checks.
- All finalizer test cases replayed against retained fixtures with writes redirected to memory: **255 passed, 0 failed**.
- **60 additional adversarial variants**, followed by **five confirming job-CLI probes** for the accepted cases described above.

The adapted harness used process-local cache accommodations, disabled Numba JIT, checked liveness through `/proc`, and routed CLI tests through `main()`. It therefore does **not** independently establish real filesystem publication, atomic-renaming or launcher lifecycle behavior. The Coder’s reported **523 passed, 3 skipped** is not claimed here as an independently reproduced unmodified full-suite result.

## Exact blocking list

The Coder must fix **findings 1 and 2 in this review** before round 2a closes:

1. Require exact artifact-mapping agreement, rejecting extra null entries.
2. Reject invalid-measurement counters that contradict per-sample observations and omitted-T60 semantics.

Finding **3** is a nit to batch into round 2b.

**Round 2b consequence confirmed:** The HAA pipeline must keep its `launch.pid` at the **job root** (`seed<s>/launch.pid`). Child directories must not contain the live pipeline launcher PID.
