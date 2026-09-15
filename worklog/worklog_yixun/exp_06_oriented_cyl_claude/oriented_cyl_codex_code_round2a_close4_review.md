**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

# Round 2a close review 4

**Verdict: approve with changes. Blocking findings: none.**

Reviewed `a977a68..12e06fc`, including every changed line, the required SOP, approved plan and review history, experiment records, composed tooling, and this cycle’s Coder prompt/report. The mandatory close-3 reproductions are fixed. One nonblocking detail remains in the omitted-T60 representation check.

## Close-3 findings: disposition

| Finding | Status | Evidence |
|---|---|---|
| **1 — Extra null artifact entries** | **Resolved** | [tools/exp06_finalize.py:1019](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:1019) requires identical key sets, then compares directly indexed values with `strict_equal`. Extra null entries and equal-key-set mappings containing one incorrect hash were rejected in **each of all nine children**, covering stage 1, stage 2 and evaluation roles. |
| **2 — Counter reconciliation** | **Partial: counter fixes resolved; nonblocking NaN-only detail below** | [tools/exp06_finalize.py:824](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:824) enforces exact C50 counts, EDT/T60 upper bounds, and zero T60 invalid counts in omitted rooms. All four close-3 counter reproductions now fail standalone and job admission, including refreshed artifact hashes. Omitted-room infinities remain accepted. |
| **3 — Numeric overflow** | **Resolved** | [tools/exp06_finalize.py:592](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:592) translates conversion overflow into a field-named `ValueError`. CLI probes with `10**400` returned **2**, with no completion published. |

### Required boundary variants

Both standalone and job admission were exercised; altered job artifacts carried refreshed hashes.

- Counters equal to the nonfinite-observation count: **accepted**. One above: **rejected**.
- EDT/T60 counters below that count: legitimate missing measurements remain accepted.
- Omitted room with one finite T60 value: **rejected**.
- Omitted room with all-NaN T60 and `t60_invalid=2`: **rejected**; count zero: **accepted**.
- Non-omitted room with all-NaN T60 and `t60_invalid=3`: **accepted**, preserving `{mean: null, median: null, n: 0}`. Count four: **rejected**. This checks measurement consistency; statistical cohort eligibility remains a later producer’s responsibility.
- `10**400`: rejected across all six per-sample metric fields, every summary’s mean/median/n, invalid counters, sample count, numeric protocol fields, HAA history lr/train loss/validation loss/epoch, and heading phi/k.

## Numbered findings

### 1. Nit — Omitted-T60 validation accepts infinity as the NaN sentinel

**Location:** [tools/exp06_finalize.py:832](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:832).

`not finite` proves that no finite T60 measurements exist, but also accepts positive and negative infinity. The fix prompt requires every omitted-room T60 observation to be **NaN**, matching the pinned writer’s initialization and skipped measurement branch.

**Reproduction:** For `dampened_room`, both:

```text
t60 = [Infinity, NaN, NaN]
t60 = [-Infinity, -Infinity, -Infinity]
```

with summary `null` and `t60_invalid=0` pass standalone and refreshed-hash job admission: CLI status **0**, `admissible_arm: true`.

**Why it matters:** The accepted representation exceeds what the writer produces. This is nonblocking because the summary remains null and this room’s T60 is excluded from the registered analyses.

**Minimal fix:** Require `math.isnan(value)` for every omitted-room T60 value after numeric validation. Add positive/negative-infinity regressions for standalone and job admission. Batch into round 2b.

## Fixture assessment

The change at [tests/test_exp06_finalize.py:669](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp06_finalize.py:669) is **correct**.

The pinned writer initializes T60 to NaN and skips its computation for `NO_T60_ROOMS`. The fixture now reproduces that behavior while retaining explicit overrides for corruption tests. It corrects an unrealistic success fixture and preserves the negative cases.

## Verification performed

All checks were CPU-only and read-only. No files were modified, GPU work started, or processes signalled.

### Tests

The requested command was attempted with the prescribed interpreter and environment:

```bash
python -m pytest tests/test_exp06_*.py tests/test_provenance.py -q -p no:cacheprovider
```

It failed before collection because default pytest capture requires a writable temporary directory.

Read-only adaptations produced:

| Check | Result |
|---|---|
| Pytest with `--capture=sys`, excluding write-dependent tests | **183 passed, 3 CUDA skips, 357 deselected** |
| All finalizer cases, retained fixtures with writes redirected to memory | **272 passed, 0 failed** |
| Historical red-test replay using `git show`, without checkout | **13 expected failures reproduced** |
| Additional adversarial CLI probes | **90 probes**, followed by **10 focused confirmations** |

The CPU subset covered scene-only symmetry breaking, joint heading cancellation, bit-exact factory/trainer parity and RNG isolation.

The memory harness used process-local cache accommodations, disabled Numba JIT, checked liveness through `/proc`, and invoked CLI cases through `main()`. It does **not** independently establish real filesystem publication, atomic renaming or launcher lifecycle behavior. The Coder’s reported **540 passed, 3 skipped** is therefore not claimed as an independently reproduced unmodified full-suite result.

### Source, pins and commit discipline

- `git diff a977a68..12e06fc` — read completely; only the finalizer and its tests changed.
- `git diff --check a977a68..12e06fc` — passed.
- `git diff main...exp06-window --diff-filter=M --name-only` — empty.
- **42 pinned/composed files**, including `tests/test_provenance.py`, compared against `main` — **zero byte mismatches**.
- Exp_03’s 12-file closure digest remains `5ba818d8…`; the trainer’s 11-file closure digest remains `5b2da250…`.
- In-memory `compile()` of both changed Python files and `bash -n tools/exp06_launch.sh` — passed.
- Documentation commit’s executable AST — unchanged.
- Worktree remained clean at `12e06fcdd8c633db7c8af6b494762c32cd351bf6`.

| Red commit / changed lines | Implementation / changed lines | Reproduced red failures |
|---|---|---:|
| `3eaec27` / 26 | `c91e014` / 9 | 2 |
| `c4ec284` / 63 | `6418e11` / 18 | 9 |
| `027bd5b` / 21 | `f0898a1` / 16 | 2 |

Documentation commit `12e06fc` changes **6 lines**. Every commit meets the ≤200 changed-line limit and carries the required Claude Opus 5 coauthor and session trailers. Each implementation immediately follows its red-test commit.

## Exact blocking list

**Empty. Round 2a may close.**

Batch **finding 1 above—the omitted-T60 NaN-only check and its regressions—into round 2b**.
