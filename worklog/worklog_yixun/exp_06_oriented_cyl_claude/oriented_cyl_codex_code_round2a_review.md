**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15

# exp_06 oriented_cyl — round 2a review

**Verdict: request changes. Blocking findings: 1–4.**

Reviewed `a950605..4f92f54` on `exp06-window`, ending at `4f92f54e554ba210563c61d36cdf06fb570785ba`. The training composition, parser parity, model initialization, heading readback, and pinned-file checks pass the checks described below. The completion machinery has material admission gaps.

No files were modified, GPU work started, or processes signalled. Adversarial finalizer checks used production functions with file reads and writes backed by **in-memory fixtures**, not fabricated directories on disk.

## 1. Findings

### 1. Blocker — full completion never revalidates execution inputs

**Where:** [tools/exp06_finalize.py:126](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:126)

`closure_record(...)[1]` hashes the **reviewed Git blobs**. Those hashes remain unchanged when working-tree files change. The finalizer discards the fresh per-file records and never calls `provenance.revalidate`. It also neither requires nor validates `train_data_identity`.

Verified counterexamples:

- Changed training-data bytes: inherited `revalidate(record)` reports a mismatch; finalization still returns `admissible_arm: true`.
- Changed working-tree trainer bytes: the same outcome.
- Missing data identity: accepted.
- Empty training closure with its correctly computed empty-list digest: accepted.

**Why it matters:** Completion does not establish the input-revalidation contract in plan §5. Its “source closure drift” check cannot detect ordinary working-tree drift.

**Minimal fix:** Require complete provenance, verify the actual expected closure membership, compare reviewed/spawn/current source hashes, and revalidate all required data and mutable inputs before publishing completion. Add tests that change the input bytes, rather than only corrupting the recorded Git-blob digest.

### 2. Blocker — startup arguments and checkpoint arguments are not validated consistently

**Where:** [tools/exp06_finalize.py:137](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:137), [tools/exp06_recipe.py:138](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_recipe.py:138)

The finalizer checks `args.json`, ignores `provenance.json.effective_args`, and compares `last.pth["args"]` using ordinary dictionary equality.

Verified:

- Provenance records `max_train_batches=3`, while retained args/checkpoint metadata claim zero: **accepted**.
- `last.pth["args"]` contains `tf32=1` and `no_save=0`, while JSON contains `True` and `False`: **accepted**.
- Floating-point parameter counts pass the purportedly type-strict derived check.
- `check_all` accepts missing operational fields and missing exp_06 provenance fields.

**Why it matters:** A retained configuration can disagree with the startup configuration. The explicitly required schema check on checkpoint arguments is bypassed by Python’s bool/int equality.

**Minimal fix:** Validate startup, JSON, and checkpoint arguments separately; require recursive type-strict agreement and consistent provenance bindings. Require the current-run fields, while retaining the deliberately narrower historical recipe-normalization path. Validate parameter-count values as native integers.

### 3. Blocker — HAA completion accepts missing provenance and contradictory evidence

**Where:** [tools/exp06_finalize.py:178](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:178), [tools/exp06_finalize.py:206](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:206), [tools/exp06_finalize.py:235](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:235)

The HAA branches do not load provenance or verify its run type, closures, or inputs. Additional gaps contradict the frozen contract:

- Heading bindings do not require `phi_deg` or verify its relationship to `k`.
- Evaluation checks only the **presence** of `meta.heading` and `checkpoint_sha256`; neither must agree with the arguments or bound inputs.
- Side labels are not required.
- Training history/summary/checkpoints are only hashed, even when their contents are malformed.
- Job completion accepts either HAA child type at every expected path and trusts `admissible_arm: true` without validating the child completion’s evidence.

Verified admissible counterexamples included:

- HAA training without provenance or `phi_deg`, with non-JSON history/summary and invalid checkpoint bytes.
- HAA evaluation with args heading `k=128`, metadata heading `k=0`, conflicting checkpoint hashes, and no side labels.
- A nine-child “fine-tuning job” whose children contain only `{"run_type":"haa_eval","admissible_arm":true}`.

**Why it matters:** These are existing round-2a finalizer branches, not absent round-2b implementation. Freezing them now would allow invalid or diagnostic evidence to receive admissible completion.

**Minimal fix:** Apply provenance/run-type/input validation to every applicable branch; enforce the declared heading, checkpoint, and side-label contracts; validate structured artifacts; and verify child completion schemas, path-specific roles, identities, and bound artifact hashes.

### 4. Blocker — the marker does not establish that the log is closed

**Where:** [tools/exp06_launch.sh:87](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_launch.sh:87), [tools/exp06_finalize.py:80](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:80)

Direct redirection avoids the particular race with a concurrently writing `tee`. **`tail` is only a reader**, so it does not itself threaten marker ordering.

However, the launcher waits only for the direct child. A descendant retaining stdout/stderr can write after that child exits. The finalizer checks marker syntax, then hashes the log separately; it does not establish writer termination or recheck the log before publication. Recovery finalization also ignores `launch.pid`.

An otherwise accepted fixture remained admissible with `launch.pid` identifying the live review process.

**Why it matters:** The marker is not guaranteed to remain the last line under all supported finalization paths. The finalizer can certify a log while a surviving writer can still change it, contrary to §5.

**Minimal fix:** Establish completion of the writer lifecycle before appending the marker—for example, drain a logging pipe to EOF and wait for its sink. Bind verifiable child-exit evidence, reject active attempts during recovery, and validate/hash a stable log snapshot before publication. Add a CPU test with a descendant that retains the output descriptor.

### 5. Should-fix — diagnostic and recovery orchestration omit parts of the lifecycle

**Where:** [tools/exp06_launch.sh:106](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_launch.sh:106), [tools/exp06_launch.sh:139](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_launch.sh:139), [tools/exp06_finalize.py:153](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:153)

Smoke/probe modes create no tracked launch PID, closed child log, or diagnostic completion. Consequently, the live-launch guard cannot identify those launches through its own PID-file mechanism. Diagnostic finalization accepts an empty directory and no receipt, without checking `no_save`.

Recovery `finalize` accepts `--reviewed-commit` and `--gpu` but does not use them, and successful recovery does not update `final`. Failure handling renames the attempt but leaves the run log without the SOP’s `_ABORTED_<reason>` suffix.

**Minimal fix:** Share lifecycle handling across modes, validate diagnostic receipts and the applicable no-save contract, retain launcher ownership through finalization, and reuse promotion/abort handling in recovery. Explicitly distinguish failed diagnostic receipts from successful diagnostics.

### 6. Should-fix — the launcher does not pin the registered data root

**Where:** [tools/exp06_launch.sh:56](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_launch.sh:56), [tools/exp06_train.py:44](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_train.py:44)

The launcher sets GPU/hash-seed/thread variables but neither sets nor validates `XRIR_DATA_PATH`.

With that variable absent, I verified:

```text
exp06_train.DATA_ROOT:
  /home/yixunhu/data_cache/AcousticRooms

dataset BASE_DATA_PATH:
  /home/yixunhu/codespace/xRIR_code_wt/data
```

Thus the training dataset and provenance helper have different fallback roots. The latter path does not exist in this worktree.

**Minimal fix:** Export or require the registered root in the launcher, and derive provenance’s root from the same resolved configuration used by the dataset. Include it in the dry-run golden.

### 7. Should-fix — smoke bounds and returned failure statuses are insufficiently checked

**Where:** [tools/exp06_smoke.py:69](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_smoke.py:69)

`alarm_seconds=0` is accepted and disables the alarm. A stub entry returning status `7` is recorded as `"ok"` because its return value is ignored.

The memory check correctly rejects an excessive peak **after the entry finishes**, as the Coder prompt requested. It does not enforce the plan’s co-tenant memory ceiling during execution.

**Minimal fix:** Validate positive finite budgets before invoking the entry; handle integer return statuses; and resolve the distinction between retrospective peak rejection and an enforced allocation ceiling before co-tenant smokes. Keep the receipt wording accurate.

### 8. Nit — parameter-count validation changes global Torch RNG state

**Where:** [tools/exp06_recipe.py:114](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_recipe.py:114)

A cold `expected_param_counts` call constructs a randomly initialized model and advances Torch’s global RNG. A warm cached call does not. I verified the state change.

This does not affect the current training main, which counts its existing model, but makes validation depend on cache history if composed before subsequent random operations.

**Minimal fix:** Construct the counting model inside `torch.random.fork_rng(devices=[])`.

### 9. Should-fix — malformed finalizer inputs escape the documented refusal interface

**Where:** [tools/exp06_finalize.py:141](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:141), [tools/exp06_finalize.py:382](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:382)

A checkpoint missing `model` raises uncaught `KeyError`, rather than the promised named `ValueError`/CLI exit 2. Other malformed container or checkpoint contents have similar paths.

**Why it matters:** These failures remain fail-closed, but violate the orchestration interface.

**Minimal fix:** Validate loaded structures and normalize expected input-decoding/schema errors into named refusals. Test malformed containers as well as missing files.

### 10. Nit — commit-size and TDD deviations should remain explicit

**Where:** `tests/test_exp06_recipe.py:1` / commit `0a1dd8d`; round commit history.

All eleven commits contain the required trailers, and tests land with their implementations. Seven exceed 200 changed lines: **220, 218, 208, 335, 206, 279, 225**.

I treat size as a nit because the SOP says “generally < 200,” although the Coder prompt was stricter. The disclosed cycle-1a red run, obtained by moving an already-written implementation aside, does not demonstrate the prescribed red-before-implementation sequence.

**Minimal fix:** Preserve the disclosure; use smaller subsequent cycles and obtain meaningful failing tests before implementing the fixes. No history rewrite is needed.

## 2. Disposition of the Coder’s 20 discrepancies

| Item | Assessment |
|---|---|
| D-1 — commit sizes | Nit; finding 10. |
| D-2 — red/green bookkeeping | Test corrections are reasonable; the manufactured cycle-1a red is a process deviation, finding 10. |
| D-3 — heading admissibility | Accept the established worklog-only exception plus closure-equals-HEAD requirement. This label does not itself constitute reviewed-digest approval. |
| D-4 — schema version 1 | Accept: `ckpt/exp06` does not exist, so no produced heading records need migration. |
| D-5 — deferred nits | Accept for this scope. The stale `36.8` literal remains a nit; realized readback is unchanged. |
| D-6 — normalization tuple / missing `no_save` | Accept. Explicitly reporting unrecorded fields is appropriate. |
| D-7 — `check_all` / cached counts | Optional budget arguments are reasonable; validation gaps and RNG behavior are findings 2 and 8. |
| D-8 — `prepare_args` signature/deletions | Accept. It mutates a local args object, not the trainer’s module namespace. M-only refusal matches the prompt. |
| D-9 — provenance destination / run-type coupling | Destination rule is reasonable. Diagnostic/no-save enforcement remains incomplete, finding 5. |
| D-10 — inventory-cache handling | Accept the read-only shared-cache reuse and temporary fallback design. |
| D-11 — no completion wall-clock timestamp | Accept; using the recorded child-exit time supports byte-identical reruns. |
| D-12 — frozen HAA contracts | Reject as implemented; finding 3. Several claimed requirements are not enforced. |
| D-13 — smoke implementation details | Timer choice and exception approach are reasonable; finding 7 covers remaining gaps. |
| D-14 — redirection plus `tail` | Reasonable logging choice, but the claimed closure guarantee is incomplete; finding 4. |
| D-15 — probe 2400 s / 46 GiB | Accept as a separate exclusive-card probe budget, not a rung-4 smoke budget. GPU fit remains unverified. |
| D-16 — omitted HAA smokes | Accept: round-2b entries are explicitly out of scope. |
| D-17 — recovery/abort/promotion | Partial; finding 5. |
| D-18 — preflight strictness | Helper refusals verified. PID ownership across launcher modes remains incomplete. |
| D-19 — CLI shape | Accept the optional subcommand shape; malformed-input refusal claim is overstated, finding 9. |
| D-20 — GPU checks not run | Appropriate and honestly disclosed. They remain required on the later ladder. |

## 3. Verification performed

### Context and diff

Read the SOP, approved v4 plan, three Codex plan reviews, Fable round-1 review, notebook including the role change, round-2a prompt/report, prior experiment results/analysis and September 14 diagnostics, and the relevant composed trainer, provenance, evaluation, HAA, profile, and gate code.

Read the complete round diff:

```text
git diff a950605..4f92f54
```

Scope: **12 files, 2,361 insertions, 22 deletions**.

### Environment and test replay

Checks used the specified interpreter and environment:

```text
/home/yixunhu/miniconda3/envs/xRIR/bin/python
PYTHONPATH=/home/yixunhu/codespace/xRIR_code_wt
XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
CUDA_VISIBLE_DEVICES=''
OMP_NUM_THREADS=4
PYTHONDONTWRITEBYTECODE=1
NUMBA_CACHE_DIR=/tmp/exp06_codex_review_numba
```

| Check | Result |
|---|---|
| `python -m pytest tests/test_exp06_*.py tests/test_provenance.py -q -p no:cacheprovider -rs` | Stopped before collection: no writable temporary directory for pytest capture. |
| Retry with `-s --maxfail=1` | Collection stopped on Librosa/Numba’s unavailable writable cache. |
| Read-only subset via `pytest.main`, `--capture=sys`, `-p no:cacheprovider`, excluding temporary-file fixtures | **151 passed, 3 skipped, 111 deselected**, 40.66 s. |
| `bash -n tools/exp06_launch.sh` | Passed. |
| In-memory `compile(...)` of all changed Python files | **11 passed**; no bytecode files written. |
| `git diff --check a950605..4f92f54` | Clean. |
| Launcher dry runs: `full`, `probe`, `smoke`, `finalize` | Passed; argv inspected against the plan/prompt. |

The read-only subset used `NUMBA_DISABLE_JIT=1` and a process-local Matplotlib cache-access shim to read existing caches; closure subprocesses received the same import accommodation. This was **not an unmodified full-suite replay**. The Coder’s **262 passed / 3 skipped** total was not independently reproduced here.

### Contracts and adversarial checks

- **Schema:** 33/33 real exp_05 fields classified; 38/38 total declared fields are disjoint. Both exp_01 recipes pass after normalization. The S-tier exp_05 recipe correctly differs in its four capacity fields.
- **Trainer:** Shared parser defaults and flag types match; yaw augmentation 1 and unknown backbones are refused. Simple model state matches the trainer bit-for-bit under the same seed. Trainer module bindings remain unchanged. Provenance construction preserved Torch RNG state.
- **Full finalizer, in-memory I/O:** Ordinary truncated args, mid-epoch checkpoint, foreign epoch-state mismatch, trailing log output, and nonzero child exit were refused without publication. Identical reruns preserved completion bytes; differing existing completion was refused and preserved. Findings 1–4 document accepted counterexamples.
- **Preflight:** Wrong/full-versus-abbreviated HEAD, dirty tree, live PID, and busy GPU refusal branches exercised. Process-existence and GPU responses were mocked; no signals or GPU work occurred.
- **Smoke:** Alarm callback and excessive-memory paths yield exit 3; unknown entry refused. CPU fixture state strictly loads into `build_xrir_exp06('cylindrical_oriented', 8)`. Actual timed alarm delivery and filesystem fixture publication were not replayed.
- **Launcher:** Full argv matches the registered trainer flags, including `--epoch-ckpt-every 1`, `--save-every 500`, TF32, and batch 32 × accumulation 2. GPU/hash-seed/thread assignments are present. Data-root omission is finding 6. Actual abort renames, symlink promotion, and child/log lifecycle were inspected statically, not executed.

### Heading readback and cycle 0

All four rooms remain **estimated −90°, `k=128`, 12/12 stable refits**:

| Room | 5 ms contrast | 50 ms contrast |
|---|---:|---:|
| Classroom | 12.9211 | 9.0362 |
| Dampened | 11.3880 | 8.6690 |
| Hallway | 18.4526 | 11.2928 |
| Complex | 11.9530 | 10.8063 |

RIR calculations used only training rows from read-only memory maps. Current closure and HEAD digests agree. Input revalidation succeeds on unchanged caches and rejects a simulated changed metadata hash. Worklog-only dirtiness remains confirmatory; outside-worklog dirtiness becomes diagnostic. S2’s distinct refusal code is present in the reviewed CLI implementation.

The selected encoder/factory tests also passed the active-yaw, heading-cancellation, bit-exact identity, parameter-delta, and CPU dtype checks.

### Pinned files and commit hygiene

- **41 pinned/composed files checked byte-for-byte against `main`: zero mismatches.**
- Exp_03 closure: **12 files**, unchanged digest:

```text
5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48
```

- Trainer closure: **11 files**, unchanged digest:

```text
5b2da2504e220b63bfc932cf0d83c143261e117edea1838e6907159c3b12eb13
```

- All eleven commits have both required trailers; tests appear with implementation.
- Final worktree status is clean.

**Diff-range clarification:** `git diff main...exp06-window --diff-filter=M --name-only` is **empty**, correctly: the heading files are additions relative to the merge base with `main`. The **round-range** command, `git diff a950605..4f92f54 --diff-filter=M --name-only`, lists exactly:

```text
tests/test_exp06_heading.py
tools/exp06_heading.py
```

## 4. Required fixes before round 2a closes

The exact blocking set is:

1. **Revalidate complete source/data/input provenance during full finalization.**
2. **Bind and type-strictly validate startup, JSON, and checkpoint arguments.**
3. **Enforce HAA provenance, heading/output, and child-completion contracts.**
4. **Establish closed-writer/log evidence before certifying completion, including recovery.**

Fix these with adversarial regression tests and re-review the fixes before starting round 2b.

**Final verdict: request changes.**
