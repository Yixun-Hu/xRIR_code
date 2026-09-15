**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

# exp_06 `full_train` integrative review

**Verdict: request changes.**

Reviewed `main...exp06-window` at **`12e06fcdd8c633db7c8af6b494762c32cd351bf6`**, including every added line, the required SOP, approved plan and amendments, review history, Coder prompts/reports, experiment records, and composed tooling.

The encoder, heading estimator, and normal synthetic training-completion path work. The merge changes no existing source file. However, additional adversarial checks expose training admission and diagnostic-launch failures that the passing tests do not cover.

## Numbered findings

### 1. Blocker — Training approvals and orchestration source bindings are missing

**Locations:** [tools/exp06_train.py:152](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_train.py:152), [tools/exp06_finalize.py:414](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:414), [tools/exp06_finalize.py:1240](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:1240).

The trainer records its current HEAD as `reviewed_commit`; preflight compares HEAD with a caller-supplied commit. Neither checks approved code digests. The approval template and `tools/exp06_profiles.py` are absent. Full finalization validates exactly one Python training closure, which excludes both the launcher and finalizer.

**Reproduction:** Independently changing the recorded repository’s `tools/exp06_launch.sh` or `tools/exp06_finalize.py` bytes in the memory overlay still produced **`admissible_arm: true`**.

**Why it matters:** §6.4 explicitly requires pre-execution code approvals and binding shell orchestration as files. This matters particularly when later rounds can change the finalizer during a long training run. A1 defers later modules but supplies no replacement for these training-time bindings.

**Minimal fix:** Implement the training-critical subset of code approvals now; bind the launcher and finalizer alongside the training closure and verify those identities at launch and completion. Leave later evaluation approvals pending. Any alternative lifecycle needs an explicit preregistered amendment and equivalent reviewed evidence.

### 2. Should-fix — Training provenance omits geometry inputs

**Locations:** [tools/exp06_train.py:143](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_train.py:143), [tools/exp06_finalize.py:259](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:259).

The reused `train_data_identity` inventories training WAVs only. Its membership check therefore establishes no identity for the metadata and depth maps consumed by the dataset.

**Reproduction:** In the synthetic training split, changing a metadata file changed the pinned dataset’s returned source position from `[1, 2, 3]` to `[9, 8, 7]`. Full finalization accepted both versions with the same recorded WAV inventory.

**Why it matters:** Changed model inputs can pass §5’s input-revalidation contract and receive an admissible completion.

**Minimal fix:** Add an exp_06-owned inventory of the consumed metadata and receiver depth maps, with independently derived membership and finalization-time rehashing. Also account for inputs used to produce the recorded epoch test losses. Keep the pinned helper unchanged.

### 3. Should-fix — A live child named in the exit receipt can be admitted

**Locations:** [tools/exp06_finalize.py:313](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:313), [tools/exp06_finalize.py:358](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:358).

Liveness checks inspect optional `launch.pid` and `child.pid` files. The required `child_exit.json.child_pid` is validated as a positive integer but never checked for liveness or agreement with `child.pid`.

**Reproduction:** With PID sidecars absent, setting the receipt’s child PID to the currently running reviewer process still yielded **`admissible_arm: true`**.

**Why it matters:** A partially restored attempt can pass the closed-writer contract while its recorded child remains alive.

**Minimal fix:** Check the receipt’s child PID independently and enforce agreement with any child PID sidecar. Preserve the owner exception solely for the launcher.

### 4. Should-fix — The operational schema checks presence without validating values or types

**Location:** [tools/exp06_recipe.py:110](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_recipe.py:110).

Consistent corruption across all three argument copies bypasses validation.

**Reproductions:** Each returned `check_all(...) == []` and an admissible full completion:

```text
epoch_ckpt_every = 2
epoch_ckpt_every = True
save_every = None
num_workers = "twelve"
```

**Why it matters:** §5 specifies a type-strict schema and `epoch_ckpt_every=1`. Agreement between copies does not establish valid arguments.

**Minimal fix:** Validate operational field types and applicable bounds; enforce the registered current-run checkpoint cadence. Preserve the explicitly narrower historical path.

A separate, harmless textual deviation is that provenance fields form a **fifth** schema class, whereas §5 places them within the four-class scheme.

### 5. Blocker — Failed smoke/probe children leave the launcher successful

**Location:** [tools/exp06_launch.sh:69](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_launch.sh:69).

`diagnostic()` records the child’s failure through the finalizer, then returns the finalizer’s successful publication status. It never propagates `CHILD_STATUS` or performs the registered aborted-run renames.

**Reproduction:**

- The real finalizer CLI accepted a failed diagnostic record with child status **3**, wrote `passed: false`, and returned **0**.
- The launcher’s diagnostic function, with that child/finalizer behavior supplied through adapters, printed `LAUNCHER_CONTINUED status=0` and exited **0**.

Consequently, smoke mode can continue after a failed rung, and a failed probe command appears successful.

**Minimal fix:** After recording diagnostic completion, propagate a nonzero child status and rename the failed attempt/log. Handle finalizer refusal similarly. Retain the useful `passed: false` completion evidence.

### 6. Should-fix — A diagnostic can “pass” without timing or memory evidence

**Location:** [tools/exp06_finalize.py:556](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:556).

This receipt is sufficient for a successful diagnostic completion:

```json
{"diagnostic": true, "argv": ["--no-save"], "exit_status": 0}
```

**Reproduction:** With a valid closed log and exit receipt, it produced **`passed: true`**.

**Why it matters:** §6.4 requires peak-memory and timing evidence; §9 uses that evidence to gate training. Diagnostic dispatch also bypasses the stated provenance/run-type requirement.

**Minimal fix:** Require and validate the runner’s identity, timing, memory, budget and outcome fields. Bind diagnostic provenance, or explicitly amend the plan to define a sufficiently complete receipt substitute.

### 7. Should-fix — The finalizer still does not enforce the heading consumer contract

**Location:** [tools/exp06_finalize.py:661](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:661).

`_heading_binding` calls `read_heading_json` without the room directory and does not require confirmatory admissibility or an approved heading-code identity.

**Reproductions using the real hallway heading:**

- After changing `meta.json` in memory, `read_heading_json(..., room_dir=...)` refused it; `_heading_binding` accepted it.
- A valid record explicitly marked `admissibility: "diagnostic"` was accepted.

**Minimal fix:** Resolve and bind the HAA cache root, invoke input revalidation, and require the approved confirmatory heading identity.

**Disposition:** Finish in **round 2b before HAA execution**. This is the remaining consumer portion of Fable S3 and does not block simulation pretraining, whose heading is fixed at zero.

### 8. Nit — Tensor equality does not enforce matching checkpoint dtypes

**Location:** [tools/exp06_finalize.py:546](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:546).

`torch.equal` accepts equal-valued tensors across different dtypes.

**Reproduction:** Converting only `epoch_012.pth` tensors from float32 to float64 still admitted the attempt.

**Minimal fix:** Require matching dtype and shape before exact value comparison. Nonblocking for the native writer, which writes matching dtypes.

### 9. Nit — The probe does not literally use every full-recipe setting

**Location:** [tools/exp06_launch.sh:214](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_launch.sh:214).

The probe inherits `decay_epochs=50`, while §5 specifies 3 and §9 describes the full-recipe probe. It also inherits `log_interval=20` rather than 50.

**Minimal fix:** Pass the registered values explicitly. The scheduler discrepancy does not change the 200 training updates within this single-epoch probe.

## Answers to the six review questions

| Question | Assessment |
|---|---|
| **1. Plan implementation** | **§2.2 passes:** channel construction, parameter delta, factory identity, active-yaw symmetry breaking, joint heading cancellation and zero-shift exactness. **§4 estimator passes** on all four real rooms. **§§5/6.4 training admission is incomplete** for findings 1–6. Other identified deviations are recorded above. |
| **2. CPU composition and refusals** | The normal synthetic composition passed through `prepare_args`, startup provenance, `check_all`, and `exp06_finalize --run-type full`, with real serialized checkpoint/artifact hashes. Existing refusal cases passed, but the additional counterexamples above prevent an unconditional fail-closed claim. Heading generation/readback works; finalizer heading revalidation remains finding 7. |
| **3. Launch commands** | Full dry-run prints the registered arguments, including `--epoch-ckpt-every 1`, `--save-every 500`, and the required environment. Smoke prints the exact three training smokes and CPU fixture command. Probe prints the intended 200-batch, 32×2, TF32, no-save configuration, with finding 9’s literal discrepancy. HAA smokes are appropriately deferred under A1. |
| **4. Pinned files** | **Byte identity passes.** Forty-one pinned source files plus both pinned test files match `main`. The modified-file triple-dot diff is empty. Independent test coverage and its sandbox limits are detailed below. |
| **5. Earlier findings** | Fable S1/S2 are resolved; S3’s helper exists, but consumer enforcement remains finding 7. The previously closed round-2a regressions passed replay. The close-4 **omitted-T60 NaN-only nit remains deferred to round 2b**, as instructed, and does not block training. |
| **6. Peer merge safety** | **Pass for source compatibility:** all 16 changes are additions under `model/`, `tools/`, and `tests/`. No exp_04/05 training or evaluation source closure changes. Current `main`’s additional record changes do not constitute source changes from this branch. |

The four real HAA headings are later HAA prerequisites; they are not inputs to simulation pretraining’s zero-heading model path.

## Verification performed

All work was CPU-only. No repository, checkpoint, cache or retained fixture file was modified; no GPU work was started and no process was signalled.

### Static and source checks

- `git diff main...exp06-window` — all **6,307 added lines across 16 files** read.
- `git diff --check main...exp06-window` — passed.
- `git diff main...exp06-window --diff-filter=M --name-only` — **empty**.
- Direct `main` versus branch comparison of modified Python/shell files — **empty**.
- `bash -n tools/exp06_launch.sh` — passed.
- In-memory `compile()` of all **15 added Python files** — passed.
- Byte comparisons against `git show main:<path>` — **43 files matched**, including both pinned tests.
- Final HEAD remained `12e06fcdd8c633db7c8af6b494762c32cd351bf6`; worktree status remained clean.

Recomputed closure digests:

```text
train_xRIR_backbone: 11 files
5b2da2504e220b63bfc932cf0d83c143261e117edea1838e6907159c3b12eb13

eval_yaw_rotation: 12 files
5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48
```

### Tests and execution checks

Used the prescribed interpreter/environment, empty CUDA visibility, disabled bytecode, and pytest cache disabled.

The requested command:

```bash
python -m pytest tests/test_exp06_*.py tests/test_provenance.py \
  tests/test_exp03_record_tools.py -q -p no:cacheprovider
```

failed **before collection** because default capture could not obtain a writable temporary directory.

Read-only adaptations produced these results; counts overlap:

| Check | Result |
|---|---:|
| Pytest with `--capture=sys` and write-dependent cases deselected | **202 passed, 4 skipped, 369 deselected** |
| All finalizer cases replayed with retained fixtures and memory-only writes | **272 passed, 0 failed** |
| Exp_03 record tests through memory adapters | **31 passed, 1 GPU-dependent skip**, across replay and corrected stat-adapter rerun |
| Launcher child lifecycle with anonymous pipes/memory-backed descriptors | **4 passed**: success, child failure, descendant draining, failed sink |
| Real hallway wrapper, training rows only | **12 samples each at k=0 and k=128 passed** |
| Real heading generation and JSON readback | **4 rooms passed**, all −90°, k=128, 12/12 stable leave-one-out |
| Actual preparation/provenance/schema → synthetic full finalizer CLI | **Passed**, `admissible_arm: true`; artifact hashes independently matched |

The selected pytest cases include the exp_03 closure pin and five no-write provenance cases. I do **not** claim an independently reproduced unmodified full `tests/test_provenance.py` run. The notebook separately records the Planner’s **571 passed / 4 skipped** at this HEAD.

Adapters disabled Numba JIT, accommodated existing import caches, redirected writes to memory, and replaced signal-zero liveness queries with `/proc` checks. Filesystem publication and rename atomicity were not independently established. Initial adapter failures were corrected; they were not product-code failures.

Also executed:

```bash
bash tools/exp06_launch.sh full  --dry-run --gpu 1 --reviewed-commit 12e06fc
bash tools/exp06_launch.sh probe --dry-run --gpu 1 --reviewed-commit 12e06fc
bash tools/exp06_launch.sh smoke --dry-run --gpu 0 --reviewed-commit 12e06fc
```

All returned zero. Real execution requires the full 40-character reviewed commit.

## Exact blocking list

**Before this `full_train` gate closes and before confirmatory training launches, fix findings 1–6:**

1. Training code approvals and launcher/finalizer source bindings.
2. Missing training geometry-input inventory.
3. Liveness validation of the receipt’s child PID.
4. Operational argument type/value validation.
5. Nonzero launcher outcome and aborted naming for failed diagnostics.
6. Required, validated diagnostic timing/memory evidence and its provenance contract.

**May follow in round 2b:** finding 7, the explicitly deferred omitted-T60 NaN-only nit, and nonblocking findings 8–9.

After fixes, the authorized operator must still complete the registered GPU ladder and timing acceptance criterion. This CPU review supplies no GPU fit, loss-parity, or runtime measurement.
