**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

# exp_06 `full_train` review 3

**Verdict: approve for merge and training launch.**

Reviewed `main...exp06-window` at **`a313253b7e210096104ed901f751a37d2eab2ea9`**, including the eleven commits in `f18383a..a313253`, the required briefing records, and the composed tooling. The latest changes resolve all four launch-blocking findings from review 2.

**Exact blocking list: empty.** The registered approval-fill commit, GPU ladder, timing criterion and peer handover remain execution prerequisites.

## 1. Disposition of review 2’s four findings

| Previous finding | Disposition | Independent evidence |
|---|---|---|
| **1. Registered smokes fail at startup** | **Resolved** | Parsed the launcher’s actual dry-run output and dispatched all three smoke children and the probe through `exp06_smoke._invoke` and the real entry points. All reached the model/CUDA boundary with dataset/model adapters. Full startup without `--approved` still refused. Diagnostic admission without approvals requires `--no-save`. |
| **2. Approvals are not committed bindings** | **Resolved** | Full preflight and full finalization both refused an external copy, an uncommitted in-repository copy, and modified working-tree bytes. For finalization, I also updated provenance to match the altered approval hash: the committed-blob check still refused. A matching committed approval file passed. |
| **3. Test WAVs and complete geometry membership are unbound** | **Resolved** | After provenance, changing a test WAV or held-out depth map caused finalizer status **2**, with no completion. A freshly hashed `splits=['train']` geometry record also refused. Regression replay covered duplicate entries, omissions, additions and wrong roots. |
| **4. Contradictory receipts are marked passed** | **Resolved** | All twelve requested receipt mutations refused, plus `outcome='ok'` with nonzero status. A valid success produced `passed: true`; a consistent failed diagnostic produced `passed: false`, always `admissible_arm: false`. |

Relevant implementation: [startup admission](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_train.py:111), [committed approval bytes](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_profiles.py:114), [data membership and rehashing](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:300), [receipt consistency](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:824).

### Assessment of the Coder’s choices

- **No approvals only with diagnostic `--no-save`: acceptable.** It restores registered startup without weakening full-run admission.
- **Over-budget `outcome: ok` refuses: correct.** The runner now converts retrospective time overruns into `aborted_alarm`, preserving valid failure receipts.
- **One-second end allowance: acceptable.** It accommodates the child-exit timestamp’s second-level precision. It does not relax the separate wall-time budget.
- **Committed-byte checks at preflight and finalization, without another at spawn: sufficient for the registered launcher.** Spawn still hashes the approval file; finalization verifies that hash and its reviewed blob. This does not certify direct trainer invocation as a substitute for preflight.
- **`passed` advisory to the shell: acceptable for this path.** The real runner exits nonzero on failure, and the launcher propagates that failure after recording completion. I independently checked this propagation.
- **Deferred `probe_align` registration: acceptable under A1**, provided it lands before the later probe executes.
- **Finalization rehashing: justified.** The extra read establishes input identity. The real geometry-plus-held-out inventory measurement took approximately **64 seconds** here; that is not a measurement of complete production finalization.

## 2. Answers to the six review questions

### (1) Do rounds 1 and 2a implement the training contract?

**Yes, with the nonblocking textual differences and deferred items below.**

- **§2.2:** The encoder appends the fixed azimuth channels after inherited gauge alignment, adds exactly **526,336 parameters**, and preserves the pinned factory routes. CPU tests cover the camera convention, scene-only symmetry breaking, joint heading cancellation, zero-shift exactness and RNG isolation.
- **§4:** All four real-room estimates reproduced **−90°, `k=128`**, with **12/12** stable leave-one-out decisions. Training rows alone entered the estimator.
- **§5:** Recipe, operational types, production constraints, three argument copies, twelve-epoch budget and checkpoint equality are enforced. The native `epoch_012.pth` contract composes successfully.
- **§6.4, training portion:** The eight training-critical approval keys, training/orchestration closures, data identities and external completion lifecycle are bound and revalidated.

Literal differences from the plan text:

1. Provenance fields occupy a **fifth schema class**, rather than the plan’s operational class.
2. Launch commands add execution/approval flags and exclusive attempt paths. Exp_06 diagnostic children now additionally receive `--run-type` and, when applicable, `--exploratory`.
3. The no-save probe does not spell out every full-run checkpoint option; its inherited checkpoint cadence is inert.
4. Later producer approvals, heading-consumer enforcement and HAA commands remain deferred under **§10a A1**.

None changes the registered pretraining recipe or blocks this training review.

### (2) Does the CPU composition work, including refusal paths?

**Yes, within the read-only adaptations described below.**

I exercised:

- Real heading estimation → four serialized JSON records → `read_heading_json(..., room_dir=...)`.
- Changed heading input bytes → readback refusal.
- Parsed full-launch arguments → `prepare_args` → fresh provenance → `exp06_recipe.check_all`.
- Tiny serialized `last.pth` and `epoch_012.pth`, twelve history rows and real artifact hashes → the actual finalizer CLI dispatch.

The synthetic full completion returned **0**, recorded **`admissible_arm: true`**, matched all **five artifact hashes**, and was idempotent.

Independent CLI mutations refused changed training WAVs, test WAVs, held-out geometry and source bytes; reduced geometry splits; missing provenance; a live receipt PID; stale log bytes; an eleven-epoch history; and boolean/integer argument drift. Refusals returned **2** without publishing completion.

The shell lifecycle drained surviving descendant output before continuing, preserved child status **7**, and returned **3** on a failing log sink before closing or certifying the run.

**Scope distinction:** HAA headings are not inputs to simulation pretraining, where heading is fixed at zero. The finalizer’s HAA heading-consumer gap remains deferred, as listed below.

### (3) Are the printed launch commands correct?

**Yes for the training scope, with the explicit differences above.**

Executed:

```bash
bash tools/exp06_launch.sh full  --dry-run --gpu 1 --reviewed-commit a313253b7e210096104ed901f751a37d2eab2ea9
bash tools/exp06_launch.sh probe --dry-run --gpu 1 --reviewed-commit a313253b7e210096104ed901f751a37d2eab2ea9
bash tools/exp06_launch.sh smoke --dry-run --gpu 0 --reviewed-commit a313253b7e210096104ed901f751a37d2eab2ea9
```

All returned **0**.

The environment printed the registered `XRIR_DATA_PATH`, `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=8` and selected GPU. Full printed the complete §5 recipe, including **`--epoch-ckpt-every 1 --save-every 500`**. Smoke retained its registered **`--save-every 0 --no-save`**, TF32-off budget. Probe printed 200 training batches, 32×2, TF32, no-save, `decay_epochs=3` and `log_interval=50`.

The pinned trainer’s child arguments remain unchanged. HAA smoke commands await their separately reviewed entry points.

### (4) Are pinned files unchanged and their checks green?

**Byte identity passes.** Direct comparisons against `git show main:<path>` matched **41 pinned source files plus both pinned test files**.

```text
git diff main...exp06-window --diff-filter=M --name-only
→ empty
```

Recomputed pins:

```text
train_xRIR_backbone — 11 files
5b2da2504e220b63bfc932cf0d83c143261e117edea1838e6907159c3b12eb13

eval_yaw_rotation — 12 files
5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48
```

Independent adapted testing passed **31 exp_03 record tests, with one skip**, and **15 provenance tests**. Seventeen write-dependent provenance cases were deselected; I do **not** claim an independent unmodified full provenance-suite run. The Coder reports that suite green within **679 passed / 4 skipped**.

### (5) Are earlier should-fixes still open?

**No training-launch should-fix remains open.** Fable S1/S2 and the earlier training-admission, operational-schema, liveness, diagnostic-propagation and checkpoint-dtype findings are resolved.

Fable S3’s consumer portion and the omitted-T60 nit remain deferred to round 2b. Their scope is described below.

### (6) Is merging safe for the peer’s live jobs?

**Yes for source compatibility.**

The branch adds **20 files**: two under `model/`, eight under `tools/`, and ten under `tests/`. It modifies no existing `main` file.

I independently recomputed the exp_04 evaluator/launcher, exp_05 evaluator, HAA trainer/evaluator and backbone-trainer closures. Their files match `main`; none imports an added exp_06 file. The merge therefore preserves the peer’s training and evaluation source identities.

## 3. Remaining numbered findings — none blocks training

1. **Should-fix, deferred to round 2b — HAA heading consumer enforcement.**  
   [tools/exp06_finalize.py:982](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:982) reads heading JSON without `room_dir` and does not enforce confirmatory heading admission. A changed cache or diagnostic heading can consequently pass this consumer. **Minimal fix:** bind the HAA root, revalidate heading inputs, and enforce approved confirmatory heading identity before HAA execution.

2. **Nit — Five schema classes versus four in §5.**  
   [tools/exp06_recipe.py:55](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_recipe.py:55) separates `exp06` provenance fields. Validation remains strict. **Minimal fix:** reconcile the plan’s classification wording or place those fields under operational metadata.

3. **Nit, deferred to round 3b — Missing `probe_align` approval key.**  
   [tools/exp06_profiles.py:45](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_profiles.py:45) and the template omit the §6.4 helper. **Minimal fix:** register and bind its closure before the mirror probe is admitted.

4. **Nit, deferred to round 2b — Omitted T60 accepts any nonfinite value.**  
   [tools/exp06_finalize.py:1152](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:1152) requires no finite observations, allowing infinity where the pinned writer emits NaN. **Minimal fix:** require the writer’s NaN-only representation for omitted T60.

## 4. Verification record and limits

All checks used the prescribed xRIR interpreter and CPU environment. No file was modified, no GPU work started, and no process was signalled.

| Independent check | Result |
|---|---:|
| Selected pytest cases, `--capture=sys -p no:cacheprovider` | **231 passed, 5 skipped, 479 deselected** |
| Finalizer/gate cases using retained fixtures and memory-only writes, including corrected CLI replay | **179 passed** |
| Pinned suites through memory adapters | **46 passed, 1 skipped, 17 deselected** |
| Printed smoke/probe child startup dispatches | **4 passed** |
| Real heading generation and readback | **4 rooms passed** |
| Synthetic full preparation → finalizer CLI | **Passed; hashes matched; idempotent** |
| Shell lifecycle and diagnostic propagation | **6 scenarios passed** |
| In-memory compilation | **18 Python files passed** |
| `bash -n tools/exp06_launch.sh` | **Passed** |
| `git diff --check main...exp06-window` | **Passed** |
| Final branch status | **Clean; HEAD unchanged** |

Counts overlap and must not be summed into a suite total.

The real inventories contained **312,321 geometry files**—302,671 metadata records and 9,650 depth maps—and **6,337 held-out WAVs**, with **zero missing files**.

Read-only adaptations redirected publication to memory, accommodated import caches, disabled Numba JIT, and replaced signal-zero liveness queries with `/proc` checks. Shell tests used anonymous pipes and storage adapters. Initial adapter failures were corrected; they were not product failures. Filesystem publication/rename atomicity and GPU forward, loss-parity, memory-fit and timing behavior were not independently established.

**Final verdict: approve for merge and training launch. Blocking findings the Coder must fix before this round closes: none.**
