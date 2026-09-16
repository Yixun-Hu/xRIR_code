**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-16.

**Verdict: request changes.**

Reviewed all 31 lines of the untracked Planner runbook against main `b9f6ebe6c853471b4d31ea6406fcda2eaedf17c2`, the SOP, approved plan, notebook, prior reviews, and actual launcher/wrapper/finalizer contracts. Runbook SHA256: `c02b7520a70c44b1a1b8318ad0e43d04b1c186349e74be1fed0672be61cdf703`.

1. **Blocker — abbreviated commit makes every launcher mode refuse.**  
   [launch_day_runbook.sh:6](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/planner_probes/launch_day_runbook.sh:6)

   `SHA=$(git rev-parse --short HEAD)` passes seven characters to preflight, which requires exact equality with the full HEAD at [exp06_finalize.py:1704](/home/yixunhu/codespace/xRIR_code/tools/exp06_finalize.py:1704). I reproduced exit **2** for `smoke`, `probe`, and `full`:
   ```
   HEAD b9f6ebe6c853471b4d31ea6406fcda2eaedf17c2 is not the reviewed commit 'b9f6ebe'
   ```
   Dry-runs return zero because they only print preflight.

   **Minimal fix:** use `SHA=$(git rev-parse HEAD)` and retain that full SHA in launch evidence.

2. **Blocker — HAA smokes cannot produce the required diagnostic completions.**  
   [launch_day_runbook.sh:23](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/planner_probes/launch_day_runbook.sh:23), also line 25.

   These commands write runner receipts and provenance, but arrange no child PID evidence, closed-log marker, `child_exit.json`, or finalizer invocation. Consequently, neither produces the notebook’s required `completion.json` with `passed: true`.

   Adding a finalizer call alone fails: [exp06_finalize.py:794](/home/yixunhu/codespace/xRIR_code/tools/exp06_finalize.py:794) requires `--no-save` in every smoke/probe receipt. Both actual HAA argv are rejected by this guard. Adding `--no-save` instead makes both HAA parsers exit **2**, and fine-tuning must retain `best.pth` for the subsequent evaluation.

   **Minimal fix:** provide a reviewed HAA diagnostic completion path that permits the registered disposable artifacts under `_smoke`, preserves `diagnostic: true` and `admissible_arm: false`, and performs the child/log/receipt lifecycle. Use exclusive attempt directories and finalize fine-tuning before evaluating its checkpoint. Any source changes require updated reviewed digests before launch. This cannot be corrected solely by adding an existing flag.

3. **Blocker — failures do not reliably stop the step.**  
   [launch_day_runbook.sh:3](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/planner_probes/launch_day_runbook.sh:3), lines 11–17 and 23–25.

   `pipefail` determines pipeline status but does not stop execution without an explicit check or `errexit`. The dirty-tree branch prints `STOP` successfully; failed approval Python is followed by successful GPU queries; GPU occupancy is displayed without being rejected. A failed fine-tune likewise proceeds to evaluation, potentially using `best.pth` saved before the failure.

   Four isolated shell reproductions all exited **0**: dirty tree, failed approval check, busy GPU, and failed fine-tuning followed by successful evaluation.

   **Minimal fix:** enable `set -euo pipefail`, explicitly reject dirty state and nonempty GPU compute-process output, and require successful fine-tuning **and its completion** before evaluation. Merely adding `-e` does not make the printed dirty/busy conditions failures.

4. **Should-fix, required before full launch — the supervising launcher is not detached.**  
   [launch_day_runbook.sh:28](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/planner_probes/launch_day_runbook.sh:28)

   The comment says “detached by the launcher,” but the runbook invokes it in the foreground. The launcher detaches the Python child, then waits for both child and log sink at [exp06_launch.sh:119](/home/yixunhu/codespace/xRIR_code/tools/exp06_launch.sh:119). Its supervising shell and sink still need to survive the approximately 31-hour run to close evidence, finalize, and promote it.

   **Minimal fix:** detach the entire launcher with `nohup setsid`, redirect its output to a timestamped record log, disconnect stdin, and record/check its PID and successful startup.

5. **Nit — logging does not exactly follow the SOP convention.**  
   [launch_day_runbook.sh:10](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/planner_probes/launch_day_runbook.sh:10), lines 23 and 25.

   HAA logs reach the correct record folder, but use compact UTC timestamps instead of the SOP’s local `YYYY-MM-DD_HH:MM:SS` convention. `gate` has no retained log within this script. Delegated launcher logs also use compact UTC names.

   **Minimal fix:** retain gate output and use SOP-formatted launch logs, documenting any retained inherited naming convention without renaming already hash-bound logs.

**What verified successfully**

- Every supplied option exists and is spelled correctly. Both HAA parsers accept the complete supplied argv.
- `--heading-json-dir` is present; omitting it for the oriented backbone refuses. `--job-spec` is optional; `--seed 0` parses as an integer. Default `--root` resolves here to `/home/yixunhu/data_cache/HAA_xrir`.
- Fixture naming agrees exactly: `ckpt/exp06/_smoke/fixture_cylor.pth`. The evaluation checkpoint matches the fine-tune save directory.
- The gate’s Python API usage is correct: `load_approved_digests(...)` returns `(approved, identity)`, and `require(approved, TRAINING_KEYS)` is valid. Actual output: **mismatches `[]`, require `[]`**. All **20** approved code digests match HEAD.
- All four heading JSON hashes match approvals; records remain confirmatory with `phi_deg=-90`, `k=128`, and the approved heading closure.
- All five GPU smoke commands specify **300 seconds / 3 GiB**. The runner has an execution watchdog and retrospective checks. The probe intentionally uses **2400 seconds / 46 GiB**.
- Explicit experiment output paths stay within `_smoke`, the pretraining root, and the record folder. One conditional external write exists in composed code: `exp06_train.data_identity()` uses a temporary directory if the shared inventory is absent. That inventory currently exists. A literal restriction covering temporary files and dependency caches needs those locations configured too.

**Verification commands and results**

- `bash -n <runbook> tools/exp06_launch.sh`: passed.
- `git diff --check`: passed.
- `git diff --no-index /dev/null <runbook>`: inspected every added line; exit 1 correctly indicates differences.
- `git diff main...exp06-window --diff-filter=M --name-only`: empty.
- Launcher dry-runs for `smoke`, `probe`, and `full`, with `--gpu 1 --reviewed-commit b9f6ebe --dry-run`: all exited 0; printed argv inspected.
- Actual preflight with that short SHA: **3/3 refused**, exit 2.
- Smoke and both HAA entry points’ `main(["--help"])`: passed with the cache accommodation below.
- Both HAA diagnostic receipts: reproduced `--no-save` refusal; both parsers reject adding that flag.
- Shell failure-injection checks: **4/4 reproduced masked failures**.
- In-memory budget boundaries and missing-heading refusals: **17 passed**, without arming alarms.
- Read-only pytest selection from `test_exp06_{profiles,recipe,train,haa,launch}.py`, the unknown-smoke-entry test, and the exp_03 provenance-pin test, using `-q --capture=sys -p no:cacheprovider --tb=short`: **157 passed, 67 deselected**, 70.79 seconds. Deselected cases require writable fixtures; selected launcher tests cover dry-runs and CPU startup dispatch.
- **41 protected source paths** match working bytes, HEAD, main, and pre-experiment `77ef292`. Closure digests remain:
  - exp_03, 12 files: `5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`
  - Original trainer, 11 files: `5b2da2504e220b63bfc932cf0d83c143261e117edea1838e6907159c3b12eb13`

The initial unwrapped help command encountered read-only Numba caching restrictions. Subsequent Python checks disabled Numba JIT and used an in-memory Matplotlib cache-directory accommodation. No files were modified, RIR arrays loaded, processes signalled, or GPU work started.

**Acceptance checks after correction**

| Step | Evidence required before advancing |
|---|---|
| `gate` | Full reviewed SHA equals HEAD; clean outside worklog; committed approvals match; no training-key deviations; GPU 1 has no compute processes after hand-over. Retain the gate log. |
| `smoke` | Three completions: `run_type="smoke"`, `diagnostic=true`, `admissible_arm=false`, `child_exit=0`, `passed=true`, `approval_deviations=[]`. Receipts have `outcome="ok"`, `exit_status=0`, `wall_s≤300`, `peak_bytes≤3×2³⁰`. Simple-backbone printed losses agree; oriented training reaches an optimizer step; fixture and sidecar exist. |
| `haa-smoke` | After finding 2 is resolved, both diagnostic completions pass the same resource/status checks. Fine-tuning completes two epochs with finite train/validation losses and the expected checkpoint; evaluation uses that successful attempt’s `best.pth`, records exactly four hallway samples and the expected heading. |
| `probe` | Passing diagnostic completion, 200 training micro-batches at batch 32 × accumulation 2, TF32 enabled, no checkpoints, finite losses, peak within 46 GiB. Separately record the timing extrapolation: **≤179.4 min/epoch** (`1.15×156`). Completion alone does not certify that projection. |
| `full` | At startup: durable supervisor, correct SHA/GPU/recipe and provenance, then an optimizer step without OOM/NaN. Epoch 1 meets the timing criterion. At completion: `run_type="full"`, `child_exit=0`, `diagnostic=false`, `admissible_arm=true`; complete twelve-epoch evidence and checkpoint hashes, exact `epoch_012.pth == last.pth["model"]`, and correct `final` promotion. |

**Exact blocking fix list before this runbook round closes: findings 1, 2, 3, and 4.** Finding 5 is nonblocking bookkeeping.
