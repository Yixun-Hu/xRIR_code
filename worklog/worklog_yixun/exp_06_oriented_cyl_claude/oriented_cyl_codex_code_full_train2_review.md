**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

## Verdict: request changes

Reviewed **f18383a32c90985a197e5fb82a4584f6e1529794**, including `main...exp06-window` and the 23-commit fix cycle `12e06fc..f18383a`, against the SOP, approved v4 plan/A1, prior reviews, experiment records, Coder prompt/report, and composed tooling.

The fixes address the original reproductions substantially. **Four findings below still block closure of this merge-and-training gate.** The branch passes the separate pinned-file and peer-job compatibility checks.

No files were modified, GPU work started, or processes signaled. The round-2b worktree was not accessed.

## Findings

### 1. Blocker — the registered exp_06 training smokes now fail at startup

**Locations:** [tools/exp06_launch.sh:250](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_launch.sh:250), [tools/exp06_train.py:256](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_train.py:256).

The launcher supplies `--run-type smoke` and `--approved` to **the wrapper**, but neither reaches the exp_06 trainer child. Its parser therefore selects `run_type='full'`, and the new admission guard raises:

```text
ValueError: a full run must name the approvals it is admitted under (--approved)
```

I reproduced this by calling the actual trainer entry with each printed smoke child argv: both `simple` and `cylindrical_oriented` fail before dataset construction or CUDA use.

**Why it matters:** Plan §9’s trainer-parity and oriented smoke rungs cannot complete. The dry-run golden tests pass because they check strings without exercising this dispatch.

**Minimal fix:** Make the exp_06 child’s diagnostic mode agree with the wrapper, while preserving the pinned trainer’s shared numerical arguments. Add a CPU integration regression that dispatches the printed smoke command through the actual exp_06 startup path.

### 2. Should-fix, launch-blocking — approvals need not be committed at the reviewed revision

**Locations:** [tools/exp06_profiles.py:113](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_profiles.py:113), [tools/exp06_finalize.py:1455](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:1455).

`load_approved_digests` validates shape and hashes, but never establishes that the approval bytes belong to a reviewed commit. Preflight does not supply that guarantee: its cleanliness check expressly excludes `worklog/`, and `--approved` also accepts an external path.

**Reproduction:** Full preflight accepted the retained, filled approvals fixture outside the clean `f18383a` repository, returning:

```text
approval_deviations: []
dirty_outside_worklog: false
```

The GPU-availability query was stubbed empty; approval loading, digest computation and git checks were real. An uncommitted worklog-path variant also passed.

**Why it matters:** Plan §6.4 requires approval filling in a second reviewed commit before confirmatory execution. Hashing an arbitrary file at spawn establishes subsequent consistency, not that prerequisite.

**Minimal fix:** Bind confirmatory approvals to an explicit reviewed repository/commit and require their bytes to equal the committed blob. Verify that binding at preflight and finalization. External or uncommitted approvals may remain available for explicitly exploratory diagnostics.

### 3. Should-fix, launch-blocking — consumed-data coverage remains incomplete and partly record-controlled

**Locations:** [tools/exp06_train.py:193](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_train.py:193), [tools/exp06_finalize.py:309](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:309).

Two variants still produce an admissible full completion:

- **Changed test WAV:** Changing `single_channel_ir/Bathrooms/Bathrooms_idx_18/S000_R000_hybrid_IR.wav` passed finalization. The existing WAV inventory covers training files; the new inventory covers geometry. Test waveforms consumed by `test_epoch` remain unbound.
- **Reduced geometry split:** Replacing the geometry identity with a correctly hashed `splits=['train']` inventory, then changing the held-out depth map, also passed. The finalizer accepts a subset of `GEOMETRY_SPLITS` and derives expected membership from that declaration.

Both returned **`admissible_arm=True`**.

**Why it matters:** The training execution consumes both splits, including inputs producing the recorded epoch test losses and best-checkpoint decisions. Its provenance cannot establish complete input identity when those bytes are omitted or the record can reduce the required inventory.

**Minimal fix:** Inventory and revalidate the test waveforms through exp_06-owned code, and derive the required geometry splits independently from the full-run contract. Require both splits with no omissions or duplicates. Keep the pinned dataset and provenance helpers unchanged.

### 4. Should-fix, launch-blocking — contradictory diagnostic receipts are still marked passed

**Locations:** [tools/exp06_finalize.py:727](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:727), [tools/exp06_finalize.py:787](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_finalize.py:787).

The new checks require receipt fields, but do not sufficiently validate their relationships. With otherwise valid diagnostic provenance and a closed zero-status child log, each following mutation was accepted by the CLI with **exit 0 and `passed=True`**:

- `outcome='aborted_memory'` or `'failed'`;
- peak allocation **4 GiB** against a **3 GiB** budget;
- `wall_s=301` against `alarm_seconds=300`;
- zero alarm or memory budgets;
- `entry='invented'`;
- `argv=['--no-save', 17, {}]`;
- a changed entry/argv inconsistent with the startup command.

Receipt timestamps outside the child’s execution window were also accepted.

**Why it matters:** Plan §6.4 requires diagnostic timing/memory evidence, and §9 uses bounded diagnostics as launch prerequisites. Field presence plus two zero statuses does not establish that the named diagnostic passed its contract.

**Minimal fix:** Validate supported entry/module identity and string argv against startup provenance; require positive budgets; bind timing to the child execution; and derive `passed` from a consistent successful outcome, statuses and resource evidence. Preserve valid failure receipts with `passed=False`.

## Disposition of full_train review 1

| Previous finding | Status | Evidence |
|---|---|---|
| **1 — approvals/orchestration** | **Partial** | Launcher/finalizer byte edits now refuse; null/missing approvals and digest drift refuse. Committed approval identity remains missing: finding 2 above. |
| **2 — geometry inputs** | **Partial** | Original metadata/depth edits now refuse; real geometry inventory verified. Remaining coverage holes: finding 3. |
| **3 — live receipt child PID** | **Resolved** | Live receipt PID refuses without a sidecar; disagreeing sidecar refuses; agreeing dead PID passes. |
| **4 — operational schema** | **Resolved** | All four original corruptions refuse even when all three argument copies agree: cadence `2`, cadence `True`, `save_every=None`, workers `"twelve"`. Additional type/boundary cases pass their rejection tests. |
| **5 — failed diagnostic launch status** | **Resolved** | Failed child status propagates and abort naming is correct; finalizer refusal exits 2. Finding 1 above is a separate new startup regression. |
| **6 — minimal diagnostic receipt** | **Partial** | Original minimal receipt and missing-field variants refuse. Contradictory complete receipts still pass: finding 4. |
| **8 — checkpoint dtype/shape** | **Resolved** | Both are checked before `torch.equal`; float64 and reshaped variants refuse. |
| **9 — probe flags** | **Resolved** | Dry-run prints `--decay-epochs 3 --log-interval 50`. |

Fable S1/S2 and the S3 input-verification helper are implemented. The **consumer-side** heading readback requirement remains open in `_heading_binding`: it calls `read_heading_json` without `room_dir`. This is the already-deferred finding 7, belongs to round 2b, and does not independently block AcousticRooms pretraining. The deferred omitted-T60 issue likewise remains outside this gate.

## Answers to the six integration questions

### 1. Implementation against the plan

**§2.2:** Implemented for the reviewed CPU contract: parent gauge alignment, two azimuth channels, inherited transformer structure, exact **526,336** parameter increase, active-yaw discrimination, joint heading cancellation and identity checks. Pinned factory routes preserve class identity and seeded state dictionaries. CUDA forward-parity tests remain skipped.

**§4:** The heading estimator and verification helper implement the frozen training-only rule. All four real rooms select **−90°, `k=128`**, with all twelve leave-one-out winners equal to `−y`.

**§5 and training portions of §6.4:** The positive composition works, but findings 1–4 prevent full compliance.

Other plan-text differences and Coder choices:

- Provenance fields form a **fifth schema class**, although §5 places them under operational fields. This is a documented, nonblocking organizational deviation; align the plan wording.
- `probe_align` is absent from the approval template despite §6.4 listing it. Acceptable deferral for this training-only gate; add it before its later consumer is approved.
- Binding shell files separately and binding the finalizer’s complete import closure covers the required training orchestration bytes.
- Enforcing admission in `main()` is acceptable in principle; the diagnostic composition regression must be fixed.
- Printing geometry rehash duration preserves completion idempotence and is acceptable.
- The finalizer’s trainer import and process-local closure cache introduce no demonstrated additional training blocker. The cache is a stat-based optimization; fresh-process verification remains important.

### 2. CPU end-to-end composition and rejection paths

**Positive path verified.** I generated and round-tripped all four real heading records in memory, checked input-aware readback, then independently composed:

```text
parse_args → provenance_fields → prepare_args → check_all
→ tiny history/last.pth/epoch_012.pth → finalize --run-type full
```

The finalizer returned **exit 0**, `admissible_arm=True`, and artifact hashes matched independently recomputed hashes. Headings are a separate HAA preparation path; AcousticRooms pretraining uses the registered zero heading.

The original source edits, metadata/depth edits, training-WAV edit, truncated budget, bool/int drift, missing provenance, live PID and stale-log paths reject. Failed-child and failed-sink control flow also rejects. **The additional accepted variants in findings 2–4 remain contract failures.**

### 3. Exact launch commands

Dry-runs for `full`, `probe` and `smoke` all exited 0 and printed:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=8
XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
```

- **Full:** Registered numerical recipe, `--epoch-ckpt-every 1`, `--save-every 500`, exclusive attempt path, plus admission flags.
- **Probe:** Registered bounded recipe, including the repaired decay/log flags.
- **Smoke:** Printed numerical child arguments match §9, including `--save-every 0 --no-save`; their exp_06 dispatch nevertheless fails as finding 1 describes.

### 4. Pinned files

**Byte identity verified.**

```text
git diff main...exp06-window --diff-filter=M --name-only
```

returned nothing. Direct comparison against `git show main:<path>` found **zero mismatches across 41 pinned/composed files**, including all twelve exp_03 files, the trainer closure, `sim_to_real/*.py`, exp_04/05 tooling and the named shared producers.

Recomputed digests include:

- Trainer: `5b2da2504e220b63bfc932cf0d83c143261e117edea1838e6907159c3b12eb13`
- Exp_03 evaluator: `5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`

Test execution qualifications are below; I do not claim an unrestricted full-suite rerun.

### 5. Remaining previous should-fixes

Previous findings **1, 2 and 6 remain partially open** through the concrete variants above. Findings **3, 4, 5, 8 and 9 are resolved**. Deferred heading-consumer work remains a round-2b requirement.

### 6. Peer-job merge safety

**Pass for source compatibility.** The branch adds exactly **20 files**, all under `model/`, `tools/` and `tests/`. Existing exp_04/05 training and evaluation closure bytes remain unchanged. No shared registry or existing import path is modified.

This compatibility result does not override the training-gate findings.

## Verification record

All Python checks used the requested interpreter/environment with CUDA hidden and bytecode disabled.

| Check | Result |
|---|---|
| Both requested git diffs; `git diff --check` | Inspected; whitespace check passed |
| Fix-cycle commits/trailers | 23 commits; maximum 189 changed lines; required trailers present |
| In-memory compilation; `bash -n` | 18 Python files passed; launcher passed |
| Broad read-only pytest subset, `--capture=sys -p no:cacheprovider` | **231 passed, 5 skipped, 439 deselected** |
| Training finalizer tests replayed with memory-only writes | **107 passed** |
| New finalizer gate tests, same accommodation | **46 passed** |
| Exp_03 record tests, memory-only filesystem/CLI accommodation | **31 passed, 1 skipped** |
| Additional read-only pytest over the two pinned test files | **24 passed, 1 skipped, 39 deselected**; overlaps earlier counts |
| Exp_03 pinned-closure test | Passed |
| Launcher diagnostic control flow | Failure exits 3; finalizer refusal exits 2; success continues |
| Anonymous-pipe lifecycle checks | Descendant output drained; child status retained; injected sink failure exits 3 before closing/publishing |
| Approval template copies | Byte-identical; SHA-256 `0ec10825bce5f9156537b5eb42934ab5d987b218d439f15e94e971501f3e018e` |

Independent real geometry hashing found **302,671 metadata JSONs + 9,650 depth maps**, **zero missing files**, **10,178,959,699 bytes**, in **60.9 seconds**. Digest:

```text
aae056d749366e9fa0eb437a4f1a7060e97fee87ad01e2f5ceada6eef67b1e17
```

**Execution limits:** Ordinary pytest capture initially failed because no writable temporary directory was available. I used memory-backed adaptations for write-requiring checks, disabled JIT cache writes, and replaced signal-based liveness checks with `/proc` inspection. These checks do not independently certify physical filesystem atomicity, actual disk-full behavior, or GPU watchdog operation. The Coder reports the unrestricted **639 passed/4 skipped** run; that is separate evidence.

## Exact blocking list

Before this round closes and merge/training launch is approved, the Coder must fix:

1. **Exp_06 smoke child mode/admission dispatch.**
2. **Committed reviewed approval-byte binding.**
3. **Complete consumed-data coverage, including test WAVs and mandatory geometry splits.**
4. **Diagnostic receipt consistency and truthful pass/fail determination.**

The later approval-fill review and registered GPU smoke/probe ladder remain required. Deferred round-2b/3 work is not added to this blocking list.
