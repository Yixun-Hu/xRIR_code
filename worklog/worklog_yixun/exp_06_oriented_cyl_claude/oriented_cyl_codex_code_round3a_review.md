**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

# Round 3a review — request changes

Reviewed `3df20b8..895f512` on `exp06-round2b`, including every changed source/test line, the required briefing, prior reviews, experiment records, and composed helpers. The numerical implementation is sound in the checks completed. Three input-binding and failure-cleanup findings block closure.

No files were modified, GPU work started, or processes signaled. The prohibited worktree was not accessed.

## Findings

### 1. **Blocker — checkpoint hashes do not bind the bytes used for inference**

**Location:** [exp06_mirror_probe.py:478](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_mirror_probe.py:478), [exp06_mirror_probe.py:501](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_mirror_probe.py:501).

`build_record` runs both probes **before** hashing checkpoints. It likewise hashes the heading JSON and captures the source closure after computation. There is no final input revalidation before publication.

Using the actual `build_record` body with in-memory input changes and stubbed inference produced:

| Change | Recorded outcome | Result |
|---|---|---|
| Checkpoint changes after loading, before hashing | `pass` | Recorded hash identifies different weights from those evaluated |
| Checkpoint changes immediately after hashing | `pass` | Recorded hash already disagrees with retained checkpoint bytes |

A heading replacement can similarly combine an earlier parsed `k` with a later JSON hash. Changes between legacy reproduction and the full gate are also undetected.

**Why it matters:** Subsequent approval of the recorded hashes cannot establish which inputs produced G1.

**Minimal fix:** Capture checkpoint, heading, cache, and source identities before either probe; use those identities throughout; revalidate immediately before publication and refuse changes. Bind loading/parsing to the captured bytes where practical. Add regressions for changes during inference and during record construction.

### 2. **Should-fix, mandatory — the panorama is absent from G1’s input bindings**

**Location:** [exp06_mirror_probe.py:503](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_mirror_probe.py:503).

The record copies the heading estimator’s four input hashes:

```text
meta.json, rirs.npy, speaker_xyz.npy, xyzs.npy
```

The probe additionally consumes **`depth.npy`**, through `HAADataset`, but neither hashes nor revalidates it. Heading verification correctly excludes depth because heading estimation does not consume it; that identity is insufficient for the mirror probe.

I verified that the heading-input verifier reads only those four files. The probe’s record-building hash calls add checkpoints and the heading JSON, but still omit depth.

**Why it matters:** Different panoramas can produce different geometry features and G1 statistics under identical declared cache identities.

**Minimal fix:** Add an exp_06 probe cache binding covering all five consumed files, including `depth.npy`, and revalidate it before publication. Keep the heading schema unchanged. Add a regression changing only depth while preserving the heading JSON and its four matching input hashes.

### 3. **Should-fix, mandatory — failed preparation can modify a root the attempt did not acquire**

**Location:** [exp06_haa_pipeline.sh:130](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_haa_pipeline.sh:130), [exp06_haa_pipeline.sh:143](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_haa_pipeline.sh:143).

When `open_job` fails, `prepare_failed` nevertheless creates/writes inside the requested root and attempts to rename it. It does not establish that this attempt created or owns the directory.

With filesystem-writing commands replaced by printing shell stubs, an `open_job` refusal against an existing directory still requested:

```text
write <root>/preparation_failure.json
mv <root> <root>_ABORTED_prepare_open_job
```

For example, failure to write an existing `launch.pid` does not imply that renaming its enclosing directory is impossible. A failed retry can also encounter a populated prior job.

**Why it matters:** Reporting a failed preparation can relocate another or previously completed job and invalidate its recorded paths.

**Minimal fix:** Track whether this attempt exclusively created/acquired the root. Only write its receipt and abort-rename it in that case. Record failures involving unowned existing roots separately. Add populated-root and failed-acquisition regressions; preserve nonzero propagation and zero child launches.

### 4. **Nit for round 3b — enforce confirmatory heading admission when wiring approvals**

**Location:** [exp06_mirror_probe.py:448](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_mirror_probe.py:448).

`_verified_heading` checks cache consistency and a usable decision, but accepts `admissibility: "diagnostic"`. The emitted heading binding omits that field.

**Why it matters:** Cache consistency alone does not satisfy the existing heading tool’s “diagnostic never binds a run” contract.

**Minimal fix:** During round-3b producer-admission integration, require confirmatory heading admission and the registered approval bindings before confirmatory G1 execution. This is nonblocking for the explicitly separated approvals round.

### 5. **Nit — three test commits exceed the requested size**

**Location:** [test_exp06_mirror_probe.py:1](/home/yixunhu/codespace/xRIR_code_wt2b/tests/test_exp06_mirror_probe.py:1), [test_exp06_bootstrap.py:1](/home/yixunhu/codespace/xRIR_code_wt2b/tests/test_exp06_bootstrap.py:1).

`792a16d`, `d6f19f6`, and `402a42b` change **225, 287, and 229 lines**, respectively. All are test-only, and the report discloses them.

**Minimal fix:** Split future test groups below 200 changed lines. Do not rewrite this history.

## Verification performed

All Python checks used the requested interpreter and environment:

```text
/home/yixunhu/miniconda3/envs/xRIR/bin/python
PYTHONPATH=/home/yixunhu/codespace/xRIR_code_wt2b
XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
CUDA_VISIBLE_DEVICES=''
OMP_NUM_THREADS=4
PYTHONDONTWRITEBYTECODE=1
NUMBA_CACHE_DIR=/tmp/exp06_codex_review_numba
```

### Repository and static checks

| Command/check | Result |
|---|---|
| `git diff 3df20b8..895f512` | Ten files; every changed line reviewed |
| `git diff --check 3df20b8..895f512` | Passed |
| `bash -n tools/exp06_haa_pipeline.sh` | Passed |
| In-memory `compile()` on changed Python files | **9 passed**, without bytecode writes |
| `git diff main...895f512 --diff-filter=MD --name-only` | Empty |
| Working-byte comparison against `main` | **90 pre-existing tracked Python/shell files identical**, covering the requested protected source |
| Enumerated exp_03 closure versus reviewed commit | All 12 files and their subsequent histories unchanged |
| Final HEAD/status | `895f512`, clean |

The exp_03 digest remains:

```text
5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48
```

Dynamic import-closure discovery was not successfully replayed because of the cache limitation below.

### Unmodified pytest replay

```text
python -m pytest tests/test_exp06_bootstrap.py -q -s -p no:cacheprovider
```

**53 passed in 12.57 seconds.** This includes all eleven canonical exp_02 cells: point estimates and both interval schemes reproduced with **maximum absolute difference 0.0**.

The affected six-module replay with normal capture failed before collection because no writable temporary directory exists. Retrying alignment, mirror-probe, and provenance tests with `-s` stopped at **two collection errors** from Numba’s cache initialization.

Consequently, **823 passed / 6 skipped remains the Coder’s reported total**, not my reproduced total.

### Additional read-only checks

These used inline `python -` harnesses with actual source definitions extracted through AST. Filesystem operations were represented in memory where necessary; these are distinct from an unmodified pytest replay.

- **Alignment:** 26 CPU test cases passed, including the independent NumPy reference, float32/float64, delay signs, complete clipping, gains, dtype, and input preservation. Arithmetic matches the pinned method line by line.
- **Composed forward:** prediction and target were **bit-exact** against the pinned numerical forward for tiny SimpleViT, cylindrical, and oriented encoders on CPU. A test subclass replaced only the pinned CUDA-only alignment allocation.
- **Time grid:** the composed forward preserves `times.unsqueeze(0).to(ref_locs.device)`.
- **Cohort:** both frozen 24-id cohorts exactly match the historical rule applied to the real hallway metadata/coordinates. Counts are **185 +y / 238 −y**. Actual y values contain ties; additional tied-y cases matched the diagnostic’s `argsort`/rank rule.
- **Frames:** mirror construction commuted bit-exactly with `k=0` and `k=128`. In-memory dataset checks produced finite mirror statistics and identical room-frame reference-side counts across frames.
- **G1:** **19 boundary cases passed**, covering every exact threshold and its immediately adjacent floating-point values.
- **Bootstrap adversarial inputs:** NaN measurements and mismatched lengths refused; global NumPy/Torch RNG states unchanged. Single-seed and one-cluster inputs behaved as generic estimators; a one-observation interval was rejected by convergence. Sparse query/seed layouts exercised zero-weight draws and `nanpercentile`.
- **Heading/cache:** training-only mmap estimation selected hallway `k=128`; the four input hashes verified. Applying that same record to the dampened-room cache refused.
- **Cycle-0 completion fix:** **8 in-memory cases passed**, including changed persisted hashes, missing/malformed records, nested type differences, and an unserialisable returned mapping. Nit 7 is resolved.
- **Publication:** inspected `write_manifest`’s exclusive `xb` creation; overwrite refusal is correctly implemented. Actual filesystem publication/rename tests could not run.

The real-checkpoint legacy neural reproduction was **not rerun** because its normal imports require writable caches. I checked its cohort, reference draw, arithmetic, checkpoint routing, and batch/device defaults against the diagnostic. The reported **0.9745/0.8510** and **0.4354/0.2382** remain Coder measurements.

Both CUDA tests are appropriately constructed: alignment asserts `torch.equal` against the untouched pinned method; forward parity compares both outputs at `1e-5` and exercises the time-grid transfer. Neither was executed here.

### Commit discipline and reported deviations

All **16 commits** contain both required trailers. Tests precede the relevant implementations; exact historical red-run totals were not independently replayed.

Disposition of the report’s numbered deviations:

- **1–3:** Accepted. `nanpercentile` preserves exp_02, adjusted-tail tolerance accounts for floating-point ordering, and zero-width refusal follows the plan.
- **4:** Accepted. Failed legacy reproduction should withhold G1; forcing `inconclusive` is consistent with its validity role.
- **5:** Injection hooks are acceptable for bounded synthetic testing.
- **6:** Size violations are finding 5.
- **7–8:** Authorized regression-test edits and import-only helper reuse are acceptable, including `_pick_refs` verification.
- **9:** The hardening changes are appropriate.
- **10:** CUDA nonexecution is correctly disclosed; parity still needs GPU verification before production G1.

## Verdict and exact blocking list

**Request changes. Round 3a may not close yet.**

The Coder must fix and reverify exactly:

1. **Finding 1:** Bind G1 to input identities captured before computation and reject changes before publication.
2. **Finding 2:** Hash-bind and revalidate the consumed panorama.
3. **Finding 3:** Prevent failure cleanup from writing into or renaming an unowned existing job root.

Findings **4–5** are nonblocking follow-ups for round 3b.
