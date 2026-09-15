**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

# Round 3a close review

**Verdict: approve with changes. Round 3a may close.**

Reviewed all changed lines in `895f512..7f29877` on `exp06-round2b`, against the approved plan, prior reviews, notebook, Coder prompt/report, and composed tooling. No files were modified, GPU work started, or processes signaled. The prohibited worktree was not accessed.

## Findings 1–4: disposition

### 1. **Resolved — original blocker: identities captured after inference**

**Location:** [exp06_mirror_probe.py:503](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_mirror_probe.py:503), [exp06_mirror_probe.py:568](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_mirror_probe.py:568).

Checkpoint and heading hashes now identify captured bytes. Checkpoints load through `BytesIO`; heading validation parses the captured JSON. Both legacy reproduction and every full-gate arm receive captured state dictionaries. Missing captured states raise rather than silently reopening checkpoints.

The final recheck covers checkpoints, heading, all five cache files, captured source files, and git identity. Changed or deleted inputs produce `SystemExit('refusing: …')` before publication.

Verified with in-memory filesystem adapters:

- Checkpoint changed after loading: refused.
- Checkpoint changed after initial hashing: refused.
- Checkpoint deleted after its captured bytes were loaded: refused.
- Valid replacement heading between legacy reproduction and full gate: original `k=128` retained; publication refused.
- Captured source-file change or HEAD change: refused.
- Both legacy models and all four gate arms load successfully from captured bytes after backing checkpoint files become invalid.

This resolves the requested capture/load/recheck contract. Finding 5 below addresses an avoidable interval after the final check.

### 2. **Resolved — mandatory: panorama unbound**

**Location:** [exp06_mirror_probe.py:480](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_mirror_probe.py:480).

`PROBE_CACHE_FILES` includes:

```text
depth.npy, meta.json, rirs.npy, speaker_xyz.npy, xyzs.npy
```

Their initial hashes are recorded and revalidated. Heading verification compares its four hashes against this same capture; the heading schema remains unchanged.

Changing only depth while the heading JSON still verifies against its four inputs refused publication. Changing depth during actual tiny-model inference also refused. Deleting each of the five cache files after capture refused.

### 3. **Resolved — mandatory: preparation cleanup touches an unowned root**

**Location:** [exp06_haa_pipeline.sh:129](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_haa_pipeline.sh:129), [exp06_haa_pipeline.sh:212](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_haa_pipeline.sh:212).

`OWNED_ROOT` resets before preparation and becomes true only after exclusive directory creation. Failure cleanup writes inside and renames only a root created by this attempt. Otherwise, its receipt goes beside the root.

Shell checks with filesystem effects represented in memory confirmed:

| Case | Result |
|---|---|
| Populated existing root; `open_job` refuses | Adjacent receipt; no root write or rename |
| Existing root adopted; `job_spec` fails | Adjacent receipt; no rename |
| Existing root; `launch.pid` acquisition fails | Adjacent receipt; no rename |
| Newly created root; preparation fails | Internal receipt and abort rename |

All cases propagate failure and launch zero children.

Adoption is reasonable for the existing resume behavior. The report’s “unowned root is never written into” wording is too broad: successful adoption refreshes `launch.pid`, and preparation can write the job specification. The implemented protection concerns **failure cleanup**.

### 4. **Resolved — diagnostic heading admission**

**Location:** [exp06_mirror_probe.py:459](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_mirror_probe.py:459).

A structurally valid diagnostic heading now refuses before either probe runs. Published heading bindings include `admissibility: "confirmatory"`.

The private validator import is appropriate here: it reuses existing validation while parsing captured bytes and respects the fix cycle’s file restrictions.

## Nonblocking nits for round 3b

### 5. **Nit — collect environment information before final revalidation**

**Location:** [exp06_mirror_probe.py:595](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_mirror_probe.py:595).

`environment()` runs after `revalidate_inputs`; output-directory preparation follows in `main`. Environment discovery includes imports and subprocess work.

An injected checkpoint change during that environment call produced exit 0 and a published record whose checkpoint hash no longer matched the retained file. The hash still correctly identified the captured weights evaluated, so this does not restore the original misidentification bug, but it unnecessarily widens the normal check-to-write race.

**Minimal fix:** Collect environment information and prepare the output parent first; assemble the record, then revalidate immediately before `write_manifest`.

### 6. **Nit — document refusal exit status accurately**

**Location:** [exp06_mirror_probe.py:620](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_mirror_probe.py:620).

The docstring says input errors exit 2, while handled input refusals use textual `SystemExit`, which exits 1.

**Minimal fix:** Document **1 for input refusals, 2 for argparse usage errors**. The chosen nonzero refusal behavior is acceptable.

## Verification

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

| Command/check | Result |
|---|---|
| `git diff 895f512..7f29877` | All four files reviewed |
| `git diff --check 895f512..7f29877` | Passed |
| `bash -n tools/exp06_haa_pipeline.sh` | Passed |
| In-memory `compile()` of changed Python files | 3 passed; no bytecode writes |
| `git diff main...7f29877 --diff-filter=MD --name-only` | Empty |
| Working bytes versus `main` | All 90 pre-existing Python/shell files, including tests, identical |
| Enumerated exp_03 closure versus reviewed commit | All 12 files and subsequent histories unchanged |
| Final HEAD/status | `7f29877bb9fb240d41c7ba725a0ba56fd9067fec`, clean |

Exp_03 digest remains:

```text
5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48
```

### Test replay

```text
python -m pytest tests/test_exp06_mirror_probe.py tests/test_exp06_haa_pipeline.py \
  -q -s -p no:cacheprovider
```

Stopped at **two collection errors** because Numba could not initialize its cache. A subsequent import diagnostic also encountered Matplotlib’s writable-cache requirement.

```text
python -m pytest tests/test_exp06_bootstrap.py -q -s -p no:cacheprovider
```

**53 passed in 12.85 seconds**, including the eleven canonical exp_02 bootstrap cells.

Additional inline, read-only harnesses verified:

- **9 new mirror-probe test cases passed**, using actual test/function bodies with in-memory filesystem operations and import adapters.
- **12 additional mutation/deletion cases refused**, including the requested checkpoint, heading, cache, and source variants.
- **6 preparation-failure cases passed** through actual Bash orchestration functions with I/O shims.
- **14 original pipeline syntax/dry-run/refusal cases passed.**
- Captured state dictionaries matched the original tensors **bit-exactly**; capture/load/revalidation preserved global NumPy and Torch RNG states.
- **2 frame/mirror commutation cases passed.**

These adapted checks are distinct from an unmodified filesystem test replay. **834 passed / 6 skipped remains the Coder’s reported full-suite result.** Dynamic hermetic closure discovery, actual filesystem publication/renaming, real-checkpoint legacy reproduction, and CUDA parity were not independently replayed successfully here.

## Integration and implementation choices

- Rehashing captured closure files plus git identity is acceptable for this fix’s source-stability check. Re-running hermetic discovery is unnecessary to detect ordinary changes to the captured implementation; approval enforcement remains round 3b work.
- Gate-record schema **2** and preparation-failure schema **2** appropriately distinguish the added bindings and nullable `aborted_dir`.
- No new round-3a test depends on a live child receipt PID. The pre-existing [pipeline fixture helper:407](/home/yixunhu/codespace/xRIR_code_wt2b/tests/test_exp06_haa_pipeline.py:407) still records `os.getpid()` and requires the planned integration fix. Genuine live-PID refusal tests should retain live markers.

## Commit hygiene

All five commits carry both required trailers. Changed-line counts, including tests, are **130 / 151 / 51 / 41 / 19**. Implementation commits follow their red-test commits. The final test-only guard landed green and is accurately disclosed; historical red-run counts were not independently replayed.

**Exact blocking findings: none. Round 3a may close.** Nits **5–6** may be carried into round 3b.
