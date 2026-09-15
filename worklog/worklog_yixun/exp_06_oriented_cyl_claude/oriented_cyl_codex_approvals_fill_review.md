**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

## Verdict: request changes

The approval digests, four heading records, merge identity, and protected files pass independent verification. One reproducible test failure prevents the required green-suite gate from closing at `53307b6`.

## Findings

1. **Should-fix — stale template-equality test fails after the authorized approvals fill.**  
   **Location:** [tests/test_exp06_profiles.py:78](/home/yixunhu/codespace/xRIR_code/tests/test_exp06_profiles.py:78)

   `test_the_record_copy_is_byte_identical_when_it_is_present` requires the committed approvals record to equal the null template byte-for-byte. Filling the approved digests necessarily breaks that assertion. At `53307b6`, it fails at byte 332 (`b'"' != b'n'`).

   **Why it matters:** The approved lifecycle now produces a failing exp_06 suite, preventing the plan’s green-suite launch gate from passing. Independent checks confirm that the populated approvals are valid.

   **Minimal fix:** Keep the null-template assertions separate. Update the record test to validate the populated record’s schema, committed-byte binding, and populated code digests while allowing the planned null sections. Preserve the template and approved values. Review the test change and replay the suite.

## Verification performed

### Approvals and commit history

- Read every changed line in `git diff 4d7a6cb..fc1ee2a` and `git diff fc1ee2a..53307b6`. History matches the notebook: `fc1ee2a` filled the approvals and introduced `fill_note`; `53307b6` removed that extra key.
- Recomputed **all 15 non-null code digests** using `compute_code_digests(repo, '53307b6')`.
- Independently recomputed them using an explicit module mapping, fresh `source_closure` calls, and `closure_record`; bound `launch_sh` and `haa_pipeline_sh` directly as files. Also independently hashed committed source blobs. **Every digest matched.**
- `load_approved_digests(path, repo, '53307b6')` successfully bound the committed bytes. Approval-file SHA-256:  
  `eba8ad3ac7713bb5deb1365fdb07d43a0ddc321bf8483dedafeff32c53bcbd9e`
- Using the actual signature, `require(approved, TRAINING_KEYS, repo=repo, commit='53307b6')` returned **`[]`**.
- Null code keys are exactly `bootstrap`, `compare`, `mirror_probe`, and `summarize_haa`; their modules are absent. All `reused` and `artifacts` leaves are null.
- Adversarial checks rejected binding the current approvals to either earlier commit and rejected the historical extra `fill_note`.

### Heading records

All four passed `read_heading_json(path, room_dir=<corresponding HAA cache>)`, including input SHA-256 revalidation.

Each records **estimated −90°, k 128, confirmatory admissibility**, HEAD `53307b6603b99a713122b80bb5c10beea3f3f031`, and no changes outside `worklog/`. Each recorded heading closure matches `code.heading`:

`69e395d0ba0bcfbe28d803553904c5fdd26bbda3dcd75ea700944802497102ae`

An independent numerical reconstruction agreed with all four decisions and all **48 leave-one-out decisions**. Maximum level discrepancy was approximately **6.17 × 10⁻⁷ dB**. All **16** simulated input-hash mismatches were rejected.

| Heading file under `ckpt/exp06/heading/` | File SHA-256 |
|---|---|
| `class_room.json` | `c37053c34d9503f6762b0129d98572315892a362f5025b6b218651838fbd562d` |
| `dampened_room.json` | `2eddf0470f80fc5011ff39faa899c4ebfef0931e2a69a47427b370fb2a75b1a1` |
| `hallway.json` | `53641c37d59165fec52ab5e307815f5e2298bcd6e789d4ed6f596714ab498ebf` |
| `complex_room.json` | `c34959101f37672b988c2aad35982e15c290125bd46c63ac7be22581ac5d8a40` |

### Merge and protected code

- `git diff 5091f6c 53307b6 -- . ':!worklog'`: **empty**.
- `git diff main...exp06-window --diff-filter=M --name-only`: **empty**.
- **44 protected paths** matched byte-for-byte across the working tree, HEAD, and pre-merge `ec7e649`.
- Closure digests remained identical at `ec7e649`, `5091f6c`, and `53307b6`:
  - exp_03, 12 files: `5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`
  - Original trainer, 11 files: `5b2da2504e220b63bfc932cf0d83c143261e117edea1838e6907159c3b12eb13`
- In-memory compilation of **27 added Python files**, `bash -n tools/exp06_launch.sh tools/exp06_haa_pipeline.sh`, and applicable `git diff --check` checks passed.

### Test replay and limits

- Targeted reproduction:
  ```text
  python -m pytest tests/test_exp06_profiles.py::test_the_record_copy_is_byte_identical_when_it_is_present -q --capture=sys -p no:cacheprovider
  ```
  **1 failed in 1.37s**, finding 1.

- Read-only selection across the eight exp_06 encoder/factory/heading/recipe/profiles/train/eval/haa test modules and the exp_03 closure-pinning test:  
  **219 passed, 1 failed, 3 skipped, 77 deselected in 85.23s.**  
  The sole failure was finding 1. Coverage included symmetry, boundary behavior, RNG isolation, bit-exactness, and fail-closed paths.

Tests requiring temporary filesystem writes were deselected; CUDA forwards were skipped. Cache accommodations were in memory, with Numba JIT disabled. The full suite and GPU ladder were not independently executed. No files were modified, no processes were signaled, and final HEAD remained `53307b6`, clean outside `worklog/`.

## Exact blocking list

**[1]** Fix the stale approvals/template equality test, obtain review of that change, and replay the required suite successfully before closing this launch gate.
