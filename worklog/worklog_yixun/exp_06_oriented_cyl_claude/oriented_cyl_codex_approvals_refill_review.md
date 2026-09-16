**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-16.

**Verdict: approve for smokes/probe/launch at b9f6ebe.**

Reviewed the approvals refill in `b9f6ebe6c853471b4d31ea6406fcda2eaedf17c2`, using the requested SOP, plan, prior reviews, reports/prompts, experiment records, and composed tooling. Read every changed line of the refill. No blocker or should-fix finding.

1. **Nit — the briefing’s explanation of `eval_launch` digest drift needs correction.** At [tools/exp06_eval_launch.py:180](/home/yixunhu/codespace/xRIR_code/tools/exp06_eval_launch.py:180), the merged round adds `_persisted()` and checks the retained completion record. This file **did change** relative to `53307b6`; `exp06_profiles.py` is **not** in its import closure. The changed `code.eval_launch` approval correctly reflects that source edit. Misattributing it would make the review’s provenance explanation inaccurate. **Minimal fix:** use this corrected explanation in the record; no code or approval-value change is required.

**Approval binding and digest verification**

- The filled JSON has exactly the merged template’s schema and keys, including `code.probe_align`.
- `load_approved_digests(path, repo, HEAD)` succeeds with committed-byte binding. The approvals file’s SHA256 is:
  `8eee050f89f218cbffdb0d0f84717d9d0633da1d6a3365b8d150845bc6120368`.
- All **20 code digests** match both `compute_code_digests()` and a separate recomputation using fresh `source_closure()` calls followed by `closure_record()`. Shell entries use their explicit file lists.
- Across those closures, all **47 unique source files** match their HEAD blobs.
- `require(approved, TRAINING_KEYS, repo, HEAD)` returns **`[]`**.
- Every `reused` leaf, every `artifacts.epoch_012` leaf, and `artifacts.gate_g1` remain **null**, as planned.

For every previously filled code key, recomputing its current closure membership against `53307b6` reproduced the old approved digest. The changed members explain the differences completely:

| Approval key | Sole changed closure member since `53307b6` |
|---|---|
| `trainer` | `tools/exp06_profiles.py` |
| `finalize` | `tools/exp06_profiles.py` |
| `smoke` | `tools/exp06_profiles.py` |
| `profiles` | `tools/exp06_profiles.py` |
| `eval_launch` | `tools/exp06_eval_launch.py` |
| `haa_pipeline_sh` | `tools/exp06_haa_pipeline.sh` |

Thus the unchanged trainer, finalizer, and smoke entry files acquire new digests solely through the profiles import. No additional changed member or unexplained digest difference was found.

**Heading admission**

All four JSON files match their approved artifact hashes:

| Room | SHA256 |
|---|---|
| `class_room` | `c37053c34d9503f6762b0129d98572315892a362f5025b6b218651838fbd562d` |
| `dampened_room` | `2eddf0470f80fc5011ff39faa899c4ebfef0931e2a69a47427b370fb2a75b1a1` |
| `hallway` | `53641c37d59165fec52ab5e307815f5e2298bcd6e789d4ed6f596714ab498ebf` |
| `complex_room` | `c34959101f37672b988c2aad35982e15c290125bd46c63ac7be22581ac5d8a40` |

Each passes `read_heading_json(path, room_dir)` against the actual cache, including input-file hash verification, and retains `admissibility="confirmatory"`, `decision="estimated"`, `phi_deg=-90`, and `k=128`.

Their producing commit is `53307b6`, but the heading closure is unchanged. Its six members are:

```text
sim_to_real/haa_dataset.py
tools/exp06_heading.py
tools/provenance.py
tools/yaw_rotation.py
treble_multi_room_dataset/treble_xRIR_dataset.py
utils/spec_utils.py
```

The recorded working-tree digest, recorded HEAD digest, current recomputation, and approved `code.heading` all equal:

`69e395d0ba0bcfbe28d803553904c5fdd26bbda3dcd75ea700944802497102ae`

The finalizer and probe require valid, confirmatory, cache-consistent heading records; they do not require the producing commit to equal the launch commit. I also exercised the actual HAA loader, finalizer heading-binding function, and probe heading consumer: **all four records passed each consumer**. **Regeneration is unnecessary.**

**Merge isolation and pins**

`git diff 21a2bf6^1 21a2bf6 --name-status` shows **12 additions and 8 modifications**, all exp_06-owned. The modified files are exactly:

```text
tests/test_exp06_eval_launch.py
tests/test_exp06_finalize.py
tests/test_exp06_haa.py
tests/test_exp06_haa_pipeline.py
tools/exp06_approved_digests_template.json
tools/exp06_eval_launch.py
tools/exp06_haa_pipeline.sh
tools/exp06_profiles.py
```

The additions are the six new exp_06 modules and their six test files. No other experiment’s closure file changed. Independently checked **44 protected paths** against working bytes, HEAD, the merge’s first parent, and `53307b6`; all matched.

The recomputed pinned closures remain:

- exp_03, 12 files: `5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`
- Original trainer, 11 files: `5b2da2504e220b63bfc932cf0d83c143261e117edea1838e6907159c3b12eb13`

Both also match `77ef292`.

**Commands and tests verified**

- `git diff b9f6ebe^ b9f6ebe -- <approved_digests.json>`: reviewed all changes; only the approvals asset changed.
- `git diff main...exp06-window --diff-filter=M --name-only`: empty, consistent with the branch already being integrated; the first-parent merge comparison above establishes the actual modified-file list.
- `git diff --check` for both the refill and merge: passed.
- `bash -n tools/exp06_launch.sh tools/exp06_haa_pipeline.sh`: passed.
- Python verification through `load_approved_digests`, `compute_code_digests`, `source_closure`, `closure_record`, `require`, and the heading consumers: all assertions passed.
- **66 additional in-memory rejection checks passed:** malformed approval schemas, altered committed bytes, wrong reviewed commit, null/drifting/absent training approvals, mismatched heading angle/roll, changed cache identities, and diagnostic headings.

Ran pytest through a read-only selection plugin with `-q --capture=sys -p no:cacheprovider --tb=short`, covering `test_exp06_{profiles,encoder,factory,heading,recipe,train,eval,haa}.py` and the exp_03 provenance-pin test:

**222 passed, 3 skipped, 78 deselected; 88.07 seconds.**

The three skips require CUDA. The 78 deselected cases require writable temporary fixtures. To accommodate the strict sandbox, Numba JIT was disabled and an in-memory import shim allowed Matplotlib to read its existing cache directories. No repository code was changed.

Executed all three launcher dry-runs:

```bash
bash tools/exp06_launch.sh smoke --gpu 1 --reviewed-commit b9f6ebe --dry-run
bash tools/exp06_launch.sh probe --gpu 1 --reviewed-commit b9f6ebe --dry-run
bash tools/exp06_launch.sh full --gpu 1 --reviewed-commit b9f6ebe --dry-run
```

All exited **0** and printed the registered argv with `--reviewed-commit b9f6ebe`: the bounded smoke settings, 200-training-batch fit/timing probe, and full 12-epoch recipe with batch 32, accumulation 2, K=8, length 9600, TF32, and registered checkpoint cadence.

No files were modified, GPU work started, or processes signalled. Actual GPU smoke/probe acceptance remains part of the registered launch ladder. The previously deferred finalizer snapshot fix remains a separate requirement before the evaluation/HAA gate.

**Exact blocking findings the Coder must fix before this refill round closes: none.**
