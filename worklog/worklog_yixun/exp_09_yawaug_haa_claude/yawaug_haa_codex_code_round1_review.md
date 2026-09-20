# Codex code review — exp_09 round 1 (exp09-yawaug 0cecf8e)

- Reviewer: OpenAI Codex `gpt-6-astra`, `reasoning effort: xhigh` (CLI override), sandbox read-only, worktree `/home/yixunhu/codespace/xRIR_code_wt` at 0cecf8e (main 3b7d829)
- Prompt: exp_06 briefing + `review_prompts/codex_code_round1_prompt.md`; run 2026-09-19 23:10:23 → 23:25:57, log `yawaug_haa_2026-09-19_23:10:23_codex_code_round1_review.log`, Codex session 01a0bccb-20e6-7fb3-9091-e4f068807c8f

## Final message

**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-19.

Reviewed exp_09 round 1 in `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp09-yawaug`, HEAD `0cecf8e4c8dd4a553d69af4a5732227fa455524d`, base `3b7d82953ac2709d8a9dee26407f461058f48952`. Review completed at approximately 23:24 EDT, 2026-09-19.

**Verdict: request changes.** The HAA historical-approval fix works, but merging and refilling the reported six keys is insufficient to approve this snapshot.

I read the SOP, exp_09 plan v2 and its review/prompt/report, the requested exp_06 plans/reviews/notebook, the prerequisite experiment records and tooling, and every changed line in the round. No files were modified, no GPU work was started, and no processes were signalled.

1. **Blocker — the shared approvals-module change invalidates historical exp_06 simulated evaluations.**  
   Location: [tools/exp06_approvals_api.py:292](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_approvals_api.py:292), consumed by [tools/exp06_compare.py:161](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_compare.py:161).

   Adding the resolver here changes `eval_launch`’s closure. Historical simulated evaluations still record writer digest `0a73f84a5a1c17533cfc3e2ef62c24928fb157103e1f4d69bf8d2474ad7ff205`. Refilling `code.eval_launch` with `8c6666d5…` makes their existing `check_route` validation fail.

   I tested all **10 real manifests** under `ckpt/exp06/sim_eval/{cyl,cyl_or}/seed{42..46}`. Their writer-approval checks pass with the current approvals and fail with the proposed refill: `approved eval_launch closure`. Independently, their source revalidation passes against current main and fails against the proposed source bytes with:
   `source.writer_exp06.tools/exp06_approvals_api.py`.

   The new HAA blob rule does not govern these checks. This violates the requested preservation of completed exp_06 runs’ verifiability.

   **Minimal fix:** restore `exp06_approvals_api.py` byte-for-byte and relocate the new resolver into an already approved closure being changed for this round, such as `exp06_finalize.py`; update its callers. Add a regression preserving these historical writer/source checks. Recompute digests afterward.

2. **Should-fix, before launch — the checkpoint resolver validates a different read from the one whose identity it approves.**  
   Location: [tools/exp06_approvals_api.py:307](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_approvals_api.py:307).

   `_sha256_file(file)` checks the reused pin, then `exp04_profiles.load_approved_digests(file)` reads the file again. The returned identity is never compared with that pin.

   An in-memory probe modeled record X being replaced and committed as Y between these reads. The real exp_04 loader validated Y’s schema and simulated Git binding; the resolver accepted Y’s checkpoint digest (`aaaa…`) while its returned approvals identity **did not equal** `reused.exp04_approved_digests_sha256`. Both pipeline and summariser depend on this resolver.

   **Minimal fix:** require the loader’s returned `identity['sha256']` to equal the reused pin before using its checkpoint. Prefer one validated snapshot; retaining the preliminary check is harmless if the parsed snapshot is also checked. Add a replacement-between-reads regression.

3. **Should-fix, before summarising — default exp_06 output is not schema-identical.**  
   Location: [tools/exp06_summarize_haa.py:1146](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1146) and [tools/exp06_summarize_haa.py:1373](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1373).

   Both the statistics JSON and CLI result JSON unconditionally gain `experiment`, including default exp_06 execution. This contradicts the explicit unchanged-default/schema contract. The new “unchanged” test asserts the added field instead of comparing against the base.

   My base-versus-HEAD synthetic analysis found exactly one added statistics field, `experiment`; after removing it, serialized results matched byte-for-byte, and rendered summaries already matched.

   **Minimal fix:** emit the new field only for exp_09, preserving exp_06’s previous JSON and CLI schemas. Add a base-schema regression. Restricting analysis to registered arms and refusing missing arms is otherwise appropriate. Exp_06’s registered set contains **five** arms, A/B/C/D/F.

I independently verified the following:

- **Scope and commits:** `git diff 3b7d829..HEAD`, `git diff main...HEAD --diff-filter=M --name-only`, and protected-path `git diff main..HEAD --name-only` checks show only the eight reviewed exp_06 source/test files changed. Protected sources, approvals templates and approval-record copies are unchanged. All 11 commits have the required trailers and contain at most **179 additions plus deletions**; the report’s maximum of 153 is inaccurate, but the 200-line limit is satisfied. `git diff --check`, `bash -n`, and in-memory Python compilation passed.
- **Pins:** the exp_03 closure test passed with its exact 12 files and digest `5ba818d8…`. All 11 files in `source_closure('train_xRIR_backbone')` are byte-identical to main.
- **Existing pipeline goldens:** actual base/HEAD dry-run output was byte-identical for `cyl_or:0`, `control_hf:1`, `cyl_hf:2`, bare `zeroshot`, and their combined queue, using an absent output root.
- **Yawaug goldens:** seed, zero-shot and complete-queue output matched the new goldens with exp_09 output/log roots, `frame=room heading=none`, and no child `--heading-json-dir`. The correct queue is `yawaug:0 yawaug:1 yawaug:2 yawaug:zeroshot`: **31 children and four jobs**. Bare `zeroshot` retains its exp_06 meaning.
- **Frame and init refusals:** in-memory probes rejected room-frame specs with headings, heading-frame specs without headings, room-frame child arguments/metadata with headings, an incorrect pipeline checkpoint, an incorrect reused approvals identity, and a summary job with an incorrect init digest. Executing `JOB_SPEC_PY` in memory produced a heading-free room spec and the real approved init digest `f8e640523892154fe744b60e2fe99178b68a522ee3947758812aa5a7dc15b299`. As plan v2 explains, this comes from approved `checkpoints.aug`; `AUG` itself has no `sha256`.
- **Historical HAA admission:** `verify_job(..., sensitivity=False, inputs={})` admitted **12/12 real jobs and 93/93 children**, with current approvals `673c36481acf208f4982d9bd9977692e26d6abfe06adbbb01724a6ffbff1cdc6`. Every child retained historical blob identity `2d258f0444352bafbfe583a5ccda17132b27b5f6b3ba97e22f4d02d0ec6c87ad`. Additional probes preserved admission when the working-file read was replaced in memory and refused both a wrong child closure and an unapproved closure in the committed blob.
- **Statistics:** E1 strict zero/margin boundaries, overlapping harm plus margin non-inferiority, and void/unconverged suppression passed. E2 has 11 cells, adjusted alpha `0.05/11`, the registered tails, and no “non-inferior” labels. Synthetic exp_09 analysis produced A/B/C/E, 11 E3 comparisons and 44 side-split cells; missing E was refused. Python, NumPy and Torch global RNG states remained unchanged.

`compute_code_digests(worktree, 'HEAD')` reproduced all six reported changes exactly:

| Key | Recomputed SHA-256 |
|---|---|
| finalize | `bb269f55bf43fd1d0698ba45c45bcbc004deba0956b337708a1bb6f20c07fd40` |
| haa_pipeline_sh | `c1c1b1149c0b065ed523b0814ced8942bc99ee77d5c043897bb2e341f4c7325c` |
| summarize_haa | `9296f0536aae206b4ac22956401d15d01bbfde5d598867a515eae0a9b25a97fe` |
| compare | `64238a56ff7c81479f76d44a9e92340ea91afa7a9d7041d16685e12c3bc92d5f` |
| eval_launch | `8c6666d5dfa5039dde7705dc062204ef2bfe55b32c9c5f7626f4e4f9cb98c939` |
| mirror_probe | `38ae34e84622c86c098a9512659f2ab47bdccc9085f3bb1c3af2cb7ec35cb6a2` |

The other 14 keys are unchanged, including exactly:

- `haa_finetune`: `9288884b4034d4510c16ee081e95855e1229aa19a23b94f22e46ce90e6b0458b`
- `haa_eval`: `1f20e36bfb29b557de020321c30ba083d5631ae6e2f1b3fbfd94ecb0e77f1134`

Test replay used the specified interpreter, CPU-only environment, at most four threads, disabled bytecode writes, and pytest `-p no:cacheprovider -s`. Native collection was blocked by unwritable Numba/Matplotlib caches. With in-memory import/cache accommodations:

- `pytest.main` over the approvals, finalizer, HAA-consumer, pipeline and summariser modules, excluding tests requiring temporary-file fixtures: **94 passed, 454 deselected**, 18.82 seconds.
- Exp_03 pin test plus exp_07 runtime-approval parametrisations: **2 passed, 2 failed**. The failures were precisely `seen_simple-simple-0` and `seen_cyl-cylindrical-0`.
- Those exp_07 parametrisations on main: **1 passed, 2 failed**, confirming the same pre-existing null-approval guards.
- Exp_06 filled-digest guard: **1 failed**, with the same six changed keys and 14 identical keys.

Live HAA validation used memory-backed temporary approvals copies and `/proc` existence checks instead of signalling PIDs; repository validators, hashes and numerical calculations remained active.

The Coder’s broader counts are **reported, not fully reproduced here**: finalizer/consumer 335 passed; approvals 56; pipeline 61; summariser 96; final combined run 213 passed; full suite **3 failed, 3256 passed, 50 skipped**. My targeted replay corroborates the three reported failure causes.

The planned two-epoch class_room fine-tune followed by four-sample hallway evaluation is an adequate bounded functional smoke for this room-frame path, provided both receipts succeed and readback confirms the expected checkpoint lineage, room frame, null headings and evaluation outputs. It remains unperformed and must pass before the full queue; it does not replace completion/admission checks.

**Exact blocking list for round closure: findings 1, 2 and 3.** Findings 1–2 must be resolved before merge/launch approval; finding 3 must be resolved before summariser approval. After the fixes, recompute the changed keys, refill at the reviewed merge tip, record the suite with only the two known exp_07 guards, and complete the bounded smoke.
tokens used
414,075
**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-19.

Reviewed exp_09 round 1 in `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp09-yawaug`, HEAD `0cecf8e4c8dd4a553d69af4a5732227fa455524d`, base `3b7d82953ac2709d8a9dee26407f461058f48952`. Review completed at approximately 23:24 EDT, 2026-09-19.

**Verdict: request changes.** The HAA historical-approval fix works, but merging and refilling the reported six keys is insufficient to approve this snapshot.

I read the SOP, exp_09 plan v2 and its review/prompt/report, the requested exp_06 plans/reviews/notebook, the prerequisite experiment records and tooling, and every changed line in the round. No files were modified, no GPU work was started, and no processes were signalled.

1. **Blocker — the shared approvals-module change invalidates historical exp_06 simulated evaluations.**  
   Location: [tools/exp06_approvals_api.py:292](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_approvals_api.py:292), consumed by [tools/exp06_compare.py:161](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_compare.py:161).

   Adding the resolver here changes `eval_launch`’s closure. Historical simulated evaluations still record writer digest `0a73f84a5a1c17533cfc3e2ef62c24928fb157103e1f4d69bf8d2474ad7ff205`. Refilling `code.eval_launch` with `8c6666d5…` makes their existing `check_route` validation fail.

   I tested all **10 real manifests** under `ckpt/exp06/sim_eval/{cyl,cyl_or}/seed{42..46}`. Their writer-approval checks pass with the current approvals and fail with the proposed refill: `approved eval_launch closure`. Independently, their source revalidation passes against current main and fails against the proposed source bytes with:
   `source.writer_exp06.tools/exp06_approvals_api.py`.

   The new HAA blob rule does not govern these checks. This violates the requested preservation of completed exp_06 runs’ verifiability.

   **Minimal fix:** restore `exp06_approvals_api.py` byte-for-byte and relocate the new resolver into an already approved closure being changed for this round, such as `exp06_finalize.py`; update its callers. Add a regression preserving these historical writer/source checks. Recompute digests afterward.

2. **Should-fix, before launch — the checkpoint resolver validates a different read from the one whose identity it approves.**  
   Location: [tools/exp06_approvals_api.py:307](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_approvals_api.py:307).

   `_sha256_file(file)` checks the reused pin, then `exp04_profiles.load_approved_digests(file)` reads the file again. The returned identity is never compared with that pin.

   An in-memory probe modeled record X being replaced and committed as Y between these reads. The real exp_04 loader validated Y’s schema and simulated Git binding; the resolver accepted Y’s checkpoint digest (`aaaa…`) while its returned approvals identity **did not equal** `reused.exp04_approved_digests_sha256`. Both pipeline and summariser depend on this resolver.

   **Minimal fix:** require the loader’s returned `identity['sha256']` to equal the reused pin before using its checkpoint. Prefer one validated snapshot; retaining the preliminary check is harmless if the parsed snapshot is also checked. Add a replacement-between-reads regression.

3. **Should-fix, before summarising — default exp_06 output is not schema-identical.**  
   Location: [tools/exp06_summarize_haa.py:1146](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1146) and [tools/exp06_summarize_haa.py:1373](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1373).

   Both the statistics JSON and CLI result JSON unconditionally gain `experiment`, including default exp_06 execution. This contradicts the explicit unchanged-default/schema contract. The new “unchanged” test asserts the added field instead of comparing against the base.

   My base-versus-HEAD synthetic analysis found exactly one added statistics field, `experiment`; after removing it, serialized results matched byte-for-byte, and rendered summaries already matched.

   **Minimal fix:** emit the new field only for exp_09, preserving exp_06’s previous JSON and CLI schemas. Add a base-schema regression. Restricting analysis to registered arms and refusing missing arms is otherwise appropriate. Exp_06’s registered set contains **five** arms, A/B/C/D/F.

I independently verified the following:

- **Scope and commits:** `git diff 3b7d829..HEAD`, `git diff main...HEAD --diff-filter=M --name-only`, and protected-path `git diff main..HEAD --name-only` checks show only the eight reviewed exp_06 source/test files changed. Protected sources, approvals templates and approval-record copies are unchanged. All 11 commits have the required trailers and contain at most **179 additions plus deletions**; the report’s maximum of 153 is inaccurate, but the 200-line limit is satisfied. `git diff --check`, `bash -n`, and in-memory Python compilation passed.
- **Pins:** the exp_03 closure test passed with its exact 12 files and digest `5ba818d8…`. All 11 files in `source_closure('train_xRIR_backbone')` are byte-identical to main.
- **Existing pipeline goldens:** actual base/HEAD dry-run output was byte-identical for `cyl_or:0`, `control_hf:1`, `cyl_hf:2`, bare `zeroshot`, and their combined queue, using an absent output root.
- **Yawaug goldens:** seed, zero-shot and complete-queue output matched the new goldens with exp_09 output/log roots, `frame=room heading=none`, and no child `--heading-json-dir`. The correct queue is `yawaug:0 yawaug:1 yawaug:2 yawaug:zeroshot`: **31 children and four jobs**. Bare `zeroshot` retains its exp_06 meaning.
- **Frame and init refusals:** in-memory probes rejected room-frame specs with headings, heading-frame specs without headings, room-frame child arguments/metadata with headings, an incorrect pipeline checkpoint, an incorrect reused approvals identity, and a summary job with an incorrect init digest. Executing `JOB_SPEC_PY` in memory produced a heading-free room spec and the real approved init digest `f8e640523892154fe744b60e2fe99178b68a522ee3947758812aa5a7dc15b299`. As plan v2 explains, this comes from approved `checkpoints.aug`; `AUG` itself has no `sha256`.
- **Historical HAA admission:** `verify_job(..., sensitivity=False, inputs={})` admitted **12/12 real jobs and 93/93 children**, with current approvals `673c36481acf208f4982d9bd9977692e26d6abfe06adbbb01724a6ffbff1cdc6`. Every child retained historical blob identity `2d258f0444352bafbfe583a5ccda17132b27b5f6b3ba97e22f4d02d0ec6c87ad`. Additional probes preserved admission when the working-file read was replaced in memory and refused both a wrong child closure and an unapproved closure in the committed blob.
- **Statistics:** E1 strict zero/margin boundaries, overlapping harm plus margin non-inferiority, and void/unconverged suppression passed. E2 has 11 cells, adjusted alpha `0.05/11`, the registered tails, and no “non-inferior” labels. Synthetic exp_09 analysis produced A/B/C/E, 11 E3 comparisons and 44 side-split cells; missing E was refused. Python, NumPy and Torch global RNG states remained unchanged.

`compute_code_digests(worktree, 'HEAD')` reproduced all six reported changes exactly:

| Key | Recomputed SHA-256 |
|---|---|
| finalize | `bb269f55bf43fd1d0698ba45c45bcbc004deba0956b337708a1bb6f20c07fd40` |
| haa_pipeline_sh | `c1c1b1149c0b065ed523b0814ced8942bc99ee77d5c043897bb2e341f4c7325c` |
| summarize_haa | `9296f0536aae206b4ac22956401d15d01bbfde5d598867a515eae0a9b25a97fe` |
| compare | `64238a56ff7c81479f76d44a9e92340ea91afa7a9d7041d16685e12c3bc92d5f` |
| eval_launch | `8c6666d5dfa5039dde7705dc062204ef2bfe55b32c9c5f7626f4e4f9cb98c939` |
| mirror_probe | `38ae34e84622c86c098a9512659f2ab47bdccc9085f3bb1c3af2cb7ec35cb6a2` |

The other 14 keys are unchanged, including exactly:

- `haa_finetune`: `9288884b4034d4510c16ee081e95855e1229aa19a23b94f22e46ce90e6b0458b`
- `haa_eval`: `1f20e36bfb29b557de020321c30ba083d5631ae6e2f1b3fbfd94ecb0e77f1134`

Test replay used the specified interpreter, CPU-only environment, at most four threads, disabled bytecode writes, and pytest `-p no:cacheprovider -s`. Native collection was blocked by unwritable Numba/Matplotlib caches. With in-memory import/cache accommodations:

- `pytest.main` over the approvals, finalizer, HAA-consumer, pipeline and summariser modules, excluding tests requiring temporary-file fixtures: **94 passed, 454 deselected**, 18.82 seconds.
- Exp_03 pin test plus exp_07 runtime-approval parametrisations: **2 passed, 2 failed**. The failures were precisely `seen_simple-simple-0` and `seen_cyl-cylindrical-0`.
- Those exp_07 parametrisations on main: **1 passed, 2 failed**, confirming the same pre-existing null-approval guards.
- Exp_06 filled-digest guard: **1 failed**, with the same six changed keys and 14 identical keys.

Live HAA validation used memory-backed temporary approvals copies and `/proc` existence checks instead of signalling PIDs; repository validators, hashes and numerical calculations remained active.

The Coder’s broader counts are **reported, not fully reproduced here**: finalizer/consumer 335 passed; approvals 56; pipeline 61; summariser 96; final combined run 213 passed; full suite **3 failed, 3256 passed, 50 skipped**. My targeted replay corroborates the three reported failure causes.

The planned two-epoch class_room fine-tune followed by four-sample hallway evaluation is an adequate bounded functional smoke for this room-frame path, provided both receipts succeed and readback confirms the expected checkpoint lineage, room frame, null headings and evaluation outputs. It remains unperformed and must pass before the full queue; it does not replace completion/admission checks.

**Exact blocking list for round closure: findings 1, 2 and 3.** Findings 1–2 must be resolved before merge/launch approval; finding 3 must be resolved before summariser approval. After the fixes, recompute the changed keys, refill at the reviewed merge tip, record the suite with only the two known exp_07 guards, and complete the bounded smoke.

