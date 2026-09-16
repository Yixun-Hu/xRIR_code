**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

## Verdict

**Approve with changes. Round 3b may close.**

The five requested fixes pass their scoped reproductions. The remaining parse/hash window in the unchanged finalizer is a **Planner post-merge should-fix requiring its own reviewed round**, described below.

Reviewed `07001a0..6ac832c` on `exp06-round2b`, including every changed line, the required briefing records, and the composed tooling.

## Findings and dispositions

1. **2a — Resolved; previous blocker.**  
   [tools/exp06_summarize_haa.py:542](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:542)

   Child provenance dependencies now enter the shared publication input map with their validated digests: both data inventories, source-closure files, and mutable inputs. Contradictory bindings are refused.

   Independently admitted a re-certified nine-child job retaining **123 inputs**, then separately changed:

   - `hallway/rirs.npy`
   - `hallway/depth.npy`
   - `tools/exp06_haa_finetune.py`
   - `tools/exp06_haa_eval.py`
   - `sim_to_real/finetune_haa.py`

   Every publication attempt raised `input changed during analysis`, leaving neither output. Prebinding the RIR dependency under a different digest raised `contradictory bindings`.

   **Further round-3b fix:** none.

2. **2b — Partial overall; the round-3b parent-binding defect is resolved.**  
   [tools/exp06_summarize_haa.py:610](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:610), [tools/exp06_summarize_haa.py:633](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:633)

   Certified spec/completion digests are bound before delegation. The completion subsequently parsed must match its certified digest, and the loader’s returned spec digest must match the parent’s declaration.

   My exact reproduction—changing stage 1’s completion log binding immediately after `verify_child` returned—was refused with `stage1/completion.json is not the bytes bound for it`. A loader returning a different `job_spec_sha256` was also refused. The committed persistent-change regressions pass.

   The inner finalizer window remains; see finding 6. **Further fix within this round:** none.

3. **3 — Resolved; previous blocker.**  
   [tools/exp06_summarize_haa.py:483](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:483)

   Effective validation rooms now match the wrapper’s defaulting rule and are checked against each stage’s registered population.

   I independently constructed synthetic histories at **1000/10 and 200/2**, set the remaining registered recipe fields, and re-certified all nine children and the job through the real finalizer. The baseline admitted without recipe deviations.

   Re-certifying either of these variants caused primary admission to refuse specifically on validation rooms:

   - Stage 1: `val_rooms=['hallway']`.
   - `stage2_hallway`: `val_rooms=['class_room']`, with the additional heading and data bindings supplied.

   Each variant admitted under sensitivity mode with exactly one reported recipe deviation.

   **Further round-3b fix:** none.

4. **5 — Resolved; previous should-fix.**  
   [tools/exp06_compare.py:401](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_compare.py:401)

   The exp_05 tier parser and digest now consume one byte buffer, reconciled with the existing and declared bindings.

   The normal legacy-M fixture admitted. The original transient `read_text` substitution was never consumed and admission refused the inconsistent `legacy_M` declaration. A new substitution through `read_bytes` was refused on digest mismatch.

   **Further round-3b fix:** none.

5. **6 — Resolved; previous should-fix.**  
   [tools/exp06_summarize_haa.py:338](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_summarize_haa.py:338)

   `launch.pid` is parsed and hashed from one buffer. Changing the marker at its first read retained the original PID and original digest; publication revalidation then refused the changed marker.

   This fixes the contradictory PID/digest evidence without introducing an inappropriate retrospective liveness check.

   **Further round-3b fix:** none.

6. **Should-fix — Deferred finalizer snapshot gap.**  
   [tools/exp06_finalize.py:1452](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_finalize.py:1452), [tools/exp06_finalize.py:1477](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_finalize.py:1477)

   `load_job_spec` still parses and subsequently hashes separate reads. I reproduced admission of substituted room ordering while the retained digest identified the original bytes; final revalidation passed.

   This violates the single-snapshot guarantee. However, substitutions changing **seed, initialization hash, room membership, or heading** were refused by the surrounding schema and child-lineage checks. The admitted ordering substitution preserved the registered room set and did not change the statistical inputs.

   **Disposition:** a separate Planner-owned training-path round before the full evaluation/HAA gate, rather than another round-3b blocker.

   **Minimal fix:** parse and hash one raw spec buffer, carry that digest through job certification, and add transient-substitution regressions. Review the dependent certification reads in that round.

## Disclosed deviations and commit hygiene

- **Finding-3 fixture:** acceptable for testing admission. My independent full-schedule synthetic re-certification additionally isolated the validation-room refusal from shortened-budget refusals. Neither fixture demonstrates actual training completion or GPU behavior.
- **Report nit:** the approximately 31-hour estimate describes pretraining. The plan estimates approximately 50 minutes per HAA fine-tuning job. Correct that parenthetical in the record.
- **Test-only `6ac832c`:** acceptable additional regression. It introduces no implementation requiring a subsequent implementation commit. Its retrospective red result is disclosed.
- **History:** 11 commits, five red/implementation pairs plus that regression; maximum **57 changed lines** per commit; both required trailers present throughout. Exactly four permitted files changed.

## Verification

Checks used the requested Python, `PYTHONPATH`, data path, CPU-only environment, four OpenMP threads, disabled bytecode, and disabled pytest cache provider.

Filesystem-dependent tests used a **process-local RAM filesystem**, current-repository fixtures, and the unchanged pipeline job-spec Python executed in-process. Numba JIT was disabled. No filesystem file was modified, no process was signalled, and no GPU work ran. Filesystem durability was not tested.

| Check / command | Result |
|---|---|
| `git diff 07001a0..6ac832c` | Every changed line reviewed |
| `git diff --check 07001a0..6ac832c` | Clean |
| `git log`, per-commit numstats and trailers | Hygiene verified as above |
| In-memory `compile(...)` of four changed Python files | Passed |
| `git show main:<path>` comparisons | **42 protected paths byte-identical** |
| `git diff main...6ac832c --diff-filter=M --name-only` | Only eight earlier exp_06-owned files |
| Compare, summarizer and approvals-API pytest replay | **193 passed** across batches of 176, 14 and 3 |
| Real exp_02 canonical regression | **All 11 cells exact** |
| Real exp_04 control admission | **All five seeds admitted**, 6,337 queries |
| Independent HAA and exp_05 adversarial probes | Outcomes recorded above |
| Repeated H3 analysis | Identical results; Python, NumPy and Torch CPU RNG states unchanged |
| Final branch / HEAD / status | `exp06-round2b`, `6ac832c62e47b03baf6b49bbb297203c813069ea`; clean |

Pytest was dispatched through `pytest.main` with `-q --capture=sys -p no:cacheprovider` for:

```text
tests/test_exp06_compare.py
tests/test_exp06_summarize_haa.py
tests/test_exp06_approvals_api.py
```

Harness-only setup errors were corrected before the affected cases passed. The pinned closure digests matched:

```text
exp_03: 5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48
trainer: 5b2da2504e220b63bfc932cf0d83c143261e117edea1838e6907159c3b12eb13
```

I did **not** independently replay every historical red revision or the Coder’s complete reported **1,218 passed / 8 skipped** suite.

## Final post-merge wiring list

1. Complete and review the separate finalizer snapshot fix above before the full evaluation/HAA gate.
2. Integrate the `code.probe_align` schema and the reviewed approvals tests. Preserve the filled approvals record; the older instruction to make it byte-identical to the null template is obsolete.
3. Recompute code digests at the final merged revision and commit the reviewed approvals. Supply the full reviewed SHA through `--approved-commit`.
4. Produce and approve the reconstructed legacy receipt; fill its path/hash and the exp_02, exp_04 and exp_05 reused identities.
5. Fill the native epoch-12 checkpoint, four heading hashes, and G1 artifact hash using the schema’s accepted keys.
6. Supply explicit approvals/G1 paths and absolute manifest paths. Retain all referenced evidence, including every job-root `launch.pid`.
7. Run primary HAA admission with the registered recipe and without `--sensitivity`; keep S1 outputs explicitly labelled. Preserve the registered H3 arm-B route and checkpoint identity.
8. Complete the post-merge full integrative review, suite replay, and applicable remaining validation ladder before evaluation/HAA execution.

## Exact blocking list

**None for closing round 3b. Round 3b may close.**
