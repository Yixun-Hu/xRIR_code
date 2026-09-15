**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-15.

# Round 2b CLOSE review

**Verdict: approve with changes. Round 2b may close.**

Reviewed every changed line in `d6a4f42..3df20b8` on `exp06-round2b`, against the SOP, approved v4 plan, prior reviews, notebook and role change, experiment records, composed tooling, and fix prompt/report. The concurrent training-gate changes, round 3 implementation, and GPU execution are outside this verdict.

No files were modified or processes signaled. The prohibited concurrent worktree was not accessed.

## Disposition of findings 1–6

1. **Original blocker — resolved: failed preparation prevents launches.**  
   **Location:** [exp06_haa_pipeline.sh:110](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_haa_pipeline.sh:110).

   `prepare_job` explicitly checks `open_job`, `job_spec`, and `check_spec`, including when Bash suppresses `errexit`. All six injected failures across fine-tuning and zero-shot paths returned nonzero with **zero launches**. The actual job-spec writer rejected a schema-valid, refused dampened-room heading before publishing a specification.

   Queue probes returned **exit 1 / `QUEUE_FAILED 2`** for two failed fine-tuning finalizers and **exit 1 / `QUEUE_FAILED 3`** for failed zero-shot finalizers. Neither reported `QUEUE_DONE`.

2. **Original should-fix — resolved: simulated outputs receive exp_06 metadata validation.**  
   **Location:** [exp06_eval_launch.py:180](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_eval_launch.py:180).

   Both outputs now require all six exp_06 fields, presence and type-sensitive equality, the inherited protocol fields, and matching completion hashes. All **24 corruption/omission cases** quarantined, including through the actual inherited `execute_run` followed by the new validator.

   Additional probes confirmed that valid output metadata with a stale completion hash quarantines for either output. An existing nonempty quarantine directory remained intact; the new quarantine received a UUID suffix. See nonblocking finding 7 concerning separately altered persisted completion bytes.

3. **Original should-fix — resolved: first-admission tests and initialization identity.**  
   **Locations:** [test_exp06_haa_pipeline.py:436](/home/yixunhu/codespace/xRIR_code_wt2b/tests/test_exp06_haa_pipeline.py:436), [exp06_finalize.py:1073](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_finalize.py:1073).

   Mutation tests remove any prior job completion, require a field-specific refusal, and assert no completion appears. Both wrappers record the specification digest and initialization identity.

   Admission probes refused all eight frozen-field mutations. The requested initialization mutation failed specifically with:

   > child stage1 was launched under the job_init 'cyl_or', not the 'somebody_elses_pretrain' of this job spec

   A byte-different specification and a single child with a different specification digest, initialization label, or initialization digest were also refused. A post-completion child-args mutation failed artifact-hash reconciliation.

4. **Original should-fix — resolved: validation-only headings are revalidated.**  
   **Location:** [exp06_finalize.py:647](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_finalize.py:647).

   The finalizer checks the training/validation union while retaining training membership separately. Hallway training with classroom validation bound both headings. Replacing only the classroom JSON with invalid text—even with its recorded hash refreshed—was refused. Missing bindings and empty or mistyped `val_rooms` were also refused.

5. **Original nit — resolved: checkpoint quoting and job syntax.**  
   **Locations:** [exp06_haa_pipeline.sh:193](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_haa_pipeline.sh:193), [exp06_haa_pipeline.sh:256](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_haa_pipeline.sh:256).

   A checkpoint path containing spaces survived as **one actual `--init` argument**. All nine malformed-job probes returned exit 2 before launching, including `cyl_or:anything:0`, `cyl_or:0:1`, and `cyl_or:-1`.

6. **Original nit — resolved: commit-size accounting corrected.**  
   **Location:** [fix report:107](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_2026-09-15_07:28:25_coder_round2b_fix_opus_report.md:107).

   The correction identifies all three earlier oversized commits, including the 227-line test commit. No history rewrite is needed.

## Nonblocking nits for round 3

7. **Nit — the post-run gate does not reread persisted `completion.json`.**  
   **Location:** [exp06_eval_launch.py:233](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_eval_launch.py:233).

   Replacing one hash in the on-disk completion while preserving the correct mapping returned by `execute_run` is not detected by `certify_outputs`.

   This is nonblocking because the normal inherited writer atomically publishes the mapping it returns; the reproduction additionally alters the persisted record. Nevertheless, validating the retained artifact would strengthen the completion contract.

   **Minimal fix:** Reread `completion.json`, compare it with the returned mapping, and validate its hashes. Add a regression distinct from ordinary output corruption.

8. **Nit — failed preparation lacks a structured job failure record.**  
   **Location:** [exp06_haa_pipeline.sh:110](/home/yixunhu/codespace/xRIR_code_wt2b/tools/exp06_haa_pipeline.sh:110).

   A fresh refused preparation leaves its directory and, after queue exit, a dead `launch.pid`; its cause survives only in terminal output. This does **not** block closure: preparation launches no child, failure propagates, and no new job completion is published.

   **Minimal fix:** Retain a structured preparation-failure receipt or document this recovery state. A job-level `_ABORTED_` rename is not necessary for the correctness fix.

## Assessment of deviations

- **`python -c … load_job_spec`: accepted.** The prompt explicitly permits an equivalent import; this executes the finalizer’s validator.
- **`exp06_haa_finetune.load_headings`: accepted.** It invokes the heading validator and cache-input verification, then enforces decided and confirmatory records.
- **Identity read from child `args.json`: accepted.** Child validation compares args with provenance, and job admission reconciles their artifact hashes against the child completion before checking identity.
- **Failed preparation without job-level rename: nonblocking**, as explained in finding 8.
- **Shared fixtures with prior completion removed: accepted.** Specific refusal assertions now prevent completion-publication conflicts from satisfying mutation tests.

## Verification record

All Python checks used the requested interpreter and environment, including empty CUDA visibility, four OpenMP threads, disabled bytecode, and the specified Numba-cache path.

| Check | Result |
|---|---|
| `git diff d6a4f42..3df20b8` | Every changed line reviewed |
| `git diff --check d6a4f42..3df20b8` | Clean |
| `git status --porcelain=v1` | Clean |
| `git diff main...exp06-round2b --diff-filter=M --name-only` | Empty |
| Direct working-byte comparison against `main` | **43 pinned/composed files, including pin tests, identical** |
| Enumerated exp_03 closure and reviewed history | All 12 unchanged; digest `5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48` |
| In-memory `compile()` on changed Python files | **9 passed**; avoided `.pyc` writes |
| `bash -n` on both exp_06 launch scripts | Passed |
| Direct output-test-body replay with in-memory filesystem | **31 passed**, including all **30 newly added output cases** |
| Inherited launcher → new validator composition | **24/24** corrupted outputs quarantined |
| Shell, heading, and job-admission probes | Results detailed above |
| Fine-tuning/evaluation numerical loops and inversion helper | AST-identical across this fix range |

### Replay limitations

The full requested command was attempted:

```text
python -m pytest tests/test_exp06_*.py tests/test_provenance.py -q -p no:cacheprovider
```

It failed before collection because pytest capture could not create a temporary file. Retrying the three affected test modules with `--capture=sys` produced **three collection errors** from Matplotlib/Numba cache requirements.

Consequently:

- **No unmodified pytest suite completed.**
- Adapted checks used actual source functions and test bodies, with filesystem operations confined to memory and expensive unchanged dependencies stubbed where stated.
- The filesystem-backed nine-child integration, actual quarantine rename, and atomic-publication behavior were not independently replayed.
- **687 passed / 3 skipped** remains the Coder’s reported total, not my reproduced total.

### Commit discipline

| Red commit | Lines | Implementation | Lines |
|---|---:|---|---:|
| `a49e3a3` | 58 | `c1f9829` | 24 |
| `93b397f` | 106 | `052bf34` | 105 |
| `a1a851d` | 138 | `497824b` | 185 |
| `5189b0d` | 92 | `3df20b8` | 106 |

All eight commits carry both required trailers. Each test-only red commit immediately precedes its implementation. Read-only historical probes reproduced the validation-union, unfrozen-init, and nine-launch preparation failures; the output-validation API is absent at its red snapshot. The report’s exact aggregate red counts were not independently reproduced.

## Exact blocking list

**None. Round 2b may close.** Carry nits **7–8** into round 3; combined integration and the later full review remain necessary before HAA/evaluation launches.
