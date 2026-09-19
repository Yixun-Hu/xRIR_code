# Codex review — exp_06 integrative `full` review (plan §10a A1) at 08a03e7

- Reviewer: OpenAI Codex `gpt-6-astra`, `reasoning effort: xhigh` (CLI override), sandbox read-only, main tree `/home/yixunhu/codespace/xRIR_code` at HEAD 08a03e7369c209459fcd5b8aad30580fcc19d6a3 (diff base 77ef292)
- Prompt: `review_prompts/codex_code_review_briefing.md` + `review_prompts/codex_code_full_prompt.md` (SHA_FULL = 08a03e7)
- Run: 2026-09-17 21:31:51 → 21:51:37 local, log `oriented_cyl_2026-09-17_21:31:51_codex_code_full_review.log`, Codex session 01a0b224-3b5b-71e2-9061-87ab76bfffbd

## Final message

**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-17.

Tree: `/home/yixunhu/codespace/xRIR_code`, branch `main`. Reviewed snapshot: `08a03e7369c209459fcd5b8aad30580fcc19d6a3`, against `77ef292`. Verification cutoff: **2026-09-17 21:49:43 EDT**. During review, HEAD advanced to `5de272c0bb2d702a965c816f130e1ad878787292`; the sole committed difference is an exp_07 notebook change. Exp_06 source, tests, and approvals remain identical to the requested snapshot.

**Verdict: request changes.** Findings F1–F3 affect launch readiness. F4–F5 must be fixed before the H3 comparer publishes results. The finalized pretraining evidence and current approval hashes passed the checks described below.

No files were modified, GPU work started, or processes signaled.

**Findings**

1. **F1 — blocker: the first simulated evaluations fail because their parent directories do not exist.**

   Location: [tools/exp06_eval_launch.py:251](/home/yixunhu/codespace/xRIR_code/tools/exp06_eval_launch.py:251), [tools/exp04_eval_launch.py:55](/home/yixunhu/codespace/xRIR_code/tools/exp04_eval_launch.py:55), and the runbook’s generated simulation commands.

   The exp_06 launcher delegates immediately to `execute_run`, which calls `run.mkdir()` without `parents=True`. The runbook creates neither `ckpt/exp06/sim_eval/cyl_or` nor `ckpt/exp06/sim_eval/cyl`; both were absent at inspection.

   A read-only probe intercepted the actual `Path.mkdir` call before any write: both arms reached it with no arguments and a nonexistent parent. Consequently, the registered first launches fail before evaluation.

   **Minimal fix:** create the parent directories in exp_06-owned orchestration while retaining exclusive creation of each run directory. Do not modify the pinned exp_04 helper. **Required before sim launch.**

2. **F2 — blocker: the runbook’s simulated outputs cannot satisfy their own comparer’s training-evidence requirement.**

   Location: [tools/exp06_eval_launch.py:43](/home/yixunhu/codespace/xRIR_code/tools/exp06_eval_launch.py:43), [tools/exp06_compare.py:320](/home/yixunhu/codespace/xRIR_code/tools/exp06_compare.py:320), [posttrain_runbook.sh:183](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/planner_probes/posttrain_runbook.sh:183).

   Neither generated arm-C nor arm-B command supplies `--bind-input`. Both parse to empty bindings. The comparer requires `train_manifest` and `train_completion` for every `exp06` route; both therefore fail with:

   ```text
   the exp_06 arm binds its training run
   ```

   There is also a literal §6.3 deviation: exp_06 validates only `heading` itself; `train_completion` receives inherited file hashing, without semantic validation or a demonstrated connection to the evaluated checkpoint. The comparer fixture masks this by binding training-evidence names to an evaluation log.

   **Minimal fix:** bind and validate C’s actual pretraining provenance/completion and checkpoint relationship. Define the appropriate historical-evidence branch for fallback B, whose exp_01 training lacks modern completion records. Add an integration test from the exact runbook argv through comparer admission for both arms. **Required before sim launch.**

3. **F3 — should-fix, launch-blocking: G1, HAA, and simulated-evaluation execution do not enforce §6.4’s approved identities.**

   Location: [tools/exp06_mirror_probe.py:568](/home/yixunhu/codespace/xRIR_code/tools/exp06_mirror_probe.py:568), [tools/exp06_haa_pipeline.sh:156](/home/yixunhu/codespace/xRIR_code/tools/exp06_haa_pipeline.sh:156), [tools/exp06_finalize.py:1143](/home/yixunhu/codespace/xRIR_code/tools/exp06_finalize.py:1143), [tools/exp06_eval_launch.py:127](/home/yixunhu/codespace/xRIR_code/tools/exp06_eval_launch.py:127).

   These paths capture identities or check consistency with their selected commit, but do not compare execution inputs against the committed producer approvals. The runbook’s `producer_ok` calls `require_producer`, which checks populated fields, not equality with current code/artifacts.

   In memory, replacing the approved mirror-probe digest and epoch-checkpoint hash with incorrect, well-formed hashes still yielded:

   ```text
   mirror_probe: []
   haa_children: []
   sim_eval: []
   ```

   Current approvals independently match HEAD; that does not implement the plan’s mandatory refusal of unapproved execution.

   **Minimal fix:** enforce the committed producer matrix and compare actual code/checkpoint/heading identities before confirmatory work; retain and revalidate the approval binding. Preserve explicitly diagnostic paths. **Required before G1/HAA/sim launch.**

4. **F4 — should-fix: descriptive exp_04 cylindrical evaluations are admitted as H3 arm B.**

   Location: [tools/exp06_compare.py:84](/home/yixunhu/codespace/xRIR_code/tools/exp06_compare.py:84), [tools/exp06_compare.py:124](/home/yixunhu/codespace/xRIR_code/tools/exp06_compare.py:124), [tools/exp06_compare.py:247](/home/yixunhu/codespace/xRIR_code/tools/exp06_compare.py:247).

   With the committed approvals, the real call equivalent to:

   ```python
   exp06_compare.admit_run(
       Path("ckpt/yaw_aug/eval/cyl_k8_seed42_k0"), "B", approved,
       inputs={}, stats={}
   )
   ```

   **admitted all 6,337 queries**, despite absent checkpoint-role and checkpoint-epoch fields. B’s unrestricted route falls back to `exp04`, then receives the historical missing-epoch exception.

   Section 6.3 registers exp_05 M-tier evaluations or an exp_06 fallback for B. The exp_04 reuse exception belongs to A.

   **Minimal fix:** restrict B to its two registered routes and enforce their respective role/epoch evidence. Add this real descriptive-run refusal regression. **Required before H3 comparison/pages; does not itself block data collection.**

5. **F5 — should-fix: H3 does not implement the registered convergence rule.**

   Location: [tools/exp06_compare.py:485](/home/yixunhu/codespace/xRIR_code/tools/exp06_compare.py:485).

   H3 calls the pinned `paired_compare.convergence` once. That helper accepts identical zero-width intervals, and the exp_06 wrapper never quadruples the bootstrap count after failure. Section 7 explicitly requires zero-width refusal and one quadrupling attempt.

   An in-memory example with five seeds and 24 identical queries per arm produced `rho=0`, interval `[0, 0]`, `passed=True`, and **“non-inferior”**, with no convergence failures.

   **Minimal fix:** add the exp_06 convergence policy around the inherited H3 resampling: refuse zero width, retry once at four times the draws, and withhold unsuccessful verdicts. Record the attempts. Keep `paired_compare.py` unchanged. **Required before H3 comparison/pages.**

**Answers to the requested checks**

**1. Plan fidelity.** The numerical and recipe contracts largely match; F2–F5 are substantive plan-text deviations.

- **§6.1:** inspected the frozen legacy cohort, anchors `0.975/0.85` and `0.435/0.24`, tolerance `0.01`, exact mirrors `(-x,-y,z)`, strict bracketing thresholds, and strict oriented pass thresholds `<0.80` and `<0.50`. Cache-coordinate readback confirms **185 positive-y test microphones**. Failed legacy reproduction or bracketing makes G1 inconclusive. See [mirror_probe.py:154](/home/yixunhu/codespace/xRIR_code/tools/exp06_mirror_probe.py:154), [mirror_probe.py:220](/home/yixunhu/codespace/xRIR_code/tools/exp06_mirror_probe.py:220), and [mirror_probe.py:399](/home/yixunhu/codespace/xRIR_code/tools/exp06_mirror_probe.py:399). I did **not** rerun real-checkpoint anchor inference; the Coder’s recorded `0.9745/0.8510` and `0.4354/0.2382` remain reported evidence.
- **§6.2:** stage-1 rooms and `1000/10/TF32`, per-room stage-2 `200/2/TF32`, K=8, evaluation seed 0, unseeded primary Griffin–Lim, default depth, all three heading-frame initializations, and zero-shot layout match. Parser-default checks and the exact dry run support this.
- **§6.3:** delegation to `run_exp04`, evaluation flags, roles and epoch match. Launch/admission composition fails as F1–F2 describe; B’s admission is too broad under F4.
- **§6.4:** the three-section schema, completion units, reconstructed legacy branch, and dependency matrix exist. Execution enforcement is incomplete under F3.
- **§7:** HAA estimands, finite-across-all-seeds query cohorts, H1/H1b void rules, strict `+0.23 dB`/zero margins, Bonferroni-11 screen, and HAA convergence/retry machinery match. All eleven historical cells reproduce exactly. H3’s ratio estimand and strict `<+3%` bound match, but its convergence implementation fails F5.
- The executed encoder/factory/heading tests also cover the §2.2 symmetry and bit-exactness contracts, including heading cancellation and the **526,336** parameter increment.

**2. CPU composition and admission.**

The requested command:

```bash
bash tools/exp06_haa_pipeline.sh 1 cyl_or:0 zeroshot --dry-run
```

exited 0 and printed **4 jobs, 5 training children, 16 evaluation children, and 25 finalizer calls**, including job specifications and registered directories. No printed GPU command was executed.

The HAA fixtures do use real child/job finalizers and summarizer validators, with cases for live PIDs, missing completions, altered specifications, recipe deviations, non-confirmatory headings, and missing/incorrect side labels. However, their writable filesystem fixtures could not be replayed in this sandbox; I therefore do **not** certify the complete CPU fixture chain as freshly green.

Actual read-only admission succeeded for **all five** reused exp_04 control runs, seeds 42–46, with 6,337 queries each. The requested refusal of descriptive cylindrical B **failed**, as F4 documents.

**3. Simulation argv.**

For both C and fallback B, parser/serializer checks reproduce the registered evaluator flags:

```text
--conditions P --yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols
--batch-size 16 --num-workers 6 --max-samples 0
--gl-seed <s> --threads 4 --log-interval 10 --decomposition-batches 0
```

TF32 is false; C has role `arm`, B `baseline`, both epoch 12, room frame, and no heading. The runbook computes semantic manifest hashes using `tools.reference_manifest.manifest_hash`; seed 42 recomputed as:

```text
9cf5c8a236afa4e38aa111d8658769767543389b169d6b16f7042c092d3faa10
```

**The evaluator argv is correct; orchestration and evidence bindings need F1–F2.**

**4. Approvals.**

Independent `exp06_profiles.compute_code_digests(repo, target_commit)` recomputation matched **all 20 code keys** at `08a03e7`. Shell closures bind the two shell files explicitly.

Relative to `b9f6ebe`, exactly four keys changed:

| Key | Independently confirmed cause |
|---|---|
| `finalize` | Finalizer edits plus changed imported smoke code |
| `smoke` | Its own edits |
| `launch_sh` | Its own shell-file edits |
| `summarize_haa` | Import-closure propagation from finalizer/smoke; summarizer bytes unchanged |

No other cause was found.

Recomputed reused identities match: exp_04 entrypoint `0b05245c…`, writer `15f9c984…`, exp_04 approvals `2e452117…`, exp_05 approvals `293c5683…`, exp_02 stats `98062d76…`, and summary `736f9fb8…`. All five controls carry the expected evaluator/writer identities.

The reconstructed receipt’s path/hash match; its hash is `5a124946…`, and **all 126 enumerated artifact hashes match**. The epoch artifact has epoch 12 and the correct checkpoint hash.

`require_producer` returns no deviations for legacy receipt, mirror probe, HAA children, and sim evaluation. **It also returns none for `compare`: contrary to the question’s parenthetical, §6.4 does not make H3 depend on G1.** The summarizer correctly awaits `artifacts.gate_g1_sha256`.

These successful population/hash checks do not remove F3.

**5. Finalized pretraining.**

For `attempt_20260916T161415`, independently verified:

- `completion.json`: `run_type="full"`, `admissible_arm=true`, diagnostic false, child exit 0, twelve epochs, no approval deviations.
- All five recorded artifact hashes match, including:

  ```text
  epoch_012.pth:
  3df2ceb53d1be1402cb6672f05ded092187f1ba871b53fc1114ca088612296c1
  ```

- History contains epochs **1–12 exactly**, with finite required losses/timings.
- `exp06_recipe.check_all(args)` returns `[]`; normalized exp_01 recipe comparisons also return `[]`.
- Recipe: batch 32 × accumulation 2, TF32, 12 epochs, learning rate `1e-3`, decay `3/0.1`, seed 0, and 9,261 training batches per epoch.
- The bound historical approvals equal the committed bytes at `6dc0b8ec3553c226f0e2e6e1f587eb806f4ff113`.
- `final` points to the specified attempt.
- Launcher and child PIDs are absent from `/proc`; no signal-based checks were used.
- The closed log/receipt agree and end with `EXP06_CHILD_EXIT 0 2026-09-17T23:16:15+00:00`.

[full_evidence:762](/home/yixunhu/codespace/xRIR_code/tools/exp06_finalize.py:762) checks checkpoint keys, dtypes, shapes, and tensor equality against `last.pth["model"]`. I verified the retained hashes and inspected that assertion; **I did not independently load the checkpoints to repeat tensor equality**, or rehash the entire training-data inventory.

**6. Pins and neighbours.**

The requested protected-path diff against `77ef292` is empty, including `sim_to_real`, all twelve exp_03 files, the original trainer, and the named exp_04/05/provenance/statistics tooling. The trainer’s eleven-file import closure also matches the base.

The exp_03 closure recomputes to:

```text
5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48
```

`git diff main...exp06-window --diff-filter=M --name-only` is empty.

Neither neighbouring approvals file was changed by exp_06. Qualification: exp_04’s approvals file differs from `77ef292` because of its **own** approvals-fill commit `fa219ef`; “byte-identical to the base” would therefore be inaccurate for that record. Exp_05’s approvals file is unchanged.

Neighbour test coverage was partial, as listed below; I cannot claim the complete neighbour suites are freshly green.

**7. Test execution and limits.**

All Python checks used the requested interpreter, hidden CUDA, four threads, and disabled bytecode/cache-provider writes.

| Executed check | Result |
|---|---|
| `python -m pytest tests -q -p no:cacheprovider` | Failed before collection: `FileNotFoundError: No usable temporary directory found` |
| `python -m pytest tests/test_exp06_bootstrap.py -q -s -p no:cacheprovider` | **53 passed**; eleven historical cells exact, maximum difference 0 |
| Read-only selection from encoder/factory/heading/probe-align/recipe/eval/mirror-probe tests | **249 passed, 6 skipped, 59 deselected** |
| Read-only selection from exp_03 record, provenance, exp_04 and exp_05 tests | **39 passed, 657 deselected** |
| Read-only selection from approvals/profiles/comparer/HAA-summarizer tests | **66 passed, 151 deselected** |
| `bash -n tools/exp06_launch.sh tools/exp06_haa_pipeline.sh` | Passed |
| `git diff 77ef292..08a03e7 --check -- model tools tests` | Passed |
| Exact HAA dry run and real control-admission checks | Passed as detailed above |

The selected suites total **407 passed, 6 skipped**; this is **not** a full-suite count. Adapted checks used in-memory import/cache handling, `NUMBA_DISABLE_JIT=1`, and deselection of filesystem-writing/process-control fixtures. Those adaptations did not replace the production admission/statistics predicates used for the reported counterexamples.

At cutoff, the Planner’s separate CPU suite log had reached **76% without a terminal summary**. It is external, incomplete evidence. **Full-suite green at HEAD is not established by this review.**

**8. Prior findings.**

The cited prior fixes are present: round-2b preparation/metadata/heading checks; round-3a captured checkpoint bytes, panorama binding and ownership-safe cleanup; round-3b recipe, evidence retention, committed approvals and snapshot corrections; premerge mandatory job ownership; and prelaunch smoke-budget, confinement and job-spec snapshot fixes.

In particular, [load_job_spec:1553](/home/yixunhu/codespace/xRIR_code/tools/exp06_finalize.py:1553) now parses and hashes one buffer and rejects subsequent changes, closing the explicitly deferred snapshot issue.

I found no additional reopened prior should-fix beyond the integration gaps identified here. Classification:

- **Before launch:** F1–F2 for sim; F3 for G1/HAA/sim.
- **Before comparer/results pages:** F4–F5.
- **Reporting-only:** historical commit-size/process disclosures and the optional S1 limitation below.
- G1’s future artifact approval remains an expected prerequisite for summarization.

**9. Optional seeded-phase sensitivity S1.**

Confirmed: [exp06_summarize_haa.py:1228](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:1228) always loads historical A/B through the fixed legacy branch; `--sensitivity` is passed only to new-arm admission. There is no supported loader for seeded re-evaluations of historical A/B.

**S1 as written is not implementable through the existing end-to-end interface without new code.** It remains optional and descriptive, and does not block the primary comparison.

**Exact closure requirements:** fix **F1, F2, F3, F4 and F5**. F4–F5 may follow data collection, but must precede H3 publication. Before launch approval, complete F1–F3 as applicable, refresh affected approval digests, and obtain a green authorized full-suite replay plus the missing integration regressions. The current snapshot is **not approved for G1/HAA/sim launch**.
tokens used
751,312
**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-17.

Tree: `/home/yixunhu/codespace/xRIR_code`, branch `main`. Reviewed snapshot: `08a03e7369c209459fcd5b8aad30580fcc19d6a3`, against `77ef292`. Verification cutoff: **2026-09-17 21:49:43 EDT**. During review, HEAD advanced to `5de272c0bb2d702a965c816f130e1ad878787292`; the sole committed difference is an exp_07 notebook change. Exp_06 source, tests, and approvals remain identical to the requested snapshot.

**Verdict: request changes.** Findings F1–F3 affect launch readiness. F4–F5 must be fixed before the H3 comparer publishes results. The finalized pretraining evidence and current approval hashes passed the checks described below.

No files were modified, GPU work started, or processes signaled.

**Findings**

1. **F1 — blocker: the first simulated evaluations fail because their parent directories do not exist.**

   Location: [tools/exp06_eval_launch.py:251](/home/yixunhu/codespace/xRIR_code/tools/exp06_eval_launch.py:251), [tools/exp04_eval_launch.py:55](/home/yixunhu/codespace/xRIR_code/tools/exp04_eval_launch.py:55), and the runbook’s generated simulation commands.

   The exp_06 launcher delegates immediately to `execute_run`, which calls `run.mkdir()` without `parents=True`. The runbook creates neither `ckpt/exp06/sim_eval/cyl_or` nor `ckpt/exp06/sim_eval/cyl`; both were absent at inspection.

   A read-only probe intercepted the actual `Path.mkdir` call before any write: both arms reached it with no arguments and a nonexistent parent. Consequently, the registered first launches fail before evaluation.

   **Minimal fix:** create the parent directories in exp_06-owned orchestration while retaining exclusive creation of each run directory. Do not modify the pinned exp_04 helper. **Required before sim launch.**

2. **F2 — blocker: the runbook’s simulated outputs cannot satisfy their own comparer’s training-evidence requirement.**

   Location: [tools/exp06_eval_launch.py:43](/home/yixunhu/codespace/xRIR_code/tools/exp06_eval_launch.py:43), [tools/exp06_compare.py:320](/home/yixunhu/codespace/xRIR_code/tools/exp06_compare.py:320), [posttrain_runbook.sh:183](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/planner_probes/posttrain_runbook.sh:183).

   Neither generated arm-C nor arm-B command supplies `--bind-input`. Both parse to empty bindings. The comparer requires `train_manifest` and `train_completion` for every `exp06` route; both therefore fail with:

   ```text
   the exp_06 arm binds its training run
   ```

   There is also a literal §6.3 deviation: exp_06 validates only `heading` itself; `train_completion` receives inherited file hashing, without semantic validation or a demonstrated connection to the evaluated checkpoint. The comparer fixture masks this by binding training-evidence names to an evaluation log.

   **Minimal fix:** bind and validate C’s actual pretraining provenance/completion and checkpoint relationship. Define the appropriate historical-evidence branch for fallback B, whose exp_01 training lacks modern completion records. Add an integration test from the exact runbook argv through comparer admission for both arms. **Required before sim launch.**

3. **F3 — should-fix, launch-blocking: G1, HAA, and simulated-evaluation execution do not enforce §6.4’s approved identities.**

   Location: [tools/exp06_mirror_probe.py:568](/home/yixunhu/codespace/xRIR_code/tools/exp06_mirror_probe.py:568), [tools/exp06_haa_pipeline.sh:156](/home/yixunhu/codespace/xRIR_code/tools/exp06_haa_pipeline.sh:156), [tools/exp06_finalize.py:1143](/home/yixunhu/codespace/xRIR_code/tools/exp06_finalize.py:1143), [tools/exp06_eval_launch.py:127](/home/yixunhu/codespace/xRIR_code/tools/exp06_eval_launch.py:127).

   These paths capture identities or check consistency with their selected commit, but do not compare execution inputs against the committed producer approvals. The runbook’s `producer_ok` calls `require_producer`, which checks populated fields, not equality with current code/artifacts.

   In memory, replacing the approved mirror-probe digest and epoch-checkpoint hash with incorrect, well-formed hashes still yielded:

   ```text
   mirror_probe: []
   haa_children: []
   sim_eval: []
   ```

   Current approvals independently match HEAD; that does not implement the plan’s mandatory refusal of unapproved execution.

   **Minimal fix:** enforce the committed producer matrix and compare actual code/checkpoint/heading identities before confirmatory work; retain and revalidate the approval binding. Preserve explicitly diagnostic paths. **Required before G1/HAA/sim launch.**

4. **F4 — should-fix: descriptive exp_04 cylindrical evaluations are admitted as H3 arm B.**

   Location: [tools/exp06_compare.py:84](/home/yixunhu/codespace/xRIR_code/tools/exp06_compare.py:84), [tools/exp06_compare.py:124](/home/yixunhu/codespace/xRIR_code/tools/exp06_compare.py:124), [tools/exp06_compare.py:247](/home/yixunhu/codespace/xRIR_code/tools/exp06_compare.py:247).

   With the committed approvals, the real call equivalent to:

   ```python
   exp06_compare.admit_run(
       Path("ckpt/yaw_aug/eval/cyl_k8_seed42_k0"), "B", approved,
       inputs={}, stats={}
   )
   ```

   **admitted all 6,337 queries**, despite absent checkpoint-role and checkpoint-epoch fields. B’s unrestricted route falls back to `exp04`, then receives the historical missing-epoch exception.

   Section 6.3 registers exp_05 M-tier evaluations or an exp_06 fallback for B. The exp_04 reuse exception belongs to A.

   **Minimal fix:** restrict B to its two registered routes and enforce their respective role/epoch evidence. Add this real descriptive-run refusal regression. **Required before H3 comparison/pages; does not itself block data collection.**

5. **F5 — should-fix: H3 does not implement the registered convergence rule.**

   Location: [tools/exp06_compare.py:485](/home/yixunhu/codespace/xRIR_code/tools/exp06_compare.py:485).

   H3 calls the pinned `paired_compare.convergence` once. That helper accepts identical zero-width intervals, and the exp_06 wrapper never quadruples the bootstrap count after failure. Section 7 explicitly requires zero-width refusal and one quadrupling attempt.

   An in-memory example with five seeds and 24 identical queries per arm produced `rho=0`, interval `[0, 0]`, `passed=True`, and **“non-inferior”**, with no convergence failures.

   **Minimal fix:** add the exp_06 convergence policy around the inherited H3 resampling: refuse zero width, retry once at four times the draws, and withhold unsuccessful verdicts. Record the attempts. Keep `paired_compare.py` unchanged. **Required before H3 comparison/pages.**

**Answers to the requested checks**

**1. Plan fidelity.** The numerical and recipe contracts largely match; F2–F5 are substantive plan-text deviations.

- **§6.1:** inspected the frozen legacy cohort, anchors `0.975/0.85` and `0.435/0.24`, tolerance `0.01`, exact mirrors `(-x,-y,z)`, strict bracketing thresholds, and strict oriented pass thresholds `<0.80` and `<0.50`. Cache-coordinate readback confirms **185 positive-y test microphones**. Failed legacy reproduction or bracketing makes G1 inconclusive. See [mirror_probe.py:154](/home/yixunhu/codespace/xRIR_code/tools/exp06_mirror_probe.py:154), [mirror_probe.py:220](/home/yixunhu/codespace/xRIR_code/tools/exp06_mirror_probe.py:220), and [mirror_probe.py:399](/home/yixunhu/codespace/xRIR_code/tools/exp06_mirror_probe.py:399). I did **not** rerun real-checkpoint anchor inference; the Coder’s recorded `0.9745/0.8510` and `0.4354/0.2382` remain reported evidence.
- **§6.2:** stage-1 rooms and `1000/10/TF32`, per-room stage-2 `200/2/TF32`, K=8, evaluation seed 0, unseeded primary Griffin–Lim, default depth, all three heading-frame initializations, and zero-shot layout match. Parser-default checks and the exact dry run support this.
- **§6.3:** delegation to `run_exp04`, evaluation flags, roles and epoch match. Launch/admission composition fails as F1–F2 describe; B’s admission is too broad under F4.
- **§6.4:** the three-section schema, completion units, reconstructed legacy branch, and dependency matrix exist. Execution enforcement is incomplete under F3.
- **§7:** HAA estimands, finite-across-all-seeds query cohorts, H1/H1b void rules, strict `+0.23 dB`/zero margins, Bonferroni-11 screen, and HAA convergence/retry machinery match. All eleven historical cells reproduce exactly. H3’s ratio estimand and strict `<+3%` bound match, but its convergence implementation fails F5.
- The executed encoder/factory/heading tests also cover the §2.2 symmetry and bit-exactness contracts, including heading cancellation and the **526,336** parameter increment.

**2. CPU composition and admission.**

The requested command:

```bash
bash tools/exp06_haa_pipeline.sh 1 cyl_or:0 zeroshot --dry-run
```

exited 0 and printed **4 jobs, 5 training children, 16 evaluation children, and 25 finalizer calls**, including job specifications and registered directories. No printed GPU command was executed.

The HAA fixtures do use real child/job finalizers and summarizer validators, with cases for live PIDs, missing completions, altered specifications, recipe deviations, non-confirmatory headings, and missing/incorrect side labels. However, their writable filesystem fixtures could not be replayed in this sandbox; I therefore do **not** certify the complete CPU fixture chain as freshly green.

Actual read-only admission succeeded for **all five** reused exp_04 control runs, seeds 42–46, with 6,337 queries each. The requested refusal of descriptive cylindrical B **failed**, as F4 documents.

**3. Simulation argv.**

For both C and fallback B, parser/serializer checks reproduce the registered evaluator flags:

```text
--conditions P --yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols
--batch-size 16 --num-workers 6 --max-samples 0
--gl-seed <s> --threads 4 --log-interval 10 --decomposition-batches 0
```

TF32 is false; C has role `arm`, B `baseline`, both epoch 12, room frame, and no heading. The runbook computes semantic manifest hashes using `tools.reference_manifest.manifest_hash`; seed 42 recomputed as:

```text
9cf5c8a236afa4e38aa111d8658769767543389b169d6b16f7042c092d3faa10
```

**The evaluator argv is correct; orchestration and evidence bindings need F1–F2.**

**4. Approvals.**

Independent `exp06_profiles.compute_code_digests(repo, target_commit)` recomputation matched **all 20 code keys** at `08a03e7`. Shell closures bind the two shell files explicitly.

Relative to `b9f6ebe`, exactly four keys changed:

| Key | Independently confirmed cause |
|---|---|
| `finalize` | Finalizer edits plus changed imported smoke code |
| `smoke` | Its own edits |
| `launch_sh` | Its own shell-file edits |
| `summarize_haa` | Import-closure propagation from finalizer/smoke; summarizer bytes unchanged |

No other cause was found.

Recomputed reused identities match: exp_04 entrypoint `0b05245c…`, writer `15f9c984…`, exp_04 approvals `2e452117…`, exp_05 approvals `293c5683…`, exp_02 stats `98062d76…`, and summary `736f9fb8…`. All five controls carry the expected evaluator/writer identities.

The reconstructed receipt’s path/hash match; its hash is `5a124946…`, and **all 126 enumerated artifact hashes match**. The epoch artifact has epoch 12 and the correct checkpoint hash.

`require_producer` returns no deviations for legacy receipt, mirror probe, HAA children, and sim evaluation. **It also returns none for `compare`: contrary to the question’s parenthetical, §6.4 does not make H3 depend on G1.** The summarizer correctly awaits `artifacts.gate_g1_sha256`.

These successful population/hash checks do not remove F3.

**5. Finalized pretraining.**

For `attempt_20260916T161415`, independently verified:

- `completion.json`: `run_type="full"`, `admissible_arm=true`, diagnostic false, child exit 0, twelve epochs, no approval deviations.
- All five recorded artifact hashes match, including:

  ```text
  epoch_012.pth:
  3df2ceb53d1be1402cb6672f05ded092187f1ba871b53fc1114ca088612296c1
  ```

- History contains epochs **1–12 exactly**, with finite required losses/timings.
- `exp06_recipe.check_all(args)` returns `[]`; normalized exp_01 recipe comparisons also return `[]`.
- Recipe: batch 32 × accumulation 2, TF32, 12 epochs, learning rate `1e-3`, decay `3/0.1`, seed 0, and 9,261 training batches per epoch.
- The bound historical approvals equal the committed bytes at `6dc0b8ec3553c226f0e2e6e1f587eb806f4ff113`.
- `final` points to the specified attempt.
- Launcher and child PIDs are absent from `/proc`; no signal-based checks were used.
- The closed log/receipt agree and end with `EXP06_CHILD_EXIT 0 2026-09-17T23:16:15+00:00`.

[full_evidence:762](/home/yixunhu/codespace/xRIR_code/tools/exp06_finalize.py:762) checks checkpoint keys, dtypes, shapes, and tensor equality against `last.pth["model"]`. I verified the retained hashes and inspected that assertion; **I did not independently load the checkpoints to repeat tensor equality**, or rehash the entire training-data inventory.

**6. Pins and neighbours.**

The requested protected-path diff against `77ef292` is empty, including `sim_to_real`, all twelve exp_03 files, the original trainer, and the named exp_04/05/provenance/statistics tooling. The trainer’s eleven-file import closure also matches the base.

The exp_03 closure recomputes to:

```text
5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48
```

`git diff main...exp06-window --diff-filter=M --name-only` is empty.

Neither neighbouring approvals file was changed by exp_06. Qualification: exp_04’s approvals file differs from `77ef292` because of its **own** approvals-fill commit `fa219ef`; “byte-identical to the base” would therefore be inaccurate for that record. Exp_05’s approvals file is unchanged.

Neighbour test coverage was partial, as listed below; I cannot claim the complete neighbour suites are freshly green.

**7. Test execution and limits.**

All Python checks used the requested interpreter, hidden CUDA, four threads, and disabled bytecode/cache-provider writes.

| Executed check | Result |
|---|---|
| `python -m pytest tests -q -p no:cacheprovider` | Failed before collection: `FileNotFoundError: No usable temporary directory found` |
| `python -m pytest tests/test_exp06_bootstrap.py -q -s -p no:cacheprovider` | **53 passed**; eleven historical cells exact, maximum difference 0 |
| Read-only selection from encoder/factory/heading/probe-align/recipe/eval/mirror-probe tests | **249 passed, 6 skipped, 59 deselected** |
| Read-only selection from exp_03 record, provenance, exp_04 and exp_05 tests | **39 passed, 657 deselected** |
| Read-only selection from approvals/profiles/comparer/HAA-summarizer tests | **66 passed, 151 deselected** |
| `bash -n tools/exp06_launch.sh tools/exp06_haa_pipeline.sh` | Passed |
| `git diff 77ef292..08a03e7 --check -- model tools tests` | Passed |
| Exact HAA dry run and real control-admission checks | Passed as detailed above |

The selected suites total **407 passed, 6 skipped**; this is **not** a full-suite count. Adapted checks used in-memory import/cache handling, `NUMBA_DISABLE_JIT=1`, and deselection of filesystem-writing/process-control fixtures. Those adaptations did not replace the production admission/statistics predicates used for the reported counterexamples.

At cutoff, the Planner’s separate CPU suite log had reached **76% without a terminal summary**. It is external, incomplete evidence. **Full-suite green at HEAD is not established by this review.**

**8. Prior findings.**

The cited prior fixes are present: round-2b preparation/metadata/heading checks; round-3a captured checkpoint bytes, panorama binding and ownership-safe cleanup; round-3b recipe, evidence retention, committed approvals and snapshot corrections; premerge mandatory job ownership; and prelaunch smoke-budget, confinement and job-spec snapshot fixes.

In particular, [load_job_spec:1553](/home/yixunhu/codespace/xRIR_code/tools/exp06_finalize.py:1553) now parses and hashes one buffer and rejects subsequent changes, closing the explicitly deferred snapshot issue.

I found no additional reopened prior should-fix beyond the integration gaps identified here. Classification:

- **Before launch:** F1–F2 for sim; F3 for G1/HAA/sim.
- **Before comparer/results pages:** F4–F5.
- **Reporting-only:** historical commit-size/process disclosures and the optional S1 limitation below.
- G1’s future artifact approval remains an expected prerequisite for summarization.

**9. Optional seeded-phase sensitivity S1.**

Confirmed: [exp06_summarize_haa.py:1228](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:1228) always loads historical A/B through the fixed legacy branch; `--sensitivity` is passed only to new-arm admission. There is no supported loader for seeded re-evaluations of historical A/B.

**S1 as written is not implementable through the existing end-to-end interface without new code.** It remains optional and descriptive, and does not block the primary comparison.

**Exact closure requirements:** fix **F1, F2, F3, F4 and F5**. F4–F5 may follow data collection, but must precede H3 publication. Before launch approval, complete F1–F3 as applicable, refresh affected approval digests, and obtain a green authorized full-suite replay plus the missing integration regressions. The current snapshot is **not approved for G1/HAA/sim launch**.

