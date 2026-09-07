**Reviewer:** OpenAI Codex (codex-cli 0.144.1, `codex exec`, read-only sandbox, model `gpt-5.6-sol`, reasoning effort `ultra`; session 01a077f0-8cf3-72b2-b974-7b276c1f1b27) · **Date:** 2026-09-06 14:32 (−04:00) · **Round:** `full` integrative review of `5bf4640..62c9107` before the first expensive launch · **Briefing:** SOP, plan v2, whole notebook, params, commands, per-round reviews, exp_01 corrected record; prompt `review_prompts/code_full_prompt.md`.

> Planner answer to the question: code/checkpoint digests go into a hash-bound `provenance.json` sidecar written next to each run's artefacts by the launcher (git SHA, checkpoint sha256, manifest hash, command line), so the numerically valid gate runs are kept; the sweep runs get the same sidecar. Verdict handling: blocking items 1–2 → Coder fix round on the summarizer; item 3 → Planner fixes the launcher (new file; the running instance is not edited).

## Summary verdict

**Approve with changes before launch.**

The numerical evaluator path is sound: rotation, P/E isolation, deterministic references, seeded Griffin–Lim, 8,000-sample window, losses, TF32 policy, and canonical padding match the approved recipe or documented amendments.

**I found no blocking numerical defect in the evaluator; the current k=0 evaluations do not need to be rerun for the findings below.** The completed control/cyl artifacts each have 6,337 unique queries, the approved manifest hash, batch 16, TF32 off, finite metrics, exact P=E at k=0, and zero delay flips.

## Blocking-for-launch findings

1. **The k=0 gate is fail-open, and its documented command fails.**

   [tools/summarize_yaw.py:540](/home/yixunhu/codespace/xRIR_code/tools/summarize_yaw.py:540), [tools/summarize_yaw.py:840](/home/yixunhu/codespace/xRIR_code/tools/summarize_yaw.py:840), [tests/test_summarize_yaw.py:857](/home/yixunhu/codespace/xRIR_code/tests/test_summarize_yaw.py:857), [yaw_rotation_degradation_command.md:26](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:26)

   Missing cylindrical/released runs and only one noise run can still produce `gate_pass=True`. Noise runs are not checked for checkpoint, registered seed/hash, full query set, k=0-only grid, batch 16, TF32, or phase seed. An arbitrary high-variance “noise” run can inflate the band and mask any discrepancy.

   The command also defaults labels to `gate_control/gate_cyl/gate_released` but supplies exp_01 mappings for `control/cyl`, which is rejected.

   **Fix:** add `--labels control cyl released`; implement a fail-closed gate validator requiring all three roles, exactly the two registered control noise manifests, complete/aligned arrays, exact settings, P=E at k=0, and `delay_flips["0"] == 0`.

2. **Canonical `full` mode does not actually pin the preregistration.**

   [tools/summarize_yaw.py:520](/home/yixunhu/codespace/xRIR_code/tools/summarize_yaw.py:520), [tools/summarize_yaw.py:620](/home/yixunhu/codespace/xRIR_code/tools/summarize_yaw.py:620), [tools/summarize_yaw.py:663](/home/yixunhu/codespace/xRIR_code/tools/summarize_yaw.py:663), [tools/summarize_yaw.py:784](/home/yixunhu/codespace/xRIR_code/tools/summarize_yaw.py:784)

   Full mode accepts any caller-supplied manifest hash, any manifest seed, batch size 1/8 despite known shape drift, epoch 11 under the expected directory, and altered alpha/threshold/equivalence margin/bootstrap count. A read-only adversarial probe confirmed `validate_full()` returned no reasons for an arbitrary hash, seed 9, batch 8, and epoch-11 checkpoints.

   **Fix:** pin seed-0 hash `47637a55…153d`, batch size 16, exact epoch-12 checkpoint paths/content hashes, `alpha=.05`, threshold `.10`, equivalence margin `.02`, bootstrap seed 0, and `n_boot >= 20000`. Deviations belong only in exploratory mode. Also enforce cross-model delay-audit equality and zero at k=0; currently only key presence is checked.

3. **The launcher masks failures, permits stale output, and does not enforce the gate.**

   [launch_yaw_sweep.sh:13](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep.sh:13), [launch_yaw_sweep.sh:23](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep.sh:23)

   `run_one` returns the status of its final `echo`, bare `wait` discards child failures, and `*_DONE` prints unconditionally. Fixed output directories can retain old atomic JSON after a failed retry, while `sweep` can run without a passing gate.

   **Fix:** propagate every evaluator status, explicitly check both background PIDs, use unique/refused output directories, and require a validated, hash-bound `gate_stats.json` before sweep. Record/assert Git SHA `62c9107` and checkpoint SHA-256s; evaluator metadata currently records paths only.

## Non-blocking

- [eval_yaw_rotation.py:588](/home/yixunhu/codespace/xRIR_code/eval_yaw_rotation.py:588) computes the k=32 cylindrical decomposition during a nominal k=0-only run. It occurs after the batch’s gate metrics and does not mutate model state or RNG, so it does not invalidate them. Use `--decomposition-batches 0` for gates or require k=32 in `yaw_cols`.
- [tools/summarize_yaw.py:64](/home/yixunhu/codespace/xRIR_code/tools/summarize_yaw.py:64) should use `allow_nan=False`.
- T14 is strong and covers all three trained checkpoints. T15 covers only four queries with a random SimpleViT; T16 uses a partial grid. No test executes the documented gate command or adversarially checks registered full-mode parameters.
- `metrics_yaw.json` is never loaded or reconciled with the per-sample source-of-truth JSON.

## Residual risks to record in the analysis

- The k=0 band is a coarse sanity heuristic: two control-only alternate draws are used for every model and do not separately cover exp_01’s unseeded phase noise or the intentional TF32/batch-shape changes.
- Angle-dependent invalid acoustic metrics create survivor-selection risk; report prominently if newly-invalid fractions are material.
- CUDA deterministic algorithms and complete dataset/checkpoint/code-content bindings are not presently pinned.
- E−P includes tiny direct-energy scaling drift as well as audited integer-delay flips.

## Questions

Should code/checkpoint digests be embedded in evaluator metadata—which would require rerunning the gate—or recorded in a hash-bound sidecar for the completed gate artifacts? The latter preserves the numerically valid current runs.
