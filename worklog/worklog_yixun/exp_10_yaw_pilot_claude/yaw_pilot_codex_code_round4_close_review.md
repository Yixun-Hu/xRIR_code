# exp_10 yaw_pilot — Codex code review, round 4 (close; Planner tooling fixes)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="ultra"' -s read-only -C /home/yixunhu/codespace/xRIR_code_wt10 --skip-git-repo-check`, stdin closed) · **Reviewed:** `1ef60f0..1507e05` (11 commits, Claude Opus 5) · **Prompt:** `review_prompts/code_round4_close_prompt.md` · **Raw log:** `yaw_pilot_2026-09-27_04:38:06_codex_code_round4_close.log` · **Date:** 2026-09-27 · **Verdict:** request changes; not approved for merge; finish not yet usable (blocking 1–7)

**Round-4 verdict: request changes. Merge verdict: not approved. The finish script must not yet be used to build the final record.**

Reviewed `exp10-yaw-pilot`, `1ef60f0..1507e05148a22b3a93f0beadc6255b94cec7b2c9`. Seven findings remain blocking.

Paths below are relative to `worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/`.

1. **Blocker — finish still accepts GPU validation reports for the CPU record.**  
   `scripts/exp10_finish.sh:49`, `validate_runs.py:254`. The unconditional relocation alias accepts reports naming `ckpt/exp10/released_k8_all`, which now identifies the distinct GPU execution. Copying the scratch GPU reports into the CPU record reached `FINISH DONE`, exit **0**, and published CPU parity **true**. MAIN’s 04:39 entry confirms the CPU reports were regenerated with their current paths; this exception is obsolete.  
   **Minimal fix:** remove the relocation argument from finish. Any retained relocation support must require matching execution or payload identity. [Reproduction](/home/yixunhu/codespace/xRIR_code_wt10/.review-round4/finish/11_gpu_reports_in_cpu/results.json).

2. **Blocker — a publication failure leaves mixed generations after deleting the rollback copy.**  
   `scripts/exp10_finish.sh:136–138`. The previous assets are deleted before both reports are published. Injecting failure into the final Markdown rename returned **43**, leaving **new assets, new HTML, old Markdown**. Cleanup incorrectly claimed the publication was unchanged.  
   **Minimal fix:** retain backups of assets and both reports until every publication succeeds; restore all three on failure. [Reproduction](/home/yixunhu/codespace/xRIR_code_wt10/.review-round4/finish/10_final_publication_fails/finish.log).

3. **Should-fix — the original foreign-figure substitution still succeeds.**  
   `make_results_html.py:213–229`, `validate_runs.py:452`. Filename filtering excludes a foreign arm’s basename, but does not bind image contents to the summary. Replacing the interim combined PNG with the real CPU-only combined PNG again returned **0** and published it under “All arms” beside three GPU-arm tables. The new test checks an extra filename, not this original substitution.  
   **Minimal fix:** regenerate figures from the validated summary, or verify an asset manifest binding their digests to that summary’s digest. [Generated reproduction](/home/yixunhu/codespace/xRIR_code_wt10/.review-round4/root/render/cpu_figure_substitution/page.html).

4. **Should-fix — renderer binding checks accept incomplete inputs and contradictory supplemental identities.**  
   `validate_runs.py:301–329,411–440`. Keeping only the control input binding while retaining all three rendered arms succeeds in both renderers. Separately, parity and online reports with the correct path but deliberately wrong `execution_id`, `protocol_id`, or `per_sample_sha256` also succeed.  
   **Minimal fix:** require exactly one matching input binding per rendered arm; compare its identity with the arm and live metadata. Validate every supplied supplemental identity, reusing `bind_report` where appropriate. [Reproductions](/home/yixunhu/codespace/xRIR_code_wt10/.review-round4/root/render/results.json).

5. **Should-fix — backend comparisons still accept different statistical protocols.**  
   `make_backend_table.py:98–126`. Run metadata is compared, but summary `alpha`, `n_boot`, seeds and convergence settings are not. Using the unmodified canonical summarizer on identical synthetic EDT observations, GPU `alpha=.05` produced “denominator uncertain”; CPU `alpha=.5` produced “defined.” The backend table accepted both and claimed only the inference device differed. Matching `.05` controls produced identical statuses.  
   **Minimal fix:** require matching summary statistical settings and record them among the verified fields. [Canonical reproduction](/home/yixunhu/codespace/xRIR_code_wt10/.review-round4/backend/canonical_alpha/evidence.json).

6. **Should-fix — a rejected concurrent chain can overwrite an active chain’s source baseline.**  
   `scripts/exp10_arm_chain.sh:27,73–76,84`. Source pins are written to a shared arm filename before reserving the probe directory. Reproduced: pause chain A after its probe; commit an evaluator change in scratch; start chain B. B overwrites the pins, then refuses the existing probe with **2**. A accepts the replacement pins, launches full with changed evaluator bytes, and finishes with **0**.  
   **Minimal fix:** keep an immutable baseline per invocation, acquiring the arm reservation before publishing shared state, or hold an exclusive arm lock. [Reproduction](/home/yixunhu/codespace/xRIR_code_wt10/.review-round4/shell/pin_race_results.json).

7. **Should-fix — Markdown corrupts the T60 absolute qualification.**  
   `make_results_md.py:77–78`. Absolute-value `|` characters are inserted unescaped into a pipe table. Rendering the regenerated real Markdown produces Δ = “change in” and G = “T60(prediction) − T60(GT)”; the actual G definition and units disappear.  
   **Minimal fix:** escape pipes in table cells and test the rendered qualification cells. [Rendered reproduction](/home/yixunhu/codespace/xRIR_code_wt10/.review-round4/root/qualification_table_rendered.html).

## Verified

All execution used **Python 3.8.20**, with GPUs disabled or mocked. Tracked files remained unchanged; writes were confined to `.review-round4/`. The live `ckpt/exp10/released_k8_all` was not accessed.

- **94 tests passed:** `python -m pytest tests/test_exp10_record_tools.py -q -p no:cacheprovider`, with scratch `--basetemp`.
- All four shell scripts passed `bash -n`; all six scoped Python files passed `py_compile`.
- `git diff --check` passed for the round diff, working tree and rehearsed merge.
- Regenerated and inspected the real interim [HTML](/home/yixunhu/codespace/xRIR_code_wt10/.review-round4/root/render/interim/page.html) and [Markdown](/home/yixunhu/codespace/xRIR_code_wt10/.review-round4/root/render/interim/page.md).
- Finish dry-run used permitted real metadata/report copies, small bound synthetic payloads, and the Coder’s summarizer stub. At the production **6,337-query** requirement, missing/incomplete released full runs were refused; the complete synthetic case published **46 assets**, all checksums verified, and stale assets disappeared. [Fixture details and limits](/home/yixunhu/codespace/xRIR_code_wt10/.review-round4/finish/12_real_metadata_6337/README.txt).
- Rehearsed `git merge --no-ff --no-commit` onto MAIN **`28fb4f1b929d74ae40cba7da24438e6d258c22d0`** in a disposable clone. Merge was clean; **exactly the ten scoped files changed**. The clone was deleted. [Merge evidence](/home/yixunhu/codespace/xRIR_code_wt10/.review-round4/root/static_merge.json).

Every prior reproduction was rerun:

| Prior finding | Observed result |
|---|---|
| 1 — unbound evidence | Missing reports, wrong-run online evidence and failed controls refused; CPU alias bypass remains, finding 1 above. |
| 2 — stale assets | Copy/summarizer/missing-file failures preserve the existing publication; successful assembly excludes stale assets. Final-publication rollback remains broken, finding 2. |
| 3 — backend protocols/populations | Same execution, 272 versus 6,337 queries, population, batch, implementation, library and precision mismatches refused. Statistical mismatch remains, finding 5. |
| 4 — mixed renderer inputs | Wrong-arm supplements and bad hashes refused; foreign basenames excluded. Original figure substitution and binding gaps remain, findings 3–4. |
| 5 — notes/qualifications | Note values and T60 denominators now appear in both outputs; Markdown T60 absolute row remains broken, finding 7. |
| 6 — room status | **Closed:** control EDT 90° shows **17.8× [8.81, 95.3]** beside **denominator uncertain** room status. |
| 7 — online false | **Closed:** probe/full online false exits **4**; parity false exits **5**. |
| 8 — predecessor bypass | **Closed:** omission resolves the table; K=8 `none`, wrong predecessors and unknown arms are refused. |
| 9 — partial output | **Closed:** waveform-only directory refused with **2**, preserving its contents. |
| 10 — source changes | Dirty tracked worklog script refused with **3**; ordinary between-stage evaluator edit refused with **8**. Concurrent pin replacement remains, finding 6. |
| 11 — truncated hashes | **Closed:** complete supplemental hashes verified in both provenance sections. |
| 12 — missing links | **Closed:** tables, JSON, CSV, CPU record and backend links resolve in both published scratch reports. |
| 13 — wrapper exits | **Closed:** queue attempts all arms and returns nonzero on failure; after-queue wrapper preserves child failure. |

**Exact blocking list: findings 1–7 above.** Round 4 remains open; the clean merge rehearsal does not establish approval. **Not approved for merge or for using finish to build the final record.**
