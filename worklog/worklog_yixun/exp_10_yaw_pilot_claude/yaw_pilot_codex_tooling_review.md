# exp_10 yaw_pilot — Codex review of the Planner tooling (record generators, chain / queue / finish scripts)

**Reviewer identity:** OpenAI Codex `gpt-6-astra` at `ultra` (codex-cli 0.154.0, `codex exec -c 'model_reasoning_effort="ultra"' -s read-only -C /home/yixunhu/codespace/xRIR_code --skip-git-repo-check`, stdin closed) · **Reviewed:** `yaw_pilot_results_assets/{make_results_md,make_results_html,make_backend_table}.py`, `scripts/{exp10_arm_chain,exp10_gpu_queue,exp10_released_k1_after_queue,exp10_finish}.sh` at main `d2b2db1` · **Prompt:** `review_prompts/tooling_review_prompt.md` · **Raw log:** `yaw_pilot_2026-09-27_03:36:17_codex_tooling_review.log` · **Date:** 2026-09-27 · **Verdict:** request changes (findings 1–11 before record closure; 12–13 nits)

**Verdict: request changes before using these scripts to close the record.**

Paths below are relative to `worklog/worklog_yixun/exp_10_yaw_pilot_claude/yaw_pilot_results_assets/`.

1. **Blocker — finish accepts missing or unrelated validation evidence.**  
   `scripts/exp10_finish.sh:16–17,28–29` checks complete/count/device and an unbound `check_online.ok`; full parity is optional, and probe controls/parity are never checked. A scratch reproduction reached `FINISH DONE` with missing full parity reports, failed probe controls, and online reports naming another run.  
   **Minimal fix:** require and validate the full/probe evidence for each prescribed arm, bind it to current execution/protocol/input hashes, and require both probe and full parity for K=8. Validate the CPU record separately while preserving its documented parity failure.

2. **Blocker — failed assembly can certify stale assets.**  
   `scripts/exp10_finish.sh:4,23,27–35` reuses `generated/` and does not stop on copy or checksum failures. In the reproduction, a failed probe-metadata copy retained old destination contents; an unrelated `stale_summary_interim.json` also survived. The script returned success and hashed both into `SHA256SUMS`.  
   **Minimal fix:** fail on every assembly error; build an explicit asset set in fresh staging, verify completeness and checksums, then replace the published output.

3. **Should-fix — backend comparison does not enforce matching protocols or populations.**  
   `make_backend_table.py:47–61` checks only checkpoint, manifest and GL seed. The real 272-query GPU probe versus 6,337-query CPU full summary was accepted with “only the inference device differs.” Supplying the same CPU execution twice also succeeded. Batch size, implementation hash, precision, library and query-list mismatches were accepted.  
   **Minimal fix:** require CUDA/CPU roles, distinct executions, identical query populations, matching estimator/protocol settings except the allowed backend difference, and valid summary input bindings.

4. **Should-fix — renderers accept mixed summaries, reports and figures.**  
   `make_results_html.py:128–142` copies neighbouring figures without binding them to the selected summary; both renderers load supplemental reports without checking their association. Replacing the combined figure with the CPU figure produced GPU tables beneath an “All arms” CPU plot. Attaching cylindrical parity to the control arm also succeeded. Both renderers accepted a deliberately incorrect summary input hash.  
   **Minimal fix:** verify summary input bindings and supplemental execution identities; regenerate figures from that summary or verify an asset manifest bound to its digest. Hashing arbitrary copied bytes does not establish their consistency.

5. **Should-fix — canonical qualifications disappear from the reports.**  
   `make_results_md.py:61–62` and `make_results_html.py:148–149` iterate the `notes` dictionary’s keys, printing `band`, `broader_population`, `gl_free`, and `pipeline` instead of their explanatory values. Markdown additionally omits `metric_qualifications`, including the distinct T60 denominators.  
   **Minimal fix:** render note values and the canonical metric qualifications in both outputs.

6. **Should-fix — HTML hides room status beside reportable multiples.**  
   `make_results_html.py:69–74` returns the multiple before appending `room_status`; the separate status column contains only query status. Actual example: control EDT at 90° displays **17.8× [8.81, 95.3]**, omitting the canonical room status **denominator uncertain**.  
   **Minimal fix:** always display separate query and room statuses, including reportable cells.

7. **Should-fix — failed online checks do not stop the chain.**  
   `scripts/exp10_arm_chain.sh:24` checks process exit status, but the approved comparator exits successfully when its JSON reports `ok=false`. The reproduction launched full and printed `ARM DONE` despite failed online checks.  
   **Minimal fix:** require the newly generated report’s `ok` to be Boolean `true` after each check. Keep the subsequent controls/parity conjunction unchanged.

8. **Should-fix — parity can be bypassed for any arm.**  
   `scripts/exp10_arm_chain.sh:7,34–36` defaults the predecessor argument to `none` without restricting that exemption to `released_k1`. Omitting it for `control_k8` yielded `parity_ok=n/a` and launched full.  
   **Minimal fix:** resolve predecessors from the validated arm table; permit `none` only for `released_k1`.

9. **Should-fix — existing partial outputs are not protected.**  
   `scripts/exp10_arm_chain.sh:19` refuses only an existing `meta.json`. A directory containing an existing waveform but no metadata was accepted and overwritten. Concurrent chains also lack an atomic reservation.  
   **Minimal fix:** refuse existing output directories and atomically reserve each stage’s directory before evaluation.

10. **Should-fix — source changes can bypass the cleanliness protection.**  
    `scripts/exp10_arm_chain.sh:15,30–38` exempts all of `worklog/`, including these executable scripts, and checks cleanliness only once. Both dirty tooling under `worklog/` and an evaluator becoming dirty after the probe allowed full launch.  
    **Minimal fix:** exempt only intended narrative/log outputs and verify relevant source hashes before each stage against the probe’s implementation. This can accommodate the documented unrelated HEAD changes.

11. **Should-fix — Markdown lacks complete supplemental-input hashes.**  
    `make_results_md.py:89,101,103–104` truncates probe/parity hashes and records no check-online hash.  
    **Minimal fix:** add full SHA256 provenance for every supplied input, matching the HTML footer.

12. **Nit — copied supplementary results are not linked.**  
    `make_results_html.py:132–135,160–164` copies full tables/JSON/CSV but prints filenames without links; Markdown likewise supplies no table link. The finish-generated backend comparison is also absent from the page’s navigation.  
    **Minimal fix:** add relative links to the full tables, canonical data and CPU/backend record so exclusions, broader-population G and secondary results are discoverable.

13. **Nit — wrapper exit statuses hide arm failures.**  
    `scripts/exp10_gpu_queue.sh:18–21` returned success after all mocked arms failed; `scripts/exp10_released_k1_after_queue.sh:8–9` logged failure but returned success.  
    **Minimal fix:** preserve aggregate/child failure status. Continuing independent arms can remain intentional.

**Verified**

- Executed both presentation generators with the real three-arm interim summary and corresponding probe/parity/online inputs; inspected the generated Markdown and HTML. Outputs: [Markdown](/tmp/claude-1013/-home-yixunhu-codespace-xRIR-code/279323f7-0bed-4b41-8631-99735eadc6f1/scratchpad/codex_tooling_review/generators/interim.md), [HTML](/tmp/claude-1013/-home-yixunhu-codespace-xRIR-code/279323f7-0bed-4b41-8631-99735eadc6f1/scratchpad/codex_tooling_review/generators/interim.html).
- All four scripts passed `bash -n`; all three generators parsed under Python 3.8 grammar.
- Fake failing probes confirmed controls=false, parity=false, both=false, and evaluator process failure prevent full launch. Existing metadata and initially dirty source outside `worklog/` were correctly refused. [Shell results](/tmp/claude-1013/-home-yixunhu-codespace-xRIR-code/279323f7-0bed-4b41-8631-99735eadc6f1/scratchpad/codex_tooling_review/shell/results.json).
- Finish preflight correctly rejected incomplete metadata, CPU device, 272-query count and `check_online.ok=false`. Finish orchestration was exercised only in relocated scratch fixtures with expensive summarization stubbed; the three record generators ran unmodified.
- Both manifest hashes, queue arm parameters and all 12 exp_03 source pins match. Actual checkpoint hashes match exp_03. **Documentation discrepancy:** plan §3’s released checkpoint prefix `1762c702…` differs from the actual and exp_03-bound `6cdb0276…`.
- Scratch `SHA256SUMS` verified all 38 listed assets—including the reproduced stale asset, demonstrating the distinction between checksum coverage and correct assembly.
- No GPU execution, repository writes, or access to live `ckpt/exp10/released_k8_all`.

Findings **1–11 should be resolved and reverified before record closure**.
