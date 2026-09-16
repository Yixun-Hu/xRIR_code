**Reviewer:** OpenAI Codex (codex-cli 0.154.0, model gpt-6-astra, reasoning ultra, `codex exec`, read-only sandbox) · **Date:** 2026-09-16.

**Verdict: request changes.** A4 is acceptable, and the SHA, HAA sequencing, and launcher detachment corrections are sound. Two gate fixes remain.

Reviewed all 49 runbook lines and the A4 amendment against main `b9f6ebe6c853471b4d31ea6406fcda2eaedf17c2`. Runbook SHA256: `f77d06c1cbb60d51d3d6993f131889f6ec2fe8312064947bdc32520c97365265`.

1. **Blocker — GPU-query failure can produce “GATE OK.”**  
   [launch_day_runbook.sh:23](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/planner_probes/launch_day_runbook.sh:23), also line 25.

   `|| true` discards failure of the compute-process query. Failure of the memory query inside `log "$(…)"` is also masked by the successful `log` command. With query errors on stderr and empty stdout, I reproduced **exit 0 and “GATE OK”**, including when both queries failed.

   This does not establish that GPU 1 is free. The later full/probe preflight protects those modes, but smoke preflight deliberately permits co-tenancy, and the HAA smoke invokes no equivalent GPU gate.

   **Minimal fix:** explicitly check each query’s exit status, retain its error output, and refuse with exit 2 on failure. Alternatively, delegate the gate to the existing `exp06_finalize.py preflight --mode full` CLI, which already refuses an unqueryable GPU.

2. **Should-fix, required for this close review — approvals refusals exit 1, not the specified 2.**  
   [launch_day_runbook.sh:15](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/planner_probes/launch_day_runbook.sh:15), through line 22.

   `SystemExit('REFUSE: …')` exits **1**; exceptions from approval loading or `require()` likewise exit 1. `errexit` now correctly stops execution, but does not implement the requested exit-2 refusal contract. Executing the exact heredoc with a substituted trainer digest reproduced exit 1.

   **Minimal fix:** explicitly handle failure of the Python/tee pipeline, log the refusal, and exit 2. Delegating to the canonical preflight CLI also resolves this.

3. **Nit — gate and fallback fixture generation inherit ambient thread settings.**  
   [launch_day_runbook.sh:6](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_06_oriented_cyl_claude/oriented_cyl_results_assets/planner_probes/launch_day_runbook.sh:6), also line 29.

   Training launcher children and HAA commands specify `OMP_NUM_THREADS=8`, but the gate and standalone fixture fallback do not. **Minimal fix:** export the registered production thread limit before the first Python invocation. The reviewer’s checks used four threads throughout.

**Disposition of the previous findings**

- **Full SHA: resolved.** All three launcher dry-runs carry the complete SHA. Actual preflight accepts it for smoke/probe/full with only the GPU-query function mocked as idle; abbreviated SHA still refuses.
- **HAA completion conflict: resolved for rung 4 by A4.** Unfinalised diagnostics are adequate to demonstrate execution and output production within budget. Exclusive attempt directories, diagnostic receipts and provenance are retained; no `completion.json` or job admission results. The finalised HAA diagnostic path and `load_job_spec` fix remain required after pretraining finalisation and before the HAA stage.
- **Failure propagation: substantially resolved.** Dirty tree and occupied GPU refuse with exit 2. Fine-tune failure, failed receipt assertion, missing `best.pth`, and directory collision all prevent evaluation. Findings 1–2 above remain.
- **Detachment: resolved.** The whole supervisor receives `nohup setsid`, closed stdin, a record log and PID file. Its inner `setsid` does not inherently fork away from the PID being waited on in this noninteractive shell. A file-free process analogue verified distinct supervisor/child sessions, matching recorded PID handles, and sink drainage through a delayed descendant’s final write before both waits completed. No hangup signal was sent.
- **Logging: substantially resolved.** New outer logs use local `YYYY-MM-DD_HH:MM:SS`, and gate output is retained. Delegated child logs retain the existing UTC naming convention. The 20-second full check establishes supervisor liveness; the operator must still confirm preflight success, provenance and the first optimizer step.

**Verification performed**

- `bash -n` on the runbook and launcher: passed.
- `git diff --check`: passed.
- `git diff --no-index /dev/null <runbook>` and the plan’s working diff: every added line inspected.
- `git diff main...exp06-window --diff-filter=M --name-only`: empty.
- Launcher `smoke`, `probe`, and `full --dry-run`, using the full reviewed SHA: **3/3 exit 0**.
- Real preflight with mocked idle-GPU queries: **3/3 accepted**, no approval deviations; short-SHA negative controls: **3/3 refused**.
- All **20 approved code digests**, four heading artifact hashes, and **41 protected source paths** verified. Exp_03 and original-trainer closure digests remain unchanged.
- Six isolated gate scenarios exposed the two findings above.
- Seven isolated HAA sequencing scenarios: expected success/refusal behavior in every case.
- Thirteen in-memory budget checks: passed, including exact 300-second/3-GiB boundaries and just-over-budget rejection; no watchdog armed.
- Read-only pytest selection covering profiles, recipe, trainer, HAA, launcher, unknown smoke entry, and the exp_03 pin, with `-q --capture=sys -p no:cacheprovider --tb=short`: **157 passed, 67 deselected**, 70.58 seconds. Deselected cases require writable fixtures. Numba JIT was disabled and an in-memory Matplotlib cache accommodation was used.

The notebook now explains the earlier failed branch replay as stale approval-record/schema mismatch and records 62 passing focused tests on main. Its definitive full replay remains pending; that green result is still required before launch.

No files were modified, RIR arrays loaded, processes signalled, or GPU work started.

**Exact blocking fix list for this runbook round: findings 1 and 2.** Finding 3 is nonblocking. These fixes can remain entirely in the runbook; A4 does not require changing the training-bound finalizer before pretraining.
