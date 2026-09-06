# Experiment SOP (portable)

A generalizable standard operating procedure for AI-assisted research experiments. Drop this file into any project and reference it from that project's `CLAUDE.md` (e.g. "Follow `worklog/experiment_SOP.md` for all experiment work"). Written for Claude Code, but the roles are model-agnostic.

## Roles (three-model separation of duties)

| Role | Who | Duty |
|---|---|---|
| **Planner / Analyst** | The main-session model (strongest reasoning tier; currently Claude Fable 5) | Writes plans, judges reliability, writes analyses. Does NOT write implementation code directly. |
| **Coder** | A subagent on the strongest coding tier at max effort (currently Claude Opus 4.8, max effort) | Implements exactly what the approved plan specifies. |
| **Reviewer** | **The opposite model family from the Coder** (mandatory cross-model review; see reciprocity note below). | Reviews all newly written code; review is saved, not just read. If the reviewer is unavailable, say so — never silently substitute. |

> **Reviewer reciprocity — no model reviews its own code.** If the main session (Planner/Coder) is **Claude**, the Reviewer is **OpenAI Codex** (`codex mcp-server`; CLI fallback `codex exec`). If the main session is **Codex**, the Reviewer is **Claude Opus 4.8 at max effort**, invoked via the `claude` CLI. The Coder and Reviewer must always be different model families, so review is genuinely independent.

> **Reviewer briefing — load context before judging.** Every review prompt (plan or code) must direct the Reviewer to read, before reviewing: (1) this SOP and all `worklog/worklog_<username>/announcement/` directives; (2) the experiment's `plan_*.md` and `_worklog.md` notebook (what was decided and why, including plan amendments); (3) the results/analyses of the PRIOR experiments the work builds on (e.g. baseline numbers and noise floor); (4) a one-paragraph statement of what the current Coder round was tasked to do and what is explicitly out of scope for this round. A reviewer without this context produces generic reviews, flags out-of-scope "gaps", and misses violations of experiment-specific decisions.

## Directory layout

All experiment bookkeeping lives in `worklog/` at the repo root, **namespaced per user** (rule updated 2026-07-12):

- `worklog/experiment_SOP.md` — **this file lives directly under `worklog/`, NOT inside any user namespace** (it is shared/portable across users).
- `worklog/worklog_<username>/` — one namespace folder per user; ALL of that user's experiment logging goes inside it (in this project `<username>` = `yixun` → `worklog/worklog_yixun/`). Every logging path that previously sat directly under `worklog/` now sits under `worklog_<username>/`:
  - `worklog/worklog_<username>/announcement/<NN>_<topic>.md` — standing directives from the user. **Read every announcement before planning or running anything.** New standing instructions get the next number.
  - `worklog/worklog_<username>/exp_<NN>_<exp name>_claude/` — one folder per experiment, `<NN>` zero-padded and sequential.
  - `worklog/worklog_<username>/archive_<reason>_<date>/` — archived/superseded code (see Development discipline).

## Per-experiment artifacts (in lifecycle order)

Inside `worklog/worklog_<username>/exp_<NN>_<exp name>_claude/`:

1. `<exp name>_<username>_query.md` — the user's driving queries: each one verbatim, plus a summary, the user's assumption/hypothesis, and why the experiment needs to run. Started at scaffold time, appended as new queries arrive. (`<username>` is the same name as in the `worklog_<username>` namespace folder — here `yixun`, e.g. `<exp name>_yixun_query.md`.)
2. `plan_<exp name>.md` — written by the Planner BEFORE any code: the English plan AND the planned code laid out per file (each existing file to edit, each new file to create). Surfaced for user approval before implementation.
3. `<exp name>_<reviewer>_plan_review.md` — the Reviewer's review of the PLAN, before user approval and before any implementation: method soundness, hidden assumptions, missing tests/controls, run design. Same reviewer-naming and identity-header rules as the code review below. Planner addresses the findings (revising the plan) before the user signs off.
4. *(the implementation itself)* — the code, written by the Coder subagent per the approved plan. No dedicated markdown artifact; this step is the source-code changes.
5. `<exp name>_<reviewer>_code_<marker>_review.md` — **one small review per Coder round, not one big review at the end**: after EVERY round of Coder output (a TDD red→green cycle, a code snippet, or a coherent commit group), the Reviewer reviews that round's commits BEFORE the next round starts. `<reviewer>` names the reviewing model (`codex` / `opus`); `<marker>` identifies the round — the TDD-cycle or snippet name, or the short name of the main commit (e.g. `<exp name>_codex_code_cyl_pose_review.md`, `<exp name>_codex_code_dispatch_review.md`). **A round is CLOSED only when its full loop has run: code written → review returned → strengthening/fixes applied by the Coder for every blocking finding → fixes re-verified (tests green; re-review or Planner verification of the fix) → loop outcome logged in `_worklog.md` with the fix commit SHAs.** No new round starts, and no run launches, while any round is open. Nits may be batched into the next round; blocking findings may not. A final integrative review of the whole experiment diff (marker `full`) still runs before the first expensive launch. Record "N/A — no code written" for code-free experiments. Every review file MUST open with a header giving the reviewer's exact identity: model/product name and version, invocation method, and review date — e.g. `**Reviewer:** OpenAI Codex (codex-cli 0.144.1, \`codex exec\`, read-only sandbox) · **Date:** 2026-07-10`. **Reviews (plan and code) always use the Reviewer family's strongest available model at its highest reasoning setting** — currently `gpt-5.6-sol` at Extra High (`-c model_reasoning_effort=xhigh`) for Codex (per Yixun 2026-07-10; supersedes `gpt-5.5`; requires codex-cli ≥ 0.144), Claude Opus 4.8 at max effort for Claude.
6. `<exp name>_params_set_up.md` — full hyperparameters/configuration, written at launch.
7. `<exp name>_command.md` — exact reproduction command(s). **Rule: EVERY experiment/run launched gets its command added to this file AT LAUNCH TIME, not retroactively** — including reruns, amended iterations, probes that produce run artifacts, and diagnostics; failed/superseded runs stay in the file (marked as such) because reproducing the failure is part of the record. The notebook records *why*, this file records *how to reproduce*.
8. `<exp name>_worklog.md` — an append-only, timestamped **lab notebook: one entry per action**, not per run. Started at scaffold, appended continuously through implementation, debugging, and every launch. This is where the validation ladder, parity audit, and failure triage (below) get recorded as they happen. Entry format in **Worklog entry template** below. Complements `_results.md` (final numbers) and `_analysis.md` (final judgment) by capturing the decision/debug trail in between.
9. `<exp name>_<YYYY-MM-DD_HH:MM:SS>.log` — ALL terminal output, one timestamped log per run (tee/redirect every training/eval command into the folder). Aborted runs keep their log, renamed with an `_ABORTED_<reason>` suffix.
10. `<exp name>_results.md` — results, appended as runs finish.
11. `<exp name>_analysis.md` — written by the Planner after results land: analyze all code + configuration + results; judge whether the result is reliable; state the outcome and the recommended next step.
12. `commits_<exp name>.md` — SHA + one-line description of every commit belonging to this experiment.
13. `<exp name>_<idx>_results.html` — **rich HTML results visualization, produced after the experiment finishes** (alongside, not replacing, `_results.md`): tables, equations (MathJax/KaTeX or clean HTML math), figures/plots, color-coded verdict blocks, run-trajectory charts — use the full expressive range of HTML/CSS to make the results maximally human-readable. `<idx>` is chosen by the author to separate distinct parts of the visualization when one page would overload (e.g. `..._01_results.html` for gate tables, `..._02_results.html` for mechanism decomposition); a single `_01_` page is fine for simple experiments. **All sourced assets (generated plot images, data extracts, JS/CSS) live in a corresponding folder inside the experiment directory** (e.g. `exp_<NN>_<exp name>_claude/<exp name>_results_assets/`) and are referenced relatively, so the HTML renders offline from a repo checkout. Charts follow the project's visualization standards (load the dataviz guidance before writing chart code). The HTML must be self-consistent with `_results.md` numbers — it is a presentation layer, never a place where numbers appear first.

## Worklog entry template

Each `_worklog.md` entry is one action, headed by an ISO-8601 timestamp in **local time with UTC offset** (same clock as the run-log filenames) and a short title, then these fields. Use the full set for substantive actions (code, launches, fixes); a lightweight **Goal / Result / Analysis / Next** is enough for routine monitoring checks.

`## <YYYY-MM-DDThh:mm:ss±hh:mm> — <short title>`

- **Goal** — what this action is for. *(every entry)*
- **Hypothesis** — the belief being tested, stated *before* the evidence. *(only when testing something)*
- **Change** — exact files/behavior touched. *(when code changed)*
- **Version Control** — branch, `base_commit`, `implementation_commit`, push/pull, changed_files. Every SHA inline.
- **Command / Validation** — exact commands (or static checks), job ids, run dirs, log + artifact paths.
- **Acceptance criteria** — the exact conditions that count as success, written *before* a launch (see **Running & failure discipline**).
- **Result** — status (`passed` / `partial` / `in_progress` / `launched` / `fix_ready`) + metrics/artifacts + key evidence.
- **Analysis** — interpretation; in particular classify any failure as *infrastructure vs. real bug*.
- **Next** — the immediate next step.

## Development discipline

- Develop **commit by commit, experiment by experiment** from a known-good base commit. No long-lived uncommitted state: an experiment concludes by committing its code and its worklog folder.
- **Each commit generally < 200 changed lines of code.** Several small commits per experiment are preferred over one large one. Log every SHA in `commits_<exp name>.md`.
- **Test-driven development (mandatory for all new functions)** — per [TDD](https://en.wikipedia.org/wiki/Test-driven_development): for each small function, write its test FIRST (pytest, in a dedicated `tests/` folder — **ask the user where to place it** the first time tests are added to a project; do not assume repo root), run it to confirm it fails (red), then implement the minimal function that passes (green), then refactor with tests green. Each red→green cycle maps naturally onto one small commit (tests may land in the same commit as the implementation or the commit immediately before it — never after). Tests are permanent regression assets: they stay in `tests/` and must keep passing in later experiments; run the relevant test subset as rung 1½ of the validation ladder.
- **Universal review coverage (2026-07-06):** EVERY piece of code — Coder-written source, AND Planner-written one-off scripts (page/visual generators, probe drivers, analysis one-offs) — goes through the Reviewer loop before its round closes. Small scripts may be batched into one consolidated review round; nothing executable that informs a decision or artifact ships unreviewed.
- Superseded or exploratory code is archived (patch + files) under `worklog/worklog_<username>/archive_<reason>_<date>/` before being removed from the working tree — never destroyed.

## Validation ladder (cheapest-first)

Never jump straight to the expensive run. Climb this ladder; advance only when the current rung passes, and record each rung in `_worklog.md`.

1. **Static checks** — `python -m py_compile <changed .py>`, config parse (`yaml.safe_load`), `bash -n <changed .sh>`, `git diff --check` (whitespace). Seconds, no accelerator.
2. **Tiny synthetic forward** — smallest module instantiation + one forward/step on a small/cheap device with synthetic tensors. Catches graph/mesh/shape/dtype/timestep errors without loading full weights or data.
3. **Small real-data readback** — parse a few real records; assert shapes, byte lengths, and min/max/std match the schema. Catches data-pipeline mismatches.
4. **Bounded data build** — if the experiment produces a dataset, build the val split (or a bounded slice) first and read it back before the full build.
5. **Smoke run** — a few steps at the smallest batch on the target hardware, checkpointing/final-save **disabled** (storage-light), just to reach one completed optimizer step and produce logs.
6. **Fit / batch-size probe** — find the max batch that fits, still storage-light (no checkpoints), before committing to the full run.
7. **Full run** — only after 1–6 pass and the parity audit below is clean.

## Parity audit before scaling

Before spending real compute on a new method, audit the implementation **component by component against the reference** (paper code / upstream repo), and diff the *numbers*, not just the shapes:

- **Numeric recipe defaults** — LayerNorm eps, weight decay, betas, LR schedule, loss type, sigma/noise schedule, CFG handling. Silent mismatches (e.g. eps `1e-6` vs `1e-5`, weight decay `0` vs `1e-2`) don't crash; they quietly corrupt results.
- **Structural parity** — which params are trainable vs frozen, residual/injection points, stop-gradient boundaries, any pinning/masking.
- **Data parity** — take one concrete source example and its processed/cached counterpart; confirm identical dtype, byte counts, and min/max/std.

Record the audit in `_worklog.md`; launch the full-scale run only from the audited commit.

## Running & failure discipline

- **Pre-launch acceptance criteria.** Before every launch, write in `_worklog.md` the exact conditions that count as success: the commit SHA the worker must report, device/host count, per-device and global batch, parallelism axes, which params are trainable, and "reaches ≥1 optimizer step with no OOM / NaN / parse failure." Judge the run against these, not vibes.
- **Infrastructure vs. real bug.** In every failure Analysis, classify the cause: *infrastructure* (spot preemption, host maintenance, download/network stall, launch-env/PATH, quota) vs. *real bug* (wrong shape/sharding, wrong objective, bad schedule). Only real bugs get a code fix; infra failures get retry/resume. Prevents thrashing on non-bugs.
- **Commit + push before running remotely.** Any remote (TPU/cluster) run executes a *pushed* SHA; verify the SHA the worker actually checked out. Never run uncommitted code on a remote.
- **Never edit a script while it is running.** Wait for a safe boundary, then patch; fix orchestration bugs separately from the data/model path.
- **Resume from the last safe boundary.** On a mid-pipeline crash, find the last fully-committed unit (contiguous shard / checkpoint) and resume from the next; never duplicate or lose work, never reuse stale staging.
- **Storage guardrail.** Long data/checkpoint jobs hold only a bounded working set (one batch/segment resident), clean up after each unit, and stay above an explicit free-space floor; contiguous coverage is the correctness check.
- **Shared-resource etiquette.** Inspect a shared machine's processes before using it; don't interrupt others' jobs without approval; verify zero stale processes/watchers/queued-resources after a failed run before relaunching.

## Evaluation integrity

- Always compare against the baseline method on its **full published evaluation configuration** — the complete split files and dataset configs the baseline's paper used. Never invent new eval configurations (subsampled items, hand-picked examples, reduced splits) for comparisons; subsets are for debugging only and never appear in `_results.md`.
- Reproduce the baseline's reported numbers (within its reported variance) **before** evaluating any new method, to calibrate the pipeline and establish the noise floor.
- Match the baseline's aggregation convention (e.g. per-scene means) and its variance protocol (e.g. mean ± std over N generations/seeds).
- If a run is launched and the code state then changes (revert, edit), kill and relaunch rather than mixing code states across a sweep; document the abort in `_params_set_up.md`.

## Sequencing summary

scaffold folder + `_yixun_query.md` + `_worklog.md` → `plan_*.md` (Planner, incl. per-function test list) → `_<reviewer>_plan_review.md` (Reviewer, strongest model at highest reasoning) → Planner revises → user approves → **TDD loop**, per round: (Coder: test first → red → implement → green, one small commit per cycle) → `_<reviewer>_code_<marker>_review.md` (Reviewer, per-round small review; blocking findings fixed before the next round) → next round … → **validation ladder** (static → tests → smoke → probe) → `_<reviewer>_code_full_review.md` (Reviewer, integrative) → **parity audit** vs reference → `_params_set_up.md` + `_command.md` + **acceptance criteria** → launch with teed timestamped logs, triaging *infra-vs-bug* on failure → `_results.md` → `_analysis.md` (Planner) → **`<exp name>_<idx>_results.html`** (+ assets folder) → commit(s) + `commits_*.md`. Log every action in `_worklog.md` as it happens.
