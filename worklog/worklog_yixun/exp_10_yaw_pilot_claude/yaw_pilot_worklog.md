# exp_10 yaw_pilot — lab notebook

## 2026-09-26T17:24:59-04:00 — scaffold
- **Goal** — start the xRIR analogue of FLAC exp_02 (yaw non-invariance pilot) per Yixun's 2026-09-26 request.
- **Result** — folder scaffolded; query file written; facts gathered: FLAC exp_02 = released checkpoint, unseen K = 1, α ∈ {0, 90, 180, 270}°, Metric 2 (vs GT) + Metric 1 (P_α vs P_0: waveform rel-L2 / MAD, T60/C50/EDT gaps with P_0 as GT), rot0 determinism control; figure = degradation vs shift bars per angle with a 2σ noise band. xRIR exp_03 has Metric 2 for control / cyl / released at K = 8 on the acoustic grid (k = 0, ±8, ±32, ±64, ±128, 256; P condition, manifest `47637a55…`, gl_seed 0, batch 16, TF32 off) and a log-spec consistency, but no stored waveforms; its K = 1 runs are k = 0 only (no released K = 1 run). Both GPUs held by cylindrical-dinov3 exp_13 tier-L trainings (day 6); asked that session for ETA / co-tenancy; CPU path prepared (runtime device-agnostic copy of `model.xRIR.apply_delay`, as in the qualitative-figure script).
- **Next** — plan → Codex plan review → Coder round (Opus 5, worktree) → Codex code review → runs.

## 2026-09-26T17:25:22-04:00 — plan v1 + Codex plan review launched
- **Goal** — pre-registered descriptive pilot (plan §1–§9) reviewed before any code.
- **Version Control** — plan committed `ac06bca` on main; Coder worktree `/home/yixunhu/codespace/xRIR_code_wt10` (branch `exp10-yaw-pilot` from `ac06bca`) created.
- **Command / Validation** — `codex exec -c 'model_reasoning_effort="xhigh"' -s read-only -C xRIR_code …` with `review_prompts/plan_prompt.md`; log `yaw_pilot_2026-09-26_17:24:59_codex_plan_review.log`.
- **Result** — in_progress.
- **Next** — revise the plan per the review, then Coder round 1 (tools/exp10_yaw_pilot.py + tests).

## 2026-09-26T17:32:34-04:00 — plan review round 1 → plan v2 → review round 2
- **Result** — Codex round 1 (`yaw_pilot_codex_plan_review.md`): REQUEST CHANGES, 2 blockers (paired-bootstrap contract; CPU/GPU parity gate too loose) + 9 should-fixes. All addressed in plan v2 (§10 map): shared mask + one draw per replicate + ratio status; GPU 1 primary backend (cylindrical-dinov3 session agreed to a 4 GB co-tenant on GPU 1 at 17:3x: its trainings end ≈ Sep 29 (GPU 1) / Oct 3 (GPU 0), ≈ 30 % slowdown expected); probe = 17 room-spanning intact canonical batches; gate on paired Δ; fresh-inference controls incl. a nonzero repeat; released K = 1 relabelled; band = context only; identity contract; GL-free panel and signed bars; timing breakdown; reviews at `ultra` (verified accepted).
- **Command / Validation** — round-2 review launched at `ultra`: log `yaw_pilot_2026-09-26_17:32:32_codex_plan_review_round2.log`.
- **Next** — on approval: Coder round 1 in wt10.

## 2026-09-26T17:33:46-04:00 — pre-Coder checks while the round-2 review runs
- **Result** — `tests/test_exp03_record_tools.py`: 31 passed, 1 skipped (exp_03's pinned closure unchanged on main); manifest rooms contiguous in canonical order; probe batches (first intact batch inside each of the 17 rooms) = [0, 16, 32, 94, 110, 125, 141, 157, 214, 277, 292, 307, 322, 338, 353, 369, 381] (272 queries); `ckpt/exp10` absent; 50 GB free. Coder round-1 prompt drafted (`coder_prompts/round1_prompt.md`).

## 2026-09-26T17:33:58-04:00 — GPU window correction from the cylindrical-dinov3 session
- **Result** — GPU 1 has only ≈ 4.8 GB free until its exp_24 fine-tune exits (≈ 22:30); GPU 0 ≈ 3.6 GB free until ≈ 23:00. Agreed: no launch on either card before that session's ping; CPU for smoke/probe meanwhile; the 4–5 h GPU 1 window starts after the ping.

## 2026-09-26T17:37:54-04:00 — plan review round 2 → plan v3 → review round 3
- **Result** — Codex round 2 at ultra (`yaw_pilot_codex_plan_review_round2.md`): REQUEST CHANGES, 1 blocker (CPU fallback must not mix a CPU run's shifts with exp_03's GPU errors) + 5 should-fixes (parity subset alignment; live pin verification against `ckpt/yaw_rotation/binding_report.json`; offline recomputation limited to waveform/acoustic gaps; FLAC/xRIR estimator-policy audit; separate query/room ratio statuses) + 2 nits (zero-width convergence; GPU window). All incorporated in plan v3 (§11).
- **Command / Validation** — round-3 review at ultra: log `yaw_pilot_2026-09-26_17:37:52_codex_plan_review_round3.log`.
- **Note** — the round-2 and round-3 review prompts carry the updated file lists (plan version, prior review files) but the intended extra sentence "verify that the previous findings are resolved" was not inserted (a line-anchored substitution on a single-line prompt matched nothing); the round-2 reviewer nevertheless reported resolution status finding by finding, and round 3 has the round-2 review in its reading list. Prompts are recorded as sent.

## 2026-09-26T17:43:00-04:00 — plan APPROVED WITH CHANGES (round 3) → Coder round 1 launched
- **Result** — Codex round 3 at ultra: no methodological blocker; 3 should-fixes (validity masks binding in parity; execution identity; ratio zero/convergence rules) + 1 nit (`log_mse` contract) → plan v3.1 §12 A1–A4, binding for the Coder. Reviewer confirmed all 12 live pinned files match their reviewed hashes and storage ≈ 4.87 GB.
- **Command / Validation** — Coder = Claude Opus 5 (Agent, max effort) in `/home/yixunhu/codespace/xRIR_code_wt10` with `coder_prompts/round1_prompt.md`; tests run with `CUDA_VISIBLE_DEVICES=""`.
- **Acceptance criteria** — tests 1–16 green; `py_compile`; `git diff --check`; exp_03/exp_04 record suites green; CPU smoke on batch 0 with controls matching to 1e-6; no pre-existing file changed.
