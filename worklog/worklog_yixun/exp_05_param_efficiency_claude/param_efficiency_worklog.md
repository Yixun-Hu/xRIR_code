# Worklog — param_efficiency (exp_05)

## 2026-09-12T15:33:05-04:00 — scaffold and plan v1 (written for handoff)
- **Goal** — the capacity-curve experiment the user asked for (query file); written by the exp_04 session's Planner, to be executed by another Claude Code agent.
- **Result** — plan v1 with measured tier parameter counts (S 2.77/2.78 M, M 19.70/19.75 M, L 43.71/43.78 M encoder; full 15.1/32.1/56.2 M) and a CPU forward check that failed only because the baseline's `apply_delay` hard-codes `.cuda()` (GPU forwards are required; noted in §7). Codex plan review launched.
- **Next** — Codex plan review → revisions → user approval → executing agent takes over.

## 2026-09-12T15:39:21-04:00 — Codex plan review round 1 (request changes, 9 findings) → plan v2
- **Goal** — close the review before the user is asked to approve the handoff plan.
- **Change** — `plan_param_efficiency.md` v2 (changelog §11; v1 kept as `plan_param_efficiency_v1_superseded.md`): tiered evaluator entry point; 32 × 2 mandatory (no 16 × 4 fallback); H2 as one four-cell family with explicit tails and TOST levels; mandatory exp_04-equivalent provenance and dedicated profiles; experiment-owned parameter counting (no edits to exp_03's closure members); fit/timing probes and both-backbone parity; exact counts and a ≤ 0.5 % matching tolerance; registered cohorts; corrected timing basis and budget; scoped H1 wording.
- **Command / Validation** — review `param_efficiency_codex_plan_review.md` (`review_prompts/plan_prompt.md`).
- **Result** — `fix_ready`; round-2 review launched.

## 2026-09-12T15:44:32-04:00 — Codex plan review round 2: approve with changes → plan v3; presented for approval
- **Result** — `param_efficiency_codex_plan_round2_review.md`: round-1 findings 1–4, 8, 9 closed, 5–7 partially; new should-fix 1–4 (H2 target cohort, three exact counts, epoch-1 gate, parity schema) and nits 5–6 (budget arithmetic, byte-identity wording) — all applied in v3 (changelog §12; v2 kept as `plan_param_efficiency_v2_superseded.md`).
- **Next** — user approval (plan §9 decisions); then handoff to the executing agent.

## 2026-09-12T15:47:40-04:00 — plan v3 approved by the user; record committed for handoff
- **Result** — user decisions (verbatim summary): "Use S/M/L only. Do not add S+ for this round. … Reuse the exp_01 M checkpoints. Clearly disclose the historical-run differences and the limitation that training realizations are not exactly paired. Keep K=8 primary and K=1 secondary. Report both completely, including any differences in the conclusions. Retain the two fixed-target comparisons: Cyl-S vs Base-M and Cyl-M vs Base-L." Plan status and §5/§9 updated accordingly.
- **Version Control** — record folder committed (SHA in `commits_param_efficiency.md`); execution handed to another Claude Code agent (plan §10).
- **Next (executing agent)** — scaffold `_params_set_up.md`/`_command.md` as work starts; Coder round 1 = `tools/exp05_params.py` + trainer tier flags (TDD, Codex `gpt-6-astra`), Fable 5.1 review; then the probes, parity and the four training launches.
