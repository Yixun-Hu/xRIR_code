# Lab notebook — yaw_rotation_degradation

## 2026-09-05T22:59:09-04:00 — scaffold experiment

- **Goal** — Start exp_03 under the SOP (numbered after the retrospective exp_01 backbone comparison and exp_02 sim-to-real records): scaffold folder, record the driving query, note the repo state the experiment builds on.
- **Version Control** — branch `main`, HEAD `9119a46` (`add cylindrical vit`). Working tree carries uncommitted pre-SOP work from the backbone comparison and sim-to-real pipeline (`model/xRIR_cyl.py`, `train_xRIR_backbone.py`, `eval_xRIR_backbone.py`, `tools/`, `sim_to_real/{prepare,haa_dataset,finetune,eval,run,summarize}_*`, `.gitignore` fix). Base-commit handling is a decision for the user at plan approval.
- **Command / Validation** — `mkdir -p worklog/worklog_yixun/{announcement,exp_03_yaw_rotation_degradation_claude}`; no announcements exist yet; Codex reviewer available (`codex-cli 0.144.1`); no `tests/` folder exists yet (placement to be asked).
- **Result** — `passed`: folder + query file + this notebook created.
- **Analysis** — Prior results this experiment builds on: baseline reproduction (`ckpt/baseline_reproduction.json`: released xRIR K=8 unseen 0.055 s / 1.36 dB / 9.69 %), backbone comparison (`ckpt/backbone_comparison_summary.txt`: CylindricalViT vs SimpleViT control, epochs 5-12 averaged, EDT -3.7 %*, test loss -1.2 %*, C50/T60 ties; per-epoch noise floor for EDT about ±0.001 s between adjacent epochs).
- **Next** — Write `plan_yaw_rotation_degradation.md`, then request the Codex plan review.

## 2026-09-05T23:09:35-04:00 — plan v1 written; retrospective records for exp_01/exp_02 created
- **Goal** — Produce `plan_yaw_rotation_degradation.md` (question, mechanism, pre-registered hypotheses and "substantial" threshold, design, per-function test list, validation ladder, parity audit, run/cost/acceptance, decisions for the user) and hand it to the Reviewer.
- **Change** — plan file only (no code). Also created, per the user's request in this session, retrospective SOP record sets for the two pre-SOP experiments: `exp_01_cylvit_vs_simplevit_claude/` (complete incl. results page) and `exp_02_sim2real_transfer_claude/` (results/analysis pending the running pipeline). Reviews there are marked N/A, not fabricated.
- **Result** — `in_progress`: plan review requested from Codex (`codex exec`, read-only sandbox, model per `~/.codex/config.toml`: gpt-5.6-sol at the config's reasoning setting).
- **Analysis** — Key design fact surfaced while planning: neither full model is yaw-invariant by construction (`lin_proj_0` is a learned pool over token positions; the coordinate embedding is not rotation-invariant), so the experiment measures actual, not structural, sensitivity; H2 is phrased accordingly.
- **Next** — Address review findings, revise plan, present to the user for approval (tests folder location, base commit).

## 2026-09-05T23:28:47-04:00 — Codex plan review received; plan v2
- **Goal** — Close the plan-review loop before user approval.
- **Result** — Review: approve with changes; 6 blocking findings (numerical non-invariance of `shift_and_align` under float32 rounding — reviewer found a real pair whose integer delay flips at k=4; unrecorded/worker-local references break per-sample parity with exp_01 at batch 16 and the reused losses reduce over the batch; Griffin-Lim random phase init unpaired across angles; the 10 % rule lacked a relative-degradation bootstrap, multiplicity control and an inference target; H1/H2 undecidable, angle grid asymmetric; evaluator lacked TDD). Saved as `yaw_rotation_degradation_codex_plan_review.md` with the identity header (codex-cli 0.144.1, gpt-5.6-sol, effort ultra).
- **Change** — `plan_yaw_rotation_degradation.md` rewritten as v2 with a finding-by-finding changelog (§11); v1 archived as `plan_yaw_rotation_degradation_v1_superseded.md`; review prompt kept in `review_prompts/`.
- **Analysis** — All blocking findings are addressed by design changes (cached k=0 alignment + end-to-end secondary condition + delay-flip audit; hashed deterministic reference manifest + per-sample loss slices + batch-1/16 equality test + distributional k=0 gate; per-sample seeded Griffin-Lim; r_k-in-bootstrap with Bonferroni(18) lower bound > 10 %, fixed-split primary / cluster-bootstrap secondary, pre-registered invalid mask; primary/replication SimpleViT wording, DiD for H2, ±2 % equivalence, signed grids; evaluator factored into four testable modules with 16 tests).
- **Next** — Present v2 to the user for approval with the five decisions in §10; then Coder rounds under TDD.

## 2026-09-05T23:38:22-04:00 — exp_01 pairing correction folded into this plan
- **Goal** — Record that the Codex review of the results pages found exp_01's evaluation references were not identical across models (24 % of queries) and phases unseeded; plan §8 now states that the k=0 gate is the properly paired re-evaluation of the epoch-12 checkpoints, reported with query-level and room-cluster CIs.
- **Result** — `passed` (plan v2 amended in place; v2 changelog unchanged otherwise). Still awaiting user approval.
