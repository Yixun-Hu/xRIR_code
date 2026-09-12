# Worklog — yaw_aug_xrir (exp_04)

## 2026-09-12T11:10:13-04:00 — scaffold
- **Goal** — start the SOP record for the yaw-augmentation xRIR experiment requested on 2026-09-12 (query file); read the FLAC exp_17 reference (plan, params, augmentation code) and this repo's training/evaluation entry points before planning.
- **Version Control** — `main` at `19a04efe8b618aea7f6c540908f9e96eb7a4b69b` (exp_03 record committed; no code change yet). No announcements exist in `worklog/worklog_yixun/announcement/` (folder empty); the FLAC namespace's directives 04 (living results table, mean over eval seeds, no single-seed screens) and 05 (declare the eval protocol in every manifest) are adopted as principles here.
- **Result** — reference design captured: FLAC `draw_yaw_offsets` (d ~ Uniform{0..W−1}, dedicated generator, global RNG untouched), `offsets_to_radians`, `rotate_scene_metadata` (roll + Rz of the depth panorama, Rz of every pose, audio untouched, exact), counter-based seed `(seed, global_step, rank)` through a keyed 32-bit bijection, applied at every training step to every sample, banner on rank 0, never at validation/evaluation; exp_17 recipe = P1 vanilla + one config delta, 40k optimizer steps at effective batch 64, single seed 42. Repo hook: `train_xRIR_backbone.py::compute_loss` receives `(_, src_loc, depth_coord, tgt_wav, ref_irs, ref_locs)`; `tools/yaw_rotation.rotate_scene_yaw(depth_coord, src, ref_locs, k, W)` already implements the exact scene yaw used by exp_03's evaluator.
- **Next** — plan → Codex plan review → user approval.

## 2026-09-12T11:31:03-04:00 — plan v1 written, Codex plan review round 1 (request changes), plan v2
- **Goal** — a reviewed plan before asking the user to approve.
- **Change** — `plan_yaw_aug_xrir.md` v1 (1 815 words) → Codex review `yaw_aug_xrir_codex_plan_review.md` (12 findings: seed mapping not pinned, resume claim false for this trainer, single-seed evaluation against the adopted directive 04, one-sided quantiles not executable through the two-sided helper, H2/TOST families incomplete, no immutable eval manifest (directive 05), producer not specified fail-closed, single-delta overstated, alignment not numerically invariant, narrow tests/gates, wrong counts/budget, stale test count) → v2 (2 791 words; v1 kept as `plan_yaw_aug_xrir_v1_superseded.md`; changelog §11). Facts verified for v2: 296 334 training samples, 9 261 micro-batches, 4 631 updates per epoch; the control's 12 epochs took 28.2 h; the control's checkpoint files enumerated.
- **Command / Validation** — review: `codex exec -s read-only` with `review_prompts/plan_prompt.md` (SOP briefing: SOP, prior results exp_01/exp_03, the FLAC exp_17 reference code and record, scope statement).
- **Result** — `fix_ready`: v2 addresses all 12 findings; round-2 review launched.
- **Next** — round-2 review → user approval (decisions in plan §10) → Coder rounds.

## 2026-09-12T11:48:00-04:00 — Codex plan review round 2 (request changes, 7 items) → plan v3
- **Goal** — close the round-2 items before asking for approval.
- **Change** — `plan_yaw_aug_xrir.md` v3 (3 836 words; v2 kept as `plan_yaw_aug_xrir_v2_superseded.md`; changelog §11): five-seed data for every TOST cell (augmented arm evaluated at the eight-angle union), TOST one-arm; the convergence gate defined on the companion intervals whose upper endpoint is the decision bound; an operationally immutable, evaluator-bound evaluation manifest (exclusive run directory, `--eval-manifest` handshake, digest echoed into outputs, post-run recheck, completion sidecar) with training/evaluator/writer/producer closures; the producer driven by hard-coded profiles with an exact CLI; exact smoke/full argv with immutable attempt directories and a 43 h cumulative ceiling; the alignment policy fixed (rotated geometry; integer-delay-only descriptive audit on a deterministic cohort); executable gates (seed boundaries, `effective_args` allowlist, `PYTHONHASHSEED=0`, numeric GPU/disk thresholds, epoch-1 2.431 h). The reviewer's independently computed golden seeds/offsets are pinned in the plan.
- **Command / Validation** — round-2 review `yaw_aug_xrir_codex_plan_round2_review.md` (`review_prompts/plan_round2_prompt.md`); round-3 review launched with `review_prompts/plan_round3_prompt.md`.
- **Result** — `fix_ready`.
- **Next** — round-3 verdict → user approval (plan §10) → Coder round 1.

## 2026-09-12T12:01:50-04:00 — Codex plan review round 3 (request changes, 7 operational items) → plan v4; presented for approval
- **Goal** — close the round-3 items and present the plan.
- **Change** — `plan_yaw_aug_xrir.md` v4 (v3 kept as `plan_yaw_aug_xrir_v3_superseded.md`; changelog §11): complete argv/environment and one normalised schema, launcher-owned provenance finalisation with an execution-bound training manifest, H1's mask, profile-pinned closures/schema/dataset identity with fail-closed producers and downstream verification, `--conditions P` mode and canonical input selection, the alignment audit wired to a hashed artefact, the superiority interval pinned. Rounds 2–3 raised no scientific-design objection; round 1's design findings are closed (reviewer's closure tables).
- **Command / Validation** — round-3 review `yaw_aug_xrir_codex_plan_round3_review.md` (`review_prompts/plan_round3_prompt.md`).
- **Result** — `fix_ready`; awaiting the user's approval of plan §10.
- **Next** — on approval: base commit of the record folder, Coder round 1 (rotation batch + seeding, TDD), per-round Codex reviews.

## 2026-09-12T15:03:17-04:00 — user correction: training source closure must include utils/lr_scheduler.py
- **Goal** — fix an omission the user found in plan v4 §4 before approval.
- **Change** — verified `train_xRIR_backbone.py:30` imports `ExponentialLR` from `utils/lr_scheduler.py`; the plan's training closure list omitted it. Plan §4 now lists it and, more robustly, defines the training closure as the set of repo-local modules loaded by importing the trainer (the evaluator-closure mechanism), with the enumerated files as a required minimum (changelog §11, item 0).
- **Result** — `passed` (plan text only; no code exists).
- **Next** — awaiting the user's approval of plan §10.

## 2026-09-12T15:04:28-04:00 — plan v4 approved by the user; base commit
- **Goal** — record the approval and freeze the base for the TDD rounds.
- **Result** — user: "Then I approve the plan, Please implement and run" (2026-09-12), after the role changes (Coder = Codex `gpt-6-astra`, code Reviewer = Claude Fable 5.1) and the `utils/lr_scheduler.py` correction. Decisions §10 taken as recommended: 12 epochs with epoch 12 confirmatory and epoch 9 diagnostic; SimpleViT only; five evaluation seeds {42–46} with fresh comparator evaluations; tests in `tests/` and the living table `worklog/worklog_yixun/model_comparison.md`; R@k undefined.
- **Version Control** — base commit below (record folder, SOP role update, exp_03 K = 1 follow-up entries); the exp_01/exp_02 commit-manifest edits stay uncommitted (unrelated).
- **Next** — Coder round 1: `tools/yaw_rotation.rotate_scene_yaw_batch` + `tools/yaw_aug.py` with tests.
