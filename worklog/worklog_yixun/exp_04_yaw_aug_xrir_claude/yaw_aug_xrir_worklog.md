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

## 2026-09-12T15:05:25-04:00 — Coder round 1 launched (Codex gpt-6-astra): rotate_scene_yaw_batch + tools/yaw_aug.py
- **Goal** — plan §5 rows 1–2 (batched per-sample rotation; verbatim seed mixer, counter, offsets, YawAug config) with their tests, TDD, one commit per cycle.
- **Version Control** — base `8cb87d1e0e8cbd85913911fdefd64aa69ced931d`.
- **Command / Validation** — `codex exec -s workspace-write --skip-git-repo-check "$(cat coder_prompts/round1_prompt.md)" < /dev/null` (log `yaw_aug_xrir_2026-09-12_15:05:25_coder_round1.log`).
- **Acceptance criteria** — two commits; the golden seeds/offsets pinned in the tests; injectivity over 111 132 counters; RNG isolation; focused and full suites green; `git diff --check` clean.
- **Result** — `launched`.
- **Next** — Fable 5.1 review of the round (`yaw_aug_xrir_fable_code_round1_review.md`).

## 2026-09-12T15:07:11-04:00 — reference manifests for the five evaluation seeds built
- **Goal** — the K-specific manifests the evaluation matrix (plan §4) needs, built before any evaluation and hashed.
- **Command / Validation** — see `_command.md`; outputs `ckpt/yaw_aug/reference_manifest_k{8,1}_seed{42..46}.json` (6 337 entries each) and the index `ckpt/yaw_aug/reference_manifests_index.json` (semantic hash + file sha256 per manifest). Semantic hashes: K = 8 seeds 42–46 `9cf5c8a2…`, `6f0a81fc…`, `42e4662a…`, `3fb941d8…`, `03f2ad94…`; K = 1 `b2955bdd…`, `0d9e033a…`, `9f7238cd…`, `f2727209…`, `61fdcba4…`.
- **Result** — `passed`. These hashes go into `tools/exp04_profiles.py` (round 4).

## 2026-09-12T15:19:31-04:00 — Coder round 1 delivered cycle 1; cycle 2 blocked on two spec conflicts → Planner decisions (plan amendment)
- **Goal** — resolve the two conflicts Codex reported before it commits cycle 2.
- **Result** — Codex (log `yaw_aug_xrir_2026-09-12_15:05:25_coder_round1.log`, exit 0): cycle 1 committed `fda92858b789ab33ae5dfdd53d34715d4e1e866e` (`rotate_scene_yaw_batch` in `tools/yaw_rotation.py`, 133 lines; focused 63 passed / 7 skipped); cycle 2 (`tools/yaw_aug.py`, `tests/test_yaw_aug.py`) implemented but uncommitted; all FLAC goldens and the 111 132-seed injectivity check pass. Conflicts: (1) the plan's "every bin within 5 % of expectation over 10^6 draws" rejects a correct `torch.randint` (21 bins outside ±5 %, max 7.57 %: with 1 953 expected per bin the binomial SD is 2.3 %, so ±5 % is 2.2 SD and ~14 excursions are expected); (2) editing `tools/yaw_rotation.py` changes a member of exp_03's reviewed evaluator closure, so exp_03's committed acceptance tests (which pin that closure's digests) fail on the working tree.
- **Analysis / decisions** — (1) *Planner mis-specification*: replace the per-bin rule with a chi-square sanity bound over the 512 bins, χ² < 700 (511 degrees of freedom: mean 511, SD ≈ 32, so 700 is ≈ +5.9 SD; observed 524.8); also assert every value in [0, W). (2) *Plan amendment*: `rotate_scene_yaw_batch` lives in `tools/yaw_aug.py`, and `tools/yaw_rotation.py` is restored byte-for-byte to its exp_03 state (commit `fda9285`'s change to it reverted in the next commit; its tests move to `tests/test_yaw_aug.py`), so exp_03's closure pin stays valid and exp_04's training closure = exp_03's unchanged `tools/yaw_rotation.py` + the new `tools/yaw_aug.py`. Both recorded in plan §11 as amendment items.
- **Next** — Codex continuation: apply the two decisions, commit cycle 2, full suite green.

## 2026-09-12T15:19:51-04:00 — Coder round 1b launched (continuation under amendments A1/A2)
- **Goal** — commit cycle 2 with the chi-square bound and the relocation of `rotate_scene_yaw_batch` into `tools/yaw_aug.py`; restore `tools/yaw_rotation.py` and `tests/test_yaw_rotation.py` to their `8cb87d1` content.
- **Command / Validation** — `codex exec -s workspace-write … coder_prompts/round1b_prompt.md` (log `yaw_aug_xrir_2026-09-12_15:19:31_coder_round1b.log`).
- **Acceptance criteria** — one commit touching exactly the four files; `git diff 8cb87d1 -- tools/yaw_rotation.py tests/test_yaw_rotation.py` empty; exp_03 record tests green again; full suite green.
- **Result** — `launched`.

## 2026-09-12T15:30:23-04:00 — round 1 code landed as one commit (`410f205`); `fda9285` dropped before push
- **Goal** — close the round's code state under amendments A1/A2 without breaking exp_03's closure pin.
- **Result** — Codex round 1b (log `yaw_aug_xrir_2026-09-12_15:19:31_coder_round1b.log`, exit 0) implemented A1/A2 and left the four-file change uncommitted: its last failing test was exp_03's acceptance check, which lists any commit after `62c9107` touching `tools/yaw_rotation.py` as drift even when the bytes are restored (`fda9285` had edited the file), and it needs both GPUs visible (the sidecars record two A6000s) while Codex was restricted to GPU 1. Planner action (git plumbing, no code): `git reset --soft 8cb87d1` (dropping the unpushed `fda9285`) and one commit `410f205ff3fa340f1eb9f371e8bf226e5a935d11` "exp_04: seeding, offsets and batched rotation in tools/yaw_aug.py (A1 chi-square bound, A2 relocation)" — `tools/yaw_aug.py` (148 lines), `tests/test_yaw_aug.py` (189 lines); `tools/yaw_rotation.py` and `tests/test_yaw_rotation.py` identical to `8cb87d1` (not in the commit). `git log 62c9107..HEAD -- tools/yaw_rotation.py` is empty again.
- **Version Control** — `main`: `8cb87d1` → `410f205`; the dropped SHA `fda9285` is recorded here and in `commits_yaw_aug_xrir.md` as superseded (never pushed).
- **Command / Validation** — `pytest tests/test_yaw_aug.py tests/test_yaw_rotation.py tests/test_exp03_record_tools.py` with both GPUs visible: 139 passed; Codex's own runs: χ² = 524.79 < 700, all offsets in [0, 512), goldens and 111 132-seed injectivity pass, `py_compile` and `git diff --check` clean, full suite 325 passed + the one exp_03 test that the reset fixes.
- **Next** — Fable 5.1 review of `8cb87d1..410f205`.

## 2026-09-12T15:43:21-04:00 — round 1 CLOSED: Fable 5.1 review approve (no blockers); round 2 launched
- **Result** — `yaw_aug_xrir_fable_code_round1_review.md`: **approve**, no blocking findings. Verified by the reviewer: the mixer is a verbatim port (three implementations agree on 200k inputs; goldens recomputed independently), every counter/rank/type boundary, RNG isolation across Python/NumPy/torch CPU/both CUDA devices, bit-exactness of `rotate_scene_yaw_batch` against exp_03's per-sample function on CPU and both GPUs for every k and many adversarial inputs, A1/A2 as recorded, test quality by mutation (23/23 non-equivalent mutants killed), focused 107 passed, full suite 326 passed. Non-blocking: 1 should-fix carried to round 2 (assert `len(train_loader) <= 9261` at startup; pass native ints), 5 nits (recorded in the review; addressed opportunistically).
- **Version Control** — round 1 = `410f205ff3fa340f1eb9f371e8bf226e5a935d11` on `8cb87d1`.
- **Next** — Coder round 2 (trainer flags + parity + alignment audit), prompt `coder_prompts/round2_prompt.md` with the carried items appended.

## 2026-09-12T15:47:20-04:00 — Coder round 2, cycle 1 TDD
- **Goal** — integrate yaw augmentation with disabled-path parity and write-free smokes.
- **Change** — train_xRIR_backbone.py and tests/test_train_yaw_aug.py only; strict resume/save-every guards, fixed-counter loader bound, native-int calls, width check, banner and args environment.
- **Command / Validation** — xRIR Python 3.8, PYTHONPATH repo, XRIR_DATA_PATH local cache, CUDA_VISIBLE_DEVICES=1, OMP_NUM_THREADS=2, NUMBA_CACHE_DIR=/tmp/xrir_numba_cache; python -m pytest tests/test_train_yaw_aug.py -q -p no:cacheprovider. Red: 10 failed against baseline (parity assertions already passed); first green: 10 passed. Final strengthening adds finite augmented gradients, green log /tmp/exp04_cycle1_green.log. py_compile and git diff --check pass.
- **Analysis** — sandbox has no GPU device access and numba cannot cache in its read-only installed package; temporary cache fixes imports, CPU tests replace cuda transfers with identity while retaining the real SimpleViT and full real-shaped tensors. External read-only nvidia-smi confirms GPU 1 available. The required len(train_loader) <= 9261 guard also rejects the approved batch-4 real-data smoke despite max-train-batches, so the explicit latest user guard takes precedence.
- **Next** — commit cycle 1; implement cycle 2 audit; Fable review belongs to the Planner after this Coder round (no Fable tool is available here).

## 2026-09-12T15:50:24-04:00 — cycle 2 audit TDD and bounded GPU smoke launch
- **Change** — tools/yaw_aug.py adds AuditBatch path metadata, alignment_audit and its CLI; tests/test_yaw_aug.py adds 9 audit cases. Cycle 1 committed as cd7dc99; concurrent Planner bookkeeping commits 2d2a386 and 8dd5aa6 retained.
- **Command / Validation** — cycle 2 red: 9 failed (missing APIs); focused green: 94 passed, 4 CUDA skips in sandbox. py_compile passes; scoped git diff --check passes. Unscoped diff --check reports trailing whitespace only in the concurrently tracked live Coder transcript; it is left intact.
- **Acceptance criteria** — CLI runs on GPU 1, float32 model/geometry, seed 0, PYTHONHASHSEED=0, 12 workers as approved training; exactly 3 batches x 32 queries x 8 references = 768 pairs, valid digest/count schema, no model forward/training; only ckpt/yaw_aug/_audit_smoke.json and its log written. No 50-batch audit.
- **Command** — see yaw_aug_xrir_command.md; log yaw_aug_xrir_2026-09-12_15:50:01_audit_smoke.log.
- **Analysis** — ordinary trainer six-tuples cannot identify query paths; AuditBatch adds metadata without modifying tensors or dataset. The CLI matches model initialization before shuffle, worker seeding and approved workers=12, with override supported. Cross-process replay requires matching PYTHONHASHSEED because dataset scene order uses a set.
- **Next** — smoke result, complete suite with CUDA_VISIBLE_DEVICES=1, cycle 2 commit.

## 2026-09-12T15:51:19-04:00 — audit smoke passed; full suite running
- **Result** — GPU-1 smoke passed: 768 pairs, 0 changed delays, fraction 0.0, cohort_sha256 bca1d9576a55db3efa612d7ad521c3e5140de58355cdc0204a7b4801dd143152. Artifact ckpt/yaw_aug/_audit_smoke.json. No 50-batch audit or full training run launched.
- **Command / Validation** — same environment as preceding smoke, python -m pytest tests -q -p no:cacheprovider; log /tmp/exp04_round2_full_suite.log. py_compile on all four changed Python files passes; scoped diff --check passes. Cycle 2 diff currently 183 changed lines including tests.
- **Next** — inspect complete suite results, commit exactly tools/yaw_aug.py and tests/test_yaw_aug.py.

## 2026-09-12T15:55:48-04:00 — Coder round 2 delivered
- **Version Control** — main; cycle 1 cd7dc994f17b656910ecce7b65633afda10fb806 (trainer + tests, 174 insertions / 6 deletions = 180 changed lines); cycle 2 34d98816a4bc4b79002aa17db78aad7ba94b0eed (audit + tests, 182 insertions / 1 deletion = 183 changed lines). Both carry the requested OpenAI Codex gpt-6-astra trailer. No push, no ckpt/ or worklog/ in either Coder commit; unrelated concurrent Planner commits preserved.
- **Result** — cycle 1: 10 passed; cycle 2 focused CPU: 94 passed / 4 skipped, including 9 new audit cases. Full suite on GPU 1: 344 passed / 1 failed, 242.19 seconds; all new cases pass. Durable log: yaw_aug_xrir_2026-09-12_15:50:01_full_suite.log. GPU smoke: 768 pairs, 0 changes. No 50-batch audit. All four Python files compile; committed code diffs pass whitespace checks.
- **Analysis** — sole full-suite failure is test_exp03_record_tools.py::test_skipping_the_gate_recomputation_is_not_acceptance: subprocess returns ACCEPTANCE FAIL (13) because recorded exp_03 environment includes two NVIDIA RTX A6000 devices and CUDA_VISIBLE_DEVICES=1 exposes only one. This is the already documented round-1 environment restriction, not a code regression; preserved exp_03 source/records. Unscoped diff --check reports whitespace in the live tracked transcript only.
- **Spec clarifications** — strict full-loader <=9261 bound intentionally rejects the plan's batch-4 smoke even when max-train-batches is set. AuditBatch adds query_paths to the unchanged six tensor fields, since raw batches lack paths. CLI defaults to the approved run's 12 workers (trainer argparse default is 16); use matching --num-workers and PYTHONHASHSEED for other training configurations. Exact digest definition is documented in alignment_audit.
- **Next** — Planner/Fable 5.1 cross-family review of this coherent round. No Fable invocation capability is available to the Coder; no Codex collaborator was used as a substitute Reviewer.

## 2026-09-12T15:56:54-04:00 — Coder round 2 delivered (two commits); one spec decision; Fable review launched
- **Result** — Codex (log `yaw_aug_xrir_2026-09-12_15:43:21_coder_round2.log`, exit 0): `cd7dc994f17b656910ecce7b65633afda10fb806` "integrate counter-seeded yaw augmentation into training" (`train_xRIR_backbone.py` +43/−, `tests/test_train_yaw_aug.py` 137 lines; 10 tests) and `34d98816a4bc4b79002aa17db78aad7ba94b0eed` "audit alignment delays on the seeded training cohort" (`tools/yaw_aug.py` +93, `tests/test_yaw_aug.py` +90; 94 passed / 4 skipped focused). Full suite 344 passed, 1 failed = exp_03's environment check, which needs both GPUs visible (Codex was restricted to GPU 1; to be re-run by the reviewer with both GPUs). GPU-1 audit smoke on 3 batches: 768 pairs, 0 changed delays (`ckpt/yaw_aug/_audit_smoke.json`). Spec clarifications reported: (1) the startup bound `len(train_loader) <= 9261` rejects the plan's batch-4 smoke; (2) the audit takes an `AuditBatch` carrying query paths (plain six-tuples lack paths) — accepted; (3) the audit CLI defaults to 12 workers and needs `PYTHONHASHSEED` pinned for reproducibility — the launcher pins it.
- **Analysis / decision (1)** — the plan's counter is defined for the confirmatory run (9 261 micro-batches per epoch). Decision: the trainer sets `batches_per_epoch = len(train_loader)` for the counter, asserts `epochs × batches_per_epoch < 2^20` (injectivity), records `batches_per_epoch` in `args.json`, and the round-3 launcher asserts `batches_per_epoch == 9261` for the confirmatory run (smokes are not confirmatory and may use any loader length). Carried into the round-2 fix list if the reviewer agrees, otherwise into round 3.
- **Next** — Fable 5.1 review of `410f205..34d9881` (exp_04 commits `cd7dc99`, `34d9881`; the interleaved `2d2a386`/`8dd5aa6` are exp_05 record commits, out of scope).

## 2026-09-12T15:58:38-04:00 — amendments A3 (new evaluation entry point instead of editing the pinned evaluator) and A4 (loader-derived counter length) recorded; round-3 prompts drafted
- **Goal** — keep exp_03's closure pins valid for every pinned file (not only `tools/yaw_rotation.py`) and settle the loader-bound question before round 3.
- **Change** — plan §11: A3 = the manifest handshake, `--conditions P` and the provenance metadata live in `tools/exp04_eval.py`, which imports `eval_yaw_rotation`'s functions unchanged; A4 = `batches_per_epoch = len(train_loader)` in the trainer, `9261` asserted by the confirmatory launcher. Prompts `coder_prompts/round3a_prompt.md` (provenance helper, evaluation entry point, evaluation launcher) and `round3b_prompt.md` (training launcher + guard tests, real smoke and throughput probe on GPU 1) drafted; round 3a starts when the Fable review of round 2 closes.

## 2026-09-12T16:31:33-04:00 — Fable round-2 code review received: approve with changes (2 blockers); Codex fix round launched
- **Review** — `yaw_aug_xrir_fable_code_round2_review.md` (Claude Fable 5.1 subagent). Verdict **approve with changes**. Verified: disabled path bit-identical to `8cb87d1` (losses/buffers/RNG; gradients under deterministic cuDNN, otherwise within the baseline's own spread); enabled path equals the reference function on the scene rotated with exp_03's helper, bit-exact; cohort digest `bca1d957…` reproduced from the real `main()` and the CLI; 36 mutants: 23 killed / 13 survived; full suite 345 passed with both GPUs visible.
- **Blockers** — (1) the 9261 pin rejects the approved rung-5 smoke and is not the A4 decision (`batches_per_epoch = len(train_loader)`, overflow assertion `epochs × B < 2^20`, live banner value); (2) no trainer-level test pins the augmented loss to the reference rotation (mutants sign-flip / depth-only / coords-only survive). Should-fix 3 → plan amendment A5 (launcher diff handles `train_batches_per_epoch` and `env`); 4–6 test strengthening; 7–9 nits; 10–11 observations (cuDNN gradient spread; audit sound).
- **Action** — plan §11: A4 extended with the §2 wording change, A5 recorded. Fix prompt `coder_prompts/round2_fix_prompt.md` (cycle 1 = finding 1; cycle 2 = findings 2, 4, 5, 6 + mutant re-check; cycle 3 = nits 7–8) launched via Codex `gpt-6-astra`; log `yaw_aug_xrir_2026-09-12_16:31:31_coder_round2fix.log`.

## 2026-09-12T16:34:11-04:00 — close-out cycle 1 green; rung-5 smoke launched
- **Goal** — implement A4 and verify the approved batch-4 smoke.
- **Version Control** — `a619bc8`, four code/test files, +45/-18; no worklog/ or ckpt/ committed, no push.
- **Command / Validation** — both touched pytest files: red 17 failed / 97 passed / 4 CUDA skips (sandbox); green on GPU 1: 118 passed. Logs `/tmp/exp04_closeout_cycle1_{red,green}.log`; py_compile and scoped diff --check pass. Full-tree whitespace findings pre-exist in the prior Coder transcript.
- **Acceptance criteria** — rung-5 exact plan argv except authorized GPU 1 and /tmp save directory; loader B=74084, banner once before training, three finite train micro-batches, two test batches, no saved files.
- **Analysis** — GPU access requires sandbox escalation; shared foreign jobs were inspected and left untouched. T6 at the top of the epoch loop requires a two-epoch exactly-once test; ordering alone with one epoch cannot detect it.
- **Next** — cycle 2 semantic equality / edge tests and isolated mutations.

## 2026-09-12T16:36:02-04:00 — rung-5 passed; cycle-2 tests green
- **Goal** — close the smoke blocker and pin trainer rotation semantics.
- **Result** — smoke exit 0: banner `yaw_aug ENABLED W=512 seed=0 counter=(epoch-1)*74084+batch_idx`; train losses 1.2363 / 1.1575 / 0.7239, test loss 0.17587 over 2 batches, save directory absent. Log `yaw_aug_xrir_2026-09-12_round2_closeout_rung5.log`.
- **Change / Validation** — cycle 2 changes only the two test files; CPU reference helper supports bit-exact loss/STFT/decay equality. Focused suite with GPU 1 visible: 121 passed (`/tmp/exp04_closeout_cycle2_green.log`).
- **Next** — isolated-copy T1/T2/T3/T6/T7/A3 mutation checks (red evidence for test strengthening), then cycle-2 commit.

## 2026-09-12T16:37:44-04:00 — cycle 2 committed; all six requested mutants killed
- **Goal** — make the reviewed semantic regressions observable without changing production code.
- **Version Control** — `da1becc`, test files only, +57/-19; focused GPU-1 suite 121 passed.
- **Command / Validation** — `/home/yixunhu/miniconda3/envs/xRIR/bin/python /tmp/exp04_round2_closeout_mutants/run_mutants.py`; isolated source copy, PYTHONPATH points only at the copy, import-origin assertions for every trainer-closure module. Logs and machine-readable results in `/tmp/exp04_round2_closeout_mutants/`.
- **Result** — baseline/restored copy each 117 passed / 4 CUDA skips; T1 sign flip, T2 depth-only and T3 coords-only each fail the new reference equality; T6 inside-loop banner fails the two-epoch count; T7 late banner fails ordering; A3 unrotated refs fails changed-delay count (1 instead of 2).
- **Next** — cycle 3 TDD for strict counters, enabled contract, exclusive audit output and args/env metadata.

## 2026-09-12T16:38:57-04:00 — cycle 3 committed; final full suite launched
- **Goal** — enforce native counters/enabled contract and preserve audit artifacts with reproduction metadata.
- **Version Control** — `35f55ad`, +72/-6; four code/test files only, no push.
- **Command / Validation** — red 17 failed / 115 passed / 4 skipped in sandbox; green with GPU 1 visible: 136 passed. Logs `/tmp/exp04_closeout_cycle3_{red,green}.log`; py_compile and scoped diff --check pass.
- **Acceptance criteria** — run the full suite once with both GPUs visible, default compute device 1, all tests pass including exp_03 record acceptance. Then final compilation and whitespace checks; audit-smoke bytes and mtime unchanged.
- **Analysis** — integer coercion removed in compute_loss as well as train_epoch, so direct callers cannot bypass the strict counter contract. CLI args/env wrap the unchanged alignment_audit result schema.
- **Next** — final suite and requested close-out report.

## 2026-09-12T16:46:56-04:00 — round-2 close-out validation complete
- **Goal** — report the three fix cycles with final regression evidence.
- **Result** — full suite once, both GPUs visible/current device 1: **373 passed, 28 warnings, 413.79 s**, no failures/skips. Log `yaw_aug_xrir_2026-09-12_round2_closeout_full_suite.log`; cycle focused counts 118 / 121 / 136 passed.
- **Validation** — xRIR Python 3.8 py_compile on all four changed files passes; `git diff --check 34d9881..HEAD` and scoped working-tree check pass. Full-tree `git diff --check` exits 2 solely for pre-existing whitespace in `yaw_aug_xrir_2026-09-12_15:43:21_coder_round2.log`, preserved unchanged.
- **Integrity** — three commits have 63 / 76 / 78 changed lines and exact trailers, only the four allowed files (cycle 2 test-only). Every pinned exp_03 file matches the starting HEAD and has no intervening edit/revert history. `_audit_smoke.json` hash remains `9ae5d2343493ec84a5aa1ab682e8cc5fcf8e0b8a0a21fa1919ce9108f5392414`, mtime 2026-09-12 15:50:33.581772174 -0400.
- **Spec / review clarification** — no unresolved implementation discrepancy under A1-A5; T6 at the top of the epoch loop cannot be caught by one-epoch ordering alone, so the strengthened test uses two epochs. Metadata belongs to the CLI artifact (the pure alignment_audit helper keeps its original schema).
- **Next** — Planner/Fable verification of the fixes; no Fable invocation tool is available in this Coder session and Codex assistance was not used as a substitute Reviewer. No push or full training run.

## 2026-09-12T16:48:10-04:00 — Codex round-2 fix delivered (`a619bc8`, `da1becc`, `35f55ad`); Fable re-review launched
- **Delivery** — cycle 1 `a619bc8` (A4: `YawAug.batches_per_epoch` threaded into the counter, overflow refusal `epochs × B ≥ 2^20`, live banner value, audit CLI guard `n_batches ≤ len(loader)`); cycle 2 `da1becc` (test-only: rotated-batch equality against the reference function, banner ordering, boundary audit fixture, 2-batch CLI, explicit `--yaw-aug-seed 3`, `--resume ""`, `--yaw-aug 0 --save-every 1 --no-save`); cycle 3 `35f55ad` (native-int guards in `train_epoch`, `assert yaw.enabled`, exclusive-create audit artefact with `args`/`env` blocks). Codex report: focused 118/121/136 passed; full suite **373 passed** (both GPUs visible, compute on GPU 1); rung-5 smoke (batch 4) runs and prints `yaw_aug ENABLED W=512 seed=0 counter=(epoch-1)*74084+batch_idx` with nothing saved; isolated mutants T1/T2/T3/T6/T7/A3 all killed (T6 needs a two-epoch case, which the parametrisation now includes). No spec discrepancy reported.
- **Action** — Fable 5.1 re-review of the three fix commits (scope: the review's blockers 1–2 and should-fix 4–6, nits 7–8; verify goldens unchanged and the mutant kills) → `yaw_aug_xrir_fable_code_round2_close_review.md`.

## 2026-09-12T17:02:40-04:00 — round 2 CLOSED: Fable close-out review = approve (no blockers)
- **Review** — `yaw_aug_xrir_fable_code_round2_close_review.md`: both blockers resolved exactly per A4 / finding 2; all 12 × 9261 confirmatory counter cells draw the same offsets under the new code, the pre-fix code and the FLAC mixer re-derived from the user's worktree; goldens hold; rung-5 smoke runs on GPU 1 with `*74084+` banner and writes nothing; mutants T1/T2/T3/T6/T7/A3 killed, 11 reviewer mutants killed except one non-blocking gap (product `epochs × B` unpinned) and two equivalents; full suite 373 passed (both GPUs visible). Carried forward: finding 1 (product test) → round 3b; findings 2–5 (early `--out` refusal, test cwd, git-dirty flag in provenance, stub test-split length) → round 3a cycle 0 / provenance helper. Prompts `round3a_prompt.md` / `round3b_prompt.md` updated accordingly.
- **Planner commit** — exp_04 bookkeeping (review files, prompts, plan amendments A3–A5, notebook, commits file) committed next; then Codex round 3a launched.
