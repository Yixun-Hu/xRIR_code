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
