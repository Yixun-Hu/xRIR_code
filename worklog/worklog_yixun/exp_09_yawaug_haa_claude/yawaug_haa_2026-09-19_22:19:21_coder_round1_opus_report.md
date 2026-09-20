# exp_09 `yawaug_haa` — Coder round 1 report

- **Coder:** Claude Opus 5 (`claude-opus-5[1m]`), Claude Code session `01NT7CPAUaJ71xgzoS8JGDVw`
- **Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp09-yawaug` (from `main` `3b7d829`). `main` untouched, nothing merged.
- **Inputs:** `plan_yawaug_haa.md` v1 and the Codex plan review of 2026-09-19 21:27 (`yawaug_haa_codex_plan_review.md`). All four of its required changes are implemented (checkpoint binding, E-only pipeline route, `--experiment exp09` on the exp_06 producer, separate E1 conclusions).
- **Environment:** CPU only (`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`), `PYTHONPATH` = the worktree, python `/home/yixunhu/miniconda3/envs/xRIR/bin/python`. No GPU touched, no foreign process signalled, no file written outside the worktree except this report.

## 1. Commits (oldest first; red-first pairs)

| SHA | Subject | Changed lines |
|---|---|---|
| `0c91651` | (red) a certified HAA child must stay verifiable after an approvals refill | tests/test_exp06_finalize.py +67/−1 |
| `455f79e` | `haa_approvals` reads the approvals blob of the child's reviewed commit | tools/exp06_finalize.py +23/−3 |
| `61b4eb0` | (red) arm E's initialisation is exp_04's approved `checkpoints.aug` | tests/test_exp06_approvals_api.py +76 |
| `a3cdd8a` | `exp04_aug_checkpoint`, the one place arm E's initialisation is resolved | tools/exp06_approvals_api.py +29 |
| `87d7a52` | (red) the HAA queue in the room frame, under exp_09's own roots | tests/test_exp06_haa_pipeline.py +121/−40 |
| `4db086e` | the HAA pipeline runs arm E in the room frame under exp_09's roots | pipeline.sh +83/−35, approvals_api +6/−3, tests +37 |
| `9909653` | (red) the summariser's arm E, its roots and its two E1 conclusions | tests/test_exp06_summarize_haa.py +141/−5 |
| `694a1cc` | confirm the finalizer certifies a room-frame arm end to end | tests/test_exp06_finalize.py +106/−12 |
| `0afdaee` | the summariser registers arm E and reads each arm under its own root | tools/exp06_summarize_haa.py +72/−11 |
| `78f4816` | E1's two conclusions, the exp_09 tables and the `--experiment` flag | tools/exp06_summarize_haa.py +94/−22, tests +59/−4 |
| `0cecf8e` | exercise the room-frame branch of `child_per_sample` directly | tests/test_exp06_summarize_haa.py +19/−6 |

Every commit carries the two required trailers. Largest commit: 153 changed lines (`78f4816`); all are under the SOP's 200.

Files touched: `tools/exp06_haa_pipeline.sh`, `tools/exp06_finalize.py`, `tools/exp06_approvals_api.py`, `tools/exp06_summarize_haa.py`, and the four test modules `tests/test_exp06_{finalize,approvals_api,haa_pipeline,summarize_haa}.py`. No pinned or forbidden file was modified; `tools/exp04_profiles.py`, `tools/exp06_profiles.py`, the entry points `exp06_haa_{finetune,eval}.py`, the model modules and every exp_03/04/05/07 file are untouched.

## 2. Design decisions

**(a) Summariser: `--experiment exp09` on `tools/exp06_summarize_haa.py`, not a new module.** Decisive reason: the approvals schema is closed. `tools/exp06_profiles.py` is a forbidden file, and `load_approved_digests` refuses any `code` key outside its `CODE_SPECS`, so a new module could never be given its own approved digest; and `check_producer_identity` compares the *entry module's* own closure with `code.summarize_haa`, which a thin wrapper would not match (it would execute unbound bytes). One producer, one closure, one key. Both experiments' behaviour is frozen in a module-level `EXPERIMENTS` table (arms, decision contrasts, screen, descriptive set, which decisions carry the classification, and the canonical outputs) that is never mutated at runtime; exp_06's entry is exactly what it published (`H1`, `H1b`, `H2`, `D`, `ckpt/exp06/{stats.json,summary.txt}`). `analyse` now publishes **only** the arms its configuration registers and refuses if one is missing, so an exp_09 run cannot carry arm D or F into a table and an exp_06 run cannot carry E.

**(b) Room-frame producer rule: no approvals-API change.** `HEADING_COMPLETE` and `PRODUCER_REQUIREMENTS` are exactly as reviewed. A room-frame job still offers the four approved heading JSONs to the *producer gate* (they exist, are approved, and hashing them is cheap) while **no child** receives `--heading-json-dir`; the job spec records `frame: room` and no `heading` key. A room-frame exception in the API would have widened a reviewed rule for no evidence.

**(c) Arm E's identity: exp_04's approved `checkpoints.aug`, never `artifacts.epoch_012`.** `exp04_profiles.AUG` has no `sha256` (the review's finding 1). The single resolver is `exp06_approvals_api.exp04_aug_checkpoint(approved, path=None)`: it hashes exp_04's approvals record, requires that hash to equal exp_06's approved `reused.exp04_approved_digests_sha256` (normalised from `approved_digests_exp04`), then loads it through exp_04's own schema (which also requires it to be committed at HEAD) and requires `epoch == 12` and a well-formed `sha256`. The identity check is deliberately **first**, so a forged record is refused as "not the approved reused identity" rather than for some later property. Checked in two places: the pipeline gate (`APPROVALS_PY`, before the job root is acquired, with `checkpoint=None` passed to `enforce_producer` so `artifacts.epoch_012` is never consulted for E) and the summariser (`expected_inits` → `load_new_arm`, which refuses any job whose `init_sha256` differs). The exp_04 record's path and digest are bound into the published `inputs` and revalidated before publication.

**(d) Blob-at-commit approvals.** New `exp06_finalize.approvals_at_commit(path, repo, commit)`: `exp06_profiles.committed_bytes` returns the blob at the child's own reviewed commit, and those exact bytes are validated by `exp06_profiles.load_approved_digests` on a private temporary copy — the same schema code, no re-implementation. The completion receipt keeps its shape and values: `path` (the repository path), `sha256` = the blob's digest — which is what every certified exp_06 child already recorded — and `committed_at` = the commit, so `verify_child`'s field-by-field comparison with the original completion still holds. The working-tree file is no longer read or required to exist; an untracked path is still refused by name ("approvals … are not tracked at &lt;commit&gt;", the one existing test whose expected message changed). The launch-time gate (`enforce_producer`) still reads the working tree, unchanged. **Side effect worth noting:** exp_06's 2026-09-18 sequencing constraint ("the approvals file must not change while any HAA child is still to finalise") is now unnecessary — a child's `reviewed_commit` is fixed at spawn and its blob there is immutable — although a *newly launched* job still needs a working file that matches its own HEAD.

**(e) Pipeline route.** `init_of` assigns `INIT_FRAME` on **every** call (a mixed queue resets it per job); `yawaug` → backbone `simple`, frame `room`, checkpoint `${EXP09_YAWAUG_CKPT:-ckpt/xRIR_simple_yawaug_8_shot/final/epoch_012.pth}`. `<init>:zeroshot` is one arm's zero-shot set; bare `zeroshot` keeps its exp_06 meaning (cyl_or, control_hf, cyl_hf). `EXP06_HAA_OUT` and `EXP06_HAA_RECORD` override the output and record roots (defaults unchanged), and the child-log prefix follows the record root (`yawaug_haa_…` when it is exp_09's, `oriented_cyl_…` otherwise), so exp_09 writes nothing into exp_06's tree. The exp_09 queue is `yawaug:0 yawaug:1 yawaug:2 yawaug:zeroshot` → 31 children + 4 jobs, asserted by a test.

**(f) E1's two conclusions.** `classify_cell` reads the **converged seed-0 two-way 95 % interval** (the same one exp_06's margin verdict reads) and publishes `category` ∈ {detected harm (lower &gt; 0), detected improvement (upper &lt; 0), no detected difference} and `non_inferior_at_margin` (upper &lt; 0.23), both strict, independent, and both `null` when the cell is void or did not converge — even though a nominal interval may still exist. The H1-style `verdict` keeps exp_06's semantics untouched. E2 is `screen_cells` with the Bonferroni-11 family (adjusted α = 0.05/11, percentile tails 0.05/22 and 1−0.05/22, hallway C50 included) and never a "non-inferior" label; E3 is `descriptive_cells` over `(('yawaug','cyl_or'),)` plus the zero-shot side split over A, B, C, E. The void rule (1 % excess / 99 % cohort) applies to E1 exactly as to H1; exp_06's descriptive treatment of the screen is unchanged.

## 3. Live re-verification of the real exp_06 children (the should-fix)

Read-only, against the **current** working-tree approvals `673c3648…`, which differ from the blob the children were certified under (`2d258f04…` at e.g. `b0bae26`):

- Before the change, reproduced on `ckpt/exp06/sim2real/cyl_or/seed0/stage1`: `refusing: approvals …/approved_digests.json differ from the bytes committed at b0bae26d…`.
- After the change, `tools/exp06_summarize_haa.verify_job` (which re-runs `finalizer.verify_child` for every child) over **all 12 real jobs** — `cyl_or`, `control_hf`, `cyl_hf` × `seed0/1/2/zeroshot`, 93 children — **ADMITTED 12, REFUSED 0**, with the registered recipe and protocol checks on (`sensitivity=False`), and each stage-1 receipt still carrying its historical `approvals.sha256` `2d258f04…`. No file was restored and no completion was rewritten.

## 4. Recomputed `code` digests at `0cecf8e4c8dd4a553d69af4a5732227fa455524d`

`tools.exp06_profiles.compute_code_digests(worktree, HEAD)` versus the committed approvals — six keys move, and they must be re-filled at the merge tip before any confirmatory run:

| key | recomputed | approved (stale after this branch) |
|---|---|---|
| `finalize` | `bb269f55bf43fd1d0698ba45c45bcbc004deba0956b337708a1bb6f20c07fd40` | `1edbaf2c…` |
| `haa_pipeline_sh` | `c1c1b1149c0b065ed523b0814ced8942bc99ee77d5c043897bb2e341f4c7325c` | `2ad3e562…` |
| `summarize_haa` | `9296f0536aae206b4ac22956401d15d01bbfde5d598867a515eae0a9b25a97fe` | `3f497fc4…` |
| `compare` | `64238a56ff7c81479f76d44a9e92340ea91afa7a9d7041d16685e12c3bc92d5f` | `43b3461f…` |
| `eval_launch` | `8c6666d5dfa5039dde7705dc062204ef2bfe55b32c9c5f7626f4e4f9cb98c939` | `0a73f84a…` |
| `mirror_probe` | `38ae34e84622c86c098a9512659f2ab47bdccc9085f3bb1c3af2cb7ec35cb6a2` | `ac503c0f…` |

Unchanged: `bootstrap, encoder, eval, evaluator_exp03, factory, haa_eval, haa_finetune, heading, launch_sh, probe_align, profiles, recipe, smoke, trainer`.

**Why six and not three.** `compare`, `eval_launch` and `mirror_probe` move only because `tools/exp06_approvals_api.py` is in their import closures and it gained `exp04_aug_checkpoint`; their behaviour is unchanged. The alternative — putting the resolver in `tools/exp06_finalize.py` — would have moved three keys instead of six, but the finalizer never uses it and the approvals adapter is its natural home; I chose correctness of placement over a smaller re-fill and am flagging the trade-off for the reviewer. Crucially, **`haa_finetune` and `haa_eval` do not move**, which is what keeps exp_06's 105 certified completions verifiable and lets exp_09's children run the same approved entry points.

## 5. Tests

- Modules iterated: `tests/test_exp06_finalize.py` + `test_exp06_finalize_haa_consumer.py` (335 passed), `tests/test_exp06_approvals_api.py` (56), `tests/test_exp06_haa_pipeline.py` (61), `tests/test_exp06_summarize_haa.py` (96). At the final commit `0cecf8e` the three modules the last commit touches were re-run together: **213 passed** in 6 min 25 s.
- New coverage: refill/blob-at-commit (3), exp_04 aug binding (6, incl. the real record), room-frame children/jobs/specs (4), pipeline room-frame goldens and roots (7 + 2 gate cases + 1 skip-guarded checkpoint identity), summariser arm E / experiments / roots / E1 boundaries / E2 tails / canonical outputs / exp_09 CLI / historical-arm-under-refill integration (17).
- Static: `bash -n tools/exp06_haa_pipeline.sh`, `py_compile` on all three changed Python files, `git diff --check` — all clean.
- **Full suite** (`python -m pytest tests -q`, CPU, at `78f4816` + the tests-only `0cecf8e` verified separately by a 96-test module re-run): **3 failed, 3256 passed, 50 skipped in 47 min 05 s**. The three failures are the expected ones and no other:
  1. `tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes` — the exp_06 approvals drift guard, failing on exactly the keys of §4 (it names `haa_pipeline_sh`, `summarize_haa`, `finalize`, `compare`, `eval_launch`, `mirror_probe`; the other 14 are identical). It goes green with the re-fill.
  2. `tests/test_exp07_profiles.py::test_the_new_arm_checkpoint_digests_come_from_the_runtime_approval[seen_simple-simple-0]`
  3. `tests/test_exp07_profiles.py::test_the_new_arm_checkpoint_digests_come_from_the_runtime_approval[seen_cyl-cylindrical-0]`

  Both exp_07 nodes were re-run on `main` in the untouched main checkout and fail identically there (`2 failed, 1 passed` in 2.6 s), so they are the peer's pre-existing guard, not exp_09's. Note that it is **two** parametrisations, not the one named in the round prompt.

## 6. Validation ladder step 2 (printed dry run, inspected)

`EXP06_HAA_OUT=ckpt/exp09/sim2real EXP06_HAA_RECORD=worklog/worklog_yixun/exp_09_yawaug_haa_claude tools/exp06_haa_pipeline.sh <gpu> yawaug:0 yawaug:1 yawaug:2 yawaug:zeroshot --dry-run` prints, for every child: no `--heading-json-dir`, `--save-dir ckpt/exp09/sim2real/yawaug/…`, logs `worklog/…/exp_09_yawaug_haa_claude/yawaug_haa_<UTC>_haa_yawaug_…log`, and `JOBSPEC … backbone=simple frame=room heading=none`; 31 `RUN nohup setsid` lines and 4 job finalisations.

## 7. Open items

1. **Approvals re-fill required** at the merge tip for the six keys in §4 before any confirmatory exp_09 run: the `haa_children` gate requires `code.finalize` and `code.haa_pipeline_sh`, and the summariser requires `code.summarize_haa`. Until then `tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes` fails by design (it is the drift guard for exactly these keys) — this is the exp_06 failure to expect, named rather than waved through.
2. **Peer's exp_07 guard** — the two exp_07 nodes named in §5, verified pre-existing on `main`.
3. **No GPU validation.** The room-frame path of `exp06_haa_finetune.py` / `exp06_haa_eval.py` is exercised only by unit evidence and by exp_02's historical room-frame runs; the plan's bounded room-frame smoke/readback before the full queue is still to be done on a card.
4. **exp_09's own output paths are not enforced**, only exp_06's are protected: `--experiment exp09 --json ckpt/exp09/stats.json --summary ckpt/exp09/summary.txt` is the operator's responsibility (the runbook line should carry it verbatim).
5. **Deliberate strengthening to report to the reviewer:** `analyse` now refuses when an arm of the selected configuration is missing and drops arms outside it. exp_06's five are unaffected, but this is a behaviour change to a reviewed function.
6. The exp_06 golden dry-run helpers in `tests/test_exp06_haa_pipeline.py` were made frame-aware (the golden *lines* for `cyl_or`/`control_hf`/`cyl_hf` are byte-identical and their tests pass unchanged).
