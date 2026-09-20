# exp_09 `yawaug_haa` — Coder round 1 **fix cycle** report

- **Coder:** Claude Opus 5 (`claude-opus-5[1m]`), Claude Code session `01NT7CPAUaJ71xgzoS8JGDVw`
- **Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp09-yawaug`. Round-1 tip `0cecf8e` → fix tip `9e41c65`. `main` (`3b7d829`) untouched, nothing merged, no GPU touched, no foreign process signalled. The only file written in the main checkout is this report.
- **Input:** `yawaug_haa_codex_code_round1_review.md` (verdict *request changes*, findings 1–3). Everything the reviewer verified is unchanged; the three findings are addressed below.
- **Environment:** CPU only (`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`, `PYTHONDONTWRITEBYTECODE=1`), `PYTHONPATH` = the worktree, python `/home/yixunhu/miniconda3/envs/xRIR/bin/python`.

## 1. Commits (oldest first; red-first pairs)

| SHA | Subject | Files | Changed lines (add+del) |
|---|---|---|---|
| `f7f3de0` | (red) the completed exp_06 evaluations stay verifiable at this tip | `tests/test_exp09_sim_eval_closures.py` +76 | **76** |
| `8aa4b48` | the arm-E resolver moves out of the shared approvals module | `tools/exp06_approvals_api.py` −32, `tools/exp06_finalize.py` +37, `tools/exp06_haa_pipeline.sh` +7/−1, `tools/exp06_summarize_haa.py` +1/−1, three test modules +11/−8 | **98** |
| `b4847cd` | the resolver's tests follow it out of the approvals module | `tests/test_exp06_approvals_api.py` −77, `tests/test_exp09_aug_checkpoint.py` +109 | **186** |
| `68c3f03` | (red) the checkpoint must come from the snapshot that was approved | `tests/test_exp09_aug_checkpoint.py` +38 | **38** |
| `192fef7` | one validated snapshot of the exp_04 approvals record | `tools/exp06_finalize.py` +13/−6 | **19** |
| `27341c3` | (red) an exp_06 run must publish exp_06's own schema | `tests/test_exp06_summarize_haa.py` +49 | **49** |
| `db4121e` | the `experiment` field is exp_09's alone | `tools/exp06_summarize_haa.py` +26/−21, tests +1/−1 | **49** |
| `9e41c65` | the base-schema regression covers the rendered summary too | `tests/test_exp06_summarize_haa.py` +1 | **1** |

Every commit carries both required trailers (checked with `git log --format='%(trailers:key=…)'` over all 19 commits of the branch). Largest commit **186** changed lines, counted as additions **plus** deletions, as the reviewer counts them.

Files the branch changes versus `main` (nine, one fewer than round 1 and a different set):
`tools/exp06_{finalize.py,haa_pipeline.sh,summarize_haa.py}` and `tests/test_exp06_{approvals_api,finalize,haa_pipeline,summarize_haa}.py`, `tests/test_exp09_{aug_checkpoint,sim_eval_closures}.py`.
`git diff main HEAD -- tools/exp06_approvals_api.py` is **empty**: the file no longer appears in the branch's changed-file list at all.

## 2. Finding 1 (blocker) — the shared approvals module is restored byte for byte

**What was done.** `git checkout main -- tools/exp06_approvals_api.py` (verified empty diff against `main`), and `exp04_aug_checkpoint` / `EXP04_AUG_EPOCH` relocated to `tools/exp06_finalize.py`. Callers updated: the pipeline gate (`APPROVALS_PY` in `tools/exp06_haa_pipeline.sh`), the summariser (`expected_inits`), and the tests. The behaviour, the refusal messages and the six original regressions are unchanged.

**Why the finalizer is the right home, verified rather than asserted** — `exp06_profiles._paths_of` listings at the tip:

| key | contains `tools/exp06_finalize.py` | contains `tools/exp06_approvals_api.py` |
|---|---|---|
| `eval` | no | no |
| `eval_launch` | **no** | yes |
| `compare` | **no** | yes |
| `mirror_probe` | **no** | yes |
| `summarize_haa` | yes | yes |
| `finalize` | yes | no |

So the resolver is outside every closure the completed evaluations are checked against, and inside the two that this round changes anyway. It is still **approved bytes wherever it runs**: `finalize` is one of `PRODUCER_CODE_KEYS['haa_children']`, so `enforce_producer` in the same gate call checks these bytes before any child is started; and the summariser executes it inside its own `code.summarize_haa` closure. The pipeline imports the finalizer **only** on the `yawaug` branch, to keep a torch import off every other init's gate.

**Regressions added** (`tests/test_exp09_sim_eval_closures.py`, three tests):
1. no evaluation-writer closure contains `tools/exp06_finalize.py`, while three of them do contain `tools/exp06_approvals_api.py` (the rule, stated both ways);
2. `compute_code_digests(REPO, HEAD, keys=('eval','eval_launch','compare','mirror_probe'))` equals the same call at `main` — the reviewer's "unchanged vs main" assertion, as a test;
3. the ten real manifests under `ckpt/exp06/sim_eval/{cyl,cyl_or}/seed{42..46}`: each recorded `source_closures.writer_exp06.sha256` equals the `eval_launch` digest a re-fill at this tip would publish, and every recorded `working_tree_sha256` of all three closures still holds (the source revalidation the reviewer ran). Skips cleanly on a checkout without the artefacts.

Tests 2 and 3 were red at `0cecf8e` and at the red commit; they pass from `8aa4b48` on.

## 3. Finding 2 — one validated snapshot of the exp_04 approvals record

`exp04_aug_checkpoint` now checks the **parsed snapshot's own identity** as well: `exp04_profiles.load_approved_digests(file)` returns `(value, identity)` and `identity['sha256']` must equal the reused pin `reused.exp04_approved_digests_sha256` before `value['checkpoints']['aug']` is read. The preliminary hash of the file as read is kept — it refuses an unapproved file *as* an unapproved file, before anything is parsed — and each of the two refusals names which read disagreed (`as read` / `as parsed`).

Regression (`tests/test_exp09_aug_checkpoint.py`): a replacement-between-reads probe monkeypatches the loader to return a schema-valid, git-bound record whose identity is **not** the pin while the first read saw the approved record → refused with `reused.exp04_approved_digests_sha256`; a control returning the *approved* identity resolves, so the refusal is about the identity and not about the probe.

## 4. Finding 3 — exp_06's default output is schema-identical again

`analyse` emits `experiment` only when the experiment is not `exp06` (`result = dict({'schema_version': 1}, **({} if experiment == 'exp06' else {'experiment': experiment}))`), and `main`'s printed CLI JSON adds it only for exp_09, after `summary_sha256`, leaving exp_06's five keys in their original order. `render` already read the field back with `exp_06` as its default, so it needed no change. exp_06's registered arm set is the five arms **A/B/C/D/F** (`control, cyl, cyl_or, control_hf, cyl_hf`), as the reviewer states.

Regressions added to `tests/test_exp06_summarize_haa.py`:
- **base-schema**: `tools/exp06_summarize_haa.py` *as `main` carries it* is imported from its git blob into a temporary file beside this round's copy (`analyse` reads only the arms it is given and its own contrast constants, so a copy outside the repository answers for the base schema exactly). Both are run over the same synthetic arms, restricted to exp_06's five: the statistics must agree **key for key and byte for byte** (`json.dumps(..., sort_keys=True)`), and — added after the reviewer's observation that the summaries already matched — `render` must agree too.
- **CLI**: a default `main()` run prints exactly `['json', 'sha256', 'summary_sha256', 'H1', 'H1b']`, in that order.
- the round-1 "unchanged" test that asserted the added field now asserts `'experiment' not in result`.

Both new tests were red at `27341c3` (statistics: `At index 1 diff: 'experiment' != 'exploratory'`; CLI: `At index 2 diff: 'experiment' != 'summary_sha256'`) and green from `db4121e`.

## 5. Recomputed `code` digests at `9e41c65`

`tools.exp06_profiles.compute_code_digests(worktree, 'HEAD')`, all twenty keys, compared with the same call at `main`. **Three keys move, not six:**

| key | digest at this tip | vs `main` |
|---|---|---|
| `finalize` | `ea8146d84e5056d3b82fb76ead33bf37ec01a17ba36915761da880c5fd92c093` | **moved** |
| `haa_pipeline_sh` | `4a401ee3d80f2e3c77ecf68fd538e0108d6c649c92680c1b0373824626ad8c08` | **moved** |
| `summarize_haa` | `3fc1e82d374c8e5fb8130d5223b7d744168b0462dabc1d22017423b4a1ad3376` | **moved** |
| `eval_launch` | `0a73f84a5a1c17533cfc3e2ef62c24928fb157103e1f4d69bf8d2474ad7ff205` | same (= the ten manifests' `writer_exp06`) |
| `compare` | `43b3461f334aaeef0ce16d1a9f166c91917c73d4b57ce250fe2c6ab8fe921dae` | same |
| `mirror_probe` | `ac503c0f265576c478c45dff4c3376cfa029d47c434fcd11db3e5c7182309597` | same |
| `eval` | `42975b3acb4e73ab10f0cde7ed15fd438874595dd972c3f33d3e8fa4aae78316` | same |
| `haa_finetune` | `9288884b4034d4510c16ee081e95855e1229aa19a23b94f22e46ce90e6b0458b` | same |
| `haa_eval` | `1f20e36bfb29b557de020321c30ba083d5631ae6e2f1b3fbfd94ecb0e77f1134` | same |

Also unchanged: `bootstrap, encoder, evaluator_exp03, factory, heading, launch_sh, probe_align, profiles, recipe, smoke, trainer`. The re-fill at the merge tip therefore touches **three** keys (`finalize`, `haa_pipeline_sh`, `summarize_haa`) — `summarize_haa` is the only other key whose closure contains the finalizer — and the `haa_children` gate plus the summariser gate are exactly the producers that need them.

## 6. Live re-verification at this tip (read-only)

Run from the worktree with this branch's modules on `PYTHONPATH` (`tools/exp06_finalize.py` confirmed to be the worktree's file), against the **current** working approvals `673c3648…`:

- **The 12 real exp_06 HAA jobs / 93 children** — `exp06_summarize_haa.verify_job(..., sensitivity=False, inputs={})` over `cyl_or`, `control_hf`, `cyl_hf` × `seed0/1/2/zeroshot`: **ADMITTED 12, REFUSED 0, 93 children**, every child still carrying its historical approvals identity `2d258f0444352bafbfe583a5ccda17132b27b5f6b3ba97e22f4d02d0ec6c87ad` across nine distinct reviewed commits — with the working file at `673c3648…`, i.e. the blob-at-commit path of round 1 doing its job. Nothing was restored or rewritten.
- **The ten completed simulated evaluations** — `exp06_compare.check_route('exp06', …)` with the current approvals, plus a re-hash of every file in all three recorded closures: **10 routed and revalidated, 0 refused**. At `0cecf8e` the same check refuses all ten (`approved eval_launch closure`, `source.writer_exp06.tools/exp06_approvals_api.py`); the pytest guard reproduces exactly this.

**Observation worth recording (pre-existing, not introduced this round).** These re-verifications must name the checkout the runs were recorded in (`/home/yixunhu/codespace/xRIR_code`). A completion's `approvals.path` is an **absolute** path, and `verify_child` compares it field for field, so passing the worktree as `repo` refuses all 12 jobs on the path alone although the `sha256` and `committed_at` are identical. That is a property of the completion schema on `main`, unchanged by this round, but a reviewer re-running the check from the worktree will see it.

## 7. Static checks and tests

- `bash -n tools/exp06_haa_pipeline.sh`, `py_compile` on every changed Python file, `git diff --check` — all clean.
- Targeted modules at the fix tip: `tests/test_exp06_summarize_haa.py` **98 passed** (4 m 35 s), `tests/test_exp09_aug_checkpoint.py` **8 passed**, `tests/test_exp09_sim_eval_closures.py` **3 passed**, `tests/test_exp06_approvals_api.py` **50 passed** (56 before the six moved out).
- **Full suite**, once, at `db4121e` (`CUDA_VISIBLE_DEVICES='' python -m pytest tests -q -p no:cacheprovider`): **3 failed, 3263 passed, 50 skipped** in 46 m 01 s. Exactly the permitted failures and no other:
  1. `tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes` — the exp_06 approvals-drift guard, naming exactly the three keys of §5 (`haa_pipeline_sh` `4a401ee3…` vs `2ad3e562…`, `finalize` `ea8146d8…` vs `1edbaf2c…`, `summarize_haa` `3fc1e82d…` vs `3f497fc4…`) with **17 identical** keys — three fewer differing keys than round 1's six. It goes green with the re-fill.
  2. `tests/test_exp07_profiles.py::…[seen_simple-simple-0]` and 3. `…[seen_cyl-cylindrical-0]` — the peer's pre-existing null-approval guards, which round 1 re-ran on `main` and which fail identically there.
- The last commit `9e41c65` adds one assertion inside an already-passing test, so the four exp_06 modules this round touches were re-run together **at the final tip `9e41c65`**: **494 passed** (13 m 33 s) — `test_exp06_summarize_haa.py`, `test_exp06_finalize.py`, `test_exp06_finalize_haa_consumer.py`, `test_exp06_haa_pipeline.py`. The three modules of the bullet above were re-run at the tip as well.

## 8. Open items

1. **Approvals re-fill** at the merge tip for the three keys of §5 (`finalize`, `haa_pipeline_sh`, `summarize_haa`) before any confirmatory run; until then the exp_06 drift guard fails by design on exactly those keys.
2. **Peer's exp_07 guard** — `seen_simple-simple-0` and `seen_cyl-cylindrical-0`, pre-existing on `main` (round 1 re-ran them there).
3. **No GPU validation yet.** The bounded room-frame smoke (two-epoch `class_room` fine-tune + four-sample `hallway` evaluation, with readback of checkpoint lineage, `frame=room`, null headings) is still to be done before the full queue, as the reviewer requires.
4. The round-1 open items 4–6 stand unchanged (exp_09's own output paths are the operator's responsibility; `analyse` refuses a missing registered arm; the pipeline golden helpers are frame-aware with byte-identical golden lines).
