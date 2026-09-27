# exp_11 orientation_cue_fairness — Coder round 1 report

**Coder:** Claude Opus 5 (1M context), Agent-tool subagent at max effort
**Date:** 2026-09-26 18:26:47 EDT
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, base commit `c6233e3`
**Scope:** plan v3 §4 "Round 1" items 1–4 — Phase 1 (arm G `yawaug_hf`) only.
Round 2 (the exp_11-owned family: models, profiles, finalizer, entry points, pipeline,
launcher, summariser arms H/I/J/K) is **not** in this round.
**Environment:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`,
`PYTHONPATH=/home/yixunhu/codespace/xRIR_code_wt`, `CUDA_VISIBLE_DEVICES=''` for every
command. No GPU was used, no process was signalled, nothing outside the worktree was
written (the report directory here is the worktree's copy; the record in `main` is
untracked and was not touched).

## 1. Commits

Base `c6233e3` → HEAD `5db0c82`. Fourteen commits, each with its tests in the same commit.
**Wording narrowed after the Codex review:** every *behavioural* change was preceded by a
failing test; the two acceptance-only commits at the end (`73a8732`, `5db0c82`) add
characterisation tests for guards that already held, so they did not go red first. The
round-1 fix cycle continues in
`orientation_cue_fairness_2026-09-26_20:47:07_coder_round1_fix_opus_report.md`, which
supersedes the suite result and the `summarize_haa` digest below.

| SHA | Subject | changed lines |
|---|---|---|
| `10661ed` | exp11: pipeline init yawaug_hf (arm G, heading frame) | 113 |
| `fd1503c` | exp11: register arm G in the HAA summariser | 116 |
| `fff7948` | exp11: the decision notation of plan section 3, generically | 202 |
| `968e167` | exp11: the four-arm interaction cell N1i | 208 |
| `bc8e762` | exp11: three screen families under exp_11's suppression | 102 |
| `824dce9` | exp11: R1 historical rows, copied under a source-field map | 248 |
| `afd5a71` | exp11: the frozen phase-1 configuration and its tables | 202 |
| `6af6688` | exp11: render and publish the phase-1 record | 183 |
| `91a48eb` | exp11: the unavailable path is the degenerate interval and nothing else | 31 |
| `e857e04` | exp11: keep the section-7 comment on the section-7 contrasts | 3 |
| `850936f` | exp11: name the new module constants for their experiment | 12 |
| `825c7ba` | exp11: hoist the test module's imports | 7 |
| `73a8732` | exp11: assert arm G's job declaration carries the four heading rolls | 14 |
| `5db0c82` | exp11: assert arm G is admitted only in the heading frame | 15 |

Four commits exceed the SOP's ≤ 200-line guideline (202, 208, 248, 202), each because the
test that had to land with the implementation is large; the implementation halves are 95,
94, 101 and 121 lines. Files touched (`git diff c6233e3..HEAD --numstat`):
`tools/exp06_haa_pipeline.sh` (+18/−6), `tools/exp06_summarize_haa.py` (+494/−35),
`tests/test_exp06_haa_pipeline.py` (+101/−2), `tests/test_exp06_summarize_haa.py` (+5/−4),
`tests/test_exp11_summarize_haa.py` (new, 713 lines). Nothing else in the repository was
modified.

## 2. Moved code-key digests

Recomputed with `tools.exp06_profiles.compute_code_digests(wt, 'HEAD')` at `5db0c82`
(the value is the same at `73a8732`: the last two commits touch tests only, which are in
no closure):

| key | digest at `5db0c82` | was (`c6233e3`) |
|---|---|---|
| `haa_pipeline_sh` | `f33d4ffe015ab76c1fc1115f8918983d6048351ca12bf96c47739e89a8eb573e` | `4a401ee3d80f2e3c77ecf68fd538e0108d6c649c92680c1b0373824626ad8c08` |
| `summarize_haa` | `32937f96910dc5b0452204ad457231422cd32b4ce84867bed3ef7276f9a62fc2` | `3fc1e82d374c8e5fb8130d5223b7d744168b0462dabc1d22017423b4a1ad3376` |

Unchanged (verified by the same call): `finalize`
`ea8146d84e5056d3b82fb76ead33bf37ec01a17ba36915761da880c5fd92c093`, and the four sim-eval
closure keys `eval`, `eval_launch`, `compare`, `mirror_probe`, plus `haa_finetune`,
`haa_eval`, `bootstrap`, `profiles`. `tests/test_exp09_sim_eval_closures.py` therefore
stays green.

**These are the working-tree digests of this branch.** They will change again at the merge
commit if any of these files move; re-fill `code.haa_pipeline_sh` and `code.summarize_haa`
in `oriented_cyl_results_assets/approved_digests.json` **at the merge commit**, exactly as
exp_09 did, before any confirmatory exp_11 producer runs. `code.finalize` needs no re-fill.

## 3. What was implemented

### 3.1 Pipeline init `yawaug_hf` (`tools/exp06_haa_pipeline.sh`, `10661ed`)

* `init_of` gains `yawaug_hf`: backbone `simple`, checkpoint
  `${EXP09_YAWAUG_CKPT:-ckpt/xRIR_simple_yawaug_8_shot/final/epoch_012.pth}` (one variable
  for E and G, so G provably starts from E's bytes), frame **heading** — taken from the
  reset `INIT_FRAME=heading` at the head of `init_of`, which is what makes a mixed queue
  give each job its own frame.
* `APPROVALS_PY`'s identity resolver now applies `finalizer.exp04_aug_checkpoint(approved)`
  to `yawaug` **and** `yawaug_hf`; the heading gate and `enforce_producer('haa_children')`
  are unchanged.
* `child_log` maps a record root containing `exp_11` to the prefix
  `orientation_cue_fairness_haa` (the `exp_09` case is untouched).
* Usage/help text and the header comment list the new init; bare `zeroshot` keeps its
  historical three-arm expansion.

Tests (`tests/test_exp06_haa_pipeline.py`, +101 lines): the `INITS` table; the golden
dry-run of `yawaug_hf:0` (identical to `finetune_job('yawaug_hf', 0)`, `frame=heading
heading=<dir>`, all nine children carrying `--heading-json-dir`); `yawaug_hf:zeroshot`'s
own golden, with bare `zeroshot` still naming only the three exp_06 arms; the mixed queue
`yawaug:0 yawaug_hf:0` in both orders (room then heading, heading then room); the exp_11
record prefix; the real approvals gate refusing wrong checkpoint bytes for `yawaug_hf`
with exp_04's `checkpoints.aug` sha and never `artifacts.epoch_012`; a missing heading
directory refusing `yawaug_hf` at `job_spec` while the room-frame `yawaug` job with the
same missing directory still declares `frame: room` and no heading; and a positive control
asserting G's written `job_spec.json` carries `{room: 128}` for all four rooms, the
SimpleViT backbone and E's initialisation. The exp_06 and exp_09 golden dry runs are
unchanged and still assert byte-for-byte.

### 3.2 Summariser (`tools/exp06_summarize_haa.py`)

**(a) Arm G** (`fd1503c`). `ARMS['yawaug_hf']` = label `G`, branch `new`, root
`ckpt/exp11/sim2real/yawaug_hf`, backbone `simple`, frame `heading`, experiment `exp11`,
`init_sha256: None`. `expected_inits` resolves that null for **both** `EXP04_AUG_ARMS`
(`yawaug`, `yawaug_hf`) from exp_04's approved `checkpoints.aug` through 6.4's reused pin
and binds the record once — an `ARMS` entry alone would have left G unchecked, which the
round-1 Codex review called out explicitly. `arm_directory` already routes an arm to its
own experiment's tree, so only `--exp11-root` (default `ckpt/exp11/sim2real`) and the
`roots` map in `main` were needed.

**(b) `EXPERIMENTS['exp11']`, phase 1** (`fff7948`, `968e167`, `bc8e762`, `824dce9`,
`afd5a71`, `6af6688`, `91a48eb`).

*Schema extension shape.* The historical entries keep their exact structure
(`decisions` 5-tuples, one `screen`, one `descriptive`, `classified`). exp_11 adds a
`phase` key, and `analyse` / `render` dispatch on its presence: with `phase` they take the
exp_11 path, without it the historical path runs the code it always ran. `exp06`'s and
`exp09`'s blocks in `render` were moved verbatim into `render_decisions` (no
re-indentation, no edits) so the branch costs the historical path nothing.
`EXPERIMENTS['exp11']['decisions']` and `['classified']` are empty tuples, so every loop
the historical path runs over them is a no-op for exp_11.

*Decisions, generically.* `decision_fields(interval, margin, fields)` implements the plan's
notation block once, for a contrast X − Y with interval [L, U] at margin m: `category`
(two-sided, exactly `h2_label`'s semantics), `y_non_inferior_at_margin` (−L < m),
`x_margin_advantage` (U < −m), `equivalent_at_margin` (L > −m and U < m). The direction is
the cell's own `contrast` string (and its `x` / `y` fields); every inequality is strict,
so an endpoint exactly on a boundary establishes nothing; a margin-bearing field without a
margin, or an unregistered field name, is refused. `exp11_cell` applies the invalidity
policy, the convergence gate and those fields to one cell. exp_11 cells carry a `status`
(`reported` / `void` / `not_converged` / `unavailable` / `suppressed (draft)`) and **no**
`verdict`, because exp_11's decisions are the statements, not a margin pass/fail. Every
field is `None` when the status withholds the interval.

Registered: **N1** (G − D, `category`), **N1i** (interaction, `category`), **N2** (C − G,
`category` + `y_non_inferior_at_margin` + `x_margin_advantage`), **N3** (G − E, `category`
+ `equivalent_at_margin`). Margin `0.23` only on N2/N3 (the two that report a
margin-bearing field); N1/N1i record `margin: null`. All four are hallway C50; each cell
also carries a `reading` string, N1i's being "positive = a larger heading-frame penalty
after yaw pretraining, not harm by any single arm".

*Interaction cell (N1i).* `interaction_rows(arms, (G, E, D, A), room, metric)` builds the
four-arm cohort once: each arm paired against the first with exp_02's own
`assert_pairing` (query `index`, `ir_path`, `eval_seed`, `num_shot`), one seed-order check,
**one** mask finite across all four arms and all three seeds, each arm's own exclusions
reported, and the rows returned as `a = G − E`, `b = D − A` so the estimate is the frozen
`bootstrap.paired_intervals(g − e, d − a, clusters, seeds)` — jointly resampled, never two
independently bootstrapped intervals subtracted (a test asserts equality with a direct
call to the frozen helper on the same arrays). `interaction_void_reasons` adds the ≥ 99 %
joint-cohort rule and keeps the component checks of G − E and D − A, each reason prefixed
by its component. A zero-width interval (cancellation) is a defined `unavailable` cell:
`converged_two_way`'s refusal is caught, the helper is untouched, the summary does not
abort — and `91a48eb` narrows that catch so only a genuinely degenerate interval becomes
`unavailable` while any other refusal of the frozen helper is re-raised.

*Screens.* `exp11_screens` runs one eleven-cell Bonferroni family per declared contrast
(G − D, C − G, G − E), keyed by the contrast. `exp11_screen_cells` calls the historical
`screen_cells` unchanged and then applies exp_11's universal suppression: a cell the
invalidity policy voids or the convergence check refuses has `label: null`, `withheld:
true` and its `void_reasons`. A test asserts the same data keeps its label under
`screen_cells` (exp_06/exp_09 semantics unchanged) and loses it under `exp11_screen_cells`.

*R1.* `EXP11_HISTORICAL` is the source-field map: C − D and D − A from exp_06's descriptive
`D` list (`diff`, `nominal_two_way`), C − A from `H1` and C − F from `H1b` (`diff`,
`two_way`, `convergence`, `verdict`), E − A from exp_09's `E1` (those plus `category`,
`non_inferior_at_margin`). Rows are selected by contrast/room/metric assertions (a
duplicate match is refused, a reordered list changes nothing), copied verbatim with their
original names and interval convention, and every other decision-bearing field is `null`
and listed in `not_recorded`. Each row records `source`, `source_sha256`, `selector`,
`kind` (`descriptive` / `decision`, never promoted) and `inference: "none (copied)"`. A
missing source is refused, and so is a `summary.txt` that does not hash to the record's own
`summary_sha256`. Both source files and both summaries are bound into the analysis inputs
and revalidated at publication. A test runs the map against the **real**
`ckpt/exp06/stats.json` and `ckpt/exp09/stats.json` (C − D = −0.6289887046267822,
D − A = +0.6307393052674443, C − A `pass`, E − A `fail` / `detected harm` /
`non_inferior_at_margin false`).

*Outputs.* `ckpt/exp11/phase1/stats.json` and `ckpt/exp11/phase1/summary.txt`, with
`experiment: "exp11"` and `phase: "phase1"` in the record and on the CLI's printed line
(plus one status per registered statement). `check_output_paths` refuses an exp_11 run
targeting exp_06's or exp_09's canonical record and either of those targeting exp_11's
phase files; `write_outputs` still refuses to overwrite.

**(c) Descriptive.** No code was needed: `side_split` and `arm_rows` are driven by the
experiment's arm list, so G appears in the zero-shot room-frame −y/+y split (the split is
computed from the room-frame cache geometry for every arm, heading-frame arms included)
and carries per-seed `per_run` values. Both are asserted.

### 3.3 `tools/exp06_finalize.py`

Unchanged, as expected. G's children are ordinary heading-frame `haa_train` / `haa_eval`
children; `load_job_spec` types `init` as a non-empty string with no whitelist, the
backbone `simple` and frame `heading` are already admissible, and `exp04_aug_checkpoint`
needed no change to answer for a second arm. `code.finalize` is byte-identical.

## 4. Design decisions worth the reviewer's attention

1. **Dispatch on `phase`, not on the experiment name.** `analyse` and `render` branch on
   `'phase' in config`, so the historical path is reached by exactly the configurations it
   always was and a future exp_11 phase (1b, final) is one more entry rather than another
   branch.
2. **`status` instead of `verdict` for exp_11 cells.** A margin `verdict` would have had to
   mean something for N1 and N1i, which have no margin. The four statements are the
   decisions; `status` says only whether the interval exists.
3. **Field names are generic, direction comes from the contrast.** `y_non_inferior_at_margin`
   / `x_margin_advantage` are read against the cell's `contrast` ("cyl_or - yawaug_hf") and
   its `x` / `y` fields, so no field name has to encode an arm and the notation block is
   reusable for Q1–Q4 and P1–P4′ in round 2 unchanged.
4. **The interaction is assembled, not subtracted.** `interaction_rows` returns rows in the
   shape `cell_rows` returns, so `intervals`, `converged_two_way` and `exp11_cell` are the
   same code paths the two-arm cells use, and the frozen bootstrap does the joint
   resampling.
5. **The `unavailable` path is narrow.** Only a zero-width interval becomes a defined
   cell; every other `ValueError` out of the frozen convergence helper propagates.
6. **R1 binds a source's summary.** The plan asked for source hashes; requiring the sibling
   `summary.txt` to match the record's own `summary_sha256` additionally proves the JSON
   being copied from is the one whose rendered summary was published.
7. **Regression by comparison with `main`.** `test_exp06_and_exp09_publish_exactly_what_main_publishes`
   imports `tools/exp06_summarize_haa.py` as `main` carries it and asserts both experiments'
   full `analyse` payloads (`json.dumps(..., sort_keys=True)`) **and** their rendered
   summaries are identical, on the same fixtures, and that no exp_11 key leaks into them.

## 5. Tests

New file `tests/test_exp11_summarize_haa.py` (49 tests) reusing
`tests/test_exp06_summarize_haa.py`'s fixtures; `tests/test_exp06_haa_pipeline.py` +8 tests
(72 total); `tests/test_exp06_summarize_haa.py` updated for the registry (98 tests, green).

Coverage of the round-1 acceptance list: the `INITS` table; golden dry runs for
`yawaug_hf:0` and `yawaug_hf:zeroshot`; unchanged room-frame `yawaug` golden; mixed queues
in both orders; wrong checkpoint; missing heading; record prefix; job-spec positive
control; a room-frame G child refused (plan section 4.3's "wrong frame"); the
EXPERIMENTS freeze for exp_11; every decision field withheld on void and on
unconverged cells; the direction semantics of the four fields on nine synthetic intervals
including three boundary-equality cases; the interaction's value against the frozen
bootstrap, its pairing refusals (reversed index, changed `eval_seed`), its single mask and
per-arm exclusions, the 99 % rule, the retained component checks, the empty joint cohort
and the exact cancellation; the three screen families and their suppression versus the
historical screens; R1's five rows, its selector, its refusals and its real-file check; the
phase outputs, the CLI, and the exp06/exp09 regression.

`bash -n tools/exp06_haa_pipeline.sh`, `python -m py_compile tools/exp06_summarize_haa.py`
and `git diff --check` are clean.

**Full CPU suite** (`python -m pytest tests -q -p no:cacheprovider -rf`,
`CUDA_VISIBLE_DEVICES=''`, `PYTHONPATH=<worktree>`): at HEAD `5db0c82`,
**1 failed, 3323 passed, 50 skipped** in 62 min (log:
`orientation_cue_fairness_2026-09-26_19:20_suite_full_cpu_5db0c82.log`).

The one failure is the **expected exp_06 approvals drift guard**, exactly as exp_09's
round 1 recorded it:
`tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes`
compares every filled `code.*` digest in
`oriented_cyl_results_assets/approved_digests.json` with what this checkout computes. Of
the 20 filled keys, **18 are identical** and exactly the two of §2 differ
(`haa_pipeline_sh` `4a401ee3…` → `f33d4ffe…`, `summarize_haa` `3fc1e82d…` → `32937f96…`).
It goes green with the re-fill at the merge commit. No other test fails, and
`tests/test_exp09_sim_eval_closures.py` is green.

An earlier full run at `6af6688` (before the last five commits) gave the same single
failure: 1 failed, 3319 passed, 50 skipped in 61 min.

## 6. Open questions / notes for the Planner and the Reviewer

1. **Approvals re-fill at merge.** `code.haa_pipeline_sh` and `code.summarize_haa` move;
   `code.finalize` does not. The digests in §2 are this branch's — recompute and fill at the
   merge commit (blob-at-commit binding), as exp_09 did, before any exp_11 producer runs.
   Existing exp_06/exp_09 children stay verifiable: they bound the approvals blob of their
   own commit.
2. **`ckpt/` is a symlink in the worktree** (`ckpt -> /home/yixunhu/codespace/xRIR_code/ckpt`),
   so R1's two source files resolve to the main checkout's paths and are bound under those
   resolved names. Nothing under `ckpt/` was written.
3. **Screen-family keys** are the contrast strings (`"yawaug_hf - control_hf"`, …), not
   `S1a/S1b/S1c`. If the results page wants stable short names, say so and I will register
   them in the config.
4. **`result['decisions']`** is the ordered list of exp_11 statement names in the published
   record, while `config['decisions']` remains the historical 5-tuple table (empty for
   exp_11). The names are close; renaming the result key (`statements`?) is a one-line
   change if the Planner prefers.
5. **Margin on N1/N1i is `null`.** They report only `category`, so no margin is meaningful;
   if the record should still show +0.23 dB for context, that is a config edit.
6. **Not done in this round** (by scope): everything exp_11-owned in round 2, the record
   assets `make_results_{md,html}.py`, `planner_probes/exp11_runbook.sh`, the merge into
   `main` and the approvals re-fill, and the bounded heading-frame smoke of G on a card.
7. **Commit index.** All fourteen SHAs above are on `exp11-cue` only; they belong in
   `commits_orientation_cue_fairness.md` (branch column `exp11-cue`) when the Planner
   updates the record, and the branch is not merged.
8. **This report lives in the worktree's record directory.** The `main` checkout's copy of
   `worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/` is untracked there and
   was deliberately not written to (read-only for the Coder); copy this file across if the
   Planner wants it beside the plan.
