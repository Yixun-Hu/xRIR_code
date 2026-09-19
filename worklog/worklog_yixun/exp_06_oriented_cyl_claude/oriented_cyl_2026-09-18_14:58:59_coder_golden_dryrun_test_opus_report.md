# Coder report — exp_06 `oriented_cyl`, tests-only micro-round (golden dry run vs. the live tree), branch `exp06-fullfix`

**Coder:** Claude Opus 5 (Claude Code, Agent tool, max effort) · **Date:** 2026-09-18, local
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp06-fullfix`, base of this round
`ec7b7bf`, **tip `90ca0fb`**. `main` untouched; nothing merged, no GPU used (`CUDA_VISIBLE_DEVICES=''`
throughout), no foreign process signalled, nothing written in the main tree except this report.
**Round:** `coder_prompts/golden_dryrun_test_opus_prompt.md` — make
`tests/test_exp06_haa_pipeline.py::test_the_dry_run_of_one_finetune_seed_is_the_golden_queue`
independent of the live `ckpt/exp06/sim2real`. **No source file changed**:
`tools/exp06_haa_pipeline.sh` is byte-identical to `main`
(`sha256 6cb9a2cbc296cf38cfdf0ad93986d73dd735453828df5d809a4a78954f270b33`, `git diff main -- <it>`
empty), and `git diff main...HEAD --stat` is one file: `tests/test_exp06_haa_pipeline.py`
(+75 / −21).

## 1. What was wrong

The test ran the CLI — `bash tools/exp06_haa_pipeline.sh 7 cyl_or:0 --dry-run` with `cwd` at the
repo root — and compared its whole output with the golden queue. The CLI's `OUT` is the literal
`ckpt/exp06/sim2real` (line 32, no environment override, and `exp06_launch.sh` `cd`s to the repo
root when sourced), and in the worktree `ckpt` is a symlink to the main tree's. `child()` skips a
directory that already carries `completion.json`, so with the confirmatory runs in place the dry run
printed

    SKIP ckpt/exp06/sim2real/cyl_or/seed0/stage1

in place of the six lines of that child's block, nine times over, and the comparison failed on a
perfectly healthy pipeline. Reproduced in this worktree at `ec7b7bf`:
`At index 8 diff: 'SKIP …/stage1' != 'MKDIR …/stage1'`, right side 45 items longer.

## 2. What changed (tests only, two commits)

| commit | subject | changed lines |
|---|---|---|
| `059d28e` | exp06: the golden queue is what the pipeline prints, not what one tree holds | 68 (+59 / −9) |
| `90ca0fb` | exp06: the two remaining dry runs stop consulting the live output root | 28 (+16 / −12) |

**The mechanism.** A new helper runs the queue in library mode, exactly as the prompt asked and as
Codex reproduced the golden lines in its close review:

```
DRY_ADAPTER = '''EXP06_PIPELINE_LIB=1 source tools/exp06_haa_pipeline.sh
DRY=1; STAMP='<UTC>'; OWNER='<pid>'      # the placeholders the CLI dry-run path sets
GPU="$QUEUE_GPU"; OUT="$WORK/sim2real"; RECORD="$WORK/record"
run_queue $QUEUE_JOBS
'''
```

`run_dry(work, *jobs, gpu='7', **environment)` runs that with `cwd` at the repo root (so the sourced
`exp06_launch.sh` still defines `say`/`run`/`finalize`/`PYTHON` the same way), then substitutes
`<work>/sim2real` → `ckpt/exp06/sim2real` and `<work>/record` → the record directory in the printed
lines. **The golden helpers are untouched** — `header`, `finetune_job`, `zeroshot_job`,
`child_block`, `train_command`, `eval_command`, `job_finalize`, `log_of`, `approvals_line` are all
exactly as they were, and the comparison is still line-for-line against them (62 lines for
`cyl_or:0`, the same 62 Codex replayed). The substitution is a pure path substitution applied to the
*produced output*, not to the golden.

`DRY=1` means nothing is executed and nothing is written: `approvals_ok`, `job_spec`, `check_spec`
skip their `$PYTHON` calls, `open_job` returns before `mkdir`, `child()` returns after `finalize`
prints. The only filesystem access left in the queue is `child()`'s `-f "$dir/completion.json"`,
which now looks inside the temporary root.

**Tests after the round** (all in `tests/test_exp06_haa_pipeline.py`):

- `test_the_dry_run_of_one_finetune_seed_is_the_golden_queue(tmp_path)` — library mode, 62 golden
  lines, no live tree.
- `test_a_completed_child_is_skipped_and_the_rest_of_the_queue_is_unchanged(tmp_path)` — **new**;
  writes `<tmp>/sim2real/cyl_or/seed0/stage1/completion.json` and asserts the output is the golden
  queue with exactly that child's six-line block replaced by one `SKIP <dir>` (the block width is
  read from `child_block`, not hard-coded). This is the live tree's behaviour, asserted on a tree
  the test owns.
- `test_the_dry_run_of_the_zeroshot_job_covers_every_init(tmp_path)` — same conversion. It passes
  today only because no zeroshot job has run yet; it would have failed the same way as soon as one
  did.
- `test_the_cli_announces_the_gpu_the_jobs_and_the_output_root()` — **new**; keeps CLI coverage of
  the two header lines (`EXP06_HAA_PIPELINE …` / `ENV …`), the `<UTC>`/`<pid>` placeholders and the
  argument parsing. These are printed before the queue inspects anything, so no state under the
  output root can change them; it is the only assertion left that runs the script the way the
  runbook does.
- `test_the_heading_directory_and_pretrained_checkpoint_are_overridable(tmp_path)` and
  `test_a_checkpoint_path_with_spaces_is_never_split(tmp_path)` — converted in commit `90ca0fb`
  (see §4).

Unchanged: `test_the_script_parses`, the refusal tests (they exit 2 before the header, never
reaching `OUT`), and every `run_lib`-based test, which already used a private `OUT="$WORK/out"`.

## 3. Verification

| check | result |
|---|---|
| `CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_exp06_haa_pipeline.py -q -p no:cacheprovider` (worktree, live run dirs present) | **49 passed** in 152 s |
| the same file in a clone of `90ca0fb` with **no** `ckpt/exp06/sim2real` at all | **49 passed** in 129 s |
| mutation check: in a second clone, `SKIP` → `SKIPPED` and `child.pid` → `kid.pid` in the script | both goldens **fail**, naming the changed line |
| `git diff --check` | clean |
| `git diff main...HEAD --stat` | `tests/test_exp06_haa_pipeline.py` only, +75 / −21 |
| `bash -n tools/exp06_haa_pipeline.sh` | ok, and the file is byte-identical to `main` |

The "without live run dirs" run is a `git clone --local` of the branch into the scratchpad whose
`ckpt/` contains only symlinks to the frozen inputs (`ckpt/exp06/heading`, `ckpt/exp06/pretrain`,
`ckpt/xRIR_cyl_8_shot`, `ckpt/xRIR_simple_8_shot`) and deliberately **no** `ckpt/exp06/sim2real`;
`git diff main --stat` in the worktree also lists `main`'s own newer record commits (the branch is
based on `ec7b7bf`, which `main` carries), which is why the branch-only comparison above is the
three-dot one.

## 4. The other tests that read the live tree (the requested grep)

`grep -rn "sim2real\|gate_g1" tests/*.py` plus a sweep of `ROOT/REPO / 'ckpt…'` constants:

- **`ckpt/exp06/gate_g1.json`: no test reads the real one.** Every `gate_g1` occurrence is a
  `tmp_path` file (`test_exp06_mirror_probe.py:464`, `test_exp06_summarize_haa.py:663`) or a
  synthetic digest in a fixture (`test_exp06_compare.py`, `test_exp06_profiles.py`,
  `test_exp06_finalize.py`).
- **`ckpt/exp06/sim2real`: after this round, no test reads the real one either.** Before it, four
  did — the two goldens and these two, which the second commit converted:
  - `test_a_checkpoint_path_with_spaces_is_never_split` asserted `--init <path with spaces> --rooms`,
    which appears **only** in the stage-1 `RUN` line of `cyl_or:2`. `ckpt/exp06/sim2real/cyl_or/seed2/stage1`
    is training right now; the moment it is finalised that line becomes `SKIP` and the test would
    have failed exactly like the golden. (The new SKIP test demonstrates the mechanism.)
  - `test_the_heading_directory_and_pretrained_checkpoint_are_overridable` asserted substrings that
    survive a `SKIP`, so it would not have failed, but it raced the live tree for no benefit.
  - `test_exp06_summarize_haa.py:141,449` only compare the `ARMS` constant strings
    `'ckpt/exp06/sim2real/<arm>'`; they never open them.
- Other exp_06 tests do read the real `ckpt/`, but only **frozen artefacts of finished
  experiments**, none of which any running process writes: `ckpt/sim2real` (exp_02;
  `test_exp06_bootstrap.py:13`, `test_exp06_finalize.py:807`, `test_exp06_summarize_haa.py:235/425/585`,
  the last guarded by `skipif`), `ckpt/yaw_aug/eval` (exp_04; `test_exp06_compare.py:753/1194`),
  `ckpt/xRIR_cyl_8_shot` (exp_01; `test_exp06_legacy_train_receipt.py:242`,
  `test_exp06_sim_evidence.py:263`), `ckpt/exp05/S_simple/attempt_20260913T123905/args.json`
  (exp_05; `test_exp06_recipe.py:16`). I left all of these alone: they are inputs the record binds
  by hash, not live output roots.
- `tests/test_exp06_haa_pipeline.py`'s real-approvals tests (`REAL_ADAPTER`) still read the record's
  `approved_digests.json` and `ckpt/exp06/heading/*.json` through the producer gate. Those are the
  approved, hash-bound artefacts the gate exists to check, not run state, so they stay.

## 5. Notes for the reviewer

1. The one place that still invokes the CLI at the repo root is
   `test_the_cli_announces_the_gpu_the_jobs_and_the_output_root`. The dry run it starts does `stat`
   the nine children under the live `ckpt/exp06/sim2real` (that is `child()`'s skip check), but the
   test only asserts the first two lines and the exit status, both of which are produced before any
   child is inspected and are identical whether or not those directories exist. I kept it because
   dropping it would have left the CLI header, the `<UTC>`/`<pid>` placeholders and the queue's
   argument handling with no coverage at all. If the Planner prefers zero contact with that root, the
   test can be deleted and the header covered only by `header()`'s use in the goldens — say the word.
2. `run_dry` sets `OUT` and `RECORD` but not `HEADING_DIR`/`HAA_ROOT`: those are already environment
   parameters of the script (`EXP06_HEADING_DIR`, `HAA_XRIR_ROOT`), the golden lines print the real
   `ckpt/exp06/heading` string, and under `DRY=1` nothing opens them.
3. The golden file/helpers were not modified, so a future change to what the queue prints still
   fails these tests for the right reason.
