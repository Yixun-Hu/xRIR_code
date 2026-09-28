# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 15**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `24fdd2b`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU
**Started:** 2026-09-27 23:4x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close14_review.md` via `coder_prompts/round2_fix15_opus_prompt.md`

**Signals:** the only processes signalled are helpers these tests started — `sleep` stubs
and `flock` holders — through their own `Popen` handles. `kill -0` remains a probe.

This cycle was asked to close a **family**, not two instances: *a failed filesystem
inspection read as a definite absence*. It does that by giving every path question three
answers in both languages, routing every guard through them, and auditing the rest.

## Commits

| SHA | Subject |
|---|---|
| `a5ab870` | exp11 fix15 (RED): a symlinked arm root is an arm with nothing in it |
| `20336b5` | exp11 fix15 (GREEN): enumerate the directory the root denotes |
| `37321c9` | exp11 fix15 (RED): four guards that read a failed inspection as an absence |
| `1b9eff5` | exp11 fix15 (GREEN): three states for every path the launcher asks about |
| `d8c71f8` | exp11 fix15 (RED): the trainer reads an uninspectable path as an absent one |
| `cfe9565` | exp11 fix15 (GREEN): the trainer asks os.stat, and accepts three answers |
| `53d068a` | exp11 fix15 (RED): a reader that never returns hangs the launcher |
| `7a2ff48` | exp11 fix15 (GREEN): one temp directory, one trap, a bounded reader, a probed resolver |

## Blocker 1 [P1] — the root is a directory, not a name

`find -P` does not descend a symlink handed to it as a starting point, while `-d`, `-r`
and `-x` all follow one: the listing came back **empty and successful** for an arm with a
live registered trainer in it. `list_attempts` now canonicalises first, with a checked
`realpath -e` — a root that cannot be resolved is not an empty arm — and enumerates the
canonical path. RED `a5ab870` drives both scans and a real resolution through a symlink
root; GREEN `20336b5` makes both scans see the trainer and the resolution refuse.

## Blocker 2 [P1] — three states, in both languages

### The shell

```bash
probe_path <path>   # 0 it is there (PROBE_MODE = its %f mode) | 1 it is NOT there | 2 nobody could say
file_present <path> # 0 a regular file is there | 1 no regular file is there | 2 unknown
probe_link_target <path>  # 0 + canonical target | 1 there is no such link | 2 unknown
```

An **absence is a claim**, and `probe_path` makes it only about a directory it could
enter: a `stat -L` that fails is followed by a `stat` of the name itself (a symlink that
dangles, loops, or points somewhere we may not go is *not* an absence), then by a `stat`
of the parent and a search-permission check. Only when all of that holds does it say
"not there". `probe_link_target` asks about the **link**, then resolves it with a checked
`realpath -e`; `readlink -f` used to answer "nothing" both for "there is no `final`" and
for "there is one I cannot resolve".

### The trainer

`probe_path` / `file_state` / `directory_state` over `os.stat` + `os.lstat`, with the
same three states and the same rule about the parent, and `demand_known` refusing with
exit 3 rather than acting on a path the process could not inspect. `Path.is_file()` is
used for no decision anywhere in the registration path: it swallows the `ELOOP` of a
self-referential `<attempt>.resolved` and returns False, which let a trainer whose launch
had been retired register anyway.

## The audit (blocker 2c)

Every existence/type test in `tools/exp11_launch.sh` and in the trainer's registration
path, at `7a2ff48`. "Routed" means the answer comes from a three-state probe and an
unknown refuses; "definite" means a failure there cannot produce absent/dead/quiet/
unpublished.

### `tools/exp11_launch.sh`

| line | site | disposition |
|---|---|---|
| 330, 337, 340, 341 | `probe_path` itself (`stat -L`, `stat`, parent `stat`, `-x`) | **the probe**: every failure it cannot explain is unknown |
| 360–368 | `probe_link_target` (`stat` of the link, parent, `realpath -e`) | **the probe**: an unresolvable link is unknown, never "no link" |
| 381 | `pid_record`: `stat -L -c %f` classification of the pid file | definite only when it SUCCEEDS; a failure delegates to the module, which distinguishes a confirmed absence from an inability to look |
| 394 | `pid_record`: `stat -c %s` of the verdict file | a temp file this process just created in its own directory; a failure is unknown |
| 494–497 | `list_attempts`: `realpath -e`, `-d`, `-r`, `-x`, `find` | every failure → 1 → the caller's unknown ("cannot be enumerated") |
| 510, 537 | `scan_arm`: `realpath` of the resolution target and of each entry, with a literal fallback | identity comparison only. A failure makes the two spellings unequal, so the attempt is **not** excluded from the scan — the strict direction |
| 527 | `scan_arm`: per-entry `stat -L -c %f` | routed: a `stat` that fails is unknown, not "not a directory" |
| 548–550 | `scan_arm`: `launching`, `train.pid`, `train.exit` | **routed** (`file_present`); any unknown closes the arm |
| 611 | `clear_launching`: `train.exit` | routed; absent **or** unknown keeps the marker |
| 628 | `resolve_unregistered`: is the target a directory | routed (`probe_path` + mode bits); unknown refuses |
| 636, 639 | `resolve_unregistered`: `realpath -e` of the target and the arm root | checked; a failure refuses as unknown |
| 659 | `resolve_unregistered`: `stat -L` of `<attempt>/.` | routed: a directory we cannot enter is unknown |
| 666 | `resolve_unregistered`: the `launching` marker | **routed**; unknown refuses |
| 683 | `resolve_unregistered`: `completion.json` | **routed**; unknown refuses (no retirement) |
| 694 | `resolve_unregistered`: the published `final` | **routed** (`probe_link_target`); unknown refuses |
| 801 | `published_target` | **is** `probe_link_target` |
| 808–812 | `already_published` | **routed**; unknown propagates and the caller exits 2 |
| 819 | `check_final_conflict` | **routed**; an unresolvable `final` is never replaced |
| 842–843 | `abort_recovery` | **routed**; unknown refuses to abort (it may be published or complete) |
| 776 | `scan_barrier`: `while [ -e "$file" ]` | justified: a test-roots-only affordance, reachable only with `EXP11_SCAN_BARRIER` set **and** `EXP11_TEST_ROOTS=1`; a failed test releases the barrier, which grants nothing |
| 944 | `[ ! -d "$DATA_ROOT" ]` | justified: a failure **refuses the invocation** ("is not a directory"), which is the closed direction |
| — | `promote` (`ln -s` + `mv -Tf`) | justified: it tests nothing; the rename is atomic and unconditional |
| — | `preflight`, `check_attempt` | justified: both are Python (`tools/exp11_finalize.py`, the inline `CHECK_ATTEMPT_PY`) and refuse on any `OSError`, as close review 14 confirmed |
| — | `hold_arm_lock`, `drop_arm_lock`, the registration lock | justified: every failure refuses access rather than granting it |

### `tools/exp11_train.py` (the registration path)

| site | disposition |
|---|---|
| `probe_path` / `file_state` / `directory_state` | the probe: `os.stat`, then `os.lstat`, then the parent |
| `register_trainer`: the `--no-save` tombstone check | **routed**; unknown refuses (exit 3) |
| `register_trainer`: the run directory | **routed** (`directory_state`) |
| `register_trainer`: the arm root | **routed** (added this cycle) |
| `register_trainer`: attempt-in-arm identity | `os.path.realpath` comparison of two paths both already probed present |
| `register_trainer`: the tombstone, under the lock | **routed**; unknown refuses |
| `register_trainer`: the `launching` marker | **routed**; unknown refuses |
| `register_trainer`: `child.pid` | **routed** through `exp11_pidrecord.inspect_record`; unknown refuses with its own reason |
| `run()`: the two "did the run directory vanish" checks | **routed** (`directory_state`); unknown refuses |
| `registration_lock`, `record_trainer_exit` | justified: a failure to open the lock refuses; the exit receipt swallows `OSError` by design, and a missing receipt only ever keeps an arm closed |

No site remains where a failure to inspect yields absent, dead, quiet or unpublished.

## The regressions (blocker 2d)

`tests/test_exp11_shell.py`, all non-root (they skip when the euid is 0):

| test | the reviewer's case |
|---|---|
| `test_a_symlinked_arm_root_is_still_enumerated` | both scans see the live trainer through a symlink root; the resolution refuses and leaves the attempt, marker and `train.pid` alone |
| `test_a_dangling_symlink_root_is_unknown` | a root that cannot be resolved is unknown, not empty |
| `test_a_marker_we_cannot_inspect_closes_the_arm` | `launching` as a symlink into a mode-000 directory, every pid record definitely absent → exit 2 |
| `test_a_receipt_we_cannot_inspect_is_never_retired` | `completion.json` behind a mode-000 directory → the resolution refuses; no tombstone, no rename |
| `test_a_final_we_cannot_resolve_is_never_replaced` | `final` reaching its target through an inaccessible intermediate → the recovery refuses, no `PROMOTE`, no `REPLACING` |
| `test_the_publication_guards_refuse_an_unresolvable_final` (3) | `already_published`, `check_final_conflict` and `abort_recovery` each return 2 and change nothing |
| `test_an_entry_the_scan_cannot_stat_is_unknown` | rewritten so the arm holds **only** the inaccessible entry: the answer cannot come from another one whichever order the listing arrives in |
| `test_a_reader_that_never_returns_is_unknown` | a `$PYTHON` that sleeps forever → unknown within the bound, not a hung launcher |

`tests/test_exp11_train.py`:

| test | the reviewer's case |
|---|---|
| `test_a_tombstone_that_cannot_be_inspected_refuses_registration` | a self-referential `<attempt>.resolved` (ELOOP) → exit 3, nothing registered |
| `test_the_diagnostic_tombstone_check_is_three_state_too` | the same for the `--no-save` branch |
| `test_a_launch_record_that_cannot_be_inspected_refuses_registration` | the `launching` marker behind a mode-000 door → exit 3 |
| `test_a_run_directory_that_cannot_be_inspected_refuses_registration` | a run directory nobody may look at is not one that vanished |

## The nonblocking items

* **Temp files.** Every listing and verdict of an invocation now lives in one directory
  created at start-up, and `trap 'rm -rf -- "$EXP11_TEMPDIR"' EXIT` takes all of them at
  once — including on the abrupt paths, where the individual `rm`s never ran.
  `hold_arm_lock`'s EXIT trap runs both handlers.
* **A bounded reader.** `timeout "$READER_TIMEOUT_S"` (30 s, `EXP11_READER_TIMEOUT_S` for
  the regression) wraps every reader invocation — the verdict reads and both health
  probes. An answer that never comes is not an answer: expiry is unknown.
* **A deterministic inaccessible-entry test**, as above.
* **The shell test count** is **163 collected** in `tests/test_exp11_shell.py` at this
  tip (154 was the count before this cycle's nine additions).

## Line counts

| commit | production | tests |
|---|---:|---:|
| `a5ab870` (RED) | — | +41 |
| `20336b5` (GREEN) | `exp11_launch.sh` +10/−4 | — |
| `37321c9` (RED) | — | +95 |
| `1b9eff5` (GREEN) | `exp11_launch.sh` +134/−20 | — |
| `d8c71f8` (RED) | — | +67 |
| `cfe9565` (GREEN) | `exp11_train.py` +87/−9 | — |
| `53d068a` (RED) | — | +24/−6 |
| `7a2ff48` (GREEN) | `exp11_launch.sh` +36/−10 | — |
| range `24fdd2b..HEAD` | **+267 / −43 = 310 changed** | +227/−6 |

Every commit is under 200 changed lines (the largest is `1b9eff5` at 154); no exception
this cycle.

## Frozen files

`tools/exp06_launch.sh` and `tools/exp06_finalize.py` are byte-identical to `9f98bbb`.
No `model/*.py`, no `tools/exp06_*` file, no `tools/exp10_*`, no exp_03-pinned file was
touched. `bash -n` passes on both shells, `py_compile` on every exp_11 module, and
`git diff --check 24fdd2b..HEAD` is silent.

## Digests

Taken at `7a2ff48`, the cycle's last production commit.

**exp_06** — one key differs from the record re-filled at `9f98bbb`, and it is the only
one allowed to move (`tools/exp06_summarize_haa.py` imports `exp11_finalize` and
`exp11_profiles`, so the trainer's closure reaches it):

| key | approved at `9f98bbb` | fix 14 | now |
|---|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `40626dd26bde…` | `22a9858b067e…` |

The other 19 exp_06 keys are identical to the approved record.

**exp_11** — all eight keys, against fix 14's tip `24fdd2b`:

| key | digest | vs `24fdd2b` |
|---|---|---|
| `train` | `b7eef052dc62a8223e57abc248af4c5838c503246e5747649578750e9dec0029` | **moved** |
| `finalize` | `4894acefecc9e82d9f2b66785361c439e9463e5f339ee78ef4589b4fa82babb1` | **moved** |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` | unchanged |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` | unchanged |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` | unchanged |
| `launch_sh` | `0a4141488f35ec626648afd7a16389893bccb58a8f95684df3f4749c37b45a2e` | **moved** |
| `smoke` | `53246ae8f6c970bb9732de9f9a2c7174b8d90e7a6f002ba1ac74fa77c8c9c0cf` | **moved** |
| `summarize_haa` | `22a9858b067ed1fdbf7e2d6670d81e60d4d1be7fb903d802696e7a52d43c0489` | **moved** |

The familiar five: `launch_sh` through its own shell bytes, and `train`, `finalize`,
`smoke`, `summarize_haa` because `tools/exp11_train.py` is in all four closures — as the
prompt predicted, both the launcher and the trainer moved this cycle. The three HAA keys
are untouched. exp_11's approvals record remains all-null and every producer refuses
today.
