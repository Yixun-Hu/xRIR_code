# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 12**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `fdb74fc`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU
**Started:** 2026-09-27 19:3x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close11_review.md` via `coder_prompts/round2_fix12_opus_prompt.md`

**Signals:** the only processes signalled are helpers these tests started — `sleep` stubs
and `flock` holders — through their own `Popen` handles. `kill -0` remains a probe.

## Commits

| SHA | Subject |
|---|---|
| `ae6d977` | exp11 fix12 (RED): the reader's exit 1 is both "no record" and "I broke" |
| `1ecd1f9` | exp11 fix12 (GREEN): the reader answers, or it does not answer at all |
| `f3e5504` | exp11 fix12 (RED): a broken reader retires a registered, running trainer |
| `9d742ad` | exp11 fix12 (GREEN): nothing is decided on an answer nobody gave |
| `05bff3e` | exp11 fix12 (tests): the tombstone says what was established |
| `80df5d6` | exp11 fix12 (docs): the rule in the launcher's own header |
| `a954fe0` | exp11 fix12 (RED): a verdict line whose pid is not a pid is still trusted |
| `2fc5300` | exp11 fix12 (GREEN): a verdict that is not digits is not a verdict |
| `a700a24`, `f29e429`, `9ea21f5` | exp_11 bookkeeping: this report, close review 11, the fix-12 prompt, and two superseded suite logs dropped (record only) |

## The blocker [P1] — a failed reader was read as "not alive"

Exit 1 meant two different things: *this file holds no record*, and *I could not run at
all*. An unimportable module exits 1. A missing interpreter exits 127 through a shell
that then reports 1. A crash exits 1. The launcher could not tell any of them from a
verdict, so a broken reader made every pid file in the arm look dead — and
`resolve_unregistered`, which deliberately ignores its own target's marker (the
tombstone is what makes that safe), then tombstoned and renamed the directory of a
**registered, running** trainer.

Reproduced first, as the RED `f3e5504`, three ways — a reader that cannot run, a real
`ImportError` of the real module, and one that answers everything except `train.pid`:

```
LOCK …/.publish.lock
LOCKED …
RESOLVED …/attempt_20260927T161616 registered no child and nothing of it is alive
```

— with the trainer's `train.pid` naming a live process the whole time.

### The verdict protocol

`tools/exp11_pidrecord.py`'s CLI now answers, or does not answer at all. Both answers
are answers: one line on stdout and exit 0.

```
record <pid>     this file holds that pid record
norecord         it holds no record -- missing, unreadable, a directory, bad bytes
```

Anything else — a non-zero exit, an empty stdout, a different first word, more than one
line, a misuse (now exit 2, not 1) — is the reader **failing**, never a verdict about
the file. `pid_record(path)` is unchanged for the trainer.

### Three states in the launcher

`pid_record <file>` returns 0 and the pid, 1 for no record, or `READER_UNKNOWN=2`, and
never folds 2 into 1; `pid_alive` and `registration_complete` carry the third state.
`scan_arm` probes through one helper,

```bash
probe_live "$attempt/train.pid" "$name has a live trainer (train.pid)" || return $?
```

which sets `SCAN_REASON` and returns 1 (live) or 2 (unknown) for the caller to
propagate unchanged. Any unknown from any attempt aborts the scan; `require_quiet_arm`
and `resolve_unregistered` return 2 and print

```
refusing: liveness unknown: pid reader failed on <file>. Nothing is scanned, retired or
published on an answer nobody gave
```

Nothing is written and the arm is left exactly as it was. `clear_launching` leaves the
marker standing on unknown — a marker is withdrawn on an answer, never on the absence of
one. The launcher's header now states the rule: **a broken reader closes the arm to every
automated decision; it never opens it.**

### The health check

Before any decision — at the top of `require_quiet_arm`, which is the one scan both
modes run, and at the top of `resolve_unregistered`, which does not go through it —
`check_pid_reader` asks the reader two questions whose answers are known: a file holding
`1\n` must come back `record 1`, an empty file must come back `norecord`. It uses the
same `$PYTHON` and the same invocation as every real read. A reader stuck on one verdict
never *fails*, so only a probe catches it, and one probe is not enough: `record 1` for
everything is caught by the empty file, `norecord` for everything by the file holding 1.
On any mismatch the invocation refuses with `pid reader unhealthy` before touching
anything.

### The regressions

All against the real file, sourced as a library, with a `$PYTHON` wrapper that breaks
**only** the pid reader — everything else, the lock holder above all, keeps the real
interpreter, so the publication lock is taken normally and the reader fails afterwards,
which is the reviewer's schedule exactly.

| test | what it does |
|---|---|
| `test_a_broken_reader_never_retires_a_live_trainers_attempt` (3 kinds) | marker + complete dead wrapper pid + live registered trainer; lock taken, reader broken, `resolve_unregistered` → exit 2, no tombstone, no rename, marker intact, `train.pid` still naming the live stub |
| `test_a_broken_reader_stops_the_scan_of_either_mode` (`full`, `finalize`) | the same injection around `require_quiet_arm` → exit 2, `liveness unknown`, nothing written |
| `test_a_broken_reader_stops_a_recovery_through_the_cli` | the launcher itself, via `EXP11_PYTHON` → exit 2, no `PROMOTE`, the attempt untouched |
| `test_the_interpreter_override_is_refused_outside_the_test_roots` | `EXP11_PYTHON` without the test roots and a dry run → refused |
| `test_the_health_check_catches_a_reader_that_answers_everything_the_same` (2) | always-`record 1` caught by the empty probe, always-`norecord` by the `1\n` probe |
| `test_a_broken_reader_leaves_the_marker_standing` | `clear_launching` with a complete `child.pid` and a `train.exit` but a broken reader → the marker stays |

The three kinds of broken reader are: **cannot run** (`exit 1` with nothing on stdout —
what a missing interpreter, an unimportable module and a crash all look like), **import
error** (a wrapper that `cd`s into a temporary tree holding a `tools/exp11_pidrecord.py`
that raises, so the real module path really fails to import — no worktree file is
touched), and **fails on the trainer** (healthy on the probes, blind to `train.pid`,
which is what exercises the scan's unknown branch rather than the health check).

`EXP11_PYTHON` is the one new affordance: it replaces the interpreter that reads pid
files so the fail-closed paths can be exercised against this launcher rather than a copy
of it. Like `EXP11_PRETRAIN_ROOT` it is honoured **only** with `EXP11_TEST_ROOTS=1` and
`--dry-run`, and the invocation is refused outright otherwise — a real launch decides
liveness with the interpreter the pinned library names.

### The byte table is still definite

`tests/test_exp11_pidrecord.py` (72 tests) now asserts of every row that the CLI exits 0
with exactly `record <pid>\n` or `norecord\n` and nothing on stderr, that the shell's
`pid_record` returns 0 or 1 and **never** 2, and that the shell and Python still never
disagree. A misuse of the CLI exits 2 with no verdict line.

## Wording (nonblocking, both from close review 11)

* The tombstone's `reason` line said "no complete child.pid", and the `RESOLVED` line
  said "registered no child and nothing of it is alive". Neither is true of the
  wrapper-loss state a resolution may now retire. They now say what was actually
  established — a launching marker with no finished trainer, and no *recorded* pid of
  the arm alive — and name the tombstone as the defence against a trainer that never
  registered and so could not be seen at all.
* The fix-11 report's "in no approval key at all" is corrected to what was true: the
  reader already belonged to the transitive Python closures of `train`, `finalize`,
  `smoke` and `summarize_haa`; it was missing from `launch_sh` specifically.

## Line counts

| commit | production | tests | record |
|---|---:|---:|---:|
| `ae6d977` (RED) | — | +47/−17 | — |
| `1ecd1f9` (GREEN) | `exp11_launch.sh` +31/−11, `exp11_pidrecord.py` +17/−6 | — | — |
| `f3e5504` (RED) | — | +174 | — |
| `9d742ad` (GREEN) | `exp11_launch.sh` +91/−15 | — | fix-11 report +4/−2 |
| `05bff3e` (tests) | — | +5/−1 | — |
| `80df5d6` (docs) | `exp11_launch.sh` +6 | — | — |
| `a954fe0` (RED) | — | +8/−1 | — |
| `2fc5300` (GREEN) | `exp11_launch.sh` +7/−1 | — | — |
| range `fdb74fc..HEAD` | **+151 / −32 = 183 changed** | +233/−18 | +4/−2 |

Every commit is under 200 changed lines; no exception this cycle (the three record-only
bookkeeping commits are larger, as they always are).

## Frozen files

`tools/exp06_launch.sh` and `tools/exp06_finalize.py` are byte-identical to `9f98bbb`
(`git diff 9f98bbb..HEAD` reports nothing for either). No `model/*.py`, no `tools/exp06_*`
file, no `tools/exp10_*`, no exp_03-pinned file was touched. `bash -n` passes on both
shells, `py_compile` on every exp_11 module, and `git diff --check fdb74fc..HEAD` is
silent.

## Digests

Taken at `2fc5300`, the cycle's last production commit.

**exp_06** — exactly **one** key differs from the record re-filled at `9f98bbb`, and it is
the only one allowed to move (`tools/exp06_summarize_haa.py` imports `exp11_finalize` and
`exp11_profiles`, so every exp_11 change reaches its closure):

| key | approved at `9f98bbb` | fix 11 | now |
|---|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `2f2f4383c94d…` | `650a5a6d97cb…` |

The other 19 exp_06 keys are identical to the approved record.

**exp_11** — all eight keys, and which moved against fix 11's tip `fdb74fc`:

| key | digest | vs `fdb74fc` |
|---|---|---|
| `train` | `6cd346cfc892ea27f1133909fadffb5087fbba343778a7eba9668ea2fc11e46f` | **moved** |
| `finalize` | `f808e7f0c9a711be32670c54fd22dea5daf0a95772329a9fca722a54198a06f3` | **moved** |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` | unchanged |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` | unchanged |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` | unchanged |
| `launch_sh` | `4153a2c749be5df4f092374d139dbcc73a8118e2966a3ce2be208c244fbfe2e0` | **moved** |
| `smoke` | `60571786e0acb996d847d15d747a819c3b8e6616cdb54247b347e3a8d00749f9` | **moved** |
| `summarize_haa` | `650a5a6d97cbc8a3c4c54d647a3c75a993d9f79eb22b2c9b10034d357950a660` | **moved** |

The same five as the last four cycles, for the same two reasons: `launch_sh` through its
own shell bytes and `tools/exp11_pidrecord.py`, which its spec now binds; `train`,
`finalize`, `smoke` and `summarize_haa` because the reader is in all four closures
through the trainer's import. The three HAA keys are untouched. exp_11's approvals record
remains all-null and every producer refuses today.

## Test results

Every run CPU-only (`CUDA_VISIBLE_DEVICES=''`), in the `xRIR` env.

| run | result |
|---|---|
| `tests/test_exp11_shell.py`, `test_exp11_pidrecord.py`, `test_exp11_train.py`, `test_exp11_lock_holder.py` | **255 passed** (128 s), and **200 passed** for the two files again after the last GREEN |
| the other exp_11 files + `tests/test_exp06_summarize_haa.py` + `tests/test_exp09_sim_eval_closures.py` | **291 passed** (729 s) |
| full CPU suite at the tip `2fc5300`, detached | _(below)_ |

One older fixture changed with the wording: `test_the_resolution_writes_a_tombstone_...`
asserted the word "unregistered" in the tombstone, and now asserts the substance the
rewritten reason carries (`05bff3e`).
