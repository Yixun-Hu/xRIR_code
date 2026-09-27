# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 11**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `51e34b1`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU
**Started:** 2026-09-27 17:0x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close10_review.md` via `coder_prompts/round2_fix11_opus_prompt.md`

**Signals:** the only processes signalled are ones this cycle's own tests started — the
`flock` holders through their `Popen` handles, and, new this cycle, the test's own GNU
`timeout` wrapper, whose pid is checked against the command line the test itself built
(`/proc/<pid>/cmdline` must contain `timeout` and this test's stub path) before anything
is sent to it. `kill -0` remains a probe, never a signal.

## Commits

| SHA | Subject |
|---|---|
| `7f253f8` | exp11 fix11 (RED): one pid-record reader, and the shell must not be the other |
| `68f1f70` | exp11 fix11 (GREEN): `tools/exp11_pidrecord.py` is the only reader of a pid file |
| `afefa2e` | exp11 fix11 (RED): a dead wrapper pid does not account for the trainer |
| `1f0ca5e` | exp11 fix11 (GREEN): the marker stands until the trainer itself is accounted for |
| `75fd446` | exp11 fix11 (RED): a wrapper-loss arm can never be reopened |
| `6e3a490` | exp11 fix11 (GREEN): the operator can answer a wrapper-loss attempt |
| `01ceecf` | exp11 fix11 (RED): the bounded registration wait is measured on the wall clock |
| `b3872a2` | exp11 fix11 (GREEN): the registration wait runs on `time.monotonic()` |
| `ddeeb82` | exp_11 bookkeeping: fix-10 report counts corrected; close review 10 and the fix-11 prompt (record only) |
| `678d9df` | exp11 fix11 (RED): the launcher's new reader is bound by no approval key |
| `275be24` | exp11 fix11 (GREEN): `launch_sh` binds the pid reader it runs |
| `6d7363b` | exp11 fix11 (tests): a generous window for the real stub trainer |
| `03ccee4` | exp_11 bookkeeping: this report (record only) |

## Blocker 1 [P1] — one pid-record *reader*, not two implementations of one grammar

Fix 10 wrote the grammar once in prose and twice in code. The shell copy read the file
*through Bash*, and command substitution drops NUL bytes: `<digits>\0`, `\0<digits>` and
`145\x007170` all reached the length check as plain digits, so `registration_complete`
said yes where Python said no. A dead pid with a NUL therefore made an arm read quiet
with a trainer in it, and withdrew the marker.

There is now exactly one reader, `tools/exp11_pidrecord.py` (49 lines):

```python
PID_RECORD = re.compile(rb'[0-9]{1,10}\n?')

def pid_record(path):
    """The pid ``path`` holds, or ``None``. Bytes, never text: CRLF is not a newline."""
    try:
        with open(str(path), 'rb') as handle:
            data = handle.read(64)
    except OSError:
        return None
    return int(data.decode('ascii')) if PID_RECORD.fullmatch(data) else None
```

with a CLI that prints the pid and exits 0, or exits 1 and prints nothing, and that
never raises — a missing file, a directory, a permission error are all simply "not a
record". The trainer imports `pid_record` from it. The launcher calls it:

```bash
pid_record() {
    [ -f "$1" ] || return 1
    "$PYTHON" -m tools.exp11_pidrecord "$1" 2>/dev/null
}
```

The shell no longer reads file **content** anywhere: only the module's stdout crosses
back, and that is digits by construction. `tests/test_exp11_pidrecord.py` (70 tests)
runs the reviewer's 17-row table — extended with leading, internal and trailing NUL and
with `DEAD\0` — through four suites: the module, the CLI, the trainer's imported
function, and a never-disagree suite that sources the launcher and compares its answer
with Python's on every row. One further test asserts the launcher's text contains no
`$(cat --`, `tr -d '\n'` or `$(< ` — the constructs the NULs went through.

## Blocker 2 [P1] — the wrapper is not the trainer

Full mode runs the child under GNU `timeout`, and the frozen lifecycle records the
**wrapper's** pid in `child.pid`. So a complete, dead `child.pid` says only that the
wrapper is gone; it says nothing at all about the trainer. The scan read it as
"registered", ignored the `launching` marker and found no `train.pid` — quiet — and both
modes proceeded over a live, unregistered trainer.

`scan_arm` now decides in the order the review specified:

* live `child.pid` → live;
* live `train.pid` → live;
* a `launching` marker with `train.pid` absent **or** `train.exit` absent → unresolved;
* otherwise quiet.

and `clear_launching` withdraws the marker only with a complete `child.pid` **and** a
`train.exit`. A trainer that crashes before registering therefore leaves the attempt
unresolved — fail closed, by design, and said so in the marker's header comment.

### What that change broke, and the second GREEN

Writing the end-to-end regression exposed a defect the new rule created:
`resolve_unregistered` refused every attempt whose `child.pid` was complete —

```
refusing: <attempt> recorded a child; it is a finished or running launch, not an unresolved one
```

— which is exactly the wrapper-loss state the scan now calls unresolved. The arm was
closed by the scan and unanswerable by the operator: shut for good. Committed as a RED
(`75fd446`, two of the three new tests failing on that refusal) and fixed in `6e3a490`.
The gate is now the question it was always asking — *did this run actually finish?* — and
the answer is a completion receipt or the published `final`:

```bash
    if [ -f "$attempt/completion.json" ]; then ... refuse ...
    if [ "$(published_target || true)" = "$attempt" ]; then ... refuse ...
```

Everything else is decided, as before, by the publication lock, the non-blocking
registration lock, a quiet scan of the whole arm, and the marker's age. The superseded
`test_resolving_refuses_an_attempt_that_is_not_unresolved` was rewritten against the new
contract (no marker / a completion receipt / published as `final`), so both new refusals
are themselves regressed.

### The regressions

Three new tests in `tests/test_exp11_shell.py` run the real thing: the sourced library,
the real FIFO sink, a real `timeout 120` wrapper, the real publication and registration
locks, and a stub trainer (`stub_trainer.py`, written per test) that registers through
the **real** `register_trainer` and records its exit through the real
`record_trainer_exit`, paced by two gate files.

* `test_the_wrapper_loss_schedule_end_to_end` — the launcher declares the marker, forks
  the wrapper, writes `child.pid`, and is lost at that instant (its own TERM handler,
  replayed; exit 143). The test KILLs its own wrapper (KILL, not TERM: `timeout`
  *forwards* a TERM to the process it manages, and the point of the schedule is the
  wrapper dying while the trainer lives). Then: `child.pid` complete and dead,
  no `train.pid` → the scan refuses **unresolved**. That scan is `require_quiet_arm`,
  which is the one call both modes make (`|| exit 2` at the launcher's full-mode line 619
  and its finalize line 743), and the test drives it through the sourced library and,
  separately, through the finalize CLI;
  open the first gate → `train.pid` live → refuses on **live trainer (train.pid)**;
  open the second gate → `train.exit` written, nothing alive, marker aged past the grace
  → `resolve_unregistered` succeeds, writes the tombstone, retires the attempt, and the
  arm promotes again.
* `test_an_ordinary_run_still_withdraws_its_marker` — the same lifecycle with both gates
  open and the launcher waiting for its child, i.e. full mode's `run_child` →
  `clear_launching`: `train.pid` and `train.exit` are there, the marker is gone, the scan
  is quiet, recovery promotes.
* `test_a_trainer_that_never_reported_its_exit_keeps_the_arm_shut` — the same, with the
  stub exiting without a receipt: the marker stands, the scan and the next launch refuse
  as unresolved, and only the operator's resolution reopens the arm.

## Minor — the bounded registration wait

`registration_lock`'s deadline was computed from `time.time()`, which NTP and a resuming
VM step in either direction. It is a duration, so it is now `time.monotonic()`. RED
`01ceecf` jumps a fake monotonic clock past the bound while the wall clock stands still
and shows the wait running its full 20 s anyway; GREEN `b3872a2` ends it at once.

## Report correction

Fix 10's per-commit table is corrected to close review 10's counts (`ea92e3b` tests
**+98**, `64b0484` production **+46/−18**, `56e67b0` **+55**, `d0f9b95` **+78**,
`d7c9e62` production **+99/−15**), and the "under 200 changed lines" statement now says
**production/test** commits, since the record-only `5046ca4` and `51e34b1` exceed it.

## Line counts

| commit | production | tests |
|---|---:|---:|
| `7f253f8` (RED) | — | +107 |
| `68f1f70` (GREEN) | `exp11_pidrecord.py` +49, `exp11_launch.sh` +7/−16, `exp11_train.py` +5/−17 | — |
| `afefa2e` (RED) | — | +70 |
| `1f0ca5e` (GREEN) | `exp11_launch.sh` +18/−5 | +3 |
| `75fd446` (RED) | — | +185 |
| `6e3a490` (GREEN) | `exp11_launch.sh` +16/−4 | +14/−1 |
| `01ceecf` (RED) | — | +36 |
| `b3872a2` (GREEN) | `exp11_train.py` +5/−2 | — |
| `678d9df` (RED) | — | +27 |
| `275be24` (GREEN) | `exp11_profiles.py` +5/−2 | +1/−1 |
| `6d7363b` (tests) | — | +2/−1 |
| range `51e34b1..HEAD` | **+105 / −46 = 151 changed** | +444/−2 |

Every production/test commit is under 200 changed lines; no exception this cycle. The
record-only `ddeeb82` is larger, as bookkeeping commits are.

## Frozen files

`tools/exp06_launch.sh` and `tools/exp06_finalize.py` are byte-identical to `9f98bbb`
(`git diff 9f98bbb..HEAD --` reports nothing for either). No `model/*.py`, no
`tools/exp06_*` file (the `summarize_haa` digest moves because that summariser imports
`exp11_finalize` and `exp11_profiles`, not because its own bytes changed), no
`tools/exp10_*`, no exp_03-pinned file was touched. `bash -n` passes on both shells, `py_compile` on every exp_11 module,
and `git diff --check 51e34b1..HEAD` is silent.

## Digests

Taken at `275be24`, the cycle's last production commit; the two commits after it
(`6d7363b` tests, `03ccee4` record) touch no file in any closure.

**exp_06** — exactly **one** key differs from the record re-filled at `9f98bbb`, and it is
the only one allowed to move (`tools/exp06_summarize_haa.py` imports `exp11_finalize` and
`exp11_profiles`, so every exp_11 change of theirs reaches it):

| key | approved at `9f98bbb` | fix 10 | now |
|---|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `a1c9b623dcfb…` | `2f2f4383c94d…` |

The other 19 exp_06 keys are identical to the approved record.

**exp_11** — all eight keys, and which moved against fix 10's tip `51e34b1`:

| key | digest | vs `51e34b1` |
|---|---|---|
| `train` | `2d22f3d190de1a1a1aaefea9593f8f6d3317945f0591f4aa47e1b060825e517d` | **moved** |
| `finalize` | `1a008a904d3a402cc91498ddff0f9646015e45f7d49e8efcf89e683cabb5b357` | **moved** |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` | unchanged |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` | unchanged |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` | unchanged |
| `launch_sh` | `78145bdf2de434c8cc2845d442a6e7fb74d6c2724e63007384aa42e7574947bb` | **moved** |
| `smoke` | `af2a9079166fb9c37b28cb50ae3de6a4afe7dc6d00086e27667c458aaf84ed32` | **moved** |
| `summarize_haa` | `2f2f4383c94deb76d1a294616539d8a6dddf34bc5e9b9f90db104b8118f2abd2` | **moved** |

The same five as the last three cycles. `launch_sh` moves through its own shell bytes
**and** through its spec, which now lists `tools/exp11_pidrecord.py`; `train`, `finalize`,
`smoke` and `summarize_haa` move because `tools/exp11_train.py` and
`tools/exp11_profiles.py` are in all four source closures — and the new
`tools/exp11_pidrecord.py` has entered them, as the trainer imports it. exp_11's
approvals record remains all-null and every producer refuses today.

## Found while reporting the digests — the launcher's helper was bound by nothing

Checking the closure membership for this section showed that `tools/exp11_pidrecord.py`
entered `train`, `finalize`, `smoke` and `summarize_haa` (the trainer imports it) but
**not** `launch_sh`: a shell has no imports for the closure walker to follow, so the
helpers the launcher runs as subprocesses have to be listed in its spec, which is why
`tools/exp11_lock_holder.py` is there. The module that now decides what every pid file in
the arm means was in no approval key at all — its grammar could change under an approved
`launch_sh` digest. RED `275be24^` asserts that every `tools/exp11_*` module named in the
launcher's text is covered by some key, and that the two helpers with no key of their own
are in `launch_sh`'s; GREEN `275be24` adds the file to the spec. That is what moved every
exp_11 digest a second time, and the table above is the one that stands.

## Notes for the reviewer

* **The shell's reader is now a subprocess.** `pid_record` forks
  `"$PYTHON" -m tools.exp11_pidrecord` per pid file, which is how the NUL problem is
  closed for good: nothing but digits crosses back. Two consequences are worth stating.
  (i) The launcher `cd`s to the repo root and exports `PYTHONPATH="$PWD"` at the top, so
  the module always resolves; in a checkout where it did not, every other Python step of
  the launcher would fail too. (ii) A reader that cannot run reads as "not a record",
  which is the safe direction for `registration_complete` (the marker stays) and the
  unsafe one for `pid_alive` (an attempt would read as not-live) — bounded now by the
  marker rule, which keeps an arm closed until the trainer's own `train.exit` is there.
  Cost is a few interpreter starts per scan (three pid files per attempt), on a path that
  runs a handful of times per launch.
* **`clear_launching` and a `--no-save` diagnostic.** A `--no-save` trainer writes no
  sidecars by design, so a full-mode launch of one would leave its marker standing. Full
  mode never passes `--no-save` (the flag belongs to the smoke path, which runs under
  `diagnostic` and has no marker), so this is not reachable today; it is stated here
  rather than guarded, because a guard would have to guess at a future caller.
* The grace in `resolve_unregistered` is unchanged, and, as the review says, it now only
  bounds how soon an operator may act: the tombstone is what makes a resolution final.

## Test results

Every run CPU-only (`CUDA_VISIBLE_DEVICES=''`), in the `xRIR` env.

| run | result |
|---|---|
| `tests/test_exp11_shell.py`, `test_exp11_train.py`, `test_exp11_pidrecord.py`, `test_exp11_lock_holder.py` | **243 passed** (122 s) |
| the other exp_11 files + `tests/test_exp06_summarize_haa.py` + `tests/test_exp09_sim_eval_closures.py` | **290 passed** (629 s) |
| `tests/test_exp11_profiles.py` after the closure fix | **18 passed** |
| full CPU suite at the tip `6d7363b`, detached | **1 failed, 3864 passed, 55 skipped** (4200 s) |

The three new shell regressions take about 20 s together: each starts a real `timeout`
wrapper and a stub that imports the trainer (and so torch), and the waits on it are
bounded at 90 s so a loaded machine cannot turn them into false failures.

The full suite ran detached at the tip into
`orientation_cue_fairness_2026-09-27_18:09_suite_full_cpu_6d7363b.log` in this record
(70 minutes: the machine was carrying another session's FLAC evaluations and a second
pytest for most of it). The **one** failure is the standing, expected one:

```
FAILED tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes
  {'summarize_haa': '2f2f4383c94d…'} != {'summarize_haa': '645e74c03e20…'}
```

— exp_06's approvals record still holds the digest re-filled at `9f98bbb`, and
`tools/exp06_summarize_haa.py`'s closure has moved with every exp_11 change since. The
re-fill belongs at the reviewed merge, as round 1's `9f98bbb` did, not on this branch.
Every other test in the repository passes.
