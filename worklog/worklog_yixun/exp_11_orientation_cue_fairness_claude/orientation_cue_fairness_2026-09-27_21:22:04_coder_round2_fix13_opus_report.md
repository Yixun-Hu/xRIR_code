# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 13**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `0885c41`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU
**Started:** 2026-09-27 21:1x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close12_review.md` via `coder_prompts/round2_fix13_opus_prompt.md`

**Signals:** the only processes signalled are helpers these tests started — `sleep` stubs
and `flock` holders — through their own `Popen` handles. `kill -0` remains a probe.

## Commits

| SHA | Subject |
|---|---|
| `ec6dcfd` | exp11 fix13 (RED): a file we cannot read is read as a file that is not there |
| `1e989d2` | exp11 fix13 (GREEN): only a confirmed absence, or bytes we read, are an answer |
| `c0ff8e8` | exp11 fix13 (RED): Bash tidies the reader's bytes into a verdict |
| `24c6c9f` | exp11 fix13 (GREEN): the verdict is the bytes, and a path we cannot inspect is unknown |
| `73d8c60` | exp11 fix13 (RED): the file classification reads a translated phrase |
| `5593d69` | exp11 fix13 (GREEN): the mode bits classify the path, in every language |

## Blocker 1 [P1] — being unable to look is not an answer

`pid_record` caught every `OSError` and returned `None`, so the CLI printed `norecord`
(exit 0) for a `chmod 000` `train.pid` — a file that exists and may well name a living
trainer — and the resolution retired the attempt of a registered, running one.

`inspect_record(path)` now classifies in three, and classifies **before it opens**:

* **definite `norecord`** — the name is confirmed absent *and* the directory it would be
  in could be inspected; the path exists but is not a regular file (a directory, FIFO,
  socket or device — none of them is a pid record); or the bytes were read and are not
  the grammar;
* **unknown** (exit 2, one line of reason on stderr, **nothing on stdout**) — a stat that
  failed, a parent that cannot be inspected, an open or a read that failed: permission
  denied, EIO, ELOOP, ENOTDIR;
* **`record <pid>`** — unchanged.

`pid_record(path)` is unchanged for the trainer (pid or `None`); the caller that must
tell "no record" from "could not look" is the launcher, through the CLI.

The stat-before-open also fixes a hang the reviewer found: the CLI blocked forever on a
FIFO, because opening one waits for a writer. A FIFO is now classified from its mode and
never opened.

In the launcher the `-f` guard is gone. `[ -f ]` on a path whose parent is unreadable is
false, which read as "absent"; the guard is now a `stat` whose **failure decides
nothing** — the module then tells a confirmed absence from a path it was not allowed to
inspect. And `resolve_unregistered` refuses as unknown when the attempt directory itself
cannot be inspected, instead of reporting "no launching marker" about a directory it
could not read.

## Blocker 2 [P1] — the bytes the reader printed, not what Bash made of them

Command substitution strips trailing newlines and NUL bytes, so `norecord\n\n` and
`norecord\0\n` — neither of them the protocol, both exit 0, both healthy on the two
probes — arrived as clean verdicts and retired a live trainer's attempt.

The verdict is now checked as raw bytes, in a file, before any shell expansion sees it:

```bash
    "$PYTHON" -m tools.exp11_pidrecord "$1" > "$out" 2>/dev/null || status=$?
    if [ "$status" -ne 0 ]; then rm -f -- "$out"; return "$READER_UNKNOWN"; fi
    if cmp -s "$out" <(printf 'norecord\n'); then rm -f -- "$out"; return 1; fi
    pid="$(tr -cd '0-9' < "$out")"
    case "$pid" in ''|*[!0-9]*) rm -f -- "$out"; return "$READER_UNKNOWN" ;; esac
    [ "${#pid}" -le 10 ] || { rm -f -- "$out"; return "$READER_UNKNOWN"; }
    if cmp -s "$out" <(printf 'record %s\n' "$pid"); then ... return 0; fi
    return "$READER_UNKNOWN"
```

Only the digits of a *candidate* pid cross back through an expansion — digits survive
intact — and the whole line must then match byte for byte, which is what refuses
padding, NULs, CR and trailing junk. The two health probes are compared the same way,
with `cmp` against files holding exactly `record 1\n` and `norecord\n`.

## A third defect, found in my own diff: the classification was in English

`stat -c %F` prints a **translated** phrase ("regular file", `fichier régulier`, …), so
under any other locale an ordinary `train.pid` would have matched none of the regular
cases and been classified "not a pid record" — a live trainer's record read as no
record. RED `73d8c60`, GREEN `5593d69`: the classification reads `%f`, the mode in hex,
and tests the S_IFMT bits, which are the same in every language.

## Nonblocking — the tombstone and RESOLVED wording

Both claimed "no finished trainer (no train.exit)", which the resolution does not check
(and should not: a stale marker beside a finished trainer is resolvable, as fix 11's
end-to-end regression shows). They now state exactly the checked conditions — held under
this arm's publication and registration locks, a launching marker older than the grace,
no recorded pid of the arm alive, no completion receipt, not the published `final` — and
name the tombstone as the defence against a trainer that never registered. The fixture
asserts the new substance and that the word `train.exit` does **not** appear.

## The regressions

`tests/test_exp11_pidrecord.py` (75 tests):

| test | what it establishes |
|---|---|
| `test_an_absent_file_and_a_directory_are_definite` | the two things that really are answers about a path we could inspect |
| `test_a_fifo_is_definite_and_is_never_opened` | classified from its mode, with a 30 s subprocess timeout proving it does not block |
| `test_a_file_we_cannot_read_is_not_an_absent_file` | the reviewer's `chmod 000`: exit 2, **no** stdout, a reason on stderr, no traceback |
| `test_a_parent_we_cannot_inspect_is_not_an_absent_file` | an unreadable directory says nothing about what is or is not inside it — for an existing name and an absent one alike |

Both permission tests skip when the euid is 0, since root ignores the bits they turn off.

`tests/test_exp11_shell.py` (144 tests):

| test | what it establishes |
|---|---|
| `test_only_the_exact_verdict_bytes_are_a_verdict` (6 payloads) | `norecord\n\n`, `norecord\0\n`, `record 123\n\n`, `record 123\0\n`, `record 123 \n`, `\nrecord 123\n` emitted **only** for `train.pid` (so both probes pass) → unknown → the resolution refuses with exit 2, no tombstone, no rename, the marker and `train.pid` intact, and the scan refuses too |
| `test_the_health_probes_are_byte_exact_too` (6 payloads) | the same bytes for every read → `pid reader unhealthy` |
| `test_a_reader_that_pads_every_answer_is_unhealthy` | correct answers with one extra newline — the case that passes a check made after command substitution and fails a byte-exact one |
| `test_a_pid_file_the_launcher_cannot_read_is_never_dead` (`file`, `parent`) | `chmod 000` on a registered live trainer's record, and on its directory: `pid_record` 2, `pid_alive` 2, the resolution and both scans refuse, nothing written |
| `test_the_file_classification_does_not_depend_on_a_translated_word` | the type comes from the mode bits, and a FIFO is still definite and immediate |

The byte table is unchanged: every row still produces a definite verdict through the
module, the CLI, the trainer and the shell, and never the unknown status 2.

## Line counts

| commit | production | tests |
|---|---:|---:|
| `ec6dcfd` (RED) | — | +52/−2 |
| `1e989d2` (GREEN) | `exp11_pidrecord.py` +60/−14 | — |
| `c0ff8e8` (RED) | — | +126 |
| `24c6c9f` (GREEN) | `exp11_launch.sh` +73/−34 | +7/−3 |
| `73d8c60` (RED) | — | +17 |
| `5593d69` (GREEN) | `exp11_launch.sh` +8/−6 | — |
| range `0885c41..HEAD` | **+135 / −48 = 183 changed** | +202/−5 |

Every commit is under 200 changed lines; no exception this cycle.

## Frozen files

`tools/exp06_launch.sh` and `tools/exp06_finalize.py` are byte-identical to `9f98bbb`
(`git diff 9f98bbb..HEAD` reports nothing for either). No `model/*.py`, no `tools/exp06_*`
file, no `tools/exp10_*`, no exp_03-pinned file was touched. `bash -n` passes on both
shells, `py_compile` on every exp_11 module, and `git diff --check 0885c41..HEAD` is
silent.

## Digests

Taken at `5593d69`, the cycle's last production commit.

**exp_06** — exactly **one** key differs from the record re-filled at `9f98bbb`, and it is
the only one allowed to move (`tools/exp06_summarize_haa.py` imports `exp11_finalize` and
`exp11_profiles`):

| key | approved at `9f98bbb` | fix 12 | now |
|---|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `650a5a6d97cb…` | `40626dd26bde…` |

The other 19 exp_06 keys are identical to the approved record.

**exp_11** — all eight keys, and which moved against fix 12's tip `0885c41`:

| key | digest | vs `0885c41` |
|---|---|---|
| `train` | `0f518d3f79cb6b992d0ff6057888f93c0bae0d9520eeb8c0522d1ab40d11b9c2` | **moved** |
| `finalize` | `256349e7dfa55880ee37eb6c97192ade560af1aff322941962e0b861c860be73` | **moved** |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` | unchanged |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` | unchanged |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` | unchanged |
| `launch_sh` | `cfb86bc5a4f27d46c810d8e5346171ce7a085dc51f7646b3c429c407e3935d9b` | **moved** |
| `smoke` | `9940aadc525cbb01ceb5925b38056657630145f2998690c8d077399a1608ccbf` | **moved** |
| `summarize_haa` | `40626dd26bdeac80f1cbfc01d243e43d5ab0017d59bb9129c11b86771ef6495d` | **moved** |

The same five as the last five cycles: `launch_sh` through its own shell bytes and the
reader its spec binds; `train`, `finalize`, `smoke` and `summarize_haa` because
`tools/exp11_pidrecord.py` is in all four closures through the trainer's import. The
three HAA keys are untouched. exp_11's approvals record remains all-null and every
producer refuses today.
