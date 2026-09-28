# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 16**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `2ca784a`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU
**Started:** 2026-09-28 01:0x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close15_review.md` via `coder_prompts/round2_fix16_opus_prompt.md`

**Signals:** the only processes signalled are helpers these tests started — `sleep` stubs
and `flock` holders — through their own `Popen` handles. `kill -0` remains a probe.

The family is closed with **one** implementation this time: bash classifies nothing and
decides nothing about a path any more.

## Commits

| SHA | Subject |
|---|---|
| `cfcda7b` | exp11 fix16 (RED): there is no errno-aware path probe |
| `9013a8f` | exp11 fix16 (GREEN): `tools/exp11_pathprobe.py`, where the errno is |
| `a451fb7` | exp11 fix16 (RED): the shell's probes decide without an errno |
| `9b4f25d` | exp11 fix16 (GREEN): the shell asks the path probe, and validates its answer |
| `ca42b42` | exp11 fix16 (RED): a dangling `train.pid` link reads as a dead trainer |
| `48b9cab` | exp11 fix16 (GREEN): one classification, asked by both readers |
| `0f6a3c8` | exp11 fix16 (RED): an alias lets a trainer register past its arm's tombstone |
| `fa13f63` | exp11 fix16 (GREEN): the attempt, the lock and the tombstone are of one arm |
| `0c80c56` | exp11 fix16 (GREEN): the trainer's probe IS the launcher's |
| `e5acfcf` | exp11 fix16 (GREEN): `launch_sh` binds the path probe it runs |
| `e8b1cd2` | exp11 fix16 (RED): the reader timeout is whatever the environment says |
| `067474a` | exp11 fix16 (GREEN): the bound on a liveness question is a bound |
| `f60e5ac` | exp11 fix16 (tests): the timeout regression never touches the real roots |
| `b9315aa` | exp11 fix16 (GREEN): the shell classifies nothing at all |

## Blocker 2 (done first, because it closes the family) — one errno-aware probe

`stat` in bash carries no errno: ENAMETOOLONG, EIO, an ACL that hides a name and a plain
"no such file" all come back the same way, and the shell called that a confirmed absence
whenever the parent looked fine. The classification now lives in
**`tools/exp11_pathprobe.py`** (137 lines), on the pid reader's protocol — one verdict
line and exit 0, or no verdict at all:

```
probe <path>   present <mode-hex> <canonical-path>  |  absent
link  <path>   link <canonical-target>  |  notlink  |  nolink
```

An **absence** is the strongest claim, so it is the most guarded: `FileNotFoundError`
from `os.stat` **and** from `os.lstat` — a name that lstats but does not stat is a
dangling or looping symlink, and the name *is* there — plus a parent that could be
stat'ed, is a directory, and can be searched (`os.access(X_OK)`). Every other errno is
unknown, with the errno name in the reason.

The shell's `probe_path`, `file_present` and `probe_link_target` are now thin wrappers:
they run the module under `timeout --kill-after=5 "$READER_TIMEOUT_S"` and read the
answer through `one_line`, which reads the line back, reprints it and requires the
result to equal the file **byte for byte** — so padding, a second line, NULs, another
protocol's words and a crafted verdict are unknown, not an answer. The health check asks
the path probe two questions whose answers are known (a file that is there, a name that
is not) beside the pid reader's two.

The trainer imports the same module: `probe_path`, `file_state` and `directory_state`
are literally `exp11_pathprobe`'s (50 lines of duplicate classification deleted).

## Blocker 1 — the pid reader asks the same question

`inspect_record` no longer decides whether the path is there; it calls the probe. A
dangling or looping `train.pid` is therefore unknown, where it used to be "no record" —
and since a resolution excludes its own target's marker by design, those records are all
that speak for the trainer. RED `ca42b42` is the reviewer's schedule (a live registered
trainer whose record is replaced by a link to a file that is then removed): the
resolution used to retire it; it now refuses with exit 2 and writes nothing.

`pid_record`'s own `stat` shortcut in the shell is gone too (`b9315aa`): the module
classifies before it opens anything, so a FIFO is still never opened, and there is one
classification instead of two that must agree.

## Blocker 3 — the attempt, the lock and the tombstone are of one arm

`register_trainer` compared the **alias's** parent, so with
`arm_alias/attempt_X -> arm_canonical/attempt_X` it took the alias parent's registration
lock and checked the alias parent's tombstone while every write landed in the canonical
attempt — whose tombstone said the launch had been retired. It now canonicalises the
attempt it was given, derives the arm as that canonical parent, requires it to equal the
canonical configured root, and uses the canonical attempt for the lock, the tombstone
and the write. An attempt reached through **its own arm's** alias still registers (a
symlinked arm root is the production case on the NAS); an alias **into another arm**
refuses with exit 3 and writes nothing, and no lock of the wrong arm is created.

## The audit, updated

Every existence/type test in `tools/exp11_launch.sh` and in the trainer's registration
path, at `b9315aa`. **Routed** = the answer comes from `tools/exp11_pathprobe.py` (or, for
pid files, from `tools/exp11_pidrecord.py`, which asks the same probe) and an unknown
refuses. **Justified** = a failure there cannot yield absent / dead / quiet / unpublished.

### `tools/exp11_launch.sh`

| line | site | disposition |
|---|---|---|
| — | `probe_path`, `file_present`, `probe_link_target` | **routed**: thin wrappers over the module, with the timeout and the byte-exact `one_line` validation |
| — | `pid_record` | **routed**: it classifies nothing itself any more (`b9315aa`); the module answers, and the verdict is validated byte for byte |
| 425 | `pid_record`: `stat -c %s` of the verdict file | justified: a temp file this process just created in its own directory; a failure is unknown |
| 533–536 | `list_attempts`: `realpath -e`, `-d`, `-r`, `-x`, `find` | justified: every failure → 1 → the caller's unknown ("cannot be enumerated") |
| 549, 576 | `scan_arm`: `realpath` for identity, with a literal fallback | justified: comparison only. A failure makes the spellings unequal, so the attempt is **not** excluded from the scan — the strict direction |
| 566 | `scan_arm`: per-entry `stat -L -c %f` | justified: a `stat` that **fails** is unknown (the entry may be a dangling link or behind a door); only a `stat` that succeeds classifies, from the locale-independent mode bits |
| 698 | `resolve_unregistered`: `stat -L` of `<attempt>/.` | justified: a failure is unknown ("cannot be inspected") |
| — | `scan_arm`'s marker / `train.pid` / `train.exit` | **routed** (`file_present`); any unknown closes the arm |
| — | `clear_launching`: `train.exit`, and the arm's enumeration | **routed**; absent **or** unknown keeps the marker |
| — | `resolve_unregistered`: the target's type, its marker, its receipt, the published `final` | **routed**; each unknown refuses with exit 2 |
| 675, 678 | `resolve_unregistered`: `realpath -e` of the target and the arm root | justified: checked; a failure refuses as unknown |
| — | `published_target`, `already_published`, `check_final_conflict`, `abort_recovery` | **routed** through `probe_link_target` / `file_present`; unknown never reads as "not published" |
| 815 | `scan_barrier`: `while [ -e "$file" ]` | justified: a test-roots-only affordance (`EXP11_SCAN_BARRIER` **and** `EXP11_TEST_ROOTS=1`); a failed test releases a barrier, which grants nothing |
| 983 | `[ ! -d "$DATA_ROOT" ]` | justified: a failure **refuses the invocation** |
| — | `promote` (`ln -s` + `mv -Tf`) | justified: it tests nothing; the rename is atomic and unconditional |
| — | `preflight`, `check_attempt` | justified: Python, and both refuse on any `OSError` |
| — | `hold_arm_lock`, `drop_arm_lock`, the registration lock | justified: every failure refuses access rather than granting it |

### `tools/exp11_train.py` (the registration path)

| site | disposition |
|---|---|
| `probe_path` / `file_state` / `directory_state` | **are** `tools/exp11_pathprobe.py`'s functions, imported |
| the `--no-save` tombstone check | **routed**; unknown refuses (exit 3) |
| the run directory (before and under the lock) | **routed** |
| the arm root | **routed** |
| the attempt↔arm containment | `os.path.realpath` of a directory already probed present, compared with the canonical configured root (`fa13f63`) |
| the tombstone, under the lock | **routed**; unknown refuses |
| the `launching` marker | **routed**; unknown refuses |
| `child.pid` | **routed** through `exp11_pidrecord.inspect_record`, which asks the probe |
| `run()`: the two "did the run directory vanish" checks | **routed** |
| `registration_lock`, `record_trainer_exit` | justified: a failure to open the lock refuses; a missing exit receipt only ever keeps an arm closed |

No site remains where a failure to inspect yields absent, dead, quiet or unpublished.

## The nonblocking items

* `timeout --kill-after=5` on every reader and probe invocation.
* `EXP11_READER_TIMEOUT_S` is validated: a whole number of seconds between 1 and 300, or
  the invocation refuses. Set-but-empty, `0`, a negative, `400`, `abc` and `3s` are all
  refused (regression); unset is 30.
* The header now states that the bound is **per question**, and that a scan asks several
  in sequence — three probes for one attempt's marker, `train.pid` and `train.exit`, so
  its own worst case is a small multiple (≈ 60 s at the default with two of them
  hanging).

## Line counts

| commit | production | tests |
|---|---:|---:|
| `cfcda7b` (RED) | — | +133 |
| `9013a8f` (GREEN) | `exp11_pathprobe.py` +137 | — |
| `a451fb7` (RED) | — | +91 |
| `9b4f25d` (GREEN) | `exp11_launch.sh` +69/−37 | — |
| `ca42b42` (RED) | — | +25 |
| `48b9cab` (GREEN) | `exp11_pidrecord.py` +15/−13 | — |
| `0f6a3c8` (RED) | — | +36 |
| `fa13f63` (GREEN) | `exp11_train.py` +25/−10 | — |
| `0c80c56` (GREEN) | `exp11_train.py` +11/−50 | — |
| `e5acfcf` (GREEN) | `exp11_profiles.py` +2/−1 | +4/−2 |
| `e8b1cd2` (RED) | — | +11 |
| `067474a` (GREEN) | `exp11_launch.sh` +21/−5 | — |
| `f60e5ac` (tests) | — | +2/−1 |
| `b9315aa` (GREEN) | `exp11_launch.sh` +5/−14 | — |
| range `2ca784a..HEAD` | **+285 / −130 = 415 changed** | +301/−2 |

Every commit is under 200 changed lines; no exception this cycle.

## Frozen files

`tools/exp06_launch.sh` and `tools/exp06_finalize.py` are byte-identical to `9f98bbb`.
No `model/*.py`, no `tools/exp06_*` file, no `tools/exp10_*`, no exp_03-pinned file was
touched. `bash -n` passes on both shells, `py_compile` on every exp_11 module, and
`git diff --check 2ca784a..HEAD` is silent.

## Digests

Taken at `b9315aa`, the cycle's last production commit.

**exp_06** — one key differs from the record re-filled at `9f98bbb`, and it is the only
one allowed to move:

| key | approved at `9f98bbb` | fix 15 | now |
|---|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `22a9858b067e…` | `720862918e13…` |

The other 19 exp_06 keys are identical to the approved record.

**exp_11** — all eight keys, against fix 15's tip `2ca784a`:

| key | digest | vs `2ca784a` |
|---|---|---|
| `train` | `21e08ffe412c1adaf80888fde36f0faca7227563da72d7e062d37c0f4229aae1` | **moved** |
| `finalize` | `ee39e005182368d999f9d7824ab805ae5cd22e51f1a6d31e5d96fa514813a6e7` | **moved** |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` | unchanged |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` | unchanged |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` | unchanged |
| `launch_sh` | `a9fd96978609ab038ea68aed51971de69ccd09d2a2ad01af9ef1f1de14b68955` | **moved** |
| `smoke` | `3fd783ccc24daec89e88445fe2cdf2814d3da7c0e8663f583e32e28a943529b6` | **moved** |
| `summarize_haa` | `720862918e131449e111602b0e3684b4c29be1a36dae9a44542de47472e83b45` | **moved** |

The familiar five. `tools/exp11_pathprobe.py` is in **all five** closures — `launch_sh`
because its spec now lists it (`e5acfcf`, caught by the coverage test written in fix 11:
"no approval key covers: ['tools/exp11_pathprobe.py']"), and the four Python keys through
the trainer's and the pid reader's imports; verified with `closure_of` for each. The
three HAA keys are untouched. exp_11's approvals record remains all-null and every
producer refuses today.
