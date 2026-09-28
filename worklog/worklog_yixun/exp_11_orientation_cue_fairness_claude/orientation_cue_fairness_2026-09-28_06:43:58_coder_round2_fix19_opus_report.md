# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 19**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `f2f6943`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU
**Started:** 2026-09-28 06:3x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close18_review.md` via `coder_prompts/round2_fix19_opus_prompt.md`

**Signals:** the only processes signalled are helpers these tests started — `sleep` stubs
and `flock` holders — through their own `Popen` handles. `kill -0` remains a probe.

**Disk:** every suite in this cycle ran with `--basetemp=/tmp/pytest-exp11-<ts>` and that
directory was removed afterwards; the root filesystem stayed at ≈ 35 GB free throughout.

Both remaining items were **conservative rejections** — the machinery refusing something
it should accept — so each fix had to be shown not to loosen the gate it sits in.

## Commits

| SHA | Subject |
|---|---|
| `653bc1d` | exp11 fix19 (RED): a string comparison still stands in front of the identity gate |
| `6597474` | exp11 fix19 (GREEN): containment is decided once, by identity |
| `44656e1` | exp11 fix19 (RED): the default entry refuses `attempt_X/sub/..` |
| `f251525` | exp11 fix19 (GREEN): the arm a path names, and the arm its attempt is in |

## P2-1 — the attempt check compared two canonical strings

The embedded Python ran

```python
directory, base = Path(attempt).resolve(), Path(root).resolve()
if directory.parent != base: ...
```

**before** the `same_parent` gate the shell asks about the very same directory. Two
spellings can be one directory — a read-only bind mount gives identical `st_dev`/`st_ino`
with unequal canonical strings — so that comparison could only ever reject what the gate
accepts. It is gone: the embedded Python now resolves the directory it must validate,
finalize and promote, checks the `attempt_<UTC>` name and the recorded profile, and
leaves containment to the identity gate. `grep` finds no `.resolve()` and no equality
between two path names anywhere in the launcher.

Because the case itself cannot be built without mounting something, the regressions are:
a white-box assertion that no canonical string is derived to be compared; the gate
exercised on both sides (an arm reached through a symlinked root is accepted and promotes
the real attempt; an attempt of another directory is refused); and a bind-mount check
that **skips unless** the system already has a read-only mount with two spellings of one
inode — it creates and mounts nothing, it reads `/proc/self/mountinfo`. On this machine
it skips.

## P2-2 — the default trainer entry refused `attempt_X/sub/..`

The arm came from `given.parent`, the string, so the last-component `..` spelling named
an arm that was the attempt itself. Fix 18's regression passed `arm_root` explicitly and
so never saw it. The rule is now the reviewer's:

```python
given_abs      = os.path.join(os.getcwd(), given)   # absolute, NOT normalised
canonical      = os.path.realpath(given_abs)
arm            = os.path.dirname(canonical)
lexical_parent = os.path.dirname(os.path.normpath(given_abs))
accept iff os.path.samestat(os.stat(lexical_parent), os.stat(arm))
```

Collapsing `..` is safe **there and only there**: it feeds an identity comparison that
can never accept a wrong arm, and no existence decision is taken from it. An explicit
`arm_root`, when supplied, must additionally be identity-equal to the arm.

Through the **default** call: `arm/attempt_X/sub/..`, `arm//attempt_X`, `arm/./attempt_X`
and `arm/attempt_X/` all register; the NAS layout `arm_alias/attempt_X` (the arm is a
symlink, the attempt inside it real) registers; an attempt symlinked into another arm
(`arm_other/attempt_X -> arm/attempt_X`) refuses with exit 3, no lock of the wrong arm
and no sidecar; and a spelling whose lexical parent is not there refuses.

## Line counts

| commit | production | tests |
|---|---:|---:|
| `653bc1d` (RED) | — | +84 |
| `6597474` (GREEN) | `exp11_launch.sh` +5/−5 | — |
| `44656e1` (RED) | — | +43/−2 |
| `f251525` (GREEN) | `exp11_train.py` +33/−16 | — |
| range `f2f6943..HEAD` | **+38 / −21 = 59 changed** | +127/−2 |

Every commit is under 200 changed lines; no exception this cycle.

## Frozen files

`tools/exp06_launch.sh` and `tools/exp06_finalize.py` are byte-identical to `9f98bbb`.
No `model/*.py`, no `tools/exp06_*` file, no `tools/exp10_*`, no exp_03-pinned file was
touched. `bash -n` passes on both shells, `py_compile` on every exp_11 module, and
`git diff --check f2f6943..HEAD` is silent.

## The timeout statement, corrected

The earlier report gave "six base 30 s budgets per fully scanned attempt, ≈ 180 s".
That is not a bound and it is withdrawn. What is true is the **per-question** bound:
every reader and probe invocation runs under `timeout --kill-after=5
"$READER_TIMEOUT_S"` (30 s by default, validated as 1–300), so each individual question
ends within that budget plus the five-second kill grace. How many questions an
invocation asks is not fixed — the health check adds four, each attempt contributes its
pid records and sidecar probes, a resolution adds the receipt, the link and the identity
questions, and a scan stops at the first live or unresolved attempt — so no single
figure bounds a scan, and none is claimed.

## Digests

Taken at `6597474`, the cycle's last production commit.

**exp_06** — one key differs from the record re-filled at `9f98bbb`, the only one allowed
to move:

| key | approved at `9f98bbb` | fix 18 | now |
|---|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `85caf08ce7cb…` | `3956692a6ae3…` |

**exp_11** — all eight keys, against fix 18's tip `f2f6943`:

| key | digest | vs `f2f6943` |
|---|---|---|
| `train` | `56004f44a2f88625882904ac622c6179c126cfef89333714eca5ff7ccc6fc0a8` | **moved** |
| `finalize` | `6a747806cdd5a84809c4f2b2529245e4fb0ca9f777ef4b174287dae653bca574` | **moved** |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` | unchanged |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` | unchanged |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` | unchanged |
| `launch_sh` | `a932e5bfd087394ee467700e803aa78973037be62cfa0ef9a7fd2d92efd5caac` | **moved** |
| `smoke` | `22f989b7541c90dfca53420bb775a6c82b6927999d0c430943dff475ced03092` | **moved** |
| `summarize_haa` | `3956692a6ae37d0c7bb59dcc11941f7e186b110e85937d144bf7cdd128d1de0a` | **moved** |

The familiar five: `launch_sh` through its own bytes, the four Python keys through
`tools/exp11_train.py`. The three HAA keys are untouched; exp_11's approvals record
remains all-null.

## Test results

Every run CPU-only, with a throwaway `--basetemp` removed afterwards.

| run | result |
|---|---|
| every targeted file in one pass — `test_exp11_shell.py` (213), `test_exp11_pathprobe.py`, `test_exp11_pidrecord.py`, `test_exp11_train.py` (65), `test_exp11_lock_holder.py`, the other ten exp_11 files, `test_exp06_summarize_haa.py`, `test_exp09_sim_eval_closures.py` | **673 passed, 1 skipped** (845 s) |
| full CPU suite, detached, at **`659747480963eb103f43650a4d15dfd24c6d440e`** | **1 failed, 4003 passed, 56 skipped** (3595 s) |

The one skip is the bind-mount check, which runs only where the system already has a
read-only mount with two spellings of one inode.

The log is `orientation_cue_fairness_2026-09-28_06:44_suite_full_cpu_6597474.log`, whose
first line is `HEAD 659747480963eb103f43650a4d15dfd24c6d440e` — written by
`git rev-parse HEAD` at launch. It ran with `--basetemp=/tmp/pytest-exp11-1790592247`,
which held 5.1 GB and was removed as soon as the run finished; the root filesystem is
back at 35 GB free. The 56th skip is the new bind-mount check.

The **one** failure is the standing, expected one:

```
FAILED tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes
  {'summarize_haa': '3956692a6ae3…'} != {'summarize_haa': '645e74c03e20…'}
```

— exp_06's approvals record still holds the digest re-filled at `9f98bbb`, and that
summariser imports `exp11_finalize` and `exp11_profiles`, so its closure moves with every
exp_11 change. The re-fill belongs at the reviewed merge, as round 1's `9f98bbb` did.
Neither the exp_04 threading flake of fix 17 nor any other failure appeared.

## Notes for the reviewer

* **Both fixes loosen a gate, so both are shown from both sides.** The attempt check now
  accepts an arm reached through a symlinked root *and still refuses* an attempt of
  another directory; the trainer accepts four odd spellings and the NAS alias layout *and
  still refuses* an attempt symlinked into another arm, with no lock of the wrong arm and
  no sidecar written.
* **Where lexical collapsing is now allowed, and why.** Exactly one place: the trainer's
  `lexical_parent`, which feeds an identity comparison. It can never *accept* a wrong arm
  — the comparison is `samestat` against the canonical attempt's real parent — and no
  existence decision is taken from it. Everywhere else, `..` is still the kernel's
  business.
* **The bind-mount regression is honest about its limits.** It reads
  `/proc/self/mountinfo`, mounts and creates nothing, and skips when the system has no
  read-only mount with two spellings of one inode — which is the case here. The property
  it would check is instead asserted white-box: no canonical string is derived to be
  compared.
* **Unchanged:** `scan_arm` stops at the first live or unresolved attempt; the shell
  trusts the pinned, health-checked probe module and defends only against a malformed
  answer.
