# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 17**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `f966461`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU
**Started:** 2026-09-28 02:3x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close16_review.md` via `coder_prompts/round2_fix17_opus_prompt.md`

**Signals:** the only processes signalled are helpers these tests started — `sleep` stubs
and `flock` holders — through their own `Popen` handles. `kill -0` remains a probe.

## Commits

| SHA | Subject |
|---|---|
| `009836d` | exp11 fix17 (RED): the probe answers about paths it never looked up |
| `4f1d267` | exp11 fix17 (GREEN): look the path up, and validate every field printed |
| `29500a8` | exp11 fix17 (RED): a verdict line passes with any fields at all |
| `8003435` | exp11 fix17 (GREEN): the fields of a verdict, checked before it counts |
| `48e3d84` | exp11 fix17 (RED): the trainer trusts whatever canonicalisation returns |
| `e918f4b` | exp11 fix17 (GREEN): the trainer checks the paths its containment rests on |

## The blocker [P1] — a verdict line is not a verdict until its fields are

The byte-exact line check said "this is one line of the protocol"; it said nothing about
what was *in* it. `link garbage` was therefore a published target, and a resolution
retired the attempt `final` really pointed at while `abort_recovery` renamed a published
one; `present 41ed relative` was a marker's verdict and left the arm quiet; `ffffffff`
was a file mode.

Every field is now checked, in all three places — before the module prints it, after the
shell reads it, and in the trainer that consumes it:

* **mode** — three to six hex digits whose S_IFMT bits name a type that exists (fifo,
  character device, directory, block device, regular, symlink, socket). `f000` and
  `ffffffff` are not modes.
* **path** (canonical path and link target) — absolute, no `//`, no `/./`, no `..`
  component, no trailing `/` (unless it is `/`), no CR/LF/NUL, at most PATH_MAX = 4096
  bytes.
* **`present` is about the name that was probed** — its basename must come back. The
  module makes that true by construction (`canonical()` resolves the *directory* of an
  ordinary name and keeps the name), and the shell checks it. The exception is stated
  where the check is: a probed name that is *itself* a symlink resolves to a target with
  another name, so the rule is skipped only where the shell has no name of its own to
  compare (`.`/`..` endings) — a symlinked sidecar therefore reads as unknown, which is
  the closed direction and a layout exp_11 never creates.

Any violation is unknown: nothing is scanned, retired, replaced, aborted or registered
on it.

## P2-a — the probe must not normalize what the kernel would not

`abspath` collapses `..` lexically, so `base/missing/../leaf` and
`base/dangling/../leaf` were answered from `base` — *absent*, although the lookup the
kernel performs fails. The probe now makes a relative path absolute by joining
`os.getcwd()` and normalizes nothing; the parent comes from `dirname` of the
**unnormalized** string, so the directory that is checked is the one actually traversed.
A trailing slash is its own rule — it demands that the last component be a directory, and
ENOTDIR and ENOENT are different news — so only a successful stat of a directory settles
it and everything else is unknown. A `..` path that really does resolve is still
answered, present or absent: the rule is "look it up", not "refuse every `..`".

## P2-b — the timeout bound, bounded as a string first

`EXP11_READER_TIMEOUT_S` is matched as a decimal string (`[1-9]`, `[1-9][0-9]`,
`[1-9][0-9][0-9]`) before any arithmetic, exactly as `check_ceiling` does for the
36-hour ceiling: bash errors on a 36-digit comparison, and an errored test is a false
one, which read as "in range". Three digits at most, then `-gt 300`.

## The regressions

`tests/test_exp11_shell.py` (207 collected), with a `$PYTHON` wrapper that answers one
probe with a crafted verdict and delegates everything else:

| test | what it establishes |
|---|---|
| `test_a_crafted_marker_verdict_leaves_the_arm_closed` (10 verdicts) | `link garbage`, `link relative/target`, `present 41ed relative`, `present zzzz /abs`, `present f000 /abs`, `present ffffffff /abs`, `/abs/../x`, `//abs`, a 5000-character path and a trailing slash — each on `launching` → the scan refuses with exit 2 |
| `test_a_crafted_receipt_verdict_retires_nothing` (3) | the same on `completion.json` → the resolution refuses; the attempt, its marker and the absence of a tombstone are preserved |
| `test_a_crafted_final_verdict_never_retires_the_published_attempt` | the reviewer's schedule: `final` really publishes the attempt being resolved, `link garbage` is injected → exit 2, the attempt is there and `final` still resolves to it |
| `test_a_crafted_final_verdict_publishes_nothing` (3 verdicts × `already_published`, `check_final_conflict` with and without `REPLACE_FINAL`, `abort_recovery`) | each returns 2, nothing is aborted or replaced, the attempt and the `final` symlink are intact |
| `test_an_overflowing_reader_timeout_is_refused` | 36 digits → the invocation refuses |

`tests/test_exp11_pathprobe.py` (18): the four P2-a cases (`missing/../leaf`,
`dangling/../leaf`, a `..` path that really resolves, a trailing slash on a dangling
link — for `probe` and for `link`), the field validators in both directions, and that a
canonical path keeps the probed name.

`tests/test_exp11_train.py` (60): a canonicalisation that returns `relative/x`,
`/abs/../x`, `//abs` or a 5000-character path refuses with exit 3 and writes nothing.

## Report corrections asked for by close review 16

* **The trainer's canonicalisation** happens **before** the first directory probe, not
  after it: `register_trainer` canonicalises the given path (≈ 309), validates that
  canonical path, derives the arm from it, and only then probes the run directory
  (≈ 317) and the arm root. The fix-16 audit table listed those rows in the other order.
* **The scan's identity fallbacks** (`realpath … || printf '%s'`): the fix-15/16 tables
  said a failure "makes the spellings unequal". More precisely: it falls back to the
  literal strings, which *can* still be equal — if the caller's spelling and the glob's
  are identical — so the exclusion may still apply. What makes this safe is the order:
  the three pid probes run **before** the exclusion, so a live or unreadable record of
  the attempt being resolved is seen whatever the comparison decides.
* **The marker's timestamp** (≈ 816) was missing from the audit:
  `date -u -r "$attempt/launching"` falls back to *now* when it fails, which makes the
  marker look zero seconds old and the grace *not* elapsed — a refusal, the conservative
  direction. It is the one place where a failed inspection substitutes a value, and it
  substitutes the one that refuses.
* **The claim narrowed.** "No inspection failure can yield absent/dead/quiet/
  unpublished" is what the audit *intends*; what the tests *establish* is that for the
  guards they exercise — the enumeration, the per-entry classification, the marker,
  `train.pid`, `train.exit`, `completion.json`, `final`, the pid records, the tombstone,
  the run directory and the arm root — an inspection failure, a reader failure, a
  crafted verdict and an invalid field each refuse, under the errno matrix
  (ENAMETOOLONG/EACCES/EIO) close review 16 replayed. The rest of the audit is argued,
  not tested.
* **The padding statement narrowed.** The byte-exact line check rejects a second line,
  NULs and trailing junk; it does **not** by itself reject a space or a CR *inside* a
  path field, because such a line is still one line. That is what the field validation
  added this cycle is for.
* **The timeout bounds.** The bound is **per question**: `timeout --kill-after=5
  $READER_TIMEOUT_S` (30 s by default) wraps each reader or probe invocation. A single
  guard is one or two questions; one attempt in a scan is up to seven (three pid records
  — each a path probe inside the module plus the reader — and the marker, `train.pid`,
  `train.exit` probes), so the worst cases are multiples of the bound, not "≈ 60 s": on
  the order of **70 s** for one attempt's sidecar probes, **105 s** with its pid records,
  and **140 s** for a resolution that also probes the receipt and `final` — per attempt
  scanned, with a hanging reader at every step.

## Line counts

| commit | production | tests |
|---|---:|---:|
| `009836d` (RED) | — | +66 |
| `4f1d267` (GREEN) | `exp11_pathprobe.py` +83/−5 | — |
| `29500a8` (RED) | — | +96 |
| `8003435` (GREEN) | `exp11_launch.sh` +59/−14 | — |
| `48e3d84` (RED) | — | +17 |
| `e918f4b` (GREEN) | `exp11_train.py` +11/−2 | — |
| range `f966461..HEAD` | **+153 / −21 = 174 changed** | +179 |

Every commit is under 200 changed lines; no exception this cycle.

## Frozen files

`tools/exp06_launch.sh` and `tools/exp06_finalize.py` are byte-identical to `9f98bbb`.
No `model/*.py`, no `tools/exp06_*` file, no `tools/exp10_*`, no exp_03-pinned file was
touched. `bash -n` passes on both shells, `py_compile` on every exp_11 module, and
`git diff --check f966461..HEAD` is silent.

## Digests

Taken at `e918f4b`, the cycle's last production commit.

**exp_06** — one key differs from the record re-filled at `9f98bbb`, the only one allowed
to move:

| key | approved at `9f98bbb` | fix 16 | now |
|---|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `720862918e13…` | `0f5fdeab9c57…` |

The other 19 exp_06 keys are identical to the approved record.

**exp_11** — all eight keys, against fix 16's tip `f966461`:

| key | digest | vs `f966461` |
|---|---|---|
| `train` | `be309c629c72a541e636075c4dbeec9c80687e22bbcf075597870b39f73d747a` | **moved** |
| `finalize` | `d30f729992ff3690ced21fd268f1af6cb147d095b245f8a2f07bcd18bc3ff8cf` | **moved** |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` | unchanged |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` | unchanged |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` | unchanged |
| `launch_sh` | `2149526de3524f5602a1f3e5775f676e8b10fda309010be744fa8089085cb8a8` | **moved** |
| `smoke` | `e81f8389019770cf88fc4193bda223a95e1e5c22d497fedb72aa773d3f5ed0da` | **moved** |
| `summarize_haa` | `0f5fdeab9c57de5702ab3e773209bbaaf95536b131eb57140dece834f21c05df` | **moved** |

The familiar five, for the familiar reasons: `launch_sh` through its own shell bytes,
and the four Python keys because `tools/exp11_pathprobe.py` and `tools/exp11_train.py`
are in their closures. The three HAA keys are untouched. exp_11's approvals record
remains all-null and every producer refuses today.
