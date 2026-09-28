# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 18**
### Planner design change — plan §11 amendment A2

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `2e109d0`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU
**Started:** 2026-09-28 04:5x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close17_review.md` and plan §11 A2, via `coder_prompts/round2_fix18_opus_prompt.md`

**Signals:** the only processes signalled are helpers these tests started — `sleep` stubs
and `flock` holders — through their own `Popen` handles, and this cycle's own superseded
suite run, identified by the log file it held open. `kill -0` remains a probe.

Close review 17 found four more gaps, all in the **grammar of a path field**: space-bearing
and overlong fields, a terminal `..` canonicalised wrongly, a `Path()` normalisation before
validation, a trailing-slash attempt refused by the basename rule. The Planner's answer is
to remove the grammar: **the probe returns no path strings at all.**

## Commits

| SHA | Subject |
|---|---|
| `2e76a95` | exp11 fix18 (RED): the verdicts still carry paths, and identity is still a string |
| `8cf5810` | exp11 fix18 (GREEN): the probe answers identity, and hands out no paths |
| `622529f` | exp11 fix18 (RED): the trainer still parses canonical strings |
| `30cd986` | exp11 fix18 (GREEN): the trainer resolves in process and compares identity |
| `87f4e4d` | exp11 fix18 (GREEN): the launcher asks identity questions, and carries no paths |
| `d420ddb` | exp11 fix18 (GREEN): the resolution's containment is an identity question too |

## The protocol, after A2

```
probe <path>          present <mode-hex>   |  absent
link  <path>          link | notlink | nolink
same <a> <b>          same | different
sameparent <p> <d>    same | different
```

— one line, exit 0, and nothing else ever crosses; anything else is unknown. `same` and
`sameparent` are decided by `os.path.samestat` on the **followed** stats: device and
inode, which settle "is this the attempt `final` publishes?" and "is this attempt in this
arm?" exactly, where a string comparison needed a grammar that three reviews in a row
found holes in. A dangling link, an absent name or a parent nobody may enter make the
question *unanswerable*, never "different".

The header of both files now states what the shell defends against: **a malformed
answer, not a lying module.** `tools/exp11_pathprobe.py` is bound with the launcher by
the `launch_sh` approvals key and health-checked before every scan and every resolution;
what the shell validates is the keyword set and, for `present`, the mode.

## Every decision that used to compare paths

| site | now |
|---|---|
| `published_target` | **gone**. `publishes <attempt>` = `probe_link` says a link is there **and** `same_path "$ARM_ROOT/final" "<attempt>"` |
| `already_published` | `publishes` + the completion receipt |
| `check_final_conflict` | `probe_link`, then `same_path`: `different` is the conflict, unknown refuses |
| `abort_recovery` | `publishes` + the receipt; unknown refuses to abort |
| `resolve_unregistered`'s published guard | `publishes` |
| `resolve_unregistered`'s arm containment | `same_parent "$attempt" "$ARM_ROOT"` (it canonicalised both sides and compared strings) |
| the scan's exclusion of the resolution target | `same_path` per entry; unknown → the scan refuses |
| `check_attempt`'s containment | `same_parent`, plus the `attempt_<UTC>` pattern on the slash-stripped basename |
| the trainer's arm containment | in-process `os.path.realpath` + `exp11_pathprobe.same` |

The only canonical spelling still produced in the shell is the one `resolve_unregistered`
needs to **rename** the directory and name the tombstone beside it, and the `realpath -e`
`list_attempts` uses as `find`'s starting point. Neither is compared with another path.
The `REPLACING` log line prints `readlink` for the operator, with a comment saying no
decision reads it.

## The trainer

`os.path.realpath` in process — the filesystem's own meaning of `..` and of every symlink
on the way — then `Path` built from that result and nothing else, and containment by
`samestat`. So `arm/attempt_X/sub/..` **is** `arm/attempt_X` and finds the tombstone on it
(close review 17's case), while `//`, `/./`, a trailing slash and `sub/..` spellings of a
legitimate attempt register normally. `valid_path` is deleted from the module and from
the shell; there is no path grammar left to get wrong.

## The regressions

`tests/test_exp11_pathprobe.py` (23): no verdict carries a path (asserted on every
question), the mode is still validated before printing, `valid_path` and `canonical` are
**gone**; the same file by two names — through a symlink, a `//`, a `/./`, a trailing
slash and a `..` — is `same`; two directories are `different`; a dangling link, an absent
name and a shut door are unknown; `sameparent` takes the dirname as written; misuse is
never a verdict.

`tests/test_exp11_shell.py` (209): the crafted-verdict suites are now **malformed
answers** — `present` with no mode, `present 81a4 /forged/path` (a field the protocol no
longer has), `present zzzz`, `present f000`, `present ffffffff`, a trailing space,
`absent yes`, `link /forged/target`, `same different`, `yes` — each on `launching`,
`completion.json` and `final`, against the scan, the resolution, `already_published`,
`check_final_conflict` (± `REPLACE_FINAL`) and `abort_recovery`: all refuse with exit 2,
the attempt and the publication preserved. Two new spelling regressions show `//` and a
trailing slash resolving normally and a published `final` recognised through three
spellings. A forged path cannot be expressed at all any more, which is why the fix-17
injections are replaced rather than kept.

`tests/test_exp11_train.py` (62): the alias layout still refuses (exit 3, no lock of the
wrong arm); `arm/attempt_X/sub/..` with a tombstone on `attempt_X` refuses; `//`, `/./`,
a trailing slash and `sub/..` spellings of a legitimate attempt register normally; and
the trainer holds no path grammar (`valid_path` absent, `canonical(` unused).

The health check now asks the identity question too: a file against itself must be
`same`, and against another file `different`.

## Report corrections asked for by close review 17

* **The suite HEAD.** Fix 17's two suite logs ran at **`fef89ba`** — a record-only commit
  above `e918f4b`, with identical production bytes — not at `e918f4b` as the report said.
  This cycle's log records `git rev-parse HEAD` in its **first line**, so the two can
  never drift again.
* **The timeout accounting.** There is no universal bound; each *question* is bounded by
  `timeout --kill-after=5 $READER_TIMEOUT_S` (30 s by default). A fully scanned attempt
  makes **six** outer timeout calls — three pid records and three sidecar probes
  (`launching`, `train.pid`, `train.exit`) — so the sidecar group alone is ≈ 105 s at the
  default with every call hanging, and the whole attempt ≈ 180 s; a resolution adds the
  receipt, the link and the identity questions. The earlier "≈ 60 s" figure understated
  it.
* **The exp_04 flake.** `test_completion_transaction_and_failed_promotion[True-promote]`
  reads `capsys` while a completion transaction runs on a background thread: it is
  **timing-sensitive**, and no exp_04 file has changed since `9f98bbb`. Attributing it to
  co-tenancy was a guess; the honest statement is that it is a timing-sensitive test
  unrelated to exp_11, which passes on its own and in its own file.
* **Bookkeeping commit sizes.** The record-only commits `a700a24` (624 lines) and
  `cad6018` (538 lines) exceed 200 changed lines. They carry a close review, a prompt and
  a report; the ≤ 200 rule is recorded here as applying to production and test commits,
  with these two named.

## Line counts

| commit | production | tests |
|---|---:|---:|
| `2e76a95` (RED) | — | +89/−21 |
| `8cf5810` (GREEN) | `exp11_pathprobe.py` +63/−61 | — |
| `622529f` (RED) | — | +36/−11 |
| `30cd986` (GREEN) | `exp11_train.py` +9/−11 | — |
| `87f4e4d` (GREEN) | `exp11_launch.sh` +124/−73 | +44/−11 |
| `d420ddb` (GREEN) | `exp11_launch.sh` +15/−6 | — |
| range `2e109d0..HEAD` | **+211 / −151 = 362 changed** | +169/−43 |

Every production/test commit is under 200 changed lines; no exception this cycle. (The
two oversized record-only commits of earlier cycles are named under the report
corrections above.)

## Frozen files

`tools/exp06_launch.sh` and `tools/exp06_finalize.py` are byte-identical to `9f98bbb`.
No `model/*.py`, no `tools/exp06_*` file, no `tools/exp10_*`, no exp_03-pinned file was
touched. `bash -n` passes on both shells, `py_compile` on every exp_11 module, and
`git diff --check 2e109d0..HEAD` is silent.

## The audit, updated — no path-field sites remain

| site | disposition |
|---|---|
| `probe_path`, `file_present`, `probe_link`, `same_path`, `same_parent` | **the probe**: a keyword from a fixed set, and a validated mode. No path crosses |
| `publishes`, `already_published`, `check_final_conflict`, `abort_recovery` | **identity questions**; unknown refuses, and no target string is read |
| `scan_arm`: the marker, `train.pid`, `train.exit` | **routed** (`file_present`); unknown closes the arm |
| `scan_arm`: the exclusion of the resolution target | **identity question**; unknown refuses |
| `scan_arm`: per-entry `stat -L -c %f` (≈ 636) | justified: a `stat` that **fails** is unknown; only a successful one classifies, from locale-independent mode bits |
| `resolve_unregistered`: target type, marker, receipt, published `final`, containment | **routed / identity**; each unknown refuses |
| `resolve_unregistered`: `realpath -e` (≈ 757) | justified: the one canonical spelling needed to **rename** the directory and name the tombstone; never compared with another path |
| `resolve_unregistered`: `stat -L` of `<attempt>/.` (≈ 780) | justified: a failure is unknown |
| `resolve_unregistered`: the marker's timestamp (≈ 850) | justified: `date -u -r … || now` — a failure makes the marker look zero seconds old, so the grace has *not* elapsed and the resolution refuses. The one substituted value, and it is the conservative one |
| `check_attempt` | **identity question** + the `attempt_<UTC>` name pattern; the inline Python still validates the profile and resolves the directory that is then asked about, finalized and promoted |
| `list_attempts`: `realpath -e`, `-d`, `-r`, `-x`, `find` (≈ 606) | justified: every failure → the caller's unknown ("cannot be enumerated"); the canonical path is `find`'s starting point, compared with nothing |
| `pid_record`: `stat -c %s` of the verdict file (≈ 491) | justified: a temp file this process just created; a failure is unknown |
| `scan_barrier` `[ -e ]` (≈ 898) | justified: test-roots-only; a failed test releases a barrier, which grants nothing |
| `check_final_conflict`'s `readlink` (≈ 966) | justified: informational log text; no decision reads it |
| `[ ! -d "$DATA_ROOT" ]` (≈ 1079) | justified: a failure refuses the invocation |
| trainer: run directory, arm root, tombstone, marker, `child.pid` | **routed** through the probe; unknown refuses with exit 3 |
| trainer: canonicalisation and containment | **in process**: `os.path.realpath` then `os.path.samestat`; no string is parsed or compared |
| `promote`, `preflight`, the locks | justified: `promote` tests nothing (atomic rename); the others refuse on any error |

## Digests

Taken at `d420ddb`, the cycle's last production commit.

**exp_06** — one key differs from the record re-filled at `9f98bbb`, the only one allowed
to move:

| key | approved at `9f98bbb` | fix 17 | now |
|---|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `0f5fdeab9c57…` | `85caf08ce7cb…` |

**exp_11** — all eight keys, against fix 17's tip `2e109d0`:

| key | digest | vs `2e109d0` |
|---|---|---|
| `train` | `d712fa89fc7b35f7785f970cb1562d432c5120cfdab55848e32318ddc5c0dc30` | **moved** |
| `finalize` | `197f5bdeb6753e48cad1075299f2cd4bd86fd4ba1a7cfe7639ee95a39142c213` | **moved** |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` | unchanged |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` | unchanged |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` | unchanged |
| `launch_sh` | `d4dab1e4a51fcac65b5ea6f307c66f43e601939f9670361219000d24b1202d19` | **moved** |
| `smoke` | `c8710213e7eadf16193e8f664e7dd7c60ecee13fd34ad6ffd01f307eac2a83ff` | **moved** |
| `summarize_haa` | `85caf08ce7cbf2d479349fbf67825c8cb0feff9be8822d41ef76f08d2f4716d7` | **moved** |

The familiar five: `launch_sh` through its own bytes and the probe its spec binds, and
the four Python keys through `tools/exp11_pathprobe.py` and `tools/exp11_train.py`. The
three HAA keys are untouched. exp_11's approvals record remains all-null.

## Test results

| run | result |
|---|---|
| `test_exp11_shell.py` (209), `test_exp11_pathprobe.py` (23), `test_exp11_pidrecord.py`, `test_exp11_train.py` (62), `test_exp11_lock_holder.py` | **376 passed** (191 s) |
| the other ten exp_11 files + `test_exp06_summarize_haa.py` + `test_exp09_sim_eval_closures.py` | **291 passed** (559 s) |
| full CPU suite, detached, at **`d420ddb4443cf3212899b01534608b3f39491165`** | **1 failed, 3997 passed, 55 skipped** (3455 s) |

The log is `orientation_cue_fairness_2026-09-28_05:29_suite_full_cpu_d420ddb.log`, whose
**first line** is `HEAD d420ddb4443cf3212899b01534608b3f39491165` — written by
`git rev-parse HEAD` at launch, so the run and the commit can never drift apart in the
record again. An earlier run of this cycle, started at `87f4e4d`, was superseded by the
resolver commit `d420ddb`; it was ended through its own pid (identified by the log file
it held open) and its partial log removed.

The **one** failure is the standing, expected one:

```
FAILED tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes
  {'summarize_haa': '85caf08ce7cb…'} != {'summarize_haa': '645e74c03e20…'}
```

— exp_06's approvals record still holds the digest re-filled at `9f98bbb`, and that
summariser imports `exp11_finalize` and `exp11_profiles`, so its closure moves with every
exp_11 change. The re-fill belongs at the reviewed merge, as round 1's `9f98bbb` did. The
exp_04 threading flake of fix 17 did not recur.

## Notes for the reviewer

* **What A2 buys.** The questions the guards ask are now the questions the kernel can
  answer exactly: *is this path there, and what kind of thing is it* (a mode), *is there
  a link here*, and *are these two names one file* (device and inode). None of them has a
  textual answer, so none of them has a grammar, and the three classes of defect the last
  three reviews found — relative paths, spaces and byte counts, terminal `..` and
  trailing slashes — cannot be expressed in the protocol.
* **What the shell still trusts.** The module itself: a lying `exp11_pathprobe` could say
  `same` where the truth is `different`. That is not defended against, and is stated in
  both headers: the module is pinned with the launcher by the `launch_sh` approvals key,
  and health-checked (four questions now, including the identity one) before every scan
  and every resolution.
* **Unchanged from close review 12:** `scan_arm` stops at the first live or unresolved
  attempt, so an unknown in a later entry may not be reached; it is still a refusal.
* **One canonical spelling remains in the shell**, in `resolve_unregistered`, because the
  directory has to be *renamed* and a tombstone named beside it. It is produced by a
  checked `realpath -e` and compared with nothing.
