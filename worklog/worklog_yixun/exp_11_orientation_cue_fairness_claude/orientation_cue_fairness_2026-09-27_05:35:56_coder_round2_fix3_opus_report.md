# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 **FIX CYCLE 3**

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`, from the reviewed tip `612e3dd`
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU, no process signals
**Started:** 2026-09-27 05:2x EDT
**Input:** `orientation_cue_fairness_codex_code_round2_close2_review.md` (request changes, one remaining blocker) and `coder_prompts/round2_fix3_opus_prompt.md`

## Commits

| SHA | Subject |
|---|---|
| `0edb7cf` | exp11 fix3 (RED): recovery can invalidate a published attempt and races itself |
| `c8b2ff0` | exp11 fix3 (GREEN): recovery is owned, serialised, idempotent and explicit |
| `155d695` | exp11 fix3: say plainly that the alternate exp_06 fixture is not git-committed |

Every commit in this cycle is under 200 changed lines.

## The blocker [P2] — recovery safety in `tools/exp11_launch.sh`

Three reproductions, three distinct causes, one fix each.

### (a) A refused recovery must not invalidate a finished run

Passing the *canonical* directory into `abort` — correct for the alias defect fix 2
solved — meant a malformed recovery (a missing `--log`, say) renamed the real attempt to
`attempt_A_ABORTED_finalize_refused`, leaving `final` dangling and carrying an existing
completion away with it.

**`abort_recovery`** now decides ownership before it acts. An attempt that is **already
published** (`final` resolves to it) or **already completed** (`completion.json` present)
is `PRESERVED`: directory, completion and link untouched, the finalizer's own cause left
standing on stderr, exit non-zero. A malformed invocation does not establish that the run
it names failed. Full mode keeps aborting unconditionally — it launched the attempt it is
aborting — and an orphan attempt (neither published nor completed) is still aborted, so
the path is narrowed rather than removed.

### (b) No ownership: two recoveries could publish a dangling link

**`take_lock`** serialises alias resolution, validation, finalization *and* promotion per
arm, and **full-mode promotion takes the same lock**. `mkdir` is the primitive: it
succeeds for exactly one caller. A held lock **refuses immediately** — it never waits —
naming the owner and whether its pid is still live. A `trap` releases it on every exit
path. A **stale lock is evidence of an interrupted publication, not litter**: it is
cleared only by an explicit `--break-lock`, which logs `BREAKLOCK` with the previous
owner, and never on a timer or a heuristic.

A dry run touches nothing outside the test roots: the lock is announced everywhere
(`LOCK …`) and really taken when the run is not a dry run, or when `EXP11_TEST_ROOTS=1`
puts the roots in a temporary directory — which is exactly where the concurrency tests
exercise it.

### (c) A different existing `final` was replaced silently

**`already_published`** makes a same-target recovery a no-op: if `final` already resolves
to the canonical attempt *and* that attempt has completed, the launcher reports
`ALREADY PUBLISHED` and exits 0 without re-finalizing or re-linking.

**`check_final_conflict`** refuses (exit 2, nothing changed) when `final` publishes a
*different* attempt, unless `--replace-final` is given; with it the replaced target is
logged (`REPLACING … -> <old> with <new>`) **before** the atomic `mv -Tf`. Silent
replacement is not a recovery policy. The conflict rule is recovery's: a full run
publishes its own fresh attempt and legitimately supersedes the previous `final`.

Alias resolution and the `attempt_*` / arm-root checks from fix 2 are unchanged.

### Tests

Dry-run, through the CLI: a published-but-incomplete attempt is `PRESERVED` on refusal
and never `_ABORTED_`; a completed attempt likewise; an orphan attempt still aborts; a
same-target recovery is idempotent with no `PROMOTE` and no finalizer call; a different
`final` refuses naming `--replace-final`, and with the flag succeeds and logs the replaced
target; a held lock refuses both recovery and full-mode promotion; a stale lock is not
auto-cleared and survives the refusal, and `--break-lock` clears it with a `BREAKLOCK`
line; the lock is released after both a refusing and a succeeding invocation.

Filesystem-level, with the launcher sourced as a library (no mode, no preflight), so the
effects are real rather than announced: `abort_recovery` on all three owned shapes
(published, completed, both) leaves the directory, the log and the `final` link in place
and creates no `*_ABORTED_*` sibling, while an unowned attempt really is renamed; and
`take_lock` is a real mutual exclusion — the first caller's trap releases it, the second
is refused, and a refused caller never removes a lock it does not own.

**Red-first.** All eight CLI tests failed on the behaviour under test before the fix: an
`ABORT` where `PRESERVED` was required, a `PROMOTE` where "already published" was
required, and exit 0 where a conflict or a held lock had to refuse.

## Report corrections carried over from the reviews

1. **The alternate exp_06 fixture is not git-committed.** `test_an_alternate_exp06_record_is_refused`
   (renamed from `…_alternate_committed_…`) writes the alternate record to `tmp_path`. That
   test isolates the **digest comparison** this gate owns. The separate requirement — that
   a consumed record be a *tracked path whose blob at the reviewed commit is
   byte-identical* — is enforced where the record is loaded and tested there
   (`exp06_profiles.load_approved_digests(repo, commit)`;
   `tests/test_exp11_profiles.py::test_committed_blob_binding` for exp_11's own record).
   The docstrings now say this plainly instead of claiming "committed".
2. **Two exp_11 keys moved against `2c64a2a`, not three.** Fix cycle 2's report said
   "Three keys moved … `launch_sh` …, `summarize_haa` … and nothing else", which
   contradicted its own list. The correct statement is **two**: `launch_sh` (blockers 2
   and 3) and `summarize_haa` (blocker 1).
3. **`0b10d7c` was fix cycle 2's last production commit.** The commit after it
   (`1c4498e`) adds only a record-scoped `.gitattributes`; it changes no production code
   and is not part of any approvals closure.

## Digests

Taken at `3284068`; the last commit that touches production code is `c8b2ff0` (the two
commits after it change a test file's docstrings and this record, neither of which is in
any approvals closure), so these values stand for the whole cycle.

**exp_06** — exactly **one** key moved against the record re-filled at `9f98bbb`; the
other nineteen are unchanged:

| key | approved at 9f98bbb | now |
|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `deff6d49d8e9…` |

**exp_11** — all eight keys:

| key | digest |
|---|---|
| `train` | `d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571` |
| `finalize` | `3eb15b59b672d97ac257054e66f8bc835abd27b85e24f9a417a5a1aaa9e25b9e` |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` |
| `haa_pipeline_sh` | `3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4` |
| `launch_sh` | `32a23ee058e2e21da2a922a0042673e092d053b6f8fc9b73eb5fb15595f08e4b` |
| `smoke` | `62133ba07a35d42026c19fd1484b37c6c06ace0ebedc1cdd8484690df323450e` |
| `summarize_haa` | `deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a` |

**Exactly one key moved against the fix-2 tip `612e3dd`: `launch_sh`**
(`737414cc…` → `32a23ee0…`), which is the whole of this cycle's production change.
`summarize_haa` is unchanged from fix 2 (`deff6d49…`), as are the other six. These are
values for the **eventual reviewed merge**, not an authorisation to fill the approvals
now: exp_11's record stays all-null and every exp_11 producer refuses today.

## Tests

* **Targeted set** — all `tests/test_exp11_*.py` plus `tests/test_exp06_summarize_haa.py`
  and `tests/test_exp09_sim_eval_closures.py`: **353 passed** in 477 s. The pinned
  exp_06/exp_09 and exp_11 phase-1 oracles still reproduce their payloads and rendered
  summaries byte for byte.
* `bash -n` on both shells, `py_compile` on every exp_11 module and test, and
  `git diff --check` over `612e3dd..HEAD`: clean.

## Full CPU suite

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest tests -q -rf` on the frozen
tree (clean outside `worklog/`):

```
1 failed, 3683 passed, 55 skipped, 307 warnings in 3154.00s (0:52:34)
FAILED tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes
```

The single failure is the **expected exp_06 approvals re-fill** — `code.summarize_haa`
has moved and the record is stale until a reviewer re-fills it at the reviewed merge,
exactly as round 1 did with `9f98bbb`. It is the mechanism that detects the one key this
round is allowed to move, not a regression.

Progression across this round's four runs: 3 614 → 3 651 → 3 669 → **3 683** passing,
with 2 → 1 → 1 → 1 failures (all of them that same approvals-drift detector after the
first run's load-average SIGTERM flake, which has not recurred).

## Status

The one remaining blocker and the three report corrections are addressed. Ready for
re-review at this tip.
