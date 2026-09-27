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
