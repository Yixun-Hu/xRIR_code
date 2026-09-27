**Reviewer:** OpenAI Codex, GPT-6 per exposed developer metadata. The exact invocation model identifier and reasoning-effort setting are not exposed; I cannot attest to a more specific value.  
**Sandbox:** Read-only worktree; approved temporary writes only under `/tmp`; CUDA disabled; one numerical thread per process, within the four-thread limit; no process signals.  
**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`.  
**HEAD:** `48b7bd4cd211575844a84568ef717e68fcd849e6`.  
**Review window:** 2026-09-27, 10:41–10:46 UTC / 06:41–06:46 EDT.  
**Scope:** All six commits in `612e3dd..48b7bd4`, their combined diff, the fix3 report, independent reproductions, tests, and closure calculations.

**Verdict: request changes. Round 2 cannot be CLOSED, and `48b7bd4` should not be merged.** The original recovery reproductions now pass, but two defects in the new lock handling still permit publication without exclusive ownership.

1. **[P2] INT/TERM handlers release the lock without terminating the protected work.**  
   [tools/exp11_launch.sh:231](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:231)

   `trap release_lock EXIT INT TERM` installs cleanup-only signal handlers. Bash can resume execution after the handler returns. For example, TERM received by the launcher while its synchronous finalizer runs can release the lock after that child returns, then allow promotion to continue without the lock.

   Respecting the prohibition on signals, I replayed the *registered handler body directly*. This demonstrates the handler’s behavior; actual signal delivery was not tested:

   ```text
   REGISTERED: trap -- 'release_lock' SIGTERM
   CONTINUED_AFTER_HANDLER LOCK_HELD=0 LOCK_EXISTS=no
   PROMOTE …/final -> attempt_A
   FINAL_TARGET=attempt_A
   exit: 0
   ```

   **Required correction:** Make INT/TERM handlers terminate execution, with ownership-safe cleanup on EXIT. Add regression coverage demonstrating that handler execution cannot return to finalization or promotion.

2. **[P2] Concurrent stale-lock breakers can delete fresh locks, and an earlier owner’s exit can delete its successor’s lock.**  
   [break_lock:206](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:206), [release_lock:198](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:198)

   `break_lock` removes the current pathname without coordinating with another breaker or verifying that it still represents the stale owner. `release_lock` subsequently trusts only its process-local `LOCK_HELD` flag.

   A controlled reproduction used the real functions, with a barrier in `say` after reading the stale owner:

   ```text
   B reads/logs stale owner; pauses
   A reads/logs the same stale owner; removes it; acquires lock
   B resumes; removes A’s fresh lock; acquires lock
   A and B are both alive inside the protected operation
   A exits normally; its EXIT trap removes B’s lock
   B remains alive; lock no longer exists
   C performs ordinary take_lock: exit 0; THIRD_ENTERED
   ```

   Both breakers initially observed a stale lock, so this does not require an operator knowingly overriding a live owner. Separately, the current flag also permits that live override directly.

   **Required correction:** Serialize stale-lock replacement, protect a newly acquired lock from an in-flight breaker, and bind cleanup to the caller’s own lock generation. Add the concurrent stale-breaker and previous-owner-exit regressions.

The requested ordinary recovery behaviors otherwise verify:

| Check | Independent result |
|---|---|
| Published attempt, nonexistent `--log` path, recovery through `final` | Exit **2**; real finalizer reports `EXP11_FINALIZE_REFUSED unreadable log`; canonical directory and link unchanged; `final` remains valid; no aborted sibling. |
| Completed, unpublished attempt with nonexistent log | Exit **2**; completion bytes and filesystem identity unchanged; no aborted sibling. |
| Published and completed attempt | Exit **0**; idempotent skip; no finalizer call or promotion; directory, completion and link unchanged. |
| Orphan attempt with nonexistent log | Exit **2**; renamed to `attempt_A_ABORTED_finalize_refused`. |
| Ordinary concurrent recoveries | First exits **0**, second **2** with owner and `live=yes`; refused caller leaves owner bytes and attempt intact; first publishes a valid `final`. |
| Different published target | Exit **2** without `--replace-final`; both attempts and existing link preserved. |
| Explicit replacement | Exit **0**; `REPLACING … -> attempt_A with attempt_B` precedes promotion; actual atomic replacement leaves a valid link. |
| Full-mode child/finalizer failure | The newly launched attempt and log are aborted; previous `final` survives; lock released. |
| Full-mode success | Fresh attempt supersedes the previous `final`; lock released. |

The missing-log reproductions used the actual finalizer. Successful expensive finalization and training were stubbed where needed; the launcher functions and literal mode branches performed the filesystem operations. Preflight was stubbed for temporary roots. No training or GPU work ran.

**The full/recovery asymmetry is appropriate:** a full invocation explicitly launches and owns a fresh attempt, promoting it only after successful finalization. Recovery replays an existing attempt and should require explicit authorization to replace another publication.

The atomic `mkdir` lock is per arm and precedes recovery alias resolution, attempt validation, finalization and promotion. Full mode participates before launching its child and retains the lock through promotion. Ordinary held-lock refusal is immediate and preserves the owner’s lock. Stale locks survive ordinary refusal; explicit breaking logs `BREAKLOCK`. An ordinary dry-run announces `LOCK` without creating its root; test-root dry-runs take the real lock. Normal success and refusal cleanup passed. The two blockers above prevent approving the broader ownership and exit-path guarantees.

The tests provide useful coverage, with these limits:

- The recovery section contains **eight changed-behavior CLI tests plus an orphan-abort control**. Dry-run preservation tests verify announcements, not a real finalizer refusal or its exit status.
- [The sourced filesystem tests](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:489) exercise real preservation for published-only, completed-only and both-owned shapes, including no `*_ABORTED_*` sibling. My reproductions additionally checked completion contents and filesystem identity.
- [The mutual-exclusion test](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:522) exercises real acquisition, EXIT cleanup and refusal without deleting another owner’s lock. Its callers are sequential, with a manually populated held lock; it does not exercise concurrent live callers or breakers.

RED-for-the-intended-reason evidence was independently replayed using the unchanged historical `0edb7cf` launcher and historical test functions under `/tmp`, with current dependencies:

| Historical test | Observed RED |
|---|---|
| Same-target idempotency | Old launcher returned 0 but announced `ABORT` and `PROMOTE`, without “already published”; intended assertion failed. |
| Different-final conflict | Old launcher returned 0 and announced promotion instead of refusing with 2; intended assertion failed. |

Neither replay failed because of imports, paths or fixture setup.

**Selected verification: 84 passed, no skips.**

| Selection | Passed |
|---|---:|
| Shell | 54 |
| Approval enforcement | 13 |
| Sim-eval closures | 3 |
| Named committed-blob binding test | 1 |
| Registry and pinned oracles | 13 |

All pytest runs used worktree `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, disabled CUDA, thread limits and `/tmp` fixtures/caches. Both shells pass `bash -n`; `git diff --check 612e3dd..48b7bd4` passes. The full CPU suite was not rerun; its reported single stale-exp_06-approval failure is consistent with the independently verified digest change.

The report corrections are valid:

- The alternate fixture is explicitly **not Git-committed** and isolates consumed-record digest comparison. The separate tracked, reviewed-blob requirement is enforced by [exp06_profiles.py:115](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_profiles.py:115) and [its loader:173](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_profiles.py:173). Confirmatory summarization supplies the repository and commit. The named exp_11 committed-blob test passes.
- Exactly **two** exp_11 keys differ from `2c64a2a`: `launch_sh` and `summarize_haa`.
- `0b10d7c` was fix2’s last production-code commit; `1c4498e` changes only record-local `.gitattributes`.

One additional, nonblocking report correction remains: [report:17](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-27_05:35:56_coder_round2_fix3_opus_report.md:17) says every commit is below 200 changed lines. `c8b2ff0` has **524 insertions and 14 deletions**, including copied review material; launcher plus tests alone total **205** changed lines. Acknowledge the exception rather than claim compliance.

Closure hygiene **passes at `48b7bd4`**:

- Exactly **one of twenty exp_06 code keys** differs from the approvals at `9f98bbb`: `summarize_haa`. The other nineteen match.
- All four sim-eval closures—`eval`, `eval_launch`, `compare`, `mirror_probe`—are unchanged.
- Frozen files are untouched. The only modified existing production source versus `9f98bbb` is `tools/exp06_summarize_haa.py`; exp_11 production files are additions.
- `BACKBONES_EXP06` is unchanged; its digest remains `f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169`.
- Pinned `c6233e3` exp_06/exp_09 and `c491396` exp_11 Phase-1 comparisons reproduce serialized statistical payloads and rendered summaries byte-for-byte.
- This fix cycle changes only production `tools/exp11_launch.sh`; only exp_11 `code.launch_sh` moves versus `612e3dd`.

The exact fill below applies to the **unchanged `48b7bd4` candidate contents**. It is not merge authorization: the required launcher corrections will require recomputing the affected closure at the eventual reviewed merge.

Replace only exp_06 `code.summarize_haa`:

```text
old: 645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619
new: deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a
```

Replacing that single digest while preserving every other byte yields a **3,190-byte** exp_06 approval record with SHA-256:

```text
80134832646672411740f779835fa7c06f8bb354b46aab558884503b39a1b666
```

The complete exp_11 candidate fill, including all eight code keys, reused inputs and null artifacts, is:

```json
{
  "schema_version": 1,
  "code": {
    "train": "d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571",
    "finalize": "3eb15b59b672d97ac257054e66f8bc835abd27b85e24f9a417a5a1aaa9e25b9e",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "32a23ee058e2e21da2a922a0042673e092d053b6f8fc9b73eb5fb15595f08e4b",
    "smoke": "62133ba07a35d42026c19fd1484b37c6c06ace0ebedc1cdd8484690df323450e",
    "summarize_haa": "deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "80134832646672411740f779835fa7c06f8bb354b46aab558884503b39a1b666",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "artifacts": {
    "simpor_epoch_012": {
      "epoch": null,
      "path": null,
      "sha256": null
    },
    "simpor_yaw_epoch_012": {
      "epoch": null,
      "path": null,
      "sha256": null
    }
  }
}
```

HEAD and repository status remained unchanged, including the pre-existing untracked close3 review prompt. No review file or repository changes were saved.
tokens used
75,711
**Reviewer:** OpenAI Codex, GPT-6 per exposed developer metadata. The exact invocation model identifier and reasoning-effort setting are not exposed; I cannot attest to a more specific value.  
**Sandbox:** Read-only worktree; approved temporary writes only under `/tmp`; CUDA disabled; one numerical thread per process, within the four-thread limit; no process signals.  
**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`.  
**HEAD:** `48b7bd4cd211575844a84568ef717e68fcd849e6`.  
**Review window:** 2026-09-27, 10:41–10:46 UTC / 06:41–06:46 EDT.  
**Scope:** All six commits in `612e3dd..48b7bd4`, their combined diff, the fix3 report, independent reproductions, tests, and closure calculations.

**Verdict: request changes. Round 2 cannot be CLOSED, and `48b7bd4` should not be merged.** The original recovery reproductions now pass, but two defects in the new lock handling still permit publication without exclusive ownership.

1. **[P2] INT/TERM handlers release the lock without terminating the protected work.**  
   [tools/exp11_launch.sh:231](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:231)

   `trap release_lock EXIT INT TERM` installs cleanup-only signal handlers. Bash can resume execution after the handler returns. For example, TERM received by the launcher while its synchronous finalizer runs can release the lock after that child returns, then allow promotion to continue without the lock.

   Respecting the prohibition on signals, I replayed the *registered handler body directly*. This demonstrates the handler’s behavior; actual signal delivery was not tested:

   ```text
   REGISTERED: trap -- 'release_lock' SIGTERM
   CONTINUED_AFTER_HANDLER LOCK_HELD=0 LOCK_EXISTS=no
   PROMOTE …/final -> attempt_A
   FINAL_TARGET=attempt_A
   exit: 0
   ```

   **Required correction:** Make INT/TERM handlers terminate execution, with ownership-safe cleanup on EXIT. Add regression coverage demonstrating that handler execution cannot return to finalization or promotion.

2. **[P2] Concurrent stale-lock breakers can delete fresh locks, and an earlier owner’s exit can delete its successor’s lock.**  
   [break_lock:206](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:206), [release_lock:198](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:198)

   `break_lock` removes the current pathname without coordinating with another breaker or verifying that it still represents the stale owner. `release_lock` subsequently trusts only its process-local `LOCK_HELD` flag.

   A controlled reproduction used the real functions, with a barrier in `say` after reading the stale owner:

   ```text
   B reads/logs stale owner; pauses
   A reads/logs the same stale owner; removes it; acquires lock
   B resumes; removes A’s fresh lock; acquires lock
   A and B are both alive inside the protected operation
   A exits normally; its EXIT trap removes B’s lock
   B remains alive; lock no longer exists
   C performs ordinary take_lock: exit 0; THIRD_ENTERED
   ```

   Both breakers initially observed a stale lock, so this does not require an operator knowingly overriding a live owner. Separately, the current flag also permits that live override directly.

   **Required correction:** Serialize stale-lock replacement, protect a newly acquired lock from an in-flight breaker, and bind cleanup to the caller’s own lock generation. Add the concurrent stale-breaker and previous-owner-exit regressions.

The requested ordinary recovery behaviors otherwise verify:

| Check | Independent result |
|---|---|
| Published attempt, nonexistent `--log` path, recovery through `final` | Exit **2**; real finalizer reports `EXP11_FINALIZE_REFUSED unreadable log`; canonical directory and link unchanged; `final` remains valid; no aborted sibling. |
| Completed, unpublished attempt with nonexistent log | Exit **2**; completion bytes and filesystem identity unchanged; no aborted sibling. |
| Published and completed attempt | Exit **0**; idempotent skip; no finalizer call or promotion; directory, completion and link unchanged. |
| Orphan attempt with nonexistent log | Exit **2**; renamed to `attempt_A_ABORTED_finalize_refused`. |
| Ordinary concurrent recoveries | First exits **0**, second **2** with owner and `live=yes`; refused caller leaves owner bytes and attempt intact; first publishes a valid `final`. |
| Different published target | Exit **2** without `--replace-final`; both attempts and existing link preserved. |
| Explicit replacement | Exit **0**; `REPLACING … -> attempt_A with attempt_B` precedes promotion; actual atomic replacement leaves a valid link. |
| Full-mode child/finalizer failure | The newly launched attempt and log are aborted; previous `final` survives; lock released. |
| Full-mode success | Fresh attempt supersedes the previous `final`; lock released. |

The missing-log reproductions used the actual finalizer. Successful expensive finalization and training were stubbed where needed; the launcher functions and literal mode branches performed the filesystem operations. Preflight was stubbed for temporary roots. No training or GPU work ran.

**The full/recovery asymmetry is appropriate:** a full invocation explicitly launches and owns a fresh attempt, promoting it only after successful finalization. Recovery replays an existing attempt and should require explicit authorization to replace another publication.

The atomic `mkdir` lock is per arm and precedes recovery alias resolution, attempt validation, finalization and promotion. Full mode participates before launching its child and retains the lock through promotion. Ordinary held-lock refusal is immediate and preserves the owner’s lock. Stale locks survive ordinary refusal; explicit breaking logs `BREAKLOCK`. An ordinary dry-run announces `LOCK` without creating its root; test-root dry-runs take the real lock. Normal success and refusal cleanup passed. The two blockers above prevent approving the broader ownership and exit-path guarantees.

The tests provide useful coverage, with these limits:

- The recovery section contains **eight changed-behavior CLI tests plus an orphan-abort control**. Dry-run preservation tests verify announcements, not a real finalizer refusal or its exit status.
- [The sourced filesystem tests](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:489) exercise real preservation for published-only, completed-only and both-owned shapes, including no `*_ABORTED_*` sibling. My reproductions additionally checked completion contents and filesystem identity.
- [The mutual-exclusion test](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:522) exercises real acquisition, EXIT cleanup and refusal without deleting another owner’s lock. Its callers are sequential, with a manually populated held lock; it does not exercise concurrent live callers or breakers.

RED-for-the-intended-reason evidence was independently replayed using the unchanged historical `0edb7cf` launcher and historical test functions under `/tmp`, with current dependencies:

| Historical test | Observed RED |
|---|---|
| Same-target idempotency | Old launcher returned 0 but announced `ABORT` and `PROMOTE`, without “already published”; intended assertion failed. |
| Different-final conflict | Old launcher returned 0 and announced promotion instead of refusing with 2; intended assertion failed. |

Neither replay failed because of imports, paths or fixture setup.

**Selected verification: 84 passed, no skips.**

| Selection | Passed |
|---|---:|
| Shell | 54 |
| Approval enforcement | 13 |
| Sim-eval closures | 3 |
| Named committed-blob binding test | 1 |
| Registry and pinned oracles | 13 |

All pytest runs used worktree `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, disabled CUDA, thread limits and `/tmp` fixtures/caches. Both shells pass `bash -n`; `git diff --check 612e3dd..48b7bd4` passes. The full CPU suite was not rerun; its reported single stale-exp_06-approval failure is consistent with the independently verified digest change.

The report corrections are valid:

- The alternate fixture is explicitly **not Git-committed** and isolates consumed-record digest comparison. The separate tracked, reviewed-blob requirement is enforced by [exp06_profiles.py:115](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_profiles.py:115) and [its loader:173](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_profiles.py:173). Confirmatory summarization supplies the repository and commit. The named exp_11 committed-blob test passes.
- Exactly **two** exp_11 keys differ from `2c64a2a`: `launch_sh` and `summarize_haa`.
- `0b10d7c` was fix2’s last production-code commit; `1c4498e` changes only record-local `.gitattributes`.

One additional, nonblocking report correction remains: [report:17](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-27_05:35:56_coder_round2_fix3_opus_report.md:17) says every commit is below 200 changed lines. `c8b2ff0` has **524 insertions and 14 deletions**, including copied review material; launcher plus tests alone total **205** changed lines. Acknowledge the exception rather than claim compliance.

Closure hygiene **passes at `48b7bd4`**:

- Exactly **one of twenty exp_06 code keys** differs from the approvals at `9f98bbb`: `summarize_haa`. The other nineteen match.
- All four sim-eval closures—`eval`, `eval_launch`, `compare`, `mirror_probe`—are unchanged.
- Frozen files are untouched. The only modified existing production source versus `9f98bbb` is `tools/exp06_summarize_haa.py`; exp_11 production files are additions.
- `BACKBONES_EXP06` is unchanged; its digest remains `f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169`.
- Pinned `c6233e3` exp_06/exp_09 and `c491396` exp_11 Phase-1 comparisons reproduce serialized statistical payloads and rendered summaries byte-for-byte.
- This fix cycle changes only production `tools/exp11_launch.sh`; only exp_11 `code.launch_sh` moves versus `612e3dd`.

The exact fill below applies to the **unchanged `48b7bd4` candidate contents**. It is not merge authorization: the required launcher corrections will require recomputing the affected closure at the eventual reviewed merge.

Replace only exp_06 `code.summarize_haa`:

```text
old: 645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619
new: deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a
```

Replacing that single digest while preserving every other byte yields a **3,190-byte** exp_06 approval record with SHA-256:

```text
80134832646672411740f779835fa7c06f8bb354b46aab558884503b39a1b666
```

The complete exp_11 candidate fill, including all eight code keys, reused inputs and null artifacts, is:

```json
{
  "schema_version": 1,
  "code": {
    "train": "d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571",
    "finalize": "3eb15b59b672d97ac257054e66f8bc835abd27b85e24f9a417a5a1aaa9e25b9e",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "32a23ee058e2e21da2a922a0042673e092d053b6f8fc9b73eb5fb15595f08e4b",
    "smoke": "62133ba07a35d42026c19fd1484b37c6c06ace0ebedc1cdd8484690df323450e",
    "summarize_haa": "deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "80134832646672411740f779835fa7c06f8bb354b46aab558884503b39a1b666",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "artifacts": {
    "simpor_epoch_012": {
      "epoch": null,
      "path": null,
      "sha256": null
    },
    "simpor_yaw_epoch_012": {
      "epoch": null,
      "path": null,
      "sha256": null
    }
  }
}
```

HEAD and repository status remained unchanged, including the pre-existing untracked close3 review prompt. No review file or repository changes were saved.

