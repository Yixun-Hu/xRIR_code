**Reviewer:** OpenAI Codex, GPT-6 per developer metadata; exact model identifier and reasoning-effort setting are not exposed.  
**Sandbox:** Read-only worktree; writes confined to `/tmp`; CUDA disabled; numerical threads limited to one per worker, at most four concurrent workers; no signals sent.  
**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`.  
**HEAD:** `86b1a465ba746da47058a2c90088b053b1d8dac3`.  
**Review window:** 2026-09-27, 09:15–09:20 EDT / 13:15–13:20 UTC.  
**Scope:** Five commits in `4f72ac2..86b1a46`, combined diff, fix5 report, independent schedules, selected tests and closure calculations.

**Verdict: request changes. Round 2 cannot be CLOSED, and `86b1a46` should not be merged.** Missing-owner signal cleanup is fixed, but acquisition still permits multiple holders, and normal EXIT can leak the breaking meta-lock.

1. **[P2] Acquisition can adopt or corrupt a successor’s lock after the grace expires.**  
   [Inode capture:397](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:397), [owner write:405](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:405), [verification:417](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:417).

   The inode is read after `mkdir`, and both owner-write operations resolve the shared pathname again. Atomic `.owner.$$` → `owner` replacement does not bind the write to the directory A created.

   I replayed the first close4 schedule using unchanged production functions and a file barrier immediately after the actual `mkdir`. With the default grace unchanged, I waited **31.267 seconds**, without adjusting directory timestamps:

   ```text
   A creates lock; pauses before inode capture and owner write
   B breaks the now-old ownerless directory and acquires its replacement
   A resumes, records B's inode, overwrites B's owner, passes verification

   A_AND_B_HOLD True True
   A_OVERWROTE_B_OWNER True
   A_EXIT 0
   B_STILL_HOLDING True
   LOCK_EXISTS False

   C acquires
   B_AND_C_HOLD True True
   ```

   A’s normal EXIT removed B’s lock. B’s later EXIT preserved C’s generation, but mutual exclusion had already failed twice.

   A second replay paused A **after capturing its original inode**. A subsequently overwrote B’s owner and then refused because the inode differed. After A exited, C’s breaker saw A’s dead PID and acquired while B remained active. Thus “refuses and removes nothing” still allows successor corruption.

   **Required correction:** Coordinate initialization and retirement so a delayed acquirer cannot adopt a replacement directory or write into its successor. Cover pauses both before inode capture and after it, including grace expiry, third callers and normal EXIT.

2. **[P2] The breaking meta-lock does not exclude callers that passed its earlier check, and can survive normal EXIT.**  
   [Meta-lock check:382](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:382), [mkdir failure:387](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:387), [late trap installation:400](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:400).

   A caller can pass the `.breaking` check, pause before `mkdir`, then create the exposed pathname during retirement. Independent replay with actual `mkdir`/`mv` operations and file barriers produced:

   ```text
   C passes .breaking check; pauses before mkdir
   B acquires .breaking and retires the stale directory
   C creates the exposed pathname, writes owner, then refuses on .breaking
   B's replacement mkdir fails
   B exits 2 before installing its EXIT trap

   breaking_survived_normal_EXIT: true
   D --break-lock exits 2: a break is already in progress
   breaking_still_present: true
   ```

   No holder remains, but subsequent acquisitions and breakers are blocked. The failure return at line 394 does not release the already-held meta-lock.

   The same prechecked-caller interleaving also makes [the `UNRESTORED` branch:361](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:361) reachable. B validated an empty generation; A then wrote its owner and refused on `.breaking`; B retired the directory; C created the exposed pathname before B attempted restoration:

   ```text
   LOCK_ABSENT_WITH_META True True
   C_EXPOSED_PATH_CREATED True
   UNRESTORED .../.publish.lock.broken.empty could not be put back
   UNRESTORED_reached: true
   retired_left: true
   ```

   **Required correction:** Make acquisition participate in exclusion across the check-to-`mkdir` interval. Establish cleanup before acquiring `.breaking`, and release it on every failure and EXIT path, including replacement-`mkdir` failure.

The requested checks therefore have mixed results:

| Check | Result |
|---|---|
| Atomic `.breaking` creation; second breaker refuses | Pass for callers observing the existing meta-lock. |
| Meta-lock excludes every acquisition and cleans up on every exit | **Fail**, blocker 2. |
| Default 30-second initialization grace | Young ownerless lock refuses as “acquisition in progress”; older lock is breakable. |
| Self-verification preserves successor ownership | **Fail**, blocker 1. |
| Rename retirement and post-rename generation check | Present; `UNRESTORED` remains reachable. |
| Both barrier variables are paths, never evaluated | Pass. |
| Owner readers tolerate missing/unreadable files | Pass. |
| Missing-owner TERM exits 143 and preserves lock | Pass. |
| Close4 schedules, third callers and normal EXIT preserve mutual exclusion | **Fail after grace expiry.** |

The short, within-grace schedules pass: B refuses A’s unfinished acquisition; a third caller refuses; A’s normal EXIT removes its own lock. With B actually observing empty before A completes, B subsequently refuses the live owner and A remains the sole holder. These passing cases do not establish safety after grace expiry or after B has already validated the empty generation.

The committed regressions miss those intervals. [Schedule one:740](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:740) uses `EXP11_ACQUIRE_BARRIER`, whose hook is **after the owner write**. [Schedule two:765](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:765) waits for an owner before calling `break_lock ""`; it does not pause B after validating empty and before retirement.

**The missing-owner handler defect is closed.** Replaying the registered handler body, without sending a signal, produced:

```text
REGISTERED on_signal TERM 15
SIGNAL TERM
UNLOCK SKIPPED ... has no owner: this invocation owns nothing there
status: 143
lock_preserved: True
```

Missing-owner INT returned 130 and preserved the directory. Intact-owner TERM returned 143 and removed its lock. An unreadable owner also returned 143 and preserved the directory, logging `UNLOCK SKIPPED … generation none`. No replay continued past the handler.

**RED evidence independently reproduced.** The committed missing-owner regression against the byte-exact `ad9317e` launcher and its dependency failed for the intended reason:

```text
LOCK ...
SIGNAL TERM
status: 2
UNLOCK SKIPPED absent
lock preserved: True
```

**The exception record still needs correction.** [Fix5 report:96](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-27_08:10:41_coder_round2_fix5_opus_report.md:96) assigns the combined RED+GREEN count to GREEN alone:

| Commit/range | Verified accounting |
|---|---|
| `519b48f` alone | **149 insertions / 31 deletions — 180 changed lines**, below 200. |
| `4f72ac2..519b48f` | **306 insertions / 27 deletions — 333 changed lines.** |
| `16b1eff` | **96 insertions / 9 deletions**, correctly recorded. |
| `a2cb3b6` | **437 record-only insertions**, correctly recorded. |
| `c8b2ff0` launcher plus tests | **205 changed lines**, correctly distinguished from insertions. |

For completeness, `c8b2ff0` totals 524 insertions / 14 deletions. Its remaining 333 insertions comprise 331 record lines and two golden-file insertions; the report’s “other 332” statement is also inaccurate. These bookkeeping corrections are nonblocking independently of the lock defects.

**Selected verification: 100 distinct tests passed, no skips.**

| Selection | Passed |
|---|---:|
| Shell | 70 |
| Approval enforcement | 13 |
| Committed-blob binding | 1 |
| Sim-eval closures | 3 |
| Registry | 6 |
| Pinned-oracle identity and parity | 7 |

There were 101 test executions because committed-blob binding ran in both verification selections. Pytest used worktree `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, CPU-only settings and `/tmp` caches/fixtures. Both shells passed `bash -n`; `git diff --check 4f72ac2 86b1a46` passed. I did not rerun the full CPU suite.

**Closure hygiene passes at `86b1a46`:**

- Against the approval record committed at `9f98bbb`, only **exp_06 `code.summarize_haa`** differs among twenty keys; the other nineteen match.
- All four sim-eval closures—`eval`, `eval_launch`, `compare`, `mirror_probe`—are unchanged; their three tests pass.
- Frozen files remain untouched. The only modified existing production source versus `9f98bbb` is `tools/exp06_summarize_haa.py`; exp_11 production files are additions.
- `BACKBONES_EXP06`, inherited class identities and outputs remain unchanged. Its digest is `f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169`.
- Statistical payloads and rendered summaries are byte-identical against pinned `c6233e3` exp_06/exp_09 and `c491396` exp_11 Phase-1 oracles.
- This cycle changes only production `tools/exp11_launch.sh`; only exp_11 `code.launch_sh` moves.
- The checked-in exp_11 approval record remains identical to the all-null template.

**Exact fill values for the unchanged candidate follow.** Because merge is blocked, this is not an approved merge manifest. Recompute affected closures after fixing the launcher and verify them at the eventual merge commit.

Replace only exp_06 `code.summarize_haa`:

```text
old: 645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619
new: deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a
```

Preserving every other byte produces a **3,190-byte** exp_06 approval record with SHA-256:

```text
80134832646672411740f779835fa7c06f8bb354b46aab558884503b39a1b666
```

All eight exp_11 code keys, independently verified reused identities, and null artifact leaves:

```json
{
  "schema_version": 1,
  "code": {
    "train": "d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571",
    "finalize": "3eb15b59b672d97ac257054e66f8bc835abd27b85e24f9a417a5a1aaa9e25b9e",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "fbc012454f0d7962d1c58fd778baa670d1340e56cab71131ba769e95f214d4c8",
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

HEAD and worktree contents remained unchanged, including the pre-existing untracked close5 prompt. No review file or approval record was saved.
tokens used
57,408
**Reviewer:** OpenAI Codex, GPT-6 per developer metadata; exact model identifier and reasoning-effort setting are not exposed.  
**Sandbox:** Read-only worktree; writes confined to `/tmp`; CUDA disabled; numerical threads limited to one per worker, at most four concurrent workers; no signals sent.  
**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`.  
**HEAD:** `86b1a465ba746da47058a2c90088b053b1d8dac3`.  
**Review window:** 2026-09-27, 09:15–09:20 EDT / 13:15–13:20 UTC.  
**Scope:** Five commits in `4f72ac2..86b1a46`, combined diff, fix5 report, independent schedules, selected tests and closure calculations.

**Verdict: request changes. Round 2 cannot be CLOSED, and `86b1a46` should not be merged.** Missing-owner signal cleanup is fixed, but acquisition still permits multiple holders, and normal EXIT can leak the breaking meta-lock.

1. **[P2] Acquisition can adopt or corrupt a successor’s lock after the grace expires.**  
   [Inode capture:397](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:397), [owner write:405](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:405), [verification:417](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:417).

   The inode is read after `mkdir`, and both owner-write operations resolve the shared pathname again. Atomic `.owner.$$` → `owner` replacement does not bind the write to the directory A created.

   I replayed the first close4 schedule using unchanged production functions and a file barrier immediately after the actual `mkdir`. With the default grace unchanged, I waited **31.267 seconds**, without adjusting directory timestamps:

   ```text
   A creates lock; pauses before inode capture and owner write
   B breaks the now-old ownerless directory and acquires its replacement
   A resumes, records B's inode, overwrites B's owner, passes verification

   A_AND_B_HOLD True True
   A_OVERWROTE_B_OWNER True
   A_EXIT 0
   B_STILL_HOLDING True
   LOCK_EXISTS False

   C acquires
   B_AND_C_HOLD True True
   ```

   A’s normal EXIT removed B’s lock. B’s later EXIT preserved C’s generation, but mutual exclusion had already failed twice.

   A second replay paused A **after capturing its original inode**. A subsequently overwrote B’s owner and then refused because the inode differed. After A exited, C’s breaker saw A’s dead PID and acquired while B remained active. Thus “refuses and removes nothing” still allows successor corruption.

   **Required correction:** Coordinate initialization and retirement so a delayed acquirer cannot adopt a replacement directory or write into its successor. Cover pauses both before inode capture and after it, including grace expiry, third callers and normal EXIT.

2. **[P2] The breaking meta-lock does not exclude callers that passed its earlier check, and can survive normal EXIT.**  
   [Meta-lock check:382](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:382), [mkdir failure:387](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:387), [late trap installation:400](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:400).

   A caller can pass the `.breaking` check, pause before `mkdir`, then create the exposed pathname during retirement. Independent replay with actual `mkdir`/`mv` operations and file barriers produced:

   ```text
   C passes .breaking check; pauses before mkdir
   B acquires .breaking and retires the stale directory
   C creates the exposed pathname, writes owner, then refuses on .breaking
   B's replacement mkdir fails
   B exits 2 before installing its EXIT trap

   breaking_survived_normal_EXIT: true
   D --break-lock exits 2: a break is already in progress
   breaking_still_present: true
   ```

   No holder remains, but subsequent acquisitions and breakers are blocked. The failure return at line 394 does not release the already-held meta-lock.

   The same prechecked-caller interleaving also makes [the `UNRESTORED` branch:361](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:361) reachable. B validated an empty generation; A then wrote its owner and refused on `.breaking`; B retired the directory; C created the exposed pathname before B attempted restoration:

   ```text
   LOCK_ABSENT_WITH_META True True
   C_EXPOSED_PATH_CREATED True
   UNRESTORED .../.publish.lock.broken.empty could not be put back
   UNRESTORED_reached: true
   retired_left: true
   ```

   **Required correction:** Make acquisition participate in exclusion across the check-to-`mkdir` interval. Establish cleanup before acquiring `.breaking`, and release it on every failure and EXIT path, including replacement-`mkdir` failure.

The requested checks therefore have mixed results:

| Check | Result |
|---|---|
| Atomic `.breaking` creation; second breaker refuses | Pass for callers observing the existing meta-lock. |
| Meta-lock excludes every acquisition and cleans up on every exit | **Fail**, blocker 2. |
| Default 30-second initialization grace | Young ownerless lock refuses as “acquisition in progress”; older lock is breakable. |
| Self-verification preserves successor ownership | **Fail**, blocker 1. |
| Rename retirement and post-rename generation check | Present; `UNRESTORED` remains reachable. |
| Both barrier variables are paths, never evaluated | Pass. |
| Owner readers tolerate missing/unreadable files | Pass. |
| Missing-owner TERM exits 143 and preserves lock | Pass. |
| Close4 schedules, third callers and normal EXIT preserve mutual exclusion | **Fail after grace expiry.** |

The short, within-grace schedules pass: B refuses A’s unfinished acquisition; a third caller refuses; A’s normal EXIT removes its own lock. With B actually observing empty before A completes, B subsequently refuses the live owner and A remains the sole holder. These passing cases do not establish safety after grace expiry or after B has already validated the empty generation.

The committed regressions miss those intervals. [Schedule one:740](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:740) uses `EXP11_ACQUIRE_BARRIER`, whose hook is **after the owner write**. [Schedule two:765](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:765) waits for an owner before calling `break_lock ""`; it does not pause B after validating empty and before retirement.

**The missing-owner handler defect is closed.** Replaying the registered handler body, without sending a signal, produced:

```text
REGISTERED on_signal TERM 15
SIGNAL TERM
UNLOCK SKIPPED ... has no owner: this invocation owns nothing there
status: 143
lock_preserved: True
```

Missing-owner INT returned 130 and preserved the directory. Intact-owner TERM returned 143 and removed its lock. An unreadable owner also returned 143 and preserved the directory, logging `UNLOCK SKIPPED … generation none`. No replay continued past the handler.

**RED evidence independently reproduced.** The committed missing-owner regression against the byte-exact `ad9317e` launcher and its dependency failed for the intended reason:

```text
LOCK ...
SIGNAL TERM
status: 2
UNLOCK SKIPPED absent
lock preserved: True
```

**The exception record still needs correction.** [Fix5 report:96](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-27_08:10:41_coder_round2_fix5_opus_report.md:96) assigns the combined RED+GREEN count to GREEN alone:

| Commit/range | Verified accounting |
|---|---|
| `519b48f` alone | **149 insertions / 31 deletions — 180 changed lines**, below 200. |
| `4f72ac2..519b48f` | **306 insertions / 27 deletions — 333 changed lines.** |
| `16b1eff` | **96 insertions / 9 deletions**, correctly recorded. |
| `a2cb3b6` | **437 record-only insertions**, correctly recorded. |
| `c8b2ff0` launcher plus tests | **205 changed lines**, correctly distinguished from insertions. |

For completeness, `c8b2ff0` totals 524 insertions / 14 deletions. Its remaining 333 insertions comprise 331 record lines and two golden-file insertions; the report’s “other 332” statement is also inaccurate. These bookkeeping corrections are nonblocking independently of the lock defects.

**Selected verification: 100 distinct tests passed, no skips.**

| Selection | Passed |
|---|---:|
| Shell | 70 |
| Approval enforcement | 13 |
| Committed-blob binding | 1 |
| Sim-eval closures | 3 |
| Registry | 6 |
| Pinned-oracle identity and parity | 7 |

There were 101 test executions because committed-blob binding ran in both verification selections. Pytest used worktree `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, CPU-only settings and `/tmp` caches/fixtures. Both shells passed `bash -n`; `git diff --check 4f72ac2 86b1a46` passed. I did not rerun the full CPU suite.

**Closure hygiene passes at `86b1a46`:**

- Against the approval record committed at `9f98bbb`, only **exp_06 `code.summarize_haa`** differs among twenty keys; the other nineteen match.
- All four sim-eval closures—`eval`, `eval_launch`, `compare`, `mirror_probe`—are unchanged; their three tests pass.
- Frozen files remain untouched. The only modified existing production source versus `9f98bbb` is `tools/exp06_summarize_haa.py`; exp_11 production files are additions.
- `BACKBONES_EXP06`, inherited class identities and outputs remain unchanged. Its digest is `f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169`.
- Statistical payloads and rendered summaries are byte-identical against pinned `c6233e3` exp_06/exp_09 and `c491396` exp_11 Phase-1 oracles.
- This cycle changes only production `tools/exp11_launch.sh`; only exp_11 `code.launch_sh` moves.
- The checked-in exp_11 approval record remains identical to the all-null template.

**Exact fill values for the unchanged candidate follow.** Because merge is blocked, this is not an approved merge manifest. Recompute affected closures after fixing the launcher and verify them at the eventual merge commit.

Replace only exp_06 `code.summarize_haa`:

```text
old: 645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619
new: deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a
```

Preserving every other byte produces a **3,190-byte** exp_06 approval record with SHA-256:

```text
80134832646672411740f779835fa7c06f8bb354b46aab558884503b39a1b666
```

All eight exp_11 code keys, independently verified reused identities, and null artifact leaves:

```json
{
  "schema_version": 1,
  "code": {
    "train": "d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571",
    "finalize": "3eb15b59b672d97ac257054e66f8bc835abd27b85e24f9a417a5a1aaa9e25b9e",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "fbc012454f0d7962d1c58fd778baa670d1340e56cab71131ba769e95f214d4c8",
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

HEAD and worktree contents remained unchanged, including the pre-existing untracked close5 prompt. No review file or approval record was saved.

