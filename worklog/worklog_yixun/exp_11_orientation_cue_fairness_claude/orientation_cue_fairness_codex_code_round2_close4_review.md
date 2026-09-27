**Reviewer:** OpenAI Codex, GPT-6 per exposed developer metadata. Exact model identifier and reasoning-effort setting are not exposed.  
**Sandbox:** Read-only worktree; temporary writes only under `/tmp`; `CUDA_VISIBLE_DEVICES=''`; numerical thread limits set to 1; no signals sent.  
**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`.  
**HEAD:** `4f72ac27ac54e1b766c346ddbfdef29397361683`.  
**Review window:** 2026-09-27, 11:56–12:00 UTC / 07:56–08:00 EDT.  
**Scope:** Five commits in `48b7bd4..4f72ac2`, combined diff, fix4 report, function/handler replays, selected tests and closure calculations.

**Verdict: request changes. Round 2 cannot be CLOSED, and `4f72ac2` should not be merged.** The original stale-generation reproductions pass, but acquisition still has a race that permits multiple holders.

1. **[P2] A breaker can retire a live acquisition before its owner file exists.**  
   [take_lock:300](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:300), [break_lock:258](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:258)

   `mkdir` makes the lock visible before `take_lock` writes its owner. During that interval, `break_lock` accepts an empty observed generation and finds no live PID.

   I reproduced this with unchanged production functions, pausing A immediately after its actual `mkdir`. No owner records were fabricated:

   ```text
   A creates .publish.lock; pauses before writing owner
   B --break-lock retires the empty directory to .publish.lock.broken.
   B acquires replacement lock, writes B's owner, enters protected work
   A resumes, writes A's owner into B's directory, enters protected work

   A_AND_B_ALIVE True True
   A_OVERWROTE_B_OWNER True
   A_EXIT 0; UNLOCK
   B_STILL_ALIVE True LOCK_PRESENT False
   C ordinary take_lock succeeds
   B_AND_C_ALIVE True True LOCK_PRESENT True
   ```

   The nonce check cannot protect B here: A overwrote the owner file, so A’s cleanup sees its own nonce and removes the shared pathname.

   A second interleaving confirms that the restoration check is insufficient: B checks the empty generation; A completes acquisition; B retires A’s now-live lock; C acquires the exposed pathname. B then detects the changed nonce, fails restoration, logs `UNRESTORED`, and exits 2. **A and C remain live holders.**

   **Required correction:** Coordinate initialization, breaking and acquisition so an incomplete live acquisition cannot be treated as stale, and retirement cannot expose an occupied lock as available. Add deterministic regressions covering both schedules, including normal EXIT cleanup and refusal of a third caller.

2. **[P2] Missing-owner cleanup bypasses the signal handler’s required exit status.**  
   [release_lock:224](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:224), [on_signal:239](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:239)

   `now="$(lock_nonce_of "$LOCK_DIR")"` propagates `sed`’s failure when the owner file is absent. With `set -e`, execution exits before `UNLOCK SKIPPED` and before `exit $((128 + $2))`.

   After the first reproduction naturally removed B’s lock, I replayed B’s registered TERM body:

   ```text
   trap -- 'on_signal TERM 15' SIGTERM
   SIGNAL TERM
   exit: 2
   UNLOCK_SKIPPED_LOGGED False
   ```

   The handler did not resume publication, but its promised exit 143 and ownership-loss cleanup behavior failed.

   **Required correction:** Handle an absent/unreadable owner explicitly as nonownership, preserve any successor lock, and ensure cleanup cannot bypass the handler’s specified exit. Cover registered INT/TERM replay with missing ownership.

The ordinary signal case verifies. [The handlers](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:236) set `INTERRUPTED=1`, log `SIGNAL`, release and terminate. INT/TERM are registered separately; EXIT retains cleanup. With an intact owned lock, direct replay produced **143 for TERM and 130 for INT**, removed the lock and never reached continuation or promotion.

`not_interrupted` guards finalization and promotion in both production branches: [full:491](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:491) and [finalize:582](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:582). The registered-body regression reads `trap -p`, rather than assuming the installed handler.

The requested existing-generation cases also verify:

| Check | Result |
|---|---|
| Owner format | `pid … nonce …` written on successful acquisition. |
| Previous-owner fixture | EXIT logs `UNLOCK SKIPPED`; successor owner and directory survive. |
| Changed observed generation | Breaker refuses; current owner bytes remain unchanged. |
| Live recorded PID | Breaker refuses. |
| Two breakers observing the same nonempty stale generation | With a readiness handshake, fast caller remains alive holding the lock; slow caller exits 2 because retirement fails; owner bytes remain intact. |
| Retirement | Uses `<lock>.broken.<observed>`, then rereads its owner and attempts restoration on mismatch. |
| Barrier | `EXP11_LOCK_BARRIER` is a quoted file path, waited on for at most approximately ten seconds; never evaluated as a command. |

Those passing cases do not cover the owner-initialization interval identified above.

**RED evidence was independently replayed.** Using the historical launcher at `fa12f00` with the regression’s actual test function:

```text
UNLOCK …
CONTINUED_AFTER_HANDLER LOCK_HELD=0
PROMOTE …/final -> attempt_X
exit: 0
EXPECTED RED ASSERTION:
the TERM handler returned and execution resumed without the lock
```

The failure was the intended assertion, not an import or fixture failure. Current registered-handler replay passes for intact ownership.

The commit-size record needs a nonblocking correction at [report:85](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-27_06:52:49_coder_round2_fix4_opus_report.md:85):

- `c8b2ff0` is **524 insertions / 14 deletions**. Launcher plus tests account for **205 changed lines**, not 205 insertions. Its exception is now acknowledged.
- `16b1eff` itself is **96 insertions / 9 deletions**, touching only the launcher. **224 insertions / 9 deletions** is the combined RED-plus-GREEN total; `fa12f00` separately added 128 test lines.
- `a2cb3b6` adds **437 record-only lines** and is the actual over-200-line commit in this cycle that should be recorded.

**Selected verification: 91 passed, no skips.**

| Selection | Passed |
|---|---:|
| Shell tests | 61 |
| Approval enforcement | 13 |
| Committed-blob binding | 1 |
| Sim-eval closures | 3 |
| Registry | 6 |
| Pinned-oracle identity and parity | 7 |

Both shells pass `bash -n`; `git diff --check 48b7bd4 4f72ac2` passes. Pytest used worktree `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, CPU-only settings and `/tmp` fixtures/caches.

I did not rerun the full CPU suite. The report’s **3,690 passed / 55 skipped / one stale-exp_06-approval failure** is consistent with the independently verified approval drift; the passing suite does not cover the new race.

Closure hygiene passes at this candidate:

- Against the approval record at `9f98bbb`, exactly **one of twenty exp_06 keys** differs: `summarize_haa`. The other nineteen match.
- All four sim-eval closures—`eval`, `eval_launch`, `compare`, `mirror_probe`—are unchanged; their three tests pass.
- Frozen files are untouched. The only modified existing production source versus `9f98bbb` is `tools/exp06_summarize_haa.py`; exp_11 production files are additions.
- `BACKBONES_EXP06` remains unchanged, including inherited class/output parity. Its digest remains `f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169`.
- Serialized statistical payloads and rendered summaries are byte-identical against pinned `c6233e3` exp_06/exp_09 and `c491396` exp_11 Phase-1 oracles.
- This cycle changes only production `tools/exp11_launch.sh`; only exp_11 `code.launch_sh` moves versus `48b7bd4`.
- Exp_11 approval leaves remain null in the repository.

The exact fill below applies to **unchanged `4f72ac2` candidate contents**. Because merge is blocked, it is not an approved merge manifest. Recompute affected closures after correcting the launcher.

Replace only exp_06 `code.summarize_haa`:

```text
old: 645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619
new: deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a
```

Replacing that single value while preserving every other byte yields a **3,190-byte** exp_06 approval record with SHA-256:

```text
80134832646672411740f779835fa7c06f8bb354b46aab558884503b39a1b666
```

All eight exp_11 code keys, reused identities and null artifact leaves:

```json
{
  "schema_version": 1,
  "code": {
    "train": "d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571",
    "finalize": "3eb15b59b672d97ac257054e66f8bc835abd27b85e24f9a417a5a1aaa9e25b9e",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "3cff85cc57cfedb3682fc9c110de3ce78e9aae413a8399b8fda62ecfe0b9279f",
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

HEAD and worktree contents remained unchanged, including the pre-existing untracked close4 prompt. No review file or approval record was saved.
tokens used
70,304
**Reviewer:** OpenAI Codex, GPT-6 per exposed developer metadata. Exact model identifier and reasoning-effort setting are not exposed.  
**Sandbox:** Read-only worktree; temporary writes only under `/tmp`; `CUDA_VISIBLE_DEVICES=''`; numerical thread limits set to 1; no signals sent.  
**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`.  
**HEAD:** `4f72ac27ac54e1b766c346ddbfdef29397361683`.  
**Review window:** 2026-09-27, 11:56–12:00 UTC / 07:56–08:00 EDT.  
**Scope:** Five commits in `48b7bd4..4f72ac2`, combined diff, fix4 report, function/handler replays, selected tests and closure calculations.

**Verdict: request changes. Round 2 cannot be CLOSED, and `4f72ac2` should not be merged.** The original stale-generation reproductions pass, but acquisition still has a race that permits multiple holders.

1. **[P2] A breaker can retire a live acquisition before its owner file exists.**  
   [take_lock:300](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:300), [break_lock:258](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:258)

   `mkdir` makes the lock visible before `take_lock` writes its owner. During that interval, `break_lock` accepts an empty observed generation and finds no live PID.

   I reproduced this with unchanged production functions, pausing A immediately after its actual `mkdir`. No owner records were fabricated:

   ```text
   A creates .publish.lock; pauses before writing owner
   B --break-lock retires the empty directory to .publish.lock.broken.
   B acquires replacement lock, writes B's owner, enters protected work
   A resumes, writes A's owner into B's directory, enters protected work

   A_AND_B_ALIVE True True
   A_OVERWROTE_B_OWNER True
   A_EXIT 0; UNLOCK
   B_STILL_ALIVE True LOCK_PRESENT False
   C ordinary take_lock succeeds
   B_AND_C_ALIVE True True LOCK_PRESENT True
   ```

   The nonce check cannot protect B here: A overwrote the owner file, so A’s cleanup sees its own nonce and removes the shared pathname.

   A second interleaving confirms that the restoration check is insufficient: B checks the empty generation; A completes acquisition; B retires A’s now-live lock; C acquires the exposed pathname. B then detects the changed nonce, fails restoration, logs `UNRESTORED`, and exits 2. **A and C remain live holders.**

   **Required correction:** Coordinate initialization, breaking and acquisition so an incomplete live acquisition cannot be treated as stale, and retirement cannot expose an occupied lock as available. Add deterministic regressions covering both schedules, including normal EXIT cleanup and refusal of a third caller.

2. **[P2] Missing-owner cleanup bypasses the signal handler’s required exit status.**  
   [release_lock:224](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:224), [on_signal:239](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:239)

   `now="$(lock_nonce_of "$LOCK_DIR")"` propagates `sed`’s failure when the owner file is absent. With `set -e`, execution exits before `UNLOCK SKIPPED` and before `exit $((128 + $2))`.

   After the first reproduction naturally removed B’s lock, I replayed B’s registered TERM body:

   ```text
   trap -- 'on_signal TERM 15' SIGTERM
   SIGNAL TERM
   exit: 2
   UNLOCK_SKIPPED_LOGGED False
   ```

   The handler did not resume publication, but its promised exit 143 and ownership-loss cleanup behavior failed.

   **Required correction:** Handle an absent/unreadable owner explicitly as nonownership, preserve any successor lock, and ensure cleanup cannot bypass the handler’s specified exit. Cover registered INT/TERM replay with missing ownership.

The ordinary signal case verifies. [The handlers](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:236) set `INTERRUPTED=1`, log `SIGNAL`, release and terminate. INT/TERM are registered separately; EXIT retains cleanup. With an intact owned lock, direct replay produced **143 for TERM and 130 for INT**, removed the lock and never reached continuation or promotion.

`not_interrupted` guards finalization and promotion in both production branches: [full:491](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:491) and [finalize:582](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:582). The registered-body regression reads `trap -p`, rather than assuming the installed handler.

The requested existing-generation cases also verify:

| Check | Result |
|---|---|
| Owner format | `pid … nonce …` written on successful acquisition. |
| Previous-owner fixture | EXIT logs `UNLOCK SKIPPED`; successor owner and directory survive. |
| Changed observed generation | Breaker refuses; current owner bytes remain unchanged. |
| Live recorded PID | Breaker refuses. |
| Two breakers observing the same nonempty stale generation | With a readiness handshake, fast caller remains alive holding the lock; slow caller exits 2 because retirement fails; owner bytes remain intact. |
| Retirement | Uses `<lock>.broken.<observed>`, then rereads its owner and attempts restoration on mismatch. |
| Barrier | `EXP11_LOCK_BARRIER` is a quoted file path, waited on for at most approximately ten seconds; never evaluated as a command. |

Those passing cases do not cover the owner-initialization interval identified above.

**RED evidence was independently replayed.** Using the historical launcher at `fa12f00` with the regression’s actual test function:

```text
UNLOCK …
CONTINUED_AFTER_HANDLER LOCK_HELD=0
PROMOTE …/final -> attempt_X
exit: 0
EXPECTED RED ASSERTION:
the TERM handler returned and execution resumed without the lock
```

The failure was the intended assertion, not an import or fixture failure. Current registered-handler replay passes for intact ownership.

The commit-size record needs a nonblocking correction at [report:85](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-27_06:52:49_coder_round2_fix4_opus_report.md:85):

- `c8b2ff0` is **524 insertions / 14 deletions**. Launcher plus tests account for **205 changed lines**, not 205 insertions. Its exception is now acknowledged.
- `16b1eff` itself is **96 insertions / 9 deletions**, touching only the launcher. **224 insertions / 9 deletions** is the combined RED-plus-GREEN total; `fa12f00` separately added 128 test lines.
- `a2cb3b6` adds **437 record-only lines** and is the actual over-200-line commit in this cycle that should be recorded.

**Selected verification: 91 passed, no skips.**

| Selection | Passed |
|---|---:|
| Shell tests | 61 |
| Approval enforcement | 13 |
| Committed-blob binding | 1 |
| Sim-eval closures | 3 |
| Registry | 6 |
| Pinned-oracle identity and parity | 7 |

Both shells pass `bash -n`; `git diff --check 48b7bd4 4f72ac2` passes. Pytest used worktree `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider`, CPU-only settings and `/tmp` fixtures/caches.

I did not rerun the full CPU suite. The report’s **3,690 passed / 55 skipped / one stale-exp_06-approval failure** is consistent with the independently verified approval drift; the passing suite does not cover the new race.

Closure hygiene passes at this candidate:

- Against the approval record at `9f98bbb`, exactly **one of twenty exp_06 keys** differs: `summarize_haa`. The other nineteen match.
- All four sim-eval closures—`eval`, `eval_launch`, `compare`, `mirror_probe`—are unchanged; their three tests pass.
- Frozen files are untouched. The only modified existing production source versus `9f98bbb` is `tools/exp06_summarize_haa.py`; exp_11 production files are additions.
- `BACKBONES_EXP06` remains unchanged, including inherited class/output parity. Its digest remains `f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169`.
- Serialized statistical payloads and rendered summaries are byte-identical against pinned `c6233e3` exp_06/exp_09 and `c491396` exp_11 Phase-1 oracles.
- This cycle changes only production `tools/exp11_launch.sh`; only exp_11 `code.launch_sh` moves versus `48b7bd4`.
- Exp_11 approval leaves remain null in the repository.

The exact fill below applies to **unchanged `4f72ac2` candidate contents**. Because merge is blocked, it is not an approved merge manifest. Recompute affected closures after correcting the launcher.

Replace only exp_06 `code.summarize_haa`:

```text
old: 645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619
new: deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a
```

Replacing that single value while preserving every other byte yields a **3,190-byte** exp_06 approval record with SHA-256:

```text
80134832646672411740f779835fa7c06f8bb354b46aab558884503b39a1b666
```

All eight exp_11 code keys, reused identities and null artifact leaves:

```json
{
  "schema_version": 1,
  "code": {
    "train": "d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571",
    "finalize": "3eb15b59b672d97ac257054e66f8bc835abd27b85e24f9a417a5a1aaa9e25b9e",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "3cff85cc57cfedb3682fc9c110de3ce78e9aae413a8399b8fda62ecfe0b9279f",
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

HEAD and worktree contents remained unchanged, including the pre-existing untracked close4 prompt. No review file or approval record was saved.

