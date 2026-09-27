**OpenAI Codex — exp_11 `orientation_cue_fairness`, round 2, close review 6**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not supplied in the visible invocation metadata; I cannot attest the plan’s requested `ultra` setting |
| Sandbox | Read-only repository; authorized test/cache writes confined to `/tmp` |
| Worktree / branch | `/home/yixunhu/codespace/xRIR_code_wt` / `exp11-cue` |
| HEAD | `94efbc2b75c64e8ab8927bad0173adb549ae540e` |
| Git tree | `2cd5787101ac92dafde00203ce65fddc07acd4db` |
| Last verification | 2026-09-27 15:05:20 UTC / 11:05:20 EDT |

**Verdict: request changes. Round 2 cannot be CLOSED, and `94efbc2` should not be merged as reviewed.**

I found one production blocker and one regression-coverage blocker.

1. **[P1] An orphaned FIFO sink can retain the publication lock indefinitely.**  
   [tools/exp11_launch.sh:208](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:208) opens an inheritable descriptor. The shared lifecycle passes it into both the background sink at [tools/exp06_launch.sh:138](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_launch.sh:138) and the `nohup setsid` child at line 141.

   I reproduced this using the real sourced `run_child` and real `flock`. A DEBUG hook replayed the registered TERM handler after the sink started, immediately before `CHILD_STARTED_AT` and before the training child launched:

   ```text
   launcher exit:                    143
   child.pid:                       absent
   sink fd 10:                      <arm-root>/.publish.lock
   sink /proc/.../wchan:             wait_for_partner
   independent flock -n:            1 (refused)
   after opening/closing own FIFO:  0 (acquired)
   ```

   The orphaned sink waits for a FIFO writer that will never arrive while retaining the lock. This creates an indefinitely locked arm without a running trainer. The 36-hour training timeout cannot help because the trainer never started.

   A second replay confirmed ordinary child inheritance: after the launcher exited 143, the detached Python helper still held fd 10 and acquisition failed until the child and sink ended.

   Retaining exclusion while an orphan trainer finishes can be defensible, but **the sink deadlock makes the current implementation unacceptable**. The header’s launcher-exit release promise is also false: release occurs when the last inherited descriptor closes. Fix descriptor propagation in exp_11-owned code, ensuring the sink cannot retain the fd while opening the FIFO, and test this exact boundary. Keep the frozen exp_06 file untouched.

2. **[P2] Still-required recovery and signal regressions were deleted with the obsolete lock tests.**  
   The deletions around [tests/test_exp11_shell.py:394](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:394) and [line 437](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:437) removed coverage for incomplete-orphan recovery, same-target idempotency, different-final refusal/`--replace-final`, TERM termination, INT termination, and `not_interrupted`.

   These behaviors remain requirements of A1. Restore their tests with flock-compatible fixtures and add the production-lifecycle regression above. The cycle removed 24 test functions and added eight; the net reduction of 16 does not mean all removed coverage was obsolete. The fix6 report’s lines 80–83 and 165–166 incorrectly describe that coverage as retained or obsolete.

The remaining requested checks passed:

| Check | Verified result |
|---|---|
| Acquisition count and scope | Exactly one call in full mode, at line 375, and recovery, at line 491. Full acquires before attempt creation/child launch; recovery acquires before canonical resolution. The parent retains its fd through promotion. |
| Lock-file handling | `exec {LOCK_FD}>>"$LOCK_FILE"` opens/creates without truncation; acquisition writes no bytes. `flock -n` refuses immediately with a clear message. |
| Held-lock refusal | With my own exact `flock lockfile sleep 60` helper, recovery and full test-root dry-runs returned 2 in approximately 21 ms and 20 ms. Neither announced child launch or promotion. |
| Refusal preservation | Attempt and completion bytes, `final`, directory entries, and lock-file bytes/inode/mtime/ctime remained unchanged. |
| Standalone holder release | After that helper ended naturally, acquisition and recovery succeeded. Standalone registered-handler replays also released their lock on exit. This does **not** establish production release because of blocker 1. |
| Old mechanism | No executable old lock machinery remains. Removed identifiers occur only in negative absence assertions and historical documentation. `OWNER`/`--owner-pid` remains for unrelated finalizer ownership. |
| Recovery behavior | Canonical aliases, published/completed-attempt preservation, orphan abort, idempotency, conflict refusal, and explicit replacement behavior passed selected tests or additional direct replays. |
| Ceiling/root restrictions | Bounded ceiling validation and the requirement for both `EXP11_TEST_ROOTS=1` and `--dry-run` remain intact and tested. |
| Signals and guards | Registered TERM/INT bodies returned 143/130 with no continuation. `not_interrupted` refused when set. Guards remain before real full finalization/promotion at lines 410/415 and recovery finalization/promotion at lines 501/506. |
| Ordinary dry-run | Announces `LOCK` without acquiring outside test roots. Test-root dry-runs acquire the real lock. |
| Other execution paths | I found no production full/recovery publication path that bypasses acquisition. Command substitutions and ordinary subprocesses do not discard the parent’s lock. Descriptor inheritance extends its lifetime instead. |

Independent verification produced **64 passing HEAD tests**: 54 shell tests, all three `tests/test_exp09_sim_eval_closures.py` tests, and seven pinned-oracle checks. Tests used `-p no:cacheprovider`, the worktree `PYTHONPATH`, disabled bytecode, empty CUDA visibility, one-thread numerical-library limits, and `/tmp` fixtures/caches. Shell syntax checks and `git diff --check 86b1a46 94efbc2` passed.

I replayed RED commit **`76d82f3`** from files extracted under `/tmp`, without checking out or modifying the worktree. Its acquisition-preservation test failed exactly as intended:

```text
assert b'' == b'not the launcher\n'
```

Recovery success and `PROMOTE` assertions had already passed. This confirms the RED evidence for the truncation fixed by `94efbc2`.

The Planner’s full suite remains pending; its log had reached approximately 19% at my last read. I did not signal or otherwise interfere with it. Its eventual result cannot resolve the demonstrated lifecycle defect.

The [fix6 report’s line-count table](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-27_09:27:14_coder_round2_fix6_opus_report.md:121) needs these corrections:

| Commit | Actual production changes | Actual test changes |
|---|---:|---:|
| `811de75` | — | +92 / −386 |
| `f80574a` | +44 / −235 | +7 / −3 |
| `04c703e` | — | +17 / −0 |
| `531db17` | +0 / −1 | — |
| `76d82f3` | — | +16 / −0 |
| `94efbc2` | +2 / −2 | — |
| Entire committed range | +43 / −235 | +124 / −381 |

`f80574a` changes **279 production lines**, despite shrinking the launcher by 191. The statement that no production commit approaches 200 changed lines is incorrect and does not satisfy the plan’s stated bound.

Committed record changes total **+509/−2**, as reported; including the modified working-tree report makes that **+596/−2**. The carried-over corrections are correct: `519b48f` is +149/−31 = 180 changed lines; `4f72ac2..519b48f` is +306/−27; and `c8b2ff0`’s remaining 333 insertions comprise 331 record lines plus two golden lines.

Closure hygiene at `94efbc2` passed:

- All 20 exp_06 code digests were recomputed. **Only `code.summarize_haa` differs from the record at `9f98bbb`; the other 19 match.**
- All eight exp_11 digests were recomputed. Only `launch_sh` moved in this fix cycle.
- The `eval`, `eval_launch`, `compare`, and `mirror_probe` closures match `9f98bbb`.
- Frozen existing files remain byte-identical, including all pre-existing model files, the specified datasets/evaluators/helpers, and `tools/exp06_launch.sh`.
- Importing the exp_11 registry preserves `BACKBONES_EXP06` and inherited class identities.
- The immutable exp_06/exp_09 oracle at `c6233e3` and exp_11 phase-1 oracle at `c491396` still produce byte-identical payloads and rendered summaries.
- The active exp_11 approval record remains identical to its all-null template.

The following are the exact fill values for a **content-identical merge of `94efbc2`**. They are not approval to fill or merge this revision. Required production changes must be reviewed and their digests recomputed before the actual merge.

For exp_06, replace only `code.summarize_haa`:

```text
old: 645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619
new: deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a
```

The existing exp_06 record SHA256 is:

```text
eb6e58e856bd915955fc008aef52ad6bd50142f8c52b125ccfa4e28ff6807f6a
```

Replacing only that digest, preserving the existing formatting and trailing newline, gives **3190 bytes** and this SHA256:

```text
80134832646672411740f779835fa7c06f8bb354b46aab558884503b39a1b666
```

The exact exp_11 fill manifest is:

```json
{
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
  },
  "code": {
    "finalize": "3eb15b59b672d97ac257054e66f8bc835abd27b85e24f9a417a5a1aaa9e25b9e",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "3d432c1de510e54ab1c48f19f264829bf53f907935563756b2a73f8d47cf9a80",
    "smoke": "62133ba07a35d42026c19fd1484b37c6c06ace0ebedc1cdd8484690df323450e",
    "summarize_haa": "deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a",
    "train": "d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "80134832646672411740f779835fa7c06f8bb354b46aab558884503b39a1b666",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

Serialized with sorted keys, two-space indentation, and one trailing newline, this is **1316 bytes**, SHA256:

```text
62d1c2286cd1e5ae400bd88eb5825e268d7b6c41f0238bfa47081a359abeea6a
```

No repository files were changed, and this review was not saved.
tokens used
77,112
**OpenAI Codex — exp_11 `orientation_cue_fairness`, round 2, close review 6**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not supplied in the visible invocation metadata; I cannot attest the plan’s requested `ultra` setting |
| Sandbox | Read-only repository; authorized test/cache writes confined to `/tmp` |
| Worktree / branch | `/home/yixunhu/codespace/xRIR_code_wt` / `exp11-cue` |
| HEAD | `94efbc2b75c64e8ab8927bad0173adb549ae540e` |
| Git tree | `2cd5787101ac92dafde00203ce65fddc07acd4db` |
| Last verification | 2026-09-27 15:05:20 UTC / 11:05:20 EDT |

**Verdict: request changes. Round 2 cannot be CLOSED, and `94efbc2` should not be merged as reviewed.**

I found one production blocker and one regression-coverage blocker.

1. **[P1] An orphaned FIFO sink can retain the publication lock indefinitely.**  
   [tools/exp11_launch.sh:208](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:208) opens an inheritable descriptor. The shared lifecycle passes it into both the background sink at [tools/exp06_launch.sh:138](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_launch.sh:138) and the `nohup setsid` child at line 141.

   I reproduced this using the real sourced `run_child` and real `flock`. A DEBUG hook replayed the registered TERM handler after the sink started, immediately before `CHILD_STARTED_AT` and before the training child launched:

   ```text
   launcher exit:                    143
   child.pid:                       absent
   sink fd 10:                      <arm-root>/.publish.lock
   sink /proc/.../wchan:             wait_for_partner
   independent flock -n:            1 (refused)
   after opening/closing own FIFO:  0 (acquired)
   ```

   The orphaned sink waits for a FIFO writer that will never arrive while retaining the lock. This creates an indefinitely locked arm without a running trainer. The 36-hour training timeout cannot help because the trainer never started.

   A second replay confirmed ordinary child inheritance: after the launcher exited 143, the detached Python helper still held fd 10 and acquisition failed until the child and sink ended.

   Retaining exclusion while an orphan trainer finishes can be defensible, but **the sink deadlock makes the current implementation unacceptable**. The header’s launcher-exit release promise is also false: release occurs when the last inherited descriptor closes. Fix descriptor propagation in exp_11-owned code, ensuring the sink cannot retain the fd while opening the FIFO, and test this exact boundary. Keep the frozen exp_06 file untouched.

2. **[P2] Still-required recovery and signal regressions were deleted with the obsolete lock tests.**  
   The deletions around [tests/test_exp11_shell.py:394](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:394) and [line 437](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:437) removed coverage for incomplete-orphan recovery, same-target idempotency, different-final refusal/`--replace-final`, TERM termination, INT termination, and `not_interrupted`.

   These behaviors remain requirements of A1. Restore their tests with flock-compatible fixtures and add the production-lifecycle regression above. The cycle removed 24 test functions and added eight; the net reduction of 16 does not mean all removed coverage was obsolete. The fix6 report’s lines 80–83 and 165–166 incorrectly describe that coverage as retained or obsolete.

The remaining requested checks passed:

| Check | Verified result |
|---|---|
| Acquisition count and scope | Exactly one call in full mode, at line 375, and recovery, at line 491. Full acquires before attempt creation/child launch; recovery acquires before canonical resolution. The parent retains its fd through promotion. |
| Lock-file handling | `exec {LOCK_FD}>>"$LOCK_FILE"` opens/creates without truncation; acquisition writes no bytes. `flock -n` refuses immediately with a clear message. |
| Held-lock refusal | With my own exact `flock lockfile sleep 60` helper, recovery and full test-root dry-runs returned 2 in approximately 21 ms and 20 ms. Neither announced child launch or promotion. |
| Refusal preservation | Attempt and completion bytes, `final`, directory entries, and lock-file bytes/inode/mtime/ctime remained unchanged. |
| Standalone holder release | After that helper ended naturally, acquisition and recovery succeeded. Standalone registered-handler replays also released their lock on exit. This does **not** establish production release because of blocker 1. |
| Old mechanism | No executable old lock machinery remains. Removed identifiers occur only in negative absence assertions and historical documentation. `OWNER`/`--owner-pid` remains for unrelated finalizer ownership. |
| Recovery behavior | Canonical aliases, published/completed-attempt preservation, orphan abort, idempotency, conflict refusal, and explicit replacement behavior passed selected tests or additional direct replays. |
| Ceiling/root restrictions | Bounded ceiling validation and the requirement for both `EXP11_TEST_ROOTS=1` and `--dry-run` remain intact and tested. |
| Signals and guards | Registered TERM/INT bodies returned 143/130 with no continuation. `not_interrupted` refused when set. Guards remain before real full finalization/promotion at lines 410/415 and recovery finalization/promotion at lines 501/506. |
| Ordinary dry-run | Announces `LOCK` without acquiring outside test roots. Test-root dry-runs acquire the real lock. |
| Other execution paths | I found no production full/recovery publication path that bypasses acquisition. Command substitutions and ordinary subprocesses do not discard the parent’s lock. Descriptor inheritance extends its lifetime instead. |

Independent verification produced **64 passing HEAD tests**: 54 shell tests, all three `tests/test_exp09_sim_eval_closures.py` tests, and seven pinned-oracle checks. Tests used `-p no:cacheprovider`, the worktree `PYTHONPATH`, disabled bytecode, empty CUDA visibility, one-thread numerical-library limits, and `/tmp` fixtures/caches. Shell syntax checks and `git diff --check 86b1a46 94efbc2` passed.

I replayed RED commit **`76d82f3`** from files extracted under `/tmp`, without checking out or modifying the worktree. Its acquisition-preservation test failed exactly as intended:

```text
assert b'' == b'not the launcher\n'
```

Recovery success and `PROMOTE` assertions had already passed. This confirms the RED evidence for the truncation fixed by `94efbc2`.

The Planner’s full suite remains pending; its log had reached approximately 19% at my last read. I did not signal or otherwise interfere with it. Its eventual result cannot resolve the demonstrated lifecycle defect.

The [fix6 report’s line-count table](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-27_09:27:14_coder_round2_fix6_opus_report.md:121) needs these corrections:

| Commit | Actual production changes | Actual test changes |
|---|---:|---:|
| `811de75` | — | +92 / −386 |
| `f80574a` | +44 / −235 | +7 / −3 |
| `04c703e` | — | +17 / −0 |
| `531db17` | +0 / −1 | — |
| `76d82f3` | — | +16 / −0 |
| `94efbc2` | +2 / −2 | — |
| Entire committed range | +43 / −235 | +124 / −381 |

`f80574a` changes **279 production lines**, despite shrinking the launcher by 191. The statement that no production commit approaches 200 changed lines is incorrect and does not satisfy the plan’s stated bound.

Committed record changes total **+509/−2**, as reported; including the modified working-tree report makes that **+596/−2**. The carried-over corrections are correct: `519b48f` is +149/−31 = 180 changed lines; `4f72ac2..519b48f` is +306/−27; and `c8b2ff0`’s remaining 333 insertions comprise 331 record lines plus two golden lines.

Closure hygiene at `94efbc2` passed:

- All 20 exp_06 code digests were recomputed. **Only `code.summarize_haa` differs from the record at `9f98bbb`; the other 19 match.**
- All eight exp_11 digests were recomputed. Only `launch_sh` moved in this fix cycle.
- The `eval`, `eval_launch`, `compare`, and `mirror_probe` closures match `9f98bbb`.
- Frozen existing files remain byte-identical, including all pre-existing model files, the specified datasets/evaluators/helpers, and `tools/exp06_launch.sh`.
- Importing the exp_11 registry preserves `BACKBONES_EXP06` and inherited class identities.
- The immutable exp_06/exp_09 oracle at `c6233e3` and exp_11 phase-1 oracle at `c491396` still produce byte-identical payloads and rendered summaries.
- The active exp_11 approval record remains identical to its all-null template.

The following are the exact fill values for a **content-identical merge of `94efbc2`**. They are not approval to fill or merge this revision. Required production changes must be reviewed and their digests recomputed before the actual merge.

For exp_06, replace only `code.summarize_haa`:

```text
old: 645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619
new: deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a
```

The existing exp_06 record SHA256 is:

```text
eb6e58e856bd915955fc008aef52ad6bd50142f8c52b125ccfa4e28ff6807f6a
```

Replacing only that digest, preserving the existing formatting and trailing newline, gives **3190 bytes** and this SHA256:

```text
80134832646672411740f779835fa7c06f8bb354b46aab558884503b39a1b666
```

The exact exp_11 fill manifest is:

```json
{
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
  },
  "code": {
    "finalize": "3eb15b59b672d97ac257054e66f8bc835abd27b85e24f9a417a5a1aaa9e25b9e",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "3d432c1de510e54ab1c48f19f264829bf53f907935563756b2a73f8d47cf9a80",
    "smoke": "62133ba07a35d42026c19fd1484b37c6c06ace0ebedc1cdd8484690df323450e",
    "summarize_haa": "deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a",
    "train": "d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "80134832646672411740f779835fa7c06f8bb354b46aab558884503b39a1b666",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

Serialized with sorted keys, two-space indentation, and one trailing newline, this is **1316 bytes**, SHA256:

```text
62d1c2286cd1e5ae400bd88eb5825e268d7b6c41f0238bfa47081a359abeea6a
```

No repository files were changed, and this review was not saved.

