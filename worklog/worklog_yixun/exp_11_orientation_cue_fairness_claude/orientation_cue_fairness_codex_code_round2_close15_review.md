**OpenAI Codex — exp_11 `orientation_cue_fairness`, ROUND 2, CLOSE review 15**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; approved execution for `/tmp` fixtures and owned helpers |
| Runtime | Non-root UID 1013; CUDA disabled; CPU thread limits ≤4 |
| Worktree / branch | `/home/yixunhu/codespace/xRIR_code_wt` / `exp11-cue` |
| HEAD | `2ca784a02e9152ba25738d31b6e0c40421850568` |
| Git tree | `1a2f0ad1b7c542d8759235a9158349c43242f6de` |
| Completed | 2026-09-28 05:09:47 UTC / 01:09:47 EDT |

**Verdict: request changes. ROUND 2 cannot be CLOSED; `2ca784a` should not be merged.**

The specific close-14 permission reproductions are repaired. Three blockers remain: two gaps in the claimed inspection-failure family closure, and a new trainer containment regression.

1. **[P1] A dangling PID record still permits retirement of a live trainer’s attempt.**

   [tools/exp11_pidrecord.py:58](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_pidrecord.py:58) handles followed-stat `FileNotFoundError` by checking the parent and returning `NORECORD`. It never checks whether the PID pathname itself remains a dangling symlink. The launcher delegates to this reader at [tools/exp11_launch.sh:388](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:388).

   During resolution, [tools/exp11_launch.sh:541](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:541) excludes the target from the subsequent `file_present` checks. Thus the new probe never corrects this answer.

   I reproduced the destructive outcome with the actual sourced `run_child`, FIFO/log sink, GNU `timeout`, real locks, and a stub importing real `register_trainer`:

   ```text
   launcher → timeout → registered trainer
   owned timeout ended through its pidfd
   owned launcher ended through its Popen handle
   trainer remained alive

   symlink-root full scan:     1, refused
   symlink-root finalize scan: 1, refused
   resolution:                1, refused

   move actual train.pid elsewhere
   replace train.pid with a symlink to that record
   remove the symlink target

   resolution:                0
   TOMBSTONE / ABORT / RESOLVED emitted
   attempt renamed
   trainer still alive, verified through its pidfd
   ```

   The launcher and wrapper records contained their actual, now-dead PIDs. The marker exceeded the production grace. Cleanup released the trainer through a file gate and reaped all owned descendants.

   **Required:** make the shared PID reader distinguish dangling links from confirmed absence, and add this live-trainer resolution regression. The audit-table justification for launcher line 381 is currently false.

2. **[P1] The shell probes still interpret unexplained leaf-inspection failures as absence.**

   At [tools/exp11_launch.sh:337](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:337), failed followed and unfollowed `stat` calls become “absent” whenever the parent checks succeed. [tools/exp11_launch.sh:360](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:360) has the same problem for publication links. Neither distinguishes `ENOENT` from an I/O or other inspection error.

   An accessible parent does not establish that a failed lookup of its child means the child is absent. The trainer’s new Python probe correctly preserves this distinction by accepting only `FileNotFoundError` as a candidate absence.

   With targeted `stat` fault injection—only the selected leaf failed; parent operations, lifecycle functions and locks were real—I reproduced:

   | Failed leaf inspection | Observed result |
   |---|---|
   | Existing `launching` | Scan returned **0**, quiet |
   | Existing `completion.json` | Resolution returned **0**, tombstoned and renamed the completed attempt |
   | Existing `final` | Conflict check returned **0** with `REPLACE_FINAL=0`; actual `promote` replaced it |
   | Existing `final` publishing the recovery target | `already_published` returned **1**; `abort_recovery` renamed the published attempt, leaving `final` dangling |

   These were controlled error injections, not a claim that the underlying disk experienced EIO. Separately, a real overlong filename produced `ENAMETOOLONG`, and both shell probes returned **1**, confirming that they do not distinguish the error classes.

   **Required:** use an errno-aware inspection result and claim absence only after confirmed missing-name results. Preserve unknown for every other inspection failure. Add behavioral regressions covering the affected guards.

3. **[P1] The new trainer identity check accepts an attempt symlink outside the locked arm.**

   [tools/exp11_train.py:357](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_train.py:357) changed the comparison from the canonical attempt’s parent to `realpath(directory.parent)`. This validates the alias’s parent while writes follow the attempt symlink elsewhere.

   With the production default `register_trainer(alias)` call, I verified:

   ```text
   arm_alias/attempt_X → arm_canonical/attempt_X

   arm_canonical/.registration.lock: held by parent process
   arm_canonical/attempt_X.resolved: present
   arm_canonical/attempt_X/launching: present

   register_trainer(alias): exit 0
   registration elapsed: 0.000312 seconds
   arm_canonical/attempt_X/train.pid: written
   canonical lock: still held
   canonical tombstone: still present
   ```

   The trainer acquired the alias parent’s lock and checked the alias parent’s tombstone. The canonical lock and tombstone were bypassed. The comparable `24fdd2b` implementation refused with exit **3**.

   This requires an attempt-level symlink; ordinary newly created attempts are unaffected.

   **Required:** restore canonical attempt-parent containment while retaining the three-state probes. Ensure the validated attempt, registration lock and tombstone refer to the same arm. Add a regression using the default production call.

I searched the actual launcher and trainer sources for every requested predicate and path operation. The audit’s call-site inventory is substantially accurate, but its claim that the entire family is closed is not.

| Audited area | Assessment |
|---|---|
| `list_attempts`: checked `realpath -e`, directory/access checks, checked `find` | Correct; failures refuse enumeration |
| Per-entry `stat -L` and mode classification | Correct; failed inspection is unknown |
| Marker, receipt, `train.exit`, publication and recovery guards | Routed correctly, but inherit blocker 2 |
| PID classification and delegated reader | Delegation has the dangling-link gap in blocker 1 |
| Resolution’s canonicalization and directory-entry checks | Failures refuse |
| `scan_arm` identity fallbacks | Identity comparisons; PID checks precede exclusion. The excluded-target gap is blocker 1 |
| `clear_launching` | Failed inspection retains the marker |
| Test barrier `-e`, data-root `-d`, marker timestamp lookup | Justifications hold; timestamp failure prevents retirement with the production grace |
| Inline attempt validation, preflight and lock acquisition | Errors refuse access |
| Trainer `stat`/`lstat`, file/directory classifications and `demand_known` guards | Correct three-state handling in exercised cases |
| Trainer arm identity | Regression in blocker 3 |
| Remaining `is_file` occurrences | Comments only; no executable decision |
| `os.path.join`, shell `read` flags and mutation-command flags | Filename construction or command options, not existence decisions |

The requested repaired cases passed:

| Case | Result |
|---|---|
| Symlink arm root with genuinely registered live trainer | Both scans and resolution refuse; attempt preserved |
| Dangling arm root | Both scans and resolution return unknown **2** |
| Missing name with accessible parent | Absent |
| Missing parent, dangling/looping/inaccessible symlink | New path probes return unknown |
| `probe_link_target`: absent/non-link, resolved link, unresolvable link | **1**, **0 + canonical target**, **2**, respectively |
| Marker through mode-000 directory | Scan refuses **2** |
| Receipt through mode-000 directory | Resolution refuses **2**; no retirement |
| `final` through inaccessible intermediate | All three publication guards refuse **2**; no promotion, replacement or abort |
| ELOOP trainer tombstone | Saved and diagnostic branches both exit **3**; no registration |
| Isolated inaccessible attempt entry | Deterministic unknown **2** |

For dangling-root function checks, the real root was locked before switching `ARM_ROOT` to the dangling alias. A lock cannot be acquired directly beneath that dangling alias.

The reader timeout works for an ordinary sleeping interpreter. At the default setting, `pid_record` returned unknown after **30.02 seconds**; an individual health probe failed after **30.02 seconds**. Both health probes execute sequentially, so an unhealthy scan can spend approximately 60 seconds probing before returning unknown.

There is a **nonblocking timeout limitation**: `timeout` sends TERM without a subsequent forced-kill deadline. A TERM-ignoring helper with a one-second configured timeout returned only after its natural four-second exit. Also, `EXP11_READER_TIMEOUT_S` is unvalidated and accepts values such as zero. A validated positive bound and bounded TERM-to-KILL escalation would support the report’s stronger “bounded reader” claim.

Temporary-directory cleanup passed ordinary exit, error exit and TERM tests, both before and after `hold_arm_lock` installed its combined EXIT trap. Leftover fixture files disappeared, and the publication lock was released. SIGKILL cannot execute an EXIT trap.

The deliberate **dangling `final` = unknown** behavior is sound. It refuses even with `REPLACE_FINAL=1`. Removing the confirmed dangling fixture link under the publication lock allowed recovery to proceed. The operator path is documented in the [fix-15 report at line 246](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_00:00:32_coder_round2_fix15_opus_report.md:246); it is absent from launcher help and refusal guidance.

Independent selected verification completed:

| Selection | Result |
|---|---:|
| Shell tests | **162 passed, 1 deselected** |
| PID reader and lock-holder tests | **82 passed** |
| Trainer tests | **54 passed** |
| Sim-eval closures | **3 passed** |
| Pinned oracles and registry identity | **8 passed** |
| Total distinct selected cases | **309 passed** |

Exactly **163 shell tests** collect. The deselected historical wrapper-loss test signals a saved PID directly; the supplemental lifecycle replay above used owned process handles instead. Permission tests ran non-root without skips. All runs used worktree `PYTHONPATH`, disabled bytecode/CUDA, thread limits, `/tmp` fixtures and `-p no:cacheprovider`. Shell syntax and `git diff --check` passed. I did not rerun the full suite.

The exact `a5ab870` shell RED replay reproduced both scans incorrectly returning **0** and resolution retiring a still-live, genuinely registered trainer through a symlink root.

Git confirms **ten commits**, comprising eight implementation/test commits and two bookkeeping commits. Production changes are **+267/−43**; tests are **+227/−6**. The largest implementation/test commit is `1b9eff5`, **154 changed lines**. The “every commit under 200” wording must be qualified: the bookkeeping commits exceed that size.

The saved full-suite log supports **1 failed, 3920 passed, 55 skipped**, with the sole stale exp_06 summarizer approval failure. Its [recorded HEAD at line 64](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_00:01_suite_full_cpu_7a2ff48.log:64) is `5cd31157e51d70a4830f876115e94094fc638bfb`, rather than an isolated `7a2ff48` checkout. Their production bytes are identical; the report should clarify this provenance.

Closure hygiene passes:

- All **28** exp_06/exp_11 code digests agree between `7a2ff48` and `2ca784a`.
- Against approvals at `9f98bbb`, only exp_06 `code.summarize_haa` differs; the other **19** match.
- Exactly five exp_11 keys moved: `train`, `finalize`, `launch_sh`, `smoke`, `summarize_haa`. The three HAA keys are unchanged.
- All four sim-eval closures remain unchanged; all ten actual simulation manifests verify.
- Frozen sources remain untouched, allowing the previously permitted exp_06 summarizer change.
- Pinned serialized analyses and rendered summaries remain byte-identical.
- Exp_11 approvals still equal the all-null template.

Unchanged identities:

```text
BACKBONES_EXP06
f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169

tools/exp06_launch.sh
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479
```

The following are the **exact fill values for a content-identical merge of `2ca784a`**. They do not authorize merging this blocked candidate. Recompute affected closures after repairs.

Replace only exp_06 `code.summarize_haa`:

```text
old:
645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619

new:
22a9858b067ed1fdbf7e2d6670d81e60d4d1be7fb903d802696e7a52d43c0489
```

Preserving every other byte of the **3190-byte** exp_06 approvals record produces SHA256:

```text
f5d83dd29287bfa72f8cba6f2bce6580cb38da165d2bfcad6aa6a7d7f2903008
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
    "finalize": "4894acefecc9e82d9f2b66785361c439e9463e5f339ee78ef4589b4fa82babb1",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "0a4141488f35ec626648afd7a16389893bccb58a8f95684df3f4749c37b45a2e",
    "smoke": "53246ae8f6c970bb9732de9f9a2c7174b8d90e7a6f002ba1ac74fa77c8c9c0cf",
    "summarize_haa": "22a9858b067ed1fdbf7e2d6670d81e60d4d1be7fb903d802696e7a52d43c0489",
    "train": "b7eef052dc62a8223e57abc248af4c5838c503246e5747649578750e9dec0029"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "f5d83dd29287bfa72f8cba6f2bce6580cb38da165d2bfcad6aa6a7d7f2903008",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

With sorted keys, two-space indentation and one trailing newline: **1316 bytes**, SHA256:

```text
94ded1951d0a038e28c8d7ae9a1fd8002aa9eb606af345ddcbf76e6d3bf32a34
```

No repository files were changed and no review was saved. Worktree status retains only the pre-existing untracked close-15 prompt.
tokens used
82,482
**OpenAI Codex — exp_11 `orientation_cue_fairness`, ROUND 2, CLOSE review 15**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; approved execution for `/tmp` fixtures and owned helpers |
| Runtime | Non-root UID 1013; CUDA disabled; CPU thread limits ≤4 |
| Worktree / branch | `/home/yixunhu/codespace/xRIR_code_wt` / `exp11-cue` |
| HEAD | `2ca784a02e9152ba25738d31b6e0c40421850568` |
| Git tree | `1a2f0ad1b7c542d8759235a9158349c43242f6de` |
| Completed | 2026-09-28 05:09:47 UTC / 01:09:47 EDT |

**Verdict: request changes. ROUND 2 cannot be CLOSED; `2ca784a` should not be merged.**

The specific close-14 permission reproductions are repaired. Three blockers remain: two gaps in the claimed inspection-failure family closure, and a new trainer containment regression.

1. **[P1] A dangling PID record still permits retirement of a live trainer’s attempt.**

   [tools/exp11_pidrecord.py:58](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_pidrecord.py:58) handles followed-stat `FileNotFoundError` by checking the parent and returning `NORECORD`. It never checks whether the PID pathname itself remains a dangling symlink. The launcher delegates to this reader at [tools/exp11_launch.sh:388](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:388).

   During resolution, [tools/exp11_launch.sh:541](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:541) excludes the target from the subsequent `file_present` checks. Thus the new probe never corrects this answer.

   I reproduced the destructive outcome with the actual sourced `run_child`, FIFO/log sink, GNU `timeout`, real locks, and a stub importing real `register_trainer`:

   ```text
   launcher → timeout → registered trainer
   owned timeout ended through its pidfd
   owned launcher ended through its Popen handle
   trainer remained alive

   symlink-root full scan:     1, refused
   symlink-root finalize scan: 1, refused
   resolution:                1, refused

   move actual train.pid elsewhere
   replace train.pid with a symlink to that record
   remove the symlink target

   resolution:                0
   TOMBSTONE / ABORT / RESOLVED emitted
   attempt renamed
   trainer still alive, verified through its pidfd
   ```

   The launcher and wrapper records contained their actual, now-dead PIDs. The marker exceeded the production grace. Cleanup released the trainer through a file gate and reaped all owned descendants.

   **Required:** make the shared PID reader distinguish dangling links from confirmed absence, and add this live-trainer resolution regression. The audit-table justification for launcher line 381 is currently false.

2. **[P1] The shell probes still interpret unexplained leaf-inspection failures as absence.**

   At [tools/exp11_launch.sh:337](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:337), failed followed and unfollowed `stat` calls become “absent” whenever the parent checks succeed. [tools/exp11_launch.sh:360](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:360) has the same problem for publication links. Neither distinguishes `ENOENT` from an I/O or other inspection error.

   An accessible parent does not establish that a failed lookup of its child means the child is absent. The trainer’s new Python probe correctly preserves this distinction by accepting only `FileNotFoundError` as a candidate absence.

   With targeted `stat` fault injection—only the selected leaf failed; parent operations, lifecycle functions and locks were real—I reproduced:

   | Failed leaf inspection | Observed result |
   |---|---|
   | Existing `launching` | Scan returned **0**, quiet |
   | Existing `completion.json` | Resolution returned **0**, tombstoned and renamed the completed attempt |
   | Existing `final` | Conflict check returned **0** with `REPLACE_FINAL=0`; actual `promote` replaced it |
   | Existing `final` publishing the recovery target | `already_published` returned **1**; `abort_recovery` renamed the published attempt, leaving `final` dangling |

   These were controlled error injections, not a claim that the underlying disk experienced EIO. Separately, a real overlong filename produced `ENAMETOOLONG`, and both shell probes returned **1**, confirming that they do not distinguish the error classes.

   **Required:** use an errno-aware inspection result and claim absence only after confirmed missing-name results. Preserve unknown for every other inspection failure. Add behavioral regressions covering the affected guards.

3. **[P1] The new trainer identity check accepts an attempt symlink outside the locked arm.**

   [tools/exp11_train.py:357](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_train.py:357) changed the comparison from the canonical attempt’s parent to `realpath(directory.parent)`. This validates the alias’s parent while writes follow the attempt symlink elsewhere.

   With the production default `register_trainer(alias)` call, I verified:

   ```text
   arm_alias/attempt_X → arm_canonical/attempt_X

   arm_canonical/.registration.lock: held by parent process
   arm_canonical/attempt_X.resolved: present
   arm_canonical/attempt_X/launching: present

   register_trainer(alias): exit 0
   registration elapsed: 0.000312 seconds
   arm_canonical/attempt_X/train.pid: written
   canonical lock: still held
   canonical tombstone: still present
   ```

   The trainer acquired the alias parent’s lock and checked the alias parent’s tombstone. The canonical lock and tombstone were bypassed. The comparable `24fdd2b` implementation refused with exit **3**.

   This requires an attempt-level symlink; ordinary newly created attempts are unaffected.

   **Required:** restore canonical attempt-parent containment while retaining the three-state probes. Ensure the validated attempt, registration lock and tombstone refer to the same arm. Add a regression using the default production call.

I searched the actual launcher and trainer sources for every requested predicate and path operation. The audit’s call-site inventory is substantially accurate, but its claim that the entire family is closed is not.

| Audited area | Assessment |
|---|---|
| `list_attempts`: checked `realpath -e`, directory/access checks, checked `find` | Correct; failures refuse enumeration |
| Per-entry `stat -L` and mode classification | Correct; failed inspection is unknown |
| Marker, receipt, `train.exit`, publication and recovery guards | Routed correctly, but inherit blocker 2 |
| PID classification and delegated reader | Delegation has the dangling-link gap in blocker 1 |
| Resolution’s canonicalization and directory-entry checks | Failures refuse |
| `scan_arm` identity fallbacks | Identity comparisons; PID checks precede exclusion. The excluded-target gap is blocker 1 |
| `clear_launching` | Failed inspection retains the marker |
| Test barrier `-e`, data-root `-d`, marker timestamp lookup | Justifications hold; timestamp failure prevents retirement with the production grace |
| Inline attempt validation, preflight and lock acquisition | Errors refuse access |
| Trainer `stat`/`lstat`, file/directory classifications and `demand_known` guards | Correct three-state handling in exercised cases |
| Trainer arm identity | Regression in blocker 3 |
| Remaining `is_file` occurrences | Comments only; no executable decision |
| `os.path.join`, shell `read` flags and mutation-command flags | Filename construction or command options, not existence decisions |

The requested repaired cases passed:

| Case | Result |
|---|---|
| Symlink arm root with genuinely registered live trainer | Both scans and resolution refuse; attempt preserved |
| Dangling arm root | Both scans and resolution return unknown **2** |
| Missing name with accessible parent | Absent |
| Missing parent, dangling/looping/inaccessible symlink | New path probes return unknown |
| `probe_link_target`: absent/non-link, resolved link, unresolvable link | **1**, **0 + canonical target**, **2**, respectively |
| Marker through mode-000 directory | Scan refuses **2** |
| Receipt through mode-000 directory | Resolution refuses **2**; no retirement |
| `final` through inaccessible intermediate | All three publication guards refuse **2**; no promotion, replacement or abort |
| ELOOP trainer tombstone | Saved and diagnostic branches both exit **3**; no registration |
| Isolated inaccessible attempt entry | Deterministic unknown **2** |

For dangling-root function checks, the real root was locked before switching `ARM_ROOT` to the dangling alias. A lock cannot be acquired directly beneath that dangling alias.

The reader timeout works for an ordinary sleeping interpreter. At the default setting, `pid_record` returned unknown after **30.02 seconds**; an individual health probe failed after **30.02 seconds**. Both health probes execute sequentially, so an unhealthy scan can spend approximately 60 seconds probing before returning unknown.

There is a **nonblocking timeout limitation**: `timeout` sends TERM without a subsequent forced-kill deadline. A TERM-ignoring helper with a one-second configured timeout returned only after its natural four-second exit. Also, `EXP11_READER_TIMEOUT_S` is unvalidated and accepts values such as zero. A validated positive bound and bounded TERM-to-KILL escalation would support the report’s stronger “bounded reader” claim.

Temporary-directory cleanup passed ordinary exit, error exit and TERM tests, both before and after `hold_arm_lock` installed its combined EXIT trap. Leftover fixture files disappeared, and the publication lock was released. SIGKILL cannot execute an EXIT trap.

The deliberate **dangling `final` = unknown** behavior is sound. It refuses even with `REPLACE_FINAL=1`. Removing the confirmed dangling fixture link under the publication lock allowed recovery to proceed. The operator path is documented in the [fix-15 report at line 246](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_00:00:32_coder_round2_fix15_opus_report.md:246); it is absent from launcher help and refusal guidance.

Independent selected verification completed:

| Selection | Result |
|---|---:|
| Shell tests | **162 passed, 1 deselected** |
| PID reader and lock-holder tests | **82 passed** |
| Trainer tests | **54 passed** |
| Sim-eval closures | **3 passed** |
| Pinned oracles and registry identity | **8 passed** |
| Total distinct selected cases | **309 passed** |

Exactly **163 shell tests** collect. The deselected historical wrapper-loss test signals a saved PID directly; the supplemental lifecycle replay above used owned process handles instead. Permission tests ran non-root without skips. All runs used worktree `PYTHONPATH`, disabled bytecode/CUDA, thread limits, `/tmp` fixtures and `-p no:cacheprovider`. Shell syntax and `git diff --check` passed. I did not rerun the full suite.

The exact `a5ab870` shell RED replay reproduced both scans incorrectly returning **0** and resolution retiring a still-live, genuinely registered trainer through a symlink root.

Git confirms **ten commits**, comprising eight implementation/test commits and two bookkeeping commits. Production changes are **+267/−43**; tests are **+227/−6**. The largest implementation/test commit is `1b9eff5`, **154 changed lines**. The “every commit under 200” wording must be qualified: the bookkeeping commits exceed that size.

The saved full-suite log supports **1 failed, 3920 passed, 55 skipped**, with the sole stale exp_06 summarizer approval failure. Its [recorded HEAD at line 64](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-28_00:01_suite_full_cpu_7a2ff48.log:64) is `5cd31157e51d70a4830f876115e94094fc638bfb`, rather than an isolated `7a2ff48` checkout. Their production bytes are identical; the report should clarify this provenance.

Closure hygiene passes:

- All **28** exp_06/exp_11 code digests agree between `7a2ff48` and `2ca784a`.
- Against approvals at `9f98bbb`, only exp_06 `code.summarize_haa` differs; the other **19** match.
- Exactly five exp_11 keys moved: `train`, `finalize`, `launch_sh`, `smoke`, `summarize_haa`. The three HAA keys are unchanged.
- All four sim-eval closures remain unchanged; all ten actual simulation manifests verify.
- Frozen sources remain untouched, allowing the previously permitted exp_06 summarizer change.
- Pinned serialized analyses and rendered summaries remain byte-identical.
- Exp_11 approvals still equal the all-null template.

Unchanged identities:

```text
BACKBONES_EXP06
f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169

tools/exp06_launch.sh
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479
```

The following are the **exact fill values for a content-identical merge of `2ca784a`**. They do not authorize merging this blocked candidate. Recompute affected closures after repairs.

Replace only exp_06 `code.summarize_haa`:

```text
old:
645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619

new:
22a9858b067ed1fdbf7e2d6670d81e60d4d1be7fb903d802696e7a52d43c0489
```

Preserving every other byte of the **3190-byte** exp_06 approvals record produces SHA256:

```text
f5d83dd29287bfa72f8cba6f2bce6580cb38da165d2bfcad6aa6a7d7f2903008
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
    "finalize": "4894acefecc9e82d9f2b66785361c439e9463e5f339ee78ef4589b4fa82babb1",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "0a4141488f35ec626648afd7a16389893bccb58a8f95684df3f4749c37b45a2e",
    "smoke": "53246ae8f6c970bb9732de9f9a2c7174b8d90e7a6f002ba1ac74fa77c8c9c0cf",
    "summarize_haa": "22a9858b067ed1fdbf7e2d6670d81e60d4d1be7fb903d802696e7a52d43c0489",
    "train": "b7eef052dc62a8223e57abc248af4c5838c503246e5747649578750e9dec0029"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "f5d83dd29287bfa72f8cba6f2bce6580cb38da165d2bfcad6aa6a7d7f2903008",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

With sorted keys, two-space indentation and one trailing newline: **1316 bytes**, SHA256:

```text
94ded1951d0a038e28c8d7ae9a1fd8002aa9eb606af345ddcbf76e6d3bf32a34
```

No repository files were changed and no review was saved. Worktree status retains only the pre-existing untracked close-15 prompt.

