**OpenAI Codex — exp_11 `orientation_cue_fairness`, round 2, close review 7**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata; I cannot attest `ultra` |
| Sandbox | Read-only worktree; helper and test writes confined to `/tmp` |
| Worktree / branch | `/home/yixunhu/codespace/xRIR_code_wt` / `exp11-cue` |
| HEAD | `e19bb442b208a3520627b58c99570d3f34a4d0c3` |
| Git tree | `2d077bfadb72b666b0a07f30b6feacf65862db1c` |
| Last verification | 2026-09-27 16:40:15 UTC / 12:40:15 EDT |

**Verdict: request changes. Round 2 cannot be CLOSED, and `e19bb44` should not be merged.**

The close6 inherited-lock defect is fixed, and the required behavior coverage is restored. However, the proposed division between publication locking and trainer liveness is **not sound as implemented**. Two paths allow recovery to promote another attempt of the same arm while a trainer remains running.

1. **[P1] Recovery’s live-pid check can become stale before lock acquisition.**  
   [tools/exp11_launch.sh:524](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:524) runs `preflight` before `hold_arm_lock`. Full mode has the same ordering at [line 406](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:406).

   A recovery can pass its scan, pause, and then acquire the lock after another launcher starts a trainer and dies. The new holder correctly releases on launcher death, but recovery never repeats the arm-wide scan. [tools/exp11_finalize.py:1002](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_finalize.py:1002) checks only the selected recovery directory, so an independently valid older attempt can pass while the newer trainer runs.

   **Required:** perform the relevant live-pid check under the acquired publication lock, before protected work, and add this interleaving regression.

2. **[P1] Launcher death between child creation and PID recording leaves an invisible trainer.**  
   [tools/exp11_launch.sh:437](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:437) invokes the frozen `run_child`, which forks the detached child at [tools/exp06_launch.sh:141](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_launch.sh:141) and writes `child.pid` at line 143.

   Death between those operations leaves a dead `launch.pid`, no `child.pid`, and a live trainer. The holder releases as designed. **Even a fresh preflight scan then finds no live launch**, allowing recovery of another attempt to proceed. Moving the scan under the lock alone does not fix this case.

   **Required:** make startup registration crash-safe in exp_11-owned code, so recovery cannot overlook a detached trainer during this interval. Test this exact boundary and preserve the frozen exp_06 file.

Both failures were reproduced using real sourced `run_child`, the real holder and `flock`, the actual live-pid functions, and actual canonicalization/conflict/guard/promotion functions:

| Isolated guard replay | Launcher exit | `child.pid` | Fresh live scan | Stub trainer | Recovery result |
|---|---:|---|---|---|---|
| TERM before PID recording | 143 | Absent | Empty | Alive | Exit 0; `final → attempt_old` |
| TERM after PID recording, with earlier preflight retained | 143 | Present | Finds live child | Alive | Exit 0; `final → attempt_old` |

These were publication/liveness replays, **not complete scientific finalizations**. Artifact validation was omitted, assuming an independently valid older attempt. The real target-only liveness check accepted that older directory; it supplies no missing arm-wide protection. Current all-null approvals still prevent an ordinary approved production launch.

The requested holder and lifecycle checks otherwise passed:

| Check | Verified result |
|---|---|
| TERM after sink starts, before child launch | Actual sourced `run_child`; exit 143; no `child.pid`; independent `flock -n` succeeded approximately 3 ms after exit. |
| Detached child, TERM | Exit 143; lock available approximately 4 ms after exit while the stub child remained alive. |
| Detached child, normal launcher exit | Exit 0; lock available while the stub child remained alive. |
| Ordinary completed lifecycle | Exit 0; explicit `drop_arm_lock` released while the launcher itself remained alive. |
| Descriptor tables | At the fork boundary, launcher and sink held neither the lock descriptor nor the verdict read end. The holder alone had `.publish.lock` on fd 3. The detached child inherited neither. |
| Inner `exec` | With it, the holder was a direct child and retained exclusion. Without it, the holder announced acquisition but the lock became free while the launcher remained alive. |
| Launcher SIGKILL | Killing only my helper’s `Popen` handle released the lock in approximately 0.49 seconds. |
| Verdict protocol | Acquisition, contention refusal, and startup failure were distinguished. A silent helper exercised the actual bounded read: failure after 30.05 seconds with “did not start,” not “another invocation.” |
| Read-end closure | Confirmed after acquisition and after timeout. |
| Release ownership check | The `/proc/<pid>/stat` ppid check and non-child bystander regression passed. Ordinary release terminates and waits for the holder. |
| Signal behavior | TERM/INT replays returned 143/130 without continuation. `not_interrupted` guards remain before real finalization and promotion. |
| Holder implementation | CLOEXEC flags, nonblocking flock, unchanged lock-file bytes, parent lease, SIGTERM exit, and seven unit tests passed. |

**Independent selected verification: 105 passed.** This comprises 65 shell tests, seven holder tests, 17 exp_11 profile tests, six registry tests, all three sim-eval closure tests, and seven pinned-oracle checks. Runs used `-p no:cacheprovider`, worktree `PYTHONPATH`, disabled bytecode and CUDA, and one numerical-library thread per worker, within the four-thread limit.

Restored coverage includes incomplete-orphan abort, idempotency, different-final refusal and `--replace-final`, TERM 143, INT 130, and `not_interrupted`. Both new lifecycle cases pass; their fixture asserts that the sink and child spawn lines remain present verbatim in frozen `tools/exp06_launch.sh`. I additionally replayed the actual sourced function as described above.

RED commit **`4da8b55`** was replayed from `/tmp`. Its no-verdict test failed as intended: a directory at `.publish.lock` caused `Errno 21`, yet that revision incorrectly reported “held by another invocation.”

One test needs a hygiene correction: [tests/test_exp11_lock_holder.py:122](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_lock_holder.py:122) creates `tools/.cloexec_probe.lock` inside the repository and later unlinks it. Use `tmp_path`. I ran these tests from a byte-identical `/tmp` snapshot to satisfy the review’s write restrictions.

The accounting is mostly corrected:

- `f80574a`: **+44/−235 = 279 production lines**, now explicitly recorded as the exception.
- `94efbc2..e19bb44`: **+120/−23 production**, **+357/−3 tests**, correctly reported.
- Three entries in the [fix7 report](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-27_11:27:01_coder_round2_fix7_opus_report.md:121) still need correction:

| Commit | Actual | Report |
|---|---:|---:|
| `4da8b55` | Tests +16 | +17 |
| `091b3b9` | Tests +20 | +21 |
| `63c1bed` | Production +10/−3 | +9/−3 |

Shell syntax and production/test `git diff --check` pass. The full current range fails whitespace checking solely in committed suite logs. The recorded full-suite result is **1 failed, 3701 passed, 55 skipped**, with the expected unfilled exp_06 `summarize_haa` mismatch; I inspected that log rather than rerunning the full suite.

Closure hygiene passed at both `63c1bed` and `e19bb44`; their production/test bytes are identical:

- All 20 exp_06 keys were recomputed. Only `summarize_haa` differs from the `9f98bbb` record; the other 19 match.
- Exactly five exp_11 keys moved against `94efbc2`: `launch_sh`, `train`, `finalize`, `smoke`, `summarize_haa`.
- More precisely, `exp11_profiles.py` belongs to the **four Python closures** above. `launch_sh` moves through its changed shell bytes and explicit addition of the holder to its three-file closure.
- Sim-eval closures `eval`, `eval_launch`, `compare`, and `mirror_probe` remain unchanged.
- Frozen files, all pre-existing model files, and `BACKBONES_EXP06` remain unchanged.
- `tools/exp06_launch.sh` remains byte-identical; SHA256 is `d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09`.
- Pinned historical oracle `c6233e3` and phase-1 oracle `c491396` still produce byte-identical payloads and rendered summaries.
- Active exp_11 approvals remain byte-identical to the all-null template.

The following are exact fill values for a **content-identical merge of `e19bb44`**. They are reference values, not approval to merge or fill. Required production fixes must be reviewed and these digests recomputed.

For exp_06, replace only `code.summarize_haa`:

```text
old: 645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619
new: 2443b9312798e286f1dfa4dabd721040689bdbdf914c26a9112e9c9cb2d96bcc
```

The existing exp_06 record is 3190 bytes, SHA256:

```text
eb6e58e856bd915955fc008aef52ad6bd50142f8c52b125ccfa4e28ff6807f6a
```

Replacing only those digest bytes, preserving formatting and the trailing newline, produces 3190 bytes, SHA256:

```text
fecb3a1d338668739c5d04327326b3b2fe89cbd93fba2d8eff5ddf42728cbd55
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
    "finalize": "08c4d41a25bdc731f23c01b7dd02e904055586908ed541aeccbf2fa54e406717",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "6dc160f805af0991371e754ffd400ac4a2235b310db76b1d7c6a3ef426f0868d",
    "smoke": "7dea6b3661f639c5ed86755b9119b7fa5310729fa54b8e84be338efc91292938",
    "summarize_haa": "2443b9312798e286f1dfa4dabd721040689bdbdf914c26a9112e9c9cb2d96bcc",
    "train": "fb955253b97b88be807c6ec98349e4eb3597fe85fbeba2ec5397d30d30f167ad"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "fecb3a1d338668739c5d04327326b3b2fe89cbd93fba2d8eff5ddf42728cbd55",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

Serialized with sorted keys, two-space indentation, and one trailing newline: **1316 bytes**, SHA256:

```text
cabbcf440ac19ad0e61acdeb60718cd70e32f3fc085e16685592f5d15e1d374f
```

No repository files were changed. No unrelated processes were signalled. This review was not saved.
tokens used
71,425
**OpenAI Codex — exp_11 `orientation_cue_fairness`, round 2, close review 7**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata; I cannot attest `ultra` |
| Sandbox | Read-only worktree; helper and test writes confined to `/tmp` |
| Worktree / branch | `/home/yixunhu/codespace/xRIR_code_wt` / `exp11-cue` |
| HEAD | `e19bb442b208a3520627b58c99570d3f34a4d0c3` |
| Git tree | `2d077bfadb72b666b0a07f30b6feacf65862db1c` |
| Last verification | 2026-09-27 16:40:15 UTC / 12:40:15 EDT |

**Verdict: request changes. Round 2 cannot be CLOSED, and `e19bb44` should not be merged.**

The close6 inherited-lock defect is fixed, and the required behavior coverage is restored. However, the proposed division between publication locking and trainer liveness is **not sound as implemented**. Two paths allow recovery to promote another attempt of the same arm while a trainer remains running.

1. **[P1] Recovery’s live-pid check can become stale before lock acquisition.**  
   [tools/exp11_launch.sh:524](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:524) runs `preflight` before `hold_arm_lock`. Full mode has the same ordering at [line 406](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:406).

   A recovery can pass its scan, pause, and then acquire the lock after another launcher starts a trainer and dies. The new holder correctly releases on launcher death, but recovery never repeats the arm-wide scan. [tools/exp11_finalize.py:1002](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_finalize.py:1002) checks only the selected recovery directory, so an independently valid older attempt can pass while the newer trainer runs.

   **Required:** perform the relevant live-pid check under the acquired publication lock, before protected work, and add this interleaving regression.

2. **[P1] Launcher death between child creation and PID recording leaves an invisible trainer.**  
   [tools/exp11_launch.sh:437](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:437) invokes the frozen `run_child`, which forks the detached child at [tools/exp06_launch.sh:141](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_launch.sh:141) and writes `child.pid` at line 143.

   Death between those operations leaves a dead `launch.pid`, no `child.pid`, and a live trainer. The holder releases as designed. **Even a fresh preflight scan then finds no live launch**, allowing recovery of another attempt to proceed. Moving the scan under the lock alone does not fix this case.

   **Required:** make startup registration crash-safe in exp_11-owned code, so recovery cannot overlook a detached trainer during this interval. Test this exact boundary and preserve the frozen exp_06 file.

Both failures were reproduced using real sourced `run_child`, the real holder and `flock`, the actual live-pid functions, and actual canonicalization/conflict/guard/promotion functions:

| Isolated guard replay | Launcher exit | `child.pid` | Fresh live scan | Stub trainer | Recovery result |
|---|---:|---|---|---|---|
| TERM before PID recording | 143 | Absent | Empty | Alive | Exit 0; `final → attempt_old` |
| TERM after PID recording, with earlier preflight retained | 143 | Present | Finds live child | Alive | Exit 0; `final → attempt_old` |

These were publication/liveness replays, **not complete scientific finalizations**. Artifact validation was omitted, assuming an independently valid older attempt. The real target-only liveness check accepted that older directory; it supplies no missing arm-wide protection. Current all-null approvals still prevent an ordinary approved production launch.

The requested holder and lifecycle checks otherwise passed:

| Check | Verified result |
|---|---|
| TERM after sink starts, before child launch | Actual sourced `run_child`; exit 143; no `child.pid`; independent `flock -n` succeeded approximately 3 ms after exit. |
| Detached child, TERM | Exit 143; lock available approximately 4 ms after exit while the stub child remained alive. |
| Detached child, normal launcher exit | Exit 0; lock available while the stub child remained alive. |
| Ordinary completed lifecycle | Exit 0; explicit `drop_arm_lock` released while the launcher itself remained alive. |
| Descriptor tables | At the fork boundary, launcher and sink held neither the lock descriptor nor the verdict read end. The holder alone had `.publish.lock` on fd 3. The detached child inherited neither. |
| Inner `exec` | With it, the holder was a direct child and retained exclusion. Without it, the holder announced acquisition but the lock became free while the launcher remained alive. |
| Launcher SIGKILL | Killing only my helper’s `Popen` handle released the lock in approximately 0.49 seconds. |
| Verdict protocol | Acquisition, contention refusal, and startup failure were distinguished. A silent helper exercised the actual bounded read: failure after 30.05 seconds with “did not start,” not “another invocation.” |
| Read-end closure | Confirmed after acquisition and after timeout. |
| Release ownership check | The `/proc/<pid>/stat` ppid check and non-child bystander regression passed. Ordinary release terminates and waits for the holder. |
| Signal behavior | TERM/INT replays returned 143/130 without continuation. `not_interrupted` guards remain before real finalization and promotion. |
| Holder implementation | CLOEXEC flags, nonblocking flock, unchanged lock-file bytes, parent lease, SIGTERM exit, and seven unit tests passed. |

**Independent selected verification: 105 passed.** This comprises 65 shell tests, seven holder tests, 17 exp_11 profile tests, six registry tests, all three sim-eval closure tests, and seven pinned-oracle checks. Runs used `-p no:cacheprovider`, worktree `PYTHONPATH`, disabled bytecode and CUDA, and one numerical-library thread per worker, within the four-thread limit.

Restored coverage includes incomplete-orphan abort, idempotency, different-final refusal and `--replace-final`, TERM 143, INT 130, and `not_interrupted`. Both new lifecycle cases pass; their fixture asserts that the sink and child spawn lines remain present verbatim in frozen `tools/exp06_launch.sh`. I additionally replayed the actual sourced function as described above.

RED commit **`4da8b55`** was replayed from `/tmp`. Its no-verdict test failed as intended: a directory at `.publish.lock` caused `Errno 21`, yet that revision incorrectly reported “held by another invocation.”

One test needs a hygiene correction: [tests/test_exp11_lock_holder.py:122](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_lock_holder.py:122) creates `tools/.cloexec_probe.lock` inside the repository and later unlinks it. Use `tmp_path`. I ran these tests from a byte-identical `/tmp` snapshot to satisfy the review’s write restrictions.

The accounting is mostly corrected:

- `f80574a`: **+44/−235 = 279 production lines**, now explicitly recorded as the exception.
- `94efbc2..e19bb44`: **+120/−23 production**, **+357/−3 tests**, correctly reported.
- Three entries in the [fix7 report](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-27_11:27:01_coder_round2_fix7_opus_report.md:121) still need correction:

| Commit | Actual | Report |
|---|---:|---:|
| `4da8b55` | Tests +16 | +17 |
| `091b3b9` | Tests +20 | +21 |
| `63c1bed` | Production +10/−3 | +9/−3 |

Shell syntax and production/test `git diff --check` pass. The full current range fails whitespace checking solely in committed suite logs. The recorded full-suite result is **1 failed, 3701 passed, 55 skipped**, with the expected unfilled exp_06 `summarize_haa` mismatch; I inspected that log rather than rerunning the full suite.

Closure hygiene passed at both `63c1bed` and `e19bb44`; their production/test bytes are identical:

- All 20 exp_06 keys were recomputed. Only `summarize_haa` differs from the `9f98bbb` record; the other 19 match.
- Exactly five exp_11 keys moved against `94efbc2`: `launch_sh`, `train`, `finalize`, `smoke`, `summarize_haa`.
- More precisely, `exp11_profiles.py` belongs to the **four Python closures** above. `launch_sh` moves through its changed shell bytes and explicit addition of the holder to its three-file closure.
- Sim-eval closures `eval`, `eval_launch`, `compare`, and `mirror_probe` remain unchanged.
- Frozen files, all pre-existing model files, and `BACKBONES_EXP06` remain unchanged.
- `tools/exp06_launch.sh` remains byte-identical; SHA256 is `d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09`.
- Pinned historical oracle `c6233e3` and phase-1 oracle `c491396` still produce byte-identical payloads and rendered summaries.
- Active exp_11 approvals remain byte-identical to the all-null template.

The following are exact fill values for a **content-identical merge of `e19bb44`**. They are reference values, not approval to merge or fill. Required production fixes must be reviewed and these digests recomputed.

For exp_06, replace only `code.summarize_haa`:

```text
old: 645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619
new: 2443b9312798e286f1dfa4dabd721040689bdbdf914c26a9112e9c9cb2d96bcc
```

The existing exp_06 record is 3190 bytes, SHA256:

```text
eb6e58e856bd915955fc008aef52ad6bd50142f8c52b125ccfa4e28ff6807f6a
```

Replacing only those digest bytes, preserving formatting and the trailing newline, produces 3190 bytes, SHA256:

```text
fecb3a1d338668739c5d04327326b3b2fe89cbd93fba2d8eff5ddf42728cbd55
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
    "finalize": "08c4d41a25bdc731f23c01b7dd02e904055586908ed541aeccbf2fa54e406717",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "6dc160f805af0991371e754ffd400ac4a2235b310db76b1d7c6a3ef426f0868d",
    "smoke": "7dea6b3661f639c5ed86755b9119b7fa5310729fa54b8e84be338efc91292938",
    "summarize_haa": "2443b9312798e286f1dfa4dabd721040689bdbdf914c26a9112e9c9cb2d96bcc",
    "train": "fb955253b97b88be807c6ec98349e4eb3597fe85fbeba2ec5397d30d30f167ad"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "fecb3a1d338668739c5d04327326b3b2fe89cbd93fba2d8eff5ddf42728cbd55",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

Serialized with sorted keys, two-space indentation, and one trailing newline: **1316 bytes**, SHA256:

```text
cabbcf440ac19ad0e61acdeb60718cd70e32f3fc085e16685592f5d15e1d374f
```

No repository files were changed. No unrelated processes were signalled. This review was not saved.

