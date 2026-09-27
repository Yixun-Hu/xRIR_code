**OpenAI Codex — exp_11 `orientation_cue_fairness`, ROUND 2, CLOSE review 10**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; authorized temporary writes confined to `/tmp` |
| Runtime | CUDA disabled; numerical libraries limited to one thread per worker |
| Worktree / branch | `/home/yixunhu/codespace/xRIR_code_wt` / `exp11-cue` |
| HEAD | `51e34b1416fd584436ead05fc81b400de99cc6ed` |
| Git tree | `c994a729a7e7563e55d37a79ccce4704917c7c86` |
| Verification completed | 2026-09-27 21:25:09 UTC / 17:25:09 EDT |

**Verdict: request changes. ROUND 2 cannot be CLOSED; `51e34b1` should not be merged.**

The registration/resolution race from close review 9 is fixed. The PID grammar still has a byte-handling defect, and the justification for leaving scans outside the registration lock is incomplete.

1. **[P1] The shell accepts NUL-containing PID records and withdraws unresolved protection.**

   At [tools/exp11_launch.sh:275](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:275), Bash command substitution discards NUL bytes. The length check at line 280 permits one missing byte, assuming it was a trailing LF. Consequently, digits containing exactly one NUL and no LF pass validation.

   Python’s bytes `fullmatch` correctly rejects those records. The shell prints `warning: command substitution: ignored null byte in input`, then accepts them.

   Replayed through the real sourced launcher with real publication locks:

   ```text
   child.pid = b'1607267\x00'    # exited, reaped helper
   Python registration_complete: false
   shell registration_complete: true
   shell pid_alive: false
   scan_arm: QUIET
   clear_launching: marker removed
   ```

   Leading and internal NULs also pass. Thus, the whole-file grammar and “invalid records never withdraw the marker” requirements remain unmet.

   **Required:** validate the actual bytes without lossy command substitution. Add leading, internal, and trailing NUL regressions, including a dead PID plus NUL that must preserve unresolved protection.

2. **[P1] A surviving trainer can disappear from both recovery and full-mode scans when its timeout wrapper and launcher die.**

   The new justification at [tools/exp11_launch.sh:389](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:389) assumes an in-flight registration is covered by either the marker or a complete **live** `child.pid`. However, full mode launches GNU `timeout` at [line 611](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:611), and the frozen lifecycle records that wrapper’s PID.

   At [line 317](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:317), a complete record makes the scan ignore `launching` even when the recorded wrapper is dead.

   Deterministic replay used the real sourced `run_child`, actual GNU `timeout`, real locks, and a stub importing real `register_trainer`:

   ```text
   launcher: 1612451
   timeout wrapper: 1612461
   trainer: 1612462

   trainer paused inside registration:
     .registration.lock held
     atomic temporary file open
     train.pid absent

   terminate our verified timeout through its pidfd handle
   launcher remains alive: publication lock still protects the arm

   terminate our launcher through its Popen handle
   publication holder exits

   remaining state:
     trainer alive
     launching present
     child.pid = b'1612461\n'   # complete, dead wrapper
     train.pid absent
     .registration.lock held

   hold_arm_lock full     + require_quiet_arm: exit 0, SCAN_PASSED
   hold_arm_lock finalize + require_quiet_arm: exit 0, SCAN_PASSED
   ```

   Releasing registration produces the live `train.pid`, after which scanning refuses. The interval before that is incorrectly treated as quiet.

   **Required:** preserve fail-closed protection until trainer registration or completion is accounted for. Merely probing the registration lock would address the demonstrated in-flight interval but would still leave a delayed trainer that has not acquired it yet.

   This replay demonstrated false quietness, not training epochs or scientific publication. Only owned helpers were terminated, through handles; all replay helpers subsequently exited.

The independent grammar replay covered 17 byte cases. Here, `P` denotes a live owned helper and `D` an exited, reaped helper.

| File bytes | Python complete | Shell complete | Shell alive | Arm quiet | Marker after clear |
|---|---:|---:|---:|---:|---|
| `P` | Yes | Yes | Yes | No | Removed |
| `P\n` | Yes | Yes | Yes | No | Removed |
| `1234567890\n` | Yes | Yes | No | Yes | Removed |
| `\nP\n` | No | No | No | No | Kept |
| `12\n34` | No | No | No | No | Kept |
| ` P\n` | No | No | No | No | Kept |
| `P\r` | No | No | No | No | Kept |
| `P\r\n` | No | No | No | No | Kept |
| Empty | No | No | No | No | Kept |
| `\n\n` | No | No | No | No | Kept |
| `P\n\n` | No | No | No | No | Kept |
| Eleven digits | No | No | No | No | Kept |
| `P\0` | **No** | **Yes** | Yes | No | **Removed** |
| `\0P` | **No** | **Yes** | Yes | No | **Removed** |
| NUL inside `P` | **No** | **Yes** | Yes | No | **Removed** |
| `P\0\n` | No | No | No | No | Kept |
| `D\0` | **No** | **Yes** | No | **Yes** | **Removed** |

The two shell readers now share their parser; their remaining disagreement with Python is byte fidelity. Completeness and liveness naturally differ for valid records naming dead processes.

`train.pid` now contains only `"<pid>\n"`. Repository-wide inspection found no production consumer requiring its former timestamp field.

The registration lock itself addresses the previous retirement race correctly:

| Verification | Result |
|---|---|
| Trainer checks | Directory existence, canonical parent, tombstone, and launch record are checked after acquiring the lock |
| Trainer write | Temporary file opened under the lock; `os.replace` publishes the complete PID atomically |
| Trainer-first schedule | Resolver refuses “registration in progress”; attempt and marker remain; no tombstone |
| After trainer registers | Scan refuses the live `train.pid`; resolution succeeds only after the helper exits |
| Resolver-first schedule | Trainer waits while resolver holds both locks; after retirement, trainer exits 3; no original or retired `train.pid` or temporary file |
| Default timeout against a held lock | Exit 3 after **60.002 seconds**; attempt still contains only `launching` |
| Full launcher still alive | Competing resolver refuses the publication lock |
| Root lock-file creation | No interference with the attempt walk or finalizer |

These schedules used file-existence barriers, never evaluated barrier contents.

The trainer takes only `.registration.lock`; it does not require `.publish.lock`. Full mode holds publication ownership from before its scan through the child run and publication. The frozen lifecycle also waits for the log sink, so wrapper death alone did not release that protection.

For an orphan with an incomplete `child.pid`, the marker and registration/resolution lock cover delayed registration correctly. The wrapper-loss case in blocker 2 invalidates the broader claim that all startup states are covered.

A stuck registration-lock holder causes an ordinary trainer startup to fail after approximately 60 seconds. Scans do not queue on that lock, and resolution acquires it non-blockingly. The resolver’s filesystem operations have no separate hard deadline. Also, [tools/exp11_train.py:238](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_train.py:238) uses `time.time()`; switching to `time.monotonic()` would make the elapsed-time bound robust against backward clock adjustments.

The arm-root `.registration.lock` is a regular file outside the attempts. It cannot match `scan_arm`’s `attempt_*` directory walk. Finalizer preflight reads `*/launch.pid` and `*/child.pid`, while full evidence is checked inside the selected attempt; neither treats this root lock file as run evidence.

**Independent selected verification: 178 passed.**

| Coverage | Passed |
|---|---:|
| `test_exp11_shell.py` | 106 |
| `test_exp11_train.py` | 49 |
| `test_exp11_lock_holder.py` | 7 |
| Sim-eval closures | 3 |
| Registry | 6 |
| Pinned-oracle checks | 7 |

Tests used `-p no:cacheprovider`, disabled bytecode/CUDA, appropriate `PYTHONPATH`, and `/tmp` temporary/cache paths. Runtime replays above are additional. Both launcher syntax checks and `git diff --check 55dc3dc 51e34b1` passed.

RED commit **`ea92e3b`** was exported under `/tmp`. Replaying these two semantic tests produced **2 failed**, as expected:

```text
tests/test_exp11_shell.py::test_the_two_readers_never_disagree
tests/test_exp11_shell.py::test_junk_in_child_pid_keeps_the_marker
```

The first returned `COMPLETE`/`GONE` for a two-record file. The second incorrectly returned recovery status 0 and emitted `PROMOTE`. These failures do not depend on the RED commit’s absent new API or its subsequently corrected substring assertion.

Git confirms the reported aggregate production **+144/−32** and tests **+224/−5**. The [fix10 report’s counts table](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-27_16:14:30_coder_round2_fix10_opus_report.md:108) needs three corrections:

| Commit | Actual production | Actual tests |
|---|---:|---:|
| `ea92e3b` | — | **+98**, not +90 |
| `64b0484` | +46/−18 | +5/−6 |
| `56e67b0` | — | **+55**, not +57 |
| `d0f9b95` | — | **+78**, not +73 |
| `d7c9e62` | +99/−15 | +5/−16 |

The statement at report line 117 should specify **production/test commits** under 200 changed lines. Record-only commits `5046ca4` and `51e34b1` exceed that count.

**`d7c9e62` is the last production commit.** `5046ca4` is bookkeeping, as the report correctly states.

I inspected the recorded full-suite result rather than rerunning it: **1 failed, 3782 passed, 55 skipped**. The sole failure is the expected pending exp_06 approvals refill:

```text
tests/test_exp06_profiles.py::
test_every_filled_record_digest_is_the_one_this_checkout_computes
```

Closure hygiene passes at both **`5046ca4` and `51e34b1`**:

- All 20 exp_06 closure keys were recomputed against the approvals at `9f98bbb`; only `summarize_haa` moved.
- Exactly five exp_11 keys moved against `55dc3dc`: `train`, `finalize`, `launch_sh`, `smoke`, and `summarize_haa`. The three HAA keys remain unchanged. All eight values appear below.
- Sim-eval closures `eval`, `eval_launch`, `compare`, and `mirror_probe` remain unchanged; all three dedicated tests pass.
- Thirty-two frozen files were verified byte-identical, including existing model files and exp_10 tools.
- Pinned oracles `c6233e3` and `c491396` retain byte-identical serialized payloads and rendered summaries.

`BACKBONES_EXP06` remains:

```text
f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169
```

The frozen lifecycle files remain byte-identical:

```text
tools/exp06_launch.sh
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479
```

The following are the **exact fill values for a content-identical merge of `51e34b1`**. They do not authorize that merge. Fixing the blockers requires recomputing affected values.

Replace only exp_06 `code.summarize_haa`:

```text
old: 645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619
new: a1c9b623dcfbebba4bfc39ba3ef8a81753871935dc7cffbe92f16a70c95e8708
```

The existing exp_06 approvals record is **3190 bytes**, SHA256:

```text
eb6e58e856bd915955fc008aef52ad6bd50142f8c52b125ccfa4e28ff6807f6a
```

Replacing only those 64 digest bytes, preserving every other byte, produces **3190 bytes**, SHA256:

```text
4d47ac6dbeb8a1f9256221013b2aca6f548b2701436c44e1e6068124640081b0
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
    "finalize": "8a416d9754f3fcef65fd566e01db5fab87ba7a2148d814546a8ddf056190a76e",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "9952138e58ee1ceee3086b0fd5a70f4c92a0edaa8f0fc478d7162100c9d4599f",
    "smoke": "bbdd9a5f907d2822ee8e72c90d1448fffc58ec3a598d49ea2ea779d21e68c1c1",
    "summarize_haa": "a1c9b623dcfbebba4bfc39ba3ef8a81753871935dc7cffbe92f16a70c95e8708",
    "train": "633d13605c584dd9e0f0a3d3fe854dbc88f4c8d7aada2c2b6254a2431a0e2519"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "4d47ac6dbeb8a1f9256221013b2aca6f548b2701436c44e1e6068124640081b0",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

With the displayed sorted keys, two-space indentation, and one trailing newline: **1316 bytes**, SHA256:

```text
f4a4656ffe5759a323e7f2cbcf7da1480b2b404e1d8a35bb5556672bb39ec721
```

The actual exp_11 approvals remain all-null. No repository files were changed, and no review was saved.
tokens used
93,480
**OpenAI Codex — exp_11 `orientation_cue_fairness`, ROUND 2, CLOSE review 10**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; authorized temporary writes confined to `/tmp` |
| Runtime | CUDA disabled; numerical libraries limited to one thread per worker |
| Worktree / branch | `/home/yixunhu/codespace/xRIR_code_wt` / `exp11-cue` |
| HEAD | `51e34b1416fd584436ead05fc81b400de99cc6ed` |
| Git tree | `c994a729a7e7563e55d37a79ccce4704917c7c86` |
| Verification completed | 2026-09-27 21:25:09 UTC / 17:25:09 EDT |

**Verdict: request changes. ROUND 2 cannot be CLOSED; `51e34b1` should not be merged.**

The registration/resolution race from close review 9 is fixed. The PID grammar still has a byte-handling defect, and the justification for leaving scans outside the registration lock is incomplete.

1. **[P1] The shell accepts NUL-containing PID records and withdraws unresolved protection.**

   At [tools/exp11_launch.sh:275](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:275), Bash command substitution discards NUL bytes. The length check at line 280 permits one missing byte, assuming it was a trailing LF. Consequently, digits containing exactly one NUL and no LF pass validation.

   Python’s bytes `fullmatch` correctly rejects those records. The shell prints `warning: command substitution: ignored null byte in input`, then accepts them.

   Replayed through the real sourced launcher with real publication locks:

   ```text
   child.pid = b'1607267\x00'    # exited, reaped helper
   Python registration_complete: false
   shell registration_complete: true
   shell pid_alive: false
   scan_arm: QUIET
   clear_launching: marker removed
   ```

   Leading and internal NULs also pass. Thus, the whole-file grammar and “invalid records never withdraw the marker” requirements remain unmet.

   **Required:** validate the actual bytes without lossy command substitution. Add leading, internal, and trailing NUL regressions, including a dead PID plus NUL that must preserve unresolved protection.

2. **[P1] A surviving trainer can disappear from both recovery and full-mode scans when its timeout wrapper and launcher die.**

   The new justification at [tools/exp11_launch.sh:389](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:389) assumes an in-flight registration is covered by either the marker or a complete **live** `child.pid`. However, full mode launches GNU `timeout` at [line 611](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:611), and the frozen lifecycle records that wrapper’s PID.

   At [line 317](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:317), a complete record makes the scan ignore `launching` even when the recorded wrapper is dead.

   Deterministic replay used the real sourced `run_child`, actual GNU `timeout`, real locks, and a stub importing real `register_trainer`:

   ```text
   launcher: 1612451
   timeout wrapper: 1612461
   trainer: 1612462

   trainer paused inside registration:
     .registration.lock held
     atomic temporary file open
     train.pid absent

   terminate our verified timeout through its pidfd handle
   launcher remains alive: publication lock still protects the arm

   terminate our launcher through its Popen handle
   publication holder exits

   remaining state:
     trainer alive
     launching present
     child.pid = b'1612461\n'   # complete, dead wrapper
     train.pid absent
     .registration.lock held

   hold_arm_lock full     + require_quiet_arm: exit 0, SCAN_PASSED
   hold_arm_lock finalize + require_quiet_arm: exit 0, SCAN_PASSED
   ```

   Releasing registration produces the live `train.pid`, after which scanning refuses. The interval before that is incorrectly treated as quiet.

   **Required:** preserve fail-closed protection until trainer registration or completion is accounted for. Merely probing the registration lock would address the demonstrated in-flight interval but would still leave a delayed trainer that has not acquired it yet.

   This replay demonstrated false quietness, not training epochs or scientific publication. Only owned helpers were terminated, through handles; all replay helpers subsequently exited.

The independent grammar replay covered 17 byte cases. Here, `P` denotes a live owned helper and `D` an exited, reaped helper.

| File bytes | Python complete | Shell complete | Shell alive | Arm quiet | Marker after clear |
|---|---:|---:|---:|---:|---|
| `P` | Yes | Yes | Yes | No | Removed |
| `P\n` | Yes | Yes | Yes | No | Removed |
| `1234567890\n` | Yes | Yes | No | Yes | Removed |
| `\nP\n` | No | No | No | No | Kept |
| `12\n34` | No | No | No | No | Kept |
| ` P\n` | No | No | No | No | Kept |
| `P\r` | No | No | No | No | Kept |
| `P\r\n` | No | No | No | No | Kept |
| Empty | No | No | No | No | Kept |
| `\n\n` | No | No | No | No | Kept |
| `P\n\n` | No | No | No | No | Kept |
| Eleven digits | No | No | No | No | Kept |
| `P\0` | **No** | **Yes** | Yes | No | **Removed** |
| `\0P` | **No** | **Yes** | Yes | No | **Removed** |
| NUL inside `P` | **No** | **Yes** | Yes | No | **Removed** |
| `P\0\n` | No | No | No | No | Kept |
| `D\0` | **No** | **Yes** | No | **Yes** | **Removed** |

The two shell readers now share their parser; their remaining disagreement with Python is byte fidelity. Completeness and liveness naturally differ for valid records naming dead processes.

`train.pid` now contains only `"<pid>\n"`. Repository-wide inspection found no production consumer requiring its former timestamp field.

The registration lock itself addresses the previous retirement race correctly:

| Verification | Result |
|---|---|
| Trainer checks | Directory existence, canonical parent, tombstone, and launch record are checked after acquiring the lock |
| Trainer write | Temporary file opened under the lock; `os.replace` publishes the complete PID atomically |
| Trainer-first schedule | Resolver refuses “registration in progress”; attempt and marker remain; no tombstone |
| After trainer registers | Scan refuses the live `train.pid`; resolution succeeds only after the helper exits |
| Resolver-first schedule | Trainer waits while resolver holds both locks; after retirement, trainer exits 3; no original or retired `train.pid` or temporary file |
| Default timeout against a held lock | Exit 3 after **60.002 seconds**; attempt still contains only `launching` |
| Full launcher still alive | Competing resolver refuses the publication lock |
| Root lock-file creation | No interference with the attempt walk or finalizer |

These schedules used file-existence barriers, never evaluated barrier contents.

The trainer takes only `.registration.lock`; it does not require `.publish.lock`. Full mode holds publication ownership from before its scan through the child run and publication. The frozen lifecycle also waits for the log sink, so wrapper death alone did not release that protection.

For an orphan with an incomplete `child.pid`, the marker and registration/resolution lock cover delayed registration correctly. The wrapper-loss case in blocker 2 invalidates the broader claim that all startup states are covered.

A stuck registration-lock holder causes an ordinary trainer startup to fail after approximately 60 seconds. Scans do not queue on that lock, and resolution acquires it non-blockingly. The resolver’s filesystem operations have no separate hard deadline. Also, [tools/exp11_train.py:238](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_train.py:238) uses `time.time()`; switching to `time.monotonic()` would make the elapsed-time bound robust against backward clock adjustments.

The arm-root `.registration.lock` is a regular file outside the attempts. It cannot match `scan_arm`’s `attempt_*` directory walk. Finalizer preflight reads `*/launch.pid` and `*/child.pid`, while full evidence is checked inside the selected attempt; neither treats this root lock file as run evidence.

**Independent selected verification: 178 passed.**

| Coverage | Passed |
|---|---:|
| `test_exp11_shell.py` | 106 |
| `test_exp11_train.py` | 49 |
| `test_exp11_lock_holder.py` | 7 |
| Sim-eval closures | 3 |
| Registry | 6 |
| Pinned-oracle checks | 7 |

Tests used `-p no:cacheprovider`, disabled bytecode/CUDA, appropriate `PYTHONPATH`, and `/tmp` temporary/cache paths. Runtime replays above are additional. Both launcher syntax checks and `git diff --check 55dc3dc 51e34b1` passed.

RED commit **`ea92e3b`** was exported under `/tmp`. Replaying these two semantic tests produced **2 failed**, as expected:

```text
tests/test_exp11_shell.py::test_the_two_readers_never_disagree
tests/test_exp11_shell.py::test_junk_in_child_pid_keeps_the_marker
```

The first returned `COMPLETE`/`GONE` for a two-record file. The second incorrectly returned recovery status 0 and emitted `PROMOTE`. These failures do not depend on the RED commit’s absent new API or its subsequently corrected substring assertion.

Git confirms the reported aggregate production **+144/−32** and tests **+224/−5**. The [fix10 report’s counts table](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-27_16:14:30_coder_round2_fix10_opus_report.md:108) needs three corrections:

| Commit | Actual production | Actual tests |
|---|---:|---:|
| `ea92e3b` | — | **+98**, not +90 |
| `64b0484` | +46/−18 | +5/−6 |
| `56e67b0` | — | **+55**, not +57 |
| `d0f9b95` | — | **+78**, not +73 |
| `d7c9e62` | +99/−15 | +5/−16 |

The statement at report line 117 should specify **production/test commits** under 200 changed lines. Record-only commits `5046ca4` and `51e34b1` exceed that count.

**`d7c9e62` is the last production commit.** `5046ca4` is bookkeeping, as the report correctly states.

I inspected the recorded full-suite result rather than rerunning it: **1 failed, 3782 passed, 55 skipped**. The sole failure is the expected pending exp_06 approvals refill:

```text
tests/test_exp06_profiles.py::
test_every_filled_record_digest_is_the_one_this_checkout_computes
```

Closure hygiene passes at both **`5046ca4` and `51e34b1`**:

- All 20 exp_06 closure keys were recomputed against the approvals at `9f98bbb`; only `summarize_haa` moved.
- Exactly five exp_11 keys moved against `55dc3dc`: `train`, `finalize`, `launch_sh`, `smoke`, and `summarize_haa`. The three HAA keys remain unchanged. All eight values appear below.
- Sim-eval closures `eval`, `eval_launch`, `compare`, and `mirror_probe` remain unchanged; all three dedicated tests pass.
- Thirty-two frozen files were verified byte-identical, including existing model files and exp_10 tools.
- Pinned oracles `c6233e3` and `c491396` retain byte-identical serialized payloads and rendered summaries.

`BACKBONES_EXP06` remains:

```text
f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169
```

The frozen lifecycle files remain byte-identical:

```text
tools/exp06_launch.sh
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479
```

The following are the **exact fill values for a content-identical merge of `51e34b1`**. They do not authorize that merge. Fixing the blockers requires recomputing affected values.

Replace only exp_06 `code.summarize_haa`:

```text
old: 645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619
new: a1c9b623dcfbebba4bfc39ba3ef8a81753871935dc7cffbe92f16a70c95e8708
```

The existing exp_06 approvals record is **3190 bytes**, SHA256:

```text
eb6e58e856bd915955fc008aef52ad6bd50142f8c52b125ccfa4e28ff6807f6a
```

Replacing only those 64 digest bytes, preserving every other byte, produces **3190 bytes**, SHA256:

```text
4d47ac6dbeb8a1f9256221013b2aca6f548b2701436c44e1e6068124640081b0
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
    "finalize": "8a416d9754f3fcef65fd566e01db5fab87ba7a2148d814546a8ddf056190a76e",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "9952138e58ee1ceee3086b0fd5a70f4c92a0edaa8f0fc478d7162100c9d4599f",
    "smoke": "bbdd9a5f907d2822ee8e72c90d1448fffc58ec3a598d49ea2ea779d21e68c1c1",
    "summarize_haa": "a1c9b623dcfbebba4bfc39ba3ef8a81753871935dc7cffbe92f16a70c95e8708",
    "train": "633d13605c584dd9e0f0a3d3fe854dbc88f4c8d7aada2c2b6254a2431a0e2519"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "4d47ac6dbeb8a1f9256221013b2aca6f548b2701436c44e1e6068124640081b0",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

With the displayed sorted keys, two-space indentation, and one trailing newline: **1316 bytes**, SHA256:

```text
f4a4656ffe5759a323e7f2cbcf7da1480b2b404e1d8a35bb5556672bb39ec721
```

The actual exp_11 approvals remain all-null. No repository files were changed, and no review was saved.

