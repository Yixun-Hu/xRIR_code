**OpenAI Codex — exp_11 `orientation_cue_fairness`, ROUND 2, CLOSE review 12**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; approved escalation for `/tmp` test fixtures only |
| Runtime | CUDA disabled; one CPU thread per worker, at most four workers |
| Worktree / branch | `/home/yixunhu/codespace/xRIR_code_wt` / `exp11-cue` |
| HEAD | `0885c41bd0a651f402e00edc20c5b9a15f0a90be` |
| Git tree | `723cde57373a20170646877fee2c11fab1ade6a4` |
| Verification completed | 2026-09-28 01:06:14 UTC / 2026-09-27 21:06:14 EDT |

**Verdict: request changes. ROUND 2 cannot be CLOSED; `0885c41` should not be merged.**

The execution/import failure from close review 11 is repaired in the tested paths. Two remaining paths still let resolution retire a registered live trainer.

1. **[P1] An unreadable existing PID record becomes definite “not alive.”**

   [tools/exp11_pidrecord.py:42](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_pidrecord.py:42) catches every `OSError`, including `PermissionError`, and returns `None`. The CLI consequently emits `norecord`, exit 0. The shell interprets that as a definite absence of a record.

   Independently reproduced with the actual sourced `run_child`, GNU `timeout`, real publication and registration locks, and a stub importing real `register_trainer`:

   ```text
   owned timeout and launcher stopped
   real register_trainer succeeds; train.pid names the live trainer
   train.pid made unreadable with chmod 000

   check_pid_reader -> 0
   pid_record       -> 1
   pid_alive        -> 1
   resolve_unregistered -> 0

   tombstone created
   attempt renamed
   launching marker removed
   registered trainer still alive
   ```

   A second independent replay applied the permission change after acquiring the publication lock and produced the same result. The trainer subsequently passed its continuation gate.

   **Required:** distinguish confirmed absence or malformed readable bytes from inability to inspect an existing record. Permission and I/O failures must produce unknown and prevent resolution. The shell’s [`-f` guard](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:302) also needs care where inaccessible metadata would otherwise look absent.

   The [“unreadable file” test](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_pidrecord.py:122) tests only a missing path and a directory; it never creates a permission-denied regular file.

2. **[P1] Bash normalizes malformed reader output into a valid verdict.**

   [tools/exp11_launch.sh:303](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:303) captures stdout through command substitution. Bash removes trailing newlines and NUL bytes before the subsequent validation sees the answer.

   Wrappers that delegate normally except for `train.pid`, where they emit either of these byte sequences with exit 0, bypass the protocol:

   ```text
   norecord\n\n
   norecord\0\n
   ```

   Both pass the two health probes. Both become plain `norecord` in Bash. With the real lifecycle and registered live trainer described above, both yielded:

   ```text
   pid_record -> 1
   pid_alive  -> 1
   resolve_unregistered -> 0
   tombstone=True, renamed=True, trainer_alive=True
   ```

   The NUL case prints Bash’s ignored-null-byte warning but proceeds with retirement.

   **Required:** validate the raw output before any lossy shell conversion. Reject additional blank lines and NUL bytes as unknown. Apply equivalent validation to the [health-check captures](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:349), which have the same normalization issue.

The independent lifecycle replay confirms that the intended fix works for these failures:

| Reader condition | Resolution | Both mode scans | CLI resolution |
|---|---:|---:|---:|
| Missing interpreter after publication lock | 2 | 2 | 2 |
| Real module ImportError through the temporary-tree wrapper | 2 | 2 | 2 |
| Failure only on `train.pid` | 2 | 2 | 2 |
| Always `record 1` | 2 | 2 | 2 |
| Always `norecord` | 2 | 2 | 2 |
| `record not-a-pid` only on `train.pid` | 2 | 2 | 2 |
| Empty stdout, wrong first word, nonblank second line, or nonzero exit with a verdict | 2 | 2 | 2 |

Every refusal retained the attempt, marker, and `train.pid`, without a tombstone or rename. Ordinary CLI recovery also returned 2 without promotion for the first six conditions. The lock holder continued using the real interpreter.

Direct state checks produced:

| Input/result | `pid_record` | `pid_alive` | `registration_complete` |
|---|---:|---:|---:|
| Valid live PID | 0 | 0 | 0 |
| Malformed readable bytes | 1 | 1 | 1 |
| Reader execution failure | 2 | 2 | 2 |

`scan_arm` sets `SCAN_REASON` and returns 2 for an encountered unknown. `require_quiet_arm` and `resolve_unregistered` preserve it; `clear_launching` retains the marker.

There is one qualification to “ANY unknown from ANY attempt”: `scan_arm` stops at the first live or unresolved attempt. An unknown in a later attempt may therefore never be inspected, and a sourced caller receives 1. I reproduced this ordering. It remains a refusal and permits no retirement or publication; CLI callers normalize either failure to exit 2.

The health check uses the same `$PYTHON -m tools.exp11_pidrecord` invocation for `1\n` and an empty file, before each production scan/resolution decision. Both constant-answer readers are detected. It does **not** establish that the real files are readable.

| Real-file condition | Observed behavior |
|---|---|
| Unreadable regular file | Healthy probes; definite `norecord` — blocker 1 |
| Directory | Healthy probes; shell returns 1 through its nonregular-file guard |
| FIFO | Healthy probes; shell returns 1 without opening it; direct Python CLI blocks |
| 1 GiB sparse file | Healthy probes; bounded `read(64)` produces definite `norecord` |

The directory/FIFO behavior is an explicit classification in the current implementation, rather than health-check coverage. The huge-file case is bounded and correctly rejects invalid file bytes. The original byte table remains definite: none of its rows returns unknown.

One nonblocking wording defect remains at [tools/exp11_launch.sh:541](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:541): the tombstone asserts “no train.exit,” but resolution does not enforce that condition. A stale marker with dead recorded PIDs and an existing `train.exit` resolved successfully in my replay, retaining `train.exit` in the retired directory while writing that assertion. The reason and `RESOLVED` message should describe only conditions actually checked.

Independent selected pytest verification produced **266 unique passes**:

| Selection | Passed |
|---|---:|
| `test_exp11_pidrecord.py` | 72 |
| `test_exp11_shell.py` | 128 |
| `test_exp11_train.py` | 50 |
| `test_exp11_lock_holder.py` | 7 |
| `test_exp09_sim_eval_closures.py` | 3 |
| Registry preservation | 1 |
| Pinned-oracle checks | 5 |

All runs used `-p no:cacheprovider`, worktree `PYTHONPATH`, disabled bytecode/CUDA, and temporary/cache directories under `/tmp`. Shell syntax and `git diff --check` passed.

**RED evidence:** replaying the exact `a954fe0` archive under `/tmp`, the gibberish-on-trainer regression failed as expected:

```text
expected resolve returncode 2
actual returncode 0
publication lock acquired
RESOLVED emitted
tombstone and _ABORTED_unregistered directory present
```

That case passes at HEAD. The new broken-reader fixtures themselves write a live sleep PID directly; the independent replays above supply the stronger real-registration and real-lifecycle evidence.

Git confirms **13 commits**, production **+151/−32**, tests **+233/−18**, and last production commit `2fc5300`. The recorded full-suite log supports **1 failed, 3878 passed, 55 skipped**, with only the documented stale exp_06 `summarize_haa` approval. I inspected that log; I did not rerun the full repository suite.

The two earlier suite cleanups are corroborated by the [scoped Coder transcript](/home/yixunhu/.claude/projects/-home-yixunhu-codespace-xRIR-code/7773656a-f89c-41f7-adc3-63d05a17d044/subagents/agent-aff672cc49719b2fb.jsonl:11557). It records the suite launches, matches their open logs, command lines and worktree to PIDs **1952017** and **1961431**, then records `kill -TERM 1952017 1961431` and successful termination. Only those two identified Coder-owned suite PIDs were targeted; no process-group or process-name kill was used.

Closure hygiene passes:

- All **28** exp_06/exp_11 code digests match between `2fc5300` and `0885c41`.
- Against exp_06 approvals at `9f98bbb`, only `code.summarize_haa` changes; the other **19** keys match.
- Exactly five exp_11 keys move versus `fdb74fc`: `train`, `finalize`, `launch_sh`, `smoke`, `summarize_haa`. The three HAA keys remain unchanged.
- All four sim-eval closures remain unchanged; all ten actual manifests pass verification.
- Frozen files are unchanged. The permitted exp_06 summarizer is excluded from that frozen-file assertion.
- Pinned serialized payloads and rendered-summary oracles remain byte-identical.

`BACKBONES_EXP06` remains:

```text
f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169
```

Frozen lifecycle SHA256s remain:

```text
tools/exp06_launch.sh
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479
```

The following are the **exact fill values for a content-identical merge of `0885c41`**. They do not constitute merge approval; fixing the blockers requires recomputing affected closures.

Replace only exp_06 `code.summarize_haa`:

```text
old:
645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619

new:
650a5a6d97cbc8a3c4c54d647a3c75a993d9f79eb22b2c9b10034d357950a660
```

Preserving every other byte of the **3190-byte** exp_06 approvals record produces SHA256:

```text
f3c79c383ebf13bac86eb0157bb5154263eeec9f6917c1fd21928c2393cea44e
```

The exp_11 fill manifest is:

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
    "finalize": "f808e7f0c9a711be32670c54fd22dea5daf0a95772329a9fca722a54198a06f3",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "4153a2c749be5df4f092374d139dbcc73a8118e2966a3ce2be208c244fbfe2e0",
    "smoke": "60571786e0acb996d847d15d747a819c3b8e6616cdb54247b347e3a8d00749f9",
    "summarize_haa": "650a5a6d97cbc8a3c4c54d647a3c75a993d9f79eb22b2c9b10034d357950a660",
    "train": "6cd346cfc892ea27f1133909fadffb5087fbba343778a7eba9668ea2fc11e46f"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "f3c79c383ebf13bac86eb0157bb5154263eeec9f6917c1fd21928c2393cea44e",
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
d5c03bf9c0c458701d7edb485c52220e59678b3f167efbbf734167dc85656141
```

No repository files were changed and no review file was saved. Worktree status retains only the pre-existing untracked close-12 prompt.
tokens used
74,439
**OpenAI Codex — exp_11 `orientation_cue_fairness`, ROUND 2, CLOSE review 12**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; approved escalation for `/tmp` test fixtures only |
| Runtime | CUDA disabled; one CPU thread per worker, at most four workers |
| Worktree / branch | `/home/yixunhu/codespace/xRIR_code_wt` / `exp11-cue` |
| HEAD | `0885c41bd0a651f402e00edc20c5b9a15f0a90be` |
| Git tree | `723cde57373a20170646877fee2c11fab1ade6a4` |
| Verification completed | 2026-09-28 01:06:14 UTC / 2026-09-27 21:06:14 EDT |

**Verdict: request changes. ROUND 2 cannot be CLOSED; `0885c41` should not be merged.**

The execution/import failure from close review 11 is repaired in the tested paths. Two remaining paths still let resolution retire a registered live trainer.

1. **[P1] An unreadable existing PID record becomes definite “not alive.”**

   [tools/exp11_pidrecord.py:42](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_pidrecord.py:42) catches every `OSError`, including `PermissionError`, and returns `None`. The CLI consequently emits `norecord`, exit 0. The shell interprets that as a definite absence of a record.

   Independently reproduced with the actual sourced `run_child`, GNU `timeout`, real publication and registration locks, and a stub importing real `register_trainer`:

   ```text
   owned timeout and launcher stopped
   real register_trainer succeeds; train.pid names the live trainer
   train.pid made unreadable with chmod 000

   check_pid_reader -> 0
   pid_record       -> 1
   pid_alive        -> 1
   resolve_unregistered -> 0

   tombstone created
   attempt renamed
   launching marker removed
   registered trainer still alive
   ```

   A second independent replay applied the permission change after acquiring the publication lock and produced the same result. The trainer subsequently passed its continuation gate.

   **Required:** distinguish confirmed absence or malformed readable bytes from inability to inspect an existing record. Permission and I/O failures must produce unknown and prevent resolution. The shell’s [`-f` guard](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:302) also needs care where inaccessible metadata would otherwise look absent.

   The [“unreadable file” test](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_pidrecord.py:122) tests only a missing path and a directory; it never creates a permission-denied regular file.

2. **[P1] Bash normalizes malformed reader output into a valid verdict.**

   [tools/exp11_launch.sh:303](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:303) captures stdout through command substitution. Bash removes trailing newlines and NUL bytes before the subsequent validation sees the answer.

   Wrappers that delegate normally except for `train.pid`, where they emit either of these byte sequences with exit 0, bypass the protocol:

   ```text
   norecord\n\n
   norecord\0\n
   ```

   Both pass the two health probes. Both become plain `norecord` in Bash. With the real lifecycle and registered live trainer described above, both yielded:

   ```text
   pid_record -> 1
   pid_alive  -> 1
   resolve_unregistered -> 0
   tombstone=True, renamed=True, trainer_alive=True
   ```

   The NUL case prints Bash’s ignored-null-byte warning but proceeds with retirement.

   **Required:** validate the raw output before any lossy shell conversion. Reject additional blank lines and NUL bytes as unknown. Apply equivalent validation to the [health-check captures](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:349), which have the same normalization issue.

The independent lifecycle replay confirms that the intended fix works for these failures:

| Reader condition | Resolution | Both mode scans | CLI resolution |
|---|---:|---:|---:|
| Missing interpreter after publication lock | 2 | 2 | 2 |
| Real module ImportError through the temporary-tree wrapper | 2 | 2 | 2 |
| Failure only on `train.pid` | 2 | 2 | 2 |
| Always `record 1` | 2 | 2 | 2 |
| Always `norecord` | 2 | 2 | 2 |
| `record not-a-pid` only on `train.pid` | 2 | 2 | 2 |
| Empty stdout, wrong first word, nonblank second line, or nonzero exit with a verdict | 2 | 2 | 2 |

Every refusal retained the attempt, marker, and `train.pid`, without a tombstone or rename. Ordinary CLI recovery also returned 2 without promotion for the first six conditions. The lock holder continued using the real interpreter.

Direct state checks produced:

| Input/result | `pid_record` | `pid_alive` | `registration_complete` |
|---|---:|---:|---:|
| Valid live PID | 0 | 0 | 0 |
| Malformed readable bytes | 1 | 1 | 1 |
| Reader execution failure | 2 | 2 | 2 |

`scan_arm` sets `SCAN_REASON` and returns 2 for an encountered unknown. `require_quiet_arm` and `resolve_unregistered` preserve it; `clear_launching` retains the marker.

There is one qualification to “ANY unknown from ANY attempt”: `scan_arm` stops at the first live or unresolved attempt. An unknown in a later attempt may therefore never be inspected, and a sourced caller receives 1. I reproduced this ordering. It remains a refusal and permits no retirement or publication; CLI callers normalize either failure to exit 2.

The health check uses the same `$PYTHON -m tools.exp11_pidrecord` invocation for `1\n` and an empty file, before each production scan/resolution decision. Both constant-answer readers are detected. It does **not** establish that the real files are readable.

| Real-file condition | Observed behavior |
|---|---|
| Unreadable regular file | Healthy probes; definite `norecord` — blocker 1 |
| Directory | Healthy probes; shell returns 1 through its nonregular-file guard |
| FIFO | Healthy probes; shell returns 1 without opening it; direct Python CLI blocks |
| 1 GiB sparse file | Healthy probes; bounded `read(64)` produces definite `norecord` |

The directory/FIFO behavior is an explicit classification in the current implementation, rather than health-check coverage. The huge-file case is bounded and correctly rejects invalid file bytes. The original byte table remains definite: none of its rows returns unknown.

One nonblocking wording defect remains at [tools/exp11_launch.sh:541](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:541): the tombstone asserts “no train.exit,” but resolution does not enforce that condition. A stale marker with dead recorded PIDs and an existing `train.exit` resolved successfully in my replay, retaining `train.exit` in the retired directory while writing that assertion. The reason and `RESOLVED` message should describe only conditions actually checked.

Independent selected pytest verification produced **266 unique passes**:

| Selection | Passed |
|---|---:|
| `test_exp11_pidrecord.py` | 72 |
| `test_exp11_shell.py` | 128 |
| `test_exp11_train.py` | 50 |
| `test_exp11_lock_holder.py` | 7 |
| `test_exp09_sim_eval_closures.py` | 3 |
| Registry preservation | 1 |
| Pinned-oracle checks | 5 |

All runs used `-p no:cacheprovider`, worktree `PYTHONPATH`, disabled bytecode/CUDA, and temporary/cache directories under `/tmp`. Shell syntax and `git diff --check` passed.

**RED evidence:** replaying the exact `a954fe0` archive under `/tmp`, the gibberish-on-trainer regression failed as expected:

```text
expected resolve returncode 2
actual returncode 0
publication lock acquired
RESOLVED emitted
tombstone and _ABORTED_unregistered directory present
```

That case passes at HEAD. The new broken-reader fixtures themselves write a live sleep PID directly; the independent replays above supply the stronger real-registration and real-lifecycle evidence.

Git confirms **13 commits**, production **+151/−32**, tests **+233/−18**, and last production commit `2fc5300`. The recorded full-suite log supports **1 failed, 3878 passed, 55 skipped**, with only the documented stale exp_06 `summarize_haa` approval. I inspected that log; I did not rerun the full repository suite.

The two earlier suite cleanups are corroborated by the [scoped Coder transcript](/home/yixunhu/.claude/projects/-home-yixunhu-codespace-xRIR-code/7773656a-f89c-41f7-adc3-63d05a17d044/subagents/agent-aff672cc49719b2fb.jsonl:11557). It records the suite launches, matches their open logs, command lines and worktree to PIDs **1952017** and **1961431**, then records `kill -TERM 1952017 1961431` and successful termination. Only those two identified Coder-owned suite PIDs were targeted; no process-group or process-name kill was used.

Closure hygiene passes:

- All **28** exp_06/exp_11 code digests match between `2fc5300` and `0885c41`.
- Against exp_06 approvals at `9f98bbb`, only `code.summarize_haa` changes; the other **19** keys match.
- Exactly five exp_11 keys move versus `fdb74fc`: `train`, `finalize`, `launch_sh`, `smoke`, `summarize_haa`. The three HAA keys remain unchanged.
- All four sim-eval closures remain unchanged; all ten actual manifests pass verification.
- Frozen files are unchanged. The permitted exp_06 summarizer is excluded from that frozen-file assertion.
- Pinned serialized payloads and rendered-summary oracles remain byte-identical.

`BACKBONES_EXP06` remains:

```text
f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169
```

Frozen lifecycle SHA256s remain:

```text
tools/exp06_launch.sh
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479
```

The following are the **exact fill values for a content-identical merge of `0885c41`**. They do not constitute merge approval; fixing the blockers requires recomputing affected closures.

Replace only exp_06 `code.summarize_haa`:

```text
old:
645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619

new:
650a5a6d97cbc8a3c4c54d647a3c75a993d9f79eb22b2c9b10034d357950a660
```

Preserving every other byte of the **3190-byte** exp_06 approvals record produces SHA256:

```text
f3c79c383ebf13bac86eb0157bb5154263eeec9f6917c1fd21928c2393cea44e
```

The exp_11 fill manifest is:

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
    "finalize": "f808e7f0c9a711be32670c54fd22dea5daf0a95772329a9fca722a54198a06f3",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "4153a2c749be5df4f092374d139dbcc73a8118e2966a3ce2be208c244fbfe2e0",
    "smoke": "60571786e0acb996d847d15d747a819c3b8e6616cdb54247b347e3a8d00749f9",
    "summarize_haa": "650a5a6d97cbc8a3c4c54d647a3c75a993d9f79eb22b2c9b10034d357950a660",
    "train": "6cd346cfc892ea27f1133909fadffb5087fbba343778a7eba9668ea2fc11e46f"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "f3c79c383ebf13bac86eb0157bb5154263eeec9f6917c1fd21928c2393cea44e",
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
d5c03bf9c0c458701d7edb485c52220e59678b3f167efbbf734167dc85656141
```

No repository files were changed and no review file was saved. Worktree status retains only the pre-existing untracked close-12 prompt.

