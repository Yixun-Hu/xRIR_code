**OpenAI Codex — exp_11 `orientation_cue_fairness`, ROUND 2, CLOSE review 13**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; auto-reviewed escalation for `/tmp` fixtures and owned helpers |
| Runtime | CUDA disabled; CPU thread limit ≤4 |
| Worktree / branch | `/home/yixunhu/codespace/xRIR_code_wt` / `exp11-cue` |
| HEAD | `f81a9f76c4598dce6954c28949694e2cd24c4093` |
| Git tree | `9444549e4e2215678f0fdefd88eeb581189a4916` |
| Verification completed | 2026-09-28 02:26:12 UTC / 2026-09-27 22:26:12 EDT |

**Verdict: request changes. ROUND 2 cannot be CLOSED; `f81a9f7` should not be merged.**

Both close-12 blockers are repaired in the verified paths. One additional permission case still permits resolution to retire a registered, live trainer.

1. **[P1] An arm root that cannot be listed is treated as an empty, quiet arm.**

   At [tools/exp11_launch.sh:414](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:414), `"$root"/attempt_*` silently fails to enumerate a directory without read permission. The unmatched pattern fails the following `-d` check, and `scan_arm` returns success.

   A root with mode `0300` retains write and search permissions. Consequently, known attempt paths remain accessible, both locks can be acquired, and tombstone creation and directory renaming still work.

   I reproduced this with the **actual sourced `run_child`**, GNU `timeout`, a stub importing real `register_trainer`, and real publication and registration locks. After ending only the owned launcher and timeout wrapper through handles, the registered trainer remained alive. The marker was aged beyond the production **120-second** grace.

   ```text
   arm root chmod 0300
   publication lock acquired in full mode
   require_quiet_arm -> 0

   publication lock acquired in finalize mode
   pid_alive train.pid -> 0
   require_quiet_arm -> 0
   resolve_unregistered -> 0

   tombstone created
   attempt renamed to *_ABORTED_unregistered
   registered trainer still alive
   ```

   The production resolution branch calls this resolver before preflight at [tools/exp11_launch.sh:886](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:886), so a later gate does not prevent the retirement.

   **Required:** establish that arm enumeration succeeded before declaring the arm quiet. Enumeration failures must return unknown and prevent retirement/publication. Add regressions covering an unlistable but searchable/writable root under both mode locks and resolution. Successful `stat` of a known pathname does not establish that the directory can be enumerated.

The requested close-12 replays otherwise pass:

| Condition | Observed result |
|---|---|
| `chmod 000` on registered trainer’s PID file | `pid_record=2`, `pid_alive=2`; both scans and resolution refuse |
| `chmod 000` on its attempt directory | Same refusal; no retirement |
| Unreadable parent, existing or absent name | CLI exit 2, empty stdout, one stderr line |
| Absent name with inspectable parent | Definite `norecord` |
| Directory, FIFO, socket, `/dev/null` | Definite `norecord` |
| Symlink to regular PID file | Reads the target’s record |
| Symlink loop | Unknown |
| FIFO | Returns promptly without opening/blocking |

The permission lifecycle replays retained the attempt, marker and PID record, with no tombstone or rename. Tests ran as non-root UID 1013; permission cases were not skipped.

For reader wrappers emitting these bytes **only on `train.pid`**, both health probes passed, but the actual read returned unknown:

```text
norecord\n\n
norecord\0\n
record 123\n\n
record 123\0\n
record 123 \n
\nrecord 123\n
```

Each produced `pid_record=2`, `pid_alive=2`, refusal by both mode scans, and resolution refusal with the registered trainer preserved. Applying each payload separately to either health probe made the reader unhealthy. A reader adding one newline to every otherwise-correct answer was also unhealthy.

Additional output checks passed:

- Empty output, an eleven-digit PID, eleven digits without the protocol prefix, CRLF, and a missing terminal newline were rejected.
- Exact one-digit and ten-digit PID answers were accepted.
- `mktemp` failure produced unknown; health-check temporary-directory failure produced unhealthy.
- Completed checks left no verdict/probe temporary files.
- A wrapper attempting **2 GiB** of digit output, constrained by a **128 KiB file-size limit**, produced unknown and refusal throughout. This exercised output failure without filling `/tmp`.

The raw-byte comparison at [tools/exp11_launch.sh:319](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:319) avoids the previous normalization defect: stdout stays in a file, and `cmp` validates the complete bytes. Command substitution handles only the digit candidate; the original output must still match exactly.

The accepted candidate is limited to **1–10 digits**, but its **extraction is not bounded**: [line 325](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:325) materializes all digits before checking length. The Coder report’s “only disk” characterization is therefore incomplete. Rogue output can consume disk, shell memory, CPU, and time. I found no route from that resource exposure to accepting a malformed verdict; with the pinned reader’s bounded output, this remains a nonblocking availability concern.

The other recorded residuals are also nonblocking:

- An earlier live/unresolved attempt can hide a later unknown because `scan_arm` stops early. The result remains a refusal.
- Additional processes and temporary files increase per-read cost without changing the decision.

These residuals are distinct from blocker 1, where failed enumeration returns **success**.

The `%f` mode-bit classification is independent of translated `%F` descriptions. Runtime classifications agreed under installed C, C.utf8 and en_US.utf8 locales; no non-English locale was installed. The tombstone/`RESOLVED` wording now describes the intended checks and no longer falsely asserts absence of `train.exit`. Blocker 1 can still invalidate its claim that no recorded PID is alive.

Independent selected verification produced **284 unique passes**:

| Selection | Passed |
|---|---:|
| `test_exp11_pidrecord.py` | 75 |
| `test_exp11_shell.py` | 144 |
| `test_exp11_train.py` | 50 |
| `test_exp11_lock_holder.py` | 7 |
| `test_exp09_sim_eval_closures.py` | 3 |
| Pinned-oracle checks | 5 |

Runs used `-p no:cacheprovider`, worktree `PYTHONPATH`, disabled bytecode/CUDA, and temporary/cache paths under `/tmp`. Both shell syntax checks and `git diff --check` passed.

**RED evidence:** replaying the relevant `73d8c60` files under `/tmp` produced the expected failure in `test_the_file_classification_does_not_depend_on_a_translated_word`, detecting `stat -L -c %F`. That regression combines a source assertion with a FIFO check; it is not a translated-locale runtime replay. The new blocker above also has direct failing behavioral evidence at HEAD.

Git confirms **eight commits**, production **+135/−48**, and last production commit `5593d69`. Later commits contain worklog records only. The saved full-suite log supports **1 failed, 3897 passed, 55 skipped**, with only the stale exp_06 `summarize_haa` approval failure. Its recorded `head` fixture is `bc0b363`, whose production sources equal `5593d69`; it is not evidence of an isolated, fixed-HEAD `5593d69` run. I did not rerun the full suite.

Closure hygiene passes:

- All **28** exp_06/exp_11 code digests agree between `5593d69` and `f81a9f7`.
- Against exp_06 approvals at `9f98bbb`, only `code.summarize_haa` moves; the other **19** keys match.
- Exactly five exp_11 keys move versus `0885c41`: `train`, `finalize`, `launch_sh`, `smoke`, `summarize_haa`. The three HAA keys remain unchanged.
- All four sim-eval closures remain unchanged; all ten actual simulation manifests verify.
- Frozen files remain untouched, excluding the permitted exp_06 summarizer change.
- Pinned exp_06/exp_09 serialized analyses and rendered summaries remain byte-identical.

Unchanged identities:

```text
BACKBONES_EXP06
f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169

tools/exp06_launch.sh
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479
```

The following are the **exact fill values for a content-identical merge of `f81a9f7`**. They do not authorize merging this blocked candidate. Recompute affected closures after the fix.

Replace only exp_06 `code.summarize_haa`:

```text
old:
645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619

new:
40626dd26bdeac80f1cbfc01d243e43d5ab0017d59bb9129c11b86771ef6495d
```

Preserving every other byte of the **3190-byte** exp_06 approvals record gives SHA256:

```text
a514b1ece05ce107a1f27386396c3345caa87bb03347d1e8b4e49ddefc32f119
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
    "finalize": "256349e7dfa55880ee37eb6c97192ade560af1aff322941962e0b861c860be73",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "cfb86bc5a4f27d46c810d8e5346171ce7a085dc51f7646b3c429c407e3935d9b",
    "smoke": "9940aadc525cbb01ceb5925b38056657630145f2998690c8d077399a1608ccbf",
    "summarize_haa": "40626dd26bdeac80f1cbfc01d243e43d5ab0017d59bb9129c11b86771ef6495d",
    "train": "0f518d3f79cb6b992d0ff6057888f93c0bae0d9520eeb8c0522d1ab40d11b9c2"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "a514b1ece05ce107a1f27386396c3345caa87bb03347d1e8b4e49ddefc32f119",
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
539f08b98f9f7b072c810a51eca5f170b47fd3295386db29c3569ce98452ee5e
```

No repository files were changed and no review was saved. Worktree status retains only the pre-existing untracked close-13 prompt.
tokens used
68,640
**OpenAI Codex — exp_11 `orientation_cue_fairness`, ROUND 2, CLOSE review 13**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; auto-reviewed escalation for `/tmp` fixtures and owned helpers |
| Runtime | CUDA disabled; CPU thread limit ≤4 |
| Worktree / branch | `/home/yixunhu/codespace/xRIR_code_wt` / `exp11-cue` |
| HEAD | `f81a9f76c4598dce6954c28949694e2cd24c4093` |
| Git tree | `9444549e4e2215678f0fdefd88eeb581189a4916` |
| Verification completed | 2026-09-28 02:26:12 UTC / 2026-09-27 22:26:12 EDT |

**Verdict: request changes. ROUND 2 cannot be CLOSED; `f81a9f7` should not be merged.**

Both close-12 blockers are repaired in the verified paths. One additional permission case still permits resolution to retire a registered, live trainer.

1. **[P1] An arm root that cannot be listed is treated as an empty, quiet arm.**

   At [tools/exp11_launch.sh:414](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:414), `"$root"/attempt_*` silently fails to enumerate a directory without read permission. The unmatched pattern fails the following `-d` check, and `scan_arm` returns success.

   A root with mode `0300` retains write and search permissions. Consequently, known attempt paths remain accessible, both locks can be acquired, and tombstone creation and directory renaming still work.

   I reproduced this with the **actual sourced `run_child`**, GNU `timeout`, a stub importing real `register_trainer`, and real publication and registration locks. After ending only the owned launcher and timeout wrapper through handles, the registered trainer remained alive. The marker was aged beyond the production **120-second** grace.

   ```text
   arm root chmod 0300
   publication lock acquired in full mode
   require_quiet_arm -> 0

   publication lock acquired in finalize mode
   pid_alive train.pid -> 0
   require_quiet_arm -> 0
   resolve_unregistered -> 0

   tombstone created
   attempt renamed to *_ABORTED_unregistered
   registered trainer still alive
   ```

   The production resolution branch calls this resolver before preflight at [tools/exp11_launch.sh:886](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:886), so a later gate does not prevent the retirement.

   **Required:** establish that arm enumeration succeeded before declaring the arm quiet. Enumeration failures must return unknown and prevent retirement/publication. Add regressions covering an unlistable but searchable/writable root under both mode locks and resolution. Successful `stat` of a known pathname does not establish that the directory can be enumerated.

The requested close-12 replays otherwise pass:

| Condition | Observed result |
|---|---|
| `chmod 000` on registered trainer’s PID file | `pid_record=2`, `pid_alive=2`; both scans and resolution refuse |
| `chmod 000` on its attempt directory | Same refusal; no retirement |
| Unreadable parent, existing or absent name | CLI exit 2, empty stdout, one stderr line |
| Absent name with inspectable parent | Definite `norecord` |
| Directory, FIFO, socket, `/dev/null` | Definite `norecord` |
| Symlink to regular PID file | Reads the target’s record |
| Symlink loop | Unknown |
| FIFO | Returns promptly without opening/blocking |

The permission lifecycle replays retained the attempt, marker and PID record, with no tombstone or rename. Tests ran as non-root UID 1013; permission cases were not skipped.

For reader wrappers emitting these bytes **only on `train.pid`**, both health probes passed, but the actual read returned unknown:

```text
norecord\n\n
norecord\0\n
record 123\n\n
record 123\0\n
record 123 \n
\nrecord 123\n
```

Each produced `pid_record=2`, `pid_alive=2`, refusal by both mode scans, and resolution refusal with the registered trainer preserved. Applying each payload separately to either health probe made the reader unhealthy. A reader adding one newline to every otherwise-correct answer was also unhealthy.

Additional output checks passed:

- Empty output, an eleven-digit PID, eleven digits without the protocol prefix, CRLF, and a missing terminal newline were rejected.
- Exact one-digit and ten-digit PID answers were accepted.
- `mktemp` failure produced unknown; health-check temporary-directory failure produced unhealthy.
- Completed checks left no verdict/probe temporary files.
- A wrapper attempting **2 GiB** of digit output, constrained by a **128 KiB file-size limit**, produced unknown and refusal throughout. This exercised output failure without filling `/tmp`.

The raw-byte comparison at [tools/exp11_launch.sh:319](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:319) avoids the previous normalization defect: stdout stays in a file, and `cmp` validates the complete bytes. Command substitution handles only the digit candidate; the original output must still match exactly.

The accepted candidate is limited to **1–10 digits**, but its **extraction is not bounded**: [line 325](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:325) materializes all digits before checking length. The Coder report’s “only disk” characterization is therefore incomplete. Rogue output can consume disk, shell memory, CPU, and time. I found no route from that resource exposure to accepting a malformed verdict; with the pinned reader’s bounded output, this remains a nonblocking availability concern.

The other recorded residuals are also nonblocking:

- An earlier live/unresolved attempt can hide a later unknown because `scan_arm` stops early. The result remains a refusal.
- Additional processes and temporary files increase per-read cost without changing the decision.

These residuals are distinct from blocker 1, where failed enumeration returns **success**.

The `%f` mode-bit classification is independent of translated `%F` descriptions. Runtime classifications agreed under installed C, C.utf8 and en_US.utf8 locales; no non-English locale was installed. The tombstone/`RESOLVED` wording now describes the intended checks and no longer falsely asserts absence of `train.exit`. Blocker 1 can still invalidate its claim that no recorded PID is alive.

Independent selected verification produced **284 unique passes**:

| Selection | Passed |
|---|---:|
| `test_exp11_pidrecord.py` | 75 |
| `test_exp11_shell.py` | 144 |
| `test_exp11_train.py` | 50 |
| `test_exp11_lock_holder.py` | 7 |
| `test_exp09_sim_eval_closures.py` | 3 |
| Pinned-oracle checks | 5 |

Runs used `-p no:cacheprovider`, worktree `PYTHONPATH`, disabled bytecode/CUDA, and temporary/cache paths under `/tmp`. Both shell syntax checks and `git diff --check` passed.

**RED evidence:** replaying the relevant `73d8c60` files under `/tmp` produced the expected failure in `test_the_file_classification_does_not_depend_on_a_translated_word`, detecting `stat -L -c %F`. That regression combines a source assertion with a FIFO check; it is not a translated-locale runtime replay. The new blocker above also has direct failing behavioral evidence at HEAD.

Git confirms **eight commits**, production **+135/−48**, and last production commit `5593d69`. Later commits contain worklog records only. The saved full-suite log supports **1 failed, 3897 passed, 55 skipped**, with only the stale exp_06 `summarize_haa` approval failure. Its recorded `head` fixture is `bc0b363`, whose production sources equal `5593d69`; it is not evidence of an isolated, fixed-HEAD `5593d69` run. I did not rerun the full suite.

Closure hygiene passes:

- All **28** exp_06/exp_11 code digests agree between `5593d69` and `f81a9f7`.
- Against exp_06 approvals at `9f98bbb`, only `code.summarize_haa` moves; the other **19** keys match.
- Exactly five exp_11 keys move versus `0885c41`: `train`, `finalize`, `launch_sh`, `smoke`, `summarize_haa`. The three HAA keys remain unchanged.
- All four sim-eval closures remain unchanged; all ten actual simulation manifests verify.
- Frozen files remain untouched, excluding the permitted exp_06 summarizer change.
- Pinned exp_06/exp_09 serialized analyses and rendered summaries remain byte-identical.

Unchanged identities:

```text
BACKBONES_EXP06
f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169

tools/exp06_launch.sh
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479
```

The following are the **exact fill values for a content-identical merge of `f81a9f7`**. They do not authorize merging this blocked candidate. Recompute affected closures after the fix.

Replace only exp_06 `code.summarize_haa`:

```text
old:
645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619

new:
40626dd26bdeac80f1cbfc01d243e43d5ab0017d59bb9129c11b86771ef6495d
```

Preserving every other byte of the **3190-byte** exp_06 approvals record gives SHA256:

```text
a514b1ece05ce107a1f27386396c3345caa87bb03347d1e8b4e49ddefc32f119
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
    "finalize": "256349e7dfa55880ee37eb6c97192ade560af1aff322941962e0b861c860be73",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "cfb86bc5a4f27d46c810d8e5346171ce7a085dc51f7646b3c429c407e3935d9b",
    "smoke": "9940aadc525cbb01ceb5925b38056657630145f2998690c8d077399a1608ccbf",
    "summarize_haa": "40626dd26bdeac80f1cbfc01d243e43d5ab0017d59bb9129c11b86771ef6495d",
    "train": "0f518d3f79cb6b992d0ff6057888f93c0bae0d9520eeb8c0522d1ab40d11b9c2"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "a514b1ece05ce107a1f27386396c3345caa87bb03347d1e8b4e49ddefc32f119",
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
539f08b98f9f7b072c810a51eca5f170b47fd3295386db29c3569ce98452ee5e
```

No repository files were changed and no review was saved. Worktree status retains only the pre-existing untracked close-13 prompt.

