**OpenAI Codex — exp_11 `orientation_cue_fairness`, ROUND 2, CLOSE review 11**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; auto-approved escalation for temporary tests and replays |
| Runtime | CUDA disabled; numerical libraries limited to one thread per worker; ≤4 workers |
| Worktree / branch | `/home/yixunhu/codespace/xRIR_code_wt` / `exp11-cue` |
| HEAD | `fdb74fc42ac28d065bf09fe8684cddc741d813c9` |
| Git tree | `4d7ff72b3d3ac00e44890fc9a99dcb35412a0f8e` |
| Verification completed | 2026-09-27 23:27:28 UTC / 19:27:28 EDT |

**Verdict: request changes. ROUND 2 cannot be CLOSED; `fdb74fc` should not be merged.**

The two close-10 defects are repaired. One blocker remains in the handling of reader execution failures.

1. **[P1] A failed reader lets resolution retire an already-registered live trainer.**

   At [tools/exp11_launch.sh:277](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:277), every reader failure becomes “not alive.” The scan then skips the resolution target’s marker at [line 307](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:307). Consequently, the resolver accepts this failed liveness observation at [line 419](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:419) and tombstones and renames the live trainer’s attempt.

   Independently reproduced using the actual sourced `run_child`, real GNU `timeout`, real publication and registration locks, and a stub importing real `register_trainer`:

   ```text
   KILL owned timeout wrapper through pidfd
   KILL owned launcher through Popen
   release trainer registration gate
   real register_trainer succeeds; train.pid names live trainer

   working reader:
     resolve_unregistered -> refuses live train.pid

   reader import failure:
     registration_complete -> false
     clear_launching -> marker retained
     ordinary scan -> refuses unresolved
     resolve_unregistered -> exit 0
     tombstone created; attempt renamed; marker removed
     registered trainer remains alive

   release trainer continuation gate:
     trainer continues after resolution
   ```

   The import-failure injection affected only `-m tools.exp11_pidrecord`; the lock holder continued using the real interpreter. A missing `$PYTHON` **after acquiring the publication lock** reproduced the same defect. A missing interpreter **before lock acquisition** correctly prevented the holder from starting.

   The marker therefore bounds reader failure during an ordinary unfinished-launch scan, but **does not bound resolution**, which deliberately excludes its target’s marker. The registration lock is already released after registration, and the tombstone is not checked again by a registered trainer.

   Restoring the reader detects the live trainer in the retired directory. This limits subsequent damage after restoration, but does not undo the incorrect retirement.

   **Required:** distinguish a completed reader verdict from failure to execute/import the reader, and fail closed on unknown liveness in scans and resolution. Add regressions for a registered live trainer under both execution and import failure. Import errors can themselves exit 1, so simply interpreting exit 1 as “invalid record” is insufficient.

The independent byte replay passed through the Python reader, CLI, and sourced shell. Here, `P` is a live owned helper and `D` an exited, reaped helper.

| File bytes | Python / shell complete | CLI exit | Shell alive |
|---|---:|---:|---:|
| `P` | Yes / Yes | 0 | Yes |
| `P\n` | Yes / Yes | 0 | Yes |
| `1234567890\n` | Yes / Yes | 0 | No |
| `\nP\n` | No / No | 1 | No |
| `12\n34` | No / No | 1 | No |
| ` P\n` | No / No | 1 | No |
| `P\r` | No / No | 1 | No |
| `P\r\n` | No / No | 1 | No |
| Empty | No / No | 1 | No |
| `\n\n` | No / No | 1 | No |
| `P\n\n` | No / No | 1 | No |
| Eleven digits | No / No | 1 | No |
| `P\0` | No / No | 1 | No |
| `\0P` | No / No | 1 | No |
| NUL inside `P` | No / No | 1 | No |
| `P\0\n` | No / No | 1 | No |
| `D\0` | No / No | 1 | No |

All these unfinished attempts remained nonquiet and retained their markers. Additional rows `D`, `\0D\n`, and `P\0\0` also agreed. `D\0` retained its marker even with a `train.exit` file present.

[tools/exp11_pidrecord.py:24](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_pidrecord.py:24) uses bytes `fullmatch(rb'[0-9]{1,10}\n?')`. Missing, unreadable, and directory paths each returned CLI exit 1, empty output, and no traceback. The launcher no longer substitutes PID-file contents through Bash; the forbidden content-reading constructs are absent.

The lifecycle replays gave these results:

| Real lifecycle schedule | Result |
|---|---|
| Kill wrapper, then launcher, before registration | Both modes refuse **unresolved** |
| Release registration gate | Both modes refuse **live trainer (`train.pid`)** |
| Trainer records exit; marker older than grace | Resolution succeeds |
| Ordinary `run_child` with real `timeout` | Marker cleared; scan quiet |
| Registered trainer exits without `train.exit` | Both modes remain unresolved; explicit resolution succeeds after grace |
| Trainer crashes before registration | No `train.pid` or `train.exit`; both modes remain unresolved until resolution |
| Live unregistered trainer delayed beyond grace | Resolution succeeds; subsequent registration refuses |

These replays called the real sourced lifecycle rather than copying its prologue. Only owned helpers were terminated, through handles. Stub file gates were checked for existence, never evaluated.

The delayed-unregistered case confirms the intended tombstone defense. After resolution, even recreating the original directory and marker did not permit registration: the persistent tombstone caused refusal under the registration lock. No trainer registration or continuation occurred.

The completion-receipt and published-`final` guards also pass. Publication locking serializes publication decisions; registration locking serializes registration against retirement; the quiet scan and marker age govern other attempts.

Registration occurs at [tools/exp11_train.py:331](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_train.py:331), before admission, dataset construction, loaders, model construction, and `.cuda()`. Heavy imports precede registration, so “nothing expensive happens first” would be inaccurate. A fresh import took 2.780 seconds and reported `torch.cuda.is_initialized() == False`; inspection found no repository import-time dataset/model construction or explicit CUDA allocation. The monotonic registration deadline is correct.

Independent selected pytest verification produced **256 unique passing tests**:

| Selection | Passed |
|---|---:|
| `test_exp11_shell.py` | 116 |
| `test_exp11_pidrecord.py` | 70 |
| `test_exp11_train.py` | 50 |
| `test_exp11_lock_holder.py` | 7 |
| Selected profile checks | 2 |
| `test_exp09_sim_eval_closures.py` | 3 |
| Registry preservation check | 1 |
| Pinned-oracle checks | 7 |

Runs used `-p no:cacheprovider`, disabled bytecode, and temporary test/cache directories. Shell syntax checks and `git diff --check` passed.

**RED evidence:** loading the exact `tools/exp11_profiles.py` blob from `678d9df` and running the unchanged launcher-coverage test produced the expected failure:

```text
AssertionError: no approval key covers: ['tools/exp11_pidrecord.py']
1 failed
```

The same test passes at HEAD. [tools/exp11_profiles.py:54](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_profiles.py:54) explicitly binds the reader in `launch_sh`. The coverage test enumerates dotted and slash-form exp_11 module references and checks their approval coverage.

Git confirms **14 commits**, production **+105/−46**, tests **+444/−2**, and last production commit `275be24`. Per-commit counts match the report. The recorded full-suite log contains **1 failed, 3864 passed, 55 skipped**, with the documented exp_06 `summarize_haa` approval mismatch. I inspected that evidence; I did not rerun the full repository suite.

Two nonblocking wording corrections remain:

- [tools/exp11_launch.sh:438](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:438) records “no complete child.pid,” and line 446 claims nothing remains alive. Valid resolution can now involve a complete dead wrapper record and a living unregistered trainer.
- The [fix11 report:234](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-27_17:57:59_coder_round2_fix11_opus_report.md:234) overstates “no approval key at all.” The reader already belonged to several transitive Python closures; its omission was specifically from `launch_sh`.

Closure hygiene passes:

- All 28 exp_06/exp_11 code digests match between `275be24` and `fdb74fc`.
- Against the exp_06 approvals at `9f98bbb`, only `code.summarize_haa` changes; the other 19 keys remain identical.
- Five exp_11 keys moved since `51e34b1`: `train`, `finalize`, `launch_sh`, `smoke`, `summarize_haa`. The three HAA keys remain unchanged.
- All four sim-eval closures remain unchanged. All ten actual sim-eval manifests pass verification.
- Frozen files remain unchanged; the permitted exp_06 summarizer is excluded from that frozen-file assertion.
- `BACKBONES_EXP06` remains `f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169`.
- Pinned serialized payloads and rendered-summary oracles remain byte-identical.
- Frozen lifecycle file hashes are:

```text
tools/exp06_launch.sh
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479
```

The following are the **exact fill values for a content-identical merge of the reviewed HEAD**. They are not merge approval; fixing blocker 1 requires recomputing affected closures.

Replace only exp_06 `code.summarize_haa`:

```text
old:
645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619

new:
2f2f4383c94deb76d1a294616539d8a6dddf34bc5e9b9f90db104b8118f2abd2
```

Preserving every other byte of the 3190-byte exp_06 record, that replacement produces:

```text
782c4a1b252e60f149d9c14ef812f88d09b5937c34c9f7806ac75b529e3299cd
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
    "finalize": "1a008a904d3a402cc91498ddff0f9646015e45f7d49e8efcf89e683cabb5b357",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "78145bdf2de434c8cc2845d442a6e7fb74d6c2724e63007384aa42e7574947bb",
    "smoke": "af2a9079166fb9c37b28cb50ae3de6a4afe7dc6d00086e27667c458aaf84ed32",
    "summarize_haa": "2f2f4383c94deb76d1a294616539d8a6dddf34bc5e9b9f90db104b8118f2abd2",
    "train": "2d22f3d190de1a1a1aaefea9593f8f6d3317945f0591f4aa47e1b060825e517d"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "782c4a1b252e60f149d9c14ef812f88d09b5937c34c9f7806ac75b529e3299cd",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

With sorted keys, two-space indentation, and one trailing newline, this is 1316 bytes, SHA256:

```text
723342049f96c24e5e48d6fe9b0698041343eaae86e05c0693967fef419b2963
```

Execution caveat: initial exploratory imports lacked explicit Numba cache redirection and may have created immediately removed cache-locator probes outside `/tmp`. The definitive byte and lifecycle replays were repeated with all caches redirected to `/tmp`, reproducing every result.

No repository files were changed and no review file was saved. Worktree status retains only the pre-existing untracked close-11 prompt.
tokens used
78,126
**OpenAI Codex — exp_11 `orientation_cue_fairness`, ROUND 2, CLOSE review 11**

| Identity | Value |
|---|---|
| Model | GPT-6 / OpenAI Codex |
| Reasoning effort | Not exposed in the supplied invocation metadata |
| Sandbox | Managed read-only; auto-approved escalation for temporary tests and replays |
| Runtime | CUDA disabled; numerical libraries limited to one thread per worker; ≤4 workers |
| Worktree / branch | `/home/yixunhu/codespace/xRIR_code_wt` / `exp11-cue` |
| HEAD | `fdb74fc42ac28d065bf09fe8684cddc741d813c9` |
| Git tree | `4d7ff72b3d3ac00e44890fc9a99dcb35412a0f8e` |
| Verification completed | 2026-09-27 23:27:28 UTC / 19:27:28 EDT |

**Verdict: request changes. ROUND 2 cannot be CLOSED; `fdb74fc` should not be merged.**

The two close-10 defects are repaired. One blocker remains in the handling of reader execution failures.

1. **[P1] A failed reader lets resolution retire an already-registered live trainer.**

   At [tools/exp11_launch.sh:277](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:277), every reader failure becomes “not alive.” The scan then skips the resolution target’s marker at [line 307](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:307). Consequently, the resolver accepts this failed liveness observation at [line 419](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:419) and tombstones and renames the live trainer’s attempt.

   Independently reproduced using the actual sourced `run_child`, real GNU `timeout`, real publication and registration locks, and a stub importing real `register_trainer`:

   ```text
   KILL owned timeout wrapper through pidfd
   KILL owned launcher through Popen
   release trainer registration gate
   real register_trainer succeeds; train.pid names live trainer

   working reader:
     resolve_unregistered -> refuses live train.pid

   reader import failure:
     registration_complete -> false
     clear_launching -> marker retained
     ordinary scan -> refuses unresolved
     resolve_unregistered -> exit 0
     tombstone created; attempt renamed; marker removed
     registered trainer remains alive

   release trainer continuation gate:
     trainer continues after resolution
   ```

   The import-failure injection affected only `-m tools.exp11_pidrecord`; the lock holder continued using the real interpreter. A missing `$PYTHON` **after acquiring the publication lock** reproduced the same defect. A missing interpreter **before lock acquisition** correctly prevented the holder from starting.

   The marker therefore bounds reader failure during an ordinary unfinished-launch scan, but **does not bound resolution**, which deliberately excludes its target’s marker. The registration lock is already released after registration, and the tombstone is not checked again by a registered trainer.

   Restoring the reader detects the live trainer in the retired directory. This limits subsequent damage after restoration, but does not undo the incorrect retirement.

   **Required:** distinguish a completed reader verdict from failure to execute/import the reader, and fail closed on unknown liveness in scans and resolution. Add regressions for a registered live trainer under both execution and import failure. Import errors can themselves exit 1, so simply interpreting exit 1 as “invalid record” is insufficient.

The independent byte replay passed through the Python reader, CLI, and sourced shell. Here, `P` is a live owned helper and `D` an exited, reaped helper.

| File bytes | Python / shell complete | CLI exit | Shell alive |
|---|---:|---:|---:|
| `P` | Yes / Yes | 0 | Yes |
| `P\n` | Yes / Yes | 0 | Yes |
| `1234567890\n` | Yes / Yes | 0 | No |
| `\nP\n` | No / No | 1 | No |
| `12\n34` | No / No | 1 | No |
| ` P\n` | No / No | 1 | No |
| `P\r` | No / No | 1 | No |
| `P\r\n` | No / No | 1 | No |
| Empty | No / No | 1 | No |
| `\n\n` | No / No | 1 | No |
| `P\n\n` | No / No | 1 | No |
| Eleven digits | No / No | 1 | No |
| `P\0` | No / No | 1 | No |
| `\0P` | No / No | 1 | No |
| NUL inside `P` | No / No | 1 | No |
| `P\0\n` | No / No | 1 | No |
| `D\0` | No / No | 1 | No |

All these unfinished attempts remained nonquiet and retained their markers. Additional rows `D`, `\0D\n`, and `P\0\0` also agreed. `D\0` retained its marker even with a `train.exit` file present.

[tools/exp11_pidrecord.py:24](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_pidrecord.py:24) uses bytes `fullmatch(rb'[0-9]{1,10}\n?')`. Missing, unreadable, and directory paths each returned CLI exit 1, empty output, and no traceback. The launcher no longer substitutes PID-file contents through Bash; the forbidden content-reading constructs are absent.

The lifecycle replays gave these results:

| Real lifecycle schedule | Result |
|---|---|
| Kill wrapper, then launcher, before registration | Both modes refuse **unresolved** |
| Release registration gate | Both modes refuse **live trainer (`train.pid`)** |
| Trainer records exit; marker older than grace | Resolution succeeds |
| Ordinary `run_child` with real `timeout` | Marker cleared; scan quiet |
| Registered trainer exits without `train.exit` | Both modes remain unresolved; explicit resolution succeeds after grace |
| Trainer crashes before registration | No `train.pid` or `train.exit`; both modes remain unresolved until resolution |
| Live unregistered trainer delayed beyond grace | Resolution succeeds; subsequent registration refuses |

These replays called the real sourced lifecycle rather than copying its prologue. Only owned helpers were terminated, through handles. Stub file gates were checked for existence, never evaluated.

The delayed-unregistered case confirms the intended tombstone defense. After resolution, even recreating the original directory and marker did not permit registration: the persistent tombstone caused refusal under the registration lock. No trainer registration or continuation occurred.

The completion-receipt and published-`final` guards also pass. Publication locking serializes publication decisions; registration locking serializes registration against retirement; the quiet scan and marker age govern other attempts.

Registration occurs at [tools/exp11_train.py:331](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_train.py:331), before admission, dataset construction, loaders, model construction, and `.cuda()`. Heavy imports precede registration, so “nothing expensive happens first” would be inaccurate. A fresh import took 2.780 seconds and reported `torch.cuda.is_initialized() == False`; inspection found no repository import-time dataset/model construction or explicit CUDA allocation. The monotonic registration deadline is correct.

Independent selected pytest verification produced **256 unique passing tests**:

| Selection | Passed |
|---|---:|
| `test_exp11_shell.py` | 116 |
| `test_exp11_pidrecord.py` | 70 |
| `test_exp11_train.py` | 50 |
| `test_exp11_lock_holder.py` | 7 |
| Selected profile checks | 2 |
| `test_exp09_sim_eval_closures.py` | 3 |
| Registry preservation check | 1 |
| Pinned-oracle checks | 7 |

Runs used `-p no:cacheprovider`, disabled bytecode, and temporary test/cache directories. Shell syntax checks and `git diff --check` passed.

**RED evidence:** loading the exact `tools/exp11_profiles.py` blob from `678d9df` and running the unchanged launcher-coverage test produced the expected failure:

```text
AssertionError: no approval key covers: ['tools/exp11_pidrecord.py']
1 failed
```

The same test passes at HEAD. [tools/exp11_profiles.py:54](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_profiles.py:54) explicitly binds the reader in `launch_sh`. The coverage test enumerates dotted and slash-form exp_11 module references and checks their approval coverage.

Git confirms **14 commits**, production **+105/−46**, tests **+444/−2**, and last production commit `275be24`. Per-commit counts match the report. The recorded full-suite log contains **1 failed, 3864 passed, 55 skipped**, with the documented exp_06 `summarize_haa` approval mismatch. I inspected that evidence; I did not rerun the full repository suite.

Two nonblocking wording corrections remain:

- [tools/exp11_launch.sh:438](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:438) records “no complete child.pid,” and line 446 claims nothing remains alive. Valid resolution can now involve a complete dead wrapper record and a living unregistered trainer.
- The [fix11 report:234](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/orientation_cue_fairness_2026-09-27_17:57:59_coder_round2_fix11_opus_report.md:234) overstates “no approval key at all.” The reader already belonged to several transitive Python closures; its omission was specifically from `launch_sh`.

Closure hygiene passes:

- All 28 exp_06/exp_11 code digests match between `275be24` and `fdb74fc`.
- Against the exp_06 approvals at `9f98bbb`, only `code.summarize_haa` changes; the other 19 keys remain identical.
- Five exp_11 keys moved since `51e34b1`: `train`, `finalize`, `launch_sh`, `smoke`, `summarize_haa`. The three HAA keys remain unchanged.
- All four sim-eval closures remain unchanged. All ten actual sim-eval manifests pass verification.
- Frozen files remain unchanged; the permitted exp_06 summarizer is excluded from that frozen-file assertion.
- `BACKBONES_EXP06` remains `f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169`.
- Pinned serialized payloads and rendered-summary oracles remain byte-identical.
- Frozen lifecycle file hashes are:

```text
tools/exp06_launch.sh
d20ca47a0baabeb9ab799688aa61b16ca1b2bb8837d57e46611b28761e9f8a09

tools/exp06_finalize.py
9237ed6d4529d6af21127b6d54d1b27d7e339782dfd7781557901dbcd3078479
```

The following are the **exact fill values for a content-identical merge of the reviewed HEAD**. They are not merge approval; fixing blocker 1 requires recomputing affected closures.

Replace only exp_06 `code.summarize_haa`:

```text
old:
645e74c03e20425e361c2efcab564f2f2b2557c98574df538cd7dd439f8b3619

new:
2f2f4383c94deb76d1a294616539d8a6dddf34bc5e9b9f90db104b8118f2abd2
```

Preserving every other byte of the 3190-byte exp_06 record, that replacement produces:

```text
782c4a1b252e60f149d9c14ef812f88d09b5937c34c9f7806ac75b529e3299cd
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
    "finalize": "1a008a904d3a402cc91498ddff0f9646015e45f7d49e8efcf89e683cabb5b357",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "78145bdf2de434c8cc2845d442a6e7fb74d6c2724e63007384aa42e7574947bb",
    "smoke": "af2a9079166fb9c37b28cb50ae3de6a4afe7dc6d00086e27667c458aaf84ed32",
    "summarize_haa": "2f2f4383c94deb76d1a294616539d8a6dddf34bc5e9b9f90db104b8118f2abd2",
    "train": "2d22f3d190de1a1a1aaefea9593f8f6d3317945f0591f4aa47e1b060825e517d"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "782c4a1b252e60f149d9c14ef812f88d09b5937c34c9f7806ac75b529e3299cd",
    "legacy_receipt": {
      "path": "ckpt/exp06/legacy_receipt.json",
      "sha256": "5a12494657665d336cf5f952c6ef918abbeae44e451f28821790ac5c46232540"
    }
  },
  "schema_version": 1
}
```

With sorted keys, two-space indentation, and one trailing newline, this is 1316 bytes, SHA256:

```text
723342049f96c24e5e48d6fe9b0698041343eaae86e05c0693967fef419b2963
```

Execution caveat: initial exploratory imports lacked explicit Numba cache redirection and may have created immediately removed cache-locator probes outside `/tmp`. The definitive byte and lifecycle replays were repeated with all caches redirected to `/tmp`, reproducing every result.

No repository files were changed and no review file was saved. Worktree status retains only the pre-existing untracked close-11 prompt.

