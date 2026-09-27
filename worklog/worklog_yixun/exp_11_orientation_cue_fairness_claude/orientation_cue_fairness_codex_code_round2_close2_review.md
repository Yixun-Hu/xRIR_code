**Reviewer:** OpenAI Codex, GPT-6 per available developer metadata. The exact invocation model identifier and reasoning-effort setting are not exposed; I cannot attest to a more specific setting.

**Sandbox:** Read-only worktree; test/fixture writes confined to `/tmp`; CUDA disabled; numerical threads limited to one per test process; no process signals. Selected pytest runs used worktree `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE=1`, and `-p no:cacheprovider`.

**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`.  
**HEAD:** `612e3dd023eb390b8a818d307a1b85d821b8ed4f`.  
**Review window:** 2026-09-27, approximately 05:20–05:25 EDT / 09:20–09:25 UTC.  
**Range:** All eight commits in `2c64a2a..612e3dd`.

**Verdict: request changes. Round 2 cannot be CLOSED, and `612e3dd` should not be merged.** The receipt and ceiling blockers are resolved. Canonicalization fixes the original alias-basename defect, but recovery remains unsafe.

1. **[P2] Failed recovery can invalidate an already published attempt; concurrent recovery can then report success while publishing a dangling link.**  
   [tools/exp11_launch.sh:383](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:383), [promotion:172](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:172), [recovery ownership:250](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:250).

   With `final -> attempt_A`, recovery through `--attempt final` now correctly resolves `attempt_A`. However, a missing `--log` makes the real finalizer refuse, after which line 384 renames **the canonical directory** to `attempt_A_ABORTED_finalize_refused`. `final` remains dangling, and any existing completion moves with the directory.

   Independent reproduction:

   ```text
   recovery exit:       2
   final is symlink:    true
   final target exists: false
   canonical moved:    true
   cause: EXP11_FINALIZE_REFUSED unreadable log
   ```

   This is a new consequence of passing the resolved directory into `abort`. A malformed recovery invocation does not establish that the previously completed run failed.

   Recovery also has no lock or ownership marker. A controlled interleaving paused one recovery after a successful-finalization stub, let another recovery’s real missing-log refusal rename the shared attempt, then resumed the first:

   ```text
   first recovery exit:  0
   second recovery exit: 2
   final target:         attempt_A
   final target exists:  false
   ```

   Separately, with `final -> attempt_A` and an external alias resolving to `attempt_B`, promotion returned success and replaced `final` with `attempt_B`. `mv -Tf` provides atomic replacement, but no conflict protection or recovery serialization.

   **Required correction:** Make refused recovery preserve existing attempts and logs. Serialize alias resolution, validation and publication per arm, with full-launch promotion participating in the same coordination. Make same-target recovery idempotent and explicitly handle a different existing `final`; silent replacement should not be an accidental recovery policy. Add mutation tests for these failure and concurrency cases.

   These reproductions ran the current launcher functions and literal recovery branch against `/tmp` fixtures. Preflight was stubbed for temporary roots; successful expensive validation was stubbed where stated. Missing-log refusals used the real finalizer, which fails before checkpoint validation.

The requested fixes otherwise verify as follows:

| Item | Independent result |
|---|---|
| Consumed exp_06 approval identity | **Resolved.** The receipt is required at runtime; omission refuses despite the signature’s `None` default. No default-path fallback remains. |
| `main()` wiring and published inputs | `main()` loads `args.approved`, passes `approvals_receipt`, and the check binds that consumed path/digest into inputs. Call site independently asserted using AST. |
| Alternate committed record | A genuinely committed temporary record with changed artifact bytes was refused against the default record’s pin. |
| Byte-identical alternate path | A committed copy with identical bytes was admitted, binding its own resolved path. |
| Ceiling | Bounded decimal-string validation precedes arithmetic: no leading zeros, at most six digits, then `(0,129600]`. |
| Canonical recovery path | Realpath, direct-parent arm root, `attempt_*` basename and arm profile are checked. Existing-`final` and external aliases use the canonical finalizer/promotion path. Failure ownership remains the blocker above. |
| Root override | `EXP11_PRETRAIN_ROOT` is honored only with both `EXP11_TEST_ROOTS=1` and `--dry-run`; other nonempty overrides refuse. |

The independent ceiling dry-runs produced:

```text
9223372036854775808                       exit 2
999999999999999999999999999999999999999  exit 2
129600                                  exit 0; timeout uses 129600
129601                                  exit 2
```

None emitted `integer expression expected`. Zero, leading-zero values and seven-digit values also refused.

The record-local [.gitattributes](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/.gitattributes:5) exemption is **acceptable**. It preserves copied review bytes and applies only to matching review Markdown under that record directory. Source, tests, the coder report and sibling records remain outside the exemption. Both `git diff --check 2c64a2a..612e3dd` and `c8acda2..612e3dd` pass. The broader `9f98bbb` range still has historical copied-log whitespace.

The qualified red-first account is supported by replay:

| Historical selection | Observed result |
|---|---|
| `8506456`: alternate-record and `main()` call-site tests | Both fail intended contract assertions: missing receipt parameter and absent receipt forwarding. |
| `a6b3200`: six ceiling cases plus existing-`final` alias | Four intended failures, three passes. Oversized integers and leading-zero input pass incorrectly; alias promotion prints `final -> final`. |
| `e98b490`: cross-arm recovery test | Fails on `NameError: json is not defined` before recovery executes. |

These were historical changed files replayed against current dependencies, not complete historical environments.

Minor report corrections are nonblocking: the committed unit test’s “committed” alternate fixture is not itself Git-committed—my reproduction was; the call-site RED test fails its own assertion; **two** exp_11 keys changed against `2c64a2a`, not three; `3119e22` is bookkeeping, while `0b10d7c` is the last production-code commit.

Closure hygiene **passes**:

- Exactly **one of twenty exp_06 keys** differs from the `9f98bbb` approvals: `code.summarize_haa`.
- Historical closure reconstruction independently reproduces the old digest; the other nineteen current keys match.
- All four simulated-evaluation closures—`eval`, `eval_launch`, `compare`, `mirror_probe`—are unchanged.
- Frozen production files are untouched. The only modified existing production source since `9f98bbb` is `tools/exp06_summarize_haa.py`.
- `BACKBONES_EXP06` is unchanged; registry digest remains `f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169`.
- Pinned `c6233e3` exp_06/exp_09 and `c491396` exp_11 Phase-1 oracle comparisons reproduce serialized statistical payloads and rendered summaries byte-for-byte.

Current selected verification totaled **100 passed, no skips**:

| Selection | Passed |
|---|---:|
| Shell | 40 |
| Approval enforcement | 13 |
| Finalizer and adapter zero-shot | 31 |
| Sim-eval closures, including ten actual manifests | 3 |
| Registry and pinned oracles | 13 |

Both shells pass `bash -n`. I did not rerun the full suite; the coder’s full-suite result remains reported evidence.

The following is the **exact fill for the unchanged `612e3dd` candidate tree**. Since recovery requires changes, it is not merge approval; recompute at the eventual reviewed merge.

Replace only exp_06 `code.summarize_haa` with:

```text
deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a
```

Replacing that single digest in the current exp_06 record, preserving every other byte, yields a 3,190-byte record with SHA-256:

```text
80134832646672411740f779835fa7c06f8bb354b46aab558884503b39a1b666
```

The complete exp_11 candidate fill is:

```json
{
  "schema_version": 1,
  "code": {
    "train": "d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571",
    "finalize": "3eb15b59b672d97ac257054e66f8bc835abd27b85e24f9a417a5a1aaa9e25b9e",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "737414cc75bd89e11c045c203f26be02a9c9f4fed7b923b565adce8ade4bcd17",
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

HEAD and repository status remained unchanged, including the pre-existing untracked review prompt. No review file or repository changes were saved.
tokens used
89,358
**Reviewer:** OpenAI Codex, GPT-6 per available developer metadata. The exact invocation model identifier and reasoning-effort setting are not exposed; I cannot attest to a more specific setting.

**Sandbox:** Read-only worktree; test/fixture writes confined to `/tmp`; CUDA disabled; numerical threads limited to one per test process; no process signals. Selected pytest runs used worktree `PYTHONPATH`, `PYTHONDONTWRITEBYTECODE=1`, and `-p no:cacheprovider`.

**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`.  
**HEAD:** `612e3dd023eb390b8a818d307a1b85d821b8ed4f`.  
**Review window:** 2026-09-27, approximately 05:20–05:25 EDT / 09:20–09:25 UTC.  
**Range:** All eight commits in `2c64a2a..612e3dd`.

**Verdict: request changes. Round 2 cannot be CLOSED, and `612e3dd` should not be merged.** The receipt and ceiling blockers are resolved. Canonicalization fixes the original alias-basename defect, but recovery remains unsafe.

1. **[P2] Failed recovery can invalidate an already published attempt; concurrent recovery can then report success while publishing a dangling link.**  
   [tools/exp11_launch.sh:383](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:383), [promotion:172](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:172), [recovery ownership:250](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:250).

   With `final -> attempt_A`, recovery through `--attempt final` now correctly resolves `attempt_A`. However, a missing `--log` makes the real finalizer refuse, after which line 384 renames **the canonical directory** to `attempt_A_ABORTED_finalize_refused`. `final` remains dangling, and any existing completion moves with the directory.

   Independent reproduction:

   ```text
   recovery exit:       2
   final is symlink:    true
   final target exists: false
   canonical moved:    true
   cause: EXP11_FINALIZE_REFUSED unreadable log
   ```

   This is a new consequence of passing the resolved directory into `abort`. A malformed recovery invocation does not establish that the previously completed run failed.

   Recovery also has no lock or ownership marker. A controlled interleaving paused one recovery after a successful-finalization stub, let another recovery’s real missing-log refusal rename the shared attempt, then resumed the first:

   ```text
   first recovery exit:  0
   second recovery exit: 2
   final target:         attempt_A
   final target exists:  false
   ```

   Separately, with `final -> attempt_A` and an external alias resolving to `attempt_B`, promotion returned success and replaced `final` with `attempt_B`. `mv -Tf` provides atomic replacement, but no conflict protection or recovery serialization.

   **Required correction:** Make refused recovery preserve existing attempts and logs. Serialize alias resolution, validation and publication per arm, with full-launch promotion participating in the same coordination. Make same-target recovery idempotent and explicitly handle a different existing `final`; silent replacement should not be an accidental recovery policy. Add mutation tests for these failure and concurrency cases.

   These reproductions ran the current launcher functions and literal recovery branch against `/tmp` fixtures. Preflight was stubbed for temporary roots; successful expensive validation was stubbed where stated. Missing-log refusals used the real finalizer, which fails before checkpoint validation.

The requested fixes otherwise verify as follows:

| Item | Independent result |
|---|---|
| Consumed exp_06 approval identity | **Resolved.** The receipt is required at runtime; omission refuses despite the signature’s `None` default. No default-path fallback remains. |
| `main()` wiring and published inputs | `main()` loads `args.approved`, passes `approvals_receipt`, and the check binds that consumed path/digest into inputs. Call site independently asserted using AST. |
| Alternate committed record | A genuinely committed temporary record with changed artifact bytes was refused against the default record’s pin. |
| Byte-identical alternate path | A committed copy with identical bytes was admitted, binding its own resolved path. |
| Ceiling | Bounded decimal-string validation precedes arithmetic: no leading zeros, at most six digits, then `(0,129600]`. |
| Canonical recovery path | Realpath, direct-parent arm root, `attempt_*` basename and arm profile are checked. Existing-`final` and external aliases use the canonical finalizer/promotion path. Failure ownership remains the blocker above. |
| Root override | `EXP11_PRETRAIN_ROOT` is honored only with both `EXP11_TEST_ROOTS=1` and `--dry-run`; other nonempty overrides refuse. |

The independent ceiling dry-runs produced:

```text
9223372036854775808                       exit 2
999999999999999999999999999999999999999  exit 2
129600                                  exit 0; timeout uses 129600
129601                                  exit 2
```

None emitted `integer expression expected`. Zero, leading-zero values and seven-digit values also refused.

The record-local [.gitattributes](/home/yixunhu/codespace/xRIR_code_wt/worklog/worklog_yixun/exp_11_orientation_cue_fairness_claude/.gitattributes:5) exemption is **acceptable**. It preserves copied review bytes and applies only to matching review Markdown under that record directory. Source, tests, the coder report and sibling records remain outside the exemption. Both `git diff --check 2c64a2a..612e3dd` and `c8acda2..612e3dd` pass. The broader `9f98bbb` range still has historical copied-log whitespace.

The qualified red-first account is supported by replay:

| Historical selection | Observed result |
|---|---|
| `8506456`: alternate-record and `main()` call-site tests | Both fail intended contract assertions: missing receipt parameter and absent receipt forwarding. |
| `a6b3200`: six ceiling cases plus existing-`final` alias | Four intended failures, three passes. Oversized integers and leading-zero input pass incorrectly; alias promotion prints `final -> final`. |
| `e98b490`: cross-arm recovery test | Fails on `NameError: json is not defined` before recovery executes. |

These were historical changed files replayed against current dependencies, not complete historical environments.

Minor report corrections are nonblocking: the committed unit test’s “committed” alternate fixture is not itself Git-committed—my reproduction was; the call-site RED test fails its own assertion; **two** exp_11 keys changed against `2c64a2a`, not three; `3119e22` is bookkeeping, while `0b10d7c` is the last production-code commit.

Closure hygiene **passes**:

- Exactly **one of twenty exp_06 keys** differs from the `9f98bbb` approvals: `code.summarize_haa`.
- Historical closure reconstruction independently reproduces the old digest; the other nineteen current keys match.
- All four simulated-evaluation closures—`eval`, `eval_launch`, `compare`, `mirror_probe`—are unchanged.
- Frozen production files are untouched. The only modified existing production source since `9f98bbb` is `tools/exp06_summarize_haa.py`.
- `BACKBONES_EXP06` is unchanged; registry digest remains `f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169`.
- Pinned `c6233e3` exp_06/exp_09 and `c491396` exp_11 Phase-1 oracle comparisons reproduce serialized statistical payloads and rendered summaries byte-for-byte.

Current selected verification totaled **100 passed, no skips**:

| Selection | Passed |
|---|---:|
| Shell | 40 |
| Approval enforcement | 13 |
| Finalizer and adapter zero-shot | 31 |
| Sim-eval closures, including ten actual manifests | 3 |
| Registry and pinned oracles | 13 |

Both shells pass `bash -n`. I did not rerun the full suite; the coder’s full-suite result remains reported evidence.

The following is the **exact fill for the unchanged `612e3dd` candidate tree**. Since recovery requires changes, it is not merge approval; recompute at the eventual reviewed merge.

Replace only exp_06 `code.summarize_haa` with:

```text
deff6d49d8e903d7e0d5005c5d40199b53ca35cdedfd71a90938df9815d0004a
```

Replacing that single digest in the current exp_06 record, preserving every other byte, yields a 3,190-byte record with SHA-256:

```text
80134832646672411740f779835fa7c06f8bb354b46aab558884503b39a1b666
```

The complete exp_11 candidate fill is:

```json
{
  "schema_version": 1,
  "code": {
    "train": "d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571",
    "finalize": "3eb15b59b672d97ac257054e66f8bc835abd27b85e24f9a417a5a1aaa9e25b9e",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "737414cc75bd89e11c045c203f26be02a9c9f4fed7b923b565adce8ade4bcd17",
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

HEAD and repository status remained unchanged, including the pre-existing untracked review prompt. No review file or repository changes were saved.

