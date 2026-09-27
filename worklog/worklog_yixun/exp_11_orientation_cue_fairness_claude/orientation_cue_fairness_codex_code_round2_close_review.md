**Reviewer:** OpenAI Codex, GPT-6 per available metadata. The exact invocation model identifier and reasoning-effort setting are not exposed; I cannot attest to more specific values.

**Environment:** Read-only repository review; authorized fixture/cache writes confined to `/tmp`; CUDA disabled; CPU concurrency ≤4; no process signals. Selected pytest runs used `-p no:cacheprovider`, disabled bytecode writes, and `PYTHONPATH` set to the worktree.

**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`.  
**HEAD:** `2c64a2a2c327a75eedfd0a5b408f62d9bea8e78b`.  
**Time:** 2026-09-27, 03:54:30–04:00:16 EDT (07:54:30–08:00:16 UTC).  
**Range:** `c8acda2..2c64a2a`, **15 commits**, not 13. The pre-existing untracked close-review prompt remained unchanged.

**Verdict: request changes. Round 2 cannot be CLOSED, and `2c64a2a` should not be merged.** Three blockers remain.

1. **[P1] The exp_06 reuse check hashes the default record rather than the record selected by `--approved`.**  
   [tools/exp06_summarize_haa.py:1448](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1448), [tools/exp06_summarize_haa.py:2300](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:2300)

   `main()` loads `args.approved`, but `check_exp11_reused()` independently hashes `approvals_api.approved_path_default()`. Consequently, exp_11’s pin can match the default file while the summarizer consumes different committed approvals.

   My temporary committed fixture demonstrated:

   ```text
   Consumed alternate exp06 SHA: a99f44f91802…
   exp11-pinned default SHA:     eb6e58e856bd…
   Both producer checks:        accepted
   check_exp11_reused:           accepted
   expected_inits['cyl_or']:     alternate record's changed checkpoint digest
   ```

   This was a gate reproduction, not a complete publication run. It establishes that the newly added binding does not enforce the identity of the consumed record.

   Pass the loaded `approvals_receipt` into this check, compare its digest with `reused.approved_digests_exp06`, and bind that same consumed identity. Add a regression using an alternate committed `--approved` record with differing bytes.

2. **[P2] The ceiling check fails open on integers outside Bash’s numeric range.**  
   [tools/exp11_launch.sh:47](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:47)

   Independent full dry-runs with `EXP11_FULL_CEILING_S=9223372036854775808` and a 39-digit integer returned **0**, emitted `integer expression expected`, and printed:

   ```text
   timeout --kill-after=60 9223372036854775808 …
   ```

   Both comparisons error, so the `if` condition is false and validation succeeds. Zero, empty, and ordinary over-limit values now refuse correctly, but the required `(0,129600]` contract remains incomplete.

   Use bounded decimal-string validation or arbitrary-precision integer parsing, and make parsing errors refuse. Add an overflow regression.

3. **[P2] Recovery validates the resolved attempt but promotes the original path’s basename.**  
   [tools/exp11_launch.sh:65](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:65), [tools/exp11_launch.sh:345](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:345)

   The original cross-arm reproduction now refuses. However, a symlink resolving to a valid direct-child attempt passes `check_attempt`, after which promotion uses the unresolved basename.

   With `H-root/final -> attempt_review`, recovery through `--attempt H-root/final` returned **0** and printed:

   ```text
   ATTEMPT ok …/final arm=H profile=H_RECIPE
   PROMOTE …/final -> final
   ```

   Real promotion would replace `final` with a self-referencing link. An external alias to the same valid attempt similarly produced a dangling target under the arm root.

   Preserve the canonical resolved attempt through finalization and promotion, or reject aliases. Cover both existing-`final` and external-alias inputs.

   Also, [tools/exp11_launch.sh:27](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:27) reads `EXP11_PRETRAIN_ROOT` unconditionally. “Only for tests” is currently a convention: it changes real launch roots and preflight scan roots too. Restrict this override to the intended test/dry-run context.

The five original blockers were checked as follows:

| Original blocker | Verification and disposition |
|---|---|
| **1. Adapter zero-shot admission** | **Resolved.** Base mode requires `simple_adapter` and a valid declared initialization digest equal to the evaluated checkpoint digest. It requires exactly the pinned base parameter set, with no adapter state. Other cases remain strict. `checkpoint_mode` is recorded and required in child completion. |
| **2. Approval enforcement** | **Partially resolved.** Wrong producer, reused-record and receipt identities now refuse on the default path; the alternate exp_06 record gap above remains. |
| **3. Diagnostic receipt identity** | **Resolved.** `git_head` is required; runner closure, HEAD and exploratory status are compared with provenance through `diagnostic_evidence`. |
| **4. Ceiling** | **Partially resolved.** Only an unset variable defaults; empty, zero, negative, non-integer and ordinary excessive values refuse. Overflow still passes. |
| **5. Recovery arm/profile** | **Partially resolved.** Cross-arm and wrong-profile attempts refuse before finalization/promotion; canonical-path promotion remains incorrect. |

For blocker 1, the independent reproduction used the **actual historical checkpoints**, at `num_shot=8`:

```text
Parameter counts: base=266, adapter=268

J: 651e3a377e110022fbd89332ff1f1c2648ef398bd83840bf369b05eb5decbedf
   admitted, checkpoint_mode=base

K: f8e640523892154fe744b60e2fe99178b68a522ee3947758812aa5a7dc15b299
   admitted, checkpoint_mode=base
```

For both J and K, base-only evaluation payloads whose digests differed from their declared initialization were refused with the original **266-versus-268** mismatch. Tests also cover adapter state supplied in base mode, complete strict adapter checkpoints, missing initialization declarations, mode recording and job lineage. Relevant implementation: [tools/exp11_finalize.py:153](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_finalize.py:153).

For blocker 2, committed temporary fixtures independently confirmed rejection of an all-zero exp_11 producer digest, incorrect exp_04/exp_06 digests, an incorrect receipt digest, and a nonexistent receipt path. Matching records passed and all three consumed identities appeared in the assembled input bindings. The existing exp_06 producer check remains active.

I also executed the embedded K queue gate with controlled approval fixtures. Incorrect exp_04 and exp_06 pins refused **before** `exp04_aug_checkpoint` ran; matching pins resolved K once and passed. The approval loader was redirected to the fixture; the embedded gate and closure checks executed normally.

For blocker 3, separate mutations exercised the diagnostic dispatcher: wrong runner closure, wrong HEAD, missing HEAD and contradictory exploratory status each refused; the matching control passed. Relevant implementation: [tools/exp11_finalize.py:889](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_finalize.py:889).

**Process evidence is substantially improved, with one qualification.** Historical test/module replays produced:

| Change | RED commit and result | GREEN commit and result |
|---|---|---|
| Adapter admission | `c7a4602`: 8 failed, 2 passed | `3e64f6f`: 10 passed |
| Approval enforcement | `441db03`: 7 failed, 1 passed | `c8e6c4e`: 8 passed |
| Receipt identity | `ddfdfad`: 5 failed, 1 passed | `eea0cca`: 6 passed |
| Ceiling/recovery selection | `e98b490`: 9 failed, 5 passed | `72839af`: 14 passed |
| Adapter smoke assertion | `ef0dd3a`: 1 failed | `fca7a4f`: 1 passed |

These were historical changed modules/tests replayed against otherwise current dependencies, not complete historical installations.

The **three recovery tests in committed RED `e98b490` fail on `NameError: json is not defined` before exercising recovery**. The import arrives in GREEN. Adding only that import in a temporary historical copy exposes the intended cross-arm and wrong-profile failures. Thus the report’s blanket claim of meaningful red-first evidence for every blocker needs qualification. The other pairs establish the expected transitions; several begin with missing-API failures.

Both requested coder prompts are now tracked. All ten oversized production commits are explicitly recorded as exceptions:

`769a6ef`, `b59e234`, `3458146`, `b39b7d0`, `216aef3`, `c7b1d7b`, `c0b10a9`, `8f5ff9a`, `e978c2e`, `e7cb7f4`.

That disclosure is acceptable. Some reported counts mix production changes with tests/generated records; notably, `e978c2e` contains **401 production insertions**, independently exceeding the limit. The last production commit is `fca7a4f`; `58bd991` is report-only.

The bundled adapter rungs **(e)/(f)** are acceptable additions: they use `simple_adapter`, the adapter-heading flag, the training checkpoint, and a passed-completion gate before evaluation. Regenerating the H golden is appropriate. Replacing `launch_finalize_I`’s nonexistent-attempt golden with the real acceptance fixture is also appropriate; [tests/test_exp11_shell.py:204](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:204) checks preflight, finalization arguments and promotion.

Those adapter smokes start from a **complete adapter fixture**, so they exercise strict adapter loading. They do not themselves demonstrate the base initialization path; that path has the separate admission reproductions above.

**Closure hygiene passes at this tip.**

- Against `9f98bbb`, exactly **one of 20 exp_06 keys** changed: `code.summarize_haa`. Its computed value is `2868e66c116752f205af1299408998b0c3fcebbfc61e26878a0f4a7ac1ff614f`.
- The other nineteen exp_06 keys match, including the four simulated-evaluation closures: `eval`, `eval_launch`, `compare`, `mirror_probe`.
- Frozen production files are untouched. The only modified existing production source is `tools/exp06_summarize_haa.py`; the other production additions are exp_11 modules.
- `BACKBONES_EXP06` remains unchanged, with registry digest `f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169`.
- `tests/test_exp09_sim_eval_closures.py` passes, including checks of the ten actual simulated-evaluation manifests.
- Pinned exp_06/exp_09 oracle comparisons against `c6233e3` and exp_11 Phase-1 comparisons against `c491396` reproduce serialized analysis payloads and rendered summaries byte-for-byte.

Current-tip selected verification totaled **165 passed, no skips**:

| Selection | Result |
|---|---:|
| Approval enforcement, profiles, exp_11 summarizer | 96 passed |
| Adapter zero-shot and finalizer | 31 passed |
| Shell | 27 passed |
| Sim-eval closures, registry, Phase-1 oracle | 11 passed |

Both shells pass `bash -n`; production/test changes pass `git diff --check`. The whole-range whitespace check reports only copied prior-review Markdown whitespace and its trailing blank line. That record also contains duplicated review text.

I did not rerun the full suite or signaling tests. The coder’s full-suite result remains reported evidence.

The following is the **exact candidate-tree refill manifest**, independently computed at `2c64a2a`. Because this candidate requires changes, these values are **not authorization to fill approvals or merge**. Recompute them after the fixes at the actual reviewed merge.

Exp_06: replace only `code.summarize_haa` with:

```text
2868e66c116752f205af1299408998b0c3fcebbfc61e26878a0f4a7ac1ff614f
```

Preserving every other byte of the current exp_06 record yields SHA-256:

```text
e19f022762dacf36128bdb073d8252982ddf5c243873fb26ba87cd4004eee7eb
```

That is the refilled-record digest used below. Reserializing the record with different whitespace requires hashing the resulting bytes again.

Exp_11’s complete candidate fill is:

```json
{
  "schema_version": 1,
  "code": {
    "train": "d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571",
    "finalize": "3eb15b59b672d97ac257054e66f8bc835abd27b85e24f9a417a5a1aaa9e25b9e",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "0bdbb5d3d5c291bb2d0e10a9cb09d3ebd66c7078b28f56118e697c4eef4b44cd",
    "smoke": "62133ba07a35d42026c19fd1484b37c6c06ace0ebedc1cdd8484690df323450e",
    "summarize_haa": "2868e66c116752f205af1299408998b0c3fcebbfc61e26878a0f4a7ac1ff614f"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "e19f022762dacf36128bdb073d8252982ddf5c243873fb26ba87cd4004eee7eb",
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

HEAD and repository status remained unchanged. No review file or repository changes were saved.
tokens used
97,592
**Reviewer:** OpenAI Codex, GPT-6 per available metadata. The exact invocation model identifier and reasoning-effort setting are not exposed; I cannot attest to more specific values.

**Environment:** Read-only repository review; authorized fixture/cache writes confined to `/tmp`; CUDA disabled; CPU concurrency ≤4; no process signals. Selected pytest runs used `-p no:cacheprovider`, disabled bytecode writes, and `PYTHONPATH` set to the worktree.

**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`.  
**HEAD:** `2c64a2a2c327a75eedfd0a5b408f62d9bea8e78b`.  
**Time:** 2026-09-27, 03:54:30–04:00:16 EDT (07:54:30–08:00:16 UTC).  
**Range:** `c8acda2..2c64a2a`, **15 commits**, not 13. The pre-existing untracked close-review prompt remained unchanged.

**Verdict: request changes. Round 2 cannot be CLOSED, and `2c64a2a` should not be merged.** Three blockers remain.

1. **[P1] The exp_06 reuse check hashes the default record rather than the record selected by `--approved`.**  
   [tools/exp06_summarize_haa.py:1448](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:1448), [tools/exp06_summarize_haa.py:2300](/home/yixunhu/codespace/xRIR_code_wt/tools/exp06_summarize_haa.py:2300)

   `main()` loads `args.approved`, but `check_exp11_reused()` independently hashes `approvals_api.approved_path_default()`. Consequently, exp_11’s pin can match the default file while the summarizer consumes different committed approvals.

   My temporary committed fixture demonstrated:

   ```text
   Consumed alternate exp06 SHA: a99f44f91802…
   exp11-pinned default SHA:     eb6e58e856bd…
   Both producer checks:        accepted
   check_exp11_reused:           accepted
   expected_inits['cyl_or']:     alternate record's changed checkpoint digest
   ```

   This was a gate reproduction, not a complete publication run. It establishes that the newly added binding does not enforce the identity of the consumed record.

   Pass the loaded `approvals_receipt` into this check, compare its digest with `reused.approved_digests_exp06`, and bind that same consumed identity. Add a regression using an alternate committed `--approved` record with differing bytes.

2. **[P2] The ceiling check fails open on integers outside Bash’s numeric range.**  
   [tools/exp11_launch.sh:47](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:47)

   Independent full dry-runs with `EXP11_FULL_CEILING_S=9223372036854775808` and a 39-digit integer returned **0**, emitted `integer expression expected`, and printed:

   ```text
   timeout --kill-after=60 9223372036854775808 …
   ```

   Both comparisons error, so the `if` condition is false and validation succeeds. Zero, empty, and ordinary over-limit values now refuse correctly, but the required `(0,129600]` contract remains incomplete.

   Use bounded decimal-string validation or arbitrary-precision integer parsing, and make parsing errors refuse. Add an overflow regression.

3. **[P2] Recovery validates the resolved attempt but promotes the original path’s basename.**  
   [tools/exp11_launch.sh:65](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:65), [tools/exp11_launch.sh:345](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:345)

   The original cross-arm reproduction now refuses. However, a symlink resolving to a valid direct-child attempt passes `check_attempt`, after which promotion uses the unresolved basename.

   With `H-root/final -> attempt_review`, recovery through `--attempt H-root/final` returned **0** and printed:

   ```text
   ATTEMPT ok …/final arm=H profile=H_RECIPE
   PROMOTE …/final -> final
   ```

   Real promotion would replace `final` with a self-referencing link. An external alias to the same valid attempt similarly produced a dangling target under the arm root.

   Preserve the canonical resolved attempt through finalization and promotion, or reject aliases. Cover both existing-`final` and external-alias inputs.

   Also, [tools/exp11_launch.sh:27](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_launch.sh:27) reads `EXP11_PRETRAIN_ROOT` unconditionally. “Only for tests” is currently a convention: it changes real launch roots and preflight scan roots too. Restrict this override to the intended test/dry-run context.

The five original blockers were checked as follows:

| Original blocker | Verification and disposition |
|---|---|
| **1. Adapter zero-shot admission** | **Resolved.** Base mode requires `simple_adapter` and a valid declared initialization digest equal to the evaluated checkpoint digest. It requires exactly the pinned base parameter set, with no adapter state. Other cases remain strict. `checkpoint_mode` is recorded and required in child completion. |
| **2. Approval enforcement** | **Partially resolved.** Wrong producer, reused-record and receipt identities now refuse on the default path; the alternate exp_06 record gap above remains. |
| **3. Diagnostic receipt identity** | **Resolved.** `git_head` is required; runner closure, HEAD and exploratory status are compared with provenance through `diagnostic_evidence`. |
| **4. Ceiling** | **Partially resolved.** Only an unset variable defaults; empty, zero, negative, non-integer and ordinary excessive values refuse. Overflow still passes. |
| **5. Recovery arm/profile** | **Partially resolved.** Cross-arm and wrong-profile attempts refuse before finalization/promotion; canonical-path promotion remains incorrect. |

For blocker 1, the independent reproduction used the **actual historical checkpoints**, at `num_shot=8`:

```text
Parameter counts: base=266, adapter=268

J: 651e3a377e110022fbd89332ff1f1c2648ef398bd83840bf369b05eb5decbedf
   admitted, checkpoint_mode=base

K: f8e640523892154fe744b60e2fe99178b68a522ee3947758812aa5a7dc15b299
   admitted, checkpoint_mode=base
```

For both J and K, base-only evaluation payloads whose digests differed from their declared initialization were refused with the original **266-versus-268** mismatch. Tests also cover adapter state supplied in base mode, complete strict adapter checkpoints, missing initialization declarations, mode recording and job lineage. Relevant implementation: [tools/exp11_finalize.py:153](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_finalize.py:153).

For blocker 2, committed temporary fixtures independently confirmed rejection of an all-zero exp_11 producer digest, incorrect exp_04/exp_06 digests, an incorrect receipt digest, and a nonexistent receipt path. Matching records passed and all three consumed identities appeared in the assembled input bindings. The existing exp_06 producer check remains active.

I also executed the embedded K queue gate with controlled approval fixtures. Incorrect exp_04 and exp_06 pins refused **before** `exp04_aug_checkpoint` ran; matching pins resolved K once and passed. The approval loader was redirected to the fixture; the embedded gate and closure checks executed normally.

For blocker 3, separate mutations exercised the diagnostic dispatcher: wrong runner closure, wrong HEAD, missing HEAD and contradictory exploratory status each refused; the matching control passed. Relevant implementation: [tools/exp11_finalize.py:889](/home/yixunhu/codespace/xRIR_code_wt/tools/exp11_finalize.py:889).

**Process evidence is substantially improved, with one qualification.** Historical test/module replays produced:

| Change | RED commit and result | GREEN commit and result |
|---|---|---|
| Adapter admission | `c7a4602`: 8 failed, 2 passed | `3e64f6f`: 10 passed |
| Approval enforcement | `441db03`: 7 failed, 1 passed | `c8e6c4e`: 8 passed |
| Receipt identity | `ddfdfad`: 5 failed, 1 passed | `eea0cca`: 6 passed |
| Ceiling/recovery selection | `e98b490`: 9 failed, 5 passed | `72839af`: 14 passed |
| Adapter smoke assertion | `ef0dd3a`: 1 failed | `fca7a4f`: 1 passed |

These were historical changed modules/tests replayed against otherwise current dependencies, not complete historical installations.

The **three recovery tests in committed RED `e98b490` fail on `NameError: json is not defined` before exercising recovery**. The import arrives in GREEN. Adding only that import in a temporary historical copy exposes the intended cross-arm and wrong-profile failures. Thus the report’s blanket claim of meaningful red-first evidence for every blocker needs qualification. The other pairs establish the expected transitions; several begin with missing-API failures.

Both requested coder prompts are now tracked. All ten oversized production commits are explicitly recorded as exceptions:

`769a6ef`, `b59e234`, `3458146`, `b39b7d0`, `216aef3`, `c7b1d7b`, `c0b10a9`, `8f5ff9a`, `e978c2e`, `e7cb7f4`.

That disclosure is acceptable. Some reported counts mix production changes with tests/generated records; notably, `e978c2e` contains **401 production insertions**, independently exceeding the limit. The last production commit is `fca7a4f`; `58bd991` is report-only.

The bundled adapter rungs **(e)/(f)** are acceptable additions: they use `simple_adapter`, the adapter-heading flag, the training checkpoint, and a passed-completion gate before evaluation. Regenerating the H golden is appropriate. Replacing `launch_finalize_I`’s nonexistent-attempt golden with the real acceptance fixture is also appropriate; [tests/test_exp11_shell.py:204](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:204) checks preflight, finalization arguments and promotion.

Those adapter smokes start from a **complete adapter fixture**, so they exercise strict adapter loading. They do not themselves demonstrate the base initialization path; that path has the separate admission reproductions above.

**Closure hygiene passes at this tip.**

- Against `9f98bbb`, exactly **one of 20 exp_06 keys** changed: `code.summarize_haa`. Its computed value is `2868e66c116752f205af1299408998b0c3fcebbfc61e26878a0f4a7ac1ff614f`.
- The other nineteen exp_06 keys match, including the four simulated-evaluation closures: `eval`, `eval_launch`, `compare`, `mirror_probe`.
- Frozen production files are untouched. The only modified existing production source is `tools/exp06_summarize_haa.py`; the other production additions are exp_11 modules.
- `BACKBONES_EXP06` remains unchanged, with registry digest `f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169`.
- `tests/test_exp09_sim_eval_closures.py` passes, including checks of the ten actual simulated-evaluation manifests.
- Pinned exp_06/exp_09 oracle comparisons against `c6233e3` and exp_11 Phase-1 comparisons against `c491396` reproduce serialized analysis payloads and rendered summaries byte-for-byte.

Current-tip selected verification totaled **165 passed, no skips**:

| Selection | Result |
|---|---:|
| Approval enforcement, profiles, exp_11 summarizer | 96 passed |
| Adapter zero-shot and finalizer | 31 passed |
| Shell | 27 passed |
| Sim-eval closures, registry, Phase-1 oracle | 11 passed |

Both shells pass `bash -n`; production/test changes pass `git diff --check`. The whole-range whitespace check reports only copied prior-review Markdown whitespace and its trailing blank line. That record also contains duplicated review text.

I did not rerun the full suite or signaling tests. The coder’s full-suite result remains reported evidence.

The following is the **exact candidate-tree refill manifest**, independently computed at `2c64a2a`. Because this candidate requires changes, these values are **not authorization to fill approvals or merge**. Recompute them after the fixes at the actual reviewed merge.

Exp_06: replace only `code.summarize_haa` with:

```text
2868e66c116752f205af1299408998b0c3fcebbfc61e26878a0f4a7ac1ff614f
```

Preserving every other byte of the current exp_06 record yields SHA-256:

```text
e19f022762dacf36128bdb073d8252982ddf5c243873fb26ba87cd4004eee7eb
```

That is the refilled-record digest used below. Reserializing the record with different whitespace requires hashing the resulting bytes again.

Exp_11’s complete candidate fill is:

```json
{
  "schema_version": 1,
  "code": {
    "train": "d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571",
    "finalize": "3eb15b59b672d97ac257054e66f8bc835abd27b85e24f9a417a5a1aaa9e25b9e",
    "haa_finetune": "bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150",
    "haa_eval": "f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20",
    "haa_pipeline_sh": "3ebc8e156177796c1ab99808273cda1c88004e5292fb23e6192fb123705d65a4",
    "launch_sh": "0bdbb5d3d5c291bb2d0e10a9cb09d3ebd66c7078b28f56118e697c4eef4b44cd",
    "smoke": "62133ba07a35d42026c19fd1484b37c6c06ace0ebedc1cdd8484690df323450e",
    "summarize_haa": "2868e66c116752f205af1299408998b0c3fcebbfc61e26878a0f4a7ac1ff614f"
  },
  "reused": {
    "approved_digests_exp04": "2e4521173bb230fc8114116ba0c94b4afd0ccdd3b0b034faaf54516461ee0314",
    "approved_digests_exp06": "e19f022762dacf36128bdb073d8252982ddf5c243873fb26ba87cd4004eee7eb",
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

HEAD and repository status remained unchanged. No review file or repository changes were saved.

