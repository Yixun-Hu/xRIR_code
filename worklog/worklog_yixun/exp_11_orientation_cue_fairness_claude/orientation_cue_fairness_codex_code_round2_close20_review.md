**Reviewer:** OpenAI Codex, GPT-6; exact model ID and reasoning-effort setting are not exposed in the supplied invocation metadata.  
**Sandbox:** read-only; temporary test execution auto-approved.  
**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`  
**HEAD:** `4ea9ad6826900edd4eebd9bdbd6b2482961250de`  
**Time:** 2026-09-28 10:24:51 EDT

**Verdict: approve with changes — tests-only merge after the coverage correction below; no re-fill.**

1. **P2: The complementary admission test is not a complete exp_11 drift guard.** [test_exp11_finalize.py:262](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_finalize.py:262) invokes a gate that checks only `train`, `finalize`, `launch_sh`, and `smoke`. A change to `tools/exp11_haa_pipeline.sh` can leave this test green. An in-memory replay confirmed that a mismatched `haa_pipeline_sh` digest is admitted, while a mismatched `train` digest is refused. Add a comparison of **every filled committed code key** against its computed digest—all eight at this HEAD. Keep this separate from training admission so partial approvals remain meaningful.

2. **P3: Qualify the new signal explanation.** [test_exp11_shell.py:398](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:398) incorrectly attributes ignored SIGINT to `nohup`/`setsid` themselves and says TERM registers regardless. Replay confirmed:
   - Inherited ignored INT: `trap … INT` succeeds but `trap -p INT` remains `trap -- '' SIGINT`.
   - Default dispositions through `nohup setsid`: INT remains trappable.
   - Background execution without job control causes inherited ignored INT.
   - Inherited ignored TERM likewise cannot be trapped.

   TERM is operative for the recorded launch conditions. The production header and conditional `on_signal` behavior remain accurate; the new comments/report need qualification.

3. **Optional strengthening:** [test_exp11_profiles.py:169](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_profiles.py:169) checks that historical and current hashes differ, but independently verifies only the current hash. Also compare the historical hash with SHA256 of the first commit’s bytes.

The remaining checks pass:

- `git diff main HEAD -- tools/ model/` is empty. All **20 exp_06 + 8 exp_11** computed code digests match the records committed at HEAD. Branch-only changes are three tests plus records; the larger endpoint diff reflects independent changes on `main`.
- The committed all-null template exercises refusal correctly. Live shape, `committed_at`, repo-relative path, SHA256 identity, and both binding refusals remain. The temporary repository correctly retrieves the earlier null record despite a subsequently filled working file.
- Child SIGINT/SIGTERM resets are correct. Selected tests passed with **pytest itself entering with SIGINT ignored: 253 passed, 1 skipped**, in 205.28 seconds.
- The full-suite log starts with HEAD `464a943688123ef58e1b1f0a5af04e0c005027ad` and `/tmp/pytest-exp11-20260928-091024`; it ends with **4197 passed, 56 skipped, zero failures**.
- Cycle 21 correctly identifies `tests/test_exp06_launch.py:71`: move the fake into an absolute, colon-free temporary directory and add the colon-path regression. Grep confirms no exp_11 test prepends to `PATH`.

CPU affinity was limited to four CPUs, numerical thread limits were one, and CUDA was disabled. Review scratch directories were removed; repository status is unchanged. No review file was saved.
tokens used
60,788
**Reviewer:** OpenAI Codex, GPT-6; exact model ID and reasoning-effort setting are not exposed in the supplied invocation metadata.  
**Sandbox:** read-only; temporary test execution auto-approved.  
**Tree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue`  
**HEAD:** `4ea9ad6826900edd4eebd9bdbd6b2482961250de`  
**Time:** 2026-09-28 10:24:51 EDT

**Verdict: approve with changes — tests-only merge after the coverage correction below; no re-fill.**

1. **P2: The complementary admission test is not a complete exp_11 drift guard.** [test_exp11_finalize.py:262](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_finalize.py:262) invokes a gate that checks only `train`, `finalize`, `launch_sh`, and `smoke`. A change to `tools/exp11_haa_pipeline.sh` can leave this test green. An in-memory replay confirmed that a mismatched `haa_pipeline_sh` digest is admitted, while a mismatched `train` digest is refused. Add a comparison of **every filled committed code key** against its computed digest—all eight at this HEAD. Keep this separate from training admission so partial approvals remain meaningful.

2. **P3: Qualify the new signal explanation.** [test_exp11_shell.py:398](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_shell.py:398) incorrectly attributes ignored SIGINT to `nohup`/`setsid` themselves and says TERM registers regardless. Replay confirmed:
   - Inherited ignored INT: `trap … INT` succeeds but `trap -p INT` remains `trap -- '' SIGINT`.
   - Default dispositions through `nohup setsid`: INT remains trappable.
   - Background execution without job control causes inherited ignored INT.
   - Inherited ignored TERM likewise cannot be trapped.

   TERM is operative for the recorded launch conditions. The production header and conditional `on_signal` behavior remain accurate; the new comments/report need qualification.

3. **Optional strengthening:** [test_exp11_profiles.py:169](/home/yixunhu/codespace/xRIR_code_wt/tests/test_exp11_profiles.py:169) checks that historical and current hashes differ, but independently verifies only the current hash. Also compare the historical hash with SHA256 of the first commit’s bytes.

The remaining checks pass:

- `git diff main HEAD -- tools/ model/` is empty. All **20 exp_06 + 8 exp_11** computed code digests match the records committed at HEAD. Branch-only changes are three tests plus records; the larger endpoint diff reflects independent changes on `main`.
- The committed all-null template exercises refusal correctly. Live shape, `committed_at`, repo-relative path, SHA256 identity, and both binding refusals remain. The temporary repository correctly retrieves the earlier null record despite a subsequently filled working file.
- Child SIGINT/SIGTERM resets are correct. Selected tests passed with **pytest itself entering with SIGINT ignored: 253 passed, 1 skipped**, in 205.28 seconds.
- The full-suite log starts with HEAD `464a943688123ef58e1b1f0a5af04e0c005027ad` and `/tmp/pytest-exp11-20260928-091024`; it ends with **4197 passed, 56 skipped, zero failures**.
- Cycle 21 correctly identifies `tests/test_exp06_launch.py:71`: move the fake into an absolute, colon-free temporary directory and add the colon-path regression. Grep confirms no exp_11 test prepends to `PATH`.

CPU affinity was limited to four CPUs, numerical thread limits were one, and CUDA was disabled. Review scratch directories were removed; repository status is unchanged. No review file was saved.

