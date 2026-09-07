**Reviewer:** OpenAI Codex (codex-cli 0.144.1, `codex exec`, read-only sandbox, model `gpt-5.6-sol`, reasoning effort `ultra`; session 01a07755-3c0d-7703-8266-33aae2e39919) · **Date:** 2026-09-06 11:37 (−04:00) · **Round:** Coder round 1 (`tools/yaw_rotation.py`, tests 1–8) · **Briefing:** SOP, plan v2, notebook incl. the Coder report, plan review, exp_01 results; commits fcac8be..362c76c; prompt `review_prompts/code_round1_prompt.md`.

> Planner answers to the questions: (1) commit trailers follow this session's attribution rule (the Coder is an Opus subagent running inside the Fable session; the notebook records the Coder identity per round); (2) the 50,696×9 delay-flip audit and the k=64 measurement were exploratory (no logged command) — labelled as such; the reproducible audit is the manifest-based one in round 3.

**Reviewer:** OpenAI Codex (GPT-5 API workspace agent; exact serving minor version not exposed, read-only review) · **Date:** 2026-09-06

## Summary verdict

**Approve with changes; keep round 1 open.**

The active-yaw implementation is correct: `roll(+k)` followed by \(R_z(+2\pi k/W)\) matches the dataset projection convention, including negative/modulo-\(W\) angles. Source, receiver, and reference views covary correctly; floating-point inputs retain dtype/device and are not mutated. `fixed_alignment` works for the planned serial, eager evaluation.

Validation passed under Python 3.8.20 with CUDA: **15 passed, 8 upstream warnings**. The commit range adds only the three approved files; `git diff --check` is clean.

## Blocking findings

1. [tools/yaw_rotation.py:131](/home/yixunhu/codespace/xRIR_code/tools/yaw_rotation.py:131) — `integer_delays` does not preserve the model’s dtype.

   `shift_and_align` produces `torch.int32` via `.int()`, but line 132 widens it to `int64`. Values and rounding order match, so this would not bias counts, but it violates the claimed exact parity.

   **Fix:** return `delay_unit` directly and assert `dtype == torch.int32`.

2. [tests/test_yaw_rotation.py:331](/home/yixunhu/codespace/xRIR_code/tests/test_yaw_rotation.py:331) — the central fixed-alignment isolation control from plan §6.6 is missing.

   Every tested forward is already pinned; there is no comparison of an ordinary k=0 forward against the same k=0 forward using cached alignment.

   **Fix:** run plain k=0, cache its alignment, run fixed k=0, and require `torch.equal`. Since k=W is reduced to identical inputs, strengthen [line 343](/home/yixunhu/codespace/xRIR_code/tests/test_yaw_rotation.py:343) to exact equality too.

3. [tests/test_yaw_rotation.py:190](/home/yixunhu/codespace/xRIR_code/tests/test_yaw_rotation.py:190) — model parity is tested only on synthetic coordinates.

   The “real-coordinate” test at [line 239](/home/yixunhu/codespace/xRIR_code/tests/test_yaw_rotation.py:239) merely calls the helper and prints an unasserted `0/160`; it never compares against `shift_and_align`, contrary to approved test 6.

   **Fix:** freeze a real query/reference pair, use an impulse or seeded reference signal, and compare the model’s shifted output with independently applied helper delays.

## Non-blocking

- `fixed_alignment` restores inherited/instance methods on normal exit, exceptions, and nesting. Inside the context, however, the replacement is a plain instance callable rather than a bound method, and overlapping contexts across threads/tasks are unsafe. Bind with `types.MethodType`, or document that it supports only serial eager evaluation.
- [tools/yaw_rotation.py:59](/home/yixunhu/codespace/xRIR_code/tools/yaw_rotation.py:59) overstates dtype support: integer vectors promote to float32. Require floating tensors or narrow the documentation.
- [tests/conftest.py:16](/home/yixunhu/codespace/xRIR_code/tests/conftest.py:16) hard-codes one user-specific data cache and suppresses the repository’s portable fallback.
- TDD was not strictly plan-wide red→green: only cycles A/C reportedly observed red, the oracle followed the implementation, and the first commit had 208 insertions. All new functions did receive pre-implementation red coverage, so this is a process deviation rather than a code blocker.

## Test gaps

- Add ±0.5/±1.5 ties-to-even delay cases and a CPU test for exact expression, dtype, and device parity.
- Add explicit float32/float64 CPU/CUDA rotation cases and invalid shape/non-integral `k`/non-positive `W` cases.
- Tighten the oracle from `atol=1e-4` with default `rtol` to approximately `atol=1e-5, rtol=0`; observed error is only \(2.2\times10^{-6}\).
- Add a direct negative-k oracle. The current negative test compares two calls sharing the same modulo path.
- CUDA skips are appropriate because baseline `apply_delay` hard-codes `.cuda()`; randomness is adequately seeded.

## Questions

1. All commit trailers attribute Claude Fable 5.1, while the worklog identifies the Coder as Opus/max effort. Did Fable only commit Opus-produced code, or does the provenance need correction?
2. Where are the reproducible command/log and per-angle counts for the reported 50,696×9 delay audit and k=64 measurement? If none exist, label them exploratory and redo them against the round-2 manifest.
