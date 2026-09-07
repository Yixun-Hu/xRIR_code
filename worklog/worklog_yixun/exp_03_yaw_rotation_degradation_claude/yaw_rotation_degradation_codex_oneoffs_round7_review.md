# Codex review — Planner one-offs and final documents, round 7

Reviewer: OpenAI Codex CLI (`codex exec -s read-only`, model gpt-5.6-sol, reasoning ultra), independent of the Planner (Claude) and Coder (Claude Opus subagent).
Prompt: review_prompts/oneoffs_round7_prompt.md · started 2026-09-06 23:29:09

---

Verdict: **request changes — do not commit yet.**

The confirmatory numbers, verdicts, and current canonical/copy bytes are consistent. However, the binding and acceptance implementations still contain reproducible fail-closed defects, so the overall scientific evidence path is not yet defect-free.

### Round-6 closure table

| Finding | Round-7 status | Assessment |
|---|---|---|
| 1. Immutable, transactional provenance binding | **Partial — blocking** | Recursive drift detection and full projection before writes are present. However, `last_migration` is omitted from the next projection, so the first identical bind after a migration refuses it as a removal. An allowed sidecar addition also changes `sidecar_sha256`, which report drift rejects. The retained record does not preserve the original v3 binding timestamps/history. See [bind_provenance.py](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/bind_provenance.py:188). |
| 2. Clean launcher workflow | **De-scoped, not closed** | The header accurately excludes the launcher from the evidence path. It still does not provide an executable clean workflow: copy operations are prose, the binding-report copy is not refreshed after binding, and the decision producer remains working-tree/manual. See [launch_yaw_sweep_v3.sh](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/launch_yaw_sweep_v3.sh:3). |
| 3. Checkpoint/log identity and preflight | **Partial — blocking** | All current checkpoint identities, digests, and canonical log associations are correct, and the launcher uses full validation. But acceptance only compares `run_start` when it is truthy, so missing, null, empty, or `False` values pass. See [check_sweep_acceptance.py](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/check_sweep_acceptance.py:244). |
| 4. Renderer fail-open behavior/presentation | **Partial — auxiliary** | The retained Markdown, HTML, and `data.json` regenerate byte-identically and contain the complete current grids. General coverage remains fail-open for deletion of a single row, a whole TOST payload represented by `null`, or empty final bound/verdict payloads. Raw/truncated spectral identifiers also remain. See [make_results_md.py](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_md.py:36) and [make_results_html.py](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_results_assets/make_results_html.py:81). |
| 5. Residual disclosures | **Closed** | The post-hoc reconstruction, launcher status, CUDA nondeterminism, and unserialized denominator are substantively disclosed in [the analysis](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_analysis.md:13), params, and command record. |
| 6. Command/worklog completeness | **Partial — record blocker** | The recent substantive worklog entries omit required SOP fields and combine multiple actions. The static-check command uses literal placeholders such as `...` and “whitespace scan,” while its retained log does not echo exact commands or return codes. See [worklog](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_worklog.md:428) and [command record](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_command.md:143). |
| 7. Narrow acceptance paths | **Partial — blocking** | All decision copies, closure digests, and recomputation exit status are now checked. But `inventory_missing=False` passes as zero, and gate metadata never requires `batch_canonical is True`; missing/false/integer forms can therefore pass. |

### Scientific evidence-path decision

**Yes, defects remain in the scientific evidence path**, specifically in binding and acceptance.

I found no discrepancy in:

- The confirmatory H1/H2 numbers or verdicts.
- The four canonical/copy pairs: full statistics, summary, gate decision, and binding report.
- The current 13 sidecars, their report records, checkpoints, metrics, and log digests.
- The retained recomputing acceptance result in [the final acceptance log](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude/yaw_rotation_degradation_2026-09-06_23:27:20_acceptance_v6_final.log:1).

Nevertheless, the binder cannot reliably perform and then revalidate the migrations it claims to support, and the acceptance checker admits malformed provenance/data metadata. Passing current inputs do not cure those validator defects.

### Auxiliary tooling

| Area | Remaining item | Classification |
|---|---|---|
| Launcher | Missing executable copy/refresh sequence and dependence on working-tree/manual steps | **Auxiliary defect**, not a defect in the historical evidence |
| Launcher | Never exercised for the retained runs; real evaluator, rename, and injected-failure branches untested | **Residual to record** |
| Renderer | Incomplete row/payload grids can still render as `FINAL`; HTML hardcodes current label keys | **Auxiliary defect** |
| Renderer presentation | Some detailed Markdown columns retain raw or truncated metric identifiers | **Presentation defect** |
| Binder tests | No real migration→identical-rerun, history-preservation, or zero-write-on-refusal integration test | **Blocking coverage defect** |
| Acceptance tests | Missing adversarial cases for `run_start`, boolean counts, and `batch_canonical` | **Blocking coverage defect** |
| Renderer/launcher tests | Missing single-row/payload corruption tests and clean end-to-end launcher exercise | **Auxiliary coverage residual**, provided it is explicitly recorded |
| Binder crash atomicity | Files are replaced sequentially after validation; an I/O failure can leave a detectable partial transaction | **Residual to record**, unless claiming crash-atomic behavior |

### Required before commit

1. Fix binder migration idempotence and authorize `sidecar_sha256` changes only when fully explained by an allowed additive sidecar migration.
2. Add filesystem-level binder tests covering migration, identical rerun, original timestamp/history preservation, and no writes after drift refusal.
3. Make acceptance require:
   - a nonempty exact `run_start`;
   - strict non-boolean integer inventory counts, including `inventory_missing == 0`;
   - `batch_canonical is True`.
4. Add adversarial acceptance tests for those cases.
5. Append SOP-complete corrective worklog entries, record the exact static commands and statuses, and retract the inaccurate history-preservation and launcher-order claims.
6. Explicitly disclose the already-lost pre-v6 binding timestamp/history; do not reconstruct precision unsupported by retained logs.
7. Retain fresh binder, recomputing acceptance, focused-test, full-suite, and exact static-check logs; refresh any changed report copy and reverify all four canonical/copy pairs.
8. Either fix the auxiliary launcher/renderer gaps or list them explicitly as open auxiliary defects/residuals.

No evaluator, sweep, or confirmatory-statistics rerun is required. I made no file changes.
codex exit 0 at 23:43:28
