**Reviewer:** OpenAI Codex `gpt-6-astra`, codex-cli 0.154.0, `codex exec`  
**Reasoning effort:** `ultra`, confirmed by this invocation’s log  
**Sandbox/resources:** read-only; CUDA disabled; CPU computations limited to one thread each; no process signals  
**Tree:** `/home/yixunhu/codespace/xRIR_code`, branch `main`; existing worklog modifications and untracked records  
**HEAD:** `c6233e370c0d355ad00e438e2db91092e76a132d`, unchanged during review  
**Date/time:** September 26, 2026, approximately 17:44–17:48 EDT  

**Verdict: approve with changes.**

V2 resolves the substantive round-1 scientific objections and chooses the right boundary for the new implementation. The remaining changes below concern precise adapter contracts, statistical edge cases, and executable provenance routing. They do not require additional experimental arms or a redesign.

I read the SOP, v2, superseded v1, round-1 review, query and notebook, relevant predecessor records, and implementation sources. The announcements directory is empty.

| Round-1 item | Round-2 assessment |
|---|---|
| Positional-code redundancy claim; D’s label | **Resolved.** Information availability is distinguished from architectural equivalence; D is an implicit-conditioning control. |
| H/I as literal tests; primary P3 = C − H | **Resolved.** H consistently occupies the requested explicit-cue baseline row; I is secondary. |
| J/K adapter design | **Partially resolved.** Technically feasible and correctly registered as a different design; clarify loading, angle/routing, and interpretation as change 1. |
| Yaw invariance argument; G ≈ E; “refutation” | **Resolved in substance.** Fine-tuning can recover orientation dependence; equivalence is a hypothesis; D − A no longer refutes explicit conditioning. Remaining causal wording is covered by change 1. |
| Notation, N1, N2 direction, separate superiority | **Resolved.** The inequalities and separation of claims are correct. |
| N1i interaction | **Partially resolved.** Correct estimand and historical disclosure; specify four-arm cohort construction and unavailable-cell handling. |
| N3; P1/P2 equivalence | **Resolved.** Strict 95% containment is explicitly chosen and correctly distinguished from conventional TOST. |
| Three S1 families; Q1–Q4; P3/P3′; P4/P4′ | **Resolved statistically.** Only P3 establishes the headline; other comparisons need the stated conditional interpretations. |
| Void/unconverged suppression | **Resolved as policy; partially resolved in implementation specification.** Historical screen helpers do not implement the proposed universal suppression. |
| R1 historical copying | **Partially resolved.** Correct sources and prohibition on recomputation, but some promised source fields do not exist. |
| Single overall reading; fixed publication rows | **Resolved.** No result-dependent replacement of D/G or substitution of A′ for A. |
| Round-1 pipeline, resolver, `expected_inits`, schema regression, census | **Resolved in design.** The proposed changes are minimal and appropriately tested. |
| Exp_11-owned round-2 family | **Partially resolved.** Correct ownership boundary; remaining routing and helper-composition gaps are changes 4–5. |
| Historical completion preservation | **Resolved for preserving current HAA/sim admission.** Unqualified replay of *every* historical completion remains **open and explicitly deferred**, because of pre-existing C-pretraining drift. |
| A/B receipt; C/D/E/F completions; A′ external row | **Resolved.** These are the correct admission paths. |
| Separate phase outputs; historical-output semantics | **Resolved.** Distinct paths avoid overwrite; regression expectations correctly exclude changing producer metadata. |
| H_RECIPE/I_RECIPE and completion contract | **Resolved in specification.** Approval and diagnostic lifecycle details remain to be pinned. |
| Cost, schedule, risks, namespace, reviewer effort | **Resolved.** Estimates are credible allowances; exp_11 avoids the peer’s namespace; this review uses `ultra`. |

The required changes are:

1. **Finish the adapter contract and restrict its scientific interpretation.**

   The proposed insertion after the patch embedding’s final LayerNorm is sound. It avoids the joint geometry/azimuth normalization change identified in round 1. Zero **weight and bias** give 1,536 added parameters at dimension 512 and can preserve the initial function. A bounded CPU prototype using the actual SimpleViT components produced equal outputs; this establishes feasibility, not correctness of the future implementation. See the original [embedding and forward path](/home/yixunhu/codespace/xRIR_code/model/simple_vit.py:129).

   Pin the token-angle convention. A natural choice for the 16 token columns is
   \[
   \theta_j=2\pi(j+\tfrac12)/16-\pi.
   \]
   Convert the bound `phi_deg` to radians. All four current records have `phi_deg=-90`, `k=128`, and confirmatory admissibility, so the continuous and quantized headings agree here.

   Also specify how the heading reaches the adapter. The unchanged dataset/loss interface supplies no room identifier or heading to the model. For this experiment, the least-change route is to validate that every participating room’s bound record supplies the same heading, install that shared cue, and reject mixed headings. Generic per-room conditioning would require additional routing beyond the stated scope.

   Distinguish two checkpoint-loading modes:

   - A/E initialization: validate every inherited key strictly and initialize only the new adapter weight/bias to zero.
   - Subsequent stage/evaluation checkpoints: require the complete adapter state strictly.

   “Pinned state dict loads unchanged” must not become unrestricted `strict=False`, or silently zero a missing trained adapter.

   With one fixed heading, the adapter cannot identify the benefit of heading information itself:
   \[
   W\,a_j(\phi)+b=W R(-\phi)a_j(0)+b.
   \]
   The heading is absorbable into the learned projection. J/K therefore test an added learnable circular positional representation during fine-tuning. They are fair and informative supplementary controls, but Q1/Q3 cannot establish that measured heading information caused a gain.

   Similarly, P4 = H − A bundles new five-channel pretraining with the room-to-heading frame change. P1 = H − D holds the HAA frame fixed and is the better matched channel-family comparison, although it still includes added parameters and a different pretrained realization. Keep P3 as the requested pipeline comparison. Describe simulated pretraining as using an absolute azimuth anchor; its loudspeaker-relative meaning comes from the HAA heading transform.

2. **Specify N1i’s common cohort and exp_11’s suppression behavior.**

   The frozen bootstrap can implement the interaction directly:
   ```text
   paired_intervals(g - e, d - a, query_clusters, finetuning_seeds)
   ```
   This jointly resamples the interaction and preserves covariance. Do not subtract independently bootstrapped intervals.

   The existing [cell assembler](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:776) handles only two arms. Register an exp_11 four-arm assembler that checks query indices, IR paths, protocol metadata, and seed alignment across A/D/E/G; uses one query mask finite across all four arms and all three seeds; and reports original per-arm exclusions. Require the joint cohort to cover at least 99% of the test queries, retaining the component invalidity checks for G − E and D − A.

   Interpret a positive interaction as a larger heading-frame penalty after yaw pretraining, rather than automatically calling it harm by one individual arm.

   Add a defined unavailable result for degenerate intervals. The frozen [convergence helper](/home/yixunhu/codespace/xRIR_code/tools/exp06_bootstrap.py:123) raises on zero width; cancellation in an interaction can produce this. Handle it in the new wrapper without aborting the entire summary or changing the historical helper.

   Finally, explicitly include or exclude descriptive screen labels from the universal suppression rule. I recommend including them for exp_11. Existing [screen logic](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:890) suppresses empty cohorts but does not apply the full void rule—consistent with exp_09’s registered behavior. Any stronger gate must be exp_11-specific. Test four-arm missingness, cancellation, boundaries, unconverged cells, and unchanged historical screen semantics.

3. **Replace R1’s generic field promise with a source-field map.**

   The actual canonical structures differ:

   | Historical contrast | Hallway C50 source | Fields available |
   |---|---|---|
   | C − D | exp_06 `D[6]` | `diff`, `nominal_two_way`; no convergence or label |
   | D − A | exp_06 `D[17]` | `diff`, `nominal_two_way`; no convergence or label |
   | C − A | exp_06 `H1` | `diff`, `two_way`, `convergence`, `verdict` |
   | C − F | exp_06 `H1b` | Same relevant structure as H1 |
   | E − A | exp_09 `E1` | Includes `category`, `verdict`, convergence, intervals, and non-inferiority |

   These are in the actual [exp_06 canonical JSON](/home/yixunhu/codespace/xRIR_code/ckpt/exp06/stats.json) and [exp_09 canonical JSON](/home/yixunhu/codespace/xRIR_code/ckpt/exp09/stats.json).

   Copy extant fields exactly, retain their original meaning and interval convention, and mark absent fields `null`/“not recorded.” Record source hashes and selectors, with contrast/room/metric assertions. Do not manufacture convergence or retrospectively classify descriptive rows as new decisions.

4. **Make the round-2 composition and summarizer dispatch executable.**

   The exp_11-owned family is the least disruptive design, but two concrete gaps remain.

   First, §4.9 cannot literally import the old fine-tuning loop: optimizer construction, epoch iteration, validation selection, and checkpoint writing are embedded inside [the old `main`](/home/yixunhu/codespace/xRIR_code/tools/exp06_haa_finetune.py:250). Specify an exp_11-owned faithful orchestration loop importing the unchanged numerical helpers. Verify optimizer-step order, validation cadence, epoch-zero selection, strict improvement selection, and final checkpoint behavior. The evaluation loop is importable, but exp_11 must own its provenance and metadata writer. Keep the frozen files unchanged.

   Second, per-arm code-key mapping is necessary but insufficient. The current summarizer requires `haa_job` and directly invokes exp_06’s specification loader, child verifier, role resolver, and lineage functions in [job admission](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:648).

   Register a per-arm admission adapter covering those calls. Historical arms must retain their old validators; H/I/J/K must use exp_11 validators. Distinguish serialized run types from semantic training/evaluation roles used by protocol checks.

   Adapter headings also need their own admission fields. Current [room-frame metadata validation](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:566) rejects heading records. Permit and bind adapter input headings specifically for J/K while preserving that rejection for historical room-frame arms.

   No old-finalizer registration appears necessary. Prefer leaving it untouched; an exception should require a demonstrated dependency, not speculative convenience. Also correct §2.2’s statement that every new arm uses the exp_11 pipeline: G intentionally uses the extended exp_06 route.

5. **Complete approval coverage and phase-specific lifecycle rules.**

   The §4.7 code-key list omits `exp11_launch.sh`. Shell launchers are outside Python import closures, as the existing [profiles implementation](/home/yixunhu/codespace/xRIR_code/tools/exp06_profiles.py:15) explicitly documents. Add `launch_sh` or an explicit closure extra, checked at startup and finalization.

   Register a producer/arm requirement matrix that consumes only available inputs:

   - Pretraining cannot require its own output checkpoint pin.
   - J/K and Phase 1b must work while both H/I checkpoint pins remain null.
   - H must work while I’s checkpoint remains unavailable.
   - Final publication requires all planned arms.

   The broad historical approval requirements cannot simply be copied.

   Finally, cover the launcher’s probe and the HAA training/evaluation smokes. Either enumerate distinct diagnostic run types or bind an explicit diagnostic kind and entry point within `exp11_smoke`. Their budgets, artifact contracts, and permanent non-admissibility must be explicit. These are acceptance details for the proposed family, not reasons to broaden it.

The admission and publication design otherwise needs no change. A/B remain covered by the reconstructed receipt and inherited protocol checks; C/D/E/F retain their original child/job and approval-blob evidence; A′ remains a hash-bound external summary row and never enters paired inference. Phase 1, Phase 1b, and final canonical outputs are appropriately separated.

H_RECIPE and I_RECIPE now contain the important distinctions: I’s saving/resume restrictions and augmentation counters; H’s declared retention change; the common numerical recipe and loader length; derived parameter counts; strict argument/history/checkpoint validation; and external completion before promotion. Preserve these exactly when implementing the new validators.

The approximately **75 GPU-hour** estimate is consistent with the component estimates, and **95 hours reserved** is a reasonable allowance. Two to three days for round-2 implementation/review is more credible than the earlier thin-wrapper estimate. October 4–5 remains conditional on actual exclusive handover. Running I sequentially on the first available card may finish earlier than waiting for the second. The exp_11 namespace move resolves the collision.

The final interpretation should remain conditional on these pretrained realizations, the previously examined rooms, and the primary hallway C50 cell. Three fine-tuning seeds do not estimate pretraining variability.

**Verification:** All 126 legacy-receipt artifact hashes matched. Recorded source hashes matched for 31 children each of C/D/F/E and all ten simulated-evaluation manifests. These checks support preservation, but are not complete finalizer replay. The closure-test suite failed during dependency imports because required cache/temp writes were unavailable in the read-only sandbox; I do not report it as passing. No GPU work, process signalling, finalizer execution, or file saving was performed.
tokens used
91,599
**Reviewer:** OpenAI Codex `gpt-6-astra`, codex-cli 0.154.0, `codex exec`  
**Reasoning effort:** `ultra`, confirmed by this invocation’s log  
**Sandbox/resources:** read-only; CUDA disabled; CPU computations limited to one thread each; no process signals  
**Tree:** `/home/yixunhu/codespace/xRIR_code`, branch `main`; existing worklog modifications and untracked records  
**HEAD:** `c6233e370c0d355ad00e438e2db91092e76a132d`, unchanged during review  
**Date/time:** September 26, 2026, approximately 17:44–17:48 EDT  

**Verdict: approve with changes.**

V2 resolves the substantive round-1 scientific objections and chooses the right boundary for the new implementation. The remaining changes below concern precise adapter contracts, statistical edge cases, and executable provenance routing. They do not require additional experimental arms or a redesign.

I read the SOP, v2, superseded v1, round-1 review, query and notebook, relevant predecessor records, and implementation sources. The announcements directory is empty.

| Round-1 item | Round-2 assessment |
|---|---|
| Positional-code redundancy claim; D’s label | **Resolved.** Information availability is distinguished from architectural equivalence; D is an implicit-conditioning control. |
| H/I as literal tests; primary P3 = C − H | **Resolved.** H consistently occupies the requested explicit-cue baseline row; I is secondary. |
| J/K adapter design | **Partially resolved.** Technically feasible and correctly registered as a different design; clarify loading, angle/routing, and interpretation as change 1. |
| Yaw invariance argument; G ≈ E; “refutation” | **Resolved in substance.** Fine-tuning can recover orientation dependence; equivalence is a hypothesis; D − A no longer refutes explicit conditioning. Remaining causal wording is covered by change 1. |
| Notation, N1, N2 direction, separate superiority | **Resolved.** The inequalities and separation of claims are correct. |
| N1i interaction | **Partially resolved.** Correct estimand and historical disclosure; specify four-arm cohort construction and unavailable-cell handling. |
| N3; P1/P2 equivalence | **Resolved.** Strict 95% containment is explicitly chosen and correctly distinguished from conventional TOST. |
| Three S1 families; Q1–Q4; P3/P3′; P4/P4′ | **Resolved statistically.** Only P3 establishes the headline; other comparisons need the stated conditional interpretations. |
| Void/unconverged suppression | **Resolved as policy; partially resolved in implementation specification.** Historical screen helpers do not implement the proposed universal suppression. |
| R1 historical copying | **Partially resolved.** Correct sources and prohibition on recomputation, but some promised source fields do not exist. |
| Single overall reading; fixed publication rows | **Resolved.** No result-dependent replacement of D/G or substitution of A′ for A. |
| Round-1 pipeline, resolver, `expected_inits`, schema regression, census | **Resolved in design.** The proposed changes are minimal and appropriately tested. |
| Exp_11-owned round-2 family | **Partially resolved.** Correct ownership boundary; remaining routing and helper-composition gaps are changes 4–5. |
| Historical completion preservation | **Resolved for preserving current HAA/sim admission.** Unqualified replay of *every* historical completion remains **open and explicitly deferred**, because of pre-existing C-pretraining drift. |
| A/B receipt; C/D/E/F completions; A′ external row | **Resolved.** These are the correct admission paths. |
| Separate phase outputs; historical-output semantics | **Resolved.** Distinct paths avoid overwrite; regression expectations correctly exclude changing producer metadata. |
| H_RECIPE/I_RECIPE and completion contract | **Resolved in specification.** Approval and diagnostic lifecycle details remain to be pinned. |
| Cost, schedule, risks, namespace, reviewer effort | **Resolved.** Estimates are credible allowances; exp_11 avoids the peer’s namespace; this review uses `ultra`. |

The required changes are:

1. **Finish the adapter contract and restrict its scientific interpretation.**

   The proposed insertion after the patch embedding’s final LayerNorm is sound. It avoids the joint geometry/azimuth normalization change identified in round 1. Zero **weight and bias** give 1,536 added parameters at dimension 512 and can preserve the initial function. A bounded CPU prototype using the actual SimpleViT components produced equal outputs; this establishes feasibility, not correctness of the future implementation. See the original [embedding and forward path](/home/yixunhu/codespace/xRIR_code/model/simple_vit.py:129).

   Pin the token-angle convention. A natural choice for the 16 token columns is
   \[
   \theta_j=2\pi(j+\tfrac12)/16-\pi.
   \]
   Convert the bound `phi_deg` to radians. All four current records have `phi_deg=-90`, `k=128`, and confirmatory admissibility, so the continuous and quantized headings agree here.

   Also specify how the heading reaches the adapter. The unchanged dataset/loss interface supplies no room identifier or heading to the model. For this experiment, the least-change route is to validate that every participating room’s bound record supplies the same heading, install that shared cue, and reject mixed headings. Generic per-room conditioning would require additional routing beyond the stated scope.

   Distinguish two checkpoint-loading modes:

   - A/E initialization: validate every inherited key strictly and initialize only the new adapter weight/bias to zero.
   - Subsequent stage/evaluation checkpoints: require the complete adapter state strictly.

   “Pinned state dict loads unchanged” must not become unrestricted `strict=False`, or silently zero a missing trained adapter.

   With one fixed heading, the adapter cannot identify the benefit of heading information itself:
   \[
   W\,a_j(\phi)+b=W R(-\phi)a_j(0)+b.
   \]
   The heading is absorbable into the learned projection. J/K therefore test an added learnable circular positional representation during fine-tuning. They are fair and informative supplementary controls, but Q1/Q3 cannot establish that measured heading information caused a gain.

   Similarly, P4 = H − A bundles new five-channel pretraining with the room-to-heading frame change. P1 = H − D holds the HAA frame fixed and is the better matched channel-family comparison, although it still includes added parameters and a different pretrained realization. Keep P3 as the requested pipeline comparison. Describe simulated pretraining as using an absolute azimuth anchor; its loudspeaker-relative meaning comes from the HAA heading transform.

2. **Specify N1i’s common cohort and exp_11’s suppression behavior.**

   The frozen bootstrap can implement the interaction directly:
   ```text
   paired_intervals(g - e, d - a, query_clusters, finetuning_seeds)
   ```
   This jointly resamples the interaction and preserves covariance. Do not subtract independently bootstrapped intervals.

   The existing [cell assembler](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:776) handles only two arms. Register an exp_11 four-arm assembler that checks query indices, IR paths, protocol metadata, and seed alignment across A/D/E/G; uses one query mask finite across all four arms and all three seeds; and reports original per-arm exclusions. Require the joint cohort to cover at least 99% of the test queries, retaining the component invalidity checks for G − E and D − A.

   Interpret a positive interaction as a larger heading-frame penalty after yaw pretraining, rather than automatically calling it harm by one individual arm.

   Add a defined unavailable result for degenerate intervals. The frozen [convergence helper](/home/yixunhu/codespace/xRIR_code/tools/exp06_bootstrap.py:123) raises on zero width; cancellation in an interaction can produce this. Handle it in the new wrapper without aborting the entire summary or changing the historical helper.

   Finally, explicitly include or exclude descriptive screen labels from the universal suppression rule. I recommend including them for exp_11. Existing [screen logic](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:890) suppresses empty cohorts but does not apply the full void rule—consistent with exp_09’s registered behavior. Any stronger gate must be exp_11-specific. Test four-arm missingness, cancellation, boundaries, unconverged cells, and unchanged historical screen semantics.

3. **Replace R1’s generic field promise with a source-field map.**

   The actual canonical structures differ:

   | Historical contrast | Hallway C50 source | Fields available |
   |---|---|---|
   | C − D | exp_06 `D[6]` | `diff`, `nominal_two_way`; no convergence or label |
   | D − A | exp_06 `D[17]` | `diff`, `nominal_two_way`; no convergence or label |
   | C − A | exp_06 `H1` | `diff`, `two_way`, `convergence`, `verdict` |
   | C − F | exp_06 `H1b` | Same relevant structure as H1 |
   | E − A | exp_09 `E1` | Includes `category`, `verdict`, convergence, intervals, and non-inferiority |

   These are in the actual [exp_06 canonical JSON](/home/yixunhu/codespace/xRIR_code/ckpt/exp06/stats.json) and [exp_09 canonical JSON](/home/yixunhu/codespace/xRIR_code/ckpt/exp09/stats.json).

   Copy extant fields exactly, retain their original meaning and interval convention, and mark absent fields `null`/“not recorded.” Record source hashes and selectors, with contrast/room/metric assertions. Do not manufacture convergence or retrospectively classify descriptive rows as new decisions.

4. **Make the round-2 composition and summarizer dispatch executable.**

   The exp_11-owned family is the least disruptive design, but two concrete gaps remain.

   First, §4.9 cannot literally import the old fine-tuning loop: optimizer construction, epoch iteration, validation selection, and checkpoint writing are embedded inside [the old `main`](/home/yixunhu/codespace/xRIR_code/tools/exp06_haa_finetune.py:250). Specify an exp_11-owned faithful orchestration loop importing the unchanged numerical helpers. Verify optimizer-step order, validation cadence, epoch-zero selection, strict improvement selection, and final checkpoint behavior. The evaluation loop is importable, but exp_11 must own its provenance and metadata writer. Keep the frozen files unchanged.

   Second, per-arm code-key mapping is necessary but insufficient. The current summarizer requires `haa_job` and directly invokes exp_06’s specification loader, child verifier, role resolver, and lineage functions in [job admission](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:648).

   Register a per-arm admission adapter covering those calls. Historical arms must retain their old validators; H/I/J/K must use exp_11 validators. Distinguish serialized run types from semantic training/evaluation roles used by protocol checks.

   Adapter headings also need their own admission fields. Current [room-frame metadata validation](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:566) rejects heading records. Permit and bind adapter input headings specifically for J/K while preserving that rejection for historical room-frame arms.

   No old-finalizer registration appears necessary. Prefer leaving it untouched; an exception should require a demonstrated dependency, not speculative convenience. Also correct §2.2’s statement that every new arm uses the exp_11 pipeline: G intentionally uses the extended exp_06 route.

5. **Complete approval coverage and phase-specific lifecycle rules.**

   The §4.7 code-key list omits `exp11_launch.sh`. Shell launchers are outside Python import closures, as the existing [profiles implementation](/home/yixunhu/codespace/xRIR_code/tools/exp06_profiles.py:15) explicitly documents. Add `launch_sh` or an explicit closure extra, checked at startup and finalization.

   Register a producer/arm requirement matrix that consumes only available inputs:

   - Pretraining cannot require its own output checkpoint pin.
   - J/K and Phase 1b must work while both H/I checkpoint pins remain null.
   - H must work while I’s checkpoint remains unavailable.
   - Final publication requires all planned arms.

   The broad historical approval requirements cannot simply be copied.

   Finally, cover the launcher’s probe and the HAA training/evaluation smokes. Either enumerate distinct diagnostic run types or bind an explicit diagnostic kind and entry point within `exp11_smoke`. Their budgets, artifact contracts, and permanent non-admissibility must be explicit. These are acceptance details for the proposed family, not reasons to broaden it.

The admission and publication design otherwise needs no change. A/B remain covered by the reconstructed receipt and inherited protocol checks; C/D/E/F retain their original child/job and approval-blob evidence; A′ remains a hash-bound external summary row and never enters paired inference. Phase 1, Phase 1b, and final canonical outputs are appropriately separated.

H_RECIPE and I_RECIPE now contain the important distinctions: I’s saving/resume restrictions and augmentation counters; H’s declared retention change; the common numerical recipe and loader length; derived parameter counts; strict argument/history/checkpoint validation; and external completion before promotion. Preserve these exactly when implementing the new validators.

The approximately **75 GPU-hour** estimate is consistent with the component estimates, and **95 hours reserved** is a reasonable allowance. Two to three days for round-2 implementation/review is more credible than the earlier thin-wrapper estimate. October 4–5 remains conditional on actual exclusive handover. Running I sequentially on the first available card may finish earlier than waiting for the second. The exp_11 namespace move resolves the collision.

The final interpretation should remain conditional on these pretrained realizations, the previously examined rooms, and the primary hallway C50 cell. Three fine-tuning seeds do not estimate pretraining variability.

**Verification:** All 126 legacy-receipt artifact hashes matched. Recorded source hashes matched for 31 children each of C/D/F/E and all ten simulated-evaluation manifests. These checks support preservation, but are not complete finalizer replay. The closure-test suite failed during dependency imports because required cache/temp writes were unavailable in the read-only sandbox; I do not report it as passing. No GPU work, process signalling, finalizer execution, or file saving was performed.

