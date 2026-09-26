**Reviewer:** OpenAI Codex `gpt-6-astra`, codex-cli 0.154.0  
**Reasoning effort:** `xhigh` — confirmed by this invocation; the SOP specifies `ultra`.  
**Sandbox/resources:** read-only; CUDA disabled; CPU computation capped at four threads; no process signals.  
**Tree:** `/home/yixunhu/codespace/xRIR_code`, branch `main`, with existing worklog modifications and untracked files.  
**HEAD:** started at `885240c52df2c4931c311582068b4733d6d8be4b`; last checked at `c52ee94ee10b38dcf9d600575e959e674b4cc5b4`. Intervening changes were peer worklog files; reviewed implementation files were unchanged.  
**Date/time:** September 26, 2026, 17:31–17:36 EDT.

**Verdict: request changes.**

Phase 1 is a useful, inexpensive extension of exp_09. It does not establish the scientific conclusion currently claimed for it. The principal blockers are the equivalence argument, the causal interpretation, the decision rules, and incomplete Phase-2 provenance routing.

1. **The positional-code argument establishes information availability, not architectural or experimental equivalence.**

   [Plan §2.1](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_orientation_cue_fairness_claude/plan_orientation_cue_fairness.md:17) is correct in a limited sense: on a fixed grid, the azimuth planes contain no additional sample-dependent heading measurement. Token column plus known patch offsets determines their values. SimpleViT already receives absolute token position.

   The next step—therefore D faithfully substitutes for a literal channel-equipped SimpleViT—is unsupported by the actual [patch embedding](/home/yixunhu/codespace/xRIR_code/model/simple_vit.py:129).

   Its computation is:

   \[
   z_j=\operatorname{LN}_{512}
       \left(W\,\operatorname{LN}_{1536}(x_j)+b\right)+p_j.
   \]

   The proposed five-channel version instead computes:

   \[
   z'_j=\operatorname{LN}_{512}
       \left(W'\,\operatorname{LN}_{2560}([x_j,a_j])+b'\right)+p_j.
   \]

   Here \(a_j\) is fixed for patch column \(j\), but the normalization of **both geometry and azimuth values depends jointly on the geometry and those fixed values**. The planes also vary across the 32 pixel columns within a patch; they are not spatially constant planes.

   Consequently, the added channels change normalization, the learned projection, gradients, parameter count, and the accessibility of orientation information. They can supply a fixed reference against which geometry statistics are represented. The final LayerNorm does not generally undo this change. The physical circular azimuth code also differs from SimpleViT’s token-index sinusoidal frequencies and seam behavior.

   I checked the normalization issue with a small CPU example using the actual patch-embedding modules. After copying the original projection weights and setting every added-channel projection weight to zero, the embeddings still differed by a maximum absolute value of approximately **0.140**. The parameter increase was **526,336**, as the plan states. This is an implementation counterexample to a function-preserving expansion, not a performance result.

   **Required change:** call D “SimpleViT in the heading frame” or “implicit orientation conditioning.” A paper reviewer could accept D as an informative control, but should not accept the current argument as sufficient to omit an explicit-cue baseline.

   H and I are sensible literal controls. They should be the planned direct test of the reviewer’s request, with **C − H the primary literal comparison**, rather than optional confirmation of an asserted equivalence. Their comparisons with D/G test practical performance equivalence for these trained models; they cannot prove architectural redundancy.

   A cheaper alternative exists: preserve the pretrained three-channel embedding and add a small, zero-initialized projection of heading-relative azimuth **after** its normalization. Fine-tune this explicit adapter from A and E under the same HAA schedule. This preserves the initial checkpoint function and costs roughly two HAA queues instead of two pretraining runs. It is a useful additional-conditioning control, but must be registered as a different design: it does not replicate CERPA’s five-channel pretraining. Naively expanding the existing projection with zero weights does **not** preserve the checkpoint function because of LayerNorm.

2. **The yaw argument and the proposed “refutation” overstate what the experiment can establish.**

   The exact statement “an invariant predictor cannot change its output under a heading-frame rotation” is correct. The implemented experiment does not guarantee such a predictor.

   `train_xRIR_backbone.compute_loss` applies yaw augmentation during simulated pretraining. In contrast, [HAA fine-tuning](/home/yixunhu/codespace/xRIR_code/tools/exp06_haa_finetune.py:259) optimizes the model’s parameters and calls `compute_loss(model, batch)` without augmentation. Orientation dependence can therefore be learned or recovered during fine-tuning. Output invariance also does not imply that every intermediate feature discards azimuth.

   Thus **G ≈ E is a reasonable prospective hypothesis, not a consequence of the code**. Equal aggregate errors would not establish prediction-level invariance or absence of cue use.

   G is fair when described precisely as “yaw-pretrained SimpleViT, fine-tuned and evaluated in the heading frame.” It is not established as the “strongest” or “best cue-equipped SimpleViT.” Exp_09 already found that E was worse than A on hallway C50, and the explicit-cue alternatives remain untested.

   There is a further inconsistency: I’s proposed augmentation rotates geometry while its azimuth planes remain fixed. On omnidirectional simulated data, its training objective likewise encourages insensitivity to geometry rotation relative to that fixed anchor. Adding channels does not automatically resolve the alleged conflict. The oriented cylindrical result also does not prove that its architecture uniquely resolves it.

   **Required change:** remove the claim that D − A showing no improvement, together with C outperforming G, refutes the additional-conditioning explanation. D − A mixes orientation canonicalization with a coordinate-frame change relative to pretraining. A harmful implementation of conditioning does not show that conditioning cannot help another implementation.

   This qualification matters numerically: exp_06’s hallway C50 errors were **A 1.135, C 1.137, D 1.766 dB**. C’s advantage over D is real in those results, but C approximately matches A. It is not evidence of an overall advantage over the same-budget room-frame baseline.

   All four rooms also use the same inferred axis and the same `k=128`. These experiments compare pipelines on fixed rooms; they do not independently establish generalization across varying loudspeaker headings.

   Cheap supporting analyses include fine-tuned side-split summaries from retained per-query outputs and explicitly labelled within-checkpoint orientation-sensitivity diagnostics. The frozen mirror-probe CLI cannot simply supply the latter for G/D: its SimpleViT `control` slot is hard-coded to the room frame. Use a reviewed exp_10 driver around its helpers if those diagnostics remain planned.

3. **The contrasts are mostly well defined, but the decision rules need correction and fixed publication choices.**

   The paired estimand, historical bootstrap conventions, convergence check, and disclosed single pretraining realization are appropriate for conditional comparisons. Three fine-tuning seeds do not estimate pretraining variability.

   The following corrections are necessary:

   | Item | Assessment and required correction |
   |---|---|
   | **N1: G − D** | Valid comparison of these pretrained initializations under heading-frame fine-tuning. It does not alone establish better cue use. If the mechanism is central, preregister the interaction \((G-E)-(D-A)\), while disclosing that D − A was already observed. |
   | **N2: C − G** | Specify direction explicitly in the implementation. For interval \([L,U]\) on C − G, G’s non-inferiority is **\(-L<0.23\)**. Applying existing `classify_cell` directly would instead assess C’s non-inferiority against G. |
   | **N2 conclusion** | Failure to establish G’s non-inferiority is not evidence that G is worse by at least 0.23 dB. Report superiority and non-inferiority separately. A margin-sized advantage for C requires **\(U<-0.23\)** on C − G. |
   | **N3: G − E** | “No detected difference” does not support the prediction G ≈ E. Add a preregistered equivalence statement if approximate equality is the prediction; retain the ordinary category separately. |
   | **S1** | Bonferroni-11 gives protection within each contrast’s 11-cell family. It does not provide experiment-wide protection across three contrasts. Explicitly declare three descriptive families, or use a 33-cell family for an overall screen claim. |
   | **R1** | Appropriate as historical re-reporting. C − D was already observed: −0.629 dB, nominal interval [−0.764, −0.500]. Preserve its descriptive status and original interval convention. |
   | **P1/P2** | Define `equivalent_at_margin` as strict containment: \(L>-0.23\) and \(U<0.23\). A 95% interval-containment rule is conservative relative to conventional TOST at α=.05, which corresponds to a 90% interval. Choose and name the rule explicitly. |
   | **P3** | Register C − H and C − I separately, including which is primary and how multiplicity is handled if either can establish the headline claim. Use the corrected reverse-direction non-inferiority calculation. |

   Every decision-bearing output—category, non-inferiority, equivalence, and any combined conclusion—must be unavailable when the relevant cell is void or unconverged. Equality at a decision boundary should establish neither claim. The 0.23 dB margin applies to hallway C50; it cannot be carried into EDT or T60 screens.

   **Required change:** remove “if either prediction fails, replace D/G.” Failure to establish equivalence can mean insufficient precision. It does not establish a difference. More importantly, the four-row presentation should not depend on which results materialize. If H/I run, report them consistently as the literal-cue arms, with D/G retained as frame-only controls.

   Also reconcile §3’s two overall rules: N2 includes a failed non-inferiority condition, whereas the “Overall reading” drops it. Replace both with one prespecified, appropriately scoped interpretation.

   R1 should copy named fields from hash-bound exp_06/exp_09 canonical JSON, recording source hashes and field paths. §4 currently describes reconstructing those rows from completions instead. Revalidation is welcome, but recomputation must not silently turn a historical descriptive interval into a new confirmatory decision. New G/H/I contrasts remain prospective on **previously examined test rooms**, not independent replication on untouched data.

4. **Round 1 is feasible with minimal changes; Phase 2 needs explicit versioned routing rather than unspecified thin wrappers.**

   The frozen-file policy and prohibition on mutating `BACKBONES_EXP06` are correct. Clarify that “every `model/*.py` file” means every **existing** file; adding new modules is allowed. Freeze the complete relevant dependency closures, not only the hand-written list.

   **Round 1:** the proposed pipeline and summarizer extensions are sufficient in principle. No new finalizer run type is needed for G.

   The pipeline must apply the `exp04_aug_checkpoint` identity resolver to both `yawaug` and `yawaug_hf`, reset the frame on every init selection, retain bare `zeroshot`’s historical expansion, and pass the approved heading records for G. In the summarizer, extend `expected_inits` explicitly: adding an `ARMS` entry with `init_sha256=None` alone would leave G without the intended checkpoint identity check.

   Keep `exp06_profiles.py`, `exp06_approvals_api.py`, both numerical HAA entry points, and the sim-eval closures unchanged. Add tests for mixed queues, wrong checkpoint, wrong frame, missing heading, and correct exp_04 approval binding.

   **Phase 2:** the proposed new model modules and detached `BACKBONES_EXP10` are sound. However, the old HAA entry points hard-code their parsers, factories, registry digest, provenance identity, and run types. Calling their `main` functions cannot implement the proposed new routes. Use exp_10-owned parsers, preparation/provenance code, and orchestration that compose the unchanged numerical helpers. Do not monkeypatch old module globals.

   The least disruptive boundary is an **exp_10-owned finalizer/adapter**, with exp_10 training, smoke/probe, HAA-child, and job routes. It can reuse suitable validation helpers while leaving historical dispatch unchanged. If the implementation instead extends `exp06_finalize.py`, the plan must enumerate every affected dispatcher and prove that the old paths reconstruct identical evidence.

   In either design, explicitly cover:

   - Registry selection, factory/state validation, entry-module identity, and recipe selection by run type.
   - Exp_10 approvals schema, committed-blob binding, and distinct H/I checkpoint artifact identities.
   - HAA job-spec validation, child roles, lineage checks, and pipeline entry-point selection.
   - Summarizer admission for old exp_06-family children alongside new exp_10 children.
   - An exp_10 smoke route: the frozen exp_06 smoke runner does not automatically recognize new entries.

   [The current `check_arm_identities`](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:468) compares all children against one approval mapping’s `haa_finetune`/`haa_eval` keys. Those keys cannot simultaneously identify different old and new entry-point closures. Add an explicit per-arm profile/code-key mapping; do not replace historical keys with exp_10 digests or disable the checks.

   There is also an existing limitation to the requested “every historical completion” guarantee. The full-pretraining validator checks the current launcher/finalizer bytes and current approvals file. I found **pre-existing drift** for C’s pretraining record in `exp06_finalize.py`, `exp06_smoke.py`, `exp06_launch.sh`, and the approvals file. The HAA blob-at-commit fix does not cover that full-pretraining path. This is not an exp_10 regression, but the plan must distinguish preservation of current HAA/sim admission from historical pretraining replay. A broader guarantee needs an explicit verification route using the recorded code and approvals snapshots, without rewriting original records or weakening launch-time checks.

   The optional future simulated evaluation of H/I also cannot use the frozen exp_06 route unchanged: its factory recognizes only `BACKBONES_EXP06`. Deferring that evaluation is reasonable; defer its new routing explicitly too.

5. **Historical reuse is admissible for A/B/C/D/E/F, but A′ requires a different presentation path. Publication needs separate phase outputs.**

   The current admission situation is:

   | Arms | Admissible reuse |
   |---|---|
   | **A, B** | Existing reconstructed legacy receipt plus inherited exp_02 completeness/protocol checks. They do not have exp_06-style child completions. |
   | **C, D, F, E** | Existing HAA child/job records, unchanged execution dependencies, original approval blobs, checkpoint identities, headings, and protocol checks. |
   | **A′ released** | **Not covered by the existing per-query legacy receipt.** Available as a clearly labelled external row copied from exp_02’s approved canonical statistics. |

   I inspected the actual receipt: it declares exactly `['control', 'cyl']`, contains **126 files**, and contains no released-arm artifact entries. All 126 hashes matched. Its inclusion of exp_02’s canonical `stats.json` permits a bound historical summary row for A′; it does not certify A′’s individual artifacts for new paired inference.

   Do not add `released` to global `LEGACY_ARMS`: the existing receipt validator requires exact arm and file membership. If new per-query inference involving A′ is needed, create a separate, explicitly approved supplemental receipt and admission path.

   For C/D/E/F, static checks found matching recorded source-file hashes and approval-blob hashes for **31 children per arm**. The shared registry digest remains `f9852f85…`. This supports the proposed reuse, provided those executable dependencies stay unchanged. Blob-at-commit approval reads permit approval refills; they do not permit arbitrary source changes.

   For the scientific table, A is the matched baseline for D/H. If A′ remains the prominently named “xRIR” row, distinguish its released initialization and include A visibly enough that readers cannot interpret A′ → D/H as a cue-only intervention. A′ may remain an external reference row, but must not silently substitute for A in the ablation.

   Two publication fixes are also required:

   - Phase 1 and Phase 2 cannot both publish to the same already-existing `stats.json`/`summary.txt`; `write_outputs` deliberately refuses overwrites. Register separate phase outputs, or reserve the final canonical paths until all planned arms finish.
   - “Byte-identical exp06/exp09 outputs” should mean unchanged statistical payloads and rendered summaries under controlled fixtures, plus untouched historical canonical files. A newly produced JSON necessarily has different producer/provenance fields after a code change.

   The new summarizer configuration also needs multiple screens and distinct P3 decisions; the existing schema supports only one screen pair per experiment. Extend that structure without changing old experiment semantics.

6. **Phase-2 recipe parity must include exp_04’s saving constraints and a fully specified completion contract.**

   I compared the requested [exp_01 args](/home/yixunhu/codespace/xRIR_code/ckpt/xRIR_simple_8_shot/args.json) and [exp_04 args](/home/yixunhu/codespace/xRIR_code/ckpt/xRIR_simple_yawaug_8_shot/final/args.json). Their common numerical training recipe agrees. The relevant differences are:

   | Field | exp_01 | exp_04 |
   |---|---|---|
   | `save_every` | **500** | **0** |
   | `epoch_ckpt_every` | **5** | **1** |
   | `yaw_aug` | Absent; historical unaugmented run | **1** |
   | `yaw_aug_seed`, `yaw_aug_width` | Absent | **0, 512** |
   | `vit_*` | Absent; historical M defaults | **512 / 12 / 8 / 512** |
   | `no_save` | Absent | **false** |
   | `train_batches_per_epoch`, `tier`, `param_counts`, `env` | Absent | Recorded |
   | `save_dir` | Original run directory | Exclusive attempt directory |

   **Required change:** define two named recipe profiles, rather than merely allowing `yaw_aug ∈ {0,1}`. In particular, I must require `save_every=0` and `resume=None`. The pinned trainer explicitly rejects yaw augmentation otherwise; copying exp_06’s `save_every=500` would violate this contract.

   H may use `epoch_ckpt_every=1` instead of historical 5 as a declared operational difference to retain the native epoch-12 checkpoint. Missing historical fields must remain documented normalizations or unavailable metadata, not be presented as originally recorded facts.

   Both profiles should fix the common 12-epoch recipe, batch 32 × accumulation 2, seed 0, M dimensions, optimizer/scheduler settings, TF32, full splits, and loader length **9,261**. I additionally fixes the augmentation seed/width and counter constraints. Derived parameter counts must come from the new factory; the encoder increment is exactly **526,336**.

   Carry over the full completion contract explicitly: agreeing startup/retained/checkpoint arguments; strict field types; no truncation or forbidden resume; epochs 1–12 exactly once with finite history; final epoch/batch metadata; tensor keys, shapes, dtypes, and values matching between `epoch_012.pth` and `last.pth["model"]`; input and closure revalidation; committed approvals; zero child exit; closed log and receipt; and external completion before checkpoint promotion.

   HAA completion must likewise preserve validation-based selection, initialization → stage 1 → stage 2 → evaluation lineage, heading binding, full test indices, and job completeness. The old contract is suitable; the new registry and approval routing must actually enforce it.

**Cost and schedule assessment.** The GPU estimates are credible planning allowances. Retained histories give approximately **28.2 h** for exp_01 SimpleViT, **27.4 h** for exp_04 yaw augmentation, and **31.0 h** for exp_06 oriented cylindrical pretraining. Thus 31/28 h plus two HAA queues is a plausible ≈65 GPU-hour Phase-2 estimate, pending the new-model timing probes. Including Phase 1 gives approximately **68 GPU-hours expected**, rather than 65 for the whole experiment.

Given the supplied owner ETAs, Phase 1 can plausibly finish September 29 after an actual exclusive handover. Sequential H/I work on the first card can finish around October 2 under expected timings; waiting for the second card pushes I into October 4. October 3–5 is a reasonable buffered target, not a reservation.

The Phase-2 implementation/review estimate is optimistic because it includes a new approval and completion family, not just model wrappers. The HAA pipeline also does not invoke the standalone launcher’s GPU preflight merely by sourcing its shell helpers; the runbook must explicitly enforce the promised exclusive-card condition.

Resolve the concurrent experiment namespace before implementation: the peer’s `exp_10_yaw_pilot_claude` plan also uses `ckpt/exp10/` and `tools/exp10_*`. Distinct subdirectories may avoid immediate collisions, but the shared experiment identifier and ownership need an explicit agreement.

Finally, use the SOP’s **ultra** setting for subsequent review closure, or record an authorized exception. This review ran at **xhigh** and should not be labelled otherwise.

**Verification limits:** I performed read-only source/record inspection, the bounded CPU embedding check, legacy-artifact hashing, historical HAA source/approval checks, and source-file hash checks across all ten completed simulated-evaluation manifests. Their recorded sim source files matched. I did not invoke finalizers, run the full suite, launch GPU work, signal processes, or save files.
tokens used
209,233
**Reviewer:** OpenAI Codex `gpt-6-astra`, codex-cli 0.154.0  
**Reasoning effort:** `xhigh` — confirmed by this invocation; the SOP specifies `ultra`.  
**Sandbox/resources:** read-only; CUDA disabled; CPU computation capped at four threads; no process signals.  
**Tree:** `/home/yixunhu/codespace/xRIR_code`, branch `main`, with existing worklog modifications and untracked files.  
**HEAD:** started at `885240c52df2c4931c311582068b4733d6d8be4b`; last checked at `c52ee94ee10b38dcf9d600575e959e674b4cc5b4`. Intervening changes were peer worklog files; reviewed implementation files were unchanged.  
**Date/time:** September 26, 2026, 17:31–17:36 EDT.

**Verdict: request changes.**

Phase 1 is a useful, inexpensive extension of exp_09. It does not establish the scientific conclusion currently claimed for it. The principal blockers are the equivalence argument, the causal interpretation, the decision rules, and incomplete Phase-2 provenance routing.

1. **The positional-code argument establishes information availability, not architectural or experimental equivalence.**

   [Plan §2.1](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_10_orientation_cue_fairness_claude/plan_orientation_cue_fairness.md:17) is correct in a limited sense: on a fixed grid, the azimuth planes contain no additional sample-dependent heading measurement. Token column plus known patch offsets determines their values. SimpleViT already receives absolute token position.

   The next step—therefore D faithfully substitutes for a literal channel-equipped SimpleViT—is unsupported by the actual [patch embedding](/home/yixunhu/codespace/xRIR_code/model/simple_vit.py:129).

   Its computation is:

   \[
   z_j=\operatorname{LN}_{512}
       \left(W\,\operatorname{LN}_{1536}(x_j)+b\right)+p_j.
   \]

   The proposed five-channel version instead computes:

   \[
   z'_j=\operatorname{LN}_{512}
       \left(W'\,\operatorname{LN}_{2560}([x_j,a_j])+b'\right)+p_j.
   \]

   Here \(a_j\) is fixed for patch column \(j\), but the normalization of **both geometry and azimuth values depends jointly on the geometry and those fixed values**. The planes also vary across the 32 pixel columns within a patch; they are not spatially constant planes.

   Consequently, the added channels change normalization, the learned projection, gradients, parameter count, and the accessibility of orientation information. They can supply a fixed reference against which geometry statistics are represented. The final LayerNorm does not generally undo this change. The physical circular azimuth code also differs from SimpleViT’s token-index sinusoidal frequencies and seam behavior.

   I checked the normalization issue with a small CPU example using the actual patch-embedding modules. After copying the original projection weights and setting every added-channel projection weight to zero, the embeddings still differed by a maximum absolute value of approximately **0.140**. The parameter increase was **526,336**, as the plan states. This is an implementation counterexample to a function-preserving expansion, not a performance result.

   **Required change:** call D “SimpleViT in the heading frame” or “implicit orientation conditioning.” A paper reviewer could accept D as an informative control, but should not accept the current argument as sufficient to omit an explicit-cue baseline.

   H and I are sensible literal controls. They should be the planned direct test of the reviewer’s request, with **C − H the primary literal comparison**, rather than optional confirmation of an asserted equivalence. Their comparisons with D/G test practical performance equivalence for these trained models; they cannot prove architectural redundancy.

   A cheaper alternative exists: preserve the pretrained three-channel embedding and add a small, zero-initialized projection of heading-relative azimuth **after** its normalization. Fine-tune this explicit adapter from A and E under the same HAA schedule. This preserves the initial checkpoint function and costs roughly two HAA queues instead of two pretraining runs. It is a useful additional-conditioning control, but must be registered as a different design: it does not replicate CERPA’s five-channel pretraining. Naively expanding the existing projection with zero weights does **not** preserve the checkpoint function because of LayerNorm.

2. **The yaw argument and the proposed “refutation” overstate what the experiment can establish.**

   The exact statement “an invariant predictor cannot change its output under a heading-frame rotation” is correct. The implemented experiment does not guarantee such a predictor.

   `train_xRIR_backbone.compute_loss` applies yaw augmentation during simulated pretraining. In contrast, [HAA fine-tuning](/home/yixunhu/codespace/xRIR_code/tools/exp06_haa_finetune.py:259) optimizes the model’s parameters and calls `compute_loss(model, batch)` without augmentation. Orientation dependence can therefore be learned or recovered during fine-tuning. Output invariance also does not imply that every intermediate feature discards azimuth.

   Thus **G ≈ E is a reasonable prospective hypothesis, not a consequence of the code**. Equal aggregate errors would not establish prediction-level invariance or absence of cue use.

   G is fair when described precisely as “yaw-pretrained SimpleViT, fine-tuned and evaluated in the heading frame.” It is not established as the “strongest” or “best cue-equipped SimpleViT.” Exp_09 already found that E was worse than A on hallway C50, and the explicit-cue alternatives remain untested.

   There is a further inconsistency: I’s proposed augmentation rotates geometry while its azimuth planes remain fixed. On omnidirectional simulated data, its training objective likewise encourages insensitivity to geometry rotation relative to that fixed anchor. Adding channels does not automatically resolve the alleged conflict. The oriented cylindrical result also does not prove that its architecture uniquely resolves it.

   **Required change:** remove the claim that D − A showing no improvement, together with C outperforming G, refutes the additional-conditioning explanation. D − A mixes orientation canonicalization with a coordinate-frame change relative to pretraining. A harmful implementation of conditioning does not show that conditioning cannot help another implementation.

   This qualification matters numerically: exp_06’s hallway C50 errors were **A 1.135, C 1.137, D 1.766 dB**. C’s advantage over D is real in those results, but C approximately matches A. It is not evidence of an overall advantage over the same-budget room-frame baseline.

   All four rooms also use the same inferred axis and the same `k=128`. These experiments compare pipelines on fixed rooms; they do not independently establish generalization across varying loudspeaker headings.

   Cheap supporting analyses include fine-tuned side-split summaries from retained per-query outputs and explicitly labelled within-checkpoint orientation-sensitivity diagnostics. The frozen mirror-probe CLI cannot simply supply the latter for G/D: its SimpleViT `control` slot is hard-coded to the room frame. Use a reviewed exp_10 driver around its helpers if those diagnostics remain planned.

3. **The contrasts are mostly well defined, but the decision rules need correction and fixed publication choices.**

   The paired estimand, historical bootstrap conventions, convergence check, and disclosed single pretraining realization are appropriate for conditional comparisons. Three fine-tuning seeds do not estimate pretraining variability.

   The following corrections are necessary:

   | Item | Assessment and required correction |
   |---|---|
   | **N1: G − D** | Valid comparison of these pretrained initializations under heading-frame fine-tuning. It does not alone establish better cue use. If the mechanism is central, preregister the interaction \((G-E)-(D-A)\), while disclosing that D − A was already observed. |
   | **N2: C − G** | Specify direction explicitly in the implementation. For interval \([L,U]\) on C − G, G’s non-inferiority is **\(-L<0.23\)**. Applying existing `classify_cell` directly would instead assess C’s non-inferiority against G. |
   | **N2 conclusion** | Failure to establish G’s non-inferiority is not evidence that G is worse by at least 0.23 dB. Report superiority and non-inferiority separately. A margin-sized advantage for C requires **\(U<-0.23\)** on C − G. |
   | **N3: G − E** | “No detected difference” does not support the prediction G ≈ E. Add a preregistered equivalence statement if approximate equality is the prediction; retain the ordinary category separately. |
   | **S1** | Bonferroni-11 gives protection within each contrast’s 11-cell family. It does not provide experiment-wide protection across three contrasts. Explicitly declare three descriptive families, or use a 33-cell family for an overall screen claim. |
   | **R1** | Appropriate as historical re-reporting. C − D was already observed: −0.629 dB, nominal interval [−0.764, −0.500]. Preserve its descriptive status and original interval convention. |
   | **P1/P2** | Define `equivalent_at_margin` as strict containment: \(L>-0.23\) and \(U<0.23\). A 95% interval-containment rule is conservative relative to conventional TOST at α=.05, which corresponds to a 90% interval. Choose and name the rule explicitly. |
   | **P3** | Register C − H and C − I separately, including which is primary and how multiplicity is handled if either can establish the headline claim. Use the corrected reverse-direction non-inferiority calculation. |

   Every decision-bearing output—category, non-inferiority, equivalence, and any combined conclusion—must be unavailable when the relevant cell is void or unconverged. Equality at a decision boundary should establish neither claim. The 0.23 dB margin applies to hallway C50; it cannot be carried into EDT or T60 screens.

   **Required change:** remove “if either prediction fails, replace D/G.” Failure to establish equivalence can mean insufficient precision. It does not establish a difference. More importantly, the four-row presentation should not depend on which results materialize. If H/I run, report them consistently as the literal-cue arms, with D/G retained as frame-only controls.

   Also reconcile §3’s two overall rules: N2 includes a failed non-inferiority condition, whereas the “Overall reading” drops it. Replace both with one prespecified, appropriately scoped interpretation.

   R1 should copy named fields from hash-bound exp_06/exp_09 canonical JSON, recording source hashes and field paths. §4 currently describes reconstructing those rows from completions instead. Revalidation is welcome, but recomputation must not silently turn a historical descriptive interval into a new confirmatory decision. New G/H/I contrasts remain prospective on **previously examined test rooms**, not independent replication on untouched data.

4. **Round 1 is feasible with minimal changes; Phase 2 needs explicit versioned routing rather than unspecified thin wrappers.**

   The frozen-file policy and prohibition on mutating `BACKBONES_EXP06` are correct. Clarify that “every `model/*.py` file” means every **existing** file; adding new modules is allowed. Freeze the complete relevant dependency closures, not only the hand-written list.

   **Round 1:** the proposed pipeline and summarizer extensions are sufficient in principle. No new finalizer run type is needed for G.

   The pipeline must apply the `exp04_aug_checkpoint` identity resolver to both `yawaug` and `yawaug_hf`, reset the frame on every init selection, retain bare `zeroshot`’s historical expansion, and pass the approved heading records for G. In the summarizer, extend `expected_inits` explicitly: adding an `ARMS` entry with `init_sha256=None` alone would leave G without the intended checkpoint identity check.

   Keep `exp06_profiles.py`, `exp06_approvals_api.py`, both numerical HAA entry points, and the sim-eval closures unchanged. Add tests for mixed queues, wrong checkpoint, wrong frame, missing heading, and correct exp_04 approval binding.

   **Phase 2:** the proposed new model modules and detached `BACKBONES_EXP10` are sound. However, the old HAA entry points hard-code their parsers, factories, registry digest, provenance identity, and run types. Calling their `main` functions cannot implement the proposed new routes. Use exp_10-owned parsers, preparation/provenance code, and orchestration that compose the unchanged numerical helpers. Do not monkeypatch old module globals.

   The least disruptive boundary is an **exp_10-owned finalizer/adapter**, with exp_10 training, smoke/probe, HAA-child, and job routes. It can reuse suitable validation helpers while leaving historical dispatch unchanged. If the implementation instead extends `exp06_finalize.py`, the plan must enumerate every affected dispatcher and prove that the old paths reconstruct identical evidence.

   In either design, explicitly cover:

   - Registry selection, factory/state validation, entry-module identity, and recipe selection by run type.
   - Exp_10 approvals schema, committed-blob binding, and distinct H/I checkpoint artifact identities.
   - HAA job-spec validation, child roles, lineage checks, and pipeline entry-point selection.
   - Summarizer admission for old exp_06-family children alongside new exp_10 children.
   - An exp_10 smoke route: the frozen exp_06 smoke runner does not automatically recognize new entries.

   [The current `check_arm_identities`](/home/yixunhu/codespace/xRIR_code/tools/exp06_summarize_haa.py:468) compares all children against one approval mapping’s `haa_finetune`/`haa_eval` keys. Those keys cannot simultaneously identify different old and new entry-point closures. Add an explicit per-arm profile/code-key mapping; do not replace historical keys with exp_10 digests or disable the checks.

   There is also an existing limitation to the requested “every historical completion” guarantee. The full-pretraining validator checks the current launcher/finalizer bytes and current approvals file. I found **pre-existing drift** for C’s pretraining record in `exp06_finalize.py`, `exp06_smoke.py`, `exp06_launch.sh`, and the approvals file. The HAA blob-at-commit fix does not cover that full-pretraining path. This is not an exp_10 regression, but the plan must distinguish preservation of current HAA/sim admission from historical pretraining replay. A broader guarantee needs an explicit verification route using the recorded code and approvals snapshots, without rewriting original records or weakening launch-time checks.

   The optional future simulated evaluation of H/I also cannot use the frozen exp_06 route unchanged: its factory recognizes only `BACKBONES_EXP06`. Deferring that evaluation is reasonable; defer its new routing explicitly too.

5. **Historical reuse is admissible for A/B/C/D/E/F, but A′ requires a different presentation path. Publication needs separate phase outputs.**

   The current admission situation is:

   | Arms | Admissible reuse |
   |---|---|
   | **A, B** | Existing reconstructed legacy receipt plus inherited exp_02 completeness/protocol checks. They do not have exp_06-style child completions. |
   | **C, D, F, E** | Existing HAA child/job records, unchanged execution dependencies, original approval blobs, checkpoint identities, headings, and protocol checks. |
   | **A′ released** | **Not covered by the existing per-query legacy receipt.** Available as a clearly labelled external row copied from exp_02’s approved canonical statistics. |

   I inspected the actual receipt: it declares exactly `['control', 'cyl']`, contains **126 files**, and contains no released-arm artifact entries. All 126 hashes matched. Its inclusion of exp_02’s canonical `stats.json` permits a bound historical summary row for A′; it does not certify A′’s individual artifacts for new paired inference.

   Do not add `released` to global `LEGACY_ARMS`: the existing receipt validator requires exact arm and file membership. If new per-query inference involving A′ is needed, create a separate, explicitly approved supplemental receipt and admission path.

   For C/D/E/F, static checks found matching recorded source-file hashes and approval-blob hashes for **31 children per arm**. The shared registry digest remains `f9852f85…`. This supports the proposed reuse, provided those executable dependencies stay unchanged. Blob-at-commit approval reads permit approval refills; they do not permit arbitrary source changes.

   For the scientific table, A is the matched baseline for D/H. If A′ remains the prominently named “xRIR” row, distinguish its released initialization and include A visibly enough that readers cannot interpret A′ → D/H as a cue-only intervention. A′ may remain an external reference row, but must not silently substitute for A in the ablation.

   Two publication fixes are also required:

   - Phase 1 and Phase 2 cannot both publish to the same already-existing `stats.json`/`summary.txt`; `write_outputs` deliberately refuses overwrites. Register separate phase outputs, or reserve the final canonical paths until all planned arms finish.
   - “Byte-identical exp06/exp09 outputs” should mean unchanged statistical payloads and rendered summaries under controlled fixtures, plus untouched historical canonical files. A newly produced JSON necessarily has different producer/provenance fields after a code change.

   The new summarizer configuration also needs multiple screens and distinct P3 decisions; the existing schema supports only one screen pair per experiment. Extend that structure without changing old experiment semantics.

6. **Phase-2 recipe parity must include exp_04’s saving constraints and a fully specified completion contract.**

   I compared the requested [exp_01 args](/home/yixunhu/codespace/xRIR_code/ckpt/xRIR_simple_8_shot/args.json) and [exp_04 args](/home/yixunhu/codespace/xRIR_code/ckpt/xRIR_simple_yawaug_8_shot/final/args.json). Their common numerical training recipe agrees. The relevant differences are:

   | Field | exp_01 | exp_04 |
   |---|---|---|
   | `save_every` | **500** | **0** |
   | `epoch_ckpt_every` | **5** | **1** |
   | `yaw_aug` | Absent; historical unaugmented run | **1** |
   | `yaw_aug_seed`, `yaw_aug_width` | Absent | **0, 512** |
   | `vit_*` | Absent; historical M defaults | **512 / 12 / 8 / 512** |
   | `no_save` | Absent | **false** |
   | `train_batches_per_epoch`, `tier`, `param_counts`, `env` | Absent | Recorded |
   | `save_dir` | Original run directory | Exclusive attempt directory |

   **Required change:** define two named recipe profiles, rather than merely allowing `yaw_aug ∈ {0,1}`. In particular, I must require `save_every=0` and `resume=None`. The pinned trainer explicitly rejects yaw augmentation otherwise; copying exp_06’s `save_every=500` would violate this contract.

   H may use `epoch_ckpt_every=1` instead of historical 5 as a declared operational difference to retain the native epoch-12 checkpoint. Missing historical fields must remain documented normalizations or unavailable metadata, not be presented as originally recorded facts.

   Both profiles should fix the common 12-epoch recipe, batch 32 × accumulation 2, seed 0, M dimensions, optimizer/scheduler settings, TF32, full splits, and loader length **9,261**. I additionally fixes the augmentation seed/width and counter constraints. Derived parameter counts must come from the new factory; the encoder increment is exactly **526,336**.

   Carry over the full completion contract explicitly: agreeing startup/retained/checkpoint arguments; strict field types; no truncation or forbidden resume; epochs 1–12 exactly once with finite history; final epoch/batch metadata; tensor keys, shapes, dtypes, and values matching between `epoch_012.pth` and `last.pth["model"]`; input and closure revalidation; committed approvals; zero child exit; closed log and receipt; and external completion before checkpoint promotion.

   HAA completion must likewise preserve validation-based selection, initialization → stage 1 → stage 2 → evaluation lineage, heading binding, full test indices, and job completeness. The old contract is suitable; the new registry and approval routing must actually enforce it.

**Cost and schedule assessment.** The GPU estimates are credible planning allowances. Retained histories give approximately **28.2 h** for exp_01 SimpleViT, **27.4 h** for exp_04 yaw augmentation, and **31.0 h** for exp_06 oriented cylindrical pretraining. Thus 31/28 h plus two HAA queues is a plausible ≈65 GPU-hour Phase-2 estimate, pending the new-model timing probes. Including Phase 1 gives approximately **68 GPU-hours expected**, rather than 65 for the whole experiment.

Given the supplied owner ETAs, Phase 1 can plausibly finish September 29 after an actual exclusive handover. Sequential H/I work on the first card can finish around October 2 under expected timings; waiting for the second card pushes I into October 4. October 3–5 is a reasonable buffered target, not a reservation.

The Phase-2 implementation/review estimate is optimistic because it includes a new approval and completion family, not just model wrappers. The HAA pipeline also does not invoke the standalone launcher’s GPU preflight merely by sourcing its shell helpers; the runbook must explicitly enforce the promised exclusive-card condition.

Resolve the concurrent experiment namespace before implementation: the peer’s `exp_10_yaw_pilot_claude` plan also uses `ckpt/exp10/` and `tools/exp10_*`. Distinct subdirectories may avoid immediate collisions, but the shared experiment identifier and ownership need an explicit agreement.

Finally, use the SOP’s **ultra** setting for subsequent review closure, or record an authorized exception. This review ran at **xhigh** and should not be labelled otherwise.

**Verification limits:** I performed read-only source/record inspection, the bounded CPU embedding check, legacy-artifact hashing, historical HAA source/approval checks, and source-file hash checks across all ten completed simulated-evaluation manifests. Their recorded sim source files matched. I did not invoke finalizers, run the full suite, launch GPU work, signal processes, or save files.

