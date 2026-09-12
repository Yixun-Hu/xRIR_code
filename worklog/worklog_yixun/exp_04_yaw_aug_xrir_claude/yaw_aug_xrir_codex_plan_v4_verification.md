# Verification of exp_04 plan v4

**Reviewer:** OpenAI Codex, interactive workspace review with three parallel independent audits (augmentation/training, statistics, operations). **Date:** 2026-09-12 14:10 -0400. **Scope:** verify the existing plan, its three earlier reviews, FLAC exp_17 implementation, and current xRIR code and experiment records. This is a plan review; the proposed implementation does not yet exist.

**Verdict: request localized changes before approval.** The augmentation semantics and registered statistical rules are sound for the stated comparison of fixed checkpoints on this split. V4 closes most earlier findings, but several execution contracts remain contradictory or incomplete. No new training arm or scientific redesign is required to address the findings below.

1. **Must fix: the prescribed GPU setting fails the exact parity gate.**

   [Plan line 15](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_04_yaw_aug_xrir_claude/plan_yaw_aug_xrir.md:15) includes `CUDA_VISIBLE_DEVICES` in the normalized configuration and omits it from the exact exclusion list. The [reported exp_01 SimpleViT command](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_command.md:38) used GPU 1; the treatment command at plan line 56 requires GPU 0. The intended launch therefore fails the required comparison. Explicitly allow and record physical GPU placement as an operational difference, retaining checks for the intended one-GPU device class. Update the single shared exclusion/normalization rule everywhere it is cited.

2. **Must fix: complete the training source closure and remove the profile hash cycle.**

   The training closure explicitly enumerated at plan line 32 omits `utils/lr_scheduler.py`, although `train_xRIR_backbone.py:30` imports its custom `ExponentialLR`. Its contents affect optimization, so the promised before/after training checks must bind this file too. Use the complete transitive repository import closure of the trainer, rather than the incomplete enumerated list.

   Separately, plan lines 41–44 generalize exp_03's import-based source closure and place the producer's approved closure digest inside `tools/exp04_profiles.py`, which the producer must read. The existing helper (`yaw_rotation_degradation_results_assets/bind_provenance.py:64–73` in exp_03) includes transitive local imports. A direct implementation therefore includes the profile containing the expected producer digest in the bytes being hashed; filling the digest changes those bytes again. A second commit alone does not resolve that dependency.

   Specify separate immutable analysis-profile data and code-closure digests, linked by an external reviewed binding. The profile data must still be authenticated, but must not be a member of the closure whose expected hash it contains. Define the same separation for later checkpoint-hash additions so they do not invalidate previously executed code bindings.

3. **Must fix: give bootstrap convergence an executable escalation policy.**

   Plan line 20 requires raising `n_boot` until the gate passes, but lines 43–44 hard-code 20,000 in an immutable profile and prohibit analytical overrides. Register a deterministic sequence, for example 20,000 → 40,000 → 80,000, with progression based only on numerical convergence, both bootstrap seeds rerun at each stage, and the used count recorded. Specify that exhausting the sequence produces no confirmatory verdict. This preserves preregistration without requiring a post-result profile/code amendment that can invalidate provenance.

   Also reuse the existing zero-width semantics in `tools/summarize_yaw.py::convergence_cell`: zero movement with zero width passes; positive movement with zero width fails. The invariant-arm synthetic test would otherwise encounter an undefined 0/0 gate.

4. **Correct the elapsed-time claim or explicitly stage the evaluation bindings.**

   Plan line 43 pins the augmented checkpoint after training completes and before *any* confirmatory evaluation; lines 31 and 64 promise evaluation overlapping training. Taken literally, the former prevents the latter. The per-run estimates sum to approximately 16.4 evaluation hours (reasonably rounded to 17), so the written lifecycle implies roughly 30 + 17 = 47 elapsed hours before final analysis.

   If overlap is intended, freeze comparator-specific bindings before training and permit their evaluations while training runs; bind the augmented checkpoint after completion without changing prior run manifests. Approximately 6.4 hours of historical-comparator evaluation can overlap. Approximately 9.2 hours of augmented epoch-12 evaluation necessarily follows training. The approximately 0.8-hour epoch-9 diagnostic can overlap only if its own checkpoint/provenance lifecycle explicitly permits it. A simple staged schedule is therefore around 39–40 hours, plus validation, hashing and orchestration overhead; these are estimates, not measurements of the new implementation. Keep GPU-hours separate from elapsed hours.

5. **Update code-review assignments to the current SOP.**

   Plan lines 34 and 52 assign per-round and final code review to Codex. The current [experiment SOP](/home/yixunhu/codespace/xRIR_code/worklog/experiment_SOP.md:10) assigns Codex as Coder and requires the opposite model family, Claude, to review that code. State the actual Coder/Reviewer pairing and use it consistently. Codex review of the Claude-authored plan remains appropriate; this finding does not invalidate the earlier plan reviews.

6. **Clarify the descriptive table's finite-value population.**

   Plan line 46 builds table cells from per-seed aggregate metric JSONs. The current `eval_yaw_rotation.py::_summarize` (line 523) averages each seed's individually finite queries. H1 at plan line 22 instead uses queries finite across all five seeds and both arms. If invalid metrics occur, the displayed table means and inferential means can describe different populations.

   Either explicitly label the table as descriptive seed-specific finite means with validity counts, and report H1's matched complete-case means/counts separately, or derive table values from per-sample JSON using a registered common mask. The broad “every reported cell” masking language at plan line 20 must agree with the selected policy. No extra evaluation runs are needed.

The trainer row at plan line 40 also retains stale wording that rejects only `save_every > 0`. Make it agree with the authoritative rule elsewhere: reject every nonzero value.

**Verified evidence:**

- The positive column roll plus active positive `Rz` matches xRIR's coordinate convention and FLAC's scene transform. Rotating depth coordinates, query-source coordinates and every reference-source coordinate while preserving raw target/reference audio is correct. The disclosed float32 alignment effect is consistent with the current model.
- An independent CPU check using the pinned Torch 2.0.1 generator reproduced all four mixer seeds and first offsets. All 111,132 reportable rank-0 mixer outputs are unique. The micro-batch counter is a reasonable xRIR adaptation; the mixer is verbatim, while counter units and training seed are experiment-specific.
- 296,334 samples give 9,261 micro-batches and 4,631 optimizer updates per epoch, including the short final batch. Epochs 9 and 12 correspond to 41,679 and 55,572 updates. The historical control's history sums to 28.2152 hours; epoch 1 took 2.31526 hours, giving a +5% threshold of 2.43103 hours.
- H1 uses the correct one-sided family-2 upper quantile, 0.975. H2 uses the correct family-8 upper quantile, 0.99375. Family-14 TOST uses tails `0.05/14` and `1−0.05/14`; TOST combines the two one-sided bounds into a `100(1−2α)%` interval, as described in [NIST Technical Note 2106, section 5.4](https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.2106.pdf). This verifies the tail allocation, not exact finite-sample coverage of a percentile bootstrap.
- The paired all-seed masks, ordinary per-query seed means, ratios recomputed within shared resamples, separate claim families, and explicit epoch-12 selection are internally consistent. The expanded augmented-arm evaluation block covers every registered TOST cell.

**Scientific interpretation to retain:** five evaluation seeds quantify reference-selection and Griffin–Lim variability for these checkpoints. They do not estimate variability across training runs. Query-level intervals under the registered within-split resampling scheme do not establish replication across new rooms; the room-cluster analysis is secondary. The plan already discloses the historical-control and single-training-seed limitations, so an additional control retrain is not required for its deliberately limited claim.

**Recommendation on the five user decisions:** retain 12 epochs with epoch 12 confirmatory and epoch 9 diagnostic; augment only SimpleViT; use five evaluation seeds for every table row, including fresh comparator evaluations; keep tests in the existing `tests/` directory and generate the living table; keep retrieval R@1/5/10 undefined. Preserve earlier single-seed outputs as historical diagnostics when replacing displayed table values. Approval should follow the localized contract corrections above and retain the planned validation ladder and final independent code review.

Only this review artifact was added. The plan and implementation were not modified, and no training or GPU evaluation was launched.
