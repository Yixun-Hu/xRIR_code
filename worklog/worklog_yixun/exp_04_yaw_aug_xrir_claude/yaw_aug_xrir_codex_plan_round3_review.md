# Codex plan review — yaw_aug_xrir (exp_04), plan v3, round 3

**Reviewer:** OpenAI Codex CLI (`codex exec -s read-only`, model gpt-5.6-sol, reasoning ultra) · **Date:** 2026-09-12 11:48:00 -0400 · prompt: review_prompts/plan_round3_prompt.md

---

# Codex plan review — exp_04 “yaw_aug_xrir”, round 3

## Round-2 closure

| # | Status | Assessment |
|---|---|---|
| 1 | Closed | All five augmented seeds cover `{0,32,64,128,256,384,448,480}`; TOST is one-arm and the ≈17 GPU-hour budget includes it. |
| 2 | Closed | Bootstrap seeds 0/1, the correct companion intervals, endpoint-movement ratio, 0.10 threshold, and artifact-suppression rule are explicit. |
| 3 | **Not closed** | The manifest fields are strong, but training provenance is not execution-bound, evaluation finalization cannot safely hash its live log, post-run revalidation is incomplete, and the requested fault tests are absent. |
| 4 | **Partially closed** | Profiles, exact CLI, one-arm TOST, five-seed admission, masks for H2/TOST, and non-self-referential provenance are added. H1’s mask and external closure/schema admission remain incomplete. |
| 5 | **Partially closed** | Cadence, native names, attempt preservation and cumulative budget are fixed. The exact argv/environment and directory-state contracts still cannot pass as written. |
| 6 | **Partially closed** | Rotated-geometry alignment is now fixed with no adaptive branch, but the audit remains inconsistently worded and has no executable producer/artifact. |
| 7 | **Partially closed** | Boundary tests, `PYTHONHASHSEED`, resource thresholds and the 2.431-hour gate are present. Effective-args equality, data-root pinning, strict cadence admission and evaluation-launch tests remain defective. |

## Previously partial/not-closed round-1 items

| Round-1 # | Status | Assessment |
|---|---|---|
| 1 | Closed | Counter, mixer, golden vectors, full-domain collision test and invalid type/range cases are specified correctly. |
| 5 | Closed | H2 is C50-primary/EDT-supportive; TOST has seven angles, two metrics and family 14. |
| 6 | **Not closed** | See finding 2: the immutable-manifest execution lifecycle is incomplete. |
| 7 | **Partially closed** | See findings 3–4: confirmatory admission remains underspecified. |
| 9 | **Partially closed** | The scientific alignment policy is fixed; its audit is not executable. |
| 10 | **Partially closed** | Training/evaluation launch contracts and evaluation validation remain incomplete. |
| 11 | **Partially closed** | Counts, checkpoint policy and numerical budget are corrected; matrix execution and canonical input selection remain ambiguous. |

Verified independently:

- The seed-0 mixer outputs are `{1254532375,2068746652,2466316700,368013681}`, with first W=512 offsets `{474,395,358,401}` at `t={0,1,9261,111131}`; all 111,132 reportable seeds are unique.
- `296,334` samples give `9,261` micro-batches and `4,631` updates per epoch: `41,679` through epoch 9 and `55,572` through epoch 12.
- H1’s \(Q_{0.975}\), H2’s \(Q_{0.99375}\), H2’s common mask, and the family-14 TOST rule are correct.
- The historical control totals 28.215 hours; its epoch 1 is 2.315 hours and +5% is 2.431 hours. The stated evaluation components total roughly 16.4 hours, so ≈17 hours is reasonable.

## New findings

1. **Blocker — §§2, 5–7 — The exact training launch and parity gate cannot pass.**

   **What is wrong:** The full argv omits `--log-interval 50`; the control records 50, while the current trainer defaults to 20. It also omits the actual control data environment, notably `XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`; without that variable the dataset falls back to the repository’s effectively empty `data/` directory. The control used `OMP_NUM_THREADS=2`, which is also neither pinned nor disclosed. Section 2 and §6 give different args-diff exclusion lists, while “empty output directory” conflicts with writing `effective_args.json` before spawn.

   The no-mid-epoch-save guard rejects only `save_every > 0`; a negative value remains armed by the current trainer’s truthiness/modulo logic.

   **Why it matters:** Preflight either refuses the canonical launch or trains from the wrong data root/configuration.

   **Minimal change:** Pin the interpreter and complete environment in the golden launch; add `--log-interval 50` and all control-relevant flags explicitly; use one normalized args/environment schema in §§2, 5 and 6; define the directory as absent before reservation and containing only whitelisted pre-run files afterward; require `resume is None` and `save_every == 0`, with negative-cadence tests. See [the plan](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_04_yaw_aug_xrir_claude/plan_yaw_aug_xrir.md:55), [trainer defaults](/home/yixunhu/codespace/xRIR_code/train_xRIR_backbone.py:53), and [control command](/home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/exp_01_cylvit_vs_simplevit_claude/cylvit_vs_simplevit_command.md:33).

2. **Blocker — §§4–5 — The provenance lifecycle is not operationally immutable.**

   **What is wrong:** `eval_yaw_rotation.py` is not listed in the planned-files table despite requiring substantial changes. It is assigned responsibility for hashing its own run log and writing the completion sidecar, but that log is still being written/drained by `tee` until the child exits. Only the checkpoint and evaluation-manifest digests are explicitly rechecked afterward; the reference manifest, data inventory and source closures are not.

   The augmented training attempt likewise has no pre-run manifest plus post-run completion record binding its source/data closure, effective args, closed log, history and checkpoint hashes. Symlink promotion alone does not make the attempt immutable. Historical exp_01 training provenance must be described as reconstructed, since it was not execution-bound at launch.

   The stale-directory and changed-manifest/source/checkpoint tests explicitly requested in round 2 are not enumerated.

   **Why it matters:** A completion sidecar may bind an incomplete log or associate outputs with inputs that changed during execution.

   **Minimal change:** Add explicit planned rows for `eval_yaw_rotation.py` and a named reusable provenance helper. Have launchers—not the running child—finalize sidecars only after the child and log sink close, then revalidate every declared mutable digest. Add an augmented-training manifest/completion sidecar before promoting `final`; label legacy training provenance honestly. Test existing directories, child failure/missing output, and mid-run changes to manifest, reference manifest, checkpoint, source and data.

3. **Blocker — §§3, 5 — H1 has no defined common finite mask.**

   **What is wrong:** The introductory paragraph refers to “a cell’s mask,” but only H2 and TOST define theirs. H1 never states whether every one of the five seed values in both arms must be finite.

   **Why it matters:** Separate arm masks or partial-seed averaging would break the paired estimand and could give the numerator and denominator different query populations.

   **Minimal change:** Define the H1 mask as finite for all five seeds in both arms at k=0, then take ordinary five-value means. Refuse an empty/undefined cell, report exclusions, and test non-finiteness in either arm and every seed position.

4. **Blocker — §§4–5, 8 — Confirmatory producers can still admit self-consistent but unapproved provenance.**

   **What is wrong:** Profiles pin checkpoints and reference manifests, but do not explicitly pin the approved manifest/sidecar schema, evaluator and launcher/writer closures, dataset/query identity, or asymmetric per-role H2 grids. `paired_compare.py` is not expressly required to compare its current producer/profile closure with the reviewed digest carried by every input manifest.

   `results_table.py` has no exact CLI/input schema, checkpoint/role admission, completion-sidecar validation or provenance sidecar. It is described as writing Markdown only, although §8 says downstream results consume its “canonical JSON.”

   **Why it matters:** Wrong-code, stale, exploratory or tampered five-seed data could still enter a verdict or paper table.

   **Minimal change:** Put schema versions, externally approved closure/data digests and exact role-specific grids in each profile; validate them independently of the run’s own declarations. Require exclusive/atomic canonical outputs and provenance verification by downstream readers. Give the living-table producer an exact profile-bound CLI, canonical JSON plus sidecar, and tampering/wrong-role/overwrite tests.

5. **Should-fix — §§3–5 — The written evaluation matrix does not match the evaluator or uniquely select canonical k=0 inputs.**

   **What is wrong:** The plan says the registered sweeps have “no E condition,” but the current evaluator always computes and serializes both P and E. The standalone runs and yaw blocks also create two complete K=8 k=0 datasets for control and augmentation, without saying which feeds H1 and the living table. Finally, §3 promises descriptive cross-arm \(D_k\) at other acoustic and spectral angles, but no fresh full-grid control run exists.

   **Why it matters:** Manifests may disagree with actual output, and duplicated realizations allow accidental source selection.

   **Minimal change:** Either implement and test a strict P-only mode, or explicitly bind and label the extra E spectral outputs as non-confirmatory. Pin which k=0 set feeds H1/table versus H2/TOST. Narrow the descriptive \(D_k\) statement to available paired angles, or add and budget a matched control full-grid run. See [the evaluator loop](/home/yixunhu/codespace/xRIR_code/eval_yaw_rotation.py:413).

6. **Should-fix — §§2, 5–6 — The alignment audit remains contradictory and unwired.**

   **What is wrong:** Section 2 correctly makes the audit integer-delay-only, but rung 4 still says “integer-delay/gain changes,” reproducing the previously undefined gain criterion. No reviewed function, command or artifact schema produces the claimed deterministic 50-batch audit.

   **Why it matters:** The stated cohort, numerator and denominator cannot be reproduced or checked.

   **Minimal change:** Delete `/gain`; assign the audit to a named reviewed helper; record a cohort/offset digest, changed-delay count and denominator `50×32×8 = 12,800`, using the same rotation/device/op sequence as training.

7. **Should-fix — §3 — The optional superiority claim has no executable bound.**

   **What is wrong:** H1’s non-inferiority bound is correctly \(Q_{0.975}\), but “the two-sided adjusted upper bound” for superiority is not defined or included in the convergence gate. For a family of two at familywise 0.05, that two-sided endpoint would be \(Q_{0.9875}\), not \(Q_{0.975}\).

   **Why it matters:** Implementation could silently choose a different superiority test.

   **Minimal change:** Remove the superiority claim, or pin `[Q_0.0125,Q_0.9875]` and gate that interval’s bootstrap convergence as well.

## Verdict

**Request changes.**

Before the user is asked to approve, v3 must:

1. Repair and fully pin the training argv, environment, normalized args comparison, directory lifecycle, and strict resume/cadence guards.
2. Add execution-bound training completion provenance and a launcher-owned evaluation finalization lifecycle, with the missing mutation/stale/failure tests.
3. Define H1’s all-seed, both-arm common finite mask.
4. Externally pin and validate evaluator/writer/producer/data/schema provenance, and make both canonical producers and downstream consumers fail closed.
5. Reconcile P-only execution, select the canonical duplicated k=0 inputs, and correct the unavailable descriptive \(D_k\) promise.
6. Wire the integer-delay-only alignment audit to a reproducible hashed artifact.
7. Remove or fully specify the superiority test.

No files were modified.
codex exit 0 at 11:59:30
