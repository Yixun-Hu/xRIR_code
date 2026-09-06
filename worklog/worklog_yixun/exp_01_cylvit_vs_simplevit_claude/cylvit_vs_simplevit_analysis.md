# Analysis — cylvit_vs_simplevit

## Outcome

**On this test split, the user's hypothesis is supported on early-reflection accuracy (EDT) and the training objective, with no detected difference on clarity (C50) and reverberation time (T60). Generalisation of that advantage to the population of unseen rooms is not established: with the 17 held-out rooms as the resampling unit every interval includes zero.** Averaged over the eight post-warm-up checkpoint pairs (epochs 5–12), CylindricalViT + xRIR versus the same-budget SimpleViT + xRIR control:

- EDT error −3.7 % (0.0454 vs 0.0472 s): query-level CI [−0.0024, −0.0011]*, room-cluster CI [−0.0039, +0.0001]; cylindrical lower in 8/8 epochs and in 11/17 rooms.
- Test loss −1.2 %: query-level *, room-cluster [−0.00039, +0.00002]; 8/8 epochs. log-STFT MSE −1.3 %: query-level *, room-cluster includes 0; 7/8.
- C50 −0.7 % and T60 −0.4 %: no detected difference under either bootstrap.

The best-checkpoint comparison (cyl ep 9 vs ctrl ep 10) gives the same picture (EDT −3.4 % query-level *, room-cluster [−0.0035, +0.0000]; MSE and loss significant under both bootstraps; C50/T60 no detected difference). The seen split (epoch 9) orders the models the same way on all five metrics. "No detected difference" is not evidence of equivalence; no equivalence margin was pre-registered for this experiment.

## Is the result reliable?

**Design strengths.** Identical data order, loss, optimizer, schedule, precision and budget for both models; only the geometry encoder differs (verified: same token layout, 32.12 M vs 32.08 M params). Every comparison is paired on the same 6337 queries with bootstrap CIs; the final judgment averages over eight checkpoints so single-epoch noise (T60 swung by up to 1.4 points between adjacent epochs) cannot drive it. The control is a strong baseline: it reproduces or beats the paper's Table 1 from epoch 3 (EDT 0.049 vs 0.055, C50 1.25 vs 1.46), and the released checkpoints themselves reproduced within noise beforehand.

**Corrected claim (review finding, 2026-09-05).** The evaluation references were *not* identical across models (24 % of queries differed; see `_results.md`), and Griffin-Lim phases were unseeded. Both are nuisance factors with zero expected effect on the paired difference (their realised effect in these arrays is unknown), so in expectation they widen the intervals rather than bias them; but the original "identical references" statement was wrong and the paired test is less powerful than claimed. The epoch-12 checkpoints will be re-evaluated with a hashed reference manifest and seeded Griffin-Lim as the k=0 gate of exp_03, which is the properly paired version of this comparison.

**Limitations.** (1) One training seed per model: the 3–4 % EDT margin is established across epochs but not across seeds; seed-to-seed variance of the final metrics is unknown. (2) The schedule is a 12-epoch compression of the authors' 200 epochs (both models plateaued by epoch 5, so this mostly affects absolute level, but the small C50/T60 margins could move under longer annealing). (3) Batch 64 was realised as 32×2 accumulation (audio-encoder BatchNorm sees 32) and TF32 was on; both apply equally to both models. (4) Two orchestration slips (epoch-2 snapshot race, epoch-8 compare crash) were recovered without affecting model weights or metrics, but are logged. (5) No cross-model code review was performed (pre-SOP); verification was by self-tests, smoke runs and agreement of the new evaluator with the original script.

**Effect size versus noise.** Adjacent-epoch EDT movement is ≈0.001 s for a fixed model (checkpoint drift, not evaluation noise); the between-model EDT gap is 0.0017–0.0022 s and has the same sign in 10 consecutive epochs (3–12). C50 and T60 gaps are below their epoch-to-epoch noise. I judge the EDT/objective result reliable for this fixed split at the single-seed level; the room-cluster analysis shows the effect is concentrated in a few rooms (Cafe, Auditorium, Office_idx_10, Restaurants_idx_24) and is not yet demonstrated to generalise across rooms. C50/T60: no detectable effect at this budget.

## Interpretation

The equivariant encoder's gain concentrates in EDT, the metric most sensitive to the geometry of early reflections, which is exactly what the panorama encoder feeds. Late reverberation (T60) and clarity (C50) are dominated by the reference RIRs the model re-weights (xRIR's output is a weighted sum of reference log-spectrograms), so the encoder swap has little leverage there. Training loss curves were indistinguishable; the difference is in generalisation to unseen rooms.

## Recommended next steps

1. The exp_03 k=0 gate: a properly paired re-evaluation (hashed reference manifest, seeded Griffin-Lim) of the epoch-12 checkpoints and the released checkpoint, reported with query-level and room-cluster CIs.
2. A second training seed per model (≈30 h per pair on two A6000s) to put a seed-level CI on the EDT margin; more held-out rooms would be needed for room-level inference.
3. exp_03 (yaw-rotation robustness) to test the mechanism directly.
4. Retrospective Codex review of the experiment code (the results-page generators were reviewed 2026-09-05: reject → fixed; see `_worklog.md`), and committing it as the base for later work.
