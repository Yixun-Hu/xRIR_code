# Query log — cylvit_vs_simplevit (user: yixun)

> **Retrospective record.** This experiment ran on 2026-09-04/05 before `worklog/experiment_SOP.md` was adopted in this repo. Queries are reproduced verbatim from the session; summaries/hypotheses are as understood at the time.

## Q1 — 2026-09-04T12:18-04:00
**Verbatim:** "Could you please use the main branch to first reproduce the xRIR results? @xRIR_pdf.md"
**Summary:** Reproduce the paper's Table 1 (xRIR, K=8, seen and unseen AcousticRooms splits) with the released checkpoints on `main`.
**Assumption:** The released checkpoints and the repo's eval scripts reproduce the paper's numbers; this establishes the baseline and noise floor before any new method.
**Why:** Evaluation-integrity prerequisite: calibrate the pipeline on the published configuration before comparing anything to it.

## Q2 — 2026-09-04T12:24-04:00
**Verbatim:** "Please substitute the simplevit with our @model/cylindrical_vit.py to perform the same few-shot synthesis task. The hypothesis is that our method will outperform the xrir as the baseline method."
**Summary:** Replace xRIR's SimpleViT geometry encoder with the user's azimuth-equivariant CylindricalViT and compare on the same few-shot RIR task.
**User's hypothesis:** CylindricalViT + xRIR outperforms SimpleViT + xRIR.
**Why:** Tests whether azimuth equivariance of the geometry encoder improves cross-room RIR prediction; the core research claim of the user's method.

## Q3–Q6 — 2026-09-04T13:18 … 2026-09-05T00:10-04:00 (status queries)
**Verbatim:** "What is your current status" · "What is the training and evaluation resutls for cylindrical vit?" · "What is the paper's checkpoint (how many epochs)" · "What is current status>" · "How many epochs?" · "How many training steps?"
**Summary:** Progress and budget questions during the run (answered from logs; see `_worklog.md`).

## Q7 — 2026-09-05T15:0x-04:00
**Verbatim:** "So the table you gave me "Epoch 9, full unseen split, paired on 6337 samples …" was from how many training steps, trainig on what dataset and eval on what dataset split?"
**Summary:** Requests the exact provenance (steps, train set, eval split) of the epoch-9 table. Recorded in `_params_set_up.md` and `_results.md`.

## Q8 — 2026-09-05T15:10-04:00
**Verbatim:** "Also evaluate both on the seen split"
**Summary:** Add a seen-split evaluation of both models (done at epoch 9; see results and the contamination caveat).

## Q9 — 2026-09-05T15:3x-04:00
**Verbatim:** "I saw the training epoch is `num_epoch = 200` inside @train_xRIR_unseen.py , why previously for simplevit + xRIR and cylindricalvit + xRIR you run 9 epoch? and the decay_epochs etc. params are different?"
**Summary:** Challenge on the training budget/schedule deviation from the authors' script. Answered in `_params_set_up.md` (compressed 12-epoch schedule, same LR endpoints, same 4:1 ratio).

## Q10 — 2026-09-05T15:4x-04:00
**Verbatim:** "So @xRIR_pdf.md is using single card for 200 epoch training? Is that acheievable with H200？"
**Summary:** Feasibility of the authors' budget on one H200 (profiled: ~10 days as-is, ~6 with vectorized STFT; paper gives no hardware/epochs).

## Q11 — 2026-09-05T15:5x-04:00
**Verbatim:** "If epoch=200 is not a explicit setting, is our current simplevit + xRIR match the table 1's results inside @xRIR_pdf.md ?"
**Summary:** Whether the retrained control matches Table 1 (yes: at/above from epoch 3; see `_results.md`).

## Q12 — 2026-09-05T16:0x-04:00
**Verbatim:** "Since currently our 9 epoch version match the released performance, do we still need to vectorize the STFT computing?"
**Summary:** Decided not to change the baseline STFT loop for this experiment (provenance; only matters for 200-epoch runs).
