# Plan — sim2real_transfer (as presented to and approved by the user, 2026-09-05)

> Presented in chat 2026-09-05 ≈15:35, approved ≈16:00 ("Approved, run the sim2real experiments after training finishes"). Reviewer loop: none (pre-SOP). Recorded here verbatim in substance.

## Protocol (paper §5.5, suppl. §9, and the repo's `sim_to_real/` scripts)

- **Rooms**: classroomBase, dampenedBase, hallwayBase, complexBase from Hearing Anything Anywhere (HAA); all four on the diskstation (630 / 276 / 576 / 408 mic positions).
- **Role swap**: the fixed loudspeaker is the model's "receiver" (panorama rendered at the speaker; its `speaker_xyz` equals the repo's hard-coded receiver position), the mic positions are the "sources".
- **Splits**: DiffRIR's official per-room `train_indices` (12 RIRs; identical to the repo's hard-coded lists), `valid_indices`, test = remainder (463 / 198 / 423 / 198).
- **Preprocessing**: 22.05 kHz mono, each room divided by the max amplitude over its 12 training RIRs (README), first 9600 samples used.
- **Stage 1** (repo `finetune_all_rooms.py`): classroom + hallway + complex, dampened excluded; 36 targets, K=8 references re-drawn per iteration from the other 11 training RIRs; init from an AcousticRooms checkpoint; AdamW lr 1e-4, wd 1e-4, ExponentialLR 0.1× per 50 epochs; one full-batch step per epoch, 1000 epochs.
- **Stage 2** (repo `finetune_classroom.py`): per room, all four; init from the stage-1 checkpoint; 12 targets, one step per epoch, 200 epochs, same optimizer.
- **Checkpoint selection**: lowest loss on the DiffRIR **validation** split (paper), not the test split (repo scripts).
- **Evaluation**: per-room test split, K=8 references from the 12 training RIRs with a fixed seed shared by every model (paired), Griffin-Lim inversion, EDT / C50 / T60 on the 9600-sample waveforms as in `eval_classroom.py`; T60 omitted for dampened (paper).

## Runs

| Init checkpoint | Purpose | Fine-tune seeds |
|---|---|---|
| Released `xRIR_unseen.pth` (authors' SimpleViT) | Reproduce Table 2 | 3 |
| Our SimpleViT control, epoch 12 | Same-budget control | 3 |
| Our CylindricalViT, epoch 12 | Hypothesis | 3 |
| All three, no fine-tuning | Zero-shot rows (paper Table 6 style) | – |
| Released, as-released depth maps | Side row (see asset bug below) | 1 |

Reported per room: mean over seeds, plus paired bootstrap CIs (cyl vs control) on per-sample metrics pooled over seeds. Paper targets (xRIR K=8): classroom 0.093 / 1.628 / 6.25, dampened 0.044 / 3.302 / –, hallway 0.062 / 0.954 / 3.20, complex 0.077 / 1.688 / 4.x.

## New files (baseline files untouched)

`sim_to_real/prepare_haa.py`, `sim_to_real/haa_dataset.py`, `sim_to_real/finetune_haa.py`, `sim_to_real/eval_haa.py`, `sim_to_real/run_haa_pipeline.sh`, `sim_to_real/summarize_haa.py`; `tools/compare_eval.py` gains optional metrics / empty-group skipping.

## Decisions confirmed by the user (approved as proposed)

1. Selection on the validation split rather than the test split.
2. DiffRIR's official test split rather than the repo's odd-index split.
3. Repo depth maps, with frame verification first (outcome: released `complex_room.npy` is byte-identical to `hallway.npy`; released `dampened_room.npy` is inconsistent with the mic grid; the cache therefore uses the user's HAA_processed renders for those two rooms and the repo maps for classroom/hallway, with both variants stored).
4. Dampened excluded from stage 1, fine-tuned and reported in stage 2.
5. Three fine-tuning seeds per init; our models at epoch 12 (no test-set selection).

## Acceptance criteria

- Cache: split sizes equal DiffRIR's; propagation delay preserved (onset tracks distance); normalisation scale = training-set peak.
- Every stage reaches its epoch count with finite losses and writes `best.pth`/`summary.json`; every eval writes `metrics_<room>.json` + per-sample file.
- Released-checkpoint fine-tuned rows land near Table 2; zero-shot released classroom near the paper's "pretrained" row (0.204 / 3.43).
