# K = 8 training dynamics: xRIR (SimpleViT) vs xRIR (CylindricalViT)

`xrir_training_dynamics_k8.{png,pdf}`, `training_dynamics_k8.csv`, generator `plot_training_dynamics.py`
(2026-09-24, CPU, no new evaluation).

Data: exp_01's per-epoch evaluations of the same-budget SimpleViT control and the CylindricalViT
(`ckpt/xRIR_{simple,cyl}_8_shot/eval_unseen_epochNN.json`, epochs 1-12): mean EDT / C50 / T60 error
over all 6 337 queries of the AcousticRooms unseen split at K = 8, epoch checkpoints only (the trainer
keeps no step-level checkpoints; `last.pth` is an overwritten resume snapshot). x-axis: optimizer steps
= epoch x 296 334 / 64 (batch 32 x 2 accumulation), i.e. ~4.6k steps per epoch, 55.6k at epoch 12.

Caveats for a caption: exp_01's per-epoch evaluations drew the K = 8 references at random per model
and did not seed Griffin-Lim, so the two curves are not paired (exp_03's paired k = 0 comparison at
epoch 12 is the confirmatory endpoint: CylindricalViT EDT -4.4 %, C50 no difference). No K = 1
per-epoch evaluations exist; the xRIR family has no retrieval metric.
