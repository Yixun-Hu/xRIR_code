# Qualitative RIR comparison: xRIR (SimpleViT) vs xRIR (CylindricalViT)

Generated 2026-09-23 by `make_qualitative_figure.py` (CPU; run from the repo root in the
`xRIR` env with `PYTHONPATH` set and `XRIR_DATA_PATH=~/data_cache/AcousticRooms`).

Files: `qualitative_sim_real.{pdf,png}` (combined, 8 columns), `qualitative_sim.{pdf,png}`,
`qualitative_real.{pdf,png}`, `qualitative_selection.json` (selected queries, the canonical
per-sample metrics they were selected on, the metrics recomputed for the displayed
predictions, checkpoints, manifests, seeds).

Layout (as xRIR Fig. 4 / DALL-E Fig. 3): columns = queries, rows = depth panorama at the
query receiver (model input; red star = query source direction, green circles = the K = 8
reference sources), waveform (ground truth / SimpleViT / CylindricalViT, ground truth in
grey behind each prediction, EDT/C50/T60 errors in the box) and log-magnitude STFT in dB
(shared colour scale per column, 60 dB range below the ground truth's 99.5th percentile,
log-STFT MSE in the box).

Sources. Sim: exp_04's confirmatory K = 8, seed-42, condition P, k = 0 evaluations of
exp_01's same-budget SimpleViT control and CylindricalViT (epoch 12) on the unseen split
(`ckpt/yaw_aug/eval/{control,cyl}_k8_seed42_k0`), references from
`ckpt/yaw_aug/reference_manifest_k8_seed42.json`, per-query seeded Griffin-Lim (seed 42),
metrics on the first 8000 samples — the recomputed values equal the canonical per-sample
values to the printed precision. Real: exp_02's seed-0 two-stage fine-tuned checkpoints
(`ckpt/sim2real/{control,cyl}/seed0/stage2_<room>/best.pth`), DiffRIR test split, eval_seed 0
references, metrics on the full 9600 samples; exp_02's Griffin-Lim was unseeded, so the
displayed (seeded) predictions differ slightly from the canonical values used for selection
(both are in the JSON).

Selection rule (both environments): among queries where the cylindrical error is lower on
every available metric (EDT, C50, T60; T60 skipped for `dampened_room`) and the cylindrical
EDT error is at most the median baseline EDT error, rank by
0.5 * mean relative improvement + 0.5 * tanh(mean improvement in median-baseline-error units / 2);
sim: best query per room category, top 4 categories; real: best test query per room.
These are deliberately the largest per-query wins of the cylindrical model, not typical
queries: in sim the cylindrical model is better on all three metrics for 1 374 / 6 337
queries (EDT −4.4 % on average, exp_03 paired k = 0), and on HAA after fine-tuning the
SimpleViT control is better on aggregate in 6 / 11 room×metric cells (exp_02), so the real
columns show per-query wins in a setting the cylindrical model loses overall.

## Classroom environment panels (`make_classroom_env.py`, 2026-09-23)

`classroom_environment.{png,pdf}`: (a) the authors' photo of the classroom (HAA project page,
`static/images/Classroom.jpg`, Wang et al. CVPR 2024 — credit it), (b) a 3D render of the
DiffRIR planar model from the authors' repository (`rooms/classroom.py`: 7.12 × 7.92 × 2.74 m
box + three table slabs) with the 630 microphone positions, the 12 training RIRs used as
references and the speaker, (c) the floor map, (d) the depth panorama the xRIR models see
(rendered at the speaker; `~/data_cache/HAA_xrir/class_room/depth.npy`) with the reference
microphones' directions marked. Run with `--photo <path to Classroom.jpg>` (not stored in the
repository). The HAA dataset itself ships no photos, meshes or panoramas — only RIRs and
coordinates; the geometry is the code-defined plane list.
