# exp_05 param_efficiency — parameters and set-up (retrospective, 2026-09-18)

Reconstructed from `effective_args.json`, `train_manifest.json`, probe receipts and `tools.exp05_params` after the runs (Codex record review finding 6). The M pair is exp_01's (`ckpt/xRIR_simple_8_shot/epoch_12.pth`, `ckpt/xRIR_cyl_8_shot/epoch_12.pth`, 2026-09-04/05, launcher v1, PYTHONHASHSEED unrecorded).

| Arm | Tier | Backbone | dim/depth/heads/mlp | Encoder / full params | Recipe | Reviewed commit | GPU | Probe T_run (h) / peak (GiB) | Cumulative hours incl. probes (ledger) | Certified training wall (h) | Epoch-12 test loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S_simple | S | simple | 256/6/4/256 | 2766080 / 15073213 | bs 32×2, lr 0.001, wd 0.0001, decay 3/0.1, 12 ep, seed 0, TF32 True, workers 12 | f19b9b6 | 1 | 11.0 / 11.3 | 11.4 | 11.3964 | 0.0159933 |
| S_cylindrical | S | cylindrical | 256/6/4/256 | 2777984 / 15085117 | bs 32×2, lr 0.001, wd 0.0001, decay 3/0.1, 12 ep, seed 0, TF32 True, workers 12 | f19b9b6 | 1 | 11.9 / 11.3 | 12.0 | 11.9857 | 0.0159770 |
| L_simple | L | simple | 768/12/12/768 | 43709184 / 56147389 | bs 32×2, lr 0.001, wd 0.0001, decay 3/0.1, 12 ep, seed 0, TF32 True, workers 12 | 862923e | 0 | 41.4 / 44.4 | 41.8 | 41.6595 | 0.0162409 |
| L_cylindrical | L | cylindrical | 768/12/12/768 | 43780608 / 56218813 | bs 32×2, lr 0.001, wd 0.0001, decay 3/0.1, 12 ep, seed 0, TF32 True, workers 12 | f19b9b6 | 1 | 46.0 / 44.4 | 46.2 | 46.0840 | 0.0159641 |

Environment (all four): `OMP_NUM_THREADS=2 PYTHONHASHSEED=0 XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`; torch 2.0.1+cu117, Python 3.8.20, RTX A6000 48 GB; training data identity per manifest (`train_data_identity`), 296 334 samples, 9 261 micro-batches per epoch.

Evaluation: unseen split 6 337 queries, K = 8 and K = 1, seeds 42–46 (manifests `ckpt/yaw_aug/reference_manifest_k{8,1}_seed*.json`), batch 16, TF32 off, `--conditions P --yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols --decomposition-batches 0`, plus one seed-42 K = 8 yaw block per arm (k ∈ {0, 32, 64, 448, 480}); entry point `tools/exp05_eval.py --tier`; reviewed commits `3e2c28b` (L arms) and `ddfcb66` (S and M arms). Approval pins: `param_efficiency_results_assets/approved_digests.json` (filled at `61f4d1b`).


Wall-time definition: "Certified training wall (h)" = `train_manifest.started_at` → `execution.ended_at` of the certified attempt (launcher-side stamps, includes the final test epoch and save); the generated results table reports the trainer-side sum of epoch times, which differs by a few minutes.
