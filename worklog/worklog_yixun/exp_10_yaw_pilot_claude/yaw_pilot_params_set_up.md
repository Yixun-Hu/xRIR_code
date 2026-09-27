# exp_10 yaw_pilot — parameters and set-up (draft before launch; the reviewed commit, execution ids and timings are filled in at each launch)

| Item | Value |
|---|---|
| Tool | `tools/exp10_yaw_pilot.py` (new; reviewed branch tip `1ebb87d`, merged as `2408fb9`), comparator `tools/exp10_compare.py`, summariser `tools/exp10_summarize.py` |
| Split / references | AcousticRooms unseen test split, 6 337 queries / 17 rooms, canonical manifest order; K = 8 manifest `ckpt/yaw_rotation/reference_manifest.json` (sha256 `47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d`, seed 0); K = 1 manifest `ckpt/yaw_rotation/reference_manifest_k1.json` (`f6d71f86…`, seed 0) |
| Arms | `released_k8` (`checkpoints/xRIR_unseen.pth`, simple, K = 8) → `released_k1` (same, K = 1, "trained K = 8, evaluated K = 1") → `control_k8` (`ckpt/xRIR_simple_8_shot/epoch_12.pth`) → `cyl_k8` (`ckpt/xRIR_cyl_8_shot/epoch_12.pth`, cylindrical) |
| Angles | k ∈ {0, 64, 128, 256, 384} columns of W = 512 (0°, 45°, 90°, 180°, 270°), condition P (alignment pinned at k = 0) |
| Controls (probe only) | `ctrl_zero_repeat` (fresh k = 0), `ctrl_k128_repeat` (fresh k = 128), `ctrl_full_turn` (k = 512 ≡ 0); tolerance 1e-6 waveform / log-spec, 1e-9 acoustic |
| Probe | 17 canonical batches [0, 16, 32, 94, 110, 125, 141, 157, 214, 277, 292, 307, 322, 338, 353, 369, 381] (272 queries, one intact batch per room) |
| Numerics | batch 16 with canonical padding, float32, TF32 off, cuDNN deterministic, `gl_seed 0` (per-query phase `sample_seed(0, query)`), Griffin-Lim 32 iterations on CPU, native 9 579 samples padded to 9 600, acoustic metrics on the first 8 000 samples; torch 2.0.1+cu117, numpy 1.23.5, Python 3.8.20 |
| Backend | GPU 1 (A6000, co-tenant ≈ 4 GB after the cylindrical-dinov3 ping) or CPU (16 threads) — recorded per arm in `meta.json` (`device`, `protocol_id`, `execution_id`) |
| Outputs | `ckpt/exp10/<arm>_<probe|all>/{per_sample.json, metrics.json, meta.json, wav_k{0,64,128,256,384}.npy}` |
| Statistics | paired query-level percentile bootstrap, 10 000 replicates, seed 0 (convergence seed 1), 95 % two-sided; room-cluster bootstrap over 17 rooms; historical band = exp_04 five-seed SDs (SimpleViT K = 8 EDT 0.186 ms / C50 0.0053 dB / T60 0.009 %; K = 1 0.609 / 0.0133 / 0.101; CylindricalViT K = 8 0.253 / 0.0059 / 0.014), context only |
| Parity gate vs exp_03 | `ckpt/yaw_rotation/sweep_{released,control,cyl}/per_sample_yaw.json`; criteria plan §7 (a)–(c) + A1 identical validity masks |


## Executed runs (from each run's `meta.json`)

| arm | stage | device | n | HEAD recorded at completion (git metadata is captured after inference) | execution_id | protocol_id | wall time |
|---|---|---|---|---|---|---|---|
| released_k8 | probe | cuda | 272 | 5016b63 | `20260927T072809041814Z-bbd9dec45b8547f1a6fbd9a171c76e38` | `ece07c70be99…` | 4.7 min (inference 152 s, inversion 114 s) |
| released_k8 | all | cuda | 6337 | bc0b9df | `20260927T073313236199Z-45c485084fee44f6b795b4f2f5a71ff4` | `ece07c70be99…` | 66.6 min (inference 2140 s, inversion 1708 s) |
| released_k1 | probe | cuda | 272 | f31d8e6 | `20260927T062510342040Z-080109365da74a318a0424ba1f6fed8d` | `95d453169f37…` | 3.1 min (inference 40 s, inversion 131 s) |
| released_k1 | all | cuda | 6337 | f31d8e6 | `20260927T062839218781Z-93fc61b3802c4e01bb2aa2d55e76c591` | `95d453169f37…` | 45.1 min (inference 543 s, inversion 1998 s) |
| control_k8 | probe | cuda | 272 | af49827 | `20260927T023813895915Z-6b0a883b52604bb090477542219691c4` | `e138ca2d0813…` | 20.6 min (inference 189 s, inversion 1025 s) |
| control_k8 | all | cuda | 6337 | 0a2a2e6 | `20260927T025953876010Z-4b35102e25014d87875828367b73a0ff` | `e138ca2d0813…` | 124.0 min (inference 2182 s, inversion 5080 s) |
| cyl_k8 | probe | cuda | 272 | f290f42 | `20260927T050438791078Z-ab38c2dced99479889a63425c95c9cd5` | `de0bf584f061…` | 5.2 min (inference 159 s, inversion 136 s) |
| cyl_k8 | all | cuda | 6337 | d9c8130 | `20260927T051017740231Z-7d2e7e6db2994f4fa57533077c4c80a0` | `de0bf584f061…` | 73.7 min (inference 2259 s, inversion 2000 s) |
| released_k8 (CPU protocol) | all | cpu | 6337 | f31d8e6 | `20260927T004009003970Z-8796b160e3a84e97b43774e2d4aac00e` | `06d5d3f72f5a…` | 405.6 min (inference 21381 s, inversion 2830 s) |
| released_k8 (CPU protocol) | probe | cpu | 272 | 33d4f75 | `20260927T002126828803Z-f4eccdf1a8c94d4883d9279ab1fc6057` | `06d5d3f72f5a…` | 18.3 min |

Checkpoint sha256s (from meta): released `6cdb02767b4c11e1be78c3438e1b67bfb9351db4e07ee9c72125f68e337325c5`; control / cyl as recorded per run. Tool sha256 (`tools/exp10_yaw_pilot.py`) identical for all runs (merged `2408fb9`). Exp_03 pin verification recorded in every meta (`verify_exp03_pins.ok = true`, 12 files at `62c9107b…`).

## Chain-start HEADs (first line of each chain log) and the reviewed implementation

| chain | started | HEAD at chain start | log |
|---|---|---|---|
| released_k8, CPU (probe → full) | 2026-09-26 20:21 | `ee31cbc` | `yaw_pilot_2026-09-26_20:21:22_released_k8_chain_cpu.log` |
| GPU queue: released_k1 (failed, K = 1 hash bug) then control_k8 | 2026-09-26 22:38 | `0a770d1` | `yaw_pilot_2026-09-26_22:38:03_released_k1_chain_cuda_ABORTED_k1_hash_bug.log`, `yaw_pilot_2026-09-26_22:38:09_control_k8_chain_cuda.log`, `yaw_pilot_2026-09-26_22:37:58_gpu1_queue.log` |
| cyl_k8 (queue) | 2026-09-27 01:04 | `0a2a2e6` | `yaw_pilot_2026-09-27_01:04:35_cyl_k8_chain_cuda.log` |
| released_k1 (after-queue) | 2026-09-27 02:25 | `d9c8130` | `yaw_pilot_2026-09-27_02:25:07_released_k1_chain_cuda.log` |
| released_k8, GPU re-run | 2026-09-27 03:28 | `5016b63` | `yaw_pilot_2026-09-27_03:28:06_released_k8_chain_cuda.log` |

These HEADs differ only in `worklog/` bookkeeping; the evaluator and its closure are the merged tools of `2408fb9` throughout: `tools/exp10_yaw_pilot.py` sha256 `e727cba016dcc2d78c7ba193fca85b75116c40f9e9d567be784970a2974c8823` is recorded identically in every run's `meta.json` (`tool_sha256`), and every run's `verify_exp03_pins` block matches the 12 reviewed files at `62c9107b…`. The "HEAD recorded at completion" column above is the commit checked out when each run finished writing, not the launch commit.
