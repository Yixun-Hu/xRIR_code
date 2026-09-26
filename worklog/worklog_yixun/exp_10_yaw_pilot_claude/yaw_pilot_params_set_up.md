# exp_10 yaw_pilot — parameters and set-up (draft before launch; the reviewed commit, execution ids and timings are filled in at each launch)

| Item | Value |
|---|---|
| Tool | `tools/exp10_yaw_pilot.py` (new; reviewed commit filled at launch), comparator `tools/exp10_compare.py`, summariser `tools/exp10_summarize.py` |
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
