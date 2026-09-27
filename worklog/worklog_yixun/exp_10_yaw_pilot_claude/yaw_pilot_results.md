# Results — exp_10 yaw_pilot

Descriptive pilot (plan v3.1). Per arm and angle: Δ = paired mean change of the error vs ground truth (α − 0°); G = mean shift of the prediction at α relative to the prediction at 0° (no ground truth); both on the shared comparison mask, 10000 bootstrap replicates (seeds [0, 1]), 95 % percentile intervals; query-level CI first, room-cluster CI (17 rooms) second. A multiple G/Δ is printed only where the headline is reportable (Δ interval excludes 0, convergence passed, seed statuses agree); otherwise the cell shows its status. Canonical JSON: `ckpt/exp10/summary/yaw_pilot_summary.json` (sha256 `826498f58780dfe42a4e6228075b0fcb2be845a18a0986d0e835302faee85b32`).

> **band:** historical baseline evaluation variability (references and phases redrawn), context only
> **broader_population:** every query with a finite g_alpha (not the paired mask)
> **gl_free:** logspec_mad and mag_rel_l2 are the Griffin-Lim-free readout of the same shift, computed on the model's direct log-magnitude output
> **pipeline:** waveform and acoustic gaps measure the sensitivity of the model-plus-Griffin-Lim pipeline, conditional on the per-query phase seed; the shared phase initialisation removes initialisation noise, but Griffin-Lim is a nonlinear inverse and can amplify a magnitude change

**Metric qualifications** (canonical, from the summariser): what a Δ and a G mean for each metric, including T60's two denominators.

| metric | Δ (change in the error vs ground truth) | G (shift from the 0° prediction) |
|---|---|---|
| C50 | change in the C50 error against the ground truth, dB | C50 distance from the k = 0 prediction, dB |
| EDT | change in the EDT error against the ground truth, seconds | EDT distance from the k = 0 prediction, seconds |
| T60 | change in the GT-normalised error, percentage points | percent of the baseline prediction's T60 |
| T60_abs | change in \|T60(prediction) - T60(GT)\|, seconds | \|T60(P_alpha) - T60(P_0)\|, seconds |

## released_k8

`checkpoints/xRIR_unseen.pth` (sha256 `6cdb02767b4c…`), backbone simple, K = 8, device cuda, 6337 queries (batches `all`), manifest `47637a55ccc5…`, gl_seed 0, execution `20260927T073313236199Z-45c485084fee44f6b795b4f2f5a71ff4`.

### EDT (ms)

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 55.1 | 54.6 | -0.585 [-1.77, 0.59] | -0.585 [-2.73, 1.7] | 26.2 [25, 27.5] | 26.2 [10.6, 43.7] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |
| 90° | 55.1 | 53.5 | -1.59 [-2.71, -0.472] | -1.59 [-3.3, 0.37] | 24.9 [23.7, 26.1] | 24.9 [8.79, 42.9] | 6337 | improvement / denominator uncertain | predictions shift while net error improves |
| 180° | 55.1 | 53.9 | -1.22 [-2.37, -0.0636] | -1.22 [-3.51, 0.529] | 25.6 [24.3, 26.9] | 25.6 [9.5, 43.5] | 6337 | improvement / denominator uncertain | predictions shift while net error improves |
| 270° | 55.1 | 54.2 | -0.937 [-1.97, 0.089] | -0.937 [-1.83, 0.0417] | 23.8 [22.6, 24.9] | 23.8 [9.01, 39.9] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |

### C50 (dB)

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 1.36 | 1.39 | 0.0331 [0.00569, 0.0603] | 0.0331 [-0.00927, 0.0748] | 0.644 [0.618, 0.672] | 0.644 [0.283, 0.933] | 6337 | defined / denominator uncertain | **19.5×** [10.3, 85] |
| 90° | 1.36 | 1.36 | -0.00277 [-0.0304, 0.0252] | -0.00277 [-0.03, 0.0194] | 0.606 [0.579, 0.635] | 0.606 [0.247, 0.923] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |
| 180° | 1.36 | 1.37 | 0.00845 [-0.0192, 0.0351] | 0.00845 [-0.0534, 0.0713] | 0.643 [0.614, 0.67] | 0.643 [0.264, 0.954] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |
| 270° | 1.36 | 1.38 | 0.0217 [-0.00338, 0.0469] | 0.0217 [-0.0345, 0.0833] | 0.586 [0.56, 0.613] | 0.586 [0.253, 0.866] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |

### T60 (pp (Δ) / % (G))

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 9.7 | 10.2 | 0.473 [0.324, 0.62] | 0.473 [0.0783, 0.825] | 3.78 [3.66, 3.91] | 3.78 [2.33, 5] | 6337 | defined / defined | **8×** [6.13, 11.6] |
| 90° | 9.7 | 9.96 | 0.259 [0.131, 0.383] | 0.259 [-0.0909, 0.652] | 3.19 [3.08, 3.3] | 3.19 [1.98, 3.95] | 6337 | defined / denominator uncertain | **12.3×** [8.31, 24.3] |
| 180° | 9.7 | 9.47 | -0.236 [-0.364, -0.108] | -0.236 [-0.61, 0.063] | 3.29 [3.18, 3.39] | 3.29 [1.99, 4.1] | 6337 | improvement / denominator uncertain | predictions shift while net error improves |
| 270° | 9.7 | 9.87 | 0.164 [0.0406, 0.292] | 0.164 [-0.084, 0.375] | 3.15 [3.04, 3.26] | 3.15 [1.97, 3.89] | 6337 | defined / denominator uncertain | **19.2×** [10.6, 71.6] |

### T60 absolute (ms)

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 49.3 | 51.5 | 2.24 [1.46, 2.99] | 2.24 [0.149, 4.13] | 21.7 [21, 22.4] | 21.7 [12.1, 28.9] | 6337 | defined / defined | **9.7×** [7.26, 14.8] |
| 90° | 49.3 | 50.4 | 1.11 [0.458, 1.75] | 1.11 [-0.444, 2.69] | 18.2 [17.5, 18.8] | 18.2 [9.95, 23.3] | 6337 | defined / denominator uncertain | **16.4×** [10.4, 39.5] |
| 180° | 49.3 | 48.1 | -1.23 [-1.91, -0.563] | -1.23 [-2.99, 0.286] | 19 [18.4, 19.7] | 19 [10.3, 24.3] | 6337 | improvement / denominator uncertain | predictions shift while net error improves |
| 270° | 49.3 | 49.8 | 0.465 [-0.178, 1.11] | 0.465 [-0.685, 1.58] | 18 [17.4, 18.6] | 18 [10.3, 22.8] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |

### log-spec MAD (GL-free) (dimensionless)

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | – | – | – | – | 0.18 [0.177, 0.183] | 0.18 [0.128, 0.228] | 6337 | undefined / undefined | no paired error for this metric |
| 90° | – | – | – | – | 0.161 [0.158, 0.165] | 0.161 [0.105, 0.215] | 6337 | undefined / undefined | no paired error for this metric |
| 180° | – | – | – | – | 0.17 [0.167, 0.174] | 0.17 [0.107, 0.223] | 6337 | undefined / undefined | no paired error for this metric |
| 270° | – | – | – | – | 0.157 [0.153, 0.16] | 0.157 [0.103, 0.205] | 6337 | undefined / undefined | no paired error for this metric |

Probe run: `ckpt/exp10/released_k8_probe` (272 queries, execution `20260927T072809041814Z-bbd9dec45b8547f1a6fbd9a171c76e38`, sha256 of its summary `44c871b4bdeb…`).

**Controls:** ok = True; ctrl_full_turn (k=512) max|Δwave| 0, max|Δlogspec| 0; ctrl_k128_repeat (k=128) max|Δwave| 0, max|Δlogspec| 0; ctrl_zero_repeat (k=0) max|Δwave| 0, max|Δlogspec| 0

**Parity vs exp_03** (`ckpt/exp10/released_k8_all/parity_exp03.json`, sha256 `8e1b613196af…`): ok = True, rows 6337, missing required []; largest per-query |Δ|: c50_err 0, decay 0, edt_err 0, log_mse 0, logspec_mad 1.53e-07, loss 0, stft 0, t60_err 0.

**check_online** (`ckpt/exp10/released_k8_all/check_online.json`): ok = True.

## released_k1 (trained K = 8, evaluated K = 1)

`checkpoints/xRIR_unseen.pth` (sha256 `6cdb02767b4c…`), backbone simple, K = 1, device cuda, 6337 queries (batches `all`), manifest `f6d71f86d5d3…`, gl_seed 0, execution `20260927T062839218781Z-93fc61b3802c4e01bb2aa2d55e76c591`.

### EDT (ms)

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 84 | 83.9 | -0.124 [-0.677, 0.409] | -0.124 [-1.17, 0.851] | 12.5 [12, 13] | 12.5 [6.03, 18.9] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |
| 90° | 84 | 83.2 | -0.786 [-1.34, -0.244] | -0.786 [-2.55, 0.614] | 11.8 [11.3, 12.3] | 11.8 [4.87, 19.2] | 6337 | improvement / denominator uncertain | predictions shift while net error improves |
| 180° | 84 | 82.9 | -1.1 [-1.57, -0.637] | -1.1 [-1.76, -0.244] | 10.6 [10.1, 11] | 10.6 [5.36, 15.4] | 6337 | improvement / improvement | predictions shift while net error improves |
| 270° | 84 | 83.8 | -0.217 [-0.782, 0.338] | -0.217 [-1.43, 0.73] | 11.9 [11.4, 12.4] | 11.9 [4.99, 19.2] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |

### C50 (dB)

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 2.08 | 2.09 | 0.0092 [0.000433, 0.0178] | 0.0092 [-0.00215, 0.0188] | 0.228 [0.221, 0.235] | 0.228 [0.126, 0.311] | 6337 | defined / denominator uncertain | **24.7×** [11.2, 136] |
| 90° | 2.08 | 2.12 | 0.0315 [0.0225, 0.0403] | 0.0315 [-0.00518, 0.0758] | 0.227 [0.22, 0.234] | 0.227 [0.118, 0.318] | 6337 | defined / denominator uncertain | **7.21×** [5.65, 10] |
| 180° | 2.08 | 2.13 | 0.0471 [0.0374, 0.0568] | 0.0471 [-0.00169, 0.116] | 0.245 [0.237, 0.253] | 0.245 [0.133, 0.345] | 6337 | defined / denominator uncertain | **5.2×** [4.36, 6.48] |
| 270° | 2.08 | 2.11 | 0.0221 [0.0144, 0.0299] | 0.0221 [-0.0147, 0.0671] | 0.213 [0.207, 0.219] | 0.213 [0.115, 0.293] | 6337 | defined / denominator uncertain | **9.6×** [7.12, 14.8] |

### T60 (pp (Δ) / % (G))

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 13.8 | 13.9 | 0.0551 [-0.00341, 0.116] | 0.0551 [-0.0403, 0.132] | 1.46 [1.42, 1.51] | 1.46 [1.06, 1.75] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |
| 90° | 13.8 | 13.8 | 0.0245 [-0.0298, 0.08] | 0.0245 [-0.0532, 0.111] | 1.32 [1.28, 1.36] | 1.32 [0.868, 1.7] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |
| 180° | 13.8 | 13.8 | 0.0522 [-0.00541, 0.11] | 0.0522 [-0.0626, 0.197] | 1.44 [1.39, 1.49] | 1.44 [0.916, 1.98] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |
| 270° | 13.8 | 13.8 | 0.0311 [-0.0226, 0.0862] | 0.0311 [-0.0224, 0.087] | 1.31 [1.27, 1.35] | 1.31 [0.896, 1.64] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |

### T60 absolute (ms)

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 67.8 | 68.1 | 0.331 [0.0154, 0.651] | 0.331 [-0.22, 0.776] | 8.39 [8.13, 8.66] | 8.39 [5.51, 10.2] | 6337 | defined / denominator uncertain | **25.3×** [11.4, 149] |
| 90° | 67.8 | 67.9 | 0.117 [-0.173, 0.414] | 0.117 [-0.25, 0.567] | 7.54 [7.28, 7.8] | 7.54 [4.23, 9.87] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |
| 180° | 67.8 | 68.1 | 0.291 [-0.0254, 0.607] | 0.291 [-0.309, 1.08] | 8.13 [7.84, 8.42] | 8.13 [4.59, 11.1] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |
| 270° | 67.8 | 67.9 | 0.117 [-0.172, 0.412] | 0.117 [-0.0954, 0.311] | 7.39 [7.15, 7.64] | 7.39 [4.39, 9.49] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |

### log-spec MAD (GL-free) (dimensionless)

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | – | – | – | – | 0.109 [0.107, 0.111] | 0.109 [0.075, 0.135] | 6337 | undefined / undefined | no paired error for this metric |
| 90° | – | – | – | – | 0.0917 [0.0901, 0.0934] | 0.0917 [0.0608, 0.118] | 6337 | undefined / undefined | no paired error for this metric |
| 180° | – | – | – | – | 0.0959 [0.0938, 0.0981] | 0.0959 [0.0609, 0.128] | 6337 | undefined / undefined | no paired error for this metric |
| 270° | – | – | – | – | 0.0914 [0.0898, 0.0931] | 0.0914 [0.0603, 0.118] | 6337 | undefined / undefined | no paired error for this metric |

Probe run: `ckpt/exp10/released_k1_probe` (272 queries, execution `20260927T062510342040Z-080109365da74a318a0424ba1f6fed8d`, sha256 of its summary `dcd366910c11…`).

**Controls:** ok = True; ctrl_full_turn (k=512) max|Δwave| 0, max|Δlogspec| 0; ctrl_k128_repeat (k=128) max|Δwave| 0, max|Δlogspec| 0; ctrl_zero_repeat (k=0) max|Δwave| 0, max|Δlogspec| 0

**check_online** (`ckpt/exp10/released_k1_all/check_online.json`): ok = True.

## control_k8

`ckpt/xRIR_simple_8_shot/epoch_12.pth` (sha256 `651e3a377e11…`), backbone simple, K = 8, device cuda, 6337 queries (batches `all`), manifest `47637a55ccc5…`, gl_seed 0, execution `20260927T025953876010Z-4b35102e25014d87875828367b73a0ff`.

### EDT (ms)

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 47 | 47.6 | 0.63 [-0.117, 1.36] | 0.63 [-0.129, 1.49] | 16.4 [15.7, 17.1] | 16.4 [5.51, 27.5] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |
| 90° | 47 | 47.6 | 0.588 [0.0527, 1.1] | 0.588 [-0.277, 1.28] | 10.5 [9.95, 11] | 10.5 [4.42, 16.5] | 6337 | defined / denominator uncertain | **17.8×** [8.81, 95.3] |
| 180° | 47 | 47 | -0.0271 [-0.555, 0.501] | -0.0271 [-0.866, 0.521] | 10.1 [9.61, 10.6] | 10.1 [4.14, 15.5] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |
| 270° | 47 | 47.3 | 0.317 [-0.208, 0.844] | 0.317 [-0.611, 1.1] | 10.5 [10, 11] | 10.5 [4.57, 16.4] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |

### C50 (dB)

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 1.24 | 1.3 | 0.0629 [0.0434, 0.0816] | 0.0629 [0.00927, 0.13] | 0.442 [0.423, 0.462] | 0.442 [0.154, 0.683] | 6337 | defined / defined | **7.03×** [5.43, 10.2] |
| 90° | 1.24 | 1.26 | 0.024 [0.0104, 0.0374] | 0.024 [-0.014, 0.0679] | 0.3 [0.286, 0.314] | 0.3 [0.12, 0.452] | 6337 | defined / denominator uncertain | **12.5×** [8.01, 28.8] |
| 180° | 1.24 | 1.23 | -0.00334 [-0.0188, 0.0116] | -0.00334 [-0.0384, 0.0309] | 0.308 [0.293, 0.325] | 0.308 [0.114, 0.479] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |
| 270° | 1.24 | 1.25 | 0.0116 [-0.00284, 0.0262] | 0.0116 [-0.0164, 0.0462] | 0.302 [0.288, 0.317] | 0.302 [0.124, 0.452] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |

### T60 (pp (Δ) / % (G))

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 9.56 | 9.59 | 0.0278 [-0.0766, 0.132] | 0.0278 [-0.23, 0.291] | 2.48 [2.39, 2.57] | 2.48 [1.41, 3.19] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |
| 90° | 9.56 | 9.44 | -0.12 [-0.201, -0.0395] | -0.12 [-0.303, 0.032] | 1.67 [1.6, 1.74] | 1.67 [1.02, 2.21] | 6337 | improvement / denominator uncertain | predictions shift while net error improves |
| 180° | 9.56 | 9.22 | -0.341 [-0.421, -0.263] | -0.341 [-0.736, -0.0633] | 1.54 [1.47, 1.6] | 1.54 [0.871, 2.11] | 6337 | improvement / improvement | predictions shift while net error improves |
| 270° | 9.56 | 9.39 | -0.171 [-0.246, -0.0976] | -0.171 [-0.375, -0.00568] | 1.66 [1.59, 1.72] | 1.66 [1.05, 2.11] | 6337 | improvement / improvement | predictions shift while net error improves |

### T60 absolute (ms)

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 48.7 | 48.8 | 0.0658 [-0.503, 0.629] | 0.0658 [-1.37, 1.48] | 14.6 [14.1, 15.2] | 14.6 [7.4, 19.7] | 6337 | undefined / denominator uncertain | unresolved Monte Carlo uncertainty |
| 90° | 48.7 | 48.1 | -0.579 [-0.991, -0.175] | -0.579 [-1.52, 0.242] | 9.64 [9.26, 10] | 9.64 [5.41, 12.9] | 6337 | improvement / denominator uncertain | predictions shift while net error improves |
| 180° | 48.7 | 46.9 | -1.76 [-2.18, -1.35] | -1.76 [-3.86, -0.315] | 8.92 [8.54, 9.31] | 8.92 [4.7, 12.3] | 6337 | improvement / improvement | predictions shift while net error improves |
| 270° | 48.7 | 47.8 | -0.907 [-1.31, -0.497] | -0.907 [-2.01, -0.0285] | 9.59 [9.23, 9.95] | 9.59 [5.62, 12.4] | 6337 | improvement / improvement | predictions shift while net error improves |

### log-spec MAD (GL-free) (dimensionless)

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | – | – | – | – | 0.127 [0.124, 0.131] | 0.127 [0.0611, 0.189] | 6337 | undefined / undefined | no paired error for this metric |
| 90° | – | – | – | – | 0.0769 [0.0751, 0.0788] | 0.0769 [0.0446, 0.106] | 6337 | undefined / undefined | no paired error for this metric |
| 180° | – | – | – | – | 0.0712 [0.0694, 0.073] | 0.0712 [0.0388, 0.0986] | 6337 | undefined / undefined | no paired error for this metric |
| 270° | – | – | – | – | 0.0768 [0.0749, 0.0787] | 0.0768 [0.045, 0.106] | 6337 | undefined / undefined | no paired error for this metric |

Probe run: `ckpt/exp10/control_k8_probe` (272 queries, execution `20260927T023813895915Z-6b0a883b52604bb090477542219691c4`, sha256 of its summary `0341ff9a206e…`).

**Controls:** ok = True; ctrl_full_turn (k=512) max|Δwave| 0, max|Δlogspec| 0; ctrl_k128_repeat (k=128) max|Δwave| 0, max|Δlogspec| 0; ctrl_zero_repeat (k=0) max|Δwave| 0, max|Δlogspec| 0

**Parity vs exp_03** (`ckpt/exp10/control_k8_all/parity_exp03.json`, sha256 `1b5152104f41…`): ok = True, rows 6337, missing required []; largest per-query |Δ|: c50_err 0, decay 0, edt_err 0, log_mse 0, logspec_mad 1.42e-07, loss 0, stft 0, t60_err 0.

**check_online** (`ckpt/exp10/control_k8_all/check_online.json`): ok = True.

## cyl_k8

`ckpt/xRIR_cyl_8_shot/epoch_12.pth` (sha256 `8ba344ad25d4…`), backbone cylindrical, K = 8, device cuda, 6337 queries (batches `all`), manifest `47637a55ccc5…`, gl_seed 0, execution `20260927T051017740231Z-7d2e7e6db2994f4fa57533077c4c80a0`.

### EDT (ms)

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 44.9 | 46.1 | 1.15 [0.518, 1.78] | 1.15 [0.564, 1.55] | 14.3 [13.6, 14.9] | 14.3 [4.74, 23.6] | 6337 | defined / defined | **12.4×** [8.01, 27.4] |
| 90° | 44.9 | 45.9 | 0.919 [0.488, 1.35] | 0.919 [0.0283, 1.85] | 8.69 [8.26, 9.12] | 8.69 [2.77, 14.7] | 6337 | defined / defined | **9.46×** [6.47, 17.7] |
| 180° | 44.9 | 45.4 | 0.422 [0.0114, 0.839] | 0.422 [-0.00873, 0.818] | 7.81 [7.43, 8.21] | 7.81 [2.46, 13.2] | 6337 | defined / denominator uncertain | **18.5×** [7.77, 112] |
| 270° | 44.9 | 45.8 | 0.888 [0.471, 1.3] | 0.888 [-0.0994, 1.94] | 8.52 [8.12, 8.94] | 8.52 [2.82, 14.2] | 6337 | defined / denominator uncertain | **9.6×** [6.56, 18.2] |

### C50 (dB)

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 1.23 | 1.29 | 0.0636 [0.0474, 0.0797] | 0.0636 [0.012, 0.12] | 0.381 [0.366, 0.397] | 0.381 [0.12, 0.597] | 6337 | defined / defined | **5.99×** [4.8, 7.96] |
| 90° | 1.23 | 1.26 | 0.032 [0.0206, 0.0437] | 0.032 [-0.00368, 0.0862] | 0.251 [0.24, 0.263] | 0.251 [0.0717, 0.406] | 6337 | defined / denominator uncertain | **7.85×** [5.76, 12.2] |
| 180° | 1.23 | 1.23 | 0.00337 [-0.00871, 0.0157] | 0.00337 [-0.0156, 0.026] | 0.235 [0.222, 0.248] | 0.235 [0.0631, 0.391] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |
| 270° | 1.23 | 1.25 | 0.0224 [0.0115, 0.0335] | 0.0224 [-0.0064, 0.0595] | 0.234 [0.223, 0.244] | 0.234 [0.0726, 0.37] | 6337 | defined / denominator uncertain | **10.4×** [7.02, 20.3] |

### T60 (pp (Δ) / % (G))

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 9.44 | 9.59 | 0.157 [0.0555, 0.258] | 0.157 [-0.0906, 0.483] | 2.32 [2.23, 2.4] | 2.32 [1.33, 2.93] | 6337 | defined / denominator uncertain | **14.8×** [9, 41.8] |
| 90° | 9.44 | 9.39 | -0.0416 [-0.107, 0.0241] | -0.0416 [-0.139, 0.0417] | 1.45 [1.4, 1.5] | 1.45 [0.789, 1.97] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |
| 180° | 9.44 | 9.13 | -0.305 [-0.369, -0.241] | -0.305 [-0.688, -0.0158] | 1.23 [1.18, 1.28] | 1.23 [0.604, 1.79] | 6337 | improvement / improvement | predictions shift while net error improves |
| 270° | 9.44 | 9.36 | -0.0715 [-0.135, -0.00775] | -0.0715 [-0.284, 0.113] | 1.42 [1.37, 1.47] | 1.42 [0.795, 1.88] | 6337 | improvement / denominator uncertain | unresolved Monte Carlo uncertainty |

### T60 absolute (ms)

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | 47.8 | 48.7 | 0.822 [0.277, 1.36] | 0.822 [-0.555, 2.63] | 13.7 [13.2, 14.2] | 13.7 [7.14, 17.9] | 6337 | defined / denominator uncertain | **16.6×** [10.1, 49] |
| 90° | 47.8 | 47.6 | -0.259 [-0.615, 0.0962] | -0.259 [-0.726, 0.13] | 8.5 [8.19, 8.82] | 8.5 [4.03, 11.7] | 6337 | denominator uncertain / denominator uncertain | denominator uncertain |
| 180° | 47.8 | 46.2 | -1.62 [-1.96, -1.28] | -1.62 [-3.56, -0.0707] | 7.31 [6.99, 7.64] | 7.31 [3.09, 10.8] | 6337 | improvement / improvement | predictions shift while net error improves |
| 270° | 47.8 | 47.4 | -0.465 [-0.812, -0.121] | -0.465 [-1.55, 0.403] | 8.33 [8.02, 8.64] | 8.33 [4.18, 11.1] | 6337 | improvement / denominator uncertain | predictions shift while net error improves |

### log-spec MAD (GL-free) (dimensionless)

| angle | mean 0° | mean α | Δ query CI | Δ room CI | G query CI | G room CI | n | status query / room | multiple or reason |
|---|---|---|---|---|---|---|---|---|---|
| 45° | – | – | – | – | 0.114 [0.111, 0.117] | 0.114 [0.0557, 0.168] | 6337 | undefined / undefined | no paired error for this metric |
| 90° | – | – | – | – | 0.0668 [0.065, 0.0686] | 0.0668 [0.0322, 0.0981] | 6337 | undefined / undefined | no paired error for this metric |
| 180° | – | – | – | – | 0.0605 [0.0589, 0.0621] | 0.0605 [0.0278, 0.0891] | 6337 | undefined / undefined | no paired error for this metric |
| 270° | – | – | – | – | 0.0646 [0.0629, 0.0663] | 0.0646 [0.0323, 0.0938] | 6337 | undefined / undefined | no paired error for this metric |

Probe run: `ckpt/exp10/cyl_k8_probe` (272 queries, execution `20260927T050438791078Z-ab38c2dced99479889a63425c95c9cd5`, sha256 of its summary `f77d3c884366…`).

**Controls:** ok = True; ctrl_full_turn (k=512) max|Δwave| 0, max|Δlogspec| 0; ctrl_k128_repeat (k=128) max|Δwave| 0, max|Δlogspec| 0; ctrl_zero_repeat (k=0) max|Δwave| 0, max|Δlogspec| 0

**Parity vs exp_03** (`ckpt/exp10/cyl_k8_all/parity_exp03.json`, sha256 `91b98e1dd9f3…`): ok = True, rows 6337, missing required []; largest per-query |Δ|: c50_err 0, decay 0, edt_err 0, log_mse 0, logspec_mad 1.15e-07, loss 0, stft 0, t60_err 0.

**check_online** (`ckpt/exp10/cyl_k8_all/check_online.json`): ok = True.

## Supplementary results and data

- [full tables (every angle, every metric, exclusions, broader-population G)](yaw_pilot_results_assets/generated/yaw_pilot_tables.md)
- [canonical summary JSON (the source of every number here)](yaw_pilot_results_assets/generated/yaw_pilot_summary.json)
- [figure data (CSV)](yaw_pilot_results_assets/generated/yaw_pilot_gaps.csv)
- [backend sensitivity: GPU (primary) vs CPU protocol](yaw_pilot_results_assets/generated/backend_sensitivity_released_k8.md)
- [CPU-protocol record](yaw_pilot_results_assets/generated/cpu_protocol/yaw_pilot_tables.md)

## Provenance

Every input of this report, with its full sha256 (the HTML page's footer lists the same values):

- canonical summary: `ckpt/exp10/summary/yaw_pilot_summary.json` sha256 `826498f58780dfe42a4e6228075b0fcb2be845a18a0986d0e835302faee85b32`
- control_k8 probe summary: `ckpt/exp10/control_k8_probe/summary/yaw_pilot_summary.json` sha256 `0341ff9a206e2c3feb1d8bdf096e3f2e0af8722988f6b2086f478d13992ee06c`
- cyl_k8 probe summary: `ckpt/exp10/cyl_k8_probe/summary/yaw_pilot_summary.json` sha256 `f77d3c884366514cf1646ef3dcafc81e6cba6072b0e7217b5a3e46fba8b29858`
- released_k1 probe summary: `ckpt/exp10/released_k1_probe/summary/yaw_pilot_summary.json` sha256 `dcd366910c112dcb1bd064323829282ac550261a38860b1815745500338e9071`
- released_k8 probe summary: `ckpt/exp10/released_k8_probe/summary/yaw_pilot_summary.json` sha256 `44c871b4bdeb207135ad362c6d86e68daa804102e936919bfb670f80daf28c61`
- control_k8 parity_exp03: `ckpt/exp10/control_k8_all/parity_exp03.json` sha256 `1b5152104f41615ad97a9c2e4e5b013f3129480205106daab111d2ac8e0f22fd`
- cyl_k8 parity_exp03: `ckpt/exp10/cyl_k8_all/parity_exp03.json` sha256 `91b98e1dd9f3eea434afcbad63e491745befaa21005b452b6ba1110e45cf835b`
- released_k8 parity_exp03: `ckpt/exp10/released_k8_all/parity_exp03.json` sha256 `8e1b613196afa33a196e9a09393fc3fadfddb08bd00717a7954fca6883ba9276`
- control_k8 check_online: `ckpt/exp10/control_k8_all/check_online.json` sha256 `6bff799664312f0b85bb42119b68bb42def9e0402e281748c9807232bbb5c9cb`
- cyl_k8 check_online: `ckpt/exp10/cyl_k8_all/check_online.json` sha256 `5bd4ebcbb9575fa84f741c07959feb4d42eae324a9f67308914b3e9c8986fa7f`
- released_k1 check_online: `ckpt/exp10/released_k1_all/check_online.json` sha256 `a3fe176ff0dec96d22b9a1b3f174f643bc3c65243a399805fea916df0a80af32`
- released_k8 check_online: `ckpt/exp10/released_k8_all/check_online.json` sha256 `dcf17eecb2029bacc1f3c54bbeb029e89d630f287a546c008deca7b5ec167dfd`
- backend sensitivity table: `yaw_pilot_results_assets/generated/backend_sensitivity_released_k8.md` sha256 `94d806b138fa33f63d4f6a34d45ecce218b10ee5ffee0a625a3344d2055afdee`
- CPU-protocol record: `yaw_pilot_results_assets/generated/cpu_protocol/yaw_pilot_tables.md` sha256 `541d890fe33afd88433f423c3b6462563770e97f6e8e6aa03cae3a37f6fe890c`

Runs the canonical summary was computed from:

- released_k8: execution `20260927T073313236199Z-45c485084fee44f6b795b4f2f5a71ff4`, per_sample sha256 `e8332e702ce4ed922d1a69cd5e2776e239c7ccf128609be357f05ae77a180500`, meta sha256 `e9dcdb36ed608273ea4bab67bb9a00a541ab06f5931f217141f3ef2396fb35d7`, run dir `/home/yixunhu/codespace/xRIR_code/ckpt/exp10/released_k8_all`
- released_k1: execution `20260927T062839218781Z-93fc61b3802c4e01bb2aa2d55e76c591`, per_sample sha256 `098838cf5cb39ba775d8772d12540370cc24083680af58a26f99e5dff09d8511`, meta sha256 `1331242a8cf5125da6d51d9a631b38b01c419372697aeccdf77018d0b20492bd`, run dir `/home/yixunhu/codespace/xRIR_code/ckpt/exp10/released_k1_all`
- control_k8: execution `20260927T025953876010Z-4b35102e25014d87875828367b73a0ff`, per_sample sha256 `c5c4b2089a1748bd603b63529dc3f58c2c9c5cbf800ba80926c12c37dc759109`, meta sha256 `b0813cc93f3fce3313b8fd48462de71d67e53408f1d591c1bfeeb1ca033978aa`, run dir `/home/yixunhu/codespace/xRIR_code/ckpt/exp10/control_k8_all`
- cyl_k8: execution `20260927T051017740231Z-7d2e7e6db2994f4fa57533077c4c80a0`, per_sample sha256 `b34ced778523ba048a3beefaecebd2a6f60e3149366bebf9311c39fa865c133a`, meta sha256 `3cb247bfdc88b3d3b3094d637538c231a2254693995c1f40af2a7236af21afb1`, run dir `/home/yixunhu/codespace/xRIR_code/ckpt/exp10/cyl_k8_all`
