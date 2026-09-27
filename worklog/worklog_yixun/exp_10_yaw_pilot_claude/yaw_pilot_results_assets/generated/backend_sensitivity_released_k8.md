# Backend sensitivity — released_k8: GPU (primary) vs CPU protocol

Same checkpoint, query population (`query_list_sha256` `a7e7b2e6131a871376a63591359da7df5531dfdeaac01ccde5b6c8f72fc2b4fe`), references (manifest `47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d`), Griffin-Lim seed, K, batch shape, precision flags, evaluator implementation (`tool_sha256` `e727cba016dcc2d78c7ba193fca85b75116c40f9e9d567be784970a2974c8823`) and torch / numpy versions — every one of them verified equal; only the inference device differs (GPU run: cuda, execution `20260927T073313236199Z-45c485084fee44f6b795b4f2f5a71ff4`, n = 6337; CPU run: cpu, execution `20260927T004009003970Z-8796b160e3a84e97b43774e2d4aac00e`, n = 6337). Both columns come from one statistical protocol, also verified equal: alpha 0.05, 10000 bootstrap replicates, seeds [0, 1], convergence tolerance 0.1. Query-level bootstrap intervals; descriptive only.

| metric | angle | Δ GPU | Δ CPU | G GPU | G CPU | status GPU / CPU |
|---|---|---|---|---|---|---|
| EDT (ms) | 45° | -0.585 [-1.77, 0.59] | -0.582 [-1.76, 0.594] | 26.2 [25, 27.5] | 26.2 [25, 27.5] | denominator uncertain / denominator uncertain |
| EDT (ms) | 90° | -1.59 [-2.71, -0.472] | -1.59 [-2.71, -0.471] | 24.9 [23.7, 26.1] | 24.9 [23.7, 26.1] | improvement / improvement |
| EDT (ms) | 180° | -1.22 [-2.37, -0.0636] | -1.22 [-2.37, -0.0582] | 25.6 [24.3, 26.9] | 25.6 [24.3, 26.9] | improvement / improvement |
| EDT (ms) | 270° | -0.937 [-1.97, 0.089] | -0.937 [-1.97, 0.0894] | 23.8 [22.6, 24.9] | 23.8 [22.6, 24.9] | denominator uncertain / denominator uncertain |
| C50 (dB) | 45° | 0.0331 [0.00569, 0.0603] | 0.0331 [0.00569, 0.0603] | 0.644 [0.618, 0.672] | 0.644 [0.618, 0.671] | defined / defined |
| C50 (dB) | 90° | -0.00277 [-0.0304, 0.0252] | -0.00276 [-0.0304, 0.0252] | 0.606 [0.579, 0.635] | 0.606 [0.579, 0.635] | denominator uncertain / denominator uncertain |
| C50 (dB) | 180° | 0.00845 [-0.0192, 0.0351] | 0.00846 [-0.0192, 0.0351] | 0.643 [0.614, 0.67] | 0.643 [0.614, 0.67] | denominator uncertain / denominator uncertain |
| C50 (dB) | 270° | 0.0217 [-0.00338, 0.0469] | 0.0217 [-0.00338, 0.0469] | 0.586 [0.56, 0.613] | 0.586 [0.56, 0.613] | denominator uncertain / denominator uncertain |
| T60 (pp / %) | 45° | 0.473 [0.324, 0.62] | 0.473 [0.324, 0.62] | 3.78 [3.66, 3.91] | 3.78 [3.66, 3.91] | defined / defined |
| T60 (pp / %) | 90° | 0.259 [0.131, 0.383] | 0.259 [0.131, 0.383] | 3.19 [3.08, 3.3] | 3.19 [3.08, 3.3] | defined / defined |
| T60 (pp / %) | 180° | -0.236 [-0.364, -0.108] | -0.235 [-0.364, -0.108] | 3.29 [3.18, 3.39] | 3.29 [3.18, 3.39] | improvement / improvement |
| T60 (pp / %) | 270° | 0.164 [0.0406, 0.292] | 0.164 [0.0406, 0.292] | 3.15 [3.04, 3.26] | 3.15 [3.04, 3.26] | defined / defined |
| T60_abs (ms) | 45° | 2.24 [1.46, 2.99] | 2.24 [1.46, 2.99] | 21.7 [21, 22.4] | 21.7 [21, 22.4] | defined / defined |
| T60_abs (ms) | 90° | 1.11 [0.458, 1.75] | 1.11 [0.458, 1.75] | 18.2 [17.5, 18.8] | 18.2 [17.5, 18.8] | defined / defined |
| T60_abs (ms) | 180° | -1.23 [-1.91, -0.563] | -1.23 [-1.91, -0.563] | 19 [18.4, 19.7] | 19 [18.4, 19.7] | improvement / improvement |
| T60_abs (ms) | 270° | 0.465 [-0.178, 1.11] | 0.465 [-0.178, 1.11] | 18 [17.4, 18.6] | 18 [17.4, 18.6] | denominator uncertain / denominator uncertain |
| logspec_mad | 45° | – | – | 0.18 [0.177, 0.183] | 0.18 [0.177, 0.183] | undefined / undefined |
| logspec_mad | 90° | – | – | 0.161 [0.158, 0.165] | 0.161 [0.158, 0.165] | undefined / undefined |
| logspec_mad | 180° | – | – | 0.17 [0.167, 0.174] | 0.17 [0.167, 0.174] | undefined / undefined |
| logspec_mad | 270° | – | – | 0.157 [0.153, 0.16] | 0.157 [0.153, 0.16] | undefined / undefined |
| mag_rel_l2 | 45° | – | – | 0.234 [0.229, 0.239] | 0.234 [0.229, 0.239] | undefined / undefined |
| mag_rel_l2 | 90° | – | – | 0.208 [0.202, 0.213] | 0.208 [0.202, 0.213] | undefined / undefined |
| mag_rel_l2 | 180° | – | – | 0.22 [0.214, 0.226] | 0.22 [0.214, 0.226] | undefined / undefined |
| mag_rel_l2 | 270° | – | – | 0.201 [0.196, 0.206] | 0.201 [0.196, 0.206] | undefined / undefined |
| wave_rel_l2 | 45° | – | – | 0.877 [0.871, 0.884] | 0.877 [0.871, 0.884] | undefined / undefined |
| wave_rel_l2 | 90° | – | – | 0.824 [0.817, 0.831] | 0.824 [0.817, 0.831] | undefined / undefined |
| wave_rel_l2 | 180° | – | – | 0.834 [0.827, 0.842] | 0.834 [0.827, 0.842] | undefined / undefined |
| wave_rel_l2 | 270° | – | – | 0.816 [0.809, 0.823] | 0.816 [0.809, 0.823] | undefined / undefined |
| wave_mad | 45° | – | – | 0.000933 [0.000913, 0.000953] | 0.000933 [0.000913, 0.000953] | undefined / undefined |
| wave_mad | 90° | – | – | 0.000853 [0.000835, 0.000871] | 0.000853 [0.000835, 0.000871] | undefined / undefined |
| wave_mad | 180° | – | – | 0.000862 [0.000844, 0.000881] | 0.000862 [0.000844, 0.000881] | undefined / undefined |
| wave_mad | 270° | – | – | 0.000849 [0.000831, 0.000867] | 0.000849 [0.000831, 0.000867] | undefined / undefined |

Inputs: `ckpt/exp10/summary/yaw_pilot_summary.json` sha256 `8c11a043c34db4f55309f10a4011dad2af2e6ed788569c2e87f1ee53f0aa2353`; `ckpt/exp10/cpu_protocol/summary/yaw_pilot_summary.json` sha256 `e77bf22f11d730ea9ee6ba16ccdc01561a05a4548b9a45065922dee91e327c9e`; live metas `/home/yixunhu/codespace/xRIR_code/ckpt/exp10/released_k8_all/meta.json` sha256 `e9dcdb36ed608273ea4bab67bb9a00a541ab06f5931f217141f3ef2396fb35d7` and `/home/yixunhu/codespace/xRIR_code/ckpt/exp10/cpu_protocol/released_k8_all/meta.json` sha256 `4697d7377fde8202bae05af248a34a99da6bfca246fcbe4990b856f25db41733`.
