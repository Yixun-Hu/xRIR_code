# exp_10 `yaw_pilot` -- descriptive summary

10000 bootstrap resamples, 95 % percentile intervals, seeds [0, 1] (seed 0 reports, seed 1 checks convergence).

The grey band in the figures is the historical baseline evaluation variability (references and phases redrawn), context only.

**How to read the shift.** waveform and acoustic gaps measure the sensitivity of the model-plus-Griffin-Lim pipeline, conditional on the per-query phase seed; the shared phase initialisation removes initialisation noise, but Griffin-Lim is a nonlinear inverse and can amplify a magnitude change. logspec_mad and mag_rel_l2 are the Griffin-Lim-free readout of the same shift, computed on the model's direct log-magnitude output.

**Denominators.** EDT: delta = change in the EDT error against the ground truth, seconds, G = EDT distance from the k = 0 prediction, seconds; C50: delta = change in the C50 error against the ground truth, dB, G = C50 distance from the k = 0 prediction, dB; T60: delta = change in the GT-normalised error, percentage points, G = percent of the baseline prediction's T60; T60_abs: delta = change in |T60(prediction) - T60(GT)|, seconds, G = |T60(P_alpha) - T60(P_0)|, seconds.

## released_k8

`ckpt/exp10/released_k8_all` -- 6337 queries, 17 rooms, execution `20260927T073313236199Z-45c485084fee44f6b795b4f2f5a71ff4`.

Context band: historical baseline evaluation variability (SimpleViT, exp_04), context only -- C50 0.0053, EDT 0.186, T60 0.009.

### Metric 2 -- accuracy against the ground truth

| angle | metric | unit | mean k=0 | mean k | delta | 95 % CI | n_mask | excluded (0 / alpha / gap) |
|---|---|---|---|---|---|---|---|---|
| 0° | EDT | s | 0.0551355 | 0.0551355 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | C50 | dB | 1.3605 | 1.3605 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | T60 | % of T60 | 9.70306 | 9.70306 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | T60_abs | s | 0.0492949 | 0.0492949 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | log_mse | log-magnitude^2 | 0.438771 | 0.438771 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | loss | test loss | 0.0169725 | 0.0169725 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 45° | EDT | s | 0.0551355 | 0.0545506 | -0.000584839 | [-0.00176748, 0.000590129] | 6337 | 0 / 0 / 0 |
| 45° | C50 | dB | 1.3605 | 1.39359 | 0.0330897 | [0.00568722, 0.0602688] | 6337 | 0 / 0 / 0 |
| 45° | T60 | % of T60 | 9.70306 | 10.1759 | 0.472875 | [0.323553, 0.620222] | 6337 | 0 / 0 / 0 |
| 45° | T60_abs | s | 0.0492949 | 0.0515312 | 0.00223626 | [0.00146373, 0.00299072] | 6337 | 0 / 0 / 0 |
| 45° | log_mse | log-magnitude^2 | 0.438771 | 0.436909 | -0.00186232 | [-0.00548035, 0.00183321] | 6337 | 0 / 0 / 0 |
| 45° | loss | test loss | 0.0169725 | 0.0174668 | 0.000494344 | [0.000340865, 0.00064037] | 6337 | 0 / 0 / 0 |
| 90° | EDT | s | 0.0551355 | 0.0535451 | -0.00159036 | [-0.00271272, -0.000472087] | 6337 | 0 / 0 / 0 |
| 90° | C50 | dB | 1.3605 | 1.35773 | -0.00277474 | [-0.0304373, 0.0252082] | 6337 | 0 / 0 / 0 |
| 90° | T60 | % of T60 | 9.70306 | 9.96212 | 0.259065 | [0.131265, 0.382995] | 6337 | 0 / 0 / 0 |
| 90° | T60_abs | s | 0.0492949 | 0.0504027 | 0.0011078 | [0.000458017, 0.00174761] | 6337 | 0 / 0 / 0 |
| 90° | log_mse | log-magnitude^2 | 0.438771 | 0.439623 | 0.000851942 | [-0.00286025, 0.00455387] | 6337 | 0 / 0 / 0 |
| 90° | loss | test loss | 0.0169725 | 0.0170292 | 5.67318e-05 | [-8.2131e-05, 0.000192283] | 6337 | 0 / 0 / 0 |
| 180° | EDT | s | 0.0551355 | 0.0539133 | -0.00122219 | [-0.00236904, -6.36431e-05] | 6337 | 0 / 0 / 0 |
| 180° | C50 | dB | 1.3605 | 1.36895 | 0.00844679 | [-0.0191752, 0.0350625] | 6337 | 0 / 0 / 0 |
| 180° | T60 | % of T60 | 9.70306 | 9.46754 | -0.235516 | [-0.364141, -0.108482] | 6337 | 0 / 0 / 0 |
| 180° | T60_abs | s | 0.0492949 | 0.0480649 | -0.00123007 | [-0.00190878, -0.000562965] | 6337 | 0 / 0 / 0 |
| 180° | log_mse | log-magnitude^2 | 0.438771 | 0.447126 | 0.00835481 | [0.00394752, 0.0128548] | 6337 | 0 / 0 / 0 |
| 180° | loss | test loss | 0.0169725 | 0.0168637 | -0.000108805 | [-0.000246372, 2.7375e-05] | 6337 | 0 / 0 / 0 |
| 270° | EDT | s | 0.0551355 | 0.0541982 | -0.000937288 | [-0.00197193, 8.90183e-05] | 6337 | 0 / 0 / 0 |
| 270° | C50 | dB | 1.3605 | 1.38217 | 0.021674 | [-0.00337805, 0.0468926] | 6337 | 0 / 0 / 0 |
| 270° | T60 | % of T60 | 9.70306 | 9.86731 | 0.164253 | [0.0406409, 0.291998] | 6337 | 0 / 0 / 0 |
| 270° | T60_abs | s | 0.0492949 | 0.0497598 | 0.000464844 | [-0.000178121, 0.00111109] | 6337 | 0 / 0 / 0 |
| 270° | log_mse | log-magnitude^2 | 0.438771 | 0.444949 | 0.00617727 | [0.00243289, 0.00984271] | 6337 | 0 / 0 / 0 |
| 270° | loss | test loss | 0.0169725 | 0.0169684 | -4.0571e-06 | [-0.000141598, 0.000131446] | 6337 | 0 / 0 / 0 |

### Metric 1 -- how far the prediction moved

| angle | metric | unit | G | 95 % CI | n |
|---|---|---|---|---|---|
| 0° | EDT | s | 0 | [0, 0] | 6337 |
| 0° | C50 | dB | 0 | [0, 0] | 6337 |
| 0° | T60 | % of T60 | 0 | [0, 0] | 6337 |
| 0° | T60_abs | s | 0 | [0, 0] | 6337 |
| 0° | logspec_mad | log-magnitude | 0 | [0, 0] | 6337 |
| 0° | mag_rel_l2 | relative | 0 | [0, 0] | 6337 |
| 0° | wave_rel_l2 | relative | 0 | [0, 0] | 6337 |
| 0° | wave_mad | amplitude | 0 | [0, 0] | 6337 |
| 45° | EDT | s | 0.026247 | [0.0250286, 0.0275045] | 6337 |
| 45° | C50 | dB | 0.643992 | [0.618352, 0.671514] | 6337 |
| 45° | T60 | % of T60 | 3.78213 | [3.65756, 3.90867] | 6337 |
| 45° | T60_abs | s | 0.0216814 | [0.020961, 0.0224207] | 6337 |
| 45° | logspec_mad | log-magnitude | 0.179964 | [0.176698, 0.183221] | 6337 |
| 45° | mag_rel_l2 | relative | 0.233855 | [0.228989, 0.238756] | 6337 |
| 45° | wave_rel_l2 | relative | 0.877463 | [0.87101, 0.883833] | 6337 |
| 45° | wave_mad | amplitude | 0.00093279 | [0.000912658, 0.000953082] | 6337 |
| 90° | EDT | s | 0.0248886 | [0.0236795, 0.0261449] | 6337 |
| 90° | C50 | dB | 0.606292 | [0.578635, 0.635187] | 6337 |
| 90° | T60 | % of T60 | 3.18612 | [3.07735, 3.29713] | 6337 |
| 90° | T60_abs | s | 0.0181566 | [0.0175421, 0.0187919] | 6337 |
| 90° | logspec_mad | log-magnitude | 0.161193 | [0.157857, 0.164662] | 6337 |
| 90° | mag_rel_l2 | relative | 0.20766 | [0.202466, 0.213132] | 6337 |
| 90° | wave_rel_l2 | relative | 0.82367 | [0.816772, 0.830818] | 6337 |
| 90° | wave_mad | amplitude | 0.000853046 | [0.000835072, 0.000871177] | 6337 |
| 180° | EDT | s | 0.0255795 | [0.0243195, 0.0269076] | 6337 |
| 180° | C50 | dB | 0.642592 | [0.614312, 0.670439] | 6337 |
| 180° | T60 | % of T60 | 3.28893 | [3.18092, 3.39442] | 6337 |
| 180° | T60_abs | s | 0.0190145 | [0.0183647, 0.0196608] | 6337 |
| 180° | logspec_mad | log-magnitude | 0.170161 | [0.166533, 0.173813] | 6337 |
| 180° | mag_rel_l2 | relative | 0.21976 | [0.214045, 0.225554] | 6337 |
| 180° | wave_rel_l2 | relative | 0.834209 | [0.826766, 0.841811] | 6337 |
| 180° | wave_mad | amplitude | 0.000862235 | [0.000843899, 0.00088067] | 6337 |
| 270° | EDT | s | 0.0237631 | [0.0226031, 0.0249254] | 6337 |
| 270° | C50 | dB | 0.586039 | [0.559599, 0.613146] | 6337 |
| 270° | T60 | % of T60 | 3.14913 | [3.04155, 3.2588] | 6337 |
| 270° | T60_abs | s | 0.0180057 | [0.0174056, 0.01862] | 6337 |
| 270° | logspec_mad | log-magnitude | 0.156719 | [0.15345, 0.159922] | 6337 |
| 270° | mag_rel_l2 | relative | 0.201177 | [0.196101, 0.206397] | 6337 |
| 270° | wave_rel_l2 | relative | 0.815905 | [0.808864, 0.823139] | 6337 |
| 270° | wave_mad | amplitude | 0.000848959 | [0.000830712, 0.000867209] | 6337 |

### Standalone G on the broader population

The paired table above drops a query whose error is invalid at either angle; this one keeps every query with a finite g_alpha (not the paired mask). It is reported separately and never mixed into the paired delta / G / R.

| angle | metric | unit | G | 95 % CI (query) | n | 95 % CI (room) | rooms | converged |
|---|---|---|---|---|---|---|---|---|
| 0° | EDT | s | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 0° | C50 | dB | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 0° | T60 | % of T60 | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 0° | T60_abs | s | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 45° | EDT | s | 0.026247 | [0.0250286, 0.0275045] | 6337 | [0.0106274, 0.0436831] | 17 | True |
| 45° | C50 | dB | 0.643992 | [0.618352, 0.671514] | 6337 | [0.282762, 0.932867] | 17 | True |
| 45° | T60 | % of T60 | 3.78213 | [3.65756, 3.90867] | 6337 | [2.33341, 5.00074] | 17 | True |
| 45° | T60_abs | s | 0.0216814 | [0.020961, 0.0224207] | 6337 | [0.0121307, 0.0289498] | 17 | True |
| 90° | EDT | s | 0.0248886 | [0.0236795, 0.0261449] | 6337 | [0.00879433, 0.0429234] | 17 | True |
| 90° | C50 | dB | 0.606292 | [0.578635, 0.635187] | 6337 | [0.247113, 0.923185] | 17 | True |
| 90° | T60 | % of T60 | 3.18612 | [3.07735, 3.29713] | 6337 | [1.97778, 3.94786] | 17 | True |
| 90° | T60_abs | s | 0.0181566 | [0.0175421, 0.0187919] | 6337 | [0.00995384, 0.0232963] | 17 | True |
| 180° | EDT | s | 0.0255795 | [0.0243195, 0.0269076] | 6337 | [0.00950046, 0.0435136] | 17 | True |
| 180° | C50 | dB | 0.642592 | [0.614312, 0.670439] | 6337 | [0.26442, 0.954047] | 17 | True |
| 180° | T60 | % of T60 | 3.28893 | [3.18092, 3.39442] | 6337 | [1.99242, 4.10044] | 17 | True |
| 180° | T60_abs | s | 0.0190145 | [0.0183647, 0.0196608] | 6337 | [0.0103195, 0.0243471] | 17 | True |
| 270° | EDT | s | 0.0237631 | [0.0226031, 0.0249254] | 6337 | [0.00901262, 0.0399028] | 17 | True |
| 270° | C50 | dB | 0.586039 | [0.559599, 0.613146] | 6337 | [0.252683, 0.865539] | 17 | True |
| 270° | T60 | % of T60 | 3.14913 | [3.04155, 3.2588] | 6337 | [1.96865, 3.89376] | 17 | True |
| 270° | T60_abs | s | 0.0180057 | [0.0174056, 0.01862] | 6337 | [0.0102731, 0.0228427] | 17 | True |

### R2 -- the ratio G / delta, with its status

| angle | metric | R (query) | 95 % CI | status (query) | status (room) | converged | headline |
|---|---|---|---|---|---|---|---|
| 0° | EDT | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 0° | C50 | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 0° | T60 | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 0° | T60_abs | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 45° | EDT | -44.879 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 45° | C50 | 19.462 | [10.2688, 84.9662] | defined | denominator uncertain | True | yes |
| 45° | T60 | 7.99817 | [6.1339, 11.6257] | defined | defined | True | yes |
| 45° | T60_abs | 9.69539 | [7.25983, 14.7784] | defined | defined | True | yes |
| 90° | EDT | -15.6497 | [-50.2146, -9.12988] | improvement | denominator uncertain | True | predictions shift while net error improves |
| 90° | C50 | -218.504 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 90° | T60 | 12.2985 | [8.30624, 24.2861] | defined | denominator uncertain | True | yes |
| 90° | T60_abs | 16.3898 | [10.3696, 39.5389] | defined | denominator uncertain | True | yes |
| 180° | EDT | -20.9292 | [-122.13, -9.65907] | improvement | denominator uncertain | True | predictions shift while net error improves |
| 180° | C50 | 76.0753 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 180° | T60 | -13.9648 | [-30.0604, -9.0322] | improvement | denominator uncertain | True | predictions shift while net error improves |
| 180° | T60_abs | -15.4581 | [-33.7307, -10.0124] | improvement | denominator uncertain | True | predictions shift while net error improves |
| 270° | EDT | -25.353 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 270° | C50 | 27.0388 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 270° | T60 | 19.1724 | [10.6286, 71.6059] | defined | denominator uncertain | True | yes |
| 270° | T60_abs | 38.7349 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |

### Room-cluster intervals (secondary)

Room-cluster intervals resample the rooms with replacement and keep every query of a drawn room with its multiplicity (query-weighted, never averaged per room). With so few clusters they are wide, and a room-level ratio inherits its denominator's status: they qualify the query-level intervals, they do not replace them.

| angle | metric | delta CI (room) | G CI (room) | R (room) | 95 % CI (room) | status | rooms | converged (room) |
|---|---|---|---|---|---|---|---|---|
| 0° | EDT | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 0° | C50 | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 0° | T60 | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 0° | T60_abs | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 45° | EDT | [-0.00272698, 0.00169677] | [0.0106274, 0.0436831] | -44.879 | [-, -] | denominator uncertain | 17 | True |
| 45° | C50 | [-0.0092749, 0.0748251] | [0.282762, 0.932867] | 19.462 | [-, -] | denominator uncertain | 17 | True |
| 45° | T60 | [0.0783208, 0.825391] | [2.33341, 5.00074] | 7.99817 | [4.54802, 30.5391] | defined | 17 | True |
| 45° | T60_abs | [0.000148687, 0.00413327] | [0.0121307, 0.0289498] | 9.69539 | [4.91674, 52.914] | defined | 17 | True |
| 90° | EDT | [-0.00330233, 0.000370408] | [0.00879433, 0.0429234] | -15.6497 | [-, -] | denominator uncertain | 17 | True |
| 90° | C50 | [-0.0299972, 0.0194338] | [0.247113, 0.923185] | -218.504 | [-, -] | denominator uncertain | 17 | True |
| 90° | T60 | [-0.0908587, 0.651985] | [1.97778, 3.94786] | 12.2985 | [-, -] | denominator uncertain | 17 | True |
| 90° | T60_abs | [-0.000444291, 0.0026897] | [0.00995384, 0.0232963] | 16.3898 | [-, -] | denominator uncertain | 17 | True |
| 180° | EDT | [-0.00351141, 0.000528551] | [0.00950046, 0.0435136] | -20.9292 | [-, -] | denominator uncertain | 17 | True |
| 180° | C50 | [-0.0534468, 0.0713397] | [0.26442, 0.954047] | 76.0753 | [-, -] | denominator uncertain | 17 | True |
| 180° | T60 | [-0.610305, 0.0630343] | [1.99242, 4.10044] | -13.9648 | [-, -] | denominator uncertain | 17 | True |
| 180° | T60_abs | [-0.00298752, 0.000286425] | [0.0103195, 0.0243471] | -15.4581 | [-, -] | denominator uncertain | 17 | True |
| 270° | EDT | [-0.00182741, 4.16598e-05] | [0.00901262, 0.0399028] | -25.353 | [-, -] | denominator uncertain | 17 | True |
| 270° | C50 | [-0.0345056, 0.0833118] | [0.252683, 0.865539] | 27.0388 | [-, -] | denominator uncertain | 17 | True |
| 270° | T60 | [-0.084025, 0.37455] | [1.96865, 3.89376] | 19.1724 | [-, -] | denominator uncertain | 17 | True |
| 270° | T60_abs | [-0.00068506, 0.00157708] | [0.0102731, 0.0228427] | 38.7349 | [-, -] | denominator uncertain | 17 | True |

### Controls

| control | k | repeats | max \|dwave\| | max \|dlogspec\| | max \|dacoustic\| | validity mismatches | non-finite | problems | ok |
|---|---|---|---|---|---|---|---|---|---|

## released_k1

`ckpt/exp10/released_k1_all` -- 6337 queries, 17 rooms, execution `20260927T062839218781Z-93fc61b3802c4e01bb2aa2d55e76c591`.

Context band: historical baseline evaluation variability (SimpleViT, exp_04), context only -- C50 0.0133, EDT 0.609, T60 0.101.

### Metric 2 -- accuracy against the ground truth

| angle | metric | unit | mean k=0 | mean k | delta | 95 % CI | n_mask | excluded (0 / alpha / gap) |
|---|---|---|---|---|---|---|---|---|
| 0° | EDT | s | 0.0839851 | 0.0839851 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | C50 | dB | 2.0846 | 2.0846 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | T60 | % of T60 | 13.7973 | 13.7973 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | T60_abs | s | 0.0677937 | 0.0677937 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | log_mse | log-magnitude^2 | 0.809389 | 0.809389 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | loss | test loss | 0.0253663 | 0.0253663 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 45° | EDT | s | 0.0839851 | 0.0838608 | -0.00012431 | [-0.00067678, 0.000409459] | 6337 | 0 / 0 / 0 |
| 45° | C50 | dB | 2.0846 | 2.0938 | 0.00920235 | [0.000433264, 0.0178473] | 6337 | 0 / 0 / 0 |
| 45° | T60 | % of T60 | 13.7973 | 13.8524 | 0.0550954 | [-0.00341475, 0.11632] | 6337 | 0 / 0 / 0 |
| 45° | T60_abs | s | 0.0677937 | 0.0681247 | 0.000330936 | [1.54111e-05, 0.00065138] | 6337 | 0 / 0 / 0 |
| 45° | log_mse | log-magnitude^2 | 0.809389 | 0.810312 | 0.000922712 | [-0.00210924, 0.00395869] | 6337 | 0 / 0 / 0 |
| 45° | loss | test loss | 0.0253663 | 0.0254019 | 3.55534e-05 | [-2.47704e-05, 9.43562e-05] | 6337 | 0 / 0 / 0 |
| 90° | EDT | s | 0.0839851 | 0.0831996 | -0.000785539 | [-0.00134415, -0.00024372] | 6337 | 0 / 0 / 0 |
| 90° | C50 | dB | 2.0846 | 2.11605 | 0.0314509 | [0.0225118, 0.0403416] | 6337 | 0 / 0 / 0 |
| 90° | T60 | % of T60 | 13.7973 | 13.8218 | 0.0245366 | [-0.0298465, 0.0799912] | 6337 | 0 / 0 / 0 |
| 90° | T60_abs | s | 0.0677937 | 0.0679104 | 0.000116603 | [-0.000172918, 0.000413631] | 6337 | 0 / 0 / 0 |
| 90° | log_mse | log-magnitude^2 | 0.809389 | 0.810608 | 0.00121836 | [-0.00159288, 0.00402467] | 6337 | 0 / 0 / 0 |
| 90° | loss | test loss | 0.0253663 | 0.0254398 | 7.34447e-05 | [2.17516e-05, 0.000125507] | 6337 | 0 / 0 / 0 |
| 180° | EDT | s | 0.0839851 | 0.0828827 | -0.00110248 | [-0.00157125, -0.000637276] | 6337 | 0 / 0 / 0 |
| 180° | C50 | dB | 2.0846 | 2.1317 | 0.0471005 | [0.0374021, 0.0568101] | 6337 | 0 / 0 / 0 |
| 180° | T60 | % of T60 | 13.7973 | 13.8495 | 0.0521761 | [-0.00540737, 0.110465] | 6337 | 0 / 0 / 0 |
| 180° | T60_abs | s | 0.0677937 | 0.0680852 | 0.000291432 | [-2.53613e-05, 0.000607047] | 6337 | 0 / 0 / 0 |
| 180° | log_mse | log-magnitude^2 | 0.809389 | 0.816883 | 0.00749389 | [0.00459475, 0.0104267] | 6337 | 0 / 0 / 0 |
| 180° | loss | test loss | 0.0253663 | 0.025391 | 2.46706e-05 | [-2.72575e-05, 7.71548e-05] | 6337 | 0 / 0 / 0 |
| 270° | EDT | s | 0.0839851 | 0.0837684 | -0.00021676 | [-0.000782233, 0.000338416] | 6337 | 0 / 0 / 0 |
| 270° | C50 | dB | 2.0846 | 2.10673 | 0.0221302 | [0.0143669, 0.0299444] | 6337 | 0 / 0 / 0 |
| 270° | T60 | % of T60 | 13.7973 | 13.8284 | 0.0310536 | [-0.0226386, 0.0862165] | 6337 | 0 / 0 / 0 |
| 270° | T60_abs | s | 0.0677937 | 0.0679111 | 0.000117354 | [-0.000171559, 0.000412415] | 6337 | 0 / 0 / 0 |
| 270° | log_mse | log-magnitude^2 | 0.809389 | 0.815222 | 0.00583253 | [0.00345579, 0.008263] | 6337 | 0 / 0 / 0 |
| 270° | loss | test loss | 0.0253663 | 0.0254901 | 0.000123791 | [7.2058e-05, 0.000175427] | 6337 | 0 / 0 / 0 |

### Metric 1 -- how far the prediction moved

| angle | metric | unit | G | 95 % CI | n |
|---|---|---|---|---|---|
| 0° | EDT | s | 0 | [0, 0] | 6337 |
| 0° | C50 | dB | 0 | [0, 0] | 6337 |
| 0° | T60 | % of T60 | 0 | [0, 0] | 6337 |
| 0° | T60_abs | s | 0 | [0, 0] | 6337 |
| 0° | logspec_mad | log-magnitude | 0 | [0, 0] | 6337 |
| 0° | mag_rel_l2 | relative | 0 | [0, 0] | 6337 |
| 0° | wave_rel_l2 | relative | 0 | [0, 0] | 6337 |
| 0° | wave_mad | amplitude | 0 | [0, 0] | 6337 |
| 45° | EDT | s | 0.0125152 | [0.0120292, 0.0130116] | 6337 |
| 45° | C50 | dB | 0.227541 | [0.220675, 0.234798] | 6337 |
| 45° | T60 | % of T60 | 1.46307 | [1.42081, 1.50778] | 6337 |
| 45° | T60_abs | s | 0.00838642 | [0.0081271, 0.00866198] | 6337 |
| 45° | logspec_mad | log-magnitude | 0.108922 | [0.107031, 0.110804] | 6337 |
| 45° | mag_rel_l2 | relative | 0.0828363 | [0.081041, 0.0846716] | 6337 |
| 45° | wave_rel_l2 | relative | 0.330942 | [0.327611, 0.334247] | 6337 |
| 45° | wave_mad | amplitude | 0.000376491 | [0.000368474, 0.000384484] | 6337 |
| 90° | EDT | s | 0.0117699 | [0.0112549, 0.0122835] | 6337 |
| 90° | C50 | dB | 0.226879 | [0.219687, 0.234232] | 6337 |
| 90° | T60 | % of T60 | 1.31834 | [1.27646, 1.36037] | 6337 |
| 90° | T60_abs | s | 0.00753723 | [0.00727719, 0.00779515] | 6337 |
| 90° | logspec_mad | log-magnitude | 0.0917413 | [0.0900731, 0.0933964] | 6337 |
| 90° | mag_rel_l2 | relative | 0.0651904 | [0.0635487, 0.0668577] | 6337 |
| 90° | wave_rel_l2 | relative | 0.308823 | [0.305417, 0.312282] | 6337 |
| 90° | wave_mad | amplitude | 0.00034743 | [0.000340283, 0.000354594] | 6337 |
| 180° | EDT | s | 0.0105805 | [0.0101345, 0.0110344] | 6337 |
| 180° | C50 | dB | 0.245019 | [0.237149, 0.252968] | 6337 |
| 180° | T60 | % of T60 | 1.43724 | [1.38974, 1.48629] | 6337 |
| 180° | T60_abs | s | 0.00812642 | [0.00783824, 0.00842007] | 6337 |
| 180° | logspec_mad | log-magnitude | 0.0959089 | [0.0938303, 0.0981082] | 6337 |
| 180° | mag_rel_l2 | relative | 0.0738707 | [0.0712981, 0.0765852] | 6337 |
| 180° | wave_rel_l2 | relative | 0.329597 | [0.325348, 0.333944] | 6337 |
| 180° | wave_mad | amplitude | 0.000364718 | [0.000357191, 0.000372169] | 6337 |
| 270° | EDT | s | 0.011923 | [0.0114055, 0.0124477] | 6337 |
| 270° | C50 | dB | 0.212525 | [0.206534, 0.218752] | 6337 |
| 270° | T60 | % of T60 | 1.30602 | [1.26589, 1.34642] | 6337 |
| 270° | T60_abs | s | 0.00739138 | [0.00714671, 0.00763547] | 6337 |
| 270° | logspec_mad | log-magnitude | 0.0914167 | [0.0898159, 0.0930758] | 6337 |
| 270° | mag_rel_l2 | relative | 0.0651467 | [0.0634676, 0.0668854] | 6337 |
| 270° | wave_rel_l2 | relative | 0.310548 | [0.307125, 0.314084] | 6337 |
| 270° | wave_mad | amplitude | 0.000348367 | [0.000341198, 0.000355326] | 6337 |

### Standalone G on the broader population

The paired table above drops a query whose error is invalid at either angle; this one keeps every query with a finite g_alpha (not the paired mask). It is reported separately and never mixed into the paired delta / G / R.

| angle | metric | unit | G | 95 % CI (query) | n | 95 % CI (room) | rooms | converged |
|---|---|---|---|---|---|---|---|---|
| 0° | EDT | s | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 0° | C50 | dB | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 0° | T60 | % of T60 | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 0° | T60_abs | s | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 45° | EDT | s | 0.0125152 | [0.0120292, 0.0130116] | 6337 | [0.00602795, 0.0189339] | 17 | True |
| 45° | C50 | dB | 0.227541 | [0.220675, 0.234798] | 6337 | [0.125524, 0.310716] | 17 | True |
| 45° | T60 | % of T60 | 1.46307 | [1.42081, 1.50778] | 6337 | [1.06342, 1.74934] | 17 | True |
| 45° | T60_abs | s | 0.00838642 | [0.0081271, 0.00866198] | 6337 | [0.00550609, 0.0101841] | 17 | True |
| 90° | EDT | s | 0.0117699 | [0.0112549, 0.0122835] | 6337 | [0.00487101, 0.0191797] | 17 | True |
| 90° | C50 | dB | 0.226879 | [0.219687, 0.234232] | 6337 | [0.117626, 0.317588] | 17 | True |
| 90° | T60 | % of T60 | 1.31834 | [1.27646, 1.36037] | 6337 | [0.86846, 1.69796] | 17 | True |
| 90° | T60_abs | s | 0.00753723 | [0.00727719, 0.00779515] | 6337 | [0.00423268, 0.00986536] | 17 | True |
| 180° | EDT | s | 0.0105805 | [0.0101345, 0.0110344] | 6337 | [0.00535696, 0.0154187] | 17 | True |
| 180° | C50 | dB | 0.245019 | [0.237149, 0.252968] | 6337 | [0.133054, 0.34499] | 17 | True |
| 180° | T60 | % of T60 | 1.43724 | [1.38974, 1.48629] | 6337 | [0.916329, 1.97974] | 17 | True |
| 180° | T60_abs | s | 0.00812642 | [0.00783824, 0.00842007] | 6337 | [0.00459233, 0.0110773] | 17 | True |
| 270° | EDT | s | 0.011923 | [0.0114055, 0.0124477] | 6337 | [0.00499212, 0.0192397] | 17 | True |
| 270° | C50 | dB | 0.212525 | [0.206534, 0.218752] | 6337 | [0.115458, 0.293107] | 17 | True |
| 270° | T60 | % of T60 | 1.30602 | [1.26589, 1.34642] | 6337 | [0.896247, 1.64208] | 17 | True |
| 270° | T60_abs | s | 0.00739138 | [0.00714671, 0.00763547] | 6337 | [0.00439379, 0.00948845] | 17 | True |

### R2 -- the ratio G / delta, with its status

| angle | metric | R (query) | 95 % CI | status (query) | status (room) | converged | headline |
|---|---|---|---|---|---|---|---|
| 0° | EDT | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 0° | C50 | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 0° | T60 | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 0° | T60_abs | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 45° | EDT | -100.677 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 45° | C50 | 24.7264 | [11.1608, 135.723] | defined | denominator uncertain | True | yes |
| 45° | T60 | 26.5551 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 45° | T60_abs | 25.3415 | [11.4073, 148.777] | defined | denominator uncertain | True | yes |
| 90° | EDT | -14.9832 | [-45.5417, -8.73417] | improvement | denominator uncertain | True | predictions shift while net error improves |
| 90° | C50 | 7.21376 | [5.65459, 10.0113] | defined | denominator uncertain | True | yes |
| 90° | T60 | 53.7294 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 90° | T60_abs | 64.6402 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 180° | EDT | -9.597 | [-16.6342, -6.77564] | improvement | improvement | True | predictions shift while net error improves |
| 180° | C50 | 5.20204 | [4.35784, 6.47895] | defined | denominator uncertain | True | yes |
| 180° | T60 | 27.546 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 180° | T60_abs | 27.8845 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 270° | EDT | -55.0055 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 270° | C50 | 9.6034 | [7.11924, 14.7612] | defined | denominator uncertain | True | yes |
| 270° | T60 | 42.057 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 270° | T60_abs | 62.9835 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |

### Room-cluster intervals (secondary)

Room-cluster intervals resample the rooms with replacement and keep every query of a drawn room with its multiplicity (query-weighted, never averaged per room). With so few clusters they are wide, and a room-level ratio inherits its denominator's status: they qualify the query-level intervals, they do not replace them.

| angle | metric | delta CI (room) | G CI (room) | R (room) | 95 % CI (room) | status | rooms | converged (room) |
|---|---|---|---|---|---|---|---|---|
| 0° | EDT | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 0° | C50 | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 0° | T60 | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 0° | T60_abs | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 45° | EDT | [-0.00116644, 0.000851368] | [0.00602795, 0.0189339] | -100.677 | [-, -] | denominator uncertain | 17 | True |
| 45° | C50 | [-0.00215058, 0.0187678] | [0.125524, 0.310716] | 24.7264 | [-, -] | denominator uncertain | 17 | True |
| 45° | T60 | [-0.040277, 0.131604] | [1.06342, 1.74934] | 26.5551 | [-, -] | denominator uncertain | 17 | True |
| 45° | T60_abs | [-0.000220274, 0.000775589] | [0.00550609, 0.0101841] | 25.3415 | [-, -] | denominator uncertain | 17 | True |
| 90° | EDT | [-0.0025478, 0.000614319] | [0.00487101, 0.0191797] | -14.9832 | [-, -] | denominator uncertain | 17 | True |
| 90° | C50 | [-0.00518317, 0.0757976] | [0.117626, 0.317588] | 7.21376 | [-, -] | denominator uncertain | 17 | True |
| 90° | T60 | [-0.0531795, 0.111082] | [0.86846, 1.69796] | 53.7294 | [-, -] | denominator uncertain | 17 | True |
| 90° | T60_abs | [-0.000249905, 0.000566508] | [0.00423268, 0.00986536] | 64.6402 | [-, -] | denominator uncertain | 17 | True |
| 180° | EDT | [-0.0017615, -0.000243758] | [0.00535696, 0.0154187] | -9.597 | [-22.6144, -7.68047] | improvement | 17 | True |
| 180° | C50 | [-0.00168716, 0.115572] | [0.133054, 0.34499] | 5.20204 | [-, -] | denominator uncertain | 17 | True |
| 180° | T60 | [-0.0626254, 0.196783] | [0.916329, 1.97974] | 27.546 | [-, -] | denominator uncertain | 17 | True |
| 180° | T60_abs | [-0.00030881, 0.00108023] | [0.00459233, 0.0110773] | 27.8845 | [-, -] | denominator uncertain | 17 | True |
| 270° | EDT | [-0.00143183, 0.000730268] | [0.00499212, 0.0192397] | -55.0055 | [-, -] | denominator uncertain | 17 | True |
| 270° | C50 | [-0.014736, 0.0670635] | [0.115458, 0.293107] | 9.6034 | [-, -] | denominator uncertain | 17 | True |
| 270° | T60 | [-0.0223772, 0.0870071] | [0.896247, 1.64208] | 42.057 | [-, -] | denominator uncertain | 17 | True |
| 270° | T60_abs | [-9.54117e-05, 0.000311082] | [0.00439379, 0.00948845] | 62.9835 | [-, -] | denominator uncertain | 17 | True |

### Controls

| control | k | repeats | max \|dwave\| | max \|dlogspec\| | max \|dacoustic\| | validity mismatches | non-finite | problems | ok |
|---|---|---|---|---|---|---|---|---|---|

## control_k8

`ckpt/exp10/control_k8_all` -- 6337 queries, 17 rooms, execution `20260927T025953876010Z-4b35102e25014d87875828367b73a0ff`.

Context band: historical baseline evaluation variability (SimpleViT, exp_04), context only -- C50 0.0053, EDT 0.186, T60 0.009.

### Metric 2 -- accuracy against the ground truth

| angle | metric | unit | mean k=0 | mean k | delta | 95 % CI | n_mask | excluded (0 / alpha / gap) |
|---|---|---|---|---|---|---|---|---|
| 0° | EDT | s | 0.0469877 | 0.0469877 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | C50 | dB | 1.23512 | 1.23512 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | T60 | % of T60 | 9.55864 | 9.55864 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | T60_abs | s | 0.0486895 | 0.0486895 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | log_mse | log-magnitude^2 | 0.422307 | 0.422307 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | loss | test loss | 0.0161415 | 0.0161415 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 45° | EDT | s | 0.0469877 | 0.0476181 | 0.000630398 | [-0.000117155, 0.00136141] | 6337 | 0 / 0 / 0 |
| 45° | C50 | dB | 1.23512 | 1.29799 | 0.0628755 | [0.0434337, 0.0816143] | 6337 | 0 / 0 / 0 |
| 45° | T60 | % of T60 | 9.55864 | 9.58649 | 0.0278458 | [-0.0765533, 0.132203] | 6337 | 0 / 0 / 0 |
| 45° | T60_abs | s | 0.0486895 | 0.0487553 | 6.5848e-05 | [-0.000503063, 0.000628842] | 6337 | 0 / 0 / 0 |
| 45° | log_mse | log-magnitude^2 | 0.422307 | 0.424314 | 0.00200775 | [-0.00166602, 0.00563719] | 6337 | 0 / 0 / 0 |
| 45° | loss | test loss | 0.0161415 | 0.0164867 | 0.000345164 | [0.000239496, 0.000451783] | 6337 | 0 / 0 / 0 |
| 90° | EDT | s | 0.0469877 | 0.0475755 | 0.000587801 | [5.27224e-05, 0.00110465] | 6337 | 0 / 0 / 0 |
| 90° | C50 | dB | 1.23512 | 1.25916 | 0.0240375 | [0.0103761, 0.0373959] | 6337 | 0 / 0 / 0 |
| 90° | T60 | % of T60 | 9.55864 | 9.43914 | -0.119504 | [-0.200896, -0.0394727] | 6337 | 0 / 0 / 0 |
| 90° | T60_abs | s | 0.0486895 | 0.0481101 | -0.000579407 | [-0.000990987, -0.000175421] | 6337 | 0 / 0 / 0 |
| 90° | log_mse | log-magnitude^2 | 0.422307 | 0.423684 | 0.00137752 | [-0.000379855, 0.00313065] | 6337 | 0 / 0 / 0 |
| 90° | loss | test loss | 0.0161415 | 0.0162195 | 7.79506e-05 | [-4.25718e-06, 0.000159418] | 6337 | 0 / 0 / 0 |
| 180° | EDT | s | 0.0469877 | 0.0469606 | -2.70949e-05 | [-0.000555267, 0.000500551] | 6337 | 0 / 0 / 0 |
| 180° | C50 | dB | 1.23512 | 1.23178 | -0.00334354 | [-0.018753, 0.0116002] | 6337 | 0 / 0 / 0 |
| 180° | T60 | % of T60 | 9.55864 | 9.21746 | -0.341186 | [-0.420665, -0.262917] | 6337 | 0 / 0 / 0 |
| 180° | T60_abs | s | 0.0486895 | 0.0469338 | -0.00175572 | [-0.00217621, -0.00135305] | 6337 | 0 / 0 / 0 |
| 180° | log_mse | log-magnitude^2 | 0.422307 | 0.420359 | -0.00194747 | [-0.00363697, -0.000297915] | 6337 | 0 / 0 / 0 |
| 180° | loss | test loss | 0.0161415 | 0.0161148 | -2.66958e-05 | [-0.000114423, 6.02747e-05] | 6337 | 0 / 0 / 0 |
| 270° | EDT | s | 0.0469877 | 0.0473052 | 0.000317496 | [-0.000208439, 0.000843863] | 6337 | 0 / 0 / 0 |
| 270° | C50 | dB | 1.23512 | 1.24667 | 0.0115502 | [-0.00283678, 0.0261682] | 6337 | 0 / 0 / 0 |
| 270° | T60 | % of T60 | 9.55864 | 9.3878 | -0.170845 | [-0.246106, -0.0975554] | 6337 | 0 / 0 / 0 |
| 270° | T60_abs | s | 0.0486895 | 0.0477826 | -0.000906865 | [-0.00131435, -0.000496959] | 6337 | 0 / 0 / 0 |
| 270° | log_mse | log-magnitude^2 | 0.422307 | 0.423955 | 0.00164791 | [-0.000238588, 0.00359953] | 6337 | 0 / 0 / 0 |
| 270° | loss | test loss | 0.0161415 | 0.0161989 | 5.73751e-05 | [-3.20332e-05, 0.000145364] | 6337 | 0 / 0 / 0 |

### Metric 1 -- how far the prediction moved

| angle | metric | unit | G | 95 % CI | n |
|---|---|---|---|---|---|
| 0° | EDT | s | 0 | [0, 0] | 6337 |
| 0° | C50 | dB | 0 | [0, 0] | 6337 |
| 0° | T60 | % of T60 | 0 | [0, 0] | 6337 |
| 0° | T60_abs | s | 0 | [0, 0] | 6337 |
| 0° | logspec_mad | log-magnitude | 0 | [0, 0] | 6337 |
| 0° | mag_rel_l2 | relative | 0 | [0, 0] | 6337 |
| 0° | wave_rel_l2 | relative | 0 | [0, 0] | 6337 |
| 0° | wave_mad | amplitude | 0 | [0, 0] | 6337 |
| 45° | EDT | s | 0.0163912 | [0.0156634, 0.0171474] | 6337 |
| 45° | C50 | dB | 0.441897 | [0.423234, 0.461617] | 6337 |
| 45° | T60 | % of T60 | 2.47823 | [2.38767, 2.57175] | 6337 |
| 45° | T60_abs | s | 0.0146337 | [0.0140824, 0.0151975] | 6337 |
| 45° | logspec_mad | log-magnitude | 0.127426 | [0.123983, 0.130939] | 6337 |
| 45° | mag_rel_l2 | relative | 0.165002 | [0.160943, 0.16918] | 6337 |
| 45° | wave_rel_l2 | relative | 0.736425 | [0.7291, 0.743655] | 6337 |
| 45° | wave_mad | amplitude | 0.000709374 | [0.000695255, 0.000723648] | 6337 |
| 90° | EDT | s | 0.0104617 | [0.00995132, 0.0110005] | 6337 |
| 90° | C50 | dB | 0.299991 | [0.286222, 0.314052] | 6337 |
| 90° | T60 | % of T60 | 1.67039 | [1.60447, 1.73585] | 6337 |
| 90° | T60_abs | s | 0.00964106 | [0.0092638, 0.0100212] | 6337 |
| 90° | logspec_mad | log-magnitude | 0.0769292 | [0.0750698, 0.0788253] | 6337 |
| 90° | mag_rel_l2 | relative | 0.10417 | [0.101309, 0.107071] | 6337 |
| 90° | wave_rel_l2 | relative | 0.616957 | [0.610577, 0.623508] | 6337 |
| 90° | wave_mad | amplitude | 0.000598375 | [0.000586176, 0.00061034] | 6337 |
| 180° | EDT | s | 0.0100989 | [0.00961157, 0.0106251] | 6337 |
| 180° | C50 | dB | 0.308278 | [0.292709, 0.32472] | 6337 |
| 180° | T60 | % of T60 | 1.53514 | [1.47026, 1.60091] | 6337 |
| 180° | T60_abs | s | 0.00891832 | [0.00854362, 0.00930552] | 6337 |
| 180° | logspec_mad | log-magnitude | 0.0711573 | [0.0693631, 0.0730232] | 6337 |
| 180° | mag_rel_l2 | relative | 0.0989147 | [0.09601, 0.101933] | 6337 |
| 180° | wave_rel_l2 | relative | 0.584408 | [0.577663, 0.59114] | 6337 |
| 180° | wave_mad | amplitude | 0.000547374 | [0.000536493, 0.000558409] | 6337 |
| 270° | EDT | s | 0.0105038 | [0.0100146, 0.011015] | 6337 |
| 270° | C50 | dB | 0.302369 | [0.2884, 0.316952] | 6337 |
| 270° | T60 | % of T60 | 1.65513 | [1.59299, 1.7185] | 6337 |
| 270° | T60_abs | s | 0.00958584 | [0.00922733, 0.00995238] | 6337 |
| 270° | logspec_mad | log-magnitude | 0.0768196 | [0.0749484, 0.0787294] | 6337 |
| 270° | mag_rel_l2 | relative | 0.104 | [0.101132, 0.106954] | 6337 |
| 270° | wave_rel_l2 | relative | 0.617744 | [0.611247, 0.624238] | 6337 |
| 270° | wave_mad | amplitude | 0.000605655 | [0.000592982, 0.000618086] | 6337 |

### Standalone G on the broader population

The paired table above drops a query whose error is invalid at either angle; this one keeps every query with a finite g_alpha (not the paired mask). It is reported separately and never mixed into the paired delta / G / R.

| angle | metric | unit | G | 95 % CI (query) | n | 95 % CI (room) | rooms | converged |
|---|---|---|---|---|---|---|---|---|
| 0° | EDT | s | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 0° | C50 | dB | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 0° | T60 | % of T60 | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 0° | T60_abs | s | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 45° | EDT | s | 0.0163912 | [0.0156634, 0.0171474] | 6337 | [0.00551172, 0.0275383] | 17 | True |
| 45° | C50 | dB | 0.441897 | [0.423234, 0.461617] | 6337 | [0.153886, 0.683115] | 17 | True |
| 45° | T60 | % of T60 | 2.47823 | [2.38767, 2.57175] | 6337 | [1.41078, 3.18768] | 17 | True |
| 45° | T60_abs | s | 0.0146337 | [0.0140824, 0.0151975] | 6337 | [0.00740083, 0.0196873] | 17 | True |
| 90° | EDT | s | 0.0104617 | [0.00995132, 0.0110005] | 6337 | [0.00441662, 0.0164951] | 17 | True |
| 90° | C50 | dB | 0.299991 | [0.286222, 0.314052] | 6337 | [0.120234, 0.45158] | 17 | True |
| 90° | T60 | % of T60 | 1.67039 | [1.60447, 1.73585] | 6337 | [1.01684, 2.20617] | 17 | True |
| 90° | T60_abs | s | 0.00964106 | [0.0092638, 0.0100212] | 6337 | [0.00541256, 0.0128769] | 17 | True |
| 180° | EDT | s | 0.0100989 | [0.00961157, 0.0106251] | 6337 | [0.00413645, 0.0154509] | 17 | True |
| 180° | C50 | dB | 0.308278 | [0.292709, 0.32472] | 6337 | [0.11371, 0.479489] | 17 | True |
| 180° | T60 | % of T60 | 1.53514 | [1.47026, 1.60091] | 6337 | [0.870701, 2.10811] | 17 | True |
| 180° | T60_abs | s | 0.00891832 | [0.00854362, 0.00930552] | 6337 | [0.00470109, 0.0122881] | 17 | True |
| 270° | EDT | s | 0.0105038 | [0.0100146, 0.011015] | 6337 | [0.00457033, 0.016367] | 17 | True |
| 270° | C50 | dB | 0.302369 | [0.2884, 0.316952] | 6337 | [0.123898, 0.451513] | 17 | True |
| 270° | T60 | % of T60 | 1.65513 | [1.59299, 1.7185] | 6337 | [1.05301, 2.10965] | 17 | True |
| 270° | T60_abs | s | 0.00958584 | [0.00922733, 0.00995238] | 6337 | [0.00562276, 0.0123636] | 17 | True |

### R2 -- the ratio G / delta, with its status

| angle | metric | R (query) | 95 % CI | status (query) | status (room) | converged | headline |
|---|---|---|---|---|---|---|---|
| 0° | EDT | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 0° | C50 | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 0° | T60 | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 0° | T60_abs | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 45° | EDT | 26.0013 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 45° | C50 | 7.02813 | [5.4277, 10.163] | defined | defined | True | yes |
| 45° | T60 | 88.9983 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 45° | T60_abs | 222.234 | [-, -] | undefined | denominator uncertain | True | unresolved Monte Carlo uncertainty |
| 90° | EDT | 17.798 | [8.80993, 95.2825] | defined | denominator uncertain | True | yes |
| 90° | C50 | 12.4801 | [8.01309, 28.7737] | defined | denominator uncertain | True | yes |
| 90° | T60 | -13.9777 | [-40.6316, -8.28093] | improvement | denominator uncertain | True | predictions shift while net error improves |
| 90° | T60_abs | -16.6395 | [-52.8591, -9.68814] | improvement | denominator uncertain | True | predictions shift while net error improves |
| 180° | EDT | -372.721 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 180° | C50 | -92.2011 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 180° | T60 | -4.49943 | [-5.74511, -3.6875] | improvement | improvement | True | predictions shift while net error improves |
| 180° | T60_abs | -5.07958 | [-6.49892, -4.16265] | improvement | improvement | True | predictions shift while net error improves |
| 270° | EDT | 33.0833 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 270° | C50 | 26.1788 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 270° | T60 | -9.68795 | [-16.9685, -6.77934] | improvement | improvement | True | predictions shift while net error improves |
| 270° | T60_abs | -10.5703 | [-19.0804, -7.34891] | improvement | improvement | True | predictions shift while net error improves |

### Room-cluster intervals (secondary)

Room-cluster intervals resample the rooms with replacement and keep every query of a drawn room with its multiplicity (query-weighted, never averaged per room). With so few clusters they are wide, and a room-level ratio inherits its denominator's status: they qualify the query-level intervals, they do not replace them.

| angle | metric | delta CI (room) | G CI (room) | R (room) | 95 % CI (room) | status | rooms | converged (room) |
|---|---|---|---|---|---|---|---|---|
| 0° | EDT | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 0° | C50 | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 0° | T60 | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 0° | T60_abs | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 45° | EDT | [-0.000128557, 0.00148779] | [0.00551172, 0.0275383] | 26.0013 | [-, -] | denominator uncertain | 17 | True |
| 45° | C50 | [0.00927337, 0.129502] | [0.153886, 0.683115] | 7.02813 | [4.19796, 35.4402] | defined | 17 | True |
| 45° | T60 | [-0.229826, 0.290793] | [1.41078, 3.18768] | 88.9983 | [-, -] | denominator uncertain | 17 | True |
| 45° | T60_abs | [-0.00136705, 0.00147685] | [0.00740083, 0.0196873] | 222.234 | [-, -] | denominator uncertain | 17 | True |
| 90° | EDT | [-0.000276709, 0.00127507] | [0.00441662, 0.0164951] | 17.798 | [-, -] | denominator uncertain | 17 | True |
| 90° | C50 | [-0.0140458, 0.0678862] | [0.120234, 0.45158] | 12.4801 | [-, -] | denominator uncertain | 17 | True |
| 90° | T60 | [-0.302507, 0.0319974] | [1.01684, 2.20617] | -13.9777 | [-, -] | denominator uncertain | 17 | True |
| 90° | T60_abs | [-0.00152004, 0.000241896] | [0.00541256, 0.0128769] | -16.6395 | [-, -] | denominator uncertain | 17 | True |
| 180° | EDT | [-0.000865537, 0.00052103] | [0.00413645, 0.0154509] | -372.721 | [-, -] | denominator uncertain | 17 | True |
| 180° | C50 | [-0.0383505, 0.0308554] | [0.11371, 0.479489] | -92.2011 | [-, -] | denominator uncertain | 17 | True |
| 180° | T60 | [-0.736126, -0.0632696] | [0.870701, 2.10811] | -4.49943 | [-16.7726, -2.70825] | improvement | 17 | True |
| 180° | T60_abs | [-0.0038595, -0.000315115] | [0.00470109, 0.0122881] | -5.07958 | [-20.2309, -2.96443] | improvement | 17 | True |
| 270° | EDT | [-0.00061096, 0.00110356] | [0.00457033, 0.016367] | 33.0833 | [-, -] | denominator uncertain | 17 | True |
| 270° | C50 | [-0.0164487, 0.0462401] | [0.123898, 0.451513] | 26.1788 | [-, -] | denominator uncertain | 17 | True |
| 270° | T60 | [-0.374872, -0.00567775] | [1.05301, 2.10965] | -9.68795 | [-73.1037, -4.70643] | improvement | 17 | True |
| 270° | T60_abs | [-0.00201261, -2.84948e-05] | [0.00562276, 0.0123636] | -10.5703 | [-92.712, -4.90975] | improvement | 17 | True |

### Controls

| control | k | repeats | max \|dwave\| | max \|dlogspec\| | max \|dacoustic\| | validity mismatches | non-finite | problems | ok |
|---|---|---|---|---|---|---|---|---|---|

## cyl_k8

`ckpt/exp10/cyl_k8_all` -- 6337 queries, 17 rooms, execution `20260927T051017740231Z-7d2e7e6db2994f4fa57533077c4c80a0`.

Context band: historical baseline evaluation variability (CylindricalViT, exp_04), context only -- C50 0.0059, EDT 0.253, T60 0.014.

### Metric 2 -- accuracy against the ground truth

| angle | metric | unit | mean k=0 | mean k | delta | 95 % CI | n_mask | excluded (0 / alpha / gap) |
|---|---|---|---|---|---|---|---|---|
| 0° | EDT | s | 0.0449422 | 0.0449422 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | C50 | dB | 1.22558 | 1.22558 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | T60 | % of T60 | 9.43614 | 9.43614 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | T60_abs | s | 0.0478317 | 0.0478317 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | log_mse | log-magnitude^2 | 0.416068 | 0.416068 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | loss | test loss | 0.0158925 | 0.0158925 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 45° | EDT | s | 0.0449422 | 0.0460893 | 0.00114709 | [0.00051828, 0.00177775] | 6337 | 0 / 0 / 0 |
| 45° | C50 | dB | 1.22558 | 1.28919 | 0.0636176 | [0.0474481, 0.0796725] | 6337 | 0 / 0 / 0 |
| 45° | T60 | % of T60 | 9.43614 | 9.59276 | 0.156622 | [0.0554993, 0.258373] | 6337 | 0 / 0 / 0 |
| 45° | T60_abs | s | 0.0478317 | 0.0486533 | 0.00082163 | [0.000277233, 0.00135802] | 6337 | 0 / 0 / 0 |
| 45° | log_mse | log-magnitude^2 | 0.416068 | 0.419687 | 0.00361855 | [0.000875839, 0.00638729] | 6337 | 0 / 0 / 0 |
| 45° | loss | test loss | 0.0158925 | 0.0163074 | 0.000414893 | [0.000322595, 0.000504673] | 6337 | 0 / 0 / 0 |
| 90° | EDT | s | 0.0449422 | 0.0458614 | 0.00091921 | [0.000488252, 0.00134846] | 6337 | 0 / 0 / 0 |
| 90° | C50 | dB | 1.22558 | 1.25759 | 0.0320111 | [0.0205599, 0.0436839] | 6337 | 0 / 0 / 0 |
| 90° | T60 | % of T60 | 9.43614 | 9.39454 | -0.0416013 | [-0.106816, 0.0241007] | 6337 | 0 / 0 / 0 |
| 90° | T60_abs | s | 0.0478317 | 0.047573 | -0.00025869 | [-0.000614727, 9.62139e-05] | 6337 | 0 / 0 / 0 |
| 90° | log_mse | log-magnitude^2 | 0.416068 | 0.419627 | 0.00355883 | [0.00192193, 0.0052251] | 6337 | 0 / 0 / 0 |
| 90° | loss | test loss | 0.0158925 | 0.0159359 | 4.33907e-05 | [-1.38883e-05, 0.000101588] | 6337 | 0 / 0 / 0 |
| 180° | EDT | s | 0.0449422 | 0.0453642 | 0.000421968 | [1.13683e-05, 0.0008387] | 6337 | 0 / 0 / 0 |
| 180° | C50 | dB | 1.22558 | 1.22895 | 0.00336787 | [-0.00870703, 0.0157027] | 6337 | 0 / 0 / 0 |
| 180° | T60 | % of T60 | 9.43614 | 9.13155 | -0.304588 | [-0.369236, -0.240844] | 6337 | 0 / 0 / 0 |
| 180° | T60_abs | s | 0.0478317 | 0.0462091 | -0.00162256 | [-0.00196312, -0.00128447] | 6337 | 0 / 0 / 0 |
| 180° | log_mse | log-magnitude^2 | 0.416068 | 0.415417 | -0.000651226 | [-0.00206196, 0.000723395] | 6337 | 0 / 0 / 0 |
| 180° | loss | test loss | 0.0158925 | 0.0158567 | -3.58175e-05 | [-8.80132e-05, 1.75021e-05] | 6337 | 0 / 0 / 0 |
| 270° | EDT | s | 0.0449422 | 0.0458303 | 0.000888079 | [0.000471349, 0.00129558] | 6337 | 0 / 0 / 0 |
| 270° | C50 | dB | 1.22558 | 1.24794 | 0.0223647 | [0.0114769, 0.0335391] | 6337 | 0 / 0 / 0 |
| 270° | T60 | % of T60 | 9.43614 | 9.3646 | -0.0715392 | [-0.134511, -0.00775223] | 6337 | 0 / 0 / 0 |
| 270° | T60_abs | s | 0.0478317 | 0.0473668 | -0.000464865 | [-0.000812111, -0.000120745] | 6337 | 0 / 0 / 0 |
| 270° | log_mse | log-magnitude^2 | 0.416068 | 0.417655 | 0.0015864 | [0.000152854, 0.00302288] | 6337 | 0 / 0 / 0 |
| 270° | loss | test loss | 0.0158925 | 0.0160327 | 0.000140167 | [8.24931e-05, 0.000197346] | 6337 | 0 / 0 / 0 |

### Metric 1 -- how far the prediction moved

| angle | metric | unit | G | 95 % CI | n |
|---|---|---|---|---|---|
| 0° | EDT | s | 0 | [0, 0] | 6337 |
| 0° | C50 | dB | 0 | [0, 0] | 6337 |
| 0° | T60 | % of T60 | 0 | [0, 0] | 6337 |
| 0° | T60_abs | s | 0 | [0, 0] | 6337 |
| 0° | logspec_mad | log-magnitude | 0 | [0, 0] | 6337 |
| 0° | mag_rel_l2 | relative | 0 | [0, 0] | 6337 |
| 0° | wave_rel_l2 | relative | 0 | [0, 0] | 6337 |
| 0° | wave_mad | amplitude | 0 | [0, 0] | 6337 |
| 45° | EDT | s | 0.0142501 | [0.0136324, 0.0148737] | 6337 |
| 45° | C50 | dB | 0.3809 | [0.365588, 0.396671] | 6337 |
| 45° | T60 | % of T60 | 2.31659 | [2.23289, 2.40344] | 6337 |
| 45° | T60_abs | s | 0.0136796 | [0.0131887, 0.0141893] | 6337 |
| 45° | logspec_mad | log-magnitude | 0.114243 | [0.111295, 0.117297] | 6337 |
| 45° | mag_rel_l2 | relative | 0.147279 | [0.143721, 0.150882] | 6337 |
| 45° | wave_rel_l2 | relative | 0.712583 | [0.705398, 0.719858] | 6337 |
| 45° | wave_mad | amplitude | 0.000696327 | [0.000681783, 0.000710711] | 6337 |
| 90° | EDT | s | 0.00869207 | [0.00825957, 0.0091201] | 6337 |
| 90° | C50 | dB | 0.251351 | [0.240046, 0.262885] | 6337 |
| 90° | T60 | % of T60 | 1.45117 | [1.39826, 1.50392] | 6337 |
| 90° | T60_abs | s | 0.00850444 | [0.0081869, 0.00882435] | 6337 |
| 90° | logspec_mad | log-magnitude | 0.0667636 | [0.0649841, 0.06858] | 6337 |
| 90° | mag_rel_l2 | relative | 0.0884345 | [0.0858682, 0.0910688] | 6337 |
| 90° | wave_rel_l2 | relative | 0.573091 | [0.566709, 0.579634] | 6337 |
| 90° | wave_mad | amplitude | 0.000549608 | [0.000538604, 0.000560567] | 6337 |
| 180° | EDT | s | 0.00781137 | [0.00742689, 0.0082123] | 6337 |
| 180° | C50 | dB | 0.23476 | [0.222483, 0.247699] | 6337 |
| 180° | T60 | % of T60 | 1.22736 | [1.1762, 1.27991] | 6337 |
| 180° | T60_abs | s | 0.00730915 | [0.00698738, 0.00763904] | 6337 |
| 180° | logspec_mad | log-magnitude | 0.060469 | [0.058864, 0.0621072] | 6337 |
| 180° | mag_rel_l2 | relative | 0.0792786 | [0.0768938, 0.081741] | 6337 |
| 180° | wave_rel_l2 | relative | 0.533427 | [0.526857, 0.54007] | 6337 |
| 180° | wave_mad | amplitude | 0.000500946 | [0.000490232, 0.000511504] | 6337 |
| 270° | EDT | s | 0.00852387 | [0.0081152, 0.00894274] | 6337 |
| 270° | C50 | dB | 0.233607 | [0.222962, 0.244277] | 6337 |
| 270° | T60 | % of T60 | 1.42346 | [1.37145, 1.47455] | 6337 |
| 270° | T60_abs | s | 0.00833133 | [0.00801759, 0.00864168] | 6337 |
| 270° | logspec_mad | log-magnitude | 0.0645632 | [0.0628789, 0.0662694] | 6337 |
| 270° | mag_rel_l2 | relative | 0.0854295 | [0.0830579, 0.0878456] | 6337 |
| 270° | wave_rel_l2 | relative | 0.569161 | [0.56287, 0.57551] | 6337 |
| 270° | wave_mad | amplitude | 0.000553629 | [0.00054225, 0.000564938] | 6337 |

### Standalone G on the broader population

The paired table above drops a query whose error is invalid at either angle; this one keeps every query with a finite g_alpha (not the paired mask). It is reported separately and never mixed into the paired delta / G / R.

| angle | metric | unit | G | 95 % CI (query) | n | 95 % CI (room) | rooms | converged |
|---|---|---|---|---|---|---|---|---|
| 0° | EDT | s | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 0° | C50 | dB | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 0° | T60 | % of T60 | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 0° | T60_abs | s | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 45° | EDT | s | 0.0142501 | [0.0136324, 0.0148737] | 6337 | [0.0047354, 0.0236453] | 17 | True |
| 45° | C50 | dB | 0.3809 | [0.365588, 0.396671] | 6337 | [0.12006, 0.59681] | 17 | True |
| 45° | T60 | % of T60 | 2.31659 | [2.23289, 2.40344] | 6337 | [1.32754, 2.92596] | 17 | True |
| 45° | T60_abs | s | 0.0136796 | [0.0131887, 0.0141893] | 6337 | [0.00714228, 0.0178987] | 17 | True |
| 90° | EDT | s | 0.00869207 | [0.00825957, 0.0091201] | 6337 | [0.00277008, 0.0147342] | 17 | True |
| 90° | C50 | dB | 0.251351 | [0.240046, 0.262885] | 6337 | [0.071667, 0.406152] | 17 | True |
| 90° | T60 | % of T60 | 1.45117 | [1.39826, 1.50392] | 6337 | [0.789203, 1.96629] | 17 | True |
| 90° | T60_abs | s | 0.00850444 | [0.0081869, 0.00882435] | 6337 | [0.00403275, 0.0116909] | 17 | True |
| 180° | EDT | s | 0.00781137 | [0.00742689, 0.0082123] | 6337 | [0.00245798, 0.0131786] | 17 | True |
| 180° | C50 | dB | 0.23476 | [0.222483, 0.247699] | 6337 | [0.0631116, 0.391198] | 17 | True |
| 180° | T60 | % of T60 | 1.22736 | [1.1762, 1.27991] | 6337 | [0.604372, 1.78625] | 17 | True |
| 180° | T60_abs | s | 0.00730915 | [0.00698738, 0.00763904] | 6337 | [0.00309367, 0.0107782] | 17 | True |
| 270° | EDT | s | 0.00852387 | [0.0081152, 0.00894274] | 6337 | [0.00281795, 0.0141893] | 17 | True |
| 270° | C50 | dB | 0.233607 | [0.222962, 0.244277] | 6337 | [0.0726499, 0.370205] | 17 | True |
| 270° | T60 | % of T60 | 1.42346 | [1.37145, 1.47455] | 6337 | [0.795106, 1.8797] | 17 | True |
| 270° | T60_abs | s | 0.00833133 | [0.00801759, 0.00864168] | 6337 | [0.00417643, 0.0111429] | 17 | True |

### R2 -- the ratio G / delta, with its status

| angle | metric | R (query) | 95 % CI | status (query) | status (room) | converged | headline |
|---|---|---|---|---|---|---|---|
| 0° | EDT | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 0° | C50 | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 0° | T60 | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 0° | T60_abs | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 45° | EDT | 12.4228 | [8.01077, 27.3799] | defined | defined | True | yes |
| 45° | C50 | 5.98734 | [4.79727, 7.96139] | defined | defined | True | yes |
| 45° | T60 | 14.791 | [8.99519, 41.7619] | defined | denominator uncertain | True | yes |
| 45° | T60_abs | 16.6494 | [10.0677, 49.0401] | defined | denominator uncertain | True | yes |
| 90° | EDT | 9.45602 | [6.4703, 17.7454] | defined | defined | True | yes |
| 90° | C50 | 7.85199 | [5.76314, 12.2155] | defined | denominator uncertain | True | yes |
| 90° | T60 | -34.8828 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 90° | T60_abs | -32.875 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 180° | EDT | 18.5118 | [7.77255, 112.011] | defined | denominator uncertain | True | yes |
| 180° | C50 | 69.7058 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 180° | T60 | -4.02956 | [-5.01685, -3.38114] | improvement | improvement | True | predictions shift while net error improves |
| 180° | T60_abs | -4.5047 | [-5.60246, -3.79069] | improvement | improvement | True | predictions shift while net error improves |
| 270° | EDT | 9.5981 | [6.55975, 18.1814] | defined | denominator uncertain | True | yes |
| 270° | C50 | 10.4453 | [7.01877, 20.3187] | defined | denominator uncertain | True | yes |
| 270° | T60 | -19.8976 | [-103.417, -9.77588] | improvement | denominator uncertain | False | unresolved Monte Carlo uncertainty |
| 270° | T60_abs | -17.922 | [-64.0406, -10.1705] | improvement | denominator uncertain | True | predictions shift while net error improves |

### Room-cluster intervals (secondary)

Room-cluster intervals resample the rooms with replacement and keep every query of a drawn room with its multiplicity (query-weighted, never averaged per room). With so few clusters they are wide, and a room-level ratio inherits its denominator's status: they qualify the query-level intervals, they do not replace them.

| angle | metric | delta CI (room) | G CI (room) | R (room) | 95 % CI (room) | status | rooms | converged (room) |
|---|---|---|---|---|---|---|---|---|
| 0° | EDT | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 0° | C50 | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 0° | T60 | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 0° | T60_abs | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 45° | EDT | [0.00056446, 0.00155395] | [0.0047354, 0.0236453] | 12.4228 | [4.93033, 23.3278] | defined | 17 | True |
| 45° | C50 | [0.0119654, 0.119942] | [0.12006, 0.59681] | 5.98734 | [3.94483, 16.877] | defined | 17 | True |
| 45° | T60 | [-0.0906119, 0.483351] | [1.32754, 2.92596] | 14.791 | [-, -] | denominator uncertain | 17 | True |
| 45° | T60_abs | [-0.000554862, 0.0026348] | [0.00714228, 0.0178987] | 16.6494 | [-, -] | denominator uncertain | 17 | True |
| 90° | EDT | [2.83117e-05, 0.00184944] | [0.00277008, 0.0147342] | 9.45602 | [4.77915, 43.4429] | defined | 17 | True |
| 90° | C50 | [-0.00368255, 0.086214] | [0.071667, 0.406152] | 7.85199 | [-, -] | denominator uncertain | 17 | True |
| 90° | T60 | [-0.139274, 0.0416814] | [0.789203, 1.96629] | -34.8828 | [-, -] | denominator uncertain | 17 | True |
| 90° | T60_abs | [-0.000725837, 0.000129986] | [0.00403275, 0.0116909] | -32.875 | [-, -] | denominator uncertain | 17 | True |
| 180° | EDT | [-8.72701e-06, 0.000818498] | [0.00245798, 0.0131786] | 18.5118 | [-, -] | denominator uncertain | 17 | True |
| 180° | C50 | [-0.0155747, 0.0259848] | [0.0631116, 0.391198] | 69.7058 | [-, -] | denominator uncertain | 17 | True |
| 180° | T60 | [-0.688142, -0.0157597] | [0.604372, 1.78625] | -4.02956 | [-29.7482, -2.44117] | improvement | 17 | True |
| 180° | T60_abs | [-0.00356271, -7.06875e-05] | [0.00309367, 0.0107782] | -4.5047 | [-30.8893, -2.70283] | improvement | 17 | True |
| 270° | EDT | [-9.94416e-05, 0.00193538] | [0.00281795, 0.0141893] | 9.5981 | [-, -] | denominator uncertain | 17 | True |
| 270° | C50 | [-0.00639523, 0.0594822] | [0.0726499, 0.370205] | 10.4453 | [-, -] | denominator uncertain | 17 | True |
| 270° | T60 | [-0.283948, 0.112524] | [0.795106, 1.8797] | -19.8976 | [-, -] | denominator uncertain | 17 | True |
| 270° | T60_abs | [-0.00155174, 0.000402999] | [0.00417643, 0.0111429] | -17.922 | [-, -] | denominator uncertain | 17 | True |

### Controls

| control | k | repeats | max \|dwave\| | max \|dlogspec\| | max \|dacoustic\| | validity mismatches | non-finite | problems | ok |
|---|---|---|---|---|---|---|---|---|---|

