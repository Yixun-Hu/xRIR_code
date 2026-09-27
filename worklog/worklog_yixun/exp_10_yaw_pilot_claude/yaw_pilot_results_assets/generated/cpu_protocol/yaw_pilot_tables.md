# exp_10 `yaw_pilot` -- descriptive summary

10000 bootstrap resamples, 95 % percentile intervals, seeds [0, 1] (seed 0 reports, seed 1 checks convergence).

The grey band in the figures is the historical baseline evaluation variability (references and phases redrawn), context only.

**How to read the shift.** waveform and acoustic gaps measure the sensitivity of the model-plus-Griffin-Lim pipeline, conditional on the per-query phase seed; the shared phase initialisation removes initialisation noise, but Griffin-Lim is a nonlinear inverse and can amplify a magnitude change. logspec_mad and mag_rel_l2 are the Griffin-Lim-free readout of the same shift, computed on the model's direct log-magnitude output.

**Denominators.** EDT: delta = change in the EDT error against the ground truth, seconds, G = EDT distance from the k = 0 prediction, seconds; C50: delta = change in the C50 error against the ground truth, dB, G = C50 distance from the k = 0 prediction, dB; T60: delta = change in the GT-normalised error, percentage points, G = percent of the baseline prediction's T60; T60_abs: delta = change in |T60(prediction) - T60(GT)|, seconds, G = |T60(P_alpha) - T60(P_0)|, seconds.

## released_k8

`ckpt/exp10/cpu_protocol/released_k8_all` -- 6337 queries, 17 rooms, execution `20260927T004009003970Z-8796b160e3a84e97b43774e2d4aac00e`.

Context band: historical baseline evaluation variability (SimpleViT, exp_04), context only -- C50 0.0053, EDT 0.186, T60 0.009.

### Metric 2 -- accuracy against the ground truth

| angle | metric | unit | mean k=0 | mean k | delta | 95 % CI | n_mask | excluded (0 / alpha / gap) |
|---|---|---|---|---|---|---|---|---|
| 0° | EDT | s | 0.0551329 | 0.0551329 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | C50 | dB | 1.36049 | 1.36049 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | T60 | % of T60 | 9.70305 | 9.70305 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | T60_abs | s | 0.0492949 | 0.0492949 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | log_mse | log-magnitude^2 | 0.438771 | 0.438771 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 0° | loss | test loss | 0.0169724 | 0.0169724 | 0 | [0, 0] | 6337 | 0 / 0 / 0 |
| 45° | EDT | s | 0.0551329 | 0.0545508 | -0.000582133 | [-0.00176336, 0.000593552] | 6337 | 0 / 0 / 0 |
| 45° | C50 | dB | 1.36049 | 1.39359 | 0.0331066 | [0.005695, 0.0602867] | 6337 | 0 / 0 / 0 |
| 45° | T60 | % of T60 | 9.70305 | 10.176 | 0.472903 | [0.323589, 0.620294] | 6337 | 0 / 0 / 0 |
| 45° | T60_abs | s | 0.0492949 | 0.0515312 | 0.00223626 | [0.0014636, 0.00299118] | 6337 | 0 / 0 / 0 |
| 45° | log_mse | log-magnitude^2 | 0.438771 | 0.43691 | -0.00186093 | [-0.00547859, 0.00183776] | 6337 | 0 / 0 / 0 |
| 45° | loss | test loss | 0.0169724 | 0.0174668 | 0.000494363 | [0.000340913, 0.000640385] | 6337 | 0 / 0 / 0 |
| 90° | EDT | s | 0.0551329 | 0.0535444 | -0.00158847 | [-0.00270921, -0.000471027] | 6337 | 0 / 0 / 0 |
| 90° | C50 | dB | 1.36049 | 1.35772 | -0.00276327 | [-0.0304327, 0.0252181] | 6337 | 0 / 0 / 0 |
| 90° | T60 | % of T60 | 9.70305 | 9.96206 | 0.259012 | [0.13125, 0.383001] | 6337 | 0 / 0 / 0 |
| 90° | T60_abs | s | 0.0492949 | 0.0504025 | 0.00110763 | [0.000457712, 0.00174755] | 6337 | 0 / 0 / 0 |
| 90° | log_mse | log-magnitude^2 | 0.438771 | 0.439624 | 0.000853671 | [-0.00286012, 0.00455831] | 6337 | 0 / 0 / 0 |
| 90° | loss | test loss | 0.0169724 | 0.0170292 | 5.67346e-05 | [-8.20941e-05, 0.000192233] | 6337 | 0 / 0 / 0 |
| 180° | EDT | s | 0.0551329 | 0.0539134 | -0.00121953 | [-0.00236673, -5.81833e-05] | 6337 | 0 / 0 / 0 |
| 180° | C50 | dB | 1.36049 | 1.36894 | 0.0084566 | [-0.0191565, 0.0350822] | 6337 | 0 / 0 / 0 |
| 180° | T60 | % of T60 | 9.70305 | 9.46759 | -0.23546 | [-0.36414, -0.108376] | 6337 | 0 / 0 / 0 |
| 180° | T60_abs | s | 0.0492949 | 0.0480651 | -0.00122986 | [-0.00190846, -0.000562646] | 6337 | 0 / 0 / 0 |
| 180° | log_mse | log-magnitude^2 | 0.438771 | 0.447126 | 0.00835577 | [0.00394738, 0.012856] | 6337 | 0 / 0 / 0 |
| 180° | loss | test loss | 0.0169724 | 0.0168637 | -0.000108768 | [-0.000246302, 2.74085e-05] | 6337 | 0 / 0 / 0 |
| 270° | EDT | s | 0.0551329 | 0.0541956 | -0.000937288 | [-0.00197223, 8.93672e-05] | 6337 | 0 / 0 / 0 |
| 270° | C50 | dB | 1.36049 | 1.38216 | 0.021672 | [-0.00337735, 0.0468637] | 6337 | 0 / 0 / 0 |
| 270° | T60 | % of T60 | 9.70305 | 9.8673 | 0.164252 | [0.0406094, 0.291946] | 6337 | 0 / 0 / 0 |
| 270° | T60_abs | s | 0.0492949 | 0.0497598 | 0.000464908 | [-0.000177952, 0.00111106] | 6337 | 0 / 0 / 0 |
| 270° | log_mse | log-magnitude^2 | 0.438771 | 0.444948 | 0.00617754 | [0.00243285, 0.00984295] | 6337 | 0 / 0 / 0 |
| 270° | loss | test loss | 0.0169724 | 0.0169684 | -4.05032e-06 | [-0.000141587, 0.000131442] | 6337 | 0 / 0 / 0 |

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
| 45° | EDT | s | 0.026244 | [0.025024, 0.0274994] | 6337 |
| 45° | C50 | dB | 0.643972 | [0.618331, 0.671485] | 6337 |
| 45° | T60 | % of T60 | 3.78216 | [3.65755, 3.90869] | 6337 |
| 45° | T60_abs | s | 0.0216814 | [0.020961, 0.0224206] | 6337 |
| 45° | logspec_mad | log-magnitude | 0.179965 | [0.176698, 0.183222] | 6337 |
| 45° | mag_rel_l2 | relative | 0.233855 | [0.228989, 0.238755] | 6337 |
| 45° | wave_rel_l2 | relative | 0.877459 | [0.871007, 0.883829] | 6337 |
| 45° | wave_mad | amplitude | 0.000932788 | [0.000912657, 0.000953077] | 6337 |
| 90° | EDT | s | 0.0248851 | [0.0236787, 0.0261399] | 6337 |
| 90° | C50 | dB | 0.606278 | [0.578627, 0.635179] | 6337 |
| 90° | T60 | % of T60 | 3.18609 | [3.07732, 3.29716] | 6337 |
| 90° | T60_abs | s | 0.0181564 | [0.017542, 0.0187919] | 6337 |
| 90° | logspec_mad | log-magnitude | 0.161193 | [0.157856, 0.164662] | 6337 |
| 90° | mag_rel_l2 | relative | 0.207658 | [0.202465, 0.213128] | 6337 |
| 90° | wave_rel_l2 | relative | 0.823666 | [0.816772, 0.830816] | 6337 |
| 90° | wave_mad | amplitude | 0.000853041 | [0.000835068, 0.000871176] | 6337 |
| 180° | EDT | s | 0.0255805 | [0.0243195, 0.0269089] | 6337 |
| 180° | C50 | dB | 0.642582 | [0.614295, 0.670431] | 6337 |
| 180° | T60 | % of T60 | 3.28901 | [3.181, 3.39447] | 6337 |
| 180° | T60_abs | s | 0.019015 | [0.0183654, 0.019661] | 6337 |
| 180° | logspec_mad | log-magnitude | 0.170161 | [0.166534, 0.173815] | 6337 |
| 180° | mag_rel_l2 | relative | 0.219759 | [0.214045, 0.225553] | 6337 |
| 180° | wave_rel_l2 | relative | 0.83421 | [0.826766, 0.84181] | 6337 |
| 180° | wave_mad | amplitude | 0.000862235 | [0.000843898, 0.000880669] | 6337 |
| 270° | EDT | s | 0.0237646 | [0.0226047, 0.0249271] | 6337 |
| 270° | C50 | dB | 0.586018 | [0.559592, 0.613127] | 6337 |
| 270° | T60 | % of T60 | 3.1492 | [3.04158, 3.25885] | 6337 |
| 270° | T60_abs | s | 0.0180061 | [0.0174062, 0.0186205] | 6337 |
| 270° | logspec_mad | log-magnitude | 0.15672 | [0.153452, 0.159923] | 6337 |
| 270° | mag_rel_l2 | relative | 0.201181 | [0.196104, 0.2064] | 6337 |
| 270° | wave_rel_l2 | relative | 0.815908 | [0.808866, 0.823138] | 6337 |
| 270° | wave_mad | amplitude | 0.000848951 | [0.000830696, 0.000867199] | 6337 |

### Standalone G on the broader population

The paired table above drops a query whose error is invalid at either angle; this one keeps every query with a finite g_alpha (not the paired mask). It is reported separately and never mixed into the paired delta / G / R.

| angle | metric | unit | G | 95 % CI (query) | n | 95 % CI (room) | rooms | converged |
|---|---|---|---|---|---|---|---|---|
| 0° | EDT | s | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 0° | C50 | dB | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 0° | T60 | % of T60 | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 0° | T60_abs | s | 0 | [0, 0] | 6337 | [0, 0] | 17 | True |
| 45° | EDT | s | 0.026244 | [0.025024, 0.0274994] | 6337 | [0.0106268, 0.043676] | 17 | True |
| 45° | C50 | dB | 0.643972 | [0.618331, 0.671485] | 6337 | [0.282746, 0.932849] | 17 | True |
| 45° | T60 | % of T60 | 3.78216 | [3.65755, 3.90869] | 6337 | [2.33346, 5.00075] | 17 | True |
| 45° | T60_abs | s | 0.0216814 | [0.020961, 0.0224206] | 6337 | [0.0121309, 0.02895] | 17 | True |
| 90° | EDT | s | 0.0248851 | [0.0236787, 0.0261399] | 6337 | [0.00879416, 0.0429155] | 17 | True |
| 90° | C50 | dB | 0.606278 | [0.578627, 0.635179] | 6337 | [0.24711, 0.923156] | 17 | True |
| 90° | T60 | % of T60 | 3.18609 | [3.07732, 3.29716] | 6337 | [1.97771, 3.9479] | 17 | True |
| 90° | T60_abs | s | 0.0181564 | [0.017542, 0.0187919] | 6337 | [0.00995361, 0.0232962] | 17 | True |
| 180° | EDT | s | 0.0255805 | [0.0243195, 0.0269089] | 6337 | [0.00950026, 0.043517] | 17 | True |
| 180° | C50 | dB | 0.642582 | [0.614295, 0.670431] | 6337 | [0.26441, 0.954033] | 17 | True |
| 180° | T60 | % of T60 | 3.28901 | [3.181, 3.39447] | 6337 | [1.99266, 4.10051] | 17 | True |
| 180° | T60_abs | s | 0.019015 | [0.0183654, 0.019661] | 6337 | [0.0103198, 0.0243471] | 17 | True |
| 270° | EDT | s | 0.0237646 | [0.0226047, 0.0249271] | 6337 | [0.00901303, 0.0399064] | 17 | True |
| 270° | C50 | dB | 0.586018 | [0.559592, 0.613127] | 6337 | [0.252678, 0.86549] | 17 | True |
| 270° | T60 | % of T60 | 3.1492 | [3.04158, 3.25885] | 6337 | [1.96877, 3.89383] | 17 | True |
| 270° | T60_abs | s | 0.0180061 | [0.0174062, 0.0186205] | 6337 | [0.0102733, 0.0228431] | 17 | True |

### R2 -- the ratio G / delta, with its status

| angle | metric | R (query) | 95 % CI | status (query) | status (room) | converged | headline |
|---|---|---|---|---|---|---|---|
| 0° | EDT | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 0° | C50 | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 0° | T60 | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 0° | T60_abs | - | [-, -] | undefined | undefined | True | observed delta = 0 |
| 45° | EDT | -45.0825 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 45° | C50 | 19.4515 | [10.2675, 84.8631] | defined | denominator uncertain | True | yes |
| 45° | T60 | 7.99774 | [6.13336, 11.6268] | defined | defined | True | yes |
| 45° | T60_abs | 9.69541 | [7.26008, 14.7849] | defined | defined | True | yes |
| 90° | EDT | -15.6661 | [-50.1295, -9.12878] | improvement | denominator uncertain | True | predictions shift while net error improves |
| 90° | C50 | -219.406 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 90° | T60 | 12.3009 | [8.30722, 24.2949] | defined | denominator uncertain | True | yes |
| 90° | T60_abs | 16.3922 | [10.3711, 39.5546] | defined | denominator uncertain | True | yes |
| 180° | EDT | -20.9757 | [-122.273, -9.65135] | improvement | denominator uncertain | True | predictions shift while net error improves |
| 180° | C50 | 75.9859 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 180° | T60 | -13.9685 | [-30.0854, -9.03377] | improvement | denominator uncertain | True | predictions shift while net error improves |
| 180° | T60_abs | -15.4611 | [-33.751, -10.0139] | improvement | denominator uncertain | True | predictions shift while net error improves |
| 270° | EDT | -25.3547 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 270° | C50 | 27.0404 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |
| 270° | T60 | 19.173 | [10.6275, 71.524] | defined | denominator uncertain | True | yes |
| 270° | T60_abs | 38.7304 | [-, -] | denominator uncertain | denominator uncertain | True | denominator uncertain |

### Room-cluster intervals (secondary)

Room-cluster intervals resample the rooms with replacement and keep every query of a drawn room with its multiplicity (query-weighted, never averaged per room). With so few clusters they are wide, and a room-level ratio inherits its denominator's status: they qualify the query-level intervals, they do not replace them.

| angle | metric | delta CI (room) | G CI (room) | R (room) | 95 % CI (room) | status | rooms | converged (room) |
|---|---|---|---|---|---|---|---|---|
| 0° | EDT | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 0° | C50 | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 0° | T60 | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 0° | T60_abs | [0, 0] | [0, 0] | - | [-, -] | undefined | 17 | True |
| 45° | EDT | [-0.00271882, 0.00169704] | [0.0106268, 0.043676] | -45.0825 | [-, -] | denominator uncertain | 17 | True |
| 45° | C50 | [-0.00923617, 0.0748301] | [0.282746, 0.932849] | 19.4515 | [-, -] | denominator uncertain | 17 | True |
| 45° | T60 | [0.0783826, 0.825439] | [2.33346, 5.00075] | 7.99774 | [4.54714, 30.5302] | defined | 17 | True |
| 45° | T60_abs | [0.000148746, 0.00413357] | [0.0121309, 0.02895] | 9.69541 | [4.91638, 52.9921] | defined | 17 | True |
| 90° | EDT | [-0.00329687, 0.000371221] | [0.00879416, 0.0429155] | -15.6661 | [-, -] | denominator uncertain | 17 | True |
| 90° | C50 | [-0.0299885, 0.0194234] | [0.24711, 0.923156] | -219.406 | [-, -] | denominator uncertain | 17 | True |
| 90° | T60 | [-0.0909282, 0.651961] | [1.97771, 3.9479] | 12.3009 | [-, -] | denominator uncertain | 17 | True |
| 90° | T60_abs | [-0.000444337, 0.00268947] | [0.00995361, 0.0232962] | 16.3922 | [-, -] | denominator uncertain | 17 | True |
| 180° | EDT | [-0.00350851, 0.000534129] | [0.00950026, 0.043517] | -20.9757 | [-, -] | denominator uncertain | 17 | True |
| 180° | C50 | [-0.0534199, 0.0713381] | [0.26441, 0.954033] | 75.9859 | [-, -] | denominator uncertain | 17 | True |
| 180° | T60 | [-0.610268, 0.0631243] | [1.99266, 4.10051] | -13.9685 | [-, -] | denominator uncertain | 17 | True |
| 180° | T60_abs | [-0.00298728, 0.000287155] | [0.0103198, 0.0243471] | -15.4611 | [-, -] | denominator uncertain | 17 | True |
| 270° | EDT | [-0.00182714, 4.17315e-05] | [0.00901303, 0.0399064] | -25.3547 | [-, -] | denominator uncertain | 17 | True |
| 270° | C50 | [-0.0345241, 0.0833263] | [0.252678, 0.86549] | 27.0404 | [-, -] | denominator uncertain | 17 | True |
| 270° | T60 | [-0.084007, 0.374523] | [1.96877, 3.89383] | 19.173 | [-, -] | denominator uncertain | 17 | True |
| 270° | T60_abs | [-0.000684842, 0.00157696] | [0.0102733, 0.0228431] | 38.7304 | [-, -] | denominator uncertain | 17 | True |

### Controls

| control | k | repeats | max \|dwave\| | max \|dlogspec\| | max \|dacoustic\| | validity mismatches | non-finite | problems | ok |
|---|---|---|---|---|---|---|---|---|---|

