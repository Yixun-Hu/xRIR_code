# exp_10 yaw_pilot — driving queries

## 2026-09-26 (verbatim)

> I need you to run this similar to exp_02 in FLAC for xrir: ## 3. xRIR pilot study
>
> mimic FLAC, doing the rotation experiments of xRIR on AR dataset to finish all the pilot study

**Summary.** The ICLR 2027 paper's pilot-study section has a FLAC part (FLAC repo exp_02 `yaw_noninvariance`, 2026-07-04: released FLAC_EMA checkpoint, AcousticRooms unseen K = 1 split, conditioning yaw-rotated by 90/180/270°; Metric 2 = accuracy vs GT per angle, Metric 1 = prediction shift P_α vs P_0 without GT; figure `FLAC/analysis/pilot_study/plot_pilot_yaw_gaps.py`, commit 435407f: "predictions move ~5x more than net accuracy loses"). Yixun wants the same pilot for xRIR on AcousticRooms so the section is complete for both model families.

**Assumption / hypothesis (Yixun's).** Like FLAC, the released xRIR is not yaw-invariant: rotating the receiver-frame conditioning (depth panorama + coordinates) changes its prediction substantially even where the aggregate accuracy loss is small.

**Why it needs to run.** exp_03 (2026-09-06) already measured Metric 2 for xRIR at many angles (small degradation, peaks at diagonal headings) and a log-spectrogram consistency, but never stored predicted waveforms, so FLAC's Metric 1 (waveform rel-L2 and EDT/C50/T60 gaps with P_0 as the reference) and the degradation-vs-shift figure do not exist for xRIR.
