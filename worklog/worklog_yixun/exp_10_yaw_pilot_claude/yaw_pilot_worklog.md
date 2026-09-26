# exp_10 yaw_pilot — lab notebook

## 2026-09-26T17:24:59-04:00 — scaffold
- **Goal** — start the xRIR analogue of FLAC exp_02 (yaw non-invariance pilot) per Yixun's 2026-09-26 request.
- **Result** — folder scaffolded; query file written; facts gathered: FLAC exp_02 = released checkpoint, unseen K = 1, α ∈ {0, 90, 180, 270}°, Metric 2 (vs GT) + Metric 1 (P_α vs P_0: waveform rel-L2 / MAD, T60/C50/EDT gaps with P_0 as GT), rot0 determinism control; figure = degradation vs shift bars per angle with a 2σ noise band. xRIR exp_03 has Metric 2 for control / cyl / released at K = 8 on the acoustic grid (k = 0, ±8, ±32, ±64, ±128, 256; P condition, manifest `47637a55…`, gl_seed 0, batch 16, TF32 off) and a log-spec consistency, but no stored waveforms; its K = 1 runs are k = 0 only (no released K = 1 run). Both GPUs held by cylindrical-dinov3 exp_13 tier-L trainings (day 6); asked that session for ETA / co-tenancy; CPU path prepared (runtime device-agnostic copy of `model.xRIR.apply_delay`, as in the qualitative-figure script).
- **Next** — plan → Codex plan review → Coder round (Opus 5, worktree) → Codex code review → runs.
