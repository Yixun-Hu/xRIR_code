# oriented_cyl (exp_06) — Yixun's driving queries

## Query 1 (2026-09-14, session xrir-code-6e) — verbatim
> Could you please tell me what are current results for "…" [a Codex-generated summary of xRIR §5.5 / Table 2 / supplement §9, §12: sim-to-real evaluation on the four Hearing-Anything-Anywhere rooms with 12-RIR fine-tuning] on xRIR and our CylindricalViT of xRIR on the HAA dataset?

**Summary.** Asked for the current exp_02 numbers. Answered from `ckpt/sim2real/stats.json`: after the paper's two-stage fine-tuning the SimpleViT control is lower on 6/11 room×metric cells (hallway C50 +81 %, EDT +43 %, T60 +55 %; classroom EDT/C50 +9 %; complex EDT +7 %), the cylindrical model on none.

## Query 2 — verbatim
> Could you please give me the reason to explain why in the real-world dataset that our method performs worse than the baseline?

**Summary / finding (recorded in exp_02's notebook, 2026-09-14, `diagnostics_2026-09-14_hallway_side/`).** The HAA loudspeaker is directional and faces −y in every room (hallway: measured C50 ≈ +3 dB at −y vs ≈ −7 dB at +y, 2× energy). The hallway panorama at the speaker is nearly 180°-rotation symmetric (9 % mean pixel difference under a half-turn roll). The cylindrical encoder is azimuth-equivariant and its learned pool leaves the geometry feature nearly invariant, so a mic at +y and its mirror image at −y give the same feature (cosine 0.975; SimpleViT 0.435). The reference attention then selects references by distance regardless of end (85 % of the mixing weight on −y references for +y queries; SimpleViT 24 %) and predicts a −y-like RIR for +y mics (signed spectral C50 error +7.4 dB). With same-side references (oracle) the cylindrical model is as good as or slightly better than the SimpleViT (C50 0.99 vs 1.04 dB; lower log-spectrogram L1).

## Query 3 — verbatim
> So after this analysis, is there a way for you to make cylindricalViT has better results and simplevit of xRIR on real world benchmark?

**Summary.** Three options were laid out: (A) pretraining fix — an orientation-relative azimuth channel in the cylindrical encoder (joint scene + orientation equivariance); (B) fine-tune-only patch (explicit angle-to-axis feature); (C) protocol variant (orientation-aware reference selection). Expected ceiling on HAA as released: parity or a small advantage, because each room has one fixed loudspeaker heading.

## Query 4 (approval of the experiment) — verbatim
> I think physically add an orientation-relative channel to the cylindrical encoder as the next experiment make sense. Please do this appropriately as the next experiment

**User's hypothesis.** Giving the cylindrical encoder the azimuth of each column relative to the source's orientation removes the front/back ambiguity that caused the hallway failure, so the cylindrical xRIR should match or beat the SimpleViT after the paper's real-room fine-tuning while keeping its simulated-room accuracy.

**Why the experiment needs to run.** The mechanism is established descriptively on the realised exp_02 runs; whether a pretrained encoder actually *uses* the channel (absolute azimuth is uninformative for omnidirectional AcousticRooms sources) and whether the fix carries to fine-tuned real-room metrics are empirical questions that need a new pretraining run and the exp_02 protocol.
