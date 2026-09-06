# Query log — yaw_rotation_degradation (user: yixun)

## Q1 — 2026-09-05T22:59:09-04:00

**Verbatim:** "@worklog/experiment_SOP.md Use this SOP to conduct an experiment to see whether \"jointly rotating the panorama and source--receiver coordinates in yaw can substantially degrade RIR synthesis quality.\" in xRIR pipeline"

**Summary:** Run an SOP-compliant experiment in this repo that tests whether applying the same yaw rotation to the receiver-centred depth panorama and to the source / reference-source coordinates (a symmetry of the acoustics, so the ground-truth RIR is unchanged) substantially degrades xRIR's predicted RIRs.

**User's assumption / hypothesis:** The xRIR pipeline is not yaw-equivariant, so a jointly rotated (physically identical) scene yields a materially worse RIR prediction. Implicit corollary from the preceding work in this repo: the azimuth-equivariant CylindricalViT geometry encoder should suffer less than the SimpleViT baseline.

**Why the experiment needs to run:** The receiver-frame panorama is the only way room geometry enters xRIR. If a mere change of azimuthal reference frame degrades predictions, the model has learned a canonical-orientation prior rather than geometry, which matters for deployment (real captures have arbitrary heading) and motivates equivariant encoders. The result also quantifies how much of the CylindricalViT gain measured in the earlier backbone comparison comes from robustness to heading.
