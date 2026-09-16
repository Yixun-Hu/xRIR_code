# exp_08 — `xRIR_CylInvariant`: end-to-end C16-invariant RIR generation (Stage A: structure)

Worktree `/home/yixunhu/codespace/xRIR_code_wt08`, branch `exp08-invariant-readout`.
Stage A is **structural validation only**: CPU, no training, no GPU, no checkpoint written.

## Goal

Make **one** forward pass of xRIR produce a rotation-invariant RIR, so the model does not need
multi-angle inference and averaging (Route 1, which the user excluded). The physical
transformation is a whole-scene yaw about the receiver's vertical axis: the panorama rolls, and
the xyz channels, the query source and every reference source rotate by `Rz(2*pi*k/512)`. For an
omnidirectional source and a monaural receiver that is a symmetry of the acoustics, so the target
RIR is unchanged and any change in the prediction is model sensitivity.

The guarantee this stage delivers is **C16** — `k` a multiple of 32 columns, i.e. multiples of
22.5 deg — because the azimuth patch is 32 columns wide. Off-lattice angles are measured and
reported under their own label; they are **not** covered by the claim, and the paper must not
present C16 as strict invariance under arbitrary continuous yaw (handoff section 1).

## Pointers

* **Binding design + validation contract**:
  `worklog/worklog_yixun/xrir_invariant_readout_route2_handoff.md` (Codex, 2026-09-16) —
  section 3 implementation, section 4 validation checklist, section 5 training/recipe facts,
  section 6 evaluation framing. Where anything disagrees with it, it wins.
* **What was already known to be broken**: exp_03 (`yaw_rotation_degradation`) measured the
  pinned models' rotation sensitivity on all 6,337 unseen queries under the same reference
  manifest this stage reuses (`manifest_hash 47637a55ccc5`).
* **The readout family this is one member of**: the CylDINO repo's
  `worklog/worklog_yixun/PHYSICS_CONSTRAINTS_synthesis_exp05_to_exp10.md`, constraint **C2**.
  exp_08's `InvariantReadout` is the `phi = identity` case — a *linear* surface integral over
  azimuth. The proposal's nonlinear upgrades (DeepSets `mean_u phi(x_u)`, per-channel deciles,
  low-`m` azimuth Fourier magnitudes, source-steered Fourier pooling) are all exactly invariant
  too and are the registered next move **if** the accuracy cost of this readout turns out to
  matter. CylDINO exp_11 is the cautionary datum: fixed C32-invariant readouts did **not** beat
  the plain token mean at linearly decoding T60/C50/EDT, so a more elaborate invariant pool is
  not automatically better.

## What changed, and only what changed

The pinned forward had exactly two paths without an invariance (handoff section 2):

1. `lin_proj_0` — a learned linear pool over the 256 token **positions**, applied to the
   query-source view, the receiver view and every reference view of the geometry encoder.
2. `src_coord_proj(dist_embedder(...))` — a Fourier embedding of the **raw** query and reference
   xyz, which rotates with the scene.

exp_08 replaces those two and nothing else:

* `InvariantReadout` (`model/xRIR_cyl_invariant.py`) — tokens `[B, 256, C]` on the encoders'
  h-major `(elevation, azimuth)` grid, `m[b,h,c] = mean_w F[b,h,w,c]`, then
  `p[b,c] = sum_h a[h] m[b,h,c] + bias`. Output `[B, 1, C]`, the same shape `lin_proj_0`
  produced, so the fusion head is untouched. The azimuth mean is accumulated in float64, which
  makes the pool **bitwise** invariant to a roll instead of invariant up to a 1-ulp change of
  summation order. Warm start: `a[h] = sum_w W[h,w]`, bias carried.
* `intrinsic_scene_coords` — the coordinate-embedding branch (query **and** every reference) is
  expressed in the scene's own horizontal basis
  `a = (src_x, src_y)/||(src_x, src_y)||`:
  `p -> (a_x p_x + a_y p_y, a_x p_y - a_y p_x, p_z)`. Degenerate rules exactly as specified:
  near-vertical query (horizontal radius `<= 1e-6` m) falls back to the reference with the
  largest horizontal radius, ties to the lowest reference index; if every horizontal component is
  at or below the threshold the horizontal outputs are zero. The geometry encoder keeps its
  original `(location - depth_coord)/5` input — its semantics are **not** swapped.

Everything else in `xRIR.forward` is byte-for-byte the pinned code: reference selection and
order, `shift_and_align`, the reference audio encoder, the geometry/audio fusion, the reference
weights and the spectrogram head. `model/xRIR.py` itself is **not edited**; its hard-coded
`.cuda()` in `apply_delay` is shimmed for CPU validation by `model/exp08_delay.py`, which is
proven equivalent to the pinned function elementwise.

## Deliverables of Stage A

| file | what it is |
|---|---|
| `model/xRIR_cyl_invariant.py` | `InvariantReadout`, the intrinsic coordinate transform, `xRIR_InvariantBase`, `xRIR_CylInvariant` (the method), `xRIR_SimpleInvariant` (the 2x2 control) |
| `model/exp08_factory.py` | `BACKBONES_EXP08` (pinned `simple`/`cylindrical` kept by object identity) + `build_xrir_exp08` |
| `model/exp08_delay.py` | device-preserving `apply_delay` + scoped context manager |
| `tools/exp08_warmstart.py` | `epoch_12.pth` -> invariant model, with a conserved per-tensor accounting |
| `tools/exp08_validate.py` | the section-4 measurements (spectra, waveforms, off-lattice, delay flips, gradients, regression, non-vacuity control) |
| `tests/test_exp08_invariant.py` | the section-4 checklist as pytest, CPU, deterministic |
| `validation_report.md` | every section-4 item with its measured numbers |

## What Stage A does **not** settle

* **Accuracy.** The warm start is an initialisation with a recorded provenance, not a resumption:
  `lin_proj_0` changes shape and `src_coord_proj`'s input *semantics* change even though its shape
  does not. Nothing here predicts T60/C50/EDT or the spectral loss.
* **The 2x2.** `simple_invariant` exists so the readout/coordinate change can be attributed
  separately from the encoder, but it carries **no** invariance guarantee — SimpleViT's tokens are
  not azimuth equivariant. Its measured residual is reported for exactly that reason.
* **The training budget.** A warm-started fine-tune is cheaper but is *not* a same-budget
  comparison against the old 12-epoch models; handoff section 5 requires either a from-scratch
  exp_01-recipe run or a matched extra-budget control, with the endpoint, seeds and
  initialisation source fixed in advance.
* **GPU numerics.** These residuals are CPU/FP32/TF32-off. Handoff section 6 requires the
  invariance to be re-measured on the trained model in the production GPU configuration; a CPU
  smoke test does not substitute for it.
