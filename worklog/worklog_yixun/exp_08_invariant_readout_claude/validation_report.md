# exp_08 validation report — Stage A (structure), CPU, no training

**Round 3** — after Codex review round 2. Round 1 raised two blockers (§11, §12); round 2
accepted the B2 *implementation* but re-opened **B1** as a P1 (the round-2 fallback was still
discontinuous, and its threshold was an absolute length in a scene tens of metres across) and
required three reporting corrections to B2. Both are addressed below and every number has been
re-measured.

Every number below was produced by `tools/exp08_validate.py` in this worktree and is stored,
per angle and per sample, in `exp08_validation.json`; the console transcript is
`exp08_validation.log`. `tests/test_exp08_invariant.py` recomputes the same quantities
independently and asserts on them (**216 passed, 0 failed, 480 s**; the pre-existing suite is
unaffected — `tests/test_exp06_*`, `test_yaw_rotation`, `test_per_sample_metrics`,
`test_reference_manifest`, `test_eval_yaw_rotation`: 102 passed, 18 skipped (GPU-only)).

## 0. Identity of the run

| | |
|---|---|
| worktree / branch | `/home/yixunhu/codespace/xRIR_code_wt08`, `exp08-invariant-readout` |
| python / torch | 3.8.20, 2.0.1+cu117 |
| device | CPU only, `CUDA_VISIBLE_DEVICES=""`, `torch.set_num_threads(8)` |
| precision | FP32, `allow_tf32` **False** for both cuBLAS and cuDNN, models in `eval()` |
| seeds | `torch.manual_seed(0)` / `np.random.seed(0)`; Griffin-Lim seed 0, per query via `tools.per_sample_metrics.sample_seed`; delay-flip ensemble seed 12345 |
| checkpoints (read-only, main checkout) | `ckpt/xRIR_cyl_8_shot/epoch_12.pth`, `ckpt/xRIR_simple_8_shot/epoch_12.pth` |
| reference manifest | `ckpt/yaw_rotation/reference_manifest.json`, `manifest_hash = 47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d`, file sha256 `5eaea3727dd840b1cb7cd77cb821ff955004c05b0f09c74fc39849e59ba5da2a` — both equal to the values the exp_03 provenance log pinned |
| real scenes | 3 queries, 3 different room **types**: `Apartments/Apartments_idx_42/S001_R001`, `Auditorium/Auditorium_idx_1/S001_R001`, `Bathrooms/Bathrooms_idx_14/S001_R001` (manifest indices 0, 500, 1500), K = 8 |
| runtime | validation tool 656 s, exp_08 suite 480 s |

The criterion is the handoff's: **relative Frobenius residual of the whole predicted
log-spectrum against `k = 0`, target `<= 1e-5`**, with the absolute residual and the
denominator `||out_0||_F` reported next to it.

## 1. The C16 guarantee, stated precisely

> With `CylindricalViT` + the azimuth-invariant readout + intrinsic coordinate conditioning, one
> forward pass of `xRIR_CylInvariant` is invariant under the C16 scene yaw (`k` a multiple of 32
> panorama columns, i.e. multiples of 22.5°), to the float32 input-rounding floor of the network
> — measured worst case **`2.80e-7`** relative on the battery and **`2.94e-7`** on the worst
> adversarial geometry, against a `1e-5` target.
>
> The geometry-to-basis map is **branch-free**: no threshold selects a case and no comparison
> selects a reference, so there is no configuration at which the representation can jump. The
> only remaining exactness loss is the **integer direct-path delay**: geometries whose pre-round
> value `(||src|| - ||ref||) / 343 * 22050` falls inside a narrow band around a `.5` tie can
> have that delay move by one sample under a rotation that preserves the distances, shifting a
> reference RIR by one sample and breaking exactness on the true end-to-end (condition E) path.
> That band has **finite width**, not zero: under the sampled ensemble of §12 it catches
> `1.04e-5` of (scene, reference, angle) triples and `223` of `1,000,000` scenes. Across all
> real validation scenes it was not hit (**0 delay flips** at every angle). Off-lattice angles
> are not covered by this statement at all.

The delay exception is a property of the discretisation, not of the readout or the coordinates:
with the alignment held fixed (condition P), the adversarial delay scene is invariant to
`1.25e-7`.

## 2. Verdict table

| handoff §4 item | outcome | worst measured |
|---|---|---|
| 1. readout invariant to azimuth rolls, still sensitive to content/elevation | **PASS** | **bitwise exact** (`torch.equal`) on random tokens for all 15 rolls; rel `7.6e-7` on real warm-started encoder tokens |
| 2. intrinsic coordinates: joint-yaw invariance, sensitivity, every degenerate branch | **PASS** | rel `1.2e-7` in FP32; `1.8e-15` with FP64 inputs (the maths is exact); all 3 branches, both thresholds, tie-free fallback |
| 3. whole model, 15 C16 angles, random-init **and** warm-started, real + synthetic, batch > 1, reduced K | **PASS** | **`2.80e-7`** worst over all 8 cases x 15 angles (target `1e-5`); **`2.94e-7`** worst over the 4 adversarial geometries |
| 4. full spectrum **and** seeded-Griffin-Lim waveform, true end-to-end path | **PASS** (waveform: see §6) | spectrum `1.18e-7` on real scenes; waveform rel-L2 `9.57e-5`, *below* the same-size non-rotational control `1.90e-4` |
| 5. real forward + backward + optimizer step, finite gradients reaching the new parameters | **PASS** | loss `1.1927` finite; 212/216 trainable tensors get a non-zero gradient and move |
| 6. pinned entries unchanged | **PASS** | `simple` and `cylindrical` **bit-identical** (`max|d| = 0.0`) and `is`-identical classes |
| **B1** degenerate basis, rounds 1 **and** 2 | **FIXED (branch-free)** | equal-radius tie `1.478e-1` -> `2.07e-7`; sum-exactly-zero `2.270e-1` -> `2.14e-7`; near-cancellation `5.151e-2` -> `2.94e-7`; production path **bitwise unchanged** |
| **B2** integer-delay boundary | **NARROWED + QUALIFIED** | flip rate `1.335e-5` -> `1.040e-5` (1.284x, **3.64σ** on paired per-scene counts); the exception is stated in §1 and measured in §12 |
| off-lattice (diagnostic, no pass/fail) | reported separately | rel `5.2e-4 … 1.8e-3` |
| non-vacuity control | **the harness detects non-invariance** | pinned `cylindrical` scores `4.61e-2`, i.e. `4600x` the target |

## 3. Item 1 — the invariant readout

* **Random non-constant tokens** (`[3, 256, 64]`, std ≈ 1): for every azimuth roll
  `m = 1 … 15` the tokens change and `InvariantReadout` returns a **bitwise identical**
  tensor (`torch.equal`). This is a consequence of accumulating the azimuth mean in float64:
  a naive float32 mean of the same tokens differs by up to `2.4e-7` between roll orders,
  which would have leaked a spurious "rotation effect" at exactly the scale being measured.
* **Real tokens from the warm-started encoder**: `src_proj(source_network((src − depth)/5))`
  on the three real panoramas, at `k = 0` and at all 15 C16 angles. Tokens move by up to
  `1.46` (max elementwise) and, after undoing the token-column roll, agree to `5.4e-6` — i.e.
  the encoder really is azimuth-equivariant. The readout output moves by rel Frobenius
  `7.6e-7`.
* **Sensitivity** (so the above is not vacuous): perturbing one token's content changes the
  output; permuting the 16 elevation rows changes the output. Both asserted.
* **Warm start**: `a[h] = sum_w W[h,w]` exactly (`rtol = atol = 0`), bias carried bitwise; and
  when the old pool happens to be azimuth-constant the new readout reproduces it to `1e-5`.

## 4. Item 2 — the intrinsic coordinate representation

The basis rule, after the round-3 B1 fix (§11) — one expression, no conditionals:

```
w = smoothstep((||q_h||/||q|| - lo) / (hi - lo))          lo = 1e-3, hi = 1e-2 (dimensionless)
a = w * q_h/max(||q_h||, lo*||q||) + (1-w) * sum_i p_h_i / sum_i ||p_h_i||
```

`||a|| <= 1` always (triangle inequality), `a` is exactly rotation-equivariant, and it is C2 in
the inputs wherever the references are not all on the vertical axis.

* **Joint yaw**: over all 15 C16 angles *and* the off-lattice angles (the transform is exact
  for any yaw), query and reference intrinsic coordinates move by at most `9.5e-7` absolute on
  a coordinate scale of `8.26` m → **rel `1.2e-7`**.
* **Exactness**: repeating the same sweep with float64 inputs gives `1.8e-15`. The FP32 number
  is therefore the rounding of the rotated *inputs*, not of the transform.
* **Sensitivity**: moving one reference to a genuinely different azimuth changes the
  representation by `> 1e-2`; rotating **only** the query (not a rigid motion) also changes it
  by `> 1e-2`, so the transform has not simply discarded azimuth.
* **Both sides transformed**: for `src = (3, 4, 1.25)` and `ref = (−4, 3, −0.5)` the query maps
  to `(5, 0, 1.25)` and the reference to `(0, 5, −0.5)` — the reference keeps its +90° relative
  azimuth.
* **Degenerate regime**, all asserted: a vertical query uses the reference aggregate, which is
  yaw-invariant across all 15 angles and cannot be moved by reordering the references
  (`atol 1e-12`); exactly cancelling references give `a = 0`, exactly zero horizontal output and
  bitwise-identical coordinates at every angle; all references on the axis hits the single
  remaining `R == 0` guard, whose output *is* the continuous limit; a zero-length query (source
  coincident with the receiver) is finite.
* **No thresholds remain in the fallback.** The primary/fallback transition is the smoothstep
  band `[1e-3, 1e-2]` on the *dimensionless* ratio `||q_h||/||q||`, so it means the same thing
  at 0.5 m and at 40 m — a test sweeps three ranges and asserts the blend agrees to `1e-9`. The
  blend saturates **exactly** at 0 and 1 outside the band (asserted), which is what keeps the
  production path bit-exact. An invalid band is refused.
* **Smoothness is measured, not asserted.** Two 400-configuration sweeps walk across the old
  thresholds and the new band; the largest single step is reported against the same sweep under
  the superseded round-2 rule (§11).

## 5. Item 3 — the whole model at all 15 C16 angles

Condition **E** is the true end-to-end path: the scene is rotated and the *entire* forward runs,
so `shift_and_align` recomputes `||src||`, `||ref||`, the direct-path gains and the **integer**
delays from the rotated coordinates. Condition **P** pins the `k = 0` alignment
(`tools.yaw_rotation.fixed_alignment`) and exists only to separate alignment rounding from the
rest. A separate test proves the recomputation happened: `shift_and_align` on rotated
coordinates is **not** bitwise equal to the `k = 0` alignment while staying within float32
rounding of it.

| case | condition | worst rel | at k (deg) | abs Frobenius | denom (F-norm of out_0) | max elementwise | worst single row |
|---|---|---|---|---|---|---|---|
| `random_init/real` | E | 1.129e-07 | 160 (112.5) | 7.226e-06 | 63.98 | 1.788e-07 | 1.459e-07 |
| `random_init/real` | P | 1.053e-07 | 160 (112.5) | 6.738e-06 | 63.98 | 1.490e-07 | 1.386e-07 |
| `random_init/synthA` | E | 1.288e-07 | 288 (202.5) | 3.940e-06 | 30.6 | 1.658e-07 | 1.488e-07 |
| `random_init/synthA` | P | 9.796e-08 | 32 (22.5) | 2.997e-06 | 30.6 | 1.192e-07 | 1.046e-07 |
| `random_init/synthB` | E | 1.220e-07 | 352 (247.5) | 3.587e-06 | 29.4 | 4.619e-07 | 1.465e-07 |
| `random_init/synthB` | P | 9.679e-08 | 224 (157.5) | 2.846e-06 | 29.4 | 1.192e-07 | 1.033e-07 |
| `random_init/synthK3` | E | 1.179e-07 | 160 (112.5) | 3.430e-06 | 29.09 | 4.023e-07 | 1.295e-07 |
| `random_init/synthK3` | P | 1.150e-07 | 448 (315.0) | 3.347e-06 | 29.09 | 1.192e-07 | 1.321e-07 |
| `warm_start/real` | E | 1.181e-07 | 96 (67.5) | 1.903e-04 | 1611 | 3.815e-06 | 1.647e-07 |
| `warm_start/real` | P | 1.205e-07 | 288 (202.5) | 1.940e-04 | 1611 | 2.861e-06 | 1.715e-07 |
| `warm_start/synthA` | E | 2.798e-07 | 384 (270.0) | 2.852e-04 | 1019 | 1.383e-05 | 3.627e-07 |
| `warm_start/synthA` | P | 2.798e-07 | 384 (270.0) | 2.852e-04 | 1019 | 1.383e-05 | 3.627e-07 |
| `warm_start/synthB` | E | 1.626e-07 | 128 (90.0) | 1.659e-04 | 1020 | 5.007e-06 | 1.774e-07 |
| `warm_start/synthB` | P | 1.626e-07 | 128 (90.0) | 1.659e-04 | 1020 | 5.007e-06 | 1.774e-07 |
| `warm_start/synthK3` | E | 9.880e-08 | 256 (180.0) | 9.892e-05 | 1001 | 4.530e-06 | 1.161e-07 |
| `warm_start/synthK3` | P | 9.880e-08 | 256 (180.0) | 9.892e-05 | 1001 | 4.530e-06 | 1.161e-07 |

Per-angle detail for the headline case (`warm_start/real`, condition E):

| k | deg | rel | abs | max elementwise | delay flips |
|---|---|---|---|---|---|
| 32 | 22.5 | 8.998e-08 | 1.449e-04 | 1.907e-06 | 0 |
| 64 | 45.0 | 1.003e-07 | 1.616e-04 | 9.060e-06 | 0 |
| 96 | 67.5 | 1.181e-07 | 1.903e-04 | 3.815e-06 | 0 |
| 128 | 90.0 | 9.521e-08 | 1.533e-04 | 2.384e-06 | 0 |
| 160 | 112.5 | 9.358e-08 | 1.507e-04 | 2.384e-06 | 0 |
| 192 | 135.0 | 1.077e-07 | 1.734e-04 | 9.060e-06 | 0 |
| 224 | 157.5 | 1.039e-07 | 1.673e-04 | 3.815e-06 | 0 |
| 256 | 180.0 | 1.142e-07 | 1.839e-04 | 2.861e-06 | 0 |
| 288 | 202.5 | 8.526e-08 | 1.373e-04 | 2.861e-06 | 0 |
| 320 | 225.0 | 9.615e-08 | 1.549e-04 | 1.001e-05 | 0 |
| 352 | 247.5 | 1.043e-07 | 1.680e-04 | 4.292e-06 | 0 |
| 384 | 270.0 | 8.473e-08 | 1.365e-04 | 2.384e-06 | 0 |
| 416 | 292.5 | 7.997e-08 | 1.288e-04 | 2.384e-06 | 0 |
| 448 | 315.0 | 1.012e-07 | 1.630e-04 | 8.583e-06 | 0 |
| 480 | 337.5 | 8.108e-08 | 1.306e-04 | 3.815e-06 | 0 |

All-angle summary, condition E:

| case | denom | min rel | median rel | max rel | max abs | delay flips total |
|---|---|---|---|---|---|---|
| `random_init/real` | 63.98 | 5.940e-08 | 1.037e-07 | 1.129e-07 | 7.226e-06 | 0 |
| `random_init/synthA` | 30.6 | 6.276e-08 | 1.046e-07 | 1.288e-07 | 3.940e-06 | 0 |
| `random_init/synthB` | 29.4 | 6.162e-08 | 1.158e-07 | 1.220e-07 | 3.587e-06 | 0 |
| `random_init/synthK3` | 29.09 | 8.148e-08 | 1.093e-07 | 1.179e-07 | 3.430e-06 | 0 |
| `warm_start/real` | 1611 | 7.997e-08 | 9.615e-08 | 1.181e-07 | 1.903e-04 | 0 |
| `warm_start/synthA` | 1019 | 1.213e-07 | 1.641e-07 | 2.798e-07 | 2.852e-04 | 0 |
| `warm_start/synthB` | 1020 | 8.978e-08 | 1.267e-07 | 1.626e-07 | 1.659e-04 | 0 |
| `warm_start/synthK3` | 1001 | 4.623e-08 | 8.747e-08 | 9.880e-08 | 9.892e-05 | 0 |

* Every one of the 8 cases x 15 angles is `<= 2.80e-7`, i.e. **at least 35x below the `1e-5`
  target**, and the same holds per *row* (worst single sample `3.63e-7`).
* **Delay flips: 0** at every angle, in every case, counted with the float64 reduction the
  invariant arms actually use (`tools.exp08_validate.delay_flip_counts_for` selects the
  reduction from the model's class, so an invariant arm is never audited against a boundary it
  no longer has).
* The `random_init` cases matter because invariance must not depend on the weights: they pass
  with the same residual as the warm-started model, on a denominator ~25x smaller.

### Why the residual is `~1e-7` and not zero

| k | input rel (depth) | input rel (src) | input rel (refs) | output rel | output abs | denom |
|---|---|---|---|---|---|---|
| 32 | 6.845e-08 | 6.739e-09 | 5.293e-08 | **8.219e-08** | 1.324e-04 | 1611 |
| 96 | 6.853e-08 | 3.014e-09 | 8.586e-08 | **8.438e-08** | 1.359e-04 | 1611 |
| 224 | 6.853e-08 | 3.014e-09 | 8.586e-08 | **8.438e-08** | 1.359e-04 | 1611 |

Rotating the scene and rotating it **straight back** is the identity in exact arithmetic, so the
returned scene is geometrically *unrotated* and differs from the original only by the float32
rounding a rotation introduces. Pushing that through the model moves the output by `8.2e-8`,
the same order as a real C16 rotation. The C16 residual is therefore the float32 input-rounding
noise floor of the network, not a property of the architecture.

(At `k = 128` and `k = 256` the float32 `Rz` is exact, so this control degenerates to zero and
says nothing there; the angles used are 32, 96 and 224. It is also a *lower* bound: it re-rounds
the coordinates but leaves the panorama column/gauge pairing alone, which a real roll also
perturbs.)

## 6. Item 4 — waveforms

Seeded Griffin-Lim (32 iterations) on the predicted magnitudes `exp(out) − 1e-8`, with a
per-query **yaw-independent** phase seed, at `k = 0` and at every angle, on the three real
scenes.

rows 45 (15 angles x 3 queries); rel-L2 min 7.200e-06, median 4.575e-05, max 9.565e-05

| query | worst rel-L2 | at k | abs L2 | denom (L2 of w_0) |
|---|---|---|---|---|
| `Apartments/Apartments_idx_42/S001_R001_hybrid_IR.wav` | 5.602e-05 | 96 | 2.349e-05 | 0.4194 |
| `Auditorium/Auditorium_idx_1/S001_R001_hybrid_IR.wav` | 9.565e-05 | 128 | 8.337e-06 | 0.08716 |
| `Bathrooms/Bathrooms_idx_14/S001_R001_hybrid_IR.wav` | 9.447e-05 | 256 | 7.404e-05 | 0.7838 |

control (same relative size 1.181e-07, non-rotational perturbation, identical seeds): rel-L2 ['1.898e-04', '1.027e-04', '5.192e-05'] ; Griffin-Lim deterministic at a fixed seed: True

**Reading this honestly.** The brief's target was "waveform residual of the same order as the
spectral residual". It is not: the spectrum moves by `1.2e-7` and the waveform by up to
`9.6e-5`. That amplification is **Griffin-Lim's**, not the model's, and the control above proves
it: perturbing the `k = 0` magnitudes by iid noise of exactly the same relative size, with **no
rotation at all** and identical seeds, moves the waveform by up to `1.90e-4` — *more* than the
rotation does. The inversion was also checked to be bitwise reproducible at a fixed seed.

## 7. Off-lattice diagnostic — **not** covered by the C16 claim

| k | deg | rel | abs | max elementwise | delay flips |
|---|---|---|---|---|---|
| 1 | 0.70 | 5.223e-04 | 8.412e-01 | 3.079e-02 | 0 |
| 5 | 3.52 | 1.253e-03 | 2.019e+00 | 6.864e-02 | 0 |
| 8 | 5.62 | 1.282e-03 | 2.065e+00 | 5.888e-02 | 0 |
| 16 | 11.25 | 1.702e-03 | 2.741e+00 | 7.502e-02 | 0 |
| 17 | 11.95 | 1.793e-03 | 2.887e+00 | 7.516e-02 | 0 |
| 31 | 21.80 | 6.747e-04 | 1.087e+00 | 4.333e-02 | 0 |

`k` values that are not multiples of 32, i.e. rotations the 32-column azimuth patch cannot
represent as a token permutation. Three to four orders of magnitude above the C16 residual;
reported under their own label and never folded into the claim. `k = 1` (0.70°) is *lower* than
`k = 16` (11.25°): the error grows with the distance to the nearest patch boundary, not with the
angle — the signature of a patch-discretisation effect.

## 8. Controls

* pinned `cylindrical` (non-vacuity control): rel over the 15 C16 angles min 2.016e-02, median 3.724e-02, **max 4.610e-02** (at k=192, abs 2.965e+00, denom 64.32)
* `simple_invariant` (2x2 control arm, no guarantee): rel over the 15 C16 angles min 1.722e-03, median 4.283e-03, **max 5.654e-03** (at k=288, abs 9.160e+00, denom 1620)

* The **pinned `cylindrical`** entry is the non-vacuity control: same harness, same real batch,
  same 15 angles, `2.0e-2 … 4.6e-2` — five orders above the invariant model and 4600x the
  target. A harness that could not see that would prove nothing.
* **`simple_invariant`** is the 2x2 control arm. Same invariant readout, same intrinsic
  coordinates, same float64 alignment, SimpleViT encoder — `1.7e-3 … 5.7e-3`. The handoff's
  prediction confirmed, and the number needed to attribute the result to the *combination*.

## 9. Warm-start parameter accounting

checkpoint: `/home/yixunhu/codespace/xRIR_code/ckpt/xRIR_cyl_8_shot/epoch_12.pth`

| bucket | tensors | params |
|---|---|---|
| loaded | 276 | 32132944 |
| consumed | 2 | 257 |
| dropped_shape | 0 | 0 |
| dropped_absent | 0 | 0 |
| new_warm | 2 | 17 |
| new_random | 0 | 0 |
| checkpoint total | 278 | 32133201 |
| new model total | 278 | 32132961 |

consumed: ['lin_proj_0.proj.weight', 'lin_proj_0.proj.bias']
new_warm: [('invariant_readout.weight', [16]), ('invariant_readout.bias', [1])]
strict_load missing: ['invariant_readout.bias', 'invariant_readout.weight']

* **0** shape mismatches, **0** checkpoint tensors absent, **0** new tensors left at random
  init. The buckets are asserted to partition both key sets.
* `lin_proj_0.proj.{weight,bias}` (257 params) is **consumed**: `a[h] = sum_w W[h,w]` into
  `invariant_readout.weight [16]`, bias into `invariant_readout.bias [1]`. A test re-reads the
  checkpoint and asserts the readout weight is bitwise the folded pool.
* `lin_proj_0` is removed; net 32,132,961 vs the checkpoint's 32,133,201 (−240).
* `simple_invariant` from `xRIR_simple_8_shot/epoch_12.pth`: 264 loaded / 2 consumed / 2
  warm-started / 0 dropped / 0 random.
* **Caveat (handoff §3.3):** `src_coord_proj` keeps its shape but its input *semantics* change.
  The old optimizer state cannot be restored. This is a new run with a recorded initialisation
  source, not a resumption.

## 10. Items 5 and 6 — gradient step and pinned regression

loss 1.192709 (stft 0.972052 + decay 0.220658); all grads finite True; 212/216 trainable tensors got a non-zero gradient; 212 changed after the step

| parameter | grad L2 | max abs delta after one AdamW step |
|---|---|---|
| `invariant_readout.weight` | 1.2412e-02 | 1.0000e-03 |
| `invariant_readout.bias` | 5.4275e-02 | 9.9999e-04 |
| `src_coord_proj.proj.weight` | 5.9735e-02 | 1.0000e-03 |
| `src_coord_proj.proj.bias` | 1.8292e-02 | 1.0000e-03 |
| `src_proj.proj.weight` | 1.2880e-01 | 1.0000e-03 |
| `lin_proj_1.proj.weight` | 6.9531e-05 | 9.9407e-04 |
| `lin_proj_2.proj.weight` | 1.2867e+00 | 1.0000e-03 |
| `time_proj.proj.weight` | 1.1443e-01 | 1.0000e-03 |
| `audio_enc.cnn.conv1.weight` | 3.5017e-01 | 1.0000e-03 |
| `source_network.to_patch_embedding.2.weight` | 1.9830e-01 | 9.9999e-04 |

no gradient (pre-existing dead heads in the pinned xRIR, not introduced by exp_08): ['audio_enc.cnn.fc_backup.bias', 'audio_enc.cnn.fc_backup.weight', 'source_proj.proj.bias', 'source_proj.proj.weight']

One forward + backward + `AdamW(lr=1e-3, weight_decay=1e-4)` step on the real batch using
`train_xRIR_backbone.compute_loss`'s objective with only the `.cuda()` transfers removed. All
gradients finite; the new readout and the coordinate branch both receive non-zero gradients and
move; the pinned parameters keep training. The four silent tensors are `source_proj.*` and
`audio_enc.cnn.fc_backup.*` — pre-existing dead heads in `model/xRIR.py`, pinned by a test.

```json
{
 "simple": {
  "class_identity": true,
  "bitwise_equal": true,
  "max_abs_diff": 0.0
 },
 "cylindrical": {
  "class_identity": true,
  "bitwise_equal": true,
  "max_abs_diff": 0.0
 }
}
```

`build_xrir_exp08("simple"/"cylindrical", 8)` produce predictions **bitwise identical**
(`max|d| = 0.0`) to `xRIR(...)` / `xRIR_Cyl(...)` on the fixed real batch; the registry entries
are the same *objects*; `model.xRIR_cyl.BACKBONES` is not mutated; and `model/xRIR.py` still
contains its hard-coded `torch.zeros_like(signal).cuda()` — asserted, because the CPU shim must
remain a shim. The float64 alignment override lives on the exp_08 invariant arms only, so the
pinned arms are untouched by it.

## 11. Blocker B1 — three rules, two of which had a discontinuity

A discontinuity in the basis *is* a discontinuity in the model's output under rotation, because
float32 rounding of the rotated coordinates is enough to step across it. Two versions of this
rule shipped with one, and both were caught by review rather than by me.

| version | fallback rule | how it broke |
|---|---|---|
| round 1 | the reference with the largest horizontal radius, ties to the lowest index | a **selection**: three references at radius exactly 5, rounding chose the winner — coords jumped **4.4 m**, spectrum `1.478e-1` |
| round 2 | `s / \|s\|` with `s` the vector sum, guarded by `\|s\| > 1e-6` m | the sum removed the selection, but (i) normalising by its *own* length is ill-conditioned exactly where the sum is small, and (ii) the guard is a hard branch on an **absolute length** — meaningless for a scene 20–36 m across. Sum exactly zero: rounding pushed `\|s\|` across the guard, branch flipped, coords jumped **35.7 m**, spectrum `2.270e-1`. Near-cancellation without a branch flip: conditioning alone moved coords **1.5 m**, spectrum `5.151e-2` |
| **round 3** | `a = s / R` with `R = sum_i \|p_h_i\|`, blended into the primary by a smoothstep on `\|q_h\|/\|q\|` | — |

**Why `R` and not `\|s\|`.** `R` is rotation *invariant* and `s` rotation *equivariant*, so `a`
is exactly equivariant. `R >= \|s\|` by the triangle inequality, so `\|a\| <= 1` and the
division can never amplify: rounding perturbs `s` by `O(1e-7 R)`, hence `a` by `O(1e-7)`
**absolute** and the coordinates by `O(1e-7 ||p_h||)` — machine level at any scene scale. It is
smooth wherever `R > 0`, so round 2's "all horizontal components vanish" branch is now simply
the continuous limit `a -> 0`; the only guard left is an exact `R == 0`, and it returns precisely
that limit.

**The primary branch had the same defect** — a hard `||q_h|| > eps` test — so it received the
same treatment: a C2 smoothstep blend on the *dimensionless* `||q_h||/||q||`, band `[1e-3, 1e-2]`.
The blend saturates exactly at 0 and 1, so outside the band the arithmetic is unchanged.

**What it means physically.** When the references' directions cancel there is genuinely no
horizontal direction the scene distinguishes, and `||a|| < 1` makes the representation
*attenuate* its horizontal components in proportion to how ambiguous they are, reaching zero
exactly when they cancel. Heights pass through untouched and the attenuation is itself rotation
invariant. That is the honest encoding of "this scene has no preferred azimuth".

| sweep family | configurations | range | largest single step, **round 3** | largest single step, superseded round-2 rule |
|---|---|---|---|---|
| `cancellation` | 400 | [1e-10, 1e-02] | **1.131e-05** | 1.000e+00 |
| `query_blend` | 400 | [1e-08, 1e-01] | **5.622e-02** | 1.231e+00 |

| case | blend | \|a\| | scene scale | coord drift, **round 3** | coord drift, round-2 rule | worst E | worst P |
|---|---|---|---|---|---|---|---|
| `equal_radius_tie` | 0.0 | 0.628932 | 5.0 m | **4.768e-07 m** | 4.768e-07 m | 2.073e-07 | 1.426e-07 |
| `sum_exactly_zero` | 0.0 | 0.000000 | 32.0 m | **8.983e-07 m** | 3.569e+01 m | 2.140e-07 | 2.134e-07 |
| `near_cancellation` | 0.0 | 0.000000 | 8.0 m | **1.647e-07 m** | 1.491e+00 m | 2.942e-07 | 2.835e-07 |
| `all_horizontal_zero` | 0.0 | 0.000000 | 5.0 m | **0.000e+00 m** | 0.000e+00 m | 1.220e-07 | 1.220e-07 |

production path untouched: blend in [1.0, 1.0] on the real battery, basis bitwise equal to `q_h/||q_h||`: **True**, smallest horizontal fraction 0.5957 against a band top of 1e-02 (a factor of 60)

Every adversarial geometry now sits at the same float32 floor as an ordinary scene, in **both**
conditions, and the two sweeps show the jump is gone rather than merely moved: the largest
single step across 400 configurations is `1.1e-5` (cancellation) and `5.6e-2` (query blend),
against `1.00` and `1.23` for the same sweeps under the round-2 rule.

**The production path is bit-exact.** Every real query is 60x above the top of the blend band, so
`w = 1` exactly, the basis is bitwise `q_h/||q_h||`, and two tests assert it: one on the basis
and coordinates, one **end to end** — the round-2 coordinate rule is patched into the module the
forward pass resolves it from and the two predictions are compared with `torch.equal`. The whole
battery in §5 is numerically identical to round 2, angle for angle.

**Tests** (`tests/test_exp08_invariant.py`): the four adversarial geometries at coordinate level
over all 15 angles and again through the full model in both conditions
(`test_adversarial_basis_coordinates_are_rotation_invariant`,
`test_adversarial_basis_full_model_is_invariant`); both smoothness sweeps with the round-2
contrast built into the assertion (`test_basis_has_no_branch_on_the_reference_aggregate`,
`..._on_the_query_horizontal_fraction`); `||a|| <= 1` over 500 random reference sets;
order-independence; exact cancellation and the `R == 0` guard; a zero-length query; exact blend
saturation outside the band; scale-relativity across 0.5 / 1 / 40 m; and the two bitwise
regressions above.

## 12. Blocker B2 — the integer-delay rounding boundary

**What was wrong.** The round-1 test only asserted that the delay-flip count was `>= 0`, and
every battery scene recorded 0 — so it could never fail. Codex's case: query
`(1.054444432258606, 0, 0)`, references `(1, 0, 0)`, `(0.7, 1.1, 0.2)`, `(-1.5, 0.3, -0.1)`. The
first reference's pre-round delay is `3.4999992` samples — `7.8e-7` from the `3.5` tie — so at
`k = 32` it rounds to 4 instead of 3, shifting that reference RIR by one sample. E residual
`2.006e-2` against a P residual of `8.5e-8`.

**Fix (a) — stabilise.** `xRIR_InvariantBase.shift_and_align` reduces the distances in
**float64** before rounding (and the direct-path gain likewise, cast back), documented as part of
the invariant arms' architecture rather than as a test shim. `apply_delay` is still resolved from
`model.xRIR`'s globals at call time, so the CPU shim reaches it and `model/xRIR.py` stays
untouched. Review round 2 accepted this implementation.

*Changes nothing on the battery*: bitwise identical integer delays and aligned reference audio
agreeing to rel `9.1e-8` — a precision change, not a behaviour change.

*The boundary narrows — measured, with two corrections.* My first metric was the largest
pre-round deviation a rotation induces, and it made float64 look **worse** (`6.10e-5` vs
`6.97e-5` samples). That metric is unreadable: the float32 value is itself quantised to float32,
so two rotated scenes often collapse onto the same representable number and the deviation comes
out optically smaller than the truth. It is superseded by a direct flip count, and the stale
"roughly 3x" claim it produced has been removed from the code docstring as well.

| reduction | flips | comparisons | rate | scenes with >=1 flip |
|---|---|---|---|---|
| float32 | 1602 | 120000000 | 1.335e-05 | 297 / 1000000 |
| float64 | 1248 | 120000000 | 1.040e-05 | 223 / 1000000 |

shrink factor **1.284**; paired per-scene difference 354 +- 97.18 -> **3.64 sigma**. (An independent-Poisson treatment would claim 6.63 sigma; flips correlate within a scene -- 199 scenes favour float32, 117 favour float64 -- so the paired figure is the honest one.)

Ensemble: 1000000 scenes x 8 references x 15 angles, seed 12345, positions uniform in a +-6 m box with heights scaled by 0.25. The rate is specific to this sampled distribution, not a universal constant.

The second correction is statistical, and it cuts my own claim down. Flips **correlate within a
scene** — one scene sitting on a tie flips at most of the 15 angles — so treating the two totals
as independent Poisson counts overstates the significance. Recomputed on the **paired per-scene
difference**, the result is **3.64σ**, not the 6.63σ an independent treatment would claim. The
test now asserts on the paired figure and additionally asserts that it is the more conservative
of the two, so the loose version cannot creep back.

**Fix (b) — qualify.** The guarantee in §1 no longer calls the exception "measure-zero" or places
it "within float rounding": it is a band of **finite width**, and the frequency quoted for it is
a property of the sampled ensemble (uniform positions in a ±6 m box, heights scaled by 0.25) —
not a universal constant. The adversarial case is reported in full rather than smoothed away:

pre-round max deviation (NOT the operative measure -- float32 is masked by its own quantisation):
  real: float32 6.104e-05 samples, float64 6.975e-05
  codex_case: float32 7.629e-06 samples, float64 9.113e-06

adversarial 3.5-tie scene: pre-round [3.499999217, -17.012968558, -30.762420547] samples -> delays [3, -17, -31]
  12 of 15 angles flip exactly one delay by exactly one sample: k = [32, 64, 96, 160, 192, 224, 288, 320, 352, 416, 448, 480]
  3 angles do not flip (Rz exact in float32): k = [128, 256, 384], rel there 8.112e-08 .. 9.092e-08
  condition E at the flipped angles: 2.006061e-02 .. 2.006063e-02
  condition P (alignment held at k=0) at every angle: max 1.254e-07

Float64 does **not** push this particular case off the boundary; it is engineered to sit `7.8e-7`
samples from the tie. What the numbers localise is *where* exactness is lost: condition P is
invariant to `1.25e-7` at every angle, so the readout, the intrinsic coordinates and the encoder
are exactly as invariant here as everywhere else, and the entire `2.0e-2` is the one-sample shift
of one reference RIR. The three non-flipping angles (90°, 180°, 270°, where float32 `Rz` is
exact) sit at `9.1e-8`, the normal floor.

**Test.** `test_codex_delay_boundary_case_is_either_invariant_or_a_documented_flip` asserts a
disjunction per angle — no flip ⇒ the E residual meets `1e-5`; a flip ⇒ exactly **one** delay
moving by exactly **one** sample, with condition P still meeting `1e-5` — then compares the
observed flip set against a pinned constant, so neither a disappearing nor a new flip can pass
silently.

**What this does not claim.** The C16 guarantee is not exact on the E path for every possible
geometry, and this report does not say it is. It is exact to the float32 floor outside a
finite-width band whose frequency is now measured under a stated distribution, and the band was
not hit by any real validation scene. A future run that does hit it will show a non-zero
delay-flip count in the per-angle tables, which are recorded for exactly that reason.


## 13. Deviations from the brief and from the review, and why

1. **Manifest hash wording.** The brief asks that the manifest's *sha256* equal `47637a55…`;
   that value is the manifest's **content** hash (`manifest_hash`, which excludes `ir_root`) and
   the file's sha256 is `5eaea3727dd8…`. Both verified, both matching the exp_03 record.
2. **File naming.** Repo casing was followed: `model/xRIR_cyl_invariant.py` (the brief wrote
   `xrir_cyl_invariant.py`, mis-casing the exp_06 file the same way). Factory at
   `model/exp08_factory.py`, which the brief allows explicitly.
3. **Test-node count went 222 -> 216 while coverage went up.** Round 2 parametrised one
   adversarial geometry over 15 angles (15 nodes) and had two 5-way threshold sweeps; round 3
   covers **four** adversarial geometries, each looping all 15 angles inside the test, plus the
   full model in both conditions, two smoothness sweeps, two bitwise regressions and five new
   degenerate-case tests. Fewer nodes, roughly four times the geometry.
4. **Review B2 says "in `xRIR_CylInvariant` ONLY"; the override is on `xRIR_InvariantBase`**, so
   both invariant arms get it. Reason: `simple_invariant` exists to isolate the *encoder* in the
   2x2 (handoff §5). If only the cylindrical arm reduced in float64, the two invariant arms would
   differ in two places at once and the attribution would be muddied. The constraint the review
   was protecting — that the **pinned** `simple` / `cylindrical` keep bit-identical behaviour and
   `model/xRIR.py` stays untouched — is satisfied either way, and §10 re-verifies it. Flagged
   here rather than assumed.
5. **Review B2 says "the boundary width shrinks accordingly".** It shrinks by **1.284x**, not by
   a large factor; my first way of measuring it showed the opposite sign because of a
   quantisation artefact; and the significance I first quoted (6.6σ) was inflated by treating
   correlated flips as independent. All three corrections are in §12, not just the flattering
   numbers.
6. **Waveform target.** "Same order as the spectral residual" is not attainable through
   Griffin-Lim by *any* model (§6), so the test asserts the control comparison instead. Raw
   numbers reported.
7. **Readout-on-real-tokens tolerance** is the handoff's `1e-5` rather than a self-imposed
   `1e-6`: the measured value is `7.6e-7` and drifts tens of percent with the thread count. The
   closed-form coordinate transform keeps the tighter `1e-6` (measured `1.2e-7`).
8. **`lin_proj_0` deleted** from the invariant models — fully consumed by the readout, recorded
   in the accounting as `consumed`.
9. **Unrequested additions** kept because the claim is weaker without them: the non-vacuity
   control, the input-rounding noise floor control, the Griffin-Lim conditioning control, the
   encoder-equivariance check inside the readout test, and the flip-rate ensemble.

## 14. Ready for training? — yes, with one gap to build

**Nothing blocks Stage B on correctness grounds.**

1. **There is no exp_08 training entry point yet.** `train_xRIR_backbone.py` takes `--backbone`
   from `model.xRIR_cyl.BACKBONES` (`simple` / `cylindrical` only). exp_06 solved this with
   `tools/exp06_train.py`, which owns a parser over `BACKBONES_EXP06` and calls the pinned
   trainer's `seed_everything` / `train_epoch` / `test_epoch` / `save_checkpoint` without
   modifying it. exp_08 needs the same wrapper over `BACKBONES_EXP08`, plus a `--warm-start`
   flag routing through `tools.exp08_warmstart.build_warm_started` and writing the accounting
   into the run directory.
2. **The CPU delay shim is not needed for training.** `model/exp08_delay.py` exists only so
   `shift_and_align` can run with no GPU visible; on a GPU the pinned
   `torch.zeros_like(signal).cuda()` is a no-op and `model/xRIR.py` is used unmodified. Do not
   install `device_preserving_delay()` in a training run. (The float64 distance reduction is
   *not* part of the shim — it is in the model and travels with it to the GPU.)
3. **GPU scheduling is not free.** Both visible GPUs were at 100% utilisation with other
   experiments and were not touched. The ~31 h figure is a rough historical estimate, not a
   measured cost for this model.

And two things Stage A deliberately does **not** license:

* **No accuracy claim.** The warm start changes `src_coord_proj`'s input semantics; nothing here
  predicts T60 / C50 / EDT or the spectral loss. Acceptance criteria must be fixed before the
  evaluation, and the recipe/checkpoint must not be selected on the 6,337-query test set.
* **No substitute for GPU validation.** These residuals are CPU/FP32/TF32-off. The invariance
  must be re-measured on the trained model in the production configuration — including the
  delay-flip count over the whole 6,337-query split, which is the population §12's measure-zero
  statement should ultimately be checked against.
