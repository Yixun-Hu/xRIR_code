# exp_08 validation report — Stage A (structure), CPU, no training

Every number below was produced by `tools/exp08_validate.py` in this worktree and is stored,
per angle and per sample, in `exp08_validation.json`; the console transcript is
`exp08_validation.log`. `tests/test_exp08_invariant.py` recomputes the same quantities
independently and asserts on them (**195 passed, 0 failed, 218 s**; the pre-existing
suite is unaffected -- `tests/test_exp06_*`, `test_yaw_rotation`, `test_per_sample_metrics`,
`test_reference_manifest`, `test_eval_yaw_rotation`: 102 passed, 18 skipped (GPU-only)).

## 0. Identity of the run

| | |
|---|---|
| worktree / branch | `/home/yixunhu/codespace/xRIR_code_wt08`, `exp08-invariant-readout` |
| python / torch | 3.8.20, 2.0.1+cu117 |
| device | CPU only, `CUDA_VISIBLE_DEVICES=""`, `torch.set_num_threads(8)` |
| precision | FP32, `allow_tf32` **False** for both cuBLAS and cuDNN, models in `eval()` |
| seeds | `torch.manual_seed(0)` / `np.random.seed(0)`; Griffin-Lim seed 0, per query via `tools.per_sample_metrics.sample_seed` |
| checkpoints (read-only, main checkout) | `ckpt/xRIR_cyl_8_shot/epoch_12.pth`, `ckpt/xRIR_simple_8_shot/epoch_12.pth` |
| reference manifest | `ckpt/yaw_rotation/reference_manifest.json`, `manifest_hash = 47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d`, file sha256 `5eaea3727dd840b1cb7cd77cb821ff955004c05b0f09c74fc39849e59ba5da2a` — both equal to the values the exp_03 provenance log pinned |
| real scenes | 3 queries, 3 different room **types**: `Apartments/Apartments_idx_42/S001_R001`, `Auditorium/Auditorium_idx_1/S001_R001`, `Bathrooms/Bathrooms_idx_14/S001_R001` (manifest indices 0, 500, 1500), K = 8 |

The criterion is the handoff's: **relative Frobenius residual of the whole predicted
log-spectrum against `k = 0`, target `<= 1e-5`**, with the absolute residual and the
denominator `||out_0||_F` reported next to it.

## 1. Verdict table

| handoff §4 item | outcome | worst measured |
|---|---|---|
| 1. readout invariant to azimuth rolls, still sensitive to content/elevation | **PASS** | **bitwise exact** (`torch.equal`) on random tokens for all 15 rolls; rel `7.6e-7` on real warm-started encoder tokens |
| 2. intrinsic coordinates: joint-yaw invariance, sensitivity, every degenerate branch | **PASS** | rel `1.2e-7` in FP32; `1.8e-15` with FP64 inputs (the maths is exact); all 3 branches + threshold boundary covered |
| 3. whole model, 15 C16 angles, random-init **and** warm-started, real + synthetic, batch > 1, reduced K | **PASS** | **`2.15e-7`** worst over all 8 cases x 15 angles (target `1e-5`) |
| 4. full spectrum **and** seeded-Griffin-Lim waveform, true end-to-end path | **PASS** (waveform: see §5) | spectrum `1.39e-7` on real scenes; waveform rel-L2 `9.84e-5`, which is *below* the same-size non-rotational control `2.22e-4` |
| 5. real forward + backward + optimizer step, finite gradients reaching the new parameters | **PASS** | loss `1.1927` finite; 212/216 trainable tensors get a non-zero gradient and move |
| 6. pinned entries unchanged | **PASS** | `simple` and `cylindrical` **bit-identical** (`max|d| = 0.0`) and `is`-identical classes |
| off-lattice (diagnostic, no pass/fail) | reported separately | rel `5.2e-4 … 1.8e-3` — three to four orders above the C16 residual, exactly as expected |
| non-vacuity control | **the harness detects non-invariance** | pinned `cylindrical` scores `4.61e-2`, i.e. `4600x` the target |

## 2. Item 1 — the invariant readout

* **Random non-constant tokens** (`[3, 256, 64]`, std ≈ 1): for every azimuth roll
  `m = 1 … 15` the tokens change and `InvariantReadout` returns a **bitwise identical**
  tensor (`torch.equal`). This is a consequence of accumulating the azimuth mean in float64:
  a naive float32 mean of the same tokens differs by up to `2.4e-7` between roll orders,
  which would have leaked a spurious "rotation effect" at exactly the scale being measured.
* **Real tokens from the warm-started encoder**: `src_proj(source_network((src − depth)/5))`
  on the three real panoramas, at `k = 0` and at all 15 C16 angles. Tokens move by up to
  `1.46` (max elementwise) and, after undoing the token-column roll, agree to `5.4e-6` — i.e.
  the encoder really is azimuth-equivariant. The readout output moves by rel Frobenius
  `7.6e-7` (max elementwise `9.5e-7`, `||base||_F` scale `1.12` max element).
* **Sensitivity** (so the above is not vacuous): perturbing one token's content changes the
  output; permuting the 16 elevation rows changes the output. Both asserted.
* **Warm start**: `a[h] = sum_w W[h,w]` exactly (`rtol = atol = 0`), bias carried bitwise; and
  when the old pool happens to be azimuth-constant the new readout reproduces it to `1e-5`.

## 3. Item 2 — the intrinsic coordinate representation

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
* **Degenerate branches**, all asserted:
  * query on the vertical axis → basis from the reference with the largest horizontal radius
    (`mode = BASIS_REFERENCE`), and that fallback is itself yaw-invariant over all 15 angles
    (residual `<= 1e-5`);
  * tie in the horizontal radii → **lowest reference index** wins, and reordering the
    references changes the winner (so the rule is about the index, not the geometry);
  * every horizontal component at or below the threshold → basis `(0, 0)`, horizontal outputs
    exactly zero, `z` untouched, no NaN.
* **Threshold**: `DEFAULT_BASIS_EPS = 1e-6` m on the horizontal radius, applied with a strict
  `>` in float64. Parametrised boundary test: `10 eps` and `1.5 eps` → query basis; **exactly
  `eps`**, `0.5 eps` and `0.0` → fallback. `eps` is a constructor argument
  (`build_xrir_exp08(..., basis_eps=...)`) and a non-positive value is refused.
* Two knife edges are documented rather than hidden: a query radius within float rounding of
  `eps`, and two references whose horizontal radii agree to within float rounding (the arg-max
  could flip). Both are measure-zero and numerical, not architectural.

## 4. Item 3 — the whole model at all 15 C16 angles

Condition **E** is the true end-to-end path: the scene is rotated and the *entire* forward runs,
so `shift_and_align` recomputes `||src||`, `||ref||`, the direct-path gains and the **integer**
delays from the rotated coordinates. Condition **P** pins the `k = 0` alignment
(`tools.yaw_rotation.fixed_alignment`) and exists only to separate alignment rounding from the
rest — it is *not* what the claim rests on. A separate test proves the recomputation happened:
`shift_and_align` on rotated coordinates is **not** bitwise equal to the `k = 0` alignment
(so no cache was reused) while staying within float32 rounding of it.

| case | condition | worst rel | at k (deg) | abs Frobenius | denom (F-norm of out_0) | max elementwise | worst single row |
|---|---|---|---|---|---|---|---|
| `random_init/real` | E | 1.312e-07 | 288 (202.5) | 8.392e-06 | 63.98 | 2.384e-07 | 1.531e-07 |
| `random_init/real` | P | 1.130e-07 | 160 (112.5) | 7.229e-06 | 63.98 | 1.788e-07 | 1.545e-07 |
| `random_init/synthA` | E | 1.659e-07 | 448 (315.0) | 5.076e-06 | 30.6 | 1.788e-07 | 1.925e-07 |
| `random_init/synthA` | P | 1.016e-07 | 160 (112.5) | 3.109e-06 | 30.6 | 1.192e-07 | 1.131e-07 |
| `random_init/synthB` | E | 1.228e-07 | 192 (135.0) | 3.611e-06 | 29.4 | 1.937e-07 | 1.460e-07 |
| `random_init/synthB` | P | 9.654e-08 | 224 (157.5) | 2.838e-06 | 29.4 | 1.192e-07 | 1.035e-07 |
| `random_init/synthK3` | E | 1.315e-07 | 64 (45.0) | 3.826e-06 | 29.09 | 2.682e-07 | 1.517e-07 |
| `random_init/synthK3` | P | 1.076e-07 | 160 (112.5) | 3.130e-06 | 29.09 | 8.941e-08 | 1.203e-07 |
| `warm_start/real` | E | 1.389e-07 | 288 (202.5) | 2.236e-04 | 1611 | 8.106e-06 | 1.583e-07 |
| `warm_start/real` | P | 1.335e-07 | 288 (202.5) | 2.150e-04 | 1611 | 2.861e-06 | 1.552e-07 |
| `warm_start/synthA` | E | 2.150e-07 | 96 (67.5) | 2.191e-04 | 1019 | 3.004e-05 | 2.556e-07 |
| `warm_start/synthA` | P | 2.034e-07 | 320 (225.0) | 2.073e-04 | 1019 | 7.629e-06 | 2.535e-07 |
| `warm_start/synthB` | E | 1.760e-07 | 128 (90.0) | 1.795e-04 | 1020 | 4.768e-06 | 2.006e-07 |
| `warm_start/synthB` | P | 1.760e-07 | 128 (90.0) | 1.795e-04 | 1020 | 4.768e-06 | 2.006e-07 |
| `warm_start/synthK3` | E | 1.142e-07 | 224 (157.5) | 1.144e-04 | 1001 | 2.670e-05 | 1.231e-07 |
| `warm_start/synthK3` | P | 1.005e-07 | 192 (135.0) | 1.006e-04 | 1001 | 5.007e-06 | 1.211e-07 |

Per-angle detail for the headline case (`warm_start/real`, condition E):

| k | deg | rel | abs | max elementwise | delay flips |
|---|---|---|---|---|---|
| 32 | 22.5 | 1.094e-07 | 1.762e-04 | 7.629e-06 | 0 |
| 64 | 45.0 | 9.396e-08 | 1.513e-04 | 7.153e-06 | 0 |
| 96 | 67.5 | 9.689e-08 | 1.561e-04 | 8.106e-06 | 0 |
| 128 | 90.0 | 8.629e-08 | 1.390e-04 | 2.384e-06 | 0 |
| 160 | 112.5 | 9.026e-08 | 1.454e-04 | 6.199e-06 | 0 |
| 192 | 135.0 | 8.402e-08 | 1.353e-04 | 7.153e-06 | 0 |
| 224 | 157.5 | 1.038e-07 | 1.671e-04 | 2.861e-06 | 0 |
| 256 | 180.0 | 1.052e-07 | 1.694e-04 | 2.384e-06 | 0 |
| 288 | 202.5 | 1.389e-07 | 2.236e-04 | 8.106e-06 | 0 |
| 320 | 225.0 | 9.208e-08 | 1.483e-04 | 8.106e-06 | 0 |
| 352 | 247.5 | 1.382e-07 | 2.225e-04 | 7.629e-06 | 0 |
| 384 | 270.0 | 1.257e-07 | 2.025e-04 | 2.861e-06 | 0 |
| 416 | 292.5 | 1.146e-07 | 1.846e-04 | 8.106e-06 | 0 |
| 448 | 315.0 | 8.375e-08 | 1.349e-04 | 7.153e-06 | 0 |
| 480 | 337.5 | 8.371e-08 | 1.348e-04 | 2.861e-06 | 0 |

All-angle summary, condition E:

| case | denom | min rel | median rel | max rel | max abs | delay flips total |
|---|---|---|---|---|---|---|
| `random_init/real` | 63.98 | 6.224e-08 | 1.055e-07 | 1.312e-07 | 8.392e-06 | 0 |
| `random_init/synthA` | 30.6 | 6.412e-08 | 1.423e-07 | 1.659e-07 | 5.076e-06 | 0 |
| `random_init/synthB` | 29.4 | 6.121e-08 | 1.060e-07 | 1.228e-07 | 3.611e-06 | 0 |
| `random_init/synthK3` | 29.09 | 7.934e-08 | 1.255e-07 | 1.315e-07 | 3.826e-06 | 0 |
| `warm_start/real` | 1611 | 8.371e-08 | 9.689e-08 | 1.389e-07 | 2.236e-04 | 0 |
| `warm_start/synthA` | 1019 | 1.043e-07 | 1.354e-07 | 2.150e-07 | 2.191e-04 | 0 |
| `warm_start/synthB` | 1020 | 9.066e-08 | 1.270e-07 | 1.760e-07 | 1.795e-04 | 0 |
| `warm_start/synthK3` | 1001 | 7.275e-08 | 9.140e-08 | 1.142e-07 | 1.144e-04 | 0 |

* Every one of the 8 cases x 15 angles is `<= 2.15e-7`, i.e. **at least 46x below the `1e-5`
  target**, and the same holds per *row* (worst single sample `2.56e-7`) so nothing hides
  behind a batch Frobenius norm.
* **Delay flips: 0** at every angle, in every case — no integer direct-path delay moved under
  any C16 rotation on these scenes. (The rounding boundary in `shift_and_align` is real; on
  this sample it simply was not hit. The count is recorded per angle so a future run that does
  hit it is visible immediately.)
* Condition P is slightly tighter than E (e.g. `1.335e-7` vs `1.389e-7` on `warm_start/real`),
  which is the expected sign: E additionally carries the float32 re-computation of the
  direct-path gains.
* The `random_init` cases matter because invariance must not depend on the weights: they pass
  with the same residual as the warm-started model, on a denominator ~25x smaller.

### Why the residual is `~1e-7` and not zero

| k | input rel (depth) | input rel (src) | input rel (refs) | output rel | output abs | denom |
|---|---|---|---|---|---|---|
| 32 | 6.845e-08 | 6.739e-09 | 5.293e-08 | **1.202e-07** | 1.936e-04 | 1611 |
| 96 | 6.853e-08 | 3.014e-09 | 8.586e-08 | **9.885e-08** | 1.592e-04 | 1611 |
| 224 | 6.853e-08 | 3.014e-09 | 8.586e-08 | **9.885e-08** | 1.592e-04 | 1611 |

Rotating the scene and rotating it **straight back** is the identity in exact arithmetic, so the
returned scene is geometrically *unrotated* and differs from the original only by the float32
rounding a rotation introduces. Pushing that through the model moves the output by the *same*
`~1.2e-7` as a real C16 rotation does. The C16 residual is therefore the float32 input-rounding
noise floor of the network, not a property of the architecture — no amount of training or
readout redesign would remove it, and running in float64 would.

(At `k = 128` and `k = 256` the float32 `Rz` is exact, so this control degenerates to zero and
says nothing there; the angles used are 32, 96 and 224. The control is also a *lower* bound: it
re-rounds the coordinates but leaves the panorama column/gauge pairing alone, which a real roll
also perturbs.)

## 5. Item 4 — waveforms

Seeded Griffin-Lim (`tools.per_sample_metrics.griffin_lim_seeded`, 32 iterations) on the
predicted magnitudes `exp(out) − 1e-8`, with a per-query **yaw-independent** phase seed, at
`k = 0` and at every angle, on the three real scenes.

rows 45 (15 angles x 3 queries); rel-L2 min 8.850e-06, median 4.062e-05, max 9.843e-05

| query | worst rel-L2 | at k | abs L2 | denom (L2 of w_0) |
|---|---|---|---|---|
| `Apartments/Apartments_idx_42/S001_R001_hybrid_IR.wav` | 6.193e-05 | 192 | 2.597e-05 | 0.4194 |
| `Auditorium/Auditorium_idx_1/S001_R001_hybrid_IR.wav` | 9.843e-05 | 224 | 8.579e-06 | 0.08716 |
| `Bathrooms/Bathrooms_idx_14/S001_R001_hybrid_IR.wav` | 6.860e-05 | 64 | 5.376e-05 | 0.7838 |

control (same relative size 1.389e-07, non-rotational perturbation, identical seeds): rel-L2 ['2.220e-04', '1.264e-04', '7.644e-05'] ; Griffin-Lim deterministic at a fixed seed: True

**Reading this honestly.** The brief's target was "waveform residual of the same order as the
spectral residual". It is not: the spectrum moves by `1.4e-7` and the waveform by up to
`9.8e-5`, ~700x more. That amplification is **Griffin-Lim's**, not the model's, and the control
above proves it: perturbing the `k = 0` magnitudes by iid noise of exactly the same relative
size, with **no rotation at all** and identical seeds, moves the waveform by up to `2.22e-4` —
*more* than the rotation does. Griffin-Lim is a non-convex iterative phase retrieval; a fixed
phase initialisation pins its randomness but not its condition number. The inversion was also
checked to be bitwise reproducible at a fixed seed.

So the defensible statement is: *the rotation moves the reconstructed waveform no more than an
equally sized non-rotational perturbation does, and both are at the `1e-4` relative level that
Griffin-Lim's conditioning imposes on any `1e-7` spectral change.* The test asserts exactly
that (rotation `<= 10x` control, and `< 1e-2` absolute), not a target that no model could meet.

## 6. Off-lattice diagnostic — **not** covered by the C16 claim

| k | deg | rel | abs | max elementwise | delay flips |
|---|---|---|---|---|---|
| 1 | 0.70 | 5.223e-04 | 8.412e-01 | 3.079e-02 | 0 |
| 5 | 3.52 | 1.253e-03 | 2.019e+00 | 6.864e-02 | 0 |
| 8 | 5.62 | 1.282e-03 | 2.065e+00 | 5.888e-02 | 0 |
| 16 | 11.25 | 1.702e-03 | 2.741e+00 | 7.502e-02 | 0 |
| 17 | 11.95 | 1.793e-03 | 2.887e+00 | 7.517e-02 | 0 |
| 31 | 21.80 | 6.747e-04 | 1.087e+00 | 4.333e-02 | 0 |

These are `k` values that are not multiples of 32, i.e. rotations the 32-column azimuth patch
cannot represent as a token permutation. They are three to four orders of magnitude above the
C16 residual and are reported here under their own label; they must never be folded into the
invariance claim or plotted with the same marker (handoff §4, last line, and §6).

Note `k = 1` (0.70°) is *lower* than `k = 16` (11.25°): the off-lattice error grows with the
distance to the nearest patch boundary, not with the angle, which is exactly the signature of a
patch-discretisation effect rather than a modelling error.

## 7. Controls

* pinned `cylindrical` (non-vacuity control): rel over the 15 C16 angles min 2.016e-02, median 3.724e-02, **max 4.610e-02** (at k=192, abs 2.965e+00, denom 64.32)
* `simple_invariant` (2x2 control arm, no guarantee): rel over the 15 C16 angles min 1.722e-03, median 4.283e-03, **max 5.654e-03** (at k=288, abs 9.160e+00, denom 1620)

* The **pinned `cylindrical`** entry is the non-vacuity control: the same harness, the same real
  batch, the same 15 angles. It scores `2.0e-2 … 4.6e-2`, i.e. **five orders of magnitude**
  above the invariant model and 4600x the target. A harness that could not see that would prove
  nothing about `cylindrical_invariant`.
* **`simple_invariant`** is the 2x2 control arm (handoff §5). It gets the same invariant readout
  and the same intrinsic coordinates but keeps SimpleViT, whose tokens are **not** azimuth
  equivariant. It scores `1.7e-3 … 5.7e-3` — about 8x better than the pinned cylindrical model
  and four orders worse than the invariant one. That is the handoff's prediction confirmed
  ("SimpleViT with the same invariant readout does not automatically get the guarantee"), and it
  is the number the paper needs in order to attribute the result to the *combination* rather
  than to the readout alone.

## 8. Warm-start parameter accounting

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

* Nothing is silently dropped: **0** shape mismatches, **0** checkpoint tensors absent from the
  new model, **0** new tensors left at random initialisation. The buckets are asserted to
  partition both key sets (`loaded + consumed = checkpoint_total`,
  `loaded + new_warm = model_total`).
* `lin_proj_0.proj.{weight,bias}` (257 params) is **consumed**, not loaded: `a[h] = sum_w W[h,w]`
  into `invariant_readout.weight [16]`, bias carried into `invariant_readout.bias [1]`. The test
  re-reads the checkpoint and asserts the readout weight is bitwise the folded pool.
* `lin_proj_0` is **removed** from the new model, so the invariant arms have no dead parameter.
  Net parameter count 32,132,961 vs the checkpoint's 32,133,201 (−240).
* `simple_invariant` from `xRIR_simple_8_shot/epoch_12.pth`: identical structure —
  264 loaded / 2 consumed / 2 warm-started / 0 dropped / 0 random, 266 tensors each side.
* **Caveat carried from the handoff (§3.3, closing note):** `src_coord_proj` keeps its shape but
  its input *semantics* change — it now sees intrinsic coordinates instead of raw xyz. Shape
  compatibility does not imply the old accuracy is preserved. The old optimizer state cannot be
  restored either (the parameter set differs), so this is a **new run with a recorded
  initialisation source**, not a resumption.

## 9. Item 5 — gradient step

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
| `audio_enc.cnn.conv1.weight` | 3.5034e-01 | 1.0000e-03 |
| `source_network.to_patch_embedding.2.weight` | 1.9830e-01 | 9.9999e-04 |

no gradient (pre-existing dead heads in the pinned xRIR, not introduced by exp_08): ['audio_enc.cnn.fc_backup.bias', 'audio_enc.cnn.fc_backup.weight', 'source_proj.proj.bias', 'source_proj.proj.weight']

One forward + backward + `AdamW(lr=1e-3, weight_decay=1e-4)` step on the **real** batch
(B = 3, K = 8), using `train_xRIR_backbone.compute_loss`'s objective (STFT log-magnitude L1 +
Schroeder energy-decay L1) with only the `.cuda()` transfers removed.

* All gradients finite.
* The two **new** readout parameters and both **coordinate-branch** parameters receive non-zero
  gradients and move by `~1e-3` (the AdamW step size) after the step.
* The pinned parameters keep training: `src_proj`, `lin_proj_1`, `lin_proj_2`, `time_proj`,
  `audio_enc.cnn.conv1`, `source_network.to_patch_embedding` all have non-zero gradients and
  all move.
* 212 of 216 trainable tensors get a non-zero gradient, and exactly those 212 change. The four
  silent ones are `source_proj.proj.{weight,bias}` and `audio_enc.cnn.fc_backup.{weight,bias}`
  — **pre-existing dead heads in the pinned `model/xRIR.py`** (`source_proj` is never called in
  `forward`; `fc_backup` is torchvision's saved ResNet classifier). exp_08 introduces none of
  them, and a test pins that list so a future change cannot quietly add a fifth.

## 10. Item 6 — regression on the pinned entries

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

Built from the same seed, `build_xrir_exp08("simple", 8)` and `build_xrir_exp08("cylindrical", 8)`
produce predictions **bitwise identical** (`torch.equal`, `max|d| = 0.0`) to `xRIR(...)` and
`xRIR_Cyl(...)` on the fixed real batch. The registry entries are the same *objects*
(`BACKBONES_EXP08["simple"] is model.xRIR.xRIR`), `model.xRIR_cyl.BACKBONES` is not mutated, and
`model/xRIR.py` still contains its hard-coded `torch.zeros_like(signal).cuda()` — a test asserts
that, because the CPU shim must remain a shim and never become an edit.

## 11. Deviations from the brief, and why

1. **Manifest hash wording.** The brief asks that the manifest's *sha256* equal
   `47637a55…`. That value is the manifest's **content** hash
   (`tools.reference_manifest.manifest_hash`, which excludes `ir_root`); the file's sha256 is
   `5eaea3727dd8…`. Both were verified and both match the exp_03 provenance record
   (`manifest_hash 47637a55ccc5, file sha 5eaea3727dd8`). No conflict, just two different
   hashes with the same purpose.
2. **File naming.** The brief writes `model/xrir_cyl_invariant.py`; the repo's file is
   `model/xRIR_cyl.py` / `model/xRIR_cyl_oriented.py` (the brief mis-cased the exp_06 file the
   same way). Repo casing was followed: `model/xRIR_cyl_invariant.py`. The factory is at
   `model/exp08_factory.py`, which the brief allows explicitly.
3. **Tolerance on the "real encoder tokens" readout check.** An initial self-imposed `1e-6`
   was replaced by the handoff's own `1e-5` for quantities that pass through the 12-layer
   encoder: the measured value is `7.6e-7` and it drifts by tens of percent with the thread
   count, so `1e-6` would have been a coin flip rather than a criterion. The closed-form
   coordinate transform keeps the tighter `1e-6` bound (measured `1.2e-7`, 8x margin). Both
   thresholds and both measurements are stated above.
4. **Waveform target.** See §5: "same order as the spectral residual" is not attainable through
   Griffin-Lim by *any* model, so the test asserts the control comparison instead. The raw
   numbers are reported, not hidden.
5. **`lin_proj_0` deleted from the invariant models.** Not requested, but it is fully consumed
   by the readout and would otherwise be an untrained tensor in the new checkpoint identity. It
   is recorded in the accounting as `consumed`, so nothing is unexplained.
6. **Extra work not asked for**, added because the claim would be weaker without it: the
   non-vacuity control (pinned `cylindrical` through the same harness), the input-rounding noise
   floor control (§4), the Griffin-Lim conditioning control (§5), and the encoder-equivariance
   check inside the readout test.

## 12. Ready for training? — yes, with one gap to build

**Nothing blocks Stage B on correctness grounds.** The structure is validated, the warm start is
accounted for, the gradients flow, and the pinned arms are untouched.

Two practical items before a run can start:

1. **There is no exp_08 training entry point yet.** `train_xRIR_backbone.py` takes
   `--backbone` from `model.xRIR_cyl.BACKBONES` (`simple` / `cylindrical` only), so
   `cylindrical_invariant` cannot be selected from it. exp_06 solved exactly this with
   `tools/exp06_train.py`, which owns a parser over `BACKBONES_EXP06` and calls the pinned
   trainer's `seed_everything` / `train_epoch` / `test_epoch` / `save_checkpoint` without
   modifying or monkey-patching it. exp_08 needs the same ~150-line wrapper over
   `BACKBONES_EXP08`, plus a `--warm-start` flag that routes through
   `tools.exp08_warmstart.build_warm_started` and writes the accounting into the run directory.
2. **The CPU delay shim is not needed for training.** `model/exp08_delay.py` exists only so
   `shift_and_align` can run with no GPU visible; on a GPU the pinned
   `torch.zeros_like(signal).cuda()` is a no-op and `model/xRIR.py` is used unmodified. Do not
   install `device_preserving_delay()` in a training run -- it would be a silent behaviour
   change relative to every existing checkpoint.
3. **GPU scheduling is not free.** Both visible GPUs were at 100% utilisation
   (46.3 GB and 33.0 GB used) while this stage ran; they belong to other experiments and were
   not touched. The handoff's estimate of ~31 h for a full 12-epoch cylindrical run is a rough
   historical figure (~155.8 min/epoch x 12), not a measured cost for this model, and a
   fine-tune has to be timed separately.

And two things Stage A deliberately does **not** license:

* **No accuracy claim.** The warm start changes `src_coord_proj`'s input semantics; nothing here
  predicts T60 / C50 / EDT or the spectral loss. Acceptance criteria for an accuracy cost must
  be fixed *before* the evaluation, and the recipe/checkpoint must not be selected on the
  6,337-query test set (handoff §5).
* **No substitute for GPU validation.** These residuals are CPU/FP32/TF32-off. The invariance
  has to be re-measured on the trained model in the production configuration before anything is
  claimed in the paper (handoff §6, last bullet).
