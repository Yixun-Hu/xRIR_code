# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 (the exp_11-owned family)

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue` (from main `c491396`)
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU, no process signals
**Started:** 2026-09-26 23:3x EDT
**Prompt:** `coder_prompts/round2_opus_prompt.md` (items 1–12), plan v3 §2.2/§2.3/§3/§4.6–14/§5, Codex plan reviews (round 1 §4–6; round 2 changes 1, 4, 5)

This file is written incrementally: each item is appended as it is committed.

## Commits

All on branch `exp11-cue`, in order from the branch base `c491396`.

| SHA | Subject | files / lines |
|---|---|---|
| `9951084` | exp11: SimpleViTOriented and xRIR_SimpleOriented (arms H/I encoder) | 3 files changed, 163 insertions(+) |
| `4abb921` | exp11: SimpleViTAdapter / xRIR_SimpleAdapter (arms J/K explicit cue) | 2 files changed, 301 insertions(+) |
| `a6da0d9` | exp11: BACKBONES_EXP11 registry and build_xrir_exp11 | 2 files changed, 95 insertions(+) |
| `487bfe8` | exp11 bookkeeping: Coder round-2 report, items 1-3 (record only) | 1 file changed, 91 insertions(+) |
| `5c4c587` | exp11: tests for the exp_11 approvals record (red) | 1 file changed, 174 insertions(+) |
| `781cf83` | exp11: exp_11 approvals record - keys, schema and committed-blob binding | 3 files changed, 220 insertions(+) |
| `765c175` | exp11: approvals closures and the producer/arm requirement matrix | 1 file changed, 141 insertions(+) |
| `0c449ff` | exp11: tests for the two named pretraining profiles (red) | 1 file changed, 150 insertions(+) |
| `769a6ef` | exp11: tools/exp11_recipe.py - H_RECIPE and I_RECIPE | 2 files changed, 216 insertions(+), 1 deletion(-) |
| `68b1aaa` | exp11: tests for the exp_11 pretraining entry point (red) | 1 file changed, 117 insertions(+) |
| `b59e234` | exp11: tools/exp11_train.py - the exp_06 loop on BACKBONES_EXP11 | 1 file changed, 275 insertions(+) |
| `3458146` | exp11: finalizer part 1 - run types, closures, orchestration and approvals | 1 file changed, 260 insertions(+) |
| `9bcca27` | exp11: finalizer part 2 - the twelve-epoch pretraining contract | 1 file changed, 86 insertions(+) |
| `b39b7d0` | exp11: finalizer part 3 - HAA children and the adapter-heading binding | 1 file changed, 203 insertions(+) |
| `216aef3` | exp11: finalizer part 4 - job admission, spec, lineage and children | 1 file changed, 208 insertions(+) |
| `c7b1d7b` | exp11: tools/exp11_smoke.py - enumerated diagnostic kinds with budgets | 1 file changed, 236 insertions(+) |
| `4845f36` | exp11: finalizer part 5 - diagnostic evidence for the enumerated kinds | 1 file changed, 151 insertions(+), 2 deletions(-) |
| `13b815f` | exp11: finalizer part 6 - finalize(), preflight, passed and the CLI | 1 file changed, 166 insertions(+) |
| `5b67c68` | exp11: tests for the exp_11 finalizer | 2 files changed, 234 insertions(+), 1 deletion(-) |
| `c0b10a9` | exp11: HAA entry points with the faithful orchestration loop and cue routing | 2 files changed, 442 insertions(+) |
| `cf9081a` | exp11: CPU parity test for the fine-tuning orchestration loop | 1 file changed, 205 insertions(+) |
| `5614533` | exp11 bookkeeping: Coder round-2 report, items 4-8 (record only) | 1 file changed, 150 insertions(+) |
| `8f5ff9a` | exp11: tools/exp11_launch.sh - smoke \| probe \| full \| finalize, --arm H\|I | 3 files changed, 283 insertions(+), 2 deletions(-) |
| `e978c2e` | exp11: tools/exp11_haa_pipeline.sh and golden dry-run tests | 12 files changed, 970 insertions(+) |
| `7ca4e4e` | exp11 bookkeeping: Coder round-2 report, items 9 and 11 (record only) | 1 file changed, 50 insertions(+) |
| `e7cb7f4` | exp11: summariser arms H/I/J/K, the admission adapter, the phases and A' | 1 file changed, 351 insertions(+), 33 deletions(-) |
| `505d5bf` | exp11: tests for the admission adapter, the phases and the external row | 3 files changed, 262 insertions(+), 4 deletions(-) |
| `a6f51d6` | exp11 bookkeeping: Coder round-2 report, items 10 and 12 (record only) | 1 file changed, 89 insertions(+) |
| `f795908` | exp11: tidy - SERIALIZED_ROLE replaces the inline map, drop an unused import | 2 files changed, 5 insertions(+), 5 deletions(-) |
| `cd356ae` | exp11: plan section 5 acceptance - a phase never overwrites a published record | 1 file changed, 19 insertions(+) |
| `77f19b9` | exp11: plan section 5 acceptance - phase-1 oracle and the cross-family child wall | 2 files changed, 73 insertions(+) |
| `e215b5a` | exp11 bookkeeping: Coder round-2 report, design decisions and open questions (record only) | 1 file changed, 56 insertions(+) |
| `9e2f421` | exp11: a child that installs no adapter heading may record no adapter_phi_deg | 1 file changed, 3 insertions(+) |
| `e855bba` | exp11 bookkeeping: report the expected exp_06 approvals re-fill (record only) | 1 file changed, 23 insertions(+) |
| `8d86d43` | exp11: pass exp11_approved by keyword; the arm registry test admits round 2's arms | 2 files changed, 6 insertions(+), 5 deletions(-) |

Two commits exceed the 200-changed-line guideline and are called out here rather than
buried: `e978c2e` (+970) is almost entirely the ten **generated golden dry-run files**
plus the two shells, and `e7cb7f4` (+351/-33) is the single summariser change of item 10,
which could not be split without leaving the module in a state that does not import (the
phase registry, the admission table and the external row reference one another).
`b59e234` (+275) and `c0b10a9` (+442) are entry points mirroring exp_06 modules of the
same size whose `main` is one unit.

## Item 1 — `model/simple_vit_oriented.py`, `model/xRIR_simple_oriented.py`

`SimpleViTOriented(SimpleViT)` keeps the parent's signature exactly (checked in the test,
as exp_06's encoder test does) and builds the parent with `channels + 2 = 5`. The two
planes are `azimuth_channels(H, W)` **imported from `model/cylindrical_vit_oriented.py`**
rather than recomputed, registered as a **non-persistent** buffer (`az_channels` appears
in `named_buffers` but not in `state_dict`, and the state-dict key set equals the pinned
`SimpleViT`'s). Tests assert the planes equal (a) `convert_equirect_to_camera_coord`'s
normalised horizontal look direction, (b) `cos/sin` of `theta = (c + 0.5)*2*pi/W - pi`,
and (c) `CylindricalViTOriented.az_channels` (same function).

**Encoder parameter increment, derived (not assumed): 526 336** at the recipe tier
(`image_size (256,512)`, `patch_size (16,32)`, `dim 512`). It coincides with the
cylindrical figure because the two patch embeddings have the same shape
(`LayerNorm(patch_dim) -> Linear(patch_dim, dim) -> LayerNorm(dim)`), so the increment is
`2*p1*p2*dim + 2*(5-3)*p1*p2 = 524 288 + 2 048`. The test computes it from
`xRIR(...).source_network` vs `xRIR_SimpleOriented(...).source_network` at the real tier
and also checks the general formula at a second patch size/dim.

`xRIR_SimpleOriented(xRIR)` swaps only `source_network`; `SimpleViT` exposes no
`num_tokens`, so the 256-token pool guard computes `(H//p1)*(W//p2)` itself and refuses a
mismatch by name (`lin_proj_0 …`).

## Item 2 — `model/xRIR_simple_adapter.py`

`SimpleViTAdapter` **wraps** a pinned `SimpleViT` as `self.vit` (composition:
`type(model.source_network.vit) is SimpleViT` is asserted) and repeats its three forward
steps so the cue can be inserted after the patch embedding's final LayerNorm:

```
x   = vit.to_patch_embedding(img)
out = vit.transformer(x + (vit.pos_embedding.to(x) + heading_code))
```

The cue is folded into the positional code before the single add, so at initialisation
(`heading_code` exactly zero) the sum is bit-for-bit the pinned `x += pos_embedding`.

* `token_azimuth(W_tok)` pins the convention `theta_j = 2*pi*(j + 1/2)/W_tok - pi`.
* `heading_proj = Linear(2, dim)`, **weight and bias both zeroed** → 1 536 parameters at
  `dim 512` (`2*512 + 512`), asserted both directly and as the difference against a
  pinned `SimpleViT`.
* `heading_code` returns `[tokens, dim]`: the per-column code broadcast down the
  `token_rows` (token order is row-major, matching the `Rearrange` in the patch
  embedding). A test with `weight = eye(dim, 2)`, `bias = 0.25` checks each token's
  first two components are `cos/sin(theta_j - phi) + 0.25` for every row.
* `set_heading(phi_deg)` accepts only a finite native `int`/`float` (`True`, `None`,
  `nan`, `'0'` are refused by name); a forward without a heading raises.

Two strict loading modes on `xRIR_SimpleAdapter`:

* `load_base_checkpoint(state)` — mode (i): refuses any state that already carries
  `source_network.vit.*` or the adapter keys (that is a stage checkpoint), remaps
  `source_network.` → `source_network.vit.`, zeroes the adapter and calls
  `nn.Module.load_state_dict(..., strict=True)` with the adapter's own zeros, so a
  missing or extra inherited key is a `RuntimeError` from torch's strict loader.
* `load_state_dict(state, strict=True)` — mode (ii): refuses `strict=False` by name and
  refuses a state missing either adapter key, then loads strictly. A trained adapter can
  never be silently re-zeroed.

**Design note (CPU):** the pinned `xRIR.forward` calls `.cuda()` unconditionally
(`apply_delay`), so the "bit-exact at init" test is made at the geometry encoder — the one
component the adapter replaces — plus a full state-dict comparison (every inherited
tensor equal under the prefix remap). Nothing downstream of `source_network` differs.

## Item 3 — `model/xrir_exp11_registry.py`

`BACKBONES_EXP11 = {**BACKBONES_EXP06, 'simple_oriented': …, 'simple_adapter': …}`,
`build_xrir_exp11`, and an exp_11 `registry_sha256()` defined exactly as
`tools.exp06_train.registry_sha256` defines its own. Tests pin exp_06's registry digest
`f9852f8568fb62aad5ec39a955b8e254dad6b5f53bbde71c69b048efa8b3e169` (the value recorded in
every completed exp_06/exp_09 child) against `exp06_finalize`, `exp06_train` and
`exp06_haa_finetune`, assert the inherited routes are the *same class objects* (`is`) and
that models built through either factory produce bit-identical encoder outputs.

## Item 4 — `tools/exp11_profiles.py` (+ the all-null approvals record)

Commits `781cf83` (schema and binding, 154 lines) and `765c175` (closures and matrix).

Eight code keys: `train`, `finalize`, `haa_finetune`, `haa_eval`, `haa_pipeline_sh`,
`launch_sh`, `smoke`, `summarize_haa`. The two shell orchestrators have no Python import
closure and are bound as files alone (`(None, ('tools/exp11_launch.sh',))`), which is
Codex round-2 change 5's `launch_sh` requirement. `summarize_haa` points at
`tools.exp06_summarize_haa` — the one shared producer — so exp_11 pins the closure **it**
runs rather than reusing exp_06's pin.

`reused` = `approved_digests_exp04`, `approved_digests_exp06` (sha256 of those records)
and `legacy_receipt` (`{path, sha256}` of `ckpt/exp06/legacy_receipt.json`).
`artifacts` = `simpor_epoch_012`, `simpor_yaw_epoch_012`, each `{epoch, path, sha256}`.

**Committed-blob binding** mirrors exp_06 exactly: `load_approved_digests(path, repo,
commit)` refuses bytes that differ from the blob committed at that commit and refuses a
path outside the repository, and `approvals_at_commit` parses a private copy of a child's
own reviewed blob (so re-filling the approvals for a later producer never invalidates an
already certified child). `tools/exp06_profiles.py` is imported **read-only** for
`committed_bytes` and the test asserts both exp_06 approvals modules are byte-identical
to their committed blobs.

**Requirement matrix** (`PRODUCER_REQUIREMENTS` / `PRODUCER_OUTPUTS` / `PRODUCER_CODE_KEYS`):

| producer | requires | produces |
|---|---|---|
| `pretrain_simple_or`, `pretrain_simple_or_yaw` | `code.{train,finalize,launch_sh,smoke}` | its own `artifacts.simpor*` |
| `haa_control_adapter`, `haa_yawaug_adapter` | the HAA code keys + all `reused` | — |
| `haa_simple_or` | the above + `artifacts.simpor_epoch_012` | — |
| `haa_simple_or_yaw` | the above + `artifacts.simpor_yaw_epoch_012` | — |
| `summarize_phase1b` | `code.summarize_haa` + HAA code keys + `reused` | — |
| `summarize_final` | every code, reused and artifact leaf | — |

A parametrised test asserts no producer requires its own output; another asserts the
adapter queues, both pretrainings and the phase-1b summary all pass on a record whose
**both** H/I pins are null, that `haa_simple_or` passes with only H's pin filled, and
that `summarize_final` names exactly the missing arm leaf.

## Item 5 — `tools/exp11_recipe.py` (commit `769a6ef`)

One shared `NUMERICAL` recipe (exp_01's: lr 1e-3, wd 1e-4, decay 3, gamma 0.1, 12 epochs,
batch 32 × accum 2, seed 0, TF32, 512/12/8/512) and two profiles:

* `H_RECIPE` — `yaw_aug 0`, `save_every 500`, `epoch_ckpt_every 1`; `yaw_aug_seed`/
  `yaw_aug_width` recorded but **inert** (integers only).
* `I_RECIPE` — exp_04's: `yaw_aug 1`, `yaw_aug_seed 0`, `yaw_aug_width 512`,
  `save_every 0`, `resume None`, plus `check_counter` enforcing
  `epochs × batches < 2**20`.

The profile is chosen by `select_profile(args)` from the run's **own** record (backbone +
`yaw_aug`), never from a caller's claim, and `check_all` additionally refuses an
`exp11_profile` field that disagrees with that selection — so a mixture (I's augmentation
with H's saving cadence, or the reverse) is a named deviation. A test reads the real
`ckpt/xRIR_simple_8_shot/args.json` and `ckpt/xRIR_simple_yawaug_8_shot/final/args.json`
and asserts each profile agrees with the historical record it is taken from. Derived
parameter counts come from `build_xrir_exp11`; `strict_equal`, `compare_sources` and
`check_budget` are imported from the unedited `tools/exp06_recipe.py`.

## Item 6 — `tools/exp11_train.py` (commits `68b1aaa`, `b59e234`)

exp_06's training entry point on `BACKBONES_EXP11`. The pinned trainer's
`seed_everything`/`seed_worker`/`train_epoch`/`test_epoch`/`save_checkpoint` and exp_06's
`resolve_data_root`/`data_identity`/`geometry_identity`/`heldout_wav_identity` are
imported unchanged. `--yaw-aug` accepts 0 **and** 1 and re-applies exp_04's post-conditions
(`--save-every 0`, no `--resume`, the 2**20 counter bound, re-checked once the loader
length is known). `prepare_args` records six `exp11_*` fields including
`exp11_profile`, refuses a capacity outside tier M and deletes the launcher-only flags, so
`args.json` holds exactly the trainer's fields plus exp_11's provenance class. Note: 275
lines in one commit — the module is a single entry point mirroring `tools/exp06_train.py`
(382 lines) and its `main` is one loop, so it was not split.

## Item 7 — `tools/exp11_finalize.py` (six commits, 1 072 lines)

`tools/exp06_finalize.py` is **imported, never monkeypatched**: log closing, exit
receipts, liveness, identity schemas, `haa_history`, `haa_metrics`, `child_identity`,
`rehash_bound_evidence`, `check_job_identity`, `expected_children`, `job_log`,
`write_completion`, `confined`, the GPU census and `child-exit` all come from it. exp_11
re-owns exactly the functions that read exp_06's module globals.

**Serialized run types vs semantic roles** (Codex change 4): `--run-type` takes
`exp11_train | exp11_smoke | exp11_haa_finetune | exp11_haa_eval | exp11_haa_job`; the HAA
children record `exp11_haa_train` / `exp11_haa_eval` in their own `provenance.json`, so
neither finalizer's CLI or validators accept the other's children (tested both ways).
`child_role` maps a path's semantic role (`stage1`, `eval/<room>`) onto exp_11's run type.

**Heading binding.** `heading_records(args, field, rooms, repo)` is exp_06's per-room
validator with the record field as a parameter. `frame_binding` is fail-closed:
heading-frame children may install no adapter; only the `simple_adapter` backbone may bind
an `adapter_heading`; **all bound records must declare the same `phi_deg`** and
`adapter_phi_deg` must equal it. Tested against the four real confirmatory records in
`ckpt/exp06/heading` and the live HAA cache, plus a stubbed mixed cohort (all four real
records declare −90, so a genuine mix cannot be constructed).

**Completion contract** — enumerated in full in the module docstring and identical to
exp_06's: child exit 0, a closed log with the end marker, the exit receipt binding those
bytes, a stale-log re-hash, three-way closure agreement, input revalidation, approvals
committed at the child's reviewed commit, the recipe schema on all three argument copies,
epochs 1–12 exactly once with finite history, `epoch_012.pth` equal to `last.pth["model"]`
in keys/dtypes/shapes/values, artifact hashes, and (for HAA) validation-based selection,
`summary.json` agreement, full test indices via the summariser, protocol meta and job
completeness.

**Diagnostics.** The receipt names its `kind`; the kind decides the entry, the provenance
run type, the ceiling a receipt may claim and the artifact contract inside
`ckpt/exp11/_smoke`. `diagnostic: true` and `admissible_arm: false` are required of both
the receipt and the diagnostic provenance.

## Item 11 (partial) — `tools/exp11_smoke.py`

Three enumerated kinds with explicit budgets and artifact contracts: `probe`
(`tools.exp11_train`, 900 s, 40 GiB, `--no-save` enforced), `haa_train_smoke`
(1800 s, 40 GiB, the stage files) and `haa_eval_smoke` (1800 s, 40 GiB, the room files).
A caller may tighten a budget but never loosen it. The bounded machinery (budget check,
peak allocation, watchdog, publication) is imported from the pinned `tools/exp06_smoke.py`.
`--make-fixture` writes a CPU `simple_oriented`/`simple_adapter` state dict, because the
exp_01 checkpoints cannot load into the five-channel patch embedding.

## Item 8 — `tools/exp11_haa_finetune.py`, `tools/exp11_haa_eval.py` + parity test

`fine_tune()` is exp_06's embedded loop transcribed step for step, with the **pinned
numerical helpers as the module-level defaults** of its keyword arguments
(`train_xRIR_backbone.compute_loss`, `sim_to_real.finetune_haa.evaluate`, `torch.save`,
`ExponentialLR`) — a test asserts the defaults **are** those objects, so production never
runs a substitute.

**Parity-test design.** Neither loop can run end-to-end on CPU (`xRIR.forward` and exp_06's
`main` call `.cuda()` unconditionally), and monkeypatching exp_06's module globals is
forbidden. So the test carries `reference_loop`, a line-for-line transcription of exp_06's
`main` epoch loop with only the device moves and record writing removed, runs both loops
on one tiny synthetic problem (a `Linear(4,1)`, three batches, `accum_steps 2` so the
accumulation boundary and the last-batch step both fire) with a scripted validation
sequence `(5, 4, 4, 6, 3)` that exercises improvement, a **tie** (not an improvement), a
rise and a final-epoch best, and compares: every `history.jsonl` row, the evaluate-call
trace, the summary fields and the **sha256 of `best.pth` and `last.pth`**. The source
digest of exp_06's `main` (`88973e01…`) is pinned, so a change to the old loop fails the
parity guard instead of letting the transcription drift.

`select_frame` returns `(frame, k_by_room, heading, adapter_heading, adapter_phi_deg)` and
is fail-closed: never both cues; `simple_oriented`/`cylindrical_oriented` require
`--heading-json-dir`; `simple_adapter` requires `--adapter-heading-json-dir`; a cohort whose
records declare different headings is refused. `load_init` picks between the two strict
loading modes by what the checkpoint carries (`base` for an exp_01/exp_04 init, `adapter`
for a later stage), and records which mode it used.

`tools/exp11_haa_eval.py` **imports** exp_06's `evaluate_room`, `room_summary` and
`write_room_outputs` unchanged — the published numbers come from the very code exp_06 and
exp_09 published theirs from — and owns only the parser, the records and the per-sample
`meta` (which adds `adapter_heading`, `adapter_phi_deg` and `exp11_source_closure_sha256`).

## Item 11 — `tools/exp11_launch.sh`

`smoke | probe | full | finalize`, `--arm H|I`. It **sources** `tools/exp06_launch.sh`
as a library (the drained pipe, the end marker, the exit receipt, `abort`, `own_launch`)
because that lifecycle is shared evidence, not an exp_11 decision — forking it would fork
the very bytes `closed_log` re-validates — and overrides everything that names an
experiment: `preflight`/`finalize`/`require_passed` call `tools/exp11_finalize.py`,
`APPROVED` is exp_11's record (re-assigned unconditionally, because the sourced library
had already resolved exp_06's), the attempt roots are
`ckpt/exp11/pretrain/xRIR_simpor_8_shot` and `…/xRIR_simpor_yawaug_8_shot`, the record and
log prefix are exp_11's, and `arm_of` fixes the arm's recipe (H: `--yaw-aug 0
--save-every 500`; I: `--yaw-aug 1 --save-every 0`). Because of that override the
`launch_sh` approvals key binds **both** shell files.

`full` runs the child under `timeout --kill-after=60 $EXP11_FULL_CEILING_S` (default
129 600 s = **36 h**, plan §7's reservation), so a stalled attempt fails and is aborted
rather than holding the card. Promotion is the atomic `ln -s` + `mv -Tf` of
`final → attempt_<UTC>`. `preflight` passes all three roots (both attempt roots and the
smoke tree) so no live `launch.pid`/`child.pid` anywhere admits a second launch.

## Item 9 — `tools/exp11_haa_pipeline.sh`

Inits `simple_or` / `simple_or_yaw` (heading frame; `${EXP11_SIMPOR_CKPT:-…}`,
`${EXP11_SIMPOR_YAW_CKPT:-…}`) and `control_adapter` / `yawaug_adapter` (room frame with
`--adapter-heading-json-dir`). `cue_args` picks the flag from the frame, reset on every
`init_of`, so a mixed queue never inherits the previous job's cue. Roots
`ckpt/exp11/sim2real/<init>/`, record log prefix `orientation_cue_fairness_haa`,
`<init>:<seed>` and `<init>:zeroshot`, and the whole queue is refused before anything runs
if any job names an unknown init or a malformed seed.

The job spec declares `heading` (heading frame) **or** `adapter_heading` +
`adapter_phi_deg` (adapter arms, with the shared-heading check), and is validated by the
very loader that will admit the job (`exp11_finalize.load_job_spec`).

The identity gate resolves each init through `tools/exp11_profiles.py`'s producer matrix:
`simple_or`/`simple_or_yaw` against exp_11's artifact keys, `control_adapter` against
`exp04_profiles.CONTROL['sha256']`, `yawaug_adapter` against
`exp06_finalize.exp04_aug_checkpoint` — and before doing so it checks that exp_06's
approvals file is the `reused.approved_digests_exp06` bytes exp_11 approved.

`prepare_job` runs an **explicit exclusive-card census** (`nvidia-smi --query-compute-apps`
must be empty) before every job, because the pipeline does not inherit the launcher's
preflight; then the approvals gate, then the exclusive job root, then the spec and its
validation. A refused preparation writes a `preparation_failure.json` receipt and renames
the root `_ABORTED_prepare_<reason>` only when this attempt created it.

Golden dry runs (`orientation_cue_fairness_results_assets/golden/`) freeze every argv for
both shells: the launcher's four modes and both arms, each pipeline init at a seed, a
zero-shot set, and a three-job mixed queue (asserted to run one census per job).

## Item 10 — `tools/exp06_summarize_haa.py` (the only exp_06 file this round touches)

**Arms.** `simple_or` (H) and `simple_or_yaw` (I) — `simple_oriented`, heading frame,
`cue: 'planes'`; `control_adapter` (J) and `yawaug_adapter` (K) — `simple_adapter`, **room**
frame, `cue: 'adapter'`. All four carry `'admission': 'exp11'` and roots
`ckpt/exp11/sim2real/<init>`. `EXP04_AUG_ARMS` gains K (its init is exp_04's approved
`checkpoints.aug`); J's init is the `exp04_profiles.CONTROL` literal, as `control_hf`'s is.

**Admission adapter shape.** A module-level `ADMISSION` table maps each family to
`{finalizer, job_run_type, extra_job_fields}`, and `arm_admission(arm)` picks it from the
arm's own `'admission'` key (default `exp06`, so every historical arm is untouched, G
included). The dispatch covers all four calls Codex named:

* `job_completion` requires the arm's own **serialized** run type (`haa_job` vs
  `exp11_haa_job`) and the extra job fields (`adapter_heading`, `adapter_phi_deg`) of
  exp_11's families;
* `verify_job` calls `validators.load_job_spec`, `validators.verify_child` and
  `validators.job_lineage` where `validators = admission['finalizer']`;
* the **semantic** role used by `child_protocol`, `child_recipe`, `arm_closures` and
  `ROLE_CODE_KEY` stays exp_06's path-derived `haa_train`/`haa_eval` for both families —
  the serialized/semantic distinction Codex asked for.

`check_arm_identities` now reads **each arm's own** approvals for the `code` keys (exp_06's
record for historical arms, exp_11's for H/I/J/K) while the heading records stay checked
against exp_06's `artifacts.heading` for every arm that binds one, because those records
are exp_06's approved artefacts whoever consumes them. No historical key is replaced or
disabled.

**Adapter-heading admission.** `child_per_sample` permits `adapter_heading` /
`adapter_phi_deg` **only** for arms registered with `cue == 'adapter'`, requires the rolls
to be the registered `HEADING_K` with a json identity per room, requires
`adapter_phi_deg` to be finite and to equal the single heading the bound records declare,
and refuses an adapter cue on any other arm — while the historical room-frame rejection of
a frame `heading` stands unchanged for A, B and E. `arm_adapter_headings` then requires one
record per room and **one installed heading across every child of the arm**.

**Phases.** `EXPERIMENTS` gains `exp11_phase1b` (arms A–G + J, K; Q1–Q4; screen families
J−A, C−J, K−E, C−K; outputs `ckpt/exp11/phase1b/…`) and `exp11_final` (all eleven arms;
N1, N1i, N2, N3, Q1–Q4, P3, P3′, P1, P2, P4, P4′ — fourteen statements; thirteen screen
families; outputs `ckpt/exp11/{stats.json,summary.txt}`). `--phase {phase1,phase1b,final}`
selects one through `experiment_key`, `--experiment` keeps its three public names, and
`check_output_paths` now also stops a phase from writing another phase's canonical record.
exp_06, exp_09 and exp_11 phase 1 publish exactly what they always did: `external` is only
set where a config declares it, and `exp11_approved_digests` only where exp_11's approvals
were actually read.

**A′.** `external_rows` copies exp_02's `released|fine-tuned|<room>|<metric>` means and sd
from `ckpt/sim2real/stats.json` through `historical_source` (which re-verifies the
sibling `summary.txt` against the record's own `summary_sha256`), recording the source
path, its sha256 and the per-cell selector, with `paired: False` and
`inference: 'none (external reference row)'`. It is not in `ARMS`, so no decision, screen
or interval can read it.

**exp_11's approvals in the summariser.** `exp11_approvals(...)` reads exp_11's record
only when the selected configuration actually carries an exp_11-admitted arm, binds it to
the blob committed at the reviewed commit, and runs the producer matrix for that phase
(`summarize_phase1b` / `summarize_final`).

## Item 12 — static checks, digests and the suite

`bash -n` on both shells, `py_compile` on every new module and test, `git diff --check`:
all clean; the tree is clean outside `worklog/`.

**exp_06 code digests recomputed at this branch's HEAD (`7b1a7af`)** — exactly **one**
key moved against the record re-filled at `9f98bbb`, as the round-2 scope allows:

| key | approved at 9f98bbb | now |
|---|---|---|
| `summarize_haa` | `645e74c03e20…` | `e10d7aa3c75d…` |

The other nineteen exp_06 keys are unchanged.

**exp_11 code digests at `7b1a7af`** (all eight keys present in this checkout):

| key | digest |
|---|---|
| `train` | `d9cbb9de8eb0ec85ac85664d9e0da7afe156e5b169bbe89096953d863cf15571` |
| `finalize` | `baea4d94ac72ae42497791b8ed72e59cbdb6a5bdb47adbf15f9e440ad0e8f99f` |
| `haa_finetune` | `bd8074e71e54600bacc79c907df1ad61ab57e655df2169119193758cca1f5150` |
| `haa_eval` | `f822365cb0c8d80efb9d4105b743824cb2b66bc19c1822346b0d8c75b994fe20` |
| `haa_pipeline_sh` | `163d1fda1bc5700c3faf7812d9f924972a76e172e335a90254c4fd4e7c08ee14` |
| `launch_sh` | `5e950c224e212c6f932a2068ad2083be8ceb2792c09cca546e762154b5a6d9ab` |
| `smoke` | `62133ba07a35d42026c19fd1484b37c6c06ace0ebedc1cdd8484690df323450e` |
| `summarize_haa` | `e10d7aa3c75db2e2ea18cc4ba7ff91135232681ae865546f612a0b67811a9e86` |

(An earlier draft of this report listed the values at `505d5bf`; four keys moved again
with the last three commits — the `launch_sh` binding of the HAA producer, the heading
gate in the pipeline and the disjoint diagnostic run types — which is exactly why the
re-fill belongs at the reviewed commit and not here.)

These are the values a reviewer fills into
`orientation_cue_fairness_results_assets/approved_digests.json` at the **second** reviewed
commit; the committed record is still all-null, so every exp_11 producer refuses today.

## Late refinements (after the item-by-item pass)

Four self-review findings, each committed on its own:

1. **`launch_sh` joins the HAA producer's code keys** (`7b1a7af`). The queue sources
   `tools/exp11_launch.sh` (and through it the pinned exp_06 library) for the child
   lifecycle, so the pair that decides an exp_11 child must be pinned by the producer that
   starts it — otherwise the lifecycle code would be unbound at queue time.
2. **The queue gates the four heading records** (same commit). It first checks that
   exp_06's approvals file is the `reused.approved_digests_exp06` bytes exp_11 approved,
   then that every room's heading JSON hashes to exp_06's approved `artifacts.heading` —
   exactly what exp_06's own queue checks, and for both frames, so the room-frame adapter
   jobs are gated on the same four approved records.
3. **The diagnostic run types are disjoint too** (`8c…`, "the diagnostic run types are
   disjoint"): `exp11_probe` / `exp11_haa_smoke_train` / `exp11_haa_smoke_eval`, so a
   diagnostic record of one family can never be read as the other's. (The runner identity
   and the enumerated `kind` already refused an exp_06 receipt; this makes the recorded
   evidence self-describing as well.)
4. **Symmetric frame/backbone guards in `frame_binding`** (`7c88814`). The finalizer now
   refuses from the admission side exactly what the entry point refuses at launch: the
   `simple_adapter` backbone may not be admitted in the heading frame, and a
   heading-frame backbone (`simple_oriented`, `cylindrical_oriented`) may not be admitted
   in the room frame, where it would deliver no cue at all. Both entry point and finalizer
   read the same two constants, asserted equal in the test.

## Design decisions worth a reviewer's attention

1. **The parity test compares two loops, not a loop against itself.** exp_06's fine-tuning
   `main` cannot be called on CPU (it `.cuda()`s unconditionally) and its module globals
   may not be monkeypatched, so the test carries a line-for-line transcription of that
   loop and **pins the sha256 of `inspect.getsource(exp06_haa_finetune.main)`**
   (`88973e01…`). If exp_06's loop ever changes, the guard fails and the transcription
   must be re-derived; it cannot drift silently. A companion test asserts that exp_11's
   loop defaults **are** the pinned helper objects, so nothing is substituted in production.
2. **Admission adapter shape.** One `ADMISSION` table keyed by the arm's own
   `'admission'` field, holding the validator module, the serialized job run type and the
   extra job fields. `verify_job` takes `validators = admission['finalizer']` once and uses
   it for the spec loader, the child verifier and the lineage; the *semantic* role stays
   exp_06's path-derived `haa_train`/`haa_eval` everywhere the protocol and recipe checks
   read it. Nothing historical is re-keyed or disabled.
3. **Requirement matrix.** Expressed as explicit **leaf paths** (`code.train`,
   `artifacts.simpor_epoch_012.sha256`, …) rather than section names, because the plan's
   rule — "pretraining never requires its own output pin; H runs while I's is null" — is a
   statement about leaves. A parametrised test asserts the disjointness of every
   producer's requirements and its outputs.
4. **Adapter heading routing.** The cue is one value for every room, so it travels three
   ways and each is checked: the job spec declares `adapter_heading` + `adapter_phi_deg`;
   the entry point validates the cohort's records agree and installs the shared value with
   `model.set_heading(phi)`; `args.json` and the per-sample `meta` record both fields, and
   the finalizer re-reads every record against the cache it was estimated from and refuses
   a mixed heading. In the summariser the fields are admitted **only** for arms registered
   with `cue == 'adapter'`, and the historical room-frame rejection of a frame `heading`
   is untouched.
5. **`launch_sh` binds two files.** exp_11's launcher sources the pinned
   `tools/exp06_launch.sh` for the child lifecycle, so the approvals key binds the pair —
   what decides an exp_11 launch is both files, not the new one alone.
6. **`child-exit` is delegated, not forked.** exp_11's finalizer routes the `child-exit`
   subcommand to `exp06_finalize.child_exit_main`, so the end marker text and the receipt
   contract stay the bytes `closed_log` re-validates.

## Open questions for the Planner and the reviewer

1. **The `EXP06_CHILD_EXIT` marker is shared.** exp_11's logs carry exp_06's marker text
   because the pinned closer writes it. That is deliberate (see 6 above) but it means a log
   does not name its experiment; the run directory and the completion do.
2. **`tools/exp11_train.py` is 275 lines in one commit** and `tools/exp11_finalize.py`
   totals 1 072 across six commits. Both mirror exp_06 modules of comparable size whose
   `main`/evidence functions are single units; splitting them further would have produced
   states that do not import.
3. **The smoke mode's training rung uses the `probe` kind with tightened budgets.** The
   plan enumerates exactly three diagnostic kinds, and a bounded training smoke is a probe
   with a smaller budget rather than a fourth kind. Say if a distinct `train_smoke` kind is
   wanted.
4. **`ckpt/exp11/_smoke` is exp_11's disposable tree**, mirroring `ckpt/exp06/_smoke`; the
   finalizer refuses to certify a diagnostic outside it.
5. **The exp_11 approvals record is all-null**, so every exp_11 producer refuses today.
   The digests in the table above are what a reviewer fills in at the second reviewed
   commit; they will move again if the reviewer asks for changes.
6. **No GPU work was done.** Every check in this round ran with `CUDA_VISIBLE_DEVICES=''`;
   the bounded card smokes of plan §5 rung 2 remain to be run before the first queue.

## Expected failure inside the branch: the exp_06 approvals re-fill

`tests/test_exp06_profiles.py::test_every_filled_record_digest_is_the_one_this_checkout_computes`
**fails on this branch, by design and exactly as in round 1.** It asserts that every
*filled* digest in exp_06's approvals record equals what the checkout computes; editing
`tools/exp06_summarize_haa.py` (item 10, the one exp_06 file this round is allowed to
touch) moves `code.summarize_haa`, so the record is stale until it is re-filled. Round 1
met the same thing and resolved it with commit `9f98bbb`
("approvals re-fill after the exp_11 round-1 merge (haa_pipeline_sh,
summarize_haa:645e74c0)").

The re-fill belongs **after** the reviewed merge, not here: `code.summarize_haa`'s closure
now contains `tools/exp11_finalize.py` and `tools/exp11_profiles.py`, so the digest moves
again with every exp_11 module change — including any review fix cycle. The keys to
re-fill after this round's merge are exactly:

* exp_06's record: `code.summarize_haa` (one key; the other nineteen are unchanged);
* exp_11's record: all eight `code` keys, at the second reviewed commit.

Everything else in `tests/test_exp06_profiles.py`, `tests/test_exp06_approvals_api.py` and
`tests/test_exp09_sim_eval_closures.py` (the pinned `eval` / `eval_launch` / `compare` /
`mirror_probe` closures) is green: **76 passed, 1 failed** in that group.
