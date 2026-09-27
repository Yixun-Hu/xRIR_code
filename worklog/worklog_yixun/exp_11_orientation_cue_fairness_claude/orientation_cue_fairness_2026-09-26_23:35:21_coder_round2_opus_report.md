# Coder report — exp_11 `orientation_cue_fairness`, ROUND 2 (the exp_11-owned family)

**Coder:** Claude Opus 5 (1M context), Claude Code
**Worktree:** `/home/yixunhu/codespace/xRIR_code_wt`, branch `exp11-cue` (from main `c491396`)
**Interpreter:** `/home/yixunhu/miniconda3/envs/xRIR/bin/python`, `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=''` throughout — no GPU, no process signals
**Started:** 2026-09-26 23:3x EDT
**Prompt:** `coder_prompts/round2_opus_prompt.md` (items 1–12), plan v3 §2.2/§2.3/§3/§4.6–14/§5, Codex plan reviews (round 1 §4–6; round 2 changes 1, 4, 5)

This file is written incrementally: each item is appended as it is committed.

## Commits

| # | SHA | Subject | +/- lines |
|---|---|---|---|
| 1 | `9951084` | exp11: SimpleViTOriented and xRIR_SimpleOriented (arms H/I encoder) | +163 |
| 2 | `4abb921` | exp11: SimpleViTAdapter / xRIR_SimpleAdapter (arms J/K explicit cue) | +292 |
| 3 | `a6da0d9` | exp11: BACKBONES_EXP11 registry and build_xrir_exp11 | +91 |

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
