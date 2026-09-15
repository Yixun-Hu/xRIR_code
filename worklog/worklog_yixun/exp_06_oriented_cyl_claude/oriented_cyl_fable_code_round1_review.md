**Reviewer:** Claude Fable 5.1 (Agent subagent, model fable) · **Date:** 2026-09-14

# exp_06 oriented_cyl — code review, Coder round 1 (encoder, factory, heading tool)

**Scope.** Branch `exp06-window` of the worktree `/home/yixunhu/codespace/xRIR_code_wt`, nine commits `f9300cf..a950605` on top of main `77ef292`, delivered by OpenAI Codex gpt-6-astra against plan v4 §8 "Round 1" (as amended by Codex plan review R3.1). Six new files (974 lines): `model/cylindrical_vit_oriented.py`, `model/xRIR_cyl_oriented.py`, `tools/exp06_heading.py`, `tests/test_exp06_{encoder,factory,heading}.py`. Out of scope (rounds 2–3): entry points, finalizer, recipe schema, probe, bootstrap, summariser, producers, GPU tests, notebook edits. Reviewed on an isolated `--shared` clone of the worktree under my scratchpad (HEAD `a950605`, the six files byte-identical to the worktree); CPU only (`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`); no tracked file modified, no process signalled, nothing written under `ckpt/`.

**Verdict: approve with changes.** No blocking findings. The round meets the plan's §2.2 encoder contract, the R3.1 symmetry test brief, the §4 heading rule (reproduced on the real training rows, and against my own independent re-implementation to 4e-7 dB), and the §3 non-modification rule. The findings below are provenance/robustness should-fixes and nits; S1 should land before the first heading JSON is produced because the record schema is frozen at write time.

---

## 1. Findings

### S1 — should-fix — the heading record's source-closure digest carries no git identity (Coder's "closure hashes bind current file bytes" resolution)
**Where.** `tools/exp06_heading.py:238-242, 255-256` (record), `:295-297` (validator requires `basis == 'working_tree'`).
**What.** The record stores `sha256_file` of each closure file *as it sits in the working tree* plus a digest over `[[path, sha256], …]`, `basis='working_tree'`, and the absolute repo path — but no `HEAD`, no dirty flag, no per-file reviewed-blob hash. I verified that a record produced from a tree with an uncommitted edit to `tools/exp06_heading.py` gets a different digest, **still validates**, and contains no field from which the dirtiness or the producing commit can be read (`probe_heading_edge.py §8`).
**Why it matters.** Plan §4 asks for "the tool's source-closure digest" and §6.4 makes the heading tool a producer whose `code` approval is a closure digest; exp_04/05 manifests record `git_state` (HEAD, dirty, diff hash) alongside every closure so a record can be traced to a commit. Here traceability is only *indirect*: I confirmed that on a clean tree the record's digest equals `tools.provenance.closure_record(files, 'HEAD', repo)[1]` (`8dcdc315fdb46957…` at `a950605`, identical JSON structure and file order), so matching it against an approved reviewed-blob digest does bind the *content* — but a mismatch cannot be diagnosed (dirty tree vs. wrong commit), and the record itself cannot be tied to a SHA.
**Minimal fix (≈ 12 lines + validator + test).** In `estimate_room_heading`: `state = git_state(repo)`; `records, head_digest = closure_record(files, state['HEAD'], repo)`; store `files=[{path, sha256: working_tree_sha256, head_blob_sha256: reviewed_blob_sha256}]`, keep `sha256` (working-tree digest, unchanged semantics), add `head_sha256=head_digest` and `git={'HEAD', 'dirty', 'dirty_outside_worklog', 'diff_sha256'}` from `git_state`. Validator: `git.HEAD` 40-hex, `dirty` bool, `head_sha256` 64-hex. Test: `git.HEAD == git rev-parse HEAD`, and `sha256 == head_sha256` iff clean. Do this before `ckpt/exp06/heading/*.json` exist (schema_version 1 is frozen on write; a later field addition would invalidate files already written).

### S2 — should-fix — CLI refusal and CLI usage/input errors share exit status 2
**Where.** `tools/exp06_heading.py:365-368`. `parser.error(...)` exits 2 (argparse convention); a *heading refusal* (evidence JSON written) also returns 2.
**What.** Verified: missing room dir → 2 (nothing written); `--heading-deg` without reason → 2; NaN heading → 2; unwritable `--out` → 2; genuine refusal → 2 (JSON written) (`probe_heading_edge.py §7`).
**Why it matters.** Round 2's `tools/exp06_haa_pipeline.sh` must stop a room on refusal and escalate (plan §4/§11); with the collision it cannot distinguish "refused, evidence on disk" from "tool misuse, nothing on disk" without parsing the JSON.
**Minimal fix.** Return a distinct status for refusal (e.g. `3`), document it in `main`'s docstring, assert `returncode == 3` in `test_cli[refuse]` and add one case asserting an argparse error is `2` with no file written.

### S3 — should-fix (round-2 consumer contract; helper can land now) — `read_heading_json` validates internal consistency only
**Where.** `tools/exp06_heading.py:349-351`, validator `:260-340`.
**What.** The validator recomputes the decision from `per_mic` and checks the closure digest against its own file list, so an inconsistent tamper is refused (verified: changed `k`, changed decision, 1e-9 change in the table, changed file hash — all refused). A *consistent* tamper (edit `per_mic`, recompute `estimator`, set `phi/k`) passes, as it must; the only guard is `input_sha256`, which nothing in round 1 compares to the cache files, and the closure digest is not compared to any approval.
**Why it matters.** Plan §6.4 assigns input revalidation to the producers/finalizer, so this is not a round-1 defect, but round 2's `exp06_haa_finetune`/`exp06_haa_eval` must not treat a heading JSON as trusted just because `read_heading_json` accepted it.
**Minimal fix.** Add `verify_heading_inputs(record, room_dir)` (compare the four `input_sha256` with `sha256_file` of the cache files; raise on mismatch) with a tamper test now, and require round 2 to call it plus the §6.4 digest check before using `k`.

### N1 — nit — malformed caches surface as tracebacks, not `parser.error`
`tools/exp06_heading.py:214-218, 365`: out-of-range `train` indices raise `IndexError`; a `meta.json` without `train`/`sr` raises `KeyError`; neither is in the `(OSError, ValueError)` catch. Fail-closed (no JSON), but noisy. Validate `train.max() < len(rirs)` and catch `KeyError`/`IndexError` in `main`.

### N2 — nit — `heading_roll_k` accepts `bool`; strings raise `TypeError`
`tools/exp06_heading.py:62-67`: `heading_roll_k(True) == 511` (True → 1.0°); `heading_roll_k('90')` raises `TypeError` rather than `ValueError`. `_width` already refuses bools — mirror that (`isinstance(phi_deg, bool) or not isinstance(phi_deg, numbers.Real)` → `ValueError`). Unreachable from the CLI (`type=float`).

### N3 — nit — `test_cli` depends on the ambient `PYTHONPATH`
`tests/test_exp06_heading.py:329-336`: the subprocess inherits the environment; without `PYTHONPATH` the CLI exits 1 (`ModuleNotFoundError: sim_to_real`). Pass `env={**os.environ, 'PYTHONPATH': str(repo)}`.

### N4 — nit — readback constants lag the frozen floor windows
`tests/test_exp06_heading.py:252-254` and plan §4 give hallway runner-up margins 36.8 / 22.6; with the frozen `N = int(w·sr)` windows the realised values are **36.905 / 22.586** (the round-3 review's numbers; 36.8 was the half-open-window value from round 2). The ±0.2 tolerance masks it; update the constant and plan text to 36.9 so the pre-registered numbers match the frozen definition.

### N5 — nit — hard-coded cache path in the readback test
`tests/test_exp06_heading.py:256`: use `sim_to_real.haa_dataset.DEFAULT_ROOT` (honours `HAA_XRIR_ROOT`) instead of the literal `/home/yixunhu/data_cache/HAA_xrir`.

### N6 — nit — non-atomic JSON write
`tools/exp06_heading.py:346`: `Path.write_text` directly; an interrupted write leaves a truncated file (which then fails validation — fail-closed, fine). `tools.provenance.write_manifest`'s temp-file + `os.replace` pattern would be consistent with the rest of the tooling.

### N7 — nit — joint-cancellation tolerance is generous
`tests/test_exp06_encoder.py:96, 99`: `atol=1e-4`; the realised float32 discrepancy is ≤ 1.4e-6 across 40 random headings/offsets at input scales up to 10 (float64: 3.6e-15). `1e-5` would still pass with 7× margin. Harmless either way — a wrong column offset produces ≫ 1e-2.

### N8 — nit — a silent training RIR aborts instead of writing a refusal record
`tools/exp06_heading.py:52-53`: zero early energy raises `ValueError` (CLI → `parser.error`, nothing written). Reasonable (a data defect is not an indecisive estimator) but record the semantics in the notebook so a future room with a dead channel is triaged as infrastructure, not as a §4 refusal.

### N9 — nit — `from __future__ import annotations` in `model/cylindrical_vit_oriented.py:2` is unused (no PEP 604/585 hints). Harmless.

---

## 2. What I verified

### 2.1 Replay and static checks (isolated clone, xRIR env, CPU)
- `pytest tests/test_exp06_encoder.py tests/test_exp06_factory.py tests/test_exp06_heading.py tests/test_provenance.py -q -p no:cacheprovider` → **112 passed, 3 skipped** (the three CUDA-only full-forward shape tests) in 61.8 s; `tests/test_provenance.py::test_exp03_closure_and_reviewed_digest_are_unchanged` green at HEAD `a950605` (exp_03 digest `5ba818d8…`, 12 files, no later commits).
- `python -m py_compile` on the six files: ok. `git diff --check 77ef292...HEAD`: clean.
- `git diff main...exp06-window --diff-filter=M --name-only`: **empty** (no tracked file modified; `--stat` lists the six new files only, +974).
- Commit sizes (adds + dels): 171, 88, 172, 127, 82, 152, 131, 43, 18 — all < 200. Every commit carries `Co-Authored-By: OpenAI Codex gpt-6-astra <noreply@openai.com>`. Tests land in the same commit as their implementation in all nine commits. The Coder log shows a collection-error "red" run before each implementation commit and two mid-cycle failing runs (3 failed / 1 failed) resolved before committing.
- Worktree HEAD `a950605`, `git status` clean; six files sha256-identical between worktree and clone.

### 2.2 Plan §2.2 encoder contract (`probe_symmetry.py`)
- `az_channels` == `stack(cos_theta, sin_theta)` broadcast (bit-exact) **and** == normalised horizontal look direction of `convert_equirect_to_camera_coord(ones)` to **1.19e-7** at 256×512, 32×64, 64×128, 16×512; `azimuth_channels()` bit-equal to the buffer; float64 variant 3e-7 vs a float64 camera conversion; buffer non-persistent (absent from `state_dict`), shape `[2, H, W]`.
- Parameters: parent 19 750 912, child 20 277 248, **delta 526 336**; state-dict keys identical; only `to_patch_embedding.1.{weight,bias}` (1536 → 2560) and `to_patch_embedding.2.weight` ([512,1536] → [512,2560]) change shape; strict reload of a child state dict works; strict load of a parent state dict is refused (expected, matches R3.3). `in_channels ∈ {1,2,4,5}` refused via the parent. Only `forward` is overridden; the parent's gauge-alignment, `pos_embedding` and `transformer(tokens, rel_index)` lines appear verbatim in the child.
- dtype/device: `.double()` propagates to `az_channels` (materialised contiguous), float64 forward ok, |f32 − f64| ≤ 2.7e-6 at default size; default-size forward `[2, 256, 512]` finite. CUDA propagation not exercised (no GPU in this review; the buffers move with `.cuda()` by construction — round-2 smoke covers it).

### 2.3 Symmetry statements (R3.1) under adversarial inputs
- **Scene-only active yaw** (`rotate_scene_yaw`, random patch-multiple `k`, random input scales 0.5–8, three sizes incl. default 256×512/depth 12, float32 and float64): parent tokens are a row permutation to **≤ 3.6e-6** (float32) / 2.4e-6 (float64); child tokens differ by **≥ 0.65** in every trial. Non-patch-multiple `k = 5` gives the parent no permutation (differences ≈ 1.7–3.4), consistent with "patch-level" equivariance.
- **Heading cancellation**: `(heading_roll_k(φ_c + j·360/W) + j) mod W == heading_roll_k(φ)` for 20 000 random `(φ ∈ U(−2000, 2000), j)` at W = 512 — **0 failures**; also 0/5000 at W ∈ {64, 100, 360, 1000}. Tensor level (40 random headings/offsets, scales up to 10): worst input difference 1.43e-6, worst token difference 1.34e-6 (float32); 3.6e-15 / 3.3e-15 (float64); **j = 0 bit-exact** for inputs and tokens in every case. Round-half-up ties: 0.3515625° → 0, 1.0546875° → 511, −0.3515625° → 1 (test parametrisation confirmed).
- **Mirror discrimination** (centred box, 8 random extents and sources): parent = exact 8-patch roll to 1.6e-6; child differs by ≥ 3.9.

### 2.4 Factory
`build_xrir_exp06('simple'|'cylindrical')` return the pinned classes by identity (`type(...) is xRIR / xRIR_Cyl`, `BACKBONES_EXP06[name] is BACKBONES[name]`), identical state dicts under the same seed, bit-identical `source_network` outputs on a fixed batch; `cylindrical_oriented` − `cylindrical` = 526 336 parameters; unknown name refused; `dim/depth/heads/mlp_dim` reach the encoder; the `lin_proj_0` token-count guard fires for a non-256-token geometry. (Tests replayed; the full xRIR forward needs CUDA and is skipped.)

### 2.5 Heading tool vs plan §4 (`probe_readback.py`, `probe_heading_edge.py`, `probe_dataset.py`)
- Windows `N = int(w·sr)` = **110 / 1102**; onset = first `|h| > 0.2·max|h|`; levels `10 log10(d²E)`; candidates `{+x:0, +y:90, −x:180, −y:−90}` with inclusive ±45° sectors (boundary mics count for both adjacent candidates, as "within ±45°" implies); evaluable ≥ 3 in / ≥ 3 out; decision = both-window agreement, contrast ≥ 3 dB (exactly 3.0 accepted), runner-up margin ≥ 3 dB (tie → margin 0 → refused), singleton allowed with `competitor_count 0` / `runner_up_margin_db null`; leave-one-out re-applies `decide_heading` (all thresholds) in all 12 refits; non-finite / mismatched / three-window inputs refused.
- **Real readback (training rows only, mmap; 2.8–3.1 s per room incl. the closure subprocess):**

| room | sectors (+x,+y,−x,−y) | decision | contrast 5 ms / 50 ms | runner-up margin | LOO | min LOO winning contrast |
|---|---|---|---|---|---|---|
| class_room | 2,1,2,7 | −y, k=128 | 12.921 / 9.036 | singleton | 12/12 −y | 8.39 dB |
| dampened_room | 4,0,3,5 | −y, k=128 | 11.388 / 8.669 (+x −7.37/−5.71, −x −6.03/−4.48) | 17.420 / 13.144 | 12/12 −y | 7.88 dB |
| hallway | 0,5,0,7 | −y, k=128 | 18.453 / 11.293 (+y −18.45/−11.29) | 36.905 / 22.586 | 12/12 −y | 9.83 dB |
| complex_room | 2,1,1,8 | −y, k=128 | 11.953 / 10.806 | singleton | 12/12 −y | 8.58 dB |

  All within 0.05 dB of the pre-registered numbers (hallway margin 36.905 vs the plan's 36.8 — see N4). My independent float64 re-implementation of §4 agrees with the tool to **≤ 3.6e-7 dB** on every contrast and ≤ 6.2e-7 dB on every per-mic level. Descriptive extras reproduce the round-1 review's ill-posedness (hallway continuous fit −44° at 50 ms, LOO range −147…−23°, mean direction −90.4°) — reported, not decisive.
- Record fields present: `phi_deg, k, decision, reason, override_*`, `estimator{winner, table(contrasts per candidate × window, competitor_count, runner_up_margin_db), leave_one_out_winners, leave_one_out_stable}`, `descriptive` (both windows), `per_mic`, `input_sha256` of the four inputs (equal to `sha256_file`), `source_closure` (6 files: `sim_to_real/haa_dataset.py, tools/exp06_heading.py, tools/provenance.py, tools/yaw_rotation.py, treble_multi_room_dataset/treble_xRIR_dataset.py, utils/spec_utils.py`; digest `8dcdc315fdb46957ba47117894e45b86f8644c358cd20ca1a55cc27061767fe2` at `a950605`, equal to `closure_record(files, 'a950605', repo)`), tz-aware timestamp. Refusal → `phi_deg = k = null`, evidence retained, JSON round-trips; override → decision `override`, estimator evidence retained, blank reason refused; `HeadingFrameDataset` refuses `None`, floats, 512, −1 (fail-closed), accepts `np.int64(128)`.
- Training-rows-only access: the test's `np.load` guard (mmap required, exactly `arange(12)` indexed once) replayed green; `sha256_file(rirs.npy)` streams the whole file for the hash only.
- **`HeadingFrameDataset` on the real hallway cache** (423 test queries, `eval_seed 0`, 47 sampled): `.items` identical; `.data` tensors identical; `_pick_refs` is the parent's method and draws identical; **k = 0 bit-identical** to `HAADataset`; **k = 128 tuple == `rotate_scene_yaw`** on the unsqueezed tuple; `tgt_wav`/`ref_irs`/`listener` untouched; `side_label` equals the room-frame sign of y and every −y mic has x′ > 0 after k = 128 (heading → +x; `Rz(k=128)(0,−1,0) = (1, −6e-17, 0)`); side split **185 +y / 238 −y** as in exp_02's notebook. Train-split draws remain global-RNG (parent behaviour).

### 2.6 Commands (all with `PYTHONPATH=<clone> XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=<scratch>`, interpreter `/home/yixunhu/miniconda3/envs/xRIR/bin/python`)
`git clone -q --shared -b exp06-window …_wt <scratch>/wt_review`; `git log --oneline main..HEAD`; `git diff main...HEAD --stat / --diff-filter=M --name-only / --check`; `git log --format='%(trailers)'`; `git show --shortstat` per commit; `python -m py_compile <6 files>`; `python -m pytest <4 test files> -q -p no:cacheprovider -rs`; scratch probes `probe_readback.py`, `probe_symmetry.py`, `probe_dataset.py`, `probe_heading_edge.py` (the last one temporarily appended a comment to `tools/exp06_heading.py` **in the scratch clone only** to demonstrate S1, then restored it byte-for-byte; verified). Files read: SOP; plan v4 §1–§14; the three Codex plan reviews; notebook; query/command/commits files; round-1 Coder prompt and the Coder log's final report; exp_02 notebook 2026-09-14 entries and `diagnostics_2026-09-14_hallway_side/gt_sides.py`; exp_03 analysis (equivariance/pool paragraphs); `model/{cylindrical_vit,xRIR_cyl}.py`, `model/xRIR.py` (constructor/forward), `tools/yaw_rotation.py`, `treble_xRIR_dataset.convert_equirect_to_camera_coord`, `sim_to_real/haa_dataset.py`, `tools/provenance.py` (`sha256_file`, `source_closure`, `closure_record`, `git_state`), `tests/test_provenance.py` (pinned-closure test).

### 2.7 Not verified here
CUDA device propagation and the full `xRIR` forward with the oriented encoder (3 tests skipped; pinned `apply_delay` is `.cuda()`-only) — ladder rung 4 (round 2 smokes). Round 2/3 contracts (approvals, consumers) — out of scope; S3 records what they must do with this round's JSON.

---

## 3. Verdict

**Approve with changes.** The round implements plan v4 §8 round 1 faithfully; all contract, symmetry and readback claims survived adversarial checks; hygiene is clean. Fix S1 and S2 in a small follow-up commit (with tests) before any `ckpt/exp06/heading/*.json` is produced; S3 is a contract for round 2 (helper may be added now). Nits N1–N9 may be batched into round 2.

**Blocking findings the Coder must fix before the round closes: none.**
