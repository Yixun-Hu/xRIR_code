# exp_08 commits — worktree `/home/yixunhu/codespace/xRIR_code_wt08`, branch `exp08-invariant-readout`

Local only; nothing pushed. Branch point: `5c53ee1` (main, "exp_07 worklog: exp_06 pretraining
launched on GPU 1 (6dc0b8e, 12:14); GPU-0 plan").

| SHA | one-liner |
|---|---|
| `f033c8d` | exp_08 Stage A: `xRIR_CylInvariant` -- one forward pass, C16-invariant RIR (structure validated, CPU, no training). Adds `model/xRIR_cyl_invariant.py`, `model/exp08_factory.py`, `model/exp08_delay.py`, `tools/exp08_warmstart.py`, `tools/exp08_validate.py`, `tests/test_exp08_invariant.py` and this worklog (plan, validation report, measured JSON + log). |
| `dee22a6` | exp_08 worklog: record the Stage A commit SHA, what it touched, and the verification commands. |
| `0dbc31e` | **Codex review round 1 fixes**: B1 -- replace the max-radius *selection* in the degenerate basis fallback with the normalised **vector sum** of the references' horizontal components (equivariant, continuous, order-independent, tie-free); B2 -- reduce the distance-to-integer-delay computation in **float64** on the invariant arms and **qualify** the C16 statement with the measure-zero rounding-boundary exception, its measured size and the adversarial case. 27 new tests (222 total). |
| (this file's own commit) | exp_08 worklog: record the review-round-1 SHA. A commit cannot contain its own hash, so read it off `git log --oneline 5c53ee1..HEAD`. |

Full message of `f033c8d`: `git -C /home/yixunhu/codespace/xRIR_code_wt08 show -s f033c8d`.

## What each commit touched

`f033c8d` adds **only new files** -- no line of the pinned `model/xRIR.py`,
`model/xRIR_cyl.py`, `model/cylindrical_vit.py`, `model/simple_vit.py`,
`train_xRIR_backbone.py`, `eval_*.py` or any exp_03/05/06/07 tool was modified
(`git show --stat f033c8d` lists 11 additions, 0 modifications, 0 deletions).  `0dbc31e`
modifies only files `f033c8d` created; across the whole branch
`git diff --name-status 5c53ee1..HEAD` reports **`A` for every path and nothing else**.  The pinned
`simple` / `cylindrical` entries are re-exported by object identity, and a test asserts both
that identity and that their predictions on a real batch are bit-identical to the classes'.

## Verification commands

```bash
cd /home/yixunhu/codespace/xRIR_code_wt08
export PYTHONPATH=$(pwd) XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
PY=~/miniconda3/envs/xRIR/bin/python

# the section-4 checklist as tests (222 passed, 254 s)
$PY -m pytest tests/test_exp08_invariant.py -q

# the measured report (333 s) -> exp08_validation.{json,log}
$PY -m tools.exp08_validate \
    --out-json worklog/worklog_yixun/exp_08_invariant_readout_claude/exp08_validation.json

# the warm-start accounting on its own
$PY -m tools.exp08_warmstart --backbone cylindrical_invariant
$PY -m tools.exp08_warmstart --backbone simple_invariant

# nothing of the pre-existing suite regressed (102 passed, 18 GPU-skipped)
$PY -m pytest tests/test_exp06_factory.py tests/test_exp06_encoder.py \
    tests/test_yaw_rotation.py tests/test_per_sample_metrics.py \
    tests/test_reference_manifest.py tests/test_eval_yaw_rotation.py -q
```

Nothing outside this worktree was written. The pinned checkpoints
(`/home/yixunhu/codespace/xRIR_code/ckpt/xRIR_{simple,cyl}_8_shot/epoch_12.pth`) and the exp_03
reference manifest (`.../ckpt/yaw_rotation/reference_manifest.json`) were read through absolute
paths and never modified; no `ckpt/` directory anywhere was touched; no GPU was used
(`CUDA_VISIBLE_DEVICES=""` throughout).
