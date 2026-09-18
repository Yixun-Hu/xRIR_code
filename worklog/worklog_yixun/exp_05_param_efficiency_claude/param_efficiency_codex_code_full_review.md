**Reviewer:** OpenAI Codex (GPT-6, API agent; read-only source review with isolated scratch execution) · **Date:** 2026-09-18

Round-8 close review of `82e0f56`, `078548f`, `0d2bebd`, `377cc2c`, plus the integrative **`full`** review of `exp05-record` against main `64f3ad7`.

**Round-8 verdict: approve. Integrative verdict: approved for merge and record use. Exact blocking list: none.**

1. **Blocker from round 7 — resolved.** [bind_provenance.py:96](/home/yixunhu/codespace/xRIR_code_wt05/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets/bind_provenance.py:96), [attempt_record:234](/home/yixunhu/codespace/xRIR_code_wt05/worklog/worklog_yixun/exp_05_param_efficiency_claude/param_efficiency_results_assets/bind_provenance.py:234). Resolving an attempt previously moved sibling lookups and dependency identities to the archive location, invalidating unchanged evidence. The implemented fix preserves logical names, obtains the attempt name from the completion-bound launcher manifest, and checks checkpoint/directory identity with `samefile`. The inherited digest and output-containment checks remain active. **Minimal fix: implemented; no further change required.** No new blocking, should-fix, or nit findings in the reviewed branch.

**Verified**

- **Relocation:** unchanged `check_record` and fresh `collect` reproduce the identical report after moving an attempt, a whole arm, `ckpt/exp05` including its products, products alone, one evaluation, or the evaluation parent. Copying all four attempts to new inodes also passes. Original paths, `final` CLI paths, and an archive-path attempt argument produce the same launcher-named attempt. Regenerated documents retain logical paths; relocated-product figures reproduce the original PNGs.
- **Refusals:** changed relocated bytes, a repointed `final` even with identical evidence and a hardlinked checkpoint, an alternate attempt directory with that hardlinked checkpoint, and an escaping output symlink are refused. All **23** committed binder forgeries pass their refusal tests, including the repaired manifest-name forgery.
- **Round-7 protections:** independently replayed **216 document + 112 figure overwrite refusals**, then repeated both matrices after relocation; all input bytes unchanged. Eight direct figure collisions also refused. Actual producer-source drift, repaired evaluator/writer swaps, forged closures, K=123, missing/foreign runs, and changed bound throughput/loss remain refused. Explicit legacy-M admission remains verified only at helper level.
- **Presentation:** all twelve HTML/Markdown tables agree and numerical fields trace to canonical products or digest-checked attempt evidence. Both pages cite all nine required digests; M rows use exp_05 evaluations. Unequal EDT/C50/T60 cohorts remain **12/11/8 queries**, **3/3/2 rooms**, **0/1/4 exclusions per seed**, for both K families. H1/H2 cohort fields and intervals agree; inline SVG colours retain precedence. All four live PNGs are byte-identical to round 7. H1 remains K8 **partial/partial**, K1 **not supported/not supported**.
- **Live record:** read-only MAIN-tree bind passed in **429.9 s**, reproducing round 7's unchanged rendered-input report field for field except `git_HEAD`; independent checker passed in **429.8 s**. Coverage: **66 evaluations, four trainings, five ledger-listed probes, two historical M checkpoints, five products, two documents, eight figures**. Separately exercised all four real attempts through the Planner's `final` routes and obtained exactly the prior attempt records. Fresh documents and figures were generated only in scratch.
- **Fresh tests:** `static_checks.sh` **exit 0**; **94 record tests passed**. Both complete collection orders: **315 passed / 1 skipped** each. The initial forward run skipped three missing-sample checks; it was rerun with read-only access to the historical sample. Rehearsed merged-tree record tests: **94 passed**. Coder's completed full-suite transcript records **3,199 passed / 49 skipped / 2 failed**; this was inspected, not rerun. On unmerged current main, exp_06's old failure now passes after its approval update; exp_07's null `seen_simple` pin still fails independently of this branch.
- **Integration/hygiene:** `git merge --no-ff --no-commit 377cc2c` onto main `d0b9f00`, refreshed onto `64f3ad7`, is conflict-free with exactly **ten additions**. The intervening main changes are record-only. Merge-base scope and whitespace checks pass; all twelve pinned exp_03 files retain empty history after `62c9107`. Python **3.8.20**, CPU only, scratch-only execution writes; live evidence and reviewed sources unchanged.

`provenance.revalidate` is unaffected, and the launcher's internal-symlink refusal still holds. Exp_07's existing relocation defect was reproduced in a synthetic helper check; exp_04's dependency-comparison defect is confirmed by source inspection. Those unchanged binders require separate fixes before equivalent migrations of their records.

After merging, run the Planner's sequence from MAIN (the five canonical producer outputs already exist):

```bash
set -euo pipefail
cd /home/yixunhu/codespace/xRIR_code
export PYTHONPATH=$PWD XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
PY=/home/yixunhu/miniconda3/envs/xRIR/bin/python
W=worklog/worklog_yixun/exp_05_param_efficiency_claude
A=$W/param_efficiency_results_assets
R=ckpt/exp05/results
ATTEMPTS=(ckpt/exp05/{S_simple,S_cylindrical,L_simple,L_cylindrical}/final)
JSONS=("$R"/{CURVE_K8,CURVE_K1,TARGETS_K8,TARGETS_K1,YAW_K8_SEED42}.json)
FLAGS=(--curve-k8 "${JSONS[0]}" --curve-k1 "${JSONS[1]}"
       --targets-k8 "${JSONS[2]}" --targets-k1 "${JSONS[3]}"
       --yaw-k8 "${JSONS[4]}")

"$PY" "$A/make_results_md.py" "${FLAGS[@]}" --attempt "${ATTEMPTS[@]}" --out "$W/param_efficiency_results.md"
"$PY" "$A/make_results_html.py" "${FLAGS[@]}" --attempt "${ATTEMPTS[@]}" --out "$W/param_efficiency_01_results.html"
mkdir -p "$W/param_efficiency_figures" ckpt/exp05/reports
"$PY" "$A/make_figures.py" --curve "${JSONS[0]}" "${JSONS[1]}" --outdir "$W/param_efficiency_figures"
"$PY" "$A/bind_provenance.py" --runs ckpt/exp05/eval/* \
  --attempt "${ATTEMPTS[@]}" --results "${JSONS[@]}" \
  --rendered "$W/param_efficiency_results.md" "$W/param_efficiency_01_results.html" \
  --figures "$W"/param_efficiency_figures/* --out ckpt/exp05/reports
"$PY" "$A/check_record.py" ckpt/exp05/reports
```

Capture the sequence in the timestamped record-run log, as the Planner's script does. After a successful check, commit the two generated documents, all eight PNG/PDF files, the run log, this review, and updated command/worklog/commits records. Preserve the timestamped binding report under `ckpt/exp05/reports`.

**Yes: the four `ckpt/exp05/*/attempt_*` directories may be moved to the NAS AFTER binding without breaking `check_record`.** Preserve every byte, leave directory symlinks at the original logical attempt paths, and retain the local arm ledgers, probe receipts, sibling attempts and `final` links. Verify the unchanged report after moving; no rebind is needed for the relocation alone.

[Commands, reproductions and evidence limits](/tmp/exp05-round8-review-Zxaa7R/review_notes.md).
