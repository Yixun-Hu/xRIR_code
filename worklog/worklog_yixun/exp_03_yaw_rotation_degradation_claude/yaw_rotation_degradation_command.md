# Commands — yaw_rotation_degradation

All from the repo root with `export PYTHONPATH=$PYTHONPATH:$(pwd); export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms`; `python` = `/home/yixunhu/miniconda3/envs/xRIR/bin/python`. Every evaluator run below is bound post hoc (binder v6, `_results_assets/bind_provenance.py`) to the 12-file source closure (`eval_yaw_rotation.py` plus its 11 repo-local imports) byte-identical to commit `62c9107b4150e44c4ac410ff4cab359c1e71cc10` (source-closure digest `5ba818d83eddc6e71055926ea64cb104ebb8d3a1c347ab1bc12dd95866c1be48`); the binding (closure, environment, content-hashed dataset inventory, log digests, chronology from log names and mtimes) proves the consistency of the retained bytes with the reviewed commit, not execution-time identity — a residual limitation recorded in the analysis. The launcher HEAD at each launch is given per section (later commits touched only `tools/summarize_yaw.py`, `tests/` and the record folder). Variables used below:

```bash
E=worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude
A=$E/yaw_rotation_degradation_results_assets
H0=47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d   # manifest seed 0
H1=6ca8164e5727d5f4db84569d5effa840b2cb6e6f77819a69bbb37bb30c6ab11e   # manifest seed 1
H2=757e5a966ee2d069a07accecbc43e46bd564ea7c788a28c219773df64e986234   # manifest seed 2
STD="--batch-size 16 --num-workers 6 --threads 4 --log-interval 10"
K0="--yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols --decomposition-batches 0"
CTRL="--backbone simple --checkpoint ckpt/xRIR_simple_8_shot/epoch_12.pth"
CYL="--backbone cylindrical --checkpoint ckpt/xRIR_cyl_8_shot/epoch_12.pth"
REL="--backbone simple --checkpoint checkpoints/xRIR_unseen.pth"
EXP01="--exp01-per-sample control=ckpt/xRIR_simple_8_shot/per_sample_unseen_epoch12.json cyl=ckpt/xRIR_cyl_8_shot/per_sample_unseen_epoch12.json"
GATE3="--runs ckpt/yaw_rotation/gate_control ckpt/yaw_rotation/gate_cyl ckpt/yaw_rotation/gate_released"
SWEEP="--runs ckpt/yaw_rotation/sweep_control ckpt/yaw_rotation/sweep_cyl ckpt/yaw_rotation/sweep_released"
```

## Reference manifests (seed 0 pinned 2026-09-06T12:36:37-04:00; seeds 1–2 built 2026-09-06T13:42:54-04:00; HEAD `62c9107b4150e44c4ac410ff4cab359c1e71cc10`)
```bash
python -c "from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset; from tools.reference_manifest import build_manifest, save_manifest, manifest_hash; ds=xRIR_Dataset(split='test', max_len=9600, num_shot=8); m=build_manifest(ds, seed=0, num_shot=8); save_manifest(m, 'ckpt/yaw_rotation/reference_manifest.json'); print(manifest_hash(m))"          # -> $H0
python -c "from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset; from tools.reference_manifest import build_manifest, save_manifest, manifest_hash; ds=xRIR_Dataset(split='test', max_len=9600, num_shot=8); m=build_manifest(ds, seed=1, num_shot=8); save_manifest(m, 'ckpt/yaw_rotation/reference_manifest_seed1.json'); print(manifest_hash(m))"    # -> $H1
python -c "from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset; from tools.reference_manifest import build_manifest, save_manifest, manifest_hash; ds=xRIR_Dataset(split='test', max_len=9600, num_shot=8); m=build_manifest(ds, seed=2, num_shot=8); save_manifest(m, 'ckpt/yaw_rotation/reference_manifest_seed2.json'); print(manifest_hash(m))"    # -> $H2
```

## Validation-ladder smoke (2026-09-06T13:23:49-04:00, GPU 1, HEAD `62c9107b4150e44c4ac410ff4cab359c1e71cc10`)
```bash
CUDA_VISIBLE_DEVICES=1 python eval_yaw_rotation.py $CYL --manifest ckpt/yaw_rotation/reference_manifest.json --manifest-hash $H0 --out-dir ckpt/yaw_rotation/_smoke_cyl_32q --max-samples 32 $STD   # log yaw_rotation_degradation_2026-09-06_13:23:49_smoke_cyl_32q.log
```

## k=0 gate runs (launched 2026-09-06T14:17:27-04:00 by `$A/launch_yaw_sweep.sh gate`; HEAD `62c9107b4150e44c4ac410ff4cab359c1e71cc10`)
```bash
# GPU 0 chain, then GPU 1 chain (each sequential). Sidecar "command" fields: the five launcher-v1 runs (gate_control, gate_cyl, gate_released, gate_noise_seed1, gate_noise_seed2) carry the v1 template with a `<0|1>` GPU placeholder; the five later gate runs (phase_seed1, tf32, noise_cyl_seed1, noise_cyl_seed2, shape_cyl_b1) have sidecars written post hoc without a command field — their commands below are reconstructions from the worklog and the logs, not verbatim launcher output (limitation recorded in binding_report.json `command_note`)
CUDA_VISIBLE_DEVICES=0 python eval_yaw_rotation.py $CTRL --manifest ckpt/yaw_rotation/reference_manifest.json       --manifest-hash $H0 --out-dir ckpt/yaw_rotation/gate_control     $STD $K0
CUDA_VISIBLE_DEVICES=0 python eval_yaw_rotation.py $REL  --manifest ckpt/yaw_rotation/reference_manifest.json       --manifest-hash $H0 --out-dir ckpt/yaw_rotation/gate_released    $STD $K0
CUDA_VISIBLE_DEVICES=0 python eval_yaw_rotation.py $CTRL --manifest ckpt/yaw_rotation/reference_manifest_seed1.json --manifest-hash $H1 --out-dir ckpt/yaw_rotation/gate_noise_seed1 $STD $K0
CUDA_VISIBLE_DEVICES=1 python eval_yaw_rotation.py $CYL  --manifest ckpt/yaw_rotation/reference_manifest.json       --manifest-hash $H0 --out-dir ckpt/yaw_rotation/gate_cyl         $STD $K0
CUDA_VISIBLE_DEVICES=1 python eval_yaw_rotation.py $CTRL --manifest ckpt/yaw_rotation/reference_manifest_seed2.json --manifest-hash $H2 --out-dir ckpt/yaw_rotation/gate_noise_seed2 $STD $K0
# gate decision v1 (2026-09-06T14:53:35-04:00; HEAD b9884c740dd8d63d7a3cf92acf671cf4eafafa4f): FAILED 2/11 -> yaw_rotation_degradation_2026-09-06_14:53:35_gate_decision_v1_FAILED.txt
python tools/summarize_yaw.py --mode k0-gate --band-rule v1 $GATE3 --manifest-hash $H0 $EXP01 --noise-runs ckpt/yaw_rotation/gate_noise_seed1 ckpt/yaw_rotation/gate_noise_seed2 --released-baseline 0.0549,1.358,9.69 --json ckpt/yaw_rotation/gate_stats.json --summary ckpt/yaw_rotation/gate_summary.txt   # exit 1 (at the time --band-rule did not exist and v1 was the only rule; pass --band-rule v1 to reproduce with the current code)
```
Logs: `yaw_rotation_degradation_2026-09-06_14:18:14_gate_control.log`, `yaw_rotation_degradation_2026-09-06_14:18:14_gate_cyl.log`, `yaw_rotation_degradation_2026-09-06_14:27:39_gate_released.log`, `yaw_rotation_degradation_2026-09-06_14:28:00_gate_noise_seed2.log`, `yaw_rotation_degradation_2026-09-06_14:36:58_gate_noise_seed1.log`.

## Gate band-v2 nuisance runs (launched 2026-09-06T14:54:22-04:00; HEAD `b9884c740dd8d63d7a3cf92acf671cf4eafafa4f`)
```bash
CUDA_VISIBLE_DEVICES=0 python eval_yaw_rotation.py $CTRL --manifest ckpt/yaw_rotation/reference_manifest.json --manifest-hash $H0 --out-dir ckpt/yaw_rotation/gate_phase_seed1 $STD $K0 --gl-seed 1   # log yaw_rotation_degradation_2026-09-06_14:53:35_gate_phase_seed1.log
CUDA_VISIBLE_DEVICES=1 python eval_yaw_rotation.py $CTRL --manifest ckpt/yaw_rotation/reference_manifest.json --manifest-hash $H0 --out-dir ckpt/yaw_rotation/gate_tf32        $STD $K0 --tf32        # log yaw_rotation_degradation_2026-09-06_14:53:35_gate_tf32.log
# gate decision v2 (2026-09-06T15:07:30-04:00; HEAD a1db897dd0bfc96c3263f894683b250458ca3c5f): FAILED 1/11 -> yaw_rotation_degradation_2026-09-06_15:07:30_gate_decision_v2_FAILED.txt
python tools/summarize_yaw.py --mode k0-gate --band-rule v2 $GATE3 --manifest-hash $H0 $EXP01 --noise-runs ckpt/yaw_rotation/gate_noise_seed1 ckpt/yaw_rotation/gate_noise_seed2 --nuisance-runs ckpt/yaw_rotation/gate_phase_seed1 ckpt/yaw_rotation/gate_tf32 --released-baseline 0.0549,1.358,9.69 --json ckpt/yaw_rotation/gate_stats.json --summary ckpt/yaw_rotation/gate_summary.txt   # exit 1 (v2 was the default at the time; pass --band-rule v2 to reproduce)
```

## Gate band-v3 runs (launched 2026-09-06T15:07:30-04:00; HEAD `a1db897dd0bfc96c3263f894683b250458ca3c5f`)
```bash
CUDA_VISIBLE_DEVICES=0 python eval_yaw_rotation.py $CYL --manifest ckpt/yaw_rotation/reference_manifest_seed1.json --manifest-hash $H1 --out-dir ckpt/yaw_rotation/gate_noise_cyl_seed1 $STD $K0                                                        # log yaw_rotation_degradation_2026-09-06_15:07:30_gate_noise_cyl_seed1.log
CUDA_VISIBLE_DEVICES=0 python eval_yaw_rotation.py $CYL --manifest ckpt/yaw_rotation/reference_manifest_seed2.json --manifest-hash $H2 --out-dir ckpt/yaw_rotation/gate_noise_cyl_seed2 $STD $K0                                                        # log yaw_rotation_degradation_2026-09-06_15:17:21_gate_noise_cyl_seed2.log
CUDA_VISIBLE_DEVICES=1 python eval_yaw_rotation.py $CYL --manifest ckpt/yaw_rotation/reference_manifest.json       --manifest-hash $H0 --out-dir ckpt/yaw_rotation/gate_shape_cyl_b1 --batch-size 1 --num-workers 6 --threads 4 --log-interval 100 $K0   # log yaw_rotation_degradation_2026-09-06_15:07:30_gate_shape_cyl_b1.log
# gate decision v3 (2026-09-06T15:40:32-04:00; HEAD 5febf471318e37e60477ebe72afd48f03879c67e): PASSED 11/11 -> yaw_rotation_degradation_2026-09-06_15:40:32_gate_decision_v3_PASSED.txt (byte-identical to ckpt/yaw_rotation/gate_summary.txt)
python tools/summarize_yaw.py --mode k0-gate --band-rule v3 $GATE3 --manifest-hash $H0 $EXP01 --noise-runs ckpt/yaw_rotation/gate_noise_seed1 ckpt/yaw_rotation/gate_noise_seed2 --noise-runs-cyl ckpt/yaw_rotation/gate_noise_cyl_seed1 ckpt/yaw_rotation/gate_noise_cyl_seed2 --nuisance-runs ckpt/yaw_rotation/gate_phase_seed1 ckpt/yaw_rotation/gate_tf32 --shape-runs ckpt/yaw_rotation/gate_shape_cyl_b1 --released-baseline 0.0549,1.358,9.69 --json ckpt/yaw_rotation/gate_stats.json --summary ckpt/yaw_rotation/gate_summary.txt   # exit 0, gate_pass true
```

## Full sweep (launched 2026-09-06T15:41:49-04:00 by `nohup $A/launch_yaw_sweep_v2.sh sweep`; HEAD `5febf471318e37e60477ebe72afd48f03879c67e`; SWEEP_DONE 19:41:20)
```bash
# exact strings from the provenance.json "command" fields; GPU 0 ran control then released, GPU 1 ran cyl
CUDA_VISIBLE_DEVICES=0 /home/yixunhu/miniconda3/envs/xRIR/bin/python eval_yaw_rotation.py --backbone simple      --checkpoint ckpt/xRIR_simple_8_shot/epoch_12.pth --manifest ckpt/yaw_rotation/reference_manifest.json --manifest-hash 47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d --out-dir ckpt/yaw_rotation/sweep_control  --batch-size 16 --num-workers 6 --threads 4 --log-interval 10   # 120.8 min, log yaw_rotation_degradation_2026-09-06_15:41:19_sweep_control.log
CUDA_VISIBLE_DEVICES=0 /home/yixunhu/miniconda3/envs/xRIR/bin/python eval_yaw_rotation.py --backbone simple      --checkpoint checkpoints/xRIR_unseen.pth           --manifest ckpt/yaw_rotation/reference_manifest.json --manifest-hash 47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d --out-dir ckpt/yaw_rotation/sweep_released --batch-size 16 --num-workers 6 --threads 4 --log-interval 10   # 119.0 min, log yaw_rotation_degradation_2026-09-06_17:42:11_sweep_released.log
CUDA_VISIBLE_DEVICES=1 /home/yixunhu/miniconda3/envs/xRIR/bin/python eval_yaw_rotation.py --backbone cylindrical --checkpoint ckpt/xRIR_cyl_8_shot/epoch_12.pth    --manifest ckpt/yaw_rotation/reference_manifest.json --manifest-hash 47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d --out-dir ckpt/yaw_rotation/sweep_cyl      --batch-size 16 --num-workers 6 --threads 4 --log-interval 10   # 125.5 min, log yaw_rotation_degradation_2026-09-06_15:41:19_sweep_cyl.log
```

## Post-sweep, first pass (2026-09-06 19:42–19:55; HEAD `5febf471318e37e60477ebe72afd48f03879c67e`) — superseded by the sections below, kept for the record
```bash
python $A/bind_provenance.py --runs ckpt/yaw_rotation/sweep_control ckpt/yaw_rotation/sweep_cyl ckpt/yaw_rotation/sweep_released     # binder v1 (evaluator file only): BIND OK
python $A/check_sweep_acceptance.py --runs ckpt/yaw_rotation/sweep_control ckpt/yaw_rotation/sweep_cyl ckpt/yaw_rotation/sweep_released --manifest-hash $H0   # checker v1: ACCEPTANCE PASS
python tools/summarize_yaw.py --mode full $SWEEP --manifest-hash $H0 --n-boot 20000 --json ckpt/yaw_rotation/full_stats.json --summary ckpt/yaw_rotation/full_summary.txt   # v1 canonical: exit 0 after 426 s; log yaw_rotation_degradation_2026-09-06_19:43:30_full_summary.log (run detached with nohup after the harness killed a first attempt on a false low-memory signal)
python $A/make_results_md.py --stats ckpt/yaw_rotation/full_stats.json; python $A/make_results_html.py --stats ckpt/yaw_rotation/full_stats.json
```

## After the Codex round-2 review (2026-09-06 20:16–20:41; summarizer commits `7f3532ccb617df56109697f5de788bbb9b5247c2`, `631050da6796f4b21818bffe5b972f590c47d561`, `cfc5e4f8bcc776b9cd396fd0af14578f986df1a6`)
```bash
python $A/bind_provenance.py --runs ckpt/yaw_rotation/sweep_control ckpt/yaw_rotation/sweep_cyl ckpt/yaw_rotation/sweep_released 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_20:20:18_bind_provenance.log     # binder v2 (12-file closure): BIND OK
python $A/check_sweep_acceptance.py --runs ckpt/yaw_rotation/sweep_control ckpt/yaw_rotation/sweep_cyl ckpt/yaw_rotation/sweep_released 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_20:20:42_acceptance.log   # checker v2: ACCEPTANCE PASS (69 checks)
python tools/summarize_yaw.py --mode full $SWEEP --manifest-hash $H0 --n-boot 20000 --json ckpt/yaw_rotation/full_stats.json --summary ckpt/yaw_rotation/full_summary.txt   # v2 canonical (labels, TOST words, h1.bounds; numbers unchanged): exit 0 after 458 s; log yaw_rotation_degradation_2026-09-06_20:32:16_full_summary_v2.log
python $A/make_results_md.py --stats ckpt/yaw_rotation/full_stats.json; python $A/make_results_html.py --stats ckpt/yaw_rotation/full_stats.json
cp ckpt/yaw_rotation/full_summary.txt $E/yaw_rotation_degradation_full_summary.txt; cp ckpt/yaw_rotation/full_stats.json $A/full_stats.json
```

## After the Codex round-3 review (2026-09-06 21:03–21:27; summarizer schema commits `13fb539`, `1413439`, `bfea00d`, `4fbcc026072b781beff17551793bb6b8f6cc6ed8`; full SHAs in `commits_yaw_rotation_degradation.md`)
```bash
python $A/bind_provenance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_21:06:20_bind_provenance_v3.log   # binder v3: all 13 runs (3 sweep + 10 gate) by exact path, full reviewed SHA, environment incl. torchvision/einops, data identity (manifest file sha256 + inventory of 13 318 referenced files); BIND OK; report ckpt/yaw_rotation/binding_report.json (idempotent: a rerun prints "report unchanged"); three earlier attempts aborted on tool bugs in the derived data paths (logs yaw_rotation_degradation_2026-09-06_21:03:45_bind_provenance_v3_ABORTED_toolbug_depthpath.log, yaw_rotation_degradation_2026-09-06_21:04:38_bind_provenance_v3_ABORTED_toolbug_depthpath.log, yaw_rotation_degradation_2026-09-06_21:05:14_bind_provenance_v3_ABORTED_toolbug_datapath.log)
cp ckpt/yaw_rotation/binding_report.json $A/binding_report.json
python $A/check_sweep_acceptance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_21:10:11_acceptance_v3.log   # checker v3 (independent band-v3 gate validation, manifest order, finite content, reconciliation, P==E at k=0, provenance depth): ACCEPTANCE PASS
python -m pytest tests/test_exp03_record_tools.py -q      # adversarial validator tests: 9 passed
$A/launch_yaw_sweep_v3.sh sweep; $A/launch_yaw_sweep_v3.sh gate   # self-tests: closure ok, gate independently validated, atomic REFUSED, nothing started (logs yaw_rotation_degradation_2026-09-06_21:12:27_sweep_launcher_v3r2_refuse_selftest.log, yaw_rotation_degradation_2026-09-06_21:12:27_gate_launcher_v3r2_refuse_selftest.log, yaw_rotation_degradation_2026-09-06_21:12:43_sweep_launcher_v3r2_refuse_selftest.log, yaw_rotation_degradation_2026-09-06_21:13:00_gate_launcher_v3r2_refuse_selftest.log)
python tools/summarize_yaw.py --mode full $SWEEP --manifest-hash $H0 --n-boot 20000 --json ckpt/yaw_rotation/full_stats.json --summary ckpt/yaw_rotation/full_summary.txt   # v3 canonical (schema fields; numbers and summary text unchanged): exit 0 after 473 s; log yaw_rotation_degradation_2026-09-06_21:17:53_full_summary_v3.log
python $A/make_results_md.py --stats ckpt/yaw_rotation/full_stats.json; python $A/make_results_html.py --stats ckpt/yaw_rotation/full_stats.json
cp ckpt/yaw_rotation/full_summary.txt $E/yaw_rotation_degradation_full_summary.txt; cp ckpt/yaw_rotation/full_stats.json $A/full_stats.json; cp ckpt/yaw_rotation/binding_report.json $A/binding_report.json
```
python $A/check_sweep_acceptance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_21:27:38_acceptance_v3_final.log   # ACCEPTANCE PASS (final v3 rerun after every change)

## After the Codex round-4 review (2026-09-06 21:44–22:10; rule-string commits `9cb5127`, `f7a0c0a`, `a938467`; full SHAs in `commits_yaw_rotation_degradation.md`)
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
E=worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude; A=$E/yaw_rotation_degradation_results_assets
SWEEP="--runs ckpt/yaw_rotation/sweep_control ckpt/yaw_rotation/sweep_cyl ckpt/yaw_rotation/sweep_released"; H0=47637a55ccc594a32c35362f970e25296e352ccc81778f9523ce882ff930153d
python $A/bind_provenance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_21:45:06_bind_provenance_v4.log   # binder v4: content-hashed inventory (13 318 files, 1 082 499 862 bytes), run logs, gate inputs (exp_01 per-sample files, gate JSON/summary, decision copies) and the gate producer blob (tools/summarize_yaw.py at 5febf471318e37e60477ebe72afd48f03879c67e); BIND OK in 19 s; rerun -> report unchanged
cp ckpt/yaw_rotation/binding_report.json $A/binding_report.json
python $A/check_sweep_acceptance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_21:47:49_acceptance_v4.log   # checker v4: exact band rule, gate decision RECOMPUTED with the pinned producer (summary byte-identical to the retained copy; rows/terms/pass equal), k=0 content of the gate runs, live environment/data recomputation, binding-report consistency; ACCEPTANCE PASS in 54 s
python -m pytest tests/test_exp03_record_tools.py -q -p no:cacheprovider      # 16 passed
EVALUATOR_OVERRIDE=$A/eval_stub.py $A/launch_yaw_sweep_v3.sh faulttest   # abort-path fault test with the retained stub (eval_stub.py: exits with --stub-exit N, writes nothing): evaluator exit 7 and a silent exit 0 both leave *_ABORTED_* logs and non-zero codes; FAULTTEST_DONE (logs yaw_rotation_degradation_2026-09-06_21:50:39_faulttest_launcher_selftest.log, yaw_rotation_degradation_2026-09-06_21:50:51_faulttest_stub_exit7_ABORTED_evaluator_exit7.log, yaw_rotation_degradation_2026-09-06_21:50:54_faulttest_stub_noout_ABORTED_provenance_exit1.log)
python tools/summarize_yaw.py --mode full $SWEEP --manifest-hash $H0 --n-boot 20000 --json ckpt/yaw_rotation/full_stats.json --summary ckpt/yaw_rotation/full_summary.txt   # v4 canonical (rule strings): 21:53:54 at working tree 9e63387, exit 0 after 449 s, log yaw_rotation_degradation_2026-09-06_21:53:54_full_summary_v4.log; confirmation rerun at HEAD a938467: 22:01:59, exit 0 after 437 s, log yaw_rotation_degradation_2026-09-06_22:01:59_full_summary_v4b.log, JSON byte-identical
python $A/make_results_md.py --stats ckpt/yaw_rotation/full_stats.json; python $A/make_results_html.py --stats ckpt/yaw_rotation/full_stats.json
cp ckpt/yaw_rotation/full_summary.txt $E/yaw_rotation_degradation_full_summary.txt; cp ckpt/yaw_rotation/full_stats.json $A/full_stats.json; cp ckpt/yaw_rotation/binding_report.json $A/binding_report.json; cp ckpt/yaw_rotation/gate_stats.json $A/gate_stats.json
python -m pytest tests -q -p no:cacheprovider    # 222 passed (detached; log in the session scratchpad; counts in the worklog)
python $A/bind_provenance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_22:10:01_bind_provenance_v4_final.log     # report unchanged, BIND OK
python $A/check_sweep_acceptance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_22:10:20_acceptance_v4_final.log    # ACCEPTANCE PASS (gate recomputed)
```

## After the Codex round-5 review (2026-09-06 22:47–23:00) — commands as run (written into this record at 23:26, from the notebook and the retained logs)
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
E=worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude; A=$E/yaw_rotation_degradation_results_assets
python $A/bind_provenance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_22:47:24_bind_provenance_v5.log       # binder v5 (drift detection, gate-producer closure): report migration (additive: gate_inputs.gate_producer_closure); BIND OK; rerun -> report unchanged
python $A/check_sweep_acceptance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_22:49:22_acceptance_v5.log      # checker v5: ACCEPTANCE PASS (53 s); `--no-recompute-gate` -> ACCEPTANCE INCOMPLETE, exit 2
EVALUATOR_OVERRIDE=$A/eval_stub.py $A/launch_yaw_sweep_v3.sh faulttest   # five abort branches: codes 7 1 1 1 3 1, four new *_ABORTED_* logs, FAULTTEST_DONE (logs yaw_rotation_degradation_2026-09-06_22:53:11_faulttest_launcher_selftest.log, yaw_rotation_degradation_2026-09-06_22:53:22_faulttest_stub_exit7_ABORTED_evaluator_exit7.log, yaw_rotation_degradation_2026-09-06_22:53:25_faulttest_stub_noout_ABORTED_provenance_exit1.log, yaw_rotation_degradation_2026-09-06_22:53:32_faulttest_stub_evchg_ABORTED_evaluator_changed_exit1.log, yaw_rotation_degradation_2026-09-06_22:53:35_faulttest_stub_ckchg_ABORTED_checkpoint_changed_exit1.log)
$A/launch_yaw_sweep_v3.sh sweep; $A/launch_yaw_sweep_v3.sh gate    # refusal self-tests (logs yaw_rotation_degradation_2026-09-06_22:53:42_sweep_launcher_v4_refuse_selftest.log, yaw_rotation_degradation_2026-09-06_22:54:33_gate_launcher_v4_refuse_selftest.log)
python $A/make_results_md.py --stats ckpt/yaw_rotation/full_stats.json; python $A/make_results_html.py --stats ckpt/yaw_rotation/full_stats.json   # generators v5 (roles, run descriptions everywhere, both TOST levels); JSON unchanged
python -m pytest tests/test_exp03_record_tools.py -q -p no:cacheprovider     # 24 passed (no retained log for this invocation; the 23:25 log below covers the current suite)
python $A/bind_provenance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_22:57:53_bind_provenance_v5_final.log   # report unchanged, BIND OK
python $A/check_sweep_acceptance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_22:58:14_acceptance_v5_final.log  # ACCEPTANCE PASS
python -m pytest tests -q -p no:cacheprovider    # 229 passed (detached; no retained log for this invocation; superseded by the 23:25 retained run)
```

## After the Codex round-6 review (2026-09-06 23:15–) — commands as run
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
E=worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude; A=$E/yaw_rotation_degradation_results_assets
python $A/bind_provenance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_23:15:24_bind_provenance_v6.log       # binder v6 (recursive drift incl. deletions and path changes, monotonic gate->all, all-or-nothing transaction, preserved bound_at + binding_history, checkpoint identity tie, canonical log-name rule): no drift, report unchanged, BIND OK
python $A/check_sweep_acceptance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_23:19:29_acceptance_v6.log      # checker v6 (checkpoint identity ties, canonical log names, strict non-boolean integer counts/scalars, all three decision copies and closure working-tree digests in the report check, recomputation exit 0 required, full_gate_validation() for the launcher): ACCEPTANCE PASS
$A/launch_yaw_sweep_v3.sh sweep    # preflight now runs full_gate_validation(recompute=True); then REFUSED (log yaw_rotation_degradation_2026-09-06_23:20:47_sweep_launcher_v5_refuse_selftest.log)
python $A/make_results_md.py --stats ckpt/yaw_rotation/full_stats.json; python $A/make_results_html.py --stats ckpt/yaw_rotation/full_stats.json   # generators v6 (exact coverage validation, strict TOST levels, producer names in every header, data.json beside --out); JSON unchanged
python -m pytest tests/test_exp03_record_tools.py -q -p no:cacheprovider 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_23:25:00_record_tools_tests.log   # 29 passed
{ python -m py_compile $A/*.py tools/summarize_yaw.py tests/test_exp03_record_tools.py; bash -n $A/launch_yaw_sweep_v3.sh; bash -n $A/launch_yaw_sweep_v2.sh; git diff --check; for f in $(git ls-files --others --exclude-standard $E tests); do grep -lP '[ \t]+$' "$f"; done; git rev-parse HEAD; } 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_23:25:00_static_checks.log   # all ok; the log echoes only headings and results, not exit codes (superseded by static_checks.sh below)
python -m pytest tests -q -p no:cacheprovider > $E/yaw_rotation_degradation_2026-09-06_23:25:00_full_suite.log 2>&1   # detached; count in the log's last lines
# final passes after everything (logs named in the worklog's closing entry)
python $A/bind_provenance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_23:26:59_bind_provenance_v6_final.log
python $A/check_sweep_acceptance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_23:27:20_acceptance_v6_final.log
cp ckpt/yaw_rotation/binding_report.json $A/binding_report.json
```

## After the Codex round-7 review (2026-09-06 23:47–) — commands as run
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
E=worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude; A=$E/yaw_rotation_degradation_results_assets
python $A/bind_provenance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_23:47:47_bind_provenance_v7.log       # binder v7 (migration idempotence, additive-only sidecar digest changes, commit_transaction): report unchanged, BIND OK; rerun identical
cp ckpt/yaw_rotation/binding_report.json $A/binding_report.json                                                       # unchanged (f4dcb56018c3dfde…)
python $A/check_sweep_acceptance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_23:48:25_acceptance_v7.log      # checker v7 (non-empty run_start, integer inventory counts, batch_canonical on gate meta): ACCEPTANCE PASS
$A/static_checks.sh 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_23:49:42_static_checks.log                     # every command echoed with its exit status; STATIC_CHECKS PASS
python -m pytest tests/test_exp03_record_tools.py -q -p no:cacheprovider 2>&1 | tee $E/yaw_rotation_degradation_2026-09-06_23:49:42_record_tools_tests.log   # 31 passed
python -m pytest tests -q -p no:cacheprovider > $E/yaw_rotation_degradation_2026-09-06_23:49:42_full_suite.log 2>&1   # detached; count in the log's last lines
```

## After the Codex round-8 review (2026-09-07 00:07–) — commands as run
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
E=worklog/worklog_yixun/exp_03_yaw_rotation_degradation_claude; A=$E/yaw_rotation_degradation_results_assets
{ python $A/bind_provenance.py; echo "exit $?"; python $A/bind_provenance.py; echo "exit $?"; } 2>&1 | tee $E/yaw_rotation_degradation_2026-09-07_00:07:21_bind_provenance_v8.log   # binder v8 (append-only history in the drift rule, project()/commit_transaction()): both invocations report unchanged, BIND OK, exit 0
cp ckpt/yaw_rotation/binding_report.json $A/binding_report.json                                                       # unchanged (f4dcb56018c3dfde…)
python $A/check_sweep_acceptance.py 2>&1 | tee $E/yaw_rotation_degradation_2026-09-07_00:07:59_acceptance_v8.log      # ACCEPTANCE PASS (gate recomputed)
$A/static_checks.sh 2>&1 | tee $E/yaw_rotation_degradation_2026-09-07_00:08:52_static_checks.log                     # STATIC_CHECKS PASS
python -m pytest tests/test_exp03_record_tools.py -q -p no:cacheprovider 2>&1 | tee $E/yaw_rotation_degradation_2026-09-07_00:08:52_record_tools_tests.log   # 32 passed
python -m pytest tests -q -p no:cacheprovider > $E/yaw_rotation_degradation_2026-09-07_00:08:52_full_suite.log 2>&1   # detached; count in the log's last lines
```

## Follow-up: K = 1 evaluation for the paper table (2026-09-12 11:00–11:20; HEAD `19a04ef`; evaluator/closure unchanged)
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd); export XRIR_DATA_PATH=/home/yixunhu/data_cache/AcousticRooms
python -c "from treble_multi_room_dataset.treble_xRIR_dataset import xRIR_Dataset; from tools.reference_manifest import build_manifest, save_manifest, manifest_hash; ds=xRIR_Dataset(split='test', max_len=9600, num_shot=1); m=build_manifest(ds, seed=0, num_shot=1); save_manifest(m, 'ckpt/yaw_rotation/reference_manifest_k1.json'); print(manifest_hash(m))"   # f6d71f86d5d313f2a982fe6f8cb801116a20ea1c37b0a65d290c088934fd73e3
H1=f6d71f86d5d313f2a982fe6f8cb801116a20ea1c37b0a65d290c088934fd73e3; K0="--yaw-cols 0 --acoustic-cols 0 --e-acoustic-cols --decomposition-batches 0"; STD="--batch-size 16 --num-workers 6 --threads 4 --log-interval 10"
CUDA_VISIBLE_DEVICES=0 python eval_yaw_rotation.py --backbone simple      --checkpoint ckpt/xRIR_simple_8_shot/epoch_12.pth --manifest ckpt/yaw_rotation/reference_manifest_k1.json --manifest-hash $H1 --out-dir ckpt/yaw_rotation/k1_control  $STD $K0   # log yaw_rotation_degradation_2026-09-12_11:00:21_k1_control.log
CUDA_VISIBLE_DEVICES=0 python eval_yaw_rotation.py --backbone simple      --checkpoint checkpoints/xRIR_unseen.pth           --manifest ckpt/yaw_rotation/reference_manifest_k1.json --manifest-hash $H1 --out-dir ckpt/yaw_rotation/k1_released $STD $K0   # log ..._k1_released.log (same timestamp prefix)
CUDA_VISIBLE_DEVICES=1 python eval_yaw_rotation.py --backbone cylindrical --checkpoint ckpt/xRIR_cyl_8_shot/epoch_12.pth    --manifest ckpt/yaw_rotation/reference_manifest_k1.json --manifest-hash $H1 --out-dir ckpt/yaw_rotation/k1_cyl      $STD $K0   # log ..._k1_cyl.log
python tools/summarize_yaw.py --mode exploratory --runs ckpt/yaw_rotation/k1_control ckpt/yaw_rotation/k1_cyl --labels control cyl --manifest-hash $H1 --n-boot 20000 --json ckpt/yaw_rotation/k1_stats.json --summary ckpt/yaw_rotation/k1_summary.txt
```
