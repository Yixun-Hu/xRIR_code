<!-- results_table:begin -->
# Model comparison

Mean ± sample SD over five evaluation seeds (42–46). Each seed selects the K-specific reference manifest and the Griffin-Lim phase. Each seed mean uses its finite queries; per-seed finite counts are recorded in the canonical JSON. Values are rounded to 6 significant digits; the JSON is canonical.

| Model | K | T60 (%) | C50 (dB) | EDT (ms) | Loss (objective) | Log-STFT MSE | Protocol |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SimpleViT (seen) | 8 | 6.88653 ± 0.0152101 | 0.908064 ± 0.00467568 | 34.4122 ± 0.169286 | 0.0155171 ± 1.47618e-05 | 0.416091 ± 0.000941012 | K = 8; seen; 6217 queries; 5 seeds; epoch 12; P; k = 0; batch 16; TF32 off |
| SimpleViT (seen) | 1 | 10.6119 ± 0.211754 | 1.48967 ± 0.0173209 | 57.5455 ± 0.572044 | 0.0246492 ± 0.000182 | 0.750553 ± 0.00301974 | K = 1; seen; 6217 queries; 5 seeds; epoch 12; P; k = 0; batch 16; TF32 off |
| CylindricalViT (seen) | 8 | 6.92062 ± 0.0155034 | 0.889695 ± 0.00522766 | 34.0998 ± 0.289578 | 0.0153224 ± 1.84601e-05 | 0.412145 ± 0.00117031 | K = 8; seen; 6217 queries; 5 seeds; epoch 12; P; k = 0; batch 16; TF32 off |
| CylindricalViT (seen) | 1 | 11.1156 ± 0.264221 | 1.52694 ± 0.0116447 | 60.5997 ± 0.554903 | 0.0248675 ± 0.000206951 | 0.760659 ± 0.003319 | K = 1; seen; 6217 queries; 5 seeds; epoch 12; P; k = 0; batch 16; TF32 off |
| YawAugxRIR (seen) | 8 | 6.84367 ± 0.0120603 | 0.916749 ± 0.0052992 | 35.0698 ± 0.12201 | 0.0157672 ± 1.62336e-05 | 0.421191 ± 0.000592833 | K = 8; seen; 6217 queries; 5 seeds; epoch 12; P; k = 0; batch 16; TF32 off |
| YawAugxRIR (seen) | 1 | 10.2735 ± 0.207799 | 1.47215 ± 0.0142216 | 56.7474 ± 0.694573 | 0.0244035 ± 0.000170569 | 0.739832 ± 0.00224529 | K = 1; seen; 6217 queries; 5 seeds; epoch 12; P; k = 0; batch 16; TF32 off |
| xRIR released (reference) | 8 | 7.26566 ± 0.00485214 | 1.02423 ± 0.00206912 | 38.3982 ± 0.103268 | 0.0163086 ± 1.29564e-05 | 0.410789 ± 0.000603014 | K = 8; seen; 6217 queries; 5 seeds; epoch unknown (external); P; k = 0; batch 16; TF32 off |
| xRIR released (reference) | 1 | 10.3676 ± 0.2139 | 1.61325 ± 0.0188606 | 62.7636 ± 0.967621 | 0.0254012 ± 0.000162574 | 0.766777 ± 0.00367064 | K = 1; seen; 6217 queries; 5 seeds; epoch unknown (external); P; k = 0; batch 16; TF32 off |

## Protocol

Rows report K, seen split query count, seed count, epoch, condition P, standalone k = 0, canonical batch 16, and TF32 off. max_samples = 0; no test-time augmentation.
Finite-count tolerance: 2 queries between seeds in each row/metric; empty seeds are refused.

## Provenance

Canonical JSON: `/home/yixunhu/codespace/xRIR_code/ckpt/exp07/results/TABLE_SEEN_V1.json`; sha256: `197c8b4f5f2b5dea36028ee7bd93ed7dd5722a7e1ba00fcb62434fc317c314fc`.
Profile digest: `d8d8d566f820bef7f53037f5e54543e2d9690c98b013e4bf0b4c3de11b257a6c`.

Generation command:
```sh
exp07_table.py --profile TABLE_SEEN_V1 --runs /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_simple_k8_seed42_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_simple_k8_seed43_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_simple_k8_seed44_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_simple_k8_seed45_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_simple_k8_seed46_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_simple_k1_seed42_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_simple_k1_seed43_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_simple_k1_seed44_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_simple_k1_seed45_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_simple_k1_seed46_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_cyl_k8_seed42_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_cyl_k8_seed43_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_cyl_k8_seed44_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_cyl_k8_seed45_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_cyl_k8_seed46_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_cyl_k1_seed42_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_cyl_k1_seed43_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_cyl_k1_seed44_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_cyl_k1_seed45_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_cyl_k1_seed46_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_aug_k8_seed42_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_aug_k8_seed43_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_aug_k8_seed44_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_aug_k8_seed45_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_aug_k8_seed46_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_aug_k1_seed42_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_aug_k1_seed43_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_aug_k1_seed44_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_aug_k1_seed45_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/seen_aug_k1_seed46_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/released_seen_k8_seed42_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/released_seen_k8_seed43_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/released_seen_k8_seed44_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/released_seen_k8_seed45_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/released_seen_k8_seed46_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/released_seen_k1_seed42_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/released_seen_k1_seed43_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/released_seen_k1_seed44_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/released_seen_k1_seed45_k0 /home/yixunhu/codespace/xRIR_code/ckpt/exp07/eval/released_seen_k1_seed46_k0 --json /home/yixunhu/codespace/xRIR_code/ckpt/exp07/results/TABLE_SEEN_V1.json --md /home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/model_comparison_seen.md
```
<!-- results_table:sha256:c94140743abf2c9a0f11a3b75ab292f7198f1f46d5a6c6d648e5a9a7046d7f92 -->
<!-- results_table:end -->
