<!-- results_table:begin -->
# Model comparison

Mean ± sample SD over five evaluation seeds (42–46). Each seed selects the K-specific reference manifest and the Griffin-Lim phase. Each seed mean uses its finite queries; per-seed finite counts are recorded in the canonical JSON. Values are rounded to 6 significant digits; the JSON is canonical.

| Model | K | T60 (%) | C50 (dB) | EDT (ms) | Loss (objective) | Log-STFT MSE | Protocol |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SimpleViT | 8 | 9.57524 ± 0.00894945 | 1.23833 ± 0.00532373 | 47.0176 ± 0.185805 | 0.0161741 ± 1.24785e-05 | 0.423136 ± 0.00122912 | K = 8; unseen; 6337 queries; 5 seeds; epoch 12; P; k = 0; batch 16; TF32 off |
| SimpleViT | 1 | 13.3828 ± 0.101397 | 1.88976 ± 0.0132673 | 68.2756 ± 0.609058 | 0.0231375 ± 7.94701e-05 | 0.800276 ± 0.00544638 | K = 1; unseen; 6337 queries; 5 seeds; epoch 12; P; k = 0; batch 16; TF32 off |
| CylindricalViT | 8 | 9.44256 ± 0.0135858 | 1.23094 ± 0.00593518 | 45.0629 ± 0.253312 | 0.0159191 ± 2.24268e-05 | 0.417353 ± 0.000937605 | K = 8; unseen; 6337 queries; 5 seeds; epoch 12; P; k = 0; batch 16; TF32 off |
| CylindricalViT | 1 | 13.5775 ± 0.0945649 | 1.94186 ± 0.0136765 | 69.7608 ± 0.786699 | 0.0235446 ± 0.000105503 | 0.780161 ± 0.00580484 | K = 1; unseen; 6337 queries; 5 seeds; epoch 12; P; k = 0; batch 16; TF32 off |
| YawAugxRIR | 8 | 9.32189 ± 0.00591373 | 1.29365 ± 0.00645626 | 46.3901 ± 0.146651 | 0.0160374 ± 2.21286e-05 | 0.412958 ± 0.00048156 | K = 8; unseen; 6337 queries; 5 seeds; epoch 12; P; k = 0; batch 16; TF32 off |
| YawAugxRIR | 1 | 13.3859 ± 0.0774579 | 1.96877 ± 0.0153867 | 69.5245 ± 0.725792 | 0.0233678 ± 8.1606e-05 | 0.73931 ± 0.00341285 | K = 1; unseen; 6337 queries; 5 seeds; epoch 12; P; k = 0; batch 16; TF32 off |

## Protocol

Rows report K, unseen split query count, seed count, epoch, condition P, standalone k = 0, canonical batch 16, and TF32 off. max_samples = 0; no test-time augmentation.
Finite-count tolerance: 2 queries between seeds in each row/metric; empty seeds are refused.

## Provenance

Canonical JSON: `/home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/results/TABLE_V1.json`; sha256: `46c6a6ae999aca9d8aefcb0ea174942ecb6c65739c7e4dd41d35b0135ed1ed41`.
Profile digest: `be0449f59e559da234bf91b6d7d9552af535aafe0910728363bbdd802339b1ac`.

Generation command:
```sh
results_table.py --profile TABLE_V1 --runs /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/control_k8_seed42_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/control_k8_seed43_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/control_k8_seed44_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/control_k8_seed45_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/control_k8_seed46_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/control_k1_seed42_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/control_k1_seed43_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/control_k1_seed44_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/control_k1_seed45_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/control_k1_seed46_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/cyl_k8_seed42_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/cyl_k8_seed43_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/cyl_k8_seed44_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/cyl_k8_seed45_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/cyl_k8_seed46_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/cyl_k1_seed42_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/cyl_k1_seed43_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/cyl_k1_seed44_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/cyl_k1_seed45_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/cyl_k1_seed46_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/aug_k8_seed42_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/aug_k8_seed43_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/aug_k8_seed44_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/aug_k8_seed45_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/aug_k8_seed46_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/aug_k1_seed42_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/aug_k1_seed43_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/aug_k1_seed44_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/aug_k1_seed45_k0 /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/eval/aug_k1_seed46_k0 --json /home/yixunhu/codespace/xRIR_code/ckpt/yaw_aug/results/TABLE_V1.json --md /home/yixunhu/codespace/xRIR_code/worklog/worklog_yixun/model_comparison.md
```
<!-- results_table:sha256:b37f33a40b9f3d23df85512600f5a7783fef72483b2f23472a8beba3bb8175a4 -->
<!-- results_table:end -->
