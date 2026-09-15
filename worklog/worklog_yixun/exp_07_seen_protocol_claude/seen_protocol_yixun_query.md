# Query — seen_protocol (exp_07)

Yixun, 2026-09-15 (after the exp_04 table draft): "Could you please give me the latex of seen and unseen split of xRIR (SimpleViT) versus YawAug-xRIR and CylindricalViT? just like Table 1" → the seen columns cannot be filled from the unseen-protocol models (their training set contains 5 124 of the 6 217 seen-split queries; exp_01 recorded the contamination) → "So We don't have seen split evaluation results yet?" → "but why your training dataset has the seen split?" (answer: the authors train one model per protocol; the seen protocol's training set is every file except the 6 217 seen-test pairs) → **"go for the exp_07"**.

Deliverable: the seen-split columns (EDT s, C50 dB, T60 %) at K = 1 and K = 8 for the same-budget xRIR (SimpleViT), CylindricalViT and YawAug-xRIR, produced with the same five-seed protocol as the unseen columns of exp_04, plus the paper-style LaTeX table combining both splits.
