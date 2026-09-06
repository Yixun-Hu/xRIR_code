"""Paired comparison of two eval_xRIR_backbone.py --save-per-sample files.

Both files must come from the same split, subset, seed and worker count so the
i-th entries are the same query with the same reference RIRs. Reports mean and
median per metric, the paired mean difference (A - B) with a bootstrap 95% CI,
and the fraction of samples where A is lower (better).

    python tools/compare_eval.py --a ckpt/xRIR_cyl_8_shot/per_sample.json --b ckpt/xRIR_simple_8_shot/per_sample.json
"""
import argparse
import json

import numpy as np

METRICS = [("edt", "EDT error (s)"), ("c50", "C50 error (dB)"), ("t60", "T60 error (%)"),
           ("env", "envelope err"), ("stft_mse", "log-STFT MSE"), ("loss", "test loss")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="per-sample JSON for model A")
    p.add_argument("--b", required=True, help="per-sample JSON for model B")
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--by-test-rooms", action="store_true",
                   help="also report the subset of samples from the unseen-split TEST_ROOMS and the complement "
                        "(useful on the seen split, where non-TEST_ROOMS queries were in the unseen-split training set)")
    args = p.parse_args()
    A, B = json.load(open(args.a)), json.load(open(args.b))
    la = args.label_a or A["meta"]["backbone"]
    lb = args.label_b or B["meta"]["backbone"]
    if A["index"] != B["index"]:
        raise SystemExit("sample indices differ; the two evals were not run on the same subset/order")
    rng = np.random.default_rng(args.seed)
    n = len(A["index"])
    print(f"paired samples: {n}   A = {la} ({A['meta']['checkpoint']})   B = {lb} ({B['meta']['checkpoint']})")
    groups = [("all samples", np.ones(n, bool))]
    if args.by_test_rooms:
        from treble_multi_room_dataset.treble_xRIR_dataset import TEST_ROOMS
        in_test = np.array([any(f"/{r}/" in f"/{path}" for r in TEST_ROOMS) for path in A["ir_path"]])
        groups += [("rooms in TEST_ROOMS (never trained on)", in_test), ("other rooms (in unseen-split training set)", ~in_test)]
    for gname, gmask in groups:
        if not gmask.any():
            continue
        print(f"--- {gname}: {int(gmask.sum())} samples ---")
        print(f"{'metric':16s} {'A mean':>9s} {'B mean':>9s} {'A med':>8s} {'B med':>8s} {'A-B mean':>10s} {'95% CI':>20s} {'A<B':>6s} {'valid':>6s}")
        for key, name in METRICS:
            if key not in A or key not in B:
                continue
            a, b = np.asarray(A[key], float)[gmask], np.asarray(B[key], float)[gmask]
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            if ok.sum() == 0:
                continue
            d = a - b
            boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.n_boot)])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"{name:16s} {a.mean():9.4f} {b.mean():9.4f} {np.median(a):8.4f} {np.median(b):8.4f} "
                  f"{d.mean():+10.4f} [{lo:+8.4f}, {hi:+8.4f}]{sig} {np.mean(a < b):6.2f} {ok.sum():6d}")
    print("* = 95% bootstrap CI of the paired difference excludes 0; negative A-B favours A (lower is better).")


if __name__ == "__main__":
    main()
