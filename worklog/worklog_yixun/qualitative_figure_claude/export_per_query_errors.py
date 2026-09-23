"""Per-query EDT / C50 / T60 errors of xRIR SimpleViT vs CylindricalViT, as CSV (for the joint FLAC + xRIR figure).

(a) HAA: exp_02 seed-0 two-stage fine-tuned checkpoints, DiffRIR test split, all four rooms
    (ckpt/sim2real/{control,cyl}/seed0/eval/per_sample_<room>.json; T60 absent for dampened_room).
(b) AcousticRooms unseen split: exp_04 K=8 seed-42 condition-P k=0 canonical evaluations of
    exp_01's same-budget models (ckpt/yaw_aug/eval/{control,cyl}_k8_seed42_k0/per_sample_yaw.json).
Columns: env, room, query_id, model, edt_s, c50_db, t60_pct (empty = NaN / not computed)."""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "joint_export", "per_query_errors.csv")
ARMS = {"simplevit": "control", "cylvit": "cyl"}
ROOMS = ["class_room", "dampened_room", "hallway", "complex_room"]


def cell(v):
    return "" if v is None or v != v else "%.6g" % v


def main():
    rows = []
    for model, arm in ARMS.items():
        for room in ROOMS:
            ps = json.load(open(os.path.join(REPO, "ckpt/sim2real", arm, "seed0", "eval", "per_sample_%s.json" % room)))
            for j, idx in enumerate(ps["index"]):
                rows.append(["haa", room, "mic%d" % idx, model, cell(ps["edt"][j]), cell(ps["c50"][j]),
                             "" if room == "dampened_room" else cell(ps["t60"][j])])
        ps = json.load(open(os.path.join(REPO, "ckpt/yaw_aug/eval", "%s_k8_seed42_k0" % arm, "per_sample_yaw.json")))
        P = ps["P"]["0"]
        for j, q in enumerate(ps["query"]):
            cat, room, fn = q.split("/")
            s, r = fn.split("_")[:2]
            rows.append(["acousticrooms_unseen", room, "S%d_R%d" % (int(s[1:]), int(r[1:])), model,
                         cell(P["edt"][j]), cell(P["c50"][j]), cell(P["t60"][j])])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "room", "query_id", "model", "edt_s", "c50_db", "t60_pct"])
        w.writerows(rows)
    print("wrote", OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
