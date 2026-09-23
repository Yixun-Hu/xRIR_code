"""Environment panels for the HAA classroom (classroomBase): photo, 3D planar model, floor map, depth panorama.

Geometry is the DiffRIR planar model from the authors' repository (rooms/classroom.py,
github.com/maswang32/hearinganythinganywhere): a 7.12 x 7.92 x 2.74 m box plus three
table slabs; the speaker position and the 630 microphone positions come from the dataset
(scenes_metadata.json / xyzs.npy in HAA_processed). The depth panorama is the one the xRIR
models see (rendered at the speaker, which plays the model's "receiver" role).
The photo is the authors' (project page static/images/Classroom.jpg) and must be credited.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

IN = 0.0254
MAX_X, MAX_Y, MAX_Z = 7.1247, 7.9248, 2.7432
SPEAKER = np.array([3.5838, 5.7230, 1.2294])
TABLES = {  # (x0, x1, y0, y1) at table height 29 in, from rooms/classroom.py
    "left": (0.0, 30 * IN, 96 * IN, MAX_Y),
    "right": (MAX_X - 30 * IN, MAX_X, 23.75 * IN, MAX_Y),
    "middle": (2.935758, 4.474256, 89 * IN, MAX_Y),
}
TABLE_Z = 29 * IN
HAA = "/media/diskstation/yixunhu/HAA_processed"
CACHE = os.path.expanduser("~/data_cache/HAA_xrir/class_room")


def load():
    meta = json.load(open(os.path.join(HAA, "metadata", "scenes_metadata.json")))["classroomBase"]
    xyz = np.load(os.path.join(CACHE, "xyzs.npy"))
    depth = np.load(os.path.join(CACHE, "depth.npy"))
    return meta, xyz, depth


def quad(x0, x1, y0, y1, z0, z1):
    return [[x0, y0, z0], [x1, y0, z0], [x1, y1, z1], [x0, y1, z1]]


def draw_3d(ax, meta, xyz):
    walls = [quad(0, MAX_X, 0, MAX_Y, 0, 0),                                       # floor
             [[0, MAX_Y, 0], [MAX_X, MAX_Y, 0], [MAX_X, MAX_Y, MAX_Z], [0, MAX_Y, MAX_Z]],   # far wall (y = max_y)
             [[MAX_X, 0, 0], [MAX_X, MAX_Y, 0], [MAX_X, MAX_Y, MAX_Z], [MAX_X, 0, MAX_Z]]]   # far wall (x = max_x)
    ax.add_collection3d(Poly3DCollection(walls, facecolors=["#e6e6e6", "#f3f3f3", "#f8f8f8"], edgecolors="#999999", linewidths=0.5, alpha=0.35))
    # remaining edges of the box as wireframe
    for a, b in [((MAX_X, 0, 0), (MAX_X, MAX_Y, 0)), ((MAX_X, MAX_Y, 0), (0, MAX_Y, 0)), ((MAX_X, 0, 0), (MAX_X, 0, MAX_Z)),
                 ((MAX_X, MAX_Y, 0), (MAX_X, MAX_Y, MAX_Z)), ((0, MAX_Y, 0), (0, MAX_Y, MAX_Z)),
                 ((0, 0, MAX_Z), (MAX_X, 0, MAX_Z)), ((MAX_X, 0, MAX_Z), (MAX_X, MAX_Y, MAX_Z)),
                 ((MAX_X, MAX_Y, MAX_Z), (0, MAX_Y, MAX_Z)), ((0, MAX_Y, MAX_Z), (0, 0, MAX_Z))]:
        ax.plot(*zip(a, b), color="#999999", lw=0.5)
    tables = [quad(x0, x1, y0, y1, TABLE_Z, TABLE_Z) for (x0, x1, y0, y1) in TABLES.values()]
    ax.add_collection3d(Poly3DCollection(tables, facecolors="#c9b18c", edgecolors="#8a6d3b", linewidths=0.6, alpha=0.75))
    train = np.array(meta["train_indices"])
    mask = np.zeros(len(xyz), bool); mask[train] = True
    ax.scatter(xyz[~mask, 0], xyz[~mask, 1], xyz[~mask, 2], s=2, c="#4d4d4d", alpha=0.5, depthshade=False, label="microphones (630)")
    ax.scatter(xyz[mask, 0], xyz[mask, 1], xyz[mask, 2], s=28, c="#009E73", marker="o", edgecolors="white", linewidths=0.5, depthshade=False, label="12 training RIRs (references)")
    ax.scatter(*SPEAKER, s=220, c="#CC0000", marker="*", edgecolors="white", linewidths=0.6, depthshade=False, label="speaker", zorder=10)
    ax.set_xlim(0, MAX_X); ax.set_ylim(0, MAX_Y); ax.set_zlim(0, MAX_Z)
    ax.set_box_aspect((MAX_X, MAX_Y, MAX_Z))
    ax.view_init(elev=24, azim=-58)
    ax.set_xlabel("x (m)", fontsize=7, labelpad=-4); ax.set_ylabel("y (m)", fontsize=7, labelpad=-4); ax.set_zlabel("z (m)", fontsize=7, labelpad=-6)
    ax.tick_params(labelsize=6, pad=-2)
    ax.legend(loc="upper left", fontsize=6, frameon=False, bbox_to_anchor=(-0.05, 1.02))
    ax.set_title("3D planar model (DiffRIR surfaces) + measurement layout", fontsize=8)


def draw_floor(ax, meta, xyz):
    ax.add_patch(Rectangle((0, 0), MAX_X, MAX_Y, fc="#f4f4f4", ec="#555555", lw=1.0))
    for (x0, x1, y0, y1) in TABLES.values():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#c9b18c", ec="#8a6d3b", lw=0.6))
    train = np.array(meta["train_indices"]); mask = np.zeros(len(xyz), bool); mask[train] = True
    ax.scatter(xyz[~mask, 0], xyz[~mask, 1], s=3, c="#4d4d4d", alpha=0.6)
    ax.scatter(xyz[mask, 0], xyz[mask, 1], s=30, c="#009E73", edgecolors="white", linewidths=0.5)
    ax.scatter(SPEAKER[0], SPEAKER[1], s=150, c="#CC0000", marker="*", edgecolors="white", linewidths=0.6)
    ax.set_xlim(-0.3, MAX_X + 0.3); ax.set_ylim(-0.3, MAX_Y + 0.3); ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=7); ax.set_ylabel("y (m)", fontsize=7); ax.tick_params(labelsize=6)
    ax.set_title("floor map (top view)", fontsize=8)


def draw_panorama(ax, meta, xyz, depth):
    from treble_multi_room_dataset.treble_xRIR_dataset import convert_equirect_to_camera_coord
    import torch
    coord = convert_equirect_to_camera_coord(torch.from_numpy(depth.astype(np.float32)), 256, 512).permute(2, 0, 1).numpy()
    u = coord / np.maximum(np.sqrt((coord ** 2).sum(0)), 1e-6)[None]
    ax.imshow(depth, cmap="gray_r", aspect="auto", vmin=0, vmax=np.percentile(depth, 99))
    for i in meta["train_indices"]:
        v = xyz[i] - SPEAKER; v = v / np.linalg.norm(v)
        r, c = np.unravel_index(int(np.argmax((u * v[:, None, None]).sum(0))), depth.shape)
        ax.plot(c, r, marker="o", ms=5, mfc="none", mec="#009E73", mew=1.2, ls="none")
    ax.set_xticks([0, 128, 256, 384, 511]); ax.set_xticklabels(["-180°", "-90°", "0°", "90°", "180°"], fontsize=6)
    ax.set_yticks([0, 128, 255]); ax.set_yticklabels(["+90°", "0°", "-90°"], fontsize=6)
    ax.set_title("depth panorama at the speaker (xRIR input)", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", default="", help="optional path to the authors' Classroom.jpg")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "classroom_environment"))
    args = ap.parse_args()
    meta, xyz, depth = load()
    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.25, 0.9], height_ratios=[1, 1], left=0.03, right=0.99, top=0.93, bottom=0.06, wspace=0.18, hspace=0.32)
    ax = fig.add_subplot(gs[:, 0])
    if args.photo and os.path.exists(args.photo):
        ax.imshow(plt.imread(args.photo)); ax.set_title("photo (Wang et al., HAA project page)", fontsize=8)
    else:
        ax.text(0.5, 0.5, "photo: masonlwang.com/hearinganythinganywhere\nstatic/images/Classroom.jpg", ha="center", va="center", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    draw_3d(fig.add_subplot(gs[:, 1], projection="3d"), meta, xyz)
    draw_floor(fig.add_subplot(gs[0, 2]), meta, xyz)
    draw_panorama(fig.add_subplot(gs[1, 2]), meta, xyz, depth)
    fig.suptitle("HAA classroom (classroomBase): 7.12 × 7.92 × 2.74 m, 630 microphone positions, one speaker", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (args.out, ext), dpi=250 if ext == "png" else None)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
