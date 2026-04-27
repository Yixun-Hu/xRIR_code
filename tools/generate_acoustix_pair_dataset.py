"""Generate a minimal arbitrary tx/rx pair dataset with AcoustiX.

The output is intentionally simple and model-facing:

data/acoustix_pair_dataset/<dataset_name>/
  metadata.json
  points.npy
  orientations.npy
  depth/point_000.npy        # equirectangular panorama depth [H, W]
  depth/point_000_coord.npy  # xRIR camera-coordinate map [3, H, W]
  rir/tx_000_rx_001.npz
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_ACOUSTIX_ROOT = _REPO_ROOT / "AcoustiX"
_SIBLING_ACOUSTIX_ROOT = _REPO_ROOT.parent / "AcoustiX"
ACOUSTIX_ROOT = (
    Path(os.environ["ACOUSTIX_ROOT"])
    if "ACOUSTIX_ROOT" in os.environ
    else _SIBLING_ACOUSTIX_ROOT
    if _SIBLING_ACOUSTIX_ROOT.exists()
    else _SIBLING_ACOUSTIX_ROOT
    if not _LOCAL_ACOUSTIX_ROOT.exists()
    else _LOCAL_ACOUSTIX_ROOT
)
sys.path.insert(0, str(ACOUSTIX_ROOT))

import mitsuba as mi  # noqa: E402
from sionna.rt.scene import Scene  # noqa: E402
from simu_utils import config_scene, ir_simulation, load_cfg  # noqa: E402


def default_scene_file() -> Path:
    """
    Return the default AcoustiX scene file.

    Parameters
    ----------
    None
        This function takes no parameters.

    Returns
    -------
    Path
        Path to the preferred AcoustiX scene XML file.
    """
    pomaria_scene = ACOUSTIX_ROOT / "extract_scene" / "Pomaria_2_int" / "output_scene.xml"
    if pomaria_scene.exists():
        return pomaria_scene
    return ACOUSTIX_ROOT / "sionna" / "sionna" / "rt" / "scenes" / "box" / "box.xml"


def parse_xyz_bounds(bounds: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a compact xyz bound string.

    Parameters
    ----------
    bounds : str
        Bounds in the form ``xmin,ymin,zmin:xmax,ymax,zmax``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Minimum and maximum xyz vectors, each with shape ``(3,)``.

    Raises
    ------
    ValueError
        Raised when the bound string does not contain two xyz vectors.
    """
    parts = bounds.split(":")
    if len(parts) != 2:
        raise ValueError("Bounds must have the form xmin,ymin,zmin:xmax,ymax,zmax")
    xyz_min = np.array([float(value) for value in parts[0].split(",")], dtype=np.float32)
    xyz_max = np.array([float(value) for value in parts[1].split(",")], dtype=np.float32)
    if xyz_min.shape != (3,) or xyz_max.shape != (3,):
        raise ValueError("Both min and max bounds must contain exactly three values")
    return xyz_min, xyz_max


def sample_points(
    rng: np.random.Generator,
    num_points: int,
    xyz_min: np.ndarray,
    xyz_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample candidate tx/rx endpoint positions and horizontal orientations.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator used for reproducible sampling.
    num_points : int
        Number of candidate endpoint positions to sample.
    xyz_min : np.ndarray
        Lower xyz bound with shape ``(3,)``.
    xyz_max : np.ndarray
        Upper xyz bound with shape ``(3,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Positions and orientations. Positions have shape ``(num_points, 3)``;
        orientations have shape ``(num_points, 3)``.
    """
    positions = rng.uniform(xyz_min, xyz_max, size=(num_points, 3)).astype(np.float32)
    yaw = rng.uniform(-np.pi, np.pi, size=num_points)
    orientations = np.stack([np.cos(yaw), np.sin(yaw), np.zeros(num_points)], axis=-1)
    return positions.astype(np.float32), orientations.astype(np.float32)


def make_ordered_pairs(
    rng: np.random.Generator,
    num_points: int,
    num_pairs: int,
    include_reciprocal: bool,
) -> list[tuple[int, int]]:
    """
    Sample ordered tx/rx query pairs.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator used for reproducible pair sampling.
    num_points : int
        Number of candidate endpoint positions.
    num_pairs : int
        Number of base ordered pairs before optional reciprocal augmentation.
    include_reciprocal : bool
        If ``True``, add ``(rx, tx)`` for every sampled ``(tx, rx)`` pair.

    Returns
    -------
    list[tuple[int, int]]
        Ordered pairs represented by ``(tx_index, rx_index)``.
    """
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(pairs) < num_pairs:
        tx_idx = int(rng.integers(0, num_points))
        rx_idx = int(rng.integers(0, num_points))
        if tx_idx == rx_idx:
            continue
        pair = (tx_idx, rx_idx)
        if pair in seen:
            continue
        pairs.append(pair)
        seen.add(pair)

    if include_reciprocal:
        reciprocal_pairs = [(rx_idx, tx_idx) for tx_idx, rx_idx in pairs]
        for pair in reciprocal_pairs:
            if pair not in seen:
                pairs.append(pair)
                seen.add(pair)
    return pairs


def grouped_by_tx(pairs: Sequence[tuple[int, int]]) -> dict[int, list[int]]:
    """
    Group receiver indices by transmitter index.

    Parameters
    ----------
    pairs : Sequence[tuple[int, int]]
        Ordered ``(tx_index, rx_index)`` pairs.

    Returns
    -------
    dict[int, list[int]]
        Mapping from transmitter index to receiver indices.
    """
    groups: dict[int, list[int]] = {}
    for tx_idx, rx_idx in pairs:
        if tx_idx not in groups:
            groups[tx_idx] = []
        groups[tx_idx].append(rx_idx)
    return groups


def equirectangular_ray_directions(height: int, width: int) -> np.ndarray:
    """
    Build AcousticRooms-style equirectangular ray directions.

    Parameters
    ----------
    height : int
        Panorama height in pixels.
    width : int
        Panorama width in pixels.

    Returns
    -------
    np.ndarray
        Unit ray directions with shape ``(height * width, 3)``. The convention
        matches ``convert_equirect_to_camera_coord`` in the xRIR dataset code.
    """
    row_idx, col_idx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    theta_map = (col_idx + 0.5) * 2.0 * np.pi / width - np.pi
    phi_map = (row_idx + 0.5) * np.pi / height - np.pi / 2.0
    cos_phi = np.cos(phi_map)
    directions = np.stack(
        [
            cos_phi * np.cos(theta_map),
            cos_phi * np.sin(theta_map),
            -np.sin(phi_map),
        ],
        axis=-1,
    )
    return directions.reshape(-1, 3).astype(np.float32)


def render_panorama_depth(
    scene: Scene,
    position: np.ndarray,
    height: int,
    width: int,
    chunk_size: int = 32768,
) -> np.ndarray:
    """
    Render an equirectangular panorama depth map from one endpoint.

    Parameters
    ----------
    scene : Scene
        Loaded Sionna/AcoustiX scene.
    position : np.ndarray
        Camera position with shape ``(3,)``.
    height : int
        Panorama height in pixels.
    width : int
        Panorama width in pixels.
    chunk_size : int, optional
        Number of rays to intersect per chunk (default is 32768).

    Returns
    -------
    np.ndarray
        Panorama depth map with shape ``(height, width)``.
    """
    directions = equirectangular_ray_directions(height=height, width=width)
    depths = np.zeros(directions.shape[0], dtype=np.float32)
    origin = np.asarray(position, dtype=np.float32)

    for start_idx in range(0, directions.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, directions.shape[0])
        cur_dirs = directions[start_idx:end_idx]
        num_rays = cur_dirs.shape[0]
        ray = mi.Ray3f(
            o=mi.Point3f(
                [origin[0]] * num_rays,
                [origin[1]] * num_rays,
                [origin[2]] * num_rays,
            ),
            d=mi.Vector3f(cur_dirs[:, 0], cur_dirs[:, 1], cur_dirs[:, 2]),
        )
        surface_interaction = scene.mi_scene.ray_intersect(ray)
        cur_depth = np.asarray(surface_interaction.t, dtype=np.float32)
        depths[start_idx:end_idx] = np.nan_to_num(
            cur_depth,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    return depths.reshape(height, width)


def convert_equirect_to_camera_coord(depth_map: np.ndarray) -> np.ndarray:
    """
    Convert equirectangular depth to xRIR camera-coordinate map.

    Parameters
    ----------
    depth_map : np.ndarray
        Equirectangular depth map with shape ``(height, width)``.

    Returns
    -------
    np.ndarray
        Camera-coordinate map with shape ``(3, height, width)``.
    """
    height, width = depth_map.shape
    directions = equirectangular_ray_directions(height=height, width=width)
    coords = directions.reshape(height, width, 3) * depth_map[:, :, None]
    return np.moveaxis(coords, -1, 0).astype(np.float32)


def run_ir_simulation_aligned(
    scene_file: Path,
    rx_positions: np.ndarray,
    tx_position: np.ndarray,
    rx_orientations: np.ndarray,
    tx_orientation: np.ndarray,
    simu_config: dict[str, object],
    add_noise: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run AcoustiX and keep outputs aligned to the requested receivers.

    Parameters
    ----------
    scene_file : Path
        Path to the AcoustiX scene XML.
    rx_positions : np.ndarray
        Requested receiver positions with shape ``(num_receivers, 3)``.
    tx_position : np.ndarray
        Transmitter position with shape ``(3,)``.
    rx_orientations : np.ndarray
        Receiver orientations with shape ``(num_receivers, 3)``.
    tx_orientation : np.ndarray
        Transmitter orientation with shape ``(3,)``.
    simu_config : dict[str, object]
        AcoustiX simulation configuration.
    add_noise : bool
        If ``True``, ask AcoustiX to add simulation noise when supported.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Aligned IR samples with shape ``(num_receivers, ir_len)`` and a valid
        mask with shape ``(num_receivers,)``.
    """
    simulation_parameters = inspect.signature(ir_simulation).parameters
    if "filter_invalid" in simulation_parameters:
        ir_samples, _, _, valid_mask = ir_simulation(
            scene_path=str(scene_file),
            rx_pos=rx_positions,
            tx_pos=tx_position,
            rx_ori=rx_orientations,
            tx_ori=tx_orientation,
            simu_config=simu_config,
            filter_invalid=False,
            add_noise=add_noise,
        )
        return ir_samples.astype(np.float32), valid_mask.astype(np.int64)

    ir_len = int(simu_config["ir_len"])
    aligned_ir = np.zeros((rx_positions.shape[0], ir_len), dtype=np.float32)
    valid_mask = np.zeros(rx_positions.shape[0], dtype=np.int64)
    returned_ir, returned_rx_pos, _ = ir_simulation(
        scene_path=str(scene_file),
        rx_pos=rx_positions,
        tx_pos=tx_position,
        rx_ori=rx_orientations,
        tx_ori=tx_orientation,
        simu_config=simu_config,
    )
    for returned_idx, returned_position in enumerate(returned_rx_pos):
        distances = np.linalg.norm(rx_positions - returned_position, axis=1)
        matched_idx = int(np.argmin(distances))
        if distances[matched_idx] < 1.0e-5:
            aligned_ir[matched_idx] = returned_ir[returned_idx].astype(np.float32)
            valid_mask[matched_idx] = 1
    return aligned_ir, valid_mask


def write_json(path: Path, payload: dict[str, object]) -> None:
    """
    Write a JSON payload with indentation.

    Parameters
    ----------
    path : Path
        Destination path.
    payload : dict[str, object]
        JSON-serializable payload.

    Returns
    -------
    None
        The file is written to disk.
    """
    with path.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.

    Parameters
    ----------
    None
        This function takes no parameters.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the dataset generation command.
    """
    parser = argparse.ArgumentParser(description="Generate arbitrary tx/rx pair data with AcoustiX.")
    parser.add_argument("--scene-file", type=Path, default=default_scene_file())
    parser.add_argument("--config-file", type=Path, default=ACOUSTIX_ROOT / "simu_config" / "basic_config.yml")
    parser.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "data" / "acoustix_pair_dataset")
    parser.add_argument("--dataset-name", type=str, default="debug_pairs")
    parser.add_argument("--num-points", type=int, default=8)
    parser.add_argument("--num-pairs", type=int, default=12)
    parser.add_argument("--bounds", type=str, default="-7.0,-3.0,1.2:1.0,5.5,1.7")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rt-num-samples", type=int, default=2048)
    parser.add_argument("--rt-max-depth", type=int, default=3)
    parser.add_argument("--depth-width", type=int, default=512)
    parser.add_argument("--depth-height", type=int, default=256)
    parser.add_argument(
        "--depth-spp",
        type=int,
        default=1,
        help="Deprecated compatibility flag; panorama depth uses ray intersections.",
    )
    parser.add_argument(
        "--depth-fov",
        type=float,
        default=90.0,
        help="Deprecated compatibility flag; panorama depth is full 360x180.",
    )
    parser.add_argument("--include-reciprocal", action="store_true")
    parser.add_argument("--skip-depth", action="store_true")
    parser.add_argument("--add-noise", action="store_true")
    return parser


def generate_dataset(args: argparse.Namespace) -> Path:
    """
    Generate endpoint depth maps and ordered tx/rx RIR samples.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments from ``build_arg_parser``.

    Returns
    -------
    Path
        Path to the generated dataset directory.
    """
    dataset_dir = args.output_dir / args.dataset_name
    depth_dir = dataset_dir / "depth"
    rir_dir = dataset_dir / "rir"
    depth_dir.mkdir(parents=True, exist_ok=True)
    rir_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    xyz_min, xyz_max = parse_xyz_bounds(args.bounds)
    points, orientations = sample_points(rng, args.num_points, xyz_min, xyz_max)
    pairs = make_ordered_pairs(rng, args.num_points, args.num_pairs, args.include_reciprocal)

    np.save(dataset_dir / "points.npy", points)
    np.save(dataset_dir / "orientations.npy", orientations)

    previous_cwd = Path.cwd()
    os.chdir(ACOUSTIX_ROOT)
    try:
        simu_config = load_cfg(config_file=str(args.config_file))
        simu_config["rt_config"]["num_samples"] = args.rt_num_samples
        simu_config["rt_config"]["max_depth"] = args.rt_max_depth

        if not args.skip_depth:
            depth_scene = config_scene(str(args.scene_file))
            for point_idx, position in enumerate(points):
                panorama_depth = render_panorama_depth(
                    scene=depth_scene,
                    position=position,
                    height=args.depth_height,
                    width=args.depth_width,
                )
                np.save(depth_dir / f"point_{point_idx:03d}.npy", panorama_depth)
                depth_coord = convert_equirect_to_camera_coord(panorama_depth)
                np.save(depth_dir / f"point_{point_idx:03d}_coord.npy", depth_coord)

        pair_groups = grouped_by_tx(pairs)
        pair_records: list[dict[str, object]] = []
        for tx_idx, rx_indices in pair_groups.items():
            rx_indices_arr = np.array(rx_indices, dtype=np.int64)
            ir_samples, valid_mask = run_ir_simulation_aligned(
                scene_file=args.scene_file,
                rx_positions=points[rx_indices_arr],
                tx_position=points[tx_idx],
                rx_orientations=orientations[rx_indices_arr],
                tx_orientation=orientations[tx_idx],
                simu_config=simu_config,
                add_noise=args.add_noise,
            )
            for local_idx, rx_idx in enumerate(rx_indices):
                file_name = f"tx_{tx_idx:03d}_rx_{rx_idx:03d}.npz"
                np.savez_compressed(
                    rir_dir / file_name,
                    ir=ir_samples[local_idx].astype(np.float32),
                    tx_index=np.array(tx_idx, dtype=np.int64),
                    rx_index=np.array(rx_idx, dtype=np.int64),
                    tx_position=points[tx_idx],
                    rx_position=points[rx_idx],
                    tx_orientation=orientations[tx_idx],
                    rx_orientation=orientations[rx_idx],
                    valid=np.array(valid_mask[local_idx], dtype=np.int64),
                )
                pair_records.append(
                    {
                        "tx_index": tx_idx,
                        "rx_index": rx_idx,
                        "rir_file": str(Path("rir") / file_name),
                        "valid": int(valid_mask[local_idx]),
                    }
                )
    finally:
        os.chdir(previous_cwd)

    metadata = {
        "scene_file": str(args.scene_file),
        "config_file": str(args.config_file),
        "acoustix_root": str(ACOUSTIX_ROOT),
        "num_points": args.num_points,
        "num_pairs": len(pairs),
        "include_reciprocal": args.include_reciprocal,
        "points_file": "points.npy",
        "orientations_file": "orientations.npy",
        "depth_format": "point_{idx:03d}.npy with shape [H, W]",
        "depth_coord_format": "point_{idx:03d}_coord.npy with shape [3, H, W]",
        "rir_format": "tx_{tx:03d}_rx_{rx:03d}.npz",
        "pairs": pair_records,
    }
    write_json(dataset_dir / "metadata.json", metadata)
    return dataset_dir


def main() -> None:
    """
    Run the AcoustiX pair dataset generator.

    Parameters
    ----------
    None
        Command-line arguments are read from ``sys.argv``.

    Returns
    -------
    None
        Dataset files are written under the requested output directory.
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    dataset_dir = generate_dataset(args)
    print(f"Wrote AcoustiX pair dataset to {dataset_dir}")


if __name__ == "__main__":
    main()
