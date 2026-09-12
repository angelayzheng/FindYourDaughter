#!/usr/bin/env python3
"""Generate controlled synthetic aorta and daughter-vessel NIfTI cases.

Each case contains:

* ``origN.nii``: a CT-like volume with background, parent, and daughters;
* ``maskN.nii``: a binary mask containing only the parent aorta; and
* ``truthN.json``: daughter geometry in physical millimetres.

The geometry is created in physical space before rasterization, so changing
voxel spacing does not change the intended vessel sizes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class Tube:
    centerline_xyz_mm: np.ndarray
    radius_mm: float
    intensity: float


@dataclass(frozen=True)
class DaughterTruth:
    ostium_xyz_mm: list[float]
    seed_xyz_mm: list[float]
    radius_mm: float
    direction_xyz: list[float]
    centerline_xyz_mm: list[list[float]]


def normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / length


def point_list(points: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 4) for value in point] for point in points]


def tube_distance_mask(
    shape_zyx: tuple[int, int, int],
    spacing_xyz_mm: tuple[float, float, float],
    tube: Tube,
) -> np.ndarray:
    """Rasterize a round tube around a polyline into native z/y/x array order."""
    points = tube.centerline_xyz_mm
    spacing = np.asarray(spacing_xyz_mm, dtype=float)
    lower = np.maximum(np.floor((points.min(axis=0) - tube.radius_mm) / spacing).astype(int), 0)
    upper = np.minimum(np.ceil((points.max(axis=0) + tube.radius_mm) / spacing).astype(int), np.asarray(shape_zyx[::-1]) - 1)
    if np.any(lower > upper):
        return np.zeros(shape_zyx, dtype=bool)

    ix, iy, iz = np.meshgrid(
        np.arange(lower[0], upper[0] + 1),
        np.arange(lower[1], upper[1] + 1),
        np.arange(lower[2], upper[2] + 1),
        indexing="ij",
    )
    coordinates = np.stack((ix, iy, iz), axis=-1).astype(float) * spacing
    distance_squared = np.full(coordinates.shape[:-1], np.inf, dtype=float)
    for start, end in zip(points[:-1], points[1:]):
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared == 0:
            candidate = np.sum((coordinates - start) ** 2, axis=-1)
        else:
            fraction = np.clip(np.sum((coordinates - start) * segment, axis=-1) / length_squared, 0, 1)
            closest = start + fraction[..., None] * segment
            candidate = np.sum((coordinates - closest) ** 2, axis=-1)
        distance_squared = np.minimum(distance_squared, candidate)

    local = distance_squared <= tube.radius_mm**2
    result = np.zeros(shape_zyx, dtype=bool)
    result[lower[2] : upper[2] + 1, lower[1] : upper[1] + 1, lower[0] : upper[0] + 1] = local.transpose(2, 1, 0)
    return result


def sample_branch(
    aorta_center_xyz_mm: np.ndarray,
    aorta_radius_mm: float,
    z_mm: float,
    length_mm: float,
    radius_mm: float,
    angle: float,
    max_curvature_mm: float,
    rng: np.random.Generator,
) -> tuple[Tube, DaughterTruth]:
    radial = np.array([np.cos(angle), np.sin(angle), 0.0])
    direction = normalize(radial + np.array([0.0, 0.0, rng.uniform(-0.3, 0.3)]))
    center = aorta_center_xyz_mm.copy()
    center[2] = z_mm
    ostium = center + radial * aorta_radius_mm
    curvature_axis = normalize(np.array([-radial[1], radial[0], 0.0]))
    curvature = float(rng.uniform(-max_curvature_mm, max_curvature_mm))
    control_1 = ostium + direction * length_mm * 0.34
    control_2 = ostium + direction * length_mm * 0.72 + curvature_axis * curvature
    end = ostium + direction * length_mm
    parameter = np.linspace(0, 1, 33)
    points = (
        ((1 - parameter) ** 3)[:, None] * ostium
        + (3 * (1 - parameter) ** 2 * parameter)[:, None] * control_1
        + (3 * (1 - parameter) * parameter**2)[:, None] * control_2
        + (parameter**3)[:, None] * end
    )
    tube = Tube(points, radius_mm, 500.0)
    arc_lengths = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    seed_distance = min(5.0, float(arc_lengths[-1]))
    seed = np.array([np.interp(seed_distance, arc_lengths, points[:, axis]) for axis in range(3)])
    truth = DaughterTruth(
        ostium_xyz_mm=[round(float(value), 4) for value in ostium],
        seed_xyz_mm=[round(float(value), 4) for value in seed],
        radius_mm=round(float(radius_mm), 4),
        direction_xyz=[round(float(value), 6) for value in direction],
        centerline_xyz_mm=point_list(points),
    )
    return tube, truth


def generate_case(
    case_index: int,
    *,
    seed: int,
    daughters: int,
    size_xyz: tuple[int, int, int],
    spacing_xyz_mm: tuple[float, float, float],
    curvature_mm: float = 5.0,
    noise_std: float = 8.0,
) -> dict:
    rng = np.random.default_rng(seed + case_index)
    nx, ny, nz = size_xyz
    shape_zyx = (nz, ny, nx)
    physical_size = np.asarray(size_xyz, dtype=float) * spacing_xyz_mm
    aorta_center = physical_size[:2] / 2
    aorta_center_xyz = np.array([aorta_center[0], aorta_center[1], 0.0])
    aorta_radius = float(rng.uniform(10.0, 14.0))

    main_points = np.array(
        [
            [aorta_center_xyz[0], aorta_center_xyz[1], 0.0],
            [aorta_center_xyz[0], aorta_center_xyz[1], physical_size[2]],
        ]
    )
    parent = Tube(main_points, aorta_radius, 350.0)
    parent_mask = tube_distance_mask(shape_zyx, spacing_xyz_mm, parent)
    image = np.full(shape_zyx, -1000.0, dtype=np.float32)
    image[parent_mask] = parent.intensity

    z_min = 30.0
    z_max = physical_size[2] - 30.0
    z_positions = np.linspace(z_min, z_max, daughters + 2)[1:-1] if daughters else np.array([])
    daughter_truth: list[DaughterTruth] = []
    daughter_tubes: list[Tube] = []
    used_angles: list[float] = []
    minimum_angle = max(0.55, 2 * np.pi / max(daughters, 1) * 0.55)
    for z_mm in z_positions:
        for _ in range(100):
            angle = float(rng.uniform(0, 2 * np.pi))
            angularly_separated = all(
                abs(np.arctan2(np.sin(angle - previous), np.cos(angle - previous))) >= minimum_angle
                for previous in used_angles
            )
            if not angularly_separated:
                continue
            radius = float(rng.uniform(2.5, 4.5))
            length = float(rng.uniform(18.0, 32.0))
            daughter, truth = sample_branch(
                aorta_center_xyz,
                aorta_radius,
                float(z_mm),
                length,
                radius,
                angle,
                curvature_mm,
                rng,
            )
            clearance = radius + max((other.radius_mm for other in daughter_tubes), default=0.0) + 1.0
            too_close = any(
                np.min(np.linalg.norm(daughter.centerline_xyz_mm[:, None] - other.centerline_xyz_mm[None, :], axis=2))
                < clearance
                for other in daughter_tubes
            )
            if too_close:
                continue
            used_angles.append(angle)
            daughter_tubes.append(daughter)
            daughter_truth.append(truth)
            break
        else:
            raise RuntimeError("Could not place non-overlapping daughter tubes; reduce daughter count or curvature")

    for daughter in daughter_tubes:
        daughter_mask = tube_distance_mask(shape_zyx, spacing_xyz_mm, daughter)
        image[daughter_mask] = daughter.intensity
    if noise_std > 0:
        image += rng.normal(0, noise_std, size=shape_zyx).astype(np.float32)

    case_id = f"subject{case_index:03d}"
    return {
        "case_id": case_id,
        "generator": {
            "seed": seed + case_index,
            "noise_std": round(float(noise_std), 4),
            "max_curvature_mm": round(float(curvature_mm), 4),
        },
        "parent": {"instance_id": "aorta", "radius_mm": round(aorta_radius, 4)},
        "daughters": [
            {
                "instance_id": f"branch_{index:03d}",
                "parent_instance_id": "aorta",
                **truth.__dict__,
            }
            for index, truth in enumerate(daughter_truth, start=1)
        ],
        "_image": image,
        "_mask": parent_mask.astype(np.uint8),
        "_spacing": spacing_xyz_mm,
    }


def write_case(case: dict, output_dir: Path) -> None:
    case_id = case["case_id"]
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    spacing = case.pop("_spacing")
    image = sitk.GetImageFromArray(case.pop("_image"))
    mask = sitk.GetImageFromArray(case.pop("_mask"))
    for itk_image in (image, mask):
        itk_image.SetSpacing(spacing)
        itk_image.SetOrigin((0.0, 0.0, 0.0))
        itk_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    number = int(case_id[-3:])
    sitk.WriteImage(image, str(case_dir / f"orig{number}.nii"), useCompression=False)
    sitk.WriteImage(mask, str(case_dir / f"mask{number}.nii"), useCompression=False)
    (case_dir / f"truth{number}.json").write_text(json.dumps(case, indent=2) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("synthetic_dataset"))
    parser.add_argument("--cases", type=int, default=10, help="Number of cases to generate")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--min-daughters", type=int, default=1)
    parser.add_argument("--max-daughters", type=int, default=4)
    parser.add_argument("--curvature", type=float, default=5.0, help="Maximum daughter centerline bend in mm (default: 5)")
    parser.add_argument("--noise-std", type=float, default=8.0, help="Gaussian voxel noise standard deviation (default: 8)")
    parser.add_argument("--size", type=int, nargs=3, default=(160, 160, 256), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--spacing", type=float, nargs=3, default=(0.8, 0.8, 1.0), metavar=("SX", "SY", "SZ"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cases < 1 or args.min_daughters < 0 or args.min_daughters > args.max_daughters:
        raise SystemExit("Require --cases >= 1 and 0 <= --min-daughters <= --max-daughters")
    if any(value < 1 for value in args.size) or any(value <= 0 for value in args.spacing):
        raise SystemExit("--size values must be at least 1 and --spacing values must be positive")
    if args.curvature < 0 or args.noise_std < 0:
        raise SystemExit("--curvature and --noise-std must be nonnegative")
    if args.size[2] * args.spacing[2] <= 60:
        raise SystemExit("The physical z extent must be greater than 60 mm")

    output_dir = args.output.expanduser().resolve()
    for index in range(1, args.cases + 1):
        daughters = int(np.random.default_rng(args.seed + index).integers(args.min_daughters, args.max_daughters + 1))
        case = generate_case(
            index,
            seed=args.seed,
            daughters=daughters,
            size_xyz=tuple(args.size),
            spacing_xyz_mm=tuple(args.spacing),
            curvature_mm=args.curvature,
            noise_std=args.noise_std,
        )
        write_case(case, output_dir)
        print(f"Wrote {case['case_id']} with {daughters} daughter(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
