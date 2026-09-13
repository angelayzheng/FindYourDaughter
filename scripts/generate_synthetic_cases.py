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
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class Tube:
    centerline_xyz_mm: np.ndarray
    radius_mm: float
    intensity: float
    radii_mm: np.ndarray | None = None


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


def cubic_bezier(start: np.ndarray, control_1: np.ndarray, control_2: np.ndarray,
                 end: np.ndarray, parameter: np.ndarray) -> np.ndarray:
    """Evaluate a cubic Bezier segment at many parameter values."""
    return (
        ((1 - parameter) ** 3)[:, None] * start
        + (3 * (1 - parameter) ** 2 * parameter)[:, None] * control_1
        + (3 * (1 - parameter) * parameter**2)[:, None] * control_2
        + (parameter**3)[:, None] * end
    )


def smooth_centerline(points: np.ndarray, samples_per_segment: int = 24) -> np.ndarray:
    """Create a C1-smooth cubic spline through anatomical landmarks."""
    tangents = np.empty_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents[1:-1] = (points[2:] - points[:-2]) * 0.5
    parameter = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    segments = []
    for index in range(len(points) - 1):
        segments.append(cubic_bezier(
            points[index], points[index] + tangents[index] / 3,
            points[index + 1] - tangents[index + 1] / 3, points[index + 1], parameter,
        ))
    return np.vstack((*segments, points[-1]))


def resample_centerline(points: np.ndarray, count: int) -> np.ndarray:
    """Resample a smooth path uniformly by physical arc length."""
    distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    target = np.linspace(0.0, float(distances[-1]), count)
    return np.column_stack([np.interp(target, distances, points[:, axis]) for axis in range(3)])


def interpolate_path_at_z(points: np.ndarray, z_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a parent point and tangent at a longitudinal location."""
    index = int(np.clip(np.searchsorted(points[:, 2], z_mm), 1, len(points) - 1))
    lower, upper = points[index - 1], points[index]
    fraction = (z_mm - lower[2]) / max(upper[2] - lower[2], 1e-6)
    center = lower + np.clip(fraction, 0.0, 1.0) * (upper - lower)
    return center, normalize(upper - lower)


def create_body_like_volume(
    shape_zyx: tuple[int, int, int],
    spacing_xyz_mm: tuple[float, float, float],
    rng: np.random.Generator,
    noise_std: float,
) -> np.ndarray:
    """Create a level-dependent torso cross-section with CT-like tissue texture."""
    nz, ny, nx = shape_zyx
    physical_size = np.asarray((nx, ny, nz), dtype=float) * spacing_xyz_mm
    x = (np.arange(nx, dtype=float) + 0.5) * spacing_xyz_mm[0]
    y = (np.arange(ny, dtype=float) + 0.5) * spacing_xyz_mm[1]
    xx, yy = np.meshgrid(x, y, indexing="xy")
    center_x, center_y = physical_size[:2] / 2
    image = np.full(shape_zyx, -2048.0, dtype=np.float32)
    tissue = np.zeros(shape_zyx, dtype=np.uint8)

    def ellipse(center: tuple[float, float], radii: tuple[float, float]) -> np.ndarray:
        return (((xx - center[0]) / radii[0]) ** 2
                + ((yy - center[1]) / radii[1]) ** 2) <= 1.0

    def assign(mask: np.ndarray, value: float, tissue_class: int) -> None:
        mask &= body
        image[z_index, mask] = value
        tissue[z_index, mask] = tissue_class

    for z_index in range(nz):
        fraction = (z_index + 0.5) / nz
        thoracic = fraction < 0.48
        taper = 0.88 + 0.10 * np.sin(np.pi * fraction)
        body_x = physical_size[0] * (0.41 if thoracic else 0.44) * taper
        body_y = physical_size[1] * (0.44 if thoracic else 0.47) * taper
        body_center = np.array([
            center_x + 2.5 * np.sin(2 * np.pi * fraction + 0.4),
            center_y + 2.0 * np.sin(np.pi * fraction),
        ])
        body = (((xx - body_center[0]) / body_x) ** 2
                + ((yy - body_center[1]) / body_y) ** 2) <= 1.0
        image[z_index, body] = 45.0 if thoracic else 65.0
        tissue[z_index, body] = 1

        # Subcutaneous fat and a thinner muscular ring provide a nonuniform body wall.
        inner_body = ellipse(tuple(body_center), (body_x * 0.82, body_y * 0.84))
        fat = body & ~inner_body
        image[z_index, fat] = -95.0
        tissue[z_index, fat] = 4
        muscle = body & ~ellipse(tuple(body_center), (body_x * 0.91, body_y * 0.91))
        image[z_index, muscle] = 55.0
        tissue[z_index, muscle] = 5

        posterior = body_center[1] + physical_size[1] * 0.22
        spine = ellipse((body_center[0], posterior),
                        (physical_size[0] * 0.070, physical_size[1] * 0.080))
        assign(spine, 650.0, 3)
        canal = ellipse((body_center[0], posterior - physical_size[1] * 0.015),
                        (physical_size[0] * 0.026, physical_size[1] * 0.030))
        assign(canal, 40.0, 1)

        if thoracic:
            lung_offset = physical_size[0] * 0.19
            lung_radii = (physical_size[0] * 0.13, physical_size[1] * 0.235)
            for lung_center_x in (body_center[0] - lung_offset, body_center[0] + lung_offset):
                lung = ellipse((lung_center_x, body_center[1] - physical_size[1] * 0.035), lung_radii)
                assign(lung, -780.0, 2)

            # Curved high-density ribs are represented by smooth elliptical arcs.
            rib_outer = ellipse(tuple(body_center), (body_x * 0.91, body_y * 0.88))
            rib_inner = ellipse(tuple(body_center), (body_x * 0.86, body_y * 0.83))
            assign(rib_outer & ~rib_inner, 480.0, 3)
        else:
            liver = ellipse((body_center[0] + physical_size[0] * 0.13,
                             body_center[1] - physical_size[1] * 0.05),
                            (physical_size[0] * 0.19, physical_size[1] * 0.22))
            assign(liver, 75.0, 6)
            spleen = ellipse((body_center[0] - physical_size[0] * 0.19,
                              body_center[1] - physical_size[1] * 0.04),
                             (physical_size[0] * 0.09, physical_size[1] * 0.14))
            assign(spleen, 95.0, 6)
            kidney_radii = (physical_size[0] * 0.065, physical_size[1] * 0.11)
            for kidney_center_x in (body_center[0] - physical_size[0] * 0.12,
                                    body_center[0] + physical_size[0] * 0.12):
                assign(ellipse((kidney_center_x, body_center[1] + physical_size[1] * 0.06), kidney_radii), 45.0, 1)

            bowel = ellipse((body_center[0], body_center[1] - physical_size[1] * 0.02),
                            (physical_size[0] * 0.18, physical_size[1] * 0.13))
            assign(bowel, 5.0, 7)

        # A small smooth body-motion field prevents perfectly identical axial slices.
        slice_bias = 3.5 * np.sin(2 * np.pi * fraction * 1.7 + 0.5)
        image[z_index, body] += slice_bias

    if noise_std <= 0:
        return image

    # Low-frequency scanner/body variation plus tissue-dependent quantum noise.
    smooth_noise = gaussian_filter(rng.normal(size=shape_zyx), sigma=(3.0, 2.5, 2.5), mode="reflect")
    smooth_noise /= max(float(smooth_noise.std()), 1e-6)
    image += (smooth_noise * noise_std * 0.9).astype(np.float32)
    tissue_noise = np.select(
        [tissue == 0, tissue == 1, tissue == 2, tissue == 3,
         tissue == 4, tissue == 5, tissue == 6, tissue == 7],
        [0.8, 1.2, 1.6, 2.0, 1.0, 1.3, 1.15, 1.5],
        default=1.0,
    )
    image += (rng.normal(size=shape_zyx) * noise_std * 1.25 * tissue_noise).astype(np.float32)
    return image


def tube_distance_mask(
    shape_zyx: tuple[int, int, int],
    spacing_xyz_mm: tuple[float, float, float],
    tube: Tube,
) -> np.ndarray:
    """Rasterize a round tube around a polyline into native z/y/x array order."""
    points = tube.centerline_xyz_mm
    spacing = np.asarray(spacing_xyz_mm, dtype=float)
    radii = tube.radii_mm
    if radii is None:
        radii = np.full(len(points), tube.radius_mm, dtype=float)
    maximum_radius = float(np.max(radii))
    lower = np.maximum(np.floor((points.min(axis=0) - maximum_radius) / spacing).astype(int), 0)
    upper = np.minimum(np.ceil((points.max(axis=0) + maximum_radius) / spacing).astype(int), np.asarray(shape_zyx[::-1]) - 1)
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
    for point_index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared == 0:
            candidate = np.sum((coordinates - start) ** 2, axis=-1)
            local_radius = radii[point_index]
        else:
            fraction = np.clip(np.sum((coordinates - start) * segment, axis=-1) / length_squared, 0, 1)
            closest = start + fraction[..., None] * segment
            candidate = np.sum((coordinates - closest) ** 2, axis=-1)
            local_radius = radii[point_index] + fraction * (radii[point_index + 1] - radii[point_index])
        distance_squared = np.minimum(distance_squared, candidate - local_radius**2)

    local = distance_squared <= 0
    result = np.zeros(shape_zyx, dtype=bool)
    result[lower[2] : upper[2] + 1, lower[1] : upper[1] + 1, lower[0] : upper[0] + 1] = local.transpose(2, 1, 0)
    return result


def ellipsoid_mask(
    shape_zyx: tuple[int, int, int],
    spacing_xyz_mm: tuple[float, float, float],
    center_xyz_mm: np.ndarray,
    radii_xyz_mm: np.ndarray,
) -> np.ndarray:
    """Rasterize one solid ellipsoid into a native z/y/x array."""
    spacing = np.asarray(spacing_xyz_mm, dtype=float)
    shape_xyz = np.asarray(shape_zyx[::-1])
    lower = np.maximum(np.floor((center_xyz_mm - radii_xyz_mm) / spacing).astype(int), 0)
    upper = np.minimum(np.ceil((center_xyz_mm + radii_xyz_mm) / spacing).astype(int), shape_xyz - 1)
    if np.any(lower > upper):
        return np.zeros(shape_zyx, dtype=bool)
    ix, iy, iz = np.meshgrid(
        np.arange(lower[0], upper[0] + 1),
        np.arange(lower[1], upper[1] + 1),
        np.arange(lower[2], upper[2] + 1),
        indexing="ij",
    )
    coordinates = np.stack((ix, iy, iz), axis=-1).astype(float) * spacing
    local = np.sum(((coordinates - center_xyz_mm) / radii_xyz_mm) ** 2, axis=-1) <= 1
    result = np.zeros(shape_zyx, dtype=bool)
    result[lower[2] : upper[2] + 1, lower[1] : upper[1] + 1, lower[0] : upper[0] + 1] = local.transpose(2, 1, 0)
    return result


def centerline_distance(first: np.ndarray, second: np.ndarray) -> float:
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return float(distances.min())


def add_organ_blobs(
    image: np.ndarray,
    shape_zyx: tuple[int, int, int],
    spacing_xyz_mm: tuple[float, float, float],
    physical_size_xyz_mm: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> None:
    """Add solid organs or lesions with varied size and CT-like intensity."""
    center_xy = physical_size_xyz_mm[:2] / 2
    for _ in range(count):
        center = np.array([
            center_xy[0] + rng.uniform(-physical_size_xyz_mm[0] * 0.25, physical_size_xyz_mm[0] * 0.25),
            center_xy[1] + rng.uniform(-physical_size_xyz_mm[1] * 0.25, physical_size_xyz_mm[1] * 0.25),
            rng.uniform(physical_size_xyz_mm[2] * 0.12, physical_size_xyz_mm[2] * 0.88),
        ])
        radii = np.array([
            rng.uniform(5.0, physical_size_xyz_mm[0] * 0.13),
            rng.uniform(5.0, physical_size_xyz_mm[1] * 0.13),
            rng.uniform(4.0, min(22.0, physical_size_xyz_mm[2] * 0.10)),
        ])
        intensity = float(rng.choice((20.0, 80.0, 130.0, 260.0, 700.0)))
        image[ellipsoid_mask(shape_zyx, spacing_xyz_mm, center, radii)] = intensity


def sample_distractor_tube(
    parent_centerline_xyz_mm: np.ndarray,
    parent_radii_mm: np.ndarray,
    physical_size_xyz_mm: np.ndarray,
    rng: np.random.Generator,
) -> Tube:
    """Create a bright, disconnected tube that may resemble a daughter."""
    z_mm = float(rng.uniform(physical_size_xyz_mm[2] * 0.16, physical_size_xyz_mm[2] * 0.84))
    parent_center, tangent = interpolate_path_at_z(parent_centerline_xyz_mm, z_mm)
    parent_radius = float(np.interp(z_mm, parent_centerline_xyz_mm[:, 2], parent_radii_mm))
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tangent, reference))) > 0.92:
        reference = np.array([0.0, 1.0, 0.0])
    radial_1 = normalize(np.cross(tangent, reference))
    radial_2 = normalize(np.cross(tangent, radial_1))
    angle = float(rng.uniform(0, 2 * np.pi))
    radial = normalize(np.cos(angle) * radial_1 + np.sin(angle) * radial_2)
    gap = float(rng.uniform(6.0, 16.0))
    start = parent_center + radial * (parent_radius + gap)
    direction = normalize(radial * rng.uniform(0.55, 0.9) + tangent * rng.uniform(-0.65, 0.65))
    side = normalize(np.cross(direction, radial))
    length = float(rng.uniform(35.0, 70.0))
    hook = float(rng.uniform(5.0, 14.0))
    hook_start = start + direction * length * 0.72
    landmarks = np.array([
        start,
        start + direction * length * 0.34,
        hook_start,
        hook_start + direction * length * 0.16 + side * hook,
        hook_start + direction * length * 0.16 + side * hook * 1.35,
    ])
    points = resample_centerline(smooth_centerline(landmarks, samples_per_segment=12), 49)
    radius = float(rng.uniform(1.5, 4.5))
    intensity = float(rng.choice((170.0, 280.0, 700.0, 900.0)))
    return Tube(points, radius, intensity)


def sample_branch(
    aorta_center_xyz_mm: np.ndarray,
    aorta_radius_mm: float,
    z_mm: float,
    length_mm: float,
    radius_mm: float,
    angle: float,
    max_curvature_mm: float,
    min_branch_angle_deg: float,
    max_branch_angle_deg: float,
    hook_mm: float,
    rng: np.random.Generator,
    *,
    parent_centerline_xyz_mm: np.ndarray | None = None,
    parent_radii_mm: np.ndarray | None = None,
) -> tuple[Tube, DaughterTruth]:
    if parent_centerline_xyz_mm is None:
        center = aorta_center_xyz_mm.copy()
        center[2] = z_mm
        tangent = np.array([0.0, 0.0, 1.0])
        parent_radius = aorta_radius_mm
    else:
        center, tangent = interpolate_path_at_z(parent_centerline_xyz_mm, z_mm)
        parent_radius = float(np.interp(z_mm, parent_centerline_xyz_mm[:, 2], parent_radii_mm))

    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tangent, reference))) > 0.92:
        reference = np.array([0.0, 1.0, 0.0])
    radial_1 = normalize(np.cross(tangent, reference))
    radial_2 = normalize(np.cross(tangent, radial_1))
    radial = normalize(np.cos(angle) * radial_1 + np.sin(angle) * radial_2)
    branch_angle = np.deg2rad(rng.uniform(min_branch_angle_deg, max_branch_angle_deg))
    direction = normalize(np.cos(branch_angle) * tangent + np.sin(branch_angle) * radial)
    ostium = center + radial * parent_radius
    curvature_axis = normalize(np.cross(direction, radial))
    curvature = float(rng.uniform(-max_curvature_mm, max_curvature_mm))
    control_1 = ostium + direction * length_mm * 0.30 + radial * radius_mm * 0.15
    hook_start = ostium + direction * length_mm * 0.68
    hook_mid = hook_start + direction * length_mm * 0.17 + curvature_axis * (curvature + hook_mm)
    end = hook_start + direction * length_mm * 0.17 + curvature_axis * (curvature + hook_mm * 1.35)
    points = smooth_centerline(
        np.array([ostium, control_1, hook_start, hook_mid, end]),
        samples_per_segment=12,
    )
    points = resample_centerline(points, 49)
    # A broad root collar overlaps the parent wall and eases into the daughter lumen.
    root_radius = max(radius_mm * 1.35, radius_mm + parent_radius * 0.08)
    progress = np.linspace(0.0, 1.0, len(points))
    radii = np.where(
        progress < 0.22,
        root_radius + (radius_mm * 1.08 - root_radius) * (progress / 0.22),
        radius_mm * (1.08 - 0.20 * (progress - 0.22) / 0.78),
    )
    tube = Tube(points, radius_mm, 500.0, radii)
    arc_lengths = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    seed_distance = min(5.0, float(arc_lengths[-1]))
    seed = np.array([np.interp(seed_distance, arc_lengths, points[:, axis]) for axis in range(3)])
    truth = DaughterTruth(
        ostium_xyz_mm=[round(float(value), 4) for value in ostium],
        seed_xyz_mm=[round(float(value), 4) for value in seed],
        radius_mm=round(float(radius_mm), 4),
        direction_xyz=[round(float(value), 6) for value in normalize(seed - ostium)],
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
    min_branch_angle_deg: float = 35.0,
    max_branch_angle_deg: float = 75.0,
    min_daughter_length_mm: float = 35.0,
    max_daughter_length_mm: float = 70.0,
    hook_mm: float = 8.0,
    distractor_tubes: int = 8,
    organ_blobs: int = 8,
) -> dict:
    rng = np.random.default_rng(seed + case_index)
    nx, ny, nz = size_xyz
    shape_zyx = (nz, ny, nx)
    physical_size = np.asarray(size_xyz, dtype=float) * spacing_xyz_mm
    aorta_center = physical_size[:2] / 2
    aorta_center_xyz = np.array([aorta_center[0], aorta_center[1], 0.0])
    aorta_radius = float(rng.uniform(10.5, 13.5))
    scale = min(1.0, float(np.min(physical_size[:2])) / 128.0)
    arch_landmarks = np.array([
        [aorta_center[0], aorta_center[1] + 2.0 * scale, 0.0],
        [aorta_center[0] + 3.0 * scale, aorta_center[1] + 1.0 * scale, physical_size[2] * 0.18],
        [aorta_center[0] + 8.0 * scale, aorta_center[1] - 2.0 * scale, physical_size[2] * 0.38],
        [aorta_center[0] + 11.0 * scale, aorta_center[1] - 5.0 * scale, physical_size[2] * 0.52],
        [aorta_center[0] + 5.0 * scale, aorta_center[1] - 6.0 * scale, physical_size[2] * 0.63],
        [aorta_center[0] - 5.0 * scale, aorta_center[1] - 4.0 * scale, physical_size[2] * 0.74],
        [aorta_center[0] - 8.0 * scale, aorta_center[1] - 1.0 * scale, physical_size[2] * 0.87],
        [aorta_center[0] - 7.0 * scale, aorta_center[1] + 2.0 * scale, physical_size[2]],
    ])
    main_points = resample_centerline(smooth_centerline(arch_landmarks), 193)
    parent_radii = aorta_radius * (1.04 - 0.14 * np.linspace(0.0, 1.0, len(main_points)))
    parent_radii *= 1.0 + 0.025 * np.sin(np.linspace(0.0, 3.0 * np.pi, len(main_points)))
    parent = Tube(main_points, aorta_radius, 350.0, parent_radii)
    parent_mask = tube_distance_mask(shape_zyx, spacing_xyz_mm, parent)
    image = create_body_like_volume(shape_zyx, spacing_xyz_mm, rng, noise_std)
    add_organ_blobs(image, shape_zyx, spacing_xyz_mm, physical_size, organ_blobs, rng)
    image[parent_mask] = parent.intensity

    z_min = max(22.0, physical_size[2] * 0.16)
    z_max = min(physical_size[2] - 22.0, physical_size[2] * 0.86)
    z_positions = np.linspace(z_min, z_max, daughters + 2)[1:-1] if daughters else np.array([])
    daughter_truth: list[DaughterTruth] = []
    daughter_tubes: list[Tube] = []
    used_angles: list[float] = []
    minimum_angle = max(0.65, 2 * np.pi / max(daughters, 1) * 0.65)
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
            length = float(rng.uniform(min_daughter_length_mm, max_daughter_length_mm))
            daughter, truth = sample_branch(
                aorta_center_xyz,
                aorta_radius,
                float(z_mm),
                length,
                radius,
                angle,
                curvature_mm,
                min_branch_angle_deg,
                max_branch_angle_deg,
                hook_mm,
                rng,
                parent_centerline_xyz_mm=main_points,
                parent_radii_mm=parent_radii,
            )
            clearance = radius + max((other.radius_mm for other in daughter_tubes), default=0.0) + 1.0
            too_close = any(
                np.min(np.linalg.norm(
                    daughter.centerline_xyz_mm[6:, None] - other.centerline_xyz_mm[None, 6:], axis=2,
                )) < clearance
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

    distractor_list: list[Tube] = []
    for _ in range(distractor_tubes):
        for _attempt in range(100):
            distractor = sample_distractor_tube(main_points, parent_radii, physical_size, rng)
            parent_clearance = max(parent_radii) + distractor.radius_mm + 2.0
            if centerline_distance(distractor.centerline_xyz_mm, main_points) <= parent_clearance:
                continue
            if any(
                centerline_distance(distractor.centerline_xyz_mm, other.centerline_xyz_mm)
                <= distractor.radius_mm + other.radius_mm + 1.0
                for other in (*daughter_tubes, *distractor_list)
            ):
                continue
            distractor_list.append(distractor)
            break
        else:
            raise RuntimeError("Could not place disconnected distractor tubes")
    for distractor in distractor_list:
        image[tube_distance_mask(shape_zyx, spacing_xyz_mm, distractor)] = distractor.intensity

    case_id = f"subject{case_index:03d}"
    return {
        "case_id": case_id,
        "generator": {
            "seed": seed + case_index,
            "noise_std": round(float(noise_std), 4),
            "max_curvature_mm": round(float(curvature_mm), 4),
            "branch_angle_deg": [round(float(min_branch_angle_deg), 4), round(float(max_branch_angle_deg), 4)],
            "daughter_length_mm": [round(float(min_daughter_length_mm), 4), round(float(max_daughter_length_mm), 4)],
            "hook_mm": round(float(hook_mm), 4),
            "distractor_tubes": distractor_tubes,
            "organ_blobs": organ_blobs,
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
    parser.add_argument("--min-branch-angle", type=float, default=35.0, help="Minimum daughter angle from the parent axis in degrees")
    parser.add_argument("--max-branch-angle", type=float, default=75.0, help="Maximum daughter angle from the parent axis in degrees")
    parser.add_argument("--min-daughter-length", type=float, default=35.0, help="Minimum daughter length in mm")
    parser.add_argument("--max-daughter-length", type=float, default=70.0, help="Maximum daughter length in mm")
    parser.add_argument("--hook", type=float, default=8.0, help="Terminal daughter hook size in mm")
    parser.add_argument("--distractor-tubes", type=int, default=8, help="Disconnected bright tubes added to the CT only")
    parser.add_argument("--organ-blobs", type=int, default=8, help="Solid organ/lesion blobs added to the CT only")
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
    if not 0 < args.min_branch_angle <= args.max_branch_angle < 90:
        raise SystemExit("Branch angles must satisfy 0 < min <= max < 90 degrees")
    if args.min_daughter_length <= 5 or args.min_daughter_length > args.max_daughter_length:
        raise SystemExit("Daughter lengths must satisfy 5 < min <= max mm")
    if args.hook < 0 or args.distractor_tubes < 0 or args.organ_blobs < 0:
        raise SystemExit("Hook size and object counts must be nonnegative")
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
            min_branch_angle_deg=args.min_branch_angle,
            max_branch_angle_deg=args.max_branch_angle,
            min_daughter_length_mm=args.min_daughter_length,
            max_daughter_length_mm=args.max_daughter_length,
            hook_mm=args.hook,
            distractor_tubes=args.distractor_tubes,
            organ_blobs=args.organ_blobs,
        )
        write_case(case, output_dir)
        print(f"Wrote {case['case_id']} with {daughters} daughter(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
