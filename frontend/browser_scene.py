"""Bounded physical-space scene for the CPU-only browser preview."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core import ScanCase


def _limit_points(indices: np.ndarray, limit: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    return indices[np.linspace(0, len(indices) - 1, limit, dtype=np.intp)]


def scene_payload(case: ScanCase, *, frame: int = 0, window: float = 400,
                  level: float = 40, show_volume: bool = True,
                  show_mask: bool = True, sampling_limit: int = 72,
                  prediction: dict | None = None, show_branches: bool = True,
                  selected_branch: str | None = None) -> dict:
    """Sample CT and mask in NIfTI RAS coordinates without copying the full volume."""
    volume = case.image.volume(frame)
    affine = case.image.geometry.affine_ras
    shape = np.asarray(volume.shape)
    corners_ijk = np.array(np.meshgrid(*[(0, size - 1) for size in shape], indexing="ij"))
    corners_ijk = corners_ijk.reshape(3, -1).T
    corners = corners_ijk @ affine[:3, :3].T + affine[:3, 3]
    center = corners.mean(axis=0)
    span = max(float(np.ptp(corners, axis=0).max()), 1.0)

    def normalize(positions: np.ndarray) -> list[list[float]]:
        return np.round((positions - center) / span, 4).tolist()

    def transform(indices: np.ndarray) -> list[list[float]]:
        return normalize(indices @ affine[:3, :3].T + affine[:3, 3])

    ct_points: list[list[float]] = []
    if show_volume:
        step = max(1, int(np.ceil(max(shape) / sampling_limit)))
        sampled = volume[::step, ::step, ::step]
        low, high = level - window / 2, level + window / 2
        selected = np.isfinite(sampled) & (sampled >= low) & (sampled <= high)
        indices = _limit_points(np.argwhere(selected), 18000)
        if len(indices):
            values = sampled[tuple(indices.T)]
            brightness = np.clip((values - low) / window, 0, 1)
            coordinates = transform(indices * step)
            ct_points = [point + [int(55 + value * 140)]
                         for point, value in zip(coordinates, brightness)]

    mask_points: list[list[float]] = []
    if show_mask and case.mask is not None:
        mask = case.mask.volume(frame if case.mask.data.ndim == 4 else 0) > 0
        occupied = [np.flatnonzero(np.any(mask, axis=tuple(other for other in range(3)
                                                         if other != axis)))
                    for axis in range(3)]
        if all(len(axis) for axis in occupied):
            starts = np.array([axis[0] for axis in occupied])
            bounds = tuple(slice(int(axis[0]), int(axis[-1]) + 1) for axis in occupied)
            cropped = mask[bounds]
            # Only exposed voxels form the surface; an interior point cannot be seen.
            interior = cropped.copy()
            for axis in range(3):
                before = [slice(None)] * 3
                after = [slice(None)] * 3
                before[axis] = slice(1, None)
                after[axis] = slice(None, -1)
                interior[tuple(before)] &= cropped[tuple(after)]
                interior[tuple(after)] &= cropped[tuple(before)]
                boundary = [slice(None)] * 3
                boundary[axis] = 0
                interior[tuple(boundary)] = False
                boundary[axis] = -1
                interior[tuple(boundary)] = False
            indices = _limit_points(np.argwhere(cropped & ~interior), 14000)
            mask_points = transform(indices + starts)

    branches = []
    if show_branches and prediction is not None:
        lps_to_ras = np.array([-1., -1., 1.])
        for daughter in prediction.get("daughters", []):
            ostium = np.asarray(daughter["ostium_xyz_mm"], dtype=float) * lps_to_ras
            seed = np.asarray(daughter["seed_xyz_mm"], dtype=float) * lps_to_ras
            direction = np.asarray(daughter["direction_xyz"], dtype=float) * lps_to_ras
            direction /= np.linalg.norm(direction)
            reference = np.eye(3)[np.argmin(np.abs(direction))]
            u = np.cross(direction, reference)
            u /= np.linalg.norm(u)
            v = np.cross(direction, u)
            angles = np.linspace(0, 2 * np.pi, 49)
            radius = float(daughter["radius_mm"])
            ring = seed + radius * (np.cos(angles[:, None]) * u + np.sin(angles[:, None]) * v)
            arrow = seed + direction * max(2.0, radius * 1.5)
            branches.append({"id": daughter["instance_id"], "ostium": normalize(ostium),
                             "seed": normalize(seed), "arrow": normalize(arrow),
                             "ring": normalize(ring), "radius_mm": radius,
                             "parent_id": daughter.get("parent_instance_id", "aorta"),
                             "ostium_lps_mm": daughter["ostium_xyz_mm"],
                             "seed_lps_mm": daughter["seed_xyz_mm"],
                             "direction_lps": daughter["direction_xyz"]})

    return {"ct": ct_points, "mask": mask_points, "branches": branches,
            "selected_branch": selected_branch,
            "corners": transform(corners_ijk), "count_ct": len(ct_points),
            "count_mask": len(mask_points)}


def scene_html(payload: dict) -> str:
    """Return a self-contained Canvas 2D viewer; no WebGL or network assets."""
    template = (Path(__file__).with_name("browser_scene.html")
                .read_text(encoding="utf-8"))
    data = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    return template.replace("__SCENE_DATA__", data)
