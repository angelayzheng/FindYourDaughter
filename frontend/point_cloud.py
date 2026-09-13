"""CPU fallback preview when the native VTK scene cannot render."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from core import ScanCase


def _points(affine: np.ndarray, selected: np.ndarray, step: int,
            color: tuple[int, int, int, int]) -> pd.DataFrame:
    indices = np.argwhere(selected)
    if len(indices) > 45000:
        indices = indices[::int(np.ceil(len(indices) / 45000))]
    indices *= step
    positions = indices @ affine[:3, :3].T + affine[:3, 3]
    points = pd.DataFrame(positions, columns=["x", "y", "z"])
    points[["r", "g", "b", "a"]] = color
    return points


def sample_scene(case: ScanCase, *, frame: int = 0, window: float = 400,
                 level: float = 40, show_volume: bool = True,
                 show_mask: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample display points in physical RAS coordinates."""
    image = case.image.volume(frame)
    affine = case.image.geometry.affine_ras
    empty = pd.DataFrame(columns=["x", "y", "z", "r", "g", "b", "a"])
    ct_points = mask_points = empty
    if show_volume:
        step = max(1, int(np.ceil(max(image.shape) / 64)))
        sampled = image[::step, ::step, ::step]
        low, high = level - window / 2, level + window / 2
        chosen = np.isfinite(sampled) & (sampled >= low) & (sampled <= high)
        ct_points = _points(affine, chosen, step, (190, 198, 210, 100))
    if show_mask and case.mask is not None:
        mask = case.mask.volume(frame if case.mask.data.ndim == 4 else 0)
        # Use every foreground voxel before capping points, so thin masks survive.
        mask_points = _points(affine, mask > 0, 1, (240, 48, 60, 255))
    return ct_points, mask_points


def render_projection(case: ScanCase, *, frame: int = 0, window: float = 400,
                      level: float = 40, show_volume: bool = True,
                      show_mask: bool = True, azimuth: float = 30,
                      elevation: float = 25, branches: list[dict] | None = None,
                      show_branches: bool = True,
                      selected_branch: str | None = None) -> tuple[bytes, int, int]:
    """Make a 3-D perspective preview without browser WebGL or host OpenGL."""
    ct_points, mask_points = sample_scene(case, frame=frame, window=window, level=level,
                                          show_volume=show_volume, show_mask=show_mask)
    shape = np.asarray(case.image.shape[:3])
    affine = case.image.geometry.affine_ras
    corners = np.array(np.meshgrid(*[(0, size - 1) for size in shape], indexing="ij"))
    corners = corners.reshape(3, -1).T @ affine[:3, :3].T + affine[:3, 3]
    center = corners.mean(axis=0)
    yaw, pitch = np.deg2rad([azimuth, elevation])
    right = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    up = np.array([-np.sin(pitch) * np.cos(yaw), -np.sin(pitch) * np.sin(yaw), np.cos(pitch)])
    depth = np.cross(right, up)
    axes = np.stack((right, up, depth), axis=1)
    box = (corners - center) @ axes
    width, height = 1000, 700
    span = np.ptp(box[:, :2], axis=0)
    scale = min((width - 100) / max(span[0], 1), (height - 100) / max(span[1], 1))

    def project(positions: np.ndarray) -> np.ndarray:
        view = (positions - center) @ axes
        return np.column_stack((width / 2 + view[:, 0] * scale,
                                height / 2 - view[:, 1] * scale, view[:, 2]))

    canvas = Image.new("RGB", (width, height), (13, 18, 29))
    draw = ImageDraw.Draw(canvas, "RGBA")
    for points, radius, color in ((ct_points, 2, (190, 198, 210, 100)),
                                  (mask_points, 3, (240, 48, 60, 255))):
        if points.empty:
            continue
        positions = points[["x", "y", "z"]].to_numpy(dtype=float)
        projected = project(positions)
        for x, y, _ in projected[np.argsort(projected[:, 2])]:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    box_xy = project(corners)
    for first in range(8):
        for second in range(first + 1, 8):
            if bin(first ^ second).count("1") == 1:
                draw.line((*box_xy[first, :2], *box_xy[second, :2]), fill=(100, 120, 145, 150), width=1)
    if show_branches:
        flip = np.array([-1.0, -1.0, 1.0])
        for branch in branches or []:
            ostium = np.asarray(branch["ostium_xyz_mm"], dtype=float) * flip
            seed = np.asarray(branch["seed_xyz_mm"], dtype=float) * flip
            direction = np.asarray(branch["direction_xyz"], dtype=float) * flip
            direction /= np.linalg.norm(direction)
            radius = float(branch["radius_mm"])
            arrow = seed + direction * max(2.0, radius * 1.5)
            o, s, a = project(np.stack((ostium, seed, arrow)))
            alpha = 255 if selected_branch is None or branch["instance_id"] == selected_branch else 90
            draw.line((o[0], o[1], a[0], a[1]), fill=(84, 198, 211, alpha), width=3)
            angle = np.arctan2(a[1] - s[1], a[0] - s[0])
            wing = np.array([np.cos(angle), np.sin(angle)])
            side = np.array([-wing[1], wing[0]])
            draw.polygon([tuple(a[:2]), tuple(a[:2] - 10 * wing + 4 * side),
                          tuple(a[:2] - 10 * wing - 4 * side)], fill=(84, 198, 211, alpha))
            reference = np.eye(3)[np.argmin(np.abs(direction))]
            u = np.cross(direction, reference)
            u /= np.linalg.norm(u)
            v = np.cross(direction, u)
            angles = np.linspace(0, 2 * np.pi, 49)
            ring = seed + radius * (np.cos(angles[:, None]) * u + np.sin(angles[:, None]) * v)
            draw.line([tuple(point[:2]) for point in project(ring)],
                      fill=(168, 233, 232, alpha), width=2, joint="curve")
            for point, color, size in ((o, (109, 188, 232, alpha), 5),
                                       (s, (168, 233, 232, alpha), 4)):
                draw.ellipse((point[0] - size, point[1] - size,
                              point[0] + size, point[1] + size), fill=color)
    output = BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue(), len(ct_points), len(mask_points)
