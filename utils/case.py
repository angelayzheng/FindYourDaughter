"""Input loading and physical-coordinate helpers."""

from __future__ import annotations

from typing import Any

from backend.inputs import CaseData, load_case

__all__ = ["CaseData", "load_case", "index_to_physical_point"]


def index_to_physical_point(image: Any, index: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert an x/y/z voxel index using SimpleITK's physical coordinate system."""
    point = image.TransformIndexToPhysicalPoint(tuple(int(value) for value in index))
    return tuple(float(value) for value in point)
