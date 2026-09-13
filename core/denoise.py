"""Small local-coherence denoising filter for visualization only."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def denoise_volume(data: np.ndarray, *, tolerance: float = 40.0, min_neighbors: int = 2) -> np.ndarray:
    """Replace locally isolated intensities with a 3-D neighborhood median.

    A voxel is considered supported when at least ``min_neighbors`` of its 26
    immediate neighbors differ by no more than ``tolerance``. This is intended
    to suppress visualization haze and speckle, not to alter evaluator inputs.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if not isinstance(min_neighbors, (int, np.integer)) or not 1 <= min_neighbors <= 26:
        raise ValueError("min_neighbors must be an integer between 1 and 26")
    source = np.asarray(data, dtype=np.float32)
    if source.ndim != 3:
        raise ValueError("denoise_volume expects a 3-D volume")

    finite = np.isfinite(source)
    support = np.zeros(source.shape, dtype=np.uint8)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dx or dy or dz):
                    continue
                source_slices = tuple(
                    slice(max(0, -offset), min(size, size - offset))
                    for offset, size in zip((dz, dy, dx), source.shape)
                )
                neighbor_slices = tuple(
                    slice(max(0, offset), min(size, size + offset))
                    for offset, size in zip((dz, dy, dx), source.shape)
                )
                similar = (
                    finite[source_slices]
                    & finite[neighbor_slices]
                    & (np.abs(source[source_slices] - source[neighbor_slices]) <= tolerance)
                )
                support[source_slices] += similar

    result = source.copy()
    isolated = finite & (support < min_neighbors)
    if np.any(isolated):
        median = ndi.median_filter(source, size=3, mode="nearest")
        result[isolated] = median[isolated]
    return result
