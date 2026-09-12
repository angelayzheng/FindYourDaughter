"""Optional Matplotlib rendering for the core scan models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np

from .scans import PreviewMode, PreviewOptions, Scan, ScanCase


def display_limits(volume: np.ndarray) -> tuple[float, float]:
    """Robust grayscale limits without modifying the underlying tensor."""
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, (1, 99))
    return (float(low), float(high)) if high > low else (float(low), float(low + 1))


def _draw_images(figure, grid, row: int, scan: Scan, mask: Scan | None, options: PreviewOptions) -> None:
    volume = scan.volume(options.frame)
    limits = display_limits(volume)
    for column, dim in enumerate((2, 1, 0)):
        ax = figure.add_subplot(grid[row, column * 2:column * 2 + 2])
        index = volume.shape[dim] // 2
        if dim == options.axis and options.slice_index is not None:
            index = options.slice_index
        data = scan.slice(dim, index, frame=options.frame).T
        remaining = [axis for axis in range(3) if axis != dim]
        spacing = scan.geometry.spacing
        aspect = spacing[remaining[1]] / spacing[remaining[0]]
        ax.imshow(data, cmap="gray", origin="lower", vmin=limits[0], vmax=limits[1], aspect=aspect)
        if mask is not None:
            mask_frame = options.frame if mask.data.ndim == 4 else 0
            mask_slice = mask.slice(dim, index, frame=mask_frame).T
            overlay = np.ma.masked_where(~np.isfinite(mask_slice) | (mask_slice <= 0), mask_slice)
            ax.imshow(overlay, cmap=ListedColormap(["red"]), origin="lower",
                      alpha=options.mask_alpha, interpolation="nearest", aspect=aspect)
        ax.set_title(f"Axis {dim} ({scan.geometry.axis_codes[dim]}) | index {index}")
        ax.axis("off")


def _draw_tensor(figure, grid, row: int, scan: Scan, options: PreviewOptions) -> None:
    index = scan.shape[options.axis] // 2 if options.slice_index is None else options.slice_index
    data = scan.slice(options.axis, index, frame=options.frame)
    remaining = [axis for axis in range(3) if axis != options.axis]
    selection = [":"] * 3
    selection[options.axis] = str(index)
    if scan.data.ndim == 4:
        selection.append(str(options.frame))
    expression = "data[" + ", ".join(selection) + "]"
    height, width = (min(options.patch_size, size) for size in data.shape)
    start_row, start_col = options.patch_origin or ((data.shape[0] - height) // 2, (data.shape[1] - width) // 2)
    if start_row + height > data.shape[0] or start_col + width > data.shape[1]:
        raise IndexError(f"The {height}x{width} patch at {(start_row, start_col)} exceeds slice shape {data.shape}")
    patch = data[start_row:start_row + height, start_col:start_col + width]
    finite = data[np.isfinite(data)]
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if low == high:
        high = low + 1
    full_ax = figure.add_subplot(grid[row, :3])
    patch_ax = figure.add_subplot(grid[row, 3:])
    heatmap = full_ax.imshow(np.ma.masked_invalid(data), origin="upper", cmap="viridis",
                             vmin=low, vmax=high, interpolation="nearest")
    full_ax.add_patch(Rectangle((start_col - 0.5, start_row - 0.5), width, height,
                               fill=False, edgecolor="red", linewidth=1.5))
    full_ax.set_title(f"{expression} | shape {data.shape}")
    full_ax.set_xlabel(f"Array axis {remaining[1]} (column index)")
    full_ax.set_ylabel(f"Array axis {remaining[0]} (row index)")
    figure.colorbar(heatmap, ax=full_ax, shrink=0.8, label="Voxel value (full slice range)")

    patch_map = patch_ax.imshow(np.ma.masked_invalid(patch), origin="upper", cmap="viridis",
                                vmin=low, vmax=high, interpolation="nearest")
    for (r, c), value in np.ndenumerate(patch):
        rgb = patch_map.cmap(patch_map.norm(value))[:3] if np.isfinite(value) else (1, 1, 1)
        luminance = sum(a * b for a, b in zip(rgb, (0.299, 0.587, 0.114)))
        patch_ax.text(c, r, f"{value:.6g}", ha="center", va="center",
                      color="black" if luminance > 0.5 else "white",
                      fontsize=max(5, min(11, 66 / max(height, width))))
    patch_ax.set_xticks(range(width), labels=range(start_col, start_col + width))
    patch_ax.set_yticks(range(height), labels=range(start_row, start_row + height))
    patch_ax.set_xlabel(f"Array axis {remaining[1]} (column index)")
    patch_ax.set_ylabel(f"Array axis {remaining[0]} (row index)")
    patch_ax.set_title(f"Numeric patch | rows {start_row}:{start_row + height}, columns {start_col}:{start_col + width}\n"
                       "Voxel values shown to 6 significant digits")


def export_preview(scan: Scan, path: str | Path, options: PreviewOptions, *, mask: Scan | None = None) -> Path:
    """Save an image overlay, native tensor view, or both as a PNG."""
    ScanCase(scan, mask)  # Reject spatially misaligned overlays.
    scan.slice(options.axis, options.slice_index, frame=options.frame)
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Preview export path must end with .png")
    both = options.mode == PreviewMode.BOTH
    figure = plt.figure(figsize=(14, 10 if both else 5), constrained_layout=True)
    grid = figure.add_gridspec(2 if both else 1, 6)
    try:
        if options.mode in (PreviewMode.IMAGE, PreviewMode.BOTH):
            _draw_images(figure, grid, 0, scan, mask, options)
        if options.mode in (PreviewMode.TENSOR, PreviewMode.BOTH):
            _draw_tensor(figure, grid, 1 if both else 0, scan, options)
        title = f"{scan.name} | shape={scan.shape} | dtype={scan.dtype} | stored={scan.storage_dtype}"
        if mask is not None and options.mode != PreviewMode.TENSOR:
            title += f" | mask: {mask.name}"
        spacing = ", ".join(f"{value:.4g}" for value in scan.geometry.spacing)
        title += f"\nNative array axes={scan.geometry.axis_codes} | spacing=({spacing}) {scan.geometry.spatial_unit}"
        if scan.data.ndim == 4:
            title += f" | frame={options.frame}"
        figure.suptitle(title, fontsize=11)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=options.dpi, bbox_inches="tight")
        if options.show:
            plt.show()
    finally:
        plt.close(figure)
    return path
