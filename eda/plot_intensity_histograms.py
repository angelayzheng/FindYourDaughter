#!/usr/bin/env python3
"""Export voxel-intensity histograms for a directory of NIfTI volumes.

The histogram uses scaled NIfTI values. For calibrated CT data these are
typically Hounsfield units. Volumes are loaded one at a time through ``core``
so gzip-compressed files with a misleading ``.nii`` extension are supported.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow this file to be executed directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import Scan


NIFTI_SUFFIXES = (".nii", ".nii.gz")


def is_nifti(path: Path) -> bool:
    return path.name.lower().endswith(NIFTI_SUFFIXES)


def is_mask(path: Path) -> bool:
    return "mask" in path.stem.lower()


def find_mask(path: Path, all_paths: list[Path]) -> Path | None:
    """Find the single neighboring mask associated with a volume."""
    if is_mask(path):
        return path
    masks = [candidate for candidate in all_paths if candidate.parent == path.parent and is_mask(candidate)]
    return masks[0] if len(masks) == 1 else None


def histogram_values(
    volume: np.ndarray,
    bins: int,
    padding_floor: float,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    values = np.asarray(volume, dtype=np.float32).ravel()
    values = values[np.isfinite(values)]
    values = values[values > padding_floor]
    if values.size == 0:
        raise ValueError("volume contains no finite non-padded voxel values")
    low, high = float(values.min()), float(values.max())
    if low == high:
        low -= 0.5
        high += 0.5
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    return counts, edges, low, high, int(values.size)


def output_name(path: Path) -> str:
    stem = path.name.removesuffix(".nii.gz").removesuffix(".nii")
    return f"{path.parent.name}_{stem}_histogram.png"


def plot_histogram(
    path: Path,
    output_dir: Path,
    *,
    bins: int = 256,
    padding_floor: float = -2048.0,
    linear_y: bool = False,
    mask_path: Path | None = None,
) -> Path:
    """Load one volume and export its voxel-intensity histogram."""
    scan = Scan.from_nifti(path)
    counts, edges, low, high, finite_count = histogram_values(
        scan.volume(), bins, padding_floor
    )
    centers = (edges[:-1] + edges[1:]) / 2
    output_path = output_dir / output_name(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask_voxel_count: int | None = None
    masked_counts: np.ndarray | None = None
    if mask_path is not None:
        mask = Scan.from_nifti(mask_path).volume()
        if mask.shape != scan.volume().shape:
            raise ValueError(f"mask shape {mask.shape} differs from image shape {scan.volume().shape}")
        foreground = np.isfinite(mask) & (mask != 0)
        mask_voxel_count = int(np.count_nonzero(foreground))
        if not is_mask(path):
            masked_values = scan.volume()[foreground]
            masked_values = masked_values[np.isfinite(masked_values)]
            masked_counts, _ = np.histogram(masked_values, bins=edges)

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.step(centers, counts, where="mid", color="#1f5a85", linewidth=1.2)
    axis.fill_between(centers, counts, step="mid", color="#5b9bd5", alpha=0.25)
    if masked_counts is not None:
        axis.step(
            centers,
            masked_counts,
            where="mid",
            color="#c0392b",
            linewidth=1.4,
            label="Mask-selected intensities",
        )
        axis.fill_between(centers, masked_counts, step="mid", color="#e74c3c", alpha=0.2)
    axis.set_title(f"Voxel intensity distribution: {path.parent.name} / {path.name}")
    axis.set_xlabel("Voxel intensity (scaled NIfTI value)")
    axis.set_ylabel("Voxel count")
    axis.set_xlim(low, high)
    if not linear_y:
        axis.set_yscale("log")
    if mask_voxel_count is not None and mask_voxel_count > 0:
        axis.axhline(
            mask_voxel_count,
            color="#c0392b",
            linestyle="--",
            linewidth=1.4,
            label=f"Mask voxels ({mask_voxel_count:,})",
        )
        axis.legend(loc="upper right")
    axis.grid(True, alpha=0.2)
    axis.text(
        0.99,
        0.98,
        "\n".join(
            (
                f"n non-padded = {finite_count:,}",
                f"mask voxels = {mask_voxel_count:,}" if mask_voxel_count is not None else "mask voxels = unavailable",
                f"range shown = [{low:.3g}, {high:.3g}]",
            )
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("nifti_histograms"))
    parser.add_argument("--bins", type=int, default=256, help="Number of histogram bins")
    parser.add_argument(
        "--padding-floor",
        type=float,
        default=-2048.0,
        help="Exclude finite values at or below this CT padding floor (default: -2048)",
    )
    parser.add_argument("--include-masks", action="store_true", help="Also plot mask voxel values")
    parser.add_argument("--linear-y", action="store_true", help="Use a linear voxel-count axis")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset}")
    if args.bins < 1:
        raise SystemExit("--bins must be positive")
    all_paths = sorted(path for path in dataset.rglob("*") if path.is_file() and is_nifti(path))
    paths = [path for path in all_paths if args.include_masks or not is_mask(path)]
    if not paths:
        raise SystemExit(f"No NIfTI volumes found under: {dataset}")

    output_dir = args.output_dir.expanduser().resolve()
    written = 0
    for path in paths:
        try:
            mask_path = find_mask(path, all_paths)
            output_path = plot_histogram(
                path,
                output_dir,
                bins=args.bins,
                padding_floor=args.padding_floor,
                linear_y=args.linear_y,
                mask_path=mask_path,
            )
        except (OSError, ValueError) as error:
            print(f"Skipped {path}: {error}")
            continue
        print(f"Wrote {output_path}")
        written += 1

    print(f"Generated {written} histogram(s) in {output_dir}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
