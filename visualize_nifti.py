"""Create quick-look PNG previews for every NIfTI volume in a dataset.

By default the script searches ``dataset/`` recursively.  When a directory
contains one image (for example ``orig1.nii``) and one mask (``mask1.nii``),
the mask is drawn over the image in red.  Other NIfTI files are still rendered
on their own, so the script is also useful for differently named datasets.

Examples
--------
    python visualize_nifti.py
    python visualize_nifti.py --dataset path/to/data --output-dir previews
    python visualize_nifti.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np


NIFTI_SUFFIXES = (".nii", ".nii.gz")


def is_nifti(path: Path) -> bool:
    return path.name.lower().endswith(NIFTI_SUFFIXES)


def middle_slices(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return central axial, coronal, and sagittal slices of a 3-D volume."""
    if volume.ndim > 3:
        volume = volume[..., 0]  # preview the first frame of a 4-D scan
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D or 4-D volume, got shape {volume.shape}")

    x, y, z = (axis // 2 for axis in volume.shape)
    # Transposing makes all plots follow the conventional image orientation.
    return volume[:, :, z].T, volume[:, y, :].T, volume[x, :, :].T


def display_limits(volume: np.ndarray) -> tuple[float, float]:
    """Use robust limits so a few extreme voxels do not wash out the preview."""
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, (1, 99))
    return (float(low), float(high)) if high > low else (float(low), float(low + 1))


def nii_to_png(
    nii_path: str | Path,
    output_path: str | Path | None = None,
    *,
    axis: int = 2,
    slice_index: int | None = None,
) -> Path:
    """Save one grayscale PNG slice from a specific NIfTI file.

    Parameters
    ----------
    nii_path:
        Path to a ``.nii`` or ``.nii.gz`` file.
    output_path:
        Destination PNG. If omitted, it is written beside the NIfTI file.
    axis:
        Slice axis: 0 (sagittal), 1 (coronal), or 2 (axial).
    slice_index:
        Index along ``axis``. The middle slice is used when omitted.
    """
    nii_path = Path(nii_path)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")

    volume = nib.load(str(nii_path)).get_fdata(dtype=np.float32)
    if volume.ndim > 3:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D or 4-D volume, got shape {volume.shape}")

    index = volume.shape[axis] // 2 if slice_index is None else slice_index
    if not 0 <= index < volume.shape[axis]:
        raise IndexError(f"slice_index must be between 0 and {volume.shape[axis] - 1}")
    slice_data = np.take(volume, index, axis=axis).T
    vmin, vmax = display_limits(volume)

    if output_path is None:
        name = nii_path.name.removesuffix(".nii.gz").removesuffix(".nii")
        output_path = nii_path.with_name(f"{name}_axis{axis}_slice{index}.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, slice_data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    return output_path


def mask_for_image(image_path: Path, paths: list[Path]) -> Path | None:
    """Find a likely mask next to an image, without assuming exact numbering."""
    if "mask" in image_path.stem.lower():
        return None
    candidates = [path for path in paths if path.parent == image_path.parent and "mask" in path.stem.lower()]
    return candidates[0] if len(candidates) == 1 else None


def render_volume(image_path: Path, all_paths: list[Path], output_dir: Path, show: bool) -> Path:
    image = nib.load(str(image_path))
    volume = image.get_fdata(dtype=np.float32)
    image_slices = middle_slices(volume)
    vmin, vmax = display_limits(volume)

    mask_path = mask_for_image(image_path, all_paths)
    mask_slices = None
    if mask_path is not None:
        mask = nib.load(str(mask_path)).get_fdata(dtype=np.float32)
        if mask.shape[:3] == volume.shape[:3]:
            mask_slices = middle_slices(mask)
        else:
            print(f"Skipping overlay for {image_path}: mask shape {mask.shape} differs from image shape {volume.shape}")

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for index, (axis, slice_data, plane) in enumerate(zip(axes, image_slices, ("Axial", "Coronal", "Sagittal"))):
        axis.imshow(slice_data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        if mask_slices is not None:
            mask_slice = mask_slices[index]
            overlay = np.ma.masked_where(~np.isfinite(mask_slice) | (mask_slice <= 0), mask_slice)
            # A constant binary foreground otherwise maps to the pale end of Reds.
            axis.imshow(overlay, cmap=ListedColormap(["red"]), origin="lower", alpha=0.45, interpolation="nearest")
        axis.set_title(plane)
        axis.axis("off")

    # Preserve subject names and avoid collisions such as two ``orig.nii`` files.
    output_name = f"{image_path.parent.name}_{image_path.name.replace('.nii.gz', '').replace('.nii', '')}.png"
    output_path = output_dir / output_name
    title = image_path.parent.name + " / " + image_path.name
    if mask_path is not None and mask_slices is not None:
        title += f"  (mask: {mask_path.name})"
    figure.suptitle(title)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render central slices from all NIfTI files in a dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Directory to search (default: dataset)")
    parser.add_argument("--output-dir", type=Path, default=Path("nifti_previews"), help="Directory for generated PNGs")
    parser.add_argument("--show", action="store_true", help="Also open each figure interactively")
    args = parser.parse_args()

    paths = sorted(path for path in args.dataset.rglob("*") if path.is_file() and is_nifti(path))
    if not paths:
        raise SystemExit(f"No .nii or .nii.gz files found under: {args.dataset}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [path for path in paths if "mask" not in path.stem.lower()]
    # Render masks too if they do not have a corresponding non-mask image.
    image_paths.extend(path for path in paths if "mask" in path.stem.lower() and not any(other.parent == path.parent and "mask" not in other.stem.lower() for other in paths))

    for image_path in image_paths:
        try:
            output_path = render_volume(image_path, paths, args.output_dir, args.show)
            print(f"Wrote {output_path}")
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
            print(f"Skipped {image_path}: {error}")

    print(f"Rendered {len(image_paths)} preview(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
