"""Create quick-look PNG previews for every NIfTI volume in a dataset.

By default the script searches ``dataset/`` recursively.  When a directory
contains one image (for example ``orig1.nii``) and one mask (``mask1.nii``),
the mask is drawn over the image in red.  Other NIfTI files are still rendered
on their own, so the script is also useful for differently named datasets.

Examples
--------
    python scripts/visualize_nifti.py
    python scripts/visualize_nifti.py --dataset path/to/data --output-dir previews
    python scripts/visualize_nifti.py --show
    python scripts/visualize_nifti.py --mode both --export-numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# Support both `python scripts/visualize_nifti.py` and module imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import PreviewMode, PreviewOptions, Scan, ScanCase
from core.preview import display_limits


NIFTI_SUFFIXES = (".nii", ".nii.gz")


def is_nifti(path: Path) -> bool:
    return path.name.lower().endswith(NIFTI_SUFFIXES)


def load_volume(path: str | Path) -> np.ndarray:
    """Compatibility wrapper returning the core scan's scaled NumPy tensor."""
    return Scan.from_nifti(path).data


def middle_slices(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return central axial, coronal, and sagittal slices of a 3-D volume."""
    if volume.ndim > 3:
        volume = volume[..., 0]  # preview the first frame of a 4-D scan
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D or 4-D volume, got shape {volume.shape}")

    x, y, z = (axis // 2 for axis in volume.shape)
    # Transposing makes all plots follow the conventional image orientation.
    return volume[:, :, z].T, volume[:, y, :].T, volume[x, :, :].T


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

    volume = load_volume(nii_path)
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


def render_volume(
    image_path: Path, all_paths: list[Path], output_dir: Path, show: bool,
    *, options: PreviewOptions | None = None, export_numpy: bool = False,
) -> Path:
    options = options or PreviewOptions(show=show)
    mask_path = mask_for_image(image_path, all_paths)
    case = ScanCase.from_nifti(image_path, mask_path)
    # Preserve subject names and avoid collisions such as two ``orig.nii`` files.
    stem = image_path.name[:-7] if image_path.name.lower().endswith(".nii.gz") else image_path.stem
    output_stem = f"{image_path.parent.name}_{stem}"
    suffix = "" if options.mode == PreviewMode.IMAGE else f"_{options.mode.value}"
    output_path = case.export_preview(output_dir / f"{output_stem}{suffix}.png", options)
    if export_numpy:
        print(f"Wrote {case.image.export_numpy(output_dir / f'{output_stem}.npy')}")
        if case.mask is not None:
            print(f"Wrote {case.mask.export_numpy(output_dir / f'{output_stem}_mask.npy')}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render central slices from all NIfTI files in a dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Directory to search (default: dataset)")
    parser.add_argument("--output-dir", type=Path, default=Path("nifti_previews"), help="Directory for generated PNGs")
    parser.add_argument("--show", action="store_true", help="Also open each figure interactively")
    parser.add_argument("--mode", choices=[mode.value for mode in PreviewMode], default="image",
                        help="Preview image slices, a NumPy tensor slice, or both (default: image)")
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=2, help="Native tensor slice axis (default: 2)")
    parser.add_argument("--slice-index", type=int, help="Slice index along --axis (default: middle)")
    parser.add_argument("--frame", type=int, default=0, help="Frame for 4-D scans (default: 0)")
    parser.add_argument("--patch-size", type=int, default=6, help="Numeric patch edge length, 1-16 (default: 6)")
    parser.add_argument("--patch-origin", type=int, nargs=2, metavar=("ROW", "COLUMN"),
                        help="Numeric patch origin in the native 2-D slice (default: centered)")
    parser.add_argument("--export-numpy", action="store_true", help="Also save complete image/mask tensors as .npy")
    args = parser.parse_args()
    try:
        options = PreviewOptions(mode=args.mode, axis=args.axis, slice_index=args.slice_index,
                                 frame=args.frame, patch_size=args.patch_size,
                                 patch_origin=tuple(args.patch_origin) if args.patch_origin is not None else None,
                                 show=args.show)
    except ValueError as error:
        parser.error(str(error))

    paths = sorted(path for path in args.dataset.rglob("*") if path.is_file() and is_nifti(path))
    if not paths:
        raise SystemExit(f"No .nii or .nii.gz files found under: {args.dataset}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [path for path in paths if "mask" not in path.stem.lower()]
    # Render masks too if they do not have a corresponding non-mask image.
    image_paths.extend(path for path in paths if "mask" in path.stem.lower() and not any(other.parent == path.parent and "mask" not in other.stem.lower() for other in paths))

    rendered = 0
    for image_path in image_paths:
        try:
            output_path = render_volume(image_path, paths, args.output_dir, args.show,
                                        options=options, export_numpy=args.export_numpy)
            rendered += 1
            print(f"Wrote {output_path}")
        except (OSError, EOFError, ValueError, IndexError, nib.filebasedimages.ImageFileError) as error:
            print(f"Skipped {image_path}: {error}")

    print(f"Rendered {rendered} preview(s) to {args.output_dir}; skipped {len(image_paths) - rendered}")


if __name__ == "__main__":
    main()
