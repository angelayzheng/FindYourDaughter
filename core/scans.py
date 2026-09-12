"""Scan tensors in native NIfTI voxel order: data[i, j, k, optional_frame].

NIfTI affines use NiBabel's RAS convention, in the header's spatial units.
They are metadata, not evaluator coordinates: the evaluator continues to use
SimpleITK and TransformIndexToPhysicalPoint for physical LPS coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import gzip
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.typing import NDArray


class PreviewMode(str, Enum):
    IMAGE = "image"
    TENSOR = "tensor"
    BOTH = "both"


@dataclass(frozen=True)
class VolumeViewOptions:
    """Desktop volume display settings; they never modify the source tensor.

    max_dimension bounds the sampled CT grid; masks and slices remain full size.
    window and level refer to scaled voxel values (HU for calibrated CT scans).
    """

    frame: int = 0
    max_dimension: int = 192
    window: float = 400.0
    level: float = 40.0
    min_intensity: float | None = None
    opacity: float = 0.12

    def __post_init__(self) -> None:
        if not isinstance(self.frame, (int, np.integer)) or self.frame < 0:
            raise ValueError("frame must be a nonnegative integer")
        if not isinstance(self.max_dimension, (int, np.integer)) or not 16 <= self.max_dimension <= 512:
            raise ValueError("max_dimension must be an integer between 16 and 512")
        if not np.isfinite((self.window, self.level, self.opacity)).all() or self.window <= 0:
            raise ValueError("window must be positive and all display settings must be finite")
        if self.min_intensity is not None and not np.isfinite(self.min_intensity):
            raise ValueError("min_intensity must be finite when provided")
        if not 0 <= self.opacity <= 0.5:
            raise ValueError("opacity must be between 0 and 0.5")


@dataclass(frozen=True)
class PreviewOptions:
    """Settings used by Scan.export_preview and ScanCase.export_preview.

    axis/slice_index select a tensor slice; frame selects a 4-D time frame.
    patch_origin is (row, column) within the untransposed tensor slice.
    """

    mode: PreviewMode = PreviewMode.IMAGE
    axis: int = 2
    slice_index: int | None = None
    frame: int = 0
    patch_size: int = 6
    patch_origin: tuple[int, int] | None = None
    mask_alpha: float = 0.45
    dpi: int = 150
    show: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", PreviewMode(self.mode))
        for name in ("axis", "frame", "patch_size", "dpi"):
            if not isinstance(getattr(self, name), (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        if self.slice_index is not None and (
            not isinstance(self.slice_index, (int, np.integer)) or self.slice_index < 0
        ):
            raise ValueError("slice_index must be a nonnegative integer")
        if self.frame < 0 or not 1 <= self.patch_size <= 16 or self.dpi <= 0:
            raise ValueError("frame must be nonnegative, patch_size 1-16, and dpi positive")
        if not 0 <= self.mask_alpha <= 1:
            raise ValueError("mask_alpha must be between 0 and 1")
        if self.patch_origin is not None and (
            len(self.patch_origin) != 2
            or any(not isinstance(value, (int, np.integer)) or value < 0 for value in self.patch_origin)
        ):
            raise ValueError("patch_origin must contain two nonnegative integer indices")


@dataclass(frozen=True, eq=False)
class ScanGeometry:
    """Voxel-to-RAS affine and explicit units; unknown units stay unknown."""

    affine_ras: NDArray[np.float64] = field(repr=False)
    spatial_unit: str = "unknown"

    def __post_init__(self) -> None:
        affine = np.array(self.affine_ras, dtype=np.float64, copy=True)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError("affine_ras must be a finite 4x4 matrix")
        if not np.allclose(affine[3], [0, 0, 0, 1]) or np.linalg.matrix_rank(affine[:3, :3]) != 3:
            raise ValueError("affine_ras must be an invertible spatial affine")
        affine.setflags(write=False)
        object.__setattr__(self, "affine_ras", affine)

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Voxel spacing in spatial_unit (not necessarily millimetres)."""
        return tuple(float(value) for value in np.linalg.norm(self.affine_ras[:3, :3], axis=0))

    @property
    def axis_codes(self) -> tuple[str, str, str]:
        return nib.aff2axcodes(self.affine_ras)

    def matches(self, other: ScanGeometry) -> bool:
        return self.spatial_unit == other.spatial_unit and np.allclose(
            self.affine_ras, other.affine_ras, rtol=0, atol=1e-6
        )


@dataclass(eq=False)
class Scan:
    """A 3-D or 4-D scan with scaled float32 voxels and source metadata.

    data is mutable and never reoriented or normalized for display. NumPy
    inputs already in float32 are reused; frame and slice access return views.
    """

    data: NDArray[np.float32] = field(repr=False)
    geometry: ScanGeometry
    source_path: Path | None = None
    storage_dtype: str | None = None

    def __post_init__(self) -> None:
        original = np.asarray(self.data)
        if original.ndim not in (3, 4) or any(size == 0 for size in original.shape):
            raise ValueError(f"Expected a nonempty 3-D or 4-D tensor, got shape {original.shape}")
        if original.dtype.kind not in "biuf":
            raise ValueError("Scan data must contain real numeric voxels")
        self.storage_dtype = self.storage_dtype or str(original.dtype)
        self.data = np.asarray(original, dtype=np.float32)
        if self.source_path is not None:
            self.source_path = Path(self.source_path)

    @classmethod
    def from_nifti(cls, path: str | Path) -> Scan:
        """Load NIfTI-1/2, including gzip files incorrectly named .nii."""
        path = Path(path)

        def materialize(image: nib.spatialimages.SpatialImage) -> Scan:
            return cls(
                data=image.get_fdata(dtype=np.float32),
                geometry=ScanGeometry(image.affine, image.header.get_xyzt_units()[0]),
                source_path=path,
                storage_dtype=str(image.get_data_dtype()),
            )

        with path.open("rb") as stream:
            is_gzip = stream.read(2) == b"\x1f\x8b"
        if not is_gzip:
            return materialize(nib.load(str(path)))
        with gzip.open(path, "rb") as stream:
            header = stream.read(540)
            stream.seek(0)
            for image_type in (nib.Nifti1Image, nib.Nifti2Image):
                if image_type.header_class.may_contain_header(header):
                    file_map = image_type.make_file_map()
                    file_map["image"] = nib.FileHolder(fileobj=stream)
                    # Load voxels before closing the decompression stream.
                    return materialize(image_type.from_file_map(file_map))
        raise nib.filebasedimages.ImageFileError(f"Not a NIfTI image: {path}")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def name(self) -> str:
        return self.source_path.name if self.source_path is not None else "NumPy scan"

    def volume(self, frame: int = 0) -> NDArray[np.float32]:
        frames = self.shape[3] if self.data.ndim == 4 else 1
        if not isinstance(frame, (int, np.integer)) or not 0 <= frame < frames:
            raise IndexError(f"frame must be between 0 and {frames - 1}")
        return self.data[..., frame] if self.data.ndim == 4 else self.data

    def slice(self, axis: int = 2, index: int | None = None, *, frame: int = 0) -> NDArray[np.float32]:
        """Return an untransposed 2-D tensor view in native array-axis order."""
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        volume = self.volume(frame)
        index = volume.shape[axis] // 2 if index is None else index
        if not isinstance(index, (int, np.integer)) or not 0 <= index < volume.shape[axis]:
            raise IndexError(f"slice_index must be between 0 and {volume.shape[axis] - 1}")
        selection = [slice(None)] * 3
        selection[axis] = index
        return volume[tuple(selection)]

    def export_numpy(self, path: str | Path) -> Path:
        """Save the complete scaled tensor as .npy, without display transforms.

        The .npy contains only voxel values; spatial metadata stays on geometry.
        """
        path = Path(path)
        if path.suffix.lower() != ".npy":
            raise ValueError("NumPy export path must end with .npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.save(stream, self.data, allow_pickle=False)
        return path

    def export_preview(
        self, path: str | Path, options: PreviewOptions | None = None, *, mask: Scan | None = None
    ) -> Path:
        """Export a PNG using optional Matplotlib, imported only on demand."""
        from .preview import export_preview

        return export_preview(self, path, options or PreviewOptions(), mask=mask)

    def show_3d(self, options: VolumeViewOptions | None = None, *, mask: Scan | None = None) -> None:
        """Open a native VTK volume viewer; block until its window closes."""
        ScanCase(self, mask).show_3d(options)


@dataclass(frozen=True)
class ScanCase:
    """A scan and optional mask on matching voxel and physical grids."""

    image: Scan
    mask: Scan | None = None

    def __post_init__(self) -> None:
        if self.mask is not None:
            if self.image.shape[:3] != self.mask.shape[:3] or not self.image.geometry.matches(self.mask.geometry):
                raise ValueError("Image and mask must have matching spatial shapes, affines, and units")
            if self.mask.data.ndim == 4 and self.mask.shape != self.image.shape:
                raise ValueError("A 4-D mask must have the same shape and frame count as the image")

    @classmethod
    def from_nifti(cls, image_path: str | Path, mask_path: str | Path | None = None) -> ScanCase:
        return cls(Scan.from_nifti(image_path), Scan.from_nifti(mask_path) if mask_path is not None else None)

    @property
    def case_id(self) -> str:
        path = self.image.source_path
        if path is None:
            return "array"
        if path.parent.name.lower().startswith("subject"):
            return path.parent.name
        return path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem

    def export_preview(self, path: str | Path, options: PreviewOptions | None = None) -> Path:
        return self.image.export_preview(path, options, mask=self.mask)

    def show_3d(self, options: VolumeViewOptions | None = None) -> None:
        """Open the optional desktop viewer without importing web UI code."""
        from desktop.volume_viewer import VolumeViewer

        VolumeViewer(self, options).show()
