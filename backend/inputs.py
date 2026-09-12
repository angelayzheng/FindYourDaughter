"""NIfTI input loading in SimpleITK's physical LPS coordinate system."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import shutil
import tempfile
from typing import Iterator
import warnings

import nibabel as nib
import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class CaseData:
    """A validated CT/aorta-mask pair on a shared physical grid."""

    image_path: Path
    mask_path: Path
    image: sitk.Image
    aorta_mask: sitk.Image

    @property
    def case_id(self) -> str:
        return self.image_path.parent.name or self.image_path.stem


@contextmanager
def _readable_nifti(path: Path) -> Iterator[Path]:
    """Give ITK a correctly suffixed gzip file without modifying the source."""
    with path.open("rb") as source:
        compressed = source.read(2) == b"\x1f\x8b"
        if not compressed or path.name.lower().endswith(".nii.gz"):
            yield path
            return

        source.seek(0)
        # Close the temporary file before ITK opens it, including on Windows.
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                shutil.copyfileobj(source, temporary, length=1024 * 1024)
            except BaseException:
                temporary.close()
                temporary_path.unlink(missing_ok=True)
                raise
        try:
            yield temporary_path
        finally:
            temporary_path.unlink(missing_ok=True)


def _affine_lps_mm(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    """Use the coded sform, then coded qform, in physical LPS millimetres."""
    affine, code = image.get_sform(coded=True)
    if not code:
        affine, code = image.get_qform(coded=True)
    if not code:
        raise ValueError("Nonorthogonal NIfTI recovery requires a coded sform or qform")
    if not np.isfinite(affine).all() or np.linalg.matrix_rank(affine[:3, :3]) != 3:
        raise ValueError("NIfTI physical affine must be finite and invertible")
    # ITK's NIfTI reader also treats unspecified spatial units as millimetres.
    factor = {"meter": 1000.0, "mm": 1.0, "micron": 0.001, "unknown": 1.0}[
        image.header.get_xyzt_units()[0]
    ]
    return np.diag([-factor, -factor, factor, 1.0]) @ affine


def _resample_pair(image_path: Path, mask_path: Path) -> tuple[sitk.Image, sitk.Image]:
    """Recover a nonorthogonal source grid by resampling in physical space."""
    image = nib.load(image_path)
    mask = nib.load(mask_path)
    if len(image.shape) != 3 or len(mask.shape) != 3:
        raise ValueError("The CT image and aorta mask must both be 3-D NIfTI volumes.")
    affine = _affine_lps_mm(image)
    mask_affine = _affine_lps_mm(mask)
    if image.shape != mask.shape or not np.allclose(affine, mask_affine, rtol=0, atol=1e-6):
        raise ValueError("Image and aorta mask must have identical physical geometry before resampling.")

    basis = affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    # Polar decomposition gives the closest orthogonal orientation, retaining
    # reflections. Sampling through the *original* affine preserves anatomy.
    left, _, right = np.linalg.svd(basis / spacing)
    direction = left @ right
    corners = np.array(list(product(*[(-0.5, size - 0.5) for size in image.shape])))
    projected = corners @ basis.T @ direction
    lower, upper = projected.min(axis=0), projected.max(axis=0)
    size = np.ceil((upper - lower) / spacing).astype(int)
    origin = affine[:3, 3] + direction @ (lower + spacing / 2)

    def resample(source: nib.spatialimages.SpatialImage, source_affine: np.ndarray, *, label: bool) -> sitk.Image:
        # NiBabel applies NIfTI slope/intercept. ITK arrays use z,y,x order.
        values = np.asanyarray(source.dataobj) if label else source.get_fdata(dtype=np.float32)
        if values.dtype.kind not in "biuf":
            raise ValueError("The CT image and aorta mask must contain real scalar voxels.")
        values = np.asarray(values, dtype=values.dtype.newbyteorder("="))
        source_grid = sitk.GetImageFromArray(values.transpose(2, 1, 0), isVector=False)
        inverse = np.linalg.inv(source_affine[:3, :3])
        transform = sitk.AffineTransform(3)
        # Output LPS -> source index coordinates on source_grid's unit grid.
        transform.SetMatrix(inverse.ravel().tolist())
        transform.SetTranslation((-inverse @ source_affine[:3, 3]).tolist())
        resampler = sitk.ResampleImageFilter()
        resampler.SetNumberOfWorkUnits(4)
        resampler.SetSize(size.tolist())
        resampler.SetOutputOrigin(origin.tolist())
        resampler.SetOutputSpacing(spacing.tolist())
        resampler.SetOutputDirection(direction.ravel().tolist())
        resampler.SetTransform(transform)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if label else sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        result = resampler.Execute(source_grid)
        result.SetMetaData("branchseed_geometry", "resampled_nonorthogonal_nifti")
        result.SetMetaData("branchseed_source_spatial_unit", source.header.get_xyzt_units()[0])
        return result

    return resample(image, affine, label=False), resample(mask, mask_affine, label=True)


def _same_geometry(image: sitk.Image, mask: sitk.Image) -> bool:
    return (
        image.GetSize() == mask.GetSize()
        and np.allclose(image.GetOrigin(), mask.GetOrigin(), rtol=0, atol=1e-6)
        and np.allclose(image.GetSpacing(), mask.GetSpacing(), rtol=0, atol=1e-6)
        and np.allclose(image.GetDirection(), mask.GetDirection(), rtol=0, atol=1e-6)
    )


def load_case(image_path: Path, mask_path: Path) -> CaseData:
    """Load a 3-D pair, recovering gzip naming and nonorthogonal NIfTI grids.

    Ordinary images retain ITK's native loading behavior. Only its specific
    orientation failure triggers joint CT/mask resampling; other errors remain
    errors. Source geometry is checked before resampling can align the pair.
    """
    image_path, mask_path = Path(image_path), Path(mask_path)
    with _readable_nifti(image_path) as readable_image, _readable_nifti(mask_path) as readable_mask:
        try:
            image = sitk.ReadImage(str(readable_image))
            mask = sitk.ReadImage(str(readable_mask))
        except RuntimeError as error:
            if "ITK only supports orthonormal direction cosines" not in str(error):
                raise
            image, mask = _resample_pair(readable_image, readable_mask)
            warnings.warn(
                f"Resampled nonorthogonal NIfTI pair {image_path} and {mask_path} "
                "onto a shared orthogonal LPS grid (linear CT, nearest-neighbor mask).",
                RuntimeWarning,
                stacklevel=2,
            )
    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError("The CT image and aorta mask must both be 3-D NIfTI volumes.")
    if image.GetNumberOfComponentsPerPixel() != 1 or mask.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("The CT image and aorta mask must contain scalar voxels.")
    if not _same_geometry(image, mask):
        raise ValueError("The CT image and aorta mask must have identical physical geometry.")
    return CaseData(image_path, mask_path, image, mask)
