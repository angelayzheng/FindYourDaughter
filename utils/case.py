"""Input loading and physical-coordinate helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseData:
    """A validated CT/aorta-mask pair and their SimpleITK metadata."""

    image_path: Path
    mask_path: Path
    image: Any
    aorta_mask: Any

    @property
    def case_id(self) -> str:
        return self.image_path.parent.name if self.image_path.parent.name else self.image_path.stem


def _simpleitk() -> Any:
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise RuntimeError(
            "SimpleITK is required to run the prototype. "
            "Install dependencies with 'python3 -m pip install -r requirements.txt'."
        ) from error
    return sitk


def load_case(image_path: Path, mask_path: Path) -> CaseData:
    """Read and validate a CT volume and its parent-aorta mask."""
    sitk = _simpleitk()
    image = sitk.ReadImage(str(image_path))
    aorta_mask = sitk.ReadImage(str(mask_path))

    if image.GetDimension() != 3 or aorta_mask.GetDimension() != 3:
        raise ValueError("Both image and aorta mask must be 3-D volumes")
    if image.GetSize() != aorta_mask.GetSize():
        raise ValueError(
            f"Image and aorta mask grids differ: {image.GetSize()} vs {aorta_mask.GetSize()}"
        )
    if image.GetSpacing() != aorta_mask.GetSpacing():
        raise ValueError("Image and aorta mask spacing differs")
    if image.GetOrigin() != aorta_mask.GetOrigin():
        raise ValueError("Image and aorta mask origin differs")
    if image.GetDirection() != aorta_mask.GetDirection():
        raise ValueError("Image and aorta mask direction differs")

    return CaseData(image_path, mask_path, image, aorta_mask)


def index_to_physical_point(image: Any, index: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert an x/y/z voxel index using SimpleITK's physical coordinate system."""
    point = image.TransformIndexToPhysicalPoint(tuple(int(value) for value in index))
    return tuple(float(value) for value in point)
