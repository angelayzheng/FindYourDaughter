"""Backend pipeline boundary.

Branch detection is intentionally not implemented during infrastructure setup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import SimpleITK as sitk


def _case_id(path: Path) -> str:
    name = path.name
    stem = name[:-7] if name.lower().endswith(".nii.gz") else path.stem
    if path.parent.name.lower().startswith("subject"):
        return path.parent.name
    return stem


def _same_geometry(image: sitk.Image, mask: sitk.Image) -> bool:
    tolerance = 1e-6
    return (
        image.GetDimension() == mask.GetDimension()
        and image.GetSize() == mask.GetSize()
        and all(abs(left - right) <= tolerance for left, right in zip(image.GetOrigin(), mask.GetOrigin()))
        and all(abs(left - right) <= tolerance for left, right in zip(image.GetSpacing(), mask.GetSpacing()))
        and all(abs(left - right) <= tolerance for left, right in zip(image.GetDirection(), mask.GetDirection()))
    )


def run_case(image_path: Path, mask_path: Path) -> dict[str, Any]:
    """Validate one case and return the challenge output envelope.

    The empty daughter list is a deliberate scaffold. Detection will replace it
    without changing the command-line or JSON interfaces.
    """
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError("The CT image and aorta mask must both be 3-D NIfTI volumes.")
    if not _same_geometry(image, mask):
        raise ValueError("The CT image and aorta mask must have identical physical geometry.")

    return {
        "case_id": _case_id(image_path),
        "parent": {"instance_id": "aorta"},
        "daughters": [],
    }
