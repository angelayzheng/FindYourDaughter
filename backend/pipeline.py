"""Evaluator boundary for the experimental direct-daughter detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.inputs import load_case
from backend.detectors import detect


def _case_id(path: Path) -> str:
    name = path.name
    stem = name[:-7] if name.lower().endswith(".nii.gz") else path.stem
    if path.parent.name.lower().startswith("subject"):
        return path.parent.name
    return stem


def run_case(image_path: Path, mask_path: Path, *, detector: str = "refined",
             parameters: dict | None = None) -> dict[str, Any]:
    """Load one case and return experimental detections in the required schema."""
    case = load_case(image_path, mask_path)
    result = detect(case.image, case.aorta_mask, detector=detector,
                    **({"parameters": parameters} if parameters is not None else {}))

    return {
        "case_id": _case_id(image_path),
        "parent": {"instance_id": "aorta"},
        "daughters": result.daughters(),
    }
