"""Backend pipeline boundary.

Branch detection is intentionally not implemented during infrastructure setup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.inputs import load_case


def _case_id(path: Path) -> str:
    name = path.name
    stem = name[:-7] if name.lower().endswith(".nii.gz") else path.stem
    if path.parent.name.lower().startswith("subject"):
        return path.parent.name
    return stem


def run_case(image_path: Path, mask_path: Path) -> dict[str, Any]:
    """Validate one case and return the challenge output envelope.

    The empty daughter list is a deliberate scaffold. Detection will replace it
    without changing the command-line or JSON interfaces.
    """
    load_case(image_path, mask_path)

    return {
        "case_id": _case_id(image_path),
        "parent": {"instance_id": "aorta"},
        "daughters": [],
    }
