"""Find local NIfTI scans for the optional inspection dashboard."""

from __future__ import annotations

from pathlib import Path


def nifti_files(folder: Path) -> list[Path]:
    """List files in one subject folder without reading large image volumes."""
    return sorted(
        (path for path in folder.iterdir()
         if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz"))),
        key=lambda path: path.name.lower(),
    )


def find_subjects(root: Path) -> dict[str, list[Path]]:
    """Index root-level scans and immediate subject folders by directory."""
    root = Path(root).expanduser()
    if not root.is_dir():
        return {}
    folders = [root] + sorted((path for path in root.iterdir() if path.is_dir()),
                              key=lambda path: path.name.lower())
    subjects = {}
    for folder in folders:
        images = [path for path in nifti_files(folder) if "mask" not in path.name.lower()]
        if images:
            subjects[folder.name] = images
    return subjects


def mask_choices(image: Path) -> list[Path]:
    """Offer neighboring masks; ScanCase validates the selected geometry."""
    return [path for path in nifti_files(image.parent) if "mask" in path.name.lower()]
