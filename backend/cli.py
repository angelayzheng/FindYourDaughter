"""Command-line interface for the offline evaluator."""

from __future__ import annotations

import argparse
import json
import gzip
import re
from pathlib import Path
from typing import Sequence

from backend.pipeline import run_case
from backend.detectors import DETECTOR_NAMES
from backend.output import validate_prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect direct daughter arteries of an aorta.")
    parser.add_argument("--image", type=Path, help="Input CT NIfTI volume")
    parser.add_argument("--aorta-mask", type=Path, help="Binary parent-aorta NIfTI mask")
    parser.add_argument("--subject", type=Path, help="Subject folder; discover its CT and aorta mask automatically")
    parser.add_argument("--output", type=Path, required=True, help="Destination prediction JSON")
    parser.add_argument("--detector", choices=DETECTOR_NAMES,
                        help="Experimental algorithm to run (default: unchanged baseline)")
    parser.add_argument("--config", type=Path, help="Detector JSON configuration; omitted fields use original defaults")
    return parser


def _looks_like_nifti(path: Path) -> bool:
    """Accept ordinary NIfTI names and gzip-compressed files with odd names."""
    name = path.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        return True
    try:
        with path.open("rb") as stream:
            if stream.read(2) != b"\x1f\x8b":
                return False
        with gzip.open(path, "rb") as stream:
            header = stream.read(4)
        return header in (b"\x5c\x01\x00\x00", b"\x00\x00\x01\x5c", b"\x1c\x02\x00\x00", b"\x00\x00\x02\x1c")
    except (OSError, EOFError):
        return False


def _number(path: Path) -> str | None:
    match = re.search(r"(\d+)", path.stem)
    return match.group(1) if match else None


def discover_subject_inputs(subject: Path) -> tuple[Path, Path]:
    """Find one CT and one parent mask in a subject folder."""
    subject = subject.expanduser().resolve()
    if not subject.is_dir():
        raise ValueError(f"Subject folder does not exist: {subject}")
    paths = sorted(path for path in subject.rglob("*") if path.is_file() and _looks_like_nifti(path))
    masks = [path for path in paths if "mask" in path.name.lower()]
    images = [path for path in paths if "mask" not in path.name.lower()]
    if not masks or not images:
        raise ValueError(f"Could not find both a CT image and mask under {subject}")

    preferred_images = [path for path in images if path.name.lower().startswith(("orig", "image", "ct"))]
    if preferred_images:
        images = preferred_images
    if len(images) != 1:
        raise ValueError(f"Expected one CT image under {subject}, found: {', '.join(map(str, images))}")
    image = images[0]
    matching_masks = [path for path in masks if _number(path) == _number(image)]
    if matching_masks:
        masks = matching_masks
    if len(masks) != 1:
        raise ValueError(f"Expected one aorta mask under {subject}, found: {', '.join(map(str, masks))}")
    return image, masks[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.subject is not None and (args.image is not None or args.aorta_mask is not None):
        parser.error("--subject cannot be combined with --image or --aorta-mask")
    if args.subject is None and (args.image is None or args.aorta_mask is None):
        parser.error("provide either --subject or both --image and --aorta-mask")
    if args.subject is not None:
        try:
            args.image, args.aorta_mask = discover_subject_inputs(args.subject)
        except ValueError as error:
            parser.error(str(error))
    from backend.configuration import configuration, load_configuration
    try:
        config = load_configuration(args.config, detector=args.detector) if args.config else configuration(args.detector or "baseline")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    for label, path in (("image", args.image), ("aorta mask", args.aorta_mask)):
        if not path.is_file():
            raise SystemExit(f"Input {label} does not exist: {path}")

    prediction = validate_prediction(run_case(args.image, args.aorta_mask, detector=config["detector"], parameters=config["parameters"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prediction, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote prediction to {args.output}")
    return 0
