"""Command-line interface for the offline evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from backend.pipeline import run_case
from backend.detectors import DETECTOR_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect direct daughter arteries of an aorta.")
    parser.add_argument("--image", type=Path, required=True, help="Input CT NIfTI volume")
    parser.add_argument("--aorta-mask", type=Path, required=True, help="Binary parent-aorta NIfTI mask")
    parser.add_argument("--output", type=Path, required=True, help="Destination prediction JSON")
    parser.add_argument("--detector", choices=DETECTOR_NAMES, default="baseline",
                        help="Algorithm to run (default: unchanged baseline; contact is experimental)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for label, path in (("image", args.image), ("aorta mask", args.aorta_mask)):
        if not path.is_file():
            raise SystemExit(f"Input {label} does not exist: {path}")

    prediction = run_case(args.image, args.aorta_mask, detector=args.detector)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote prediction to {args.output}")
    return 0

