"""Command-line parsing and validation for the prototype runner."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect direct daughter arteries from a CT volume and a parent-aorta mask."
    )
    parser.add_argument("--image", required=True, type=Path, help="Input CT NIfTI volume")
    parser.add_argument(
        "--aorta-mask", required=True, type=Path, help="Binary parent-aorta mask NIfTI volume"
    )
    parser.add_argument("--output", required=True, type=Path, help="Prediction JSON output path")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    for name in ("image", "aorta_mask"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"{name.replace('_', '-')} does not exist: {path}")
        setattr(args, name, path)
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args
