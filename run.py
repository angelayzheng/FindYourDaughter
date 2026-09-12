#!/usr/bin/env python3
"""Run the branch-seed prototype on one CT/aorta-mask pair."""

from __future__ import annotations

import sys

from utils.case import load_case
from utils.cli import parse_args
from utils.prediction import Prediction, write_prediction


def detect_daughters(case: object) -> list:
    """Return daughter predictions.

    This is intentionally a valid baseline until the branch detector is added.
    """
    del case
    return []


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        case = load_case(args.image, args.aorta_mask)
        prediction = Prediction(case_id=case.case_id, daughters=detect_daughters(case))
        write_prediction(prediction, args.output)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {len(prediction.daughters)} daughter prediction(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
