"""Evaluate baseline and contact against eval_set's draft daughter references."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.detectors import DETECTOR_NAMES
from evaluation.draft_set import SCOPE, evaluate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("eval_set"))
    parser.add_argument("--output-dir", type=Path, default=Path("nifti_previews/evaluation"))
    parser.add_argument("--detector", choices=("all", *DETECTOR_NAMES), default="all")
    parser.add_argument("--tolerance-mm", type=float, default=3., help="Development ostium matching tolerance (default: 3 mm)")
    parser.add_argument("--cases", nargs="+", type=int, help="Optional case numbers, e.g. 19 23")
    args = parser.parse_args(argv)
    selected = DETECTOR_NAMES if args.detector == "all" else (args.detector,)
    print(SCOPE, flush=True)
    try:
        report = evaluate_dataset(args.dataset, args.output_dir, detectors=selected,
                                  tolerance_mm=args.tolerance_mm, cases=args.cases)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    for name, result in report["summary"].items():
        print(f"{name}: {result['matched']}/{result['reference_count']} draft references matched; "
              f"{result['unmatched_predictions']} unmatched predictions; {result['failed_cases']} failed cases")
    print(f"Wrote {args.output_dir / 'report.json'} and {args.output_dir / 'summary.md'}")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
