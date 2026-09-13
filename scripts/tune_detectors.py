"""Search a declared parameter grid and validate frozen selections offline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.tuning import run_sweep, validate_selection


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sweep = commands.add_parser("sweep", help="Evaluate all configurations and check the shortlist")
    sweep.add_argument("--plan", type=Path, default=Path("configs/tuning_grid.json"))
    sweep.add_argument("--dataset", type=Path, default=Path("eval_set"))
    sweep.add_argument("--synthetic-development", type=Path, required=True)
    sweep.add_argument("--output-dir", type=Path, help="Must not exist; default is a timestamped directory")
    validate = commands.add_parser("validate", help="Score frozen defaults/winners without retuning")
    validate.add_argument("--run-dir", type=Path, required=True)
    validate.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "sweep":
            output = args.output_dir or Path("nifti_previews/tuning") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            plan = json.loads(args.plan.read_text(encoding="utf-8"))
            result = run_sweep(plan, args.dataset, output, synthetic_development=args.synthetic_development)
            print(f"Saved experiment to {output}")
        else:
            result = validate_selection(args.run_dir, args.dataset)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
