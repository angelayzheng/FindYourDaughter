"""Development-only matching against generated tube truth, not the challenge scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import linear_sum_assignment

from backend.pipeline import run_case
from backend.detectors import DETECTOR_NAMES
from backend.configuration import configuration, load_configuration


def evaluate(dataset: Path, tolerance_mm: float = 3.0, *, detector: str = "baseline", parameters: dict | None = None) -> dict:
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("Matching tolerance must be finite and positive")
    config = configuration(detector, parameters)
    truth_files = sorted(dataset.glob("subject*/truth*.json"))
    if not truth_files:
        raise ValueError(f"No generated truth files found under {dataset}")
    rows, errors = [], []
    true_positive = false_positive = false_negative = 0
    for truth_path in truth_files:
        truth_document = json.loads(truth_path.read_text(encoding="utf-8"))
        if "generator" not in truth_document:
            raise ValueError(f"Expected synthetic generator metadata in {truth_path}")
        truth = truth_document["daughters"]
        suffix = truth_path.stem.removeprefix("truth")
        prediction = run_case(truth_path.with_name(f"orig{suffix}.nii"),
                              truth_path.with_name(f"mask{suffix}.nii"), detector=detector,
                              parameters=config["parameters"])["daughters"]
        matches = []
        if truth and prediction:
            distances = np.linalg.norm(
                np.array([branch["ostium_xyz_mm"] for branch in prediction])[:, None]
                - np.array([branch["ostium_xyz_mm"] for branch in truth])[None, :], axis=-1)
            # Prefer the largest valid matching before minimizing total distance.
            penalty = (min(len(truth), len(prediction)) + 1) * tolerance_mm
            left, right = linear_sum_assignment(np.where(distances <= tolerance_mm, distances, penalty))
            matches = [(i, j) for i, j in zip(left, right) if distances[i, j] <= tolerance_mm]
            for i, j in matches:
                errors.append({
                    "ostium_mm": float(distances[i, j]),
                    "seed_mm": float(np.linalg.norm(np.array(prediction[i]["seed_xyz_mm"])
                                                     - truth[j]["seed_xyz_mm"])),
                    "radius_mm": abs(prediction[i]["radius_mm"] - truth[j]["radius_mm"]),
                })
        true_positive += len(matches)
        false_positive += len(prediction) - len(matches)
        false_negative += len(truth) - len(matches)
        rows.append({"case_id": truth_document["case_id"], "truth": len(truth),
                     "predictions": len(prediction), "matched": len(matches)})
    return {"scope": "Generated tubes only; development tolerance, not official challenge scoring", "detector": detector,
            "configuration": config,
            "matching_tolerance_mm": tolerance_mm, "true_positive": true_positive,
            "false_positive": false_positive, "false_negative": false_negative,
            "mean_errors": {key: float(np.mean([error[key] for error in errors]))
                            for key in errors[0]} if errors else {}, "cases": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tolerance-mm", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=Path("nifti_previews/synthetic_metrics.json"))
    parser.add_argument("--detector", choices=DETECTOR_NAMES)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    try:
        config = load_configuration(args.config, detector=args.detector) if args.config else configuration(args.detector or "baseline")
        result = evaluate(args.dataset, args.tolerance_mm, detector=config["detector"], parameters=config["parameters"])
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
