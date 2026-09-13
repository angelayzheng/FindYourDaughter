"""Export per-branch disagreement evidence from a saved draft evaluation run.

This is a development review tool, never a detector input or anatomical labeler.
Optional previews require the existing frontend matplotlib dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import SimpleITK as sitk

from evaluation.draft_set import SCOPE, _sha256, read_reference


def _nearest(point, records, point_key="ostium_xyz_mm"):
    if not records:
        return None, None
    distances = [float(np.linalg.norm(np.asarray(point) - r[point_key])) for r in records]
    i = int(np.argmin(distances))
    return records[i], distances[i]


def review_rows(case_id: str, detector: str, references: list, predictions: list,
                scores: dict, diagnostics: dict, tolerance_mm: float) -> list[dict]:
    """Keep scorer assignments separate from nearest-contact diagnostic clues."""
    matches = {r["reference_id"]: r for r in scores["matches"]}
    predictions_by_id = {p["instance_id"]: p for p in predictions}
    candidates = diagnostics.get("candidates", [])
    contacts = diagnostics.get("source_contact_records", diagnostics.get("contacts", []))
    rows = []
    for reference in references:
        match = matches.get(reference["instance_id"])
        contact, distance = _nearest(reference["ostium_xyz_mm"], contacts)
        row = {"case_id": case_id, "detector": detector,
               "status": "matched" if match else "unmatched_reference",
               "reference_id": reference["instance_id"],
               "prediction_id": match["prediction_id"] if match else None,
               "ostium_error_mm": match["ostium_error_mm"] if match else None,
               "seed_error_mm": match["seed_error_mm"] if match else None,
               "direction_error_deg": match["direction_error_deg"] if match else None,
               "radius_error_mm": match["radius_error_mm"] if match else None,
               "nearest_contact_status": contact["status"] if contact else None,
               "nearest_contact_distance_mm": distance,
               "contact_within_tolerance": distance <= tolerance_mm if distance is not None else None,
               "ostium_xyz_mm": reference["ostium_xyz_mm"],
               "reference_review_status": reference.get("review_status"),
               "reference_confidence": reference.get("confidence"),
               "contact_evidence": contact}
        if not match:
            nearest, pdistance = _nearest(reference["ostium_xyz_mm"], predictions)
            row["nearest_prediction_id"] = nearest["instance_id"] if nearest else None
            row["nearest_prediction_distance_mm"] = pdistance
        else:
            row["candidate_evidence"] = next((c for c in candidates if c["instance_id"] == match["prediction_id"]), None)
        rows.append(row)
    for unmatched in scores["unmatched_predictions"]:
        p = predictions_by_id[unmatched["prediction_id"]]
        rows.append({"case_id": case_id, "detector": detector, "status": "unmatched_prediction",
                     "reference_id": None, "prediction_id": p["instance_id"],
                     "nearest_reference_id": unmatched["nearest_reference_id"],
                     "nearest_reference_distance_mm": unmatched["nearest_ostium_distance_mm"],
                     "ostium_xyz_mm": p["ostium_xyz_mm"],
                     "candidate_evidence": next((c for c in candidates if c["instance_id"] == p["instance_id"]), None)})
    return rows


def _previews(reference, diagnostics: dict, rows: list[dict], output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = reference.case.image
    ct = sitk.GetArrayViewFromImage(image)
    parent = sitk.GetArrayViewFromImage(reference.case.aorta_mask)
    spacing = np.asarray(image.GetSpacing())[::-1]
    refs = {r["instance_id"]: r for r in reference.annotation["daughters"]}
    branches = {c["instance_id"]: c["branch"] for c in diagnostics.get("candidates", [])
                if c["status"] == "accepted"}
    mismatches = [r for r in rows if r["status"] != "matched"]
    paths = []
    for page in range((len(mismatches) + 3) // 4):
        batch = mismatches[page * 4:page * 4 + 4]
        fig, axes = plt.subplots(len(batch), 3, figsize=(12, 3.4 * len(batch)),
                                 squeeze=False, layout="constrained")
        for i, row in enumerate(batch):
            missed = row["status"] == "unmatched_reference"
            identifier = row["reference_id"] if missed else row["prediction_id"]
            branch = refs[identifier] if missed else branches.get(identifier)
            if branch is None:
                raise ValueError("Preview paths require fusion/refined candidate diagnostics")
            point = np.asarray(image.TransformPhysicalPointToContinuousIndex(row["ostium_xyz_mm"])[::-1])
            guide = np.asarray([image.TransformPhysicalPointToContinuousIndex(p)[::-1]
                                for p in branch["centerline_xyz_mm"]])
            low = np.maximum(0, np.floor(point - 12 / spacing).astype(int))
            high = np.minimum(ct.shape, np.ceil(point + 12 / spacing).astype(int) + 1)
            region = tuple(slice(a, b) for a, b in zip(low, high))
            for col, axis in enumerate((0, 1, 2)):
                v, h = [a for a in range(3) if a != axis]
                at = int(round(point[axis] - low[axis]))
                slab = [slice(None)] * 3
                slab[axis] = slice(max(0, at - 1), min(high[axis] - low[axis], at + 2))
                view = ct[region][tuple(slab)].max(axis=axis)
                mask = parent[region][tuple(slab)].max(axis=axis)
                extent = ((low[h] - .5) * spacing[h], (high[h] - .5) * spacing[h],
                          (high[v] - .5) * spacing[v], (low[v] - .5) * spacing[v])
                ax = axes[i, col]
                ax.imshow(view, cmap="gray", vmin=-100, vmax=500, extent=extent)
                if mask.any() and not mask.all():
                    ax.contour(np.arange(low[h], high[h]) * spacing[h],
                               np.arange(low[v], high[v]) * spacing[v], mask, levels=[.5], colors=["red"])
                ax.plot(guide[:, h] * spacing[h], guide[:, v] * spacing[v], color="magenta" if missed else "cyan")
                ax.scatter(point[h] * spacing[h], point[v] * spacing[v], c="yellow", s=18)
                ax.set_title(f"{identifier}: {row['status'].replace('_', ' ')}", fontsize=9)
                ax.set_aspect("equal")
        case_id = reference.annotation["case_id"]
        fig.suptitle(f"{case_id} | {rows[0]['detector']} draft disagreement review\n"
                     "Red parent; yellow origin; cyan prediction; magenta draft. Three-voxel slabs; paths projected.\n"
                     "Axes are native-grid distances (mm), not LPS coordinates. Not expert adjudication.", fontsize=10)
        path = output / f"{case_id}_{page + 1}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path.name)
    return paths


def build_review(report_path: Path, output: Path, *, detector="refined", dataset=None, previews=False) -> dict:
    report_path, output = report_path.resolve(), output.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dataset = Path(dataset or report["dataset"]).resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError("Review output must be outside the reference dataset")
    if output == report_path.parent / detector:
        raise ValueError("Review output must not replace prediction exports")
    if detector not in report["detectors"]:
        raise ValueError("Selected detector is absent from the saved report")
    output.mkdir(parents=True, exist_ok=True)
    rows, images, failures = [], [], []
    for case in report["cases"]:
        run = case["detectors"][detector]
        if run["status"] != "ok":
            failures.append({"case_id": case["case_id"], "error": run.get("error", "failed input")})
            continue
        # Tie review images to the evaluated inputs rather than a subsequently
        # edited annotation package. Do not rerun or alter any detector.
        for name, digest in case["input_sha256"].items():
            if _sha256(dataset / name) != digest:
                raise ValueError(f"Evaluated input has changed: {name}")
        reference = read_reference(dataset / case["case_id"])
        folder = report_path.parent / detector
        diagnostics = json.loads((folder / f"{case['case_id']}_diagnostics.json").read_text(encoding="utf-8"))
        predictions = json.loads((folder / f"{case['case_id']}_prediction.json").read_text(encoding="utf-8"))["daughters"]
        case_rows = review_rows(case["case_id"], detector, reference.annotation["daughters"],
                                predictions, run["scores"], diagnostics, report["matching_tolerance_mm"])
        rows.extend(case_rows)
        if previews:
            images.extend(_previews(reference, diagnostics, case_rows, output))
    result = {"scope": SCOPE, "source_report": str(report_path), "source_report_sha256": _sha256(report_path),
              "detector": detector, "matching_tolerance_mm": report["matching_tolerance_mm"],
              "note": "Nearest contact is a diagnostic clue, not anatomical identity or a scored match.",
              "rows": rows, "failed_cases": failures, "preview_files": images}
    (output / "review.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["case_id", "detector", "status", "reference_id", "prediction_id", "ostium_error_mm", "seed_error_mm",
              "direction_error_deg", "radius_error_mm", "nearest_contact_status", "nearest_contact_distance_mm",
              "contact_within_tolerance", "nearest_prediction_id", "nearest_prediction_distance_mm",
              "nearest_reference_id", "nearest_reference_distance_mm"]
    with (output / "review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--detector", default="refined")
    parser.add_argument("--previews", action="store_true", help="Render fusion/refined disagreement paths")
    args = parser.parse_args()
    try:
        result = build_review(args.report, args.output_dir, detector=args.detector,
                              dataset=args.dataset, previews=args.previews)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {len(result['rows'])} branch records and {len(result['preview_files'])} preview pages to {args.output_dir}")
    return 1 if result["failed_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
