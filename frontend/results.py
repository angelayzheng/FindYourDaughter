"""Discovery and loading helpers for local benchmark CSV outputs."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path


def discover_csv_files(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def read_csv_file(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return [], []
        rows = list(reader)
        return list(reader.fieldnames), rows


def synthetic_comparison_rows(report: dict) -> list[dict[str, object]]:
    """Flatten scorer matches and unmatched truth/prediction entries for a table."""
    rows: list[dict[str, object]] = []
    for case in report.get("cases", []):
        truth = case.get("truth_daughters", [])
        predictions = case.get("predicted_daughters", [])
        matches = {
            item["truth_index"]: item
            for item in case.get("matches", [])
        }
        matched_predictions = {item["prediction_index"] for item in case.get("matches", [])}
        for index, daughter in enumerate(truth):
            match = matches.get(index)
            prediction = predictions[match["prediction_index"]] if match else None
            rows.append({
                "case_id": case["case_id"],
                "status": "matched" if match else "missed truth",
                "truth_id": daughter.get("instance_id", f"truth_{index + 1:03d}"),
                "prediction_id": prediction.get("instance_id", "") if prediction else "",
                "truth_ostium_xyz_mm": json.dumps(daughter.get("ostium_xyz_mm", [])),
                "prediction_ostium_xyz_mm": json.dumps(prediction.get("ostium_xyz_mm", [])) if prediction else "",
                "ostium_error_mm": match.get("ostium_error_mm", "") if match else "",
                "seed_error_mm": match.get("seed_error_mm", "") if match else "",
                "radius_error_mm": match.get("radius_error_mm", "") if match else "",
            })
        for index, prediction in enumerate(predictions):
            if index not in matched_predictions:
                rows.append({
                    "case_id": case["case_id"],
                    "status": "false positive",
                    "truth_id": "",
                    "prediction_id": prediction.get("instance_id", f"prediction_{index + 1:03d}"),
                    "truth_ostium_xyz_mm": "",
                    "prediction_ostium_xyz_mm": json.dumps(prediction.get("ostium_xyz_mm", [])),
                    "ostium_error_mm": "",
                    "seed_error_mm": "",
                    "radius_error_mm": "",
                })
    return rows


def comparison_csv(rows: list[dict[str, object]]) -> str:
    stream = io.StringIO(newline="")
    fields = list(rows[0]) if rows else ["case_id", "status"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()
