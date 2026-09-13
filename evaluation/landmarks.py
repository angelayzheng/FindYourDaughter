"""One-to-one physical landmark matching against potentially incomplete references."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
import SimpleITK as sitk


def validate_branches(branches: list[dict], *, reference: bool = False) -> None:
    """Reject malformed landmarks rather than silently corrupting a score."""
    if not isinstance(branches, list):
        raise ValueError("Daughters must be a list")
    identifiers = set()
    for branch in branches:
        if not isinstance(branch, dict):
            raise ValueError("Each daughter must be an object")
        identifier = branch.get("instance_id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Branch instance IDs must be nonempty and unique within each set")
        identifiers.add(identifier)
        if branch.get("parent_instance_id") != "aorta":
            raise ValueError(f"{identifier}: expected parent_instance_id aorta")
        for key in ("ostium_xyz_mm", "seed_xyz_mm", "direction_xyz"):
            vector = np.asarray(branch.get(key), dtype=float)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError(f"{identifier}: {key} must have three finite physical components")
        if not np.isclose(np.linalg.norm(branch["direction_xyz"]), 1, atol=1e-3, rtol=0):
            raise ValueError(f"{identifier}: direction_xyz must be a unit vector")
        radius = branch.get("radius_mm")
        if reference and radius is None:
            continue
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not np.isfinite(radius) or radius <= 0:
            raise ValueError(f"{identifier}: radius_mm must be positive (or null in a reference)")


def usable_radius(branch: dict) -> bool:
    """A missing or envelope-limited measurement is not a zero radius."""
    return (branch.get("radius_mm") is not None
            and not branch.get("seed_cross_section_touches_search_limit", False)
            and branch.get("radius_measurement_status") in (None, "approximate_threshold_estimate"))


def _statistics(values: list[float]) -> dict:
    return {"count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "max": float(np.max(values)) if values else None}


def summarize(matches: list[dict], predictions: int, references: int) -> dict:
    """Micro-average counts and summarize matched landmarks only."""
    matched = len(matches)
    return {
        "reference_count": references, "prediction_count": predictions, "matched": matched,
        "unmatched_predictions": predictions - matched, "unmatched_references": references - matched,
        "reference_precision": matched / predictions if predictions else None,
        "reference_recall": matched / references if references else None,
        "reference_f1": 2 * matched / (predictions + references) if predictions + references else None,
        "errors": {key: _statistics([m[key] for m in matches if m.get(key) is not None])
                   for key in ("ostium_error_mm", "seed_error_mm", "direction_error_deg", "radius_error_mm",
                               "seed_distance_to_label_voxel_mm")},
        "seed_in_matched_label": {
            "count": sum(m.get("seed_in_matched_label") is not None for m in matches),
            "inside": sum(m.get("seed_in_matched_label") is True for m in matches),
        },
    }


def score_landmarks(predictions: list[dict], references: list[dict], *, tolerance_mm: float = 3.,
                    labels: sitk.Image | None = None) -> dict:
    """Maximize matches within a physical ostium tolerance, then minimize distance.

    Branch IDs are local identifiers, never a correspondence rule. Unmatched
    predictions are not established false positives when the reference is draft.
    Optional label measurements use the label image's own physical grid.
    """
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("Matching tolerance must be finite and positive")
    validate_branches(predictions)
    validate_branches(references, reference=True)
    predictions = sorted(predictions, key=lambda b: b["instance_id"])
    references = sorted(references, key=lambda b: b["instance_id"])
    pairs = []
    distances = np.empty((len(predictions), len(references)))
    if predictions and references:
        distances = np.linalg.norm(
            np.asarray([p["ostium_xyz_mm"] for p in predictions])[:, None]
            - np.asarray([r["ostium_xyz_mm"] for r in references])[None, :], axis=-1)
        penalty = (min(len(predictions), len(references)) + 1) * tolerance_mm
        left, right = linear_sum_assignment(np.where(distances <= tolerance_mm, distances, penalty))
        pairs = [(int(i), int(j)) for i, j in zip(left, right) if distances[i, j] <= tolerance_mm]
    label_array = sitk.GetArrayViewFromImage(labels) if labels is not None else None
    matches = []
    for i, j in pairs:
        predicted, reference = predictions[i], references[j]
        predicted_direction = np.asarray(predicted["direction_xyz"])
        reference_direction = np.asarray(reference["direction_xyz"])
        cosine = float(predicted_direction @ reference_direction
                       / (np.linalg.norm(predicted_direction) * np.linalg.norm(reference_direction)))
        match = {
            "prediction_id": predicted["instance_id"], "reference_id": reference["instance_id"],
            "reference_review_status": reference.get("review_status", "unspecified"),
            "reference_confidence": reference.get("confidence"),
            "ostium_error_mm": float(distances[i, j]),
            "seed_error_mm": float(np.linalg.norm(np.asarray(predicted["seed_xyz_mm"]) - reference["seed_xyz_mm"])),
            "direction_error_deg": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
            "radius_error_mm": abs(predicted["radius_mm"] - reference["radius_mm"]) if usable_radius(reference) else None,
            "reference_radius_status": reference.get("radius_measurement_status", "unspecified"),
        }
        if labels is not None:
            label = reference["label_value"]
            seed = [float(value) for value in predicted["seed_xyz_mm"]]
            index = labels.TransformPhysicalPointToIndex(seed)
            inside = all(0 <= index[k] < labels.GetSize()[k] for k in range(3))
            match["seed_nearest_label_value"] = int(labels[index]) if inside else None
            match["seed_in_matched_label"] = bool(inside and labels[index] == label)
            voxels = np.argwhere(label_array == label)
            if not len(voxels):
                raise ValueError(f"Reference label {label} is absent from the daughter mask")
            points = np.array([labels.TransformIndexToPhysicalPoint(tuple(int(v) for v in p[::-1])) for p in voxels])
            match["seed_distance_to_label_voxel_mm"] = float(np.linalg.norm(points - seed, axis=1).min())
        matches.append(match)
    matched_predictions, matched_references = {i for i, _ in pairs}, {j for _, j in pairs}
    unmatched = []
    for i, prediction in enumerate(predictions):
        if i in matched_predictions:
            continue
        nearest = int(np.argmin(distances[i])) if references else None
        unmatched.append({"prediction_id": prediction["instance_id"],
                          "nearest_reference_id": references[nearest]["instance_id"] if nearest is not None else None,
                          "nearest_ostium_distance_mm": float(distances[i, nearest]) if nearest is not None else None})
    return {"summary": summarize(matches, len(predictions), len(references)), "matches": matches,
            "unmatched_predictions": unmatched,
            "unmatched_reference_ids": [r["instance_id"] for j, r in enumerate(references) if j not in matched_references]}
