"""Validation for the evaluator-facing prediction JSON envelope."""

from __future__ import annotations

import math
import re
from typing import Any


_DAUGHTER_FIELDS = {
    "instance_id",
    "parent_instance_id",
    "ostium_xyz_mm",
    "seed_xyz_mm",
    "radius_mm",
    "direction_xyz",
}


def _finite_vector(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a list of three numbers")
    if not all(isinstance(component, (int, float)) and math.isfinite(component) for component in value):
        raise ValueError(f"{name} must contain only finite numbers")


def validate_prediction(prediction: Any) -> dict[str, Any]:
    """Validate and return a prediction before it is serialized as JSON."""
    if not isinstance(prediction, dict) or set(prediction) != {"case_id", "parent", "daughters"}:
        raise ValueError("Prediction must contain exactly case_id, parent, and daughters")
    if not isinstance(prediction["case_id"], str) or not prediction["case_id"]:
        raise ValueError("case_id must be a non-empty string")
    if prediction["parent"] != {"instance_id": "aorta"}:
        raise ValueError('parent must be {"instance_id": "aorta"}')
    daughters = prediction["daughters"]
    if not isinstance(daughters, list):
        raise ValueError("daughters must be a list")

    instance_ids: set[str] = set()
    for daughter in daughters:
        if not isinstance(daughter, dict) or set(daughter) != _DAUGHTER_FIELDS:
            raise ValueError(f"Each daughter must contain exactly {sorted(_DAUGHTER_FIELDS)}")
        instance_id = daughter["instance_id"]
        if not isinstance(instance_id, str) or re.fullmatch(r"branch_\d{3}", instance_id) is None:
            raise ValueError("Each daughter instance_id must match branch_NNN")
        if instance_id in instance_ids:
            raise ValueError(f"Duplicate daughter instance_id: {instance_id}")
        instance_ids.add(instance_id)
        if daughter["parent_instance_id"] != "aorta":
            raise ValueError("Each daughter parent_instance_id must be aorta")
        _finite_vector(daughter["ostium_xyz_mm"], "ostium_xyz_mm")
        _finite_vector(daughter["seed_xyz_mm"], "seed_xyz_mm")
        _finite_vector(daughter["direction_xyz"], "direction_xyz")
        radius = daughter["radius_mm"]
        if not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius_mm must be a positive finite number")
        direction_length = math.sqrt(sum(component * component for component in daughter["direction_xyz"]))
        if not math.isclose(direction_length, 1.0, rel_tol=0, abs_tol=1e-3):
            raise ValueError("direction_xyz must be a unit vector")
    return prediction
