"""Strict, serializable detector settings; omitted values keep existing defaults."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path


def option_types(detector: str) -> dict:
    from backend.detection import DetectionOptions
    from backend.contact_detection import ContactOptions
    from backend.fusion_detection import FusionOptions
    from backend.refined_detection import RefinedOptions

    groups = {
        "baseline": {"baseline": DetectionOptions},
        "contact": {"contact": ContactOptions},
        "fusion": {"baseline": DetectionOptions, "contact": ContactOptions, "fusion": FusionOptions},
        "refined": {"baseline": DetectionOptions, "contact": ContactOptions,
                    "fusion": FusionOptions, "refined": RefinedOptions},
    }
    if not isinstance(detector, str) or detector not in groups:
        raise ValueError(f"Unknown detector {detector!r}")
    return groups[detector]


def resolve_parameters(detector: str, parameters: dict | None = None) -> dict:
    """Validate names/types/ranges and return all effective parameters."""
    types = option_types(detector)
    supplied = {} if parameters is None else parameters
    if not isinstance(supplied, dict) or set(supplied) - set(types):
        raise ValueError(f"Parameter groups for {detector} must be drawn from {list(types)}")
    result = {}
    for group, cls in types.items():
        values = supplied.get(group, {})
        defaults = asdict(cls())
        if not isinstance(values, dict) or set(values) - set(defaults):
            raise ValueError(f"Unknown parameters in {group}; supported: {list(defaults)}")
        for name, value in values.items():
            expected = type(defaults[name])
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or (expected is int and type(value) is not int)):
                raise ValueError(f"{group}.{name} must be a finite {'integer' if expected is int else 'number'}")
        result[group] = asdict(cls(**{name: int(value) if type(defaults[name]) is int else float(value)
                                     for name, value in values.items()}))
    return result


def configuration(detector: str, parameters: dict | None = None) -> dict:
    return {"detector": detector, "parameters": resolve_parameters(detector, parameters)}


def load_configuration(path: Path, *, detector: str | None = None) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"detector", "parameters"}:
        raise ValueError("Configuration requires exactly 'detector' and 'parameters'")
    if detector is not None and detector != value["detector"]:
        raise ValueError("--detector conflicts with the configuration's detector")
    return configuration(value["detector"], value["parameters"])
