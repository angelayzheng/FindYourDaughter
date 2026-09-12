"""Prediction schema and JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DaughterPrediction:
    instance_id: str
    parent_instance_id: str
    ostium_xyz_mm: list[float]
    seed_xyz_mm: list[float]
    radius_mm: float
    direction_xyz: list[float]


@dataclass
class Prediction:
    case_id: str
    daughters: list[DaughterPrediction] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "parent": {"instance_id": "aorta"},
            "daughters": [asdict(daughter) for daughter in self.daughters],
        }


def write_prediction(prediction: Prediction, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(prediction.as_dict(), indent=2) + "\n")
