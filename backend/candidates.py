"""Reviewable internal evidence and conservative cross-detector correspondence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from backend.detection import DetectedBranch, _point_at


@dataclass
class Candidate:
    candidate_id: str
    source: str
    branch: DetectedBranch
    features: dict = field(default_factory=dict)
    status: str = "candidate"
    reasons: list[str] = field(default_factory=list)
    group_id: str | None = None
    instance_id: str | None = None
    selected_candidate_id: str | None = None

    def diagnostic(self) -> dict:
        def safe(value):
            if isinstance(value, dict):
                return {key: safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple, np.ndarray)):
                return [safe(item) for item in value]
            if isinstance(value, np.generic):
                value = value.item()
            return None if isinstance(value, float) and not np.isfinite(value) else value
        return safe(asdict(self))


def group_candidates(candidates: list[Candidate], *, ostium_mm: float, path_mm: float) -> list[list[Candidate]]:
    """Pair baseline/contact observations without merging roots from one source.

    Require close origins, aligned outward directions, and close proximal paths.
    The path allowance is the larger of a grid-scale floor and the smaller
    measured lumen radius: two medial traces within a broad tube can differ.
    A bipartite assignment prevents transitive chains from swallowing adjacent
    independent ostia. No point or path averaging is performed.
    """
    if not np.isfinite([ostium_mm, path_mm]).all() or min(ostium_mm, path_mm) <= 0:
        raise ValueError("Duplicate distances must be finite and positive")
    ordered = sorted(candidates, key=lambda c: c.candidate_id)
    if any(c.source not in ("baseline", "contact") for c in ordered):
        raise ValueError("Candidate grouping expects baseline/contact observations")
    left = [c for c in ordered if c.source == "baseline"]
    right = [c for c in ordered if c.source == "contact"]
    costs = np.full((len(left), len(right)), np.inf)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            root = float(np.linalg.norm(np.asarray(a.branch.ostium_xyz_mm) - b.branch.ostium_xyz_mm))
            if root > ostium_mm or np.dot(a.branch.direction_xyz, b.branch.direction_xyz) < .8:
                continue
            a_path, b_path = np.asarray(a.branch.centerline_xyz_mm), np.asarray(b.branch.centerline_xyz_mm)
            distances = [np.linalg.norm(_point_at(a_path, d) - _point_at(b_path, d)) for d in (1., 3., 5.)]
            path_limit = max(path_mm, min(a.branch.radius_mm, b.branch.radius_mm))
            if max(distances) <= path_limit:
                costs[i, j] = root + float(np.mean(distances))
    pairs = []
    if left and right:
        maximum = float(np.max(costs[np.isfinite(costs)])) if np.isfinite(costs).any() else 0.
        penalty = (min(len(left), len(right)) + 1) * (maximum + 1)
        a_indices, b_indices = linear_sum_assignment(np.where(np.isfinite(costs), costs, penalty))
        pairs = [(int(i), int(j)) for i, j in zip(a_indices, b_indices) if np.isfinite(costs[i, j])]
    used = {c.candidate_id for i, j in pairs for c in (left[i], right[j])}
    groups = [[left[i], right[j]] for i, j in pairs] + [[c] for c in ordered if c.candidate_id not in used]
    groups.sort(key=lambda group: min(c.candidate_id for c in group))
    for number, group in enumerate(groups, 1):
        for c in group:
            c.group_id = f"group_{number:03d}"
    return groups
