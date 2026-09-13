"""Optional evidence-based fusion of the unchanged baseline and contact detectors."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy import ndimage as ndi
import SimpleITK as sitk

from backend.candidates import Candidate, group_candidates
from backend.contact_detection import ContactOptions, _contact_radius, _fit_axis, _parent_caps, detect_contacts
from backend.detection import DetectionOptions, DetectionResult, _point_at, detect_daughters


@dataclass(frozen=True)
class FusionOptions:
    """Fixed development settings; scores are evidence summaries, not probabilities."""

    duplicate_ostium_mm: float = 2.5
    duplicate_path_mm: float = 1.5
    minimum_support_fraction: float = .8
    max_radius_cv: float = .35
    max_expansion_ratio: float = 2.

    def __post_init__(self):
        if not np.isfinite(tuple(vars(self).values())).all():
            raise ValueError("Fusion settings must be finite")
        if min(self.duplicate_ostium_mm, self.duplicate_path_mm) <= 0:
            raise ValueError("Duplicate distances must be positive millimetres")
        if not 0 < self.minimum_support_fraction <= 1 or not 0 < self.max_radius_cv < 1:
            raise ValueError("Invalid support fraction or radius variation limit")
        if self.max_expansion_ratio <= 1:
            raise ValueError("Expansion ratio must be greater than one")


class _EvidenceVolume:
    """Small parent ROI for shared physical measurements, without resegmentation."""

    def __init__(self, image: sitk.Image, mask: sitk.Image, diagnostics: dict):
        self.image = image
        self.spacing = np.asarray(image.GetSpacing())[::-1]
        parent = sitk.GetArrayViewFromImage(mask)
        bounds = []
        for axis in range(3):
            active = np.flatnonzero(np.any(parent, axis=tuple(i for i in range(3) if i != axis)))
            padding = int(np.ceil(15 / self.spacing[axis]))
            bounds.append((max(0, int(active[0]) - padding), min(parent.shape[axis], int(active[-1]) + padding + 1)))
        selection = tuple(slice(lo, hi) for lo, hi in bounds)
        self.offset = np.asarray([lo for lo, _ in bounds])
        self.parent = parent[selection] > 0
        ct = np.asarray(sitk.GetArrayViewFromImage(image)[selection], dtype=np.float32)
        self.ct = ndi.gaussian_filter(ct, np.maximum(.5 / self.spacing, .35))
        self.inside = ndi.distance_transform_edt(self.parent, sampling=self.spacing)
        self.outside = ndi.distance_transform_edt(~self.parent, sampling=self.spacing)
        self.signed = self.outside - self.inside
        self.normals = np.gradient(ndi.gaussian_filter(self.signed, .5), *self.spacing)
        self.caps = _parent_caps(self.parent, self.inside, self.spacing, image, self.offset)
        self.lower = diagnostics["threshold"]
        self.upper = diagnostics.get("upper_intensity", diagnostics["blood_intensity"] + 200)

    def indices(self, points: np.ndarray) -> np.ndarray:
        return np.asarray([self.image.TransformPhysicalPointToContinuousIndex(p.tolist())[::-1]
                           for p in points]) - self.offset

    def sample(self, array: np.ndarray, points: np.ndarray, *, order=1, outside=np.nan) -> np.ndarray:
        return ndi.map_coordinates(array, self.indices(points).T, order=order, mode="constant", cval=outside)

    def measure(self, candidate: Candidate, options: FusionOptions) -> None:
        branch = candidate.branch
        path = np.asarray(branch.centerline_xyz_mm, dtype=float)
        origin = np.asarray(branch.ostium_xyz_mm)
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        features = candidate.features
        features.update(observed_length_mm=length, source_radius_mm=branch.radius_mm,
                        source_radius_method=branch.radius_method, mean_vesselness=branch.vesselness)
        if length < 5 or length > 10 + 1e-5 or not np.isfinite(path).all():
            candidate.reasons.append("invalid_proximal_path")
            candidate.status = "rejected"
            return
        seed = _point_at(path, 5)
        vector = seed - origin
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            candidate.reasons.append("degenerate_direction")
            candidate.status = "rejected"
            return
        direction = vector / norm
        samples = np.asarray([_point_at(path, d) for d in np.linspace(2, 5, 7)])
        ct_values = self.sample(self.ct, samples)
        parent_values = self.sample(self.parent.astype(np.uint8), samples, order=0, outside=1)
        supported = np.isfinite(ct_values) & (ct_values >= self.lower) & (ct_values <= self.upper) & (parent_values == 0)
        support = float(supported.mean())
        wall_distances = self.sample(self.outside, samples)
        origin_distance = float(self.sample(self.signed, origin[None])[0])
        normal = np.array([self.sample(field, origin[None])[0] for field in self.normals])[::-1]
        normal = np.asarray(self.image.GetDirection()).reshape(3, 3) @ normal
        normal /= max(float(np.linalg.norm(normal)), 1e-8)
        end_distance = min((float(np.linalg.norm(origin - center)) for center, _, _ in self.caps), default=None)
        cap = any(np.linalg.norm(origin - center) <= reach and (origin - center) @ tangent > 0
                  and normal @ tangent > .7 for center, tangent, reach in self.caps)
        features.update(support_fraction=support, ct_samples=ct_values.tolist(),
                        parent_wall_distances_mm=wall_distances.tolist(), parent_surface_distance_mm=origin_distance,
                        parent_end_distance_mm=end_distance, parent_normal_direction_cosine=float(normal @ direction),
                        parent_cap=bool(cap))
        if cap:
            candidate.reasons.append("parent_end_cap")
        if not np.isfinite(origin_distance) or abs(origin_distance) > self.spacing.max():
            candidate.reasons.append("unsupported_origin")
        if support < options.minimum_support_fraction:
            candidate.reasons.append("insufficient_path_support")
        # Measure at the seed and nearby *observed* positions, never beyond a fork.
        sections = []
        for distance in (4., 5., 6.):
            if distance > length + 1e-6:
                continue
            point = _point_at(path, distance)
            local = np.asarray([_point_at(path, d) for d in np.linspace(max(0, distance - 1), min(length, distance + 1), 7)])
            tangent = _fit_axis(local, point, local[-1] - local[0])
            radius, method = _contact_radius(self.image, self.ct, self.parent, self.offset,
                                              point, tangent, self.lower, self.upper, 6.)
            axis = np.eye(3)[np.argmin(np.abs(tangent))]
            horizontal = np.cross(tangent, axis)
            horizontal /= np.linalg.norm(horizontal)
            vertical = np.cross(tangent, horizontal)
            angles = np.arange(16) * 2 * np.pi / 16
            ring = point + max(4., 2 * branch.radius_mm) * (np.cos(angles)[:, None] * horizontal
                                                           + np.sin(angles)[:, None] * vertical)
            background = self.sample(self.ct, ring)
            background = background[np.isfinite(background)]
            peak = float(self.sample(self.ct, point[None])[0])
            contrast = peak - max(0., float(np.median(background))) if len(background) and np.isfinite(peak) else None
            sections.append({"arc_length_mm": distance, "radius_mm": radius, "method": method,
                             "closed": radius is not None, "area_mm2": float(np.pi * radius**2) if radius is not None else None,
                             "sampled_ring_contrast": contrast})
        radii = [s["radius_mm"] for s in sections if s["closed"]]
        cv = float(np.std(radii) / np.mean(radii)) if len(radii) >= 2 else None
        seed_section = next(s for s in sections if s["arc_length_mm"] == 5)
        stable = len(radii) >= 2 and cv <= options.max_radius_cv and seed_section["closed"]
        quality = "stable_sections" if stable else "unstable_sections" if seed_section["closed"] else "source_fallback"
        features.update(sections=sections, radius_cv=cv, radius_quality=quality)
        proximal = next((s["radius_mm"] for s in sections if s["arc_length_mm"] == 4), None)
        distal = next((s["radius_mm"] for s in sections if s["arc_length_mm"] == 6), None)
        expansion = distal / proximal if proximal is not None and distal is not None else None
        features["radius_expansion_ratio"] = expansion
        # A narrow bridge opening abruptly into a broad pool can look tubular
        # near its root. Require both a large distal section and rapid expansion;
        # ordinary proximal taper or a small uncertain section alone is not vetoed.
        if expansion is not None and distal > 6 and expansion > options.max_expansion_ratio:
            candidate.reasons.append("expands_into_broad_pool")
        if seed_section["closed"]:
            if not .8 <= seed_section["radius_mm"] <= 6.:
                candidate.reasons.append("radius_out_of_range")
            else:
                branch.radius_mm = seed_section["radius_mm"]
                branch.radius_method = "fusion_" + seed_section["method"]
        # Score ranks observations within an identified duplicate group only.
        # Borderline section measurements remain visible; this is not a learned
        # confidence probability or an independent artery classification score.
        features["review_score"] = float(.5 * np.clip(branch.vesselness, 0, 1) + .25 * support
                                         + .25 * (len(radii) / len(sections)))
        branch.seed_xyz_mm = seed.tolist()
        branch.direction_xyz = direction.tolist()
        candidate.status = "rejected" if candidate.reasons else "eligible"


def detect_fusion(image: sitk.Image, aorta_mask: sitk.Image, options: FusionOptions | None = None, *,
                  baseline_options: DetectionOptions | None = None,
                  contact_options: ContactOptions | None = None) -> DetectionResult:
    """Fuse observed proposals; reference annotations are never inputs."""
    started = time.perf_counter()
    options = options or FusionOptions()
    # Contact validates scalar geometry, binary masks and finite ROI intensities.
    contact = detect_contacts(image, aorta_mask, contact_options) if contact_options is not None else detect_contacts(image, aorta_mask)
    baseline = detect_daughters(image, aorta_mask, baseline_options) if baseline_options is not None else detect_daughters(image, aorta_mask)
    sources = {"baseline": baseline, "contact": contact}
    candidates = [Candidate(f"{name}_{i:03d}", name, deepcopy(branch))
                  for name, result in sources.items() for i, branch in enumerate(result.branches, 1)]
    result = DetectionResult(diagnostics={
        "method": "experimental_fusion_v1", "options": asdict(options), "rejected": {},
        "source_options": {"baseline": asdict(baseline_options or DetectionOptions()),
                           "contact": asdict(contact_options or ContactOptions())},
        "source_counts": {name: len(source.branches) for name, source in sources.items()},
        "source_rejections": {name: source.diagnostics.get("rejected", {}) for name, source in sources.items()},
        "blood_intensity": contact.diagnostics.get("blood_intensity"),
        "deferred_common_trunks": [dict(record, status="needs_seed_policy", source="contact")
                                   for record in contact.diagnostics.get("contacts", []) if record["status"] == "early_bifurcation"],
        "source_contact_records": contact.diagnostics.get("contacts", []),
    })
    if candidates:
        volume = _EvidenceVolume(image, aorta_mask, contact.diagnostics)
        for candidate in candidates:
            volume.measure(candidate, options)
        groups = group_candidates(candidates, ostium_mm=options.duplicate_ostium_mm, path_mm=options.duplicate_path_mm)
        selected = []
        for group in groups:
            eligible = [c for c in group if c.status == "eligible"]
            if not eligible:
                continue
            chosen = max(eligible, key=lambda c: (c.features["radius_quality"] == "stable_sections",
                                                  c.features["review_score"], c.source == "contact", c.candidate_id))
            chosen.status = "accepted"
            chosen.selected_candidate_id = chosen.candidate_id
            selected.append(chosen)
            for candidate in group:
                if candidate is chosen:
                    continue
                candidate.selected_candidate_id = chosen.candidate_id
                if candidate.status == "eligible":
                    candidate.status = "duplicate"
                    candidate.reasons.append("same_proximal_lumen")
        selected.sort(key=lambda c: tuple(c.branch.ostium_xyz_mm[::-1]))
        for i, candidate in enumerate(selected, 1):
            candidate.instance_id = f"branch_{i:03d}"
        result.branches = [c.branch for c in selected]
        result.diagnostics["candidate_groups"] = len(groups)
    for candidate in candidates:
        for reason in candidate.reasons:
            counts = result.diagnostics["rejected"]
            counts[reason] = counts.get(reason, 0) + 1
    result.diagnostics["candidates"] = [c.diagnostic() for c in candidates]
    result.diagnostics["elapsed_seconds"] = time.perf_counter() - started
    return result
