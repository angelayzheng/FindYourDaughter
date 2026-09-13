"""Opt-in refinement; baseline, contact and fusion remain independent controls."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import time

import numpy as np
from scipy import ndimage as ndi
import SimpleITK as sitk

from backend.candidates import Candidate, group_candidates
from backend.contact_detection import ContactOptions, detect_contacts
from backend.detection import DetectedBranch, DetectionOptions, DetectionResult, _point_at, _vesselness, detect_daughters
from backend.fusion_detection import FusionOptions, _EvidenceVolume


def _extended_caps(volume: _EvidenceVolume) -> list[tuple]:
    """Account for skeleton retreat inside a broad or oblique parent cut face."""
    caps = []
    for center, tangent, reach in volume.caps:
        radius = reach - 3 * volume.spacing.max()
        distances = np.arange(0, 4 * radius + volume.spacing.max(), volume.spacing.min() / 2)
        ray = center + distances[:, None] * tangent
        values = volume.sample(volume.parent.astype(np.uint8), ray, order=0, outside=0)
        exits = np.flatnonzero(values == 0)
        retreat = float(distances[exits[0]]) if len(exits) else radius
        caps.append((center, tangent, max(radius, retreat) + 3 * volume.spacing.max()))
    return caps


def _repair_parent_chord(path: np.ndarray, volume: _EvidenceVolume) -> tuple[np.ndarray | None, str]:
    """Reanchor only an initially parent-crossing chord at its observed exit.

    A patch point on a concave wall can be unrelated to the skeleton root it
    was joined to. Limit correction to one physical voxel diagonal and retain
    the existing external path. Long jumps, later parent returns, and paths
    with less than five millimetres left are not rescued.
    """
    chord = float(np.linalg.norm(path[1] - path[0]))
    if chord > 2 * np.linalg.norm(volume.spacing):
        return None, "long_initial_chord"
    fractions = np.linspace(0, 1, max(3, int(np.ceil(chord / .25)) + 1))
    samples = path[0] + fractions[:, None] * (path[1] - path[0])
    parent_float = volume.parent.astype(np.float32)
    parent = volume.sample(parent_float, samples)
    inside = np.flatnonzero(parent >= .5)
    # Ignore the origin's expected half-mask contact itself.
    if not np.any(parent[1:-1] >= .5) or not len(inside) or inside[-1] == len(parent) - 1:
        return None, "no_initial_parent_crossing"
    left, right = samples[inside[-1]], samples[inside[-1] + 1]
    for _ in range(16):
        middle = (left + right) / 2
        if volume.sample(parent_float, middle[None])[0] >= .5:
            left = middle
        else:
            right = middle
    origin = (left + right) / 2
    if np.linalg.norm(origin - path[0]) > np.linalg.norm(volume.spacing):
        return None, "large_origin_shift"
    repaired = np.vstack([origin, path[1:]])
    length = float(np.linalg.norm(np.diff(repaired, axis=0), axis=1).sum())
    if length < 5:
        return None, "short_repaired_path"
    samples = np.asarray([_point_at(repaired, d) for d in np.r_[np.arange(.5, length, .25), length]])
    ct = volume.sample(volume.ct, samples)
    parent = volume.sample(volume.parent.astype(np.uint8), samples, order=0, outside=1)
    if np.any(parent) or not np.isfinite(ct).all() or np.any(ct < volume.lower) or np.any(ct > volume.upper):
        return None, "unsupported_repaired_path"
    return repaired, "repaired_parent_chord"


def _selection_key(candidate: Candidate) -> tuple:
    features = candidate.features
    # A closed section obtained by cutting away the parent is weaker evidence
    # of separation than a naturally closed section. Resolve that geometric
    # ambiguity before comparing brightness/tubularity scores.
    independent_sections = all(s["closed"] and s["method"] == "orthogonal_section"
                               for s in features["sections"])
    return (features["radius_quality"] == "stable_sections", independent_sections,
            features["review_score"], candidate.source == "contact", candidate.candidate_id)


def detect_refined(image: sitk.Image, aorta_mask: sitk.Image) -> DetectionResult:
    """Refine source evidence with no reference labels or learned parameters."""
    started = time.perf_counter()
    options = FusionOptions()
    contact = detect_contacts(image, aorta_mask)
    baseline = detect_daughters(image, aorta_mask)
    candidates = [Candidate(f"{name}_{i:03d}", name, deepcopy(branch))
                  for name, source in (("baseline", baseline), ("contact", contact))
                  for i, branch in enumerate(source.branches, 1)]
    diagnostics = {
        "method": "experimental_refined_v1", "options": asdict(options),
        "source_options": {"baseline": asdict(DetectionOptions()), "contact": asdict(ContactOptions())},
        "source_counts": {"baseline": len(baseline.branches), "contact": len(contact.branches)},
        "source_rejections": {"baseline": baseline.diagnostics.get("rejected", {}),
                              "contact": contact.diagnostics.get("rejected", {})},
        "source_contact_records": deepcopy(contact.diagnostics.get("contacts", [])),
        "deferred_common_trunks": [dict(r, status="needs_seed_policy", source="contact")
                                   for r in contact.diagnostics.get("contacts", [])
                                   if r["status"] == "early_bifurcation"],
        "repairs": [], "rejected": {},
    }
    result = DetectionResult(diagnostics=diagnostics)
    has_paths = any(r.get("centerline_xyz_mm") for r in diagnostics["source_contact_records"])
    if candidates or has_paths:
        volume = _EvidenceVolume(image, aorta_mask, contact.diagnostics)
        volume.caps = _extended_caps(volume)
        vesselness = None
        for number, record in enumerate(diagnostics["source_contact_records"], 1):
            if record["status"] != "unsupported_path":
                continue
            path, reason = _repair_parent_chord(np.asarray(record["centerline_xyz_mm"]), volume)
            audit = {"contact_record": number, "original_ostium_xyz_mm": record["ostium_xyz_mm"],
                     "status": reason}
            diagnostics["repairs"].append(audit)
            if path is None:
                continue
            if vesselness is None:
                selection = tuple(slice(int(o), int(o + size)) for o, size in zip(volume.offset, volume.ct.shape))
                ct = np.asarray(sitk.GetArrayViewFromImage(image)[selection], dtype=np.float32)
                vesselness = _vesselness(ct, volume.spacing, contact.diagnostics["blood_intensity"],
                                        contact.diagnostics["background_intensity"])
            samples = np.asarray([_point_at(path, d) for d in np.arange(2, 5.01, .5)])
            score = float(volume.sample(vesselness, samples).mean())
            if not np.isfinite(score) or score < ContactOptions().min_vesselness:
                audit["status"] = "weak_repaired_tubularity"
                continue
            seed = _point_at(path, 5)
            direction = seed - path[0]
            length = float(np.linalg.norm(direction))
            if length < 1e-8:
                audit["status"] = "degenerate_repaired_direction"
                continue
            direction /= length
            # Radius is measured below; this placeholder cannot be exported
            # because recovered candidates require a valid closed seed section.
            branch = DetectedBranch(path[0].tolist(), seed.tolist(), 1., direction.tolist(),
                                    path.tolist(), score, "pending_section")
            c = Candidate(f"contact_repair_{number:03d}", "contact", branch,
                          features={"origin_method": "repaired_parent_chord", "contact_record": number})
            candidates.append(c)
            audit["candidate_id"] = c.candidate_id
            audit["repaired_ostium_xyz_mm"] = path[0].tolist()
        for c in candidates:
            volume.measure(c, options)
            sections = c.features.get("sections", [])
            if sections and not any(s["closed"] for s in sections):
                c.reasons.append("no_closed_lumen_sections")
            if c.features.get("origin_method") == "repaired_parent_chord":
                seed_section = next((s for s in sections if s["arc_length_mm"] == 5), None)
                if seed_section is None or not seed_section["closed"]:
                    c.reasons.append("unmeasured_repaired_seed")
            if c.reasons:
                c.status = "rejected"
        groups = group_candidates(candidates, ostium_mm=options.duplicate_ostium_mm,
                                  path_mm=options.duplicate_path_mm)
        selected = []
        for group in groups:
            eligible = [c for c in group if c.status == "eligible"]
            if not eligible:
                continue
            chosen = max(eligible, key=_selection_key)
            selected.append(chosen)
            chosen.status = "accepted"
            chosen.selected_candidate_id = chosen.candidate_id
            for c in group:
                if c is chosen:
                    continue
                c.selected_candidate_id = chosen.candidate_id
                if c.status == "eligible":
                    c.status = "duplicate"
                    c.reasons.append("same_proximal_lumen")
        selected.sort(key=lambda c: tuple(c.branch.ostium_xyz_mm[::-1]))
        for i, c in enumerate(selected, 1):
            c.instance_id = f"branch_{i:03d}"
        result.branches = [c.branch for c in selected]
        diagnostics["candidate_groups"] = len(groups)
    diagnostics["candidates"] = [c.diagnostic() for c in candidates]
    for c in candidates:
        for reason in c.reasons:
            diagnostics["rejected"][reason] = diagnostics["rejected"].get(reason, 0) + 1
    diagnostics["elapsed_seconds"] = time.perf_counter() - started
    return result
