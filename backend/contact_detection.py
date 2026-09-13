"""Experimental contact detector: early tube evidence and supported parent exits.

An adaptation of Danilov et al. (2016), Riffaud et al. (2022), and Tahoces
et al. (2020), not a reproduction of their anatomical labeling algorithms.
The original detector remains in detection.py. Only its result types and
low-level image/graph measurements are shared here.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import SimpleITK as sitk

from backend.detection import (
    DetectedBranch, DetectionResult, _components, _graph, _point_at,
    _vesselness,
)
from backend.inputs import _same_geometry


@dataclass(frozen=True)
class ContactOptions:
    """Physical settings for the optional contact detector; not eligibility rules."""

    margin_mm: float = 15.0
    growth_mm: float = 12.0
    intensity_fraction: float = 0.35
    support_band_mm: float = 2.0
    tube_seed: float = 0.15
    tube_reach_mm: float = 2.0
    min_vesselness: float = 0.10
    min_radius_mm: float = 0.8
    max_radius_mm: float = 6.0

    def __post_init__(self) -> None:
        if not np.isfinite(tuple(vars(self).values())).all():
            raise ValueError("Contact settings must be finite")
        if not 10 <= self.growth_mm <= self.margin_mm:
            raise ValueError("Require margin_mm >= growth_mm >= 10")
        if not 0 < self.support_band_mm < self.growth_mm:
            raise ValueError("Support band must be inside the growth region")
        if not 0 < self.intensity_fraction < 1:
            raise ValueError("Intensity fraction must be between zero and one")
        if not 0 < self.min_vesselness <= self.tube_seed <= 1:
            raise ValueError("Require 0 < min_vesselness <= tube_seed <= 1")
        if not 0 < self.tube_reach_mm < self.growth_mm:
            raise ValueError("Tube reach must be inside the growth region")
        if not 0 < self.min_radius_mm <= self.max_radius_mm:
            raise ValueError("Invalid contact radius limits")


def _supported_wall(vessels: np.ndarray, distance: np.ndarray,
                    spacing: np.ndarray, band_mm: float) -> np.ndarray:
    """Retain near-wall voxels only if they link to retained outer layers.

    The outer region is explicitly preserved as support. Physical layers may
    skip bins across anisotropic neighbors. This is applied only near the wall,
    not as a requirement that an entire daughter travel radially outward.
    """
    retained = vessels & (distance > band_mm)
    step = float(spacing.min())
    for upper in np.arange(band_mm, 0, -step):
        layer = vessels & (distance <= upper) & (distance > max(0, upper - step))
        retained |= layer & ndi.binary_dilation(retained, structure=np.ones((3, 3, 3)))
    return retained


def _fit_axis(points: np.ndarray, anchor: np.ndarray,
              forward: np.ndarray) -> np.ndarray:
    """Fit an anchored physical axis and orient it in the observed forward direction."""
    delta = points - anchor
    values, vectors = np.linalg.eigh(delta.T @ delta)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        raise ValueError("Cannot orient a direction without observed progress")
    # Isotropic/degenerate point clouds do not define a trustworthy principal axis.
    axis = vectors[:, -1] if values[-1] > 1e-8 and values[-1] > 1.2 * values[-2] else forward / norm
    if abs(float(axis @ forward)) < 1e-8:
        axis = forward / norm
    return axis if axis @ forward >= 0 else -axis


def _physical(image: sitk.Image, indices: np.ndarray, offset: np.ndarray) -> np.ndarray:
    if np.issubdtype(indices.dtype, np.integer):
        return np.asarray([image.TransformIndexToPhysicalPoint(tuple(int(v) for v in (point + offset)[::-1]))
                           for point in indices])
    return np.asarray([image.TransformContinuousIndexToPhysicalPoint(
        (point + offset)[::-1].tolist()) for point in indices])


def _contact_radius(image: sitk.Image, ct: np.ndarray, parent: np.ndarray,
                    offset: np.ndarray, seed: np.ndarray, tangent: np.ndarray,
                    lower: float, upper: float, max_radius: float) -> tuple[float | None, str]:
    """Measure a closed CT lumen section, excluding air from the background model.

    The shape-gated candidate mask can erode thick vessels; measure the original
    smoothed CT in a sufficiently wide physical plane instead of its EDT radius.
    A parent-connected section gets one retry with the supplied parent excluded.
    """
    axis = np.eye(3)[np.argmin(np.abs(tangent))]
    horizontal = np.cross(tangent, axis)
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(tangent, horizontal)
    center = np.asarray(image.TransformPhysicalPointToContinuousIndex(seed.tolist()))[::-1] - offset
    basis = np.asarray([image.TransformPhysicalPointToContinuousIndex((seed + direction).tolist())[::-1]
                        for direction in (horizontal, vertical)]) - offset - center
    step = .4
    half_width = int(np.ceil(max(4., 2 * max_radius) / step))
    positions = np.arange(-half_width, half_width + 1) * step
    u, v = np.meshgrid(positions, positions)
    indices = center + u[..., None] * basis[0] + v[..., None] * basis[1]
    values = ndi.map_coordinates(ct, indices.reshape(-1, 3).T, order=1,
                                 mode="constant", cval=np.nan).reshape(u.shape)
    finite = np.isfinite(values)
    ring = (np.hypot(u, v) >= half_width * step * .8) & finite
    method = "orthogonal_section"
    if not np.any(ring):
        return None, method
    # HU air/fat should not pull the lumen threshold below soft tissue. This is
    # a provisional CT assumption, not a probability or a general MRI rule.
    background = max(0., float(np.median(values[ring])))
    peak = float(values[half_width, half_width])
    if not np.isfinite(peak) or peak - background < 40:
        return None, method
    section = finite & (values >= max(lower, (peak + background) / 2)) & (values <= upper)
    for exclude_parent in (False, True):
        if exclude_parent:
            method = "parent_excluded_section"
            parent_section = ndi.map_coordinates(parent, indices.reshape(-1, 3).T, order=0,
                                                 mode="constant", cval=0).reshape(u.shape)
            section &= ~parent_section
        labels, _ = ndi.label(section)
        label = labels[half_width, half_width]
        if not label:
            return None, method
        component = labels == label
        if (component[0].any() or component[-1].any() or component[:, 0].any() or component[:, -1].any()
                or np.any(ndi.binary_dilation(component) & ~finite)):
            continue
        return float(np.sqrt(component.sum() * step**2 / np.pi)), method
    return None, method


def _parent_caps(parent: np.ndarray, inside: np.ndarray, spacing: np.ndarray,
                 image: sitk.Image, offset: np.ndarray) -> list[tuple]:
    """Local endpoint tangents of the supplied mask; no parent resegmentation."""
    coords, adjacency = _graph(skeletonize(np.pad(parent, 1), method="lee")[1:-1, 1:-1, 1:-1])
    points = _physical(image, coords, offset)
    caps = []
    for endpoint, neighbors in enumerate(adjacency):
        if len(neighbors) != 1:
            continue
        radius = float(inside[tuple(coords[endpoint])])
        chain, current, previous, length = [endpoint], endpoint, -1, 0.0
        while length < max(6.0, 1.5 * radius):
            choices = adjacency[current] - {previous}
            if len(choices) != 1:
                break
            following = min(choices)
            if following in chain:
                break
            length += float(np.linalg.norm(points[following] - points[current]))
            chain.append(following)
            previous, current = current, following
        if len(chain) >= 3:
            direction = _fit_axis(points[chain], points[endpoint], points[endpoint] - points[chain[-1]])
            caps.append((points[endpoint], direction, radius + 3 * spacing.max()))
    return caps


def _trace_contact(root: int, origin: np.ndarray, points: np.ndarray,
                   adjacency: list[set[int]], other_roots: set[int]) -> tuple[np.ndarray, str]:
    """Trace at most 10 mm, stopping at a true fork or an ambiguous connection."""
    current, previous = root, -1
    visited: set[int] = set()
    path = [origin, points[root]]
    length = float(np.linalg.norm(points[root] - origin))

    def sustained(start: int, parent: int) -> bool:
        pending = [(start, parent, 0.0)]
        seen = set(visited) | {parent}
        while pending:
            node, last, distance = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            distance += float(np.linalg.norm(points[node] - points[last]))
            if distance >= 2.0 or node in other_roots:
                return True
            pending.extend((n, node, distance) for n in sorted(adjacency[node] - seen))
        return False

    while length < 10.0:
        visited.add(current)
        neighbors = adjacency[current] - {previous}
        if neighbors & other_roots:
            return np.asarray(path), "other_contact"
        if neighbors & visited:
            return np.asarray(path), "cycle"
        choices = sorted(neighbors - visited)
        if len(choices) > 1:
            choices = [node for node in choices if sustained(node, current)]
        if not choices:
            return np.asarray(path), "end"
        if len(choices) > 1:
            return np.asarray(path), "bifurcation"
        following = choices[0]
        length += float(np.linalg.norm(points[following] - points[current]))
        path.append(points[following])
        previous, current = current, following
    array = np.asarray(path)
    return np.vstack([array[:-1], _point_at(array, 10.0)]), "length_limit"


def detect_contacts(image: sitk.Image, aorta_mask: sitk.Image,
                    options: ContactOptions | None = None) -> DetectionResult:
    """Propose unnamed direct daughters on matching 3-D SimpleITK LPS grids.

    Arrays use ZYX; every exported point/direction uses physical LPS millimetres.
    Diagnostics are JSON-safe and separate from evaluator daughter objects.
    """
    started = time.perf_counter()
    options = options or ContactOptions()
    if (image.GetDimension() != 3 or aorta_mask.GetDimension() != 3
            or image.GetNumberOfComponentsPerPixel() != 1
            or aorta_mask.GetNumberOfComponentsPerPixel() != 1
            or not _same_geometry(image, aorta_mask)):
        raise ValueError("Contact detection requires matching scalar 3-D image and mask geometry")
    mask = sitk.GetArrayViewFromImage(aorta_mask)
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("The aorta mask must be binary (0 or 1)")
    result = DetectionResult(diagnostics={"method": "experimental_contact_v1", "rejected": {}, "contacts": []})

    def finish() -> DetectionResult:
        result.branches.sort(key=lambda branch: tuple(branch.ostium_xyz_mm[::-1]))
        identifiers = {tuple(branch.ostium_xyz_mm): f"branch_{i:03d}"
                       for i, branch in enumerate(result.branches, 1)}
        for record in result.diagnostics["contacts"]:
            if record["status"] == "accepted":
                record["instance_id"] = identifiers[tuple(record["ostium_xyz_mm"])]
        result.diagnostics["elapsed_seconds"] = time.perf_counter() - started
        return result

    def reject(reason: str, record: dict | None = None) -> None:
        counts = result.diagnostics["rejected"]
        counts[reason] = counts.get(reason, 0) + 1
        if record is not None:
            record["status"] = reason

    if not np.any(mask):
        return finish()
    spacing = np.asarray(image.GetSpacing())[::-1]
    bounds = []
    for axis in range(3):
        active = np.flatnonzero(np.any(mask, axis=tuple(i for i in range(3) if i != axis)))
        pad = int(np.ceil(options.margin_mm / spacing[axis]))
        bounds.append((max(0, int(active[0]) - pad), min(mask.shape[axis], int(active[-1]) + pad + 1)))
    selection = tuple(slice(lo, hi) for lo, hi in bounds)
    offset = np.asarray([lo for lo, _ in bounds])
    parent = mask[selection] > 0
    ct = np.asarray(sitk.GetArrayViewFromImage(image)[selection], dtype=np.float32)
    if min(ct.shape) < 3 or np.count_nonzero(parent) < 3:
        reject("insufficient_3d_support")
        return finish()
    if not np.isfinite(ct).all():
        raise ValueError("CT values around the aorta must be finite")
    inside = ndi.distance_transform_edt(parent, sampling=spacing)
    outside = ndi.distance_transform_edt(~parent, sampling=spacing)
    core = parent & (inside >= 2.0)
    blood = float(np.median(ct[core if np.any(core) else parent]))
    shell = (outside >= 4) & (outside <= 10)
    background = float(np.median(ct[shell])) if np.any(shell) else float(np.min(ct))
    lower = max(40.0, background + options.intensity_fraction * (blood - background))
    result.diagnostics.update(blood_intensity=blood, background_intensity=background,
                              threshold=lower, roi_shape_zyx=list(ct.shape))
    if blood - background < 40 or blood <= lower:
        reject("insufficient_contrast")
        return finish()
    smooth = ndi.gaussian_filter(ct, np.maximum(0.5 / spacing, 0.35))
    core_values = smooth[core if np.any(core) else parent]
    noise = 1.4826 * float(np.median(np.abs(core_values - np.median(core_values))))
    upper = blood + max(200.0, 4 * noise)
    vesselness = _vesselness(ct, spacing, blood, background)
    weak = (~parent & (outside <= options.growth_mm) & (smooth >= lower) & (smooth <= upper))
    markers = weak & (outside > options.support_band_mm) & (vesselness >= options.tube_seed)
    if not np.any(markers):
        reject("no_tube_support")
        return finish()
    # Bound weak reconstruction around observed tube evidence. Without this
    # envelope, one thin bridge can flood an entire contrast-compatible organ.
    near_tube = ndi.distance_transform_edt(~markers, sampling=spacing) <= options.tube_reach_mm
    weak &= near_tube | (outside <= options.support_band_mm)
    vessels = ndi.binary_propagation(markers, structure=np.ones((3, 3, 3)), mask=weak)
    before_cleanup = int(np.count_nonzero(vessels))
    vessels = _supported_wall(vessels, outside, spacing, options.support_band_mm)
    labels, _ = ndi.label(parent | vessels, structure=np.ones((3, 3, 3)))
    touching = np.unique(labels[parent])
    vessels &= np.isin(labels, touching[touching != 0])
    lumen = parent | vessels
    contacts, count = ndi.label(vessels & ndi.binary_dilation(parent, structure=np.ones((3, 3, 3))),
                                structure=np.ones((3, 3, 3)))
    result.diagnostics.update(upper_intensity=upper, tube_seed_voxels=int(markers.sum()),
                              removed_voxels=before_cleanup - int(vessels.sum()), contact_candidates=int(count))
    if not count:
        return finish()
    radius_field = ndi.distance_transform_edt(lumen, sampling=spacing)
    coords, adjacency = _graph(skeletonize(np.pad(lumen, 1), method="lee")[1:-1, 1:-1, 1:-1])
    raw_points = _physical(image, coords, offset)
    in_parent = parent[tuple(coords.T)]
    external_nodes = set(np.flatnonzero(~in_parent).tolist())
    raw_roots = {n for n in external_nodes if any(in_parent[k] for k in adjacency[n])}
    external = [neighbors & external_nodes for neighbors in adjacency]
    junctions = {n for n in external_nodes if len(external[n]) >= 3}
    groups = _components(junctions, external) + [[n] for n in sorted(external_nodes - junctions)]
    mapping = {node: group_id for group_id, group in enumerate(groups) for node in group}
    points = np.asarray([raw_points[g][np.argmin(np.linalg.norm(raw_points[g] - raw_points[g].mean(axis=0), axis=1))]
                         for g in groups])
    reduced = [set() for _ in groups]
    for node in sorted(external_nodes):
        reduced[mapping[node]].update(mapping[n] for n in external[node] if mapping[n] != mapping[node])
    roots = {mapping[n] for n in raw_roots}
    components = _components(set(range(len(groups))), reduced)
    contact_counts = {}
    for component in components:
        component_contacts = {int(contacts[tuple(coords[k])]) for m in component for k in groups[m]
                              if k in raw_roots}
        contact_counts.update((n, len(component_contacts)) for n in component)
    caps = _parent_caps(parent, inside, spacing, image, offset)
    normal_fields = np.gradient(ndi.gaussian_filter(outside - inside, 0.5), *spacing)
    direction_matrix = np.asarray(image.GetDirection()).reshape(3, 3)
    parent_float = parent.astype(np.float32)
    for label, region in enumerate(ndi.find_objects(contacts), 1):
        if region is None:
            continue
        patch = np.argwhere(contacts[region] == label) + np.asarray([s.start for s in region])
        # Distance is computed in the combined lumen, so the cut at P's surface
        # is not treated as a vessel side wall. Break plateaus by patch centrality.
        radii = radius_field[tuple(patch.T)]
        central = patch[np.isclose(radii, radii.max(), atol=1e-6)]
        voxel = central[np.argmin(np.linalg.norm((central - patch.mean(axis=0)) * spacing, axis=1))]
        around = np.asarray([(voxel + d) for d in np.ndindex(3, 3, 3)]) - 1
        valid = np.all((around >= 0) & (around < np.asarray(parent.shape)), axis=1)
        around = around[valid]
        neighbors = around[parent[tuple(around.T)]]
        inward = neighbors[np.argmin(np.linalg.norm((neighbors - voxel) * spacing, axis=1))].astype(float)
        # Locate the half-mask interface along a supported inside/outside edge.
        left, right = inward, voxel.astype(float)
        for _ in range(12):
            middle = (left + right) / 2
            if ndi.map_coordinates(parent_float, middle[:, None], order=1, mode="nearest")[0] >= .5:
                left = middle
            else:
                right = middle
        origin_index = (left + right) / 2
        origin = _physical(image, origin_index[None], offset)[0]
        record = {"ostium_xyz_mm": origin.tolist(), "contact_voxels": len(patch), "status": "pending"}
        result.diagnostics["contacts"].append(record)
        normal = np.array([ndi.map_coordinates(f, origin_index[:, None], order=1, mode="nearest")[0]
                           for f in normal_fields])
        normal = direction_matrix @ normal[::-1]
        normal /= max(float(np.linalg.norm(normal)), 1e-8)
        if any(np.linalg.norm(origin - center) <= reach and (origin - center) @ tangent > 0
               and normal @ tangent > .7 for center, tangent, reach in caps):
            reject("parent_end_cap", record)
            continue
        starts = {mapping[n] for n in raw_roots if contacts[tuple(coords[n])] == label}
        if not starts:
            reject("no_centerline", record)
            continue
        root = min(starts, key=lambda n: (np.linalg.norm(points[n] - origin), n))
        record["connected_contacts"] = contact_counts[root]
        path, stop = _trace_contact(root, origin, points, reduced, roots - starts)
        record["centerline_xyz_mm"] = path.tolist()
        record["stop_reason"] = stop
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        record["observed_length_mm"] = length
        if stop in ("cycle", "other_contact"):
            reject("ambiguous_connection", record)
            continue
        if length < 5:
            reject("early_bifurcation" if stop == "bifurcation" else "short_path", record)
            continue
        samples = np.asarray([_point_at(path, d) for d in np.arange(1.0, length, 0.5)])
        indices = np.array([image.TransformPhysicalPointToContinuousIndex(p.tolist())[::-1] for p in samples]) - offset
        observed = ndi.map_coordinates(smooth, indices.T, order=1, mode="nearest")
        parent_samples = ndi.map_coordinates(parent_float, indices.T, order=0, mode="nearest")
        if np.any(parent_samples) or np.any(observed < lower) or np.any(observed > upper):
            reject("unsupported_path", record)
            continue
        score = float(np.mean(ndi.map_coordinates(vesselness, indices.T[:, 3:], order=1, mode="nearest")))
        record["mean_vesselness"] = score
        if score < options.min_vesselness:
            reject("weak_tubularity", record)
            continue
        seed = _point_at(path, 5.0)
        local = np.asarray([_point_at(path, d) for d in np.linspace(3, min(length, 7), 9)])
        tangent = _fit_axis(local, seed, local[-1] - local[0])
        radius, radius_method = _contact_radius(image, smooth, parent, offset, seed, tangent,
                                                lower, upper, options.max_radius_mm)
        record["radius_method"] = radius_method
        if radius is None:
            reject("open_section", record)
            continue
        record["radius_mm"] = radius
        if not options.min_radius_mm <= radius <= options.max_radius_mm:
            reject("radius", record)
            continue
        vector = seed - origin
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            reject("degenerate_direction", record)
            continue
        result.branches.append(DetectedBranch(origin.tolist(), seed.tolist(), radius, (vector / norm).tolist(),
                                              path.tolist(), score, radius_method))
        record["status"] = "accepted"
    return finish()
