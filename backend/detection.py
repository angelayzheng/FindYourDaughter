"""Experimental CPU baseline: aorta-connected lumen and short 3-D centerlines."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import time

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import SimpleITK as sitk

from backend.inputs import _same_geometry


@dataclass(frozen=True)
class DetectionOptions:
    """Provisional sensitivity settings, pending annotated development cases."""

    margin_mm: float = 15.0
    intensity_fraction: float = 0.55
    min_radius_mm: float = 0.8
    max_radius_mm: float = 6.0
    min_vesselness: float = 0.1

    def __post_init__(self) -> None:
        if not np.isfinite(tuple(vars(self).values())).all():
            raise ValueError("Detection settings must be finite")
        if self.margin_mm < 12 or not 0 < self.intensity_fraction < 1:
            raise ValueError("margin_mm must be at least 12 and intensity_fraction between 0 and 1")
        if not 0 < self.min_radius_mm <= self.max_radius_mm or not 0 <= self.min_vesselness <= 1:
            raise ValueError("Invalid radius or vesselness limits")


@dataclass
class DetectedBranch:
    ostium_xyz_mm: list[float]
    seed_xyz_mm: list[float]
    radius_mm: float
    direction_xyz: list[float]
    centerline_xyz_mm: list[list[float]]
    vesselness: float
    radius_method: str

    def prediction(self, instance_id: str) -> dict:
        return {"instance_id": instance_id, "parent_instance_id": "aorta",
                "ostium_xyz_mm": self.ostium_xyz_mm, "seed_xyz_mm": self.seed_xyz_mm,
                "radius_mm": self.radius_mm, "direction_xyz": self.direction_xyz}


@dataclass
class DetectionResult:
    branches: list[DetectedBranch] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def daughters(self) -> list[dict]:
        return [branch.prediction(f"branch_{i:03d}") for i, branch in enumerate(self.branches, 1)]


def _components(nodes: set[int], adjacency: list[set[int]]) -> list[list[int]]:
    remaining = set(nodes)
    groups = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack, group = [seed], []
        while stack:
            current = stack.pop()
            group.append(current)
            fresh = remaining & adjacency[current]
            remaining.difference_update(fresh)
            stack.extend(sorted(fresh, reverse=True))
        groups.append(sorted(group))
    return groups


def _graph(skeleton: np.ndarray) -> tuple[np.ndarray, list[set[int]]]:
    coordinates = np.argwhere(skeleton)
    lookup = {tuple(point): i for i, point in enumerate(coordinates)}
    offsets = [np.array(offset) for offset in product((-1, 0, 1), repeat=3) if any(offset)]
    adjacency = []
    for point in coordinates:
        adjacency.append({neighbor for offset in offsets if (neighbor := lookup.get(tuple(point + offset))) is not None})
    return coordinates, adjacency


def _point_at(points: np.ndarray, distance: float) -> np.ndarray:
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return np.array([np.interp(distance, arc, points[:, axis]) for axis in range(3)])


def _trace(root: int, origin: np.ndarray, points: np.ndarray, adjacency: list[set[int]],
           other_roots: set[int]) -> tuple[np.ndarray, str]:
    """Follow the proximal graph, pruning short spurs and stopping at forks."""
    visited = set(other_roots)
    path, current, length = [origin, points[root]], root, float(np.linalg.norm(points[root] - origin))

    def continues(start: int, previous: int, blocked: set[int]) -> bool:
        stack = [(start, previous, 0.0)]
        seen = set(blocked) | {previous}
        while stack:
            node, parent, distance = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            distance += float(np.linalg.norm(points[node] - points[parent]))
            if distance >= 2.0:
                return True
            stack.extend((next_node, node, distance) for next_node in sorted(adjacency[node] - seen))
        return False

    while length < 10.0:
        visited.add(current)
        choices = sorted(adjacency[current] - visited)
        if len(choices) > 1:
            choices = [node for node in choices if continues(node, current, visited)]
        if not choices:
            return np.asarray(path), "end"
        if len(choices) > 1:
            return np.asarray(path), "bifurcation"
        following = choices[0]
        length += float(np.linalg.norm(points[following] - points[current]))
        path.append(points[following])
        current = following
    array = np.asarray(path)
    return np.vstack([array[:-1], _point_at(array, 10.0)]), "length_limit"


def _vesselness(ct: np.ndarray, spacing_zyx: np.ndarray, blood: float, background: float) -> np.ndarray:
    normalized = np.clip((ct - background) / max(blood - background, 40.0), -1, 3).astype(np.float32)
    response = np.zeros(ct.shape, dtype=np.float32)
    objectness = sitk.ObjectnessMeasureImageFilter()
    objectness.SetNumberOfWorkUnits(4)
    objectness.SetBrightObject(True)
    objectness.SetObjectDimension(1)
    objectness.SetScaleObjectnessMeasure(False)
    objectness.SetGamma(0.2)
    for sigma in (0.8, 1.5, 2.5):
        # Sigma is physical. Multiplication provides scale-normalized Hessians.
        smoothed = ndi.gaussian_filter(normalized, sigma / spacing_zyx) * sigma**2
        volume = sitk.GetImageFromArray(smoothed)
        volume.SetSpacing(spacing_zyx[::-1].tolist())
        enhanced = objectness.Execute(volume)
        np.maximum(response, sitk.GetArrayViewFromImage(enhanced), out=response)
    return response


def _section_radius(image: sitk.Image, ct: np.ndarray, offset: np.ndarray,
                    seed: np.ndarray, tangent: np.ndarray, rough_radius: float,
                    upper_intensity: float) -> float | None:
    """Area-equivalent radius of the local lumen perpendicular to its path."""
    tangent = tangent / np.linalg.norm(tangent)
    axis = np.eye(3)[np.argmin(np.abs(tangent))]
    horizontal = np.cross(tangent, axis)
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(tangent, horizontal)
    center_index = np.array(image.TransformPhysicalPointToContinuousIndex(seed.tolist()))[::-1] - offset
    horizontal_index = np.array(image.TransformPhysicalPointToContinuousIndex((seed + horizontal).tolist()))[::-1] - offset - center_index
    vertical_index = np.array(image.TransformPhysicalPointToContinuousIndex((seed + vertical).tolist()))[::-1] - offset - center_index
    step = 0.4
    half_width = int(np.ceil(max(4.0, min(12.0, 2.5 * rough_radius)) / step))
    positions = np.arange(-half_width, half_width + 1) * step
    u, v = np.meshgrid(positions, positions)
    indices = center_index + u[..., None]*horizontal_index + v[..., None]*vertical_index
    values = ndi.map_coordinates(ct, indices.reshape(-1, 3).T, order=1, mode="constant", cval=np.nan).reshape(u.shape)
    finite = np.isfinite(values)
    ring = (np.hypot(u, v) >= half_width * step * .8) & finite
    if not np.any(ring):
        return None
    background = float(np.median(values[ring]))
    peak = float(values[half_width, half_width])
    if not np.isfinite(peak) or peak - background < 40:
        return None
    section = finite & (values >= (peak + background) / 2) & (values <= upper_intensity)
    labels, _ = ndi.label(section)
    label = labels[half_width, half_width]
    if not label:
        return None
    component = labels == label
    if (component[0].any() or component[-1].any() or component[:, 0].any() or component[:, -1].any()
            or np.any(ndi.binary_dilation(component) & ~finite)):
        return None
    return float(np.sqrt(np.count_nonzero(component) * step**2 / np.pi))


def detect_daughters(image: sitk.Image, aorta_mask: sitk.Image,
                     options: DetectionOptions | None = None) -> DetectionResult:
    """Propose direct daughters; this baseline has no measured clinical accuracy.

    Voxel indices below use NumPy z,y,x order. Graph lengths and exported points
    use the input image's physical LPS millimetres. No anatomical names or fixed
    branch count are assumed.
    """
    started = time.perf_counter()
    options = options or DetectionOptions()
    if (image.GetDimension() != 3 or aorta_mask.GetDimension() != 3
            or image.GetNumberOfComponentsPerPixel() != 1
            or aorta_mask.GetNumberOfComponentsPerPixel() != 1
            or not _same_geometry(image, aorta_mask)):
        raise ValueError("Detection requires matching scalar 3-D image and mask geometry")
    mask_values = sitk.GetArrayViewFromImage(aorta_mask)
    if not np.all((mask_values == 0) | (mask_values == 1)):
        raise ValueError("The aorta mask must be binary (0 or 1)")
    result = DetectionResult(diagnostics={"method": "experimental_connected_lumen_skeleton", "rejected": {}})

    def reject(reason: str) -> None:
        counts = result.diagnostics["rejected"]
        counts[reason] = counts.get(reason, 0) + 1

    if not np.any(mask_values):
        result.diagnostics["elapsed_seconds"] = time.perf_counter() - started
        return result
    spacing = np.array(image.GetSpacing())[::-1]
    bounds = []
    for axis in range(3):
        active = np.flatnonzero(np.any(mask_values, axis=tuple(i for i in range(3) if i != axis)))
        margin = int(np.ceil(options.margin_mm / spacing[axis]))
        bounds.append((max(0, int(active[0]) - margin), min(mask_values.shape[axis], int(active[-1]) + margin + 1)))
    selection = tuple(slice(lo, hi) for lo, hi in bounds)
    offset = np.array([lo for lo, _ in bounds])
    parent = mask_values[selection] > 0
    ct = np.asarray(sitk.GetArrayViewFromImage(image)[selection], dtype=np.float32)
    if min(ct.shape) < 3 or np.count_nonzero(parent) < 3:
        reject("insufficient_3d_support")
        result.diagnostics["elapsed_seconds"] = time.perf_counter() - started
        return result
    if not np.isfinite(ct).all():
        raise ValueError("CT values around the aorta must be finite")
    inside_distance = ndi.distance_transform_edt(parent, sampling=spacing)
    outside_distance = ndi.distance_transform_edt(~parent, sampling=spacing)
    core = parent & (inside_distance >= 2.0)
    blood = float(np.median(ct[core if np.any(core) else parent]))
    shell = (outside_distance >= 4) & (outside_distance <= 10)
    background = float(np.median(ct[shell])) if np.any(shell) else float(np.min(ct))
    threshold = max(60.0, background + options.intensity_fraction * (blood - background))
    result.diagnostics.update(blood_intensity=blood, background_intensity=background,
                              threshold=threshold, roi_shape_zyx=list(ct.shape))
    if blood <= threshold:
        reject("insufficient_contrast")
        result.diagnostics["elapsed_seconds"] = time.perf_counter() - started
        return result
    smooth = ndi.gaussian_filter(ct, np.maximum(0.6 / spacing, 0.65))
    core_values = smooth[core if np.any(core) else parent]
    spread = 1.4826 * float(np.median(np.abs(core_values - np.median(core_values))))
    upper = blood + max(200.0, 4 * spread)
    result.diagnostics["upper_intensity"] = upper
    lumen = parent | ((smooth >= threshold) & (smooth <= upper) & (outside_distance <= 12.0))
    labels, _ = ndi.label(lumen, structure=np.ones((3, 3, 3)))
    connected_labels = np.unique(labels[parent])
    lumen = np.isin(labels, connected_labels[connected_labels != 0])
    skeleton = skeletonize(np.pad(lumen, 1), method="lee")[1:-1, 1:-1, 1:-1]
    coordinates, adjacency = _graph(skeleton)
    result.diagnostics["skeleton_voxels"] = len(coordinates)
    if not len(coordinates):
        result.diagnostics["elapsed_seconds"] = time.perf_counter() - started
        return result
    raw_points = np.array([image.TransformIndexToPhysicalPoint(tuple(int(v) for v in (point + offset)[::-1]))
                           for point in coordinates])
    in_parent = parent[tuple(coordinates.T)]
    outside = set(np.flatnonzero(~in_parent).tolist())
    root_nodes = {node for node in outside if any(in_parent[neighbor] for neighbor in adjacency[node])}
    root_groups = _components(root_nodes, adjacency)
    result.diagnostics["root_candidates"] = len(root_groups)

    # Merge neighboring junction voxels so grid diagonals do not create false
    # bifurcations. Root contacts are found before this graph simplification.
    external = [neighbors & outside for neighbors in adjacency]
    junctions = {node for node in outside if len(external[node]) >= 3}
    groups = _components(junctions, external) + [[node] for node in sorted(outside - junctions)]
    mapping = {node: group_id for group_id, group in enumerate(groups) for node in group}
    # A representative must remain on the skeleton: a centroid of a curved
    # junction region can lie inside the parent or outside the observed lumen.
    points = np.array([raw_points[group][np.argmin(np.linalg.norm(
        raw_points[group] - raw_points[group].mean(axis=0), axis=1))] for group in groups])
    reduced = [set() for _ in groups]
    for node in sorted(outside):
        reduced[mapping[node]].update(mapping[neighbor] for neighbor in external[node]
                                     if mapping[neighbor] != mapping[node])
    all_roots = {mapping[node] for node in root_nodes}
    radius_field = ndi.distance_transform_edt(lumen, sampling=spacing)
    vesselness = _vesselness(ct, spacing, blood, background) if root_groups else None
    parent_coordinates = np.argwhere(parent) * spacing
    center = parent_coordinates.mean(axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov((parent_coordinates - center).T))
    parent_axis = eigenvectors[:, -1]
    projections = parent_coordinates @ parent_axis
    low, high = float(projections.min()), float(projections.max())
    normal_fields = np.gradient(ndi.gaussian_filter(outside_distance - inside_distance, 0.7), *spacing)

    for group in root_groups:
        crossings = [(raw_points[node] + raw_points[neighbor]) / 2 for node in group
                     for neighbor in adjacency[node] if in_parent[neighbor]]
        origin = np.mean(crossings, axis=0)
        origin_index = np.array(image.TransformPhysicalPointToContinuousIndex(origin.tolist()))[::-1] - offset
        root_projection = float((origin_index * spacing) @ parent_axis)
        normal = np.array([ndi.map_coordinates(field, origin_index[:, None], order=1, mode="nearest")[0]
                           for field in normal_fields])
        cap_alignment = abs(float(normal @ parent_axis)) / max(float(np.linalg.norm(normal)), 1e-6)
        if min(abs(root_projection - low), abs(root_projection - high)) <= 2 * spacing.max() and cap_alignment > 0.7:
            reject("parent_end_cap")
            continue
        starts = {mapping[node] for node in group}
        root = min(starts, key=lambda node: (np.linalg.norm(points[node] - origin), node))
        path, stop = _trace(root, origin, points, reduced, all_roots - starts)
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        if length < 5.0:
            reject("early_bifurcation" if stop == "bifurcation" else "short_path")
            continue
        seed = _point_at(path, 5.0)
        seed_index = np.array(image.TransformPhysicalPointToContinuousIndex(seed.tolist()))[::-1] - offset
        clearance = float(ndi.map_coordinates(outside_distance, seed_index[:, None], order=1, mode="nearest")[0]) - spacing.min() / 2
        if clearance < 2.0:
            reject("follows_parent_wall")
            continue
        witness = np.array([_point_at(path, distance) for distance in np.arange(1.0, length, 0.5)])
        witness_indices = np.array([image.TransformPhysicalPointToContinuousIndex(point.tolist())[::-1]
                                    for point in witness]) - offset
        observed = ndi.map_coordinates(smooth, witness_indices.T, order=1, mode="nearest")
        touches_parent = ndi.map_coordinates(parent.astype(np.uint8), witness_indices.T, order=0, mode="nearest")
        if np.any(touches_parent) or np.any(observed < threshold) or np.any(observed > upper):
            reject("unsupported_path")
            continue
        rough_radius = float(ndi.map_coordinates(radius_field, seed_index[:, None], order=1, mode="nearest")[0])
        samples = np.array([_point_at(path, distance) for distance in np.linspace(2, min(length, 8), 12)])
        sample_indices = np.array([image.TransformPhysicalPointToContinuousIndex(point.tolist())[::-1] for point in samples]) - offset
        score = float(np.mean(ndi.map_coordinates(vesselness, sample_indices.T, order=1, mode="nearest")))
        if score < options.min_vesselness:
            reject("weak_tubularity")
            continue
        tangent = _point_at(path, min(length, 6.0)) - _point_at(path, 4.0)
        if np.linalg.norm(tangent) < 1e-6:
            reject("degenerate_direction")
            continue
        radius = _section_radius(image, smooth, offset, seed, tangent, rough_radius, upper)
        radius_method = "orthogonal_section"
        if radius is None or radius > 1.75 * rough_radius + spacing.min():
            # A nearby blood pool can join the section to the parent or another
            # vessel. Retain a conservative local distance estimate in that case.
            radius = rough_radius - spacing.min() / 2
            radius_method = "distance_transform_fallback"
        if not options.min_radius_mm <= radius <= options.max_radius_mm:
            reject("radius")
            continue
        direction = seed - origin
        direction_length = np.linalg.norm(direction)
        if direction_length < 1e-6:
            reject("degenerate_direction")
            continue
        direction /= direction_length
        result.branches.append(DetectedBranch(origin.tolist(), seed.tolist(), max(radius, 0.0),
                                              direction.tolist(), path.tolist(), score, radius_method))
    result.branches.sort(key=lambda branch: tuple(branch.ostium_xyz_mm[::-1]))
    result.diagnostics["elapsed_seconds"] = time.perf_counter() - started
    return result
