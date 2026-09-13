"""Display backend LPS-mm detections on the native viewer's RAS-mm scene."""

from __future__ import annotations

import numpy as np
import vtk

from core import Scan, ScanCase, ScanGeometry


LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])
PATH_COLOR = (0.35, 0.7, 1.0)
OSTIUM_COLOR = (1.0, 0.85, 0.15)
SEED_COLOR = (0.1, 1.0, 0.85)


def scan_case_from_backend(case) -> ScanCase:
    """Copy the exact working grid, including any loader resampling, for display."""
    import SimpleITK as sitk

    origin = np.array(case.image.TransformIndexToPhysicalPoint((0, 0, 0)))
    affine = np.eye(4)
    affine[:3, 3] = origin
    for axis in range(3):
        index = tuple(int(i == axis) for i in range(3))
        affine[:3, axis] = np.array(case.image.TransformIndexToPhysicalPoint(index)) - origin
    affine[:3] *= LPS_TO_RAS[:, None]
    geometry = ScanGeometry(affine, "mm")
    return ScanCase(
        Scan(sitk.GetArrayFromImage(case.image).transpose(2, 1, 0), geometry, case.image_path),
        Scan(sitk.GetArrayFromImage(case.aorta_mask).transpose(2, 1, 0), geometry, case.mask_path),
    )


def _polydata(segments=(), markers=()):
    points, lines, vertices = vtk.vtkPoints(), vtk.vtkCellArray(), vtk.vtkCellArray()
    points.SetDataTypeToDouble()
    for segment in segments:
        lines.InsertNextCell(len(segment))
        for point in segment:
            lines.InsertCellPoint(points.InsertNextPoint(point))
    for point in markers:
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(points.InsertNextPoint(point))
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetLines(lines)
    mesh.SetVerts(vertices)
    return mesh


def _actor(renderer, mesh, color, *, width=2.5):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(width)
    actor.GetProperty().SetPointSize(9)
    actor.GetProperty().LightingOff()
    actor.PickableOff()
    renderer.AddActor(actor)
    return actor


def _sphere(renderer, center, color, radius):
    source = vtk.vtkSphereSource()
    source.SetCenter(center)
    source.SetRadius(radius)
    source.SetThetaResolution(20)
    source.SetPhiResolution(16)
    source.Update()
    return _actor(renderer, source.GetOutput(), color)


def slice_segments(voxels: np.ndarray, dim: int, index: int) -> list[np.ndarray]:
    """Clip each path segment to a one-voxel slab, then project onto its center."""
    segments = []
    for left, right in zip(voxels[:-1], voxels[1:]):
        delta = right - left
        if abs(delta[dim]) < 1e-12:
            if abs(left[dim] - index) > 0.5:
                continue
            low, high = 0.0, 1.0
        else:
            crossings = sorted(((index - .5 - left[dim]) / delta[dim],
                                (index + .5 - left[dim]) / delta[dim]))
            low, high = max(0.0, crossings[0]), min(1.0, crossings[1])
            if low > high:
                continue
        segment = np.array([left + low * delta, left + high * delta])
        segment[:, dim] = index
        segments.append(segment)
    return segments


class DetectionOverlay:
    """Short centerlines and measured seed rings; no inferred vessel surfaces."""

    def __init__(self, viewer, result):
        if viewer.case.image.geometry.spatial_unit != "mm" or viewer.data.ndim != 3:
            raise ValueError("Detector overlays require a 3-D display grid in millimetres")
        self.viewer, self.result = viewer, result
        self.selected = 0 if result.branches else None
        self.visible = True
        self.actors, self.slice_actors, self.paths, self.voxels = [], [], [], []
        self.ostia, self.seeds, self.rings = [], [], []
        inverse = np.linalg.inv(viewer.affine)
        for branch in result.branches:
            path = np.array(branch.centerline_xyz_mm) * LPS_TO_RAS
            ostium = np.array(branch.ostium_xyz_mm) * LPS_TO_RAS
            seed = np.array(branch.seed_xyz_mm) * LPS_TO_RAS
            self.paths.append(path)
            self.voxels.append((np.c_[path, np.ones(len(path))] @ inverse.T)[:, :3])
            self.ostia.append(ostium)
            self.seeds.append(seed)
            direction = np.array(branch.direction_xyz) * LPS_TO_RAS
            # Match the detector's local 4-6 mm tangent for the seed section.
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            tangent = np.array([np.interp(min(6, arc[-1]), arc, path[:, axis])
                                - np.interp(4, arc, path[:, axis]) for axis in range(3)])
            tangent /= np.linalg.norm(tangent)
            u = np.cross(tangent, np.eye(3)[np.argmin(np.abs(tangent))])
            u /= np.linalg.norm(u)
            v = np.cross(tangent, u)
            angles = np.linspace(0, 2 * np.pi, 65)
            ring = seed + branch.radius_mm * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
            self.rings.append(ring)
            actors = [_actor(viewer.scene, _polydata([path]), PATH_COLOR, width=4),
                      _sphere(viewer.scene, ostium, OSTIUM_COLOR, .7),
                      _sphere(viewer.scene, seed, SEED_COLOR, .5),
                      _actor(viewer.scene, _polydata([ring]), SEED_COLOR)]
            cone = vtk.vtkConeSource()
            cone.SetCenter(seed - .5 * direction)
            cone.SetDirection(direction)
            cone.SetHeight(1.0)
            cone.SetRadius(.4)
            cone.SetResolution(20)
            cone.Update()
            actors.append(_actor(viewer.scene, cone.GetOutput(), SEED_COLOR))
            for actor in actors:
                actor.PickableOn()
            self.actors.append(actors)
            self.slice_actors.append([
                [_actor(renderer, _polydata(), color) for color in (PATH_COLOR, OSTIUM_COLOR, SEED_COLOR)]
                for renderer in viewer.slice_renderers
            ])
        self.label = vtk.vtkBillboardTextActor3D()
        self.label.GetTextProperty().SetColor(SEED_COLOR)
        self.label.GetTextProperty().SetFontSize(17)
        self.label.GetTextProperty().SetBackgroundColor(.035, .045, .065)
        self.label.GetTextProperty().SetBackgroundOpacity(.8)
        self.label.SetDisplayOffset(12, 12)
        self.label.PickableOff()
        viewer.scene.AddActor(self.label)
        self.title = viewer._text(viewer.scene, "", (.025, .88), 17)
        self.detail = viewer._text(viewer.scene, "", (.025, .825), 15)
        viewer._text(viewer.scene, "Yellow: opening   Green: 5 mm seed + radius ring   Blue: centerline", (.025, .05), 14)
        viewer._text(viewer.scene, "Click branch / Left / Right: select   D overlay   A overview   J focus seed", (.025, .015), 14)
        self.update()

    def update(self):
        for index, actors in enumerate(self.actors):
            for actor in actors:
                actor.SetVisibility(self.visible)
                actor.GetProperty().SetOpacity(1 if index == self.selected else .5)
            actors[0].GetProperty().SetLineWidth(5 if index == self.selected else 3)
        self.label.SetVisibility(self.visible and self.selected is not None)
        method = self.result.diagnostics.get("detector", "refined")
        self.title.SetInput(f"EXPERIMENTAL {method} / {len(self.actors)} candidates / overlay {'ON' if self.visible else 'OFF'}")
        if self.selected is None:
            self.detail.SetInput("No candidates passed the current detector filters")
        else:
            branch = self.result.branches[self.selected]
            name = f"branch_{self.selected + 1:03d}"
            self.label.SetInput(name)
            self.label.SetPosition(self.seeds[self.selected])
            fallback = " (approximate fallback)" if branch.radius_method == "distance_transform_fallback" else ""
            self.detail.SetInput(f"{name} / {self.selected + 1} of {len(self.actors)} / radius {branch.radius_mm:.2f} mm{fallback}")
        self.update_slices()

    def update_slices(self):
        inverse = np.linalg.inv(self.viewer.affine)
        for index, voxels in enumerate(self.voxels):
            for slot, dim in enumerate((2, 1, 0)):
                plane_index = self.viewer.indices[dim]
                axes = [axis for axis in range(3) if axis != dim]
                normal = np.cross(self.viewer.affine[:3, axes[0]], self.viewer.affine[:3, axes[1]])
                normal = normal / np.linalg.norm(normal) * .01

                def world(points):
                    return (np.c_[points, np.ones(len(points))] @ self.viewer.affine.T)[:, :3] + normal

                segments = [world(segment) for segment in slice_segments(voxels, dim, plane_index)]
                meshes = [_polydata(segments)]
                for point in (self.ostia[index], self.seeds[index]):
                    voxel = (inverse @ np.r_[point, 1])[:3]
                    inside = abs(voxel[dim] - plane_index) <= .5
                    voxel[dim] = plane_index
                    meshes.append(_polydata(markers=world([voxel]) if inside else []))
                for actor, mesh in zip(self.slice_actors[index][slot], meshes):
                    actor.GetMapper().SetInputData(mesh)
                    actor.SetVisibility(self.visible and mesh.GetNumberOfPoints() > 0)
                    actor.GetProperty().SetOpacity(1 if index == self.selected else .45)
                self.viewer.slice_renderers[slot].ResetCameraClippingRange()

    def select(self, index: int, *, focus=True):
        if not self.actors:
            return
        self.selected = index % len(self.actors)
        voxel = np.linalg.solve(self.viewer.affine, np.r_[self.seeds[self.selected], 1])[:3]
        self.viewer.slice_focus_voxel = voxel.copy() if focus else None
        for dim in range(3):
            self.viewer.set_slice(dim, voxel[dim])
        self.update()
        if focus:
            center = self.seeds[self.selected]
            extent = max(15.0, self.result.branches[self.selected].radius_mm * 4)
            bounds = [value for coordinate in center for value in (coordinate - extent, coordinate + extent)]
            self.viewer.scene.ResetCamera(bounds)
            for renderer in self.viewer.slice_renderers:
                renderer.GetActiveCamera().SetParallelScale(extent)
                renderer.ResetCameraClippingRange()
        self.viewer._render()

    def toggle(self):
        self.visible = not self.visible
        self.update()
        self.viewer._render()

    def overview(self):
        self.viewer.slice_focus_voxel = None
        self.viewer._reset_camera(focus_mask=True)
        for dim in range(3):
            self.viewer._update_slice(dim, reset_camera=True)
        self.update_slices()
        self.viewer._render()

    def pick(self, x: int, y: int) -> bool:
        if not self.visible:
            return False
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(.008)
        picker.PickFromListOn()
        for actors in self.actors:
            for actor in actors:
                picker.AddPickList(actor)
        if picker.Pick(x, y, 0, self.viewer.scene):
            for index, actors in enumerate(self.actors):
                if picker.GetActor() in actors:
                    self.select(index)
                    return True
        return False
