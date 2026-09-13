"""Native VTK volume viewer with full-resolution, linked slice views.

VTK image coordinates are native voxel indices. All 3-D props use the same
NIfTI voxel-to-RAS affine; this also preserves oblique and anisotropic scans.
These display coordinates do not replace the evaluator's SimpleITK LPS system.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
except ImportError as error:
    raise ImportError("The desktop viewer needs VTK. Install requirements-frontend.txt or use the offline installer.") from error

from core import ScanCase, VolumeViewOptions
from core.denoise import denoise_volume


def vtk_matrix(affine: np.ndarray):
    result = vtk.vtkMatrix4x4()
    for row in range(4):
        for column in range(4):
            result.SetElement(row, column, float(affine[row, column]))
    return result


def vtk_image(data: np.ndarray, *, spacing=(1, 1, 1), origin=(0, 0, 0)):
    """Copy a native (i,j,k) tensor into VTK's i-fastest scalar layout."""
    image = vtk.vtkImageData()
    image.SetDimensions(data.shape)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.GetPointData().SetScalars(numpy_to_vtk(data.ravel(order="F"), deep=True))
    return image


def mask_surface(mask: np.ndarray):
    """Extract the supplied foreground at full resolution, including scan edges."""
    foreground = np.isfinite(mask) & (mask > 0)
    occupied = [np.flatnonzero(foreground.any(axis=tuple(a for a in range(3) if a != dim))) for dim in range(3)]
    if not occupied[0].size:
        return None
    start = [int(indices[0]) for indices in occupied]
    stop = [int(indices[-1]) + 1 for indices in occupied]
    cropped = foreground[tuple(slice(a, b) for a, b in zip(start, stop))]
    # Padding closes a surface at cropped scan faces; it does not infer anatomy.
    image = vtk_image(np.pad(cropped, 1).astype(np.uint8), origin=tuple(a - 1 for a in start))
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, 0.5)
    contour.ComputeScalarsOff()
    contour.Update()
    mesh = vtk.vtkPolyData()
    mesh.ShallowCopy(contour.GetOutput())
    return mesh


class VolumeViewer:
    """One case per native window. Constructing a viewer does not open a window."""

    def __init__(self, case: ScanCase, options: VolumeViewOptions | None = None, *, detection=None) -> None:
        self.case = ScanCase(case.image, case.mask)
        if detection is not None and (case.image.data.ndim != 3 or case.image.geometry.spatial_unit != "mm"):
            raise ValueError("Detector overlays require a 3-D display grid in millimetres")
        self.options = options or VolumeViewOptions()
        self.raw_data = case.image.volume(self.options.frame)
        self.data = (
            denoise_volume(
                self.raw_data,
                tolerance=self.options.denoise_tolerance,
                min_neighbors=self.options.denoise_min_neighbors,
            )
            if self.options.denoise
            else self.raw_data
        )
        if min(self.data.shape) < 2:
            raise ValueError("3-D visualization requires at least two voxels along each spatial axis")
        self.mask = None if case.mask is None else case.mask.volume(
            self.options.frame if case.mask.data.ndim == 4 else 0
        )
        self.affine = case.image.geometry.affine_ras
        self.transform = vtk_matrix(self.affine)
        self.indices = [size // 2 for size in self.data.shape]
        self.slice_focus_voxel = None
        self.level, self.width = self.options.level, self.options.window
        self.min_intensity = self.options.min_intensity
        self.opacity = self.options.opacity
        self.clip_enabled = False
        self.planes_visible = False
        self.mip = False
        self.ready = False
        self.detection_overlay = None
        self.sliders = []
        self.slice_actors, self.slice_labels, self.plane_sources = [], [], []
        self.crosshair_lines = []
        self.default_screenshot = Path("nifti_previews") / f"{case.case_id}_3d.png"

        self.window = vtk.vtkRenderWindow()
        self.window.SetWindowName(f"Branchseed 3D / {case.case_id}")
        self.window.SetSize(1400, 900)
        self.window.SetMultiSamples(0)
        self.controls_height = .28 if detection is not None else .24
        self.scene = self._renderer((0, self.controls_height, 0.68, 1))
        self.controls = self._renderer((0, 0, 0.68, self.controls_height))
        self.slice_renderers = [self._renderer((0.68, low, 1, high)) for low, high in ((0.67, 1), (0.34, 0.67), (0, 0.34))]
        scene_title = "3D aorta + branch candidates" if detection is not None else "3D volume + supplied mask"
        self._text(self.scene, f"{case.case_id} / {scene_title}", (0.025, 0.94), 20)
        self.status = self._text(self.controls, "", (0.03, 0.9 if detection is not None else .82), 15)
        self._text(self.controls, "Drag: rotate   Wheel: zoom / scroll slices   Click slice: move cross-section", (0.03, .8 if detection is not None else .70), 13)
        self._text(self.controls, "V volume   M mask   P planes   C cut at K   B MIP   F focus mask   R reset   S save   Q close", (0.03, .71 if detection is not None else .60), 12)

        self._build_volume()
        self._build_mask()
        self._build_outline()
        self._build_slices()
        self._update_transfer()
        self._reset_camera()

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(self.style)
        self.style.AddObserver("LeftButtonPressEvent", self._left_click)
        self.style.AddObserver("MouseWheelForwardEvent", lambda *_: self._wheel(1))
        self.style.AddObserver("MouseWheelBackwardEvent", lambda *_: self._wheel(-1))
        self.style.AddObserver("CharEvent", lambda *_: None)
        self.interactor.AddObserver("KeyPressEvent", self._key)
        axes = vtk.vtkAxesActor()
        axes.SetXAxisLabelText("R")
        axes.SetYAxisLabelText("A")
        axes.SetZAxisLabelText("S")
        self.orientation = vtk.vtkOrientationMarkerWidget()
        self.orientation.SetOrientationMarker(axes)
        self.orientation.SetInteractor(self.interactor)
        self.orientation.SetCurrentRenderer(self.scene)
        self.orientation.SetViewport(0.02, 0.02, 0.16, 0.20)
        self._build_sliders()
        if detection is not None:
            from desktop.detection_overlay import DetectionOverlay

            self.detection_overlay = DetectionOverlay(self, detection)
            self.default_screenshot = Path("nifti_previews") / f"{case.case_id}_detection_3d.png"
            method = detection.diagnostics.get("detector", "refined")
            if method != "baseline":
                self.default_screenshot = Path("nifti_previews") / f"{case.case_id}_{method}_detection_3d.png"
            self.volume.SetVisibility(False)
            self.outline_actor.SetVisibility(False)
            if self.mask_actor:
                self.mask_actor.GetProperty().SetOpacity(.45)
            self.detection_overlay.select(0)
            self._reset_camera(focus_mask=True)
        self._update_status()

    def _renderer(self, viewport):
        renderer = vtk.vtkRenderer()
        renderer.SetViewport(*viewport)
        renderer.SetBackground(0.035, 0.045, 0.065)
        self.window.AddRenderer(renderer)
        return renderer

    @staticmethod
    def _text(renderer, text, position, size):
        actor = vtk.vtkTextActor()
        actor.SetInput(text)
        actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        actor.SetPosition(*position)
        actor.GetTextProperty().SetFontSize(size)
        actor.GetTextProperty().SetColor(0.86, 0.9, 0.96)
        actor.GetTextProperty().SetBackgroundColor(.035, .045, .065)
        actor.GetTextProperty().SetBackgroundOpacity(.8)
        renderer.AddViewProp(actor)
        return actor

    def _build_volume(self):
        self.stride = max(1, int(np.ceil(max(self.data.shape) / self.options.max_dimension)))
        self.strides = tuple(min(self.stride, size - 1) for size in self.data.shape)
        sampled = self.data[tuple(slice(None, None, step) for step in self.strides)]
        finite = sampled[np.isfinite(sampled)]
        if not finite.size:
            raise ValueError("The selected frame has no finite intensity values")
        self.scalar_range = (float(finite.min()), float(finite.max()))
        sampled = np.nan_to_num(sampled, nan=self.scalar_range[0], posinf=self.scalar_range[1], neginf=self.scalar_range[0])
        self.volume_image = vtk_image(sampled, spacing=self.strides)
        self.volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        self.volume_mapper.SetNumberOfThreads(1)
        self.volume_mapper.SetInputData(self.volume_image)
        self.volume_mapper.SetBlendModeToComposite()
        self.volume_mapper.SetSampleDistance(float(self.stride))
        self.volume_mapper.SetInteractiveSampleDistance(float(self.stride * 2))
        self.volume_mapper.SetImageSampleDistance(1.5)
        self.volume_mapper.SetMaximumImageSampleDistance(4)
        self.volume_property = vtk.vtkVolumeProperty()
        self.volume_property.SetInterpolationTypeToLinear()
        self.volume_property.ShadeOff()
        self.volume_property.SetScalarOpacityUnitDistance(float(self.stride))
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.volume_mapper)
        self.volume.SetProperty(self.volume_property)
        self.volume.SetUserMatrix(self.transform)
        self.scene.AddVolume(self.volume)

    def _build_mask(self):
        self.mask_actor = None
        mesh = None if self.mask is None else mask_surface(self.mask)
        if mesh is not None:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(mesh)
            mapper.ScalarVisibilityOff()
            self.mask_actor = vtk.vtkActor()
            self.mask_actor.SetMapper(mapper)
            self.mask_actor.SetUserMatrix(self.transform)
            self.mask_actor.GetProperty().SetColor(1, 0.12, 0.16)
            self.mask_actor.GetProperty().SetSpecular(0.25)
            self.scene.AddActor(self.mask_actor)

    def _build_outline(self):
        outline = vtk.vtkOutlineSource()
        outline.SetBounds(0, self.data.shape[0] - 1, 0, self.data.shape[1] - 1, 0, self.data.shape[2] - 1)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(outline.GetOutputPort())
        self.outline_actor = vtk.vtkActor()
        self.outline_actor.SetMapper(mapper)
        self.outline_actor.SetUserMatrix(self.transform)
        self.outline_actor.GetProperty().SetColor(0.25, 0.34, 0.43)
        self.scene.AddActor(self.outline_actor)

    def _slice_texture(self, dim):
        selection = [slice(None)] * 3
        selection[dim] = self.indices[dim]
        data = self.data[tuple(selection)]
        low = self.level - self.width / 2
        gray = np.nan_to_num(np.clip((data - low) / self.width, 0, 1), nan=0)
        if self.min_intensity is not None:
            gray[data < self.min_intensity] = 0
        rgb = np.repeat(gray[..., None], 3, axis=2)
        if self.mask is not None:
            mask = self.mask[tuple(selection)]
            foreground = np.isfinite(mask) & (mask > 0)
            rgb[foreground] = rgb[foreground] * 0.55 + np.array([1, 0, 0]) * 0.45
        pixels = np.asarray(rgb * 255, dtype=np.uint8)
        image = vtk.vtkImageData()
        image.SetDimensions(data.shape[0], data.shape[1], 1)
        image.GetPointData().SetScalars(numpy_to_vtk(pixels.transpose(1, 0, 2).reshape(-1, 3), deep=True))
        texture = vtk.vtkTexture()
        texture.SetInputData(image)
        texture.SetColorModeToDirectScalars()
        texture.InterpolateOff()
        return texture

    def _build_slices(self):
        for slot, dim in enumerate((2, 1, 0)):
            plane = vtk.vtkPlaneSource()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(plane.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().LightingOff()
            plane_actor = vtk.vtkActor()
            plane_actor.SetMapper(mapper)
            plane_actor.GetProperty().LightingOff()
            plane_actor.SetVisibility(False)
            self.slice_renderers[slot].AddActor(actor)
            self.scene.AddActor(plane_actor)
            self.slice_actors.append((actor, plane_actor))
            self.plane_sources.append(plane)
            self.slice_labels.append(self._text(self.slice_renderers[slot], "", (0.025, 0.90), 16))
            self._update_slice(dim, reset_camera=True)
            lines = []
            for _ in range(2):
                line = vtk.vtkLineSource()
                line_mapper = vtk.vtkPolyDataMapper()
                line_mapper.SetInputConnection(line.GetOutputPort())
                line_actor = vtk.vtkActor()
                line_actor.SetMapper(line_mapper)
                line_actor.GetProperty().SetColor(0.1, 0.85, 0.9)
                line_actor.GetProperty().SetOpacity(0.65)
                line_actor.GetProperty().LightingOff()
                line_actor.PickableOff()
                self.slice_renderers[slot].AddActor(line_actor)
                lines.append(line)
            self.crosshair_lines.append(lines)
        self._update_crosshairs()

    def _update_crosshairs(self):
        for slot, dim in enumerate((2, 1, 0)):
            remaining = [a for a in range(3) if a != dim]
            normal = np.cross(self.affine[:3, remaining[0]], self.affine[:3, remaining[1]])
            normal = normal / np.linalg.norm(normal) * min(self.case.image.geometry.spacing) * 0.001
            for line, axis in zip(self.crosshair_lines[slot], remaining):
                start, end = np.array(self.indices, dtype=float), np.array(self.indices, dtype=float)
                start[axis], end[axis] = -0.5, self.data.shape[axis] - 0.5
                line.SetPoint1((self.affine @ np.append(start, 1))[:3] + normal)
                line.SetPoint2((self.affine @ np.append(end, 1))[:3] + normal)

    def _update_slice(self, dim, *, reset_camera=False):
        slot = 2 - dim
        axes = [a for a in range(3) if a != dim]
        origin = np.full(3, -0.5)
        origin[dim] = self.indices[dim]
        corner = (self.affine @ np.append(origin, 1))[:3]
        u, v = (self.affine[:3, a] * self.data.shape[a] for a in axes)
        plane = self.plane_sources[slot]
        plane.SetOrigin(corner)
        plane.SetPoint1(corner + u)
        plane.SetPoint2(corner + v)
        texture = self._slice_texture(dim)
        for actor in self.slice_actors[slot]:
            actor.SetTexture(texture)
        code = self.case.image.geometry.axis_codes[dim]
        self.slice_labels[slot].SetInput(f"Axis {dim} ({code}) / index {self.indices[dim]} / {self.data.shape[dim] - 1}")
        renderer = self.slice_renderers[slot]
        camera = renderer.GetActiveCamera()
        center = corner + (u + v) / 2
        if self.slice_focus_voxel is not None and not reset_camera:
            focus = self.slice_focus_voxel.copy()
            focus[dim] = self.indices[dim]
            center = (self.affine @ np.r_[focus, 1])[:3]
        normal = np.cross(u, v)
        normal /= np.linalg.norm(normal)
        distance = max(np.linalg.norm(u), np.linalg.norm(v)) * 2
        camera.SetPosition(center + normal * distance)
        camera.SetFocalPoint(center)
        camera.SetViewUp(v / np.linalg.norm(v))
        camera.ParallelProjectionOn()
        if reset_camera:
            renderer.ResetCamera()
            camera.Zoom(0.88)
        renderer.ResetCameraClippingRange()

    def _update_transfer(self):
        low, high = self.level - self.width / 2, self.level + self.width / 2
        color = vtk.vtkColorTransferFunction()
        color.AddRGBPoint(low, 0, 0, 0)
        color.AddRGBPoint(self.level, 0.68, 0.55, 0.47)
        color.AddRGBPoint(high, 1, 0.96, 0.90)
        alpha = vtk.vtkPiecewiseFunction()
        alpha.AddPoint(low, 0)
        if self.min_intensity is None:
            alpha.AddPoint(self.level, self.opacity * 0.08)
            alpha.AddPoint(high, self.opacity)
        elif self.min_intensity >= high:
            alpha.AddPoint(high, 0)
            alpha.AddPoint(self.min_intensity, 0)
        else:
            cutoff = max(low, self.min_intensity)
            alpha.AddPoint(cutoff, 0)
            first_visible = cutoff + max((high - low) * 1e-6, 1e-6)
            alpha.AddPoint(first_visible, self.opacity * 0.02)
            if self.level > cutoff:
                alpha.AddPoint(self.level, self.opacity * 0.08)
            alpha.AddPoint(high, self.opacity)
        self.volume_property.SetColor(color)
        self.volume_property.SetScalarOpacity(alpha)

    def set_slice(self, dim, index):
        self.indices[dim] = int(np.clip(round(index), 0, self.data.shape[dim] - 1))
        self._update_slice(dim)
        self._update_crosshairs()
        if self.detection_overlay is not None:
            self.detection_overlay.update_slices()
        if self.sliders:
            self.sliders[dim].GetRepresentation().SetValue(self.indices[dim])
        if self.clip_enabled:
            self._update_clip()
        self._render()

    def set_window(self, level=None, width=None):
        if level is not None:
            self.level = float(level)
        if width is not None:
            self.width = max(float(width), 1e-6)
        self._update_transfer()
        for dim in range(3):
            self._update_slice(dim)
        self._render()

    def set_opacity(self, opacity):
        self.opacity = float(np.clip(opacity, 0, 1))
        self._update_transfer()
        self._render()

    def set_min_intensity(self, value):
        self.min_intensity = float(value)
        self._update_transfer()
        for dim in range(3):
            self._update_slice(dim)
        self._render()

    def _slider(self, title, low, high, value, x1, x2, y, callback):
        representation = vtk.vtkSliderRepresentation2D()
        representation.SetMinimumValue(low)
        representation.SetMaximumValue(high)
        representation.SetValue(value)
        representation.SetTitleText("")
        label_position = (x1 / .68, max(.02, y / self.controls_height - (.11 if self.controls_height == .28 else .13)))
        if title == "Minimum intensity" and self.controls_height == .28:
            label_position = (.35, y / self.controls_height + .07)
        self._text(self.controls, title, label_position, 13)
        representation.SetLabelFormat("%.0f" if title != "Volume opacity" else "%.2f")
        representation.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        representation.GetPoint1Coordinate().SetValue(x1, y)
        representation.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        representation.GetPoint2Coordinate().SetValue(x2, y)
        representation.SetSliderLength(0.04)
        representation.SetSliderWidth(0.08)
        representation.SetTubeWidth(0.012)
        representation.SetTitleHeight(0.10)
        representation.SetLabelHeight(0.07)
        representation.GetTitleProperty().SetColor(0.85, 0.9, 0.96)
        representation.GetLabelProperty().SetColor(0.85, 0.9, 0.96)
        widget = vtk.vtkSliderWidget()
        widget.SetInteractor(self.interactor)
        widget.SetCurrentRenderer(self.controls)
        widget.SetRepresentation(representation)
        widget.SetAnimationModeToJump()
        widget.AddObserver("InteractionEvent", lambda obj, _: callback(obj.GetRepresentation().GetValue()))
        self.sliders.append(widget)

    def _build_sliders(self):
        for dim, label in enumerate(("I / axis 0", "J / axis 1", "K / axis 2")):
            self._slider(label, 0, self.data.shape[dim] - 1, self.indices[dim],
                         0.04 + dim * 0.21, 0.20 + dim * 0.21, 0.105,
                         lambda value, dim=dim: self.set_slice(dim, value))
        low, high = self.scalar_range
        low, high = min(low, self.level), max(high, self.level + 1)
        self._slider("Window level", low, high, self.level, 0.04, 0.20, 0.035, lambda value: self.set_window(level=value))
        self._slider("Window width", 1, max(high - low, self.width, 2), self.width, 0.25, 0.41, 0.035, lambda value: self.set_window(width=value))
        self._slider("Volume opacity", 0, 0.5, self.opacity, 0.46, 0.62, 0.035, self.set_opacity)
        threshold_value = low if self.min_intensity is None else np.clip(self.min_intensity, low, high)
        self._slider("Minimum intensity", low, high, threshold_value, 0.04, 0.62,
                     .16 if self.controls_height == .28 else .14, self.set_min_intensity)

    def _reset_camera(self, focus_mask=False):
        camera = self.scene.GetActiveCamera()
        camera.SetPosition(1, -1.6, 0.8)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)
        self.scene.ResetCamera(self.mask_actor.GetBounds() if focus_mask and self.mask_actor else self.outline_actor.GetBounds())
        camera.Zoom(0.9)
        self.scene.ResetCameraClippingRange()

    def _update_clip(self):
        bounds = list(self.volume_image.GetBounds())
        bounds[5] = min(bounds[5], max(bounds[4] + 0.01, self.indices[2]))
        self.volume_mapper.SetCroppingRegionPlanes(bounds)
        self.volume_mapper.SetCroppingRegionFlagsToSubVolume()
        self.volume_mapper.SetCropping(self.clip_enabled)

    def _update_status(self):
        self.status.SetInput(
            f"Volume {'ON' if self.volume.GetVisibility() else 'OFF'} / "
            f"Mask {'ON' if self.mask_actor and self.mask_actor.GetVisibility() else 'OFF / empty'} / "
            f"Planes {'ON' if self.planes_visible else 'OFF'} / Cut {'ON' if self.clip_enabled else 'OFF'} / "
            f"{'MIP' if self.mip else 'Composite'} / CT stride {self.stride} / "
            f"Min {self.min_intensity if self.min_intensity is not None else 'OFF'} / "
            f"Denoise {'ON' if self.options.denoise else 'OFF'} / "
            f"{self.case.image.geometry.spatial_unit}"
        )

    def _render(self):
        self._update_status()
        if self.ready:
            self.window.Render()

    def _poked_renderer(self):
        return self.interactor.FindPokedRenderer(*self.interactor.GetEventPosition())

    def _wheel(self, amount):
        renderer = self._poked_renderer()
        if renderer in self.slice_renderers:
            dim = 2 - self.slice_renderers.index(renderer)
            self.set_slice(dim, self.indices[dim] + amount)
        elif renderer == self.scene:
            if amount > 0:
                self.style.OnMouseWheelForward()
            else:
                self.style.OnMouseWheelBackward()

    def _left_click(self, *_):
        renderer = self._poked_renderer()
        if renderer in self.slice_renderers:
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.005)
            x, y = self.interactor.GetEventPosition()
            if picker.Pick(x, y, 0, renderer):
                voxel = np.linalg.solve(self.affine, np.append(picker.GetPickPosition(), 1))[:3]
                for dim in range(3):
                    self.set_slice(dim, voxel[dim])
        elif renderer == self.scene:
            if self.detection_overlay is not None and self.detection_overlay.pick(*self.interactor.GetEventPosition()):
                return
            self.style.OnLeftButtonDown()

    def _key(self, *_):
        key = self.interactor.GetKeySym().lower()
        overlay = self.detection_overlay
        if overlay is not None:
            if key in ("bracketleft", "bracketright", "left", "right"):
                overlay.select((overlay.selected or 0) + (-1 if key in ("bracketleft", "left") else 1))
                return
            if key == "d":
                overlay.toggle()
                return
            if key == "a":
                overlay.overview()
                return
            if key == "j":
                overlay.select(overlay.selected or 0)
                return
        if key in ("q", "escape"):
            self.interactor.TerminateApp()
            return
        if key == "v":
            self.volume.SetVisibility(not self.volume.GetVisibility())
        elif key == "m" and self.mask_actor:
            self.mask_actor.SetVisibility(not self.mask_actor.GetVisibility())
        elif key == "p":
            self.planes_visible = not self.planes_visible
            for _, actor in self.slice_actors:
                actor.SetVisibility(self.planes_visible)
        elif key == "c":
            self.clip_enabled = not self.clip_enabled
            self._update_clip()
        elif key == "b":
            self.mip = not self.mip
            if self.mip:
                self.volume_mapper.SetBlendModeToMaximumIntensity()
            else:
                self.volume_mapper.SetBlendModeToComposite()
        elif key in ("r", "f"):
            self._reset_camera(focus_mask=key == "f")
        elif key == "s":
            self.save_screenshot(self.default_screenshot)
        self._render()

    def save_screenshot(self, path: str / Path) -> Path:
        path = Path(path)
        if path.suffix.lower() != ".png":
            raise ValueError("Screenshot path must end with .png")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.window.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(self.window)
        capture.ReadFrontBufferOff()
        capture.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(capture.GetOutputPort())
        writer.Write()
        print(f"Wrote {path}")
        return path

    def show(self, *, screenshot: str / Path / None = None, offscreen=False) -> None:
        if offscreen and screenshot is None:
            raise ValueError("Offscreen rendering requires a screenshot path")
        try:
            self.window.SetOffScreenRendering(offscreen)
            self.interactor.Initialize()
            self.ready = True
            self.orientation.EnabledOn()
            self.orientation.InteractiveOff()
            for slider in self.sliders:
                slider.EnabledOn()
            self.window.Render()
            if screenshot is not None:
                self.default_screenshot = Path(screenshot)
                self.save_screenshot(screenshot)
            if not offscreen:
                self.interactor.Start()
        finally:
            self.close()

    def close(self):
        self.ready = False
        for slider in self.sliders:
            slider.EnabledOff()
        self.orientation.EnabledOff()
        self.window.Finalize()
