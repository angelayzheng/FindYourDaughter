"""Isolated offscreen rendering of the native VTK scene for Streamlit."""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory
from threading import Lock, Thread
from weakref import finalize

from core import ScanCase, VolumeViewOptions


class VTKUnavailable(RuntimeError):
    """The local VTK render window cannot create a usable OpenGL context."""


def _render_environment() -> dict[str, str]:
    """Select the bundled Mesa CPU OpenGL backend on Windows x64."""
    environment = os.environ.copy()
    if sys.platform != "win32" or sysconfig.get_platform() != "win-amd64":
        return environment
    vendor = Path(__file__).resolve().parents[1] / "vendor" / "mesa"
    manifest_path = vendor / "manifest.json"
    if not manifest_path.is_file():
        raise VTKUnavailable("Bundled Mesa software OpenGL manifest is missing.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        platform = manifest["platform"]
        files = manifest["files"]
    except (OSError, ValueError, KeyError) as error:
        raise VTKUnavailable("Bundled Mesa software OpenGL manifest is invalid.") from error
    if platform != sysconfig.get_platform():
        raise VTKUnavailable("Bundled Mesa software OpenGL platform does not match this host.")
    mesa = vendor / "win_amd64"
    for name, expected in files.items():
        path = mesa / name
        if not path.is_file():
            raise VTKUnavailable(f"Bundled Mesa software OpenGL library is missing: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise VTKUnavailable(f"Bundled Mesa software OpenGL library failed verification: {name}")
    environment["PATH"] = str(mesa) + os.pathsep + environment.get("PATH", "")
    environment["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkOSOpenGLRenderWindow"
    environment["GALLIUM_DRIVER"] = "softpipe"
    return environment


@lru_cache(maxsize=1)
def probe_vtk() -> tuple[bool, str]:
    """Check a real offscreen context in another process, so failures cannot hang Streamlit."""
    try:
        environment = _render_environment()
        result = subprocess.run([sys.executable, "-m", "frontend.render", "--probe"],
                                capture_output=True, text=True, timeout=12, check=False,
                                env=environment)
    except VTKUnavailable as error:
        return False, str(error)
    except subprocess.TimeoutExpired:
        return False, "VTK's offscreen OpenGL check timed out."
    if result.returncode:
        diagnostic = (result.stderr or result.stdout).lower()
        if "osmesa.dll not found" in diagnostic:
            return False, "Win32 OpenGL failed and the OSMesa software OpenGL library is missing."
        return False, "VTK could not create an offscreen OpenGL context on this host."
    return True, ""


def render_3d(
    case: ScanCase,
    options: VolumeViewOptions,
    *,
    azimuth: float = 0,
    elevation: float = 0,
    zoom: float = 1,
    show_volume: bool = True,
    show_mask: bool = True,
    mip: bool = False,
    cut_at_k: int | None = None,
    show_slices: bool = False,
    show_planes: bool = False,
    plane_opacity: float = 0.85,
    focus_mask: bool = False,
    slice_indices: tuple[int, int, int] | None = None,
    session: VTKRenderSession | None = None,
) -> bytes:
    """Return a VTK scene PNG, or raise VTKUnavailable for a safe fallback."""
    image = case.image.source_path
    mask = case.mask.source_path if case.mask is not None else None
    if image is None or (case.mask is not None and mask is None):
        raise ValueError("VTK dashboard rendering requires NIfTI source paths")
    settings = dict(frame=options.frame, max_dimension=options.max_dimension,
                    window=options.window, level=options.level, opacity=options.opacity,
                    azimuth=azimuth, elevation=elevation, zoom=zoom,
                    show_volume=show_volume, show_mask=show_mask, mip=mip, cut_at_k=cut_at_k,
                    show_slices=show_slices, show_planes=show_planes,
                    plane_opacity=plane_opacity, focus_mask=focus_mask,
                    min_intensity=options.min_intensity,
                    slice_indices=slice_indices or tuple(size // 2 for size in case.image.shape[:3]))
    if session is not None:
        return session.render(settings)
    supported, reason = probe_vtk()
    if not supported:
        raise VTKUnavailable(reason)
    with TemporaryDirectory() as directory:
        output = Path(directory) / "volume.png"
        command = [sys.executable, "-m", "frontend.render", "--image", str(image),
                   "--output", str(output), "--settings", json.dumps(settings)]
        if mask is not None:
            command.extend(("--mask", str(mask)))
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=180, check=False, env=_render_environment())
        except subprocess.TimeoutExpired as error:
            raise VTKUnavailable("VTK volume rendering timed out after 180 seconds.") from error
        if result.returncode or not output.is_file():
            raise VTKUnavailable("VTK could not complete the offscreen volume render.")
        png = output.read_bytes()
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise VTKUnavailable("VTK did not produce a valid PNG image.")
        return png


class VTKRenderSession:
    """Keep one isolated VTK scene alive while Streamlit controls change."""

    def __init__(self, image: Path, mask: Path | None, options: VolumeViewOptions) -> None:
        self.key = (image, mask, options.frame, options.max_dimension)
        self._directory = TemporaryDirectory()
        self._responses: Queue[str | None] = Queue()
        self._lock = Lock()
        command = [sys.executable, "-m", "frontend.render", "--serve", "--image", str(image),
                   "--settings", json.dumps(dict(frame=options.frame,
                                                   max_dimension=options.max_dimension,
                                                   window=options.window, level=options.level,
                                                   opacity=options.opacity))]
        if mask is not None:
            command.extend(("--mask", str(mask)))
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                         env=_render_environment())
        self._finalizer = finalize(self, self._shutdown, self._process, self._directory)
        assert self._process.stdout is not None
        Thread(target=self._read_responses, args=(self._process.stdout, self._responses),
               daemon=True).start()
        try:
            self._receive(30)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _read_responses(stream, responses: Queue[str | None]) -> None:
        try:
            for line in stream:
                responses.put(line)
        except (OSError, ValueError):
            pass
        responses.put(None)

    def _receive(self, timeout: int) -> None:
        try:
            line = self._responses.get(timeout=timeout)
        except Empty as error:
            raise VTKUnavailable(f"VTK did not respond within {timeout} seconds.") from error
        if line is None:
            raise VTKUnavailable("The VTK render process exited unexpectedly.")
        try:
            response = json.loads(line)
        except ValueError as error:
            raise VTKUnavailable("The VTK render process returned an invalid response.") from error
        if not response.get("ok"):
            raise VTKUnavailable(str(response.get("error", "VTK could not render this view.")))

    def render(self, settings: dict) -> bytes:
        with self._lock:
            if self._process.poll() is not None:
                raise VTKUnavailable("The VTK render process has stopped.")
            output = Path(self._directory.name) / "volume.png"
            try:
                assert self._process.stdin is not None
                self._process.stdin.write(json.dumps({"settings": settings, "output": str(output)}) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise VTKUnavailable("The VTK render process stopped responding.") from error
            self._receive(90)
            try:
                png = output.read_bytes()
            except OSError as error:
                raise VTKUnavailable("VTK did not produce a PNG image.") from error
            if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise VTKUnavailable("VTK did not produce a valid PNG image.")
            return png

    def close(self) -> None:
        self._finalizer()

    @staticmethod
    def _shutdown(process: subprocess.Popen, directory: TemporaryDirectory) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        directory.cleanup()


def _configure_scene(viewer) -> None:
    viewer.window.SetSize(1000, 760)
    viewer.scene.SetViewport(0, 0, 1, 1)
    viewer.controls.SetDraw(False)
    for renderer in viewer.slice_renderers:
        renderer.SetDraw(False)
    viewer.window.SetOffScreenRendering(True)


def _apply_settings(viewer, settings: dict) -> None:
    show_slices = bool(settings["show_slices"])
    slices_were_visible = bool(viewer.slice_renderers[0].GetDraw())
    contrast_changed = (viewer.level != float(settings["level"]) or
                        viewer.width != float(settings["window"]) or
                        viewer.min_intensity != settings.get("min_intensity"))
    viewer.scene.SetViewport(0, 0, 0.68 if show_slices else 1, 1)
    for renderer in viewer.slice_renderers:
        renderer.SetDraw(show_slices)
    viewer.planes_visible = bool(settings["show_planes"])
    for _, plane_actor in viewer.slice_actors:
        plane_actor.SetVisibility(viewer.planes_visible)
        plane_actor.GetProperty().SetOpacity(float(settings.get("plane_opacity", 0.85)))
    viewer.level = float(settings["level"])
    viewer.width = float(settings["window"])
    viewer.opacity = float(settings["opacity"])
    viewer.min_intensity = settings.get("min_intensity")
    viewer._update_transfer()
    if show_slices or viewer.planes_visible:
        for dim, index in enumerate(settings["slice_indices"]):
            if index != viewer.indices[dim] or contrast_changed or not slices_were_visible:
                viewer.set_slice(dim, index)
    viewer.volume.SetVisibility(settings["show_volume"])
    if viewer.mask_actor is not None:
        viewer.mask_actor.SetVisibility(settings["show_mask"])
    if settings["mip"]:
        viewer.volume_mapper.SetBlendModeToMaximumIntensity()
    else:
        viewer.volume_mapper.SetBlendModeToComposite()
    viewer.clip_enabled = settings["cut_at_k"] is not None
    if viewer.clip_enabled:
        slice_k = viewer.indices[2]
        viewer.indices[2] = max(0, min(int(settings["cut_at_k"]), viewer.data.shape[2] - 1))
        viewer._update_clip()
        viewer.indices[2] = slice_k
    else:
        viewer._update_clip()
    viewer._reset_camera(focus_mask=bool(settings.get("focus_mask")))
    camera = viewer.scene.GetActiveCamera()
    camera.Azimuth(float(settings["azimuth"]))
    camera.Elevation(float(settings["elevation"]))
    camera.Zoom(float(settings["zoom"]))
    viewer.scene.ResetCameraClippingRange()


def _serve(case: ScanCase, options: VolumeViewOptions) -> None:
    from desktop.volume_viewer import VolumeViewer

    viewer = VolumeViewer(case, options)
    _configure_scene(viewer)
    print(json.dumps({"ok": True}), flush=True)
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                _apply_settings(viewer, request["settings"])
                # The native viewer prints a desktop status line after saving.
                from contextlib import redirect_stdout
                with redirect_stdout(sys.stderr):
                    viewer.save_screenshot(request["output"])
                response = {"ok": True}
            except Exception as error:
                response = {"ok": False, "error": str(error)}
            print(json.dumps(response), flush=True)
    finally:
        viewer.close()


def _direct_render(case: ScanCase, options: VolumeViewOptions, settings: dict, output: Path) -> None:
    from desktop.volume_viewer import VolumeViewer

    viewer = VolumeViewer(case, options)
    try:
        _configure_scene(viewer)
        _apply_settings(viewer, settings)
        viewer.save_screenshot(output)
    finally:
        viewer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--settings")
    args = parser.parse_args()
    if args.probe:
        import vtk

        window = vtk.vtkRenderWindow()
        window.SetOffScreenRendering(True)
        window.SetSize(16, 16)
        window.Render()
        if not window.SupportsOpenGL():
            raise SystemExit("No usable OpenGL render window")
        window.Finalize()
        return
    if args.image is None or args.settings is None or (not args.serve and args.output is None):
        parser.error("--image and --settings are required; --output is required unless serving")
    settings = json.loads(args.settings)
    options = VolumeViewOptions(**{key: settings[key] for key in
                                   ("frame", "max_dimension", "window", "level", "opacity")})
    case = ScanCase.from_nifti(args.image, args.mask)
    if args.serve:
        _serve(case, options)
    else:
        _direct_render(case, options, settings, args.output)


if __name__ == "__main__":
    main()
