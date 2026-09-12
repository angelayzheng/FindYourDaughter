"""Local Streamlit dashboard for inspecting CT NIfTI volumes and aorta masks."""

from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
from time import monotonic, sleep

import streamlit as st

# Streamlit runs this file as a script, so make repository packages importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import ScanCase, VolumeViewOptions
from frontend.cases import find_subjects, mask_choices
from frontend.point_cloud import render_projection
from frontend.render import VTKRenderSession, VTKUnavailable, render_3d


@st.cache_resource(show_spinner=False)
def load_case(image: Path, mask: Path | None, image_mtime: int, mask_mtime: int | None) -> ScanCase:
    """Reuse loaded tensors across camera and display changes."""
    return ScanCase.from_nifti(image, mask)


def prepare_view(case: ScanCase, options: VolumeViewOptions, *, azimuth: int,
                 elevation: int, zoom: float, show_volume: bool, show_mask: bool,
                 mip: bool, cut_at_k: int | None, show_slices: bool,
                 show_planes: bool, slice_indices: tuple[int, int, int],
                 session: VTKRenderSession | None) -> tuple[bytes, str, str, tuple[int, int] | None, VTKRenderSession | None]:
    """Prefer the native VTK scene and use CPU projection when OpenGL is unavailable."""
    try:
        if session is None:
            session = VTKRenderSession(case.image.source_path, case.mask.source_path if case.mask else None,
                                       options)
        png = render_3d(case, options, azimuth=azimuth, elevation=elevation, zoom=zoom,
                        show_volume=show_volume, show_mask=show_mask, mip=mip,
                        cut_at_k=cut_at_k, show_slices=show_slices,
                        show_planes=show_planes, slice_indices=slice_indices,
                        session=session)
        return png, "VTK volume", "", None, session
    except VTKUnavailable as error:
        if session is not None:
            session.close()
        png, ct_count, mask_count = render_projection(
            case, frame=options.frame, window=options.window, level=options.level,
            show_volume=show_volume, show_mask=show_mask,
            azimuth=azimuth, elevation=elevation,
        )
        return png, "CPU preview", str(error), (ct_count, mask_count), None


st.set_page_config(page_title="Branchseed scan viewer", page_icon="🫀", layout="wide")
st.title("Patient scan viewer")
st.caption("Inspect local CT scans and supplied aorta masks. This viewer does not detect daughter arteries.")

with st.sidebar:
    st.header("Case")
    dataset_text = st.text_input("Dataset folder", value=str(ROOT / "dataset"))
    dataset = Path(dataset_text).expanduser()
    subjects = find_subjects(dataset)
    if not subjects:
        st.info("No scans found. Enter a folder with NIfTI files or subject subfolders.")
        st.stop()
    subject = st.selectbox("Subject", list(subjects), key=f"subject:{dataset.resolve()}")
    image = st.selectbox("CT scan", subjects[subject], format_func=lambda path: path.name,
                         key=f"image:{dataset.resolve()}:{subject}")
    masks = [None, *mask_choices(image)]
    mask = st.selectbox("Aorta mask", masks, format_func=lambda path: "None" if path is None else path.name,
                        index=1 if len(masks) == 2 else 0, key=f"mask:{image.resolve()}")

try:
    case = load_case(image, mask, image.stat().st_mtime_ns, mask.stat().st_mtime_ns if mask else None)
except (OSError, ValueError, IndexError) as error:
    st.error(f"Could not load this case: {error}")
    st.stop()

shape = case.image.shape
spacing = case.image.geometry.spacing
st.write(f"**{case.case_id}** · {shape[0]} × {shape[1]} × {shape[2]} voxels · "
         f"spacing {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} "
         f"{case.image.geometry.spatial_unit}")

with st.sidebar:
    st.header("3D view")
    frame = st.slider("Frame", 0, shape[3] - 1, 0) if len(shape) == 4 else 0
    window = st.slider("Window width", 1, 2000, 400)
    level = st.slider("Window level", -1000, 1500, 40)
    opacity = st.slider("Volume opacity", 0.0, 0.5, 0.12, 0.01)
    max_dimension = st.select_slider("CT sampling limit", options=[64, 96, 128, 192, 256], value=128)
    azimuth = st.slider("Starting azimuth", -180, 180, 30)
    elevation = st.slider("Starting elevation", -90, 90, 25)
    zoom = st.slider("Zoom", 0.5, 2.5, 1.0, 0.1)
    show_volume = st.checkbox("Show CT volume", True)
    show_mask = st.checkbox("Show mask surface", True, disabled=case.mask is None)
    mip = st.checkbox("Maximum intensity projection")
    cut = st.checkbox("Cut at voxel K")
    cut_at_k = st.slider("Voxel K", 0, shape[2] - 1, shape[2] // 2) if cut else None
    st.header("Orthogonal slices")
    show_slices = st.checkbox("Show three slice views", True)
    show_planes = st.checkbox("Show slice planes in 3D", True, disabled=not show_slices)
    slice_indices = tuple(
        st.slider(f"Axis {axis} slice", 0, shape[dim] - 1, shape[dim] // 2,
                  disabled=not show_slices)
        for dim, axis in enumerate("IJK")
    )

case_key = (image, mask, image.stat().st_mtime_ns, mask.stat().st_mtime_ns if mask else None)
view_key = (case_key, frame, window, level, opacity, max_dimension, azimuth, elevation,
            zoom, show_volume, show_mask, mip, cut_at_k, show_slices, show_planes,
            slice_indices)
session_key = (case_key, frame, max_dimension)
if st.session_state.get("vtk_session_key") != session_key:
    old_session = st.session_state.pop("vtk_session", None)
    if old_session is not None:
        old_session.close()
    st.session_state["vtk_session_key"] = session_key
if st.session_state.get("rendered_view") != view_key:
    try:
        options = VolumeViewOptions(frame=frame, max_dimension=max_dimension,
                                    window=window, level=level, opacity=opacity)
        started = monotonic()
        progress = st.progress(0, text="Rendering VTK volume · 0.0s elapsed")
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(prepare_view, case, options, azimuth=azimuth,
                                  elevation=elevation, zoom=zoom, show_volume=show_volume,
                                  show_mask=show_mask, mip=mip, cut_at_k=cut_at_k,
                                  show_slices=show_slices, show_planes=show_planes,
                                  slice_indices=slice_indices,
                                  session=st.session_state.get("vtk_session"))
            while not pending.done():
                elapsed = monotonic() - started
                progress.progress(min(95, 5 + int(elapsed * 3)),
                                  text=f"Rendering 3D view · {elapsed:.1f}s elapsed")
                sleep(0.15)
            png, mode, reason, counts, session = pending.result()
        elapsed = monotonic() - started
        progress.progress(100, text=f"3D view ready · {elapsed:.1f}s")
        st.session_state["preview_png"] = png
        st.session_state["render_mode"] = mode
        st.session_state["fallback_reason"] = reason
        st.session_state["point_counts"] = counts
        st.session_state["render_seconds"] = elapsed
        st.session_state["rendered_view"] = view_key
        st.session_state["vtk_session"] = session
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        st.session_state.pop("preview_png", None)
        st.error(f"3D view failed: {error}")

if st.session_state.get("rendered_view") == view_key and "preview_png" in st.session_state:
    mode = st.session_state["render_mode"]
    st.success(f"{mode} ready in {st.session_state['render_seconds']:.1f}s")
    if st.session_state["fallback_reason"]:
        st.info(f"VTK render unavailable: {st.session_state['fallback_reason']} Showing the CPU preview.")
    if st.session_state["point_counts"] is not None:
        ct_count, mask_count = st.session_state["point_counts"]
        if ct_count + mask_count == 0:
            st.warning("No CT voxels fall in the selected window and the mask is empty or hidden. Adjust the window or select a mask.")
    st.image(st.session_state["preview_png"], width="stretch")
    st.caption(f"{mode} · NIfTI RAS display geometry. Sliders update the view automatically.")
    st.download_button("Download view as PNG", st.session_state["preview_png"],
                       file_name=f"{case.case_id}_3d.png", mime="image/png", on_click="ignore")
