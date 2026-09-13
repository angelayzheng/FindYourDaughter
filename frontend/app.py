"""Local, CPU-only inspection dashboard for CT scans and aorta branches."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys
from time import monotonic, sleep

import streamlit as st
import streamlit.components.v1 as components

# Streamlit runs this file as a script, so make repository packages importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.detectors import DETECTOR_NAMES
from backend.pipeline import run_case
from core import ScanCase, VolumeViewOptions
from frontend.browser_scene import scene_html, scene_payload
from frontend.cases import find_subjects, mask_choices
from frontend.point_cloud import render_projection
from frontend.render import VTKRenderSession, VTKUnavailable, render_3d


@st.cache_resource(show_spinner=False)
def load_case(image: Path, mask: Path | None, image_mtime: int,
              mask_mtime: int | None) -> ScanCase:
    """Reuse loaded tensors across camera and display changes."""
    return ScanCase.from_nifti(image, mask)


@st.cache_data(show_spinner=False)
def load_prediction(image: Path, mask: Path, image_mtime: int, mask_mtime: int,
                    detector: str) -> dict:
    """Run the evaluator detector once per input revision and algorithm."""
    return run_case(image, mask, detector=detector)


def prepare_view(case: ScanCase, options: VolumeViewOptions, *, azimuth: int,
                 elevation: int, zoom: float, show_volume: bool, show_mask: bool,
                 mip: bool, cut_at_k: int | None, show_slices: bool,
                 show_planes: bool, plane_opacity: float, focus_mask: bool,
                 slice_indices: tuple[int, int, int],
                 session: VTKRenderSession | None) -> tuple[bytes, str, str, tuple[int, int] | None, VTKRenderSession | None]:
    """Render the native VTK scene or a safe CPU projection fallback."""
    try:
        if session is None:
            session = VTKRenderSession(case.image.source_path,
                                       case.mask.source_path if case.mask else None, options)
        png = render_3d(case, options, azimuth=azimuth, elevation=elevation, zoom=zoom,
                        show_volume=show_volume, show_mask=show_mask, mip=mip,
                        cut_at_k=cut_at_k, show_slices=show_slices,
                        show_planes=show_planes, plane_opacity=plane_opacity,
                        focus_mask=focus_mask, slice_indices=slice_indices,
                        session=session)
        return png, "VTK volume", "", None, session
    except VTKUnavailable as error:
        if session is not None:
            session.close()
        png, ct_count, mask_count = render_projection(
            case, frame=options.frame, window=options.window, level=options.level,
            show_volume=show_volume, show_mask=show_mask,
            azimuth=azimuth, elevation=elevation)
        return png, "CPU preview", str(error), (ct_count, mask_count), None


st.set_page_config(page_title="Find Your Daughter", page_icon="🩻", layout="wide")
st.markdown("""<style>
  .stApp { background: #171417; color: #f0e7e9; }
  [data-testid="stSidebar"] { background: #211b20; border-right: 1px solid #75515f; }
  [data-testid="stMainBlockContainer"] { padding-top: 2rem !important; }
  [data-testid="stSidebarHeader"] { height: 2.25rem !important; margin-bottom: 0 !important; }
  [data-testid="stSidebarUserContent"] { padding-top: 0 !important; }
  .stApp, .stApp button, .stApp input, .stApp textarea { font-family: Consolas, 'Courier New', monospace; }
  h1, h2, h3 { font-family: Consolas, 'Courier New', monospace !important; letter-spacing: .015em; color: #f6edef !important; }
  h1 { font-size: 2.2rem !important; }
  [data-testid="stExpander"] { border: 1px solid #75515f; border-radius: 0; background: #272027; }
  [data-testid="stTabs"] button { text-transform: uppercase; letter-spacing: .03em; }
  [data-testid="stImage"] img { border: 1px solid #75515f; border-radius: 0; }
  [data-testid="stCaptionContainer"] { color: #d6bdc5 !important; }
  [data-testid="stCodeBlock"] { border: 1px solid #75515f; border-radius: 0; }
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### FIND YOUR DAUGHTER")
    st.caption("CT review · direct branches from the aorta")
    with st.expander("01 · CASE FILES", expanded=True):
        dataset_text = st.text_input("Dataset folder", value=str(ROOT / "dataset"),
                                     help="Searches this folder and its subject subfolders for NIfTI scans.")
        dataset = Path(dataset_text).expanduser()
        subjects = find_subjects(dataset)
        if not subjects:
            st.info("No NIfTI scans were found here. Check the dataset folder.")
            st.stop()
        subject = st.selectbox("Subject", list(subjects), key=f"subject:{dataset.resolve()}")
        image = st.selectbox("CT image", subjects[subject], format_func=lambda path: path.name,
                             key=f"image:{dataset.resolve()}:{subject}")
        masks = [None, *mask_choices(image)]
        mask = st.selectbox("Aorta mask", masks,
                            format_func=lambda path: "None" if path is None else path.name,
                            index=1 if len(masks) == 2 else 0, key=f"mask:{image.resolve()}")

try:
    image_mtime = image.stat().st_mtime_ns
    mask_mtime = mask.stat().st_mtime_ns if mask else None
    case = load_case(image, mask, image_mtime, mask_mtime)
except (OSError, ValueError, IndexError) as error:
    st.error(f"Could not open the selected scan and mask: {error}")
    st.stop()

shape = case.image.shape
spacing = case.image.geometry.spacing
can_detect = mask is not None and len(shape) == 3
with st.sidebar:
    with st.expander("02 · BRANCH DETECTION", expanded=True):
        detector = st.selectbox("Algorithm", DETECTOR_NAMES,
                                help="Baseline is the default evaluator algorithm. Contact is an alternate experimental method.")
        if not can_detect:
            st.caption("Select a 3-D CT and its aorta mask to run detection.")
        prediction = None
        detection_error = None
        if can_detect:
            try:
                with st.spinner(f"Detecting candidate branches ({detector})…"):
                    prediction = load_prediction(image, mask, image_mtime, mask_mtime, detector)
            except (OSError, ValueError, RuntimeError) as error:
                detection_error = str(error)
        daughters = prediction["daughters"] if prediction else []
        branch_names = [daughter["instance_id"] for daughter in daughters]
        selected_label = st.selectbox("Highlight branch", ["All candidates", *branch_names],
                                      disabled=not branch_names)
        selected_branch = None if selected_label == "All candidates" else selected_label
        show_branches = st.checkbox("Show branch markers", True,
                                    help="Show the estimated opening, 5 mm seed, direction arrow, and radius ring.",
                                    disabled=not branch_names)

with st.sidebar:
    with st.expander("03 · VIEW", expanded=True):
        viewer_mode = st.radio("Viewer", ["Fast interactive", "Detailed VTK"],
                               help="Fast interactive lets you turn a sampled scan with the mouse. Detailed VTK renders the CT volume and three cross-sections on the CPU.")

st.title("Find Your Daughter")
st.caption("Review branches that may arise directly from the supplied aorta mask. Check each finding against the CT scan.")
st.markdown(f"**{case.case_id}** &nbsp; | &nbsp; CT: `{image.name}` &nbsp; | &nbsp; "
            f"{shape[0]} × {shape[1]} × {shape[2]} voxels &nbsp; | &nbsp; "
            f"spacing {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} "
            f"{case.image.geometry.spatial_unit} &nbsp; | &nbsp; "
            f"{len(daughters)} {'branch' if len(daughters) == 1 else 'branches'} found")
if detection_error:
    st.error(f"Detection failed: {detection_error}")

if viewer_mode == "Fast interactive":
    old_session = st.session_state.pop("vtk_session", None)
    if old_session is not None:
        old_session.close()
    st.session_state.pop("vtk_session_key", None)
    with st.sidebar:
        with st.expander("04 · CT APPEARANCE", expanded=False):
            frame = st.slider("Frame", 0, shape[3] - 1, 0, key="fast_frame") if len(shape) == 4 else 0
            window = st.slider("Window width", 1, 2000, 600 if can_detect else 400,
                               help="Width of the visible CT intensity range. Values are HU when the scan is calibrated.",
                               key="fast_window")
            level = st.slider("Window center", -1000, 1500, 200 if can_detect else 40,
                              help="Middle of the visible CT intensity range.",
                              key="fast_level")
            density = st.select_slider("CT sampling limit", options=[48, 64, 72, 96], value=72,
                                       help="Higher values show more CT points but take longer to prepare.")
            show_volume = st.checkbox("Show CT sample", True, key="fast_volume")
            show_mask = st.checkbox("Show aorta surface", True,
                                    disabled=case.mask is None, key="fast_mask")
    view_tab, results_tab = st.tabs(["3D VIEW", "RESULTS"])
    with view_tab:
        with st.spinner("Preparing the scan view…"):
            payload = scene_payload(case, frame=frame, window=window, level=level,
                                    show_volume=show_volume, show_mask=show_mask,
                                    sampling_limit=density, prediction=prediction,
                                    show_branches=show_branches, selected_branch=selected_branch)
        components.html(scene_html(payload), height=604, scrolling=False)
        st.caption(f"{payload['count_ct']:,} CT points · {payload['count_mask']:,} aorta surface points. "
                   "The blue dot marks the opening; the teal arrow points into the branch; the pale teal ring shows its estimated radius. "
                   "Drag to turn the scan. This view samples the CT; use Detailed VTK to inspect the full volume.")
else:
    with st.sidebar:
        with st.expander("04 · CT APPEARANCE", expanded=False):
            frame = st.slider("Frame", 0, shape[3] - 1, 0, key="vtk_frame") if len(shape) == 4 else 0
            window = st.slider("Window width", 1, 2000, 600 if can_detect else 400,
                               help="Width of the visible CT intensity range. Values are HU when the scan is calibrated.",
                               key="vtk_window")
            level = st.slider("Window center", -1000, 1500, 200 if can_detect else 40,
                              help="Middle of the visible CT intensity range.",
                              key="vtk_level")
            use_threshold = st.checkbox("Use minimum intensity", False)
            threshold = st.slider("Minimum intensity", -1000, 1500, 100,
                                  disabled=not use_threshold)
            opacity = st.slider("Volume opacity", 0.0, 0.5, 0.12, 0.01)
            max_dimension = st.select_slider("CT sampling limit", [64, 96, 128, 192, 256], value=128)
            show_volume = st.checkbox("Show CT volume", False if can_detect else True)
            show_mask = st.checkbox("Show aorta surface", True, disabled=case.mask is None)
            mip = st.checkbox("Maximum intensity projection", False)
        with st.expander("05 · VIEW ANGLE", expanded=False):
            azimuth = st.slider("Azimuth (degrees)", -180, 180, 30)
            elevation = st.slider("Elevation (degrees)", -90, 90, 25)
            zoom = st.slider("Zoom", 0.5, 2.5, 1.0, 0.1)
            focus_mask = st.checkbox("Focus camera on aorta", False,
                                     disabled=case.mask is None)
        with st.expander("06 · CROSS-SECTIONS", expanded=True):
            show_slices = st.checkbox("Show three orthogonal slice panels", True)
            show_planes = st.checkbox("Show slice planes in 3D", True)
            plane_opacity = st.slider("3D plane opacity", 0.0, 1.0, 0.85, 0.05,
                                      disabled=not show_planes)
            st.caption("I, J, and K move through the scan one voxel at a time. The letter in parentheses gives the physical axis direction.")
            slice_indices = tuple(
                st.slider(f"{axis} / axis {dim} ({case.image.geometry.axis_codes[dim]})",
                          0, shape[dim] - 1, shape[dim] // 2,
                          disabled=not (show_slices or show_planes))
                for dim, axis in enumerate("IJK")
            )
            cut = st.checkbox("Clip CT volume at K", False)
            cut_at_k = st.slider("CT cut position / K", 0, shape[2] - 1, shape[2] // 2,
                                 disabled=not cut) if cut else None

    view_tab, results_tab = st.tabs(["3D VIEW", "RESULTS"])
    with view_tab:
        case_key = (image, mask, image_mtime, mask_mtime)
        view_key = (case_key, frame, window, level, threshold if use_threshold else None,
                    opacity, max_dimension, azimuth, elevation, zoom, focus_mask,
                    show_volume, show_mask, mip, cut_at_k, show_slices, show_planes,
                    plane_opacity, slice_indices)
        session_key = (case_key, frame, max_dimension)
        if st.session_state.get("vtk_session_key") != session_key:
            old_session = st.session_state.pop("vtk_session", None)
            if old_session is not None:
                old_session.close()
            st.session_state["vtk_session_key"] = session_key
        if st.session_state.get("rendered_view") != view_key:
            try:
                options = VolumeViewOptions(frame=frame, max_dimension=max_dimension,
                                            window=window, level=level,
                                            min_intensity=threshold if use_threshold else None,
                                            opacity=opacity)
                started = monotonic()
                progress = st.progress(0, text="Rendering VTK volume · 0.0s")
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pending = pool.submit(prepare_view, case, options, azimuth=azimuth,
                                          elevation=elevation, zoom=zoom,
                                          show_volume=show_volume, show_mask=show_mask,
                                          mip=mip, cut_at_k=cut_at_k,
                                          show_slices=show_slices, show_planes=show_planes,
                                          plane_opacity=plane_opacity, focus_mask=focus_mask,
                                          slice_indices=slice_indices,
                                          session=st.session_state.get("vtk_session"))
                    while not pending.done():
                        elapsed = monotonic() - started
                        progress.progress(min(95, 5 + int(elapsed * 3)),
                                          text=f"Rendering VTK volume · {elapsed:.1f}s")
                        sleep(0.15)
                    png, mode, reason, counts, session = pending.result()
                elapsed = monotonic() - started
                progress.progress(100, text=f"3D view ready · {elapsed:.1f}s")
                st.session_state.update(preview_png=png, render_mode=mode,
                                        fallback_reason=reason, point_counts=counts,
                                        render_seconds=elapsed, rendered_view=view_key,
                                        vtk_session=session)
            except (OSError, ValueError, RuntimeError, ImportError) as error:
                st.session_state.pop("preview_png", None)
                st.error(f"3D view failed: {error}")
        if st.session_state.get("rendered_view") == view_key and "preview_png" in st.session_state:
            mode = st.session_state["render_mode"]
            st.caption(f"{mode} rendered in {st.session_state['render_seconds']:.1f}s. "
                       "Move the I, J, or K sliders to inspect another cross-section.")
            if st.session_state["fallback_reason"]:
                st.info(f"VTK unavailable: {st.session_state['fallback_reason']} CPU preview shown.")
            st.image(st.session_state["preview_png"], width="stretch")
            st.download_button("Save PNG", st.session_state["preview_png"],
                               file_name=f"{case.case_id}_3d.png", mime="image/png",
                               on_click="ignore")

with results_tab:
    st.subheader("Branch results")
    st.caption("These are candidate branches from the selected detector. They have not been checked against reference annotations.")
    if prediction is not None:
        st.download_button("Export evaluator JSON", json.dumps(prediction, indent=2) + "\n",
                           file_name=f"{case.case_id}_{detector}_prediction.json",
                           mime="application/json", on_click="ignore")
    if detection_error:
        st.error(detection_error)
    elif not can_detect:
        st.info("Select a 3-D CT and its aorta mask to look for branches.")
    elif not daughters:
        st.info("No branches met this detector's criteria in the selected scan.")
    else:
        st.write(f"{len(daughters)} {'branch' if len(daughters) == 1 else 'branches'} found with the {detector} detector.")
        st.caption("The opening is where the branch meets the aorta. The seed is about 5 mm farther into the branch. "
                   "Positions use physical LPS coordinates in millimetres; direction is a unit vector pointing away from the aorta.")
        ordered = sorted(daughters, key=lambda item: item["instance_id"] != selected_branch)
        for daughter in ordered:
            fmt = lambda values: "  ".join(f"{value:+.2f}" for value in values)
            st.markdown(f"**{daughter['instance_id'].upper()}**")
            st.code(f"parent       {daughter['parent_instance_id']}\n"
                    f"ostium LPS   {fmt(daughter['ostium_xyz_mm'])} mm\n"
                    f"seed LPS     {fmt(daughter['seed_xyz_mm'])} mm\n"
                    f"radius       {daughter['radius_mm']:.2f} mm\n"
                    f"direction    {fmt(daughter['direction_xyz'])} (unit)", language=None)
